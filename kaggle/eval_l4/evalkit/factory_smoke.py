"""
FS1 factory rollout smoke (GPU). Pre-registered in docs/experiments.md (Oct 28).
60 of the verified factory-v0 tasks (dataset gemma4-factory-v0; 20 per library, stratified, seed 0):
  1. rewrite   Gemma (thinking off) turns each template issue into a user-style bug report, using the hidden tests'
               inputs/expectations as the reproducer. A rewrite that quotes a gold-patch added line, names a test, or
               is too short falls back to the template (counted).
  2. rollouts  the submission under test (v14) unchanged: the first 6 tasks at concurrency 1 (throughput baseline +
               health check), the other 54 at concurrency 6.
Output: /kaggle/working/factory_smoke_report.json + eval_smoke_c{1,6}.tgz (traces) + smoke_tasks.jsonl.
Run inside the venv: python -m evalkit.factory_smoke
"""
import json
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import litellm

from .compat import patch_sandbox_venv, use_scorer_sandbox
from .config import INPUT, WORK, Settings
from .inputs import Inputs
from .report import Report
from .runner import ChunkedEvaluation
from .server import ScorerServer, make_sandbox_offline, set_scorer_env
from .submission import Submission
from .util import log

PER_LIB, N_BASE, CONC = 20, 6, 6
TASKS_OUT = WORK / "smoke_tasks.jsonl"
REWRITE_PROMPT = """You are a user of the Python library `{pkg}` filing a GitHub issue about a bug you hit.

Below are (1) a terse draft of the issue and (2) the maintainers' internal test code that exposes the bug.
Write the GitHub issue as a real user would: a short title line, what you were doing, a minimal standalone code
snippet that calls the public API (derive the inputs and expected values from the test code), what you got
(exception or wrong output) and what you expected.

Rules:
- Do NOT mention tests, test names, test classes, pytest, or any file under tests/.
- Do NOT say where in the library's source the bug is, and do NOT suggest a fix.
- Plain markdown, under 250 words. Output only the issue text.

(1) Draft issue:
{issue}

(2) Test code:
```python
{tests}
```"""


def lib_of(iid):
    return iid.split("_", 1)[1].rsplit("_", 1)[0]


def pick(tasks, meta):
    """PER_LIB tasks per library, singles and pairs in their natural proportion (shuffled with seed 0)."""
    rng = random.Random(0)
    out = []
    for lib in sorted({lib_of(t["instance_id"]) for t in tasks}):
        pool = [t for t in tasks if lib_of(t["instance_id"]) == lib]
        rng.shuffle(pool)
        out += pool[:PER_LIB]
    rng.shuffle(out)
    return out


def added_lines(patch):
    return [l[1:].strip() for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++")]


def check_rewrite(text, task):
    if len(text) < 200:
        return "short"
    if re.search(r"\btests?/|def test_|pytest|\bTestCase\b", text):
        return "names_tests"
    f2p = task["FAIL_TO_PASS"] if isinstance(task["FAIL_TO_PASS"], list) else json.loads(task["FAIL_TO_PASS"])
    if any(n.rsplit("::", 1)[-1] in text for n in f2p):
        return "names_tests"
    if any(len(l) >= 8 and l in text for l in added_lines(task["patch"])):
        return "leaks_gold"
    return None


def chat(base_url, model, prompt):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "top_p": 0.95,
            "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=600))
    return (out["choices"][0]["message"].get("content") or "").strip()


def rewrite_all(server, tasks):
    base_url = server.server.base_url
    model = json.load(urllib.request.urlopen(f"{base_url}/models"))["data"][0]["id"]

    def one(t):
        tests = "\n".join(l[1:] for l in t["test_patch"].splitlines() if l.startswith("+") and not l.startswith("+++"))
        prompt = REWRITE_PROMPT.format(pkg=t["repo"].split("/")[-1], issue=t["problem_statement"], tests=tests[:3000])
        for _ in range(2):  # one retry on a rejected rewrite
            try:
                text = chat(base_url, model, prompt)
            except Exception as e:  # noqa: BLE001
                text, why = "", f"error: {type(e).__name__}"
            else:
                why = check_rewrite(text, t)
            if why is None:
                return text, None
        return None, why

    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        outs = list(ex.map(one, tasks))
    stats = {"n": len(tasks), "rewritten": 0, "fallback": {}, "minutes": 0}
    stats["templates"] = {t["instance_id"]: t["problem_statement"] for t in tasks}
    for t, (text, why) in zip(tasks, outs):
        if text:
            t["problem_statement"] = text
            stats["rewritten"] += 1
        else:
            stats["fallback"][why] = stats["fallback"].get(why, 0) + 1
    stats["minutes"] = round((time.time() - t0) / 60, 1)
    stats["samples"] = {t["instance_id"]: t["problem_statement"] for t in tasks[:3]}
    log("REWRITE", json.dumps({k: v for k, v in stats.items() if k != "samples"}))
    return stats


def phase(s, inp, sub, models, adapters, label, ids, conc, chunk=18):
    ps = replace(s, name=label, task_ids=ids, concurrency=conc, chunk_size=len(ids) if conc == 1 else chunk,
                 max_tasks=None, repeats=1)
    t = time.time()
    ChunkedEvaluation(ps, inp, sub, models, adapters).run()
    wall = time.time() - t
    report = Report(ps, inp)
    summary = report.build()
    p = report.root / "rep0" / "merged" / "task_results.jsonl"
    rows = [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []
    # not report.package(): it deletes submission_under_test, which the next phase still needs
    log.sh(f"cd {WORK} && tar -czf {ps.label}.tgz {ps.label} && rm -rf {ps.label}")
    return {"tasks": len(ids), "done": len(rows), "resolved": sum(bool(r.get("resolved")) for r in rows),
            "wall_min": round(wall / 60, 1), "tasks_per_h": round(len(rows) / (wall / 3600), 2) if rows else 0,
            "task_sec_median": sorted(r.get("duration_seconds") or 0 for r in rows)[len(rows) // 2] if rows else None,
            "summary": summary}, rows


def breakdown(rows, meta, budget):
    by = {}
    cap = (budget.get("max_time_minutes") or 8) * 60  # pass the effective budget (rollouts override the cap)
    for r in rows:
        m = meta.get(r["instance_id"], {})
        for key in (f"lib:{lib_of(r['instance_id'])}", f"kind:{m.get('kind')}", *(f"op:{o}" for o in m.get("ops", []))):
            b = by.setdefault(key, [0, 0])
            b[0] += bool(r.get("resolved"))
            b[1] += 1
    return {"by": {k: f"{a}/{n}" for k, (a, n) in sorted(by.items())},
            "time_cap_hits": sum((r.get("duration_seconds") or 0) >= cap - 5 for r in rows),
            "call_cap_hits": sum((r.get("tool_calls") or 0) >= (budget.get("max_tool_calls") or 28) - 1 for r in rows),
            "empty_patch": sum(not r.get("agent_patch_size") for r in rows),
            "errors": sum(bool(r.get("error")) for r in rows)}


def main():
    s = Settings.load()
    set_scorer_env()
    patch_sandbox_venv()
    litellm.drop_params = True
    comp = Inputs.locate(False)
    use_scorer_sandbox(comp.data)
    fac = sorted(p for p in INPUT.rglob("tasks.jsonl") if (p.parent / "task_meta.jsonl").exists())
    assert fac, "attach dataset gemma4-factory-v0"
    fdir = fac[0].parent
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(fdir / "task_meta.jsonl")}
    tasks = pick([json.loads(l) for l in open(fac[0]) if l.strip()], meta)
    log(f"factory: {fdir}, picked {len(tasks)} tasks")

    sub = Submission().unpack()
    declared, adapters = sub.declared_model(), sub.adapters()
    server = ScorerServer(comp.model_path, adapters).start()
    models = server.registry(declared)
    report = {"tasks": [t["instance_id"] for t in tasks]}
    try:
        report["rewrite"] = rewrite_all(server, tasks)
        with open(TASKS_OUT, "w") as f:  # same keys as the competition rows; templates are in the report
            f.writelines(json.dumps(t) + "\n" for t in tasks)
        make_sandbox_offline()
        inp = Inputs(TASKS_OUT, comp.data, fdir / "snapshots", comp.model_path)
        ids = [t["instance_id"] for t in tasks]
        report["c1"], rows1 = phase(s, inp, sub, models, adapters, "smoke_c1", ids[:N_BASE], 1)
        log("PHASE c1", json.dumps({k: v for k, v in report["c1"].items() if k != "summary"}))
        if report["c1"]["done"] and s.hours_left() > 0.5:
            report["c6"], rows6 = phase(s, inp, sub, models, adapters, "smoke_c6", ids[N_BASE:], CONC)
            log("PHASE c6", json.dumps({k: v for k, v in report["c6"].items() if k != "summary"}))
        else:
            rows6 = []
        rows = rows1 + rows6
        report["overall"] = {"resolved": sum(bool(r.get("resolved")) for r in rows), "n": len(rows),
                             **breakdown(rows, meta, sub.eval_budget())}
        if report.get("c6") and report["c1"]["tasks_per_h"]:
            report["speedup_c6_vs_c1"] = round(report["c6"]["tasks_per_h"] / report["c1"]["tasks_per_h"], 2)
    finally:
        (WORK / "factory_smoke_report.json").write_text(json.dumps(report, indent=1))
        log("FS1 REPORT", json.dumps({k: v for k, v in report.items() if k not in ("tasks", "rewrite")}))
        log.sh(f"cd {WORK} && tar -czf factory_smoke.tgz factory_smoke_report.json smoke_tasks.jsonl run.log "
               f"$(ls eval_smoke_*.tgz 2>/dev/null)")
        server.stop()
        log(f"RUN_DONE in {(time.time() - s.t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
