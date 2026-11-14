"""
G1 held-out gate (GPU): v14 vs v14 + rs2_lora as exported, no k_eq_v copy (submissions/v18_rs2_lora) on the 95 toolz + pathspec factory tasks that
RS2 held out of training. Pre-registered in docs/experiments.md (Nov 1).
  - tasks: R2's rewritten issues (rollout_tasks.jsonl of the attached gemma4-factory-rollout-s0/s1 outputs), so both
    arms see exactly the prompts the RS2 val episodes were sampled from; snapshots + meta from gemma4-factory-v0/v1
  - one vLLM server (scorer settings) with the adapter; the base arm runs with no adapter manifest, like the scorer
  - arms alternate in blocks of BLOCK tasks (base block i, lora block i, ...), so a budget stop leaves them paired
  - R2's cap and concurrency (Settings.max_time_minutes, CONC 6); time-cap hits per arm show any LoRA slowdown
--dry (CPU): tasks, snapshots, both submissions and the adapter manifest, no server.
Output: /kaggle/working/lora_gate_report.json + lora_gate.tgz. Run inside the venv: python -m evalkit.lora_gate [--dry]
"""
import json
import math
import pathlib
import random
import sys
import time

from .compat import patch_sandbox_venv, use_scorer_sandbox
from dataclasses import replace

from .config import INPUT, SUBMISSION_ZIP, WORK, Settings
from .factory_smoke import breakdown, phase
from .inputs import Inputs
from .server import ScorerServer, make_sandbox_offline, set_scorer_env
from .submission import Submission
from .util import log

HOLDOUT = ("toolz", "pathspec")  # kaggle/rftset/rs2_core.py
CONC, BLOCK = 6, 24
BASE_ZIP = WORK / "base_src.zip"  # v14, written by the notebook's inputs cell
TASKS_OUT = WORK / "gate_tasks.jsonl"
SNAPS = pathlib.Path("/tmp/gate_snapshots")
GATE_SET = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--set=")), None) or Settings.load().gate_set
TAG = "lora_gate" if GATE_SET == "heldout" else f"lora_gate_{GATE_SET}"  # report + tgz names


def lib_of(i):
    return i.split("_", 1)[1].rsplit("_", 1)[0]


def load_tasks():
    """Held-out tasks with R2's rewritten issues, their snapshots (symlinked into one dir) and factory meta."""
    if GATE_SET == "f3":
        return load_f3()
    rows = {}
    for p in sorted(INPUT.rglob("rollout_tasks.jsonl")):
        for l in open(p):
            if l.strip():
                t = json.loads(l)
                if lib_of(t["instance_id"]) in HOLDOUT:
                    rows.setdefault(t["instance_id"], t)
    assert rows, "attach kernel outputs gemma4-factory-rollout-s0 / -s1"
    meta, snaps = {}, {}
    for tp in sorted(p for p in INPUT.rglob("tasks.jsonl") if (p.parent / "task_meta.jsonl").exists()):
        meta.update({json.loads(l)["id"]: json.loads(l) for l in open(tp.parent / "task_meta.jsonl") if l.strip()})
        snaps.update({s.stem: s for s in (tp.parent / "snapshots").glob("*.tgz")})
    SNAPS.mkdir(parents=True, exist_ok=True)
    for i in rows:
        assert i in snaps, f"missing snapshot for {i} (attach gemma4-factory-v0 / -v1)"
        (SNAPS / f"{i}.tgz").unlink(missing_ok=True)
        (SNAPS / f"{i}.tgz").symlink_to(snaps[i])
    tasks = sorted(rows.values(), key=lambda t: t["instance_id"])
    random.Random(0).shuffle(tasks)
    with open(TASKS_OUT, "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in tasks)
    return tasks, meta


def load_f3():
    """G2's real-issue arm: the verified F3 tasks (real bug-fix commits from libraries in no training set), with
    their own issue text; tasks.jsonl + task_meta.jsonl + snapshots/ from the attached gemma4-task-factory-f3."""
    tp = [p for p in sorted(INPUT.rglob("tasks.jsonl")) if (p.parent / "task_meta.jsonl").exists()]
    rows = {}
    for p in tp:
        for l in open(p):
            if l.strip() and json.loads(l)["instance_id"].startswith("f3_"):
                rows.setdefault(json.loads(l)["instance_id"], (json.loads(l), p.parent))
    assert rows, "attach kernel output gemma4-task-factory-f3"
    meta = {}
    SNAPS.mkdir(parents=True, exist_ok=True)
    for i, (_, d) in rows.items():
        meta.update({json.loads(l)["id"]: json.loads(l) for l in open(d / "task_meta.jsonl")
                     if l.strip() and json.loads(l)["id"] == i})
        assert (d / "snapshots" / f"{i}.tgz").exists(), f"missing snapshot for {i}"
        (SNAPS / f"{i}.tgz").unlink(missing_ok=True)
        (SNAPS / f"{i}.tgz").symlink_to(d / "snapshots" / f"{i}.tgz")
    tasks = sorted((t for t, _ in rows.values()), key=lambda t: t["instance_id"])
    random.Random(0).shuffle(tasks)
    with open(TASKS_OUT, "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in tasks)
    return tasks, meta


def arm_subs(names):
    """One submission per adapter arm: the embedded zip (which declares names[0]) with `adapter:` rewritten."""
    import re
    import shutil
    subs = {}
    for n in names:
        sub = Submission(SUBMISSION_ZIP, WORK / f"sub_{n}")
        if n == names[0]:
            sub.unpack()
        else:
            shutil.rmtree(sub.dir, ignore_errors=True)
            shutil.copytree(subs[names[0]].dir, sub.dir, ignore=shutil.ignore_patterns("adapters"))
            for y in sub.dir.rglob("*.yaml"):
                y.write_text(re.sub(rf"^(\s*adapter:\s*){re.escape(names[0])}\s*$", rf"\g<1>{n}", y.read_text(), flags=re.M))
            sub.fetch_adapters()
        subs[n] = sub
    return subs


def server_manifest(subs, names):
    """The server loads every arm's adapter: manifest of names[0]'s submission plus the other arms' adapters/ dirs."""
    import shutil
    if len(names) == 1:
        return subs[names[0]].adapters()
    from adk_submission import discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
    d = WORK / "sub_all"
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(subs[names[0]].dir, d)
    for n in names[1:]:
        shutil.copytree(subs[n].dir / "adapters" / n, d / "adapters" / n)
    return discover_adapters(str(d), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)


def mcnemar(a, b):
    """Exact two-sided McNemar p from the discordant counts."""
    n = a + b
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n)


def summarize(rows, meta, budget):
    secs = sorted(r.get("duration_seconds") or 0 for r in rows)
    return {"resolved": sum(bool(r.get("resolved")) for r in rows), "n": len(rows),
            "task_sec_median": secs[len(secs) // 2] if secs else None, **breakdown(rows, meta, budget)}


def main(dry=False):
    s = replace(Settings.load(), gate_set=GATE_SET)
    set_scorer_env()
    patch_sandbox_venv()
    comp = Inputs.locate(False, need_model=not dry)
    use_scorer_sandbox(comp.data)
    tasks, meta = load_tasks()
    ids = [t["instance_id"] for t in tasks]
    libs = sorted({lib_of(i) for i in ids})
    log(f"gate[{s.gate_set}] adapter {s.lora_adapter}: {len(ids)} tasks "
        f"({', '.join(f'{h} {sum(lib_of(i) == h for i in ids)}' for h in libs)}), "
        f"cap {s.max_time_minutes} min, concurrency {CONC}, blocks of {BLOCK}")
    names = s.lora_adapter.split(",")  # one arm per adapter; a single adapter keeps G1's arm name "lora"
    arm_of = {n: ("lora" if len(names) == 1 else n) for n in names}
    loras = [arm_of[n] for n in names]
    subs = {"base": Submission(BASE_ZIP, WORK / "sub_base").unpack(),
            **{arm_of[n]: v for n, v in arm_subs(names).items()}}
    mans = {a: sub.adapters() for a, sub in subs.items()}
    declared = {a: sub.declared_model() for a, sub in subs.items()}
    assert len(set(declared.values())) == 1, declared
    declared["lora"] = declared["base"]
    all_man = server_manifest({n: subs[arm_of[n]] for n in names}, names)
    report = {"gate_set": s.gate_set, "adapter": s.lora_adapter, "tasks": ids, "declared": declared["lora"],
              "adapters": {a: sorted(getattr(m, "adapters", {}) or {}) for a, m in mans.items()}}
    log("G1 setup", json.dumps({k: v for k, v in report.items() if k != "tasks"}))
    report["server_adapters"] = sorted(getattr(all_man, "adapters", {}) or {})
    assert all(report["adapters"][arm_of[n]] == [n] for n in names) and not report["adapters"]["base"], report["adapters"]
    assert report["server_adapters"] == sorted(names), report["server_adapters"]
    if dry:  # CPU: gold must resolve and the empty patch must not, in this kernel's own task sandbox
        from .preflight import verify
        make_sandbox_offline()
        inp = Inputs(TASKS_OUT, comp.data, SNAPS, comp.model_path)
        g = verify(inp, {t["instance_id"]: t.get("patch") or "" for t in tasks}, f"{TAG}_gold")
        e = verify(inp, {i: "" for i in ids}, f"{TAG}_empty")
        bad_gold = sorted(i for i in ids if not g.get(i, {}).get("resolved"))
        bad_empty = sorted(i for i in ids if e.get(i, {}).get("resolved", True))
        report["dry"] = {"n": len(ids), "gold_ok": len(ids) - len(bad_gold), "empty_fails": len(ids) - len(bad_empty),
                         "gold_failed": bad_gold, "empty_resolved": bad_empty,
                         "gold_fail_tails": {i: (g.get(i, {}).get("error") or g.get(i, {}).get("tail", ""))[-300:]
                                             for i in bad_gold[:8]}}
        log("GATE DRY", json.dumps({k: v for k, v in report["dry"].items() if k != "gold_fail_tails"}))
        (WORK / f"{TAG}_report.json").write_text(json.dumps(report, indent=1))
        return
    import litellm
    litellm.drop_params = True
    server = ScorerServer(comp.model_path, all_man).start()
    models = server.registry(declared["lora"])
    make_sandbox_offline()
    inp = Inputs(TASKS_OUT, comp.data, SNAPS, comp.model_path)
    rows = {a: [] for a in ("base", *loras)}
    report["blocks"] = []
    budget = {**subs[loras[0]].eval_budget(), **({"max_time_minutes": s.max_time_minutes} if s.max_time_minutes else {})}
    try:
        for b in range(0, len(ids), BLOCK):
            for arm in ("base", *loras):
                if s.hours_left() < 0.6:
                    log(f"budget: stop before block {b // BLOCK} {arm}")
                    raise StopIteration
                info, r = phase(s, inp, subs[arm], models, mans[arm], f"gate_{arm}_b{b // BLOCK}",
                                ids[b:b + BLOCK], CONC, BLOCK)
                rows[arm] += r
                report["blocks"].append({"arm": arm, "block": b // BLOCK,
                                         **{k: v for k, v in info.items() if k != "summary"}})
                log("BLOCK", json.dumps(report["blocks"][-1]))
    except StopIteration:
        pass
    finally:
        res = {a: {r["instance_id"]: bool(r.get("resolved")) for r in rows[a]} for a in rows}
        report["arms"] = {a: summarize(rows[a], meta, budget) for a in rows}
        report["paired_by"] = {}
        for x, y in [("base", a) for a in loras] + ([(loras[0], loras[1])] if len(loras) > 1 else []):
            both = sorted(set(res[x]) & set(res[y]))
            lo = sum(res[y][i] and not res[x][i] for i in both)
            bo = sum(res[x][i] and not res[y][i] for i in both)
            report["paired_by"][f"{y}_vs_{x}"] = {
                "n": len(both), x: sum(res[x][i] for i in both), y: sum(res[y][i] for i in both),
                f"{y}_only": lo, f"{x}_only": bo, "mcnemar_p": round(mcnemar(lo, bo), 4)}
        report["paired"] = report["paired_by"][f"{loras[0]}_vs_base"]
        report["arm_order"] = ["base", *loras]
        report["per_task"] = {i: [res[a].get(i) for a in ("base", *loras)] for i in ids}
        (WORK / f"{TAG}_report.json").write_text(json.dumps(report, indent=1))
        log("GATE REPORT", json.dumps({"paired": report["paired_by"],
                                     **{a: {k: v for k, v in x.items() if k != "by"} for a, x in report["arms"].items()}}))
        log.sh(f"cd {WORK} && tar -czf {TAG}.tgz {TAG}_report.json gate_tasks.jsonl run.log "
               f"$(ls eval_gate_*.tgz 2>/dev/null)")
        server.stop()
        log(f"RUN_DONE in {(time.time() - s.t0) / 3600:.1f} h")


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
