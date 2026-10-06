"""
Build a de-familiarised dev set from the competition's public tasks, and verify it with the official scorer code.

Why: our local set (fastapi / rich / requests) is public, so Gemma has likely seen it. The hidden set is private
repos. Local gains (15 -> 26 -> 33 / 129) no longer show up on the leaderboard (0.00 -> 0.08 -> 0.08).
This script renames each repo's package and its distinctive names (same length, word-boundary, consistently
across code, tests, issue text, gold patch and test patch) and rebuilds git history from scratch. Memorised
names stop matching, while the code's behaviour stays identical.

Run on Kaggle (CPU is enough; Internet ON helps for test deps):
  Inputs: competition data + dataset `gemma-4-developer-agent-wheelhouse` (owner: metric).
  !python build_devset.py
Output: /kaggle/working/devset/ = tasks.jsonl (verified tasks only), snapshots/<id>.tgz, devset_report.json.
Then: "Save Version", open the output, and use "New Dataset" -> call it gemma-devset-renamed. Attach that dataset
to the L4 notebook and set DEVSET=True in run_official_l4.py.
"""
import asyncio
import csv
import glob
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

CFG = {
    "LIMIT": None,           # e.g. 10 for a quick trial
    "VERIFY": True,          # run gold-patch and empty-patch checks with the official verifier
    "CONCURRENCY": 4,
    "OUT": "/kaggle/working/devset",
    "WORK": "/tmp/devset_build",
}

# same-length replacements so fixed-width rendered outputs (rich tables, tracebacks) keep their alignment
RENAMES = {
    "fastapi/fastapi": ("glimmer/glimmer", [("FastAPI", "Glimmer"), ("FASTAPI", "GLIMMER"), ("Fastapi", "Glimmer"),
                                            ("fastapi", "glimmer"), ("tiangolo", "quorvexa")]),
    "Textualize/rich": ("lumaco/luma", [("Textualize", "Lumacosoft"), ("rich", "luma"), ("Rich", "Luma")]),
    "psf/requests": ("conduit/conduits", [("requests", "conduits"), ("Requests", "Conduits"),
                                          ("kennethreitz", "jordanbelkyy")]),
}
TEXT_EXT = {".py", ".pyi", ".toml", ".cfg", ".ini", ".txt", ".md", ".rst", ".yml", ".yaml", ".json", ".in",
            ".html", ".css", ".js", ".sh", ".typed", ".lock", ".csv", ".svg", ""}
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def first(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None


def compile_rules(pairs):
    return [(re.compile(r"\b" + re.escape(a) + r"\b"), b) for a, b in pairs]


def apply_rules(text, rules):
    for rx, b in rules:
        text = rx.sub(b, text)
    return text


# ------------------------------------------------------------------ 0. inputs + official harness
_all = sorted(glob.glob("/kaggle/input/**/tasks.jsonl", recursive=True))
_comp = [p for p in _all if (pathlib.Path(p).parent / "graphs").is_dir()]
tasks_path = (_comp or _all or [None])[0]
assert tasks_path, "attach the competition data"
DATA = pathlib.Path(tasks_path).parent
log("competition data:", DATA)


def ensure_harness():
    try:
        import swegemma  # noqa: F401
        from swegemma.evaluate import Evaluator  # noqa: F401
        return True
    except Exception as e:  # noqa: BLE001
        log("installing harness from wheelhouse:", type(e).__name__, e)
    wh = first("/kaggle/input/**/swegemma-*.whl")
    if not wh:
        log("wheelhouse dataset not attached -> cannot verify")
        return False
    skip = ("torch", "vllm", "nvidia", "triton", "cutlass", "xformers", "flashinfer", "flash_attn", "cupy")
    whls = [str(w) for w in pathlib.Path(wh).parent.glob("*.whl") if not w.name.lower().startswith(skip)]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall"] + whls)
    try:
        import importlib
        importlib.invalidate_caches()
        from swegemma.evaluate import Evaluator  # noqa: F401
        return True
    except Exception as e:  # noqa: BLE001
        log("harness import still failing:", type(e).__name__, e)
        return False


HAVE_HARNESS = ensure_harness()
if HAVE_HARNESS:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "inline-snapshot", "dirty-equals",
                    "asttokens", "executing"])
    dep = first("/kaggle/input/**/inline_snapshot-*.whl")
    if dep:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps"] +
                       [str(p) for p in pathlib.Path(dep).parent.glob("*.whl")])

# ------------------------------------------------------------------ 1. tasks (+ hidden fields)
tasks = [json.loads(l) for l in open(tasks_path) if l.strip()]


def secret_maps():
    tp, gp = {}, {}
    for anc in [DATA, *DATA.parents]:
        sec = anc / "secret"
        if not sec.is_dir():
            continue
        try:
            import pandas as pd
            for name, target, col in (("solution", tp, "test_patch"), ("perfect_submission", gp, None)):
                for ext in ("parquet", "csv"):
                    f = sec / f"{name}.{ext}"
                    if f.exists():
                        df = pd.read_parquet(f) if ext == "parquet" else pd.read_csv(f)
                        for r in df.to_dict(orient="records"):
                            iid = str(r.get("id") or r.get("instance_id") or "")
                            val = r.get(col) if col else (r.get("prediction") or r.get("patch"))
                            if iid and isinstance(val, str):
                                target[iid] = val
                        break
        except Exception as e:  # noqa: BLE001
            log("secret read failed:", e)
        break
    return tp, gp


tp_map, gp_map = secret_maps()
for t in tasks:
    t["test_patch"] = t.get("test_patch") or tp_map.get(t["instance_id"], "")
    t["patch"] = t.get("patch") or gp_map.get(t["instance_id"], "")
tasks = [t for t in tasks if t["repo"] in RENAMES]
if CFG["LIMIT"]:
    tasks = tasks[: CFG["LIMIT"]]
log(f"{len(tasks)} tasks to rename; with test_patch: {sum(bool(t['test_patch']) for t in tasks)}; "
    f"with gold patch: {sum(bool(t['patch']) for t in tasks)}")

# ------------------------------------------------------------------ 2. materialise + rename + repack
OUT, WORK = pathlib.Path(CFG["OUT"]), pathlib.Path(CFG["WORK"])
shutil.rmtree(OUT, ignore_errors=True)
(OUT / "snapshots").mkdir(parents=True)
WORK.mkdir(parents=True, exist_ok=True)

from swegemma.deduplication import resolve_task_snapshot_paths  # noqa: E402
from swegemma.harness.container_setup import extract_snapshot  # noqa: E402
from swegemma.sandbox.subprocess import SubprocessManager  # noqa: E402

mgr = SubprocessManager(timeout_seconds=900)


def materialise(t, dst):
    snap, base, patch = resolve_task_snapshot_paths(DATA / "snapshots", t["instance_id"], t["repo"])
    sid = mgr.start()
    try:
        extract_snapshot(mgr, sid, snap, patch_path=patch, base_snapshot_path=base)
        ws = pathlib.Path(mgr._sandboxes[sid]["workspace"])
        shutil.copytree(ws, dst, symlinks=True)
    finally:
        mgr.stop(sid)


def rename_tree(root, rules):
    stats = {"files_changed": 0, "paths_renamed": 0}
    shutil.rmtree(root / ".git", ignore_errors=True)
    for dp, dns, fns in os.walk(root, topdown=False):
        for fn in fns:
            p = pathlib.Path(dp) / fn
            if p.is_symlink():
                continue
            if p.suffix.lower() in TEXT_EXT or fn in ("Makefile", "LICENSE", "AUTHORS", "CHANGES"):
                try:
                    raw = p.read_bytes()
                    if b"\x00" in raw[:8192]:
                        raise ValueError
                    txt = raw.decode("utf-8")
                    new = apply_rules(txt, rules)
                    if new != txt:
                        p.write_text(new, encoding="utf-8")
                        stats["files_changed"] += 1
                except (UnicodeDecodeError, ValueError):
                    pass
            nfn = apply_rules(fn, rules)
            if nfn != fn:
                p.rename(p.with_name(nfn))
                stats["paths_renamed"] += 1
        for dn in dns:
            nd = apply_rules(dn, rules)
            if nd != dn:
                (pathlib.Path(dp) / dn).rename(pathlib.Path(dp) / nd)
                stats["paths_renamed"] += 1
    env = dict(os.environ, GIT_AUTHOR_NAME="dev", GIT_AUTHOR_EMAIL="dev@example.com", GIT_COMMITTER_NAME="dev",
               GIT_COMMITTER_EMAIL="dev@example.com", GIT_AUTHOR_DATE="2026-01-01T00:00:00",
               GIT_COMMITTER_DATE="2026-01-01T00:00:00")
    for cmd in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"], ["git", "commit", "-q", "-m", "snapshot"]):
        subprocess.run(cmd, cwd=root, env=env, capture_output=True)
    return stats


new_tasks, build_rows = [], []
T0 = time.time()
for i, t in enumerate(tasks, 1):
    new_repo, pairs = RENAMES[t["repo"]]
    rules = compile_rules(pairs)
    new_id = apply_rules(t["instance_id"], rules)
    if new_id == t["instance_id"]:
        new_id = "dev_" + t["instance_id"]
    tmp = WORK / new_id
    shutil.rmtree(tmp, ignore_errors=True)
    row = {"orig": t["instance_id"], "new": new_id}
    try:
        materialise(t, tmp)
        row.update(rename_tree(tmp, rules))
        out = OUT / "snapshots" / f"{new_id}.tgz"
        subprocess.run(["tar", "-czf", str(out), "-C", str(tmp), "."], check=True)
        row["tgz_mb"] = round(out.stat().st_size / 1e6, 1)
        nt = dict(t)
        nt.update(instance_id=new_id, repo=new_repo, orig_instance_id=t["instance_id"], orig_repo=t["repo"],
                  problem_statement=apply_rules(t.get("problem_statement", ""), rules),
                  hints_text=apply_rules(t.get("hints_text", "") or "", rules),
                  patch=apply_rules(t["patch"], rules), test_patch=apply_rules(t["test_patch"], rules))
        for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            if isinstance(nt.get(k), str):
                nt[k] = apply_rules(nt[k], rules)
            elif isinstance(nt.get(k), list):
                nt[k] = [apply_rules(x, rules) for x in nt[k]]
        new_tasks.append(nt)
        row["ok"] = True
    except Exception as e:  # noqa: BLE001
        row.update(ok=False, error=f"{type(e).__name__}: {e}"[:300])
    shutil.rmtree(tmp, ignore_errors=True)
    build_rows.append(row)
    if i % 10 == 0 or i == len(tasks):
        log(f"built {i}/{len(tasks)} ({(time.time() - T0) / 60:.1f} min); last: {row}")

with open(OUT / "tasks_all.jsonl", "w") as f:
    for nt in new_tasks:
        f.write(json.dumps(nt) + "\n")
leaks = sum(1 for nt in new_tasks for a, _ in RENAMES[nt["orig_repo"]][1]
            if re.search(r"\b" + re.escape(a) + r"\b", nt["problem_statement"] + nt["test_patch"] + nt["patch"]))
log(f"built {len(new_tasks)} tasks; old names left in issue/patch texts: {leaks} (should be 0)")

# ------------------------------------------------------------------ 3. verify with the official verifier
verify = {}
if CFG["VERIFY"] and HAVE_HARNESS:
    from adk_submission import ModelRegistry
    from adk_eval_core.tracing import SessionTrace
    from swegemma.config import EvalConfig
    from swegemma.evaluate import Evaluator

    dummy_sub = WORK / "dummy_submission"
    dummy_sub.mkdir(exist_ok=True)

    def run_mode(tasks_file, snaps, ids, mode):
        async def fake_agent(self, task, snapshot_path, task_index, total_tasks, **kw):
            return (task.patch if mode == "gold" else ""), None, SessionTrace()
        Evaluator._run_agent_sandbox = fake_agent
        cfg = EvalConfig(tasks_path=pathlib.Path(tasks_file), snapshots_dir=pathlib.Path(snaps),
                         results_dir=WORK / f"verify_{mode}_{int(time.time())}", submission_dir=dummy_sub,
                         models=ModelRegistry(), sandbox="subprocess", wheels_dir=DATA / "wheels",
                         task_ids=list(ids), concurrency=CFG["CONCURRENCY"], display_mode="quiet",
                         max_time_minutes=30, timeout_seconds=600)
        res = asyncio.run(Evaluator(cfg).run())
        return {r.instance_id: bool(r.resolved) for r in res.task_results}

    ids = [nt["instance_id"] for nt in new_tasks]
    t = time.time()
    gold = run_mode(OUT / "tasks_all.jsonl", OUT / "snapshots", ids, "gold")
    log(f"gold check: {sum(gold.values())}/{len(gold)} pass ({(time.time() - t) / 60:.1f} min)")
    t = time.time()
    empty = run_mode(OUT / "tasks_all.jsonl", OUT / "snapshots", ids, "empty")
    log(f"empty-patch check: {sum(empty.values())}/{len(empty)} pass (should be 0) ({(time.time() - t) / 60:.1f} min)")
    bad = [nt for nt in new_tasks if not gold.get(nt["instance_id"])]
    orig_gold = {}
    if bad:
        # is the failure caused by the renaming, or does the original task also fail locally?
        orig_file = WORK / "orig_tasks.jsonl"
        with open(orig_file, "w") as f:
            for t0 in tasks:
                f.write(json.dumps(t0) + "\n")
        orig_gold = run_mode(orig_file, DATA / "snapshots", [nt["orig_instance_id"] for nt in bad], "gold")
        log(f"originals of failing renamed tasks that pass gold: {sum(orig_gold.values())}/{len(orig_gold)}")
    for nt in new_tasks:
        i = nt["instance_id"]
        verify[i] = {"gold": gold.get(i), "empty": empty.get(i), "orig_gold": orig_gold.get(nt["orig_instance_id"])}
    keep = [nt for nt in new_tasks if gold.get(nt["instance_id"]) and not empty.get(nt["instance_id"])]
else:
    log("verification skipped -> tasks.jsonl contains all built tasks (unverified)")
    keep = new_tasks

with open(OUT / "tasks.jsonl", "w") as f:
    for nt in keep:
        f.write(json.dumps(nt) + "\n")
report = {"n_source": len(tasks), "n_built": len(new_tasks), "n_verified": len(keep),
          "by_repo": {r: sum(1 for nt in keep if nt["repo"] == r) for r in {nt["repo"] for nt in keep}},
          "renames": {k: [v[0], v[1]] for k, v in RENAMES.items()}, "build": build_rows, "verify": verify,
          "total_mb": round(sum(p.stat().st_size for p in (OUT / "snapshots").glob("*.tgz")) / 1e6, 1)}
for nt in new_tasks:  # drop snapshots of tasks that failed verification
    if nt not in keep:
        (OUT / "snapshots" / f"{nt['instance_id']}.tgz").unlink(missing_ok=True)
(OUT / "devset_report.json").write_text(json.dumps(report, indent=1))
(OUT / "build.log").write_text("\n".join(LOG))
log(f"DONE: {len(keep)} verified tasks, {report['by_repo']}, {report['total_mb']} MB -> {OUT}")
