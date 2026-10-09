"""
Execution-graph instrument (docs/plan_autonomy.md): trace the competition's own tasks in the scorer-faithful env.

For each public task whose gold patch passes in the scorer-like env (54/128, kernel gemma4-devset-build v5), run the
official verification (swegemma subprocess sandbox, scorer site-packages) with a fake agent, in three modes:
  empty  : no patch            -> the failing run; executed call graph of the fail-to-pass tests on the buggy code
  gold   : the gold patch      -> the passing run
  loo<i> : gold minus the hunks of gold function group i (only for fixes touching >= 2 functions) -> necessity
A pytest plugin (sys.monitoring, PY_START) records, per test node, the workspace functions executed and the
caller -> callee edges between workspace functions (nearest workspace frame as caller), plus the test outcome.

Kaggle: CPU, Internet ON. Inputs: competition data, dataset metric/gemma-4-developer-agent-wheelhouse,
dataset shreysharma07/test-deps-wheels, kernel output gemma4-devset-build (scorer_site.tgz).
Output: /kaggle/working/xg/ = results.jsonl, tasks_meta.jsonl, traces/<id>__<mode>.json, src/<id>.tgz (base .py
files), graphs/<commit>.json (the provided graph for each task's commit), run_report.json.
"""
import asyncio
import glob
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import time

T0 = time.time()
OUT = pathlib.Path("/kaggle/working/xg")
WORK = pathlib.Path("/tmp/xg_work")
PLUG = pathlib.Path("/kaggle/working/_xgplugin")  # outside /tmp: the subprocess sandbox rewrites /tmp in commands
for d in (OUT / "traces", OUT / "src", OUT / "graphs", WORK, PLUG):
    d.mkdir(parents=True, exist_ok=True)
MAX_HOURS = 10.0
IDS = json.loads(os.environ.get("XG_IDS", "null") or "null")  # default: the 54 gold-passing tasks (below)
GOLD_OK = ['fastapi_14258', 'fastapi_14372', 'fastapi_14851', 'fastapi_14873', 'fastapi_14986', 'fastapi_15280',
           'fastapi_15588', 'fastapi_15589', 'fastapi_15661', 'fastapi_15800', 'requests_6644', 'requests_7205',
           'requests_7309', 'requests_7315', 'requests_7427', 'rich_2725', 'rich_2943', 'rich_3006', 'rich_3043',
           'rich_3052', 'rich_3061', 'rich_3063', 'rich_3067', 'rich_3105', 'rich_3180', 'rich_3278', 'rich_3454',
           'rich_3468', 'rich_3470', 'rich_3471', 'rich_3480', 'rich_3506', 'rich_3518', 'rich_3521', 'rich_3535',
           'rich_3675', 'rich_3676', 'rich_3718', 'rich_3772', 'rich_3777', 'rich_3882', 'rich_3894', 'rich_3905',
           'rich_3930', 'rich_3934', 'rich_3935', 'rich_3938', 'rich_3942', 'rich_3944', 'rich_3953', 'rich_4006',
           'rich_4075', 'rich_4076', 'rich_4077']
LOG = open(OUT / "run.log", "a", buffering=1)


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")


def first(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None


# ------------------------------------------------------------------ pytest tracer plugin
(PLUG / "xgtrace.py").write_text(r'''
import json, os, sys
import pytest

ROOT = os.path.realpath(os.getcwd())
OUT = os.environ["XG_OUT"]
_mon = getattr(sys, "monitoring", None)
TOOL = 4
_bucket = {"__collect__": [set(), set()]}
_cur = ["__collect__"]
_outcome = {}
_keys = {}


def _key(code):
    k = _keys.get(code)
    if k is None:
        raw = code.co_filename
        fn = os.path.realpath(raw)
        ok = not raw.startswith("<") and os.path.isfile(fn) and fn.startswith(ROOT + os.sep) and "site-packages" not in fn
        k = (os.path.relpath(fn, ROOT) + "::" + code.co_qualname) if ok else ""
        _keys[code] = k
    return k


def _on_start(code, offset):
    k = _key(code)
    if not k:
        return _mon.DISABLE
    b = _bucket.setdefault(_cur[0], [set(), set()])
    b[0].add(k)
    f = sys._getframe(1).f_back
    while f is not None:
        ck = _key(f.f_code)
        if ck:
            b[1].add((ck, k))
            break
        f = f.f_back
    return None


if _mon is not None:
    _mon.use_tool_id(TOOL, "xgtrace")
    _mon.register_callback(TOOL, _mon.events.PY_START, _on_start)
    _mon.set_events(TOOL, _mon.events.PY_START)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    _cur[0] = item.nodeid
    yield
    _cur[0] = "__collect__"


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        _outcome[report.nodeid] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    if _mon is not None:
        _mon.set_events(TOOL, 0)
    data = {"root": ROOT, "python": sys.version.split()[0], "monitoring": _mon is not None, "exit": int(exitstatus),
            "tests": {t: {"outcome": _outcome.get(t, "none"), "nodes": sorted(b[0]), "edges": sorted(b[1])}
                      for t, b in _bucket.items()}}
    with open(OUT, "w") as fh:
        json.dump(data, fh)
''')

# ------------------------------------------------------------------ competition data + harness (from build_devset.py)
_all = sorted(glob.glob("/kaggle/input/**/tasks.jsonl", recursive=True))
_comp = [p for p in _all if (pathlib.Path(p).parent / "graphs").is_dir()]
tasks_path = (_comp or _all or [None])[0]
assert tasks_path, "attach the competition data"
DATA = pathlib.Path(tasks_path).parent
log("competition data:", DATA, "| python", sys.version.split()[0])


def ensure_ensurepip():
    chk = [sys.executable, "-c", "import ensurepip"]
    if subprocess.run(chk, capture_output=True).returncode == 0:
        return
    v = f"{sys.version_info.major}.{sys.version_info.minor}"
    subprocess.run("apt-get update -qq", shell=True, capture_output=True)
    for pkg in (f"python{v}-venv", "python3-venv"):
        subprocess.run(f"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq {pkg}", shell=True, capture_output=True)
        if subprocess.run(chk, capture_output=True).returncode == 0:
            return


def ensure_harness():
    try:
        from swegemma.evaluate import Evaluator  # noqa: F401
        return
    except Exception:  # noqa: BLE001
        pass
    wh = first("/kaggle/input/**/swegemma-*.whl")
    assert wh, "attach dataset metric/gemma-4-developer-agent-wheelhouse"
    skip = ("torch", "vllm", "nvidia", "triton", "cutlass", "xformers", "flashinfer", "flash_attn", "cupy")
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    ok_tags = set(sys_tags())
    whls, pins = [], []
    for w in pathlib.Path(wh).parent.glob("*.whl"):
        if w.name.lower().startswith(skip):
            continue
        name, ver, _, tags = parse_wheel_filename(w.name)
        (whls.append(str(w)) if tags & ok_tags else pins.append(f"{name}=={ver}"))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall"] + whls)
    if pins:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps"] + pins)
        if r.returncode:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps"] + [p.split("==")[0] for p in pins])
    import importlib
    importlib.invalidate_caches()


ensure_ensurepip()
ensure_harness()

tasks = [json.loads(l) for l in open(tasks_path) if l.strip()]
sec = next((a / "secret" for a in [DATA, *DATA.parents] if (a / "secret").is_dir()), None)
tp_map, gp_map = {}, {}
if sec:
    import pandas as pd
    for name, target, col in (("solution", tp_map, "test_patch"), ("perfect_submission", gp_map, None)):
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
for t in tasks:
    t["test_patch"] = t.get("test_patch") or tp_map.get(t["instance_id"], "")
    t["patch"] = t.get("patch") or gp_map.get(t["instance_id"], "")
ids = IDS or GOLD_OK
tasks = [t for t in tasks if t["instance_id"] in set(ids) and t["patch"] and t["test_patch"]]
log(f"{len(tasks)} tasks with gold + test patch")
TASKS_FILE = WORK / "tasks.jsonl"
with open(TASKS_FILE, "w") as f:
    for t in tasks:
        f.write(json.dumps(t) + "\n")

# scorer env: restore the site-packages built by gemma4-devset-build (same as the scorer's), else fail loudly
SCORER_SITE = pathlib.Path("/tmp/scorer_site")
site_tgz = first("/kaggle/input/**/scorer_site.tgz")
assert site_tgz, "attach the output of kernel gemma4-devset-build (scorer_site.tgz)"
SCORER_SITE.mkdir(parents=True, exist_ok=True)
subprocess.run(["tar", "-xzf", site_tgz, "-C", str(SCORER_SITE)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "inline-snapshot", "dirty-equals",
                "asttokens", "executing"])

import swegemma.evaluate as ev  # noqa: E402
import swegemma.harness.container_setup as cs  # noqa: E402
import swegemma.harness.verification as vf  # noqa: E402
import swegemma.sandbox.subprocess as sp  # noqa: E402

orig_init, orig_start = sp.SubprocessManager.__init__, sp.SubprocessManager.start


def _init(self, *a, **kw):
    kw["system_site_packages"] = False
    orig_init(self, *a, **kw)


def _start(self, *a, **kw):
    sid = orig_start(self, *a, **kw)
    for spd in pathlib.Path(self._sandboxes[sid]["venv"]).glob("lib/python*/site-packages"):
        (spd / "_scorer_site.pth").write_text(f"{SCORER_SITE}\n")
    return sid


sp.SubprocessManager.__init__, sp.SubprocessManager.start = _init, _start
setup_py = DATA / "sandbox" / "setup.py"
orig_deps = cs.install_test_dependencies


def install_test_dependencies(docker, container_id, repo="", *, fast_path=True, config=None):
    if type(docker).__name__ not in {"SubprocessManager", "SubprocessSandbox"}:
        return orig_deps(docker, container_id, repo, fast_path=fast_path, config=config)
    flag = "--fast-path" if fast_path else "--no-fast-path"
    docker.exec(container_id, f"python3 {setup_py} {flag} --workspace /workspace {repo}".strip(), timeout=600)


for mod in (cs, vf, ev):
    if hasattr(mod, "install_test_dependencies"):
        mod.install_test_dependencies = install_test_dependencies

# ------------------------------------------------------------------ tracing hook on the verification pytest command
import contextvars  # noqa: E402

CUR = contextvars.ContextVar("xg_cur", default=None)  # (instance_id, mode), set by the fake agent per task
orig_exec = vf.sandbox_exec


async def traced_exec(docker, eval_id, cmd, *a, **kw):
    cur = CUR.get()
    if cur and "-m pytest" in cmd and "--junitxml" in cmd:
        out = OUT / "traces" / f"{cur[0]}__{cur[1]}.json"
        cmd = cmd.replace("python3 -s -m pytest", f"XG_OUT={out} PYTHONPATH={PLUG}${{PYTHONPATH:+:$PYTHONPATH}} "
                                                  f"python3 -s -m pytest -p xgtrace", 1)
    return await orig_exec(docker, eval_id, cmd, *a, **kw)


vf.sandbox_exec = traced_exec

# ------------------------------------------------------------------ base sources + gold function groups
import ast  # noqa: E402
from swegemma.deduplication import resolve_task_snapshot_paths  # noqa: E402
from swegemma.harness.container_setup import extract_snapshot  # noqa: E402

mgr = sp.SubprocessManager(timeout_seconds=900)


def materialise(t):
    dst = WORK / "src" / t["instance_id"]
    if dst.exists():
        return dst
    snap, base, patch = resolve_task_snapshot_paths(DATA / "snapshots", t["instance_id"], t["repo"])
    sid = mgr.start()
    try:
        extract_snapshot(mgr, sid, snap, patch_path=patch, base_snapshot_path=base)
        ws = pathlib.Path(mgr._sandboxes[sid]["workspace"])
        dst.mkdir(parents=True)
        with tarfile.open(OUT / "src" / f"{t['instance_id']}.tgz", "w:gz") as tf:
            for p in ws.rglob("*.py"):
                if ".git" in p.parts:
                    continue
                rel = p.relative_to(ws)
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst / rel)
                tf.add(p, arcname=str(rel))
    finally:
        mgr.stop(sid)
    return dst


def split_patch(patch):
    """-> [(file_header_lines, [(old_start, old_len, hunk_lines)])] per file"""
    files, cur, hunk = [], None, None
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git") or (line.startswith("--- ") and (cur is None or cur[1])):
            cur = [[line], []]
            files.append(cur)
            hunk = None
        elif cur is None:
            continue
        elif line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))?", line)
            hunk = [int(m.group(1)), int(m.group(2) or 1), [line]]
            cur[1].append(hunk)
        elif hunk is None:
            cur[0].append(line)
        else:
            hunk[2].append(line)
    return files


def path_of(header):
    for l in header:
        if l.startswith("--- a/"):
            return l[6:].strip()
    m = re.match(r"diff --git a/(\S+)", header[0])
    return m.group(1) if m else None


def func_spans(src):
    out = []
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return out

    def walk(node, stack):
        for c in ast.iter_child_nodes(node):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = ".".join(stack + [c.name])
                out.append((c.lineno, c.end_lineno or c.lineno, q))
                walk(c, stack + [c.name, "<locals>"])
            elif isinstance(c, ast.ClassDef):
                walk(c, stack + [c.name])
            else:
                walk(c, stack)
    walk(tree, [])
    return out


def hunk_functions(path, src_root, hunk):
    """innermost function(s) containing the hunk's changed old lines (qualname format = co_qualname)."""
    f = src_root / path
    spans = func_spans(f.read_text(errors="replace")) if f.exists() else []
    old, changed = hunk[0], []
    for l in hunk[2][1:]:
        if l.startswith("-"):
            changed.append(old)
            old += 1
        elif l.startswith("+"):
            changed.append(max(old - 1, 1))
        elif not l.startswith("\\"):
            old += 1
    fns = set()
    for ln in changed:
        inside = [(e - s, q) for s, e, q in spans if s <= ln <= e]
        fns.add(f"{path}::{min(inside)[1]}" if inside else f"{path}::<module>")
    return fns


def build_variants(t):
    src_root = materialise(t)
    files = split_patch(t["patch"])
    groups = {}  # function key -> list of (file_idx, hunk_idx)
    for fi, (hdr, hunks) in enumerate(files):
        p = path_of(hdr)
        for hi, h in enumerate(hunks):
            key = " | ".join(sorted(hunk_functions(p, src_root, h)))
            groups.setdefault(key, []).append((fi, hi))
    func_groups = [k for k in groups if "<module>" not in k]

    def assemble(drop):
        out = []
        for fi, (hdr, hunks) in enumerate(files):
            kept = [h for hi, h in enumerate(hunks) if (fi, hi) not in drop]
            if kept:
                out.append("".join(hdr) + "".join("".join(h[2]) for h in kept))
        return "".join(out)

    variants = {"empty": "", "gold": t["patch"]}
    if len(func_groups) >= 2:
        for i, g in enumerate(sorted(func_groups)):
            variants[f"loo{i}"] = assemble(set(groups[g]))
    return variants, sorted(func_groups), sorted(groups)


# ------------------------------------------------------------------ run the official verifier per mode
from adk_submission import ModelRegistry  # noqa: E402
from adk_eval_core.tracing import SessionTrace  # noqa: E402
from swegemma.config import EvalConfig  # noqa: E402
from swegemma.evaluate import Evaluator  # noqa: E402

PATCHES = {}
dummy_sub = WORK / "dummy_submission"
dummy_sub.mkdir(exist_ok=True)
MODE = [None]


async def fake_agent(self, task, snapshot_path, task_index, total_tasks, **kw):
    CUR.set((task.instance_id, MODE[0]))
    return PATCHES[(task.instance_id, MODE[0])], None, SessionTrace()


Evaluator._run_agent_sandbox = fake_agent


def run_mode(mode, task_ids):
    MODE[0] = mode
    cfg = EvalConfig(tasks_path=TASKS_FILE, snapshots_dir=DATA / "snapshots", results_dir=WORK / f"res_{mode}",
                     submission_dir=dummy_sub, models=ModelRegistry(), sandbox="subprocess", wheels_dir=DATA / "wheels",
                     task_ids=list(task_ids), concurrency=4, display_mode="quiet", max_time_minutes=30,
                     timeout_seconds=600)
    res = asyncio.run(Evaluator(cfg).run())
    with open(OUT / "results.jsonl", "a") as f:
        for r in res.task_results:
            f.write(json.dumps({"id": r.instance_id, "mode": mode, "resolved": bool(r.resolved),
                                "exit": r.test_exit_code, "tail": (r.test_output or "")[-800:]}) + "\n")
    return {r.instance_id: bool(r.resolved) for r in res.task_results}


meta = {}
for t in tasks:
    try:
        variants, fgroups, allgroups = build_variants(t)
    except Exception as e:  # noqa: BLE001
        log("variant build failed", t["instance_id"], type(e).__name__, e)
        continue
    for m, p in variants.items():
        PATCHES[(t["instance_id"], m)] = p
    meta[t["instance_id"]] = {"repo": t["repo"], "base_commit": t.get("base_commit"), "modes": sorted(variants),
                              "gold_function_groups": fgroups, "all_groups": allgroups,
                              "FAIL_TO_PASS": t.get("FAIL_TO_PASS"), "PASS_TO_PASS": t.get("PASS_TO_PASS"),
                              "problem_statement": t.get("problem_statement", ""), "patch": t["patch"]}
with open(OUT / "tasks_meta.jsonl", "w") as f:
    for iid, m in meta.items():
        f.write(json.dumps({"id": iid, **m}) + "\n")
log(f"variants built for {len(meta)} tasks; multi-function (LOO) tasks: "
    f"{sum(len(m['gold_function_groups']) >= 2 for m in meta.values())}")

# provided graph per task commit (for the provided-vs-executed comparison)
for iid, m in meta.items():
    c = m["base_commit"]
    g = first(str(DATA / "graphs" / f"*{c}*")) if c else None
    if g and not (OUT / "graphs" / pathlib.Path(g).name).exists():
        shutil.copy2(g, OUT / "graphs" / pathlib.Path(g).name)

summary = {}
for mode in json.loads(os.environ.get("XG_MODES", "null") or "null") or ["empty", "gold"]:  # v3: re-trace with the frozen-frame fix (loo results in v2)
    if (time.time() - T0) / 3600 > MAX_HOURS:
        log("time cap reached before", mode)
        break
    ids_m = [i for i in meta if (i, mode) in PATCHES]
    if not ids_m:
        continue
    t = time.time()
    r = run_mode(mode, ids_m)
    summary[mode] = {"n": len(r), "resolved": sum(r.values()), "min": round((time.time() - t) / 60, 1),
                     "traced": sum((OUT / "traces" / f"{i}__{mode}.json").exists() for i in ids_m)}
    log("mode", mode, summary[mode])
    (OUT / "run_report.json").write_text(json.dumps({"summary": summary, "hours": round((time.time() - T0) / 3600, 2)},
                                                    indent=1))
log("DONE", json.dumps(summary))
