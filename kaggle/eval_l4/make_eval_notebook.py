"""
Build the competition notebook (check-l4, the only kernel that gets 4x L4) that evaluates submissions/<name> locally
with the scorer's stack, from the evalkit package. Cells, in order:
  1. settings          Settings -> /kaggle/working/eval_cfg.json (edit here to change the run)
  2. embedded inputs   submission_src.zip + trace_analyzer.py, written from base64
  3. evalkit modules   one %%writefile cell per module (kaggle/eval_l4/evalkit/*.py, the source of truth)
  4. scorer env        ScorerEnv().prepare() in the notebook's own Python 3.13 (stdlib only)
  5. evaluate          python -m evalkit.main inside the restored Py3.12 scorer venv, streamed
  6. results           summary.json
--preflight builds a free CPU kernel instead (kaggle/eval_l4/preflight_kernel): same cells 1-4 without the GPU check,
then evalkit.preflight (gold patches through the official Evaluator, no model). Run it after any env/sandbox change.
Usage: python3 kaggle/eval_l4/make_eval_notebook.py v7_noedit [--max-tasks 8] [--repeats 1] [--devset] [--budget-h 10.5]
       [--preflight]
Then:  .venv/bin/kaggle kernels push -p kaggle/probes/check_l4   (or -p kaggle/eval_l4/preflight_kernel)
"""
import argparse
import ast
import base64
import io
import json
import pathlib
import sys
import zipfile

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "kaggle"))
from owner import mine, src as shared  # noqa: E402
KIT = REPO / "kaggle" / "eval_l4" / "evalkit"
GPU_DIR = REPO / "kaggle" / "probes" / "check_l4"
CPU_DIR = REPO / "kaggle" / "eval_l4" / "preflight_kernel"
GATE_DIR = REPO / "kaggle" / "eval_l4" / "gate_kernel"
VALID = REPO / "data" / "runs" / "eval_sweep_v2" / "valid_tasks.json"     # 114 tasks verifiable with the overlays (Oct 14)
PRE_OVERLAY_VALID = REPO / "data" / "runs" / "eval_sweep" / "valid_tasks.json"  # 54 without them: decides which tasks get pins
OVL_DIR = REPO / "kaggle" / "eval_l4" / "overlays_kernel"
NEC_DIR = REPO / "kaggle" / "eval_l4" / "necessity_kernel"
SWAP_DIR = REPO / "kaggle" / "eval_l4" / "siteswap_kernel"  # A12 site-swap + A5 typology (--siteswap)
SWAP2_DIR = REPO / "kaggle" / "eval_l4" / "siteswap2_kernel"  # A12v validated site-swap (--siteswap2)
# A3/A4 (--necessity): agent patches + per-task rows of these runs, and the old-54 LOO results (exec-graph run)
NEC_RUNS = ["eval_v7_new60/x/eval_v7_noedit/rep0", "eval_v14_new60_peek/x/eval_v14_thinking/rep0"] + \
    [f"rs_v7_valid/x/eval_v7_noedit/rep{i}" for i in range(5)]
MODULES = ["__init__", "config", "util", "envsetup", "inputs", "submission", "compat", "server", "runner", "report",
           "preflight", "overlays", "gate", "necessity", "siteswap", "siteswap2", "main"]
GATE_PREFIXES = REPO / "data" / "runs" / "gate" / "prefixes.json"
SOURCE_MAX_BYTES = 1_000_000  # Kaggle rejects notebook source above ~1 MB (HTTP 400; found by Atul, Oct 15)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


def zip_submission(name):
    src = REPO / "submissions" / name
    assert (src / "agent.yaml").exists(), f"no submission at {src}"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store":
                z.write(p, p.relative_to(src))
    return base64.b64encode(buf.getvalue()).decode()


def module_doc(name):
    text = (KIT / f"{name}.py").read_text()
    doc = text.split('"""')[1].strip().splitlines()[0] if text.startswith('"""') else ""
    return doc


def run_cells(a, step, result):
    return [
        md("## 4. Scorer environment\nRestore the Py3.12 venv, upgrade changed wheelhouse wheels, add pip and test deps, "
           "and prepare the Python 3.13 task sandbox (scorer site)."),
        code(f"""sys.path.insert(0, "/kaggle/working")
from evalkit.envsetup import ScorerEnv
ScorerEnv().prepare(gpus={not a.preflight})"""),
        md("## 5. " + ("Preflight\nGold patches through the official Evaluator with a fake agent (no model)."
                      if a.preflight else
                      "Evaluate\nRuns inside the venv (vLLM + Evaluator); output is streamed here and to `run.log`.")),
        code(f"""import subprocess, threading, time, glob
proc = subprocess.Popen(["/tmp/scorer_host/bin/python", "-m", "evalkit.{step}"], cwd="/kaggle/working",
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
# hard stop: the in-run budget only stops NEW chunks, so a hang could run to Kaggle's 12 h cap (Oct 18: ~21 quota h lost)
deadline = EVAL_CFG["t0"] + (EVAL_CFG["session_budget_h"] + 0.25) * 3600
killed = []
def _watchdog():
    while proc.poll() is None:
        if time.time() > deadline:
            print("WATCHDOG: session budget exceeded, killing the evaluator", flush=True)
            killed.append(1)
            proc.kill()
            return
        time.sleep(30)
threading.Thread(target=_watchdog, daemon=True).start()
for line in proc.stdout:
    print(line, end="", flush=True)
print("exit code", proc.wait())
if killed:  # keep whatever finished: traces and results of completed chunks
    subprocess.run("tar czf /kaggle/working/partial_eval.tgz -C /kaggle/working $(ls -d eval_* 2>/dev/null | grep -v tgz) run.log",
                   shell=True)"""),
        md("## 6. Results"),
        code(f"""import os
p = "/kaggle/working/{result}"
print(open(p).read()[:20000] if os.path.exists(p) else "no result file (see WATCHDOG / run.log)")"""),
    ]


def build(a):
    a.siteswap = a.siteswap or a.siteswap2  # same payload
    a.preflight = a.preflight or a.sweep or a.overlays or a.necessity or a.siteswap
    settings = {"name": a.name, "max_tasks": a.max_tasks, "repeats": a.repeats, "devset": a.devset,
                "session_budget_h": a.budget_h, "sweep": a.sweep, "valid_only": a.valid_only}
    gate_b64 = system_b64 = tools_b64 = ""
    if a.gate:  # behaviour gate: prefixes + v7's exact system prompt and declared tools
        import gzip
        sys.path.insert(0, str(REPO / "posttrain"))
        import render_atif
        settings.update(gate_adapters=[n for n in a.gate_adapters.split(",") if n], gate_samples=a.gate_samples)
        gate_b64 = base64.b64encode(gzip.compress(GATE_PREFIXES.read_bytes())).decode()
        system_b64 = base64.b64encode((REPO / "submissions" / "v7_noedit" / "prompts" / "system.md").read_bytes()).decode()
        tools_b64 = base64.b64encode(json.dumps(render_atif.declared_tools(
            REPO / "submissions" / "v7_noedit" / "agent.yaml")).encode()).decode()
    valid = pathlib.Path(a.valid_file) if a.valid_file else PRE_OVERLAY_VALID if a.overlays else VALID
    valid_b64 = base64.b64encode(valid.read_bytes()).decode() if (a.valid_only or a.overlays or a.necessity or a.siteswap) else ""
    nec_b64 = ""
    if a.necessity or a.siteswap:
        import tarfile
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for r in NEC_RUNS:
                d = REPO / "data" / "runs" / r
                tf.add(d / "analysis" / "per_task.jsonl", arcname=f"{r}/analysis/per_task.jsonl")
                for pt in sorted((d / "merged" / "patches").glob("*.patch")):
                    tf.add(pt, arcname=f"{r}/merged/patches/{pt.name}")
            for f in ("results.jsonl", "tasks_meta.jsonl"):
                tf.add(REPO / "data" / "runs" / "xg" / f, arcname=f"xg/{f}")
            if a.siteswap:  # A3 necessity labels (new-60) and oblig.py (A5 typology)
                tf.add(REPO / "results" / "runs" / "a3a4" / "necessity_new60.json", arcname="a3/necessity_new60.json")
                tf.add(REPO / "tools" / "oblig.py", arcname="tools/oblig.py")
            if a.siteswap2:  # A12 swaps to validate
                tf.add(REPO / "results" / "runs" / "a12" / "swaps.json", arcname="a12/swaps.json")
        nec_b64 = base64.b64encode(buf.getvalue()).decode()
    assert valid.exists() or not (a.valid_only or a.overlays), f"needs {valid} (run --sweep first)"
    cells = [
        md(f"""# Scorer-faithful local eval: `{a.name}`
Generated by `kaggle/eval_l4/make_eval_notebook.py` from `kaggle/eval_l4/evalkit/` (edit there, not here).
Stack: vLLM 0.19.1 on `gemma-4-31b-it-qat-w4a16-ct` (TP=4), official swegemma 0.2.11 Evaluator, subprocess sandbox,
offline agent, the submission's own `eval_config.yaml` budgets."""),
        md("## 1. Settings\nEverything the run needs is in `eval_cfg.json`; the budget clock starts here."),
        code(f"""import json, os, sys, time
os.makedirs("/kaggle/working/evalkit", exist_ok=True)
EVAL_CFG = {settings!r}
EVAL_CFG["t0"] = time.time()
open("/kaggle/working/eval_cfg.json", "w").write(json.dumps(EVAL_CFG, indent=1))
EVAL_CFG"""),
        md("## 2. Embedded inputs\nThe submission under test (zip of `submissions/" + a.name + "`) and the trace analyzer."),
        code(f"""import base64, pathlib
SUBMISSION_B64 = {zip_submission(a.name)!r}
ANALYZER_B64 = {base64.b64encode((REPO / 'analysis' / 'trace_analyzer.py').read_bytes()).decode()!r}
pathlib.Path("/kaggle/working/submission_src.zip").write_bytes(base64.b64decode(SUBMISSION_B64))
pathlib.Path("/kaggle/working/trace_analyzer.py").write_bytes(base64.b64decode(ANALYZER_B64))
VALID_B64 = {valid_b64!r}  # tasks the scorer env can verify (preflight sweep); empty unless --valid-only
if VALID_B64:
    pathlib.Path("/kaggle/working/valid_tasks.json").write_bytes(base64.b64decode(VALID_B64))
NEC_B64 = {nec_b64!r}  # A3/A4 only: agent patches + per-task rows + old-54 LOO results
if NEC_B64:
    pathlib.Path("/kaggle/working/a3a4_payload.tgz").write_bytes(base64.b64decode(NEC_B64))
GATE_B64, SYSTEM_B64, TOOLS_B64 = {gate_b64!r}, {system_b64!r}, {tools_b64!r}  # behaviour gate only
if GATE_B64:
    import gzip
    pathlib.Path("/kaggle/working/gate_prefixes.json").write_bytes(gzip.decompress(base64.b64decode(GATE_B64)))
    pathlib.Path("/kaggle/working/gate_system.md").write_bytes(base64.b64decode(SYSTEM_B64))
    pathlib.Path("/kaggle/working/gate_tools.json").write_bytes(base64.b64decode(TOOLS_B64))
print("submission zip", len(base64.b64decode(SUBMISSION_B64)), "bytes")"""),
        md("## 3. evalkit modules\nOne cell per module."),
    ]
    for m in MODULES:
        cells.append(md(f"### `evalkit/{m}.py`\n{module_doc(m)}"))
        cells.append(code(f"%%writefile /kaggle/working/evalkit/{m}.py\n" + (KIT / f"{m}.py").read_text()))
    step, result = (("gate", "gate_report.json") if a.gate else ("necessity", "necessity_report.json") if a.necessity
                    else ("siteswap2", "siteswap2_report.json") if a.siteswap2
                    else ("siteswap", "siteswap_report.json") if a.siteswap
                    else ("preflight", "preflight.json") if a.preflight
                    else ("main", "summary.json"))
    if a.overlays:  # CPU + Internet: build extras + per-task pins with the notebook's own Python 3.13 (no venv)
        cells += [
            md("## 4. Build overlays\nTest-only extras + per-task version pins (see `evalkit/overlays.py`)."),
            code("""import subprocess
sys.path.insert(0, "/kaggle/working")
proc = subprocess.Popen([sys.executable, "-m", "evalkit.overlays"], cwd="/kaggle/working",
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end="", flush=True)
print("exit code", proc.wait())"""),
            md("## 5. Results"),
            code("""print(open("/kaggle/working/overlays/report.json").read()[:20000])"""),
        ]
    else:
        cells += run_cells(a, step, result)
    for c in cells:  # every code cell must be valid Python before it costs a GPU session
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            ast.parse(src.split("\n", 1)[1] if src.startswith("%%writefile") else src)
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    wire = len(json.dumps({**nb, "cells": [{**c, "source": "".join(c["source"])} for c in cells]}).encode())
    assert wire < SOURCE_MAX_BYTES, f"notebook source {wire:,} bytes >= {SOURCE_MAX_BYTES:,}: move inputs to a dataset"
    out_dir = GATE_DIR if a.gate else OVL_DIR if a.overlays else NEC_DIR if a.necessity else SWAP2_DIR if a.siteswap2 else SWAP_DIR if a.siteswap else CPU_DIR if a.preflight else GPU_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_name = ("gemma4-gate.ipynb" if a.gate else "gemma4-test-overlays.ipynb" if a.overlays else
               "gemma4-a3a4-necessity.ipynb" if a.necessity else "gemma4-a12v-validate.ipynb" if a.siteswap2 else "gemma4-a12-siteswap.ipynb" if a.siteswap else
               "gemma4-eval-preflight.ipynb" if a.preflight else "check-l4.ipynb")
    (out_dir / nb_name).write_text(json.dumps(nb, indent=1))

    meta_path = out_dir / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "id": mine("gemma4-eval-preflight"), "title": "gemma4-eval-preflight", "code_file": nb_name,
        "language": "python", "kernel_type": "notebook", "is_private": True}
    meta["dataset_sources"] = ["metric/gemma-4-developer-agent-wheelhouse", shared("test-deps-wheels")]
    meta["kernel_sources"] = [shared("gemma4-scorer-env"), shared("gemma4-devset-build")]  # scorer_site
    for k in a.adapter_from or []:  # training outputs holding adapters/<name>/
        meta["kernel_sources"].append(k)
    if a.overlays_from and not a.overlays:  # extras + pins for the task sandboxes (the 114-task valid set needs them)
        meta["kernel_sources"].append(a.overlays_from)
    meta["model_sources"] = [] if a.preflight else ["google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2"]
    meta["competition_sources"] = ["gemma-4-developer-agent"]
    if not a.preflight:
        meta["id"], meta["title"] = mine("check-l4"), "check L4"
    else:
        meta["id"] = mine("gemma4-eval-preflight")
    meta["enable_internet"] = False
    meta["enable_gpu"] = not a.preflight
    if a.overlays:  # PyPI downloads; inputs = tasks + scorer site only
        meta.update(id=mine("gemma4-test-overlays"), title="gemma4-test-overlays", code_file=nb_name,
                    enable_internet=True, dataset_sources=[], kernel_sources=[shared("gemma4-devset-build")])
    if a.siteswap:  # CPU, offline, same inputs as A3/A4
        meta.update(id=mine("gemma4-a12-siteswap"), title="gemma4-a12-siteswap", code_file=nb_name)
    if a.siteswap2:
        meta.update(id=mine("gemma4-a12v-validate"), title="gemma4-a12v-validate", code_file=nb_name)
    if a.necessity:  # CPU, offline, same inputs as the preflight sweep + overlays
        meta.update(id=mine("gemma4-a3a4-necessity"), title="gemma4-a3a4-necessity", code_file=nb_name)
    if a.gate:  # GPU (L4 via the competition source), no task sandboxes needed
        meta.update(id=mine("gemma4-gate"), title="gemma4-gate", code_file=nb_name)
    if a.preflight:
        meta.pop("machine_shape", None)
    else:
        meta["machine_shape"] = "NvidiaL4"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out_dir / nb_name}: {len(cells)} cells, settings {settings}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--devset", action="store_true")
    ap.add_argument("--budget-h", type=float, default=10.5)
    ap.add_argument("--preflight", action="store_true", help="CPU kernel: gold-patch check, no model")
    ap.add_argument("--sweep", action="store_true", help="preflight over all tasks: gold + empty -> valid_tasks.json")
    ap.add_argument("--valid-only", action="store_true", help="evaluate only data/runs/eval_sweep/valid_tasks.json")
    ap.add_argument("--overlays", action="store_true", help="CPU+Internet kernel that builds test extras + version pins")
    ap.add_argument("--overlays-from", default=shared("gemma4-test-overlays"), help="kernel whose output holds overlays/ (e.g. <owner>/gemma4-test-overlays)")
    ap.add_argument("--valid-file", default=None, help="valid task list to embed (default: the 114-task data/runs/eval_sweep_v2)")
    ap.add_argument("--adapter-from", action="append", default=None,
                    help="kernel whose output holds adapters/<name>/ (repeatable; e.g. <owner>/gemma4-train-s3-compact)")
    ap.add_argument("--gate", action="store_true", help="GPU behaviour gate: base vs adapters at fixed decision points")
    ap.add_argument("--siteswap2", action="store_true", help="CPU kernel: A12v validated site-swap + A5 keyword-free")
    ap.add_argument("--siteswap", action="store_true", help="CPU kernel: A12 site-swap counterfactual + A5 typology")
    ap.add_argument("--necessity", action="store_true", help="CPU kernel: A3 LOO necessity on new-60 + A4 completion counterfactual")
    ap.add_argument("--gate-adapters", default="", help="comma-separated adapter names for --gate")
    ap.add_argument("--gate-samples", type=int, default=4)
    build(ap.parse_args())
