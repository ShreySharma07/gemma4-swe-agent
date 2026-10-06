"""
Gemma 4 Developer Agent — environment probe (run BEFORE the baseline).

Run in a Kaggle notebook with the competition dataset attached (CPU is enough):
    !python probe_env.py              # discovery only (~1 min)
    !RUN_SKIP=1 python probe_env.py   # + harness smoke test on 2 tasks, no LLM needed (~5-10 min)

Answers:
  1. Is swegemma installed / installable offline from the bundled wheels?
  2. What does the sample submission look like (agent.yaml, prompts, sub-agents, budgets)?
  3. models.yaml format  -> can local eval point at an API instead of a local GPU?
  4. How skill scripts run  -> can they read /workspace and the graph files?
  5. Does the subprocess sandbox + Phase-2 pytest pipeline work here (--skip-agent-patch)?
  6. Is coverage / sys.settrace usable inside the sandbox (dynamic tracing at inference)?

Writes /kaggle/working/probe_report.txt  -> upload that file back to Claude.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("DATA_ROOT", "/kaggle/input"))
OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "probe_report.txt"
lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def section(title):
    log("\n" + "=" * 72)
    log(title)
    log("=" * 72)


def sh(cmd, timeout=600, max_chars=6000):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr.strip() else "")
        return r.returncode, out[-max_chars:]
    except subprocess.TimeoutExpired:
        return -1, f"[timeout after {timeout}s]"


def show_file(p, max_chars=4000):
    try:
        txt = Path(p).read_text(errors="replace")
    except Exception as e:  # noqa: BLE001
        txt = f"[unreadable: {e}]"
    log(f"--- {p} ({len(txt)} chars)")
    log(txt[:max_chars] + ("\n[...truncated]" if len(txt) > max_chars else ""))


# ------------------------------------------------------------------ 1. locate data
section("1. DATA LAYOUT")
all_paths = list(ROOT.rglob("*"))
def find(pred):
    return sorted(p for p in all_paths if pred(p))

tasks = find(lambda p: p.name.endswith(".jsonl") and "task" in p.name.lower())
snap_dirs = find(lambda p: p.is_dir() and p.name.lower() in ("snapshots", "snapshot"))
sample_dirs = find(lambda p: p.is_dir() and "sample_submission" in p.name.lower())
wheel_dirs = sorted({p.parent for p in all_paths if p.suffix == ".whl"})
graph_dirs = find(lambda p: p.is_dir() and p.name.lower() == "graphs")
emb_dirs = find(lambda p: p.is_dir() and p.name.lower() == "embeddings")
for name, v in [("tasks", tasks), ("snapshots", snap_dirs), ("sample_submission", sample_dirs),
                ("wheel dirs", wheel_dirs), ("graphs", graph_dirs), ("embeddings", emb_dirs)]:
    log(f"{name:18}: {[str(x) for x in v][:5]}")
harness_wheels = sorted(p.name for d in wheel_dirs for p in d.glob("*.whl")
                        if re.search(r"swegemma|adk|eval", p.name, re.I))
log("harness-related wheels:", harness_wheels)
other_files = find(lambda p: p.is_file() and p.suffix in (".md", ".yaml", ".yml", ".py", ".txt", ".toml")
                   and "snapshot" not in str(p).lower() and "sample_submission" not in str(p).lower())
log("other text files (first 40):", [str(p.relative_to(ROOT)) for p in other_files[:40]])

# ------------------------------------------------------------------ 2. harness install
section("2. HARNESS INSTALL")
code, out = sh("pip show swegemma adk-submission adk-eval-core 2>&1 | grep -E 'Name|Version|Location'")
log(out or "(not installed)")
# The official harness ships in a SEPARATE Kaggle dataset: metric/gemma-4-developer-agent-wheelhouse
wh = sorted(glob.glob("/kaggle/input/**/swegemma-*.whl", recursive=True))
if not wh:
    log("!! Official wheelhouse NOT attached. In the notebook: Add Input -> search "
        "'gemma-4-developer-agent-wheelhouse' (owner: metric) -> attach, then rerun.")
elif "swegemma" not in out.lower():
    WH = Path(wh[0]).parent
    log("WHEELHOUSE =", WH)
    log("wheelhouse contents:", sorted(p.name for p in WH.glob("*.whl"))[:60])
    # CPU-only probe: install the three harness packages + their pure-Python deps; skip vllm/torch.
    pkgs = [str(p) for p in WH.glob("*.whl") if re.match(r"(swegemma|adk_submission|adk_eval_core|google_adk)-", p.name)]
    log("installing:", [Path(p).name for p in pkgs])
    code, out = sh(f"pip install -q --find-links={WH} {' '.join(pkgs)} 2>&1 | tail -20", timeout=1800)
    log(out or "(pip ok)")
    code, out = sh("pip show swegemma adk-submission adk-eval-core google-adk litellm 2>&1 | grep -E 'Name|Version'")
    log(out)
code, out = sh("which swegemma; swegemma --help; swegemma eval --help", timeout=120)
log(out)

# ------------------------------------------------------------------ 3. sample submission
section("3. SAMPLE SUBMISSION")
for sd in sample_dirs[:1]:
    for p in sorted(sd.rglob("*")):
        rel = p.relative_to(sd)
        if p.is_file():
            size = p.stat().st_size
            log(f"  {rel}  ({size} B)")
    for p in sorted(sd.rglob("*")):
        if p.is_file() and p.suffix in (".yaml", ".yml", ".md", ".txt", ".py") and p.stat().st_size < 60_000:
            show_file(p)

# ------------------------------------------------------------------ 4. harness source facts
section("4. HARNESS SOURCE: models.yaml, skills, graph paths, tracing")
code, loc = sh("python -c \"import swegemma,os;print(os.path.dirname(swegemma.__file__))\"")
pkg_dirs = [loc.strip().splitlines()[-1]] if code == 0 else []
for mod in ("adk_submission", "adk_eval_core"):
    c, l = sh(f"python -c \"import {mod},os;print(os.path.dirname({mod}.__file__))\"")
    if c == 0:
        pkg_dirs.append(l.strip().splitlines()[-1])
log("package dirs:", pkg_dirs)

def grep(pattern, ctx=2, max_hits=12):
    hits = 0
    for d in pkg_dirs:
        for f in glob.glob(f"{d}/**/*.py", recursive=True):
            try:
                src = Path(f).read_text(errors="replace").splitlines()
            except Exception:  # noqa: BLE001
                continue
            for i, line in enumerate(src):
                if re.search(pattern, line):
                    lo, hi = max(0, i - ctx), min(len(src), i + ctx + 1)
                    log(f"  {f}:{i+1}")
                    for j in range(lo, hi):
                        log(f"    {j+1:5}| {src[j][:160]}")
                    hits += 1
                    if hits >= max_hits:
                        return
    if hits == 0:
        log(f"  (no hits for /{pattern}/)")

for d in pkg_dirs:
    for f in glob.glob(f"{d}/**/*models*.y*ml", recursive=True) + glob.glob(f"{d}/**/*.y*ml", recursive=True)[:5]:
        show_file(f, 2500)
log("\n>> models.yaml parsing:")
grep(r"models_yaml|models\.yaml|load_models", ctx=4)
log("\n>> skill script execution:")
grep(r"run_skill_script|skill.*script|scripts/|SkillRegistry", ctx=4)
log("\n>> graph / embedding file resolution:")
grep(r"data/graphs|graphs_dir|embeddings_dir|\.npz", ctx=3)
log("\n>> ATIF trace step schema:")
grep(r"class .*Step|def add_step|atif", ctx=3, max_hits=6)
log("\n>> concurrency in scoring inference:")
grep(r"concurrency|max_workers|Semaphore", ctx=2, max_hits=8)

# ------------------------------------------------------------------ 5. hardware
section("5. HARDWARE")
log(sh("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv 2>&1")[1])
log(sh("python -c \"import jax;print(jax.devices())\" 2>&1 | tail -2")[1])
log(sh("nproc; free -g | head -2; df -h /kaggle/working | tail -1")[1])
log(sh("pip show vllm 2>&1 | grep -E 'Name|Version'")[1] or "vllm not installed")

# ------------------------------------------------------------------ 6. harness smoke test (no LLM)
if os.environ.get("RUN_SKIP") == "1" and tasks and snap_dirs and sample_dirs and shutil.which("swegemma"):
    section("6. SMOKE TEST: --skip-agent-patch on 2 tasks (Phase 2 only; baseline tests should FAIL)")
    ids = [json.loads(l)["instance_id"] for l in open(tasks[0]) if l.strip()][:2]
    res = OUT / "probe_skip_run"
    cmd = (f"swegemma eval --tasks {tasks[0]} --snapshots-dir {snap_dirs[0]} "
           f"--submission-dir {sample_dirs[0]} --results-dir {res} --sandbox subprocess "
           f"--skip-agent-patch --display plain --task-ids {' '.join(ids)}")
    log(cmd)
    code, out = sh(cmd, timeout=1800, max_chars=8000)
    log(f"exit={code}\n{out}")
    for f in [res / "summary.json"] + sorted((res / "test_outputs").glob("*.log"))[:1]:
        if f.exists():
            show_file(f, 3000)

    section("7. SANDBOX CAPABILITIES (coverage / settrace inside the task venv)")
    venvs = sorted(glob.glob("/tmp/swegemma_sandbox_*/venv/bin/python")) + [sys.executable]
    py = venvs[0]
    log("python used:", py)
    log(sh(f"{py} -c \"import coverage;print('coverage',coverage.__version__)\" 2>&1")[1])
    log(sh(f"{py} -c \"import sys;sys.settrace(lambda *a:None);print('settrace ok')\" 2>&1")[1])
    log(sh(f"{py} -m pytest --version 2>&1")[1])
else:
    log("\n(skip smoke test; set RUN_SKIP=1 to run it)")

# ------------------------------------------------------------------ 8. Gemma via Google AI Studio: does tool calling work?
section("8. AI STUDIO GEMMA API: models + tool-calling check")
key = os.environ.get("GEMINI_API_KEY")
if not key:
    try:
        from kaggle_secrets import UserSecretsClient  # Kaggle: Add-ons -> Secrets -> label GEMINI_API_KEY
        key = UserSecretsClient().get_secret("GEMINI_API_KEY")
    except Exception:  # noqa: BLE001
        key = None
if not key:
    log("(no GEMINI_API_KEY secret; skipped)")
else:
    import urllib.request
    BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
    def call(path, payload=None):
        req = urllib.request.Request(BASE + path, method="POST" if payload else "GET",
                                     data=json.dumps(payload).encode() if payload else None,
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            body = getattr(e, "read", lambda: b"")().decode(errors="replace")[:800]
            return {"error": f"{e} {body}"}
    ms = call("/models")
    gemma = sorted(m["id"] for m in ms.get("data", []) if "gemma" in m.get("id", "").lower())
    log("gemma models on the API:", gemma or ms)
    tool = {"type": "function", "function": {"name": "read_file", "description": "Read a file",
            "parameters": {"type": "object", "properties": {"filepath": {"type": "string"},
            "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["filepath"]}}}
    for mid in [m for m in gemma if "31b" in m][:2] or gemma[:1]:
        r = call("/chat/completions", {"model": mid, "tools": [tool], "tool_choice": "auto",
                 "messages": [{"role": "user", "content": "Read lines 1-20 of src/app.py using the tool."}]})
        msg = (r.get("choices") or [{}])[0].get("message", {})
        log(f"{mid}: tool_calls={json.dumps(msg.get('tool_calls'))[:400]}  content={str(msg.get('content'))[:200]}  "
            f"usage={r.get('usage')}  error={str(r.get('error'))[:400]}")

# ------------------------------------------------------------------ 9. multi-turn tool loop through LiteLLM (what the harness uses)
section("9. LITELLM MULTI-TURN TOOL LOOP (harness path: LiteLlm openai/<model>)")
if key:
    try:
        import litellm
        litellm.drop_params = True
        tool = {"type": "function", "function": {"name": "read_file", "description": "Read a file",
                "parameters": {"type": "object", "properties": {"filepath": {"type": "string"},
                "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["filepath"]}}}
        kw = dict(model="openai/gemma-4-31b-it", api_base="https://generativelanguage.googleapis.com/v1beta/openai",
                  api_key=key, tools=[tool], temperature=0.2)
        msgs = [{"role": "user", "content": "Use read_file on src/app.py lines 1-5, then tell me what the file does."}]
        r1 = litellm.completion(messages=msgs, **kw)
        m1 = r1.choices[0].message
        log("turn1 tool_calls:", str(m1.tool_calls)[:300])
        if m1.tool_calls:
            msgs.append(m1.model_dump(exclude_none=True))
            msgs.append({"role": "tool", "tool_call_id": m1.tool_calls[0].id,
                         "content": json.dumps({"status": "ok", "content": "from fastapi import FastAPI\napp = FastAPI()\n"})})
            try:
                r2 = litellm.completion(messages=msgs, **kw)
                log("turn2 OK:", str(r2.choices[0].message.content)[:300])
            except Exception as e:  # noqa: BLE001
                log("turn2 FAILED (thought_signature / format issue?):", str(e)[:800])
        t0 = __import__("time").time()
        for _ in range(3):
            litellm.completion(messages=[{"role": "user", "content": "Say ok."}], **{k: v for k, v in kw.items() if k != "tools"})
        log(f"latency: {(__import__('time').time()-t0)/3:.1f}s per short call")
    except Exception as e:  # noqa: BLE001
        log("litellm test error:", str(e)[:800])
else:
    log("(no key; skipped)")

REPORT.write_text("\n".join(lines))
print(f"\nSaved -> {REPORT}  (upload this file back to Claude)")
