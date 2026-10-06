"""
Gemma 4 Developer Agent — local BASELINE on Kaggle TPU v5e-8 (vLLM-TPU) + the OFFICIAL harness.

Why TPU: v5e-8 has 8 x 16 GB HBM = 128 GB, so Gemma-4-31B fits in bf16 across 8 chips. vLLM on TPU is roughly
~10x faster per stream than llama.cpp on T4, and it can serve several tasks at once. So all 129 public tasks fit
in one ~9 h session, and the GPU quota stays free for other work.
Fidelity: same server family and parsers as the scorer (vLLM, gemma4 tool/reasoning parsers). The weights are
the W4A16 competition checkpoint if vLLM-TPU can load it, otherwise the bf16 Gemma-4-31B-it (not bit-identical
to the scorer's INT4).

Notebook setup:
  - Accelerator: TPU v5e-8.   Internet: ON.
  - Inputs: competition data · dataset `gemma-4-developer-agent-wheelhouse` (owner: metric) ·
            Model `google/gemma-4` -> gemma-4-31b-it-qat-w4a16-ct  AND (recommended) the bf16 `gemma-4-31b-it`
            variant as fallback. Optional secret HF_TOKEN for a Hugging Face fallback download.
  - Run via "Save Version -> Save & Run All (Commit)" (TPU sessions cap at 9 h).
Outputs in /kaggle/working: run.log · baseline_tpu.tgz (traces, logs, patches, results) · submission.zip
"""

import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

CFG = {
    "task_ids": None,           # None = all 129 public tasks
    "concurrency": 4,           # tasks in parallel (the scorer runs sequentially; parallel is fine for analysis)
    "max_num_seqs": 8,          # vLLM batch slots (analyzer sub-agent calls share them)
    "local_max_minutes": 10,
    "max_tool_calls": 40,
    "max_turns": 100,
    "timeout_seconds": 180,
    "results": "baseline_tpu",
    "session_budget_h": 8.4,    # stop scheduling before Kaggle's 9 h TPU cap
    "scorer_eval": {"timeout_seconds": 180, "max_tool_calls": 40, "max_time_minutes": 4.5, "max_turns": 100},
}
W = pathlib.Path("/kaggle/working")
VT = pathlib.Path("/tmp/vt")    # venv: vLLM-TPU server
HV = pathlib.Path("/tmp/hv")    # venv: official harness (kept separate to avoid dependency clashes)
ALIAS = "gemma-4-31b-it-qat-w4a16-ct"
PORT = 8000
LOG = open(W / "run.log", "a", buffering=1)
T0 = time.time()


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")


def sh(cmd, timeout=None, check=False):
    log("$", cmd[:220])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = ((r.stdout or "")[-1500:] + ("\n" + r.stderr[-1500:] if (r.stderr or "").strip() else "")).strip()
    if out:
        log(out)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED rc={r.returncode}: {cmd[:120]}")
    return r


def first(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None


def secret(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ 0. hardware + inputs
sh("nproc; free -g | head -2; df -h /tmp | tail -1; ls /dev | grep -i -E 'accel|vfio' | head")
tasks_path = first("/kaggle/input/**/tasks.jsonl")
assert tasks_path, "competition data not attached"
DATA = pathlib.Path(tasks_path).parent
whl = first("/kaggle/input/**/swegemma-*.whl")
assert whl, "attach dataset 'gemma-4-developer-agent-wheelhouse' (owner: metric)"
WH = pathlib.Path(whl).parent
cfgs = [pathlib.Path(p).parent for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if "31b" in p.lower()]
w4a16 = next((str(p) for p in cfgs if "w4a16" in str(p).lower()), None)
bf16 = next((str(p) for p in cfgs if "w4a16" not in str(p).lower() and "gguf" not in str(p).lower()), None)
log("DATA =", DATA, "| WHEELHOUSE =", WH, "| w4a16 =", w4a16, "| bf16 =", bf16)

# ------------------------------------------------------------------ 1. two isolated venvs
sh(f"{sys.executable} -m pip install -q uv", check=True)
if not (VT / "bin" / "vllm").exists():
    sh(f"uv venv --seed {VT}", check=True)
    sh(f"uv pip install --python {VT}/bin/python vllm-tpu "
       "--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match",
       timeout=3600, check=True)
sh(f"{VT}/bin/python -c 'import vllm, jax; print(\"vllm\", vllm.__version__, \"| devices\", jax.devices())'", check=True)
if not (HV / "bin" / "swegemma").exists():
    sh(f"uv venv --seed {HV}", check=True)
    sh(f"uv pip install --python {HV}/bin/python --find-links {WH} "
       "--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match "
       "swegemma adk-submission adk-eval-core 'google-adk==1.36.1' litellm", timeout=3600, check=True)
sh(f"{HV}/bin/python -c 'import swegemma, google.adk; print(\"harness ok, adk\", google.adk.__version__)'", check=True)

# ------------------------------------------------------------------ 2. start vLLM-TPU (try W4A16 first, then bf16, then HF)
hf_token = secret("HF_TOKEN")
candidates = [m for m in (w4a16, bf16) if m] + ["google/gemma-4-31B-it"]
BASE_FLAGS = ["--tensor-parallel-size", "8", "--max-model-len", "32768", "--max-num-seqs", str(CFG["max_num_seqs"]),
              "--enable-auto-tool-choice", "--tool-call-parser", "gemma4", "--reasoning-parser", "gemma4",
              "--served-model-name", ALIAS, "--host", "127.0.0.1", "--port", str(PORT)]
OPTIONAL = [["--default-chat-template-kwargs", '{"enable_thinking": true}'],   # scorer default
            ["--limit-mm-per-prompt", '{"image": 0, "audio": 0, "video": 0}'],  # text-only: faster startup
            ["--disable_chunked_mm_input"]]
server = None
MODEL_USED = None


def start(model, extra, tag):
    env = {**os.environ, "VLLM_NO_USAGE_STATS": "1", "HF_HOME": "/tmp/hf", "VLLM_ENGINE_READY_TIMEOUT_S": "3000"}
    if hf_token:
        env["HF_TOKEN"] = hf_token
    cmd = [str(VT / "bin" / "vllm"), "serve", model, *BASE_FLAGS, *extra]
    log(f"[{tag}] starting:", " ".join(cmd)[:400])
    p = subprocess.Popen(cmd, stdout=open(W / f"vllm-{tag}.log", "a"), stderr=subprocess.STDOUT, env=env)
    for i in range(240):  # up to 60 min (weights load + XLA compilation)
        time.sleep(15)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5).read()
            log(f"[{tag}] vLLM up after {(i + 1) * 15 / 60:.1f} min")
            return p
        except Exception:  # noqa: BLE001
            if p.poll() is not None:
                log(f"[{tag}] vLLM died rc={p.returncode}:", (W / f"vllm-{tag}.log").read_text()[-2500:])
                return None
    p.kill()
    return None


attempt = 0
for model in candidates:
    for drop in range(len(OPTIONAL) + 1):          # retry with fewer optional flags if an unknown flag kills startup
        extra = [x for grp in OPTIONAL[drop:] for x in grp]
        attempt += 1
        server = start(model, extra, str(attempt))
        if server:
            MODEL_USED = model
            break
        tail = (W / f"vllm-{attempt}.log").read_text()[-3000:].lower()
        if not any(k in tail for k in ("unrecognized arguments", "no such option", "invalid choice", "error: argument")):
            break                                  # not a flag problem -> try the next checkpoint
    if server:
        break
assert server, "vLLM-TPU failed for every checkpoint; send run.log + vllm-*.log to Claude"
(W / "model_used.txt").write_text(MODEL_USED)
log("MODEL_USED =", MODEL_USED)

# sanity + speed: one tool call, one 256-token generation
def chat(payload):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


tool = {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {
    "type": "object", "properties": {"filepath": {"type": "string"}, "start_line": {"type": "integer"},
                                     "end_line": {"type": "integer"}}, "required": ["filepath"]}}}
t = time.time()
r = chat({"model": ALIAS, "temperature": 0.2, "tools": [tool],
          "messages": [{"role": "user", "content": "Read lines 1-5 of src/app.py."}]})
log(f"sanity ({time.time() - t:.0f}s): tool_calls={json.dumps(r['choices'][0]['message'].get('tool_calls'))[:300]}")
t = time.time()
r = chat({"model": ALIAS, "max_tokens": 256, "temperature": 0.2,
          "chat_template_kwargs": {"enable_thinking": False},
          "messages": [{"role": "user", "content": "Explain Python decorators in detail."}]})
n = r.get("usage", {}).get("completion_tokens", 0)
log(f"speed: {n} tokens in {time.time() - t:.1f}s = {n / max(time.time() - t, 1e-6):.1f} tok/s (single stream)")


def watchdog():
    global server
    while True:
        time.sleep(30)
        if server is not None and server.poll() is not None:
            log(f"WATCHDOG: vLLM exited rc={server.returncode}; restarting")
            server = start(MODEL_USED, [], "restart")


threading.Thread(target=watchdog, daemon=True).start()

# ------------------------------------------------------------------ 3. baseline submission (sample agent, no adapters)
SAMPLE = DATA / "sample_submission"
SUB = W / "baseline_submission"
shutil.rmtree(SUB, ignore_errors=True)
for d in ("prompts", "configs", "sub_agents"):
    (SUB / d).mkdir(parents=True)
for f in ("system.md", "analyzer.md"):
    shutil.copy(SAMPLE / "prompts" / f, SUB / "prompts" / f)
(SUB / "configs" / "sampling.yaml").write_text(
    "temperature: 0.2\ntop_p: 0.95\ntop_k: 40\nmax_output_tokens: 8192\n"
    "thinking_config:\n  include_thoughts: false\n  thinking_budget: 0\n")
TOOLS = ["run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch",
         "get_code_neighbors", "search_similar_code", "get_code_subgraph"]
(SUB / "agent.yaml").write_text(
    f"name: swe_baseline_agent\nmodel: {ALIAS}\ninstruction: !include prompts/system.md\ntools:\n"
    + "".join(f"  - {t}\n" for t in TOOLS)
    + "  - agent_tool:\n      config_path: sub_agents/code_analyzer.yaml\n      skip_summarization: false\n"
    "generate_content_config: !include configs/sampling.yaml\n")
(SUB / "sub_agents" / "code_analyzer.yaml").write_text(
    "name: code_analyzer_agent\ndescription: Analyzes repository source files and symbol graphs to locate root causes.\n"
    f"model: {ALIAS}\ninstruction: !include ../prompts/analyzer.md\ntools:\n"
    "  - read_file\n  - search_similar_code\n  - get_code_neighbors\n  - get_code_subgraph\n"
    "generate_content_config: !include ../configs/sampling.yaml\n")
(SUB / "eval_config.yaml").write_text("evaluation:\n" + "".join(f"  {k}: {v}\n" for k, v in CFG["scorer_eval"].items()))
if (W / "submission.zip").exists():
    (W / "submission.zip").unlink()
shutil.make_archive(str(W / "submission"), "zip", SUB)
log("wrote submission.zip with scorer budgets", CFG["scorer_eval"])

# ------------------------------------------------------------------ 4. official Evaluator (inside the harness venv)
ids = CFG["task_ids"] or [json.loads(l)["instance_id"] for l in open(tasks_path) if l.strip()]
(W / "task_ids.json").write_text(json.dumps(ids, indent=1))
INNER = pathlib.Path("/tmp/inner_eval.py")
INNER.write_text('''
import asyncio, json, pathlib, sys
import litellm
litellm.drop_params = True
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
from swegemma.models.registry import setup_gemma_model_registry
a = json.loads(sys.argv[1])
models = setup_gemma_model_registry(api_base=a["api_base"], api_key="EMPTY", served_model=a["alias"])
limits, gen = build_submission_limits()
d = pathlib.Path(a["data"])
cfg = EvalConfig(
    tasks_path=pathlib.Path(a["tasks"]), snapshots_dir=d / "snapshots", results_dir=pathlib.Path(a["results"]),
    submission_dir=pathlib.Path(a["sub"]), models=models, sandbox="subprocess", wheels_dir=d / "wheels",
    graph_dir=str(d / "graphs"), embeddings_dir=str(d / "embeddings"),
    timeout_seconds=a["timeout"], max_time_minutes=a["minutes"], max_tool_calls=a["calls"], max_turns=a["turns"],
    limits=limits, generation_constraints=gen, adapter_manifest=None,
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(compaction_interval=5, overlap_size=2,
                                                    token_threshold=14336, event_retention_size=5),
    task_ids=a["ids"], concurrency=a["concurrency"], display_mode="quiet", verbose=False)
res = asyncio.run(Evaluator(cfg).run())
print("EVAL_DONE", repr(res)[:2000], flush=True)
''')
args = {"api_base": f"http://127.0.0.1:{PORT}/v1", "alias": ALIAS, "tasks": tasks_path, "data": str(DATA),
        "results": str(W / CFG["results"]), "sub": str(SUB), "timeout": CFG["timeout_seconds"],
        "minutes": CFG["local_max_minutes"], "calls": CFG["max_tool_calls"], "turns": CFG["max_turns"],
        "ids": ids, "concurrency": CFG["concurrency"]}
env = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True", "OTEL_SDK_DISABLED": "true",
       "PATH": f"{HV}/bin:" + os.environ["PATH"]}
remaining_s = CFG["session_budget_h"] * 3600 - (time.time() - T0)
log(f"EVAL start: {len(ids)} tasks, concurrency {CFG['concurrency']}, {remaining_s / 3600:.1f} h left in session")
t = time.time()
with open(W / "eval_stdout.log", "w") as f:
    try:
        rc = subprocess.run([str(HV / "bin" / "python"), str(INNER), json.dumps(args)], stdout=f,
                            stderr=subprocess.STDOUT, env=env, timeout=max(remaining_s, 600)).returncode
    except subprocess.TimeoutExpired:
        rc = "session-budget timeout (partial results kept: task_results.jsonl is append-only)"
log(f"EVAL finished rc={rc} in {(time.time() - t) / 60:.0f} min")
log((W / "eval_stdout.log").read_text()[-3000:])

# ------------------------------------------------------------------ 5. failure analysis (if trace_analyzer.py was uploaded next to this script)
ana = next((p for p in [pathlib.Path("trace_analyzer.py"), *map(pathlib.Path, glob.glob("/kaggle/input/**/trace_analyzer.py", recursive=True))] if p.exists()), None)
if ana:
    sh(f"{sys.executable} {ana} --results {W / CFG['results']} --tasks {tasks_path} --out {W / 'analysis'}")
else:
    log("trace_analyzer.py not found next to the script; skipping analysis (upload it and rerun only this step)")

# ------------------------------------------------------------------ 6. package
res_dir = W / CFG["results"]
if (res_dir / "summary.json").exists():
    log("SUMMARY", (res_dir / "summary.json").read_text()[:3000])
sh(f"cd {W} && tar -czf {CFG['results']}.tgz {CFG['results']} task_ids.json run.log eval_stdout.log "
   f"model_used.txt vllm-*.log analysis 2>/dev/null; ls -la {W}/*.tgz")
if server is not None:
    server.terminate()
log(f"RUN_DONE in {(time.time() - T0) / 3600:.1f} h")
