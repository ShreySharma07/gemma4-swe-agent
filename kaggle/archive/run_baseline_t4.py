"""
Gemma 4 Developer Agent — local BASELINE run on Kaggle T4 x2 with the real 31B model.

  Model   : official Gemma-4-31B QAT weights as Q4_0 GGUF, served by llama.cpp (vLLM's W4A16 kernels need
            Ampere+, so they don't run on T4). Same QAT weights, different 4-bit packing and server.
  Harness : the OFFICIAL swegemma 0.2.7 from the `metric/gemma-4-developer-agent-wheelhouse` dataset,
            subprocess sandbox, tasks run one at a time (like the scorer).
  Agent   : organisers' sample agent (coder + read-only analyzer), adapters removed, public-consensus
            sampling (temp 0.2, thinking off).

Notebook setup:
  - Accelerator: GPU T4 x2.  Internet: ON.
  - Inputs: competition data + `gemma-4-developer-agent-wheelhouse` (owner: metric).
  - Optional secret HF_TOKEN (only needed for the official gated GGUF; otherwise an ungated mirror is used).
  - Run with "Save Version -> Save & Run All (Commit)" so it runs in the background (~6-7 h for 36 tasks).

Outputs in /kaggle/working:
  run.log · baseline_t4.tgz (traces, logs, patches, task_results, summary) · task_ids.json
  submission.zip  (the same agent with scorer budgets -> can be uploaded to the leaderboard as-is)
"""

import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

CFG = {
    "n_tasks": 36,              # stratified sample of the 129 public tasks
    "per_repo": {"fastapi": 18, "rich": 12, "requests": 6},   # matched on repo short name
    "seed": 0,
    "local_max_minutes": 12,    # T4 ~12 tok/s is ~2-3x slower than the scorer, so give wall time and cap by calls
    "max_tool_calls": 40,
    "max_turns": 100,
    "timeout_seconds": 180,
    "results": "baseline_t4",
    # budgets written into submission.zip for the real scorer (sequential ~120 tasks in 12 h)
    "scorer_eval": {"timeout_seconds": 180, "max_tool_calls": 40, "max_time_minutes": 4.5, "max_turns": 100},
}
W = pathlib.Path("/kaggle/working")
HV = pathlib.Path("/tmp/hv")
ALIAS = "gemma-4-31b-it-qat-w4a16-ct"
PORT = 8000
LOG = open(W / "run.log", "a", buffering=1)
T0 = time.time()


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")


def sh(cmd, timeout=None, check=False, quiet=False):
    if not quiet:
        log("$", cmd[:220])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = ((r.stdout or "")[-1500:] + ("\n" + r.stderr[-1500:] if (r.stderr or "").strip() else "")).strip()
    if out and not quiet:
        log(out)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED rc={r.returncode}: {cmd[:120]}\n{out}")
    return r


def find1(pattern):
    r = subprocess.run(f"find /kaggle/input -path '{pattern}' 2>/dev/null | head -1", shell=True,
                       capture_output=True, text=True)
    return r.stdout.strip()


def secret(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ 0. hardware + inputs
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | head -2; df -h /tmp | tail -1")
ngpu = int(sh("nvidia-smi -L | wc -l", quiet=True).stdout.strip() or 0)
if ngpu < 2:
    raise SystemExit("Need GPU T4 x2 (Settings -> Accelerator).")
tasks_path = find1("*/tasks.jsonl")
assert tasks_path, "competition data not attached"
DATA = pathlib.Path(tasks_path).parent
whl = find1("*/swegemma-*.whl")
assert whl, "attach the dataset 'gemma-4-developer-agent-wheelhouse' (owner: metric)"
WH = pathlib.Path(whl).parent
log("DATA =", DATA, "| WHEELHOUSE =", WH)

# ------------------------------------------------------------------ 1. official harness in an isolated venv
if not (HV / "bin" / "swegemma").exists():
    sh(f"{sys.executable} -m pip install -q uv", check=True)
    sh(f"uv venv --seed {HV}", check=True)
    sh(f"uv pip install --python {HV}/bin/python --find-links {WH} "
       "--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match "
       "swegemma adk-submission adk-eval-core 'google-adk==1.36.1' litellm", timeout=3600, check=True)
sh(f"{HV}/bin/python -c 'import swegemma, google.adk; print(\"harness ok, adk\", google.adk.__version__)'", check=True)

# ------------------------------------------------------------------ 2. llama.cpp (CUDA, T4 = sm75)
SERVER_BIN = pathlib.Path("/tmp/llama.cpp/build/bin/llama-server")
prebuilt = find1("*/llama-server")  # reuse a build saved as a dataset in a later run
if prebuilt:
    SERVER_BIN = pathlib.Path(prebuilt)
    sh(f"chmod +x {SERVER_BIN}")
elif not SERVER_BIN.exists():
    sh("mkdir -p /tmp/cudalib && d=$(dirname $(find /usr/lib /usr/local -name 'libcuda.so.1' 2>/dev/null | head -1)) "
       "&& ln -sf $d/libcuda.so.1 /tmp/cudalib/libcuda.so")
    sh("cd /tmp && rm -rf llama.cpp && git clone -q --depth 1 https://github.com/ggml-org/llama.cpp", check=True)
    sh("cd /tmp/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF "
       "-DBUILD_SHARED_LIBS=OFF -DCMAKE_LIBRARY_PATH='/tmp/cudalib;/usr/local/cuda/lib64/stubs' "
       "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc > /tmp/cmake.log 2>&1; "
       "cmake --build build --config Release -j 4 --target llama-server 2>&1 | tail -3", timeout=5400)
    assert SERVER_BIN.exists(), "llama.cpp build failed (see /tmp/cmake.log)"
    shutil.copy(SERVER_BIN, W / "llama-server")  # static binary: save it as a dataset later to skip the ~20 min build
log("llama-server:", SERVER_BIN)

# ------------------------------------------------------------------ 3. weights (Q4_0 GGUF of the QAT checkpoint)
gguf = find1("*31[bB]*[qQ]4_0*.gguf")
if not gguf:
    sh(f"{sys.executable} -m pip install -q huggingface_hub", check=True)
    from huggingface_hub import hf_hub_download, list_repo_files
    token = secret("HF_TOKEN")
    candidates = [("google/gemma-4-31B-it-qat-q4_0-gguf", token),
                  ("mradermacher/gemma-4-31B-it-qat-q4_0-unquantized-GGUF", None)]
    for repo, tok in candidates:
        try:
            files = [f for f in list_repo_files(repo, token=tok) if f.endswith(".gguf")
                     and "q4_0" in f.lower() and "mmproj" not in f.lower()]
            if not files:
                log("no Q4_0 file in", repo)
                continue
            t = time.time()
            gguf = hf_hub_download(repo, sorted(files, key=len)[0], local_dir="/tmp/gguf", token=tok)
            log(f"weights from {repo} in {time.time() - t:.0f}s ->", gguf)
            break
        except Exception as e:  # noqa: BLE001
            log(f"cannot download from {repo}: {str(e)[:200]}")
assert gguf, "no GGUF weights available"
(W / "weights_source.txt").write_text(gguf)

# ------------------------------------------------------------------ 4. serve (OpenAI-compatible, tool calls via --jinja)
SERVER_CMD = [str(SERVER_BIN), "-m", gguf, "-ngl", "999", "-sm", "layer", "-c", "32768", "-np", "1", "--jinja",
              "-fa", "on", "--reasoning-format", "auto", "--no-mmap", "--host", "127.0.0.1", "--port", str(PORT),
              "--alias", ALIAS]
server = None


def start_server(tag):
    p = subprocess.Popen(SERVER_CMD, stdout=open(W / f"llama-{tag}.log", "a"), stderr=subprocess.STDOUT)
    for i in range(100):
        time.sleep(15)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5).read()
            log(f"llama-server up ({tag}) after {(i + 1) * 15}s")
            return p
        except Exception:  # noqa: BLE001
            if p.poll() is not None:
                log(f"llama-server died ({tag}):", (W / f"llama-{tag}.log").read_text()[-1500:])
                return None
    return None


server = start_server("0")
if server is None:
    raise SystemExit("llama-server failed to start")


def watchdog():  # the server occasionally gets OOM-killed on Kaggle; restart it so the run continues
    global server
    n = 0
    while True:
        time.sleep(20)
        if server is not None and server.poll() is not None:
            n += 1
            log(f"WATCHDOG: server exited rc={server.returncode}, restart #{n}")
            server = start_server(str(n))


threading.Thread(target=watchdog, daemon=True).start()

# sanity: one tool call through the server
req = {"model": ALIAS, "temperature": 0.2, "messages": [{"role": "user", "content": "Read lines 1-5 of src/app.py."}],
       "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {
           "type": "object", "properties": {"filepath": {"type": "string"}, "start_line": {"type": "integer"},
                                            "end_line": {"type": "integer"}}, "required": ["filepath"]}}}]}
t = time.time()
r = json.loads(urllib.request.urlopen(urllib.request.Request(
    f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(req).encode(),
    headers={"Content-Type": "application/json"}), timeout=600).read())
m = r["choices"][0]["message"]
log(f"sanity ({time.time() - t:.0f}s): tool_calls={json.dumps(m.get('tool_calls'))[:300]} usage={r.get('usage')}")

# ------------------------------------------------------------------ 5. baseline submission (sample agent, no adapters)
SAMPLE = DATA / "sample_submission"
SUB = W / "baseline_submission"
shutil.rmtree(SUB, ignore_errors=True)
(SUB / "prompts").mkdir(parents=True)
(SUB / "configs").mkdir()
(SUB / "sub_agents").mkdir()
shutil.copy(SAMPLE / "prompts" / "system.md", SUB / "prompts" / "system.md")
shutil.copy(SAMPLE / "prompts" / "analyzer.md", SUB / "prompts" / "analyzer.md")
(SUB / "configs" / "sampling.yaml").write_text(
    "temperature: 0.2\ntop_p: 0.95\ntop_k: 40\nmax_output_tokens: 8192\n"
    "thinking_config:\n  include_thoughts: false\n  thinking_budget: 0\n")
(SUB / "agent.yaml").write_text(f"""name: swe_baseline_agent
model: {ALIAS}
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
  - agent_tool:
      config_path: sub_agents/code_analyzer.yaml
      skip_summarization: false
generate_content_config: !include configs/sampling.yaml
""")
(SUB / "sub_agents" / "code_analyzer.yaml").write_text(f"""name: code_analyzer_agent
description: Analyzes repository source files and symbol graphs to locate root causes.
model: {ALIAS}
instruction: !include ../prompts/analyzer.md
tools:
  - read_file
  - search_similar_code
  - get_code_neighbors
  - get_code_subgraph
generate_content_config: !include ../configs/sampling.yaml
""")
se = CFG["scorer_eval"]
(SUB / "eval_config.yaml").write_text("evaluation:\n" + "".join(f"  {k}: {v}\n" for k, v in se.items()))
if (W / "submission.zip").exists():
    (W / "submission.zip").unlink()
shutil.make_archive(str(W / "submission"), "zip", SUB)
log("wrote", W / "submission.zip", "(scorer budgets:", se, ")")

# ------------------------------------------------------------------ 6. task sample
tasks = [json.loads(l) for l in open(tasks_path) if l.strip()]
rng = random.Random(CFG["seed"])
ids = []
for repo, k in CFG["per_repo"].items():
    pool = sorted(t["instance_id"] for t in tasks if t["repo"].split("/")[-1].lower() == repo)
    ids += rng.sample(pool, min(k, len(pool)))
(W / "task_ids.json").write_text(json.dumps(ids, indent=1))
log(f"{len(ids)} tasks:", ids)

# ------------------------------------------------------------------ 7. run the official Evaluator inside the venv
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
cfg = EvalConfig(
    tasks_path=pathlib.Path(a["tasks"]), snapshots_dir=pathlib.Path(a["data"]) / "snapshots",
    results_dir=pathlib.Path(a["results"]), submission_dir=pathlib.Path(a["sub"]), models=models,
    sandbox="subprocess", wheels_dir=pathlib.Path(a["data"]) / "wheels",
    graph_dir=str(pathlib.Path(a["data"]) / "graphs"), embeddings_dir=str(pathlib.Path(a["data"]) / "embeddings"),
    timeout_seconds=a["timeout"], max_time_minutes=a["minutes"], max_tool_calls=a["calls"], max_turns=a["turns"],
    limits=limits, generation_constraints=gen, adapter_manifest=None,
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(compaction_interval=5, overlap_size=2,
                                                    token_threshold=14336, event_retention_size=5),
    task_ids=a["ids"], concurrency=1, display_mode="quiet", verbose=False)
res = asyncio.run(Evaluator(cfg).run())
print("EVAL_DONE", repr(res)[:2000], flush=True)
''')
args = {"api_base": f"http://127.0.0.1:{PORT}/v1", "alias": ALIAS, "tasks": tasks_path, "data": str(DATA),
        "results": str(W / CFG["results"]), "sub": str(SUB), "timeout": CFG["timeout_seconds"],
        "minutes": CFG["local_max_minutes"], "calls": CFG["max_tool_calls"], "turns": CFG["max_turns"], "ids": ids}
env = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True", "OTEL_SDK_DISABLED": "true",
       "PATH": f"{HV}/bin:" + os.environ["PATH"]}
log("EVAL start:", {k: v for k, v in args.items() if k != "ids"})
t = time.time()
with open(W / "eval_stdout.log", "w") as f:
    rc = subprocess.run([str(HV / "bin" / "python"), str(INNER), json.dumps(args)], stdout=f,
                        stderr=subprocess.STDOUT, env=env).returncode
log(f"EVAL finished rc={rc} in {(time.time() - t) / 60:.0f} min")
log((W / "eval_stdout.log").read_text()[-3000:])

# ------------------------------------------------------------------ 8. package
res_dir = W / CFG["results"]
if (res_dir / "summary.json").exists():
    log("SUMMARY", (res_dir / "summary.json").read_text()[:3000])
sh(f"cd {W} && tar -czf {CFG['results']}.tgz {CFG['results']} task_ids.json run.log eval_stdout.log "
   f"weights_source.txt llama-*.log 2>/dev/null; ls -la {W}/*.tgz")
if server is not None:
    server.terminate()
log(f"RUN_DONE in {(time.time() - T0) / 3600:.1f} h")
