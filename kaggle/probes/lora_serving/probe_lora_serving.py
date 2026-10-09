"""
Phase 0 go/no-go: can the scorer stack serve a PEFT LoRA on gemma-4-31b-it-qat-w4a16-ct, and does `adapter:` route to it?

Kaggle: GPU L4 x4 (offline). Start it from the web editor (the API can't allocate L4 on this account).
Inputs: output of kernel gemma4-scorer-env (scorer_env.tgz) + wheelhouse dataset + model gemma-4-31b-it-qat-w4a16-ct.

Steps
  1. restore the prebuilt Python 3.12 scorer env (vLLM 0.19.1 + harness, wheelhouse --no-deps + filled deps)
  2. read the checkpoint's language-model linear modules + true (out, in) shapes
  3. write two synthetic PEFT adapters (rank 8): zero_lora (B = 0 -> must equal base) and noise_lora (B != 0 -> must differ)
  4. start the official VllmServer exactly like the scorer (TP=4, 32k, gpu 0.80, gemma4 parsers, enable_lora, 8 x r128)
  5. base vs zero vs noise at temperature 0 (plain + tool call), tokens/s, GPU memory
  6. compile a submission with `adapter: noise_lora` with the official compiler and check the agent's model id
Output: /kaggle/working/lora_probe.json (+ run.log). Verdict line at the end.
"""
import glob
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

W = pathlib.Path("/kaggle/working")
LOG = open(W / "run.log", "a", buffering=1)
REPORT = {}
T0 = time.time()


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")


def sh(cmd, timeout=None, check=False):
    log("$", cmd[:220])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = ((r.stdout or "")[-1500:] + ("\n" + r.stderr[-2500:] if (r.stderr or "").strip() else "")).strip()
    if out:
        log(out)
    if check and r.returncode != 0:
        save()
        raise SystemExit(f"FAILED rc={r.returncode}: {cmd[:120]}")
    return r


def save():
    REPORT["elapsed_min"] = round((time.time() - T0) / 60, 1)
    (W / "lora_probe.json").write_text(json.dumps(REPORT, indent=1, default=str))


def first(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None


# ------------------------------------------------------------------ 0. inputs
gpus = sh("nvidia-smi --query-gpu=name --format=csv,noheader").stdout.split("\n")
gpus = [g for g in gpus if g.strip()]
REPORT["gpus"] = gpus
if len(gpus) < 4 or not all("L4" in g for g in gpus):  # W4A16 needs sm80+; don't burn quota on the wrong machine
    save()
    raise SystemExit(f"need 4x L4, got {gpus} -> check machine_shape in kernel-metadata.json")
whl = first("/kaggle/input/**/swegemma-*.whl")
assert whl, "attach dataset metric/gemma-4-developer-agent-wheelhouse"
WH = pathlib.Path(whl).parent


def is_w4a16(cfg_path):
    if "w4a16" in cfg_path.lower():
        return True
    try:
        c = json.load(open(cfg_path))
        q = c.get("quantization_config") or c.get("text_config", {}).get("quantization_config") or {}
        return "31b" in cfg_path.lower() and "compressed" in str(q.get("quant_method", "")).lower()
    except Exception:  # noqa: BLE001
        return False


cfgs = [p for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if is_w4a16(p)]
assert cfgs, "attach model google/gemma-4 -> gemma-4-31b-it-qat-w4a16-ct"
MODEL_PATH = str(pathlib.Path(sorted(cfgs)[0]).parent)
log("MODEL =", MODEL_PATH, "| WHEELHOUSE =", WH)

# ------------------------------------------------------------------ 1. scorer stack on this Python
for k, v in {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "TRANSFORMERS_NO_TF": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
             "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1", "VLLM_ENGINE_READY_TIMEOUT_S": "1200",
             "VLLM_NO_USAGE_STATS": "1", "OTEL_SDK_DISABLED": "true",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}.items():
    os.environ[k] = v
VENV = pathlib.Path("/tmp/scorer_host")
INSTALL_INFO = VENV / "install_info.json"
if pathlib.Path(sys.prefix) != VENV:
    # L4 sessions are offline: restore the env built by kaggle/env/build_scorer_env.py (CPU kernel, Internet ON),
    # attached via kernel_sources. It unpacks to the same absolute paths it was built at.
    env_tgz = first("/kaggle/input/**/scorer_env.tgz")
    assert env_tgz, "attach the output of the gemma4-scorer-env kernel (kernel_sources)"
    t = time.time()
    sh(f"tar -xzf {env_tgz} -C /", timeout=1800, check=True)
    REPORT["env_restore_min"] = round((time.time() - t) / 60, 1)
    LOG.close()
    os.execv(f"{VENV}/bin/python", [f"{VENV}/bin/python", os.path.abspath(__file__)])
REPORT.update(json.loads(INSTALL_INFO.read_text()))
REPORT["python"] = sys.version.split()[0]
r = sh(f"{sys.executable} -c 'import torch, vllm, swegemma, google.adk, adk_submission; "
       "print(\"torch\", torch.__version__, \"vllm\", vllm.__version__, \"adk\", google.adk.__version__)'")
REPORT["import_check"] = (r.stdout or r.stderr).strip()[-400:]
if r.returncode != 0:
    save()
    raise SystemExit("scorer stack does not import in the venv")

# ------------------------------------------------------------------ 2. language-model linear modules + shapes
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SKIP = ("vision", "audio", "multi_modal", "embed_", "mm_")
idx_file = pathlib.Path(MODEL_PATH) / "model.safetensors.index.json"
weight_map = json.load(open(idx_file))["weight_map"] if idx_file.exists() else \
    {k: "model.safetensors" for k in safe_open(str(pathlib.Path(MODEL_PATH) / "model.safetensors"), "pt").keys()}
modules = {}
for key, fname in weight_map.items():
    mod, leaf = key.rsplit(".", 1)
    if not leaf.startswith("weight") or mod.rsplit(".", 1)[-1] not in TARGETS or any(s in mod for s in SKIP):
        continue
    modules.setdefault(mod, {})[leaf] = fname
shapes = {}
handles = {}
for mod, leaves in sorted(modules.items()):
    if "weight_shape" in leaves:
        f = handles.setdefault(leaves["weight_shape"], safe_open(str(pathlib.Path(MODEL_PATH) / leaves["weight_shape"]), "pt"))
        shapes[mod] = tuple(int(x) for x in f.get_tensor(f"{mod}.weight_shape").tolist())
    elif "weight" in leaves:
        f = handles.setdefault(leaves["weight"], safe_open(str(pathlib.Path(MODEL_PATH) / leaves["weight"]), "pt"))
        shapes[mod] = tuple(f.get_slice(f"{mod}.weight").get_shape())
REPORT["n_target_modules"] = len(shapes)
REPORT["module_examples"] = {m: shapes[m] for m in list(shapes)[:8]}
REPORT["distinct_shapes"] = sorted({(m.rsplit(".", 1)[-1],) + s for m, s in shapes.items()})
log(f"{len(shapes)} target modules; distinct shapes: {REPORT['distinct_shapes']}")
assert shapes, "no target modules found in the checkpoint"

# ------------------------------------------------------------------ 3. synthetic PEFT adapters
SUB = W / "probe_submission"
R = 8
g = torch.Generator().manual_seed(0)
for name, scale in (("zero_lora", 0.0), ("noise_lora", 0.05)):
    d = SUB / "adapters" / name
    d.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for mod, (out_f, in_f) in shapes.items():
        tensors[f"base_model.model.{mod}.lora_A.weight"] = (torch.randn(R, in_f, generator=g) / in_f ** 0.5).to(torch.bfloat16)
        tensors[f"base_model.model.{mod}.lora_B.weight"] = (torch.randn(out_f, R, generator=g) * scale).to(torch.bfloat16)
    save_file(tensors, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": R, "lora_alpha": 2 * R, "lora_dropout": 0.0,
        "bias": "none", "fan_in_fan_out": False, "target_modules": list(TARGETS), "base_model_name_or_path": MODEL_PATH,
        "inference_mode": True}, indent=1))
    REPORT[f"{name}_mb"] = round((d / "adapter_model.safetensors").stat().st_size / 1e6, 1)
(SUB / "agent.yaml").write_text(
    "name: lora_probe_agent\nmodel: gemma-4-31b-it-qat-w4a16-ct\nadapter: noise_lora\n"
    "instruction: You are a software engineer. Fix the issue and call submit_patch.\n"
    "tools:\n  - run_command\n  - submit_patch\n"
    "generate_content_config:\n  temperature: 0.2\n  max_output_tokens: 1024\n")
(SUB / "eval_config.yaml").write_text("evaluation:\n  max_tool_calls: 5\n  max_time_minutes: 1\n")
save()

# ------------------------------------------------------------------ 4. official VllmServer, scorer settings
import litellm  # noqa: E402

litellm.drop_params = True
from adk_submission import VllmConfig, VllmServer, discover_adapters  # noqa: E402
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, build_submission_limits  # noqa: E402

adapters = discover_adapters(str(SUB), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
REPORT["discovered_adapters"] = list(getattr(adapters, "adapters", {}) or {})
log("adapters:", REPORT["discovered_adapters"])
server = VllmServer(VllmConfig(
    model=MODEL_PATH, port=8000, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
    default_chat_template_kwargs={"enable_thinking": True}, max_model_len=32768,
    dtype="bfloat16" if torch.cuda.is_bf16_supported() else "auto", gpu_memory_utilization=0.80,
    enable_auto_tool_choice=True, enable_lora=True, max_loras=8, max_lora_rank=128,
    tensor_parallel_size=4, startup_timeout=60 * 25), adapter_manifest=adapters)
t = time.time()
try:
    server.start()
    REPORT["server_start_s"] = round(time.time() - t)
except Exception as e:  # noqa: BLE001
    msg = f"{type(e).__name__}: {e}"
    REPORT["server_error"] = msg[:300] + "\n...\n" + msg[-6000:]  # the cause is at the end of the server output
    log("SERVER ERROR TAIL:\n" + msg[-6000:])
    save()
    sh("ls -t /tmp/*.log /kaggle/working/*.log 2>/dev/null | head; tail -60 $(ls -t /tmp/*vllm*.log 2>/dev/null | head -1) 2>/dev/null")
    raise SystemExit("vLLM failed to start with the adapters -> NO-GO (see server_error)")
log(f"vLLM up after {REPORT['server_start_s']}s")
REPORT["gpu_mem_after_start"] = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader").stdout.split("\n")
BASE = server.base_url.rstrip("/")


def api(path, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


served = [m["id"] for m in api("/models")["data"]]
REPORT["served_models"] = served
log("served models:", served)
base_id = next(m for m in served if m not in ("zero_lora", "noise_lora"))

# ------------------------------------------------------------------ 5. base vs zero vs noise
PROMPT = [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number, then explain it in two sentences."}]
TOOLS = [{"type": "function", "function": {"name": "run_command", "description": "Run a shell command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
TOOL_MSG = [{"role": "user", "content": "List the Python files under /workspace/src using the run_command tool."}]
outs = {}
for mid in (base_id, "zero_lora", "noise_lora"):
    rec = {}
    try:
        t = time.time()
        r = api("/chat/completions", {"model": mid, "messages": PROMPT, "temperature": 0, "max_tokens": 256,
                                      "chat_template_kwargs": {"enable_thinking": False}})
        dt = time.time() - t
        txt = r["choices"][0]["message"].get("content") or ""
        ntok = r.get("usage", {}).get("completion_tokens", 0)
        rec.update(text=txt[:600], completion_tokens=ntok, tok_per_s=round(ntok / dt, 1))
        r = api("/chat/completions", {"model": mid, "messages": TOOL_MSG, "tools": TOOLS, "temperature": 0,
                                      "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}})
        msg = r["choices"][0]["message"]
        rec["tool_calls"] = [(c["function"]["name"], c["function"]["arguments"]) for c in (msg.get("tool_calls") or [])]
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:800]
    outs[mid] = rec
    log(mid, json.dumps(rec)[:500])
REPORT["outputs"] = outs
b, z, n = outs[base_id], outs["zero_lora"], outs["noise_lora"]
REPORT["zero_equals_base"] = "error" not in z and z.get("text") == b.get("text")
REPORT["noise_differs_from_base"] = "error" not in n and n.get("text") != b.get("text")
save()

# ------------------------------------------------------------------ 6. official compiler: does `adapter:` route?
try:
    from adk_submission.compiler import compile_submission
    from swegemma.models.discovery import validate_single_declared_model

    declared = validate_single_declared_model(SUB)
    models = server.create_model_registry(aliases=[declared], model_prefix="openai/", api_key="EMPTY")

    def mk(nm):
        def f(**kw):
            return "{}"
        f.__name__ = nm
        f.__doc__ = nm
        return f

    tools = {nm: mk(nm) for nm in ("run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch")}
    limits, gen = build_submission_limits()
    from swegemma.models import resolve_swegemma_adapter  # what swegemma's agent_runner passes on the scorer

    agent = compile_submission(SUB, tools, models, limits=limits, generation_constraints=gen, adapter_manifest=adapters,
                               adapter_resolver_fn=lambda base, info: resolve_swegemma_adapter(base, info, models))
    m = getattr(agent, "model", None)
    REPORT["compiled_agent_model"] = getattr(m, "model", None) or str(m)
    REPORT["adapter_routed"] = "noise_lora" in str(REPORT["compiled_agent_model"])
except Exception as e:  # noqa: BLE001
    REPORT["compile_error"] = f"{type(e).__name__}: {e}"[:1500]
save()

go = REPORT.get("zero_equals_base") and REPORT.get("noise_differs_from_base") and REPORT.get("adapter_routed")
REPORT["verdict"] = "GO" if go else "CHECK"
save()
log("VERDICT:", REPORT["verdict"], {k: REPORT.get(k) for k in
    ("zero_equals_base", "noise_differs_from_base", "adapter_routed", "compiled_agent_model", "server_start_s")})
server.stop()
