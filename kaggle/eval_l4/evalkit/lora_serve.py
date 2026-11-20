"""
B0 LoRA serving probe on the ATTACHED model (E4B for B0; Phase-0 method of Oct 7, which was run on the 31B only).
  1. target linear modules (q/k/v/o/gate/up/down of the language model, no vision/audio) + true (out, in) shapes
  2. two synthetic rank-8 PEFT adapters: zero_lora (B = 0 -> must equal base) and noise_lora (B != 0 -> must differ)
  3. ScorerServer with both adapters (the scorer adds --enable-lora only when a submission ships adapters, so the
     plain eval that follows starts its own server without them, as the scorer would for v26)
  4. base / zero / noise at temperature 0: plain text + one tool call, thinking off
  5. decode speed at concurrency 1, base vs zero_lora: prompts of ~2k and ~12k tokens (fresh nonce each call),
     256 decode tokens, 3 repeats, arms alternating; the base numbers also give the model's tok/s for runtime estimates
  6. official compiler: an agent declaring the submission's model + `adapter: noise_lora` routes to the adapter
--dry (CPU): steps 1-2 and adapter discovery only.
Never raises: a server or compile failure is a result (NO-GO), and the eval step after it still runs.
Output: /kaggle/working/lora_serve_report.json. Run inside the venv: python -m evalkit.lora_serve [--dry]
"""
import json
import pathlib
import random
import statistics
import sys
import time
import traceback
import urllib.request

from .config import WORK, Settings
from .inputs import Inputs
from .server import ScorerServer, set_scorer_env
from .submission import Submission
from .util import log

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SKIP = ("vision", "audio", "multi_modal", "embed_", "mm_")
RANK = 8
REPORT_PATH = WORK / "lora_serve_report.json"
PROBE_SUB = WORK / "lora_serve_sub"
PARA = ("def merge(a, b):\n    # Merge two sorted lists into one sorted list without using sort().\n"
        "    out, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n"
        "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n"
        "            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]\n\n")
TIMING = {"short": 22, "long": 130}  # paragraph repeats: about 2k and 12k prompt tokens
DECODE, REPEATS = 256, 3


def target_shapes(model_path):
    """{module name: (out, in)} for every target linear in the checkpoint (compressed-tensors keeps weight_shape)."""
    from safetensors import safe_open
    root = pathlib.Path(model_path)
    idx = root / "model.safetensors.index.json"
    if idx.exists():
        weight_map = json.loads(idx.read_text())["weight_map"]
    else:
        with safe_open(str(root / "model.safetensors"), "pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}
    modules = {}
    for key, fname in weight_map.items():
        mod, leaf = key.rsplit(".", 1)
        if not leaf.startswith("weight") or mod.rsplit(".", 1)[-1] not in TARGETS or any(s in mod for s in SKIP):
            continue
        modules.setdefault(mod, {})[leaf] = fname
    shapes, handles = {}, {}
    for mod, leaves in sorted(modules.items()):
        if "weight_shape" in leaves:
            f = handles.setdefault(leaves["weight_shape"], safe_open(str(root / leaves["weight_shape"]), "pt"))
            shapes[mod] = tuple(int(x) for x in f.get_tensor(f"{mod}.weight_shape").tolist())
        elif "weight" in leaves:
            f = handles.setdefault(leaves["weight"], safe_open(str(root / leaves["weight"]), "pt"))
            shapes[mod] = tuple(f.get_slice(f"{mod}.weight").get_shape())
    return shapes


def write_adapters(shapes, model_path, declared):
    """zero_lora + noise_lora under PROBE_SUB/adapters/, plus an agent that routes to noise_lora (step 6)."""
    import torch
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(0)
    sizes = {}
    for name, scale in (("zero_lora", 0.0), ("noise_lora", 0.05)):
        d = PROBE_SUB / "adapters" / name
        d.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for mod, (out_f, in_f) in shapes.items():
            tensors[f"base_model.model.{mod}.lora_A.weight"] = (torch.randn(RANK, in_f, generator=g) / in_f ** 0.5).to(torch.bfloat16)
            tensors[f"base_model.model.{mod}.lora_B.weight"] = (torch.randn(out_f, RANK, generator=g) * scale).to(torch.bfloat16)
        save_file(tensors, str(d / "adapter_model.safetensors"))
        (d / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": RANK, "lora_alpha": 2 * RANK, "lora_dropout": 0.0,
            "bias": "none", "fan_in_fan_out": False, "target_modules": list(TARGETS),
            "base_model_name_or_path": model_path, "inference_mode": True}, indent=1))
        sizes[name] = round((d / "adapter_model.safetensors").stat().st_size / 1e6, 1)
    (PROBE_SUB / "agent.yaml").write_text(
        f"name: lora_serve_agent\nmodel: {declared}\nadapter: noise_lora\n"
        "instruction: You are a software engineer. Fix the issue and call submit_patch.\n"
        "tools:\n  - run_command\n  - submit_patch\n"
        "generate_content_config:\n  temperature: 0.2\n  max_output_tokens: 1024\n")
    (PROBE_SUB / "eval_config.yaml").write_text("evaluation:\n  max_tool_calls: 5\n  max_time_minutes: 1\n")
    return sizes


def api(base, path, body=None, timeout=900):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read()), time.perf_counter() - t


def behaviour(base, arms):
    """Step 4: same request to every arm at temperature 0."""
    prompt = [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number, "
                                          "then explain it in two sentences."}]
    tool_msg = [{"role": "user", "content": "List the Python files under /workspace/src using the run_command tool."}]
    tools = [{"type": "function", "function": {"name": "run_command", "description": "Run a shell command",
              "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
    off = {"chat_template_kwargs": {"enable_thinking": False}}
    outs = {}
    for arm, mid in arms.items():
        rec = {}
        try:
            r, _ = api(base, "/chat/completions", {"model": mid, "messages": prompt, "temperature": 0, "max_tokens": 256, **off})
            rec["text"] = (r["choices"][0]["message"].get("content") or "")[:600]
            r, _ = api(base, "/chat/completions", {"model": mid, "messages": tool_msg, "tools": tools, "temperature": 0,
                                                   "max_tokens": 256, **off})
            msg = r["choices"][0]["message"]
            rec["tool_calls"] = [(c["function"]["name"], c["function"]["arguments"]) for c in (msg.get("tool_calls") or [])]
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:800]
        outs[arm] = rec
        log("ARM", arm, json.dumps(rec)[:400])
    return outs


def timing(base, arms):
    """Step 5: prefill + decode at concurrency 1, base vs zero_lora (zero_lora runs the LoRA kernels like a real one)."""
    rng, rows = random.Random(0), []
    for mid in arms.values():  # warm-up: CUDA graphs, first LoRA load
        api(base, "/completions", {"model": mid, "prompt": PARA * 4, "max_tokens": 8})
    for label, n in TIMING.items():
        for rep in range(REPEATS):
            for arm in rng.sample(list(arms), len(arms)):
                prompt = f"# run {rng.randrange(10**9)}\n" + PARA * n  # nonce first: no prefix-cache hit across calls
                _, t_pre = api(base, "/completions", {"model": arms[arm], "prompt": prompt, "max_tokens": 1, "temperature": 0})
                r, t_dec = api(base, "/completions", {"model": arms[arm], "prompt": prompt, "max_tokens": DECODE,
                                                      "min_tokens": DECODE, "ignore_eos": True, "temperature": 0})
                u = r["usage"]
                row = {"arm": arm, "len": label, "prompt_tokens": u["prompt_tokens"], "rep": rep,
                       "prefill_s": round(t_pre, 3), "decode_s": round(t_dec, 3),
                       "decode_tok_s": round(u["completion_tokens"] / t_dec, 2)}
                rows.append(row)
                log("CALL", json.dumps(row))
    summ = {}
    for label in TIMING:
        for arm in arms:
            rs = [r for r in rows if r["len"] == label and r["arm"] == arm]
            summ[f"{arm}@{label}"] = {"prompt_tokens": rs[0]["prompt_tokens"],
                                      "prefill_s": round(statistics.median(r["prefill_s"] for r in rs), 3),
                                      "decode_tok_s": round(statistics.median(r["decode_tok_s"] for r in rs), 2)}
        summ[f"lora_decode_slowdown@{label}"] = round(
            summ[f"base@{label}"]["decode_tok_s"] / summ[f"zero_lora@{label}"]["decode_tok_s"], 3)
    return rows, summ


def routes(server, declared, adapters):
    """Step 6: compile the probe agent the way agent_runner does on the scorer; which model does it call?"""
    from adk_submission import compile_submission
    from swegemma.config import build_submission_limits
    from swegemma.harness.agent_runner import resolve_swegemma_adapter
    models = server.registry(declared)

    def mk(nm):
        def f(**kw):
            return "{}"
        f.__name__ = nm
        f.__doc__ = nm
        return f

    tools = {nm: mk(nm) for nm in ("run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch")}
    limits, gen = build_submission_limits()
    agent = compile_submission(PROBE_SUB, tools, models, limits=limits, generation_constraints=gen,
                               adapter_manifest=adapters,
                               adapter_resolver_fn=lambda b, info: resolve_swegemma_adapter(b, info, models))
    m = getattr(agent, "model", None)
    return getattr(m, "model", None) or str(m)


def main(dry=False):
    s = Settings.load()
    set_scorer_env()
    rep = {"dry": dry}

    def save():
        rep["elapsed_min"] = round((time.time() - s.t0) / 60, 1)
        REPORT_PATH.write_text(json.dumps(rep, indent=1, default=str))

    server = None
    try:
        model_path = Inputs.locate(False).model_path
        declared = Submission().unpack().declared_model()
        rep.update(model_path=model_path, declared_model=declared)
        cfg = json.loads((pathlib.Path(model_path) / "config.json").read_text())
        rep["architectures"] = cfg.get("architectures")
        shapes = target_shapes(model_path)
        rep["n_target_modules"] = len(shapes)
        rep["per_target"] = {t: sum(m.endswith("." + t) for m in shapes) for t in TARGETS}
        rep["distinct_shapes"] = sorted({(m.rsplit(".", 1)[-1],) + s_ for m, s_ in shapes.items()})
        rep["module_examples"] = list(shapes)[:4]
        log("LORA_SERVE modules", rep["n_target_modules"], rep["per_target"], rep["distinct_shapes"])
        assert shapes, "no target modules found in the checkpoint"
        rep["adapter_mb"] = write_adapters(shapes, model_path, declared)
        from adk_submission import discover_adapters
        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
        adapters = discover_adapters(str(PROBE_SUB), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
        rep["discovered"] = sorted(getattr(adapters, "adapters", {}) or {})
        save()
        if dry:
            rep["verdict"] = "DRY_OK"
            return
        t = time.time()
        try:
            server = ScorerServer(model_path, adapters).start()
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            rep["server_error"] = msg[:300] + "\n...\n" + msg[-6000:]  # the cause is at the end of vLLM's output
            rep["verdict"] = "NO-GO: vLLM did not start with LoRA on this model"
            log("SERVER ERROR TAIL:\n" + msg[-6000:])
            return
        rep["server_start_s"] = round(time.time() - t)
        base = server.server.base_url.rstrip("/")
        served = [m["id"] for m in api(base, "/models")[0]["data"]]
        rep["served_models"] = served
        arms = {"base": next(m for m in served if m not in ("zero_lora", "noise_lora")),
                "zero_lora": "zero_lora", "noise_lora": "noise_lora"}
        outs = behaviour(base, arms)
        rep["outputs"] = outs
        rep["zero_equals_base"] = "error" not in outs["zero_lora"] and outs["zero_lora"].get("text") == outs["base"].get("text")
        rep["noise_differs_from_base"] = "error" not in outs["noise_lora"] and outs["noise_lora"].get("text") != outs["base"].get("text")
        rep["base_tool_call_ok"] = bool(outs["base"].get("tool_calls"))
        save()
        try:
            rep["timing_rows"], rep["timing"] = timing(base, {"base": arms["base"], "zero_lora": "zero_lora"})
        except Exception as e:  # noqa: BLE001
            rep["timing_error"] = f"{type(e).__name__}: {e}"[:800]
        save()
        try:
            rep["compiled_agent_model"] = routes(server, declared, adapters)
            rep["adapter_routed"] = "noise_lora" in str(rep["compiled_agent_model"])
        except Exception as e:  # noqa: BLE001
            rep["compile_error"] = f"{type(e).__name__}: {e}"[:1500]
        go = rep.get("zero_equals_base") and rep.get("noise_differs_from_base") and rep.get("adapter_routed")
        rep["verdict"] = "GO" if go else "CHECK"
    except Exception as e:  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
        rep["traceback"] = traceback.format_exc()[-3000:]
        rep.setdefault("verdict", "ERROR")
    finally:
        save()
        log("LORA_SERVE VERDICT:", rep.get("verdict"), json.dumps({k: rep.get(k) for k in (
            "n_target_modules", "server_start_s", "zero_equals_base", "noise_differs_from_base", "adapter_routed",
            "timing", "error")}, default=str)[:1500])
        if server:
            server.stop()
            time.sleep(20)  # let the GPUs free before the eval step starts its own server


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
