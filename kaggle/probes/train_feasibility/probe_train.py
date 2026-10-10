"""
Phase 0b go/no-go, v3: can we LoRA-train Gemma-4-31B on Kaggle 4x L4, and how fast?  (docs/training_feasibility_4xl4.md)

v2 (Oct 9, experiments.md p0b) settled: head_dim-512 SDPA fwd+bwd OK to 32k; LoRA export = the 410 served modules;
layer-split over 4 GPUs = 82 tok/s at 8k, OOM at 16k. Still open: FSDP throughput and NF4-vs-W4A16 agreement.
v3 fixes: every GPU stage is its own process (the parent never touches CUDA, so nothing is left on the GPUs);
per-rank logs are kept; loss is computed inside the FSDP forward (the tied lm_head is sharded outside it);
activation offload to host RAM (188 GB) for 24k/32k; bf16 adapter export; vLLM script has a __main__ guard.

Kaggle: GPU L4 x4, offline, via the competition notebook `check-l4` (only it gets L4).
Inputs (attached by make_notebook.py):
  - kernel output gemma4-scorer-env   (scorer_env.tgz: Py3.12 venv, torch 2.10 cu128, transformers 5.13.1, vLLM 0.19.1)
  - kernel output gemma4-train-wheels (peft 0.21.2 + bitsandbytes 0.50.2)
  - model google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2  (bf16 QAT weights -> NF4 on load)
  - model google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2                (served base, for the mismatch check)

Stages (each skipped if the time budget PROBE_BUDGET_MIN, default 75, can't cover it):
  A. FSDP, 4 ranks, data parallel: NF4 load (+ optional NF4 save on rank 0), LoRA r16 on the 410 linears,
     NF4 next-token predictions on real code (zero adapter), then train steps at 8k/16k (plain) and
     24k/32k (activations offloaded to CPU): tokens/s and peak memory per rank; bf16 adapter export.
  B. NF4 vs W4A16 (vLLM 0.19.1, prompt_logprobs): top-1 agreement on the same tokens.
Output: /kaggle/working/train_probe.json (+ run.log, fsdp_logs/, adapter_probe/). Verdict at the end.

PROBE_SMOKE=1 runs stage A on CPU (gloo, 2 ranks) with a tiny random Gemma-4 built from the real config.
"""
import glob
import json
import os
import pathlib
import subprocess
import sys
import time

SMOKE = os.environ.get("PROBE_SMOKE") == "1"
W = pathlib.Path(os.environ.get("PROBE_OUT", "/tmp/probe_smoke" if SMOKE else "/kaggle/working"))
W.mkdir(parents=True, exist_ok=True)
LOG = open(W / "run.log", "a", buffering=1)
REPORT = {"probe_version": 3}
T0 = float(os.environ.get("PROBE_T0", time.time()))  # survives the re-exec into the venv
os.environ["PROBE_T0"] = str(T0)
BUDGET_MIN = float(os.environ.get("PROBE_BUDGET_MIN", "95"))


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")


def sh(cmd, timeout=None, check=False):
    log("$", cmd[:240])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = ((r.stdout or "")[-3000:] + ("\n" + r.stderr[-3000:] if (r.stderr or "").strip() else "")).strip()
    if out:
        log(out)
    if check and r.returncode != 0:
        save()
        raise SystemExit(f"FAILED rc={r.returncode}: {cmd[:120]}")
    return r


def save():
    REPORT["elapsed_min"] = round((time.time() - T0) / 60, 1)
    (W / "train_probe.json").write_text(json.dumps(REPORT, indent=1, default=str))


def left_min():
    return BUDGET_MIN - (time.time() - T0) / 60


def model_dir(*must):
    for cfg in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        if all(m in cfg.lower() for m in must):
            return str(pathlib.Path(cfg).parent)
    return None


# ------------------------------------------------------------------ 0. inputs + offline env (no CUDA in this process)
VENV = pathlib.Path("/tmp/scorer_host")
ON_KAGGLE = pathlib.Path("/kaggle/input").exists()
if not SMOKE:
    gpus = [g for g in sh("nvidia-smi --query-gpu=name --format=csv,noheader").stdout.split("\n") if g.strip()]
    REPORT["gpus"] = gpus
    if len(gpus) < 4 or not all("L4" in g for g in gpus):  # don't burn quota on the wrong machine
        save()
        raise SystemExit(f"need 4x L4, got {gpus} -> set Accelerator to GPU L4 x4")
if ON_KAGGLE:
    if pathlib.Path(sys.prefix) != VENV:
        env_tgz = sorted(glob.glob("/kaggle/input/**/scorer_env.tgz", recursive=True))
        assert env_tgz, "attach the output of kernel gemma4-scorer-env (kernel_sources)"
        whls = sorted(glob.glob("/kaggle/input/**/train_wheels/*.whl", recursive=True))
        assert whls, "attach the output of kernel gemma4-train-wheels (kernel_sources)"
        t = time.time()
        sh(f"tar -xzf {env_tgz[0]} -C /", timeout=1800, check=True)
        r = sh(f"{sys.executable} -m pip --python {VENV}/bin/python install --no-deps --no-index -q " + " ".join(whls),
               timeout=600)
        if r.returncode != 0:  # fallback: wheels are zip archives, unpack into site-packages
            site = sh(f"{VENV}/bin/python -c 'import site; print(site.getsitepackages()[0])'").stdout.strip()
            for w in whls:
                sh(f"cd {site} && unzip -oq {w}", check=True)
        REPORT["env_restore_min"] = round((time.time() - t) / 60, 1)
        save()
        LOG.close()
        os.execv(f"{VENV}/bin/python", [f"{VENV}/bin/python", os.path.abspath(__file__)])
    mem = dict(l.split(":", 1) for l in open("/proc/meminfo").read().splitlines() if ":" in l)
    REPORT["host_ram_gb"] = round(int(mem["MemTotal"].split()[0]) / 2 ** 20, 1)
    REPORT["cpus"] = os.cpu_count()
r = sh(f"{sys.executable} -c 'import torch, transformers, peft; "
       "print(torch.__version__, transformers.__version__, peft.__version__)'")
REPORT["versions"] = (r.stdout or r.stderr).strip()[-200:]
save()

if SMOKE:
    MODEL_PATH = str(W / "tiny_model")
else:
    MODEL_PATH = model_dir("31b", "unquantized") or model_dir("31b", "qat-q4_0")
    assert MODEL_PATH, "attach model google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"
log("MODEL =", MODEL_PATH)

# ------------------------------------------------------------------ A. FSDP worker (one process per GPU)
FSDP_WORKER = r'''
import datetime, glob, json, os, sys, time
import torch, torch.distributed as dist, torch.nn as nn, torch.nn.functional as F
from functools import partial
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import transformers
from transformers import AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model

SMOKE = os.environ.get("PROBE_SMOKE") == "1"
MODEL, OUT, SEQS, OFFLOAD_FROM = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")], int(sys.argv[4])
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_RE = r".*language_model.*\.(" + "|".join(TARGETS) + r")$"
dist.init_process_group("gloo" if SMOKE else "nccl", timeout=datetime.timedelta(minutes=8))  # a rank that OOMs must not hang the rest
rank, world = dist.get_rank(), dist.get_world_size()
if SMOKE:
    dev, dt = torch.device("cpu"), torch.float32
else:
    torch.cuda.set_device(rank)
    dev, dt = torch.device("cuda", rank), torch.bfloat16
    # flash caps head_dim at 256 and math materialises T x T: force the memory-efficient kernel
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(True)
res = {"world": world, "seqs": {}}


def dump():
    if rank == 0:
        open(OUT, "w").write(json.dumps(res, indent=1, default=str))


def peak():
    return 0.0 if SMOKE else round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)


def max_over_ranks(x):
    t = torch.tensor([float(x)], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return round(float(t), 2)


# 1. load: every rank quantises the full model onto its own GPU, FSDP then shards it
kw = {"dtype": dt, "attn_implementation": "sdpa"}
if not SMOKE:
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dt,
        bnb_4bit_quant_storage=dt,  # FSDP can only shard 4-bit weights stored in the compute dtype
        llm_int8_skip_modules=["vision_tower", "embed_vision", "multi_modal_projector", "audio_tower", "embed_audio", "lm_head"])
    kw["device_map"] = {"": rank}
t = time.time()
m = AutoModelForImageTextToText.from_pretrained(MODEL, **kw)
res["load_s"] = round(time.time() - t)
res["gpu_gb_after_load"] = max_over_ranks(peak())
cfg = m.config.get_text_config()
CAP = getattr(cfg, "final_logit_softcapping", None)
dump()

# 1b. optional: keep the NF4 checkpoint so later sessions skip the ~12 min quantisation
nf4_dir = os.environ.get("PROBE_NF4_DIR")
if nf4_dir and rank == 0 and not SMOKE:
    t = time.time()
    m.save_pretrained(nf4_dir)
    sz = sum(os.path.getsize(p) for p in glob.glob(f"{nf4_dir}/*.safetensors"))
    res["nf4_save"] = {"dir": nf4_dir, "gb": round(sz / 1e9, 2), "save_s": round(time.time() - t)}
dist.barrier()


CH = 64 if SMOKE else 256  # loss chunk: 256 x 262k fp32 logits = 0.27 GB


class ChunkedCE(torch.autograd.Function):
    """Next-token CE with Gemma's final-logit softcap, 1k tokens at a time: T x vocab logits never exist at once.
    The head is frozen, so only dL/dh is needed; it is computed chunk by chunk in forward and kept (T x hidden)."""

    @staticmethod
    def forward(ctx, h, w, y):
        n = y.numel()
        gh = torch.empty_like(h)
        tot = torch.zeros((), device=h.device, dtype=torch.float32)
        for i in range(0, n, CH):
            hc = h[i:i + CH].detach().requires_grad_(True)
            with torch.enable_grad():
                z = (hc @ w.T).float()
                if CAP:
                    z = torch.tanh(z / CAP) * CAP
                l = F.cross_entropy(z, y[i:i + CH], reduction="sum")
                g, = torch.autograd.grad(l, hc)
            gh[i:i + CH] = g
            tot += l.detach()
        ctx.save_for_backward(gh)
        ctx.n = n
        return tot / n

    @staticmethod
    def backward(ctx, go):
        gh, = ctx.saved_tensors
        return gh * (go / ctx.n), None, None


class Wrap(nn.Module):
    """FSDP root: the tied lm_head/embedding are unsharded only inside the root forward, so the loss lives here."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids, mode="loss"):
        base = self.m.base_model.model  # PeftModel -> LoraModel -> Gemma4ForConditionalGeneration
        h = base.model.language_model(input_ids=ids, use_cache=False).last_hidden_state[0]  # no logits, no hidden-state tuple
        w = base.get_output_embeddings().weight
        if mode == "logits":  # NF4 reference predictions (no grad)
            z = (h @ w.T).float()
            if CAP:
                z = torch.tanh(z / CAP) * CAP
            lp = torch.log_softmax(z, -1)
            return lp.argmax(-1), lp[torch.arange(ids.shape[1] - 1, device=lp.device), ids[0, 1:]]
        return ChunkedCE.apply(h[:-1], w, ids[0, 1:])


# 2. LoRA r16 on the 410 language-model linears; adapters in the base dtype (FSDP flat params need one dtype)
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
m.enable_input_require_grads()
m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules=LORA_RE,
                                 task_type="CAUSAL_LM"))
for n_, p in m.named_parameters():
    if "lora_" in n_:
        p.data = p.data.to(dt)
lm = next(mm for n_, mm in m.named_modules() if n_.endswith("language_model") and "vision" not in n_)
res["lora_modules"] = sum(1 for n_, _ in m.named_parameters() if ".lora_A." in n_)
layer_cls = type(lm.layers[0])
vision = [mm for n_, mm in m.named_modules() if n_.endswith("vision_tower") or n_.endswith("embed_vision")]
wrap_types = tuple({layer_cls} | {type(v) for v in vision})  # vision gets its own (never-gathered) unit
wrap = partial(lambda_auto_wrap_policy, lambda_fn=lambda mod: isinstance(mod, wrap_types))
fm = FSDP(Wrap(m), auto_wrap_policy=wrap, sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True,
          device_id=torch.device("cpu") if SMOKE else torch.cuda.current_device(), limit_all_gathers=True, sync_module_states=False)
if not SMOKE:
    torch.cuda.empty_cache()
res["gpu_gb_after_wrap"] = max_over_ranks(0 if SMOKE else torch.cuda.memory_allocated() / 2 ** 30)
dump()

# 3. NF4 reference predictions on real code (zero adapter: LoRA B = 0), same number of forwards on every rank
if SMOKE:
    texts = [torch.randint(0, cfg.vocab_size, (64,)).tolist() for _ in range(2 * world)]
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    files = sorted(glob.glob(os.path.join(os.path.dirname(transformers.__file__), "models", "*", "modeling_*.py")))
    texts = []
    for f in files[::7]:
        ids = tok(open(f).read())["input_ids"]
        if len(ids) >= 1024:
            texts.append(ids[:1024])
        if len(texts) == 2 * world:
            break
mine = texts[rank::world]
fm.eval()
out = []
with torch.no_grad():
    for ids in mine:
        am, tlp = fm(torch.tensor([ids], device=dev), mode="logits")
        out.append({"ids": ids, "nf4_argmax": am[:-1].tolist(), "nf4_true_lp": tlp.tolist()})
allv = [None] * world
dist.all_gather_object(allv, out)
if rank == 0:
    ref = [x for part in allv for x in part]
    torch.save({k: [r[k] for r in ref] for k in ("ids", "nf4_argmax", "nf4_true_lp")}, os.path.join(os.path.dirname(OUT), "nf4_ref.pt"))
res["nf4_ref_texts"] = len(texts)
dump()

# 4. train steps: data parallel, one sequence of T tokens per rank
fm.train()
opt = torch.optim.AdamW([p for p in fm.parameters() if p.requires_grad], lr=1e-5)
for T in SEQS:
    offload = T >= OFFLOAD_FROM
    try:
        if not SMOKE:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        times = []
        for step in range(2):  # step 0 = warm-up
            ids = torch.randint(0, cfg.vocab_size, (1, T), device=dev)
            if not SMOKE:
                torch.cuda.synchronize()
            t = time.time()
            if offload:  # saved activations (checkpoint inputs) live in pinned host RAM
                with torch.autograd.graph.save_on_cpu(pin_memory=not SMOKE):
                    loss = fm(ids)
            else:
                loss = fm(ids)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if not SMOKE:
                torch.cuda.synchronize()
            times.append(time.time() - t)
        res["seqs"][T] = {"ok": True, "offload": offload, "step_s": round(times[-1], 1),
                          "tok_per_s_total": round(world * T / times[-1], 1), "peak_gb_max_rank": max_over_ranks(peak()),
                          "loss": round(float(loss.detach()), 3)}
    except Exception as e:  # noqa: BLE001
        res["seqs"][T] = {"ok": False, "offload": offload, "error": f"{type(e).__name__}: {e}"[:400],
                          "peak_gb_this_rank": peak()}
        opt.zero_grad(set_to_none=True)
        loss = None
        if not SMOKE:
            torch.cuda.empty_cache()
    print(f"rank{rank} T={T}", res["seqs"][T], flush=True)
    dump()
    ok = torch.tensor([1.0 if res["seqs"][T].get("ok") else 0.0], device=dev)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if float(ok) == 0 and not offload:  # plain OOM at T: skip straight to the offloaded lengths
        SEQS = [s for s in SEQS if s >= OFFLOAD_FROM or s <= T]

# 5. bf16 adapter export. All LoRA weights live in the decoder layers, and each layer is one FSDP unit, so gather
#    one layer at a time (~0.3 GB) instead of the whole model; every rank takes part, rank 0 keeps the LoRA tensors.
from safetensors.torch import save_file
unit_name = {id(mod): n_ for n_, mod in fm.named_modules()}
t = {}
for unit in FSDP.fsdp_modules(fm):
    if not isinstance(unit.module, layer_cls):
        continue
    with FSDP.summon_full_params(unit, recurse=False, writeback=False):
        if rank == 0:
            base = unit_name[id(unit)]
            for n_, p in unit.module.named_parameters():
                if "lora_" in n_:
                    full = f"{base}.{n_}".replace("_fsdp_wrapped_module.", "").removeprefix("m.").replace(".default.", ".")
                    t[full] = p.detach().to("cpu", torch.bfloat16).clone()
if rank == 0:
    adir = os.path.join(os.path.dirname(OUT), "adapter_probe")
    os.makedirs(adir, exist_ok=True)
    save_file(t, os.path.join(adir, "adapter_model.safetensors"))
    m.peft_config["default"].save_pretrained(adir)
    mods = sorted({k.removeprefix("base_model.model.").rsplit(".lora_", 1)[0] for k in t})
    shapes_ok = all(v.ndim == 2 and min(v.shape) == 16 for v in t.values())
    res["export"] = {"n_modules": len(mods), "shapes_ok": shapes_ok, "key_example": next(iter(t)),
                     "mb": round(os.path.getsize(os.path.join(adir, "adapter_model.safetensors")) / 1e6, 1)}
dump()
dist.barrier()
dist.destroy_process_group()
'''

if SMOKE:  # tiny random Gemma-4 from the real config, saved like a checkpoint
    r = sh(f"""{sys.executable} - <<'EOF'
import json, torch
from transformers import AutoConfig, AutoModelForImageTextToText
real = json.load(open({os.environ['PROBE_SMOKE_CONFIG']!r}))
tc, n = real["text_config"], 6
tc.update(hidden_size=64, intermediate_size=128, num_hidden_layers=n, num_attention_heads=4, num_key_value_heads=2,
          num_global_key_value_heads=1, head_dim=16, global_head_dim=32, vocab_size=512, vocab_size_per_layer_input=512,
          sliding_window=16, layer_types=["sliding_attention"] * (n - 1) + ["full_attention"])
real["vision_config"].update(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=2, head_dim=16, global_head_dim=16)
torch.manual_seed(0)
m = AutoModelForImageTextToText.from_config(AutoConfig.for_model(**{{k: v for k, v in real.items() if k != "transformers_version"}}))
m.save_pretrained({MODEL_PATH!r})
EOF""", check=True)

SEQS, OFFLOAD_FROM = ("64,128,256", 128) if SMOKE else ("4096,8192,16384,24576,32768", 8192)
NPROC = 2 if SMOKE else 4
if left_min() < 30:
    REPORT["fsdp"] = {"skipped": f"time budget ({left_min():.0f} min left)"}
else:
    wf, out = W / "fsdp_worker.py", W / "fsdp_result.json"
    wf.write_text(FSDP_WORKER)
    logs = W / "fsdp_logs"
    env = "PROBE_SMOKE=1 " if SMOKE else "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    if os.environ.get("PROBE_SAVE_NF4") == "1":
        env += f"PROBE_NF4_DIR={W / 'nf4_ckpt'} "
    try:
        r = sh(f"{env}{sys.executable} -m torch.distributed.run --nproc_per_node {NPROC} --log-dir {logs} "
               f"--redirects 3 --tee 3 {wf} {MODEL_PATH} {out} {SEQS} {OFFLOAD_FROM}",
               timeout=int(max(5, min(55, left_min() - 16)) * 60))
        REPORT["fsdp"] = json.loads(out.read_text()) if out.exists() else {"error": "no result"}
        REPORT["fsdp"]["rc"] = r.returncode
    except subprocess.TimeoutExpired:
        REPORT["fsdp"] = json.loads(out.read_text()) if out.exists() else {}
        REPORT["fsdp"]["error"] = "timeout"
        sh("pkill -f fsdp_worker.py")
    if REPORT["fsdp"].get("rc") or REPORT["fsdp"].get("error"):  # keep each rank's own traceback
        tails = {}
        for f in sorted(logs.glob("**/stderr.log")):
            txt = f.read_text()[-1200:]
            if txt.strip():
                tails[str(f.relative_to(logs))] = txt
        REPORT["fsdp"]["rank_stderr_tails"] = tails
log("fsdp", json.dumps(REPORT["fsdp"], default=str)[:2000])
save()

# ------------------------------------------------------------------ B. which training base matches the served W4A16?
# B1: HF references on the same token ids: bf16 QAT weights (unquantised) and HF-dequantised W4A16 (compressed-tensors,
#     decompressed to bf16). B2: vLLM (the scorer's engine) on W4A16, compared against every reference.
#     HF-W4A16 vs vLLM = implementation noise floor; NF4 / bf16 vs vLLM = what a training base would cost.
HF_REF = r"""
import json, sys, torch
from transformers import AutoModelForImageTextToText
path, kind, ref_in, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
ref = torch.load(ref_in)
kw = dict(dtype=torch.bfloat16, device_map="balanced", attn_implementation="sdpa")
if kind == "w4a16_dq":
    from transformers import CompressedTensorsConfig
    kw["quantization_config"] = CompressedTensorsConfig(run_compressed=False)
m = AutoModelForImageTextToText.from_pretrained(path, **kw).eval()
cap = m.config.get_text_config().final_logit_softcapping
head = m.get_output_embeddings()
am, tl = [], []
with torch.no_grad():
    for ids in ref["ids"]:
        x = torch.tensor([ids], device=m.device)
        h = m.model.language_model(input_ids=x, use_cache=False).last_hidden_state[0].to(head.weight.device)
        z = (h @ head.weight.T).float()
        z = torch.tanh(z / cap) * cap if cap else z
        lp = torch.log_softmax(z, -1)
        am.append(lp.argmax(-1)[:-1].tolist())
        tl.append(lp[torch.arange(len(ids) - 1, device=lp.device), torch.tensor(ids[1:], device=lp.device)].tolist())
torch.save({"ids": ref["ids"], "argmax": am, "true_lp": tl}, out)
print(json.dumps({"ok": True, "kind": kind}))
"""
MISMATCH = r"""
import glob, json, os, sys


def main():
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    nf4 = torch.load(sys.argv[2])
    refs = {"nf4": (nf4["nf4_argmax"], nf4["nf4_true_lp"])}
    for f in glob.glob(os.path.join(os.path.dirname(sys.argv[2]), "hfref_*.pt")):
        r = torch.load(f)
        refs[os.path.basename(f)[6:-3]] = (r["argmax"], r["true_lp"])
    llm = LLM(model=sys.argv[1], tensor_parallel_size=4, max_model_len=2048, gpu_memory_utilization=0.85,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in nf4["ids"]],
                        SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0))
    res = {}
    for name, (AM, TL) in refs.items():
        agree = n = 0
        dlp = []
        for o, ids, am, tlp in zip(outs, nf4["ids"], AM, TL):
            pl = o.prompt_logprobs  # position j: {token: Logprob} for token j given tokens < j
            for j in range(1, len(ids)):
                d = pl[j]
                top = max(d.items(), key=lambda kv: kv[1].logprob)[0]
                agree += int(top == am[j - 1])
                n += 1
                dlp.append(abs(d[ids[j]].logprob - tlp[j - 1]))
        dlp.sort()
        res[name] = {"positions": n, "top1_agreement_vs_vllm": round(agree / n, 4),
                     "median_abs_dlogp_true": round(dlp[len(dlp) // 2], 4), "p99_abs_dlogp_true": round(dlp[int(0.99 * len(dlp))], 4)}
    print(json.dumps(res), flush=True)


if __name__ == "__main__":  # vLLM spawns workers that re-import this file
    main()
"""
W4 = None if SMOKE else model_dir("31b", "w4a16")
REPORT["mismatch"] = {}
if SMOKE or not W4 or not (W / "nf4_ref.pt").exists():
    REPORT["mismatch"] = {"skipped": "smoke" if SMOKE else ("no W4A16 model" if not W4 else "no nf4_ref.pt from stage A")}
else:
    hf = W / "hf_ref.py"
    hf.write_text(HF_REF)
    for kind, path in (("bf16_qat", MODEL_PATH), ("w4a16_dq", W4)):
        if left_min() < 20:
            REPORT["mismatch"][f"hf_{kind}"] = {"skipped": "time budget"}
            continue
        try:
            r = sh(f"{sys.executable} {hf} {path} {kind} {W / 'nf4_ref.pt'} {W / f'hfref_{kind}.pt'}",
                   timeout=int(min(25, left_min() - 15) * 60))
            REPORT["mismatch"][f"hf_{kind}"] = {"rc": r.returncode, "tail": ((r.stdout or "") + (r.stderr or ""))[-600:]}
        except subprocess.TimeoutExpired:
            REPORT["mismatch"][f"hf_{kind}"] = {"error": "timeout"}
        save()
    if left_min() >= 12:
        mf = W / "mismatch.py"
        mf.write_text(MISMATCH)
        try:
            r = sh(f"VLLM_WORKER_MULTIPROC_METHOD=spawn {sys.executable} {mf} {W4} {W / 'nf4_ref.pt'}",
                   timeout=int((left_min() - 1) * 60))
            line = [l for l in (r.stdout or "").splitlines() if l.startswith("{")]
            REPORT["mismatch"]["vs_vllm"] = json.loads(line[-1]) if line else {"error": (r.stdout or "")[-1500:] + (r.stderr or "")[-1500:]}
        except subprocess.TimeoutExpired:
            REPORT["mismatch"]["vs_vllm"] = {"error": "timeout"}
log("mismatch", REPORT["mismatch"])
save()

# ------------------------------------------------------------------ verdict
seqs = {int(k): v for k, v in REPORT.get("fsdp", {}).get("seqs", {}).items()}
ok = {T: v for T, v in seqs.items() if v.get("ok")}
REPORT["verdict"] = {
    "fsdp_max_seq_plain": max([T for T, v in ok.items() if not v.get("offload")], default=0),
    "fsdp_max_seq_offload": max([T for T, v in ok.items() if v.get("offload")], default=0),
    "fsdp_tok_per_s": {T: v["tok_per_s_total"] for T, v in ok.items()},
    "export_modules": REPORT.get("fsdp", {}).get("export", {}).get("n_modules"),
    "agreement_vs_vllm_w4a16": {k: v.get("top1_agreement_vs_vllm") for k, v in
                                (REPORT.get("mismatch", {}).get("vs_vllm") or {}).items() if isinstance(v, dict)},
}
save()
log("VERDICT", REPORT["verdict"])
