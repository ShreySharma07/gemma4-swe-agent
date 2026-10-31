"""
LC1 probe: are the long-context fixes exact, and does exact full backprop fit at 24-32k on 4x L4?
Smoke (CPU, fp32, tiny Gemma-4, window 16, MLP tile 32): the whole new stack (block attention, tiled MLP,
checkpointed norms, host head + StreamCE, replicated LoRA + ShardedAdamW, global normalisation) against plain HF
autograd on an unpatched copy of the model (HF masks, full logits, torch AdamW); each runtime fix switched off alone;
then a short end-to-end train run on a synthetic rs2-format file (plan, validation, export).
L4 (real RS2 tokens; Gemma-4-31B NF4):
  A  StreamCE vs full-logit autograd on 2k real hidden states (loss, dL/dh)
  B  LoRA gradients at T=4096 with each runtime fix alone and all together vs all off (run twice = noise floor)
  C  all fixes on, one fwd+bwd at T = 8k, 24k, 28k, 32k with a per-component memory breakdown at the peak (allocator
     trace replay), stop at the first OOM; then one ShardedAdamW step on the last gradients
Rank 0 writes OUT_DIR/lc_probe.json after every test.
"""
import gc
import glob
import json
import time
import zlib

import numpy as np
import torch
import torch.distributed as dist

from . import fix_attention, fix_head
from .config import LC, OUT_DIR
from .data import IGNORE, TrainData
from .fix_head import StreamCE, reference_ce
from .fix_lora import ShardedAdamW

FLAGS = tuple(LC)
_EPS = {}


def set_flags(fm, **on):
    for k in FLAGS:
        LC[k] = bool(on.get(k, False))
    fix_attention.install(fm)


def _key(n):
    return n.replace("_fsdp_wrapped_module.", "").removeprefix("m.").replace(".default.", ".")


def _randomize(named, std=0.05, only="lora_"):
    """Same deterministic values for the same LoRA tensor in any model and on every rank (PEFT's B = 0 init gives A
    no gradient and leaves the forward untouched by LoRA)."""
    for n, p in named:
        if only in n:
            g = torch.Generator().manual_seed(zlib.crc32(_key(n).encode()))
            p.data.copy_((torch.randn(p.shape, generator=g) * std).to(p.dtype))


# ------------------------------------------------------------------ inputs
def _episodes():
    if "eps" not in _EPS:
        f = sorted(glob.glob("/kaggle/input/**/rs2.npz", recursive=True))
        _EPS["eps"] = []
        if f:
            d = np.load(f[0])
            st, ids, w = d["ep_starts"], d["ids"], d["weight"]
            _EPS["eps"] = [(ids[st[k]:st[k + 1]], w[st[k]:st[k + 1]]) for k in range(len(st) - 1)]
            _EPS["eps"].sort(key=lambda e: -len(e[0]))
    return _EPS["eps"]


def _seq(T, k, dev, vocab=None, tail=False):
    """Sequence k of length T: a real RS2 episode (longest first; concatenated beyond the longest), else random."""
    eps = _episodes()
    if eps:
        long = [e for e in eps if len(e[0]) >= T]
        parts = [long[k % len(long)]] if long else [eps[(k + j) % len(eps)] for j in range(len(eps))]
        cut = slice(len(parts[0][0]) - T, None) if tail and long else slice(0, T)
        ids = torch.from_numpy(np.concatenate([e[0] for e in parts])[cut].astype(np.int64))
        wts = torch.from_numpy(np.concatenate([e[1] for e in parts])[cut][1:].astype(np.float32))
        assert ids.numel() == T, f"RS2 holds fewer than {T} tokens"
    else:
        g = torch.Generator().manual_seed(1000 + k)
        ids = torch.randint(10, vocab, (T,), generator=g)
        wts = (torch.rand(T - 1, generator=g) < 0.6).float() * torch.rand(T - 1, generator=g) * 3
    lab = ids[1:].clone()
    lab[wts == 0] = IGNORE
    return ids[None].to(dev), lab.to(dev), wts.to(dev)


# ------------------------------------------------------------------ gradient comparison
def _grads(tr):
    return [p.grad.detach().float().clone() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32)
            for p in tr.lora]


def _compare(a, b, reduce=True):
    """a vs reference b (lists of tensors): cosine, norm ratio, worst per-tensor relative error; summed over ranks
    (each rank holds its own sequence's grads) when reduce."""
    dev = a[0].device
    acc = torch.zeros(3, device=dev, dtype=torch.float64)
    worst = torch.zeros(1, device=dev, dtype=torch.float64)
    for x, y in zip(a, b):
        acc += torch.stack([(x * y).sum(), (x * x).sum(), (y * y).sum()]).double()
        if y.norm() > 0:
            worst = torch.maximum(worst, ((x - y).norm() / y.norm()).double().reshape(1))
    if reduce:
        dist.all_reduce(acc)
        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    return {"cosine": round(float(acc[0] / (acc[1].sqrt() * acc[2].sqrt() + 1e-30)), 6),
            "norm_ratio": round(float(acc[1].sqrt() / (acc[2].sqrt() + 1e-30)), 6), "worst_tensor_rel": round(float(worst), 6)}


HOST = {}


def _host_avail_gb():
    with open("/proc/meminfo") as f:
        return round(int(next(l for l in f if l.startswith("MemAvailable")).split()[1]) / 2 ** 20, 1)


def _fwd_bwd(tr, ids, lab, wts, norm, smoke):
    for p in tr.lora:
        p.grad = None
    with torch.autograd.graph.save_on_cpu(pin_memory=False):  # pageable: pinned blocks round up (host OOM at 28k, v2)
        loss = tr.fm(ids, lab, wts, norm)
    HOST["avail_after_fwd_gb"] = _host_avail_gb()  # all checkpoint inputs are on the host now (node-wide figure)
    loss.backward()
    return float(loss)


# ------------------------------------------------------------------ attention unit test (no model)
def _attn_unit(dev, smoke):
    """lc_attention (bias mask, bool mask) and HF's sdpa path with its dense mask, each vs an fp32 reference, at the
    real head shapes: forward error and q/k/v gradient cosine. Lengths include one that is not a multiple of 16."""
    from types import SimpleNamespace
    torch.manual_seed(0)
    cases = ([("sliding", 4, 2, 16, 16), ("global", 4, 1, 32, None)] if smoke else
             [("sliding", 32, 16, 256, 1024), ("global", 32, 4, 512, None)])
    rows, dt = [], torch.float32 if smoke else torch.bfloat16
    for kind, H, KV, D, W in cases:
        for T in ((100, 103) if smoke else (4096, 4100)):
            q, k, v = (torch.randn(1, n, T, D, device=dev) * D ** -0.25 for n in (H, KV, KV))
            mod = SimpleNamespace(is_causal=True, sliding_window=W, num_key_value_groups=H // KV)
            i, j = torch.arange(T, device=dev)[:, None], torch.arange(T, device=dev)[None, :]
            allowed = (j <= i) & (j > i - W) if W else (j <= i)
            qr, kr, vr = (x.clone().requires_grad_(True) for x in (q, k, v))
            ref = []
            for h in range(H):  # fp32 reference, one head at a time
                sc = (qr[0, h] @ kr[0, h // (H // KV)].T).masked_fill(~allowed, float("-inf"))
                ref.append(torch.softmax(sc, -1) @ vr[0, h // (H // KV)])
            ref = torch.stack(ref, 1)  # [T, H, D]
            go = torch.randn_like(ref)
            (ref * go).sum().backward()
            gref = [x.grad for x in (qr, kr, vr)]
            for variant in ("lc_bias", "lc_bool", "hf_dense"):
                qx, kx, vx = (x.to(dt).clone().requires_grad_(True) for x in (q, k, v))
                LC["attn"] = variant != "hf_dense"
                fix_attention.MASK = "bool" if variant == "lc_bool" else "bias"
                mask = None if variant != "hf_dense" or not W else allowed[None, None]
                o, _ = fix_attention.lc_attention(mod, qx, kx, vx, mask, scaling=1.0, sliding_window=W)
                (o.float() * go[None]).sum().backward()
                err = float((o[0].float() - ref).norm() / ref.norm())
                cos = [float(torch.nn.functional.cosine_similarity(a.grad.float().flatten(), b.flatten(), 0))
                       for a, b in zip((qx, kx, vx), gref)]
                rows.append({"kind": kind, "T": T, "variant": variant, "fwd_rel_err": round(err, 6),
                             "grad_cos_qkv": [round(c, 6) for c in cos]})
            del q, k, v, qr, kr, vr, ref, go, gref
    fix_attention.MASK = "bias"
    return rows


# ------------------------------------------------------------------ memory breakdown (allocator trace replay)
_KEEP = ("trainkit/", "transformers/", "peft/", "bitsandbytes/", "torch/utils/checkpoint", "torch/distributed/fsdp",
         "torch/autograd/graph", "torch/nn/functional")


def _frame_key(frames):
    for f in frames:  # innermost first
        fn = f.get("filename", "")
        if any(k in fn for k in _KEEP):
            short = fn.split("site-packages/")[-1].split("working/")[-1]
            return f"{short}:{f.get('line')}:{f.get('name')}"
    return "other"


def _breakdown(base_bytes, top=18):
    """Replay this device's allocator trace: the moment of peak live bytes and what was live then, by allocating
    frame. base_bytes = memory allocated when recording started (weights etc., not in the trace)."""
    snap = torch.cuda.memory._snapshot()
    trace = snap["device_traces"][torch.cuda.current_device()]
    live, cur, peak, peak_i = {}, 0, 0, -1
    for i, e in enumerate(trace):
        a = e.get("action")
        if a == "alloc":
            live[e["addr"]] = e["size"]
            cur += e["size"]
            if cur > peak:
                peak, peak_i = cur, i
        elif a in ("free_requested", "free_completed") and e["addr"] in live:
            cur -= live.pop(e["addr"])
    live, by = {}, {}
    for e in trace[:peak_i + 1]:
        a = e.get("action")
        if a == "alloc":
            live[e["addr"]] = (e["size"], _frame_key(e.get("frames", [])))
        elif a in ("free_requested", "free_completed"):
            live.pop(e["addr"], None)
    for size, k in live.values():
        by[k] = by.get(k, 0) + size
    rows = sorted(by.items(), key=lambda kv: -kv[1])[:top]
    return {"base_gib": round(base_bytes / 2 ** 30, 2), "trace_peak_gib": round(peak / 2 ** 30, 2),
            "total_peak_gib": round((base_bytes + peak) / 2 ** 30, 2), "trace_events": len(trace),
            "top": [[k, round(v / 2 ** 30, 3)] for k, v in rows]}


# ------------------------------------------------------------------ probe
def probe(tr, smoke):
    out = {"smoke": smoke, "flags": list(FLAGS)}
    path = OUT_DIR / "lc_probe.json"
    dev = tr.dev

    def save():
        if tr.rank == 0:
            path.write_text(json.dumps(out, indent=1))

    def reset():
        gc.collect()
        if not smoke:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def peak():
        return 0 if smoke else round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)

    if smoke:
        out["U_attention"] = _attn_unit(dev, True)
        save()
        _smoke(tr, out, save)
        out["done"] = True
        save()
        return

    # LoRA B ~ N(0, 1e-3) (A keeps PEFT's kaiming init, identical on all ranks): BA*alpha/r ~ 1e-4 per weight entry,
    # ~0.5% of the base weights, so LoRA shapes the forward and both A and B get gradients
    _randomize(tr.fm.named_parameters(), std=1e-3, only="lora_B")

    out["U_attention"] = _attn_unit(dev, False)  # before anything else: decides whether B/C below mean anything
    for r in out["U_attention"]:
        tr.log(event="lc_U", **r)
    save()

    def save_rank(key, row):  # every rank keeps its own copy: an OOM on rank r > 0 can hang rank 0 before it saves
        (OUT_DIR / f"lc_probe_rank{tr.rank}.json").write_text(json.dumps({**rank_rows, key: row}, indent=1))
        rank_rows[key] = row
    rank_rows = {}

    # A. StreamCE exactness on real hidden states; P_CHUNK 512 so the position chunks are exercised too (the CPU smoke
    # had one vocab tile and one chunk). Reference: full-logit autograd 512 positions at a time (2 GB fp32 per chunk)
    reset()
    ids, lab, wts = _seq(2049, tr.rank, dev, tail=True)  # the episode's end: model-written tokens (v2: the head was all prompt)
    set_flags(tr.fm, **{k: True for k in FLAGS})
    with torch.no_grad():
        tr.fm.eval()
        h = tr.fm(ids, lab, wts, hidden=True)  # through the FSDP root (root-owned params are only valid in there)
        tr.fm.train()
    head, cap = tr.fm.module.head[0], tr.fm.module.cap
    norm = float(wts.sum())
    hr = h.detach().clone().requires_grad_(True)
    chunk0, fix_head.P_CHUNK = fix_head.P_CHUNK, 512
    ls = StreamCE.apply(hr, head, lab.clamp(min=0), wts, cap, norm)
    fix_head.P_CHUNK = chunk0
    ls.backward()
    hf = h.detach().clone().requires_grad_(True)
    wg, lf = head.to(dev), 0.0
    for a in range(0, hf.shape[0], 512):
        part = reference_ce(hf[a:a + 512], wg, lab[a:a + 512].clamp(min=0), wts[a:a + 512], cap, norm)
        part.backward()
        lf += float(part.detach())
    del wg
    out["A_streamce"] = {"loss_stream": round(float(ls), 6), "loss_ref": round(lf, 6),
                         "abs_diff": round(abs(float(ls) - lf), 7), "peak_gb": peak(), **_compare([hr.grad.float()], [hf.grad.float()]),
                         "n_trained": int((wts > 0).sum())}
    tr.log(event="lc_A", **out["A_streamce"])
    del h, hr, hf, ls, lf
    save()

    # B. exactness of each runtime fix at T=4096 (real tokens, one episode per rank)
    T = 4096
    ids, lab, wts = _seq(T, tr.rank, dev)
    norm = float(wts.sum())
    runs = [("off", {}), ("off_again", {})] + [(k, {k: True}) for k in FLAGS] + [("all", {k: True for k in FLAGS})]
    ref, out["B_grad"] = None, []
    for name, on in runs:
        set_flags(tr.fm, **on)
        reset()
        t = time.perf_counter()
        loss = _fwd_bwd(tr, ids, lab, wts, norm, smoke)
        row = {"run": name, "loss": round(loss, 6), "sec": round(time.perf_counter() - t, 1), "peak_gb": peak()}
        g = _grads(tr)
        if ref is None:
            ref, ref_loss = g, loss
        else:
            row.update(_compare(g, ref), loss_abs_diff=round(abs(loss - ref_loss), 7))
        out["B_grad"].append(row)
        save_rank(f"B_{name}", row)
        tr.log(event="lc_B", **row)
        save()
    del ref, g

    # C. memory / time scan, all fixes on
    set_flags(tr.fm, **{k: True for k in FLAGS})
    out["C_scan"] = []
    last_ok = None
    for T in (28672, 32768):  # 8k: 8.24 GiB, 24k: 14.95 GiB peak (v2)
        ids, lab, wts = _seq(T, tr.rank, dev)
        reset()
        base = torch.cuda.memory_allocated()
        torch.cuda.memory._record_memory_history(max_entries=400000, stacks="python")
        row = {"T": T, "real_tokens": bool(_episodes())}
        t = time.perf_counter()
        try:
            loss = _fwd_bwd(tr, ids, lab, wts, float(wts.sum()), smoke)
            torch.cuda.synchronize()
            sec = time.perf_counter() - t
            row.update(ok=True, loss=round(loss, 4), sec=round(sec, 1), tok_s_rank=round(T / sec, 1), peak_gb=peak(),
                       reserved_gb=round(torch.cuda.max_memory_reserved() / 2 ** 30, 2))
            last_ok = T
        except torch.cuda.OutOfMemoryError as e:
            row.update(ok=False, error=f"OOM: {str(e)[:240]}", peak_gb=peak())
        try:
            row["breakdown"] = _breakdown(base)
        except Exception as e:  # the breakdown must never cost the result
            row["breakdown_error"] = repr(e)[:200]
        torch.cuda.memory._record_memory_history(enabled=None)
        flag = torch.tensor([0.0 if row["ok"] else 1.0], device=dev)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        row["any_rank_failed"] = bool(float(flag))
        row["host_avail_after_fwd_gb"] = HOST.get("avail_after_fwd_gb")
        out["C_scan"].append(row)
        save_rank(f"C_{T}", row)
        tr.log(event="lc_C", **{k: v for k, v in row.items() if k != "breakdown"})
        save()
        if row["any_rank_failed"]:
            break
    # one optimizer step on the last successful gradients (the trainer's real update path)
    if last_ok and not out["C_scan"][-1]["any_rank_failed"]:
        opt = ShardedAdamW(tr.lora, lr=1e-5)
        reset()
        t = time.perf_counter()
        gn = opt.step(1.0)
        torch.cuda.synchronize()
        out["C_opt_step"] = {"after_T": last_ok, "grad_norm": round(float(gn), 4), "sec": round(time.perf_counter() - t, 2),
                             "peak_gb": peak()}
        tr.log(event="lc_opt", **out["C_opt_step"])
    out["done"] = True
    save()


def _smoke(tr, out, save):
    """CPU fp32: new stack (FSDP, 2 gloo ranks) vs plain HF autograd on an unpatched copy, then a tiny train run."""
    from .model import add_lora, load_base
    dev, vocab = tr.dev, tr.vocab
    ref = add_lora(load_base(tr.model_path, tr.rank, True), tr.s.rank, tr.s.alpha, torch.float32)
    _randomize(ref.named_parameters())
    _randomize(tr.fm.named_parameters())
    ref_lora = {_key(n): p for n, p in ref.named_parameters() if "lora_" in n}
    name_of = {id(p): _key(n) for n, p in tr.fm.named_parameters() if "lora_" in n}
    mine = [name_of[id(p)] for p in tr.lora]  # tr.lora order (what _grads and ShardedAdamW use)
    assert sorted(mine) == sorted(ref_lora), "LoRA tensors differ between the FSDP model and the reference"
    T = 100  # > 6 sliding windows of 16 and > 3 MLP tiles of 32
    seqs = [_seq(T, k, dev, vocab) for k in range(tr.world)]
    norm = sum(float(w.sum()) for _, _, w in seqs)

    # reference: every rank computes all ranks' sequences on the unpatched model (flags off -> HF paths)
    for k in FLAGS:
        LC[k] = False
    ref.train()
    rlm = ref.base_model.model.model.language_model
    head = ref.base_model.model.get_output_embeddings().weight
    cap = tr.fm.module.cap
    rloss = 0.0
    for ids, lab, wts in seqs:
        h = rlm(input_ids=ids, use_cache=False).last_hidden_state[0][:-1]
        loss = reference_ce(h, head, lab.clamp(min=0), wts, cap, norm)
        loss.backward()
        rloss += float(loss)
    rgrad = [ref_lora[k].grad.detach().float().clone() for k in mine]

    def new_stack(**on):
        set_flags(tr.fm, **on)
        ids, lab, wts = seqs[tr.rank]
        loss = _fwd_bwd(tr, ids, lab, wts, norm, True)
        g = [x.clone() for x in _grads(tr)]
        for x in g:
            dist.all_reduce(x)  # what ShardedAdamW's reduce-scatter sums
        tot = torch.tensor([loss], dtype=torch.float64)
        dist.all_reduce(tot)
        return float(tot), g

    out["S_exact"] = []
    for name, on in [("all", {k: True for k in FLAGS})] + [(f"all_but_{k}", {j: j != k for j in FLAGS}) for k in FLAGS] + \
                    [("none", {})]:
        loss, g = new_stack(**on)
        row = {"run": name, "loss": round(loss, 7), "loss_ref": round(rloss, 7), "loss_abs_diff": round(abs(loss - rloss), 8),
               **_compare(g, rgrad, reduce=False)}
        row["pass"] = row["cosine"] > 0.99999 and row["worst_tensor_rel"] < 1e-3 and row["loss_abs_diff"] < 1e-4
        out["S_exact"].append(row)
        tr.log(event="lc_S", **row)
        save()

    # one optimizer step: ShardedAdamW (all fixes on) vs torch AdamW + clip on the reference, fed the same summed
    # gradients (Adam's first step is ~lr * sign(g), so near-zero grads that differ by noise would flip by 2 lr)
    _, g = new_stack(**{k: True for k in FLAGS})
    opt = ShardedAdamW(tr.lora, lr=1e-2)
    gn = opt.step(1.0)
    rparams = [ref_lora[k] for k in mine]
    for p, x in zip(rparams, g):
        p.grad = x.clone()
    rn = torch.nn.utils.clip_grad_norm_(rparams, 1.0)
    ropt = torch.optim.AdamW(rparams, lr=1e-2, weight_decay=0.0)
    ropt.step()
    diff = max(float((a.detach() - b.detach()).abs().max()) for a, b in zip(tr.lora, rparams))
    out["S_opt"] = {"grad_norm": round(float(gn), 6), "grad_norm_ref": round(float(rn), 6), "max_param_abs_diff": diff,
                    "pass": diff < 1e-5 and abs(float(gn) - float(rn)) < 1e-4 * max(1.0, float(rn))}
    tr.log(event="lc_S_opt", **out["S_opt"])
    save()
    del ref

    # tiny end-to-end train run on a synthetic rs2-format file
    f = OUT_DIR / "smoke_rs2.npz"
    if tr.rank == 0:
        rng = np.random.default_rng(0)
        lens = rng.integers(40, 120, size=14)
        ids = rng.integers(10, vocab, size=int(lens.sum())).astype(np.int32)
        w = (rng.random(int(lens.sum())) < 0.5).astype(np.float16) * 2
        np.savez(f, ids=ids, weight=w, ep_starts=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
                 val=np.array([0] * 10 + [1] * 4, dtype=np.uint8))
    dist.barrier()
    data = TrainData(f, vocab=vocab, seed=0, max_len=110)
    tr.s.epochs, tr.s.grad_accum, tr.s.lr, tr.s.warmup_steps = 2.0, 2, 1e-3, 1
    tr.run(data)
    out["S_train"] = {**data.stats(), "plan_steps": len(data.plan(data.train_w, tr.world, 2, 2.0))}
    save()
