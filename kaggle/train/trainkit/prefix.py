"""
Idea A (PA1 probe): prefix-cached truncated backprop, so late-run steps train in their real 20-30k context.
The first P tokens run under no_grad (chunked prefill into the model's own HF cache), the cache is frozen (update()
returns cache + new keys without storing them, so checkpoint recompute in backward sees the same state), then
forward + backward runs on the last S tokens only. LoRA shapes the prefix forward; gradients stop at the cache.
Gemma-4-31B: 50 of 60 layers are sliding-window (1024), so the cache is ~0.8 GB for those + ~1-2 GB per 24k tokens
for the 10 global layers (4 KV heads x 512).
probe(): correctness (cache path loss == full-sequence loss on the same targets), a short overfit run, and peak
memory / time over a (P, S) grid on random tokens. Rank 0 writes OUT_DIR/prefix_probe.json after every test.
"""
import contextlib
import functools
import json
import time
import types

import torch
import torch.nn as nn

from .config import OUT_DIR
from .data import IGNORE

PREFILL_CHUNK = 4096  # L4 v1: 8192-token no-grad chunks + a full-length cache OOMed at P=9216
TIMING = {}  # last prefill wall time (probe: prefill vs trained tokens/s)


def freeze_cache(cache):
    """Make every cache layer read-only: update() returns (stored + new) without storing, lengths stay at P."""
    for layer in cache.layers:
        k0, v0 = layer.keys.detach(), layer.values.detach()

        def update(self, key_states, value_states, cache_kwargs=None, k0=k0, v0=v0):
            return torch.cat([k0, key_states], dim=-2), torch.cat([v0, value_states], dim=-2)
        layer.update = types.MethodType(update, layer)
    return cache


def prefix_cache(lm, ids, chunk=PREFILL_CHUNK):
    """No-grad chunked prefill of ids [1, P] into the model's own cache (sliding layers keep only their window)."""
    t = time.perf_counter()
    was = lm.training
    lm.eval()  # HF turns use_cache off under gradient checkpointing in train mode
    from transformers import DynamicCache
    try:  # sliding layers then keep only their window (the model's own default may not)
        cache = DynamicCache(config=lm.config)
    except TypeError:
        cache = None
    with torch.no_grad():
        for a in range(0, ids.shape[1], chunk):
            pos = torch.arange(a, min(a + chunk, ids.shape[1]), device=ids.device)
            cache = lm(input_ids=ids[:, a:a + chunk], past_key_values=cache, use_cache=True, cache_position=pos,
                       position_ids=pos.unsqueeze(0)).past_key_values
    lm.train(was)
    if ids.is_cuda:
        torch.cuda.synchronize()
    TIMING["prefill_s"] = time.perf_counter() - t
    TIMING["cache_gb"] = sum(x.numel() * x.element_size() for layer in cache.layers for x in (layer.keys, layer.values)) / 2 ** 30
    TIMING["cache_layers"] = {k: sum(type(l).__name__ == k for l in cache.layers) for k in {type(l).__name__ for l in cache.layers}}
    return freeze_cache(cache)


@contextlib.contextmanager
def checkpoint_keeps_cache():
    """HF's GradientCheckpointingLayer sets past_key_values=None in train mode (a cache that grows would be appended
    twice by the recompute). Ours is frozen, so recompute is safe: checkpoint the layer with its kwargs unchanged."""
    from transformers.modeling_layers import GradientCheckpointingLayer
    orig = GradientCheckpointingLayer.__call__

    def call(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            return self._gradient_checkpointing_func(functools.partial(nn.Module.__call__, self, **kwargs), *args)
        return nn.Module.__call__(self, *args, **kwargs)
    GradientCheckpointingLayer.__call__ = call
    try:
        yield
    finally:
        GradientCheckpointingLayer.__call__ = orig


def suffix_hidden(lm, ids, prefix):
    """Hidden states of ids[:, prefix:] attending to a frozen cache of ids[:, :prefix]."""
    cache = prefix_cache(lm, ids[:, :prefix])
    pos = torch.arange(prefix, ids.shape[1], device=ids.device)
    with checkpoint_keeps_cache():
        return lm(input_ids=ids[:, prefix:], past_key_values=cache, use_cache=False, cache_position=pos,
                  position_ids=pos.unsqueeze(0)).last_hidden_state[0]


# ------------------------------------------------------------------ probe
def _example(T, P, vocab, device, seed):
    g = torch.Generator().manual_seed(seed)  # same tokens on every rank (FSDP needs equal shapes, not equal data)
    ids = torch.randint(10, vocab, (1, T), generator=g)
    full_lab = ids[0, 1:].clone()
    full_lab[:P] = IGNORE  # full sequence: train only targets after the prefix
    return ids.to(device), full_lab.to(device), ids[0, P + 1:].clone().to(device)


def probe(tr, smoke):
    fm, dev, log = tr.fm, tr.dev, tr.log
    vocab = min(tr.vocab, 50000)
    out = {"smoke": smoke}
    path = OUT_DIR / "prefix_probe.json"

    def save():
        if tr.rank == 0:
            path.write_text(json.dumps(out, indent=1))

    def peak():
        return 0 if smoke else round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)

    def reset():
        if not smoke:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # 1. correctness: same targets, full sequence vs prefix cache (crosses the sliding window when P > window)
    corr = []
    for P, S in ([(40, 24), (64, 48)] if smoke else [(3072, 2048), (9216, 2048)]):
        ids, full_lab, suf_lab = _example(P + S, P, vocab, dev, seed=P)
        fm.eval()
        with torch.no_grad():
            lf = float(fm(ids, full_lab))
            lp = float(fm(ids, suf_lab, prefix=P))
        fm.train()  # train mode = the gradient-checkpointing path (v1 smoke crashed only here); no_grad: a forward without
        with torch.no_grad():  # backward would leave FSDP's root unsharded and its graph alive (L4 v1 OOM)
            lt = float(fm(ids, suf_lab, prefix=P))
        cache_info = {"cache_gb": round(TIMING["cache_gb"], 3), "cache_layers": TIMING["cache_layers"]}
        with torch.no_grad():  # negative control: same targets with no prefix at all must differ from full
            ls = float(fm(ids[:, P:], suf_lab))
        corr.append({"P": P, "S": S, "loss_full": round(lf, 5), "loss_prefix": round(lp, 5), "abs_diff": round(abs(lf - lp), 5),
                     "loss_prefix_train": round(lt, 5), "abs_diff_train": round(abs(lf - lt), 5),
                     "loss_suffix_only": round(ls, 5), "abs_diff_suffix_only": round(abs(lf - ls), 5), **cache_info,
                     "peak_gb": peak()})
        reset()
        log(event="prefix_correct", **corr[-1])
    out["correctness"] = corr
    cfg = fm.module.m.config.get_text_config()
    out["config"] = {k: getattr(cfg, k, None) for k in ("attn_logit_softcapping", "final_logit_softcapping", "sliding_window",
                                                       "attention_k_eq_v")}
    if not smoke:
        out["config"]["sdpa"] = {"flash": torch.backends.cuda.flash_sdp_enabled(), "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                                 "math": torch.backends.cuda.math_sdp_enabled()}
    save()

    # 1b. gradient fidelity: LoRA grads of the truncated path vs full backprop through the prefix (same targets)
    P, S = (32, 32) if smoke else (4096, 4096)
    ids, full_lab, suf_lab = _example(P + S, P, vocab, dev, seed=11)
    grads = []
    for path_ in ("full", "prefix"):
        with torch.autograd.graph.save_on_cpu(pin_memory=not smoke):
            loss = fm(ids, full_lab) if path_ == "full" else fm(ids, suf_lab, prefix=P)
        loss.backward()
        grads.append([q.grad.detach().float().clone() if q.grad is not None else None
                      for q in fm.parameters() if q.requires_grad])
        for q in fm.parameters():
            q.grad = None
    acc = torch.zeros(3, device=dev, dtype=torch.float64)  # dot, |full|^2, |prefix|^2 over this rank's shards
    for a, b in zip(*grads):
        if a is not None and b is not None:
            acc += torch.stack([(a * b).sum(), (a * a).sum(), (b * b).sum()]).double()
    torch.distributed.all_reduce(acc)
    out["grad_fidelity"] = {"P": P, "S": S, "cosine": round(float(acc[0] / (acc[1].sqrt() * acc[2].sqrt() + 1e-30)), 4),
                            "norm_ratio_prefix_over_full": round(float(acc[2].sqrt() / (acc[1].sqrt() + 1e-30)), 4)}
    log(event="prefix_grad_fidelity", **out["grad_fidelity"])
    del grads
    save()

    # 2. overfit sanity: a few optimizer steps on one example through the prefix path; loss must fall
    P, S = (48, 32) if smoke else (16384, 4096)
    ids, _, suf_lab = _example(P + S, P, vocab, dev, seed=7)
    opt = torch.optim.AdamW([p for p in fm.parameters() if p.requires_grad], lr=1e-3 if smoke else 2e-4)
    losses = []
    reset()
    for _ in range(4):
        with torch.autograd.graph.save_on_cpu(pin_memory=not smoke):
            loss = fm(ids, suf_lab, prefix=P)
        loss.backward()
        fm.clip_grad_norm_(1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(round(float(loss), 4))
    out["overfit"] = {"P": P, "S": S, "losses": losses, "falls": losses[-1] < losses[0], "peak_gb": peak()}
    log(event="prefix_overfit", **out["overfit"])
    save()
    del opt

    # 3. memory / time grid, ascending total length; stop at the first OOM (later configs are larger)
    grid = [(0, 32), (32, 32), (64, 32)] if smoke else [(0, 8192), (16384, 4096), (16384, 8192), (24576, 4096), (24576, 6144),
                                                      (28672, 4096), (24576, 8192)]
    out["grid"] = []
    for P, S in grid:
        ids, full_lab, suf_lab = _example(P + S, P, vocab, dev, seed=P + S)
        reset()
        t = time.perf_counter()
        row = {"P": P, "S": S}
        try:
            with torch.autograd.graph.save_on_cpu(pin_memory=not smoke):
                loss = fm(ids, suf_lab, prefix=P) if P else fm(ids, full_lab)
            loss.backward()
            if not smoke:
                torch.cuda.synchronize()
            sec = time.perf_counter() - t
            pre = TIMING.get("prefill_s", 0.0) if P else 0.0
            row.update(ok=True, sec=round(sec, 1), prefill_s=round(pre, 1), peak_gb=peak(), loss=round(float(loss), 4),
                       cache_gb=round(TIMING.get("cache_gb", 0.0), 3) if P else 0.0,
                       prefill_tok_s=round(P / pre, 1) if P and pre else None,
                       trained_tok_s=round(S / max(1e-6, sec - pre), 1), host_pinned_gb=None if smoke else
                       round(torch.cuda.memory_stats().get("host_alloc_bytes.all.peak", 0) / 2 ** 30, 2))
        except torch.cuda.OutOfMemoryError as e:
            row.update(ok=False, error=f"OOM: {str(e)[:200]}", peak_gb=peak())
        for p in fm.parameters():
            p.grad = None
        flag = torch.tensor([0.0 if row["ok"] else 1.0], device=dev)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        row["any_rank_failed"] = bool(float(flag))
        out["grid"].append(row)
        log(event="prefix_grid", **row)
        save()
        if row["any_rank_failed"]:
            break
    # 4. unequal prefixes across ranks (same number of prefill chunks), as real batches will have; watched by NCCL timeout
    if not any(r["any_rank_failed"] for r in out["grid"]):
        P = (32 + 4 * tr.rank) if smoke else (17408 + 1024 * tr.rank)
        S = 32 if smoke else 4096
        ids, _, suf_lab = _example(P + S, P, vocab, dev, seed=99 + tr.rank)
        reset()
        t = time.perf_counter()
        with torch.autograd.graph.save_on_cpu(pin_memory=not smoke):
            loss = fm(ids, suf_lab, prefix=P)
        loss.backward()
        for q in fm.parameters():
            q.grad = None
        out["unequal_prefix"] = {"ok": True, "P_rank0": P, "S": S, "sec": round(time.perf_counter() - t, 1), "peak_gb": peak()}
        log(event="prefix_unequal", **out["unequal_prefix"])
    out["done"] = True
    save()
