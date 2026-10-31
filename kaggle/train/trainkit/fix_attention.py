"""
Fix: attention without dense T x T masks (exact). Replaces the "sdpa" entry HF's Gemma-4 attention looks up.
  sliding layers (50, window W=1024): queries in blocks of W attend to keys [block start - W + 1, block end) with a
      small (W x 2W) mask -- the same rule as HF's sliding mask (q - W < k <= q). HF instead builds a dense bool T x T
      mask (1 GiB at 32k), keeps it in the checkpoint kwargs, turns it into a 2T^2 bf16 bias inside SDPA, and computes
      all T x T scores: ~8x the attention FLOPs at 32k.
  global layers (10, 4 KV heads x 512): head_dim 512 > 256 makes HF skip SDPA's GQA and repeat K/V to all 32 heads
      (4 GB of K/V copies at 32k, kept for backward). Here: one causal SDPA call per KV group (8 query heads).
  Every block/group is its own checkpoint, so backward keeps only views of q/k/v, never the expanded copies.
Masks are disabled by passing NO_MASKS (a dict) as attention_mask: HF then skips mask creation for every layer.
Anything else (a real mask, a KV cache with q_len != k_len, decoding) goes to HF's own sdpa_attention_forward.
Also: the last full and sliding layer keep their full-length K/V in shared_kv_states for KV sharing; Gemma-4-31B
has num_kv_shared_layers=0, so nothing reads them (0.75 GiB at 32k, held by the checkpoint kwargs) -> turned off.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import LC

NO_MASKS = {"full_attention": None, "sliding_attention": None}
MASK = "bias"  # "bias": additive mask in q's dtype, row stride padded to 16 (L4 v2: the bool-mask variant's gradient
               # had cosine 0.55 vs HF's dense-mask path at 4k); "bool": the v2 variant, kept for the probe's unit test
_HF_SDPA = None
BLOCK = None  # query block for sliding layers (None = the window); the probe varies it as a rounding-only control
CAPTURE = None  # probe only: a list -> every call appends its arguments (lcprobe._real_width_diag)


def _grouped_causal(q, k, v, scale):
    """q [1, g, T, D]; k, v [1, 1, T, D] -> causal attention of g query heads over one KV head."""
    g = q.shape[1]
    return F.scaled_dot_product_attention(q, k.expand(-1, g, -1, -1).contiguous(), v.expand(-1, g, -1, -1).contiguous(),
                                          is_causal=True, scale=scale)


def _window_block(query, key, value, s, e, window, scale):
    """Queries [s, e) over keys [s - W + 1, e) with HF's rule q - W < k <= q. Slicing and the GQA expansion happen in
    here, so the checkpoint keeps only the full q/k/v it was given, not a per-block copy."""
    ks, g = max(0, s - window + 1), query.shape[1] // key.shape[1]
    q = query[:, :, s:e]
    k = key[:, :, ks:e].repeat_interleave(g, dim=1)  # same head order as HF's repeat_kv
    v = value[:, :, ks:e].repeat_interleave(g, dim=1)
    i = torch.arange(s, e, device=q.device)[:, None]
    j = torch.arange(ks, e, device=q.device)[None, :]
    mask = (j <= i) & (j > i - window)
    if MASK == "bool":
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None], scale=scale)
    L = e - ks  # the memory-efficient kernel wants the mask's row stride aligned to 16: pad, then slice
    bias = torch.full((e - s, -(-L // 16) * 16), float("-inf"), dtype=q.dtype, device=q.device)[:, :L]
    bias.masked_fill_(mask, 0.0)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias[None, None], scale=scale)


def _ckpt(fn, *args):
    return checkpoint(fn, *args, use_reentrant=False) if torch.is_grad_enabled() else fn(*args)


def lc_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    T, Tk = query.shape[2], key.shape[2]
    if CAPTURE is not None:
        CAPTURE.append({"layer": getattr(module, "layer_idx", None), "T": T, "Tk": Tk, "q": query, "k": key, "v": value,
                        "mask": attention_mask, "dropout": dropout, "scaling": scaling, "is_causal": is_causal,
                        "kwargs": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else type(v).__name__)
                                   for k, v in kwargs.items()}})
    if (not LC["attn"] or attention_mask is not None or T != Tk or T == 1 or dropout
            or not getattr(module, "is_causal", True)):
        return _HF_SDPA(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal,
                        **kwargs)
    window = kwargs.get("sliding_window") or getattr(module, "sliding_window", None)
    hkv, g = key.shape[1], query.shape[1] // key.shape[1]
    if window and T > window:
        step = BLOCK or window
        out = torch.cat([_ckpt(_window_block, query, key, value, s, min(s + step, T), window, scaling)
                         for s in range(0, T, step)], dim=2)
    else:
        out = torch.cat([_ckpt(_grouped_causal, query[:, h * g:(h + 1) * g], key[:, h:h + 1], value[:, h:h + 1], scaling)
                         for h in range(hkv)], dim=1)
    return out.transpose(1, 2).contiguous(), None


def install(model):
    """Register lc_attention under "sdpa" (once) and set the K/V-store switch on every text attention module."""
    global _HF_SDPA
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if _HF_SDPA is None:
        _HF_SDPA = sdpa_attention_forward
        ALL_ATTENTION_FUNCTIONS["sdpa"] = lc_attention
    for mod in model.modules():
        if type(mod).__name__ == "Gemma4TextAttention":
            if not hasattr(mod, "_hf_store_kv"):
                mod._hf_store_kv = mod.store_full_length_kv
            mod.store_full_length_kv = mod._hf_store_kv and not LC["no_kv_store"]
