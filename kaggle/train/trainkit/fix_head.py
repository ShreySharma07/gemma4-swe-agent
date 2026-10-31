"""
Fix: the tied embedding / lm_head (262144 x 5376 bf16 = 2.82 GB) lives in pinned host RAM, outside FSDP (exact).
Before: it sat in the FSDP root unit, which FULL_SHARD never reshards after forward, so every rank held the whole
2.62 GiB through the step (plus its 0.7 GB shard). Now:
  embedding  rows are looked up on the CPU and copied to the GPU (T x 5376: 0.3 GB at 28k); frozen, no grad.
  loss       StreamCE: weighted next-token CE with Gemma's final-logit softcap, the vocabulary streamed through the
             GPU in tiles, two passes (1: running logsumexp + target logit, 2: dL/dh). Exact up to summation order;
             dL/dh is accumulated in fp32. ~5.6 GB of host->GPU copies per sequence (~0.5 s on PCIe 4).
Loss normalisation is the caller's: loss = sum_t w_t * ce_t / norm (norm = the optimizer step's total weight, so
every trained token counts the same however the step's sequences are split across ranks and micro-steps).
"""
import torch
import torch.nn as nn

V_TILE = 16384  # vocab rows per upload (176 MB bf16)
P_CHUNK = 4096  # positions per matmul (fp32 logits tile 4096 x 16384 = 268 MB)


class HostEmbedding(nn.Module):
    """Gemma4TextScaledWordEmbedding with its weight in (pinned) host RAM as a plain attribute, so FSDP never sees it."""

    def __init__(self, emb, device):
        super().__init__()
        w = emb.weight.detach().to("cpu")
        self.w = w.pin_memory() if device.type == "cuda" else w
        self.scale = emb.embed_scale.to(w.dtype)  # HF: lookup * embed_scale cast to the weight dtype
        self.device = device

    def forward(self, input_ids):
        rows = nn.functional.embedding(input_ids.cpu(), self.w) * self.scale
        return rows.to(self.device, non_blocking=True)


def move_to_host(m, device):
    """Swap the text embedding for HostEmbedding and drop the tied lm_head. Returns the host weight (= lm_head)."""
    base = m.base_model.model  # PeftModel -> LoraModel -> Gemma4ForConditionalGeneration
    lm = base.model.language_model
    head = base.get_output_embeddings()
    assert head.weight.data_ptr() == lm.embed_tokens.weight.data_ptr(), "expected tied embedding / lm_head"
    lm.embed_tokens = HostEmbedding(lm.embed_tokens, device)
    base.lm_head = nn.Identity()
    del head
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return lm.embed_tokens.w


def _tiles(w, dev):
    for v0 in range(0, w.shape[0], V_TILE):
        yield v0, w[v0:v0 + V_TILE].to(dev, non_blocking=True)


class StreamCE(torch.autograd.Function):
    """h [T, H] (GPU), w [V, H] (host), y [T] targets, wt [T] float weights (0 = not trained), cap softcap, norm."""

    @staticmethod
    def forward(ctx, h, w, y, wt, cap, norm):
        idx = (wt > 0).nonzero().squeeze(1)
        hs, ys, ws = h[idx], y[idx], wt[idx].float()
        n, dev = idx.numel(), h.device
        m = torch.full((n,), -float("inf"), device=dev)
        s = torch.zeros(n, device=dev)
        zt = torch.zeros(n, device=dev)

        def logits(a, wv):
            z = (hs[a:a + P_CHUNK] @ wv.T).float()
            return torch.tanh(z / cap) * cap if cap else z

        for v0, wv in _tiles(w, dev):  # pass 1: logsumexp and target logits
            for a in range(0, n, P_CHUNK):
                z = logits(a, wv)
                mz = torch.maximum(m[a:a + P_CHUNK], z.max(1).values)
                s[a:a + P_CHUNK] = s[a:a + P_CHUNK] * torch.exp(m[a:a + P_CHUNK] - mz) + torch.exp(z - mz[:, None]).sum(1)
                m[a:a + P_CHUNK] = mz
                yy = ys[a:a + P_CHUNK] - v0
                hit = (yy >= 0) & (yy < wv.shape[0])
                zt[a:a + P_CHUNK][hit] = z[hit.nonzero().squeeze(1), yy[hit]]
        lse = m + torch.log(s)
        loss = (ws * (lse - zt)).sum() / norm
        gs = torch.zeros(n, h.shape[1], device=dev, dtype=torch.float32)
        coef = ws / norm
        for v0, wv in _tiles(w, dev):  # pass 2: dL/dh = sum_v coef * (p_v - [v = y]) * dz_v/dl_v * w_v
            for a in range(0, n, P_CHUNK):
                z = logits(a, wv)
                g = torch.exp(z - lse[a:a + P_CHUNK, None])
                yy = ys[a:a + P_CHUNK] - v0
                hit = (yy >= 0) & (yy < wv.shape[0])
                g[hit.nonzero().squeeze(1), yy[hit]] -= 1.0
                if cap:
                    g *= 1.0 - (z / cap) ** 2  # d(cap * tanh(l / cap))/dl
                g *= coef[a:a + P_CHUNK, None]
                gs[a:a + P_CHUNK] += (g.to(h.dtype) @ wv).float()  # bf16 matmul as autograd would; fp32 sum over tiles
        gh = torch.zeros_like(h)
        gh[idx] = gs.to(h.dtype)
        ctx.save_for_backward(gh)
        return loss

    @staticmethod
    def backward(ctx, go):
        gh, = ctx.saved_tensors
        return gh * go, None, None, None, None, None


def reference_ce(h, w, y, wt, cap, norm):
    """Plain autograd over full logits (tests only: T x 262144 fp32)."""
    z = (h @ w.T).float()
    if cap:
        z = torch.tanh(z / cap) * cap
    ce = nn.functional.cross_entropy(z, y, reduction="none")
    return (wt.float() * ce).sum() / norm
