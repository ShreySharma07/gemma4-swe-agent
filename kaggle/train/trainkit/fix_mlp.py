"""
Fix: tiled MLP (exact). Gemma-4's MLP is per position, so running it over sequence tiles gives the same outputs.
During a layer's backward recompute, HF keeps gate, up, act(gate) and their product for the whole sequence:
4 x T x 21504 bf16 = 5.6 GB at 32k, plus as much again in backward temporaries. With each tile its own checkpoint,
only the tile being differentiated is materialised (tile 4096: ~0.7 GB). Cost: one extra MLP forward per layer.
LoRA gradients from the tiles add up in .grad exactly as one big matmul would (up to summation order).
"""
import torch
from torch.utils.checkpoint import checkpoint

from .config import LC

TILE = 4096  # set from TrainSettings.mlp_tile by install()
_HF_FORWARD = None


def tiled_forward(self, x):
    T = x.shape[-2]
    if not (LC["mlp_tile"] and torch.is_grad_enabled() and T > TILE):
        return _HF_FORWARD(self, x)
    return torch.cat([checkpoint(_HF_FORWARD, self, x[..., a:a + TILE, :], use_reentrant=False)
                      for a in range(0, T, TILE)], dim=-2)


def install(tile):
    global _HF_FORWARD, TILE
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP
    TILE = tile
    if _HF_FORWARD is None:
        _HF_FORWARD = Gemma4TextMLP.forward
        Gemma4TextMLP.forward = tiled_forward
