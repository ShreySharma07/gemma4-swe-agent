"""
Fix: checkpointed RMSNorm (exact: the same function, recomputed). Gemma4RMSNorm upcasts to fp32, so autograd keeps
fp32 copies of its input and intermediates: ~0.7 GB per hidden-size norm at 32k (4 per layer) plus the q/k/v norms.
As its own checkpoint a norm keeps only its bf16 input. Applies only when the input needs grad (training).
"""
import torch
from torch.utils.checkpoint import checkpoint

from .config import LC

_HF_FORWARD = None


def ckpt_forward(self, hidden_states):
    if LC["norm_ckpt"] and torch.is_grad_enabled() and hidden_states.requires_grad:
        return checkpoint(_HF_FORWARD, self, hidden_states, use_reentrant=False)
    return _HF_FORWARD(self, hidden_states)


def install():
    global _HF_FORWARD
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
    if _HF_FORWARD is None:
        _HF_FORWARD = Gemma4RMSNorm.forward
        Gemma4RMSNorm.forward = ckpt_forward
