"""Model: NF4 Gemma-4-31B + LoRA r16 + FSDP, and the masked, chunked, soft-capped loss (probe v4 recipe)."""
import re
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from .config import TARGETS
from .data import IGNORE

LORA_RE = r".*language_model.*\.(" + "|".join(TARGETS) + r")$"
CHUNK = 256  # loss chunk: 256 x 262k fp32 logits = 0.27 GB


def load_base(path, rank, smoke):
    """Every rank quantises the bf16 QAT weights to NF4 on its own GPU (~11 min); FSDP then shards them."""
    from transformers import AutoModelForImageTextToText
    if smoke:
        return AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float32, attn_implementation="sdpa")
    from transformers import BitsAndBytesConfig
    q = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,  # FSDP can only shard 4-bit weights stored in the compute dtype
        llm_int8_skip_modules=["vision_tower", "embed_vision", "multi_modal_projector", "audio_tower", "embed_audio",
                               "lm_head"])
    return AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                       quantization_config=q, device_map={"": rank})


def add_lora(m, r, alpha, dtype):
    from peft import LoraConfig, get_peft_model
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    m = get_peft_model(m, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none", target_modules=LORA_RE,
                                     task_type="CAUSAL_LM"))
    for n, p in m.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(dtype)  # FSDP flat params need one dtype per unit
    return m


class MaskedCE(torch.autograd.Function):
    """Next-token CE over the trained positions only, with Gemma's final-logit softcap, CHUNK positions at a time.
    The head is frozen, so only dL/dh is needed; it is computed in forward and kept (T x hidden)."""

    @staticmethod
    def forward(ctx, h, w, y, cap):
        idx = (y != IGNORE).nonzero().squeeze(1)
        gh = torch.zeros_like(h)
        tot = torch.zeros((), device=h.device, dtype=torch.float32)
        n = max(1, idx.numel())
        for i in range(0, idx.numel(), CHUNK):
            j = idx[i:i + CHUNK]
            hc = h[j].detach().requires_grad_(True)
            with torch.enable_grad():
                z = (hc @ w.T).float()
                if cap:
                    z = torch.tanh(z / cap) * cap
                loss = F.cross_entropy(z, y[j], reduction="sum")
                g, = torch.autograd.grad(loss, hc)
            gh[j] = g
            tot += loss.detach()
        ctx.save_for_backward(gh)
        ctx.n = n
        return tot / n

    @staticmethod
    def backward(ctx, go):
        gh, = ctx.saved_tensors
        return gh * (go / ctx.n), None, None, None


class LossWrap(nn.Module):
    """FSDP root: the tied lm_head/embedding is unsharded only inside the root forward, so the loss lives here."""

    def __init__(self, m, cap):
        super().__init__()
        self.m, self.cap = m, cap

    def forward(self, ids, labels, prefix=0):
        """prefix > 0: ids[:, :prefix] only feed a frozen KV cache (no grad); labels cover ids[:, prefix + 1:]."""
        base = self.m.base_model.model  # PeftModel -> LoraModel -> Gemma4ForConditionalGeneration
        lm = base.model.language_model
        if prefix:
            from .prefix import suffix_hidden
            h = suffix_hidden(lm, ids, prefix)
        else:
            h = lm(input_ids=ids, use_cache=False).last_hidden_state[0]
        return MaskedCE.apply(h[:-1], base.get_output_embeddings().weight, labels, self.cap)


def wrap_fsdp(m, smoke):
    """One FSDP unit per decoder layer, the vision tower in its own (never gathered) unit, the rest in the root."""
    lm = next(mm for n, mm in m.named_modules() if n.endswith("language_model") and "vision" not in n)
    layer_cls = type(lm.layers[0])
    vision = [mm for n, mm in m.named_modules() if n.endswith("vision_tower") or n.endswith("embed_vision")]
    types = tuple({layer_cls} | {type(v) for v in vision})
    cap = getattr(m.config.get_text_config(), "final_logit_softcapping", None)
    fm = FSDP(LossWrap(m, cap), auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=lambda x: isinstance(x, types)),
              sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True,
              device_id=torch.device("cpu") if smoke else torch.cuda.current_device(), limit_all_gathers=True,
              sync_module_states=False)
    return fm, layer_cls


def lora_modules(m):
    return sum(1 for n, _ in m.named_parameters() if ".lora_A." in n and re.match(LORA_RE, n.split(".lora_A.")[0]))
