"""Model: NF4 Gemma-4-31B + LoRA r16 + FSDP with the long-context fixes (fix_*.py), and the weighted streamed loss."""
import re
from functools import partial

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from . import fix_attention, fix_head, fix_lora, fix_mlp, fix_norm
from .config import LC, TARGETS
from .data import IGNORE
from .fix_attention import NO_MASKS
from .fix_head import StreamCE

LORA_RE = r".*language_model.*\.(" + "|".join(TARGETS) + r")$"


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
    # no enable_input_require_grads: non-reentrant checkpointing does not need it, and the embedding is frozen
    m = get_peft_model(m, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none", target_modules=LORA_RE,
                                     task_type="CAUSAL_LM"))
    for n, p in m.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(dtype)  # FSDP flat params need one dtype per unit
    return m


class LossWrap(nn.Module):
    """FSDP root. The tied embedding / lm_head is in host RAM (fix_head), so the root holds only norms and buffers."""

    def __init__(self, m, cap, head):
        super().__init__()
        self.m, self.cap = m, cap
        self.head = [head]  # a list, so it is neither a parameter nor a buffer FSDP would move to the GPU

    def forward(self, ids, labels, wts=None, norm=None, prefix=0, hidden=False):
        """loss = sum_t wts_t * ce_t / norm over labels [T-1] (wts default: 1 where labels != IGNORE; norm default:
        the local weight sum, i.e. this sequence's mean). prefix > 0: ids[:, :prefix] only feed a frozen KV cache.
        hidden: return the final hidden states h[:-1] instead (probe)."""
        lm = self.m.base_model.model.model.language_model
        if wts is None:
            wts = (labels != IGNORE).float()
        if norm is None:
            norm = wts.sum().clamp(min=1.0)
        if prefix:
            from .prefix import suffix_hidden
            h = suffix_hidden(lm, ids, prefix)
        else:
            h = lm(input_ids=ids, attention_mask=NO_MASKS if LC["attn"] else None, use_cache=False).last_hidden_state[0]
        if hidden:
            return h[:-1]
        y = labels.clamp(min=0)
        return StreamCE.apply(h[:-1], self.head[0], y, wts, self.cap, float(norm))


def wrap_fsdp(m, smoke, mlp_tile=4096):
    """Install the long-context fixes, then one FSDP unit per decoder layer, the vision tower in its own (never
    gathered) unit, the rest in the root. LoRA tensors are FSDP-ignored (fix_lora); the head lives on the host."""
    dev = torch.device("cpu") if smoke else torch.device("cuda", torch.cuda.current_device())
    fix_attention.install(m)
    fix_mlp.install(mlp_tile)
    fix_norm.install()
    head = fix_head.move_to_host(m, dev)
    lora = fix_lora.lora_params(m)
    fix_lora.broadcast_init(lora)  # replicated tensors: every rank must start from the same init
    lm = next(mm for n, mm in m.named_modules() if n.endswith("language_model") and "vision" not in n)
    layer_cls = type(lm.layers[0])
    vision = [mm for n, mm in m.named_modules() if n.endswith("vision_tower") or n.endswith("embed_vision")]
    types = tuple({layer_cls} | {type(v) for v in vision})
    cap = getattr(m.config.get_text_config(), "final_logit_softcapping", None)
    fm = FSDP(LossWrap(m, cap, head), auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=lambda x: isinstance(x, types)),
              sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True,
              device_id=torch.device("cpu") if smoke else torch.cuda.current_device(), limit_all_gathers=True,
              sync_module_states=False, ignored_states=lora)
    return fm, layer_cls, lora


def lora_modules(m):
    return sum(1 for n, _ in m.named_parameters() if ".lora_A." in n and re.match(LORA_RE, n.split(".lora_A.")[0]))
