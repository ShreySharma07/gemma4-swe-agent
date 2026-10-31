"""Export: the (replicated) LoRA weights -> PEFT adapter in the format vLLM served in p0."""
import json
import os

import torch

from .config import TARGETS


def export_adapter(fm, layer_cls, out_dir, rank, r, alpha, base_path):
    """LoRA tensors are FSDP-ignored and identical on every rank (fix_lora), so rank 0 writes them directly. Keys are
    base_model.model.<checkpoint module>.lora_{A,B}.weight, e.g. ...model.language_model.layers.N.self_attn.q_proj."""
    from safetensors.torch import save_file
    if rank != 0:
        return None
    tensors = {}
    for n, p in fm.named_parameters():
        if "lora_" in n:
            key = n.replace("_fsdp_wrapped_module.", "").removeprefix("m.").replace(".default.", ".")
            tensors[key] = p.detach().to("cpu", torch.bfloat16).clone()
    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"))
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": alpha, "lora_dropout": 0.0,
                   "bias": "none", "fan_in_fan_out": False, "target_modules": list(TARGETS),
                   "base_model_name_or_path": base_path, "inference_mode": True}, f, indent=1)
    mods = {k.removeprefix("base_model.model.").rsplit(".lora_", 1)[0] for k in tensors}
    return {"dir": out_dir, "n_modules": len(mods), "n_tensors": len(tensors), "key_example": next(iter(tensors)),
            "shapes_ok": all(v.ndim == 2 and min(v.shape) == r for v in tensors.values()),
            "mb": round(os.path.getsize(os.path.join(out_dir, "adapter_model.safetensors")) / 1e6, 1)}
