"""
LP2: HF-side ablation of the global layers' V delta (pre-registered in docs/experiments.md, Nov 1). Gemma-4's 10 global
layers have no v_proj: HF computes V = k_proj(x), so a k_proj LoRA moves K and V. LP1 found vLLM scores the RS2 val
episodes slightly better with the V delta absent (raw) than present (kv copy). Here, on the training stack (NF4 + HF +
LC1 fixes), the same comparison:
  full     T1's adapter as trained (K and V get the k_proj delta): must reproduce T1's end val 0.0666
  v_base   global layers' V = base k_proj(x) (LoRA skipped for V only; K keeps it): the vLLM 'raw' forward
  base     every lora_B zeroed: must reproduce T1's start val 0.1011
Weighted val loss pooled and per episode (as Trainer.validate). If v_base < full here too, the vLLM copy reproduces
HF and the V delta itself is mildly harmful on held-out episodes; if v_base > full, vLLM's kv arm is not HF's forward.
Smoke (CPU, tiny random model, random LoRA): the V patch is exact (full == v_base when the global k_proj delta is
zero) and changes the loss when it is not.
"""
import glob
import json

import torch
import torch.distributed as dist

from .config import INPUT, OUT_DIR

ADAPTER = "rs2_lora"


def norm_name(n):
    """FSDP-wrapped name -> export / checkpoint key prefix (as export.export_adapter)."""
    return n.replace("_fsdp_wrapped_module.", "").removeprefix("m.").replace(".default.", ".")


def global_attn(fm):
    """{key prefix of self_attn: (attention module, its k_proj LoRA module)} for every text attention layer without
    v_proj (the k_eq_v global layers)."""
    return {norm_name(n): (mod, mod.k_proj) for n, mod in fm.named_modules()
            if type(mod).__name__ == "Gemma4TextAttention" and mod.v_proj is None}


def set_v_base(pairs, on):
    """on: V = k_proj.base_layer(x) (no LoRA); off: back to HF's V = k_proj(x). object.__setattr__, so the base layer
    is not registered as a second submodule (FSDP already holds its weights in this layer's flat parameter)."""
    for attn, k in pairs.values():
        object.__setattr__(attn, "v_proj", k.base_layer if on else None)


def lora_named(tr):
    return {norm_name(n): p for n, p in tr.fm.named_parameters() if "lora_" in n}


def load_adapter(tr, smoke):
    """T1's export into the replicated LoRA tensors (smoke: random lora_B, std 0.05)."""
    named = lora_named(tr)
    if smoke:
        g = torch.Generator().manual_seed(1)
        for n, p in named.items():
            if "lora_B" in n:
                p.data.copy_((torch.randn(p.shape, generator=g) * 0.05).to(p.dtype))
        return {"source": "random lora_B (smoke)", "loaded": len(named)}
    from safetensors.torch import load_file
    found = sorted(glob.glob(f"{INPUT}/**/adapters/{ADAPTER}/adapter_model.safetensors", recursive=True))
    assert len(found) == 1, f"expected one attached adapters/{ADAPTER}/, found {found}"
    sd = load_file(found[0])
    missing = [n for n in named if n not in sd]
    extra = [k for k in sd if k not in named]
    assert not missing and all(".v_proj." in k for k in extra), (missing[:3], extra[:3])
    for n, p in named.items():
        assert sd[n].shape == p.shape, (n, sd[n].shape, p.shape)
        p.data.copy_(sd[n].to(p.device, p.dtype))
    return {"source": found[0], "loaded": len(named), "ignored_extra": len(extra)}


def per_episode(tr, data, tag):
    """Weighted val loss pooled + per episode; ranks split the episodes, equal counts per rank (FSDP)."""
    tr.fm.eval()
    mine = data.val_w[tr.rank::tr.world]
    n_real = len(mine)
    mine = mine + mine[:1] * (len(data.val_w[::tr.world]) - n_real)
    rows = torch.zeros(len(data.val_w), 2, dtype=torch.float64, device=tr.dev)
    with torch.no_grad():
        for k, w in enumerate(mine):
            ids, lab, wts = data.example(w, tr.dev)
            loss = float(tr.loss(ids, lab, wts, 1.0))  # sum_t w_t * ce_t
            if k < n_real:
                rows[data.val_w.index(w)] = torch.tensor([loss, float(wts.sum())], dtype=torch.float64)
    dist.all_reduce(rows)
    tot = rows.sum(0)
    out = {"loss": round(float(tot[0] / tot[1]), 5), "per_episode": [round(float(a / b), 5) for a, b in rows.tolist()]}
    tr.log(event="vablate", tag=tag, loss=out["loss"], windows=len(data.val_w))
    return out


def probe(tr, data, smoke):
    pairs = global_attn(tr.fm)
    report = {"global_layers": len(pairs), "adapter": load_adapter(tr, smoke), "arms": {}}
    assert pairs, "no k_eq_v layer found (expected v_proj None on global layers)"
    named = lora_named(tr)
    arms = report["arms"]
    if smoke:  # exactness: with the global k_proj delta zeroed, V is the base output either way
        glob_k = [f"{pre}.k_proj.lora_B.weight" for pre in pairs]
        kB = {n: named[n].data.clone() for n in glob_k}
        for n in glob_k:
            named[n].data.zero_()
        arms["kzero_full"] = per_episode(tr, data, "kzero_full")
        set_v_base(pairs, True)
        arms["kzero_v_base"] = per_episode(tr, data, "kzero_v_base")
        set_v_base(pairs, False)
        for n in glob_k:
            named[n].data.copy_(kB[n])
        report["smoke_exact_max_diff"] = max(abs(a - b) for a, b in zip(arms["kzero_full"]["per_episode"],
                                                                           arms["kzero_v_base"]["per_episode"]))
        report["smoke_global_k_zeroed"] = len(glob_k)
    arms["full"] = per_episode(tr, data, "full")
    set_v_base(pairs, True)
    arms["v_base"] = per_episode(tr, data, "v_base")
    set_v_base(pairs, False)
    arms["full_again"] = per_episode(tr, data, "full_again")  # the patch is undone cleanly
    for n, p in named.items():
        if "lora_B" in n:
            p.data.zero_()
    arms["base"] = per_episode(tr, data, "base")
    f, v = arms["full"]["per_episode"], arms["v_base"]["per_episode"]
    report["v_base_minus_full"] = round(arms["v_base"]["loss"] - arms["full"]["loss"], 5)
    report["v_base_better_episodes"] = sum(a < b for a, b in zip(v, f))
    report["full_again_max_diff"] = max(abs(a - b) for a, b in zip(f, arms["full_again"]["per_episode"]))
    if tr.rank == 0:
        (OUT_DIR / "vablate.json").write_text(json.dumps(report, indent=1))
    tr.log(event="vablate_done", **{k: v for k, v in report.items() if k not in ("arms", "adapter")})
