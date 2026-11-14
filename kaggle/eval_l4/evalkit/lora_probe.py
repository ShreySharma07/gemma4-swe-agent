"""
LP1 adapter probe on the SERVED base (GPU): weighted loss of the RS2 val episodes under vLLM 0.19.1 on the W4A16 model,
for base / rs2_lora as trained (raw) / rs2_lora with the k_eq_v v_proj copy (kv, evalkit/kvdup.py). Pre-registered in
docs/experiments.md (Nov 1). Answers two questions before any rollout:
  1. k_eq_v: does the v_proj copy matter on the served path (kv vs raw)?
  2. NF4 -> W4A16: how much of T1's val gain (measured on the NF4 training base, 0.1011 -> 0.0666) survives on the
     base the scorer serves?
Loss = sum_t w_t * -log p(id_t | ids_<t) / sum_t w_t over all val episodes (as trainkit.worker.validate), from
prompt_logprobs. Its own vLLM engine (offline LLM, TP=4): prefix caching off (cached tokens get no prompt logprobs),
1024-token prefill chunks (full-vocab fp32 logits per chunk), so the scorer's server settings are not used here.
--dry (CPU): load episodes + adapters, write the report without a model.
Output: /kaggle/working/lora_probe_report.json. Run inside the venv: python -m evalkit.lora_probe [--dry]
"""
import json
import sys
import time

import numpy as np

from .config import INPUT, WORK, Settings
from .inputs import Inputs
from .kvdup import kv_dup
from .util import log

CFG = Settings.load()
ADAPTERS, DATA = CFG.lora_adapter.split(","), CFG.lora_data
ADAPTER = ADAPTERS[0]
# end-of-training on-policy val on the NF4 base (trainkit): T1, T2a, T2b
NF4_VAL = {"start": 0.1011, "end": {"rs2_lora": 0.0666, "t2a_lora": 0.0657, "t2b_lora": 0.0695}.get(ADAPTER)}
KV = ADAPTER == "rs2_lora"  # the k_eq_v copy arm: LP1 showed it does not help, later adapters ship raw only


def episodes():
    p = sorted(INPUT.rglob(f"{DATA}.npz"))
    assert p, f"attach kernel output gemma4-rftset-{DATA}"
    d = np.load(p[0])
    st, val = d["ep_starts"], d["val"].astype(bool)
    if "corrected" in d.files:  # rs4: the on-policy val episodes only (the set NF4_VAL was measured on)
        val &= ~d["corrected"].astype(bool)
    eps = [(d["ids"][st[i]:st[i + 1]].astype(np.int64).tolist(), d["weight"][st[i]:st[i + 1]].astype(np.float64))
           for i in range(len(st) - 1) if val[i]]
    return [(ids, w) for ids, w in eps if w[1:].sum() > 0], str(p[0])


def adapters():
    """-> {arm: dir}: the T1 export as trained (raw) and with the k_eq_v copy (kv)."""
    found = sorted(INPUT.glob(f"**/adapters/{ADAPTER}/adapter_model.safetensors"))
    assert len(found) == 1, f"expected one attached adapters/{ADAPTER}/, found {found}"
    if not KV:  # T2: one raw arm per adapter ("raw" for the first, the adapter's name for the others)
        arms = {"raw": str(found[0].parent)}
        for n in ADAPTERS[1:]:
            f = sorted(INPUT.glob(f"**/adapters/{n}/adapter_model.safetensors"))
            assert len(f) == 1, f"expected one attached adapters/{n}/, found {f}"
            arms[n] = str(f[0].parent)
        return arms, {}
    kv = WORK / "adapters_kv" / ADAPTER
    info = kv_dup(found[0].parent, kv)
    log("kv_dup:", json.dumps(info))
    return {"raw": str(found[0].parent), "kv": str(kv)}, info


def score(llm, eps, lora):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1, detokenize=False)  # 1: the actual token is always included
    num = den = 0.0
    per = []
    t = time.time()
    for ids, w in eps:
        out = llm.generate([{"prompt_token_ids": ids}], sp, lora_request=lora, use_tqdm=False)[0]
        lp = out.prompt_logprobs
        nll = np.array([-lp[k][ids[k]].logprob for k in range(1, len(ids))])
        a, b = float((w[1:] * nll).sum()), float(w[1:].sum())
        num, den = num + a, den + b
        per.append(round(a / b, 5))
    toks = sum(len(i) for i, _ in eps)
    return {"loss": round(num / den, 5), "per_episode": per, "tokens": toks,
            "tok_per_s": round(toks / (time.time() - t), 1)}


def main(dry=False):
    eps, src = episodes()
    arms, info = adapters()
    lens = sorted(len(i) for i, _ in eps)
    report = {"adapter": ADAPTER, "rs2": src, "episodes": len(eps), "tokens": sum(lens), "max_len": lens[-1], "kv_dup": info,
              "adapters": arms, "nf4_val": NF4_VAL, "arms": {}}
    log("LP1", json.dumps({k: v for k, v in report.items() if k != "arms"}))
    if not dry:
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        model = Inputs.locate(False).model_path
        t = time.time()
        llm = LLM(model=model, tensor_parallel_size=4, max_model_len=32768, gpu_memory_utilization=0.80,
                  dtype="bfloat16", enable_lora=True, max_loras=max(2, len(arms)), max_lora_rank=16, enable_prefix_caching=False,
                  max_num_batched_tokens=1024, max_num_seqs=1)
        report["load_s"] = round(time.time() - t)
        loras = {a: LoRARequest(a, i + 1, d) for i, (a, d) in enumerate(arms.items())}
        for arm in ("base", "kv", "raw") if KV else ("base", *arms):
            try:
                report["arms"][arm] = score(llm, eps, loras.get(arm))
            except Exception as e:  # noqa: BLE001  keep the arms already scored
                report["arms"][arm] = {"error": f"{type(e).__name__}: {e}"[:500]}
            log(arm, json.dumps({k: v for k, v in report["arms"][arm].items() if k != "per_episode"}))
            (WORK / "lora_probe_report.json").write_text(json.dumps(report, indent=1))
        b, k, r = (report["arms"].get(a, {}).get("loss") for a in ("base", "kv", "raw"))
        if b and k:
            report["kv_gain"] = round(1 - k / b, 4)
            report["nf4_gain"] = round(1 - NF4_VAL["end"] / NF4_VAL["start"], 4)
            report["kv_episodes_better"] = sum(x < y for x, y in zip(report["arms"]["kv"]["per_episode"],
                                                                     report["arms"]["base"]["per_episode"]))
        if r and k:
            report["raw_minus_kv"] = round(r - k, 5)
        if b and r:
            report["raw_gain"] = round(1 - r / b, 4)
            report["nf4_gain"] = round(1 - NF4_VAL["end"] / NF4_VAL["start"], 4)
            report["raw_episodes_better"] = sum(x < y for x, y in zip(report["arms"]["raw"]["per_episode"],
                                                                      report["arms"]["base"]["per_episode"]))
            report["gains"] = {("raw" if a == "raw" else a): round(1 - v["loss"] / b, 4)
                               for a, v in report["arms"].items() if a != "base" and v.get("loss")}
    (WORK / "lora_probe_report.json").write_text(json.dumps(report, indent=1))
    log("LP1 REPORT", json.dumps({k: v for k, v in report.items() if k not in ("arms", "adapters")}))


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
