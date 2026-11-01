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

from .config import INPUT, WORK
from .inputs import Inputs
from .kvdup import kv_dup
from .util import log

ADAPTER = "rs2_lora"
NF4_VAL = {"start": 0.1011, "end": 0.0666}  # T1 (trainkit, NF4 base, HF attention path)


def episodes():
    p = sorted(INPUT.rglob("rs2.npz"))
    assert p, "attach kernel output gemma4-rftset-rs2"
    d = np.load(p[0])
    st, val = d["ep_starts"], d["val"].astype(bool)
    eps = [(d["ids"][st[i]:st[i + 1]].astype(np.int64).tolist(), d["weight"][st[i]:st[i + 1]].astype(np.float64))
           for i in range(len(st) - 1) if val[i]]
    return [(ids, w) for ids, w in eps if w[1:].sum() > 0], str(p[0])


def adapters():
    """-> {arm: dir}: the T1 export as trained (raw) and with the k_eq_v copy (kv)."""
    found = sorted(INPUT.glob(f"**/adapters/{ADAPTER}/adapter_model.safetensors"))
    assert len(found) == 1, f"expected one attached adapters/{ADAPTER}/, found {found}"
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
    report = {"rs2": src, "episodes": len(eps), "tokens": sum(lens), "max_len": lens[-1], "kv_dup": info,
              "adapters": arms, "nf4_val": NF4_VAL, "arms": {}}
    log("LP1", json.dumps({k: v for k, v in report.items() if k != "arms"}))
    if not dry:
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        model = Inputs.locate(False).model_path
        t = time.time()
        llm = LLM(model=model, tensor_parallel_size=4, max_model_len=32768, gpu_memory_utilization=0.80,
                  dtype="bfloat16", enable_lora=True, max_loras=2, max_lora_rank=16, enable_prefix_caching=False,
                  max_num_batched_tokens=1024, max_num_seqs=1)
        report["load_s"] = round(time.time() - t)
        loras = {"raw": LoRARequest("raw", 1, arms["raw"]), "kv": LoRARequest("kv", 2, arms["kv"])}
        for arm in ("base", "kv", "raw"):
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
    (WORK / "lora_probe_report.json").write_text(json.dumps(report, indent=1))
    log("LP1 REPORT", json.dumps({k: v for k, v in report.items() if k not in ("arms", "adapters")}))


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
