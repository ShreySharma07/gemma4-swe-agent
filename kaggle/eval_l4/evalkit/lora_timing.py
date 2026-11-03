"""
LT1 LoRA serving-cost probe (GPU). Pre-registered in docs/experiments.md (Nov 3). v18 (v14 + rs2_lora) timed out on
the LB while v14 with the same caps finished; G1 timed the adapter at concurrency 6, where batching hides per-token
LoRA cost. Here: the scorer's exact server (ScorerServer: TP 4, max_loras 8, max_lora_rank 128, prefix caching as
served), ONE request at a time, through /v1/completions with raw token ids from real RS2 val episodes:
  arms     base | rs2_lora (T1, 410 modules) | rs2_attn (T1's q/k/v/o LoRA only, 230 modules: what an
           attention-only adapter would cost; behaviour not evaluated)
  lengths  prompt prefixes of LENGTHS tokens of the longest val episode, a fresh nonce token at position 1 per call
           so no call hits another's prefix cache
  per call prefill (max_tokens 1), then decode DECODE tokens on the now-cached prompt (ignore_eos)
Arms alternate inside every (length, repeat), so drift hits them equally.
--dry (CPU): episodes, both adapters built and discovered, no server.
Output: /kaggle/working/lora_timing_report.json. Run inside the venv: python -m evalkit.lora_timing [--dry]
"""
import json
import random
import statistics
import sys
import time
import urllib.request

from .config import WORK, Settings
from .inputs import Inputs
from .kvdup import read, write
from .lora_probe import episodes
from .server import ScorerServer, set_scorer_env
from .submission import Submission
from .util import log

LENGTHS = (2048, 8192, 16384, 24000)
DECODE, REPEATS = 256, 3
ATTN = (".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.")


def attn_only(src, dst):
    """rs2_lora restricted to its attention projections (stdlib safetensors I/O from kvdup)."""
    header, data = read(src / "adapter_model.safetensors")
    meta = header.pop("__metadata__", None)
    keep = {k: (v["dtype"], v["shape"], data[v["data_offsets"][0]:v["data_offsets"][1]])
            for k, v in header.items() if any(a in k for a in ATTN)}
    dst.mkdir(parents=True, exist_ok=True)
    write(dst / "adapter_model.safetensors", keep, meta)
    cfg = json.loads((src / "adapter_config.json").read_text())
    cfg["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    return len({k.rsplit(".lora_", 1)[0] for k in keep})


def post(base_url, body):
    req = urllib.request.Request(f"{base_url}/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    out = json.load(urllib.request.urlopen(req, timeout=900))
    return time.perf_counter() - t, out["usage"]


def main(dry=False):
    s = Settings.load()
    set_scorer_env()
    eps, src = episodes()
    ep = max((ids for ids, _ in eps), key=len)
    sub = Submission().unpack()
    n_attn = attn_only(sub.dir / "adapters" / "rs2_lora", sub.dir / "adapters" / "rs2_attn")
    adapters = sub.adapters()
    names = sorted(getattr(adapters, "adapters", {}) or {})
    report = {"rs2": src, "episode_len": len(ep), "lengths": [L for L in LENGTHS if L < len(ep)], "decode": DECODE,
              "repeats": REPEATS, "adapters": names, "attn_modules": n_attn, "calls": []}
    log("LT1", json.dumps({k: v for k, v in report.items() if k != "calls"}))
    assert names == ["rs2_attn", "rs2_lora"], names
    if dry:
        (WORK / "lora_timing_report.json").write_text(json.dumps(report, indent=1))
        return
    server = ScorerServer(Inputs.locate(False).model_path, adapters).start()
    base_url = server.server.base_url
    try:
        served = [m["id"] for m in json.load(urllib.request.urlopen(f"{base_url}/models"))["data"]]
        arms = {"base": next(m for m in served if m not in names), "rs2_lora": "rs2_lora", "rs2_attn": "rs2_attn"}
        log("served:", served)
        rng = random.Random(0)
        post(base_url, {"model": arms["base"], "prompt": ep[:512], "max_tokens": 8})  # warm-up (graphs, first LoRA)
        for a in ("rs2_lora", "rs2_attn"):
            post(base_url, {"model": arms[a], "prompt": ep[:512], "max_tokens": 8})
        for L in report["lengths"]:
            for rep in range(REPEATS):
                for arm in rng.sample(list(arms), 3):
                    prompt = [ep[0], rng.randrange(1000, 200000)] + ep[1:L - 1]  # nonce: uncached prefix
                    t_pre, _ = post(base_url, {"model": arms[arm], "prompt": prompt, "max_tokens": 1, "temperature": 0})
                    t_dec, u = post(base_url, {"model": arms[arm], "prompt": prompt, "max_tokens": DECODE,
                                               "min_tokens": DECODE, "ignore_eos": True, "temperature": 0})
                    row = {"arm": arm, "len": L, "rep": rep, "prefill_s": round(t_pre, 3), "decode_s": round(t_dec, 3),
                           "decode_tokens": u["completion_tokens"], "decode_tok_s": round(u["completion_tokens"] / t_dec, 2)}
                    report["calls"].append(row)
                    log("CALL", json.dumps(row))
            (WORK / "lora_timing_report.json").write_text(json.dumps(report, indent=1))
        summ = {}
        for L in report["lengths"]:
            for arm in arms:
                rows = [r for r in report["calls"] if r["len"] == L and r["arm"] == arm]
                summ[f"{arm}@{L}"] = {"prefill_s": round(statistics.median(r["prefill_s"] for r in rows), 3),
                                      "decode_tok_s": round(statistics.median(r["decode_tok_s"] for r in rows), 2)}
        for arm in ("rs2_lora", "rs2_attn"):
            summ[f"{arm}_decode_slowdown"] = {L: round(summ[f"base@{L}"]["decode_tok_s"] / summ[f"{arm}@{L}"]["decode_tok_s"], 3)
                                              for L in report["lengths"]}
            summ[f"{arm}_prefill_slowdown"] = {L: round(summ[f"{arm}@{L}"]["prefill_s"] / summ[f"base@{L}"]["prefill_s"], 3)
                                               for L in report["lengths"]}
        report["summary"] = summ
    finally:
        (WORK / "lora_timing_report.json").write_text(json.dumps(report, indent=1))
        log("LT1 REPORT", json.dumps(report.get("summary", {})))
        server.stop()
        log(f"RUN_DONE in {(time.time() - s.t0) / 3600:.2f} h")


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
