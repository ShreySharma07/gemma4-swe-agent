"""Everything after the env step, inside the scorer venv: python -m trainkit.main (cwd /kaggle/working)."""
import glob
import json
import shutil

from .config import DATA_FILE, INPUT, OUT_DIR, SMOKE_CONFIG, WORK, TrainSettings
from .launcher import launch, rank_tails


def real_model_path(model):
    """The attached bf16 QAT checkpoint of `model` (31b | e4b); the *-assistant variants are not the chat model."""
    cfgs = sorted(p for p in glob.glob(f"{INPUT}/**/config.json", recursive=True)
                  if f"{model}-" in p.lower() and "unquantized" in p.lower() and "assistant" not in p.lower())
    return cfgs[0].rsplit("/", 1)[0] if cfgs else None


def tiny_config(real):
    """Shrink a real Gemma-4 config to a 6-layer random model with the same architecture features (KV sharing and
    per-layer inputs when the real model has them, as E4B does)."""
    tc, n = real["text_config"], 6
    shared = 2 if tc.get("num_kv_shared_layers") else 0  # E4B: the last layers reuse K/V of the last full/sliding layer
    tc.update(hidden_size=64, intermediate_size=128, num_hidden_layers=n, num_attention_heads=4, num_key_value_heads=2,
              head_dim=16, global_head_dim=32, vocab_size=512, vocab_size_per_layer_input=512, sliding_window=16,
              num_kv_shared_layers=shared,
              layer_types=(["sliding_attention"] * 3 + ["full_attention", "sliding_attention", "full_attention"]) if shared
              else ["sliding_attention"] * (n - 1) + ["full_attention"])
    if tc.get("num_global_key_value_heads"):
        tc["num_global_key_value_heads"] = 1
    if tc.get("hidden_size_per_layer_input"):
        tc["hidden_size_per_layer_input"] = 8
    real["vision_config"].update(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                 num_key_value_heads=2, head_dim=16, global_head_dim=16)
    if real.get("audio_config"):
        real["audio_config"].update(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, output_proj_dims=32,
                                    subsampling_conv_channels=[8, 4])
    return real


def model_path(smoke, model="31b"):
    if smoke:  # tiny random Gemma-4 from the real config (same architecture, 6 layers)
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText
        real = tiny_config(json.loads(SMOKE_CONFIG.read_text()))
        attached = real_model_path(model)  # smoke kernels may attach the real model only to check the handle
        print("real checkpoint attached:", attached, flush=True)
        torch.manual_seed(0)

        def build(cfg_dict):
            cfg = AutoConfig.for_model(**{k: v for k, v in cfg_dict.items() if k != "transformers_version"})
            return AutoModelForImageTextToText.from_config(cfg)
        try:
            tiny = build(real)
        except Exception as e:  # noqa: BLE001  the shrunk audio tower may violate its own constraints
            print("tiny model with audio failed:", type(e).__name__, str(e)[:300], "-> retry without the audio tower", flush=True)
            real["audio_config"] = None
            tiny = build(real)
        path = WORK / "tiny_model"
        tiny.save_pretrained(path)
        tc = tiny.config.get_text_config()
        print("tiny text config:", {k: getattr(tc, k, None) for k in ("num_kv_shared_layers", "hidden_size_per_layer_input",
                                                                      "num_hidden_layers", "layer_types")}, flush=True)
        return path
    found = real_model_path(model)
    assert found, f"attach model google/gemma-4/Transformers/gemma-4-{model}-it-qat-q4_0-unquantized/2"
    return found


def data_path():
    if DATA_FILE.exists():
        return DATA_FILE
    found = sorted(glob.glob(f"{INPUT}/**/train_*.npz", recursive=True)) + sorted(glob.glob(f"{INPUT}/**/rs[234].npz", recursive=True))
    assert found, "no training data: embed it (notebook inputs cell) or attach a dataset with train_*.npz / rs2.npz"
    return found[0]


def main():
    s = TrainSettings.load()
    mp, dp = model_path(s.smoke, s.model), ("none" if s.prefix_probe or s.lc_probe else data_path())
    print("model", mp, "| data", dp, "| settings", json.dumps(s.__dict__), flush=True)
    rc = launch(mp, dp, 2 if s.smoke else 4, timeout_s=int(max(0.2, s.hours_left() + 0.3) * 3600))
    report = {"rc": rc, "settings": s.__dict__}
    mfile = OUT_DIR / "metrics.jsonl"
    rows = [json.loads(l) for l in open(mfile)] if mfile.exists() else []
    report["val"] = [r for r in rows if r.get("event") == "val"]
    report["val_sets"] = [r for r in rows if r.get("event") == "val_set"]
    report["exports"] = [r for r in rows if r.get("event") == "export"]
    steps = [r for r in rows if r.get("event") == "step"]
    report["steps"] = {"n": len(steps), "first": steps[:3], "last": steps[-3:]}
    for probe in ("prefix_probe", "lc_probe", "vablate"):
        pp = OUT_DIR / f"{probe}.json"
        if pp.exists():
            report[probe] = json.loads(pp.read_text())
    if rc != 0:
        report["rank_tails"] = rank_tails()
    final = OUT_DIR / "adapter_final"
    if final.exists():  # ready to drop into a submission: adapters/<name>/
        dest = WORK / "adapters" / s.name
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(final, dest)
        report["adapter"] = str(dest)
    (WORK / "train_report.json").write_text(json.dumps(report, indent=1, default=str))
    print("TRAIN_REPORT", json.dumps({k: report[k] for k in ("rc", "val", "val_sets", "exports", "prefix_probe", "lc_probe", "vablate") if k in report}), flush=True)


if __name__ == "__main__":
    main()
