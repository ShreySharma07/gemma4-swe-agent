"""Everything after the env step, inside the scorer venv: python -m trainkit.main (cwd /kaggle/working)."""
import glob
import json
import shutil

from .config import DATA_FILE, INPUT, OUT_DIR, SMOKE_CONFIG, WORK, TrainSettings
from .launcher import launch, rank_tails


def model_path(smoke):
    if smoke:  # tiny random Gemma-4 from the real config (same architecture, 6 layers)
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText
        real = json.loads(SMOKE_CONFIG.read_text())
        tc, n = real["text_config"], 6
        tc.update(hidden_size=64, intermediate_size=128, num_hidden_layers=n, num_attention_heads=4,
                  num_key_value_heads=2, num_global_key_value_heads=1, head_dim=16, global_head_dim=32, vocab_size=512,
                  vocab_size_per_layer_input=512, sliding_window=16,
                  layer_types=["sliding_attention"] * (n - 1) + ["full_attention"])
        real["vision_config"].update(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                     num_key_value_heads=2, head_dim=16, global_head_dim=16)
        torch.manual_seed(0)
        cfg = AutoConfig.for_model(**{k: v for k, v in real.items() if k != "transformers_version"})
        path = WORK / "tiny_model"
        AutoModelForImageTextToText.from_config(cfg).save_pretrained(path)
        return path
    cfgs = sorted(p for p in glob.glob(f"{INPUT}/**/config.json", recursive=True)
                  if "31b" in p.lower() and "unquantized" in p.lower())
    assert cfgs, "attach model google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"
    return cfgs[0].rsplit("/", 1)[0]


def data_path():
    if DATA_FILE.exists():
        return DATA_FILE
    found = sorted(glob.glob(f"{INPUT}/**/train_*.npz", recursive=True))
    assert found, "no training data: embed it (notebook inputs cell) or attach a dataset with train_*.npz"
    return found[0]


def main():
    s = TrainSettings.load()
    mp, dp = model_path(s.smoke), data_path()
    print("model", mp, "| data", dp, "| settings", json.dumps(s.__dict__), flush=True)
    rc = launch(mp, dp, 2 if s.smoke else 4, timeout_s=int(max(0.2, s.hours_left() + 0.3) * 3600))
    report = {"rc": rc, "settings": s.__dict__}
    mfile = OUT_DIR / "metrics.jsonl"
    rows = [json.loads(l) for l in open(mfile)] if mfile.exists() else []
    report["val"] = [r for r in rows if r.get("event") == "val"]
    report["exports"] = [r for r in rows if r.get("event") == "export"]
    steps = [r for r in rows if r.get("event") == "step"]
    report["steps"] = {"n": len(steps), "first": steps[:3], "last": steps[-3:]}
    if rc != 0:
        report["rank_tails"] = rank_tails()
    final = OUT_DIR / "adapter_final"
    if final.exists():  # ready to drop into a submission: adapters/<name>/
        dest = WORK / "adapters" / s.name
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(final, dest)
        report["adapter"] = str(dest)
    (WORK / "train_report.json").write_text(json.dumps(report, indent=1, default=str))
    print("TRAIN_REPORT", json.dumps({k: report[k] for k in ("rc", "val", "exports") if k in report}), flush=True)


if __name__ == "__main__":
    main()
