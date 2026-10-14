"""
Rendered windows (posttrain/render_atif.py JSONL) -> one compact training file for kaggle/train/trainkit.

Each window's text is tokenized exactly as rendered (no extra special tokens; the chat template already starts with
<bos>), and a token is trained iff its first character lies inside one of the window's `train_spans` (model-written
tool calls / text; tool responses and prompts are context only). Windows are capped at MAX_TOKENS.
Validation = VAL_FRAC of TASKS (not windows), so no task is on both sides.
Output .npz: ids int32[N], train uint8[N], starts int64[W+1] (window w = ids[starts[w]:starts[w+1]]),
task (str[W]), val uint8[W]; plus a JSON summary next to it.
Usage: python posttrain/tokenize_windows.py WINDOWS.jsonl OUT.npz [--tokenizer DIR] [--val-frac 0.05]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

MAX_TOKENS = 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windows")
    ap.add_argument("out")
    ap.add_argument("--tokenizer", default="/tmp/g4tok")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = [json.loads(l) for l in open(a.windows)]
    tasks = sorted({r["id"].rsplit("__", 1)[0] for r in rows})
    rng = random.Random(a.seed)
    val_tasks = set(rng.sample(tasks, max(1, round(len(tasks) * a.val_frac)))) if a.val_frac > 0 else set()
    ids, train, starts, task, val = [], [], [0], [], []
    for r in rows:
        enc = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)
        spans = sorted(r["train_spans"])
        mask, j = [], 0
        for s, _ in enc["offset_mapping"]:
            while j < len(spans) and spans[j][1] <= s:
                j += 1
            mask.append(1 if j < len(spans) and spans[j][0] <= s < spans[j][1] else 0)
        x = enc["input_ids"][:MAX_TOKENS]
        ids += x
        train += mask[:len(x)]
        starts.append(len(ids))
        t = r["id"].rsplit("__", 1)[0]
        task.append(t)
        val.append(t in val_tasks)
    out = Path(a.out)
    np.savez_compressed(out, ids=np.array(ids, np.int32), train=np.array(train, np.uint8),
                        starts=np.array(starts, np.int64), task=np.array(task), val=np.array(val, np.uint8))
    v = np.array(val, bool)
    lens = np.diff(starts)
    summary = {"windows": len(rows), "tasks": len(tasks), "val_tasks": len(val_tasks), "val_windows": int(v.sum()),
               "tokens": int(lens.sum()), "train_tokens": int(sum(train)), "max_len": int(lens.max()),
               "trained_share": round(sum(train) / len(ids), 3)}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
