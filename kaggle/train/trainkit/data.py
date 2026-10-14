"""TrainData: tokenized windows (posttrain/tokenize_windows.py) -> per-rank schedules of (input_ids, labels)."""
import random

import numpy as np
import torch

IGNORE = -100


class TrainData:
    def __init__(self, path, max_windows=None, vocab=None, seed=0):
        d = np.load(path)
        self.ids, self.train, self.starts = d["ids"], d["train"], d["starts"]
        self.task, self.val = d["task"], d["val"].astype(bool)
        n = len(self.starts) - 1
        w = list(range(n))
        random.Random(seed).shuffle(w)
        self.train_w = [i for i in w if not self.val[i]][:max_windows]
        self.val_w = [i for i in w if self.val[i]]
        self.vocab = vocab  # smoke: fold ids into the tiny model's vocabulary

    def length(self, w):
        return int(self.starts[w + 1] - self.starts[w])

    def example(self, w, device):
        """-> (ids [1, T], labels [T-1]): labels[t] = ids[t+1] if that token is model-written, else IGNORE."""
        s, e = self.starts[w], self.starts[w + 1]
        ids = torch.from_numpy(self.ids[s:e].astype(np.int64))
        if self.vocab:
            ids = ids % self.vocab
        lab = ids[1:].clone()
        lab[torch.from_numpy(self.train[s + 1:e] == 0)] = IGNORE
        return ids.unsqueeze(0).to(device), lab.to(device)

    def schedule(self, windows, rank, world, epochs=1.0):
        """Equal-length per-rank lists (every rank must run the same number of steps under FSDP). Windows are dealt
        in length order so the ranks of one step get similar lengths (no rank waits long for another)."""
        n_total = int(len(windows) * epochs)
        reps = [windows[i % len(windows)] for i in range(n_total)]
        per_step = []
        for i in range(0, len(reps) - world + 1, world):
            group = sorted(reps[i:i + world], key=self.length)
            per_step.append(group)
        return [g[rank] for g in per_step]

    def stats(self):
        lens = [self.length(w) for w in self.train_w]
        return {"train_windows": len(self.train_w), "val_windows": len(self.val_w),
                "train_tokens": int(sum(lens)), "max_len": max(lens) if lens else 0}
