"""TrainData: whole episodes (rs2.npz: per-token float weights) or tokenized windows (train_*.npz: 0/1 mask) ->
length-bucketed steps of (input_ids, labels, weights) with each step's total weight known in advance."""
import random

import numpy as np
import torch

IGNORE = -100


class TrainData:
    def __init__(self, path, max_windows=None, vocab=None, seed=0, max_len=None, corr="keep", corr_share=0.0):
        d = np.load(path)
        self.ids = d["ids"]
        if "weight" in d:  # rs2.npz (kaggle/rftset/rs2_core.py): weight[t] = loss weight of token t, 0 = context
            self.w, self.starts, val = d["weight"], d["ep_starts"], d["val"].astype(bool)
        else:  # windows (posttrain/tokenize_windows.py): train[t] = 1 if token t is model-written
            self.w, self.starts, val = d["train"], d["starts"], d["val"].astype(bool)
        n = len(self.starts) - 1
        # rs4.npz (kaggle/rftset/rs4_core.py): corrected[i] = 1 for episodes built from failed runs (T2)
        self.corr = d["corrected"].astype(bool) if "corrected" in d else np.zeros(n, dtype=bool)
        self.scale = np.ones(n, dtype=np.float64)
        self.dropped_long = [i for i in range(n) if max_len and self.length(i) > max_len]
        drop = set(self.dropped_long)
        w = [i for i in range(n) if i not in drop and self.weight_sum(i) > 0]
        random.Random(seed).shuffle(w)
        self.train_w = [i for i in w if not val[i] and not (corr == "exclude" and self.corr[i])][:max_windows]
        self.val_w = [i for i in w if val[i]]
        self.val_sets = ({"onpolicy": [i for i in self.val_w if not self.corr[i]], "corrected": [i for i in self.val_w if self.corr[i]]}
                         if any(self.corr[i] for i in self.val_w) else {})
        wc = sum(self.weight_sum(i) for i in self.train_w if self.corr[i])
        wt = sum(self.weight_sum(i) for i in self.train_w)
        self.corr_share_raw = wc / wt if wt else 0.0
        if corr_share and wc and wt > wc:  # scale corrected episodes so they carry corr_share of the train weight
            self.scale[[i for i in self.train_w if self.corr[i]]] = corr_share * (wt - wc) / ((1 - corr_share) * wc)
        self.vocab = vocab  # smoke: fold ids into the tiny model's vocabulary
        self.seed = seed

    def length(self, w):
        return int(self.starts[w + 1] - self.starts[w])

    def weight_sum(self, w):
        """Total loss weight of window w's labels (tokens 1..T-1)."""
        return float(self.w[self.starts[w] + 1:self.starts[w + 1]].astype(np.float64).sum() * self.scale[w])

    def example(self, w, device):
        """-> (ids [1, T], labels [T-1], weights [T-1] float32): labels[t] predicts ids[t+1] with weight w[t+1]."""
        s, e = self.starts[w], self.starts[w + 1]
        ids = torch.from_numpy(self.ids[s:e].astype(np.int64))
        if self.vocab:
            ids = ids % self.vocab
        wts = torch.from_numpy(self.w[s + 1:e].astype(np.float32) * np.float32(self.scale[w]))
        lab = ids[1:].clone()
        lab[wts == 0] = IGNORE
        return ids.unsqueeze(0).to(device), lab.to(device), wts.to(device)

    def plan(self, windows, world, accum, epochs=1.0):
        """Every rank computes the same plan: a list of optimizer steps, each a list of `accum` micro-steps, each a list
        of `world` windows (one per rank). Per epoch: drop a random remainder, sort by length, cut into steps of
        world * accum consecutive windows (ranks of a step get similar lengths), shuffle the steps."""
        steps, per = [], world * accum
        n_ep = max(1, int(np.ceil(epochs)))
        for ep in range(n_ep):
            rng = random.Random(self.seed * 1000 + ep)
            ws = list(windows)
            rng.shuffle(ws)
            ws = sorted(ws[:len(ws) - len(ws) % per], key=self.length)
            ep_steps = [[ws[a + i * world:a + (i + 1) * world] for i in range(accum)] for a in range(0, len(ws), per)]
            rng.shuffle(ep_steps)
            if ep == n_ep - 1 and epochs < n_ep:
                ep_steps = ep_steps[:round(len(ep_steps) * (epochs - ep))]
            steps += ep_steps
        return steps

    def step_norm(self, step):
        """The step's total label weight over all ranks and micro-steps: the loss divisor that makes every trained
        token count the same."""
        return max(1e-6, sum(self.weight_sum(w) for micro in step for w in micro))

    def stats(self):
        lens = [self.length(w) for w in self.train_w]
        return {"train_windows": len(self.train_w), "val_windows": len(self.val_w), "dropped_long": len(self.dropped_long),
                "train_tokens": int(sum(lens)), "max_len": max(lens) if lens else 0,
                "train_weight": round(sum(self.weight_sum(w) for w in self.train_w), 1),
                "train_corrected": int(sum(self.corr[w] for w in self.train_w)), "val_corrected": len(self.val_sets.get("corrected", [])),
                "corr_share_raw": round(self.corr_share_raw, 4), "corr_scale": round(float(self.scale.max()), 3),
                "corr_share": round(sum(self.weight_sum(w) for w in self.train_w if self.corr[w]) /
                                    max(1e-6, sum(self.weight_sum(w) for w in self.train_w)), 4)}
