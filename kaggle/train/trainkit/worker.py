"""
Torchrun entry point (one process per GPU): python -m torch.distributed.run --nproc_per_node 4 -m trainkit.worker MODEL
Data parallel: each rank runs one window per micro-step; the loss is divided by the step's total label weight (all
ranks and micro-steps), LoRA grads are summed across ranks by ShardedAdamW (fix_lora), which steps every grad_accum
micro-steps with linear warmup then cosine decay. Validation loss (held-out tasks) at the start, every
val_every steps and at the end. Rank 0 writes metrics.jsonl and the adapter(s).
"""
import ctypes
import datetime
import gc
import json
import math
import sys
import threading
import time

import torch
import torch.distributed as dist

from .config import OUT_DIR, TrainSettings
from .data import TrainData
from .export import export_adapter
from .fix_lora import ShardedAdamW
from .model import add_lora, load_base, lora_modules, wrap_fsdp


def host_gb():
    """(this process's RSS, node MemAvailable) in GB, from /proc."""
    def kb(path, key):
        with open(path) as f:
            return int(next(l for l in f if l.startswith(key)).split()[1]) / 2 ** 20
    return round(kb("/proc/self/status", "VmRSS"), 1), round(kb("/proc/meminfo", "MemAvailable"), 1)


def release_heap():
    """Return freed heap to the OS: loading 62 GB of bf16 shards leaves glibc holding freed CPU memory (LC1 v4: a rank
    was SIGKILLed for host RAM 50 s after load, at the first 4k step)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


class HostWatch(threading.Thread):
    """Samples RSS / MemAvailable every 5 s into OUT_DIR/host_rank{r}.json (written each sample, so it survives a
    SIGKILL): the minimum MemAvailable, the maximum RSS and the last 120 samples."""

    def __init__(self, rank, t0):
        super().__init__(daemon=True)
        self.path, self.t0 = OUT_DIR / f"host_rank{rank}.json", t0
        self.min_avail, self.max_rss, self.tail = 1e9, 0.0, []

    def run(self):
        while True:
            rss, avail = host_gb()
            self.min_avail, self.max_rss = min(self.min_avail, avail), max(self.max_rss, rss)
            self.tail = (self.tail + [[round((time.time() - self.t0) / 60, 2), rss, avail]])[-120:]
            try:
                self.path.write_text(json.dumps({"min_avail_gb": self.min_avail, "max_rss_gb": self.max_rss,
                                                 "tail_min_rss_avail": self.tail}))
            except OSError:
                pass
            time.sleep(5)


class Trainer:
    def __init__(self, s, model_path):
        self.s, self.model_path = s, model_path
        dist.init_process_group("gloo" if s.smoke else "nccl", timeout=datetime.timedelta(minutes=10))
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        if s.smoke:
            self.dev, self.dt = torch.device("cpu"), torch.float32
        else:
            torch.cuda.set_device(self.rank)
            self.dev, self.dt = torch.device("cuda", self.rank), torch.bfloat16
            # flash caps head_dim at 256 and math materialises T x T: force the memory-efficient kernel
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.host = None
        if not s.smoke:
            self.host = HostWatch(self.rank, s.t0)
            self.host.start()
        self.metrics = open(OUT_DIR / "metrics.jsonl", "a") if self.rank == 0 else None

    def log(self, **kw):
        kw["t_min"] = round((time.time() - self.s.t0) / 60, 1)
        print(f"rank{self.rank}", json.dumps(kw), flush=True)
        if self.metrics:
            self.metrics.write(json.dumps(kw) + "\n")
            self.metrics.flush()

    def build(self):
        t = time.time()
        m = load_base(self.model_path, self.rank, self.s.smoke)
        load_s = round(time.time() - t)
        m = add_lora(m, self.s.rank, self.s.alpha, self.dt)
        n_lora = lora_modules(m)
        self.fm, self.layer_cls, self.lora = wrap_fsdp(m, self.s.smoke, 32 if self.s.smoke else self.s.mlp_tile)  # smoke: T~100
        self.fm.train()  # from_pretrained returns eval mode, where HF skips gradient checkpointing (LC1 v4/v5: a 4k
        # step then kept every layer's activations and dequantized weights, ~40 GB/rank of host RAM via save_on_cpu)
        if not self.s.smoke:
            torch.cuda.empty_cache()
        self.vocab = m.config.get_text_config().vocab_size
        rss0, _ = host_gb()
        release_heap()
        rss1, avail = host_gb()
        self.log(event="model", load_s=load_s, lora_modules=n_lora, lora_params=sum(p.numel() for p in self.lora),
                 gpu_gb=0 if self.s.smoke else round(torch.cuda.memory_allocated() / 2 ** 30, 2),
                 rss_gb_before_trim=rss0, rss_gb=rss1, node_avail_gb=avail)

    def loss(self, ids, lab, wts, norm):
        """sum_t wts_t * ce_t / norm for one sequence; checkpoint inputs of long sequences live in pinned host RAM."""
        if ids.shape[1] >= self.s.offload_from:
            with torch.autograd.graph.save_on_cpu(pin_memory=False):  # pinned blocks round up: host OOM at 28k (LC1 v2)
                return self.fm(ids, lab, wts, norm)
        return self.fm(ids, lab, wts, norm)

    def val_totals(self, data, ws):
        """-> [weighted loss sum, weight sum] over the windows ws, all ranks."""
        tot = torch.zeros(2, device=self.dev, dtype=torch.float64)
        n_real = len(ws[self.rank::self.world])
        mine = ws[self.rank::self.world] + ws[:1] * (len(ws[::self.world]) - n_real)  # equal counts per rank (FSDP); repeat, weight 0
        with torch.no_grad():
            for k, w in enumerate(mine):
                ids, lab, wts = data.example(w, self.dev)
                real = k < n_real
                loss = self.loss(ids, lab, wts, 1.0)
                tot += torch.tensor([float(loss) * real, float(wts.sum()) * real], device=self.dev, dtype=torch.float64)
        dist.all_reduce(tot)
        return tot

    def validate(self, data, tag):
        """Weighted mean loss on the held-out tasks (every trained token counts the same); ranks split the windows.
        With corrected held-out episodes (rs4) the on-policy and corrected sets are also reported separately."""
        if not data.val_w:
            return None
        self.fm.eval()
        sets = {k: ws for k, ws in data.val_sets.items() if ws} or {"all": data.val_w}
        tot = torch.zeros(2, device=self.dev, dtype=torch.float64)
        for name, ws in sets.items():
            t = self.val_totals(data, ws)
            tot += t
            if name != "all":
                self.log(event="val_set", tag=tag, set=name, loss=round(float(t[0] / max(1e-6, float(t[1]))), 4), windows=len(ws))
        self.fm.train()
        v = float(tot[0] / max(1e-6, float(tot[1])))
        self.log(event="val", tag=tag, loss=round(v, 4), windows=len(data.val_w))
        return v

    def export(self, tag):
        out = OUT_DIR / ("adapter_" + tag)
        info = export_adapter(self.fm, self.layer_cls, str(out), self.rank, self.s.rank, self.s.alpha, self.model_path)
        dist.barrier()
        if info:
            self.log(event="export", tag=tag, **info)

    def lr_at(self, step, total):
        if step < self.s.warmup_steps:
            return self.s.lr * (step + 1) / self.s.warmup_steps
        p = (step - self.s.warmup_steps) / max(1, total - self.s.warmup_steps)
        return self.s.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    def run(self, data):
        plan = data.plan(data.train_w, self.world, self.s.grad_accum, self.s.epochs)
        total = len(plan)
        self.log(event="plan", optimizer_steps=total, micro_steps=total * self.s.grad_accum, **data.stats())
        opt = ShardedAdamW(self.lora, lr=self.s.lr)
        self.validate(data, "start")
        for step, micro in enumerate(plan):
            stop = torch.tensor([1.0 if self.s.hours_left() < 0.4 else 0.0], device=self.dev)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)  # every rank stops at the same step
            if float(stop):
                self.log(event="budget_stop", optimizer_step=step)
                break
            t0, norm, tok, lsum = time.perf_counter(), data.step_norm(micro), 0, 0.0
            for windows in micro:
                ids, lab, wts = data.example(windows[self.rank], self.dev)
                loss = self.loss(ids, lab, wts, norm)
                loss.backward()
                lsum += float(loss.detach())
                tok += ids.shape[1]
            for g in opt.param_groups:
                g["lr"] = self.lr_at(step, total)
            gn = opt.step(1.0)
            opt.zero_grad()
            if not self.s.smoke:
                torch.cuda.synchronize(self.dev)
            dt = time.perf_counter() - t0
            local_peak = 0 if self.s.smoke else torch.cuda.max_memory_allocated() / 2 ** 30
            red = torch.tensor([tok, lsum, 0.0, 0.0], device=self.dev, dtype=torch.float64)
            dist.all_reduce(red)  # tokens and the step's weighted mean loss (each rank's part is already / norm)
            mx = torch.tensor([dt, local_peak], device=self.dev, dtype=torch.float64)
            dist.all_reduce(mx, op=dist.ReduceOp.MAX)
            self.log(event="step", step=step + 1, of=total, loss=round(float(red[1]), 4), lr=round(self.lr_at(step, total), 7),
                     grad_norm=round(float(gn), 3), tokens=int(red[0]), elapsed_s=round(float(mx[0]), 3),
                     tok_per_s=round(float(red[0]) / float(mx[0]), 1), peak_gb=round(float(mx[1]), 2),
                     max_len=max(data.length(w) for ws in micro for w in ws))
            last = step + 1 == total  # the end-of-run validation and export follow anyway
            if self.s.val_every and (step + 1) % self.s.val_every == 0 and not last:
                self.validate(data, f"step{step + 1}")
            if self.s.save_every and (step + 1) % self.s.save_every == 0 and not last:
                self.export(f"step{step + 1}")
        self.validate(data, "end")
        self.export("final")

    def close(self):
        dist.barrier()
        dist.destroy_process_group()


def main():
    s = TrainSettings.load()
    tr = Trainer(s, sys.argv[1])
    tr.build()
    if s.prefix_probe or s.lc_probe:
        if s.prefix_probe:
            from .prefix import probe
        else:
            from .lcprobe import probe
        probe(tr, s.smoke)
        tr.close()
        return
    data = TrainData(sys.argv[2], s.max_windows, vocab=tr.vocab if s.smoke else None, seed=s.seed, max_len=s.max_len,
                     corr=s.corr, corr_share=s.corr_share)
    if s.vablate:
        from .vablate import probe
        probe(tr, data, s.smoke)
    else:
        tr.run(data)
    tr.close()


if __name__ == "__main__":
    main()
