"""
Torchrun entry point (one process per GPU): python -m torch.distributed.run --nproc_per_node 4 -m trainkit.worker MODEL
Data parallel: each rank runs one window per micro-step; FSDP averages gradients across ranks; AdamW steps every
grad_accum micro-steps with linear warmup then cosine decay. Validation loss (held-out tasks) at the start, every
val_every steps and at the end. Rank 0 writes metrics.jsonl and the adapter(s).
"""
import datetime
import json
import math
import sys
import time

import torch
import torch.distributed as dist

from .config import OUT_DIR, TrainSettings
from .data import TrainData
from .export import export_adapter
from .model import add_lora, load_base, lora_modules, wrap_fsdp


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
        self.fm, self.layer_cls = wrap_fsdp(m, self.s.smoke)
        if not self.s.smoke:
            torch.cuda.empty_cache()
        self.vocab = m.config.get_text_config().vocab_size
        self.log(event="model", load_s=load_s, lora_modules=n_lora,
                 gpu_gb=0 if self.s.smoke else round(torch.cuda.memory_allocated() / 2 ** 30, 2))

    def step_loss(self, data, w):
        ids, lab = data.example(w, self.dev)
        if ids.shape[1] >= self.s.offload_from:  # checkpoint inputs live in pinned host RAM (188 GB)
            with torch.autograd.graph.save_on_cpu(pin_memory=not self.s.smoke):
                return self.fm(ids, lab), int((lab != -100).sum())
        return self.fm(ids, lab), int((lab != -100).sum())

    def validate(self, data, tag):
        """Mean trained-token loss on the held-out tasks; every rank evaluates its share, results are reduced."""
        if not data.val_w:
            return None
        self.fm.eval()
        tot = torch.zeros(2, device=self.dev, dtype=torch.float64)
        mine = data.schedule(data.val_w, self.rank, self.world) or []
        with torch.no_grad():
            for w in mine:
                loss, n = self.step_loss(data, w)
                tot += torch.tensor([float(loss) * n, n], device=self.dev, dtype=torch.float64)
        dist.all_reduce(tot)
        self.fm.train()
        v = float(tot[0] / max(1.0, float(tot[1])))
        self.log(event="val", tag=tag, loss=round(v, 4), windows=len(mine) * self.world)
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
        sched = data.schedule(data.train_w, self.rank, self.world, self.s.epochs)
        total = math.ceil(len(sched) / self.s.grad_accum)
        self.log(event="plan", micro_steps=len(sched), optimizer_steps=total, **data.stats())
        opt = torch.optim.AdamW([p for p in self.fm.parameters() if p.requires_grad], lr=self.s.lr, weight_decay=0.0)
        self.validate(data, "start")
        step, t_win, tok, loss_acc = 0, time.perf_counter(), 0, []
        for i, w in enumerate(sched):
            stop = torch.tensor([1.0 if self.s.hours_left() < 0.4 else 0.0], device=self.dev)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)  # every rank stops at the same micro-step
            if float(stop):
                self.log(event="budget_stop", micro_step=i, optimizer_step=step)
                break
            loss, n = self.step_loss(data, w)
            (loss / self.s.grad_accum).backward()
            loss_acc.append(float(loss.detach()))
            tok += data.length(w)
            if (i + 1) % self.s.grad_accum == 0 or i + 1 == len(sched):
                for g in opt.param_groups:
                    g["lr"] = self.lr_at(step, total)
                gn = self.fm.clip_grad_norm_(1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                # Variable-length windows make world * rank0_tokens an underestimate.
                # Keep the actual workload and slowest rank's duration for auditing.
                if not self.s.smoke:
                    torch.cuda.synchronize(self.dev)  # include queued optimizer work in elapsed time
                dt = time.perf_counter() - t_win
                local_peak = 0 if self.s.smoke else torch.cuda.max_memory_allocated() / 2 ** 30
                tokens = torch.tensor([tok], device=self.dev, dtype=torch.int64)
                resources = torch.tensor([dt, local_peak], device=self.dev, dtype=torch.float64)
                dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
                dist.all_reduce(resources, op=dist.ReduceOp.MAX)
                total_tokens, elapsed_s = int(tokens.item()), float(resources[0])
                self.log(event="step", step=step, of=total, loss=round(sum(loss_acc) / len(loss_acc), 4),
                         lr=round(self.lr_at(step - 1, total), 7), grad_norm=round(float(gn), 3),
                         tok_per_s=round(total_tokens / elapsed_s, 1),
                         tokens=total_tokens, elapsed_s=round(elapsed_s, 6),
                         rank_tokens=tok, rank_elapsed_s=round(dt, 6),
                         peak_gb=round(float(resources[1]), 2))
                t_win, tok, loss_acc = time.perf_counter(), 0, []
                if self.s.val_every and step % self.s.val_every == 0:
                    self.validate(data, f"step{step}")
                if self.s.save_every and step % self.s.save_every == 0:
                    self.export(f"step{step}")
        self.validate(data, "end")
        self.export("final")

    def close(self):
        dist.barrier()
        dist.destroy_process_group()


def main():
    s = TrainSettings.load()
    tr = Trainer(s, sys.argv[1])
    tr.build()
    data = TrainData(sys.argv[2], s.max_windows, vocab=tr.vocab if s.smoke else None, seed=s.seed)
    tr.run(data)
    tr.close()


if __name__ == "__main__":
    main()
