"""
Fix: LoRA weights outside FSDP, fp32 master weights, exact global loss normalisation.
Before: each layer's FSDP flat parameter mixed frozen NF4 storage with LoRA, so FSDP allocated gradients for the whole
flat parameter (~3.4 GiB/rank inferred), AdamW updated bf16 LoRA weights (updates below 2^-8 relative were lost), and
FSDP averaged per-rank mean losses (a 4k episode's tokens weighed ~7x a 28k episode's).
Now the LoRA tensors (~0.3 GB bf16) are FSDP-ignored and replicated: every rank starts from rank 0's init, backward
accumulates bf16 grads locally, and ShardedAdamW reduce-scatters them in fp32 (SUM: the loss is already divided by
the step's total weight), clips by the global norm, updates its 1/world fp32 master shard and all-gathers the result.
"""
import torch
import torch.distributed as dist


def lora_params(m):
    return [p for n, p in m.named_parameters() if "lora_" in n]


def broadcast_init(params):
    for p in params:
        dist.broadcast(p.data, 0)


def _gloo():
    return dist.get_backend() == "gloo"


class ShardedAdamW:
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = params
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.numel = sum(p.numel() for p in params)
        self.n = -(-self.numel // self.world)  # shard size (padded)
        dev = params[0].device
        flat = self._flat([p.data for p in params], dev)
        self.master = torch.nn.Parameter(flat[self.rank * self.n:(self.rank + 1) * self.n].clone())
        self.opt = torch.optim.AdamW([self.master], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.param_groups = self.opt.param_groups

    def _flat(self, tensors, dev):
        out = torch.zeros(self.n * self.world, device=dev, dtype=torch.float32)
        a = 0
        for t in tensors:
            out[a:a + t.numel()] = t.reshape(-1).float()
            a += t.numel()
        return out

    def step(self, max_norm=1.0):
        """-> global grad norm before clipping. Missing grads count as zero."""
        dev = self.master.device
        g = self._flat([p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params], dev)
        shard = torch.empty(self.n, device=dev, dtype=torch.float32)
        if _gloo():  # gloo has no reduce_scatter
            dist.all_reduce(g)
            shard.copy_(g[self.rank * self.n:(self.rank + 1) * self.n])
        else:
            dist.reduce_scatter_tensor(shard, g)
        del g
        sq = (shard * shard).sum()
        dist.all_reduce(sq)
        norm = sq.sqrt()
        shard.mul_(torch.clamp(max_norm / (norm + 1e-6), max=1.0))
        self.master.grad = shard
        self.opt.step()
        self.master.grad = None
        full = torch.empty(self.n * self.world, device=dev, dtype=self.params[0].dtype)
        if _gloo():
            parts = [torch.empty(self.n, device=dev, dtype=full.dtype) for _ in range(self.world)]
            dist.all_gather(parts, self.master.data.to(full.dtype))
            full = torch.cat(parts)
        else:
            dist.all_gather_into_tensor(full, self.master.data.to(full.dtype))
        a = 0
        for p in self.params:
            p.data.copy_(full[a:a + p.numel()].view_as(p))
            a += p.numel()
        return norm

    def zero_grad(self):
        for p in self.params:
            p.grad = None
