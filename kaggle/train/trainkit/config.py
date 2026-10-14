"""Training settings and fixed paths. Stdlib only."""
import json
import pathlib
import time
from dataclasses import asdict, dataclass, field

WORK = pathlib.Path("/kaggle/working")
INPUT = pathlib.Path("/kaggle/input")
CFG_FILE = WORK / "train_cfg.json"
DATA_FILE = WORK / "train_data.npz"       # written by the notebook's inputs cell (or found under /kaggle/input)
OUT_DIR = WORK / "train_out"              # adapters, metrics, per-rank logs
SMOKE_CONFIG = WORK / "g4cfg.json"        # real Gemma-4 config, for the tiny smoke model
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class TrainSettings:
    name: str = "conv_lora"           # adapter name (adapters/<name>/ in the submission)
    rank: int = 16
    alpha: int = 32
    lr: float = 1e-4
    warmup_steps: int = 5
    grad_accum: int = 2               # optimizer step every grad_accum windows per rank (batch = 4 ranks x accum)
    epochs: float = 1.0
    max_windows: int | None = None    # cap (pipeline smoke runs)
    offload_from: int = 4097          # windows longer than this keep checkpoint activations in host RAM
    val_every: int = 0                # optimizer steps between validation passes (0 = only at the end)
    save_every: int = 0               # optimizer steps between adapter exports (0 = only at the end)
    budget_h: float = 10.5            # stop taking steps after this, then export (session cap 12 h)
    seed: int = 0
    smoke: bool = False               # CPU, gloo, 2 ranks, tiny random model
    t0: float = field(default_factory=time.time)

    def hours_left(self):
        return self.budget_h - (time.time() - self.t0) / 3600

    def save(self, path=CFG_FILE):
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path=CFG_FILE):
        return cls(**json.loads(path.read_text()))
