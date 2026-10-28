"""Run settings and fixed paths. Stdlib only, so the env step can import it outside the venv."""
import json
import pathlib
import time
from dataclasses import asdict, dataclass, field

WORK = pathlib.Path("/kaggle/working")
INPUT = pathlib.Path("/kaggle/input")
VENV = pathlib.Path("/tmp/scorer_host")
TARGET_MODEL = "gemma-4-31b-it-qat-w4a16-ct"
CFG_FILE = WORK / "eval_cfg.json"
SUBMISSION_ZIP = WORK / "submission_src.zip"
ANALYZER = WORK / "trace_analyzer.py"
SANDBOX_ENV = WORK / "sandbox_env.json"   # task-sandbox interpreter + site, written by envsetup
SCORER_SITE = pathlib.Path("/tmp/scorer_site")
SANDBOX_PIP = pathlib.Path("/tmp/sandbox_pip")
VALID_TASKS = WORK / "valid_tasks.json"       # from the preflight sweep; embedded by make_eval_notebook.py


@dataclass
class Settings:
    name: str = "v7_noedit"           # submissions/<name>; label of every output
    task_ids: list | None = None      # None -> all tasks, interleaved by repo (any prefix is a stratified sample)
    max_tasks: int | None = None      # cap the task list (shakedown runs)
    seed: int = 0
    chunk_size: int = 8
    chunk_timeout_min: float = 60     # floor; the runner raises it to chunk_size x (task max_time + 3 min) (Oct 19)
    session_budget_h: float = 10.5    # stop scheduling chunks after this; the session cap is 12 h
    est_task_min: float = 5.5         # 4.5 min agent budget + setup/verification
    concurrency: int = 1              # tasks in flight per Evaluator call (1 = scorer; >1 only for data generation)
    repeats: int = 1                  # >1: run the list again (rep-major) -> pass@k, trajectories for rejection sampling
    devset: bool = False              # True -> renamed dev set (kernel output gemma4-devset-build)
    valid_only: bool = False          # True -> only tasks in valid_tasks.json (gold passes, empty fails; preflight sweep)
    sweep: bool = False               # preflight: gold + empty over ALL tasks -> valid_tasks.json (else 7-task smoke)
    gate_adapters: list | None = None  # behaviour gate: adapter names to compare with base (evalkit/gate.py)
    gate_samples: int = 4             # behaviour gate: samples per decision point
    t0: float = field(default_factory=time.time)  # notebook start; budget counts from here across processes

    @property
    def label(self):
        return f"eval_{self.name}" + ("_dev" if self.devset else "")

    def hours_left(self):
        return self.session_budget_h - (time.time() - self.t0) / 3600

    def save(self, path=CFG_FILE):
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path=CFG_FILE):
        return cls(**json.loads(path.read_text()))
