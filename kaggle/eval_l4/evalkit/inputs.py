"""Inputs: where the tasks, snapshots, graphs and served model live under /kaggle/input."""
import json
import pathlib
from dataclasses import dataclass

from .config import INPUT
from .util import log


@dataclass
class Inputs:
    tasks_path: pathlib.Path
    data: pathlib.Path        # competition data dir (graphs/, embeddings/, wheels/)
    snapshots: pathlib.Path
    model_path: str

    @classmethod
    def locate(cls, devset=False, need_model=True):
        all_tasks = sorted(INPUT.rglob("tasks.jsonl"))
        comp = [p for p in all_tasks if (p.parent / "graphs").is_dir()]
        assert comp or all_tasks, "competition data not attached"
        data = (comp or all_tasks)[0].parent
        tasks_path = data / "tasks.jsonl"
        if devset:
            dev = [p for p in all_tasks if (p.parent / "devset_report.json").exists()]
            assert dev, "devset=True but the gemma4-devset-build output is not attached"
            tasks_path = dev[0]
        cfgs = sorted(p for p in INPUT.rglob("config.json") if "w4a16" in str(p).lower())
        assert cfgs or not need_model, "attach model google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct"
        inp = cls(tasks_path, data, tasks_path.parent / "snapshots", str(cfgs[0].parent) if cfgs else "")
        log("inputs:", inp)
        return inp

    def load_tasks(self):
        return [json.loads(l) for l in open(self.tasks_path) if l.strip()]
