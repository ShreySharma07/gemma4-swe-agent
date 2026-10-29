"""
R2 on-policy factory rollouts (GPU). Pre-registered in docs/experiments.md (Oct 29).
All verified factory tasks from the attached datasets (gemma4-factory-v0: fac_*, gemma4-factory-v1: f1_*), shuffled
with seed 0 and split into n_shards; this kernel runs tasks[shard::n_shards]:
  1. rewrite   the FS1 issue rewrite (Gemma, thinking off), same leak checks and template fallback
  2. rollouts  the submission under test (v14) unchanged except the per-task time cap (Settings.max_time_minutes),
               concurrency 6 (FS1: 2.65x throughput; 4/54 timeouts at the 8 min cap came from contention)
Output: /kaggle/working/factory_rollout_report.json + factory_rollout.tgz (report, rewritten tasks, run.log, traces).
Run inside the venv: python -m evalkit.factory_rollout
"""
import json
import pathlib
import random
import time

import litellm

from .compat import patch_sandbox_venv, use_scorer_sandbox
from .config import INPUT, WORK, Settings
from .factory_smoke import breakdown, phase, rewrite_all
from .inputs import Inputs
from .server import ScorerServer, make_sandbox_offline, set_scorer_env
from .submission import Submission
from .util import log

CONC, CHUNK = 6, 30  # bigger chunks than FS1 (18): fewer idle slots while a chunk's last tasks finish
TASKS_OUT = WORK / "rollout_tasks.jsonl"
SNAPS = pathlib.Path("/tmp/rollout_snapshots")  # symlinks to both datasets' snapshots (Inputs takes one dir)


def load_factory():
    """Every verified factory task + meta across the attached factory datasets, ids unique."""
    tasks, meta, seen = [], {}, set()
    SNAPS.mkdir(parents=True, exist_ok=True)
    for tp in sorted(p for p in INPUT.rglob("tasks.jsonl") if (p.parent / "task_meta.jsonl").exists()):
        rows = [json.loads(l) for l in open(tp) if l.strip()]
        meta.update({json.loads(l)["id"]: json.loads(l) for l in open(tp.parent / "task_meta.jsonl")})
        for t in rows:
            assert t["instance_id"] not in seen, f"duplicate id {t['instance_id']}"
            seen.add(t["instance_id"])
            src = tp.parent / "snapshots" / f"{t['instance_id']}.tgz"
            assert src.exists(), f"missing snapshot {src}"
            (SNAPS / src.name).unlink(missing_ok=True)
            (SNAPS / src.name).symlink_to(src)
        tasks += rows
        log(f"factory: {tp.parent} -> {len(rows)} tasks")
    assert tasks, "attach datasets gemma4-factory-v0 / gemma4-factory-v1"
    return tasks, meta


def main():
    s = Settings.load()
    set_scorer_env()
    patch_sandbox_venv()
    litellm.drop_params = True
    comp = Inputs.locate(False)
    use_scorer_sandbox(comp.data)
    tasks, meta = load_factory()
    tasks.sort(key=lambda t: t["instance_id"])
    random.Random(0).shuffle(tasks)
    tasks = tasks[s.shard::s.n_shards]
    log(f"shard {s.shard}/{s.n_shards}: {len(tasks)} tasks, cap {s.max_time_minutes} min, concurrency {CONC}")

    sub = Submission().unpack()
    declared, adapters = sub.declared_model(), sub.adapters()
    server = ScorerServer(comp.model_path, adapters).start()
    models = server.registry(declared)
    report = {"shard": s.shard, "n_shards": s.n_shards, "tasks": [t["instance_id"] for t in tasks]}
    rows = []
    try:
        report["rewrite"] = rewrite_all(server, tasks)
        with open(TASKS_OUT, "w") as f:  # rewritten issues = the prompts the trajectories were sampled from
            f.writelines(json.dumps(t) + "\n" for t in tasks)
        make_sandbox_offline()
        inp = Inputs(TASKS_OUT, comp.data, SNAPS, comp.model_path)
        report["run"], rows = phase(s, inp, sub, models, adapters, f"rollout_s{s.shard}",
                                    [t["instance_id"] for t in tasks], CONC, CHUNK)
        log("RUN", json.dumps({k: v for k, v in report["run"].items() if k != "summary"}))
    finally:
        budget = {**sub.eval_budget(), **({"max_time_minutes": s.max_time_minutes} if s.max_time_minutes else {})}
        report["overall"] = {"resolved": sum(bool(r.get("resolved")) for r in rows), "n": len(rows),
                             **breakdown(rows, meta, budget)}
        (WORK / "factory_rollout_report.json").write_text(json.dumps(report, indent=1))
        log("R2 REPORT", json.dumps({k: v for k, v in report.items() if k not in ("tasks", "rewrite")}))
        log.sh(f"cd {WORK} && tar -czf factory_rollout.tgz factory_rollout_report.json rollout_tasks.jsonl run.log "
               f"$(ls eval_rollout_*.tgz 2>/dev/null)")
        server.stop()
        log(f"RUN_DONE in {(time.time() - s.t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
