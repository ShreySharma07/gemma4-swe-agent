"""
Preflight (CPU, offline, no model): the official Evaluator with a fake agent that submits the GOLD patch for tasks
known to pass in the scorer env (exec-graph run, 54/54), and an EMPTY patch for one of them. Proves task sandboxes,
test deps, verification and result files before a GPU session is spent. Expect gold -> resolved, empty -> not.
Run inside the venv: python -m evalkit.preflight
"""
import asyncio
import json
import pathlib

import pandas as pd

from .compat import patch_sandbox_venv, use_scorer_sandbox
from .config import WORK, Settings
from .inputs import Inputs
from .server import make_sandbox_offline, set_scorer_env
from .util import log

GOLD_OK = ["fastapi_14258", "requests_6644", "rich_2725", "fastapi_14372", "requests_7205", "rich_2943"]
EMPTY = "rich_3006"  # gold-passing task submitted with no patch: must NOT resolve


def gold_patches(inp):
    """Gold patches: secret/perfect_submission.* anywhere under /kaggle/input, else the tasks' own `patch` field.
    Also reports where the verifier's test patches (secret/solution.*) live: without them nothing can be verified."""
    from .config import INPUT
    log("data dir:", sorted(p.name for p in inp.data.iterdir()))
    log("task keys:", sorted(inp.load_tasks()[0]))
    sols = sorted(INPUT.rglob("solution.parquet")) + sorted(INPUT.rglob("solution.csv"))
    log("secret solution files:", [str(p) for p in sols] or "NONE (verifier has no test patches)")
    for f in sorted(INPUT.rglob("perfect_submission.parquet")) + sorted(INPUT.rglob("perfect_submission.csv")):
        df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        log("gold from", f)
        return {str(r.get("id") or r.get("instance_id")): r.get("prediction") or r.get("patch")
                for r in df.to_dict(orient="records")}
    log("gold from tasks.jsonl `patch`")
    return {t["instance_id"]: t.get("patch") or "" for t in inp.load_tasks()}


def verify(inp, patches, label):
    """Official Evaluator with a fake agent that returns patches[instance_id]; -> {id: row}."""
    from adk_eval_core.tracing import SessionTrace
    from adk_submission import ModelRegistry
    from swegemma.config import EvalConfig
    from swegemma.evaluate import Evaluator

    async def fake_agent(self, task, snapshot_path, task_index, total_tasks, **kw):
        return patches[task.instance_id], None, SessionTrace()

    Evaluator._run_agent_sandbox = fake_agent
    dummy = WORK / "preflight_submission"
    dummy.mkdir(exist_ok=True)
    cfg = EvalConfig(tasks_path=inp.tasks_path, snapshots_dir=inp.snapshots, results_dir=WORK / f"preflight_{label}",
                     submission_dir=dummy, models=ModelRegistry(), sandbox="subprocess", wheels_dir=inp.data / "wheels",
                     graph_dir=str(inp.data / "graphs"), embeddings_dir=str(inp.data / "embeddings"),
                     task_ids=sorted(patches), concurrency=4, display_mode="quiet", max_time_minutes=30,
                     timeout_seconds=600)
    res = asyncio.run(Evaluator(cfg).run())
    return {r.instance_id: {"resolved": bool(r.resolved), "exit": r.test_exit_code, "error": r.error,
                            "tail": (r.test_output or "")[-600:]} for r in res.task_results}


def main():
    s = Settings.load()
    set_scorer_env()
    patch_sandbox_venv()
    inp = Inputs.locate(False, need_model=False)
    use_scorer_sandbox(inp.data)
    make_sandbox_offline()
    gold = gold_patches(inp)
    ids = sorted(gold) if s.sweep else GOLD_OK + [EMPTY]
    g = verify(inp, {i: gold[i] for i in ids if i != EMPTY or s.sweep} | ({} if s.sweep else {EMPTY: ""}), "gold")
    e = verify(inp, {i: "" for i in ids}, "empty") if s.sweep else {EMPTY: g.pop(EMPTY)}
    (WORK / "preflight.json").write_text(json.dumps({"gold": g, "empty": e}, indent=1))
    if s.sweep:
        valid = sorted(i for i in ids if g.get(i, {}).get("resolved") and not e.get(i, {}).get("resolved", True))
        (WORK / "valid_tasks.json").write_text(json.dumps(valid, indent=1))
        by_repo = {}
        for i in ids:
            r = by_repo.setdefault(i.rsplit("_", 1)[0], [0, 0, 0])
            r[0] += 1
            r[1] += bool(g.get(i, {}).get("resolved"))
            r[2] += i in valid
        log(f"SWEEP: valid {len(valid)}/{len(ids)} (gold passes and empty fails); per repo [n, gold, valid]: {by_repo}")
        return
    gold_ok = sum(g.get(i, {}).get("resolved", False) for i in GOLD_OK)
    empty_ok = not e.get(EMPTY, {}).get("resolved", True)
    verdict = "PASS" if gold_ok >= len(GOLD_OK) - 1 and empty_ok else "FAIL"
    log(f"PREFLIGHT {verdict}: gold resolved {gold_ok}/{len(GOLD_OK)}, empty patch unresolved={empty_ok}")
    for i, r in (g | e).items():
        log(i, r["resolved"], r["exit"], r["error"], r["tail"][-200:].replace("\n", " | "))


if __name__ == "__main__":
    main()
