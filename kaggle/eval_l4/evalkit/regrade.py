"""
R1 stray-file re-grade (CPU, offline, no model). Pre-registered in docs/experiments.md (Oct 28).
Question: do stray scratch files (repro.py, reproduce_issue.py, non-.py files; trace_analyzer's `stray` rule) make
otherwise-correct patches fail, or are they only a symptom of runs that ran out of budget?
For every valid-task run in the embedded payload (same runs as A3/A4) whose patch has stray files:
  orig      the agent's patch as submitted (control: re-grade noise)
  stripped  the same patch minus the stray files' diff blocks
Resolved cases are included too (stripping must not break them). Output: /kaggle/working/regrade/report.json.
Run inside the venv: python -m evalkit.regrade
"""
import json
import time

from .compat import patch_sandbox_venv, use_scorer_sandbox
from .config import VALID_TASKS, WORK
from .inputs import Inputs
from .necessity import path_of, runs_from_payload, split_patch, verify_full
from .server import make_sandbox_offline, set_scorer_env
from .util import log

OUT = WORK / "regrade"


def strip_files(patch, drop):
    """The patch without the diff blocks of the files in `drop`."""
    keep = []
    for header, hunks in split_patch(patch):
        if path_of(header) in drop or any(h.startswith("+++ b/") and h[6:].strip() in drop for h in header):
            continue
        keep.append("".join(header) + "".join("".join(h[2]) for h in hunks))
    return "".join(keep)


def main():
    t0 = time.time()
    set_scorer_env()
    patch_sandbox_venv()
    inp = Inputs.locate(False, need_model=False)
    use_scorer_sandbox(inp.data)
    make_sandbox_offline()
    OUT.mkdir(parents=True, exist_ok=True)
    valid = set(json.loads(VALID_TASKS.read_text()))
    runs, _ = runs_from_payload()
    cases = []
    for run, rows in sorted(runs.items()):
        for iid, (row, patch) in sorted(rows.items()):
            stray = row.get("stray_files_in_patch") or []
            if iid not in valid or not stray or not patch.strip():
                continue
            stripped = strip_files(patch, set(stray))
            cases.append({"run": run, "id": iid, "resolved_in_run": bool(row["resolved"]), "stray": stray,
                          "orig": patch, "stripped": stripped, "stripped_empty": not stripped.strip()})
    log(f"R1: {len(cases)} stray-file cases ({sum(not c['resolved_in_run'] for c in cases)} unresolved in run)")
    results = []
    for mode in ("orig", "stripped"):  # one Evaluator call per (mode, run): task ids must be unique per call
        for run in sorted({c["run"] for c in cases}):
            batch = [c for c in cases if c["run"] == run and not (mode == "stripped" and c["stripped_empty"])]
            if not batch:
                continue
            res = verify_full(inp, {c["id"]: c[mode] for c in batch}, f"r1_{mode}_{run}")
            for c in batch:
                r = res.get(c["id"], {})
                results.append({"run": run, "id": c["id"], "mode": mode, "resolved_in_run": c["resolved_in_run"],
                                "stray": c["stray"], "resolved": r.get("resolved"), "exit": r.get("exit"),
                                "error": r.get("error"), "tail": (r.get("tail") or "")[-1500:]})
            log(f"  {mode} {run}: {sum(bool(results[-k]['resolved']) for k in range(1, len(batch) + 1))}/{len(batch)}")
    (OUT / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (OUT / "patches.json").write_text(json.dumps(cases, indent=1))

    def tally(unresolved_in_run, mode):
        rs = [r for r in results if r["mode"] == mode and r["resolved_in_run"] is not unresolved_in_run]
        return {"n": len(rs), "resolved": sum(bool(r["resolved"]) for r in rs)}
    key = lambda r: (r["run"], r["id"])
    orig = {key(r): r["resolved"] for r in results if r["mode"] == "orig"}
    rescued = [r for r in results if r["mode"] == "stripped" and r["resolved"] and not orig.get(key(r))]
    broken = [r for r in results if r["mode"] == "stripped" and not r["resolved"] and orig.get(key(r))]
    report = {"cases": len(cases), "stripped_empty": sum(c["stripped_empty"] for c in cases),
              "unresolved_in_run": {"orig": tally(True, "orig"), "stripped": tally(True, "stripped")},
              "resolved_in_run": {"orig": tally(False, "orig"), "stripped": tally(False, "stripped")},
              "rescued_by_stripping": [key(r) for r in rescued], "broken_by_stripping": [key(r) for r in broken],
              "orig_disagrees_with_run": [k for k, v in orig.items()
                                          if bool(v) != next(c["resolved_in_run"] for c in cases if (c["run"], c["id"]) == k)],
              "minutes": round((time.time() - t0) / 60, 1)}
    (WORK / "regrade_report.json").write_text(json.dumps(report, indent=1))
    log("R1 REPORT", json.dumps(report))


if __name__ == "__main__":
    main()
