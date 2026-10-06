Analyze an evaluation run without flooding the context. Argument: path to an `official_l4_*.tgz` (or an extracted dir) and a run id, e.g. `data/official_l4_v2_sed_dev.tgz r5_v2_dev`.

1. Extract into `data/runs/$RUN_ID/` (the tgz contains `merged/`, `run.log`, optionally `analysis/`). Never cat traces or logs.
2. Print only the last 15 lines of `run.log`. Check for crashes, budget stops and the RESULT line.
3. If `analysis/` is missing and the matching tasks.jsonl with gold patches is available, run `analysis/trace_analyzer.py`.
4. Run `python analysis/compare_runs.py <previous best run> data/runs/$RUN_ID/merged`, using the run listed as current best in `docs/experiments.md`.
5. Copy the small artefacts (task_results.jsonl, summary.json, report.md) into `results/runs/$RUN_ID/`.
6. Add a row to `docs/experiments.md`: hypothesis, change, local result, paired p-value, key behaviour shifts, one-line takeaway.
7. Report to the user in ≤10 lines: result, significance, the 2–3 behaviour changes that explain it, and the proposed next single change.
