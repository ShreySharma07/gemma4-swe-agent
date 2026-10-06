# Experiment log

Local = scorer-faithful run on Kaggle L4×4 (`run_official_l4.py`, subprocess sandbox), 129 public tasks unless noted. LB = public leaderboard (hidden private-repo tasks; 1 task ≈ 0.8 pp).

| ID | Date | Submission | Change (one per run) | Local | LB | Takeaway |
|---|---|---|---|---|---|---|
| r1 | Oct 1 | baseline_v1 | organisers' sample agent, temp 0.2, thinking off, 4.5 min / 40 calls | 15/129 (15/97 verifiable) | 0.00 | 42% empty patches. `read_file` args garbled (47% of reads) → read loops; context resets |
| r2 | Oct 3 | v2_sed | remove `read_file`; read via grep/sed; test deps fixed locally | 26/129 | **0.08** | Empty patches 41 → 20, repeats 48% → 28%. New top failure: wrong fix never tested (47) |
| — | Oct 3 | v2b_budget80 | v2 with 80 calls / 160 turns | not run | not submitted | |
| — | Oct 3 | v3_verify | v2 + mandatory budget plan + one targeted test | not run | not submitted | |
| — | Oct 3 | v4_pipeline | localizer (read-only + graph) → fixer (fresh context), 80 calls | not run | not submitted | |
| r3 | Oct 4 | v5_graph | v4 + `code-map` ast skill + exact-name graph usage | 33/129 (+12/−5 vs v2, p=0.14) | 0.08 | Graph tools 0 calls. Localizer overran (48 calls) and starved the fixer. 61% of edit calls garbled. 21 timeouts. The skill probably never ran on the scorer |
| r4 | Oct 5 | v6_trace | localizer whose only tool is the code-map skill (22-call hard budget) + rich handoff brief | not run | **0.00, ~10 h** | Nearly every task timed out → the skill most likely fails in the scorer's Docker sandbox. v6 line dropped |

## Fidelity notes
- Before Oct 5 the local runner used compaction interval 5 / threshold 14336. The scorer uses 15 / 32768, so effectively no compaction. Fixed. Replaying v5 without compaction, only 3–5 resolved tasks would have overflowed the context.
- About 20/129 tasks are environment-unstable locally (competitor report: gold passes 109/129). The dev-set builder records a gold check per task.
