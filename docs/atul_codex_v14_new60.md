Shrey asks you (Atul) to run ONE GPU job: **v14 (thinking on) on the 60 new tasks**. Shrey's quota is used up until Oct 24.

Why: the top public notebooks (0.22–0.24) all run with thinking ON (4,096-token budget), 8 min / 28 calls per task. A forum study on the same 129 tasks reports thinking off solving 23 vs thinking on 48–51. Our v7 runs with thinking OFF (19/129 locally; **23/60 on the new-60**, `data/runs/eval_v7_new60/`). `submissions/v14_thinking` = v7 + the submit_patch prompt fix (v13) + thinking on + the public budgets (8 min, 28 calls, 240 s command timeout, 80 turns). Nothing else changes, so the comparison with v7's 23/60 on the same tasks is direct.

First `git pull`. The eval notebook has safeguards: each 8-task chunk is abandoned after 8 × (task limit + 3 min), and the evaluator is killed at `--budget-h` + 15 min, saving finished work as `partial_eval.tgz`.

Setup as before: Kaggle CLI in `.venv`, already signed in by you; never create, print or handle tokens; `export KAGGLE_OWNER=atulbaliyan`.

Steps:
1. Check nothing of yours is queued or running on Kaggle (one 4×L4 session per account; a second push waits in the queue).
2. Build and push. Expected ~4.5–6 session-h ≈ 9–12 quota-h; hard cap 7.75 session-h ≈ 15.5 quota-h. You have ~21 h.
   ```bash
   python3 kaggle/eval_l4/make_eval_notebook.py v14_thinking --valid-only --valid-file data/runs/eval_sweep_v2/new_tasks.json --budget-h 7.5
   .venv/bin/kaggle kernels push -p kaggle/probes/check_l4
   ```
   Check before pushing: `kernel-metadata.json` id `atulbaliyan/check-l4`, no adapter sources; settings show `session_budget_h: 7.5`; the embedded submission's `configs/sampling.yaml` has `thinking_budget: 4096`.
3. Poll `kaggle kernels status atulbaliyan/check-l4` every 15 min. If it is still RUNNING after 8 h, tell me (Atul) so I can cancel it from the notebook page (the CLI cannot cancel).
4. When it finishes, download `summary.json`, `run.log` and `eval_v14_thinking.tgz` (or `partial_eval.tgz`) into `data/runs/atul_eval_v14_new60/`. Commit `summary.json`, `run.log` and `eval_v14_thinking/rep0/analysis/` (summary.json, per_task.jsonl, report.md), not the tgz.
5. Add the run to §8 of `docs/post_training_guide.md` with the actual quota charge.

Report: resolved/60 (and how many of the 60 ran, if it stopped early); the report.md metric table (empty_patch_pct, tasks_overflow_pct, repeated_identical_calls_share_pct, explicit_submit_pct, tool_calls_median, tasks_budget_exhausted_pct); per-chunk timing from run.log; any WATCHDOG or chunk-timeout lines; and the per-task overlap with v7 (`data/runs/eval_v7_new60/x/eval_v7_noedit/rep0/merged/task_results.jsonl`, key `instance_id`): solved by both / v14 only / v7 only.

Then STOP. No other runs, no leaderboard submission. Commit on a branch as before.
