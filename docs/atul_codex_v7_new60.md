Shrey asks you (Atul) to run ONE GPU job: the v7 arm of the Step 3 paired eval. Shrey's quota is used up.

Why: v7 + S3b (`submissions/v12_s3b`) already ran on the 60 new tasks on Shrey's account: **9/60 solved**, 77% empty patches, 23% context overflows, 47% repeated identical calls (`data/runs/eval_s3b_new60/`). The fresh v7 arm on the same 60 tasks hung on Shrey's account: it ran ~10.5 session-h (~21 quota-h) with an empty log, and used up his quota. We need v7's number on the same tasks to judge S3b.

First `git pull`. The eval notebook now has two safeguards (commit "eval watchdog"): each 8-task chunk is abandoned after 60 min, and the evaluator is killed at `--budget-h` + 15 min, saving finished work as `partial_eval.tgz`.

Setup as before: Kaggle CLI in `.venv`, already signed in by you; never create, print or handle tokens; `export KAGGLE_OWNER=atulbaliyan`.

Steps:
1. Check nothing of yours is queued or running on Kaggle.
2. Build and push (expected ~2 session-h ≈ 4 quota-h; hard cap 3.75 session-h ≈ 7.5 quota-h):
   ```bash
   python3 kaggle/eval_l4/make_eval_notebook.py v7_noedit --valid-only --valid-file data/runs/eval_sweep_v2/new_tasks.json --budget-h 3.5
   .venv/bin/kaggle kernels push -p kaggle/probes/check_l4
   ```
   Check `kernel-metadata.json` first: id `atulbaliyan/check-l4`, no adapter sources; the notebook's settings show `session_budget_h: 3.5` and 60 tasks.
3. Poll `kaggle kernels status atulbaliyan/check-l4` every 10 min. **If it is still RUNNING after 3 h, tell me (Atul) at once** so I can cancel it from the notebook page (the CLI cannot cancel). Shrey's S3b arm on the same tasks took 1.85 h.
4. When it finishes, download `summary.json`, `run.log` and `eval_v7_noedit.tgz` (or `partial_eval.tgz`) into `data/runs/atul_eval_v7_new60/`. Do not commit the tgz; commit `summary.json`, `run.log` and `eval_v7_noedit/rep0/analysis/` (summary.json, per_task.jsonl, report.md).
5. Add the run to §8 of `docs/post_training_guide.md` with the actual quota charge.

Report: resolved/60, the report.md metric table (empty_patch_pct, tasks_overflow_pct, repeated_identical_calls_share_pct, explicit_submit_pct, tool_calls_median), the timing of each chunk from run.log, and any WATCHDOG or chunk-timeout lines.

Then STOP. No other runs, no leaderboard submission. Commit on a branch as before.
