> **Updated Oct 21:** run **v15b_spec8k**, not v15_spec (v15 errored on the LB).

Shrey asks you (Atul) to run ONE GPU job: **v15b on the 60 new tasks**, the same protocol as your v14 run (`docs/atul_v14_new60_report_20261019.md`).

Why: v14 (thinking on) solved 30/60 locally vs v7 23/60 (your run) and scored **0.17** on the leaderboard (best so far; v7 0.15). `submissions/v15b_spec8k` = v14 + ONE change (v15, which also cut `max_output_tokens` to 6144, failed on the leaderboard with "Notebook Threw Exception" after ~4 h; v14's longest response was 5,913 tokens, so 6144 likely truncated a tool call on a hidden task. v15b keeps v14's 8192 to isolate the prompt):
Prompt rules adapted from the public 0.24 notebook: treat the issue as the specification (exact names/messages/types/edge cases), apply the change to every parallel implementation, scratch files only in /tmp, `git status --short && git diff` before submit, and never use the graph tools (`search_similar_code` can return >100k chars). v14 failures: 15 wrong/incomplete source fixes, 6 with stray scratch files, 4 with only scratch files, 5 empty.
Sampling (incl. max_output_tokens 8192, thinking 4096), tools, 8 min / 28 calls / 240 s / 80 turns are identical to v14, so v15b vs v14 on the same 60 tasks measures the prompt change alone. Also report any truncated / unparseable tool calls or uncaught exceptions in run.log.

Your quota: ~13.0 h available (16.96 of 30 used after v14). v14 took 4.02 session-h ≈ 8.0 quota-h. This run uses `--budget-h 5`: the evaluator is killed at 5.25 h (≤ ~10.5 quota-h), and new chunks stop being scheduled when fewer than ~1.2 h remain, so if v15 is much slower than v14 the last chunk may not run. Report coverage if so.

Setup as before: Kaggle CLI in `.venv`, signed in by you; never create, print or handle tokens; `export KAGGLE_OWNER=atulbaliyan`. `git pull` first.

Steps:
1. Check nothing of yours is queued or running (one 4×L4 session per account).
2. Build and push:
   ```bash
   python3 kaggle/eval_l4/make_eval_notebook.py v15b_spec8k --valid-only --valid-file data/runs/eval_sweep_v2/new_tasks.json --budget-h 5
   .venv/bin/kaggle kernels push -p kaggle/probes/check_l4
   ```
   Check before pushing: id `atulbaliyan/check-l4`, no adapter sources, `session_budget_h: 5.0`, 60 tasks, and the embedded `configs/sampling.yaml` has `max_output_tokens: 8192` and `thinking_budget: 4096`.
3. Poll every 15 min. If still RUNNING after 5.5 h, tell me (Atul) so I can cancel it from the notebook page.
4. When it finishes, download `summary.json`, `run.log`, `eval_v15b_spec8k.tgz` (or `partial_eval.tgz`) into `data/runs/atul_eval_v15_new60/`; commit `summary.json`, `run.log` and `eval_v15b_spec8k/rep0/analysis/` only. Add the run to §8 of `docs/post_training_guide.md` with the actual quota charge.

Report (same as v14): resolved/60 and coverage; empty_patch_pct, tasks_overflow_pct, repeated_identical_calls_share_pct, explicit_submit_pct, tool_calls_median; BudgetExceeded events; per-chunk minutes; WATCHDOG/timeout lines; and **paired overlap with v14** (`data/runs/atul_eval_v14_new60/.../task_results.jsonl` or the v14 archive, key `instance_id`): both / v15 only / v14 only / neither, exact McNemar p. Also classify v15's failed patches like the v14 analysis (empty, scratch-only, source + stray scratch, source wrong/incomplete) and count context-overflow tasks.

Then STOP. No other runs, no leaderboard submission. Commit on a branch and open a PR as before.
