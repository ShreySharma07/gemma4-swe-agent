You're helping me (Atul) run the next GPU steps of our Kaggle "Gemma 4 Developer Agent" project from this repo.

First `git pull`, then read `docs/step0_handoff_20261017.md` (the exact commands) and `docs/finetune_strategy_plan.md` (why). Rules from `docs/post_training_guide.md` §7.2 still apply.

Setup: Kaggle CLI in `.venv`, already signed in by me. Never create, print or handle API tokens. `export KAGGLE_OWNER=atulbaliyan` in every shell.

1. Verify data hashes against the table in the handoff (`shasum -a 256 data/runs/train_s3_compact.npz data/runs/train_s1_own.npz data/runs/gate/prefixes.json`). Stop if any differ.
2. Step 1a, train S3 (~3 quota-h): build with `kaggle/train/make_train_notebook.py ... --name s3_compact` exactly as written, check `kernel-metadata.json` (id `atulbaliyan/gemma4-train-s3-compact`, source < 1 MB), push. Poll `kaggle kernels status` every ~10 min. When COMPLETE, download outputs to `data/runs/atul_s3_compact/`. Pass: `train_report.json` rc 0, final export 410 modules, validation loss down.
3. Step 1b, train S1 (~3.5 quota-h): same procedure with `--name s1_own` → `data/runs/atul_s1_own/`. Push only after 1a is COMPLETE.
4. Step 2, behaviour gate (~1.2 quota-h): build with `make_eval_notebook.py v7_noedit --gate ...` as written, push `kaggle/eval_l4/gate_kernel`, download `gate_report.json` to `data/runs/atul_gate/`. This harness has never run on GPU: if requests error, read the log, explain the cause, propose a fix in `kaggle/eval_l4/evalkit/gate.py`, and ask me before re-pushing.
5. In parallel on CPU (no torch, no model): E1 replay of the S3 episodes' edits as described in the handoff. Report any edit that does not apply.

Then STOP and report: per-run quota used, val-loss curve, and from `gate_report.json` each adapter's verdict, pass rate, edit rate at file_in_view, batched/no-call counts, and the per-type table (flag the `early` row if edits appear there).

Rules:
- Never hand-edit a generated notebook on Kaggle; fix code under `kaggle/` and rebuild.
- Never push while another run is queued or running.
- Before every push, state expected quota hours and add a row to §8 of `docs/post_training_guide.md`. Ask me before anything over 2 session-h or if total use passes 10 quota-h.
- No paired task eval, no leaderboard submission, no retraining. Those come after the gate and need my OK.
- Commit only small results (`train_report.json`, `metrics.jsonl`, `run.log`, `gate_report.json`), not adapters or tarballs.
