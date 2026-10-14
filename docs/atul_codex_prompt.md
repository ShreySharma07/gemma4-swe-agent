You're helping me (Atul) run the GPU steps of our Kaggle "Gemma 4 Developer Agent" project from this repo.

Read `docs/post_training_guide.md` first, especially §7 (the handoff for me) and §8 (GPU quota ledger). Then:

1. Check my setup per §7.1:
   - Kaggle CLI in `.venv`, already signed in by me. Never create, print or handle API tokens.
   - `export KAGGLE_OWNER=<my-kaggle-username>` in every shell.
   - Confirm the generated `kernel-metadata.json` uses my username for the notebook id and `shreysharma07/...` for the inputs.
2. Run Step A (training pipeline check, ~0.7 GPU-h) exactly as written in §7.3:
   - build the notebook with `kaggle/train/make_train_notebook.py`, push with the Kaggle CLI;
   - poll `kaggle kernels status` every few minutes; when it finishes, download the outputs to `data/runs/<run-name>/`;
   - check the pass conditions.
3. If A passes, do Step B (serving check, ~1 GPU-h). **Stop and ask me before Step C** (~9–10 GPU-h).

Rules:
- Never hand-edit a generated notebook on Kaggle; fix the code under `kaggle/` and rebuild.
- Never push a new version while one is queued or running.
- Before every push, state the expected GPU hours and add a row to §8. Ask me before anything over 2 h.
- Nothing heavy runs locally (no torch/model runs on this machine); everything runs on Kaggle.
- If a run fails, read the log against the failure table in §7.5, explain the cause, and propose the fix before re-pushing.
- Commit only small results (`train_report.json`, `metrics.jsonl`, `run.log`, `summary.json`), not tarballs or adapters. Use clear messages.
