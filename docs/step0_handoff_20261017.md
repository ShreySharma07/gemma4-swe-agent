# Fine-tuning plan: Step 0 done (CPU) and the GPU steps for Atul (Oct 17)

Plan: `docs/finetune_strategy_plan.md`. Basis: `docs/e0_e2_audit_20261016.md` and Atul's Step D analysis.

## What Step 0 built (all CPU, all in git)

| Piece | What it is | Result |
|---|---|---|
| Reasoning-preserving conversion | `posttrain/convert_public.py --reasoning --only-trajectories --save-rows` re-converted **the same 200 trajectories**. Identical to Step C's apart from the teacher's short decision text | 200/200 identical apart from reasoning; raw source rows saved (`data/runs/public_200_rows_20261017.tar.gz`, needed for E1) |
| **S3 data** | `posttrain/compact_episodes.py`: one grounded window per run. Searches that located the file → `sed -n` read of the edited region (real lines) → source edits → one existing-test run → submit. Scratch-file writes dropped. Informative reasoning kept as a leading `# ...` comment in the command, because Gemma's template puts message text *after* the call | 147/200 runs kept (40 edited a region never shown, 11 ungrounded, 2 over 8k). 0.74M tokens, median 4.9k per episode |
| S3 vs Step C (E2 re-run) | `data/runs/audit_e2_s3/e2_report.json` | Mid-run windows 80% → **0%**; ungrounded edits 79% → **0/182**; fix tokens 17% → **75%**; scratch-script tokens 56% → **0%**; masks clean |
| **S1 data** | v7's own 82 solved runs (old-54 tasks, disjoint from the new-60) with the local sandbox's `wrapt` warning stripped from 783 responses | 154 windows, 1.08M tokens, 21 tasks (3 held out) |
| **Behaviour gate** | `posttrain/gate_prefixes.py` → `data/runs/gate/prefixes.json`: 30 fixed decision points from v7 runs on tasks S1 doesn't train on. `kaggle/eval_l4/evalkit/gate.py` serves base + adapters in one vLLM and scores the next call | 8 file_in_view (expect edit), 8 test_failed (edit or read), 6 test_passed (submit), 8 early (read/search). Exactly 1 call per response. 4 samples each |
| Builders | `make_train_notebook.py` gives one notebook per adapter (`gemma4-train-<name>`); `make_eval_notebook.py --gate` | Both notebooks under 1 MB (Atul's check, now also in the eval builder) |

Hashes:

| File | SHA-256 |
|---|---|
| `train_s3_compact.npz` | `1893c901e1c6c8116e0696f32cdb13041fc7252a012cefd964674ae0021625b4` |
| `train_s1_own.npz` | `d3ade3498e4b3b10048fa983c3bda4ddbccdbd302cd0f028769d0d258a8655bc` |
| `gate/prefixes.json` | `d07b5ef760aa4dd917800c2eb7641337d62e4b095b9fc26a31583bd4acf95e12` |
| `public_200_rows_20261017.tar.gz` | `b0ceaac824c543d8a417b7a4e45bbd5c1d62bf8791c5eb1047942100afdf8607` |

**Not done:**
- **NF4 caching (saving the converted model to cut the 37-min load).** Not needed for these short runs; worth it before any long retrain.
- **The gate on GPU.** It has never run there. Its request format (OpenAI chat with tool history) is untested against vLLM, so expect possibly one fix-up run.

## Atul: E1 (CPU, in parallel with the GPU steps)

Replay the converted edits of the **S3 episodes** (the 147 in `data/runs/s3_compact/`) on the real repos at their base commits. Every kept edit must apply (`s.count(old) == 1`) and change the intended file. Raw rows with `model_patch` and instance IDs are in `public_200_rows_20261017.tar.gz`. Report any episode whose edit does not apply. Those get dropped before training.

## Atul: GPU steps (quota reset Oct 17; 4×L4 billed at 2× session time)

Set `export KAGGLE_OWNER=atulbaliyan` in every shell. Run steps one at a time, and push the next only after the previous one shows COMPLETE.

**Step 1a: train S3** (~1.4 session-h ≈ **3 quota-h**; lr 5e-5, 2 epochs, checkpoint every 9 steps):
```bash
python3 kaggle/train/make_train_notebook.py data/runs/train_s3_compact.npz --name s3_compact --lr 5e-5 --epochs 2 --grad-accum 2 --save-every 9 --budget-h 3
.venv/bin/kaggle kernels push -p kaggle/train/l4_s3_compact
```

**Step 1b: train S1**, same settings (~1.7 session-h ≈ **3.5 quota-h**):
```bash
python3 kaggle/train/make_train_notebook.py data/runs/train_s1_own.npz --name s1_own --lr 5e-5 --epochs 2 --grad-accum 2 --save-every 9 --budget-h 3
.venv/bin/kaggle kernels push -p kaggle/train/l4_s1_own
```

For each run, check `train_report.json`: `rc 0`, final export with 410 modules, validation loss down.

**Step 2: behaviour gate** (base vs S1 vs S3; ~0.6 session-h ≈ **1.2 quota-h**):
```bash
python3 kaggle/eval_l4/make_eval_notebook.py v7_noedit --gate --gate-adapters s1_own,s3_compact --adapter-from atulbaliyan/gemma4-train-s1-own --adapter-from atulbaliyan/gemma4-train-s3-compact
.venv/bin/kaggle kernels push -p kaggle/eval_l4/gate_kernel
```
Read `gate_report.json`:
- Each adapter's `verdict` (PASS = edit rate at file_in_view ≥ base, overall pass rate ≥ base, 0 batched responses, 0 errors).
- Per-type rates, the `early` row in particular: a high edit rate there means "edits before reading", which is the S3 risk.

**Total ≈ 8 quota-h** of Atul's 30. Stop after Step 2 and report. Step 3 (paired eval on the new-60, ~16 quota-h) runs on whichever account has quota, and only for adapters that PASS.

**Stop rule:** if both adapters FAIL the gate, no task evaluation and no retraining. Diagnose from the per-type table.

## Step 5 data, built ahead (Oct 18, CPU): S3-1k

Ready in case S3 passes the gate, so the scale-up can start without a 2–3 day delay. **Not for training yet.**

| | S3 (Step 1a) | **S3-1k** |
|---|---|---|
| Converted public runs | 200 | 1,150 distinct tasks (35,434 rows scanned; same settings: `--reasoning --one-per-task`, held-out repos excluded) |
| Grounded episodes kept | 147 | **840** (203 edited a region never shown, 94 ungrounded, 13 over 8k). Superset: all 147 Step-1a episodes are included |
| Tokens | 0.74M | **4.31M** (median 4,983 per episode; 12.8% supervised) |
| Train / val | | 798 / 42 episodes (5% of tasks) |
| E0 | | REPRODUCED (`data/runs/audit_e0_s3_1k/e0_report.json`) |
| E2 | 0/182 ungrounded, 75% of loss on edits | 0% mid-run windows; **edits 75.3% of loss**, 0 batched calls; audit flags 3/1,048 edits as "file not shown", all non-`.py` files (`.ini`, `README.md`, `.rst`) that the audit's `.py`-only read regex can't see; the compactor's grounding check passed them |
| File | `train_s3_compact.npz` | `train_s3_1k.npz` (1.74 MB) sha256 `d0b5e59edce9f21f1b9f02a9d55744f7638fe05535f3276c0fc09464b52ab8a9` |

Episodes: `data/runs/s3_1k_episodes_20261018.tar.gz`. The raw source rows (241 MB, needed for E1 on the new 693 episodes) stay local on Shrey's machine; ask if needed.

**Training it:** 1.74 MB is over the 0.6 MB embed limit, so it must be attached as a private Kaggle dataset (`--dataset owner/gemma4-train-s3-1k`), as in Step C. Cost at Step C's measured rate (~1 session-h per 1M tokens + 0.6 h load): **1 epoch ≈ 4.7 session-h ≈ 9.5 quota-h**, ~100 optimizer steps at batch 8. Over the 2 h rule: needs explicit agreement, and only after S3 passes the gate and Step 3's paired eval.
