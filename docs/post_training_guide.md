# Post-training Gemma-4-31B on Kaggle 4×L4: a living guide

This explains how we tested whether the competition model can be fine-tuned on our hardware: what each piece does, what we measured, what broke, and how we fixed it. **It's updated as fine-tuning progresses.** Newest decisions go in the log at the end.

Code: `kaggle/probes/train_feasibility/probe_train.py` (the probe), `make_notebook.py` (wraps it for the L4 notebook), `make_smoke_kernel.py` (CPU smoke test). Results: `docs/experiments.md` rows p0b, p0b-v4, e2. Raw outputs: `data/runs/p0b_*`.

---

## 1. The problem in one paragraph
- **Hard constraints:**
  - The scorer serves **one fixed model**: `gemma-4-31b-it-qat-w4a16-ct`, a 31-billion-parameter Gemma 4 whose weights are stored in 4 bits.
  - The only way to change its behaviour is a **LoRA adapter**: small extra matrices loaded next to the frozen model by vLLM, the serving engine.
  - Training has to happen on Kaggle: **4 NVIDIA L4 GPUs, 24 GB each (about 22 GB usable)**, offline, 12-hour sessions, about 30 GPU-hours per week.
- **What the probe answered:**
  1. Can we train a LoRA on this model at all?
  2. How long can each training sequence be?
  3. How fast does it train?
  4. Will the adapter load on the scorer?
  5. How different is our training copy of the model from the one the scorer serves?

## 2. Concepts you need (short)
| Term | What it is | Why it matters here |
|---|---|---|
| **LoRA** | Instead of changing a weight matrix W (huge), learn two thin matrices A (r × in) and B (out × r) and use W + (α/r)·B·A. r = "rank", e.g. 16 | 31B params frozen; we train ~122M. The adapter file is small (≈245 MB at r16 in bf16) |
| **Target modules** | Which matrices get LoRA: q/k/v/o (attention) + gate/up/down (MLP) in every layer | Gemma-4-31B has **410** such matrices (global layers share K and V, so they have no v_proj) |
| **Quantisation** | Storing weights in fewer bits | The served model is **W4A16**: weights 4-bit, maths in 16-bit |
| **QAT** | Quantisation-aware training: Google trained the model knowing it would be rounded to 4 bits | The served int4 weights are a deliberately tuned version of the model, not plain rounding |
| **NF4 / QLoRA** | A 4-bit format (bitsandbytes) that PEFT can train on. QLoRA = LoRA on top of an NF4 base | We can't train directly on the served compressed-tensors format, so we train on NF4 and serve on W4A16. That mismatch had to be measured |
| **FSDP** | Fully Sharded Data Parallel: each of the 4 GPUs stores ¼ of the model and they swap pieces as needed | The whole 4-bit model is ~17 GB, so each GPU keeps only ~4.6 GB and each trains on its own sequence |
| **Gradient checkpointing** | Don't keep every layer's intermediate results for the backward pass; keep only each layer's input and recompute the rest | Cuts memory massively, costs ~30% extra compute |
| **Activation offload** | Move those saved layer inputs to CPU RAM (`save_on_cpu`) | Kaggle's L4 machine has **188 GB** RAM, so long sequences can park activations there |
| **Attention kernels (SDPA)** | PyTorch's fast attention implementations: FlashAttention, memory-efficient, math | Gemma-4's global layers use **head_dim 512**. FlashAttention only supports ≤256, and "math" uses T×T memory (OOM). So we force the **memory-efficient** kernel |
| **Chunked loss** | Compute next-token loss 256 tokens at a time | The vocabulary is 262k: logits for 8k tokens in fp32 would be ~8.6 GB |
| **Tied embeddings** | The input embedding matrix is reused as the output layer | 2.8 GB that must be fully present when the loss is computed, which affects how FSDP can split the model |
| **PEFT export** | Saving the adapter as `adapter_model.safetensors` + `adapter_config.json` | Key names must match what vLLM expects: `base_model.model.model.language_model.layers.N.<module>.lora_A.weight` |

## 3. How the probe works
Every GPU stage runs as its own process, so nothing is left on the GPUs between stages. That was a lesson from run 2.

1. **Environment (offline):** L4 sessions have no Internet. Two CPU kernels with Internet build the software beforehand:
   - `gemma4-scorer-env`: the scorer's exact Python environment (torch 2.10, transformers 5.13.1, vLLM 0.19.1), 5.8 GB.
   - `gemma4-train-wheels`: adds peft 0.21.2 and bitsandbytes 0.50.2, checked to import inside the scorer env.

   The L4 notebook restores both, then re-launches itself inside that environment.
2. **Model:** Google publishes the QAT weights in bf16 (`gemma-4-31b-it-qat-q4_0-unquantized`, 62.5 GB). Each GPU loads them and quantises to NF4 as they stream in. This takes ~9–11 min, the biggest fixed cost per training session.
3. **LoRA + FSDP:** apply LoRA r16 to the 410 language-model matrices (not the vision tower). Then wrap with FSDP:
   - one unit per decoder layer;
   - the vision tower in its own unit;
   - everything else in the root;
   - LoRA weights cast to bf16 (FSDP needs one dtype per unit).
4. **Loss inside the forward pass:** under FSDP the tied embedding is only fully present during the forward pass. So the model wrapper computes the chunked, soft-capped cross-entropy inside its own forward, with a custom autograd function that keeps only dLoss/dHidden.
5. **Sweep:** for each sequence length, 2 optimiser steps (the first is a warm-up). Record step time, tokens/second across the 4 GPUs and peak memory. From 8k up, activations are offloaded to CPU.
6. **Reference predictions:** before training, run 8 real code texts through the NF4 model and save its next-token predictions.
7. **Served-model comparison:**
   - Run the same token ids through the bf16 QAT weights in HF.
   - Run them through the **served engine** (vLLM 0.19.1 on the W4A16 checkpoint, tensor-parallel over 4 GPUs).
   - Compare top-1 next-token agreement and log-prob differences.
8. **Smoke test first:** every code change runs on a free Kaggle CPU kernel with a tiny random Gemma-4 (same architecture, 6 layers) and FSDP on 2 CPU processes. That catches API and shape bugs before spending GPU time. (Never on the Mac: it crashed Python once.)

## 4. What happened, run by run
| Run | What we learned | Fix for the next run |
|---|---|---|
| v2 (Oct 9) | Layer-split across the 4 GPUs (no FSDP) works but runs at only 82 tok/s at 8k, because only one GPU works at a time. 16k OOMs on the last GPU. head_dim-512 attention works. The adapter's module names match vLLM's. 188 GB host RAM. FSDP crashed with no traceback; the vLLM check failed on a missing `__main__` guard | Isolate each stage in its own process, keep per-rank logs, add the guard |
| v3 (Oct 10) | FSDP loads and shards (4.6 GB/GPU) but 8k OOMs at 20 GB. NF4 vs served agreement is only 82.7% | Smaller loss chunks, separate vision unit, start at 4k, offload from 8k, add a bf16 reference |
| **v4 (Oct 10)** | **4k: 421 tok/s (15 GB). 8k with offload: 244 tok/s (18.5 GB). 16k+: OOM even with offload.** Agreement with served model: NF4 81.7%, bf16 QAT 85.1% | See decisions |

**Why the 82–85% agreement is fine.** Even the original bf16 weights only agree with the served int4 model 85% of the time, so most of the gap comes from serving in 4 bits, not from our training format. NF4 costs ~3.5 points on top. An adapter trained on NF4 learns *directions* (e.g. "write tool calls in the native format") that should carry over. Whether they do is checked at evaluation time on the real served stack.

**Why 16k fails even with offload.** Offload moves the *saved* activations to CPU. During the backward pass, each layer must still recompute its full activations on the GPU. At 16k tokens, Gemma's MLP (21,504 wide) alone needs several GB per layer. That's added to the model shard, the 2.8 GB tied embedding that must be present, and FSDP's prefetched layer.

## 5. Decisions log (newest first)
| Date | Decision | Reason |
|---|---|---|
| Oct 18 | **S3-1k built ahead of Step 5 (CPU only):** 1,150 public runs converted → **840 grounded episodes, 4.31M tokens** (`train_s3_1k.npz`, sha256 `d0b5e59e…`), superset of Step 1a's 147. E0 reproduced; E2: edits 75% of loss, 0 mid-run, 0 batched. Not trained | So a scale-up can start right after S3 passes the gate and paired eval. Training cost ≈ 9.5 quota-h/epoch; needs a private dataset and explicit agreement (>2 h) |
| Oct 17 | **Fine-tuning Step 0 done (CPU); Step 1–2 handed to Atul** (`docs/step0_handoff_20261017.md`, Codex prompt `docs/atul_codex_step1.md`). Two new adapters instead of retraining on Step C data: **S3** (147 public runs, each compressed to one grounded ≤8k episode: locate → `sed -n` read of the edited region → source edit → existing-test run → submit; scratch scripts dropped; informative teacher reasoning kept as a `# …` comment) and **S1** (v7's own 82 solved runs, old-54 tasks). Dose cut: lr 5e-5 (C: 1e-4), ~36 optimizer steps (C: 122), checkpoints every 9. New 30-prefix behaviour gate (base vs adapters, next-call kind) must PASS before any task eval | E2 showed Step C trained edits without the code in view (79%), started 80% of windows mid-run, and spent 56% of loss on scratch scripts; S3 re-audit: 0/182 ungrounded, 0% mid-run, 75% of loss on fixes. S1 is the control ("does our own style alone help?"). Lower dose limits drift from v7's policy. Gate catches collapse for ~1 quota-h instead of a 9 h task eval |
| Oct 16 | **Step D (Atul): v7 + C adapter = 0/114 resolved, 114/114 empty patches. Adapter rejected; do not submit v8.** Format intact (0 drift, edit syntax fine, serving fine) but behaviour collapsed: grep/sed reading 54% → 0% of calls, pytest 1% → 44%, `cd /workspace && …` 47%; opening ritual `cat README / pytest.ini / pyproject`, then the same pytest command repeated (median 6 identical repeats per task); 28 context overflows, 7 timeouts. 8.99 quota h for 4.49 session h (2.00×); Atul has 2.88 h left; D0 not run | Low val loss (0.29) measured imitation, not acting. Likely causes, by evidence: (1) windows 2–5 start mid-run with no history, so the model learned 'after the task, run tests/read files' without what those actions found or when to edit (fits the identical repeats); (2) the converter dropped Qwen's `think`/prose, i.e. the decision points; (3) OpenHands style (README/configs/tests first) doesn't fit 40 calls / 4.5 min. Step A's adapter on our own v7 runs did not collapse. Redesign before any retraining (CPU): keep reasoning as short text before calls, whole-run context (harder output truncation or a carried summary), mix in our own solved runs |
| Oct 16 | **Step C details + Step B PASS** (pulled from `atulbaliyan/gemma4-train-lora` and `atulbaliyan/check-l4`; Atul's committed folders `data/runs/atul_step_c_corrected_20261015_v2`, `atul_step_b_20261015_v1`). Val loss start 0.772 → step40 0.313 → step80 0.291 → end 0.288: most of the gain in the first 40 steps, then a plateau, no overfitting. Train loss 0.89 → ~0.4; peak 18.9 GB; exports at steps 40/80/120/final all 410 modules. **Model load took 37 min** (probe: 11). Step B (Step A's adapter, 8 tasks): adapter copied from the training output, vLLM up with `adapters: ['conv_lora']`, health all true, 2/8 (not meaningful: trained on these tasks); 1 task hit the 100-turn limit. Quota: 4×L4 is charged **2× session time** (Atul measured C exactly: 7.82 session h = 15.64 quota h; this also explains Shrey's ~30 h for ~18 h of runs) | The adapter path from training to the scorer stack is proven. Next (approved Oct 16): Atul's paired 24-task pilot instead of Step D/D0 (`docs/step_c_evaluation_proposal_20261015.md`): v8 vs a fresh v7 on 24 newly valid tasks, ≤10.4 quota h; the full 114-task comparison follows on Shrey's quota after the Saturday reset if the pilot isn't negative |
| Oct 15 | **Step C v2 PASS / COMPLETE:** rc 0, all 122 planned optimizer steps, no budget stop; validation 0.7718 → 0.3133 → 0.2909 → 0.2883 → 0.2882. All checkpoints/final/serving files pass 410-module / 820 BF16 rank-16 tensor integrity and exact shape checks; final/serving SHA-256 matches. Global peak 18.88 GiB; raw median 294.8 tok/s, steady median 295.6 (247.7–324.0). Three intervals include validation/export overhead. Existing group-of-four scheduler used 972/974 train and 40/43 val windows while retaining all 190/10 tasks. Actual 7.8203 session h / 15.64 quota h, versus expected 9–10 / 18–20; quota 11.87 h remains. All outputs preserved and final report `docs/step_c_20261015_report.md`. No retry, D, or leaderboard run; monitor paused. | A/B PASS remain preserved; corrected submission explicitly approved after the earlier HTTP-400 rejection (zero GPU cost). No additional experiments authorized. |
| Oct 15 | **C failed to launch: SaveKernel returned HTTP 400; no retry performed.** Shrey approval was explicitly confirmed; A/B COMPLETE; 27.51 quota h available; exact 200-task data, 10.5 h budget, val/save every 40, and corrected instrumentation verified. Free CPU smoke PASS confirmed SUM tokens / MAX elapsed across two ranks. C's push was rejected; old A lastRunTime/status remains unchanged and quota stays 27.51 h. | Likely source-size rejection: CLI-normalized C source 4,560,176 bytes (embedded NPZ 3,385,750 bytes), versus successful A 558,425 bytes; matches reported Kaggle 1 MB source-limit failures (https://www.kaggle.com/discussions/product-feedback/236663). Error body unavailable, so this is an inference. Builder now guards source size; prepared unchanged data as private dataset `atulbaliyan/gemma4-train-public-200` and a 45,840-byte attached-input notebook. Neither dataset upload nor corrected re-push performed; await authorization under no-retry instruction. Report: `data/runs/atul_step_c_20261015_v2/summary.json`. |
| Oct 15 | **Step B PASS; C prepared but not launched.** `atulbaliyan/check-l4` v1 uses the unchanged Step A adapter, verified by matching SHA-256 in downloaded `submission.zip`. COMPLETE, exit 0, adapter copied and declared, vLLM started in 409 s, all health checks true. Eight tasks/traces, tools on all 8, tests on 7; one 100-turn-budget error, 2/8 resolved (not a generalization score). Runtime 31.91 min, quota 1.06 h, 27.51 h left. | Downloaded archive and `run.log` independently verified. C's exact 10.5 h-budget / val-every-40 / save-every-40 notebook is staged in `data/runs/atul_step_c_prepared_20261015/`, preserving existing generated files. Explicit Shrey approval for 9–10 session h (roughly 18–20 quota h observed accounting) is still required. |
| Oct 15 | **Step A reassessed PASS without rerunning, per Atul's instruction.** Exact schedule reconstruction matches all four logged plans: 40 windows / 270,421 input tokens. Using every rank's rounded rate to bound elapsed time and taking the slowest rank gives step throughput intervals 271.14–271.23, 287.55–287.66, 295.52–295.61, 276.95–277.04, 257.50–257.62 tok/s; all within 240–420. Source logging now sums tokens, takes maximum synchronized elapsed time and memory across ranks, and records token counts/durations. Original metrics and initial validation are preserved. | `analysis/reconstruct_training_throughput.py` verifies schedule, plans, rank-0 metrics equality, source hashes, and rounding bounds using stdlib only; synthetic uneven-workload/slow-rank checks PASS. B authorized using A v1's unchanged adapter; C still needs B PASS and explicit Shrey approval. |
| Oct 15 | **Atul Step A v1: training/export succeeded; strict throughput gate FAIL, so B/C were not launched.** `atulbaliyan/gemma4-train-lora`: rc 0, val 0.2557 → 0.1818, final 410 modules / 820 tensors, shapes valid, 245 MB; peak logged memory 17.61 GB. Logged throughput 175.3–259.2 tok/s (median 193.2), below §7.3's approximate 240–420 range. Reconstructing the seed-0 schedule identifies a metric bug: rank 0 receives each group's shortest window, while `Trainer.run` reports `world * rank0_tokens / dt`. All-rank token counts imply approximately 257.6–295.6 tok/s instead. Proposed fix: sum token counts across ranks and use maximum elapsed time before logging; validate on CPU before any authorized GPU retry. | Do not silently replace the documented gate with a corrected estimate or retry a GPU job. No §7.5 failure signature, OOM, or traceback; apparent stall was delayed log delivery. Quota charged 1.43 h versus 42.7 min notebook execution (~0.71 session h); use quota accounting for future capacity checks. Results: `data/runs/atul_step_a_20261015_v1/`. |
| Oct 14 | **Local valid set 54 → 114 tasks** (CPU only): overlays kernel `gemma4-test-overlays` adds test-only packages for every task sandbox and per-task Starlette/Pygments pins dated to each task's issue; gold/empty sweep: fastapi 59, rich 45, requests 9, httpx 1, none of the old 54 lost. Now the default for evals (`make_eval_notebook.py` embeds `data/runs/eval_sweep_v2/valid_tasks.json` and attaches the overlays) | ~2× the tasks → roughly half the noise in the local score, and fastapi (67 tasks) is finally measured. Comparisons need v7 on the 60 new tasks (`data/runs/eval_sweep_v2/new_tasks.json`) |
| Oct 13 | **Training pipeline built; CPU smoke PASS; first data set = 200 public runs, all windows** (user: nothing that trades learning). `posttrain/tokenize_windows.py` → `.npz` (loss only on model-written tokens; val = 5% of tasks): 200 tasks, 974 train + 43 val windows, 7.37M tokens (16% trained), 3.4 MB, embedded in the notebook. `kaggle/train/trainkit` (config/data/model/export/worker/launcher/main) + `make_train_notebook.py` (multi-cell; `--smoke` CPU kernel, else L4 kernel `gemma4-train-lora`): probe v4 recipe, AdamW lr 1e-4, warmup 5 + cosine, batch 8 windows (4 ranks × accum 2), grad clip 1.0, offload above 4k tokens, loss on trained positions only (chunked, softcap), val loss start/end, per-layer export in the p0-served PEFT format. Smoke (tiny Gemma-4, 2 CPU ranks): train → val → export OK, 41 modules / 82 tensors with the vLLM key names. Eval side: `Submission.fetch_adapters()` copies `adapters/<name>` from an attached training run; `submissions/v8_lora_smoke` = v7 + `adapter: conv_lora` | ~8.4 GPU-h per epoch at the probe's 244 tok/s. **Blocked: weekly 30 GPU-h quota used up** (evals + 5× rejection sampling); next = L4 pipeline check (40 own windows → serve in v7 on 8 tasks), then the 200-run training |
| Oct 13 | **Converter built + 50-trajectory sample passes review** (`posttrain/convert_public.py` → ATIF in v7's exact calls/JSON responses → `render_atif.py --agent`). Yield 50/295 resolved (17%; mostly over the 40-call budget) → ~5k usable trajectories in the full set. 1,814 converted calls, **0 drift**; edit scripts round-trip exactly (quotes, backslashes, triple quotes). **Cost:** 5.3 windows (~7.7k tokens) per trajectory = 44 GPU-h per 1k trajectories. Selecting the conversion phase (windows from the first edit, max 2) = 15k tokens/trajectory → **~500 trajectories ≈ 1k windows ≈ 8.5 GPU-h** | Matches the target (edit → test → submit) and the 9–10 GPU-h budget. Known gaps: tree is one level (OpenHands lists 2 levels from /workspace); final messages are long summaries (cut at 1.5k chars) |
| Oct 13 | **External data source = `nebius/SWE-rebench-openhands-trajectories`** (CC-BY-4.0; Qwen3-Coder-480B, Apache-2.0; 67k trajectories, 32k resolved, 3.8k issues, 1.8k Python repos, SWE-bench Verified excluded). Competition rules allow public, free external data; the winner Apache-2.0 obligation is compatible (attribution). OpenHands tools: `execute_bash`, `str_replace_editor` (view/create/str_replace), `think`, `finish`, all mappable to v7. **Constraint found:** resolved runs are long (calls p10/p50/p90 = 41/59/84; tool output ~150k chars per run), only ~9% fit v7's 40-call budget → filter to ≤40 calls (~2.9k runs est.), truncate tool outputs, drop fastapi/rich/requests/httpx (held-out eval) | Learning 59-call habits under a 40-call / 4.5-min budget would grow v7's biggest failure (no patch before the limit). SWE-smith trajectories (MIT, but Claude-generated) and nebius SWE-agent trajectories (Llama-3.1 licence terms) set aside |
| Oct 13 | **Own-run rejection sampling is not enough data on its own.** pass@5 = 21/54 (pass^5 9/54); 82 solved trajectories → 185 windows from 21 tasks. Renderer now takes the agent's declared tools (`--agent agent.yaml`), since v7 declares no edit_file | Training and evaluating on the same 21 public tasks would measure memorisation; the hidden set is private repos. Needs many more distinct tasks (external trajectories converted to v7's tool format) or a held-out split |
| Oct 12 | **Local metric = resolved on the 54 valid tasks** (gold passes, empty fails, CPU sweep `--sweep`); GPU runs use `--valid-only`. v7 = 18/54 (v5 14, v2 12), same order as the LB. **Step 2 started: rejection sampling** v7 × 5 reps on the 54 (check-l4 v10) -> pass@k + solved trajectories for the conversion LoRA | 75/129 tasks can't pass even with the gold patch in the scorer's env (fastapi: missing test-only deps + Starlette 1.6), so they only add noise and waste GPU time |
| Oct 12 | **Task sandboxes made scorer-faithful + CPU preflight gate.** Each task sandbox is a fresh Python 3.13 venv (the scorer's sandbox is python:3.13-slim) seeing only the scorer site (`scorer_site.tgz` from gemma4-devset-build) plus a pip-only dir, with the competition's `sandbox/setup.py`; Debian venv's `sys.exit` is turned into an exception (`evalkit/compat.py`). Every env change must pass `--preflight` (free CPU kernel: gold patches through the official Evaluator, no model) before a GPU run. **Preflight PASS: gold 6/6 resolved, empty patch unresolved** | First full run died before task 1: swegemma's subprocess sandbox inherits the host env (our Py3.12 harness venv: no pytest) and Debian's venv exits the process when ensurepip is missing. Known noise: Kaggle's `sitecustomize` prints a `wrapt` warning on stderr in the sandbox |
| Oct 12 | **Local eval = `evalkit` package, built into a multi-cell notebook** (`kaggle/eval_l4/evalkit/`, `make_eval_notebook.py`): restores the Py3.12 scorer venv, upgrades only the wheels the Oct 9 wheelhouse changed (swegemma 0.2.11, adk-submission 0.2.13), runs the official Evaluator on any `submissions/<name>`, with `repeats` for pass@k / rejection sampling. Shakedown first (8 tasks), then full runs | Every training candidate needs a local signal: the leaderboard allows one pending submission and queues ~33 h. The old single-script runner broke on Python 3.13 and the removed `swegemma.models.discovery` |
| Oct 12 | **Base agent for training = v7_noedit (LB 0.15).** Training trajectories must use v7's tool set: no edit_file; edits via python snippet-replace through run_command or write_file. The first LoRA's target moves from "format repair" to **fix conversion** (diagnose → edit → targeted test → submit) | v7 doubled v2 (0.08 → 0.15) without training, so a format adapter for edit_file is no longer needed |
| Oct 11 | **Training data rendered with `enable_thinking=False`** | adk-submission 0.2.13: `thinking_budget: 0` sends `enable_thinking: false`; `include_thoughts: false` alone now leaves thinking on with a 4,096-token budget. Train/serve prompts must match token for token |
| Oct 10 | **Measured drift (m1-drift): it is an `edit_file` problem**: 61.5% of edit_file calls drift vs ≤1.2% for run_command/write_file. Two levers: (1) zero-training variant without edit_file (edits via write_file / patch script), (2) LoRA on native-format edit_file calls | 129 v5 traces, 8,576 calls; drifted calls error 82.6% of the time; tasks with drift resolve 8% vs 30% |
| Oct 10 | **First adapter = tool-format LoRA** | The re-render check (e2) showed malformed tool calls are Gemma drifting into JSON-style quoting (`"start_line":10`), which vLLM parses into garbage keys. The scorer controls vLLM, so only the model can fix it |
| Oct 10 | **Training base = NF4 of the bf16 QAT weights**; HQQ/exact-int4 deferred | NF4 is only ~3.5 points worse than bf16 at matching the served model, and it is the documented FSDP-trainable path |
| Oct 10 | **Sequences ≤ 8k tokens**; prefer 4k windows when possible (1.7× faster, no offload) | Measured envelope (v4). Long trajectories are cut into windows that end on assistant turns, with the loss only on the assistant tokens of the window |
| Oct 10 | **Budget rule: ~1k windows of 8k ≈ 9 GPU-h per epoch**; plus ~10 min model load per session | From 244 tok/s. One serious run ≈ one third of a week's quota, so every run is gated by a cheap offline check first |
| Oct 7 | Adapters are served unmerged by vLLM via `adapter:` in agent.yaml | LoRA serving probe (p0) |

## 6. Where things stand (kept current, Oct 14)
| Step | Status | Where |
|---|---|---|
| Base agent | **v7_noedit**: LB 0.15; local 16.4/54 mean over 5 reps (pass@5 21/54) | `submissions/v7_noedit` |
| Local eval | Works on 4×L4; metric = resolved on the **114** valid tasks (54 before the overlays) | `kaggle/eval_l4/` (§7.3) |
| Training data | 200 public tasks → 974 train + 43 val windows, 7.37M tokens, 3.4 MB (embedded in the notebook) | `data/runs/train_public_200.npz` |
| Smoke data | our own 185 v7 windows (21 tasks) | `data/runs/rs_v7_valid/train_v7_solved.npz` |
| Training code | Step A GPU pipeline PASS after verified throughput correction Oct 15; original adapter reused for B | `kaggle/train/`; `data/runs/atul_step_a_20261015_v1/` |
| Next GPU steps | pipeline check → serving check → 200-task training → eval (§7.3) | Atul's Kaggle |
| Valid set | **114/129** with the overlays (done Oct 14, CPU) | `data/runs/eval_sweep_v2/` |

## 7. Handoff: running the GPU steps on Atul's Kaggle
Shrey's weekly GPU quota is used up until Saturday 00:00 UTC, so Atul runs the GPU steps on their own account.
Everything is generated from this repo; **never edit a generated notebook on Kaggle**, change the code here and rebuild.

Atul's Kaggle username: **atulbaliyan** (notebooks `atulbaliyan/gemma4-train-lora`, `atulbaliyan/check-l4`; both are visible to Shrey's account).

### 7.1 One-time setup
**Shrey (on kaggle.com, signed in):** share these with Atul (open each → *Share* → add Atul as a collaborator, "can view"):
- notebooks `gemma4-scorer-env`, `gemma4-train-wheels`, `gemma4-devset-build` (their **outputs** are inputs: the scorer's Python 3.12 env, peft/bitsandbytes wheels, the task sandbox's package set `scorer_site.tgz`);
- notebook `gemma4-test-overlays` (output = test-only packages + per-task version pins that make 114/129 public tasks verifiable);
- dataset `test-deps-wheels`.

Nothing is rebuilt: Atul's notebooks keep reading them as `shreysharma07/...`.

**Atul:**
1. Clone the shared git repo (Atul is a collaborator) and pull before each step; commit results as in §7.4.
2. Join the competition *Google – The Gemma 4 Developer Agent* and accept its rules (required to attach its data and to get 4×L4).
3. Install the Kaggle CLI (`python3 -m venv .venv && .venv/bin/pip install kaggle`) and sign in with **your own** Kaggle API token (Kaggle → Settings → API → *Create New Token*; never share it).
4. In every terminal you use for this, set your username so the builders name the notebooks after your account:
```bash
export KAGGLE_OWNER=<atul-kaggle-username>
```
5. Check: `python3 kaggle/train/make_train_notebook.py data/runs/rs_v7_valid/train_v7_solved.npz --max-windows 40 --budget-h 2` must print `<atul>/gemma4-train-lora` in `kaggle/train/l4_kernel/kernel-metadata.json`.

### 7.2 Rules (learned the hard way, Oct 12–13)
- **Quota:** 30 GPU-h per week, reset Saturday 00:00 UTC. Write every run's expected and actual hours in §8 **before** pushing; anything over 2 h is agreed with Shrey first.
- **One version at a time:** don't push a new version of a notebook while the previous one is queued or running (it may run both).
- **Close the Kaggle editor:** an open GPU notebook in the browser burns quota while idle. Push from the CLI and only *view* results.
- **Builders check every code cell compiles** before writing the notebook; if a build fails, fix the code, don't hand-edit the notebook.
- Long runs stop themselves before the 12 h session cap (`--budget-h`), and the eval aborts after its first 8 tasks if something systemic is wrong (health check), so a broken run costs ~1 h, not 10.

### 7.3 The GPU steps, in order (each gated)
Watch any run with (replace the slug):
```bash
.venv/bin/kaggle kernels status $KAGGLE_OWNER/gemma4-train-lora
```
Fetch its outputs into the repo when it shows `COMPLETE` or `ERROR`:
```bash
.venv/bin/kaggle kernels output $KAGGLE_OWNER/gemma4-train-lora -p data/runs/<run-name>
```

**Step A: training pipeline check (~0.7 GPU-h).** Trains 40 of our own v7 windows; proves the full model trains on 4×L4 and the adapter exports.
```bash
python3 kaggle/train/make_train_notebook.py data/runs/rs_v7_valid/train_v7_solved.npz --max-windows 40 --budget-h 2 --name conv_lora
.venv/bin/kaggle kernels push -p kaggle/train/l4_kernel
```
Pass when `train_report.json` shows:
- `"rc": 0`;
- `val` end loss < start loss;
- one export with `"n_modules": 410` and `"shapes_ok": true`;
- in `train_out/metrics.jsonl`, `tok_per_s` ≈ 240–420 and `peak_gb` < 21.

If the log's first lines say `need GPU L4 x4, got [...T4...]`, the notebook didn't get L4s (it aborts in seconds). Tell Shrey; the fallback is to run training inside your `check-l4` competition notebook.

**Step B: serving check (~1 GPU-h).** Serves that adapter in v7 (`submissions/v8_lora_smoke` = v7 + `adapter: conv_lora`) on 8 valid tasks. The adapter is taken straight from Step A's output, so don't re-run Step A in between.
```bash
python3 kaggle/eval_l4/make_eval_notebook.py v8_lora_smoke --valid-only --max-tasks 8 --budget-h 2 --adapter-from $KAGGLE_OWNER/gemma4-train-lora
.venv/bin/kaggle kernels push -p kaggle/probes/check_l4
```
Pass when `run.log` has:
- `adapter conv_lora <-`;
- `adapters: ['conv_lora']`;
- `vLLM up`;
- `HEALTH after first chunk` with all checks `true`.

The score doesn't count: the adapter was trained on these very tasks.

**Step C: real training (~9–10 GPU-h; agree with Shrey first).** 200 public tasks, all windows, one epoch.
```bash
python3 kaggle/train/make_train_notebook.py data/runs/train_public_200.npz --budget-h 10.5 --name conv_lora --val-every 40 --save-every 40
.venv/bin/kaggle kernels push -p kaggle/train/l4_kernel
```
Check `train_report.json`:
- the validation loss (10 held-out tasks) falls and does not turn back up;
- `"rc": 0`;
- a final export with 410 modules.

**Step D0: v7 baseline on the 60 newly valid tasks (~4 GPU-h).** v7 has 5 runs on the original 54 tasks but none on the 60 tasks the overlays made verifiable; this fills the gap so Step D compares like with like.
```bash
python3 kaggle/eval_l4/make_eval_notebook.py v7_noedit --valid-only --valid-file data/runs/eval_sweep_v2/new_tasks.json --repeats 2 --budget-h 10.5
.venv/bin/kaggle kernels push -p kaggle/probes/check_l4
```

**Step D: does it help? (~8 GPU-h; 2 passes over 114 tasks, can be split into two sessions).** v7 + the trained adapter.
```bash
python3 kaggle/eval_l4/make_eval_notebook.py v8_conv_lora --valid-only --repeats 2 --budget-h 10.5 --adapter-from $KAGGLE_OWNER/gemma4-train-lora
.venv/bin/kaggle kernels push -p kaggle/probes/check_l4
```
- **v7 reference:** on the old 54 tasks, 16, 16, 16, 16, 18 (mean 16.4). On the 60 new tasks, Step D0.
- **Gate for a leaderboard submission:** the adapter's mean beats v7's mean on the same 114 tasks by ≥ 5 tasks, with no rise in empty patches.
- **Leaderboard submission (Shrey):** the run writes the exact upload as `submission.zip`, adapter included.
- Evals attach `shreysharma07/gemma4-test-overlays` and embed the 114-task list automatically; nothing to add.

### 7.4 What to send back after each step
Put the downloaded folder under `data/runs/<run-name>/` in the shared repo (or zip it to Shrey):
- `train_report.json`, `train_out/metrics.jsonl`, `run.log`, the notebook log `*.log`;
- for evals, also `summary.json` and the `eval_*.tgz`.

Add a row to §8 and, for anything surprising, a line in the decisions log (§5).

### 7.5 Known failure signatures (and what they meant)
| Symptom in the log | Cause | Fix |
|---|---|---|
| `NameError: name 'null' is not defined` in cell 1 | JSON pasted into Python (fixed Oct 12) | rebuild from this repo |
| `ensurepip is not available` / run ends before task 1 | Debian venv exits the process (fixed: `evalkit/compat.py`) | rebuild; never hand-edit |
| `ModuleNotFoundError: pluggy` / `wrapt` warnings | harmless noise from the image's `sitecustomize` | ignore |
| `attach dataset metric/gemma-4-developer-agent-wheelhouse` | input missing in kernel metadata | rebuild (the builders add it) |
| `Maximum weekly GPU quota ... reached` on push | quota | wait for Saturday 00:00 UTC |
| `need GPU L4 x4, got ...` | notebook didn't get 4×L4 | see Step A |
| `rank_tails` in `train_report.json` with `CUDA out of memory` | a window too long for memory | tell Shrey (fix: `offload_from` lower or cap length) |

## 8. GPU quota ledger
Oct 15 output verification: all Step A outputs downloaded. Both adapter copies have SHA-256 `09877873743ec96d0ef7d362e042ac512deb8331f1022bf5f5c02e4d1313d1a0`; stdlib-only header checks confirm all 820 BF16 tensors, rank-16 shapes, and complete file length. Per-rank logs show peak memory 17.61 / 18.46 / 18.69 / 18.81 GB, all below 21 GB. Step A was initially stopped on the faulty rank-0 throughput metric, then reassessed PASS after all-rank reconstruction bounded every step within target; original measurements retained.

| Date | Account | Run | Expected h | Actual h | Result |
|---|---|---|---|---|---|
| Oct 18 | Atul | **E1 CPU PASS; S3 GPU preflight prepared, awaiting explicit runtime approval** | S3 ~2.1–2.4 session h / ~4.2–4.8 quota h, unchanged 3 h budget; S1 ~2.6–2.9 / ~5.2–5.8, gate ~0.6 / ~1.2; sequence ~10.6–11.8 quota h, forecasts only | **0 new GPU session/quota h; no notebook pushed** | E1: exact benchmark bases recovered for 147/147 episodes; 182/182 edits replayed in disposable no-network namespace; 151/151 initial and final episode/file blobs match teacher patch indexes; empty-file edit valid; no dataset change. Corrected initial overbroad E1 gate interpretation: handoff explicitly allows parallel CPU replay and requires confirmed failures removed before training. New independent `.venv` has Kaggle CLI 2.2.4 + dependencies only, existing login verified without credential inspection/copy. At 02:08:10 IST quota remains **30.00 h available, 0 used/reserved**; project notebooks COMPLETE. Exact S3 notebook built: 459,501 bytes; owner/data/model/flags verified. Await >2-session-h and >10-cumulative-quota-h approvals; refresh quota/status before push. Source scheduler plans validation on 4/7 S3 and 20/21 S1 windows. [Consolidated report §§7.1/8.1](atul_step1a_1b_step2_gate_report_20261018.md). |
| Oct 18 | Atul | S3/S1/gate preflight; **no notebook submitted** | Handoff: S3 ~1.4 session / 3 quota h; S1 ~1.7 / 3.5; gate ~0.6 / 1.2; total ~7.7 quota h. Two-epoch token/load scenario suggests ~10.17 quota h before validation/export overhead; revisit approvals before launch | **0 new session h / 0 new quota h**; no run-boundary charge ratio | BLOCKED before Step 1a: E1 lacks verified base commits/source snapshots and established disposable isolation. Four handoff hashes match. At Oct 18 01:25:27 IST, authenticated quota = 30.00 total, 0 used, 0 reserved, **30.00 available**; three project notebooks COMPLETE, two unrelated notebook sessions 404. No S3/S1/gate outputs. See the [single consolidated report](atul_step1a_1b_step2_gate_report_20261018.md). Refresh quota and apply >2 session-h / >10 cumulative quota-h approval rules before resuming. |
| Oct 15 | Atul | Step C corrected v2: `gemma4-train-lora`, private dataset-backed 200-task input, 10.5 h budget, val/save every 40 | 9–10 session h (~18–20 quota h) | **7.8203 session h / 15.64 quota h** (18.13 − 2.49; 11.87 h left) | **Step C v2 PASS / COMPLETE:** rc 0, all 122 planned optimizer steps, no budget stop; validation 0.7718 → 0.3133 → 0.2909 → 0.2883 → 0.2882. All checkpoints/final/serving files pass 410-module / 820 BF16 rank-16 tensor integrity and exact shape checks; final/serving SHA-256 matches. Global peak 18.88 GiB; raw median 294.8 tok/s, steady median 295.6 (247.7–324.0). Three intervals include validation/export overhead. Existing group-of-four scheduler used 972/974 train and 40/43 val windows while retaining all 190/10 tasks. Actual 7.8203 session h / 15.64 quota h, versus expected 9–10 / 18–20; quota 11.87 h remains. All outputs preserved and final report `docs/step_c_20261015_report.md`. No retry, D, or leaderboard run; monitor paused. Earlier HTTP-400 upload created no run and consumed 0 GPU quota. |
| Oct 15 | Atul | Instrumentation CPU smoke: `gemma4-train-smoke`, tiny random Gemma-4 / 2 gloo ranks / 4 windows | 0 GPU h | 0 GPU h | PASS: COMPLETE, rc 0, token SUM and elapsed MAX verified against both rank logs, rates agree, valid 41-module tiny export. No local torch/model execution. |
| Oct 15 | Atul | Step B v1: `check-l4`, `v8_lora_smoke`, 8 valid tasks, adapter from unchanged A v1, 2 h session budget | ~1 session h (~2 quota h based on A accounting) | 1.06 quota h (31.91 min session) | PASS: COMPLETE, exit 0, adapter copied/declared, vLLM up in 409 s, all 5 health checks true. Archive: 8 results/traces, tools on 8, tests on 7; one 100-turn-limit error; 2/8 resolved (smoke only). Serving adapter checksum matches A. Quota 28.57 → 27.51 h remaining. C prepared only; Shrey approval unconfirmed. |
| Oct 15 | Atul | Step A v1: `gemma4-train-lora`, 40 own v7 windows, `conv_lora`, 2 h budget | 0.7 | 1.43 quota h (~0.71 session h) | COMPLETE, rc 0, 42.7 min. Val 0.2557 → 0.1818; 410 modules, shapes valid; peak logged 17.61 GB. Initially stopped on logged throughput; reassessed PASS after exact all-rank correction to 257.50–295.61 tok/s. A not repeated; B passed; C pending approval. Pre-push quota 30.00 h; post-run 28.57 h. Existing generated files preserved unchanged. |
| Oct 10 | Shrey | training probe v3 + v4 | 1.5 | ~1.7 | training envelope (≤8k) |
| Oct 12 | Shrey | eval attempts 1–3 (my bugs) | — | ~0.8 | fixed: cell literal, Debian venv, health check |
| Oct 12 | Shrey | v7 full local eval (129 tasks) | 10 | 4.4 | 19/129; 18/54 valid |
| Oct 12–13 | Shrey | v7 rejection sampling, 5 reps × 54 | 10.5 | 9.7 | pass@5 21/54, 82 solved runs |
| Oct 10–13 | Shrey | Kaggle start-up/output overhead, ~7 sessions | — | ~1.5 | — |
| | | **accounted for** | | **~18** | ~12 h of the 30 h not explained by these runs: check *Versions* run times on kaggle.com |
| Oct 14–15 | Atul | Steps A + B + C (pipeline check, serving check, 200-task training) | ~11.7 | ~18.1 (30 − 11.87 left) | C: val loss 0.77 → 0.29, 7.82 h |
