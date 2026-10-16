# Atul: Gemma 4 post-training A/B/C report — October 15, 2026

All three authorized gates PASS. Step C version 2 completed with return code 0 and all 122 planned optimizer steps. No additional run, Step D, or leaderboard submission was launched.

| Stage | Result | Expected usage | Actual usage | Evidence |
|---|---|---|---|---|
| A, pipeline | PASS after throughput audit; no repeat run | ~0.7 session h | ~0.71 session h / 1.43 quota h | Loss 0.2557 → 0.1818; 410 modules / 820 tensors; global peak 18.81 GiB. Original 193.2 tok/s median preserved; exact all-rank reconstruction establishes 257.50–295.61 tok/s, corrected median about 277.0. |
| B, serving | PASS | ~1 session h | 0.532 session h / 1.06 quota h | Step A adapter loaded; vLLM started; all five health checks true; eight tasks completed evaluation. One task hit its turn limit, two resolved. Training overlap means the score is only a smoke result. |
| C v2, full training | PASS | 9–10 session h / ~18–20 quota h | 7.8203 session h / 15.64 quota h | COMPLETE, rc 0, 122/122 optimizer steps, no budget stop; 17 validation checks passed. |

## Step C configuration and execution

One explicitly approved corrected submission used private dataset `atulbaliyan/gemma4-train-public-200`, SHA-256 `b3f51bc449c31de2a3b163f98c5fdec2725b4427bde15625af7b4c75447d2488`. All 200 tasks and 974 training / 43 validation windows were attached. The downloaded runtime settings confirm `smoke=false`, `max_windows=null`, one epoch, `conv_lora`, rank 16, alpha 32, learning rate 0.0001, accumulation 2, validation and save every 40 steps, and budget 10.5 h. Startup confirmed four NVIDIA L4s and the approved Gemma 4 31B model. Shared inputs remain under `shreysharma07`.

The original HTTP-400 embedded-source upload created no run and consumed no GPU quota. Its evidence and A/B outputs remain in their original folders. The successful corrected version used compact source and the private dataset. No automatic retry occurred.

## Validation and throughput

| Validation point | Loss | Actual validation windows |
|---|---:|---:|
| Start | 0.7718 | 40 |
| Step 40 | 0.3133 | 40 |
| Step 80 | 0.2909 | 40 |
| Step 120 | 0.2883 | 40 |
| End | 0.2882 | 40 |

Loss fell at every recorded validation and never reversed. All 122 optimizer steps completed, with 243 scheduled microsteps per rank. A stdlib reconstruction of the seed-0 schedule matched every rank's input tokens, the SUM of four ranks, maximum elapsed time, and rounded throughput for all 122 intervals. Exactly 7,039,924 input tokens were processed. Input-token throughput includes context tokens, rather than only tokens contributing to the supervised loss.

Raw interval throughput: median 294.8 tok/s, range 113.3–324.0. Steps 41, 81 and 121 include the preceding validation/export because the timer resets before that work; their rates are 113.3, 128.0 and 115.7 tok/s. All other intervals are within target at 247.7–324.0 tok/s, median 295.6. Token-total / elapsed-total interval throughput is 282.60 tok/s. These interpretations are documented separately; original metrics were not rewritten. Global peak allocated GPU memory is 18.88 GiB, below the 21 GiB gate.

The existing equal-rank scheduler discards incomplete groups of four. It processed 972/974 training windows and 40/43 validation windows; two training windows totaling 12,726 input tokens and three validation windows were omitted. All 190 training tasks and all 10 held-out tasks were still represented, with no train/validation task overlap. This is a documented scheduling limitation of the authorized configuration, not an incomplete run or missing attached data.

## Adapter integrity

Checkpoint exports at steps 40, 80, 120, the final export and the serving copy all passed safetensors header, contiguous/non-overlapping offsets, BF16 byte counts, file length, PEFT configuration, A/B pairing, and all 410 module / 820 rank-16 tensor checks. Every exact tensor key, dtype and shape also matches the previously validated Step A adapter layout. Each adapter file is 244,982,352 bytes.

Final and serving-copy SHA-256: `a116b6be160fffcc98cdd2a4ab6cfaffafcfc54ac6709a8a5d95d95ebab544c6`.

The new C adapter has not been served or evaluated for issue-solving performance. Step B verified serving with the earlier A adapter. Additional serving/evaluation jobs require authorization.

## Accounting and retained outputs

Quota before C: 2.49 h used / 27.51 h remaining. After completion: 18.13 h used / 11.87 h remaining. Difference: **15.64 quota h**. The notebook event timeline spans 28,153.13 s (**7.8203 h**), consistent with observed 2× session quota accounting. Quota readings have 0.01 h display precision. Total A+B+C charge is 18.13 quota h; CPU instrumentation smoke and rejected upload cost zero GPU hours.

All completed-run outputs were downloaded successfully to `data/runs/atul_step_c_corrected_20261015_v2/`, including `train_report.json`, `train_cfg.json`, `run.log`, `gemma4-train-lora.log`, rank logs, metrics, checkpoint adapters, final adapter and serving copy. Validation evidence is in `validation/summary.json`; the run record is `summary.json`. Raw reports are preserved. A/B remain under `atul_step_a_20261015_v1` and `atul_step_b_20261015_v1`.

No traceback, CUDA out-of-memory error, or budget stop occurred. The standard OMP_NUM_THREADS startup warning is benign. No model or GPU workload ran locally, and no credentials were displayed or modified.

**Next stage:** no additional training, Step D, or leaderboard submission is authorized. The completed-run monitor is paused.
