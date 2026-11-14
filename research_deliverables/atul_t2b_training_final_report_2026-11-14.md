# T2b LoRA training final report — 14 November 2026

T2b finished with trainer exit code 0 and all 66 planned optimizer steps. The final on-policy validation loss was 0.0695 and corrected validation loss was 0.0509. The final adapter and step-33 checkpoint passed independent integrity checks. The validation results show that T2b learned the corrected target actions, while T2a retained lower on-policy loss. Task resolution remains a separate measurement.

## Run identity and inputs

Authenticated Kaggle recovery confirmed the private, offline NvidiaL4 notebook at version 1, with source cells matching the hash-verified launch copy and terminal status COMPLETE. The startup log records four NVIDIA L4s. The training exit code and optimizer-step sequence were independently checked in the platform log, training report and raw metrics. The recovered report matches Shrey's committed T2b report as parsed JSON.

Source commit at launch was `ec16cdbac734d4220acae04cb5dcd77998ac088f`. The input was the RS4c training archive from version 3 of the dataset-building notebook, SHA-256 `0dd24e6284208ab4df628230286c729a26ebd4bb9161793e372fccd5f200e801`. The NF4 trainer loaded the approved unquantized Gemma 4 31B QAT checkpoint version 2. Settings were one epoch, LoRA rank 16 and alpha 32, learning rate 1e-4, five warmup steps with cosine decay, clipping at 1.0, gradient accumulation 2 over four ranks, maximum length 28,672, and session budget 10.5 hours.

The data held 391 on-policy and 141 corrected training episodes, plus 40 on-policy and 12 corrected validation episodes. The longest episode was 26,831 tokens and none were dropped for length. Corrected training weight rose from 4.88% to the configured 15% before batching, using a multiplier of 3.443. The planner drops the four-episode remainder for full batches of eight, so 528 of the 532 eligible episodes fit the 66-step schedule. Raw step logs record 6,357,282 processed tokens versus 6,398,527 tokens in the eligible pool.

## Training and validation

| Run and split | Start | Midpoint | Final |
|---|---:|---:|---:|
| T2b on-policy | 0.1011 | 0.0719 | 0.0695 |
| T2b corrected | 0.7073 | 0.0602 | 0.0509 |
| T2b combined | 0.1302 | 0.0714 | 0.0687 |
| T2a on-policy | 0.1011 | 0.0678 | 0.0657 |
| T2a corrected | 0.7073 | 0.7743 | 0.7731 |
| T2a combined | 0.1302 | 0.1016 | 0.0996 |

T2b midpoint was step 33 and T2a midpoint was step 24. Logged T2b optimizer loss was 0.1641 at step 1 and 0.0646 at step 66. These are different batches. Median step throughput was 349.0 tokens/s (292.6–402.2). Across 18,942 seconds of timed optimizer steps, aggregate throughput was about 336 tokens/s. Pre-clip gradient norm exceeded 5 at five steps, peaking at 17.478. T2a had five such steps out of 48 with a maximum of about 28.6, so these spikes also occurred without corrected training episodes. Peak reported GPU memory was 17.47 GiB per rank. The 66 step metrics were finite, and the retained platform and per-rank logs contained no NaN, OOM, interrupted-rank, or budget-stop signatures.

Final export was recorded at 354.9 minutes on the trainer clock. The platform log reached exit 0 about 355.1 minutes after its clock origin. At the observed 2× L4 quota rate this corresponds to about 11.8 quota-hours. The latest account snapshot after the weekly refresh showed 30.0 available and zero used or reserved. The refresh prevents an exact charge from being reconstructed from the captured account readings. Recurring monitoring was not active during training. The terminal metrics come from the recovered outputs.

The preregistered step count, approximate runtime, initial on-policy loss, finite training and 17.5 GiB peak ceiling held. Final on-policy loss exceeded the 0.0666 target and was 5.8% above T2a, missing the within-5% target by about 0.8 percentage points. Corrected loss was 93.4% below T2a, meeting the 30% reduction target. The corrected targets use templated edits, tests and submission steps. Lower loss on those targets does not establish that the agent chooses the right actions or resolves more bugs on its own.

## Adapter verification and preserved evidence

Each weights file is 244,982,352 bytes. The validator independently checked 410 complete LoRA A/B module pairs, 820 BF16 tensors, exact key and shape correspondence with the verified T1 reference export, contiguous data offsets, expected byte counts and finite tensor payloads. All three configs specify rank 16 and alpha 32 and share the same SHA-256. Final and serving weights are byte-identical by SHA-256. Step 33 is distinct.

| Recovered artifact | SHA-256 |
|---|---|
| `train_out/adapter_step33/adapter_model.safetensors` | `db05e128da2ea211616baba54440d479017e44fc99af8caf406ca9ad27097e92` |
| `train_out/adapter_final/adapter_model.safetensors` and `adapters/t2b_lora/adapter_model.safetensors` | `ad6062130ba0efbd2c7c79734906fc5158cabbcb3439190969a4e4ee177eb8e3` |
| All three `adapter_config.json` files | `6b0a7255228061cb6f69a91baff7d8e0c7ec27e090ddd6ef2f9a0b0c73d2d5cc` |

The private recovery bundle includes `train_report.json`, `train_cfg.json`, `train_out/metrics.jsonl`, eight rank logs, host records, platform output, adapter files, `receipt.json`, and `verified_training.json`. The explicit numeric version selector returned HTTP 404. Recovery used the current output listing with successful source, version 1 and terminal checks before and after download. The receipt records hashes and no transfer errors.

## LP3 and G2

Shrey's main at `ec9a13ff492eb982e46b5a27976405a5337f3616` documents that the original T2 rule would exclude T2b from G2. Shrey amended the design before GPU evaluation, adding T2b to LP3 and G2 because the on-policy miss was narrow on one seed and T2b is the arm testing the corrected signal. [Experiment registry](https://github.com/ShreySharma07/gemma4-swe-agent/blob/ec9a13ff492eb982e46b5a27976405a5337f3616/docs/experiments.md)

LP3 was complete in those records. On served W4A16, on-policy losses were 0.0866 base, 0.0517 T2a and 0.0566 T2b. Both passed the 15% serving threshold. LP3 did not measure corrected validation. G2 held-out gate version 1 was documented RUNNING at 12:18 UTC. That is the committed status, with no terminal G2 result in the inspected main. Review G2's paired resolves and runtime once available. The preregistered F3 follow-up depends on a held-out gain of at least two resolves. [Project status](https://github.com/ShreySharma07/gemma4-swe-agent/blob/ec9a13ff492eb982e46b5a27976405a5337f3616/docs/HQ.md)
