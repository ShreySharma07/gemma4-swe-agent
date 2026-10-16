# Step D evaluation and Step D0 quota gate — October 16, 2026

Step D completed its single authorized pass: **0/114 resolved (0%)**, with **114/114 empty patches**. Execution and artifact checks passed, but the adapter failed the performance gate. **D0 was not submitted**: final remaining quota is **2.88 h**, below its approximately 3.5 h estimate plus safety reserve. No paired trained-versus-baseline effect can be calculated on the new 60 tasks. Further action requires instructions; the monitor is paused.

## Configuration and adapter identity

`atulbaliyan/check-l4` version **2**, `v8_conv_lora`, 114 valid tasks, one repetition, 10.5 h evaluation budget. The frozen submission files match `v7_noedit` exactly except for `adapter: conv_lora`. Test overlays and the scorer environment were attached; four L4s were confirmed. No training notebook was pushed.

Before D, `atulbaliyan/gemma4-train-lora` latest version was verified as successful Step C **v2**, COMPLETE. A fresh download of its serving adapter matched the preserved C final and serving copies, with 410 modules / 820 BF16 rank-16 tensors. The adapter inside D’s downloaded `submission.zip` has the same SHA-256:

`a116b6be160fffcc98cdd2a4ab6cfaffafcfc54ac6709a8a5d95d95ebab544c6`

This differs from Step A’s adapter (`09877873743ec96d0ef7d362e042ac512deb8331f1022bf5f5c02e4d1313d1a0`). All original configurations and A/B/C artifacts remain preserved. Existing generated evaluation files were backed up before building D, and D’s exact submitted configuration is archived separately.

## Execution verification and results

Kaggle COMPLETE was independently checked against exit code **0**, correct adapter source/declaration, vLLM startup (437 s), all five first-chunk health checks true, and the complete expected task-ID set. There was no first-chunk abort or evaluation budget stop. Exactly one rep contains 114 unique expected results, 114 traces and 114 per-task logs. Seventy-seven tasks reached verifier execution and have test-output logs; 37 task-level errors occurred before testing. No patch files were generated, consistent with every result reporting zero patch bytes. The gzip/tar archive decoded and extracted successfully; `submission.zip` passed CRC checks and its adapter checksum matched C.

| Evaluation set | Step D | Step D0 |
|---|---:|---:|
| All valid tasks | 0/114 (0%) | Not applicable |
| Newly valid common 60 | 0/60 (0%) | **Not run** |
| Original 54 | 0/54 (0%) | Not part of D0 |

The historical v7 reference on the old 54 is 16, 16, 16, 16, 18 resolved (mean 16.4). This is historical context, not a fresh paired run or a substitute for the missing 60-task baseline. No McNemar p-value or paired gain/loss count is reported because D0 observations do not exist. The 114-task and 60-task percentages are not compared as equivalent sets.

## Failure patterns

All 114 patches are empty. There are 77 results without a top-level harness error, 28 context-window overflows, seven 4.5-minute session timeouts and two 100-turn-budget errors. Verifier exit codes: 67 code 1, seven code 2, three code 124; 37 tasks have -1 (not run). This is not merely a run that failed before evaluating tasks.

The existing trace analyzer reports 5,684 `run_command` calls, 81 `submit_patch` calls, no `write_file` calls, 30.3% repeated identical calls, and loops on 57.9% of tasks. All `run_command` argument mappings use the expected `command` key. A separate aggregate heuristic detects 2,544 commands mentioning pytest/unittest on 109 tasks, but edit-intent patterns on only one task (27 commands). That heuristic is incomplete and can match text rather than an actual edit; the definitive outcome is that no task produced a nonempty final patch.

Analyzer failure stages: 52 never located the gold file, 34 found the file but not its lines, and 28 saw gold lines without an identified edit. It reports 1,142 CommandError and 174 BudgetExceeded observations; these are observations, not distinct failed tasks. Of CommandError-type observations, the analyzer separately flags 573 benign grep no-match errors. Its 41-call median counts trace events, while the harness result median is 40; retain this distinction.

The observed failure is a failure to convert investigation/testing into saved edits. Adapter loading and tool activity are proven, but these checks do not prove correct learning or a healthy semantic policy. The specific training/conversion root cause remains unproven. No checkpoint sweep, adapter retry, baseline run, or additional experiment was launched to investigate it. Public repositories and one attempt per task also limit generalization claims. The adapter does not satisfy the documented leaderboard gate.

## Runtime, quota and safety intervention

| Item | Step D | Step D0 |
|---|---:|---:|
| Expected session hours | ~4 | ~2, planning only |
| Expected quota hours | ~7 at 1.75×; conservative ~8 at 2× | ~3.5; conservative ~4 |
| Actual session hours | **4.4916** event span; **4.4949** last event from session start | Not run |
| Quota used before / after | **18.13 / 27.1242 h** | No charge |
| Actual quota charge | **8.9942 h** (~8.99) | 0 |
| Quota hours per session hour | **2.0024×** using event span (~2.00×) | Not measurable |
| Final quota remaining | **2.8758 h** (~2.88) | Insufficient to launch |

The before reading has 0.01 h display precision; do not overinterpret the extra arithmetic digits. Notebook event times run from 11.6721 s to 16,181.5101 s; their difference gives the event span, while the last event gives session-start runtime. Both imply approximately 2× charging, not 1.75×.

Live logs were initially unavailable, then bounded snapshots exposed startup and health. Early pacing (16 tasks in 47.2 min) projected insufficient remaining quota; a manual stop was requested. The installed cancellation API required a verified current session ID absent from the available notebook metadata/status, and the browser was signed out. No guessed-ID cancellation or notebook deletion was attempted. At 00:51 UTC (06:21 IST), RUNNING telemetry showed only 1.60 h remaining and at least 96 completed tasks, prompting an urgent repeat stop request. The later COMPLETE snapshot showed 2.88 h remaining; the live reading had been more pessimistic than final accounting. The session was not stopped by this agent; completed logs show a normal end. Interim requests and readings are retained rather than rewritten as successful safety intervention.

D exceeded both the ~7 h requested estimate and conservative ~8 h planning estimate, but finished above the one-hour reserve. Monitoring/execution delays and unavailable programmatic cancellation limited enforcement; the evaluation budget was not treated as a hard quota limit. D0 remains blocked by the actual final quota, and must not automatically start after the weekly reset.

## Preserved artifacts

All downloaded output files are under `data/runs/atul_step_d_20261016/outputs/`, including `check-l4.log`, `run.log`, `summary.json`, `eval_cfg.json`, `task_ids_order.json`, `valid_tasks.json`, `submission.zip`, `submission_src.zip`, sandbox settings, captured evalkit source and `eval_v8_conv_lora.tgz`. The CLI download exited 0. Extracted results, traces, task logs and verifier logs are under `extracted/eval_v8_conv_lora/rep0/merged/`; the original archive remains unchanged.

Small reviewable reports: `data/runs/atul_step_d_20261016/summary.json`, `validation_summary.json`, and `behavior_summary.json`. Original run files and every timestamped monitor/live-log snapshot remain local. Raw logs, traces, adapters, archives, generated notebooks and unrelated user changes are excluded from the results commit. No GitHub push, leaderboard submission, Step E, local model workload or credential inspection was performed.
