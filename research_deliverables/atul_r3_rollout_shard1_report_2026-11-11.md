# R3 Factory V2 rollout — shard 1 preflight

**Initial preflight decision: launch blocked.** The authenticated owned-notebook inventory listed 11 notebooks under `atulbaliyan`, with no active or queued run and no entry for `gemma4-r3-rollout-s1`. The direct metadata and status requests for that exact slug both returned HTTP 403, with the status client reporting `Permission 'kernels.get' was denied`. This could be Kaggle's response for a new slug, but the response does not independently establish that no saved version exists. The launch was withheld under the requested identity and duplicate-run gate.

## Verified inputs and configuration

- GitHub main was `9f94c98846fdd0712bb35857189601e0d2da117d`. The working research checkout lacked R3 and its Git metadata was read-only in the sandbox, so the latest main was fetched into an isolated checkout.
- The approved notebook's code cells parsed successfully. The embedded rollout module matched the repository implementation, the embedded submission matched `v14_thinking`, and there was no adapter. The two shard notebooks' code cells differed only in the shard selector. Notebook SHA-256 `30e0da4201e785ec8ef73fa8873e5c953bb6ba94b1ee802232e3b536bbe886e8`; metadata SHA-256 `e0d8f4d928eaed8495320b0c06f1a70d1f1ea6015b563f7eb9ee61318cd0e2d3`.
- The metadata requests a private offline NvidiaL4 run. Startup enforces four L4s and serving uses tensor parallelism 4. Concurrency is 6 with 30-task chunks and one episode per task. The task cap is 12 minutes, command timeout 240 seconds, session budget 7 hours, watchdog 7 hours 15 minutes.
- The shared Factory V2 dataset was accessible. Its downloaded manifest contains 527 unique verified tasks, with a listed snapshot for each. Sorting by ID, shuffling with seed 0 and taking alternating entries matched the locally reconstructed preregistration split exactly. Shard 1 contains 263 tasks (63 singles, 166 pairs, 34 triples). Shard 0 contains 264 tasks. The two sets are disjoint.
- The wheel datasets, competition file listing, W4A16 model version 2 and required kernel sources were accessible. Kernel source versions were scorer-env 1, devset-build 5 and test-overlays 3.
- Kaggle reported 30.0 quota-hours available and 0 reserved. The preregistered estimate is about 5–6.3 hours elapsed (10–12.6 quota-hours), with up to about 14.5 quota-hours at the watchdog. Quota passed the resource check.

## Evidence and next step

The account receipts, manifests and shard IDs are retained privately. No notebook push was attempted. There is no accepted version, runtime charge or rollout result to analyze, and no monitor was started. No repository branch or tracked content was changed.

Confirm in the signed-in Kaggle account that the target slug has no saved version, or restore read access to its metadata and status. Once rollout results exist, apply Shrey's preregistered multi-site threshold across both shards. For RS3 screening, keep verifier-confirmed episodes with clean patches and reliable post-edit verification, and check context and duplicate trajectories before rendering.

## Launch receipt — 11 November 2026

Atul manually confirmed that the target slug had no existing notebook and explicitly approved its creation. The launch guard rechecked current GitHub main, the exact notebook and metadata hashes, owned notebooks, GPU quota and input access. The owned inventory still contained no target or active GPU job. One push from the isolated checkout was accepted as private version 1 at approximately 01:52 IST. Kaggle independently confirmed RUNNING, private, GPU enabled, internet disabled, and source cells matching the verified files. The initial live log showed four NVIDIA L4s and scorer restoration. Task execution and outcome coverage were not yet visible in that snapshot.

Quota moved from 30.000 available before push to 29.993 available at the first status check, with no reservation in either snapshot. The raw push receipt, post-launch status and initial log are preserved in `data/runs/atul_r3_s1/`. No recurring app monitor was installed. Terminal results are pending.

## Verified terminal results — 12 November 2026

The private version 1 notebook was COMPLETE at the 22:05 IST recovery check. The platform log recorded evaluator exit code 0, with no watchdog kill or failed chunk found. The first-chunk health checks passed, and the run log recorded four L4s at startup, vLLM readiness and scorer test execution. The run finished at about 07:25 IST on 11 November, approximately 5 h 33 min after launch. Evaluation chunks took 316.8 minutes at 49.82 tasks/hour. The terminal account snapshot showed 11.109 quota-hours used, 18.891 available and zero reserved. The change from the pre-launch balance was 11.109 hours, consistent with the rollout duration at the expected accounting rate. The account-level readings do not itemize charges by notebook.

The report and rewritten task manifest match the approved shard order. The merged results contain one outcome per expected ID, all 263 traces, and no missing, duplicate or extra task outcomes.

| Outcome | Result |
|---|---:|
| Attempted and completed | 263/263 |
| Resolved | 81/263 (30.8%) |
| Unresolved | 182/263 (69.2%) |
| Singles | 44/63 (69.8%) |
| Pairs | 31/166 (18.7%) |
| Triples | 6/34 (17.6%) |
| Multi-site total | 37/200 (18.5%) |

The shard fell below the preregistered overall 35–47% range, driven by pairs falling below the 28–40% range. Singles and triples fell within their respective ranges. Full-template issues resolved 41/104 (39.4%), no-name issues 18/45 (40.0%), and terse issues 22/114 (19.3%). The predicted no-name disadvantage did not appear on this shard. Coupled multi-site tasks resolved 26/108 (24.1%) versus 11/92 (12.0%) for independent tasks. These subgroup rates are descriptive. Shard 1 contributed 37 of the 80 resolved multi-site episodes required by the combined-shard RS3 decision rule. Shard 0 must contribute at least 43 for that threshold to be met.

## Patch, test and runtime findings

- The evaluator recorded 243 nonempty patches and 20 empty patches. Of the empty-patch episodes, 11 had verifier exit −1 and nine had exit 1. The analyzer flagged stray files on 75/263 tasks (28.5%).
- The verifier recorded 81 exits at 0, 164 at 1, seven at 124 and 11 at −1. There were 252 verifier output files. A scan of retained verifier and task logs found no explicit patch-application failure signature and no pytest collection-error signature. Exit 1 alone does not identify a root cause. The logs do not support a claim that every nonempty patch applied cleanly.
- The harness recorded 33 task session timeouts (12.5%), four context overflows (1.5%), and two task terminations for tool-call budget. The analyzer separately recorded 30 tool-level BudgetExceeded events on 23 tasks and 17 tool-level TimeoutExceeded events on seven tasks. The rollout report's 36 time-cap and 63 call-cap hits are threshold proxies. Rewrite succeeded on 261 tasks and fell back twice (0.8%). Throughput, overflow and rewrite fallback were within their predicted ranges. Session timeouts exceeded the predicted 8% cap-hit ceiling even when counted by explicit harness errors alone.
- The mutation-aware trace analysis found recognizable local test invocations on 89 tasks and a returned local PASS on 59. Seventy-four tasks invoked a local test after their last detected source mutation, and 53 recorded a local PASS after it. Among the 81 resolved episodes, those counts were 37 and 27. These are heuristic detections from trace commands and output. The verifier result remains the evidence for resolution.

## RS3 evidence and review

The first-pass shortlist has 25 episodes (18 singles, six pairs, one triple) with a verifier-confirmed resolve, no analyzer-flagged stray files, recorded peak prompt under 20,000 tokens and a detected local PASS after the last source mutation. The shortlist is narrower than the 81 resolved episodes and is a review queue. Before rendering, inspect its patches and test scope, check for duplicate trajectories and leakage, and confirm the split against the R2 and shard 0 manifests. Preserve rejected and unsuccessful episodes for failure analysis. The joint multi-site decision requires verified shard 0 results.

Raw recovery files and the task-level shortlist are retained privately. The numeric version selector returned HTTP 404. The recovery used the current output listing bracketed by matching source, version 1 and terminal status checks. The raw archive, evaluator archive, standalone report, rewritten task manifest, settings and platform log are hashed in the receipt. The nested evaluator archive and standalone copy match byte for byte. The archived run log is a prefix of the standalone log, which includes the later completion line. Both copies are preserved. Recurring 15-minute monitoring was not active during the run, so live observations cover startup and the terminal check. The runtime and task counts come from the recovered outputs.
