# Full T1 LoRA versus v14 on the original 60 tasks

## Terminal paired GPU result — 7 November 2026

**VERIFIED.** The single approved private Kaggle run, version 6 reached COMPLETE. The evaluator exited 0, reported 60 matched pairs and did not trip the 10.75 h watchdog. The platform log places evaluator termination about 8 h 59 min after its clock origin. The first successful terminal status check was 10 h 28 min after push. These clocks measure different things. Both arms used the W4A16 model, concurrency 1, one repeat, the same predeclared task manifest and equal limits (240 s command timeout, 8 min/task, 28 tool calls, 80 turns). The full arm alone added T1. Its runtime SHA-256 assertion and independent local read agreed on `2b22c3f393f178f0c8885df99229a501da826ec1150bda83d8bfd7f4d3d56510`. The verified source checks 820 tensors and rejects global-layer V-projection copies. The raw adapter was not part of the results archive.

| Outcome | Count |
|---|---:|
| v14 resolved | 27/60 |
| v14 + full T1 resolved | 26/60 |
| Both solved | 24 |
| Full only | 2 |
| Base only | 3 |
| Neither solved | 31 |
| Missing paired outcomes | 0 |

Exact two-sided McNemar p = 1.00 (5 discordant pairs). Full minus base was −1/60 (−1.7 percentage points), with a descriptive 95% paired percentile-bootstrap interval of −8.3 to +5.0 points (50,000 resamples of task pairs). The interval does not account for generation variability across repeated runs. Both arms solved 22 FastAPI and 3 Requests tasks. The net difference was one Rich task (2 vs 1 resolved), with only six Rich tasks in the cohort.

**Interpretation.** This run provides no evidence that full T1 improved repair success over the fresh baseline. It does not establish equivalence. The historical v14 30/60 and v17 27/60 are separate runs (see `docs/experiments.md`). The fresh v14's 27/60 also illustrates why the historical v14 score is not the control for this experiment. The full arm consumed 275.33 chunk minutes vs 252.20 for base (+23.13 min, +9.2%); this includes tool and verifier time and does not isolate LoRA decode overhead. Harness token totals are unavailable as a reliable usage measure. Recorded context-limit errors were 9 vs 10 (base vs full).

**Provenance.** Recovery evidence is retained privately under `data/runs/atul_full_lora_new60_20261104/gpu_recovery_v6_20261107T075101Z/`. The numeric version selector returned HTTP 404. The current-version listing was bracketed by successful checks of version 6, privacy and source cells. Six listing pages were retained in the receipt and their logs preserved. The two recovered archives are byte-identical (SHA-256 `ea79279432f51830c2b8ae18dc15fee5a842812e2fa8d52a3ab0e7ee12b57cae`); no terminal partial archive was listed. Archive paths and member types were checked before extraction. The merged rows contain one result per ID per arm. Both arms have the same merged row order, which differs from manifest order within chunks. All eight chunk memberships match the manifest slices, and the run passed the per-chunk identity checks. The archived settings and pair results agree with the separately downloaded JSON. All 2,298 protected-file hashes matched before appending these two untracked reports.

**Quota.** Kaggle reported 30.0 available, 0 used and 0 reserved both before launch and at the terminal check, an observed net account change of 0.0 h. Used quota rose to 1.18 h in the early snapshots and then returned to zero during execution. These readings do not establish the actual charge.

## Historical plan and initial CPU preflight

## NF4 / W4A16 compatibility decision

**VERIFIED:** T1 trained on `google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2`, the bf16 QAT checkpoint, quantized at load to NF4 with double quantization and bf16 compute. Competition serving uses `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2`. These use different quantization grids and implementations; the adapter architecture is compatible.

LP1 already tested the adapter on W4A16: 40 validation episodes, 612,179 tokens, weighted loss **0.08663 → 0.05075** with raw T1, a 41.42% reduction. A global K-to-V copy gave 0.05092. G1 ran raw T1 through the W4A16 official evaluator: base 40/95 versus full 42/95, six gains and four losses, exact paired p=0.75390625. LP2 on HF/NF4 found full loss 0.06661 versus global-V-base 0.06637, with 26/40 episodes favoring removal of the V delta. The raw 410-module, 820-tensor export is therefore the reference; no synthesized global V factors are added.

**INFERENCE:** Existing evidence supports unchanged T1 reuse for the planned W4A16 comparison. W4A16-specific retraining or conversion is not justified merely by the quantization difference. Teacher-forced loss transfer does not prove improved repair performance or universal absence of quantization effects. No retraining or conversion is planned.

Final/serving SHA-256 reverified locally: `2b22c3f393f178f0c8885df99229a501da826ec1150bda83d8bfd7f4d3d56510`. Original weights and research remain unchanged. Latest remote main inspected read-only: `c6ae20ecc3a62c160878cc1c2a3532f73ca2e7ce`. Relevant training, export, evaluator and LP1/LP2 files match local Git blob hashes; subsequent commits update research documentation.

## Registered experiment and launch gates

Plan: `data/runs/atul_full_lora_new60_20261104/preflight/compatibility_and_plan.md`.

- Fresh A: v14 without an adapter. Fresh B: identical policy plus full raw T1.
- Original 60 IDs and exact task order; 120 task episodes in total. Historical v14's 30/60 remains a separate observation.
- Concurrency 1; eight-minute tasks, 28 tool calls, 80 turns, 240-second command timeout; thinking 4096 and output reservation 8192.
- Existing evalkit implementation, one server, alternating paired chunks. Source differs only by the root adapter reference.
- Session scheduling budget 10.5 hours, evaluator watchdog 10.75 hours, manual cancellation alert at 11 hours. Expected consumption is 8–9 session hours / 16–18 quota-hours; launch requires at least 23.5 quota-hours and no other active GPU job.

Thirty generated GPU-notebook code cells compile. Embedded submission, sampling, task identities and orchestrator match their intended sources. Original notebooks are untouched. The private CPU preflight was accepted as version 1 at full-lora-new60-cpu-preflight. Its source, version, privacy and GPU-disabled setting are verified. Input discovery, adapter hash and official compiler must pass before the single guarded GPU push.

Before the CPU report was written, Kaggle showed 30.0 quota-hours available, with zero used and zero reserved. Timestamped snapshots are retained privately.

## Competition decision is separate

The earlier v18 submission is COMPLETE with an empty score; HQ records a notebook timeout. LT1 measured approximately 11% decode overhead and no timing advantage for attention-only LoRA. No unchanged resubmission is planned. A separately named full-LoRA runtime candidate requires measured runtime, correctness and quota validation; at most one candidate may be submitted. None has been submitted in this task.

Authenticated official pages recovered in `preflight/competition.json` establish a 12-hour patch-generation limit including sandbox setup **but excluding patch validation**, one submission per day, December 2 final deadline, November 12 optional paper deadline, and offline L4×4 sessions charged at twice wall time. These dated observations must be refreshed before submission. Local end-to-end runtime cannot be equated with the timed generation phase.

## Pending evidence and preservation

New A/B coverage, resolves, paired outcomes, uncertainty, runtime, tokens and overflows are **UNKNOWN**. No new leaderboard score is claimed. Traces and verifier outputs will remain private. Existing untracked R2 evidence is preserved. No staging, commit, push, PR, weight change, training or GPU launch has occurred. Detailed preflight evidence is in `data/runs/atul_full_lora_new60_20261104/`.


## Terminal CPU preflight result — 6 November 2026

**BLOCKED — implementation error in the new dry-run path.** Kaggle version 1 is terminal COMPLETE, but the evaluator child exited **1**. It failed at `from swegemma.models.registry import setup_gemma_model_registry` with `ModuleNotFoundError: No module named 'swegemma.models.registry'`. The restored scorer version does not provide this legacy API. No `paired_dry_report.json` or official compiler PASS exists. Notebook COMPLETE therefore does not mean preflight success.

Scorer preparation completed, test dependencies were found, base and full submissions unpacked, and the adapter was discovered from T1 output before this failure. The exception occurred after the script's adapter hash, 820-tensor, settings and task-identity assertions; this supports those checks, but official compilation remains unverified. The failure is in the newly authored preflight implementation; it is **not evidence of an incompatible T1 adapter or justification for retraining**. Earlier LP1/LP2/G1 evidence remains unchanged. Shared network cost-map warnings fell back locally and are not the terminal cause.

Original report/config/log recovery: `data/runs/atul_full_lora_new60_20261104/cpu_recovery_20261105T224049Z/`. The recorded run.log SHA-256 is `7a726a3d29d4f69b04195ebc925d5ce1d223d435e9835ea39495ebaa9ba29f2d`. Platform logs preserve the traceback and child exit code.

Per the registered failure-stop rule, **no GPU push, retry, training, conversion or competition submission was performed**. New A/B task counts remain UNKNOWN because no task evaluation occurred. Last actual account snapshot: 30.0 quota-hours available, 0.0 used and 0.0 reserved. The next necessary step is to replace the legacy dry-run compiler registry construction with the installed scorer's supported API and validate that preflight before any GPU launch. This turn stops at the recorded blocker without retrying. The scheduled monitor is being deleted to prevent automatic launch or repeated polling.
