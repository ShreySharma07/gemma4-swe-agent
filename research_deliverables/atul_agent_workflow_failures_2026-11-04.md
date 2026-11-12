# Agent workflow failures: full T1 versus v14

## Terminal workflow evidence — Kaggle v6, 7 November 2026

Both arms contributed 60 unique results and traces. Counts below are base / full T1. The evaluator returned 27 / 26 resolved. The archived original analyzer output was checked and `analysis/mutation_test_metrics.py` was run on the recovered evidence (see `gpu_recovery_v6_20261107T075101Z/workflow/` under the run directory).

| Signal | Base | Full T1 |
|---|---:|---:|
| Nonempty final patches | 58 | 55 |
| Verifier outputs present | 59 | 59 |
| Verifier exit 0 / 1 / 2 / −1 | 27 / 26 / 6 / 1 | 26 / 28 / 5 / 1 |
| Harness context overflow tasks | 9 | 10 |
| BudgetExceeded tool events (affected tasks) | 12 (6) | 10 (9) |
| Task-level tool-budget termination | 1 | 1 |
| Task-level session timeout | 0 | 1 |
| Analyzer-detected malformed tool args (affected tasks) | 2 (1) | 0 (0) |
| Total tool calls (median per task) | 1,318 (23) | 1,264 (22) |
| Tasks with a detected source mutation | 54 | 52 |
| Tasks that invoked a recognizable local test | 10 | 14 |
| Tasks with a returned local PASS | 0 | 2 |
| Tasks that invoked a test after last detected source mutation | 8 | 11 |
| Tasks with local PASS after last detected source mutation | 0 | 1 |

The mutation tagger found 11 / 7 attempted source writes classified as unapplied, which is distinct from final patch application. No explicit patch-application failure signature was found in the retained verifier/task logs. That scan cannot prove every patch applied. The verifier exit codes and final artifacts are stronger evidence of end-to-end outcomes than a command string or a tagger label. The primary tagger recognized writes to 78/78 and 62/63 final-patch source files. Local test calls returned 0 PASS / 10 FAIL / 1 UNKNOWN for base and 2 PASS / 19 FAIL / 1 UNKNOWN for full. Ad hoc repro scripts may fall outside the recognizable-test definition. The original analyzer labeled 22 / 19 failures “wrong_fix_unverified.” These are heuristic workflow labels rather than a count of proven wrong fixes.

**Sandbox diagnosis.** Both first-chunk health checks passed and 59 verifier logs per arm were written, so the fallback did not disable scoring. The no-pip fallback and missing `wrapt` startup warning recurred. `wrapt` also appears in passing verifier output, so it is not a sufficient explanation for failures. There were actual pytest collection errors on 6 / 5 tasks (four shared), including imports of symbols from the task checkout. Their causes are not independently separated into environment, baseline checkout and patch effects. The 9 / 10 context-limit failures were recorded task outcomes, and did not abort the evaluator.

**Interpretation.** Full T1 produced more recognizable test invocations and two returned passes, but only one after the final detected mutation. This is too sparse to establish that the adapter improved verification or that testing caused repair success. It is consistent with MM1's finding that PASS-after-edit is rare, though the test-after-mutation share here (13.3% / 18.3%) is higher than some historical MM1 strata. Neither the MM1 cohorts nor v17 are controls for this run. The paired result and uncertainty are recorded in `atul_full_lora_new60_evaluation_2026-11-04.md`.

## Historical record before the paired GPU run

The terminal paired workflow findings appear first. The remaining sections document the historical v14 analysis and CPU preflight.

## Historical v14 CPU reanalysis

Reused `analysis/mutation_test_metrics.py` against the preserved `data/runs/atul_eval_v14_new60/x/eval_v14_thinking/` directory. Outputs are under `data/runs/atul_full_lora_new60_20261104/historical_v14_workflow_v2/`. The first invocation used too narrow a directory and found no runs; its empty output is preserved separately and excluded from conclusions. The corrected invocation finds 60 traces and 30 resolved tasks.

**VERIFIED output of the heuristic MM1 tagger:**

| Metric | Historical v14 |
|---|---:|
| Source mutation detected | 50/60 (83.3%) |
| Test invoked after last detected source mutation | 5/60 (8.3%) |
| Recognized passing test return after that mutation | 0/60 |
| Any test invoked | 8/60 (13.3%) |
| Mutation detected among resolved tasks | 30/30 |
| Mutation detected among unresolved tasks | 20/30 |
| After-mutation test, resolved / unresolved | 3/30 / 2/30 |
| Final source-patch files seen by the tagger | 68/69 (98.6%) |

One task has a nonempty final source patch without a primary mutation detected. That miss supports keeping uncertain labels available for manual review.

**INFERENCE:** Low observed test-after-edit frequency is a workflow opportunity; it does not prove that forcing tests improves repairs. Test invocation differs from returned PASS, and a reproduction script exiting zero is not proof that an existing test passed. Historical verifier success, 30/60, is a separate measure.

## Required taxonomy for the new comparison

Pending per arm: malformed tool calls and arguments; reading/navigation errors; attempted versus applied/unapplied writes; patch capture and application; empty/source/scratch/stray patches; test discovery and invocation; returned PASS/FAIL/UNKNOWN; collection/dependency/environment errors versus assertion failures; testing before versus after the last actual mutation; context overflow; actual BudgetExceeded events versus proximity to a cap; tool/session/platform timeouts; and calls, tokens and runtime.

Use actual final diffs and verifier output alongside traces. Manually inspect uncertain Python/heredoc/shell writes, indirect scripts and combined edit/test commands. These events are not mutually exclusive causes without adjudication. Shared wrapt warnings alone do not establish failure causality.

## Comparators, recommendations and limits

MM1 remains heuristic. v17 changes tools and prompts, so its results are descriptive context rather than an isolated LoRA control. The new A/B retains identical v14 policies; a runtime-optimized competition candidate is a separate decision.

Before new outcomes, priorities are to preserve identical inputs/settings, use MM1 plus final-diff/verifier checks, distinguish unapplied writes and actual test returns, and enforce correctness/runtime gates on any submission variant. Full paired failure counts and causal conclusions are **BLOCKED** until new traces, patches and verifier outcomes are recovered. No training, weight change or Git publication occurred.


## Terminal CPU preflight result — 6 November 2026

**BLOCKED — implementation error in the new dry-run path.** Kaggle version 1 is terminal COMPLETE, but the evaluator child exited **1**. It failed at `from swegemma.models.registry import setup_gemma_model_registry` with `ModuleNotFoundError: No module named 'swegemma.models.registry'`. The restored scorer version does not provide this legacy API. No `paired_dry_report.json` or official compiler PASS exists. Notebook COMPLETE therefore does not mean preflight success.

Scorer preparation completed, test dependencies were found, base and full submissions unpacked, and the adapter was discovered from T1 output before this failure. The exception occurred after the script's adapter hash, 820-tensor, settings and task-identity assertions; this supports those checks, but official compilation remains unverified. The failure is in the newly authored preflight implementation; it is **not evidence of an incompatible T1 adapter or justification for retraining**. Earlier LP1/LP2/G1 evidence remains unchanged. Shared network cost-map warnings fell back locally and are not the terminal cause.

Original report/config/log recovery: `data/runs/atul_full_lora_new60_20261104/cpu_recovery_20261105T224049Z/`. The recorded run.log SHA-256 is `7a726a3d29d4f69b04195ebc925d5ce1d223d435e9835ea39495ebaa9ba29f2d`. Platform logs preserve the traceback and child exit code.

Per the registered failure-stop rule, **no GPU push, retry, training, conversion or competition submission was performed**. New A/B task counts remain UNKNOWN because no task evaluation occurred. Last actual account snapshot: 30.0 quota-hours available, 0.0 used and 0.0 reserved. The next necessary step is to replace the legacy dry-run compiler registry construction with the installed scorer's supported API and validate that preflight before any GPU launch. This turn stops at the recorded blocker without retrying. The scheduled monitor is being deleted to prevent automatic launch or repeated polling.
