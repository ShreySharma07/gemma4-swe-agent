# v14 optimization research handoff — October 20, 2026

**Recommendation: preserve v14 and first test a minimal inference-only bounded-output policy.** Context failure has a precise, reproducible boundary; training has repeatedly improved imitation loss without improving complete tasks. Correct repair remains the larger unresolved capability problem. Reducing overflow is a tractable first intervention, not a claim that eight additional tasks will be solved.

This is CPU-only retrospective research. No existing submission, dataset, configuration, result or historical report was changed. The only new files are in this directory. The user's read-only Git policy supersedes CLAUDE's requests to update HQ and the brief's fetch/worktree instructions. A read-only `git ls-remote` check found remote HEAD/main and local HEAD at `125ce5a3e35bf6c4254611cf2d0c07b347d5a10b`; no fetch was used.

## Findings with confidence

| Finding | Confidence and supporting evidence |
|---|---|
| v14 solves 30/60 versus v7 23/60; 22 both, 8 gains, 1 loss, 29 neither | High for these artifacts. Independently joined original v7 archive and v14 task rows; exact McNemar p=0.0390625. [CPU results](validation_results.json), [original report](../atul_v14_new60_report_20261019.md). |
| All eight context exceptions reject at least 24,577 input + 8,192 reserved output against 32,768 | High. All eight terminal errors agree; zero recorded compactions in all 60 traces. Successful-request prompt maxima understate the rejected prompt. [Failure analysis](failure_analysis.md). |
| Existing edit/verification diagnostics are incomplete | High. Python mutations are a supported v14 edit route but the old analyzer recognizes only write_file/edit_file. New audit identifies repository mutation intent in 57 tasks; 28 tasks have a recognized test or reproduction/verification-script invocation after the last recognized repository mutation. This is broader activity, **not 28 proven meaningful verifications**. [Tool](audit.py), [evidence](trace_audit.json). |
| The earlier four-task missing-wrapt explanation is unsupported | High correction. The wrapt warning occurs in **all 60 verifier logs, including all 30 successes**. Four collection-failure logs also expose missing API symbols/arguments; v7 has similar API errors and one explicit syntax error. Do not remove these tasks or blame wrapt alone. [Detailed correction](failure_analysis.md). |
| S3b still strongly teaches test-first behavior | High descriptive evidence: preserved E2 audit has pytest first in 371/826 supervised windows; new classifier sees executable pytest first in 382/826 converted episodes under a broader command definition. These are different definitions. S3b's 63.1% edit-token share did not control its rollout action mix. [Dataset audit](dataset_audit.md). |
| Thinking-only causality and hidden-repository gains are unestablished | High limitation. v14 bundles prompt and budget changes, and 50/60 tasks are FastAPI. Current Kaggle rules/leaderboard pages returned no readable body, so no new leaderboard score or fresh rule verification is claimed. |
| v14's worst-case runtime needs a separate competition decision | High arithmetic, using repository constraints: 120 × 8 min =16 h, exceeding a 12 h aggregate cap before overhead. Actual local chunk average is 3.85 min/task, but it is not a worst-case guarantee. |

The paired task bootstrap interval for the observed net gain is approximately **+3.3 to +21.7 percentage points** (20,000 resamples, fixed seed); v14's Wilson interval is 37.7–62.3%. These describe this public task sample and omit variation across runs/repositories. They are not a hidden-score forecast.

## Direct answers to the thirteen research questions

1. **Single biggest remaining failure reason:** failure to turn located evidence into a complete, correct implementation. Gold-file exposure is 59/60, yet 30 fail. Among failures, 5 patches are empty, 5 contain only scratch artifacts, and 20 contain implementation/documentation changes that do not resolve the task. These are observable outcomes; semantic understanding cannot be measured directly from them. Context failures overlap eight of the failures.
2. **Strongest fixable architectural weakness:** history growth with ineffective in-task compaction and an 8,192-token output reserve. The effective prompt boundary is 24,576 tokens, not the nominal 32,768. Context reduction must operate within the actual submission interface.
3. **Primary bottleneck:** repair quality is the broader capability bottleneck; inference context/output management is the best first engineering investment. Data quality is the primary prerequisite before any further training. These statements concern different decisions and are compatible.
4. **Why S3b failed despite its gate:** a one-response rubric cannot observe recurrent pytest selection or eventual source changes. Test-first state frequency stayed high; real post-edit failure recovery remained scarce. The adapter solved 9/60 versus v7 23/60, with 46/60 empty patches. This supports a distribution/control-flow hypothesis, not a unique causal attribution to one data field.
5. **Next dataset, if funded:** a small, provenance-pinned corpus of authentic v14-compatible successful and recovery trajectories from tasks outside every diagnostic/evaluation pool, with source visibility and oracle verification. Start with an acceptance target of 200 distinct tasks from at least 20 repositories, not a promise that qualifying data already exists. Details and two competing designs: [vNext](dataset_vNext_design.md).
6. **Add/remove demonstrations:** add real failed-test→diagnosis→revised-source→verification chains, safe handling of collection errors, rejected replacements, and budget-aware evidence retention. Remove repetitive test runs, fabricated success feedback, ungrounded edits, unrelated scratch-code targets and default test→submit scripts. Preserve useful pre-edit tests only when the state warrants them.
7. **Avoid repetitive pytest:** audit state-conditioned transitions and opening actions, deduplicate at task/commit/trajectory level, cap per-task effective loss, and mask redundant tests. Do not flatten the entire action distribution into artificial equal counts.
8. **Thinking compatibility:** use the exact v14 prompt/tool/template provenance and thinking-enabled serialization; supervise native observable calls plus concise factual state summaries. Do not synthesize hidden reasoning. Current renderer and one-step gate explicitly disable thinking; compatibility needs a new CPU serialization check and then an approved serving probe.
9. **Generalization:** freeze unseen tasks, repository groups and snapshots before development; quarantine the new-60 and old diagnostic/gate sets; near-deduplicate issues and diffs; verify the environment with isolated gold/empty oracles before evaluation. No independent final benchmark is established by this audit.
10. **Cheapest meaningful disproof:** first, CPU replay rejects output compression if it drops necessary error/patch evidence or cannot be expressed through the permitted scorer interface. If that passes, a separately approved 8-task-per-arm full-rollout smoke gate can expose collapse or context regressions. A small gate cannot establish a modest resolution gain; confirmation needs an independent paired set.
11. **Effort:** estimates for one engineer: 1–2 days for output-policy/ledger prototypes and evidence retention tests; 1–2 additional days for scorer-interface checks and a frozen small-task protocol. A verified diverse training corpus is roughly 1–2 weeks of curation plus environment work, highly dependent on available authentic recovery cases. These are planning estimates.
12. **Reject:** unchanged Step C/S3b retraining or scaling, loss-only selection, fixed edit deadlines, mandatory pytest at every prefix, larger call budgets as the main remedy, graph-heavy localization by default, invented preference labels, hidden-test/leaderboard tuning, and blind thinking/output reductions.
13. **Work first:** correct observability, review bounded-output safety, and freeze the independent rollout gate. Then request approval for one isolated inference-only experiment. Implementing runtime compaction or a training campaign is not the next authorized action.

## Handoff map

- [Per-task failures and all eight context cases](failure_analysis.md)
- [Step C/S1/S3/S3b supervision audit](dataset_audit.md)
- [Three competing dataset designs and contamination controls](dataset_vNext_design.md)
- [Architecture changes and ranked candidates](architecture_improvements.md)
- [Frozen experiment proposals, statistics and costs](experiment_matrix.md)
- [One-week roadmap and rule-verification limits](competition_roadmap.md)
- [Paper opportunities and causal claims still missing](research_paper_opportunities.md)
- [Reproduction, validation and file inventory](README.md)

**Stop after this handoff. No GPU run or Git write operation is authorized.**
