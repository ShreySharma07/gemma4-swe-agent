# Three competing dataset designs

No training dataset is built or modified in this task. The [diagnostic quarantine](diagnostic_quarantine.json) identifies the new-60 tasks for exclusion; it is an analysis manifest, not a training artifact. The designs below have acceptance criteria and reproducible preparation steps. Qualifying data volume, target-environment teacher success and full licenses remain prerequisites.

## Design A — v14-compatible verified demonstrations

**Question:** can a small set of authentic, well-conditioned complete solutions teach useful repair without S3b's workflow collapse?

Use successful trajectories collected under the frozen v14 agent/tool schema, thinking enabled, thinking budget 4096, context 32768, output 8192, 8-minute and 28-chargeable-call budgets. Start with an acceptance target of 200 distinct training tasks from ≥20 repositories and a separate 40-task development pool. These are planning targets, not existing qualifying episodes or permission to collect them. The diagnostic new-60, old benchmark runs and gate prefixes cannot be repurposed as these tasks.

Each demonstration requires a pinned repository/base commit and environment hash; a nonempty implementation diff; a trace that shows the edited region and relevant callers/types; command/response IDs; actual test command, stdout/stderr and exit result; and successful independent oracle verification without editing evaluation tests. A final shell exit of zero or teacher's `resolved=true` is insufficient. Keep alternate valid patches; do not force a gold-file match as the only acceptance criterion. Preserve the original row, collection policy and trajectory hashes.

Keep investigation decisions needed for the fix, each source edit, relevant verification and justified termination. Context-only observations remain verbatim or reproducibly abbreviated with an explicit pointer to the original; they are never fabricated. Scratch/reproduction code may be context if it supplies evidence, but default loss on its creation is zero. Do not train a command template simply because it appears in all successful episodes.

If ≤8k training windows are required, use an observable factual state record plus the latest complete relevant source read before each supervised edit. The record includes issue constraints, evidence locations, current diff, failed attempts, last test status and budgets. It is derived only from earlier observations; it cannot contain the gold patch, future result or hindsight source location. If evidence cannot fit, exclude the target or split at a state-preserving boundary; do not hard-clip an edit's conditioning evidence.

**Selection criterion:** choose A only when enough authentic v14-compatible solutions exist and its opening/phase transitions remain close to a productive base policy. Advantage: direct policy compatibility. Risk: it reinforces only easy tasks and can exclude hard but valuable recovery behavior. Do not lower quality requirements just to fill 200 rows.

## Design B — authentic hard-state recovery episodes

**Question:** can targeted recovery support the transition that S3/S3b barely supervise?

Collect 100 distinct tasks with naturally observed failed assertions, wrong-file hypotheses, rejected edits, tool-argument errors, collection problems or context pressure. Target at least 25 genuine **post-source-edit failed-test→investigation→revised source** chains across multiple repositories before training this design. If those cases are unavailable, mark the design data-infeasible instead of substituting pre-edit test runs. End-to-end success is preferred; failed episodes may supply context and valid recovery decisions but cannot receive a successful-resolution label.

Annotate state at the time of each decision:

| State | Required evidence | Appropriate target classes |
|---|---|---|
| Assertion failed after actual source edit | Current source diff and specific assertion/traceback | Inspect cause/caller, revise implicated source, then reverify |
| Test could not collect | Import/API diagnostics and environment provenance | Separate unsupported API/patch regression from dependency issue; preserve uncertainty |
| Replacement assertion failed | Exact old snippet, file read and actual failure | Reread narrow region, correct snippet/arguments; do not repeat the same call |
| Gold-independent localization hypothesis contradicted | Source/test evidence visible then | Search/read a different relevant implementation path |
| Patch only contains a reproducer | Actual diff/status evidence | Continue implementation if supported; do not mark scratch activity as repair |
| Context headroom small | Actual usage/known bound, essential source state | Save concise factual evidence, reduce output; fresh-context handoff only if supported |
| Relevant test passed after valid change | Test identity, unchanged tests, nonempty source diff | Submit when coverage and task constraints justify it |

These are state-conditioned possibilities, not a fixed required script. An import error may be caused by a missing implementation symbol, as this audit found; labeling every import error “environment” would teach another failure. A failing test sometimes warrants inspection rather than an immediate edit. A successful test may leave a multi-file/API feature incomplete.

Sample by recovery type **and distinct task**, with an anchor of verified successful trajectories from A. Initially cap one accepted recovery chain and one terminal successful chain per task; report any repeats explicitly. Use modest bounded weights for rare valid decisions, not synthetic repetition until actions are equal. Keep the full predecessor state for recovery targets. Stop acceptance if the same test/submission shortcut dominates opening states.

**Selection criterion:** prefer B only if its failure outcomes are authentic, its repair decisions have visible supporting evidence and environment failures are classified reliably. Main risk: noisy failure attribution and synthetic state enrichment; validation must catch hindsight leakage.

## Design C — verified candidate selection before preference optimization

**Question:** can we improve candidate acceptance or build reliable preferences without imitating a teacher's whole workflow?

Use authentic candidate patches/trajectories for the **same issue, base commit, initial policy state, environment and resource budget**, from an independent task pool. Retrospective same-task v7/v14 or v7/S3b pairs on the new-60 are diagnostic examples only and must not enter training. Existing v7 repetitions cover few tasks and do not meet diversity requirements for this design.

First investigate rejection sampling or a deterministic verification-based selection procedure. Require a real oracle; do not claim a reward from a model-written success summary, nonempty patch, gold-file overlap or behavior-gate class. Candidates that fail due to an unstable environment do not produce confident quality preferences. Prefer a passing patch over a failing patch only when identical verifier conditions make the comparison interpretable. Ties and uncertain comparisons are omitted from preference loss or retained as unlabeled evidence.

Collection may require multiple rollouts, and inference-time candidate selection multiplies compute. This is a major competition cost/12-hour risk. The cheapest CPU phase is to audit already existing eligible candidates and oracle consistency; no new sampling is authorized here. A preference pair needs full provenance for both alternatives, including matched starting prefix if the objective is a next-action preference. Two full trajectories sharing an issue do not automatically form a valid preference at every intermediate state.

Only after reliable pair coverage and an adequate A/B anchor should a bounded DPO-style or other preference experiment be proposed. Serving-compatible LoRA preference training still needs implementation/memory validation; it is not implemented here. Online RL is deferred because reward quality, model-consuming rollouts, resettable environments and budget are unresolved.

**Selection criterion:** choose C only if matched oracle labels are reliable and comparable candidates exist at useful scale. Advantage: less direct imitation of undesirable teacher habits. Risks: environment confounding, length/format shortcuts, overfitting the verifier and multiple-candidate inference cost. A CPU-only ranking audit cannot establish model gains.

## Shared sampling and loss rules

1. Split before selecting trajectories. Group by repository/fork family, issue, base/neighboring commits and near-duplicate patch cluster. Keep every rollout of a task in one split.
2. Deduplicate exact raw/converted trajectories and normalized command-response sequences. Compute issue-text MinHash or character n-gram similarity and normalized-diff similarity for candidates; use a conservative screening threshold (e.g. Jaccard ≥0.8 as a proposed review trigger), then manually inspect cross-split near matches. Threshold is a screening policy, not a proof of independence.
3. Sample tasks uniformly first, then choose one qualifying episode/window per phase. Cap repository and repeated-task effective loss; report before/after weighting. An initial repository cap of 10% and ≤2 training presentations/task/epoch is a proposal to review against available diversity, not an established optimum.
4. Mask user/system/tool results. Supervise native tool calls and necessary concise factual state descriptions. Mask redundant reads/tests, unneeded scratch writes and unsupported success claims. Keep one terminal submission decision per completed episode; do not repeat it after every prefix.
5. Measure target tokens **and** sum of target fraction per scheduled window by action, task and phase. Match adapter comparisons on effective target exposure and input-token cost. Monitor first-action rates and pre/post-edit transition probabilities; a fixed global test-token quota cannot prevent a short test-call default.
6. Do not force equal numbers of reads, tests, edits and submits. Preserve valid workflow variability and branching. Inspect any large shift from productive base-rollout states before accepting the corpus.
7. Include failed-test recovery as real chains; do not transform passing feedback into a fake failure, invent a revised patch or suppress a tool error. Every summary must be reproducible from prior observed evidence.

## Thinking-enabled serialization and factual memory

The existing renderer's `THINKING=False` and the gate's thinking-off requests are mismatched to v14. A future converter should pin tokenizer/template revisions, v14 system prompt hash, complete declared tool order/schema (including analyzer tool if declared), reasoning parser and serving version. Verify native tool delimiters, masks and enable_thinking behavior on CPU. `declared_tools` currently warns that agent_tool schemas are missing; this gap must be resolved or explicitly omitted from both matched configurations.

Do not pretend to possess private chain-of-thought. Native exposed reasoning may be retained only if genuinely recorded and its provenance/terms allow use; this audit does not establish a corpus of permissible reasoning traces. Prefer concise factual state context over invented thinking. Serializing a thinking-enabled prompt without recorded reasoning is not evidence that training and serving are behaviorally aligned; an approved serving/full-task gate is still required.

## Independent evaluation protocol and reproducible preparation

Prepare a **new manifest**, without changing benchmark artifacts, with repository, fork group, issue ID, base/test commit, issue date, environment/package hashes, license/attribution, source row/revision, raw/converted hashes, collection policy, split, duplicate cluster and quarantine reason. Add language, changed-file/hunk counts, issue/patch size and dependency-depth estimates with their measurement method and missing-value flags. Do not fill absent provenance by guessing.

Proposed order:

1. Inventory and license-pin potential independent task sources. Exclude all 129 familiar benchmark tasks, the diagnostic 60 and gate tasks, plus overlapping fixes/forks; the new-60 are a subset of the repeatedly used public benchmark, not fresh holdout data now.
2. Freeze repository-group train/development/final assignments and hashes. Target ≥120 final tasks across ≥10 repository groups where feasible; retain an additional out-of-language group if available. A short development gate is separate from final evaluation.
3. Verify each environment with isolated empty/gold oracle runs and freeze rules for invalid tasks **before any candidate comparison**. Gold/test artifacts are sealed from agent input, training and retrieval. Oracle replay may itself require approved environment/network resources; none is executed here.
4. Collect or import authentic eligible trajectories; retain original provenance and real feedback. Replay source changes safely with pinned source blobs and audit-owned operations; do not execute untrusted teacher commands.
5. Derive states using only prior observations. Render matched tools/thinking settings; validate spans and labels; report per-window and scheduled effective-loss exposure. Produce reject reasons, not merely an accepted count.
6. Run exact/near-duplicate and contamination checks across all splits; have a second manual review of borderline clusters. Near-duplicate detection does not prove a pretrained model never saw a public repository.
7. Freeze the dataset, manifest, acceptance report, tokenizer/template and oracle versions before requesting training approval. Select dose/checkpoint rules without final-test feedback.

No truly independent final manifest, new verified recovery corpus or completed license audit is produced by this retrospective task. Therefore none of these designs currently supports an honest generalization claim. The preferred near-term path remains an inference-only experiment; A/B are conditional future research, and C is exploratory.
