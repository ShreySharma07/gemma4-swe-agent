# Experiment matrix — proposals only

No GPU training/inference, notebook launch, leaderboard submission or new experiment is authorized or executed. Every GPU row below requires a separate explicit approval, fresh live quota, exact immutable settings and hard runtime/watchdog bounds. The last recorded 13.037684 quota-hours is historical capacity, not a fresh balance or permission.

## Baseline, tasks and metrics

Freeze the exact v14 configuration and source hashes from `input_manifest.json`: 31B W4A16, TP4, vLLM .19.1, model context32768, thinking4096, output8192, temperature .2/top_p .95/top_k40, no adapter, 8min/28 charged calls/80turns/240s command timeout. Keep the subprocess environment and overlay hashes identical in local treatment/control; record that this is not the scorer's Docker environment. Retain v7 as historical evidence only. Fresh baseline execution is necessary to compare any treatment on new tasks; the existing new-60 cannot serve as an independent final test.

A new independent manifest is a prerequisite. Freeze task IDs, repository/fork groups, base/test snapshots, licenses, environmental oracle outcomes, split membership and near-duplicate exclusions before generation. Gold patches/tests may be used by an isolated evaluator to establish valid environments, never shown to the agent or used as supervised features. The final set is sealed during development. If no independent pool exists, stop at diagnostic claims; do not relabel the old new-60 as independent.

Primary outcome is **resolved issues / all preregistered eligible tasks**. Freeze treatment-failure handling: timeout, context overflow, absent patch, malformed response and runtime failure count as unsolved; do not drop difficult failures after the fact. Environment exclusions require pre-comparison oracle criteria. Report intention-to-evaluate counts, missing artifacts and any protocol deviations.

Secondary metrics: nonempty **implementation** patch rate separately from scratch-only patches; gold-independent source localization where possible and retrospective gold file/line overlap only as a diagnostic; context-exception incidence; wrong-fix/API-regression rate; proposed, associated, charged and denied calls; repeated calls/output; explicit pytest versus probe frequency; post-edit meaningful tests; real failed-test→inspection/revision→reverification transitions; submission correctness; duration; input/completion tokens (thought tokens only if actually provided); GPU runtime and account quota deltas. Never use declining loss or fixed-prefix accuracy as primary acceptance evidence.

## Ranked candidates and isolated tests

| Priority / experiment | Hypothesis and exact isolated treatment | Control / tasks | CPU prerequisite and primary falsifier | GPU cost / complexity |
|---|---|---|---|---|
| P0 instrumentation | Python edit intent, source/scratch distinction and response association correct misleading diagnostics | Preserved outputs; no model | Current audit + synthetic tests; reject misassociation, false pass labels, raw evidence mutations | **0 GPU**, low; completed in this task |
| P1 output policy | A single bounded-output instruction reduces accumulated history without sacrificing repair evidence | Exact v14 except one prompt section; new independent development tasks | Review cap replay; assert relevant code, assertion, exit status and diff facts survive; reject if scorer-declarative interface cannot express the change | Evaluation only; low/moderate engineering |
| P2 factual state replacement | One supported bounded fresh-context handoff preserves the active repair state | v14 + P1 held fixed in both arms; treatment adds only handoff | Demonstrate real history replacement, no hidden gold, schema/compiler support, required-state invariants; reject added history masquerading as compaction | Evaluation only; moderate/high |
| P3 conditional recovery | Diff-aware termination and error-sensitive next actions reduce scratch-only/empty outcomes | v14 + chosen frozen output policy in both arms; one recovery prompt section | Retrospective state review without training on diagnostic labels; reject forced early edits or universal environment excuses | Evaluation only; low/moderate |
| P4 evidence retrieval | Reading actual API/caller contracts improves multi-file repair at matched budget | Same baseline; treatment adds selective source-neighbor retrieval rule only | Show marginal relevant evidence over narrow grep/sed on a new dev pool; reject generic graph overhead or exact-name query mismatch | Evaluation only; moderate; do not introduce unverified skill runtime |
| P5 verified A/B SFT | Small v14-compatible, state-preserving demonstrations improve full rollouts | Same frozen scaffold/base; only adapter differs, plus explicit serving identity | Corpus acceptance, licenses, state-conditioned transitions, scheduled loss-dose audit, rendering and oracle checks; reject if no authentic recovery/diversity | ~3–5 quota h for a hypothetical 1M-input-token pass **plus** serving/evaluation; high |
| P6 preferences/selection | Matched verified quality labels improve candidate choice without imitating full workflows | Same state/env/budgets and candidates | Reward/oracle reliability and pair comparability; reject uncertain environment comparisons and unmatched prefixes | Cannot responsibly fix a training cost before accepted pair lengths/counts; collection costs scale with rollout count; high |
| P7 hybrid | A minimal inference fix plus a proven adapter is complementary | Factorial control, inference only, adapter only, both | Only after P1 and P5 independently pass; reject confounding a bundle with a single mechanism | About four-arm evaluation cost; deferred |

All estimated costs below use the observed ~2× quota/log-hour relationship. Fresh platform accounting can differ. Dollars are not projected; the project cash cap remains a constraint. Training estimates count **input tokens**, not merely target tokens. No proposal consumes the historical balance automatically.

## E0: cheapest meaningful CPU rejection gate

Run the stdlib audit and tests on preserved evidence; inspect all eight context cases and a success/failure sample for information that a display cap would hide. Reconstruct the exact serving tokenizer/template if legitimately available and compare retained-input tokens. Pin the abbreviation algorithm and retain source pointers. CPU character-cap savings already produced here are a **scenario**, not this full preservation/tokenizer acceptance test.

Reject output management before GPU if any essential API/patch/failing-test fact is lost in the reviewed replay, if the ledger leaks future/gold information, if test exit codes become ambiguous, or if the proposed interface relies on custom scorer runtime support that does not exist. This test can falsify feasibility/information retention; it cannot prove resolved-issue improvement. Cost: zero GPU, estimated 1–2 engineering days for complete prototype/review.

## E1: first future full-rollout smoke gate

**Question:** does P1 produce usable complete trajectories without collapse? Use **8 new development tasks**, paired across frozen v14 and v14+one output-policy section, one full rollout per arm, order alternated by task. Choose tasks and available context/complexity strata before outcome observation. Do not choose only the eight historical context failures. Keep all other model/config/environment settings identical.

Predeclare a first-stage stop if treatment creates an obvious systematic failure (source-edit rate collapses; malformed/batched calls; repeated test loops; missing source patches on otherwise solvable tasks) or exceeds the fixed runtime cap. For a conservative advance gate, require no fewer resolved tasks than the matched control, no excess context failures, and no rise in empty/scratch-only patches. These are engineering acceptance criteria on a tiny sample, not statistical evidence of superiority. Do not change the rubric after seeing the results.

Planning estimate from the v14 run: roughly **1.2–1.8 aggregate session h /2.4–3.6 quota h** for both arms with setup uncertainty. A proposed aggregate hard cap is **2.8 session h /5.6 quota h**, split into explicit per-session bounds and with fresh quota ≥6.6h to preserve a 1h reserve. Per-task worst-case generation totals 16×8/60=2.133h before startup/setup, so a shorter nominal estimate cannot serve as a hard cap. Approval must resolve exact notebook watchdogs and enforce no automatic retry.

Full generation calls, raw arguments, response IDs/status, final diff and verifier logs are retained. A one-response gate is supplementary. If the runtime limit prevents complete paired coverage, report incomplete coverage and do not compare partial denominators. No automatic leaderboard submission follows a pass.

## E2: independent paired resolution test

Only after E1 and scorer-path feasibility: freeze **120 independent tasks across ≥10 repository groups where feasible**, with at least 20% multi-file/API/dependency tasks and stable valid environments. These are design targets requiring inventory validation. If only a smaller pool exists, report it, reduce the claim and recalculate power; do not manufacture tasks or silently reuse contaminated cases.

Run paired v14/control and one treatment, same serving stack, one prespecified seed/repetition initially and alternated order. Report both/gain/loss/neither; exact two-sided McNemar p; per-arm Wilson intervals; paired bootstrap interval for the resolution difference; and repository-cluster sensitivity intervals. Ten repository groups is still a modest number; show repository-specific results. With only three public repos, cluster uncertainty is not reliably estimated. If repeated runs are funded, preregister them and aggregate within task rather than treating runs as independent tasks.

Do not promise power for a score improvement whose discordance rate is unknown. With 10 discordant pairs, an 8:2 split has exact two-sided p≈0.109; 9:1 gives p≈0.0215. Small changes need more tasks or repetitions than a convenient 60-task set. Run a prospective exact-binomial power calculation over plausible discordant fractions **before approval**, choosing the effect of practical interest with the user; this audit does not invent an effect forecast.

Choose one primary comparison and alpha=.05 before execution. Apply Holm adjustment if several treatments share confirmatory status. Avoid repeated significance checks/optional stopping; operational failure stops are allowed but disclosed. For advance, require a positive prespecified paired resolution difference, acceptable uncertainty, no new systematic failure mode and runtime compatible with eventual scorer limits. A non-significant result is inconclusive unless it rules out the chosen practical effect; it is not proof of equivalence.

At observed public pacing, 240 task rollouts are roughly 16 session h /32 quota h with separate setup overhead. A conservative per-task bound at8min is32 session h before setup, so the entire plan needs explicit multi-session budgets and fresh quota allocation. The historical 13.04 quota h does not cover this confirmation. A 60-task development comparison would roughly halve cost but remain development evidence and still require new independent IDs.

## E3: future adapter gate and causal controls

For A/B data: preregister accepted tasks, training windows, one low-dose rank16/alpha32 schedule, exact learning rate/optimizer settings, token exposure and checkpoint-selection rule after the corpus exists. Compare to the same frozen thinking-enabled inference policy. Training loss is diagnostic only. First verify adapter export/hash and serving identity; then use a **12-new-task full-rollout gate** paired with the base, distinct from training/development prefixes. Predeclare nonempty-source-patch and loop/overflow non-regression plus resolved-count acceptance; inspect first-action/test frequencies over complete trajectories.

Do not promote a candidate because it emits an edit at an isolated prefix. Do not choose an earlier checkpoint after seeing final-test outcomes. If real-task behavior collapses, stop that recipe; preserve negative evidence without retraining it automatically. A successful small gate permits proposing E2; it does not authorize it.

To isolate the dataset mechanism, compare A versus A+authentic recovery with matched **scheduled normalized-loss dose** and task/repository counts, not merely matched input bytes. A separate opener-balancing ablation can test the test-first hypothesis. Do not change thinking, data, LR and scaffold together and attribute the result to one factor.

## Competition and approval boundaries

A local runner compaction change is a harness research experiment until scorer support is verified. Testing architecture using unavailable custom runtime would not produce a deployable competition candidate. Offline data accessibility, teacher-output terms, adapter constraints and submission format need live official verification before any proposed deployment.

For approximately120 tasks under12h, an8-minute per-task cap gives a16h worst case. Any later public-scorer configuration must satisfy a **separate** timing decision with setup margin; a cap near5–5.5min would bound generation around10–11h, but changing it now would confound the v14 comparison. Average local runtime alone does not guarantee scorer completion. No timing or deployment change is made here.
