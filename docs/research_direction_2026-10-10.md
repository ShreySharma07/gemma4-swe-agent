# Atul's recommended research direction — 10 October 2026

Recommendation for review, not a record of an adopted decision or a completed experiment.

## Recommendation

Prioritize reliable software-repair execution with the existing v2 scaffold and a small, targeted SFT LoRA. Keep one paper candidate: execution-derived necessity of edits as a yardstick for evaluating offline training credit. Continue validating the execution-graph instrument already frozen in `plan_autonomy.md`; use its measurements to decide whether that paper candidate deserves a larger experiment.

Suggested effort split: 70% competition pipeline and validation, 30% paper pilot. These are planning estimates. Do not start online RL, another multi-agent scaffold, or a large graph architecture now.

## Evidence behind the recommendation

The experiment log reports baseline 15/129 locally and 0.00 LB, v2 26/129 and 0.08 LB, v5 33/129 and 0.08 LB, and v6 0.00 LB. The paired v2-to-v5 change was +12/-5, p=0.14. These numbers support retaining v2 as the simplest leaderboard-tested baseline; they do not establish that graph methods cannot help.

Failure analysis reports 47/129 v2 tasks with a wrong fix never tested and only about 15% testing after the last edit. v5 reports malformed arguments in 61% of edit calls, concentrated in 14 tasks. Call-level percentages exaggerate prevalence when failures loop; report affected tasks and first-attempt error rates as well.

The Oct 9 training probe supports 4k/8k layer-split training at approximately 100/82 tokens per second, but 16k OOMed. FSDP and training-to-serving fidelity remain unresolved. The earlier feasibility document's 400–650 tokens/s and 16k memory estimates are projections, not measured capacity. Synthetic adapter serving is verified; useful trained-adapter transfer is not yet verified.

The behavioural-matching pilot scored 0.514 with repo-clustered 95% CI 0.40–0.59 and failed its registered threshold. Graph repair produced reach@2 0.65 to 0.71 on only 17 eligible tasks, with repair recall below its validation threshold. Respect both stopping decisions.

Execution-graph analysis reports only 17% provided-graph coverage of executed repo edges and 19/41 removable gold functions in multi-function fixes. These are provisional instrument-dependent observations. In particular, a removable gold function is not an agent step-credit label, and passing selected tests is not proof of semantic redundancy.

## First 48 hours: remove uncertainty cheaply

1. **Validate evaluation.** The checked-in L4 runner still has the older host-install workflow; the offline scorer-host and task-dependency environment described in HQ must be integrated and checked. Run empty/gold controls and record collection failures separately. Local subprocess parity does not prove Docker parity. Start with a small, fixed task subset before a full evaluation.
2. **Audit tool-call rendering.** Build a fixture set of intended calls with nested quotes, multiline replacements, backslashes, Unicode and long strings. Render through the actual Gemma tokenizer/template, then replay through the pinned scorer parser and schema validation. Inspect raw generated text on a small reproduction to distinguish model emission, truncation, parser and schema failures. Do not reconstruct supposedly correct replacements automatically when intent is ambiguous. Parser fixes used locally are diagnostics unless the official submission path permits them.
3. **Finish training feasibility.** Run the existing v3 probe with isolated stages and captured rank logs. Verify finite loss and gradients, peak memory, useful tokens/s, export, and inference through the official W4A16 route. If long-context training fails, begin with 4k–8k action windows containing sufficient issue, source and error context. Do not make the project depend on 16k–32k support.
4. **Check the execution instrument.** Confirm repo-frame filtering, repeatability and mapping on the v3 re-trace. Where tests execute in child processes, record whether they were traced; missing subprocess frames must not be interpreted as unexecuted code.

Deliverable: one short report saying which capabilities are proven and which remain blockers. No new leaderboard variant until its scorer path is validated.

## Competition experiments, in order

**E1 — Edit-format reliability.** Train a rank-8 or rank-16 LoRA on a proposed starter set of 500–1,000 diverse short examples, sized after the throughput probe. Include ordinary valid calls, difficult quoting, and correct recovery after an actual tool error. Deduplicate by source task; split repos and template families before generating variations. Match the scorer's rendering and tool schema. Mask tool observations and supervise assistant actions.

Use existing competition traces for diagnosis. Keep public competition tasks separate from the external training pool so historical comparisons remain interpretable. A repo-disjoint external evaluation reduces direct training overlap; it cannot establish absence of pretraining memorization.

Compare unchanged v2 with v2 plus the format adapter. Proposed engineering gate, registered before running: at least halve first-attempt malformed calls on held-out cases, with no loss of replacement-content correctness. Then evaluate closed-loop tasks at the same time/call limits. Better serialization alone is not sufficient evidence of better repair.

**E2 — Verification behaviour.** First compare unchanged v2 against the existing v3 verification prompt under a matched budget. If testing increases without improving resolution, inspect test relevance and available repair time. Only then add short successful reproduce–edit–test–repair examples to the SFT mix, retaining E1 as an ablation.

**E3 — Diverse successful repair trajectories.** Scale only after E1/E2 show a useful effect. Convert public trajectories to the actual tool interface, check licences and generation restrictions, and filter for meaningful executable repairs. Prefer one adapter with a controlled data mixture before experimenting with role-specific adapters.

For every comparison report paired task outcomes, exact McNemar results, affected-task error rate, empty patches, tests after final edit, repeated calls, latency and context failures. Tool calls within a trajectory are not independent samples; use repo/task clustering for uncertainty. A small 20–30 task pilot screens large regressions, not gains of a few percentage points.

Budget with measured throughput. For example, 500 examples × 4,000 processed tokens is 2M tokens: at 80–100 tokens/s, one epoch is approximately 5.6–6.9 node-hours before loading, evaluation and other overhead. Log processed tokens and supervised tokens separately. Clarify whether Kaggle quota is billed in session/node-hours or GPU-hours before scheduling.

## Paper candidate and stop conditions

Question: **Do offline credit signals prioritize the edits needed by the final repair, beyond what edit survival and simple diff features already explain?**

Use final-patch subset interventions and write provenance to label surviving edits. Call the estimand test-suite-relative contribution to the final artifact. Do not call it causal policy advantage or claim it measures the value of reads, failed attempts or all intermediate reasoning.

Run a bounded CPU pilot after instrument validation. Begin with a small replay smoke test, then the existing report's 100-trajectory pilot if replay works. Check full/empty patch behaviour repeatedly, exact replay fidelity, patch applicability, and provenance. Bound the number of edit units: exact Shapley requires exponentially many subsets. If sampling is needed, disclose approximation and uncertainty. Syntax-invalid or dependency-invalid subsets are not automatically evidence that an edit is semantically necessary.

Retain the report's proposed gates: replay fidelity ≥80%; at least 40 surviving-but-unnecessary steps and meaningful non-additive support; stop the headline audit if a held-out simple diff baseline reaches AUROC ≥0.90. Compare necessity with survival, diff features, shuffled labels, and one feasible credit signal. Split by repo and report uncertainty. Treat these thresholds as proposed protocol until formally registered.

If the pilot fails, retain a careful instrument/measurement analysis if the findings are reproducible and useful. Do not automatically launch another named research direction. If it passes, scale the audit; training with necessity weights is an optional later ablation at matched data and compute. A successful main-track adapter is not contingent on the paper pilot.

## Current literature checked for this recommendation

- [SWE-agent](https://arxiv.org/abs/2405.15793) already establishes the importance of the agent-computer interface.
- [SERA](https://proceedings.mlr.press/v306/shen26o.html) demonstrates efficient SFT-based coding-agent training. It motivates an affordable SFT baseline; it does not guarantee transfer to this quantized model or task distribution.
- [LocAgent](https://aclanthology.org/2025.acl-long.426/) already studies graph-guided localization. Merely adding a call graph is not a novelty claim.
- [TRIM](https://arxiv.org/abs/2607.18161) already minimizes trajectories to remove unnecessary code. An edit-removal instrument alone is not novel.
- [vLLM issue #54256](https://github.com/vllm-project/vllm/issues/54256) reports a Gemma parser failure involving reasoning-channel transitions. It supports checking the parser, but does not establish the cause of this repo's quote/key corruption.
- [Pinned vLLM 0.19.1 parser documentation](https://docs.vllm.ai/en/v0.19.1/api/vllm/tool_parsers/gemma4_utils/) is the relevant parsing reference for fixtures.

The earlier report names additional close work, including Counterfactual Supervision. Its OpenReview page was inaccessible in this check; do not treat earlier absence-of-prior-art claims or reviewer scores as independently verified. Read those primary sources before claiming novelty. Kaggle's overview loaded without readable details, so dates and detailed rules here remain repository-recorded, not freshly verified.

## Immediate order

Evaluation and parser audit → training/export probe → small format LoRA → matched closed-loop evaluation. In the paper allocation: validate execution traces → bounded edit-necessity pilot → retain or stop based on registered gates.

This checkout contains scripts, notes and small result artifacts, but not the raw competition snapshots or full trajectory archives. This review launches no GPU jobs, external APIs, training or submissions.
