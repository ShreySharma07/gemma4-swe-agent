# Gemma 4 developer-agent data and post-training research

**Atul Baliyan's research contribution — October 16, 2026**

**Status: research recommendations; no corrective implementation or new model experiment.**

## 1. Executive summary

Step D establishes a failure to turn investigation into repository-source changes: **0/114 resolved, 114/114 empty patches, no identified source edits, and 81 explicit submissions returning zero bytes**. The completed [Step D diagnosis](step_d_failure_analysis_20261016.md) is the primary experimental evidence. This report proposes a solution and tests; it does not replace or repeat that investigation.

The strongest next investment is **recovering and auditing the exact preprocessing chain**, not adding data indiscriminately or changing the optimizer immediately. Current source contains consequential fidelity gaps: converted responses are inferred rather than measured, later windows repeat the issue without preceding filesystem/action state, and role/mask boundaries are inferred from rendered text. Their prevalence and causal contribution in the actual 200-task corpus remain unverified. The converter already selects teacher-reported successful runs; “the dataset contains no editing” is not an established finding.

Three recommendations follow:

1. **Make every successful training edit and submission auditable against filesystem evidence.** Recover original rows, converted ATIF, rendered windows, tokenizer/template and environment revisions; compare regenerated arrays with the preserved NPZ. Replay validation is a separately authorized next step, not work performed here.
2. **Teach state-conditioned transitions through verified, budget-compatible episodes.** Preserve the evidence needed for locate→edit, failed-test→correct, and empty-diff→recover. Balance exposure by action, supervised tokens, and actual loss coefficients. Keep full workflow examples so “edit emphasis” does not become blind editing.
3. **Select adapters by behavior before aggregate loss.** Require actual tracked-source edits, appropriate verification, nonempty collected patches, and feedback use in a tiny fixed gate before any broad evaluation. Retain the current rank-16 SFT stack initially. DPO and online RL are conditional later options, not a remedy for an unvalidated data representation.

The last measured budget is **2.8758 Kaggle quota-hours**, not a fresh weekly allocation. No GPU job is justified merely because a point estimate fits that number. Artifact recovery and static audits cost **zero GPU quota-hours**. A proposed fixed-prefix inference comparison is approximately **0.5–1.5 quota-hours**, subject to separate authorization, fresh quota verification and enforceable local timeouts. It cannot establish end-to-end resolution.

## 2. Problem definition and previous results

The target behavior is:

**Understand issue → locate relevant source → make a justified source edit → test the changed code → recover from failures → verify and submit a nonempty patch.**

Testing before editing can be useful. The failure is never reaching a source edit, not the mere presence of tests. Nonempty patches are necessary on these repair fixtures, but are not sufficient: Step B included an edited, unresolved task.

| Evidence | Observed result | Interpretation |
|---|---|---|
| Step A/B | Five training steps on 40 own-agent windows; B preserved 3/8 nonempty patches and resolved 2/8 | The general serving/editing path can work. Training overlap and small sample prevent a generalization claim. |
| Step C | 200 tasks; 190 train/10 validation; 974/43 available windows; 122 optimizer steps; LR 1e-4; rank 16, alpha 32; 410 modules | Training/export worked. Scheduler actually used 972 train/40 validation windows, retaining all tasks. |
| C validation | 0.7718 → 0.2882 | Better teacher-forced likelihood on this representation, not proof of correct autonomous actions. |
| C resources | 7.8203 session hours, 15.64 quota-hours; peak 18.88 GiB; steady median 295.6 input tokens/s | Use measured quota accounting and input tokens for planning. |
| Step D | 0/114 resolved; all patches empty; 77 normally completed tasks still empty | Missing edits are not explained universally by context/time errors. |
| B/D same eight IDs | 3 versus 0 nonempty patches; 2 versus 0 resolves | Association with adapter/data/training changes; exact McNemar p=0.50 for resolution, not significant causal evidence. |
| Serving identity | C adapter hash verified; all 4,251 D steps with model identity route to `conv_lora`; B/D scaffold and six evalkit files match | Loading/routing success does not establish policy quality, but generic routing failure is poorly supported. |

Sources: [C report](step_c_20261015_report.md), [D report](step_d_d0_20261016_report.md), [diagnosis §§1–5](step_d_failure_analysis_20261016.md). Older guide/HQ statements that C was “untested” describe their historical time, not the current state. D0 was never run; no new60 baseline exists.

## 3. What the evidence proves—and leaves unknown

Use four evidence labels throughout: **observed** (retained experimental evidence), **static** (current code behavior), **published** (external result), and **hypothesis/proposal** (not demonstrated in this project).

**Observed:** D emits read/search/test behavior without identified source mutation; 81 live submissions already report an empty patch before packaging. All 28 context errors have 24,577 input + 8,192 reserved output = 32,769 against a 32,768 limit. Eight first responses contain 55–253 calls; 389 responses reject calls for budget exhaustion. The diagnosis distinguishes 5,684 shell proposals from 4,106 harness-counted calls. These counts must not be relabeled as executions.

**Unknown:** whether actual converted edits fail; exact phase/action supervision shares; whether special-token offsets mask the wrong boundaries; compiled train/serve prompt equivalence; whether an earlier C checkpoint retains editing; and whether quantization transfer changes action choice. No retained D worktrees or complete execution record exists for six aborted batches. The original tokenizer, rendered windows and converted 200 trajectories were not located in the earlier audit.

**Not established:** overtraining, universal mask corruption, all teacher feedback being false, a collector discarding successful edits, or LoRA universally increasing repetition. B has substantial repetition too; D's repetition share on B's eight IDs is lower while editing is worse. A declining validation curve does not distinguish these hypotheses.

## 4. Research passes and independent findings

The investigation used five specialized responsibilities, staggered within the available four concurrent slots. Three delegated agents completed dataset, pipeline and methodology memos after reading the complete diagnosis. The literature and synthesis agents were launched but hit the account usage limit before returning research. The primary researcher completed those two passes, including primary-source checks and a separate recommendation review. Thus the literature and synthesis passes are not independent agent corroboration. Findings are synthesized below and stress-tested in the roadmap.

| Responsibility | Main finding | Consequence |
|---|---|---|
| 1. Dataset design | Teacher success is a useful filter, but successful source-state transitions must survive conversion, windowing and environment changes. Removing failed actions can remove the context that makes recovery intelligible. | Prefer verified workflow episodes and corrected continuations, with repository/fork disjoint splits. |
| 2. Conversion/supervision | Synthetic responses and state-free later windows are real static risks. Training averages window means while validation averages supervised-token loss. | Audit effective training coefficients in addition to action and token shares; preserve exact artifacts. |
| 3. Post-training methods | Existing SFT is feasible; online agent RL requires an environment/rollout stack and useful positive rewards that D currently lacks. | Start with corrected SFT; gate DPO/RL on demonstrated editing and validated preferences/rewards. |
| 4. Literature | Environment-grounded SFT, curation and recovery are supported in other systems. Long-context, large-compute results are not evidence that this 8k conversion is faithful. | Borrow verification principles; do not import headline scores or large-scale recipes as predictions. |
| 5. Synthesis | Release gates must reject an active but non-editing agent. Pipeline correction, dataset redesign and objective changes answer different hypotheses. | Separate them into Paths A/B/C with predeclared controls and no automatic GPU progression. |

## 5. Literature review: what transfers to this failure

Primary sources checked October 16, 2026. Results below are authors' reported measurements under their own models, scaffolds and benchmarks, not expected Gemma performance. Older foundational objective papers are included where necessary. “et al.” abbreviates author lists.

| Work, authors, year and original source | Contribution and reported setting/result | Limitation; practical relevance |
|---|---|---|
| **Training Software Engineering Agents and Verifiers with SWE-Gym**, Jiayi Pan et al., preprint 2024 / ICML 2025 ([paper](https://arxiv.org/abs/2412.21139), [implementation](https://github.com/SWE-Gym/SWE-Gym)) | Executable real issue environments and successful interaction traces; paper reports 32.0% Verified with verifier-based inference scaling. | This includes selection/extra inference, not plain SFT alone. Supports environment verification; large repeated sampling is unsuitable for the remaining quota. |
| **SWE-smith: Scaling Data for Software Engineering Agents**, John Yang et al., 2025 ([paper](https://arxiv.org/abs/2504.21798), [code](https://github.com/SWE-bench/SWE-smith)) | Test-breaking synthetic tasks from 128 Python repositories; 50k instances; SWE-agent-LM-32B reports 40.2% pass@1 on Verified. | Synthetic mutations need realism/diversity checks; another scaffold and training scale. Use small verified tasks for transition coverage, not an assumed benefit from volume. |
| **R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents**, Naman Jain et al., 2025 ([v1 paper](https://arxiv.org/abs/2504.07164), [code](https://github.com/R2E-Gym/R2E-Gym)) | Executable task synthesis and complementary test/execution-free verification; v1 reports 34.4% pass@1 for 32B and 51% with hybrid scaling. | Version-specific numbers; learned verifiers can favor style and tests can fail to distinguish patches. Real replay and protected independent tests remain essential here. |
| **SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution**, Yuxiang Wei et al., 2025 ([paper](https://arxiv.org/abs/2502.18449)) | Rule-based rewards including similarity to developer solutions; Llama3-SWE-RL-70B reports 41.0% Verified. | Not proof of interactive filesystem-conditioned recovery; textual similarity can reject valid alternatives. Useful objective comparison, not a feasible immediate 31B Kaggle RL recipe. |
| **SWE-Master: Unleashing the Potential of Software Engineering Agents via Post-Training**, Huatong Song et al., 2026 ([paper](https://arxiv.org/html/2602.03411v2), [code](https://github.com/RUCAIBox/SWE-Master)) | Curated long-horizon SFT reports 57.8%, execution RL 61.4%, TTS@8 70.8% Verified. Difficulty-filter ablation: 54.2→57.8. | Its long trajectories and extra inference differ sharply from 40 calls/8k training. Its RL truncation-mask findings conflict with another recipe, showing such rules are initialization-dependent. Supports controlled curation, not copying RL defaults. |
| **When Agents go Astray: Course-Correcting SWE Agents with PRMs**, Shubham Gandhi et al., 2025 ([paper](https://arxiv.org/abs/2509.02360)) | Taxonomy-guided process feedback; closed-source PRMs improve Verified resolution 40.0→50.6%. | Additional inference model/cost and different policy. Borrow failure taxonomy for recovery data; a second online critic is not a demonstrated fit for offline Kaggle serving. |
| **Wink: Recovering from Misbehaviors in Coding Agents**, Rahul Nanda et al., 2026 ([paper](https://arxiv.org/abs/2602.17037)) | Asynchronous interventions for specification drift, reasoning and tool failures; over 10k production trajectories; reports 90% recovery for misbehaviors requiring one intervention. | Conditional misbehavior-recovery rate, not issue-resolution rate. Supports explicit recovery evaluation; production system is not our declarative scorer. |
| **Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents**, Yueqi Song et al., 2025/2026 revision ([paper](https://arxiv.org/abs/2510.24702)) | Intermediate representation unifies 13 datasets; reports roughly 20% average improvement over corresponding bases across domains. | Aggregate across unlike settings, not Gemma SWE resolve gain. A common schema helps provenance but cannot certify semantic tool equivalence without replay. |
| **Revisiting Catastrophic Forgetting in Large Language Model Tuning**, Hongyu Li, Liang Ding, Meng Fang and Dacheng Tao, 2024 ([paper](https://arxiv.org/abs/2406.04836)) | Studies loss-landscape flatness and sharpness-aware minimization on three fine-tuning datasets. | Not a coding-agent diagnosis; abstract provides no single portable effect size. Motivates retention controls, not labeling C as forgotten/overfit or adding a more expensive optimizer first. |
| **SWE-rebench OpenHands trajectories**, Nebius, 2025 dataset release ([pinned card](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories/blob/ebb5599ac2b045fe09e98540519bfc60a53d8e0f/README.md)) | Qwen3-Coder-480B/OpenHands v0.54.0 trajectories: 67,074 total, 32,161 successful; fields include original conversation and model patch. | Dataset statistics and teacher verdicts are not replay results for our converted subset. Pin the exact source revision; this cited card revision is not asserted to be C's original revision. |
| **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**, Rafael Rafailov et al., 2023/2024 revision ([paper](https://arxiv.org/abs/2305.18290)) | Uses preference pairs to optimize a classification objective without online sampling or a separate reward model; original experiments address sentiment, summarization and single-turn dialogue. | No SWE-agent result or Gemma memory estimate in that evidence. Useful for matched-state edit/empty-finish preferences only after both branches are verified. |
| **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**, Zhihong Shao et al., 2024 ([paper](https://arxiv.org/abs/2402.03300)) | Introduces GRPO; reports 51.7% MATH for DeepSeekMath 7B without voting/tools and 60.9% with 64-sample self-consistency. | Combined math-data/training result, not an isolated GRPO effect or repository-agent test. Avoid treating critic removal as proof that rollout training fits our quota. |

The direct common lesson is to verify the environment-action-state chain. None of these papers proves the cause of D. Do not use an LLM's completion claim or a simulated tool success as equivalent to a measured source diff.

The [Nebius original technical report](https://nebius.com/blog/posts/openhands-trajectories-with-qwen3-coder-480b) supplies especially relevant counterevidence to categorical blame of the raw dataset: its 30B RFT model reports 49.7% pass@1 at 100 turns and 50.3% at 500 turns on a 484-task Verified subset, with 131k maximum training sequence length. It excluded 16 evaluation tasks for infrastructure reasons. Its patch-applicability filtering and successful RFT show that the source can be useful; neither validates our conversion into 8k windows and 40-call execution. The report also explicitly warns that tool-argument deserialization and scaffold versions affect behavior.

## 6. Detailed critique of the existing pipeline

### 6.1 Conversion: syntax fidelity is weaker than execution fidelity

[`posttrain/convert_public.py`](../posttrain/convert_public.py) filters unresolved/non-submitting runs, excludes the four public evaluation repositories, rejects some unsupported/interactive operations and caps converted calls at 40. These are valuable safeguards and contradict a simplistic claim that all data is unsuccessful or unrestricted.

However, `replace_script` checks exact single occurrence against the actual file only **when executed**; the converter does not execute it. Its response is generated from a teacher response containing “has been edited.” `create` always synthesizes success; `execute_bash` defaults to exit 0 when no exit footer is found; `insert` can synthesize an IndexError even though Python `list.insert` clamps out-of-range indices. `finish` creates submission size from the original stored patch rather than the diff produced by converted actions. These are static semantic gaps, not measurements of their frequency in C.

Other risks: editor paths are directly interpolated into Python/shell strings; teacher cwd/interpreter/session state may not match a fresh `run_command`; global path rewriting can affect output text; directory views and truncated output may omit needed information; think/task-tracker/prose is dropped. A round-trip test of quoting the replacement strings does not verify path handling, file existence, mutation outcome or final patch equivalence.

**Required audit:** compare original and converted action prerequisites, cwd, source hashes, exit classification, observed modified files and final diff. Distinguish shell proposals, executed calls, edit intent, source mutation, fixture/test-only changes, retained collector patch and verified resolution. Never train a successful source-edit label from command text alone.

### 6.2 Windowing: a later action needs its preceding state

[`render_atif.py:107`](../posttrain/render_atif.py#L107) starts every window with the original task message and a consecutive subset of later steps. It does not carry prior edits, current file contents, earlier failed tests, or budget consumption into later windows. A test/submit target can therefore be conditioned on what looks like a fresh issue. This is a demonstrated design property; the number of harmful windows is unknown.

Conversely, not every later window is invalid: it may contain enough local observations and its own edit to be self-contained. Classify sufficient state by dependencies, not by window number alone. Dropping malformed calls and observations may likewise remove the cause of a later correction. Retain such failures as masked context when the continuation depends on them; do not positively supervise the failing action.

The renderer can emit an overlong singleton; tokenization uses a silent `[:8192]` cap. **The new read-only NPZ audit found maximum length 8,191, with no row at 8,192. Under the inspected code, any clipped row would have length 8,192. Therefore this hard clipping is excluded as a C explanation, conditional on matching preprocessing-code provenance.** Keep it as a future robustness gap. Output trimming at 5,000 characters and omitted preceding history remain separate issues from token truncation. The renderer estimates length with default special-token handling while final tokenization uses `add_special_tokens=False`; the historical tokenizer is needed to establish whether those counts differed.

### 6.3 Rendering, turn boundaries and masks

The renderer uses native chat templates and `enable_thinking=False`. It warns that the analyzer `agent_tool` schema is omitted. The current v7 sampling file explicitly includes `thinking_budget: 0`; the server default is thinking enabled, so **the final compiled request**, not the default alone, decides fidelity. Historical prose in CLAUDE about the older sub-agent validation behavior must not override inspected current configuration.

`model_spans` scans for textual model-turn and response delimiters. `tokenize_windows.py` marks a token from its start offset; special tokens may have unusual offsets, and content can contain delimiter-like strings. `TrainData.example` shifts labels to predict the next token and masks from the corresponding next-token mask. That shift is coherent in code, but only token-level fixtures can verify boundaries under the actual tokenizer/template.

Required fixtures: one call→one observation→next call; successful edit→test→submit; failed edit→corrected edit; empty submission→recovery; assistant final text; quoted delimiter strings in source/output; missing observation; multi-call input; Unicode; a singleton just below/above the limit. Compare the training prefix to the **serving generation prefix** byte-for-byte/token-for-token; distinguish expected training response endings from the inference open assistant prefix. Confirm correct end-of-call, response and turn tokens are learned exactly where intended, and environment responses carry zero action-loss weight. Do not guess token IDs or change stop tokens from memory.

There is also a general response-association defect in `conversations`: it resolves a response's call ID but appends its message using the last assistant call's ID. Multiple calls on one step can therefore misassociate the first response and omit later responses. Converted public traces alternate singleton calls and responses, protecting that path; **this is not an established cause of C or D's batching**. The converter separately serializes original teacher batches into alternating calls/responses, potentially implying feedback-conditioned decisions that were originally generated together. Recover original decision-group IDs and test both input forms before reuse.

### 6.4 The actual objective and what loss reduction means

`MaskedCE.forward` in [`trainkit/model.py`](../kaggle/train/trainkit/model.py) returns mean CE over supervised positions in each window. [`worker.py`](../kaggle/train/trainkit/worker.py) averages these across ranks and accumulated microsteps. Validation instead sums loss×supervised-count and divides by total supervised tokens. Training therefore optimizes approximately **equal-window mean loss**, while validation reports **token-weighted mean loss**.

For action category `a`, compute all three quantities on the **scheduled training windows**:

- Action count/share: number of eligible actions of category `a`.
- Token share: `sum(n[w,a]) / sum(n[w])`, with `n[w]` the shifted supervised-label count.
- Objective coefficient share: normalize `sum(c[w] * n[w,a] / n[w])`, where `c[w]` includes actual sampling multiplicity, rank and accumulation factors. This is a coefficient audit, not the realized gradient norm or causal influence of an action.

A short final-summary window can receive as much window weight as a longer editing window. That does not establish that summaries dominate C; it establishes why token counts alone cannot answer the question. Include task-level macro statistics so long multi-window tasks cannot silently dominate. The final partial accumulation and discarded incomplete rank groups must be recorded, not confused with missing attached data.

The 1,017-window NPZ contains 7,372,758 input and 1,170,308 masked-in tokens (15.9%) according to preserved summaries and the completed structural audit. Binary masks and nonzero supervision do not identify phase labels or prove correct boundaries. Shared tool syntax, repeated prefixes, predictable tests and final prose can become easier to predict while edit decisions deteriorate. This is a possible explanation for loss/behavior divergence, not a decoded finding about C.

The new stdlib-only audit measures supervised labels per window: **minimum 23, median 895, maximum 4,726**. Counts by bins are 49 windows with 1–99 labels, 365 with 100–499, 109 with 500–999, 273 with 1,000–1,999, and 221 with ≥2,000. The scheduled 972 training windows contain **1,114,033 supervised labels** (median 879). Thus a token in the smallest-supervision window has up to **205.5×** the nominal within-window coefficient of a token in the largest, before accumulation, clipping and content differences. This does not reveal whether those tokens are edits, tests or summaries. The first position is masked in every window, so shifted and stored supervised totals agree. See the [reproducible CPU audit](atul_data_training_cpu_audit_20261016.md).

### 6.5 Evaluation and adapter loading

[`runner.py:69`](../kaggle/eval_l4/evalkit/runner.py#L69) checks results, some non-error result, tool activity, saved traces and verifier execution. It does not require a source mutation or nonempty patch. All five checks can pass on D by design.

[`analysis/trace_analyzer.py`](../analysis/trace_analyzer.py) and [`analysis/compare_runs.py`](../analysis/compare_runs.py) identify named edit tools but miss the v7 shell-edit route. Their existing post-edit testing metric is therefore unsuitable as the release gate. Future measurement should use actual file deltas plus ordered call/observation records. Reuse the completed diagnosis instead of rerunning those flawed metrics over all D tasks.

`Submission.fetch_adapters` chooses the first matching attached adapter if a copy is not shipped. This is a future reproducibility hazard if multiple sources exist, but D's exact adapter was already verified. Export integrity, module shape and routing checks must remain, alongside semantic behavior checks; do not reframe verified D identity as an open root cause.

## 7. Proposed improved dataset construction process

**Raw trajectories → provenance verification → source-edit replay → behavioral labeling → quality filtering → state-preserving rendering → token/mask validation → training → serving smoke test → edit-behavior gate → independent evaluation.**

Every arrow has a retained manifest and a rejection reason. The following is a proposed process, not an implemented pipeline.

| Stage | Verification artifact and acceptance condition | Cost/authorization |
|---|---|---|
| Raw ingestion | Pin dataset revision, row/trajectory IDs and hashes; save raw tool schemas, teacher version, issue/base commit and license attribution. | CPU/data retrieval; no teacher generation authorized. |
| Provenance | Resolve repository/fork identity; verify snapshot hash; deduplicate issue/patch families before splitting; exclude evaluation repositories and duplicates. | CPU; missing source is quarantined, not silently substituted. |
| Source-edit replay | In disposable isolated environments, replay approved converted actions; record cwd, before/after source hashes, exit status and diff. Compare intended final source tree with teacher patch applied separately to the same base. | CPU/GPU=0 for replay; environment retrieval and execution require separately scoped authorization. Never execute saved commands on the host workspace. |
| Behavioral labeling | Label observed locate/edit/test/recover/submit transitions; separate fixtures from source and proposal from execution. Record causal predecessors. | CPU plus manual review of ambiguous cases. |
| Quality filtering | Positive examples need real source edits and valid collector patches; success tier additionally needs independent tests. Failures may supply masked context or rejected preference branches, not positive completion targets. | CPU; preserve all excluded raw data and reason codes. |
| State-preserving rendering | Prefer complete native episodes ≤8k. Otherwise preserve relevant prior tool exchanges, current snippets/diff and failures. Every target's prerequisite state must be available. | CPU; reject unsafe cuts instead of silently chopping actions. |
| Token/mask validation | Exact template/tokenizer hashes; assert length, turn boundaries, tool-response exclusion and full action coverage; compare arrays and label shift fixtures. | Tokenizer/stdlib only; no model or PyTorch needed. |
| Training | Fixed manifest, schedule, coefficient audit, checkpoint budget and behavior-retention set; retain all exports and resource counters. | Future GPU, separately approved. |
| Serving smoke | Exact exported hash, target module shapes, parser/tool schema and final request consistency; replay a valid source-edit proposal. | Static part CPU; generation GPU separately approved. |
| Edit gate | Actual tracked edit + appropriate test after final edit + nonempty collector patch; recovery fixtures and bounded calls. | Tiny inference gate before broader evaluation. |
| Independent evaluation | Frozen repository/fork-disjoint holdout, matched controls, same environment/budget, independent test outcome and paired statistics. | Later budget approval; no D/D0 retry implied. |

### Selection and diversity

Start with existing authentic trajectories whose converted edits can be verified. Do not generate thousands more teacher runs before measuring usable yield. Select at the issue/repository level before windows; cap repeated easy tasks. Own v7 successes are useful format examples but historically represent only 21 distinct tasks, with nine tasks contributing many successes. They cannot also be an independent test set.

Use bug fixes, small features, single- and multi-file changes, stale exact-match repairs, syntax/test regressions, missing-path recovery, empty-diff recovery and dependency/environment failures. A fixture should never teach modifying protected tests or fabricating successful verification. Environment errors need a distinct label so the model does not repeatedly “fix” source for an unavailable dependency.

Proposed first redesign pilot: 40–80 windows across at least 20 distinct issues and multiple non-evaluation repository families. This is a pipeline/behavior pilot, not sufficient generalization data. A provisional target mixture is 50% complete compact success episodes, 30% successful recovery continuations, 20% evidence-to-edit decision episodes. These are sampling hypotheses, not literature-derived optimal ratios. Report resulting action/token/coefficient shares and compare to unweighted verified SFT; do not force exactly equal phases. No final-only episode may pretend to start from an unchanged repository.

### State-preserving choices under the measured 8k limit

1. **Complete compact episodes:** highest fidelity and simplest; biased toward easier/shorter tasks. Establish this baseline first.
2. **Overlapping native history:** retain the relevant earlier edit and observation, but mask already-supervised overlap to avoid multiplying its weight. Costs more tokens and can still omit distant dependencies.
3. **State-restored decision episodes:** supply an explicit current-state observation grounded in a replayed snapshot, then supervise one transition or short continuation. Never insert future patch/test knowledge. A new summary format creates train/serve shift unless the deployed agent actually receives that representation; under a frozen scaffold prefer existing native tool observations and skip windows that cannot fit.
4. **Longer contexts:** not an immediate option. Project probes OOM at 16k even with offload; external 80k/131k recipes do not remove that limit.

For recovery, keep the failed action and observed error as context with loss zero on the failed action, then supervise the verified corrective continuation. For empty submissions, contrast unchanged source/zero diff with edited source/nonzero diff. Do not teach that every empty diff is malicious or that any arbitrary edit is a solution; these fixtures explicitly require a code change.

## 8. Recommended schema and trajectory-quality criteria

An intermediate schema should make claims inspectable before rendering. ATIF can remain the interchange layer; add a versioned sidecar rather than destroying original traces.

| Record | Required fields |
|---|---|
| Identity/provenance | schema version; source dataset/revision; raw row hash; full trajectory ID; issue ID; repository canonical/fork-family ID; base commit/snapshot hash; teacher model/scaffold/version; license/attribution |
| Environment | image digest; dependency lock/hash; interpreter; cwd; tool schema hash; timeout/output limits; collector version; allowed/protected paths |
| Action | unique call ID; action index; original and converted tool/args; phase labels; target path class; expected prerequisites; execution status; start/end times; remaining calls/time |
| Observation | paired call ID; actual exit code/stdout/stderr and truncation marker; before/after source hashes; observed diff; origin=`measured` or `synthetic_unverified` |
| Outcome | source change count; full and collector-filtered patch hashes/bytes; independent test IDs/results; environment failure class; claimed versus measured success; recovery links |
| Window | parent trajectory ID; included message/action IDs; window start state; preserved dependency IDs; snapshot/diff references; token count; dropped-context reasons; target action IDs |
| Supervision | tokenizer/template hashes; token IDs; offset mappings; per-token role/phase; loss mask/weight; shifted-label checks; boundary assertions; scheduler inclusion and multiplicity |
| Split | train/dev/test; repository/fork cluster; issue/patch-near-duplicate cluster; date policy; contamination checks; immutable manifest hash |

Three quality tiers: **syntax-valid** means parseable, **execution-valid** means measured state transitions/collector consistency, and **task-valid** means independent task tests pass without prohibited edits. Only task-valid examples receive a “resolved” positive label. Execution-valid but unresolved trajectories can teach narrow verified operations or provide negative branches, with their limited label scope explicit.

Acceptance for positive resolved trajectories: every claimed successful source mutation is measured; each observation pairs to the right action; final patch survives collector rules and applies to a clean base; gold/known fix passes and empty fails for the evaluation environment; protected files are excluded from success evidence; recovery has a real predecessor failure; no future knowledge enters prompts; every supervised target is complete and causally supported. One disagreement quarantines that example until explained.

## 9. Recommended post-training methodology

Retain the known-working NF4/FSDP rank-16 LoRA stack and native v7 edit tools for the first corrective experiment. This avoids changing data, optimizer, adapter capacity and serving interface together. Keep successful source-edit behavior as the first gate, with resolution as a separate gate.

**Initial objective:** verified SFT with explicit window-level accounting. Add phase/task weighting only after the recovered corpus demonstrates imbalance or a controlled pilot supports it. A possible action-normalized objective averages token CE within each action and then weights actions/episodes; this prevents long prose from dominating solely through length, but can overweight trivial short actions. It is a testable alternative, not an automatic correction to equal-window loss.

**Retention:** include a small, disjoint set of verified native-tool examples as a training mixture only in a separate ablation. Keep a distinct held-out retention gate. Do not mix Step B test IDs into the training set and then claim preserved generalization. Fewer steps or lower LR may preserve behavior, but cannot correct false feedback or missing state.

**Checkpoint selection:** predeclare one early checkpoint (C step 40 is retained) and final C for diagnostic comparison; do not sweep all checkpoints until one happens to pass. For new training, reject checkpoints with lost editing even if validation loss improves. Report phase-specific loss alongside autonomous outcomes; a supervised edit token is not an executed edit.

**NF4→W4A16:** the project measured 0.817 versus 0.851 top-1 agreement with served W4A16 for NF4 versus bf16 QAT on a small text probe. This demonstrates numerical disagreement, not its effect on editing. Step A/B's successful edits through the same transfer path are contradictory evidence to a universal transfer failure. A later controlled test must compare identical token prefixes and the same adapter across training and serving bases, after prompt equivalence is established. Do not switch to an unproven memory-heavy training base on the strength of this hypothesis.

### Controlled adaptation ablations, only after data validation

| Ablation | Evidence and contrary evidence | Cheapest discriminating test; expected compute and gate |
|---|---|---|
| Fewer updates | C40 exists; A edited after five steps. Different data prevents attributing behavior to update count. | Compare one retained early checkpoint with final C at fixed sufficient-state prefixes (E3); no training. Editing restored early supports a dose association; it does not isolate LR from changed trajectory exposure. |
| Lower LR | 1e-4 updates all 410 modules; smaller changes may preserve base behavior. No high-LR failure has been proven. | One future fixed-data/update comparison, e.g. 3e-5 vs 1e-4, same seed/rank/alpha. Each tiny training arm ~1.4–3.0 quota h plus serving. Require restored edits/recovery and retained task correctness; lower CE alone fails. |
| Lower rank | Rank 16 is proven to fit/export; rank 8 would approximately halve adapter storage. Lower capacity could preserve behavior or underfit. | Export/shape checks on CPU first; later r8/alpha16 versus r16/alpha32 at matched data/steps, preserving alpha/r=2. Training runtime is still base/activation dominated, so do not promise a 2× speedup. Budget each arm separately; same behavioral gates. |
| Adapter strength | A reversible delta scaling can test whether behavior tracks adaptation magnitude; interpolation can also destroy useful circuits or native syntax. | Only if the serving API supports verified scaling, compare fixed strengths 0, 0.5, 1 on identical decision prefixes. Otherwise prepare separately named scaled copies after authorization; never alter original C. ~0.5–1.5 quota h diagnostic scope, not an automatic sweep. Gate: valid edits/recovery without parse regressions; a middle-strength win is suggestive, not proof of overtraining. |
| Retention mixture or KL | Behavior preservation is plausible given B/D association; raw base policy also has empty-submit/repetition failures. | First CPU inventory of disjoint verified native traces. Test a small retention-mixture arm before a reference-logit KL implementation. Same-token SFT cost plus any reference pass; require no regression on a separate retention gate and improved missing-edit fixtures. |

Do not run this table as a grid. Pick the one experiment supported by the preceding audit, predeclare it and retain a control. Expanding rank, adding modules or increasing epochs has no direct evidence-backed priority for the missing-edit failure.

## 10. Competing dataset and training alternatives

### Dataset alternatives

| Alternative | Expected benefit against D | Counterevidence/risk | Cheapest falsification and success criterion |
|---|---|---|---|
| Existing successful public rows, replay-validated | Reuses diverse teacher data; detects converted non-edits/false success | Successful teacher runs may already replay correctly; no policy fix if phase representation is wrong | CPU matched replay: zero unexplained state/result mismatches in accepted data; if all pass, demote conversion-failure hypothesis. |
| Complete compact native episodes | Preserves edit→test→submit dependencies without new summary interface | Easy-task bias, low yield | CPU measure retained issue/repo/recovery coverage at ≤8k; fail if coverage becomes too narrow; later same-budget behavior comparison. |
| State-preserved/overlapping decision windows | Retains essential previous edits and failures under 8k | More context cost; summaries can introduce serve mismatch or future leakage | CPU dependency audit: 100% targets have prerequisites; compare to existing windows on fixed decisions later. |
| Own-agent rejection-sampled successes | Native format and tool behavior | Only 21 unique historical successful tasks; strong memorization/duplication risk; B also repeats empty submits | Deduplicate and audit held-out family coverage first; reject using overlaps as generalization evidence. |
| Synthetic verified repairs/recovery | Cheap controlled coverage of empty diff, edit assertion and regression recovery | Toy-task shortcuts; teacher-free correctness only on narrow fixtures | Reject any task where empty patch passes; validate transfer on real unseen issues, not only synthetic fixtures. |
| Larger unverified corpus | More surface diversity | Can amplify false success, phase bias and compute waste | No recommendation to train; first measure verified yield and effective action coverage per compute unit. |

### Training alternatives

Costs are planning assumptions, not measured new-method benchmarks. Same-token estimates exclude new teacher rollout generation unless stated.

| Method | Data/objective and benefit | Risk, implementation and memory fit | Compute and decision |
|---|---|---|---|
| Verified SFT | Replay-validated successful actions; masked CE; teaches actual edits and sequencing | Lowest change; known 8k/r16/FSDP fit, ~18.9 GiB measured; bad labels remain harmful | Tiny 40–80-window pilot ~1.4–3.0 quota h plus serving; full C-size anchor 15.64. First choice after CPU gates. |
| Rejection-sampled SFT | Accept environment-tested successful rollouts | Filtering existing data is cheap; new policy sampling costly and can reinforce easy-task bias | Training similar to SFT; own previous five-pass sampling took 9.7 session h. Reuse verified data before generating more. |
| Quality/phase-weighted SFT | Weight measured edit/recovery actions or select balanced episodes | Low/moderate code change; wrong weights teach blind edits or lose localization. Similar memory at same length | Similar token budget; CPU coefficient audit first. Compare one weighting change against verified unweighted SFT. |
| Preference optimization/DPO | Chosen vs rejected continuation from identical state; prefer successful edit/recovery over empty finish | Needs verified pairs and sequence log-prob objective/reference; current CE wrapper is insufficient. Sequential pair processing/cached reference avoids assuming two bases fit | Rough planning 2–3× SFT compute per matched decision example when each branch has comparable length, including reference processing; token/implementation dependent, no measured fit. Conditional only after SFT gate. |
| Policy-preserving adaptation | Smaller update dose, native example replay, or KL to a fixed reference | Low-dose easiest; KL needs reference pass/logits and correct masks. Can preserve an inadequate base policy | LR/steps no extra memory; KL adds reference compute/storage. Prefer one early-checkpoint diagnostic before training a grid. |
| RL with verifiable rewards | Reward independent tests after real source editing and collection | Needs sandbox rollouts, reward validation, policy log-probs, scheduler, export; sparse success at D policy; reward hacks possible | Not budgeted for current quota. Collect a tiny reward-yield estimate only under future approval before estimating full training. |
| GRPO-style RL | Group-relative advantages from multiple environment rollouts; avoids separate learned value model | Still requires policy/reference/inference and long trajectories; all-zero groups have no useful relative reward; 8k training versus longer rollout context mismatch | Group size multiplies rollout work; no current 4×L4 compatibility proof. Defer, not a “cheap RL” replacement. |
| Curriculum | First verified source edit; then failed-test correction; then full workflows | Multiple phase changes confound attribution and can forget earlier phases | CPU episode design first; eventual same-token comparison retaining all earlier skills in mixture. No new training now. |

DPO's foundational source is [Rafael Rafailov et al., Direct Preference Optimization, 2023](https://arxiv.org/abs/2305.18290); GRPO originates in [Zhihong Shao et al., DeepSeekMath, 2024](https://arxiv.org/abs/2402.03300). Neither provides a Gemma-4 SWE cost estimate. The numbers above are engineering estimates constrained by this project's measured SFT envelope.

For clarity, proposed DPO would optimize `-log sigmoid(beta * [(log pi(chosen|state)-log pi_ref(chosen|state)) - (log pi(rejected|state)-log pi_ref(rejected|state))])`, using assistant-action tokens only and exactly the same observed starting state. Do not compare an already-edited positive state to an unchanged negative state. Prefer short verified continuations to whole-trajectory pairs with many unrelated differences. Cheapest rejection test: CPU evidence review finds contradictory state/outcome labels, or preferences reward irrelevant edits; do not train until all pilot pairs pass. Later acceptance requires beating verified SFT on held-out recovery without increasing harmful edits. Cached reference probabilities reduce reference residency, but policy log-probabilities still require a new compatible implementation and resource check.

For RLVR/GRPO, the cheapest falsification of near-term feasibility is a cost/reward-yield audit using retained outcomes and a declared rollout plan. A hypothetical 16-task × 4-rollout cohort, each consuming the existing 4.5-minute cap, allows **4.8 session hours / ~9.6 quota-hours of rollouts alone**, before startup, scoring and updates. This is an illustrative worst-case rollout allowance, not measured average cost or a proposal to run it. GRPO's all-equal-reward groups provide no outcome-based relative discrimination; D supplies no successful terminal examples to justify assuming a useful yield. Proceed only after a separately funded pilot demonstrates nonzero verified rewards, uncontaminated reward controls and a complete train/export/serve cycle; then require better independent resolution per quota-hour than verified SFT. No current GPU allocation is assumed for this work.

For future RL, a positive terminal reward must require a valid collected source patch and independent tests, with test tampering/invalid artifacts rejected. A nonempty-diff bonus alone is exploitable by whitespace or irrelevant edits. Process labels should diagnose progress; if shaping is tried, keep terminal resolution dominant and audit policies that maximize shaping while failing tasks. Exact teacher patch matching is useful for conversion-fidelity audit, but is inappropriate as the sole correctness reward because multiple fixes can be valid.

## 11. Ranked hypotheses and falsification tests

These hypotheses can coexist; rank reflects current evidence and information gain, not a posterior probability.

| Rank / hypothesis | Support | Contradictory evidence / missing data | Cheapest falsification, compute and pass criterion |
|---|---|---|---|
| 1. Unsuitable state/phase supervision contributes to missing edits | D switches from B's edits to reads/tests; later windows omit preceding state | Teacher runs were success-filtered; actual phase shares and state sufficiency unknown | Recover windows; dependency/phase/coefficient audit, 0 GPU. If every target has needed state and edit/recovery coverage is substantial, demote. Correction succeeds only with improved held-out transition behavior, not counts alone. |
| 2. Converted action/feedback semantics are unfaithful | Responses inferred; create success/default exit/insert mismatch; original patch reused for submit | No actual C replay failure demonstrated; quoting checks passed | Replay same original rows/base snapshots in isolated CPU environments, 0 GPU. All claimed successful edits must mutate intended source and match expected final tree; any mismatch invalidates that sample, not automatically the whole causal theory. |
| 3. Boundaries/masks/train–serve mismatch alter control flow | Text scans/offset masks, omitted analyzer schema, giant feedback-blind batches | Native template intended; masks structurally valid; most D steps are single-call | Exact tokenizer + compiled request fixtures, 0 GPU. Zero tool-response label leakage, correct ending labels and prefix parity. Passing all fixtures sharply narrows this hypothesis. |
| 4. Update strength damages a useful policy | 122 C updates versus 5 A; behavior differs while validation improves | Data differs too; no controlled strength or checkpoint comparison | First static data gates, then fixed-prefix base/C40/Cfinal comparison (~0.5–1.5 quota h scoped plan). Early edits/final no-edits supports association with dose; absence across arms weakens it. Does not prove overtraining uniquely. |
| 5. NF4-to-W4A16 transfer amplifies adaptation error | Measured next-token disagreement | A/B edits survive same path; text agreement is not action accuracy | Later identical-prefix, same-adapter cross-base comparison; CPU prompt check first. GPU diagnostic requires separate budget; demote if action results agree despite token drift. |
| 6. Budget/context boundary suppresses late opportunities | 389 rejections; 28 identical one-token overflows; large batches | 77 normal completions still empty | Static request arithmetic and batching fixtures, 0 GPU; later one-call→feedback gate. Removing boundary alone cannot be called a fix without source edits. |
| 7. Tool/collector defect loses source changes | Logical possibility with lost worktrees | No attempted source edit found; B/D same scaffold retains B patches; verified C routing | CPU official-collector fixture with source, fixture-only, untracked, staged and reverted cases. Must capture eligible source patch and explain exclusions. Low-priority causal investigation. |

## 12. GPU-efficient roadmap and authorization boundaries

### Ranked paths

| Path | Rank, evidence and expected impact | Effort/risk | Exit condition |
|---|---|---|---|
| **A — Minimal pipeline correction** | First: directly tests static fidelity/state/mask risks with zero GPU | Low–medium engineering; exact originals may be missing. Correct only demonstrated discrepancies, one at a time. | Reproducible validated data plus a behavior-gated tiny candidate. If audits pass, do not manufacture a bug; move to policy-dose/data-selection hypotheses. |
| **B — Dataset redesign** | Second: best structural response if faithful existing data still overweights unsupported reads/tests/finish or lacks recovery | Medium effort; replay environments and diversity dominate cost; harder than simply collecting more rows | Verified distinct issues, measured phase coverage, stateful episodes; better edit/recovery behavior on disjoint tasks at matched compute. |
| **C — Improved objective** | Third: useful only if trustworthy data and SFT still fail, or specific preference/retention evidence supports it | Low for dose ablation; medium/high for DPO; very high for online RL | Beat verified SFT at matched budget without higher empty-patch or regression rates. RL remains a later research program. |

### Pre-registered experiment sequence

| ID | Question and control | Planned cost | Pass/fail and stop rule | Authorization |
|---|---|---|---|---|
| E0 | Can exact C preprocessing be recovered? Compare arrays/manifests to preserved NPZ, not a newly sampled corpus. | 0 GPU; hours to days of artifact work depending on missing environments | Stop claiming exact reconstruction at first unrecoverable provenance gap. Regeneration is a separately labeled candidate. | Read-only local audit permitted now; remote recovery/rebuild not performed here. |
| E1 | Are converted transitions faithful? Original teacher actions/patch vs converted replay at same base. Stratify edit types and failures before expanding to all accepted data. | 0 GPU; initial ~1 CPU working day estimate, environment provisioning may dominate | Any false successful edit/submission quarantines example. Stop first pilot on systemic mismatch; otherwise validate all training positives. | Separate isolated replay approval; no host execution of teacher commands. |
| E2 | Do training targets have correct state/masks? Compare original rendered windows to state-dependency annotations and serving request fixtures. | 0 GPU; ~0.5–2 working days estimate after artifacts exist | 100% tested boundary fixtures pass; zero silent incomplete actions; all accepted target prerequisites present. | Future audit utility/fixture implementation scoped separately. |
| E3 | Does final C prefer non-edit actions at identical sufficient-state prefixes? Base vs final C, optionally predeclared C40 if budget permits. | **0.5–1.5 quota h** incl. startup; no retraining | Four fixed prefixes per arm: localized-source edit, failed replacement, failed test after edit, empty diff. Record proposal validity separately from execution. Stop on infrastructure failure; no automatic retry/arm expansion. | Explicit new GPU authorization and fresh quota needed. |
| E4 | Does one demonstrated correction restore editing? Tiny corrected SFT vs unchanged data at fixed updates/token exposure; base retention reference | **1.4–3.0 quota h per training arm**, plus **1.2–2.0** for bounded serving gate; controls may need additional training | ≤8k; peak <21 GiB; hard step cap; stop on NaN/OOM/budget; candidate must pass edit/recovery gate. No budget for a two-arm causal study is presumed. | Defer until sufficient quota and separate training approval. |
| E5 | Do restored edits resolve independent issues? Frozen holdout, paired unadapted v7/candidate under identical environment/budgets | Full 114-task pass measured ~8.99 quota h; paired run is not a 2.88 h experiment | Predeclare IDs/seeds/metrics; no post-hoc task deletion; resolution + empty patches + intervals/paired tests. | Later independent evaluation proposal, not permission to retry D or D0. |

**Recommended next experiment: E0 followed by the CPU E1/E2 audit package.** It separates fidelity defects from optimization hypotheses without spending the remaining GPU quota. If provenance cannot be recovered, build a new explicitly versioned verified pilot; do not present it as proving what was in C.

For E3, prepare a concrete script/config for separate review before launch. A proposed cap is **35 minutes total session wall time**, including environment restoration and server startup, with a 10-minute startup deadline, bounded completion tokens (e.g. 512 per fixed decision), per-request timeouts and a process watchdog. At observed 2× accounting this is ~1.17 quota h before platform overhead; request a 1.5 h ceiling and preserve at least 1.0 h reserve. If fresh quota cannot cover ceiling+reserve, do not start. Reduced completion limits must be identical across arms and make this a narrow decision probe, not the full D generation configuration.

The existing manual cancellation route was ineffective during D. A Python/process timeout bounds workload but does not guarantee Kaggle stops charging after process exit. Require validated startup-time accounting, watchdog termination and session shutdown before authorizing a job. Treat remote stop as best-effort unless its exact mechanism is verified. Never promise a hard quota cap from `--budget-h` alone; the training loop checks budget only between microsteps and validation/export also take time.

Cost model for future planning: `quota_h ≈ measured_accounting_factor × (startup_s + input_tokens / measured_input_tok_s + validation/export/eval_s) / 3600`. Use about 2× as an observed estimate, with headroom, not a contractual rate. At 244–296 tok/s, 7.04M processed input tokens already require about 6.6–8.0 session hours before other overhead. Longer state prefixes increase compute even when their loss mask is zero.

## 13. Quantitative success/failure gates

These are **proposed engineering acceptance thresholds**, not empirically optimized constants or statistical claims.

| Gate | Required measurement | Proposed threshold |
|---|---|---|
| Data provenance | Accepted samples linked to raw row, base commit, tokenizer/template and conversion revision | 100%; otherwise quarantine or label reconstructed candidate |
| Replay fidelity | Claimed edit success matches actual eligible source mutation; final collector output agrees with measured tree | 100% of accepted positive examples; zero unexplained synthetic-success labels |
| Representation | Paired observations, state prerequisites, complete targets, correct boundary masks | All fixtures pass; zero tool-response supervision; zero silent truncation |
| Split integrity | Repository/fork and issue/patch duplicate overlap | Zero known train/test overlap; report unknown pretraining contamination separately |
| First behavior gate | Four independent repair fixtures per arm, including empty-diff and failed-test recovery | Candidate edits eligible source and returns nonempty patch on 4/4; both recovery fixtures use feedback; appropriate verification after final edit on 4/4 |
| Control flow | Calls per model response; repeated failures; empty completion claims | One tool call before feedback on these sequential fixtures; no repeated empty submission; no false success claim; no context/budget error |
| Resolution | Known deterministic repair fixtures | All four required targeted tests pass, with source edits retained; these easy gates do not estimate generalization |
| Retention | Base vs candidate matched fixed prefixes/tasks | No loss of a previously successful edit/recovery fixture; record paired wins/losses rather than aggregate-only comparison |
| Later independent evaluation | Resolved tasks, empty patches, time/calls, repeat/context rates, same test environment | Retain historical ≥5 additional resolves on matched 114 as a practical screen, no rise in empty patches; also report paired exact McNemar and uncertainty. Five net wins alone are not statistical significance. |

Stop a proposed gate early if its mandatory threshold is already impossible; retain all attempted outcomes. Do not silently replace failures with easier tasks. If the base fails a supposedly trivial fixture too, investigate fixture validity rather than attributing candidate failure solely to training. Track both unconditional success and conditional transitions such as edit given sufficient localization, recovery given actual failure, and nonempty submission given source mutation.

For generalization, keep a final repository-disjoint test set untouched by checkpoint selection. The 114 public tasks are now heavily inspected diagnostic data, not a clean unseen test set. Use task-paired comparisons and repository-clustered uncertainty where enough repositories exist; the four public repository groups are too few for strong cross-repository claims. Repeat seeds only when the budget and pre-registration justify them. Do not imply a successful 4/4 smoke establishes reliability across private tasks.

## 14. Risks, missing artifacts and reproducibility

### Recovery manifest

| Required artifact | Current evidence / recovery route | What remains impossible without it |
|---|---|---|
| Exact raw 200 teacher rows, full trajectory IDs, dataset revision and selection order | Not located in diagnosis; search teammate preprocessing outputs/caches and original dataset export. NPZ task IDs can narrow candidate rows but do not uniquely identify trajectories. | Actual edit distribution, teacher-output fidelity and exact conversion reconstruction |
| Converted `trace_*.json` and `convert_report.json` | Request original conversion directory; otherwise rerun pinned converter on exact rows into a new directory after authorization | Distinguishing original conversion defects from changes in current source |
| Rendered JSONL with text/spans/window indices | Recover original output; otherwise deterministic render with pinned source/template/tokenizer | Per-window causal state, exact token/action mapping, omitted observation audit |
| Tokenizer vocabulary/config/special-token files and chat template | Original default was `/tmp/g4tok`; obtain exact source revision/files, not simply latest model tokenizer | Decoding IDs and validating role/end-of-call labels with confidence |
| Template trace, system prompt, agent/tool schemas and preprocessing commands | Recover exact inputs and their hashes; current v7 and source are candidate references, not proof of historical identity | Byte-identical train/serve prompt and deterministic window reproduction |
| Source snapshots/base commits, dependencies, teacher environment and collector implementation | Obtain pinned dataset environments and official wheel/source matching retained run; preserve licenses/digests | Environment-grounded replay and collector-filter fidelity |
| C input NPZ and summary | Present at `data/runs/train_public_200.npz` and copies; known SHA-256 below | Sufficient for structure/schedule checks, insufficient for semantic provenance alone |
| C checkpoints 40/80/120, final; training source/config | Present per artifact/C reports; training code match established by diagnosis | Available for future authorized comparison; no regeneration needed |
| Original compiled serving requests/parser/tokenizer | Need captured request fixtures or exact-version reconstruction | Final prompt/stop/reservation equivalence |

Compare regenerated **array contents, ordering, masks, starts, task IDs and split flags** to the old NPZ. ZIP-level bytes can differ because serialization metadata changes; preserve original hash and compare canonical array hashes. A full match supports preprocessing identity but still does not validate filesystem semantics. If row selection cannot be reconstructed exactly, clearly label the dataset “new candidate”; do not overwrite the original.

Preserved NPZ SHA-256: `b3f51bc449c31de2a3b163f98c5fdec2725b4427bde15625af7b4c75447d2488`.

Final C adapter SHA-256: `a116b6be160fffcc98cdd2a4ab6cfaffafcfc54ac6709a8a5d95d95ebab544c6`.

Primary source at research start: Git HEAD `18356102b618280c50f9e7e1587e02695cad1421`, with pre-existing uncommitted HQ/probe and diagnosis/artifact changes. A Git commit alone therefore does not identify every inspected file. Preserve per-file hashes for any subsequent reproducible build. Do not reset, clean, or overwrite existing work.

Other limitations: CPU replay may still require expensive environment downloads and setup; teacher test success may not transfer across Python/dependency versions; protecting test files changes which patches survive collection; source diff verification does not prove correctness; recovery oversampling can induce unnecessary failures; short-task selection can suppress hard-task competence; summaries can leak solutions; public-data repository splits cannot exclude unknown pretraining memorization. Cost forecasts for DPO/RL have no local throughput measurement. Dataset license metadata and provider terms need pinned-version review before adoption; this report is not legal clearance.

## 15. Dedicated Step D finding → correction → validation map

| Major diagnosis finding | Candidate corrective action | Validation, counterevidence and cost |
|---|---|---|
| No identified source edits; 114 empty patches | Verified locate→edit demonstrations and source-delta gate | CPU replay + 4/4 edit gate; teacher selection already reports success, so measure converted state. 0 GPU audit; future bounded inference. |
| 81 successful tool responses report empty patch; no recovery | Empty-diff→inspect→edit→verify continuation examples | Two state-matched empty/nonempty fixtures; no arbitrary-edit reward. Collector “ok” is not task success. 0 GPU data audit; gate within E3/E4. |
| 77 normal completions still empty | Make nonempty eligible source patch mandatory in behavior gate | Gate rejects activity-only candidates; fixing timeout alone cannot pass. Static gate design 0 GPU. |
| Eight enormous pre-feedback batches | Preserve call/observation boundaries; sequential tool examples | Token/parser fixtures then one-call→observation behavior test. Most D tasks are already single-call, so batching is not universal cause. |
| 389 budget-rejected calls; calls after submission | State-aware remaining-budget and failure-recovery examples | Bounded fixture must stop/replan after rejection, not repeat. More calls is not the control treatment. |
| All 28 context failures at 32,769 | Audit final request reservation and margin | Input + output reserve + margin ≤32,768; inspect actual serializer. CPU arithmetic first; no claim it restores editing. |
| Temporary fixture writes fooled edit heuristic | Classify filesystem deltas by tracked source/protected/test/temp path | Fixture-only mutation must score zero source edits; positive source edit must survive collection. CPU replay. |
| Old analyzers miss shell edits/misassociate observations | Use ordered IDs and state deltas in new audits | Synthetic multi-call pairing fixtures, explicit missing observation status. Do not rerun old analyzer to infer source edits. |
| B retains three patches under same scaffold | Preserve tool interface as first control; compare adapter policy | Base/C fixed-prefix test and narrow collector fixture; infrastructure failure is lower ranked. |
| C loss improves while behavior fails | Phase/coefficient loss accounting and behavioral checkpoint selection | Token/schema gates plus held-out editing and resolution; aggregate CE cannot approve release. |
| Correct C routing/hash and complete exports | Retain integrity checks but separate from behavior | Exact artifact identity before comparison; no redundant routing-root-cause investigation. |
| Missing preprocessing artifacts | Recover provenance chain before changing data | Canonical regenerated arrays match original or declare new candidate. 0 GPU. |
| D0 absent; historical/task-overlap confounds | Frozen matched future controls and independent repositories | No significance/generalization claim from B's 8 IDs or old aggregates. Later evaluation needs separate budget. |

## 16. Final recommendations for Shrey

Approve a **CPU-first provenance and data-fidelity investigation**, scoped to exact preprocessing recovery, isolated edit replay, state/mask audits and effective phase-weight accounting. Its result should determine whether Path A needs a correction or Path B needs a redesigned corpus. Do not repeat the completed D investigation, expand budgets, restore `edit_file`, or retrain from the plausible-defect list alone.

After those gates, choose **one** small causal comparison. If data defects are demonstrated, correct one defect and compare under fixed token/update budgets. If fidelity checks pass, compare base, one early C checkpoint and final C at predeclared edit/recovery prefixes to test policy degradation. Maintain native-tool and export compatibility; advanced objectives must earn their additional cost against a verified SFT baseline.

### Proposed HQ summary — research recommendations, not experimentally verified results

Atul's October 16 data/post-training research uses the completed Step D diagnosis as primary evidence. Recommended order: recover exact preprocessing artifacts; audit converted source edits/feedback, state preservation and token boundaries; measure action/token/effective-loss exposure; then use verified compact/recovery trajectories and a source-edit/nonempty-patch behavior gate before broad evaluation. Current code shows fidelity risks, but their actual C prevalence and causal role remain unverified. Keep rank-16 SFT initially; defer DPO/RL until editing and reward validity pass. Last measured quota remains 2.88 h; no new allocation or run is assumed. Research only; no pipeline fix, GPU run, model load, notebook push or submission performed.

### Message ready for Shrey — not sent

> Shrey, the research report is in `docs/atul_data_training_research_20261016.md`. D's missing source edits are established, but the upstream cause is still open. The next step should cost zero GPU hours: recover the exact preprocessing artifacts, verify converted edits and feedback, and audit window state, masks and effective phase weights. Then test one correction with a strict edit→test→nonempty-patch gate. I recommend verified SFT first and deferring DPO/RL. The report includes three paths, falsification tests, compute estimates and a mapping from every major D finding. No jobs or implementation changes were made.
