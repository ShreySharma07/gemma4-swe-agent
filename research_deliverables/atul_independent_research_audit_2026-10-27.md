# Independent Research Audit — Gemma 4 Developer Agent

**Prepared for Atul and Shrey · 27 October 2026**

**Repository audited:** `/home/atul-baliyan/Desktop/gemma4-swe-agent-main`  
**Local branch and evidence cutoff:** `main`, commit `313f50e9abdcd8703ac92e7eee66f8bf1054659b`. This is the latest commit available in this checkout; remote freshness was not established by fetching.  
**Audit scope:** research documents, implementations, locally available measurements, CPU recomputation, and primary literature. No model inference, training, repository test-suite replay, Kaggle operation, Git history change, or modification to existing evidence was performed. The requested new Markdown report is the sole authorized repository addition.

Evidence labels used throughout:

- **Verified:** directly checked against available records or implementation; this does not imply an independent rerun of the underlying experiment.
- **Reported:** present in an existing report but underlying records needed for full reproduction were unavailable.
- **Interpretation:** an explanation consistent with some observations, with alternatives still open.
- **Unsupported:** stronger than the design, measurement, or available evidence permits.
- **Proposed:** future analysis or experiment; not performed in this audit.

All repository paths below are relative to the repository root unless an absolute path is given. Paths into `data/runs/` identify local evidence; their presence here does not guarantee that every raw file is tracked or available to a public reviewer. The final paper needs a distributable evidence manifest rather than links that silently depend on this workstation.

## Part 1 — Executive verdict

**Verdict: promising but incomplete. The experiment portfolio supports a careful empirical paper; the strongest current causal interpretation is not yet secure.** The current outline is substantially better than the earlier architecture proposals because it preserves failed predictions. Its measurements often reproduce exactly. Its principal vulnerability is the inference from a test-failing hybrid patch to an intrinsically incorrect local repair.

The strongest defensible direction is **Direction B: repair correctness, framed as a narrow empirical diagnosis within the existing “Found, Not Fixed” project**. The contribution should be a validated distinction among incomplete repairs, incompatible counterfactual composition, incorrect local behavior, verification failures, and resource termination. It should not assume that these categories have already been separated.

Five findings matter most:

1. **The main outcome counts reproduce.** On the same 60 tasks, v7 resolves 23, v14 resolves 30, and v16 resolves 27. v7/v14 discordances are 8 versus 1, exact two-sided McNemar `p=0.0390625`; v14/v16 discordances are 4 versus 1, `p=0.375`. The first result concerns a configuration bundle; the second supplies no demonstrated resolution improvement from the small-output prompt.
2. **A4 challenges a simple “just add the missing edits” explanation.** Completion passes in 6/26 selected patch cases on 2/13 distinct tasks. The five pure-incompleteness successes are repeated attempts on one task, `rich_3052`. This is useful negative evidence about the particular intervention. It is not a mathematical upper bound on every compatible completion or on an in-run assistance method.
3. **A12's arithmetic is sound, but its semantic interpretation needs validation.** The saved primary result is 52/71 failing protocol-labelled clean, necessary-site swaps on 20 selected tasks; 19/20 have at least one failure. A synthetic construct check demonstrated that `build_swap()` can label a namespace-incompatible hybrid “clean.” This proves an instrumentation weakness, not the prevalence of that weakness in the 71 real swaps. “73% of local edits are wrong” currently overstates the result.
4. **Several explanatory claims exceed their measurements.** Command mentions and intended read ranges are counted as exposure; bare function names approximate edited-site coverage; a submit anywhere in a trajectory is treated as voluntary termination; budget detection overlooks actual denied calls. Four unresolved, nonempty v14 trajectories contain a BudgetExceeded response followed by submission. A failure to understand, plan, execute, or verify cannot be uniquely assigned from these proxies.
5. **The typed-retrieval negative is credible within its implementation, while broad novelty is limited.** E0b recomputes to typed hit@6 `0.1886` versus BM25 `0.4211` over 1,192 ordered pairs from 120 tasks/repositories. A task-cluster bootstrap difference excludes zero. That is strong evidence against this frozen graph/rule configuration on gold co-change labels, not against graph retrieval generally. Recent work already covers localization versus repair, change-impact planning, counterfactual diagnosis, and repair verification.

The paper should continue only with a tighter claim boundary. Preserve the historical negatives, downgrade the “oracle bound,” “voluntary stopping,” and universal graph claims, and validate the A12 intervention before making local-edit correctness the headline. No new GPU campaign is needed to make that decision.

## Part 2 — Complete research journey

The chronology below separates successive hypotheses rather than treating every old proposal as a live plan. `docs/HQ.md`, `docs/experiments.md`, and `docs/findings.md` are the principal historical records. Reused identifiers require dates and namespaces: September graph-analysis A1/A2 are different experiments from October's size and failure analyses.

| Period and research stage | Hypothesis or proposal | Evidence and research decision | Audit status and remaining qualification |
|---|---|---|---|
| September 27: provided repository graphs | Gold repair locations should be reachable through repository structure. | HQ records gold-node coverage on 118/129 tasks, reciprocal edges, mixed containment/call semantics, and weak entry grounding. Issue-to-gold reachability remained limited despite substantial graph coverage. | Historical measurements inspected through reports/code context. Node coverage is not evidence of agent repair competence; exact-name entry and graph semantics were practical obstacles. |
| September 28–29: name invariance and graph reasoning | Train agents to resist naming shortcuts, inspired by structural-invariance work; use evidence-conditioned graph reasoning. | `docs/HQ.md` and the research surveys identify distribution-shift, proxy-reward, novelty, and compute risks. Public code familiarity and evaluation validity became central concerns. | A research motivation, not a demonstrated SWE method. Do not claim that renaming alone would remove contamination or that graph-reasoning rewards transfer to code repair. |
| Early October: scaffold and interface repairs | Read loops and tool-format drift might explain poor task completion. | The development history documents changes to tool presentation and the v7 scaffold, including removal of problematic channels/edit interfaces. | Useful engineering observations. Several scaffold/environment changes co-occurred, preventing clean attribution to one interface decision. |
| October 7 onward: evaluation redesign | Gold-passing tasks and fresh tasks should support more trustworthy evaluation. | Old-54, expanded validation, and new-60 collections replaced earlier unstable subsets. The current new-60 has 50 FastAPI, 6 Rich, and 4 Requests tasks. | Better internal reliability, with strong selection and repository concentration. Public, newer tasks are not private or demonstrably uncontaminated tasks. |
| October 9: graph-diagnostic pilots | Graph structure should separate useful repair relationships sufficiently to justify a mechanism. | HQ/findings record a BMM separation estimate around 0.514 with an interval spanning weak discrimination, and a graph-repair pilot whose approximately 41% recall failed its gate. | Negative evidence against those tested mechanisms. It does not establish a general impossibility result for structural information. |
| October 15–18: post-training feasibility and failure | Better imitation data and lower loss should improve an acting policy. | Step C/D reports document a 0/114, all-empty-patch collapse after successful training/export. S1/S3 training succeeds technically, but the 30-prefix behavior gate favors base over both adapters. S3b later passes Gate v2 yet resolves only 9/60 versus v7's 23/60. | Strong warning about the surrogate objectives and gate's predictive validity. Exact causes of collapse were not isolated experimentally. “SFT fails” is much too broad. |
| October 19: v14 configuration | Thinking, prompt, submission, and budget changes might improve the base agent. | v14 reaches 30/60 versus v7 23/60; empty patches fall from 15 to 5. Paired arithmetic reproduces. | A nominally significant bundle comparison, not a thinking-only experiment. New-60 is reused, not a pristine final holdout. |
| Around October 20: EBST architecture | Evidence-conditioned stopping and structural control could deliver a novel architecture. | `research_deliverables/paper_outline.md` contains a proposed EBST narrative, theoretical claims, and prospective-looking numerical benefits without an implemented evaluated method supporting them. | Superseded. Values such as an 87.5% to below-10% change and 85% token savings must not migrate into the current paper as results. Entropy does not establish a calibrated false-commit guarantee. |
| October 22: literature correction | Audit the original novelty story before investing in it. | `literature_review.md`, `references.bib`, and the five surveys correct earlier bibliographic/novelty mistakes and identify close prior work. | A valuable correction. Historical survey warnings should not be misrepresented as proof that the current bibliography still contains every old error. |
| October 23: obligation frontier | After a first edit, typed dependency rules should retrieve remaining necessary changes. | E0a reaches 0.45, below its gate; E0b reaches 0.189 versus BM25 0.421. The proposed closed-loop E1–E4 sequence was not justified and was not run. | A meaningful rejected method. Predictions and results for E0a/E0b share commits, so independently timestamped preregistration is not established. |
| October 23–25: “Found, Not Finished” | Failures after exposure mainly omit required changes. | A3 finds 19 new-60 tasks with at least two gold-relative necessary groups. A1/A13 show size confounding. A4 yields little completion rescue and only one pure-incompleteness task. | The preferred simple incompleteness explanation lost support under this protocol. The small selected sample and composition validity still matter. |
| October 24–25: “Found, Not Fixed” | Wrong edits at necessary sites are the dominant remaining obstacle. | A12 yields 52/71 failing clean/necessary hybrids; A5 describes co-change relationships; A7/A2 analyze stopping; A8 examines lexical advantage; A8b changes rules post hoc. | A reasonable next hypothesis, currently presented too conclusively. Compatibility, trace measurement, clustered inference, and verification failure categories need repair. |
| October 23 results, available at audit cutoff: v16 | Keeping tool output small should reduce context pressure and improve repair. | All 60 results are recoverable: 27 resolved; context errors fall from 8 to 5; resolution does not improve. | Preserve this negative result and incorporate its task records. The outline's assertion that A9 cannot run because v16 per-task results are absent is stale in this checkout. |

### Which experiments were preregistered?

The current outline's Git provenance table is useful and should remain. A1's prediction commit `e7f559f` precedes its reported result commit `9d99618`; A3/A4's `62e4068` precedes `7beef0e`; A12/A13 use `92dabb6`; A5/A7/A2/A8/A8b use `7ad401b`. A8b is explicitly a registered **post-hoc** extension, not an independent confirmation of the original rules. E0a prediction and result share `e8e9db3`; E0b shares `672bc30`. Git establishes publication order in some cases, not unseen researcher knowledge, complete analysis-plan compliance, or absence of previous tuning.

### Why the rejected directions should remain rejected

The original frontier gate failed before an expensive downstream run; later v2 rules were tested on the same seven-task pilot and still missed the 0.80 threshold. EBST lacked the implemented mechanism and empirical support required for its novelty score and guarantees. The “mostly unfinished” hypothesis received little rescue from A4. Post-training showed a clear mismatch between imitation loss/gate scores and task outcomes. These are different negative results with different scopes. Combining them into “structure and training cannot help” would discard what the experiments actually taught.

Sources: the primary historical documents above; `research_deliverables/thesis_found_not_finished.md`, `gaps_2026-10-22.md`, `novelty_stress_test.md`, `paper_outline_find_vs_finish.md`; `results/runs/`; and the named run reports in Part 3.

## Part 3 — Verified evidence table

### Audit method, availability, and calculation rules

CPU recomputation used Python standard-library parsing, exact combinatorial tests, descriptive aggregation, and seeded task-cluster resampling. No saved result was rewritten. The original training/evaluation processes and repository test suites were not independently rerun. Synthetic construct checks executed small isolated functions in memory; they did not execute recorded agent commands or alter source repositories.

For paired runs, tasks were joined by exact `instance_id`, with unique IDs and identical 60-task sets checked. For discordances `b,c`, the exact two-sided McNemar value was calculated as `min(1, 2 * sum(C(b+c,k), k=0..min(b,c)) / 2**(b+c))`. Fisher tests sum fixed-margin hypergeometric probabilities no larger than the observed table probability. Reported Wilson intervals use 95% coverage. A12 task bootstrap used seed 0 and 10,000 resamples; retrieval sensitivity used seed 0 and 4,000 task resamples. These intervals describe selected samples, not random sampling of all repository issues.

Important source bundles:

| Bundle | Exact primary paths | Availability and verification boundary |
|---|---|---|
| v7 new-60 | `data/runs/eval_v7_new60/x/eval_v7_noedit/rep0/merged/task_results.jsonl`; sibling `analysis/per_task.jsonl` | Task outcomes and derived rows available; full original traces absent locally. Budget-event prevalence is **unknown**, not zero. |
| v14 new-60 | `data/runs/atul_eval_v14_new60/x/eval_v14_thinking/rep0/merged/task_results.jsonl`; sibling `merged/traces/`, `analysis/per_task.jsonl`; `docs/atul_v14_new60_report_20261019.md` | 60 outcomes and 60 traces available; arithmetic and actual tool-response budget errors checked. |
| v16 new-60 | `data/runs/atul_eval_v16_new60/recovered/eval_v16_small_output/rep0/merged/task_results.jsonl`; sibling traces; `data/runs/atul_eval_v16_new60/eval_v16_small_output/rep0/analysis/comparison.json` and `report.md` | 60 outcomes and traces available. The comparison artifact reports version 5, source matching, no adapters, archive hash, and all IDs matching v14. This audit recalculated outcomes, not a new remote provenance check. |
| A1/A13/A7/A2 | `results/runs/a1/a1_report.json`; `results/runs/a13/a1_a13_a7_a2_report.json`; `kaggle/a1_confound/a1_body.py` | Aggregate reports and implementation available; the feature/design rows `a1_rows.json` needed to independently refit models were not found. Necessity-stratum counts can be recomputed using other available task records. |
| A3/A4 | `results/runs/a3a4/{cases.json,groups.json,necessity_new60.json,necessity_report.json,results.jsonl}`; `kaggle/eval_l4/evalkit/necessity.py` | Case-level and verifier exit records available. Full original hybrid patches and test-output tails are not retained in this committed bundle. |
| A12/A5 | `results/runs/a12/{swaps.json,siteswap_report.json,results.jsonl,a5_pairs.json}`; `necessity.py::build_swap`, `main_siteswap`, `a5_typology` | Swap classes/outcomes and pair records available. `_patch` is deliberately omitted when saving swaps; execution output tails are not saved by the wrapper. |
| E0a/E0b/A8b | `results/runs/e0a/e0a_rows.json`; `results/runs/e0b/e0b_pairs.json`; `results/runs/a8b/e0a_v2_rows.json`; `analysis/{e0a_obligation_recall.py,e0b_external_recall.py}`; `tools/oblig.py` | Retrieval metrics independently aggregated. Repository indexing/retrieval over original source snapshots was not rerun. Current `oblig.py` SHA-256 is `54f37498699867692755dace156ecfb1d6f063bb12aeaf8dba6bee86683c31a2`; distinguish this v2 source from E0's original rules. |
| A8 | `results/runs/a8/a8_report.json`; `kaggle/a8_bm25/a8_body.py` | Report and model implementation available; `a8_rows.json` feature rows absent. Coefficients and their significance were not independently refit. |
| Training/gates | `data/runs/atul_s1_own/train_report.json`; `data/runs/atul_s3_compact/train_report.json`; `data/runs/atul_gate/gate_report.json`; `data/runs/s3b_train/train_report.json`; `data/runs/gate_v2/gate_report.json` | Training summaries and gate rows available. Gate rows measure classified next-call decisions, not execution or repaired-task outcomes. |
| S3b evaluation | `data/runs/eval_s3b_new60/x/eval_v12_s3b/rep0/analysis/per_task.jsonl` | All 60 IDs match v7; paired counts independently verified. |
| Old-54 repeats / Step D | `data/runs/rs_v7_valid/x/summary.json`; five `rep*/analysis/summary.json`; `docs/step_d_d0_20261016_report.md`; `docs/step_d_failure_analysis_20261016.md` | Repeat aggregate counts available; full old-54 task records/traces and original Step D output directory are absent here. Some historical claims are therefore report-level, not independently reconstructed. |

### Main experimental results

| Experiment / claim | Independently checked evidence | Statistical reading | Limitations and permitted conclusion |
|---|---|---|---|
| v7 → v14 | 23/60 → 30/60; 22 both, 8 v14-only, 1 v7-only, 29 neither. | Net +7/60 = +11.67 percentage points; exact `p=0.0390625`. | One stochastic draw per arm; thinking, prompt, budgets, and submission changes bundled. Nominal significance within a large exploratory portfolio. |
| v14 → v16 | 30/60 → 27/60; 26 both, 1 v16-only, 4 v14-only, 29 neither. | Net −3/60 = −5 points; exact `p=0.375`. | No demonstrated resolve benefit or statistically established harm. “No significant gain” does not establish equivalence. |
| Repository distribution | All three arms: FastAPI 50, Rich 6, Requests 4. | 83.3% of tasks from one repository. | Strong within-benchmark pairing; weak inference to arbitrary repositories or private tasks. |
| v14 single-/multi-file | 27/43 versus 3/17. | Fisher `p=0.0034212`; unadjusted single-versus-multi odds ratio 7.875, approximate Wald 95% interval 1.96–31.68. | Strong raw association, not an effect of file count independent of difficulty/size. |
| A1 size strata, reported design rows | v7 small/mid/large: 12/20, 10/19, 0/18; v14: 15/20, 12/19, 2/18. Fifteen of 17 multi-file tasks fall in the large stratum. | Reported pooled log-size OR 0.468, `p=0.0463`; v14 OR 0.367, `p=0.0250`. | Feature rows absent; coefficients not refit. Gold change size predicts difficulty but is not randomized. |
| A1 adjusted multi-file | Reported pooled OR 0.321, CI 0.0406–2.541. | Clustered Wald `p=0.2819`; ordinary LR `p=0.2081`. | The outline combines the CI with LR `p≈0.21` without distinguishing the tests. Wide uncertainty does not prove a null effect. |
| A3 necessity | 22 multi-group tasks, gold 22/22, empty 0/22; 19/60 have ≥2 necessary groups; labels present for 56 tasks. | A real gold-relative multi-site phenomenon exists. | Multi-group cases receive removal tests; single-group labels are assigned/inferred. Four tasks lack labels. “Necessary” means necessary in this gold-patch decomposition under this suite. |
| A13 necessity strata | On the 57-row analysis subset: v7 4/19 versus 18/38; v14 6/19 versus 23/38; v16 6/19 versus 20/38. | Published pooled Fisher 10/38 versus 41/76 gives `p=0.00560`, but treats repeated task outcomes as independent. | Counts reproduce; the independent Fisher inference is inappropriate for pooled paired arms. Adjusted necessity OR 1.68, CI 0.446–6.333, clustered Wald `p=0.4433`, remains uncertain. |
| A4 completion | 26 cases, 13 tasks; agent alone 0/26, completion 6/26 on 2 tasks, missed-only 3/26 on 2 tasks, pure incompleteness 5/26 on 1 task. | Completion-task Wilson CI 4.33–42.23%; pure-task CI 1.37–33.31%. | A selected intervention sample; repeated attempts are correlated. Do not headline the narrower case-level interval as task-level certainty. |
| A12 swaps | 128 swaps, 61 distinct `(run,task)` patch cases, 29 distinct tasks; 85 clean, 20 coupled, 17 conflict, 6 identical. | Gold rerun passes 29/29; selected primary 52/71 failures; task-bootstrap CI 44.94–100%. | “Clean” is an algorithmic class with demonstrated false-clean potential. Sixty-one is not the number of distinct tasks. |
| A12 task summary | 19/20 primary tasks have ≥1 failing swap; task-macro failure fraction 0.925. | Any-failure Wilson interval 76.39–99.11%; exploratory macro bootstrap 0.80–1.00. | Any-failure probability increases with number of swaps. Selected tasks and swap counts vary; this is not a population local-edit error rate. |
| E0a original | 28 ordered necessary-group pairs, 7 tasks: typed 0.4500, untyped 0.5414, BM25 0.3571 hit@6. | Task-macro values: typed 0.4429, untyped 0.4355, BM25 0.3214. | Typed/untyped ordering reverses under task-macro aggregation. Pilot is tiny and had rule-development exposure. Gate failure remains clear. |
| E0b external | 1,192 ordered gold co-change pairs, 120 tasks/repositories: typed 0.1886, untyped 0.2438, BM25 0.4211. | Task-cluster bootstrap typed−BM25 CI −0.2855 to −0.1837; macro values 0.2418, 0.2872, 0.5306. | Strong negative for this index/rule system. Labels are co-change, not test necessity; seeds are oracle gold edits, not naturally occurring agent edits. |
| A8b v2, post hoc | Same 28 pilot pairs: typed 0.6286, untyped 0.6257, BM25 0.3571; macro typed 0.5024. | Improvement on a reused pilot; both typed/untyped round to 0.63, but are not exactly equal. | Gate still fails. This is not external validation or proof that typing helps. |
| A5 relations | 596 ordered pairs on 19 tasks; one task contributes 306 pairs. Macro shared edited-token rate 0.9495; micro 0.7785. | Macro same-file 0.613, direct-edge 0.345, two-hop 0.512; micro same-file 0.3322, direct-edge 0.1560. | Aggregation radically changes interpretation. Shared-token extraction includes keywords and generic names; it is not a semantic dependency annotation. |
| A8 explanation | Report: no-shared-edit stratum 532 pairs, typed 0.149 versus BM25 0.308; gap 0.159. | The predicted gap below 0.10 fails. BM25 identifier coefficient reported 21.901. | Raw design rows absent; coefficients lack task-cluster inference, use different predictor scales, and do not establish why BM25 wins. |
| S1/S3 training | S1 33 steps, validation 0.2395→0.1362; S3 35 steps, 0.6089→0.1807. | Successful optimization/export in those reports. | Different validation data/coverage; loss is neither comparable across datasets nor a task-resolution metric. |
| S1/S3 gate | Base 74/120, S1 63/120, S3 66/120 over 30 prefixes. | Four samples per prefix across 23 source tasks are dependent. | Both adapters miss the comparative gate; classification is a proxy, with no executed repair. |
| S3b full tasks | 9/60 versus v7 23/60; 9 both, 0 S3b-only, 14 v7-only, 37 neither. | Exact McNemar `p=0.00012207`. | Strong within-set regression of this adapter. The result does not identify which training ingredient caused it. |
| Old-54 sampling | Aggregate repeats: 16,16,16,16,18 solves; 21/54 solved at least once, 9/54 every time. | Reported 0/7 multi-necessary tasks solved in five attempts. | Original full task/necessity records unavailable here; five attempts on seven tasks do not create 35 independent tasks. |
| Step D | Historical reports state 0/114 resolved, 114 empty patches. | Large documented failure under its setup. | Raw output absent locally; do not claim an independent task-level recount or a general negative theorem about fine-tuning. |

### Training and gate interpretation

The gate rows confirm file-in-view edit classifications of 24/32 for base, 19/32 for S1, and **21/32 for S3**. S3 submits after a failed-test prefix in 28/32 samples, passes the recovery criterion in 0/32, submits after passed-test prefixes in 24/24, and edits early in 10/32 versus zero early edits for base/S1. This supports an unwanted next-action pattern, not a measured count of bad executed patches. The classifier has narrow syntax recognition for edits and tests, and the gate uses short, thinking-off generations unlike v14's configuration. Four samples on one prefix are not four independent software issues. S3b's later task regression despite a Gate v2 pass is evidence that passing this surrogate was insufficient.

S3 validation covers 4/7 held-out windows, S1 20/21; their different loss values cannot rank agent quality. The E1 source-edit replay recorded in `docs/atul_step1a_1b_step2_gate_report_20261018.md` verifies 147/147 episodes and 182/182 source mutations, with matching source blobs. It does not independently establish that teacher tests or reasoning were semantically correct: that replay did not execute teacher code or repository tests. The historical training failure explanations remain hypotheses involving windowing, missing reasoning, style, and dose, rather than isolated causal findings.

### Analysis-status boundary

A2/A7 share the failure-analysis implementation and are not independent replications. A6 was dropped; its exposure question is reopened by this audit's measurement findings. A9 was not run as the planned third-arm analysis, but existing v16 records now permit CPU reanalysis. A10's in-run lexical hint was not tested. No separate active A11 experiment/result was identified in the audited ledger; no result is inferred from the numbering gap. The abandoned frontier E1–E4 sequence is distinct from the October 18 training-data **E1 replay**, which did occur. The original seven-task obligation pilot is implemented in `analysis/obligation_pilot.py`; its pilot reuse should accompany the E0a description.

### A4: what the intervention actually establishes

The six successful completions occur on `rich_3052` and `rich_3782`. Five are the five old-54 repeats of `rich_3052`. Missed-only patches pass on `rich_3782` and `rich_4070`; for two `rich_4070` cases, adding the agent's existing edits turns a passing missed-only patch into a failing completion. That is direct evidence of antagonism in those compositions, but it does not isolate a universal cause of agent failure.

There are **16**, not 15, old-54 cases in `cases.json`; ten are new-60 cases, comprising three v14 and seven v7 cases. The outline's repeated-case count needs correction. Composition uses 13 straight `git-apply+git-apply` cases, 8 fuzzy-patch cases, and 5 `no-module` fallback cases. Completion exit codes are six 0s, fourteen 1s, and six 2s. Exit 2 is compatible with collection/usage failures and cannot be called an assertion-level semantic failure without logs. The missing tails prevent resolving that ambiguity in this audit.

The term “upper bound” is too strong even for a frozen agent patch: the implementation tries a specified combination of gold hunks, module hunks, and fallback application. Patch composition is non-monotonic. Other compatible completions or repairs to interfaces may behave differently. Use **“observed gold-completion rescue rate under the specified protocol.”** The experiment is still a valuable falsification of a simple and concrete completion strategy.

### A12: arithmetic replication versus construct validity

The primary 52 failures comprise 49 exit-1 results, two exit-2 results on `fastapi_14978` in v14/v7, and one exit-124 result on `rich_4006` old-54 rep0. Excluding the exit-2/124 cases leaves 49/68 = 72.06%; this sensitivity shows that those three statuses alone do not erase the high hybrid failure rate. It does **not** show that the remaining failures are intrinsically incorrect agent functions. Exit 1 still needs diagnostic context.

The single task `rich_3052` supplies 15 primary swaps, all passing, explaining part of the difference between the pooled 52/71 rate and the 0.925 task-macro rate. There are only six v14 tasks in the primary subset, contributing seven failing swaps. Claims about all 30 v14 failures cannot be inferred from that subset.

`necessity.py::build_swap()` computes `gold_def` as a union of bare defined names over files. A verified synthetic example has an agent function in `a.py` calling an agent-created helper; the gold tree defines a helper with the same name only in `b.py`, with no corresponding import into `a.py`. The function is labelled **clean**, yet its hybrid raises `NameError`. Cross-file name equality is not namespace compatibility. Existing helper behavior, attributes, imports, signatures, decorators, and nested definitions pose additional plausible coupling risks that have not all been shown to occur in this dataset.

The 17 saved conflicts are missing-function cases, not evidence that all 17 were compile failures. Whitespace normalization for the identical class also removes whitespace inside string literals. AST function spans omit leading decorators. These are reasons to validate the instrument; they are not estimates of the real misclassification rate. Passing the 29 gold controls establishes useful environment viability, not hybrid compatibility.

### Trace, budget, and failure-taxonomy audit

The v14/v16 comparison artifact records the following common-definition measurements. Outcome counts, context-error counts, and actual BudgetExceeded responses were independently checked; the detailed output-character totals are source-artifact measurements rather than a second full serialization reimplementation.

| Metric | v14 | v16 | Reading |
|---|---:|---:|---|
| Context-window task errors | 8 | 5 | Fewer observed errors; three fewer does not demonstrate their causal contribution to resolve. |
| Actual BudgetExceeded events / affected tasks | 10 / 8 | 11 / 6 | Contradicts a literal interpretation of the original analyzer's 0% budget-exhaustion metric. |
| `cat` command proposals | 99 | 59 | Approximately 40% reduction in proposals; proposals need not equal successful whole-file reads. |
| Mean raw tool-response characters/task | 24,104.75 | 23,345.97 | Approximately 3.15% reduction; distinct from stdout/stderr payload only. |
| Mean stdout/stderr characters/task | 20,259.82 | 18,958.15 | Approximately 6.42% reduction under that narrower definition. |
| Original empty-patch percentage | 8.3% | 5.0% | Five versus three empty final patches. |
| Original repeated-identical-call share | 6.5% | 7.7% | A behavior proxy, not an independent measure of reasoning quality. |
| Original explicit-submit percentage | 88.3% | 90.0% | At least one recognized submit, not necessarily a voluntary final stopping reason. |
| Original median tool calls | 24 | 25 | More calls did not yield more resolved tasks in this comparison. |
| Failed final patches: empty / scratch-only / source+scratch / source unresolved | 5 / 5 / 5 / 15 | 3 / 1 / 7 / 22 | Content taxonomy; “source unresolved” can be wrong or incomplete and may coexist with verifier/environment issues. |

Original exposure counts reproduce: v14 gold-file 59/60 and gold-line 55/60; v16 gold-file 59/60 and gold-line 52/60. However, `analysis/trace_analyzer.py::analyze` counts paths mentioned in arguments or raw observations and infers line coverage from intended command ranges. A synthetic one-line response to `cat pkg/a.py` is labelled as exposure to a gold change at line 1000. This demonstrates potential false exposure; it is not a corrected estimate for these 60 tasks. All v14 first-file-hit events inspected had nonempty stdout, so it would also be incorrect to assert that the measured 59 hits were all failed reads. v16 includes first-hit events with empty stdout/errors.

`analysis/find_vs_finish.py` uses unqualified function names and hunk headers as coverage proxies. The v14 15-partial/9-elsewhere/5-empty/1-full-but-wrong decomposition is a heuristic partition, not an independently adjudicated causal taxonomy. `tested_after_last_edit` misses mutations through Python `run_command` and can misread successful command exit as passing relevant tests.

A7's reported 57-task subset has v7 21 unresolved nonempty cases, 20 with any explicit submit, versus v14 24/17. Across all 60, the counts are v7 22/21, v14 25/18, v16 30/27. Three executable FastAPI `docs_src` tasks are excluded by the analysis filter, not missing from the evaluation. Actual v14 denials precede submission in unresolved nonempty `fastapi_11355`, `fastapi_14246`, `fastapi_14266`, and `rich_3782`. The implemented `budget_exhausted` checks the final error string for “budget”; it does not inspect all tool-response denials. **“Submitted voluntarily, not starved” is unsupported.** v7's missing raw traces prevent the corresponding denial audit.

The source v16 report lists no recorded chunk watchdog/timeout, malformed recorded arguments, or unmatched responses. Its chunk durations are 37.6, 33.1, 35.0, 30.3, 40.1, 28.7, 38.9, and 15.8 minutes. Shared `wrapt` warnings alone are not a causal explanation: v14 has warnings on all 60 tasks and four collection-error tasks; v16 has warnings on 58 and five collection-error tasks. Preserve the actual diagnostic distinction.

### Historical quota and cancellation evidence

This is an audit of saved accounting, not a live account check. `data/runs/atul_eval_v15_new60/summary.json` records version 4 cancelled with **no recoverable task records**: completed/resolved/failed counts are unknown, 60 task-result artifacts missing, and no valid partial comparison. This must not become 0/60. Its stable post-cancellation account-used delta is 0.0 hours, after earlier rising telemetry reverted; that is not proof of zero physical GPU runtime.

The v16 comparison artifact records used quota 16.962316→25.949728 hours, delta **8.987412 account hours**, versus a maximum platform-log elapsed proxy of **4.493467 hours**. Available quota at that historical observation was 4.050272 hours, not today's quota. These measurements are approximately 2:1 but are not an attributable invoice. Actual monitor gaps reached about 65.3 minutes; a configured 15-minute heartbeat is not continuous observation. No quota ledger or monitor was changed during this audit.

### Other independent sensitivities

- A13's unadjusted **task-level** exploratory permutation, using each task's mean v7/v14 outcome, gives other-minus-multi-necessary difference 0.2763, Monte Carlo two-sided `p≈0.0477` with 10,000 permutations. It does not adjust for size/repository or repair selection bias; it illustrates why the pooled-independent Fisher result is not the right primary evidence.
- If v7/v14 and v14/v16 were treated as one two-comparison family, Holm adjustment would change 0.0390625 to 0.078125. That family is illustrative, not an invented retrospective official analysis plan. The paper needs its actual family and exploratory status declared.
- A5's saved shared-token lists include 38 pairs whose retained tokens consist only of Python keywords/`self`/`cls`. Removing those yields a conservative task-macro sensitivity of 0.8940, versus 0.9495 originally. Lists are capped at eight tokens, so hidden valid tokens could make the corrected rate higher; this is not a definitive re-estimate. Token overlap remains common but is not a manually verified semantic relationship.

## Part 4 — Main scientific weaknesses

Severity reflects impact on the **current paper's claims**, not whether the project was worthwhile.

| Severity | Issue and exact evidence | Why a reviewer should care | Recommended correction |
|---|---|---|---|
| **CRITICAL** | A12 clean/necessary failure is equated with intrinsically wrong local edits. `necessity.py::build_swap`, `results/runs/a12/swaps.json`; namespace-union synthetic false-clean demonstrated. | The strongest proposed new contribution may measure incompatible hybrid composition as well as repair error. | Recover exact hybrids and diagnostics; validate compatibility with namespace/interface checks and counterfactual controls. Until then state only protocol-labelled hybrid failure. |
| **CRITICAL** | A4 is called an oracle upper bound, with “content is the binding failure” inferred from six completion successes. `paper_outline_find_vs_finish.md` §3.3/C6; composition code in `necessity.py`. | A selected, non-monotonic intervention is not a bound over all completions; environmental/application failures are not isolated. | Rename the estimand; stratify application/exit modes; test gold-to-gold and round-trip controls; reserve causal attribution for adjudicated cases. |
| **HIGH** | Exposure is described as returned code, while `trace_analyzer.py` accepts argument mentions and intended ranges. `find_vs_finish.py` also approximates groups by leaf names. | “Found” can be overcounted, and exposure cannot establish localization or understanding. | Reconstruct returned source spans with file/commit/line provenance; use qualified AST sites; label mentions, successful reads, and decision-relevant exposure separately. |
| **HIGH** | A7 C14 says zero budget denials and voluntary stopping; four failed nonempty v14 traces show denial then submit. `a1_body.py`, v14 traces. | A direct factual/measurement error weakens the proposed explanation of failure. | Separate final task error, any denial, last paid tool attempt, later free submit, and context termination. Do not infer confidence from a tool name. |
| **HIGH** | Pooled A13 Fisher counts repeat tasks; A8 uses unclustered pair-level logistic inference. `a1_a13_a7_a2_report.json`; `a8_body.py`. | Apparent sample size and precision are overstated. | Task-level paired/clustered inference; resample whole tasks, preserve matched dependencies; report pair and task estimands separately. |
| **HIGH** | Lack of significant adjusted multi-file/necessity effect is presented as disappearance; A1 and A13 intervals are very wide. | Low power and confounding cannot demonstrate equivalence or that size is the sole mechanism. | State that size and multiplicity cannot be separated reliably here; show effect intervals and a predeclared practically meaningful range. |
| **HIGH** | Matching retains imbalanced sizes and reused controls: eight file-count sets have median 60 versus 32 lines. `a1_body.py::matched` logic and A1 report. | “Matched” does not guarantee exchangeability; independent set shuffles can violate dependence with reused controls. | Show balance/calipers/overlap, cluster reused units, and drop unsupported matching claims if adequate controls do not exist. |
| **HIGH** | Selection combines gold-passing suites, unresolved nonempty patches, edit coverage, buildability, and non-identical functions. `necessity.py::main_siteswap`. | A12 and A4 do not represent all failures or all tasks; exclusions can change the observed error distribution. | Publish an inclusion flow with all exclusions and exact task IDs; report arm-specific denominators and missingness. |
| **HIGH** | Counterfactual hybrid patches and output tails are discarded; feature rows needed for A1/A8 refits are absent. | Aggregate reproducibility is not intervention or model reproducibility. | Recover immutable input/output bundles with hashes, exact tests/environments, and structured failure diagnostics. If unavailable, limit claims explicitly. |
| **HIGH** | Gold patch size is treated as explanation and v14 gain as “thinking helped commit.” Current outline §§3.2/3.5. | Size is correlated with several latent difficulties; v14 changes are bundled and behavioral medians do not identify a causal mediator. | Use association language; identify controlled versus bundled interventions; do not equate fewer empty patches with a proven reasoning mechanism. |
| **HIGH** | Generalization from one quantized model, three repositories, and reused evaluation sets. HQ and task records. | Model-specific limitations, repository familiarity, and repeated hypothesis development may dominate results. | Scope every headline to the tested configuration/distribution; disclose public-data and tuning exposure; avoid private-repo claims. |
| **HIGH** | Novelty claims overlap recent localization/repair and verification studies. Current outline versus Part 5 primary sources. | A familiar conclusion plus a small single-model study may be incremental without validated new diagnostics. | Make the controlled diagnostic artifact the contribution; directly compare closest counterfactual/verification work. |
| **MEDIUM** | Necessity labels mix actual leave-one-group-out tests with inferred single-group labels; modules/nested groups complicate “site.” A3 and `necessity.py`. | Necessary in a chosen decomposition is not globally necessary across alternative correct repairs. | Publish label provenance per group; define groups and module handling; avoid calling gold edits the unique correct repair. |
| **MEDIUM** | A5 compares old pair-micro and new task-macro rates; generic token extraction inflates “identifier” semantics. `a5_typology`. | Claimed typology replication/failure can partly be an aggregation artifact. | Report both aggregations on each dataset; AST/tokenize with keyword exclusions; blinded second coding with disagreement examples. |
| **MEDIUM** | A8 causal explanation uses coefficient magnitude on incomparable scales, binary thresholding of fractional hits, and post-hoc gold-edit features. | Predictive associations do not establish a lexical mechanism; some features are unavailable to a deployed agent. | Standardize effects, use task-cluster intervals and held-out prediction, preserve fractional outcomes or justify thresholding, separate oracle explanatory features. |
| **MEDIUM** | Typed and untyped frontier arms change weights, allowed edge types, and locality behavior simultaneously. `tools/oblig.py`; E0 scripts. | “Typing hurts” attributes a bundle difference to one ingredient. | Describe the exact bundles; use a common candidate graph and one-factor comparisons for future CPU ablations. |
| **MEDIUM** | E0b selects one candidate per repository before eligibility; uses public gold anchors and restricted existing-function tasks. `e0b_external_recall.py`. | A broad-repository sample is still conditional; gold first edits differ from failed agent first edits. | Document selection flow, pin source revision/hashes, report excluded classes, and avoid an end-to-end agent-benefit claim. |
| **MEDIUM** | Many tests, thresholds, and post-hoc interpretations; E0 prediction/result commits coincide. | Nominal p-values and “preregistered” language can imply more confirmation than exists. | Separate confirmatory from exploratory analyses; disclose all tested gates; identify timestamp evidence and actual multiplicity family. |
| **MEDIUM** | Gate passes conflate call classifications with good decisions; S3b passes Gate v2 then regresses on tasks. `data/runs/atul_gate/evalkit/gate.py`; gate reports. | A surrogate gate is not a validated measure of behavior or deployment value. | Retain the negative transfer result; audit prefix/task dependence, syntax classifier coverage, and real feedback validity. |
| **LOW** | Outline says v16 task outcomes are unavailable, gives 15 old repeat cases, and compresses v2 typed/untyped to an exact tie. | Small inconsistencies erode trust and obscure reusable evidence. | Correct to available 60-task v16 data, 16 repeats, and unrounded values during an authorized later revision. |
| **LOW** | Historical draft/survey claims and current plans are interleaved; bibliography lacks some closest active papers. | Reviewers may mistake abandoned mechanisms or prospective numbers for contributions. | Mark superseded documents explicitly in a future authorized edit and provide a short authoritative claims/evidence index. |

### Disposition of the current claims ledger

- **C1/C1b:** retain only as original-proxy exposure/coverage descriptions until returned-span validation. Actual localization and understanding remain unmeasured.
- **C2:** retain a scoped association with gold change size. **C3:** causal multiplicity is unestablished, not disproved.
- **C4:** retain gold-relative necessary-group prevalence with label-provenance caveats.
- **C5:** the simple completion explanation failed its specified empirical test; do not claim every incompleteness mechanism was rejected.
- **C6:** retain measured A4 rescue counts; remove the universal upper-bound wording.
- **C7:** downgrade from Supported to **conditional interpretation pending compatibility and diagnostic validation**.
- **C8/C8b/C9:** preserve bundle outcomes and old-repeat limits; avoid a thinking-only or sampling impossibility mechanism.
- **C10/C11/C11b:** retain exact graph/token/recall observations; qualify aggregation and correlational mechanism. “Typing hurts” means the tested typed bundle performed worse, not an isolated edge-type effect.
- **C12:** preserve rejection of the original typed-rule prediction. **C13:** private-repository transfer remains unshown.
- **C14:** withdraw the voluntary/no-budget conclusion and replace it with actual event-level termination evidence.

No change to those existing documents was made during this audit.

## Part 5 — Novelty analysis

### Closest primary literature and claim boundaries

The following primary sources were checked online during the audit. Full-text inspection was used for the central localization comparison and specific CodeAnchor passages; some newer comparisons are based on verified abstracts and are labelled accordingly. This is a focused novelty check, not a systematic review proving absence of every competing method. ArXiv submission dates or “submitted” comments do not establish conference acceptance.

| Proposed contribution | Closest prior art and verified overlap | What could still be contributed here |
|---|---|---|
| Code is found but repair fails | [From Finding to Fixing](https://arxiv.org/abs/2610.25826), submitted October 19, 2026, studies six localization approaches, three models, and 16,248 trajectories. Improved localization does not reliably improve repair; causal descriptions matter. | Replication on this quantized Gemma setup and an intervention-based decomposition. High exposure alone is not new. Its 11.6% differing-file result concerns 127/1,095 successful model-target pairs, not 11.6% of all tasks. |
| Post-edit change-impact/obligation planning | [CodePlan](https://arxiv.org/abs/2309.12499) uses repository-level planning, incremental dependency analysis, and change impact. | A specifically tested gold-relative obligation metric and negative rule-retrieval result. General dependency-conditioned editing is established. |
| Graph retrieval and graph-guided localization | [RepoGraph](https://arxiv.org/abs/2410.14684) and [LocAgent](https://arxiv.org/abs/2503.09089) use repository graphs for code navigation/localization. LocAgent's reported downstream Pass@10 is not Pass@1. | A controlled failure analysis of a particular graph index and typed rules, with strong lexical baselines. No general refutation of graph methods follows. |
| Passive structural context versus optional graph tools | [CodeAnchor](https://arxiv.org/html/2606.26979) discusses an optional-tool pilot on 20 tasks and evaluates injected structural context. | The repository's “never offered an optional tool” characterization is too absolute. A motivating pilot exists; it is not a full controlled optional-versus-passive effect-size study. Its localization Func@5 metric must not be presented as issue-resolution Pass@1. |
| Diagnostic localization may help | [SHERLOC](https://arxiv.org/abs/2606.24820) reports repair gains from structured diagnostic findings. | A reason to separate location from useful causal evidence. The current negatives do not establish that localization or diagnostics are generally ineffective. |
| Necessary edit minimization and patch interventions | [TRIM](https://arxiv.org/abs/2607.18161) minimizes trajectories/edits with test revalidation. [RGFL](https://arxiv.org/abs/2601.18044), checked at abstract level, explicitly proposes counterfactual upper-bound analysis of localization stages. | A4 and A12 may provide a useful paired completion/replacement diagnostic, but “first counterfactual SWE analysis” or “first necessity analysis” is not justified. |
| Dependent and multi-hunk repair | [ChainSWE](https://arxiv.org/abs/2607.02606) studies sequential dependent multi-bug issues; [SiblingRepair](https://arxiv.org/abs/2605.06209) retrieves semantically related siblings for multi-hunk repair. | Single-issue gold-relative groups under a constrained model are a different setting. A4 does not bound ChainSWE's mechanism; lexical co-change retrieval has substantial prior art. |
| Test-grounded diagnosis | [SWE-Doctor](https://arxiv.org/abs/2607.00990) studies bug-reproducing tests and diagnostic assistance. | A careful measurement of whether this agent uses reliable test evidence. An untested diagnosis architecture is not a contribution yet. |
| Reliable repair, verification, and post-training | [Teaching Agents to Code Reliably](https://arxiv.org/abs/2610.03984), submitted October 2, 2026, checked at abstract level, addresses limited location/edit diversity, misleading self-tests, patch-versus-reverted-tree scoring, and training a verifier. The page says submitted to ICLR 2027. | This is especially close to a proposed “repair correctness” method. The present project can contribute validated failure diagnostics and negative controls; it cannot claim to originate verification-aware repair or its training recipe. |
| Tests versus semantic correctness | [Are “Solved Issues” in SWE-bench Really Solved Correctly?](https://arxiv.org/abs/2503.15223) develops PatchDiff and studies plausible but incorrect patches. | “Resolved” here must mean target-test-passing, not semantically proven correct. Hybrid failure must likewise be diagnostically distinguished from semantic wrongness. |
| Contamination, freshness, and public training | [Schrödinger's Repo](https://arxiv.org/abs/2609.27891) examines repository cue sensitivity; [SWE-rebench](https://arxiv.org/abs/2505.20411) expands public issue evaluation; [SWE-smith](https://arxiv.org/abs/2504.21798) demonstrates large-scale agent training. | The study can disclose familiarity/selection threats. Neither fresh public tasks nor renaming proves contamination removal; these small SFT failures do not contradict successful training elsewhere. |

### What is new, confirmatory, or incremental?

**Potentially distinctive, conditional on validation:** the paired use of missed-gold completion and gold-background site replacement to interrogate an agent's partial repairs, with explicit incompatibility and verifier controls. The particular empirical combination is worth studying. It is not a new universal theory of repair, and the instrument must work before its failure rates can bear the claim.

**Confirmatory but useful:** exposure does not ensure success; larger changes are harder; small-output prompting can alter behavior without improving task resolution; imitation loss and prefix gates can diverge from end-task performance. These observations are valuable when carefully scoped and accompanied by task IDs, controls, and negative results.

**Incremental engineering evidence:** this typed graph/rule implementation loses to BM25 on E0b; v2 rules improve a reused pilot; optional tools see low use. The artifact, selection disclosure, and fair baselines can make these worthwhile secondary findings. They cannot carry an architecture novelty claim.

**Not established:** an EBST advantage, a guarantee about stopping error, graph-theoretic explanation of LLM behavior, a general impossibility of multi-site repair, a universal oracle ceiling, causal patch-size effects, or private-repository transfer.

The current corrected `references.bib` and literature review should not be accused of retaining every historical citation error flagged by the surveys. However, the active bibliography and paper need direct entries and precise discussion for the closest localization/completion/verification work above, particularly the newer reliability paper. A subjective novelty score such as 98/100 in `novelty_stress_test.md` has no evidential standing. Novelty should be assessed claim by claim.

## Part 6 — Strongest research direction

### Ranking

| Rank | Direction | Novelty / importance | Current support | Feasibility before November 5 | Principal reviewer risk |
|---|---|---|---|---|---|
| **1** | **B — repair correctness through validated counterfactual diagnosis** | Moderate potential methodological value; important problem with strong prior art. | A4/A12, complete v14/v16 traces, meaningful failure contrasts. | Good for CPU recovery/adjudication if original interventions can be reconstructed. | “Your hybrids are incompatible; you did not identify wrong local edits.” |
| **2** | **A — strengthen the broad Found, Not Fixed empirical paper** | Modest novelty; credible negative/replication study if restrained. | Broadest existing measurement portfolio. | Highest feasibility; many statistical/presentation fixes need no new evaluations. | “Single-model anecdotes, too many RQs, familiar conclusion.” |
| **3** | **C — counterfactual-diagnostic validity/reproducibility study** | A concrete gap in validation of hybrid repair diagnostics; novelty needs further comparison. | Synthetic false-clean case, missing intervention provenance, aggregation inconsistencies. | Feasible as an artifact/case study; a broad new benchmark is unrealistic by freeze. | “Synthetic pathology without measured real-world prevalence.” |

### Direction B: recommended primary plan

The question becomes: **under what verified conditions can partial-repair interventions distinguish missing edits from incorrect local behavior?** The technical contribution would be the validated intervention protocol, controls, task-level uncertainty, and diagnostically adjudicated results. Strong baselines are original agent patches, complete gold patches, missed-only patches, self-swaps, identity/round-trip transformations, and compatibility-preserving hybrids. Relevant prior work includes the counterfactual, patch-verification, and diagnostic studies in Part 5.

Required work is evidence recovery, deterministic CPU reconstruction, outcome classification, corrected exposure/termination analysis, and transparent statistics. Compute cost can remain CPU-only; the main cost is human diagnosis and environment reconstruction. Do not add fine-tuning or claim a new corrective policy before the deadline. The reward is a coherent empirical contribution; the risk is that actual compatibility failures weaken the headline. That is a legitimate possible result, not something to hide.

### Direction A: fallback and paper framework

A carefully scoped audit of a constrained agent has scientific value, especially when it documents failed predictions rather than selecting successes. Its technical content is the evaluation artifact and controlled comparisons, with lexical retrieval, size controls, and the training/gate mismatch as secondary evidence. It requires better measurement, inclusion flows, and reproducible feature rows; no additional GPU model run is necessary.

Its publication potential is strongest as a concise empirical/negative-results submission. It is weaker as a broad research breakthrough, and a 3,000-word paper cannot responsibly carry five major RQs plus a training study. If A12 remains unrecoverable, use this narrower descriptive framing rather than promising causal diagnoses the artifacts cannot support.

### Direction C: justified alternative, not another architecture

A compact study of diagnostic validity could systematically test whether code-splicing instruments preserve namespaces, signatures, decorators, imports, state initialization, and test identity. Negative controls and invariance tests are its baselines. Its importance is methodological: unsupported localization-versus-repair conclusions can arise from invalid interventions. Current support is a demonstrated potential defect, not a measured prevalence study.

Limit this to a reproducible validation artifact and real-case adjudication, roughly the work needed for B anyway. Do not announce a general benchmark or a new training algorithm without a representative case collection and literature comparison. If hybrids often fail for composition reasons, C becomes a more honest contribution than retaining “73% wrong local edits.”

**Recommendation:** pursue B within A's existing dataset and paper infrastructure, with C as a predeclared outcome if validation changes the conclusion. This keeps the research decision tied to evidence instead of selecting a title first.

## Part 7 — Proposed improvements

These are the **five highest-priority improvements before the deadline**, in order. Each specifies a testable hypothesis, alternatives, and a decision. They are proposals; this audit did not modify the implementations or run the proposed repository tests.

### Priority 1 — Validate A12/A4 intervention compatibility and failure causes

1. **Problem:** protocol-labelled clean hybrids may fail because of incompatible composition; A4 completion may fail for application or environment reasons.
2. **Precise hypothesis:** among the 71 primary A12 swaps, a substantial failure fraction remains after namespace/interface validation and successful control transformations, and those failures exhibit target-behavior defects attributable to the swapped site. The original ≥60% threshold can be reported as historical; any new decision threshold must be frozen before adjudication.
3. **Current evidence:** 52/71 failures, 19/20 tasks; passing gold controls; two exit-2 and one exit-124 primary outcomes; the demonstrated synthetic false-clean case; A4's low rescue and `rich_4070` antagonism.
4. **Required data:** exact base commits/source, gold/agent patches, serialized hybrid patches, original verifier commands/environments, full stdout/stderr, timestamps, necessity labels and their provenance. Recover original evidence before trying to reconstruct it.
5. **Procedure:** enumerate all primary swaps and A4 cases; validate gold→gold self-swaps, agent→agent round trips, and namespace-aware symbol/interface compatibility; compile and run the frozen target tests in disposable CPU environments only after separate authorization if needed. Two reviewers classify assertion failure, import/collection error, timeout, transformation error, incompatible hybrid, or unresolved ambiguity. Preserve every original and reconstructed artifact separately. Start with the six v14 primary tasks, then cover all 20 primary tasks/71 swaps without selecting easy examples.
6. **Statistics:** task-cluster intervals for swap-weighted and task-macro failure; report any-failure task count with opportunity counts; sensitivity treating ambiguous cases as all failures and all nonfailures. Keep invalid/unknown counts visible instead of silently dropping them.
7. **Baselines/controls:** original gold and agent reruns, identity transformations, missed-only controls, actual dependency-preserving replacements, and the known namespace false-clean construct.
8. **Competing outcomes:** persistent assertion failures under compatible swaps support local repair defects; failures disappearing after compatibility fixes support composition confounding; unstable gold controls support an environment explanation.
9. **Falsification:** withdraw C7's strong local-error interpretation if real clean failures cannot be distinguished from transformation/coupling failures, or if the corrected conclusion depends on ambiguous cases. Do not revise rules until a preferred rate returns.
10. **Effort:** approximately 16–24 person-hours if inputs are available; stop recovery after a bounded effort if critical artifacts cannot be obtained, and publish the resulting limitation.
11. **GPU requirement:** none. CPU verifier replay may need cached dependencies/sandbox setup and explicit permission beyond this read-only audit.
12. **Paper impact:** highest value. It can establish the core methodological result or force a scientifically necessary change of thesis.

### Priority 2 — Measure actual evidence consumption, mutation, and termination

1. **Problem:** the current “found,” “edited,” “tested,” and “voluntary submit” proxies mix intent with execution.
2. **Precise hypothesis:** a measurable subset of failures has successfully returned source for the necessary sites before mutation yet still fails under reliable verification; separately, budget/context termination explains some failures currently labelled voluntary.
3. **Current evidence:** original v14 59/60 file and 55/60 line exposure; synthetic line-exposure false positive; four failed nonempty v14 denial→submit trajectories; complete v14/v16 traces; missing v7 raw traces.
4. **Required data:** v14/v16 call IDs, arguments, successful response content, truncation flags, task snapshots/line maps, actual final patches and test outcomes. Treat v7 quantities requiring absent traces as unavailable.
5. **Procedure:** build an offline event ledger associating calls/responses; distinguish mention, successful source read, returned target-span overlap, and all-required-site coverage before first real mutation. Detect edits via file tools, Python/shell mutations, and final diff reconciliation. Classify final error, any denial, last paid operation, submit after denial, test execution, and relevant passing/failing assertions. Restore A6 as this observational analysis. Manually cross-check a stratified sample and all headline cases with a second reader.
6. **Statistics:** paired v14/v16 event prevalence; task-level intervals; confusion tables for old versus corrected labels; agreement/disagreements across reviewers. No claim that exposure causes repair success from conditional rates alone.
7. **Baselines/controls:** original analyzer outputs, successful tasks, known failed reads, truncated `cat`, duplicate function names across classes/files, Python-based edits, and denial→free-submit examples.
8. **Competing outcomes:** high verified pre-edit exposure supports an execution/diagnosis problem after access; low verified coverage supports remaining information/planning deficits; reliable tests followed by unchanged wrong patches suggest verification-use failure.
9. **Falsification:** reject “found the necessary code” when only file mentions or partial snippets are established; reject “voluntary stopping” when denial/context/timeout precedes completion; keep mixed/unknown cases.
10. **Effort:** approximately 10–16 person-hours; reuse parsers but create a separately versioned analysis later rather than overwriting historical outputs.
11. **GPU requirement:** none; existing traces suffice.
12. **Paper impact:** repairs the title's key measurement and the erroneous budget claim; may support a more useful distinction between access, action, and verification without inventing a cognition measure.

### Priority 3 — Correct task dependence, uncertainty, selection, and statistical claims

1. **Problem:** repeated-task pooling, correlated pairs, weak matching, missing feature rows, and wide intervals are being interpreted too strongly.
2. **Precise hypothesis:** the size association remains descriptively robust under task-aware inference, while the evidence cannot precisely separate size from multiplicity. A statistically null independent multiplicity effect is not assumed.
3. **Current evidence:** A1/A13 wide OR intervals; 15/17 multi-file tasks are large; raw paired outcomes and necessity labels reproduce; pooled Fisher and matching limitations identified.
4. **Required data:** recover A1/A8 feature rows and exact transformation definitions, all 60 task IDs, exclusion flags, source/gold-size measures, issue-length features, repository identity, and run identifiers.
5. **Procedure:** publish a 60→57 inclusion table; reproduce model design matrices; use a small predeclared model with run fixed effects and task-aware uncertainty; examine penalized estimates/separation; display overlap and matching balance, remove invalid independent-set permutations. Add v16 as an explicitly exploratory third configuration using its existing records. Report leave-one-repository-out descriptive sensitivities without pretending that three repositories support precise repository-cluster inference.
6. **Statistics:** exact paired tests for configuration outcomes; task bootstrap or appropriate task-cluster methods for repeated runs; effect intervals and marginal risk differences; a declared multiplicity family for confirmatory questions. Use regularized-model/bootstrap sensitivity rather than claiming one small-sample p-value settles causality.
7. **Baselines/controls:** unadjusted task-level results, original reported models, size-only and multiplicity-only models, all-60 descriptive strata, and the same model restricted to the documented 57-task subset.
8. **Competing outcomes:** stable size association with uncertain multiplicity supports a difficulty correlate; a stable independent multiplicity effect supports additional coordination difficulty; instability supports a primarily descriptive conclusion.
9. **Falsification:** withdraw strong adjusted claims if feature rows cannot be reconstructed, effects reverse under reasonable specifications, or interval width leaves practically important alternatives open. Nonsignificance is not falsification of an effect.
10. **Effort:** approximately 8–12 person-hours, dependent on recovering feature rows. Cap model search; more specifications are not more confirmation.
11. **GPU requirement:** none.
12. **Paper impact:** improves credibility and protects against a straightforward statistical rejection; may reduce the number of claimed mechanisms, which strengthens the paper's honesty.

### Priority 4 — Separate retrieval observations from lexical/structural mechanism claims

1. **Problem:** graph bundle differences, aggregation, keyword overlap, oracle seeds, and correlated regression rows are conflated with a general explanation of co-change retrieval.
2. **Precise hypothesis:** BM25's advantage over the frozen original typed system persists under task-macro aggregation and non-generic token definitions, but the current data do not isolate typing itself as the cause.
3. **Current evidence:** E0b's negative clustered difference, A5 macro/micro differences, A8's failed no-shared-edit prediction, and post-hoc A8b improvement without passing the gate.
4. **Required data:** existing retrieval rows, uncapped edited-token sets, original/v2 rules with hashes, candidate rankings/sets, source revisions, and A8 feature rows. Real first-agent-edit queries can be examined where available, with failures/absent cases disclosed.
5. **Procedure:** recompute micro and macro metrics under the same definitions; remove keywords/generic names; stratify by candidate count, repository, cross-file status, and co-change size. Audit BM25 documents/query content for edit/future-information leakage. If a small additional CPU ablation is warranted later, hold candidates and locality weights fixed while changing only edge-type allowances. Keep original and v2 analyses separate; freeze new rules before testing any unused collection.
6. **Statistics:** task-level paired retrieval differences and bootstrap intervals; clustered explanatory models with standardized effects and held-out evaluation. Gold-edit features remain explanatory oracle variables rather than deployable query features.
7. **Baselines/controls:** BM25, same-file proximity/random, original untyped and typed bundles, v2 bundles, and common-candidate one-factor ablations. Do not omit lexical baselines after a rule revision.
8. **Competing outcomes:** persistent BM25 advantage supports lexical similarity as a useful baseline here; disappearance after fair weighting identifies bundle confounding; poor real-agent-query performance shows the oracle-seed task was too favorable or mismatched.
9. **Falsification:** abandon “typing hurts” as an isolated mechanism unless single-factor comparisons support it. Do not revive the frontier method from improved scores on the reused seven-task pilot.
10. **Effort:** approximately 6–10 person-hours for existing-row analysis; cap new indexing work if pinned source inputs cannot be recovered.
11. **GPU requirement:** none; no new closed-loop agent evaluation is recommended before freeze.
12. **Paper impact:** makes a strong implementation-specific negative result reproducible and prevents an unnecessary general attack on graph methods. It is secondary to intervention validation.

### Priority 5 — Produce an auditable, narrow paper and public evidence package

1. **Problem:** missing raw intervention evidence, stale outline statements, overlapping prior art, and too many RQs undermine an otherwise useful empirical study.
2. **Precise hypothesis:** every headline claim can be traced to a frozen input, explicit inclusion set, reproducible calculation, and appropriately scoped prior-work comparison without relying on private workstation files.
3. **Current evidence:** many reported counts reproduce; the current bibliography is improved; some raw rows/patches are absent; v16 evidence exists despite the stale A9 statement; close verification papers narrow novelty.
4. **Required data:** a minimal manifest of source revisions, hashes, run/experiment IDs, exclusions, public-versus-local availability, commands, analysis versions, and citation metadata. No credentials, private authentication files, or unnecessary large archives belong in a paper package.
5. **Procedure:** create a future authorized evidence index linking each claim to an immutable row/table calculation; preserve all failed predictions; add exact missing-artifact statements; second-read the 19-task A5 typology with disagreements retained. Rewrite a three-RQ, approximately 3,000-word paper and verify every citation against its primary source and metric. Have Atul independently reproduce the headline tables from the frozen package before the final draft.
6. **Statistics:** a complete claim/estimand/unit-of-analysis ledger with multiplicity labels; descriptive agreement for manual labels; check that each displayed interval and p-value uses the advertised denominator and dependence structure.
7. **Baselines/controls:** existing public artifacts, a clean-environment CPU reproduction, original negative results, and a reviewer checklist including the strongest alternative explanations.
8. **Competing outcomes:** a compact reproducible package supports submission; unresolved private dependencies or uncheckable central claims require narrowing the paper rather than filling gaps with inferred values.
9. **Falsification:** any headline number lacking recoverable rows or an explicit report-only label fails the evidence gate; any originality claim with a close established equivalent must be rewritten.
10. **Effort:** approximately 6–10 technical person-hours plus writing/review time shared across the team. This is essential publication work, not an optional cosmetic pass.
11. **GPU requirement:** none.
12. **Paper impact:** distinguishes a credible negative-results contribution from a narrative assembled after many experiments; reduces preventable novelty and reproducibility objections.

## Part 8 — Research roadmap

The November 5 freeze, November 12 deadline, and approximately 3,000-word limit are taken from the project's current planning documents and the user's instruction. This audit did not independently verify competition rules or authorize any submission. The plan prioritizes existing CPU evidence and has explicit fallback decisions.

| Priority / dates | Atul's responsibility | Shrey's responsibility | Required handoff or decision |
|---|---|---|---|
| **P0: October 27–29** | Independently audit the six v14 A12 tasks first, then the remaining primary cases. Record compatibility, actual test failure, missing evidence, and disagreements without seeing the preferred conclusion as a label. | Recover exact counterfactual inputs/patches/test tails and frozen environments; design namespace-aware and identity controls in a separately authorized branch/artifact. | A 71-swap ledger with original class, validated class, failure cause, provenance, and unknowns. Do not continue drafting C7 as settled while this is unresolved. |
| **P0: October 28–30** | Reconcile denial→submit and source-read events in v14/v16; check headline examples manually. | Implement offline event reconciliation and qualified site/span mapping; recover A1/A8 design rows. | Corrected measurement tables, documented confusion with old labels, and a clear list of unrecoverable records. |
| **P1: October 30–November 1** | Recompute headline paired/clustered statistics independently; second-code A5 typology and check denominator arithmetic. | Freeze a small statistical plan, regenerate models/intervals and selection diagrams, include existing v16 outcomes as exploratory evidence. | Agreement on units, estimands, multiple-comparison labels, model limitations, and every displayed count. |
| **P1: November 1–3** | Verify closest-paper claims and source availability; test reproduction instructions on the frozen small package. | Complete bounded retrieval sensitivities; draft the three-RQ narrative and figures. | If A12 validity remains uncertain, switch to Direction A/C's descriptive/diagnostic framing. No new architecture or GPU run to rescue the story. |
| **P1: November 3–4** | Adversarial review: try to invalidate each causal sentence; verify literature metrics and all claim-to-evidence links. | Remove unsupported claims, resolve reviewer disagreements visibly, prepare a reproducible public artifact with missingness disclosures. | A go/no-go claims ledger: validated, descriptive-only, or excluded from the paper headline. |
| **November 5: freeze** | Sign off on arithmetic, provenance, exclusions, and original evidence preservation. | Freeze analysis code/data versions, results, claims, figures, and paper scope. | No new result-driven rule changes; corrections remain allowed and must be logged transparently. |
| **November 6–9** | Reproduce final tables and edit methods/limitations for precision. | Lead manuscript integration to approximately 3,000 words and align contributions with evidence. | Complete draft with readable figures, direct prior-art comparisons, and all negative outcomes retained in paper/artifact. |
| **November 10–11** | Perform a final skeptical review and independent evidence lookup. | Address only correctness, clarity, formatting, and citation issues; check actual submission requirements. | Final document ready for the team's own approval. |
| **November 12** | Joint final review. | Joint final review and user-authorized submission handling outside this audit. | Submit only through the team's normal authorization process; this audit performs no publication action. |

Estimated technical work is roughly 46–72 combined person-hours, depending on evidence recovery, plus concise writing and review. This is a planning range, not a measured runtime. At roughly five available research hours per day, prioritize P0 and defer retrieval refinements if time is scarce. Do not interpret two named owners as guaranteed full-time availability from both people.

**Scope cuts:** no new adapter training, leaderboard tuning, closed-loop frontier run, broad renamed-repository benchmark, or second-model campaign before freeze. If frozen environments cannot be rebuilt within about a day of focused recovery, record that limit and narrow the intervention claim. A trustworthy descriptive paper is preferable to an uninterpretable late experiment.

## Part 9 — Paper improvements

### Title and thesis

Recommended working title: **“When Partial Repairs Still Fail: Counterfactual Diagnostics of a Quantized Code Agent.”** A shorter “Found, Not Fixed” title can remain a subtitle after exposure and counterfactual validity are established. The current title risks equating file exposure with successful localization.

Recommended thesis, using only currently defensible evidence:

> In a public, repository-concentrated evaluation of one constrained quantized code agent, adding missed gold edits rescues few selected partial repairs, and replacing gold sites with agent versions frequently produces failing hybrids. These observations motivate compatibility-aware diagnosis of repair failure; they do not by themselves establish a universal localization bottleneck, completion bound, or semantic error rate.

After successful validation, replace the provisional second clause with the measured compatible-swap result and its uncertainty. Do not insert a guessed corrected rate now.

### Three research questions

1. **RQ1 — Access and action:** what source evidence was actually returned before mutation, and how do observed actions, verification, and termination relate to unresolved tasks?
2. **RQ2 — Counterfactual diagnosis:** under controlled compatible transformations, how often can gold completion rescue a partial repair, and which agent-site replacements fail for behavior, compatibility, or environment reasons?
3. **RQ3 — Scope and alternatives:** how robust are these findings to task/run aggregation, gold change size, and the lexical-versus-structural retrieval comparison?

The current five-RQ outline spreads attention across exposure, size, completion, sampling, and retrieval without giving the counterfactual validity enough space. Treat post-training and leaderboard history as contextual negative evidence or supplemental material.

### Contributions that the evidence can support

- **Primary, conditional:** an evaluated completion/replacement diagnostic with documented assumptions, identity controls, incompatibility categories, and task-level uncertainty.
- **Empirical:** a scoped assessment of this agent's failures using complete task pairing and corrected trace/termination measurements, preserving negative outcomes and uncertain causes.
- **Reproducibility:** an evidence package that exposes inclusion decisions, intervention identities, original verifier outcomes, and the limits of proxy metrics.

The typed-frontier failure is a secondary empirical result with strong lexical baselines. A new general repair algorithm, new stopping theory, or general graph-reasoning advance is not currently supported.

### Suggested 3,000-word allocation

| Section | Words | Content |
|---|---:|---|
| Introduction | 280 | Concrete repair-diagnosis problem, closest prior work, scoped contribution. |
| Setup and provenance | 450 | Model/harness, 60-task composition, old-54 use, gold selection, run pairing, exposure definitions. |
| Counterfactual method and controls | 450 | A4/A12 construction, necessity, compatibility, identity controls, failure statuses. |
| Results and uncertainty | 850 | Validated intervention results, task/case distinctions, trace/termination evidence, size associations. |
| Retrieval alternative | 250 | E0b primary comparison and its limitations; E0a/A8b as developmental context. |
| Related work | 300 | Localization versus repair, patch interventions/verification, graph/co-change retrieval. |
| Limitations | 300 | Selection, single model, concentration, contamination, imperfect tests, uncertainty, missing evidence. |
| Conclusion | 120 | What was learned and what remains untested. |
| **Total** | **3,000** | Bibliography/figure accounting must be checked against actual track rules. |

### Figures and tables with scientific purpose

1. **Inclusion and denominator diagram:** separate all-60 outcomes, the 57-task feature subset, A4's 13 tasks/26 cases, and A12's 29 candidate-bearing tasks/20 primary tasks/71 swaps. These are overlapping populations, not one simple causal funnel.
2. **Counterfactual outcome table:** agent, missed-only, completion, compatible swap, incompatible swap, verifier error, and unknown; report cases and distinct tasks side by side. Put original and validated classes in separate columns.
3. **Task-level intervention plot:** each task's swap counts/failures with run labels, revealing `rich_3052`'s 15 successful swaps and the dependence structure. Do not draw 71 independent Bernoulli error bars.
4. **Compact paired/robustness table:** v7/v14/v16 discordances, size associations with intervals, and micro/macro E0b comparison. Move the long A5 typology and training history to supplementary evidence if space allows.

The current proposed exposure→edit→resolve diagram should be revised so arrows denote observed sequence, not established causal mediation. Likewise, A4 should not be labelled an oracle bound in a figure, and A12 should not label every hybrid test failure “wrong local edit.”

### Related work and limitations that must be explicit

Discuss the close localization study as prior evidence, not merely a supporting citation. Compare completion/replacement diagnostics to TRIM and counterfactual localization analysis. Compare any correctness or test-based proposal to PatchDiff, SWE-Doctor, and Teaching Agents to Code Reliably. CodePlan and SiblingRepair precede post-edit dependency/co-change assistance. Keep measured localization metrics distinct from end-to-end repair metrics.

Limitations should include one model/quantization and scaffold, 32k context/resource constraints, 50/60 FastAPI concentration, public benchmark familiarity, selected gold-passing tasks, reused evaluation data, oracle first-edit retrieval, gold-relative necessity, potentially non-unique correct patches, non-exhaustive tests, repeated-task dependence, mixed inferred/tested labels, and missing original intervention diagnostics. No abstract should imply private-repository transfer or semantic correctness from target-suite passage alone.

**Publication assessment:** with valid controls and a compact artifact, this could be a useful empirical contribution for a constrained paper track. As written, a skeptical reviewer can reject the central interpretation on construct validity and judge the broad thesis as established by stronger prior work. The deadline is best spent on discriminating those objections, not accumulating another unpaired score.

## Part 10 — Final decision

**Should the team continue with “Found, Not Fixed”?** Yes, as a provisional empirical program with a narrower paper. Keep the name only with an operational definition of “found.” The current strong C7/C14 interpretations should not survive unchanged.

**What is the strongest publishable contribution?** A validated, reproducible completion-versus-site-replacement diagnostic of partial repairs, accompanied by a scoped negative result about this agent and transparent alternative explanations. At present the strongest verified empirical component is low rescue under the specified A4 completion protocol, supported by correctly paired task outcomes and an implementation-specific retrieval negative. The full semantic interpretation of A12 remains conditional.

**What is the biggest weakness?** Counterfactual construct validity: a failing hybrid labelled clean is treated as proof of an intrinsically wrong local edit, without sufficient compatibility checks or retained diagnostic evidence. Exposure and termination measurement errors compound that problem.

**What is the single most important next analysis?** Recover and adjudicate the exact **71 primary A12 hybrids**, starting with the six v14 tasks, using namespace-aware compatibility checks, gold self-swaps, agent round trips, and original verifier diagnostics. Determine how many failures remain attributable to local repair behavior rather than hybrid incompatibility or verification failure. Do not run another model evaluation to answer this question.

**Which directions should be abandoned or deferred?** Abandon EBST's unsupported empirical/theoretical claims; stop treating the original typed frontier as a promising primary method after its failed gates; abandon “mostly unfinished” as the default explanation and “voluntary stopping” as an established result. Defer new SFT scale-up, renamed-repository campaigns, closed-loop frontier assistance, and general graph/quantization impossibility claims. Keep all negative data and hypotheses in the historical record.

**What should Atul do next?** Independently inspect the A12/A4 evidence and denial→submit examples, document missing artifacts and failure categories, and second-read the manual typology. Atul's role should include trying to falsify the preferred interpretation rather than confirming Shrey's labels.

**What should Shrey do next?** Recover and freeze source/patch/test provenance, repair the diagnostic instrument in separately authorized work, export the missing feature rows, and narrow the manuscript claims. Shrey should lead a concise methods-first draft after the compatibility decision, with Atul reproducing its tables independently.

### Decision gates before the experiment freeze

- **If compatible, well-controlled swaps retain frequent behavior failures:** proceed with Direction B and report the corrected estimand, uncertainty, and selected scope.
- **If composition failures explain an important or uncertain portion:** report the diagnostic limitation as a finding, shift to Direction C/A, and remove the intrinsic-local-error headline.
- **If original interventions cannot be reconstructed:** retain aggregate protocol results as reported measurements with missingness disclosure; do not promote them to causal or semantic conclusions.
- **In every outcome:** preserve negative results, publish correct task dependence and denominators, and avoid an expensive late experiment that does not resolve the central validity question.

### Reproduction and preservation record for this audit

Audit-only working calculations were kept outside the repository in `/tmp/atul-independent-audit-20261027/`: `recompute.py`, `recomputed.json`, `coupling_construct_check.json`, and a preservation inventory. They cover exact paired tests, selected Fisher/Wilson calculations, A4/A12 outcome aggregation from saved rows, task-cluster sensitivity, E0 aggregation, A5 token sensitivity, gate-row aggregation, and two synthetic measurement checks. These temporary files are not a promised public artifact; the key methods and findings are included in this report.

Before analysis the checkout was clean on `main` at `313f50e9abdcd8703ac92e7eee66f8bf1054659b`. A 1,998-file preservation inventory recorded original file metadata, selected evidence/source hashes, symlinks, refs, stashes, and worktrees outside the repository. No existing document, script, notebook, dataset, experiment output, branch, stash, or Git history was intentionally modified. The report was assembled only after completing the analysis. Final checks verify its nonempty contents, all ten requested parts, unchanged original inventory/Git state, and the single new untracked report.

**Research recommendation:** use the remaining time to validate repair diagnosis and reduce claim scope. The scientific value is the reliable distinction among possible failure causes, including outcomes that contradict the preferred narrative.
