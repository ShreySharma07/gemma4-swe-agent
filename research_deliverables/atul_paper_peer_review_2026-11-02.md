# Independent paper peer review — Found, Not Fixed

**Audit date:** 2 November 2026 (IST). **Evidence cutoff:** local `main`, `8f8dd657467709ab27eb76eae43e6408986ecaa6`. No fetch was performed: this is the latest locally verified state, not a claim that remote main cannot be newer. The working tree initially had only the pre-existing untracked `data/runs/atul_r2_s1/` evidence. Existing sources, manuscript, results, datasets, Git index/history and Kaggle resources were not modified. No notebook, inference, training, benchmark replay or leaderboard submission was launched. Only new Markdown reports are repository additions.

**Evidence labels:** **VERIFIED** = independently inspected code or recomputed from available records, not a rerun of the experiment; **REPORTED** = documented result whose underlying measurement was not reproduced; **INFERENCE** = interpretation supported but not uniquely established; **HYPOTHESIS** = proposal requiring testing; **BLOCKED** = evidence/access insufficient. An arithmetic verification does not turn a recorded test classification into a proven causal diagnosis. Full repository paths and the shorthand path aliases below refer to the cutoff commit; local-only evidence paths are not public download links. No private issue bodies, fixture contents, patches, traces or credentials are reproduced.

## 1. Executive summary and scientific verdict

**INFERENCE — Major revision; not submission-ready today.** There is a credible, narrow empirical paper here. Its best contribution is the paired use of missed-gold completion and gold-background function replacement to diagnose selected partial repairs, including a valuable account of how the first replacement instrument failed validation. The current manuscript overstates what static compatibility checks and target-test failures establish, confuses exposure with verified code consumption, and carries an outdated blanket fine-tuning narrative. These are repairable with evidence presentation and CPU analysis; a new GPU training run is not a prerequisite.

**VERIFIED — Main numerical claims largely reproduce.** v7/v14 resolve 23/60 and 30/60; paired exact McNemar p=0.0390625. Completion rescues six cases from two of 13 tasks. A12v records 43 behavioral-classified failures, seven passes and one timeout among 51 selected swaps; its task-clustered percentile interval reproduces [0.625,1.000]. Those counts deserve publication, but they are not an estimate that 84% of arbitrary agent edits are intrinsically wrong.

**VERIFIED observations:** only four of the 51 primary swaps are v14, while 41 come from old-54 v7 repeats. The isolated checker probes return `validated-clean` despite changed defaults, a nested-scope error or a sync/async change. The identity helper reconstructs a function span from the same tree rather than invoking the actual hybrid builder.

**INFERENCE:** these observations limit the instrument's compatibility guarantees; a changed default need not itself be an invalid repair. The ten-case diagnostic read raises one reverse-composition concern and identifies one error-message-only failure. The newer training records also require qualifying a universal imitation-collapse interpretation. These are judgments about scope, not independently verified intrinsic-error classifications.

## 2. Scope, strengths and novelty

### 2.1 Strengths

- **VERIFIED:** the complete manuscript `research_deliverables/paper_draft.md:1–285` was reviewed, including references. **INFERENCE:** its failed predictions, population limitations, gold-relative labels, A12 correction and A4 non-monotonicity caveat are substantial strengths.
- **VERIFIED:** original outcomes exist for A3/A4, A12/A12v and external retrieval, permitting arithmetic checks and uncertainty reconstruction. The paper does not need to disguise these negative results as a new successful agent architecture.
- **INFERENCE:** the manuscript's three main RQs are substantially more coherent than the broader superseded outline/thesis. `paper_outline_find_vs_finish.md` and `thesis_found_not_finished.md` retain historical claims; they must not override the corrected manuscript.
- **VERIFIED:** the Oct 27 audit already rejected an unrestricted A4 upper bound, A12's namespace-union clean label and pooled independent-task Fisher inference. Most of that correction reached the draft. The outline still contains stale “upper bound” wording and should not be copied back into the paper.

### 2.2 Contribution-by-contribution novelty

| Contribution | Assessment | Defensible positioning |
|---|---|---|
| Exposure/localization | **INFERENCE:** useful replication, weak standalone novelty. File/path exposure is weaker than inspected-source reach in prior work. | Replication in one small quantized model/scaffold, with explicit measurement differences. |
| Completion A4 | **INFERENCE:** a useful concrete negative intervention, potentially distinctive in this pairing. Not a new theory of incompleteness. | Specified frozen-patch gold-completion protocol with composition modes and task-level rescue counts. |
| Compatible site swap A12v | **INFERENCE:** strongest methodological component, conditional on compatibility and artifact adjudication. | Gold-background replacement diagnostic and its validation/failure boundaries, not intrinsic semantic correctness. |
| Structural-retrieval negative | **VERIFIED** for this index/rule set and gold co-change labels; **INFERENCE** as a general explanation. | External oracle-seeded retrieval comparison that rejects this frontier, not all graph methods or closed-loop assistance. |

Closest prior work was checked against primary arXiv pages. [From Finding to Fixing](https://arxiv.org/abs/2610.25826), Shi, Yang and Chen (2026), already separates high reach from repair utility and warns that a single gold patch is incomplete location ground truth. [RGFL](https://arxiv.org/abs/2601.18044), Sepidband et al. (2026), explicitly includes counterfactual localization-stage analysis; its draft bibliography title is inaccurate. [TRIM](https://arxiv.org/abs/2607.18161), Mathai et al. (2026), tests trajectory/patch minimization. [CodePlan](https://arxiv.org/abs/2309.12499), Bairi et al. (2023), already propagates dependency/change-impact information through a planner. **INFERENCE:** avoid “first counterfactual SWE diagnostic” and “novel change-impact obligations.” The narrower paired diagnosis may be distinctive; no exhaustive priority claim is established.

## 3. Claim-to-evidence and statistics audit

Computations used existing JSON/JSONL only, exact `instance_id` joins, no task reruns. Exact two-sided McNemar is `min(1, 2*sum(comb(b+c,k), k=0..min(b,c))/2**(b+c))`. Wilson intervals use z=1.959964. Task bootstrap resamples task IDs with replacement, preserving every swap/repeat within a task; 10,000 samples, seed 0, sorted indices 250 and 9750, matching the implementation. These are selected-task intervals, not population sampling guarantees.

| Claim | Independent result/status | Evidence and implication |
|---|---|---|
| 23/60 → 30/60 | **VERIFIED:** both22, v7-only1, v14-only8, neither29; p=.0390625. | v7 `data/runs/eval_v7_new60/x/eval_v7_noedit/rep0/analysis/per_task.jsonl`; v14 `data/runs/atul_eval_v14_new60/eval_v14_thinking/rep0/analysis/per_task.jsonl`. Bundled scaffold change, not isolated thinking effect. |
| v14 vs v16 30/60 → 27/60 | **VERIFIED:** both26, v14-only4, v16-only1, neither29; p=.375. | v16 `data/runs/atul_eval_v16_new60/eval_v16_small_output/rep0/analysis/original_per_task.jsonl`. Not proof of equivalence or that context management cannot help. |
| Exposure 98.3%,91.7%; failures29/30 | **VERIFIED:** 59/60 file,55/60 intended-line proxy; Wilson29/30=[.8333,.9941]. | Analyzer input records reproduce, but the construct is path/range occurrence, not verified returned code or attention. |
| Empty patches25% →8% | **VERIFIED:** empty final patches are 15/60 → 5/60; nonempty unresolved counts are 22 → 25. **INFERENCE:** these totals do not establish the manuscript’s separately undefined category of 16 “wrong patches.” | Draft:96 needs a definition/subset for16; if “wrong patches” means nonempty unresolved, the statement is false. Final patch emptiness is independent of the heredoc-edit bug. |
| A3 ≥2 necessary groups 19/60 | **VERIFIED:**19 from `necessity_new60.json`;56 tasks have labels. All 22 multi-group gold controls pass and 22 empty controls fail. | `necessity.py:279–301` tests multi-group tasks and infers singleton necessity from earlier validity. Four unlabeled tasks must not silently count as known single-site. LOO failure does not prove no alternative fix exists. |
| A4 completion 2/13 tasks | **VERIFIED:**26 cases/13 tasks,6 completion passes on 2 tasks;5 pure cases on 1 task; agent 0/26; missed-only 3 passes. Wilson 2/13=[.0433,.4223]. | `results/runs/a3a4/{cases.json,results.jsonl}`. Any-success-per-task has unequal repeated opportunities. Five successful cases are repeated rich_3052. |
| A12 original 52/71 | **VERIFIED** as recorded arithmetic only; invalid clean instrument. | `results/runs/a12/`; use as validation history, not corroborating independent evidence. |
| A12v 43/51 | **VERIFIED:**43 behavioral-classified,7 pass, 1 timeout;19 tasks,18 with ≥1 failure; task-macro .894737; task-cluster micro-rate CI [.625,1]. | `results/runs/a12v/swaps_v2.json`; all 51 have recorded self_swap_pass, none fail recorded agent identity. Timeout remains in denominator: 43/51 is not 44/51 all-failure rate. 43/50 among non-timeouts is a separate post-hoc estimand. |
| 25 previously-clean swaps incompatible | **VERIFIED:**10 namespace + 14 interface + 1 attribute reclassifications. | `siteswap2_report.json`; class counts total 128:61 clean, 24 namespace, 15 interface, 5 attribute, 17 conflict, 6 identical. |
| Reverse-coupling exclusion 42/50 | **VERIFIED:** reported rule removes one primary behavioral failure; any-failure tasks 17/19 afterward. | Diagnostic sensitivity, not independent confirmation; reverse flag is incomplete regex-based, added post hoc. |
| A1 size/multiplicity | **REPORTED:** pooled size OR .468,p.0463; v14 size OR .367,p.02496; multi-file adjusted OR .321 CI[.0406,2.5409]. | `results/runs/a1/a1_report.json`, code `kaggle/a1_confound/a1_body.py:114+`. No independent refit: local per-task design matrix/gold-size payload not reconstructed. Association only, few repositories and near-collinearity. |
| A13 adjusted multiplicity | **REPORTED:**OR1.680 CI[.4458,6.3334], robust Wald p=.4433. Raw pooled Fisher p=.0056 is withdrawn, not valid confirmatory evidence. | `results/runs/a13/a1_a13_a7_a2_report.json`. Task-permutation p≈.048 remains **REPORTED/UNVERIFIED** in this audit; raw clustered permutation not rerun. Three-arm statement at draft:91 must define which arms/test produced p=.048. |
| E0b external typed vs BM25 | **VERIFIED:**1,192 pairs / 120 tasks / repositories; typed.188574,untyped.243809,BM25.421141,random.210906; typed−BM25 task bootstrap[−.285524,−.183683]. | `results/runs/e0b/e0b_pairs.json`;4,000 resamples, seed 0, indices 100/3900. Pair-weighted mean and task-macro are different estimands. |

**VERIFIED — Model/support imbalance:** A12v primary swaps comprise v14_new60=4, v7_new60=6, old-54 v7 repeats=41 (8/8/7/9/9). Only three distinct v14 tasks contribute. Five passing swaps are repeated rich_3052. A12v therefore cannot be headlined as a broad diagnostic of v14's 30 failures. Draft:9 switches to best-configuration failures, then pools a different population in the abstract; explicitly introduce that population switch.

**INFERENCE — Statistical limits:** task clustering correctly avoids treating 51 swaps as independent. It does not solve task selection, repository concentration, multiple analyses, repeated use of the same benchmark, or selection of counterfactuals from gold-relative coverage. Report numerator/denominator flow for every arm; use effect sizes/CIs ahead of p-values. Do not call the A1 borderline p=.046 a robust mechanism discovery. Matching has only 8 multi-file sets and 13 necessity sets, so absence of significance is not absence of a multiplicity effect. Repository-level generalization cannot be established from mostly fastapi tasks.

## 4. Counterfactual construct validity

### 4.1 A3 and A4

**VERIFIED:** `necessity.py:279–306` drops a gold group and tests the remainder; `:307–333` composes missed-group gold edits with an agent patch and may retry without module hunks. `cases.json` records 13 straight applies, 8 fuzzy applies, 5 no-module fallbacks. Completion executions contain six exit 0, fourteen exit 1 and six exit 2. **BLOCKED:** A4 has no retained output tails sufficient to distinguish assertion failures from collection/composition failures for exit 2. No causal semantic diagnosis should be assigned to those six cases from exit code alone.

**INFERENCE:** gold-relative necessary groups are a valid diagnostic label, not a canonical minimal semantic fix. Joint alternatives, group-merging rules, module-level hunks and fuzzy composition affect the intervention. A4 tests one frozen-patch completion strategy. Low rescue does not exclude an in-run localization/completion aid that changes existing edits, nor establish a mathematical upper bound even over all frozen-patch completions.

### 4.2 A12v checker: improvements and verified residual gaps

**VERIFIED:** A12v improves A12 by same-file names, decorator-inclusive spans, signature categories, self/cls attribute checks, gold controls and saved tails (`necessity.py:737–985`). These reduce important obvious false-clean cases.

**VERIFIED — Three safe synthetic CPU probes**, using only isolated pure AST helpers from `necessity.py` (no real task execution):

1. Gold `f(x=1)` versus an agent `f(x=2)` with a different body is `validated-clean`. `signature():863–866` compares argument names and default counts, not default values. It also omits annotations, variadic parameter names and synchronous-versus-asynchronous kind.
2. An agent function whose nested helper binds a variable, then loads that variable outside the nested scope, is `validated-clean`. `bound_in():812–831` flattens nested bindings through `ast.walk`, falsely treating a nested local as visible to the outer function.
3. Sync gold function versus async agent function with the same argument shape is `validated-clean`, despite changed call semantics.

These demonstrate checker incompleteness, **not** how often these faults occur in the real 51. No revised failure rate is invented. Attribute checks only track certain self/cls patterns and can miss aliases, inherited/dynamic contracts and reverse dependencies. Compile success is not namespace or runtime compatibility.

**VERIFIED — Identity-control limitation:** `identity_swap():934–948` reconstructs `lines[:s-1]+lines[s-1:e]+lines[e:]` from the same tree. For valid bounds this is the original text by construction. It does not run `build_swap_v2(tree,tree,group)` and compare actual builder output, and it does not invert the actual mixed-tree swap. Gold rerun 29/29 establishes baseline test viability, not validity of the splicing path. `agent_roundtrip_identical=113/128` is likewise a text identity report; all 51 primary cases pass the reported flag. Draft:147,158 should say what was actually tested rather than imply a fully exercised inverse instrument.

### 4.3 Second-reader review: ten independently chosen distinct-task failures

**VERIFIED — Selection:** filter `swaps_v2.json` to class `validated-clean`, necessary, self_swap_pass, observed result and failure_type `behavioural`. Sort by SHA-256 of `Atul-Nov2-independent-reader|` plus `patch_file`; take the first case for each unseen task until ten. This was independent of the earlier reader's unspecified random sample but not blinded to stored classifications. Full available tails, not just the final failure-type string, were read. Case identifiers below are already public metadata; private test bodies and exact fixture values are not reproduced.

**BLOCKED at the audit cutoff:** a search of this checkout found `patch_file` names and tails but not the retained `.patch` files or a local A12v archive. Thus this is a second read of failure diagnostics, not a full source-level adjudication of ten hybrids. The instrument writes patches during the run (`necessity.py:1035–1048`), but references in JSON do not establish reviewers can retrieve them.

| Exact patch/case ID | Independent classification and disagreement |
|---|---|
| `104_v7_old54_rep3_fastapi_14986.patch` | **VERIFIED:** root-path/server-list assertions fail; substantive target-test disagreement in hybrid. **INFERENCE:** consistent with behavioral label; intrinsic agent error vs composition not resolved without patch. |
| `018_v14_new60_requests_7328.patch` | **VERIFIED:** redirect/history tests report 3 failures with many passes. **BLOCKED:** retained tail is chiefly summary/warnings/server logs, insufficient to localize root cause. Agree test failure, withhold intrinsic-error adjudication. |
| `034_v7_new60_fastapi_14455.patch` | **VERIFIED:** generated security-schema assertion mismatch plus an error. Agree target-behavior mismatch; mixed failure must not be reduced to a single clean semantic cause. |
| `098_v7_old54_rep2_rich_3506.patch` | **VERIFIED:** display-cell segmentation/width assertions mismatch. Consistent substantive hybrid-behavior failure. |
| `121_v7_old54_rep4_rich_3061.patch` | **VERIFIED:** tab-expansion span assertions fail. Consistent substantive hybrid-behavior failure. |
| `014_v14_new60_fastapi_14609.patch` | **VERIFIED:** missing module attribute in a model-field compatibility path; already flagged reverse_coupling. **INFERENCE:** composition/namespace concern, not clean evidence of intrinsic edit incorrectness. Disagree with a strong semantic-error reading; retain in registered count but separately adjudicate/exclude only as sensitivity. |
| `113_v7_old54_rep3_rich_4006.patch` | **VERIFIED:** grapheme-span assertions fail. Consistent substantive hybrid-behavior failure. |
| `006_v14_new60_fastapi_14479.patch` | **VERIFIED:** assertion failures shown concern expected error-message wording. It is a target-test/API-contract failure, but not demonstrated incorrect accepted/rejected-input behavior. Disagree with calling this broad semantic repair failure without qualification. |
| `124_v7_old54_rep4_rich_3675.patch` | **VERIFIED:** terminal-override detection assertion fails. Consistent substantive hybrid-behavior failure. |
| `100_v7_old54_rep2_rich_3676.patch` | **VERIFIED:** trace-extraction failures include NoneType errors and note representation mismatch. Supports hybrid test failure; source-level coupling still unadjudicated. |

**INFERENCE — Summary:** seven substantive hybrid failure diagnostics, one summary-only/inconclusive cause, one message-contract-only and one explicit composition concern. All ten support nonpass under the target verifier; they do not establish ten independent intrinsic semantic mistakes. There is no defensible inter-rater kappa: the original reader's same-case annotations and blinded labels are unavailable. Shared wrapt warnings occur in passing gold controls too and are not the failure explanation. Do not revise 43/51 from this deliberately distinct-task audit sample; it is not an unbiased estimate of all swaps or tasks.

## 5. The heredoc analysis bug: exact manuscript dependencies

**VERIFIED:** `analysis/trace_analyzer.py:29,207–216` recognizes only edit_file/write_file as edits. v14 commonly mutates source through run_command Python/heredocs/sed. This breaks first/last-edit chronology, test-after-last-edit, repro-before-edit, edit-failure counters and verification-stage labels (`:270–286`). A tool status `ok` is also not proof tests passed. RS2's improved regex tagger is not a validated execution-state oracle either.

| Passage/table | Required treatment |
|---|---|
| Draft:96 “Thinking moved commitment, not correctness” | Remove isolated-thinking causal attribution; define “wrong patches 16.” Empty-patch counts themselves survive the bug, but final nonempty unresolved counts are 22 vs 25. |
| Draft:102 “15 failures touched some sites but not others” and:177 “9 edited elsewhere” | These are derived from final patch coverage in `analysis/find_vs_finish.py:41–92`, not chronology, so the heredoc bug does **not** automatically invalidate them. Function-name/hunk-header coverage is approximate and must remain labeled heuristic. Avoid throwing away unaffected evidence. |
| Draft:171 manual failures and:175 correctness interpretation | Not directly caused by edit chronology; affected by independent counterfactual/typing validity issues above. |
| Any copied RS1/R2 “~5% tested-after-last-edit,” failure_stage tables or “wrong_fix_unverified” explanation | Must not be imported into the paper as validated behavior. RS2 report records 79 train episodes with recognized pytest-after vs old analyzer's low counts; definitions and populations differ. Recompute source mutations and successful returned tests before new causal claims. |
| Draft:214,218 blanket failed-imitation conclusion | Update with T1/LP1/G1; this is stale evidence framing, not itself a heredoc bug. |
| Exposure statements:9,70–74,255 | Not an edit-tagger bug: analyzer:175–196 combines command arguments with output and intended ranges. Exposure is not guaranteed successful returned source. |

## 6. Section-by-section peer review and replacement wording

| Section | Finding and concrete recommendation |
|---|---|
| Title | **INFERENCE:** memorable and appropriate as a question-led diagnostic. Keep subtitle; do not make it an unconditional correctness theorem. |
| Abstract:9–15 | **INFERENCE:** give pooled intervention population and “static-screened gold-background hybrids”; replace “binding failure” with observed failure under protocol. Avoid best-v14 exposure implying all swaps are from v14. |
| Introduction:21–36 | **INFERENCE:** the hypotheses are clearly separated; claim 3 is strongest but conditional; contribution 1 “associated more than site count” should emphasize non-identifiability, not explanatory dominance. “Each pre-registered” needs E0 caveat earlier. |
| Setup:42–64 | **VERIFIED:** model, quantization, TP4, 32768 context documented. **INFERENCE:** specify effective 24576 prompt ceiling, tool/time/call budgets, package/source versions and exclusions; distinguish preregistration Git evidence from authors' asserted order. |
| RQ1:70–96 | **INFERENCE:** association is credible; confidence limited by 57 tasks/three dominant repos and correlated arms. Clarify likelihood-ratio p-values versus robust Wald p-values, missing labels, raw/adjusted estimands. Correct causal thinking title and wrong-patch count. |
| RQ2:102–177 | **INFERENCE:** central contribution. Add inclusion flow, per-arm primary counts, composition modes, classifier limitations, timeout treatment and second-reader caveat. Explicitly mention error-message-only failures and retained-patch access gap. |
| RQ3:183–206 | **VERIFIED:** the typed index loses to BM25 in the recorded E0b comparison. **INFERENCE:** no closed-loop frontier run occurred, so “giving structure does not fix it” is not a measured agent outcome. Distinguish gold co-change vs necessity labels and oracle first-edit seeds. |
| Other negatives:212–218 | **INFERENCE:** compress. LB differences from separate submissions are noisy and not clean ablations. Qualify old LoRA collapse with later successful loss transfer/inconclusive behavior. |
| Related work:224–239 | **VERIFIED:** close relevant references exist. **INFERENCE:** integrate precise comparisons; RGFL title wrong, seven titles blank, ChainSWE sequential setting differs. No comprehensive novelty priority claim. |
| Limitations:245–251 | **INFERENCE:** already good. Add checker default/scope/async gaps, primary arm imbalance, selected-success validation and data access. Replace no second reader with completed tail-only audit plus source blocked. |
| Conclusion:255 | **INFERENCE:** “on screen,” “neither more context nor structure” and “lever remains correctness” exceed the measured design. End on observed protocols and open intervention hypotheses. |

Suggested replacements (proposals only; manuscript untouched):

- **Abstract/conclusion:** “In this selected population, gold-file paths appeared in tool calls or outputs for 29 of 30 v14 failures. Across pooled failed patches, the specified gold-completion protocol rescued 2 of 13 tasks, while 43 of 51 necessary-group hybrids passing our static screens failed target tests. These results diagnose limits of those completion and replacement protocols; residual hybrid incompatibility remains a threat.”
- **Draft:96:** “The bundled v14 configuration produced fewer empty final patches than v7 (5 vs 15 of 60), while resolution rose from 23 to 30. This comparison does not isolate thinking or establish unchanged local-edit correctness.”
- **Draft:131:** “A failing hybrid shows that the transplanted agent function does not satisfy the target suite in the reference-patch background; it may reflect local behavior, interface contracts or residual hybrid incompatibility.”
- **Draft:175:** “The registered failure-rate criterion was met for static-screened necessary swaps; this supports frequent target-test failure in that selected hybrid population.”
- **Draft:214:** “Earlier imitation adapters collapsed behavior. A subsequent on-policy full-context adapter reduced held-out weighted likelihood loss on both training and served bases, but a 95-task paired gate was inconclusive (40 vs 42, p=.754).”
- **Draft:255:** “The evidence motivates better diagnosis and verification of edits, but does not establish that localization, in-run completion, context management or graph-assisted policies cannot help.”

## 7. Citations, rules, reproducibility and word budget

**VERIFIED:** primary arXiv titles/IDs checked on November 2. Full author lists were visible; selected first author/year and exact titles below are usable bibliography corrections, not verification of publication venues. Current `references.bib` contains broader historical literature; manuscript's shorthand reference table still needs complete consistent entries.

| arXiv | Verified exact title; first author |
|---|---|
| [2309.12499](https://arxiv.org/abs/2309.12499) | CodePlan: Repository-level Coding using LLMs and Planning — Ramakrishna Bairi |
| [2410.14684](https://arxiv.org/abs/2410.14684) | RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph — Siru Ouyang |
| [2503.09089](https://arxiv.org/abs/2503.09089) | LocAgent: Graph-Guided LLM Agents for Code Localization — Zhaoling Chen |
| [2503.15223](https://arxiv.org/abs/2503.15223) | Are “Solved Issues” in SWE-bench Really Solved Correctly? An Empirical Study — You Wang |
| [2601.18044](https://arxiv.org/abs/2601.18044) | RGFL: Reasoning Guided Fault Localization for Automated Program Repair Using Large Language Models — Melika Sepidband |
| [2605.06209](https://arxiv.org/abs/2605.06209) | SiblingRepair: Sibling-Based Multi-Hunk Repair with Large Language Models — Xinyu Liu |
| [2606.26979](https://arxiv.org/abs/2606.26979) | How Much Static Structure Do Code Agents Need? A Study of Deterministic Anchoring — Zhihao Lin |
| [2606.24820](https://arxiv.org/abs/2606.24820) | SHERLOC: Structured Diagnostic Localization for Code Repair Agents — Hovhannes Tamoyan |
| [2606.07297](https://arxiv.org/abs/2606.07297) | SWE-Explore: Benchmarking How Coding Agents Explore Repositories — Shaoqiu Zhang |
| [2607.00990](https://arxiv.org/abs/2607.00990) | SWE-Doctor: Guiding Software Engineering Agents with Runtime Diagnosis from Multi-Faceted Bug Reproduction Tests — Yaoqi Guo |
| [2607.02606](https://arxiv.org/abs/2607.02606) | ChainSWE: Benchmarking Coding Agents on Multi-Bug Software Maintenance — Qirui Jin |
| [2607.18161](https://arxiv.org/abs/2607.18161) | TRIM: Reducing AI-Generated CodeSlop via Agent Trajectory Minimization — Alex Mathai |
| [2609.27891](https://arxiv.org/abs/2609.27891) | Schrödinger's Code Repository: Have LLMs Learned SWE-bench or Memorized It? — Silin Chen |
| [2610.03984](https://arxiv.org/abs/2610.03984) | Teaching Agents to Code Reliably — Muhammad Ahmed Mohsin |
| [2610.25826](https://arxiv.org/abs/2610.25826) | From Finding to Fixing: How Useful is Fault Localization for Agentic Program Repair? — Xinyu Shi |
| [2505.20411](https://arxiv.org/abs/2505.20411) | SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents — Ibragim Badertdinov |

**VERIFIED:** ChainSWE's full text concerns sequential dependent issues with accumulated repository state; it does discuss under/over-edits, but is not the same experiment as simultaneous multi-site repair in one issue. From Finding to Fixing's full text defines reach through inspection; our mention/range exposure cannot be called an identical measurement. RGFL's counterfactual work is relevant prior art, not a “Counterfactual analysis of localization” paper title. **BLOCKED:** no exhaustive full-text novelty search or venue/DOI audit for every citation; do not claim it.

**VERIFIED:** [official main competition overview](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview) still states paper deadline **November 12, 2026, 23:59 UTC = November 13, 05:29 IST**, as checked November 2. **REPORTED:** ≤3,000 words remains the active project limit (`paper_draft.md:3`,outline Constraints,`docs/HQ.md:274–278`). **BLOCKED:** separate paper-track overview was inaccessible through the browsing tool, so exact counting rules/reference exclusions and current submission UI mechanics were not independently reverified. Check them in the signed-in official paper-track UI before submission. November 5 is an internal experiment freeze, not an organizer deadline; no superseding local freeze was found.

**VERIFIED:** whitespace split counts are 3,232 whitespace tokens/words for the whole Markdown and 3,035 before References. These count table delimiters/Markdown syntax and are not the official editor's count. **INFERENCE:** the draft has essentially no safe word-budget margin; cut 200–350 words rather than adding a training paper. Replace long intervention class tables with a compact flow/linked public supplementary table if official rules allow it.

**INFERENCE — Reproducibility minimum:** pin commit; list exact task/run IDs and populations; define exposure and failtypes; publish aggregate JSONs and a safe case manifest; specify bootstrap estimand/seed/repetitions; cite preregistration/result commits with deviations; document restricted raw artifacts and legitimate access route. A local `patch_file` name or `/kaggle/working` path is not a public artifact. Do not attach private fixtures or traces to a public paper without explicit rights clearance.

## 8. What to include from the new training work

- **B: acknowledge briefly — INFERENCE.** Add one paragraph updating the older imitation claims: R2/RS2/T1/LP1 establish a viable on-policy data, training and serving path, but G1 does not establish a behavioral improvement. This prevents an unjustified universal negative and clarifies diagnosis versus intervention.
- **C: keep engineering details separate — INFERENCE.** RS2 masks, LC1 memory fixes, NF4/W4A16 compatibility, LP2 K/V ablations and training cost belong in the companion methodology report or a future training paper. They should not displace the diagnostic paper's central question.
- **A: include a narrow contrast only if space allows — INFERENCE.** G1's 40/95 versus 42/95 can illustrate that lower demonstration loss need not produce a verified repair gain. State selection/calibration caveats. It is not a test of verification supervision and cannot validate A12v's compatibility checker.
- **BLOCKED:** v18 submission ref 57892829 is documented in `docs/HQ.md:3–6`, but no committed score receipt or independently accessible score is available. Mark it **PENDING/UNVERIFIED**, not zero and not the v14/v17 score.

## 9. Ordered changes before submission

1. **Shrey, mandatory:** narrow the abstract, site-swap claim and conclusion to gold-background target-test failures. Insert the 51-swap arm composition and the population inclusion flow.
2. **Atul and Shrey, mandatory:** recover exact A12v hybrids and full diagnostics through an authorized existing-output transfer. Adjudicate the ten IDs above and namespace, attribute and message-only concerns. This audit launches no recovery job or verifier. If source artifacts remain unavailable, state the limitation and remove claims depending on source-level review.
3. **Shrey, mandatory:** correct compatibility and control descriptions. Default values, scope and async gaps must be explicit. Freeze typing rules before future additional testing; retain registered 43/51 and label sensitivities separately.
4. **Atul, mandatory CPU-only work:** establish mutation/test metric provenance independent of the old tagger. Distinguish final-patch coverage from action chronology, and test invocation from successful returned verification. Do not propagate old “unverified” counts.
5. **Shrey, mandatory:** replace causal thinking wording, clarify the wrong-patch count of 16, update the LoRA paragraph, and remove universal context/structure conclusions.
6. **Shrey, mandatory:** finish citation metadata, count words in the official editor, add a public evidence manifest and disclose preregistration deviations. Reconfirm deadline and word-count mechanics in the official paper-track UI.
7. **Optional:** task-balanced sensitivity, leave-one-repository-out descriptions, message-only sensitivity and clearer confidence-interval presentation using existing records. These do not justify a GPU training run.

**INFERENCE — Final decision:** preserve the diagnosis paper, make the five central corrections, and submit only with honest claim scope and artifact limitations. Neither a new architecture nor another LoRA epoch is required for a defensible revision. Training should continue only under a separately registered behavioral hypothesis, not to rescue the narrative.


## Publication corrections and privacy review — 2 November 2026

This appendix records only changes to this review before its user-authorized, review-only GitHub PR. The opening “No fetch was performed” describes the original audit, not this publication pass. The publication pass fetched `origin`, confirmed `main` and `origin/main` both at `8f8dd657467709ab27eb76eae43e6408986ecaa6`, and uses a dedicated review branch. The scientific evidence cutoff remains unchanged. No source paper, training implementation, experiment output or Kaggle resource was changed.

| Correction | Reason and scope |
|---|---|
| P1 — Separate observations from judgments | Split the executive findings into recomputed counts/code observations and interpretation. Relabeled presentation quality and the strength of retrieval evidence as interpretation rather than empirical verification. |
| P2 — Narrow synthetic compatibility claims | State what the checker accepted; a changed default can indicate an unchecked interface difference without proving an invalid real repair. Real-case prevalence remains unknown. |
| P3 — Preserve the incomplete second-reader evidence boundary | Clarified that the patch search was within this checkout at the cutoff. Ten diagnostic reads do not become a full hybrid-source review. |
| P4 — Avoid an unsupported replacement taxonomy | Kept the verified empty/nonempty unresolved counts but clarified that they do not determine the manuscript's undefined “16 wrong patches” category. No outcome count changed. |
| P5 — Repair citation/path and copy clarity | Added the path/access legend; fixed spacing and a dangling Wald-p phrase. All 16 unique arXiv abstract URLs and the official Kaggle overview URL returned HTTP 200 with matching page titles in this publication check. Link availability does not independently validate every scientific claim or the inaccessible paper-track rules. |
| P6 — Separate audit from publication authorization | Preserved the original read-only audit history while documenting this later branch/commit/PR authorization. No permission for training or evaluation is implied. |

**VERIFIED publication privacy check:** this report contains aggregate results, already-public case identifiers, synthetic checker examples and paraphrased diagnostic categories. It contains no private task/fixture bodies, non-public patch text, raw traces, credentials or access tokens. Local-only evidence is explicitly marked as provenance rather than a public download. Unavailable raw evidence and the unverified leaderboard score remain blocked/pending. The PR is for Shrey's review and is not to be merged automatically.


### Source-path and access legend

Short references such as `paper_draft.md`, `paper_outline_find_vs_finish.md`, `references.bib` and `thesis_found_not_finished.md` resolve under `research_deliverables/`. `necessity.py`, `lora_gate.py`, `lora_probe.py` and `kvdup.py` resolve under `kaggle/eval_l4/evalkit/`; `rft_core.py` and `rs2_core.py` under `kaggle/rftset/`; `data.py`, `worker.py`, `model.py`, `config.py`, `export.py`, `launcher.py` and `fix_*.py` under `kaggle/train/trainkit/`. A12v JSON filenames resolve under `results/runs/a12v/`, and A3/A4 label/case files under `results/runs/a3a4/`. Explicit `data/runs/` paths describe workstation evidence, not guaranteed public artifacts. Citations to these files disclose provenance, not permission to redistribute their private inputs.
