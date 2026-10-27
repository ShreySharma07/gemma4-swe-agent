# Found, Not Fixed: Counterfactual Diagnostics of Partial Repairs in a Quantized Code Agent

*Draft v1, Oct 27, 2026. Kaggle Gemma 4 paper track (≤ 3,000 words). Every number is traceable to a row in `docs/experiments.md` and an artifact in `results/runs/`. Claim status follows the ledger in `paper_outline_find_vs_finish.md`.*

---

## Abstract

Do code agents fail because they cannot find the code that must change? We test this on the competition's own stack: Gemma-4-31B (W4A16), 32k context, no compaction. Over 60 recent tasks, our best configuration was exposed to a file of the reference fix in 29 of its 30 failures. Resolution falls steeply with the size of the required change. Size and the number of sites to change cannot be separated in these data.

Two pre-registered counterfactual interventions locate the failure:
- **Completion.** Adding the reference edits the agent missed to its own partial patch rescues only 2 of 13 tasks.
- **Site swap.** Replacing a single reference function with the agent's version of it, with every other site correct, fails the target behaviour in 43 of 51 compatibility-checked hybrids (18 of 19 tasks).

A pre-registered structural co-change retriever lost to lexical retrieval on 120 external repositories. The binding failure is *what the agent writes* at the right place.

---

## 1. Introduction

Repository-level code agents spend much of their effort on localization: graph tools, retrieval indices and structural summaries that help them find where to edit [RepoGraph; LocAgent; CodeAnchor]. Recent evidence questions how much this matters. Across six localization methods and three models, better localization rarely improves repair, and descriptions of a bug's *cause* help more than its location [From Finding to Fixing].

We ask the follow-up question for a small, quantized, open model under the tight budget of the Kaggle Gemma 4 Developer Agent competition. If the agent sees the right code but still fails, *why*? Three explanations are possible:
1. it edits the wrong place;
2. it edits the right places but not all of them, i.e. it is unfinished;
3. its edits at the right places are wrong.

We separate them with two counterfactual interventions on the agent's own failed patches, built from leave-one-out *necessity* labels: a reference function is necessary if removing its edits from the gold patch makes the target tests fail.

**Contributions.**
1. **Exposure is near-saturated, and failure tracks change size.** We replicate the high-exposure finding on a new model and stack, and show the resolve gap is associated with change size more than with site count.
2. **A completion counterfactual.** Supplying the missed gold edits rescues 2/13 tasks (§4.1).
3. **A validated site-swap counterfactual.** Compatible hybrids fail the target behaviour in 43/51 swaps (§4.2). We also report how an earlier, unvalidated version of this instrument mislabeled 25 incompatible hybrids as "clean".
4. **A pre-registered negative result for structure-derived co-change recovery** against lexical baselines (§5).

Each analysis was pre-registered with a prediction and decision rule. We report the predictions that failed.

---

## 2. Setup

**Model and harness.** Gemma-4-31B-it QAT W4A16 is served by vLLM 0.19.1 (TP=4, 4×L4), using the official harness and tool set: `run_command`, `write_file`, `submit_patch`. The context is 32,768 tokens and no compaction occurs inside a task.

We study two agent configurations:

| Agent | Description | New-60 resolved | Public LB |
|---|---|---|---|
| v7 | Single agent; edits via Python snippets in `run_command` (the drift-prone `edit_file` tool removed) | 23/60 | 0.15 |
| v14 | v7 + thinking (4,096 tokens), revised budgets, one-line submit fix | 30/60 | 0.17 |

v7 → v14 is a bundled change: 8 tasks gained and 1 lost, exact McNemar p = 0.039. A third configuration, v16 (bounded tool output), resolves 27/60 (vs v14, p = 0.375) and is used as a secondary check.

**Tasks.** All tasks are verified: the gold patch passes the target tests and an empty patch fails.
- **new-60:** 60 public tasks (50 fastapi, 6 rich, 4 requests).
- **old-54:** an earlier set (39 rich, 9 fastapi, 5 requests, 1 httpx). v7 was run on it five times.

The repository concentration is a central limitation (§8).

**Labels.**
- *Exposure:* a gold file path, or an intended line range, appears in a tool call or output.
- *Function groups:* each gold hunk mapped to its innermost enclosing function.
- *Necessity:* group *g* is necessary if gold-minus-*g* fails the target tests. These labels are relative to the gold patch; other correct fixes may exist.

**Pre-registration.** Every analysis has a row in `docs/experiments.md`, committed before the analysis ran: prediction, decision rule and, after the run, the result. For the two retrieval tests (E0a, E0b), prediction and result share a commit, so their order rests on our word. All counterfactual runs used the scorer-faithful CPU sandbox with the competition's test environment.

---

## 3. RQ1: Exposed, but not resolved

**Exposure is near-saturated.** Exposure is measured by a proxy: mentions and intended read ranges, not verified returned spans.
- In v14, 98.3% of tasks were exposed to a gold file and 91.7% to gold lines.
- 29 of the 30 failures were exposed to a gold file (95% CI 83–99%).

This replicates prior work; here it rules out "never looked" as the main failure.

**Resolve falls with change size.** On the 57 tasks whose gold patch changes non-test Python, tasks were split into terciles of changed lines:

| Gold change size | v7 | v14 |
|---|---|---|
| Small (≤ 6 lines) | 12/20 | 15/20 |
| Medium (7–28) | 10/19 | 12/19 |
| Large (> 28) | **0/18** | **2/18** |

In a logistic model with issue length and repository as controls, each log-unit of changed lines lowers the odds of resolution (pooled OR 0.47, p = 0.046; v14 alone OR 0.37, p = 0.025). This is an association, not an effect of size.

**Site count is not separable from size.** Multi-file fixes resolve far less often in raw counts (v14: 3/17 vs 27/43). But 15 of the 17 multi-file tasks are also in the largest size tercile.
- After size control, the multi-file odds ratio is 0.32 (95% CI 0.04–2.54).
- The pre-registered prediction (OR < 0.4 with p < 0.05) failed.

The same holds with necessity labels. 19 of 60 tasks need ≥ 2 necessary groups.
- Raw outcomes for those 19 tasks against the other 38: v7 4/19 vs 18/38, v14 6/19 vs 23/38, v16 6/19 vs 20/38. A task-level permutation test gives p ≈ 0.048.
- After size control the OR is 1.68 (95% CI 0.45–6.33).

We conclude only that size and multiplicity cannot be separated in these data, not that multiplicity has no effect.

**Thinking moved commitment, not correctness.** v14's gain over v7 came mainly from fewer empty patches (25% → 8%). The number of wrong patches was unchanged (16). Gains were spread across small and medium changes; large changes went 0 → 2.

---

## 4. RQ2: Unfinished, or wrong?

The failure decomposition in §3 suggested partial coverage: in 15 of v14's 30 failures the patch touched some gold sites but not others. We test what that means with two interventions on unresolved, non-empty agent patches. Both are run through the official verifier.

### 4.1 Completion: add what the agent missed

**Design.** We took every unresolved patch from v7, v14 and the five v7 repeats that edits ≥ 1 gold function group and misses ≥ 1. That gave 26 cases from 13 distinct tasks. Three variants were run:

| Variant | Contents | Purpose |
|---|---|---|
| (a) agent | The agent's patch as-is | Control; must fail |
| (b) completion | Agent patch + gold hunks of the missed groups (+ gold module-level hunks) | The intervention |
| (c) missed-only | Those gold hunks without the agent's patch | Is the agent's part needed? |

**Result.**

| Outcome | Cases (of 26) | Distinct tasks (of 13) |
|---|---|---|
| (a) agent resolves | 0 | 0 |
| (b) completion resolves | 6 | 2 (95% CI 4–42%) |
| (c) missed-only resolves | 3 | 2 |
| Pure incompleteness: (b) passes, (c) fails | 5 | **1** |

Five of the six successful completions are the five repeats of a single task (rich_3052), which adds a `case_sensitive` parameter that must be threaded through a constructor, a factory classmethod and two consumers. On rich_4070, adding the agent's own edits turns a passing missed-only patch into a failing one.

The pre-registered predictions failed (completion ≥ 50%, pure incompleteness ≥ 35%). The decision rule we set in advance fired: the paper does not describe these failures as unfinished.

This measures one completion protocol on a frozen patch. It is **not** an upper bound on all completion aids: patch composition is non-monotonic, and an in-run aid could also change the edits the agent did make.

### 4.2 Site swap: keep everything correct except one function

**Design.** The mirror intervention. Start from the gold-patched tree and replace one gold function group with the agent's own version of that function. If the hybrid fails, the agent's function fails *even with every other site correct*.

The first version of this instrument (A12) produced 52/71 failures among "clean" swaps of necessary groups. An independent audit by a team member found that its "clean" label was not a valid compatibility check:
- it resolved names against all gold files at once, so a helper defined in another, unimported file counted as visible;
- it ignored decorators;
- it discarded the hybrids and the test output.

**Validation (A12v).** We pre-registered a validation and rebuilt all 128 swaps. A swap counts as *compatible* only if it passes four checks:

| Check | Rule |
|---|---|
| Names | Every name the agent's function loads resolves in the *same file* of the gold tree, or is local or builtin |
| Attributes | It reads no attribute that only the agent's other edits create |
| Signature | It keeps the gold signature (otherwise classed *interface-changed*) |
| Decorators | Decorators are included in the swapped span |

We added two identity controls: a gold→gold self-swap and an agent→agent round trip. Every hybrid patch and 4,000 characters of test output were kept, and failures were typed by a rule fixed in advance.

| A12v class | Swaps |
|---|---|
| Compatible ("validated-clean") | 61 |
| Name unresolved in gold file | 24 |
| Signature changed | 15 |
| Reads agent-added attribute | 5 |
| Cannot be built (function missing) | 17 |
| Identical to gold | 6 |

25 of the 85 swaps the first instrument called "clean" were incompatible. The controls behaved: the self-swap left the tree unchanged in all 125 buildable cases, and the gold patch re-ran cleanly on all 29 tasks.

**Result.** Among compatible hybrids of necessary groups whose self-swap control passed:

| Measure | Value |
|---|---|
| Fail the target behaviour | **43/51 (84%)** |
| Pass | 7 |
| Time out | 1 |
| Unclassifiable | 0 |
| Tasks with ≥ 1 behavioural failure | **18 of 19** |
| Task-bootstrap 95% CI | 0.63–1.00 |

A manual read of 10 random failures found genuine behaviour failures in all ten, e.g. a width computation returning 7 instead of 5 for wide characters, or an `IndexError` inside the agent's own function.

Excluding failures that could reflect gold's *other* sites depending on the gold version of the swapped function (a post-hoc check) leaves 42/50.

The pre-registered predictions held, and by the decision rule the claim "the agent's edits at the right sites are wrong" is supported for this population.

**Taken together.** Supplying the sites the agent missed rarely helps. With every other site correct, the agent's own function usually still fails. The agent is exposed to the right code, edits it, and writes the wrong behaviour. In 9 of v14's 30 failures it also edited a different place altogether, a separate failure mode we do not analyse further.

---

## 5. RQ3: Can code structure supply the missing co-changes?

Our original thesis held that agents fail to *finish* multi-site changes. It proposed computing an "obligation frontier" after each edit: the callers, constructor consumers, forwarded parameters and overrides that should change too, taken from a typed, directed graph of the repository. The idea of change-impact obligations is not new [CodePlan]. Our proposed contribution was delivering them passively inside an autonomous agent's edit loop.

**Pre-registered recall tests.** Before any agent run, we tested whether the frontier from a simulated first edit (the gold version of one function) contains the other functions that must change. The gate was hit@6 ≥ 0.80 and ≥ 0.20 above the best alternative.

| Test | Label | Typed graph | Untyped graph | BM25 | Same file, random | Provided graph |
|---|---|---|---|---|---|---|
| E0a: 7 tasks, 28 pairs | test-necessary | 0.45 | 0.54 | 0.36 | 0.30 | 0.27 |
| E0b: 120 external repos, 1,192 pairs | gold co-change | **0.19** | 0.24 | **0.42** | 0.21 | n/a |

Both gates failed. On SWE-rebench tasks from 120 repositories (one task each, none ours), lexical similarity to the edited function beat the typed frontier. The repository-clustered difference, typed − BM25, was −0.29 to −0.18. Adding two rules derived from E0a's misses, evaluated post hoc on the same seven tasks, raised the typed frontier to 0.63, still short of the gate and level with the untyped graph.

**What co-changes look like.** We read the gold diffs of the 19 new-60 tasks with ≥ 2 necessary groups:
- **Parameter or option plumbing (7 tasks):** e.g. a `strict_content_type` option threaded from the application through router, route and request handler.
- **The same edit repeated across siblings (7 tasks):** e.g. `field.alias` replaced with a validation-alias helper at seven call sites.
- **The rest:** helper/caller pairs (3), a constructor field (1) and one repository-wide refactor.

| Relation between necessary pairs | Task-averaged | Pair-level |
|---|---|---|
| Share an edited identifier (keywords excluded) | 89% | 65% |
| Directly linked by a structural edge | 35% | 16% |

Co-changes are thus more often textually than structurally related, which is consistent with BM25's advantage. That explanation is associational; a pre-registered prediction that shared edits alone would account for it failed.

Since completing patches with the *correct* co-changes rescues only 2/13 tasks (§4.1), we did not spend GPU budget on closed-loop frontier runs; their in-run effect remains untested.

---

## 6. What else did not help

Measured during the competition:
- **Optional graph tools:** zero calls in a version that offered them. The provided similarity search matches only fully qualified names and returned nothing for any of 461 phrase or bare-name queries.
- **Imitation fine-tuning (LoRA):** validation loss fell while behaviour collapsed. One adapter produced 114 empty patches out of 114. Another resolved 9/60 against the base agent's 23/60 (McNemar p < 0.001).
- **Larger budgets and forced early edits:** 70 calls scored 0.12 on the public leaderboard, and forcing an early edit plus a test scored 0.10, both below v7's 0.15.
- **Bounded tool output (v16):** context overflows fell from 8 to 5, with no gain in resolution (27/60).

None of these made the edits right.

---

## 7. Related work

**Localization versus repair.** From Finding to Fixing (2610.25826) shows across models that localization aids rarely improve repair and that cause descriptions help more. Our exposure result replicates this; our contribution is the counterfactual decomposition of partial repairs. SHERLOC (2606.24820) shows structured diagnostic findings help weaker repair models.

**Structure for agents.**
- RepoGraph (2410.14684) and LocAgent (2503.09089) use repository graphs for navigation.
- CodeAnchor (2606.26979) injects static structure as source comments for localization.
- CodePlan (2309.12499) propagates change-impact obligations through a dependency graph in a planner.
- Our negative result concerns one typed index and rule set on gold co-change labels, not graph methods in general.

**Incompleteness and multi-hunk repair.** ChainSWE (2607.02606) documents agents under-editing dependent changes; SiblingRepair (2605.06209) retrieves related sites for multi-hunk repair. In our setting, completion is real but rarely sufficient.

**Patch interventions and correctness.**
- TRIM (2607.18161) minimises agent edits by re-testing; counterfactual localization analyses also exist (2601.18044). Our interventions differ in applying *gold-relative* completion and replacement to the agent's own failed patches, with compatibility validation.
- Test passage is not semantic correctness (2503.15223); our "behavioural failure" means failing the target tests.
- Recent work trains verification-aware repair [Teaching Agents to Code Reliably, 2610.03984] and test-grounded diagnosis [SWE-Doctor, 2607.00990]. These point the same way as our findings.

**Memorization.** Renaming studies (2609.27891) show SWE-bench performance partly reflects familiarity with the repository. Our tasks are public, a threat we cannot remove.

---

## 8. Limitations

- **One model, one scaffold, Python only.** The tasks are concentrated in public repositories: 50/60 of new-60 are fastapi. Results may not transfer to private repositories, which is what the competition's hidden set uses.
- **Selected populations.** The interventions apply only to unresolved patches that edit gold function groups: 13 tasks for completion, 19 tasks with compatible necessary swaps. Repeats of v7 runs are correlated, so we headline task-level numbers. Small n gives wide intervals.
- **Gold-relative labels.** Necessity and "the right site" are defined against one reference patch; other correct fixes exist. A failing hybrid shows the agent's function does not work *with this gold design*. Our compatibility checks reduce, but do not eliminate, cases where the agent pursued a different valid design.
- **Proxies.** Exposure counts mentions and intended read ranges, not verified returned code. Failure typing relies on test output, and a second reader has not yet reviewed it.
- **Retrieval tests.** They use gold first edits as seeds, not real agent edits. E0b's labels are co-change, not necessity.
- **Size is observational.** We do not claim it causes failure.

---

## 9. Conclusion

For a quantized 31B agent under competition constraints, the right code is nearly always on screen. Failures grow with the size of the required change. When the agent edits a necessary site, its edit usually fails even with every other site correct, and supplying the sites it missed rarely helps. Neither more context nor code structure addresses this. The lever that remains is the correctness of the edit itself: understanding causes, and verifying behaviour before submitting.

---

## References (arXiv IDs from our Oct 22–23 surveys and Atul's Oct 27 audit; blank titles still to be checked against the source before submission)

| Ref | arXiv | Title |
|---|---|---|
| CodePlan | 2309.12499 | Repository-level coding using LLMs and planning |
| RepoGraph | 2410.14684 | Enhancing AI software engineering with repository-level code graph |
| LocAgent | 2503.09089 | Graph-guided LLM agents for code localization |
| SWE-bench solved? | 2503.15223 | Are "solved issues" in SWE-bench really solved correctly? |
| RGFL | 2601.18044 | Counterfactual analysis of localization |
| SiblingRepair | 2605.06209 | |
| CodeAnchor | 2606.26979 | How much static structure do code agents need? |
| SHERLOC | 2606.24820 | Structured diagnostic localization for code repair agents |
| SWE-Explore | 2606.07297 | |
| SWE-Doctor | 2607.00990 | |
| ChainSWE | 2607.02606 | |
| TRIM | 2607.18161 | |
| Schrödinger's Repo | 2609.27891 | Have LLMs learned SWE-bench or memorized it? |
| Teaching Agents to Code Reliably | 2610.03984 | |
| From Finding to Fixing | 2610.25826 | |
| SWE-rebench | 2505.20411 | Task collection and decontaminated evaluation |
