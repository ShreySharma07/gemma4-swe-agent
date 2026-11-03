# Found, Not Fixed: Counterfactual Diagnostics of Partial Repairs in a Quantized Code Agent

*Draft v2, Nov 2, 2026. Kaggle Gemma 4 paper track (≤ 3,000 words). Revised after Atul's Nov 2 peer review (`atul_paper_peer_review_2026-11-02.md`). Every number is traceable to a row in `docs/experiments.md` and an artifact in `results/runs/`.*

---

## Abstract

Do code agents fail because they cannot find the code that must change? On the competition's stack (Gemma-4-31B W4A16, 32k context, no compaction), a gold-file path appeared in tool calls or outputs for 29 of 30 failures of our best configuration on 60 recent tasks. Resolution falls steeply with change size, which cannot be separated from the number of sites.

Two pre-registered counterfactuals on failed partial patches, pooled across three agent runs, probe what remains:
- **Completion.** Adding the reference edits the agent missed to its own patch rescued 2 of 13 tasks.
- **Site swap.** Placing the agent's version of one necessary reference function into the otherwise gold-patched tree failed the target tests in 43 of 51 hybrids that passed our static compatibility screens (18 of 19 tasks).

A pre-registered structural co-change retriever lost to lexical retrieval on 120 external repositories. These results show the limits of these protocols; residual hybrid incompatibility remains a threat.

---

## 1. Introduction

Repository-level code agents spend much effort on localization: graph tools, retrieval indices and structural summaries that help them find where to edit [RepoGraph; LocAgent; CodeAnchor]. Recent evidence questions how much this matters. Across six localization methods and three models, better localization rarely improves repair, and descriptions of a bug's *cause* help more than its location [From Finding to Fixing].

We ask the follow-up question for a small, quantized, open model under the Kaggle Gemma 4 Developer Agent competition's budget. If the agent sees the right files but still fails, *why*? It may:
1. edit the wrong place;
2. edit some right places but not all (unfinished);
3. make wrong edits at the right places.

We separate these with two counterfactual interventions on the agent's own failed patches, built from leave-one-out *necessity* labels: a reference function is necessary if removing its edits from the gold patch makes the target tests fail.

**Contributions.**
1. A replication of near-saturated exposure on a new model and stack, with failure tracking change size; size and site count are not separable in our data (§3).
2. A completion counterfactual: supplying the missed gold edits rescues 2/13 tasks (§4.1).
3. A site-swap counterfactual: static-screened gold-background hybrids fail the target tests in 43/51 swaps (§4.2). We report how an earlier version mislabeled 25 incompatible hybrids as "clean", and what the corrected screens still miss.
4. A pre-registered negative result for structure-derived co-change recovery against lexical baselines (§5).

Each analysis had a pre-registered prediction and decision rule (§2); we report the predictions that failed.

---

## 2. Setup

**Model and harness.** Gemma-4-31B-it QAT W4A16 is served by vLLM 0.19.1 (TP=4, 4×L4), using the official harness and tool set: `run_command`, `write_file`, `submit_patch`. The context is 32,768 tokens, of which about 24.6k are available to the prompt; no compaction occurs inside a task.

| Agent | Description | New-60 resolved | Public LB |
|---|---|---|---|
| v7 | Single agent; edits via Python snippets in `run_command` (the drift-prone `edit_file` tool removed) | 23/60 | 0.15 |
| v14 | v7 + thinking (4,096 tokens), revised budgets, one-line submit fix | 30/60 | 0.17 |

v7 → v14 is a bundled change: 8 tasks gained and 1 lost, exact McNemar p = 0.039. v16 (bounded tool output) resolves 27/60 (vs v14, p = 0.375) and serves as a secondary check.

**Tasks.** All tasks are verified: the gold patch passes the target tests and an empty patch fails.
- **new-60:** 60 public tasks (50 fastapi, 6 rich, 4 requests).
- **old-54:** an earlier set (39 rich, 9 fastapi, 5 requests, 1 httpx). v7 was run on it five times.

**Labels.**
- *Exposure:* a gold file path, or an intended line range, appears in a tool call or output. This is weaker than verified returned code.
- *Function groups:* each gold hunk mapped to its innermost enclosing function.
- *Necessity:* group *g* is necessary if gold-minus-*g* fails the target tests. 56 of 60 new-60 tasks received labels. Labels are relative to the gold patch; other correct fixes may exist.

**Pre-registration.** Every analysis has a row in `docs/experiments.md` giving prediction and decision rule, committed before the analysis ran, with the result and any deviation added after. For the retrieval tests E0a and E0b, prediction and result share a commit. Other deviations: five completion cases fall back to dropping module-level hunks; the site-swap identity control is a textual re-splice; one site-swap sensitivity flag and the source-level read were added post hoc. The evidence manifest maps every claim to its registration, result commit and aggregate files; raw trajectories and test logs remain restricted competition material. Counterfactual runs used the scorer-faithful CPU sandbox with the competition's test environment.

---

## 3. RQ1: Exposed, but not resolved

**Exposure is near-saturated.** In v14, 59/60 tasks were exposed to a gold file and 55/60 to gold lines. 29 of the 30 failures were exposed to a gold file (Wilson 95% CI 83–99%). Under this proxy, "never looked" is not the main failure.

**Resolve falls with change size.** On the 57 tasks whose gold patch changes non-test Python, split into terciles of changed lines:

| Gold change size | v7 | v14 |
|---|---|---|
| Small (≤ 6 lines) | 12/20 | 15/20 |
| Medium (7–28) | 10/19 | 12/19 |
| Large (> 28) | **0/18** | **2/18** |

In a logistic model with issue length and repository as controls, each log-unit of changed lines lowers the odds of resolution (pooled OR 0.47, p = 0.046; v14 alone OR 0.37, p = 0.025). With 57 tasks and three dominant repositories, this is an association, not a mechanism.

**Site count is not separable from size.** Multi-file fixes resolve less often in raw counts (v14: 3/17 vs 27/43), but 15 of the 17 multi-file tasks are also in the largest tercile. After size control the multi-file OR is 0.32 (95% CI 0.04–2.54); the pre-registered prediction (OR < 0.4, p < 0.05) failed.

With necessity labels, 19 tasks need ≥ 2 necessary groups. They resolve less often in all three arms (v7 4/19 vs 18/38, v14 6/19 vs 23/38, v16 6/19 vs 20/38). After size control the OR is 1.68 (95% CI 0.45–6.33). With few matched sets (8 multi-file, 13 necessity), absence of an adjusted effect is not evidence of no multiplicity effect.

**The bundled change.** v14 produced fewer empty final patches than v7 (5 vs 15 of 60) while resolution rose from 23 to 30; non-empty unresolved patches went from 22 to 25. This does not isolate thinking.

---

## 4. RQ2: Unfinished, or wrong?

By final-patch coverage (function-level matching, heuristic), 15 of v14's 30 failures touched some gold sites but not others. We test what partial coverage means with two interventions on unresolved, non-empty patches, both run through the official verifier. Because v14 alone yields few such patches, both pool three sources:

| Population | v14 new-60 | v7 new-60 | v7 old-54 repeats | Tasks |
|---|---|---|---|---|
| Completion cases | 3 | 7 | 16 | 13 |
| Primary site swaps | 4 | 6 | 41 | 19 |

Repeats of one task are correlated, so we headline task-level counts and use task-clustered intervals.

### 4.1 Completion: add what the agent missed

**Design.** Every unresolved patch that edits ≥ 1 gold function group and misses ≥ 1: 26 cases from 13 tasks. Three variants: (a) the agent's patch alone (control); (b) completion: agent patch plus the gold hunks of the missed groups and gold module-level hunks; (c) missed-only: those gold hunks without the agent's patch. Composition applied cleanly in 13 cases, needed fuzzy matching in 8, and fell back to omitting module-level hunks in 5.

| Outcome | Cases (of 26) | Tasks (of 13) |
|---|---|---|
| (a) agent resolves | 0 | 0 |
| (b) completion resolves | 6 | 2 (Wilson 95% CI 4–42%) |
| (c) missed-only resolves | 3 | 2 |
| Pure incompleteness: (b) passes, (c) fails | 5 | **1** |

Five of the six successes are repeats of one task (rich_3052), which threads a `case_sensitive` parameter through a constructor, a factory classmethod and two consumers. On rich_4070, adding the agent's own edits turns a passing missed-only patch into a failing one.

The pre-registered predictions failed (completion ≥ 50%, pure incompleteness ≥ 35%), and the decision rule fired: we do not describe these failures as unfinished. This tests one frozen-patch completion protocol. It is not an upper bound on completion aids: composition is non-monotonic, and an in-run aid could also change the edits the agent made.

### 4.2 Site swap: keep everything correct except one function

**Design.** Start from the gold-patched tree and replace one necessary gold function group with the agent's version of it. A failing hybrid shows that the transplanted function does not satisfy the target tests in the reference-patch background. That may reflect local behaviour, an interface contract, or residual incompatibility between the agent's function and gold's other sites.

**Validation history.** The first instrument (A12) reported 52/71 failures among "clean" swaps, but an internal audit found that label invalid: it resolved names against all gold files at once, ignored decorators, and kept neither hybrids nor test output. A pre-registered validation (A12v) rebuilt all 128 swaps. A swap passes the static screens only if every name it loads resolves in the same gold file (or is local or builtin), it reads no attribute that only the agent's other edits create, it keeps the gold argument names and default count, and its span includes decorators.

Of 128 swaps, 61 passed; 24 had unresolved names, 15 changed signatures, 5 read agent-added attributes, 17 could not be built and 6 were identical to gold. 25 of the 85 swaps A12 called "clean" failed these screens.

**What the screens miss.** A second reader built synthetic cases that pass as compatible: a changed default value, a name bound only in a nested helper but loaded outside it, and a sync function replaced by an async one. Attribute checks cover only some `self`/`cls` patterns. The source read below found two more: module attributes and field types that gold's other sites change or rely on. How often such cases occur among the 51 is unknown. The identity control re-splices each function span from the same tree and so checks span bounds, not the hybrid builder itself. The gold patch re-ran cleanly on all 29 tasks.

**Result.** Among necessary-group hybrids that passed the screens:

| Measure | Value |
|---|---|
| Fail the target tests | **43/51** |
| Pass | 7 |
| Time out (kept in denominator) | 1 |
| Tasks with ≥ 1 failure | **18 of 19** |
| Task-cluster bootstrap 95% CI | 0.63–1.00 |

A first read of ten random failure outputs found target-behaviour failures in all. A second sample of ten distinct tasks was read at source level against the gold patch. In eight, the agent's own function computes what the failing assertion rejects, e.g. it never sets a redirect history, or handles `TTY_COMPATIBLE=1` but not `0`; one of these also returns `None` where gold's renderer iterates a list. One matches gold's logic and fails only on message wording; one (fastapi_14609) calls a module attribute that gold's edits remove. Excluding failures where gold's other sites may depend on the gold version of the swapped function (a post-hoc check) leaves 42/50.

The registered failure-rate criterion was met. This supports frequent target-test failure in this selected, static-screened hybrid population, not an estimate that 84% of agent edits are intrinsically wrong.

**Taken together.** Supplying the sites the agent missed rarely helped; with gold's other sites in place, the agent's function usually still failed the tests. In 9 of v14's 30 failures the agent edited a different place altogether, a mode we do not analyse further.

---

## 5. RQ3: Can code structure supply the missing co-changes?

Our original thesis held that agents fail to *finish* multi-site changes, and proposed showing an "obligation frontier" after each edit: the callers, constructor consumers, forwarded parameters and overrides that should change too, from a typed repository graph. Change-impact propagation is established [CodePlan]; our proposal was to deliver it passively inside an agent's edit loop.

**Pre-registered recall tests.** Before any agent run, we tested whether the frontier from a simulated first edit (the gold version of one function) contains the other functions that must change. The gate was hit@6 ≥ 0.80 and ≥ 0.20 above the best alternative.

| Test | Label | Typed graph | Untyped graph | BM25 | Same file, random | Provided graph |
|---|---|---|---|---|---|---|
| E0a: 7 tasks, 28 pairs | test-necessary | 0.45 | 0.54 | 0.36 | 0.30 | 0.27 |
| E0b: 120 external repos, 1,192 pairs | gold co-change | **0.19** | 0.24 | **0.42** | 0.21 | n/a |

Both gates failed. On SWE-rebench tasks from 120 repositories (one task each, none ours), lexical similarity to the edited function beat the typed frontier; the task-clustered difference, typed − BM25, was −0.29 to −0.18. Two rules derived from E0a's misses, evaluated post hoc on the same seven tasks, raised the typed frontier to 0.63, still short of the gate.

**What co-changes look like.** Among the 19 new-60 tasks with ≥ 2 necessary groups, 7 thread a parameter or option through several layers, 7 repeat the same edit across sibling call sites, 3 are helper/caller pairs, 1 adds a constructor field and 1 is a repository-wide refactor.

| Relation between necessary pairs | Task-averaged | Pair-level |
|---|---|---|
| Share an edited identifier (keywords excluded) | 89% | 65% |
| Directly linked by a structural edge | 35% | 16% |

Co-changes are more often textually than structurally related, consistent with BM25's advantage; a pre-registered prediction that shared edits alone would explain it failed. Given §4.1, we spent no GPU budget on closed-loop frontier runs, so the in-run effect of structural assistance is untested.

---

## 6. Other interventions

From separate competition submissions, not clean ablations:
- **Optional graph tools** received zero calls; the provided similarity search returned nothing for 461 phrase or bare-name queries.
- **Imitation fine-tuning.** Early imitation adapters collapsed behaviour (one produced 114/114 empty patches). A later on-policy, full-context adapter cut held-out loss by 34–41% on both the training and served bases, but a 95-task paired gate was inconclusive (42 vs 40, p = 0.75).
- **Larger budgets and forced early edits** scored 0.12 and 0.10 on the public leaderboard, below v7's 0.15.
- **Bounded tool output (v16)** cut context overflows from 8 to 5 without a resolution gain.

---

## 7. Related work

**Localization versus repair.** From Finding to Fixing (2610.25826) shows localization aids rarely improve repair and that one gold patch is incomplete location ground truth; it measures reach by inspection, a stronger construct than our exposure proxy. SHERLOC (2606.24820) shows structured diagnoses help weaker repair models.

**Structure for agents.** RepoGraph (2410.14684) and LocAgent (2503.09089) use repository graphs for navigation; CodeAnchor (2606.26979) injects static structure as source anchors; CodePlan (2309.12499) propagates change impact through a dependency graph in a planner. Our negative result concerns one typed index and rule set on gold co-change labels, not graph methods in general.

**Incompleteness and multi-hunk repair.** ChainSWE (2607.02606) studies sequential dependent issues and reports under- and over-editing; our setting is simultaneous multi-site repair within one issue. SiblingRepair (2605.06209) retrieves sibling sites for multi-hunk repair.

**Patch interventions and correctness.** TRIM (2607.18161) minimises agent trajectories and patches; RGFL (2601.18044) includes counterfactual analysis of the localization stage. Ours apply gold-relative completion and replacement to the agent's own failed patches. Test passage is not semantic correctness (2503.15223). Verification-aware training (2610.03984) and reproduction-test diagnosis (SWE-Doctor, 2607.00990) point the same way as our findings.

**Memorization.** Renaming studies (2609.27891) show SWE-bench performance partly reflects repository familiarity. Our tasks are public, a threat we cannot remove.

---

## 8. Limitations

- **One model, one scaffold, Python only.** 50/60 of new-60 are fastapi; results may not transfer to the private repositories of the competition's hidden set.
- **Selected, imbalanced populations.** The interventions cover 13 and 19 tasks, mostly from old-54 v7 repeats; only 3 distinct v14 tasks contribute swaps. Small n gives wide intervals.
- **Gold-relative labels.** Necessity and "the right site" are defined against one reference patch. A failing hybrid shows the agent's function does not work *with this gold design*.
- **Incomplete compatibility screens.** Default values, nested scopes, sync/async kind and many attribute patterns are unchecked (§4.2). Completion runs kept no test output, so their exit-code failures are untyped.
- **Proxies and adjudication.** Exposure counts mentions and intended read ranges. Failure typing relies on test-output tails; only ten hybrids were read at source level, by an unblinded reader and a blinded second reader of the same model family (8/10 identical verdicts, the rest differing only in label).
- **Retrieval tests** use gold first edits as seeds, not real agent edits. E0b's labels are co-change, not necessity.
- **Edit timing.** A shell-aware heuristic re-tagging of v7 and v16 trajectories finds a test run after the final edit in 5–13% of tasks; no earlier verification labels are used.
- **Size is observational.** We do not claim it causes failure.

---

## 9. Conclusion

For a quantized 31B agent under competition constraints, gold files nearly always appear in the agent's trajectory, and failures grow with the size of the required change. When the agent edits a necessary site, its function usually fails the target tests even with gold's other sites in place, and supplying the sites it missed rarely helps. The evidence motivates better diagnosis and verification of the edit itself. It does not establish that localization, in-run completion, context management or graph-assisted policies cannot help.

---

## References

| Ref | arXiv | Title (first author) |
|---|---|---|
| CodePlan | 2309.12499 | CodePlan: Repository-level Coding using LLMs and Planning (Bairi) |
| RepoGraph | 2410.14684 | RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph (Ouyang) |
| LocAgent | 2503.09089 | LocAgent: Graph-Guided LLM Agents for Code Localization (Chen) |
| SWE-bench solved? | 2503.15223 | Are "Solved Issues" in SWE-bench Really Solved Correctly? An Empirical Study (Wang) |
| SWE-rebench | 2505.20411 | SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents (Badertdinov) |
| RGFL | 2601.18044 | RGFL: Reasoning Guided Fault Localization for Automated Program Repair Using Large Language Models (Sepidband) |
| SiblingRepair | 2605.06209 | SiblingRepair: Sibling-Based Multi-Hunk Repair with Large Language Models (Liu) |
| SWE-Explore | 2606.07297 | SWE-Explore: Benchmarking How Coding Agents Explore Repositories (Zhang) |
| SHERLOC | 2606.24820 | SHERLOC: Structured Diagnostic Localization for Code Repair Agents (Tamoyan) |
| CodeAnchor | 2606.26979 | How Much Static Structure Do Code Agents Need? A Study of Deterministic Anchoring (Lin) |
| SWE-Doctor | 2607.00990 | SWE-Doctor: Guiding Software Engineering Agents with Runtime Diagnosis from Multi-Faceted Bug Reproduction Tests (Guo) |
| ChainSWE | 2607.02606 | ChainSWE: Benchmarking Coding Agents on Multi-Bug Software Maintenance (Jin) |
| TRIM | 2607.18161 | TRIM: Reducing AI-Generated CodeSlop via Agent Trajectory Minimization (Mathai) |
| Schrödinger's Repo | 2609.27891 | Schrödinger's Code Repository: Have LLMs Learned SWE-bench or Memorized It? (Chen) |
| Teaching Agents | 2610.03984 | Teaching Agents to Code Reliably (Mohsin) |
| From Finding to Fixing | 2610.25826 | From Finding to Fixing: How Useful is Fault Localization for Agentic Program Repair? (Shi) |
