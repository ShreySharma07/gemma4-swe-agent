# Paper outline: Found, Not Finished

**Status:** Oct 23, 2026. This outline replaces `paper_outline.md` (EBST). The thesis's method claim (the obligation frontier) failed both pre-registered gates, E0a and E0b, so this paper is the planned fallback in §8 of `thesis_found_not_finished.md`. It is an analysis paper with one pre-registered negative result.

> **A1 result (Oct 23):** the pre-registered prediction failed. After controlling for gold patch size, "multi-file" is no longer significant (pooled OR 0.32, CI 0.04–2.54, p = 0.21); patch size is (OR 0.47 per log line, p = 0.046). 15/17 multi-file new-60 tasks are in the largest size tercile, so the two are nearly collinear. §3 now frames the gap as **large changes**, with multi-site as one aspect that cannot be separated in these data. A4 (completion counterfactual) is now the analysis that decides whether "not finished" is the right word.

> **A3/A4 result (Oct 23):** the pre-registered predictions failed and the decision rule fired.
> - **Completion rarely rescues a partial patch.** Adding the gold edits the agent missed resolves 6/26 partial patches (23%), on just 2 of 13 distinct tasks.
> - **Pure incompleteness is rare.** Only 1/13 tasks fits it (rich_3052). In most partial patches the agent's *own* edits are wrong as well.
> - **The paper therefore drops "not finished" as the explanation.** Working title becomes ***Found, Not Fixed***: localisation is saturated, failure grows with change size, and neither completion nor structure-derived hints account for it.
> - **Multi-site fixes are common.** 19/60 new-60 tasks need ≥2 test-necessary function groups.
> - **Next:** §3 and the claims ledger must be rewritten around "wrong edits at the right site" (cf. 2610.25826: a cause description beats location).

**Constraints:**
- Kaggle paper track, ≤3,000 words.
- Experiment freeze Nov 5; due Nov 12.
- Every number below names its source. Rows marked **NEEDED** link to the analysis list in §5.

---

## 1. Working title and claim

**Title:** *Found, Not Finished: a small code agent locates the fix but cannot complete multi-site changes, and code structure is not what it is missing*

**One-sentence claim.** On the competition's own stack (Gemma-4-31B W4A16, 32k context, no compaction), the agent almost always reaches the code it must change. It fails because it changes only part of what must change together. Neither more reasoning, more samples, nor structure-derived hints about what else must change close that gap. In our tests, lexical similarity predicts co-change better than any graph we built or were given.

**Why it is worth a paper.** The code-graph literature spends structure on localisation. We show, on a small open model and with pre-registered tests, that:
1. localisation is saturated;
2. the residual failure is completion;
3. the relations that bind co-changes are mostly not call edges, and on external code they are better recovered lexically than structurally.

Points 1 and 3 replicate and extend recent work (2610.25826, 2607.02606). Point 2, measured against test-necessity labels, and the negative structural result are ours.

---

## 2. Research questions

| RQ | Question | Primary evidence |
|---|---|---|
| RQ1 | Where in the find → edit → complete → resolve funnel does a small agent fail? | v7 / v14 on new-60, v7 ×5 on old-54 |
| RQ2 | Is the gap about *completion* or just harder tasks? | Strata by gold files / test-necessary groups, controlled for patch size (**NEEDED** A1) |
| RQ3 | Do more reasoning or more samples close it? | v7 → v14 (thinking), pass@5 by stratum |
| RQ4 | What relates the program elements that must change together, and can a graph supply them? | LOO necessity typology, E0a, E0b (pre-registered) |

---

## 3. Sections, word budget, and what each needs

### §1 Introduction (300 words)
- Hook: the agent saw a gold file in 29/30 of its failed tasks, yet single-file tasks resolve at 63% and multi-file tasks at 18%. *Have: v14 new-60.*
- Contributions (4 bullets):
  1. the find/finish funnel;
  2. test-necessity labels;
  3. the obligation typology;
  4. a pre-registered negative result for structural frontiers.
- Name CodePlan (2309.12499), From Finding to Fixing (2610.25826) and ChainSWE (2607.02606) here, not in a footnote.

### §2 Setup (250 words)

**Model and serving:** Gemma-4-31B-it QAT W4A16, vLLM 0.19.1 TP=4 on 4×L4, the scorer harness (`run_official_l4.py` / evalkit), 32k context, no compaction.

**Agents.** Each agent is one config; we describe what it does, not the ablation history:

| Agent | What it is | LB |
|---|---|---|
| v7 | `edit_file` removed; edits through Python in `run_command` | 0.15 |
| v14 | v7 plus thinking and budget changes | 0.17 |

**Task sets:**

| Set | Valid tasks | Repos |
|---|---|---|
| old-54 | 54 | 39 rich, 9 fastapi, 5 requests, 1 httpx |
| new-60 | 60 | 50 fastapi, 6 rich, 4 requests |

All are gold-verified (gold passes, empty fails). Public repos are a limitation, stated in §7.

**Labels:**
- gold files/functions from trace_analyzer;
- **test-necessary function groups** from the leave-one-group-out runs of the execution instrument (`kaggle/exec_graph`).

### §3 RQ1–RQ3: found, not finished (850 words). The core

**Fig. 1: funnel**, per stratum (single- vs multi-site): saw gold file → saw gold lines → edited a gold file → edited *all* necessary sites → resolved.
- Have: v14/v7 file-level funnel (`analysis/find_vs_finish.py`).
- **NEEDED:** line- and function-level exposure per stratum (A2) and necessity-level completion on new-60 (A3).

**Table 1: resolve by stratum:**

| Stratum | v7 | v14 | Source / status |
|---|---|---|---|
| 1 gold file | 22/43 | 27/43 | have |
| ≥2 gold files | 1/17 | 3/17 | have (Fisher p = 0.001 / 0.003) |
| 1 test-necessary group (old-54, ×5) | 66/185 runs, 17/38 tasks | — | have |
| ≥2 test-necessary groups (old-54, ×5) | 0/35 runs, 0/7 tasks | — | have (p = 0.034) |
| ≥2 test-necessary groups, new-60 | ? | ? | **NEEDED** A3 |

**Failure decomposition of unresolved tasks:**

| Failure | v14 tasks (of 30) | Status |
|---|---|---|
| Partial coverage | 15 | have |
| Saw gold but edited elsewhere | 9 | have |
| Empty patch | 5 | have |
| Complete but wrong | 1 | have |

Two refinements are **NEEDED**: whether the agent ever looked at the missed sites (A6), and whether it submitted with budget left (A7).

**The completion counterfactual (the strongest single analysis).** For partial patches, add the gold edits of only the *missed* necessary groups to the agent's own patch and re-run the fail-to-pass tests. If most then pass, the failure is purely incompleteness. **NEEDED** A4.

**RQ2 confound control.** Multi-site tasks are bigger and may simply be harder. Model resolve against stratum, controlling for gold patch size (added/removed lines, hunks), issue length and repo, with a logistic regression and a repo-cluster bootstrap. **NEEDED** A1.

**RQ3:**
- **Thinking (v7 → v14):** +5 single-file, +2 multi-file. *Have.*
- **Sampling:** pass@5 on old-54 by stratum (0/7 multi vs 17/38 single). *Have.* Extend with the v16 run if it completes (A9).

### §4 RQ4: what binds co-changes? (650 words)

**Typology (Table 2)** from the 7 old-54 tasks with ≥2 test-necessary groups:
- parameter plumbing and field def→use;
- constructor field → consumer;
- override family;
- same-class siblings;
- helper/callee;
- independent issue facets.

*Have.* Extend to new-60 (A3, A5).

**Where the pairs sit:**
- same file 26/28;
- executed-call 1-hop 4/28;
- provided graph 1-hop 4/28, 2-hop 20/28 (at a median of 38 nodes).

*Have:* `analysis/obligation_pilot.py`.

**Pre-registered recall tests (Table 3):** hit@6 of the first-edit frontier.

| Test | Label | Typed AST | Untyped AST | BM25 | Same-file random | Executed calls | Provided |
|---|---|---|---|---|---|---|---|
| E0a, 7 tasks | test-necessary | 0.45 | 0.54 | 0.36 | 0.30 | 0.29 | 0.27 |
| E0b, 120 external repos | gold co-change | **0.19** | 0.24 | **0.42** | 0.21 | n/a | n/a |

- Both gates failed (bar 0.80).
- Typing the graph by edge kind hurts in both tests.
- Lexical similarity wins on external code.

**NEEDED:**
- A8: why BM25 wins (shared identifiers vs repeated edit pattern);
- E0b split by relation and by single- vs multi-file;
- E0a re-scored with v2 rules, labelled post hoc.

Report this as a **negative result**, with protocol, predictions and hashes. Note that the structural graph beat BM25 on the necessity label (CI [+0.04, +0.24]) but on only 7 tasks.

### §5 What does not help, briefly (250 words)

Measured negatives from the competition. Each is one or two sentences with its number:
- **Optional graph tools are ignored:** 0 graph-tool calls in v5. Exact-name graph search returns nothing for phrases or bare names (0/461 calls).
- **Imitation SFT collapses behaviour while loss falls:** Step D resolved 0/114; S3b resolved 9/60 vs v7's 23/60.
- **More calls do not convert into fixes:** v9, with 70 calls, scored LB 0.12.
- **Forcing an early edit or test hurts:** v10b scored LB 0.10.

### §6 Related work (300 words)

| Thread | Papers |
|---|---|
| Graph-for-agents localisation | RepoGraph 2410.14684; LocAgent 2503.09089; CodeAnchor 2606.26979; LARGER 2605.16352 |
| Localisation-is-not-the-bottleneck | 2610.25826; SWE-Explore 2606.07297; SHERLOC 2606.24820 |
| Incompleteness | ChainSWE 2607.02606; 2609.34317; multi-hunk repair, e.g. SiblingRepair 2605.06209 |
| Change impact | CodePlan 2309.12499; CoEdPilot 2408.01733 |
| Necessity / minimal patches | TRIM 2607.18161 |
| Memorisation | SchrödingerRepo 2609.27891 (motivates §7) |

All IDs are checked: `surveys_2026-10-22/`.

### §7 Limitations and outlook (250 words)
- Python only.
- Three public repos, with fastapi at 50/60 of new-60. That is memorisation risk and repo concentration, so repo-cluster CIs are reported.
- LB link is indirect: the hidden set is private and small, and the LB noise is about ±0.03.
- Small n for necessity labels.
- Gold co-change ≠ necessity on E0b.
- Outlook: lexical co-change hints as the next intervention (§6, A10).

**Total ≈ 2,850 words.**

---

## 4. Claims ledger

The paper may state only claims marked "supported".

| Claim | Status | Evidence / needed |
|---|---|---|
| The agent sees the gold file in nearly all failures | Supported | v14 29/30 |
| Multi-site tasks resolve far less often | **Raw gap supported; not separable from patch size** (A1) | Raw Fisher p = 0.003 / 0.001; size-controlled OR 0.32 [0.04, 2.54]. Say "large, multi-site changes", never "because multi-site" |
| Larger gold patches resolve less often | Supported (A1) | Pooled OR 0.47 per log line, p = 0.046 |
| Failures are incompleteness, not wrong diagnosis | **Rejected** (A4: completion 6/26 cases, 2/13 tasks) | A4 (completion counterfactual) decides |
| Thinking does not close the gap | Weakly supported | +5 vs +2, bundled changes; say "did not close", not "cannot" |
| Necessary co-changes are mostly not call edges | Supported on 7 tasks | Extend with A3/A5 |
| Structural frontiers do not recover co-changes better than lexical ones | Supported (E0b, pre-registered) | Caveat: co-change label |
| Typing edges by change kind helps | **Rejected** | E0a, E0b |
| Results transfer to private repos | Not shown | Limitation only |

---

## 5. Analyses still to run

Ordered by priority. CPU = local or Kaggle CPU, no GPU quota. "Local" = runnable now on files in `data/runs/`.

| ID | Analysis | Why it matters | Data / instrument | Compute | Priority |
|---|---|---|---|---|---|
| **A1 ✓ done Oct 23** | Confound control: resolve ~ multi-site + gold size (lines, hunks, functions) + issue length + repo, as a logistic regression with a repo-cluster bootstrap; also matched pairs of single/multi tasks of similar size | Without it, "multi-site is the problem" could just be "big patches are hard" | per_task.jsonl (local); gold patches for new-60 (Kaggle task dataset, download once) | CPU, local, <1 h | **P0** |
| **A3 ✓ done Oct 23** | Run the exec-graph instrument (empty / gold / LOO) on **new-60** with the test overlays (`evalkit/overlays.py`) | Test-necessity labels for 60 more tasks; multi-necessary n goes from 7 to ~20; Table 1 last row | `kaggle/exec_graph/build_exec_graphs.py`, extended to the overlay env + new-60 list | Kaggle CPU, ~2–4 h | **P0** |
| **A4 ✓ done Oct 23** | Completion counterfactual: for every partial patch (v7, v14, rs ×5), apply agent patch + gold hunks of only the *missed* necessary groups and run fail-to-pass | Turns "partial coverage" into "the agent's part was right; it just didn't finish" (or refutes it) | New mode `agent+missing` in the instrument; agent patches are local | Kaggle CPU, ~1–2 h (same kernel as A3) | **P0** |
| **A2** | Funnel at line and function level per stratum (seen lines, seen gold function, edited each necessary group) | Fig. 1 | trace_analyzer fields + necessity labels | CPU, local | P1 |
| **A6** | For each missed necessary site: did the agent ever view it (grep/sed/cat hits in traces) before submitting? | Splits completion failures into "never looked" (search) vs "looked, didn't edit" (conversion) | Traces in run tgz (local, extract into `data/runs/<id>/`) | CPU, local, ~2 h | P1 |
| **A7** | Budget at submit for partial patches: calls left, prompt tokens, time; submitted explicitly vs ran out | Premature submit vs budget exhaustion | per_task + traces (local) | CPU, local | P1 |
| **A5** | Typology on new-60's multi-necessary tasks (manual reading of gold diffs plus the AST relations from `oblig.py`) | Table 2 with n ≈ 20 instead of 7 | A3 output | CPU, ~2 h manual | P1 |
| **A8** | Why BM25 wins on E0b: share of co-changed pairs that share identifiers or an edited token; repeated-edit-pattern detection; E0b split by same-/cross-file | Turns a negative result into an explanation; seeds the outlook | `results/runs/e0b/e0b_pairs.json` + cached snapshots (scratchpad) | CPU, local, ~1 h | P1 |
| A8b | E0a re-scored with v2 rules, **labelled post hoc** | Completeness of the negative-result section | Existing harness | CPU, minutes | P2 |
| A9 | Add v16 new-60 (if its eval completes) as a third agent in Table 1 | One more config; checks the gap is stable | v16 run (currently errored / re-queued on Atul's account) | GPU already scheduled; no new spend | P2 |
| A10 | *Optional* GPU intervention: v14 + lexical co-change hint (after an edit, grep the changed identifiers and list hits) on the multi-site stratum only (17 new-60 + 7 old-54 tasks), paired with existing v14 | Only intervention suggested by E0b. Gives the outlook one data point, but must be pre-registered and is underpowered (n ≈ 24) | New prompt variant; validate with `kaggle/validate` | ~4 quota-h | P3, only if A1/A4 finish by Oct 30 |

**Not needed** (decided): E1–E4 closed-loop frontier runs (gates failed); renamed-repo experiments (no method left to test the interaction on); new LB submissions for the paper.

---

## 6. Timeline to freeze (Nov 5)

| Date | Work |
|---|---|
| Oct 24 | A1 (local); fetch new-60 gold patches; extend the exec-graph kernel for overlays + `agent+missing` mode |
| Oct 25–26 | A3 + A4 on Kaggle CPU; A8, A8b, A7 locally |
| Oct 27–28 | A2, A6, A5 |
| Oct 29 | Claims ledger review: drop or soften every claim still "plausible" |
| Oct 30 | Go/no-go on A10 |
| Oct 31–Nov 5 | Figures (funnel, strata table, recall curves) and freeze |
| Nov 5–12 | Write and revise; submit |

## 7. Figures and tables

| Item | Content | Data |
|---|---|---|
| Fig. 1 | Funnel by stratum | A2, A3 |
| Table 1 | Resolve by stratum × agent, with CIs | have + A1, A3 |
| Fig. 2 | Completion counterfactual: share of partial patches that pass once the missed sites are added | A4 |
| Table 2 | Obligation typology | have + A5 |
| Table 3 / Fig. 3 | Pre-registered recall tests E0a/E0b, recall@k curves per arm | have |
