# Paper outline: Found, Not Fixed

**Status:** Oct 24, 2026. This outline replaces `paper_outline.md` (EBST). It is an analysis paper with pre-registered negative results. The thesis's method claim, the obligation frontier, failed both pre-registered gates (E0a, E0b). The fallback framing "found, not finished" was then tested and rejected by its own pre-registered checks (A1, A4). This version is rewritten around what survived.

**Review edits (Oct 24):**
- C1 narrowed to *sees* (exposure). 9 of the 30 failures saw a gold file and edited elsewhere, which is a separate finding.
- C6 narrowed: A4 bounds patching the agent's *final* patch, not every in-run completion aid.
- A12 hardened against wrong-vs-incompatible (pre-registered in `docs/experiments.md`).
- Wilson 95% CIs added.
- Pre-registration commit hashes added (§2).
- The thinking result (v7 → v14, p = 0.039) is now stated in §3.5.
- Deadline confirmed as Nov 12 (§6).

**What changed on Oct 23–24:**
- **A1:** the multi-file gap is not separable from patch size. Size carries it: OR 0.47 per log line (p = 0.046). For multi-file the OR is 0.32 with CI 0.04–2.54.
- **A3:** 19/60 new-60 tasks need ≥2 test-necessary function groups.
- **A4:** adding the gold edits the agent missed rescues 6/26 partial patches, on 2/13 distinct tasks. Pure incompleteness fits 1/13 tasks.

**Constraints:**
- Kaggle paper track, ≤3,000 words.
- Experiment freeze Nov 5; due **Nov 12** (confirmed by Shrey, Oct 24).
- **All remaining analyses run on Kaggle CPU kernels, not locally** (Shrey, Oct 23).
- Every number names its source. **NEEDED** rows link to §5.

---

## 1. Working title and claim

**Title:** *Found, Not Fixed: a small code agent sees the code it must change, but its edits are wrong, increasingly so as the change grows. Completing its patch or giving it code structure does not fix that.*

**One-sentence claim.** On the competition's own stack (Gemma-4-31B W4A16, 32k context, no compaction), the agent almost always *sees* the code that must change (gold-file exposure 98%), yet a third of its failures edit somewhere else. Its resolve rate falls with the size of the required change. When it edits part of a multi-site fix, adding the missing gold edits to its final patch rarely rescues it: 2/13 tasks (95% CI 4–42%). So the failure is mostly *wrong edits at the right place*, not unfinished ones. Structure-derived hints about co-changes do no better than lexical similarity.

**Why it is worth a paper.** On a small open model, with pre-registered tests that were allowed to fail (and several did):
1. **Exposure is saturated, editing location is not.** The agent sees a gold file in 29/30 failures (replicating 2610.25826 on a new model and stack), but 9 of those failures edit elsewhere.
2. **Failure grows with change size, not with the number of sites per se** (A1).
3. **An oracle completion bound** (A4, new). Adding the correct missing edits to the agent's final partial multi-site patch rescues 2/13 tasks (CI 4–42%). This bounds *post-hoc completion of a frozen patch*. An in-run aid could also change the edits the agent did make, so it is not a bound on every aid. It is still strong evidence that missing sites are not the main failure.
4. **A pre-registered negative result for structural co-change recovery** (E0a/E0b). Lexical similarity beats typed graph frontiers on 120 external repos.

---

## 2. Research questions

| RQ | Question | Evidence | Status |
|---|---|---|---|
| RQ1 | Does the agent reach the code it must change? | v14 exposure; failure decomposition | have |
| RQ2 | What predicts failure: number of sites or size of change? | A1 (logit + size-matched) | have |
| RQ3 | Are failures unfinished or wrong? | A4 completion counterfactual; A12 site-swap | A4 have, **A12 NEEDED** |
| RQ4 | Do more reasoning or more samples help, and where? | v7 → v14 by size tercile; pass@5 | have; A13 extends |
| RQ5 | Can code structure supply the co-changes? | Typology, E0a, E0b | have; A5, A8 extend |

---

## 3. Sections, word budget, and what each needs

### §1 Introduction (300 words)
- **Hook.** In v14, our best leaderboard config, the agent saw a gold file in 29/30 of its failed tasks. Single-file tasks resolve at 63% and multi-file at 18%. Yet when we hand it the edits it missed, 2/13 tasks pass.
- **Contributions** (4 bullets, as in §1 of this outline):
  1. saturated localisation;
  2. failure tracks change size;
  3. the oracle completion bound;
  4. the structural negative.
- **Name early:** CodePlan (2309.12499), From Finding to Fixing (2610.25826), ChainSWE (2607.02606).

### §2 Setup (250 words)

**Model and serving:** Gemma-4-31B-it QAT W4A16, vLLM 0.19.1 TP=4 on 4×L4, the scorer harness (`kaggle/eval_l4/evalkit/`), 32k context, no compaction.

**Agents.** Each agent is one config; we describe what it does, not the ablation history:

| Agent | What it is | LB |
|---|---|---|
| v7 | `edit_file` removed; edits through Python in `run_command` | 0.15 |
| v14 | v7 plus thinking (4,096), budgets (8 min / 28 calls / 240 s) and v13's one-line `submit_patch` prompt fix | 0.17 |

**Task sets:**

| Set | Valid tasks | Repos |
|---|---|---|
| old-54 | 54 | 39 rich, 9 fastapi, 5 requests, 1 httpx |
| new-60 | 60 | 50 fastapi, 6 rich, 4 requests |

All are gold-verified (gold passes, empty fails). Public repos are a limitation (§7).

**Labels:**
- **Gold files/functions** from trace_analyzer.
- **Test-necessary function groups.** Each gold hunk is mapped to its innermost enclosing function. A group is necessary if gold minus that group fails the fail-to-pass tests. Sources: old-54 from the exec-graph instrument, new-60 from A3.

**Pre-registration.** Every analysis in §3–§4 has a pre-registered row in `docs/experiments.md` with a prediction, written before running. We report which predictions failed. Git evidence:

| Analysis | Pre-registration commit | Result commit | Verifiable from git? |
|---|---|---|---|
| A1 | `e7f559f` | `9d99618` | yes |
| A3/A4 | `62e4068` | `7beef0e` | yes |
| E0a | `e8e9db3` (same commit as the result) | `e8e9db3` | **no**: disclose |
| E0b | `672bc30` (same commit as the result) | `672bc30` | **no**: disclose |
| A12, A13 | `92dabb6` (committed before any kernel was built) | — | yes, if results land in a later commit |

For E0a/E0b, state plainly that the prediction and result were committed together, so the order rests on our word. Do not call them "pre-registered" without that caveat.

### §3 Found, not fixed (850 words). The core

**§3.1 Found (RQ1, ~150 words).**

| Metric (v14, new-60) | Value | Source |
|---|---|---|
| Gold-file exposure, all tasks | 98.3% | v14 report |
| Gold-line exposure, all tasks | 91.7% | v14 report |
| Failed tasks that saw a gold file | 29/30 (CI 83–99%) | `find_vs_finish.py` |

- **Replication:** this matches 2610.25826 (88.6–99.4%). State it as a replication, not a finding.
- **Wording:** "sees", not "reaches" or "localises". Exposure means a gold file appeared in a tool output, not that the agent acted on it.
- **How the 30 failures break down:**

| Failure | Tasks |
|---|---|
| Edited some gold sites, missed others | 15 |
| Saw a gold file, edited elsewhere | 9 |
| Empty patch | 5 |
| Covered all gold sites, still wrong | 1 |

These are heuristic labels; §3.3 tests what they mean. **The 9 "saw a gold file, edited elsewhere" cases are a finding in their own right:** about a third of failures are wrong-location edits despite exposure. Report them beside the "wrong at the right site" cases (§3.4), not folded into "found".

**§3.2 Not fixed, and worse for larger changes (RQ2, ~250 words).**

**Table 1, resolve by stratum** (new-60; overflows count as failures). The file-count rows use all 60 tasks. The size rows use the 57 tasks with non-test `.py` gold, with terciles at ≤6 and ≤28 changed lines:

| Stratum | v7 | v14 | Source |
|---|---|---|---|
| 1 gold file | 22/43 | 27/43 | have |
| ≥2 gold files | 1/17 | 3/17 | have (raw Fisher p = 0.001 / 0.003) |
| Small change (≤6 changed lines) | 12/20 | 15/20 | A1 |
| Mid (7–28 lines) | 10/19 | 12/19 | A1 (single + multi) |
| Large (>28 lines) | 0/18 | 2/18 | A1 (15 of the 18 are multi-file) |
| ≥2 test-necessary groups (19 tasks) | **NEEDED** | **NEEDED** | A13 (from A3 labels) |

**Size-controlled models (A1)**, each with controls for issue length and repo:

| Model | Odds ratio | 95% CI | p |
|---|---|---|---|
| log(changed lines), pooled v7 + v14 | 0.47 per log line | — | 0.046 |
| log(changed lines), v14 alone | 0.37 | — | 0.025 |
| Multi-file, after size | 0.32 | 0.04–2.54 | 0.21 |

- **Size-matched comparisons** (same repo, size caliper): 5–10 sets, none significant.
- **How to word it:**
  - Say: *"resolve falls with the size of the required change. Multi-site fixes are mostly large ones, and in these data the two cannot be separated."*
  - Do not say "because multi-site". The pre-registered prediction (multi-file OR < 0.4, p < 0.05) failed.

**§3.3 Not unfinished: the oracle completion bound (RQ3, ~300 words). The paper's central new result.**

**Design (A4, pre-registered).** Take every unresolved agent patch that edits ≥1 gold function group and misses ≥1: 26 cases from 13 distinct tasks, across v7, v14 and v7 ×5. Run three variants:
- **(a) agent alone.** Control; must fail.
- **(b) agent + gold hunks of the missed groups**, plus gold module-level hunks: the completion.
- **(c) the missed gold hunks alone.** Control: was the agent's own part needed?

**Fig. 2 / Table 2, results:**

| Outcome | Cases (of 26) | Distinct tasks (of 13) |
|---|---|---|
| (a) agent alone resolves | 0 | 0 |
| (b) completion resolves | 6 (23%; CI 11–42%) | 2 (CI 4–42%) |
| (c) missed-only resolves | 3 | 2 |
| **Pure incompleteness**: (b) passes and (c) fails | 5 (19%; CI 9–38%) | **1** (rich_3052, all 5 reps; CI 1–33%) |

- **Restricted** to cases that missed a test-necessary group: completion 6/20, pure 5/20.
- **On 2 tasks (rich_3782, rich_4070) the agent's own edits were not needed:** the missed gold hunks pass alone.
- **Reading.** Supplying the *correct* missing edits to the agent's final patch is an upper bound on **post-hoc completion** of that patch: 2/13 tasks. It is **not** a bound on in-run aids (a structural frontier, a lexical co-change hint, a "check callers" prompt), because those can also change the edits the agent made. What it does show is that coverage is rarely what separates these patches from passing. The content of the edits the agent did make is the binding failure.
- **Cases are correlated:** 15 of the 26 come from v7's 5 repeats. Headline the task-level numbers.
- **Predictions that failed:** completion ≥ 50%, pure ≥ 35%. The pre-registered decision rule fired (completion < 30%).

**§3.4 Wrong at the right site? (RQ3 continued, ~100 words). NEEDED: A12** (pre-registered in `docs/experiments.md`).
- **Design:** the mirror counterfactual. Take the gold patch, but replace one group's gold hunks with the agent's own edits to that group.
- **The trap:** an agent edit that is right *for its own other edits* (a renamed parameter, a helper added elsewhere) can fail next to gold's other hunks without being wrong. So the swap separates three outcomes:
  - **conflict:** the swap does not apply or does not import;
  - **coupled:** the agent's edit references names that exist only in its own edits outside the group;
  - **clean:** neither of those.
- **Primary claim only on clean swaps** of test-necessary groups. Single-group tasks are the cleanest stratum.
- **Reading:** a clean swap that fails means the agent's edit at that site is wrong even with everything else correct.
- **What it gives:** per-site "wrong at the right site" rates, which test the paper's claim directly instead of inferring it from A4.

**§3.5 Reasoning and sampling (RQ4, ~80 words).**

**The one significant positive result:**
- **Overall:** v7 → v14 (thinking on) on new-60 went **23 → 30/60** (22 both, 8 v14-only, 1 v7-only; exact McNemar **p = 0.039**).
- **Empty patches:** fell from 25% to 8%.
- **Wrong fixes:** stayed at 16 in both.

So reasoning helped the agent *commit* to an edit, not get the edit right, consistent with §3.3. Two counterpoints for the same section:
- **v15b**, a public 0.24-notebook prompt on v14's tools: LB 0.08.
- **v16**, bounded tool output: 27 vs 30, p = 0.375. Overflows fell from 8 to 5, but no extra solves.

v7 → v14 by size tercile (A1 strata):

| Stratum | v7 → v14 |
|---|---|
| Small | 12 → 15 |
| Mid | 10 → 12 |
| Large | 0 → 2 |

- **Thinking, prompt and budget changes are bundled,** so say "did not close", not "cannot".
- **pass@5 on old-54:** 0/7 tasks with ≥2 necessary groups, but their size-matched single-group controls are also ~0 (A1 matched: 0.00 vs 0.01). So sampling does not reach large changes either.
- **A13** re-tabulates both on the new-60 necessity labels.

### §4 Can structure supply the co-changes? (RQ5, 600 words)

**Typology (Table 3)** from the 7 old-54 tasks with ≥2 test-necessary groups:
- parameter plumbing and field def→use;
- constructor field → consumer;
- override family;
- same-class siblings;
- helper/callee;
- independent issue facets.

*Have.* Extend to new-60's 19 tasks (A5).

**Where the pairs sit:**
- same file 26/28;
- executed-call 1-hop 4/28;
- provided graph 1-hop 4/28, 2-hop 20/28 (at a median of 38 nodes).

*Have:* `analysis/obligation_pilot.py`.

**Pre-registered recall tests (Table 4):** hit@6 of the first-edit frontier.

| Test | Label | Typed AST | Untyped AST | BM25 | Same-file random | Executed calls | Provided |
|---|---|---|---|---|---|---|---|
| E0a, 7 tasks | test-necessary | 0.45 | 0.54 | 0.36 | 0.30 | 0.29 | 0.27 |
| E0b, 120 external repos | gold co-change | **0.19** | 0.24 | **0.42** | 0.21 | n/a | n/a |

- Both gates failed (bar 0.80).
- Typing the graph by edge kind hurts in both tests.
- Lexical similarity wins on external code.

**Tie to §3.3.** Post-hoc completion with perfect co-change information rescues 2/13 tasks. Recall is therefore unlikely to be the main lever, which is why no closed-loop frontier run was justified. In-run effects remain untested. **NEEDED:** A8 (why BM25 wins), A8b (E0a under v2 rules, post hoc).

### §5 What does not help, briefly (250 words)

Measured negatives from the competition, each in one or two sentences with its number:
- **Optional graph tools are ignored:** 0 graph-tool calls in v5. Exact-name graph search returns nothing for phrases or bare names (0/461 calls).
- **Imitation SFT collapses behaviour while loss falls:** Step D resolved 0/114; S3b resolved 9/60 vs v7's 23/60.
- **More calls do not convert into fixes:** v9 with 70 calls scored LB 0.12.
- **Forcing an early edit or test hurts:** v10b scored LB 0.10.

### §6 Related work (300 words)

| Thread | Papers |
|---|---|
| Graph-for-agents localisation | RepoGraph 2410.14684; LocAgent 2503.09089; CodeAnchor 2606.26979; LARGER 2605.16352 |
| Localisation-is-not-the-bottleneck | 2610.25826; SWE-Explore 2606.07297; SHERLOC 2606.24820 |
| Incompleteness | ChainSWE 2607.02606 (we bound its "undershoot" mechanism: real but rarely sufficient); 2609.34317; multi-hunk repair, e.g. SiblingRepair 2605.06209 |
| Change impact | CodePlan 2309.12499; CoEdPilot 2408.01733 |
| Necessity / minimal patches | TRIM 2607.18161 |
| Memorisation | SchrödingerRepo 2609.27891 |

All IDs are checked in `surveys_2026-10-22/`.

### §7 Limitations and outlook (250 words)
- Python only.
- Three public repos, with fastapi at 50/60 of new-60. That is memorisation risk and repo concentration.
- **Small n:** 13 distinct A4 tasks; 19 multi-necessary tasks.
- Necessity is relative to the gold patch; alternative valid fixes exist (2610.25826: 11.6% of pairs).
- Gold co-change ≠ necessity on E0b.
- LB link is indirect.
- **Outlook:** the lever is edit correctness at the site (cause understanding, verification), not finding or completing. That points to cause descriptions (2610.25826) and test-grounded diagnosis (SWE-Doctor 2607.00990).

**Total ≈ 2,800 words.**

---

## 4. Claims ledger

Only rows marked **Supported** may be stated as findings. "Weak" rows need hedged wording, as given.

| # | Claim | Status | Evidence | Wording / condition |
|---|---|---|---|---|
| C1 | The agent **sees** the code it must change | **Supported** (replication) | v14: 98.3% gold-file and 91.7% gold-line exposure; 29/30 failures saw a gold file (CI 83–99%) | "sees", not "reaches" or "localises"; cite 2610.25826 as the original finding |
| C1b | Exposure does not mean editing the right place | **Supported** (descriptive) | 9/30 v14 failures saw a gold file and edited elsewhere | Heuristic label; report beside C7 |
| C2 | Resolve falls with the size of the required change | **Supported** | A1: OR 0.47 per log line, p = 0.046 (pooled); v14 alone OR 0.37, p = 0.025 | Correlational; controls for issue length and repo only |
| C3 | Multi-site fixes resolve less often *because* they are multi-site | **Not supported** | A1: OR 0.32 [0.04, 2.54] after size; matched gaps n.s.; pre-registered prediction failed | Say "multi-site fixes are mostly large; the two cannot be separated here" |
| C4 | Multi-site fixes are common | **Supported** | A3: 19/60 new-60 tasks need ≥2 test-necessary groups; 7/54 old-54 | Relative to gold patches |
| C5 | Failures are mostly unfinished patches | **Rejected** | A4: completion rescues 6/26 cases, 2/13 tasks; pure incompleteness 1/13 | State as a rejected pre-registered hypothesis |
| C6 | Adding the correct missing edits to the agent's final patch rescues 2/13 tasks (the oracle completion bound) | **Supported** | A4 with controls (agent-alone 0/26); task CI 4–42% | Bounds **post-hoc completion of a frozen patch** on these 13 tasks. **Not** a bound on in-run aids, which can change the agent's own edits too |
| C7 | The agent's edits at the right sites are wrong | **Plausible** | Implied by C5/C6 | **A12 decides, on clean swaps only** (conflict/coupled swaps reported separately); until then say "the patch content, not its coverage, fails" |
| C8 | Thinking helps small and mid changes, not large ones | **Weak** | v7 → v14: small +3, mid +2, large +2 (of 18) | Bundled changes, tiny strata; descriptive only |
| C8b | The v14 bundle (thinking + budgets + submit fix) raises resolve | **Supported** | new-60 23 → 30, exact McNemar p = 0.039; LB 0.15 → 0.17 | Bundle, not thinking alone; one run per arm |
| C9 | Sampling does not reach large changes | **Weak** | pass@5 0/7 multi-necessary; size-matched controls also ~0 | A13 on new-60 |
| C10 | Necessary co-changes are mostly not call edges | **Supported on 7 tasks** | Same file 26/28; executed-call 1-hop 4/28 | A5 extends to 19 tasks |
| C11 | Structural frontiers do not recover co-changes better than lexical ones | **Supported** (pre-registered) | E0b: typed 0.19 vs BM25 0.42; CI excludes 0 | Gold co-change label, not necessity |
| C12 | Typing edges by change kind helps | **Rejected** | E0a, E0b | — |
| C13 | Results transfer to private repos | **Not shown** | — | Limitation only |

---

## 5. Analyses still to run (all on Kaggle CPU)

**Done:** A1 (size confound), A3 (new-60 necessity), A4 (completion counterfactual), E0a, E0b.

| ID | Analysis | Why it matters | How | Priority |
|---|---|---|---|---|
| **A12** | **Site-swap counterfactual.** For every unresolved agent patch that edits ≥1 gold group: gold patch with that group's hunks replaced by the agent's edits to the same group. Fails → the agent's edit at that site is wrong. Control: gold alone, re-run. **Pre-register first** | Tests C7 directly; turns "found, not fixed" from inference into measurement | New mode in `evalkit/necessity.py` (same kernel; agent hunks per group via `hunk_functions`) | **P0** |
| **A13** | Table 1 and RQ4 on the new-60 necessity labels: resolve by necessary-group count, with the A1 size control (≥2 vs 1 necessary) | Fills Table 1's last row; checks C3 with necessity instead of file count | Re-run the A1 kernel with `necessity_new60.json` added to its payload | **P0** |
| A7 | Budget at submit for unresolved patches: calls left, prompt tokens, time; explicit submit vs ran out | Distinguishes "stopped early" from "out of budget" for wrong edits | Kaggle CPU kernel over per_task + traces (traces embedded or as a private dataset) | P1 |
| A5 | Typology of the 19 new-60 multi-necessary tasks (read gold diffs; AST relations from `oblig.py`) | Table 3 with n = 26 tasks instead of 7 | Gold diffs from `results/runs/a3a4/`; reading, not running | P1 |
| A8 | Why BM25 wins on E0b: share of co-changed pairs that share identifiers or an edited token; repeated-edit patterns; same- vs cross-file split | Explains the negative result | Kaggle CPU kernel over `results/runs/e0b/e0b_pairs.json` + SWE-rebench snapshots (internet) | P1 |
| A2 | Funnel at line and function level per size stratum | Fig. 1 | Kaggle CPU kernel over per_task fields | P2 (§3.1 table may suffice) |
| A8b | E0a re-scored with v2 rules, **labelled post hoc** | Completeness of §4 | Kaggle CPU kernel with the xg data | P2 |
| A9 | Add v16 new-60 as a third agent, if its eval completes | Stability of C2 | Already scheduled; no new spend | P2 |

**Dropped:**
- **A6** (did the agent look at the missed sites). A4 shows completion is not the lever, so *why* sites were missed matters little.
- **A10** (lexical co-change hint, GPU). Dropped for budget. C6 makes a large gain unlikely but does not strictly cap an in-run hint, so say "not tested", not "bounded".
- **E1–E4** closed-loop frontier runs, renamed-repo runs, and new LB submissions for the paper.

---

## 6. Timeline to freeze (Nov 5)

**Decided Oct 24: Nov 12.** The compressed Nov 2 plan below is kept for reference only:

| Date | Work |
|---|---|
| Oct 24 | Commit A12 + A13 pre-registrations (before building); build the site-swap mode |
| Oct 25–26 | A12 + A13 on Kaggle CPU; A5 reading |
| **Oct 27** | **Freeze.** Drop A7, A8, A8b, A2 (or move them to "future work") |
| Oct 27–30 | Write (≤3,000 words) and make the tables |
| Oct 31–Nov 1 | Revise (Atul reviews) |
| **Nov 2** | Submit |

**Plan in use:**

| Date | Work |
|---|---|
| Oct 24 | Pre-register A12 and A13; extend `necessity.py` (site-swap mode) and the A1 kernel |
| Oct 25–26 | A12 + A13 on Kaggle CPU; A5 reading |
| Oct 27–28 | A7, A8 kernels |
| Oct 29 | Claims-ledger review: every claim still "plausible" or "weak" gets its final wording |
| Oct 30–Nov 5 | Figures and tables; freeze |
| Nov 5–12 | Write and revise; submit |

## 7. Figures and tables

| Item | Content | Data |
|---|---|---|
| Fig. 1 | Exposure → edit → resolve by size stratum | A1 strata, §3.1 (A2 optional) |
| Table 1 | Resolve by stratum × agent | have + A13 |
| Fig. 2 / Table 2 | Oracle completion bound: (a) / (b) / (c) and pure incompleteness, case- and task-level | A4 |
| Table 3 | Obligation typology | have + A5 |
| Table 4 / Fig. 3 | Pre-registered recall tests E0a/E0b, recall@k per arm | have |
| (Fig. 4) | Site-swap: share of agent-edited sites that are wrong | A12 |
