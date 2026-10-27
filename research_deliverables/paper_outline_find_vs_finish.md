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

**Oct 25: all remaining analyses done** (Kaggle CPU; rows in `docs/experiments.md`):
- **A12 → A12v:** A12's 52/71 used a flawed "clean" label (audit). The pre-registered validation A12v keeps only compatible hybrids: **43/51 fail behaviourally (84%), 18/19 tasks**. **C7 is Supported (validated).**
- **A13:** with necessity labels there is a raw gap (task-level permutation p ≈ 0.048). After size the OR is 1.68, CI 0.45–6.33: too wide to separate size from multiplicity. The pooled Fisher test was withdrawn because it treated the two runs as independent.
- **A7:** **C14 withdrawn** (audit). Four failed v14 runs hit BudgetExceeded and then submitted; A7 only counted budget errors that ended a run. Still standing: v14 large changes overflow the context (6/14).
- **A5:** necessary co-changes are parameter plumbing (7/19) or the same edit repeated across siblings (7/19). Shared edited identifiers: 95% macro / 78% micro, inflated by keywords (keyword-free ≈ 89% macro, audit). Direct graph links: 35% macro / 16% micro.
- **A8:** BM25 wins on E0b because co-changed functions already look alike, not only through a shared edit. The prediction failed.
- **A8b:** E0a under v2 rules gives typed 0.6286 vs untyped 0.6257 (post hoc; gate still fails).

**Oct 27: Atul's independent audit** (`research_deliverables/atul_independent_research_audit_2026-10-27.md`) reproduced the counts and found the measurement and inference problems corrected here. These were C7, C14, the A13 statistics, A5 keywords, exposure wording, A4's "upper bound", the old-54 repeat count, and the stale v16 note. The audit recommends narrowing to three RQs and validating A12 first (A12v, pre-registered in `docs/experiments.md`).

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
3. **A low observed gold-completion rescue rate** (A4, new; formerly called an "oracle completion bound", a wording the audit rightly rejected). Adding the correct missing edits to the agent's final partial multi-site patch rescues 2/13 tasks (CI 4–42%). This bounds *post-hoc completion of a frozen patch*. An in-run aid could also change the edits the agent did make, so it is not a bound on every aid. It is still strong evidence that missing sites are not the main failure.
4. **A pre-registered negative result for structural co-change recovery** (E0a/E0b). Lexical similarity beats typed graph frontiers on 120 external repos.

---

## 2. Research questions

| RQ | Question | Evidence | Status |
|---|---|---|---|
| RQ1 | Does the agent reach the code it must change? | v14 exposure; failure decomposition | have |
| RQ2 | What predicts failure: number of sites or size of change? | A1 (logit + size-matched) | have |
| RQ3 | Are failures unfinished or wrong? | A4 completion counterfactual; A12 site-swap | have (A4, A12) |
| RQ4 | Do more reasoning or more samples help, and where? | v7 → v14 by size tercile; pass@5; A7 | have |
| RQ5 | Can code structure supply the co-changes? | Typology (A5), E0a, E0b, A8, A8b | have |

---

## 3. Sections, word budget, and what each needs

### §1 Introduction (300 words)
- **Hook.** In v14, our best leaderboard config, the agent saw a gold file in 29/30 of its failed tasks. Single-file tasks resolve at 63% and multi-file at 18%. Yet when we hand it the edits it missed, 2/13 tasks pass.
- **Contributions** (4 bullets, as in §1 of this outline):
  1. saturated localisation;
  2. failure tracks change size;
  3. the observed completion rescue rate (A4) and the site-swap diagnostic (A12, pending validation);
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
| A12, A13 | `92dabb6` (committed before any kernel was built) | A13 `09085fc`; A12 in the commit after `d5845c4` | yes |
| A5, A7/A2, A8, A8b | `7ad401b` | A7/A2, A8 `09085fc`; A8b `d5845c4`; A5 with A12 | yes (A8b registered as post hoc) |

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
| ≥2 test-necessary groups (19 of the 57) | 4/19 | 6/19 | A13; others 18/38 and 23/38; v16 6/19 vs 20/38; task-level permutation p ≈ 0.048 (audit) |

**Size-controlled models (A1)**, each with controls for issue length and repo:

| Model | Odds ratio | 95% CI | p |
|---|---|---|---|
| log(changed lines), pooled v7 + v14 | 0.47 per log line | — | 0.046 |
| log(changed lines), v14 alone | 0.37 | — | 0.025 |
| Multi-file, after size | 0.32 | 0.04–2.54 | 0.21 |
| ≥2 test-necessary groups, after size (A13, pooled) | 1.68 | 0.45–6.33 | 0.44 |
| log(changed lines) in the A13 model, pooled | 0.31 | — | 0.0003 |

- **Size-matched comparisons** (same repo, size caliper): 5–10 sets by file count, 13 by necessity; none significant (A13: +8 / −5 pp).
- **How to word it:**
  - Say: *"resolve falls with the size of the required change. Multi-site fixes are mostly large ones, and in these data the two cannot be separated."*
  - Do not say "because multi-site". The pre-registered prediction (multi-file OR < 0.4, p < 0.05) failed.

**§3.3 Not unfinished: observed completion rescue (RQ3, ~300 words).**

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
- **Cases are correlated:** 16 of the 26 come from v7's 5 old-54 repeats (10 are new-60: 3 v14, 7 v7). Headline the task-level numbers.
- **Composition modes:** 13 cases applied cleanly with git, 8 needed fuzzy patching and 5 needed the no-module fallback. Completion exit codes were six 0, fourteen 1 and six 2; exit 2 may be collection/usage failures, and no output tails were kept (audit).
- **Predictions that failed:** completion ≥ 50%, pure ≥ 35%. The pre-registered decision rule fired (completion < 30%).

**§3.4 Wrong at the right site (RQ3 continued, ~150 words). A12 + validation A12v, both pre-registered.**
- **Design:** the mirror of A4. Start from the gold-patched tree and replace one gold function group with the agent's own version of that function (AST segment). Before any test runs, each swap is classed:
  - *identical:* the agent's version equals gold;
  - *conflict:* the swap does not build or compile;
  - *coupled:* the agent's version uses names that only its own edits elsewhere define;
  - *clean:* none of these.
- **Swaps (128):** from 61 unresolved patch cases (29 distinct tasks) across v7, v14 and v7 ×5:

| Class | Swaps |
|---|---|
| clean | 85 |
| coupled | 20 |
| conflict | 17 |
| identical | 6 |

  Conflict plus coupled is 29%, within the ≤30% the swap needed to be informative. The gold re-run control passed on 29/29 tasks.
- **Result (protocol-labelled):** among clean swaps of test-necessary groups, **52/71 hybrids fail the target tests** (task-bootstrap CI 45–100%). 19/20 tasks have at least one failing hybrid. Breakdown of the 52: 49 exit 1, 2 exit 2 (`fastapi_14978`), 1 exit 124 (`rich_4006`); excluding these gives 49/68. `rich_3052` contributes 15 primary swaps, all passing. Only 6 v14 tasks are in the primary subset.
- **Secondary:** all clean swaps 54/85 (64%); coupled swaps 18/20 (reported separately).
- **Predictions held arithmetically**, but Atul's audit showed the "clean" label was not a valid compatibility check. It resolved names against all gold files at once and ignored decorators, and no hybrids or output tails were kept.
- **A12v (pre-registered validation, Oct 27).** Same 128 swaps, rebuilt with:
  - namespace-aware name resolution in the same file;
  - checks for attributes the agent added elsewhere in the class;
  - a separate *interface-changed* class for changed signatures;
  - decorators included in the function span;
  - gold→gold self-swap and agent→agent round-trip controls;
  - every hybrid patch and 4,000 chars of test output kept;
  - failure typing by a rule fixed in advance.

| A12v class | Swaps |
|---|---|
| validated-clean | 61 |
| namespace-incompatible | 24 |
| interface-changed | 15 |
| attribute-incompatible | 5 |
| conflict | 17 |
| identical | 6 |

  25 of A12's 85 "clean" swaps were incompatible. Controls: self-swap identical 125/128, the 3 others being conflicts; gold re-run 29/29.
- **Result:** among compatible hybrids of test-necessary groups, **43/51 fail the target behaviour (84%; task-macro 0.90; 18/19 tasks; task-bootstrap CI 0.63–1.00)**. 7 pass and 1 times out; none are ambiguous. A post-hoc exclusion of reverse coupling gives 42/50. A manual read of 10 random tails found real behaviour failures: assertion mismatches, and errors raised inside the agent's own function.
- **Predictions held and the decision rule fired: C7 is Supported (validated).**
- **Disclosures:**
  - a coupling-rule bug in A12's smoke run (fixed then);
  - the post-hoc reverse-coupling flag was added after A12v's smoke run and is not used by the decision;
  - one reader so far (Atul second-read pending);
  - the population is A12's selected cases, and 16 swaps are v7 ×5 repeats.
- **Reading, with §3.3:** with every other site correct, the agent's own edit to a site the tests need usually fails the target behaviour. Supplying its missing sites rarely helps. The binding failure is what the agent writes, not where or how much.

**§3.5 Reasoning and sampling (RQ4, ~80 words).**

**The one significant positive result:**
- **Overall:** v7 → v14 (thinking on) on new-60 went **23 → 30/60** (22 both, 8 v14-only, 1 v7-only; exact McNemar **p = 0.039**).
- **Empty patches:** fell from 25% to 8%.
- **Wrong fixes:** stayed at 16 in both.

So reasoning helped the agent *commit* to an edit, not get the edit right, consistent with §3.3 and §3.4.

**How wrong patches end (A7):**
- **v7:** 20/21 unresolved non-empty patches contain an explicit submit; raw traces are unavailable, so budget denials cannot be checked.
- **v14:** 17/24 contain an explicit submit. **4 of them hit BudgetExceeded before submitting** (audit). There were 8 context overflows, 6 of them in large changes.
- **C14 ("voluntary stopping") is withdrawn.** A submit call does not show the agent chose to stop.

The agent stops because it believes it is done, not because it runs out. For v14 the 32k context is a binding limit on large changes.

Two counterpoints for the same section:
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
- **A13** on new-60 necessity labels agrees: tasks with ≥2 necessary groups resolve 10/38 vs 41/76 across the two runs, but the gap disappears once size is controlled.

### §4 Can structure supply the co-changes? (RQ5, 600 words)

**Typology (Table 3).** 26 tasks with ≥2 test-necessary groups: the 7 old-54 pilot tasks and the 19 new-60 tasks (A5). Categories come from reading the gold diffs (dominant relation per task):

| Relation | old-54 (7) | new-60 (19) | Example |
|---|---|---|---|
| Parameter / option plumbing (ctor → field → consumer → callee) | 2 | 7 | `strict_content_type` through FastAPI → APIRouter → APIRoute → handler (fastapi_14978) |
| Same edit repeated across sibling functions or classes | 1 (override family) | 7 | `field.alias` → `get_validation_alias(field)` at 7 call sites (fastapi_14371) |
| Helper / callee | 1 | 3 | `prepare_body` + `_encode_files` / `_encode_params` (requests_7505) |
| Constructor field → consumer | 1 | 1 | `ModelField.__post_init__` + `get_model_fields` (fastapi_14482) |
| Same-class siblings | 2 | — | |
| Independent issue facets | 1 | — | |
| Repo-wide refactor | — | 1 | dropping pydantic v1 (fastapi_14609; 306 of the 596 new-60 pairs) |

**Where the pairs sit:**

| Measure | old-54 pilot (pairs, of 28) | new-60 (A5, task-macro) |
|---|---|---|
| Same file | 26 | 61% |
| Same class | — | 17% |
| Direct `oblig.py` edge, any type | — | 35% |
| Executed-call 1-hop | 4 | — |
| Two-hop | — | 51% |
| Share an edited identifier (keyword-free, A12v) | — | 89% macro / 65% micro (with keywords: 95% / 78%) |

The shared identifier is typically the plumbed parameter or the repeated token. The pilot's same-file share did not replicate (61% vs 26/28): new-60 co-changes cross files more often.

**Pre-registered recall tests (Table 4):** hit@6 of the first-edit frontier.

| Test | Label | Typed AST | Untyped AST | BM25 | Same-file random | Executed calls | Provided |
|---|---|---|---|---|---|---|---|
| E0a, 7 tasks | test-necessary | 0.45 | 0.54 | 0.36 | 0.30 | 0.29 | 0.27 |
| E0b, 120 external repos | gold co-change | **0.19** | 0.24 | **0.42** | 0.21 | n/a | n/a |
| A8b: E0a with v2 rules (post hoc) | test-necessary | 0.6286 | 0.6257 | 0.36 | 0.30 | 0.28 | 0.28 |

- Both gates failed (bar 0.80).
- Typing the graph by edge kind hurts in both tests.
- Lexical similarity wins on external code.

**Why lexical wins (A8; its pre-registered prediction failed):**

| E0b pairs | n | TYPED hit@6 | BM25 hit@6 |
|---|---|---|---|
| Share an edited identifier | 660 | 0.22 | 0.51 |
| Share none | 532 | 0.15 | 0.31 |
| Cross-file | 506 | 0.11 | 0.27 |

- The predicted < 0.10 gap in pairs with no shared edit is in fact **0.16**.
- In a logistic regression, BM25 hits are driven mostly by **identifier overlap of the existing code** (coef 21.9, p < 0.001), more than by a shared edit (0.73).
- **Reading (associational):** co-changed functions tend to look alike, consistent with A5's shared-identifier rate (78–95%, keyword-inflated; see C10). The A8 logit is pair-level and unclustered, so it is not a mechanism.

**Tie to §3.3–§3.4.** Gold completion rescues 2/13 tasks, and compatible site hybrids fail behaviourally in 43/51 swaps. Recall of co-changes is therefore a secondary lever, which is why no closed-loop frontier run was justified. In-run effects remain untested.

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
| C1 | The agent is **exposed to** the code it must change | **Supported, as a proxy** (replication) | v14: 98.3% gold-file and 91.7% gold-line exposure; 29/30 failures (CI 83–99%). Exposure = path mentioned in a call or output, lines = intended read range (`trace_analyzer.py`); a one-line read of a file counts as line exposure (audit, synthetic check) | Say "exposed to", not "sees", "reaches" or "localises"; returned-span exposure is unmeasured. Cite 2610.25826 |
| C1b | Exposure does not mean editing the right place | **Supported** (descriptive) | 9/30 v14 failures saw a gold file and edited elsewhere | Heuristic label; report beside C7 |
| C2 | Resolve falls with the size of the required change | **Supported** | A1: OR 0.47 per log line, p = 0.046 (pooled); v14 alone OR 0.37, p = 0.025 | Correlational; controls for issue length and repo only |
| C3 | Multi-site fixes resolve less often *because* they are multi-site | **Not established** (A1, A13) | A1: OR 0.32 [0.04, 2.54] after size. A13 (necessity label): v7 4/19 vs 18/38, v14 6/19 vs 23/38, v16 6/19 vs 20/38; task-level permutation on mean v7/v14 outcome, difference 0.28, p ≈ 0.048 (audit, unadjusted). After size: OR 1.68 [0.45, 6.33] | Wide intervals: size and multiplicity cannot be separated here. That is neither proof of no effect nor a "disappearance". The earlier pooled Fisher p = 0.006 treated the two runs as independent and is withdrawn |
| C4 | Multi-site fixes are common | **Supported** | A3: 19/60 new-60 tasks need ≥2 test-necessary groups; 7/54 old-54 | Relative to gold patches |
| C5 | Failures are mostly unfinished patches | **Rejected** | A4: completion rescues 6/26 cases, 2/13 tasks; pure incompleteness 1/13 | State as a rejected pre-registered hypothesis |
| C6 | Adding the missing gold edits to the agent's final patch rescues few partial repairs | **Supported, as an observed rate** | A4: 6/26 cases on 2/13 tasks (task CI 4–42%); agent-alone 0/26. Five of the six successes are repeats of rich_3052. On rich_4070, adding the agent's edits turns a passing missed-only patch into a failure | "Observed gold-completion rescue rate under the specified protocol". **Not** an upper bound: composition is non-monotonic and other completions may behave differently |
| C7 | The agent's edits at the right sites are wrong | **Supported (validated, A12v)** | Compatible hybrids only (namespace, attribute and interface checks; decorators; gold→gold self-swap control): **43/51 fail the target behaviour (84%; 18/19 tasks; task CI 0.63–1.00)**. 10/10 random tails confirmed by hand. Original A12: 52/71 protocol-labelled; 25 of its "clean" swaps were in fact incompatible | Word it as "with every other site gold, the agent's version of a test-necessary function fails the target behaviour in 43/51 compatible swaps (18/19 tasks)". The population is A12's selected cases; 16 swaps are v7 ×5 repeats; Atul second-read pending |
| C8 | Thinking helps small and mid changes, not large ones | **Weak** | v7 → v14: small +3, mid +2, large +2 (of 18) | Bundled changes, tiny strata; descriptive only |
| C8b | The v14 bundle (thinking + budgets + submit fix) raises resolve | **Supported** | new-60 23 → 30, exact McNemar p = 0.039; LB 0.15 → 0.17 | Bundle, not thinking alone; one run per arm |
| C9 | Sampling does not reach large changes | **Weak** | pass@5 0/7 multi-necessary on old-54; size-matched controls also ~0 | Old-54 only (no new-60 repeats); descriptive |
| C10 | Necessary co-changes are mostly not direct structural edges and often share an edited identifier | **Supported, aggregation-dependent** | Pilot: executed-call 1-hop 4/28. A5 (new-60): direct edge 35% task-macro, 16% pair-micro. Shared edited identifier 95% macro and 78% micro, but the token extractor kept keywords (`self`, `None`, `return`). Keyword-free (A12v): 89% macro, 65% micro. Same file 61% macro, 33% micro | Report macro and micro side by side; headline the keyword-free values (89% macro / 65% micro). One refactor task holds 306/596 pairs |
| C11 | Structural frontiers do not recover co-changes better than lexical ones | **Supported** | E0b: typed 0.19 vs BM25 0.42; CI excludes 0. A8b: v2 rules reach 0.63 on E0a's necessity pairs but only tie untyped | Gold co-change label, not necessity; E0a/E0b prediction and result share a commit, so disclose that |
| C11b | Co-changed functions are found lexically because they already look alike | **Weak** (A8; its prediction failed) | BM25 hit associated with identifier overlap of existing code (coef 21.9); gap 0.16 even without a shared edit | Pair-level logit without task clustering, on unstandardised predictors with oracle gold-edit features. Associational only; no mechanism claim |
| C14 | Wrong patches are submitted voluntarily, not cut off by the budget | **Withdrawn** | A7 counted only budget errors that *ended* a run. Four unresolved non-empty v14 runs hit BudgetExceeded and then submitted (`fastapi_11355`, `fastapi_14246`, `fastapi_14266`, `rich_3782`). v14 overall: 10 BudgetExceeded events in 8 tasks. v7 traces are not available to check | Keep only: v14 large changes ended in 6/14 context overflows. Event-level run-ending analysis is future work |
| C12 | Typing edges by change kind helps | **Rejected** | E0a, E0b | — |
| C13 | Results transfer to private repos | **Not shown** | — | Limitation only |

---

## 5. Analyses (all on Kaggle CPU)

**All done as of Oct 25:** A1, A3, A4, A12, A13, A5, A7, A2, A8, A8b, E0a, E0b. Results, predictions and kernel ids are in `docs/experiments.md`; outputs are in `results/runs/`.

**Not run:**
- **A9** (v16 as a third agent): its per-task outcomes are now in the repo (Atul, PR #7: `data/runs/atul_eval_v16_new60/`). Its A13 strata (6/19 vs 20/38) are in Table 1; a full v16 refit is optional.
- **A6** (did the agent look at the missed sites): dropped, because A4 + A12 show the edits themselves are wrong.
- **A10** (lexical co-change hint, GPU): not tested. C6 makes a large gain unlikely but does not strictly cap an in-run hint, so say "not tested", not "bounded".
- **E1–E4** closed-loop frontier runs, renamed-repo runs, and new LB submissions for the paper.

**Open, optional before freeze:** a second reader for the A5 typology (one rater so far).

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
| Oct 25 | **Done:** A12 + A13 + A5 + A7/A2 + A8 + A8b on Kaggle CPU |
| Oct 26–28 | Draft §3–§4 from the ledger; Atul second-reads the A5 typology |
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
| Fig. 4 | Site-swap: share of agent-edited sites that are wrong (clean / coupled / conflict / identical) | A12 (have) |
