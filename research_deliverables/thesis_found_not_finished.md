# Found, Not Finished: obligation graphs for closing multi-site changes in small code agents

**Thesis proposal: paper track (≤3,000 words, due Nov 12) and main-track candidate (finals Dec 2).** Written Oct 23, 2026. **Adopted Oct 23.**

> **Status (Oct 23, final): method claim FAILED its pre-registered gates. Do not run E1–E4.**
>
> | Test | Typed frontier hit@6 | Gate needed | Result |
> |---|---|---|---|
> | E0a (7 tasks, test-necessary pairs) | 0.45 | ≥ 0.80 | FAIL |
> | E0b (120 external repos, v2 rules) | 0.19 | ≥ 0.80 | FAIL |
>
> On E0b, BM25 (text similarity) scored 0.42 and random same-file 0.21. Per §8 the paper falls back to *Found, not finished*: the find-vs-finish analysis (H1, supported) plus the obligation analysis. The obligation analysis is now a mixed result: obligations are not runtime call edges, but on external code they are better predicted lexically than structurally. `tools/oblig.py` is kept as a tool, not a claim. See `docs/experiments.md` (E0a, E0b).

**Inputs:**
- `gaps_2026-10-22.md` and its four surveys.
- Our run history (`docs/experiments.md`, `docs/findings.md`, `docs/HQ.md`).
- Two new analyses run for this proposal: [`analysis/find_vs_finish.py`](../analysis/find_vs_finish.py) and [`analysis/obligation_pilot.py`](../analysis/obligation_pilot.py).

Every number below that comes from our data names the run it came from. Pilot numbers are labelled as pilots, with their n.

---

## 0. The thesis in one paragraph

**The problem.** A 31B quantised code agent on a 32k context almost always **finds** the code it has to change. In our best leaderboard configuration (v14), it had seen a gold file in 29 of the 30 tasks it failed. What it cannot do is **finish**. When a fix must touch several program elements that depend on each other, the agent changes one of them and submits. Those elements are a constructor parameter and the methods that read it, a callee and the caller that must pass a new argument, or sibling overrides of one protocol method.

**Why current code graphs miss it.** Today's code-graph methods spend their structure on *localisation*, before the first edit. That is the phase our agent has already saturated, and the phase where memorised names already do the work. The relations that carry these co-change *obligations* look different:
- field def→use;
- parameter forwarding;
- override families.

Mostly they are not runtime call edges. The competition's provided graph, and graphs built from executed calls, largely miss them.

**What we propose.** Use the graph *after* the first edit:
- type the change the agent just made;
- propagate it along typed, directed obligation edges;
- hand the agent a small **obligation frontier** passively, inside the output of the edit itself ("structure on write");
- run a cheap closure check before submit.

**Why it should transfer.** The frontier is computed from the repository in front of the agent, not from memory. So it should help most exactly where memorised cues fail: on private, unseen, or renamed repositories.

---

## 1. Which gaps, and why these

From the 24 gaps in `gaps_2026-10-22.md`, these six are both closest to what we observed in the competition and testable by Nov 12:

| # | Gap (from the survey) | Open question we take on | Why it is ours |
|---|---|---|---|
| Q1 | **CC7 + X4.** Same-model localisation handoff is unstudied; nobody has decomposed SWE failures into evidence-hit vs conversion | When a small agent has *seen* the right code, what stops it from producing a complete fix? | Gold-file exposure is ~97% in our failures (v14); localisation is saturated, conversion is not |
| Q2 | **CC4.** Graph correctness in dynamic languages: edge errors are unmeasured downstream | Which relations actually connect the program elements that must change together, and do existing graphs contain them? | Provided graph covers 17% of executed call edges, 0% of `self.m()`/constructor edges (xg, Oct 9); obligations travel via fields, overrides and forwarding |
| Q3 | **CC2 + CC3 + X3.** Structure under a hard context cap for weak tool-callers; name-resolution failures of graph lookups | How can structure reach an agent that ignores optional tools and cannot name nodes, inside 32k with no compaction? | 0 graph-tool calls in v5; `search_similar_code` empty for every phrase or bare name (0/461 calls); v14 overflows 8/60 |
| Q4 | **CC1.** Structure × familiarity has never been crossed | Is "find" memorised while "finish" is structural, and does structure help more on unseen code? | Local gains (15→26→33) did not transfer to the hidden LB (0→0.08→0.08); hidden repos are private |
| Q5 | **GR4 + X2.** No standard test that a model *uses* topology | Is any gain due to the graph's topology, or to extra text? | Shuffle/relabel controls are cheap for us and rare in code-graph papers |
| Q6 | **X1 + GR8.** Code as a free source of ground-truth graphs; path verification | Can test-necessity give a ground-truth obligation set, removing the "proxy reward" objection? | We already have leave-one-gold-function-out runs for 54 tasks (xg instrument) |

Gaps we deliberately do **not** take: GR1 (10^5–10^7-node graphs), CC9 (embeddings), CC10 (repo memory). They are real, but not where our agent fails, and not testable by Nov 12.

---

## 2. What our competition data already shows

### 2.1 Finding is saturated; finishing is not

[`analysis/find_vs_finish.py`](../analysis/find_vs_finish.py) uses trace_analyzer's per-task output (gold files, gold functions, patch files, agent patches). Same 60 new tasks, all counted (overflows count as failures):

| | v7 (LB 0.15) | v14 (LB 0.17) |
|---|---|---|
| Resolved | 23/60 | 30/60 |
| Gold fix touches **1 file** | 22/43 (51%) | 27/43 (63%) |
| Gold fix touches **≥2 files** | **1/17 (6%)** | **3/17 (18%)** |
| Fisher exact, single vs multi | p = 0.001 | p = 0.003 |
| Unresolved tasks where the agent saw a gold file | 29/37 | **29/30** |
| Multi-file tasks where a gold file was edited: patch covers *all* gold files | 2/11 | 3/15 |

How v14's 30 failures break down:

| Failure | Tasks |
|---|---|
| Partial: some gold files untouched | 10 |
| Saw the gold file but edited only elsewhere | 9 |
| Partial: all gold files, but some gold functions untouched | 5 |
| Empty patch | 5 |
| Complete coverage, still wrong | 1 |

**Most failures sit after "find".** 24 of the 30 saw the right code and either did not edit all of it or chose a different site.

**v14's improvements over v7 went mostly to easy tasks.** Thinking, budgets and prompt fixes gained +5 single-file tasks but only +2 multi-file. Better reasoning alone does not close multi-site changes.

### 2.2 Ground-truth obligations: test-necessity pilot

The xg instrument (Oct 9) ran the fail-to-pass tests with each gold function group removed. A group whose removal fails the tests is **test-necessary**. 7 of the 54 old valid tasks have ≥2 test-necessary groups. [`analysis/obligation_pilot.py`](../analysis/obligation_pilot.py) crossed them with v7's five repeated runs:

| | Runs resolved | Tasks solved at least once in 5 runs |
|---|---|---|
| 1 test-necessary group | 66/185 (36%) | 17/38 |
| ≥2 test-necessary groups | **0/35** | **0/7** (Fisher p = 0.034, task level) |

The 7 tasks show what the relations look like (read from the gold diffs):

| Task | What must change together | Relation type |
|---|---|---|
| rich_3052 | `case_sensitive` added to `PromptBase.__init__` → stored on `self` → forwarded by classmethod `ask(..., case_sensitive=)` → read in `check_choice` → called from `process_response` | **parameter plumbing + field def→use** |
| fastapi_15800 | `APIRouter.__init__` option → consumed in `_IncludedRouter._build_effective_context` (+ `applications.py`) | **constructor field → consumer** |
| rich_3942 | `Heading` / `HorizontalRule` / `TableElement.__rich_console__` | **override family** (same protocol method on sibling classes) |
| rich_3061, rich_3468 | Sibling methods of `Text` / `Console` sharing state | **same-class siblings** |
| rich_3953 | `_cell_len` ↔ `get_character_cell_size` | **helper/callee** |
| fastapi_14986 | `FastAPI.setup.openapi` (root-path servers) **and** `docs.get_swagger_ui_html` (HTML escaping) | **two independent issue facets**, not a structural link |

fastapi_14986 is the cleanest example. In **all 6** v7 runs that reached it, the agent fixed `get_swagger_ui_html` and never touched `FastAPI.setup`.

**Where the obligation pairs sit:** across the 28 ordered pairs of necessary groups,
- 26/28 are in the same file;
- 4/28 are within 1 hop on the *executed* call graph, and 16/28 within 2 hops;
- 4/28 are within 1 hop on the provided graph, and 20/28 within 2 hops, where a 2-hop frontier is a median of 38 nodes (too big to inject).

The relations that bind co-changes are mostly **not runtime calls**. Same-file proximity finds them but is not selective: a whole file is far too much text for a 32k context.

**Caveats.** n = 7 tasks. Two tasks dominate the agent-run statistics. This is motivating evidence, not a result.

### 2.3 Facts that constrain any method

| Fact | Source | Consequence for the design |
|---|---|---|
| Optional graph tools get 0 calls; graph search requires fully qualified names | v5, findings Oct 4 | No new tool to call and no node naming: the **edit itself must be the query** |
| No compaction; 32k context; v14 overflow 8/60 | HQ, v14 report | The frontier must be tiny (≤8 lines, ≤150 tokens) |
| `edit_file` drifts 61%; `write_file` 100% clean; `run_command` 98.8% clean | m1-drift | Deliver through `run_command` output, bootstrapped with `write_file` |
| Skill scripts unverified on the scorer (v6 0.00) | v6 | Do not depend on `skills/`; carry the tool in the prompt |
| Imitation SFT lowered loss but collapsed behaviour (Step D 0/114; S3b 9/60 vs 23/60) | Step D, S3b | Training-free method first; training is optional and last |
| Local gains stopped transferring (15→26→33 local vs 0→0.08→0.08 LB) | experiments | Must test on renamed/unseen repos, not only public ones |

---

## 3. Thesis, hypotheses and how each could fail

**Thesis.** For small agents on unfamiliar code, the binding failure is *incomplete propagation of a change*, not failure to locate it. Code structure helps when it is (a) typed by obligation relation, (b) computed from the edit the agent just made, and (c) delivered passively inside the edit's own output. Its benefit grows when memorised cues are removed.

| ID | Hypothesis | Prediction (pre-registered before the GPU runs) | Falsified if |
|---|---|---|---|
| H1 | Find is saturated; finish is the bottleneck | On new-60 and old-54, conversion (resolve given gold seen) is far below exposure; multi-obligation tasks resolve ≤ ⅓ as often as single ones | Multi-site resolve rate ≥ ½ of single-site in the baseline |
| H2 | Obligation edges ≠ call edges | Typed obligation graph (field, forward, override, call) recovers ≥ 80% of test-necessary co-change pairs at frontier ≤ 8 nodes; provided and executed-call graphs recover ≤ 50% at the same size | Executed-call or provided graph matches typed recall at equal frontier size |
| H3 | Passive frontier converts | v14 + obligation frontier resolves more multi-site tasks than v14 (paired), with no loss on single-site tasks and no rise in overflow | Paired McNemar shows no gain on multi-site tasks, or overflow rises by > 3 tasks |
| H4 | The gain comes from topology | A same-size frontier of random same-file symbols (shuffle control) gives no gain | The shuffle control is within 1 task of the real frontier |
| H5 | Find is lexical/memorised; finish is structural | Under namespace remapping, *exposure* and single-site resolve drop; the frontier's gain on multi-site tasks is unchanged or larger | The frontier's gain shrinks under renaming by as much as baseline find does |

H1 is already supported by §2. H2 can be settled offline on CPU. H3–H5 need GPU runs.

---

## 4. Method

### 4.1 Obligation graph (typed, directed, built per repository at task time)

Nodes are module-level functions, classes, methods, class attributes, and module constants. Edges are directed and typed. Each type has a transfer rule saying which kinds of change propagate along it:

| Edge type τ | From → to | Extracted by | Propagates these change kinds |
|---|---|---|---|
| `CALL(argmap)` | call site → callee, with the positional/keyword binding recorded | AST + import/alias resolution + `self.`/`cls.`/`super()` resolution | signature change (param add/remove/rename/default), return-contract change |
| `FWD` | function forwarding a parameter by name (`f(..., x=x)` / `**kwargs` pass-through) → callee | AST dataflow over parameter names | new or renamed parameter (the plumbing chain) |
| `FIELD` | writer of `self.a` (usually `__init__`) → every reader of `self.a` in the class hierarchy | AST attribute def/use | new field, changed field semantics |
| `OVR` | method → same-named method in sibling or parent/child classes | class hierarchy + name | protocol/behaviour change in one implementation |
| `EXPORT` | definition → `__init__.py` re-export / `__all__` | AST imports | rename, new public symbol |
| `TEST` | test function → covered symbol (static import/call) | AST | any change: tells the agent which existing test to run |

**Why not reuse a graph that already exists:**
- The provided graph has one mixed label, mirrors every edge, and has no field, forwarding or override edges.
- An executed-call graph needs tests to run and misses non-call relations (§2.2).

Our extractor needs only `ast`, which is in the Python standard library, and runs in under a second on fastapi/rich-scale repositories (v5's `code-map` skill: 0.1–0.4 s).

**Calibration (offline, never on test tasks).** We fit a weight w_τ per edge type: the probability that a co-change crosses an edge of type τ. The data is gold patches from repositories disjoint from our evaluation set (SWE-smith and SWE-rebench task repos; AST only, so no execution and CPU only). This is the only learned part. It has six parameters, so it cannot memorise a repository.

### 4.2 Change typing from the diff

After each edit, diff the edited function before and after, and classify the change:
- signature (parameter added, removed or renamed, default changed);
- new or removed `self` field;
- return or raise contract (new `return`/`raise` shapes);
- protocol method body (method name shared with siblings);
- body-only.

This picks which edge types the change propagates along: body-only changes propagate only along `OVR` and `TEST`.

### 4.3 Frontier = typed, budgeted propagation

Starting from the edited node(s), run a typed personalised-PageRank / bounded BFS. Only edge types allowed by the change kind are used, weighted by w_τ, with depth ≤ 3. Hub nodes are damped by in-degree, the cheap version of the over-squashing concern in the earlier proposal.

The result is the **top-k frontier (k ≤ 6)**. Each item is one line, rendered as:

```
OBLIGATIONS after your edit of PromptBase.__init__ (new parameter case_sensitive):
  1. rich/prompt.py:118 PromptBase.ask(...) constructs cls(...) without case_sensitive  [FWD]
  2. rich/prompt.py:205 PromptBase.check_choice reads self.choices; new field unused   [FIELD]
  3. tests/test_prompt.py::test_prompt_str covers PromptBase                            [TEST]
```

Items already changed in the current `git diff` are dropped, so the frontier shrinks as the agent closes obligations.

### 4.4 Delivery: "structure on write", with no new tool and no node naming

The scorer does not allow Python entry points, and skills are unverified, so the tool travels inside the prompt:
1. The system prompt (`!include`d markdown) contains `oblig.py` (~150 lines, standard library only).
2. **Bootstrap:** the agent writes the file once with `write_file`, which had 100% clean calls, then moves it out of the workspace with `run_command`, so it never appears in the patch.
3. **Edit wrapper:** v7/v14 already edit through a Python snippet-replace in `run_command`. The wrapper `python3 /tmp/oblig.py edit FILE` performs that same exact-snippet replace, then prints the obligation frontier. Every edit returns its own obligations, so the agent never has to choose to query structure or know a node name (Q3).
4. **Closure check:** `python3 /tmp/oblig.py check` runs before `submit_patch`. It lists unaddressed obligations of the whole `git diff` plus the `TEST` items, and is capped at 8 lines.
5. **Issue-facet line (for fastapi_14986-type tasks):** the agent states the issue's distinct requested behaviours in one line at the start. The check echoes any facet with no edit. This is our cheap version of SWE-Doctor's multi-faceted reproduction (2607.00990), and it targets the non-structural share of obligations.

All tool output stays under 1,000 characters (cap 5,000), and no history is rewritten.

### 4.5 Why the closure is computed outside the model (the graph-reasoning argument)

Obligation closure is typed multi-hop reachability with change-dependent transfer functions, close to an IFDS-style dataflow problem. The graph-reasoning theory surveyed in gap GR3 predicts that this is exactly what a fixed-depth transformer does badly:
- connectivity capacity grows only with depth (diameter ≤ 3^L, 2510.19753);
- past capacity, models fall back on degree heuristics;
- length generalisation is "recursively unstable" (2604.15306).

In-context alternatives cost what we cannot afford. Pasting the file or a 2-hop subgraph (median 38 nodes) is expensive in a 32k context with no compaction, and RepoGraph found that a flattened 2-hop subgraph *hurts* (26.0 vs 27.3).

So the division of labour is: a symbolic engine computes the closure; the LLM decides which obligations are real and how to edit them. We measure that split directly:

> P(resolve) ≈ P(obligation ∈ frontier) · P(agent acts on it | shown) · P(edit correct)
>                 [graph recall, offline]   [conversion, from traces]

These are GraphDancer's evidence-hit and conversion, applied to SWE agents (X4).

### 4.6 What we do not do (and why)

- **No LoRA in the core claim.** Imitation SFT collapsed the agent three times (Step D, S3, S3b). A frontier-conditioned adapter is listed as future work, not as a dependency.
- **No soft graph tokens or GNN encoder.** That route cannot be served as PEFT, and graph tokens are shown to go unused (2606.03712).
- **No graph for localisation.** That phase is saturated (§2.1), and published gains there are small and confounded by memorisation.

---

## 5. Experiments

### 5.1 Offline (CPU only, no LLM). Settles H2 and calibrates w_τ

| ID | Data | Measure | Arms |
|---|---|---|---|
| E0a | Old-54 (xg, LOO necessity already run) plus **new-60 with the xg instrument re-run** (Kaggle CPU kernel `kaggle/exec_graph`, ~1–2 CPU-h) | Pairwise obligation recall@k (k = 2, 4, 6, 8) from each necessary group to the others; median frontier size in tokens | Typed obligation graph · provided graph · executed-call graph · same-file (random order) · BM25 lexical neighbours · random |
| E0b | Gold patches of 2–5k SWE-smith / SWE-rebench tasks on repos disjoint from ours (AST only; no execution) | Co-change recall@k of the frontier from one gold function to the others; per-edge-type co-change lift (gives w_τ) | Same arms; repo-disjoint split for calibration vs test |
| E0c | Every partial run in v7 / v14 / rs_v7 traces | Would the frontier from the agent's *own* first edit have contained a missed obligation, and at what rank? | Typed vs provided vs same-file |

**Pass gate before any GPU spend.** Typed recall@6 ≥ 0.8 on E0a/E0b, and at least +0.2 above the best non-typed arm at equal size. If it fails, stop H3–H5 and publish H1/H2 as an analysis paper (see §8).

### 5.2 Closed loop on the scorer-faithful L4 runner (GPU)

The base is v14, or v16 if v16 lands as the new best; that is decided before E1. Tasks: new-60 plus old-54 (114 valid tasks, one pass each), with multi-site tasks pre-flagged from gold so the strata are fixed in advance.

| ID | Arm | Tests | Cost (quota-h, measured 8/60-task arm) |
|---|---|---|---|
| E1 | base vs **base + OF** (frontier + closure check + facet line) | H3 | 2 × ~15 |
| E2 | **base + OF-shuffled**: same wrapper, same number of lines, random same-file symbols with plausible reasons | H4 | ~15 |
| E3 | **base + lexical frontier**: the wrapper prints `grep -n` hits of the edited name instead of the typed frontier | Structure vs lexical propagation | ~15 (cut first if quota is short) |
| E4 | **Renamed** new-60 (Schrödinger namespace remapping, code released, or our `build_devset.py` renamer extended to all identifiers), × {base, base + OF} | H5 | 2 × ~8 |
| E5 | One LB submission of base + OF | Scorer-path validation; main-track signal (LB noise ~±0.03) | 1 day |

**Total ≈ 90 quota-h** if run on all 114 tasks, which does not fit before Nov 5. **Minimum viable set:** E1 + E2 + E4-OF-only on new-60, ≈ 40 quota-h over two accounts in ~2 weeks. The 17 multi-file tasks in new-60 are the primary stratum. With 17 tasks, a 3→9 change is detectable (exact McNemar p ≈ 0.03 if there are no losses). Smaller effects will need old-54 added.

**Metrics:**
- resolve rate;
- find (gold-file exposure);
- **finish** (coverage of test-necessary groups);
- partial-patch rate;
- frontier conversion (shown → edited);
- tokens, overflow, calls, wall time;
- per-arm worst-case runtime (scorer ≤ 12 h).

Stats are paired McNemar plus a repo-clustered bootstrap. Predictions are frozen in `docs/experiments.md` before each run (house rule: one change per experiment).

---

## 6. Why this is a winning contender

1. **It explains our own leaderboard, not just a benchmark.** The opening result is that a model sees the fix 97% of the time but finishes it 18% of the time when the fix spans files, and 0/35 runs when ≥2 functions are test-necessary. "Sees it" replicates From Finding to Fixing (2610.25826) on a new model. "Doesn't finish it", measured against test-necessity, is ours. It comes from the model and scorer stack the judges care about.
2. **It sits exactly at the intersection the track asks for.** The code-comprehension side is a typed obligation graph from AST. The graph-reasoning side is typed, budgeted propagation with a measured recall/conversion split, plus topology controls (H4) that most code-graph papers skip.
3. **It turns the field's weak result around.** Published graph-for-agents gains are +1–3 pp on memorised repos and target localisation. We argue, and test, that structure belongs *after* the first edit, and that its value is hidden by memorisation (H5). The idea of edit obligations is CodePlan's (2309.12499). What is new is delivering them passively inside an autonomous agent's edit loop, and showing causally (shuffle control) that topology, not extra text, does the work.
4. **It ships under the competition's constraints.** Declarative only, no skills, no adapter, about 150 tokens per edit. One LB submission validates the scorer path. Few paper-track entries will have both a paper claim and a scorer-verified artifact.
5. **It is honest about negatives, and they strengthen it.** Graph-tools-unused, imitation-SFT collapse and the local→LB gap become motivation, not embarrassment.
6. **It is cheap to falsify.** The CPU gate (E0) decides before any GPU spend, so the downside is bounded (§8).

### Novelty position against the closest work

The prior-art check is in `surveys_2026-10-22/novelty_obligation.md`. It was checked through alphaXiv; the arXiv API was down, and "abstract only" items are marked there.

| Prior work | What it already covers | What remains ours |
|---|---|---|
| **CodePlan** (2309.12499, FSE'24) | **The core mechanism.** A typed dependency graph (calls, overrides, instantiates, imports, uses, construct), a change may-impact analysis after a seed edit, per-change-type rules, and a plan graph of "edit obligations" | The planner drives GPT-4 block by block, with given seed edits, on migration/temporal-edit tasks. It is static-only, uses no SWE-bench, has no autonomous agent and no small model. We move the idea into **passive post-edit context for an autonomous small agent**, add the field/forwarding edges our pilot shows matter, and evaluate on issue resolution |
| **From Finding to Fixing** (2610.25826, Oct 19 2026) | "Localization is not the bottleneck": agents reach the gold file 88.6–99.4% of the time; a bug-cause description beats location | Our ~97% exposure is a **replication** on a 31B quantised open model, not a headline. The *finish-given-find* split by single- vs multi-site, with test-necessity labels, is not in it |
| **ChainSWE** (2607.02606) | "Undershoot": agents edit only the files the issue names. 48% of downstream failures are chain errors; under-edits outnumber over-edits ~9:1 | Diagnosis only. No edit-triggered intervention; no test-necessity ground truth; no topology control |
| **CodeAnchor** (2606.26979, ISSTA'26) | Passive, grep-visible static structure works where optional tools are ignored | It injects *before* any edit, for localisation (+2.2 pp Func@5). We inject *after* an edit, triggered by the diff, aimed at finishing |
| **TRIM** (2607.18161), delta debugging | Removing hunks from *agent* patches and re-testing to cut unnecessary code | We apply leave-one-out to *gold* patches to **define** the necessary co-change set. Caveat: alternative valid patches use different files in 11.6% of pairs (2610.25826), so we report necessity *relative to the gold patch* |
| **SWE-Doctor** (2607.00990) | Multi-faceted reproduction tests to avoid partial patches | Requirement-side completeness; we cite it for the facet line (§4.4) and own only the structural side |
| **CABRA** (2610.10610) | Frontier agents are near-perfect on *synthetic* call-graph propagation | Scopes our claim: **small models on real repositories**, where obligations run through fields and forwarding, not only calls |
| Software World Models (2610.04940); SiblingRepair (2605.06209); MULocBench (2509.25242); CoEdPilot (2408.01733); incomplete-changes data (2609.34317) | Learned execution-based impact (cross-service); multi-hunk sibling repair; multi-location benchmark; edit-propagation prediction; prevalence of incomplete human changes | Cited as related; none is an edit-triggered frontier inside an autonomous agent with a necessity-labelled, topology-controlled evaluation |

**Claim exactly this:** an **edit-triggered, passive obligation frontier for an autonomous small-model SWE agent**, computed from the agent's own diff over a typed graph whose weights are calibrated on external co-change data. It is evaluated with:
- a finish-given-find decomposition;
- a test-necessity ground truth (gold leave-one-out);
- a shuffled-frontier topology control;
- a renaming interaction.

**Do not claim any of these:** change-impact propagation itself, "localisation isn't the bottleneck", passive graph injection in general, or execution-informed impact in general.

**One-line framing for the paper:** CodePlan's obligations, moved from a planner that drives the LLM into the edit loop of an autonomous agent, with a causal control and an execution-grounded necessity label.

---

## 7. How this relates to earlier decisions and proposals

- **The Sep 29 rejection of T1** (name-invariance + structure reward) gave four reasons. Each answered:
  1. *"Optimises for a shift absent from the test set."* We do not optimise for renaming. Renaming is only an *instrument* to separate memorised find from structural finish. The hidden set is private repos, which is a shift in kind.
  2. *"No ground-truth chains; the reward is a proxy."* Test-necessity (LOO) gives ground truth, and nothing is trained on it.
  3. *"GRAIN for code is an application."* There is no GRAIN-style training here; the contribution is where and how structure is used.
  4. *"129 tasks can't power the claims."* There is a paired design on pre-registered strata, and H2 is tested on thousands of external gold patches.
- **The Oct 7 drop of the renamed dev set** reasoned that renaming keeps structure. That is precisely what H5 needs: keep structure, remove names. SchrödingerRepo (2609.27891) shows the renaming drop vanishes on post-cutoff tasks, so renaming does isolate memorisation.
- **The Oct 20 EBST proposal** (`paper_outline.md`, `architecture_proposal.md`) has four problems as a submission:
  - it needs a synchronous CPU engine the scorer cannot run;
  - it targets localisation precision, which is not our bottleneck;
  - its curvature/over-squashing premise is unmeasured on code;
  - its stopping rule targets an adapter pathology (S3 submitting after failed tests) that the base agent does not show.

  Two parts carry over: non-monotonic evidence from tests becomes the `TEST` items in the closure check, and hub damping is kept in §4.3. **Recommendation:** supersede EBST with this thesis.
- **Atul's Oct 10 edit-necessity proposal** is absorbed. Its LOO instrument becomes our ground truth for obligations, and its gates and replay protocol carry over unchanged.

---

## 8. Risks, kill criteria, fallbacks

| Risk | Signal | Response |
|---|---|---|
| Typed recall is not better than same-file | E0 gate fails | Stop GPU work. The paper becomes *"Found, not finished: a find/finish decomposition of a small SWE agent"*: H1, H2-negative, the obligation typology and topology analysis. This is still a solid analysis paper, at near-zero GPU cost |
| Agent ignores frontier lines | Low shown→edited conversion in E1 traces | Move the frontier into the closure check, which runs once before submit, and make submit conditional in the prompt ("if `check` lists items, address or justify each") |
| Overflow rises | Overflow > base + 3 | Cut k to 3; print only file:line and reason |
| Bootstrap file fails on the scorer (Docker/gVisor) | E5 LB < base by > noise | The paper uses the local runner (scorer-faithful); E5 is a main-track bonus, not a paper dependency |
| Too few multi-site tasks for power | n < 15 per stratum | Add old-54; report effect sizes with CIs; frame the GPU results as a pilot alongside the CPU-scale E0b |
| Prior art covers the core | Checked Oct 23: CodePlan covers obligation propagation itself; 2610.25826 covers "localisation isn't the bottleneck" | Already narrowed (§6): edit-triggered passive delivery in an autonomous agent, test-necessity labels, topology control. Cite CodePlan prominently in the intro, not in a footnote |
| Reviewers see renaming as weak (it preserves structure) | Review comment | That is the design: H5 needs structure kept and names removed. State it in one sentence and cite SchrödingerRepo's post-cutoff result |

---

## 9. Timeline (today Oct 23 → paper Nov 12)

| Dates | Work | GPU |
|---|---|---|
| Oct 23–25 | `oblig.py` (graph, change typing, frontier, check) + unit tests on the 7 pilot tasks; E0a on old-54; xg re-run on new-60 (CPU kernel) | 0 |
| Oct 25–27 | E0b on SWE-smith/rebench gold patches (CPU kernel with internet); fit w_τ; **E0 gate** | 0 |
| Oct 27–28 | v14 + OF submission compiled and validated (`kaggle/validate`); renamed new-60 built | 0 |
| Oct 28–Nov 3 | E1, E2 (Shrey + Atul accounts in parallel), E4; E5 LB submission | ~40 quota-h |
| Nov 3–5 | Analysis, figures; freeze | 0 |
| Nov 5–12 | Write (≤3,000 words): intro 300, find/finish evidence 600, method 700, results 900, related work and limitations 500 | 0 |
| Nov 12–Dec 2 | Main track: if E5 holds, tune k and the facet line; pick 2 finals | LB only |

### Paper skeleton (3,000 words)

1. **Introduction.** "Found, not finished".
2. **Find vs finish in a 31B agent.** §2 evidence; Figure 1 = funnel, single- vs multi-site.
3. **Obligations are not call edges.** Typology and E0 recall curves (Figure 2).
4. **Structure on write.** Method and delivery (Figure 3, frontier example).
5. **Closed-loop results.** E1–E4 (Table 1); conversion analysis.
6. **Related work.**
7. **Limitations.** Python only; public repos plus renaming as the unseen-code proxy; small n.
