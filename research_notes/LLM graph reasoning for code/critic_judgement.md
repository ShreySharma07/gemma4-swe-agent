# Critic judgement: which research gap survives? (8 Oct 2026)

Role: adversarial area chair. Inputs: the four notes in this folder, `docs/HQ.md`, `docs/findings.md`, `docs/plan.md`, `CLAUDE.md`, `analysis/gold_graph_analysis.py`, `analysis/verify_graph.py`, and `results/runs/*`. Every query I ran is in the **Search log** at the end.

Scores run 0–3 on four axes: **N** novelty (after my own searches), **F** feasibility under the hard constraints by Nov 5, **P** payoff (hidden-test generalisation + graph-ML paper), **Fa** falsifiability with dense metrics. Any 0 kills. A user veto also kills.

---

## 0. Three local facts that change the judging (computed this session from `results/runs/`)

These were computed on CPU from files in the repo. They are VERIFIED as arithmetic. The labels are noisy: `gold_funcs` comes from the trace analyser's regex parsing of the gold patch.

| # | Fact | Numbers | Consequence |
|---|---|---|---|
| L1 | In v5, getting the *function* right in the localiser's diagnosis does **not** predict resolve. | Diagnoses with gold funcs: 109. FILE right 88/109 (81%). FUNCTION right 52/109 (48%). Resolved given function hit: **14/52 (27%)**. Given a miss: **16/57 (28%)**. | Any method whose payoff is "better function-level localisation" (most of the literature: CodeScout, A²Agent, SweRank, C2C) is aimed at a **non-bottleneck** for Gemma. Localisation-only rewards get P ≤ 1 for the leaderboard. |
| L2 | Touching the gold *file* is close to necessary. | v2: patch touches a gold file → 24/80 resolved. Otherwise 2/49. | File-level retrieval matters, and it is already 81%. Better file maps add little. |
| L3 | Fixes spanning several functions are a dead zone. | v2 resolve by number of gold functions: 1 → 19/57 (33%); 2 → 5/20 (25%); **≥3 → 2/45 (4%)**. 45/129 tasks have ≥3. | One-to-many fixes are the biggest pool of failures. Some of these 45 are feature additions whose new functions are not in any graph yet. |

Also re-read: the dominant v2 failure is **wrong fix, never tested** (47/129), and gold lines were seen in 80% of v2 tasks. The bottleneck is **conversion (diagnosis → correct, verified fix)**, not search. Every candidate below is judged against this.

---

## Round 1: candidate gaps, attacked

Prior work cited here was either read this session (S#) or is VERIFIED in the four notes.

| # | Candidate gap / hypothesis | N | F | P | Fa | Verdict | One-line reason |
|---|---|---|---|---|---|---|---|
| C1 | Repair the provided graph (direction + edge typing) as the headline | 1 | 3 | 1 | 3 | **KILL (veto)** | The user rejected it as a headline. Keep it as an ingredient. Codebase-Memory already has confidence-typed edges. |
| C2 | Degraded-graph controlled study (G1: undirected/mislabelled graph as the variable) | 2 | 3 | **0** | 3 | **KILL** | This is "imperfect graph measurement" (vetoed). It cannot move the hidden test, where graph files may be absent. |
| C3 | Graph tools vs no graph under identical post-training (G2) | 2 | 1 | **0** | 2 | **KILL** | Agents make 0 graph calls (v5). Hidden graphs are unknown. Either outcome leaves the leaderboard unchanged. It also needs two LoRA trainings. |
| C4 | Graph localisation on quantised ~30B models (G3) | 1 | 3 | **0** | 2 | **KILL** | Characterisation only. Flat-Score already covers INT4 Gemma tool errors. |
| C5 | Dense entry retriever + directed expansion (G4, SweRank-style) | 1 | **0** | 2 | 3 | **KILL** | No text encoder can ship (LoRA-only, fixed tools, skills unverified). `search_similar_code` is a name lookup. |
| C6 | Discovery-commit reward + passive graph injection into observations (G5) | 0–1 | **0** | 1 | 3 | **KILL** | Tool outputs cannot be altered in a declarative YAML submission. The DC reward alone is A²Agent's. L1 says commit precision is not the bottleneck. |
| C7 | Static call-graph-checked evidence-chain reward/filter for RFT | 2 | 2 | 1 | 2 | survive (weak) | VulAgentRL (static CPG node-ID F1) and P2T (observed-symbol check) are close. A static graph cannot certify edges (incomplete-graph theorem, 2606.30247). L1 caps payoff if the chain only targets location. |
| C8 | Chain-validity / faithfulness metric for agent diagnoses (eval) | 2 | 3 | 1 | 3 | survive as ingredient | No benchmark scores stated chains against a graph (eval note, my S2/S11). On its own it is measurement, not a method. |
| C9 | Graph-privileged on-policy self-distillation (teacher sees the gold path) | 3 | 1 | 2 | 2 | survive | Untried for code (RC-OPD and OPSD are math only). Teacher log-probs on 31B over ≤32k tokens on L4 is heavy. RC-OPD's "reference borrowing" risk applies. |
| C10 | Batched, direction-filtered, paged neighbour tool / provenance collapse | 2 | **0** | 1 | 3 | **KILL** | Tools are fixed by the harness. Skill scripts are unverified on the scorer (v6 = 0.00). |
| C11 | Shortcut-aware C×P filter (PathRouter ported to code) | 2 | 2 | 1 | 2 | **KILL (veto)** | Penalising name-matched hits without a structural path is T1 "structure over names" again. |
| C12 | Bug-injection, many-repo dense dev/train set (SWE-smith / TRACE pattern) | 1 | 2 | 3 | 3 | survive as infrastructure | Not novel (CodeScout trains on 39K SWE-smith tasks). It is the only way to power claims (≥30 repos), so it is required, not a headline. |
| C13 | Free-fork counterfactual credit for read-only localisation | 2 | 1 | 1 | 2 | survive → R2 | Many 31B rollouts per fork. Localisation-only payoff (L1). |
| C14 | Non-LLM community/architecture map in the prompt | 1 | 1 | **0** | 2 | **KILL** | Targets file-level retrieval, which is already 81% (L2). The map would have to be computed in-sandbox per hidden repo. |
| C15 | On-demand dynamic trace graph built by the agent (repro → stdlib tracer via `run_command`) | **1** | 2 | 2 | 3 | survive as ingredient | Inference-time execution localisation is crowded: XAgent (S4), SWE-Doctor (S3), ADI/FramePilot (S7), CT-Repair (S16), EviACT (S17), SieveFL (S10). |
| C16 | Grep-built static subgraph idioms taught via LoRA ("callers via grep") | **0** | 3 | 2 | 2 | **KILL** | Covered by RepoNavigator's single `jump` tool, CodeScout (bash only, RL) and FastContext (grep explorer, RL). |
| C17 | Structural support of a diagnosis as a label-free verifier / best-of-k selector | 2 | 2 | 1 | 3 | survive → R2 | R4P does group-wise patch verification, Agentless does majority voting. L1 says choosing a better location does not raise resolve. |
| C18 | Text → provided-embedding bridge so natural-language queries work | 1 | **0** | 1 | 2 | **KILL** | The tool accepts names, not vectors. Nothing deployable. |
| C19 | **Execution-witnessed graph** (built offline by tracing failing tests) as the *verifier / supervision* for evidence-backed diagnosis | 2 | 2 | 2 | 3 | survive | TraceEval and RepoReasoner use execution-witnessed call structure only as eval labels or for trace-prediction fine-tuning. Nobody uses it to certify an agent's stated chain during training (S2, S5, S9, S11). |
| C20 | One-to-many structural completion (graph impact closure for multi-function fixes) | 1 | 2 | 2 | 3 | survive | Biggest failure pool (L3). Partly covered: GraphLocator (one-to-many CIG), MultiFixer, SiblingRepair, the SWE-Doctor impact field (S8, S15). |

Round 1 survivors: C7, C8, C9, C12, C13, C15, C17, C19, C20.

---

## Round 2: combine known pieces plus ONE new piece, then attack again

| # | Combination (known pieces + **one new piece**) | N | F | P | Fa | Verdict | One-line reason |
|---|---|---|---|---|---|---|---|
| R2-a | C19 + C7 + C8 + C15 + C12 + **new: an execution-witnessed chain is the training-time verifier (filter/weight) for a diagnosis LoRA that runs graph-free at test time** | 2 | 2 | 2 | 3 | **SURVIVE (top)** | New searches S5/S9/S11/S13 found no training method that checks an agent's *stated* symptom→cause chain against an execution-witnessed graph. Inference-time cousins exist (SWE-Doctor's runtime-grounding check, CT-Repair's TEG), so the delta is the training signal plus the graph analysis. Inference needs neither graph files nor skills (the traceback from `run_command` is a natural chain fragment), so it ships on private repos. The chain carries the *mechanism*, so it aims at conversion, not L1. |
| R2-b | C9 + C19: OPSD where the teacher is privileged with the witnessed chain | 3 | 1 | 2 | 2 | **MERGE into R2-a** | Same premise as R2-a (privileged evidence must help the teacher). It becomes R2-a's training-method ablation, not a separate thesis. Cost: teacher log-probs for every student token on 31B. |
| R2-c | C17 + C15: test-time graph self-consistency over k parallel diagnoses with repro traces | 1 | 1 | 1 | 3 | **KILL** | Selection is crowded (R4P, Agentless, XAgent ranking). Optimises location (L1). k parallel runs plus tracing will not fit ~5 min. |
| R2-d | C20 + **new: co-change closure measured on the witnessed + repaired graph; brief lists graph-adjacent co-edit candidates (OTHER_LOCATIONS)** | 1–2 | 2 | 2 | 3 | **SURVIVE (runner-up)** | Targets L3 directly. Novelty is weak (GraphLocator, MultiFixer, SiblingRepair, change-impact analysis). It survives because it is cheap and its kill test is independent of R2-a's. |
| R2-e | C12 + C8: dense chain-faithfulness benchmark on injected-bug unseen repos | 2 | 2 | 1 | 3 | **MERGE into R2-a** (evaluation) | On its own it is measurement, and P is low. As R2-a's evaluation it is what makes the claims well-powered. |
| R2-f | C13 alone | 2 | 1 | 1 | 2 | **KILL** | GPU cost out of budget. Localisation-only payoff. |

---

## Round 3: the pick

### Top hypothesis H1: Execution-Witnessed Evidence Chains (EWEC)

**Falsifiable claim.** For Gemma-4-31B-W4A16 on unseen repos, the binding failure is conversion, not localisation (L1, L2, findings). Diagnoses whose symptom→cause chain is **certified by an execution-witnessed call graph** carry information that location alone does not. Concretely:

- **(i) Premise / upper bound.** With the oracle location held fixed, adding a compact witnessed chain lifts resolve by **≥ +8 pp**, versus giving the location only.
  - The chain is ≤600 tokens: failing-test entry → … → gold function, plus exception type and argument/return values at the gold frame, with no test source and no expected values.
- **(ii) Method.** RFT / advantage-weighted SFT that keeps or up-weights trajectories whose stated chain has hop-precision ≥0.5 on the witnessed graph, compared with outcome-only RFT at matched sample count. Predictions on ≥500 injected-bug tasks from ≥30 held-out repos:
  - resolve **+4 pp**;
  - chain hop-precision **+20 pp**.
- Predicted sizes are **GUESSED**. They are anchored on SWE-Doctor (+8.0–8.9 pp on SWE-bench Pro from runtime diagnosis, frontier models), CT-Repair (w/o TEG −8.6% repaired), and the 3–9-point process-reward gains in EoG, PathRouter and VulAgentRL.

**Solution = known pieces + one new piece.**

| Piece | Source | Status |
|---|---|---|
| Execution-witnessed call structure as ground truth, built by tracing tests | TraceEval 2605.11006, RepoReasoner 2607.25996 (eval note) | known |
| Failing-path / differential traces narrow candidates | XAgent 2609.09769 (S4), CT-Repair TEG 2607.12605 (S16), EviACT 2605.27238 (S17) | known, inference-only |
| Runtime-grounded diagnosis record (location, symptom, propagation path, values) | SWE-Doctor 2607.00990 (S3) | known, inference-only, prompt check |
| Evidence-grounded process reward against a code graph | VulAgentRL 2607.26656 (static CPG node-ID F1, vulnerability detection) | known |
| Groundedness filter for SFT curation | P2T 2605.21996 (symbols must have been *observed*) | known |
| Soft grounding under incomplete graphs | 2606.30247 (theorem: hard grounding fails) | known |
| Offline-compatible weighted SFT | A²Agent 2608.29831 (advantage-weighted MLE, LoRA r16) | known |
| Direction repair of the provided graph (86–90% recoverable) | A2 in HQ | known, allowed as an ingredient |
| **NEW: the agent's *stated* chain is scored edge-by-edge against an execution-witnessed graph built offline for each training task. The score filters/weights a diagnosis LoRA, with soft credit for statically supported but unwitnessed hops. At test time no graph or tracer is needed: the model cites traceback frames from its own repro.** | — | not found (S2, S5, S9, S11, S12, S13, S14) |

**Graph-ML content for judges:**
1. A measured property of real code graphs: what fraction of the provided static graph is "live" on failing paths, and how often its shortest seed→gold paths use unwitnessed or hub hops.
2. Chain hop-P/R as a structural faithfulness metric.
3. Soft grounding.
4. A degree-preserving rewire control.

None of this relies on the vetoed "structure over names" framing.

**Novelty risk (harsh).** A reviewer can call this "SWE-Doctor's runtime-grounding check moved to training time." The defence depends on (a) the training-signal ablation (witnessed vs static vs outcome-only filter) and (b) the static-vs-witnessed graph analysis. If Stage A finds static ≈ witnessed, the defence collapses. That is why it is a kill criterion.

**VERIFIED vs GUESSED**

| Claim | Status |
|---|---|
| L1–L3 numbers above | VERIFIED (computed this session; `gold_funcs` labels are regex-derived) |
| Tasks carry `FAIL_TO_PASS`, `PASS_TO_PASS`, `test_patch` | VERIFIED (`kaggle/devset/build_devset.py` l.181, l.272) |
| 54/128 originals pass gold in the scorer-like env; `scorer_site.tgz` exists | VERIFIED (HQ; `data/runs/devset_v5/`) |
| v5 localiser diagnoses (FILE/FUNCTION/ROOT_CAUSE) exist locally for 121 tasks | VERIFIED (`results/runs/r3_v5_graph/behaviour_per_task.json`) |
| Raw trajectories and agent patches for r1–r3 are retrievable from Kaggle kernel outputs | GUESSED (not local; only summaries are in the repo) |
| Task sandbox is `python:3.13-slim` with `run_command` working, so `python -m trace` / tracebacks are available at inference | VERIFIED (HQ, CLAUDE.md); runtime overhead in the sandbox GUESSED |
| No prior work trains on witnessed-chain validity | VERIFIED only as absence across ~17 searches (alphaXiv coverage of the last ~10 days may lag; OpenReview ICLR-2027 submissions not searchable) |
| Effect sizes (+8 pp oracle, +4 pp trained, +20 pp hop-precision) | GUESSED |
| Gemma-31B-4bit will use runtime evidence. Counter-evidence: Qwen3-32B used ADI in only 28–31% of tasks vs 71–73% for Claude-3.7 (S7) | GUESSED / risk |
| `run_official_l4.py` works on the Py 3.13 image for Stage B | NOT YET (HQ says it needs the same fix as `build_devset.py`) |

---

## The cheapest experiment that could kill H1

It runs in two stages. Stage A is CPU-only and is a prerequisite: it builds the witnessed graphs that Stage B needs. Each stage has a pre-registered kill rule.

### Stage A: witnessed-graph premise (CPU, ~1 working day, 0 GPU-h)

**Data**
- All tasks whose gold patch passes in the scorer-like env (54; rich 39, fastapi 10, requests 5). Optionally widen to the ~109 host-env-stable tasks as a sensitivity set.
- For each task: `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, the provided graph (`.npz`/graph files on Kaggle), the gold patch, and the v5 diagnoses (local).

**Procedure** (one Kaggle CPU kernel; reuses `build_devset.py`'s scorer env and `gold_graph_analysis.py`'s parsers)
1. Check out the base commit and apply `test_patch`. Run the F2P tests under a `sys.setprofile` pytest plugin. Record caller→callee edges between repo functions (qualified names) and the test entry. Do the same for P2P tests in the same test files.
2. Map gold hunks to the enclosing pre-existing `def`s via `ast`. Mark gold functions added by the patch as "new".
3. Compute:
   - **M1 coverage**: share of tasks where ≥1 pre-existing gold function is executed by F2P.
   - **M2 depth**: minimum call depth test→gold; share with depth ≥2.
   - **M3 differential set**: functions executed by F2P but not by P2P. Report median size, and whether gold is inside it.
   - **M4 static agreement**: for witnessed edges on test→gold paths, the share present in the provided graph and the share whose source-text direction matches. Also the share of the provided graph's shortest seed→gold paths that contain ≥1 unwitnessed or hub hop.
   - **M5 Gemma near-miss**: among the 57 v5 diagnoses with the wrong function, the share whose function lies on the witnessed failing path (upstream "symptom" node) vs off it.
4. Run a degree-preserving rewire of the witnessed graph as a null for M4/M5.

**Sample size / power**
- n=54 tasks gives a 95% CI of about ±13 pp on proportions near 50%. That is enough for the coarse thresholds below, which sit ≥20 pp from the predicted values.
- M5 uses ~57 wrong diagnoses (±13 pp).
- The 109-task set narrows CIs to about ±9 pp.

**Kill criteria (any one kills H1's novel piece)**
- **K1** M1 < 50%: witnessed chains are missing for most tasks (feature-heavy or import-time failures). The training labels would not exist.
- **K2** median M2 ≤ 1 **and** median M3 > 100 functions: the chain is trivial (the test calls the gold function directly) or uninformative (the trace is too broad).
- **K3** M4 ≥ 90% edge agreement **and** ≥ 95% direction agreement: the witnessed graph adds nothing over static direction repair. The "new construction" collapses into the vetoed obvious ingredient.
- (Informative, not a kill) M5 < 20%: Gemma's wrong diagnoses are off-path, so chains add little over F1. This shifts weight to R2-d.

**Predictions (GUESSED):** M1 ≈ 70–80%; M2 depth ≥2 in ≈ 60%; M3 median ≈ 15–40; M4 edge agreement ≈ 60–75%; M5 ≈ 35–50%.

**Cost:** ~3 h to write the tracer and mapping, ~1–2 h of Kaggle CPU, ~1 h of analysis.

### Stage B: oracle-evidence upper bound (GPU, ~5–6 GPU-h, only if Stage A passes)

**Data:** the same 54 scorer-faithful tasks.

**Procedure:** a fixer-only v2-style agent (no localiser), temperature 0.2, scorer budgets. Two arms receive an identical brief header:
- **A0**: oracle FILE + FUNCTION only.
- **A2**: A0 plus the ≤600-token witnessed chain from Stage A (frames + exception type + values at the gold frame; no test code, no expected values).
- Optional third arm if quota allows: **A1** = A0 + static repaired-graph chain. This is the graph-ML contrast.

**Metrics**
- Paired resolve (exact McNemar).
- Dense secondaries: the edit touches gold hunk lines; a test was run after the last edit.

**Power:** with ~20% discordance, the SE of Δ is ≈ 6 pp at n=54. This is an *upper-bound* test, so the bar is set accordingly.

**Kill rule**
- **Kill:** Δ(A2−A0) ≤ 0 pp. Even oracle witnessed evidence does not help Gemma convert, so a trained model emitting imperfect chains cannot either. This also kills R2-b (privileged OPSD).
- **Pass:** Δ ≥ +10 pp with one-sided p < 0.1.
- **Ambiguous:** anything in between. Add a second seed, or extend to the host-env tasks.

**Cost:** 2 arms × 54 × ~2.5 min ≈ 4.5 h, plus ~0.5 h vLLM start. This fits next week's 30 h quota. Only 5.5 h remain until Oct 10, so Stage A runs first.

**Infrastructure dependency:** port the Py 3.13 install fix and `use_scorer_env()` into `run_official_l4.py` first (≈ half a day).

### What to do after each outcome

| Outcome | Next step |
|---|---|
| **A pass + B pass** | Build R2-a. (1) Mint ≥500 injected-bug tasks on ≥30 permissive repos, with symptom text taken from failing-test output, not from the diff. (2) Sample Gemma rollouts (vLLM, k=4) whose diagnosis brief requires a CHAIN field. (3) Train witnessed-filtered vs outcome-only LoRAs on matched counts. (4) Evaluate with repo-clustered bootstrap. (5) Spend 1 LB submission on the winner. The paper's figure set: static vs witnessed graph (M4), chain hop-P/R, Stage B upper bound, RFT ablation. |
| **A pass + B fail** | Evidence-backed diagnosis does not help this model convert. Drop the chain thesis. Keep the Stage A graph analysis as a paper section, and move to the runner-up (R2-d) or to plain tool-format/verification RFT for the leaderboard. |
| **A fail (K1/K2)** | Witnessed labels are unavailable or trivial on real tasks. Run the runner-up test (same kernel output, ~2 h extra). |
| **A fail (K3)** | Static repair ≈ execution. The novel construction is gone. Fall back to the runner-up, using the static graph. |
| **Ambiguous** | Widen to the 109 host-env tasks (CPU) before spending any GPU. |

---

## Runner-up H2: co-change closure for one-to-many fixes (R2-d)

**Claim.** Gemma resolves 4% of tasks with ≥3 gold functions (L3). On *failed* multi-function tasks where the agent edited ≥1 gold function, **≥50% of the missing pre-existing gold functions lie within 2 non-hub hops** of an edited one, on the direction-repaired (or witnessed) graph. That makes a graph-derived co-edit list (in OTHER_LOCATIONS, via `get_code_neighbors` when graphs exist, or grep-callers otherwise) a cheap completion step.

- Predicted effect: +3–5 pp resolve on the multi-function subset at most (GUESSED).
- Novelty is weak (GraphLocator, MultiFixer, SiblingRepair). Its value is that it attacks the largest observed failure pool.

**Cheapest test (CPU, ~3 h, can share Stage A's kernel)**
- **Data:** the 65 tasks with ≥2 gold functions, the provided graphs, the gold patches, and the r2/r3 agent patches (Kaggle outputs).
- **Procedure:** for each task:
  - (a) the share of gold functions that are newly added (not in the base graph);
  - (b) pairwise hop distances among pre-existing gold functions on the raw, repaired and hub-free graphs;
  - (c) for failed tasks with a gold function touched, the hop distance from the touched to the untouched gold functions.
  - Compare against a null of random same-file function pairs.
- **Kill:**
  - >50% of gold functions in ≥3-function tasks are new: graph completion cannot help; or
  - <40% of the missing gold functions are within 2 non-hub hops; or
  - the closeness is no better than the same-file null (AUC < 0.6): proximity is a file effect, not a graph effect.
- **Next:**
  - Pass → add co-edit candidates to the v2 brief and run a paired L4 check on the multi-function subset (~3 GPU-h).
  - Fail → one-to-many failures are a fix-complexity problem, not a structure problem. Stop pursuing graph headlines and put effort into tool-format/verification RFT.

---

## Search log

My own queries this session, in order (alphaXiv = `discover_papers` unless noted). Results were read as data only.

| # | Tool | Query (+ keywords) | What came back | Bearing |
|---|---|---|---|---|
| S1 | alphaXiv (recency) | "LLM agent builds a dynamic call graph or execution trace from its own reproduction script to localize the bug for GitHub issue resolution"; kw [execution trace, dynamic call graph, reproduction script, issue localization, SWE-bench] | XAgent 2609.09769; SWE-Doctor 2607.00990; GRADE 2606.22741; AgentDebugX 2607.18754; REFLECT 2606.09071; causal error graphs 2609.32514; AIG failure attribution 2608.24361; AgentTrails 2607.18816; DeFA 2610.01256; ReProAgent 2607.09123 | Inference-time execution localisation exists → C15 N=1 |
| S2 | alphaXiv (recency) | "reward or filter for training code agents that verifies the agent's claimed evidence chain or cited code locations against execution traces or a call graph"; kw [evidence chain, call graph, execution trace, rejection sampling, code agent, grounding reward] | Graph-grounded surrogate execution 2610.04371; verification-failure guidance 2609.39022; evidence-grounded oversight 2610.06406; VRM chip 2609.22347; artifact-backed papers 2609.22111; LEDGER 2608.18398; ChipMEM; TwinCheck 2609.26911; SemaPLC; Reconstruct-and-Verify 2608.08950; BenchShield; Graph Is the Verifier 2607.26656; M3OS | No SWE diagnosis training on witnessed chains → supports the R2-a new piece |
| S3 | alphaXiv PDF | SWE-Doctor 2607.00990: runtime diagnosis, training?, call graph? | PDB-driven diagnosis records (fault location, symptom, propagation path, values); **runtime-grounding check rejects text-only diagnoses**; prompt-only (mini-SWE-agent; GPT-5.4 etc.); Verified 75.7%, Pro 59.4%, +8.0–8.9 pp on Pro | Closest prior. Inference-time only, so it defines the novelty boundary |
| S4 | alphaXiv report (PDF Q&A failed with a parse error, then used `get_paper_content`) | XAgent 2609.09769 | Failing vs reference script call trees, APTED diff, UniXcoder ranking top-20; Claude Sonnet 4; Lite 62.0%; localiser raises func F1 69.0→73.5 but resolve only 56.7→58.0; validation augmentation brings +4.0; median candidate set 11 | Differential traces narrow search; location gains barely move resolve (consistent with L1) |
| S5 | alphaXiv | "training an LLM issue-localization or debugging agent with supervision or rewards derived from execution traces / dynamic call graphs of failing tests"; kw [execution trace, dynamic call graph, fault localization, reinforcement learning, supervised fine-tuning, failing test] | BoostAPR 2605.09134; Socratic-SWE 2606.07412; ADI 2604.24212; C2C 2609.38402; REFLECT; FailForge 2608.08570; CausalFlow; DHRCL 2607.26457; Repair-or-Resample; ASAD | No witnessed-chain training for localisation/diagnosis |
| S6 | alphaXiv | "selecting among multiple sampled bug localization candidates using code graph structure, consensus or self-consistency, or a label-free verifier"; kw [self-consistency, fault localization, call graph, reranking, verifier, SWE-bench] | SemNav 2609.31176; ARISE 2605.03117; GraphLocator 2512.22469; SHERLOC 2606.24820; RGFL 2601.18044; C2C; Verification Trap 2610.05170; transductive test generation 2609.12489; ExeCRE; ArcticSwarm | Selection is crowded → R2-c killed |
| S7 | alphaXiv PDF | ADI / FramePilot 2604.24212 | Function-level debugger with call-tree navigation; inference-only; Verified 63.8% (Claude-3.7); **ADI invoked 71–73% (Claude) vs 28–31% (Qwen3-32B)** | Risk: open ~30B models under-use runtime tools |
| S8 | alphaXiv PDF | GraphLocator 2512.22469 | Causal issue graph over a typed repository graph; prompt-only (Claude-3.5); handles symptom-to-cause and one-to-many | Causal-chain inference exists, so the novelty must come from training. Also weakens R2-d |
| S9 | alphaXiv (after 2026-01-01) | "fine-tuning or RL for SWE agents where trajectories are filtered or rewarded by whether the root-cause diagnosis is grounded in runtime evidence"; kw [root cause, reproduction, rejection sampling, SWE-bench, process reward, debugging] | CRR 2609.33875; SWE-Doctor; FC-SWE 2610.07898; TRACE 2609.10315; SWE-MILE 2609.32631; Verifier errors in RLVR 2609.35677; My FAULT 2610.01161; LongRCA 2608.15242; ReProAgent; FLARE 2609.23808; Socratic-SWE; SWE-Prime 2608.27449 | None filters on runtime-grounded chains → supports the R2-a new piece |
| S10 | alphaXiv (historical) | "PageRank or graph propagation over call graph combined with test coverage or execution spectra for fault localization; graph-based aggregation of multiple LLM localization candidates"; kw [PageRank, spectrum-based fault localization, call graph, coverage, LLM] | CoSIL 2503.22424; LocAgent; SieveFL 2605.13491; SemLoc 2603.29109; graph-retrieval multi-agent FL 2409.13642; ConFL; LIDL; FL-RAG 2509.20552; HyperFL; execution features in SFL 2606.30324 | Graph-propagated SBFL is crowded. PRFL (ISSTA'17) is recalled from memory, not searched (INFERRED) |
| S11 | WebSearch (extended) | `training SWE agent localization reward verified against execution trace call chain "fail-to-pass" dynamic call graph supervision 2026 arXiv` | Socratic-SWE; Teaching Agents to Code Reliably 2610.03984; Verification Horizon 2606.26300; SWE-TRACE 2604.14820; Fail-Fast 2608.03222; SWE-RM 2512.21919; R4P 2510.22775; category-aware experts 2609.23377 | No match for the new piece |
| S12 | WebSearch | `openreview ICLR 2027 issue localization evidence chain reward call graph LoRA rejection sampling` | Catalyst-papers survey; credit assignment with resets 2605.25507; ToolTrain 2508.03012; GraphLocator mirror; ICLR 2026/2027 pages; "Chaining the Evidence" (deep-search rubric rewards, not code) | No concurrent OpenReview hit (submissions likely not yet public: GUESSED) |
| S13 | alphaXiv PDF | R4P 2510.22775 | Group-wise patch verifier (RL, Qwen2.5-Coder-32B, 72.2% acc); Mini-SE follows symptom → root-cause entities with no chain verification | Not a chain verifier |
| S14 | alphaXiv | "fine-tuning LLMs to predict runtime call structure or execution traces of code improves downstream bug localization or issue resolution"; kw [TraceEval, execution trace prediction, call structure, fine-tuning, transfer] | TraceEval 2605.11006; CodeAlchemy; Neural Debugger 2603.09951; WitnessGym 2609.36635; ReProAgent; Know Before Fix 2607.11111; semantic drift 2607.18550; XAgent; SkillForge; trace-guided debate 2604.02647; TraceCoder 2602.06875; execution-first traces 2607.29175 | No transfer-to-localisation result. Trace labels are only used for eval/trace prediction |
| S15 | alphaXiv | "LLM agents produce partial patches on multi-location fixes; using call graph or co-change structure to find the remaining edit locations"; kw [multi-hunk, multi-location, partial patch, co-change, call graph, program repair] | MultiFixer 2607.26591; PhoenixRepair; SiblingRepair 2605.06209; CT-Repair 2607.12605; ARISE; multi-file exploration 2606.11976; EviACT 2605.27238; RepoAtlas; Kozuchi; multi-hunk dynamics 2511.11012 | R2-d novelty is weak |
| S16 | alphaXiv PDF | CT-Repair 2607.12605 | Temporal Execution Graph + Joern CPG behind query tools; inference-only; Defects4J (Java), perfect FL; w/o TEG −8.6% repaired | Execution *graph* as inference evidence exists. Training use is absent |
| S17 | alphaXiv PDF | EviACT 2605.27238 | RED failure trace → code-graph expansion → compile and RED/GREEN gates; inference-only; failures: misloc 46.5%, mis-edit 38.5% | Same boundary |

Local computations (this session, CPU): the L1/L2/L3 tables from `results/runs/r3_v5_graph/behaviour_per_task.json` and `results/runs/r{1,2}_*/per_task.jsonl`.

Inherited searches (not re-run): the four notes' logs (code_graphs_swe Q-log 1–11, general_graph_reasoning 1–11, post_training_navigation 1–21, evaluation_benchmarks 1–13) cover LocAgent, CodeAnchor, RepoNavigator, A²Agent, SweRank, OrcaLoca, GRA, SOG-R1, PathRouter, EoG, GraphCert, VulAgentRL, P2T, CRR, CodeScout, TraceEval, RepoReasoner. Their verdicts were used for C1–C14.

Not searched (budget): ACL Anthology/ICSE 2027 proceedings; GitHub code search for witnessed-chain rewards in verl/SkyRL; full text of Socratic-SWE and FailForge.

---

# Round 4 — final judgement (8 Oct 2026, after `prior_art_redteam.md` and `cross_domain_ideas.md`)

## R4.0 What changed

**H1 lost its headline novelty.**
- Comet-9B (2609.34359) already trains a repo-level SWE agent (rejection-sampled SFT + GRPO) to write an "Evidence-Grounded Bug Propagation Chain".
- I read its judge prompt (App. G.1.8; T8 in the log below). The judge sees the gold patch, the official tests and the agent's tool evidence, and scores "propagation correctness" and "repo grounding" on a rubric.
- It does **not** check traces, call graphs or cited edges.
- Its task is *regression-test writing*, not fixing, and nothing in it is multi-location.

What H1 still owns is narrow: the verifier is programmatic and graph-structured instead of an LLM judge. That is N≈1.5.

**H2 is dead as a method.** CodePlan and SynFix cover it (closeness 3), and Jia et al. and CG-Diff predict its premise fails.

**The "partial-patch interventions" idea (I2) is weaker than claimed.** My search found **IBugDC** (FSE 2024: "Towards Effective Multi-Hunk Bug Repair… Indivisible Bugs"; T6, T10). It enumerates hunk subsets on 281 Defects4J multi-hunk bugs and finds 139 *divisible* (independent bugs with different failures). So "apply subsets of the gold patch and run tests" is known (Java, hunk level, used for benchmark cleaning).

**Still open after my searches:**
1. *Predicting* fix-set causal structure (necessity, divisibility, joint necessity) from **execution-graph structure** instead of enumeration. T9 found nothing. PolyHunk (2511.11012, T7) explicitly lists "call-graph proximity" as future work: its hunk divergence is lexical/AST/file only.
2. Using interventional fix-set labels as a **training verifier for an LLM agent**. T1, T11: no hit. RECAP (2608.13292, T3) trains with DPO in the *opposite* direction: gold ≻ inflated patch, for concision.

## R4.1 Kill round 4 (attack the new candidates)

Scores 0–3. P = hidden-test payoff + graph-ML paper payoff.

| # | Candidate | N | F | P | Fa | Verdict | One-line reason |
|---|---|---|---|---|---|---|---|
| A | H1-narrow: executed-graph chain verifier vs LLM judge, for diagnosis RFT | 1.5 | 2 | 2 | 3 | survive | The delta over Comet is only the verifier type. It must beat a judge arm. |
| B | I2 alone: partial-gold-patch necessity labels | 1 | 3 | 1 | 3 | **KILL as headline** | IBugDC did the enumeration (FSE'24). Measurement only. Keep as the label generator. |
| C | I1 fix-motif census | 1.5 | 3 | 1 | 3 | **KILL as headline** | Hunk4J/PolyHunk spatial classes + CG-Diff connectivity. Measurement (vetoed alone). Fold into analysis. |
| D | I3 node-dropping incomplete-patch negatives | 2 | 2 | 2 | 2 | survive → merge | No prior uses partial-gold negatives (T11). Needs I2 labels. Over-editing risk (RECAP, "Models Edit Too Much"). |
| E | I4 edit-ablation credit | 1.5 | 2 | 1 | 2 | **KILL** | Close to CRR/leave-one-out. Gemma's dead-zone patches are often single-edit or empty, so there is nothing to ablate. |
| F | I10 as-needed vs systematic (seen vs never-seen missed gold) | 2 | 2 | 2 | 3 | survive (diagnostic) | Cheap and decisive about *which* method. Graph content is thin. Needs raw traces (GUESSED retrievable). |
| G | I5 dominator fix-site | 2 | 3 | 1 | 3 | merge | Same tracer. Becomes a sub-analysis of the top pick. |
| H | I16 test-time do() interventions by the agent | 2 | 1 | 2 | 2 | **KILL** | Gemma rarely even reproduces (6/47). Open ~30B models under-use runtime tools (ADI, S7). Does not fit ~5 min. |
| I | I6/I7 resistance/Steiner difficulty, curriculum | 2 | 2 | 0–1 | 2 | **KILL** | Measurement (vetoed). 2608.18280 reaches AUC 0.863 without graphs, a hard baseline. The curriculum needs 2 trainings. |
| J | I14 GNN → LoRA co-edit distillation | 2 | 1 | 1 | 2 | **KILL** | The student never sees the graph. Jia/CG-Diff say static call features are weak co-edit predictors. |
| K | **NEW: Interventional Fix Graphs (A + B + D + G + one new piece)** | 2–2.5 | 2 | 3 | 3 | **survive** | See R4.2. |

## R4.2 Kill round 5 (strengthen K, then attack it)

**K = "Interventional Fix Graphs" (IFG).** For each training task, build two objects:
- the **executed call graph** of the failing (F2P) tests on the buggy commit: directed caller→callee edges actually seen at runtime;
- **interventional labels** on the gold functions, from function-level partial-patch do() runs: necessary, sufficient subsets, divisible components, and jointly necessary pairs (hyperedges).

**The new piece:**
1. The *causal structure of a fix is predictable from the executed graph*: jointly necessary functions co-occur on the same failing paths and sit close in executed hops, while divisible components sit on disjoint failing-test subgraphs. This beats static call-graph distance, file proximity (Hunk4J/PolyHunk) and hunk divergence.
2. IFG becomes a **programmatic verifier** for post-training:
   - chain reward: the stated chain's edges must exist in the executed graph, and its endpoint must be an *interventionally necessary* node, not just "a gold node";
   - completeness: the edit set must hit a necessary node in every failing-test component;
   - node-dropped negatives come from necessary nodes.

This replaces Comet's gold-patch LLM judge and gives multi-location fixes partial credit that a single F2P pass/fail cannot.

**Attacks**

| Attack | Answer | Residual risk |
|---|---|---|
| "IBugDC already enumerates subsets." | We use enumeration only to *label* (function level, Python repo tasks). The claim is that executed-graph structure *predicts* the labels, plus their use as an agent-training verifier. Neither was found (T9, T11). | Medium. A reviewer may call the labelling engineering. |
| "Comet already rewards propagation chains." | Comet is a judge with gold-patch access, a test-writing task, no multi-location, no edge check. TraceLift shows judge rewards saturate (95.5% perfect scores). The paper must include a Comet-style judge arm. | High if the judge arm wins. |
| "The hitting-set property is a tautology." | True. Any changed outcome must touch an executed function, so "each failing test's path contains an edited node" is not a claim. The claim is **which pairs are *jointly* necessary** and whether execution structure predicts that better than static/spatial baselines. | — |
| "CG-Diff: changed functions are mostly not call-connected." | That is the static call graph over PRs (median CPR 0.3). Executed failing-path co-occurrence is a different relation (2606.21187 shows runtime coupling adds over history in JavaScript impact analysis, T12). Unknown for Python bug fixes. That is exactly the kill test. | High. It may fail. |
| "n is too small (≈20–45 multi-function tasks)." | Pairwise analysis gives ~100–200 gold-function pairs, clustered by task. Scale with injected multi-function bugs (SWE-smith "combine"-style, INFERRED feasible) for the training set. | Medium. |
| "Hidden test has no traces." | Test-time behaviour only needs the agent's own repro/test runs (tracebacks). No graph files or skills are needed. The verifier is training-time only. | Gemma rarely reproduces. Stage B checks whether evidence helps at all. |
| "Feasible by Nov 5?" | CPU premise ~1.5 days. Stage B ~5 GPU-h. RFT data generation ~20 GPU-h (two weeks of quota). One LoRA training. | Tight. Only the premise and upper-bound results are guaranteed by Nov 5. |

Verdict: **K survives as the top pick.** A (H1-narrow) becomes its single-location special case and the runner-up. F (I10) stays third as the cheap method-deciding diagnostic.

---

## R4.3 Final ranking

### 1. TOP — Interventional Fix Graphs (IFG): execution-graph structure predicts the causal structure of fixes, and serves as a programmatic training verifier

**Falsifiable claim**
- **(a) Premise, graph-central.** On multi-function gold fixes, executed-graph features (co-coverage by the same failing F2P tests, executed hop distance, shared failing path) predict pairwise **joint necessity** with AUROC ≥ 0.75. They beat the best of these by ≥ 0.10 AUROC:
  - static provided/repaired call-graph hop distance;
  - same-file / Hunk4J proximity class;
  - hunk divergence.
- **(b) Method.** An IFG-verified RFT LoRA, compared at matched sample count with outcome-only RFT and with a Comet-style judge-filtered RFT, gives:
  - **+5 pp** resolve on held-out multi-function tasks;
  - **≥ +2 pp** overall;
  - **+20 pp** chain hop-precision.
- All effect sizes are **GUESSED**.

**Known pieces + new piece**

| Piece | Source | Status |
|---|---|---|
| Subset enumeration to find divisible/indivisible multi-hunk bugs | IBugDC, FSE 2024 (T6, T10) | known (Java, hunks, benchmark cleaning) |
| Multi-hunk difficulty = lexical/AST/file divergence and spatial proximity; call-graph proximity is "future work" | Hunk4J 2506.04418 (T4), PolyHunk 2511.11012 (T7) | known; opening stated by the authors |
| Static call connectivity of changed functions in PRs (median CPR 0.3) | CG-Diff 2609.32057 | known |
| Executed call structure as ground truth | TraceEval 2605.11006, RepoReasoner 2607.25996 | known (eval/trace prediction) |
| Execution graphs as inference-time repair evidence | CT-Repair 2607.12605, XAgent, SWE-Doctor, EviACT (S3–S17) | known, inference-only |
| LLM-judge reward on propagation chains | Comet-9B 2609.34359 (T8) | known; judge-based, test-writing |
| Static-graph evidence reward | VulAgentRL 2607.26656 | known |
| Gold ≻ degraded patch preference | RECAP 2608.13292 (T3), reverse direction | known |
| Node-dropping augmentation | GraphCL (INFERRED) | known |
| Offline advantage-weighted SFT | A²Agent 2608.29831 | known |
| **NEW: (i) executed-graph prediction of interventional joint-necessity / divisibility of real fixes; (ii) its use as a programmatic verifier (necessity-checked chain endpoint + per-failing-component completeness + necessary-node-dropped negatives) to post-train an SWE agent** | — | not found: S2, S5, S9, S11, T1, T2, T9, T11, T12 (absence only; OpenReview not searchable) |

**VERIFIED vs GUESSED**

| Item | Status |
|---|---|
| L3: ≥3 gold functions → 2/45 resolved; 65 tasks have ≥2 gold functions | VERIFIED (regex-derived labels) |
| F2P/P2P/`test_patch` available; 54 tasks gold-pass in scorer-like env; ~109 in host env | VERIFIED (repo files, HQ) |
| How many of the 54 are multi-function (estimate ~20–25) | GUESSED |
| Function-level subsets apply cleanly (new helpers bundled with their first caller) | GUESSED, measured by K1 |
| Raw r2/r3 trajectories and agent patches retrievable from Kaggle outputs | GUESSED |
| Comet's judge does not check traces/graphs | VERIFIED (prompt read) |
| ~50% of multi-hunk bugs are divisible (Defects4J: 139/281) | VERIFIED as reported by IBugDC (snippet-level) |
| All effect sizes; AUROC thresholds hold | GUESSED |

**Cheapest kill experiment (CPU only, ~1.5 working days, 0 GPU-h)**

- **Data.**
  - Primary: the 54 scorer-faithful tasks.
  - Sensitivity: the ~109 host-env-stable tasks, which raises the multi-function count to ~45.
  - Gold patches, `test_patch`, F2P/P2P, the provided graphs, `scorer_site.tgz`.
- **Procedure** (one Kaggle CPU kernel extending Stage A):
  1. Run the F2P tests on the buggy commit (+`test_patch`) under a `sys.setprofile` tracer. Build the executed graph, and record per-test node sets.
  2. For each task with k ≥ 2 pre-existing gold functions, apply every function-level subset (2^k for k ≤ 5, ddmin above) and run F2P+P2P. Bundle added functions with their first using hunk. Mark subset runs failing on Import/NameError as *contaminated*.
  3. Labels: necessary set, minimal sufficient subsets, divisible components (IBugDC definition), pairwise joint necessity.
  4. Pairwise features:
     - executed: Jaccard of covering failing tests, executed hop distance, same failing path;
     - baselines: undirected provided-graph hop, direction-repaired hop, same file, Hunk4J class, hunk divergence;
     - null: a degree-preserving rewire of the executed graph.
  5. (If patches are retrievable) For Gemma's failed multi-function tasks, compute the share of *missed necessary* functions that lie on an executed failing path not hit by Gemma's edits.
- **Metric.**
  - Pairwise AUROC for joint necessity, with task-clustered bootstrap CI.
  - The distribution of structure types (core + incidental / conjunctive / divisible).
- **Power.** ~100–200 pairs clustered over 25–45 tasks gives an AUROC SE of ≈0.05–0.07. That is enough to see a 0.10–0.15 gap, not 0.05. This is pre-registered.
- **Kill criteria**
  - **K1 (labels):** <50% of multi-function tasks give uncontaminated subset outcomes, or fewer than 15 usable tasks in the 109 set. Labels cannot be obtained on real tasks. Pivot to injected multi-function bugs, or drop the idea.
  - **K2 (graph is decorative):** executed-graph AUROC < 0.70, or ≤ best baseline + 0.05, or no better than the rewired null. The graph headline is dead and we go to the runner-up.
  - **K3 (degenerate structure):** ≥80% of multi-function tasks are a single sufficient function plus incidental edits, **or** ≥80% are fully conjunctive with all pairs co-covered. There is no structural signal beyond F2P pass/fail, so the completeness half is dead.
  - **K4 (Gemma relevance, if patches available):** <30% of Gemma's missed necessary functions lie on executed failing paths. Trace-guided completeness cannot be learned from repro evidence, so the completeness half is dead.
- **Hours.** ~6 h code (tracer + subset runner, reusing `build_devset.py` env and `gold_graph_analysis.py` parsers), ~3–6 h Kaggle CPU (≤32 subsets × ~10 s × ~45 tasks), ~2 h analysis.
- **Next steps**
  - **All pass:** run Stage B (§ above) with a third arm, "+ necessary-set/component hint", ~6 GPU-h. Then the RFT ablation on injected + real multi-function tasks: outcome-only vs Comet-style judge vs static-graph verifier vs IFG verifier, at matched samples. Spend 1 LB submission on the best.
  - **K2 fails:** go to the runner-up (single-location chain verifier). Keep the interventional labels as an analysis section.
  - **K3 fails:** keep only per-test partial credit (HER-style relabelling, I12) as a training trick, with no graph claim.
  - **K1 fails:** run the 3rd pick and drop multi-location.
  - **Ambiguous:** widen to 109 tasks, then to injected bugs, before any GPU spend.

### 2. RUNNER-UP — Executed-graph chain verifier vs LLM judge (H1-narrow)

**Claim.** For single- and multi-location diagnoses alike, an edge-level verifier on the executed failing graph discriminates Gemma's good from bad diagnoses better than a Comet-style gold-patch judge (here played by Gemma). Predictions:
- AUROC for downstream fix success **≥ judge + 0.05**;
- within-task score ties **< 30%**, versus TraceLift's 95.5% saturation for judges.

If true, the RFT filter beats judge filtering by **+2–3 pp** resolve (GUESSED).

**Known + new.** Comet-9B (judge), TraceLift (judge saturation), VulAgentRL (static graph), SWE-Doctor (inference-time grounding). New: an edge-level executed-graph verifier as the *training* signal, compared head-to-head with a judge. No hit in T8/T9/S2/S11.

**Cheapest kill** (~1 day CPU for Stage A + ~7–8 GPU-h)
1. Stage A as in Round 3.
2. Sample k=4 Gemma diagnoses with a CHAIN field on the 54 tasks (~2–3 GPU-h).
3. Score each diagnosis with:
   - executed-graph hop-precision;
   - static repaired-graph hop-precision;
   - Gemma-as-judge with the gold patch, using Comet's rubric (~1 GPU-h).
4. Run one fixer per diagnosis on 27 tasks (108 runs, ~4.5 GPU-h).

**Metric:** AUROC of each score for fix success; tie rate within groups.

**Kill:** executed AUROC ≤ judge AUROC, or executed ties > 70%, **or** Stage A's K3 (static ≈ executed graph).
- Pass → it becomes the verifier inside the top pick's training.
- Fail → drop graph-verified training. Post-training reverts to tool-format/verification RFT for the leaderboard.

### 3. THIRD — As-needed vs systematic comprehension → structurally incomplete negatives (I10 + I3)

**Claim.** Classify each gold function Gemma missed on a failed multi-function task as never displayed or displayed-but-not-edited. Prediction: ≥60% of these misses fall in a single class. That class decides the method:
- **never-seen** → impact-triggered systematic exploration;
- **seen-not-edited** → node-dropped negatives (DPO/weighted SFT on full ≻ minus-one-necessary).

Predicted effect: +3 pp on the multi-function subset (GUESSED).

**Known + new.**
- Littman et al. 1986 / Soloway 1988 (citation VERIFIED, content INFERRED); Koenemann & Robertson (as cited by CG-Diff); PolyHunk (agent behaviour on multi-hunk bugs, no seen/unseen split).
- Delta debugging; GraphCL; RECAP (reverse-direction DPO).
- New: the seen/never-seen split of missed fix nodes in LLM agents, plus necessary-node-dropped negatives. T11 and S-G found none.

**Cheapest kill** (CPU, ~3 h + trace retrieval)
- **Data:** r2/r3 raw traces for the 65 tasks with ≥2 gold functions.
- **Metric:** share of missed gold functions whose `def` lines appeared in any tool output before `submit_patch`.
- **Kill:** neither class reaches 60% (no dominant mechanism), **or** misses are dominated by newly-added functions (no node to "see").
- **Next:**
  - never-seen dominant → merge into the top pick's completeness reward;
  - seen dominant → run I3 zero-shot discrimination (~1.5 GPU-h; kill if Gemma already discriminates ≥ 90%).

---

## R4.4 Honest verdict on paper novelty for a graph-ML audience

- **Where graph reasoning is central (top pick, only if K2 passes).**
  - The paper's object is a graph: the directed executed call graph of failing tests. Its labels are interventional hyperedges: which node sets must change *jointly*.
  - The core result is a structure-prediction result. Can graph structure (co-coverage, executed distance, cut and dominator position) predict causal fix structure better than static and spatial structure, with a degree-preserving rewire null?
  - The method uses that structure as the verifier that trains the LLM.
  - Remove the graph and both the claim and the verifier disappear. That is the test of "central, not decorative".
- **Where it would be decorative.**
  - If K2 fails, executed structure adds nothing over file proximity or static hops. A verifier built on it would then be a well-engineered SE reward, and graph-ML judges would see a reward function, not graph reasoning.
  - H1-narrow alone has the same weakness: graph content = "check edges exist". That is VulAgentRL-level graph use.
- **Prior-art exposure.** IBugDC (enumeration), Comet-9B (chain reward), and Hunk4J/PolyHunk/CG-Diff (multi-hunk structure) must all be cited up front. The defensible novelty is the *junction*: interventional labels × executed-graph structure × agent post-training. No single component is new.
- **Realistic ceiling.**
  - With ≤45 real multi-function tasks plus injected bugs, and one LoRA run by Nov 5: a solid workshop or short paper with a clean structural finding and a modest training ablation.
  - A main-track graph-ML acceptance would need the K2 effect to be large (≥0.15 AUROC) and replicated on injected bugs across ≥30 repos.
- **Leaderboard honesty.** The largest LB levers remain tool-format reliability, verification and budget (findings). IFG targets the 35% multi-function pool, where Gemma solves 4%. A few tasks gained there are invisible at LB SE ≈ 2.5 pp.

---

## Search log — Round 4 additions (my own queries; R1–R14 are in `prior_art_redteam.md`, S-A..S-P in `cross_domain_ideas.md`)

| # | Tool | Query (+ keywords) | Key results | Bearing |
|---|---|---|---|---|
| T1 | alphaXiv | "analysis of SWE-bench gold patches: which hunks or functions are actually necessary for the fail-to-pass tests, applying subsets of the reference patch…"; kw [gold patch, SWE-bench, hunks, fail-to-pass, partial patch, necessary] | RECAP 2608.13292; SWE-Gate 2609.04167; PAIChecker 2607.28587; MPC-Patch-Bench; **Hunk4J 2506.04418**; "Solved Issues" 2503.15223; security backporting; Single Patch Not Enough 2607.01597; Reconstruct-and-Verify 2608.08950; Probe-to-Generate 2604.01518; ProgramDistill; SemaDiff 2607.13111 | No necessity labelling of SWE gold patches |
| T2 | alphaXiv | "multiple fault localization as minimal hitting set or vertex cut over failing test executions… applied to LLM repair or patch completeness"; kw [hitting set, multiple faults, spectrum-based reasoning, Barinel, model-based diagnosis, patch completeness] | Minimal repair families via counterfactual replay 2608.29228 (multi-agent messages); EDGE 2609.01360; AIG 2608.24361; Who Broke the System 2607.07989; telemetry faults; SFL execution features 2606.30324; MBD for C 2512.02898; APR experimental settings 2609.17993; Model or Harness 2607.28802; SHERLOC | "Jointly necessary repair families" exists for agent *messages*, not code. The hitting-set framing is classic (and tautological here) |
| T3 | alphaXiv PDF | RECAP 2608.13292 | SFT+DPO refiner, gold ≻ inflated patch (concision); untangling/delta debugging are baselines; +4 to +42 resolved on Verified | Reverse direction from I3. Cite |
| T4 | alphaXiv PDF | Hunk4J 2506.04418 | Divergence (lexical/AST/file) + spatial classes; no call-graph measure; refers to "hunk divisibility" prior work | Opening for the graph claim |
| T5 | alphaXiv (historical) | "divisibility of multi-hunk patches… partial fixes passing some failing tests"; kw [multi-hunk, divisible, partial fix, Defects4J, independent hunks] | Hunk4J; PolyHunk 2511.11012; MultiFixer; SiblingRepair; MultiMend 2501.16044; HAFixAgent 2511.01047; BoostAPR; DEAR 2205.01859; Hercules 1906.08903 | — |
| T6 | WebSearch | `"multi-hunk" patches "divisible" OR "divisibility" hunks independently fix failing tests Defects4J empirical study` | **IBugDC, FSE 2024** ("Towards Effective Multi-Hunk Bug Repair… Indivisible Bugs"); MultiMend; Hunk4J (ASE'25) | Kills I2's "no one enumerates" novelty |
| T7 | alphaXiv PDF | PolyHunk / "Beyond Accuracy" 2511.11012 | 404 bugs (32 SWE-bench Python); 4 agents; divergence/proximity predict failure; **call-graph proximity listed as future work** | Supports the graph claim's opening |
| T8 | alphaXiv PDF | Comet-9B 2609.34359, judge prompt G.1.8 | Judge gets issue, reasoning file, gold patch, official tests, agent tool evidence; rubric scores; no trace/edge check; SWT test-writing task | Confirms the H1 delta = verifier type |
| T9 | alphaXiv | "predicting whether hunks… are independent or jointly necessary using test coverage, execution traces or dependency graphs instead of enumerating…"; kw [indivisible bugs, multi-hunk, test coverage, dependency, IBugDC] | MultiFixer; PolyHunk; Hunk4J; CT-Repair; trace-guided debate 2604.02647; COMPASS 2609.39783; SiblingRepair; CausalRepair 2608.10613; TraceFixer 2304.12743; MultiMend | **No predictor of joint necessity from execution graphs** → supports the IFG new piece |
| T10 | WebSearch | `IBugDC "indivisible" multi-hunk bugs enumeration failing tests FSE 2024 …` | 281 D4J multi-hunk → 139 divisible; 249 isolated bugs (105 multi-hunk) | Prior anchor (~50% divisible) |
| T11 | alphaXiv | "preference optimization or contrastive training for code repair agents using incomplete or partial patches… as negative examples"; kw [DPO, incomplete patch, negative examples, patch completeness, SWE agent, multi-location] | Counterexample-supplemented sketches 2607.15854; ContraFix 2605.17450; ComPass 2602.07561; Co-Evolving Agents (hard negatives) 2511.22254; Single Patch 2607.01597; SWE-Gate; FC-SWE; SWE-Pruner Pro; MASS-DPO 2605.10784 | No necessary-node-dropped negatives → supports I3 |
| T12 | alphaXiv (historical) | "dynamic coupling or runtime call graph from test executions used to predict which code entities must change together…"; kw [dynamic coupling, runtime call graph, co-change, change impact analysis, test execution] | JS history + runtime change impact 2606.21187; IA + PDG 2607.23355; CG-Diff; CodeFlow 2408.02816; DiffTestGen 2607.16024; OpenStack change deps 2508.05034; ripple/impact visions | Runtime coupling for impact analysis exists (JS, not fix necessity, not agent training) → closeness 1–2; cite |
