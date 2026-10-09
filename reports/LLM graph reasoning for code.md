# Execution graphs, not code maps, should decide the fix

**Bottom line.** The literature on LLM reasoning over large graphs is crowded wherever our project has no bottleneck and thin where it does. Code-graph localisers, KG agents and graph process rewards all mostly improve *finding* the right place. Our own data on Gemma-4-31B-W4A16 shows that finding is not the binding constraint: resolve is **27% when the diagnosis names the right function and 28% when it names the wrong one**. The dead zone is **multi-function fixes: 2 of 45 tasks with three or more gold functions were resolved**. An adversarial critic ran five kill rounds over about 30 candidates. Most died to prior art or to that bottleneck. Execution-witnessed chain rewards (H1) were narrowed by Comet-9B. Graph co-change completion (H2) was killed by CodePlan and SynFix. Partial-patch interventions (I2) were pre-empted by IBugDC. The survivor is **Interventional Fix Graphs (IFG)**. Its claim is that the executed call graph of the failing tests predicts which edited functions of a fix are *jointly necessary*, better than static, file or hunk proximity. That structure then becomes a programmatic verifier for post-training the agent. No single piece is new. The junction is, as far as about 50 logged searches can show. The cheapest experiment that could kill it is CPU-only: **about 1.5 working days, 0 GPU-hours**, with four pre-registered kill criteria (K1–K4). It has **not been run**. A go/no-go from Shrey is the next step. Honest framing for later sections: even if IFG survives, it is a workshop-to-short-paper contribution, and the largest leaderboard levers (tool-format reliability, verification, budget) lie elsewhere.

---

## Our bottleneck is conversion, which most of the literature does not target

Three facts computed on CPU from the repo's own run files frame the whole comparison. They are VERIFIED as arithmetic. The labels are noisy, because gold functions come from regex parsing of the gold patch.

| Fact | Numbers | Source |
|---|---|---|
| **L1: a function-level hit does not predict resolve** | In the v5 localiser's diagnoses, the file is right 88/109 (81%) and the function is right 52/109 (48%). Resolved given a function hit: **14/52 (27%)**. Given a miss: **16/57 (28%)**. | `results/runs/r3_v5_graph/behaviour_per_task.json` |
| **L2: touching a gold file is close to necessary** | v2 patches that touch a gold file resolve 24/80. Patches that don't: 2/49. | `results/runs/r2_*/per_task.jsonl` |
| **L3: multi-function fixes are a dead zone** | Resolved by number of gold functions: 1 → 19/57 (33%); 2 → 5/20 (25%); **≥3 → 2/45 (4%)**. 65 of 129 tasks have ≥2 gold functions. | `results/runs/r{1,2}_*/per_task.jsonl` |

The dominant v2 failure is **"wrong fix, never tested" (47/129)**, and gold lines were seen in 80% of v2 tasks. So any method whose payoff is better function-level localisation aims at a non-bottleneck for this model.

XAgent shows the same thing at frontier scale. Its differential-trace localiser raised function F1 from 69.0 to 73.5 but moved resolve only from 56.7% to 58.0% ([XAgent](https://www.alphaxiv.org/abs/2609.09769)).

Constraints for the competition and paper are in `docs/HQ.md`:
- fixed 4-bit 31B model, LoRA only, 32k context, about 5 minutes per task;
- a declarative YAML submission, so tool outputs cannot be altered;
- skills unverified on the scorer (v6 scored 0.00);
- hidden repos are private;
- about 30 L4 GPU-hours a week, $50 cash.

---

## Four literatures compared: where they help and where they stop

**Code graphs for SWE agents.** The field has moved from "big typed graph plus many tools" toward fewer tools, trained use, and passive structure.

- **RepoNavigator (ICML 2026)** trained a single `jump` tool with GRPO. Adding three more structural tools under RL cut function IoU from **24.28 to 13.71** ([RepoNavigator](https://arxiv.org/abs/2512.20957)).
- **CodeAnchor (ISSTA 2026)** injects call-graph tags passively, through grep. Matched head to head, grep-first Codex beat graph-agent LocAgent, **Func@5 0.832 vs 0.595**. An optional call-graph tool saw low use ([CodeAnchor](https://arxiv.org/abs/2606.26979)).
- **SweRank (ICLR 2026)** uses no graph at all. On issues with 2–3 gold functions, LocAgent fell from 82.0 to 47.1 while SweRank's embedder held 77.1 ([SweRank](https://arxiv.org/abs/2505.07849)).
- **A²Agent** keeps graph tools and adds a per-turn discovery reward and a discovery–commit reward at 4B with LoRA r16. It finds the discovery–commit gap behind **over 40% of failures** ([A²Agent](https://www.alphaxiv.org/abs/2608.29831)).
- On our exact base model, **INT4 quantisation doubles tool-name hallucination, 19.5% → 38.3% of calls**, while the headline score stays flat ([Flat Score, Amplified Failures](https://arxiv.org/abs/2607.27275)).

Lesson: on ≤32B open models, the gains come from training and evidence presentation, not from graph richness.

**General LLM graph reasoning.** The strongest 2026 result is a warning. On KG question answering, a graph-free control comes within **+0.3 to +1.9 pp** of the graph agent. The gain comes from *selective access*, not topology. The agentic advantage reverses once tool-call failure exceeds a few percent ([GRA](https://www.alphaxiv.org/abs/2608.15834)). At scale, global ranking before agency wins: Agent+BM25 reached **69.4%** where the raw file agent ran out of budget ([BM25 Wins at Scale](https://www.alphaxiv.org/abs/2607.26497)).

Trained KG agents are efficient but need an SFT warm start. SOG-R1 answers in **2–5 batched, direction-typed calls**, and RL-only training collapses to 63.0 vs 86.1 with SFT ([SOG-R1](https://www.alphaxiv.org/abs/2607.18481)).

A degree-preserving rewire is the clean proof that structure is used. In CNY it dropped walk accuracy from 76.6 to 73.0 ([Call Neighbours Yourself](https://www.alphaxiv.org/abs/2608.29588)). On incomplete graphs, a theorem shows **hard grounding cannot both keep true-but-unobserved paths and reject false ones**, so path rewards must be soft ([Incomplete Graph Evidence](https://www.alphaxiv.org/abs/2606.30247)).

**Post-training for graph navigation and evidence-backed diagnosis.** Process terms on top of outcome rewards help consistently but modestly, by **about 3–9 points**:
- Removing EoG's path reward cut CWQ F1 from 73.9 to 70.8 ([EoG](https://www.alphaxiv.org/abs/2602.21728)).
- PathRouter retains **95.7%** across datasets vs 85.8% for Graph-R1 ([PathRouter](https://www.alphaxiv.org/abs/2606.16409)).
- VulAgentRL's static CPG node-ID F1 term added **+8.9 pair-correct** ([Graph Is the Verifier](https://www.alphaxiv.org/abs/2607.26656)).
- GraphCert's certification filter is worth **0.612 → 0.722 F1** ([GraphCert](https://www.alphaxiv.org/abs/2609.38798)).

For SWE agents:
- P2T's groundedness check ("every cited symbol must already have been observed") is worth **7.2 points**, within an overall +10.8 Pass@1 ([P2T](https://www.alphaxiv.org/abs/2605.21996)).
- Counterfactual forks work (CRR: **41.7% vs 36.4%** for GRPO) but cost **321 vs 188 H100-hours** ([CRR](https://www.alphaxiv.org/abs/2609.33875)). That is out of reach for us.
- Efficiency-only process rewards can backfire. GraphEcho's PAPT gained 41.5 pp on synthetic tasks but **lost 3.6 pp on SciFact transfer** ([GraphEcho](https://www.alphaxiv.org/abs/2609.17695)).

What fits our budget is rejection-sampled or advantage-weighted SFT generated offline with vLLM, which is A²Agent's objective.

**Evaluation.** Structural benchmarks with execution-derived ground truth now exist:
- **TraceEval**: 10,583 programs, with a pipeline that turns any repo into instances ([TraceEval](https://arxiv.org/abs/2605.11006)).
- **RepoReasoner**: call-chain prediction scored against pytest traces. Models are high-precision, low-recall (GPT-4.1-Mini **81.4% precision vs 31.8% recall**) ([RepoReasoner](https://www.alphaxiv.org/abs/2607.25996)).

No benchmark scores an agent's *stated* causal chain against a graph (evaluation search #10). Power is the binding constraint. A paired 4 pp difference at 20% disagreement needs about **981 independent items** (evaluation-note arithmetic, following [Miller, Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640)). Our 129 tasks and the ~120-task hidden leaderboard (SE ≈ 2.5 pp) cannot resolve such effects. Claims need many repos.

| Strand | What reliably works | Where it stops | Implication for us |
|---|---|---|---|
| Code graphs for SWE | Minimal tools, RL on dense localisation rewards, passive structure | Optimises localisation; graph tools go unused | Localisation is not our bottleneck (L1) |
| General graph reasoning | Selective access, global ranker then local expansion, soft grounding | Topology adds little over selective access | Any graph claim needs graph-free and rewire controls |
| Post-training | Verifiable process rewards over outcome, SFT before RL, groundedness filters | All verify *location or observation*, not causal structure | Room exists for a structural verifier of the fix |
| Evaluation | Execution-witnessed call structure as ground truth | No chain-faithfulness metric; little power | Execution traces are the non-circular oracle |

---

## Round 0: four literature sweeps produced the first gap list

**Findings.** The four sweeps proposed these gaps, each backed by its own search log:

| Gap | Backing searches |
|---|---|
| G1. Degraded or undirected graphs as a controlled variable, and graph repair | code-graph log #2, #5 |
| G2. Graph tools vs no graph under identical post-training ("No papers found") | code-graph log #10 |
| G3. Graph localisation on quantised ~30B models | code-graph log #6, #8 |
| G4. Dense entry retrieval plus directed expansion | — |
| G5. Discovery–commit reward plus passive graph injection | — |
| Rewarding an agent's claimed call-graph chain ("No papers found") | post-training log #9, #16 |
| Graph-privileged on-policy self-distillation for code | post-training log #11 |
| Chain validity as a faithfulness metric | evaluation log #10 |

General-graph work added PathRouter-style shortcut routing, provenance collapse and rewire controls as "not found for code" (general-graph log #7, #9).

**Decision.** Hand all candidates to an adversarial critic. The critic scored each on novelty (N), feasibility by Nov 5 (F), payoff (P: hidden-test plus graph-ML paper) and falsifiability (Fa), each 0–3. Any 0 kills, and a user veto kills.

**Next step.** Re-judge every gap against L1–L3 and the hard constraints.

---

## Round 1: twenty candidates, eleven killed by veto, payoff or feasibility

**Findings.** Four killing patterns explain almost every death:

- **User veto.**
  - Graph repair as headline (C1).
  - Shortcut-aware C×P filter (C11), which is "structure over names" again.
- **Zero payoff on the hidden test.**
  - Degraded-graph study (C2) is vetoed "imperfect graph measurement".
  - Graph vs no-graph ablation (C3): agents make 0 graph calls in v5.
  - Quantised-localisation characterisation (C4).
  - Community "architecture map" (C14): file retrieval is already 81%, so it targets a solved problem.
- **Zero feasibility under a fixed YAML harness.**
  - A dense retriever cannot ship (C5).
  - Tool outputs cannot be altered (C6, C10).
  - The embedding tool accepts names, not vectors (C18).
- **Zero novelty.** Grep-built subgraph idioms (C16) are covered by RepoNavigator, CodeScout and FastContext.

**Decision.** Nine candidates survived:
- static-graph chain reward (C7);
- chain-faithfulness metric (C8);
- graph-privileged self-distillation (C9);
- bug-injection dev set (C12, required infrastructure, not a headline);
- free-fork credit (C13);
- agent-built dynamic trace (C15, crowded at inference time per searches S1, S4, S7, S10);
- structural verifier/selector (C17);
- **execution-witnessed graph as a training verifier (C19)**;
- one-to-many structural completion (C20).

**Next step.** Combine known pieces with exactly one new piece each, and attack again.

**What it taught us.** Three of the four sweeps' headline gaps (G1–G3) were real absences but measured the wrong thing. They are about how the agent searches, while L1 says the failure happens after search.

---

## Round 2: combinations, and the first top pick (H1)

**Findings.** The critic built six combinations:

- **R2-a (survive, top).** Execution-witnessed graph + static chain reward + chain metric + agent-built trace + injected-bug data. The new piece: the agent's stated chain is scored edge by edge against an execution-witnessed graph built offline, as a training-time filter for a diagnosis LoRA that runs graph-free at test time. Searches S2, S5, S9, S11 and S13 found no training method that checks a stated symptom→cause chain against witnessed execution.
- **R2-b and R2-e.** Merged into R2-a as its training ablation and its evaluation.
- **R2-c (killed).** Test-time graph self-consistency. Selection is crowded (S6: R4P, Agentless, ARISE) and it optimises location.
- **R2-f (killed).** Free-fork credit alone: out of GPU budget, with a localisation-only payoff.
- **R2-d (survive, runner-up).** Co-change closure for multi-function fixes, kept because it attacks L3 directly.

**Decision.** H1 = Execution-Witnessed Evidence Chains (R2-a). H2 = graph co-change closure (R2-d).

**Next step.** Specify falsifiable claims and a cheapest kill test for H1.

---

## Round 3: H1 specified with a two-stage kill test

**Findings.** The closest prior work is all inference-only:
- **SWE-Doctor**: runtime-grounded diagnosis records, with a prompt-level check that rejects text-only diagnoses; **+8.0–8.9 pp** on SWE-bench Pro with frontier models ([SWE-Doctor](https://www.alphaxiv.org/abs/2607.00990)).
- **CT-Repair**: a temporal execution graph; removing it costs 8.6% of repairs ([CT-Repair](https://www.alphaxiv.org/abs/2607.12605)).
- **EviACT** ([EviACT](https://www.alphaxiv.org/abs/2605.27238)).

The critic predicted "SWE-Doctor's check moved to training time" as the reviewer's attack. It also flagged a real risk: open ~30B models under-use runtime tools. Qwen3-32B invoked a debugger in **28–31%** of tasks vs **71–73%** for Claude-3.7 ([ADI/FramePilot](https://www.alphaxiv.org/abs/2604.24212)).

**Decision.** H1's kill test has two stages:
- **Stage A (CPU).** Do witnessed chains exist, are they non-trivial, and do they differ from the static graph? Kill criteria include witnessed ≈ static at ≥90% edge agreement.
- **Stage B (about 5–6 GPU-h).** An oracle-evidence upper bound: does handing Gemma the witnessed chain beat handing it the location alone?

**Next step.** Send H1 and H2 to a dedicated prior-art red team before spending anything.

---

## Round 4: the red team wounds H1 and kills H2; cross-domain ideas appear

**Findings on H1.** Red-team query R1 found **Comet-9B**, posted 28 Sep 2026, ten days before the search. It post-trains a repo-level SWE agent with rejection-sampled SFT and GRPO to write an **"Evidence-Grounded Bug Propagation Chain"**. That gives **+7.25 pp on SWE-bench Pro** from adding reasoning tasks to SFT, reaching 30.51% at 9B ([Comet-9B](https://www.alphaxiv.org/abs/2609.34359)).

The critic read the judge prompt (search T8). Comet's chain is graded by an **LLM judge that sees the gold patch and tests**. It checks no traces, call graphs or edges, and the task is regression-test writing, not multi-location fixing. H1's novelty dropped from N=2 to **N≈1.5**. What it still owns is only the verifier type: programmatic and graph-structured instead of a judge.

Two things help that defence:
- LLM-judge reasoning rewards saturate. TraceLift saw **573/600 (95.5%) perfect scores**, which makes them non-discriminative inside GRPO groups ([TraceLift](https://www.alphaxiv.org/abs/2605.03862)).
- No search found a dynamic-trace chain reward (R7, R10).

**Findings on H2.** It is dead as a method. CodePlan already propagates edits to graph-dependent code after each LLM edit ([CodePlan](https://www.alphaxiv.org/abs/2309.12499)). SynFix applies one-hop RelationGraph synchronisation on SWE-bench, **52.33% Lite** ([SynFix](https://preview.aclanthology.org/setup/2025.findings-acl.252); search R12, R14).

The empirical premise also looks weak:
- In 634K Java PRs, call dependency has **permutation importance 0.0082 vs 0.384 for co-change history** ([Jia et al.](https://www.alphaxiv.org/abs/2411.19099)).
- In 7,490 Python PRs, the median share of changed functions that are call-connected is **0.3** ([CG-Diff](https://www.alphaxiv.org/abs/2609.32057)).
- For **61%** of Defects4J sibling bugs, the failing tests do not cover all siblings ([SiblingRepair](https://www.alphaxiv.org/abs/2605.06209)).

**Findings from the cross-domain pass.** Sixteen transplants were proposed. The cross-domain note ranked **I2, partial-patch interventions** first (N3 P3 F3): apply every subset of the gold patch at function level, run the failing tests, and label necessity, sufficiency and joint necessity. Its novelty rested on search S-K ("No papers found" for patch minimisation and untangling) and S-A.

The critic then ran T6 and T10 and found **IBugDC (FSE 2024)**. It enumerates hunk subsets on 281 Defects4J multi-hunk bugs and finds 139 *divisible* (from search snippets; no URL was recorded in the notes). So enumeration itself is known: in Java, at hunk level, for benchmark cleaning.

Other transplants died too:
- Edit-ablation credit (I4): close to CRR and leave-one-out.
- Test-time do() interventions (I16): Gemma rarely reproduces (6/47).
- Resistance/Steiner difficulty (I6/I7): measurement only, against a graph-free baseline that already reaches **AUC 0.863** ([What Makes SWE Tasks Difficult](https://www.alphaxiv.org/abs/2608.18280)).
- GNN-to-LoRA co-edit distillation (I14): the student never sees the graph.

**Decision.**
- Keep I2 only as a label generator.
- Keep node-dropped negatives (I3; T11 found no partial-gold negatives), the dominator sub-analysis (I5) and the seen-vs-never-seen diagnostic (I10).
- Search the one thing nobody had done. Searches T9 and T12 found **no predictor of joint necessity from execution graphs**. PolyHunk explicitly lists call-graph proximity as future work, because its hunk divergence is lexical/AST/file only ([PolyHunk](https://www.alphaxiv.org/abs/2511.11012); [Hunk4J](https://www.alphaxiv.org/abs/2506.04418)).

**Next step.** Assemble the survivors into one candidate and attack it.

### What each death taught us

| Idea | Killed or weakened by | Search IDs | Lesson |
|---|---|---|---|
| H1, execution-witnessed chain reward | Comet-9B (LLM-judged propagation chains, already trained at repo level) | R1, R3, T8 | Absence claims are perishable. A 10-day-old paper overturned "not found (S2–S14)". The surviving delta must be *verifier type*, and a judge arm is mandatory. |
| H2, graph co-change closure | CodePlan, SynFix; Jia et al., CG-Diff, SiblingRepair predict its premise fails | R6, R8, R9, R11, R12, R14 | "Propagate to k-hop neighbours after an edit" is a hyperparameter of known work. Static call links barely predict co-change. |
| I2, partial-patch interventions | IBugDC (FSE 2024) | T6, T10 (overturning S-K) | A single empty alphaXiv query is weak evidence for older SE venues. Enumeration is known; *predicting* its outcome from graph structure is not. |
| C1/C11, repair and shortcut filter as headline | User veto | — | Graph repair stays an ingredient only. |
| C2–C4, C14, I6 | Zero hidden-test payoff, or measurement only | S10, S-E | Measurement must lead into a method, or it is vetoed. |
| C5, C6, C10, C18 | YAML harness: no tool changes, no shipped encoders | — | Anything graph-shaped must live in training, not inference. |
| I4, I16 | Close to CRR; Gemma rarely reproduces | S-C, S7 | Methods that need Gemma to run more experiments at test time fight its weakest behaviour. |

---

## Round 5: Interventional Fix Graphs survives the final attack

**Findings.** IFG builds two objects for each training task:
1. **The executed call graph** of the failing (F2P) tests on the buggy commit: directed caller→callee edges actually observed, plus per-test node sets.
2. **Interventional labels** on the gold functions from function-level partial-patch runs: necessary set, minimal sufficient subsets, divisible components, and jointly necessary pairs (hyperedges).

**The new piece has two parts:**
- **(i) Prediction.** Executed-graph structure predicts the causal structure of real fixes. Jointly necessary functions should share failing paths and sit close in executed hops. Divisible components should sit on disjoint failing-test subgraphs. This should beat static call-graph distance, file proximity and hunk divergence.
- **(ii) Verifier.** IFG serves as a programmatic training verifier:
  - the stated chain's edges must exist in the executed graph;
  - its endpoint must be an *interventionally necessary* node;
  - the edit set must hit a necessary node in every failing-test component;
  - negatives drop exactly one necessary node.

This replaces Comet's gold-patch judge, and gives multi-location fixes partial credit that a single pass/fail cannot.

**Attacks and answers:**

| Attack | Answer | Residual risk |
|---|---|---|
| "The hitting-set property is a tautology." | Conceded. Any outcome change must touch an executed function. The claim is *which pairs are jointly necessary*, and whether execution predicts that better than baselines. | — |
| "CG-Diff says changed functions are mostly not call-connected." | That is the *static* graph over PRs. Executed failing-path co-occurrence is a different relation. Runtime coupling does add over history in JavaScript impact analysis ([2606.21187](https://www.alphaxiv.org/abs/2606.21187)), but this is unknown for Python fixes. | **High**: this is exactly what the kill test measures. |
| "Comet already rewards chains." | The paper must beat a Comet-style judge arm. | High if the judge wins. |
| Small n | | Medium; rescued by injected multi-function bugs for training. |

**Decision.** IFG is the top pick:
- **Runner-up:** H1-narrow (executed-graph chain verifier vs LLM judge). It becomes IFG's single-location special case.
- **Third:** the I10 + I3 diagnostic. It splits Gemma's missed gold functions into never-seen and seen-but-not-edited, and routes to node-dropped negatives if the latter dominates.

**Next step.** The CPU kill experiment below, pending Shrey's go.

---

## The top pick: Interventional Fix Graphs as known pieces plus one new piece

**Falsifiable claim.**

- **(a) Premise, graph-central.** On multi-function gold fixes, executed-graph features predict pairwise joint necessity with **AUROC ≥ 0.75**. They beat the best of these baselines by **≥ 0.10**:
  - static or direction-repaired call-graph hop distance;
  - same-file or Hunk4J proximity class;
  - hunk divergence.
- **(b) Method.** An IFG-verified RFT LoRA, compared at matched sample count with outcome-only RFT and with Comet-style judge-filtered RFT, gives:
  - **+5 pp** resolve on held-out multi-function tasks;
  - **≥ +2 pp** overall;
  - **+20 pp** chain hop-precision.
- All effect sizes are GUESSED.

| Piece | Source | Status |
|---|---|---|
| Subset enumeration finds divisible vs indivisible multi-hunk bugs | IBugDC, FSE 2024 (searches T6, T10) | Known: Java, hunk level, benchmark cleaning |
| Multi-hunk difficulty from lexical/AST/file divergence; call-graph proximity left as future work | [Hunk4J](https://www.alphaxiv.org/abs/2506.04418), [PolyHunk](https://www.alphaxiv.org/abs/2511.11012) | Known; the opening is stated by the authors |
| Static call connectivity of changed functions (median 0.3) | [CG-Diff](https://www.alphaxiv.org/abs/2609.32057) | Known |
| Execution-witnessed call structure as ground truth | [TraceEval](https://arxiv.org/abs/2605.11006), [RepoReasoner](https://www.alphaxiv.org/abs/2607.25996) | Known (eval / trace prediction only) |
| Execution graphs as repair evidence | [CT-Repair](https://www.alphaxiv.org/abs/2607.12605), [SWE-Doctor](https://www.alphaxiv.org/abs/2607.00990), [XAgent](https://www.alphaxiv.org/abs/2609.09769), [EviACT](https://www.alphaxiv.org/abs/2605.27238) | Known, inference-only |
| LLM-judge reward on propagation chains | [Comet-9B](https://www.alphaxiv.org/abs/2609.34359) | Known; judge-based, test writing |
| Static-graph evidence reward | [VulAgentRL](https://www.alphaxiv.org/abs/2607.26656) | Known |
| Groundedness filter for SFT curation | [P2T](https://www.alphaxiv.org/abs/2605.21996) | Known |
| Gold ≻ degraded patch preference (opposite direction, for concision) | [RECAP](https://www.alphaxiv.org/abs/2608.13292) | Known |
| Offline advantage-weighted SFT | [A²Agent](https://www.alphaxiv.org/abs/2608.29831) | Known |
| Soft grounding under incomplete graphs | [2606.30247](https://www.alphaxiv.org/abs/2606.30247) | Known |
| **NEW: (i) executed-graph prediction of interventional joint necessity and divisibility in real fixes; (ii) its use as a programmatic verifier (necessity-checked chain endpoint, per-component completeness, necessary-node-dropped negatives) to post-train an SWE agent** | — | **Not found** |

**The search evidence behind "not found":**

| Novelty component | Searches with no match |
|---|---|
| No training method checks a stated chain against witnessed execution | S2, S5, S9, S11, R7, R10 |
| No necessity labelling of SWE gold patches | T1 |
| No hitting-set or vertex-cut framing applied to LLM patch completeness (only to agent messages, 2608.29228) | T2 |
| **No predictor of joint necessity from execution graphs or coverage** | **T9** |
| No partial-gold-patch negatives | T11 |
| Runtime coupling exists for JavaScript impact analysis, but not for fix necessity or agent training | T12 |

These are absence claims from alphaXiv's arXiv-centric index and a few WebSearches. OpenReview ICLR-2027 submissions were not searchable, and arXiv coverage of the last ~10 days lags. The defensible phrasing is "no work found uses execution-graph structure to predict, or partial-patch interventions to label, the causal structure of fixes for **training LLM agents**". A claim that "no one has minimised patches" would be false.

**Runner-up: H1-narrow.** The claim is that an edge-level executed-graph verifier separates Gemma's good from bad diagnoses better than a Comet-style gold-patch judge played by Gemma:
- AUROC for downstream fix success ≥ judge + 0.05;
- within-task score ties under 30%, vs TraceLift's 95.5% saturation.

New piece: verifier type, compared head to head (no hit in T8, T9, S2, S11). Cheapest kill: Stage A plus about 7–8 GPU-h of diagnosis sampling, judging and fixer runs. It dies if the executed AUROC ≤ the judge's, if ties exceed 70%, or if static ≈ executed graph.

**Third: as-needed vs systematic comprehension, leading to incomplete negatives (I10 + I3).** The claim is that ≥60% of Gemma's missed gold functions on failed multi-function tasks fall into one class:
- **never displayed** → impact-triggered exploration;
- **displayed but not edited** → necessary-node-dropped negatives.

Novelty rests on searches S-G (no as-needed/systematic theory applied to agents) and T11 (no partial-gold negatives). The comprehension-theory citations are VERIFIED as citations, but their content is recalled. Cheapest kill: about 3 h CPU on r2/r3 traces, if those traces can be retrieved.

---

## The cheapest kill experiment: CPU only, about 1.5 days, 0 GPU-hours

**Data.**
- **Primary:** the 54 tasks whose gold patch passes in the rebuilt scorer-like environment (rich 39, fastapi 10, requests 5; `data/runs/devset_v5/`, `scorer_site.tgz`).
- **Sensitivity:** about 109 host-env-stable tasks, which raises the multi-function count to about 45.
- **Inputs:** gold patches, `test_patch`, FAIL_TO_PASS / PASS_TO_PASS, the provided graphs.
- **For K4 only:** Gemma's r2/r3 agent patches from Kaggle kernel outputs.

**Procedure.** One Kaggle CPU kernel, reusing `build_devset.py`'s scorer env and `gold_graph_analysis.py`'s parsers.

1. **Executed graph.** Run the F2P tests on the buggy commit plus `test_patch` under a `sys.setprofile` tracer. Record caller→callee edges and per-test node sets.
2. **Subset runs.** For each task with k ≥ 2 pre-existing gold functions, apply every function-level subset (2^k for k ≤ 5, ddmin above) and run F2P+P2P.
   - Bundle added helpers with their first caller.
   - Mark runs that fail on ImportError/NameError as *contaminated*.
3. **Labels.** Necessary set, minimal sufficient subsets, divisible components (IBugDC definition), pairwise joint necessity.
4. **Pairwise features.**
   - Executed: co-coverage Jaccard over failing tests, executed hop distance, same failing path.
   - Baselines: undirected provided-graph hops, direction-repaired hops, same file, Hunk4J class, hunk divergence.
   - Null: degree-preserving rewire of the executed graph.
5. **K4 check.** If patches are retrievable, find the share of Gemma's missed necessary functions that sit on an executed failing path its edits did not hit.

**Metric and power.**
- Pairwise AUROC for joint necessity, with task-clustered bootstrap CIs.
- Distribution of structure types: core + incidental, conjunctive, divisible.
- About 100–200 pairs over 25–45 tasks gives **AUROC SE ≈ 0.05–0.07**. That can see a 0.10–0.15 gap, not 0.05. This limitation is pre-registered.

### Kill criteria

| ID | Kills if | What it means |
|---|---|---|
| **K1 (labels)** | Fewer than 50% of multi-function tasks give uncontaminated subset outcomes, or fewer than 15 usable tasks in the 109-task set | Labels cannot be obtained on real tasks |
| **K2 (graph is decorative)** | Executed-graph AUROC < 0.70, or ≤ best baseline + 0.05, or no better than the rewired null | The graph headline is dead |
| **K3 (degenerate structure)** | ≥80% of multi-function tasks are one sufficient function plus incidental edits, or ≥80% are fully conjunctive with all pairs co-covered | No structural signal beyond pass/fail; the completeness half is dead |
| **K4 (Gemma relevance)** | Fewer than 30% of Gemma's missed necessary functions lie on executed failing paths (only if patches are retrievable) | Trace-guided completeness cannot be learned from repro evidence |

**Cost.** About 6 h of code (tracer and subset runner), 3–6 h of Kaggle CPU (≤32 subsets × ~10 s × ~45 tasks), and 2 h of analysis. That is about 1.5 working days and **0 GPU-hours**. It does not touch the 5.5 GPU-hours left before the Oct 10 quota reset.

### What to do after each outcome

| Outcome | Next step |
|---|---|
| **All pass** | Run H1's Stage B oracle-evidence test with a third arm, "+ necessary-set/component hint" (about 6 GPU-h; requires porting the Py 3.13 install fix and `use_scorer_env()` into `run_official_l4.py`, about half a day). Then run the RFT ablation on injected and real multi-function tasks: outcome-only vs Comet-style judge vs static-graph verifier vs IFG verifier, at matched samples. Spend one leaderboard submission on the best. |
| **K2 fails** | Move to the runner-up (single-location chain verifier vs judge). Keep the interventional labels as an analysis section. |
| **K3 fails** | Keep only per-test partial credit (hindsight relabelling by test subset) as a training trick, with no graph claim. |
| **K1 fails** | Run the third pick (seen vs never-seen) and drop multi-location as a thesis. Consider injected multi-function bugs with known structure. |
| **K4 fails** | Drop the completeness reward. Keep the premise result as analysis. |
| **Ambiguous** | Widen to the 109 tasks, then to injected bugs, before any GPU spend. |

---

## What is verified and what is guessed

| Item | Status |
|---|---|
| L1–L3 numbers; 65 tasks with ≥2 gold functions | **VERIFIED** (computed from repo run files; gold labels regex-derived) |
| Tasks carry F2P, P2P and `test_patch`; 54/128 pass gold in the scorer-like env; ~109 in the host env | **VERIFIED** (`build_devset.py`, HQ) |
| Task sandbox is `python:3.13-slim` with `run_command`, so tracebacks are available at inference | **VERIFIED** (HQ); runtime overhead GUESSED |
| Comet-9B's judge sees the gold patch and checks no traces or edges | **VERIFIED** (judge prompt read, T8) |
| CodePlan / SynFix mechanisms | CodePlan **VERIFIED**; SynFix **SNIPPET** (full text not read) |
| IBugDC: 139/281 Defects4J multi-hunk bugs divisible | **SNIPPET** (WebSearch T6/T10; no primary URL recorded) |
| No prior work predicts joint necessity from executed graphs, or trains agents on interventional fix labels | **VERIFIED only as absence** across the logged searches; OpenReview not searchable |
| How many of the 54 tasks are multi-function (~20–25) | **GUESSED** |
| Function-level subsets apply cleanly | **GUESSED**; measured by K1 |
| r2/r3 raw trajectories and agent patches retrievable from Kaggle outputs | **GUESSED** |
| Gemma-31B-4bit will use runtime evidence it gathers | **GUESSED / risk** (ADI: Qwen3-32B used the debugger in 28–31% of tasks) |
| All effect sizes (AUROC ≥ 0.75, +5 pp, +2 pp, +20 pp hop-precision) | **GUESSED** |
| Classic SE/PL claims (delta debugging, Littman 1986 as-needed finding, GraphCL, HER, dominators) | **INFERRED** from memory; check before citing |

---

## Graph-ML novelty is narrow, and the leaderboard lives elsewhere

**Graph-ML novelty, honestly.** If K2 passes, graph reasoning is central rather than decorative:
- The paper's object is a directed executed graph.
- Its labels are interventional hyperedges.
- Its core result is structure prediction, against static, spatial and rewired-null baselines.
- Its verifier is that structure.

Remove the graph and both the claim and the method disappear. If K2 fails, the verifier is a well-engineered SE reward and graph-ML judges will see "check that edges exist", which is VulAgentRL-level graph use. H1-narrow alone has the same weakness.

IBugDC, Comet-9B and Hunk4J/PolyHunk/CG-Diff must be cited up front. The defensible novelty is the *junction* of interventional labels, executed-graph structure and agent post-training, not any component. Realistic ceiling: with ≤45 real multi-function tasks plus injected bugs and one LoRA run by Nov 5, **a solid workshop or short paper**. A main-track graph-ML acceptance would need a K2 gap of ≥0.15 AUROC, replicated on injected bugs across ≥30 repos.

**Leaderboard levers lie elsewhere.** The run history points to these:
- tool-format reliability (61% malformed edits in v5; INT4 doubles tool-name hallucination);
- verification ("wrong fix, never tested" in 47/129);
- loops and budget.

IFG targets the roughly 35% multi-function pool, where Gemma solves 4%. A few tasks gained there are invisible at leaderboard SE ≈ 2.5 pp. IFG is a paper bet. It should not displace the tool-format and verification RFT that carries the main track.

---

## Conclusion

The literature's centre of gravity — richer graphs, smarter traversal, denser localisation rewards — answers a question this model does not ask. Gemma already reaches the right file 81% of the time and gains nothing from hitting the right function. The useful shift is to treat the *fix*, not the repository, as the graph worth reasoning about. Interventions on the fix label its causal structure. Execution on the failing tests supplies a non-circular graph that might predict that structure.

The kill rounds also left a methodological lesson. Every "not found" in this project has a shelf life:
- Comet-9B appeared ten days before it was searched.
- IBugDC hid behind an alphaXiv query that returned nothing.

So the novelty claim must be phrased as a junction and re-searched before submission. The cheapest experiment is designed so that its likeliest failure, K2 (graph decorative), still yields a publishable negative result and routes cleanly to the runner-up.

---

## Decision needed from Shrey

**Run the Stage-A CPU kill experiment for Interventional Fix Graphs: yes or no?** It costs about 1.5 working days (≈6 h code, 3–6 h Kaggle CPU, 2 h analysis) and **0 GPU-hours**. It reuses the existing scorer env (`scorer_site.tgz`) and parsers. **It has not been run.**

Three sub-choices come with a yes:
1. **Task set.** Primary 54 scorer-faithful tasks only, or include the ~109 host-env tasks as the sensitivity set from the start. The 109 set is recommended, since K1's threshold is defined on it.
2. **K4.** Whether to spend time retrieving r2/r3 agent patches and traces from Kaggle outputs. This also enables the third pick's seen/never-seen check.
3. **Priority.** Whether this runs before, or in parallel with, the leaderboard-side tool-format and verification work. That work is where main-track points most likely come from.

A "no" means the paper falls back to a narrower, already-weakened H1 framing, or to a leaderboard-first plan with no graph headline.
