# Cross-domain ideas: graph reasoning × code comprehension × agent post-training (8 Oct 2026)

Scope: generate 12–20 candidate paper ideas by transplanting mechanisms from other fields onto **our measured bottleneck**, then check each one's novelty against the literature. Inputs: `docs/HQ.md`, `CLAUDE.md`, the five notes in this folder (especially `critic_judgement.md`), and `results/runs/*/per_task.jsonl` (checked to exist: it has `gold_funcs`, `resolved`, `n_tool_calls` and step-of-first-gold fields per task).

Labels:
- **VERIFIED**: read in the paper text or full alphaXiv report this session.
- **SNIPPET**: seen only in a search-result abstract or snippet.
- **INFERRED**: my reasoning, or recalled from memory and not re-checked this session. Every classic SE/PL paper marked INFERRED should be checked before it is cited in the paper.

**The bottleneck we are aiming at** (from critic_judgement.md §0, VERIFIED arithmetic on repo files):
- L1: when the diagnosis names the right function, resolve is 27%; when it names the wrong one, 28%.
- L2: touching a gold file is close to necessary (24/80 resolved vs 2/49).
- L3: fixes spanning ≥3 gold functions are a dead zone (2/45 resolved, vs 19/57 for single-function fixes).
- Other dominant failures: wrong fix never tested (47/129), exact-repeat loops (28–48% of calls), malformed edit calls (61% in v5).

So the target is **diagnosis → a correct, complete, verified multi-location fix**, not localisation.

Constraints applied to every idea:
- Model and serving: Gemma-4-31B at 4 bits, LoRA only, 32k context, ~5 min per task.
- Inference: offline, declarative YAML. We cannot change tool outputs at inference, and skills are unverified on the scorer.
- Budget: ~30 L4 GPU-hours a week, $50 cash.
- Already vetoed: name invariance (T1), direction repair as the headline, imperfect-graph measurement on its own.

---

## Q1. Which cross-domain transplants could become a novel paper, and what does the evidence say about each?

### Takeaway
The most promising transplants treat **the fix itself as a small graph**, not the repository. The mechanisms are:
- causal interventions over subsets of the gold patch (delta debugging / do-calculus);
- motif census of the patch subgraph on the call graph;
- contrastive "node-dropping" negatives;
- edit-level counterfactual credit from test execution only.

Novelty searches found no paper that labels the **causal structure of multi-location fixes**, i.e. which edited functions are necessary or jointly sufficient for the failing tests to pass, or that uses it to post-train an agent.

Adjacent work exists and must be cited:
- Co-edit and next-edit prediction is crowded: CoEdPilot, Coeditor, GitCF, co-changed-method learning-to-rank, CodePlan.
- Multi-hunk repair: SiblingRepair, MultiFixer.
- Call-graph connectivity of real PRs: CG-Diff.
- Static task-difficulty prediction for agents (2608.18280) uses **no graph features**, which leaves room for graph-structural difficulty.

Localisation-side transplants (IR query expansion, graph encodings, community maps) are low-payoff given L1 and are listed as rejected.

### The candidates

Format for each idea: mechanism → known pieces → NEW piece → why it targets our bottleneck → graph-ML judge's view → novelty searches (log IDs in Q3) → cheapest falsification test.

---

#### I1. Fix-motif census: multi-location fixes as patch subgraphs (network science: motifs and subgraph census)

- **Mechanism.** Embed each gold patch as the subgraph induced by its edited functions on the (direction-repaired) call graph. Classify it into motifs:
  - **chain**: f→g→h all edited;
  - **star-callers**: a callee's contract changed and its callers were updated;
  - **siblings**: no edge, but parallel implementations, e.g. sync/async twins or per-backend methods;
  - **containment**: class plus methods;
  - **new-node**: added functions;
  - **disconnected**.

  Then teach the agent one completion operator per motif, via LoRA SFT on motif-labelled trajectories, applied as a pre-submit step: "callers of the changed callee", "siblings by name/AST shape", and so on.
- **Known pieces.**
  - CG-Diff (2609.32057), on 7,490 Python PRs: median Cluster Participation Ratio (CPR) is 0.3. Only 39.6% of PRs have more than half of their changed functions connected by calls, and 50.8% have a cluster of ≥5 functions. VERIFIED.
  - SiblingRepair (2605.06209) covers the sibling motif for multi-hunk repair (SNIPPET).
  - MultiFixer (2607.26591) uses a coordinator–proposer for multi-hunk bugs (SNIPPET).
  - CodePlan (2309.12499) does change-may-impact propagation for repository-level edits (SNIPPET; known from memory as FSE'24, INFERRED).
  - Network motifs: Milo et al., Science 2002 (INFERRED).
- **NEW piece.** A motif taxonomy of *bug-fix* subgraphs, linked to *agent failure* by motif, with motif-specific completion operators taught by post-training. CG-Diff measures connectivity for code review, not repair and not agents. SiblingRepair handles one motif with a dedicated pipeline, not a trained agent.
- **Why it targets our bottleneck.** L3 is the largest failure pool: 45/129 tasks, 4% resolved. If most ≥3-function fixes fall into one or two motifs, one trained operator could convert a share of them.
- **Graph-ML judge's view.** Positive. Subgraph census and motif statistics are native graph-ML. A "fix motif spectrum" with a degree-preserving rewired null is a clean figure. The risk is that it reads as an SE empirical study unless it is tied to a training result.
- **Novelty searches.** S-B, S-I, S-J. Closest: CG-Diff, SiblingRepair, CoEdPilot, GitCF. No motif census of fix subgraphs linked to agent outcomes was found.
- **Cheapest falsification test (CPU, ~4 h, 0 GPU).**
  - **Data:** our 129 gold patches plus the provided graphs; for density, also public gold patches (SWE-bench Verified 500, SWE-Gym, SWE-smith instances; INFERRED to be freely downloadable). Our r1/r2/r3 outcomes give a per-motif resolve rate for Gemma.
  - **Metric:** motif distribution for ≥2- and ≥3-function patches. Per-motif Gemma resolve rate. ΔAUC of motif features over the hunk/line features from 2608.18280 in predicting resolve across our 3 runs.
  - **Kill:**
    - more than 60% of ≥3-function patches are "disconnected non-sibling" or "new-node" (no structural operator applies); or
    - motif adds ΔAUC < 0.03 over num_hunks + lines_deleted.

---

#### I2. Causal structure of fix sets via partial-patch interventions (delta debugging × do-calculus × Shapley)

- **Mechanism.** For each task with k ≥ 2 gold functions, apply every subset of the gold patch at function granularity (2^k ≤ 32 for k ≤ 5; use ddmin beyond that) and run the FAIL_TO_PASS and PASS_TO_PASS tests. Each run is an intervention do(fix S). Its outcome yields:
  - per-function **necessity** and **sufficiency**;
  - a **Shapley value** over F2P tests passed;
  - the **interaction structure**: conjunctive "AND" (all functions needed), "core + incidental" (one function flips the tests, the others are cleanup, docs or refactors), or disjunctive.

  This turns each multi-location fix into a labelled hypergraph over functions.
- **Known pieces.**
  - Delta debugging (Zeller & Hildebrandt, TSE 2002; INFERRED).
  - Tangled or non-essential changes in commits (Herzig & Zeller MSR 2013; Kawrykow & Robillard ICSE 2011; INFERRED).
  - Causal Testing (Johnson, Brun & Meliou, ICSE 2020; INFERRED).
  - Weak validation evidence in repair agents: "Validation Evidence in LLM Repair Agents" (2607.28871) measures how often passing tests actually test the bug (SNIPPET).
  - Shapley attribution is standard.
- **NEW piece.** A dataset-level causal decomposition of **gold** multi-location fixes, used as (a) a diagnostic for *why* agents fail them and (b) labels for post-training (I3, I4, I12). The discover_papers search for patch minimisation / untangling-into-necessary-hunks returned **zero papers** (S-K).
- **Why it targets our bottleneck.** It directly explains L3.
  - If ≥3-function fixes are mostly "core + incidental", the dead zone is a **core-finding** problem: the tests only need one function.
  - If they are mostly conjunctive, it is a **completeness** problem.

  The two outcomes call for different training. Either is a publishable finding about multi-location repair.
- **Graph-ML judge's view.** Good, provided the fix-set structure is presented as a learned hypergraph or interaction graph (pairwise Shapley interactions as edge weights) and related to call-graph distance: "do causally interacting edits sit close on the call graph?" That last question is the graph-ML hook.
- **Novelty searches.** S-A, S-K. No hit for partial-gold-patch necessity labels.
- **Cheapest falsification test (CPU, ~1 day, 0 GPU).**
  - **Data:** the scorer-faithful tasks with ≥2 gold functions among the 54 that pass gold in the scorer-like env (`data/runs/devset_v5/`). Expect ~20–25 tasks; widen to the ~109 host-env tasks for power.
  - **Metric:**
    - share of tasks where some strict subset of the gold functions already turns all F2P tests green (core + incidental);
    - share where every function is necessary (conjunctive);
    - Spearman correlation between pairwise interaction and call-graph hop distance.
  - **Kill (for the "core-finding" reframing):** fewer than 20% of multi-function tasks have a sufficient strict subset. **Kill (for the graph hook):** interaction vs hop-distance |ρ| < 0.15.
  - **Hours:** ~3 h to write it (reuse `build_devset.py`'s env and gold-patch hunk parser), ~2–4 h of Kaggle CPU.

---

#### I3. Structurally incomplete negatives: node-dropping contrastive post-training for completeness (graph contrastive learning → preference / RFT)

- **Mechanism.** Build preference pairs (full gold patch ≻ gold patch minus one *necessary* function, using I2's labels) and train a LoRA either:
  - (a) with DPO or weighted SFT on a "pre-submit completeness check" turn ("which other locations does this change require?"); or
  - (b) as negatives in rejection-sampled SFT on agent trajectories that submitted incomplete patches.

  This is node-dropping augmentation from graph contrastive learning, applied to the patch subgraph.
- **Known pieces.**
  - Node-dropping augmentation: GraphCL, You et al., NeurIPS 2020 (INFERRED).
  - DPO / preference training for SWE agents exists (INFERRED; not searched in depth).
  - Hard negatives from mutation are common in code models (INFERRED).
  - PathRouter-style route-scaled advantages (general_graph_reasoning.md, VERIFIED there).
- **NEW piece.** Negatives defined by the **causal patch subgraph**: each negative misses exactly one execution-verified necessary node. This trains completeness of multi-location fixes rather than correctness of a single hunk.
- **Why it targets our bottleneck.** It targets L3 directly, and aims at "submit without checking the remaining locations".
- **Graph-ML judge's view.** Recognisable mechanism (GCL augmentations) in a new domain with an execution-verified notion of "semantics-changing augmentation". That is a nice inversion of GCL's usual "semantics-preserving" assumption.
- **Novelty searches.** S-A, S-C. No paper uses partial gold patches as negatives.
- **Cheapest falsification test (GPU ~1.5 h, inference only).**
  - **Data:** ~50 (full, minus-one-necessary) patch pairs from I2 plus SWE-bench Verified multi-function gold patches (pairs built statically; necessity unverified, used as a sensitivity set).
  - **Metric:** zero-shot Gemma-31B pairwise discrimination: "Which patch fully resolves the issue?", order randomised. Also an open-ended "what else must change?" completion recall.
  - **Kill:**
    - Gemma already discriminates ≥ 90% (nothing to teach); or
    - after a 1-epoch LoRA on ~500 pairs (≈3–4 GPU-h), held-out discrimination improves by < 10 pp.

---

#### I4. Edit-ablation counterfactual credit: execution-only credit assignment over graph-structured actions (RL theory: counterfactual credit, leave-one-out)

- **Mechanism.** For each rollout's final patch, revert each edit hunk (or each edited function) one at a time and re-run the tests. The change in F2P/P2P pass counts is that edit's **counterfactual credit**. Use it to weight the edit-producing turns in advantage-weighted SFT: up-weight necessary edits, down-weight harmful or idle ones. This needs zero extra LLM tokens, only test runs.
- **Known pieces.**
  - Counterfactual Rollout Replay (2609.33875): forkable environments as process rewards (SNIPPET; the critic notes ~1.7× rollout compute).
  - SWE-MILE (2609.32631): milestone credit (SNIPPET).
  - Dependency-aware policy optimisation for terminal agents (2610.03634) (SNIPPET).
  - GraphHCA closed-form hindsight credit (2609.35084) (SNIPPET).
  - "Credit Without Ground Truth" (2608.19760): step-level credit signals fail against executed replay in ALFWorld (SNIPPET).
  - A²Agent advantage-weighted MLE (VERIFIED in post_training_navigation.md).
- **NEW piece.** The counterfactual is taken over the **patch's edit set** (environment state) rather than over the policy's future actions, so it is exact, cheap, and needs no LLM re-rollouts. That suits a 31B model on 30 GPU-h a week.
- **Why it targets our bottleneck.** In multi-edit rollouts it separates the edit that made progress from the noise. Failed multi-function rollouts that pass *some* F2P tests become partially credited data instead of being discarded.
- **Graph-ML judge's view.** Neutral to positive. It is RL credit assignment, and becomes graph-ML when the edit set is treated as a graph and credit is propagated along it. Weaker graph content than I1/I2.
- **Novelty searches.** S-C. Every hit counterfactualises actions or rollouts. None reverts patch hunks for credit.
- **Cheapest falsification test (CPU, ~4 h).**
  - **Data:** agent patches from r2/r3 (on Kaggle kernel outputs; the critic marks retrieval as GUESSED), ~100 non-empty patches.
  - **Metric:** share of patches where per-edit credit is non-degenerate, i.e. at least one edit has credit different from the others.
  - **Kill:** credit is degenerate (all edits jointly necessary, or none matters) in > 85% of patches. The signal would then equal outcome-only RFT.

---

#### I5. Fix at the dominator: compiler dominator trees on witnessed failing paths as a fix-site prior

- **Mechanism.** Build the dynamic call graph of the F2P tests (the critic's Stage A tracer) and compute the dominator tree from the test entry. Ask where gold sits:
  - at the crash frame;
  - at an immediate dominator of the crash frame (an upstream decision point);
  - at the lowest common dominator of several failure frames;
  - off the path (data-flow origin).

  If gold is mostly the lowest common dominator of multiple symptoms, teach the diagnosis to name "the dominator of the symptom set" as the fix site. This could also collapse some multi-location fixes into one.
- **Known pieces.**
  - Dominator trees: Lengauer–Tarjan 1979 (INFERRED).
  - CausalRepair dual slicing (2608.10613) (SNIPPET).
  - PathFix expected-path repair (2510.14341) (SNIPPET).
  - EviACT, CT-Repair temporal execution graph (VERIFIED in critic S16/S17).
  - SWE-Doctor's propagation path (VERIFIED in critic S3).
- **NEW piece.** A dominator-theoretic characterisation of where real fixes sit relative to symptoms, used as a training prior for diagnosis. The search found no dominator-based fix-site work (S-F).
- **Why it targets our bottleneck.** Partial. It targets diagnosis quality *beyond* function hit (L1 says a function hit alone does not convert). It might convert multi-symptom fixes into single-site ones.
- **Graph-ML judge's view.** It is a classic graph-algorithm concept on a real-data graph, but the judges may see it as PL rather than graph learning.
- **Cheapest falsification test (CPU, shares the critic's Stage A kernel, +2 h).**
  - **Metric:** distribution of gold position: crash frame, dominator, LCD, off-path.
  - **Kill:**
    - gold = deepest failing frame in > 70% of tasks (trivial); or
    - gold is off the dominator chain in > 60% of tasks (prior useless).

---

#### I6. Graph-structural task difficulty: effective resistance and Steiner cost of the fix set (graph ML: over-squashing, resistance distance)

- **Mechanism.** Define task difficulty as:
  - the Steiner-tree cost connecting {issue-anchor nodes} ∪ {gold functions} on the (hub-pruned, direction-repaired) call graph; and
  - the effective resistance between anchors and gold. This is the over-squashing quantity: a low-resistance path means many parallel routes.

  Test whether these predict agent failure beyond the static patch features.
- **Known pieces.**
  - 2608.18280, on 45,769 tasks from 1,553 repos with CoderForge-Preview Qwen3-Coder-480B trajectories: static patch + repository + prompt features give AUC 0.863. The top features are `patch_lines_deleted`, `patch_num_hunks` and `patch_hunk_gap_mean`. **No call-graph features** are used. VERIFIED.
  - Effective-resistance rewiring (2603.11944) and sheaf effective resistance (2610.04157) for GNN over-squashing (SNIPPET).
  - Predicting task difficulty without rollouts (2608.05797) (SNIPPET).
- **NEW piece.** Graph-theoretic (resistance/Steiner) difficulty for agentic repair, plus the claim that **LLM agents suffer an "over-squashing" analogue**: fixes reachable only through bottleneck hubs are harder.
- **Why it targets our bottleneck.** Indirectly. It explains L3 structurally and can drive I7. On its own it is measurement, and the user has rejected measurement as a headline.
- **Graph-ML judge's view.** Strong on vocabulary (effective resistance, over-squashing). The risk is "correlational study".
- **Cheapest falsification test (CPU, ~1 day).**
  - **Data:** our 129 tasks × 3 runs (small); for density, a subset of CoderForge-Preview tasks with public trajectories, building static call graphs with Python `ast` for ~100 repos (INFERRED feasible on Kaggle CPU).
  - **Kill:** ΔAUC < 0.02 over 2608.18280's top-3 patch features.

---

#### I7. Curriculum by graph difficulty for multi-location RFT (RL curriculum theory)

- **Mechanism.** Order the synthetic training tasks (SWE-smith-style injected multi-function bugs) by I6's Steiner/resistance cost or I1's motif, from single-node fixes to chains to stars. Prerequisite structure follows Active Curriculum Refinement's DAG-of-environments idea.
- **Known pieces.**
  - Active Curriculum Refinement (2608.26469) (SNIPPET).
  - Curriculum RL in code generation: SpecCoder 2609.06041, requirement-aware 2605.00433 (SNIPPET).
  - SWE-smith injection (VERIFIED in post_training_navigation.md).
- **NEW piece.** A graph-structural difficulty axis for the SWE curriculum.
- **Why it targets our bottleneck.** L3. Feasibility is weak: it needs two full LoRA trainings to compare.
- **Graph-ML judge's view.** Fine as an ablation, weak as a headline.
- **Cheapest falsification test.** Conditional on I6 passing. Otherwise, a 2-arm RFT on ≤300 tasks (~10 GPU-h). **Kill:** no difference in multi-function resolve on the held-out repos (expected to be underpowered, which is why it ranks low).

---

#### I8. Agents as information foragers: PFIS spreading activation (≈ personalised PageRank) as a generative model of agent navigation (cognitive science: information foraging theory)

- **Mechanism.** Fit PFIS to Gemma's navigation traces: spreading activation of "scent" (issue-text overlap) over code topology. Ask:
  - Does PFIS predict the next visited function better than recency?
  - Do loops begin where scent is flat (no gradient)?
  - Do failing multi-function tasks show "scent capture" by one region?

  Then use PFIS residuals as a trajectory filter: keep trajectories that follow scent and leave depleted patches.
- **Known pieces.**
  - Lawrance, Bogart, Burnett et al., "How programmers debug, revisited: an information foraging theory perspective", IEEE TSE 2013. Programmers' verbalisations concerned scent-following far more than hypotheses, and an executable scent model predicted navigation better than scent-free models. SNIPPET from the publisher/Memphis repository and Burnett's reprint page.
  - PFIS / PFIS2 (SNIPPET, same sources).
  - InForage / Scent of Knowledge (2505.09316) applies IFT to retrieval-augmented reasoning, not code (SNIPPET).
  - Query-aware spreading activation over KGs (2606.30133) and SYNAPSE (2601.02744) (SNIPPET).
- **NEW piece.** Using IFT/PFIS as a **quantitative model of LLM coding-agent navigation and loop onset**. The search found no application to coding agents (S-H).
- **Why it targets our bottleneck.** Loops and budget overruns, which are major failures. It does not target L3 directly.
- **Graph-ML judge's view.** Positive: PFIS is spreading activation, i.e. PPR on a text-attributed graph. "A 1990s cognitive model predicts a 2026 agent's failures" is a memorable story.
- **Cheapest falsification test (CPU, ~1 day; needs raw traces from Kaggle outputs).**
  - **Metric:** MRR of next-visited function under PFIS vs recency vs uniform-neighbour baselines; AUC of a scent-gradient feature for predicting loop onset in the next 5 calls.
  - **Kill:** MRR gain over recency < 0.05 **and** loop-onset AUC < 0.65.

---

#### I9. Marginal-value-theorem patch leaving as a stop/continue rule (behavioural ecology)

- **Mechanism.** Treat files or graph communities as "patches". Charnov's MVT says leave when the instantaneous information gain drops below the environment's average rate. Train, via SFT on rewritten trajectories, a "leave-patch / switch to editing" decision. This attacks budget overruns (the localiser averaged 48 calls against an asked 25) and loops.
- **Known pieces.**
  - "Not Worth Another Token" (2608.08389): marginal value estimation for efficient deep-research agents (SNIPPET).
  - Context Gathering Decision Process as a POMDP (2605.07042) (SNIPPET).
  - Bayesian control for coding agents (2606.24453) (SNIPPET).
  - MVT (Charnov 1976) (INFERRED).
- **NEW piece.** Only the transfer to coding agents, plus the graph-community definition of a patch. Novelty is low because marginal-value stopping already exists for research agents.
- **Why it targets our bottleneck.** Loops and overruns, not L3.
- **Graph-ML judge's view.** Lukewarm.
- **Cheapest falsification test (CPU, on the traces).** **Kill:** time-in-file beyond the MVT threshold does not predict failure (odds ratio < 1.5).

---

#### I10. As-needed vs systematic comprehension: "delocalized plans" explain the multi-location dead zone (cognitive science of program comprehension)

- **Mechanism.** Classic studies found that programmers who comprehend **as-needed** make modifications that miss non-local interactions, while **systematic** readers make correct modifications. Soloway et al. named code whose plan is spread across locations a **delocalized plan**.

  Hypothesis: Gemma is an as-needed comprehender, and the missed gold functions in L3 are functions it **never viewed**.

  Intervention: a LoRA-taught "systematic mode", triggered when the edited function's impact set (callers + siblings + contract users) is non-trivial. It enumerates and inspects that set before submitting.
- **Known pieces.**
  - Littman, Pinto, Letovsky & Soloway, "Mental models and software maintenance", Empirical Studies of Programmers, 1986. The citation is VERIFIED via search (NASA NTRS / dblp listing). The **as-needed vs systematic finding is recalled, INFERRED**: the snippet did not show it.
  - Soloway, Pinto, Letovsky, Littman & Lampert, "Designing documentation to compensate for delocalized plans", CACM 1988. Title VERIFIED via search snippet. Content INFERRED.
  - Koenemann & Robertson (CHI 1991): developers forage cues as needed rather than following a systematic strategy. VERIFIED as cited inside CG-Diff's related work.
  - LaToza & Myers, "Developers ask reachability questions" (ICSE 2010). VERIFIED as cited in CG-Diff.
  - CG-Diff: 58 of 110 necessary callees sit in a different file from their caller. VERIFIED.
- **NEW piece.**
  - The **seen-but-not-edited vs never-seen** decomposition of missed fix locations in agent trajectories, framed with comprehension theory.
  - An impact-triggered systematic strategy learned by post-training.
  - The search found no application of systematic/as-needed theory to LLM coding agents (S-G).
- **Why it targets our bottleneck.** L3 directly, and it is decisive about *which* intervention is needed:
  - never-seen → exploration/impact-closure training (I1/I11);
  - seen-but-not-edited → planning/completeness training (I3).
- **Graph-ML judge's view.** Moderate. The graph content is the impact set (k-hop/PPR closure). The cognitive-science framing is distinctive, and a good paper "story" layer on top of I1–I3.
- **Cheapest falsification test (CPU, ~3 h).**
  - **Data:** r2/r3 traces (raw traces from Kaggle outputs) for the 65 tasks with ≥2 gold functions.
  - **Metric:** among missed gold functions in failed tasks, the share whose body was displayed (sed/grep output containing the def lines) before submit_patch.
  - **Kill (for systematic-mode training):** > 70% of missed gold functions were already viewed. Extra exploration will not help; move the weight to I3.

---

#### I11. Change-impact closure checklist (PPR from the edited set) — the critic's runner-up R2-d, re-examined

- **Mechanism.** Personalised PageRank (or k-hop non-hub closure) seeded at the edited functions gives a ranked co-edit candidate list. The agent checks it before submit.
- **Known pieces.**
  - CodePlan (2309.12499) (SNIPPET); CoEdPilot (2408.01733) (SNIPPET); Coeditor (2305.18584) (SNIPPET).
  - GitCF (2609.37045): reducing incomplete changes using commit similarities (SNIPPET).
  - Co-changed-method learning-to-rank (2411.19099) (SNIPPET).
  - Neural change prediction (2606.03378) (SNIPPET).
  - Impact analysis with transformers + PDGs (2607.23355) (SNIPPET).
  - GraphLocator, MultiFixer (critic S8, S15, VERIFIED/SNIPPET).
- **NEW piece.** Essentially none. Its value is as an ingredient. Co-edit prediction is **crowded**.
- **Cheapest falsification test.** As in critic_judgement.md (H2): kill if fewer than 40% of missed gold functions lie within 2 non-hub hops, or if AUC vs the same-file null is < 0.6.

---

#### I12. Hindsight relabelling of partial fixes by test subset (RL: Hindsight Experience Replay)

- **Mechanism.** A failed multi-function rollout whose patch turns *some* F2P tests green has achieved a different goal: "make tests T′ ⊂ T pass". Relabel the task with T′, and with symptom text derived only from T′'s failures. Keep it as a positive SFT example. Use I2's causal labels to make sure the subset is coherent.
- **Known pieces.**
  - HER (Andrychowicz et al., NeurIPS 2017) (INFERRED).
  - GraphHCA hindsight credit (2609.35084) (SNIPPET).
  - P2T reverse distillation of gold patches into process graphs (VERIFIED in post_training_navigation.md).
  - SWE-smith synthetic task generation (VERIFIED there).
- **NEW piece.** Goal relabelling at the level of **test subsets** in SWE, which mines the multi-function failure pool for usable data. The search did not surface SWE-specific HER (S-C). Absence is weakly established.
- **Why it targets our bottleneck.** It turns L3 failures into training signal. The risk is that it teaches "partial is fine".
- **Graph-ML judge's view.** Weak graph content.
- **Cheapest falsification test (CPU).** **Kill:** fewer than 15% of failed multi-function rollouts pass any F2P test.

---

#### I13. Non-backtracking exploration prior against loops (graph ML: non-backtracking walks)

- **Mechanism.** Treat a trajectory as a walk on the code graph. Exact repeats (28–48% of calls) are backtracking steps. Rewrite SFT trajectories to be non-backtracking (drop revisits whose observation is unchanged), or filter on a backtracking-rate threshold.
- **Known pieces.**
  - NBRW theory: self-intersection and birthday-paradox results for non-backtracking walks (2608.26321, 2608.09729) (SNIPPET).
  - Non-backtracking GNNs (INFERRED).
  - "Redundant or Necessary?" benchmark for redundant agent steps (2605.29893) (SNIPPET).
  - GraphEcho redundancy (VERIFIED in general_graph_reasoning.md).
- **NEW piece.** The framing only. In practice it is "dedupe repeated calls in the SFT data", which any tool-format SFT will already do.
- **Graph-ML judge's view.** Cute link, thin substance.
- **Cheapest falsification test.** **Kill:** the backtracking rate does not predict failure after controlling for trajectory length.

---

#### I14. GNN teacher → LoRA student distillation of co-edit structure (GNN-to-LLM distillation)

- **Mechanism.** Train a small GNN offline on SWE-smith / GREPO to predict co-edit sets from the call graph plus text features. Use its ranked predictions as rationale targets in SFT, so the LLM learns to propose structurally likely co-edits from code it has read via grep. No GNN ships at inference.
- **Known pieces.**
  - GNN-to-MLP KD, e.g. InfGraND 2601.08033 (SNIPPET).
  - GRIP in-parameter graph reasoning (2511.07457) (SNIPPET).
  - GREPO: GNNs vs LLMs on repository bug localisation (2602.13921) (SNIPPET; VERIFIED in earlier notes).
  - CoT distillation for graph fraud detection (2601.22949) (SNIPPET).
- **NEW piece.** GNN → agent-LLM distillation for code edits. The search found no code instance (S-L).
- **Why it targets our bottleneck.** L3, in principle. But the student cannot see the graph at test time, so the transfer mechanism is unclear.
- **Graph-ML judge's view.** Very positive on paper. Fragile in practice.
- **Cheapest falsification test (CPU/GPU-light).** **Kill:** the GNN does not beat PPR + same-file heuristics at co-edit prediction (Recall@5 gain < 5 pp). If it does not beat them, there is nothing extra to distil.

---

#### I15. Contract-delta propagation (abstract interpretation → which callers must change)

- **Mechanism.** Before a fix is completed, the agent states the abstract **contract delta** of its edit: return type, nullability, raised exceptions, argument semantics. Callers are flagged only if they depend on the changed facet. This is abstract-domain filtering of the impact set, to cut I11's false positives.
- **Known pieces.**
  - Abstract interpretation (Cousot & Cousot 1977) (INFERRED).
  - CodePlan's may-impact analysis (SNIPPET).
  - Neural change prediction (2606.03378) (SNIPPET).
- **NEW piece.** LLM-stated abstract contract deltas used as the propagation key. Plausibly new, but hard to evaluate cheaply.
- **Cheapest falsification test.** Via I1: if the "star-callers after contract change" motif is under 15% of multi-function fixes, kill.

---

#### I16. Agent-side do() interventions at test time (causal inference)

- **Mechanism.** At inference, through `run_command`, the agent monkeypatches a suspected function to a stub with the expected behaviour (or forces a branch). It then checks whether the failing test flips before it writes the real fix, so hypotheses are tested interventionally. For multi-location fixes it can intervene on pairs.
- **Known pieces.**
  - AutoSD / scientific debugging with LLMs (Kang et al. 2023) (INFERRED).
  - Causal Testing (INFERRED).
  - SWE-Doctor's runtime diagnosis records (VERIFIED in critic S3).
  - ADI/FramePilot debugger use, where Qwen3-32B used the debugger in only 28–31% of tasks (VERIFIED in critic S7).
- **NEW piece.** Interventions over **sets** of functions, learned via LoRA from I2's oracle intervention outcomes.
- **Why it targets our bottleneck.** It targets "wrong fix never tested" and L3. It costs time within ~5 min per task.
- **Cheapest falsification test.** Use I2's data. **Kill:** single-function interventions are uninformative, i.e. no function flips any F2P test, in more than 70% of multi-function tasks.

---

#### Considered and rejected quickly (low payoff given L1, or infeasible)

| Transplant | Field | Why rejected |
|---|---|---|
| Learned sparse retrieval / doc2query / Rocchio feedback over the graph | IR | Localisation is not the bottleneck (L1). File hit is already 81% (L2). |
| Laplacian/structural encodings shown to the LLM | Graph transformers | We cannot modify tool outputs in declarative YAML. LLMs do not use raw PE vectors. |
| Spectral community "architecture map" in the prompt | Spectral clustering | Already killed as C14 (critic): targets file retrieval. |
| Spectrum-based FL diffused by PageRank | SBFL + PPR | Crowded (critic S10: CoSIL, SieveFL, FL-RAG; PRFL from memory). Also localisation. |
| Potential-based shaping with graph distance to gold | RL theory (Ng 1999) | A localisation reward again (A²Agent covers dense discovery). L1 caps payoff. |
| Mutation-score adequacy of the agent's repro test | Mutation analysis | Valuable for "untested fixes", but not graph reasoning, and test-generation literature is crowded (INFERRED). Keep as a leaderboard trick. |

### Cited Findings
- CG-Diff, 9,310 PRs from 77 Python projects (7,490 with changed functions):
  - median Cluster Participation Ratio 0.3;
  - in 39.6% of PRs more than half of the changed functions are call-connected;
  - in 50.8% the largest cluster has ≥5 functions;
  - 58/110 necessary callees are in a different file from their caller.

  VERIFIED — [CG-Diff, arXiv 2609.32057](https://www.alphaxiv.org/abs/2609.32057)
- CG-Diff cites Koenemann & Robertson (CHI 1991) for "developers forage cues as needed instead of following a systematic strategy", and LaToza & Myers (ICSE 2010) for reachability questions. VERIFIED (as cited) — [CG-Diff](https://www.alphaxiv.org/abs/2609.32057)
- 2608.18280, static task-difficulty prediction for agents on 45,769 SWE-smith/SWE-rebench tasks:
  - XGBoost AUC 0.863 for any-success;
  - patch features alone 0.846, repository features alone 0.839, prompt features about 0.59;
  - top SHAP features: `patch_lines_deleted`, `patch_num_hunks`, `patch_hunk_gap_mean`;
  - no call-graph features are used.

  VERIFIED — [What Makes Software Issue Resolution Tasks Difficult for Agents?](https://www.alphaxiv.org/abs/2608.18280)
- Multi-hunk repair already has dedicated systems: SiblingRepair (similar fixes across "sibling" locations) and MultiFixer (coordinator–proposer). SNIPPET — [SiblingRepair 2605.06209](https://www.alphaxiv.org/abs/2605.06209); [MultiFixer 2607.26591](https://www.alphaxiv.org/abs/2607.26591)
- Co-edit and next-edit prediction is an established line. SNIPPET — [CoEdPilot 2408.01733](https://www.alphaxiv.org/abs/2408.01733); [Coeditor 2305.18584](https://www.alphaxiv.org/abs/2305.18584); [GitCF 2609.37045](https://www.alphaxiv.org/abs/2609.37045); [co-changed method L2R 2411.19099](https://www.alphaxiv.org/abs/2411.19099); [CodePlan 2309.12499](https://www.alphaxiv.org/abs/2309.12499); [Neural Change Prediction 2606.03378](https://www.alphaxiv.org/abs/2606.03378)
- SWE credit assignment in 2026 counterfactualises actions or rollouts. SNIPPET — [CRR 2609.33875](https://www.alphaxiv.org/abs/2609.33875); [SWE-MILE 2609.32631](https://www.alphaxiv.org/abs/2609.32631); [Dependency-aware PO 2610.03634](https://www.alphaxiv.org/abs/2610.03634); [GraphHCA 2609.35084](https://www.alphaxiv.org/abs/2609.35084); [Credit Without Ground Truth 2608.19760](https://www.alphaxiv.org/abs/2608.19760)
- A study measures how often a passing test in repair agents actually exercises the bug. SNIPPET — [Validation Evidence in LLM Repair Agents 2607.28871](https://www.alphaxiv.org/abs/2607.28871)
- Lawrance et al. (IEEE TSE 2013) found programmers' debugging verbalisations concerned scent-following far more than hypotheses. Their executable scent model (PFIS lineage) predicted navigation better than models without scent. SNIPPET — [Memphis Digital Commons](https://digitalcommons.memphis.edu/facpubs/2870); [Burnett reprint](https://web.engr.oregonstate.edu/~burnett/Reprints/TSE-IFT-2013-asprinted.pdf)
- Information foraging has been applied to search-augmented LLM reasoning (InForage), but the search found no application to coding agents. SNIPPET — [Scent of Knowledge 2505.09316](https://www.alphaxiv.org/abs/2505.09316)
- Marginal-value stopping exists for deep-research agents. SNIPPET — [Not Worth Another Token 2608.08389](https://www.alphaxiv.org/abs/2608.08389)
- Littman, Pinto, Letovsky & Soloway, "Mental models and software maintenance" (1986), and Soloway et al., "Designing Documentation to Compensate for Delocalized Plans" (CACM 1988) exist. SNIPPET (citation only) — [NASA NTRS](https://ntrs.nasa.gov/citations/19860010503); [dblp](https://dblp1.uni-trier.de/pid/92/484.html)
- Fault-site reasoning via slicing and expected paths exists for LLM repair. SNIPPET — [CausalRepair 2608.10613](https://www.alphaxiv.org/abs/2608.10613); [PathFix 2510.14341](https://www.alphaxiv.org/abs/2510.14341)
- Effective resistance is the active over-squashing measure in GNN theory. SNIPPET — [ER rewiring 2603.11944](https://www.alphaxiv.org/abs/2603.11944); [Sheaf ER 2610.04157](https://www.alphaxiv.org/abs/2610.04157)

### Inferences
- **CG-Diff's median CPR of 0.3 is a warning for any "graph-closure" story (I11).** In real PRs, most changed functions are *not* call-connected. Siblings, containment and new code may dominate multi-location fixes. If so, the graph of the fix is not the call graph, and I1's motif census is the step that tells us which graph matters. INFERRED; it is about PRs, not bug fixes.
- **I2 is the keystone.** Its partial-patch labels are prerequisites for I3 (negatives), I4 (credit validation), I12 (relabelling) and I16 (intervention oracle). It is pure CPU and reuses the scorer env built for the dev set. INFERRED.
- **I10 is the cheapest decisive diagnostic.** "Never seen" vs "seen but not edited" splits the L3 dead zone into an exploration problem and a completion problem, and tells us whether I1/I11 or I3 should carry the method. INFERRED.
- **2608.18280's AUC 0.863 without graph features sets a hard baseline** for any graph-difficulty claim (I6). Graph features must add over hunk counts, which may already proxy fix spread. INFERRED.
- **A coherent paper is possible:** "Fixes are subgraphs: the causal and structural anatomy of multi-location repairs, and structure-aware post-training that closes them." Measurement (I1 + I2 + I10) leads straight into a method (I3 and/or I4). That avoids the vetoed "measurement alone" and "structure over names" framings. INFERRED.

### Gaps
- None of the classic SE/PL mechanism papers were read this session: Littman 1986 content, delta debugging, tangled changes, Causal Testing, GraphCL, HER, dominators, AutoSD. Their claims here are INFERRED from memory and need checking before citation.
- I did not verify whether SWE-bench Verified / SWE-Gym gold patches are trivially downloadable offline for the I1 density set (INFERRED yes).
- Raw r2/r3 trajectories and agent patches are needed for I4, I8, I10 and I12. The critic marks their retrieval from Kaggle outputs as GUESSED.
- I found no published distribution of necessity/sufficiency of gold-patch hunks, so I2's predictions have no prior anchor.
- DPO/preference training specifically for SWE agents was not searched in depth. The absence claim for I3 is limited to "partial-gold-patch negatives".

---

## Q2. Ranking: which 5 to pursue, and why could each still fail?

### Takeaway
Ranked by novelty × payoff × feasibility, each scored 1–3:

1. **I2** causal fix-set structure: N3 P3 F3
2. **I10** as-needed vs systematic diagnostic → impact-triggered systematic mode: N2 P3 F3
3. **I1** fix-motif census + motif operators: N2 P2 F3
4. **I3** node-dropping incomplete-patch negatives: N3 P2 F2
5. **I4** edit-ablation counterfactual credit: N2 P2 F2

Honourable mention: I6 (graph difficulty), for its graph-ML vocabulary. I1, I2 and I10 share one CPU kernel, about 2 working days in total, and their outcomes decide between the I3 and I4 training methods.

### Ranking table

| Rank | Idea | N | P | F | Product | One-line case |
|---|---|---|---|---|---|---|
| 1 | I2 Causal fix-set structure (partial-patch interventions) | 3 | 3 | 3 | 27 | No prior hit. CPU only. Directly explains L3. Produces labels for every training idea. |
| 2 | I10 As-needed vs systematic (seen vs never-seen missed locations) | 2 | 3 | 3 | 18 | ~3 h of CPU. Decides the method. Distinctive cognitive-science framing. |
| 3 | I1 Fix-motif census + motif-conditioned completion | 2 | 2 | 3 | 12 | Native graph-ML analysis with dense public data. Partially pre-empted by CG-Diff and SiblingRepair. |
| 4 | I3 Structurally incomplete negatives (node dropping) | 3 | 2 | 2 | 12 | A real post-training method on the L3 pool. Needs I2 labels and ~4–8 GPU-h. |
| 5 | I4 Edit-ablation counterfactual credit | 2 | 2 | 2 | 8 | Cheap, exact credit. Close to CRR in spirit. Needs agent patches from Kaggle. |
| — | I6 Resistance/Steiner difficulty | 2 | 1 | 2 | 4 | Good vocabulary, but measurement only (vetoed as a headline). Use as an analysis section. |
| — | I5 Dominator fix-site | 2 | 1 | 2 | 4 | Shares the Stage A tracer. Keep as a sub-analysis. |
| — | I8 PFIS foraging model | 2 | 1 | 2 | 4 | Targets loops, not L3. Needs raw traces. |
| — | I12, I16 | 2 | 2 | 1–2 | ≤8 | Both depend on I2. Fold them in as extensions. |
| — | I7, I9, I11, I13, I14, I15 | ≤2 | ≤2 | ≤2 | ≤4 | Crowded (I9, I11), thin (I13), or infeasible to evaluate (I7, I14, I15). |

### Why each top-5 idea could still fail (honest)

- **I2 could fail because:**
  - (a) Too few scorer-faithful multi-function tasks. Only ~54 tasks pass gold, and perhaps ~20 of them have ≥2 functions, so CIs will be about ±20 pp.
  - (b) Function-granularity subsets may not apply cleanly (hunks that depend on each other syntactically, e.g. a new helper used by an edited function). Then subsets fail for import or name reasons, not causal ones, and the labels are contaminated.
  - (c) Weak F2P tests, per 2607.28871: a "sufficient" subset may pass tests without fixing the issue. Our labels would then be test-relative, not semantic.
  - (d) If the result is "all conjunctive", the finding is real but the graph hook (interaction vs hop distance) may be null.
  - (e) Reviewers may say "this is delta debugging on gold patches: an SE analysis, not graph ML".
- **I10 could fail because:**
  - (a) "Viewed" is hard to measure from grep/sed outputs: partial views, truncated 5k outputs.
  - (b) It needs the raw traces, which may not be retrievable.
  - (c) If the answer is "seen but not edited", systematic-mode training is pointless and the cognitive framing loses its payoff, leaving only a diagnostic.
  - (d) The Littman claim is recalled, not verified. If the 1986 paper says something different, the framing must change.
- **I1 could fail because:**
  - (a) CG-Diff already suggests most changed functions are not call-connected (median CPR 0.3). The motif census may say "mostly disconnected", which leaves no structural operator.
  - (b) Our graph has one mixed `calls` label and no imports or inheritance edges, so the sibling and containment motifs must be built from `ast` anyway. "Graph reasoning" then becomes "AST heuristics".
  - (c) Motif features may not beat hunk counts (2608.18280's baseline).
- **I3 could fail because:**
  - (a) Gemma may already discriminate full from partial patches when both are shown, so the failure is upstream (never generating the second edit), and pairwise training does not change generation.
  - (b) DPO on a 4-bit 31B with LoRA under 30 GPU-h a week is unproven (post_training_navigation.md Q4: no LoRA-RL result on 4-bit ~30B multi-turn).
  - (c) It may teach over-editing. The "Models Edit Too Much" result (2609.04061, SNIPPET) shows minimality matters, and could hurt single-function tasks.
- **I4 could fail because:**
  - (a) Most agent patches in the dead zone are single-edit or empty, so there is nothing to ablate.
  - (b) Credit may be all-or-nothing for most patches, which collapses to outcome RFT.
  - (c) Re-running tests per edit inside a 4 GiB / 2 vCPU sandbox for hundreds of rollouts costs CPU-hours, not GPU, but the time is still real.
  - (d) A reviewer may see it as a minor variant of CRR or leave-one-out baselines.

### Cheapest joint plan (INFERRED)
1. **Day 1 (CPU):** one Kaggle kernel runs I2 (partial-patch interventions) and I1 (motif census on our 129 tasks plus SWE-bench Verified gold patches). If the critic's Stage A tracer is ready, add I5 as a by-product.
2. **Day 2 (CPU):** fetch the r2/r3 raw traces and run I10 (seen vs never-seen) and the I4 degeneracy check.
3. **Decision:**
   - I2 finds "core + incidental" dominant → the method is core-finding (I16 interventions / diagnosis training).
   - I2 conjunctive + I10 seen-but-not-edited → I3 (incomplete negatives).
   - I2 conjunctive + I10 never-seen + I1 connected motifs → I1 operators / I11 closure.
   - Everything killed → the paper falls back to the critic's H1 (EWEC), with I2 + I1 as its analysis section.

### Cited Findings
- Over-editing harms the fidelity of minimal code edits, which bears on I3's risk. SNIPPET — [When Models Edit Too Much 2609.04061](https://www.alphaxiv.org/abs/2609.04061)
- Weak validation tests in repair agents bear on I2's label validity. SNIPPET — [2607.28871](https://www.alphaxiv.org/abs/2607.28871)
- CG-Diff's low median call connectivity bears on I1/I11 risk. VERIFIED — [2609.32057](https://www.alphaxiv.org/abs/2609.32057)
- Static patch features already predict agent success at AUC 0.846 alone, which bears on I1/I6 risk. VERIFIED — [2608.18280](https://www.alphaxiv.org/abs/2608.18280)

### Inferences
- The product ranking deliberately favours CPU-only, label-producing analyses. Under ~30 GPU-h a week, any GPU idea that cannot be pre-filtered on CPU is a bad bet before Nov 5. INFERRED.
- I2 + I3 together form the most "graph-ML-native post-training" story: execution defines which nodes of the fix subgraph are necessary, and contrastive training on node-dropped subgraphs teaches completeness. A judge will map this onto GCL augmentations and causal graph learning. INFERRED.

### Gaps
- No effect-size anchors exist for any top-5 idea on multi-location fixes. All predicted sizes would be GUESSED.
- The power of I2 on ~20 multi-function scorer-faithful tasks is weak. Building injected multi-function bugs (SWE-smith style) with known causal structure by construction would fix this. I did not check whether SWE-smith already ships multi-function injections (INFERRED partially: it has "combine" strategies, unverified).

---

## Q3. Novelty evidence: what did the searches find? (Search log)

### Takeaway
14 literature tool calls this session (10 alphaXiv `discover_papers`, 2 alphaXiv full reads, 2 WebSearch), plus one PDF fetch. Clear absences for:
- partial-gold-patch necessity labels (S-K returned zero papers);
- dominator-based fix sites (S-F);
- information-foraging models of coding agents (S-H);
- systematic/as-needed comprehension in LLM agents (S-G);
- GNN → LLM-agent distillation for code edits (S-L).

Crowded: co-edit and next-edit prediction (S-J), SWE credit assignment via counterfactual rollouts (S-C), and multi-hunk repair systems (S-A, S-B). Inherited searches S1–S17 from critic_judgement.md were not re-run.

### Cited Findings (search log)

| ID | Tool | Query (+ keywords) | Main results | Bearing |
|---|---|---|---|---|
| S-A | alphaXiv discover | "Which functions of a multi-location gold patch are causally necessary: applying subsets of a gold patch (partial patches)… or using partial/incomplete patches as hard negatives for preference training"; kw [partial patch, multi-hunk, gold patch, DPO, program repair] | Validation Evidence 2607.28871; SiblingRepair 2605.06209; MultiFixer 2607.26591; Hallucination in APR 2609.04909; Deterministic fusion of repair candidates 2607.01597; PracRepair 2606.17612 | No partial-patch necessity labels or incomplete-patch negatives → supports I2/I3 novelty |
| S-B | alphaXiv discover | "Structure of multi-location bug fixes on the call graph: taxonomy… Steiner tree or graph distance… as difficulty"; kw [multi-location, co-change, call graph, Steiner tree, SWE-bench] | **CG-Diff 2609.32057**; SiblingRepair; multi-file change localisation exploration 2606.11976; Models Edit Too Much 2609.04061; SWE-Review; SWE-Explore | CG-Diff is the closest to I1 (PR call connectivity). No fix-motif census linked to agents |
| S-C | alphaXiv discover | "Credit assignment for SWE agent training by reverting individual edits and re-running tests… or hindsight relabeling of failed partial fixes"; kw [credit assignment, counterfactual, hindsight, SWE agent, edit] | CRR 2609.33875; SWE-MILE 2609.32631; FC-SWE 2610.07898; Dependency-aware PO 2610.03634; GraphHCA 2609.35084; SHARPO 2610.00838; ASCT 2609.35215; ExecuCritic 2609.16604; Credit Without Ground Truth 2608.19760 | Counterfactuals are over actions/rollouts, not patch hunks → I4 N=2. No SWE test-subset HER → I12 weakly novel |
| S-D | alphaXiv discover | "Information foraging theory or marginal value theorem patch-leaving rule applied to LLM agents exploring code repositories…"; kw [information foraging, marginal value theorem, LLM agent, code navigation] | Not Worth Another Token 2608.08389; Context Gathering POMDP 2605.07042; SemNav 2609.31176; Latent Programming Horizons 2607.05188; Bayesian control for coding agents 2606.24453 | MVT stopping exists for research agents → I9 low novelty |
| S-E | alphaXiv discover | "Graph-theoretic difficulty measures (effective resistance, commute time, over-squashing, curvature) predicting LLM agent… failure"; kw [effective resistance, over-squashing, LLM, code repository] | Sheaf ER 2610.04157; Reachability is not enough 2609.33674; ER rewiring 2603.11944; GraphEcho; Predicting Task Difficulty Without Rollouts 2608.05797; GRADE 2606.22741 | No resistance-based difficulty for agentic repair → I6 open, but measurement only |
| S-F | alphaXiv discover | "Choosing the fix location… using dominator or post-dominator trees, common callers of failing paths, or the root of a propagation chain"; kw [dominator tree, fix location, program repair, root cause] | CausalRepair 2608.10613; Pecker 2603.02583; CT-Repair 2607.12605; PathFix 2510.14341; SemLoc 2603.29109; EffiHolmes 2608.03558 | No dominator-based fix-site prior → I5 N=2 |
| S-G | alphaXiv discover | "Programmers' systematic versus as-needed comprehension strategy… applied to LLM coding agents"; kw [as-needed, systematic strategy, program comprehension, coding agents] | Gaze transitions for comprehension strategies 2607.01042; Complexity-aware LLM comprehension 2609.37405; developer–agent misalignment 2605.29442; code review comprehension 2503.21455 | No application to agents → I10 framing novel |
| S-H | alphaXiv discover | "Modeling LLM coding agent navigation with information foraging theory or spreading activation (PFIS)…"; kw [information foraging, PFIS, spreading activation, coding agent] | Information-scent navigation model 2603.11759; Scent of Knowledge 2505.09316; query-aware spreading activation 2606.30133; SYNAPSE 2601.02744; SCOUT 2605.04496; architecture descriptors for agents 2604.13108 | IFT is used for RAG/web, not coding agents → I8 N=2 |
| S-I | alphaXiv discover | "Curriculum learning for SWE agent RL ordered by structural task difficulty… decomposing multi-function gold patches"; kw [curriculum, SWE agent, task difficulty, multi-function, reinforcement learning] | Socratic-SWE 2606.07412; Active Curriculum Refinement 2608.26469; **What makes SWE tasks difficult 2608.18280**; SWE benchmark profiling 2609.01271; SWE-rebench V2; SWE-Lego | Difficulty is predicted from static features with no graph → I6/I7 openings, with a strong baseline |
| S-J | alphaXiv discover | "Next edit location prediction after a prior edit in a repository (edit propagation, CoEdPilot-style)…"; kw [next edit, edit propagation, CoEdPilot, related edits] | CoEdPilot 2408.01733; subsequent code edits 2604.12220; GitCF 2609.37045; change dependencies on OpenStack 2508.05034; co-changed method L2R 2411.19099; Neural Change Prediction 2606.03378; Coeditor 2305.18584; NES 2508.02473 | Co-edit prediction is crowded → I11 N=1 |
| S-K | alphaXiv discover | "Which hunks of a fix are necessary: delta debugging or minimization of a patch… untangling composite commits…"; kw [delta debugging, patch minimization, untangling, hunks] | **No papers found** | Absence supports I2 (alphaXiv coverage of older SE venues is thin; classic untangling work exists from memory, INFERRED) |
| S-L | alphaXiv discover | "Distilling a graph neural network's predictions into an LLM through fine-tuning targets…"; kw [GNN-to-LLM, knowledge distillation, graph neural network, LLM] | InfGraND 2601.08033; CoT distillation for fraud 2601.22949; LLM-guided KD for TKG 2602.14428; GRIP 2511.07457; LLM as GNN 2503.03313; DistillGuard 2609.28996 | No code-edit / agent instance → I14 N=2 (feasibility weak) |
| S-M | alphaXiv discover | "Non-backtracking random walks or revisit penalties to reduce repetitive loops in LLM tool-using agents…"; kw [non-backtracking, repetition, loops, LLM agent] | GraphEcho; StateFork; Redundant-steps benchmark 2605.29893; NBRW theory 2608.26321, 2608.09729; BEAP-Agent; RE-TRAC | Only theory and redundancy benchmarks → I13 framing only |
| S-N | alphaXiv discover | "Predicting co-edit locations… graph neural networks or change-impact propagation… CodePlan"; kw [co-edit, change impact analysis, CodePlan, graph neural network, repository-level editing] | CodePlan 2309.12499; transformers + PDG impact analysis 2607.23355; graph-propagated impact analysis vision 2606.18855; Neural Change Prediction; GREPO 2602.13921; CG-Diff; partial dependency graph retrieval 2608.01927; GPT-5 change-impact dataset 2512.19481 | Change-impact + learning is established → I11/I15 limited novelty |
| S-O | WebSearch | `Littman Pinto Letovsky Soloway 1986 mental models software maintenance systematic as-needed strategy` | NASA NTRS record; dblp; Kent State slides; CACM 1988 delocalized-plans paper listed | Citation VERIFIED. The as-needed finding was not shown (INFERRED) |
| S-P | WebSearch | `information foraging theory debugging Lawrance Burnett "information scent" programmers navigation` | TSE 2013 IFT-of-debugging (Memphis repository, Burnett reprint); PFIS, PFIS2 | SNIPPET support for I8 |
| R-1 | alphaXiv get_paper_content | 2608.18280 (task difficulty) | Full report read; numbers above | VERIFIED baseline for I1/I6 |
| R-2 | alphaXiv get_paper_content | 2609.32057 (CG-Diff) | Full text read; CPR stats, necessity coding | VERIFIED; key risk for I1/I11 |
| F-1 | curl + pypdf | Kent State "Mental Models for Program Understanding" slides | Lists opportunistic vs systematic strategies, no Littman results | Not a primary source; not used for claims |

### Inferences
- The absence findings rest on alphaXiv's arXiv-centric index and a single query each. Older SE venues (MSR/ICSE 2010–2015 untangling and tangled-changes work) are under-indexed, so the I2 novelty claim must be phrased as "no work uses partial-gold-patch interventions to label or train **LLM agents**". A claim of "no one has minimised patches" would be wrong (INFERRED from memory of Herzig & Zeller, Kawrykow & Robillard).

### Gaps
- Not searched this session: OpenReview ICLR-2027 submissions (likely not public yet), ACL Anthology, ICSE/FSE 2026 proceedings directly, and Semantic Scholar citation graphs of CG-Diff and SiblingRepair (which might reveal a concurrent fix-motif study).
- No full-text read of SiblingRepair or MultiFixer this session. Their exact claims about the share of multi-hunk bugs that are "sibling" are unknown here.
