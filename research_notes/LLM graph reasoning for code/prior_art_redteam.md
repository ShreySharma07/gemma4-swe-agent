# Prior-art red team for H1 (execution-witnessed chain reward) and H2 (graph co-change completion)

Written 8 Oct 2026. Role: a hostile reviewer trying to find the paper that already did it. Searches S1–S17 in `critic_judgement.md` were not repeated; this log starts at R1.

Labels:
- **VERIFIED**: I read the relevant pages of the source this session.
- **SNIPPET**: taken only from a search-result snippet or abstract.
- **INFERRED**: my own reasoning, or recalled from memory and not re-read this session.

Closeness scale for H1 and H2:
- 0 = unrelated
- 1 = shares one ingredient
- 2 = same mechanism, different check / domain / setting
- 3 = already did it

---

## Q1. Execution-trace-aware code-LLM training: does any of it reward a stated chain against the trace at repo level?

### Takeaway
**One new paper comes very close to H1:** Comet-9B (arXiv 2609.34359, 28 Sep 2026). It post-trains a repo-level SWE agent with rejection-sampled SFT plus agentic GRPO. The agent must write an **"Evidence-Grounded Bug Propagation Chain"**, and the reward checks it.

The check is an **LLM judge** that has the gold patch and tests. It is **not** an edge-by-edge comparison against a dynamic call trace. Everything else in the trace-training literature either:
- trains on traces as input or output for standalone functions (NExT, SemCoder, TRACED, CodeExecutor, ExecVerify and similar), or
- uses traces only at inference time (LDB, SWE-Doctor, ADI, XAgent, CT-Repair, EviACT).

### Cited Findings

**Comet-9B (2609.34359), closeness to H1 = 2 (high 2), VERIFIED**
- **Task.** "Precondition–postcondition reasoning" in real repositories (SWE-Gym 2,438 issues, SWE-ReBench-Lite 1,091), with Bash access. The agent keeps a reasoning file with 7 required sections, including "Evidence-Grounded Bug Propagation Chain", and writes a regression test. — [Comet-9B](https://www.alphaxiv.org/abs/2609.34359)
- **How the chain is checked.** An LLM judge "with access to the ground-truth tests and the gold patch" scores precondition, postcondition and **propagation correctness**. In the RL continuation phase it also scores "repository grounding" ("claims are supported by the code and execution evidence available to the agent"). The total q∈[0,10] gives r_reason = q/10. It is weighted equally with an execution reward (the test fails before the fix, passes after it, and adds no new failures). — [Comet-9B §3.4, App. D.1](https://www.alphaxiv.org/abs/2609.34359)
- **Training.**
  - SFT data comes from GPT-5.5 / Qwen3.5-397B teachers with rejection sampling. A trajectory is kept only if its test passes the execution check **and** its explanation is accepted by the judge. This gives 1,138 trajectories over 617 tasks.
  - GRPO then runs on 567 tasks. — [Comet-9B §3.3–3.4](https://www.alphaxiv.org/abs/2609.34359)
- **Results (Qwen3.5-9B base).**
  - Adding the reasoning tasks to SFT gives +7.25 pp on SWE-bench Pro and +9.70 pp on SWT-Bench Verified.
  - Sequential RL adds a further +7.25 pp (Pro), +26.79 pp (SWT) and +4.67 pp (CyberGym).
  - Final scores: 30.51% on SWE-bench Pro, 45.50% on SWT-Bench Verified. — [Comet-9B abstract/§1](https://www.alphaxiv.org/abs/2609.34359)
- **What it does not do.** There is no call graph, no dynamic call trace and no edge-level scoring. The propagation is free text graded by an LLM. It does not report a chain-accuracy metric against execution. — VERIFIED by absence in the pages read (§3, App. C–D); INFERRED that it is absent from unread appendices.
- **Its related-work list confirms the earlier trace-training line is standalone-program only:**
  - T-REX (next-state prediction)
  - SemCoder and CodeI/O (I/O prediction)
  - NExT (repair with traces)
  - CodeReasoner (RL on I/O prediction)
  - ExecVerify (rewards the correct next executed statement, variable values and output)
  - StepCodeReasoner and CodeThinker (intermediate-state rewards)
  - CodeRL+
  - Maimon et al. 2026

  It says these "primarily train on standalone programs rather than through interaction with repositories." — [Comet-9B §2](https://www.alphaxiv.org/abs/2609.34359)

**NExT (Ni et al., ICML 2024, 2404.14662), closeness 1, VERIFIED**
- The traces are **input** (line-level variable states as inline comments).
- Rationales are bootstrapped by self-training. Samples are filtered **only by whether the fix passes unit tests**. The authors say "this filtering criteria does not directly consider rationale quality."
- Data is MBPP-R / HumanEvalFix-Plus, which is function-level, not repo-level.
- Results: PaLM 2-L +26.1 pp pass@1 on MBPP-R. — [NExT](https://arxiv.org/abs/2404.14662)

**SemCoder (Ding et al., NeurIPS 2024, 2406.01006), closeness 1, SNIPPET + VERIFIED-as-described-by-Comet**
- Trains output/input prediction and "monologue" semantics reasoning on standalone programs.
- There is no chain-versus-trace reward at repo level. — [SemCoder](https://www.alphaxiv.org/abs/2406.01006); [Comet-9B §2](https://www.alphaxiv.org/abs/2609.34359)

**ExecVerify / StepCodeReasoner / CodeThinker (2026), closeness 1, VERIFIED only as Comet describes them**
- These are the only works I found whose reward checks *intermediate* execution claims against real execution (next statement, variable values).
- All are on given code + input, not repositories, and not call chains. — [Comet-9B §2](https://www.alphaxiv.org/abs/2609.34359)

**TraceLift, "Correct Is Not Enough" (2605.03862), closeness 1 (method lesson), VERIFIED**
- The reasoning reward is a trained rubric reward model multiplied by measured uplift to a frozen executor. It is not checked against execution traces. Tasks are HumanEval, MBPP, LCB and math.
- **Useful for H1's defence:** a direct LLM-as-judge rubric reward saturated. 573/600 on-policy samples got a perfect score (95.5%), so it was non-discriminative inside GRPO groups. It also underperformed the trained reward model by 3.34 pp. — [TraceLift App. 8.3–8.4](https://www.alphaxiv.org/abs/2605.03862)

**Classic trace-training work, closeness 0–1, INFERRED (recalled, not re-read)**
- TRACED (Ding et al., ICSE 2024): execution-aware pre-training on traces (coverage, variable values).
- CodeExecutor (Liu et al., ACL 2023): pre-training to predict execution traces.
- "Learning to Execute" (Zaremba & Sutskever 2014).
- None are repo-level agents, and none check a stated chain.

**LDB / "Debug like a Human" (Zhong et al., 2024), closeness 1, INFERRED**
- Inference-time prompting that segments execution into basic blocks and verifies intermediate values. No training.

**Inference-only runtime-diagnosis work (SWE-Doctor, ADI/FramePilot, XAgent, CT-Repair, EviACT), closeness 1–1.5, VERIFIED in `critic_judgement.md` S3, S4, S7, S16, S17**
- SWE-Doctor's runtime-grounding check (it rejects text-only diagnoses) is the nearest *check*, but it is prompt-only.

### Inferences
- **The novelty boundary has moved since the critic wrote "not found (S2–S14)".** A reviewer will now cite Comet-9B for: "trained a repo-level SWE agent, by rejection sampling and RL, to state a bug propagation chain that is rewarded for correctness and grounding." The claim "first to reward a stated causal chain at repo level" is **dead**. INFERRED.
- **What survives is narrower.** The surviving claim is *how* the chain is verified: a programmatic, edge-level check against an execution-witnessed caller→callee graph from the failing tests, with partial credit for static-only edges. Comet uses an LLM judge with the gold patch, which is privileged and costly, and which TraceLift shows can saturate. INFERRED.
- **H1 must now include a "Comet-style LLM-judge reward" arm.** It needs to beat that arm (or match it at far lower cost), not just beat outcome-only RFT. Without this arm, an informed reviewer rejects. INFERRED.
- **Comet weakens H1's premise (i) in one respect.** It shows that propagation-chain training already transfers to SWE-bench Pro at 9B. The marginal question is therefore whether a structural verifier beats a judge, not whether chain training helps. INFERRED.

### Gaps
- I read Comet-9B §1–5 and App. C–D.1. I did not read the judge prompt (G.1.8) or the example reasoning file (H). The judge could, in principle, be told to check cited functions against observed tracebacks. That would move closeness to 2.5.
- Full text of Socratic-SWE 2606.07412 and FailForge 2608.08570 was still not read (flagged by the critic). Snippets do not suggest chain verification.
- ExecVerify, StepCodeReasoner and CodeThinker were seen only through Comet's citations. I did not locate their arXiv IDs.

---

## Q2. Process rewards checked against program analysis (static or dynamic) for agents, 2025–2026

### Takeaway
**Program-analysis-checked process rewards exist, but none use a dynamic call trace to score an SWE agent's chain.** Two works check against static structure:
- VulAgentRL checks against a static code property graph (CPG).
- ExecVerify-style rewards check line-level execution on standalone code.

The rest of 2026 SWE process rewards are credit-assignment schemes: rubric reward models, counterfactual forks, milestones and failure-conditioning. None of them is a structural verifier.

### Cited Findings
- **"Graph Is the Verifier" / VulAgentRL (2607.26656).** Process reward = F1 of the agent's cited CPG node IDs against a static code property graph, for vulnerability detection. Closeness to H1 = 2 on the "graph-checked evidence reward" axis, but static and not SWE. — VERIFIED in the inherited notes (critic C7); [arXiv 2607.26656](https://www.alphaxiv.org/abs/2607.26656)
- **ExecVerify (2026).** Rewards correct predictions of the next executed statement, variable values and types, and output, for given code and input. This is a dynamic check but not a call chain or an agent. — VERIFIED only as described in [Comet-9B §2](https://www.alphaxiv.org/abs/2609.34359)
- **Recent SWE-agent process / credit-assignment rewards.** I saw only titles and abstracts, and none mention a dynamic-trace or call-graph check:
  - SWE-TRACE (rubric process reward model, 2604.14820)
  - Counterfactual Rollout Replay (2609.33875)
  - SWE-MILE (milestone potentials, 2609.32631)
  - FC-SWE (failure-conditioned RL, 2610.07898)
  - A²Agent (action-aware RL for localisation, 2608.29831)
  - ExecCritic (2609.09133)

  SNIPPET — [SWE-TRACE](https://www.alphaxiv.org/abs/2604.14820), [CRR](https://www.alphaxiv.org/abs/2609.33875), [SWE-MILE](https://www.alphaxiv.org/abs/2609.32631), [FC-SWE](https://www.alphaxiv.org/abs/2610.07898), [A²Agent](https://www.alphaxiv.org/abs/2608.29831), [ExecCritic](https://www.alphaxiv.org/abs/2609.09133)
- **Agent-trace verification work** (AgentLTL 2607.02599, VICT 2608.28128, GRADE 2606.22741) checks the *agent's own tool-call trace*, for example temporal-logic compliance. It does not check the *program's* execution. Closeness 0–1. — SNIPPET — [AgentLTL](https://www.alphaxiv.org/abs/2607.02599), [VICT](https://www.alphaxiv.org/abs/2608.28128), [GRADE](https://www.alphaxiv.org/abs/2606.22741)

### Inferences
- **No search found the exact combination:** dynamic caller→callee graph from F2P tests, edge-level precision/recall of the agent's stated chain, and use as an SFT filter or RL reward for an SWE agent. Across S1–S17 plus R1–R8 (about 25 searches), absence holds. Coverage of OpenReview ICLR-2027 submissions is still nil. INFERRED.
- **The cleanest framing for a graph-ML reviewer** is a three-way comparison of verifier types for the same chain:
  1. **Static** (VulAgentRL-like)
  2. **Witnessed** (H1)
  3. **LLM judge** (Comet-like)

  The static-versus-witnessed analysis (Stage A, M4) becomes the empirical core. INFERRED.

### Gaps
- OpenReview was not searchable, and arXiv coverage of the last ~10 days may lag (the Comet paper is only 10 days old).
- GitHub code search of verl, SkyRL and OpenHands RL recipes for trace-based reward functions was not done.

---

## Q3. Co-change / change-impact prediction with call graphs (+LLMs): how close to H2?

### Takeaway
**H2's mechanism is not novel.** "After one edit, propose graph-adjacent functions as further edit sites" is CodePlan's change-may-impact analysis (2023, FSE'24). SynFix applies one-hop RelationGraph synchronisation to SWE-bench (ACL Findings 2025). TRACE (2604.12220), PReMM (OOPSLA'25) and MultiFixer (ASE'26) cover nearby variants.

**The empirical literature is also unfriendly to the premise.** Call-graph links are a weak co-change predictor next to history and path, and changed functions in typical PRs are mostly *not* call-connected.

### Cited Findings

**CodePlan (2309.12499), closeness to H2 = 3 on mechanism, VERIFIED**
- After each LLM edit, it classifies the change and updates a dependency graph. A change-may-impact analysis then marks affected blocks as new edit obligations:
  - callers when an escaping object or signature changes;
  - overrides and overridden-by;
  - field users.
- It handles transitive propagation.
- Evaluated on C# migration and Python temporal edits with GPT-4-32k: 5/6 repos pass validity checks versus 0 for oracle-guided baselines.
- It explicitly catches call sites a type checker does not flag. — [CodePlan](https://www.alphaxiv.org/abs/2309.12499)

**SynFix (ACL Findings 2025, arXiv 2503.21710), closeness 3 on mechanism and same benchmark family, SNIPPET**
- Builds a RelationGraph (classes, functions, variables; imports, inheritance, dependencies).
- "By analyzing one-hop nodes in the RelationGraph, SynFix ensures repairs account for dependent updates."
- Results: 52.33% on SWE-bench Lite, 55.8% on Verified. — [ACL Anthology](https://preview.aclanthology.org/setup/2025.findings-acl.252); [arXiv](https://arxiv.org/pdf/2503.21710v2)

**TRACE, subsequent-edit prediction (2604.12220), closeness 2, VERIFIED**
- Given prior edits, it predicts the next edit location by interleaving a neural locator with LSP "edit compositions": rename, def-use propagation, clone, diagnostics.
- 53.6% of 38K commits contain at least one LSP-detectable edit composition.
- Real-edit simulation: MR@1 is 35.18% versus 11.47% for CoEdPilot.
- Def-use is not the same as a ≤2-hop call-graph closure, but the "after edit k, propose edit k+1 from static relations" framing is the same. — [TRACE](https://www.alphaxiv.org/abs/2604.12220)

**CoEdPilot (2408.01733), closeness 2, SNIPPET**
- Project-wise edit-location recommendation from prior edits (learned relevance). — [CoEdPilot](https://www.alphaxiv.org/abs/2408.01733)

**PReMM (OOPSLA 2025), closeness 2, SNIPPET**
- For multi-method bugs it clusters faulty methods "based on the dependence relationship among them" for divide-and-conquer repair. This assumes the faulty methods are already localised.
- Result: 307 correct fixes on Defects4J. — [SPLASH 2025](https://2025.splashcon.org/details/OOPSLA/115/PReMM-LLM-Based-Program-Repair-for-Multi-Method-Bugs-via-Divide-and-Conquer)

**MultiFixer (2607.26591), closeness 2, VERIFIED**
- The Coordinator builds a hunk-dependency graph with static-analysis tools plus the bug report. Its relation types include "Change Propagation", for example a signature change requiring call-site edits.
- The graph schedules the repair order.
- Buggy hunks are given as input, so this addresses ordering and coordination, not *discovery* of missing locations.
- Results on Defects4J: 326/835, including 62 multi-method and 27 multi-file bugs. — [MultiFixer](https://www.alphaxiv.org/abs/2607.26591)

**SiblingRepair (2605.06209), closeness 1–2, VERIFIED; evidence against a pure call-graph premise**
- Siblings are found by TF-IDF plus embedding similarity over test-exercised statements, not by the call graph.
- **For 61% of Defects4J sibling bugs, the failing tests do not cover all siblings.**
- Commit history fills the gap for only 17% of those bugs.
- Results: 23 sibling and 76 multi-hunk bugs repaired under SBFL. — [SiblingRepair](https://www.alphaxiv.org/abs/2605.06209)

**Learning-to-rank co-changed methods (Jia et al., 2411.19099), VERIFIED; evidence against H2's premise**
- 150 Java projects, 634K PRs. Random forest reaches NDCG@5 = 0.84.
- Permutation importance by feature:

  | Feature | Importance |
  |---|---|
  | **Co-change history** | **0.384** |
  | Path similarity | 0.054 |
  | Authors | 0.039 |
  | **Code dependency (method calls)** | **0.0082** |

- File-proximity ranking alone reaches NDCG@5 = 0.64. — [Jia et al.](https://www.alphaxiv.org/abs/2411.19099)

**CG-Diff (2609.32057), VERIFIED; mixed evidence**
- Dataset: 7,490 Python PRs from 77 repos (≥10k stars).
- Median Cluster Participation Ratio is **0.3**: typically only 30% of changed functions are call-connected to another changed function.
- In 39.6% of PRs, more than half are call-connected.
- Necessary callees are split across files about half the time (58/110). — [CG-Diff](https://www.alphaxiv.org/abs/2609.32057)

**Change-impact analysis with transformers + program dependence graphs (2607.23355)** and the **JavaScript history + runtime change-impact study (2606.21187)** confirm the impact-analysis line is active. Closeness 1–2. — SNIPPET — [2607.23355](https://www.alphaxiv.org/abs/2607.23355), [2606.21187](https://www.alphaxiv.org/abs/2606.21187)

**GraphLocator (2512.22469).** Handles one-to-many (symptom→multiple causes) with a causal issue graph, prompt-only. Closeness 2. — VERIFIED in `critic_judgement.md` S8

### Inferences
- **As a method, H2 is dead to a graph-ML reviewer.** CodePlan and SynFix already do "propagate to ≤1–k graph neighbours after an edit," and SynFix does it on SWE-bench. A ≤2 non-hub-hop variant is a hyperparameter of known work. INFERRED.
- **What remains is measurement specific to Gemma's failures.** H2 could ask: on *failed* multi-function tasks, do the missing gold functions lie within 2 non-hub hops of the edited one, beating a same-file null? That is a diagnostic result, not a method contribution. INFERRED.
- **Expect the critic's kill rule "AUC < 0.6 vs same-file null" to fire.**
  - Jia et al. find call dependency adds roughly nothing beyond path and history.
  - CG-Diff's median CPR of 0.3 suggests most co-changed functions are not call-adjacent.
  - SiblingRepair's siblings are similarity-related, not call-related.

  This is INFERRED. Those datasets are Java PRs, Python PRs and Defects4J, not Gemma's failed SWE tasks.

### Gaps
- I did not read SynFix's full text, so its exact hop definition, ablations and edit-time triggering are unknown. Only the abstract-level description is available.
- I found no paper that reports the call-graph hop-distance distribution among gold functions in SWE-bench multi-function patches specifically. That measurement may still be a small empirical contribution.
- Hercules (sibling repair) and DEAR (multi-hunk deep-learning repair) are recalled from memory and were not checked (INFERRED).

---

## Q4. Verdict: closest prior work, remaining delta, would a graph-ML reviewer accept?

### Takeaway
- **H1: weakened but alive.** Closest prior is Comet-9B (2609.34359, closeness 2). The remaining delta is a programmatic, graph-structured verifier (execution-witnessed caller→callee edges with partial credit for static edges) replacing an LLM judge with gold-patch access. A graph-ML reviewer would accept it only if the paper shows witnessed-edge rewards beating both a static-graph reward (VulAgentRL-style) and an LLM-judge reward (Comet-style), plus the static-versus-witnessed graph analysis.
- **H2: not novel as a method.** Closest prior: CodePlan (closeness 3) and SynFix (closeness 3 on SWE-bench). It survives only as a diagnostic measurement, and existing empirical evidence predicts it will fail its kill test.

### Cited Findings
- Comet-9B rewards an "Evidence-Grounded Bug Propagation Chain" via an LLM judge plus execution checks, at repo level, with SFT rejection sampling and GRPO. — [Comet-9B](https://www.alphaxiv.org/abs/2609.34359)
- LLM-judge reasoning rewards can saturate (95.5% perfect scores) and underperform a calibrated reward model. — [TraceLift](https://www.alphaxiv.org/abs/2605.03862)
- NExT filters rationales only by test pass and does not check them against traces. — [NExT](https://arxiv.org/abs/2404.14662)
- CodePlan propagates edits to graph-dependent code after each edit. — [CodePlan](https://www.alphaxiv.org/abs/2309.12499)
- SynFix uses one-hop RelationGraph neighbours on SWE-bench (52.33% Lite). — [SynFix](https://preview.aclanthology.org/setup/2025.findings-acl.252)
- Call dependency is a near-zero-importance co-change feature (0.0082) next to history (0.384). — [Jia et al.](https://www.alphaxiv.org/abs/2411.19099)

### Inferences

**H1: closest work and remaining delta**

| Closest work | Shared | Delta H1 still owns |
|---|---|---|
| Comet-9B (2) | Repo-level SWE agent; rejection-sampled SFT + RL; a stated propagation chain is rewarded | The verifier is an execution-witnessed call graph scored edge by edge (programmatic, cheap, no gold patch), not an LLM judge. Graph-structured partial credit (static vs witnessed). Diagnosis feeds a fix, not test writing. |
| SWE-Doctor (1.5) | Runtime-grounded diagnosis record with a propagation path; rejects ungrounded diagnoses | Training-time use; edge-level metric |
| VulAgentRL (2) | Graph-checked evidence reward | Dynamic, not static; SWE |
| ExecVerify-style (1) | Reward checked against real execution | Repo-level call chains; agentic |
| NExT (1) | Training rationales with traces | Rationale checked against trace, not only outcome |

**H1: requirements for a graph-ML reviewer** (all INFERRED)
- **(a) Required arms.** Witnessed-edge reward vs static-edge reward vs LLM-judge reward vs outcome-only, at matched compute.
- **(b) Stage A M4 must show witnessed ≠ static** (the critic's K3). Otherwise the "graph" contribution collapses into the static-CPG line.
- **(c) Report the chain hop-P/R metric** as a structural faithfulness measure. No prior reports it.
- **(d) Cite Comet-9B prominently** and frame H1 as "a structural verifier for propagation-chain training."

**H1: novelty score**
- Downgrade from the critic's N=2 to **N=1.5**.
- Still acceptable as a workshop or short paper if (a)–(c) land.
- Risky for a main graph-ML venue, because the graph part is a reward function, not a learned graph model.

**H2: verdict** (INFERRED)
- Novelty N=0.5 as a method. A reviewer cites CodePlan and SynFix immediately.
- Keep it only as a cheap diagnostic on existing outputs (critic's CPU test). Expect the same-file / history null to win.
- If it passes, it is a finding about Gemma's failure modes, not a contribution.

### Gaps
- Concurrent ICLR-2027 OpenReview submissions are invisible. Given Comet-9B's date (28 Sep 2026), a judge-to-programmatic-verifier follow-up could appear at any time.
- The SynFix full text was not read, and neither was Comet's judge prompt. Either could tighten the verdicts.

---

## Search log

Prior searches S1–S17 are in `critic_judgement.md` and were not repeated. New queries this session:

| # | Tool | Query (+ keywords / filters) | Key hits | Bearing |
|---|---|---|---|---|
| R1 | alphaXiv discover (recency) | "training code LLMs with reward that compares model's predicted or stated execution path / call chain against the actual runtime execution trace; process reward checked by program analysis for code agents"; kw [execution trace, process reward, call chain, NExT, SemCoder, reasoning faithfulness] | **Comet-9B 2609.34359**; RLPF 2607.27271; TraceLift 2605.03862; SCOPE; CT-Repair; Self-Execution Simulation 2604.03253; SemCoder 2406.01006 | Found the closest H1 prior |
| R2 | alphaXiv discover | "predict which other functions must co-change after editing one function, using call graph and LLM, for multi-location bug fixes or change impact analysis"; kw [change impact analysis, co-change, call graph, multi-hunk, LLM] | MultiFixer; CG-Diff 2609.32057; IA + PDG 2607.23355; JS change impact 2606.21187; SiblingRepair; multi-file exploration 2606.11976 | H2 landscape |
| R3 | alphaXiv PDF | Comet-9B 2609.34359: training data, reward vs trace, SWE-bench, call chains | LLM-judge reward on the propagation chain; repo-level; SFT RS + GRPO; +7.25 pp Pro | VERIFIED, closeness 2 |
| R4 | alphaXiv PDF | TraceLift 2605.03862: what the reward checks | Rubric reward model × executor uplift; LLM-judge saturation 95.5% | VERIFIED, supports a programmatic verifier |
| R5 | alphaXiv PDF | NExT: filtering criterion; datasets | Unit-test filtering only; MBPP-R/HeFix+ | VERIFIED, closeness 1 |
| R6 | alphaXiv discover (historical) | "after an initial code edit, use dependency graph / change-may-impact analysis to propose the next locations to edit"; kw [CodePlan, CoEdPilot, edit propagation, change-may-impact, dependency graph] | CodePlan 2309.12499; CoEdPilot 2408.01733; TRACE 2604.12220; ToCS 2603.00601 | H2 killers |
| R7 | alphaXiv discover (≥2025-06-01) | "RL or rejection sampling for SWE agents where the reward checks localization reasoning or fault propagation chain against dynamic execution coverage of failing tests"; kw [fault propagation, coverage, failing test, reward, SWE-bench, root cause] | FC-SWE; SWE-MILE; Fail-Fast; SWE-TRACE; CRR; My FAULT; A²Agent; Socratic-SWE; ExecCritic | No dynamic-chain reward found |
| R8 | alphaXiv PDF ×4 (parallel) | CodePlan; TRACE 2604.12220; MultiFixer; SiblingRepair | Details in Q3 | VERIFIED |
| R9 | alphaXiv discover | "multi-method bug repair grouping faulty methods by call/data dependency (PReMM); empirical study of call-graph distance between co-changed functions"; kw [PReMM, multi-method bugs, faulty-method clustering, DEAR, co-changed functions] | CG-Diff; LtR co-change 2411.19099; D4J CFG/DFG anatomy 2502.02299 | H2 premise evidence |
| R10 | alphaXiv discover (≥2025-01-01) | "reward that compares an agent's cited stack frames or claimed call path with the actual runtime call stack / dynamic call graph, used for training"; kw [call stack, dynamic call graph, faithfulness, grounding reward, stack trace, explanation] | AgentLTL; GRADE; evidence-tracing survey 2606.04990; VICT; TRACE-credit 2607.13988; BenchShield; OTAP | None check program traces → absence holds |
| R11 | alphaXiv PDF ×2 | CG-Diff (call-connectivity of changed functions); Jia et al. 2411.19099 (features) | Median CPR 0.3; call-dependency importance 0.0082 | VERIFIED |
| R12 | WebSearch | `PReMM multi-method bug repair dependency-based faulty method clustering LLM` | PReMM (OOPSLA'25) abstract; **SynFix** (ACL Findings'25); trace-guided debate 2604.02647 | SNIPPET |
| R13 | alphaXiv PDF | "SynFix: Dependency-Aware Program Repair via RelationGraph Analysis" | **Mis-resolved** to an unrelated regex-repair paper (2510.09037); discarded | — |
| R14 | WebSearch | `SynFix RelationGraph SWE-bench Lite synchronized changes callers dependency-aware repair results` | SynFix: one-hop RelationGraph; 52.33% Lite / 55.8% Verified | SNIPPET, closeness 3 to H2 |

Not searched (budget): OpenReview ICLR-2027 submissions; full text of SynFix, Socratic-SWE and FailForge; Comet-9B judge prompt (App. G.1.8); TRACED, CodeExecutor and LDB (labelled INFERRED from memory); GitHub RL-recipe code search.
