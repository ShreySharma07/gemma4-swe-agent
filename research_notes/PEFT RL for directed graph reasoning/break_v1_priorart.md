# Breaker 3 (prior art and novelty) on architecture v1 GRAFT (8 Oct 2026)

**Role.** Breaker 3. The job is to break the novelty of GRAFT's three pieces tagged [NEW] and of its framing claim. The kill criterion is prior art that exploits the *same mechanism*. Prior art that only shares the domain does not count.

**Inputs read in full:** `architecture_v1.md` and `phase3_redteam.md`. I did not repeat the red team's queries. I reused its leads (SWE-Playground, SWM, FrogNano, CNY, 2511.16767, SWE-smith) and went deeper on them.

**Labels.**
- **VERIFIED**: I read the primary text this session, through alphaXiv full-text retrieval or the source code on GitHub.
- **SNIPPET**: abstract, search snippet, or another paper's description of the work.
- **INFERRED**: my own reasoning, or something recalled from memory and not re-read.

**Closeness scale** (same as the red team):
- 0: unrelated.
- 1: same domain or goal, different mechanism.
- 2: same mechanism family, with a material difference.
- 3: same mechanism already done.

---

## 0. Bottom line

| Piece | Closest prior (closeness) | Verdict |
|---|---|---|
| **(1) Graph→repo compiler, values flow along edges, dark edges, topology as the controlled variable** | The kernel has been done three times. **FuncBenchGen** (ICLR 2026, 2509.26553) has **2**. **DynaCode** (2503.10452) has **2**. **SWM procedural worlds** (2610.04940) has **2**: a typed graph is compiled into executable multi-service systems, with *hidden channels* (the dark-edge idea) and GRPO training. The training-transfer thesis is done in a neighbouring domain by **PhantomEnvironments** (2609.40221), closeness **2**: content-free, rule-generated graph worlds, RL, transfer to real tasks, and "hop count drives transfer". | **NARROW**. The compiler is not new. Only two things survive: *topology copied from real traced repos vs a degree- and depth-matched rewire* as the training treatment, and function-level dark-edge idioms inside an SWE-agent repair loop. |
| **(2) DPCS: degree-preserving call-target double swap as a bug operator** | Single-site, signature-compatible call-target replacement is a classic mutation operator. It appears as muJava EAM/EMM (= AMR/MMR) and as MutDafny's **MCR** (Method Call Replacement), both **VERIFIED** via 2511.15403. DeepMutants' "API misuse: wrong function call" (2107.06657, SNIPPET) is another instance. Pairing two first-order mutants is higher-order mutation (Jia & Harman, IST 2009, SNIPPET). Closeness **2**. | **NARROW**. DPCS = a second-order MCR mutant constrained to a Maslov–Sneppen double edge swap. Nothing found that uses it as a *jointly necessary* two-site SWE-agent training task, or that uses the same operator to build the null arm. The SWE-smith gap is confirmed in code (VERIFIED). |
| **(3) Rewired-context readout Δ_struct = AP(real) − AP(RW) on real code** | **CNY** (2608.29588, VERIFIED) applies a degree-preserving edge swap with texts and labels untouched, and reads the accuracy collapse as structure use. Closeness **2**. **Huang et al., TMLR 2024** (2309.16595, VERIFIED) runs adversarial ego-graph rewiring and finds LLM accuracy unchanged. Closeness **2**. The label-fixed counterfactual-context design is standard in the knowledge-conflict literature (INFERRED, 1.5). | **NARROW**. The measurement design is known. The surviving delta is the *code instantiation*: permuting call sites inside displayed real function bodies, including distractors, with labels from the true execution trace, used as a before/after-training metric. |
| **Framing: "topology-controlled curricula show whether LLM graph reasoning on code depends on real topology"** | No same-mechanism prior was found. Neighbours: PhantomEnvironments' complexity-axis ablation (VERIFIED), SynthWorlds' real-vs-synthetic mirror (SNIPPET), cross-family ER/BA/WS transfer in 2604.18092 (VERIFIED, closeness 1.5), and SWE-Playground (content, not topology). | **SURVIVES**. This is the most defensible novelty: real topology vs a null model, used as a *training-data* treatment for agent RL. |

**Overall verdict for GRAFT: NARROW, not KILL.** None of the three [NEW] tags holds as stated, since each component has a closeness-2 prior. The architecture's own wording ("the combination … no single component is claimed as a mechanism") is roughly correct, but §1 and §12 still sell the compiler and DPCS as [NEW], and that will not survive review. What survives is one experimental question, together with the instruments built to answer it.

**Single best reframing (§6):**

> *"A null-model test for synthetic agent-training environments: does real software topology carry transferable training signal beyond its degree and depth profile?"*

Under that framing:
- the compiler, DPCS and Δ_struct are presented as *instruments*, citing FuncBenchGen, DynaCode, SWM, PhantomEnvs, MCR/HOM and CNY;
- the headline is the (b) vs (c) contrast, analysed with TOST and pre-registered in both directions.

---

## 1. Piece 1: graph→repository compiler with topology as the treatment

### Closest prior work

**1a. FuncBenchGen, "Towards Reliable Benchmarking: A Contamination-Free, Controllable Evaluation Framework for Multi-step LLM Function Calling", Maekawa et al. (Megagon), ICLR 2026, arXiv 2509.26553. Closeness 2. VERIFIED.**
- Function dependencies are a hidden DAG. Each node becomes an executable function with a random name, typed inputs and one output. Values flow along edges: a function returns the correct output only when it receives the exact values produced by its parents.
- Difficulty is controlled through:
  - number of core nodes;
  - dependency depth;
  - number of *connected irrelevant nodes* (distractors that share type-compatible variables with the solution DAG);
  - number of disconnected distractors.
- Edges are linked by type and sub-type matching, not by name ("linking by exact names made the problem too easy").
- Findings:
  - success falls sharply with depth;
  - connected, type-compatible distractors are the hardest;
  - models propagate stale argument values.
- The work is evaluation plus a prompting mitigation. There is no training, no repository and no bug injection.
- **This is GRAFT C3's mechanism kernel (graph → executable functions with typed value flow → depth and distractors as controlled variables), done at the tool-call level.** GRAFT's 2:1 distractors "matched on file and degree" in C9 also echo FuncBenchGen's CIN design.

**1b. DynaCode, Hu et al., arXiv 2503.10452 (2025). Closeness 2. VERIFIED.**
- Takes 16 DAG call-graph structures (one root, at most 5 nodes) and compiles nested Python programs. Unit functions are aligned by input/output type, so "the output of a parent function becomes the input for its child function".
- Call-graph complexity is a controlled axis: M = longest path × branch count × edge count. It is crossed with code complexity.
- Findings: models handle sequential graphs well but struggle with multi-branch ones.
- The work is evaluation only. It is code *generation*, the graphs are tiny and synthetic, and there are no repositories, bugs or training.
- This is "compile a call graph into executable code with values flowing along edges, and vary topology", done.

**1c. Software World Models (SWM), Su, Hu, Zhao et al., arXiv 2610.04940 (ICLR 2027 submission LCT7iCqsXa). Closeness 2 for the compiler and the dark-edge sub-claim. VERIFIED (Appendix D read this session).**
- **Procedural systems** have 4–7 services, a shared library, datastores, event topics, configuration and access control. Synchronous calls form a **random DAG** (edge probability 0.35). They are compiled into executable worlds with repository suites and end-to-end checks.
- A **typed edge schema** has 12 relations (calls_api, consumes, reads, guarded_by, …).
- **Hidden channels** are edges dropped from the agent's observed graph at per-relation rates "reflecting real extraction failure modes" (reads 0.60, calls_api 0.35, consumes 0.30, …). This is GRAFT's dark-edge idea ("not lexically visible; share matched to real") at service granularity.
- SFT warm-up and GRPO on executed consequences, with train and eval worlds disjoint and held-out real libraries.
- The red team rated this 1.5. **I raise it to 2 for the dark-edge sub-claim.** SWM already compiles typed graphs with deliberately extractor-invisible edge types into executable worlds for LM training. It differs in three ways:
  - service granularity, not function granularity;
  - random topology, not copied from real systems;
  - consequence prediction, not repair.

**1d. PhantomEnvironments, "Training LLM Agents in Fictional Worlds", Kabra et al. (Cornell/Stanford), arXiv 2609.40221. ICLR 2027 submission lSbXxI368i; workshop version "On the Promise and Limits of Training LLM Search Agents in Fictional Worlds", LLA 2026 (SNIPPET). Closeness 2 for the framing and the transfer thesis. VERIFIED.**
- Rule-generated worlds:
  - a random social graph;
  - templated articles;
  - Prolog-verified multi-hop questions;
  - no LLM and no real facts.
- GRPO trains search agents that **transfer to real multi-hop benchmarks**.
- The **ablation of environment complexity axes** shows that *"linear hops carry most of the transfer"*.
- Real-data training wins in-domain. Synthetic training wins on newer out-of-domain benchmarks and closes the SynthWorlds knowledge-advantage gap.
- **This is GRAFT's H1 + H2 thesis ("content-free graph worlds transfer; depth/hops is the active ingredient") demonstrated for search agents.** GRAFT's H1/H2 are no longer surprising in kind, only in domain (code, SWE).
- The paper also cites three related works:
  - **SynthWorlds** (Gu et al. 2025): the same graph re-instantiated with synthetic entities, i.e. topology held fixed while content varies. This is the mirror image of GRAFT's design;
  - Kabra et al. 2026;
  - **RandomWorld** (Sullivan et al. 2025): type-guided procedural tool-use sequences.
  These three are SNIPPET.

**1e. Codoku, Li, Sun, Li, Su (ETH), arXiv 2609.34661. Closeness 1. VERIFIED.**
- Programs are synthesised from scratch from a prescribed CFG and execution path, by semantic reification (Chopra et al., PLDI 2026) with an SMT solver.
- Generation profiles control complexity.
- It is an evaluation puzzle, intra-procedural, with no call graph and no training. It establishes that "graph spec → executable program, renewable, contamination-free" is a live PL/ML technique.

**1f. Other neighbours.**
- **SWE-Playground** 2512.12216 (red team VERIFIED, 1.5).
- **FrogNano** 2609.07925 (1).
- **SPADE** 2608.19197 (VERIFIED, 1): LLM-written Gym environments, no graph compiler.
- **BugPilot** 2510.19898 (VERIFIED, 1): agentic feature-add bugs in real repos.
- **TraceEval** 2605.11006 (VERIFIED, 1.5 for C9). This one bears on the readout rather than the compiler. It is an execution-verified call-graph benchmark: 10.6k real programs, a sys.settrace oracle, and LoRA fine-tuning of Qwen2.5-Coder lifts F1 by up to +55.6. GRAFT's C9 item ("trace-labelled call set from a test") is TraceEval's task in a pooled-candidate format, and TraceEval already shows that SFT on traced call graphs moves this metric. GRAFT should cite it and contrast its own synthetic→real transfer against TraceEval's real→real fine-tuning.
- Systems analogues (SNIPPET):
  - DGG, "A Microservice Graph Generator with Production Characteristics" (CoRR 2024) generates dependency graphs that match production topology for benchmarks;
  - Goens et al. MULTIRPOG generates code from random graphs for compiler/concurrency benchmarks.

### What remains (the delta)

1. **Topology copied from real traced repositories vs a DAG-preserving, depth-matched, per-type degree-preserving rewire vs depth-1**, as the *training* treatment, with everything else equalised. No prior found. SWM, FuncBenchGen and DynaCode all use random or hand-specified topologies and never ask whether real topology matters. PhantomEnvs varies *question type*, not graph topology.
2. Function-level dark-edge idioms (dispatch, decorator, registry, callback) compiled into a *Python repository* with pytest root tests and unit contracts, used for an **SWE repair agent**. SWM's hidden channels are the same idea at the service level.
3. Copying identifiers from S_topo while bodies are templates. This is a minor design choice.

### Verdict: NARROW

- Drop "[NEW] Topology-as-treatment compiler". The compiler is engineering on a known pattern; cite FuncBenchGen, DynaCode and SWM.
- Keep "[NEW] topology-fidelity treatment": real vs degree/depth-matched null.
- State H1/H2 as *replications in the code/SWE domain* of PhantomEnvs' "content-free graph worlds transfer; hops are the active ingredient". That makes them priors, not discoveries.

---

## 2. Piece 2: DPCS (degree-preserving call-target swap)

### Closest prior work

**2a. Signature-compatible method-call replacement in mutation testing. Closeness 2. VERIFIED via MutDafny (Amaral, Mendes, Campos, arXiv 2511.15403, §3.3.13 and §4.2.2).**
- muJava's **AMR/MMR** (muJava's EAM, *accessor method change*, and EMM, *modifier method change*; Ma, Offutt, Kwon 2005/2006) "replace one method with another having the same signature".
- MutDafny's **MCR (Method Call Replacement)** replaces one call with another of matching return type and argument types, keeping the arguments (e.g. `Sum(10,20)` → `Multiply(10,20)`). It was mined from real Dafny bug-fix commits, so it is an empirically grounded *real* bug class.
- Also known (SNIPPET):
  - DeepMutants (2107.06657), "API misuse … a programmer chooses a wrong function during a function call", seeded from a call vocabulary for neural bug detectors;
  - the Defects4J study noting MuJava/PIT/Major lack a wrong-method-call mutator (orbilu TSE-MIN).
- **So the single-site, type-compatible call-target replacement that DPCS is built from is a long-established mutation operator.** The "type-compatibility removes the TypeError giveaway" argument is exactly MCR's validity constraint.

**2b. Higher-order mutation. Closeness 1.5–2. SNIPPET/INFERRED.**
- Jia & Harman, "Higher Order Mutation Testing" (IST 2009, OpenReview DBLP hit); "Constructing subtle faults using higher order mutation testing" (SCAM 2008, INFERRED).
- A second-order mutant composed of two MCRs whose fix needs both sites is a coupled HOM.
- DPCS's specific constraint, that the two replacements are each other's callees (a→d, c→b), is a Maslov–Sneppen double edge swap applied as a HOM. That conjunction was not found.

**2c. SWE-agent bug synthesis. Closeness 1. VERIFIED (code).**
- The SWE-smith Python procedural operators, read from `swesmith/bug_gen/procedural/python/__init__.py`, are:
  - ClassRemoveBases, ClassRemoveFuncs, ClassShuffleMethods;
  - ControlIfElseInvert, ControlShuffleLines;
  - RemoveAssign, RemoveConditional, RemoveLoop, RemoveWrapper;
  - OperationBreakChains, OperationChangeConstants, OperationChange, OperationSwapOperands.
  **None changes a call target.** The java, golang, js, rust and cpp folders have no call-target swap either (file names checked).
- SWE-smith's **combine** step (`bug_gen/combine/same_module.py`) merges patches into multi-file tasks, but **never runs tests and never checks joint necessity**. GRAFT's C5 strict-subset necessity check is therefore a real difference.
- **BugPilot** (VERIFIED) produces multi-file bugs naturally (4.2 files on average). It has no structural operator and no necessity guarantee.

### What remains (the delta)

1. Constraining two type-compatible MCRs to form a **degree-preserving double swap**, so the bug is invisible to any degree or fan-in/fan-out statistic.
2. Using it as a **jointly necessary two-site SWE-agent training task** with an execution-verified strict-subset check.
3. Using **the same operator** to generate both bugs and the topology-null arm (c).

No prior found for (1)–(3) in combination. Each is a small step.

### Side note (validity, INFERRED; for the critic, not a novelty point)

"Jointly necessary" is checked only against *strict subsets of the gold fix* (C5 iii), not against **alternative one-site fixes**. If b and d have no other callers, swapping the two *definitions'* names or bodies repairs the program with one edit. A related route is re-binding the callee at one site plus re-binding at the callee definition. C5 should also require that no single-site edit passes, or at least report how often an agent's accepted patch touches only one site.

### Verdict: NARROW

- Rename it honestly as "a degree-preserving second-order MCR mutant".
- Cite muJava AMR/MMR, MutDafny MCR, DeepMutants and Jia & Harman HOM.
- Claim only the degree-preservation constraint, the joint-necessity guarantee, and the operator's dual use as bug generator and null model.

---

## 3. Piece 3: rewired-context readout Δ_struct on real code

### Closest prior work

**3a. CNY, "Call Neighbours Yourself", Liu et al. (UQ), arXiv 2608.29588. Closeness 2. VERIFIED (§5.3, App. A.2).**
- "If the model were doing only semantic retrieval … randomising the graph while keeping every node's degree fixed should leave accuracy unchanged. On WikiCS we apply random edge swaps, preserving degrees but destroying topology, with texts and labels untouched."
- Accuracy falls from 76.6 to 73.0, matching the no-walk baseline. This drop is read as evidence that the model uses the real structure.
- **This is Δ_struct's exact logic:** a degree-preserving rewire, content and labels fixed, accuracy drop = structure use. It is applied to TAG neighbour-walking rather than to code bodies. The architecture already cites CNY; the issue is that it is closer than "neighbour".

**3b. Huang, Zhang, Mei, Ma, "Can LLMs Effectively Leverage Graph Structural Information through Prompts…?", TMLR 2024, arXiv 2309.16595. Closeness 2. VERIFIED.**
- Two "adversarial" probes:
  - linearise the ego-graph;
  - **randomly rewire** it (Random / Extreme / Path), comparing the LLM's drop against MPNNs.
- Results: LLMs drop −0.25% on average, MPNNs −5.55%. The authors conclude that LLMs read prompts "as contextual paragraphs rather than graphs".
- This is the base-model Δ_struct measurement, published, with a GNN as the structure-using reference. Their rewire is not degree-preserving.

**3c. "When Structure Doesn't Help", Xu, You, Ma, LoG 2025, arXiv 2511.16767. Closeness 1.5. VERIFIED.** Structure-free templates (HN, CO) match structure-aware ones. LLMs act as "unordered set readers". This is the GF-control analogue.

**3d. Code-specific counterfactual probes. Closeness 1.5.**
- **"When Names Disappear"**, Le et al., arXiv 2510.03178 (VERIFIED) is the *inverse* manipulation. It keeps structure, destroys names, and measures the drop on execution prediction (e.g. LiveCodeBench Pass@1 85.4 → 71.2). GRAFT's RW keeps names and destroys structure. Together they form a 2×2 that GRAFT could cite as the missing cell.
- **Spiess, Devanbu, Barr**, arXiv 2604.16320 (VERIFIED): meaning-preserving transforms and input perturbation on CruxEval.
- **CACP**, Hooda et al. 2024: counterfactual code mutations such as flipping branches (SNIPPET, via 2604.16320).

**3e. Counterfactual-context / knowledge-conflict methodology. Closeness 1.5. INFERRED.**
- Longpre et al. 2021 entity-substitution conflicts; context-aware decoding (CAD).
- These measure reliance on context by making the context lie while the evaluation target stays fixed.
- Δ_struct with "labels stay true" is this design. A model that reads call sites is *penalised* under RW. So Δ_struct measures context-faithfulness to displayed call sites, which may be a lexical copying skill rather than multi-hop traversal.
- This is not a novelty kill but a construct-validity risk the paper must address (INFERRED). One way: stratify Δ_struct by depth ≥2, where a single copied call site cannot recover the label.

**3f. The underlying task.** TraceEval 2605.11006 (VERIFIED, see 1f) and CG-Bench (LMPL 2025, SNIPPET) are call-graph prediction from source with LoRA fine-tuning. They are not rewired, but they are the item format's prior.

### What remains (the delta)

- Applying a degree-preserving rewire of the *static call graph* to call sites **inside displayed real function bodies, including distractors**, with labels from the *true execution trace*.
- Reporting it as a **before/after-training change** (Δ_struct(b) − Δ_struct(base)), stratified by depth and edge type.

No code-domain instance was found. The measurement concept is CNY + Huang et al.

### Verdict: NARROW

Re-tag it [NARROWED]: "CNY's degree-preserving-rewire control, ported to code bodies and used as a training-effect metric". Add depth stratification to separate copying from traversal.

---

## 4. The framing claim

**Claim:** "Topology-controlled curricula show whether LLM graph reasoning on code depends on real topology."

**Searched for:**
- real-vs-null-model topology as a *training-data* treatment, in LLMs, agents or GFMs;
- degree-preserving rewiring for LLM code or agent work;
- synthetic repositories with controlled topology transferred to real repositories.

**Found:**

| Prior | Closeness | Label | What it does |
|---|---|---|---|
| PhantomEnvs | 2 | VERIFIED | Varies *question complexity axes* (hops, comparisons, constraints), not topology fidelity. It does show "simple content-free graph worlds transfer". |
| 2604.18092 | 1.5 | VERIFIED | LoRA SLMs trained on two random-graph families (ER/BA/WS), tested on the held-out third, and fine-tuned on real TU/OGB graphs. Graph family is a variable, but it is *random-model vs random-model* and the task is property estimation, not reasoning transfer to real code. |
| SynthWorlds | 1.5 | SNIPPET | Holds the real graph fixed and swaps content. This is the orthogonal manipulation. |
| SWE-Playground | 1.5 | VERIFIED by red team | Synthetic content vs real; topology uncontrolled. |
| Bechler-Speicher et al., "GNNs use graphs when they shouldn't" | 1 | SNIPPET | Cited in 2511.16767. |
| Graph-foundation-model transfer theory (2605.29828) | 1 | SNIPPET | — |
| Web and alphaXiv searches for "degree-preserving" + call graph + LLM | — | — | Returned nothing in combination. |

### Verdict: SURVIVES

This is GRAFT's most defensible novelty. It survives *only as the (b) vs (c) contrast*. The (b) vs (d) "depth matters" contrast is now prior-expected, following PhantomEnvs, FuncBenchGen and DynaCode.

---

## 5. Overall novelty verdict for GRAFT

**NARROW.** By the paper-track rubric (novelty including analysis):

1. **Compiler (C3).** Do not tag it [NEW]. The kernel is FuncBenchGen/DynaCode (eval) and SWM (training, typed hidden channels). Novelty points come only from scaling it to SWE repos and from function-level dark idioms.
2. **DPCS (C4/O2).** Tag it [NARROWED], citing MCR/AMR/MMR and HOM. It is a minor contribution, with a validity fix needed (single-site alternative fixes).
3. **Δ_struct (C9).** Tag it [NARROWED], citing CNY and Huang et al. TMLR'24. Its value is the code instantiation plus the training-effect framing.
4. **H1/H2.** Present them as domain transfers of PhantomEnvs' findings (and SWE-Playground's scale caveat), not as novel predictions.
5. **H3 ((b) ≈ (c) under TOST).** This is the only genuinely open question. It is also where GRAFT's predicted outcome is a *null*. A null headline is acceptable only if it is (i) pre-registered both ways (already done) and (ii) powered: the B11 seed-noise replicate becomes mandatory, not descopable (D2 drops it).

## 6. The single reframing that makes GRAFT most defensibly novel

> **"Does real software topology matter for training coding agents? A degree-preserving null-model test for synthetic training environments."**

The headline contribution is a **null-model methodology for synthetic agent-training data**. Any procedurally generated training world (code, search, tool-use) should be compared against a degree- and depth-matched rewire of its source structure, so as to attribute transfer to *real structure* vs *structure statistics* vs *content*. Code is the first, best-instrumented case, because execution gives exact labels and the same DPCS operator yields both the null arm and two-site bugs.

Everything else is reported as instruments, with citations:
- the compiler (FuncBenchGen, DynaCode, SWM, PhantomEnvs);
- DPCS (MCR/HOM);
- Δ_struct (CNY, Huang et al. TMLR'24);
- the call-set items (TraceEval, CoverageEval).

**Why this framing is best:**
- It turns the most likely result (H3 null) into the finding: "degree + depth suffice, so cheap random generators matched to these statistics are enough".
- The alternative result ((b) > (c)) is also a finding: "real topology carries signal".
- It generalises beyond code, which strengthens the rubric's generality axis.
- No prior work occupies that slot.

**Required to make the reframing hold:**
- keep B11 (the seed replicate);
- report the motif, clustering and assortativity differences between (b) and (c) (A8);
- add depth-stratified Δ_struct so it measures traversal rather than call-site copying.

---

## Search log

**Tools:**
- alphaXiv `discover_papers` (DP) and `answer_pdf_queries` (READ);
- WebSearch (WS); WebFetch (WF);
- OpenReview API `api2.openreview.net/notes/search` (OR), paced at about 7 s; per-note `notes?id=` returned **HTTP 403** three times, so abstracts were taken from search results instead;
- GitHub API/raw (GH).

| # | Tool | Query / target | Key result |
|---|---|---|---|
| 1 | DP | procedurally generated synthetic programs/repos from call/dependency graphs for LLM code reasoning, topology controlled | **Codoku 2609.34661**, **PhantomEnvironments 2609.40221**, GraphInstruct, InfoSynth (not close) |
| 2 | DP | do LLMs use graph structure: shuffling/rewiring with degree preserved, structural probes | Lost in Serialization 2511.10234, graph-isomorphism failure 2606.09484, structural-distortion 2606.15633 (1) |
| 3 | DP | synthetic programs with controlled call depth for multi-hop call-chain reasoning | **FuncBenchGen 2509.26553**, RepoReasoner, SWE-QA, Codoku |
| 4 | DP | synthetic repos for SWE-agent RL, procedurally generated with injected bugs (recency) | DeNovoSWE, **SPADE**, CodeMidas, SSR, **BugPilot**, PITMuS, SWE-Synth, WitnessGym, FrogNano, FC-SWE |
| 5 | READ | 2609.34661 Codoku | CFG+EP→program via semantic reification + SMT; eval only |
| 6 | READ | 2509.26553 FuncBenchGen | DAG→typed functions; depth / connected-distractor control; eval only |
| OR1 | OR | "synthetic program generation call graph LLM" | Only call-graph-tool papers |
| OR2 | OR | "random program generator training language models" | Remgen (compiler testing) |
| OR3 | OR | "procedurally generated codebases" | InfiniteScienceGym, PGT (not close) |
| OR4 | OR | "call-target mutation" | **"A large-scale study of call graph-based impact prediction using mutation testing"** (Musco et al., SQJ 2017; SNIPPET; ERC-adjacent), **Higher Order Mutation Testing** (IST 2009) |
| OR5 | OR | "method call replacement mutation operator" | Only GA-mutation noise |
| OR6 | OR | "degree-preserving rewiring language model" | Network-science paper only |
| OR7 | OR | "graph shuffling LLM structure use" | Nothing |
| 7 | WS | muJava EAM/EMM, method call replacement with a compatible signature | muJava operator lists; MutDafny |
| 8 | WS | "wrong function call" / "callee swap" bug injection for neural bug detection | DeepBugs (arg swap), **DeepMutants API-misuse wrong-function** |
| 9 | WF/GH | muJava mutopsClass.pdf | WF returned the GMU homepage; curl returned HTML. Not read |
| 10 | READ | MutDafny 2511.15403 (resolved from the muJava title query) | **AMR/MMR same-signature replacement; MCR** |
| 11 | READ | SWE-smith 2504.21798 | SSE parse error (tool) |
| 12 | READ | BugPilot 2510.19898 | FeatAdd multi-file bugs; no structural operator |
| 13 | WF | swesmith.com guide | Operator list not on page |
| 14 | GH | SWE-smith `bug_gen/procedural/{python,java,golang,js,rust,cpp}` | **No call-target operator** (VERIFIED) |
| 15 | WF | SWE-smith `bug_gen/combine/same_module.py` | Combines patches; **no tests, no joint necessity** (VERIFIED) |
| 16 | DP | perturbing call sites / names to test reliance on structure vs names in code LLMs | **When Names Disappear 2510.03178**, Lexical Superstition 2609.26388, **2604.16320**, Jagged Frontier 2608.18389, **Paired interventions 2607.26244** |
| 17 | DP | training on synthetic graphs with random vs real topology; null models; transfer | **2604.18092**, shortest-path generalisation 2604.15306, GraphEcho (not close) |
| 18 | READ | 2608.29588 CNY | **Degree-preserving rewire control, texts/labels fixed** |
| 19 | READ | 2511.16767 | Structure-free templates match structure-aware ones |
| 20 | READ | 2309.16595 Huang et al. TMLR'24 | **Ego-graph rewiring adversarial test**; LLM drop ≈0 vs MPNN −5.6 |
| 21 | READ | 2607.26244 | Paired semantic interventions on tests (1) |
| 22 | DP | LLM call-graph construction incl. indirect calls | Indirect-call analysis 2408.04344, **TraceEval 2605.11006**, AgentFlow |
| 23 | DP | random/perturbed code-graph ablation for code agents | GRADE, **Code Isn't Memory 2606.22417**, Jagged Frontier |
| 24 | READ | 2605.11006 TraceEval | Execution-verified call-graph benchmark; LoRA +55.6 F1 |
| 25 | READ | 2606.22417 | Structural index on/off ablation; no shuffled-index control |
| 26 | DP | synthetic nested function calls of controlled depth; training transfer | Neural Debugger, ExecVerify, PhantomEnvs (not new) |
| 27 | DP | content-free procedural data and transfer | **Procedural Pretraining 2601.21725**, Logic Before Language, Structural priors (1, framing only) |
| 28 | READ | 2609.40221 PhantomEnvironments | **Rule-generated graph worlds; RL; real transfer; hops drive transfer; SynthWorlds and RandomWorld cited** |
| 29 | READ | 2608.19197 SPADE | LLM-written Gym environments; no graph compiler |
| 30 | READ | 2604.18092 | Cross-family ER/BA/WS transfer; LoRA SLMs |
| OR8 | OR | "synthetic codebase generator dependency graph" | **DGG microservice graph generator with production characteristics** (CoRR 2024; SNIPPET) |
| OR9 | OR | "rewired call graph" | GNN rewiring, AutoPruner (not close) |
| OR10 | OR | "topology controlled synthetic environments agent" | EARIS (ICLR 2027, DvQ8Kf0noB; executable synthetic environments for lookahead; 1), Agent World Model, Synthetic Sandbox |
| OR11 | OR | "fictional worlds agent training transfer" | **PhantomEnvs ICLR 2027 submission lSbXxI368i**; LLA 2026 workshop version |
| OR12 | OR | "configuration model null LLM" | Nothing |
| OR13 | OR | "second order mutant method call" | SOM clustering (COMPSAC 2021; not close) |
| OR14 | OR | "counterfactual context utilization code" | Noise |
| OR15 | OR | per-note fetch for DvQ8Kf0noB, LHhwWN1yK6, EvnDKRVZNV | **HTTP 403** ×3; used search-result abstracts |
| 31 | WS | generating an executable program from a call-graph spec; benchmarking call-graph tools | **DynaCode 2503.10452**, Goens MULTIRPOG, PyCG micro-suite, DyPyBench; "no tool found that emits a program from a call-graph spec" |
| 32 | WS | LLM training on synthetic repos from dependency-graph topology, transfer to real | UCD-Training 2602.20799 (real graphs; 1), CGM; no topology-controlled synthetic repos |
| 33 | READ | 2503.10452 DynaCode | **16 DAG call graphs → nested code, parent output feeds child input; eval only** |
| 34 | WS | Valverde & Solé motifs in software graphs, degree-preserving null | Software-motif literature uses degree-preserving nulls (SNIPPET); the PRE 2005/2007 paper could not be verified |
| 35 | READ | 2604.16320 Spiess et al. | MPT/renaming/input perturbation; cites CACP counterfactual mutations |
| 36 | READ | 2510.03178 | Name-obfuscation suite; execution-prediction drops |
| 37 | READ | 2610.04940 SWM, Appendix D | **Random-DAG typed systems; 12 relations; hidden channels at per-relation drop rates** |
| 38 | DP | real vs null-model topology as fine-tuning treatment, transfer | GFM transfer theory 2605.29828, GraphUniverse; nothing same-mechanism |
| 39 | WS | "degree-preserving" rewiring + call graph + LLM/code agent | Nothing in combination. CG-Bench (LMPL 2025) and the Venkatesh et al. LLM call-graph study 2410.00603 surfaced (SNIPPET) |

**Not done:**
- Semantic Scholar: alphaXiv and OpenReview covered the space.
- Firecrawl PDF scrapes: not needed, since every key hit was on arXiv.
- Primary reads of:
  - Jia & Harman HOM;
  - muJava's class-operator PDF (WF and curl both failed);
  - DeepMutants;
  - SynthWorlds;
  - Longpre et al. 2021;
  - CACP.
  These are labelled SNIPPET or INFERRED above.
