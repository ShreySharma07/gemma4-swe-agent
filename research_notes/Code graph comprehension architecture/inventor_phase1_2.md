# Inventor, Phases 1–2: twelve code-graph techniques, six refined (9 Oct 2026)

**Brief.** The topic is novel code-graph generation, parsing and embedding. Every technique must:
- be verified with execution as ground truth;
- be cheap to train (LoRA on Gemma-4-31B W4A16, 4×L4, offline, ≤16k tokens per sequence, about 25 GPU-h for the whole study; CPU is free);
- generalise beyond this competition.

No critic has seen this yet. Phase 3 should attack it.

**Labels.**
- **VERIFIED**: read in a primary source or computed from repo files. Here that means read either by the scouts (`scout_graph_generation.md` = **G-An**, `scout_embedding.md` = **E-An**) or in `docs/HQ.md` (**HQ**).
- **SNIPPET**: from a search-result summary only (my log is **IV-n** at the end).
- **INFERRED**: my own reasoning.
- **GUESSED**: a sizing number that a kill test must measure.

Every effect size below is GUESSED unless marked otherwise. **Nothing here has been run.**

**Traps from the last three rounds, and how this note avoids them.** Each technique's kill test was checked against this list.

| Trap (source: *PEFT RL for directed graph reasoning*, "What died") | Rule applied here |
|---|---|
| Set rewards that can be gamed by padding (K-30/31) | Score only ranks over **fixed candidate pools**: MRR, hit@k, AUROC. No set-F1 reward anywhere |
| Items solvable without the graph (contract or template shortcuts, K-04/12) | Every kill test has a **graph-free arm**, usually lexical mention or BM25. Every downstream test also runs under **consistent alpha-renaming** |
| Controls that change more than one thing (K-21 locality) | Rewires preserve **in-degree and out-degree** and, where the claim is not about locality, **module**. The label is stated per test |
| One seed; no positive control | ≥3 seeds or data resamples, with seeds as the unit. Each kill test names a positive control |
| Under-counted budgets (L5 ×2) | Budgets count epochs and forward tokens explicitly. Throughput is GUESSED at 400–650 tok/s for training and about 1.5k tok/s for HF forward passes until the probe measures it |
| "Methods" that are really SFT on items | Phase 2 says for each technique what the **artifact** is and how it is used **outside** the training items: unseen repos, another language, another model |
| Mechanisms already published (CRPO, cue decorrelation, FuncBenchGen/SWM, CNY rewire) | Nothing below uses twin or null training as its mechanism. Rewires appear only as **controls** |
| Directions already taken (CERT, IFG, null dial, Z, twin-null RFT, GRAFT) | Not proposed as headlines. KILLVEC (T10) is flagged as IFG-adjacent |
| Function hit does not predict resolve (27% vs 28%; *LLM graph reasoning for code*, L1, VERIFIED) | No technique claims a resolve gain as its headline. The headlines are comprehension artifacts, scored on execution-labelled graph or retrieval metrics. Resolve rate is a secondary, unpowered readout |

**Train/test hygiene.** Training uses only repos outside the dev set (fastapi, rich, requests and httpx are excluded): pure-Python packages with good test suites, for example from the SWE-smith or SWE-Gym repo lists. The 118 gold-mapped dev tasks are **evaluation only**.

**Shared CPU infrastructure** (built once, used by T1, T2, T3, T5, T6 and T7):
- **Tracer.** `sys.monitoring` (3.12+) or `sys.setprofile` records (caller code object, call-site line, callee code object, count), plus escaping exceptions and returns, while running each repo's test suite.
- **Static resolver.** A tree-sitter/AST resolver for the static baseline, plus PyCG where it terminates.

This is the same C1 tracer already specified for IFG and the v3 dial (VERIFIED, report §"final v3").

---

## Phase 1(a): six anomalies, three mechanisms each, and a technique that works only if that mechanism is real

I picked anomalies that (i) are verified, (ii) touch generation, parsing *or* embedding, and (iii) appear in our own data as well as the literature.

### AN1. Static call graphs recall about half of runtime edges, and 0.99 edge precision becomes 0.19 chain precision
Sources: G-A1, G-A3, G-A18, VERIFIED. Missing callees are 59% built-in-type methods, about 12% name mismatches, plus `super()`.

| # | Mechanism | Technique that works only if it is real |
|---|---|---|
| 1a | Most misses are **resolution failures the live heap has already resolved**. After `import pkg`, the runtime objects carry the MRO, `__wrapped__`, `__module__`/`__qualname__` (canonical names for re-exports), registry dicts of callables, and bound built-in types | **T1 HEAP**: build the graph by walking the imported heap, with no tests and no LLM |
| 1b | Chain collapse comes from **correlated, site-level ambiguity**, not independent edge errors. One ambiguous call site fans out to k candidates, and each one drags a whole false subtree along | **T6 CHAIN**: give each call site a probability distribution over callees that sums to at most 1, learned from traces, and rank paths by probability mass |
| 1c | The misses are **type-inference failures at call sites**. Types are where LLMs beat static tools (G-A7: return types on decorated functions, 86% vs 38%) | **T8 SITE**: an LLM call-site typer with static closure over the inferred types, trained on traced callees |

### AN2. LLMs over-extend along the static transitive graph, and reachability is more decodable than adjacency but not used
Sources: G-A4 (260 predicted vs 44 actual def–use pairs; Gemma-4-31B ≈ 4B) and E-A17 (direct AUROC 0.869 vs transitive 0.986; 0% behavioural effect), VERIFIED; E-A19, VERIFIED (qualitative).

| # | Mechanism | Technique that works only if it is real |
|---|---|---|
| 2a | The residual stream encodes **lexical reachability**: the transitive closure of name co-mention. Over-extension is that representation read out faithfully | **T2 LATENT** used as a *diagnostic*. A probe trained on traced labels should be no better than the lexical closure on runtime-only edges. If 2a holds, T2 fails as a builder, and that failure is itself the finding |
| 2b | The residual stream encodes **runtime-relevant** reachability (dispatch, decorators, `super`), but the verbal readout fails, as in E-A17/A19 | **T2 LATENT** used as a *builder*. A bilinear probe trained on execution labels turns one forward pass into a directed runtime graph that beats the model's own verbal answers |
| 2c | The model over-extends because it never sees **negative or graded evidence**: graphs show edges as present and never as rarely taken | **T6 CHAIN** weights, or **T12 EXECCARD** observed behaviour, shown in context. Over-extension (predicted reach-set size divided by true size) should shrink |

### AN3. Our supplied embeddings are low-pass, and the top-1 embedding neighbour is a graph neighbour only 3–9% of the time
Source: HQ, VERIFIED. Node ≈ mean of its neighbours (cosine 0.85–0.97). Centered cosine is about 0.85–0.92 at 1 hop and about 0 at 5+ hops. Spearman with TF-IDF is 0.22–0.32. The field analogue is over-smoothing (E-A16: MLP beats GNN adapter, 76.3 vs 67.5).

| # | Mechanism | Technique that works only if it is real |
|---|---|---|
| 3a | The vectors are a **linear low-pass filter of per-node text features**, X ≈ g(Â)·H, so the discriminative node content sits in the attenuated high frequencies | **T4 DECONV**: identify g, invert it with regularisation to recover Ĥ, and use Ĥ for retrieval and as a target for a linear query encoder. It works with the *true* Â and not with a degree-preserving rewired Â |
| 3b | Top-1 neighbours are **2-hop structural siblings**: co-callees of a hub, or methods of one class. Smoothing makes nodes with shared neighbours closer than adjacent ones | **Sibling retrieval for co-edits.** Multi-function fixes (65/129 tasks have ≥2 gold functions; ≥3 → 4% resolved, VERIFIED in *LLM graph reasoning for code*) may be siblings, so kNN in the smoothed space should retrieve the *other* gold functions. This is folded into T4's kill test as a competing explanation |
| 3c | Top-1 neighbours are **textual near-duplicates** (fastapi docs_src clones, 0.94–0.99) | **Clone-collapse before any graph operation.** This is a hygiene step, not a technique. It is a control in T4 (exclude clone pairs) |

### AN4. Issue-to-edit retrieval collapses (NDCG 70 → 20), the fix is on the query side, and only 21% of issues name the gold node
Sources: E-A1, E-A2 (frozen code vectors with a query-side LoRA, MRR 0.29 → 0.51–0.57), VERIFIED; HQ 21.2%, VERIFIED.

| # | Mechanism | Technique that works only if it is real |
|---|---|---|
| 4a | Issues describe **behaviour** ("raises KeyError", "mutates headers", "returns None"), while code embeddings encode names and content. The bridge is a shared *effect* vocabulary | **T5 EFFECT**: an execution-labelled effect parser over nodes, with effects propagated along directed edges, matched against effects extracted from the issue |
| 4b | Issues describe a **scenario**: a public entry point called with some input. The gold node lies *downstream on the runtime path* of that scenario, and the path's shape survives even when names are wrong | **T3 HYTRACE**: a LoRA maps issue plus repo skeleton to an *imagined runtime call path*, which is then decoded onto the real graph by structural (Viterbi) alignment |
| 4c | Issues quote **runtime observables** (messages, reprs, warnings) built dynamically, for example by f-strings or formatting helpers, so grep misses the producer | **OBSERVABLE**: index traced producers of strings and exceptions. It was dropped in distillation (grepping literals already captures most of it; INFERRED), and is kept as a feature inside T5 |

### AN5. A less accurate LLM-built partial graph beats an accurate static global graph, and edge value shrinks with model strength
Sources: G-A7 vs G-A8 (DyCoder, 7.4× faster), G-A10/A17 (−5.7 vs −1.0 pp), G-A21 (graphs buy efficiency), VERIFIED.

| # | Mechanism | Technique that works only if it is real |
|---|---|---|
| 5a | Value comes from **query-conditioned selection**, not construction fidelity | **T3 HYTRACE**: the graph region is chosen *by* the query (an imagined path) |
| 5b | Edges buy **token efficiency**, not reach. Value ∝ reads avoided | **T9 TRACEBACK / T12 EXECCARD**: compact formats. Predicted gain appears on the budget curve (accuracy at fixed tokens), not at unlimited budget |
| 5c | Stronger models **already hold the graph latently**, so explicit edges are redundant | **T11 RESIDUAL**: show only edges the model's latent (T2 probe) does *not* already encode. Predictions: the residual graph equals the full graph at a fraction of the tokens, and gain from explicit edges anti-correlates with probe AUROC across model sizes |

### AN6. The same graph in another format changes results, graph-only beats graph+source, and containment edges are worth about nothing
Sources: G-A12 (visual 78 vs text 71), G-A13 (AST+PDG 83.2% vs source 53.5%; PDG+source < PDG), G-A14 (containment-only 66.42 ≈ no graph 66.06; attributes hurt), VERIFIED.

| # | Mechanism | Technique that works only if it is real |
|---|---|---|
| 6a | The format effect is **distance from the pretraining distribution of structure text**. Tracebacks and stack dumps are common in pretraining; adjacency lists are rare | **T9 TRACEBACK**: serialise directed paths as synthetic Python tracebacks |
| 6b | Source **dilutes attention by token count**, and a behavioural abstract carries the decision-relevant bits | **T12 EXECCARD**: node text = traced I/O exemplars and effects instead of source |
| 6c | Containment is **redundant with qualified names**. An edge helps only when it is *not lexically inferable* | **T11 RESIDUAL**: prune every edge predictable from names or mentions and keep the non-lexical ones. Prediction: the residual graph ≥ the full graph, and a containment-plus-lexical graph ≈ no graph |

---

## Phase 1(b): eight unstated assumptions of code-graph papers, inverted

| # | Assumption | Inversion | Technique |
|---|---|---|---|
| I1 | The graph is built **before** the query | The query *generates* the graph region: an imagined execution path decoded onto structure | T3 HYTRACE; T8 SITE (lazy, per site) |
| I2 | Edges are **binary** | A call site is a **distribution** over callees, with mass ≤1, from runtime frequency. Paths have probability | T6 CHAIN |
| I3 | The graph is built by **static analysis or the LLM** | Built by **the runtime heap** at import time, or **read out of the LLM's residual stream** with an execution-trained probe | T1 HEAP; T2 LATENT |
| I4 | Nodes are **functions** | Nodes are **call sites** (resolution happens there) or **effects** (what issues describe) | T8 SITE; T5 EFFECT |
| I5 | Embeddings encode **content** | Embeddings encode **role**: effects, direction (separate source and target vectors), or the high-pass residual that sets a node apart from its neighbourhood | T5; T7 ASYM; T4 DECONV |
| I6 | The graph is **shown** to the model | The graph is **already in the model**. Read it out (probe), or show only the part it lacks (residual) | T2; T11 RESIDUAL |
| I7 | **Edge precision** is what matters | **Path mass** and **presence of the correct node** matter (G-A3, G-A16: rank order had 0.0 pp effect) | T6; T1 (recall) |
| I8 | **One graph per repo** | One graph **per execution context** (the scenario path, per-test behaviour), and one **cross-repo** readout (probe weights shared across repos) | T3; T12; T2 |

---

## Phase 1(c): the ideal-observer graph and its cheapest approximation

**Ideal observer (infinite compute).** G\* is the **interventional dependency graph**. For every pair of program points (p, q), and for every input in the realistic input distribution of the repo's users and tests, it records the causal effect on q's observable behaviour of intervening on p. That covers p's code (every mutant), p's values (activation patching of program state), and the eventual test outcomes. G\* contains, as projections:
- the static call graph (its support under some input);
- the runtime call graph (realised under the observed inputs);
- dynamic data flow (value interventions);
- the necessity structure of fixes (IFG's object).

For localisation, the ideal observer just asks: which p, under the issue's scenario input, moves the observed symptom?

**Approximation ladder.** The cost column assumes one about 10k-LOC Python repo on 1 CPU and is GUESSED unless noted.

| Level | Object | Cost | What it loses |
|---|---|---|---|
| L0 | Lexical mention graph | ms | Everything non-lexical |
| L1 | Static AST/PyCG graph | s–h; PyCG times out on 6/50 (G-A1, VERIFIED) | About 51% of runtime edges (built-ins, MRO, names) |
| **L2** | **Import-time heap graph (T1)** | **seconds; no tests needed** | Edges that need values flowing at call time |
| L3 | Traced test-suite graph | minutes; needs a runnable suite | Untested paths; issue inputs are not the test inputs |
| L4 | Mutation kill-matrix coupling graph (T10) | CPU-hours | Value-level interventions |
| L5 | Value-patching interventional graph | days | Nothing on the tested inputs |

**Cheapest useful approximation.** Use L2 at inference (seconds, no tests, works on hidden private repos where running suites inside about 5 minutes is unsafe). Add an **amortised predictor trained on L3/L4 labels from other repos**, so execution supervision transfers to repos where nothing is executed. The predictor is either the T2 probe on Gemma's own forward pass or the T7 encoder. That is the central architectural claim of this note (INFERRED): *heap for the edges the runtime already resolved, amortised execution for the rest*.

---

## Phase 1(d): the twelve techniques after distillation

Each entry gives: mechanism → technique (what is built or trained, its objective, and how execution supplies labels) → surprising prediction → kill test (≤2 days, ≤5 GPU-h, CPU preferred), with a graph-free control and a degree-preserving rewire (DPR) where it is meaningful.

### T1. HEAP: an import-time reflective graph builder
- **Mechanism (1a).** A1's misses are mostly already resolved in the live heap.
- **Technique.** `import` every module under a sandbox and walk the live objects:
  - `inspect.getmro` and `super` targets;
  - `inspect.unwrap` / `__wrapped__` chains (decorators);
  - `__module__` + `__qualname__` canonical names, which collapse re-exports;
  - module-level dicts, lists and sets of callables (registries, dispatch tables, `singledispatch.registry`);
  - `functools.partial` targets;
  - class attributes bound to functions;
  - descriptor `__get__`/`property.fget`.

  These facts **resolve the AST's unresolved call sites**: receiver `self.x` plus the class MRO gives the defining method. The output is a directed, typed graph with edge kinds call, override, wraps, registered-in, re-export and contains. No training. **Execution labels**: the traced test-suite graph (L3) is the recall/precision yardstick.
- **Surprising prediction.** Without running a single test, HEAP recovers **≥40% of the runtime edges that static analysis misses** (GUESSED), at precision within 3 pp of static. The recovered share concentrates in the `super`, built-in and re-export categories of G-A1.
- **Kill test (CPU, about 1.5 days).**
  - Setup: 10 non-dev repos.
  - Runtime-edge recall, three ways: static vs static+HEAP vs traced.
  - Precision proxy: an edge is a *likely false positive* if its call-site line executed ≥5 times and the callee was never observed from that site.
  - Graph-free control: the L0 lexical graph.
  - DPR is not meaningful for recall. It is used in the downstream half (T1 spec, Phase 2).
  - **Kill** if the recovered share is <20% of static-missing runtime edges, or the precision drop is >5 pp.

### T2. LATENT: reading a directed runtime graph out of Gemma's residual stream
- **Mechanisms (2a vs 2b, competing; the test separates them).**
- **Technique.**
  - One forward pass of Gemma-4-31B over a packed context of functions (≤16k tokens).
  - Mean-pool the residual states over each function span at 3 layers (about L/4, L/2, 3L/4; E-A20).
  - Train a low-rank **bilinear probe** s(u→v) = h_uᵀ U Vᵀ h_v (rank 32; asymmetric, so it is directed) with BCE on **traced runtime labels**: runtime call edge, and runtime reachability within 3 hops.
  - The trained probe *is* a graph builder (threshold the scores) and an asymmetric embedding pair (Uᵀh as source vector, Vᵀh as target vector).
- **Surprising prediction.** On **runtime-only edges** (traced, but not static- or lexically visible, e.g. `super()`, decorators, registries), probe AUROC is ≥0.75. Meanwhile:
  - the lexical baseline is about 0.5 by construction;
  - Gemma's *verbal* answer to "does u call v at runtime?" is ≤0.6;
  - reachability AUROC is higher than adjacency AUROC (replicating E-A17 on code).
- **Kill test (≤3 GPU-h, plus CPU).**
  - Data: 8 non-dev repos, about 1.2M forward tokens.
  - Split: train on 5 repos, test on 3 (repo-disjoint).
  - Controls:
    - graph-free lexical-mention and name-similarity baselines;
    - Hewitt–Liang control: train the probe on **DPR-permuted labels** (each node keeps its in/out label degree), so hub or degree memorisation is caught;
    - consistent alpha-renaming of the test repos;
    - a positive control: explicit "A calls B" lines in context must be probe-decodable (AUROC ≥0.95), or the pipeline is broken.
  - **Kill** if runtime-only AUROC is <0.65, or within 0.05 of the DPR control.

### T3. HYTRACE: issue → imagined runtime path, decoded onto the graph
- **Mechanisms (4b, 5a).**
- **Technique.**
  - Gemma LoRA input: issue + a compact repo skeleton (signatures, about 6k tokens).
  - Output: an ordered call path `entry → … → culprit` of symbol names.
  - **Decoder**: an HMM/Viterbi over the directed graph (T1 or static). Hidden states are nodes, transitions are directed edges (with a small skip probability), and emissions are fuzzy name and docstring similarity to each imagined step. Node posteriors give the ranking.
  - **Execution labels.** Non-dev repos are mutated with in-family operators. Failing tests are traced *under the mutant* from the test entry to the mutated function, which gives the true path. Gemma writes a pseudo-issue from the failing output, with the culprit's name and traceback frames masked.
- **Surprising prediction.** The **decoded** path beats both (i) the imagined path's names alone and (ii) BM25 on the 79% of dev tasks whose issue names no gold node. The gain **disappears on a DPR-rewired graph**. Imagined paths still help when ≥50% of their names are wrong, because structure corrects names.
- **Kill test (zero-shot first, about 2 GPU-h vLLM).**
  - Run on the 118 dev tasks with no LoRA.
  - True paths: trace the FAIL_TO_PASS tests on the base commit.
  - Compare hit@5 over a fixed pool of 50 candidates: decoded vs names-only vs BM25 (graph-free) vs decoded-on-DPR.
  - **Kill** if decoded − names-only is <3 pp, or decoded ≈ decoded-on-DPR. The LoRA is trained only if zero-shot shows a structural effect (Phase 2).

### T4. DECONV: inverting the smoothing in supplied code-graph embeddings
- **Mechanism (3a, against 3b and 3c).**
- **Technique.**
  - Identify a low-pass operator g(Â) such that X ≈ g(Â)H. The candidates are Â^k, (I+αL)⁻¹ and an APPNP-style filter.
  - Fit the hyperparameters on held-out repos by maximising agreement of Ĥ = g(Â)⁻¹X (Tikhonov-regularised) with an *independent* text view of each node: TF-IDF or a small code embedder of the node's source.
  - Then fit a **ridge map W from a small open text encoder to Ĥ**. This gives the supplied embeddings a natural-language query encoder, which they lack today: `search_similar_code` takes symbol names only (HQ, VERIFIED).
  - Execution labels: gold edits of the 109 tasks whose gold patch passes tests (VERIFIED replica count).
- **Surprising prediction.** Ĥ makes **co-edited gold functions retrieve each other** (multi-function tasks) better than X does, and W fits Ĥ with clearly higher R² than X. The gains vanish when deconvolving with a **DPR-rewired Â**: wrong-graph inversion should *hurt*, which is a sharp mechanistic signature.
- **Kill test (CPU, hours).**
  - Arms: X, centred X, Soft-ZCA(X), Ĥ(true Â), Ĥ(DPR Â), Ĥ(true Â) with clone pairs removed.
  - The graph-free control is the text encoder alone.
  - The 3b check: Jaccard of shared neighbours for top-1 pairs.
  - **Kill** if Ĥ(true) − Ĥ(DPR) is <0.03 MRR on co-edit retrieval, or Ĥ(true) does not beat Soft-ZCA.

### T5. EFFECT: an execution-labelled effect parser with directed propagation
- **Mechanism (4a).**
- **Technique.**
  - From traces, record per function a 16-bit **effect signature**:
    - escaping exception types;
    - argument mutation (shallow state diff);
    - attribute writes on `self`;
    - returns None, self, new, or an arg-typed value;
    - I/O, logging or warnings;
    - global or module state writes.
  - Train a multilabel **linear probe on Gemma's frozen pooled states** (reusing T2's activations) to predict *local* effects from source.
  - Propagate *escaping* effects along directed edges: exceptions are the union over callees minus caught types, which needs T1 direction.
  - An LLM extracts effect descriptors from the issue, and ranking is by effect match fused (RRF) with BM25.
- **Surprising prediction.**
  - Propagated effects localise the gold node in tasks that name no gold symbol.
  - Effect-match retrieval is **unchanged under alpha-renaming** while BM25 collapses.
  - Propagating over a DPR graph destroys the gain *for propagated effects only*.
- **Kill test (CPU plus ≤1 GPU-h).**
  - Coverage: on 118 dev tasks, the fraction whose issue-described effect matches the traced effect of a gold node.
  - Then MRR in fixed pools: effect alone, effect+BM25, BM25 (graph-free), local-effect only (no propagation), propagation on DPR.
  - **Kill** if coverage is <25%, or effect+BM25 − BM25 is <2 pp MRR.

### T6. CHAIN: call-site probability distributions in place of binary edges
- **Mechanisms (1b, 2c).**
- **Technique.**
  - Each static or HEAP call site s gets a distribution p(callee | s) with mass ≤1. A small CPU model (gradient-boosted trees) predicts it from features: resolution kind, fan-out at s, receiver-type certainty, same-module, HEAP support.
  - It is trained with **cross-entropy against traced per-site callee frequencies** on non-dev repos.
  - Path score = ∏ p over distinct sites. Traversal expands by path mass, not by hop count.
- **Surprising prediction.**
  - At equal kept-edge recall, site-normalised mass gives higher **chain precision** than a per-edge classifier (the AutoPruner/CGPruner family, IV-3). That holds only if errors are site-correlated.
  - With mass-ranked expansion, **2-hop beats 1-hop** for localisation at a fixed node budget, the reverse of RepoGraph's result.
- **Kill test (CPU, 1 day).**
  - Chain precision and recall from test entries on 3 held-out repos: binary static vs per-edge classifier vs CHAIN.
  - Downstream: anchor→gold hit@k on dev at budgets 10/25/50 nodes; graph-free BM25; CHAIN on DPR.
  - **Kill** if CHAIN chain-precision ≤ per-edge + 0.03 at matched recall.

### T7. ASYM: directed source and target text embeddings trained on runtime reachability
- **Mechanism (I5, I9).** Cosine is symmetric; "what reaches f" and "what f reaches" differ.
- **Technique.** A small open embedder (e.g. Qwen3-Embedding-0.6B) gets two LoRA heads (source, target). The loss is an InfoNCE in which ⟨s_u, t_v⟩ ranks traced reachable pairs over unreachable ones in the same repo. Labels come from L3 traces. CPU or 1 L4.
- **Surprising prediction.** On unseen repos, ASYM retrieves *callers of the symptom function* better than any symmetric embedder at the same parameter count.
- **Kill test.** Directed retrieval (u→?, ?→v) on 3 held-out repos. Compare symmetric InfoNCE on the same pairs (a one-change control), and DPR labels.
- *Note.* T2's bilinear probe is the 31B version of this. ASYM is its portable, small-model instantiation. Flow2Vec (IV-7) already did asymmetric transitivity for value flow, and Callee does call-site/callee Siamese matching for binaries (IV-7): closeness about 2, INFERRED.

### T8. SITE: call sites as nodes, LLM type inference plus static closure
- **Mechanism (1c, I4).**
- **Technique.** For each unresolved call site, Gemma (LoRA) is shown the site, the enclosing function and the HEAP class index, and outputs the receiver's fully qualified type. Static closure (MRO lookup) then gives the callee. Labels are the traced callee per site. The training target is the type, not the callee name, so it is a short, low-entropy output.
- **Surprising prediction.** Site-local typing avoids the "empty graph" failure of whole-file LLM graph extraction (G-A7: gemma2-9B complete on 120 cases but sound on only 4), and beats HEAP alone on decorator and registry sites.
- **Kill test.** Zero-shot first (≤1 GPU-h): recall of HEAP-unresolved runtime edges on 3 repos. **Kill** if <+10 pp over HEAP.
- Closest prior: CG-Bench / SWARM-CG (E-A21, SNIPPET), which ask LLMs for call graphs directly.

### T9. TRACEBACK: serialising directed neighbourhoods as synthetic tracebacks
- **Mechanism (6a).**
- **Technique.** Render each directed path from an anchor as `Traceback (most recent call last): File "m.py", line n, in f` frames, with the signature line. This is a pure serialiser.
- **Surprising prediction.** With identical edges, traceback form beats adjacency-list and tree-indent forms on caller-vs-callee and reachability questions. The gain shrinks to 0 when the edges are DPR-rewired. That last part shows the format helps the model *read* edges and does not just trigger priors.
- **Kill test (≤1.5 GPU-h).** 600 execution-labelled reachability questions × 3 formats × {true, DPR}, plus a graph-free arm.
- **Risk.** A format effect alone is a weak paper (closeness to G-A12/A14, INFERRED).

### T10. KILLVEC: an amortised mutation-coupling graph (ideal-observer level L4)
- **Mechanism.** Coupling via shared state or config is invisible to call graphs.
- **Technique.**
  - Build a mutant × test kill matrix (CPU).
  - Draw an edge between f and g if their kill vectors overlap (Jaccard ≥ τ).
  - Train a probe on Gemma states (as in T2) to predict coupling from source.
- **Prediction.** ≥30% of coupled pairs have no static path of ≤3 hops (GUESSED).
- **Kill.** Measure the fraction on 5 repos (CPU only).
- **Flag.** This is IFG-adjacent (fix necessity via executed graphs). Keep it only as an *edge type* inside T2. It is not a headline.

### T11. RESIDUAL: show only the edges the model cannot already infer
- **Mechanisms (5c, 6c).**
- **Technique.** Score each edge's *inferability*: a lexical-mention test plus the T2 probe's confidence. Serialise only the low-inferability edges.
- **Surprising prediction.** The residual graph matches or beats the full graph on localisation at less than 30% of the tokens. The "containment plus lexical" complement performs at about the no-graph level (reproducing G-A14 mechanistically).
- **Kill test (≤2 GPU-h).** Fixed-pool localisation QA with arms full / residual / complement / none / residual-on-DPR.
- **Dependency.** It requires T2.

### T12. EXECCARD: behaviour cards as node text
- **Mechanisms (6b, 2c).**
- **Technique.** Node text = 2–3 traced input→output exemplars plus the effect signature, in place of source or an LLM summary. Labels come from L3 traces.
- **Prediction.** Cards beat source at less than 50% of the tokens for localisation, and they shrink over-extension.
- **Kill test.** Fixed-pool localisation QA: cards vs source vs LLM summary vs none.
- **Limitation.** Hidden repos need traces at inference, which is risky inside about 5 minutes. It is a weak artifact.

**Dropped during distillation:** OBSERVABLE (4c; folded into T5) and sibling co-edit retrieval (3b; folded into T4 as a competing explanation).

---

## Phase 2: six techniques strengthened

**Selection** was by surprise of the prediction, artifact value and cost:
- **T2 LATENT**, which absorbs T7/T10/T11 as heads and uses;
- **T1 HEAP**;
- **T3 HYTRACE**;
- **T4 DECONV**;
- **T5 EFFECT**;
- **T6 CHAIN**.

**How they fit together** (INFERRED): T1 (edges the runtime has already resolved) + T6 (path mass) + T2 (an amortised execution graph read from the model) form one **builder stack**. T3, T4 and T5 are three **query-to-graph parsers or embeddings** that use it. One paper can be built on T2 + T1 with the others as ablations, or any one can stand alone.

### P1. LATENT: execution-supervised latent graph readout ("the graph is already in the model")

**Full specification.**
1. **Data (CPU).**
   - About 12 non-dev pure-Python repos with test suites of ≥70% line coverage (GUESSED availability).
   - Trace the suites to get directed runtime call edges E_rt, reachability R_rt (≤3 hops), escaping effects, and per-site callee frequencies.
   - Edge strata:
     - **lexical**: v's name appears in u's body and static resolves it;
     - **static-only**: resolved but never executed;
     - **runtime-only**: executed, but neither lexical nor statically resolved. This is the stratum that matters.
2. **Contexts.** Pack functions into ≤12k-token windows in repo order, by module cluster (a function and its static/HEAP 2-hop neighbours, filling to the cap). Every labelled pair must be co-present.
3. **Forward pass.** Gemma-4-31B through HF transformers with `output_hidden_states`, served as W4A16 compressed-tensors or bf16 (62 GB fits in 4×24 GB with `device_map`). Keep per-function mean-pooled states at 3 layers, so storage is about 3×d floats per function.
4. **Probe heads**, trained on CPU or 1 GPU in minutes:
   - (a) bilinear directed edge s(u→v);
   - (b) bilinear reachability ≤3;
   - (c) a linear multilabel effect head (shared with P5);
   - (d) an optional coupling head (T10 labels).

   Loss: BCE with pairs sampled within a window, plus hard negatives from the static-only stratum.
5. **Artifact.** The probe weights (a few MB) are the graph builder: one forward pass gives a directed runtime-graph estimate. U and V projections give **asymmetric node embeddings** (source/target) that replace the supplied symmetric, smoothed 256-d vectors.
6. **Uses.**
   - RESIDUAL serialisation (T11): show only edges with low probe confidence that are not lexical.
   - A cross-model "redundancy meter": probe AUROC vs explicit-edge gain across model sizes.

**Compute (GUESSED).**
- Kill test: about 1.2M tokens at about 1.5k tok/s ≈ 0.25 h on one 4×L4 node, so about 1–1.5 GPU-h including overhead.
- Full study (12 repos, 3 layers, 3 seeds of repo splits): about 3M tokens, about 3 GPU-h.
- Small-model replication: Gemma-4-E4B and one non-Gemma model (e.g. Qwen3-8B), <1 GPU-h each.
- Dev-set use for retrieval: about 129 tasks × 30k tokens ≈ 4M tokens, about 3 GPU-h.
- **Total about 8 GPU-h.** No LoRA is needed: the probe is the only trained object.

**Generalisation.**
- Other models: the probe is retrained per model on the same execution labels, for minutes.
- Other hardware: forward passes only.
- Other languages: the labels come from any tracer (Node `--cpu-prof`, Java JFR). The method is language-agnostic. A JS replication on 3 repos is the generality test (CPU tracing; ≤1 GPU-h).

**More than SFT-on-items.**
- No weights of the LLM change.
- The claim concerns what an *untouched* model represents, and turns that into a reusable builder evaluated on repos and languages it never saw.
- Labels are execution facts, not answers to items.

**Predicted effect (GUESSED) and falsifiers.**

| Claim | Predicted | Falsifier |
|---|---|---|
| H1: runtime-only edge AUROC | ≥0.75 | <0.65, or within 0.05 of the DPR-permuted-label control |
| H2: reachability > adjacency | +0.05 AUROC | ≤0 |
| H3: probe > the model's verbal answer on runtime-only edges | ≥+0.15 AUROC | Verbal ≥ probe |
| H4: invariance under alpha-renaming | Drop ≤0.05 | Drop >0.15, which means the probe reads names |
| H5: middle layer best | — | Last layer best (informative, not fatal) |
| H6: RESIDUAL serialisation ≥ full graph at <30% of tokens on fixed-pool QA | — | Residual < full by ≥3 pp |

**Positive control.** Explicit "A calls B" text in context must decode at ≥0.95.

**Publishable if it fails.**
- If 2a is true (the probe tracks lexical/static and not runtime reachability), that is a clean mechanistic explanation of G-A4's over-extension: *the model's latent call graph is static-lexical, so it over-extends because it faithfully reads out the wrong graph*. No paper probes code-LLM residuals for repository call graphs (E Q4 gap, VERIFIED absence by the scout; IV-1 found only tool-call and reasoning-DAG probes).
- **Prior-art risk.** Probing for structure is a known method (Hewitt–Manning; E-A17 on tool graphs). The novelty is the *runtime-vs-static label contrast on real repos* and its *use as a builder*. That is closeness 2 to E-A17 (INFERRED). The critic should search "probe call graph code LLM 2026".

### P2. HEAP: an import-time reflective graph builder (graph generation at level L2)

**Full specification.**
1. A sandboxed importer: a subprocess with a timeout, `sys.modules` isolation, network blocked. It imports every package module and records import failures, which are themselves a metric.
2. **Heap walk.** For every function, class and module object:
   - canonical id = `__module__` + `__qualname__` (re-export dedup);
   - MRO;
   - `inspect.unwrap` chains;
   - `functools.partial` / `singledispatch.registry`;
   - module-level containers of callables (registries, routing tables), including those filled by decorators at import time;
   - `property` / descriptor accessors;
   - `__init_subclass__` effects.
3. **Resolution pass.** For each AST call site with an unresolved target, resolve `self.m`, `cls.m` and `super().m` through the MRO. Resolve `name(...)` through the module's live globals, which handles re-exports and aliasing. Resolve `registry[key](...)` to all registered callables, as a typed `dispatch` edge with mass split as in T6. Resolve built-in-type receivers when the receiver's type is a literal or annotation.
4. **Output.** A directed multigraph with typed edges: `call`, `override`, `wraps`, `dispatch`, `reexport`, `contains`. Containment is kept *separate* (G-A14).
5. **Ground truth.** The traced L3 graph on the 10–12 non-dev repos, plus the 4 dev repos for the competition graph comparison. The supplied graph is undirected with a mixed label (HQ, VERIFIED).

**Compute.** CPU only. About 5–60 s per repo for import and walk (GUESSED); tracing the suites takes minutes to an hour per repo. **0 GPU-h** for construction. Downstream evaluation reuses P3/P5 GPU lines.

**Generalisation.** Any reflective runtime:
- JS: module namespace objects, prototype chains, `Function.prototype.bind`;
- Ruby: `ancestors`, `method(:x).owner`;
- Java: classloading plus reflection, closest to TamiFlex.

It is model-independent and offline, and it runs on hidden repos inside the scorer's sandbox if the package imports (the import-failure rate is a reported limitation).

**More than SFT-on-items.** It is not learned at all. It is a builder, scored against execution.

**Predicted effect (GUESSED) and falsifiers.**
- Recall of runtime edges: static about 50–70% (consistent with G-A1 at 49% and PyCG's own 69.9%, IV-2, SNIPPET); static+HEAP about +15–25 pp; precision change ≤3 pp.
- By category: `super`/MRO and re-export misses mostly recovered; built-in-type methods only partly.
- **Downstream** (dev, fixed 50-node pool, issue anchors → gold within k hops along *directed* edges), with arms:
  - supplied undirected graph;
  - static-directed;
  - HEAP-directed;
  - HEAP-directed DPR (in/out degree and module preserved);
  - BM25 (graph-free).

  Prediction: HEAP-directed > static-directed by ≥3 pp hit@10, and HEAP > HEAP-DPR.
- **Falsifiers:** recovered share <20%, or no downstream difference from static-directed.

**Publishable if it fails.** A measured decomposition: "which runtime edges are already resolved at import time vs only at call time", by category, across about 12 repos. This is the missing real-world breakdown flagged as a gap in G Q1 (no recall numbers for practical tools beyond PyCG and Jarvis).

**Prior-art risk.** TamiFlex (Java, reflection logs) and dynamic or hybrid Python graphs (DynaPyt, Scalpel) are the nearest. I found no Python import-heap call-graph builder in a light search (IV-2: no hit). That is a weak absence. The critic must search "import-time" / "heap snapshot" Python call graph and "hybrid call graph Python runtime introspection".

### P3. HYTRACE: an imagined execution path decoded onto the directed graph (a query parser)

**Full specification.**
1. **Skeleton.** A per-repo listing of module → class → `def name(sig)` with first docstring lines, cut to ≤6k tokens by dropping tests and docs_src. Clone classes are collapsed (3c).
2. **Generator.** Gemma-4-31B, prompted (zero-shot) or with LoRA (r16), emits `PATH: entry_api -> a -> b -> culprit`, 2–8 steps, plus a confidence per step.
3. **Decoder (the novel parsing piece).**
   - An HMM over the directed graph G (HEAP+static; CHAIN mass as transition probabilities, with a skip/teleport ε).
   - Observation t = imagined step t; emission = max(name-token Jaccard, embedding similarity of name+docstring).
   - Forward–backward gives node posteriors, and the culprit posterior is the ranking.
   - This is CPU work, in milliseconds per task.
4. **Training data (execution-labelled).**
   - Non-dev repos.
   - Mutation operators drawn *from each repo's own constructs* (in-family, to avoid the K-12 template shortcut).
   - Keep mutants that make ≥1 test fail; trace the failing test from entry to culprit (the true path).
   - Gemma writes a pseudo-issue from the failure output, with the culprit's name, its file and traceback frames masked. A leak checker rejects any issue containing a culprit name token.
   - About 1,200 items.
5. **Training.** LoRA SFT on the true path (teacher forcing). A **decoder-aware filter** keeps items where the true path is decodable on G. This is SFT, but the scored artifact is the *generator+decoder pair evaluated on real issues of unseen repos*, and the claim is about the decoder's structural correction.

**Compute (GUESSED).**
- Zero-shot kill: 118 tasks × about 8k tokens ≈ 1M prefill on vLLM, ≤1 GPU-h.
- LoRA: 1,200 × 6.5k ≈ 7.8M tokens per epoch at about 500 tok/s ≈ **4.3 GPU-h per epoch**. Use 1 epoch, 3 seeds only if zero-shot passes, so the minimum is 4.3 and the maximum is 13 GPU-h. *Flagged: 3 seeds alone eat half the study budget.* A cheaper seed plan is 1 full seed plus 2 half-data seeds ≈ 8.6 GPU-h.
- Eval per arm: ≤1 GPU-h.

**Generalisation.** The decoder is model-agnostic: any LLM's imagined path can be decoded, so a frontier API model can be tested zero-shot as a portability check. Language-agnostic given a directed graph. The LoRA artifact transfers across repos by construction, since test repos are disjoint.

**Predicted effect (GUESSED).**
- Zero-shot decoded hit@5 ≥ names-only + 5 pp and ≥ BM25 + 3 pp on the no-name stratum (about 93 tasks).
- Decoded on DPR ≈ names-only.
- LoRA adds +5 pp over zero-shot.
- Path-step name error rate ≥50%, yet decoding still helps.

**Falsifiers.** Decoded ≈ names-only, decoded ≈ DPR, or no gain on the no-name stratum. Positive control: the decoder fed the *true* traced path (with names noised at 50%) must recover the culprit ≥80% of the time. Otherwise the decoder, not the generator, is the problem.

**Publishable if it fails.** "Gemma's imagined runtime paths": how often a 31B model's scenario simulation hits the true executed path, and where it breaks (decorators, `__getattr__`, per SWE-FLUX's qualitative finding G-A4). This would be the first quantitative measure from issue text rather than from code.

**Prior-art risk.** HyDE-style query rewriting (GREPO, in HQ Map 1), IssueExec / OrcaLoca (real traces from executed reproductions, IV-4), and log-to-trace estimation (2506.19045, IV-4: estimating traces from failure logs at about 90% F1, SNIPPET). The distinguishing pieces are no execution at inference and *structural HMM decoding of an imagined path*. Closeness about 2 (INFERRED).

### P4. DECONV: embedding forensics, inverting the graph smoothing and adding an NL query encoder

**Full specification.**
1. **System identification.** For each candidate operator g ∈ {Â^k (k=1..4), (I+αL)⁻¹, APPNP(α,K)}, with Â = sym-normalised adjacency plus self-loops from the supplied graph:
   - compute Ĥ = argmin ‖g(Â)H − X‖² + λ‖H‖² (CG solver; sparse);
   - score with the held-out-repo criterion: mean CKA between Ĥ and an independent text view (TF-IDF of source; a small code embedder).
   - Pick (g, params, λ) on 2 repos; test on 2.
2. **Retrieval uses.**
   - (i) co-edit retrieval: gold function i → other gold functions of the same task, on the 65 multi-function tasks;
   - (ii) duplicate-code detection;
   - (iii) **NL querying**: ridge W from a small text encoder (on node source) to Ĥ; issue → W·e(issue) → kNN in Ĥ, re-smoothed by g for neighbourhood context.
3. **Artifact.** A procedure that identifies and inverts the smoothing in *any* GNN-smoothed embedding release, plus a linear NL bridge into it.

**Compute.** CPU, hours. About 4k-node sparse solves.

**Generalisation.** The approach applies to any graph-smoothed embedding product: code-graph releases, the SPECTER-style citation smoothing common in RAG indexes, and GNN embeddings shipped without encoders. The general claim to test is "smoothed embeddings shipped without encoders can be partly un-smoothed and queried".

**More than SFT-on-items.** The only trained parameters are the deconvolution hyperparameters and a ridge map.

**Predicted effect (GUESSED) and falsifiers.**

| Claim | Predicted | Falsifier |
|---|---|---|
| CKA(Ĥ, text) − CKA(X, text) | +0.1 | No gain |
| Co-edit MRR, Ĥ(true Â) over X | +0.05 | Ĥ(true) − Ĥ(DPR Â) <0.03 |
| Ĥ(DPR) vs X | Ĥ(DPR) < X (wrong-graph inversion injects noise) | Ĥ(DPR) ≈ Ĥ(true): the gain is generic high-pass/whitening |
| Ĥ(true) vs Soft-ZCA(X) | Ĥ(true) better (E-A6 is the strong baseline) | Ĥ ≤ Soft-ZCA |
| R²(W→Ĥ) vs R²(W→X) | Higher for Ĥ | No difference |
| Competing explanation 3b: top-1 pairs share ≥2 neighbours | — | If true, *X* (not Ĥ) is better for co-edits. A sign flip that is itself a finding: smoothing *helps* multi-function fixes |

**Execution role.** Weak, honestly. Labels are gold edits of patches that pass their tests. This is the least execution-grounded of the six.

**Publishable if it fails.** "What the competition's node embeddings are": an identified operator and its parameters, a short and reproducible forensic note. Small.

**Prior-art risk.** Graph deconvolution exists (GDN, NeurIPS 2021, IV-6) and model-inversion attacks on GNNs exist. Using it to *recover a retrievable text signal and an NL query bridge* for code-graph releases was not found in a light search. Closeness about 2–2.5 for the method (INFERRED). **This is the weakest on novelty, but the cheapest and sharpest kill test.**

### P5. EFFECT: an execution-labelled effect parser with directed propagation (role embeddings)

**Full specification.**
1. **Labels (CPU).** During traced test runs, for each function invocation record:
   - escaping exception type(s);
   - argument mutation: a shallow fingerprint of mutable args before and after (len, keys, `__dict__` keys, hash of repr for sizes below a cap);
   - `self` attribute writes;
   - return class ∈ {None, self, arg-identity, new instance of an arg type, other};
   - calls into I/O, `logging` or `warnings`;
   - module-global writes.

   Aggregate per function into a 16-bit *local* signature (an effect observed in ≥10% of calls) and a *propagated* signature (escaping effects attributable to callees).
2. **Predictor.** Linear multilabel heads on Gemma's pooled states (shared forward with P1). A HEAP+direction propagation rule: escaping exceptions = local ∪ ⋃ callees(escaping) − caught types, using the AST `except` clauses.
3. **Query parser.** Gemma prompted to map issue text to an effect query: exception types named, "modified/changed in place", "returns None", "warns", "writes". Score = weighted Hamming match on the propagated signature, fused with BM25 by RRF.
4. **Artifact.** A node-level effect embedding (16-d, interpretable) plus an issue→effect parser. Effects are a vocabulary shared by issue text and code, and names never enter it.

**Compute.** Labels: CPU. Probe: shares the P1 forward pass (+0 GPU-h). Issue parsing: 118 short prompts, ≤0.2 GPU-h. **Total ≤0.5 GPU-h incremental.**

**Generalisation.**
- Effect vocabularies exist in every language (exceptions, mutation, I/O). The heads can be retrained per model.
- Alpha-renaming invariance is predicted by construction. That makes it the one technique aimed straight at the "agents localise by names" finding (HQ Deep-research verdict; CABRA G-A5; E-A9).

**More than SFT-on-items.** No LLM training. The labels are execution facts about code, not answers to localisation items.

**Predicted effect (GUESSED).**
- Coverage: ≥35% of dev tasks have an issue-stated effect matching a traced gold effect.
- RRF(effect, BM25) − BM25 ≥ +3 pp MRR overall and ≥ +5 pp on the no-name stratum.
- Under renaming, BM25 drops ≥10 pp while effect drops ≤2 pp.
- Local-only effects underperform propagated ones, and propagation on DPR loses the propagated gain.

**Falsifiers.** Coverage <25%; fused gain <2 pp; or the propagated ≈ local ≈ DPR propagation (direction and graph irrelevant). Positive control: the *traced* (oracle) gold signature with the parsed issue query must reach a high MRR, which separates parser failure from representation failure.

**Publishable if it fails.** The coverage number itself: "how often do issues describe behaviour that execution can verify on the culprit". This is a new descriptive statistic on SWE data (INFERRED absence: IV-5 found exception *prediction* benchmarks, ThrowBench, and trace models, CWM/E.T., but not effect-matched localisation).

### P6. CHAIN: call-site distributions and path mass (graph generation with graded edges)

**Full specification.**
1. **Sites.** Every AST call site in static+HEAP, with its candidate callee set C(s).
2. **Model.** p(c | s) = softmax over C(s) ∪ {∅ (unobserved/external)} of f(features). The features are resolution kind (direct/MRO/registry/fuzzy/builtin), |C(s)|, type certainty, module distance, HEAP support, and whether c is a test helper. f is LightGBM. Loss: cross-entropy vs traced per-site callee frequencies (Laplace-smoothed), trained on 8 non-dev repos and tested on 4.
3. **Path mass.** m(path) = ∏ p(c_i | s_i). Expansion is best-first by mass with a node budget B, so this is graph generation as *probabilistic reachability*.
4. **Artifact.** The CPU model plus mass-weighted graphs, which T3 uses as HMM transitions and T1 as dispatch-mass splits.

**Compute.** CPU only. Downstream reuses the P2 evaluation.

**Generalisation.** Any language with a tracer and a static resolver. Model-agnostic, because the downstream consumer can be any LLM or a non-LLM retriever.

**More than SFT-on-items.** It is a calibrated estimator of runtime path mass, scored by chain precision against execution.

**Predicted effect (GUESSED) and falsifiers.**
- At matched edge recall, CHAIN chain precision ≥ per-edge classifier + 0.05 (the AutoPruner/CGPruner-style baseline, same features, per-edge BCE: a one-change control).
- Mass-ranked 2-hop > 1-hop at B = 25 nodes on dev anchor→gold hit, while unweighted 2-hop < 1-hop (reproducing RepoGraph within our data).
- Mass on DPR ≈ unweighted.
- **Falsifier:** no chain-precision gain over per-edge at matched recall. That would mean errors are not site-correlated, and 1b is false.

**Publishable if it fails.** The first chain-level (not edge-level) measurement of *why* Python call-graph errors compound: site-correlated vs independent. This extends Jarvis's 0.99 → 0.19 (G-A3).

**Prior-art risk.** ML call-graph pruning exists (AutoPruner FSE'22, CGPruner, Mir et al. MSR'24; IV-3). JIT compilers keep weighted call-target profiles (Jikes RVM `WeightedCallTargets`, IV-8). LARGER uses confidence-weighted K-hop reachability (IV-7). The new parts are learning site-normalised mass *to predict chain validity and drive LLM context expansion*. Closeness 2–2.5 (INFERRED). **This is the most crowded of the six.**

---

## Budget roll-up for the six (GUESSED; counts epochs and seeds)

| Line | GPU-h (min / central / max) |
|---|---|
| Throughput and memory probe (unmeasured FSDP ≤16k) | 0.5 / 1 / 1.5 |
| P1 LATENT: kill + full + 2 replication models + dev forward | 4 / 8 / 11 |
| P2 HEAP | 0 (CPU) |
| P3 HYTRACE: zero-shot kill + eval arms | 1.5 / 2.5 / 4 |
| P3 HYTRACE: LoRA (only if zero-shot passes; 1 full + 2 half seeds) | 0 / 8.6 / 13 |
| P4 DECONV | 0 (CPU) |
| P5 EFFECT | 0.2 / 0.5 / 1 |
| P6 CHAIN | 0 (CPU) |
| Retention or downstream sanity (one agent run per final arm; not powered) | 2 / 4 / 7 |
| **Total** | **8.2 / 24.6 / 37.5** |

**The central estimate fits 25 GPU-h only if HYTRACE's LoRA stays to one full seed. The pessimistic case does not fit.** The cheapest coherent paper is P1 + P2 + P5 + P6, at about 10–13 GPU-h. It needs no LoRA, which makes it the most "artifact" and the least "SFT".

**Kill order.** All CPU kills run in week 1 first (P2, P4, P6, the P5 coverage count, and the P1 label strata counts). Then the P1 GPU kill (about 1.5 GPU-h). Then the P3 zero-shot kill. Nothing trains until its kill passes.

---

## Search log (light checks; novelty judgement is for the critic)

| ID | Query (WebSearch, standard) | Result used |
|---|---|---|
| IV-1 | probing LLM hidden states call graph reachability code repository linear probe dynamic dispatch | Only tool-call dependency probes (2605.25310, already E-A17), the reasoning-DAG probe "From Chains to DAGs" (2601.17593), and tool-error probes (2608.27750). **No code-repo call-graph probe found** (SNIPPET-level absence) |
| IV-2 | Python call graph import-time reflection runtime objects decorators `__wrapped__` hybrid static analysis recall | PyCG (recall about 69.9% on its benchmark, SNIPPET); `callgraph` decorator tracing. **No import-heap builder found** (weak absence) |
| IV-3 | AutoPruner CGPruner pruning false positive call graph edges dynamic ground truth | AutoPruner (FSE'22), CGPruner (precision 24% → 66% with recall loss), Mir et al. MSR'24 (+25% precision / −9% recall, NYXCorpus). All per-edge Java pruning. SNIPPET |
| IV-4 | LLM predict execution trace call path from issue description fault localization hypothetical trace | IssueExec 2607.17286 (executed caller-callee graph + LLM), OrcaLoca, trace estimation from logs 2506.19045 (about 90% F1), stack-trace LLM 2501.18005. **No imagined-path decoding found.** SNIPPET |
| IV-5 | LLM infer function side effects exceptions mutation from source trained on execution traces | CWM, E.T. 2503.05703, ThrowBench 2503.04241, TRACED, CodEx (FSE'25). **No effect-matched localisation found.** SNIPPET |
| IV-6 | graph deconvolution invert GNN smoothing embeddings recover node features retrieval | GDN (NeurIPS'21) inverse filter + wavelet denoising. No retrieval use found. SNIPPET |
| IV-7 | directed asymmetric code embeddings caller callee source target embedding reachability | Flow2Vec (OOPSLA'20, asymmetric transitivity over value flow), Callee (binary call-site Siamese), kTrans, LARGER (confidence-weighted K-hop reachability), SEA. SNIPPET |
| IV-8 | probabilistic call graph call site distribution over targets runtime frequency weighted call graph LLM retrieval | Jikes RVM `WeightedCallTargets` (JIT profiles); Oracle patent "Probabilistic call-graph construction" US10719314 (method unread). No LLM-retrieval use found. SNIPPET |

**Searches the critic should run** (not done here):
- "import-time introspection call graph Python";
- "probe code LLM call graph hidden states 2026";
- "hypothetical execution path issue localization";
- "effect signature bug localization";
- "embedding deconvolution retrieval";
- an OpenReview ICLR 2027 sweep on "code graph probe" and "runtime call graph LLM".

Sources: [2601.17593](https://arxiv.org/html/2601.17593v1), [2608.27750](https://arxiv.org/pdf/2608.27750), [PyCG](https://arxiv.org/pdf/2103.00587), [callgraph](https://callgraph.readthedocs.io/), [AutoPruner](https://arxiv.org/pdf/2209.03230), [Mir et al.](https://arxiv.org/abs/2402.07294v1), [IssueExec](https://arxiv.org/pdf/2607.17286), [OrcaLoca](https://arxiv.org/pdf/2502.00350), [2506.19045](https://arxiv.org/html/2506.19045v2), [2501.18005](https://arxiv.org/abs/2501.18005v3), [CWM](https://arxiv.org/pdf/2510.02387), [E.T.](https://arxiv.org/pdf/2503.05703), [ThrowBench](https://arxiv.org/pdf/2503.04241), [GDN](https://proceedings.neurips.cc/paper/2021/hash/afa299a4d1d8c52e75dd8a24c3ce534f-Abstract.html), [Flow2Vec](https://2020.splashcon.org/details/splash-2020-oopsla/109/Flow2Vec-Value-Flow-Based-Precise-Code-Embedding), [Callee](https://arxiv.org/pdf/2111.01415), [LARGER](https://arxiv.org/pdf/2605.16352), [Jikes RVM WeightedCallTargets](https://jikesrvm.sourceforge.net/apidocs/latest/org/jikesrvm/adaptive/database/callgraph/WeightedCallTargets.html), [US10719314](https://patents.google.com/patent/US10719314).
