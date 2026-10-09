# Phase 3 red team: prior art against the six refined techniques (9 Oct 2026)

**Target.** `inventor_phase1_2.md`: Phase 2 techniques P1 LATENT, P2 HEAP, P3 HYTRACE, P4 DECONV, P5 EFFECT, P6 CHAIN, plus T7–T12 briefly.

**Question asked of each.** Has someone already exploited the *same mechanism*? Not the same words, and not the same benchmark.

**Labels.**
- **VERIFIED**: I read the primary text (abstract via the arXiv API, or the PDF text extracted locally).
- **SNIPPET**: from a search-engine summary or an OpenReview search hit only.
- **INFERRED**: my reasoning.

The search log (RT-n) is at the end.

**Closeness scale.**
- **0**: unrelated.
- **1**: same goal, different mechanism.
- **2**: same mechanism in a different domain, language or use, or the same mechanism partly done.
- **3**: same mechanism done, in substance, for this setting.

---

## Verdict table

| # | Technique | Closest prior (closeness) | Surprising prediction already answered? | Verdict |
|---|---|---|---|---|
| P1 | LATENT | Tool-call dependency probe 2605.25310 (2); data-flow-edge probes on code models, Troshin & Chirkova 2202.08975 and Anand et al. ACL'24 2406.11930 (2); reasoning-DAG probe that *reconstructs graphs* 2601.17593 (2); execution-labelled state probes, Jin & Rinard ICML'24 (1.5) | **No.** No paper found that probes a code LLM for *runtime-only* call edges on real repos, or contrasts runtime with static labels | **SURVIVE** (narrowed claim) |
| P2 | HEAP | **Approximate interpretation**, Laursen, Xu & Møller, PLDI'24 (2.5); **HeapDL**, OOPSLA'17 (2.5); **PoTo**, ECOOP'25 (2); TamiFlex (2) | **Largely yes, for JS.** Running module and library initialisation code with no tests, then feeding the facts to a static call-graph analysis, raised runtime-edge recall from 75.9% to 88.1% with precision −1.5% (VERIFIED). That recovers ≈51% of the missed edges, which is the inventor's "≥40% without tests" prediction | **NARROW** (it becomes a Python port and a decomposition; no longer a headline) |
| P3 | HYTRACE | HMM map-matching (Newson & Krumm 2009; graph path localisation by Viterbi, IEEE TSP 2018) (2 for the decoder); **RoG / GCR** for KGQA, where the LLM plans a relation path and it is grounded on the graph (2); SWE-Debate 2507.23348 and GraphLocator 2512.22469, with fault-propagation chains on dependency graphs (1.5–2); static trace estimation from logs 2506.19045 (1.5); ND-Slicer, FSE'24, predicted dynamic dependences (1.5) | **No.** Nobody measures how often an LLM's *imagined* runtime path from issue text matches the executed path, or whether structural decoding corrects ≥50% wrong names. The DPR signature is untested | **NARROW** (the decoder is a known algorithm transplanted; the claim must be the measurement) |
| P4 | DECONV | **GDN**, NeurIPS'21 (2.5 for the inversion); **blind identification / deconvolution of graph filters** (Segarra et al. TSP'17; Ye & Mateos 2412.15133, robustness to graph perturbations) (2.5 for the system-ID step); reverse-process GNNs, ICML'24 (2) | **Partly.** That deconvolution with a perturbed graph degrades is studied in graph signal processing (SNIPPET). The specific "DPR-Â inversion is worse than X" for retrieval is not. No retrieval or NL-bridge use was found | **NARROW** (forensic note or ablation; not a paper) |
| P5 | EFFECT | EXCEPT, exception-type-driven FL (1.5); SBEST 2405.00565, stack-trace exceptions plus call-graph distance (1.5); ThrowBench and CRUXEval-exception (1, prediction only); JPure, dynamic side-effect inference (1, labels only); ExecRetrieval 2609.01865 (1) | **No.** No effect-signature-matched localisation was found, and no coverage statistic of the form "issue-stated effect = traced gold effect" | **SURVIVE** |
| P6 | CHAIN | **AFLGopher** 2511.10828 (2): an LSTM gives a per-call-site softmax over statically enumerated targets, learned from traces and used to weight call-graph distance (VERIFIED). **Pereira 2015** (UFRGS MSc, Ruby) (2): static call-graph edges annotated with test-run likelihood, used to rank candidate targets at a site (VERIFIED). Jikes RVM `WeightedCallTargets` (1.5). AutoPruner and Mir et al. MSR'24 / OriginPruner (2, per-edge). Probabilistic pointer analysis, ASPLOS'06 (1.5) | **No.** Nobody tests whether *site-normalised* mass gives better **chain** precision than per-edge pruning at matched recall (the site-correlated-error hypothesis) | **NARROW** (the technique exists; only the hypothesis test plus cross-repo transfer is new) |

T7–T12 are covered briefly at the end. Overall:
- The two **builder** claims are the most exposed: HEAP is near-prior-art, and CHAIN is "a learned per-site distribution" that already exists.
- LATENT and EFFECT survive.
- HYTRACE and DECONV survive only as measurements.

---

## P1. LATENT: execution-supervised readout of a directed runtime graph from Gemma's residual stream

**Closest prior.**
1. **"Tool-Call Dependency Structure is Linearly Decodable in LLM Agent Residual Streams"**, arXiv 2605.25310; ACL ARR Aug 2026 submission (OpenReview DYVfFM2UfZ).
   - A low-capacity edge probe on Qwen3-32B decodes the agent's *runtime tool-call* dependency graph.
   - Controls: Hewitt–Liang random-label, positional baseline, activation patching.
   - The abstract says prior structural probes "have targeted static code or chain-of-thought text".
   - VERIFIED (abstract). Closeness **2**: same method family (directed pairwise edge probe with a Hewitt–Liang control), different graph (tool I/O, not code calls).
2. **Troshin & Chirkova, "Probing Pretrained Models of Source Code"**, 2202.08975 (BlackboxNLP'22): linear probes for data-flow-graph edges between tokens. **Anand et al., "A Critical Study of What Code-LLMs (Do Not) Learn"**, ACL'24, 2406.11930: token-pair probes for DFG edges. They report weak identifier–syntax relations, and *larger* models encoding less. SNIPPET. Closeness **2**: the edge-probe mechanism exists on code, but on intra-snippet static data flow with small models.
3. **AST-Probe** (ASE'22, 2206.11719): a structural probe recovers the whole AST from a syntactic subspace in the middle layers. SNIPPET. Closeness 1.5.
4. **"From Chains to DAGs: Probing the Graph Structure of Reasoning"**, 2601.17593. Probes predict adjacency and distance, and are used to **reconstruct approximate graphs**, which is the "probe as builder" use. VERIFIED (abstract). Closeness 2 for the builder framing.
5. **Jin & Rinard, ICML'24** (2305.11169): probes trained on **execution-derived labels** (intermediate program states), with an interventional baseline. SNIPPET. Closeness 1.5: execution labels as probe targets, but on a synthetic DSL with no graph.

**Remaining delta (INFERRED).**
- Directed *runtime-only* call and reachability edges are new. These are edges traced but neither lexically nor statically visible: `super`, decorators, registries.
- So are the runtime-vs-static label contrast, the setting (real multi-file repos, function spans), and a frozen 31B model.
- No hit in OpenReview, the arXiv API or web for "probe" × "call graph" × code LLM. The arXiv API query `all:probing AND all:"call graph"` returns 4 results, none of them on code-LLM call graphs (RT-17).

**Is the surprising prediction answered?** No. H1 (runtime-only AUROC ≥0.75 vs DPR control) and H3 (probe > verbal) are untested on code.
- E-A17 (2605.25310) already answered the *analogue* of H2 (reachability > adjacency) for tool graphs, so H2 on code is a replication, not a surprise.
- The "probe vs verbal" gap is also already shown in other domains: E-A19, and the 2026 CWE paper (vulnerability probes beat verbal answers; ICLR 2026 submission 2964, SNIPPET).

**Hostile notes.**
- (a) **The format confound is real.** "Linear Probes Detect Task Format, Not Reasoning Mode" (TrustNLP'26) shows that residualising surface features can drop probes to chance (SNIPPET). A mean-pooled function-span bilinear probe can read name co-occurrence and position in the packed window. The planned controls (alpha-renaming, Hewitt–Liang via DPR labels) are necessary. Add **window-position residualisation**, because packing "in repo order by module cluster" makes position a proxy for static proximity.
- (b) **Anand et al.** found *larger* models encode *less* pairwise code structure. That raises the prior for the 2a outcome ("latent graph is lexical/static"), which the inventor already pre-registers as publishable.
- (c) The runtime-only stratum may be small and dominated by a few patterns (`super`, one registry), so probe AUROC could be pattern-specific. Report per-category AUROC.

**Verdict: SURVIVE, narrowed.**
- The headline must be "runtime-only vs static-visible decodability in a frozen code LLM", with method credit to 2605.25310 and the Troshin and Anand lineage.
- The "probe as graph builder" angle should cite 2601.17593.

---

## P2. HEAP: import-time reflective Python graph builder

**Closest prior.**
1. **Laursen, Xu & Møller, "Reducing Static Analysis Unsoundness with Approximate Interpretation"**, PLDI 2024 (PACMPL 8, art. 194), tool **Jelly**. VERIFIED (PDF).
   - A forced-execution dynamic analysis runs *library API initialisation code* with no test suite. It infers determinate facts (dynamic property reads and writes) and feeds them as hints into a static call-graph analysis.
   - On 141 Node.js projects: +55.1% call edges and +17.7% resolved call sites.
   - On 36 projects with dynamic call graphs from test suites: **recall 75.9% → 88.1%, precision −1.5%**. In one case recall went 40.1% → 98.0%.
   - Closeness **2.5**: same mechanism ("the runtime state built at module initialisation resolves what static analysis misses; no tests needed"), same ground-truth protocol (dynamic call graph from test suites), different language.
2. **HeapDL: "Heaps Don't Lie: Countering Unsoundness with Heap Snapshots"**, Grech, Fourtounis, Francalanza & Smaragdakis, OOPSLA'17 (1905.02088), plus "Shooting from the Heap", ISSTA'18.
   - Whole-heap snapshots are fed as extra facts to a static points-to and call-graph analysis.
   - On DaCapo it computes **99.5% (median) of call-graph edges of unseen dynamic executions vs 76.9% for TamiFlex**.
   - VERIFIED (abstract). Closeness **2.5**: same "heap facts complete the static call graph" mechanism. Java, with snapshots taken during execution rather than at import.
3. **PoTo / PoToCG**, Rak-amnouykit, Milanova, Baudart, Hirzel & Dolby, ECOOP'25. VERIFIED (PDF, §1–3).
   - A Python Andersen-style analysis with **concrete evaluation in the Python interpreter** for expressions involving *external* imports and built-ins. Concrete objects are propagated through the points-to graph.
   - PoToCG gives more complete and more precise call graphs than PyCG.
   - Closeness **2** to HEAP's mechanism. Note that the package under analysis is analysed *abstractly*, while HEAP imports the package itself. PoTo attacks exactly the **built-in-type-method** category: 59% of PyCG misses (G-A1) are where HEAP admits only partial recovery.
4. **Heapster / "Heaps'n Leaks"**, ICSE'20: heap snapshots improve FlowDroid taint precision, with a mixed static fallback. SNIPPET. Closeness 2 (Android, taint).
5. **Pereira 2015**: Ruby static call graph enriched from test runs. See P6.

**Remaining delta (INFERRED).**
- A Python port that *actually imports the package under analysis*, rather than forcing or approximating execution, and walks MRO, `__wrapped__`, `__qualname__` re-exports, registries and `singledispatch`.
- Plus the per-category decomposition ("resolved at import vs only at call time") across about 12 repos.
- The decomposition is a modest empirical contribution. The mechanism is not new.

**Is the surprising prediction answered?**
- **Effectively yes, in a sister language.** Jelly recovers (88.1 − 75.9)/(100 − 75.9) ≈ **51% of statically missed runtime edges at −1.5% precision, without tests**. The inventor's prediction is ≥40% recovered at ≤3 pp precision loss. Python may differ, since decorator registries are import-time-heavy, so the number may be *higher*. But "you can recover about half of missing edges from initialisation-time state" is now a known result, not a surprise.
- HeapDL shows the same for Java with run-time heaps.

**Hostile notes.**
- Calling HEAP "graph generation at level L2, not found in a light search" (IV-2) was wrong at the mechanism level. The right search terms were "approximate interpretation", "heap snapshots" and "unsoundness", not "import-time".
- Also: importing arbitrary packages executes code. Jelly uses a sandboxed forced-execution design for that reason. HEAP's sandbox and import-failure rate must be reported like Jelly's.

**Verdict: NARROW.**
- Keep HEAP as **infrastructure**: a cheap, test-free recall boost that feeds CHAIN, HYTRACE and EFFECT.
- Cite Jelly, HeapDL and PoTo as the mechanism's prior.
- Do not headline it. A paper section titled "Python analogue of approximate interpretation, with a category decomposition" is honest.

---

## P3. HYTRACE: issue → imagined runtime path → HMM/Viterbi decoding onto the directed graph

**Closest prior.**
1. **The decoder is HMM map-matching.**
   - Newson & Krumm (2009) decode noisy GPS points onto a road graph by Viterbi, with segments as hidden states and edges as transitions (SNIPPET).
   - Yang, Chen et al., "Fast (Temporal) Path Localization on Graphs via Multiscale Viterbi Decoding", IEEE TSP 2018, decode a connected path signal on a general graph (SNIPPET).
   - Closeness **2** for the decoder: same algorithm, with "imagined step = noisy observation" swapped in.
2. **Plan-then-ground on a graph with an LLM.**
   - **RoG** (Luo et al., ICLR'24): the LLM generates relation paths as plans, which are then retrieved and instantiated on the KG.
   - Its follow-up **GCR** (graph-constrained reasoning, ICML'25) builds the KG into decoding because RoG's paths can be non-existent.
   - SNIPPET. Closeness **2**: same "LLM imagines a path, the graph corrects it" mechanism in KGQA. HYTRACE's HMM is a soft version of GCR's hard constraint.
3. **Code-graph fault chains.**
   - **SWE-Debate** (2507.23348): candidate fault-propagation chains are built by depth-limited traversal of a dependency graph from issue-matched entities.
   - **GraphLocator** (2512.22469, FSE'26): a causal issue graph from symptom to cause, grown by abductive reasoning over the repo graph. It reports +19.5% function recall.
   - **CoSIL** (2503.22424): LLM-driven call-graph search.
   - SNIPPET. Closeness **1.5–2**: the graph drives chain construction. None has the LLM write a full imagined path first and decode it structurally.
4. **Execution prediction without execution.**
   - 2506.19045 (TSE'26): static estimation of a test's execution trace from one failure log, about 90% F1. VERIFIED (abstract).
   - **ND-Slicer** (FSE'24 Distinguished): predicts input-specific dynamic dependences without running, trained on execution.
   - "LLM as an Execution Estimator" (2508.18721). SNIPPET.
   - Closeness **1.5**: amortised execution prediction is established, but not from issue prose to a repo call path.
5. Executed-trace localisers (opposite design point):
   - IssueExec 2607.17286 (VERIFIED abstract): test-driven, with traces executed.
   - DAIRA 2603.22048: dynamic analysis in the agent loop, 79.4% on SWE-bench Verified (VERIFIED abstract).
   - COCA 2503.23051: reconstructs execution paths from issue reports by static analysis for distributed-systems root-cause analysis (VERIFIED abstract; closeness 1.5).

**Remaining delta (INFERRED).**
- (i) A quantitative measure of how often a 31B model's imagined runtime path from *issue text* hits the executed FAIL_TO_PASS path, and where it breaks.
- (ii) Whether soft structural decoding corrects mostly-wrong names.
- (iii) The DPR collapse signature.

The decoder and the plan-and-ground idea are both prior. The combination is unoccupied on SWE data.

**Is the surprising prediction answered?** No. "Decoding helps even when ≥50% of path names are wrong, and the gain vanishes on DPR" is untested. RoG→GCR is indirect evidence that ungrounded LLM paths are often invalid and that graph constraints help in KGQA. That *raises* the prior for HYTRACE's direction, which lowers surprise.

**Hostile notes.**
- (a) **Many SWE-bench issues contain tracebacks.** The traceback *is* an executed path, so the "imagined path" may be copied from the issue for exactly the tasks where it is easy. The no-name stratum must also exclude issues with traceback frames, or HYTRACE measures traceback parsing.
- (b) Agentic graph search (CoSIL, GraphLocator, SWE-Debate) is the strong baseline, not BM25. A reviewer will ask why a one-shot decode beats an iterative LLM walk. The inventor's claim must be cost, or no execution with no agent loop, with this baseline included.
- (c) The 4.3–13 GPU-h LoRA is the main budget risk for what is mainly a decoder claim. Do zero-shot only unless the decoder positive control passes.

**Verdict: NARROW.**
- Present it as "map-matching for imagined execution paths", citing Newson–Krumm and RoG/GCR.
- The publishable core is the measurement plus the structural-correction ablation.
- The LoRA is optional and probably not worth the budget.

---

## P4. DECONV: identify and invert the low-pass filter in supplied code-graph embeddings, plus a ridge NL bridge

**Closest prior.**
1. **GDN, "Deconvolutional Networks on Graph Data"** (Li et al., NeurIPS'21): reconstructs input graph signals from GCN-smoothed representations with spectral inverse filters plus wavelet denoising, because the inverse is a high-frequency amplifier. Uses are feature imputation and structure generation. SNIPPET (abstract-level). Closeness **2.5** for the inversion.
2. **Blind identification of graph filters** (Segarra, Mateos, Marques & Ribeiro, IEEE TSP 2017) jointly estimates filter coefficients and input from the output. **Ye & Mateos, 2412.15133 (IEEE SPL 2025)**: blind deconvolution of graph signals *with stable recovery under graph perturbations*, alternating with eigenbasis denoising. SNIPPET. Closeness **2.5** for DECONV's system-identification step (pick g and its parameters, then invert).
3. **Reverse-process GNNs** (Park et al., ICML'24): learned inverse message-passing layers to undo oversmoothing. SNIPPET. Closeness 2.
4. Embedding whitening and Soft-ZCA (E-A6) is the inventor's own strong baseline. **Network deconvolution** (Feizi et al., 2013) removes transitive effects from observed graphs (INFERRED from memory, not re-searched). Closeness 1.5.

**Remaining delta.** The use, not the method:
- forensic identification of an *unknown* smoothing operator in a shipped embedding product;
- the CKA-to-independent-text-view selection criterion;
- co-edit retrieval;
- a ridge NL query bridge.

No retrieval application was found (RT-28, RT-29). The arXiv API query `abs:deconvolution AND abs:embeddings AND abs:graph AND abs:retrieval` returned 0.

**Is the surprising prediction answered?** Partly.
- "Deconvolving with the wrong graph hurts" is the expected behaviour in graph signal processing. Ye & Mateos treat small perturbations as a robustness problem. A degree-preserving rewire is a *large* perturbation, so degradation is close to guaranteed (INFERRED).
- The DPR arm is therefore a **sanity check, not a surprise**.
- The genuinely open prediction is the 3b sign flip: smoothing *helps* co-edit retrieval.

**Hostile notes.**
- (a) The X ≈ g(Â)H model assumes the supplied vectors are a polynomial or rational filter of per-node features over *the supplied graph*. If the product was trained (node2vec-style or GNN with nonlinearities), the identified g is a poor fit, and DECONV reduces to "a high-pass filter" ≈ Soft-ZCA.
- (b) Generality is claimed for "any graph-smoothed embedding release". That is a niche, so reviewers will see a dataset-forensics note.
- (c) The labels are the least execution-grounded of the six, as the inventor admits.

**Verdict: NARROW.**
- Run it as the cheap CPU kill it is.
- Report it as an appendix or forensic note citing GDN and blind graph deconvolution.
- Do not position it as a technique contribution.

---

## P5. EFFECT: execution-labelled effect signatures, propagated along directed edges, matched to issue-stated effects

**Closest prior.**
1. **EXCEPT, "Exception-Driven Fault Localization for APR"** (2201.00736): uses the exception's type and source, rather than spectra, to localise. Four Java exception types, 43 Defects4J faults, beats Ochiai. SNIPPET. Closeness **1.5**: same intuition ("the symptom's effect type points to the culprit"), but hand rules, no learned or traced signatures, and no issue-text matching.
2. **SBEST** (2405.00565): mines stack traces in bug reports. Finds 98.3% of fix intentions align with stack-trace exceptions and buggy methods within ≈0.34 calls. Ranks with call-graph distance. SNIPPET. Closeness **1.5**. This is also a hostile data point (see below).
3. **ExLocator** (Android framework exceptions; 25/27 located). SNIPPET. Closeness 1.5.
4. **Labels without localisation.**
   - ThrowBench 2503.04241: exception prediction from code, F1 19–38%.
   - "How robustly do LLMs understand execution semantics" 2604.16320: exception prediction collapses to about 15%.
   - JPure: dynamic side-effect and parameter-mutability inference from executions.
   - PyTation (ICSE'26): traces call arguments and types under test suites.
   - All SNIPPET. Closeness **1**: they supply pieces of the effect labeller, not the retrieval.
5. **Behaviour-aware retrieval.**
   - ExecRetrieval 2609.01865: embeddings cannot rank execution-correct code first.
   - Chistyakov et al. ICLR'17: behaviour-pattern embeddings from execution logs, for malware.
   - SNIPPET. Closeness 1.

**Remaining delta.** No prior found that:
- builds a per-function **effect signature** (exceptions, mutation, return identity, I/O, global writes) from traced test runs;
- propagates escaping effects interprocedurally;
- predicts the signature from frozen LLM states for untraced code;
- and matches it to effects parsed from issue prose for localisation.

The "issue-stated effect = traced gold effect" coverage statistic is also absent (RT-25, RT-26).

**Is the surprising prediction answered?** No. Renaming invariance of effect retrieval and the propagated > local > DPR ordering are untested.

**Hostile notes.**
- (a) **SBEST's numbers cut both ways.** When a report has an exception stack trace, the culprit is usually *in or next to the trace*. Grep or traceback parsing already wins there, and the effect channel adds nothing.
- EFFECT's marginal value lives in issues that state an effect *without* a traceback: "mutates headers", "returns None", "warns". That stratum may be small.
- The 25% coverage kill threshold should be computed **on traceback-free issues**, or the result is inflated by exceptions that grep already finds.
- (b) Exception propagation "local ∪ callees − caught" is standard exception-flow analysis (e.g., "Revisiting exception handling practices with exception flow analysis", SCAM'17; SNIPPET). Do not claim the propagation rule. Claim its use as a retrieval channel.

**Verdict: SURVIVE.** It is the least crowded of the six. Pre-register the traceback-free stratum.

---

## P6. CHAIN: per-call-site callee distributions (mass ≤1), learned from traces, and path mass

**Closest prior.**
1. **AFLGopher**, "Accelerating Directed Fuzzing via Feasibility-Aware Guidance" (Bai, Wu, Wu & Lu; arXiv 2511.10828, Nov 2025; no venue listed). VERIFIED (PDF §4).
   - Indirect-call targets are statically enumerated per site (MLTA).
   - An LSTM over the preceding call chain "outputs a softmax distribution over candidate targets for each call site".
   - It is trained on execution traces and updated online. The scores "are used as feasibility estimates for edge weighting in the call graph" that drives directed-fuzzing distance.
   - Closeness **2**: the same object (a site-normalised learned callee distribution from traces over a static candidate set, used to weight path distance). The differences are C binaries, per-program online learning with no cross-program transfer, fuzzing guidance as the downstream use, and no chain-precision evaluation.
2. **Pereira (2015)**, "Test-case-based Call Graph Construction in Dynamically Typed Programming Languages", MSc thesis, UFRGS. VERIFIED (PDF text).
   - Ruby. A static call graph is enriched by running test cases offline; "the edges of the CG [are annotated] with the likelihood that a call occurs at runtime".
   - The candidate targets at a site are ranked by test-run frequency to expose "collisions by coincidence".
   - Closeness **2**: per-site runtime-frequency weighting of a static call graph in a dynamic language. Not learned, and no transfer to unexecuted code.
3. **ML call-graph pruning, per edge**: AutoPruner (FSE'22), CGPruner, Mir et al. MSR'24 (2402.07294; precision +25% / recall −9%, conservative pruning) and **OriginPruner** (2412.09110). SNIPPET. Closeness 2. CHAIN's one-change baseline is exactly this family.
4. JIT profiles (Jikes RVM `WeightedCallTargets`, LLVM indirect-call promotion with per-site hotness) and **probabilistic pointer analysis** (Da Silva & Steffan, ASPLOS'06: a probability per points-to relation, built from edge profiles). SNIPPET. Closeness 1.5.
5. Static path-frequency estimation: Buse & Weimer, "The Road Not Taken", ICSE'09; Wu & Larus. SNIPPET. Closeness 1 (path mass, but intraprocedural).

**Remaining delta (INFERRED).**
- (i) **Cross-repo transfer**: a model trained on traced repos predicts site distributions on *unexecuted* repos. AFLGopher and Pereira need traces of the same program.
- (ii) The **chain-level hypothesis**: site-correlated errors make site-normalised mass beat per-edge classifiers on chain precision at matched recall. No prior evaluates chain precision this way.
- (iii) Mass-ranked expansion for LLM context, with the "2-hop beats 1-hop" reversal.

**Is the surprising prediction answered?** No. Both predictions are untested. The *technique* is not new: per-site softmax from traces exists.

**Hostile notes.**
- (a) **A softmax over candidates is not automatically better than per-edge BCE when most sites are monomorphic.** For |C(s)| = 1, site normalisation is a no-op. The chain-precision gain can only come from polymorphic, MRO and registry sites. Report the share of chain errors that originate at |C(s)| > 1 sites *before* training anything. If it is under about 30%, 1b is dead on arrival (INFERRED).
- (b) A trace-frequency target confounds "rarely taken in tests" with "false". CHAIN mass will down-weight real but untested edges, which is a known complaint against dynamic ground truth (Mir et al. note recall loss).

**Verdict: NARROW.**
- Retitle it as a test of *why chains collapse*: site-correlated vs independent errors.
- Cite AFLGopher and Pereira for the per-site distribution itself.
- The inventor already flagged it as "most crowded". This confirms that, and adds two closer priors than the inventor found.

---

## T7–T12 (brief)

| # | Technique | Closest prior (closeness) | Verdict |
|---|---|---|---|
| T7 | ASYM (source/target embeddings, InfoNCE on runtime reachability) | Flow2Vec (OOPSLA'20, asymmetric transitivity over value flow) (2); Callee (2111.01415, contrastive site↔callee matching) (2); kTrans indirect-call ranking; order embeddings (CodeScholar) (1.5). All SNIPPET | NARROW: a small-model instance of LATENT's U/V heads. Keep it as an ablation |
| T8 | SITE (LLM receiver typing + static MRO closure) | Venkatesh et al., EMSE'25 / 2410.00603, **explicitly propose** using LLM-inferred types to enrich static call-graph construction as future work (2). SEA, "Semantic-Enhanced Indirect Call Analysis with LLMs" (2408.04344, C) (2). PoTo's concrete evaluation covers part of the same sites (1.5). SNIPPET | SURVIVE-narrow: the hybrid is proposed but not evaluated for Python. Cheap zero-shot kill. Overlaps HEAP on built-ins |
| T9 | TRACEBACK serialisation | Graph-encoding format studies, e.g. "Talk like a Graph" (ICLR'24, from memory; INFERRED) and G-A12/A14 (2) | NARROW/weak, as the inventor says |
| T10 | KILLVEC (mutation kill-vector coupling) | Seshat and WITNESS (predictive kill matrices) (2); Mut4T (mutation-based test-to-code traceability, SANER'25) (1.5); IFG (internal) | NARROW: keep it as an edge type only |
| T11 | RESIDUAL (show only non-inferable edges) | No direct prior found (not deeply searched). Depends on LATENT | Untested. Survives only if LATENT survives |
| T12 | EXECCARD (traced I/O cards as node text) | NExT, E.T. 2503.05703, SemCoder, TRACED (traces as training or context) (1.5–2) | NARROW, plus the inventor's own inference-time-trace limitation |

---

## Cross-cutting findings for the inventor

1. **The "builder stack" (HEAP + CHAIN) is mostly prior art in other languages.**
   - HEAP ≈ Jelly approximate interpretation (JS) + HeapDL (Java) + PoTo concrete evaluation (Python, external code only).
   - CHAIN ≈ AFLGopher per-site softmax + Pereira runtime-likelihood annotation.
   - The only novel parts are a Python import-time port, cross-repo transfer, and the chain-precision hypothesis.
   - Recommendation (INFERRED): do not make the paper "a new graph builder".
2. **The defensible paper is P1 + P5, with P3 as a measured ablation.**
   - LATENT (runtime-only decodability) and EFFECT (effect-channel localisation, with a coverage statistic) have no closeness-3 or 2.5 prior.
   - Both are cheap: one shared forward pass.
   - That fits the cheapest-coherent plan (P1 + P2 + P5 + P6, about 10–13 GPU-h) with P2/P6 demoted to infrastructure.
3. **Two strata must be pre-registered to avoid shortcut wins.**
   - Traceback-free issues, for HYTRACE and EFFECT. SBEST shows traceback proximity alone localises most exception bugs.
   - Polymorphic sites (|C(s)| > 1), for CHAIN.
4. **Baselines reviewers will demand.**
   - Jelly-style approximate interpretation ported to Python (for HEAP).
   - AFLGopher-style per-site softmax *without* cross-repo features (for CHAIN).
   - GraphLocator or CoSIL agentic graph search (for HYTRACE).
   - GDN inverse filter (for DECONV).
   - The Troshin/Anand pairwise edge-probe protocol (for LATENT).

---

## Search log

Tools:
- **WS**: WebSearch (standard or extended).
- **WF**: WebFetch.
- **OR**: OpenReview `api2.openreview.net/notes/search`. Per-note `/notes?id=` returned 403; titles came from search-hit content.
- **AX**: the arXiv export API.
- **PDF**: a PDF fetched and its text extracted locally with pypdf.

| ID | Tool | Query or target | Result used |
|---|---|---|---|
| RT-1 | WS | probing code language model hidden states call graph edges linear probe | No call-graph probe. Troshin 2202.08975; 2212.10017; 2606.14530 (correctness probe) |
| RT-2 | WS | code models data flow probing / structural probe | Troshin & Chirkova; Anand et al. ACL'24 2406.11930 (DFG-edge pair probes; larger models encode less) |
| RT-3 | WS | probe LLM residual stream PDG reachability code 2025 2026 | 2605.25310 (tool-call), 2607.05188 (agent-state probes), CodeCircuit 2602.07080. No PDG probe |
| RT-4 | WF | arxiv 2609.24352 | Few-shot in-context world representations (graph tracking). Not code. Closeness 1 |
| RT-5 | WS ext | do LLMs encode call graph internally probing | Only external-graph agents (LocAgent, CodexGraph) |
| RT-6 | OR | "probe call graph code", "runtime call graph LLM", "code graph probe" | AutoPruner; 2605.25310 (CoRR + ARR Aug 2026); ICLR'26 reviews (CWE linear probes, sub. 2964); no code call-graph probe |
| RT-7 | OR | linear probe code model data flow; execution trace probing hidden states code; probing code LLM execution semantics | "How should code LLMs learn execution?" (COLM'26 WS); TRACE-SWE (ICLR'27 sub.); no graph probe |
| RT-8 | OR | dynamic call graph Python; hybrid call graph Python; Python call graph introspection | Jarvis, DyPyBench, **PoTo (ECOOP'25)**, Scalpel, Indirection-bounded CG (ECOOP'24). No import-heap builder |
| RT-9 | OR | call graph pruning machine learning; call site target prediction; probabilistic call graph | AutoPruner; **Mir et al. (MSR'24)**; **OriginPruner** 2412.09110 |
| RT-10 | OR | predict execution trace fault localization; simulated execution issue localization; hypothetical execution trace LLM | SweRank; classic SBFL; nothing on imagined paths |
| RT-11 | WS ext | Python call graph runtime introspection MRO hybrid | PoTo; DyPyBench; **Pereira 2015** (static CG + test likelihood) |
| RT-12 | WS | Python static CG recall decorators super empirical | DyPyBench (PyCG vs DynaPyt); Kummita'21; Sui et al. |
| RT-13 | WS | Jarvis Python call graph | 2305.05949 (precision +84%, recall +20% vs PyCG) |
| RT-14 | WS ext | call graph Python import time / live objects | None (absence) |
| RT-15 | WF + PDF | PoTo ECOOP'25 PDF | VERIFIED: concrete evaluation for *external* imports and built-ins; PoToCG > PyCG |
| RT-16 | WS ×2 | "Heaps don't lie" / "Heaps'n Leaks" | **HeapDL**: 99.5% vs TamiFlex 76.9% (VERIFIED abstract); Heapster ICSE'20 |
| RT-17 | AX | `all:probing AND all:"call graph"`; `abs:"heap snapshot"`; `abs:"import time" AND python`; `abs:"call graph" AND abs:"hidden states"`; and others | 4, 3, 0 and 0 hits. Only 2605.25310 and 2605.17246 (fidelity probes, unrelated) for probes. HeapDL is the only heap-snapshot call-graph paper |
| RT-18 | WS ext | heap snapshot static analysis Python/JS after module loading | HeapDL; "Shooting from the Heap" ISSTA'18; no Python |
| RT-19 | WS + PDF | "Reducing Static Analysis Unsoundness with Approximate Interpretation" | **VERIFIED: recall 75.9 → 88.1%, precision −1.5%, initialisation code, no tests (Jelly)** |
| RT-20 | WS ext | approximate interpretation Python 2025 2026 | No Python port found (absence) |
| RT-21 | WS | Mir et al. ML call graph pruning MSR'24 | Venue is MSR'24, not ICSE; NYXCorpus; OriginPruner |
| RT-22 | WS ext + WF + PDF | per-call-site learned target distribution, dynamic labels | **AFLGopher** 2511.10828. The WF abstract summary misreported it; the PDF §4 VERIFIED a per-site softmax from traces used for edge weighting. Also iResolveX, NeuCall, AttnCall, Callee (binaries) |
| RT-23 | WS | Buse & Weimer "The Road Not Taken" | ICSE'09 poster; Wu & Larus |
| RT-24 | WS | probabilistic pointer analysis | Da Silva & Steffan ASPLOS'06 |
| RT-25 | PDF | Pereira 2015 UFRGS thesis | VERIFIED: Ruby; edges annotated with runtime likelihood; candidate ranking |
| RT-26 | WS | SWE-Debate fault propagation traces | 2507.23348; GraphLocator 2512.22469 |
| RT-27 | WS ext | LLM predicts call chain from issue then aligns to call graph | CoSIL 2503.22424; ConFL 2608.02974; AutoFL; none predict-then-align |
| RT-28 | WS | HMM Viterbi align predicted function sequence to call graph | None for code (absence) |
| RT-29 | WF | 2506.19045; 2607.17286 | VERIFIED abstracts: static trace estimation from failure log (TSE'26, about 90% F1); IssueExec executes tests |
| RT-30 | WS | ND-Slicer predictive dynamic slicing | FSE'24 Distinguished; RecovSlicing 2508.18721; Blended Analysis FSE'25 |
| RT-31 | WS ext | LLM simulated/imagined execution localization | CoSIL, SWE-Adept, SHERLOC 2606.24820 (WF: hypothesis-driven, mechanism unspecified) |
| RT-32 | WS | HMM map matching program graph | Newson & Krumm; route map-matching 2607.22543; nothing in code |
| RT-33 | WS ×2 | RoG; GraphLocator | RoG ICLR'24 plan-then-ground; GCR follow-up; GraphLocator FSE'26 |
| RT-34 | AX | `abs:"execution path" AND issue AND localization AND LLM` | COCA 2503.23051, DAIRA 2603.22048, EffiHolmes 2608.03558 (VERIFIED abstracts for the first two) |
| RT-35 | WS ×3 | blind identification of graph filters; reverse-process GNN; GNN inversion attacks | Segarra TSP'17; Park ICML'24; GDN NeurIPS'21; GRAIN; MC-GRA 2606.08067 |
| RT-36 | WS | graph deconvolution embeddings retrieval query encoder | GDN only; no retrieval use (absence) |
| RT-37 | OR | embedding deconvolution graph smoothing | **Ye & Mateos, blind deconvolution robust to graph perturbations (IEEE SPL'25)**; KQGC |
| RT-38 | OR | imagined execution path localization; issue localization execution path graph decoding | **Multiscale Viterbi path localisation on graphs (IEEE TSP'18)**; CoSIL |
| RT-39 | WS ×2 | exception propagation bug localization; behavioural side-effect localization | EXCEPT 2201.00736; **SBEST** 2405.00565; ExLocator; MBFL interference mutants |
| RT-40 | WS ×2 | execution-derived behaviour embeddings; LLM exception prediction | ExecRetrieval 2609.01865; Chistyakov ICLR'17; ThrowBench; 2604.16320; CatchAll |
| RT-41 | WS ×2 | bug report symptom taxonomy; Python side-effect inference dynamic | Bettenburg'07, Zimmermann; JPure; PyTation ICSE'26 |
| RT-42 | OR | exception propagation localization LLM; effect signature code retrieval; weighted call graph LLM context retrieval | Exception-flow analysis SCAM'17; nothing on effect retrieval |
| RT-43 | WS ext | probing hidden states cross-file dependency repository 2026 | None; "Linear Probes Detect Task Format" TrustNLP'26 (confound warning) |
| RT-44 | WS ×3 | Jin & Rinard; AST-Probe; probe dynamic dispatch / inheritance | ICML'24 execution-label probes; ASE'22 AST-Probe; ICSE'25 NIER scope probes; nothing on dispatch |
| RT-45 | AX | id_list 2503.23051, 2603.22048, 2605.17246, 2605.25310, 2601.17593 | Abstracts VERIFIED (quoted above) |
| RT-46 | WS ×2 | LLM type inference to resolve Python call sites | Venkatesh et al. EMSE'25 (hybrid proposed); Paderborn theses; SEA 2408.04344 |
| RT-47 | WS | directed asymmetric code embeddings | Flow2Vec, Callee, kTrans, CodeScholar |
| RT-48 | WS ×2 | kill-matrix coupling; execution traces as LLM context | Seshat, WITNESS, Mut4T; E.T., NExT, SemCoder, TRACED, CoCoNUT |

**Not run (gaps).**
- Semantic Scholar API: skipped for budget; OpenReview plus the arXiv API covered the ML and SE venues.
- ICLR 2027 full-text sweep beyond the search API.
- Python-specific searches for "Pyright/Jedi interpreter-mode call graph".

An unpublished Python port of approximate interpretation, or a 2026–27 submission probing code-LLM call graphs, cannot be excluded. The OpenReview search API surfaces reviews, not all submissions.
