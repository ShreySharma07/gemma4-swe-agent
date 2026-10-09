# Code-graph generation and parsing for LLM code comprehension: anomalies scout (as of 9 Oct 2026)

**Scope.** How repository, call, dependency and data-flow graphs are generated (static, dynamic, LLM-built, hybrid) and parsed or serialised for LLMs. The goal is a list of **anomalies**: surprising, unexplained or contradictory numbers that a new technique could exploit.

**Labels.**
- **VERIFIED**: I read the claim in the primary source text (alphaXiv full text or PDF pages) this session.
- **SNIPPET**: seen only in an abstract or a search result.
- **INFERRED**: my own reasoning.
- **[PR]** = peer-reviewed. **[pre]** = preprint. Status is taken from the venue line in the paper, or from a citation in another verified paper where noted.

**Not repeated from existing notes** (`code_graphs_swe.md`, `general_graph_reasoning.md`): CodeAnchor's inverse-only links, ARISE's spurious-vs-missing asymmetry, RepoGraph's 1-hop vs 2-hop result, BM25-wins-at-scale, A²Agent, GraphCert, and the others listed there. Where a new finding contradicts one of them, I flag it.

The anomalies are numbered **A1–A22** across sections. A one-line index sits at the end of the last section.

---

## Q1. Static vs dynamic graphs: how much do Python static call graphs miss or hallucinate, and how do dynamic features affect LLMs?

### Takeaway
Python static call graphs look near-perfect on micro-benchmarks (about 90% of cases sound) but recover only about **half of runtime edges** on real projects. The misses come mostly from mundane causes (built-in-type methods, `super()`, name resolution), not exotic metaprogramming. Edge-level precision also hides chain-level collapse: about 0.99 precision per edge becomes 0.19 when scored along call chains. LLMs themselves are near-blind to runtime data flow (1–2% accuracy). They also over-extend along the *static* transitive call graph.

### Cited Findings

- **A1. PyCG recovers only 49% of runtime call edges on real projects. 92% of the misses are a missing callee, and 59% of missing callees are methods of built-in types.** VERIFIED [PR, FSE 2024]
  - Setup: 39 projects where both tools ran. Only **49%** of DynaPyt's dynamic edges appear in PyCG's static graph, and only **48.5%** of dynamic callees appear at all.
  - For **92%** of missing edges the callee is absent from the static graph. Calls on built-in types (e.g. `"abc".strip()`) account for **59%** of missing callees. `super().method()` calls are also dropped. Fully-qualified-name mismatches (wrappers, re-exports) account for about 12%.
  - PyCG failed outright on **11 of 50** projects (6 timeouts at 6 h, 3 out-of-memory at 60 GB, 2 crashes). The dynamic graph was sometimes *faster* to produce than the static one.
  - **Why surprising:** graph-RAG and LocAgent-style systems assume static parsing gives a sound superset of real calls. In fact half of the executed edges are absent. The causes are boring and fixable (built-ins, MRO, naming), not the "dynamic features" usually blamed. This is the primary source behind the 49% figure in our HQ notes (cited there via NameRTS).
  - Source: [DyPyBench, arXiv 2403.00539](https://arxiv.org/abs/2403.00539)

- **A2. Micro-benchmarks are saturated and do not predict real-world recall.** VERIFIED [PR for PyCG; pre for the others]
  - On its own 112-program micro-benchmark, PyCG is complete on **111/112** cases and sound on **103/112**.
  - A 2026 bytecode-level framework, CPyGraph, matches PyCG *exactly* on that benchmark (111 / 103). It also reports **100% recall and 98.9% call-graph precision** on its own 201-program PYGBench.
  - Yet the same class of tool reaches 49% runtime-edge recall on real code (A1).
  - CPyGraph's authors note that their benchmark and tool "were developed together".
  - **Why surprising:** the field reports call-graph quality on feature-isolating micro-benchmarks, where tools are near-ceiling. The real-world gap (about 90% sound cases vs about 49% of runtime edges) is never what downstream LLM papers measure.
  - Sources: [CPyGraph, arXiv 2609.25083](https://arxiv.org/abs/2609.25083) (TOSEM-format preprint); [PyCG, ICSE 2021, arXiv 2103.00587](https://arxiv.org/abs/2103.00587)

- **A3. Errors compound along chains: about 0.99 edge precision becomes 0.19 chain precision.** VERIFIED [pre; arXiv v5]
  - On 6 real applications (20,085 manually built ground-truth call relations, 6 person-months), PyCG reaches **0.99 precision / 0.76 recall** when scored per edge on the application only.
  - When scored as **call chains from entry functions** over the whole program (application plus libraries), PyCG precision drops to **0.19** (recall 0.48). Jarvis drops to **0.35 / 0.60**.
  - The authors' explanation: one false-positive edge makes every later edge on the chain a false positive.
  - PyCG whole-program analysis exceeded **24 h** on 4 of 6 projects and ran out of memory or hit recursion errors on the other 2.
  - **Why surprising:** multi-hop graph expansion assumes per-edge precision carries over. Chain-level precision is far lower. This may give a mechanism for "RepoGraph 2-hop worse than 1-hop" and "spurious edges hurt more than missing ones (ARISE)": spurious edges multiply under traversal. (The mechanism link is INFERRED.)
  - Source: [Jarvis, arXiv 2305.05949](https://arxiv.org/abs/2305.05949)

- **A4. LLMs cannot follow runtime data flow (1–2% accuracy), and the larger Gemma-4 is no better than the 4B.** VERIFIED [pre]
  - Benchmark: SWE-FLUX, 480 execution-grounded questions over 12 real Python repositories, with tool-assisted exploration (25 tool calls, 32k context for open models).
  - Best overall: **GPT-5.4 at 37.71%**. **Gemma-4-31B 19.17% vs Gemma-4-4B 17.29%.**
  - Runtime data flow (def–use): **1.69% (single test) / 1.82% (multi-test)** averaged across models. GPT-5.4 scores 6.78% and 0.00%.
  - Inter-procedural control flow: **16.4% / 8.0%**. Intra-procedural: 53.9%.
  - Two failure modes:
    - **Over-extension along the static transitive call graph.** On one SymPy item a model predicted **260 def–use pairs vs 44 actual**. On Python-Control it reported 41 covered functions vs 15.
    - **Truncation** when execution passes through `__getattr__`/`__setattr__`, generators or **decorators**.
  - **Why surprising:**
    - A larger model on the same family does not help; the authors say "scale alone does not fully explain performance".
    - Models use static structure as a *proxy* for runtime behaviour and overshoot. That is the opposite of the usual worry that LLMs "miss" structure.
    - Directly relevant to us: Gemma-4-31B is our base model.
  - Source: [SWE-FLUX, arXiv 2609.28449](https://arxiv.org/abs/2609.28449)

- **A5. Coding agents are near-perfect on synthetic call-graph traversal at scale, because they script around it.** VERIFIED [pre, 7 Oct 2026]
  - Benchmark: CABRA, 6,840 tasks generated from directed acyclic call graphs, with up to N=200 edit or distractor functions.
  - LLMs without tools degrade with N. Agents stay near 100% by writing `ast`/`grep`/`exec()` "meta-scripts". GPT-5.4-Mini inside a harness beats GPT-5.5 without tools for N>100.
  - On SWE-bench Verified traces, the number of *understanding* tool calls (READ+ANALYZE) predicts failure (point-biserial **r = −0.200**) better than lines edited (−0.159).
  - On a code-merging task, **removing variable renaming boosted accuracy the most**. The authors conclude agents "over-rely on surface-level naming".
  - **Why surprising:** graph traversal is a solved problem *if the agent can compute it*. The hard part is semantic alignment under renaming, which matches the "localise by names" finding in our HQ notes. Graph tools add value only where scripts cannot reach (semantic equivalence and runtime binding).
  - Source: [CABRA, arXiv 2610.10610](https://arxiv.org/abs/2610.10610)

- **A6. Syntax-only graphs leave call targets unresolved; resolving them on demand with a language server is worth about 7 pp.** VERIFIED [pre]
  - SemNav keeps AST relation sites with unresolved targets (e.g. `obj.run()`) and resolves them online with Pyright (go-to-definition / find-references).
  - With Gemma-4-E4B on SWE-bench Lite, removing online resolution costs **−7.00 pp File Hit@10** and −4.33 pp Function Hit@10. Full SemNav raises File Hit@10 from 68.33% to 82.67%.
  - **Why surprising:** most code-graph systems (RepoGraph, CodexGraph, LocAgent) build the *whole* graph eagerly from tree-sitter or AST and silently drop unresolved edges. Lazy, query-time resolution of only the edges the agent asks about beats an eager complete graph, even at 4B scale.
  - Source: [SemNav, arXiv 2609.31176](https://arxiv.org/abs/2609.31176)

### Inferences
- The 49% recall in A1 is dominated by fixable categories: built-ins (59% of missing callees), `super()`, and qualified-name mismatches (about 12%). A cheap repair pass (built-in method stubs plus MRO resolution plus name canonicalisation) might recover a large share of missing edges without dynamic analysis. Nobody reports doing this for an LLM agent. (INFERRED)
- A3 together with A4 suggests a single mechanism: structure is helpful at depth 1. Beyond that, both static tools and LLMs over-propagate (spurious chains; 260 vs 44 def–use pairs). A depth-dependent confidence decay on graph edges is a natural exploit. (INFERRED)

### Gaps
- No real-world recall numbers found for pyan, tree-sitter "calls" heuristics, Joern CPG on Python, or CodeQL Python call graphs vs runtime. All numbers above are for PyCG and Jarvis only.
- No paper found that measures LLM *localisation* accuracy split by whether the gold function is reached only through a decorator, registry, monkeypatch or dynamic dispatch. SWE-FLUX gives only qualitative evidence (truncation at decorators and `__getattr__`).

---

## Q2. LLM-generated graphs (LLM as parser, RPG-Encoder, CodexGraph, partial graphs): accuracy vs static tools, cost, failure modes

### Takeaway
There is a clear contradiction. When graded *as graphs*, LLMs build worse call graphs than PyCG. When graded *downstream*, LLM-built partial graphs beat static global graphs and run 7.4× faster. Graph-construction accuracy appears to be the wrong objective. Relevance-filtered, on-demand, entry-anchored subgraphs win.

### Cited Findings

- **A7. LLMs are worse than PyCG at call graphs but better than static tools at type inference.** VERIFIED [pre; journal submission]
  - Setup: 24 LLMs on PyCG's micro-benchmark extended to 126 cases.
  - Complete / sound cases: **PyCG 84.9% / 87.3%**; best LLM mistral-large-123B 60.3% / 62.7%; gpt-4o 50.0% / 59.5%; **gemma2-27B 23.0% / 34.1%**.
  - Degenerate output: gemma2-9B is "complete" on 120 cases but sound on only 4, and gemma2-2B is complete on 117 and sound on 1. They emit near-empty graphs, which trivially have no false positives.
  - On type inference, mistral-large beats HeaderGen on decorated functions (return type **86.2% vs 37.9%**). gpt-4o gets 806/860 annotations correct.
  - The authors' explanation: types are local and pattern-learnable, while call graphs need global closure.
  - **Why surprising:** "LLM as parser" pipelines (DyCoder, RPG-Encoder semantic lifting) assume LLMs extract dependencies acceptably. In fact small models produce mostly *empty* graphs, so their errors are omissions, not hallucinations. This suggests a hybrid: let the LLM do type inference and let a static resolver close the graph over those types. (The hybrid is INFERRED.)
  - Source: [Venkatesh et al., arXiv 2410.00603](https://arxiv.org/abs/2410.00603)

- **A8. Yet an LLM-built partial dependency graph beats a static global graph downstream, 7.4× faster.** VERIFIED [PR, ASE 2026]
  - How it works: DyRetriever lets the LLM pick entry functions, then asks it to *extract callees* (only two relation types: call and lazy import) of each visited function on the fly. Hallucinated targets are filtered by name-matching against the repository.
  - Results: DyCoder improves Pass@1 by **25.63% (CoderEval) and 59.73% (DevEval)** relative to RAG baselines. Total time is **148 s vs 1,096 s** for RepoScope, which builds a static global graph with PyType and tree-sitter (875 s indexing).
  - Failure modes reported: the LLM **invents non-existent functions** and labels stdlib APIs (e.g. `re.match`) as repository dependencies.
  - Static-rule failure reported: RepoScope's import-based traversal cannot reach **same-file dependencies**, because they are not imported.
  - **Why surprising:** this contradicts A7. A less accurate graph builder wins because it builds only the task-relevant region. This points to *selective construction* as the lever rather than parse fidelity.
  - Source: [DyCoder / DyRetriever, arXiv 2608.01927](https://arxiv.org/abs/2608.01927)

- **A9. CodexGraph's LLM-written graph queries are the bottleneck, and its schema has no call edges.** VERIFIED [PR, NAACL 2025 per citation in SemNav]
  - The schema has CONTAINS, INHERITS, HAS_METHOD, HAS_FIELD and USES. Adding function-call edges is listed as *future work*.
  - Removing the natural-language-to-Cypher "translation agent" drops GPT-4o CrossCodeEval exact match from **27.90 to 8.30**. Qwen2-72B drops from 5.00 to 0.50.
  - On SWE-bench Lite, Qwen2-72B + CodexGraph resolves **1.95%** vs 9.34% for AutoCodeRover.
  - Indexing ran out of memory on **43 SymPy instances**, which were excluded.
  - Token cost is 102k per SWE-bench task vs 76k for AutoCodeRover. The agent sometimes "fails to generate executable graph queries".
  - **Why surprising:** a call-edge-free graph is competitive (GPT-4o 22.96% = AutoCodeRover), so call edges were not the binding constraint. Query generation is, and it collapses at ≤72B. Expressive query languages are a poor interface for mid-size models.
  - Source: [CodexGraph, arXiv 2408.03910](https://arxiv.org/abs/2408.03910)

- **A10. RPG-Encoder: dependency edges matter much less for stronger models, and its efficiency advantage reverses for the strongest one.** VERIFIED [pre]
  - The RPG (Repository Planning Graph) is built by GPT-4o "semantic lifting" (feature phrases per function, an LLM-induced functional hierarchy) plus AST dependency edges.
  - Ablation on SWE-bench Live, Function Acc@1 with dependency edges removed:
    - GPT-4o: 50.5 → **44.8** (−5.7 pp).
    - GPT-4.1: 64.7 → **63.7** (−1.0 pp).
  - Removing semantic features costs more in both cases (−7.4 and −7.3 pp).
  - With Claude-4.5 the efficiency advantage reverses: RPG-Encoder takes **16.75 steps and $1.32** per task vs 7.54 steps and $0.88 for LocAgent (Table 15). Its headline 93.7% Acc@5 is with Claude-4.5.
  - Incremental updates cost **95.7% fewer tokens** (633K vs 14.7M), with only small accuracy changes (function Acc@1 −2.7 to −3.3 pp, Acc@5 +0.9 to +1.6 pp).
  - **Why surprising:** dependency edges, the "graph" part, show a diminishing marginal value as model strength rises. LLM-written summaries carry more signal than the edges.
  - Source: [RPG-Encoder, arXiv 2602.02084](https://arxiv.org/abs/2602.02084)

- **A11. Graph-based retrieval underperforms BM25 on dependency recall; a feature-guided local graph wins with 86% fewer edges.** VERIFIED [pre]
  - Benchmark: DevEval dependency recall at 15 (DR@15).
  - **RepoGraph 0.127** vs **BM25 0.296**, UniXcoder 0.415, CodexGraph 0.430, FeatLens 0.501.
  - Downstream Pass@1 (DeepSeek-V3.2): **RepoGraph 23.76% vs BM25 32.17%**.
  - FeatLens builds a *task-specific* seed graph from an LLM feature index ($0.21–0.30 per repository), expands one hop, then filters with personalised PageRank. It uses **61% fewer nodes and 86.2% fewer edges** than CodexGraph and no LLM tokens at retrieval time.
  - In its worked example, one-hop expansion raises dependency recall from 66.7% to 100% but drops precision from 26.7% to 17.1%; filtering restores precision to 37.5%.
  - **Why surprising:** a published ICLR'25 graph method loses to BM25 by more than 2× on recall. The authors attribute this to search-term or entry-point quality. That matches the HQ finding that only 21% of issues name the gold node: the graph is irrelevant if the entry is wrong.
  - Source: [FeatLens, arXiv 2609.26480](https://arxiv.org/abs/2609.26480)

### Inferences
- A7 vs A8 vs A11 show a consistent pattern. **Graph precision and recall (the construction objective) is not what downstream tasks reward. Entry anchoring plus a bounded, filtered neighbourhood is.** That supports treating the competition's noisy graph as a *candidate generator to be filtered* rather than a ground truth to be traversed. (INFERRED)
- The LLM-as-parser failure modes are specific and checkable: invented names, stdlib leakage, empty output at small scale. A name-existence filter plus a stdlib allow-list removes the first two. For Gemma at 4-bit, expect the empty-graph mode of A7. (INFERRED)

### Gaps
- I did not read RepoUnderstander's or KGCompass's construction-accuracy numbers.
- I found no paper that directly measures LLM-extracted *call-edge* precision and recall on *real* repositories (as opposed to micro-benchmarks).

---

## Q3. Granularity, schema, serialisation: what is the measured effect?

### Takeaway
Serialisation and presentation of a given graph move results as much as its content does. The examples:
- Visual rendering beats text rendering of the same graph.
- Graph-only prompts beat graph+source (context dilution).
- Tree-indented subgraphs beat adjacency lists.
- Attaching node attributes hurts.

Edge types are not equal. Containment-only adds almost nothing over no graph. Call, import and inherit edges plus multi-hop traversal carry LocAgent's gain, which contradicts RepoGraph's 1-hop result.

### Cited Findings

- **A12. The same graph rendered as an image beats it rendered as text, and text-only graphs barely beat no graph.** VERIFIED [pre]
  - Setup: DualView ablation, Kimi-K2.5 in OpenCode, SWE-bench Pro subset, four views (module coupling, function call, class hierarchy, program dependence).
  - Resolved: base **69**, textual graph **71**, visual graph **78**, both **81**.
  - The visual variant is also cheaper ($0.50 vs $0.57) and uses fewer steps (47.2 vs 51.1).
  - Across agents, graph calls average **1.8–3.7 per instance** and reduce total steps by 4.6–8.5. On SWE-bench Verified, Gemini-3-Flash goes from **75.8% to 79.8% at half the cost** ($0.36 → $0.18).
  - **Why surprising:**
    - Identical information gives a 7-resolve gap depending only on modality, so text serialisation of graphs is lossy *for the model*.
    - Agents did call the optional graph tools (1.8–3.7 per instance) when given an adaptive "when to use which view" protocol. That is in tension with CodeAnchor's "agents ignore optional graph tools". The protocol may be the difference. (INFERRED)
  - Caveat: small subset, single runs.
  - Source: [DualView, arXiv 2607.01929](https://arxiv.org/abs/2607.01929)

- **A13. Context dilution: graph-only prompts beat graph plus source code.** VERIFIED [pre]
  - Setup: RepBench, 107 PrimeVul C/C++ functions, Joern-extracted graphs, one model (qwen3.6-plus).
  - Results:
    - **AST+PDG 83.2%** vs **raw source 53.5%**.
    - Graph-only family 74.4% vs source+graph 69.4%; the latter uses 2.2× the tokens.
    - pdg+source 65.7% is *below* PDG alone at 69.0%.
  - The authors show the drop occurs without truncation.
  - **Why surprising:** every repository agent appends source to structure on the assumption that more evidence is monotonically better. Here adding the source *removes* the graph's benefit.
  - Caveats: vulnerability classification, not localisation; C/C++; small n.
  - Source: [RepBench, arXiv 2606.25356](https://arxiv.org/abs/2606.25356)

- **A14. LocAgent: containment edges alone are worth about nothing, and multi-hop beats 1-hop. Attribute-rich serialisation hurts.** VERIFIED [PR, ACL 2025 per citation in SemNav]
  - Ablation with fine-tuned Qwen2.5-7B on SWE-bench Lite, Function Acc@10:
    - Full: **71.53**.
    - No graph traversal: **66.06**.
    - **Containment edges only: 66.42** (≈ no graph).
    - **Hops fixed to 1: 66.79**.
    - No BM25 index: 60.22. No entity search: 53.28.
  - Format study on 37 hard instances: tree-indented subgraph with full entity IDs is best. **"row with entity attributes" consistently underperforms plain "row"**, which the authors attribute to noise. The integer-ID "incident" encoding helps file level but hurts module and function level.
  - **Why surprising:**
    - It contradicts RepoGraph's "2-hop worse than 1-hop" (in our notes). The difference is likely that LocAgent's agent *chooses* hops and edge types (direction and type filters) while RepoGraph dumps an ego-graph. (INFERRED)
    - The competition's mixed "calls" label is partly containment, which by this ablation carries almost no signal. (INFERRED)
    - The lexical index (−11 to −18 pp) matters more than the graph (−5 pp), which again points to the entry-point bottleneck.
  - Source: [LocAgent, arXiv 2503.09089](https://arxiv.org/abs/2503.09089)

- **A15. Semantic cards beat raw source at 48% less working context, at 4B scale.** VERIFIED [pre]
  - SemNav's issue-conditioned "Semantic Cards" (LLM summary plus supporting snippets) cut the localisation agent's working context by **48.2%** vs full source.
  - Removing them costs **−6.00 pp File Hit@10** with Gemma-4-E4B.
  - Removing the candidate workspace collapses output from 9.96 to 1.50 functions and costs −19.2 pp Function Recall@10.
  - Downstream, Qwen3.5-35B repair resolves **52.33%** with SemNav context vs 44.00% for the best baseline (LocAgent) and 67.00% with oracle files.
  - **Why surprising:** this agrees with A10 (summaries > edges) and A13 (less raw source is better). At small scale, *summarised* nodes plus resolved edges beat raw nodes.
  - Source: [SemNav, arXiv 2609.31176](https://arxiv.org/abs/2609.31176)

### Inferences
- Four independent papers (A12–A15) imply that graph serialisation is under-optimised. Three facts stand out: the same content in a different form moved resolves 71 → 78, removing source raised accuracy, and attributes hurt. For a text-only Gemma, the actionable version is a compact, tree-indented, direction-labelled neighbourhood with summarised nodes, and *no* bulk source. (INFERRED)
- A14's "containment ≈ nothing" result plus our 100%-reciprocal mixed-label graph suggests that re-typing edges (separating containment and test edges from true calls) may matter more than adding direction alone. This is cheap to test. (INFERRED)

### Gaps
- No study found that ablates *edge direction* (directed vs symmetrised) at a fixed edge set for LLM localisation. That remains open, as noted in our Map 1.
- Visual graph rendering (A12) needs a multimodal model. Whether Gemma-4-31B vision can read rendered call graphs is untested.

---

## Q4. Incompleteness, noise, staleness and repair: which edges matter, and does construction quality correlate with downstream gains?

### Takeaway
The correlation is weak or inverted in most head-to-heads:
- LLM partial graphs (less accurate) beat static global graphs (A8).
- A call-edge-free schema is competitive (A9).
- Dependency-edge ablations shrink with model strength (A10).
- Containment edges add about nothing (A14).

The one place where *validity* (as opposed to completeness) matters strongly is **staleness**. Outdated context is actively harmful (worse than none along the wrong-state dimension), and the presence of correct evidence rescues the model more than its ranking does.

### Cited Findings

- **A16. Stale context is worse than no context: it redirects failures into confident wrong-state calls.** VERIFIED [pre]
  - Setup: 17 helper-signature changes from click, flask, httpx, requests and rich.
  - Stale-only retrieval produces stale-signature calls on **15/17 (Qwen2.5-Coder-7B)** and **13/17 (gpt-4.1-mini)** samples, i.e. +88.2 and +76.5 pp over current-only retrieval.
  - No retrieval: **0/17 stale and 1/17 pass**; the models simply fail.
  - Adding the current snippet alongside the stale one drops the stale rate to 23.5–29.4%. **Rank order has 0.0 pp effect.**
  - 22 of the 28 stale outputs follow the old signature by *omitting a newly added argument*.
  - **Why surprising:** retrieval-noise work assumes bad context is at worst neutral. A plausible but stale node in a graph is not neutral. For graph construction this means the offline index's freshness matters, and coverage of the correct node matters more than ranking.
  - Caveat: n=17, static regex oracles.
  - Source: [Stale repository context, arXiv 2605.14478](https://arxiv.org/abs/2605.14478)

- **A17. Dependency-edge value shrinks as model strength rises.** VERIFIED (numbers in A10) [pre]
  - −5.7 pp (GPT-4o) vs −1.0 pp (GPT-4.1) function Acc@1 when dependency edges are removed.
  - A second data point: CodexGraph's graph helps GPT-4o and hurts Qwen2-72B on SWE-bench Lite (1.95% vs AutoCodeRover's 9.34%).
  - Together: weaker models gain *more* from edges when the interface is simple (LocAgent fine-tuned 7B: +5.5 pp from traversal), but *lose* when the interface is expressive (Cypher).
  - **Why surprising:** this is a non-monotone interaction between interface complexity and model size, not a single "graphs help small models" law.
  - Sources: [RPG-Encoder](https://arxiv.org/abs/2602.02084); [CodexGraph](https://arxiv.org/abs/2408.03910); [LocAgent](https://arxiv.org/abs/2503.09089)

- **A18. Chain-level vs edge-level quality disconnect.** VERIFIED (A3) plus INFERRED link.
  - Per-edge precision of 0.99 becomes 0.19 per chain.
  - Downstream methods that traverse multi-hop (RepoGraph ego-graphs) inherit the chain-level number, not the edge-level one.
  - This may explain why graph *quality* metrics (edge P/R) do not predict downstream gains: downstream gains depend on path validity, which nobody measures.
  - Source: [Jarvis](https://arxiv.org/abs/2305.05949)

- **A19. Implicit runtime mechanisms are where both static graphs and LLMs break.** VERIFIED for the components; the combination is INFERRED.
  - Static side: PyCG misses built-in methods and `super()` (A1). Jarvis lists functions stored in list/tuple/dict and native libraries as remaining false negatives (Jarvis §IV-C).
  - LLM side: models truncate traces at `__getattr__`, `__setattr__`, generators and decorators (SWE-FLUX Discovery 3).
  - HeaderGen (static) widens decorator-changed return types to `Any`, while LLMs infer them (A7).
  - **Exploit:** decorators, registries and attribute magic are exactly where an LLM's *type* inference (strong) could repair a static call graph's *edges* (weak). That is a hybrid nobody has evaluated for localisation.
  - Sources: [DyPyBench](https://arxiv.org/abs/2403.00539); [Jarvis](https://arxiv.org/abs/2305.05949); [SWE-FLUX](https://arxiv.org/abs/2609.28449); [Venkatesh et al.](https://arxiv.org/abs/2410.00603)

- **A20. Cost anomaly: static whole-repository graphs can be slower than running the tests.** VERIFIED [mixed]
  - DyPyBench: dynamic call graphs "can be generated quicker than static call graphs for some projects", and PyCG timed out at 6 h on 6 of 50 projects.
  - RepoScope's static graph took **875 s** to index vs 86 s for DyCoder (ASE'26).
  - CodexGraph ran out of memory on 43 SymPy instances.
  - RPG full rebuild costs 14.7M tokens over 36 commits vs 633K incremental.
  - **Why surprising:** static graphs are assumed to be the cheap option. At repository scale, eager whole-program construction is often the *expensive* path, which favours lazy or on-demand construction (A6, A8, A11).
  - Sources: [DyPyBench](https://arxiv.org/abs/2403.00539); [DyCoder](https://arxiv.org/abs/2608.01927); [CodexGraph](https://arxiv.org/abs/2408.03910); [RPG-Encoder](https://arxiv.org/abs/2602.02084)

- **A21. Agents use graphs to reduce work, not to find things they could not otherwise find.** VERIFIED for the numbers; the reading is INFERRED.
  - DualView cuts steps by 6–9 and cost by about $0.05 per instance while adding only +2.9 to +6.3 pp resolution.
  - RPG-Encoder's headline efficiency is 6.34–8.22 steps vs 20–37 for OrcaLoca (except Claude, A10).
  - CABRA agents solve call-graph tasks with scripts anyway (A5).
  - **Why surprising:** papers frame graphs as *capability*. The data reads more like graphs as *efficiency*. That matters under our ~5-minute budget more than under unlimited ones.
  - Sources: [DualView](https://arxiv.org/abs/2607.01929); [RPG-Encoder](https://arxiv.org/abs/2602.02084); [CABRA](https://arxiv.org/abs/2610.10610)

- **A22. Oracle-context gap persists even with the best localiser.** VERIFIED [pre]
  - SemNav context lets Qwen3.5-35B resolve 52.33% vs **67.00% with oracle source files**, i.e. 78.1% of oracle.
  - With a Gemma-4B repairer it is 27.67% vs 39.00%.
  - A 15-point headroom remains after localisation at 4B, so the localiser is still the bottleneck at small scale.
  - Source: [SemNav](https://arxiv.org/abs/2609.31176)

### Inferences
- **Exploitable meta-pattern (INFERRED).** Across A3, A8, A11, A14, A16 and A18, what predicts downstream gain is (a) whether the *entry* is right, (b) whether the *path* is valid (not merely the edge), and (c) whether the *correct* node is present at all (not its rank). None of the 2024–2026 construction papers optimises for these. A graph-repair objective scored on path validity from issue-derived anchors, rather than on edge P/R, is open.
- **Relation to our own anomalies.**
  - "Spurious edges hurt more than missing ones" (ARISE) and "2-hop worse than 1-hop" (RepoGraph) both follow from A3/A18 (chain precision collapse).
  - LocAgent (A14) shows multi-hop helps when the agent filters by type and direction. That suggests our undirected, mixed-label graph *blocks* the filtering that makes multi-hop pay off.
  - This is a sharper statement of thesis T1 (repair direction and type before traversal).

### Gaps
- No controlled study found that varies call-graph recall or precision (e.g. by deleting or injecting a fraction of edges) and plots LLM *localisation* accuracy. ARISE is the closest, and it is already in our notes.
- No study of undirected or symmetrised graphs as an independent variable.
- No study of LLM localisation on graphs whose edge labels are mixed (containment + call + test under one label).

---

## Anomaly index (one line each)

A1 PyCG 49% runtime-edge recall, mostly built-ins · A2 micro-benchmark saturation (103/112 sound) vs real 49% · A3 0.99 edge precision → 0.19 chain precision · A4 LLM runtime data flow 1–2%; Gemma-4-31B ≈ 4B; over-extension 260 vs 44 · A5 agents script around call-graph traversal; naming reliance · A6 lazy LSP resolution +7 pp at 4B · A7 LLMs < PyCG on call graphs, > static on types; small models emit empty graphs · A8 LLM partial graph beats static global graph, 7.4× faster · A9 CodexGraph has no call edges; query generation is the bottleneck · A10 dependency edges −5.7 pp (GPT-4o) vs −1.0 pp (GPT-4.1) · A11 RepoGraph DR@15 0.127 < BM25 0.296 · A12 visual > textual rendering of the same graph (78 vs 71) · A13 graph-only > graph+source (context dilution) · A14 containment-only ≈ no graph; multi-hop > 1-hop; attributes hurt · A15 semantic cards −48% context, +6 pp · A16 stale context worse than none; presence > rank · A17 non-monotone interface × model-size interaction · A18 path validity, not edge P/R, is what matters downstream · A19 decorators and magic methods break both static tools and LLMs · A20 static graph construction often costlier than dynamic or lazy · A21 graphs buy efficiency more than capability · A22 15-pt oracle-context headroom at 4B.

---

## Search log

| # | Tool / query | Result used |
|---|---|---|
| 1 | alphaXiv discover: "accuracy of static vs LLM-generated call graphs for Python… graph construction quality affect LLM code comprehension" (kw: call graph, PyCG, code graph, LLM, Python) | FeatLens, DyCoder, CPyGraph, PyCG |
| 2 | alphaXiv discover: "Repository graph construction… granularity, edge types, hierarchical / semantic feature graphs (RPG, CodexGraph, RepoUnderstander), ablations" | RPG-Encoder, CodexGraph, DualView, SemNav, LARGER (already covered), Thousand-Graph (already covered) |
| 3 | Full text: CPyGraph 2609.25083 | A2 |
| 4 | Full text: FeatLens 2609.26480 | A11 |
| 5 | Full text: DyCoder 2608.01927 | A8, A20 |
| 6 | Full text: RPG-Encoder 2602.02084 | A10, A17, A20 |
| 7 | alphaXiv discover: "Can LLMs generate call graphs for Python… unsoundness of static call graphs (DyPyBench)" | Venkatesh et al., DyPyBench, Jarvis |
| 8 | alphaXiv discover (recency): "code graph quality or noise affect LLM issue localization… missing / spurious edges, schema ablations" | CABRA, RepBench, stale-context study |
| 9 | Full text: Venkatesh 2410.00603; DyPyBench 2403.00539; RepBench 2606.25356; CABRA 2610.10610 | A1, A4–A5, A7, A13 |
| 10 | Full text: Jarvis 2305.05949; LocAgent 2503.09089; stale context 2605.14478; DualView 2607.01929 | A3, A12, A14, A16 |
| 11 | alphaXiv discover: "Python dynamic features (decorators, dispatch, monkeypatching, registries) affect LLM comprehension / localization" | SWE-FLUX, PyFlow (not read) |
| 12 | Full text: CodexGraph 2408.03910; SemNav 2609.31176 | A6, A9, A15, A22 |
| 13 | OpenReview API `notes/search?term=code graph call graph LLM repository` | Mostly untitled review notes. Only "Call Chain-Aware Multi-View Context" (RepoScope) and "KG-based Repository-Level Code Generation" were titled; both are covered indirectly via DyCoder. Low yield. |
| 14 | Full text: SWE-FLUX 2609.28449 | A4, A19 |

**Searched with no usable hit:**
- Real-world recall of pyan, tree-sitter, Joern-Python or CodeQL-Python call graphs vs runtime.
- Controlled edge-deletion or edge-injection curves for LLM localisation (beyond ARISE).
- Directed vs undirected graph ablation.
- Mixed-label edge ablation.
- Localisation accuracy split by decorator, registry or monkeypatch reachability.
