# Scout: how code and code graphs are embedded and encoded for LLM code comprehension. Anomalies list (as of 9 Oct 2026)

**Goal.** Collect anomalies: surprising, unexplained or contradictory empirical findings about code and code-graph embedding or encoding that a new technique could exploit. Each item gives the source, the exact number, and the assumption it contradicts.

**Labels**
- **VERIFIED**: I read the number in the primary source's arXiv HTML during this session. The page was read through WebFetch, which has a summarising model in the loop. Table values were cross-checked against the prose where the fetch flagged a mismatch.
- **SNIPPET**: the number comes from a search-engine summary or abstract only. Re-check it before citing it in the paper.
- **INFERRED**: my own reasoning.
- **[peer]** / **[pre]**: peer-reviewed / preprint, as far as the source shows.

**Not repeated from prior notes**: Agentic Graph Tokens' +29 zero-shot result, "graph tokens sink", BM25 beating agentic retrieval at scale, SweRank beating graph agents, CGM needing scale (here only its *ablation table* is new), GREPO, SHERLOC, Schrödinger's Repo, Face/Off.

**Anomaly index**: 22 anomalies, A1–A22, grouped under the five key questions.

---

## Q1. Code embedding models on repo-level retrieval and issue-to-function localisation: where do they fail (anisotropy, length, identifier reliance)?

### Takeaway
Off-the-shelf code embedders score highly on snippet search, but they collapse on issue-to-edit localisation (NDCG@10 roughly 70 → 20) and do no better than BM25 at function level. Their scores depend heavily on identifier names, and more so on long code. Two patterns are cheap to exploit:
- **Training-free whitening** and **query-side-only adaptation** both recover large gains.
- **Snippet-benchmark rank does not predict downstream rank.**

### Cited Findings
- **A1. Leaderboard rank does not transfer to issue-to-edit localisation.** VERIFIED [pre; an HF model card cites v3 as EMNLP 2026, SNIPPET]
  - Qwen3-Embedding-8B: NDCG@10 71.7 on CORE-Bench Level-1 (snippet search) vs **20.3** on Level-2 (issue-to-edit).
  - C2LLM-7B: 72.4 → 16.7. Qwen3-Embedding-4B: 72.7 → 18.3.
  - SweRankEmbed-Large has the *lowest* L1 of the 7B models (45.4) but the *best* base L2 (22.4).
  - Scaling helps little: Qwen3-Embedding 0.6B / 4B / 8B → L2 17.0 / 18.3 / 20.3.
  - In-domain PR-data SFT trades L1 for L2. Qwen3-8B goes L1 71.7 → 63.0 and L2 20.3 → 32.8. The paper does not discuss the L1 loss.
  - *Contradicts*: that general code-retrieval quality (the MTEB/CoIR style) is the right axis to optimise for SWE localisation.
  - — [CORE-Bench, arXiv 2606.11864](https://arxiv.org/html/2606.11864)
- **A2. Frozen code embedders ≈ BM25 at function level, and adapting only the query side fixes most of the gap.** VERIFIED [pre]
  - On 369 issues from 5 repos, function-level MRR: BM25 0.32, frozen CodeRankEmbed **0.29**, frozen Qwen3-Embedding 0.32, frozen Jina-Code-v2 0.35.
  - HyperFL keeps *both* encoders frozen and caches the code embeddings. It only generates query-specific LoRA for the issue encoder, and reaches MRR 0.51–0.57. That is +0.22 over the same frozen code vectors.
  - More data hurt: dropping the BM25 training-pair filter grows training to ~145K pairs but lowers MRR from 51.3% to **42.5%**.
  - *Contradicts*: that code-side representation quality is the bottleneck. The issue-text encoding is.
  - — [HyperFL, arXiv 2608.02967](https://arxiv.org/html/2608.02967)
- **A3. Identifier anonymisation halves general embedders, and the drop is worst on long code.** VERIFIED [pre]
  - CLARC (C/C++), NDCG standard → neutralised:
    - OpenAI-text-embedding-large on Group 2: 85.87 → **44.20**.
    - On "Group 3 Long": OpenAI 83.80 → 34.80; Nomic-embed-code 69.46 → 38.70; even Voyage-code-3 89.13 → 63.90.
  - Under anonymisation every model scores *lower* on Group 2 than on Group 1, which reverses the standard ordering.
  - Unexplained: Voyage-code-3 scores *higher* with random identifiers than with neutral placeholders on Long (66.40 vs 63.90).
  - *Contradicts*: that code embedders encode semantics, and that longer functions give more signal.
  - — [CLARC, arXiv 2603.04484](https://arxiv.org/html/2603.04484)
- **A4. Contradicts A3: for code-to-code retrieval, the top embedders are invariant to identifier perturbation.** VERIFIED (claim level; the paper gives no per-model renaming breakdown) [pre]
  - Recall-Before-Rerank reports that top embedders (Qwen3 Embedding, Nomic, CodeXEmbed, CodeSage-V2) "remain invariant under style, comment, and identifier perturbations", while weaker models swing up to 30 points.
  - LLM rewrite-normalisation lifts Code Llama Java P@50 from 20% to 48%.
  - — [Recall Before Rerank, arXiv 2606.27401](https://arxiv.org/html/2606.27401v1)
- **A5. A raw LLM used as an encoder is worse than a 50× smaller embedder.** VERIFIED [pre]
  - Qwen3-Embedding-0.6B beats Qwen3-Coder-30B: P@50 **97 vs 52** on CodeNet-Python and **93 vs 8** on xCodeEval-Java.
  - Inversion on BigCloneBench Type-3 (near-miss clones): lightweight encoders score 46–70%, the top semantic embedders only **29–42%**.
  - *Contradicts*: that more capable or better-ranked encoders dominate uniformly. Strong embedders lose syntactic near-duplicate signal.
  - — [Recall Before Rerank, arXiv 2606.27401](https://arxiv.org/html/2606.27401v1)
- **A6. Training-free whitening gives gains as large as fine-tuning, and contrastively trained models gain nothing.** VERIFIED [pre; ESANN 2025 per search snippet]
  - IsoScore of base CodeBERT ≈ **0.005–0.007** (near-total anisotropy). Code Llama ≈ 0.002–0.010.
  - Soft-ZCA whitening alone adds MRR **+0.10 to +0.25** (CodeBERT base) and **+0.23 to +0.48** (Code Llama, e.g. Ruby +0.476).
  - CodeT5+ (contrastively trained, IsoScore ≈ 0.14–0.40) gains **0.000–0.035**.
  - Fine-tuned CodeBERT reaches IsoScore 0.998 after whitening but gains only +0.04–0.075.
  - — [Isotropy Matters / Soft-ZCA, arXiv 2411.17538](https://arxiv.org/html/2411.17538v2)
- **A7. Field-wide anisotropy matches ours, and length is a bigger confound than anisotropy.** VERIFIED [pre]
  - Random-pair cosine across layers: StarEncoder 0.94–0.98, CodeLlama-7B 0.74–0.97, Qwen2.5-Coder-7B 0.84–0.95. "No layer approaches isotropy."
  - Length confounds code–English alignment much more strongly than anisotropy, with |ρ| up to **0.96** for Qwen2.5-Coder.
  - — [Rethinking Pivot PLs, arXiv 2609.22988](https://arxiv.org/html/2609.22988)
- **A8. Better retrieval beats *gold* context downstream, and a smaller retriever beats a larger one.** VERIFIED [peer: COLM 2025 per PDF header in search result]
  - RepoEval Pass@1 with retrieved context: CodeXEmbed-2B **66.3**, CodeXEmbed-7B 57.8, **gold 39.1**. The 7B has the higher CoIR score (70.46 vs 67.41) but loses downstream.
  - On SWE-bench-Lite every retriever is within about 1–2 tasks of 0–2.7%. "OpenAI + rerank" scores **0.0**.
  - *Contradicts*: that retrieval metrics are monotone in downstream gain, and that gold context is an upper bound.
  - — [CodeXEmbed, arXiv 2411.12644](https://arxiv.org/html/2411.12644v2)
- **A9. Names act as a retrieval key into memory, and removing them hurts reasoning that should not depend on names.** VERIFIED [pre]
  - ClassEval class-level intent: GPT-4o **87.3 → 58.7** when names are removed. On LiveCodeBench (median identifier length ~2) it is roughly unchanged (73.4 → 70.1).
  - Execution prediction, which is semantically name-free, also drops: Llama-4-Maverick on high-complexity LiveCodeBench 80.2 → **56.4** with ambiguous names.
  - Yet alpha-renaming *raised* Llama-4 on ClassEval (81.1 → 83.4).
  - GPT-4o reproduced stale dataset outputs 13× on original code and ~0 under obfuscation, so identifiers trigger memorised answers.
  - — [When Names Disappear, arXiv 2510.03178](https://arxiv.org/html/2510.03178)

### Inferences
- **Our setting is the A1 + A2 regime.** Name-keyed node embeddings plus issue text is exactly issue-to-edit localisation. The field evidence says the cheapest lever is on the **query side** (issue → symbol), not on improving the node vectors. This matches our A2 finding that the entry point is the bottleneck. (INFERRED)
- **Whitening is a one-line, falsifiable test on our 256-d vectors** (A6, A7). Our raw random-pair cosine of 0.77–0.90 falls in the field's range. Centering already brings it to ~0.04–0.1. Soft-ZCA could test whether neighbour ranking improves beyond centering. (INFERRED)
- **The A3/A4 contradiction looks query-modality dependent.** NL→code loses its only lexical bridge when names disappear. In code→code, the query and target share structure, so identifiers are not the only shared signal. I found no paper that resolves this directly. (INFERRED)

### Gaps
- I found no MTEB/CoIR ↔ SWE-bench *resolve-rate* correlation study, only A1 and A8. The MTEB "SWE-bench localisation reranking" task exists only as an unmerged results PR ([PR #732](https://github.com/embeddings-benchmark/results/pull/732), SNIPPET: bge-small ~0.52).
- I found no renaming study run on issue-to-function localisation embedders specifically.

---

## Q2. Structure-aware embeddings (GraphCodeBERT, UniXcoder, AST, data-flow, call-graph smoothing): do they beat text-only on modern tasks?

### Takeaway
The measured marginal value of structure inside an *encoder* is small: about 2 MRR points for data flow in GraphCodeBERT, and nothing for generation in UniXcoder. Graph-based retrieval for code is almost always used as a **re-ranker or expansion step on top of similarity retrieval**, and none of the papers found isolates it cleanly. A structural non-embedding signal (RepoMap) wins the one query type where text fails.

### Cited Findings
- **A10. Data flow is worth about 2 MRR points.** SNIPPET (Table 5 numbers via search summary) [peer: ICLR 2021]
  - GraphCodeBERT code search: full model **0.713** vs **0.693** without data flow. Removing the edge-prediction or node-alignment objectives costs 0.006 and 0.010.
  - UniXcoder: AST helps understanding tasks but "does not bring improvements on generation tasks", and BFS/DFS serialisation "even hurt" some tasks.
  - — [GraphCodeBERT, arXiv 2009.08366](https://arxiv.org/pdf/2009.08366); [UniXcoder, arXiv 2203.03850](https://arxiv.org/pdf/2203.03850)
- **A11. A structural map wins where dense retrieval fails, and fusing them helps.** VERIFIED [pre]
  - On trace2code queries, RepoMap (a structural repo map) reaches MRR **0.2742** vs Qwen3-4B-Embedding **0.0827**.
  - RRF of Qwen3-8B + RepoMap raises overall MRR 0.2296 → **0.2713**.
  - Retrieval proxies correlate weakly with agent outcomes: BCY@8k vs final File F1 ρ = **0.129–0.237**.
  - Live agents never touch a gold file on 27–35% of samples.
  - About 82% of corpus chunks are exact duplicates across nearby commits.
  - — [Agent Retrieval Bench, arXiv 2607.24882](https://arxiv.org/html/2607.24882)
- **Graph retrieval for code is reported as a re-ranker, never a clean replacement.** SNIPPET [peer: GraphCoder ASE 2024; CodeRAG EMNLP 2025]
  - GraphCoder re-ranks a similarity pool by code-context-graph distance: EM +6.06 / +6.23 vs RAG baselines.
  - RepoHyper reports graph search beating kNN on *hit rate* only.
  - CodeRAG treats data-flow as one retrieval path among several.
  - — [GraphCoder](https://arxiv.org/html/2406.07003v2); [RepoHyper](https://arxiv.org/pdf/2403.06095); [CodeRAG](https://arxiv.org/html/2509.16112v1)

### Inferences
- **The competition embeddings are already "structure-aware" in the GNN-smoothing sense.** The general-graph evidence (A16) says GNN smoothing often *loses* to an MLP or to centre-node-only input. So the graph-smoothed vectors may carry less issue-relevant signal than a plain text embedding of the node's own source would. (INFERRED)
- **Cheapest falsifier**: embed node source with a small text encoder (or BM25 over node text). Compare it against the provided vectors on the 118 gold-mapped tasks, using the existing A2 harness. (INFERRED)

### Gaps
- I found no 2025–26 controlled "retrieval alone vs retrieval + k-hop call-graph smoothing, same encoder" study for code. GREPO is the closest, and it is already covered.
- UniXcoder's exact w/o-AST code-search number was not retrieved.

---

## Q3. Graph encoders feeding LLMs (soft prompts, graph tokens, GNN adapters, CGM): measured gains vs cost, failure modes, size trends

### Takeaway
Across three independent 2025–26 audits:
- Graph-token LLMs mostly ignore structure (accuracy unchanged under structure attacks) or memorise token→label maps.
- A linear probe on the projector output predicts the LLM's accuracy (r = 0.96).
- MLP adapters beat GNN adapters by about 9 points.
- 7B → 13B does not improve structure use.

CGM's own ablation shows the graph attention mask is worth only 2.9–4.0 EM points. Its reconstruction pre-training and adapter are worth far more.

### Cited Findings
- **A12. CGM: graph-in-attention is the *smallest* ingredient, and backbone scale dominates.** VERIFIED (Appendix C.7 tables; the prose percentages disagree with the tables, so table values are used) [peer: NeurIPS 2025]
  - CrossCodeEval Java EM:

    | Setting | Java EM |
    |---|---|
    | Frozen | 17.91 |
    | Adapter only | 41.70 |
    | Adapter + LoRA | 49.51 |
    | Full CGM without graph mask | 48.71 |
    | Full CGM | **51.61** |
    | Full CGM without subgraph-reconstruction pre-training | 42.78 |

  - So the graph mask contributes +2.90 EM (Python +4.02), while reconstruction pre-training contributes +8.83 (Python +6.46).
  - Note that "without mask" (48.71) ≈ "adapter + LoRA" (49.51).
  - SWE-bench Lite by backbone: Qwen2.5-Coder-7B **4.00%**, Coder-32B 28.67%, Llama-3.1-70B 25.33%, Qwen2.5-72B 43.00%.
  - — [CGM, arXiv 2505.16901](https://arxiv.org/html/2505.16901)
- **A13. Graph-token LLMs are insensitive to structure attacks, or memorise.** VERIFIED [pre]
  - Under a PRBCD structure attack: TEA-GLM 45.82 → 45.80 (−0.04%), GraphGPT 49.69 → **50.73 (+2.09%)**, LLaGA −39.3%, GCN −73.7%.
  - LLaGA scores 84.29 originally, 81.54 rephrased, and only **3.03** on a "Random" instruction variant. After instruction tuning its Random score is 0.00 while its Relabel score is 89.89, which the authors read as memorised GT→label mapping.
  - Instruction tuning on variants does not generalise to unseen variants.
  - — [Revisiting Graph-Tokenizing LLMs, arXiv 2605.03514](https://arxiv.org/html/2605.03514)
- **A14. The LLM in a graph-language model is "an expensive decoder head".** VERIFIED [pre]
  - On 4 "semantically-sufficient" datasets, text-only soft-prompt Llama3-8B is within about 2 points of G-Token. Example: Computers 74.34 vs 76.13.
  - A linear probe on the *frozen projector output* tracks full GLM accuracy with Pearson **r = 0.9643**.
  - On the CLeGR reasoning benchmark, GLMs only match soft-prompted LLMs, and G-Retriever is *worse*.
  - — [A Graph Talks, But Who's Listening?, arXiv 2508.20583](https://arxiv.org/html/2508.20583)
- **A15. Structure tokens hurt, and dropping neighbours helps.** VERIFIED [pre]
  - LLaGA-style Vicuna-7B node classification: structure-aware template **69.48%** < structure-free neighbour sampling 74.49% < **centre node only 75.29%**.
  - On School: 66.43% (structure) vs 91.13% (centre only).
  - Model size: Llama2 7B → 13B gives no improvement in structure use (School structure-aware 70.98 → 69.30).
  - — [When Structure Doesn't Help, arXiv 2511.16767](https://arxiv.org/html/2511.16767)
- **A16. MLP beats GNN as the graph-token adapter, and deeper GNN is worse.** VERIFIED [pre]
  - GraphToken node classification: MLP **76.26%** vs GCN 67.50, GAT 67.07, GIN 68.24. The authors attribute the gap to over-smoothing.
  - Link prediction is ≈ equal (MLP 75.19 vs GCN 75.51).
  - — [When Structure Doesn't Help, arXiv 2511.16767](https://arxiv.org/html/2511.16767)

### Inferences
- **A16 + our measurements point to over-smoothing.** "Node ≈ mean of neighbours (cos 0.85–0.97)" plus "top-1 neighbour is a direct graph neighbour only 3–9%" is the signature of a low-pass graph filter. Under repeated neighbour averaging, nodes that *share many neighbours* (2-hop structural siblings, e.g. methods of the same class or co-callees of a hub) become closer than adjacent nodes.
  - **Prediction**: the top-1 embedding neighbour is disproportionately a 2-hop node sharing ≥2 neighbours.
  - **Falsifier**: compute the hop distance and the Jaccard of neighbour sets for top-1 pairs, using the existing `verify_graph.py` data. This takes minutes. (INFERRED)
- **For a 4-bit ~30B Gemma, the A12 size curve is the warning.** CGM at 7B is 4%. Structure-in-attention is unlikely to pay at our size; serialised or tool-delivered structure is more plausible. (INFERRED)
- **A13/A14 suggest a cheap test.** If graph embeddings were fed as soft tokens, a linear probe on them would likely predict most of the gain, so it should be tested first. (INFERRED)

### Gaps
- I found no study that varies LLM size systematically (e.g. 1B → 70B) for graph-token gains on *code* graphs.
- CGM's "no graph, text-only node encoding" ablation was not reported.

---

## Q4. LLM hidden states as code/graph embeddings: is call-graph or reachability structure linearly present, and is it used?

### Takeaway
Graph structure is linearly decodable from LLM residual streams. Two anomalies stand out:
- The transitive closure is **more** decodable than direct edges.
- The representation is present but **not behaviourally used**: patching moves it, but tool calls change 0% of the time, and zero-shot use is weak.

Representations reorganise to an in-context graph, but the mechanism may be neighbour-mixing by previous-token heads, the same "node ≈ neighbour mean" signature we see.

### Cited Findings
- **A17. Reachability is more decodable than adjacency, and the probe is not causally used.** VERIFIED [pre]
  - Qwen3-32B, τ-bench, linear edge probe for tool-call dependency: direct-edge AUROC **0.869** vs **transitive 0.986**.
  - Controls: random-label control max 0.565; position-only baseline 0.792.
  - Llama-3.3-70B: direct 0.912, transitive 0.967.
  - Activation patching shifts the probe at later layers (+0.016 at L57), but the realised tool call changes in **0%** of cases.
  - Thresholded predictions are *not* transitively coherent: 0.307 vs a 0.429 null.
  - — [Tool-Call Dependency Structure is Linearly Decodable, arXiv 2605.25310](https://arxiv.org/html/2605.25310)
- **A18. Representations reorganise to an in-context graph, but priors resist.** SNIPPET [peer: ICLR 2025]
  - Llama-3.1-8B shows a sudden reorganisation from semantic to graph-aligned representations as context grows.
  - When nodes carry strong pretrained correlations (days of the week), the context graph is present but "unable to dominate".
  - An ICLR 2026 critique attributes the effect to previous-token heads *mixing neighbour representations*, with performance explained by induction circuits.
  - — [In-Context Learning of Representations, arXiv 2501.00070](https://arxiv.org/html/2501.00070v2); [ICLR 2026 critique](https://iclr.cc/virtual/2026/poster/10012124)
- **A19. The graph is present but unused zero-shot, and few-shot shifts the readout, not the representation.** VERIFIED (qualitative; exact AUROCs are only in figures) [pre]
  - Probe AUROC for the latent graph is "clearly higher" than ~0.5 chance (Qwen3-4B-Base, layer 20). It stays "largely unchanged" across 0–32 shots.
  - Accuracy rises with shots, and the shot vector is "nearly orthogonal" to the world subspace.
  - Zero-shot predictive use is "difficult".
  - Includes **Gemma-4** in the appendix.
  - — [Few-Shot Demonstrations Elicit the Use of In-Context World Representations, arXiv 2609.24352](https://arxiv.org/html/2609.24352)
- **A20. Middle layers beat the final layer as embeddings.** SNIPPET [peer: ICML 2025]
  - Across 32 text-embedding tasks, intermediate layers "consistently provide stronger features" than the last layer.
  - Code was not tested. AST probes on code models likewise find most AST information in *middle* layers.
  - — [Layer by Layer, arXiv 2502.02013](https://arxiv.org/html/2502.02013v2); [AST-Probe, arXiv 2206.11719](https://arxiv.org/pdf/2206.11719)
- **A21. LLMs lose to static analysers on call-graph construction.** SNIPPET [peer: EMSE 2025 / LMPL@SPLASH 2025]
  - Across 24 LLMs on Python/JS, PyCG and Jelly "consistently outperform" the LLMs (SWARM-CG).
  - CG-Bench indirect-call resolution: best DeepSeek-6.7B at **69.70%** few-shot. Removing irrelevant context helps.
  - — [SWARM-CG, EMSE](https://link.springer.com/article/10.1007/s10664-025-10704-3); [CG-Bench](https://conf.researchr.org/details/icfp-splash-2025/lmpl-2025-papers/3/CG-Bench-Can-Language-Models-Assist-Call-Graph-Construction-in-the-Real-World-)

### Inferences
- **A17's direct < transitive gap suggests an exploitable target.** The model's latent code is closer to *reachability* than to *adjacency*, so a reachability-style probe or reward may align with what the model natively represents. Our localisation needs multi-hop reachability from issue seeds (1-hop 22.6%, ≤2-hop 32.3% in A2). (INFERRED)
- **A17 + A19 locate the failure in readout, not representation.** Structure is present but not used. Interventions that change the *readout*, such as demonstrations (shot vectors) or RL on the structural answer, are more promising than adding more structural input tokens. This is consistent with "graph tokens sink". (INFERRED)
- **A18's "previous-token head mixing" gives a possible explanation for the competition embeddings.** An LLM-based featurizer applied over graph walks would itself produce "node ≈ neighbour mean". This is untested speculation. (INFERRED)

### Gaps
- I found no paper that linearly probes code-LLM hidden states for a *static repository call graph* (caller/callee, reachability) at function granularity. Prior probes cover AST and data-flow (Troshin & Chirkova 2022; AST-Probe), or run-time tool graphs (A17). **This looks like an open, cheap experiment on Gemma.**
- Middle-layer vs last-layer for code *retrieval* embeddings is untested as far as I found.

---

## Q5. Findings where embedding-quality metrics do not predict downstream SWE gains

### Takeaway
At least four independent sources show a disconnect between embedding quality and downstream gains:
- CoIR rank vs RepoEval/SWE-bench resolve (A8).
- L1 vs L2 retrieval (A1).
- Retrieval proxies vs agent outcomes, ρ ≈ 0.13–0.24 (A11).
- Graph-token representation quality vs structure use (A13, A14).

### Cited Findings
- **A22. Retrieval metrics are a weak proxy for agent behaviour.** VERIFIED [pre]
  - Budgeted context yield correlates with any-gold-touched at ρ = 0.230 / 0.172, and with final File F1 at ρ = 0.237 / 0.129.
  - Oracle gold context adds +0.3115 File F1, while retrieval-derived seeds add only ~0.075 and random context adds 0.0215.
  - Rankings flip between sample weighting and repo weighting (Qwen3-8B vs Qwen3-4B) and between token budgets (4k vs 8k).
  - — [Agent Retrieval Bench, arXiv 2607.24882](https://arxiv.org/html/2607.24882)
- Also see A1 (CORE-Bench L1 "only moderately predicts" L2), A8 (CodeXEmbed 2B > 7B downstream; retrieved > gold on RepoEval; ±1 task on SWE-bench-Lite), and A14 (r = 0.96 between a projector probe and GLM accuracy, meaning the LLM adds little).

### Inferences
- **Evaluate any embedding change on our own metric, not on CoIR-style metrics.** That metric is gold-node hit within ≤2 tool calls, plus resolve rate. A2's 3.3× oracle-vs-seed gap suggests seeds matter far more than ranking quality. (INFERRED)

### Gaps
- I found no study that correlates an embedding's anisotropy or isotropy with downstream SWE resolve rate.

---

## Search log
Queries run on 9 Oct 2026, in order. All went through WebSearch (standard or extended), followed by WebFetch on arXiv HTML. Direct PDF fetches failed, because the streams are compressed and no local PDF tool was available, so the HTML versions were used instead.

1. "code embedding models identifier renaming robustness retrieval drop" → CLARC, Recall Before Rerank, When Names Disappear, Embedding Drift 2609.21203, Vulnerable Code Search 2608.26031.
2. "code retrieval embedding benchmark SWE-bench function localization embedding models fail 2025 2026" → CORE-Bench, HyperFL, CodeXEmbed, MTEB PR #732, SWE-Adept.
3. "probing LLM hidden states call graph code structure linear probe" → Tool-Call Dependency probe 2605.25310, Code Correctness probe 2606.14530.
4. "graph tokens LLM code GNN adapter soft prompt code graph structure ablation 2025" → Revisiting Graph-Tokenizing LLMs, TEA-GLM, GraphAdapter, Ex-GraphRAG.
5. Fetched the HTML of 2605.03514, 2603.04484, 2510.03178, 2411.12644, 2606.27401.
6. "Code Graph Model CGM SWE-bench Lite … ablation" → fetched 2505.16901, including Appendix C.7.
7. "structural probe pre-trained code models AST …" → AST-Probe, CAT-probing, Ma et al.
8. "HyperFL …" and "CORE-Bench …" → fetched both, plus Agent Retrieval Bench 2607.24882.
9. "intermediate layers outperform final layer … Layer by Layer"; "LLMs understand call graph … reachability"; "graph-based retrieval repository-level code completion GraphCoder RepoHyper CodeRAG".
10. "In-Context Learning of Representations …"; "structure-aware code embedding … CoIR" (no CoIR-structure study found); "anisotropy code embeddings whitening" → fetched Soft-ZCA; "SweRank … small model" (already covered, not expanded).
11. "tree positional encoding AST LLM code … ablation" → HiRoPE, Tree-Enhanced CodeBERTa. No decoder-LLM marginal-gain study was found, so this question has no entry (gap).
12. "probing whether LLM embeddings encode graph structure …" → fetched 2508.20583 and 2609.24352. Read 2605.25310 in full.
13. "GraphCodeBERT ablation without data flow …"; "graph tokens … gains diminish larger LLM scale" → fetched 2511.16767; "call graph neighbor aggregation code embedding … bug localization" (no controlled smoothing study found).
14. Fetched 2608.02967 (HyperFL tables) and 2609.22988 (anisotropy ranges).

**Not found (supports novelty)**:
- No linear probe for static repository call graphs or reachability in code LLMs.
- No controlled comparison of call-graph smoothing vs raw text embedding for issue localisation.
- No structural positional encoding ablation on decoder code LLMs.
- No study that varies LLM size for graph-token gains on code.
