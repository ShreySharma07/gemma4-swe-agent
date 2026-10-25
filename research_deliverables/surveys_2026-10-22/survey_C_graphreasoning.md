# Survey C: Improving LLM reasoning over large-scale graphs (2024 – Oct 2026)

Compiled 2026-10-22. Sources: alphaXiv `discover_papers` / `get_paper_content` / `answer_pdf_queries`, the arXiv export API (abstracts and comments), and WebSearch. Every row below was retrieved through one of these tools. Numbers come from the abstract or the full text I read. "abs" means the number is from the abstract only, so I did not see the experimental setup. "unverified" means I did not open the paper.

Venues come from arXiv comments where present. Otherwise the venue is "arXiv".

---

## 1. Graph-as-text encodings and their sensitivity

| Paper | Year / venue | ID | Method (1 line) | Key quantitative result | Limitation |
|---|---|---|---|---|---|
| Talk Like a Graph (Fatemi, Halcrow, Perozzi) | 2023 / ICLR'24 (venue unverified) | 2310.04560 | Systematic study of graph-to-text encoders (edge lists, "friendship", incident etc.) on GraphQA | Choice of encoder changes accuracy by 4.8–61.8 pts depending on task (abs) | Small synthetic graphs; pre-reasoning-model LLMs |
| NLGraph (Wang et al.) | 2023 / NeurIPS'23 spotlight | 2305.10037 | 29,370-problem NL graph benchmark (8 tasks); Build-a-Graph and Algorithmic prompting | Prompting gains 3.07–16.85%; advanced prompting/ICL benefit shrinks on complex tasks; brittle to spurious correlations (abs) | Tiny graphs; GPT-3/4 era |
| Lost in Serialization (Herbst … Frasca) | 2025 / ICML'26 GFM workshop | 2511.10234 | Splits serialization into node labeling, edge encoding and syntax; tests G1-style post-trained reasoners on Erdős | Larger non-fine-tuned models are more robust. **Fine-tuning reduces relabeling sensitivity but can increase sensitivity to structure/format changes, and does not consistently help unseen tasks** | Workshop paper; small graphs |
| LLMs Fail at Graph Isomorphism | 2026 / arXiv | 2606.09484 | Isomorphism detection with permuted node labels | Near-perfect on "detect differences", but fails when identical graphs get permuted labels, so it exploits patterns rather than structure (abs) | Single task |
| Graph Serializations for Reasoning LLMs (Oracle) | 2026 / EMNLP'26 Findings | 2610.22734 | 15 serialization formats × 9 frontier reasoning-LLM configs | Knowledge extraction is format-insensitive. Structural reasoning still depends on format. The best format depends on model × query × task (abs) | No single recommended format |
| GTA / GT-Bench | 2026 / arXiv | 2609.12265 | 24 classical problems, 44 settings, >100k examples in 4 representations; agent with preference-trained representation selector + plan/decompose | Best representation shifts with density, size, topology and model. Sensitivity persists, attenuated, in the strongest reasoning models. GTA lifts Phi-4 from 53.5→69.1% (easy) and 33.0→41.5% (hard) | Executor is frozen; graphs still in context |
| Colorful Talks with Graphs | 2026 / ACL'26 Findings | 2602.10386 | Encode WL-colour classes as human-like colour words in the prompt | "Considerable" gains, especially on global-structure tasks (abs; no number in abstract) | Pre-computation of WL is itself a graph algorithm |
| When Structure Doesn't Help | 2025 / LoG'25 | 2511.16767 | Compare structure encodings for text-attributed graphs (TAGs) | Node text alone is already strong; most structural encodings give marginal or negative gains (abs) | TAG node classification tasks only |
| Rethinking & Benchmarking LLMs for Graph Reasoning | 2025 / arXiv | 2509.24260 | Argues models should *design* algorithms, then code (Simple-RTC); new GraphAlgorithm bench (239 problems, 3,041 instances) | Simple-RTC is near-perfect on existing benchmarks (abs), so the old benchmarks are saturated by code | Shifts the problem to code execution |

## 2. Graph tokens / GNN–LLM alignment

| Paper | Year / venue | ID | Method | Key result | Limitation |
|---|---|---|---|---|---|
| GraphToken (Perozzi et al.) | 2024 / arXiv | 2402.05862 | Learned GNN encoder produces soft prompt tokens for a frozen LLM | Up to +73 pts on GraphQA node/edge/graph tasks (abs) | Small synthetic graphs; task-specific encoder |
| GraphGPT | 2023 / SIGIR'24 | 2310.13023 | Text-graph grounding + 2-stage graph instruction tuning with projector | Claims SOTA in supervised and zero-shot node tasks (abs, no number) | Citation-graph TAGs; later found to under-use tokens |
| LLaGA | 2024 / ICML'24 (venue unverified) | 2402.08170 | Node-neighbourhood templates → projector → LLM tokens | Single model across 4 datasets / 3 tasks (abs) | Sink tokens are PADs (see 2606.03712) |
| G-Retriever | 2024 / NeurIPS'24 (venue unverified) | 2402.07630 | PCST subgraph retrieval + GNN soft prompt + LoRA; GraphQA benchmark (textual graphs) | Scales to graphs beyond context and reduces hallucination (abs) | Degrades GPT-4o-mini on GraphRAG-Bench (2506.02404) |
| Revisiting Graph-Tokenizing LLMs (GTEval) | 2026 / arXiv | 2605.03514 | Format/content instruction perturbations on 6 GTokenLLMs | **Models do not fully understand graph tokens.** They are over- or under-sensitive to instruction changes and rely heavily on text. Extra instruction tuning does not fix this (abs) | Node-level TAG tasks |
| When Graph Tokens Sink | 2026 / arXiv | 2606.03712 | Mechanistic analysis (activation sinks, interventions, logit lens) of LLaGA and TEA-GLM | The top-2 activation "graph sink" tokens in LLaGA are `[PAD]` tokens. Removing them barely changes accuracy, while removing random non-sink tokens hurts more. Logit lens decodes generic words ("paper") | Two GLMs, Vicuna-7B, citation graphs |
| Graph Evidence Is Not Enough (HopQA, S²GE) | 2026 / EMNLP'26 main | 2608.30437 | Intervention triangle: readable vs shuffled vs no-graph evidence for shortest-hop distance | Existing graph-augmented baselines fail on a trivially scored hop-distance question. S²GE (query-aware sampling + endpoint ordering) improves exact match by +53.5 pts on average over the best native baseline (abs) | Single bounded diagnostic |
| Big Claims, Small Margins | 2026 / arXiv | 2610.28714 | Re-tunes cheap baselines on GraphLLM papers' own splits | Correct&Smooth on TF-IDF trails the reported LLM SOTA by a median of only 2.3 pts and is within 3 pts on 15/24 tasks. The paper documents test-set selection and label-leak bugs (abs) | Node classification only |

## 3. RL / SFT for graph reasoning

| Paper | Year / venue | ID | Method | Key result | Limitation |
|---|---|---|---|---|---|
| GraphWiz | 2024 / KDD'24 (venue unverified) | 2402.16029 | GraphInstruct SFT with explicit reasoning paths, then DPO | 65% avg on 9 tasks vs GPT-4 43.8%; notes that more data leads to overfitting (abs) | Superseded; poor transfer to other encodings |
| GraphPRM / GraphSILO | 2025 / KDD'25 | 2503.00845 | MCTS-labelled step data, used to train a graph process reward model; best-of-N + DPO | +9% for Qwen2.5-7B across 13 tasks; also transfers to GSM8K/MATH500 (abs) | PRM trained on small synthetic graphs |
| G1 (Guo, Li, Wang, Jegelka, Wang) | 2025 / arXiv (venue unverified) | 2505.18499 | Erdős dataset (50 NetworkX tasks, 100k train, real subgraphs 5–35 nodes); CoT-SFT then GRPO with rule/Jaccard/verifier rewards | G1-7B 66.2% vs o3-mini 64.9%, GPT-4o-mini 47.6%. G1-3B beats Qwen2.5-72B. On larger graphs (36–100 nodes), G1-7B scores 46.5% vs 23.9% for its base. Transfers zero-shot to GraphWiz/GraphArena. RL restores GSM8K/MATH after SFT dents them | Graphs ≤100 nodes (context-bounded). Fine-tuning raises format sensitivity (2511.10234) |
| Generalizable LLM Learning of Graph Synthetic Data | 2025 / arXiv | 2506.00845 | GRPO/DPO on synthetic graphs with solution- vs process-based rewards; evaluates *implicit-graph* real tasks (multi-hop QA, planning) | +12.9% avg on 5 datasets. **Process rewards beat solution rewards on synthetic tasks but not on real-world tasks**; compositionality still unsolved (abs) | Small effect sizes |
| NPG-Muse | 2025 / arXiv | 2508.20373 | Long-CoT SFT + RL on NP-hard graph problems as a reasoning curriculum | 7B surpasses QwQ-32B on NP-hard graph problems; gains on math, code and logic (abs) | Context-bounded graphs |
| Graph-R1 (zero-shot graph learning) | 2025 / EMNLP'25 | 2508.17387 | GNN-free: linearized graphs + rethink templates + RL for node, link and graph classification | Beats SOTA zero-shot baselines (abs, no number) | Linearization limits size |
| GRAIN / GRIT | 2026 / arXiv | 2608.27142 | Single agent: parse text→graph then call deterministic tools; RL with a **Structure Invariance Reward** that checks the extracted graph against ground truth | +16.45% over multi-agent baselines, ~24% lower latency. Halves the OOD gap of SFT (15.77→7.80%) (abs) | Requires ground-truth topology for the reward |
| Generalization in LLM Problem Solving: Shortest Path | 2026 / ICLR'26 | 2604.15306 | Controlled from-scratch study: SFT vs Dr.GRPO vs inference-time search | >90% spatial transfer to unseen maps, but **length scaling fails from recursive instability**: P(long \| both halves correct) falls 0.81→0.59. **RL stabilizes training but never exceeds the best SFT**, and search does not rescue length. ~1% slightly-longer examples rescue length 30 to ~90% | Small from-scratch models; grid maps |
| Graph Reasoning Paradigm (PASC-GRPO) | 2026 / ACL'26 | 2601.12995 | Graph-structured reasoning steps with stratified clipping | Evaluated on math/code, not graph tasks (per WebSearch summary; unverified) | Not graph-data reasoning |
| GraphPO | 2026 / arXiv | 2606.18954 | Graph-based policy optimization (merges shared reasoning steps across rollouts) | Found via WebSearch only; it is general RLVR, not graph-data reasoning (unverified) | Off-topic for graph data |

## 4. Agentic / tool-based exploration of graphs too big for context

| Paper | Year / venue | ID | Method | Key result | Limitation |
|---|---|---|---|---|---|
| Think-on-Graph 2.0 | 2024 / ICLR'25 (venue unverified) | 2407.10805 | Training-free alternation of KG retrieval and doc context retrieval | SOTA on 6/7 datasets with GPT-3.5; lifts Llama-2-13B to GPT-3.5 level (abs) | Many LLM calls per hop |
| Fast Think-on-Graph | 2025 / arXiv | 2501.14300 | "Community by community" traversal with coarse/fine pruning | Faster and more accurate than ToG (abs, no number) | Community detection on dense KGs |
| Think-on-Graph 3.0 | 2025 / arXiv | 2509.21710 | Multi-agent, query-adaptive Chunk–Triplet–Community index | Beats baselines with lightweight LLMs (abs) | Complex multi-agent cost |
| Graph Chain-of-Thought / GRBench | 2024 / ACL'24 Findings (venue unverified) | 2404.07103 | Iterative reason → graph-function call → execute over 10 domain graphs; 1,740 Qs | Consistently beats baselines (abs) | Small benchmark; prompt-only |
| Graph-constrained Reasoning (KG-Trie) | 2024 / ICML'25 | 2410.13080 | Constrain decoding to valid KG paths via trie; small KG-LLM plus a big general LLM | "Zero reasoning hallucination" by construction; zero-shot to unseen KGs (abs) | Trie must be built over the paths; size limits |
| GraphChain | 2025 / arXiv | 2511.00457 | PPO-trained tool chaining with reward for shrinking graph-description length + relevance; spectral fingerprint soft-prompt TTA | 84.7% avg vs 70.2% for GraphForge. **Holds 79.5% on a 200k-node financial graph** while GPT-4o degrades | Own benchmark; static graphs |
| KG-R1 | 2025 / arXiv | 2509.26383 | Single-agent RL KG-RAG (Qwen2.5-3B) | Higher accuracy with fewer tokens than multi-module pipelines; plug-and-play on unseen KGs (abs) | KGQA benchmarks only |
| Search-on-Graph-R1 | 2026 / arXiv | 2607.18481 | One batched 1-hop `Search` tool; SFT on SPARQL-scaffolded teacher traces, then GRPO with exact-match × turn-penalty reward | Llama-3.1-8B: WebQSP 91.7, CWQ 88.7, GrailQA 90.0, beating GPT-4-class prompting. **SFT does most of the work** (CWQ 35.3→86.1); RL adds +0.8 to +2.4 and cuts calls 4–15%. RL-only reaches just 63.0 on CWQ | Needs gold SPARQL for scaffolding; oracle entity linking; Freebase only |
| AgentGL | 2026 / ACL'26 | 2604.05846 | Graph-native multi-scale tools + search-constrained thinking + graph-conditioned curriculum RL on TAGs | Up to +17.5% node classification and +28.4% link prediction (abs) | TAG benchmarks |
| GraphDancer | 2026 / arXiv | 2602.02518 | Two-stage RL with a curriculum over trajectory structural complexity | 3B, trained on one domain, generalizes to unseen domains (abs) | GRBench-scale graphs |
| GraphCert | 2026 / arXiv | 2609.38798 | Self-generated, execution-certified evidence rubrics used as GRPO reward | Beats larger agents across 5 GRBench domains (abs) | Under review |
| ARK | 2026 / ACL'26 | 2601.13969 | Two tools only: global lexical search + 1-hop expansion; agent chooses breadth vs depth | STaRK 59.1% Hit@1, up to +31.4% over training-free methods; distilled to 8B keeps 98.5% (abs) | Retrieval only |
| Knowledge Graphs as Files (DKGI) | 2026 / arXiv | 2610.18616 | Agent writes its own Python graph-access tools over raw node/edge files + shell | Beats best published baseline by ≥6.7 Hit@1 on every graph, training-free (abs) | Needs a code-execution sandbox |
| Schema-Agnostic Graph Reasoning Agent (GRA) | 2026 / arXiv | 2608.15834 | 7 generic ls/cat/grep-like tools over a hybrid KG | 88.4% vs 83.3% for a full-context agent at <1/3 the tokens. **A graph-free control shows the gain is mostly selective access, not topology** (abs) | Single industrial benchmark |
| Wikidata Search Traces | 2026 / arXiv | 2610.06650 | 10,235 solving traces + RLM harness (batched calls, results kept in Python state) | Claims failures come mostly from evidence management, not the model (abs) | Tech report |
| GraphEcho | 2026 / arXiv | 2609.17695 | Tests whether agents treat repeated paths as extra corroboration | Redundant paths increase repeated walks in all frozen agents. Provenance-aware post-training reduces revisits but covers fewer sources, and accuracy drops on real claims (abs) | Synthetic plus one science domain |
| GABench | 2026 / arXiv | 2608.01684 | 10,400 tasks, 84 executable graph tools, agentic (graph not in prompt) | Existing agents still struggle on complex graph analysis (abs) | New, few results yet |
| Auto-Cypher / SynthCypher | 2024 / NAACL'25 | 2412.12612 | LLM-as-database-filler verification for synthetic Text2Cypher data (29.8k) | Up to +40% on Text2Cypher test, +30% on graph-SPIDER (abs) | Query-generation only. "Mind the Query" (IBM, 27.5k queries, 11 graphs) found via WebSearch only (unverified) |

## 5. GraphRAG family and its failure modes

| Paper | Year / venue | ID | Method | Key result | Limitation |
|---|---|---|---|---|---|
| MS GraphRAG (From Local to Global) | 2024 / arXiv | 2404.16130 | LLM entity graph + Leiden community summaries for global QFS | Seminal (not re-read) | High indexing cost (see 2607.26497) |
| LightRAG | 2024 / arXiv | 2410.05779 | Dual-level entity/relation keyword retrieval | Popular baseline (not re-read) | Super-linear build cost (b=1.36), fails to finish at 2,826 docs (2607.26497) |
| HippoRAG 2 | 2025 / ICML'25 | 2502.14802 | Personalized PageRank over an open-IE triple graph + passage nodes | +7% associative memory over SOTA embeddings, without the factual-QA drop seen in other structured RAG (abs) | ~3 instance-days to build at 601M tokens (2607.26497) |
| GraphRAG-Bench (Xiao et al., textbooks) | 2025 / arXiv | 2506.02404 | 1,018 college-level CS questions, 5 types; scores rationales | Some GraphRAG methods (DALK, G-Retriever) *degrade* GPT-4o-mini. **All GraphRAG methods hurt on mathematics.** RAPTOR is best on hierarchical textbooks | LLM-judge rationale metric |
| When to use Graphs in RAG (GraphRAG-Bench, Xiang et al.) | 2025 / ICLR'26 | 2506.05690 | 4 task levels on medical + novel corpora; stage-wise metrics | Vanilla RAG ≥ GraphRAG on simple fact retrieval; GraphRAG wins on multi-hop/summarization. MS-GraphRAG global prompts reach ~40k tokens vs ~1k for HippoRAG2. Needs a ≥7B reader | Corpora still small |
| RAG vs GraphRAG | 2025 / arXiv | 2502.11371 | Unified protocol across QA and summarization | Distinct strengths; hybrid selection helps (abs) | — |
| In-depth Analysis of Graph-based RAG | 2025 / arXiv | 2503.04338 | Unified framework, 12-ish methods | New method combinations beat SOTA (abs) | — |
| LinearRAG | 2025 / arXiv | 2510.10114 | Relation-free entity graph (NER + semantic links), linear build | Beats baselines with no extra LLM tokens (abs); within 1.8 pts of MS-GraphRAG at bedrock (2607.26497) | Loses relation semantics |
| Graph-R1 (agentic GraphRAG) | 2025 / ICML'26 | 2507.21892 | Hypergraph KG + multi-turn RL agent | Beats GraphRAG and RL-RAG (abs) | Standard multi-hop QA sets |
| Do We Still Need GraphRAG? (RAGSearch) | 2026 / arXiv | 2604.09666 | Dense RAG vs GraphRAG as substrate *under agentic search* | Agentic search narrows the gap, especially with RL. GraphRAG still helps complex multi-hop if offline cost is amortized (abs) | QA benchmarks |
| **BM25 Wins at Scale** | 2026 / arXiv | 2607.26497 | 28 nested corpus tiers, 1.7M→601M tokens; same reader (Qwen3.6-27B); token-metered | Above ~10M tokens BM25 beats a file-system agent, by ~20 pts at full scale (50.5 vs 30.7). Graph RAG hits **construction walls** (LightRAG fails at 2,826 docs; HippoRAG2 extrapolates to 2.9B build tokens) and stays below BM25. **Agent + BM25 = 69.4%**, so agency should come *after* global ranking | Enterprise synthetic corpus |
| Is GraphRAG Needed? | 2026 / ACL'26 GEM ws | 2606.25656 | 9 RAG scenarios + context engineering | 19–53% token reduction. **Retrieval–generation gap: better retrieval does not proportionally improve answers** (abs) | Workshop |
| EffiRAG / Structure Pricing | 2026 / arXiv | 2609.18099 | Graph only *locates* passages; answer from source text | Preferred over LightRAG-hybrid 93/120 at 57% lower cost (abs) | 120 questions |
| MOSAIC | 2026 / arXiv | 2609.11065 | Per-query traversal policy (seed, depth, stop) set by an LLM analyzer | GraphRAG-Bench Medical 76.97 / Novel 64.33 (+5.1 / +4.4). No fixed policy is consistently optimal; evaluates 81.9% fewer paths (abs) | Training-free heuristics |

## 6. Theory: expressivity, CoT limits, size generalization

| Paper | Year / venue | ID | Result | Implication |
|---|---|---|---|---|
| Understanding Transformer Reasoning via Graph Algorithms (Sanford et al.) | 2024 / NeurIPS'24 (venue unverified) | 2405.18512 | Hierarchy: retrieval tasks (node/edge count, degree) need depth 1. Parallelizable tasks (connectivity, cycle, bipartite) need Θ(log N) depth, which is necessary under the 1-vs-2-cycle conjecture. Search tasks (shortest path, diameter, reachability) need log depth *and* width N^{1/2+ε}. A 1-layer transformer for connectivity or shortest path needs width Ω̃(N) | Fixed-depth LLMs cannot do connectivity on large graphs "in one forward pass". They need CoT, tools, or depth |
| Expressive Power of Transformers with CoT (Merrill & Sabharwal) | 2023 / ICLR'24 | 2310.07923 | Log-many CoT steps add little power; linear steps reach regular languages; polynomial steps reach exactly P | CoT length must grow with graph size |
| A Little Depth Goes a Long Way | 2025 / NeurIPS'25 | 2503.03961 | Θ(log n)-depth uniform transformers express connectivity; scaling depth is more efficient than scaling width or CoT | Looped/deeper models are an alternative to long CoT |
| Transformers Provably Learn Connectivity … Only with the Right Data | 2025 / arXiv | 2510.19753 | An L-layer (disentangled) transformer has exact capacity diam ≤ 3^L. Beyond-capacity training graphs push SGD toward a **node-degree heuristic**; training only at or within capacity yields the matrix-powering algorithm and OOD generalization | Curricula should bound diameter relative to the compute budget |
| Transformers Provably Learn CoT with Length Generalization | 2025 / arXiv | 2511.07378 | (Found in search; abstract not read) unverified detail | — |
| Shortest-path generalization (Tong et al.) | 2026 / ICLR'26 | 2604.15306 | Empirical: spatial transfer OK, length scaling fails (recursive instability); RL ≤ SFT ceiling | See §3 |
| When Do Hallucinations Arise? (path reuse / compression) | 2026 / arXiv | 2604.03557 | Models next-token prediction as graph search. Hallucinations come from memorized paths overriding context (reuse) and multi-hop shortcuts (compression) (abs) | Gives a mechanism for "hallucinated edges" |
| Generalization Boundaries of fine-tuned SLMs | 2026 / arXiv | 2604.18092 | 3–4B models keep *ordinal* consistency on larger graphs, with architecture-specific degradation (abs) | Ranking survives scale better than exact answers |

## 7. Scale, latency, hallucinated paths, verification

| Paper | Year / venue | ID | Contribution | Note |
|---|---|---|---|---|
| GraphArena | 2024 / ICLR'25 | 2407.00379 | Scores outputs as correct, suboptimal, hallucinatory (well-formed but infeasible) or missing; 4 P + 6 NP-complete tasks | Top LLMs hallucinate infeasible paths on larger graphs (abs) |
| KG-Guard | 2026 / arXiv | 2606.00328 | Small GNN verifier that classifies LLM-proposed answer nodes in the KBQA subgraph | F1 82.0 / 87.4 / 84.3 on WebQSP / CWQ / PUGG with ~305× fewer params than an LLM judge (abs) |
| Conformal Path Reasoning | 2026 / arXiv | 2605.08077 | Path-level conformal calibration | +45% coverage rate, −52% set size vs conformal baselines (abs) |
| GCR (KG-Trie) | 2024 / ICML'25 | 2410.13080 | Constrained decoding: paths valid by construction | §4 |
| KGCache | 2026 / arXiv | 2608.07954 | 1-hop neighbourhood cache for ToG/RoG | Up to 1.91× retrieval speedup but only 1.06× end-to-end (abs), so the LLM is the bottleneck |
| LogosKG | 2026 / ACL'26 | 2604.18913 | Hardware-aligned k-hop retrieval on billion-edge KGs | Systems-side scaling (abs) |
| AtlasKV | 2025 / ICLR'26 | 2510.17934 | Parametric KG2KV injection of ~1B triples in <20GB VRAM | Avoids retrieval latency (abs); no path verifiability |
| GraphChain | 2025 | 2511.00457 | 200k-node graphs via tool chaining | Largest *reasoning* (not retrieval) scale I found with an accuracy number |

**On scale:** I found no paper that reports multi-step *algorithmic* reasoning accuracy on graphs with 10^6–10^7 nodes. The scale numbers I did find:
- **10^5 nodes:** GraphChain (200k nodes).
- **Freebase / Wikidata:** KGQA agents over full Freebase, but the questions need only 1–4 hops from given topic entities.
- **Corpus tokens:** BM25-at-scale (601M tokens), where graph indexes did not even finish building.
- **Billion edges:** LogosKG and AtlasKV, which are retrieval and storage systems.

---

## Open gaps (ranked; evidence → why still open)

1. **No evaluation of long-horizon algorithmic reasoning on graphs with 10^5–10^7 nodes.**
   - Evidence: G1 stops at 100 nodes because of context. GT-Bench and GraphArena use context-sized graphs. GraphChain reaches 200k nodes on its own benchmark. KGQA "large KG" work needs only 1–4 hops from oracle topic entities (SoG-R1 assumes oracle linking).
   - Why open: graph-in-context methods cannot scale, and tool methods are scored on shallow queries. Nobody pairs a huge graph with deep (≥10-hop, global) queries plus verifiable answers.

2. **RL does not obviously expand graph-reasoning capability beyond SFT, and its transfer claims rest on small graphs and the same encoding.**
   - Evidence: in the controlled study, RL ≤ best SFT and does not fix length scaling (2604.15306). In SoG-R1, SFT gives +50 pts and RL +1–2 (RL-only reaches only 63 on CWQ). Process rewards help on synthetic tasks but not on real ones (2506.00845). Fine-tuned G1-style models become *more* sensitive to format and structure (2511.10234).
   - Why open: existing transfer tests (GraphWiz, GraphArena, Cora) share a distribution with training. Nobody has run a held-out-encoding, held-out-size RL study with McNemar-style paired stats.

3. **Size/length generalization has a theoretical explanation but no practical recipe for LLMs.**
   - Evidence: capacity is diam ≤ 3^L, and beyond-capacity data induces a degree heuristic (2510.19753). Search tasks need wide, log-depth models (2405.18512). Recursive instability appears beyond training length, and ~1% slightly-longer data rescues it (2604.15306).
   - Why open: these results come from small from-scratch transformers. Nobody has tested "diameter-bounded curricula" or "slightly-longer augmentation" on pretrained LLMs doing tool-mediated search.

4. **Soft graph tokens are largely unused by the decoder, and reported GraphLLM gains are fragile.**
   - Evidence: sink tokens are PADs and removing them does nothing (2606.03712). Models rely on text (2605.03514). A trivially scored hop-distance diagnostic fails (2608.30437). Cheap baselines come within 2–3 pts, and evaluation leaks exist (2610.28714). Text-only node descriptions already match structural encodings (2511.16767).
   - Why open: there is no accepted test that a model *uses* topology, such as a shuffle-control or relabel-control. Most papers still report aggregate accuracy only.

5. **Encoding and permutation invariance is still unsolved, even for frontier reasoning models.**
   - Evidence: the best format varies with model × task × density (2609.12265, 2610.22734). Models fail isomorphism under relabeling (2606.09484). GRAIN's invariance reward halves the OOD gap but needs gold topology (2608.27142).
   - Why open: no training objective gives invariance without ground-truth graphs. Tool use sidesteps the problem only when parsing text into a graph is itself reliable.

6. **GraphRAG's value is conditional and unpriced at scale.**
   - Evidence: BM25 beats every graph index at shared tiers above ~10M tokens, and graph builds hit construction walls (2607.26497). GraphRAG hurts on simple facts and on math (2506.05690, 2506.02404). Retrieval gains do not translate to generation (2606.25656). No fixed traversal policy is optimal (2609.11065).
   - Why open: there are few controlled scaling studies (one, synthetic enterprise data) and no theory of *when* explicit structure adds signal beyond lexical ranking plus agentic refinement.

7. **Agent evidence management, not graph access, may be the real bottleneck, and it is little studied.**
   - Evidence: the GRA graph-free control shows gains come from selective access (2608.15834). Wikidata Traces attributes long-horizon failures to evidence management (2610.06650). GraphEcho shows agents re-walk redundant paths, and fixing that loses coverage (2609.17695). DKGI's self-written tools beat predefined graph APIs (2610.18616). Agent+BM25 beats the raw agent by 32 pts (2607.26497).
   - Why open: there are no benchmarks that separate graph-tool quality from context and evidence handling under a fixed token budget.

8. **Verification of reasoning paths against the graph is mostly post-hoc or KGQA-only.**
   - Evidence: GCR gives faithfulness by construction, but only for trie-able KG paths. KG-Guard and conformal methods verify answer nodes, not multi-step paths. GraphArena counts "hallucinatory" infeasible outputs, but no method trains against them at scale.
   - Why open: there is no cheap, general step-level checker (does edge (u,v) exist? is the path valid?) built into RL rewards for agentic exploration of large graphs. The closest are GraphCert's certified rubrics and SoG-R1's live-tool grounding.

9. **Latency and cost accounting is rare.**
   - Evidence: KGCache shows a 1.91× faster retrieval gives only 1.06× end-to-end, so LLM calls dominate. SoG-R1 is one of the few to report a reduction in tool calls (−4 to −15%).
   - Why open: most papers report accuracy only. Wall-clock under a fixed budget, which is what a sequential 12 h scorer cares about, is almost never measured.

---

## Transferable to code graphs

**Plausibly transferable:**
- **Two-tool breadth/depth agents (ARK 2601.13969; GRA 2608.15834):** global lexical search + 1-hop neighbour expansion maps directly onto repo graphs (grep/BM25 over symbols + callers/callees/imports). GRA explicitly frames KG tools as ls/cat/grep.
  - Blocker: GRA's own control says the gain is selective access, not topology. Expect a code graph to help mainly when the issue needs multi-file/call-chain hops. "Code Isn't Memory" (2606.22417) finds the index pays off on multi-file changes. Deterministic anchoring (2606.26979) gives only +2.2 pp Func@5, mostly by making navigation reproducible.
- **"Retrieval before agency" (2607.26497):** run global lexical ranking first, then agentic graph expansion from the top hits. This matches the earlier finding in this project that grep/sed reads helped. It also argues against graph-only localizers.
- **Batched 1-hop tool + turn-penalty RL (SoG-R1 2607.18481):** one `neighbors(symbols[], direction, edge_types)` tool with paged output and a call-count penalty fits a 5,000-char output cap and a growing, uncompacted context.
  - Blocker: SoG-R1 depends on gold SPARQL to scaffold teacher traces. For code, the analogous scaffold is the gold patch's touched symbols. Paths from the issue to those symbols could be computed offline from the call graph to generate SFT traces, much as LocAgent fine-tunes Qwen-Coder-32B on such traces.
- **Diameter/length-bounded curricula (2510.19753, 2604.15306):** keep training trajectories within the hop depth the model can reliably compose, plus ~1% slightly longer ones.
- **Verification by construction (GCR KG-Trie 2410.13080; live-tool grounding):** constrained decoding of symbol names to ones that exist in the repo graph would directly address hallucinated function and file names.
- **Agents writing their own graph tools (DKGI 2610.18616):** on code, the agent can write ad-hoc Python over AST/import graphs.
  - Blocker: this needs reliable code execution in the sandbox.

**What blocks transfer:**
1. **Name resolution.** Graph tools keyed by fully qualified names fail on bare names and phrases (the project's own finding about `search_similar_code`). Every successful KG agent here pairs graph hops with a *lexical/global search* entry point (ARK, Agent+BM25, SoG-R1 with given topic entities).
2. **Tool-call format reliability** in small or quantized models. GRA notes the effect "depends on a model able to drive tools reliably", and GraphEcho shows redundant re-walks. These match the garbled-argument and repeat-loop failures already seen in this project.
3. **Encoding sensitivity.** Serialized subgraphs fed into context inherit the format brittleness in §1. Keep graph outputs small, structured and consistent.
4. **Fine-tuning can raise format sensitivity** (2511.10234). A LoRA trained on one tool-output format may break if the scorer's graph-tool output format differs.
5. **Construction cost and staleness** do not apply to static parsers (tree-sitter graphs are cheap; Codebase-Memory 2603.27277 builds for 66 languages). But a Codebase-Memory-style graph trades accuracy for tokens: 83% vs 92% answer quality at 10× fewer tokens.

---

## Sources (beyond arXiv IDs above)
- GraphPO / GraphPRM search: https://arxiv.org/abs/2503.00845v1, https://papers.cool/arxiv/2606.18954
- Text2Cypher resources: https://huggingface.co/papers/2412.12612, https://researcher.ibm.com/publications/mind-the-query-a-benchmark-dataset-towards-text2cypher-task, https://neo4j.com/blog/developer/benchmarking-neo4j-text2cypher-dataset/, https://www.arxiv.org/pdf/2406.10710
