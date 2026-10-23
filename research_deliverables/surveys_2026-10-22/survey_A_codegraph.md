# Survey A: Code-graph generation, parsing and embedding for repository comprehension (2024 to Oct 2026)

Compiled 2026-10-22. Sources: alphaXiv `discover_papers` (about 12 queries) plus `get_paper_content` (full text or AI report) for the papers marked **[read]**. Papers marked **[abs]** were seen only as abstracts or search snippets. Numbers come from the papers themselves. Anything I could not confirm is marked "unverified". All IDs are arXiv IDs (https://arxiv.org/abs/<ID>).

Caveat on dates: several 26xx IDs are very recent preprints (Aug to Oct 2026) and have not been peer reviewed. The "ICLR 2027" header on 2608.26602 is a template and is not evidence of acceptance.

---

## 1. Table

### 1a. Graph construction and graph-as-tool for agents

| Paper | Year / venue | ID | Method (one line) | Key quantitative result | Limitation |
|---|---|---|---|---|---|
| **RepoGraph** [read] | 2024; ICLR 2025 | 2410.14684 | tree-sitter line-level def/ref graph (invoke + contain edges), filters builtins and 3rd-party calls; the k-hop ego-graph is flattened into the prompt or exposed as an agent action | SWE-bench Lite: +2.0 to +2.7 pp absolute across RAG, Agentless, AutoCodeRover and SWE-agent; Agentless+RepoGraph 29.67%. 1-hop is best; flattening the 2-hop graph hurts | Python only. Gains are small in absolute terms. Name-based resolution. GPT-4/4o only |
| **CodexGraph** [abs] | 2024; NAACL 2025 | 2408.03910 | Static analysis writes the repo into a graph DB (Neo4j); the LLM writes Cypher queries | Not read in detail (unverified) | Needs a full scan and a persistent DB; the LLM must write a query language |
| **LocAgent** [read] | 2025; ACL 2025 | 2503.09089 | Heterogeneous graph (dir/file/class/func; contain/import/invoke/inherit) + sparse hierarchical entity index (ID dict + BM25 on IDs and on content); tools SearchEntity / TraverseGraph / RetrieveEntity; SFT of Qwen2.5-Coder 7B/32B on 768 trajectories | SWE-bench Lite file Acc@5 94.16%, function Acc@10 77.37% (Claude-3.5). Fine-tuned 32B is comparable at 86% lower cost. Ablation: dropping TraverseGraph or multi-hop hurts function-level accuracy most; dropping BM25 hurts all metrics | Python. Localization is evaluated more than resolution (+12% Pass@10 downstream). LOC-BENCH repos are popular projects |
| **OrcaLoca** [abs] | 2025 | 2502.00350 | Priority-scheduled agent actions + code-graph search for localization | Used as the SOTA baseline by RPG-Encoder (beaten by 14.4 pp function Acc@5) | Not read |
| **CoSIL** [abs] | 2025 | 2503.22424 | LLM-driven iterative code-graph search (module call graph, then function call graph) | Baseline in RPG-Encoder (beaten by 11.6 pp on SWE-bench Live Lite) | Not read |
| **GraphLocator** [abs] | 2025 | 2512.22469 | Graph-guided causal reasoning for issue localization | Unverified | Not read |
| **RPG** (Repository Planning Graph) [abs, checked on arxiv.org] | 2025 | 2509.16198 | Unified graph of capabilities, files, data flows and functions as a generation blueprint (ZeroRepo) | RepoCraft (6 projects, 1,052 tasks): 81.5% coverage, 69.7% test accuracy, +27.3 / +35.8 pp over Claude Code | Generation-side. Graph built by LLM planning |
| **RPG-Encoder** [read] | 2026 | 2602.02084 | Lifts an existing repo into an RPG: LLM "semantic lifting" (verb-object feature per function), LLM reorganisation into a functional hierarchy, AST dependency edges; incremental update from git diffs; tools SearchNode / FetchNode / ExploreRPG | SWE-bench Verified function Acc@5 93.7% (Claude-4.5-Sonnet), +14.4 pp over OrcaLoca. SWE-bench Live Lite 87.8% (GPT-5). Incremental maintenance 633K vs 14.7M tokens (−95.7%). Ablation: removing features drops function Acc@1 50.5→43.1 | LLM-extracted features cost tokens and can hallucinate. Incremental update is checked for "statistical parity", not exact equivalence. Localization only. Frontier models only |
| **Codebase-Memory** [read] | 2026 | 2603.27277 | tree-sitter knowledge graph for 66 languages in a single SQLite file; 6-strategy call resolution with confidence scores (import map 0.95 … fuzzy 0.30–0.40); LSP-style type resolution for Go/C/C++; Louvain communities; 14 MCP tools | vs a grep/read agent (Opus 4.6, 31 languages): 83% vs 92% answer quality at ~10x fewer tokens and 2.1x fewer tool calls. Django indexes in ~6 s, Linux in ~3 min. Worst case macro-heavy C: 0.58 vs 1.00 | Lower answer quality than plain exploration. Not evaluated on issue resolution. Python call resolution is name-based (no type resolution) |
| **CodeAnchor** ("How much static structure…") [read] | 2026 | 2606.26979 | Injects PyCG call/inherit/import edges as structured **comments inside the source**, so a grep-first agent sees them with no new tool. Variants: Topo, Dense (+config/data-flow), Inv (inverse-only) | Func@5 +2.2 pp on Lite (McNemar p=0.041), +1.2 to 1.4 pp on Verified. −1.5 to 1.6 rounds. Run-to-run std roughly halved. Dense tags give no accuracy gain and +14–19% tokens. Inverse-only is better on large hub-heavy repos (>100k LOC) and worse on medium ones | Single agent (GPT-5.1-codex). Python. Mostly localization; the repair gain (+12.5 pp) is on a disagreement subset only |
| **ARISE** [read] | 2026 | 2605.03117 | Python `ast` graph + statement nodes + **intra-procedural def-use edges**; `get_dataflow_slice` tool; token-budgeted context bundling; plugged into SWE-agent with Qwen2.5-Coder-32B | SWE-bench Lite: file/func/line R@1 67/60/41 vs 57/43/26 for SWE-agent. Pass@1 22.0% vs 17.3%. Slicing alone adds +7 pp func R@1 and +2 pp Pass@1. The "Coarse" control (same tool schema, empty slices) gives no gain, so the data matters, not the extra tool. NL-summarised slices give no gain | Intra-procedural only; most residual failures are wrong-file (the root cause is in a callee). Python. 300 tasks |
| **SeeRepo** ("LLM Agents Can See Code Repositories") [read] | 2026 | 2606.14061 | AST graph (contains/imports/inherits/invokes) rendered with Graphviz as images for multimodal LLMs | Vision-only: −13.6 to −34.1 pp. Hybrid text+image: +0.4 to +1.8 pp (one model −2.2) at 3–46% lower cost. Graph tool used during **repair** hurts (−5.0 pp); used during localization it helps | Gains in accuracy are within noise. Structure is only useful early in the trajectory |
| **CodeNib** [read] | 2026 | 2607.25431 | Multi-view system (lexical / dense / structural graph + static vs live LSP) with incremental maintenance and a manifest per commit | One-hop graph expansion of dense seeds: ΔFile Success@10 from −4.8 to +7.1 pp, **all CIs include 0**. Static index vs live LSP: 63.2% agreement (87.4% for definitions, **39.0% for references**). Symbol-level incremental graph repair **matches a full rebuild in only 15/33 (45.5%)** transitions. Eager+compact context: 50–87% of grep/read tokens with recall preserved (includes Gemma 4-12B) | Localization only. No end-to-end resolve rate |
| **Repository Intelligence Graph (RIG)** [abs] | 2026 | 2601.10112 | Deterministic architectural map built from build systems (multilingual build/test structure) | Unverified | Not read |
| **The Thousand-Graph Hypothesis** [read] | 2026 | 2608.26602 | Argues that only entities need to persist and that relations are "materialised" by attention; two-layer entity index with zero prebuilt edges | DeepSeek-V4-Flash on SWE-bench Verified: 92.1 → 94.2 → 95.6% "success rate" (the metric is undefined; treat as **unverified / suspicious**) | One model. Mechanism not probed. Single author. Deployment evidence is anecdotal |
| **FeatLens** [abs] | 2026 | 2609.26480 | Feature-guided dynamic code-graph construction for repo-level generation | Unverified | Not read |
| **Partial Dependency Graph retrieval** [abs] | 2026 | 2608.01927 | Builds only the dependency subgraph needed for retrieval in repo-level generation | Unverified | Not read |
| **LARGER** [abs] | 2026 | 2605.16352 | Lexically anchored repository-graph exploration for localization | Unverified | Not read |

### 1b. Graph-integrated models, embeddings, adapters

| Paper | Year / venue | ID | Method | Key result | Limitation |
|---|---|---|---|---|---|
| **Code Graph Model (CGM)** [read] | 2025 (NeurIPS 2025 per repo; unverified) | 2505.16901 | Repo graph (REPO…ATTRIBUTE nodes; contain/call/import/extend edges). Each node's text goes through a CodeT5+ encoder and MLP adapter into **one node token per 512-token chunk**. **Graph-aware attention mask** among node tokens. Subgraph→code reconstruction pretraining, then noisy fine-tuning. Qwen2.5-72B + LoRA; agentless Graph-RAG pipeline (rewriter/retriever/reranker/reader) | SWE-bench Lite 43.0%, Verified 50.4% (best open-weight at the time). Removing the graph mask: −8.6 / −5.6 EM on completion. Removing reconstruction pretraining: −7.65 EM. Removing the reranker: −24.67 pp (pipeline matters more than the graph) | Needs custom attention masks and an encoder, so it cannot be served as a plain PEFT LoRA in stock vLLM. Gains are dominated by the RAG pipeline. Trained on SWE-bench-train repos (memorisation not controlled) |
| **CGBridge** [read] | 2025 | 2512.07666 | GNN (graph transformer) over a CPG (AST+CFG+DFG); Q-Former-style bridge produces a **soft graph prompt** for a frozen LLM; contrastive + edge-type pretraining | Summarisation LLM-judge +16% relative; translation execution accuracy 89.0% vs 70.6% (1.5B). **Renaming robustness: −1.6% vs −15 to −28% for LoRA/text baselines** | Function-level snippets only. Not repo-level, not agentic |
| **HGAdapter** [abs] | 2025 | 2510.17591 | Hypergraph-based adapters in PLMs for summarisation and clone detection | Unverified | Encoder-scale PLMs |
| **Code2LoRA** [read] | 2026 | 2606.06492 | Hypernetwork maps a repo embedding (Qwen3-Embedding-0.6B, mean+max pooled files) to a LoRA. "Evo" variant: a GRU over diff embeddings keeps the adapter current | Assertion completion cross-repo EM 63.8% vs 53.9% (FFT+RAG). Matches per-repo LoRA without per-repo training. RAG k=3 only 39.7%. <10 ms per adapter, zero extra tokens | Qwen2.5-Coder-1.5B, a single task (assertions). Repo encoding is a structure-free bag of files |
| **SpIDER** [read abstract] | 2025 | 2512.16956 | Dense retrieval + graph-based candidate expansion (containment, call edges) with an auditable "reason for inclusion"; SpIDER-Bench covers Python/Java/JS/TS | Recall@20 +13% relative via containment, **+27% relative (+0.11 to +0.19 abs) with call edges**, on every language | Retrieval-only metric; conflicts with CodeNib's null result for one-hop expansion |
| **jina-code-embeddings** [abs] | 2025 | 2508.21290 | Code embeddings from autoregressive code-generation backbones | Unverified | Snippet retrieval |
| **C2LLM** [abs] | 2025 | 2512.21332 | Contrastive code LLM embeddings (0.5B/7B, Qwen2.5-Coder) with PMA cross-attention pooling | Unverified | Text-only |
| **ExecRetrieval** [abs] | 2026 | 2609.01865 | Benchmark measuring the functional-correctness gap of code-embedding retrieval | Unverified | |
| **Identifier-Free Code Embedding Models** [abs] | 2026 | 2605.05251 | Embeddings that do not rely on identifiers (binary reverse-engineering setting) | Unverified | Binary domain |

### 1c. Structure-aware pre-, mid- and fine-training

| Paper | Year | ID | Method | Key result | Limitation |
|---|---|---|---|---|---|
| **AST-FIM** [read abstract] | 2025 | 2506.00204 | Masks complete AST subtrees during FIM pretraining; Real-FIM-Eval (30k commits, 12 languages) | Up to +5 pts over random-character FIM (1B/8B) | Pretraining scale. Not agentic |
| **Function-aware FIM mid-training** [read] | 2026 | 2607.12463 | PDG (call + sibling edges) selects whole-function (or 2–3 coupled functions) masks scored by complexity × inferability; CoT inside the FIM middle; 2.6B-token mid-training **before** agentic SFT; SWE-bench repos decontaminated | SWE-bench Verified +2.8 (7B), +3.0 (14B), +5.3 (SWE-Smith), +3.2 (Qwen3-8B). Multi-function same-file tasks 13.6→22.7%. "No-patch" failures fall by about 10x. Restores LiveCodeBench (+11.1) lost in post-training. **Without CoT still +1.18**. PDG+H+I selection beats random masking | Python only. Full-weight mid-training (not LoRA). Uses Gemini-3-Flash for CoT |
| **Neural debugger / execution traces** [abs] | 2026 | 2603.09951 | Training on Python execution traces for line-by-line execution prediction | Unverified | |
| **Self-Execution Simulation** [abs] | 2026 | 2604.03253 | Models simulate execution of their own code | Unverified | |
| **Dataflow-aware pretraining for slicing** [abs] | 2026 | 2604.26961 | LM static slicing with dataflow-aware pretraining + constrained decoding | Unverified | |

### 1d. Robustness: renaming, memorisation vs structure

| Paper | Year | ID | Method | Key result | Limitation |
|---|---|---|---|---|---|
| **SchrodingerRepo** [read] | 2026 | 2609.27891 | Instantiates a semantically equivalent repo at eval time: problem restatement, namespace remapping, intra-file reordering, functionality-preserving rewrites; SWE-bench Verified + SWE-QA | Resolve rate **−6.0 to −14.4%** (p<0.05). Input tokens up to +59%. >80% of the extra actions are exploration. **On post-cutoff (SWE-rebench Mar 2026) tasks, Pass@1 is unchanged (17.27%)**, so the drop reflects familiarity and the transform is not simply harder | Frontier models only. No graph-tool arm |
| **Jagged Frontier** [read abstract] | 2026 | 2608.18389 | Semantics-preserving transformations (control-flow rewrites, dead code, renaming) on SWE-bench Verified/Pro; 2 scaffolds × 4 models, paired repeated runs | Up to −6.7 pp; significant in 6/16 configs. Robustness ranking flips across scaffolds. The simpler scaffold (mini-SWE) is more robust | Effect sizes are small. Includes Qwen 3.6-27B |
| **When Names Disappear (ClassEval-Obf)** [read] | 2025 | 2510.03178 | Graded obfuscations (alpha-renaming, ambiguous, cross-domain, misleading names) | Even **execution prediction** drops: ClassEval Pass@1 85.7→76.1; LiveCodeBench 85.4→71.2; DeepSeek-V3 90.0→69.3 under ambiguous names | Function/class level |
| **Semantics-preserving mutations** [abs] | 2025 | 2505.10443 | Robustness of LLM code understanding to mutations | Unverified | |
| **Lexical superstition** [abs] | 2026 | 2609.26388 | Re-evaluates comprehension on low-lexical-quality code | Unverified | |
| **Detecting functional memorisation** [abs] | 2026 | 2606.12764 (Meta) | Functional (non-verbatim) memorisation audit | Unverified | |
| **Memorisation diagnostics should be scale-aware** [abs] | 2026 | 2608.12771 | | Unverified | |

### 1e. Benchmarks for repo comprehension

| Paper | Year | ID | What it measures | Key result | Limitation |
|---|---|---|---|---|---|
| **SWE-QA** [abs] | 2025 | 2509.14635 | Repo-level QA on SWE-bench-style repos | Unverified | Public repos |
| **SWE-QA-Pro** [abs] | 2026 | 2603.16124 | Long-tail repo QA + training recipe | Unverified | |
| **Code-QA-Bench** [read] | 2026 | 2605.29277 | Answer-first QA with three conditions: closed-book / code-only / documented | On code-derivable tasks, **closed-book already scores 0.44–0.56** vs ~0.86 with code. Docs add almost nothing (+0.006). DeepSeek-V4-Pro has the biggest code-reading gain (+0.45) | 10 SWE-bench repos, so memorisation is built in. LLM judge |
| **RepoProbe** [read] | 2026 | 2608.04783 | Architecture "how/why" QA from GitHub Discussions (repos created ≥2024) with checklist grading | Best overall ~62.7% (GPT-5.2). Perfect-solve ≤27.5%. "Edit bias" causes 10–24% of failures. Checklist grading halves judge variance | Repos are recent but public. LLM-generated checklists |
| **CABRA** ("Code understanding is a bottleneck") [read abstract] | 2026 | 2610.10610 | Synthetic call-graph-transformation tasks that scale traversal / search / runtime resolution | Bare LLMs degrade with size; **agents stay near-perfect by offloading to grep**. Tool-call count predicts SWE-bench Verified success better than lines edited | Synthetic |
| **SWE-Explore** [abs] | 2026 | 2606.07297 | Benchmarks how agents explore repos | Unverified | |
| **CORE-Bench (code retrieval)** [abs] | 2026 | 2606.11864 | Code retrieval for agentic coding | Unverified | |
| **Agent Retrieval Bench** [abs] | 2026 | 2607.24882 | Context-acquisition stage of coding agents | Unverified | |
| **RepoReasoner** [abs] | 2026 | 2607.25996 | Repo-level reasoning of long-context LMs | Unverified | |
| **Runtime-behaviour repo benchmark** [abs] | 2026 | 2609.28449 | Dynamic (runtime) reasoning at repo level | Unverified | |
| **LOC-BENCH** (in LocAgent) [read] | 2025 | 2503.09089 | Post-cutoff issues; bug/feature/security/perf mix | See LocAgent row | Popular repos |

### 1f. Adjacent (retrieval interface, not graph-specific)

| Paper | ID | Note |
|---|---|---|
| Is Grep All You Need? [read] | 2605.15184 | On LongMemEval (not code), inline grep beats vector search in every harness/model pair. The harness changes accuracy by up to ~16 pp for the same model. File-based delivery can collapse accuracy (Codex 93.1→55.2). Shows that the interface dominates. |
| The Devil Is in the Interface [abs] | 2608.11386 | Tool architecture shapes coding-agent behaviour (unverified details) |
| CodeGrep (RL-trained retrieval agent) [abs] | 2608.05886 | Unverified |
| Tool-integrated RL for Repo Deep Search [abs] | 2508.03012 | Unverified |
| One Tool Is Enough (RL repo agents) [abs] | 2512.20957 | Unverified |

---

## 2. Cross-cutting observations

1. **Graph benefit is real but small once a strong agent exists, and it shows up mainly as efficiency and stability.** RepoGraph +2–2.7 pp; CodeAnchor +1–2 pp Func@5 with variance halved; SeeRepo +0.4–1.8 pp at 25–46% lower cost; Codebase-Memory 10x fewer tokens at *lower* quality; CodeNib one-hop expansion not significant. Large localization numbers (LocAgent 94%, RPG-Encoder 93.7% Acc@5) are reported on SWE-bench repos, which are contaminated.
2. **What matters is how structure is delivered, not that a graph exists.** ARISE's "Coarse" control shows the same tool schema with empty data gives nothing. CodeAnchor delivers structure through grep, with no new tool. SeeRepo shows structure hurts during repair. CGM's ablation shows the reranker matters more than the graph mask.
3. **Fine-grained semantics matter on demand but not eagerly.** ARISE: on-demand intra-procedural slicing is the largest single gain (+7 pp func R@1). CodeAnchor: eagerly injected data-flow/config tags add nothing and cost 14–19% tokens.
4. **Memorisation is large and measurable.** Closed-book models answer about 50% of code-derivable QA on SWE-bench repos (Code-QA-Bench). Resolve rates fall 6–14% under namespace and layout remapping, but not on post-cutoff tasks (SchrodingerRepo). Renaming hurts even execution prediction (When Names Disappear). Graph-integrated encoders are much more renaming-robust (CGBridge −1.6% vs −15 to −28%), but only at snippet level.

---

## 3. Open gaps (ranked; evidence and why each is still open)

**G1. No one has measured the interaction between graph tools and repository familiarity.** All graph-for-agents papers (RepoGraph, LocAgent, CGM, RPG-Encoder, ARISE, CodeAnchor, SeeRepo) evaluate on SWE-bench / Live / Lite, built on the same ~12–40 public repos. SchrodingerRepo shows memorised cues are worth 6–14% and that removing them inflates *exploration* (>80% of the extra actions). That is exactly the phase graph tools speed up (SeeRepo: structure helps only during localization). Prediction nobody has tested: the marginal value of structure is larger on unseen or remapped repos than on SWE-bench, so the published +1–3 pp is a lower bound. No paper crosses {graph tool on/off} × {canonical / remapped or held-out repo}. *Why open:* the remapping benchmarks are new (Aug 2026) and graph papers predate them.

**G2. The value of graphs under tight context and step budgets is untested.** The positives are efficiency: 25–46% fewer tokens (SeeRepo), 10x (Codebase-Memory), 1.5 fewer rounds and half the variance (CodeAnchor), 50–87% of grep tokens (CodeNib). Accuracy gains are near zero for frontier models with 250-step / $3 budgets. Nobody converts these savings into accuracy under a hard cap (e.g., 32k context, no compaction, fixed wall-clock), where the token savings should matter most. CodeNib includes Gemma 4-12B and Qwen3.5-9B but measures only localization recall. *Why open:* benchmarks run frontier models with generous budgets.

**G3. Delivering structure to weak tool-callers without new tools.** Graph tools require entity IDs or fully qualified names (LocAgent adds BM25 fuzzy fallback because exact names fail; ARISE uses TF-IDF). Small models garble tool arguments. CodeAnchor injects edges as source comments read via grep, which needs no new tool and no name resolution, but it was tested only with GPT-5.1-codex. ARISE reports that a 32B model reads raw structured slices fine without NL summaries. Untested: comment-injected or grep-visible structure (`# called_by: …`) for ≤30B open models, and whether it beats explicit graph tools when tool-call reliability is the bottleneck. *Why open:* each line of work assumes either strong models or fine-tuned tool use.

**G4. On-demand inter-procedural data flow for dynamic languages.** ARISE's slicing is intra-procedural yet gives the largest gain, and its residual failures are mostly "wrong file: root cause in callee". CodeAnchor's dense data-flow tags (pattern-matched, eager) saturate. No agent paper offers on-demand *inter*-procedural def-use or slicing for Python, even though analyzers now exist (PyFlow 2608.07026 and CPyGraph 2609.25083, both [abs], unverified). *Why open:* soundness and cost of inter-procedural analysis in Python. The ARISE vs CodeAnchor contrast suggests a query-time tool, not eager injection, is the right interface.

**G5. Graph freshness inside a trajectory, and the cost of incorrect incremental graphs.** CodeNib: symbol-level incremental repair matches a full rebuild in only 45.5% of transitions, and static references agree with live LSP only 39% of the time. RPG-Encoder checks only "statistical parity". Codebase-Memory re-indexes on file change but never validates correctness. All issue-resolution papers build the graph once on the base commit, even though the agent then edits code. Nobody measures how often a graph query after an edit returns stale edges, or whether that misleads the fixer. *Why open:* the evaluation pipelines are localization-centric.

**G6. How graph precision and recall in dynamic languages affect downstream results.** Call resolution is heuristic: Codebase-Memory has confidence tiers down to 0.30 (fuzzy); CGM uses conservative CHA; CodeAnchor uses PyCG (precision over completeness). CodeAnchor *claims* LLMs treat tags as soft hints and are robust to unsoundness, but this is not measured. Missing: controlled edge-noise or edge-dropout experiments (false-positive vs false-negative edges) against a dynamic-trace ground truth. *Why open:* building ground-truth call graphs for Python repos requires running the tests, which most graph papers avoid.

**G7. Structure-aware training that ships in standard serving stacks.** CGM needs a custom encoder and graph attention mask (not PEFT-servable). CGBridge needs a soft-prompt bridge. Code2LoRA generates LoRAs but from a structure-free embedding, on a 1.5B model and one task. Function-aware FIM mid-training (PDG-selected masks) gives +2.8 to 5.3 pp SWE-bench Verified, but it is full-weight mid-training on 2.6B tokens. Untested: whether the PDG-FIM objective (or call-graph-neighbour prediction) still helps as a *LoRA on an instruction-tuned model*, and whether its "no-patch failures fall ~10x" effect transfers at LoRA scale. *Why open:* the papers target base-model builders, not adapter-only settings.

**G8. Graph-integrated representations have not been tested for renaming robustness at repo level.** CGBridge (GNN soft prompt) is nearly invariant to renaming (−1.6%) while text models drop 15–28%, but only on snippets. CGM (repo-level graph tokens) reports no renaming or held-out-repo test. Since memorisation is the confound in G1, a graph-token model should in principle be the de-memorised option, and nobody has checked. *Why open:* the evaluation suites differ (snippet vs SWE-bench).

**G9. No comprehension benchmark is both unseen and structurally deep.** Code-QA-Bench: 10 SWE-bench repos, closed-book ≈0.5. SWE-QA and SchrodingerRepo QA use public repos. RepoProbe uses repos created ≥2024 (better), but they are public and its questions come from Discussions, so structural depth is not controlled. CABRA controls depth synthetically, but agents solve it trivially with grep. Missing: tasks on held-out or private repos where answers require ≥2-hop call/data-flow reasoning that grep cannot short-circuit. LocAgent shows accuracy falls with hop distance but stops at "hops in SWE-bench", which are mostly 1–2 (as CodeAnchor notes).

**G10. Retrieval embeddings and graph expansion give contradictory evidence.** SpIDER: call-edge expansion +27% relative Recall@20 on every language. CodeNib: one-hop expansion CI includes zero across embedding models. The difference is probably in seed quality, edge type (call vs semantic reference) and benchmark, but no paper reconciles them. Code embedding models (C2LLM, jina-code) are trained text-only. No repo-level contrastive embedding uses graph-neighbour positives and is evaluated under renaming (identifier-free embeddings exist only for binaries, 2605.05251).

**G11. The claim that relations need not be prebuilt is untested mechanistically.** The Thousand-Graph Hypothesis (2608.26602) argues attention materialises task-specific edges, so external edges are unnecessary. Its evidence is one model with an undefined 92–95% "success" metric. A probing study (do attention patterns recover call edges between in-context entities, and does adding explicit edges change the model's behaviour?) would settle whether explicit edges add information or only routing. Treat the paper's numbers as unverified.

---

## 4. Directly actionable reading for an offline small-model SWE agent (my synthesis, not from any one paper)
- Structure should be **grep-visible text** (CodeAnchor) or a **single on-demand slice tool** (ARISE). It should not be a family of graph tools keyed by fully qualified names (the repo's own finding that `search_similar_code` returns 0 for bare names matches LocAgent's need for a BM25 fallback).
- Use structure in **localization only** (SeeRepo: −5 pp when it is used during repair).
- Use **inverse-only links** (`called_by`) in large hub-heavy repos and bidirectional links in medium ones (CodeAnchor).
- For training: PDG-selected whole-function FIM with CoT is the only structure-aware objective with SWE-bench gains after decontamination (2607.12463).
