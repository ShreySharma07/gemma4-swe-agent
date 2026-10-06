# Deep research: 2025–26 work on code graphs, code comprehension and LLM graph reasoning

Written 28 Sep 2026. Covers three threads: (1) code graphs for SWE agents, (2) code comprehension and structure-aware training, (3) general LLM graph reasoning. Each thread lists what was done, the gaps, how those gaps map onto our A1/A2 findings, and what we could change in the architecture.

**Venue honesty.** The top-venue papers in this space are mostly from 2025: RepoGraph (ICLR'25), LocAgent (ACL'25), CodexGraph (NAACL'25), CGM (NeurIPS'25), SWE-smith (NeurIPS'25 D&B), and CodeAnchor (ISSTA'26). Most 2026 work is still on arXiv, including the most relevant items (GRAIN, Schrödinger, SkillForge, SHERLOC, SemNav, AGT). I treat those as the live frontier and note wherever a claim comes from a preprint.

---

## 1. What the field did

### 1a. Code graphs and SWE agents
| Work | What | Key result | Relevance to us |
|---|---|---|---|
| RepoGraph (ICLR'25), CodexGraph (NAACL'25), LocAgent (ACL'25) | Give agents a directed, typed code graph plus graph tools | Plug-in gains on SWE-bench | Assumes a directed, typed graph. Ours is undirected with a single label. |
| CGM (NeurIPS'25) | Graph-aware attention through an architectural adapter | Strong at 72B, ~4% at 7B | Not allowed in the competition (LoRA only); needs scale. |
| CodeAnchor (ISSTA'26) | Passive structure injection vs optional graph tools | Agents rarely call optional graph tools; passive injection wins; inverse-only edges help in hub-heavy repos | Directly supports "inject, don't offer". |
| Code Isn't Memory (arXiv Jun'26) | Causal ablation of a structural index (BM25 + vector + call graph) inside one harness | Loc acc@5 44→85, resolve 41.9→50.4 (p=0.003); biggest gains on 3+-file fixes | Structure helps when retrieval is good. Our tools aren't good yet. |
| SHERLOC (NVIDIA, arXiv Jun'26) | Training-free localizer; outputs a **structured finding** (location, root cause, solution idea, …) | +5.95 pp average resolve when injected into repair agents; weaker models gain most; shuffled findings hurt in 9/10 cells; **~58% of recall is reachable from issue text alone** | Localizer/editor split plus a finding format. Gemma-31B-QAT is in the "weaker model" regime. |
| SemNav (arXiv Sep'26) | Semantic Cards plus a Candidate Workspace, on Gemma 4B | — | Overlaps our old "behavioural edge labels" idea. |
| SeeRepo / DualView / RepoAtlas (arXiv '26) | Repo graph as an image (visual modality) | — | Gemma 4 is multimodal, but images of graphs are an expensive detour at 32k context. |
| GraphLocator, CoSIL, RPG-Encoder | Causal issue graphs, iterative graph search, semantic feature graphs | — | Entry grounding and graph construction are crowded. |
| SkillForge (arXiv Aug'26) | Synthesizes issues from the repo's own tests, solves them, distils **entity-grounded skills**, injects them just in time when the agent opens a file | +5.8 pp (Verified); **skills are model-specific** (72.2 with own skills vs 65.2 with another model's) | Project-specific knowledge pays off. We must self-distil with Gemma, not a teacher. |

### 1b. Code comprehension and structure-aware training
| Work | What | Key result | Relevance to us |
|---|---|---|---|
| Schrödinger's Repo (arXiv Aug'26) | Invertible namespace remapping, layout reorder and rewrite of SWE-bench repos | Pass@1 **−6 to −14 pp**; namespace remap hurts most; **82–84% of extra actions go to exploration**; no drop on post-cutoff tasks, so the drop is memorisation | Agents localise by familiar names. The mapping tool is reusable. |
| Jagged Frontier (arXiv Aug'26) | 14 semantics-preserving transformations, incl. keyword decoys | Up to −6.7 pp; keyword decoys derail grep-style localisation | Same message: lexical localisation is brittle. |
| Face/Off "Lexical Superstition" (arXiv Sep'26) | Alpha-renaming across code-comprehension tasks | Lexical overemphasis is pervasive; **prompting and fine-tuning on anonymised data do NOT close the gap** | Plain renaming augmentation is not enough. Structural supervision is needed. |
| Function-aware FIM mid-training (arXiv Jul'26) | PDG-based fill-in-the-middle to learn "backward" integration of tool returns | Helps coding agents | Nearest to "graph literacy", but generic code filling, not graph navigation. |
| SWE-ZERO / SWE-smith / trajectory-curation study | Execution-free SFT scales; synthetic bug injection; for LoRA, **quantity beats quality below ~1k trajectories**; best filter = low error-retry | Tool-call format is a failure mode, so use constrained decoding | Shapes our data recipe. |
| Agentic code reasoning (semi-formal) | Structured reasoning templates for localisation | Internalising them via post-training is listed as future work | Template for the finding / chain format. |

### 1c. General LLM graph reasoning (graph-ML venues)
| Work | What | Key result | Relevance to us |
|---|---|---|---|
| **GRAIN** (arXiv Aug'26) | LLM parses text → explicit graph IR → tool; RL with a **structure-invariance reward** (Jaccard of the emitted graph vs ground truth) under **isomorphic renaming** | Without the structure reward: OOD gap 1.1%→11.1%, exact graph recovery 97%→63%. SFT OOD gap 15.8 → 7.8 with RL. **4B model.** | The recipe we transplant to code. Nobody has done this for repositories. |
| GraphDancer (arXiv '26) | PPO then DPO with a curriculum by structural difficulty (S-rounds = singleton lookups, E-rounds = neighbourhood expansions) | 3B beats 14B baselines cross-domain. Metrics: **Evidence Hit vs conversion P(correct given evidence seen)**. Residual errors are ~50% **premature stop** | Evaluation lens for Map 2, and a curriculum for our data. |
| Agentic Graph Tokens (arXiv Aug'26) | Model requests graph views, and a GNN encodes them as soft tokens mid-reasoning | Soft graph tokens beat text by +3 in-domain and **+29 zero-shot**; 2.8 GPU-h | Strong, but needs a GNN encoder + LLM joint training. Can't ship. Paper-only at best. |
| When Graph Tokens Sink (arXiv Jun'26) | Mechanistic study of graph-token LLMs | Salient graph tokens are not the information carriers; topology is poorly used | Caution on the soft-token route. |
| G1, GraphWiz, Graph-R1, GraphPO, AgentGL | RL/SFT on synthetic graph tasks | Transfer to real tasks is weak unless the structure is verified | Supports GRAIN's point: reward the structure, not just the answer. |

---

## 2. Gaps (cross-cutting)
1. **Names vs structure.** Three independent 2026 studies show SWE agents localise by lexical or memorised cues. They all *evaluate* only. No one *trains* an SWE agent to be name-invariant, and Face/Off shows naive anonymised SFT doesn't work.
2. **No intermediate structural supervision for SWE agents.** SWE training rewards final outcomes (resolve, file hit). GRAIN-style process supervision on the path the agent claims (is this chain real in the graph?) exists only for toy graph problems.
3. **Imperfect-graph regime is unstudied.** Every code-graph system assumes a directed, typed graph. Our measured reality: 100% reciprocal edges, one mixed label, no natural-language entry point.
4. **Optional tools go unused** (CodeAnchor, SHERLOC's zero-tool shortcuts). Delivering structure (passive, just in time, as a finding) matters as much as having it.
5. **Project knowledge: parametric vs contextual.** SkillForge proves project-specific knowledge helps, as text skills. Nobody compares it with the same knowledge in a LoRA, or tests whether that LoRA learned structure or just names.
6. **Localisation metrics are outcome-only.** GraphDancer's evidence-hit vs conversion split has not been applied to SWE agents. It separates *didn't find* from *found but didn't recognise*.

## 3. Mapping to our findings
| Our finding (A1/A2) | Literature says | Implication |
|---|---|---|
| Gold named in only 21%; clean seeds in 47.5% | Agents lean on names (Schrödinger, SHERLOC 58%) | On our tasks the name shortcut runs out fast, so structure has to carry the rest. |
| ~69% of tasks not bridged issue→fix in ~2 tool calls | Structural index + good retrieval gives +40 pp acc@5 | Retrieval and delivery are the bottleneck, not graph coverage. |
| Edges 100% reciprocal; direction recoverable 86–90% | GRAIN: build the correct graph IR, reward matching it | We can repair direction offline and use it as a free training signal. |
| Embeddings = graph-smoothed text; top-1 neighbour is a graph neighbour only 3–9% of the time | Soft graph tokens > text (AGT) but hard to ship | Keep embeddings as a candidate generator, not a reasoning medium. |
| Mixed "calls" label (containment + test edges) | Typed edges assumed everywhere | Re-typing edges (calls / contains / tests) is cheap and shippable. |

---

## 4. Candidate theses (ranked)

### T1 — "Structure over names": name-invariant, graph-verified localisation (recommended)
**Analogy.** Today's agent navigates like a tourist reading street signs. Rename the streets and it's lost. We want it to navigate like a taxi driver who knows the road network.

**Method.**
1. Repair the graph: add direction from source text and re-type edges.
2. The localiser emits an **evidence chain**: symptom node → directed path → suspected fix node(s). The chain is checkable.
3. Train signal = gold hit + **chain validity/overlap vs the repaired graph and gold nodes**. This is GRAIN's structure reward, transplanted, and it is execution-free.
4. **Consistent namespace remapping** of repo + issue + graph as training-time augmentation (Schrödinger's invertible mapping, used as a data generator). This also multiplies our 129 tasks ×k.
5. Recipe: teacher = Gemma 31B (free API), generating rollouts on original + remapped repos → filter by the structure reward → LoRA SFT (rejection-sampled). Paper ablation: small Gemma + GRPO with vs without the structure reward, and with vs without remapping.

**Why it's novel.** It is the first *training-side* answer to the 2026 memorisation/lexical findings, and the first process-level structural reward for SWE localisation. Face/Off predicts renaming alone fails. Our hypothesis is that renaming + structural reward succeeds; either outcome is publishable.

**Topics it spans.** Code comprehension, graph reasoning, agent post-training, robustness evaluation.

**Leaderboard payoff.** Better localisation plus a data multiplier.

**Risks.**
- Repo-wide consistent renaming tooling (Schrödinger's code is released).
- A chain reward could be gamed with trivial chains. Mitigate by requiring the endpoint to hit gold.

### T2 — Parametric vs contextual project knowledge (can fold into T1 as an ablation)
Compare a per-repo "graph literacy" LoRA (structural QA over its own graph plus self-synthesised issues, SkillForge-style but in weights) against SkillForge-style text skills. Use remapping to test whether the LoRA learned structure or memorised names.

**Only worth it if the hidden test repos = training repos.** Must verify.

### T3 — Diagnostic lens (a section, not a thesis)
Evidence-hit vs conversion, plus premature-stop rate, per tool configuration on our ATIF traces. Goes into Map 2 and the paper's analysis section.

Rejected as headline:
- Soft graph tokens (can't ship; sink risk).
- Visual graph modality (context cost).
- Semantic edge labels (SemNav overlap).
- Pure entry grounding (taken).

---

## 5. Architecture changes

### (a) Shippable in the submission (YAML agents, sub-agents, skills .py, ≤8 LoRAs, prompts)
1. **Localiser → Editor split.** Separate sub-agents with separate LoRAs. The localiser outputs a SHERLOC-style finding plus an evidence chain.
2. **Passive graph map at start.** A skill script computes clean seeds (A2 pipeline), a direction-repaired 2-hop neighbourhood and embedding top-k, and injects a compact map. Optional tools become the fallback.
   - *Blocked on verifying that skill scripts can read the graph files and repo.*
3. **Just-in-time entity cards** when the agent reads a file (SkillForge micro-injection), if the harness exposes a hook. Otherwise fold them into the map.
4. **Guards.** Loop detection, a premature-stop check ("did you see evidence for the fix node?"), and constrained tool-call format.
5. **Data recipe.** Quantity first (≥1k trajectories), filtered by low error-retry and chain validity; execution-free where possible.

### (b) Paper-only experiments (small Gemma, TPU)
- GRPO with the structure reward vs outcome-only (the core ablation).
- Remapped vs canonical evaluation (Schrödinger protocol) on our 129 tasks.
- Optional stretch: soft graph tokens for code graphs (AGT-style) as a "what if architecture were allowed" upper bound. Low priority.

---

## 6. Open questions blocking thesis lock
1. Are hidden test tasks from the same 4 repos? This decides T2 and how much per-repo knowledge pays off.
2. Can skill `.py` scripts read `graph/*.json` and the repo at runtime? This decides passive injection.
3. Will Gemma-31B-QAT emit parseable evidence chains zero-shot? This decides whether a warm-start SFT is needed.
4. Is repo-wide remapping on fastapi/rich feasible with Schrödinger's released tool, keeping tests green?

## Sources
- GRAIN https://arxiv.org/abs/2608.27142
- GraphDancer https://arxiv.org/abs/2602.02518
- Agentic Graph Tokens https://arxiv.org/abs/2608.00542
- When Graph Tokens Sink https://arxiv.org/abs/2606.03712
- Schrödinger's Repo https://arxiv.org/abs/2609.27891
- Jagged Frontier https://arxiv.org/abs/2608.18389
- Lexical Superstition https://arxiv.org/abs/2609.26388
- SkillForge https://arxiv.org/abs/2608.18933
- SHERLOC https://arxiv.org/abs/2606.24820
- Code Isn't Memory https://arxiv.org/abs/2606.22417
- Trajectory curation for LoRA https://arxiv.org/abs/2607.17205
- Function-aware FIM https://arxiv.org/abs/2607.12463
- SemNav https://arxiv.org/abs/2609.31176
- Dual-modal structural reasoning https://arxiv.org/abs/2607.01929
- LARGER https://arxiv.org/abs/2605.16352
- Earlier Map 1 papers (LocAgent, RepoGraph, CodexGraph, ToolTrain, CodeScout, GREPO, CGM, CodeAnchor): see `claude/Map1_literature.md`
