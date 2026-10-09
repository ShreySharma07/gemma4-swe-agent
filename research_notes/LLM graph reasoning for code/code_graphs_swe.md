# Code graphs for LLM-agent issue localisation: comparison grid and open gaps (as of 8 Oct 2026)

Scope: papers NOT already in docs/literature_2025_2026.md or docs/literature_map1.md, plus a gap check against our setting. Our setting: Gemma-4-31B at 4 bits, 32k context, offline, about 5 minutes per task, LoRA only. The provided graph is undirected, carries one mixed `calls` label and name-keyed embeddings. Agents ignore optional graph tools, and only about 21% of issues name the fix node.

Labels used throughout:
- **VERIFIED**: I read the claim in the primary source (PDF text through alphaXiv) during this session.
- **INFERRED**: my own reasoning, or a claim carried over from the project docs that I did not re-read.
- **Peer-reviewed / preprint**: taken from the venue line printed in the paper itself.

---

## Q1. How do methods build the graph, deliver it, and pick entry points? What does each gain, and at what model size?

### Takeaway
Since mid-2025 the field has moved away from "big typed graph plus many graph tools" in three directions:
1. **One structurally grounded tool trained with RL.** RepoNavigator (ICML'26) uses a single `jump` tool. A²Agent (preprint) uses LocAgent's tools plus action-level RL at 4B.
2. **Passive or post-retrieval structure.** CodeAnchor (ISSTA'26) injects tags inline. RepoNav (preprint) builds a file-centred scaffold.
3. **No graph at all.** SweRank (ICLR'26) uses retrieve-and-rerank. FastContext (preprint) uses a grep-only explorer subagent. The Thousand-Graph Hypothesis (preprint) uses an entity-only index with no stored edges.

On small and mid-size open models, the biggest gains come from **training** (RL, contrastive ranking) and from **how evidence is presented**, not from richer graphs.

### Comparison grid (new papers; ✔ = VERIFIED from the PDF)

| Method (ID, venue) | How the graph is built | How it is delivered | Entry point | Training | Model sizes | Key result |
|---|---|---|---|---|---|---|
| **RepoNavigator / "One Tool Is Enough"** (2512.20957, **ICML 2026**, peer-reviewed ✔) | No stored graph. A language server resolves symbol → definition on demand. | One tool, `jump(file, symbol)` | **The entry point (usually a test function) is included in the issue text** (App. F) | GRPO directly from the base model, no distillation; 4k SWE-smith samples, 1 epoch | Qwen2.5 7B/14B/32B | GRPO 7B beats 14B baselines on Sample-F1/IoU. 32B GRPO func IoU 32.30 vs LocAgent 14.18 on Verified. Tool ablation under RL: jump only = 24.28 IoU vs jump + GetClass + GetFunc + GetStruc = **13.71** (Table 4) |
| **A²Agent** (2608.29831, preprint, Aug'26 ✔) | Reuses LocAgent's graph index (contains/imports/invokes/inherits) plus BM25, LLM file summaries and a commit index | Six tools: TraverseGraph, SearchEntity, SearchCodeSnippet, SearchSummary, ViewSummary, SearchCommit | Agent-chosen | Teacher SFT on 1,410 trajectories (teacher Qwen3-30B-A3B), then advantage-weighted RL with a per-turn discovery reward and a **Discovery-Commit reward**; **LoRA r=16** | Qwen3-4B, Qwen3-8B | +1.58 F1 (Verified) and +8.55 F1 (Pro) over SOTA. 4B beats baselines up to 8× larger. Downstream resolve 38.07% vs LocAgent 31.78% (same 4B localiser, Qwen3-30B repairer; oracle 45.93%). Index build takes 4.7 GB / 1h56m for 500 instances |
| **CodeAnchor** (2606.26979, **ISSTA 2026** ✔) | Offline PyCG call graph + AST extractors (calls, inheritance, data-flow, config) | **Passive**: structured comments injected into the source files, read through ordinary grep | Grep hits | None | Codex (OpenAI) | +2.2 pp Func@5, −1.6 rounds, run-to-run variance roughly halved, about 10% more input tokens. An optional call-graph tool in a 20-task pilot had **low use**. Matched LocAgent trails grep Codex (Func@5 59.5 vs 83.2 on Lite) |
| **RepoNav** (2609.08355, preprint, Sep'26 ✔) | No repository-wide graph. Python `ast` on the top-5 retrieved files only; same-file call neighbourhood | Post-retrieval text scaffold ([ANCHORS] / [GLIMPSE] / [CANDIDATE_TARGETS]) + `list_symbols` | Lexical match between the agent's query and symbol names | None | 7 models incl. GPT-OSS-120B, Qwen3-Next-80B, GLM-4.7, Gemini-3-Flash | "Correct file, wrong function" is **44–58%** of function-level misses. Removing [CANDIDATE_TARGETS] costs −7.6 Func Acc@5. Larger gains on complex files (+6–8 pp) |
| **SweRank** (2505.07849, **ICLR 2026** ✔) | No graph | Retrieve (bi-encoder) + listwise LLM rerank | The whole issue as the query | Contrastive on SWELoc (GitHub PRs) + listwise reranker SFT | 137M / 7B embedder; 7B / 32B reranker | SWE-Bench-Lite Func Acc@10 88.69 (32B) vs LocAgent-Claude-3.5 77.37. On the low-lexical-overlap bucket, 7B embedder 65.2% vs LocAgent 60.9% |
| **OrcaLoca** (2502.00350, **ICML 2025** ✔) | AST CodeGraph: containment + call-reference edges, **directed** | Search APIs + priority action queue + **distance-aware context pruning** (shortest path to candidate bug nodes, keep top-12) | Stack trace from the reproducer seeds the first actions | None | Claude-3.5-Sonnet | 65.33% function match on Lite; 41.0% resolved with the Agentless repairer. Removing pruning costs −4.3 pp on a 93-task subset |
| **Agentless** (2407.01489, FSE 2025 per citation in 2603.27277, INFERRED venue) | No graph; repository tree + skeletons | Fixed 3-stage pipeline | File tree → files → classes/functions → lines | None | — | Used as a baseline everywhere. On Qwen2.5-7B Verified, function IoU 11.74 (RepoNavigator Table 1 ✔) |
| **SWE-Explore** (2606.07297, preprint ✔) | Benchmark, not a method | — | — | — | — | 848 issues, line-level ground truth. **CoSIL (iterative code-graph search) has by far the highest non-oracle line recall and F1**. LocAgent looks like general agents. Missing context hurts more than redundant context |
| **FastContext** (2606.14066, Microsoft preprint ✔) | No graph | Explorer subagent with only READ/GLOB/GREP, returns file:line citations | Agent-chosen, parallel first turn | SFT on 2,954 Sonnet-4.6 traces → GRPO (file/line F1 reward) | Qwen3-4B, Qwen3-Coder-30B-A3B | Up to +5.5% resolve and up to −60% main-agent tokens in Mini-SWE-Agent |
| **Codebase-Memory** (2603.27277, preprint ✔) | Tree-sitter, 66 languages; **6-strategy call resolution with confidence scores 0.95 → 0.30**; Louvain communities; typed edges (CALLS, IMPORTS, INHERITS, TESTS, …) | 14 MCP tools (trace_call_path, query_graph, get_architecture, …) | Agent-chosen | None | Claude Opus 4.6 | Quality 0.83 vs 0.92 for a file explorer, at 10× fewer tokens and 2.1× fewer calls. Not a SWE-bench evaluation; graded by the first author |
| **Thousand-Graph Hypothesis** (2608.26602, preprint; header says "ICLR 2027", which is not plausible for an Aug'26 arXiv post, so treat as preprint ✔) | **Entities only, no stored edges**; two-layer index (global routing → local entity selection) | Entities serialised into context | Two-layer index | None | DeepSeek-V4-Flash | Claims 92.1 → 94.2 → 95.6% on SWE-bench Verified (base / 1-layer / 2-layer). One model, one benchmark, extraordinary numbers: treat with caution |
| **C2C** (2609.38402, preprint ✔) | No graph; CodeBERT contrastive file retriever | Hierarchical RL (BiLSTM/MLP agents) over file → function → line | Stack trace | Contrastive + HRL with position-based intermediate rewards | Small non-LLM encoders | SWE-bench Verified file Hit@1 21.7%, Hit@5 76.1%. Not an LLM-agent method |

### Cited findings
- RepoNavigator gets its structure from a language-server `jump`, trained with GRPO from a base model with no teacher. It "include[s] the entry point (normally a testing function) in issue q". VERIFIED — [2512.20957](https://arxiv.org/abs/2512.20957)
- In RepoNavigator, adding GetClass/GetFunc/GetStruc to `jump` under RL lowers function IoU from 24.28 to 13.71 (Qwen2.5-7B). The authors argue success decays multiplicatively with the number of sequential tool calls and that the action space should be minimal. VERIFIED — [2512.20957](https://arxiv.org/abs/2512.20957)
- A²Agent: trajectory-level rewards cannot tell "explored but not committed" apart from "never found". The Discovery-Commit ratio DC = |E∩Y∩O| / |E∩Y| rewards committing gold locations the agent has already seen. Uses LoRA r=16 and a 40K training context. VERIFIED — [2608.29831](https://arxiv.org/abs/2608.29831)
- CodeAnchor treats the call graph as "soft hints": strong gains even with the unsound PyCG, with the agent falling back to lexical reasoning when an edge is missing. VERIFIED — [2606.26979](https://arxiv.org/abs/2606.26979)
- RepoNav "requires no additional persistent structural index or repository-wide graph". It parses only the top-k retrieved files and records same-file call context by name only. VERIFIED — [2609.08355](https://arxiv.org/abs/2609.08355)
- SweRank: retrieve-and-rerank beats graph-agent LocAgent (Claude-3.5) on Lite and LocBench at roughly 1/40 of the cost. On issues with 2–3 gold functions, LocAgent falls 82.0 → 47.1 while the SweRank embedder holds 77.1. VERIFIED — [2505.07849](https://arxiv.org/abs/2505.07849)
- OrcaLoca prunes context by average shortest-path distance (taken in either direction) between each search result and the current candidate bug nodes. It retains 12 entries. VERIFIED — [2502.00350](https://arxiv.org/abs/2502.00350)
- SWE-Explore: "Specialized localizers only help when they broaden search… CoSIL is the main exception: it achieves by far the highest non-oracle Recℓ and F1". VERIFIED — [2606.07297](https://arxiv.org/abs/2606.07297)

### Inferences
- **Entry point.** Three of the strongest structure-based methods assume an anchor that our tasks often lack:
  - RepoNavigator puts a test function in the prompt.
  - OrcaLoca starts from a reproducer stack trace.
  - RepoNav and LARGER start from lexical hits.

  With only 21% of our issues naming the fix node, a SweRank-style dense retriever plus reranker is the most robust entry generator. SweRank shows it holds up at low lexical overlap. (INFERRED)
- **What gains at our size.** On ≤32B open models, the gains come from RL with dense or partial-credit rewards (RepoNavigator, A²Agent, FastContext) and from compact text presentation (RepoNav, CodeAnchor), not from graph richness. (INFERRED)

### Gaps
- None of the RL papers reports gains on a Gemma-family model or on a 4-bit model.
- A²Agent does not report per-tool usage frequencies after RL, so it remains unknown whether TraverseGraph is still used once training is done.

---

## Q2. Which work handles noisy or incomplete graphs, graph repair, or edge direction and typing? Which handles graphs far larger than the context?

### Takeaway
- **Noise and direction.** Handled only implicitly. CodeAnchor shows LLMs tolerate unsound call graphs and that **inverse-only ("who calls me") links beat bidirectional links in hub-heavy repos**. Codebase-Memory attaches **confidence scores to call edges**. Nobody repairs an undirected or untyped graph, or trains an agent to cope with one.
- **Scale beyond context.** Handled by:
  - query-time pruning (OrcaLoca: graph distance);
  - local, on-the-fly extraction (RepoNav: top-k files only);
  - hierarchical routing (Thousand-Graph: two-layer index; Codebase-Memory: Louvain communities / get_architecture);
  - delegation to an explorer subagent (FastContext).

### Cited findings
- CodeAnchor directionality: on medium repos (Lite: mean out-degree 4.2, 8% hubs) bidirectional topology is best. On hub-heavy repos (Verified: mean out-degree 9.7, 23% hubs), the inverse-only view is best (+0.2 pp Func@5 at equal tokens). Dense forward tags "pull the agent into utility modules". Their heuristic is: avg out-degree <5 → full topology; >8 → inverse-only. VERIFIED — [2606.26979](https://arxiv.org/abs/2606.26979)
- CodeAnchor granularity: dense data-flow and config tags add no Func@5 on Lite (0.854 = 0.854) but cost +4.9 rounds and +18.8% tokens. They rescue only 15/500 implicit-dependency cases on Verified. VERIFIED — [2606.26979](https://arxiv.org/abs/2606.26979)
- CodeAnchor effective tag rate: 27.1% (Lite) and 26.6% (Verified) of tag-following hops land on a ground-truth entity. 55–59% of tasks contain at least one such hop. VERIFIED — [2606.26979](https://arxiv.org/abs/2606.26979)
- Codebase-Memory resolves call edges through a cascade: import map 0.95 → same module 0.90 → import suffix 0.85 → unique name 0.75 → suffix 0.55 → fuzzy 0.30–0.40. Strategies 1–3 resolve about 80% of calls in well-structured code. It acknowledges dynamic dispatch and reflection are not represented. VERIFIED — [2603.27277](https://arxiv.org/abs/2603.27277)
- OrcaLoca's graph is directed, but its pruning distance is min(d(u→v), d(v→u)). Its relevance computation is therefore direction-agnostic. VERIFIED — [2502.00350](https://arxiv.org/abs/2502.00350)
- The Thousand-Graph Hypothesis argues that persisting edges is unnecessary because attention "materialises" task-specific relations once entities are in context. It reports no edges and 95.6% Verified with DeepSeek-V4-Flash, but "does not directly verify the latent graph". VERIFIED (claim) — [2608.26602](https://arxiv.org/abs/2608.26602)
- RepoNav parses only the top-5 retrieved files and fully expands the top-3. No repository-wide structure ever enters context. VERIFIED — [2609.08355](https://arxiv.org/abs/2609.08355)
- From the project docs (not re-read here): ARISE reports spurious edges hurt more than missing ones, and RepoGraph found 2-hop worse than 1-hop. INFERRED (carried over) — [ARISE 2605.03117](https://arxiv.org/abs/2605.03117), [RepoGraph 2410.14684](https://arxiv.org/abs/2410.14684)

### Inferences
- **Direction repair is directly actionable for us.** Our graph has 100% reciprocal edges. CodeAnchor's inverse-only result implies that recovering direction (86–90% recoverable per our A2 analysis) is not cosmetic: it lets us show only callers on hub nodes. No paper measures what an agent loses with an undirected graph compared with a directed one. (INFERRED)
- **Edge typing has a template.** Codebase-Memory's confidence-scored resolution is the closest template for re-typing our mixed `calls` label into calls / contains / tests with a confidence attached. (INFERRED)
- **Two cheap compression primitives fit 32k.** OrcaLoca's graph-distance pruning and RepoNav's top-k local scaffold both serialise as text, so both fit our 32k budget better than whole-subgraph dumps. (INFERRED)

### Gaps
- I found no paper that:
  - evaluates code-localisation agents on a deliberately degraded graph (undirected, mislabelled, or with edges removed or added) as a controlled variable;
  - repairs edge direction or type before giving the graph to an agent;
  - uses a name-keyed embedding index as the only "semantic" signal.

  See the Search log.

---

## Q3. Is there evidence that graph tools stop helping after post-training, or for weaker or quantised models?

### Takeaway
There is indirect but consistent evidence that richer graph toolsets stop paying off once a model is trained:
- RepoNavigator: adding tools under RL halves IoU.
- CodeScout (from project docs): bash only beats graph agents.
- FastContext: a grep-only RL explorer works.
- CodeAnchor: grep Codex beats LocAgent by 23.7 pp.

The counter-example is A²Agent, which keeps LocAgent's graph tools and wins at 4B. No paper runs the clean ablation: same model, same RL data, graph-traversal tool on vs off.

For quantised models specifically, the evidence is about tool use in general, not graph tools. Gemma-4-31B at INT4 **doubles tool-name hallucination** (19.5% → 38.3% of calls) on τ²-bench while the headline score stays flat.

### Cited findings
- RepoNavigator Table 4: under outcome-only RL, IoU is 24.28 with jump only, 24.00 with jump + GetStruc, 21.44 with jump + GetClass + GetFunc, and 13.71 with all four. "Excessive tools do not boost RepoNavigator's performance." VERIFIED — [2512.20957](https://arxiv.org/abs/2512.20957)
- Training-free LocAgent on Qwen2.5-7B scores very low on SWE-bench Pro (function IoU 0.40, file precision 0.17). RepoNavigator training-free scores 9.20 IoU. Graph-tool agents degrade sharply on small untrained models. VERIFIED — [2512.20957](https://arxiv.org/abs/2512.20957)
- A²Agent (Qwen3-4B, LocAgent's graph tools + 4 others, SFT → action-aware RL) beats CodeScout and RepoNavigator. A²Agent's own tools table lists CodeScout as "Terminal + LocalizationFinish". After RL the agent "shift[s] from broad exploration to targeted search, reducing average observation length by approximately 40%". VERIFIED — [2608.29831](https://arxiv.org/abs/2608.29831)
- CodeAnchor matched head-to-head on SWE-bench Lite (N=274): LocAgent Func@5 0.595 vs grep-first Codex 0.832. VERIFIED — [2606.26979](https://arxiv.org/abs/2606.26979)
- CodeAnchor: structure helps most when the base agent already has >80% recall. It offers "limited value when the base agent is weak (<60% recall)". VERIFIED — [2606.26979](https://arxiv.org/abs/2606.26979)
- FastContext's trained 4B/30B explorer uses only READ/GLOB/GREP. RL lifts the 4B explorer's file F1 from 70.55 (SFT) to 71.48 and function F1 to 38.45. VERIFIED — [2606.14066](https://arxiv.org/abs/2606.14066)
- Flat Score, Amplified Failures (preprint, Jul'26): Gemma-4-31B, BF16 → AWQ-INT4, τ²-bench telecom.
  - Tool-name hallucination 19.51% → 38.26% of tool calls.
  - Error-budget exhaustions 5 → 46 of 456 episodes.
  - Tool calls per episode 7.29 → 9.43.
  - Final score unchanged (66.7 vs 65.4).
  - Quantisation amplifies existing failure modes (only 0.18% novel events) rather than creating new ones.
  - Re-injecting the tool inventory after a failed call cut the channel to 2.9% and lifted INT4 to 71.3%.

  VERIFIED — [2607.27275](https://arxiv.org/abs/2607.27275)
- CodeScout (pure RL, bash only, Qwen3 1.7–14B) beats graph-tool agents. INFERRED (from project docs, not re-read) — [2603.17829](https://arxiv.org/abs/2603.17829)

### Inferences
- **Keep the tool surface small and constrained.** For a 4-bit Gemma-4-31B, the strongest evidence-backed design is a minimal tool surface (RepoNavigator's argument) with constrained or validated tool names (Flat Score's mitigation). Graph structure should go into observations passively (CodeAnchor / RepoNav style), not into extra tools. This matches our "agents ignore optional graph tools" finding. (INFERRED)
- **Watch per-call error rates.** The Flat Score result is on the exact base model we use, so a flat SWE resolve rate under 4-bit may hide inflated tool-call error volume that eats our ~5-minute budget. We should log per-call error rates. (INFERRED)

### Gaps
- There is no controlled study of graph-traversal tools vs bash under identical post-training. RepoNavigator varies symbol-lookup tools, not graph traversal. A²Agent never removes TraverseGraph.
- There is no localisation study on quantised models. Flat Score is a customer-service benchmark (τ²-bench), not SWE.

---

## Q4. What remains genuinely unaddressed? (Novelty claims backed by the Search log)

### Takeaway
Five gaps survived searching. In order of confidence:
- **G1. Imperfect-graph regime.** Undirected, single-label, noisy graphs as a controlled variable, plus graph repair before use. Not found.
- **G2. Graph-tool vs no-graph ablation under identical post-training.** Not found.
- **G3. Graph-based localisation with quantised or ~30B open models under a hard token/time budget.** Not found.
- **G4. Entry grounding when the issue does not name the fix node, combined with structure.** Partly addressed by SweRank (no structure) and RepoNav (lexical anchors). Joint dense entry + directed-graph expansion with a trained commit step is not found as a single method.
- **G5. Discovery–commit / premature-anchoring as a training target with structure in the loop.** A²Agent (reward) and RepoNav (interface) each address half. Combining a DC-style reward with passive graph injection is not found.

### Cited findings
- The discovery–commit failure is documented in two independent papers:
  - A²Agent: agents observe gold regions but fail to commit them.
  - RepoNav: 44–58% of function misses happen after reaching the correct file, which it calls "premature anchoring".

  VERIFIED — [2608.29831](https://arxiv.org/abs/2608.29831), [2609.08355](https://arxiv.org/abs/2609.08355)
- SWE-Explore: low F1 "is mostly a recall problem". Missing core context is the dominant downstream failure mode. Redundant context hurts mainly when core evidence is scarce (−7 to −9 pp at α=0). VERIFIED — [2606.07297](https://arxiv.org/abs/2606.07297)
- CodeAnchor lists generalisation to end-to-end repair, other agent architectures and other languages as open. VERIFIED — [2606.26979](https://arxiv.org/abs/2606.26979)
- Codebase-Memory lists a "controlled empirical study comparing graph-based, text-based, and embedding-based exploration across SWE-bench with ablation studies" as future work. VERIFIED — [2603.27277](https://arxiv.org/abs/2603.27277)

### Inferences
- **G1 is the most defensible paper novelty** for a Graph Reasoning track. Every system I read either builds its own directed, typed graph (LocAgent, OrcaLoca, CodeAnchor, Codebase-Memory, A²Agent) or avoids graphs (SweRank, FastContext, Thousand-Graph, RepoNavigator's LSP). Nobody studies the "given a bad graph" regime that competition harnesses impose. (INFERRED)
- **G2 and G3 are cheap to test.** They can be answered with our harness and small-Gemma ablations, and either outcome is reportable. (INFERRED)

### Gaps
- alphaXiv coverage of Oct 2026 may lag. Papers posted in the last ~10 days could be missed.
- I did not read CoSIL, GraphLocator or CGM. They were excluded as already covered.
- I could not verify any Oct 2026 venue decisions.

---

## Search log

Every query run in this session, with what came back.

**1. alphaXiv discover:** "graph-based code localization for LLM agents in large repositories (call graph, dependency graph) 2026"; keywords [code graph, fault localization, SWE-bench, repository graph]; recency.
- Returned: RepoAtlas, SemNav, FeatLens (2609.26480, code generation not localisation), A²Agent, Thousand-Graph (2608.26602), SWE-bench Science, LARGER, Codebase-Memory (2603.27277), LocAgent, GraphCodeAgent.

**2. alphaXiv discover:** "robustness of LLM code agents to noisy or incomplete code graphs; do graph tools help after RL post-training or for small quantized models"; keywords [noisy graph, incomplete call graph, code localization, graph tools].
- Returned: LocAgent, Noisy test-time RL for code (2609.32172, about noisy labels not graphs), SemNav, A²Agent, Hierarchical Graph Memory (2608.05095, general agent memory), Retry/Switch/Abstain (2608.11977, tool failures in general), FrogNano 4B, Closed-loop graph algorithms with SLMs (2606.24980, synthetic), GRAIN, CodeAnchor (2606.26979), ARISE, Topology-Aware Gaussian Graph Repair (2606.03462, GNN graph repair, not code agents).
- **No code-agent paper on noisy or undirected graphs.** Supports G1.

**3. Paper reads (answer_pdf_queries):** 2512.20957 (One Tool Is Enough), 2608.29831 (A²Agent), 2606.26979 (CodeAnchor), 2608.26602 (Thousand-Graph).

**4. Paper reads:** 2609.08355 (RepoNav), 2606.07297 (SWE-Explore), 2505.07849 (SweRank), "OrcaLoca: An LLM Agent Framework for Software Issue Localization" (resolved to 2502.00350).

**5. alphaXiv discover:** "LLM code agent localization with imperfect, incomplete, undirected or untyped call graphs; repairing or inferring edge direction and edge types in repository graphs"; keywords [call graph, edge direction, graph repair, issue localization, unsound static analysis]; published after 2025-06-01.
- Returned: Dual-Modal Structural Reasoning (2607.01929), SemNav, PhoenixRepair (2607.18859, repair strategies), CodeAnchor, RepoAtlas, ARISE, SHERLOC, Partial Dependency Graph retrieval for code generation (2608.01927), Neuro-Symbolic Indirect-Call Analysis for C (2609.33547, static analysis, not agents), ContextSniper (2607.01916), AgentTrails, GRADE.
- **None repairs direction or type for agent use.** Supports G1.

**6. alphaXiv discover:** "hierarchical abstraction or query-time subgraph extraction of repository code graphs too large for the LLM context window, for issue localization; small or quantized models using graph tools"; keywords [hierarchical summarization, repository graph, subgraph extraction, context window, code localization, quantized]; published after 2025-06-01.
- Returned: Hierarchical Graph Memory, SemNav, RepoAtlas, Partial Dependency Graph, Dual-Modal, A²Agent, MemRepair (2605.17444, vulnerability repair memory), ARISE, LARGER, ConFL (2608.02974, concurrency FL), C2C (2609.38402), Codebase-Memory, MemGraphRAG, FastContext (2606.14066), an analog-circuit compression paper.
- **No paper on graph localisation with quantised models.** Supports G3.

**7. Paper reads:** 2609.38402 (C2C), 2606.14066 (FastContext), 2603.27277 (Codebase-Memory).

**8. WebSearch:** `code graph tools LLM agent quantized small model localization ablation "graph tools" unused 2026`.
- Returned only LocAgent pages, the CoSIL and LARGER PDFs, and secondary summaries (emergentmind, liner, gitee mirrors).
- **No quantised graph-tool study.** Supports G3.

**9. WebSearch:** `quantization effect on LLM agent tool calling multi-turn SWE-bench 4-bit degradation 2026`.
- Returned: "Flat Score, Amplified Failures" (2607.27275, primary, read), plus blog posts (dailyaiworld, mer.vin, developersdigest, dev.to).
- I did not use the blogs: they are secondary and unverified, and one cites a "97.2–97.5% retention" figure I could not trace to a primary source.

**10. alphaXiv discover:** "after RL or SFT post-training do repository graph tools still help code localization agents, or does a bash/grep-only agent match them"; keywords [SWE-bench, graph tools, post-training, tool ablation, localization]; published after 2026-01-01.
- Returned: **"No papers found."** Supports G2.

**11. Paper read:** 2607.27275 (Flat Score, Amplified Failures).

**Not searched (budget):** ACL/ICSE/FSE 2026 proceedings pages directly; GitHub repos of the papers; OpenReview for ICLR 2027 submissions.
