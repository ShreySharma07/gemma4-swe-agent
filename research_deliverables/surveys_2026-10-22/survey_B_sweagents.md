# Survey B: Code graphs and repository structure in LLM SWE agents (2024 to Oct 2026)

Compiled 2026-10-22. Sources were found via alphaXiv (discover_papers / get_paper_content / answer_pdf_queries) and web search.
"Read" means the full text or a detailed report was read. "Abstract" means only the abstract or snippet was seen; treat its details as less certain.
Numbers are as reported by the authors. Nothing here was re-run.

## 1. Paper table

### A. Graph-guided localization

| Paper | Year / venue | ID | Method (one line) | Key quantitative result | Limitation / caveat |
|---|---|---|---|---|---|
| LocAgent (read) | 2025, ACL 2025 | arXiv 2503.09089 | Heterogeneous graph (dir/file/class/func; contain/import/invoke/inherit) + sparse BM25 entity index; tools SearchEntity / TraverseGraph / RetrieveEntity; SFT (LoRA) of Qwen2.5-7B/32B on 433+335 Claude/self trajectories | SWE-bench Lite file Acc@5 94.2%, func Acc@10 77.4% (Claude-3.5); Qwen-32B-ft ~ Claude at $0.09 vs $0.66/instance; ablation: removing TraverseGraph or multi-hop hurts function-level most | Python only. In later independent re-runs it is weak. CoSIL reports up to 27% empty outputs. RepoNavigator: training-free Qwen2.5-7B function recall 1.0%, 32B 8.7%. LARGER: with GPT-5.2, LocBench function Acc@5 is only 11.7. Graph tools are unusable without training. |
| CoSIL (read) | 2025 | arXiv 2503.22424 | No pre-index; LLM builds 1st-order module-call and function-call subgraphs on the fly; pruner LLM rejects irrelevant nodes; "reflective alignment" short re-query to fix output format | Qwen2.5-Coder-32B Top-1 function acc 43.3% (Lite), 44.6% (Verified); +Agentless 25.67% / 34.6% resolved (GPT-4o); $0.02/inst w/ DeepSeek-V3; ablation: iterative search -20%, module graph -10%, reflection -6-10%, pruning -3% | Evaluated only on public SWE-bench. In SWE-Explore it has the highest line recall but emits whole files, so file-level noise is high. LARGER (GPT-5.2) finds it trails plain bash agents. |
| OrcaLoca (abstract + web) | 2025, ICML 2025 | arXiv 2502.00350 | Priority-queue action scheduling, action decomposition, distance-aware context pruning on code graph | Lite function match 65.33%, file 83.33%; 41.0% resolved; +6.33 pp resolve when paired w/ Agentless-1.5 editing | RepoNavigator authors could not reproduce it. In SWE-Explore its region recall is very low (HitReg 0.126). |
| GraphLocator (read) | 2025 (Dec) | arXiv 2512.22469 | Repository Dependency Fractal Structure (lazy-loaded tree-sitter graph) + Causal Issue Graph of sub-issues (Mermaid-serialized) expanded by priority queue; abductive LLM reasoning | +19.49 pp function recall, +11.89 pp function precision vs best baseline (avg, Claude-3.5) across Lite / LocBench / Multi-SWE Java; 52.9% fewer input tokens than LocAgent; CIG structure fed to Trae Agent: +28.74% (relative) resolved on Lite | Proprietary backbones only (GPT-4o, Claude-3.5). The authors themselves flag leakage risk. |
| LogicLoc / "Neurosymbolic Repo-level Code Localization" (read, partial) | 2026 | arXiv 2604.16021 | LLM synthesizes Datalog over extracted program facts; parser-gated validation + mutation diagnostics; deterministic engine does traversal | Names the "Keyword Shortcut": in over 50% of issues the description names the GT file/class/function. KA-LogicQuery (25 keyword-free queries) causes a "catastrophic" drop for SOTA localizers. LogicLoc is competitive on standard benchmarks with fewer tokens. | Diagnostic set is tiny (25). Unclear transfer to end-to-end repair. |
| RepoNavigator / "One Tool Is Enough" (read) | 2025-26, ICML 2026 | arXiv 2512.20957 | Single LSP-backed `jump` (go-to-definition) tool; GRPO from base Qwen2.5 (7/14/32B), no distillation | SWE-bench Pro Python, 32B GRPO: function F1 20.7 / file F1 57.6 vs CoSIL 7.7 / 21.0. Tool ablation (7B, function IoU): jump only 24.3; jump + GetClass + GetFunc + GetStruct 13.7, so more tools hurt. | Gives every method the gold entry file and function (App. B). That is a big deviation from standard SWE-bench and inflates absolute numbers. |
| RepoSearcher + ToolTrain (read) | 2025 | arXiv 2508.03012 | 6 simple tools (GetRepoStructure, GetImportOfFile, SearchClass/Function/ClassMethod); rejection-sampled SFT (5k Claude-3.7 trajs) then RL with nDCG@k reward on 18k issues from 600 repos | 32B function Recall@5 68.55 (above Claude-3.7's 66.38); 31.6% resolved with Qwen-32B patcher; 7B 14.0%; RL drives tool-call success to over 95% | Resolve rate is bounded by the patch model (7B patcher 14% vs 32B 31.6% at similar localization). Evaluated only on Verified. |
| CodeScout (read) | 2026 | arXiv 2603.17829 | RL (GSPO, no KL) of Qwen3 1.7B/4B/14B with bash-only scaffold (rg/sed/find) + structured LocalizationFinish tool; reward = file+module+function F1 (+ finish-within-k bonus) | 4B/14B beat Qwen2.5-32B with OrcaLoca/CoSIL by 10-33 pp F1; 1.7B ≈ RepoNavigator-7B; GPT-5/Sonnet-4.5 often better with plain bash than with RepoNavigator scaffold; feeding CodeScout locations to Qwen3-4B/Coder-30B OpenHands raises resolve and cuts steps | No graph. The result is evidence against specialised graph tools once RL is applied. Python ground truth only. |
| SHERLOC (read) | 2026 | arXiv 2606.24820 | Training-free reasoning LLM + 4 compact tools (tree, view, search, import tree) + self-recovery (loop detection, malformed-call repair, final-turn synthesis); emits 5-field diagnostic finding per location | File Recall@1 on Verified 81.3% (Qwen3-235B), 75.1% (Qwen3-30B). Injecting findings into repair agents: Qwen3-Coder-30B SWE-Agent 44.7 to 54.0 (McNemar Holm p=8.6e-5); Next-80B +10.6 / +11.8 pp; strong MiniMax-M2.5 -4 / -5 pp. Shuffled findings hurt (44.7 to 40.2). Localization tokens -36.7%. Agents spend 48% of turns (18.5 turns, about 320k tokens) localizing before the first patch. | Findings from the weak Qwen3-30B localizer did not help: 44.7 to 43.8 and 49.4 to 42.2 (Table 16). About 58% of localization recall comes from issue text alone, and the masked-path control is admitted to be weak. |
| LARGER (read) | 2026 | arXiv 2605.16352 | Passive structure: the agent's own grep/lexical hits are aligned to graph anchors, and confidence-filtered k-hop neighbours are appended inside the same search output. No new tool. | LocBench file Acc@5 87.0 vs Codex 74.1 (GPT-5.2), same backbone; wall-clock about halved; tokens 522K to 353K on MuLocBench; removing graph expansion -13.5% Acc@5. With GPT-5.2, CoSIL and LocAgent trail plain bash agents. | Uses frontier models only. Not evaluated end-to-end on resolve rate. |
| RepoMem / "Improving Code Localization with Repository Memory" (read) | 2025, ICLR 2026 | arXiv 2510.01003 | Episodic memory (BM25 over past commits: SearchCommit / ExamineCommit) + semantic memory (LLM summaries of the top-200 most-edited files) added to LocAgent | Verified Acc@5 71.6 to 76.5; SWE-bench-Live 63.1 to 66.2; resolved 37.0 to 40.4%; memory search 0.22 s | Gains depend on history depth: pytest +16.7, but repos with sparse history -13.1. Irrelevant memory "pollutes" context. GPT-4o only. |
| KGCompass (web) | 2025 | arXiv 2503.21710 | Repository-aware KG linking issues/PRs to code entities; narrows to 20 candidate functions; path-guided repair | v1: 45.67% Lite at $0.20 per repair. v2: 58.3% with Claude-4 Sonnet. 69.7% of localized bugs need multi-hop KG paths (v1). | Uses issue/PR link history, which a private-repo snapshot may lack. Details not personally verified. |
| ARISE (abstract) | 2026 | arXiv 2605.03117 | Repo-level graph representation + toolset for agentic APR / FL | Not read | Unverified |
| A^2Agent (abstract) | 2026 | arXiv 2608.29831 | Action-aware RL with per-action credit for localization agents | Not read | Unverified |
| Semantic Navigation (abstract) | 2026 | arXiv 2609.31176 | Iterative semantic navigation for issue localization | Not read | Unverified |

### B. Graph tools / structure as agent interface

| Paper | Year / venue | ID | Method | Key result | Limitation |
|---|---|---|---|---|---|
| RepoGraph (read) | 2024, ICLR 2025 | arXiv 2410.14684 | Line-level def/ref graph; k-hop ego-graph retrieval; plugged in as prompt context (procedural) or as `search_repograph` action (agent) | Lite: Agentless 27.33 to 29.67 (+7 tasks), SWE-agent 18.33 to 20.33 (+6), RAG 2.67 to 5.33. Average +32.8% relative. Flattened 2-hop ego-graph (10.5k tokens) gave 26.0, below the 27.33 baseline. 1-hop flat (2.3k tokens) was best. SWE-agent calls the tool mostly in the first 15 turns. | Gains are about 2 pp (6-8 tasks of 300) with no significance test. GPT-4 class models only; the authors say open models were too weak. |
| CodexGraph (abstract) | 2024 | arXiv 2408.03910 | LLM agent writes Cypher-like queries against a code graph DB | Not read in detail | Query-language generation is a burden for weak models (unverified) |
| Codebase-Memory (read) | 2026 | arXiv 2603.27277 | Tree-sitter KG for 66 languages in a single SQLite file, exposed as 14 MCP tools (search_graph, trace_call_path, Cypher-like query); 6-strategy call resolution with confidence | About 1k vs 10k tokens per question and 2.3 vs 4.8 tool calls, but answer quality 83% vs 92% for a grep/file explorer (Opus 4.6). Graph loses on macro-heavy C (0.58) and on full-source questions. | The benchmark is code-QA, not issue resolution. It is the authors' own benchmark with no controlled end-to-end SWE eval. |
| DualView (read) | 2026 | arXiv 2607.01929 | 4 graph views (module coupling, call, class hierarchy, PDG) returned as rendered image + text via MCP / shell commands | SWE-bench Pro public, Kimi K2.5 OpenCode: 342 to 388 resolved (+46), cost $0.55 to $0.50, steps 50.1 to 45.5. It beats RepoGraph (+3.4 pp) and CodeGraph MCP (+2.9 pp). Ablation on 150 tasks: textual-only graph 71, visual-only 78, both 81, base 69. Agents make only 1.8-3.7 graph calls per instance. | Requires a multimodal frontier model. The ablation is n=150 with no significance test. |
| The Devil Is in the Interface (read) | 2026 | arXiv 2608.11386 | Controlled tool-architecture study with fixed capability: BashOnly / Atomic / NL search / Python / scaffolding; SWE-bench-Live 65 tasks × 10 rollouts | Resolve rate is about equal across architectures. Atomic tools raise pass^k most for the weak Qwen3-Coder-30B (+0.06-0.07) by cutting mis-edits and bad syntax. A Python-code interface uses 41.6% fewer steps and 56.3% fewer input tokens. NL search broadens reads by more than 11% but lowers precision. Hypothesis-tracking and scratchpad tools are barely used. | No graph tools. Small n (65). |
| SWE-Explore (read) | 2026 | arXiv 2606.07297 | Line-level exploration benchmark (848 issues, 10 languages); explorer outputs top-5 regions; restricted-context repair validates the metrics | Context efficiency correlates r=0.95 with downstream resolve. Restricted-context resolve: CoSIL 59.3%, near Oracle 59.7%; LocAgent 44.7%, below plain mini-SWE-agent at 50.0%. Missing core context hurts more than noise. An empty-context dip suggests memorisation on canonical repos. | Built on public repos (Verified / Pro / Multilingual). |
| Small Agents with Semantic Search (read) | 2026 | arXiv 2610.05099 | 1-1.7B localizers trained with weighted SFT + RL using `colgrep` (late-interaction semantic search + regex pre-filter) vs grep | Qwen3-1.7B RL: file F1 67.3 (colgrep) vs 63.8 (grep), roughly a 32B base model. Python-only SFT generalises to 7 unseen languages: -2.1 pt with colgrep vs -14.9 with grep. 44% lower CPU latency, 29% fewer tokens. Grep is better at function level (25.1 vs 9.5 F1). | Localization only. Semantic search loses to grep on symbol-level precision. |
| Cost-Effective Repository Exploration (abstract) | 2026 | arXiv 2608.29675 | Delegating exploration to cheaper explorer models (IssueLoc-Bench: 499 Verified + 500 tasks from 153 other repos) | Cheaper explorers keep 78-94% of reference Hit@3 at 84-95% fewer tokens | Abstract only |

### C. Graph retrieval for repo-level code generation / completion

| Paper | Year | ID | Method | Key result | Limitation |
|---|---|---|---|---|---|
| GraphCoder (abstract) | 2024 | arXiv 2406.07003 | Code context graph (control-flow + data/control dependence) coarse-to-fine retrieval for completion | Not read in detail | Completion only |
| GraphCodeAgent (abstract) | 2025 | arXiv 2504.10046 | Dual graph (requirement graph + structural-semantic code graph) guiding a retrieval agent for repo-level generation | Not read in detail | Unverified |
| CGM (read) | 2025 | arXiv 2505.16901 | Graph-integrated LLM: CodeT5+ encoder compresses each node to 1 token, adapter, graph-aware attention mask between node tokens (Qwen2.5-72B, LoRA); agentless Graph-RAG (rewriter / retriever / reranker / reader) | Lite 43.0%, Verified 50.4% (top open-weight at the time). Removing the graph-aware mask: -8.6 / -5.6 EM on completion. Removing the reranker: -24.7 pp on Lite. | Needs architecture change and 72B. The reranker (plain file selection) is the dominant component. |
| CodeRAG (abstract) | 2025 | arXiv 2509.16112 | Relevant-and-necessary knowledge retrieval for repo completion | Not read | Unverified |
| GRACE (abstract) | 2025 | arXiv 2509.05980 | Graph-guided hierarchical code fusion for completion | Not read | Unverified |
| Better Call Grep (abstract) | 2026 | arXiv 2601.23254 | Grep-like lexical retrieval for repo-level completion | Snippet only: argues grep-like retrieval is competitive | Unverified |
| FeatLens; Partial Dependency Graph (abstract) | 2026 | arXiv 2609.26480; 2608.01927 | Dynamic / partial dependency graph retrieval for repo-level generation | Not read | Unverified |

### D. Evaluation: contamination, memorisation, renaming, live/private benchmarks

| Paper | Year | ID | Finding | Relevance |
|---|---|---|---|---|
| The SWE-Bench Illusion (read) | 2025 | arXiv 2506.12286 | From the issue text alone (no repo), models name the buggy file path: up to 76% on Verified, 50-68% on new issues from the same repos, under 53% on other popular repos. The gap persists when paths are filtered out of the issue. 5-gram function reproduction 34.9% on Verified vs 18.2% on SWE-Extra. | Localization benchmarks on SWE-bench repos largely measure recall of memorised paths, which is exactly what graph localizers are scored on. |
| SchrodingerRepo (read, partial) | 2026 | arXiv 2609.27891 | Semantics-preserving transforms (issue rewrite, namespace remap, layout reorder, code rewrite) cut Verified pass@1 by 6.0-14.4% (p<0.05), and 81.6-83.6% of the extra actions are exploration and localization. On temporally held-out SWE-rebench the same transforms keep Pass@1 unchanged and only raise cost. | Renaming is a valid memorisation probe. On unseen repos the cost lands mainly on exploration, which is where structure tools should matter. |
| SWE-bench Goes Live (web) | 2025 | arXiv 2505.23419 | 1,319 post-2024 tasks from 93 repos; the same agent+LLM pairs score consistently lower than on SWE-bench | Fresh-repo baseline |
| SWE-Bench Pro (web) | 2025 | arXiv 2509.16941 | Public 731 / commercial (private codebases) 276 / held-out 858. Commercial-set scores fall: GPT-5 23.1 to 14.9, Opus 4.1 22.7 to 17.8 (per Scale leaderboard summary). | Closest public analogue to a private-repo scorer |
| SWE-rebench (cited by others) | 2025 | arXiv 2505.20411 | Automated decontaminated task collection (V2: 2602.23866) | Fresh tasks for RL/eval |
| From Finding to Fixing (read) | 2026 | arXiv 2610.25826 | 6 FL methods (incl. LocAgent, CoSIL, OrcaLoca, SHERLOC, SWERank, GraphLocator) × 3 LLMs, 16,248 trajectories. FL gives almost no repair gain on Verified (Δpass@1 -3.0 to +0.4) and often costs more tokens than it saves. Agents anchor on wrong FL 34-100% of the time but recover in 35-96% of cases. With location clues removed from issues: the no-FL baseline drops 73.4 to 68.4, gold symbols + bug cause give +13.0 pp (vs +6.6 with original issues), and gold locations cut repair tokens by up to 39%. | The value of localization grows when issues lack location clues, which is likely in private repos. Bug-cause text is worth more than locations. |
| Fault-localization granularity (read) | 2026 | arXiv 2604.00167 | Perfect localization fed to Agentless + Qwen3-Coder-30B on SWE-bench-Mini (50 tasks, 10 trials): function 45.6% > line 43.6% > file 42.6% (Friedman p=0.0017). The best granularity depends on difficulty. | Even perfect FL only reaches about 46% for a 30B model. Small study. |
| Loc2Repair (abstract) | 2026 | arXiv 2606.30963 | Framework separating file-level localization from repair failure modes | Unverified |

### E. Training agents to use structure / small models

| Paper | ID | Note |
|---|---|---|
| SWE-RL (cited) | arXiv 2502.18449 | RL on software-evolution data. Not read here, and nothing graph-specific is known. |
| Self-Play SWE-RL (abstract) | arXiv 2512.18552 | Meta self-play SWE-RL. Not read. |
| FrogNano (abstract) | arXiv 2609.07925 | 4B coding agent via online task synthesis. Not read. |
| LocAgent SFT, ToolTrain, RepoNavigator, CodeScout, colgrep agents | see above | All train localization only and score with string-level rewards (F1 / nDCG / IoU), not graph-path rewards. None report LoRA-only adapters on a quantised base. |

## 2. Synthesis: what the evidence actually says

1. **Graph structure shows a consistent benefit only when it is cheap to consume.**
   - LARGER (passive neighbours inside grep output) gives +11.8 Acc@5. CoSIL and LocAgent (separate graph tools) trail plain bash agents with GPT-5.2.
   - RepoNavigator: adding graph-ish tools beside `jump` lowers IoU from 24.3 to 13.7.
   - RepoGraph: a 2-hop flattened subgraph hurts (26.0 vs 27.3 baseline), while 1-hop helps.
2. **Untrained small models cannot drive graph tools; RL-trained small models do not need them.**
   - Training-free LocAgent with Qwen2.5-7B/32B gets 1-9% function recall.
   - CodeScout (bash-only, RL) beats graph scaffolds at 4B/14B.
3. **Localization gains rarely become resolve gains for strong models, but do for weak ones, provided the localizer is stronger than the fixer.**
   - SHERLOC: a 235B localizer gives +8-12 pp to 30-80B fixers. A 30B localizer gives nothing or hurts.
   - From Finding to Fixing: about 0 pp at a 73% baseline.
4. **Almost all graph-tool evidence is on memorised public repos.** Issue-only file recall reaches 58-76%, so headline localization numbers mostly measure recall. The only transform study (SchrodingerRepo) shows the extra cost on unseen-like repos lands in exploration.
5. **Agents use graph tools sparsely.**
   - DualView: 1.8-3.7 calls per instance.
   - RepoGraph: calls mostly in the first 15 turns.
   - colgrep: the paper says tool descriptions alone are insufficient, and models keep falling back to grep until SFT teaches the new tool.

## 3. Open gaps (ranked; sharp ones first)

**G1. No causal test of graph/structure tools on unseen code with a small open model, end to end.**
- Evidence:
  - Every graph-localization paper (LocAgent, CoSIL, OrcaLoca, GraphLocator, RepoNavigator, RepoGraph, CGM, LARGER) evaluates on SWE-bench Lite/Verified/Pro-public, LocBench or MuLocBench. All are public, popular repos.
  - Illusion: up to 76% file recall from the issue alone.
  - SHERLOC concedes that about 58% of its recall is achievable from issue text, and that its masked control "bounds but does not eliminate" the confound.
  - SchrodingerRepo shows renaming removes 6-14% of pass@1 on Verified, with 82% of the extra actions spent on exploration. Nobody has re-run a graph tool under SchrodingerRepo transforms or on SWE-Bench Pro commercial.
- Why open: the hypothesis "graph tools matter more once memorised cues are gone" is plausible from SchrodingerRepo's cost shift but untested. Graph tools could also fail more, since names lose meaning.
- The cheapest decisive experiment: graph-on vs graph-off × original vs namespace-remapped repo, on one small model, with McNemar.

**G2. Same-model localization handoff (localizer = fixer = small model) is unstudied, and the existing evidence leans negative.**
- Evidence:
  - SHERLOC Table 16: Qwen3-30B findings fed to Qwen3-30B-class fixers give 44.7 to 43.8 and 49.4 to 42.2.
  - Gains appear only with a 235B localizer.
  - From Finding to Fixing: FL rarely improves repair and often costs more tokens than it saves. Agents anchor on wrong FL in up to 100% of cases.
  - Shuffled findings cost 4.5 pp in SHERLOC.
- Why open: all positive handoff results use a localizer stronger than the fixer. When one model must do both, under one context budget with no compaction, the net effect of a separate structured localization stage (and the right confidence gate) is unknown. SHERLOC's judge-filtered QF variant shows gating matters, but it uses a GPT-5.2 judge that sees the gold patch.

**G3. Passive structure injection vs optional graph tool, under a hard context cap, for weak models.**
- Evidence:
  - LARGER (passive, inside the lexical channel) beats tool-based graph agents.
  - RepoGraph shows output size flips the sign of the effect (1-hop helps, flattened 2-hop hurts).
  - RepoNavigator: more tools lower IoU.
  - Devil-in-Interface: interface shape changes consistency for a 30B model.
- Why open: LARGER uses GPT-5.2 and stops at localization metrics. No paper measures, at a fixed context ceiling (e.g. 32k) with a small model, how much passive neighbour-injection buys versus its token cost and malformed-call cost. Every cost analysis reports $/tokens, not success under a hard cap with no compaction.

**G4. RL/SFT on graph tools vs bash with matched compute, scored end to end, has not been done.**
- Evidence:
  - ToolTrain (graph-lite tools), RepoNavigator (`jump`), CodeScout (bash) and colgrep agents each train one interface.
  - All use localization rewards (nDCG / F1 / IoU). RepoNavigator also supplies the gold entry point.
  - No study trains two identical models that differ only in having graph tools and then measures resolve rate.
  - No paper uses a graph-aware reward (e.g. credit for traversing the gold call path). A^2Agent's action-level credit is the closest, and it is unverified.
  - No paper uses low-rank LoRA on a quantised base, which is the realistic constraint for small-budget teams.
- Why open: whether structure helps, or RL simply teaches efficient search with any tool, is not separable from the current literature.

**G5. Name-resolution brittleness of graph lookups.**
- Evidence:
  - LARGER frames localization as "turning lexical matches into structural entry points".
  - Codebase-Memory needs a 6-strategy fuzzy cascade (confidence down to 0.30) to resolve calls.
  - colgrep needs a hybrid regex + semantic query to beat grep. Grep still wins at function level (25.1 vs 9.5 F1).
  - LocAgent's SearchEntity relies on a BM25 fallback; removing it is the largest ablation drop.
- Why open: no paper reports the fraction of graph-tool calls that return nothing because the agent supplied a phrase or bare name instead of a fully qualified ID. Nor does any paper show how a fuzzy/anchoring layer changes that rate for small models. This is plausibly the dominant failure mode for exact-key graph tools.

**G6. Repository memory without commit history.**
- Evidence:
  - RepoMem's gains scale with history depth (+16.7 on pytest).
  - It degrades on sparse-history repos (-13.1).
  - KGCompass uses issue/PR links.
- Why open: offline snapshot settings (private repo, no git log, no issues) remove both. Snapshot-only memory (tests-to-code links, docstrings, cross-task experience within one run) is untested for localization.

**G7. Localization benchmarks still reward keyword matching.**
- Evidence:
  - Over 50% of issues name the gold location (Keyword Shortcut paper). The only keyword-free set has 25 queries.
  - From Finding to Fixing shows FL value roughly doubles when location clues are removed (gold symbols + cause: +13.0 vs +6.6 pp).
- Why open: there is no sizeable keyword-free, execution-verified benchmark on which graph vs lexical localizers have been compared. The one in-domain case where structure should matter most has the least data.

**G8. Output format of structured localization for weak fixers.**
- Evidence:
  - Perfect FL: function-level beats line and file levels by 1-3 pp for a 30B model (50 tasks, 2 repos), with the ordering changing by difficulty.
  - SHERLOC field ablation: removing solution idea or root cause costs 2.8-3.2 pp for the 30B fixer, but helps the strong MiniMax model.
  - GraphLocator says its causal graph helps Trae Agent.
- Why open: no study has crossed granularity × diagnostic fields × fixer size with real (imperfect) localizers.

**G9. The value of visual or graph-rendered structure for text-only small models is unknown.**
- Evidence: DualView's gain is largely from the visual modality (visual 78 vs textual 71 of 150). Textual graph serialisation alone gives only +2 tasks over base.
- Why open: text-only small models get only the weak half of that result. Compact textual encodings (tree-indented, as LocAgent uses; Mermaid, as GraphLocator uses) have not been compared head to head.

## 4. Cost / latency / context notes (for reference)
- Localization phase: 48% of turns, about 320k tokens per instance (SHERLOC, 5 LLMs × 2 frameworks). Reading files is 76.1% of the token budget on Verified (cited in DualView).
- Graph index build is cheap:
  - LocAgent: seconds.
  - Codebase-Memory: Django in 6 s; Linux kernel in 3 min; queries under 1 ms.
  - RepoMem: SymPy in 45 min, because it includes LLM summaries.
- Graph outputs can blow up context: RepoGraph 2-hop flattened is about 10.5k tokens per call and hurts resolve rate.
- Several papers report net cost decreases with structure, because fewer steps outweigh larger observations:
  - DualView: -$0.05 per instance, -4.6 to -8.5 steps.
  - LARGER: -32% tokens.
  - colgrep: -29% tokens.
- The token savings are reported for frontier models. For small models, the cost of malformed calls is not reported.
