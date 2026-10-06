# Gemma 4 Developer Agent — HQ

This is the single source of truth. Every new chat starts from this doc, and it gets updated at the end of each working session.

## Goals
- Win a prize on the **main track** (leaderboard, % of issues resolved).
- Submit a **solid, novel paper** to the paper track. Home topic: **Graph Reasoning** (finalised 27 Sep). The paper may intersect Code Comprehension, graph reasoning and agent post-training (Shrey, 28 Sep).
- Strategy: **safe core + ambitious add-on**. Post-training is in scope.

## Key dates (23:59 UTC)
| Date | Milestone |
|---|---|
| Sep 28 – Oct 1 | Research spike (below) |
| ~Oct 2 | Lock paper thesis + post-training method |
| ~Oct 15 | Checkpoint: agent scaffold done? |
| ~Nov 5 | Experiment freeze for paper |
| **Nov 12** | **Paper deadline** (≤3,000 words, Kaggle Writeup) |
| Nov 25 | Entry + team-merger deadline |
| **Dec 2** | **Main track final submission** |

## Constraints
- Cash: **$50 max**. It is a reserve, not a budget. Earliest spend is ~week 5, and only on what free compute can't do.
- Time: ~5 hrs/day.
- Team: solo for now (may ask chachu).
- Model fixed: `gemma-4-31b-it-qat-w4a16-ct`.
- Eval: 4×L4, offline, 12 h for all tasks.

## Rules summary (read 27 Sep)
- **1 submission/day**, and we pick **2 finals**. Every submission is precious, so a local eval harness is mandatory.
- **External data and models are allowed** if public/free or minimal cost ("reasonably accessible to all").
  - Public trajectory datasets are OK.
  - Teacher-model outputs are OK under the competition rules, but check each provider's ToS on training with outputs.
- **Winners must open-source** (OSI licence, commercial use allowed) their training + inference code and give a reproducible write-up. Keep the repo clean and reproducible from day 1.
- Team max is 5. Merging requires combined submissions ≤ days × 1, so if chachu joins, only one of us submits until we merge.

## Harness key facts (HARNESS_README, read 27 Sep)
- **Local eval exists**: the `swegemma eval` CLI runs on the published training tasks (`tasks.jsonl` + snapshots). It writes a summary, patches, test logs and full ATIF traces, which are the raw material for Map 2.
- **No Docker needed**: there is a `--sandbox subprocess` backend, so it should run on Kaggle or any box.
- **Context ceiling is 32,768 tokens** (vLLM max_model_len). ADK compaction kicks in at ~14k tokens, so training sequences must be ≤32k.
- **LoRA**: PEFT safetensors, max rank 128, max 8 adapters, total submission <3 GiB. A different adapter per agent is allowed.
- **Budgets per task** (`eval_config.yaml`):
  - Scoring defaults are 60 min / 100 tool calls.
  - The sample submission uses tiny values (1 min / 10 calls), which we must raise.
  - 12 h total across all tasks, so the per-task budget is a tuning knob.
- **Graph tools** are only offered if a graph/embeddings exist for the repo.
  - `search_similar_code` takes a **symbol name, not natural language** (offline embedding lookup by node key).
  - Node names resolve fuzzily.
- Skills may include `.py` scripts (an allowed extension). That could be a lever for custom retrieval or graph building. Still to verify where skill scripts run.
- Output caps: `read_file` 150 lines / 10k chars; command output 5k chars; 300 s command timeout.
- Scoring:
  - The patch is applied in a fresh container.
  - Test files and runner configs touched by the agent are reset.
  - Resolved = all target tests pass.
- Smaller Gemma 4 variants (12B, e4b, e2b, …) are registered for **local** eval, so small-model experiments can use the same harness.
- `--models-yaml` maps model aliases. We might be able to point local eval at a free Gemma API instead of a GPU (verify).
- Gotchas:
  - Scratch scripts go in /tmp.
  - Don't touch pytest.ini/conftest.py.
  - submit_patch and get_status are free.
  - Keep edits small to avoid tool-call truncation.

## Scorer & field intel (29 Sep — from public competitor repo github.com/happyc0der/gemma-swe-agent; VERIFY the starred items on the Kaggle Data page / forum)
- **Official harness** ships in a separate Kaggle dataset: `metric/gemma-4-developer-agent-wheelhouse`. It contains swegemma 0.2.7, adk-submission 0.2.11, adk-eval-core 0.1.0, google-adk 1.36.1 and vllm 0.19.1, and was re-uploaded 27 Sep with LoRA fixes under the same version numbers. Model on Kaggle: `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2`.
- ★ **Hidden test ≈120 tasks from PRIVATE repos (not fastapi/rich/requests/httpx)**, split 50/50 public/private LB. → Per-repo knowledge (T2) is dead; generalisation to unseen repos is what matters. Whether graphs exist for the hidden repos is unknown.
- ★ **Tasks run sequentially** under the 12 h cap → ~4.5–5 min per task. Time and tokens are the binding constraint.
- **No in-task context compaction**: ADK compacts only between invocations, so a trajectory hits the 32k wall after ~20 full-size tool outputs (ContextWindowExceededError). The fallback diff is still graded.
- **Leaderboard ~29 Sep**: top 0.13–0.15, public notebooks 0.08–0.12. All use temp 0.15–0.2, thinking off, 4–5 min/task, and a coder + analyzer AgentTool.
- **Hardware**:
  - vLLM W4A16 does NOT run on Kaggle T4 (needs sm80+).
  - llama.cpp with the official Q4_0 GGUF runs on T4×2 at ~12.8 tok/s. That is faithful-ish but slow: ~4 h per 33 tasks.
  - Kaggle L4×4 notebooks: refused for that competitor, but **available on Shrey's account** (1 Oct).
- **Harness replica**: gold patches pass on 109/129 tasks. The other 20 are environment-unstable (requests timeouts, rich pygments, fastapi starlette), so exclude them from local denominators.
- `--models-yaml` maps any alias to any LiteLLM model plus api_base/api_key (checked in the registry.py source), so the local eval can use an API backend.

## Dataset facts (verified by inspection script, 27 Sep)
- **129 training tasks**: fastapi 67 · rich 48 · requests 13 · httpx 1. Created Jul 2023 → Jun 2026. The hidden test set is separate (**same repos? unknown, must verify**).
- Each task is ≈ 0.8% of the local score. That is still too few to train on alone, so we need outside or synthetic data.
- **Graphs are per-commit**: 127 graph + 127 embedding files. base_commit matches a graph for **129/129** tasks (no stale graphs).
- **Only one edge label exists: `calls`** (452k edges total). There are no separate imports / defined_in / inheritance labels, although the README mentions them.
- Graph size:
  - fastapi ~3.9k nodes / ~2.5k edges (sparse; many docs_src tutorial nodes with no edges)
  - rich ~2k nodes / ~5.6k edges
  - requests ~730 nodes / ~1.8k edges
  - httpx 740 nodes / 1.6k edges
- Node ids drop path prefixes (e.g. fastapi docs_src files appear as `settings.tutorial001_py310.Settings`).
- Embeddings: 256-dim float32 per node, keyed by node id.
- **Every gold patch touches ≥1 .py file**. 91/129 are single-file fixes.
- Snapshots total ~21.5 GB (129 files). There are 124 wheels and a sample_submission.

## Graph & retrieval findings (27 Sep — `gold_graph_analysis.py` A1 + `verify_graph.py` A2)
Caveats: gold = code edited (including features). Issue-symbol extraction is regex-based; an LLM might extract better seeds.
- **Fix locations are in the graph**: 118/129 tasks map to ≥1 gold node (351 nodes, median 1/task). Only 4.8% of gold nodes are isolated.
- **✅ Confirmed — the graph is undirected in practice**: 100% of edges are reciprocal in all 127 graphs, plus 2,592 self-loops and 94 parallel duplicates. `get_code_neighbors` cannot tell callers from callees.
- **Direction is recoverable from source text** for 86–90% of edge pairs (A's text contains `B(` but not vice versa).
- **"calls" edges also include class → method containment** (e.g. `LookupDict → LookupDict.get`, `RouteContext → RouteContext.path`) and test → library edges, so the edge label is semantically mixed. Quantify later.
- **✅ Embeddings = text features smoothed over the graph (GNN-like)**:
  - Raw vectors are highly anisotropic: random-pair cosine 0.77–0.90, ~0.04–0.1 after centering.
  - Centered cosine decays with hop distance: ≈0.85–0.92 at 1 hop, ≈0 at 5+ hops.
  - A node's vector ≈ the mean of its neighbours' vectors (cosine 0.85–0.97, vs 0.03–0.18 for random nodes).
  - Some text signal remains: embedding-vs-TF-IDF Spearman 0.22–0.32; fastapi duplicate-code pairs 0.94–0.99.
  - Yet the top-1 embedding neighbour is a direct graph neighbour only 3–9% of the time.
- **Entry point is the bottleneck (clean seeds, A2)**: only **47.5%** of tasks name any non-generic, non-test, non-hub symbol that resolves to a node, and only **21.2%** name a gold node directly.
- **Reachability for the 31 tasks that have clean seeds but don't name gold**: 1-hop graph 22.6% · ≤2-hop graph 32.3% · embedding top-10 19.4% · **either 38.7%**. With hub-free BFS, unreachable goes 10 → 15 and 3-hop goes 8 → 5, so some "short" paths are hub shortcuts.
- **Headline**: gold named (25) + found cheaply by the tools (≈12) ≈ **31% of 118 tasks**. For the other **~69%**, the provided graph tools do not get the agent from the issue text to the fix within ~2 calls.
- **Centrality prior** (A1): gold nodes are more central than random ones. Degree AUC is 0.78 vs all nodes and 0.62 vs library-only nodes. The effect is modest and likely confounded by function size.

## Map 1 verdict (27 Sep — full grid in `claude/Map1_literature.md`)
- Lever 1 (entry grounding): **largely taken** (LocAgent BM25 index, LARGER lexical anchoring, GREPO query-rewrite anchors). Useful for the leaderboard, weak as headline novelty.
- Lever 2 (direction + typed edges): **open gap as far as found**, but Shrey judged it too obvious as a headline (28 Sep). Now used as an *ingredient* of T1.
- Lever 3 (GNN / structure ranking): partly taken (GREPO: GNNs beat LLM agents).
- Lever 4 (training to navigate): crowded (LocAgent SFT, ToolTrain SFT→RL, CodeScout RL).

## Deep research verdict (28 Sep — full doc in `claude/Research_2025_2026.md`)
- The 2026 frontier shows **SWE agents localise by names and memorised repo cues**:
  - Schrödinger's Repo: −6 to −14 pp under namespace remapping.
  - Jagged Frontier: up to −6.7 pp.
  - SHERLOC: 58% of recall is reachable from issue text alone.
  - Face/Off: fine-tuning on anonymised code does NOT fix it.
- All of these only *evaluate*. Nobody *trains* for name invariance in SWE agents.
- GRAIN (graph-ML, Aug'26) shows a **structure-invariance reward + isomorphic renaming** fixes exactly this brittleness, but only on synthetic graph problems (4B model).
- Delivery matters:
  - Agents ignore optional graph tools (CodeAnchor).
  - Structured findings transfer +6 pp to weaker repair models (SHERLOC).
  - Project-specific self-distilled skills give +5.8 pp and are model-specific (SkillForge).
- **Leading thesis T1 — "Structure over names"**: repair the graph (direction + edge types) → localiser emits a checkable evidence chain → train with a GRAIN-style structural reward + consistent namespace remapping augmentation (rejection-sampled SFT on 31B; GRPO ablation on small Gemma).
- T2 (per-repo LoRA vs text skills) only if the test repos = the train repos.
- T3 (evidence-hit vs conversion diagnostics) is a paper section.

## Free compute stack
- Kaggle TPU v5e-8 (training, JAX/Keras). Weekly quota: **TBD**
- Kaggle GPU 30 h/wk
- Google AI Studio Gemma 4 31B free tier (~1,500 req/day) · OpenRouter free (200 req/day)
- ~~Modal~~ out of scope: it needs billing enabled; only $1 credit (29 Sep) · Lightning 15 credits/mo
- TPU Research Cloud: apply
- Oracle $300 / 30 days: sign up only in the week we need it
- Kaggle leaderboard submissions = free full evals on real hardware

## Working agreement
- Claude writes the code by default. Shrey will say explicitly when he wants to write a specific part himself.
- Chats: one per topic. When a chat gets long, ask for a handoff summary, paste it into the Decision/Experiment log here, then start a new chat.
- Claude Code:
  - `CLAUDE.md` in the repo root; specs in `docs/`.
  - `/clear` between tasks.
  - Never paste raw logs or trajectories; summarise them via scripts.

## Research spike checklist (Sep 28 – Oct 1)
- [x] Rules tab → what's allowed (see Rules summary)
- [x] HARNESS_README read (see Harness key facts)
- [x] Dataset contents (see Dataset facts)
- [x] Gold-location graph analysis (A1)
- [x] Verification pass: symmetry, embedding origin, clean seeds (A2)
- [x] Literature gap (Map 1)
- [x] Deep research 2025–26 (code graphs, code comprehension, LLM graph reasoning) → `claude/Research_2025_2026.md`
- [ ] **Hidden test repos = training repos?** (rules / overview / discussion forum)
- [ ] Execution: run `swegemma eval --sandbox subprocess` on 2–3 tasks (Kaggle or local)
- [ ] Local eval via free API: can `--models-yaml` point at AI Studio / OpenRouter?
- [x] Code execution at inference: YES (run_command bash, pytest enabled in Container A, 300 s/cmd, offline, 4 GiB/2 vCPU)
- [x] Probe v1: data layout + sample submission OK. Harness is NOT in the competition data; it is in the wheelhouse dataset.
- [x] Probe v2: Gemma-4-31B on AI Studio DOES tool-call. But Shrey dropped the API route on 30 Sep: secret/venv friction, and local = faithful.
- [ ] **Baseline run `run_baseline_t4.py`** (Kaggle T4×2): llama.cpp + QAT Q4_0 GGUF, official harness in a uv venv, 36 stratified tasks, 40 calls / 12 min local. Outputs traces + submission.zip.
- [ ] Verify hidden-test claims (private repos? ~120 tasks? sequential?) on the Data page / forum
- [ ] Skill scripts: where do they run, and can they read the repo / graph files?
- [ ] Zero-shot probe: can Gemma-31B emit a parseable evidence chain (symptom → path → fix node)?
- [ ] Remapping feasibility: Schrödinger's tool on fastapi/rich with tests still green
- [ ] Baseline submission: sample config with raised budgets → leaderboard score
- [ ] TPU pilot:
  - Tiny JAX LoRA → export PEFT safetensors → does it load on the w4a16 base in vLLM?
  - Weekly TPU quota?
- [x] `trace_analyzer.py` built against the official ATIF v1.7 schema (read from the harness source) and tested on synthetic traces. It labels failure stage, evidence hit, errors + recovery, loops, nudges, context pressure, stray files.
- [ ] Leaderboard: `submission_baseline_v1.zip` (sample agent, no adapters, temp 0.2, thinking off, 4.5 min / 40 calls) → submit for the hidden-test baseline
- [ ] Failure analysis (Map 2): run the analyzer on the baseline traces
- [ ] Public trajectory data: licenses + conversion effort to the harness tool format (SWE-smith, ToolTrain's 28k localization set, CodeScout data)
- [ ] Apply to TPU Research Cloud; get an AI Studio key; sign up for Modal + Lightning

## Open decisions (lock after spike)
- Paper thesis. **T1 "Structure over names" REJECTED (29 Sep).** New direction from Shrey: (a) new retrieval / graph-generation technique for better structure + (b) post-training for **evidence-backed diagnosis**; claims must rest on dense, well-powered evaluation (not a handful of extra resolves on 129 tasks). Being scoped.
- Post-training method. Leaning towards rejection-sampled SFT (structure-filtered) on 31B, plus a small-Gemma GRPO ablation.

## Decision log
| Date | Decision | Why |
|---|---|---|
| Oct 6 | **Reset to notebook-style roadmap; scaffold frozen at v2; post-training first; work moves to a git repo + Claude Code** (`gemma4-swe-agent`: CLAUDE.md, docs/plan.md, docs/experiments.md, submissions v1–v6, Kaggle scripts, analysis tools). Training on Kaggle L4×4 only. Not public yet | scaffold changes plateaued (v2 = v5 = 0.08, v6 = 0.00); Shrey wants the traditional EDA → techniques → data → post-training flow |
| Oct 6 | **LB v6_trace = 0.00, ~10 h scorer runtime.** The skill-only localizer most likely fails in the scorer's Docker sandbox (every task ran to its timeout). v6 line dropped; skills are treated as unverified on the scorer | v5 (skill optional) = v2 on LB fits the same story |
| Oct 5 | Local v5_graph 33/129 (+12/−5 vs v2, p=0.14), LB 0.08. Localizer overran (48 calls), 61% of edits malformed. Runner compaction bug fixed (scorer = interval 15 / threshold 32768). Local set no longer predicts the LB → renamed dev set (`build_devset.py`) | `docs/findings.md` |
| Oct 4 | Graph diagnosis: `search_similar_code` = exact-name lookup (phrases/bare names 0/461 calls); built v5 with an ast `code-map` skill | `docs/findings.md` |
| Sep 25 | Enter both tracks | Prize + research credibility |
| Sep 25 | $50 hard cap | Budget |
| Sep 26 | Safe core + ambitious add-on | Guaranteed deliverables with upside |
| Sep 27 | Paper topic: Graph Reasoning | Judges are graph-ML researchers; harness ships graph tools |
| Sep 27 | Claude writes code by default | Speed |
| Sep 27 | NVIDIA×Nebius hackathon dropped | Full focus on Gemma comp |
| Sep 28 | Paper may intersect topics; "imperfect graph measurement" rejected as headline | Too obvious; want novelty across code comprehension × graph reasoning |
| Sep 29 | T1 (name-invariance + structure reward) rejected | Optimises for a shift absent from the test set; no ground-truth chains so the reward is a proxy; "GRAIN for code" = application; 129 tasks can't power the claims |
| Sep 29 | Direction: new retrieval/graph generation + post-training for evidence-backed diagnosis | Shrey's call; needs dense per-claim metrics + synthetic task scale |
| Oct 3 | Graph check: all graph/emb files load fine (0 dead), so the 0-result `search_similar_code` cause is elsewhere → `probe_graph_tools.py`. Built **v4_pipeline** (SequentialAgent: localizer [read-only + graph, structured brief via output_key] → fixer [fresh context, edit/test/submit], 80 calls), validated with the official compiler + ADK 1.36.1. Gotcha: `thinking_budget: 0` fails sub-agent schema validation in adk-submission 0.2.11, so use `include_thoughts: false` only | combined change requested by Shrey |
| Oct 3 | **Local v2_sed: 26/129 (20.2%)**; on the common 97: 15 → 21 (only-v2 8, only-baseline 2, p≈0.11). Empty patches 41 → 20, repeats 48% → 28%. New top failure: wrong fix never tested (47/129, budget used up). Next: v3_verify (mandatory budget plan + one targeted test) | `claude/Map2_findings.md` |
| Oct 2 | **LB: v2_sed = 0.08** (baseline 0.00); rank ~651/1200; top 3 = 0.24 / 0.17 / 0.17. One change (remove read_file → grep/sed reads) moved hidden-test score ≈ +5 of ~60 public tasks. Local v2 run (with test-deps fix) started | confirms the read-loop diagnosis transfers to hidden repos |
| Oct 1 | Full 129-trace analysis: 32 fastapi tasks env-unverifiable (missing inline_snapshot/dirty_equals) → **15/97 = 15.5% on verifiable**. 42% end with an empty patch, driven by read loops (malformed read_file args, 47% of reads; resets; 48% repeat calls). Also malformed edit_file args (17 tasks). Next: v2_sed (remove read_file) + env fix | `claude/Map2_findings.md` |
| Oct 1 | **Baseline run 1 done: 15/129 local (11.6%)**, same config = 0.00 LB. Root causes: malformed read_file args (`start_line\"` keys silently ignored → reads stuck at lines 1–150) + context resets at ~14k tokens → amnesia loops. Localisation is fine (gold lines seen 70%), but conversion is 15.6% | `claude/Map2_findings.md` |
| Oct 1 | **Baseline backend = Kaggle GPU L4×4 (available to Shrey!)** with `run_official_l4.py`: scorer-faithful (wheelhouse vLLM 0.19.1, W4A16, TP=4, util 0.80, multi-LoRA, sequential, offline agent). TPU/T4 scripts shelved | Same hardware + stack as the scorer; LoRA-per-agent testable locally |
| Sep 30 | Leaderboard: submission_baseline_v1 (sample prompt, no adapters) = **0.00** | Consistent with others' sample-prompt result; pipeline works, prompt doesn't |
| Sep 30 | Baseline backend = Kaggle TPU v5e-8 + vLLM-TPU (`run_baseline_tpu.py`); T4 llama.cpp script kept as fallback | ~10× faster per stream, runs tasks in parallel → all 129 tasks in one 9 h session; same vLLM parsers as the scorer; keeps GPU hours free. TPU quota ~20 h/wk is shared with training |
| Sep 30 | API backend dropped; baseline runs locally on Kaggle T4×2 (llama.cpp Q4_0) | Faithful weights, no key friction; slow (~12 tok/s) so 36-task subset. LB top now 0.17 |
| Sep 29 | The whole trajectory must be evidence-backed; order = baseline + full failure analysis (Map 2) → full literature check → method | Design must follow observed failures. Plan in `claude/Map2_plan.md` |

## Experiment log
| ID | Date | Hypothesis | Config / commit | Result | Takeaway |
|---|---|---|---|---|---|
| A1 | Sep 27 | Gold locations are structurally reachable from issue mentions via the call graph | gold_graph_analysis.py v1, 129 tasks | 24% gold named; 41% no seed; 1-hop 22% / emb top-10 19.5%; in = out degree | Entry-point grounding, not graph coverage, is the bottleneck |
| A2 | Sep 27 | Edges mirrored? Embeddings structural? Does A1 survive clean seeds? | verify_graph.py, 127 graphs, 118 tasks | 100% reciprocal edges; direction recoverable 86–90%; embeddings = graph-smoothed text; clean seeds 47.5%, gold named 21.2%, either-tool hit 38.7% | ~69% of tasks: tools can't bridge issue → fix within ~2 calls |

## Paper — living draft status
| Section | Status | Source |
|---|---|---|
| Title / abstract | not started | written last |
| Introduction + research question | not started | spike + thesis lock |
| Related work | **grid done** (Map 1 + deep research); prose not started | `claude/Map1_literature.md`, `claude/Research_2025_2026.md` |
| Method | not started | when method freezes |
| Experiments & results | not started | experiment log A1, A2 |
| Discussion / limitations | not started | failure analysis |
