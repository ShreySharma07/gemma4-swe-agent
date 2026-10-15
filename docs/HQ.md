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

## Probes & environment findings (Oct 6–7, Claude Code session)
**Kaggle CLI** set up locally: `.venv/bin/kaggle` (CLI 2.2.4, OAuth user `shreysharma07`). Kernels are pushed with `kaggle kernels push -p kaggle/<dir>` (metadata in `kaggle/devset/kernel-metadata.json`, `kaggle/eval_l4/kernel-metadata.json`).

**Scorer/environment facts (read from the harness source + competition files):**
- **Kaggle images now run Python 3.13** (Debian, `dist-packages`, no `ensurepip`). The wheelhouse has 11 cp312-only binaries (xgrammar, outlines_core, msgspec, cbor2, setproctitle, …). An all-at-once `pip install` of the wheelhouse aborts → harness missing. Fix in `build_devset.py`: install compatible wheels, fetch the same pinned versions of the rest from PyPI. **`run_official_l4.py` has the same bug** (and is offline) → needs fixing before the next local eval.
- **The scorer's task sandbox is `python:3.13-slim`** (`docker/Dockerfile.sandbox`), so 3.13 is the faithful version.
- Wheelhouse now ships **adk_submission 0.2.12** (we had 0.2.11).
- **Scorer test environment** (`container_setup._ensure_container_site_packages` + `sandbox/setup.py --fast-path`): Dockerfile pip packages (latest pytest, pytest-timeout 2.1.0, build backends) + the **highest py313-compatible version of every competition wheel** unpacked into site-packages (pytest/pluggy/pip/setuptools/wheel skipped) + editable install + `workspace_paths.pth` (workspace and each top-level dir on sys.path).
- **In `--sandbox subprocess` mode swegemma skips all of that**: task venvs just inherit the host's site-packages. So every local run so far tested against the Kaggle image's packages, not the scorer's (fastapi vs newer starlette, rich vs pygments, requests `src/` layout never importable → original requests tasks silently tested Kaggle's installed `requests`). Our earlier inline_snapshot/dirty_equals install also diverged from the scorer, which does not have them.
- The competition wheels are **not dependency-complete** (pydantic 2.13 needs `typing_inspection`, absent) → a wheels-only env can't import pydantic.
- The scorer pins starlette **1.6.0** (highest), which breaks older fastapi commits (`Router.__init__() got an unexpected keyword argument 'on_startup'`), and has no pytest-httpbin (most requests tests error). Those public tasks are likely unscoreable on the real scorer too.

**Renamed dev-set builds (`gemma4-devset-build` kernel, Kaggle CPU):**
| Ver | Change | Built | Gold pass (renamed) | Note |
|---|---|---|---|---|
| v1 | as written | 0 | – | harness install failed (cp312 wheels on Py 3.13) |
| v2 | install compatible wheels + PyPI pins | 128/128, 0 name leaks, 23 min | 48/128 | 70/80 failing originals fail too → env, not renaming |
| v3 | + apt `python3.13-venv` (ensurepip) | 128 | 48/128 | ensurepip was not the cause (subprocess mode skips installs anyway) |
| v4 | + scorer env (wheels site, isolated venvs, `setup.py --fast-path`) | 128 | 38/128 (fastapi 1, rich 32, requests 5) | pydantic broken (`typing_inspection` missing); orig-gold check had a bug (sent empty patches) |
| v5 | + fill only missing deps from PyPI (added typing-inspection, h11, blinker, greenlet); save `scorer_site.tgz` (17.5 MB); fix orig-gold | 128 | 47/128 (renamed, unused) | **originals in scorer-like env: 54/128 gold** (rich 39/48, fastapi 10/67, requests 5/13) |

**Original public tasks under the rebuilt scorer env (v5, Oct 7): 54/128 pass gold.** fastapi fails mostly on things the scorer image also lacks or pins (no `inline_snapshot`, starlette 1.6.0 breaks older commits, collection errors); requests needs the `httpbin` fixture (excluded by the scorer); rich fails pygments snapshot tests. Either the real scorer also can't verify most public fastapi tasks, or the real public image has extra packages we can't see — unresolvable without an LB probe, and irrelevant for hidden tasks (they ship their own wheels). Competitor's "109/129" was in a host-package env, not the scorer's. **Implication:** local scores need an explicit env + denominator; the 54 tasks are the scorer-faithful solvable subset. Per-task results: `data/runs/devset_v5/` (not committed).

## Decision log
| Date | Decision | Why |
|---|---|---|
| Oct 15 | **Step C v2 PASS / COMPLETE:** rc 0, all 122 planned optimizer steps, no budget stop; validation 0.7718 → 0.3133 → 0.2909 → 0.2883 → 0.2882. All checkpoints/final/serving files pass 410-module / 820 BF16 rank-16 tensor integrity and exact shape checks; final/serving SHA-256 matches. Global peak 18.88 GiB; raw median 294.8 tok/s, steady median 295.6 (247.7–324.0). Three intervals include validation/export overhead. Existing group-of-four scheduler used 972/974 train and 40/43 val windows while retaining all 190/10 tasks. Actual 7.8203 session h / 15.64 quota h, versus expected 9–10 / 18–20; quota 11.87 h remains. All outputs preserved and final report `docs/step_c_20261015_report.md`. No retry, D, or leaderboard run; monitor paused. | A/B PASS remain preserved; corrected submission explicitly approved after the earlier HTTP-400 rejection (zero GPU cost). No additional experiments authorized. |
| Oct 15 | **Shrey-approved C upload rejected with HTTP 400; no GPU run started and no retry.** Preflight: 27.51 quota h, A/B COMPLETE, exact prepared notebook/data verified, free CPU instrumentation smoke PASS. Original A status/lastRunTime and quota unchanged after rejection. | Likely source-size failure: embedded 3.39 MB NPZ makes CLI source 4.56 MB. Builder now rejects oversized embedding and enforces a conservative 1 MB source budget. Proposed private dataset + 45,840-byte notebook prepared locally with identical data hash; no upload/re-push. Cause inferred because CLI omitted server error body. Report: `data/runs/atul_step_c_20261015_v2/summary.json`; await corrected-submission authorization. |
| Oct 15 | **Step B PASS on Atul's account; C prepared, not launched.** Same Step A adapter verified by SHA-256; `check-l4` v1 COMPLETE / exit 0, vLLM up in 409 s, all five health checks true. Downloaded archive has 8 results/traces, tool calls on 8, tests on 7; one task exceeded 100 turns. 2/8 resolved is only a serving-smoke result. Runtime 31.91 min; quota cost 1.06 h; 27.51 h remain. | Results/report: `data/runs/atul_step_b_20261015_v1/validation/summary.json`. C's 37-cell notebook is staged separately with the exact §7.3 settings, original generated files preserved. Shrey's explicit approval is unconfirmed; no C, D, or leaderboard execution. |
| Oct 15 | **Step A throughput discrepancy resolved; A reassessed PASS and B authorized without repeating A.** Exact total workload 270,421 tokens; all-rank inferred elapsed-time bounds give 257.50–295.61 tok/s (median bounded 276.95–277.04), within 240–420. Original rank-0 metrics were undercounts; raw files and initial report remain intact. Source instrumentation now sums actual rank tokens, uses slowest synchronized elapsed time, and records audit fields and maximum memory. | Stdlib audit cross-checks all four plans/rates, recorded rank-0 metrics, source hashes, and rounding uncertainty; lightweight synthetic checks PASS. B uses unchanged A v1 adapter. C requires B PASS and explicit Shrey approval; no D or leaderboard work authorized. |
| Oct 15 | **Atul Step A v1 completed: GPU training and export work; B paused on the documented throughput gate.** rc 0, val 0.2557 → 0.1818, 410 modules / 820 tensors with valid shapes, 17.61 GB logged peak. Throughput logged 175.3–259.2 tok/s (median 193.2); schedule reconstruction identifies rank-0 shortest-window token undercount, with estimated all-rank rate 257.6–295.6 tok/s. No retry, B, C, D, or leaderboard submission launched. | Follow strict stage gates. Proposed fix is distributed token/time accounting plus CPU verification before any approved retry. Quota charged 1.43 h; 28.57 h remain. Small results in `data/runs/atul_step_a_20261015_v1/`; see post-training guide decision/ledger. |
| Oct 12 | **v7_noedit = 0.15 on the leaderboard (v2 = 0.08). v7 is the new base.** The training-free change came from the m1-drift measurement. Next LoRA targets fix conversion on top of v7 (no edit_file); data must use v7's edit style (python snippet-replace / write_file) | Removing the tool with 61.5% format drift roughly doubled the hidden-test score. Harness-update confound noted (v2 not re-scored) |
| Oct 10 | **Submitted v7_noedit to the leaderboard** (v2 minus edit_file). **The official harness was updated on Oct 9:** the wheelhouse now ships swegemma **0.2.11** / adk-submission **0.2.13** (was 0.2.7 / 0.2.12). `swegemma.models.registry/discovery` moved to `adk_submission.discovery` + `swegemma.cli._build_cli_model_registry`; `submit_patch` now strips protected/test files from the patch. `tools/validate_submission.py` is stale; validate with `kaggle/validate/make_validate_kernel.py` (Kaggle CPU, official wheels). Submission files must be named exactly `submission.zip` | m1-drift showed edit_file is the drift source. The harness change may shift all scores, so compare v7 against a fresh v2 baseline if results are ambiguous |
| Oct 10 | **Research-direction review (recommendation, not an adopted change):** retain v2; validate evaluation and tool rendering, finish training/export feasibility, then test a small format-reliability LoRA. Validate the existing execution instrument and gate an edit-necessity credit audit on a bounded CPU pilot. Detailed protocol: `docs/research_direction_2026-10-10.md`. No jobs or submissions launched. | Observed edit/test failures are actionable; BMM failed its pilot and graph repair was inconclusive. Measured 8k training is ~82 tok/s; larger training and useful adapter transfer remain unproven. Novelty claims require fresh primary-source checks. |
| Oct 10 | **Training envelope fixed: FSDP-QLoRA (NF4) on 4×L4, sequences ≤8k tokens.** Measured 421 tok/s at 4k and 244 tok/s at 8k with CPU activation offload; 16k OOMs. 1k trajectories × 8k ≈ 9 GPU-h per epoch. NF4 is an acceptable base: 0.817 vs 0.851 (bf16 QAT) top-1 agreement with served W4A16. Training data must be cut to ≤8k-token windows (prefix-fit / step windows) | Probe p0b-v4. Throughput was the unknown blocking every SFT plan |
| Oct 9 | **BMM K3c pilot FAILED its pre-registered gate.** Sibling separation is 0.514 (CI 0.40–0.59), so per pre-registration C-BEH (signed behavioural evidence, BMM's novel piece) is demoted before any GPU spend. Issues almost never state "expected but absent" effects (10/833 pairs), and 29% of pairs tie under AST q. Details in experiments.md k3c and `reports/Behavioural map matching for code.md` | Cheapest test of Prop. 1 on real issues. With static effects + regex, the behavioural channel does not separate the culprit from name-similar siblings better than chance |
| Oct 9 | **Phase 0b training probe (p0b) = partial GO.** 31B NF4 + LoRA r16 trains on 4×L4. The head_dim-512 global layers work with the memory-efficient SDPA kernel up to 32k. The adapter export matches the 410 served modules. With a layer split over the 4 GPUs it reaches 8k tokens at only ~82 tok/s, and 16k OOMs on the last GPU. FSDP throughput and the NF4-vs-W4A16 agreement are **still unmeasured** (stage crashes, see experiments.md p0b). Host RAM is 188.7 GB, so offloading activations to CPU is an option. Training base = Kaggle `gemma-4-31b-it-qat-q4_0-unquantized/2`, quantised to NF4 on load (12 min). Add-on wheels come from kernel `gemma4-train-wheels`. | Throughput decides every training plan. The layer-split rate gives ~27 h per 1k×8k trajectories, so FSDP is required. Rerun the fixed FSDP and mismatch stages after the quota reset |
| Oct 7 | **✅ Phase 0 LoRA serving = GO** (check-l4 v1, 4× L4, offline, scorer env restored from `scorer_env.tgz`). Official VllmServer with scorer settings + 2 synthetic rank-8 PEFT adapters (real Gemma-4 shapes, 410 modules): zero_lora output == base (text + tool call identical), noise_lora output differs, official compiler + `resolve_swegemma_adapter` routes `adapter: noise_lora` → `openai/noise_lora`. vLLM start 480 s, 18.9 GB/GPU, ~34 tok/s single-stream decode. Report: `results/runs/p0_lora_serving/lora_probe.json` | post-training is viable on the scorer path; next = Phase 0b training feasibility |
| Oct 7 | **L4×4 needs the competition attached.** Shrey's competition notebook `check-l4` (competition_sources = gemma-4-developer-agent, machine_shape NvidiaL4) gets L4 when pushed via CLI; my probe kernels lacked competition_sources and fell back to T4. Pushed the offline LoRA probe into `check-l4` (`kaggle/probes/check_l4/`, notebook cell writes the probe to a file and runs it, since it re-execs into the venv); fixed its model to `gemma-4-31b-it-qat-w4a16-ct/2` (it had the 26B-A4B q4_0 model). Backup of the previous notebook: `data/backups/check-l4_before_push.ipynb` | GPU runs can now be launched from the CLI |
| Oct 7 | **Scorer env built** (`gemma4-scorer-env` v1, CPU, 5.5 min): `scorer_env.tgz` 5.8 GB, Python 3.12.3, 251 packages, torch 2.10.0+cu128 / vLLM 0.19.1 / ADK 1.36.1 / uvloop / litellm all import, 0 missing requirements after 4 fill rounds. Offline LoRA probe (v8) pushed with it as `kernel_sources`; needs an L4×4 run from the web editor | unblocks every offline L4 run |
| Oct 7 | **Offline L4 workflow:** L4 sessions have no Internet, so the scorer's vLLM-host env is built once on a CPU kernel with Internet (`kaggle/env/build_scorer_env.py` → kernel `gemma4-scorer-env` → `scorer_env.tgz`: uv Py 3.12 venv at /tmp/scorer_host + uv's standalone Python, restored with `tar -C /`). L4 kernels attach it via `kernel_sources` and run offline (LoRA probe updated; eval runner to follow). L4 runs are started from the web editor | the probe's online install would fail on L4 |
| Oct 7 | **L4×4 cannot be requested via the Kaggle CLI/API from this account**: probe v7 with `machine_shape: NvidiaL4` was recorded as NvidiaL4 but still allocated 2× T4 (guard aborted in seconds, ~0 quota). Workflow: push code with the CLI, then in the web editor set Accelerator → GPU L4 ×4 and Save & Run All. CPU kernels via CLI work fine | earlier L4 runs were all started from the UI |
| Oct 7 | **CLI-pushed GPU kernels silently ran on T4×2, not L4×4.** `"machine_shape": "NvidiaL4X4"` is not a valid value; Kaggle fell back to `NvidiaTeslaT4` (server metadata). LoRA probe v5–v6: install + vLLM banner fine, then `World size (4) > available GPUs (2)`. Valid value per Kaggle docs: `NvidiaL4` (set in both kernel-metadata files); probes now abort in seconds unless they see 4× L4. **GPU quota: 24.5 h used, 5.5 h left until Oct 10** | every probe result before v7 was on the wrong hardware; quota is now the binding constraint this week |
| Oct 7 | **LoRA probe v4: install now works** (Py 3.12 uv venv, wheelhouse `--no-deps` + 162 missing deps filled, 1.7 min; torch 2.10.0+cu128, vLLM 0.19.1, ADK 1.36.1 import). Synthetic adapters built from the real checkpoint: **410 target modules** (q/k/v/o/gate/up/down; two attention shapes: q 8192 or 16384, k 4096 or 2048), **122.6 MB per rank-8 adapter** → rank 64 ≈ 1 GB, so ≤2–3 rank-64 adapters fit in 3 GiB. `discover_adapters` finds both. vLLM then died on `No module named 'uvloop'` (filler ignored extras like `fastapi[standard]`). v5 follows requested extras transitively | install path for every GPU run on the new image is now known |
| Oct 7 | **LoRA probe v3 failed in 12 s on a second wheelhouse conflict** (xgrammar 0.2.5 wants transformers <4.56, vllm 0.19.1 wants ≥4.56; wheelhouse ships transformers 5.13.1). Conclusion: the wheelhouse is not a consistent dependency set; the scorer installs it unresolved. v4 mirrors that: torch normally → all wheelhouse wheels `--no-deps` (exact scorer versions) → add only missing requirements round by round, never changing an installed version | stop whack-a-mole with overrides |
| Oct 7 | **LoRA probe v2: uv failed in 6 s on a conflict inside the official wheelhouse** — google-adk 1.36.1 needs starlette <1, prometheus-fastapi-instrumentator 8.1.0 (vLLM dep) needs starlette ≥1, so the scorer must install the wheelhouse unresolved (`--no-deps`). v3 adds a uv override `starlette>=0.49.1,<1` | install fidelity; vLLM start will show whether the override is safe |
| Oct 7 | **LoRA probe v1 failed before testing anything:** pip on the Py 3.13 image hit `ResolutionTooDeep` after 52 min resolving vllm + adk against Kaggle's preinstalled packages. v2: isolated **Python 3.12 venv via uv** (wheelhouse is cp312 = likely the scorer's vLLM host Python; task sandbox is 3.13), all wheelhouse wheels as-is + torch 2.10 cu128, script re-execs inside it. Same install path must go into `run_official_l4.py` | install method is the blocker for every GPU run on the new image |
| Oct 7 | **Renamed dev set dropped as the memorisation probe** (Shrey). Build v5 to be stopped / ignored; only its by-products matter (`scorer_site.tgz` = scorer test env for the L4 runner, gold check of the 128 original tasks = solvable denominator) | renaming changes names only; file layout, code structure and style stay recognisable, so it cannot separate memorised structure from skill. Generalisation will be measured on genuinely unseen repos instead |
| Oct 7 | **Local environment ≠ scorer environment, for every run so far.** subprocess sandbox inherits Kaggle's packages; scorer uses highest-version competition wheels + fast-path setup. Rebuilding the scorer env (`use_scorer_env` in `build_devset.py`) and will port it to `run_official_l4.py` | local numbers (15/26/33) were measured in a different dependency env from the scorer |
| Oct 7 | Phase 0 LoRA serving probe started (`kaggle/probes/probe_lora_serving.py`): synthetic PEFT adapter with real Gemma-4 module shapes → served via the official VllmServer + adapter manifest → base vs adapter outputs compared | go/no-go for all post-training |
| Oct 7 | Strategy note: the 129 public tasks are a microscope (behaviour/failure analysis + harness check), not training data. Generalisation to private repos must come from repo-independent process skills (repro → search → small reliable edits → verify → submit) trained on diverse-repo trajectories. Renamed dev set is read on localisation metrics (steps-to-gold, gold-seen), not just resolves. v5 = v2 on LB is NOT evidence that local doesn't transfer (skill likely didn't run on scorer; LB public half ≈60 tasks can't see +3) | discussion with Shrey |
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


## Atul Baliyan — Gemma 4 Post-Training Experiments

Artifact audit completed after the October 15 runs (October 16 local time): **all required artifacts COMPLETE locally**, including C checkpoints 40/80/120, final and serving copies. Recovered 33 missing CPU-smoke files (2,832,825bytes); no healthy artifact or original report was overwritten. See [artifact audit](atul_training_artifact_audit_20261015.md), [completed training report](step_c_20261015_report.md), and the per-file inventory in `data/runs/atul_artifact_audit_20261015/summary.json`.

- [Step A — gemma4-train-lora v1](https://www.kaggle.com/code/atulbaliyan/gemma4-train-lora): PASS after exact throughput reconstruction; original 193.2 tok/s median preserved, corrected median about 277.0. Validation 0.2557→0.1818; both 410-module adapter copies intact.
- [Step B — check-l4 v1](https://www.kaggle.com/code/atulbaliyan/check-l4): PASS with A adapter; eight results/traces, five health checks true, 2/8 resolved. One turn-limit error; score is serving-smoke evidence, not generalization. Original archives pass integrity checks.
- [CPU instrumentation — gemma4-train-smoke v1](https://www.kaggle.com/code/atulbaliyan/gemma4-train-smoke): PASS; distributed token SUM/elapsed MAX verified. Recovered tiny model/adapter exports, configuration/data and captured sources. No GPU cost.
- [Step C — gemma4-train-lora v2](https://www.kaggle.com/code/atulbaliyan/gemma4-train-lora): COMPLETE/PASS, rc 0, 122/122 steps; validation 0.7718→0.3133→0.2909→0.2883→0.2882. Steady median 295.6 tok/s, global peak 18.88 GiB. Scheduler processed 972/974 train and 40/43 validation windows while retaining all 190/10 tasks. Actual 7.8203 session hours / 15.64 quota-hours; last recorded remaining 11.87quota-hours.

Each C export has 410 modules/ 820 BF16 rank 16 tensors and is 244,982,352 bytes. Final/serving SHA-256: `a116b6be160fffcc98cdd2a4ab6cfaffafcfc54ac6709a8a5d95d95ebab544c6`. The approved 200-task NPZ hash also matches. Current B/CPU/C remote nested output sizes match local files. Historical A re-fetch is unavailable through the installed CLI/API; its original local evidence and adapters are intact. Remote checksum headers are absent; the audit distinguishes local hash verification from remote size comparison.

The C adapter’s serving/issue-solving performance remains untested. The paired evaluation proposal is prepared separately; no evaluation, Step D, leaderboard submission or new training is authorized or launched by this artifact audit. Existing historical entries above are preserved, including their then-current approval/gate states.
