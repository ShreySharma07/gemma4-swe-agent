# CLAUDE.md: Gemma 4 Developer Agent (Kaggle)

Post-train `gemma-4-31b-it-qat-w4a16-ct` into an offline SWE agent that resolves GitHub issues.
- **Main track:** leaderboard = % of hidden issues resolved. Final submission **Dec 2, 2026**.
- **Paper track:** ≤3,000-word write-up, deadline **Nov 12, 2026**.
- Owner: Shrey (solo, ~5 h/day, $50 hard cap). Claude writes code by default.

Read `docs/plan.md` (the roadmap) and `docs/experiments.md` (every run so far) before proposing work.

## Hard constraints of the scorer (do not violate)
- **Hardware / serving:** 4×L4, vLLM 0.19.1, TP=4, `max_model_len=32768`, gpu_mem 0.80. gemma4 tool + reasoning parsers. Multi-LoRA: ≤8 adapters, rank ≤128, PEFT `.safetensors`, submission <3 GiB.
- **Tasks:** ~120 hidden tasks from **private repos** (not fastapi/rich/requests). Run **sequentially** under a **12 h total cap**. Each task runs in a locked-down Docker container (gVisor, offline).
  - Budget for worst-case runtime: `max_time_minutes × 120` must stay well under 12 h.
  - v6 took ~10 h and scored 0.00.
- **Submission format:** declarative only. `agent.yaml` (LlmAgent / Sequential / Parallel / Loop), prompts via `!include`, `sub_agents/`, `configs/`, `eval_config.yaml`, optional `skills/` and `adapters/`. No Python entrypoints.
  - Allowed extensions: yaml/yml/md/txt/py (skills only)/json/safetensors.
- **Tools:** run_command, read_file, edit_file, write_file, submit_patch, get_status, plus graph tools (`search_similar_code`, `get_code_neighbors`, `get_code_subgraph`) when graph files exist.
  - Command output is capped at 5,000 chars; 300 s per command.
- **Context:** the scorer's compaction is interval 15 / threshold 32768, so effectively **no compaction inside a task**. The context only grows. Keep tool outputs small and trajectories short.
- **Leaderboard:** 1 submission per day; we pick 2 finals. Each submission is a full ~10 h scorer run. Never submit anything whose scorer path is unverified when it matters.
- **Generation config:** `thinking_budget: 0` is rejected by sub-agent validation. Use `thinking_config: {include_thoughts: false}`.
- **ADK templating:** `{problem_description}` and optional `{var?}` work (ADK 1.36.1). `output_key` saves a sub-agent's final text to state. `include_contents: none` gives a fresh context.

## Verified findings (details: `docs/findings.md`)

### Run history
| Run | Local | LB |
|---|---|---|
| baseline_v1 | 15/129 | 0.00 |
| v2_sed | 26/129 | **0.08** |
| v5_graph | 33/129 | 0.08 |
| v6_trace | not run locally | 0.00 (~10 h) |

**Local gains have stopped transferring to the LB.** The likely cause is that the public repos are memorised by Gemma. The renamed dev set (`kaggle/devset/build_devset.py`) is meant to test this. It has not been run yet.

### What helped
- Removing `read_file`. Gemma garbles its arguments (`start_line"` keys, 47% of reads), which caused read loops.
- grep/sed reads instead.

### Known failure mechanisms
1. **Garbled tool calls.** Code containing quotes leaks into the argument keys. v5: 61% of `edit_file` calls, concentrated in ~14 tasks. This is a natural first LoRA target: tool-call format reliability.
2. **Budget overruns.** A localizer sub-agent ignores its call budget (asked ≤25, averaged 48) and starves the fixer.
3. **Exact-repeat loops.** 28–48% of calls.
4. **Wrong fix never tested** (v2: 47/129).
5. **Graph tools.** `search_similar_code` is a lookup table keyed by fully qualified node name; there is no text encoder.
   - Phrases and bare names return 0 results.
   - The installed swegemma has no `resolve_node_name`.
   - Agents mostly ignore the graph tools.
6. **Skill scripts are UNVERIFIED on the scorer.**
   - v6 made a skill the localizer's only tool and scored 0.00 in ~10 h, consistent with skill execution failing in the scorer's Docker sandbox.
   - v5 had the skill as an optional extra and scored the same as v2.
   - Treat `skills/*.py` as broken on the scorer until a probe proves otherwise.

### Local runner fidelity
- `kaggle/eval_l4/run_official_l4.py` reproduces the scorer stack, but uses the **subprocess** sandbox, not Docker.
- The local test-deps fix installs inline_snapshot and dirty_equals.
- About 20 of 129 tasks are environment-unstable locally (gold patch fails).

## Repo map
- `submissions/<name>/`: every submitted agent, unzipped. Zip the contents (not the folder) to submit.
- `kaggle/eval_l4/run_official_l4.py`: scorer-faithful local eval on Kaggle L4×4. Set `VARIANT`, or set `SUBMISSION_DIR` to an attached dataset. `DEVSET=True` evaluates on the renamed dev set. Outputs `official_l4_<name>.tgz`.
- `kaggle/devset/build_devset.py`: builds the renamed (de-memorised), gold-verified dev set (Kaggle CPU).
- `kaggle/probes/`: environment and graph probes. `kaggle/archive/`: shelved T4/TPU runners.
- `analysis/trace_analyzer.py`: per-task failure stage, evidence hits, loops, malformed calls, context resets. Needs gold patches from `tasks.jsonl`.
- `analysis/compare_runs.py`: paired comparison of runs (McNemar) plus behaviour metrics from traces.
- `tools/validate_submission.py`: compiles a submission with the official `adk_submission` compiler. Needs google-adk 1.36.1 + adk-submission from the wheelhouse.
- `tools/make_zip.py`: zips a submission directory.
- `results/runs/<id>/`: small per-run artefacts (task_results, summaries). Raw tgz files go to `data/` (gitignored).
- `docs/`: HQ (decision log), plan, experiments, findings, literature.

## Working rules
- **Never paste or print raw traces or logs into the conversation.** Results tgz files can be hundreds of MB. Extract into `data/runs/<id>/`, process with scripts, print only summaries.
- **One change per experiment.** Predict the outcome before running. Log every run in `docs/experiments.md`: hypothesis, config, local, LB, takeaway.
- **Report significance.** Report paired results with McNemar p. With ~120 hidden tasks, 1 task ≈ 0.8 pp, so don't over-read small differences.
- **Validate before handing over.** Every new submission must compile with `tools/validate_submission.py`, and its worst-case runtime must be computed.
- **Scorer path first.** Anything that depends on scorer-side execution (skills, adapters, new config fields) gets a cheap LB probe before we build on it.
- **Keep it open-sourceable.** Winners must release training + inference code under an OSI licence, so the repo must stay reproducible.
- Update `docs/HQ.md` (decision log) at the end of each working session.
