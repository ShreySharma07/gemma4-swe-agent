# Map 2 — Baseline failure analysis plan (29 Sep)

**Goal.** Before choosing a method, build an evidence-backed picture of *where* Gemma-31B fails, *why*, *what happens when it breaks*, and *what would have rescued it*. Every label must come from the trace or the gold data, not from intuition.

## Answered: can the agent execute code at inference? — YES (from HARNESS_README)
- `run_command` runs bash in `/workspace`: 300 s per command, 5k chars of output.
- `enable_sandbox_testing=True` by default, so pytest works in Container A.
- Dependencies are pre-installed and the sandbox is offline, with 4 GiB RAM / 2 vCPU per container.
- Skill scripts are "executed in a sandbox" (competition PDF), but *what they can see* is still unknown. The probe greps the source for this.
- Still to confirm: whether `coverage` is installed in the task venv (`sys.settrace` is always there). The probe checks this.

## Step 0 — Environment probe (`probe_env.py`, Kaggle CPU)
Checks:
- swegemma install
- the sample submission
- the models.yaml format
- how skill scripts run
- graph path resolution
- the ATIF schema
- a Phase-2 smoke test with `--skip-agent-patch` (no LLM needed)
- coverage / settrace availability

## Step 1 — Model backend (decide after the probe)
| Option | Fidelity | Cost | Notes |
|---|---|---|---|
| **A. Modal, vLLM with the exact QAT model + competition flags** | Highest (same weights, parser, 32k context) | Free $30/mo credits ≈ 10–14 GPU-h (L40S/A100) | Needs snapshots on a Modal volume and the subprocess sandbox |
| B. AI Studio Gemma-4-31B API via `--models-yaml` | Medium (not QAT; different tool-call path) | Free (~1.5k req/day ≈ 30–40 tasks/day) | Good for a 5-task pilot; loses the vLLM-specific truncation failures |
| C. Kaggle T4×2 | Low / unlikely | Free | W4A16 Marlin kernels need sm80+; T4 is sm75 |
| D. Kaggle leaderboard submission | Real hardware, hidden test | 1 per day | Score only, no traces. Use for the baseline number. |

**Plan:**
1. Pilot 5 tasks on B (or A).
2. Full 129-task run on A.
3. Submit the same config to the leaderboard (D).
4. Re-run 20 tasks ×3 seeds to measure run-to-run noise.

**Budget:** run generously (100 calls / 30 min) and log per-step, so shorter budgets can be simulated after the fact from one run.

## Step 2 — Trace analyzer (built after the first real trace, so it matches the real ATIF schema)
The analyzer labels each trajectory stage by stage, joining it with our gold nodes (A1).

| Stage | Automatic signals |
|---|---|
| Understand | Symbols extracted from the issue; clean-seed presence (A2) |
| Localize | Step at which the gold *file* first appears; step at which the gold *function* first appears (read range or graph output) = **evidence hit**; files and functions read that are not gold |
| Reproduce | Wrote/ran a repro? Did the repro fail before the edit (bug confirmed)? |
| Edit | Edited the gold file / gold function? Edit precision; number of edit attempts; FileEditError count |
| Verify | Ran tests after the last edit? Did they pass? Tests chosen vs target tests |
| Submit | Explicit submit / fallback / none; empty patch; stray files in the patch (repro scripts) |
| Outcome | Resolved / tests fail / patch-apply fail / no patch / budget exhausted |

**Break events** (what happens when it breaks):
- tool errors (CommandError, FileEditError, Timeout)
- the 3 nudge types (truncated tool call, MAX_TOKENS, plain stop)
- ADK compaction events, followed by re-reading the same content ("amnesia")
- repeated identical calls (loops)
- hallucinated paths or symbols
- misuse of the graph tools (natural-language query sent to `search_similar_code`)
- budget warnings

**Recovery:** after each break, did the agent recover within k steps, and how many steps and tokens did recovery cost?

**Conversion metrics** (GraphDancer lens):
- P(resolved | evidence hit), P(resolved | no evidence hit)
- P(correct edit location | evidence hit)
- These separate *never found it* from *found it but didn't recognise it* from *recognised it but broke it*.

## Step 3 — Counterfactual "what would have rescued it" replays (causal, not correlational)
On failed tasks, re-run with one oracle injected at a time:
1. Gold file.
2. Gold function(s).
3. Root-cause text written from the gold patch (no code).
4. Target test command.
5. Resume from the step before the first break, with the break removed (e.g. a clean context instead of post-compaction amnesia).

The oracle that rescues the most tasks identifies the true bottleneck stage. The pattern of which oracle rescues which task is the input to the method design.

## Statistical honesty
- 129 tasks give ±8 pp on a resolve rate, so resolve counts are **descriptive only**.
- Claims will rest on **dense units**: ~thousands of tool steps, 351 gold nodes, per-break recovery events, and per-oracle rescue counts with paired tests.
- Later: synthetic tasks (bug injection on the 4 repos) for scale.

## Deliverables
1. `probe_report.txt`
2. Pilot traces (5 tasks)
3. `trace_analyzer.py`
4. Full run + taxonomy tables
5. Oracle replays
6. Map 2 findings doc, which drives the full literature check and the method design
