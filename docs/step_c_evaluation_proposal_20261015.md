# Proposal: evaluate the completed Step C adapter

Status: proposal only, October 15, 2026. No Kaggle jobs, model runs, evaluations, uploads, or leaderboard submissions were launched during this review. Execution requires new explicit authorization. The completed training report and adapter are unchanged.

## Recommendation

Run one paired pilot on **24 fixed, previously newly valid public tasks**, comparing `v8_conv_lora` with a fresh `v7_noedit` baseline. Each arm gets one attempt per task, identical task order and evaluation settings. Integrate the Step C serving check into the candidate arm's first eight tasks, so a separate smoke job does not consume quota or reduce the main comparison's sample size.

Plan for **5.8–10.2 Kaggle quota-hours**, with a proposed maximum allocation of **10.4 quota-hours** and **1.47 hours reserved** from the last measured 11.87. This is an exploratory experiment that tests transfer to tasks and repositories absent from the adapter's data. It cannot establish leaderboard improvement or broad generalization by itself.

## What the review established

- Step C completed all 122 optimizer steps, rc 0, without a budget stop. Validation loss fell monotonically from 0.7718 to 0.2882. The scheduler omitted two training and three validation windows, while retaining all 190 training and ten validation tasks.
- Final adapter: `data/runs/atul_step_c_corrected_20261015_v2/adapters/conv_lora/`, 244,982,352-byte safetensors file; 410 modules, 820 BF16 tensors, rank 16, alpha 32. Its SHA-256 was recomputed during this read-only review and matches `a116b6be160fffcc98cdd2a4ab6cfaffafcfc54ac6709a8a5d95d95ebab544c6`. Existing validation proves matching final/serving copies and all exact tensor shapes.
- `diff -ru submissions/v7_noedit submissions/v8_conv_lora` shows only `adapter: conv_lora` added to the root agent. Prompts, sampling, tools and task budgets otherwise match. The code-analyzer subagent remains on the base model in both arms; applying LoRA to that role would be a separate experiment.
- Current evaluation source is `kaggle/eval_l4/evalkit/` plus `make_eval_notebook.py`. The old `run_official_l4.py` mentioned in older documentation is absent. `tools/validate_submission.py` uses obsolete imports; the documented replacement is the official-wheel compiler workflow under `kaggle/validate/`.
- Step B served the earlier A adapter, not the final C adapter. C therefore still needs an actual serving check. Lower NF4 training loss does not establish successful transfer to the served W4A16 checkpoint.

## Dataset independence and selection

The stdlib-only audit decoded task identifiers from the exact Step C NPZ, including both training and validation tasks. It normalized external IDs (`owner__repo-issue`) and competition IDs (`fastapi_11194`, etc.) to canonical repository/issue identities, rather than comparing incompatible raw strings.

| Existing candidate pool | Tasks | Step C task overlap | Step C repository overlap | Step A full-source task overlap |
|---|---:|---:|---:|---:|
| All overlay-valid public tasks | 114 | 0 | 0 | 21 |
| Newly valid tasks | 60 | 0 | 0 | 0 |
| Earlier valid set | 54 | 0 | 0 | 21 |

Step C's 200 tasks span 177 canonical repositories. The 114-task pool spans fastapi/fastapi, Textualize/rich, psf/requests and encode/httpx. Its repository identities are disjoint from Step C's entire training/validation set. The Step A overlap column conservatively uses the complete 21-task source NPZ, rather than only A's selected windows.

Prefer the 60 newly valid tasks: they avoid A's source task IDs and were not part of the old 54-task rejection-sampling benchmark. They are still development tasks used in environment validation, not a pristine blind test. The old 54 may be useful for later historical diagnostics but should not be the next primary pilot.

The proposed manifest is `docs/step_c_eval_pilot_tasks_20261015.json`: **14 fastapi, six rich, four requests**, selected by the smallest SHA-256 of `gemma4-C-pilot-v1|instance_id` within each repository. The selection uses no task-solving outcomes. All rich and requests tasks in the newly valid pool are included; fastapi is sampled from 50. There are no newly valid httpx tasks. This intentionally broadens repository coverage relative to the 60-task pool's 50/6/4 distribution; report pilot micro and per-repository results without extrapolating its aggregate to all 114 tasks.

Manifest SHA-256: `3bdf1f60844fd47cf31467c3d7d2234f1e4ef7b651dc58bf347ac397429f90bb`. Full ID-level audit and selection rules are in `docs/step_c_eval_leakage_audit_20261015.json`.

**Leakage gate still required before any GPU spending:** no complete competition `tasks.jsonl` or raw 200-trajectory corpus was found in this checkout. Task/repository IDs are clean, but full content independence has not yet been proven. After authorization, recover pinned provenance and compare canonical issue URLs, PR/base/fix commits, repository aliases/forks, normalized issue text, patch text and distinctive code snippets. Check near duplicates as well as exact hashes; common boilerplate is insufficient evidence of leakage. Exclude the entire C training/validation union, and do not evaluate on C's ten validation tasks. If any proposed task is contaminated or its provenance is unresolved, stop before GPU launch and revise/re-freeze the manifest transparently.

Unknown Gemma pretraining exposure cannot be ruled out with these files. The claim supported by this design is independence from our post-training/validation data, subject to the content audit—not that Gemma has never encountered these repositories. Renaming tasks does not solve that limitation. No gold patches, test patches or retrospective analysis are provided to the agent as additional hints; use the existing verifier and sandbox boundaries unchanged.

## Execution sequence, only after authorization

1. **Freeze inputs and run CPU checks.** Preserve the current B notebook, generated files and all A/B/C outputs. Record Git/source hashes, task and snapshot IDs/base commits, prompts, both submissions, wheelhouse/environment versions, overlays and task dependency pins. Obtain current account quota and confirm no queued/running `check-l4` version. Keep `KAGGLE_OWNER=atulbaliyan` in every Kaggle shell; use the existing login without reading auth files.
2. **Validate the environment and submissions without a model.** Reuse the existing gold/empty sweep evidence only if all relevant input versions/hashes match. Otherwise, request/include a free Kaggle CPU preflight that verifies gold passes and empty fails under the frozen overlays. Compile both submissions with the official-wheel compiler workflow, adapted to verify the attached real adapter's manifest/config without serving a model. The CPU preflight and compiler work are part of the proposed authorization scope, not work already executed. Current `preflight.py --sweep` covers all tasks; it does not currently honor the pilot whitelist, so a targeted check needs a reviewed source change or use the existing full CPU sweep.
3. **Build and run the candidate arm first.** Use `v8_conv_lora`, this exact whitelist, one repeat, `--adapter-from atulbaliyan/gemma4-train-lora`, offline four-L4 metadata and a proposed session budget of 2.5 h. Before vLLM startup, assert the attached completed training output is C version 2 and hash the copied `conv_lora` file against the expected hash. `Submission.fetch_adapters()` currently chooses the first matching path; add a fail-closed check for ambiguity and checksum mismatch rather than silently selecting an old adapter. Do not edit the generated notebook.
4. **Serving gate inside the candidate arm.** Require the expected adapter-copy source and declared `conv_lora` manifest, successful vLLM startup, correct adapter routing for the root agent, no missing/unexpected tensor or adapter-load errors, and all five first-chunk health checks: results present, not all errors, tool calls present, traces saved, verifier tests ran. A healthy first chunk need not solve a task. Keep these eight tasks as part of the fixed 24, rather than discarding them after inspecting outcomes. If any required check fails, stop and diagnose; do not push a retry or launch the baseline automatically.
5. **Download and validate the candidate outputs before reusing `check-l4`.** Require 24 unique expected result IDs, matching traces and patches/test logs where applicable, recorded arm configuration and unchanged adapter hash. A notebook COMPLETE status alone is insufficient: `main.py` and `runner.py` can package partial results after a health abort, budget stop or caught chunk exception. Verify result completeness explicitly.
6. **Run the fresh baseline arm.** Once the candidate serving/execution gates pass and its outputs are safely retained, build `v7_noedit` against the exact same whitelist and conditions, with no adapter enabled. Use the same model-server LoRA capability settings; the declared root-agent adapter is the sole experimental difference. One new `check-l4` version at a time. Complete and retain its outputs before comparison.
7. **Analyze locally with lightweight scripts.** Pair by task ID, preserve raw reports/traces, produce a comparison and per-task failure table, record actual quota deltas and update the ledger. No model imports or execution locally. Pause after reporting; no automatic follow-up, checkpoint sweep, retry, full Step D or leaderboard submission.

The existing builder can express the intended arms using the new manifest as `--valid-file`, `--valid-only --max-tasks 24 --repeats 1 --budget-h 2.5`; only the candidate adds `--adapter-from`. These are proposed arguments, not commands executed in this review. The runner uses seed-0 repository interleaving and then filters by whitelist membership; the JSON list order itself is not execution order. Save `task_ids_order.json` and assert that both arms have identical order. Snapshot generated notebooks in separate preparation directories before each authorized push; the builder's default output path is shared.

## Fixed serving and evaluation conditions

Use the existing vLLM 0.19.1 server on `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2`, TP=4, BF16, maximum context 32,768, memory utilization 0.80, Gemma-4 tool and reasoning parsers, automatic tool choice, LoRA enabled, max eight adapters and rank limit 128. Keep the existing thinking/template behavior identical in both arms and verify the compiler accepts the frozen sampling YAML.

Pin the actual swegemma/adk-submission versions restored for both arms, expected from prior checks to be 0.2.11/0.2.13. Keep the same Python 3.13 task sandboxes, scorer site, test-only extras, per-task pins, graphs, snapshots and verifier. Preserve temperature 0.2, top-p 0.95, top-k 40 and max output 8192. Keep task limits: 4.5 minutes, 40 tool calls, 100 turns and the existing 180-second timeout field; confirm its current harness semantics. Concurrency remains one, compaction interval 15 / threshold 32,768.

`Settings.seed=0` controls task order, not necessarily LLM sampling. If a supported inference seed is available, record and use the same seed in both arms without changing temperature; otherwise explicitly report uncontrolled generation randomness. One attempt per task is not an estimate of pass@k or a noise-free treatment effect. Use a fresh sandbox for every task and reset server state between arms.

## Measurements and analysis

| Area | Required outputs |
|---|---|
| Primary outcome | Resolved count/24, success percentage, per-repository rates, candidate-minus-baseline count and percentage-point difference; both solve / candidate only / baseline only / neither table. |
| Statistical uncertainty | Exact paired McNemar test via `analysis/compare_runs.py`, paired confidence interval for the rate difference, per-arm intervals, and discordant counts. With 24 tasks, one result is 4.17 percentage points; three repositories do not support a strong repository-level generalization claim. |
| Tool behavior | Calls/task and by role/tool, malformed arguments and tool errors by current schema, repeated-call share and loops, successful writes/replacements, test attempts and tests after final confirmed edit, explicit submit versus fallback/empty patch. |
| Resource/runtime | Task duration median/p90/total, agent and verifier time separately where recorded, startup/restoration time, prompt/output tokens where available, GPU utilization/memory if captured, arm session time and actual quota delta. |
| Reliability | Missing results/traces, adapter/vLLM errors, harness exceptions, timeouts, turn/tool-budget exhaustion, context overflow, sandbox/import/test-collection failures, no tests run and invalid patches. |
| Failure mode | Per-task localization failure, malformed/failed edit, edit reverted or empty patch, wrong patch untested, wrong patch despite local tests, budget/context failure, or environment failure; attach evidence references and confidence. |

Repair or supplement the analyzers before interpreting behavioral results: `format_drift.py` hard-codes swegemma 0.2.7 signatures, while the current harness is newer. `trace_analyzer.py` and `compare_runs.py` recognize edits through `write_file`/`edit_file` but miss many v7 Python replacements through `run_command`; they can therefore mislabel verification and failure stages. Match observations to call IDs, account for the current nested observation structure and subagent traces, and validate parsing on preserved B traces plus small synthetic fixtures. Recognize command-based writes using explicit command patterns and available execution/diff evidence; label uncertain edits unknown rather than claiming no edit. Distinguish a tool's successful execution from passing tests. These analyzer changes are proposed; no code was changed for this review.

Do not silently drop failed tasks or count missing results as success. Primary denominators remain the 24 predeclared tasks in each complete arm. Report errors as unsuccessful outcomes and separately by cause. For incomplete runs, label the experiment incomplete and show coverage; do not claim a full paired PASS from `compare_runs.py`'s default common-ID intersection. Environment failures discovered after launch must be reported symmetrically, with any sensitivity analysis clearly secondary. Gold-file localization heuristics are diagnostic, not proof of a unique correct fix.

## GPU quota estimate and limits

Use the observed **approximately two quota-hours per session hour**, not four times wall time merely because there are four L4s. Recheck actual quota immediately before every authorized push; 11.87 h is the last measured balance, not a current guarantee.

Planning model per arm: 0.25–0.35 h for restoration/startup/packaging plus 24 × 3–5.5 min/task = **1.45–2.55 session h**, or **2.9–5.1 quota h**. The previous eight-task B run took 0.532 session h including startup; the conservative range allows for harder tasks and verifier overhead.

| Option | Estimated quota h | Assessment |
|---|---:|---|
| Separate eight-task C smoke | ~1–2 | Proves serving, but alone cannot measure adapter benefit; integrate into candidate arm instead. |
| Recommended 24-task paired pilot, one attempt/arm | **5.8–10.2** | Best balance of a fresh control, repository coverage and cost. |
| 60 new tasks, both arms, one attempt | ~13–23.4 | Does not fit conservative planning for 11.87 h. |
| All 114 tasks, both arms, one attempt | ~23.8–43.2 | Defer until quota reset and a positive pilot. |

Proposed maximum allocation: two sessions capped at **2.6 h each**, approximately **10.4 quota h total**, leaving **1.47 quota h**. `--budget-h 2.5` is only a scheduling guard: current runner estimates 5.5 min/task before each eight-task chunk and does not enforce a hard notebook deadline. Before execution, add a source-level watchdog that includes startup, bounds subprocess waits, stops model work and saves outputs with a cleanup allowance before 2.6 h. Verify it with a lightweight fake-process check; do not claim a guaranteed billing cap from the existing setting. Kaggle/accounting lag may also affect the exact charge.

If measured startup or first-chunk time makes completing all 24 unlikely within the cap, stop before spending the next arm's quota and report the situation. Do not opportunistically change the sample, task budgets or repeats based on observed solves. A later smaller fixed design would require renewed authorization. The 1.47 h reserve is for uncertainty and cleanup, not an authorized extra run.

Sessions exceeding two hours require explicit Shrey agreement under guide §7.2, in addition to Atul's approval of this evaluation scope. Prior Shrey approval covered Step C training and does not authorize these new jobs. Record expected usage in a new evaluation ledger row only when execution is authorized, then record actual quota deltas separately for each arm.

## Interpretation, artifacts, and approval scope

Hypothesis: the full SFT adapter improves workflow reliability and resolves more independent tasks than v7 under the same budgets. Evidence against it includes worse resolve counts, more empty patches, tool-format drift or runtime failures despite lower training loss.

Pre-register an exploratory continuation rule: net gain of at least three tasks, no increase in empty-patch count and no systemic serving failures makes a larger confirmation worth considering. Report the McNemar result and uncertainty regardless; this rule is a follow-up prioritization rule, not statistical proof. Net changes within ±2 are inconclusive; net loss of at least three or systemic failures argues for diagnosis before further GPU spend. Do not select an earlier checkpoint by comparing outcomes on this pilot. The guide's separate leaderboard gate of at least five extra solves on the same 114 tasks is not satisfied or replaced by this pilot.

Retain separate future run folders for C-candidate and baseline, their exact task manifests/order, source/settings/input hashes, adapter hash, run logs, notebook logs, summary, task results, patches, test output and full ATIF archives. Produce `paired_summary.json` and a per-task failure report with baseline/candidate result, duration, error type, empty patch, tool metrics and evidence paths. Keep adapters and raw large archives outside Git; commit only approved small reports and relevant documentation.

**Requested future authorization scope:** complete the pending provenance/content audit and source-only safeguards, one free Kaggle CPU compiler/preflight job if needed, then at most two sequential GPU versions (candidate and fresh baseline) for these 24 tasks, up to the agreed 10.4-quota-hour allocation. Shrey must approve the over-two-hour sessions. Stop on serving failure, infrastructure failure, insufficient quota or incomplete provenance; no automatic retry or further evaluation. Until that authorization arrives, this proposal remains unexecuted.
