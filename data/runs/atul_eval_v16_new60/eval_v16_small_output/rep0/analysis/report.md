# Atul v16 new-60 Version 5 recovery and paired audit — October 23, 2026

**COMPLETE; 60/60 task outcomes recovered; 27/60 resolved (45%), 33 unresolved, zero missing outcomes.** v14 resolved 30/60 (50%). The paired net change is −3 tasks (−5 percentage points); exact two-sided McNemar p=0.375. This run does not support promoting v16 over v14.

## Identity, recovery and coverage

Notebook: https://www.kaggle.com/code/atulbaliyan/check-l4, **Version 5**. Run `atul-v16-new60-20261023T115526Z`; approved base `701ed9f6d2256494dffba5133ea6540f72328706`, branch `codex/atul-v16-new60-preflight`. Exactly one push began **11:57:36.557263 UTC /17:27:36 IST** and was accepted at 11:57:40.622000 UTC. Terminal COMPLETE was independently observed at 16:39:22 UTC and reconfirmed at 16:41:54 UTC. No second run, retry, training or leaderboard submission occurred.

All original output pages (one page, 38 files) and the 426,098-byte platform log were retained locally, along with original archive `9838e8e134f8a2180d030d026fdb862541108888f6b80aa385c0023550dff483` (SHA-256). Archive extraction checked all 250 members before writing: only relative regular files/directories, no traversal, links or special files; 6,910,660 total extracted bytes. Nothing downloaded was executed.

Remote Version 5 cells match the immutable submitted notebook; NvidiaL4, GPU enabled, Internet disabled. Both recovered submission ZIPs contain six files byte-identical to approved v16 source. `eval_cfg.json` identifies v16_small_output, session budget5.0; log declares adapters `[]`; sampling8192 output/4096 thinking and original v14 task budgets are unchanged. Every trace system prompt contains the approved output-limiting section (CPU checked). Notebook SHA-256 `c0eb741007c9ba214b55d7326826e652a6f5994d43149801a1ea339b80c8fb67`; metadata `15a58a0063c0fe4dbeb767d10e35da0045630bfbd98bae60d7d7efa247548cdd`.

The archive evaluator summary equals the separately downloaded evaluator summary:27/60. The local top-level summary is a consolidated control/result record; original evaluator summaries remain separately preserved under outputs/ and recovered/. All60 distinct instance_id values exactly match v14 and the approved manifest. There are60 parseable traces and58 verifier logs; **fastapi_14266 and fastapi_14349 have no verifier log** after context termination (test_exit_code−1). Their error outcomes are present; missing verifier logs are not missing task outcomes. No stale v14 notebook file listing was used as v16 evidence.

## Paired outcomes

| Outcome | Count |
|---|---:|
| Both solved |26|
| v16 only |1|
| v14 only |4|
| Neither solved |29|

v16-only: `fastapi_9425`.

v14-only: `fastapi_11194`, `fastapi_14349`, `fastapi_14430`, `rich_4079`.

All60 paired IDs and per-task measurements are supplied in comparison.json and v14_per_task.jsonl/v16_per_task.jsonl. Exact McNemar uses discordant counts1 and4: 2×(C(5,0)+C(5,1))/2^5=0.375. It does not establish improvement or statistically significant degradation. This repeated diagnostic set is not independent leaderboard/generalization evidence.

## Comparable metrics

| Metric | v14 | v16 |
|---|---:|---:|
| Resolved /60 | 30 | 27 |
| Failed outcomes | 30 | 33 |
| BudgetExceeded events | 10 | 11 |
| Context overflow tasks | 8 | 5 |
| Original analyzer `empty_patch_pct` | 8.3 | 5.0 |
| Original analyzer `repeated_identical_calls_share_pct` | 6.5 | 7.7 |
| Original analyzer `explicit_submit_pct` | 88.3 | 90.0 |
| Original analyzer `tool_calls_median` | 24.0 | 25.0 |
| Command-position cat proposals, total | 99 | 59 |
| Serialized tool-response chars / task, mean | 24104.75 | 23345.97 |
| stdout+stderr chars / task, mean | 20259.82 | 18958.15 |

cat proposal mean/task:1.65→0.9833; median0→0. Total raw response characters1,446,285→1,400,758 (−3.15%); median per task22,428.5→24,562.5 (increased). stdout+stderr total1,215,589→1,137,489 (−6.42%); median17,555.5→17,145. Fewer cat proposals (−40.4%) did not produce a resolved-rate improvement. Original analyzer repeat rate rose6.5→7.7%, median proposals24→25. Keep original analyzer files unchanged under original_*; their stage/localization/test heuristics do not prove understanding or successful retained edits.

Definitions applied identically to both arms: proposals are recorded tool_calls, not assertions of successful execution. cat counts one run_command proposal containing a top-level shell command-position cat, recognized after shell operators/optional command prefixes; heredoc bodies and quoted prose are excluded, nested shell/Python command construction is outside this conservative definition. Raw response characters sum serialized observation content once per associated tool response; stdout/stderr characters decode these fields (including nested details), excluding JSON envelope and other fields. These are Unicode character counts, not bytes or model tokens. ID association is preferred, with explicitly ordered same-tool fallback for ID-less observations; both arms have zero missing or unmatched observations. Values may differ from earlier studies with different counting definitions.

## Errors, context and budgets

v16 context error IDs: `fastapi_14609`, `fastapi_14266`, `fastapi_14246`, `fastapi_14349`, `fastapi_13786`. Each is a recorded ContextWindowExceeded error: ≥24,577 input tokens +8,192 reserved output exceeds32,768. These five are the only uncaught task-result errors; no additional non-context task-result exceptions were found. v14 had eight such task errors. Three persist in both arms; five v14 overflow tasks avoid overflow in v16 but remain unresolved. New overflows are fastapi_14246 and fastapi_14349; the latter is one paired regression. Lower overall overflow count is verified, but cannot be credited with additional solved tasks here.

Actual tool BudgetExceeded responses: **10 events/8 tasks→11 events/6 tasks**. v16 IDs: `fastapi_14455`, `rich_3469`, `rich_3064`, `fastapi_13207`, `fastapi_14262`, `rich_3296`. Two of these tasks (fastapi_13207, rich_3296) still resolve. The original analyzer's terminal tasks_budget_exhausted_pct is0 in both arms; this does not mean zero denied tool calls. No recorded TimeoutExceeded tool responses, WATCHDOG firing, chunk timeout, or explicit malformed/unparseable/truncated-tool-call log records were found. All recorded tool-call arguments parse; this does not prove unavailable intermediate model generations were never truncated. Bounded/truncated stdout is different from an unparseable model tool call.

Five v16 verifier logs contain collection errors (v14 four): `fastapi_14978`, `fastapi_14609`, `fastapi_14262`, `fastapi_15030`, `fastapi_14306`. Shared sitecustomize wrapt warnings occur in all58 available v16 verifier logs, including resolved tasks (and all60 v14 logs); the warning alone is not a causal explanation. Final resolved/test_exit_code and verifier logs take precedence over agent claims or shell exit0 behind a pipeline.

## Final failed-patch taxonomy

Classification uses retained +++ b/ patch paths and actual unresolved outcomes, separately from mutation intent. Repository prefixes: fastapi/, rich/, src/requests/, docs_src/, docs/. Scratch/test prefixes include tests/, /tmp/, repro*, verify*, test_*, output*. These labels do not establish semantic correctness or attribute the verifier failure to a stray file.

| Unresolved final patch |v14|v16|
|---|---:|---:|
| Empty |5|3|
| Scratch-only |5|1|
| Source plus stray scratch |5|7|
| Source wrong/incomplete (unresolved source patch) |15|22|

The final category includes unresolved source patches whose failure may involve verifier/environment issues; it is not a manually proven incorrect-fix count. AST-detected mutation intent is reported per task but never substituted for final patch content.

The paired regression fastapi_14430 ends with only repro.py despite mutation proposals; fastapi_14349 ends empty after context overflow; fastapi_11194 retains source plus reproduce_issue.py and verifier exit1; rich_4079 retains rich/markdown.py with verifier exit1. The sole gain fastapi_9425 retains fastapi/dependencies/utils.py and verifier exit0; v14 retained that source plus reproduce_issue.py with exit1. These observations do not isolate the causal mechanism of the prompt change.

## Runtime, chunks and quota

All eight chunks completed, last chunk7 (offset56, final four tasks). Minutes: **37.6,33.1,35.0,30.3,40.1,28.7,38.9,15.8**. Evaluation began12:07:33UTC; final task chunk finished16:27:02UTC; RUN_DONE16:27:09UTC. vLLM startup320s. No watchdog/timeout fired.

Platform-log maximum elapsed time **4.493467h** (16,176.48s) is a notebook runtime proxy; run.log rounds this to4.5h. v14's comparable proxy4.020013h. Do not use delayed terminal observation time as actual runtime.

Pre-launch quota:16.962316111h used,0reserved,13.037683889available of30h. Two terminal observations:25.949727778used,0reserved,**4.050272222available**. Account-used delta **8.987411667 quota hours**, about2.0001× notebook-log runtime. This is observed account accounting, not a per-job billing invoice; exact allocation boundaries are unavailable. No reset or reversal was observed during this run.

Saved monitoring cadence was15min; actual observation gaps ranged **1.43–65.33min**, with delayed heartbeat execution. Full actual timestamps are in comparison.json and summary.json. Monitoring was not continuously observed at15-minute intervals. Terminal completion occurred before5.5h and available quota did not trigger the configured threat threshold.

## Validation and recommendation

CPU checks: all60IDs/outcomes/traces parse and match; summaries agree; six-file submission source identity, remote Version5 identity/settings/no-adapter evidence and safe archive extraction pass; exact paired arithmetic cross-checks. All53 protected v15b files and backup SHA-256 remain unchanged; all238 baseline files audited remain byte-identical. Original branches/stashes and v15b evidence are preserved.

Keep v14 as the current base. v16 reduced observed overflow and cat use, but resolved fewer tasks and spent more runtime/quota. The next useful work is CPU-only examination of the four regressions, remaining context failures and scratch-only/stray-file outcomes. No further GPU run is authorized or scheduled. Original archives, traces, notebook, platform log, recovery manifests and raw outputs remain local; publication contains only small summaries, the original run.log, analysis and the quota-ledger update.
