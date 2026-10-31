# Gemma 4 Developer Agent — HQ

## Shrey/Claude — RS1 training set built: 350 thinking-ON episodes, 261 core, October 31

Rule from Shrey: nothing heavy on the 8 GB MacBook; data builds run as Kaggle kernels (saved in memory). CPU kernel `gemma4-rftset` (`kaggle/rftset/`) rendered R2's 350 resolved runs with thinking as `reasoning_content`, matching vLLM's logged prompt sizes within a constant 29 tokens (system/tool prefix). core 261 (no stray files, no drift, main agent only). Median 13.1k tokens; 84% exceed 8k (925 windows). Core is 82% single-bug; only 22 core runs test after their last edit. Files live in the kernel output (attach `shreysharma07/gemma4-rftset` as a kernel source to train). Report `results/runs/rs1/`.

## Shrey/Claude — R2 DONE: 350/601 factory rollouts resolved (58%), October 31

Shard 1 (`atulbaliyan/gemma4-factory-rollout-s1`, 5.7 h on Atul's quota): 174/300, 55.3 tasks/h, output shared with Shrey and downloaded to `data/runs/r2_s1/`. Combined, disjoint, all 601 tasks once: **350 resolved**; singles 75%, pairs 37%, triples 14%; v0 74%, v1 46%. Every R2 prediction held. Failure pressure: call cap (124 tasks used ≥27 of 28 calls) > overflow (44) > time cap (28). Behaviour to filter in training data: stray files in ~32% of patches, tests after last edit in ~5%. Next: build the training set from the 350 resolved trajectories.

## Shrey/Claude — R2 shard 0 DONE: 176/301 (58%), every prediction held, October 30

`gemma4-factory-rollout-s0` v1, 5.8 h (~11.6 quota-h), 54.3 tasks/h at concurrency 6, all 301 done. Singles 75%, pairs 33%, triples 23%; v0 tasks 75%, v1 tasks 44%. 12-min cap hits 4.3%, call-cap hits 68, overflow 8.3%, rewrite fallbacks 7. 176 resolved trajectories ≥ the 150 decision bar → training set next (drop stray-file patches: 32% of tasks). Shard 1 waits on Atul (datasets must be shared with him). `results/runs/r2_s0/`; traces in `data/runs/r2_s0/`.

## Shrey/Claude — R2 shard 0 pushed (Shrey), shard 1 prepared for Atul, October 29

Pushed `shreysharma07/gemma4-factory-rollout-s0` v1 (301 tasks, budget 7 h, ≤14.5 quota-h). Shard 1 (300 tasks) is built as `atulbaliyan/gemma4-factory-rollout-s1` in `kaggle/eval_l4/factory_rollout_kernel_s1/` (built with `KAGGLE_OWNER=atulbaliyan`; inputs stay on shreysharma07). Atul pushes it from his own login: `kaggle kernels push -p kaggle/eval_l4/factory_rollout_kernel_s1`. Needs datasets `gemma4-factory-v0` and `gemma4-factory-v1` shared with Atul first. Builder fix: factory datasets now use the source owner, not the pushing account.

## Shrey/Claude — R2 factory rollouts built (601 tasks, 2 shards), October 29

New `evalkit/factory_rollout.py` + builder flag `--factory-rollout --shard k --n-shards 2 --max-time-min 12`. Loads every verified task from both factory datasets (symlinked snapshots), seed-0 shuffle, runs tasks[shard::2]: FS1 issue rewrite, then v14 at concurrency 6 with the per-task cap raised 8 → 12 min (`Settings.max_time_minutes`, applied in `ChunkedEvaluation`; default None so scorer-faithful runs are unchanged). Shards: s0 301 tasks, s1 300. Budget 7 h each (≤14.5 quota-h per shard, ~29 for both, about one week's quota), so the shards run one after the other. Pre-registered as R2 in `docs/experiments.md`.

## Shrey/Claude — factory v1 DONE: 332 verified tasks, 57% multi-site, October 29

Kernel `gemma4-task-factory-v1` (6.5 h CPU): built 425, **verified 332** (gold 78%, empty 0%): 143 singles, 132 pairs, 57 triples, 9 libraries with ≥10 tasks. Mutant runs ~2–7 s vs v0's ~54 s. Lost: marshmallow (simplejson not in scorer site), 5 libraries whose test deps are missing on the build host, and more-itertools (coverage tracing broke 7 baseline tests → 0 tasks in 213 min). Fixed after the run: plain baseline for outcomes + test-level targets (toy-checked, not re-run on Kaggle). Private dataset `gemma4-factory-v1` uploaded with ids renamed `f1_*` (v0 uses `fac_*`, so the two sets combine without collisions): 601 verified tasks in total. Details: F1 row in `docs/experiments.md`, `results/runs/factory_v1/`.

## Shrey/Claude — factory v1 pushed (harder, faster, 17 libraries), October 28

Pushed `shreysharma07/gemma4-task-factory-v1` v1 (CPU, Internet). Pre-registered as F1. Speed: baseline run records per-test line coverage; each mutant runs only the test files that execute its lines, on 4 parallel repo copies. Harder: 20 hard-operator-first singles + up to 30 multi-site tasks (pairs and triples, each fix needed) per library. Libraries: 14 new + the v0 three; building stops early to keep verification inside 12 h. Checked end-to-end on a synthetic toy library on the Mac (3 s, 6 singles + 6 multis). Output will become dataset `gemma4-factory-v1`.

## Shrey/Claude — FS1 factory rollout smoke DONE: 45/60, 2.65× at concurrency 6, October 28

Kernel `gemma4-factory-smoke` v1 COMPLETE (1.2 h, ~2.5 quota-h). Gemma rewrote all 60 issues (0 fallbacks, 1.3 min). v14 solved **45/60 (75%)**: singles 41/49, **pairs 4/11**; boltons 19/20, more-itertools 15/20, sqlparse 11/20; cmp_swap 3/10, const_shift 4/10. 11 of 14 failures are wrong fixes never tested, the same failure as on new-60. **Concurrency 6 gives 2.65× tasks/h** (67 vs 25); 4 runs at c=6 hit the 8-min cap with only 19–27 calls (contention). Decision (pre-registered >70% rule): factory v1 must make harder tasks (multi-site bugs, harder ops). Rollouts at c=6 with a longer time cap. Results: `results/runs/fs1/`. (First output download was silently incomplete; re-download with `--force` got everything.)

## Shrey/Claude — FS1 factory rollout smoke pushed, October 28

Pushed `shreysharma07/gemma4-factory-smoke` v1 (GPU, offline, `--budget-h 2.75`, expected ~4–5 quota-h, hard max ~6). 60 factory-v0 tasks (20 per library, 11 two-bug, seed 0). Gemma (thinking off) first rewrites the template issues into user-style reports with a reproducer taken from the hidden tests; rewrites that quote a gold added line or name a test fall back to the template. Then v14 unchanged runs 6 tasks at concurrency 1 (baseline) and 54 at concurrency 6. Pre-registered as FS1 in `docs/experiments.md` (prediction 45–75% solved, ≥3× tasks/h at c=6). New code: `evalkit/factory_smoke.py`; `Settings.concurrency` (default 1, so scorer-faithful runs are unchanged); builder flag `--factory-smoke`. Graph tools return nothing on factory repos (no graphs); v14 rarely calls them.

## Shrey/Claude — factory v0 VERIFIED: 269 tasks, October 28

Kernel `gemma4-task-factory-verify` v3 (46 min, CPU): **gold 269/278, empty 0/278**. Verified set: more-itertools 49 (15 two-bug), sqlparse 105 (13), boltons 115 (14); hidden tests per task 1–10 (101 with one). Operators are spread evenly: const_shift 56, cmp_swap 47, drop_not 41, arg_swap 39, arith_swap 36, drop_guard 33, boolop_swap 32, drop_update 22, exc_swap 5. The 9 rejects are F2P names colliding with xfailed sqlparse tests, which the v1 build check prevents. Empty 0/278 means no library copy installed in the scorer site shadows the workspace copy (more-itertools/sqlparse are in the Kaggle image too). Private dataset `gemma4-factory-v0` now has `tasks.jsonl` (verified) + snapshots, ready for on-policy rollouts. Issues are template text (v0); a model rewrite is still to do. Summary: `results/runs/factory_v0/`.

## Shrey/Claude — R1 done (no cleanup rule); v17 local 27/60; task factory v0 built 278 tasks, verification re-run, October 28

**R1** (`gemma4-r1-regrade`): stripping stray files rescues 0/8 unresolved patches and breaks 0/17 resolved ones; originals re-grade identically 40/40. 15/23 unresolved stray-file patches had no source edit. Decision: no cleanup rule; target "no source edit" (10/30 v14 failures). **v17 new-60:** 27/60 vs v14 30/60 (p = 0.51), overflows 18 vs 8, so the base stays v14. **Factory v0** (`gemma4-task-factory`, CPU): more-itertools 49, sqlparse 114, boltons 115 = **278 tasks** (single + pair mutations, hidden tests removed from the snapshot). The build took 681 min, so the kernel hit the 12 h cap during verification. Gold passed 216/278; all 62 failures are sqlparse tasks broken by the verifier's prefix matching of PASS_TO_PASS names against xfailed tests. Fix: PASS_TO_PASS empty, plus rejection of xfail name collisions. The empty-patch check never ran. Re-verifying from the uploaded private dataset `gemma4-factory-v0` with kernel `gemma4-task-factory-verify` (no rebuild). **v1 to do:** ~54 s per mutant run is too slow; run only the test files that cover the mutated function, use 4 workers, and stop at ~40 tasks per library.

## Shrey/Claude — v17 new-60 pushed; v14 failure split; fine-tuning strategy v2, October 27

Pushed `shreysharma07/gemma4-eval-v17-new60` v1 (v17_superagent, new-60, --budget-h 5, ~9 quota-h, user-approved). A12v kernel is still RUNNING (the handoff said COMPLETE). **v14 failure split:** tractable tasks (1 file, ≤3 gold funcs) 26/41, large 4/19. Of the 15 tractable failures, 12 hit the 28-call cap (≥27 calls; failure median 28 vs 22 for resolved; time not binding, 249 s vs 480 s cap), 4 are empty, and 9 carry stray repro files (stray-file tasks resolve 1/11). Verification is nearly absent (tests after last edit 3.3%). Plan: `docs/strategy_20261027.md` (calls 28→40 probe; stray-file CPU re-grade; SWE-smith-lite task factory on installed libraries; concurrency-8 on-policy rollouts; verifier adapter + low-dose thinking-ON RFT with prefix-cached truncated backprop past 8k).

## Atul — v14 new-60 v3 COMPLETE;30/60 vs v7 23/60, October 19 UTC

Exactly one authorized unchanged evaluation completed with **rc0,60/60 tasks,30resolved**. Paired v7: **22both/8v14-only/1v7-only/29neither**, net+7 (+11.67pp), exact McNemar p0.0390625. Source/version3/noLoRA/provenance and all38outputs verified. Empty patches25→8.3%, repeats28.8→6.5%, median calls29→24; overflow3.3→13.3% (8actual context errors). Budget metric0% coexists with10denied tool events across8tasks;4shared wrapt collection failures remain in60denominator. Changes bundle thinking/prompt/budgets; no thinking-only or hidden-LB causal claim.

**4.020013 notebook-log h /8.040502 account quota h**, remaining **13.037684h**. All8chunks complete, no watchdog/timeout; exact backend boundaries unavailable. Saved15min monitor, actual gaps4.949–23.540min; monitor PAUSED after review. [Report, paired IDs, full metrics, warnings and quota ledger](atul_v14_new60_report_20261019.md). Only permitted small results/docs are committed. **STOP: no retry, further evaluation/training or leaderboard work.** Prior S3/S1/gate findings and recovery/stashes preserved.

## Atul — Step 2 COMPLETE; both adapters FAIL, October 18

One unchanged `atulbaliyan/gemma4-gate` version 1 completed with **rc 0, serving PASS and 360/360 expected responses**. Both actual copied adapter hashes match the approved training finals; remote source, all 14 modules and frozen prompt/tools/prefixes verify. Base **74/120 (61.67%)**, S1 **63/120 (52.50%)**, S3 **66/120 (55.00%)**; edits at file_in_view **24/32,19/32,21/32**. Both miss both comparative thresholds, despite zero batching/request errors. No missing/extra/invalid report rows; no-call counts 4/1/5. **S1 FAIL; S3 FAIL.**

S3 improves submission after success 24/24 but also submits after failures 28/32, with failure recovery 0/32 and early edits 10/32 across 3 prefixes. Base/S1 read early 32/32 and never edit there. S1 has more single calls but fewer correct edit/recovery decisions than base. Feedback audit finds all 6 passed-test prefixes have exit code 0/explicit passed counts; failure prefixes include environment/import errors. Decisions are classified, not executed source edits; generated arguments are unavailable. Fixed 30 prefixes/23 tasks with 4 samples each are not independent task-resolution outcomes or leaderboard evidence. Detailed per-type/distribution/paired-prefix tables and a measured figure are in the [one consolidated report](atul_step1a_1b_step2_gate_report_20261018.md); detailed S3/S1/E1 findings retained.

Actual gate **0.418923 notebook-log h/0.838354 quota h**; cumulative **8.921814 h**, remaining **21.078186 h**. vLLM startup 409 s; exact backend boundaries unverified. Ten-minute persisted monitor with delayed terminal observation is disclosed; paused after review. Connection-broken output downloads were recovered read-only using bounded ranges; originals preserved, no experiment retry/source fix. All 37 remote files retained, large/raw artifacts uncommitted. **Neither adapter advances: handoff says stop task evaluation/retraining when both fail.** Recommend separately approved CPU feedback/termination and prefix audit before deciding new work; no scale-up, paired eval or leaderboard launch. Earlier submission entries below are history.

## Atul — Step 2 gate v1 submitted once, October 18

Exactly one unchanged `atulbaliyan/gemma4-gate` is now authorized. Both training version1 COMPLETE and full local validators/hash identities rechecked PASS; all required Kaggle inputs accessible. Fresh quota **8.083459 used / 21.916541 available**, reserved0, no running/queued project notebook. Expected **~0.6 sessionh/~1.2quotah**, cumulative **9.283459h**, below approval boundaries. Exact builder produces40 compiled cells/217,310B source; all14 modules unchanged. Frozen renderer omits optional analyzer schema (no prefix history calls it); common10.5h config default is not an enforced gate cap or authorization extension. Those limitations are recorded in the [one report §6.1](atul_step1a_1b_step2_gate_report_20261018.md). Ledger updated before push. No source change, retry or later GPU stage approved. Push accepted version1, initial RUNNING; source-cell text and metadata verified remotely. Monitor `monitor-approved-behaviour-gate-v1` ACTIVE; latest saved cadence 10 minutes (requested10min; recheck for client normalization). Results pending; do not infer model failure/success from training loss.

## Atul — S1 Own v1 COMPLETE / training-export PASS, October 18

The sole approved S1 run completed unchanged: **rc 0, 33/33 steps, no budget stop**, validation **0.2395→0.1362** (43.13%) over 20/21 windows/all 3 held-out tasks. Exact downloaded NPZ/source/version1 identity verifies. All step9/18/27/final/serving adapters pass 410-module/820-BF16-rank16 integrity; final/serving SHA-256 `c41939e731011b0bc6fb5ab222c6e84f6f4a1512301201fb14bbe230d6c21040`. All-rank throughput reconstruction agrees 33/33; median **282.7 tokens/s**, weighted 281.568; peak **18.88 GiB**. Training first 0.3469/final 0.4562 varies by window; do not claim monotonic training convergence. Preclip norms up to 40.5 and non-fatal barrier/escape warnings are documented; no OOM/traceback.

Actual **2.104073 notebook-log h / 4.208630 quota h**, cumulative S3+S1 **8.083459 h**, remaining **21.916541 h**, no reservation. Model load 512 s vs S3 1,879s; cause of faster startup unverified. API last_run_time is stale; exact session boundaries unknown. Monitoring effective 30 min and execution delays are disclosed. A transient status 401 recovered through existing login; stalled local output retrieval resumed with bounded downloads, preserving originals and never retrying training. All outputs retained locally; permitted small evidence, measured curve and comparisons are in the [one consolidated report](atul_step1a_1b_step2_gate_report_20261018.md). Monitor paused after review. E1 remains 147/147 episodes/182/182 edits PASS and S3 findings intact. **Both training prerequisites PASS; gate NOT RUN/NOT APPROVED. Stop and wait for user approval.** Different validation sets cannot rank agent quality. Older submission entries below are historical.

## Atul — one approved S1 Own run submitted, October 18

The user separately authorized exactly one S1 Own training run, unchanged handoff settings (`--budget-h 3`, lr 5e-5, 2 epochs, accumulation 2, save every 9), approved estimate **2.6–2.9 session h / 5.2–5.8 quota h**. S3 version 1 remains COMPLETE and all required S3 local checks rerun PASS. Final pre-push quota/status at **12:20:39 IST**: used **3.874829**, available **26.125171 h**, no reservation or conflicting job; S3+S1 upper forecast **9.674829 h**, below 10. S1 data hash, decoded embedded input, compiled notebook, namespace/model/inputs and read access verify; normalized source **507,049 bytes**. Exact S3-based runtime scenarios finish ~2.43–2.52 h, with final training step below the unchanged 156-min cutoff; forecast only, no guaranteed margin.

One push accepted `atulbaliyan/gemma4-train-s1-own` **version 1**; initial status RUNNING. Expected **33 steps**, validation **20/21** windows; actual losses/exports/runtime/quota pending. Monitor `monitor-approved-s1-own-v1` is ACTIVE. An initial saved 10-min interval was subsequently normalized by the client to **30 min**; the effective saved cadence was rechecked and the report corrected. The [same consolidated report](atul_step1a_1b_step2_gate_report_20261018.md) preserves S3/E1 and records S1 preflight. No retry/config change, additional training, gate, paired eval or leaderboard submission is authorized. Stop after S1 review; **Step 2 needs separate approval**.

## Atul — S3 Compact v1 COMPLETE / training-export PASS, October 18

The sole approved S3 run completed: **rc 0, 35/35 steps, no budget stop**, validation **0.6089→0.1807**. All step9/18/27/final exports and serving copy pass **410-module / 820 BF16 rank-16** integrity checks. Final/serving SHA-256 `74c041da7cc4540d617463e769173942db80a5d2c60b2ba615c4afc38c616caa`. Exact input hash, 12 downloaded source modules and remote notebook cell sources match submission. Median throughput **299.3 tok/s**, global peak **18.86 GiB**, 1,409,618 input tokens; model load 1,879 s. E1 remains **147/147 episodes, 182/182 edits PASS**. This is training/export success; **behaviour untested**. Validation covers only **4/7 held-out windows** with two measurement points; gradient spikes and non-fatal warnings are documented in the [one consolidated report](atul_step1a_1b_step2_gate_report_20261018.md), including a measured loss figure.

Actual quota **3.874829 h**, remaining **26.125171 h**. Notebook maximum elapsed log time **1.937136 h**, below approved 3 h; quota/log-hour ratio **2.000288×**. API last_run_time is inconsistent, so exact absolute session boundaries are not asserted; do not use late COMPLETE-check time as GPU runtime. Monitoring requested 10 min but effective saved cadence was 30 min; delayed terminal processing is disclosed. Terminal outputs downloaded successfully and retained, including all adapters; monitor paused after review. No training retry, S1, behaviour gate, paired eval or leaderboard submission. **S1 and gate require separate approval.** Previous RUNNING entry below is history, superseded by this terminal result.

## Atul — single approved S3 run submitted, October 18

The user approved **exactly one Step 1a S3 Compact training run**, unchanged handoff settings and `--budget-h 3`; S1 and behaviour gate are explicitly withheld pending separate approval. Fresh pre-push quota at 02:20:31 IST was **30.00 h available, 0 used/reserved**, no queued/running notebook. All four input hashes, exact notebook settings, embedded S3 data, namespace and metadata passed; normalized source 459,501 bytes. Ledger expectation entered before push: **2.1–2.4 session h / 4.2–4.8 quota h**. One push accepted `atulbaliyan/gemma4-train-s3-compact` **version 1**; initial status RUNNING. No retry or config change. Ten-minute thread monitor `monitor-approved-s3-compact-v1` ACTIVE to recover terminal outputs and verify rc/loss/410-module exports/runtime/quota. Actual measurements pending. The [single report](atul_step1a_1b_step2_gate_report_20261018.md) now reconciles executive tables/conclusions to E1 **147/147 episode, 182/182 edit PASS**. No behavioural pass is claimed.

## Atul — E1 resolved; S3 runtime approval pending, October 18 follow-up

**E1 source-edit replay PASS: 147/147 episodes, 182/182 edits.** Exact `instance_id`/repo joins to Nebius SWE-rebench revision `89cdfbab4ab1bd8f5a658bb212d1b63624f4f881` recovered every base commit. Only three benchmark columns (~1.64 MB) and 150 distinct pinned source files (~2.65 MB) were downloaded. Replay used audit-owned AST-equivalent replacement operations in a disposable, network-disabled bubblewrap namespace; no teacher code or repository setup/tests ran. All 151 episode/file initial and final Git blobs match teacher patch indexes. The empty-file edit is valid: actual source is zero bytes. No confirmed edit/feedback failures or data correction. Detailed local evidence and the small committed [summary/complete source manifest](../data/runs/atul_e1_replay_20261018/summary.json) support the [same consolidated report](atul_step1a_1b_step2_gate_report_20261018.md).

The earlier universal pretraining E1 gate was an overly broad interpretation. The handoff explicitly permits CPU E1 in parallel and requires confirmed failures removed before training. Full replay now passes anyway. Separate primary-workspace `.venv` has only Kaggle CLI 2.2.4 and dependencies; existing login verified without copying/printing/inspecting credentials. Fresh 02:08:10 IST quota: **30.00 h available, 0 used/reserved**; no queued/running project notebook. Exact S3 notebook generated and metadata verified, **459,501 bytes**, unsubmitted. Source-derived validation limitation: unchanged scheduler covers 4/7 S3 and 20/21 S1 held-out windows.

**Await explicit runtime approval before GPU push:** S3 ~2.1–2.4 session h / ~4.2–4.8 quota h; full S3/S1/gate forecast ~10.6–11.8 quota h. These exceed the preserved >2-session-h / >10-cumulative-quota-h rules. Settings remain exactly the handoff's two epochs and 3 h budgets. No training, GPU inference, gate, paired-task eval, notebook push, retry or submission occurred; no GPU quota spent. Prior preflight entry below is preserved as history, superseded on provenance and isolation availability by this follow-up.

## Atul — S3/S1/gate preflight, October 18, 2026

The reporting requirement is **one consolidated technical report**: [Step 1a, Step 1b, Step 2 and E1](atul_step1a_1b_step2_gate_report_20261018.md). Primary workspace cleanly fast-forwarded to `fc29e53`; work is on `atul/s3-s1-gate-results`. Both NPZs, gate prefixes and preserved raw-row archive match the handoff hashes. Fresh authenticated quota at 01:25:27 IST: **30.00 h available, 0 used/reserved**; three project notebooks COMPLETE, two unrelated notebook sessions return 404.

**Stopped before any GPU push.** E1's exact source bases are not established: no `base_commit` in 200 raw rows or 147 matching candidate compact episodes. The new S3-1k archive contains candidates matching all original S3 task and trajectory IDs, but matching IDs do not prove original rendered-byte identity. Static inspection finds 823 associated call/response pairs, 182 edit calls, 181 unique old literals in visible excerpts, and one empty-old/empty-file case requiring actual file bytes. No real-repository replay or teacher-command execution occurred; disposable isolation was not established. Do not call this E1 PASS or a confirmed edit failure.

S3/S1/gate are **NOT STARTED**, not failed training runs. Exact source scheduling implies 35/33 optimizer steps at the handoff's two epochs. A token/load-based planning scenario is ~10.17 quota h before validation/export overhead, so review >2-session-h and >10-cumulative-quota-h approval limits before resuming. Ledger updated; no data regeneration, model load, notebook push, GPU use, evaluation, retry or submission. Existing findings below remain historical context.

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


## Atul — authorized Step D/D0 continuation, October 16, 2026

The human explicitly authorized exactly one pass each, **D first, then D0**, overriding §7.3’s historical two-pass commands. Step D was pushed once successfully as `atulbaliyan/check-l4` **version 2** and is RUNNING; results and startup/health gates are pending. No training notebook was pushed. Latest training notebook is still Step C v2 COMPLETE; a fresh download of its serving adapter passes integrity checks and SHA-256 equals the preserved C final/serving copies (`a116b6be160fffcc98cdd2a4ab6cfaffafcfc54ac6709a8a5d95d95ebab544c6`), different from Step A.

Pre-push quota: 18.13 used / **11.87 h remaining**. D expected ~4 session h / ~7 quota h at 1.75×; observed prior runs suggest a conservative ~8 quota h at 2×. Expected usage was entered in §8 before push. Current generated notebook was backed up before rebuilding; the submitted config is archived in `data/runs/atul_step_d_20261016/submitted_config/`. Static verification confirms one pass, 114 task IDs, exact v7 scaffold plus adapter declaration, and no embedded stale adapter. All account notebooks were checked: relevant training/eval/smoke sessions COMPLETE; older unrelated notebooks have no session (404).

Five-minute thread monitor `monitor-authorized-step-d-then-d0` is ACTIVE: save quota/log snapshots, verify startup and all five health checks, download/review D before D0. D0 requires fresh quota >=4.5 h (3.5 expected plus 1.0 safety reserve), and cannot launch on failed/incomplete D. Stop without retry on critical failure. No raw logs/traces in chat, local model work, extra experiment, leaderboard submission or GitHub push. Initial persisted and bounded live log captures were empty; health/startup remain unverified, not failed. State/evidence: `data/runs/atul_step_d_20261016/summary.json`.

October 16 quota-risk intervention during D: live logs confirm all five first-chunk health checks PASS, correct training adapter, vLLM up, 114 × 1. First 16 tasks took 47.2 min (chunks 19.9/27.3); at observed 2× accounting, remaining 98 project ~9.64 quota h versus 7.54 available at snapshot. D0 blocked. Requested safe stop of current session; SDK cancellation needs a verified session ID not exposed in notebook metadata/status, and browser was signed out. No guessed-ID cancellation, deletion, retry or new push. Monitor remains active to recover terminal outputs; D is not yet complete.

October 16 00:51 UTC / 06:21 IST: D remains RUNNING; quota 28.40 used / **1.60 h remaining**, D charge ~10.27 h. Earlier manual-stop request has not resulted in a terminal state. Urgent user intervention requested again to stop current `check-l4` v2 and protect the 1.0 h reserve. Available cancellation API still lacks verified session ID; no guessed-ID action taken. D0 prohibited. This snapshot is later than the heartbeat’s trigger due to execution/approval delay; use actual checked time.

## Step D final / D0 blocked — October 16

**D COMPLETE / rc 0; 0/114 resolved, 114 empty patches.** One authorized pass, check-l4 v2; final Step C adapter checksum verified inside downloaded submission.zip. All five health checks passed; exact 114 task results/traces/logs retained, 77 verifier logs, no health abort or budget stop. Task errors: 28 context overflows, 7 timeouts, 2 turn limits. Execution PASS does not mean performance PASS. Actual 4.4916 session h / 8.9942 quota h = ~2.0024×; used 18.13 → 27.1242, 2.8758 h remains. Final quota was less pessimistic than live telemetry; prior stop alerts remain historical evidence, no agent cancellation succeeded.

**D0 never submitted:** remaining 2.88 h is below the 3.5 h estimate plus reserve (4.5 h gate). New60 D=0/60; old54 D=0/54. No fresh baseline or paired McNemar comparison is available. No automatic launch after reset. Monitor PAUSED; instructions required for further action. All output downloads exited 0, archives/ZIP verified. [Final report](step_d_d0_20261016_report.md). No extra run, training push, leaderboard submission, credential inspection or GitHub push.

## Step D retrospective diagnosis — October 16, no GPU usage

Read-only analysis found no identifiable repository-source edit in any D trace; apparent write hits were temporary reproduction fixtures or read-only diagnostic false positives. All 81 explicit submissions returned zero bytes/zero changed files. Batched proposals and the analyzer’s response association made earlier command/error counts misleading: 5,684 proposed shell calls reconcile to 4,106 harness-counted calls, with 389 budget rejections. All 28 context errors share the 24,577 input + 8,192 allowance boundary. B/D submission and six evalkit source files are byte-identical; B preserves three source patches, supporting policy failure over a generic collection defect. Upstream training/data/rendering causes remain hypotheses pending separately authorized CPU audits. Historical comparisons are descriptive. No notebook, D0, retraining, inference, result changes or Git push. [Detailed diagnosis and ranked next tests](step_d_failure_analysis_20261016.md). Analysis complete; await authorization before implementation or experiments.

## Atul Baliyan — data/post-training research recommendations, October 16

[Research report](atul_data_training_research_20261016.md) uses the completed Step D diagnosis as primary experimental evidence. Recommended order: recover exact preprocessing artifacts; verify converted edits/feedback, window state and token boundaries; measure action/token/effective-loss exposure; then evaluate verified SFT with source-edit, recovery and nonempty-patch gates. Three delegated research agents completed their audits; literature and synthesis passes were completed by the primary researcher after delegate usage-limit failures. Pipeline, windowing, masking, over-adaptation and quantization-transfer causes remain unproven. Paths ranked: minimal demonstrated pipeline correction → verified dataset redesign → conditional objective changes; defer online RL under the last measured 2.88 quota-hours.

**New lightweight CPU evidence, not a causal result:** the preserved C NPZ has maximum window length 8,191, excluding the inspected 8,192-token hard clipping conditional on preprocessing provenance. Supervised labels/window range 23–4,726; training averages window means while validation weights supervised tokens. Actual action/phase shares remain unknown without missing rendered examples/tokenizer. [Audit and reproduction](atul_data_training_cpu_audit_20261016.md). These recommendations have not been implemented or behaviorally tested. No GPU job, model load, retraining, notebook push, D/D0 retry, submission or Git push was performed; prior findings and contributor sections are preserved.


### Publication clarification — October 16, 2026

The planning notes above and in the training guide are retained as historical context. The subsequently completed Step D run resolved **0/114 tasks**, with **114 empty patches**. D0 was **not run** because the last measured remaining quota was approximately **2.88 quota-hours**, below its launch gate. The completed diagnosis and research recommendations above supersede earlier pending-evaluation status for this run; they do not authorize another evaluation or training job. This documentation branch preserves Shrey's latest upstream work and leaves the training guide unchanged.

## Literature gap survey — October 22, 2026
- Four parallel surveys (code-graph construction/embedding, code graphs in SWE agents, LLM graph reasoning, citation audit) → `research_deliverables/gaps_2026-10-22.md`, raw notes in `research_deliverables/surveys_2026-10-22/`.
- **Citation audit:** `research_deliverables/literature_review.md` + `references.bib` have invented authors on ~20 rows, 6 wrong arXiv IDs, 1 non-existent paper and several unsupported numbers — do not cite until fixed. `docs/literature_2025_2026.md` mostly verified; its Gap 4 evidence (CodeAnchor/SHERLOC) is mischaracterised.
- SchrödingerRepo (2609.27891): renaming drop vanishes on post-cutoff tasks ⇒ renaming does isolate memorisation; revisit the Oct 7 decision to drop the renamed dev set.

## Thesis proposal "Found, Not Finished" — October 23, 2026
- Proposal: `research_deliverables/thesis_found_not_finished.md` (supersedes EBST if adopted). Prior-art check: `research_deliverables/surveys_2026-10-22/novelty_obligation.md` — CodePlan (2309.12499) owns obligation propagation itself; 2610.25826 owns "localisation isn't the bottleneck"; our delta = edit-triggered passive frontier in an autonomous small agent + test-necessity labels + shuffle control.
- **New CPU findings** (`analysis/find_vs_finish.py`, new-60): v14 resolves 27/43 single-file vs 3/17 multi-file (Fisher p=0.003); v7 22/43 vs 1/17 (p=0.001). v14 unresolved: 29/30 saw a gold file; 15/30 partial coverage, 9/30 edited elsewhere.
- **Obligation pilot** (`analysis/obligation_pilot.py`, xg LOO, rs_v7 ×5): tasks with ≥2 test-necessary gold groups 0/35 runs, 0/7 tasks vs single 66/185 runs, 17/38 tasks (p=0.034). Necessary co-change pairs: same file 26/28, executed-call 1-hop 4/28 → obligations run through fields/forwarding/overrides, not calls. n=7, pilot only.
- **Oct 23 — CPU gate built and run.** `tools/oblig.py` (stdlib; `edit`/`check`/`show`, <1 s, output ≤1,000 chars) passes functional tests on 7/7 pilot repos (`tools/test_oblig_pilot.py`). **E0a (pre-registered) GATE FAIL:** typed frontier hit@6 on test-necessary pairs 0.45 (need 0.80); untyped AST graph 0.54; BM25 0.36; same-file 0.29–0.30; executed-call 0.29; provided graph 0.27. AST structure beats lexical/same-file baselines (cluster-bootstrap CIs exclude 0), but change-typing adds nothing and recall is ~0.5. Diagnosis: the transfer rules block forwarder→callee and new-read→writer propagation. Per pre-registration: no GPU on E1–E4 until a clean external test (E0b) passes. n is small (7 tasks).
- **Oct 23 — E0b (pre-registered, deciding) GATE FAIL.** 120 SWE-rebench tasks / 120 repos, 1,192 co-change pairs, oblig.py v2 frozen. hit@6: TYPED 0.19, UNTYPED 0.24, **BM25 0.42**, same-file 0.21–0.29. Typed frontier ≈ random same-file; lexical similarity of the edit wins (repo-cluster CI TYPED−BM25 [−0.29, −0.18]). **Decision (per pre-registration): no GPU on E1–E4; obligation frontier dropped as a paper claim; paper = find-vs-finish analysis.** oblig.py kept as a tool. Open question worth a sentence in the paper: co-change is better predicted lexically than structurally.
- **Oct 23 — Paper direction locked: "Found, Not Finished" analysis paper.** Outline: `research_deliverables/paper_outline_find_vs_finish.md` (supersedes EBST `paper_outline.md`). P0 analyses before the Nov 5 freeze, all CPU: A1 confound control (multi-site vs patch size), A3 exec-graph LOO on new-60, A4 completion counterfactual (agent patch + missed gold sites). Optional A10 (lexical co-change hint, ~4 quota-h) only if P0 is done by Oct 30.
- **Oct 23 — A1 (pre-registered, Kaggle CPU) prediction FAILED.** After controlling for gold patch size, issue length and repo, multi-file is no longer significant (pooled new-60 OR 0.32 [0.04, 2.54], p = 0.21); patch size is (OR 0.47 per log line, p = 0.046). 15/17 multi-file tasks are in the largest size tercile, so the two are nearly collinear (low power). Paper §3 reframed: "large changes", not "multi-site". `results/runs/a1/a1_report.json`.
- **Oct 23 — A3/A4 (pre-registered, Kaggle CPU) predictions FAILED; decision rule triggered.** A3: 19/60 new-60 tasks have ≥2 test-necessary groups. A4: adding the missed gold sites to the agent's partial patch resolves 6/26 cases (23%; 2/13 distinct tasks); pure incompleteness in 5/26 (1/13 tasks, rich_3052). Agent-alone control 0/26. **The paper drops "not finished": the agent's own edits are mostly wrong too.** `results/runs/a3a4/`.
- **Oct 25 — All remaining paper analyses done (Kaggle CPU).**
  - **A12 site-swap: C7 Supported.** 52/71 clean necessary swaps fail (73%), on 19/20 tasks.
  - **A13:** the necessity gap vanishes after size (OR 1.68).
  - **A7:** wrong patches are submitted voluntarily; v14 large changes overflow (6/14).
  - **A5:** co-changes are plumbing or repeated edits; 95% share an edited identifier.
  - **A8:** BM25 wins via code similarity (prediction failed).
  - **A8b:** v2 rules give 0.63 on E0a, ties untyped.
  - Outline and claims ledger updated; next is writing.
- **Oct 27 — Atul's audit applied; A12v (pre-registered, Kaggle CPU) validates C7.**
  - Ledger fixes: C7 made conditional, then validated; C14 withdrawn; A13 pooled Fisher withdrawn (task-level p ≈ 0.048); A4 is no longer called an "upper bound"; A5 keyword caveat added; v16 included.
  - A12v (namespace/attribute/interface checks, decorators, identity controls, kept hybrids and tails): validated-clean 61/128; **43/51 compatible necessary hybrids fail behaviourally (84%), 18/19 tasks, CI 0.63–1.00**; 10/10 tails confirmed by hand.
  - **C7 Supported (validated).** A5 keyword-free shared identifier: 0.65 micro / 0.89 macro.
  - Atul second-read pending.
