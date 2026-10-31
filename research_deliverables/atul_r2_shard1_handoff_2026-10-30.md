# Atul R2 shard-1 handoff — October 30, 2026

**Status: the single approved run is RUNNING. This is an initial-session handoff, not a completed-rollout report.**

Recorded at 2026-10-30T18:25:05.625101+00:00. Terminal findings must be appended after independent recovery and validation. Original source files and existing experiments have not been changed.

## Execution identity and authorization

- Repository: `/home/atul-baliyan/Desktop/gemma4-swe-agent-main`.
- Branch: `main`; source commit **66f30edfbe70090219a3f906c845f2e17c1ec164**.
- Latest GitHub main was checked through a read-only API request and matches this local commit. No fetch, pull, branch switch, commit, push to GitHub, or PR was performed.
- Exactly one unchanged GPU push was invoked and accepted: **Version 1** of [atulbaliyan/gemma4-factory-rollout-s1](https://www.kaggle.com/code/atulbaliyan/gemma4-factory-rollout-s1).
- Push command started **2026-10-30T18:20:27.652692+00:00**, corresponding to **October 30, 23:50:27 IST**.
- The accepted CLI return code is 0. Durable exclusive attempt marker: `data/runs/atul_r2_s1/preflight/push_attempt.json`; captured stdout/stderr are alongside it. No automatic retry is permitted.

## Preflight verification

The installed Kaggle CLI is 2.2.4 and worked without an upgrade. Existing authentication was used normally; credentials/API tokens were never inspected, printed, or copied. All eight mine-inventory notebook refs belong to atulbaliyan. Six existing project notebooks were COMPLETE; two old unrelated notebooks have no session (404). No account notebook was observed RUNNING or QUEUED before launch.

The requested new notebook was absent from the complete eight-entry own inventory. Its precreation GetKernel returned 403 while other authenticated inputs and account APIs succeeded. This was distinguished from an authentication failure; version 1 was subsequently accepted and independently readable.

Both PRIVATE factory inputs are accessible under their actual source owner:

- `shreysharma07/gemma4-factory-v0`: 269 tasks, 281 listed files.
- `shreysharma07/gemma4-factory-v1`: 332 tasks, 335 listed files.

Only small tasks/task-meta manifests and metadata were downloaded for allocation verification, not all snapshots. Every assigned snapshot appears in the corresponding remote inventory. Source metadata states PRIVATE and license `other` for both datasets. No new distribution rights are inferred.

All additional declared inputs were accessible: metric/gemma-4-developer-agent-wheelhouse, shreysharma07/test-deps-wheels, scorer-env v1, devset-build v5, test-overlays v3, the competition files, and google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2 model files.

The combined 601 task IDs are unique. Applying the exact sorted-ID/seed-0 shuffle and stride allocation reproduces Shrey's entire completed shard-0 list. Shard 1 has exactly 300 IDs and zero overlap. Planned shard-1 strata: **181 singles, 88 pairs, 31 triples; 131 v0 and 169 v1 tasks**. These are assignment counts, not completed or solved counts.

## Notebook configuration and integrity

| Setting | Verified value |
|---|---|
| Submission | v14_thinking |
| Shard / total shards | 1 / 2 |
| Rollouts per task | 1 |
| Tasks | 300 |
| Concurrency / chunk size | 6 / 30 |
| Per-task time cap | 12 minutes |
| Session budget / watchdog allowance | 7 hours / 0.25 hours |
| GPU / machine | enabled / NvidiaL4 |
| Internet | disabled |
| Notebook visibility | PRIVATE |
| Adapters | none |

All 25 code cells parse; all 20 embedded evalkit modules match current source (ignoring trailing whitespace only); all six embedded submission files are byte-identical to submissions/v14_thinking. The analyzer also matches. No prepared notebook, metadata, model, prompt, sampling, budget or task allocation was modified or regenerated.

- Original notebook SHA-256: `a5a87629ba9c58ba6362ecfa177a431e70fd6a76e756eebdd9229b1511c7d6b0`.
- Original metadata SHA-256: `15de095c883b995d619e7456d7c98a2651d02c2946e2e0409a7cef8e875dc456`.
- Shard-1 ordered-ID digest: `d01a97fd0b8461e3c15485b3e7f82ae698f0700ac057dbc8f735614afe66e3ed`.
- Complete manifest hashes/module checks: `data/runs/atul_r2_s1/preflight/notebook_validation.json` and `inputs.json`.
- Immutable submitted source copies: `data/runs/atul_r2_s1/submitted_notebook/`.

Independent post-push metadata confirms current version 1, NvidiaL4, GPU true, Internet false, PRIVATE true, and exact source-cell equality. The numeric version-label status request initially returned 404; normal/latest status succeeds and is bound to independently verified current version/source. This was a read-only monitoring request correction, not a notebook modification or retry.

## Monitoring and current quota ledger

Persistent heartbeat monitor **monitor-approved-r2-shard-1-rollout** was created and confirmed ACTIVE, with its saved 15-minute schedule rechecked. The monitor is instructed to recover outputs, validate results, create the authorized private results dataset, attempt supported sharing, and report limitations. Scheduled cadence is not continuous observation; actual snapshot times are saved.

Latest recorded observation: **2026-10-30T18:22:45.840528+00:00**.

| Accounting field | Hours |
|---|---:|
| Pre-launch total / used / reserved | 30.000000 / 0 / 0 |
| Pre-launch available | 30.000000 |
| Latest observed used | 0.074276 |
| Latest observed reserved | 0.000000 |
| Latest observed available | 29.925724 |
| Planning upper bound, not guaranteed charge limit | 14.500000 |

Observed status is **KernelWorkerStatus.RUNNING**, with no failure message or alert in that snapshot. A bounded live-log stream was saved with no observed watchdog or first-chunk-health-abort marker. This does not prove that startup/health checks or tasks have completed. Completed/resolved counts are **UNKNOWN pending actual task evidence**; no percentage is inferred from elapsed time or quota.

Historical shard-0 runtime is approximately 5.8 session-hours. Comparable shard-1 duration is an estimate only. Budget/watchdog attention points measured from the push are approximately October 31 **06:50 / 07:05 IST**, respectively; the notebook's actual budget clock begins in its settings cell. If still RUNNING at/after 7.25 hours from push, the monitor must alert for manual attention rather than assume the watchdog stopped the platform.

Project documents describe a Saturday quota reset. If this run spans a reset/reversal, retain consecutive readings and disclose the accounting discontinuity. A final before/after account-used delta alone may not identify this run's charge; runtime proxies or a 2x estimate must not be labelled an attributable invoice.

## Recovery and research analysis — pending terminal status

No terminal outputs are yet claimed recovered or verified. Expected originals are factory_rollout_report.json, rollout_tasks.jsonl, run.log and factory_rollout.tgz with nested eval_rollout_s1.tgz. Recover all pages into a new directory, preserve originals, inspect/extract archives safely, verify source/version/shard, reconcile raw per-task outcomes/traces/test logs, and compute SHA-256 checksums. Report partial/missing/unknown results explicitly.

Requested resolution rates, cap events, context errors, patch/verification metrics, rewrite fallbacks, throughput, session duration and quota impact remain pending. Direct errors must be distinguished from threshold proxies and analyzer heuristics. Never execute recorded agent commands locally.

Shrey's unchanged committed shard-0 report and task_results.jsonl provide 301 outcomes and 176 resolved (58.47%). Reported group counts: singles 142/189, pairs 28/86, triples 6/26; rollout throughput 54.29 tasks/hour; time-cap proxies 13, call-cap proxies 68, empty patches 15. Source summary also reports overflow 8.3% and stray files 31.9%, with its analyzer's limitations. No combined 601-task success rate is available until shard 1 finishes and all IDs/definitions reconcile.

Successful trajectories are candidates for training only after quality checks. No training or additional model evaluation is authorized or started.

## Private dataset and collaborator state

**Results dataset not yet created:** creation waits for verified outputs. Requested future ID is `atulbaliyan/gemma4-factory-rollout-s1-results`, title “Gemma 4 Factory Rollout — R2 Shard 1 Results.” Its anticipated URL is https://www.kaggle.com/datasets/atulbaliyan/gemma4-factory-rollout-s1-results; this is a requested destination, not evidence of an existing dataset or version.

Only original required evidence and a checksum/provenance manifest may be staged for private upload. Preserve source rights with consistent metadata; neither resource may be made public. Never upload credentials, environment dumps, unrelated research, or redundant huge archives. An uncertain creation must not be retried blindly.

**Notebook collaborator:** not added or verified. **Dataset collaborator:** pending dataset creation. The installed public CLI/SDK exposes no collaborator-management methods. The only available browser is the in-app browser; its Kaggle home page shows Login, and the private notebook is inaccessible there. No credentials or cookies were transferred to work around that boundary. API authentication works independently.

If browser authentication remains unavailable after recovery/dataset creation, sharing requires the user to use their authenticated Kaggle session:

1. Notebook: open its URL, then Options → Sharing → Add collaborator; choose shreysharma07 with Can view, keep PRIVATE, save and verify the entry.
2. Results dataset, once actually created: open its URL, open its Sharing/collaborator controls (typically through Settings), add shreysharma07 with Can view, keep PRIVATE, save and verify the entry. UI labels must be verified in the signed-in interface; sharing the dataset does not share the notebook.

No sharing success is claimed. These are operational steps, not a request to reauthorize the already-approved work.

## Preservation and next steps

The initial Git status was clean. All 815 tracked files have preflight SHA-256 records. Git refs, stashes and worktrees were recorded for final comparison. Only the new untracked run evidence/scripts and this new untracked handoff report are intended local changes. HQ, experiments, existing research, notebooks, datasets and Git history remain untouched.

The original request and continuation guide are saved as `data/runs/atul_r2_s1/preflight/user_request.txt` and `RECOVERY_AND_SHARING.md`. Future monitor turns must read summary.json and the push marker first, **never invoke launch_once.py**, and append terminal/recovery findings to this report while preserving its initial text. Stop/delete the monitor after completion or a genuine manual-sharing blocker.

Shrey's next steps after receipt: verify private dataset access and manifest checksums, reconcile the nonoverlapping 601 IDs, quality-filter successful trajectories including stray files and verification reliability, then decide the separately authorized training-data build. Do not start training from this initial RUNNING handoff.

## Initial-session checklist

- [x] Both private source datasets accessible.
- [x] Exact notebook configuration and task allocation verified.
- [x] Exactly one run submitted; version 1 accepted.
- [x] Post-push source/version/PRIVATE visibility checked.
- [x] Persistent scheduled monitor setup confirmed.
- [ ] Terminal run status and completed-task counts.
- [ ] Outputs recovered, hashed and validated.
- [ ] Private results dataset created and verified.
- [ ] Dataset and notebook collaborator entries verified.
- [x] Initial local handoff report created; terminal appendix pending.
- [x] No additional GPU runs, training, leaderboard submissions or Git publication.


## Terminal recovery and private dataset handoff — October 31, 2026

This appendix supersedes pending statuses in the initial-session account above, while preserving that historical text. Verified terminal status: **COMPLETE**, exact notebook **Version 1**, PRIVATE, NvidiaL4, GPU enabled, Internet disabled, source cells unchanged and no adapters. Sole launch: 2026-10-30 18:20:27.652692 UTC (23:50:27 IST). First terminal observation: 2026-10-31 00:55:24.097219 UTC. Final independent source/status/quota check: 03:01:37.734385 UTC (08:31:37 IST). The notebook finished earlier than those observations; elapsed time until observation is not runtime.

Notebook: https://www.kaggle.com/code/atulbaliyan/gemma4-factory-rollout-s1

Results dataset: https://www.kaggle.com/datasets/atulbaliyan/gemma4-factory-rollout-s1-results/versions/1

### Coverage, results and shard comparison

All **300/300 distinct expected shard-1 IDs** have task results: **174 resolved (58.00%)**, **126 unresolved**, **0 missing**, **0 duplicate IDs**. Unresolved includes errors and wrong/nonpassing patches; it does not mean all 126 had infrastructure failures. Exactly 300 rewritten task records and report task entries agree with the approved IDs and source meta manifests. There is no overlap with shard 0's 301 IDs. Both shards use the same approved configuration and compatible outcome definitions; the combined unique set is 601, not repeated evaluations of identical issues.

| Stratum | Shard 1 | Shard 0 | Combined |
|---|---:|---:|---:|
| overall | 174/300 (58.00%) | 176/301 (58.47%) | 350/601 (58.24%) |
| single | 136/181 (75.14%) | 142/189 (75.13%) | 278/370 (75.14%) |
| pair | 36/88 (40.91%) | 28/86 (32.56%) | 64/174 (36.78%) |
| triple | 2/31 (6.45%) | 6/26 (23.08%) | 8/57 (14.04%) |
| v0 | 94/131 (71.76%) | 104/138 (75.36%) | 198/269 (73.61%) |
| v1 | 80/169 (47.34%) | 72/163 (44.17%) | 152/332 (45.78%) |

These are descriptive results on different shard tasks, not a paired causal comparison. Triple-task success is weak, and simple aggregate success does not validate training-data quality or issue-rewrite equivalence. Shard 0 raw task results and report were available at `results/runs/r2_s0/`; its raw traces were not available locally, so event-level or corrected behavioral metrics are not independently combined.

### Runtime, failures and counting definitions

Rollout phase: **325.3 minutes**, **55.33 completed tasks/hour**, versus shard 0's 332.7 minutes and 54.29 tasks/hour. End-to-end platform-log elapsed proxy: **5.7054563 session-hours** (about 5h42m20s); end-to-end throughput proxy 52.58 tasks/hour includes setup/rewrites. Combined processing throughput is 601 divided by the sum of the two phase times: **54.80 tasks/hour**, not throughput over simultaneous wall-clock execution.

All ten chunks fully completed. Chunk indices 0–9 took **30.7, 26.7, 35.8, 31.5, 33.1, 31.7, 25.6, 33.4, 44.8, 32.1 minutes**. Last fully complete chunk: **9**, all 300 records merged. No WATCHDOG or failed-chunk record was found. Tool-level timeouts still occurred.

| Metric and definition | Shard 1 | Shard 0 | Combined where compatible |
|---|---:|---:|---:|
| Duration ≥715 seconds (time-cap proxy) | 15/300 | 13/301 | 28/601 |
| Harness tool calls ≥27 (call-cap proxy) | 56/300 | 68/301 | 124/601 |
| Actual BudgetExceeded tool-response events | 18 across 17 tasks | 32 reported by original analyzer; raw trace audit unavailable | Not independently verified |
| Actual tool TimeoutExceeded events | 4 | 2 reported by original analyzer | Not independently verified |
| Task error records | 32 | 36 | 68/601 |
| Actual context-error task messages | 19/300 | 25/301 | 44/601 |
| Empty final patches | 14/300 (4.67%) | 15/301 | 29/601 |
| Issue rewrite fallbacks | 1/300 (`names_tests`) | 7/301 | 8/601 |
| Original stray-file heuristic | 98/300 (32.67%) | 31.9% reported | Exact combined count not asserted |

Rewrites: 299 accepted, one fallback; rewrite phase 6.7 minutes. Time/call thresholds are proxies, not proof of actual denied calls. Context errors are counted by task error messages, separate from generic warnings. No claim attributes failures to shared wrapt warnings alone.

Original analyzer metrics are preserved intact, rather than silently redefined: median tool calls **21**, repeated-identical-call share **6.4%**, explicit submit **88.7%**, empty patch **4.7%**, tests after last edit **3.7%**. Shard 0 originals are respectively 21, 6.6%, 88.0%, 5.0%, 5.6%. Original failure-stage labels are heuristics: 84 wrong-fix-unverified, 21 saw-gold/edited-elsewhere, 12 saw-gold/never-edited, 5 edited-wrong-file, 2 wrong-fix-verified-locally, 1 found-file-not-lines and 1 never-found-gold-file, plus 174 resolved. Gold-based localization labels must not be mistaken for final source mutation or independent verifier outcomes.

A separate static audit recognizes Python/shell mutation intent as well as write_file calls: 282 tasks contain recognized mutation intent; **58/300 (19.33%)** have a returned test invocation after the last recognized mutation intent, and **48/300** show passing-test text afterward. Literal source targets can be mapped for 265 tasks; 58 have a returned test after their last mapped source mutation intent. These are intent/invocation proxies, not proof that writes succeeded or tests verified the final patch. Dynamic writes may be missed; collection/environment errors can occur in test invocations. Recorded commands were never executed locally. All 6,258 tool associations used ordered same-tool matching because IDs were unavailable, with zero unmatched observations or missing responses under that definition.

### Candidate training trajectories and next steps

The 174 resolved trajectories are an upper-bound candidate pool. A conservative reproducible filter retains **132 shard-1 candidates**: resolved, nonempty final patch, official test exit code 0, trace and grader log present, and no stray files under the original documented heuristic. The other 42 resolved tasks have stray-file labels and require manual review; some files could be legitimate. This is not a ready-to-train dataset. Across both shards, **350 resolved** are potential candidates; the combined strict count is **unknown** without shard 0 raw trace/patch review.

Shrey should: (1) download private dataset Version 1 and verify manifest hashes; (2) confirm access and use the recovered source/config and original archive; (3) deduplicate the compatible 601 IDs; (4) review the 132 candidate IDs in `analysis/metrics.json`, rewrite semantics, final patches, stray-file legitimacy and verification traces; (5) recover equivalent shard-0 evidence and apply the same filter; (6) split by source repository/issue family to prevent leakage, with held-out evaluation decided before any separately authorized training. No training or training dataset conversion was started.

### Original evidence, integrity and limitations

Durable recovery directory: `data/runs/atul_r2_s1/recovery_20261031T024440Z/`. Original output directory: `original/`. All 54 listed output files plus the platform log were recovered; the inventory had one output page. Before/after remote identity checks confirmed exact current Version 1 and source. The installed CLI's output retrieval does not enforce the numeric version suffix; binding relies on independently confirmed unchanged current Version 1, with no other versions submitted. Version-specific dataset downloads do enforce Version 1.

Outer archive: four regular files; inner evaluation archive: 1,192 members, 1,184 regular files. Extraction rejected traversal, links and special files and enforced size/member bounds. Extracted bytes were checked against original archive members. All 300 task traces are present, but only **294 grader-output logs**, so six task logs are unavailable; this does not erase the explicit outcomes in merged task results. Individual chunk copies were removed by the upstream merge/package step; none survived for a duplicate chunk-versus-merged comparison. The outer archived run.log is an exact prefix of final run.log because RUN_DONE was printed after packaging. Optional submission ZIPs differ only by directory entries; all six source files agree, and the approved embedded ZIP matches the recovered source ZIP exactly.

An initial CPU validator rejected a harmless ZIP directory-entry difference; corrected validation preserved originals and verified the six actual source files. This was a validator assumption, not a failed GPU run or a reason to relaunch.

Derived analysis: `recovery_20261031T024440Z/analysis/metrics.json` and `task_audit.jsonl`. CPU reproduction scripts: `validate_recovered.py`, `analyze_results.py` and `verify_dataset_roundtrip.py` in this run directory. Archive validation: `recovery_20261031T024440Z/archive_validation.json`. Full download provenance: `download_manifest.json` and `output_inventory.json` in that directory. The static audit imports pure helpers from the existing `docs/optimization_v14/audit.py` without changing it or invoking its main function.

| Original file | SHA-256 |
| `factory_rollout_report.json` | `29e04e3235c82b69cd6dc35e2ef16fa4e399a97d5268a3b33e82f6e956bef219` |
| `rollout_tasks.jsonl` | `0a75913f1c1a59513ae17bdd386786d27e8fc234e1fdbe9c3be51717be07b319` |
| `run.log` | `e5cd96265870398d17d3a73e9907d44fd2e700bf5ec1dbd5158e9669bef61856` |
| `factory_rollout.tgz` | `0a216547987a0c2b7db4e4bb21d54678b21d16997b71db278f8bff5edc88ad2c` |

The private dataset contains the four required original files, eval_cfg, source submission material, a provenance/checksum manifest, rights note and derived metrics. No credentials, account telemetry, sandbox environment dump, unrelated evidence, or redundant inner archive were uploaded. Kaggle automatically expanded the optional `submission_src.zip` into six source files; this conversion is disclosed rather than claiming preservation of that ZIP container remotely. The original ZIP is retained locally. Required rollout archive and JSONL were preserved with `--keep-tabular`; no dataset update or second version is authorized or needed. Source dataset license metadata says `other`; this private upload uses `other`, asserts no open license, and retains upstream rights.

### Observed GPU quota ledger

Prelaunch account: **0 used, 0 reserved, 30 available of 30 quota-hours**. Last pre-reset observation at Oct 30 23:54:06.619461 UTC: **11.1196264339 used, 18.8803735661 available**. First terminal observation at Oct 31 00:55:24.097219 UTC showed **0 used, 0 reserved, 30 available**. Final independent check at **03:01:37.734385 UTC** still shows **30 available**, **0 reserved**. These readings crossed a Saturday account reset/reversal. The endpoint-used delta is 0, but **actual attributable final billed charge is UNKNOWN**, not zero. Twice the platform runtime gives **11.4109126 quota-hours**, explicitly an estimate rather than a measured bill. Shard 0's 11.6 quota-hours is a historical reported approximation, not a new measurement.

The recorded 15-minute schedule was not continuous observation. Measured earlier snapshot gaps ranged 1.35–61.29 minutes, and the final check was approximately 126.23 minutes after the previous terminal snapshot. There was no observation proving an active session after the 7.25-hour safety limit; platform logs show it finished within budget. No budget extension or guessed cancellation request was issued.

### Privacy, sharing and manual blocker

Dataset owner `atulbaliyan`, Version 1, PRIVATE are independently verified. The actual metadata collaborator entry is **`shreysharma07`, `READER`**. This supersedes the initial-session assumption that all collaborator APIs were unavailable: the installed SDK supports dataset collaborators through metadata updates. Full existing metadata and PRIVATE=true were preserved. Proof: `dataset_metadata_after_share/dataset-metadata.json`, `dataset_file_inventory.json` and `dataset_verified.json`.

Notebook privacy and Version 1 are verified through authenticated SDK reads, but **notebook sharing is NOT verified/completed**. Installed supported notebook services do not provide the analogous collaborator-setting operation; repushing a notebook to change sharing is forbidden. The only available browser is signed out: private notebook shows unavailable, Kaggle homepage visibly shows Login. No credentials were inspected or copied.

**Required manual step:** sign in as atulbaliyan, open https://www.kaggle.com/code/atulbaliyan/gemma4-factory-rollout-s1 → **Options → Sharing → Add collaborator** → enter **shreysharma07** → select **Can view** → save. Confirm the collaborator is listed and the notebook remains **PRIVATE**. Sharing the dataset does not share the notebook. No new notebook version, Run, or Save & Run is required.

### Preservation and final checklist

All **815 tracked-file hashes unchanged**; HEAD/main remains `66f30edfbe70090219a3f906c845f2e17c1ec164`. User branches/remotes/tags/stash refs, stash hashes/messages and worktrees are unchanged. Codex app-managed `refs/codex/turn-diffs/` rotated between turns; this is disclosed in `final_preservation.json`, not claimed byte-identical. The first stash display comparison used a different format; the corrected hash/message comparison passes. No agent Git write, staging, commit, push, merge, branch/worktree operation or PR occurred.

The monitor `monitor-approved-r2-shard-1-rollout` was **deleted** after terminal status/recovery confirmation, with only the manual notebook-sharing blocker remaining. No additional GPU experiment, retry, training or leaderboard submission occurred.

- [x] Both private factory datasets and required inputs verified.
- [x] Correct immutable notebook configuration; exactly one accepted version.
- [x] COMPLETE; all 300 tasks accounted for; 174 resolved, 126 unresolved, zero missing IDs.
- [x] Original outputs recovered, safely inspected and CPU-audited.
- [x] One PRIVATE dataset created and ownership/privacy/version checked.
- [x] Dataset actual READER collaborator verified for Shrey.
- [ ] Notebook collaborator entry: manual signed-in UI action required.
- [x] Local handoff report appended without replacing earlier text.
- [x] Source repository preserved; only authorized untracked run/report paths.
- [x] Monitor deleted; no additional GPU work or Git publication.

Final `git status --short`:

```text
?? data/runs/atul_r2_s1/
?? research_deliverables/atul_r2_shard1_handoff_2026-10-30.md
```


## GitHub publication verification — October 31, 2026

**Latest sharing status:** Atul confirms in the October 31 publication request that both the notebook and dataset have been shared with Shrey. Dataset PRIVATE visibility and the actual `shreysharma07` READER entry were independently reverified through authenticated metadata reads. Notebook Version 1 COMPLETE, PRIVATE visibility and unchanged source were independently reverified at **03:16:52 UTC / 08:46:52 IST**. Notebook collaborator access is user-confirmed, not independently reverified by the signed-out browser; no permissions were changed. This updates the earlier manual-sharing-pending entry without pretending an API verification occurred.

The existing original archive was found and reused; no new download, GPU execution, extraction into the project root or overwritten evidence was necessary. CPU revalidation consumed every safe outer and nested tar member, checked gzip/tar integrity, rejected unsafe paths/links/special members, and matched all **300 unique approved IDs, 300 traces, 174 resolved and 126 unresolved**. Its SHA-256 matches the previous private-dataset Version 1 round-trip download. Shard 0 remains unchanged at 176/301; zero shard overlap gives **350/601 (58.24%)**. Validation details: `data/runs/atul_r2_s1/publication_archive_validation.json`.

Public results follow `results/runs/r2_s0/` in **`results/runs/r2_s1/`**:

- `factory_rollout_report.json`: original report aggregate fields and IDs; private `rewrite.templates` (300 issue texts) and `rewrite.samples` (3 issue texts) omitted for public publication. The original complete report remains untouched locally and in Kaggle. This is a documented derivative, not a byte-identical original.
- `summary.json`: byte-identical original trace-analyzer summary, matching the shard-0 convention.
- `task_results.jsonl`: byte-identical original merged outcomes, 300 records.
- `audit_metrics.json`: previously reproduced shard/combined metrics, proxy definitions, candidate rules, missing-evidence caveats and historical quota observations. Its earlier observation timestamp remains explicit.
- `provenance.json`: public file hashes, original report hash, archive hash, dataset/notebook identity, transformations and sharing/accounting limitations.

The original TGZ is **5,121,885 bytes** (about 4.88 MiB):

```text
Local: /home/atul-baliyan/Desktop/gemma4-swe-agent-main/data/runs/atul_r2_s1/recovery_20261031T024440Z/original/factory_rollout.tgz
SHA-256: 0a216547987a0c2b7db4e4bb21d54678b21d16997b71db278f8bff5edc88ad2c
```

**TGZ is intentionally not committed.** GitHub repository visibility was verified PUBLIC. The archive contains private-source task text, patches and model traces; it belongs in the shared PRIVATE Kaggle dataset, despite being small enough for ordinary Git. `*.tgz` is also ignored; no ignore override, LFS configuration or history rewrite was used. Private rewritten task JSONL and raw traces are likewise excluded. Public per-task outcome records contain metrics, not issue bodies, patches or traces.

### Shrey: retrieve the complete original evidence

Use your existing authenticated Kaggle account with reader access to **[private results dataset Version 1](https://www.kaggle.com/datasets/atulbaliyan/gemma4-factory-rollout-s1-results/versions/1)**. In its file list select **`factory_rollout.tgz` → Download**, or run the installed supported CLI using the version-qualified dataset reference and a fresh directory:

```bash
mkdir r2_s1_original_v1
kaggle datasets download -d atulbaliyan/gemma4-factory-rollout-s1-results/1 -f factory_rollout.tgz -p r2_s1_original_v1
sha256sum r2_s1_original_v1/factory_rollout.tgz
```

Verify the checksum above, inspect the member listing, then safely unpack into a dedicated directory. `factory_rollout.tgz` contains original report/tasks/log and nested `eval_rollout_s1.tgz`; the nested archive contains all 300 merged outcomes and traces. Original final `run.log` is additionally provided separately in the dataset; the log inside the outer archive is a verified prefix because the final RUN_DONE line followed packaging. Keep the existing six-missing-grader-log limitation and upstream-removed chunk-copy limitation in the interpretation.

Fresh account quota: **30 available, 0 used, 0 reserved** at 08:46:52 IST. This does not establish zero charge: peak measured pre-reset use was 11.1196264 hours, final attributable charge remains unknown because of reset/reversal, and runtime-times-two 11.4109126 remains only an estimate. Runtime and failure metrics in the terminal appendix are unchanged; no new task was executed.

Only the five small shard-1 result files and this handoff report are authorized for the requested direct-main commit. Shrey's HQ, experiments, source, notebooks and all previous result files remain unchanged. Raw recovery evidence remains local and untracked/ignored. No PR is created. The previously deleted monitor remains stopped.
