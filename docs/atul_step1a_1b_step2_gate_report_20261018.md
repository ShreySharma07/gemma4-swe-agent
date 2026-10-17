# S3 Compact, S1 Own and behaviour gate — consolidated technical report

Date: October 18, 2026 (IST). Author: Atul's execution workspace.

**Current status: E1 PASS (147/147 episodes, 182/182 edits); S3 Compact version 1 submitted exactly once and initially RUNNING with unchanged `--budget-h 3`. S1 and the behaviour gate require separate approval.**

This is the single report for Steps 1a, 1b, 2 and CPU E1. Earlier preflight blockers are reconciled below; detailed evidence and the original initial-stage checks remain identifiable as history. No new training measurements are reported until actual output exists.

## 1. Executive Summary

The experiment objective is to test whether grounded, compact public demonstrations (S3) or the agent's own successful trajectories (S1) restore useful edit behaviour after Step C's adapter produced 114 empty patches in Step D. The intended sequence is S3 training, review, S1 training, review, then a fixed-prefix behaviour gate comparing base, S1 and S3.

The reporting correction supersedes the original request for three separate reports. No separate Step 1a, Step 1b or Step 2 report was created.

| Stage | Completed work | Current outcome |
|---|---|---|
| Workspace/Git | Authorized feature branch in the primary workspace | PASS |
| Artifact identity | Both NPZs, gate prefixes and raw row archive match all four handoff hashes | PASS; rechecked before S3 push |
| Kaggle access | Independent workspace CLI; fresh authenticated quota/inventory/status | PASS; pre-push 30.00 h available, 0 used/reserved |
| CPU E1 | Exact benchmark bases, isolated filesystem replay, initial/final teacher-blob checks | **PASS: 147/147 episodes, 182/182 edits, zero failures/unverifiable edits** |
| Step 1a S3 | Exact notebook verified and submitted once as version 1 | **RUNNING**; training/output checks pending |
| Step 1b S1 | Dataset/configuration inspected | NOT RUN; separate approval required |
| Step 2 gate | Prefix counts and harness scoring inspected | NOT RUN; separate approval required |

The original missing-provenance issue was resolved through the authoritative benchmark, not by guessing repository versions. All 150 distinct source downloads succeeded, covering 151 episode/file states. Initial and final Git blobs match teacher patch indexes in all 151 cases. The empty-file case is valid at exact zero-byte source state. E1 details and limitations are in §7.

Only one S3 run is now authorized. Its expected duration is 2.1–2.4 session-hours / 4.2–4.8 quota-hours with the unchanged 3 h budget. The user explicitly withheld authorization for S1, the gate, retries and configuration changes. Training losses, export integrity, actual runtime and quota charges remain pending until this one run produces evidence. E1 does not establish coding-agent behaviour or clear synthesized test-feedback concerns.

## 2. Background and Research Motivation

Sources: [handoff](step0_handoff_20261017.md), [strategy](finetune_strategy_plan.md), [training guide](post_training_guide.md), [Step C report](step_c_20261015_report.md), [Step D report](step_d_d0_20261016_report.md), [Step D diagnosis](step_d_failure_analysis_20261016.md), and [E0/E2 audit](e0_e2_audit_20261016.md).

Step C established that training, export and serving integration could work. Its preserved `train_report.json` records rc 0 and four valid 410-module exports. Its validation loss fell substantially. Step D then measured 0/114 resolved and 114/114 empty patches with that adapter. Consequently, lower imitation loss and adapter integrity cannot approve a coding-agent release.

The subsequent E2 audit identified missing action state and an imbalanced supervision mix: 80% of Step C windows started mid-run, 79% of edits lacked the edited file in that window, and scratch scripts received 56% of supervised tokens versus 17% for fixes. These are measured dataset properties; their individual causal contributions to collapse remain unisolated.

S3 compresses a demonstration to one grounded episode: locating searches, the edited region, source edits, an existing-test command when present, and submission. It drops scratch files and retains informative decisions as command comments. S1 supplies an own-policy control using v7's solved trajectories, reducing the cross-agent style change while accepting limited task diversity.

| Hypothesis | Intended evidence | Current status |
|---|---|---|
| Grounded source-edit supervision improves edit decisions | S3 edit rate at `file_in_view` at least base; overall gate rate at least base | Untested |
| Own-policy demonstrations preserve action selection | S1 gate compared with base and S3 | Untested |
| Lower update dose reduces drift | Rank 16, half Step C's peak learning rate, fewer optimizer steps | Configuration inspected; effect untested |
| Compact demonstrations can induce premature editing | Elevated edit frequency in `early` prefixes | Untested |
| Converted source-edit success feedback corresponds to executable changes | E1 at exact repository commits | Verified for 182/182 retained edits; test outcomes outside scope |

The exact handoff specifies **two epochs** for both runs. The general strategy's earlier one-epoch recommendation is historical context; it was not substituted for the prescribed command.

## 3. Step 1a — S3 Compact Training

### Dataset design, identity and measured structure

| Property | Value/source |
|---|---|
| NPZ | `data/runs/train_s3_compact.npz` |
| SHA-256 | `1893c901e1c6c8116e0696f32cdb13041fc7252a012cefd964674ae0021625b4` |
| File size | 310,099 bytes |
| Windows / distinct tasks | 147 / 147 |
| Train / validation windows | 140 / 7 |
| Train / validation tasks | 140 / 7 |
| Total input tokens | 744,278 |
| Train / validation input tokens | 704,809 / 39,469 |
| Supervised mask positions, whole NPZ | 93,746 (12.6%) |
| Maximum sequence length | 8,171 tokens |
| Matched candidate episode repositories | 132 |

Array counts were read directly from NPZ members with a standard-library NPY reader; no model or training framework was loaded. “Supervised mask positions” is the mask sum across both splits, not optimizer-processed input tokens, and not necessarily identical to shifted next-token loss counts.

The preserved compaction summary reports 147/200 kept, 40 rejected for an unseen edited region, 11 for ungrounded edits and 2 for exceeding 8,192 tokens. Median episode length is 4,942 tokens; median retained calls is 6. The preserved E2 report records no continuation windows, no leaked tool-response supervision, no unclosed call spans and 0/182 edits whose file was absent from prior window context. It attributes 75% of supervised tokens to snippet edits. These are Shrey's existing audit results, not a newly repeated tokenizer audit.

### Exact prescribed configuration

```bash
export KAGGLE_OWNER=atulbaliyan
python3 kaggle/train/make_train_notebook.py data/runs/train_s3_compact.npz --name s3_compact --lr 5e-5 --epochs 2 --grad-accum 2 --save-every 9 --budget-h 3
.venv/bin/kaggle kernels push -p kaggle/train/l4_s3_compact
```

The builder command and one push command above were executed with `KAGGLE_OWNER=atulbaliyan`. Kaggle accepted **version 1** of `atulbaliyan/gemma4-train-s3-compact`; the initial status query returned **RUNNING**. There was exactly one push attempt, no retry and no configuration change. Actual training measurements are pending terminal outputs.

| Setting | Prescribed value / source default |
|---|---|
| Training base | NF4 loading of `google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2` |
| Hardware | 4×L4, FSDP |
| Adapter | `s3_compact`; rank 16, alpha 32 |
| Target projection families | q/k/v/o and gate/up/down; expected 410 actual modules |
| Peak learning rate / optimizer | 0.00005 / AdamW, weight decay 0 |
| Schedule | 5 warmup steps, then cosine |
| Epochs / accumulation / global batch | 2 / 2 / 8 windows |
| Seed / max-windows cap | 0 / none |
| Gradient clipping | Norm 1.0 |
| Offload | Sequences ≥4,097 tokens |
| Checkpoint cadence | Every 9 optimizer steps plus final |
| Validation cadence | Default `val_every=0`: start and end only |
| Budget argument | 3 session-hours; not a hard guarantee of billed usage |

Source schedule arithmetic gives 280 scheduled window presentations, 70 microsteps per rank and **35 optimizer steps**. Expected checkpoint tags are step9, step18, step27 and final, conditional on a complete run. This is more precise than the handoff's approximate 36-step description. Actual step counts and checkpoints remain unmeasured.

### Losses, checkpoints, resources and export verification

S3 version 1 is now RUNNING. `train_report.json`, `train_out/metrics.jsonl` and `run.log` have not yet been downloaded to `data/runs/atul_s3_compact/`; no actual loss or export result is inferred from successful submission. Consequently:

- Train/validation loss and loss curve: unavailable.
- GPU memory, utilization, throughput, actual runtime and quota charge: pending outputs and fresh quota accounting.
- Adapter tensors, hashes, shapes and serving-copy equality: pending output verification.
- Runtime warnings, failures or checkpoint interruption: pending log review; initial RUNNING is not a health or performance verdict.

**Verdict: RUNNING, version 1; training/export checks pending.** Required future checks remain rc 0, decreasing validation loss and a valid final 410-module export; integrity must include tensor shapes/dtype and file completeness. Lower loss alone cannot demonstrate behavioural improvement.

## 4. Step 1b — S1 Own Training

The handoff identifies the data as 82 solved v7 runs on 21 old-54 tasks, with the local sandbox's `wrapt` warning stripped from 783 responses. These provenance details are inherited from the handoff; original own-run conversion was not repeated.

| Property | Measured NPZ value |
|---|---|
| NPZ | `data/runs/train_s1_own.npz` |
| SHA-256 | `d3ade3498e4b3b10048fa983c3bda4ddbccdbd302cd0f028769d0d258a8655bc` |
| File size | 345,771 bytes |
| Windows / distinct tasks | 154 / 21 |
| Train / validation windows | 133 / 21 |
| Train / validation tasks | 18 / 3 |
| Input tokens, both splits | 1,083,946 |
| Train / validation input tokens | 933,475 / 150,471 |
| Supervised mask positions, both splits | 192,809 (17.8%) |
| Maximum sequence length | 8,189 tokens |

```bash
export KAGGLE_OWNER=atulbaliyan
python3 kaggle/train/make_train_notebook.py data/runs/train_s1_own.npz --name s1_own --lr 5e-5 --epochs 2 --grad-accum 2 --save-every 9 --budget-h 3
.venv/bin/kaggle kernels push -p kaggle/train/l4_s1_own
```

These are unexecuted prescribed commands. Expected identity: `atulbaliyan/gemma4-train-s1-own`. Settings otherwise match S3. The current scheduler groups four windows and truncates incomplete groups: `floor(133×2/4)=66` microsteps per rank, yielding **33 optimizer steps** and 264 window presentations rather than all 266. Which two presentations are omitted depends on the deterministic shuffled schedule; no task-coverage conclusion was derived here. Expected checkpoint tags are step9, step18, step27 and final.

No S1 `train_report.json`, `metrics.jsonl`, `run.log`, checkpoints or exports were produced. Loss convergence, validation performance, memory, throughput and runtime warnings are unavailable. New S1 GPU consumption is zero because it was not submitted.

**Verdict: NOT STARTED / NOT AUTHORIZED.** It cannot proceed without separate user approval, even after S3 is COMPLETE and reviewed. The small set of distinct tasks limits generalization claims even if future training loss and gate results are favorable.

## 5. Comparative Analysis — S3 vs S1

| Dimension | S3 Compact | S1 Own | Step C historical reference |
|---|---|---|---|
| Demonstrations | Compact converted external trajectories | Own v7 solved runs | Converted external trajectories in multiple windows |
| Windows / tasks | 147 / 147 | 154 / 21 | 1,017 / 200 |
| Total input tokens | 744,278 | 1,083,946 | About 7.37M |
| Supervised mask share | 12.6% | 17.8% | About 16% |
| Peak learning rate | 5e-5 planned | 5e-5 planned | 1e-4 actual |
| Epochs | 2 planned | 2 planned | 1 actual |
| Optimizer steps | 35 planned | 33 planned | 122 actual |
| Validation trajectory | Unmeasured | Unmeasured | 0.7718 → 0.3133 → 0.2909 → 0.2883 → 0.2882 |
| Export integrity | Unmeasured | Unmeasured | step40/80/120/final: 410 modules, shapes valid |
| Behaviour | Unmeasured | Unmeasured | D: 114/114 empty patches, 0/114 resolved |

S1 has approximately 1.32× S3's training-split input tokens and much less task diversity. Similar optimizer-step counts therefore do not mean matched token exposure, supervised-token exposure or compute. S3's 75% edit-token share from E2 must not be confused with its 12.6% overall supervised-mask density: these use different denominators.

The historical Step C values above were checked against its preserved report and metrics. Its first logged step loss is **1.7400**, its last is **0.3897**, median logged throughput **294.8 tokens/s**, and maximum logged global peak **18.88 GiB**. These exact values take precedence here over the guide's approximate initial training-loss description. This is a historical comparison, not a rerun.

| Step C validation point | Measured loss |
|---|---:|
| Start | 0.7718 |
| Step 40 | 0.3133 |
| Step 80 | 0.2909 |
| Step 120 | 0.2883 |
| End | 0.2882 |

The table provides the actual historical loss trajectory; S3/S1 curves cannot be drawn without their metrics. Even eventual lower validation loss across S3 and S1 would not rank their behaviour: validation tasks, targets and token distributions differ. No causal attribution among grounding, compression, reasoning retention, teacher style and dose is possible from configuration inspection alone.

## 6. Step 2 — Behaviour Gate

### Setup and exact intended configuration

```bash
export KAGGLE_OWNER=atulbaliyan
python3 kaggle/eval_l4/make_eval_notebook.py v7_noedit --gate --gate-adapters s1_own,s3_compact --adapter-from atulbaliyan/gemma4-train-s1-own --adapter-from atulbaliyan/gemma4-train-s3-compact
.venv/bin/kaggle kernels push -p kaggle/eval_l4/gate_kernel
```

The gate would serve the W4A16 base and both final adapters in one scorer-configured vLLM server. Adapter provenance must be established by training version and verified export hashes before use; neither new adapter exists from this workflow. `gate_report.json` was not generated.

The verified prefix file SHA-256 is `d07b5ef760aa4dd917800c2eb7641337d62e4b095b9fc26a31583bd4acf95e12`. Direct JSON inspection confirms 30 prefixes. The handoff specifies four samples each: 120 expected responses per model, 360 total. The inspected request uses temperature 0.2, top-p 0.95, top-k 40, `max_tokens=1024`, automatic tool choice and `enable_thinking=False`. This is a one-response diagnostic, not a full task evaluation under v7's normal output allowance.

| Model | Intended identity | Verdict | Overall pass rate | Edit rate at file_in_view | Batched / no-call / request errors |
|---|---|---|---|---|---|
| Base | Unadapted served base with v7 prompt/tools | NOT RUN | N/A | N/A | N/A / N/A / N/A |
| S1 | Final `s1_own` from its separate training notebook | NOT RUN | N/A | N/A | N/A / N/A / N/A |
| S3 | Final `s3_compact` from its separate training notebook | NOT RUN | N/A | N/A | N/A / N/A / N/A |

### Complete per-type table

| Type | Prefixes | Planned responses per model | Harness pass condition, exactly one call | Base | S1 | S3 |
|---|---:|---:|---|---|---|---|
| `file_in_view` | 8 | 32 | Classified source edit | N/A | N/A | N/A |
| `test_failed` | 8 | 32 | Edit or read; command differs from prior command | N/A | N/A | N/A |
| `test_passed` | 6 | 24 | `submit_patch` | N/A | N/A | N/A |
| `early` | 8 | 32 | Read/search | N/A | N/A | N/A |
| Total | 30 | 120 | Type-specific condition plus one call | N/A | N/A | N/A |

A candidate PASS requires `edit_rate_file_in_view >= base`, overall `pass_rate >= base`, zero batched responses and zero request errors. No-call responses fail individual decisions. The base is the reference, rather than a candidate with an independently assigned adapter verdict. Future reporting must also verify expected response coverage; the harness excludes request-error rows from scored-rate denominators, so a percentage alone is insufficient.

### Early category, proposal validity and limitations

S3's compact locate→edit ordering could cause edits before sufficient investigation. That is a hypothesis to inspect in `early`, using its actual edit/read/test/no-call distribution, not merely the aggregate rate. There are currently no generated `early` responses.

The inspected `kind()` classifier identifies an edit by the presence of `PYEOF` and `write_text(` in a shell command. This is a syntactic heuristic. It does not execute the command, prove the target is eligible repository source, check a unique replacement or validate the resulting diff. A future classified edit rate must therefore not be presented as an executed source-edit success rate. `write_file` has its own tool label and is not automatically counted as an edit by this classifier.

No model-side tool failures, missing-edit rates, batching frequencies or gate infrastructure failures have been measured in this workflow. The gate's GPU request path remains untested. If infrastructure fails after a future authorized push, preserve outputs and request authorization before modifying `gate.py` or resubmitting, as the task requires.

## 7. CPU E1 Replay Audit

**Current verdict: PASS, 147/147 episodes and 182/182 source edits.** The source-provenance and isolation blockers are resolved; no dataset correction is needed. All edit assertions apply and change the intended file, with final file contents matching the teacher patch blob IDs. There are zero confirmed failures and zero unverifiable retained edits.

The initial static-only check identified 823 paired calls, 182 edits and one ambiguous empty-old operation in reconstructed display text. It did not constitute replay. The subsequent exact-commit source recovery and isolated filesystem audit below supersede its incomplete status. In particular, the empty-file case is now verified against the actual zero-byte file.

Source-edit replay does not verify tests, issue resolution or model behaviour. The compactor's synthesized test `status=ok` / exit-0 fields and tailed pipelines remain a distinct feedback-fidelity limitation, not a demonstrated source-edit failure.

### 7.1 Provenance recovery and completed CPU edit replay

**Result: PASS, 147/147 episodes and 182/182 edits; E1's source-edit replay blocker is resolved.** No episodes were removed, and all four handoff hashes remain unchanged.

#### Exact instruction and corrected decision

The [handoff](step0_handoff_20261017.md#atul-e1-cpu-in-parallel-with-the-gpu-steps) labels the work **“Atul: E1 (CPU, in parallel with the GPU steps)”** and says:

> Every kept edit must apply (`s.count(old) == 1`) and change the intended file.

It then requires reporting edits that do not apply and says **“Those get dropped before training.”** The separate [Codex prompt](atul_codex_step1.md), item 5, also says **“In parallel on CPU (no torch, no model)”**. These instructions explicitly authorize parallel execution. They require removal of confirmed failures before training; they do not state a universal rule that every episode's E1 replay must finish before any training can start. The initial universal prerequisite was our interpretation, not Shrey's explicit instruction. This follow-up corrects it without treating static checks as replay.

The distinction no longer determines whether the data can proceed: the full retained source-edit set has now been replayed successfully. Training still needs the independently specified resource approval.

#### Recovery paths examined and authoritative source

| Path | Evidence and outcome |
|---|---|
| Original teacher rows | All 200 preserved rows inspected; explicit `base_commit` is absent. 52 rows contain Git provenance-related commands, but no commits were guessed from those commands or issue numbers. |
| Recovered compact episodes | Exact task-ID join identifies 147 candidates in the 840-trace S3-1k archive; all 147 trajectory IDs match preserved teacher rows. |
| Preprocessing source and metadata | `convert_public.py` identifies the source as `nebius/SWE-rebench-openhands-trajectories`; it saves instance/trajectory IDs but omits benchmark base commits. The compact trace retains that limited source object. |
| Kaggle/local records | Existing training manifests describe tokenized inputs and outputs, not these external task base revisions. No S3 GPU run existed from which additional source state could be recovered. |
| Recovery checkout / verified backup | Read-only filename inventory and the October 16 backup manifest found no S3/rebench provenance manifest; the backup predates Step 0. Neither checkout nor backup files were modified. |
| Public benchmark metadata | The upstream trajectory card identifies SWE-rebench as its task source. An exact instance-ID/repository join against the benchmark recovered **147/147 base commits**. |

Authoritative sources: [Nebius trajectory dataset](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories), [pinned benchmark metadata](https://huggingface.co/datasets/nebius/SWE-rebench/tree/89cdfbab4ab1bd8f5a658bb212d1b63624f4f881), and [benchmark schema](https://huggingface.co/datasets/nebius/SWE-rebench/blob/89cdfbab4ab1bd8f5a658bb212d1b63624f4f881/README.md). The schema defines `base_commit` as the source HEAD before the solution PR; `environment_setup_commit` is a different field and was not substituted.

The dataset-viewer filter route returned HTTP 500. Recovery therefore used pinned Parquet HTTP range reads of only `instance_id`, `repo`, and `base_commit`, not the 245 MB whole benchmark or 2 GB trajectories export. The metadata projection transferred **1,643,764 bytes** across the two test shards and recovered all 147 tasks without ambiguity. Each repository identity was checked against its preserved teacher row.

Only files targeted by retained edits were fetched from `raw.githubusercontent.com/<repo>/<exact-base-commit>/<path>`. All **150 distinct repo/commit/path downloads** succeeded, totaling **2,645,508 source bytes**. There are **151 episode/file states** because one pinned file is shared by more than one episode. No repository clone, Docker image, model download, setup script or test suite was needed.

#### Replay method and results

A disposable bubblewrap namespace was verified using an audit-owned filesystem check outside the outer tool sandbox. The replay used a read-only Python runtime, read-only source/plan inputs, a temporary writable filesystem, a designated audit-output directory, and **no network**. No teacher shell or Python program was executed. No project setup, import or test code was executed.

Before replay, each generated command's Python AST had to equal the canonical conversion algorithm: import `pathlib`, bind a literal target path, `read_text`, bind literal `old`/`new`, assert exactly one occurrence, and `write_text(s.replace(old, new))`. Only safe relative paths were accepted. The audit-owned implementation then performed that same operation on actual temporary files initialized from the pinned source bytes. It checked mutation, exact expected output bytes, target-file containment, recorded edit feedback and final Git blob identity. This is filesystem edit replay with an equivalent validated operation, not host execution of teacher code or a full software-test evaluation.

| Required measurement | Result |
|---|---:|
| Episodes examined | 147 |
| Episodes with verified source revision and all targeted files | 147/147 |
| Episodes whose complete edit chains replayed | 147/147 |
| Edits successfully applied and changed intended source | 182/182 |
| Confirmed edit failures | 0 |
| Edits unverifiable due to missing provenance/source | 0 |
| Initial episode/file Git blobs matching teacher patch index | 151/151 |
| Final episode/file Git blobs matching teacher patch index | 151/151 |
| Mismatched source-edit success feedback | 0/182 |
| Undeclared files materialized/modified in replay trees | 0 |

The teacher patch supplies abbreviated Git index IDs; the downloaded source and replayed output were hashed with Git's complete blob algorithm and required to match those prefixes. Exact commit-addressed retrieval and agreement with both patch endpoints corroborate the source association. Full SHA-256 file hashes and full computed Git blob IDs are retained in local evidence. No merely similar source file was accepted.

#### Empty-file case — resolved at byte level

For `ithaka__apiron-40`, base commit **`a75eb697f27dcabaa7366f7859f192ad5b387b4f`**, target `apiron/__init__.py` is **exactly zero bytes**. Its SHA-256 is `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`, and Git blob ID is `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391`, matching teacher index `e69de29`.

For the actual empty string, `s.count("") == 1`; the converted call therefore passes its assertion. The replacement creates 599 characters, changes the correct file, and yields Git blob `8842928fe5733279125ceee6243370b70e4a41b7`, matching teacher final index `8842928`. Output SHA-256 is `3b5dc6ccb0c07a315d81c434c171c9e6dc93d91442a14c497775dd0d1cd7abe4`. The previous one-newline retained read was a display reconstruction artifact, not the original source bytes. There is no confirmed conversion failure here.

#### Scope and remaining limitations

E1 now verifies the retained source-edit transitions and their final source-file states. It does not verify teacher test execution, test success, issue resolution or all removed intermediate actions. The earlier observation about synthesized test/pipeline success fields remains a separate data-fidelity concern; zero source-edit feedback mismatches does not clear every kind of feedback. Matching candidate IDs plus source/final blob identities still does not claim a fresh tokenizer-level E0 reproduction of the missing original compact directory.

No data correction is justified by these replay results. No NPZ, original archive, converter, compactor or generated notebook was hand-edited. No GPU, torch or model inference was used for E1. The audit's PyArrow reader was installed only under `/tmp` for metadata access; it is not a training/runtime change.

The small [E1 summary and complete per-episode source manifest](../data/runs/atul_e1_replay_20261018/summary.json) records all 147 instance/repository/base-commit mappings, source paths, initial/final Git blob hashes and results. Full local replay evidence is `data/runs/atul_e1_replay_20261018/e1_replay_report.json`; raw downloaded source files and detailed inputs remain uncommitted. Audit-owned utility copies and SHA-256 values are preserved locally under that folder's `audit_tools/`.

## 8. GPU Runtime and Quota Accounting

Fresh authenticated SDK snapshot: **October 18, 2026, 01:25:27 IST** (`2026-10-17T19:55:27.112455+00:00`). API GPU quota: total 30.00 h, used 0.00 h, reserved 0.00 h. Available quota after subtracting used and reserved is 30.00 h.

The CLI's authenticated `--mine` inventory returned five notebooks in Atul's namespace. Status queries returned COMPLETE for `gemma4-train-smoke`, `check-l4` and `gemma4-train-lora`. The two older unrelated notebooks returned HTTP 404 for session status; these are not represented as COMPLETE. The zero-reservation observation supports the absence of a currently reserved GPU session. This snapshot must be refreshed before any future push.

| Run | Handoff expected session h | Handoff expected quota h | Actual new session/quota h | Before / after available quota | Observed charge ratio |
|---|---:|---:|---|---|---|
| S3 | ~1.4 | ~3.0 | 0 / 0; not submitted | No run-boundary pair | N/A |
| S1 | ~1.7 | ~3.5 | 0 / 0; not submitted | No run-boundary pair | N/A |
| Gate | ~0.6 | ~1.2 | 0 / 0; not submitted | No run-boundary pair | N/A |
| Total | ~3.7 | ~7.7 (rounded to ~8 in handoff) | **0 / 0** | Single snapshot: **30.00 h available** | N/A |

There is no post-run snapshot because there was no run. The 2× billing factor is a historical planning estimate, not a new measured ratio. Historical Step C was 7.8203 session-hours / 15.64 quota-hours; Step D was 4.4916 / 8.9942, approximately 2.0024×.

### Runtime estimate requiring review before resumption

The prescribed two epochs expose approximately 1.410M S3 training input tokens and 1.867M S1 training input tokens before S1's two omitted presentations. A simple scenario using 280 tokens/s and the previously measured 37-minute load gives:

| Run | Approximate training-input time | Plus 37-minute load | Approximate quota at 2× |
|---|---:|---:|---:|
| S3 | 1.40 h | 2.02 session h | 4.03 h |
| S1 | 1.85 h | 2.47 session h | 4.94 h |
| Gate, handoff estimate | — | 0.60 session h | 1.20 h |
| Combined scenario | — | 5.09 session h | 10.17 h |

These are **forecasts**, not measured runtimes or replacements for the prescribed configuration. They omit validation, checkpoint/export and setup overhead; shorter windows may run faster than 280 tokens/s. They demonstrate that the handoff's ~8 quota-hour total is uncertain for two epochs and repeated model loading. Both the >2-session-hour and >10-cumulative-quota-hour approval rules need review before resumption. A `--budget-h 3` argument is not itself permission for a longer run or a guarantee of billing termination.

The quota ledger in [the guide, §8](post_training_guide.md#8-gpu-quota-ledger) is updated separately with this stopped preflight, zero new consumption and the estimate discrepancy.

### 8.1 GPU preflight and one-run approval

A separate **primary-workspace `.venv`** was created with `include-system-site-packages=false`. It contains Kaggle CLI **2.2.4** and its dependency graph, installed through `pip install kaggle`; no training/model framework was installed there. Authentication was verified through the existing authorized login flow. No Kaggle credentials were copied, printed or inspected. The recovery checkout's virtual environment is no longer needed for handoff commands.

Fresh authenticated inventory/quota snapshot: **October 18, 2026, 02:08:10 IST** (`2026-10-17T20:38:10.088452+00:00`). Total GPU quota **30.00 h**, used **0.00 h**, reserved **0.00 h**, available **30.00 h**. The inventory belongs to `atulbaliyan`; the three project notebooks are COMPLETE and the two older unrelated notebook sessions return 404. No queued/running notebook was returned, and no S3/S1 training notebook existed in the inventory. Refresh these checks again immediately before an approved push.

The exact S3 builder command in §3 was executed locally with `KAGGLE_OWNER=atulbaliyan`. The builder compiled all code cells and generated **37 cells / 459,501 normalized source bytes**, below the 1,000,000-byte limit. Metadata was verified:

- Identity: `atulbaliyan/gemma4-train-s3-compact`; private; GPU enabled; Internet disabled; `machine_shape=NvidiaL4`.
- Data embedded from the unchanged S3 NPZ; no newer S3-1k data substituted.
- Model: `google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2`.
- Inputs: official metric wheelhouse, competition, and Shrey's scorer-env/train-wheels outputs.
- Settings: `name=s3_compact`, `lr=5e-5`, `epochs=2`, `grad_accum=2`, `save_every=9`, `val_every=0`, `budget_h=3`, `smoke=false`, no max-windows cap.

Read-only API checks at 02:11:05 IST confirmed access to both shared kernel outputs (`scorer_env.tgz` and the two training wheels) and the official wheelhouse (41 listed files, no further page). The generated notebook is 467,033 bytes on disk, SHA-256 `b93e23c6065f61c6ef6e312a0ef002ba4065950499860e3d944f44a1f79a669a`; its normalized source is 459,501 bytes. Both are below 1 MB.

The generated notebook remains a local, uncommitted build artifact, with a copy archived under `data/runs/atul_s3_compact/submitted_notebook_v1/`. Exactly one subsequent push was accepted as version 1. S1 and gate require separate user approval as well as preceding-stage checks.

The exact deterministic schedule was recalculated without importing torch:

| Measurement | S3 | S1 |
|---|---:|---:|
| Scheduled training input tokens | 1,409,618 | 1,855,104 |
| Window presentations | 280 | 264 |
| Optimizer steps | 35 | 33 |
| Omitted training presentations | 0 | 2 |
| Declared validation windows | 7 | 21 |
| Actually scheduled validation windows per pass | 4 | 20 |
| Scheduled validation input tokens per pass | 25,076 | 143,444 |

Validation uses the same group-of-four scheduler: S3's validation loss will cover **4 of 7** held-out tasks/windows, not all 7; S1 uses 20 of 21 held-out windows. This is a source-derived limitation of the unchanged configuration, not a measured new loss or a code fix. It must accompany future loss reporting.

Step C's actual model event records **2,243 s load** and `t_min=43.9`, which includes startup before that event; its initial validation completed at 48.7 min. Using actual scheduled training tokens and **280 tokens/s**, adding 43.9 min through model initialization gives approximately **2.13 session h S3** and **2.57 session h S1**, before their own validation/export overhead. Using Step C's raw median 294.8 tokens/s gives about 2.06 and 2.48 h respectively. These are estimates, not measurements of the new runs; sequence-length efficiency can change them.

For authorization planning, use **S3 ~2.1–2.4 session h / ~4.2–4.8 quota h**, **S1 ~2.6–2.9 session h / ~5.2–5.8 quota h**, and the handoff gate estimate **0.6 session h / ~1.2 quota h**: total approximately **10.6–11.8 quota h**. Setup, validation/export and Kaggle accounting remain uncertain. S1 may approach the existing 3 h budget's stop-taking-steps threshold at 2.6 h; do not change its flags silently or call a budget-stopped run complete training.

**Approval received for Step 1a only:** the user explicitly approved exactly one S3 run with the unchanged 3 h budget and exact handoff parameters. Expected duration remains below that budget. The user requires fresh preflight, one submission, no automatic retry, and separate approval for S1 and the gate. This is not approval for the cumulative >10 h plan. At pre-push time no quota has yet been spent; record actual run usage from fresh before/after snapshots.

### 8.2 Approved S3 version 1 submission and monitoring

Before push, fresh authenticated quota at **02:20:31 IST** was **30.00 h available, 0 used, 0 reserved**, with no queued/running notebook. All four hashes were rechecked. AST compilation, exact settings, metadata inputs/namespace and embedded data identity passed. The normalized notebook source is **459,501 bytes**, on-disk notebook **467,033 bytes**, unchanged SHA-256 `b93e23c6065f61c6ef6e312a0ef002ba4065950499860e3d944f44a1f79a669a`.

The user explicitly approved **exactly one** S3 run with the unchanged 3 h budget. Expected duration remained **2.1–2.4 session h / 4.2–4.8 quota h**, below the approved session budget; the quota ledger was updated **before** push. The command `.venv/bin/kaggle kernels push -p kaggle/train/l4_s3_compact` returned **“Kernel version 1 successfully pushed.”** The initial status query returned **RUNNING**. No second push was issued.

Run: [atulbaliyan/gemma4-train-s3-compact](https://www.kaggle.com/code/atulbaliyan/gemma4-train-s3-compact). Local `summary.json` and the exclusive-create `preflight_20261018/push_attempt.json` record the sole attempt and version. The ten-minute thread monitor `monitor-approved-s3-compact-v1` is ACTIVE to read status/quota and recover terminal outputs. Unchanged/non-actionable states do not trigger repeated notifications. Completion, failure, budget concerns or needed user actions will be reported.

Actual loss, checkpoint/export integrity, memory, runtime and billed quota remain **PENDING**, not zero. Successful push/RUNNING does not establish training success. On terminal state the monitor must download and verify outputs, update this same report and the quota ledger, then pause. S1, the behaviour gate, retries and configuration changes are explicitly not authorized.

## 9. Major Findings and Technical Analysis

1. **Input identity passed:** all four handoff hashes remain unchanged, including the embedded S3 data. No S3-1k substitution or data regeneration occurred.
2. **E1 passed:** 147 episodes and 182 edits replayed at authoritative source revisions, with 151/151 initial and final episode/file blob matches. The prior missing-provenance issue and empty-file ambiguity are resolved.
3. **Scope is precise:** zero retained source-edit feedback mismatches does not validate synthesized test outcomes, whole-episode execution or issue-solving behaviour.
4. **Training plan is exact:** 35 planned S3 steps and 33 S1 steps follow from the unchanged two-epoch scheduler. Validation covers 4/7 S3 and 20/21 S1 windows; report this when interpreting eventual loss.
5. **Authorization is narrow:** exactly one S3 run is approved, with no S1, gate, retry or configuration change. Forecast S3 time is below the approved 3 h budget.
6. **Loss and behaviour remain distinct:** no new model behaviour can be inferred from E1 or from future loss alone. Step C/D remains the historical counterexample.

No paired significance result is reported without new paired outcomes.

## 10. Recommendations and Future Research

Proceed with the one user-approved S3 run after recording the fresh preflight and expected charge. Stop after its output verification and reporting; S1 and the behaviour gate need separate approval.

| Next action | Evidence required / current status | GPU cost |
|---|---|---|
| Provenance recovery and isolated E1 | COMPLETE: 147/147 episodes, 182/182 edits; no correction | 0 GPU h |
| Approved S3 training, one push | Exact unchanged notebook; fresh no-conflict/quota check; COMPLETE/rc/loss/export review | Expected 4.2–4.8 quota h, not measured yet |
| S1 training | No authorization; request separately after S3 review | Forecast only, no spending |
| Behaviour gate | No authorization; request separately after required training passes | Forecast only, no spending |

Dataset improvements should follow measured E1 defects: preserve exact original success/error outcomes, represent empty-file operations correctly, and retain enough source state to make edits executable. Do not change the current NPZ merely because a code path looks suspicious. Compact episodes also need recovery examples and investigation state appropriate to early decisions; these are future design hypotheses.

Keep the prescribed rank and objective initially. First establish faithful data, a valid behavioural measurement and authorized runtime. Neither lower validation loss nor a syntactic edit label warrants higher-dose retraining, mixed S4 training, DPO/RL or S3-1k scale-up. Any later task evaluation, retry or scale-up needs its own authorization; both adapters failing the gate would trigger the existing stop rule.

## 11. Final Conclusion

E1 source-edit verification is complete and successful: 147/147 episodes, 182/182 edits, matching initial/final Git blobs for all 151 episode/file states, and no data corrections. Missing provenance and isolation are resolved. The empty-file operation is valid. This clears the source-edit audit concern; it does not demonstrate model behaviour or validate teacher test results.

Exactly one S3 training run was submitted successfully as version 1 and is initially RUNNING with unchanged handoff parameters and `--budget-h 3`. Ten-minute monitoring and terminal-output recovery are active. Its actual loss, exports, runtime and quota usage remain pending and must be reported from downloaded evidence, without extrapolating behavioural success. S1 and the behaviour gate remain unapproved. No automatic retry or configuration change is allowed.

### Evidence and delivery record

- Inspected source baseline: `fc29e53` on `main`, matching freshly fetched `origin/main` before the results branch was created.
- Authorized branch: `atul/s3-s1-gate-results`.
- Local supplemental preflight evidence, not experiment outputs: `data/runs/atul_s3_s1_gate_preflight_20261018/kaggle_preflight.json` and `static_preflight.json`.
- Historical measured loss/export source: `data/runs/atul_step_c_corrected_20261015_v2/train_report.json` and `train_out/metrics.jsonl`; historical runtime interpretation from its report/guide and Step D's final report.
- Static source checks: `posttrain/compact_episodes.py`, `posttrain/convert_public.py`, `kaggle/train/trainkit/{config,data,worker,main}.py`, and `kaggle/eval_l4/evalkit/gate.py`.
- No credentials were inspected, copied or exposed; only the existing CLI/SDK authentication flow was used. Its preserved installation was invoked with Python bytecode writes disabled.
- Recovery working files and Kaggle artifacts were preserved. Git uses administrative metadata under the recovery checkout's `.git/worktrees/` for this linked worktree; the authorized pull and feature-branch operations necessarily updated that shared metadata.
- Exactly one approved S3 notebook was pushed as version 1. No teacher-command execution, dataset regeneration, paired task evaluation, leaderboard submission, S1/gate run or automatic next-stage experiment occurred.
