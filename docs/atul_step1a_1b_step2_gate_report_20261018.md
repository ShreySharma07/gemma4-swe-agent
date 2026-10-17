# S3 Compact, S1 Own and behaviour gate — consolidated technical report

Date: October 18, 2026 (IST). Author: Atul's execution workspace.

**Status: stopped during preflight; no GPU notebook submitted.** This is the single report for Steps 1a, 1b, 2 and CPU E1. It records completed checks and the blockers, rather than reporting unexecuted experiments as successes or failures. Update this same file when execution is authorized to resume.

## 1. Executive Summary

The experiment objective is to test whether grounded, compact public demonstrations (S3) or the agent's own successful trajectories (S1) restore useful edit behaviour after Step C's adapter produced 114 empty patches in Step D. The intended sequence is S3 training, review, S1 training, review, then a fixed-prefix behaviour gate comparing base, S1 and S3.

The reporting correction supersedes the original request for three separate reports. No separate Step 1a, Step 1b or Step 2 report was created.

| Stage | Completed work | Outcome |
|---|---|---|
| Workspace/Git | Clean `main` fast-forwarded from `b90af48` to `fc29e53`; authorized feature branch created | PASS |
| Artifact identity | Both training NPZs, gate prefixes and raw teacher-row archive match handoff hashes | PASS |
| Kaggle access | Authenticated inventory and fresh accelerator-quota query | PASS; 30.00 quota-hours available, 0 reserved |
| CPU E1 | Located 147 candidate episodes in the new S3-1k archive; matched all task and trajectory IDs; inspected action/feedback structure and literal edits | PARTIAL; no real-repository replay completed |
| Step 1a S3 | Dataset and prescribed configuration inspected | NOT STARTED; E1 prerequisites unresolved |
| Step 1b S1 | Dataset and prescribed configuration inspected | NOT STARTED; predecessor not completed |
| Step 2 gate | Prefix counts and harness scoring inspected | NOT STARTED; no trained candidate outputs |

The principal blocker is missing exact repository base commits for E1: none of the 200 preserved teacher rows or 147 matched compact episodes contains `base_commit`. A task/issue ID and final patch do not establish the source tree on which the converted edits must execute. No suitable disposable execution environment was established either: Docker and Podman were unavailable on PATH, and a basic bubblewrap isolation check failed under the current sandbox.

Measured static findings: 147 candidate episodes cover 147 tasks and 132 repositories, with 182 edit calls and 823 total calls. All 823 calls have associated responses, with no duplicate response IDs. For 181 edits, the decoded old literal occurs exactly once in the retained visible excerpt after preceding simulated replacements. One empty-old-string edit needs exact file bytes to assess. These measurements do **not** establish successful filesystem transitions, correct final patches or truthful teacher feedback.

New workflow GPU consumption is **0 session-hours / 0 quota-hours**. No new loss curves, adapter exports, gate verdicts or performance measurements exist.

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
| Converted success feedback corresponds to executable source changes | E1 at exact repository commits | Blocked |

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

The commands above are the handoff commands, **not a record of execution**. The expected notebook identity is `atulbaliyan/gemma4-train-s3-compact`; no version was submitted or assigned by this workflow.

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

No S3 `train_report.json`, `train_out/metrics.jsonl` or `run.log` exists in the prescribed `data/runs/atul_s3_compact/` output location. No notebook was built or pushed after the blocker was established. Consequently:

- Train/validation loss and loss curve: unavailable.
- GPU memory, utilization, throughput, actual runtime and quota charge: unavailable; no run.
- Adapter tensors, hashes, shapes and serving-copy equality: unavailable; no export.
- Runtime warnings, failures or checkpoint interruption: no execution evidence.

**Verdict: NOT STARTED / BLOCKED, not training FAIL.** Required future checks remain rc 0, decreasing validation loss and a valid final 410-module export; integrity must include tensor shapes/dtype and file completeness. Lower loss alone cannot demonstrate behavioural improvement.

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

**Verdict: NOT STARTED.** It cannot proceed until S3 is COMPLETE and its required checks are reviewed. The small set of distinct tasks limits generalization claims even if future training loss and gate results are favorable.

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

### Recovery and provenance checks

The handoff expects 147 traces in `data/runs/s3_compact/`; only `compact_report.json` was present there. The read-only recovery checkout had no corresponding S3 directory. The newly pulled `data/runs/s3_1k_episodes_20261018.tar.gz` contains 840 trace JSONs plus a summary. macOS `._` metadata entries were excluded from inspection.

Filtering the archive by exact task IDs decoded from the original S3 NPZ found exactly 147 episodes, covering every S3 task once. All 147 trajectory IDs also match the preserved 200 teacher rows. This supports candidate episode recovery without modifying or regenerating the NPZ. It does **not** prove byte-for-byte identity with the missing original compact traces or re-establish tokenizer-level E0 for this subset.

Raw row archive SHA-256: `b0ceaac824c543d8a417b7a4e45bbd5c1d62bf8791c5eb1047942100afdf8607` (matches handoff).

### Completed static checks — not actual replays

| Check | Measured result | Interpretation |
|---|---:|---|
| Matched episodes / tasks / repositories | 147 / 147 / 132 | Candidate coverage established |
| Matching raw trajectory and instance IDs | 147/147 | Row association established |
| Total calls | 823 | Retained action count |
| Calls missing a response | 0/823 | Association exists; truthfulness untested |
| Duplicate response IDs | 0 | No duplicate IDs observed within inspected episodes |
| Edit calls parsed for literal old/new/path | 182/182 | No teacher code executed |
| Old/new literal differs | 182/182 | Intended textual mutation, not proof of applied mutation |
| Unique old literal in simulated visible state | 181/182 | Local excerpt precondition satisfied |
| Empty-old special case | 1/182 | Requires exact source bytes |
| Raw rows containing `base_commit` text | 0/200 | No explicit pinned base found |
| Matched episodes containing `base_commit` text | 0/147 | No explicit pinned base found |
| Actual repository-source transitions replayed | 0 | E1 incomplete |
| Successful / failed real replays | N/A / N/A | Neither outcome measured |
| Final source state compared with teacher patch | 0 episodes | Not established |

The static utility decoded literals with `ast.literal_eval` and applied string replacements only to retained observation text in memory. It did not evaluate teacher Python, invoke teacher shell commands, run tests or mutate a source checkout.

The special case is `ithaka__apiron-40`, `call_4`, target `apiron/__init__.py`. Its `old` literal is empty and the command asserts `s.count(old) == 1`. The reconstructed retained read is one newline character, so its empty-string count is 2. A genuinely empty file has empty-string count 1. Because the compactor appends a newline when reconstructing reads, this is **not a confirmed failed edit**. It demonstrates why excerpt simulation cannot replace real-file replay. The associated retained feedback reports `status=ok`, exit 0, but that report has not been validated against the real source state.

### Blockers and technical implications

Neither raw rows nor compact episodes supply an explicit base commit. No verified mapping or corresponding pinned source snapshots was established in the new worktree, recovery checkout or inspected desktop backup inventory. This is a local availability finding, not a claim that the upstream dataset cannot provide the commits. Do not infer a base from current repository HEAD or from an issue number.

Docker and Podman were unavailable on PATH. Bubblewrap exists, but a basic isolated `/bin/true` invocation failed while creating its network namespace (`NETLINK_ROUTE`, operation not permitted). No elevated isolation attempt was needed once base provenance was missing. The task requires disposable isolation before any untrusted trajectory execution; that requirement remains unresolved.

Source inspection also shows the compactor synthesizes a retained test response with status `ok` and exit code 0 after appending `2>&1 | tail -n 30`. That is not proof that the original pytest passed: pipeline exit status and test outcome can differ, and original tail text may describe failure. This is an observed construction rule, not a measured count of false feedback in the actual training corpus. The converter's old/new replacement commands similarly require E1 to verify their preconditions in the correct source state.

No episodes were removed and no dataset was regenerated. Missing provenance, the empty-file case, actual edit applicability, expected diffs and final-state equivalence remain open. The instruction that failed E1 episodes be dropped before training makes the audit a prerequisite for an approved dataset; changing it would require reconciling a new hash with the handoff.

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

## 9. Major Findings and Technical Analysis

1. **Artifact identity passed.** All four handoff hashes match. This prevents silent substitution with the newer S3-1k dataset, which remains outside the authorized training scope.
2. **The missing original episode directory is partly recoverable.** The new archive contains matching task/trajectory candidates for all 147 S3 tasks. Exact historical rendered/tokenized identity remains distinct from ID-level recovery.
3. **E1 has not passed.** Response IDs and visible snippets are structurally consistent in the measured ways, but the exact base source is missing and no real source edit was executed. Successful replay rate is undefined, not 0% or 100%.
4. **Empty files require byte-aware validation.** The one empty-old edit shows how an added observation newline changes precondition counts. It must not be labeled a real edit failure solely from reconstructed display text.
5. **Feedback construction deserves validation.** Synthesized shell/test success fields and tailed pipelines cannot establish test success. Measure actual mismatches in E1 rather than treating a plausible converter risk as a proven cause of Step D.
6. **Planned updates differ slightly from prose estimates.** Source scheduling gives 35 S3 and 33 S1 steps; S1 loses two window presentations to grouping. Neither training coverage nor loss is inferable from the approximate 36-step handoff wording.
7. **Resources are available but runtime authorization still matters.** The observed quota reset does not waive per-run or cumulative approval thresholds. Token-based estimates need reconciliation before launch.
8. **No behavioural claim is supported yet.** There are no S3/S1 outputs or gate measurements. Step C demonstrates why lower loss, clean format and export integrity alone cannot settle the hypothesis.

No paired significance result is reported: no new task outcomes or paired gate samples exist. Historical Step D remains evidence of the previous adapter's behaviour, not a fresh S3/S1 control experiment.

## 10. Recommendations and Future Research

**No adapter should advance at this point.** Neither new adapter has been trained or behaviourally gated.

| Next action | Evidence required / stop condition | GPU cost |
|---|---|---|
| Recover exact source provenance | A verified instance/trajectory→repository→base-commit mapping or pinned source snapshots for all 147 episodes | 0 GPU h; CPU/network/storage effort unknown |
| Complete isolated E1 | Disposable source trees; every edit precondition, intended source delta, feedback and final diff checked; explicitly resolve empty-file case | 0 GPU h; replay duration unmeasured |
| Reconcile dataset if any replay fails | Quarantine failed episodes; obtain authorization for changed data/hash and rebuild; preserve originals | 0 GPU h for preprocessing |
| Revisit resource estimates and approvals | Agreed bounds for runs expected >2 session h or cumulative >10 quota h; fresh quota/conflict check | 0 GPU h for planning |
| Execute prescribed S3 then S1 once each | Log expected quota first; validate COMPLETE/rc/loss/export before continuing | Handoff 6.5 quota h combined; scenario ~8.97 plus overhead, uncertain |
| Execute gate once after both pass | Correct adapter version/hash, full response coverage and per-type diagnostics | Handoff ~1.2 quota h, unmeasured |

Dataset improvements should follow measured E1 defects: preserve exact original success/error outcomes, represent empty-file operations correctly, and retain enough source state to make edits executable. Do not change the current NPZ merely because a code path looks suspicious. Compact episodes also need recovery examples and investigation state appropriate to early decisions; these are future design hypotheses.

Keep the prescribed rank and objective initially. First establish faithful data, a valid behavioural measurement and authorized runtime. Neither lower validation loss nor a syntactic edit label warrants higher-dose retraining, mixed S4 training, DPO/RL or S3-1k scale-up. Any later task evaluation, retry or scale-up needs its own authorization; both adapters failing the gate would trigger the existing stop rule.

## 11. Final Conclusion

Grounded compact supervision and own-policy demonstrations remain reasonable, untested responses to the measured Step C data defects. This continuation verified input identity and account readiness, recovered matching candidate episodes, and identified the source-provenance gap that prevents a valid E1 replay. It did not test whether S3 or S1 improves the agent.

The behaviour gate **has not run and has not passed**. Prioritize the exact base-commit/source manifest and disposable replay environment, then resolve runtime approvals before spending GPU quota. Continue by updating this same report with actual `train_report.json`, `metrics.jsonl`, `run.log` and `gate_report.json` measurements when those experiments are executed. Preserve this blocked-stage record as part of the engineering history.

### Evidence and delivery record

- Inspected source baseline: `fc29e53` on `main`, matching freshly fetched `origin/main` before the results branch was created.
- Authorized branch: `atul/s3-s1-gate-results`.
- Local supplemental preflight evidence, not experiment outputs: `data/runs/atul_s3_s1_gate_preflight_20261018/kaggle_preflight.json` and `static_preflight.json`.
- Historical measured loss/export source: `data/runs/atul_step_c_corrected_20261015_v2/train_report.json` and `train_out/metrics.jsonl`; historical runtime interpretation from its report/guide and Step D's final report.
- Static source checks: `posttrain/compact_episodes.py`, `posttrain/convert_public.py`, `kaggle/train/trainkit/{config,data,worker,main}.py`, and `kaggle/eval_l4/evalkit/gate.py`.
- No credentials were inspected, copied or exposed; only the existing CLI/SDK authentication flow was used. Its preserved installation was invoked with Python bytecode writes disabled.
- Recovery working files and Kaggle artifacts were preserved. Git uses administrative metadata under the recovery checkout's `.git/worktrees/` for this linked worktree; the authorized pull and feature-branch operations necessarily updated that shared metadata.
- No new notebook push, GPU/model execution, teacher-command execution, dataset regeneration, paired task evaluation, leaderboard submission or automatic next-stage experiment occurred.
