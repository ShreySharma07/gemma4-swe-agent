# Dataset and training-policy audit

## What is independently checked

The stdlib-only [validator](validate.py) reads NPZ headers and arrays without NumPy, torch or a model. It verifies CRCs, dimensions, token/mask lengths, binary masks, monotone nonempty windows ≤8,192, positive shifted-label counts and task-disjoint internal train/validation membership. All four preserved datasets pass; hashes match the historical handoffs. This verifies structure, not semantic quality or absence of benchmark contamination.

| Dataset | Distinct tasks | Windows | Input tokens | Masked target tokens | Max length | Internal train/val tasks |
|---|---:|---:|---:|---:|---:|---:|
| Step C public | 200 | 1,017 | 7,372,758 | 1,170,308 | 8,191 | 190 / 10 |
| S1 own | 21 | 154 | 1,083,946 | 192,809 | 8,189 | 18 / 3 |
| S3 compact | 147 | 147 | 744,278 | 93,746 | 8,171 | 140 / 7 |
| S3b | 826 | 826 | 5,823,825 | 632,182 | 8,192 | 785 / 41 |

These are complete attached datasets, not actual scheduled exposure. Step C processed 972/974 training windows and 40/43 validation windows. S3 trained two epochs and validated 4/7 windows; S1 trained two epochs and validated 20/21. S3b used only 400 shuffled training episodes for one epoch, not all 826. The source scheduler drops incomplete groups of four. A dataset-level percentage alone cannot reconstruct the exact selected optimizer dose. Sources: [C report](../step_c_20261015_report.md), [S1/S3 report](../atul_step1a_1b_step2_gate_report_20261018.md), `kaggle/train/trainkit/data.py`.

## Actions and token targets reward different things

Preserved token-kind figures are from `data/runs/audit_e2*/e2_report.json`, whose tokenizer-level audit is already recorded. This task did not fetch a tokenizer or rerun tokenization. New measurements below are command-classification counts from archives and must not be silently substituted for E2's supervised-span definitions.

| Property | Step C | S1 | S3 | S3b |
|---|---|---|---|---|
| Training provenance | 200 converted successful public teacher runs | 82 successful v7 trajectories from just 21 tasks | 147 compact public runs | 826 compact public episodes; 400 used in training |
| Mid-trajectory windows | 817/1,017 | Multi-window; exact phase share not remeasured here | 0/147 | 0/826 |
| Source-edit supervision | 396 actions; 16.8% token share | No preserved per-kind token audit found; do not infer from total mask | 182 edits; 75.0% token share | 1,020 edits; 63.1% token share |
| Scratch writes | 735 write_file actions; 56.4% token share, mostly scratch/test/verification files | Own successes still contain repetition and scratch work | Removed by compaction | Removed by compaction |
| Pytest targets | 1,051 actions; 3.1% tokens | Exact comparable action/token share unavailable | 127 actions; 6.2% tokens | 1,345 actions; 6.6% tokens |
| Pytest first supervised action | 120/1,017 windows | Not independently remeasured | 0/147 | **371/826 (44.9%)** |
| Read→edit visibility | 311/396 snippet edits lacked earlier file visibility | Own provenance is not a grounding proof | 0/182 flagged unseen; E1 replay passes | 3/1,020 flagged unseen; regex limited to Python |
| Main downstream evidence | Step D 0/114, all patches empty | Fixed gate below base | Gate v1 below base; v2 different rubric/outcome | Gate v2 passed; real tasks 9/60 versus v7 23/60 |

**E2 denominator correction:** the preserved Step C JSON prints `311/1131` under a field named snippet edits. Its source increments the denominator for both 396 snippets and 735 write_file actions (396+735=1,131), but increments the numerator only for snippets. The relevant snippet conditional is 311/396=78.5%, as the existing prose report states. Do not interpret 311/1,131 as a snippet-grounding rate. Files remain unchanged. See `posttrain/audit_pipeline.py`, `edits_total`/`edits_unseen` handling.

The loss also matters. `MaskedCE.forward` in `kaggle/train/trainkit/model.py` normalizes by supervised labels **within each window**; worker accumulation averages window losses across equally scheduled ranks. Validation instead accumulates token-weighted loss. Therefore summed token shares are useful descriptive exposure, but they are not exact effective gradient shares. Short windows and repeated tasks can receive disproportionate weight. A future audit must report both target-token share and each action's sum of per-window normalized target weights, plus actual selected/scheduled windows.

## New independent archive audit

[audit.py](audit.py) reads archives in memory and never extracts or executes teacher commands. It associates responses by explicit call ID when available; v14's ID-less observations use a disclosed ordered same-tool fallback. Missing observations stay unknown. It distinguishes actual pytest invocation patterns from a grep command merely mentioning pytest. Mutation detection is intent, with literal Python target paths recovered by AST parsing where possible.

- Step C archive: 200 episodes, 7,161 tool-call proposals, 177 repository keys inferred from filenames, no exact canonical duplicate episodes under this audit's hash.
- S3b archive: 826 episodes, 7,540 tool-call proposals, 564 repository keys inferred from filenames, no exact canonical duplicate episodes under this audit's hash.
- S3b executable first-action classification: 382 pytest, 370 search, 74 read (46.2% test-first). E2's 371/826 is the supervised-window measure; differing command parsing/rendering explains that these definitions need not match. A rigorous rendered-span reconciliation would require the exact tokenizer/windows, not an asserted equivalence.
- New S3b transition counts include 275 recognized post-mutation passing tests→submit, 414 post-mutation tests with unknown outcome→submit, and one recognized failed test→submit. Nested nonzero-exit responses are parsed. Two post-mutation unknown-outcome tests→mutation cases cannot be called verified failure recovery.
- Many successful commands have no pytest summary, and pipe/tail or collection failures complicate interpretation. Unknown is retained rather than relabelled pass. These are adjacent-call transitions, not end-to-end verified resolution labels.

The complete new action/transition counts, definitions and repository-key distributions are in [dataset_measurements.json](dataset_measurements.json). Existing S3b handoff reports two failed-post-edit-test→edit cases and one→submit; its broader `status != ok` failure definition differs from this audit's explicit assertion-failure classification. Neither definition demonstrates substantial verified repair-after-failure coverage.

## Why the failures are consistent with these data

**Step C:** data provenance and mask boundaries were reproduced (E0/E2), so simplistic mask corruption is not the supported explanation. The observed missing state and scratch-code target dominance provide a concrete policy-mismatch mechanism. Repeated pytest and batching at inference were not literally copied as identical batched training calls; the downstream mechanism remains unisolated.

**S3:** grounding fixed an important defect, but compression produced a narrow search/read/edit/test/submit template. S3's gate v1 submitted after failed feedback 28/32 and edited at early prefixes 10/32. Correct source bytes do not teach when to edit or whether a test result permits submission. E1's 182/182 source-edit replay proves applicability at pinned bases, not teacher tests in the target environment.

**S1:** 82 solved trajectories represent only 21 tasks; nine tasks supply most of the successes in the original repetition study. It lowers validation loss while failing the gate. Own-policy style alone is insufficient at this observed diversity/dose; it does not disprove all self-distillation.

**S3b:** additional reads and authentic test feedback corrected the one-step gate, but nearly half of episodes still open with testing and true post-edit failed-test recovery is rare. Training with 63% edit target tokens can still increase the probability of a short test call at the wrong time. Full rollouts then accumulate tests and context instead of edits: 9/60 solves, 76.7% empty patches, 47.4% repeated calls and 23.3% overflow in the preserved run summaries. These are associations consistent with decision-state imbalance, not a controlled causal decomposition.

**Gate mismatch:** `kaggle/eval_l4/evalkit/gate.py` sends `enable_thinking=False`; `posttrain/render_atif.py` sets `THINKING=False`. The fixed gate does not reproduce v14's thinking-enabled trajectory, and samples from 30 prefixes are not 120 independent tasks. Version 2 also changes scoring (including `cat -n`), so it does not erase version 1's result or establish comparable causal improvement.

## Provenance, diversity and licensing limits

`posttrain/convert_public.py` names `nebius/SWE-rebench-openhands-trajectories` and records CC-BY-4.0; its held-out repository filter excludes FastAPI, rich, requests and httpx. This is source-declared licensing, not a freshly verified upstream legal record. Pin dataset revision, row ID, original license text/attribution, teacher/provider terms, repository license and base commit before any reuse. An OSI license on our code does not replace those obligations.

Repository counts above are inferred filename keys, not a complete repository/snapshot join. Exact duplicates absent under one canonicalization do not rule out issue paraphrases, related fixes, fork leakage, repeated base commits or near-duplicate command templates. Languages, dependency depth and issue/patch complexity cannot be fully certified from masked token arrays; Python-oriented conversion is clear, but multilingual generalization is unmeasured. E1 recovers exact commits for the 147 S3 examples; S3b's full commit/license/test-environment provenance is not established by E1.

The old-54 own runs, fixed gate prefixes and new-60 diagnostic tasks are development evidence. They must all be excluded from new independent final comparisons, with related commits and forks grouped. Internal task-disjoint validation in an NPZ is necessary but does not establish repository-disjoint generalization or freedom from model pretraining exposure.

## Conditions before another adapter

Require authentic state-conditioned recovery coverage, exact source-state grounding, raw outcome retention, comparable environment verification, explicit effective-loss exposure, a thinking-compatible rendering check and a full-task smoke gate. Missing any of these is a data-acceptance failure, not a reason to scale the previous corpus. A single failed adapter does not prove post-training cannot work; the current evidence rejects repeating these recipes unchanged.
