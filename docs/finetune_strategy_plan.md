# Fine-tuning strategy plan (Oct 17)

**Goal:** a LoRA that makes v7 fix more issues on the hidden leaderboard, verified locally before it costs a submission slot. Secondary goal: a clean, checkable result for the paper (experiment freeze Nov 5).

## 1. What we know (measured, not assumed)

| Fact | Evidence |
|---|---|
| The pipeline works: train → export → serve, on 4×L4 | Steps A/B/C/D (`docs/step_c_20261015_report.md`) |
| An adapter trained on **our own v7 runs** keeps the agent editing | Step B: 3/8 non-empty patches, 2/8 solved (tiny, overlapping, but no collapse) |
| An adapter trained on **another agent's runs, converted** collapses editing | Step D: 0/114 solved, 114/114 empty patches, pytest 44% of calls |
| The Step C data taught edits **without the file in view** | E2: 79% of source edits sit in windows where the edited file wasn't shown earlier in that window |
| It mostly taught **scratch-script writing** | E2: 56% of supervised tokens write new tests / repro / verification scripts; fixes are 17% |
| 80% of windows start mid-run with no history | E2: 817 / 1,017 |
| Masks and boundaries are clean | E2: 0 leaked tool responses, 0 broken spans, 0 batched calls in the data |
| Validation loss doesn't predict behaviour | C: 0.77 → 0.29 while editing collapsed |
| Cost | ~280 training tok/s; 4×L4 billed at 2× wall-clock; model load ~37 min. **1M training tokens ≈ 1 h + 0.6 h load ≈ 3.2 quota-h** |
| Budget | 30 quota-h per account per week (Shrey + Atul = 60/week, both reset Saturday 00:00 UTC) |

## 2. Principles (what must change)

1. **Every edit must be trained with the code it edits in view.** No window may supervise an edit unless the file's most recent read is in the same window.
2. **Train the fix, not the scaffolding.** Mask or drop writing new scratch files (repro, test and verification scripts) from the loss. Make sure source edits carry a real share of it.
3. **Keep the decision.** Keep a short reasoning line before key calls, from the teacher's `think` and prose. That's where "now I edit" is decided.
4. **Stay close to our own policy.** Anchor with v7-style data, and use a lower update dose than Step C (122 steps at lr 1e-4 on rank 16).
5. **Gate on behaviour, not loss.** Every adapter must pass a cheap behaviour gate before any task evaluation.
6. **Never evaluate on training tasks.** Train on one task pool, test on a disjoint one.

## 3. Candidate strategies

| | S1: own-run RFT (self-distillation) | S2: fixed public SFT | S3: compact public episodes | S4: mix (S1 + S3) |
|---|---|---|---|---|
| **Data** | v7's own solved runs (82 runs / 21 tasks now; more from new reps) | Same 200 public runs, re-rendered with principles 1–3 | Each public run compressed into **one ≤8k window**: task, the key reads (files later edited), the edit(s), the test, submit. Scratch-file writes dropped | S1 + S3, about 1:1 by supervised tokens |
| **Fixes which defect** | Style mismatch (it is our own style) | State loss, scratch-script dominance, missing decisions | All three, and every edit is grounded by construction | Diversity from S3, style anchor from S1 |
| **Main risk** | Few distinct tasks, so small gains; mostly reinforces what v7 already solves | Still many windows per run, so costly; mid-run windows remain | Compression could drop context the teacher used; needs a careful selector | Two data sources to tune |
| **Training cost** | ~1.3M tokens ≈ **4 quota-h** | ~7M tokens ≈ **16 quota-h** | ~200 × 6k = 1.2M tokens ≈ **4 quota-h** | ~2.5M tokens ≈ **6.5 quota-h** |
| **Clean eval** | Train on the old-54 tasks, test on the new-60 (disjoint) | Any of the 114 (repos disjoint) | Any of the 114 | New-60 |

**Recommendation: S3 first, with S1 run alongside as the control. S4 only if both help.**
- S3 attacks every measured defect for a quarter of S2's cost.
- S1 tells us whether "our own style" alone already helps.
- S2 is dominated by S3: it pays more for the same fixes, keeps the mid-run windows, and doesn't ground edits by construction.

**Dose for all of them:** rank 16; lr 5e-5 (half of C's); 1 epoch; save checkpoints at 25 / 50 / 100% of the run.

## 4. The behaviour gate (new, CPU-built, ~1 quota-h per adapter)

Use fixed decision points ("prefixes") from v7's own traces, chosen before any training. The model gets the prefix and generates one response. The adapter must match or beat base v7:

| Prefix | Pass condition |
|---|---|
| The target file was just read, and its bug region is visible | Next action is a **source edit**, not a test or another read |
| Edit applied, test failed | Next action is a corrected **edit** |
| Edit applied, test passed | Next action is **`submit_patch`** |
| Mid-investigation | Next action is a targeted read or search, not a whole-suite `pytest` |
| Any | Exactly **one** tool call per response (no 50–250-call batches) |

Also: 8 real tasks with **at least as many non-empty patches as v7**. Only adapters that pass go on to a task evaluation.

## 5. Sequence, gates and quota (both accounts reset Saturday Oct 17)

| # | Step | Where | Quota-h | Gate to continue |
|---|---|---|---|---|
| 0 | **CPU**: E1 edit replay (Atul); build the S3 compactor + S1 data; behaviour-gate prefixes and harness; cache the NF4 model as an output to cut the 37-min load | CPU | 0 | S3 data: 100% of edits grounded; scratch writes masked; spot-check 20 episodes |
| 1 | Train **S3** (and **S1**, same session if it fits) | Atul | ~8 | Runs; exports 410 modules |
| 2 | Behaviour gate: base vs S1 vs S3 | Atul | ~2 | Edits at the "file in view" prefixes ≥ base; no batching |
| 3 | Paired eval: best adapter vs fresh v7, **new-60 tasks**, 1 pass each | Shrey | ~16 | Adapter ≥ v7 + 3 tasks, no rise in empty patches |
| 4 | Leaderboard submission of v7 + adapter | — | 0 | Daily slot; compare to v7 (0.15) and v11 |
| 5 | If it wins: scale S3 to 500–1,000 public runs (or S4), retrain once, repeat 2–4 | both | ~15–25 | Same gates |

That's about **26 quota-h for steps 1–3**, under half the week's 60. One paired 114-task confirmation (~18 quota-h) still fits if step 3 is promising.

**Stop rules:** if S3 and S1 both fail the behaviour gate, stop training for the leaderboard. Write up the Step D collapse and E2 analysis as the paper's post-training result, and put GPU time into local evals of prompt and config variants.

## 6. What we will not do

- Retrain on the unchanged Step C data.
- Pick checkpoints by looking at evaluation results.
- Evaluate on training tasks.
- Raise call or context budgets as a substitute for edit behaviour; v9 already tested that idea at 0.12.
- Go to DPO or RL before SFT data is validated (the research report's ordering).
- Run any GPU session over 2 h without agreeing it first; both accounts log every run in §8 of `docs/post_training_guide.md`.

## 7. Paper framing

- **Finding:** converting a stronger agent's trajectories into the target's tool format can teach perfect *format* and destroy *behaviour*. Low loss, 0/114 solved.
- **Why:** state loss in windows, edits trained without their code, and supervision dominated by scaffolding. All of it is measured (E2), and the data reproduces byte for byte (E0).
- **Fix:** grounded compact episodes (S3), checked with a behaviour gate.

That covers the rubric's "insight into existing methods" and verifiability, whatever the leaderboard says.
