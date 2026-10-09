# Failure-targeted fine-tuning for SWE agents (mid-size open models), mapped to Gemma-4-31B failure modes — as of Oct 8 2026

Labels: **VERIFIED** = read in the paper's full text or its alphaXiv structured report in this session. **SNIPPET** = seen only in an abstract or search-result snippet. **INFERRED** = my reasoning, not a sourced claim. Unless marked otherwise, every paper below is an **arXiv preprint with no peer review confirmed**. SWE-Gym is the exception: ICML 2025, peer-reviewed.

Project baseline used for mapping: `CLAUDE.md`, `docs/findings.md`, `docs/experiments.md`, `results/runs/r1_baseline_v1|r2_v2_sed/summary.json`.
- r2 (v2_sed) failure stages: `fix:wrong_fix_unverified` 47/129, repeated identical calls 28.4%, tasks with loops 59.7%, error recovery 57.4%.
- r1 (baseline): `tested_after_last_edit_pct` 7.0%, `repro_before_edit_pct` 22.5%, P(resolved | saw gold lines) 15.6% vs 2.6%.

## Key question 0: Grid of failure mode → interventions in the literature → reported effect → cost under our constraints (LoRA, ~30 L4-GPU-h/week, no online RL)

### Takeaway
The best-evidenced lever that fits offline LoRA is **quality-filtered or process-filtered SFT / rejection-sampled FT on teacher trajectories**: SWE-Prime, FailForge, SWE-Gym, SWE-smith. It raises resolve rate and also moves the process metrics we care about (redundancy, observe-before-edit, tool success, test-before-edit). Loops and verification have the most targeted evidence. Malformed tool-call args have no SWE-specific fine-tuning evidence, but our data shows they are a large, cheap-to-fix channel.

### Cited Findings (grid)

| Our failure mode (measured) | Intervention in the literature | Reported effect | Cost / fit to our constraints | Label |
|---|---|---|---|---|
| **Malformed tool-call args** (edit_file 61% garbled in v5; read_file `start_line"` keys 47%) | (a) SFT trajectory selection that scores tool-call success rate as a quality signal (SWE-Prime S_tool) | Fine-tuned models raise tool success rate by up to +0.094 and resolve rate by up to +24.2% relative vs all-resolved SFT | Offline SFT; 10% of a 32k-trajectory pool ≈ 3.2k trajectories | VERIFIED [SWE-Prime](https://www.alphaxiv.org/abs/2608.27449) |
| | (b) Reflexive repair prompt after a failed call, on Gemma-4-31B INT4 (τ²-bench telecom) | +5.9 pts score (CI +1.1, +10.8); hallucinated-tool events 1,646 → 82. Helped only the damaged cell | Zero training cost, prompt-only | VERIFIED [Flat Score, Amplified Failures](https://www.alphaxiv.org/abs/2607.27275) |
| | (c) Quantisation link: INT4 AWQ (g=32) doubles Gemma-4-31B tool-name hallucination (19.5% → 38.3%); FP8 shows no significant change. Quantisation amplifies *existing* errors (BF16/INT4 name-distribution correlation 0.97; only 0.18% novel) | Final score flat (within 1.5-pt noise) at max_errors=10. The gap grows to 16.7 pts at an error budget of K=2 | Diagnostic only | VERIFIED [Flat Score](https://www.alphaxiv.org/abs/2607.27275) |
| | (d) Function-calling SFT corpora (APIGen-MT/xLAM-2, ToolACE, Hammer, ToolLoop, WEFT) | Gains reported on BFCL/τ-bench. **I found no paper showing transfer to SWE-bench resolve** | Data is free; format mismatch with the gemma4 tool parser | SNIPPET [APIGen-MT](https://www.alphaxiv.org/abs/2504.03601), [WEFT](https://www.alphaxiv.org/abs/2609.36887), [ToolLoop](https://www.alphaxiv.org/abs/2609.09072) |
| **Exact-repeat loops** (28–48% of calls) | (a) Process-filtered SFT: drop trajectories with adjacent identical calls (S_redundancy) plus segment-level loss masking | Redundancy down by up to 0.188; turns down 4.1–12.7; higher resolve | Offline SFT; needs an LLM segment scorer | VERIFIED [SWE-Prime](https://www.alphaxiv.org/abs/2608.27449) |
| | (b) Harness: replace the verbatim failed call in context with a runtime description of the failure, and/or ban already-failed strings at the decoder | Removes 76% of the "corrective-gain inversion". Exact repeat rate 31% → 8% (ban) and loops 29% → 4% (abstract+ban). A "do not repeat" instruction does nothing; a clean restart is worst (67% loops). **Only 135M–1.7B models tested** | Zero training. In our setting it needs harness control, which declarative ADK may not allow (INFERRED) | VERIFIED [Feedback That Backfires](https://www.alphaxiv.org/abs/2608.23651) |
| | (c) Weight surgery on Gemma-4-31B: strip 1,100 MLP neurons in L36 | Enumeration "fast-commit" loops 13/64 → 0/64 (thinking on); benchmarks within ±1 pt. Did *not* fix "doom looping" (non-convergent self-correction) | Weight edit, could be expressed as a LoRA delta (INFERRED). Tested on enumeration prompts, not agent loops | VERIFIED [One Neuron / Repetition Loops](https://www.alphaxiv.org/abs/2606.13705) |
| | (d) Early-failure monitor plus restart with prior edits as an optional overlay. Policies include **Gemma4-31B** | Gemma4-31B resolve +3.0 pts to 65.2% (mini-swe-agent, SWE-bench Verified test split); saves 15–20% tokens; 0.6B LoRA monitor transfers across policies | Monitor training is cheap (0.6B LoRA). Restart needs harness control | VERIFIED [Fail-Fast, Restart-Smart](https://www.alphaxiv.org/abs/2608.03222) |
| **Wrong fix never tested** (47/129; tested-after-last-edit 7%) | (a) Train a separate Test agent (SFT on 5k CoT trajectories, then GRPO) that writes a frozen repro test; Repair agent iterates against it | Base→Gold test validity 22.2% (base) → 39.6% (SFT) → 62.2% (RL). Composed system 61.2% → 72.6% on Verified (Qwen-3.5-35B-A3B). **Bad self-generated tests hurt: 61.2% → 57.3%** | SFT stage is feasible offline; the RL stage is not. Two agents fit our multi-LoRA (≤8 adapters) and Sequential agent | VERIFIED [ExecCritic](https://www.alphaxiv.org/abs/2609.09133) |
| | (b) Process-filtered SFT that keeps only observe→edit→verify trajectories (S_workflow) | Observe-before-edit up to +9.6 pp | Offline SFT | VERIFIED [SWE-Prime](https://www.alphaxiv.org/abs/2608.27449) |
| | (c) Failure-derived procedural skills, removed before training | FailForge students test before editing in 97.2% of trajectories; Qwen3.5-4B 59.6% → 66.2% pass@1 | Needs a strong teacher, a skill-inducing agent and a judge (API cost) | VERIFIED [FailForge](https://www.alphaxiv.org/abs/2608.08570) |
| **Multi-function fixes rarely resolved** (2/45) | FailForge skill taxonomy: "change propagation" 18.5% of induced skills, "fix placement" 21.5% | Indirect: higher localisation precision (0.52 solved / 0.45 unsolved) | As above | VERIFIED [FailForge](https://www.alphaxiv.org/abs/2608.08570) |
| **Sub-agent budget overruns** (asked ≤25, averaged 48) | Training on shorter or efficiency-filtered trajectories (SWE-Prime turn reduction); supervision-horizon selection; dependency-aware trajectory refinement | Turns −4.1 to −12.7 (SWE-Prime). Others seen only as abstracts | Offline SFT | VERIFIED (SWE-Prime); SNIPPET [Supervision horizon](https://www.alphaxiv.org/abs/2610.05831), [Dependency-aware refinement](https://www.alphaxiv.org/abs/2609.18417) |
| **General / transfer to unseen repos** | Rejection-sampled SFT on many repos | SWE-Gym: up to +19% absolute from SFT; 32.0% Verified with a verifier (ICML 2025). SWE-smith: 40.2% Verified for a 32B model, 50k instances from 128 repos | Data and trajectories are public; LoRA on 31B is the cost driver | SNIPPET (abstracts) [SWE-Gym](https://arxiv.org/abs/2412.21139), [SWE-smith](https://arxiv.org/abs/2504.21798) |

### Inferences
- INFERRED: a reasonable LoRA mix is a small (~1–3k), heavily filtered set of trajectories that are tool-call-valid, contain no adjacent repeats, and end with a test run. That mix attacks three of our five failure modes in one adapter. SWE-Prime's result that 10% filtered data beats 100% supports a small set, which suits 30 GPU-h/week.
- INFERRED: the malformed-args failure is a token-level format error, shaped like quotes leaking into keys. Exact-format SFT on the gemma4 tool parser's schema is probably the cheapest single fix. Process metrics can verify it offline without running agents.
- INFERRED: Flat Score suggests our W4A16 QAT base already amplifies Gemma's tool-call failure channel. QAT is not AWQ, so this is unverified for our checkpoint. A LoRA trained *on top of the quantised base* (QLoRA-style) learns to correct the quantised logits, which matters more than BF16 training.

### Gaps
- No paper isolates the effect of function-calling SFT (BFCL-style data) on SWE-bench resolve.
- No paper tests unlikelihood or negative-example training specifically against agent action loops on SWE tasks. Agentic-DPO is closest ([abstract](https://www.alphaxiv.org/abs/2607.10601), SNIPPET).

## Key question 1: Tool-call reliability training (function-calling FT, error recovery, reflection on tool errors) — effects on agentic benchmarks?

### Takeaway
Function-calling corpora improve BFCL/τ-bench, but there is no direct SWE-bench evidence. For SWE agents, the evidence that tool reliability *moves resolve* is indirect: trajectory filtering on tool success co-moves with resolve. Error-recovery evidence says error messages in context can *increase* repetition of the failed call. Recovery is best fixed by changing what the model sees, or by RL, which we cannot run.

### Cited Findings
- VERIFIED: in a failure record, the probability of re-emitting the just-failed call *rises*.
  - Over 4 candidates it goes from 0.06 to 0.54; greedy decoding reproduces the failed call verbatim on 19% of items after the failure vs 0% before.
  - The effect holds for 6 instruction-tuned models (135M–1.7B), on 90–100% of items.
  - 83% of the damage comes from the failed call's surface form being in context.
  - Source: [Feedback That Backfires (arXiv 2608.23651)](https://www.alphaxiv.org/abs/2608.23651).
- VERIFIED: in the same paper's rollout harness experiments, a decoder ban on failed strings cut exact repeats 31% → 8% (paired −23 pts [−38, −10]) at no change in tokens. The explicit "do not repeat" prompt did not move the measure. Source: [same](https://www.alphaxiv.org/abs/2608.23651).
- VERIFIED: τ²-bench, Gemma-4-31B dense.
  - Recovery rate after a failed call is ~55% in both BF16 and INT4.
  - Quantisation raises the *number* of failures (1.45 → 3.64 per episode), not the recovery ability.
  - A mechanism-targeted repair prompt gives +5.9 pts.
  - Source: [Flat Score, Amplified Failures (2607.27275)](https://www.alphaxiv.org/abs/2607.27275).
- SNIPPET: controlled error injection (retry/switch/abstain).
  - Bayesian Tool Memory gives up to +16.8 pp without retraining; RL adds recovery behaviours.
  - 40.8–45.5% under error injection while keeping failure-free performance; 7 models, 4 families.
  - Source: [Retry, Switch, or Abstain? (2608.11977)](https://arxiv.org/abs/2608.11977).
- SNIPPET: function-calling data generators exist, but in the snippets seen they evaluate on BFCL/τ-bench, not SWE-bench: [APIGen-MT](https://www.alphaxiv.org/abs/2504.03601), [ToolLoop](https://www.alphaxiv.org/abs/2609.09072), [WEFT](https://www.alphaxiv.org/abs/2609.36887).
- VERIFIED: PACE's proxy-instance analysis selected BFCL instances repeatedly across agentic targets. "Error Recovery" was among the capabilities selected for SWE-bench Verified. Source: [PACE (2607.02032)](https://www.alphaxiv.org/abs/2607.02032).

### Inferences
- INFERRED: Feedback That Backfires predicts our 28–48% exact repeats. Gemma sees its own failed `edit_file`/grep call verbatim and copies it. A fine-tuning target follows from that: SFT examples where a failed call is followed by a *differently-formed* call. Mask loss on the failed call; train only on the recovery. This is the closest offline analogue to "reflection on tool error" training.
- INFERRED: never train on the malformed calls themselves. SWE-Prime-style segment masking (loss only on high-value segments) is the mechanism.

### Gaps
- No ToolACE/Hammer/xLAM ablation reporting SWE-bench resolve. I did not fetch the ToolACE or Hammer papers this session.
- Feedback That Backfires is limited to ≤1.7B models. Whether the inversion holds at 31B is untested.

## Key question 2: Loop / repetition and premature-stop — training interventions

### Takeaway
Evidence for SWE loops points to **data curation** rather than special losses: drop adjacent-identical calls and mask low-value segments. On the harness side, a decoder ban plus abstracted failure records works. Gemma-4 specifically has a localisable "fast-commit" loop circuit (31B: L36 MLP), but "doom looping" is entangled with reasoning.

### Cited Findings
- VERIFIED: SWE-Prime scores S_redundancy = 0 if adjacent steps make an identical tool call. It trains on the top 10% per cluster, with loss masked to segments scoring ≥7/10.
  - Qwen3-Coder-30B-A3B: 50.2% (stage 1) → 53.2% (with segment masking) on Verified.
  - Redundancy down by up to 0.188; 4.1–12.7 fewer turns.
  - Source: [SWE-Prime (2608.27449)](https://www.alphaxiv.org/abs/2608.27449).
- VERIFIED: in Gemma-4-31B, stripping 1,100 L36 MLP neurons removed enumeration loops (13/64 → 0/64, thinking on) with benchmarks within ±1 pt. A repetition penalty of 1.15–1.30 degraded code synthesis.
  - At ≥4,096-token budgets, the edit converted tight loops into endless self-correction. The authors call doom looping "a knowledge-precision problem".
  - Source: [When Can One Neuron Fix Repetition Loops (2606.13705)](https://www.alphaxiv.org/abs/2606.13705).
- VERIFIED: a clean restart (deleting the failed attempt) produced the most loops of any harness, 67% of rollouts. Source: [Feedback That Backfires](https://www.alphaxiv.org/abs/2608.23651).
- VERIFIED: Fail-Fast monitor plus RestartSmart.
  - With the prior edits as an optional overlay, Gemma4-31B gained +3.0 pts (to 65.2%). A cold restart gained only marginally (+0.2 pts on Qwen3.6-27B).
  - A prompt intervention (SWE-PRM) *hurt* the stronger Qwen3.6-27B by −3.2 pts.
  - Source: [Fail-Fast, Restart-Smart (2608.03222)](https://www.alphaxiv.org/abs/2608.03222).
- SNIPPET: refining trajectories by removing redundant rounds (failed tool calls, verification-only steps) to make multi-turn FT efficient: [Dependency-Aware Trajectory Refinement (2609.18417)](https://www.alphaxiv.org/abs/2609.18417). Studying how much of each long trajectory to supervise: [Selecting Long-Horizon Trajectories (2610.05831)](https://www.alphaxiv.org/abs/2610.05831).

### Inferences
- INFERRED: Fail-Fast reports Gemma4-31B at ~62% on Verified under mini-swe-agent, while we get 26–33/129 locally. So the ceiling is a scaffold/harness/format problem more than raw capability. Process-fixing LoRA has headroom.
- INFERRED: premature stop (empty patches, 20/129 in r2) has no dedicated training paper in this search. Filtering for trajectories that end in explicit submit-after-test is the natural SFT analogue.

### Gaps
- No unlikelihood-on-repeated-action result for SWE agents.
- No premature-stop-specific SFT study found.

## Key question 3: Verification behaviour — training agents to write and run repro tests

### Takeaway
Repro-test training raises resolve, but only if test *quality* is high. Low-quality self-written tests reduced resolve by 3.9 pts in ExecCritic. SFT alone roughly doubles test validity (22% → 40%), and RL adds a further +23 pts, which we cannot run. Frozen tests from a separate role prevent "tests that agree with the wrong patch".

### Cited Findings
- VERIFIED: ExecCritic (Qwen-3.5-35B-A3B). Source: [ExecCritic (2609.09133)](https://www.alphaxiv.org/abs/2609.09133).
  - No-test baseline 61.2%. Base-model self-tests → 57.3%; GPT-5.6 tests → 65.3%; oracle F2P → 69.4%.
  - Test agent Base→Gold validity: 22.2% → 39.6% (SFT, 5k CoT trajectories from DeepSeek-V4-Flash) → 62.2% (SFT+GRPO). SFT showed diminishing returns with more data; RL from base without SFT was unstable; CoT-visible SFT gave a better RL init.
  - Composed trained system: 72.6% (+11.4).
- VERIFIED: FailForge students tested before editing in 97.2% of trajectories, with fewer median edits (9.9). Test-contract grounding is the largest skill category (21.8%). Source: [FailForge (2608.08570)](https://www.alphaxiv.org/abs/2608.08570).
- VERIFIED: PACE found SWT-Bench (test generation) and SWE-bench Verified both draw on "Verification and Test" proxy instances. Source: [PACE](https://www.alphaxiv.org/abs/2607.02032).
- SNIPPET: repro-test generation benchmarks and agents exist: [Multi-SWT-Bench (2609.34752)](https://www.alphaxiv.org/abs/2609.34752), [SWE-Doctor (2607.00990)](https://www.alphaxiv.org/abs/2607.00990), [DPIAgent (2608.23341)](https://www.alphaxiv.org/abs/2608.23341).

### Inferences
- INFERRED: given our 47/129 "wrong fix never tested", ExecCritic's negative result is a warning. Merely forcing Gemma to test (prompt or SFT) may produce false confidence. Gate on *Base-fails* (the test must fail before the edit); this is checkable in our sandbox.
- INFERRED: a feasible design is a Test-writer LoRA (SFT only) plus a Fixer LoRA in a SequentialAgent. Repro-test validity (fails on base, passes on gold) can be measured offline on the 54 scorer-faithful tasks without full agent runs.

### Gaps
- No SFT-only (no RL) ablation of repro-test training on resolve for a ~30B dense model.
- SWE-smith and Agentless repro ablations were not fetched this session.

## Key question 4: Process skills vs task-specific knowledge — what generalises to unseen repos?

### Takeaway
Procedural supervision transfers across instances and harnesses better than instance hints. That includes leakage-filtered skills internalised by SFT and filtered process-clean trajectories. Instance-level knowledge transfers poorly.

### Cited Findings
- VERIFIED: FailForge. Source: [FailForge (2608.08570)](https://www.alphaxiv.org/abs/2608.08570).
  - Procedural skills kept 59.3% of their recovery rate when moved to a sibling instance, vs 20.8% for instance hints.
  - FailForge 66.2% vs RFT+hint 63.8% (hints recovered *more* instances, 352 vs 218).
  - Cross-harness: OpenHands-trained models evaluated under Claude Code kept the lead (62.6% Verified).
  - Train/eval had repository-level separation.
- VERIFIED: SWE-Prime gains held across Python/JS/TS/Go on SWE-bench Pro. Source: [SWE-Prime](https://www.alphaxiv.org/abs/2608.27449).
- SNIPPET: SWE-smith attributes its gains to scale and diversity (50k instances, 128 repos vs ≤11 previously). Source: [SWE-smith (2504.21798)](https://arxiv.org/abs/2504.21798).

### Inferences
- INFERRED: this matches our LB evidence. Hidden repos are private, and local gains on memorised public repos stopped transferring. Train on diverse-repo, process-clean trajectories (SWE-smith/SWE-Gym/SWE-rebench pools filtered SWE-Prime-style), never on the 129 public tasks.

### Gaps
- No study isolates "process-only" LoRA vs full SFT at 31B under a fixed GPU budget.

## Key question 5: Cheap proxies that predict resolve gains

### Takeaway
Three proxies are evidenced: (1) process-channel error rates measured with a *tightened error budget*, (2) a learned early-outcome predictor on trajectory prefixes, and (3) IRT subset selection (10% of tasks recovers rank, τ≈0.89). Next-action prediction on held-out expert trajectories has no evidence I could find. Process metrics co-move with resolve in fine-tuning papers (SWE-Prime, FailForge), but no paper reports a validated correlation coefficient for malformed-rate → resolve.

### Cited Findings
- VERIFIED: aggregate task reward can hide 2× process damage. Re-scoring with a counterfactual error budget K widens the BF16–INT4 gap from 1.3 pts (K=10) to 16.7 pts (K=2). The authors recommend reporting per-channel error rates and S(K). Source: [Flat Score (2607.27275)](https://www.alphaxiv.org/abs/2607.27275).
- VERIFIED: PTA-IRT. With a 10% calibration subset it reaches mean MAE 0.041 (best baseline 0.126) and Kendall τ 0.888. At 5% on Lite, τ = 0.768. Source: [Efficient SWE Agent Benchmarking (2609.01603)](https://www.alphaxiv.org/abs/2609.01603).
- VERIFIED: a 0.6B LoRA prefix monitor reaches 30.5% failure recall at 5% FPR and 68.3% at 25% FPR. It transfers to Gemma4-31B, and dense fail-to-pass progress targets add +6.2% recall. Source: [Fail-Fast (2608.03222)](https://www.alphaxiv.org/abs/2608.03222).
- VERIFIED: PACE predicts SWE-bench Verified from 100 non-agentic instances, but cross-*model* (14 models): MAE 2.09%, Spearman 0.67, pairwise accuracy 78.5%. Source: [PACE (2607.02032)](https://www.alphaxiv.org/abs/2607.02032).
- SNIPPET: early outcome prediction for cheaper agent evaluation: [EarlyEval (2609.02783)](https://www.alphaxiv.org/abs/2609.02783).
- VERIFIED: SWE-Prime's process metrics (redundancy, observe-before-edit, tool success) all improved in the same direction as resolve across 3 models × 2 benchmarks. Source: [SWE-Prime](https://www.alphaxiv.org/abs/2608.27449).

### Inferences
- INFERRED, ranked by cost:
  1. Malformed-call rate and exact-repeat rate on ~20 short-budget local tasks (minutes, no full resolve needed).
  2. Repro-test Base-fail / Gold-pass validity on the 54 scorer-faithful tasks (one test-writer turn, no fix).
  3. An S(K)-style re-scoring of existing traces (free).
  4. A 10–15-task IRT-selected subset of the 129 for resolve.
- INFERRED: PACE is cross-model and calibrated on frontier models. It will not resolve LoRA-vs-LoRA deltas of a few points on one base.

### Gaps
- No paper validates next-action-prediction accuracy on held-out expert trajectories as a resolve predictor.
- No paper gives a correlation coefficient between malformed-call rate and resolve on SWE tasks.
- Every "process metric ↔ resolve" link found is co-movement under intervention, not a calibrated predictor.

## Key question 6: Remaining gaps and search log

### Takeaway
Coverage is strongest on trajectory curation, loops and verification. It is weakest on function-calling → SWE transfer and on offline next-action proxies.

### Cited Findings
- SNIPPET (newest, unread): FC-SWE trains SWE agents with failure-conditioned RL. It is online RL, not applicable to us. Source: [FC-SWE (2610.07898)](https://www.alphaxiv.org/abs/2610.07898).

### Inferences
- INFERRED: we have 129 tasks with ~27–33 resolves per run. Detecting a +3-task gain is underpowered: v5 was +12/−5, p=0.14. Process proxies are needed for iteration, with the LB used only for confirmation.

### Gaps
- Not fetched or verified this session:
  - ToolACE, Hammer and xLAM full papers.
  - SWE-smith ablations on repro and diversity.
  - The Agentless repro-test selection effect.
  - Nemotron/CORTEXA or Skywork-SWE SFT scaling.
  - Agentic-DPO results.
  - Diagnosis-Before-Recovery (2608.11772) results.
- Search log:
  1. alphaXiv discover: "fine-tuning open models for SWE-bench agents… tool-call reliability, error recovery, repro tests, loop avoidance" [SWE-bench, SWE-agent, SFT, trajectories, SWE-smith, SWE-Gym]
  2. alphaXiv discover: "function calling fine-tuning effect on agentic benchmarks, tool error self-correction, repetition loops" [APIGen, xLAM, ToolACE, Hammer, BFCL, tool error, self-correction]
  3. alphaXiv discover: "training SWE agents to write and run reproduction tests…" [reproduction test, SWE-bench, test generation, verification, SWE-smith, R2E-Gym]
  4. alphaXiv discover: "cheap offline evaluation… next-action prediction… process metrics correlated with SWE-bench resolve"
  5. alphaXiv discover: "function-calling FT transfer to SWE-bench; malformed tool args; unlikelihood/negative trajectories to reduce repeats"
  6. Full reports read: 2608.08570, 2608.23651 (full text grep), 2606.13705, 2608.27449, 2607.02032, 2609.09133, 2608.03222, 2609.01603, 2607.27275
  7. WebFetch abstracts: arXiv 2504.21798 (SWE-smith), 2412.21139 (SWE-Gym), 2608.11977 (Retry/Switch/Abstain)
