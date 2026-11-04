# Gemma 4 31B: causal attribution of LoRA contributions to software repair

Author: Atul independent research analysis · 4 November 2026 · Research design only

## 1. Answer, scope and evidence standard

**INFERENCE — Main answer:** No available result identifies individual Gemma weights, attention heads, or LoRA modules as causally responsible for successful software repair. There is enough evidence to design a controlled intervention study. The most informative next GPU experiment, if separately authorized, is a **four-arm attention × MLP adapter ablation**, keeping the T1 checkpoint and v14 policy fixed: no adapter, attention only, MLP only, full adapter. This distinguishes family contributions and interactions before selecting layers or retraining. Module selection should be driven by held-out intervention outcomes, not large update norms.

**VERIFIED — Audit provenance:** Local HEAD was `d54b8902198b92c6edb65a0e2d1ffc7cca5bb78b` on `codex/atul-paper-training-audit-review`. The only initial untracked status entry was `data/runs/atul_r2_s1/`. A read-only GitHub API query found newer main `e40f30b5e965e7c35b2c096a08eaaa04a04381c1`. Latest HQ, experiments, LT1 results and LT1 implementation were read at that exact revision through temporary copies, without Git fetch, checkout, branch creation or updating existing files. Git blob comparisons show the local RS2/T1/LP1/LP2/G1 result files and existing trainkit/evalkit Python files inspected here match that pinned main; LT1 files are new remotely. “Latest” means this observed revision, not a promise that main will never advance.

Labels throughout:

- **VERIFIED:** directly inspected source, artifact contents, tensor headers or recomputed arithmetic. A recorded GPU measurement is verified as a record, not independently rerun hardware evidence.
- **INFERENCE:** interpretation supported by those records, with alternative explanations remaining.
- **HYPOTHESIS:** proposed mechanism, experiment, success threshold or resource estimate, not an observed result.
- **BLOCKED:** evidence needed for the claim was unavailable or no relevant intervention has been run.

**BLOCKED — Tensor availability:** A filesystem search under `adapters/`, `data/` and `dist/` found 13 exact-name `adapter_model.safetensors` files, but none belonging to T1 `rs2_lora`. Older S1/S3 checkpoints are available. Their headers establish the architecture inventory below, not T1's numerical updates. No T1 norm, singular spectrum, important-layer ranking, or per-weight causal score is claimed. No model was loaded, no GPU accessed, no adapter changed, and no dataset uploaded. Only this report is added inside the repository.

### Evidence index

All remote paths below are pinned to the observed main; source paths elsewhere in this report are repository-relative to that revision unless explicitly called local-only.

| ID | Exact evidence | What it establishes |
|---|---|---|
| H | [docs/HQ.md](https://github.com/ShreySharma07/gemma4-swe-agent/blob/e40f30b5e965e7c35b2c096a08eaaa04a04381c1/docs/HQ.md) | Latest decisions, LT1, reported v18 timeout and limitations |
| E | [docs/experiments.md](https://github.com/ShreySharma07/gemma4-swe-agent/blob/e40f30b5e965e7c35b2c096a08eaaa04a04381c1/docs/experiments.md) | RS2/T1/LP1/LP2/G1/LT1 registrations and deviations |
| R | `results/runs/rs2/rs2_report.json`; `kaggle/rftset/rs2_core.py` | Training population, weighting and fallback logic |
| T | `results/runs/t1/l4_v1_summary.json`; `kaggle/train/trainkit/{config,model,data,worker,fix_lora,export}.py` | Architecture, optimization, checkpoint exports and training measurements |
| P1 | `results/runs/lp1/l4_v1_report.json`; `kaggle/eval_l4/evalkit/lora_probe.py` | Served-base weighted validation loss |
| P2 | `results/runs/lp2/l4_v1.json`; `kaggle/train/trainkit/vablate.py`; `kaggle/eval_l4/evalkit/kvdup.py` | Global-layer V-delta intervention and export semantics |
| G | `results/runs/g1/l4_v1_report.json`; `kaggle/eval_l4/evalkit/{lora_gate,server,submission}.py` | Matched behavioral outcomes and serving controls |
| L | [results/runs/lt1/l4_v1_report.json](https://github.com/ShreySharma07/gemma4-swe-agent/blob/e40f30b5e965e7c35b2c096a08eaaa04a04381c1/results/runs/lt1/l4_v1_report.json); [lora_timing.py](https://github.com/ShreySharma07/gemma4-swe-agent/blob/e40f30b5e965e7c35b2c096a08eaaa04a04381c1/kaggle/eval_l4/evalkit/lora_timing.py) | 36 timing records, attention-only construction, timing methodology |
| A | Local-only `data/runs/atul_s1_own/adapters/s1_own/adapter_model.safetensors` and `data/runs/atul_s3_compact/adapters/s3_compact/adapter_model.safetensors`, adjacent configs | Header/shape corroboration; older checkpoints, not T1 |

### What the experiments actually say

**VERIFIED (R, T):** RS2 v2 retained 344 of 350 resolved episodes; 203 training, 40 validation, 101 capped out. Training includes 153 singles, 44 pairs, 6 triples; 133/203 come from boltons, sqlparse and more_itertools. Eleven of 243 rendered episodes had span mismatches and fallback weights. These heuristics can affect what any module learns. T1 trains whole episodes from `ep_starts`, not independent 4k RS2 chunks: rank 16, alpha 32, dropout 0, batch 8, two epochs/50 optimizer steps, 200 episodes per epoch from 203, no length drops, maximum 26,831 tokens. Recorded validation loss is 0.1011 → 0.0669 → 0.0666; 122,429,440 trainable parameters across 410 modules. The second epoch's small validation improvement does not by itself prove memorization or establish the best behavioral checkpoint.

**VERIFIED (P1, P2):** LP1's W4A16 validation loss is base 0.08663, raw 0.05075, K→V-copy 0.05092. The raw adapter improves this imitation metric by 41.42%. LP2's NF4/HF loss is full 0.06661, global-V-base 0.06637, full-again 0.06661, base 0.10109; 26/40 episodes favor V-base. This is an actual component intervention on **teacher-forced loss**, not a demonstration that K/V changes improve autonomous fixing. Agreement of the effect direction across stacks does not prove bitwise forward equivalence.

**VERIFIED (G; recomputed):** G1 has 95 matched IDs: both solve 36, full-only 6, base-only 4, neither 49; base 40/95, full 42/95, net +2.105 percentage points. Exact two-sided McNemar p = 0.75390625. Overflow counts are 12 versus 18; shared 8, adapter-only 10, base-only 4, p = 0.17956543. Median task duration is 328.39 versus 349.37 seconds, empty patches 8 versus 9. G1 used concurrency 6 and 12-minute tasks, whereas v14's submitted policy uses 8 minutes and scorer concurrency 1. It reused toolz/pathspec validation-domain tasks, so it is not untouched confirmation after checkpoint/serving selection.

**INFERENCE:** The data are compatible with beneficial changes, harmful changes, cancellation between modules, or sampling variation. “No significant improvement” does not mean no causal effect; likewise, nonsignificant overflow discordance does not justify declaring the adapter innocent. HQ's exact equation “24,572 = 32,768 − 8,192” has a four-token arithmetic error: the reserved-output ceiling is **24,576**, and 24,572 can only be an observed maximum below it.

## 2. Why individual weights are not bug-fixing units

**INFERENCE — Mechanistic framing:** Successful repair is a trajectory-level property of a transformer, decoding policy, tool environment, budget, repository and test oracle. Localization, correct reasoning, tool syntax, mutation, verification and termination interact. A weight can participate in many functions; the same behavior can be redundantly represented. Perturbing one parameter can change downstream token choices and the entire tool history. A damaging intervention can show that a parameter matters under that intervention, but does not label it a “bug-fixing neuron.” General language degradation must be distinguished from a repair-specific deficit.

For a LoRA linear, `y = W x + s B A x`, where `s = alpha/r = 2` here. The decomposition is nonidentifiable: for invertible R, `(B R)(R^-1 A) = B A`. Consequently A/B entry magnitudes, individual rank columns, and their separate singular values are coordinate-dependent. Even invariant ΔW statistics are not behavioral causal attribution. Nonlinear residual interactions, normalization, attention mixing and quantization make size-to-impact mappings unreliable. Correlation between a layer's update norm and success also has training-data and optimization confounders.

**HYPOTHESIS — Appropriate causal claim:** “Removing the learned MLP delta from this frozen T1 adapter changes the probability of passing these tasks by X under this serving stack and budget.” This is narrower and testable. It does not assign pretraining's bug-fixing ability to those modules. Establishing a mechanism requires convergent evidence: controlled removal, retained-only sufficiency tests, restore/rescue, and eventually activation/path interventions at specific decisions.

Methodological grounding: [LoRA](https://arxiv.org/abs/2106.09685) motivates the frozen-base low-rank parameterization; [Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042) shows that intervention choices and measurement metrics matter; [AtP*](https://arxiv.org/abs/2403.00745) motivates gradient attribution as a screening approximation that still needs direct intervention confirmation. These papers motivate methodology, not evidence about this Gemma checkpoint. Single-prefix gradients are particularly weak evidence here: LC1's repository records show substantial bf16 gradient sensitivity to rounding.

## 3. Complete adapted-module map

**VERIFIED (A, T, P1):** The inventory is 60 text transformer layers, indices 0–59. Every layer has q/k/o and gate/up/down adapters; 50 sliding layers additionally have v. Global layers are `5,11,17,23,29,35,41,47,53,59`. Thus `50×7 + 10×6 = 410`: attention 230, MLP 180; two factors per module give 820 tensors. Independent S1 and S3 header reads have the same 820 keys/shapes, BF16 dtype, valid in-file offsets and 122,429,440 scalars, agreeing with T1's recorded count. **INFERENCE:** This reconstructs the expected complete T1 key set from matching architecture and export source; T1 byte-level verification remains BLOCKED until its exact file is available.

Canonical module prefix: `base_model.model.model.language_model.layers.{i}.` followed by `self_attn.{q_proj,k_proj,v_proj,o_proj}` or `mlp.{gate_proj,up_proj,down_proj}`. Append `.lora_A.weight` or `.lora_B.weight` for tensor keys. Every row in Appendix A expands these names explicitly by layer. Global layers have **no independent trained v_proj**; a synthesized serving copy is not a 411th–420th independently learned module.

| Family/type | Layers | A shape | B shape | Total trainable scalars |
|---|---:|---|---|---:|
| q, sliding | 50 | 16 × 5376 | 8192 × 16 | 10,854,400 |
| q, global | 10 | 16 × 5376 | 16384 × 16 | 3,481,600 |
| k, sliding | 50 | 16 × 5376 | 4096 × 16 | 7,577,600 |
| k, global | 10 | 16 × 5376 | 2048 × 16 | 1,187,840 |
| v, sliding only | 50 | 16 × 5376 | 4096 × 16 | 7,577,600 |
| o, sliding | 50 | 16 × 8192 | 5376 × 16 | 10,854,400 |
| o, global | 10 | 16 × 16384 | 5376 × 16 | 3,481,600 |
| gate | 60 | 16 × 5376 | 21504 × 16 | 25,804,800 |
| up | 60 | 16 × 5376 | 21504 × 16 | 25,804,800 |
| down | 60 | 16 × 21504 | 5376 × 16 | 25,804,800 |

Attention has 45,015,040 scalars (36.77%), MLP 77,414,400 (63.23%). Raw family-norm sums must therefore not be interpreted without dimension/count normalization. Embeddings, LM head, norms, vision/audio components and biases are not targets of this text-only LoRA regex. The base remains frozen; the training implementation uses NF4, bf16 compute, FSDP base sharding, replicated LoRA, globally normalized weighted loss and gradient reduction, plus streamed loss/attention/MLP memory fixes. Export strips wrapper/default prefixes and writes BF16 PEFT tensors from rank 0 without merging them into base weights.

## 4. Read-only CPU tensor analysis protocol

**HYPOTHESIS — Proposed analysis, not executed numerical results:** First obtain provenance-verified T1 step25 and final tensors through an authorized read/download, identifying source/version, full SHA-256, config, shape manifest and file completeness. The abbreviated `2b22c3f3…` in HQ is insufficient to authenticate bytes. Keep S1/S3 in separate cohorts; their failed historical gates make them useful negative examples, not surrogates for T1. Do not execute pickles or load the 31B base merely to inspect safetensors. Use bounded header parsing and memory-mapped read-only arrays, CPU only, stream one module at a time; do not save transformed weights.

For each module with A ∈ R^(r×din), B ∈ R^(dout×r):

1. Check finite values, shapes, rank, alpha, dtype, offsets, absence of unexpected tensors, and any per-module scaling overrides. Record zero updates and numerical tolerances. BF16 bytes must be converted correctly to fp32/fp64 before arithmetic.
2. Compute the invariant update `ΔW = s BA` implicitly. Frobenius squared is `s² tr[(BᵀB)(AAᵀ)]`, using fp64 small Gram matrices; clamp only tiny negative roundoff and flag larger errors. Do not allocate dense dout×din matrices for every module.
3. For stable singular values, thin QR: `B=Qb Rb`, `Aᵀ=Qa Ra`. Nonzero singular values of ΔW are those of the r×r matrix `s Rb Raᵀ`; this avoids the conditioning loss from squaring in Gram eigenproblems. Retain σ1…σr, spectral norm, stable rank `||ΔW||F²/σ1²`, energy fractions of top 1/4/8 components, and entropy effective rank using normalized σ². Zero matrices get explicit zero/rank conventions, not division by zero.
4. Report update RMS `||ΔW||F/sqrt(dout din)`, per-family/layer medians and quantiles, total squared energy, energy per scalar and sliding/global stratification. Relative `||ΔW||F/||W||F` is BLOCKED without the exact corresponding base representation; NF4 and W4A16 denominators must not be silently interchanged. Distinguish quantization reconstruction error from the learned delta.
5. Compare step25→final in ΔW space, not A/B coordinates: signed cosine of flattened updates, change norm, and left/right subspace overlap via small cross-products. Degenerate singular vectors have no unique identity; compare subspaces. Do not subtract adapter weights from different training runs and call it an optimization trajectory.
6. If authorized calibration activations later exist, add `E||ΔW x||²`, relative output perturbation and input-covariance-weighted spectra on training-disjoint code/tool contexts. This is more functionally relevant than unweighted norms but remains descriptive. Reading already saved activations is CPU work; producing them with a 31B forward pass is a separate inference experiment.

Proposed output schema (inside a future separate analysis artifact): checkpoint/hash, exact key, layer, attention type, family, dimensions, parameter count, scale, norms, singular values, effective ranks, checkpoint difference, finite/shape checks and evidence label. Produce layer×family heatmaps and spectra, not “importance maps.” Rank modules only as candidates for testing; include matched random-layer controls, and keep discovery rankings blind to confirmation outcomes.

**VERIFIED — Work performed here:** Header/count/shape validation on S1/S3 only. No tensor values, norms or spectra were computed. Neither system Python nor the existing `.venv` exposes NumPy; no dependency was installed. A future CPU environment with NumPy or CPU PyTorch suffices. Rough planning estimate: 245 MB BF16 adapter, ~490 MB fp32 if fully expanded; streaming factors/QR should fit comfortably within a few GB RAM. Runtime is hardware-dependent; allow minutes to an hour for several checkpoints, benchmark before promising throughput. This does not require GPU quota.

## 5. Causal ablation design

**HYPOTHESIS — Frozen-checkpoint interventions:** Define mask m over module outputs: `Δy_j = m_j s_j B_j A_j x`. Active modules keep their original rank, scale and values. Suppression removes only the delta, never the base linear or entire residual branch. Do not multiply retained deltas to compensate for removed energy in the primary comparison; that would add a dose intervention.

| Arm | Retained learned deltas | Count | Causal question |
|---|---|---:|---|
| B | None; v14 policy and base | 0 | Reference behavior |
| F | Full raw T1 adapter | 410 | Total adapter effect |
| A | q/k/v/o only, raw global K convention | 230 | Attention sufficiency; removing MLP from F |
| M | gate/up/down only | 180 | MLP sufficiency; removing attention from F |
| E-only / F-minus-E | Layers 0–17 / complement | 123 / 287 | Early-group sufficiency/necessity |
| C-only / F-minus-C | Layers 18–41 / complement | 164 / 246 | Middle-group sufficiency/necessity |
| L-only / F-minus-L | Layers 42–59 / complement | 123 / 287 | Late-group sufficiency/necessity |

Groups preserve complete six-layer attention cycles. The middle group is larger; compare per-module descriptive quantities and include size/family/global-layer-matched random groups before attributing an effect to depth. A later balanced alternative is five consecutive 12-layer groups, 82 modules each. Do not run all groups immediately: first the four family arms, then a preregistered focused follow-up on independent tasks. Family ablation does not distinguish individual heads or MLP neurons.

The four-arm factorial permits `attention effect with M absent = A−B`, `with M present = F−M`, analogous MLP effects, and interaction `F−A−M+B`, all computed as within-task contrasts. A small A−B but large F−M suggests interaction, not useless attention. Removal can break co-adaptation; a large drop does not establish that the removed family alone contains the capability. Conversely retained-only success is conditional sufficiency with the entire pretrained base still present.

Controls before interpreting behavior:

- Full-mask identity and restore/rescue must reproduce the original adapter's forward outputs within an established backend tolerance; zero-mask output must match base logits. Include a **zero-delta loaded-adapter** control to separate runtime routing overhead from mathematical base behavior. These future controls require inference; not performed here.
- Verify served module membership, tensor hashes and loading logs; packed qkv / gate-up mappings can silently invalidate a nominal mask. Test tiny synthetic known deltas before 31B evaluation.
- Prefer runtime delta masks for causal experiments if supported and validated; otherwise derivative copies under future authorization, never overwriting original weights. Physical omission and zeroing can choose different kernel paths: record both when timing is part of the endpoint.
- Dose-response at λ=0, 0.5, 1 for a winning group and a same-size random group is a follow-up, not an unregistered hyperparameter sweep. Avoid individual-weight edits and rank-column ablations without specifying basis dependence.
- Optional activation patching: replay a matched repository/tool state, intervene on a module's delta output for one decision, and score a prespecified correct versus incorrect action. Keep the recipient context fixed; verify rescue by restoring the original activation. Test later free-running success separately. This is evidence about that decision and patching distribution, not automatic mediation of whole-task success.

## 6. Tasks, metrics, uncertainty and regressions

**HYPOTHESIS — Paired protocol:** Freeze task IDs, base commits, rewritten issues, dependency overlays, tool schemas, prompt, temperature, thinking budget, max output, context limit and test oracle before running. Use fresh sandboxes for every arm. Gold must pass and empty must fail in the same environment, with exclusion rules applied blind to arm. Never leak gold patches, hidden tests or sibling mutation solutions into model prompts. Split by source function/mutation family and repository where possible, not merely by instance ID. G1 can be reused for exploratory localization only; its 40 validation successes and serving-selection reuse must be declared. New-60 is also well-studied and is a regression set, not untouched confirmation.

Randomize/counterbalance arm order within task blocks, balance seed allocation, keep cache/warmup policy fixed, and alternate sufficiently to avoid temporal load confounding. Common seeds improve pairing but cannot force the same random trajectory after policies diverge. Use at least several seeds for finalists if affordable; resample tasks with all their seeds together. With only two G1 libraries, a repository-cluster confidence interval cannot support broad generalization; obtain more independent repositories for confirmation.

**Primary deployment endpoint:** resolved / all assigned eligible tasks at identical wall-clock and call budgets, concurrency 1, using unchanged v14 policy (8 minutes, 28 calls, thinking 4096, output reservation 8192). Report every task's outcome; environment failures remain explicit. Agent timeout/overflow is a deployment failure, whereas missing logs from platform failure are unknown. Report both conservative all-assigned totals and complete matched coverage; never silently drop inconvenient failures.

**Separate mechanistic endpoint:** fixed-token/fixed-call or frozen-prefix decisions under a nonbinding experimental wall-time limit, to distinguish policy competence from slower serving. This is a separate future condition, not permission to extend a job or alter the primary budget. Report raw speed alongside quality; do not adjust away overflow or time in the primary analysis because they are possible mediators of adapter effects.

Report paired both/F-only/comparator-only/neither, net gain, exact McNemar for one run per task, task-cluster bootstrap CIs for differences and factorial interaction, and per-repository estimates. For multiple preplanned arm comparisons use Holm correction or a prespecified hierarchical testing sequence. With multiple seeds, do not treat task×seed rows as independent. Evaluate all arms including worse ones; no outcome-dependent stopping without a registered sequential design. Use practical effect thresholds and noninferiority CIs, not p>0.05 as equivalence.

Secondary measures: resolved per GPU-hour; full runtime including setup and verification; completion/thinking/output tokens; peak prompt; overflow; actual BudgetExceeded denials separately from cap-proximity proxies; truncated/unparseable calls; repeated calls; final empty/stray patches; verifier regressions; and test **invoked**, test **passed**, and source **actually changed** after the last mutation as separate measures. New `analysis/mutation_test_metrics.py` (MM1 in latest H/E) improves old edit detection but is still heuristic. Python/heredocs/shell writes, commands that combine editing and testing, failed writes and scripts with indirect effects require trace and final-diff checks. No shared wrapt warning establishes failure causality.

Regression suites should include tool-format decisions, ordinary code comprehension, existing base-solved tasks, long-context tasks, API/interface compatibility and incorrect-test-feedback recovery. Blind manual review of wins/losses should use actual final patch and verifier evidence, not mutation intent or teacher-forced loss. Keep privacy-sensitive traces and source out of public reporting.

**INFERENCE — Power:** At G1's discordance rate q=10/95≈0.105, a rough normal-approximation paired planning formula `n≈(1.96+0.84)² q / δ²` gives ~330 independent tasks for a 5-point effect and ~2,060 for 2 points at 80% power. These are not exact binomial power calculations and omit multiplicity, family dependence and changed discordance; simulate the registered design before spending quota. A 60–95-task screen will often be inconclusive.

## 7. Serving semantics, context and LT1

**VERIFIED (T, P2):** NF4 training quantizes bf16 QAT weights; the scorer serves a W4A16 export. These are different numeric bases. Trainkit targets actual text linears. On global layers HF reuses `k_proj(x)` as V, so a K adapter affects both K and V in training. The raw export has no global V factors; vLLM's separate packed V slice receives base weights but no corresponding LoRA delta. `kvdup.py` can synthesize 20 extra factors, producing 420 modules/840 tensors, but LP1/LP2 led to turning this copy off by default. A stale G1 experiment-row description says “kv copy”; newer decisions, export implementation and G1 build describe **raw**. Use raw for the reference arm and record this discrepancy rather than copying the stale specification.

**HYPOTHESIS:** A future attention-only training run should explicitly match intended served global K-only semantics or separately train K and V, with clear additional parameter accounting. Changing this during the first ablation would confound family selection. LP1 supports validation-loss transfer; it does not show that NF4/W4A16 mismatch is universally harmless, nor that “fewer rounding errors” explains the base-loss difference. Test decisions on W4A16 before making deployment claims.

**VERIFIED (L, recomputed from 36 records):**

| Prefix tokens | Base decode tok/s | Full T1 | Attention only | Full/base time ratio at equal token count |
|---:|---:|---:|---:|---:|
| 2,048 | 38.63 | 34.12 | 34.27 | 1.132 |
| 8,192 | 36.28 | 32.48 | 32.53 | 1.117 |
| 16,384 | 34.83 | 31.25 | 31.30 | 1.115 |
| 24,000 | 32.85 | 29.75 | 29.78 | 1.104 |

The last column is base rate / full rate, equivalently **full/base time**. Prefill time ratio full/base is 1.026–1.034. Each record contains a one-token prefill call followed by a cached-prefix 256-token decode request; 36 records mean 72 timed HTTP requests, plus warmups. Lengths are prefixes of one validation episode, three repeats, nonce tokens and randomized arm order. “Decode” includes request overhead and cached-prefix processing, not isolated kernel time. No behavioral attention-only score was measured.

**INFERENCE:** LT1 is consistent with a substantial fixed serving-path cost and does not support proportional speedup from deleting 180 MLP modules. The exact causal explanation “Punica rank padded to 128” is not independently established by these timing rows; max_lora_rank=128 is configured, but no kernel profiler or rank/engine intervention is supplied. MLP-only, selective-layer, rank-8 and merged/requantized performance remain unmeasured. Do not promise any of them will be faster.

HQ's +0.8 h /120-task estimate uses ~7.1k completion tokens per task and ~35.6→31.9 tok/s: `120×7100×(1/31.9−1/35.6)/3600≈0.77 h`, before a small prefill increment. This is a workload projection, not measured leaderboard runtime. The reported v18 timeout means an unchanged deployment is unsafe to assume viable, but it does **not** prove v14's true runtime was >11.2 h: different hidden trajectories, infrastructure and unobserved overhead remain alternatives. At equal workload, ~10% fewer decode tokens offsets ~1.11× slowdown; 10–15% is a reasonable design target, not demonstrated repair preservation.

Context overflow is similarly coupled to behavior: a shared 24,576-token prompt ceiling can still be reached at different rates because adapters choose different actions/tool outputs. Hold context and output reservation fixed in attribution experiments. Training concise decisions or testing a lower thinking budget is a separate intervention. Do not conflate reducing thinking with reducing required max-output reservation, or silently change current competition constraints.

## 8. Selective fine-tuning after causal screening

**HYPOTHESIS:** If A preserves useful fixes while removing M improves held-out outcomes, attention-only training is a candidate; the symmetric inference applies to M. If F exceeds both, preserve cross-family interaction. If only late-layer removal helps, test that specific group on independent tasks before freezing it. Ablation of an already co-trained adapter does not predict training a restricted adapter from scratch: the retained modules may have co-adapted and could learn differently under a new objective.

A controlled follow-up should compare fresh full-target and selected-target LoRA from the same base, with the same data, seed set, per-token weights, training steps, optimization schedule and rank/scale on retained modules. Include a same-size random selection control. Report the changed total trainable capacity. A second explicitly capacity-matched design can redistribute rank; do not quietly mix it with the fixed-rank comparison. Starting from T1 and selectively continuing is another condition, not equivalent to from-base training. Preserve full exports and validate both step25 and final as exploratory checkpoints; never choose the best on the final test set.

Training examples must teach the desired decisions:

- Realistic on-policy states with source grounded edits and verifiable successful outcomes, including unfamiliar repositories and multi-site dependencies; balance size and difficulty rather than simply upweight every multi-file example.
- Recovery from failed tests, incorrect assumptions and incomplete patches, with a correct continuation available from that exact state. Failed traces are context/contrast examples, not positive imitation targets.
- Matched correct/incorrect edit or action alternatives, verified by task tests, for a future preference/corrective objective. Hold the objective fixed when testing module selection; otherwise improved data and changed modules are inseparable.
- Concise useful reasoning, bounded tool output, relevant targeted tests and timely submission, judged by actual outcomes. Avoid arbitrary deletion of reasoning or forcing early edits; prior failure modes show style imitation and premature verification can harm repair.
- Long-context recovery and termination examples below the serving ceiling, not only successful episodes up to a training max of 28,672. This must not expose held-out tests or reward submitting an incorrect patch just to avoid timeout.

Before new training, repair/evaluate RS2 span alignment on all 11 mismatch cases, audit every masked harmful action and multi-call message, and distinguish testing after an attempted edit from testing after a successful mutation. Keep full context while masking unsuitable target tokens. Use source-function/repository-disjoint confirmation data and measure patch correctness, not just loss. Selective training may reduce optimizer state and overfitting, but frozen layers may still need backward propagation to reach earlier trainable layers; compute and memory savings are not proportional to trainable parameter count.

## 9. Resources and work possible without GPUs

**HYPOTHESIS — Planning estimates, not approvals or current account balances:** Use the known 4×L4/TP4 serving stack and T1's 4×L4 training setup. Distinguish physical GPU-hours (4× wall hours) from the historical account quota convention (~2× wall hours in project logs); quota policy and actual balance must be checked before any future run.

| Work | GPU need | Approximate planning envelope |
|---|---|---|
| Header/hash/mapping, spectra, task joins, statistics, audit masks | None | CPU hours; a few GB RAM for streamed adapter analysis; model base not needed |
| Reproduce LT1-like loading/timing controls | 4×L4 inference | Recorded LT1 0.35 wall h; allow 0.5–1 wall h for additional masks/checks, 2–4 physical GPU-h |
| Four-arm, 60-task concurrency-1 behavioral screen | 4×L4 inference | 240 episodes ×231 s /3600 ×1.1 ≈16.9 wall h agent work; allow ~18–24 wall h including setup/verification and variability, 72–96 physical GPU-h; historical quota equivalent ~36–48 h |
| Four-arm, 95-task screen at G1 concurrency 6 | 4×L4 inference | Twice G1's 3.7 h ≈7.4 wall h; allow 8–10 wall h, 32–40 physical GPU-h; ~16–20 historical quota-h; **not scorer-faithful latency evidence** |
| One T1-scale epoch | 4×L4 training | Roughly 2.5–3 wall h including load/validation/export; 10–12 physical GPU-h, ~5–6 historical quota-h |
| Matched two-condition two-epoch training | 4×L4 training per condition | About 2×4.8=9.6 wall h before extra controls/evaluation; ~38.4 physical GPU-h, ~19.2 historical quota-h |

The concurrency-1 estimates use historical v14 mean task time, not a guarantee for new tasks. A worst-case 240×8-minute cap alone is 32 wall hours before setup; plan separate bounded sessions only after explicit approval. A four-arm study is expensive and should not be smuggled into one previously authorized run. If resources only support a smaller paired pilot, designate it a feasibility/localization screen, not proof of benefit. New calibration activation collection and causal patching need forward passes and separate estimates; storing all activations across 60 layers and long contexts is unnecessary—capture selected module outputs at selected decisions.

**BLOCKED:** No current quota query, T1 adapter download, GPU availability check, licensing change or future-run permission was performed or assumed. CPU-only preflight can fully validate source masks/manifests, partitions, arithmetic and proposed metrics, but cannot certify served loading/logit identity, runtime, or bug-fixing outcomes.

## 10. Single next experiment and decision criteria

**HYPOTHESIS — Recommend FAC-1: frozen-T1 family factorial, no retraining.** Register one four-arm study (B/F/A/M) on 60 fresh gold-valid, source-family-disjoint tasks, stratified across repositories, singles/coupled edits and context lengths, with one matched seed for the initial screen. Use the same v14 policy, 8-minute/28-call budget and concurrency 1 in every arm. G1/new-60 remain historical/exploratory regression evidence rather than being called fresh confirmation. First complete CPU provenance/mask checks and separately authorized GPU identity checks. Do not proceed if the exact T1 artifact or intended serving semantics cannot be authenticated.

Primary scientific question: is either family beneficial or harmful conditional on the other, and is their joint effect non-additive? Compute all four paired contrasts and the interaction. Primary deployment candidate rule, prespecified before outcomes:

1. A reduced arm must improve over full by **at least 5 percentage points** in resolved rate, with a multiplicity-adjusted 95% paired interval above zero for that contrast, and must meet a **−3 percentage-point noninferiority margin versus base** (one-sided adjusted interval lower bound >−3 pp). At n=60 these are demanding; failure to certify is usually inconclusive, not equivalence.
2. It must not increase overflow/agent-timeout incidence by more than a prespecified +3 pp safety margin, assessed with upper confidence bounds; any serious tool/schema or verification regression blocks advancement pending explanation. Report uncertainty even if point estimates look safe.
3. It must show a credible end-to-end runtime offset: target ≥10% fewer completion tokens than full with preserved correctness, and measured total time including tool/testing overhead consistent with an explicitly reserved deployment margin. LT1 alone cannot certify a 12-hour hidden workload. Without that margin, a scientific attribution result can succeed while deployment remains blocked.
4. Repeat the selected comparison on untouched tasks and additional seeds before selective retraining or deployment claims. Do not use the same 60 outcomes to select a layer group and confirm it.

**Scientific success:** identity controls pass and a stable, practically meaningful family contrast/interaction replicates with task-level uncertainty accounted for—even if the causal finding is that a component harms performance. **Deployment success:** the stricter accuracy, regression and runtime criteria above all hold on confirmation. **Failure:** invalid mask/loading controls, missing matched coverage, consistent material damage, or inability to fit runtime. **Inconclusive:** valid study with intervals spanning useful benefit and harm; keep the base and do not escalate to a broad layer/weight search. If full, A and M all fail to beat base meaningfully, return to training-example quality and objective design rather than selecting the largest-norm layers.

Priority actions: Atul should finalize the tensor provenance and CPU analysis schema and preregister task/statistical controls. Shrey should supply the exact T1 checkpoint hashes and serving artifact/config evidence, then assess quota for FAC-1. Neither role needs to edit model weights or launch a job for this report. The immediate next action is **authenticate T1 step25/final tensors and the served raw adapter**, enabling the offline spectrum analysis and future mask identity tests.

## Appendix A. Per-layer expansion of all 410 modules

`q/k/v/o` abbreviate the four `self_attn.*_proj` families; `gate/up/down` abbreviate `mlp.*_proj`. Apply the canonical prefix in §3 to every entry. This is an exhaustive architectural map, not an importance ranking. Counts are independently checked against older S1/S3 headers; exact T1 keys remain awaiting its artifact.

| Layer | Attention type | Adapted attention | Adapted MLP | Modules |
|---:|---|---|---|---:|
| 0 | sliding | q, k, v, o | gate, up, down | 7 |
| 1 | sliding | q, k, v, o | gate, up, down | 7 |
| 2 | sliding | q, k, v, o | gate, up, down | 7 |
| 3 | sliding | q, k, v, o | gate, up, down | 7 |
| 4 | sliding | q, k, v, o | gate, up, down | 7 |
| 5 | global | q, k, o | gate, up, down | 6 |
| 6 | sliding | q, k, v, o | gate, up, down | 7 |
| 7 | sliding | q, k, v, o | gate, up, down | 7 |
| 8 | sliding | q, k, v, o | gate, up, down | 7 |
| 9 | sliding | q, k, v, o | gate, up, down | 7 |
| 10 | sliding | q, k, v, o | gate, up, down | 7 |
| 11 | global | q, k, o | gate, up, down | 6 |
| 12 | sliding | q, k, v, o | gate, up, down | 7 |
| 13 | sliding | q, k, v, o | gate, up, down | 7 |
| 14 | sliding | q, k, v, o | gate, up, down | 7 |
| 15 | sliding | q, k, v, o | gate, up, down | 7 |
| 16 | sliding | q, k, v, o | gate, up, down | 7 |
| 17 | global | q, k, o | gate, up, down | 6 |
| 18 | sliding | q, k, v, o | gate, up, down | 7 |
| 19 | sliding | q, k, v, o | gate, up, down | 7 |
| 20 | sliding | q, k, v, o | gate, up, down | 7 |
| 21 | sliding | q, k, v, o | gate, up, down | 7 |
| 22 | sliding | q, k, v, o | gate, up, down | 7 |
| 23 | global | q, k, o | gate, up, down | 6 |
| 24 | sliding | q, k, v, o | gate, up, down | 7 |
| 25 | sliding | q, k, v, o | gate, up, down | 7 |
| 26 | sliding | q, k, v, o | gate, up, down | 7 |
| 27 | sliding | q, k, v, o | gate, up, down | 7 |
| 28 | sliding | q, k, v, o | gate, up, down | 7 |
| 29 | global | q, k, o | gate, up, down | 6 |
| 30 | sliding | q, k, v, o | gate, up, down | 7 |
| 31 | sliding | q, k, v, o | gate, up, down | 7 |
| 32 | sliding | q, k, v, o | gate, up, down | 7 |
| 33 | sliding | q, k, v, o | gate, up, down | 7 |
| 34 | sliding | q, k, v, o | gate, up, down | 7 |
| 35 | global | q, k, o | gate, up, down | 6 |
| 36 | sliding | q, k, v, o | gate, up, down | 7 |
| 37 | sliding | q, k, v, o | gate, up, down | 7 |
| 38 | sliding | q, k, v, o | gate, up, down | 7 |
| 39 | sliding | q, k, v, o | gate, up, down | 7 |
| 40 | sliding | q, k, v, o | gate, up, down | 7 |
| 41 | global | q, k, o | gate, up, down | 6 |
| 42 | sliding | q, k, v, o | gate, up, down | 7 |
| 43 | sliding | q, k, v, o | gate, up, down | 7 |
| 44 | sliding | q, k, v, o | gate, up, down | 7 |
| 45 | sliding | q, k, v, o | gate, up, down | 7 |
| 46 | sliding | q, k, v, o | gate, up, down | 7 |
| 47 | global | q, k, o | gate, up, down | 6 |
| 48 | sliding | q, k, v, o | gate, up, down | 7 |
| 49 | sliding | q, k, v, o | gate, up, down | 7 |
| 50 | sliding | q, k, v, o | gate, up, down | 7 |
| 51 | sliding | q, k, v, o | gate, up, down | 7 |
| 52 | sliding | q, k, v, o | gate, up, down | 7 |
| 53 | global | q, k, o | gate, up, down | 6 |
| 54 | sliding | q, k, v, o | gate, up, down | 7 |
| 55 | sliding | q, k, v, o | gate, up, down | 7 |
| 56 | sliding | q, k, v, o | gate, up, down | 7 |
| 57 | sliding | q, k, v, o | gate, up, down | 7 |
| 58 | sliding | q, k, v, o | gate, up, down | 7 |
| 59 | global | q, k, o | gate, up, down | 6 |

## Appendix B. CPU validation and preservation record

**VERIFIED:** Recomputed G1 paired counts, exact McNemar p-values and overflow counts from task rows; recomputed all LT1 median decode rates from 36 records; checked both S1/S3 820-tensor shape inventories and file bounds; verified module-count and scalar-count arithmetic. This is an evidence audit, not a new model experiment. No numeric T1 tensor analysis was claimed or run.

S1 header-source SHA-256: `c41939e731011b0bc6fb5ab222c6e84f6f4a1512301201fb14bbe230d6c21040`. Relevant remote implementation/results were checked against the pinned Git tree. The new report contains aggregate outcomes, architecture dimensions and experimental design, not task bodies, patches, fixture data or credentials.

A baseline SHA-256 inventory covered 2,272 existing tracked/evidence files before report creation. The final SHA-256 check matched every entry (0 changed/missing). Git diff --check passed; Git status showed only this report added beyond the pre-existing untracked R2 directory. Temporary API copies and validation metadata live outside the repository in `/tmp/atul-weight-audit/` and are not research-result dependencies. Existing documents, datasets, notebooks, model weights and Git history remain unchanged.
