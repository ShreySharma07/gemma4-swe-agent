# T1 LoRA effective-weight analysis — 4 November 2026

## 1. Findings and evidence labels

**VERIFIED:** All three user-supplied T1 files passed structural, checksum, configuration, mapping and finite-value checks. Final and serving adapters are byte-identical. The previous research report's T1 tensor-availability blocker is now resolved; that historical report remains unchanged.

**VERIFIED:** Final effective updates are largest in late layers, particularly layer 59's MLP up and gate projections. The five equal-size depth groups put layers 48–59 first by both total squared Frobenius norm and pooled RMS. Gate/up together account for 61.86% of final update energy. This is a size-aware descriptive result, **not** a causal importance score.

**VERIFIED:** All 410 module update norms increased from step 25 to final. The norm of the concatenated effective updates grew 9.87%, with cosine 0.98218 between checkpoints. The second-epoch increment has norm 22.11% of the step-25 update. Early-middle layers 12–23 changed more proportionally; late layers have larger but directionally more stable updates.

**INFERENCE:** Useful future causal candidates should cover contrasting properties: large concentrated late MLP updates, proportionally changing attention-output modules, and high-effective-rank early-middle modules. Selecting only the largest norms would miss this diversity.

**BLOCKED:** Base-relative update magnitude, activation-weighted influence, causal bug-fixing importance, behavioral effects of deleting modules, and whether final is behaviorally better than step25 remain unmeasured. No base-model weights, activations, inference or new task outcomes were used.

Labels: VERIFIED means directly checked artifact facts or numerical calculations; INFERENCE means interpretation; HYPOTHESIS denotes future interventions; BLOCKED identifies missing evidence. These are exact descriptive statistics of two checkpoints, not random-sample behavioral estimates. No significance tests across the 410 correlated modules are used.

## 2. Provenance and integrity

Repository inspected at HEAD `0e5e5d14371df6b2c95dc0c1267d2db9686b3778`. Read the prior `research_deliverables/atul_gemma4_weight_attribution_research.md`, current `docs/HQ.md`, `docs/experiments.md`, repository `kaggle/train/trainkit/config.py` and `export.py`, and downloaded `train_cfg.json` and adapter JSON configurations. Downloaded Python/bytecode was **never executed or imported**. Only our newly authored analysis scripts were executed. The installed dependency is NumPy 2.5.4 in an isolated `/tmp/atul-t1-cpu-env`; the existing repository environment was not changed.

Checkpoint root: `/home/atul-baliyan/Desktop/gemma4-t1-weights/`.

| Checkpoint | Relative path | Bytes | SHA-256 |
|---|---|---:|---|
| step25 | `train_out/adapter_step25/adapter_model.safetensors` | 244,982,352 | `3804c9260680c1c110372d3b3fa75f2c9b27e8ca6319fd9c6daf4658e10d751b` |
| final | `train_out/adapter_final/adapter_model.safetensors` | 244,982,352 | `2b22c3f393f178f0c8885df99229a501da826ec1150bda83d8bfd7f4d3d56510` |
| serving | `adapters/rs2_lora/adapter_model.safetensors` | 244,982,352 | `2b22c3f393f178f0c8885df99229a501da826ec1150bda83d8bfd7f4d3d56510` |

Each file is 233.633 MiB: 8-byte length prefix, 123,464-byte JSON header and 244,858,880-byte tensor payload. Checks reject duplicate JSON keys, invalid/oversized header, wrong names/shapes/dtypes, overlapping or gapped tensor ranges, out-of-bounds offsets, missing/extra modules and trailing payload bytes. All three contain exactly 820 BF16 factors, 122,429,440 finite scalar values, and no zero-valued factor entries. Absence of zero entries is an observation, not a training-health requirement. SHA-256 matches the user's independently supplied values. Final/serving equality was also checked by streaming byte comparison, independently of the hash comparison. This authenticates the supplied bytes against those reference hashes; it does not independently reconstruct the remote download chain.

All three adapter configurations are byte-identical (SHA-256 `6b0a7255228061cb6f69a91baff7d8e0c7ec27e090ddd6ef2f9a0b0c73d2d5cc`), with rank 16, alpha 32, scale 2, dropout 0, no bias, `fan_in_fan_out=false`, standard LoRA, and the seven expected target families. No rank/alpha patterns, DoRA, rsLoRA or modules-to-save overrides are active. The base path identifies `gemma-4-31b-it-qat-q4_0-unquantized/2`. Downloaded training config specifies RS2, lr 0.0001, two epochs, accumulation 2, max length 28,672, saves/validation every 25 steps, and no smoke/probe mode. Rank/alpha omitted from that runtime JSON are supplied by the repository defaults and corroborated by the exports.

**VERIFIED mapping:** text layers 0–59, with global layers 5,11,17,23,29,35,41,47,53,59. Every layer has q/k/o and gate/up/down; sliding layers also have v. Exactly 230 attention + 180 MLP modules = 410; no global v factors are present. Full keys are retained in numerical CSVs. Canonical prefix is `base_model.model.model.language_model.layers.{layer}.`, followed by `self_attn.{family}` or `mlp.{family}`.

Shape checks cover every tensor. A is rank×input and B is output×rank. Hidden width is 5376; MLP width 21504. Sliding q/o width is 8192 and k/v output width 4096; global q/o width is 16384 and k output width 2048. Thus the expected scalar counts and 820-factor architecture are independently confirmed on actual T1, rather than inferred from S1/S3.

No integrity discrepancy required stopping. Global V omission is expected raw-export behavior, not a corrupt or incomplete checkpoint.

## 3. Numerical methods and definitions

All calculations use CPU NumPy, with one BLAS thread and CUDA visibility disabled. Safetensors payloads are memory-mapped read-only, BF16 bits are expanded exactly into fp32 and then fp64. One module's factors are processed at a time. No 31B model is instantiated; no transformer forward or backward pass is run; no raw tensors are saved.

For each module, `U = ΔW = 2 BA`. We compute thin QR decompositions `B=Qb Rb`, `Aᵀ=Qa Ra`. The at-most-16 nonzero singular values of U are the singular values of the small matrix `2 Rb Raᵀ`. We do not form its full output×input matrix. Metrics:

- Energy = `||U||F² = Σσ²`; Frobenius = its square root; spectral norm = largest σ.
- Stable rank = `Σσ² / σ1²`.
- Energy-entropy effective rank = `exp(-Σp log p)`, `p=σ²/Σσ²`. This matches the prior research protocol. Because “effective rank” also commonly uses normalized singular values instead of squared values, `supplementary_metrics.csv` additionally supplies that explicitly named alternative. Do not mix the two conventions.
- Update RMS = `||U||F/sqrt(output×input)`. This normalizes by the effective dense matrix size, **not** by the number of LoRA factors. It is not a ratio to the base weights.
- Top-1/4/8 energy fractions and the minimum number of singular components required for 90%/99% energy. Full 16-value singular spectra and energy fractions are saved as numerical statistics.
- **Relative update RMS:** the final/step25 RMS ratio equals the same-module Frobenius ratio and is supplied. `RMS(U)/RMS(W_base)` is explicitly BLOCKED: the matching base matrices were not provided. No denominator is invented or borrowed from another quantization.

Checkpoint differences are computed in effective-update space: `D=Ufinal−U25 = 2 [Bfinal B25] [Afinal;−A25]`. Its spectrum uses a maximum 32×32 QR core. Cosine uses the factor cross-Gram identity for `<U25,Ufinal>`; comparisons include difference norm, RMS, spectral norm, stable/effective rank and relative change `||D||F/||U25||F`. Independent Gram identities validate the results. We do not compare raw A/B coordinates as if they were uniquely meaningful: reciprocal basis rotations can leave BA unchanged.

Aggregated energy is the sum over modules; its square root is the norm of a **concatenated list of matrices**, not the spectral norm of the transformer. Group pooled RMS is `sqrt(sum energy / sum dense entries)`, while group median RMS weights modules equally. Depth groups are five 12-layer blocks, each with 82 modules and the same projection-type mix. Global/sliding comparisons are additionally stratified by projection because shapes and family membership differ. Cross-layer energy comparisons still lack activation/normalization context; dimension normalization does not make them causal.

## 4. Layer and family distributions

**VERIFIED — final checkpoint:**

| Family | Modules | Energy | Share | Pooled update RMS | Median stable rank | Median energy effective rank |
|---|---:|---:|---:|---:|---:|---:|
| gate_proj | 60 | 8.837453 | 31.17% | 3.56943e-05 | 3.469 | 11.163 |
| up_proj | 60 | 8.701086 | 30.69% | 3.54178e-05 | 3.369 | 11.243 |
| q_proj | 60 | 3.707807 | 13.08% | 3.46805e-05 | 3.679 | 11.415 |
| down_proj | 60 | 2.175926 | 7.68% | 1.77116e-05 | 2.798 | 8.945 |
| o_proj | 60 | 2.113656 | 7.46% | 2.61845e-05 | 2.511 | 7.989 |
| k_proj | 60 | 1.461668 | 5.16% | 3.47403e-05 | 3.581 | 10.941 |
| v_proj | 50 | 1.352488 | 4.77% | 3.50487e-05 | 2.511 | 8.232 |

MLP carries 69.54% of total update energy versus attention's 30.46%, but MLP represents larger matrices. Pooled attention RMS is **3.19159e-5**, slightly above MLP **3.07799e-5**. Gate/up lead both absolute energy and family-level RMS; down projection is much smaller on RMS despite the same matrix element count as gate/up. This supports a descriptive family difference, not “MLPs perform 70% of bug fixing.”

| Layer block | Modules | Final energy | Share | Pooled RMS | Median relative change from step25 | Median checkpoint cosine |
|---|---:|---:|---:|---:|---:|---:|
| 00-11 | 82 | 4.851052 | 17.11% | 2.87785e-05 | 23.72% | 0.979458 |
| 12-23 | 82 | 4.644813 | 16.38% | 2.81601e-05 | 25.40% | 0.975207 |
| 24-35 | 82 | 5.263760 | 18.57% | 2.99777e-05 | 22.32% | 0.981971 |
| 36-47 | 82 | 6.123172 | 21.60% | 3.23324e-05 | 21.45% | 0.983394 |
| 48-59 | 82 | 7.467287 | 26.34% | 3.57052e-05 | 18.60% | 0.987490 |

Layers **48–59** lead in energy and RMS, followed by **36–47**. The highest-energy individual layers are **59, 56, 58, 57, 55**. All 60 layer rows for both checkpoints are in `groups.csv`; all 410 exact module identities and measurements are in `modules.csv`.

### Global versus sliding attention

Global attention comprises 30 adapted modules (no v), sliding attention 200. Their total energies are 1.576475 and 7.059144 respectively; this count-confounded comparison alone is not informative. Pooled RMS is 2.90218e-5 global versus 3.26893e-5 sliding.

| Family | Global pooled RMS | Sliding pooled RMS | Global / sliding median stable rank |
|---|---:|---:|---|
| q_proj | 3.48652e-05 | 3.46063e-05 | 4.153 / 3.599 |
| k_proj | 3.5702e-05 | 3.46427e-05 | 2.818 / 3.714 |
| o_proj | 2.03692e-05 | 2.81765e-05 | 2.082 / 2.722 |

Global q/k RMS is close to or slightly above sliding q/k, whereas global o RMS is lower. Thus the pooled global deficit is not uniform across families. Sliding v RMS is 3.50487e-5; no global-v numerical comparison is available because no independent factor exists. Global K is counted once as a learned matrix, not twice for its two HF usages.

## 5. Step25 to final: growth, stability and dimensionality

| Concatenated-update measure | Value |
|---|---:|
| Step25 energy | 23.48361180 |
| Final energy | 28.35008383 |
| Step25 Frobenius norm | 4.84598925 |
| Final Frobenius norm | 5.32447968 |
| Norm growth (%) | 9.87394737 |
| Energy growth (%) | 20.72284311 |
| Difference Frobenius norm | 1.07164831 |
| Difference / step25 norm | 0.22114129 |
| Checkpoint cosine | 0.98218232 |

**VERIFIED:** Every module's norm increased. Individual checkpoint cosines range **0.963514–0.991027**, and relative difference norms **15.39–33.08%**. High alignment does not mean the increment is negligible: growth and modest direction change together yield a 22.11% overall difference relative to step25.

Gate and up contribute 31.24% and 29.86% of the **difference energy**; q contributes 13.69%. This is distinct from `final energy − step25 energy`: the latter includes cross terms. Family median relative changes are close (~21.9–22.9%), with q the largest and v the smallest. The largest relative changes occur in **layer 11 o (33.08%), layer 17 o (32.13%), layer 19 o (30.52%)**. Layer 20 o has the lowest cosine (0.963514). Late MLP updates dominate absolute increment size, but layers 12–23 have the largest block median proportional increment (25.40%).

Final module median stable rank is **3.193** (range 1.175–8.243); median energy effective rank is **10.181** (range 1.973–15.127). Median top-1/4/8 energy shares are **31.32%, 61.57%, 80.45%**. Median components needed for 90% energy is **12**, and for 99% is **16**. Low stable rank does not imply the rank-16 adapter can safely be replaced by rank 3: the remaining spectrum retains considerable energy, and neither energy retention nor numerical rank certifies behavioral preservation.

At layer 59, up/gate top singular directions carry **73.97% / 74.97%** of energy, with effective ranks **3.086 / 3.027**. In contrast, layer 12 gate has effective rank **15.127**, stable rank **8.243**, and only **12.13%** top-direction energy. Large updates and high-dimensional updates are different phenomena. Layer 4 K also has high effective rank **14.770**, providing an attention-family contrast.

**INFERENCE:** The second epoch largely strengthens an existing update direction rather than replacing it wholesale. It is not numerically stationary and not proof of “stable reasoning.” T1 validation loss changed only 0.0669→0.0666 in that epoch; we cannot infer whether the added update improves repair, memorizes formatting, or introduces regressions. The same input tensors do not establish the behavior of step25, which has not been evaluated here.

## 6. Candidate modules for future causal experiments

**HYPOTHESIS — post-hoc numerical screening, not confirmed importance:** Start with the prior report's base/full/attention-only/MLP-only factorial before an expensive 410-way scan. Then preregister a small contrastive module set. All layer indices below are zero-based and expand under the canonical prefix in §2.

| Candidate | Numerical reason to include | Required interpretation/control |
|---|---|---|
| `layers.59.mlp.up_proj` | Largest Frobenius 0.522781, spectral 0.449613, RMS 4.86218e-5; concentrated spectrum | Test removal and restoration; compare matched MLP module and broad code/tool regressions |
| `layers.59.mlp.gate_proj` | Second-largest norm 0.510907; largest absolute checkpoint increment 0.116989 | Test alone and jointly with up; their gated-MLP interaction can be non-additive |
| `layers.11.self_attn.o_proj` and `layers.17.self_attn.o_proj` | Largest proportional checkpoint changes, 33.08% and 32.13%; both global | Compare same-family/type controls; test step25 versus final delta in a follow-up |
| `layers.20.self_attn.o_proj` | Largest direction change (lowest cosine 0.963514); sliding | Separates directional change from high absolute size and global attention |
| `layers.12.mlp.gate_proj` | Highest energy effective rank 15.127 | Contrast broad spectrum with concentrated layer-59 gate; neither is presumed better |
| `layers.4.self_attn.k_proj` | High-dimensional attention update, effective rank 14.770 | Sliding-K contrast; include matched K control |
| `layers.59.self_attn.k_proj` | High normalized RMS 4.23596e-5, but global K/V serving complication | Lower-priority semantic control, not a direct comparison with an invented global V module |

Late block **48–59** and higher-change block **12–23** are useful group-level contrasts; use equally sized matched-random groups and keep the full adapter rescue. These candidates were selected after inspecting weight statistics; there is no preregistered causal discovery claim. Freeze the candidate list before behavioral outcomes are collected and correct for multiple comparisons.

Suppress only the learned delta, never the base matrix. Evaluate paired identical tasks/settings, final patch correctness, base-solved losses, overflow and runtime; use the full adapter, genuine no-adapter base and a zero-delta-loaded control. Activation patching on fixed decision contexts would be a later mechanism test. No masks, replacement tensors, ablated checkpoints or inference were produced in this analysis.

## 7. Limitations and anomalies

1. **BLOCKED base-relative scale:** update RMS is dimension normalized, not relative to the actual base matrix norm. No matching base weights or activation covariances were supplied. This is the main incomplete requested metric if “relative RMS” is meant relative to the base.
2. **VERIFIED export identity, not forward identity:** final and serving files are identical, but NF4/HF and W4A16/vLLM are different computation paths. On global layers HF K deltas also influence V; raw vLLM export leaves V without that delta. Our statistics characterize the learned matrix once, not its task-conditional functional effect in either stack.
3. **INFERENCE runtime caution:** LT1 found almost identical overhead for attention-only and full adapters. These norm/spectrum results provide no reason to expect proportional speedup from module deletion. They cannot establish a safe rank reduction or merge/requantization recipe.
4. **BLOCKED behavioral attribution:** G1's 42/95 versus 40/95 is inconclusive (paired p≈0.754). No layer-specific success labels, gradients, activations, or new behavioral outcomes enter this study. A high cosine or large norm cannot certify bug-fixing competence, harmfulness or a “reasoning circuit.”
5. **VERIFIED numerical scope:** results describe BF16-exported factors multiplied in fp64, not their pre-export fp32 master values or actual bf16 kernel accumulation. Checkpoint differences include export quantization. No NaN/Inf, unexpected module, structural corruption or hash discrepancy was found.
6. **INFERENCE dimensional caveat:** singular vectors are not emitted and individual LoRA columns are not interpreted. Rank statistics depend on the stated definition, and co-adapted nonlinear components can have large joint effects despite small independent norms. Layer 59's dominant direction does not identify a head, feature or human-readable algorithm.

## 8. Reproducibility, validation and preservation

Supporting directory: `research_deliverables/atul_t1_lora_analysis_2026-11-04/`.

- `analyze.py`: independently authored strict parser, CPU QR/SVD analysis and numerical checks; no downloaded-code imports.
- `integrity.json`: exact paths, sizes, hashes, configurations and validation outcomes.
- `modules.csv`: **820 module-checkpoint rows** (410 per checkpoint), all shape-normalized metrics. Serving is not double-counted because it equals final.
- `spectra.json`: **820 numerical spectra**, not factors or model tensors.
- `checkpoint_comparison.csv`: **410 effective-update comparisons**.
- `groups.csv`: both checkpoints' family, component, layer, equal-depth-block and type aggregates; `attention_type` on MLP rows refers to the containing layer, not attention tensors.
- `supplementary_metrics.csv`: singular-entropy effective rank and final/step25 RMS ratio; the ratio is a pair property repeated on each checkpoint row.
- `summary.json`, `summarize.py`: derived rankings/aggregates and their reproducible construction.
- `validation.json`, `independent_checks.json`: numeric tolerances, synthetic controls and actual-weight slice/gauge checks.
- `provenance.json`, `preservation.json`, `output_manifest.json`: source hashes/environment, protected-file checks and hashes of the new deliverables (manifest excludes itself).

Reproduction uses a fresh output directory because the scripts refuse to overwrite their outputs:

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/atul-t1-cpu-env/bin/python research_deliverables/atul_t1_lora_analysis_2026-11-04/analyze.py --integrity-only --out /tmp/t1-reproduction-new
PYTHONDONTWRITEBYTECODE=1 /tmp/atul-t1-cpu-env/bin/python research_deliverables/atul_t1_lora_analysis_2026-11-04/analyze.py --out /tmp/t1-reproduction-new
```

`summarize.py` reads outputs beside itself; to reproduce summaries elsewhere, copy this **authored** script and `analyze.py` into the fresh output directory and run it there. NumPy 2.5.4 and Python 3.12.3 were used. No downloaded training/evaluation script is needed.

**VERIFIED numerical validation:** Random, rank-deficient and zero synthetic cases match dense SVD. All 820 checkpoint-module norms match independent Gram calculations; all 410 difference energies match cross-Gram identities. Twenty-one small real-weight slices match explicit rank-one accumulation. Three further real-weight rectangular slices match dense SVD, and orthogonal factor-basis rotations preserve the full effective spectra. Dense validation was restricted to tiny matrices; no full module update or model was materialized.

Maximum relative errors: synthetic SVD **2.45e-16**, Gram/Frobenius **8.88e-16**, checkpoint-difference energy **4.42e-14**. Main CPU analysis, including integrity recheck, took **12.69 seconds** on this host; this is not model-inference throughput.

The preservation audit hashes all 2,342 protected tracked/evidence/download files before and after the work; details are in `preservation.json`. Existing untracked R2 evidence is preserved. No Git staging, branch changes, commits, pushes or PRs occurred. No model training/inference, GPU/Kaggle job, weight modification or upload occurred. Only the new report and its numerical/code support directory are authorized repository additions.

**Recommended next action:** review the fixed candidate set and controlled family-ablation design with Shrey; do not selectively retrain or prune on the basis of these statistics alone.
