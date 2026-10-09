# Break of architecture v1 (GRAFT): methodology, statistics, compute (8 Oct 2026)

**Role:** Breaker 2 (methodology, statistics, compute). This file attacks `architecture_v1.md`. Background read in full: `docs/training_feasibility_4xl4.md`, `phase3_critic.md` and `docs/HQ.md`. I also checked `results/runs/*` and `kaggle/eval_l4/run_official_l4.py` for environment facts.

**Labels:**
- **VERIFIED**: read from repo files or computed from them in this pass.
- **INFERRED**: my own arithmetic or reasoning, or a citation from memory.
- **GUESSED**: a sizing number only.

**Severity:**
- **FATAL**: the central claim cannot be supported as designed.
- **MAJOR**: a reviewer would reject it, or a sub-claim is not identified.
- **MINOR**: it should be fixed, but the paper survives without the fix.

## Bottom line

1. **The ≤25 GPU-h budget is not credible as written.**
   - My recomputation gives **≈22 h optimistic, ≈35 h central and ≈58 h pessimistic** (§1.9).
   - Three lines carry most of the gap: training at the 250 tok/s gate floor, dev-guard wall time, and a reserve that is too small.
   - The dev-guard wall time for r3_v5 was **6.45 h sequential**; the plan budgets 2.5 h (VERIFIED).
   - The reserve is 4%, against a probe history of 8 failed versions.

2. **The experiment is cheap only because it is tiny, and that makes it underpowered.**
   - Each arm trains on about 15–40 successful trajectories plus about 20 auxiliary items.
   - With so little data, the most likely outcome is that no arm moves.
   - Then **H3 "passes" without telling us anything**: (b)≈(c) because nothing changed.
   - And **H1's failure gets misread as "content dominates"**.
   - The design has no positive control that tells "too little training" apart from "content dominates".

3. **The topology contrast is not cleanly identified.**
   - The auxiliary call-set items are in the eval format, and they differ in depth between arms.
   - Rejection-sampling equalisation reshapes each arm's difficulty distribution.
   - Repo size is recalibrated per arm.
   - The rewire in (c) breaks module locality.
   - So (b) vs (d), and to a lesser extent (b) vs (c), measure bundles, not topology.

4. **The statistics cannot deliver H3 and H4.**
   - There are only 8 clusters with about 37 items each. The design effect is plausibly 2.8–4.7, not 1.5–2.
   - There is one seed per arm.
   - Under those conditions the ±0.03 TOST is either impossible to pass or passes without telling us anything.

5. **"Near SOTA" is not measured at all.**
   - The only guard is one dev-resolve comparison against a *stored* run.
   - That run used a different environment, prompt and pipeline.
   - Only 54/128 dev tasks are gold-solvable in the scorer environment (VERIFIED, HQ Oct 7).

---

## 1. Budget realism, line by line

### 1.0 Facts the v1 budget missed (all VERIFIED unless marked)

**Weekly quota.**
- Kaggle GPU quota is **30 h/week**, shared with all main-track leaderboard work.
- 5.5 h remain until Oct 10.
- Nov 5 is the freeze, so about 3.5 weeks are left.
- The paper study cannot take the whole quota. Every main-track eval also costs 4–6.5 h.

**Probe history.** The probes took versions v1–v8 to reach GO, and **24.5 GPU-h were spent in one week, mostly on failures**. Budgeting a 1.1 h reserve (4%) ignores the project's own base rate.

**Wall time of dev runs (sum of `duration_seconds`, sequential, from `task_results.jsonl`):**

| Run | Agent time | Resolved | Median task |
|---|---|---|---|
| r1 | 4.57 h | 15/129 | — |
| r2 | 3.91 h | 26/129 | 93 s |
| **r3_v5** | **6.45 h** | **33/129** | **170 s** |

**Deployed configuration** (`run_official_l4.py` V5_FILES):
- A two-stage SequentialAgent: the localizer has run_command and graph tools, and **no read_file or edit tools**; the fixer edits.
- Sampling is temperature 0.2, top_p 0.95, top_k 40.
- `max_time_minutes: 4.5` is a **wall-clock** per-task budget.

**Measured serving numbers.** vLLM start is 480 s, memory is 18.9 GB/GPU at util 0.80, and single-stream decode is ~34 tok/s (Oct 7 probe).

### 1.1 Rollout rate (B2, B3, B5)

**Decode side (INFERRED).**
- A synthetic trajectory with ≤40 turns probably generates **2–4k output tokens**.
- On 4×L4 over PCIe (no NVLink, TP=4), batching 16 streams should give about 6–10× single-stream decode, so **≈200–340 tok/s aggregate**.
- Decode alone then allows ≈200–400 rollouts/h. Decode is not the binding term.

**KV cache (INFERRED, must be measured).**

| Layer type | KV per token |
|---|---|
| Sliding (×50) | 16 KB per layer |
| Global (×10) | 4 KB per layer (k=v) |

- If vLLM 0.19.1's hybrid allocator frees KV outside the 1,024-token window, a 12k sequence costs about 1.3 GB across TP4, so ≈30+ concurrent streams fit.
- If it does **not**, the sliding layers alone cost ~9.8 GB per 12k sequence, so only **≈4–5 concurrent streams** fit and the rate collapses to roughly single-stream × 4.
- The plan never checks this.
- **Fix:** add `num_running_seqs` and `gpu_cache_usage` at 16 streams to G1.

**Environment side (the real unknown).**
- C5 allows test suites of up to 60 s. Agents run pytest repeatedly.
- With 16 parallel CPU sandboxes on an L4×4 VM whose **vCPU count is unknown** (not in HQ), CPU contention can dominate. T4×2 machines have 4 vCPUs.
- **Fix:** record `nproc` in B1, and cap suite time at 10 s for training tasks.

**Sampling temperature (MAJOR for calibration).**
- C6 samples at T=1.0 with no top-k or min-p. Deployment uses 0.2 / top-p 0.95 / top-k 40.
- On an int4 model, untruncated T=1.0 over 40 turns gives more malformed tool calls and tail-token degeneracy (INFERRED).
- So zero-shot success and A4 (the outcome-variance band) are measured at a temperature that is never deployed.
- The stated reason, "keeps importance ratios valid", is moot: C8 recomputes π_A in the trainer anyway, and can apply the same temperature and top-k transform to its logits.
- **Fix:** sample at T=0.7, top_k=40, and apply the identical logit transform in the trainer.

**Verdict on B5.**
- 60/h central is plausible *if* the KV and CPU questions resolve well. 40/h is a fair floor.
- I keep 100 / 60 / 40 per hour as optimistic / central / pessimistic.
- The plan's B5 line is honest. Its problem is scientific (too few rollouts, §2–3), not arithmetic.

### 1.2 Session overheads

- Each session restores `scorer_env.tgz` (5.8 GB) and starts vLLM (480 s), or loads the FSDP trainer and converts W4A16 to HQQ.
- That is 0.15–0.35 h per session.
- I count about 11 sessions: B1/B2, B3, B4, round-1 generation, Stage A, round-2 generation, Stage B, B8, B9, B10 and B11.
- That is **≈2–3.5 h of pure overhead**, which v1 spreads implicitly at the low end.
- The HQQ conversion of 410 modules on every trainer load (CPU dequantise, then re-pack) is unbudgeted. Fix: cache the HQQ state dict as a Kaggle dataset, once.

### 1.3 Training (B7): is HQQ + FSDP + PEFT on Gemma-4 actually supported?

**The memory table is for the wrong path.**
- The feasibility doc's memory table and its FSDP path are for **bnb NF4 with `bnb_4bit_quant_storage=bf16`**. That is the documented way to make FSDP shard 4-bit weights.
- v1 makes **HQQ g32** the default. HQQ + FSDP in transformers and PEFT is **not a documented path** (INFERRED).
  - The Answer.AI fsdp_qlora repository had custom HQQ handling. HF `HqqConfig` + FSDP is not something I can confirm.
  - If HQQ's packed `W_q`, scale and zero end up as non-sharded buffers, each GPU holds a full ~15 GB int4 copy. Add ~10.6 GB of activation checkpoints at 16k and it **OOMs**.
- Matching the served grid exactly also needs HQQ's half-quadratic optimiser **bypassed**: scales are injected directly, with zero=8 and symmetric int4. That is custom code.

**Throughput.**
- Dense-mask SDPA on the 50 sliding layers, plus head_dim-512 global layers, adds about **+55–75% FLOPs at 16k** (INFERRED: 4·T²·d·H per layer, ×3 for forward, recompute and backward). It is about half that at 8k.
- The L4 is power-capped at 72 W, so sustained clocks fall below boost.
- 500 tok/s central is optimistic for 16k sequences.
- **G0 accepts 250 tok/s**, so the pessimistic column must use 250, not 400.

**Token count per arm (INFERRED from v1's own numbers):**

| Component | Tokens |
|---|---|
| Stage A | ≈0.35 M |
| Stage B: 80 round-2 trajectories × ~8k | ≈0.64 M |
| Stage B: items capped at 30% | ≈0.27 M |
| Stage B, ×2 epochs | ≈1.8 M forward-backward |
| π_A recompute (≈⅓ cost) | ≈0.9 M forward |
| **Total** | **≈2.4 M token-equivalents per arm** |

| Throughput | Training time (3 arms) | Plus loads | B7 total |
|---|---|---|---|
| 650 tok/s | 3.1 h | +0.6 | 3.7 |
| 500 tok/s | 4.1 h | +0.6 | 4.7 |
| 250 tok/s | 8.2 h | +0.6 | 8.8 |

v1 has 4.2 / 5.0.

**Also note:** the v1 plan generates 200 × K=4 item samples per arm per round (B6), then the ≤30% token cap lets only ≈50 of the 800 item sequences into Stage B. Most of B6 is wasted.

**Severity: MAJOR.** Fix:
- Make G0 test the **exact** path: HQQ-sharded FSDP, 16k, backward through the head_dim-512 layers.
- Pre-register NF4 + measured mismatch as the fallback if HQQ + FSDP does not shard.
- Budget B7 at 250 tok/s in the pessimistic column.

### 1.4 Item evaluation (B4, B8)

**Base model, per item, decode tokens (INFERRED):**

| Readout | Tokens |
|---|---|
| A, 3 contexts (~42 lines × 12 tokens each) | ≈1.5k |
| L | ≈0.2k |
| Lmax (8 sampled lists) | ≈1.6k |
| L-EOS | ≈0.35k |
| **Total** | **≈3.65k** |

- B4: 300 items → **1.1 M decode tokens**, about 0.5–1.1 h at 300–600 tok/s aggregate. Add 4.5 M prefill at 1.5–3k tok/s (0.4–0.8 h) and 0.15 h start-up.
- **B4 ≈ 1.0–2.1 h** (v1: 0.9 / 1.3).

**Trained arms:**
- The KV prefix cache cannot be shared across different LoRAs.
- Each arm needs ≈3.3 M unique prefill (A real / GF / RW + Lmax real), so 10 M for three arms. That is 1.1–1.9 h.
- Decode is ≈2.8 M tokens, another 1.3–2.6 h.
- Punica multi-LoRA kernels cost about 10–30% throughput (INFERRED).
- **B8 ≈ 2.0–4.7 h** (v1: 2.0 / 2.9).

### 1.5 Dev guard (B9)

**The guard is far too cheap in the plan.**
- v1 budgets 2.5 / 3.4 h. That implies about 50 tasks/h, which means **parallel** execution.
- The deployed config has a **4.5-min wall-clock** cap per task. Running 8 streams in parallel slows each stream's decode, so tasks hit the time cap after fewer steps. **The parallel guard is then not the deployed system.**
- A faithful sequential run of a v5-style pipeline is **4–6.5 h** (VERIFIED wall times above). With the extra CANDIDATES tokens it is more.

**The comparison is invalid as designed (MAJOR on its own).**
- The baseline is the **stored r3_v5 run**. It ran before the scorer-environment rebuild, in the host-package environment where local scores did not predict the leaderboard (HQ Oct 5–7).
- It used a different prompt (no CANDIDATES) and was a single stochastic run.
- "Same harness" is false.
- The guard therefore needs a **fresh base + CANDIDATES run in the same session and environment**. That doubles B9.

### 1.6 Edge-use instrument (B10) and seed replicate (B11)

**B10.**
- 80 trajectories × (6 views + 3 dose-response views) × ~8k tokens ≈ 5.8 M prefill.
- Sharing the prefix up to the issue saves perhaps 30–40%.
- **B10 ≈ 0.4–1.2 h** (v1: 0.5 / 0.8).

**B11.**
- Stage A + B for one arm takes 1.2–2.9 h of training (§1.3), plus one evaluation context (0.4–0.6 h).
- **B11 ≈ 1.6–3.0 h.**

### 1.7 Gates that silently cost GPU time

- G1: "recalibrate, then re-pilot 10 tasks (0.2)".
- G3: "re-sample pools".
- G4: "debug before training the other arms".
- G5: the DPO fallback, which is an extra full training run of ≈0.5–1.5 h.

**None of these is in the subtotal.** With this project's history, assume at least one fires.

### 1.8 Where the plan really is cheap: data volume (this is the scientific break, see §2)

**Stage A data per arm:**
- Round 1 gives 20 tasks × K=4 = 80 rollouts at a GUESSED zero-shot success of 0.2–0.4. That is ≈16–32 successes.
- Capping at 2 per task and taking the minimum across arms (floor 15) gives **≈15–25 successful trajectories**.
- Items are ≤30% of 0.35 M tokens, so **≈20 items** of ~5k tokens.

**Stage B data per arm:**
- 80 trajectories.
- With round-2 depth set to {3, 4, ≥5}, π_A's success rate is plausibly 0.05–0.2.
- P(group has outcome variance at K=4) = 1 − p⁴ − (1−p)⁴. That is **0.19–0.59**.
- So there are **≈4–12 informative groups per arm**.

**Optimiser steps.** At an effective batch of 16, the whole study is roughly **5–10 optimiser steps in Stage A and 10–25 in Stage B per arm**.

### 1.9 Recomputed total for v1 as written (GPU-h)

| # | Line | v1 central / pess. | **Optimistic** | **Central** | **Pessimistic** | Driver of the change |
|---|---|---|---|---|---|---|
| B1 | Training probe | 1.0 / 1.0 | 1.0 | 1.5 | 2.5 | HQQ+FSDP is an untested path; probe history v1–v8 |
| B2 | 30-task pilot | 0.5 / 0.75 | 0.5 | 0.7 | 1.0 | + session overhead |
| B3 | K1/K2, 90 tasks | 1.5 / 2.25 | 1.0 | 1.5 | 2.5 | 100 / 60 / 40 per h + overhead |
| B4 | Base items | 0.9 / 1.3 | 1.0 | 1.5 | 2.3 | Decode for A×3 + Lmax + L-EOS |
| B5 | 480 rollouts | 8.0 / 12.0 | 5.2 | 8.4 | 12.4 | + two session start-ups |
| B6 | Item samples | 1.2 / 1.6 | 1.2 | 2.0 | 3.0 | 2.4 M decode tokens |
| B7 | Training, 3 arms | 4.2 / 5.0 | 3.7 | 4.7 | 8.8 | 2.4 M token-equivalents per arm; G0 floor 250 tok/s |
| B8 | Trained-arm items | 2.0 / 2.9 | 2.0 | 3.2 | 4.7 | No cross-LoRA prefix cache; punica overhead |
| B9 | Dev guard, best arm | 2.5 / 3.4 | 2.5 | 4.0 | 6.7 | Sequential is 4–6.5 h (VERIFIED) |
| B9′ | **Fresh base + CANDIDATES** (required, §1.5) | — | 2.5 | 4.0 | 6.7 | Stored r3_v5 is not the same harness |
| B10 | Edge-use instrument | 0.5 / 0.8 | 0.4 | 0.7 | 1.2 | 9 views |
| B11 | Seed replicate | 1.6 / 2.0 | 1.6 | 2.0 | 3.0 | |
| G* | Gate re-runs (G1/G3/G5) | 0 | 0 | 1.0 | 2.0 | Unbudgeted in v1 |
| | **Subtotal** | 22.3 + 1.6 | 22.6 | 33.2 | 56.8 | |
| B12 | Reserve | 1.1 | 10% → 2.3 | 15% → 5.0 | 20% → 11.4 | v1 reserve is 4% |
| | **Total** | **25.0 / 33** | **≈25** | **≈38** | **≈68** | |

Without B9′ (an invalid guard) the totals are ≈22 / ≈35 / ≈58. **≤25 holds only in the optimistic column.**

### 1.10 Which descoping steps destroy the scientific claim

| Step | Effect on the claim | Severity |
|---|---|---|
| D1: fewer tasks | Shrinks an already sub-dose dataset. It raises the chance of a null on H1 *and* of an uninformative pass on H3. **Destroys the interpretability of nulls** | MAJOR |
| D2: drop B11 | H3 then has no seed term at all. The equivalence claim is unsupported | MAJOR (H3 dies) |
| D3: skip round 2 | **Harmless, even beneficial.** Stage B is predicted to add ≈0 (§6). Weakens only the word "RL" | MINOR |
| D4: 64-task guard | The guard was already uninformative (§2.5) | MINOR |
| D5: drop the three arms | **Destroys the paper's novelty.** H2 and H3 are the "topology as treatment" contribution. Without them it is "synthetic RFT does or does not transfer", an incremental SWE-Playground point | FATAL to novelty |

**The ladder is ordered wrongly.**
- It cuts science (D1, D2) before it cuts low-value spend (D3, the guard, the B10 instrument, Lmax and L-EOS on trained arms).
- **Correct order:** D3 → drop B10 → drop Lmax and GF on trained arms → D4/descriptive guard → (only then) D1. D2 and D5 are last resorts.

---

## 2. Statistics

### 2.1 Clustering: 8 repos is not enough (MAJOR)

**Design effect.**
- ≥300 items over 8 repos means m ≈ 37.5 items per cluster.
- DE = 1 + (m−1)·ICC. For ICC = 0.02 / 0.05 / 0.10 this gives **1.7 / 2.8 / 4.7**.
- v1 assumes 1.5–2. For repo-level effects of a LoRA, such as format habits and naming style, an ICC of 0.05–0.1 is ordinary (INFERRED).

**SE and MDE** (paired SD 0.15, GUESSED in v1):
- Item-level SE = 0.15/√300 = 0.0087.
- Clustered SE = **0.011 / 0.015 / 0.019**.
- With 8 clusters, inference uses about 7 degrees of freedom: t₇,0.975 = 2.36 rather than 1.96.
- 80%-power MDE ≈ 3.26·SE = **0.037 / 0.048 / 0.061**.

**Implications.**
- H2's predicted +0.05 is **at or below the MDE**.
- H4 is a difference of differences, so its variance is about 2× and its SE about √2×. The predicted +0.04 is under-powered at any ICC ≥0.02.
- A pairs (percentile) cluster bootstrap with G=8 is known to **under-cover**: 95% intervals behave more like 85–90% (Cameron–Gelbach–Miller 2008; MacKinnon–Webb 2017; INFERRED from memory).
- The mixed-model "robustness check" with 8 groups estimates the random-effect variance badly as well.

**Fix (no GPU cost):**
- Build **16–20 S_eval repos × 15–20 items**. The prefill cost is identical at fixed n. More repos only costs CPU tracing effort.
- Use a **wild cluster bootstrap-t with Webb six-point weights**, or randomisation inference over repos, as the primary method.
- Report the ICC.

### 2.2 TOST for H3 cannot pass informatively (FATAL for H3 as written)

**Variance of the (b)−(c) contrast:**

`Var = SE_item,clustered² + 2·σ_seed²`

- The plan has one seed per arm, so σ_seed is unestimated. B11 gives *one* extra (b) run: one degree of freedom, and no interval.
- B11 also **reuses the same rollouts**, so it captures only init and shuffle noise. It misses data-sampling noise: which tasks succeeded, which of the 6 source graphs were drawn.
- For LoRA fine-tuning on ~40 sequences, an AP seed SD of 0.01–0.02 is a reasonable GUESS.

**Scenario: real training effect.** Take ICC 0.05 and σ_seed = 0.01. Then SE_total = √(0.015² + 2·0.01²) ≈ 0.020. TOST at 90% needs |d| + 1.895·0.020 < 0.03, which needs |d| < −0.008. **Impossible.**

**Scenario: no training effect.** Neither arm moves. The paired SD of (b)−(c) collapses, because the two adapters make nearly identical predictions, and σ_seed is ≈0. **TOST passes.**

So H3 passes exactly when training did nothing, and fails whenever training did something. That is backwards for a claim that "fidelity is irrelevant".

**Second unit-of-analysis problem.**
- The treatment unit is the **source topology**, and there are only 6 S_topo graphs.
- "Real topology is irrelevant beyond degree and depth" generalises over graphs, but the effective n for that generalisation is 6.
- Item-level bootstrap treats the 6 graphs as fixed.

**Fix:**
- (i) Make H3 **conditional** on a manipulation check passing: (b) beats base on held-out *synthetic* deep tasks by a pre-registered margin, *and* H2 passes. Otherwise H3 is "not evaluable".
- (ii) Run **≥3 seeds** of (b) and (c), each resampling the task subset and rollout subset (data bootstrap, not just init). Include seed variance in the TOST through a hierarchical bootstrap: repos × seeds.
- (iii) Report a **leave-one-source-graph-out** sensitivity analysis.
- (iv) Set the margin from a pre-registered smallest effect size of interest (for example, ½ of the observed (b)−(d) gap), not from the MDE.

### 2.3 Multiplicity is misdesigned (MINOR to MAJOR)

**The Holm family is the wrong tool.**
- Holm over {H1, H2, H3, H4}, with H3 an equivalence test, mixes superiority and equivalence nulls.
- The central claim C is a **conjunction**. An intersection–union test needs no correction for the conjunction. Holm spends power for nothing.
- If the individual claims are each reported, use a **fixed-sequence gatekeeping order**, each test at α = 0.05, stopping at the first failure:
  1. manipulation check;
  2. H1;
  3. H2;
  4. H4;
  5. H3.

**The real multiplicity problem is unaddressed.** Secondary readouts multiply the comparisons with no correction or exploratory label:
- 4 readouts × 3 contexts;
- strata of depth × edge type × direction;
- recall at matched precision;
- IDF-filtered AP;
- the H4 sub-condition "Δ_struct(d) − Δ_struct(base) < 0.015", which has no test specified.

Pre-register them as **exploratory**, with BH-FDR within each family.

### 2.4 Pre-registration gaps (MAJOR in aggregate)

| Gap | Why it matters | Fix |
|---|---|---|
| Eval set is not frozen before training. G3 allows re-sampling pools and strata until base AP is "in band"; A9 drops repos after seeing traces | Researcher degrees of freedom; selection on base performance | Freeze and hash the S_eval item set, the sampler seed and the repo-exclusion rule **before any trained-arm evaluation**. Hold out a confirmatory half (items are cheap) |
| "Best arm = (b) unless H1 picks another" (C11) | Choosing the guard arm from the outcome | Fix the guard arm to (b) |
| "r=32 if Stage A underfits"; DPO fallback; "raise K on all arms" | Data-dependent pipeline changes | Pre-register the triggers as numbers, and apply any change to **all** arms |
| AP with 5-level verbalised p, "ties averaged over orders" | With ~42 candidates and 5 levels, most AP movement is tie-breaking noise; resolution is coarse | Pre-register the **continuous** primary score: the log-prob of the EDIT/CALLED token at each candidate line (vLLM logprobs; same generation, no extra cost). Verbalised p becomes secondary |
| H6 definition | "100 synthetic items" (B4) vs "fraction of *tasks* where both sites listed EDIT" (agent rollouts, not budgeted: ≈1.7 h per arm at 60/h) | Decide which. If rollouts, budget them; if items, call it item-level |
| K1 "negative depth slope" | Near-tautological: deeper bugs give longer traces and more turns against a turn cap of 40, so the slope is negative whatever happens | Require the slope to stay significant **after controlling for trajectory length and repo size**, or test the GF-named contrast instead (K2 already does part of this) |

### 2.5 H5 non-inferiority at 6 pp is neither meaningful nor powered (MAJOR)

**The margin.**
- Base resolve is ≈25% (r3_v5 33/129). A 6 pp margin is **a quarter of all resolves, ≈8 tasks**. Calling that "no regression" will not survive review.
- Under the scorer environment only **54/128 tasks are gold-solvable** (VERIFIED). The real denominator is about 54, so the MDE grows to ≈12–14 pp.

**Power.**
- With SE ≈3.4 pp at n=129, a one-sided 95% non-inferiority test at margin −6 needs an observed Δ ≥ −0.4 pp.
- So power is only ≈50% even when the true Δ is 0.
- A single stochastic base run adds unmodelled run-to-run variance. Even at T=0.2, ±3–5 tasks per 129 is typical (INFERRED).

**Fix:** demote resolve to **descriptive**, and run it as a paired same-session comparison of base+CANDIDATES vs (b)+CANDIDATES on the 54 solvable tasks. Put non-inferiority on dense metrics instead (§4.2).

---

## 3. Causal identification: does the three-arm design isolate topology?

### 3.1 The auxiliary items carry the signal and differ in depth by arm (FATAL for H2 as an RL-on-topology claim)

**How much of the training is items.**
- Items are in the **exact eval format** and labelled exactly.
- They are ≤30% of tokens, but with Stage A at ≈15–25 trajectories they are a large share of *supervision*.
- Each item line is a direct label, whereas a trajectory gives one outcome bit.
- In Stage B, the η-term on items is the dominant per-token signal, because the outcome advantage is zero in 40–80% of groups (§1.8).

**Why H2 reduces to label matching.**
- Arm (d)'s items come from depth-1 graphs, so **its positives sit only at depth ≤2**. Arm (b)'s items include depth ≥3 positives.
- H2 is AP(b) − AP(d) on real items stratified by depth.
- So H2 largely tests "supervised training on deeper call-set labels improves deeper call-set prediction". That is close to tautological, and it is not "multi-hop traversal demand from agent RL".

**Fix (pick one):**
- **(a) Clean:** drop auxiliary items from the main arms. Topology then reaches the model only through agent trajectories. Run a separate, cheap **items-only** factor if desired. Items need no rollouts; they can be trained on exact labels directly.
- **(b) Decomposition:** keep items, but add one arm (b−items) and report H2 on both.

Option (a) is cheaper and gives a cleaner paper.

### 3.2 The equalisation constraints are mutually inconsistent and bias difficulty (MAJOR)

**The constraints cannot all hold.** v1 requires equal distinct successful trajectories, equal failures, equal item count **and** equal total tokens (±5%).
- Arm (d)'s trajectories are much shorter: depth 1, and the failing unit contract names the bug.
- At equal trajectory count, (d) has far fewer tokens. Reaching equal tokens means either more (d) trajectories (breaking "equal successes") or more (d) item tokens.
- If items fill the gap, (d) gets **more eval-format training**. That biases H2 *towards* (d) and makes the confound direction unpredictable.

**Selection on success shifts difficulty.**
- Successes concentrate on the easiest tasks in each arm.
- (b) and (c) trajectories are then mostly depth-2 tasks, not depth-3–5. The effective depth contrast with (d) shrinks.
- If (c) is systematically harder or easier than (b), the two arms' *successful subsets* differ in depth, operator mix and size. (c) can differ because rewiring breaks callee locality (§3.3) and changes fan-in per type.

**Per-arm size recalibration** ("If an arm falls below the floor, recalibrate that arm's repo size") makes **repo size an arm-specific variable**. That is a direct confound.

**Fix:**
- Equalise **within strata**: match successes per (bug depth × operator × size) cell between (b) and (c).
- Report the realised training-set depth and operator histograms per arm. Pre-register a balance check (standardised mean difference <0.1 on depth, length and operator).
- Never recalibrate per arm; change size for all arms or none.
- Drop "equal tokens" for (d), or equalise on **optimiser steps × sequence count** and disclose it.

### 3.3 (b) vs (c) is not "only wiring" (MAJOR)

**Module locality.**
- C3 copies the source file and module layout into all arms.
- Real call graphs are module-local: callees are often in the same file or package.
- A same-type double-edge swap ignores the module partition, so (c) has **more cross-module edges**. That means more imports, more file hops and more read calls.
- (b)−(c) then confounds "fidelity of who-calls-whom" with **locality**, which is a navigation-cost variable the model may well learn from.
- A8 only checks motifs, clustering and assortativity, not the cross-module fraction.

**Name–edge coherence.**
- If (b) keeps the *real* identifier at each node position, real caller and callee names co-occur meaningfully (`send → prepare_request`). The rewire in (c) breaks that, so (b) carries naming content.
- C3 says identifiers are "sampled from a pool". It must state that the **name-to-node assignment is randomised independently of topology, with the same seed in all arms**.

**Fix:**
- Constrain swaps to preserve the cross-module-edge count, for example by swapping only edges in the same (caller-module class, callee-module class) bucket. Alternatively, re-derive the module partition per arm by the same community-detection rule.
- Add cross-module fraction to the A8 balance table.
- Randomise names independently of the graph.

### 3.4 (b) vs (d) differs in more than depth (MINOR; disclosed, but not modelled)

- In (d), the test roots become hubs with huge out-degree, the unit contracts localise bugs almost directly, and trajectories are short.
- H2's "depth is the active ingredient" is therefore "the depth + degree + localisation-difficulty bundle".
- **Fix:** add a depth-2 arm, or make the H2 claim explicitly about the bundle. A cheaper option is to analyse H2 within the subset of real items at depth 1, where (d) should not be disadvantaged. The (b)−(d) gap should then be ≈0 if depth is the mechanism. That is a free, built-in falsifier: pre-register it.

### 3.5 Construct validity of the real items (MAJOR)

**The chain may not be visible.**
- Positives at depth ≥3 are trace-reachable, but the **intermediate nodes on the path may not be in the pool**. The pool takes ≤14 sampled positives.
- Bodies are truncated to 25 lines, which can cut off the call site.
- If the chain is not visible, the label cannot be recovered from the context, and AP at depth ≥3 measures guessing and name priors, not graph reasoning.

**RW incoherence.**
- The RW context swaps callee names at call sites. Arity and keyword mismatches make the code visibly incoherent.
- A model trained on perfectly coherent compiled code may simply be more disturbed by incoherence. Δ_struct then rises from distribution shift, not edge use.
- The critic's "plausible rewire" fix was applied to C10 but **not** to C9.

**Fix:**
- Pre-register **path-complete pools**: every positive's shortest path from the test is included in the pool, and its call sites lie within the shown lines. Report the share of items that pass.
- Add a **plausible-RW** context (arity-compatible swaps only) for the H4 primary.

### 3.6 H1 vs base is confounded with format training (MAJOR for interpretation)

- AP(b) − AP(base) includes learning the A format, which only the trained arms practise.
- The outcome table maps "H1 fails" to "content dominates". That inference is invalid: failure is equally predicted by **underdosing** (§1.8) and by format effects in either direction.
- **Fix: add a positive control (p).** Train on **real-code** call-set items from the S_topo repos (real bodies, exact trace labels, same item count, tokens and format), items-only.
  - No rollouts are needed: about 0.3 h of training and 0.5 h of evaluation.
  - If (p) moves AP and (b) does not, "content dominates" is supported.
  - If neither moves, the dose is insufficient, and the paper must say so rather than claim a content effect.

---

## 4. Is the central claim the right one for the rubric and the user's goal?

### 4.1 Rubric fit

| Axis | Assessment | Severity |
|---|---|---|
| **Novelty / insight** | The topology-as-treatment factorisation is genuinely interesting, but the claim C is a **four-way conjunction**. The modal outcome is "nothing moved" (§1.8, §2.2), which yields no insight. The novelty lives in H2/H3, exactly the parts the stats cannot deliver | MAJOR |
| **Quality / generality** | Pure assertion (§5) | MAJOR |
| **Relevance** | Item AP has no demonstrated link to SWE outcomes. A6 itself notes localiser function-hit did not predict resolve (27% vs 28%). Items are ≤42-candidate local pools, **not large directed graphs**: the graph is never shown at scale | MAJOR |
| **Verifiability** | Strong on labels. Weak on pre-registration freeze (§2.4) and on the stored-baseline comparison (§1.5) | MINOR to MAJOR |
| **Clarity** | Six hypotheses, four readouts, three contexts, an instrument, a fallback and a ladder. Too much for 3,000 words. A reviewer will not see one clean figure | MINOR |

**Recommended reframing.**
- One **headline question**: "Does training on topology-controlled executable worlds increase a model's *reliance on displayed edges*, and is that reliance dose- and depth-dependent?"
- Primary readouts:
  - (i) Δ_struct on path-complete real items, using plausible-RW;
  - (ii) the depth-stratified (b)−(d) contrast with the built-in depth-1 null (§3.4);
  - (iii) the positive control (p) to anchor "content".
- H3 becomes a secondary analysis, only if the manipulation check passes.
- This framing wins on "insight" even if the gains are small, because the measurement itself is new.

### 4.2 "Near SOTA" is not measured (MAJOR)

- §8 offers only arguments: exact grid, trust region, small update.
- The only measurement is an under-powered resolve guard against an invalid baseline. The "bf16 + LoRA gap" is optional.
- The user's goal, "keep the quantised model near SOTA", needs **capability-retention and reference-point evidence**:
  1. **Retention battery** (prefill-dominated, ≈0.3–0.5 h per adapter): a 300–500-item slice of a public multiple-choice reasoning set (for example MMLU-Pro), a code set (for example HumanEval+/MBPP+ via short generation), and a tool-calling format check. Report the base vs adapter difference with TOST. The token-level **KL / NLL drift** of the adapter on held-out *base* agent trajectories is nearly free.
  2. **Reference point**: score the 300 real items with the bf16 QAT-unquantised checkpoint (or the free AI Studio Gemma-4-31B endpoint) and one frontier model through a free API. That shows where W4A16 and W4A16+LoRA sit relative to a stronger model. Without it, "near SOTA" is undefined.

### 4.3 External evidence a reviewer will demand (MAJOR)

1. **An SWE-facing dense benchmark with public provenance.**
   - Example: issue → edit-set items built from SWE-bench Verified/Lite gold patches (function-level gold, multi-function subset). Same A-readout, same AP. n≈300–500, prefill-cheap (≈0.3 h per adapter).
   - This is the bridge from "call-set reasoning" to "relevance to SWE" that A6 lacks. Contamination is equal across arms, because the comparison is paired.
2. **A public code-reasoning call-chain set.** The critic named **RepoReasoner** (call-chain items, ~169; verify the licence and format). The "TraceEval" named in the brief is unverified by me; check that it exists before citing it.
3. **A directed-graph benchmark at scale**, for the graph-ML judges and the user's "large directed graphs" goal.
   - Reachability, path and topological-order queries on *serialised* S_eval call graphs with 500–2,000 nodes (exact labels, CPU-built).
   - Optionally a public set (NLGraph / GraphArena-style directed reachability; INFERRED names, verify).
   - Test whether (b) > (d) here too.
4. **Resolve on non-dev repos.** SWE-bench-Live or SWE-rebench tasks would be ideal, but environment building offline on Kaggle is expensive. Mark as out of scope, and use (1) as the proxy.

---

## 5. Generalisation to other models and hardware (MAJOR as written)

**Everything in §9 is asserted, not shown:**
- online GRPO "drop-in";
- "any model or quantisation";
- "any executable DAG";
- the rank-by-token rule.

**Minimum to claim anything:**

| Claim | Minimum evidence | Cost |
|---|---|---|
| Effect is not Gemma-31B-int4-specific | Replicate (b) vs (d), Stage A only, on **one smaller model** (Gemma-4 12B or e4b in bf16, plain LoRA; registered in the harness) on the same 300 items | ≈2–3 GPU-h on L4 (rollouts are cheaper) |
| "Train on the served grid" matters | The A12 mismatch numbers for NF4 vs HQQ at zero adapter **and** with LoRA_A. Optionally train (b) once on NF4 and compare item AP | 0.3 h (mismatch only) |
| Online GRPO equivalence | Not testable here. **Remove the claim**, or present it as "design for future work" | — |
| Other DAG domains | Remove, or present as a conjecture | — |

**Rule:** the paper may claim only what one second model shows. Otherwise §9 becomes a limitations paragraph.

---

## 6. Validity of offline RL Stage B

1. **The ratio is not an importance ratio** (MINOR to MAJOR).
   - r_t = π_θ / π_A^train is computed against the *trainer's* π_A, not the vLLM behaviour policy.
   - It is a proximal trust region around π_A, not off-policy correction.
   - Any vLLM-vs-HQQ mismatch, including multi-LoRA punica precision with the adapter on, is **ignored**, not removed.
   - A12 checks only the zero adapter and the median.
   - **Fix:** log vLLM log-probs and apply **truncated importance sampling** with weight min(π_A^train / π_vLLM, C=2). This is the TIS correction for rollout–trainer mismatch (Yao et al. 2025, INFERRED). Report the 95th and 99th percentile of |Δ| with LoRA_A active, not just the zero-adapter median.

2. **Epoch 1 is plain policy gradient, epoch 2 is barely clipped.**
   - r=1 in epoch 1. With lr 2e-5 and about 10–25 optimiser steps, r stays near 1 in epoch 2.
   - The clip rarely binds, so the G5 rule "clip fraction ≤20%" is vacuous.

3. **Signal size.**
   - There are about 4–12 informative outcome groups per arm (§1.8).
   - The outcome-advantage gradient is negligible. Stage B's effect will come almost entirely from the **η-term on CANDIDATES and item lines**, which is per-element supervised learning against exact labels in disguise.
   - **Prediction (pre-register it):** on the real-item AP, LoRA_B − LoRA_A will lie within ±0.01. Synthetic held-out item log score improves slightly (driven by η). Synthetic task success will not differ detectably (n≈20 per arm).
   - **Test for free:** add LoRA_A of each arm to the B8 multi-LoRA session (+≈0.5 h per arm, or real-context A only, ≈0.3 h).

4. **A cheaper equivalent.**
   - Drop round 2 and Stage B. Spend the same rollout budget on **twice as many round-1 tasks**, which roughly doubles the Stage-A successes per arm.
   - Replace recombined-item best-of-K with **direct label SFT**. Labels are exact, and "projection onto support" adds nothing at K=4 with 5 levels (the critic made this point already).
   - Optionally, one DPO pass on the (τ⁺, τ⁻) pairs from round 1. It has the same data and avoids a second generation session.
   - This saves ≈3–5 GPU-h (one generation session, 6 h of Stage-B training at 250 tok/s, π_A recompute). Spend it on **seeds**.
   - Call the method "outcome-filtered self-training (RFT)". Do **not** call it RL in the title. The "RL-style" label will be attacked, as the critic already warned for CASE.

5. **K=4 per task.** It is acceptable for RFT. For group-advantage it is too small where p<0.2. If any group-RL is kept, use K=8 on fewer tasks.

---

## 7. Other breaks (brief)

| # | Break | Severity | Fix |
|---|---|---|---|
| 7.1 | **The synthetic harness is not the deployed harness.** C3 says "exactly the deployed tool set: run_command, read_file, edit_file…". Deployed v5 (VERIFIED) is a **two-stage localizer→fixer pipeline**; the localizer has no read or edit tools, read_file was removed in v2, and there is a 4.5-min wall-clock cap. The training trajectories are single-agent; the deployed agent is not | MAJOR (H5 and relevance) | Generate rollouts with the deployed SequentialAgent config (CANDIDATES in the localizer's diagnosis), at deployed sampling |
| 7.2 | The loss normaliser −(1/\|T\|) over batch tokens weights long trajectories more. That contradicts the Dr.GRPO motivation the plan cites | MINOR | Use a constant normaliser (max length) as in Dr.GRPO |
| 7.3 | Reward: unlisted non-gold is unscored, while omitted gold counts as p=0.05. This invites listing only high-confidence nodes, so recall pressure is weak | MINOR | Report listed-set size per arm over training |
| 7.4 | H6 is in-distribution: same compiler, same DPCS operator, only the topology family differs. It is a fit check, not evidence of commit completeness on real code | MINOR | Report it as a manipulation check, not a hypothesis |
| 7.5 | A single held-out topology family (a "7th repo") for H6 and the fallback trigger means n=1 source | MINOR | Use 2–3 families |
| 7.6 | Identifier overlap: S_topo pools may contain common names (`get`, `parse`) shared with S_eval | MINOR | A17's static check must be at the qualified-name level; report the overlap rate |

---

## 8. The 5 most damaging breaks, with minimal v2 fixes

| Rank | Break | Severity | Minimal v2 fix |
|---|---|---|---|
| **1** | **No positive control or dose anchor. With ≈15–25 successful trajectories and ≈20 items per arm, the modal result is "nothing moved". H1 failing is then misread as "content dominates", and H3 passes without telling us anything** (§1.8, §2.2, §3.6) | FATAL | Add arm **(p)**: real-code call-set items, items-only, same dose (≈0.8 GPU-h). Add a **manipulation check** (b > base on held-out deep synthetic tasks or items) that gates H2–H4. Make H3 conditional on it |
| **2** | **Auxiliary eval-format items differ in depth by arm, and equalisation is inconsistent and success-selected. H2 becomes supervised label-depth matching; (b)/(c) differ in difficulty, size and locality** (§3.1–3.3) | FATAL (H2/H3 identification) | Remove items from the main arms (topology enters only through trajectories). Stratified success matching per (depth × operator × size) cell. No per-arm size changes. Locality-preserving swaps. Names randomised independently of topology. Pre-registered balance table. Built-in falsifier: (b)−(d) ≈ 0 on depth-1 real items |
| **3** | **Eight clusters and one seed: the design effect is 2.8–4.7, the MDE is 0.05–0.06, and the TOST ±0.03 is unpassable when training works and passes without telling us anything when it does not** (§2.1–2.2) | MAJOR (kills H3/H4 as stated) | 16–20 S_eval repos × 15–20 items (CPU only). Wild cluster bootstrap-t. **3 seeds × data-resampling** for (b) and (c), paid for by dropping Stage B. Hierarchical bootstrap (repo × seed). Leave-one-source-graph-out. Fixed-sequence gatekeeping. Continuous log-prob primary score. Freeze and hash the eval set before training |
| **4** | **The budget is ≈35 h central, not 25. The dev guard is 2–3× under-costed and compared with an invalid stored baseline. The ladder cuts science first** (§1) | MAJOR | Drop round 2 and Stage B (predicted ≈0, §6). Drop B10 to the appendix. No Lmax, L-EOS or GF on trained arms. Reuse the B2/B3 base rollouts as arm-(b) Stage-A data. Guard = paired same-session base+CANDIDATES vs (b) on the **54 scorer-solvable** tasks, **descriptive**. Reserve 15% |
| **5** | **"Near SOTA" and relevance are unmeasured. Item AP has no SWE link, there are no external benchmarks, and the generalisation claims are asserted** (§4, §5) | MAJOR | Retention battery + KL drift (≈0.4 h per adapter). SWE-bench-Verified issue→edit-set items (≈0.3 h per adapter). Serialised large-graph reachability items (CPU-built). A bf16/reference-model point on the 300 items. One small-model replication of (b) vs (d), or cut §9 to "future work" |

Honourable mentions: the rollout-harness mismatch with the deployed two-stage agent (7.1); the T=1.0 untruncated sampling (§1.1); and the untested HQQ+FSDP path (§1.3).

---

## 9. Corrected GPU-h table for a minimal v2 (single round, RFT-only, with seeds and controls)

**Design assumed:**
- Arms (b), (c), (d) trained on trajectories only, plus the (p) items-only real-code control.
- Stage A (RFT) only, about 0.4 M tokens per run.
- 3 seeds for (b) and (c), 1 seed for (d) and (p): **8 adapters**.
- Primary readout: continuous A-readout on 320 path-complete items from 16+ repos, in real + plausible-RW contexts.
- Rates: rollouts 100 / 60 / 40 per h; training 650 / 500 / 250 tok/s.

| # | Line | Opt. | Central | Pess. | Notes |
|---|---|---|---|---|---|
| V1 | Training probe (exact HQQ-FSDP path, mismatch with LoRA, KV concurrency at 16 streams, `nproc`) + 30-task pilot, one session | 1.4 | 1.8 | 3.0 | Must pass before anything else |
| V2 | K1/K2: 60 depth-stratified base tasks (**reused as arm-(b) round-1 data**) + 30 GF-named | 1.0 | 1.6 | 2.5 | |
| V3 | Base items: 320 items × A (real / GF / RW / plausible-RW) + L + L-EOS on 100 items; SWE-bench edit-set items; large-graph items; retention battery | 1.5 | 2.2 | 3.2 | Prefill-dominated |
| V4 | Rollouts: 3 arms × 30 tasks × K=4 = 360, minus ≈60 reused from V2 | 3.2 | 5.2 | 7.7 | Single round, deployed pipeline, T=0.7 / top-k 40 |
| V5 | Training: 8 adapters × ≈0.4 M tokens, one FSDP session | 1.7 | 2.2 | 4.0 | No Stage B, no π_A recompute |
| V6 | Trained evaluation: 8 adapters × real + plausible-RW (primary); 4 main adapters × (SWE edit-set, graph items, retention) | 3.2 | 4.4 | 6.3 | One multi-LoRA session, maybe two |
| V7 | Dev guard, **descriptive**: base+CANDIDATES vs (b) on 54 solvable tasks, same session, sequential | 2.6 | 3.2 | 4.2 | 108 runs at ~170 s plus start-up |
| V8 | Gate re-runs | 0.5 | 1.0 | 2.0 | |
| | **Subtotal** | **15.1** | **21.6** | **32.9** | |
| | Reserve | 10% → 1.5 | 15% → 3.2 | 20% → 6.6 | |
| | **Total** | **16.6** | **24.8** | **39.5** | |

**v2 descoping ladder, in order** (science preserved longest):

| Step | Change | Saving | What it does to the science |
|---|---|---|---|
| 1 | Drop V7 (resolve is descriptive anyway) | −3.2 | Lose the SWE anecdote |
| 2 | Drop the retention battery and large-graph items for the (c) seeds | −0.8 | |
| 3 | Seeds 3 → 2 for (b) and (c) | −1.5 | H3 margin widens; disclosed |
| 4 | Tasks 30 → 20 per arm | −1.7 | Dose falls; the (p) control still anchors nulls |
| 5 | Drop (c) | Last resort | **H3 is lost, so the paper becomes depth-only (b vs d), with novelty reduced** |
| — | Never drop (p) or (d) | — | They are what make any outcome interpretable |

**Expected outcome to pre-register for v2 (INFERRED):**
- Real-item AP gains are small: (b)−base ≈ 0.01–0.03.
- (p) beats (b).
- Δ_struct(b) > Δ_struct(base) is the most likely positive finding. Training on content-free worlds should mainly raise *edge reliance*.
- The depth-1 null holds.

That is still a publishable "analysis" paper under the rubric's Novelty clause. The v1 framing (transfer gains + fidelity irrelevance) is not.
