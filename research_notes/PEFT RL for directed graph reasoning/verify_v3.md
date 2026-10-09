# Verification of architecture v3 (final) against all prior break reports (8 Oct 2026)

**Role:** fresh, skeptical verifier.

**Read in full:**
- `architecture_v3_final.md`;
- `break_v2_logic.md` (L-01 … L-27);
- `break_v2_priorart.md`;
- `docs/training_feasibility_4xl4.md`.

**Read in the relevant sections:**
- `break_v1_logic.md` (K-01 … K-51: §0, §1–§5 in full; §6–§11 by heading);
- `break_v1_method.md` (bottom line, §1.4–§1.7, §3.5–§5);
- `break_v1_priorart.md` (§0, §4–§6);
- `architecture_v2.md` §10 (v1→v2 change log), used to see which v1 fixes v3 inherited and which it lost.

**Labels:**
- **VERIFIED**: checked against the files (with a line reference or a grep), or arithmetic I recomputed (scipy for t and normal quantiles).
- **INFERRED**: my reasoning. It has not been run.

Nothing was trained, run or searched on the web this pass. I modified no file except this one.

---

## 0. Verdict up front

**NEEDS-FIXES.** Day-1 CPU kills K-C1 to K-C4 and K-C7 may start now, after three small amendments to the kill table (fixes F3, F4 and F6 below). No GPU spend should happen until fixes F1 and F2 are in.

Three findings change the plan materially:

1. **The training budget counts one epoch, but the recipe trains two (VERIFIED arithmetic).**
   - L5 = 26 × 0.75M = 19.5M tokens. That reproduces v3's 6.4 / 8.9 / 14.3 h exactly.
   - But §2.3 defines 0.75M as *per epoch* ("≈62 sequences per epoch") and trains 2 epochs, so the true count is 39M tokens.
   - Corrected totals: **24.0 / 33.8 / 59.3 GPU-h**, against v3's stated 17.4 / 24.6 / 43.1. Central is over the 25 h cap before anything goes wrong.
2. **v3 promoted the rewired-real-code context (E2) to the headline, but silently dropped every K-34 mitigation v2 had (VERIFIED by grep).**
   - In v2, Δ_struct (rewired context) was exploratory because of K-34. v2 used plausible-RW, a corruption probe, an oracle-normalised score, and stratification by whether the test body was rewired.
   - v3 makes E2 the metric for H1, H2 and all of Family B. None of those mitigations appears in v3. Worse, its locality conflicts deliberately use cross-module "far-type" swaps.
   - Result: H1's predicted +0.15 can come from "content-free SFT trains away plausibility checking and memory", not from better graph reasoning.
3. **The "only when" half of the [CLAIMED] dial is under-powered, and the nested design cannot separate "statistic-specific" from "monotone in null strength" (VERIFIED arithmetic + INFERRED design).**
   - HS-spec power is ≈0.14–0.47 jointly, not ≈0.7.
   - H2b's TOST has power **0** at α = 0.025 as specified.

---

## 1. Audit of the v2→v3 change log

### 1.1 L-01 … L-27

| Break | v3 claims | Verdict | Evidence / why |
|---|---|---|---|
| L-01 blind-commit credit | Rewritten into the dial | **Fixed by reframing** (VERIFIED §0) | The old claim is gone. The dial inherits L-02's logic, though; see §2, edge case N2 |
| L-02 pairing ≡ mixture | Mix_mid arm; pairing demoted to S1 | **Fixed** (VERIFIED §1, §2.4) | One residue: every "break" arm is still a 50/50 real+null *mixture*. "Break" therefore means *halving* the cue's validity, not destroying it (N2). Mix_mid is a break-Loc/Mot arm but is left out of the HS design matrix, which wastes 3 seeds |
| L-03 fitting repairs | Per-node hidden contracts, K-C7 | **Fixed for the agent part** (VERIFIED §2.5, §8) | Agent work is descriptive only |
| L-04 RFT freezes bias | RFT removed; signed reliance | **Fixed** | — |
| L-05 GRPO std-normalisation | Pair-pooled mean, no std | **Fixed (specified only)** (VERIFIED §3) | Never run |
| L-06 last-hop discordance | Change log says "hop position reported" | **Claimed but absent from the body** (VERIFIED: "hop" appears only in a prompt name and in the change log) | No hop or depth stratum is defined for synthetic or real items |
| L-07 1–2 RFT steps | 32 item-SFT steps | **Renamed** (INFERRED) | Moved from ~1 step to 32 steps at batch 64 items. Whether 32 steps at lr 1e-4 moves a 31B model is untested. The hyperparameter freeze (L3: 0.4M tokens per config) is about a quarter of one real run (1.5M tokens), so it does not freeze the real schedule |
| L-08 issue-number leak | No agent twins in the claims | **Fixed by deletion** | — |
| L-09 tracing is reading | Agent claim withdrawn | **Fixed by deletion** | — |
| L-10 constant inversion | O1 ≤3-way flip, MCR | **Fixed (agent)** | An acceptable alternative to L-10's suggested fix |
| L-11 ≈6% rewirable | Family constraint dropped; min J; far/leak nulls | **Fixed** (VERIFIED §2.2) | New risk: on small or layered graphs, near ≈ mid in what they destroy (N3) |
| L-12 tiny joint yield | K-C2 joint-yield kill | **Fixed, with an inconsistency** (VERIFIED) | K-C2 passes at ≥2,000 pools per null order. §2.3 requires a reservoir of ≥5,000 and distinct pools per seed, and demand for ρ_mid alone (M_mid 5×500 + Mix 3×500 + M_rel 2×500) is 5,000. Between 2,000 and 5,000 is a silent pool-reuse zone |
| L-13 dark edges | Registry-key swaps; "H strata split lexical vs dark" | **Half fixed** (VERIFIED: "dark" appears only at l.486) | Registry swaps are in §2.2. The lexical-vs-dark stratum exists only in the change log. The K-07 fixes inherited from v2 (≥3 overrides, computed keys, getattr, grep-distance) are not mentioned in v3 |
| L-14 contradictory labels | Path-complete in both wirings | **Fixed, but opens a new leak** (N6) | — |
| L-15 SCC levels | "SCC nodes excluded; survival and in-degree histograms" | **Partially present** (VERIFIED l.156 vs l.488) | The SCC cap is in C2. The exclusion rule and the histograms appear only in the change log |
| L-16 pooled evaluation | Headline on real E1/E2, "no home world" | **Partially fixed** (INFERRED) | E2 builds Loc conflicts with *far-type* swaps (M_far's training operator) and Mot conflicts with *near-type* swaps (M_near's training operator). So for some arms a partial home advantage comes back (N4) |
| L-17 power | 5 seeds, KR df, short chains | **Mostly fixed** | The power numbers are partly wrong; see §3.1 |
| L-18 G2 stop | Descriptive, with a pivot | **Fixed** | — |
| L-19 unsigned metric | Signed reliance | **Fixed for HS; reappears in M_rel** | L_rel rewards *any* difference between halves, which is unsigned. That is acceptable, since it is CRPO's own design, but say so |
| L-20 H3 vs rewire weakness | Nulls of increasing J; direct H3 only under Q1-B | **Fixed in spirit; H3 demoted** | `break_v1_priorart` §4 rated real-vs-null *as a training treatment* the only SURVIVES verdict. v3 §0 cites it as "what survived", yet runs it directly only under the conditional Q1-B |
| L-21 retention | KL + margins powered | **Mostly fixed** (INFERRED caveat) | "Exact full-vocabulary KL" cannot come from vLLM, whose logprobs are top-k with max_logprobs ≈20 against a 262k vocabulary. It has to be computed on the HF NF4 path, i.e. the *trainer*, not the served stack that §4 says all evaluation uses. The margins (prompt_logprobs on the served stack) are fine |
| L-22 ≈3k labels | 16k labels per adapter | **Fixed on paper; unfunded** | Two epochs are needed and not budgeted (§3.2) |
| L-23 unbudgeted lines, weekly fit | Lines added; 20% reserve | **Not fixed** (VERIFIED §3.2) | Correct L5 central is ≈17.2 h, so week 3 alone is ≈17–19 h against the ≈10 h/week cap |
| L-24 method is CAD-on-items | Declared up front | **Fixed (honest)** | It also means the user's "RL" requirement is not met (§5) |
| L-25 GRPO advantages | Specified | **Fixed (specified)** | — |
| L-26 content–wiring independence | Scope + name-informative stratum | **Fixed** | — |
| L-27 small items | Normaliser, ε, without-replacement sampling | **Fixed** | Pool-count inconsistency, see L-12 |

### 1.2 Fixes silently dropped (in v2, absent from v3, not in v3's change log)

All VERIFIED by grep on both files.

| Item | In v2 | In v3 | Severity now |
|---|---|---|---|
| **K-34 Δ_struct confounds:** plausible-RW (arity-compatible swaps among files the caller imports), corruption-detection probe, no-LLM name-reachability oracle with normalised Δ, test-body rewiring stratified | v2 §2.9 l.367–373; change log l.602 | **Absent.** E2 is now the *headline* metric, and Loc conflicts use far-type, import-adding swaps | **Major, and worse than v2** |
| **K-35 static-reachability distractor labels and stratum** | v2 l.363, l.603 | Absent. The E1 static/trace agreement filter removes some of the problem | Major → partial |
| **K-01 / K-04 / BM-§3.5 depth stratification** (v2's H2b stratum "depth ≥2 × static-reachable distractor") | v2 l.75 | Absent. No depth or hop stratum on E1/E2 | Major: depth-1 positives in E2 are a string lookup in the test body |
| **K-07 compiled dark edges** (≥3 overrides, computed registry keys, getattr, grep-distance) | v2 l.577 | Absent | Minor–major (transfer to the "dark" stratum) |
| **K-41 / B7 calibration (ECE) per arm** | v2 l.690 | Absent | Minor |
| **K-47 / B18 weak name–template correlation (30%)** | v2 change log | Replaced by "random names" | Minor. Negative name transfer is now measured, not mitigated (L-26 stratum) |
| **BM-§4.3(3) large serialised-graph benchmark** (500–2,000-node directed reachability) | v2 l.660: "deferred; noted for v3" | **Absent** | **Major for the user's goal of "large directed graphs"** |
| **BM-§5 small-model replication** (v2 L11: Gemma-4 12B, M vs U) | v2 l.467 | **Absent**, not even conditional | **Major for the user's goal of generalising to any model** |
| **BM-§2.1 / B9 ICC reporting; B17 leave-one-source-out** | v2 §8, §11 | Absent | Minor |
| **BM-§2.3 BH-FDR on exploratory families** | v2 | Replaced by "intervals, no confirmatory claim" | OK |

v1 fixes that **are** still present: K-02 SCC, K-03 task-wise growth, K-06 effect-only edges and hub cap, K-09 mocks and unique-resolution typing, K-10 dead code, K-20 roots-per-node, K-22 motif reporting, K-23 exact level preservation, K-24 dispatch not rewired, K-36 continuous score, K-45 throwaway hyperparameter family, K-49 hub cap. K-05 (value masking) no longer matters, because labels are executed call sets.

---

## 2. New breaks introduced by v3 (the null-model dial)

### N1. Nested nulls make the design matrix collinear with null strength — **major** (INFERRED)

- Each family's break set is a *suffix* of one ordering, near < mid < far < leak:
  - Loc: mid, far, leak;
  - Lev: far, leak;
  - Mot: near, mid, far, leak.
- J also rises along the same ordering (0.30 → 0.50 → 0.80 → 0.90).
- So a single non-specific mechanism, "the more rewired the training worlds, the less the model trusts *any* prior", produces HS-Loc, HS-Lev and HS-Mot all with β < 0, exactly as predicted.
- Only HS-spec can tell the two apart, and it is under-powered (§3.1).
- **Fix:**
  - Add one **non-nested null**, ρ_mod: degree + module, no level constraint. It breaks Lev and keeps Loc, the reverse of ρ_mid. Then pre-register the **double dissociation** as the specificity test: an arm × family interaction (M_mid lowers Loc more than Lev; M_mod lowers Lev more than Loc).
  - Fund it by dropping M_leak to descriptive, or by replacing it (S3 is secondary anyway).
  - Add J as a covariate in the HS regressions.

### N2. "Break" arms are 50% real; Reliance conflates edge-reading skill with prior use — **major** (INFERRED)

- Every M_x arm is 500 real + 500 null. The Loc cue therefore stays positively predictive in every break arm's data, at roughly half strength.
- By L-02's own logic, a pure prior-weighting account predicts break-arm reliance **between base and U, still above 0.5**. The predicted 0.40 (below indifference) can only come from better reading of displayed edges.
- Better edge reading lowers reliance on *every* family's conflict set equally. That is the generic competence confound.
- **Fix:**
  - Define Reliance_f as a **cue-use contrast**: score on concordant-f elements minus score on conflict-f elements, so that general reading skill cancels.
  - Report raw conflict accuracy separately.
  - State the mixture dose in the claim: break = 50% null exposure.

### N3. Small, layered or sparse source graphs: near ≈ mid ≈ far, and "destroy locality only" may be impossible — **major** (INFERRED)

- **Layered architectures.** In code where modules align with levels (utils at the bottom, API at the top), level(b) = level(d) already implies module(b) = module(d) with high probability. ρ_mid then preserves most of the locality, and M_mid is a weak "break-Loc" arm.
- **Degree correlates with level.** Hubs are deep utilities. ρ_far preserves degree, so it partly preserves Lev as well.
- **The K-C1 relaxation makes it worse.** On failure, K-C1 allows ±1-level swaps for near and mid. That breaks Lev in the *keep* arms (near, mid), which contaminates HS-Lev's keep group and HS-spec(Lev).
- **"Keep a source only if every null reaches its J" is K-23-style selection.** It favours large, non-layered graphs and drops exactly the layered ones where the families are hardest to separate.
- **Small graphs.** On 6–8 sources of a few hundred nodes, ρ_near can be near-identity, and K-C1 at ≥5 sources leaves very little diversity.
- **Fix:**
  - Pre-register a **per-null destruction threshold for each family, not only J**. For example, the intra-module edge fraction in ρ_mid must be ≤ chance + 0.1, and the level-adjacency AP in ρ_far must be ≤ base rate + 0.05.
  - Keep K-C3(b), but state it in AP terms.
  - If a ±1 relaxation is used, reclassify near and mid as partial-break for Lev.

### N4. Conflict sets overlap, and the E2 operators give some arms home advantage — **major** (INFERRED)

- A far-type swap that creates a Loc conflict usually also changes level adjacency and motif membership. So Loc-conflict elements are often Lev- and Mot-conflicts as well, and Reliance_Loc picks up Lev reliance.
- Mot conflicts use near-type swaps, which are M_near's training operator. Loc conflicts use far-type swaps, M_far's operator. C5's claim that "the conflict for one family does not come from the training operator of every break arm" holds only literally.
- **Fix:**
  - Use **pure-conflict elements**, where only family f's classifier flips. If that gives too few, fit a multi-cue regression on all three cue scores per element.
  - K-C6 should count *pure* conflicts (≥150 each).
  - Use a third, held-out swap operator (e.g. callback-only or registry-only) for at least one family.

### N5. Implausible displayed edges: base rationally trusts priors and memory (K-34 returns) — **major** (INFERRED; dropped fixes VERIFIED)

- Example: `parse_url` now calls `render_table(x)`, with an added import. The base model, which may have seen popular OSS in pretraining, rationally discounts the displayed edge.
- Every trained arm saw only content-free code, where names never predict calls. All of them learn "implausibility carries no information".
- So **H1 (vs base) is confounded** by:
  - plausibility checking being trained away;
  - memory being overridden;
  - format familiarity (BM-§3.6).
- H2 (vs U) and the HS tests are protected, because all arms share the content-free distribution. The prompt-only baseline "trust only displayed calls" partly controls H1. Credit to v3 for that.
- **Fix:**
  - Restore v2's plausible-RW constraint for every conflict built on real code: arity-compatible callees from modules the caller already imports. For Loc conflicts that cross modules, choose callees in already-imported modules.
  - Restore the corruption-detection probe and the no-LLM displayed-BFS oracle.
  - Stratify by whether the test body was rewired.
  - Report Δ_GF (memorisation) per repo and prefer less popular S_eval repos.
  - State H1 as "follows displayed edges more", not "reasons better".

### N6. Items solvable without multi-hop reasoning, despite K-C4 — **major** (INFERRED)

- **Shown-set leak.** In path-complete items, non-pool helpers are shown only if they lie on a reach path. "Called by any displayed non-pool helper ⇒ CALLED" is then a one-hop rule that bypasses tracing. If the shown helper set differs between twins, the leak is twin-specific too.
- **Depth-1 positives** are a name lookup in the test body, on both synthetic and E2 items.
- **Pool order.** If candidates are listed in file or BFS order, the order leaks reachability. v3 does not specify shuffling.
- K-C4 masks call sites and tests content, so it misses all three.
- **Fix:**
  - Add a K-C4b no-LLM **"mentioned-anywhere" and "depth-1 lookup" baseline**, with the kill set at AP ≤ base rate + 0.10 on the depth ≥2 stratum.
  - Pad every item with decoy displayed helpers that are not on any reach path, matched in count across twins.
  - Randomise candidate order.
  - Pre-register a hop ≥2 stratum for E1 and E2.

### N7. Packed 4-D masks: label leakage, position IDs, and a throughput claim that does not hold — **major for G0 / budget** (INFERRED)

- **Label leakage.** About 16 items × 16 suffixes sit in each 12k sequence, and every suffix contains a teacher-forced label token. Gemma-4 builds separate sliding and global masks. If a custom 4-D mask is honoured on one layer type but not the other, suffixes see other suffixes' labels. C11 (zero-adapter HF–vLLM agreement) can miss this, because at zero adapter a leak only shifts log-probs slightly.
- **Position IDs.** Each suffix must restart at the prefix's end position to match vLLM's single-suffix requests. This is not specified.
- **Throughput basis.** Mem-efficient SDPA with an explicit mask still computes the full 12k × 12k score blocks; masking does not skip work. Only varlen flash attention exploits packing, and the feasibility doc rules it out for head_dim 512 and for training above 192 on sm89. So "packed short items have little attention cost" is wrong. The 900 tok/s optimistic rate has no basis, and 650 is the *top* of the documented 400–650 range, not the centre.
- **Fix:**
  - A G0 unit test: perturb one suffix's label token and assert that every other suffix's logits are bitwise unchanged, on both layer types.
  - Reset position IDs per suffix.
  - Pack to ~2–4k sequences (a few items each) to cut wasted attention FLOPs.
  - Re-centre the rates at 400 / 525 / 650 until G0 measures them.

### N8. 32 optimiser steps — **minor–major** (INFERRED)

- 32 steps at batch 64 items (1,024 labels per step), lr 1e-4 with cosine decay, is a small number of updates. GW(a) failure from under-training is a live risk.
- The batch size is arbitrary. The same tokens at effective batch 2 sequences for **one epoch** give ≈31 steps for half the GPU-h; see the budget fix in §3.2.
- Over-training is unlikely at this dose. The retention gates would catch it.
- **Fix:** take the step count from the throwaway-family freeze, and run the freeze at the real token count and schedule.

### N9. Evaluation cost depends on real-item length, which v3 never states — **major for budget** (INFERRED)

- L6 assumes ≈1.27M prefill tokens per adapter across E1–E4. With 320 items × 3 real contexts, that is ≲1k tokens per real item.
- Path-complete pools of *real* function bodies (16 candidates plus intermediates) are plausibly 2–4k tokens. v2 itself budgeted ≈2k-token items.
- At 2.5k tokens per real item, L6 grows ≈2–2.5×, to ≈13–15 h central.
- **Readout risk:** Gemma tokenises digits separately (INFERRED), so "0.05" or "0.95" are not single level tokens. The readout must force the "0." prefix and read the next digit over {0, 2, 5, 8, 9}. vLLM's top-k logprobs may also omit some of the five, so score them as forced continuations.

### N10. Kenward–Roger and the units of inference for HS — **major (statistics)** (INFERRED)

- **Unit of inference.** HS regressions treat seeds from *different* arms as exchangeable within "break" or "keep". If the claim is about nulls that break family f *in general*, the unit is the arm: 3 break vs 2 keep for Loc, so df ≈ 3. Seed-level SEs are then anti-conservative. Either phrase HS as a fixed linear contrast over *these* arms, or accept arm-level df.
- **KR feasibility.** KR on `score ~ arm + (1|seed:arm) + (1|repo) + (1|repo:item)` with ≈26 × 400 × 16 ≈ 170k rows is computationally infeasible (pbkrtest needs n × n operations). Aggregate to seed × item means first.
- **Two-seed arms.** M_leak and M_rel have Welch df ≈ 1, so the Welch sensitivity analysis is uninformative for them. The pooled σ_seed assumes homoscedasticity across arms; test it.

### N11. Smaller inconsistencies (VERIFIED)

- GW(a) requires ΔAP ≥ +0.05 "in its own world" for every arm. For P, the own world *is* E1, where GW(b) uses +0.03.
- G0 passes at ≥400 tok/s, but Q2 fires at <500, and K-G0 is "ambiguous" at 400–500. Pick one.
- E4's "3 held-out source families": S_topo has only 6–8 repos, and v3 does not define where the held-out families come from.

---

## 3. Arithmetic re-check

### 3.1 Power table (σ_seed = 0.03, item component 0.008, as v3 assumes)

| Test | v3 says | Recomputed (VERIFIED) | Note |
|---|---|---|---|
| H1 (5 seeds vs base, df 4, one-sided 0.025) | SE 0.017, MDE80 0.06, power 0.99 | SE 0.0156, MDE80 0.058, power ≈1 | OK. df is 4, not "4–8" |
| H2 (5 v 5, df ≈8) | SE 0.022, MDE 0.075, power 0.85 | SE 0.021, MDE 0.070, power ≈0.9 | OK, slightly conservative |
| H1b | power 0.3 | ≈0.2–0.25 | OK in direction |
| **H2b TOST, margin 0.05, SE 0.022** | **0.6** | **0 at α = 0.025** (the 1−2α CI half-width 0.0507 is larger than the margin); **0.32 at α = 0.05** | **Wrong.** Widen the margin to ≥0.07, or drop the claim |
| HS-Loc (10 v 8) | SE 0.014, power ≈1.0 | SE 0.0142; power ≈1 at −0.20 | OK *if* seeds are the unit (N10) |
| HS-Lev (5 v 13) | SE 0.016, power 0.99 | SE 0.0158; ≈1 at −0.12 | OK, same caveat |
| HS-Mot (5 v 13) | MDE 0.047, power ≈0.5 | MDE ≈0.051; power 0.78 at Holm-first α = 0.0083 | v3 is conservative; fine |
| **HS-spec (TOST, margin 0.06)** | **≈0.7** | Loc (near 3 v U 5): **0.23 at α = 0.025**, 0.57 at 0.05. Lev (mid 5 v U 5): 0.61 / 0.81. **Both jointly: ≈0.14 / 0.47.** After ladder step P6 (near → 2 seeds): Loc **0 / 0.29** | **Wrong, and this test carries the "only when" in the [CLAIMED] sentence** |

All figures ignore the arm-level variance (N10) and any σ_seed above 0.03. Both would make them worse.

### 3.2 GPU-h table

Column sums (VERIFIED):
- core: opt 15.8, central 22.4, pess 35.9;
- reserve: 1.6 / 2.2 / 7.2;
- totals: 17.4 / 24.6 / 43.1;
- the ladder's cumulative figures (35.9 → 29.8; +20% = 35.8) are internally consistent.

**L5 error (VERIFIED):**
- 19.5M tokens at 900 / 650 / 400 tok/s gives 6.02 / 8.33 / 13.54 h, plus loads of 0.38 / 0.57 / 0.76 h. That reproduces 6.4 / 8.9 / 14.3 exactly, so v3 budgeted **one epoch**.
- §2.3 and §3 train **2 epochs** over 0.75M tokens per epoch. That is 39M tokens, and L5 becomes **12.4 / 17.2 / 27.8 h**.

| | Opt. | Central | Pess. |
|---|---|---|---|
| v3 total | 17.4 | 24.6 | 43.1 |
| **Corrected core** | 21.8 | 30.7 | 49.4 |
| **Corrected total (same reserve %)** | **24.0** | **33.8** | **59.3** |
| + N9 if real items are ≈2.5k tokens (INFERRED) | ≈+3 | ≈+8 | ≈+12 |

- The ladder (P2–P7) saves ≈5 h at central (INFERRED). That gives ≈25.4 core, or ≈28 with reserve. **Still over 25 at central**, before N7's throughput correction.
- **Week 3** at central is ≈17 h of training alone, against a ≈10 h/week cap.
- **Cheapest fix (F1):** keep 1,000 items and 16k labels, train **1 epoch at effective batch 2 sequences (≈31 steps)**. The tokens match what v3 actually budgeted (19.5M), and the step count matches what it promised. Alternatively, cut tokens per item by trimming suffixes or prefixes. Either way, re-run the budget with N7's rates and N9's measured real-item length.

### 3.3 Does the descoping ladder keep the headline?

| Claim | After P2–P7 | After Q2(b) (3-point dial) | After Q2(a) (750 items) |
|---|---|---|---|
| H1, H2 | **Kept** (U × 5, M_mid × 5) | Kept | Kept; the dose is lower, so GW(a) risk rises |
| HS-Loc | Kept (break 8 v keep 7, SE ≈0.016) | Break = mid + far; keep = **U only** | Kept |
| HS-Lev | Nominally kept, but break = M_far × 2 + M_leak × 1 (3 adapters) | Break = **far only** (≤3 seeds) | Kept |
| **HS-spec (the "only when")** | **Lost in practice** (Loc power ≤0.29) | **Impossible for Loc** (no M_near) | Same as P2–P7 |

**Conclusion (INFERRED):**
- The ladder keeps the *method* claims (H1, H2) and the *directional* dial (HS-Loc, HS-Lev).
- It does **not** keep the specificity half of the [CLAIMED] sentence. Under realistic descoping the paper's novel claim becomes "nulls reduce prior reliance", which v3's own outcome table calls "a blunt instrument".
- With the L5 correction, some descoping is near-certain at central, not just pessimistic.

---

## 4. Does v3 meet the user's target?

> "A PEFT + RL novel method that improves LLM reasoning over large directed graphs, verified on code, trainable on little compute, keeping the quantized model near SOTA, generalisable to any model/hardware."

| Requirement | Status | Why |
|---|---|---|
| PEFT | **Met** | LoRA r16 on the documented NF4 FSDP-QLoRA path, served as W4A16 + LoRA (VERIFIED §4; feasibility doc) |
| **RL** | **Not met** | Nothing RL-like is trained. The method is SFT on execution-verified labels. v3 says this honestly (§0, L-24). "Verifiable reward used as a supervised target" is supervision, not RL. The online GRPO drop-in is specified, never run, and depends on agent worlds that are only CPU-audited |
| Novel method | **Partially met** | The *method* (mix null-rewired items into synthetic SFT) is closeness-2 to null-model augmentation (Xuan et al.) and CAD. Pairing is predicted ≈0 (S1). The novelty is an *analysis instrument* (the dial plus the reliance readout), backed by 6 searches only, with the hostile red team still pending (C1). Its specificity half is under-powered (§3.1) |
| Improves reasoning over **large** directed graphs | **Not met** | Items are 16-candidate local pools of ≈0.75k tokens. The model never sees more than a few dozen nodes. The large serialised-graph benchmark that v2 deferred "for v3" was silently dropped. "Improves" is measured as following displayed edges in rewired real code, which K-34 confounds (N5) |
| Verified on code | **Partially met** | Real-code E1 (trace labels), E2 (rewired real code), and E5 SWE-V edit-set and TraceEval slices. But E2 is artificial, the headline H1 is confounded (N5), and E1 transfer is declared under-powered (H1b ≈0.2–0.3) |
| Little compute | **Met for the method; not met for the study as budgeted** | One adapter is ≈0.3–0.7 GPU-h on 4×L4, which is genuinely cheap. The 26-adapter study is ≈34 GPU-h central after the L5 correction, not 24.6 |
| Quantised model near SOTA | **Partially met** | Retention non-inferiority (KL, margins) is well powered: "does no harm" is showable. "Near SOTA" is not: there is no frontier reference (API dropped), the bf16 reference is conditional, and the published-number comparison is weak. The exact-KL "served stack" claim is infeasible (L-21 row) |
| Generalisable to any model or hardware | **Not met (asserted)** | One model, one quantisation. v2's 12B replication line was dropped without a log entry. Generality is a design claim (§6 says "specified, not tested" for GRPO) |

**Blunt summary:** what v3 can deliver is a careful, honest *analysis* paper. It asks whether synthetic call-graph SFT with nested null models shifts which topology cues a quantised 31B model relies on when reading code. It is not a PEFT + RL method for large directed graphs with demonstrated generality. Three cheap additions would close part of the gap:
1. an eval-only large serialised-graph probe (≈0.5–1 GPU-h, CPU-built);
2. a one-seed M_mid vs U replication on a smaller model (≈1.5 GPU-h);
3. reframing the title to match what is actually run.

---

## 5. Required fixes (in priority order)

| # | Fix | Cost |
|---|---|---|
| **F1** | Fix L5. Use 1 epoch at effective batch 2 (≈31 steps; 19.5M tokens), or re-budget 39M tokens honestly. Re-centre throughput at 400 / 525 / 650 until G0 (N7). State the real-item token length and recompute L6 (N9). Re-fit week 3 under 10 h | 0 GPU-h (doc) |
| **F2** | Restore the K-34 controls on E2: plausible conflicts (callees in imported modules, arity-compatible), the corruption probe, the displayed-BFS oracle, test-body stratification, Δ_GF memorisation. Restore K-35 and depth/hop strata (hop ≥2 pre-registered). Reword H1 as "follows displayed edges" | ≈+0.3 GPU-h |
| F3 | Make specificity identifiable: add a non-nested ρ_mod null (degree + module, no level). Pre-register an arm × family double-dissociation test. Use J as a covariate. Use a cue-use contrast (concordant − conflict) for Reliance. Include Mix_mid in the HS matrix. Fund by cutting M_leak to 1 seed (descriptive) | ≈+1.5 GPU-h (3 seeds) |
| F4 | Amend the CPU kills: per-family destruction thresholds per null (N3); K-C6 counts *pure* conflicts (N4); add **K-C4b**, the mentioned-anywhere / depth-1 baselines, plus decoy helpers and shuffled order (N6); reconcile K-C2's 2,000 with the 5,000 reservoir | CPU only |
| F5 | Fix the statistics: H2b margin (≥0.07, or drop it); recompute HS-spec power honestly; state HS as a fixed-arm contrast or use arm-level df; aggregate before KR; specify the H1 df as 4 | 0 |
| F6 | G0 mask test: bitwise invariance of other suffixes' logits to a perturbed label, on both sliding and global layers. Per-suffix position-ID reset. Forced-continuation level readout handling digit tokenisation | Inside L1 |
| F7 | Put back, as conditional lines: the small-model replication, and an eval-only large serialised-graph probe. Log in the change log every v2 fix dropped in v3 (§1.2) | ≈2–2.5 GPU-h conditional |
| F8 | Run the hostile prior-art pass on the dial (C1) *before* GPU. Include controlled-spurious-correlation / shortcut-dose literature and dK-series primaries | 0 GPU-h |

---

## 6. Verdict and the five most important residual risks

**Verdict: NEEDS-FIXES.**
- The CPU kill programme (K-C1 to K-C4, K-C7) is sound and can start once F4 is applied, which is a table edit.
- GPU work must wait for F1 (the budget is wrong by ≈9 h central) and F2 (the headline metric re-imports a break v2 had already fixed).
- It is not BROKEN: H1 and H2 are well specified and powered, the CPU kills are cheap, and every outcome row is reportable.

**Top 5 residual risks:**

1. **Budget (VERIFIED arithmetic).**
   - Training is under-counted by 2× (two epochs), so the corrected total is ≈33.8 central and ≈59 pessimistic.
   - On top of that: throughput assumes a packing speed-up that masked SDPA does not give (N7), and real-item length is unstated (N9).
   - The likely outcome is forced descoping that removes HS-spec.
2. **The headline metric E2 is confounded (K-34 regression).**
   - Implausible rewired real code plus pretraining memory lets content-free SFT "win" H1 by distrusting plausibility, not by reasoning better.
   - v2's controls were dropped silently.
3. **The dial's "only when" is unidentifiable and under-powered.**
   - Nested nulls are collinear with null strength.
   - Break arms are 50% real.
   - Reliance conflates reading skill with prior use.
   - HS-spec power is ≈0.14–0.47 and goes to ≈0 after descoping.
   - The [CLAIMED] sentence will likely shrink to "nulls reduce prior reliance".
4. **Construct leaks in items and conflict sets.**
   - Shown-set, depth-1 and order shortcuts (N6).
   - Overlapping conflict families and partial operator home advantage (N4).
   - Layered or small graphs where near ≈ mid (N3).
   - Any of these can produce or erase the HS pattern for reasons unrelated to the dial.
5. **The user's target is mostly unmet, and novelty is thinly checked.**
   - No RL is run.
   - No large-graph evidence (v2's deferral was dropped).
   - No second model (v2's L11 was dropped).
   - "Near SOTA" is retention only.
   - The [CLAIMED] dial rests on 6 searches, with Xuan et al. at closeness 2 and the hostile red team still pending.
