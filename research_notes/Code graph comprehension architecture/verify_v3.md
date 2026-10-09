# Verification of architecture v3 (BMM, final), 9 Oct 2026

**Target.** `architecture_v3_final.md`, checked against `break_v1_logic.md` (L1–L46), `break_v1_method.md`, `break_v1_priorart.md` (BR3), `break_v2_logic.md` (N1–N30), `break_v2_priorart.md` (BR3b), `phase3_critic.md`, `phase3_redteam.md`, plus `architecture_v2.md` §11 (to catch v2 fixes that v3 dropped), `docs/training_feasibility_4xl4.md` and `docs/HQ.md`.

**Labels.**
- **VERIFIED**: checked in the files, or by arithmetic I re-ran (python, normal approximation).
- **INFERRED**: my reasoning about how the specified algebra or procedure behaves.

Nothing was run except arithmetic.

---

## 0. Verdict in six lines

1. **NEEDS-FIXES, but none of the fixes blocks the Day-1 CPU kills.** K0, K1, K2 and K3a can start today. About 0.5 pd of spec fixes (§6) should land before any K4–K6 code is written and before the K9 GPU run.
2. The change log maps **every** N-item (N1–N30 plus N-orient, N-tune, N-ctrl, N-gold). VERIFIED. **Six are only partly fixed or renamed:** N1, N2, N8, N10, N17 and N-tune. Two v2 fixes for v1 breaks were **silently dropped**: the ε-skip over wraps/decorators (L9, and critic item 5) and several diagnostics (L4, L10, L11, L13).
3. **One claimed novelty component does nothing to the ranking.** Subtracting the per-step local median (the "local base rate") is a constant for every node at that step, so it cancels between functions, exactly as q̄ did in N1. The "IDF-like / base-rate" ingredient in §0.1(iii), §0.2 and the §0.5.3 headline only calibrates match against insert/delete. It is not a contrast between nodes. VERIFIED (algebra).
4. **The power table arithmetic is correct** (VERIFIED to ±0.01). Two caveats:
   - It rests on a GUESSED SD of 0.6, against v2's 0.9.
   - Its 250-task stratum needs a ≈42% effect share, but K3 passes at 35% (210 tasks).
5. **Budget totals are right, except** the optimistic person-day total (14.0, not 15.0), the ladder floor (14.25 pd, not ≈13.5) and G2 (≈1.9 NH central, not 1.4).
   - **The ladder does not preserve the core claim.** Rung 8 (EVAL 400) puts the C-BEH stratum below K3b's own 150-task threshold unless the effect share is ≥37.5%.
6. **Fit to Shrey's target is partial.** BMM is a localisation *decoder over* a code graph with behaviour-typed node annotations. It is not a code-graph generation, parsing or embedding technique; the graph-construction parts are explicitly "infrastructure, not claimed". It is novel at closeness 2 as a combination, verified on code and cheap in GPU. It is not cheap in person-days.

---

## 1. Audit of the v2 → v3 change log

### 1.1 N-items (break_v2_logic)

All 34 IDs appear in §9.1 (VERIFIED). Each verdict below compares the §9 claim with the v3 body text.

| ID | v3 claim | Verdict | Evidence / residual |
|---|---|---|---|
| N1 | Local median contrast + all-bit μ′ + LOC rule | **Partly fixed; the "local base rate" is renamed.** | med_v[·] in §1/§1.1 is a single number per (step, bit), the same for every v. It cancels in every function-to-function comparison and only moves match vs insert/delete/∅. The line "it makes the term a contrast between nodes rather than a constant" is **false** (VERIFIED algebra). The omnivore fix rests **entirely on μ′**. The breaker's node-specific neighbourhood normalisation (N1 fix 1) was not adopted. The η floor is fixed |
| N2 | NEG/PEER separable; missing-call via predecessor readout | **Partly fixed.** | NEG/PEER are separable (X-NEG0, X-PEER0, R7). But §1 applies the predecessor-readout choice **only to the "+exception" stratum**, while N2-ii (missing call) is a **"−" polarity** case. The claim "missing-call reading handled by predecessor readout" is not wired to "−" steps. Low peer coverage for plain functions is reported, not fixed (acceptable) |
| N3 | AST for implicit raises, g_φ for intended bits | **Fixed (the breaker's minimal option).** | Residual: KeyError, IndexError, TypeError and AttributeError can be both implicit (AST) and explicitly raised (g_φ "explicit raise C"). How the two sources combine into one bit is unspecified (§2, E-6) |
| N4 | Span-cited codes, "none", invented-code rate | **Fixed.** | Minor: the substring check verifies that the span exists, not its polarity ("doesn't raise" can be cited for +C) |
| N5 | Per-class repo-exception bits, MRO-closed | **Fixed, but opens a leak under R2** (E-5) | — |
| N6 | K3 bands 35/20; K3a + K3b; switch before EVAL | **Fixed.** | §4.3's stratum of 250 implies ≈42% share, not 35% (§3) |
| N7 | h_ψ dropped; uniform p(s\|u) | **Fixed** (happy-path bias removed) | **New:** the move threshold now scales with site count, so large functions become "sticky" (E-7) |
| N8 | Delete states; no readout mass for inserts; η₀ tuned; P1a with siblings | **Partly fixed.** | The entry no longer absorbs the readout. But v3's "delete" **consumes** the observation, so it is not a silent profile-HMM delete. Omitted intermediate steps (the dominant LLM error, since RepoReasoner recall is ≈0.31) are bridged only by the 2-hop teleport. A delete at the **last** step gives its destination no readout mass, so structural correction at the culprit end earns nothing (E-3) |
| N9 | Centred similarity | **Fixed** | §1.2 threshold arithmetic: ln(0.05/(0.80/25)) = 0.446 nats, VERIFIED |
| N10 | Local 2-hop teleport + 0.2π | **Partly fixed / renamed for large repos.** | The 2-hop neighbourhood of 20 nodes in django/sympy can hold thousands of nodes (hubs). Per-node mass 0.08/|N₂| ≈ 3e-5, against ≈0.032 for a call edge: a gap of ≈7 nats (INFERRED), close to v2's 8.7. "Emission-based so the HMM stays exact" is correct: the matrix is time-inhomogeneous but depends only on observations |
| N11 | Undirected PageRank, λ = 0 default | **Fixed** | — |
| N12 | Site-local ∅_s, registry ∅_r | **Fixed** | — |
| N13 | P1a on CAL-tune; EVAL-exec read once | **Fixed** | — |
| N-orient | Orientation max for all controls | **Fixed, with a new hole:** under a rank-1 A the two orientations tie (E-4) | — |
| N-tune | 4⁴ grid, 50 kh | **Partly fixed, possibly worse.** | The breaker suggested 4³ (64 points) or BO and estimated ≈60 kh for 12 arms. v3 uses 4⁴ = 256 points, 14 arms and two views, at 50 kh central. If every arm is tuned for tuning parity (L39), a recount gives ≈100–300 kh (INFERRED, §3.3). v3 does not say which arms are tuned |
| N14 | X-BAG = rank-1 PPR HMM | **Fixed** | Residuals: insert self-stay makes it not purely i.i.d.; orientation tie (E-4) |
| N15 | Top-10 Jaccard ≥ 0.8 | **Adopted, but under-specified** (E-4) | — |
| N16 | X-LEXEFF⁺, static/learned split | **Fixed** (descriptive) | "Non-exception bits have no lexical analogue" is false for warns (`warnings.warn`, `…Warning` classes), `return None`, `return self` and `self.x =`. Harmless, because C-BEH-static is restricted to exception steps, but the sentence should be corrected |
| N17 | Pre-rename identity map for LEXEFF⁺; builtin/repo split; R8 | **Control fixed; treatment leak not addressed** (E-5) | — |
| N18 | (BMM + BODY) − (β = 0); shared BODY | **Fixed in form; ambiguous in content.** | μ′ sits **outside** β in the §1 emission. X-β0 then still uses the stated-effect set (j ∉ E⁺∪E⁻) and q (E-2) |
| N-ctrl | DPR within package; X-NULL donors; scope states removed | **Fixed** | Graph-free = X-BAG with a *uniform* prior removes both graph and π: two changes (E-11) |
| N19 | Pool arithmetic, dedup, CAL-tune exclusion, by-source | **Fixed** | — |
| N20 | B-RECALL, docline renaming | **Fixed, with a mislabel** (E-8) | — |
| N21 | Truncated log₂-rank; X-BAG tail; gold-in-G0 | **Fixed** | — |
| N-gold | Min-over-gold; hunk rules | **Fixed** | — |
| N22 | Single metric, one control, exception-line stratum | **Fixed, with two defects.** | (a) The falsification sentence in §4.2 is garbled. "CI excludes Δ ≤ −0.1" and "CI upper bound ≥ 0" are different conditions, and failing to reject is not falsification. (b) The stratum size is inconsistent with K3 (§3.1) |
| N23 | C-USE underpowered; wild cluster bootstrap | **Fixed** | — |
| N24 | Selective prefix, token, socket sandbox | **Fixed** (Shrey's token stated as a blocker) | — |
| N25 | R1 dropped | **Moot** | — |
| N26 | Raiser vs caller stratum | **Fixed** | — |
| N27 | 19.0 pd central, floor 13.5 | **Central matches the breaker's lean target (17–19).** | Optimistic total mis-summed (14.0); floor mis-stated (14.25 from the ladder's own savings); week-1 load infeasible (§3.4) |
| N28 | G0 excludes tests, 6–20k | **Fixed** | — |
| N29 | Scope states removed | **Moot**, but the exclusion has a bias (E-9) | — |
| N30 | DEV dropped / labelled | **Fixed** | — |

Other break_v2 items:
- §9 collapse (BER's own controls; the two missing rows): fixed (BER row in §3.2; R5, R6).
- §11 "also cheap" items: all adopted.

### 1.2 v1 breaks and v2 fixes that v3 silently dropped

v3 §9 maps only N-items. Diffing against v2 §11.1 (VERIFIED by grep) shows the following.

| v1 break | v2 fix | v3 status |
|---|---|---|
| L9 (hubs, skip-as-teleport); critic HYTRACE item 5 ("a skip-transition ε is essential" for decorators, `__getattr__`, generators) | ε-skip only over wraps/decorator edges or sub-hub intermediates | **Dropped.** ε no longer exists, and δ-delete is not a skip over a wrapper. Under ladder rung 4 (HEAP dropped, AST-only graph) unresolved decorator wrappers break paths with no skip |
| L4 | C-DEC also reported by path length | **Dropped** (only X-SHUF keeps T ≥ 3). Short paths (T ≤ 2) dilute C-DEC toward 0 |
| L10 | Family-size stratum | Dropped |
| L11 | "Excluding entry" diagnostic | Dropped. It is still useful as a check on the new delete/readout rule |
| L13 | Self-loop cap | Not mentioned. Recursion plus insert-stay is unspecified |
| L18 | Residual-name share (dunders, overrides, string dispatch); C4 on low-residual repos | Narrowed to "file and module paths' residual share". The dunder/override residual is not reported |
| v2 §5 B-CRASH (TB stratum) and the CoSIL optional run; RT hostile (b) "agentic graph search (CoSIL, GraphLocator) is the strong baseline" | Optional rungs | **Dropped to "cited".** A reviewer will still ask (§5) |
| Critic HYTRACE item 1 ("first + last steps only" arm) | Not in v2 either | Absent (minor) |

L22 and L42–L46 were skipped legitimately (E-TR and Study B are cut). Everything else in L1–L46 is carried forward (VERIFIED by spot-checks: L1→N-gold, L3→N12, L5, L6, L19, L21, L23–L41).

---

## 2. New edge cases against v3's specific changes

### E-1. The local median is ranking-neutral, so N1 is "fixed" only by μ′ (Major for the novelty text)
- For step t and bit j, med_v log((1−η)q_vj+η) is one number per step, independent of v. So ℓ_v − ℓ_v′ is unchanged by it. VERIFIED algebra.
- It does change the ratio of match states to insert, delete and ∅ (a centring role, like N9).
- **Consequence 1.** §1.1's "contrast between nodes" is wrong.
- **Consequence 2.** Conjunction element (iii) in §0.1 ("normalised against a local base rate") and the §0.2/§0.5.3 wording ("log-ratios to a local base rate") present as a scoring idea something that is only a match/insert calibrator. No ablation isolates it.
- **Fix.** Either drop (iii) from the novelty text, or make the base rate node-specific: contrast against v's own peers or call-site co-candidates (N1 fix 1), and add an "X-noBR" ablation.

### E-2. μ′ is outside β, so X-β0 still sees the issue's effect statement (Major for C-BEH)
- In §1, ℓ = τ·[…] + BODY + β·E⁺ + β·E⁻ − μ′·Σ_{j∉E⁺∪E⁻}(q_vj − med q_·j)₊.
- If X-β0 sets only β = 0, the control keeps a term whose bit set depends on **which effects the issue stated**, evaluated on q. The control is not behaviour-free, so C-BEH under-measures, or measures something undefined.
- On "~value" and "none" steps (E = ∅), the μ′ term penalises every effectful function **in every arm**, a bias toward trivial getters (INFERRED). §1.1 says these "emit nothing", which contradicts the formula.
- **Fix.** Write the emission as β·(E⁺ + E⁻ − μ′·Σ…), so X-β0 zeroes all q-dependent terms. Apply μ′ only on steps with E ≠ ∅.

### E-3. Delete at the culprit end gives no credit to the node it reaches (Major for "structure corrects hallucination")
- The highest-weight step (w_T ∝ 1) is the LLM's culprit guess. This is the step most likely to be hallucinated when localisation is hard.
- If it is decoded as delete(v→callee), the observation is skipped and the position advances one hop. The callee, which the graph says is the likely culprit, receives **no readout mass**, because readout is match states only.
- If decoded as insert, v gets nothing at T either.
- The mechanism §0.5.2 sells ("a hallucinated step can be skipped instead of misleading") therefore cannot *place* a culprit. It can only stop a wrong one from scoring.
- Separately, deletes consume observations, so **omitted** intermediates (LLM paths are short, with RepoReasoner recall ≈0.31) are bridged only by the 2-hop teleport.
- **Fix.** Pre-register one of two options: give delete destinations at step T readout mass w_T·γ_T(del→v), or add a silent skip state (no observation consumed) limited to 1–2 hops. Add the dropped "excluding entry" and "by path length" diagnostics. Check E-3 directly in P1a by substituting the **last** step, not only middle steps.

### E-4. X-BAG orientation tie and the Jaccard rule
- **Orientation tie.** With A = 𝟙·PPRᵀ, the sequence likelihood Π_t Σ_v PPR(v)e^{ℓ_v(o_t)} is the same for both orientations. Only insert self-stays break the tie, and only weakly. The "orientation max" in X-BAG is then a tie, and the readout weight w_t ∝ (t/T)² falls on the first or last step depending on the tie-break. That rule is unspecified, and it can move C-DEC by about the predicted effect on reversed-narration issues (INFERRED).
  - **Fix.** Give X-BAG the orientation chosen by BMM for that sample, so orientation detection does not count as "sequence", or pre-register forward orientation and report both.
- **Jaccard.** On top-10 sets Jaccard is discrete: 9 of 10 shared = 0.818, 8 of 10 = 0.667 (VERIFIED). So "≥0.8" means "≥9 shared".
  - It is blind to reordering inside the top 10 (rank 8 → 1 gives Δm = −3 with Jaccard 1.0).
  - It is blind to ranks 11–100, which the truncated metric counts.
  - So it can relabel a **significant** C-DEC as "bag + prior". §3.2 does not say which wins.
  - **Fix.** Use rank-biased overlap or Kendall-τ over top-100 as the diagnostic, and state that a Holm-significant C-DEC is reported as a sequence effect whatever the diagnostic says.

### E-5. Under R2, BMM resolves repo exceptions through the identity map, so names leak into the treatment arm (Major; L17/N17 in a new door)
- The issue (not renamed) says "+InvalidURL". The renamed repo has `C_8f2`.
- v3 gives X-LEXEFF⁺ the identity map. It never says how **BMM** creates the "raises InvalidURL (MRO-closed)" bit for a class that no longer exists by that name.
- The only ways are the identity map, which is name information that X-β0's BODY term (renamed tokens) lacks, or falling back to "any repo exception".
- With the map, C-BEH under R2 can pass by un-renaming exception classes. R8 catches this only post hoc.
- **Fix.** Pre-register that under R2 BMM maps an issue-named repo exception to "raises any repo exception (MRO-closed)" with no identity map. Alternatively, make the **builtin-only** stratum the C-BEH primary and report repo-exception steps as secondary.

### E-6. AST implicit-raise bits saturate, and the dual-source bits are undefined (Major for the effect channel's informativeness)
- Almost every function does attribute access and calls, so q(AttributeError) and q(TypeError) ≈ 1 almost everywhere. Two-hop propagation saturates them further.
- After the median shift the contrast is ≈0 for nearly all nodes. The modal accidental bug ("'NoneType' object has no attribute …") therefore carries **no** behavioural signal, only a −3β filter on the rare functions without attribute access (INFERRED).
- KeyError and IndexError have some range. KeyError is also an "explicit raise C" (g_φ), and how the two sources combine (max, noisy-OR or AST-only) is unspecified.
- **Fix.**
  - Report the per-bit dynamic range of q over the G0 nodes on CAL-tune before K9.
  - Pre-register the combination rule (noisy-OR).
  - Consider receiver-typed AST rules ("attribute on a parameter that can be None", "subscript on a dict parameter without .get/in-guard") instead of bare possibility. That costs about 0.3 pd and is still pure AST.

### E-7. Uniform transitions over call sites make big functions sticky, and the 2-hop teleport is diluted by hubs
- The move threshold is ln(η₀·|sites|/0.80): 0.45 nats at 25 sites but **1.83 nats at 100 sites** (VERIFIED), back at v2's 1.61.
- If builtin and external calls count as sites, typical django/sympy methods have 30–100. The decoder then prefers to insert inside long functions, a size bias stacked on the LOC concern.
- Repeated calls to one helper (`self._check(x)` ×10) multiply its transition mass ×10.
- **Fix.** Count only repo-resolved sites in p(s|u), dedupe sites by callee, and report the threshold distribution on CAL-tune.

### E-8. B-RECALL "memorisation index" conflates memory with the issue naming the gold
- HQ A1 (VERIFIED) found the gold named in the issue for 24% of tasks (dev set). With the issue + repo name, Gemma "recalls" those by reading, not by memory.
- The miss stratum is still a valid *hard* stratum. But "B-RECALL hit = memorised" is a mislabel, and §0.3's interpretation ("issues Gemma cannot recall") inherits it.
- The name-matching rule (qualified vs bare names) is unspecified.
- **Fix.** Add a gold-name-in-issue string covariate. Define memorisation = B-RECALL hit ∧ gold not named in the issue. Pre-register the matching rule and the sampling temperature.

### E-9. Excluding non-function gold biases the "−" polarity stratum
"Should reject X / should raise but doesn't" fixes often edit class-level regexes, attribute tables or default dicts (for example validator regex class attributes). INFERRED, not counted. Those are exactly the omission-fault issues where PEER/NEG are meant to help. **Fix:** report the excluded share by polarity at K0. Consider mapping class-body golds to the class's methods that read the attribute, as a secondary stratum.

### Short extras
- **E-10.** N2-ii is not wired to "−" steps (see N2 above).
- **E-11.** Graph-free = "X-BAG with a uniform prior" removes PPR *and* π (BM25 ⊕ CRE): two changes. Use π itself.
- **E-12.** Ladder rung 7 (k = 5) silently changes the stratum rule "span-verified code in ≥3 of 10 samples". Pre-register ≥2 of 5.

---

## 3. Arithmetic recheck

### 3.1 Power table (VERIFIED)
Two-sided Holm-first α = 0.0167 gives z = 2.394. Power = Φ(|Δ|/(SD/√n_eff) − z). DE = 1 + (m−1)·ICC.

| Row | v3 | Recomputed | Note |
|---|---|---|---|
| C-BEH, 250 tasks / 20 repos, ICC 0.05 → 0.15 | n_eff 155→90; 0.96→0.79 | n_eff 159→92; **0.965→0.788** | ✓ |
| C-BEH at Δ = −0.15 | 0.77→0.50 | 0.775→0.500 | ✓ |
| C-DEC, 600 / 25 repos (m = 24) | 280→135; 0.99→0.82 | 279→135; 0.991→0.822 | ✓ |
| C-USE | 0.39→0.18 | 0.380→0.173 | ✓ |
| Joint C-BEH ∧ C-DEC | 0.95→0.65 | 0.956→0.648 | ✓ |
| All three | 0.37→0.12 | 0.363→0.112 | ✓ |

**Caveats** (VERIFIED arithmetic, INFERRED interpretation):
1. **Stratum size vs K3.** "250 tasks (K3 ≥ 35%)" is inconsistent: 35% of 600 = 210. At 210 tasks, n_eff is 142→87 and power 0.94→**0.76**. At ICC 0.15, n_eff falls below the "n_eff ≥ 90" condition in the §4.2 falsification rule, so the rule cannot fire.
2. **At K3b's floor (150 tasks):** n_eff 113→76, power 0.88→**0.70**.
3. **SD sensitivity.** SD(Δ) = 0.6 is GUESSED, a third below v2's 0.9 (untruncated), while Δ was cut only 20% (0.25 → 0.20).
   - At SD 0.9: C-BEH power 0.66→**0.40**.
   - K10 tolerates SDs up to 1.3× assumed (0.78). At that SD and ICC 0.15, power is ≈**0.53** with no trigger firing.
4. **The B-RECALL-miss sub-stratum** that §0.3 calls *the* result: about 125 tasks at a 50% miss rate gives power 0.82→0.65 at Holm α. It is a sub-claim, not a primary. **§0.3's "single result" is not one of the three powered primaries.** §4.2 only requires it to be "also negative" (point estimate).
5. "R5 is the most likely outcome" holds **only conditional on the GUESSED effects being real**: P(R5) ≈ 0.96×0.99×0.62 ≈ 0.59 → 0.79×0.82×0.83 ≈ 0.54. Unconditionally, R3, R4 or R6 may be more likely. Say "most likely if the effects are as predicted".

### 3.2 GPU (NH)
- Column sums: 6.9 / 9.95 / 15.55 (VERIFIED).
- **G2 is under-counted.** R2 is a full G1 re-run at k = 10 on 600 of the 750 G1 tasks. Scaling G1 by 600/750 gives **1.28 / 1.92 / 3.04**, not 1.0 / 1.4 / 2.2.
- The corrected total is ≈7.2 / **10.5** / **16.4** NH. It still fits: 16.4 / ≈120 quota-h ≈ 14%. Quota is 30 h/week per HQ (VERIFIED: "24.5 h used, 5.5 h left").

### 3.3 CPU (kernel-h)
- Sums: 115 / 235 / 538. Wall time at 5 concurrent: 23 / 47 / 108 h (VERIFIED).
- **EVAL-exec + CAL-tune traces** (100 tasks) scale from the breaker's 35 kh for 150 tasks to ≈23 kh, not 15 (INFERRED).
- **Decoding.**
  - Per tuned arm-view: 256 grid points × 100 tasks × 10 samples × 2 orientations ≈ 512k decodes. At 0.1–0.3 s per decode that is ≈14–43 vCPU-h, or ≈3.5–11 kernel-h.
  - If all 14 arms × 2 views are tuned (L39 parity), decoding alone is **≈100–300 kh**.
  - If only BMM is tuned and the controls inherit its parameters, ≈20–40 kh.
  - **Specify which arms are tuned.** Otherwise the 50 kh central is unsupported.

### 3.4 Person-days
- Column sums are **14.0** / 19.0 / 27.25. v3 states 15.0 for the optimistic case, a harmless mis-sum (VERIFIED).
- 19 / 27 days = 70% (VERIFIED). Oct 9 → Nov 5 is 27 days.
- **Floor:** the ladder's stated savings (0.5 + 0.5 + 1.5 + 1.75 + 0.5) give 19.0 − 4.75 = **14.25 pd**, not ≈13.5 (VERIFIED). The breaker's own floor was 13.75.
- **The week-1 kill load does not fit (INFERRED).**
  - K4, K5 and K6 need, by Oct 17: the EVAL builder (2.0), AST resolver + graph (1.75), the tracer and 40 CAL-tune traces (≈1.5 of 2.5), g_φ + AST rules (1.25 + 1.5 for K4), the decoder (2.5), plus the Day-1 counts (1.5).
  - That is ≈11–12 pd in 9 calendar days, at ≤1 pd/day shared with the main track.
  - K0, K1, K2 and K3a fit. **K4–K6 will realistically slip to about Oct 21–23**, which compresses K9 (Oct 22).
  - Re-date them now, or move K4 behind K9 (its fail path is harmless: AST q everywhere).

### 3.5 Does the ladder preserve the core? Partly
- **Components:** yes. The core list is never cut.
- **The core claim:** no.
  - **Rung 8 (EVAL 600 → 400):** at a 35% share the stratum is 140, below K3b's 150 threshold, which by v3's own rule **demotes C-BEH to a secondary estimate**. Power is 0.86→0.68 at n_eff 108→74. It stays primary only if the share is ≥37.5%.
  - **Rung 7 (k = 5)** changes the stratum definition (E-12).
  - **Rung 4 (no HEAP)** removes decorator unwrapping, and with ε gone (§1.2) that hurts paths through decorators. This is a C-DEC risk, not a core-claim cut.
- **P1a gate (K6/K9)** needs executed CAL-tune tasks, i.e. the tracer. The tracer is therefore de facto core but is not on the "never cut" list. Add it.

---

## 4. §0.5 "existing idea + new use" audit

### 4.1 Closeness values against the prior-art notes

| Row | v3 closeness | Source says | Verdict |
|---|---|---|---|
| E1 Newson–Krumm; TSP'18 | 2 (RT) | RT P3: 2 for the decoder | ✓ |
| E2 FSE'24; Euler | 2.5 (decoder); 2 | BR3b: 1 to the piece, 2.5 for the decoder; BR3: Euler 2 | ✓ |
| E3 EXCEPT, SBEST, CrashTracker | "1.5 each, BR3" | CrashTracker 1.5 (BR3); EXCEPT and SBEST 1.5 come from **RT P5 / CR**, which BR3 only re-cites | ✓ values; attribution slightly off. **FaR-Loc (1.5, BR3), the closest "behaviour-described retrieval", is missing from this row** |
| E4 IPA-GNN, CodeFlow | 2, 2 | BR3: 2, 2 | ✓ |
| E5 QMR-DT, possibilistic MBD | 2, 2 | BR3b: 2 (SNIPPET), 2 | ✓ |
| E6 Chucky, Engler | 2, 1.5 | BR3b: 2 (SNIPPET), 1.5 (INFERRED) | ✓ |
| E7 LocAgent | 1.5–2 | BR3: 1.5–2 | ✓ |
| E8 RepoReasoner, SWE-Flux, TraceEval | 2, 1.5, 1.5 | BR3: same | ✓ |
| §0.1 "novel piece 2, architecture 2–2.5", "≈55 queries" | — | BR3b: 1.5→2 and 2→2–2.5; BR3 ≈30 + BR3b ≈25 queries | ✓ |

**Missing rows** for existing ideas that the headline itself uses:
1. **Liblit Increase / IDF (closeness 2, BR3b).** The "local base rate" has no row, and no ablation (and E-1 shows it is ranking-neutral).
2. **SherLog / Pensieve / Pathidea / CrashLocator / COCA (1.5–2, BR3).** BR3's hostile point is that reviewers will read BMM as "SherLog/Pensieve with an LLM as the log". The "new use" against that family (imagined from prose, soft emissions, exact inference) is stated nowhere in §0.5.
3. **RoG/GCR (2, RT).** Named only in §0.1.
4. **AstroBR (2 overall, BR3b).** It already has an LLM extract S2R steps and path-map them onto a graph, which weakens E2's "new input: LLM steps" (BMM's are *imagined*, not *extracted*; say so).

### 4.2 Does each claimed change map to a real control? (VERIFIED against §3.2, §4 and §5)

| Row | Control exists? | Problem |
|---|---|---|
| E1 → C-DEC (X-BAG), X-SHUF, X-DPR | Yes | E-4 (orientation tie, Jaccard rule) |
| E2 → B-ALIGN | Yes (secondary) | "Same ℓ inputs" is ambiguous: does B-ALIGN get the behavioural terms? If not, BMM − B-ALIGN mixes inference with emissions. Specify ℓ incl. β |
| E3 → C-BEH (X-β0), X-LEXEFF⁺ | Yes | E-2 (μ′ leak) and E-5 (identity-map leak) contaminate exactly this control |
| E4 → K4, X-ASTEFF | Yes | — |
| E5 → X-NEG0 | Yes (secondary) | — |
| E6 → X-PEER0 | Yes (secondary) | Peer coverage may make it unestimable |
| E7 → C-USE | Yes | Underpowered (0.38→0.17), stated |
| E8 → M1 | Descriptive only | Fine |
| "Local base rate" (§0.1 iii, §0.2, §0.5.3) | **No control** | Ranking-neutral (E-1) |
| "Per-step attachment of effects" (§0.5.2, "which step each effect belongs to") | **No direct control** | The nearest is X-BAG (which keeps per-step attachment) and X-SHUF. Add "effects pooled over all steps" (effect set attached to every step) as a one-change secondary |
| "Name-free" (§0.5.2) | R2 + C4 | Leaks: E-5; file paths unrenamed (stated) |

### 4.3 Is the §0.5.3 headline honest? Mostly; five edits needed

1. "HMM map-matching (known … from FSE'24 …)". BR3b says FSE'24 is a Markov model with **heuristic additive** scores and DP top-k. Say "Markov/DP alignment (FSE'24) and HMM map-matching (GPS)". Do not call FSE'24 an HMM.
2. "and **make it work** for issue localisation". This presumes the result. Say "adapt it to …".
3. "scored as **local-base-rate log-ratios**". Inert for ranking (E-1). Drop it, or fix and ablate it.
4. "(AST rules plus an **execution-trained** model)". This must be conditional on K4. §5 K4's fail path says to drop it, but §0.2 and §0.5.3 state it unconditionally.
5. "combined with peer contrast for missing effects **(new)**". Read literally, this claims peer contrast is new, against must-not-claim item 15 (Chucky/Engler). Say "peer contrast (Chucky) applied to issue-stated missing effects".
- "Name-free behavioural channel" should become "a behavioural channel robust to code-side renaming" (paths and, per E-5, possibly exception names leak).

The fallback sentence (R3/R4) is honest.

---

## 5. Does v3 meet Shrey's target?

Target: *a novel technique (existing idea + new use/improvement is acceptable) for code-graph generation/parsing/embedding that enhances deep comprehension of complex repositories, verified on code, cheap, generalisable.*

| Criterion | Verdict | Why |
|---|---|---|
| Novel (existing + new use) | **MET, conditionally** | Closeness 2 as a conjunction (BR3b, VERIFIED). Each part's new use is stated, and most have a control (§4.2). It is only a *technique* paper if C-BEH passes (and C-DEC for R1/R5); otherwise it is an analysis paper (R3/R4) |
| **For code-graph generation / parsing / embedding** | **NOT MET in the strict sense; PARTIAL in a generous reading** | What is novel is **inference over** a call graph (decoding an issue onto it) for **function localisation**. Graph generation (AST resolver, HEAP, site-local ∅, registries) is declared "infrastructure, not claimed" (§0.5.1, VERIFIED), because RT found it near prior art (Jelly, HeapDL, PoTo, AFLGopher). The nearest "embedding/parsing" element is the **behaviour-typed node annotation** q_v ∈ [0,1]^B (an effect-signature node feature). It is novel only if K4 passes (else AST rules) and is not claimed as the contribution. To meet the target literally, the paper would need to be framed as "**behaviour-annotated call graphs** + issue-to-path grounding", with q_v evaluated as a representation (bit-level AP, K4) and not only through localisation |
| Enhances deep comprehension of complex repositories | **PARTIAL** | Measured only through function-level localisation rank (a comprehension *proxy*). M1 (imagined vs executed paths, typed divergences, memorisation) is the genuine "comprehension" analysis but is descriptive. EVAL includes django/sympy/sklearn (complex) |
| Verified on code | **MET (planned)** | ≥600 tasks, ≥25 repos, static gold, a 60-task executed subset, wild cluster bootstrap. Nothing run yet |
| Cheap | **PARTIAL** | GPU ≈10.5 NH central (≈9% of quota) and CPU-only decoding at inference: cheap. Person-time 19 pd central (70% of remaining days), pessimistic 27 pd not fitting: **not cheap to build** |
| Generalisable | **PARTIAL** | No tests at inference; the AST-only path runs on hidden repos; ≥25 repos; a second generator (Qwen). But Python only. HEAP needs importable envs. Main-track transfer (HQ: hidden private repos, agents read ≤10 candidates) is not planned: no agent integration or LB probe |

**Blunt summary.** v3 is a well-controlled **issue-localisation** method that *uses* a code graph. It is not a code-graph construction or embedding technique. If Shrey's target is literal, the honest options are:
- (a) accept the "graph reasoning / inference on code graphs" reading, which HQ's Graph Reasoning track supports; or
- (b) re-centre the paper on the behaviour-annotated graph (q_v as node features), which makes K4 a primary and costs the "AST-only" fallback its novelty.

---

## 6. Verdict

**NEEDS-FIXES.** These are spec-level fixes, about 0.5 pd. Do not let them block K0, K1, K2 or K3a, which can start today. Land them before K4–K6 code and the K9 GPU run.

| # | Fix | Addresses |
|---|---|---|
| F1 | Put μ′ inside β; apply μ′ only on steps with E ≠ ∅; state that X-β0 zeroes every q-dependent term | E-2 / N18 |
| F2 | Under R2, map issue-named repo exceptions to "any repo exception" with no identity map in BMM; or make builtin-only the C-BEH primary stratum | E-5 / N17 |
| F3 | Delete at step T credits its destination (or add a silent 1–2-hop skip state); restore the ε-skip over wraps/decorators; P1a also substitutes the **last** step; restore "excluding entry" and "C-DEC by path length" | E-3 / N8 / L4 / L9 / L11 |
| F4 | X-BAG uses BMM's per-sample orientation; replace or augment Jaccard with RBO/Kendall over top-100; state that the diagnostic never overrides a Holm-significant C-DEC | E-4 / N15 |
| F5 | Remove "local base rate" from the novelty sentences, or make it node-specific with an X-noBR ablation; fix the §1.1 sentence | E-1 / N1 |
| F6 | Restate C-BEH n at the 35% share (210); fix the §4.2 falsification sentence; drop "n_eff ≥ 90" or make the floor reachable; state power at SD 0.78/0.9 | §3.1 |
| F7 | Specify which arms are tuned; recost decoding CPU; recost G2 (+0.5 NH) | §3.2–3.3 |
| F8 | Re-date K4–K6 to about Oct 21–23 (or move K4 after K9); fix the optimistic total (14.0) and the floor (14.25); add the tracer to "never cut"; at rung 8, state that C-BEH demotes unless the share is ≥37.5%; set k = 5 → "≥2 of 5" | §3.4–3.5, E-12 |
| F9 | Wire the predecessor readout to "−" steps (missing call); pre-register a noisy-OR for dual-source bits; count only repo-resolved, callee-deduped sites | N2, E-6, E-7 |
| F10 | B-RECALL: add a gold-named-in-issue covariate and a matching rule; graph-free control uses π, not uniform; report non-function-gold exclusions by polarity | E-8, E-11, E-9 |
| F11 | §0.5 edits: the five headline edits in §4.3; add rows for Liblit/IDF, SherLog-family, RoG/GCR and AstroBR; add the "effects pooled across steps" ablation | §4 |

### Five most important residual risks

1. **Schedule (INFERRED).**
   - 19 pd central is 70% of the remaining days.
   - The week-1 kill set needs ≈11–12 pd by Oct 17.
   - Pessimistic (27.25) does not fit, and the real floor is 14.25.
   - The main track (Dec 2) is squeezed. This is the most likely failure mode, and it is independent of the science.
2. **The behavioural channel may be near-empty for the modal bug.**
   - AST implicit bits saturate (E-6).
   - The base rate is ranking-inert (E-1).
   - The effect-statement share is uncertain (K3: the breaker's prior is 15–22% by regex).
   - R3/R4/R6 are realistic outcomes, not tails.
3. **C-BEH validity leaks** (E-2 μ′ outside β; E-5 identity map under R2). Either can make C-BEH pass or fail for a reason other than the mechanism. They are cheap to fix and must be fixed before K9.
4. **Power fragility.**
   - The core's 0.79–0.96 depends on a GUESSED SD of 0.6 and a 250-task stratum that K3's pass bar does not guarantee.
   - At realistic values (210 tasks, SD 0.78–0.9, ICC 0.15), power is ≈0.4–0.55.
   - The headline sub-result (B-RECALL miss) is not a primary.
5. **Target fit and relevance.**
   - It is a localiser using a graph, not a graph construction or embedding technique.
   - C-USE (beats direct prompting) is ≈0.17–0.38 powered.
   - The agentic-graph baselines (CoSIL, GraphLocator) are only cited.
   - So the most likely positive paper is R5, "beats its own ablations, not Gemma prompted directly". Reviewers may read that as weak relevance unless the robustness strata (R2, gold-not-in-G0, B-RECALL miss) carry it.

No file other than this one was modified.
