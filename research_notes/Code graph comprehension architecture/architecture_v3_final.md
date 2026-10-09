# Architecture v3.1 (final): Behavioural Map-Matching (BMM), with Phase 4. 9 Oct 2026

## v3.1 changes (from v3, after `verify_v3.md`)

| Fix | What changed in v3.1 |
|---|---|
| **F1** | μ′ is now **inside β**, and applies only on steps that state an effect (E ≠ ∅). X-β0 zeroes every q-dependent term (§0.6.2 Eq. 5) |
| **F2** | Under R2, BMM maps an issue-named repo exception to "raises *any* repo exception (MRO-closed)" and uses **no identity map**. A builtin-only C-BEH estimate is reported alongside |
| **F3** | Deletes at the last step T credit their destination. A **silent ε-skip** over wraps/decorator edges is restored. P1a also substitutes the **last** step. Diagnostics restored: C-DEC by path length (L4), family-size stratum (L10), "excluding entry" (L11), self-loop cap (L13), full residual-name share (L18) |
| **F4** | X-BAG uses BMM's per-sample orientation. The bag diagnostic is **RBO (p = 0.9) over the top 100**, and it never overrides a Holm-significant C-DEC |
| **F5** | **The "local base rate" is removed from the novelty claims.** Centring is ranking-neutral at its own step (Proposition 0) and is kept only as a match-vs-insert calibrator. Omnivores are handled by the node-specific all-bit penalty μ′ |
| **F6** | The C-BEH stratum is restated at a 35% share (210 tasks). Power is given at SD 0.6 / 0.78 / 0.9. A three-way decision rule (supported / falsified-equivalent / inconclusive) replaces the garbled sentence |
| **F7** | Tuned arms are stated: BMM is tuned on a 3⁴ grid. Controls inherit its parameters except the one they remove. X-β0 also gets a sensitivity re-tune. Decoding CPU and G2 are recosted |
| **F8** | K4–K6 re-dated to Oct 21–23. Optimistic pd = 14.1 and floor = 14.75 pd. The tracer is on the never-cut list. Rung 8 states that C-BEH is demoted unless the share is ≥37.5%. With k = 5 the stratum rule becomes "≥2 of 5" |
| **F9** | Missing-call (predecessor) credit on "−" steps. Noisy-OR for dual-source bits. Receiver-typed AST rules for implicit raises. Only repo-resolved, callee-deduplicated sites are counted |
| **F10** | Memorisation = B-RECALL hit ∧ gold *not* named in the issue, with a fixed matching rule. The graph-free control uses π. Non-function gold exclusions are reported by polarity |
| **F11** | §0.5 headline: five edits. New rows: Liblit/IDF, the SherLog family, RoG/GCR, AstroBR, FaR-Loc. New ablation **X-POOL** (effects pooled across steps). B-ALIGN gets the same ℓ, including β |
| **New** | §0.7 **Topic fit**: decision (b), with reasons. §0.6 **Novelty verification (4 criteria)**: formal model, Propositions 0–4, and a status table. The CPU sibling-separation pilot **K3c** is the cheapest first demonstration |

---

**Inputs.**
- v2 and its breaks: `break_v2_logic.md` (N1–N30, plus N-orient, N-tune, N-ctrl, N-gold) and `break_v2_priorart.md` (PA2).
- `verify_v3.md` (F1–F11, E-1 to E-12).
- Earlier notes: v1, the v1 breaks (L-, M§, BR3), the phase-3 critic and red team, `docs/training_feasibility_4xl4.md`, HQ.

**Labels.**
- **VERIFIED**: read in repo docs, or by a breaker or verifier in a primary source.
- **INFERRED**: my reasoning.
- **GUESSED**: a number a kill test must measure.

**Nothing has been run.**

**Units.**
- pd = a 5-hour person-day (solo, shared with the main track).
- NH = Kaggle 4×L4 session-hour.
- kh = 4-vCPU kernel-hour.

**Dates.** Today is Thu Oct 9. Day-1 CPU kills land by Oct 14 and the K3c pilot by **Oct 17**. Graph and decoder kills land Oct 21–23. The freeze is **Nov 5**; the paper is due Nov 12.

---

## 0. Novelty

### 0.1 What is new, stated plainly

**The novelty is a combination at closeness ≈2.** The novel piece itself is closeness 2; the architecture as a whole is 2–2.5. Every sub-mechanism has a closeness-2 neighbour (§0.5). What no prior has, all at once (absence at abstract level, ≈55 queries in BR3 and BR3b):

- (i) issue-prose effects with **polarity**, attached to the steps of an **LLM-imagined** execution;
- (ii) scored against **predicted per-function effect probabilities** over a fixed alphabet. These come from AST rules, plus an execution-trained model *if K4 passes*;
- (iii) for "expected but missing" effects, a **noisy-OR negative finding plus a peer contrast** (Chucky-style) computed only for the effect the issue names;
- (iv) **compounded along a call-graph path** by exact HMM inference, with delete and skip states;
- (v) for **issue localisation, with no tests at inference**.

**Removed in v3.1 (F5).** The "local base rate" is not part of the novelty. Proposition 0 shows it does not change rankings at its own step.

### 0.2 The novel piece, one sentence

> **BMM turns the behaviour an issue reports (+) and the behaviour it expected but did not see (−) into signed observations on the steps of an LLM-imagined execution path. It scores each step against per-function effect probabilities (with a noisy-OR negative-finding term and a peer contrast for "−"), and compounds the evidence along the repository call graph by exact HMM decoding.**

**Claim sentence, used only if C-BEH passes** (PA2, edited per F5):
> *We turn the observed and expected-but-unobserved behaviour stated in an issue into signed observations of an LLM-imagined execution path and score them against per-function effect probabilities. Decoding them on the call graph improves function-level localisation over the same path decoded with names and body text alone, under code-side renaming.*

"Execution-trained" is added before "per-function" **only if K4 passes**. Do not say "first" for polarity, likelihood terms, peer contrast or decoding individually.

### 0.3 The single result that makes the paper worth reading at closeness 2

**C-BEH**, the primary claim: under R2, on the effect-stated stratum, the gold function's truncated log₂-rank under (BMM + BODY) beats the identical decoder with β = 0. Its pre-registered secondary is the **memorisation-free** sub-stratum: B-RECALL miss, or gold named in the issue (F10). That sub-stratum is reported as an estimate with a CI. It is not a separate primary (verifier §3.1.4).

Either sign is an insight into existing methods:
- **Positive:** issue-stated behaviour is a localisation channel that survives code-side renaming. LocAgent, Agentless and SweRank read issue behaviour only as text.
- **Equivalence-null** (§4.2 rule): behaviour stated in issues adds nothing beyond body tokens once names are renamed. That bounds behaviour-aware localisers (FaR-Loc, BLAgent) and the finding that expected behaviour helps agents (2607.07593) to a lexical mechanism.

### 0.4 Must NOT claim

Items 1–10 from v2:
1. LLM-imagined execution flow.
2. Symptom → path reconstruction.
3. NL steps mapped onto an execution graph.
4. HMM, map-matching or Bayesian fault localisation.
5. Exception- or effect-driven localisation.
6. Execution-trained exception prediction.
7. Repo-level predicted vs executed path measurement.
8. Search by behaviour.
9. "Execution-free". Say **"no tests at inference"**.
10. Execution-verified call-edge data.

Items 11–16 from PA2:

11. Markov/DP alignment of report steps onto a program graph, with missing-step inference (Zhang et al. FSE'24, AstroBR, Euler).
12. Using the observed/expected-behaviour structure of reports for localisation (Chaparro; BugScribe; Desc2Fix; 2607.07593).
13. Negative findings as an inference idea (QMR-DT; possibilistic MBD; software MBD).
14. Base-rate or likelihood-ratio scoring of code elements (Liblit PLDI'05; Sisman & Kak; IDF-IRBL).
15. Peer contrast: "should do X like its peers" (Chucky, Engler).
16. Pathidea is log- and stack-anchored BFS with a file-level IR boost, not NL. Do not claim "first to combine report-derived paths with IR".

**Related-work order:** SherLog → Pensieve → Pathidea → CrashLocator → COCA/2506.19045; Euler → FSE'24 → AstroBR; Newson–Krumm, RoG/GCR; QMR-DT and MBD; Liblit and IDF; Chucky and Engler; LocAgent; IPA-GNN and CodeFlow; RepoReasoner and SWE-Flux. Read the classic papers (QMR-DT, Liblit, Chucky, Engler) before writing.

### 0.5 Novelty as existing idea + new use

Shrey's framing: the thing must not yet have been *built*; the idea need not be unprecedented. Closeness values come from RT, CR, BR3 and BR3b.

#### 0.5.1 Existing ideas, what each already does, BMM's change, and the falsifying experiment

| # | Existing idea (closeness) | What it already does | BMM's change | Experiment that would show the change does not help |
|---|---|---|---|---|
| E1 | HMM map-matching, Newson & Krumm (2, RT); multiscale Viterbi on graphs, TSP'18 | Decodes noisy GPS points onto road segments | **Domain:** repository call graph. **Input:** LLM-imagined steps from prose. **States:** delete and silent-skip. **Target:** culprit posterior | **C-DEC** (BMM vs X-BAG), by path length and branching (Prop. 2); X-SHUF; X-DPR. K11; collapse rows R2/R4 |
| E2 | Zhang et al. FSE'24 report-step → UI-model Markov/DP alignment (2.5 for the decoder, BR3b); Euler greedy (2, BR3); AstroBR (2 overall, BR3b): an LLM **extracts** S2R steps and maps them onto a GUI execution model | Aligns human-written or LLM-*extracted* reproduction steps onto a GUI graph with heuristic additive scores, for **reproduction** | **Graph:** call graph. **Steps:** LLM-*imagined*, from issues with no S2R section. **Emissions:** normalised likelihoods with behavioural terms. **Target:** localisation | **BMM vs B-ALIGN** (FSE'24-style DP and greedy Euler, given the **same ℓ, including β**, F11). Secondary §4.4. If B-ALIGN ≥ BMM, exact inference adds nothing |
| E3 | Effect/exception-driven localisation: EXCEPT, SBEST (1.5, RT P5 / CR); CrashTracker (1.5, BR3). Behaviour-described retrieval: FaR-Loc (1.5, BR3) | Use the exception type, stack trace or exception summaries from an *executed failure*. FaR-Loc retrieves by a free-text description of failing behaviour, built from a failing test | **Input:** traceback-free prose effects with polarity, span-verified. **Signal:** a typed multi-effect alphabet, not free text or one exception. **Use:** per-step emissions compounded along a path | **C-BEH** ((BMM+BODY) vs X-β0, under R2; core primary, K11). **X-LEXEFF⁺** (typed vs grep). **X-POOL** (per-step attachment vs pooled). K3a/K3b/K3c gate the stratum and its signal |
| E4 | Execution-trained runtime-error prediction: IPA-GNN, CodeFlow (2, BR3) | Statically predict whether, and which kind of, error a whole program raises, with execution labels | **Target:** per-function, cross-repository probabilities over intended-effect bits, calibrated as emissions. **Split:** implicit raises come from receiver-typed AST rules; the two sources are combined by noisy-OR | **K4** (g_φ vs AST, leave-repo-out macro-AP). If it fails: AST q, and "execution-trained" is dropped. C-BEH-learned = BMM − X-ASTEFF (descriptive) |
| E5 | Negative findings: QMR-DT/Quickscore (2, BR3b); possibilistic MBD (2) | Absent expected findings lower the posterior of hypotheses that would cause them | **Source:** "expected but did not see" in issue prose, attached to an imagined step. **Object:** predicted function effects | **X-NEG0** under C-BEH. If NEG does not help, polarity becomes peer contrast (R7) |
| E6 | Peer contrast: Chucky (2, BR3b); Engler (1.5) | Flags functions that lack a check their similar neighbours have. It is issue-agnostic | **Conditioning:** only the effect the issue says is missing. **Peers:** HEAP override family, call-site co-candidates, sibling-class methods | **X-PEER0** on the "should X but doesn't" stratum, with peer coverage reported |
| E7 | LLM flow reconstruction: LocAgent prompt (1.5–2, BR3) | Free-text execution flow as chain of thought for an agent | **Form:** a typed (symbol, ±effect @span) sequence. **Consumer:** exact inference, not context | **C-USE** (BMM vs B-DIRECT\*; variant (ii) is the LocAgent prompt). Underpowered (§4.3) → R5 |
| E8 | Imagined-vs-executed comparison: RepoReasoner (2, BR3); SWE-Flux, TraceEval (1.5) | Test → predicted file set vs traced chain. Finding: precision high (≈0.80), recall low (≈0.31) | **Input:** issue prose. **Granularity:** function level. **Reference:** fail-to-pass trace. **Analysis:** divergences typed by edge kind, stratified by memorisation | **M1** on EVAL-exec. If it replicates 0.80/0.31 with no concentration by edge type, it is reported as a replication |
| E9 | Base-rate / IDF scoring: Liblit Increase (2, BR3b); IDF-IRBL (1.5) | Normalise evidence for a code element by a background rate | **No new use is claimed (F5).** The per-step centring constant is a match/insert calibrator. The node-specific omnivore control is the μ′ penalty | Proposition 0 (algebraic). μ′ = 0 is a tuning-grid point; the LOC-Spearman rule (§0.6.2) |
| E10 | Symptom → path reconstruction: SherLog, Pensieve (1.5–2), Pathidea (2), CrashLocator (1.5–2), COCA, 2506.19045 (1.5) | Rebuild executed paths from real **logs, stack traces or crash stacks** with exact anchors, by rules or BFS | **Observations** are imagined from prose with no runtime artefact. **Anchors** are soft (names + effect likelihoods). **Inference** is exact probabilistic, not rules | **B-ALIGN** (anchor-and-fill heuristic) and **X-NULL** (paths for another issue). On the optional traceback stratum, **B-CRASH** (CrashLocator-style) |
| E11 | Plan-then-ground on graphs: RoG (ICLR'24), GCR (ICML'25) (2, RT) | An LLM plans relation paths, which are grounded on a knowledge graph (hard constraint in GCR) | **Graph:** a program graph with typed call edges. **Grounding:** soft probabilistic, with delete/skip. **Plan elements:** behavioural, not relation labels | C-DEC (soft grounding vs bag). A hard-constraint GCR-style variant (ρ = 0, no insert/delete) is reported as a secondary decode |

**Infrastructure, not claimed:** HEAP (2–2.5: Jelly, HeapDL, PoTo) and per-site candidate sets (2: AFLGopher, Pereira).

#### 0.5.2 What the combination does that no part can do alone

- **E1/E2/E10/E11 alone** need name, text or artefact anchors. They fail when names are hallucinated or renamed.
- **E3–E6 alone** score one function against one piece of evidence. A ~15-bit signature leaves huge ties, and a missing effect cannot be placed without knowing which step it belongs to.
- **E7/E8** produce or measure imagined flows but never correct them.
- **Together:**
  - the sequence attaches each stated effect to a step, which X-POOL tests;
  - the graph orders and connects candidates and lets hallucinated steps be deleted or skipped, which C-DEC and Prop. 2/4 test;
  - the signed effects separate name-similar siblings, which C-BEH, Prop. 1 and K3c test.

Collapse rows R1, R2, R3 and R7 decide which parts carry the gain.

#### 0.5.3 Honest headline sentence (the five F11 edits applied)

> **We take map-matching (HMM map-matching from GPS; Markov/DP report-step alignment from FSE'24) and adapt it to issue localisation on repository call graphs. The observations are an LLM's prose-imagined execution path, whose steps carry issue-stated signed effects. These are scored against per-function effect probabilities (AST rules, plus an execution-trained model if K4 passes), using a noisy-OR negative finding and peer contrast (Chucky) applied to issue-stated missing effects. This yields a behavioural channel robust to code-side renaming, and we test whether it lowers gold-function rank beyond the same decoder with names and body text alone (C-BEH).**

**Fallback (R3/R4):** "Map-matching LLM-imagined paths onto call graphs works (or does not) as well as names and body text; issue-stated behaviour adds no localisation signal once code names are renamed."

### 0.6 Novelty verification (4 criteria)

#### 0.6.1 Criterion 1: Combination

**The exact set combined:**

| Idea | Prior-art evidence |
|---|---|
| (C1) HMM map-matching | Newson–Krumm; RT P3; RT-28/32/38; CR-2/9 |
| (C2) Markov/DP report-step alignment | FSE'24; BR3b-10/12 |
| (C3) LLM-imagined flow | LocAgent; BR3-23 |
| (C4) Typed effects from issue prose, with polarity | Desc2Fix, BugScribe, Chaparro: closeness 1; BR3-27, BR3b-3/7 |
| (C5) Predicted per-function effect probabilities | IPA-GNN, CodeFlow; BR3-9/26 |
| (C6) Noisy-OR negative findings | QMR-DT; BR3b-13 |
| (C7) Peer contrast | Chucky; BR3b-18 |

**Evidence that nobody combines them:**
- No Viterbi or map-matching on call graphs: BR3-1 (0 hits), RT-28, BR3-21/29.
- No predict-then-align from issues: BR3-7, BR3-12, BR3-25, RT-27.
- No signed-effect localiser: BR3b-8 (4 queries, 0 relevant), BR3b-15, BR3b-17, BR3b-11 (OpenReview, no hit).
- No effect-signature retrieval: BR3-8/20, BR3b-4.
- The closest pairs are at 2–2.5: FSE'24 = C1 + C2 on GUI graphs; LocAgent = C3 without decoding; Chucky = C7 without issues.

**Residual risk:**
- Absence is at abstract level only. Semantic Scholar was rate-limited, and Google Scholar and ACM DL were not crawled.
- ICSE/FSE 2027 papers not yet on arXiv are not visible.
- QMR-DT, Liblit, Chucky and Engler were read only as snippets.
- An unpublished or industrial system combining C1 + C4 cannot be excluded.

**Status: verified on paper (abstract-level absence).**

#### 0.6.2 Criterion 2: Formulation

**Graph.**
- V_r is the set of repo functions: non-test (by discovery config), non-vendored.
- For u ∈ V_r, S(u) is the set of **repo-resolved, callee-deduplicated** call sites (F9). Each site s has candidates C(s) ⊆ V_r, with p(v | s) uniform over C(s), or split by HEAP dispatch.
- Callback states ∅_s (site-local) and ∅_ρ (one per framework registry) are added.
- W ⊆ V×V is the set of wraps/decorator edges (HEAP or AST).

**Observations.** For one sampled path in one orientation:

```
o_{1:T},  o_t = (n_t, E⁺_t, E⁻_t),  E^± ⊆ B
```

- n_t is a symbol string.
- B is the effect alphabet: per-class exceptions (MRO-closed; repo classes under R2 collapse to "any repo exception", F2), returns None, returns self, mutates an argument, writes a self attribute, warns.
- Every code must cite an issue span that passes a substring check.

**Hidden state.**

```
z_t ∈ { M_v, I_v, D_v : v ∈ V_r ∪ ∅ }
```

- M_v: match. o_t is emitted by v.
- I_v: insert. o_t is noise; the position stays at v.
- D_v: delete. o_t is noise; the position has advanced to v.
- Let pos(z_t) = v.

**Transition kernel** (fixed scalars η₀, δ, ρ, ε ∈ (0,1); η₀ + δ + ρ < 1):

```
K(u,v)  = (1−ε)·(1/|S(u)|) Σ_{s∈S(u)} p(v|s)  +  ε·Σ_{w:(u,w)∈W} K⁰(w,v)/|W(u)|        (1)
          [ε-skip: a silent hop over a wrapper; K⁰ is the ε=0 kernel. Self-loop cap (L13): total stay probability at u is max(η₀, K(u,u)), not η₀ + K(u,u)]
R_t(v)  = 0.8·Unif( N₂(Top20_t) )(v) + 0.2·π(v)                                           (2)
          [Top20_t = top 20 nodes by ℓ·(o_t); N₂ = 2-hop graph neighbourhood. Depends only on o, so the HMM stays exact]
P(z_t = M_v or D_v | pos(z_{t−1}) = u) = (1−η₀−δ−ρ)·K(u,v)·[1−θ] + ρ·R_t(v)·[1−θ]   for M,
                                       = (1−η₀−δ−ρ)·K(u,v)·θ    + ρ·R_t(v)·θ         for D,   θ = δ/(1−η₀−ρ)
P(z_t = I_u | pos(z_{t−1}) = u)        = η₀                                                (3)
π(v) = softmax( BM25(issue, v) restricted to G0 files ) ⊕ CRE(issue, v)   (mixture, fixed 0.5/0.5)
```

**Emission** (log-ratio to the background b₀; I and D emit b₀, i.e. log-ratio 0):

```
ℓ_v(o_t) = τ·[ sim(n_t, v) − c_t^sim ]  +  ω·BODY(o_t, v)  +  β·[ Φ_v(o_t) − c_t^Φ ]        (4)

Φ_v(o) = Σ_{j∈E⁺} log((1−η) q_vj + η)                                   (reported effect)
       + Σ_{j∈E⁻} [ log(1 − q_vj + η)                                    (NEG: noisy-OR negative finding)
                   + log( (p̄_j(v) + η) / (q_vj + η) ) ]                  (PEER: Chucky-style contrast; 0 if no peers)
       − μ′·1[E⁺∪E⁻ ≠ ∅]·Σ_{j∉E⁺∪E⁻} q_vj                                (omnivore penalty, inside β; F1)       (5)
```

- c_t^sim and c_t^Φ are the medians over the issue's G0-file nodes. They are constants per step.
- BODY is BM25 of (n_t + the cited issue sentence) against v's body tokens, with weight ω tuned once and shared.
- p̄_j(v) is the mean q over v's peers.
- **X-β0** sets β = 0, which removes every q-dependent term.

**Effect probabilities.**

```
q_vj = 1 − (1 − q^AST_vj)(1 − q^φ_vj)          for dual-source bits (KeyError, IndexError, TypeError, AttributeError; F9)
q^AST: receiver-typed rules (subscript on a dict-typed / parameter receiver without .get / in-guard → KeyError; attribute on Optional/None-able
       parameter → AttributeError; …), q^AST ∈ {0.05, 0.6, 0.95} by rule strength
q^φ_vj = σ(φ_jᵀ x_v)  for intended-effect bits (explicit raise C, NotImplementedError, warns, returns None/self, mutates arg, writes self)
exception bits propagated ≤2 hops:  q̃_vj = 1 − (1−q_vj) Π_{u: K(v,u)>0, d(v,u)≤2} (1 − K(v,u) q̃_uj (1 − c_vj))       (6)
       [c_vj = 1 if v has an except clause covering j by issubclass]
```

**Training objective for g_φ** (CAL-train repos only, leave-one-repo-out, then isotonic calibration on the held-out repo):

```
min_φ  Σ_r Σ_{v∈V_r} w_v Σ_{j∈B_int} −[ y_vj log σ(φ_jᵀx_v) + (1−y_vj) log(1−σ(φ_jᵀx_v)) ]  +  λ‖φ‖²₂            (7)
y_vj = 1[ effect j observed for v in ≥ 2 distinct passing tests ]     (possibility label)
w_v  = 1 / max(1, log(1 + calls_v))                                  (inverse call-count propensity)
```

**Inference and readout.**
- Forward–backward over z_{1:T} gives γ_t(·).
- The orientation (o or its reverse) with the higher P(o) is kept per sample.

```
score(v) = (1/k) Σ_samples [ Σ_t w_t γ_t(M_v)  +  w_T γ_T(D_v)                       (F3: delete credit at T)
                             + Σ_{t: E⁻_t≠∅} w_t·κ_pred·Σ_u γ_t(M_u)·K(v,u)/Σ_{v'}K(v',u) ]    (missing-call credit to callers; F9)
w_t ∝ (t/T)²,  κ_pred = 0.3 (fixed);  rank = min over gold functions;  metric m = log₂ min(rank, 101)
```

**Parameters.**
- **Tuned (BMM only):** β, τ, η₀, μ′ on a 3⁴ grid with nested 5-fold CV on CAL-tune (F7).
- **Fixed:** δ = 0.05, ρ = 0.1, ε = 0.05, η = 0.05, κ_pred = 0.3, ω (tuned once and shared).
- **Controls** inherit BMM's values except the parameter they remove.
- **Sensitivity:** X-β0 is also re-tuned on (τ, η₀) over a 3² grid.

**Pre-registered LOC rule.** If Spearman(Φ, LOC) > 0.3, C-BEH is read on the LOC-quartile-matched stratum.

**X-BAG** is the same model with K(u,·) replaced by the rank-1 kernel PPR_π(·) (ρ = 1 in effect), the same I and D states, the same readout, and **BMM's chosen orientation** (F4).

**Status: verified on paper** (this section is the definition).

#### 0.6.3 Criterion 3: Mathematical mechanism

**Proposition 0 (centring is a calibrator, not a contrast).**
- Statement: adding any constant c_t to ℓ_v(o_t) for all v leaves γ_t(M_v)/γ_t(M_{v'}) unchanged for every v, v′. It changes scores only by reweighting match versus insert/delete at step t, which affects other steps.
- Proof: every path with z_t ∈ {M_·} is multiplied by e^{c_t}, and γ_t(M_v) sums only over such paths. ∎
- Consequence: the v3 "local base rate" (and v2's q̄) cannot separate two candidates at the step it is applied to. Hence F5.
- Check: none needed (algebraic). μ′ ∈ grid (including 0) carries the omnivore control.

**Proposition 1 (when signed evidence separates the culprit from a name-similar sibling).**
- Setup: let v\* (the culprit) and v′ be **structurally exchangeable**: K(u,v\*) = K(u,v′) and K(v\*,w) = K(v′,w) for all u, w; R_t(v\*) = R_t(v′); π(v\*) = π(v′). Let them be **name-tied** at every step: sim and BODY are equal.
- (a) With β = 0, γ_t(M_{v\*}) = γ_t(M_{v′}) for all t, and score(v\*) = score(v′). Every names, body, graph or PPR method ties them. The same holds for X-β0, BM25, CRE and PPR.
- (b) With β > 0, if the only observation that differs between them is o_t:

```
log γ_t(M_{v*}) − log γ_t(M_{v'}) = β·[Φ_{v*}(o_t) − Φ_{v'}(o_t)]
                                  = β·[ Δ⁺ + Δ⁻_NEG + Δ⁻_PEER − μ′·Σ_{j∉E}(q_{v*j} − q_{v'j}) ]
```

  so v\* ranks above v′ **iff**

```
Δ⁺ + Δ⁻_NEG + Δ⁻_PEER  >  μ′ · Σ_{j∉E}(q_{v*j} − q_{v'j})                              (8)
```

  where Δ⁺ = Σ_{j∈E⁺} log[((1−η)q_{v\*j}+η)/((1−η)q_{v′j}+η)], Δ⁻_NEG = Σ_{j∈E⁻} log[(1−q_{v\*j}+η)/(1−q_{v′j}+η)], and Δ⁻_PEER = Σ_{j∈E⁻} log[(q_{v′j}+η)/(q_{v\*j}+η)] when v\* and v′ share peers.
- Proof sketch: by exchangeability, the forward message into v\* equals that into v′, and so does the backward message out of them. They differ only by the emission factor at t, so the posterior ratio equals the emission ratio. The readout is linear in γ. ∎
- Corollaries:
  - (i) For "−" evidence with shared peers, NEG and PEER **both** favour the function with the lower q_j. That is right for "v should raise but doesn't". For the missing-call shape the culprit is a caller, so the credit comes from the predecessor term, not from (8).
  - (ii) Omnivores win only when (8) fails, i.e. when the culprit's unstated-effect mass exceeds its stated-effect advantage divided by μ′.
- **Check: K3c** (CPU, by Oct 17). On effect-stated EVAL issues, using regex-parsed effects and AST q only, compute for each gold function and each of its top-5 name-similar non-gold siblings (same family or file, by sim) whether (8) holds at the tuned-free point β = 1, μ′ = 0.1. The **sibling-separation rate** is the share of pairs that satisfy it. Prediction ≥0.6. At ≈0.5 (chance) the behavioural channel is empty, and the result is R3/R6.

**Proposition 2 (when sequence decoding beats the bag).**
- Setup: the true path is v₁ → … → v_T. Step t's name evidence gives v_t a log-advantage a_t over each of m distractors d, which are not children of v_{t−1}. Suppose step t−1 is decoded with posterior mass ≥ 1 − ϵ on v_{t−1}, and let b_t = |S(v_{t−1})| be its effective branching.
- Prior odds of v_t against d:
  - BMM: O_HMM ≈ [(1−η₀−δ−ρ)/b_t + ρR_t(v_t)] / [ρR_t(d)]
  - X-BAG: O_BAG = PPR(v_t)/PPR(d)
- **BMM ranks v_t above d where the bag does not iff O_HMM > O_BAG**, i.e. roughly

```
b_t  <  (1−η₀−δ−ρ) · PPR(d) / ( ρ·R_t(d)·PPR(v_t) )                                     (9)
```

- Consequences:
  - The advantage needs (a) **T ≥ 2 with a confidently decoded predecessor** (an issue-named entry, or an exact name hit), (b) **low branching** b_t, and (c) distractors that are **globally plausible** (high PPR) but **locally implausible** (outside N₂).
  - It accumulates over consecutive steps while each predecessor stays confident. Each confident step multiplies the odds by the (9) ratio. When an intermediate is hallucinated, the delete state D carries the position at cost δ instead of collapsing to the entry (the N8 regime).
- Proof sketch: condition on z_{t−1} at v_{t−1} up to ϵ. The prior for z_t then follows directly from (3) against the rank-1 kernel. Both arms share emissions, so the posterior odds equal the prior odds × e^{a_t}. ∎
- **Check:** C-DEC reported by **T** (1–2 / 3–4 / ≥5) and by **median b_t** along the decoded path (terciles); L4 is restored. Prediction: Δm ≈ 0 for T ≤ 2 and grows with T and with low b. P1a (oracle paths) tests (9) without LLM noise.

**Proposition 3 (exact reductions to baselines).**
- (a) ρ = 1 with R_t := PPR_π gives **BMM ≡ X-BAG**.
- (b) T = 1 gives a posterior ∝ [(1−η₀−δ−ρ)K(·) + ρR₁]·e^{ℓ}, a prior-weighted bag. C-DEC's mechanism is absent.
- (c) β = 0 gives **BMM + BODY ≡ X-β0**, which is name-and-text map-matching: the FSE'24 PRM transplanted to call graphs.
- (d) If all ℓ_v(o_t) are equal (uninformative observations), γ_t(M_·) ∝ (π-started propagation through (3)). The score is then a truncated random walk with restart, PPR-like, i.e. B-PPR up to the aggregator.
- (e) η₀ + δ → 1 means no match states, so score = 0 everywhere and ties are broken by BM25. That is B-BM25.
- Proof: substitute into (1)–(5). ∎
- **Check:** the tuned (ρ, η₀) values are reported. An RBO ≥0.9 with X-BAG, or ≥0.9 with B-PPR, marks a run as "reduced". This is descriptive and never overrides a significant primary.

**Proposition 4 (culprit-end correction by delete credit).**
- Setup: the last step o_T is hallucinated (ℓ_v(o_T) ≈ 0 for all v), and step T−1 is confident at u.
- Then the readout mass at T is spread over K(u,·) through D and the weak M states, giving each child ≈ w_T/b_u. X-BAG spreads w_T by PPR over all of V.
- So BMM places a culprit that is a child of u with probability ≈ 1/b_u, against PPR(v\*) under the bag.
- Proof: a corollary of (3) and the F3 delete credit. ∎
- **Check:** P1a with **last-step substitution** (F3) on executed CAL-tune tasks.

**Status: proofs and sketches on paper. Every condition is tied to a named check (K3c, C-DEC by T and b, P1a, RBO diagnostics). None has been run.**

#### 0.6.4 Criterion 4: Demonstrated advantage

**NOT established. Nothing has run.** The following would establish it:
- **C-BEH** (K11): support or equivalence-null.
- **C-DEC** (K11).
- Their mechanism checks: **K3c** (Prop. 1), C-DEC by T and b (Prop. 2), P1a last-step (Prop. 4).

**Cheapest first demonstrations, in order:**

| # | Pilot | Cost | Date | What it shows |
|---|---|---|---|---|
| 1 | **K3c sibling-separation pilot** (CPU only; no LLM, no traces). EVAL builder + AST effect rules + regex issue effects on the effect-stated stratum | ≈2 kh, ≈0.25 pd beyond the K0/K3a work | **Oct 17** | Whether signed evidence separates gold from its name-similar siblings at all (Prop. 1 condition). A rate ≤0.55 demotes C-BEH before any GPU |
| 2 | **Oracle mechanism pilot** (CPU). ~40 executed CAL-tune tasks; traced F2P paths with sibling-substituted middle and last names; traced effects. Three contrasts: BMM vs X-BAG (Props 2, 4), BMM vs X-β0 with names renamed (Prop. 1 with oracle effects), P1a gate | ≈5 kh | **Oct 22–23** (needs tracer + decoder) | The mechanism with no LLM noise. Small n, descriptive |
| 3 | **≤1 NH real-LLM pilot.** K9 (CAL-tune G0/G1/B-DIRECT\*, 0.6 NH) + K3b/K10 (50-task EVAL pilot, 0.4 NH) | ≈1.0 NH | **Oct 24–25** | First real-generator signal and SDs. Not a claim |

#### 0.6.5 Status table

| Criterion | Status |
|---|---|
| 1. Combination: no prior combines C1–C7 | **Verified on paper** (abstract-level absence; residual risk stated) |
| 2. Formulation | **Verified on paper** (§0.6.2) |
| 3. Mathematical mechanism | **Verified on paper** as Props 0–4 with proofs/sketches; **conditions pending experiment** (K3c, P1a, C-DEC by T/b) |
| 4. Demonstrated advantage | **Not met.** Pending K3c (Oct 17), the oracle pilot (Oct 22–23), the LLM pilot (Oct 24–25) and K11 (Nov 3) |

### 0.7 Topic fit: decision (b)

**Decision: (b).** Keep the localisation framing, and submit it as **inference over code graphs** for comprehension: issue → grounded execution path. That fits Code Comprehension in a generous reading ("parsing" an issue into a graph-grounded path), and Graph Reasoning in a strict one.

**Why not (a).** Reframing as a "behavioural code-graph representation" would be substantively true only if q_v were a contribution in its own right. It is not:
- It is ~15 bits.
- If K4 fails, it falls back to AST rules that are not novel.
- The execution-trained part is closeness 2 to IPA-GNN/CodeFlow.
- Retrieval with q_v alone is predictably weak (huge ties), so a "q_v vs text embeddings" retrieval test would be a strawman.

**What remains (a)-like, and is reported as such.**
- q_v is evaluated **as a representation** at K4: leave-repo-out macro-AP per bit against AST rules.
- The typed effect-annotated call graph (q_v + typed edges + per-site candidates) is released as a dataset artifact.

**Pre-registered upgrade.** If K4 passes **and** C-BEH-learned is positive, the paper may add "execution-calibrated behavioural node annotations" as a secondary artifact in the Code Comprehension framing. It is never the headline.

---

## 1. Architecture (lean scope)

```
OFFLINE (CPU; CAL-train, Py≥3.12, sys.monitoring) — K4 (Oct 23); fail ⇒ AST q only
  traced suites ─► possibility labels, intended-effect bits ─► g_φ (Eq. 7)

PER SNAPSHOT (CPU; no tests)
  AST resolver (imports, aliases, static MRO, self/cls/super, local imports, literal registries)
  ∪ HEAP where an image / pure-Python install imports (unwrap → W edges, registries, re-exports, MRO, framework registries)
  sites S(u): repo-resolved, callee-deduped; ∅_s site-local; ∅_ρ per framework registry
  q: receiver-typed AST rules ⊕(noisy-OR) g_φ; per-class exception bits (R2: "any repo exception"); propagation Eq. 6

PER ISSUE (GPU, vLLM, Gemma-4-31B W4A16, thinking off)
  G0 tree → top-10 files;  G1 skeleton(G0) → k=10 paths with ±effects @span;  span check (CPU)
  B-RECALL issue + repo name → top-10 (temperature 0)

DECODER (CPU): Eqs. 1–5, forward–backward, orientation max, readout with delete + missing-call credit
```

The formal definition is §0.6.2.

- **Non-function gold.** Tasks whose gold is a class body or module level are excluded from the headline and counted by polarity (F10).
- **The "−" stratum.** Missing-call credit applies on every step with E⁻ ≠ ∅ (F9). For "+exception" steps, the choice between the culprit-weighted readout and that plus predecessor credit is made on CAL-tune.

---

## 2. Data

| Pool | Source | Size | Execution | Role |
|---|---|---|---|---|
| **EVAL** | SWE-bench Verified ∪ Lite (≈226 after filters) + SWE-rebench (≥375, ≥14 repos); Loc-Bench only if K0 is short. Dedup by instance_id, (repo, PR) and MinHash ≥0.8. Excludes dev, CAL-train and CAL-tune repos | ≥600 traceback-free, non-new-code, function-gold; ≤40 per repo; ≥25 repos | None (static gold) | Headline; by source |
| EVAL-pilot | 50 burned EVAL tasks | 50 | None | SDs; K3b |
| **EVAL-exec** | 60 tasks; SWE-bench images, extracting only `/opt/miniconda3` and `/testbed` | 60 | F2P ×3, settrace, differential | M1, P1c, edge recall; read once after the freeze |
| CAL-train | ~8 Python ≥3.12 repos (2 large) | — | Suites | g_φ if K4 |
| **CAL-tune** | ~100 rebench/SWE-Gym tasks, disjoint repos; ~40 executed | 100 | 40 | Tuning; P1a gate; oracle pilot; hallucination profile |

**Rules.**
- Rank = min over gold functions. Decorator and def-line hunks map to that def; deletion-only hunks to the enclosing def.
- Test files are defined by discovery config.
- **R2 (code-only):** identifiers in docstrings are renamed too. Repo exception classes are renamed with **no identity map for BMM** (F2). Paths are not renamed.
- **Residual-name share (L18):** paths, dunders, external-API overrides, string-built dispatch and literals, reported per repo. C4 is also reported on low-residual repos.

**Strata.**
- effect-stated: span-verified code in ≥3 of 10 samples, or ≥2 of 5 if k = 5 (F8);
- exception-line: in C-BEH only;
- **memorisation**: B-RECALL hit ∧ gold not named in the issue. Matching rule: an identifier-boundary match of the bare or qualified gold name (F10);
- gold-in-G0;
- builtin vs repo-exception steps;
- raiser vs caller;
- peer exists;
- polarity;
- path length T and branching b (Prop. 2);
- family size (L10);
- source.

**Docker Hub.** Shrey must enter a read-only token into Kaggle Secrets himself. Without it, EVAL-exec is ≈30 rebench pip tasks and HEAP covers pure-Python repos only. The HEAP sandbox is a best-effort in-process socket monkeypatch.

---

## 3. Arms

### 3.1 Baselines

| ID | Definition |
|---|---|
| **B-DIRECT\*** | Best of 3 prompt-only rankers chosen on CAL-tune: (i) direct top-10; (ii) LocAgent-style flow-then-list; (iii) G1's final steps. Same G0 and skeleton. Tail filled by X-BAG. This is also the prompt-only best-prompt baseline. Run on original + R2 |
| B-RERANK | Gemma listwise rerank of the RRF(BM25, CRE) top 50. **Ladder rung 1.** The SweRank-LLM numbers are cited if it is dropped |
| B-BM25, B-CRE | Function-level BM25; CodeRankEmbed (CPU). SweRankEmbed-Small is cited, not run (pd) |
| B-PPR | PPR from π on G, with no imagined path |
| **B-ALIGN** | FSE'24-style DP top-k (step, node) alignment over G-reachability, score = ℓ (**including β**) + 1/shortest-path distance, with missing-step fill. Greedy Euler variant reported too |
| B-RECALL | Issue + repo name → top 10; memorisation covariate |
| Published | LocAgent, CoSIL, SweRank, Agentless, OrcaLoca function Acc@k on the matched Lite subset, with non-comparability stated. A CoSIL run (100 tasks, +2 NH) is optional, only if ≤8 NH have been spent after G2 |
| CERT / Z | **Excluded.** They are RL agent-trajectory credit signals with no function ranking, and nothing is trained here |

### 3.2 Controls and ablations (one change each)

| ID | Change | Role |
|---|---|---|
| **X-BAG** | Rank-1 PPR kernel; BMM's orientation; RBO diagnostic | **C-DEC control**; BER fallback; B-DIRECT\* tail |
| **X-β0** | (BMM + BODY) with β = 0 (all q terms off) | **C-BEH control** |
| **Graph-free** | X-BAG with π in place of PPR_π (removes the graph only; F10) | Graph-free control |
| **X-DPR** | Degree-preserving rewire within the package, with p carried. A C-DEC replicate | Degree-preserving control |
| X-NEG0 / X-PEER0 | NEG off / PEER off | Polarity (R7) |
| X-POOL | ∪_t E^±_t attached to every step (F11) | Per-step attachment |
| X-ASTEFF / X-LEXEFF⁺ | AST-only q / an exception-token grep that is MRO-closed with propagation. Restricted to exception steps; warns, return None/self and self-writes have lexical analogues, but AST rules already cover them | C-BEH-learned / C-BEH-static |
| X-SHUF, X-NULL | Shuffled steps (T ≥ 3); donor paths (disjoint gold files, ±1 minor version) | Order; matched null |
| GCR-hard | ρ = 0, no I/D (hard grounding) | E11 secondary |
| BER | X-BAG + BODY vs the same with β = 0 | Fallback's own C-BEH |

---

## 4. Phase 4: claims and power

### 4.1 Metric and inference

- **Primary metric:** m = log₂ min(rank, 101), unfused.
- **Descriptive:** MRR, hit@1/5/10 and all-gold recall@10.
- **Inference:** wild cluster bootstrap over repos; Holm over 3 primaries; one control per claim; TOST margin ±0.1.
- **C-USE** uses a two-level bootstrap.

### 4.2 Falsifiable central claim and decision rule (F6)

**C-BEH (core):** under R2, on the effect-stated stratum (+ exception-line), Δ = m(BMM + BODY) − m(X-β0) < 0. Predicted Δ = −0.20 (GUESSED).

| Verdict | Rule |
|---|---|
| **Supported** | The Holm-adjusted two-sided CI for Δ lies entirely below 0 |
| **Falsified (equivalence)** | The 90% CI lies entirely inside (−0.1, +0.1) (TOST), or the CI lies entirely above 0 (harm) |
| **Inconclusive** | Anything else. Reported as such; never called falsified |

### 4.3 Primaries and honest power

Stratum at a 35% share = **210 tasks** (K3 pass bar), ~20 repos.

| Primary | Contrast | n → n_eff (ICC 0.05 / 0.15) | Predicted Δm | Power, SD 0.6 | SD 0.78 | SD 0.9 |
|---|---|---|---|---|---|---|
| **C-BEH** | (BMM + BODY) − X-β0, R2 | 210 → 142 / 87 | −0.20 | 0.94 / 0.76 | 0.75 / 0.50 | 0.60 / 0.37 |
| **C-DEC** | BMM − X-BAG | 600 → 279 / 135 | −0.20 | (SD 0.7) 0.99 / 0.82 | — | — |
| C-USE | BMM − B-DIRECT\* | 600 → 279 / 135 | −0.15 | (SD 1.2) 0.38 / 0.17 | — | — |

**Joint power.**
- C-BEH ∧ C-DEC ≈ 0.93 → 0.62 at SD 0.6.
- At SD 0.78 it falls to ≈0.74 → 0.41.
- **At realistic values (210 tasks, SD 0.78–0.9, ICC 0.15), C-BEH power is ≈0.37–0.50** (VERIFIED arithmetic, verify_v3 §3.1). This is stated in the paper.

**R5** ("beats its ablations, not direct prompting") is the most likely outcome **only if the predicted effects are real**: P ≈ 0.54–0.59 conditional on that. Unconditionally, R3, R4 or R6 may be more likely.

**Secondary estimates:** X-NEG0, X-PEER0, X-POOL, C-BEH-static and C-BEH-learned, BMM − B-ALIGN, GCR-hard, X-SHUF, X-DPR, X-NULL, graph-free, C4, by source, by memorisation, builtin-only C-BEH (F2), C-DEC by T/b (Prop. 2), Qwen sign agreement.

**Positive controls.**
- **P1a, the gate, on executed CAL-tune:** oracle paths with middle **and last** steps sibling-substituted from the measured profile, hit@5 ≥0.80.
- P1b: edge recall (reported).
- P1c: realistic oracle on EVAL-exec after the freeze.
- P2: traced effects.
- **M1:** RepoReasoner-style precision/recall on EVAL-exec, with divergences typed by edge kind.

---

## 5. Kill table

| # | By | Test | Metric | Threshold | Cost | Pass | Fail | Ambiguous |
|---|---|---|---|---|---|---|---|---|
| K0 | Oct 11 | EVAL pool, static gold, dedup; non-function exclusions by polarity | Count | ≥600 tasks, ≥25 repos | 4 kh, 1 pd | Proceed | <400: n = available, MDE restated | 400–600: add Loc-Bench, cap 50 |
| K1 | Oct 13 | Image prefix extraction ×3 (**Shrey's token**) | F2P fail→pass | ≥2/3 | 2 kh | EVAL-exec 60 | ≈30 pip tasks | 1/3: 0.5 pd debug |
| K2 | Oct 11 | CAL-tune availability | Tasks / executable | 100 / 40 | 0.3 pd | Proceed | 60 / 25 | — |
| K3a | Oct 14 | Regex effect share (validated against 2607.09553 labels if available) | Share | ≥35% | ≈0 | Proceed | <20%: C-BEH secondary before GPU → R6 | 20–35%: wait for K3b |
| **K3c** | **Oct 17** | Sibling-separation pilot (Prop. 1), AST q + regex effects | Share of (gold, top-5 sibling) pairs satisfying (8) | ≥0.60 | 2 kh, 0.25 pd | Proceed | ≤0.55: C-BEH demoted to secondary before GPU; headline R3/R6 | 0.55–0.60: proceed; C-BEH flagged at risk |
| K5 | Oct 22 | Graph adequacy, executed CAL-tune (P1b) | True-path edge recall | ≥0.6 | 3 kh | Proceed | <0.4: C-DEC expected weak; stated | 0.4–0.6: report |
| K6 | Oct 22 | **P1a gate + oracle mechanism pilot** (Props 2, 4) | hit@5; Δm BMM vs X-BAG on oracle paths | ≥0.80; Δm < 0 | 5 kh | Proceed to GPU | Debug the decoder on CAL-tune only | — |
| K7 | Oct 22 | DPR feasibility | Repos with ≤10% edge overlap | ≥10 | CPU | X-DPR | Descriptive | — |
| K4 | Oct 23 | g_φ vs AST rules, leave-repo-out | Macro-AP Δ | ≥ +0.03, CI > 0 | 18 kh, 1.5 pd | g_φ for intended bits | AST q; drop "execution-trained". Harmless | AST q |
| K8 | Oct 20 | Batched vLLM throughput (standalone first session) | Decode tok/s | ≥250 | 0.4 NH | k = 10 | k = 5 (stratum rule ≥2 of 5) | — |
| K9 | Oct 24 | CAL-tune GPU run (G0, G1, B-DIRECT\* ×3, B-RECALL) + measured hallucination profile → P1a re-gate | Nested-CV Δm vs X-BAG and vs B-DIRECT\* | BMM not worse than both by >0.3 | 0.6 NH | Proceed | EVAL runs; headline pre-switched to R3/R4 | Proceed |
| K3b + K10 | Oct 25 | 50-task EVAL pilot | Span-verified share; invented-code rate; SDs | Share ≥35% (≥150 tasks projected); SD ≤ 0.78 | 0.4 NH | Keep §4 | <150 tasks: C-BEH secondary (R6), fixed before EVAL decoding | SD 0.78–0.9: C-BEH stays primary, low power stated |
| K11 | Nov 3 | EVAL primaries | §4.2 | §4.2 | Core | R1/R5 | R2–R8 | Inconclusive is reported |

---

## 6. Collapse table

| Row | Outcome | What BMM is | Framing |
|---|---|---|---|
| R1 | C-BEH ✓, C-DEC ✓, C-USE ✓ | A technique at closeness 2 | Technique + M1 |
| R2 | C-BEH ✓, C-DEC ✗ | Behaviour helps; sequence does not | **BER**, with its own C-BEH |
| R3 | C-DEC ✓, C-BEH falsified (equivalence) | FSE'24-style map-matching on LLM names | Insight paper (§0.3 negative) + M1 |
| R4 | C-DEC ✗, C-BEH ✗, ≈ B-DIRECT\* | **Collapses into "BM25 + LLM rerank"** | Measurement paper + negative result |
| R5 | C-BEH ✓, C-DEC ✓, C-USE ✗ | Beats its ablations, not direct prompting (most likely *if* the effects are real) | Robustness framing: R2, memorisation-free, gold-not-in-G0 |
| R6 | K3a/K3b/K3c fail or are ambiguous | C-BEH underpowered or empty | Decided before EVAL decoding: C-BEH is an estimate; headline from C-DEC + M1 |
| R7 | C-BEH ✓ only via PEER (X-NEG0 ≥ BMM) | Polarity = peer contrast | Restated as Chucky-style contrast on issue effects |
| R8 | C-BEH ✓ only on repo-exception steps, builtin-only null | Possible residual-name effect | Builtin-only result is the claim; if null, treat as R3 |
| R9 | C-BEH inconclusive | Underpowered | Reported as an estimate with CI; framing as R6 |

---

## 7. Budget

### 7.1 GPU (NH)

| Line | Opt | Central | Pess |
|---|---|---|---|
| P0v (K8) | 0.3 | 0.4 | 0.5 |
| G0, 750 tasks | 0.5 | 0.8 | 1.2 |
| G1, 750 × k = 10 | 1.6 | 2.4 | 3.8 |
| **G2, R2 600 × k = 10 (recosted, F7)** | 1.3 | 1.9 | 3.0 |
| B4: B-DIRECT\* (3 variants on CAL-tune; best on EVAL original + R2) | 0.6 | 1.0 | 1.6 |
| B-RECALL | 0.1 | 0.15 | 0.25 |
| G4: Qwen3-32B-AWQ, 150 tasks, original + R2 | 0.8 | 1.2 | 2.0 |
| Session overhead | 0.2 | 0.3 | 0.5 |
| Reserve | 1.5 | 1.8 | 2.8 |
| **Total** | **6.9** | **9.95** | **15.65** |
| opt: B-RERANK (ladder rung 1 if not cut) | +0.3 | +0.5 | +0.7 |
| opt: CoSIL, 100 tasks | +1.2 | +2.0 | +3.0 |

The pessimistic case with all options is 19.4 NH, under 25.

### 7.2 CPU (kh)

| Line | Opt | Central | Pess |
|---|---|---|---|
| Images/installs | 25 | 50 | 120 |
| HEAP walks | 20 | 40 | 90 |
| Traces (EVAL-exec 60 + CAL-tune 40, ×3) | 12 | 23 | 45 |
| AST resolver + receiver-typed rules | 20 | 40 | 80 |
| CAL-train traces (K4) | 6 | 18 | 48 |
| Embeddings (CRE × 2 views) | 4 | 8 | 20 |
| R2 renaming | 6 | 12 | 25 |
| **Decoding (F7):** BMM 3⁴ × 100 × 10 × 2 ≈ 162k decodes; X-β0 3²; EVAL ~14 arms × 2 views × 650 × 20 ≈ 360k decodes at 0.1–0.3 s | 15 | 25 | 60 |
| K3c + oracle pilot | 4 | 7 | 12 |
| **Total** | **112** | **223** | **500** |

### 7.3 Person-days (before Nov 5)

| Work | Opt | Central | Pess |
|---|---|---|---|
| Day-1 counts, K1 probe | 1.0 | 1.5 | 2.0 |
| EVAL builder | 1.5 | 2.0 | 2.75 |
| Images + HEAP | 1.25 | 1.75 | 3.0 |
| AST resolver + graph + receiver-typed rules | 1.5 | 2.0 | 2.75 |
| Tracer + CAL-tune / EVAL-exec traces | 2.0 | 2.5 | 3.5 |
| g_φ + audits | 1.0 | 1.25 | 2.0 |
| Decoder (I/D/ε-skip, missing-call credit, orientation) + arms + B-ALIGN + CV | 2.25 | 2.75 | 3.75 |
| K3c + oracle pilot | 0.2 | 0.25 | 0.5 |
| Prompts, vLLM, span check, ~8 sessions | 1.25 | 1.75 | 2.25 |
| R2 renaming | 1.0 | 1.5 | 2.0 |
| Embedding baseline (CRE only) | 0.15 | 0.25 | 0.5 |
| Stats, figures | 0.75 | 1.0 | 1.5 |
| Qwen staging, dataset release | 0.5 | 0.5 | 1.0 |
| **Pre-freeze total** | **14.35** | **19.0** | **27.5** |
| Writing (Nov 5–12) | 1.5 | 2.0 | 2.5 |

**Week-1 realism (F8).** Only K0–K3a and K3c fit by Oct 17. K4–K7 land Oct 22–23.

### 7.4 Descoping ladder

**Never cut (the core):**
- static EVAL ≥400;
- G0, G1, R2;
- B-RECALL;
- B-DIRECT\* and BM25;
- X-BAG and X-β0;
- AST effects with span verification;
- **the tracer and the executed CAL-tune tasks (P1a)**;
- the truncated-metric statistics.

| Rung | Cut | Saves | Note |
|---|---|---|---|
| 1 | B-RERANK | Already off central | — |
| 2 | Qwen → 75 tasks or off | 0.5 pd, 1 NH | — |
| 3 | g_φ → AST q | 1.5 pd | — |
| 4 | HEAP → AST-only | 1.75 pd | ε-skip then uses AST-detected decorators only |
| 5 | EVAL-exec 60 → 30 | 0.5 pd | — |
| 6 | B-ALIGN greedy only | 0.25 pd | — |
| 7 | k = 10 → 5 | — | Stratum rule becomes ≥2 of 5 |
| 8 | EVAL 600 → 400 | — | **C-BEH is demoted to secondary unless the effect share is ≥37.5%** (≥150 tasks) |

**Floor: ≈14.5 pd (19.0 − 4.5), ≈6 NH.**

---

## 8. What runs on hidden repos without tests

| Component | Runs on a hidden repo with no tests? | Requirements |
|---|---|---|
| G0, G1 prompts | **Yes** | vLLM; 2 calls; ≈20–40 s extra (INFERRED) |
| AST graph, receiver-typed AST effects, span check, decoder, BM25/CRE prior | **Yes** | Source only; CPU |
| g_φ (if K4 passes) | **Yes** | Trained offline; static features |
| HEAP | **Yes, with caveats** | An importable environment; executes module top-level code (not tests); best-effort sandbox |
| Tracing, EVAL-exec, P1/M1 | **No** | Offline only |

The wording is "no tests at inference".

---

## 9. Change logs

### 9.1 v2 → v3 (N-items)

| ID | Fix |
|---|---|
| N1 | μ′ omnivore penalty (now inside β); LOC rule. The base-rate claim is withdrawn (F5) |
| N2 | NEG/PEER separable; missing-call credit on "−" steps (F9) |
| N3 | Receiver-typed AST for implicit raises; g_φ for intended bits; noisy-OR (F9) |
| N4 | Span check; "none" option |
| N5 | Per-class bits (R2: any repo exception, F2) |
| N6 | K3 bands; K3a/b; K3c added |
| N7 | h_ψ dropped; repo-resolved, deduped sites (F9) |
| N8 | I/D states; delete credit at T; ε-skip restored (F3) |
| N9 | Centred similarity |
| N10 | Local teleport (residual hub dilution reported, verify N10) |
| N11 | Undirected PageRank, λ = 0 |
| N12 | Site-local ∅ |
| N13 | P1a on CAL-tune |
| N-orient | X-BAG inherits BMM's orientation (F4) |
| N-tune | BMM-only tuning on 3⁴; inheritance; recosted (F7) |
| N14 | Rank-1 X-BAG |
| N15 | RBO over top 100 (F4) |
| N16 | X-LEXEFF⁺; the lexical-analogue sentence corrected |
| N17 | No identity map in BMM (F2); R8 |
| N18 | μ′ inside β (F1) |
| N-ctrl | DPR within package; X-NULL donors; graph-free uses π (F10) |
| N19 | Pool arithmetic, dedup |
| N20 | B-RECALL ∧ not-named rule (F10); docstring renaming |
| N21 | Truncated metric; X-BAG tail |
| N-gold | Min-gold; hunk rules |
| N22 | Single metric; decision rule (F6) |
| N23 | Wild cluster bootstrap; C-USE underpowered |
| N24 | Prefix extraction; Shrey's token |
| N25 | Moot (R1 dropped) |
| N26 | Raiser vs caller stratum |
| N27 | 19.0 pd central |
| N28 | Tree excludes tests |
| N29 | Moot; exclusions by polarity (F10) |
| N30 | DEV labelled / optional |

### 9.2 Verifier (F1–F11, E-1 to E-12)

All applied; see the table at the top. Details:
- E-1 → F5 / Prop. 0.
- E-2 → F1.
- E-3 → F3 / Prop. 4.
- E-4 → F4.
- E-5 → F2.
- E-6 → receiver-typed rules, noisy-OR, per-bit dynamic range on CAL-tune reported before K9.
- E-7 → deduped repo sites; threshold distribution reported.
- E-8 → memorisation rule.
- E-9 → exclusions by polarity.
- E-10 → missing-call credit.
- E-11 → graph-free with π.
- E-12 → ≥2 of 5.

**Dropped v1 diagnostics restored:** L4, L10, L11, L13, L18. Critic HYTRACE item 1 (first + last steps only) remains absent (minor).

### 9.3 Prior-art items (PA2)

- Closeness stated in §0.1.
- FSE'24 → B-ALIGN, plus the Prop. 3c framing.
- The updated claim is edited per F5.
- Must-not-claim items 11–16 are in §0.4.
- NEG and PEER ablations; R7.
- Pathidea wording fixed.
- 2607.09553 used for K3a.
- Classic papers are read before writing.

---

## 10. Assumptions (short)

| # | Assumption | Checked by |
|---|---|---|
| A1 | ≥600 EVAL tasks over ≥25 repos | K0 |
| A2 | Shrey's token enables ≥60 image extractions | K1 |
| A3 | Effect share ≥35%; sibling separation ≥0.6 | K3a/K3b; K3c |
| A4 | SD ≤ 0.78; ICC ≤0.15 | K10 |
| A5 | P1a ≥0.80 with last-step substitution | K6/K9 |
| A6 | Edge recall ≥0.6 | K5 |
| A7 | 19 pd holds | Oct 17 review |
| A8 | Decode ≥250 tok/s | K8 |
| A9 | Qwen AWQ can be staged | G4 |
| A10 | Memorisation does not saturate | B-RECALL |

---

## 11. Questions for Shrey

1. **Docker Hub token.** Please create a read-only access token and add it to Kaggle Secrets yourself. I will not handle it. Without it, EVAL-exec is ≈30 tasks and HEAP is pure-Python only.
2. **Priority.** 19 pd central is ≈70% of the days to Nov 5. Is that acceptable against the main track, or should I start at the ≈14.5 pd floor?
3. **GPU.** ≈10 NH central (15.7 pessimistic), mostly Oct 20–Nov 1. Does that collide with LoRA training or 12-h evals?
4. **Topic fit.** Do you accept decision (b): a localisation and graph-inference framing, with q_v as a secondary artifact only?
5. **Framing.** Do you accept R5 as the pre-registered likely headline (conditional on real effects), and R3/R4 as analysis papers?
6. **Sessions.** About 8 manual L4 starts are needed. Which days suit you?
7. **Release.** Is a public Kaggle dataset release of generations, graphs and effect tables OK?

No file other than this one was modified.
