# Break report on architecture v2 ("twin-null RFT"): logic, edge cases, statistics, compute (8 Oct 2026)

**Role:** fresh breaker (logic, edge cases, statistics, compute). I read `architecture_v2.md` in full, including §2.3, B1–B20 and the v1→v2 change log. I skimmed `break_v1_logic.md` and `break_v1_method.md` to check whether each v1 break was fixed or only renamed (§8). Environment facts come from `docs/training_feasibility_4xl4.md` and `docs/HQ.md`. Prior art is out of scope: `break_v2_priorart.md` covers it, and I touch it only where it changes the logic.

**Severity:**
- **fatal**: the stated mechanism or a headline hypothesis cannot be supported as designed.
- **major**: a conclusion can be wrong or uninterpretable unless fixed.
- **minor**: disclose, or fix cheaply.

**Labels:** as in v2. Every number I compute here is INFERRED arithmetic on v2's own GUESSED inputs unless marked otherwise.

Breaks are numbered **L-01 … L-27**.

---

## 0. Bottom line

1. **The [CLAIMED] cancellation is vacuous in population expectation (fatal, L-01 + L-02).**
   - A wiring-blind commit to node w earns ½·(1{w=v} + 1{w=v′}), not "½ whichever node it names". So twins cancel only features that separate v from v′. Every feature the symptom-matcher *forces* them to share is still credited exactly as under U: level, module, family, arity, uncontracted status, d_obs class, and the module-level import graph.
   - Bugs are placed by the generator, not by real-world bug processes. So v (in G) and v′ (in ρG) are exchangeable draws, as far as the null preserves statistics. In expectation, pairing is then identical to an **unpaired 50/50 real+null mixture**. The M−U difference in blind credit is at most ½ × the H3 effect, and v2 predicts H3 ≈ 0.
   - v2's own predictions (H2a +0.12, H4 −0.25, H3a ≈ 0.01) are therefore mutually inconsistent, unless the effect comes from something else: the evaluation bias in L-16, or finite-sample pairing effects. Pairing is CAD, and CAD is prior art.
2. **Wiring-blind solvers that solve *both* twins get full credit (fatal for the agent/RFT half, L-03).**
   - Example: re-implement the deepest failing contract c by fitting its visible expected values. With affine-style templates, three numbers pin the function.
   - Hidden F2P only checks the root and contract outputs on fresh inputs. Nothing pins the bug node's own behaviour, so the fit passes in X and X̄ without reading a single edge.
   - §2.3 never considers actions that solve both twins.
3. **The null twin is nearly the real graph, and twin supply is tiny (major→fatal, L-11 + L-12).**
   - The swap rule needs equal (level(a), level(b), module(b), family(b), arity(b), edge type). With ≈40 families assigned independently of topology, a 500-node graph has ≈14k cells and ≈0.06 edges per cell. That gives ≈30 legal swaps in total, so ≲6% of edges change.
   - Exact symptom matching selects bugs whose only parent was swapped. Twin discordance is then *one hop, at the last edge*, and O2 twins (in-degree ≥2 required) are ≈0.
   - H3 equivalence becomes an artefact of rewire strength.
4. **No headline real-world claim is powered (major, L-16, L-17, L-21).**
   - H2a and H4 are evaluated on twins pooled over null halves that only M and N trained on, which builds in an advantage for M.
   - A 3-seed hierarchical bootstrap ignores the 2–4 degrees of freedom of the seed component.
   - Recomputed power: H1 ≈ 30–70%, H2b ≈ 8–18%. The retention battery would pass non-inferiority with probability ≈ 3–30% *even if retention were perfect*.
   - The gatekeeping chain H4→H2a→H1→H2b completes with probability ≈ 0.02–0.08.
5. **The "RL" is vestigial, and the modal outcome is a stop at G2 in week 2 (major, L-07, L-17, L-21).**
   - Stage B is 20 trajectories at an effective batch of 16, i.e. 1–2 optimiser steps, inside a 10-step warm-up. Under D6 it is less than one step.
   - Stage A carries ≈3,200 supervised level tokens per adapter. The "0.5M tokens" is mostly unsupervised context.
   - G2(a) is a logistic slope on 90 clustered rollouts, adjusted for a mediator. G2(b) is a point-estimate rule at SE ≈ 0.075. Either failing **stops the study**, and that outcome is not in the "every outcome publishable" table.

**Salvageable:** yes, as a narrower and honest study: **paired vs unpaired null-twin augmentation for code-graph reachability items**, plus the H3 null-model analysis with a strong-null instrument check. Agent RFT becomes exploratory. Minimal v3 changes are in §10.

---

## 1. The cancellation argument, written out formally

### 1.1 Setup, and what §2.3 actually proves

- Pair (X, X̄) with twin indicator Z ∈ {X, X̄}, P(Z=X) = ½.
- Pre-wiring history h, with the same observation distribution under both Z (v2's premise; L-08 breaks it).
- Q_Z(h, a): success probability of action a at h under twin Z, continuing with the behaviour policy π₀.
- Policy-gradient direction at h under balanced presentation:

  `g(h) = Σ_a π(a|h) · [Q̄(h,a) − b(h)] · ∇log π(a|h)`, with `Q̄ = ½(Q_X + Q_X̄)`.

What §2.3 proves is only that **g depends on Q̄**. Everything else depends on what Q̄ looks like for each class of action:

| Action class at h | Q_X | Q_X̄ | Q̄ | §2.3 says |
|---|---|---|---|---|
| commit "edit v" (gold in X) | c | 0 | c/2 | c/2 ✓ |
| commit "edit v′" (gold in X̄) | 0 | c | c/2 | c/2 ✓ |
| commit "edit w", w ∉ {v, v′} | 0 | 0 | **0** | "½ whichever node it names" ✗ |
| read wiring, then commit | q·c | q·c | q·c | ✓ |
| **twin-agnostic repair that solves both** (L-03) | r | r | **r** | not considered |

Here c is the probability that the patch is correct given the right node. It is not 1 (see L-10).

### L-01 Blind commits are not all worth ½: twins cancel only v-vs-v′ discrimination — **fatal for the [CLAIMED] sentence; major for the method** (B1, B2)

**Claim in §0:** "the expected outcome credit … for any decision taken before reading the wiring therefore cancels exactly". **This is false.**

For a wiring-blind heuristic σ that maps pre-wiring observations to a node, the twin-averaged credit is

`Credit_M(σ) = ½·[P(σ = v) + P(σ = v′)]`, against `Credit_U(σ) = P(σ = v)`.

- Any feature φ with φ(v) = φ(v′) is credited in M **exactly as in U**.
- The symptom matcher *forces* a long list of such features:
  - level(v) = level(v′) and module(v) = module(v′);
  - same family, operator and arity;
  - both uncontracted (both need d_obs ≥ 2 under the same contract set);
  - identical failing visible tests, and identical imports (block-preserving swaps).

**Worked example.**

```
deepest failing contract c (module m0) — shared, named in the issue
uncontracted nodes below c in module m3 (visible: file m3.py has no tests for them):
    v  (affine, arity 1)   v′ (affine, arity 1)   w1 (fold, arity 1)
other nodes below c: w2 (m3, contracted, test passes), w3 (module m2), w4 (m3, level 3)
```

Heuristic σ = "edit an uncontracted function in the module that c's module imports". It needs no edge reading, because imports are identical across twins (K-11 fix). σ picks uniformly from {v, v′, w1}:
- U: 1/3.
- N: 1/3.
- M: ½(1/3 + 1/3) = **1/3**.

So M applies **zero** extra pressure against σ. The only thing M neutralises is a preference *between* v and v′.

**Consequence.**
- The twin mechanism teaches the generator's placement fingerprint (uncontracted, same-module, family-matched, d_obs ≥ 2) as strongly as U does.
- It teaches the module-level (block) graph as strongly as U does, because the block graph is identical across twins.
- Twins are discordant only *inside* blocks.

**Fix.**
- Restate the claim exactly: "cancels credit for node-intrinsic features that differ between v and v′".
- Randomise the matched-feature set against the real-item distribution, or drop constraints that are not needed for B2. Level equality is needed for exactness; family equality is not (L-11).
- Add an audit: train a learned blind solver (gradient-boosted, on all pre-wiring features) and report its **per-half** accuracy, not only pair accuracy. If per-half accuracy is ≫ 1/|pool|, the fingerprint is learnable in every arm.

### L-02 In expectation, pairing ≡ an unpaired real+null mixture, and U has no topology shortcut to cancel — **fatal for the stated mechanism** (B1, B3, H2a, H4)

**Step 1 (exchangeability).**
- In X, the bug v is placed by C4's sampler in G. In X̄, v′ is placed in ρG by the same sampler under the same constraints.
- Real topology determines the *graph*. It does not determine *where bugs are*: no real-bug distribution is used.
- So, for any node-feature heuristic σ, P(σ = v | real) and P(σ = v′ | null) differ only through statistics that ρG does not preserve. Those are exactly the motif and higher-order statistics that H3 is about.

**Step 2 (no memorisation ⇒ population expectation).**
- Test pools are new, so the model generalises through features, not through the identity of h.
- The expected gradient over the training distribution therefore depends only on the **marginal** distribution of (features, label) over twin halves.
- Under M, that marginal is the 50/50 mixture of real and null halves, which is exactly what an **unpaired** mixture "Mix" provides (real halves from pools A, null halves from pools B, equal dose).
- The per-h cancellation only matters to a model that has memorised h.

**Step 3 (the bound).**

`Credit_M(σ) − Credit_U(σ) = ½·[P(σ=v′|null) − P(σ=v|real)] = −½·Δ_H3(σ)`

where Δ_H3 is the real-vs-null difference in the success of heuristic σ. The same identity holds per element for item cross-entropy: M's statistics-only optimum is the mixture posterior.

**The dilemma.**

| If H3 says… | Then M vs U, in expectation |
|---|---|
| real ≈ null beyond {degree, level, block} (v2 *predicts* this: H3a ≈ 0.01, H3b ≈ 0) | No topology shortcut exists in U for M to cancel. M − U ≈ 0, apart from finite-sample pairing effects |
| real carries extra signal | That signal holds **on real test code too**. Real reachability regularities are not "spurious" in the CAD sense. Halving them in M lowers the Bayes-optimal real-item AP for any model that cannot read every edge, so H2b's sign is ambiguous, possibly negative |

**Internal inconsistency.** Under v2's own H3 prediction, the mechanism predicts H2a ≈ 0 and H4 ≈ 0. Yet v2 predicts +0.12 and −0.25. If those numbers appear, they must come from:
- **(i)** M having trained on null-topology worlds that half of the evaluation set consists of (L-16, a design artefact);
- **(ii)** finite-sample pairing efficiency, which is the paired-CAD effect: prior art and an *empirical* claim, not the §2.3 identity;
- **(iii)** M's data being 50% null worlds. If N > U (one H3 outcome), M > U for that reason alone.

**§2.3's motivating example is also wrong.** "Real-topology regularities, such as 'the bug sits in the more central callee', can push [P(gold=v | statistics)] towards 1" cannot happen here. Centrality is preserved per node by the rewire, and bug placement is synthetic and symmetric across twins.

**Fix.**
- Add arm **Mix**: unpaired 50/50 real/null at equal dose and equal distinct-site count. This is the single control that separates "pairing" from "mixture".
- H2a := M vs Mix. The method claim becomes "paired twins beat the same data unpaired".
- Restate §2.3 as a finite-sample, variance-reduction argument, with CAD and paired-CAD cited as the prior mechanism.

### L-03 Twin-agnostic repairs solve both halves; §2.3 omits them — **fatal for the agent/RFT half, major for G2 and A** (B2, B3, K-17/K-18 fixes)

The reward is "hidden F2P on fresh inputs ∧ P2P ∧ no issue literals". It pins the behaviour of the **root and the failing contracts**, not of the bug node.

**Worked example (O1, affine family).**
```
gold:     v(t) = 2t + 5       x(t) = v(t) + 1      c(t) = 2·x(t) − 4   ⇒ c_gold(t) = 4t + 8
X  bug:   v(t) = 7t + 5                                               ⇒ c_X(t)    = 14t + 8
X̄ bug:   v′ mutated instead (x→v′ in ρG)                              ⇒ c_X̄(t)   = some other affine map
issue / visible contract: c(1) == 12 (got 22), c(2) == 16 (got 36)
```

Policy "rewrite c's body as `return 4*t + 8`, fitted from the visible expected values":
- It passes the hidden F2P on fresh inputs, because c now *is* gold.
- It passes P2P, because c's behaviour is gold and everything else is untouched.
- It passes the literal check, because 4 and 8 are not literals in the issue.
- It needs **no wiring**, so Q_X = Q_X̄ = r.

Under M its credit is r. Reading only reaches q·c. If templates compose into low-dimensional function classes (affine ∘ affine is affine; many arithmetic families are closed or nearly so), r can be far above q·c.

Variants that work the same way:
- a compensating edit at a shared ancestor;
- a lookup-or-fit shim inside the root's helper.

**Why the audits miss it:**
- Audit (i) checks *pair accuracy of a node-picking solver*. A shim picks no bug node.
- C5's single-site alternative library (revert, swap, exchange bodies, rename, shim) is specified only for O2, and "shim" is not "fitted re-implementation".
- G2(c) looks for pair-both-solved *above* the product. A base model that rarely writes fitters will not trip it, but RFT on successes amplifies fitters in **all** arms, and most in M, where they succeed in both halves.

**Fix.**
- Hidden **per-node** contracts on fresh inputs for **every node in c's subtree, including the bug node**, with hidden P2P on every non-bug node. Reward then equals "restore the bug node's gold behaviour, change nothing else".
- Add "fit-and-replace an ancestor" and "compensating parent edit" to C5's alternative-fix library for all operators.
- Audit an observation-free *fitting* solver's pair-both rate before training.

### L-04 RFT does not remove a base bias; it freezes it. H4 compares against an arm that amplifies — **major** (H4, B3)

- RFT is SFT on π₀ successes. Its fixed point is `π*(a|h) ∝ π₀(a|h)·Q̄(a|h)`. The expected gradient matches policy gradient only at θ = θ₀; after that, SFT regresses towards π*.
- Among blind commits with equal Q̄ (v and v′), **the base ratio π₀(v) : π₀(v′) is preserved exactly.**

**Worked example.** The base prefers v (for any reason) with π₀(v) = 0.6 and π₀(v′) = 0.2; reading has π₀ = 0.2 and q = 0.5; take c = 1.

| | M target | U target |
|---|---|---|
| blind v | 0.6·½ = 0.30 | 0.6·1 = 0.60 |
| blind v′ | 0.2·½ = 0.10 | 0 |
| read | 0.2·0.5 = 0.10 | 0.2·0.5 = 0.10 |
| normalised | v 0.60, v′ 0.20, read 0.20 (**= base**) | v 0.86, read 0.14 |

- M ≡ base on twin-invariance, while U becomes *more* wiring-blind.
- H4 (M − U ≤ −0.15) passes because U got worse, not because M "removes observation-independent answering". The H4 claim needs M vs **base** (and vs Mix).
- Reading is not favoured over base under M here either: its share stays 0.20. It only grows when q·c > V̄.

**Fix.**
- H4 := M vs base and M vs Mix, **signed**: the score must move towards the correct label (L-19), not just move.
- Report the blind-commit ratio before and after training on held-out twins.

### L-05 Two routes to "guess randomly between the two candidates" — **major** (B3; generality claim for GRPO)

**(a) Success-balanced selection in RFT.**
- If C7's matching, or a seed bootstrap that keeps only pairs with successes in both halves, equalises successes per half, then a blind-only base gives X-successes that are all "edit v" and X̄-successes that are all "edit v′".
- The contexts are identical, so the SFT target at h is **50/50 over {v, v′}**. The model is trained to guess between the matched candidates.
- v2 says "natural success counts" for Stage B but "balanced presentation" in §2.3. Pin this down.

**(b) GRPO with per-group std normalisation** (the online drop-in, §2.7). Policy: blind v 0.45, blind v′ 0.05, read 0.50 with q = 0.1. Then p_X = 0.50 (σ_X = 0.5) and p_X̄ = 0.10 (σ_X̄ = 0.3).

| Action | Adv in X | Adv in X̄ | Net (std-norm) | Net (Dr.GRPO, no std) |
|---|---|---|---|---|
| blind v | +1.00 | −0.33 | **+0.67** | +0.40 |
| blind v′ | −1.00 | +3.00 | **+2.00** | +0.40 |
| read | −0.80 | 0.00 | −0.80 | −0.40 |

- Std normalisation credits the base-disfavoured blind guess 3× more per unit probability. That pushes π(v′)/π(v) towards 1: "guess randomly between the two".
- §2.3's guarantee "never rewards a blind guess more than another" holds only **without std normalisation**, or with a **pair-pooled mean *and* std**.
- In expectation, per-group mean baselines are fine: they scale advantages by (K−1)/K.

**Fix.** The GRPO drop-in must specify a pair-pooled mean and no per-group std (Dr.GRPO-style), with both halves in every batch.

### L-06 Shared prefixes and partial reads: discordance lives on the last hop — **major** (construct; links to L-12)

- Shared early actions (read the issue, run pytest, open c's file) get Q̄ credit, the same in all arms up to the overall success rate. Fine.
- But exact symptom matching (L-12) makes the twin difference almost always **one swapped edge at the bug's parent x**: x→v in G, x→v′ in ρG.
- The path root→…→c→…→x is identical in both twins. So the "multi-hop" traversal is credited identically in M and U.
- M's *distinctive* pressure is on reading one call site at x, which is a grep-able lexical call because dark edges are never rewired (L-13).
- Partial reads that stop above x reveal nothing about Z.

The method therefore trains "read the last call site", not multi-hop graph reasoning.

**Fix.**
- Report the hop-position of every discordant edge relative to the bug path.
- Require that some twins have their discordant edge ≥2 hops above the bug. This needs relaxed matching on *passing* tests, not failing ones (L-12).

### L-07 Stage B is 1–2 optimiser steps inside warm-up; Stage A is ~12 steps — **major** (B6, the "RL" claim)

- **Stage A:** ≈200 items at an effective batch of 16 gives **12.5 steps**, with a cosine schedule and **10 warm-up steps**. The mean learning rate is ≈0.5–0.55× nominal.
- **Stage B:** ≈20 trajectories at batch 16 gives **1.25 steps**. If warm-up restarts per stage, Stage B runs at ≤1e-5. Under D6 (10–15 trajectories) it is **less than one step**.
- So the "RFT" stage cannot change behaviour. The method is item SFT (CAD on items).
- BM-§6 predicted Stage B ≈ 0. v2 kept the stage and kept "RL/RFT" in the [CLAIMED] sentence and the prior-art delta ("use is in agent RL/RFT").

**Fix.**
- Either drop Stage B from the claim (the method is "null-twin paired item SFT"; RFT is exploratory), or give it a real dose: ≥150 trajectories per arm, which requires ≥5× the rollouts (not affordable; §6).
- Use batch 4–8 and warm-up ≤10% of steps.

---

## 2. B2 indistinguishability

### L-08 The pre-wiring history is not identical: the issue numbers differ, and the twin-identity probe is ill-posed — **major** (B2)

- §2.2 admits that the twins "differ only in the expected-vs-actual numbers". The *actual* values are a function of wiring and bug, so h₀ ≠ h̄₀ from turn 0. The §2.3 premise ("no information about which twin") is false as stated.
- Statistical leaks via magnitude are plausible:
  - |actual − expected| scales with how many templates the error passes through (d_obs) and with the operator;
  - v and v′ can have different d_obs (both ≥2).
- Audit (i) never lists the numbers as solver inputs.
- The B2 check "prefill, ask which twin" is meaningless. A model shown one item has no referent for "which twin". It will be at chance whether or not a leak exists.

**Fix.**
- Audit with a *learned* cross-validated classifier: train on pairs, test on held-out pairs. Inputs: every pre-wiring observation (issue text, visible numbers, file sizes, imports, test files). Target: gold ∈ {v, v′}, and per-half gold among the pool.
- Optionally probe hidden states linearly.
- Equalise d_obs within pairs (d_obs(v) = d_obs(v′)).

### L-09 "Reading the wiring" includes executing it; there are no tracebacks, but trace tools exist — **major** (construct, B19)

- Templates are total, so failures are assertion mismatches. **No traceback ever shows the call path.** Tracebacks neither help nor trivialise the task. (This answers the question in the task.)
- But the localizer has `run_command`:
  - `python -m trace --listfuncs`, `sys.setprofile`, coverage, or print instrumentation return the **executed call set** of the failing test in one call, in both twins;
  - this resolves every "dark" edge (dispatch, registry, getattr) at once, which nullifies K-07's dark-edge difficulty.
- The cancellation cannot tell static from dynamic reads. So Stage B (if it had any dose) would reward "run a tracer", a useful SWE habit but not graph *reasoning*.
- Meanwhile every powered readout is a tool-free item (call-set prediction from shown code). The Stage-B skill has **no powered readout**: A has n = 20 pairs and is descriptive.
- "Reading the wiring" and "reading an execution trace" are therefore indistinguishable to the reward. They are distinguishable only by logging *which tool produced the decisive observation*.

**Fix.**
- Log and classify decisive reads (static open/grep vs graph tool vs dynamic trace) as a descriptive mechanism readout.
- State in the paper that the agent-level claim is "execution-or-structure use".
- If static reasoning is the target, disable tracing modules in training sandboxes, and disclose that this departs from deployment.

### L-10 The K-12 fix made repair unidentifiable without algebra — **major** (B5, G2(b))

- A bug is "another valid family instance", and bodies are chance-indistinguishable (audit ii). So the *correct* constants are knowable **only** by inverting the visible numeric constraints through the composition between v and c.
- Real bugs come with semantic specs: names, docs, intent. Here names are random and there are no docstrings. Repair is pure numeric system identification.

Consequences:
- **c (patch correctness given the right node) is low.** B5's base success band [0.15, 0.6] is unsupported. At 0.05, the 300-rollout pool gives ≈15 successes in total, ≈7 per arm after matching.
- **G2(b) is likely to fail for the wrong reason.** "Bug-named success − normal ≥ +10 pp" needs localisation to be the bottleneck. If repair is the bottleneck, both rates sit near the floor and the study **stops** (§4).
- L-03's fitting shim becomes the *easiest* successful strategy, so successes over-represent it.

**Fix.**
- Put the gold program's intended constants in reach: show a per-family spec ("`scale_*` multiplies by the module's `SCALE` constant") or a reference implementation of the family elsewhere in the repo. Localisation, not inversion, is then the bottleneck.
- Pilot c on bug-named tasks during L1, before committing.

---

## 3. Twin supply: rewire strength, symptom-matching yield, source graphs

### L-11 The swap constraints leave almost no legal swaps, so ρG ≈ G — **major → fatal for H3 and for twin supply** (B4, B17, K-21/K-22 fixes)

A swap (a→b, c→d) needs equal type ∈ {lexical, callback}, level(a) = level(c), level(b) = level(d), module(b) = module(d), **family(b) = family(d)** and arity(b) = arity(d).

**Illustrative count** (a 500-node source after C2; GUESSED structure):

| Factor | Values |
|---|---|
| lexical/callback edges | ≈900 |
| (level a, level b) pairs carrying most edges | ≈10 |
| modules | ≈12 |
| families | 40 (assigned to positions independently of topology, §2.2) |
| arities | 3 |
| **cells** | ≈14,400 → **λ ≈ 0.06 edges/cell** |

- Expected swappable edge pairs ≈ Σ E[C(n,2)] = 14,400 × λ²/2 ≈ **28**, before multi-edge and self-loop rejections. At most ≈56 of 900 edges (**≈6%**) can ever change, whatever the number of proposals (10·|E| proposals do not create new legal moves).
- Without the family constraint: ≈360 cells, λ ≈ 2.5, ≈1,100 swappable pairs. That is workable, but mixing is still confined to small within-cell bipartite blocks.

**Consequences:**
- **Discordance is scarce.** Twin items need ≥4 discordant elements in a 16-node pool, and with ≈6% of edges rewired most items have 0–1. Selecting the few that qualify picks unusual subgraphs.
- **H3 is uninformative.** A null that differs from G in ≈6% of edges makes the "real ≈ null" equivalence (the predicted outcome) true by construction. The TOST passes because the null is the real graph, not because topology does not matter.
- The reported motif counts will look "preserved" for the same reason.

**Fix.**
- Drop family from the ρG swap rule. Make all templates share one value domain (e.g. ints in a bounded range, total) so that arity is enough. This is the K-15 tension: resolve it with a common domain, not with family matching.
- Report the **fraction of rewirable edges changed** and the Jaccard distance J(E_G, E_ρG). Pre-register a minimum (e.g. ≥40% of rewirable edges).
- Add a **far null** (per-type-degree only, no level or block constraint) as an *instrument positive control* for H3: if real-vs-far-null is not distinguishable either, the H3 instrument is blind.

### L-12 Exact symptom matching selects in-degree-1 bugs and single-edge differences; O2 twins ≈0; the relaxed fallback breaks B2 — **major** (B4)

- FailVis_G(B) = FailVis_ρG(B′) requires the contracted-ancestor set of v in G to equal that of v′ in ρG, at 40% contract density, plus equal failing roots.
- For nodes with fan-in (many ancestors), exact equality is improbable. The match is near-automatic when **v and v′ each have one parent x, and the swap exchanged x's callee.** The yield therefore concentrates there (→ L-06: one-hop discordance).
- **O2** requires in-degree(b), in-degree(d) ≥ 2, the opposite of where matches occur, plus "no single-site alternative passes". Predicted O2 twin yield ≈ 0, so the 25% O2 quota cannot be met by twins. Either the operator mix differs between twins and the U pool, or O2 is silently dropped.
- **Relaxed fallback** (equal failing roots + equal contract count + Jaccard ≥0.8 on names): the failing-contract sets then differ, so **the issue text names different contracts across twins**. That is a direct twin-identity and localisation leak, and the §2.3 premise is gone. It is not "weakens B2"; it **voids** it for those pairs.
- A 5% yield on candidate nodes says nothing about the yield per *(graph, item)* with ≥4 discordant elements and both halves passing C5. The joint yield is what limits the 75 + 60 pairs.

**Fix.**
- Day-1 CPU kill criterion on the **joint** yield (pairs passing C5 with ≥4 discordant elements).
- Report the in-degree distribution of matched v against all eligible nodes.
- Drop O2 from twins (keep it, if at all, as an unpaired operator in all arms).
- Drop the relaxed fallback, or treat relaxed pairs as a separate stratum excluded from the mechanism claims.

### L-13 Dark edges are never rewired, so twins teach only lexical call sites — **major** (construct, H2b stratum)

- K-24's fix keeps dispatch, registry, decorator and getattr edges identical across twins. All discordance therefore sits on grep-able lexical/callback edges.
- The hard part of real code-graph reasoning (dark edges; HQ A2: undirected tools, ~69% of tasks unbridged) is never discordant.
- The twin mechanism puts **no pressure** on dark-edge reasoning beyond U.
- H2b's pre-registered stratum (depth ≥2 × statically-reachable distractor) should be split by the edge type of the decisive hop. The prediction for the dark stratum is M ≈ U.

**Fix.** Stratify H2b by lexical vs dark, and state the method's scope as "lexical-edge reading". Alternatively, rewire *registry keys* by swapping string constants within a registry; that keeps the class structure and is twin-safe.

### L-14 Twin items can have identical inputs and contradictory labels — **major** (Stage A validity)

- "The pool is identical across twins; only the bodies at swapped call sites … differ."
- If the swapped edge that causes an element's discordance has its call site in a function *outside* the 16-node pool, both twin items are **textually identical with opposite labels**.
- Cross-entropy then drives that element to 0.5 (hedging), and the item is unsolvable in both arms. This injects label noise that is larger in M, which sees both halves.
- Path-completeness is required for *real* items (§2.9) but not stated for twin items.

**Fix.** Twin items must be **path-complete in both wirings**: every edge on every reach path to every discordant element is shown in both twins. Measure how this interacts with the 16-node and 1.8k-token budget (L-11 makes it binding).

### L-15 Real traced source graphs: levels, SCCs and the integration-root restriction — **minor**

- Longest-path layering on traced graphs is fine, and the level-preservation proof holds: each node's multiset of predecessor levels is unchanged, so by induction every level is unchanged.
- But SCCs compiled as bounded recursion have no well-defined level. Swaps touching condensed nodes need a stated rule.
- Integration-root-only growth plus the hub cap (30%) plus the SCC cap (5%) together select tree-like, low-fan-in regions. That compounds L-12.

**Fix.** Report, per source repo, the share of nodes surviving each C2 rule and the in-degree histogram of surviving nodes.

---

## 4. Statistics, gates and power

### L-16 H2a and H4 are evaluated on twins pooled over null halves that U never trained on — **fatal for H2a as evidence of the mechanism** (H2a, H4, GW-a)

- The held-out evaluation set is "both twins pooled", so half of it is ρG worlds. M (and N) trained on ρG-style worlds; U did not.
- Any distribution shift between real and null worlds (L-11 says it is small, but it is non-zero by design) gives M a home advantage on 50% of the evaluation items.
- With Mix absent, M − U conflates the pairing effect with "trained on the test distribution".

**Fix.** Report H2a per half: **M vs U on real halves** (U's home turf) and **M vs N on null halves**. The mechanism claim requires M to win both, or M > Mix on the pooled set.

### L-17 The 3-seed bootstrap is anti-conservative; recomputed power — **major** (B8, H1, H2b)

- Resampling 3 seeds within an arm gives 10 distinct bootstrap draws and ignores that the seed component has 2 df (per arm) or ≈4 df (difference).
- Small-data LoRA fine-tuning has notoriously large seed variance (Dodge et al. 2020, INFERRED). Data-resampled seeds add the data variance on top. B8's σ_seed = 0.015 is optimistic.
- Recomputed with Satterthwaite df: item component SE ≈ 0.15·√(1.9/320) ≈ 0.0116; seed component σ/√3 per arm.

| Contrast | σ_seed | Total SE | df | MDE80 | Predicted | Power at prediction |
|---|---|---|---|---|---|---|
| H1 (M vs base) | 0.015 | 0.0145 | ≈15 | 0.044 | +0.04 | ≈0.72 |
| H1 | 0.030 | 0.021 | ≈4 | **0.077** | +0.04 | ≈0.30 |
| H2b (M vs U) | 0.015 | 0.017 | ≈14 | 0.050 | +0.02 | ≈0.18 |
| H2b | 0.030 | 0.027 | ≈5 | 0.095 | +0.02 | ≈0.08 |
| H2a (synthetic, as predicted) | 0.03 | ≈0.026 | ≈5 | ≈0.09 | +0.12 | ≈0.9 (but biased, L-16) |

- **The gatekeeping chain** H4 → H2a → H1 → H2b completes with P ≈ 0.8 × 0.8 × (0.3–0.7) × (0.08–0.18) ≈ **0.02–0.08**, *before* the G2 and GW gates.
- The one claim a practitioner cares about ("M beats the standard synthetic curriculum on real code") is the least powered, and it sits last in the chain.
- H4 heads the chain, but it is the least meaningful test (L-04, L-19). If it fails, H1 is never tested confirmatorily.

**Fix.**
- Treat the seed as the unit for the arm term: per-seed arm means, then a t-test or mixed model with Kenward–Roger df.
- Raise M, U and Mix to 5 seeds (training is ≈0.36 h central per adapter; see §6 for the trade). Cut N to 2 seeds, or fold it into Mix.
- Reorder the chain: H1 → H2a (vs Mix) → H2b, with H4 as a secondary under Holm.

### L-18 G2 is a likely, unpublishable stop — **major** (B5, §1.3 completeness)

- **G2(a) power.** 90 rollouts in 45 *pairs*, so the design effect is >1; base success maybe 0.15; d_obs ∈ {2…5}. Unadjusted, the slope z ≈ 1.5–2, giving ≈30–55% power.
- **Adjusting "for tokens consumed" conditions on a mediator.** Deeper bugs → more tokens → more failures, so the adjustment removes part of the effect it is meant to detect. Tokens are also post-treatment and outcome-dependent (successes stop early), which is a collider. Power falls further, and the estimand is wrong.
- **G2(b)** is a point-estimate rule (≥ +10 pp) with SE ≈ √(2·0.15·0.85/45) ≈ 0.075. Even a true +10 pp passes ≈50% of the time, and L-10 makes the true contrast small.
- **G2(c)** compares pair-both-solved with the product of the half rates + 0.05. With expected counts of ≈2 pairs it cannot detect anything. Its direction also only catches *leaks*: blind guessing pushes both-solved *below* the product, and L-03 fitters push it above, but only once trained.
- **P(G2 passes) ≈ 0.15–0.3.** On failure v2 says "stop and report", but the §1.3 outcome table (which claims "every row is publishable") has no G2 row.
- **The (d′) rollouts for G2(b) are not in L2** (only 90 = 45 × 2 normal rollouts). They are unbudgeted (§6).

**Fix.**
- Make G2 descriptive with a pre-registered *pivot*, not a stop: if (a) fails, report the d_obs curve and continue with items only.
- Model cap hits as a competing risk without adjusting for tokens.
- Size G2(b) from the pilot.
- Add a G2-fail row to §1.3.

### L-19 H4's twin-invariance metric is unsigned — **minor/major**

- "Score differs by <0.1 between twins" rewards *any* sensitivity to call-site text, including noise.
- A model that randomly perturbs scores whenever call-site tokens change lowers invariance with zero accuracy gain.

**Fix.** Use signed discordant accuracy: the share of discordant elements whose score moves in the label's direction, i.e. Δscore × (2y − 1) > 0.1. H2a already covers the AP side.

### L-20 H3a equivalence measures the distance between the rewire and the real graph, not the value of topology — **major** (H3, B17)

- The crossover interaction is (U→real − U→null) − (N→real − N→null). Matched-distribution advantages make it positive whenever real and null worlds differ, and ≈0 whenever they do not (L-11).
- So the "equivalent" verdict is predicted by rewire weakness. The "U > N" verdict can arise from home-advantage alone.

**Fix.** Use the far null (L-11) as the instrument positive control. Report the interaction as a function of rewire strength (two or three nulls of increasing distance). H3 then becomes a dose–response over null strength, which is also a stronger *analysis* contribution.

---

## 5. Retention, near-SOTA and dose

### L-21 The retention battery cannot show non-inferiority even when true Δ = 0 — **major** (R, the "near-SOTA" goal)

Paired SE ≈ √(discordance/n), one-sided 95%:

| Test | n | Margin | Discordance d (GUESSED) | SE | P(pass \| true Δ = 0) |
|---|---|---|---|---|---|
| MMLU-Pro slice | 300 | −2 pp | 0.02–0.05 | 0.008–0.013 | 0.46–0.79 |
| HumanEval+ greedy | 164 | −3 pp | 0.03–0.08 | 0.014–0.022 | 0.39–0.82 |
| Tool-call validity | 100 | −1 pp | 0.01–0.03 | 0.010–0.017 | **0.14–0.31** |
| **All three (≈independent)** | | | | | **≈0.03–0.2** |

- "Near-SOTA retained" will usually be *not shown*, even if true. Per adapter the outcome is a coin flip, which invites selective reporting.
- B16 relies on the AI Studio bf16 endpoint. HQ 30 Sep records that Shrey **dropped the API route** ("secret/venv friction, local = faithful"). This reference is not available without reversing a logged decision.
- The bf16-vs-W4A16+M comparison also mixes readouts: verbalised p for API models vs a level-token expectation for trained adapters. The "gap ≤0.03" compares two instruments.

**Fix.**
- Primary retention = **exact KL** (already planned, well powered) plus **per-item gold log-probability margins** on MMLU-Pro and HumanEval+ prompts (continuous and paired; SE an order of magnitude smaller). Accuracies become descriptive.
- Drop the 1 pp tool-call margin, or use ≥1,000 held-out turns (prefill only, cheap).
- Score all models in verbalised mode for the bf16 gap, or drop the gap.
- Log a decision in HQ if the API reference is reinstated.

### L-22 "0.5M tokens per adapter" is ≈3–4k supervised labels; P will probably fail GW-b — **major** (B6, GW)

- **Stage A:** 200 items × 16 level tokens = **3,200 supervised tokens** for M; 250 × 16 = 4,000 for P. Context is ≈85–90% of the "dose".
- **Stage B:** ≈20 × 2–4k assistant tokens ≈ 40–80k supervised tokens, over 1–2 optimiser steps (L-07).
- About 12 effective steps at a mean learning rate of ≈5e-5 is very little ranking change for a 31B model. GW-a requires all three arms to show ΔAP ≥ +0.05 *and* a CI excluding 0: a conjunction, ≈0.5 even at 80% each.
- GW-b failing makes H1, H2b and H3b "not evaluable". So the likely end state is row 2 of §1.3, a dose finding.
- On "can 0.5M change behaviour without hurting retention": **retention will almost certainly be fine** (KL ≪ 0.03 at this dose), but the behaviour change is likely too small to measure. The study is safe and uninformative.
- The dose rule ("scale until GW-b passes") cannot be executed within the budget.

**Fix.**
- Items are CPU-generated and short. Raise Stage A to **≥1,000–1,500 items per arm (≥16–24k labels)**, using 2 epochs or more pools. The cost is training tokens only: ≈2–3M tokens per adapter, about +1.5–3 h per arm-seed at central rates (§6).
- Fund this by dropping Stage B and the RFT pool (L-07), which saves ≈5–9 h.

---

## 6. Budget recount and the ladder

### L-23 Unbudgeted lines, and the pessimistic reserve is cut in the very case it is needed — **major** (B20)

**Missing lines** (INFERRED):

| Missing line | Opt. | Central | Pess. | Why |
|---|---|---|---|---|
| Hyperparameter freeze on a 4th throwaway family (§2.5): ≥2–3 configs × (train + 240-item eval) | 0.8 | 1.3 | 2.5 | Required by v2; no line |
| (d′) bug-named rollouts for G2(b), ≈45 | 0.5 | 0.8 | 1.1 | Not inside L2's 90 |
| G0 fallback (8k re-probe, prefix-fit trimming) if head_dim-512 backward fails | 0 | 0.3 | 1.0 | Probe history v1–v8 |
| SWE-V pools if the gold-file pool exceeds 2k tokens (likely; real files are 10–30k tokens) | 0.2 | 0.5 | 1.0 | L4 assumes ≈5k tokens per external item |
| **Additions** | **1.5** | **2.9** | **5.6** | |

**Totals:**

| | Opt. | Central | Pess. |
|---|---|---|---|
| v2 core total | 15.2 | 23.6 | 40.6 |
| + additions (with the column's reserve %) | +1.7 | +3.3 | +6.7 |
| **Recount** | **≈17** | **≈27** | **≈47** |
| Pessimistic after D1–D7 (saves 12.0 h before reserve) | | | 33.8 + 5.6 − 12.0 = 27.4 → **≈30 at 10%**, **≈33 at 20%** |

- v2's ladder lands at 24.0 only by **cutting the reserve from 20% to 10%** in the pessimistic column. That is the scenario with the most overrun risk, given the probe history of 8 failed versions.
- **Weekly fit** (B20: ≈10 h per week for this study): pessimistic week 3 is L5 8.0 + L6 7.5 = **15.5 h**, and ≈13 h after D7, which exceeds the cap. The HQ log shows week 1 already at 24.5/30 h used before Oct 10, and dev-resolve runs cost ≈6.5 h each (BM, VERIFIED), so the main track competes for the same quota.
- The next ladder step after D7 would cut seeds, which is core.

### L-24 The ladder silently converts the method into CAD-on-items — **major** (novelty)

- D6 (skip R2) is the largest pessimistic saving (−5.6 h), and the pessimistic case is triggered by slow rollouts, which is exactly when D6 fires. Below 25 rollouts/h, G1 drops Stage B entirely.
- Even at central rates, Stage B is ≈1 step (L-07).
- In every realistic branch, the deliverable is therefore "null-twin paired item SFT". Its closest priors are CAD and paired CAD (v2's own table, closeness 2). The "agent RL/RFT" part of the novelty delta does not survive.
- v2 discloses D6 ("the method then rests on the item stage") but never revises the [CLAIMED] sentence, which names RFT and policy gradient.

**Fix.** Make items the declared method from the start. Turn the rollout budget into (a) more item dose, (b) a Mix arm, (c) 5 seeds. Keep a small agent-level twin check (A) as descriptive.

**Re-budget sketch for v3** (central, INFERRED):

| Line | GPU-h |
|---|---|
| L1 probe | 1.5 |
| R1 = 90 rollouts for G2 descriptive and the A baseline | 1.3 |
| (d′) | 0.8 |
| L4 base evaluation | 1.5 |
| Training: arms {M, U, Mix} × 5 seeds + N × 2 + P × 3 = 20 adapters × 1.5M tokens (Stage A ×3 dose) | 30M tok ÷ 500 tok/s ≈ 16.7 h ✗ too much |
| → instead 20 adapters × 0.9M tokens | ≈10.0 + 0.5 loads |
| L6 (20 adapters × real + synthetic) | ≈6.5 |
| A check | 2.3 |
| Hyperparameter freeze | 1.3 |
| Gate re-runs | 1.0 |
| **Subtotal** | **≈26.7**, plus 15% ≈ **30.7** |

This is over 25. The binding trade is **seeds × arms × dose**. The cheapest honest option is to drop N as a training arm (H3 then uses U vs Mix vs far-null *evaluation-only* contrasts), or to drop P to 2 seeds. Shrey must choose. v2 avoids the choice by keeping the dose near-homeopathic.

---

## 7. Generality

### L-25 Online GRPO transfer holds only with pair-pooled, un-normalised advantages — **major** (§2.7, §6)

- See L-05(b). Per-group std normalisation breaks the "never prefers one blind guess" property whenever p_X ≠ p_X̄, which B3 expects.
- Zero-variance groups are harmless *in expectation* for mean baselines, but with std normalisation they are usually dropped. That removes the negative half of the cancellation for the twin whose group is all-fail: at K = 8 and p = 0.1, P(all fail) = 0.9⁸ ≈ 0.43.
- The fitting shims of L-03 are amplified fastest under online RL, because they succeed in both halves of every pair.

**Fix.** Specify Dr.GRPO-style advantages with a pair-pooled mean over 2K samples and no std. Add per-node hidden contracts (L-03) before any online run.

### L-26 Twins require content–wiring independence, which real graphs violate — **major** (generality; the K-47 rename)

- B2 holds in the synthetic worlds only because names and bodies carry no information about edges: names are random and bodies are family templates.
- In real code, and in knowledge graphs, citation graphs and workflow DAGs, node content *predicts* edges. A rewired twin is then implausible (K-34's corruption-detection issue), so twins are distinguishable from content and the cancellation premise fails.
- So the trick transfers only to domains where content is uninformative about wiring, which are the domains where shortcut-by-content is not a problem anyway.
- Worse, training on content-independent wiring teaches "names do not predict calls". That is v1's K-47 negative-transfer concern, renamed: the 30% name–*template* correlation does not address the name–*edge* correlation that real items reward.
- For non-code graphs without an executable symptom (static text graphs, KG QA), "symptom-matched twins" reduce to ordinary counterfactual or contrast-set augmentation. That is prior art: Gardner et al. contrast sets, and the GRAIN-style structure-invariance work noted in HQ (Sep notes) and `break_v2_priorart.md`.

**Fix.**
- Scope the generality claim to "executable DAGs where content is (made) uninformative about wiring".
- Measure negative name transfer: H2b on the lexical-name-informative stratum of real items, with the M − U prediction stated as ≤ 0.
- Keep generality as a conjecture. L11 (12B replication) tests *model* generality, not *domain* generality.

### L-27 Small things — **minor**

- **Constant normaliser L_max = 16,384** with only 16 supervised tokens per Stage-A item: per-sequence gradients are ≈1,000× smaller than Stage B's. Adam is scale-invariant to first order, but ε = 1e-8 is no longer negligible at LoRA-B grad scales ≈1e-7. Use per-stage normalisers or set ε = 1e-10.
- **"Equal item counts" vs "the same pools, G version only"** (§2.5): U needs twice as many pools as M. State that U and N draw extra pools (or that M uses half as many pools), and match pool difficulty.
- **Bootstrap-resampled seeds** duplicate ≈37% of pools per seed within one epoch. That is fine, but report the unique-pool count per seed.
- **SWE-V edit-set items:** a pool of "gold-patch files plus same-file distractors" is mostly within-file localisation (B15). Report the number of files per item.

---

## 8. Were v1's breaks fixed, renamed, or made worse?

| v1 break | v2 claims | Verdict | Why |
|---|---|---|---|
| K-04 contract bisection (fatal) | Fixed (d_obs ≥2, symptom match) | **Renamed** | Bisection now gives "uncontracted node ≥2 below the deepest failing contract in an imported module". That generator fingerprint is shared by v and v′, so twins do not cancel it (L-01) |
| K-12 template anomaly (fatal) | Fixed (in-family instances) | **Fixed, but caused new breaks** | Repair becomes numeric inversion (L-10), and fitting shims become the dominant strategy (L-03) |
| K-17 alternative fixes | Fixed (binary test reward) | **Worse** | Accepting any passing fix now accepts wrong-node ancestor shims that solve both twins (L-03) |
| K-18 hardcoding issue values | Fixed (fresh inputs, no literals) | **Renamed** | Fitted re-implementation is hardcoding by another route |
| K-20 coverage changes | Fixed (symptom match) | **Partial** | Only the bug's failing set is matched; other nodes' coverage differs, spectrum is reported only |
| K-21 locality | Fixed (block-preserving) | **Fixed at a cost** | Together with the family and arity constraints, almost no legal swaps (L-11) |
| K-22 motifs | Disclosed | OK | But L-11 makes the motif report trivially "preserved" |
| K-23 KS filter | Fixed by construction | **Fixed** | Level-preservation proof is correct |
| K-30/31/33/37 reward | Removed from RL | **Fixed** (by deletion) | |
| K-41 format transfer | Guided scaffold, P anchor | **Mostly fixed** | Calibration is not equal by construction (B7). Measure ECE per arm |
| K-47 "ignore names" | Fixed (weak name–template correlation) | **Renamed** | Name–*edge* independence is still taught (L-26) |
| K-50 pseudo-replication | 3 data-resampled seeds | **Partial** | The bootstrap ignores the seed df (L-17) |
| K-08 depth slope driven by budget | Token adjustment + competing risk | **Worse** | Mediator/collider adjustment (L-18) |
| BM top-1: no positive control / dose | P arm, dose 0.5M | **Renamed** | P will likely *reveal* the dose problem (GW-b fails) rather than fix it. "0.5M tokens" is ≈3k labels (L-22) |
| BM top-2: items carry the signal, differ by arm | Identical pools across arms | **Partial** | Depth parity is fixed, but the evaluation pools null halves (L-16). BM recommended removing items from the main arms; v2 made items *the* signal, which is now the honest method (L-24) |
| BM-§6 Stage B ≈ 0 | Single-step RFT kept | **Not fixed** | 1–2 optimiser steps (L-07), and still named in the claim |
| BM-§2.2 H3 TOST | Conditional on GW, SESOI from P | **Partial** | The H3 instrument is blind if the null is near-identity (L-11, L-20) |
| BM-§4.2 near-SOTA | Retention battery | **Measured but unpowered** | L-21 |
| BM-§1.9 budget | Recomputed | **Mostly fixed** | Unbudgeted hyperparameter freeze and (d′); reserve cut in pessimistic (L-23) |

---

## 9. Assumption → break map

| Assumption | Breaks | Status |
|---|---|---|
| B1 twin novelty | L-01, L-02, L-24, L-26 | The mechanism reduces to CAD in expectation. Novelty survives only as "graph-null-model minimal pairs for code reachability, paired vs unpaired" plus the null-strength analysis |
| B2 indistinguishability | L-03, L-08, L-12 (relaxed), L-14, L-26 | Violated at turn 0 (numbers), voided by the relaxed fallback, and bypassed by twin-agnostic repairs |
| B3 q > ½ | L-04, L-05, L-10 | The ½ threshold is right only against the v/v′ commit, and RFT freezes base bias anyway |
| B4 yield ≥5% | L-11, L-12, L-14 | The wrong yield is measured. The joint yield with ≥4 path-complete discordant elements is the binding one |
| B5 base success 0.15–0.6 | L-10 | Likely far lower |
| B6 dose | L-07, L-22 | ≈3k labels and ≈12 steps |
| B8, B9 seeds and ICC | L-17 | σ_seed likely 0.02–0.04; CIs anti-conservative |
| B16 bf16 API | L-21 | Conflicts with HQ 30 Sep decision |
| B17 H3 generalises | L-11, L-20 | The instrument is blind without a far null |
| B19 relevance to the deployed agent | L-07, L-09 | The agent-side signal has no dose and no powered readout |
| B20 quota | L-23 | Pessimistic weeks exceed 10 h |

---

## 10. Verdict

### Top 5 breaks

1. **L-02 (+ L-01): the cancellation is vacuous in expectation.**
   - Bugs are synthetic and v, v′ are exchangeable, so pairing equals an unpaired real+null mixture. M − U in blind credit is ≤ ½·Δ_H3, which v2 predicts ≈ 0.
   - Twins cancel only v-vs-v′ discrimination. They keep full credit for every feature the matcher forces v and v′ to share.
   - **Fatal for the [CLAIMED] mechanism.**
2. **L-03: twin-agnostic repairs (fitted ancestor shims) solve both halves.**
   - The hidden tests do not pin the bug node's behaviour, so the best "shortcut" earns *more* credit under M than reading.
   - **Fatal for the agent/RFT half.**
3. **L-11 + L-12: the null barely differs from the real graph, and twin supply is tiny.**
   - Family + level + module + arity cells hold ≈0.06 edges each, so ≲6% of edges are rewirable.
   - Exact matching yields in-degree-1, last-hop twins; O2 twins ≈ 0; the relaxed fallback leaks the answer.
   - H3 "equivalence" is an artefact. **Major → fatal.**
4. **L-16 + L-17 + L-21: nothing headline is powered or fairly tested.**
   - H2a and H4 are biased by evaluation pooled over null halves; H4 compares M against an amplifying U (L-04).
   - H1 ≈ 0.3–0.7 power, H2b ≈ 0.08–0.18, full chain ≈ 0.02–0.08.
   - Retention non-inferiority passes ≈ 0.03–0.2 even under perfect retention.
   - **Major.**
5. **L-07 + L-22 + L-18 + L-24: dose and gates.**
   - ≈3k supervised labels and ≈12 steps (10 warm-up); Stage B is 1–2 steps.
   - G2 stops the study with P ≈ 0.7–0.85, with no publishable G2 row.
   - Every realistic ladder branch turns the method into CAD-on-items while the claim still says RFT.
   - **Major.**

### Is v2 salvageable?

**Yes, as a narrower, honest study. Not as written.** What survives:
- the exact level-preserving rewire;
- symptom-matched twin *items*;
- the positive control;
- the served-stack evaluation discipline;
- the null-model question.

The claim must change from "twins cancel shortcut credit in agent RL" to:

> Paired null-twin minimal pairs vs the same data unpaired, for call-reachability reasoning on code, plus a **null-strength dose–response** answering "how much real topology matters".

That analysis framing is the part `break_v1_priorart` found unclaimed. It is also the part the twin pool can actually power on synthetic items.

### Minimal v3 changes (in priority order)

1. **Add arm Mix** (unpaired 50/50 real/null, equal dose and distinct sites). Redefine H2a as M vs Mix. Report M vs U on real halves and M vs N on null halves separately (L-02, L-16). Rewrite §2.3 as a finite-sample pairing argument, and correct the "½ whichever node" error (L-01).
2. **Hidden per-node contracts** (fresh inputs) on every node in c's subtree, including the bug node. Add ancestor fit-shims and compensating edits to C5's alternative library for all operators. Audit an observation-free *fitting* solver (L-03). Expose a per-family spec so that localisation, not constant inversion, is the bottleneck (L-10).
3. **Rewire fix.** Drop the family constraint (common value domain, arity only). Pre-register a minimum rewired-edge fraction. Add a **far null** as the H3 instrument control and report H3 as a dose–response over null strength (L-11, L-20).
4. **Twin supply on CPU, day 1.** Kill criterion on the *joint* yield (C5-passing pairs with ≥4 discordant elements, path-complete in both wirings). Equal d_obs within pairs. Drop O2 twins and the relaxed fallback. Report the hop-position of discordant edges (L-06, L-08, L-12, L-14).
5. **Declare the method as item SFT.** Drop Stage B from the claim (keep A as descriptive). Move the rollout savings into item dose: ≥1,000 items and ≥16k labels per arm, batch 4–8, warm-up ≤10% (L-07, L-22, L-24).
6. **Statistics.**
   - Seed as the unit with Satterthwaite or Kenward–Roger df; 5 seeds for M, U and Mix (fund this by dropping N training or P's third seed).
   - Chain H1 → H2a(vs Mix) → H2b; H4 signed, against base and Mix, under Holm.
   - Retention primary = exact KL + per-item gold log-prob margins; accuracies descriptive (L-17, L-19, L-21).
7. **Gates.** G2 becomes descriptive with a pre-registered pivot (no stop) and no token adjustment. Add a G2-fail row to §1.3. Budget the (d′) rollouts and the hyperparameter-freeze line. Keep a 20% pessimistic reserve. Re-fit the week-3 load under 10 h (L-18, L-23).
8. **Scope generality** to executable DAGs with content made uninformative about wiring. State the GRPO drop-in with pair-pooled, std-free advantages. Measure negative name transfer on real items (L-25, L-26).
