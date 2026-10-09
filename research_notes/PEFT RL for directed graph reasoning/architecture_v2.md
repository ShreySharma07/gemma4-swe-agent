# Architecture v2: GRAFT-2, training on symptom-matched null twins (8 Oct 2026)

**Role:** architect. This rebuilds v1 after three independent breaks:
- `break_v1_logic.md`: breaks K-01 to K-51, fixes F1–F11;
- `break_v1_method.md`: cited below as **BM-§x.y**;
- `break_v1_priorart.md`: cited below as **BP-n**.

All three were read in full, together with `architecture_v1.md`, `phase3_critic.md`, `phase3_redteam.md`, `docs/training_feasibility_4xl4.md` and the relevant `docs/HQ.md` rows.

**Labels** (as before):
- **VERIFIED**: read in a primary source, or computed from repo files by an earlier pass. Inherited labels are kept.
- **SNIPPET**: abstract or search result only.
- **INFERRED**: my own reasoning, or a citation from memory.
- **GUESSED**: a number used for sizing only. A gate measures it before it is relied on.

**Tags:**
- **[KNOWN]**: prior art, cited.
- **[NARROWED]**: a known mechanism; only the stated use is claimed.
- **[CLAIMED]**: the single piece for which no same-mechanism prior was found.

---

## 0. Resolving the tension: method *and* null-model experiment

The user wants a **method**: a PEFT + RL-style recipe that improves reasoning over large directed graphs, keeps a quantised model near SOTA, and generalises. The breakers push towards an **analysis**: does real software topology matter as training data, tested against a degree-preserving null model?

Breaker 3 found that only the analysis framing is unclaimed (BP-4). The synthetic compiler (FuncBenchGen, DynaCode, SWM, PhantomEnvs) and the swap operator (muJava AMR/MMR, MutDafny MCR, higher-order mutants) both have closeness-2 priors.

v2 resolves the tension by making the null model **the training signal itself**, not only a control arm.

> **The single novel piece [CLAIMED].** Every training task is issued together with a *symptom-matched null twin*. The twin has the same issue text, the same visible failing tests, the same function bodies, names, modules, levels and per-type degrees. Its call graph is a level-, block- and degree-preserving rewire of the real traced topology, and its correct answer is a different node that only the rewired edges determine. Under balanced presentation of both twins, the expected outcome credit (RFT or policy gradient) for any decision taken before reading the wiring therefore cancels exactly, leaving credit only for behaviour that reads edges. The same twin pool also supplies the real-vs-null training contrast.

**Prior-art deltas for that sentence:**

| Prior | Closeness | What it does | What v2 adds |
|---|---|---|---|
| Counterfactually augmented data, CAD (Kaushik, Hovy, Lipton, ICLR 2020/2021, SNIPPET) and paired-CAD training (ACL 2024, 2406.06633, SNIPPET) | 2, mechanism family: minimal pairs remove spurious label correlations | Pairs are human-edited text labels | Pairs come from a **graph null model**, so the nuisance features are *defined* to be the null model's preserved statistics. Pairs are **executable** and **symptom-matched**. Use is in **agent RL / RFT** with a stated cancellation condition (§2.3). No CAD-for-graph or CAD-for-agent-RL prior was found (searches S1–S3). |
| CNY (2608.29588, VERIFIED) and Huang et al. TMLR'24 (2309.16595, VERIFIED) | 2 for measurement | Degree-preserving or random rewire used as an *evaluation* control | Here the rewire generates *training data* and the credit structure. |
| FuncBenchGen 2509.26553, DynaCode 2503.10452, SWM 2610.04940, PhantomEnvs 2609.40221 (all VERIFIED, BP-1) | 2 | Synthetic executable graph worlds with random or hand-set topology | Topology is **copied from real traces and paired with its own null**. |
| Invariance-across-environments ideas: IRM, domain randomisation, PAIRED environment design (INFERRED / SNIPPET) | 1.5 | Train across environments that differ in nuisance factors | Here each environment pair is constructed to differ *only* in wiring, with the answer flipping. |

The twin mechanism has **not yet** had a dedicated red-team pass. Assumption B1 (§11) flags it.

**The experiment, which reports both outcomes.** One shared, base-generated twin pool supports three training arms that differ *only in which twin halves they train on*:
- **M**: both halves, balanced (the method);
- **U**: real-topology halves only (the standard synthetic curriculum);
- **N**: null-topology halves only (the null-model arm).

A real-code positive control **P** anchors dose and content. Every outcome is reportable (§1.3).

---

## 1. Claims, hypotheses, predicted effect sizes

### 1.1 Gate GW: did training take? Everything downstream depends on it.

| ID | Test | Pass rule | Predicted (GUESSED) |
|---|---|---|---|
| GW-a | Each of M, U, N vs base on **held-out synthetic twin items** (3 held-out source families; both twins pooled) | Seed-mean ΔAP ≥ +0.05 and the 95% CI excludes 0, for all three arms | base 0.55 → arms ≈0.70 |
| GW-b | P vs base on **held-out real items** (S_eval) | ΔAP ≥ +0.03, CI excludes 0 | +0.06 |

**If GW-a fails:** training did not take. All hypotheses are *not evaluable*. Stop and report.

**If GW-b fails:** this dose cannot move the real metric even with real content. H1, H2b and H3b are reported as "dose-limited, not evaluable as content or topology effects".

### 1.2 Hypotheses

They are tested in a fixed gatekeeping order, each at α = 0.05; testing stops at the first failure (BM-§2.3). Effect sizes are GUESSED.

| Order | ID | Claim | Metric | Predicted | Test |
|---|---|---|---|---|---|
| 1 | **H4 (mechanism)** | Twin training removes observation-independent answering | **Twin-invariance rate**: on held-out twin items, the share of *discordant* elements where the model's continuous score differs by <0.1 between the two twins (i.e. it ignores the wiring) | U 0.55, N 0.55, **M 0.30** (M − U ≤ −0.15) | Paired hierarchical bootstrap over seeds × families × items. Pass if CI upper < −0.05 |
| 2 | **H2a (in-distribution method gain)** | M beats U on the elements only wiring can decide | **Discordant-element AP** on held-out twin items | M − U = **+0.12** | As above. Pass if CI lower > +0.03 |
| 3 | **H1 (real transfer vs base)** | M improves real call-set prediction over the deployed base | AP on 320 frozen real items, real context | **+0.04** (includes format; P anchors it) | Wild-cluster bootstrap-t over repos (Webb weights) × seed resampling. Pass if CI lower > 0 |
| 4 | **H2b (real transfer vs standard synthetic)** | M ≥ U on real items, concentrated where the wiring matters | AP on real items: overall, and in the depth ≥2 × static-reachable-distractor stratum | **+0.02** overall; +0.04 in the stratum | Same. **Expected to be underpowered:** the MDE is ≈0.035–0.05 (§8). Reported as superiority if the CI lower bound is >0, else as an interval |
| — | **H3 (null-model test, two-sided, outside the chain)** | Does real topology carry training signal beyond {per-type degree, level, module block}? | (a) **Crossover interaction** on held-out synthetic items, train source {U, N} × test source {real twin, null twin}. (b) AP_real(U) − AP_real(N) | (a) ≈0.01; (b) ≈0.00 | (a) TOST with margin 0.03 (powered: paired, synthetic). (b) TOST with SESOI δ = max(0.02, ½·(AP_real(P) − AP_real(base))), plus two-sided superiority. **Conditional on GW** (BM-§2.2) |
| — | **R (retention, non-inferiority)** | Near-SOTA capability kept | MMLU-Pro slice Δacc ≥ −2 pp; HumanEval+ Δpass@1 ≥ −3 pp; tool-call validity Δ ≥ −1 pp; exact per-token KL(π_LoRA ‖ π_base) on held-out base agent trajectories: mean ≤0.03 nats, p99 ≤0.5 | Δ ≈ 0; KL ≈0.01 | One-sided 95% CIs per adapter (M seed 1 required; U and P if budget allows) |
| — | **E (external, descriptive)** | Where the model sits | SWE-bench-Verified issue→edit-set AP; TraceEval slice AP. Reference points: bf16 Gemma-4-31B and a frontier model on the same items | M ≥ base +0.01–0.02; W4A16 gap to bf16 ≤0.03 | Point estimates with CIs; no inferential claim |
| — | **A (agent level, descriptive with CI)** | The RFT stage changes behaviour on twins | Held-out twin pairs: share of pairs where **both** twins are solved (hidden fresh-input F2P), base vs M vs U, deployed pipeline | base 0.10, U 0.15, M 0.22 | Paired, n = 20 pairs. Descriptive |

Exploratory analyses, controlled with BH-FDR within each family:
- depth, edge-type and direction strata;
- the GF and plausible-RW contexts;
- normalised Δ̂_struct;
- IDF-filtered AP;
- base-rate variation (25 / 33 / 50%);
- the Stage-A-only snapshot;
- leave-one-source-graph-out.

### 1.3 What each outcome means (every row is publishable)

| GW | H4 / H2a | H1 / H2b | H3 | Reading |
|---|---|---|---|---|
| fails | — | — | — | "At ≈0.5M tokens, LoRA on a W4A16 31B does not learn even in distribution." A dose finding; stop |
| a ✓, b ✗ | any | not evaluable | not evaluable | "Synthetic training takes in distribution, but the dose cannot move real call-set prediction even with real code." This bounds every synthetic-transfer claim at this scale |
| ✓ | ✓ / ✓ | ✓ / ✓ | any | **Method works.** Null twins remove wiring-independent shortcuts and the gain transfers |
| ✓ | ✓ / ✓ | ✓ / ✗ or inconclusive | any | Twins cancel shortcuts in distribution, but **those shortcuts are not what limits real transfer**. A mechanism finding |
| ✓ | ✗ | — | any | **Mechanism refuted:** balanced twins do not change what the model learns at this dose. Twin-invariance is reported as a new diagnostic |
| ✓ | any | any | (a) equivalent, (b) equivalent | **Null-model statistics suffice.** Synthetic training worlds need only match {degree, level, block}. This is a generator-design result |
| ✓ | any | any | U > N | **Real software topology carries training signal beyond those statistics** |
| ✓ | any | any | N > U | Null rewiring adds useful diversity. Real topology's regularities are themselves shortcuts |
| ✓ | any | P ≫ M, U, N on real | any | Content dominates topology at this dose, quantified against an anchored positive control |

---

## 2. Architecture

### 2.1 Component diagram

```
CPU ─────────────────────────────────────────────────────────────────────────────────────────────────────
 S_topo (6–8 permissive non-dev repos)            S_eval (16–20 other non-dev repos)    S_ctrl (= S_topo code)
   │ C1 tracer: setprofile+threading, code-obj id,      │ C9a real item builder:           │ C9p positive-control
   │ wrapper collapse, unique-resolution lexical typing │ path-complete pools, mocks out,  │ items (real bodies,
   ▼                                                    │ hub cap, static-reach labels,    │ trace labels, same
 C2 source hygiene (F11): integration roots only,      │ truncation marks, FREEZE+HASH    │ format & pool size)
   task-wise growth, SCC cap/bounded recursion,         ▼                                  ▼
   effect-only edge share, dead code, hub-bug cap     real items ×{real, GF, plausible-RW}   P items
   │
   ▼
 C3 TWIN GENERATOR [CLAIMED]: G (real-copied) ─► ρG = level- & block- & per-type-degree-preserving
   swap of lexical+callback edges (exact by construction; no KS filter); names/bodies/modules identical
   │
   ▼
 C4 compiler [KNOWN pattern: FuncBenchGen/DynaCode/SWM] + in-distribution templates (F4), hard dark edges (K-07),
   contracts placed first, bugs placed at d_obs ≥ 2 (F3), symptom matching: FailVis_G(B) = FailVis_ρG(B′), B ≠ B′
   │   bug ops: O1 in-family body instance, O3 2-node contract, O2 = degree-preserving 2nd-order MCR (F5)
   ▼
 C5 verifier: gold passes; hidden F2P on FRESH random inputs (F2); necessity incl. alternative single-site fixes;
   determinism; suite ≤10 s; |P2P| ≥ 5; shortcut audits (observation-free solver, body-only discriminator,
   single-use-import AUC, spectrum solver rate)
   │ twin pairs (X, X̄) + twin items (same test & pool, two wirings)
GPU (4×L4) ──────────▼────────────────────────────────────────────────────────────────────────────────
 C6 ROLLOUTS: deployed pipeline (`swegemma eval`, V5 SequentialAgent localizer→fixer), deployed sampling
   T=0.2/top-p 0.95/top-k 40, base model only, K=1–2; ONE shared pool for all arms
   │ binary verified outcome (fresh-input F2P ∧ P2P ∧ no issue literals)
   ▼
 C7 ARM SELECTOR (CPU): M = both halves of pairs; U = X halves; N = X̄ halves; P = real items;
   stratified matching on (d_obs × operator × size) cells; equal items, tokens, steps
   ▼
 C8 TRAINER: FSDP + bnb NF4 (quant_storage=bf16) from dequantised W4A16 [documented path];
   Stage A item SFT (warm start) → Stage B RFT on verified trajectories; ≥3 seeds (data-resampled)
   ▼
 C9b EVAL (served stack: vLLM W4A16 + LoRA): guided-scaffold readout, continuous level score;
   real items, synthetic twins, retention battery, KL drift, SWE-V edit-set, TraceEval slice
 C10 agent-level twin check (deployed pipeline);  C11 dev resolve (conditional, descriptive)
API (0 GPU-h) ── bf16 Gemma-4-31B and frontier reference on the same items (verbalised readout)
```

### 2.2 The twin generator (C3) and symptom-matched tasks (C4)

**Source graph G** (after C2 hygiene, applying F11, K-01 to K-03, K-06 and K-10):

| Rule | Detail |
|---|---|
| Roots | Only integration tests, i.e. tests whose E(t) spans ≥3 modules. Unit-test roots are deleted before levels are computed (K-01) |
| Growth | By task: all descendants of a sampled integration root, capped at 200–1,000 nodes (K-03) |
| SCCs | Drop the source graph if its largest SCC exceeds 5% of \|V\|. Otherwise compile each SCC as bounded recursion with a `fuel` argument. Report \|largest SCC\|/\|V\| and the condensed degree (K-02) |
| Effect-only edges | A share of edges discard the return value, matched to C1's per-repo estimate of unused call results (K-06) |
| Hub bugs | A node covered by more than 30% of root tests cannot host a bug; \|P2P\| ≥ 5 (K-06) |
| Dead code | 10–20% of nodes are unreachable, and 10% are reachable only from non-F2P tests, matched to S_eval's distractor composition (K-10) |
| Depth histogram | Reported on CPU before any GPU spend |

**Null twin ρG** (applying F6 and K-20 to K-24). Directed double swaps (a→b, c→d) ⇒ (a→d, c→b) are proposed **only** when all of the following hold:
- both edges are of the same type, and that type is **lexical or callback**. Dispatch, decorator and registry edges are never rewired (K-24);
- `level(b) = level(d)` and `level(a) = level(c)`, using longest-path layering. Every edge then still points to a deeper layer, so **acyclicity and every node's level are exact by construction**. There is no KS filter and no selection of source graphs (K-23);
- `module(b) = module(d)`. Every import statement is then identical across twins (K-11, K-21), and the (caller-module, callee-module) edge counts are preserved;
- b and d come from the same template family and have the same arity (K-15, K-19).

Run 10·\|E\| proposals. Report, per twin family:
- modularity Q and the intra-file edge fraction (should be identical by construction);
- triangle, diamond and feed-forward-loop counts;
- the per-node distribution of the number of test roots reaching it;
- \|anc\| and \|desc\| distributions;
- class count and MRO depth (should be identical);
- per-arm C5 drop rates.

H3 is stated as "fidelity beyond {per-type degree, level, module block}". Motif differences are reported, not hidden (K-22).

**Names.** Identifiers come from the S_topo vocabulary. They are assigned to node *positions* independently of topology, with the **same seed in both twins**, so names never encode wiring (BM-§3.3). The verb carries a weak, fixed template-family correlation (e.g. `sum_`/`fold_` for fold templates with 30% probability), which is identical in all arms (K-47).

**Bodies.** About 40 template *families*. Each instance draws its constants and operators at random from its family, so a buggy body is another valid family instance (F4, K-12). All templates are total, never raising (K-15). Calls are positional (K-16). Local names are drawn independently of callees (K-16). No docstrings (K-16). Non-injective templates (mod, min/max, len, dict.get default) are banned on bug→observer paths, or tagged and stratified (K-05).

**Dark edges** (K-07):
- dispatch through ≥3 sibling overrides, chosen by a constructor argument set elsewhere in the graph;
- computed registry keys (`REG[prefix + kind]`);
- `getattr(mod, name_str)`;
- callbacks.

The grep-distance of each dark edge is reported. Dark edges are identical in both twins.

**Symptom-matched bug placement** (the core of the twin):
1. Sample the visible unit-contract set *first*: 40% of internal nodes.
2. Choose a bug instance B at node(s) v in G and B′ at v′ ≠ v in ρG, with the same operator and family. Require:
   - `level(v) = level(v′)` and `module(v) = module(v′)`;
   - **identical visible failing-test sets**: FailVis_G(B) = FailVis_ρG(B′);
   - d_obs ≥ 2 in both, where d_obs counts hops from the deepest failing visible contract to the bug (F3, K-04).

   The issue text then names the same root test and the same deepest failing contract. The two twins differ only in the expected-vs-actual numbers, the wiring below that contract, and which body is the buggy family instance.
3. Twin X is (G, B) and X̄ is (ρG, B′).
4. The pair is kept only if both halves pass C5. The match yield is reported; it is GUESSED at 10–30% of candidate nodes. If the yield is below 5%, relax to equal *failing root tests* plus equal failing-contract count with Jaccard ≥0.8 on contract names, and disclose.

**DPCS, renamed honestly** as a **degree-preserving second-order MCR mutant** [NARROWED; cites muJava AMR/MMR, MutDafny MCR (2511.15403), DeepMutants (2107.06657), and Jia & Harman HOM]. It is operator O2, used for 25% of tasks, with these constraints (F5, K-13 to K-17):
- a ≠ c, in different files, graph distance ≥2;
- lexical or callback edges only;
- module(b) = module(d);
- same-family callees;
- in-degree(b) ≥ 2 and in-degree(d) ≥ 2, so that a fix that only edits a definition breaks P2P.

On CPU, a library of single-site alternative fixes is enumerated: revert at a, revert at c, exchange the bodies of b and d, rename, shim. The task is kept only if **no** single-site alternative passes. Item-level gold is "the union over minimal passing edit sets". The agent reward is test-defined, so it accepts any passing fix (K-17).

The other operators:
- **O1**: an in-family body instance, 55% of tasks.
- **O3**: a two-node contract change, 20% of tasks.

**Hidden tests and reward hygiene (F2, K-18).**
- Hidden F2P runs on **fresh random inputs**, with expected values from the gold program. These inputs never appear in the issue.
- Success is **binary**: all hidden F2P pass, P2P is intact, and a static check finds no literal copied from the issue in the patch.
- Suite time ≤10 s (BM-§1.1).

**Shortcut audits.** All are CPU, except (ii), which needs ≈0.05 GPU-h of prefill inside L1/L4. Each must hold **before** any arm is trained (F3, F4, K-11):
- (i) An observation-free solver uses the deepest failing contract plus level, degree and module priors plus body surface, and never reads call sites or coverage. Its pair accuracy must be ≤0.55; by construction it is ≤0.5 plus ties.
- (ii) A body-only discriminator: base-model AUC for "is this body mutated?" on 200 single bodies must be in [0.45, 0.55].
- (iii) A single-use-import heuristic must have AUC ≈0.5 for locating O2 sites.
- (iv) A spectrum solver (Ochiai over executed nodes) has its pair accuracy *reported*. This is a legitimate structure-using strategy, not a shortcut.

**Twin items** (training and held-out synthetic evaluation):
- the test is an integration root;
- the pool has 16 candidates, at most 15-line bodies, ≈1.8k tokens;
- **≥4 discordant elements** (CALLED in one twin, NOT in the other), with the rest concordant;
- the pool is identical across twins; only the bodies at swapped call sites and the labels differ;
- labels are exact from compiled-program traces.

### 2.3 Why twins cancel shortcut credit (the method's stated condition; proof sketch, INFERRED)

Consider a state s whose history carries no information about which twin it is in. That holds before any wiring-dependent observation, because the issue, the visible test output, names, modules, levels and degrees are identical, and bodies are family-indistinguishable (audit ii). Under balanced presentation, the expected policy-gradient direction at s is:

`∇_θ Σ_a π_θ(a|s) · ½[Q_X(s,a) + Q_X̄(s,a)]`

- A wiring-blind **commit** action ("edit v") solves exactly one twin, because v ≠ v′. Its twin-averaged value is ½, whichever node it names. Any preference among blind commits learned from topology statistics therefore cancels.
- A **read-the-wiring** action that leads to success with probability q in each twin has value q.
- So the learning signal at s favours acquiring wiring evidence **iff q > ½**, and it never rewards a blind guess more than another blind guess.

Under U, by contrast, a blind commit is worth P(gold = v | statistics). Real-topology regularities, such as "the bug sits in the more central callee", can push that towards 1. Those are exactly the shortcuts U can learn and M cannot.

The same identity holds for RFT, since the success-filtered SFT gradient is Σ_a π Q ∇log π. It also holds per element for item cross-entropy: on a discordant element, any score predicted from statistics alone is rewarded in one twin and penalised in the other.

**Caveats** (listed as assumptions B1–B3):
- Twin-informative leaks other than wiring (file length, number formatting) break the premise. Audits (i)–(iii) bound them.
- When q < ½ early in training, twin balance pushes towards indifference rather than towards reading.
- The cancellation is in expectation. With about 20 trajectories it is noisy, which is one reason the item stage carries most of the signal.

### 2.4 Rollouts in the deployed pipeline (C6; BM-§7.1, BM-§1.1)

- Synthetic twin tasks are packaged as `tasks.jsonl` plus snapshots and run with the official `swegemma eval`, using the **deployed V5 SequentialAgent**. The localizer has run_command and graph tools; the fixer has the edit tools. Prompts are unchanged; v2 has **no CANDIDATES field** (§2.6).
- Graph files for synthetic repos are produced with the same builder as dev, applied to the degraded graph (undirected, merged labels), so the graph tools behave as they do in deployment.
- Sampling uses the deployed settings: **T = 0.2, top-p 0.95, top-k 40**. K = 1 for gate rollouts and K = 2 for the pool. At T = 0.2 the samples are near-duplicates; trajectories are deduplicated by patch hash. Task diversity, not K, is what matters for RFT.
- **Wall clock.** The deployed per-task cap is 4.5 min sequential. Parallel streams slow each stream's decode by a factor s, which the probe measures, so the training cap is 4.5·s min plus the deployed 100-tool-call cap. This keeps the step budget equivalent, and it is disclosed.
- **Pool.** One pool, generated by the **base** model:
  - R1: 45 pairs × 2 tasks × K=1 = 90 rollouts, which also serve as the gate;
  - R2: the same 45 pairs again, plus 30 new pairs × 2 × K=2;
  - 300 rollouts in total.
  All arms select from this pool, so the arms differ only in **selection** (BM-§3.1, §3.2).

### 2.5 Arms, selection and equalisation (C7)

| Arm | Stage A items | Stage B trajectories | Seeds |
|---|---|---|---|
| **M (method)** | Twin items: each pool shown in both its G and ρG versions, balanced | Successes from **both** halves of selected pairs, at their natural success counts | 3 |
| **U (standard synthetic)** | The same pools, G version only | Successes from X halves | 3 |
| **N (null topology)** | The same pools, ρG version only | Successes from X̄ halves | 3 |
| **P (positive control)** | **Real-code** items from S_topo repos (real bodies, trace labels, same format, pool size and path-completeness rules) | None; extra items are added so the token count matches | 3 (2 under ladder step D5) |
| base | — | — | — |

**Rules for every arm (K-45, K-46, BM-§3.2):**
- Equal item counts, equal total tokens (±3%) and equal optimiser steps.
- Stage-B trajectories are matched **per (d_obs bucket × operator × size) cell** between arms, by stratified subsampling.
- A balance table is pre-registered: standardised mean difference <0.1 on d_obs, trajectory length, operator mix, and \|V\| of the source world.
- No per-arm size recalibration, ever. Any generator change applies to all arms.
- Hyperparameters are frozen on a fourth, throwaway synthetic family before any arm is trained.
- M sees fewer distinct (graph, bug) sites than U or N at equal dose: P pairs × 2 wirings, against 2P sites. This works *against* M and is disclosed.
- **Seeds resample the data.** Each seed draws a bootstrap sample of pairs and pools, and a new LoRA initialisation and shuffle. Seeds are not init-only (BM-§2.2).

**Arm (d) from v1 is removed** (K-26 to K-28, BM-§3.4). Its "depth/hops matters" question is now prior-expected (PhantomEnvs, FuncBenchGen, DynaCode; BP-1, BP-4). The graph-free control moves to:
- the evaluation GF context (names and signatures only);
- the gate (d′): the **same twin graphs** with an issue that names the bug node, as the breaker proposed (K-27).

### 2.6 The set readout and the reward

**The policy-controlled set term is removed from the RL reward entirely.** This resolves K-30, K-31, K-33 and K-37 by construction: the RL reward is binary test success. The set objective lives only in **items with fixed, compiler-chosen pools**, where every candidate is scored and padding is impossible.

The deployed agent prompt is unchanged. That keeps the deployed pipeline and the dev comparison valid.

For anyone who wants the CANDIDATES field in an online deployment, the F1-compliant scoring is specified for future use:
- Score the fixed set gold ∪ (2·\|gold\| compiler-sampled matched negatives) ∪ listed.
- Normalise: `S = (Σ_{v∈gold} S_v + Σ_{v∈listed∖gold} S_v) / |gold|`.
- Count omitted gold at p = 0.05, including inside the group baseline.
- Apply η only to the `p=` tokens.
- Cap list length at 3·\|examined files\|.
- A reward unit test asserts that honest listing beats every padding or omission policy on the K-30 grid. That test ships with the code.

**Item readout** (training and evaluation share one format; K-41, K-36, K-32):
- The guided-decoding scaffold forces `"<qualname>: CALLED p="` per candidate, in canonical (file, line) order.
- The model chooses one of 5 level tokens {0.05, 0.2, 0.5, 0.8, 0.95}.
- **Primary score** = the expectation under the renormalised level-token distribution, which is continuous. The parse rate is 100% for every model.
- There is no EDIT|SKIP label; p is the only quantity.
- Free-generation parse failures are reported separately, for the base model only.

### 2.7 Objectives (C8)

All losses use a **constant normaliser** L_max = 16,384 tokens per sequence (Dr.GRPO-style; BM-§7.2, K-42).

**Stage A: execution-labelled SFT warm start on items** [KNOWN; TraceEval 2605.11006 is the real→real prior]:

`L_A = −(1/L_max) Σ_items Σ_{v∈pool} log π_θ(ℓ(y_v) | scaffold_<v)`

- ℓ(1) = the 0.95 token and ℓ(0) = the 0.05 token, with label smoothing 0.1 onto the adjacent level.
- Only level tokens carry loss.
- lr 1e-4, 1 epoch.
- In M, each pool appears in both wirings within the same epoch, in shuffled order, never in the same sequence. Never showing both twins in one context avoids the paired-CAD "learn the edit" shortcut (2406.06633).

**Stage B: offline RL-style step = outcome-filtered self-training (RFT)** [KNOWN: RAFT / ReST / STaR family, INFERRED]:

`L_B = −(1/L_max) Σ_{τ∈D_B} Σ_{t∈assistant(τ)} log π_θ(y_t | s_t)`

- D_B = binary-verified successes from the shared **base-generated** pool, deduplicated and stratified-matched.
- lr 5e-5, 1 epoch.
- This is the binary-reward, zero-baseline policy-improvement step from the behaviour policy. It is named RFT, not RL (BM-§6.4).
- **Removed from v1:** the round-2 regeneration, the clipped surrogate, the trainer recompute of π_A, and the DPO fallback (BM-§6, K-43, K-44). At this data size they were predicted to add ≈0 and cost 3–6 GPU-h. The savings pay for seeds.

**Online-GRPO drop-in.** This is a design, not tested here (BM-§5):
- The task server emits twin pairs, and **every batch contains both halves** of each pair.
- GRPO uses a per-task group of K = 8. A pair-pooled baseline, the mean over both twins' 2K samples, is an option; it leaves the expectation unchanged and may reduce variance.
- Truncated importance weights `w_t = min(π_train/π_vLLM, 2)`, with masking when |log w| > 1 (F10, K-40).
- Sample at the deployed temperature and top-k, applying the identical logit transform in the trainer (BM-§1.1).
- Stage A is kept as the warm start.

### 2.8 PEFT and quantisation

| Setting | Value |
|---|---|
| LoRA | r = 16, α = 32, dropout 0.05, all 410 linear modules (global layers have no v_proj) |
| Excluded | No DoRA. `bias="none"`. No `modules_to_save`. No embedding or lm_head targets |
| **Base for training (default)** | W4A16 dequantised to bf16, then **bnb NF4 with `bnb_4bit_quant_storage=bfloat16`**. This is the documented FSDP-QLoRA path (feasibility doc; BM-§1.3). FSDP FULL_SHARD, gradient checkpointing, SDPA with the math backend disabled (head_dim 512), chunked CE with softcap 30, 16k sequences. The NF4 state dict is cached once as a Kaggle dataset (BM-§1.2) |
| Train–serve skew | Measured, not assumed: next-token agreement, plus per-token KL **p50 and p99** and per-sequence sums, between the NF4 trainer and vLLM W4A16, **with the zero adapter and with a 20-step adapter** (F10, K-40, A12 gate on p99). **Every evaluation runs on the served stack** (vLLM W4A16 + LoRA), so any skew shows up in the reported numbers |
| HQQ g32 (exact-grid) | **Ablation only**, conditional: one M seed retrained if the probe shows that HQQ shards under FSDP. Not in the core |
| Batch / schedule | Effective batch 16 sequences, cosine, 10 warm-up steps. Dose ≈0.5M tokens per adapter (≈200 twin items + ≈20 trajectories; P ≈250 items) |

### 2.9 Evaluation (C9b, C10, C11)

**Real items, the primary transfer readout** (applying F8, F9, K-09, K-35, K-49 and BM-§2.1, §2.4, §3.5):
- **16–20 S_eval repos × 16–20 items = 320 confirmatory items**, plus 80 calibration/exploratory items.
- S_eval excludes the dev repos and S_topo.
- Items are frozen and **SHA-256 hashed before any trained-arm evaluation**.

Pools:
- 16 candidates, ≈2k tokens;
- **path-complete**: every positive's shortest-path intermediates are in the pool, and its call sites lie within the shown lines;
- at most 2 hub positives;
- tests that patch pool functions are excluded;
- truncation is marked `# … N more lines`;
- a call site is typed "lexical" only if it resolves uniquely in the pool;
- each distractor is labelled **statically reachable or not** (name-reachability).

Contexts:
- **real** for every adapter;
- **plausible-RW**: arity-compatible swaps among callees in files the caller imports; test body not rewired; for base and M, U, N seed 1;
- **GF** (names and signatures): base only.

Measurement aids:
- A no-LLM **name-reachability oracle** is scored in every context.
- Normalised `Δ̂_struct = (AP_real − AP_RW)/(AP^oracle_real − AP^oracle_RW)` is exploratory. It is also reported residualised on AP_real, because of K-34.
- A corruption-detection probe on base, asking whether the code was tampered with, must be near chance on plausible-RW.

**Synthetic twin items.** 240 held-out items from 3 held-out source families, each 80 items as 40 pools × 2 wirings (K-51). Used for GW-a, H4, H2a and H3a.

**Retention and near-SOTA** (BM-§4.2):
- MMLU-Pro slice, 300 items, scored over options by log-likelihood;
- HumanEval+ (164, greedy);
- tool-call validity on 100 held-out deployed-agent turns (schema-valid calls);
- **exact per-token KL** over the full vocabulary between adapter and base, computed in the trainer on 20 held-out base agent trajectories (synthetic plus SWE-V prompts). This is exact, not an estimator (K-42).

**External, SWE-facing** (BM-§4.3):
- (i) **SWE-bench-Verified issue→edit-set items**: 300 multi-function-gold items. Pool = functions in the gold-patch files plus same-file distractors; gold = functions touched by the gold patch. Same readout.
- (ii) **TraceEval slice** (2605.11006, VERIFIED by BP), 200 items converted to the pooled format.
- RepoReasoner call-chain items are optional.
- Contamination is equal across arms because the comparison is paired.

**Reference points** (0 GPU-h, API):
- the bf16 Gemma-4-31B endpoint (AI Studio; serving precision GUESSED to be bf16);
- one frontier model, on the 320 real items, SWE-V and TraceEval.

APIs give no level-token log-probs, so these use verbalised p. W4A16 base is also scored in verbalised mode (inside L4) for an apples-to-apples comparison.

**"Near SOTA" is operationalised as:** retention non-inferior **and** the W4A16+M score on SWE-V and real items within the stated gap of the bf16 reference.

**Agent level (C10).**
- 20 held-out twin pairs × 2 tasks, for base, M seed 1 and U seed 1, on the deployed pipeline.
- Metric: the share of pairs where both twins are solved.
- Descriptive with a paired CI (n = 20).

**Dev resolve (C11): conditional and descriptive.**
- base and M seed 1 on the **54 scorer-solvable dev tasks** (VERIFIED, HQ Oct 7: rich 39, fastapi 10, requests 5).
- Same session, sequential, deployed wall clock.
- No inferential claim (BM-§1.5, §2.5). The guard arm is fixed to M seed 1 in advance (BM-§2.4).

---

## 3. Data flow

1. **CPU (week 1–2).** C1 traces S_topo, S_eval and S_ctrl. C2 applies hygiene. C3 builds twins. C4 compiles worlds and places symptom-matched bugs. C5 verifies them and runs audits (i), (iii) and (iv). C9a builds and **freezes** real items. Twin items, P items, the SWE-V edit-set items and the TraceEval conversion are all built here.
2. **GPU L1 (week 1).** Training probe and pilot rollouts. Audit (ii) runs here (prefill).
3. **GPU L2 → gate G2.** R1 rollouts. G2 checks the d_obs slope (controlling for tokens, with cap hits treated as a competing risk; K-08, BM-§2.4) and the (d′) bug-named contrast.
4. **GPU L3.** R2 rollouts complete the shared pool.
5. **CPU.** Binary verification, then arm selection, stratified matching and the balance table.
6. **GPU L5.** Twelve adapters (M, U, N, P × 3 seeds), plus the exact-KL forwards, in one or two FSDP sessions.
7. **GPU L4 and L6.** Multi-LoRA evaluation sessions on the served stack.
8. **GPU L7.** Agent-level twin check.
9. **API.** Reference points.
10. **Conditional L9/L10.** Dev descriptive; HQQ ablation.

---

## 4. Gates

| Gate | When | Pass rule | If it fails |
|---|---|---|---|
| G0 probe | L1 | NF4-FSDP fits 16k at ≤22 GB/GPU. ≥250 tok/s. Mem-efficient SDPA runs forward and backward at head_dim 512. Trainer-vs-vLLM per-token KL p99 ≤1.0 nat with the zero adapter and with the 20-step adapter. The exported adapter routes in vLLM. Also record `nproc`, `num_running_seqs` and `gpu_cache_usage` at 16 streams (KV hybrid allocator; BM-§1.1) | Per the feasibility doc: 8k cap with prefix-fit trimming (disclosed); fewer streams |
| G1 throughput | First 30 rollouts | ≥40 rollouts per GPU-h on the deployed pipeline | Ladder D6 (pool shrinks). If below 25/h, Stage B is dropped and the method runs on items only; disclosed as "RFT stage not run" |
| G-audit | CPU + L1 | (i) observation-free pair accuracy ≤0.55; (ii) body-discriminator AUC in [0.45, 0.55]; (iii) import AUC in [0.45, 0.55]; twin yield ≥5% | Fix the generator (all arms) before L2 |
| G2 world validity | After R1 (90 rollouts) | **(a)** Base success falls with d_obs, with a logistic-slope CI excluding 0 **after** controlling for tokens consumed; cap hits counted separately. **(b)** (d′) bug-named success − normal ≥ +10 pp. **(c)** Base pair-both-solved rate is below the product of single-half rates + 0.05, i.e. the base is not solving twins by a shared shortcut | (a) or (b) fails: the worlds do not demand traversal, so **stop** and report. (c) fails: a twin leak; fix the generator |
| GW | After L6 | §1.1 | §1.1 |

Numeric triggers are pre-registered. Any pipeline change applies to all arms, and is logged (BM-§2.4, K-44, K-45).

---

## 5. GPU-hour budget (whole study)

**Rates** (optimistic / central / pessimistic):

| Quantity | Opt. | Central | Pess. |
|---|---|---|---|
| Rollouts per GPU-h | 100 | 60 | 40 |
| Training tok/s (pessimistic = the G0 floor) | 650 | 500 | 250 |
| Prefill tok/s | 3k | 2.25k | 1.5k |
| Aggregate decode tok/s | 600 | 400 | 250 |
| Overhead per session (h) | 0.2 | 0.3 | 0.4 |

Each line below includes its own session overheads. The method breaker's corrections are applied (BM-§1.9, §9).

| # | Line | Week | Opt. | Central | Pess. | Basis |
|---|---|---|---|---|---|---|
| L1 | Probe (NF4-FSDP 16k, skew p99 with and without adapter, KV/nproc) + 30 pilot rollouts + audit (ii) | 1 | 1.2 | **1.5** | 3.0 | Probe history v1–v8 sets the pessimistic value |
| L2 | R1 gate rollouts: 60 more, making 90 | 2 | 0.9 | 1.3 | 1.9 | |
| L3 | R2 pool rollouts: 210 | 2 | 2.4 | 3.8 | 5.6 | |
| L4 | Base evaluation: 400 real items × {real, plausible-RW, GF} + 240 synthetic twin items + SWE-V 300 + TraceEval 200 + retention + verbalised-mode real items | 2 | 1.0 | 1.5 | 2.5 | ≈6.5M prefill + ≈0.5M guided decode |
| L5 | Training: 12 adapters × 0.5M tokens + exact-KL forwards (20 trajectories × 2 passes) + 2 loads | 3 | 3.2 | 4.3 | 8.0 | 6.0M training tokens |
| L6 | Trained evaluation: 12 × real (320); 9 × synthetic twins (240); 3 × plausible-RW; 3 × (SWE-V + TraceEval + retention) for M, U, P seed 1; Stage-A snapshot of M seed 1 | 3 | 3.2 | 4.8 | 7.5 | Multi-LoRA (punica +15%); no cross-LoRA prefix cache |
| L7 | Agent-level twin check: 20 pairs × 2 × {base, M1, U1} = 120 rollouts | 4 | 1.4 | 2.3 | 3.3 | |
| L8 | Gate re-runs and failures | — | 0.5 | 1.0 | 2.0 | Unbudgeted in v1 |
| | **Core subtotal** | | **13.8** | **20.5** | **33.8** | |
| | Reserve (10 / 15 / 20%) | | 1.4 | 3.1 | 6.8 | v1 had 4% |
| | **Core total** | | **15.2** | **23.6** | **40.6 → ladder** | |
| L9 | *Conditional:* dev resolve descriptive, base vs M1 on 54 solvable tasks, sequential | 4 | 2.6 | 3.2 | 4.2 | Runs only if the projected total including it is ≤25 |
| L10 | *Conditional:* HQQ exact-grid ablation (1 adapter + real evaluation) | 4 | 0.5 | 0.6 | 1.0 | Only if HQQ shards under FSDP |
| L11 | *Conditional:* small-model replication (M vs U, Stage A only, Gemma-4 12B bf16, 1 seed each, real + synthetic items) | 4 | 1.0 | 1.5 | 2.5 | The only way to *show*, rather than assert, generality (BM-§5) |

**Weekly fit.** Week 1 is 1.5 (L1). Week 2 ≈6.6, week 3 ≈9.1, week 4 ≈3.3 plus conditional lines. Each week is far inside the ~30 h quota, which is shared with main-track evaluations.

**Descoping ladder.** Applied in order to the **pessimistic** case. Low-value lines go first; the scientific core is never dropped (BM-§1.10).

| Step | Change | Saving (pess.) | Cumulative core (pess.) | Effect on science |
|---|---|---|---|---|
| — | Start | — | 33.8 | |
| D1 | L9, L10 and L11 not run | 0 | 33.8 | Lose the dev anecdote, the HQQ ablation and replication; generality stays a design claim |
| D2 | Drop the Stage-A snapshot evaluation, and plausible-RW for U and N (keep base and M) | −0.7 | 33.1 | Exploratory only |
| D3 | External and retention for base and M only (drop U and P) | −1.5 | 31.6 | R and E still cover the method arm |
| D4 | Agent twin check: 20 → 12 pairs, base and M only | −1.9 | 29.7 | A becomes thinner and descriptive |
| D5 | P seeds 3 → 2 | −0.9 | 28.8 | GW-b slightly noisier |
| D6 | Skip R2; Stage B uses R1 successes only | −5.6 | 23.2 | The RFT stage becomes thin (≈10–15 trajectories per arm). The method then rests on the item stage; disclosed |
| D7 | Dose 0.5M → 0.4M tokens per adapter, **for all arms equally** | −1.4 | 21.8 | Lower dose; P still anchors |
| | Reserve 10% | +2.2 | **24.0 ✓** | |

**Never dropped (the core):**
- arms M, U and N at **3 data-resampled seeds**, P at ≥2 seeds, and base;
- the shared twin pool from at least R1;
- GW, H4 and H2a on the synthetic twins;
- H1 and H2b on the 320 frozen real items;
- H3a;
- retention and exact KL for M;
- SWE-V for M and base.

---

## 6. Generalisation: what changes for other models and hardware

| Dimension | This instance | Rule | Status |
|---|---|---|---|
| Loop | Offline: base-generated pool → Stage A items → Stage B RFT | Online GRPO with twin-balanced batches; same binary reward; TIS weights; Stage A kept | **Design only.** Not tested here |
| Quantised training base | bnb NF4 from dequantised W4A16; skew measured; evaluation always on the served stack | Train with the documented QLoRA path for the serving format. Always evaluate on the served stack. Report skew p99 | Shown for this instance only |
| LoRA rank and dose | r = 16, 0.5M tokens | Scale dose with the positive control: keep increasing until GW-b passes | — |
| Sampling | Deployed settings (T = 0.2 / top-p 0.95 / top-k 40) | Always sample at deployed settings; for online RL apply the same transform in the trainer | — |
| Twin generator | Python, call graphs, compiled worlds | Any executable DAG with a tracer: build systems, dataflow pipelines, workflow engines. The C3 swap constraints (level, block, type) are language-agnostic | **Conjecture** outside Python |
| Model scale | Gemma-4-31B W4A16 | — | Shown only if L11 runs |

---

## 7. Retention and near-SOTA argument (measured, not asserted)

1. **Dose.** Small: ≈0.5M tokens, r = 16, 1 epoch per stage. Stage B uses on-policy-at-base successes.
2. **Measurements:**
   - the retention battery (MMLU-Pro slice, HumanEval+, tool-call validity) as non-inferiority per adapter;
   - **exact per-token KL** from base on held-out agent trajectories (mean and p99);
   - the train–serve skew p99.
3. **Reference points.** The bf16 and frontier models, scored on the same SWE-V, TraceEval and real items, place W4A16 + LoRA on an absolute scale.
4. **Optional.** The bf16 + same-LoRA vs W4A16 + LoRA gap is not in the core. It would need a bf16 serving session (≈0.9 h) that the ladder cannot afford.

---

## 8. Statistics

**Units.** The arm effect is replicated over **3 data-resampled seeds**, and evaluated on items nested in repos (real) or in families (synthetic).

**Model:**

`score_item ~ arm + (1|repo) + (1|item) + (1|seed:arm)`

Primary inference uses a hierarchical bootstrap: resample seeds within arm, then repos with **wild-cluster bootstrap-t (Webb six-point weights)**, then items. With 16–20 repos, few-cluster inference is acceptable. The ICC is reported (F9, K-50, BM-§2.1).

**Power** (GUESSED inputs: paired item SD 0.15, ICC 0.05 at m = 16–20 → design effect 1.8–2.0, seed SD 0.015):

| Contrast | SE | MDE (80%) | Prediction | Verdict |
|---|---|---|---|---|
| Paired arm difference on real items | ≈0.016 | ≈0.045 | H2b +0.02 | Underpowered, declared |
| | | | H1 +0.04 | Borderline, declared |
| Synthetic-twin contrasts (paired twins, 240 items, 3 families) | ≈0.010 | ≈0.03 | H2a +0.12, H4 −0.25 | Well powered |
| | | | H3a TOST at 0.03 | Passable only if the true \|I\| ≤ ~0.01 |

**Other rules:**
- Equivalence is reported only when the **seed-inclusive** CI fits the margin.
- H3 is evaluated only if GW passes.
- The SESOI for H3b is tied to the positive control, not to the MDE (BM-§2.2).
- A leave-one-source-graph-out sensitivity analysis is reported for H3a.
- Pre-registered **before** trained-arm evaluation: the hashed item set, the sampler seed, the repo-exclusion rule, every numeric trigger, the gatekeeping order, and the fixed guard arm.

---

## 9. Timeline

| Week | GPU | CPU |
|---|---|---|
| **1** (Oct 8–14) | L1 = 1.5 | C1–C5; audits (i)/(iii)/(iv); twin yield; C9a freeze; P items; SWE-V and TraceEval item conversion; reward and generator unit tests |
| **2** | L2 → G2; L3; L4 | Balance tables |
| **3** | L5; L6 | Analysis scripts frozen before L6 |
| **4** | L7; L8; conditional L9–L11 | Write-up |

The Nov 5 freeze leaves about one week of slack.

---

## 10. v1 → v2 change log

**F = applied, J = skipped with justification.** BM = `break_v1_method.md`, BP = `break_v1_priorart.md`.

### Logic breaks (K-xx)

| Break | Severity | v2 response | § |
|---|---|---|---|
| K-01 union-of-tests depth ≈1 | major | F: integration-only roots; d_obs as the difficulty variable; depth histogram before GPU | 2.2 |
| K-02 SCC mega-hubs | major | F: largest-SCC cap of 5%, or bounded recursion with `fuel`; node-fraction reported | 2.2 |
| K-03 BFS frontier truncation | minor | F: task-wise growth; frontier out-degree reported | 2.2 |
| K-04 contract bisection shortcut | fatal | F: contracts placed first; bug at d_obs ≥2; symptom-matched twins make the deepest failing contract identical across twins; observation-free solver audit | 2.2, 4 |
| K-05 value masking, differential drop | major | F: non-injective templates banned on bug→observer paths (or tagged); per-side drop rate; pair kept only if both halves survive | 2.2 |
| K-06 hubs, vacuous P2P | major | F: effect-only edge share; hub-bug cap at 30% coverage; \|P2P\| ≥5 | 2.2 |
| K-07 dark edges not dark | major | F: ≥3 overrides selected elsewhere; computed registry keys; getattr; grep-distance reported | 2.2 |
| K-08 depth slope driven by budget | major | F: competing-risk analysis of cap hits; slope controlled for tokens; repos sized so the gold path fits ≤50% of the cap | 4 |
| K-09 mocks, branches, name-collision typing | major | F: mocked tests excluded; static-reachable distractor labels; unique-resolution lexical typing | 2.9 |
| K-10 no dead code | minor | F: 10–20% unreachable nodes; 10% reachable only from non-F2P tests | 2.2 |
| K-11 DPCS import fingerprint | major/fatal | F: module(b) = module(d), so imports are identical; single-use-import AUC audit | 2.2 |
| K-12 template-anomaly shortcut | fatal | F: in-family randomised instances; body-only discriminator gate | 2.2, 4 |
| K-13 a = c degenerate swap | major | F: a ≠ c, different files, distance ≥2 | 2.2 |
| K-14 DPCS on registry/decorator | major | F: lexical and callback edges only | 2.2 |
| K-15 value-range giveaways | major | F: total templates; same-family callees | 2.2 |
| K-16 kwarg, local-name and docstring leaks | minor/major | F: positional calls; independent local names; no docstrings | 2.2 |
| K-17 alternative valid fixes penalised | major | F: binary test reward accepts any fix; in-degree ≥2 so definition-only fixes break P2P; single-site alternatives enumerated and must fail; item gold = union of minimal passing sets | 2.2 |
| K-18 hardcoding issue values | major | F: hidden F2P on fresh random inputs; binary success; literal-copy rejection | 2.2 |
| K-19 drop filter favours distinguishable callees | minor | F: same-family callees; drop rate reported | 2.2 |
| K-20 coverage changes under rewire | major | F: symptom matching equalises failing visible tests per pair; roots-per-node, \|anc\|/\|desc\| and spectrum-solver rate reported | 2.2 |
| K-21 locality and imports destroyed | major/fatal | F: block-preserving swaps (module(b) = module(d)); modularity and intra-file fraction identical | 2.2 |
| K-22 motif drop | major | F (partial): motifs reported; H3 scoped to "beyond {degree, level, block}". J: no motif band, which would leave too few legal swaps; disclosed | 2.2 |
| K-23 KS depth filter invalid, selects graphs | major | F: exact level preservation by construction; no filtering | 2.2 |
| K-24 dispatch/decorator rewire needs odd class structures | major | F: dark edges never rewired; class count and MRO depth identical | 2.2 |
| K-25 summary of (b)≈(c) artefacts | — | F: each row addressed by the K-04, K-12, K-41, K-50 and K-23 fixes; detectors kept as audits | 2.2, 4, 8 |
| K-26 (d) changes bodies, length, success | major | F: arm (d) removed | 2.5 |
| K-27 (d) names the bug | major | F: (d′), the same graphs with a bug-named issue, used at gate G2(b) | 2.5, 4 |
| K-28 (d) negative transfer makes H2 pass spuriously | major | F: no (d) arm; H2b is now M vs U with an identical item distribution | 2.5 |
| K-30 set reward rewards padding | fatal | F: the policy-controlled set term is removed from the RL reward; items use fixed pools; the F1 formula and a unit test are specified for future online use | 2.6 |
| K-31 η on name tokens, omission free | major | F: η removed; the future formula applies η to p tokens only, with omitted gold in the baseline | 2.6 |
| K-32 EDIT\|SKIP redundant | minor | F: label dropped; p only | 2.6 |
| K-33 RFT filter selects long lists | major | F: RFT filter is binary test success only | 2.7 |
| K-34 Δ_struct confounds | major | F: demoted to exploratory; plausible-RW; corruption probe; oracle-normalised and residualised; test-body rewiring stratified | 2.9 |
| K-35 statically reachable distractors | major | F: static-reachability labels and stratum; the H2b stratum is defined on it | 2.9 |
| K-36 5-level ties | minor | F: continuous expected level from the renormalised level-token distribution | 2.6 |
| K-37 H6 gameable | major | F: H6 removed; replaced by item AP and agent pair-both-solved | 1.2 |
| K-40 tail mismatch ignored | major | F: offline RFT has no ratios; skew p99 gated; all evaluation on the served stack; TIS for the online drop-in. J: exact dequantised base, because the coordinator mandates NF4 by default; HQQ is a conditional ablation | 2.8 |
| K-41 format-compliance transfer | major | F: guided scaffold, 100% parse for every model; equal item format across M/U/N; P anchors format and content | 2.6 |
| K-42 KL estimator, length weighting | minor | F: exact full-vocabulary KL; constant normaliser | 2.7, 2.9 |
| K-43 zero-variance groups | minor/major | F: Stage-B surrogate and η removed | 2.7 |
| K-44 DPO fallback pitfalls | major | F: DPO removed; any trigger applies to all arms | 2.7, 4 |
| K-45 tuning on (b) first | minor | F: hyperparameters frozen on a throwaway fourth family | 2.5 |
| K-46 per-arm size, success selection | major | F: no per-arm size changes; stratified cell matching; balance table | 2.5 |
| K-47 content-free teaches "ignore names" | major | F: fixed weak name–template correlation in all arms; H2b stratified by lexical vs dark; P anchors real-name effects | 2.2 |
| K-48 spurious H1 sources | — | F: guided readout (format); base-rate variation (exploratory); truncation markers; hub cap; N's RW reported separately | 2.9 |
| K-49 hub- and integration-dominated eval | minor | F: ≤2 hub positives per item | 2.9 |
| K-50 pseudo-replication | major | F: 3 data-resampled seeds per arm; hierarchical bootstrap; 16–20 repos; wild-cluster bootstrap-t | 8 |
| K-51 one held-out family | minor | F: 3 held-out families | 2.9 |

### The breaker's fix set F1–F11

| Fix | Applied? | Where |
|---|---|---|
| F1 reward | Superseded: the set term is removed from RL; formula plus unit test specified for the online CANDIDATES variant | 2.6 |
| F2 hidden fresh inputs, binary success, no literals | Yes | 2.2 |
| F3 d_obs, contracts first, solver audit | Yes | 2.2, 4 |
| F4 in-distribution mutations, discriminator gate | Yes | 2.2, 4 |
| F5 DPCS constraints, minimal passing gold | Yes, with module(b) = module(d) | 2.2 |
| F6 level- and block-preserving swaps, dark edges not rewired | Yes, exact by construction | 2.2 |
| F7 replace (d) with (d′) | Partial: (d′) used at the gate; (d) dropped as an arm (prior-expected; budget goes to seeds) | 2.5 |
| F8 teacher-forced readout, oracle, plausible-ρ, strata | Yes (guided scaffold = teacher-forced format) | 2.6, 2.9 |
| F9 seeds, hierarchical bootstrap, clusters | Yes | 8 |
| F10 TIS, p99, joint triggers, exact dequant | TIS for online; p99 yes; joint yes. Exact dequant replaced by NF4 default plus served-stack evaluation (J above) | 2.8 |
| F11 source hygiene before GPU | Yes | 2.2 |

### Method breaks (BM)

| Break | v2 response | § |
|---|---|---|
| BM-§1.1 KV concurrency, nproc, T = 1.0 | Probe records them; deployed sampling; suite ≤10 s | 2.4, 4 |
| BM-§1.2 session overheads | Counted per line; NF4 state cached | 5, 2.8 |
| BM-§1.3 HQQ + FSDP undocumented; 250 tok/s floor | NF4 default; HQQ conditional; pessimistic column at 250 tok/s | 2.8, 5 |
| BM-§1.4 evaluation cost | Recomputed; ≈2k-token items; guided decode; GF on base only | 5 |
| BM-§1.5 dev guard costing and invalid baseline | Conditional, descriptive, same-session paired on the 54 solvable tasks | 2.9, 5 |
| BM-§1.7 unbudgeted gate costs | L8 line | 5 |
| BM-§1.8 dose too small | Dose raised to ≈0.5M tokens per adapter, mostly exact-label items; P anchors whether it is enough | 2.5, 2.8 |
| BM-§1.10 ladder cuts science first | Reordered; core listed as never dropped | 5 |
| BM-§2.1 8 clusters | 16–20 repos; wild-cluster bootstrap-t; ICC reported | 8 |
| BM-§2.2 TOST backwards; seed noise; 6 sources | H3 conditional on GW; 3 data-resampled seeds; SESOI from P; leave-one-source-out; H3a powered on synthetic | 1.2, 8 |
| BM-§2.3 multiplicity | Fixed-sequence gatekeeping; BH-FDR on exploratory families | 1.2 |
| BM-§2.4 pre-registration gaps | Freeze and hash; fixed guard arm; numeric triggers; continuous score; K1 controls for length | 4, 8 |
| BM-§2.5 H5 margin | H5 removed; dev is descriptive | 2.9 |
| BM-§3.1 items differ in depth by arm | Items are identical pools across M/U/N (only the wiring shown differs) and level-matched by construction; P anchors | 2.5 |
| BM-§3.2 equalisation inconsistent | Stratified cell matching; equal tokens and steps; balance SMD <0.1 | 2.5 |
| BM-§3.3 locality, name–edge coherence | Block-preserving swaps; names independent of topology | 2.2 |
| BM-§3.4 (b) vs (d) bundle | Arm removed | 2.5 |
| BM-§3.5 path-incomplete pools, RW incoherence | Path-complete pools; plausible-RW | 2.9 |
| BM-§3.6 no positive control | Arm P with 3 seeds; GW-b | 2.5, 1.1 |
| BM-§4.1 rubric reframing | Method plus null-model experiment, one headline mechanism (§0) | 0 |
| BM-§4.2 near-SOTA unmeasured | Retention battery, exact KL, bf16 and frontier references | 2.9, 7 |
| BM-§4.3 external benchmarks | SWE-V edit-set and TraceEval slice; RepoReasoner optional. J: large serialised-graph items deferred (budget); noted for v3 | 2.9 |
| BM-§5 generality asserted | Claims downgraded to design or conjecture; small-model replication as conditional L11 | 6 |
| BM-§6 Stage B ≈0 | Round 2, surrogate and π_A recompute removed; Stage B is single-step RFT from the shared pool | 2.7 |
| BM-§7.1 harness not deployed | Deployed V5 SequentialAgent via `swegemma eval` | 2.4 |
| BM-§7.2 loss normaliser | Constant L_max | 2.7 |
| BM-§7.3 to §7.6 | CANDIDATES removed; H6 removed; 3 held-out families; qualified-name overlap check (B10) | 2.6, 2.9 |

### Prior-art breaks (BP)

| Break | v2 response | § |
|---|---|---|
| BP-1 compiler not new (FuncBenchGen, DynaCode, SWM, PhantomEnvs) | Tagged [KNOWN pattern]; cited; no longer a claimed contribution | 0, 2.1 |
| BP-2 DPCS = MCR / HOM | Renamed "degree-preserving second-order MCR mutant" [NARROWED]; one operator among three | 2.2 |
| BP-3 Δ_struct = CNY / Huang TMLR'24 | [NARROWED]; exploratory only | 2.9 |
| BP-4 only the topology-fidelity treatment survives | Made into the training signal (null twins) **and** the H3 experiment | 0, 2.3 |
| BP-1f TraceEval | Cited as the real→real prior; used as an external slice | 2.7, 2.9 |
| H1/H2 now prior-expected (PhantomEnvs) | The depth arm is dropped; the hop effect is cited, not claimed | 2.5 |

---

## 11. Assumptions for the next breaker

| ID | Assumption | Check | If false |
|---|---|---|---|
| **B1** | **Twin mechanism novelty.** No prior uses null-model-generated, symptom-matched executable twins as an RL/RFT training signal. Only CAD-family and environment-invariance priors were found (3 searches, SNIPPET) | Needs a dedicated red-team pass: CAD for code, minimal-pair RL, invariant policy optimisation, "contrastive environments" | The claim narrows to "CAD instantiated with graph null models for agent RFT" |
| B2 | Twin halves are indistinguishable before wiring is read. No leaks from file length, number formats or issue numbers | Audits (i)–(iii); a twin-identity probe on the base model (prefill, ask which twin) must be near chance | Twin balance stops cancelling; H4 is uninterpretable |
| B3 | q > ½ is reachable. Reading the wiring is reliable enough that the cancellation favours it | Base pair-both-solved and single-half success at R1 | Balance yields indifference rather than reading; predict H4 passes but H2a fails |
| B4 | The symptom-match yield is ≥5% of candidate nodes, enough for 75 + 60 held-out pairs | CPU in week 1 | Relaxed matching (disclosed) weakens B2 |
| B5 | Base success on twins sits in roughly [0.15, 0.6], so the pool gives ≈20 successes per half | G1/G2 | Stage B is thin; the method rests on items (D6 regime) |
| B6 | About 0.5M tokens per adapter is enough to move in-distribution metrics (GW-a) | GW-a | Nothing evaluable; dose finding |
| B7 | Format and calibration effects are equal across M, U and N, because items are identical apart from the wiring shown | Parse is 100% by scaffold; report calibration (ECE) per arm | M−U contrast confounded by calibration |
| B8 | Seed SD ≈0.015 AP; 3 data-resampled seeds suffice | Observed between-seed SD | H2b and H3b wider; already declared underpowered |
| B9 | ICC ≈0.05 with 16–20 repos | Reported ICC | Wider CIs |
| B10 | No leakage path: S_topo, S_eval, S_ctrl and dev are disjoint at repo level; qualified-name overlap of identifier pools with S_eval is reported (BM-§7.6) | Static check | Re-sample pools |
| B11 | NF4 double rounding (W4A16 → bf16 → NF4) leaves skew p99 ≤1 nat, and adapters trained on NF4 transfer to the served W4A16 | G0 plus served-stack evaluation | HQQ ablation (L10) becomes core; budget is re-planned |
| B12 | Deployed sampling (T = 0.2) gives enough task-level diversity for RFT | Deduplicated success count | Raise K for all arms; temperature stays deployed |
| B13 | The deployed wall-clock cap scaled by measured slowdown s preserves the deployed step budget | Compare steps per task at 1 vs 16 streams on 10 tasks | Tool-call cap only, disclosed |
| B14 | Guided-scaffold expected level is a valid continuous score across models (same level-token set) | Report the level-token mass outside the 5 tokens (should be ≈0 under guidance) | Fall back to verbalised argmax with tie handling |
| B15 | SWE-V edit-set items are a fair SWE-facing proxy (gold = functions touched by the gold patch) | Report the share of gold patches touching new code (dropped and counted) | Descriptive only |
| B16 | The AI Studio endpoint serves bf16 Gemma-4-31B, and the API terms allow evaluation use | Check the documentation | bf16 reference via L4 session (+0.9 h) or dropped |
| B17 | H3a, the synthetic crossover, generalises the null-model question beyond the 6–8 source repos | Leave-one-source-out | H3a reported per source |
| B18 | The weak name–template correlation (30%) neither dominates nor erases the name prior | AP on the lexical stratum vs base | Re-tune for all arms (one change) |
| B19 | Removing CANDIDATES from the agent leaves the method relevant to the deployed agent through RFT and item transfer | Agent twin check (A) and the dev descriptive run | The SWE relevance claim weakens; the method stays an item/graph-reasoning result |
| B20 | Kaggle quota and the main track leave ≈10 GPU-h per week for this study in weeks 2–4 | HQ log | Stretch the timeline; the Nov 5 freeze is at risk |

---

## Search log (this pass)

| # | Tool | Query | Result |
|---|---|---|---|
| S1 | firecrawl web | counterfactual minimal pair training, graph rewiring, LLM graph reasoning, degree-preserving | Knowledge-conflict and KGE counterfactual work only; nothing on rewired-graph training pairs |
| S2 | firecrawl web | counterfactually augmented data for code models / program repair | CAD (Kaushik et al. ICLR 2020/2021); paired-CAD shortcut paper (ACL 2024, 2406.06633); iterative CDA (AAAI). No code or graph-null-model instance |
| S3 | firecrawl web | agent RL with paired counterfactual environments, identical observations, cancelling shortcut credit | Counterfactual credit assignment (MARL), PAIRED environment design, Agent World Model. None with twin environments differing only in graph wiring |

All other prior art is inherited from the BP, red-team and critic logs, with their labels.
