# Architecture v2: Behavioural Map-Matching, rebuilt after three breaks (9 Oct 2026)

**Inputs.**
- `architecture_v1.md` (v1).
- `break_v1_logic.md` (breaks **L1–L46**).
- `break_v1_method.md` (items **M§n**).
- `break_v1_priorart.md` (items **PA**).
- The phase-3 critic and red-team notes.
- `docs/training_feasibility_4xl4.md` and HQ.

**Labels.**
- **VERIFIED**: in repo docs or a breaker's primary read.
- **INFERRED**: my reasoning.
- **GUESSED**: a number that a kill test must measure.

**Nothing has been run.**

**Calendar.** Today is Oct 9; experiment freeze is Nov 5; the paper is due Nov 12.
- One person-day (pd) = 5 h, solo, shared with the main track.
- NH = one Kaggle 4×L4 session-hour.

---

## 0. Decision summary

### 0.1 What survived the breaks

- **Novel piece.** The prior-art breaker rated **behavioural emissions** at closeness **1.5**. They are the only unit with no closeness-2 prior in localisation.
- **Everything else is a known ingredient, and v2 claims none of it:**

| Ingredient | Closeness | Nearest priors |
|---|---|---|
| Decoder | 2 | Euler, Newson–Krumm, RoG/GCR |
| Imagined flow | 1.5–2 | LocAgent's prompt |
| Symptom → path | 2 | SherLog, Pensieve, Pathidea, CrashLocator |
| Effect predictor | 2 | IPA-GNN, CodeFlow |
| Measurement | 2 | RepoReasoner |

### 0.2 What the logic breaker showed

v1's emission was the wrong object.
- It encoded *normal* behaviour at ≥10% frequency (L15).
- It had no polarity, although issues report *anti*-normal behaviour (L16).
- It was indistinguishable from grepping exception tokens (L17).
- v1's decoder could not, by its own algebra, move off a well-named entry (L7), leave a missing edge except toward BM25 favourites (L8), or rank a culprit above the entry (L11).
- v1's C1 control never saw the imagined path (L35).

### 0.3 What the method breaker showed

v1 is about 38 pd and 21.6 NH, which does not fit. Two of its parts were also unsound:
- the headline was execution-validated EVAL (yield risk);
- the ≤6k skeleton cannot hold Django (hidden retrieval, M§3.2).

### 0.4 What v2 does

- **Keeps one architecture.** BMM stays: imagined (symbol, effect) path → HMM over a code graph → node posteriors.
- **Rebuilds the novel piece** as **polarity-aware, base-rate-normalised behavioural emissions**.
- **Rebuilds the decoder** (insert states, propensity-weighted sites, mixed teleport, hub normalisation, culprit-weighted readout).
- **Replaces the controls,** so each primary claim is a one-change contrast against an arm that *sees the same imagined path*.
- **Moves the headline to an execution-free, statically gold-mapped EVAL** of ≥600 traceback-free tasks over ≥25 repos. Execution is used only for training labels (CAL repos) and a 120–150-task measurement subset.
- **Cuts** Study B (LATENT), JS, E-TR, PyCG/PoTo, B-CTX, SweRankEmbed-Large and the CoSIL run.

**Budget: 9.9 NH central (15.2 pessimistic) and 15.5 pd before the freeze.**

### 0.5 Honest collapse analysis

BMM could collapse into "BM25 + LLM rerank". v2 is built so that the data, not the narrative, decide whether it does.

| Outcome on EVAL | What BMM is | What the paper is |
|---|---|---|
| C-DEC ✓, C-BEH ✓, C-USE ✓ | Novel technique (closeness 2; emission 1.5) | Technique paper with M1 as analysis |
| C-BEH ✓, C-DEC ✗ | Sequence decoding adds nothing; behaviour does | **Strongest surviving alternative: Behavioural Evidence Reranking (BER).** This is the X-BAG⊗PPR arm itself: polarity-aware effect likelihood ratios over the imagined steps, fused with a PPR prior. It is pre-registered as the fallback headline, at closeness 1.5 (CrashTracker, FaR-Loc, EXCEPT/SBEST) |
| C-DEC ✓, C-BEH ✗ | Euler / map-matching on LLM-imagined names, closeness 2–2.5 | Decoding analysis + M1; no technique novelty claimed |
| C-DEC ✗, C-BEH ✗, BMM ≈ B-DIRECT* ≈ reranker | **It collapses into "BM25 + LLM rerank"**, and I say so | Measurement paper: issue-conditioned imagined-vs-executed paths with typed divergences (PA (d): NARROW→SURVIVE, extends RepoReasoner/SWE-Flux), plus a documented negative result |

The core claim (C-BEH under R2 on the static EVAL) is never descoped. See §9.

---

## 1. The single novel piece, in one sentence

> **Polarity-aware behavioural emissions.** Each step of an LLM's prose-imagined execution of an issue carries the effects the issue *reports* (+) and the effects it *expected but did not see* (−). These are scored as likelihood ratios against the repository's base rates, on per-function effect signatures predicted by a cross-repository model trained on executed traces, and they are compounded along the path by exact HMM decoding on the call graph.

**Closeness of the novel piece itself: 1.5.**
- CrashTracker's exception-thrown summaries, FaR-Loc's described-behaviour retrieval, and EXCEPT/SBEST are all 1.5.
- None scores a typed multi-effect signature against prose-stated effects, none has polarity, and none compounds along a path.

**Closeness of the whole architecture: 2.** Its ancestors are:
- Euler (FSE'19): NL steps → execution graph with shortest-path gap filling. VERIFIED by PA.
- SherLog, Pensieve, Pathidea and CrashLocator: symptom → path.
- Newson–Krumm and RoG/GCR: the decoder and plan-then-ground.
- LocAgent: an imagined flow as a prompt step. VERIFIED by PA.
- IPA-GNN / CodeFlow: execution-trained exception prediction.

**Defensible claim sentence** (PA §0; used only if C-DEC and C-BEH pass):
> *We decode an LLM's prose-derived imagined execution of a traceback-free issue onto a repository call graph with emissions that score behaviour (per-function, polarity-aware effect signatures from a cross-repository, execution-trained model). This beats both the same evidence without sequence decoding and names-plus-body-text decoding, most of all under code-side renaming. We measure against fail-to-pass executions where, and on which edge kinds, imagined paths diverge.*

### 1.1 Must NOT claim (PA §3, adopted verbatim in substance)

1. Imagining or reconstructing an issue's execution flow with an LLM (LocAgent; CoSIL, GraphLocator, SWE-Debate).
2. Symptom-to-path reconstruction without execution (SherLog, Pensieve, Pathidea, CrashLocator, 2506.19045, COCA).
3. Aligning NL step sequences onto an execution graph with gap filling (Euler).
4. HMM/Viterbi/map-matching or Bayesian fault localisation over program structure (Newson–Krumm, Dietz NeurIPS'09, PPDG, SmartFL, Prosecutor).
5. Exception- or effect-driven localisation (EXCEPT, SBEST, CrashTracker, ExLocator, EXPECT).
6. Static prediction of raised exceptions learned from execution (IPA-GNN, CodeFlow, TRACED).
7. The first LLM-predicted vs executed path measurement (RepoReasoner, SWE-Flux, TraceEval).
8. "Search by behaviour, not names" as a new idea (Podgurski & Pierce, Satsy, Jiang SCAM'18, FaR-Loc).
9. "Execution-free" localisation. Say **"no tests at inference"**, because HEAP imports module top-level code.
10. Execution-verified call-edge data as a contribution (TraceEval).

**Related-work order** (PA §4): SherLog → Pensieve → Pathidea → CrashLocator → COCA/2506.19045; Euler; Newson–Krumm and RoG/GCR; LocAgent; IPA-GNN/CodeFlow; RepoReasoner/SWE-Flux.

---

## 2. Component diagram

```
OFFLINE (CPU, CAL-train repos only; Python ≥3.12; sys.monitoring)
  traced suites ─► possibility effect labels (observed in ≥2 distinct tests; MRO-closed exceptions)
                ─► g_φ : AST+HEAP features → q_v ∈ [0,1]^B   (leave-repo-out, isotonic)
                ─► site propensity  h_ψ : P(site executed)    and  CHAIN-lite f_θ : p(callee | site)
                ─► repo base rates are NOT learned here (computed per EVAL repo from q, §3.3)

PER SNAPSHOT (CPU; no tests run)
  G-a  AST resolver: imports, aliases, static MRO, self/cls/super, function-local imports (L24)
  G-b  HEAP: sandboxed import (+ per-repo bootstrap: django.setup, sphinx dummy app; L23), touch __all__ (PEP 562),
       unwrap/decorators, registries, re-exports, callbacks passed to external calls (∅→v, L3)
  G-c  Jedi fallback for still-unresolved sites in G0-selected files only
  G-d  States: functions ∪ scope states (class body, module top level; L1) ∪ ∅ ; vendored/generated excluded (L41)
  G-e  Node effects q̃ (g_φ + exception propagation ≤2 hops + warn/I-O wrapper propagation 1 hop; L20) ; base rates q̄_j

PER ISSUE (GPU, vLLM, Gemma-4-31B W4A16, thinking off)
  G0   file selection: repo tree + issue → top-10 files              (shared by every LLM arm and π; M§3.2, L40)
  G1   issue + skeleton(G0 files, ≤6k) → k=10 imagined paths:
         STEP i: <symbol> | +E1,+E2 ; -E3 ; ~value      (+ reported, − expected-but-missing, ~ wrong value)
  π    restart/initial: 0.5·softmax(BM25 restricted-boost to G0 files) + 0.25·uniform + 0.25·CodeRankEmbed(issue)

DECODER (CPU, ms per task)
  profile-HMM with insert states; site-propensity-weighted transitions; ε-skip only over wraps/decorator edges
  and sub-hub intermediates; teleport ρ to π; BOTH orientations decoded, max likelihood kept (L6)
  emission = name LR (unnormalised over v; L10) + POLARITY-AWARE EFFECT LR vs base rates   ◄── novel piece
  readout = Σ_t w_t γ_t(v) / π_∞(v)^λ  with culprit-end weights; mean over k                  (L9, L11)
  ─► full-repo ranking (non-test, non-vendored functions + scope states)
```

**Works with no tests at inference:** all of the above. HEAP needs an importable install and runs module top-level code inside a sandbox (network off, isolated HOME/tmp). Repos whose import fails use G-a plus G-c only, reported per repo.

**Needs execution:** CAL training labels, and the executed measurement subset (M1, P1).

---

## 3. Exact objectives

### 3.1 Effect alphabet and labels (L15, L16, L19, M§2.2)

**Alphabet, B = 15:**
- raises one of {KeyError, IndexError, AttributeError, TypeError, ValueError, AssertionError, NotImplementedError, RuntimeError, OSError, repo-defined};
- returns None;
- returns self;
- mutates an argument;
- writes a self attribute;
- warns (any category).

**Two further fields:**
- I/O-or-log is kept as an AST-only feature, never a label.
- Module-global write is **AST-only**, since the dict-version diff cannot be implemented (M§2.2).

**Exception closure.** Exceptions are closed under MRO. A repo exception subclassing ValueError fires both its own bit and ValueError. `except` coverage uses issubclass (L19).

**Labels (CAL-train only).** **Possibility**: bit = 1 if observed in ≥2 distinct test functions. A frequency bit is recorded but unused (L15).

**Fingerprints.** Mutation and self-write labels come from shallow fingerprints. They are biased low, so a 50-function manual audit is reported (M§2.2).

**Tracer.**
- CAL-train is restricted to Python ≥3.12 repos, using `sys.monitoring` RAISE/PY_UNWIND.
- The executed EVAL subset uses `sys.settrace` with call and exception events, `f_trace_lines=False` (M§2.1).
- A cross-tracer agreement check runs on one CAL repo.
- Aggregation happens in-process, with no raw event logs (M§2.3).

### 3.2 Predictors (CPU)

| Model | Form | Target | Loss | Notes |
|---|---|---|---|---|
| g_φ | Per-bit logistic regression (L2) on AST features (raise sites with MRO class, return forms, param/self stores, warns calls, try/except coverage) + HEAP context (wrapped-by, dispatch target, overrides) | Possibility bits | Class-weighted BCE | Inverse call-count propensity weights (L21); leave-one-repo-out across CAL-train; isotonic on the left-out repo |
| h_ψ | Logistic: P(site executed \| site features) | CAL traces | BCE | Gives p(s \| u) ∝ h_ψ(s), so rarely executed sites stop diluting transitions (L7) |
| f_θ (CHAIN-lite) | Site-normalised softmax over C(s) ∪ {∅} | Membership 1/\|Obs(s)\| at sites executed ≥5× | Cross-entropy | Cited to AFLGopher/Pereira; not claimed |

**Effect-bit test, C2b at bit level** (M§4.1). Leave-repo-out macro-average-precision of g_φ minus the deterministic AST-rule predictor, over thousands of functions. Paired bootstrap over functions, clustered by repo.

### 3.3 Emission (the novel piece), in log-ratio form against a background b₀

For step t, observation o_t = (n_t, E⁺_t, E⁻_t, value_t). For state v:

```
ℓ_v(o_t) = α · τ · sim(n_t, v)
         + β · Σ_{j∈E⁺_t} log[ ((1−η) q̃_vj + η) / q̄_j ]                                (reported effect, IDF-like; L14)
         + β · Σ_{j∈E⁻_t} { log[ (1 − q̃_vj + η) / (1 − q̄_j + η) ] + log[ (peer_j(v) + η) / (q̄_j + η) ] }   (contract; L16a)
         − μ · Σ_{j ∉ E⁺_t∪E⁻_t, j ∈ exc} 1[q̃_vj > 0.8] · log(q̃_vj / q̄_j)               (weak unstated penalty; L14)
```

| Symbol | Definition |
|---|---|
| sim | max of (a) guess-length-normalised subtoken containment \|n ∩ q_v\|/\|n\| with a length prior, and (b) cosine(CodeRankEmbed(n), CodeRankEmbed(signature+docline of v)). **Not softmax-normalised over v**, so it is a likelihood, not a posterior (L1, L10). Scope states use hierarchical qualnames |
| q̄_j | Repo base rate: the mean of q̃_·j over the repo's non-test functions |
| peer_j(v) | Mean q̃ over v's contract peers: the HEAP override family, other candidates at v's call sites, and same-name methods in sibling classes. "Siblings raise, v does not" is the "should raise but doesn't" signal |
| ~value | Emits nothing: neutral, but stratified (L16c) |

**Parameters.**
- Fixed a priori: α = 1, η = 0.05, μ = 0.1.
- **Tuned (4): β, τ, ρ, λ** (M§4.4: ≤4 tuned).

### 3.4 Transitions and decoding (L3, L6–L13)

**States:** functions, scope states, ∅, and an **insert** copy (v, ins) per visited v (profile-HMM style). An insert emits background and keeps the position, replacing v1's stutter δ.

**Transition from u:**
- (1 − η₀ − ε − ρ) · Σ_s p(s | u) · p(v | s), with p(s | u) ∝ h_ψ(s);
- \+ η₀ → (u, ins);
- \+ ε · skip along `wraps`/decorator edges, or through intermediates below the 95th in-degree percentile only (L9);
- \+ ρ · π.

**Special cases.**
- ∅ → v for callback-registered v, weighted by registration count (L3).
- Self-loop = max(A₁(u→u), 0) only. No added stutter (L13).

**Fixed a priori:** η₀ = 0.15, ε = 0.05, and the π mix 0.5/0.25/0.25 (L8).

**Decode.** Forward–backward in both orientations per sample; keep the higher-likelihood one (L6).

**Readout:**
- score(v) = (1/k) Σ_paths Σ_t w_t γ_t(v) / π_∞(v)^λ, where w_t ∝ (t/T)² puts weight on the culprit end, and π_∞ is the stationary distribution of the transition-only chain.
- Secondary readouts: the final-step posterior; a "predecessor" readout scoring callers of the posterior mode (missing-branch bugs, L5a); and gold rank excluding the decoded entry (L11).

**Effective-structure diagnostic** (L12). Per item, KL(BMM posterior ‖ X-BAG⊗PPR score distribution).
- Pre-registered: if the median KL is below 0.05 nats, C-DEC is reported as "bag + prior" whatever its sign.
- The tuned ρ and λ, and the median A₁ on executed true paths, are also reported.

### 3.5 Generator prompts

- **G0:** repo tree (paths only) + issue → 10 files, ranked.
- **G1:** issue + skeleton of the G0 files (module→class→def(sig)+first doc line, ≤6k tokens; truncated by G0 rank, then file order) + the alphabet with glosses.
  - Output grammar: `STEP i: <dotted symbol> | +codes ; -codes ; ~value`, 2–8 steps.
  - k = 10 (2 seeds × 5), temperature 0.7.
  - Invalid lines are dropped and logged.
- **No LoRA** anywhere in v2. Study B and every training line are cut.

---

## 4. Data

### 4.1 Pools

| Pool | Source | Execution? | Size | Role |
|---|---|---|---|---|
| **EVAL** | Verified ∪ Lite ∪ SWE-rebench ∪ Loc-Bench. Excludes the dev repos (fastapi, rich, requests, httpx + httpcore), CAL repos and vendored/generated paths | **No.** Gold = patch hunks mapped statically to functions or scope states | **≥600 traceback-free, ≥25 repos, ≤40 per repo** (M§1.4, L33) | Headline |
| EVAL-pilot | 50 EVAL tasks, burned | No | 50 | SDs and power only (M§4.4) |
| **EVAL-exec** | 120–150 EVAL tasks from SWE-bench images, rootfs extracted with `crane export` (no Docker), plus modern rebench pure-Python tasks | Yes: F2P fail→pass ×3 with pinned PYTHONHASHSEED; settrace | 120–150 | M1, P1a/b/c, edge recall, on/off-path |
| **CAL-train** | ~10 Python ≥3.12 repos with good suites, including 2–3 large ones (M§4.4); disjoint from EVAL, dev and CAL-tune | Yes, full suites | ~10 repos | g_φ, h_ψ, f_θ |
| **CAL-tune** | ~100 SWE-rebench/SWE-Gym tasks from **other** repos, disjoint from CAL-train (L28) | Static gold; ~40 executed for P1 | 100 | Nested 5-fold CV tuning of every arm (L39) |
| TB | 150 EVAL-pool tasks *with* tracebacks | No | 150 | Secondary: CrashLocator-style baseline (PA §4); optional rung |
| DEV | 118 competition dev tasks | — | 118 | Untuned in-domain readout |

### 4.2 Ground-truth rules

| Rule | Detail | Break |
|---|---|---|
| Traceback-free | No `Traceback (most recent call last)`, no `File "…", line N`, no pytest `E ` dumps, and no bare `^\w+(Error\|Exception\|Warning): ` lines. Bare-line issues form an "exception-line" stratum. The joint distribution traceback-free × effect-stated × new-code is reported before n is fixed | L34 |
| Gold | Hunks are mapped to existing functions or to **scope states** (class body, module top level). A **non-function-hunk** stratum is reported | L1 |
| New code | Tasks whose gold is entirely new code are **excluded from the headline**, not pooled | L5b |
| Test files | Defined by the repo's discovery config (`testpaths`, `python_files`, django `tests/`), not by substrings. `_pytest`, `django/test` and `sympy/testing` golds stay rankable | L32 |
| Executed-subset path | Base failures are classified as collection/import, binding, assertion, or in-repo exception. Binding failures count as on-path at the gold. The gold invocation is chosen by **differential trace** (subtree or return differs between base and gold-patched runs), falling back to the last entry before the failing assertion. Off-path = no gold function entered | L29, L30, L5c |
| M1 alignment | **Suffix-first** from the gold; valid-walk scoring. Per-step precision vs path recall, as in RepoReasoner (PA §4) | L31 |
| Renaming | **R2** (code-only, all 600) and **R1** (150 tasks; issue renamed **only inside code spans**, with the collision rate reported). File and module paths are not renamed. A per-repo **residual-name share** (dunders, external overrides, string dispatch, literals) is published, and C4 is also reported on low-residual repos | L18 |
| Strata | Defined from gold or executed facts, never from G (L27). T ≥ 3 stratum (L4); gold-in-skeleton (L40); polarity strata (+ / − / ~value) from G1 output, not from gold (L16); family-size and hub strata (L9, L10); post- vs pre-cutoff with an honest MDE (M§4.5) | L4, L9, L10, L16, L27, L40 |

---

## 5. Arms

### 5.1 Baselines

| ID | Baseline | Notes |
|---|---|---|
| **B-DIRECT\*** | Best-of-3 prompt-only rankers, chosen on CAL-tune by nested CV: (i) direct top-10 culprit list; (ii) LocAgent-style "reconstruct the execution flow, then list"; (iii) the final steps of G1's own paths as a list (= B-NAMES). All use the same G0 and skeleton, and are tail-filled by BM25 | **C-USE comparator** (M§5.2). This is also the "prompt-only best-prompt" baseline |
| B-RERANK | Gemma listwise rerank over the top-50 of RRF(BM25, CodeRankEmbed), window 20, tail-filled | SweRankLLM-style |
| B-BM25 | Function-level BM25 over identifier subtokens + docstring + body | Graph-free, LLM-free |
| B-CRE, B-SRS | CodeRankEmbed; SweRankEmbed-Small, CPU | Check SRS's licence: baseline use only, never inside BMM (M§5.1) |
| B-PPR | PPR from π on G, with no imagined path | Secondary contrast: value of the imagination |
| B-EULER | Greedy alignment: each step → its best node by the same ℓ; gaps filled by shortest path on G; rank by culprit-weighted visits | PA §4. One change vs BMM: greedy align-and-fill in place of exact inference |
| B-LOCAGENT / CoSIL / SweRank / OrcaLoca / Agentless | Published function-level Acc@k on the matched Lite subset, with the non-comparability stated (pools vs full repo; traceback filter). A CoSIL run is an optional rung | Cited, not run by default |
| B-CRASH | CrashLocator-style stack expansion on G, **TB stratum only** | PA §4; optional rung |
| CERT / Z | **Not meaningful.** They are agent-trajectory credit signals for RL training (`PEFT RL…/architecture_v3_final.md` §235). They produce no function ranking, and v2 trains nothing. Excluded | — |

### 5.2 One-change controls and ablations (each differs from BMM in one thing)

| ID | Change | Role |
|---|---|---|
| **X-BAG⊗PPR** | Same G0, G1 observations, ℓ emissions, w_t readout weights and hub normalisation. Transitions replaced by a PPR prior: score(v) = κ·log PPR_π(v) + Σ_t w_t ℓ_v(o_t) | **C-DEC control** (L35, L12). It is also the BER fallback |
| X-NOEFF+BODY | β = 0, with the name term augmented by BM25 over **body tokens** (renamed in R2) | **C-BEH control 1** (L17) |
| X-LEXEFF | Effect term replaced by a binary "body contains `raise C` / `except C` / `warnings.warn` for a class C named in the step", with no learning or execution, same polarity handling | **C-BEH control 2** (L17) |
| X-ASTEFF | q from deterministic AST rules in place of g_φ | C2b task-level estimate (descriptive) |
| X-NOPOL | E⁻ treated as E⁺ (v1 behaviour) | Polarity ablation |
| X-SHUF | Steps shuffled (T ≥ 3 stratum) | C3 |
| X-DPR | In/out-degree- and module-preserving rewire; **p(c\|s) carried with the edge** (L26). Repos with >10% edge overlap are excluded from this contrast; overlap and surviving true-path edges are reported | C3, degree-preserving control |
| X-NULL | Paths imagined **for a different issue in the same repo** (L37), in place of the random path | Matched null |
| Graph-free | X-BAG without PPR: names+effects over all nodes | Graph-free control |
| X-UNIF / X-STAT | Uniform p(c\|s); AST-only graph without HEAP. Both report true-path edge recall and mean log A on executed paths (L25) | Infrastructure readouts |

---

## 6. Claims, effect sizes, honest power

**Statistics.**
- Primary inference is the **repo-clustered bootstrap**: 10,000 resamples of repos, then tasks.
- The repo-macro mean is co-reported.
- Primary metric pair: **mean log₂-rank** (resolution in the tail) and **MRR** (top-heavy) (M§4.2).
- Holm correction over three primaries.
- C-USE adds a two-level bootstrap (tasks, then samples) (M§4.5).
- Equivalence statements use TOST margins of ±0.01 MRR and ±0.1 log₂-rank.
- All SDs come from the burned 50-task EVAL pilot.

**Power assumptions.**
- n = 600, ~25 repos, ≤40 per repo, ICC 0.05.
- Design effect ≈ 1 + 23·0.05 ≈ 2.15, so effective n ≈ 280.
- The SDs are GUESSED and replaced by pilot values.

| Claim | Contrast (all BM25-fused by RRF, one variant only) | Predicted | SD(Δ) assumed | Power at Holm α = 0.0167 | Decision rule |
|---|---|---|---|---|---|
| **C-BEH (core)** | BMM − X-NOEFF+BODY **and** BMM − X-LEXEFF, **under R2**, on the effect-stated stratum (≥1 informative ± effect in ≥3 of 10 samples, defined from G1 output; est. 200–300 tasks, n_eff ≈ 120) | Δlog₂-rank −0.25; ΔMRR +0.03 | 0.9 / 0.15 | log-rank ≈ 0.75; MRR ≈ 0.45 | Pass iff **both** log-rank CIs exclude 0 (intersection–union, no extra α) and both MRR lower bounds > −0.01 |
| **C-DEC** | BMM − X-BAG⊗PPR, all traceback-free, original names | Δlog₂-rank −0.3; ΔMRR +0.03 | 1.0 / 0.15 | log-rank ≈ 0.99; MRR ≈ 0.8 | Pass iff the log-rank CI excludes 0, the MRR lower bound > −0.01, and median KL ≥ 0.05 |
| **C-USE** | BMM − B-DIRECT\* | Δlog₂-rank −0.4; ΔMRR +0.02 | 1.5 / 0.30 | log-rank ≈ 0.95; **MRR ≈ 0.2 (MRR is non-inferiority only)** | Pass iff the log-rank CI excludes 0 and the MRR lower bound > −0.02 |
| C2b (bit-level) | g_φ − AST rules, leave-repo-out macro-AP | +0.03 | — | High (thousands of functions) | Fail → drop "execution-trained" from the sentence and use AST q. The architecture survives |
| C3 | X-SHUF, X-DPR, X-NULL on T ≥ 3 | −0.15 log-rank each | — | Moderate | TOST equivalence → "order or structure carries nothing" |
| C4 | R2 relative loss of BMM ≤ ½ of BM25's | — | — | Descriptive | — |
| M1 | Per-step precision ≈ 0.6–0.8, path recall ≈ 0.2–0.4 (prior: RepoReasoner 0.80 / 0.31) | — | — | Descriptive on EVAL-exec | — |

**What is honestly underpowered.**
- MRR for C-USE (≈0.2).
- MRR for C-BEH (≈0.45).
- Any contamination split (post-cutoff n ≈ 100: MDE ≈ 0.1 MRR).
- C3 on small T ≥ 3 strata.

These are reported as estimates with CIs and are not claimed.

**Positive controls (L36).**
- **P1a, decoder sanity:** the oracle executed path projected onto G, with 50% of names substituted from the **measured** hallucination distribution (sibling / wrong-module / non-existent, measured on CAL-tune G1). Gate: hit@5 ≥0.80 on EVAL-exec. Failing it means a decoder bug, so EVAL decoding is not read until it is fixed. It is not a paper kill.
- **P1b, graph adequacy:** true-path edge recall of G, overall and through ∅. Reported, not gated.
- **P1c, realistic ceiling:** P1a with the full measured M1 error profile injected.
- **P2, behaviour:** gold-path F2P-traced effects + G1 effects → effect-only score; MRR reported on effect-stated tasks.

---

## 7. Kill table

The CPU kills run in week 1, before any GPU spend. GPU kills follow.

| # | When | Test | Metric | Threshold | Cost | Pass | Fail | Ambiguous |
|---|---|---|---|---|---|---|---|---|
| K0 | Day 1–2 | EVAL pool count with static gold | Traceback-free, non-new-code, ≤40/repo | ≥600 tasks and ≥25 repos | CPU 4 kh, 1 pd | Proceed | <400: n = whatever is available, MDEs recomputed and stated | 400–600: cap 50/repo, add Loc-Bench |
| K1 | Day 1–2 | Rootfs extraction probe on 3 SWE-bench images | F2P fail→pass from extracted env | ≥2/3 | CPU 2 kh | EVAL-exec from images | EVAL-exec = 60 modern rebench tasks via pip; M1 weakens | — |
| K2 | Day 2 | CAL-tune availability (static) | Tasks from ≥6 repos disjoint from CAL-train | ≥100 | 0.2 pd | Proceed | 60 tasks + nested CV | — |
| K3 | Day 3–4 | Effect-statement proxy on EVAL issues (regex/keyword, ± polarity) | Share stating ≥1 informative effect (base rate <10%) | ≥20% | CPU ≈0 | Proceed | **<10%: C-BEH cannot reach power. Declare collapse risk; the headline becomes C-DEC + M1 (closeness 2), stated honestly before GPU** | 10–20%: C-BEH n shrinks; MDE stated |
| K4 | Week 1 | C2b bit-level | g_φ − AST macro-AP, leave-repo-out | CI > 0 and ≥ +0.03 | CPU 10 kh | Keep "execution-trained" | Use AST q in BMM; drop the phrase. Core unaffected | Point >0 with CI ∋ 0: keep g_φ, no claim |
| K5 | Week 1 | Graph adequacy (P1b) on EVAL-exec | True-path edge recall of G | ≥0.6 | CPU 5 kh | Proceed | <0.4: extend Jedi to all sites; C-DEC is expected weak, stated | 0.4–0.6: report; teleport mix unchanged (pre-registered) |
| K6 | Week 1–2 | P1a with uniform-sibling noise (the measured profile follows K9) | hit@5 | ≥0.80 | CPU | Proceed | Debug the decoder; no EVAL GPU until it passes | — |
| K7 | Week 1 | DPR feasibility | Repos with ≤10% edge overlap | ≥15 repos | CPU | C3-DPR on those repos | DPR reported descriptively | — |
| K8 | First GPU session | P0v vLLM batched throughput | Aggregate decode tok/s | ≥250 | 0.4 NH | k = 10 | k = 5 (pre-registered) | — |
| K9 | GPU, CAL-tune | G0+G1+B-DIRECT\* on 100 CAL-tune tasks; nested CV; hallucination profile; P1a re-run with measured noise | CV log₂-rank of BMM vs X-BAG⊗PPR and vs B-DIRECT\* | BMM not worse than both by >0.3 log-rank | ≈0.5 NH | Proceed to EVAL | Worse than both: EVAL still runs (the measurement needs it), headline pre-switched to the BER or measurement row of §0.5 | Mixed: proceed, no switch |
| K10 | GPU, pilot | 50-task EVAL pilot (burned) | SD(Δ) per primary | Within 1.3× of assumed | ≈0.3 NH | Keep §6 power | MRR claims become estimate-only; log-rank stays primary | — |
| K11 | GPU, EVAL | C-BEH, C-DEC, C-USE | §6 rules | §6 | G1, G2, B4 | §0.5 row 1 | §0.5 rows 2–4 | — |

---

## 8. Budget

### 8.1 GPU (NH; throughputs as recounted in M§3)

| Line | Content | Opt | Central | Pess |
|---|---|---|---|---|
| P0v | Batched throughput probe, folded into the first session | 0.3 | 0.4 | 0.5 |
| G0 | File selection, ~870 tasks (600 + 50 pilot + 118 DEV + 100 CAL-tune) | 0.4 | 0.6 | 0.9 |
| G1 | Imagined paths, ~870 tasks × k = 10 | 1.8 | 2.7 | 4.2 |
| G2 | R2 (600 × k = 10) + R1 (150 × k = 5). G0 is reused, since paths are unrenamed | 1.1 | 1.6 | 2.4 |
| B4 | B-DIRECT\* (3 variants on CAL-tune, best on EVAL, original + R2) + B-RERANK | 0.8 | 1.3 | 1.9 |
| G4 | Second generator: Qwen3-32B-AWQ, 300 tasks (AWQ ≠ W4A16 confound stated) | 0.8 | 1.2 | 2.0 |
| G5 | Gemma-4-E4B generator, 150 tasks | 0.2 | 0.3 | 0.5 |
| R | Reserve | 1.5 | 1.8 | 2.8 |
| | **Core total** | **6.9** | **9.9** | **15.2** |
| opt | TB stratum generation (150 tasks) | +0.3 | +0.4 | +0.6 |
| opt | CoSIL run, 100 tasks, hit@k | +1.2 | +2.0 | +3.0 |
| opt | LATENT kill only (P0 + 1.2M tokens + transfer check), only if ≥8 NH remain after G2 | +1.5 | +2.4 | +3.5 |
| | **Everything** | 9.9 | 14.7 | **22.3 ≤ 25** |

### 8.2 CPU (4-vCPU kernel-hours; free, but each session needs babysitting)

| Line | Opt / Central / Pess |
|---|---|
| Import-only installs (HEAP) + rootfs extraction | 18 / 36 / 72 |
| EVAL-exec validation ×3 + traces | 15 / 30 / 50 |
| AST resolver + Jedi fallback, ~700 snapshots (file-hash cache) | 25 / 50 / 100 |
| HEAP walks | 20 / 40 / 80 |
| CAL-train suite traces | 6 / 18 / 48 |
| Embeddings (CRE, SRS × orig/R1/R2) | 7 / 14 / 40 |
| Renaming | 5 / 10 / 20 |
| Decoding, all arms, nested CV | 10 / 30 / 70 |
| **Total** | **106 / 228 / 480** |

At 5 concurrent kernels, that is ≈21 / 46 / 96 wall-hours.

### 8.3 Person-days (5 h; before the Nov 5 freeze)

| Work | Opt | Central | Pess |
|---|---|---|---|
| Day-1 counts (K0, K2), Pathidea abstract read, rootfs probe (K1) | 0.8 | 1.0 | 1.5 |
| EVAL builder: static gold, scope states, strata, test-file rules, vendored exclusion, G0 tree | 1.2 | 1.5 | 2.0 |
| Import installs, HEAP walk, django/sphinx bootstrap, callbacks | 1.5 | 2.0 | 3.0 |
| AST resolver + Jedi fallback + graph assembly | 1.0 | 1.5 | 2.0 |
| Tracer (monitoring + settrace), CAL traces, EVAL-exec differential traces | 2.0 | 2.5 | 3.5 |
| g_φ / h_ψ / f_θ, possibility labels, MRO, peers, wrappers, leave-repo-out | 0.8 | 1.0 | 1.5 |
| Decoder, controls, Euler, PPR, nested CV | 1.5 | 2.0 | 2.5 |
| Prompts + vLLM batch scripts + Kaggle sessions | 1.2 | 1.5 | 2.0 |
| Renaming (R1 code spans, R2, residual share) | 0.8 | 1.0 | 1.5 |
| Embedding baselines | 0.3 | 0.5 | 0.8 |
| Stats, figures, kill reports | 0.8 | 1.0 | 1.5 |
| **Pre-freeze total** | **11.9** | **15.5** | **21.8** |
| Writing (Nov 5–12, after the freeze) | 1.5 | 2.0 | 2.5 |

### 9. Descoping ladder (the core claim is never dropped)

**Core, never cut:**
- static EVAL ≥ 400;
- G0 + G1 + R2;
- B-DIRECT\* and B-BM25;
- X-BAG⊗PPR, X-NOEFF+BODY, X-LEXEFF;
- the clustered statistics.

**Cut order** when pd or NH run over (projected pd > 16, or NH > 1.3× central after G1):

1. Optional rungs never start: LATENT, CoSIL, TB/B-CRASH.
2. Drop G5 (E4B).
3. Shrink G4 to 150 tasks. Generality then rests on ≥25 repos plus a second model at 150.
4. Drop R1; keep R2.
5. Drop B-RERANK, and cite SweRank instead.
6. EVAL-exec 150 → 60. M1 and P1 become weaker but stay.
7. k = 10 → 5.
8. Drop the Jedi fallback.
9. Drop HEAP; use the AST-only graph. The core claim is unaffected, and X-STAT becomes moot.
10. g_φ → AST q. C2b is dropped, and the core claim stands on X-LEXEFF / X-NOEFF+BODY.

**Floor:** ≈4.5 NH, ≈10 pd. It still tests C-BEH, C-DEC and C-USE on ≥400 tasks.

---

## 10. Generality plan

| Axis | Plan | Prediction |
|---|---|---|
| Repos (primary) | ≥25 repos, ≤40 each, repo-macro reporting | Sign holds in ≥60% of repos with ≥15 tasks |
| Second model | Qwen3-32B-AWQ as generator only; everything else frozen | C-DEC and C-BEH hold in sign. The decoding gain is larger for the weaker-naming generator |
| Scale | Gemma-4-E4B generator | Same direction; larger relative gain from structure |
| Language | **Dropped** (M§6: 7–10 pd for JS). Scope stated: the method needs a tracer, an import or approximate-interpretation builder (Jelly exists for JS), and an effect alphabet | — |
| Release | Generations, graphs, effect tables, traces and splits as a Kaggle dataset, so the headline is reproducible without rebuilding environments (M§6) | — |

---

## 11. v1 → v2 change log

### 11.1 Logic breaks (L1–L46)

| ID | Sev | v2 action |
|---|---|---|
| L1 | Major | Scope states; containment score with a length prior; non-function-hunk stratum |
| L2 | Major | Hallucination types measured on CAL-tune; P1a uses the measured noise |
| L3 | Major | ∅ → v callback transitions from HEAP and AST; share of paths through ∅ reported |
| L4 | Major | C3 only on T ≥ 3; C-DEC also reported by path length |
| L5 | Major | (a) predecessor readout; (b) new-code excluded from the headline; (c) off-path redefined |
| L6 | Minor-major | Both orientations decoded; max likelihood kept; reversed share reported |
| L7 | Major | Stutter removed. Insert states and p(s\|u) ∝ h_ψ. Median A₁ on true paths reported |
| L8 | Major | Teleport mix π/uniform/embedding; true-path edge recall (K5, P1b) |
| L9 | Major | Skip restricted to wraps or sub-hub intermediates; hub normalisation π_∞^λ applied to every arm; hub share in the top 10 reported |
| L10 | Major | Likelihood-form name score, not normalised over v; family-size stratum |
| L11 | Fatal | Culprit-weighted readout; "excluding entry" diagnostic |
| L12 | Fatal | KL effective-structure diagnostic with a pre-registered threshold; tuned ρ and λ reported |
| L13 | Minor | Self-loop capped; dispatch treated as one site |
| L14 | Major | Base-rate likelihood ratio; unstated-effect penalty μ; LOC correlation reported |
| L15 | Fatal | Possibility labels (≥2 distinct tests) |
| L16 | Fatal | ± polarity in the grammar; contract (peer) score for −; ~value neutral; polarity strata |
| L17 | Fatal | X-LEXEFF and X-NOEFF+BODY; C-BEH must beat both under R2 |
| L18 | Major | R1 renames only code spans in the issue; residual-name share published; C4 on low-residual repos |
| L19 | Major | MRO-closed exceptions; issubclass for except coverage; informativeness stratum |
| L20 | Major | One-hop warn/I-O wrapper propagation. g_φ AUROC per repo reported on CAL hold-out and EVAL-exec |
| L21 | Major | Inverse call-count weights; calibration binned by call count on EVAL-exec |
| L22 | Minor-major | **Skipped.** E-TR is cut (M§2.3) |
| L23 | Major | Django and sphinx bootstrap recipes; import success reported per repo |
| L24 | Minor-major | Static pass over function-local imports; touch `__all__`; plugin edge kind |
| L25 | Major | X-STAT reports edge recall and mean log A separately |
| L26 | Fatal | DPR carries p(c\|s); ≤10% overlap rule; surviving true-path edges reported |
| L27 | Minor | Strata from gold or executed facts only |
| L28 | Fatal | CAL-train and CAL-tune split by repo; leave-repo-out g_φ |
| L29 | Major | Base-failure classification; binding failures counted on-path |
| L30 | Major | Differential-trace gold invocation |
| L31 | Major | Suffix-first alignment; valid-walk scoring |
| L32 | Major | Test files by discovery config; pytest inner sessions traced separately |
| L33 | Minor-major | PYTHONHASHSEED pinned, 3 runs; ≤40 per repo; clustered primary |
| L34 | Major | Bare exception lines stripped into their own stratum; joint distribution before fixing n |
| L35 | Fatal | X-BAG⊗PPR is the C-DEC control; B-PPR is secondary |
| L36 | Major | P1a/b/c split; P1b not gated |
| L37 | Minor | X-NULL = paths imagined for another issue in the same repo |
| L38 | Major | C3 read only after L4, L6, L7, L11 and L26 are fixed; T ≥ 3 and overlap rules |
| L39 | Minor-major | Nested 5-fold CV; ≤4 tuned parameters; SDs from the EVAL pilot, not CAL-tune |
| L40 | Major | Explicit G0 shared by all LLM arms; gold-in-skeleton stratum |
| L41 | Minor | Vendored and generated code excluded by a fixed rule |
| L42–L46 | Fatal–Minor | **Skipped.** Study B is cut (M§7). If the optional LATENT kill runs, L42 (matched negatives), L43 (shared-name/literal subset), L44 (v-before-u conditioning, function-level shuffle), L45 (context ablation) and L46 (ECE + Platt) are mandatory, and T-LAT is not run |

### 11.2 Method breaks

| Item | v2 action |
|---|---|
| §0 Scope vs calendar | 15.5 pd central, 9.9 NH central |
| §1 Data feasibility | Static-gold EVAL ≥600, ≥25 repos, ≤40 each; EVAL-exec 120–150 via rootfs extraction (K1 probe) |
| §2.1 setprofile cannot see exceptions | settrace on <3.12; CAL on ≥3.12 monitoring; cross-tracer check |
| §2.2 Overhead, global bit, fingerprints | Global bit AST-only; 50-function audit; site-level counting only |
| §2.3 E-TR volume, raw logs | E-TR cut; in-process aggregation |
| §3.1 Throughput | P0v probe; k fallback to 5 |
| §3.2 Skeleton cannot fit 6k | Explicit G0, shared, reported as a stage |
| §3.3 Recount | §8.1 uses the breaker's rates |
| §4.1 C2b underpowered | Bit-level test; task-level descriptive |
| §4.2 MRR resolution | Log₂-rank co-primary |
| §4.3 Clustering | Clustered bootstrap primary; repo-macro; ≥25 repos |
| §4.4 Tuning leakage, small CAL, ≤4 params, one C1 variant | Cross-fit, leave-repo-out; 2–3 large CAL repos; burned pilot; Day-1 CAL-tune count; one fused variant per claim |
| §4.5 Two-level bootstrap, TOST, contamination | Adopted; post-cutoff MDE stated as ≈0.1 |
| §5.1 Licence, CoSIL readout | SRS baseline only; CoSIL optional with hit@k |
| §5.2 B-DIRECT, reranker, LocAgent, published numbers | All added; B-DIRECT\* is the C-USE comparator |
| §6 Rubric: JS, Study B, word budget, release | JS and Study B cut; dataset release; the paper has one method, one main table, one ablation table and one M1 figure |
| S3 list | All adopted (E4B as its own line, R2 at k = 10, log\|V\| normalisation of τ and ρ) |

### 11.3 Prior-art items

| Item | v2 action |
|---|---|
| Novel piece = behavioural emissions (1.5) | §1 sentence reworded around it |
| Must-not-claim list (10) | §1.1 |
| Related-work order | §1.1 |
| Euler-style aligner | B-EULER |
| CrashLocator on the traceback stratum | B-CRASH on TB (optional rung) |
| M1 in RepoReasoner precision/recall form | §6 M1 |
| g_φ is NARROW | Claim only "cross-repo, multi-effect, calibrated", and only if C2b passes |
| "No tests at inference" wording | §1.1 item 9 |
| Pathidea abstract unread (could reach 2.5) | Day-1 read (A16) |
| Euler follow-ups (BURT, AdbGPT) unswept | A17 |
| TraceEval as a data competitor | Cited; not claimed |

---

## 12. Assumptions for the final breaker (A1–A20)

| # | Assumption | Check / consequence |
|---|---|---|
| A1 | 9.9 NH central fits beside the main track within the 30 h/week quota | HQ; ladder §9 |
| A2 | Batched vLLM decode ≥250 tok/s; prefill ≈1.5–2.5k tok/s; ≈0.25 NH overhead per session | K8 |
| A3 | ≥600 traceback-free, non-new-code tasks over ≥25 repos exist with static gold under the ≤40 cap | K0 |
| A4 | SWE-bench image rootfs extraction runs F2P on Kaggle CPU as root, without Docker | K1 |
| A5 | ~10 Python ≥3.12 CAL repos with good suites exist, disjoint from EVAL, including 2–3 large ones; ≥100 CAL-tune tasks from other repos | K2 |
| A6 | ≥20% of traceback-free issues state an informative ± effect | K3 (decisive for the core claim's power) |
| A7 | G1 emits parseable ± effects that agree with the issue (precision ≥0.7 on a 50-item manual audit) | Audit in week 2. If it fails, the C-BEH null is a parser failure, reported as such |
| A8 | The contract-peer score carries signal for "should X but doesn't" (peers exist for ≥50% of − steps) | Reported coverage |
| A9 | g_φ possibility labels transfer leave-repo-out (macro-AP over AST ≥ +0.03) | K4 |
| A10 | True-path edge recall of AST ∪ HEAP ∪ Jedi is ≥0.6 | K5 |
| A11 | HEAP imports ≥70% of modules for ≥70% of EVAL repos with bootstraps | Reported per repo; ladder rung 9 |
| A12 | Tuning on CAL-tune repos transfers to EVAL repos; log\|V\| normalisation suffices | DEV untuned readout; pilot |
| A13 | SDs: log₂-rank Δ 0.9–1.5, MRR Δ 0.15–0.30; ICC ≈0.05 | K10 |
| A14 | B-DIRECT\* at matched G0 is the strongest prompt-only comparator; a prompt not in the 3 variants does not dominate | Variant selection on CAL-tune; reported |
| A15 | Memorisation of Verified gold does not drive C-DEC or C-BEH, given R2, which the generator cannot see through | Pre- vs post-cutoff estimate (underpowered, stated) |
| A16 | **Pathidea** (TSE'22) reconstructs paths from *log snippets*, not from NL descriptions. If it uses NL, the overall closeness rises to 2.5, and §1 must say so | Day-1 read |
| A17 | No Euler follow-up (BURT, AdbGPT, 2020–26) pairs an LLM with probabilistic alignment and behavioural matching on an execution graph | Final breaker sweep |
| A18 | SweRankEmbed-Small's licence permits baseline use in a paper | Day-1 check |
| A19 | Qwen3-32B-AWQ can be staged offline (≤20 GB dataset) and served by vLLM 0.19 with thinking suppressed | G4; ladder rung 3 |
| A20 | 15.5 pd central is realistic, given that v1's precedent (`build_devset.py`) ran about 2× over | Track weekly; ladder triggers at a projected 16 pd |

No file other than this one was modified.
