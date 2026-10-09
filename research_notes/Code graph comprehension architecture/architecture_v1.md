# Architecture v1: Behavioural Map-Matching (BMM), 9 Oct 2026

**Role.** Architect. I read: the two scouts (G-An, E-An), `inventor_phase1_2.md` (P1–P6), `phase3_critic.md` (CR), `phase3_redteam.md` (RT), `docs/training_feasibility_4xl4.md`, the p0b row in `docs/experiments.md`, and the "What died" table of `reports/PEFT RL for directed graph reasoning.md`.

**Labels.**
- **VERIFIED**: read in a primary source by a scout, the critic or the red team, or read in the repo.
- **INFERRED**: my reasoning.
- **GUESSED**: a sizing or effect-size number that a gate must measure.

**Nothing in this note has been run.** Every effect size is GUESSED.

**Budget unit.** 1 NH = one Kaggle session-hour on the 4×L4 node. The study cap is 25 NH, including evaluation and controls. CPU kernels are free.

---

## 0. The decision: one architecture, one novel piece

### 0.1 The disagreement

The two Phase-3 reviewers recommended different papers.

| | Critic | Red team |
|---|---|---|
| Recommended paper | "Graph as decoder": static+HEAP → CHAIN-lite → imagined path → HMM decode with name+EFFECT emissions, with LATENT as the mechanism study | LATENT + EFFECT as the paper, HYTRACE as a measured ablation, HEAP/CHAIN as infrastructure |
| Reason | The parts are weak alone but fit together, and each has a one-change ablation in one harness | The HMM decoder (Newson–Krumm, 2) and plan-then-ground (RoG/GCR, 2) are already done. LATENT and EFFECT have no closeness ≥2.5 prior |

### 0.2 The resolution (INFERRED)

The two recommendations need each other.
- **EFFECT alone is too coarse to rank.** A 16-bit signature over about 4k functions leaves huge tie classes (CR §5 item 4). A *sequence* of effects along a connected path is far more selective, because tie classes multiply down along the path. Only a structural sequence decoder gives EFFECT the resolution to be a localisation technique rather than a coverage statistic.
- **HYTRACE alone is not novel.** Its decoder is map-matching (RT P3, closeness 2), and with name emissions it may reduce to "PPR from issue anchors, reranked by names" (CR §3 item 1). A name-free **behavioural emission** is the part that is not map-matching, not RoG/GCR and not GraphLocator. It is also the part that should survive identifier renaming.
- **LATENT is not usable at inference.** vLLM exposes no mid-layer states (CR §0.6). Inside the decoder harness it becomes the measured answer to "where should amortised execution knowledge come from?". Its effect head is one emission source, and its edge probe is one transition source. Both are paper-only arms, scored on the same items as everything else.

So the **architecture** is the critic's decoder, and the **novel piece** is the red team's least-crowded survivor (EFFECT), placed where it gains resolution. HYTRACE's imagined path, HEAP, CHAIN-lite and the HMM are known pieces, cited at the red team's closeness ratings. LATENT is Study B, a mechanism study inside the same harness. If Study B dies, the architecture does not.

### 0.3 The one novel piece, in one sentence

> **BMM decodes an LLM's imagined execution of an issue, a sequence of (symbol-guess, observable-effect) steps, onto a repository call graph with an HMM whose emissions score *behaviour*: per-function effect signatures predicted from source by a model trained on executed traces of other repositories. Execution-calibrated behaviour and structure, not identifiers, therefore correct the imagination, and no tests run at inference.**

### 0.4 Prior-art deltas for that sentence

Closeness numbers are the red team's or the critic's unless marked.

| Prior | Closeness | What it already does | Delta claimed by BMM |
|---|---|---|---|
| HMM map-matching: Newson & Krumm 2009; multiscale Viterbi path localisation on graphs, IEEE TSP'18 | 2 (decoder) | Decodes noisy observations onto graph paths by HMM | Reused, not claimed. The observations are LLM-*imagined* program steps, and the emission alphabet is behavioural |
| RoG (ICLR'24) / GCR (ICML'25) | 2 | The LLM plans a relation path, which is grounded on a KG (hard constraint in GCR) | BMM uses soft probabilistic grounding over a program graph. Transitions are calibrated on execution, and observations carry effects, not relation labels |
| GraphLocator 2512.22469, SWE-Debate 2507.23348, CoSIL 2503.22424 | 1.5–2 | Iterative LLM search over code graphs for fault chains | BMM makes one generation call, then runs exact inference on CPU. There is no agent loop. CoSIL is a measured baseline |
| EXCEPT 2201.00736; SBEST 2405.00565 | 1.5 | Localise by exception type or stack-trace distance | BMM uses learned, execution-calibrated *multi-effect* signatures and effects stated in issue prose, decoded as a sequence. It is evaluated on traceback-free issues, where SBEST-style trace proximity is unavailable |
| 2506.19045 (static trace estimation from a failure log, TSE'26) | 1.5 | Maps *observed* log output to an execution trace | BMM's observations are imagined from prose, and no log exists |
| SherLog (ASPLOS'10) and log-based path reconstruction | **Not searched by the red team; I estimate 2 (INFERRED)** | Infers execution paths from real log messages by static analysis | Same delta as 2506.19045. **Breakers must search this family (A20)** |
| AFLGopher 2511.10828; Pereira 2015 | 2 | Per-call-site softmax over targets, learned from traces | Infrastructure only (CHAIN-lite transitions). Cited, not claimed |
| Jelly approximate interpretation, PLDI'24; HeapDL, OOPSLA'17; PoTo, ECOOP'25 | 2–2.5 | Initialisation-time or heap facts complete static call graphs | Infrastructure only (HEAP). Reported as "a Python analogue of approximate interpretation" |
| 2605.25310 (tool-call edge probes); Troshin & Chirkova; Anand et al. ACL'24 | 2 | Directed pairwise edge probes with Hewitt–Liang and positional controls | Study B only. The new parts are the runtime-only vs static contrast and use of the probe as a decoder source |

**Combined closeness of the novel piece: 2 (INFERRED).** No closeness-3 prior was found for behavioural emissions in imagined-path decoding. Each half has a closeness-2 neighbour in another setting. The paper must say so.

### 0.5 What the paper claims, and what it does not

**Claims:**
- (C1) BMM beats PPR-from-anchors at matched graph, prior and tuning budget.
- (C2) Behavioural emissions add to name emissions, more so under renaming. Execution-calibrated effects beat AST-only effects.
- (C3) Order and structure carry the gain. Shuffled order, a degree-preserving rewired graph (DPR) and random paths lose it.
- (M) A measurement of how a 31B model's imagined runtime paths compare with executed paths, and where they break.

**Does not claim:**
- the HMM, plan-then-ground, per-site softmax or import-time resolution as new;
- resolve-rate gains (L1 lesson: function hit does not predict resolve);
- any training method.

---

## 1. Component diagram

```
                ┌───────────────────────── OFFLINE, ONCE (CPU, other repos) ─────────────────────────┐
                │  Calibration repos (repo-disjoint from eval)                                       │
                │   traced pre-existing test suites ──► per-site callee membership ──► CHAIN-lite f_θ │
                │                                   └──► per-function effect bits  ──► EFFECT g_φ     │
                │                                         (AST+HEAP features → P(effect bit))         │
                │  Tuning split (100 real tasks, calibration repos) ──► freeze α,β,τ,ε,ρ,δ,η          │
                └────────────────────────────────────────────────────────────────────────────────────┘
                                                 │ frozen f_θ, g_φ, hyper-params
 PER REPO SNAPSHOT (CPU, no tests run)           ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ G1 Static union: Pyright go-to-def per site ∪ PyCG ∪ PoTo (where they run)    │
 │ G2 HEAP: sandboxed import + heap walk → override/wraps/dispatch/reexport edges │  graph G (directed, typed)
 │ G3 CHAIN-lite: p(callee | site) = softmax_{C(s)∪∅} f_θ(features)               │──► transitions A
 │ G4 Node effects: q_v = g_φ(AST, HEAP ctx) ∈ [0,1]^B ; propagate exceptions ≤2h │──► effect table Q
 │ G5 Skeleton: module→class→def(sig)+1st doc line, tests/docs dropped, ≤6k tok   │
 └──────────────────────────────────────────────────────────────────────────────┘
 PER ISSUE                                    │ skeleton
 ┌────────────────────────────────────────────▼─────────────────────────────────┐
 │ P1 Generator (Gemma-4-31B W4A16, vLLM, zero-shot, thinking off, k=5 samples): │
 │    STEP i: <symbol guess> | <effects ⊆ alphabet or "none">   (2–8 steps)      │
 │ P2 Initial prior π: softmax(BM25(issue → node text)) ⊕ public-API prior        │
 └────────────────────────────────────────────┬─────────────────────────────────┘
                                              ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ D  HMM forward–backward on G (CPU, ms):                                        │
 │    states = repo functions ∪ {∅ external}; π; A (G3 + ε·2-hop + ρ·teleport)   │
 │    emission b_v(o_t) = name term (α) × behavioural term (β) over Q             │  ◄── THE NOVEL PIECE
 │    score(v) = mean over k paths of max_t γ_t(v); BM25 tie-break                │
 └────────────────────────────────────────────┬─────────────────────────────────┘
                                              ▼
                             full-repo function ranking (MRR, hit@k)

 PAPER-ONLY SOURCES (not at inference; same harness):
   E-src ladder for Q: AST rules → g_φ (default) → +pre-existing-test traces → LATENT effect head → oracle F2P trace
   A-src: + LATENT edge-probe scores on call-time-only candidate pairs (100-task subset)
 STUDY B (LATENT): HF forward (NF4) → pooled spans at layers ~15/30/45 → bilinear edge + effect probes
```

**Mapping to the track's three words:**
- *Generation*: G1–G3, a directed typed graph with execution-calibrated weights.
- *Parsing*: P1, issue → imagined execution with effects.
- *Embedding*: G4, a name-free, interpretable B-dimensional behavioural node embedding; plus the LATENT U/V projections in Study B.

**What runs without execution at inference.**
- Runs: G1, G2 (needs the environment *installed* and imports module top-level code, but runs no tests; sandboxed), G3, G4 (g_φ is trained elsewhere), G5, P1, P2 and D. This is the default BMM.
- Needs execution: the "+pre-existing-test traces" emission arm.
- Positive controls only: the oracle arms.
- Needs HF mid-layer states, so not deployable in the vLLM scorer: the LATENT arms.

---

## 2. Data and execution ground truth (Kaggle CPU)

### 2.1 Task pools

| Pool | Source | Use | Size target |
|---|---|---|---|
| **EVAL** | SWE-bench Verified ∪ Lite, minus psf/requests (a dev repo), plus SWE-rebench Python tasks from repos not in CAL and not in dev (fastapi, rich, requests, httpx, and httpcore as a sibling) | Headline | **≥300 traceback-free tasks with reproduced fail→pass** (≈1,200 candidates drawn; GUESSED survival 50% env × 65% traceback-free) |
| **CAL-train** | ~12 pure-Python repos with suites of ≥70% line coverage, repo-disjoint from EVAL and dev (e.g. attrs, marshmallow, boltons, more-itertools, toolz, jsonschema, pyparsing, tomlkit, babel, python-dateutil, pluggy, cattrs; final list chosen by the coverage gate) | Train f_θ (CHAIN-lite), g_φ (effects), LATENT probes | Traced suites |
| **CAL-tune** | 100 SWE-rebench tasks from CAL-train repos | Freeze BMM and control hyperparameters; positive control | 100 |
| **DEV** | The 118 gold-mapped competition dev tasks | In-domain readout only; never tuned on | 118 |
| **JS** (stretch) | SWE-PolyBench / Multi-SWE-bench JS tasks | Language generality | ~100 |

### 2.2 How executed ground truth is produced (CPU kernels, Internet on)

1. **Environment per (repo, version).**
   - `uv python install <ver>` and a venv. Dependencies come from the swebench harness specs (SWE-bench) or the SWE-rebench install recipes.
   - Environments are cached as tarballs and shared across tasks of the same version with `git checkout`.
   - No Docker.
2. **Validation.** Apply the test patch. Run FAIL_TO_PASS on the base commit; it must fail. Apply the gold patch; it must pass.
   - Keep a task only if both hold in **2 of 2** runs (flakiness filter).
3. **Tracer.**
   - Python ≥3.12: `sys.monitoring` (PY_START, CALL with instruction offset, RAISE, PY_UNWIND, PY_RETURN).
   - Older: `sys.setprofile` + `frame.f_lasti`.
   - Records: (caller code, call-site offset, callee code, count); escaping exception classes per frame; return class ∈ {None, self, arg-identity, new, other}; sampled (≤50 calls/function) shallow fingerprints of args and `self.__dict__` before and after (mutation, attribute writes); module-global writes via a dict-version diff; calls into `logging` / `warnings` / I/O builtins.
   - Exclusions (CR §1 item 7): callees in test files or `unittest.mock`, and sites executed only under `monkeypatch` / `mock.patch` contexts.
4. **Gold mapping.** Gold patch hunks are mapped to *existing* functions at the base commit.
   - If all hunks fall in *new* functions, the task joins the **new-code stratum** (scored at file and enclosing-class level only).
   - If a hunk adds a call into a function that is not on any executed path, the task joins the **off-path stratum**.
5. **True path.** In the failing base-commit F2P run, take the call stack at the first entry into a gold function. Strip test frames; the first repo-internal frame is the entry.
   - This gives on-path vs off-path (A6).
   - It also gives the reference path for the step-level imagination measurement (§4.4).
6. **Inference-honest effects and edges.** Trace the *pre-existing* tests: those not touched by the test patch, excluding F2P. These feed the "+traces" emission arm and Study B's strata on EVAL repos. **F2P traces are used only for oracle arms and scoring.**
7. **Strata computed on CPU before any GPU spend:**
   - **traceback-free**: no `Traceback (most recent call last)`, no `File "…", line N` frames, no pytest `E ` frame dumps;
   - **named**: the issue contains a gold function's identifier;
   - **effect-covered**: the parsed issue effects ∩ traced gold-path effects ≠ ∅;
   - **on-path / off-path / new-code**;
   - **polymorphic-site share** of path edges.

CPU cost is GUESSED at ≈10 CPU-min per task after environment reuse, so ≈200 CPU-h. Spread over 5 parallel 4-core kernels, that is ≈10–15 wall-hours (A4).

### 2.3 Renamed views (CR §0.5)

Rename views, never the executed code.
- HEAP, traces and edges are computed on the original code.
- **R1 (consistent):** rename repo-internal identifiers in skeleton, node text and issue alike, from one LibCST scope-aware table.
- **R2 (code-only):** rename only the code side; the issue is untouched. This is the realistic case.

The graph is isomorphic under renaming. AST-effect features are name-invariant by construction, except raised exception classes defined in the repo: these are renamed too, and builtin exception names are kept.

---

## 3. Exact objectives

### 3.1 CHAIN-lite transitions (CPU)

- **Model.** For each static ∪ HEAP site s with candidate set C(s): p(c | s) = softmax over C(s) ∪ {∅} of f_θ(x_{s,c}).
  - f_θ is a LightGBM ranker with lambdarank on grouped sites; the probabilities come from a temperature-calibrated softmax.
  - Features: resolution kind (direct / MRO / super / registry / reexport / fuzzy / builtin), |C(s)|, receiver-type certainty, module distance, HEAP support, test-helper flag.
- **Target: set membership, not frequency** (CR §6 fix). y_{s,c} = 1/|Obs(s)| for c ∈ Obs(s).
  - Only sites executed ≥5 times are used, to keep coverage holes out of the negatives.
  - Loss: −Σ_s Σ_c y_{s,c} log p(c | s).
- **One-change control arm:** per-edge BCE + post-hoc within-site renormalisation.
- **Function-level transition:** A₁(u→v) = mean over sites s∈u of p(v | s).
- **Full transition:** A(u→v) = (1−ε−ρ−δ)·A₁(u→v) + ε·A₁²(u→v) + ρ·π(v) + δ·1[u=v].
  - The ε skip covers decorator and wrapper truncation (G-A4); ρ is a teleport to the prior; δ is a stutter.

### 3.2 Effect predictor g_φ (CPU), the execution-calibrated part

- **Alphabet B = 16 bits:**
  - raises one of {KeyError, IndexError, AttributeError, TypeError, ValueError, AssertionError, NotImplementedError, repo-defined exception, other};
  - returns None;
  - returns self;
  - mutates an argument;
  - writes a `self` attribute;
  - writes module or global state;
  - emits a warning;
  - performs I/O or logging.
- **Labels.** A local bit is 1 if it is observed in ≥10% of a function's traced calls, using only functions with ≥5 traced calls (CAL-train).
- **Model.** Per-bit logistic regression (L2) on AST features (raise sites, `return` forms, subscript or attribute stores on params and self, `global` / `nonlocal`, calls to `warnings` / `logging` / `open`, try/except coverage) plus HEAP context (wrapped-by, dispatch target, overrides).
  - Loss: class-weighted BCE.
  - Isotonic calibration on a held-out CAL repo.
  - 3 seeds over repo splits.
- **Propagation (exceptions only), depth ≤2 along G:** q̃_{v,j} = 1 − (1 − q_{v,j}) · Π_{u ∈ callees(v)} (1 − A₁(v→u) · q̃^{(1)}_{u,j} · (1 − c_{v,j})), where c_{v,j} = 1 if v has an `except` covering class j.
  - Mutation and return bits are not propagated.
- **AST-only control.** Same alphabet, deterministic rules: q = 1 if the rule fires, else 0.05.

### 3.3 Generator prompt (zero-shot, no LoRA in v1)

- **Input:** issue text + skeleton (≤6k tokens) + the effect alphabet with one-line glosses.
- **Output:** 2–8 lines of the form `STEP i: <dotted symbol guess> | <comma-separated effect codes or none>`. The order is from the public entry point to the culprit.
- Thinking is off. Temperature 0.7, k = 5 samples per task, 3 sampling seeds.
- The same output provides names-only and effects-only observations by masking. No re-generation is needed for ablations.

### 3.4 Emission and decoding (CPU)

- **Observation:** o_t = (n_t, e_t). Emission:

  log b_v(o_t) = α · log s_name(n_t, v) + β · Σ_{j∈e_t} log((1−η) q̃_{v,j} + η)

  - s_name(n, v) = softmax over v of τ · max(subtoken Jaccard(n, qualname_v), cos(CodeRankEmbed(n), CodeRankEmbed(signature_v + docline_v))).
  - Unstated effects carry **no penalty**, because issues under-report.
  - The ∅ external state emits stdlib and third-party names, with a fixed q̃.
- **Initial prior:** π = softmax(BM25(issue, node text)/T) mixed 0.8/0.2 with a public-API prior (`__all__` or exported, public module).
- **Decode:** forward–backward gives γ_t(v). score(v) = (1/k) Σ_paths max_t γ_t(v), with BM25 as tie-break.
  - Secondary readout: the final-step posterior only.
- **Tuning.** (α, β, τ, ε, ρ, δ, η, T) are tuned on CAL-tune by 200-trial TPE maximising MRR, then frozen.
  - **Every control and baseline gets the same 200-trial budget on its own free parameters.** For example, PPR tunes its restart and T.

### 3.5 Study B (LATENT) objectives, with the critic's fixes

- **Windows.** ≤12k-token windows packed from 2-hop static∪HEAP clusters.
  - Each window is packed in **two random orders**, and results are averaged.
  - Labels are only for co-present pairs.
  - Each function is mean-pooled at layers ≈15/30/45 of 60.
- **Edge probe:** s(u→v) = (P h_u)ᵀ(Q h_v), rank 32, BCE.
  - Positives: traced edges.
  - Negatives: (site-executed ≥5×, candidate never observed) pairs, plus random same-window pairs as a secondary.
- **Controls:**
  - additive probe a·h_u + b·h_v;
  - Hewitt–Liang (DPR-permuted labels);
  - position-only and token-distance-only logistic baselines;
  - window-position residualisation (RT P1 hostile note a);
  - name-mention baseline.
- **Strata (strict hierarchy):** name-mentioned ⊂ static (Pyright∪PyCG∪PoTo) ⊂ HEAP-resolvable ⊂ call-time-only.
  - The headline stratum is *call-time-only, no name mention*.
  - **Pre-gate:** ≥300 positive pairs across ≥3 EVAL repos.
- **Effect head:** linear multilabel BCE on h_v against the same 16 local bits.
- **Training and test split.** Train on CAL-train windows; test on EVAL-repo windows. Layer and L2 are chosen on a CAL hold-out.
- **Quantisation check.** Train on NF4 states and test on dequantised-W4A16 states for 0.2M tokens; report the AUROC drop (CR §0.6).
- **Positive control:** explicit "A calls B" lines must reach AUROC ≥0.95.
- **Negative control:** a random-init model of the same architecture must fail on the call-time stratum.
- **Verbal baseline H3:** ≥2 prompts, scored by log-prob of yes/no, on the same pairs.

### 3.6 LoRA: none in v1

The HYTRACE LoRA is a gated optional rung, and only if all three hold:
- FSDP is measured at ≥400 tok/s;
- C1 passes on EVAL;
- NH spent so far is ≤15.

It is never part of the headline.

---

## 4. Arms, baselines, controls, metrics

### 4.1 Baselines (graph-free and external)

| ID | Baseline | Where it runs |
|---|---|---|
| B-BM25 | Function-level BM25 over source text (identifier subtokens + docstring) | CPU |
| B-CRE | CodeRankEmbed (137M), issue → function | CPU, unique-body cache |
| B-SRS | SweRankEmbed-Small | CPU |
| B-SRL | SweRankEmbed-Large (7B) | GPU, optional (B2) |
| B-NAMES | Imagined names only, fuzzy-matched to nodes, no graph (β = 0, A = identity) | CPU |
| B-PPR | **PPR from π with restart r on G and A, no imagined path. The main one-change control** | CPU |
| B-CTX | "Graph as context": the top-30 PPR neighbourhood serialised as a tree-indented, directed list (G-A14 format); Gemma ranks the candidates | GPU, 150 tasks |
| B-CoSIL | CoSIL with Gemma via vLLM, ≤20 calls per task | GPU, 100-task subset |
| B-GL | GraphLocator, if code is released (A19); otherwise its published numbers are cited only, as not comparable | — |

All ranking arms are reported **alone** and **fused with BM25 by RRF (k = 60)**.

### 4.2 BMM ablations and controls (each changes one thing relative to BMM-full)

| ID | Change | Tests |
|---|---|---|
| X-SHUF | Step order shuffled | Sequence carries information (CR kill) |
| X-FL | Only the first and last steps kept | The middle of the path matters |
| X-RAND | Observations = names and AST effects of a random walk from π on G, length-matched | Gains are not from the decoder's prior alone; must be ≤ B-PPR |
| X-DPR | G rewired with in/out degree and module preserved; A recomputed by f_θ | Real structure, not degree or locality |
| X-NOEFF | β = 0 (names only, with graph) | **Novel piece: does behaviour add?** |
| X-ASTEFF | q from AST rules in place of g_φ | **Does execution calibration add?** (CR §5 item 2) |
| X-BAG | Effects + names matched per node, no transitions (tie-aware ranks) | Behaviour needs structure to resolve ties |
| X-UNIF | Uniform p(c\|s) in place of CHAIN-lite | Calibrated transitions add |
| X-EDGE | CHAIN-lite as per-edge BCE + renormalisation | Site-normalisation (AFLGopher-style) vs per-edge |
| X-STAT | G = static union only, no HEAP | HEAP edges add (infrastructure readout) |
| E-TR | q from pre-existing-test traces where available, g_φ elsewhere | Execution at inference; disclosed as not test-free |
| E-LAT | q from the LATENT effect head | Study B link (paper-only) |
| E-ORA | q from F2P traces + oracle true path names noised 50% | **Positive control P1**: hit@5 ≥0.80 or the decoder is broken |
| T-LAT | A augmented with LATENT edge scores for call-time-only candidates (100-task subset) | Study B link (paper-only) |

The renamed views R1 and R2 are run for B-BM25, B-CRE, B-SRS, B-PPR, B-NAMES, BMM-full, X-NOEFF and X-ASTEFF.

### 4.3 Metrics and statistics

- **Item-level primary metric.** Reciprocal rank of the best-ranked gold function among **all non-test repo functions at the base commit** (no pools, CR §0.4). Also hit@1/5/10. Tie-aware expected rank is used wherever scores tie.
- **Strata reported:** traceback-free (**headline**, n ≥300); named / no-name; effect-covered / not; on-path / off-path; new-code (file level); polymorphic-path / monomorphic-path; post- vs pre-model-cutoff task date.
- **Contrasts.**
  - Paired bootstrap over tasks, 10,000 resamples, 95% CIs.
  - A repo-clustered bootstrap is a sensitivity check.
  - Holm correction over the 3 primary contrasts: C1 = BMM − B-PPR; C2a = BMM − X-NOEFF under R2; C2b = BMM − X-ASTEFF.
  - Everything else is secondary and reported with CIs and no claims.
- **Seeds.** Generator variance uses 3 sampling seeds × k = 5. Every per-item score is averaged over seeds first. Seed-level spread is reported separately.
  - g_φ and f_θ: 3 repo-split seeds.
  - LATENT: 3 repo-split resamples × 3 initialisations.
  - Decoders are deterministic.
- **Pre-registration.** Hyperparameters, strata definitions and this arm table are hashed into `data/` before the first EVAL generation.
- **Power (A16).**
  - For C1, with SD(ΔRR) ≈0.25 and n = 300, the SE is ≈0.014. The MDE at 80% power under Holm (α ≈ 0.017) is ≈0.045.
  - For nested arms (C2), SD ≈0.12, giving an MDE of ≈0.022.
  - Both SDs are measured on CAL-tune before EVAL. If the measured SD exceeds the assumption, the affected claim is downgraded to "estimate only" **before** looking at EVAL.

### 4.4 The measurement layer (publishable whatever C1 does)

- **Step-level imagination accuracy:**
  - exact-name hit rate per step;
  - fraction of steps whose MAP-decoded node lies on the executed true path;
  - position of the first divergence.
- **Break categories at the divergence**, using HEAP edge kinds: decorator/wraps, registry/dispatch, super/MRO, `__getattr__`, generator, re-export.
- **Effect coverage statistic** on traceback-free tasks: issue-stated effect = traced gold-path effect.
- **Over-extension ratio** (G-A4): imagined path length vs true path length.

---

## 5. Falsifiable central claim and predicted effect sizes (all GUESSED)

| Claim | Prediction (EVAL, traceback-free, n ≥300) | Kill / falsifier |
|---|---|---|
| **C1** BMM-full > B-PPR (alone and BM25-fused) | ΔMRR **+0.05**, Δhit@5 +6 pp | Holm-adjusted CI lower bound ≤0 → **BMM dies as a technique**. The paper becomes the §4.4 measurement plus Study B |
| C1′ BMM-full vs external baselines | vs BM25 +0.06 MRR; vs SweRankEmbed-Small +0.02 (uncertain); vs CoSIL ≈ at ≥10× lower GPU cost | Not a kill. Reported honestly, including losses |
| **C2a** Behaviour adds, more so without shared names | BMM − X-NOEFF: +0.02 (original), **+0.04 (R2)** | CI ≤0 under R2 → **the novel piece dies**. BMM reduces to map-matching (closeness 2) |
| **C2b** Execution calibration adds over syntax | BMM − X-ASTEFF: **+0.015** | CI ≤0 → the "execution-labelled" sub-claim dies; AST effects are kept |
| C3 Order and structure carry it | X-SHUF −0.02; X-DPR −0.04; X-RAND ≤ B-PPR; X-BAG < BMM by ≥0.03 | SHUF ≈ BMM **and** DPR ≈ BMM → "bag of names and effects", not decoding |
| C4 Renaming robustness | Under R2, BM25 loses ≥40% of its MRR; BMM loses ≤ half of BM25's relative drop | BMM's relative drop ≥ BM25's |
| M1 Imagination quality | Exact-name step hit 25–40%; ≥50% of paths have ≥half their steps wrong, yet C1 holds | — (descriptive) |
| M2 Effect coverage | 25–40% of traceback-free tasks | **CPU gate:** <20% → drop C2 before GPU spend (§7) |
| B1 (Study B) | Call-time-only, no-name AUROC 0.62–0.75 | Pass bar: ≥0.70 **and** ≥ position-only + 0.08 **and** ≥ additive + 0.03, with CIs over 3 resamples. Failing it is the "latent graph is static-lexical" finding (Anand et al. prior favours it) |
| B2 (link) | Imagined-step divergences concentrate on edge strata where probe ≈ position baseline: logistic regression of step-error on stratum + probe score shows a negative probe-score coefficient | Coefficient CI includes 0 |

**Positive controls.**
- P1 (E-ORA): hit@5 ≥0.80.
- P2: oracle effects with parsed issue effects, effect-bag MRR ≥0.4 on effect-covered tasks.
- P3: LATENT text edges ≥0.95.
- P4: random-init model at chance.

If P1 fails, no C-claim is read.

---

## 6. GPU budget (NH)

Throughput assumptions:
- vLLM: start 8 min (VERIFIED); prefill ≈2k tok/s with prefix caching (GUESSED); batched decode 400–800 tok/s aggregate (GUESSED; single-stream 34 tok/s VERIFIED).
- HF forward: layer-split 250–350 tok/s (INFERRED), or ≈1.2k tok/s with per-GPU NF4 replicas if P0 shows it.
- NF4 load: 12 min per process (VERIFIED).

| Line | Content | Opt | Central | Pess |
|---|---|---|---|---|
| P0 | Forward-only throughput probe (layer-split vs 4 NF4 replicas) | 0.5 | 0.5 | 0.5 |
| G1 | Main generation: (300 EVAL + 118 DEV + 100 CAL-tune) × ~7k prefill = 3.6M; decode 518 × 5 × 3 × 250 ≈ 1.9M | 1.0 | 1.6 | 2.5 |
| G2 | R1 + R2 renamed generations (300 tasks, 1 seed, k = 5) | 0.6 | 1.0 | 1.5 |
| G3 | Issue → effect-query parse for the bag/P2 arms | 0.1 | 0.2 | 0.3 |
| G4 | Second generator model, 300 tasks (generality) | 0.5 | 0.8 | 1.2 |
| B1 | CoSIL, 100 tasks | 1.0 | 1.8 | 2.5 |
| B2 | SweRankEmbed-Large over unique EVAL function bodies (~25M tok) | 0.4 | 0.8 | 1.5 |
| B3 | Graph-as-context arm, 150 tasks × ~10k | 0.3 | 0.5 | 0.8 |
| L1 | LATENT forward: kill 1.2M + full 1.8M + R2 0.5M + positive control 0.2M (includes the 100-task T-LAT neighbourhoods) = 3.7M, + loads | 1.3 | 2.5 | 4.0 |
| L2 | NF4 → W4A16 transfer check, 0.2M bf16 layer-split | 0.3 | 0.3 | 0.4 |
| L3 | Verbal baseline H3, 2 prompts, logprob | 0.5 | 0.7 | 0.9 |
| L4 | LATENT replication on a second model | 0.3 | 0.6 | 1.0 |
| J1 | JS generality, ~100 tasks (generation only; graph and traces on CPU) | 0.4 | 0.6 | 1.0 |
| R | Reserve for one failed or rerun session | 1.0 | 1.5 | 2.5 |
| | **Total** | **8.2** | **13.4** | **20.6** |
| (opt.) | HYTRACE LoRA, 3 seeds × ~1.8M tok (only under the §3.6 gates) | +6 | +6.5 | +7 → exceeds cap; forbidden in this case |

### 6.1 Descoping ladder

Apply in order when projected spend exceeds 25 NH, or when actual spend after G1 exceeds 1.5× central:
1. LoRA never starts.
2. Drop J1 (JS). Generality rests on G4 (second model) and L4.
3. Drop B2 (SweRankEmbed-Large). Keep Small and CodeRankEmbed on CPU.
4. Cut B1 CoSIL to 50 tasks, or cite it only.
5. Drop L4 and the LATENT R2 windows.
6. Drop B3, and say "decoder vs context" is untested.
7. G2: keep R2 on all 300 tasks; R1 on 150.
8. G1: 2 sampling seeds.

**Floor paper** (P0 + G1 + R2 + L1-kill + L2): ≈4.5–7 NH. That still tests C1, C2a, C2b and B1.

---

## 7. Execution order and gates

1. **Week 1, CPU only (free).**
   - Environments, validation and traces for EVAL candidates and CAL repos.
   - HEAP and the static union. f_θ and g_φ trained.
   - Strata counted.
   - **Gates:**
     - ≥300 traceback-free EVAL tasks (A5), else extend candidates from SWE-rebench;
     - effect coverage ≥20% on CAL-tune and EVAL issues, using a cheap regex/keyword parser as a proxy (A7), else drop C2 and re-plan;
     - Study B pre-gate of ≥300 call-time-only, no-name pairs (A18), else Study B reports only static vs HEAP strata;
     - g_φ macro-AUROC over AST rules ≥ +0.03 on CAL hold-out (A9), else C2b is pre-declared dead.
2. **P0** (0.5 NH).
3. **G1 on CAL-tune only** (≈0.3 NH of G1). Positive control P1 must pass. Tune hyperparameters, measure the paired-Δ SDs (A16), freeze, hash.
4. **G1 on EVAL + DEV, then G2.** All decoding arms run on CPU.
5. **L1 kill (1.2M tokens) → L2 → the rest of Study B**, only if the kill passes its pre-gate. Otherwise stop at the kill and report it.
6. B1, B2, B3, G4, J1 in ladder order as budget allows.

---

## 8. Generality plan

| Axis | How | Prediction |
|---|---|---|
| **Second model** (required, G4) | Swap only the generator: Qwen3-32B or Qwen3-Coder-30B-A3B (W4A16) from Kaggle Models. The decoder, graph, effects and tuning stay frozen | C1 holds in sign. The gain over B-NAMES is *larger* for the generator with the lower exact-name hit rate ("structure corrects names") |
| Model scale (cheap) | Gemma-4-E4B as generator on 150 tasks (≈0.2 NH, inside the R reserve) | The decoder's absolute gain grows as the generator gets weaker |
| **Second language** (stretch, J1) | JS: static + approximate interpretation from Jelly (the HEAP analogue already exists there), a Jelly dynamic call graph or a `--require` hook tracer, and the effect alphabet mapped (throws, returns undefined/null, mutates arg, console.warn, I/O). g_φ is retrained on JS CAL repos | Same direction for C1/C2a; smaller C2b |
| Beyond issues | The decoder applies to any "imagined trace → graph" problem: crash-free bug reports, log-less incident reports, and grounding a frontier model's plan. Stated as scope, not tested | — |

---

## 9. Traps from earlier rounds and how v1 avoids them

| Trap (source) | v1 rule |
|---|---|
| Gameable set rewards (K-30/31) | No sets. Full-repo ranks only |
| Items solvable without the graph (K-04/12) | B-BM25, B-NAMES, X-BAG, and the traceback-free and no-name strata |
| Multi-change controls (K-21) | Every X-arm changes one thing. DPR preserves in/out degree **and** module. B-PPR shares π, G, A and the tuning budget |
| Pools containing gold by construction (CR §0.4) | Full-repo ranks only |
| Underpowered 3 pp kills (CR §0.3) | n ≥300, CI-based kills, SDs measured on CAL-tune first |
| Under-counted budgets (L5 ×2) | Tokens, starts, loads and a reserve all counted. No training line |
| "Method" that is SFT on items | No LLM weights change in v1. The trained objects are CPU predictors over execution facts from other repos |
| Leakage from F2P tests or test names (CR §3 item 6, §5 item 1) | F2P traces only in oracle and scoring arms. Generator inputs exclude tests and the test patch |
| Unproven mechanism prior art (RT cross-cutting 1) | HEAP and CHAIN are infrastructure, cited at 2–2.5 |

---

## 10. Assumptions for the breakers (A1–A20)

| # | Assumption | How it fails / how it is checked |
|---|---|---|
| A1 | 25 NH = Kaggle 4×L4 session-hours. CPU kernels are free and ≥5 run concurrently | Quota policy change; fewer concurrent kernels → week 1 takes longer, but NH is unchanged |
| A2 | vLLM on Gemma-4-31B W4A16 TP4: prefill ≥2k tok/s with prefix caching; batched decode ≥400 tok/s | Unmeasured. If it is 2× slower, G1+G2 move to the pessimistic column |
| A3 | Forward-only HF throughput is 250–350 tok/s layer-split, possibly ~1.2k tok/s with per-GPU NF4 replicas | P0 measures it. At <200 tok/s, L1 is cut to the kill only |
| A4 | ≥50% of candidate tasks reproduce fail→pass on Kaggle CPU with uv-installed Pythons and no Docker; ≈10 CPU-min per task after environment reuse | Old django/sympy versions with C-extension deps may fail en masse. Mitigation: shift weight to SWE-rebench |
| A5 | ≥300 traceback-free EVAL tasks survive from ≈1,200 candidates | Week-1 count |
| A6 | The gold function is on the executed F2P path in ≥60% of traceback-free tasks | If on-path <40%, C1's ceiling is low. Reported as a stratum, not a kill |
| A7 | ≥25% of traceback-free issues state an effect in the 16-bit alphabet that matches a traced gold-path effect | Week-1 gate at 20%. RT warned the stratum may be small |
| A8 | Imagined paths have a decodable entry in most tasks, and ≥30% contain ≥1 middle step that maps onto the true path | Measured in §4.4. If it fails, C3 dies (X-FL ≈ BMM) |
| A9 | g_φ transfers across repos and beats AST rules by ≥0.03 macro-AUROC on held-out repos | CAL hold-out gate. If it fails, C2b is dead before GPU spend |
| A10 | "Site executed ≥5× without observing c" marks true non-edges often enough for CHAIN-lite targets | Coverage holes bias f_θ toward tested callees. X-UNIF shows whether it matters |
| A11 | HEAP imports ≥80% of modules per EVAL repo with a stub importer, side effects contained (network off, isolated HOME/tmp) | Import-failure rate reported per repo. X-STAT bounds the dependence |
| A12 | Pyright CLI, PyCG and PoTo finish on EVAL snapshots within CPU limits, and CodeRankEmbed / SweRankEmbed-Small embed unique bodies on CPU in hours | PyCG times out on some projects (G-A1); that repo then uses Pyright ∪ PoTo, reported |
| A13 | Imagined steps are roughly caller→callee ordered, so a first-order HMM fits | X-SHUF tests it. A reversed-order variant is logged |
| A14 | Hyperparameters tuned on 100 CAL-tune tasks transfer to EVAL repos | DEV is the untuned readout. Large CAL→EVAL drops are reported |
| A15 | Gemma's memorisation of SWE-bench Verified patches does not dominate | Post-cutoff SWE-rebench stratum and R1. If BMM's gain exists only pre-cutoff, report contamination |
| A16 | SD of paired ΔRR is ≈0.25 (C1) and ≈0.12 (nested C2 arms) | Measured on CAL-tune before EVAL. If larger, claims are downgraded before unblinding |
| A17 | NF4 hidden states transfer to W4A16 served weights (AUROC drop ≤0.03) | L2 check. If it fails, Study B is reported as "about NF4 Gemma" |
| A18 | The call-time-only, no-name stratum has ≥300 positive pairs over ≥3 EVAL repos | Week-1 count (CR pre-gate) |
| A19 | CoSIL runs with Gemma via vLLM at ≤20 calls per task. GraphLocator code is not released | If CoSIL needs a frontier model's tool-calling, the baseline is cited only |
| A20 | No closeness-3 prior exists for *behavioural emissions in imagined-path decoding*. The log-to-path family (SherLog ASPLOS'10, 2506.19045, COCA 2503.23051) stays at ≤2 because its observations are real logs, not prose-imagined effects | **Not searched by RT or CR.** Breakers should search "log-based execution path reconstruction", "symptom-guided path inference", "behavioural bug report localisation HMM" and "effect-based fault localisation" |

---

## 11. What I changed relative to the Phase-3 inputs

- **Merged.** HYTRACE + EFFECT into one decoder with behavioural emissions. The novel piece is the emission, not the HMM.
- **Demoted.**
  - HEAP → infrastructure, with X-STAT as its one readout.
  - CHAIN → transitions, with X-UNIF and X-EDGE.
  - LATENT → Study B, plugged in as paper-only E-LAT and T-LAT arms.
- **Killed.** DECONV is not used; v1 does not touch the supplied embeddings.
- **Added beyond both reviews:**
  - X-BAG, the argument that effects need structure, as a test;
  - B-CTX ("graph as context"), the field's default use of graphs, to test the critic's thesis directly;
  - name-free AST-effect features with renamed views rather than renamed executions;
  - a CAL-tune split, so no hyperparameter touches EVAL or DEV;
  - SD measurement before unblinding.
- **Not done:** any new prior-art search. A20 lists what the breakers must search.

No file other than this one was modified.
