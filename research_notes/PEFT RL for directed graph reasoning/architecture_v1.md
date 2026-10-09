# Architecture v1: GRAFT, Graph-compiled, Rewire-controlled, Atomised-readout Fine-Tuning (8 Oct 2026)

**Role:** architect. This is one design built from the parts that survived Phase 3. It is meant to be attacked. §11 lists the assumptions it rests on, numbered so breakers can target them.

**Inputs read in full:**
- `inventor_phase1_2.md`;
- `phase3_critic.md`;
- `phase3_redteam.md`;
- `docs/training_feasibility_4xl4.md`;
- the header and harness facts in `docs/HQ.md`.

**Labels** (same as the inventor's file):
- **VERIFIED**: read in the primary source or computed from repo files by an earlier pass. Inherited labels are kept as they were.
- **SNIPPET**: title or abstract only.
- **INFERRED**: my reasoning, or a citation from memory.
- **GUESSED**: a number used only for sizing. A gate measures it before it is relied on.

**Tags used throughout:**
- **[KNOWN]**: the piece is prior art and is cited.
- **[NEW]**: no prior was found by the Phase-3 red team (absence of evidence only).
- **[NARROWED]**: a known mechanism; only the code- or graph-specific use is claimed.

---

## 0. One-paragraph summary

GRAFT trains a LoRA adapter on a quantised model so that it reasons along the edges of large directed program graphs. It does this **without training on any real repository**.

1. A compiler turns a typed directed graph into an executable Python repository with tests. Values flow along edges. Some edges are "dark" (dispatch, decorator, registry, callback), meaning they are not lexically visible.
2. Bugs are injected. These include a new **degree-preserving call-target swap** (DPCS) operator, which creates two-site wiring bugs that are jointly necessary to fix. Body-level mutations are also injected.
3. The model is trained in two stages, with the test-execution outcome as the reward:
   - an SFT warm start on its own verified successes (rejection sampling);
   - an offline, clipped, group-advantage stage.
   In both stages an auxiliary per-element proper-scoring term acts on a canonical set readout ("which of these functions are called / must be edited, with what probability").
4. **The experimental variable is topology.** Three training arms are identical except for the graph fed to the compiler:
   - (b) topology copied from real traced repositories;
   - (c) a DAG-preserving, depth-matched, degree-preserving rewire of those graphs;
   - (d) depth 1.
5. Transfer is measured densely on **≥300 held-out real-repository call-set items**. Each item is shown in three contexts: real, graph-free, and rewired. This gives an accuracy effect and also a *structure-use* effect, i.e. how much the model relies on the edges it is shown.
6. One development-set resolve run serves as a non-inferiority guard.

The pipeline is the same code offline (Kaggle) and online (GRPO). Only the outer loop changes.

## 1. The central claim (falsifiable) and what is new

**Central claim C.** RL-style LoRA training on content-free, topology-controlled compiled repositories improves a quantised model's per-element call-set prediction on held-out real repositories, and raises its *reliance on displayed edges*. Three sub-claims follow:
- The gain comes from **multi-hop traversal demand**: arm (b) beats arm (d).
- It does **not** come from the fidelity of the real topology beyond its degree and depth profile: (b) ≈ (c) within a pre-registered margin.
- It is achieved without harming end-to-end resolve.

| ID | Prediction (pre-registered) | Predicted value (GUESSED) | Test |
|---|---|---|---|
| H1 transfer | AP(b) − AP(base) on held-out real call-set items, real context | **+0.07** (base A-format AP ≈0.55 at a 33% base rate) | Paired, cluster bootstrap over repos. Pass if the 95% CI lower bound > +0.02 |
| H2 depth is the active ingredient | AP(b) − AP(d) | **+0.05** | Pass if the CI lower bound > 0 |
| H3 fidelity is irrelevant | \|AP(b) − AP(c)\| | **≈0.01** | TOST, margin 0.03, 90% CI inside ±0.03 |
| H4 structure use | Δ_struct(b) − Δ_struct(base), where Δ_struct = AP(real ctx) − AP(RW ctx) | **+0.04** (base Δ_struct ≈0.02); Δ_struct(d) − Δ_struct(base) < 0.015 | Paired bootstrap. Pass if the CI lower bound > 0 for (b) |
| H5 no regression (guard) | Dev resolve of the best arm vs the stored r3_v5 base, same harness | Δ ≥ −2 pp; P2P breakage not higher | TOST non-inferiority, margin 6 pp (the MDE at n=129 is 7–9 pp, critic X3) |
| H6 commit completeness (secondary, descriptive) | Two-site wiring bugs, held-out synthetic topology family: fraction of tasks where both sites are listed EDIT at p≥0.5 | base ≈0.25 → (b) ≥0.50 | Paired McNemar, n=100. Dev ≥2-gold slice reported only descriptively (n=65, MDE ≥12 pp) |

Multiplicity: H1–H4 form one Holm family. H5 and H6 are outside it.

**What each outcome means:**

| Outcome | Interpretation |
|---|---|
| H1 fails (CI includes +0.02) | The transfer premise is dead: content dominates. This matches the prior from SWE-Playground: synthetic projects match SWE-smith at 7B but trail by ≈9 pts at 32B (2512.12216, VERIFIED by the red team). Report how much content matters. |
| H3 fails with (b) > (c) | Real call-graph topology carries transferable structure beyond its degree and depth sequence. That is a graph-ML finding in itself. |
| H4 fails while H1 passes | The model got better at the items without leaning on the edges it was shown. This is "uses structure for selection only", the CNY reading (inventor §0, anomaly 4). |

**The novel piece.** It is the *combination* below. No single component is claimed as a mechanism.

1. **[NEW] Topology-as-treatment compiler.** A typed directed graph, including dark edge types, is compiled into an executable repository with tests. Everything except topology is held fixed across arms (identifier vocabulary, body templates, bug operators, token and success budgets). The red team found no compiled-topology repositories used for agent RL (SWE-Playground 1.5 and FrogNano 1 are the closest).
2. **[NEW] Degree-preserving call-target swap (DPCS) as a bug and data operator.** A double swap (a→b, c→d) ⇒ (a→d, c→b) between type-compatible callees produces **two-site, jointly necessary, purely topological** bugs. Node bodies are unchanged and every in- and out-degree is preserved. No call-target swap operator was found in SWE-smith (red team #18). The closest neighbours are Software World Models (2610.04940, consequence prediction) and the predictive-mutation-testing (PMT) lineage. The *same* operator, with acyclicity and depth constraints, generates control arm (c).
3. **[NEW measurement] Rewired-context structure-use readout on real code.** Displayed function bodies are shown with their call sites permuted by a degree-preserving rewire of the repository's static call graph, *including the distractors* (critic fix 1 for CASE K3). The labels stay true. Δ_struct then measures reliance on shown edges. The closest neighbours are CNY's rewire control (2608.29588) and edge-shuffle probes in text-attributed-graph (TAG) work (2511.16767, SNIPPET).

**Known pieces, cited:**
- Rejection-sampling fine-tuning: STaR/RFT (INFERRED).
- Clipped group-advantage surrogate: PPO/GRPO, with Dr.GRPO-style mean-centring (2503.20783, INFERRED).
- Per-element proper scoring: [NARROWED]. The mechanism prior is Ma & Chochlakis 2505.17510, which shows list emission suppresses labels. The format prior is CoverageEval 2307.13383. The per-entity reward prior is SPADER 2606.00593.
- Contrastive self-teacher instrument: [NARROWED]. RLCSD 2606.11709, CROP 2608.13387, CREDIT 2605.11613, with the Salesforce three-checks K0 from 2608.09263.
- Degree-preserving null model: Maslov–Sneppen double-edge swap (INFERRED).

---

## 2. Component diagram

```
                ┌──────────────────────── CPU (Kaggle CPU kernels, local) ───────────────────────────┐
                │                                                                                     │
 Topology       │  C1 Tracer                C2 Topology transformer           C3 Graph→Repo compiler   │
 source repos ──┼─▶ (setprofile+threading,  ─▶ (b) copy  (SCC-condense→DAG)  ─▶ typed nodes, value     │
 S_topo (6,     │   code-object identity,     (c) DPCS rewire: DAG-preserving,   flow along edges, dark │
 non-dev,       │   wrapper collapse)            depth-matched, type-preserving  edge types, vocab-only │
 non-eval)      │   G_exec, edge types        (d) depth-1 forest, same |V|       identifiers, tests,    │
                │                                                              │   unit "contracts"     │
                │                                                              ▼                        │
                │                         C4 Bug injector: body mutations (SWE-smith ops) + DPCS wiring │
                │                            swaps + 2-node contract bugs; C5 necessity check (every   │
                │                            strict subset of the fix still fails ≥1 F2P)               │
                │                                                              │                        │
 Held-out eval  │  C9a Item builder (non-dev, non-S_topo repos S_eval ≥8): trace → call-set items;      │
 repos S_eval ──┼─▶ contexts {real, GF, RW}; formats {A, L, Lmax, L-EOS-suppressed}                    │
                └───────────────────────────────────────────────┬────────────────────────────┬─────────┘
                                                                │ tasks + items              │ eval items
                ┌──────────────────────── GPU (4×L4) ───────────▼────────────────────────────▼─────────┐
                │  C6 Generator: vLLM W4A16 TP=4, multi-LoRA, temperature sampling (no top-k/min-p),  │
                │     K=4 per task; deployed tools only; log behaviour log-probs                       │
                │        │ trajectories            ▲ LoRA_k                                             │
                │        ▼                         │                                                    │
                │  C7 Verifier/Reward (CPU sandbox): F2P fraction, P2P guard, per-element set score   │
                │        │ (R, per-token Ã)                                                              │
                │        ▼                                                                               │
                │  C8 Trainer: FSDP-QLoRA on exact served grid (HQQ g32); Stage A RFT → Stage B clipped │
                │     offline surrogate; export PEFT (bias=none, no modules_to_save)                    │
                │        │                                                                               │
                │  C9b Item evaluator (prefill-heavy) ── C10 Edge-use instrument (6-view prefill, K0)  │
                │  C11 Dev guard (129 tasks, best arm only)                                             │
                └───────────────────────────────────────────────────────────────────────────────────────┘
```

**Controls built into the architecture** (graph-free = GF, rewire = RW):

| Where | GF control | RW control |
|---|---|---|
| Training data | Arm (d): depth-1, no traversal needed. Same |V|, vocabulary, templates and operators | Arm (c): DAG-preserving, depth-matched, degree-preserving rewire of (b)'s graphs |
| Gate K2 (zero-shot) | The issue names the bug node(s) | — |
| Item evaluation | Names and signatures only; no bodies, so no call sites | Bodies with call sites permuted by a degree-preserving rewire of the static call graph over pool ∪ distractors. Labels unchanged |
| Instrument | Node-list teacher δ^∅ | ρ1, ρ2, a plausible-ρ, and a dose-response (0.25 / 0.5 / 1.0) |

---

## 3. Components in detail

### C1. Tracer (CPU) [KNOWN; critic edge cases folded in]

- Uses `sys.setprofile` together with `threading.setprofile`, because the per-thread miss on httpx/anyio was flagged by the critic.
- Function identity is keyed on the **code object**, not the name. functools wrappers are collapsed onto `__wrapped__`.
- Calls into C code are invisible. The resulting under-labelling is reported per repo.
- Each edge gets a type from how the call site resolves:
  - **lexical**: the callee name followed by `(` appears in the caller's body;
  - **dispatch**: resolved through the MRO;
  - **decorator**;
  - **registry / dict**;
  - **callback**: passed as an argument.
- Output per test: `E(t)` (executed functions), `G_exec(t)`, call-site file:line, and test-IDF.

**Repository partition (fixes the contamination of critic X5 and the topology-copy leak):**

| Set | Contents | Used for |
|---|---|---|
| Dev | fastapi, rich, requests, httpx | **Excluded** from S_topo and S_eval. Evaluation only, via the C11 guard |
| `S_topo` | 6 permissive, non-dev repos with working pytest (SWE-smith / SWE-rebench pool) | Their `G_exec` topologies are copied for arm (b). Their identifiers feed the vocabulary pool. Their code is never shown |
| `S_eval` | ≥8 other permissive, non-dev repos | The held-out item set. **Disjoint from S_topo** |

### C2. Topology transformer [(b) KNOWN copy; (c) NEW constraint set; (d) control]

The input is the union of `G_exec` over a source repo's test suite. Steps:
1. Restrict to an induced connected subgraph of 200–1,500 function nodes, grown by BFS from the test roots.
2. Condense strongly connected components into single nodes, so the graph is a DAG.
3. Report the fraction of edges lost (expected <3%, GUESSED). Recursion is not modelled. This is a disclosed transfer gap (A16).

The three arms:

| Arm | Construction |
|---|---|
| **(b) real-copied** | The condensed DAG with its edge types. |
| **(c) rewired** | Directed double-edge swaps (a→b, c→d) ⇒ (a→d, c→b), only between edges of the **same edge type** (so per-type in- and out-degree is preserved). A swap is rejected if it would create a self-loop, a multi-edge, or a cycle (reject a→d if d reaches a; incremental reachability check). 10·|E| attempts. The final graph is accepted only if its longest path is within ±1 of the original **and** a KS test on the per-node depth histogram (depth from the test roots) gives p > 0.2. Otherwise retry, up to 20 seeds. If no seed passes, the source graph is dropped from **all** arms, so the arms stay paired. |
| **(d) depth-1** | Same node count. Every non-test node is called directly from a test root or from at most one leaf helper. Same type mix. It cannot be degree-matched, and is disclosed as the "no traversal needed" control. |

What the arms preserve:
- (b) vs (c) differ only in *which* node connects to which. Degree sequence (per type), depth profile, node bodies, names and bugs are all matched.
- (b) vs (d) differ in traversal depth, and necessarily also in degree.

### C3. Graph → repository compiler [NEW, the core]

**Nodes.**
- Each node gets a value type from {int, str, list[int], dict[str,int]} and a body template drawn from a library of about 40 deterministic templates (arithmetic, string transforms, folds, lookups).
- The template consumes the node's inputs and its children's return values.
- Identifiers are sampled from an S_topo identifier pool. They have no semantic link to the template. This is content-free by design, and the disclosed transfer risk is A5/A7.

**Dark edge types compile to non-lexical calls:**

| Edge type | Compiled form |
|---|---|
| dispatch | A base-class method overridden in a subclass and called through `self.m()` |
| decorator | A wrapper calls the wrapped function |
| registry | `REGISTRY[key](x)` with a string key |
| callback | The function is passed as an argument |

`grep "name("` does not find these callers. The dark-edge share is set to match S_topo, measured by C1.

**Layout.** 20–120 modules. The file and module structure is copied from the source repo (b), or the same partition is reused for (c) and (d).

**Tests:**
- **Root tests**: pytest system tests at the test roots.
- **Unit contracts**: unit tests for a random 40% of internal nodes. Expected values are computed by running the *correct* program.
- Visible to the agent: the unit-contract file, minus held-back tests.
- Hidden F2P: root tests plus held-back unit tests.
- P2P: all the rest.

These contracts are what make the intended wiring *inferable*. The agent can run them and bisect along the call chain, which is a real backward-traversal procedure.

**Issue text.** Generated from the failing root test's assertion (expected vs actual). It names the **entry function only**, which matches the dev reality that only 21.2% of tasks name a gold node (HQ A2, VERIFIED). The GF variant (gate K2 only) names the bug node(s).

**Harness.** Exactly the deployed tool set: `run_command`, `read_file`, `edit_file`, `write_file`, `submit_patch`. Optional graph tools are backed by a deliberately degraded H (undirected, merged labels), mirroring `G_prov`. The deployed prompts and caps are used unchanged (read_file 150 lines, command output 5k, ADK compaction at about 14k).

### C4. Bug injector [body operators KNOWN (SWE-smith); DPCS NEW]

Each task gets 1–3 bug nodes, mixed 50 / 30 / 20 across three operators:

| Operator | Share | What it does |
|---|---|---|
| **O1 body mutation** | 50% | Off-by-one, operator swap, constant change in a node body (SWE-smith style) |
| **O2 DPCS wiring swap** | 30% | Pick two call sites a→b and c→d whose callees b and d share a value type and arity. Swap them. Bodies are unchanged and all degrees are preserved. Type-compatibility *by construction* removes the TypeError giveaway the critic flagged for ERC. Fixing it needs **both** sites, so it is a two-element commit set by construction (it targets the F1 dead zone: 72% `OTHER_LOCATIONS: none` on ≥3-gold tasks, VERIFIED). Swaps that create recursion or reach no root test are rejected. In arm (c), swaps are drawn on the rewired graph. |
| **O3 contract change** | 20% | Two nodes must agree on a changed contract. |

The bug-depth distribution (hops from test root to the deepest bug node) is sampled identically for (b) and (c). For (d) it is 1 by definition.

### C5. Verifier (CPU) [KNOWN]

Each task must satisfy all of the following:
- (i) the gold program passes all tests;
- (ii) the bugged program fails ≥1 F2P;
- (iii) **necessity**: every strict subset of the gold fix still fails ≥1 F2P (IFG-style, critic GCSR edge case);
- (iv) determinism: three repeated runs agree;
- (v) the full suite runs in under 60 s.

Tasks failing any check are dropped and counted.

### C6. Generator [KNOWN]

- vLLM wheelhouse 0.19.1 on W4A16, TP=4, with multi-LoRA (Phase-0 serving GO, VERIFIED HQ Oct 7).
- Plain temperature 1.0. No top-k or min-p, which keeps importance ratios valid.
- K=4 samples per task.
- Turn cap 40. Trajectory token cap 12k, so every training sequence is ≤16k.
- 8–16 parallel streams.
- Behaviour log-probs are logged but **not trusted** for ratios (see C8).

### C7. Reward [execution-verified KNOWN; per-element set term NARROWED]

**Set readout in the agent (prompt-only change, allowed).** The diagnosis turn's `OTHER_LOCATIONS` field is replaced by a `CANDIDATES:` block. It has one line per function the agent has examined:

```
<qualname>: EDIT|SKIP p=0.05|0.2|0.5|0.8|0.95
```

- Verbalised probabilities are robust to W4A16 (critic CASE, quantisation row).
- A `NEW:` slot covers fixes that add new code.

**Trajectory reward:**

```
R(τ) = f_F2P(τ) · 1[P2P intact] − 0.5 · 1[P2P broken] + λ_set · S̄(τ) − 0.1 · 1[turn cap hit]
```

- `f_F2P` is the fraction of hidden F2P tests passing after `submit_patch`.
- The per-element log score is `S_v = y_v log p_v + (1−y_v) log(1−p_v)`, divided by |log 0.05|, which puts it in [−1, 0].
- `S̄(τ)` is the mean of `S_v` over the set (listed ∪ gold). **An omitted gold node is scored at p=0.05.** This is how under-enumeration gets penalised even though it emits no tokens.
- λ_set = 0.2.
- Gold comes from the compiler, so it is exact. Unlisted, non-gold nodes are not scored; this is the tri-state rule (idea 12).

**Group advantage.** Mean-centred only, with no std division (Dr.GRPO-style, which avoids a difficulty bias):

`Â_τ = R(τ) − mean_K R`

**Token advantage:**
- `Ã_t = Â_τ` on every assistant token.
- On the tokens of the CANDIDATES line for element v: `Ã_t = Â_τ + η·(S_v − S̄_v^grp)`, where `S̄_v^grp` is the mean over the group's samples that listed v, and η = 0.5.
- Environment and observation tokens are masked.

**Auxiliary call-set items** (≤30% of training tokens per arm). Each item is built from that arm's *own* compiled repos:
- a test plus a canonical pool, which is the trace positives (stratified by depth) plus twice as many distractors matched on file and degree;
- one line per candidate, `CALLED|NOT p=…`, with labels from the trace (exact).

The items are trained in two ways. First, **element-wise best-of-K recombination**. For each element, keep the sampled line with the best S_v and assemble one target. This is named honestly as *SFT on labels projected onto the model's support* (critic CASE point 4). Second, the per-element advantage term in Stage B. The items use the same format as the evaluation items. This creates a format-familiarity confound, which arm (d) absorbs: it gets the same item type and share, only shallow.

### C8. Trainer and objectives

**LoRA configuration:**

| Setting | Value |
|---|---|
| Rank / α / dropout | r = 16, α = 32, dropout 0.05 (r = 32 is the allowed alternative if Stage A underfits; one change only) |
| Targets | All 410 linear modules (q, k, v, o, gate, up, down; global layers have no v_proj) |
| Excluded | No DoRA. `bias="none"`. No `modules_to_save`. No embedding or lm_head targets |
| Frozen base | The W4A16 checkpoint dequantised to **HQQ 4-bit, group 32**, the exact served grid. NF4 only if the probe shows ≥97% next-token agreement with vLLM W4A16 |
| Parallelism / memory | FSDP FULL_SHARD over 4×L4. SDPA with the math backend disabled (head_dim 512 on the global layers). Gradient checkpointing |
| Loss computation | Chunked CE with the 30.0 final-logit softcap |
| Sequence length | Max 16k |
| Batch / schedule | Effective batch 16 sequences. Cosine schedule, 10 warm-up steps |

**Stage A: SFT warm start = rejection-sampling fine-tuning (RFT)** [KNOWN].

`D_A` combines:
- round-1 trajectories with `f_F2P = 1`, P2P intact, and an all-gold-listed CANDIDATES block, at most 2 per task;
- the recombined item targets.

Loss: `L_A = −(1/|T|) Σ_{t ∈ assistant} log π_θ(y_t | s_t)`. lr 1e-4, 1 epoch.

**Stage B: offline RL-style improvement** [KNOWN form; this is the primary objective]. Data are round-2 rollouts generated by π_A (LoRA_A served), K=4 per task, plus K=4 item samples.

```
L_B = −(1/|T|) Σ_t [ min(r_t Ã_t, clip(r_t, 1−ε, 1+ε) Ã_t) − β_KL · k3(π_θ, π_A)_t ]
r_t = π_θ(y_t|s_t) / π_A^train(y_t|s_t)
```

- ε = 0.2, β_KL = 0.02, lr 2e-5, 2 epochs.
- `π_A^train` is **recomputed in the trainer** (one no-grad forward with LoRA_A) rather than taken from vLLM. That removes the vLLM-W4A16 vs HQQ-kernel mismatch from the ratio (critic SAST quantisation row). r = 1 exactly at the start of epoch 1.
- The vLLM log-probs are kept only to *report* the mismatch: the distribution of log(π_vLLM / π_train) at the zero adapter must be within ±0.05 nats/token median (A12).
- Groups with zero outcome variance still carry the η-term. The fraction of informative groups is reported.

**Pre-registered fallback, not run unless triggered: DPO on verified pairs.**
- Trigger: on the held-out *synthetic* split, Stage B lowers item log score, or more than 20% of tokens hit the clip.
- Data: every group with ≥1 success and ≥1 failure gives a pair (τ⁺, τ⁻).
- Loss: `L_DPO = −log σ(β[(log π_θ(τ⁺) − log π_A(τ⁺)) − (log π_θ(τ⁻) − log π_A(τ⁻))])`, with β = 0.1, log-probs summed over assistant tokens only, and length-normalised.
- Reference: π_A.

**Online-GRPO drop-in, where available (other hardware).** Nothing above changes except the outer loop:
- On-policy K=8.
- `r_t` against the rollout policy (π_old), with the same R, Ã_t, ε and β_KL (against the Stage-A reference).
- C3/C4 become a task server. Each depth bucket is sampled where the running pass rate is in [0.2, 0.8] (an adaptive curriculum).
- Stage A is kept, because SOG-R1 shows SFT+RL 88.7 > SFT-only 86.1 > RL-only 63.0 (CWQ, VERIFIED inventor §0).
- The arms (b), (c) and (d) become three task-server configurations.

**Equalisation across arms (critic GCSR fix 2).** Per stage, every arm trains on:
- the same number of **distinct successful trajectories**: the minimum across arms, with a floor of 15. If an arm falls below the floor, recalibrate that arm's repo *size*, never its depth, before spending more GPU;
- the same number of failure trajectories, in Stage B;
- the same item count;
- the same total tokens, within ±5%, achieved by subsampling.

Successes are subsampled down for the easier arms, especially (d). Each arm gets the same number of tasks and rollouts.

**Curriculum.** For arms (b) and (c): round 1 uses bug depth ∈ {2, 3}; round 2 uses {3, 4, ≥5}. Arm (d) uses depth 1 throughout, with the same size schedule.

### C9. Dense item-level evaluation on held-out real repos (primary readout)

**Items (C9a, CPU).**
- ≥300 call-set items from S_eval: at least 8 repos, at most 60 items per repo.
- One item = test source plus a pool of candidates.
- Positives are E(t), stratified by shortest-path depth {1, 2, ≥3}, up to 14 per item. The sampler is pre-registered (critic CASE edge case).
- Distractors are twice as many, matched on file and degree, so the positive base rate is 33%. That limits yes-bias inflation.
- Bodies are truncated to 25 lines. Canonical order is (file, line). Each item is ≤5k tokens.
- **A further 100 synthetic items** come from a **held-out topology family**, compiled from a 7th repo outside S_topo. They are used for H6 and the Stage-B fallback trigger, never for H1–H4.

**Contexts:**

| Context | What is shown |
|---|---|
| real | Unchanged bodies |
| GF | Names and signatures only |
| RW | Bodies with call-site callee names permuted by a degree-preserving rewire of the static call graph over the whole pool plus its distractors. Labels remain the true trace |

**Readouts (all on the base model; trained arms get A in all three contexts plus Lmax in the real context):**

| Readout | Definition |
|---|---|
| **A** | Atomised, one p per candidate. This is the trained format |
| **L** | A list with the same pool, order and token budget, scored by rank order |
| **Lmax** | Max-over-generations readout (red team, from Ma & Chochlakis): p_v = max over emission steps of v's first-token probability (vLLM logprobs=20, missing = 0), averaged with v's frequency across 8 sampled lists |
| **L-EOS** | The list-closing token banned through `logit_bias`, forcing 10 more items (critic CASE fix 2). It discriminates an emission-termination prior from perception failure |

**Metrics:**
- **Primary:** item-level average precision (AP) of p, averaged over items. It is threshold-free, dense, and paired across arms.
- **Secondary:** recall at matched precision, where the threshold is set so that micro-precision equals base-L precision on a 60-item calibration split; log score; and AP on IDF-filtered positives (excluding hubs).
- **Structure use:** Δ_struct = AP(real) − AP(RW), and Δ_GF = AP(real) − AP(GF).
- Everything is stratified by depth, edge type (lexical vs dark) and direction. That covers ideas 2, 3 and 5 at no extra cost.

**Statistics.**
- Paired item differences, with a cluster bootstrap over repos (10k resamples). A mixed model with a repo random effect is the robustness check.
- Power: paired SD of the AP difference is GUESSED at 0.15. At n=300 that gives SE ≈0.009 before clustering, so the MDE is ≈0.025. With 8 clusters the design effect could be ~1.5–2 (A15). The 0.03 TOST margin was chosen with that in view.

### C10. Edge-use instrument (measurement only; never a training signal in v1) [NARROWED]

**Inputs.** 40 base trajectories and 40 Stage-B (b) trajectories, both from rollouts already generated in C6, on synthetic tasks where G is exact. No new rollouts are needed. This addresses X4 (raw dev traces are not local).

**Six prefill views** of the frozen W4A16 model:
- G;
- ρ1 and ρ2;
- plausible-ρ, which swaps only among callees named in the same caller file (critic coherence fix);
- ∅, a node list;
- none.

**Privileged block.** The 3-hop executed neighbourhood of the failing test's entry, including sibling branches that do not reach gold. It is de-leaked: selection is independent of gold. It is placed **after** the system prompt and issue so that prefixes are shared across views.

**Per-action edge credit:** `c_t = δ^G_t − δ^ρ_t`, plus a dose-response at rewired fractions 0.25 / 0.5 / 1.0.

**Gates:**
- **K0 (Salesforce three-checks Q1):** trajectory-level AUC of Σc against success must have a 95% CI excluding 0.5, with a point estimate ≥0.6. Otherwise per-action edge-use is not interpreted.
- Pre-registered effect-size kills (critic SAST fix 4):
  - Σ|c| < 2 × the 95th percentile of the ρ1−ρ2 null;
  - δ^∅ explains ≥80% of the variance of δ^G.

**Use.** If K0 passes, report the per-action edge-use concentration (≥2 hops, and CANDIDATES lines) for base vs (b). This is a mechanism figure supporting H4. It costs prefill only.

### C11. Development-set guard [KNOWN]

- The best arm (pre-registered: (b), unless H1 picks another on the item evaluation) runs the deployed harness on all 129 dev tasks with the CANDIDATES prompt.
- It is compared with the stored r3_v5 base.
- The prompt and the LoRA change together, so the guard measures the *deployable system*, not the attribution between the two. This is disclosed.
- Rich (48) and requests (13) are reported separately. Training never saw any dev repo, so there is no contamination by construction.

---

## 4. Data flow, one arm, one iteration

1. **CPU.** Draw a source graph from S_topo. C2 produces (b)/(c)/(d) siblings with the same seed. C3 compiles them. C4 injects bugs, with depth matched between (b) and (c). C5 verifies.
   - Output: 20 tasks per round per arm, plus 200 call-set items per arm (auto-labelled).
2. **GPU, generation.** One vLLM session per round serves all three arms (multi-LoRA in round 2).
   - Rollouts: 20 tasks × K=4 per arm.
   - Items: 200 × K=4 per arm (prefix-cached).
3. **CPU, reward.** Apply hidden tests, run the P2P guard, parse CANDIDATES, compute R(τ), Â_τ and Ã_t. Items get S_v and their recombined targets.
4. **GPU, training.** Stage A on round-1 data → LoRA_A. Round-2 generation with LoRA_A. Stage B → LoRA_B. Export PEFT.
5. **GPU, evaluation.** C9b runs base once and the three LoRA_B adapters in one multi-LoRA session. C10 reuses trajectories. C11 runs the best arm only.

---

## 5. GPU-hour budget (whole study; cap 25)

Throughput inputs:
- Synthetic rollouts at **60 per GPU-h central** and 40 floor (critic). Upper bound INFERRED: ~34 tok/s single-stream decode (VERIFIED HQ Oct 7) × 16 streams, with prefill at 1.5–3k tok/s, suggests 100–200/h is possible, but the floor is what is budgeted.
- Prefill at 2.25k tok/s central (critic X8: 1.5–3k).
- Training at 500 tok/s central (400–650).
- Each GPU session costs ≈0.15–0.2 h of start-up (vLLM start 480 s, VERIFIED).

| # | Line | Week | Central | Pessimistic | Notes |
|---|---|---|---|---|---|
| B1 | Training-feasibility probe (memory and speed at 4/8/16k, SDPA backend, HQQ-vs-vLLM agreement, 20-step export→serve) | 1 | 1.0 | 1.0 | Gates everything (critic X1) |
| B2 | GCSR pilot: 30 zero-shot tasks (depth 1/2/4 × 10, arm-(b) topology) for **throughput and difficulty calibration** | 1 | 0.5 | 0.75 | Abort rule G1 |
| B3 | Gate K1/K2: 60 more normal tasks (making 90 = 30 per depth) + 30 GF-named-bug tasks | 2 | 1.5 | 2.25 | Critic GCSR fix 4 |
| B4 | Base item evaluation: 300 real × {A, L, Lmax, L-EOS} real context + {A} GF and RW, + 100 synthetic | 2 | 0.9 | 1.3 | ≈4.5M unique prefill; L shares A's pool prefix |
| B5 | Arm rollouts: 3 arms × 2 rounds × 20 tasks × K=4 = 480 | 2–3 | 8.0 | 12.0 | Largest line; descoping ladder below |
| B6 | Item-sample generation for training: 3 arms × 2 rounds × 200 items × K=4 (same sessions as B5) | 2–3 | 1.2 | 1.6 | Prefix-cached across K |
| B7 | Training: 3 arms × (Stage A ≈0.35M tok + Stage B ≈1.2M tok incl. π_A recompute) + loads | 2–3 | 4.2 | 5.0 | Equal tokens across arms |
| B8 | Trained-arm item evaluation: 3 arms × (A in real/GF/RW + Lmax real) + synthetic, one multi-LoRA session | 3–4 | 2.0 | 2.9 | |
| B9 | Dev guard: best arm × 129 tasks | 4 | 2.5 | 3.4 | Base = stored r3_v5 |
| B10 | Edge-use instrument: 80 trajectories × 6 views + dose-response, prefill only | 4 | 0.5 | 0.8 | Reuses B5 trajectories |
| | **Subtotal (core)** | | **22.3** | **31.0** | |
| B11 | Seed replicate of arm (b): same rollouts, new shuffle and init, Stage A+B + item evaluation (A, real context) | 3–4 | 1.6 | 2.0 | Measures the training-noise floor against the H3 margin |
| B12 | Reserve (session failures, re-runs) | — | 1.1 | — | |
| | **Total** | | **25.0** | **33.0 → descoped below 25** | |

**Descoping ladder.** Apply the steps in order until the projected total is ≤25.

| Step | Trigger | Change | Saving |
|---|---|---|---|
| D1 | Measured rollout rate in [40, 60)/h | Rounds shrink to 14 tasks × K=4 (336 rollouts) | B5 ≈7.5–8.4 at 40–45/h |
| D2 | B11 not affordable | Drop B11. H3 is then reported with a caveat: "single seed, noise floor unmeasured" | −1.6 to 2.0 |
| D3 | Still over | Skip round 2. Stage B runs as a second, clipped pass over round-1 rollouts (off-policy against π_base, ratio recomputed). Disclosed as weaker | −⅓ of B5, −part of B6 |
| D4 | Still over | Dev guard on a pre-registered stratified 64-task subset. Non-inferiority power is lost; reported descriptively | ≈−1.2 |
| D5 | Rate < 40/h (critic abort) | **Abort the three-arm factorisation.** Run (b) vs base only. H1, H4 and H6 remain testable; H2 and H3 are dropped | — |

Week fit: week 1 is 1.5 (B1 + B2) **exactly**. Weeks 2–4 (≈23.5) are well inside ~30 per week. That leaves the Nov 5 experiment freeze about one week of slack.

---

## 6. Gates (decision points before each spend)

| Gate | When | Pass rule | If it fails |
|---|---|---|---|
| G0 probe | B1 | 16k fits in ≤22 GB per GPU. ≥250 tok/s. Mem-efficient SDPA on head_dim 512 forward and backward. HQQ-vs-W4A16 next-token agreement ≥97%. Exported adapter routes in vLLM | Per the feasibility doc: FlexAttention for global layers; 8k cap; NF4 → HQQ |
| G1 throughput and calibration | B2 | ≥40 rollouts/GPU-h (≥60 for the full plan). Zero-shot success at depth 1 in [0.3, 0.9] and at depth 4 in [0.05, 0.6] | <40 → D5. Out of band → recalibrate size and contract density on CPU, then re-pilot 10 tasks (0.2) |
| G2 K1 / K2 | B3 | K1: the logistic depth slope on 90 tasks is negative with a 95% CI excluding 0. K2: GF-named success − normal success ≥ +10 pp at depth ≥2 | K1 fails → the gym does not exercise traversal, so **stop**; report as a negative finding about compiled-topology tasks. K2 fails → the issue already leaks location; fix the issue generator |
| G3 base item sanity | B4 | Base AP(A, real) in [0.40, 0.85]. Positives per item ≥5 on ≥90% of items | Re-sample pools or depth strata on CPU |
| G4 training sanity | After Stage A of (b) | Synthetic held-out success ≥ base + 10 pp and item log score not worse | Debug before training the other arms (arms are trained after (b)'s Stage A, in the same session) |
| G5 Stage B | After Stage B | Clip fraction ≤20%. Synthetic item log score not below Stage A | DPO fallback (pre-registered) |
| G6 instrument K0 | B10 | AUC CI excludes 0.5 and point estimate ≥0.6 | Report "the self-teacher instrument does not track success here", with no edge-use claims |

---

## 7. Kill test this week (1.5 GPU-h) and the CPU work that precedes it

**CPU, days 1–4:**
- C1 tracer with threading and code-object identity, run on S_topo candidates.
- C2 with acyclicity and depth matching, with unit tests: no cycles; per-type degree preserved exactly; depth KS reported.
- C3 compiler v0, with all four dark edge types and contracts.
- C4 with O1 and O2 (DPCS); O3 can follow.
- C5 necessity checker.
- 30 pilot tasks.
- The C9a item builder started on S_eval.

**GPU, one session:**
- B1 probe (1.0).
- B2 pilot (0.5). This measures **rollouts per GPU-h** and the depth calibration. That one number decides whether the three-arm design (D5) is affordable. The critic flagged it as the binding unknown.

---

## 8. Why this keeps the W4A16 model near SOTA

1. **Exact grid.** Adapters are trained against the served int4 values (HQQ g32 from a dequantised W4A16). Ratios are computed in the trainer, so there is no kernel mismatch inside the update.
2. **On-policy data and a trust region.**
   - Stage A uses the model's own successes.
   - Stage B uses π_A's own rollouts with clip 0.2 and a k3 KL penalty at 0.02, a small lr, and r = 16.
   - This is the "RFT moves the model less than teacher SFT" argument (INFERRED). The SWE-Gym counter-risk is guarded by G4, G5 and H5.
3. **Small total update.** About 1.5M tokens per arm, with ≤30% auxiliary items.
4. **Deployment unchanged.** The agent stays declarative with the same tools. Only the LoRA and one prompt field change. The verbalised p is robust to quantisation.
5. **Reported gap (optional, if the reserve allows).** Run the B9 guard once with the bf16 QAT-unquantised checkpoint plus the same LoRA on a 32-task subset (≈0.6 h), to report (bf16 + LoRA) − (W4A16 + LoRA).

---

## 9. What changes for other models and hardware

| Dimension | Gemma-4-31B W4A16 on 4×L4 (this instance) | General rule |
|---|---|---|
| Loop | Offline generate → train, 2 rounds | Online GRPO: same R and Ã_t, K=8, task server with adaptive depth (C8 drop-in). Rounds become steps |
| Base used for training | Dequantised W4A16 → HQQ g32 (exact served grid) | Always train on the *served* quantisation grid; with bf16 serving, plain LoRA |
| LoRA | r = 16, all linear modules, no DoRA, export without bias or modules_to_save | Rank by token budget: ≤2M tokens → r = 8–16; ≥20M → r = 32–64. Targets = all linear modules the server can route |
| Sequence cap | 16k training / 32k serving; trajectory cap 12k; turn cap 40 | Trajectory cap = training cap − 25%; compiler repo size scales so that navigation still requires tools |
| Counts | 20 tasks × K=4 × 2 rounds per arm; 300 evaluation items | Scale tasks with rollouts/GPU-h; keep **equalisation** and the **≥300-item paired evaluation** fixed |
| Attention quirks | SDPA mem-efficient for head_dim 512 | Model-specific; not part of the method |
| Language | Python compiler and tracer | A port needs a compiler backend (body templates, four dark-edge idioms) and a tracer. The C2 and C4 graph operators are language-agnostic |
| Graphs beyond code | — | Any *executable* DAG (build systems, dataflow pipelines, workflow engines) supports C2 to C5 unchanged. Non-executable graphs (knowledge graphs) can use C2 and C9 but not the DPCS-as-bug operator |
| Reward | Hidden F2P tests + P2P guard | Any verifier that is exact on the compiled world |

---

## 10. Pieces deliberately left out of v1 (and why)

| Piece | Why it is out |
|---|---|
| **SAST as a training signal** | The red team found RLCSD and CROP already use it (closeness 2.5), and Salesforce found AUC 0.505 on AIME. It is kept only as the C10 instrument, gated by K0 |
| **SENSOR** (IGPO-style spectrum gain) | The red team's NARROW verdict needs a self-belief-IG arm, and sparse pre-edit experiments (40/129 ran any test, VERIFIED) make the signal thin. It could be an additive `Ã_t` term in v2: the online drop-in is already shaped for it |
| **ALIAS** | Critic-infeasible (23–30 GPU-h) and of low relevance to graphs |
| **ERC forecast items** | The kill-test tautologies (critic) and SWM prior art. Only the **DPCS operator** survives, now used as a bug generator inside compiled worlds, where type compatibility, necessity and labels are exact by construction. That is not a PMT study |
| **Real-repo RFT arm (a)** | X6 cost: 15–37 GPU-h for resolved replay, plus contamination. So P1 ("≥70% of the real-RFT gain") is **not** claimed. The claim is relative: (b) vs (c) vs (d) vs base |

---

## 11. Assumptions the design rests on (attack these)

| ID | Assumption | How it is checked | If false |
|---|---|---|---|
| A1 | Synthetic rollouts reach ≥60/GPU-h (≥40 floor) on 4×L4 with 8–16 streams | G1 (B2) | D1 → D5 |
| A2 | Prefill reaches 1.5–3k tok/s, so item evaluation stays ≈2.9 GPU-h in total | B2 logs | Fewer contexts on trained arms (drop GF there first) |
| A3 | The training stack works at 16k on L4 (head_dim 512 SDPA, memory ≈20 GB, ≥250 tok/s) | G0 | 8k cap with trimming (disclosed confound) |
| A4 | Base success on depth-2–4 compiled tasks sits in a band where K=4 groups have outcome variance (≈0.15–0.6) | G1 calibration; fraction of informative groups reported | Raise K on all arms equally, or rely on the η set term |
| A5 | **Protocol transfers without content**: skills learned on vocabulary-only bodies transfer to real code. SWE-Playground's 32B gap (−9 pts with *content-rich* synthetic data) is prior evidence against | H1 itself | H1 fails; the paper becomes "content dominates, quantified" |
| A6 | **Call-set AP on real items is a meaningful proxy for graph reasoning in the agent.** Caution: localiser function-hit did not predict resolve (27% vs 28%, VERIFIED critic L1). The claim is scoped to item-level reasoning; resolve is only a guard | H5 descriptive plus the ≥2-gold slice | The paper's relevance to SWE weakens; the item-level finding stands |
| A7 | Content-free identifiers do not make the tasks so unnatural that the agent's protocol on them differs from its protocol on real repos | Compare tool-call mix (read/grep/run) on synthetic vs dev trajectories | Add light semantic templates (one change, v1.1) |
| A8 | The depth-matched DPCS rewire changes *only* wiring-specific statistics (motifs, clustering, which-node-feeds-which). It does not change depth, per-type degree, size or bug depth | Report motif counts, clustering and assortativity for (b) vs (c) | If (c) differs on other axes, H3 attribution narrows to "that bundle" |
| A9 | Traces on S_eval are complete enough for labels (threading covered; C code invisible) | Per-repo count of under-labelled C-boundary calls; drop repos with >10% | Fewer repos; check power |
| A10 | Training-seed noise is below the H3 margin (0.03 AP) | B11 seed replicate | H3 inconclusive (reported, not claimed) |
| A11 | About 1.5M tokens of LoRA r = 16 updates move item AP measurably without regressing resolve | H1 and H5 | Raise r or tokens (one change) |
| A12 | The vLLM W4A16 vs HQQ-trainer log-prob mismatch at the zero adapter is small (median \|Δ\| ≤0.05 nats/token) | B1 probe, step 3 | Use the NF4/HQQ choice that minimises it; ratios are already recomputed in the trainer |
| A13 | Five-level verbalised p gives enough resolution for AP. Ties are handled by averaging over tie orders | Report the tie fraction; Lmax gives continuous scores | Use a log-prob readout of the EDIT token as a robustness check |
| A14 | Matched-precision and AP readouts neutralise yes-bias (per-item queries inflate "yes") | The 33% base rate; L-EOS arm; precision reported | Report precision@k as well |
| A15 | ≥8 eval repos are enough clusters for the bootstrap; design effect ≤2 | Mixed-model ICC reported | Widen the TOST margin, which is disclosed |
| A16 | SCC condensation (no recursion) and DAG-only compilation do not remove transfer-relevant structure | Fraction of edges lost (<3% GUESSED) | Bounded-recursion compiler in v1.1 |
| A17 | There is no leakage path: synthetic content cannot contain dev or S_eval code; S_topo ∩ S_eval = ∅; identifier pools come from S_topo only | Static check of identifier overlap with S_eval | Re-sample the pools |
| A18 | The CANDIDATES prompt field does not by itself hurt the base agent | Confounded in H5 (disclosed) | A 32-task base + CANDIDATES run from the reserve (≈0.6) |
| A19 | DPCS two-site bugs are solvable from contracts plus traces, rather than by guessing | Zero-shot DPCS success in B2/B3 >0 at depth 2; a solved trajectory inspected for a contract-bisection pattern | Raise contract density for O2 tasks |
| A20 | Dark-edge share and hub ratio of compiled repos match S_topo (KS on degree distribution, critic GCSR hubs row) | C2/C3 report | Re-weight the generator |

---

## 12. Paper framing (one line each)

| Rubric axis | Framing |
|---|---|
| Novelty | Topology as an experimental treatment for agent RL, through compiled executable worlds. DPCS as a structure-only, jointly-necessary bug operator. A rewired-context structure-use readout on real code |
| Generality | Any model or quantisation (train on the served grid), offline or online GRPO, any executable DAG domain |
| Verifiability | Exact labels by construction; pre-registered H1–H6; dense paired evaluation at n ≥ 300; every control built in |
| Even if it fails | H1 fails → "content dominates" (quantified at 31B, int4). H3 fails → real topology carries transferable structure beyond degree and depth. K1 fails → compiled-topology tasks do not exercise traversal. Each is reportable |
