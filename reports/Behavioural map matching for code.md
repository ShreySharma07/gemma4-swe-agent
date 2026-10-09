# Decode issue behaviour onto call graphs

**Bottom line.** The topic chosen is **Code Comprehension**. The brief gave the inventor "novel code-graph generation, parsing and embedding". The four earlier reports had already worked Tuning & Optimization and Graph Reasoning hard, and those rounds ended in audits and analysis instruments, not techniques. Tasks & Benchmarks yields a measurement, not an architecture. (The notes record the brief, not the reasons for setting the other topics aside. Those reasons are this report's reading of the earlier reports: *PEFT and RL for SWE agents* and *Interventional step credit audit* for Tuning & Optimization, and *PEFT RL for directed graph reasoning* and *LLM graph reasoning for code* for Graph Reasoning.) Twelve ideas were cut to six, and critic, red team and five break rounds reduced them to one architecture: **Behavioural Map-Matching (BMM) with signed behavioural evidence**. An LLM (Gemma-4-31B) reads a traceback-free issue and imagines its execution as a short path of steps. Each step names a guessed function and carries the effects the issue *reports* (+) and the effects it *expected but did not see* (−). BMM scores every step against per-function effect probabilities over a fixed alphabet (exception classes, returns None or self, mutates an argument, writes a self attribute, warns). For "−" steps it adds a noisy-OR negative finding and a Chucky-style peer contrast. It then compounds the evidence along the repository call graph by exact HMM forward–backward decoding, with delete and skip states, and returns a culprit-weighted function ranking. No tests run at inference. Novelty is **a combination at closeness ≈2**: every part has a closeness-2 neighbour, and no prior combines them (abstract-level absence over ≈55 logged queries). Of the four novelty criteria, **combination and formulation are verified on paper, the mechanism is proven on paper with its conditions pending, and demonstrated advantage is NOT MET, because nothing has run.** The final verifier returned **NEEDS-FIXES**. All eleven fixes (F1–F11) are applied in v3.1. The honest topic fit is decision (b): BMM is a localisation and graph-inference technique, a loose fit to Code Comprehension and a strict fit to Graph Reasoning. The cheapest first demonstration is **K3c**, a CPU-only sibling-separation pilot: about 2 kernel-hours, $0, by **Oct 17**.

**Headline sentence (v3.1 §0.5.3, with the verifier's five edits applied).**

> **We take map-matching (HMM map-matching from GPS; Markov/DP report-step alignment from FSE'24) and adapt it to issue localisation on repository call graphs. The observations are an LLM's prose-imagined execution path, whose steps carry issue-stated signed effects. These are scored against per-function effect probabilities (AST rules, plus an execution-trained model if K4 passes), using a noisy-OR negative finding and peer contrast (Chucky) applied to issue-stated missing effects. This yields a behavioural channel robust to code-side renaming, and we test whether it lowers gold-function rank beyond the same decoder with names and body text alone (C-BEH).**

In the requested "We take X and make it work for Y by Z" form: we take **map-matching (X)** and adapt it to **function-level issue localisation on Python call graphs (Y)** by **scoring the signed effects an issue states, attached to an LLM-imagined path, against predicted per-function effect probabilities (Z)**. The verifier replaced "make it work" with "adapt it to", because "make it work" presumes a result that does not exist yet ([verify_v3 §4.3](<../research_notes/Code graph comprehension architecture/verify_v3.md>)).

**Novelty status, exactly as in v3.1 §0.6.5.**

| Criterion | Status |
|---|---|
| 1. Combination: no prior combines C1–C7 | **Verified on paper** (abstract-level absence; residual risk stated) |
| 2. Formulation | **Verified on paper** (§0.6.2) |
| 3. Mathematical mechanism | **Verified on paper** as Props 0–4 with proofs/sketches; **conditions pending experiment** (K3c, P1a, C-DEC by T/b) |
| 4. Demonstrated advantage | **Not met.** Pending K3c (Oct 17), the oracle pilot (Oct 22–23), the LLM pilot (Oct 24–25) and K11 (Nov 3) |

By the user's own rule, a combination counts as new only if all four criteria are verified. **BMM is therefore not yet new.** It is a fully specified, falsifiable candidate whose fourth criterion depends on K3c and K11.

**Topic-fit decision (b).** BMM is a localisation decoder *over* a code graph. Graph construction (AST resolver, HEAP import-time walk, per-site candidates) is declared infrastructure, because the red team found it near prior art. The verifier judged the fit to "code-graph generation/parsing/embedding" **not met in the strict sense, partial in a generous reading** ([verify_v3 §5](<../research_notes/Code graph comprehension architecture/verify_v3.md>)). v3.1 therefore submits it as *inference over code graphs*: an issue parsed into a graph-grounded path. The per-function effect vector q_v is evaluated as a representation at K4 and released as a dataset, but it is never the headline. Note a labelling trap: the verifier's option "(a)" is v3.1's "(b)". Both mean "keep the graph-inference framing".

**Nothing has been run.** No GPU-hour has been spent, no kill test has executed, and every effect size below is a guess.

---

## Round 0: scouts found that names, not structure, carry localisation

**Findings.** Two scouts collected 22 anomalies each, on graph generation and parsing (G-A1..A22) and on embedding (E-A1..A22). Seven of them shaped everything downstream.

| ID | Anomaly | Number | What it contradicts |
|---|---|---|---|
| G-A1 | PyCG recovers half of runtime edges on real projects | **49%** of DynaPyt edges; 59% of missing callees are built-in-type methods ([DyPyBench](https://arxiv.org/abs/2403.00539)) | "Static parsing gives a sound superset" |
| G-A3 | Edge precision collapses along chains | **0.99 per edge → 0.19 per chain** ([Jarvis](https://arxiv.org/abs/2305.05949)) | Multi-hop expansion inherits edge precision |
| G-A4 | LLMs over-extend along the static graph | Runtime data flow **1–2%**; Gemma-4-31B 19.17% vs 4B 17.29%; **260 predicted vs 44 actual** def–use pairs ([SWE-FLUX](https://arxiv.org/abs/2609.28449)) | Bigger models read runtime structure better |
| G-A8 / A11 | Less accurate, query-selected graphs win | LLM partial graph beats static global graph, 7.4× faster ([DyCoder](https://arxiv.org/abs/2608.01927)); RepoGraph DR@15 **0.127 vs BM25 0.296** ([FeatLens](https://arxiv.org/abs/2609.26480)) | Construction fidelity drives downstream gain |
| G-A14 | Containment edges are worth nothing; lexical index matters most | Containment-only 66.42 ≈ no graph 66.06; no BM25 index −11 pp ([LocAgent](https://arxiv.org/abs/2503.09089)) | Any graph helps |
| E-A1 / A2 | Issue-to-edit retrieval collapses, and the fix is on the query side | Qwen3-Emb-8B NDCG@10 **71.7 → 20.3** ([CORE-Bench](https://arxiv.org/html/2606.11864)); query-side LoRA lifts MRR **0.29 → 0.51–0.57** over frozen code vectors ([HyperFL](https://arxiv.org/html/2608.02967)) | Code-side representation is the bottleneck |
| E-A17 | Reachability is decodable but unused | Transitive AUROC **0.986** vs direct 0.869; realised tool call changes **0%** under patching ([2605.25310](https://arxiv.org/html/2605.25310)) | Represented structure is used structure |

Our own HQ data added two facts. Only **21%** of dev issues name the gold node, and the supplied node embeddings are low-pass (each node ≈ mean of its neighbours, cosine 0.85–0.97). The scouts' synthesis was that the entry point and path validity decide downstream gains, while edge precision and recall do not, and that issues describe *behaviour* while code embeddings encode *names*.

**Decision.** Anomalies that touch generation, parsing or embedding, and that appear in both the literature and our data, go to the inventor.

**Next step.** Inventor Phase 1.

---

## Rounds 1–2: twelve mechanisms became six techniques

**Findings.** The inventor gave each of six anomalies (AN1–AN6) three competing mechanisms. It inverted eight unstated assumptions; for example, "the graph is built before the query" became "the query generates the graph region". It also sketched an ideal observer: the interventional dependency graph, with a cheapest useful approximation of "import-time heap for edges the runtime already resolved, plus an amortised execution-trained predictor for the rest". Distillation gave twelve techniques. T1 HEAP, T2 LATENT, T3 HYTRACE, T4 DECONV, T5 EFFECT and T6 CHAIN were refined. T7 ASYM, T8 SITE, T9 TRACEBACK, T10 KILLVEC, T11 RESIDUAL and T12 EXECCARD were folded in or dropped. The inventor applied the trap list from *PEFT RL for directed graph reasoning*: ranks over fixed pools instead of gameable set-F1 rewards, a graph-free arm plus renaming in every kill test, degree-preserving rewires as one-change controls, and budgets that count epochs.

| Refined | Mechanism bet | Surprising prediction (GUESSED) | Kill |
|---|---|---|---|
| **LATENT** | Gemma's residual stream encodes runtime-only call edges that it cannot verbalise | Bilinear probe AUROC ≥0.75 on runtime-only edges; probe > verbal by ≥0.15 | AUROC <0.65 or within 0.05 of the DPR-label control |
| **HEAP** | Static misses are already resolved in the imported heap | ≥40% of statically missed runtime edges recovered with no tests | <20% recovered, or precision −5 pp |
| **HYTRACE** | Issues describe a scenario whose path shape survives wrong names | Decoded imagined path beats names-only and BM25, and the gain vanishes on a DPR graph | Decoded − names-only <3 pp, or decoded ≈ DPR |
| **DECONV** | Supplied embeddings are a linear low-pass filter of text features | Inverting the filter improves co-edit retrieval, and inverting with a rewired graph hurts | Ĥ(true) − Ĥ(DPR) <0.03 MRR |
| **EFFECT** | Issues and code share an effect vocabulary that names never enter | Effect retrieval invariant under renaming while BM25 drops | Coverage <25%, or fused gain <2 pp |
| **CHAIN** | Chain collapse comes from site-correlated ambiguity | Site-normalised mass beats per-edge pruning on chain precision | No gain at matched recall |

The six-technique roll-up was **8.2 / 24.6 / 37.5 GPU-h**. HYTRACE's LoRA alone (4.3 GPU-h per epoch) threatened the cap ([inventor_phase1_2](<../research_notes/Code graph comprehension architecture/inventor_phase1_2.md>)).

**Decision.** Send all six to Phase 3. No training until its kill passes.

**Next step.** Critic and red team in parallel.

---

## Round 3: critic and red team narrowed four, killed two, and promoted LATENT and HYTRACE

**Findings.** The critic scored falsifiability, surprise, prior-art risk and generality. It also fixed three global defects: GPU units (everything in Kaggle 4×L4 node-hours, NH), power (a 3 pp threshold on 93 tasks sits inside a ≈4.8 pp SE), and renaming logic (R1 consistent vs R2 code-only renaming; consistent renaming barely hurts BM25). The red team asked one question of each technique: has someone already exploited the *same mechanism*? Both used OpenReview's API (CR-6, CR-9; RT-6..RT-10, RT-37, RT-38).

| Technique | Critic verdict (surprise 0–5) | Red-team verdict (closest prior, closeness) |
|---|---|---|
| LATENT | **SURVIVE-BY-SURPRISE**, narrowed (4) | SURVIVE, narrowed: tool-call edge probes ([2605.25310](https://arxiv.org/html/2605.25310), 2); Troshin & Chirkova and Anand et al. DFG probes (2) |
| HEAP | SURVIVE, narrowed (2) | **NARROW**: Jelly approximate interpretation (PLDI'24, 2.5) already lifts runtime-edge recall **75.9% → 88.1%** at −1.5% precision with no tests (RT-19); HeapDL (2.5); PoTo (2) |
| HYTRACE | **SURVIVE-BY-SURPRISE**, zero-shot only (4) | NARROW: Newson–Krumm map-matching (2); RoG/GCR plan-then-ground (2) |
| DECONV | **KILL** (2) | NARROW: GDN and blind graph-filter deconvolution (2.5) |
| EFFECT | NARROW, as a component (3) | **SURVIVE**: EXCEPT and SBEST (1.5); no effect-signature localiser found (RT-25, RT-42) |
| CHAIN | KILL standalone → component (2) | NARROW: per-site softmax already in [AFLGopher](https://arxiv.org/abs/2511.10828) (2) and [GRAPHIA](https://arxiv.org/abs/2506.18191) |

The top two by surprise, LATENT and HYTRACE, got the extra round. LATENT was given a strict stratum hierarchy (lexical ⊂ static ⊂ HEAP ⊂ call-time-only), executed-site negatives, order randomisation, position-only and additive-probe controls, and an NF4 → W4A16 transfer check. HYTRACE was given PPR-from-anchors, shuffled-order and random-path controls, ≥300 public tasks, full-repo ranks and an on-path/off-path stratum. Its LoRA was gated on measured FSDP throughput ≥400 tok/s. The critic proposed one combination, "graph as decoder". The red team proposed LATENT + EFFECT as the paper, with HYTRACE as an ablation ([phase3_critic](<../research_notes/Code graph comprehension architecture/phase3_critic.md>), [phase3_redteam](<../research_notes/Code graph comprehension architecture/phase3_redteam.md>)).

**Decision.** The architect saw that the two recommendations need each other. EFFECT alone is too coarse to rank: a 16-bit signature over ≈4k functions leaves huge ties. HYTRACE alone is map-matching. Effects attached to a decoded *sequence* gain resolution, and a name-free emission is the part of the decoder that is not map-matching. LATENT cannot run at inference, because vLLM exposes no mid-layer states, so it became a mechanism side-study.

**Next step.** Architect v1.

---

## Round 4: v1 broke on its own algebra, its controls and its calendar

**Findings.** v1 BMM decoded k = 5 imagined (symbol, effect) paths with an HMM whose emissions multiplied a name term by a behavioural term. The behavioural term used effect frequencies from traces of other repos. Its claims were C1 (BMM > PPR), C2a/C2b (behaviour adds; execution calibration beats AST rules), C3 (order and structure carry the gain) and Study B (LATENT). Budget: 13.4 NH central. Three breakers attacked it.

| Breaker | Top breaks |
|---|---|
| Logic, L1–L46 | **L35 (Fatal):** C1's control, PPR, never sees the imagined path, so BMM can win by reading Gemma's (possibly memorised) names. **L15+L16 (Fatal):** labels encode *normal* behaviour at ≥10% frequency, while issues report rare, buggy, polarity-inverted behaviour. **L17 (Fatal):** behaviour is lexical in disguise under R2 renaming. **L7+L8+L11:** stutter collapse, a BM25-only teleport and a max-over-steps readout make "structure corrects wrong names" impossible in the model's own algebra. **L28:** CAL-tune leaks into training traces |
| Method, M§ | **≈32–45 person-days** against 27 calendar days shared with the main track. Execution-validated EVAL yield 214–491, 60% Django/sympy, CPU undercounted 7–30×. A ≤6k skeleton cannot hold Django. The C1 comparator is wrong (no LLM baseline). C2b power 0.17–0.58. Recount **21.6 NH central** |
| Prior art, BR3 | Overall **SURVIVE, narrowed, closeness 2**. Euler (FSE'19, NL steps aligned to an execution graph, 2, [1906.07107](https://arxiv.org/abs/1906.07107)); LocAgent's prompt already says "reconstruct the execution flow" (1.5–2); the SherLog → Pensieve → Pathidea → CrashLocator family (symptom → path, 1.5–2); IPA-GNN and CodeFlow for execution-trained exception prediction (2, [2203.03771](https://arxiv.org/abs/2203.03771), [2408.02816](https://arxiv.org/abs/2408.02816)); [RepoReasoner](https://arxiv.org/abs/2607.25996) for predicted vs traced chains (2; precision ≈0.80, recall ≈0.31). **Behavioural emissions: SURVIVE, closeness 1.5**: the least crowded piece |

**Decision.** Keep one architecture and rebuild the novel piece as **polarity-aware, base-rate-normalised behavioural emissions**. Rebuild the decoder with insert states, a mixed teleport and a culprit-weighted readout. Give every primary claim a control that sees the same imagined path. Move the headline to a **static, gold-mapped EVAL of ≥600 traceback-free tasks over ≥25 repos**. Cut Study B (LATENT), JS, pre-existing-test traces, PyCG/PoTo and the CoSIL run.

**Next step.** v2 and a second break.

---

## Round 5: v2's fixes were renamed, not fixed, and the schedule did not fit

**Findings.** v2 claimed 9.9 NH and 15.5 pd, with three primaries: C-BEH (behaviour, under R2), C-DEC (sequence decoding vs a bag) and C-USE (vs direct prompting). Two breakers attacked it.

| Breaker | Top breaks |
|---|---|
| Logic, method, stats and compute, N1–N30 | **N1:** the base-rate likelihood ratio is a per-step constant that cancels between nodes, so omnivore functions are untouched. **N8:** insert states recreate the v1 stutter, and the entry readout returns. **N14:** X-BAG⊗PPR is a log-product over steps, so C-DEC changes the evidence algebra as well as the sequence. **N22:** C-BEH's four-condition pass rule has joint power ≈**0.3–0.5**. **N21:** C-USE can "win" at ranks 50–2,000 that no agent reads. **N27:** person-days are **≈26.75 central, not 15.5** |
| Prior art, BR3b | The novel piece moves **1.5 → 2**. Zhang et al. FSE'24 aligns report steps onto a UI model with a Markov model + DP and missing-step fill, at 2.5 for the decoder ([PDF](https://par.nsf.gov/servlets/purl/10544604), BR3b-12). QMR-DT and possibilistic MBD already use negative findings (2). Liblit's Increase already does base-rate scoring of code (2). Chucky already does peer contrast (2). Pathidea uses only logs and stack traces, not prose, and stays at 2 ([PDF](https://petertsehsun.github.io/papers/PathideaTSE2021.pdf), BR3b-1c). **No SWE localiser parses issues into signed effects** (BR3b-3, BR3b-8, BR3b-15, BR3b-17) |

**Decision.** v3 starts from a "v2-lean" scope (AST + Jedi graph, HEAP only where images exist, R2 only, EVAL-exec 60). It uses one control per claim and a single truncated log₂-rank metric. X-BAG becomes the same HMM with a rank-1 PPR kernel. Delete states and a local teleport are added. Implicit raises come from receiver-typed AST rules and intended raises from g_φ. C-BEH becomes (BMM + BODY) vs β = 0.

**Next step.** v3, then an independent verifier.

---

## Round 6: the verifier found one inert novelty claim and two control leaks, and v3.1 fixed them

**Findings.** The verifier mapped every N-item and found six only partly fixed (N1, N2, N8, N10, N17, N-tune). It also found that v3 had silently dropped the ε-skip over decorators and several v1 diagnostics. Its sharpest findings were E-1 to E-5. **E-1:** v3's "local base rate" (subtracting the per-step median) is also ranking-neutral. That is now Proposition 0. **E-2:** the omnivore penalty μ′ sat outside β, so the β = 0 control still saw the issue's effect statement. **E-3:** a delete at the last step credited no node. **E-5:** under R2 the identity map for repo exceptions leaked names into the treatment arm. The power arithmetic checked out to ±0.01. The stratum was mis-sized: 35% of 600 is 210 tasks, not 250. G2 was under-counted by 0.5 NH, and K4–K6 could not land by Oct 17. Verdict: **NEEDS-FIXES**, about 0.5 pd of spec work, none of which blocks the Day-1 CPU kills.

| Fix | What changed in v3.1 |
|---|---|
| F1 | μ′ moved inside β, active only on effect-stating steps; X-β0 zeroes every q-dependent term |
| F2 | Under R2, repo exceptions collapse to "any repo exception", with no identity map; builtin-only C-BEH reported alongside |
| F3 | Delete at T credits its destination; ε-skip over wraps/decorators restored; P1a substitutes the last step; L4/L10/L11/L13/L18 diagnostics restored |
| F4 | X-BAG inherits BMM's orientation; RBO (p = 0.9, top 100) diagnostic that never overrides a significant C-DEC |
| F5 | "Local base rate" **removed from the novelty claims** (Proposition 0) |
| F6 | Stratum restated at 210 tasks; power at SD 0.6 / 0.78 / 0.9; three-way decision rule |
| F7 | Only BMM is tuned (3⁴ grid); controls inherit; decoding CPU and G2 recosted |
| F8 | K4–K6 re-dated to Oct 21–23; tracer added to the never-cut list; rung 8 demotes C-BEH unless share ≥37.5% |
| F9 | Missing-call (predecessor) credit on "−" steps; noisy-OR for dual-source bits; receiver-typed AST rules; deduplicated repo sites |
| F10 | Memorisation = B-RECALL hit ∧ gold not named in issue; graph-free control uses π |
| F11 | Five headline edits; rows for Liblit/IDF, SherLog family, RoG/GCR, AstroBR, FaR-Loc; new X-POOL ablation |

v3.1 also added the §0.6 four-criterion novelty verification (formal model, Propositions 0–4) and the §0.7 topic-fit decision.

**Decision.** v3.1 is final. Day-1 CPU kills (K0, K1, K2, K3a) can start now.

**Next step.** The decisions below.

---

## What died and what each death taught

| What died | Killed by | Lesson carried forward |
|---|---|---|
| DECONV as a technique | Critic KILL; blind graph deconvolution is prior art; no execution role (CR-10, RT-35) | A forensic step is a control, not a paper |
| CHAIN as a standalone paper | AFLGopher and GRAPHIA already have per-site softmaxes (RT-22, CR-12) | Search the mechanism ("per-call-site distribution"), not the vocabulary |
| HEAP as a headline | Jelly recovers ≈51% of missed edges without tests (RT-19) | "Not found in a light search" (IV-2) was wrong; the right terms were "approximate interpretation" and "heap snapshot" |
| LATENT / Study B | Unusable at inference (no mid-layer states in vLLM); 5–7 pd, ≈6.9 NH; "a second paper" (M§6) | Scope to one claim per 3,000 words |
| HYTRACE LoRA | ≈26 NH per seed at the measured 82 tok/s (CR §3) | Zero-shot first; train only after a kill passes |
| Frequency-based effect labels (v1) | L15/L16: issues report rare, buggy, polarity-inverted behaviour | Use possibility labels and signed (±) evidence |
| PPR as the C1 control (v1) | L35: the control never saw the imagined path | Every control must see the same LLM output |
| Execution-validated EVAL headline (v1) | M§1: 40–60% yield, CPU undercounted 7–30× | Static gold for the headline; execution only for labels and a 60-task subset |
| Base-rate likelihood ratio (v2) and local median (v3) | N1, then E-1 / Proposition 0: per-step constants cancel | Prove ranking effects algebraically before claiming them |
| Insert-state decoder (v2) | N8: recreates the stutter | Delete states plus delete credit at the culprit end (Proposition 4) |
| X-BAG⊗PPR log-product (v2) | N14: changes the algebra and the sequence at once | A control must be the same model at a parameter value (Proposition 3a) |
| Four-condition C-BEH pass rule (v2) | N22: joint power ≈0.3–0.5 | One control, one metric, three-way verdict |
| μ′ outside β and the R2 identity map (v3) | E-2, E-5: the control or treatment leaks | Audit what each arm can read, not only what it computes |
| R1 renaming, JS, E4B, CoSIL run | Cost against 27 days | Generality via ≥25 repos and a second generator (Qwen) |

These deaths repeat lessons from the earlier reports. *PEFT RL for directed graph reasoning* ended with an analysis instrument at ≈0.14–0.47 power and an under-counted budget. BMM hit the same two failure modes (N22, N27) and fixed them earlier. *LLM graph reasoning for code* showed that resolve is 27% when the diagnosis names the right function and 28% when it names the wrong one. BMM therefore claims no resolve gain anywhere.

---

## Final specification (v3.1)

### Component diagram

```
OFFLINE (CPU; CAL-train repos, Py>=3.12, sys.monitoring)        K4 by Oct 23; fail => AST q only
  traced suites --> possibility labels (effect seen in >=2 passing tests) --> g_phi (Eq. 7)

PER SNAPSHOT (CPU; no tests run)
  AST resolver: imports, aliases, static MRO, self/cls/super, local imports, literal registries
  U HEAP where an image/pure-Python install imports: unwrap -> W (wraps) edges, registries, re-exports
  call sites S(u): repo-resolved, callee-deduplicated; site-local null state; per-registry null state
  q_v: receiver-typed AST rules (+) noisy-OR g_phi; per-class exception bits; <=2-hop propagation (Eq. 6)

PER ISSUE (GPU: vLLM, Gemma-4-31B W4A16, thinking off)
  G0: repo tree -> top-10 files
  G1: skeleton(G0) -> k=10 paths of (symbol, +effects, -effects @ issue span)
  span check (CPU): every effect code must cite a verbatim issue substring
  B-RECALL: issue + repo name -> top-10 (memorisation covariate)

DECODER (CPU): Eqs. 1-5, forward-backward over {Match, Insert, Delete} x V,
  orientation = argmax P(o), readout with delete credit at T + missing-call credit to callers
  --> full-repo function ranking
```

### Key equations (v3.1 §0.6.2)

Observations, one sampled path: o_t = (n_t, E⁺_t, E⁻_t), with E^± ⊆ B. Hidden state z_t ∈ {M_v, I_v, D_v}: match, insert (noise, position stays) or delete (noise, position advances).

```
K(u,v) = (1−ε)·(1/|S(u)|) Σ_{s∈S(u)} p(v|s)  +  ε·Σ_{w:(u,w)∈W} K⁰(w,v)/|W(u)|           (1)
R_t(v) = 0.8·Unif(N₂(Top20_t))(v) + 0.2·π(v)                                             (2)
P(M_v | u) = (1−η₀−δ−ρ)·K(u,v)·(1−θ) + ρ·R_t(v)·(1−θ);  P(D_v | u) = same with θ;  P(I_u | u) = η₀    (3)

ℓ_v(o_t) = τ·[sim(n_t,v) − c_t^sim] + ω·BODY(o_t,v) + β·[Φ_v(o_t) − c_t^Φ]              (4)

Φ_v(o) = Σ_{j∈E⁺} log((1−η)q_vj + η)                                    reported effect
       + Σ_{j∈E⁻} [ log(1 − q_vj + η)                                    noisy-OR negative finding
                   + log((p̄_j(v) + η)/(q_vj + η)) ]                      peer contrast (Chucky-style)
       − μ′·1[E⁺∪E⁻ ≠ ∅]·Σ_{j∉E⁺∪E⁻} q_vj                                omnivore penalty, inside β     (5)

q_vj = 1 − (1 − q^AST_vj)(1 − q^φ_vj)   for dual-source bits (KeyError, IndexError, TypeError, AttributeError)

min_φ Σ_r Σ_v w_v Σ_j BCE(y_vj, σ(φ_jᵀx_v)) + λ‖φ‖²,  y_vj = 1[effect j seen in ≥2 passing tests],  w_v = 1/max(1, log(1+calls_v))   (7)

score(v) = (1/k) Σ_samples [ Σ_t w_t γ_t(M_v) + w_T γ_T(D_v)
                             + Σ_{t:E⁻_t≠∅} w_t κ_pred Σ_u γ_t(M_u) K(v,u)/Σ_{v'} K(v',u) ],   w_t ∝ (t/T)²
metric m = log₂ min(rank, 101), rank = min over gold functions
```

Tuned (BMM only, 3⁴ grid, nested 5-fold CV on CAL-tune): β, τ, η₀, μ′. Fixed: δ = 0.05, ρ = 0.1, ε = 0.05, η = 0.05, κ_pred = 0.3. Controls inherit BMM's values except the one they remove.

### Propositions 0–4 in plain words

| Prop. | Plain statement | What checks it |
|---|---|---|
| **0** | Adding the same constant to every node's score at one step cannot change which node wins *at that step*. It only shifts match against insert/delete. So "subtract a local base rate" is a calibrator, not a contrast | Algebra (no experiment). Hence F5 |
| **1** | If the culprit and a name-similar sibling are structurally identical and tie on names and body text, every names/body/graph/PPR method ties them. With β > 0, the culprit wins **iff** its stated-effect advantage (reported + negative finding + peer) exceeds μ′ times its extra unstated-effect mass (Eq. 8) | **K3c** (CPU, Oct 17): share of (gold, top-5 sibling) pairs satisfying Eq. 8 at β = 1, μ′ = 0.1. Predicted ≥0.6; ≈0.5 means the channel is empty |
| **2** | Sequence decoding beats the bag only when a predecessor step is confidently decoded, branching b_t is low, and distractors are globally plausible but locally implausible (Eq. 9: b_t < (1−η₀−δ−ρ)·PPR(d)/(ρ·R_t(d)·PPR(v_t))). The gain grows with path length | C-DEC reported by T (1–2 / 3–4 / ≥5) and by branching terciles. P1a oracle paths |
| **3** | BMM reduces exactly to its baselines at parameter limits: ρ = 1 gives X-BAG, T = 1 gives a prior-weighted bag, β = 0 gives name-and-text map-matching (FSE'24 on call graphs), uninformative observations give PPR, and η₀ + δ → 1 gives BM25 | Reported tuned (ρ, η₀); RBO ≥0.9 with X-BAG or B-PPR flags a "reduced" run (descriptive) |
| **4** | When the last imagined step is hallucinated, delete credit spreads it over the children of the last confident node (≈1/b_u each), instead of over the whole graph as PPR does | P1a with last-step substitution on executed CAL-tune tasks |

### Existing idea + new use (v3.1 §0.5)

| # | Existing idea (closeness) | What it already does | BMM's change | Falsifying experiment |
|---|---|---|---|---|
| E1 | HMM map-matching, Newson–Krumm (2); multiscale Viterbi, TSP'18 | Decodes GPS points onto roads | Call graph; LLM-imagined steps; delete and skip states; culprit posterior | C-DEC, X-SHUF, X-DPR |
| E2 | FSE'24 Markov/DP step alignment (2.5 decoder); Euler (2); AstroBR (2) | Aligns written or LLM-*extracted* S2R steps onto GUI graphs for reproduction | LLM-*imagined* steps from issues with no S2R section; normalised likelihoods with behavioural terms; localisation | BMM vs B-ALIGN (same ℓ, including β) |
| E3 | EXCEPT, SBEST (1.5); CrashTracker (1.5); FaR-Loc (1.5, [2509.20552](https://arxiv.org/abs/2509.20552)) | Use exception type, stack trace or free-text failing behaviour from an *executed failure* | Traceback-free prose effects with polarity, span-verified; typed multi-effect alphabet; per-step emissions along a path | **C-BEH**, X-LEXEFF⁺, X-POOL; K3a/b/c |
| E4 | IPA-GNN, CodeFlow (2) | Predict whether and which error a whole program raises | Per-function cross-repo probabilities over intended-effect bits, calibrated as emissions; noisy-OR with AST rules | K4; X-ASTEFF |
| E5 | QMR-DT/Quickscore (2); possibilistic MBD (2) | Absent expected findings lower hypothesis posteriors | Source: "expected but did not see" in prose, on an imagined step; object: predicted function effects | X-NEG0 |
| E6 | Chucky (2); Engler (1.5) | Flags functions lacking a check their peers have, issue-agnostic | Conditioned only on the issue-stated missing effect | X-PEER0 |
| E7 | LocAgent prompt (1.5–2) | Free-text execution flow as chain of thought | Typed (symbol, ±effect @span) sequence consumed by exact inference | C-USE |
| E8 | RepoReasoner (2); SWE-Flux, TraceEval (1.5) | Test → predicted file set vs traced chain | Issue prose; function level; F2P reference; divergences typed by edge kind | M1 on EVAL-exec |
| E9 | Liblit Increase (2); IDF-IRBL (1.5) | Background-rate normalisation | **No new use claimed (F5)** | Proposition 0 |
| E10 | SherLog, Pensieve, Pathidea, CrashLocator, COCA (1.5–2) | Rebuild executed paths from real logs or stacks with exact anchors | Observations imagined from prose; soft anchors; exact probabilistic inference | B-ALIGN, X-NULL; B-CRASH on the traceback stratum |
| E11 | RoG, GCR (2) | LLM plans relation paths grounded on a KG | Program graph; soft grounding with delete/skip; behavioural plan elements | C-DEC; GCR-hard secondary |

HEAP (2–2.5: Jelly, HeapDL, PoTo) and per-site candidate sets (2: AFLGopher, Pereira) are infrastructure and are not claimed.

**Combination evidence (search-log IDs).** No Viterbi or map-matching on call graphs: BR3-1 (0 hits), RT-28, RT-32, BR3-21, BR3-29. No predict-then-align from issues: BR3-7, BR3-12, BR3-25, RT-27. No signed-effect localiser: BR3b-8 (4 queries, 0 relevant), BR3b-15, BR3b-17, BR3b-11 (OpenReview, no hit). No effect-signature retrieval: BR3-8, BR3-20, BR3b-4. Residual risk: the absence is at abstract level only. Semantic Scholar was rate-limited, and Google Scholar and ACM DL were not crawled. ICSE/FSE 2027 papers not yet on arXiv are invisible. QMR-DT, Liblit, Chucky and Engler were read only as snippets.

### Must not claim

| Item | Prior that owns it |
|---|---|
| LLM-imagined execution flow | LocAgent; CoSIL, GraphLocator, SWE-Debate |
| Symptom → path reconstruction | SherLog, Pensieve, Pathidea, CrashLocator, 2506.19045, COCA |
| NL steps mapped onto an execution graph; Markov/DP alignment with missing-step inference | Euler; Zhang et al. FSE'24; AstroBR |
| HMM, map-matching, Bayesian FL | Newson–Krumm; Dietz NeurIPS'09; PPDG; SmartFL; Prosecutor |
| Exception- or effect-driven localisation | EXCEPT, SBEST, CrashTracker, ExLocator, EXPECT |
| Execution-trained exception prediction | IPA-GNN, CodeFlow, TRACED |
| Repo-level predicted vs executed path measurement | RepoReasoner, SWE-Flux, TraceEval |
| Search by behaviour | Podgurski & Pierce, Satsy, Jiang SCAM'18, FaR-Loc |
| "Execution-free" (say **"no tests at inference"**; HEAP runs module top-level code) | LLMAO, SHERLOC, Agentless |
| Execution-verified call-edge data | TraceEval |
| Observed/expected behaviour structure for localisation | Chaparro ICSME'17/EMSE'19; BugScribe; Desc2Fix; [2607.07593](https://arxiv.org/abs/2607.07593) |
| Negative findings as an inference idea | QMR-DT; possibilistic MBD; software MBD |
| Base-rate or likelihood-ratio scoring of code | Liblit PLDI'05; Sisman & Kak; IDF-IRBL |
| Peer contrast | Chucky, Engler |
| "First to combine report-derived paths with IR" | Pathidea (log- and stack-anchored BFS) |
| "First" for polarity, likelihood terms, peer contrast or decoding individually | Only the conjunction may be claimed, and only if C-BEH passes |

### Baselines and controls

The baselines are as follows. **B-DIRECT\*** is the best of three prompt-only rankers chosen on CAL-tune (direct top-10, a LocAgent-style flow-then-list, and G1's final steps), with its tail filled by X-BAG. B-RERANK (Gemma listwise rerank) is ladder rung 1. B-BM25 and B-CRE (CodeRankEmbed) are retrieval baselines. B-PPR runs PPR from π with no imagined path. **B-ALIGN** is FSE'24-style DP alignment plus a greedy Euler variant, given the same ℓ including β. B-RECALL is a memorisation probe. Published LocAgent, CoSIL, SweRank, Agentless and OrcaLoca numbers on matched Lite are cited as non-comparable. CERT and Z from the earlier reports are excluded, because they are RL credit signals with no function ranking.

The controls each change one thing. **X-BAG** (rank-1 PPR kernel, BMM's orientation) is the C-DEC control and the BER fallback. **X-β0** (all q terms off) is the C-BEH control. **Graph-free** is X-BAG with π in place of PPR_π. **X-DPR** is a degree-preserving rewire within each package, with p carried over (needs ≥10 repos with ≤10% edge overlap, K7). X-NEG0 / X-PEER0 test polarity. X-POOL pools effects across steps. X-ASTEFF / X-LEXEFF⁺ separate learned and static behaviour. X-SHUF and X-NULL (donor paths) test order and provide a matched null. GCR-hard sets ρ = 0 with no I/D states.

### Primaries, honest power and decision rule

Metric m = log₂ min(rank, 101), unfused. Inference uses a wild cluster bootstrap over repos, Holm over three primaries, one control per claim and a TOST margin of ±0.1. The C-BEH stratum at a 35% effect share is **210 tasks** over ~20 repos.

| Primary | Contrast | n → n_eff (ICC 0.05 / 0.15) | Predicted Δm (GUESSED) | Power SD 0.6 | SD 0.78 | SD 0.9 |
|---|---|---|---|---|---|---|
| **C-BEH** | (BMM + BODY) − X-β0, under R2, effect-stated stratum | 210 → 142 / 87 | −0.20 | 0.94 / 0.76 | 0.75 / 0.50 | 0.60 / 0.37 |
| **C-DEC** | BMM − X-BAG | 600 → 279 / 135 | −0.20 | (SD 0.7) 0.99 / 0.82 | — | — |
| C-USE | BMM − B-DIRECT\* | 600 → 279 / 135 | −0.15 | (SD 1.2) 0.38 / 0.17 | — | — |

**At realistic values (210 tasks, SD 0.78–0.9, ICC 0.15), C-BEH power is ≈0.37–0.50.** The verifier re-ran this arithmetic. Joint C-BEH ∧ C-DEC power is ≈0.93 → 0.62 at SD 0.6 and ≈0.74 → 0.41 at SD 0.78. C-USE is openly underpowered.

| Verdict on C-BEH | Rule |
|---|---|
| Supported | Holm-adjusted two-sided CI for Δ lies entirely below 0 |
| Falsified (equivalence) | 90% CI inside (−0.1, +0.1), or CI entirely above 0 (harm) |
| Inconclusive | Anything else; reported as such, never called falsified |

Either sign of C-BEH is informative. A positive result shows that issue-stated behaviour is a localisation channel that survives code-side renaming. An equivalence-null bounds behaviour-aware localisers (FaR-Loc, BLAgent) and the finding that expected behaviour helps agents ([2607.07593](https://arxiv.org/abs/2607.07593)) to a lexical mechanism. The memorisation-free sub-stratum (B-RECALL miss, or gold named in the issue) is a pre-registered secondary estimate, not a primary.

### Kill table (K3c is the first demonstration)

| # | By | Test | Pass | Fail consequence |
|---|---|---|---|---|
| K0 | Oct 11 | EVAL pool, static gold, dedup | ≥600 tasks, ≥25 repos | <400: restate MDE |
| K2 | Oct 11 | CAL-tune availability | 100 / 40 executable | 60 / 25 |
| K1 | Oct 13 | Image prefix extraction ×3 (**Shrey's token**) | ≥2/3 F2P fail→pass | EVAL-exec ≈30 pip tasks |
| K3a | Oct 14 | Regex effect share (validated against [2607.09553](https://arxiv.org/abs/2607.09553) labels if available) | ≥35% | <20%: C-BEH secondary → R6 |
| **K3c** | **Oct 17** | **Sibling-separation pilot (Prop. 1), AST q + regex effects, ≈2 kh, ≈0.25 pd** | **≥0.60** | **≤0.55: C-BEH demoted before any GPU; headline R3/R6** |
| K8 | Oct 20 | vLLM batched decode | ≥250 tok/s | k = 5, stratum rule ≥2 of 5 |
| K5 | Oct 22 | True-path edge recall (P1b) | ≥0.6 | <0.4: C-DEC expected weak |
| K6 | Oct 22 | P1a gate + oracle mechanism pilot (Props 2, 4) | hit@5 ≥0.80; Δm < 0 | Debug decoder on CAL-tune |
| K7 | Oct 22 | DPR feasibility | ≥10 repos | Descriptive only |
| K4 | Oct 23 | g_φ vs AST rules, leave-repo-out macro-AP | ≥+0.03, CI > 0 | AST q; drop "execution-trained" (harmless) |
| K9 | Oct 24 | CAL-tune GPU run + hallucination profile | BMM not worse than X-BAG and B-DIRECT\* by >0.3 | Headline pre-switched to R3/R4 |
| K3b+K10 | Oct 25 | 50-task EVAL pilot | Share ≥35%; SD ≤0.78 | <150 tasks: C-BEH secondary (R6) |
| K11 | Nov 3 | EVAL primaries | §4.2 rule | Rows R2–R9 |

### Collapse table

| Row | Outcome | What BMM is | Framing |
|---|---|---|---|
| R1 | C-BEH ✓, C-DEC ✓, C-USE ✓ | A technique at closeness 2 | Technique + M1 |
| R2 | C-BEH ✓, C-DEC ✗ | Behaviour helps; sequence does not | BER (behavioural evidence reranking), with its own C-BEH |
| R3 | C-DEC ✓, C-BEH falsified | FSE'24-style map-matching on LLM names | Insight paper (negative §0.3) + M1 |
| R4 | C-DEC ✗, C-BEH ✗, ≈ B-DIRECT\* | **Collapses into "BM25 + LLM rerank"** | Measurement paper + negative result |
| R5 | C-BEH ✓, C-DEC ✓, C-USE ✗ | Beats its ablations, not direct prompting | Robustness framing (R2, memorisation-free, gold-not-in-G0) |
| R6 | K3a/K3b/K3c fail or are ambiguous | C-BEH empty or underpowered | C-BEH as estimate; headline from C-DEC + M1, decided before EVAL decoding |
| R7 | C-BEH ✓ only via PEER | Polarity = peer contrast | Chucky-style contrast on issue effects |
| R8 | C-BEH ✓ only on repo-exception steps | Possible residual-name effect | Builtin-only result is the claim; if null, treat as R3 |
| R9 | C-BEH inconclusive | Underpowered | Estimate with CI; framed as R6 |

R5 is the most likely outcome **only if the predicted effects are real** (P ≈ 0.54–0.59 conditional on that). Unconditionally, R3, R4 or R6 may be more likely.

### Budget and schedule risk

| Resource | Optimistic | Central | Pessimistic |
|---|---|---|---|
| GPU (NH = Kaggle 4×L4 session-hour) | 6.9 | **9.95 (≈10)** | 15.65 (19.4 with all options) |
| CPU (4-vCPU kernel-hours) | 112 | **223** | 500 |
| Person-days before Nov 5 (5 h each) | 14.35 | **19.0** | 27.5 |
| Writing, Nov 5–12 | 1.5 | 2.0 | 2.5 |

The descoping floor is **≈14.5 pd and ≈6 NH**. The internal arithmetic gives 14.25–14.75 depending on which ladder savings are counted. GPU is not the binding constraint: ≈10 NH is about 9% of the ≈120 quota-hours left. **People-time is.** 19 pd central is **≈70% of the 27 days to Nov 5**, shared with the main track (Dec 2). The pessimistic 27.5 pd does not fit at all. Only K0–K3a and K3c fit by Oct 17; K4–K7 land Oct 21–23. The verifier ranks schedule as the most likely failure mode, independent of the science. The never-cut core is: static EVAL ≥400, G0/G1/R2, B-RECALL, B-DIRECT\* and BM25, X-BAG and X-β0, AST effects with span verification, the tracer with executed CAL-tune tasks, and the truncated-metric statistics. The tracer is the same coverage tracer specified for IFG (*LLM graph reasoning for code*) and the null-model dial, so building it once serves all three lines.

---

## What is verified and what is guessed

| Claim | Status | Source |
|---|---|---|
| Scout anomalies (49% edge recall; 0.99 → 0.19; 1–2% data flow; NDCG 71.7 → 20.3; MRR 0.29 → 0.51–0.57; AUROC 0.986 vs 0.869) | VERIFIED (primary text read by scouts) | Linked inline above |
| Jelly lifts recall 75.9% → 88.1% at −1.5% precision with no tests | VERIFIED (PDF, RT-19) | Red team |
| FSE'24 is a Markov model with heuristic additive scores and DP top-k, not an HMM | VERIFIED (PDF, BR3b-12) | [PDF](https://par.nsf.gov/servlets/purl/10544604) |
| Pathidea reconstructs from logs and stack traces only | VERIFIED (PDF, BR3b-1c) | [PDF](https://petertsehsun.github.io/papers/PathideaTSE2021.pdf) |
| LocAgent's prompt already reconstructs execution flow | VERIFIED (PDF prompt text, BR3) | [2503.09089](https://arxiv.org/abs/2503.09089) |
| RepoReasoner precision ≈0.80, recall ≈0.31 | VERIFIED (abstract + HTML) | [2607.25996](https://arxiv.org/abs/2607.25996) |
| Combination absence (C1–C7 together) | VERIFIED at abstract level only | BR3-1, BR3b-8, BR3b-11, BR3b-15, BR3b-17, RT-28 |
| QMR-DT, Liblit, Chucky, Engler mechanisms | SNIPPET / recall | BR3b-13, -14, -18 |
| Proposition 0 (centring is ranking-neutral) | VERIFIED (algebra) | verify_v3 E-1 |
| Propositions 1–4 | Proofs and sketches on paper; conditions untested | v3.1 §0.6.3 |
| Power table arithmetic | VERIFIED (re-run to ±0.01) | verify_v3 §3.1 |
| Budget column sums; 19/27 = 70%; quota 30 h/week | VERIFIED | verify_v3 §3.2–3.4; HQ |
| Δm = −0.20, SD 0.6, ICC 0.05–0.15 | GUESSED | K10 measures SD |
| Effect-stated share ≥35% (breaker's regex prior: 15–22%) | GUESSED | K3a, K3b |
| Sibling separation ≥0.6 | GUESSED | K3c |
| P1a hit@5 ≥0.80; edge recall ≥0.6 | GUESSED | K6, K5 |
| vLLM ≥250 tok/s; 19 pd; 223 kh | GUESSED | K8; Oct 17 review |
| Closeness ratings; topic fit; "R5 most likely if effects are real" | INFERRED | RT, BR3, BR3b, verifier |

---

## Decision needed from Shrey

Nothing has been run. Every number in the power, budget and effect columns is a guess until K3c and K10 report.

| # | Question | Why it matters |
|---|---|---|
| 0 | **Run the K3c CPU pilot as the first demonstration?** About 2 kernel-hours, $0, ≈0.25 pd beyond the K0/K3a builder work, by Oct 17 | It is the cheapest test of whether signed evidence separates gold from name-similar siblings at all. ≤0.55 demotes C-BEH before any GPU is spent |
| 1 | **Docker Hub token.** Create a read-only access token and add it to Kaggle Secrets **yourself**. I will not handle it | Without it, EVAL-exec shrinks to ≈30 pip tasks and HEAP covers pure-Python repos only |
| 2 | **19 pd central vs the main track.** That is ≈70% of the days to Nov 5. Accept it, or start at the ≈14.5 pd floor? | Schedule is the top residual risk |
| 3 | **GPU windows.** ≈10 NH central (15.7 pessimistic), mostly Oct 20–Nov 1, over about 8 manual L4 starts. Which days avoid LoRA training and 12-h evals? | Sessions are manual |
| 4 | **Accept R5 as the likely headline** (conditional on real effects), with R3/R4 as analysis papers? | Pre-registering the framing now prevents a rewrite after the data |
| 5 | **Accept topic-fit decision (b):** a localisation/graph-inference framing, with q_v as a secondary artifact only? | Strict fit is Graph Reasoning; Code Comprehension only in a generous reading |
| 6 | **Public Kaggle dataset release** of generations, graphs and effect tables? | Verifiability (rubric) without others rerunning environments |

## Conclusion

The process did what it was built to do. It found no unoccupied mechanism, only an unoccupied conjunction, and it then proved on paper that two of that conjunction's apparent ingredients (base-rate likelihood ratios and local centring) do nothing to rankings. What remains is narrow and testable. The research question is whether the behaviour an issue states, signed and attached to the right step, separates a culprit from its name-similar siblings once names stop helping. Propositions 1 and 2 turn the success conditions into numbers K3c and C-DEC can measure, so a negative result will say *why* the channel is empty, not only that it is. The real constraint is calendar, not compute. A $0 pilot on Oct 17 decides whether 19 person-days are worth spending. Until it runs, BMM meets two of the user's four novelty criteria and does not count as new.
