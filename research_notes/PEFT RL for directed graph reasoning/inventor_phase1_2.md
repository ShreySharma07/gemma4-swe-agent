# Inventor pass, Phases 1–2: PEFT + RL-style training signals for reasoning over large directed graphs (8 Oct 2026)

Role: inventor. No critic has been applied yet, so ideas are deliberately bold. Novelty is **not** judged here; the closest prior work I happened to see is named so the critic can attack it.

Target instance:
- Gemma-4-31B, served W4A16 (QAT, sym int4, group 32). LoRA only, rank ≤128, no DoRA.
- Training on Kaggle 4×L4: FSDP-QLoRA, ~20 GB/GPU at 16k tokens, ~400–650 tok/s (INFERRED, `docs/training_feasibility_4xl4.md`).
- The loop is generate-with-vLLM, then train offline, for a few iterations.
- Deployed agent: declarative YAML with fixed tools (`run_command`, `read_file`, `edit_file`, `write_file`, `submit_patch`, plus optional `get_code_neighbors` / `search_similar_code`). Only LoRA weights and prompts can change.
- Budget this week: **1.5 GPU-h**, so every first kill test is CPU-first.

Labels:
- **VERIFIED**: I read it in the primary source this session, or computed it from repo files.
- **SNIPPET**: title or abstract from a search result only.
- **INFERRED**: my reasoning, or carried over from the project notes without re-reading.
- **GUESSED**: a number I made up to size things; the kill test measures it.

Baselines to beat, never re-proposed here:
- CERT (interventional step-credit audit);
- Z (zero-information actions);
- IFG / EWEC (critic's top pick in `LLM graph reasoning for code/critic_judgement.md`: executed-graph verifier for fix structure and chains).

Where an idea touches IFG/EWEC, I say exactly what differs.

---

## 0. Anomaly re-check and two local facts (this session)

| # | Anomaly | Status after my re-read | What the re-read adds |
|---|---|---|---|
| 1 | RepoReasoner call-chain prediction: GPT-4.1-Mini precision 0.814, recall 0.318 (50% signal ratio) | **VERIFIED** (FSE'26, Table 2) | **The decisive detail:** in the *Oracle* setting the pool contains only signal files (no distractors), yet GPT-4.1-Mini recall is still 0.355 (precision 0.912). DeepSeek-R1 Oracle recall is 0.357, Qwen3-235B 0.387. Noise is not the cause; the models simply emit short lists. Ground truth = files hit by a dynamic pytest trace, so the labels include every incidentally executed file. The task is file-level. |
| 2 | GRA: graph-free control within +0.3 to +1.9 pp | VERIFIED (inherited, `general_graph_reasoning.md`) | GRA's budget knee is at B≈30 calls. Below the knee, selective access dominates. |
| 3 | RepoNavigator: adding 3 structural tools under outcome-only RL cuts IoU 24.28 → 13.71 | VERIFIED (inherited, `code_graphs_swe.md`) | The added tools (GetClass/GetFunc/GetStruc) are largely *redundant views* of what `jump` returns. That matters for mechanism M3b. |
| 4 | CNY: degree-preserving rewire drops walk accuracy 76.6 → 73.0 | **VERIFIED** (2608.29588 §5.3) | (i) 73.0 equals the "preview+direct, no walking" control (73.4). The rewire removes *exactly* the walk's gain and nothing else. (ii) Walks per node rise 1.24 → 2.01: the model notices that steps return less useful evidence and compensates. (iii) The model picks the most embedding-similar preview only 22% of the time. **Reconciled reading (INFERRED):** structure is used, but only for *selection*, and selection is worth ~3.5 pts on WikiCS. "Barely uses structure" is true of the *accuracy budget*. "Uses structure" is true of the *mechanism*. |
| 5 | SOG-R1: RL-only 63.0 vs 86.1 | **VERIFIED, with a correction** (2607.18481 Table 3, CWQ) | The two numbers are RL-only (SOG-R1-Zero) = 63.0 and **SFT-only** = 86.1. SFT+RL = 88.7. Base = 35.3. So RL-only does not "collapse": it gains **+27.7** over base and plateaus below SFT. The text and table disagree slightly (text: SFT-only 86.3, RL-only WebQSP 81.9 vs table 80.9). The RL uplift over SFT shrinks as SFT data grows: +7.8 EM at 2.7k questions vs +2.4 at 8.7k. On Qwen3-4B, RL adds more over a weaker SFT (+6.5 CWQ). SFT data = teacher traces scaffolded by gold SPARQL. |
| 6 | Incomplete graphs (2606.30247): hard grounding cannot separate true-unobserved from false-unsupported edges | ~VERIFIED (inherited) | Code is special. **Execution is a third evidence source** that resolves the ambiguity on the *covered* part of the graph, and only there. |

**Local fact F1 (VERIFIED arithmetic, regex-noisy gold labels).** Source: `results/runs/r3_v5_graph/behaviour_per_task.json` × `r2_v2_sed/per_task.jsonl`. In v5 the localiser's structured diagnosis has an `OTHER_LOCATIONS` field. By number of gold functions:

| Gold functions | Tasks | `OTHER_LOCATIONS: none` | Mean fraction of gold functions named | Resolved |
|---|---|---|---|---|
| 1 | 55 | 49 | 0.56 | 23 |
| 2 | 16 | 11 | 0.34 | 5 |
| ≥3 | 43 | **31 (72%)** | **0.25** | **2** |

This is RepoReasoner's high-precision / low-recall pattern reproduced in our own agent, at the decision that matters: which functions to commit to editing.

**Local fact F2 (VERIFIED arithmetic, `r2_v2_sed/per_task.jsonl`).** `repro_before_edit` is true in 17/129 tasks, and only 2 of those 17 resolved, against 24/112 without reproduction. Naive reproduction does not currently pay off; any "tests as sensors" idea has to beat this. Also from findings: of 47 wrong-fix-never-tested tasks, 9 ran any test and 6 reproduced (VERIFIED, `docs/findings.md`).

---

## Shared definitions used by every idea

**Graphs.**
- `G_prov`: the provided graph. Undirected in practice, one mixed `calls` label (HQ A2).
- `G_exec(t)`: the directed caller→callee graph executed by test *t*, recorded with a `sys.setprofile` tracer on CPU.
- `G_exec(F2P)`: the union over fail-to-pass tests on the buggy commit.
- `G_lex`: edges lexically visible in source text (callee name followed by `(` in the caller's body; A2 says 86–90% of edge pairs have recoverable direction this way).

**Mandatory controls in every evaluation.**
- **Graph-free control (GF):** the same pipeline with edges removed. For prompts, the node set only (names and signatures, or bodies without the cross-reference block). For agents, no graph tools. For signals, recompute with the node set only. Wherever an idea trains, a GF-trained arm is trained with the edge-dependent part of the signal replaced by its node-only analogue.
- **Degree-preserving rewire control (RW):**
  - Directed graphs use a double-edge swap (a→b, c→d) ⇒ (a→d, c→b), rejecting self-loops and multi-edges, with 10×|E| swaps. This preserves every in- and out-degree.
  - Undirected `G_prov` uses the standard double-edge swap.
  - RW is applied to whatever graph the signal or evaluation consumes. A null distribution comes from two independent rewires (RW1 vs RW2).

**Offline loop.** Iteration *k*:
1. vLLM (wheelhouse 0.19.1, W4A16, TP=4, multi-LoRA) generates with LoRA_k. Plain temperature sampling, **no top-k / min-p**, which keeps importance ratios valid (Nebius lesson, VERIFIED in `rl_and_offline_substitutes.md`). Store per-token behaviour log-probs.
2. A CPU sandbox executes tests and computes the signals.
3. FSDP-QLoRA trains with the offline clipped surrogate

   `L = −Σ_t∈action tokens min(r_t Ã_t, clip(r_t, 1−ε, 1+ε) Ã_t)`, with `r_t = π_θ/π_gen` and ε=0.2.

   At step 0, r=1, so this reduces to advantage-weighted SFT. One or two epochs per iteration.
4. Export PEFT (`bias=none`, no `modules_to_save`). Two or three iterations in total.

**Online drop-in.** With online GRPO available, `Ã_t` is used unchanged as the per-token advantage, with the group being the K samples of one prompt.

**PEFT default.**
- LoRA r=32 (≤128), α=64, dropout 0.05, all 410 linear modules.
- lr 1e-4 for pure-SFT phases, 2e-5 for advantage-weighted phases.
- Base for training: W4A16 dequantised to **HQQ 4-bit / group 32**, the exact grid. NF4 only if the probe shows ≥97% token agreement.
- Chunked CE with the 30.0 softcap. Max sequence length 16k.

**Why these keep the W4A16 model near SOTA (shared argument, INFERRED).**
1. Adapters are trained against the served int4 grid, so there is no train–serve mismatch.
2. All behaviour data is **on-policy** from the W4A16 model itself. RFT-style updates on own samples shift the distribution less than SFT on teacher traces; the SWE-Gym evidence that naive on-policy RFT can hurt is the counter-risk, so every idea keeps a KL/replay guard.
3. Auxiliary items are at most 30% of tokens, mixed with resolved-trajectory replay.
4. Several signals target failure modes that quantisation *amplifies*. INT4 Gemma-4-31B doubles tool-name hallucination while the score stays flat (Flat Score 2607.27275, VERIFIED in `code_graphs_swe.md`), and under-commitment (F1) is plausibly amplified the same way. GUESSED, testable by also running the bf16 QAT-unquantised checkpoint.
5. Report the gap: (bf16 QAT + LoRA) − (W4A16 + LoRA).

---

# Phase 1

## (a) Three mechanisms per anomaly, each paired with a signal that works only if the mechanism is real (18)

| Anomaly | # | Mechanism | Signal that works only if this mechanism is real | Observation that discriminates it from its siblings |
|---|---|---|---|---|
| 1 RepoReasoner precision ≫ recall | M1a | **Emission-termination prior.** Set-valued answers are closed early by a list-length prior (brevity from instruction tuning; "absence = false" habits). Perception is fine. | **Atomised set emission:** force one calibrated decision per candidate (no list to close), scored by a proper log score against trace labels. If perception is the bottleneck, atomising changes nothing. | Oracle recall ≈0.36 with zero distractors (VERIFIED) already favours M1a. Decisive test: zero-shot recall under list vs atomised format. |
| | M1b | **Relevance ≠ execution.** Models report the *task-relevant* subgraph; traces include incidental hubs (utils, config, `__init__`, logging). "Low recall" is partly a label-definition artefact. | **Test-IDF-weighted labels:** weight each positive by `idf(v) = log(|T| / |{t : v ∈ E(t)}|)`. Training on IDF-weighted reward fixes recall on specific nodes only if missed nodes are mostly hubs. | Missed nodes are enriched in low-IDF (high test-coverage) nodes. |
| | M1c | **Dark-edge blindness.** Missed nodes are reached through non-lexical edges: dispatch through base classes, decorators, `super()`, properties, `__getattr__`, fixtures, registries. Models see lexical `name(` edges only. | **Dark-edge ranking reward:** for caller *u*, rank candidate callees; reward = reciprocal rank of trace-witnessed callees with no lexical cue in *u*. Helps only if dark edges explain the misses. | Recall split by edge type (lexical vs dark) at equal depth. |
| 2 GRA graph-free ≈ graph | M2a | **Lexical entailment of topology.** In text-rich graphs (and code), node text already names the neighbours, so edges are redundant with text except for the non-lexical remainder. | **Redundancy-weighted training:** up-weight items whose answer path contains ≥1 non-lexical edge (`G_exec \ G_lex`). The gain over uniform weighting appears only if redundancy is the reason structure looks useless. | Graph-vs-graph-free gap grows with the fraction of non-lexical edges on the gold path. |
| | M2b | **Selection, not relation inference.** The bottleneck is *what to read* under a budget. Any selective interface works; topology only orders reads. | **Value-of-information per token:** credit each read by new witnessed-relevant nodes per observation token. Pays off only below the budget knee. | Effect is present below the knee (our ~5 min/task) and absent above it. |
| | M2c | **Composition ceiling.** Models compose ≤1 hop and guess the rest, so graph and graph-free agents are capped at the same depth. | **Depth-inverse-propensity reward:** weight each recovered node by `1/recall_base(depth)`, with depth = executed hops from the entry. Only works if depth is the binding factor. | Zero-shot recall falls steeply with executed depth, even for lexical edges. |
| 3 RepoNavigator: more tools hurt under RL | M3a | **Credit dilution.** A larger action space makes outcome-only GRPO noisier, with fewer successful rollouts per group. | **Prefix-stratified baselines:** compute advantages within groups sharing the same tool-choice prefix (offline branching from shared prefixes). | Drop shrinks with more rollouts per group or an SFT warm start. |
| | M3b | **Alias splitting.** The new tools are observation-equivalent aliases of `jump`, so probability and credit split across syntaxes for the same information act. | **Alias-marginal likelihood:** `−log Σ_{a′∈C(a)} π(a′)` over executable, observation-equivalent aliases. | A renamed *duplicate* of `jump` would hurt as much as GetClass; a non-redundant tool would not. |
| | M3c | **Observation cost.** Structural tools return long outputs, consuming context and turns, and success decays multiplicatively with call count. | **Token-cost-aware reward** (information per token, as M2b). | Equalising output lengths removes the drop. |
| 4 CNY rewire costs only 3.6 pts | M4a | **Structure is a candidate generator only.** It decides *which* neighbour to read; the label is decided from text. | **Structure-ablated self-teacher credit:** credit = teacher log-ratio with the true vs the rewired privileged graph. Non-zero only on selection actions if M4a holds. | Rewire loss = walk gain (73.0 ≈ 73.4, VERIFIED) supports M4a. Credit should concentrate on selection tokens. |
| | M4b | **Low structural mutual information in the data.** In natural tasks, given text, topology carries little label information, so models never learn to propagate along edges. | **Executable rewire counterfactuals:** create items where the answer is carried *only* by topology (degree-preserving call-target swaps executed against tests). | Models fail on structure-necessary items while succeeding on natural items of equal size. |
| | M4c | **Robustness by compensation.** Models detect bad structure and spend more steps (walks 1.24 → 2.01, VERIFIED), so accuracy understates structure use. | **Cost-sensitive outcome reward** (SOG-R1's multiplicative turn factor). Shows structure use in cost, not accuracy. | Under RW, accuracy barely moves while steps rise. |
| 5 SOG-R1: SFT warm start needed | M5a | **Support / exploration.** RL sharpens only behaviours with non-negligible base probability. | **Privileged-scaffold generation:** generate with the executed failing subgraph in the prompt, strip it, and train with importance correction. | RL-only gains concentrate on items with base pass@k > 0. |
| | M5b | **Protocol, not content.** SFT mostly teaches the interaction protocol (probe-then-filter, batching, paging, committing). Graph knowledge is secondary. | **Graph-compiled synthetic repos:** warm start on procedurally compiled repos that share protocol but no content with eval repos. | Warm start on synthetic data recovers most of the real-data gain. |
| | M5c | **Schema / content transfer.** The 235B teacher injects schema priors (which relations exist): memorisation that transfers within Freebase only. | **Repo-held-out SFT** with a per-repo content ablation. | SFT gain shrinks on unseen schemas or repos. Relevant: our hidden repos are private (HQ). |
| 6 Incomplete graphs | M6a | **Absence-as-falsity training bias.** Hard-grounded rewards teach "not in graph ⇒ false", producing conservatism. This links to anomaly 1. | **Tri-state execution grounding:** a claim is supported (static or trace), contradicted (covered by an executed test yet absent), or unobserved (uncovered); reward +1 / −λ / 0. | Hard-grounded RFT lowers recall on true-unobserved edges; tri-state does not. |
| | M6b | **Models already soft-ground** through parametric priors. Deletions are tolerated; spurious additions hurt (ARISE, INFERRED). | **Asymmetric graph augmentation:** train with deleted-edge views, never added-edge views. | Degradation curves: random deletion shallow, random addition steep. |
| | M6c | **No confidence channel.** Without a way to express edge confidence, models collapse to one operating point (conservative). | **Proper-scoring edge emission** (log or Brier) against execution witnesses, scored only on covered edges. | Recall at fixed precision improves when confidence can be expressed. |

## (b) Eight unstated assumptions, inverted

| Assumption | Inversion | Signal it yields |
|---|---|---|
| 1. The graph should be given to the model | The model **builds** the graph it needs (callers via grep, traces via `run_command`). The provided graph is optional and may be absent on hidden repos. | Tool-surface-invariant training (alias marginal); call-set prediction trained on traces. |
| 2. Edges are the unit of reasoning | The unit is the **intervention** (does changing this edge change behaviour?) and the **set** (which nodes jointly matter). | Executable rewire counterfactuals; atomised set emission. |
| 3. Reward the answer | Reward **forecasts** (which tests will flip after my edit) and **information** (how much an experiment told me about the fix). | Impact forecasting; spectrum information gain. |
| 4. More structure helps | Structure should be **subtracted**: credit only what topology adds over the same nodes without edges. Fewer redundant tools. | Structure-ablated self-teacher (true minus rewired teacher); alias marginal. |
| 5. The model should traverse forward | Traverse **backward** from the symptom (failing assertion → callers). The traceback *is* a backward path, and forward edges are the lexically easy ones. | Backward-hop (caller) prediction from traces. |
| 6. The graph is fixed during training | The graph is **edited by the agent and by us**. Edits rewire the program, and rewires are executable. | Executable rewire counterfactuals; edit-impact forecasting. |
| 7. Graphs are input, not output | The graph is an **output** (call sets, commit sets, impact sets), checked against execution. | Atomised set emission with trace labels. |
| 8. Tests are only for reward | Tests are **sensors** (coverage spectra = graph observations), **label generators** (free dense supervision on any repo) and **intervention oracles** (rewires, partial patches). | Spectrum information gain; trace-labelled auxiliary data; rewire labels. |

## (c) Ideal observer, then the cheapest approximation

**Ideal observer (infinite compute).** For a task, maintain the exact posterior over the *fix hypergraph*: the set of functions that must change, their joint-necessity structure, and the semantic change. Also maintain the posterior over the program's dynamic call graph. Both are conditioned on everything the agent has observed.

The ideal per-action training signal is the **change in the log-posterior of the true fix** caused by the action, minus its cost:

`g_t = log p(F* | o_≤t) − log p(F* | o_<t) − λ·cost(a_t)`

Equivalently it is the optimal advantage `Q*(s_t, a_t) − V*(s_t)`, computed by exhaustive continuation rollouts from every state. With infinite compute we would:
1. enumerate candidate patches;
2. run every test under every partial patch, which gives exact necessity and sufficiency structure;
3. compute exact information gain for every possible experiment;
4. roll out K→∞ continuations per state.

**Cheapest approximations, in decreasing fidelity:**

1. **Privileged self-teacher (one prefill per view, no rollouts).** Replace "posterior given everything" with "the same model given the executed failing subgraph". The log-ratio `log π(a | s, G) − log π(a | s)` estimates pointwise information gain. Subtracting the rewired-graph teacher isolates what *edges* contribute → idea 10 / **SAST**.
2. **Explicit Bayesian posterior over candidates from execution spectra (CPU only).** SBFL on the agent's own experiments gives `p_t(v)`, and `g_t` is computed exactly on that surrogate posterior → idea 7 / **SENSOR**.
3. **Interventional labels.** Partial-patch runs define F* (as in IFG). Reuse the labels only; the signal differs.

The cheapest is (2). It is CPU-only on logged trajectories, provided the raw trajectories exist or ~30 fresh rollouts are generated.

---

## The 12 distilled ideas

Common to every row:
- **Ctrl** = both the GF and RW controls as defined above.
- "≤x GPU-h" = Gemma W4A16 inference on 4×L4. Prefill throughput is GUESSED at 3–6k tok/s with prefix caching and is measured in the first run.
- Ideas marked ★ go to Phase 2.

| # | Idea | Mechanism → | Signal (exact; offline computation; where tests enter) | Surprising prediction | Kill test (≤2 days, ≤5 GPU-h; CPU-first) |
|---|---|---|---|---|---|
| 1 ★ | **CASE: Commit-completeness via Atomised Set Emission** | M1a, M6c; inversion 7 | Per-candidate line `v: p`. Per-element log score `S_v = y_v log p_v + (1−y_v) log(1−p_v)`. Element advantage = `S_v` minus the mean over K samples, applied to that line's tokens only. Labels y from traces (call sets) and gold/necessity (commit sets). Tests enter as label generators and as final resolve. | Zero-shot atomisation alone lifts executed-call-set recall from ~0.35–0.45 to ≥0.65 at ≤0.10 precision loss. RepoReasoner's "multi-hop weakness" is mostly emission. | CPU: 150 trace-labelled items from requests/rich/httpx. ≤0.4 GPU-h: list vs atomised. Ctrl: GF = names and signatures only; RW = labels from reachability on rewired `G_exec`. Kill if recall gain <0.15, or AUROC − AUROC_GF <0.05, or AUROC_true − AUROC_RW <0.05. |
| 2 | **Depth-IPW recall reward** | M2c | Per recovered node v at executed depth d: reward `1/r̂_base(d)` (clipped at 5). Base recall-by-depth estimated on CPU-labelled items. | Base recall drops ≥40 pts from depth 1 to depth ≥3 *even for lexical edges*. | Reuses idea 1's items and zero-shot outputs (no extra GPU). Stratify recall by depth × edge type. Kill if the slope is < 15 pts or vanishes once dark edges are controlled for. Ctrl as idea 1. |
| 3 | **Dark-edge ranking** | M1c, M2a | For caller u, rank candidate callees. Reward = Σ reciprocal rank of trace-witnessed callees not in `G_lex`. Lexical edges are scored too but at weight 0.25. | ≥50% of zero-shot misses are reached via dark edges, and the provided graph (built lexically?) misses the same ones. Training on 300 dark-edge items moves SWE localisation on dispatch-heavy repos. | CPU: classify missed nodes from idea 1's outputs by edge type, and measure the dark-edge share of `G_exec` vs `G_prov`. Kill if dark edges are <25% of misses. Ctrl: GF (node list) and RW (reachability on rewired `G_exec`). |
| 4 ★ | **ERC: Executable Rewire Counterfactuals** | M4b; inversions 2, 6, 8 | Degree-preserving swaps of *executed* call targets between signature-compatible call sites. The rewired program is run against the tests, and the model predicts per-test flips (atomised log score) plus the divergence frame. An agentic variant asks "find and revert the swap". Tests enter as the label generator and the fix verifier. | A zero-parameter reachability check on `G_exec` beats zero-shot Gemma at predicting flips. ERC training raises the agent's *sensitivity to rewiring* the provided graph (it starts using edges) and its multi-function resolve, with no loss when graph tools are absent. | CPU: 300 swaps on requests and rich. Yield ≥25% non-trivial. Reachability AUROC ≥0.80 vs text-only ≤ reachability −0.10. Reachability on rewired `G_exec` ≤0.65. ≤0.5 GPU-h: Gemma zero-shot with real / GF / RW context. |
| 5 | **Backward-hop asymmetry (caller prediction)** | Inversion 5; M1a/M2c | Items: "given function f on the failing path, list its executed callers up to the test". Atomised log score against `G_exec(F2P)` reversed. Mix forward items at 1:1. | Zero-shot forward (callee) recall exceeds backward (caller) recall by ≥25 pts. Gemma's missed multi-function gold nodes are mostly *callers* of the function it edited. | CPU: r2/r3 patches vs gold on `G_exec`; classify each missed gold function as caller / callee / sibling of the edited one. ≤0.2 GPU-h: asymmetry probe. Ctrl: GF and RW. Kill if the asymmetry is <10 pts and callers are <40% of misses. |
| 6 ★ | **ALIAS: observation-equivalent action marginal likelihood** | M3b; inversions 1, 4 | `L = −w_τ · log Σ_{a′∈C(a_t)} π_θ(a′ | s_t)`, where C = executable aliases (graph tool / grep / sed / `read_file` / `python -c ast`) whose observations carry ≥0.8 Jaccard of the same (file, symbol, line) facts. Tool-surface dropout. Aliases are executed in the sandbox. | Gemma's mass over aliases is diffuse (≥1 bit). After ALIAS, resolve differs by <1 pp with and without graph tools, and graph-tool use rises only where it is cheaper. | CPU: aliaser plus execution on 300 logged actions. ≤0.3 GPU-h: alias log-probs (prefill only). Kill if the alias rate is <20% or the top alias holds ≥0.9 of the mass. Ctrl: GF = no graph tools; RW = graph tool backed by rewired `G_prov` (must fall out of equivalence classes: ≥90%). |
| 7 ★ | **SENSOR: tests as sensors (spectrum information gain)** | Ideal observer (c2); inversion 8; M2b | Surrogate posterior `p_t(v) ∝ exp(κ·Ochiai_t(v))` from the spectra of the agent's own experiments up to t. Spectra come from replaying each experiment command offline under a coverage tracer. Credit `g_t = log p_t(F*) − log p_{t−1}(F*)` on the experiment's command tokens. Tests enter as sensors, as F* via gold/necessity, and as resolve. | Σg predicts resolve better than localiser function-hit, which does not predict it (27% vs 28%, VERIFIED critic L1). Training on g raises useful reproduction although F2 shows naive reproduction does not pay. | CPU 1.5 d: replay raw r2/r3 trajectories (retrievability GUESSED; else 30 fresh rollouts ≈0.6 GPU-h). Kill if AUROC(Σg) ≤ max(function-hit, GF Σg) + 0.05, or RW Σg is within 0.03 of true. |
| 8 ★ | **GCSR: graph-compiled synthetic repos (protocol/content factorisation)** | M5b; inversions 1, 6 | Sample a typed directed graph and compile it to a Python package (values flow along edges; dark edge types included). Inject 1–3 node bugs; tests at the roots. Agent tasks use the exact harness tools. RFT warm start on own successes. Reward = tests pass (+ CASE commit log score). | Synthetic-only warm start recovers ≥70% of real-SWE RFT gain at equal tokens. Topology *fidelity* (real-graph-copied vs degree-preserving rewired) does not matter; *depth* does. | CPU 1.5 d: generator v0 plus 60 tasks at depth 1/2/4. ≤0.6 GPU-h: zero-shot Gemma on 30. Kill if success does not fall with depth (Spearman > −0.1), or the GF version (issue names the bug node) ≈ normal. RW: compile from real requests graph vs its rewire (measurement). |
| 9 | **Privileged-scaffold generation, stripped training** | M5a | Generate with the `G_exec(F2P)` edge list (no patch) in the prompt, then train on the prompt without it. Token weight = clip(π(y | s) / π(y | s, G), 0.2, 1) × outcome advantage. | Scaffolding raises pass@4 on base-pass@4=0 tasks by ≥15 pts, and the stripped student keeps ≥⅓ of it. | ≤1 GPU-h: 20 base-unsolved dev tasks × 4 samples, with and without scaffold. Ctrl: GF scaffold (node list) and RW scaffold. Kill if scaffold − GF-scaffold < 5 pts. |
| 10 ★ | **SAST: Structure-Ablated Self-Teacher** | Ideal observer (c1); M4a | `c_t = log π0(y_t | P(G), s) − log π0(y_t | P(ρG), s)`. Advantage `Ã_t = Â_τ + β·clip(c_t)`. The frozen W4A16 base via vLLM `prompt_logprobs`. G = `G_exec(F2P)` edge list. Tests enter by building G and as outcome. | Edge credit is ≈0 on the first 1–2 navigation actions (lexical) and concentrates at ≥2 executed hops and on commit-set decisions. Training on c beats plain privileged OPSD on multi-function tasks. | ≤0.3 GPU-h: score 40 trajectories × 4 views. Kill if |c| is indistinguishable from the RW1-vs-RW2 null, or node-only δ explains ≥90% of δ_G variance. |
| 11 | **Impact forecasting (edit-delta prediction)** | Inversions 3, 6 | Before `submit_patch`, the prompt asks for `FORECAST:`, an atomised list of tests and callers whose behaviour changes. Reward = log score vs the actual test run after the patch; used as an RFT weight. | Forecast accuracy predicts resolve better than localisation, and training on it raises multi-function completion without any fix reward change. | CPU: on r2/r3 patches, compute the impact sets; check whether gold multi-function nodes lie in the first-edit impact closure on `G_exec` (critic H2 premise risk: CodePlan/SynFix/CG-Diff). Ctrl: GF and RW closure. Kill if <30% coverage. |
| 12 | **Tri-state execution grounding** | M6a, M6b | Claims in diagnoses (edges/nodes): +1 if supported by `G_prov ∪ G_exec`, −λ if covered by an executed failing test yet absent, 0 if uncovered. Abstention scores 0. Soft grounding with execution as the third evidence source. | Hard-grounded RFT reduces recall on true-unobserved edges by ≥10 pts relative to tri-state, i.e. it *trains* anomaly 1. | CPU: share of gold-path edges in `G_exec` that are absent from `G_prov` (expected ~10–14%, INFERRED from A2). Kill if <5%: there is nothing for soft grounding to protect. Ctrl: GF and RW. |

**Selection for Phase 2** (the six with the most surprising predictions that can still be tested cheaply): **1 CASE, 4 ERC, 6 ALIAS, 7 SENSOR, 8 GCSR, 10 SAST.**
- Ideas 2, 3 and 5 become diagnostics inside CASE's kill test (depth, edge type and direction stratification of the same outputs).
- Idea 12 becomes CASE's labelling rule.
- Idea 9 is SAST's generation-side twin and becomes an ablation of it.
- Idea 11 is parked: its premise overlaps the critic's killed H2.

---

# Phase 2: the refined six

Constraint check: none of the six changes inference or the harness.
- CASE adds one **prompt** field, which is allowed.
- All other signals are training-time only.
- Graph tools stay optional. Every method must work when graph files are absent on hidden repos (HQ: unknown).

## R1. CASE: Commit-completeness via Atomised Set Emission

**Mechanism.** Under-commitment on set-valued graph outputs (call sets, impact sets, *the set of functions to edit*) comes from an emission-termination prior: a list is closed early. It is not a perception failure.

Evidence:
- RepoReasoner Oracle recall is 0.355 even with zero distractors (VERIFIED).
- Our own v5 diagnoses: 72% `OTHER_LOCATIONS: none` and 25% gold coverage on ≥3-function tasks, where resolve is 2/43 (F1, VERIFIED).

Hard-grounded training habits ("absence = false") reinforce the prior (M6a).

**Data generation (CPU, Kaggle CPU kernel, permissive repos only).**
1. *Trace items.* For each repo with working pytest (start with requests, rich, httpx in `scorer_site.tgz`; then 20–30 SWE-smith / SWE-rebench repos), run each test under `sys.setprofile`. Record `E(t)` (executed src functions), `G_exec(t)` (directed, with call-site file:line) and test-IDF.

   Item = test source + candidate pool. The pool is `E(t)` ∪ an equal number of distractors matched on file and degree, with bodies truncated to 25 lines, ≤8k tokens. Candidates are listed in a **fixed canonical order** (file, then line), not chosen by the model, so there is no list to terminate.
2. *Commit items.* From Gemma's own trajectories on training tasks (SWE-smith / rebench, never the 129 dev tasks), take the prefix up to the diagnosis turn. Append a `CANDIDATES` block: functions the agent has displayed ∪ functions on `G_exec(F2P)`. Labels y = gold function set (if built, refined by IFG-style necessity labels: necessary = 1, incidental = 0.5).
3. *Label rule (idea 12 folded in).* An unobserved candidate (not executed by any F2P test and not gold) gets no penalty: its `S_v` is dropped from the loss.

**Objective.** Output format, one line per candidate: `name: EDIT|SKIP p=0.05|0.2|0.5|0.8|0.95` (or `CALLED|NOT`).
- Per-element proper log score `S_v` against y (IDF-weighted variant: positive terms × idf(v)/mean idf).
- **Element-wise best-of-K recombination**, which keeps training cheap offline:
  1. sample K=4 completions per item (prefix-cached prompt);
  2. for each element pick the line with the highest `S_v`;
  3. assemble one target completion from the model's own lines;
  4. run SFT on that target, plus one negative pass with the clipped surrogate on the worst sample's lines, where Â = `S_v − mean_K`.

  This gives one sequence per prompt instead of K, cutting training tokens by K× (INFERRED).
- Agent transfer: the localiser prompt gains a `CANDIDATES` field with the same format. A prompt-only change; harness unchanged. The fixer reads it.
- Mix: 70% RFT-filtered resolved trajectories (replay) + 30% CASE items.

**PEFT and offline loop.** Default config (r=32). Iteration 0: zero-shot. Iteration 1: recombination SFT. Iteration 2: regenerate both items and agent trajectories, then retrain. Online GRPO: element advantages are per-token advantages over the K group, a direct drop-in.

**Compute on 4×L4 (GUESSED throughputs).**

| Step | Cost |
|---|---|
| CPU tracing | 3 repos ≈ 2–4 CPU-h; 25 repos ≈ 1 day on Kaggle CPU kernels |
| Generation | 600 items × K=4 at ~6k prompt tokens, prefix-cached → ~3.6M prefill + ~0.7M decode tokens ≈ 0.3–0.6 GPU-h |
| Training | 600 recombined targets × 6k = 3.6M tokens → 1.5–2.5 h. With 400 commit items and 300 replay trajectories at 12k tokens: ~10M tokens ≈ 4–7 h per iteration |

Two iterations fit about one weekly quota.

**Why it keeps W4A16 near SOTA.** Targets are the model's own lines, recombined, so there is almost no distribution shift. The proper score preserves calibration. The format is one the quantised model already produces. The replay mix protects the base behaviour.

**Generalisation.**
- *Models and hardware:* any LLM, any set-valued graph query (neighbours, reachability sets, k-hop sets, impact sets). It needs only a labeller: tracer, graph oracle or KG.
- *Graph benchmarks:* RepoReasoner call chains; KGQA multi-answer questions (SOG-R1's prompt *pleads* "examine every single one of those entities", VERIFIED p.13, which is evidence that under-enumeration is known and is being patched by prompting).

**Surprising predictions.**
1. P1: zero-shot atomisation alone gives recall ≥0.65 at ≤0.10 precision loss on executed call sets.
2. P2: after training, `OTHER_LOCATIONS: none` on ≥3-gold tasks falls from 72% to <40%, and resolve on ≥2-gold tasks rises while single-function resolve stays flat.
3. P3: the gain survives IDF weighting, so it is not just "say yes to hubs".

**Kill test (CPU ~1 day + ≤0.4 GPU-h this week).**
- Data: 150 trace items (requests/rich/httpx, function level). Gemma W4A16, greedy, under list format (L) and atomised format (A).
- Controls:
  - **GF:** same items, bodies removed (names, signatures and test only).
  - **RW:** labels recomputed as reachability from the test node in a degree-preserving rewire of `G_exec(t)`.
- Metrics: recall at matched precision; AUROC of p; all stratified by depth, edge type (lexical/dark) and direction (ideas 2, 3 and 5 for free).
- Kill if any of:
  - (K1) recall(A) − recall(L) < 0.15, or the precision drop is > 0.15;
  - (K2) AUROC(A) − AUROC_GF(A) < 0.05 (decisions are name-matching);
  - (K3) AUROC against true labels − AUROC against RW labels < 0.05 (decisions not structure-driven).
- A 0-GPU proxy run on any API model can precede it, but it says nothing about W4A16.

**Publishable even if it fails.**
- K1 fails: the recall deficit is perceptual, a clean refutation of the format account of RepoReasoner, with depth and edge-type stratification showing where perception breaks.
- K3 fails: correct call-set predictions are not structure-driven, which is direct evidence for the "barely uses structure" reading of anomaly 4 in code.

## R2. ERC: Executable Rewire Counterfactuals

**Mechanism (M4b).** In natural tasks, topology carries little label information beyond text (GRA, CNY), so outcome training never forces propagation along edges. **Code is the one domain where a degree-preserving rewire is itself an executable program**: swapping call targets yields a valid program whose behaviour the tests measure. That manufactures items whose answer is carried by topology by construction, with exact labels.

**Data generation (CPU).**
1. Resolve static call sites (AST; `jedi` for attributes) and keep those that are **executed** (on `G_exec`) with a unique resolved callee.
2. Sample double-edge swaps (a→b at site s1, c→d at site s2) ⇒ (a→d, c→b). Constraints: b and d are signature-compatible (same required positional arity, compatible kwargs); a≠c; b≠d; no self-loop; both sites covered. Rewrite the callee expression at each site, adding imports if needed. Every node keeps its in/out degree.
3. Run the covering tests (coverage-selected; 300 s cap). Label the per-test flip y_t, the exception type, and the first divergent frame (a or c side). Keep swaps with 1 ≤ #flips ≤ ½·#covering tests. Keep ~25% zero-flip swaps (equivalent mutants) as negatives.
4. Item types:
   - (i) *Forecast:* repo excerpt (bodies of a, b, c, d; covering-test names with first-level callees) + swap diff → atomised per-test FLIP p (CASE format) + divergence frame.
   - (ii) *Agentic "find the swap":* the repo has 1–3 swaps; the issue = failing test output; the agent uses the harness tools in the sandbox replica, and success = tests pass after the fix. These are SWE-like structural-bug tasks, mixed into agent RFT.

**Objective.**
- Forecast items: per-test log score + 0.5·[frame correct], trained with CASE's element-wise recombination.
- Agentic items: outcome advantage (fraction of F2P passing) + 0.3 × CASE log score of the forecast the agent writes before `submit_patch`. Prompt-only.

**PEFT and offline loop.** Default config. Data mix per iteration: 30% forecast, 20% agentic ERC, 50% real-task replay. Online GRPO: rewards are per-rollout scalars plus per-element token advantages.

**Compute.**

| Step | Cost |
|---|---|
| CPU | 3 repos × 200 swaps × 10–60 s of selected tests ≈ 2–10 CPU-h (one Kaggle CPU day with 4 parallel workers) |
| GPU generation | 1,000 forecast items × K=4 × 5k ≈ 0.5–1 h; 300 agentic rollouts ≈ 4–8 h (GUESSED 40–80 rollouts/h) |
| Training | ~5M forecast tokens + ~4M agentic tokens ≈ 4–6 h |

**Why it keeps W4A16 near SOTA.** On-policy samples; the auxiliary task shares the format with the agent's real decisions; replay mix.

**Generalisation.** Any language with tests and a tracer. Any *executable* graph (build DAGs, workflow graphs, dataflow pipelines). The rewire is the standard graph-ML null model made into a training distribution.

**Surprising predictions.**
1. P1: on forecast items, a zero-parameter baseline ("is the swapped site on `G_exec(t)`?") reaches AUROC ≥0.85, while zero-shot Gemma is ≤0.70 and ≈ its own GF prompt. BFS beats the 31B model at structural effect prediction.
2. P2: after ERC, the agent's **rewire sensitivity** (resolve drop when `G_prov` is replaced by a rewire) *increases*. It now relies on edges. Meanwhile resolve with graph tools removed does not fall, because structure is internalised rather than tool-dependent.
3. P3: transfer to RepoReasoner call-chain recall and to our ≥2-gold tasks, with no training on either.

**Kill test (CPU ≤1.5 days + ≤0.5 GPU-h).** 300 swaps on requests and rich.
- (K1) Yield: <25% of swaps are non-trivial → kill.
- (K2) The label is structural: reachability-on-`G_exec` AUROC ≥0.80, and the GF / text-only baseline (TF-IDF similarity between test source and swapped functions) must be ≤ reachability − 0.10. Otherwise kill: the labels are text-predictable.
- (K3) RW: reachability on an independent degree-preserving rewire of `G_exec` must fall to ≤0.65. This confirms the wiring, not the degree sequence, carries the label.
- (K4, GPU) Gemma zero-shot on 150 items with three contexts: real, GF (diff + test names only), RW (callee lists shown from a rewired graph). If Gemma(real) ≥ reachability − 0.02, there is nothing to teach: kill as a method, report as a positive finding.

Prior closeness for the critic: predictive mutation testing and learning-based mutant selection (2610.06055, SNIPPET; the ISSTA'16 PMT paper is INFERRED from memory, unchecked); SWE-smith bug synthesis.

**Publishable even if it fails.**
- K1/K3: a census of how often degree-preserving rewires of real call graphs are behaviourally *equivalent* (test-suite structural adequacy), new for graph-ML null models on programs.
- K4 positive: "LLMs already propagate effects along code edges when the question is structural", which counters the anomaly-4 pessimism.

## R3. SAST: Structure-Ablated Self-Teacher

**Mechanism.** This is ideal-observer credit (Phase 1(c1)) made *structural*.

A self-teacher given privileged information scores the student's actions, and the log-ratio estimates pointwise information gain. Privileged-context OPSD is crowded: CNY uses destination recaps (VERIFIED), "What should a self-teacher see?" finds more privileged information is not always better (2609.25623, VERIFIED), and agent OPSD variants exist (2609.29051, 2609.36642, 2609.40285, SNIPPET).

The new piece is the **counterfactual baseline**. Give the teacher a degree-preserving *rewire* of the same privileged graph: same nodes, same degrees, same text, wrong wiring. Then subtract. The difference is the pointwise mutual information between the action and the *edges*, controlling for node identity and degree. It is the CNY rewire control turned into a credit signal (M4a).

**Signal.** For trajectory τ (W4A16 + LoRA_k), assistant action tokens y_t (tool names and arguments, `edit_file` targets, `CANDIDATES` lines):
- `δ^G_t = log π0(y_t | P(G), s_<t) − log π0(y_t | s_<t)`
- `δ^ρ_t` and `δ^∅_t`: the same with P(ρG) (rewire) and P(V(G)) (node list, the GF control).
- Structural credit `c_t = δ^G_t − δ^ρ_t`.
- Advantage `Ã_t = Â_τ + β·clip(c_t, −2, 2)`, β ∈ {0.02, 0.05}. `Â_τ` = group-normalised outcome (fraction of F2P passing, P2P guard) over K=4 samples per task.

Details:
- **P(G)** is a compact directed edge list of `G_exec(F2P)` restricted to test → gold-node paths plus 1-hop executed neighbours, ≤150 edges, rendered `caller -> callee (file:line)`. It has **no code, no patch and no bug marker**. That limits reference borrowing (RC-OPD risk noted in the critic, C9).
- **ρ** is a directed double-edge swap on that list (10×|E|).
- π0 is the *served* W4A16 base, so the teacher is exactly the deployed model. Log-probs come from vLLM `prompt_logprobs` (prefill only). The privileged block sits at the top of the system prompt in the teacher views only.
- Ablation arms: plain privileged OPSD (`δ^G`), the GF teacher (`δ^∅`), and generation-side scaffolding (idea 9).

**Compute.**

| Step | Cost |
|---|---|
| Teacher scoring | 300 trajectories × 4 views × ~12k tokens ≈ 14M prefill ≈ 1–1.5 GPU-h (GUESSED 3–5k tok/s) |
| Training | 300 × 12k = 3.6M tokens ≈ 1.5–2.5 h |
| Generation | 300 rollouts ≈ 4–8 h |

That is ~8–12 GPU-h per iteration. Tracing `G_exec(F2P)` is CPU.

**Why it keeps W4A16 near SOTA.** The teacher *is* the frozen served model. β·c is a small, signed, token-local nudge (CNY reports |βδ| ≈ 0.013–0.027 nats, VERIFIED), which acts like a KL-anchored update.

**Generalisation.** Any model. Any graph task with a privileged graph view: KGQA gold subgraphs, TAG neighbourhoods, dependency graphs. With online GRPO it is the CNY recipe with the rewire baseline added. The cost is three extra prefills per rollout, about +60% over CNY's ~24% (INFERRED).

**Surprising predictions.**
1. P1: c_t ≈ 0 on the first 1–2 navigation actions, which are lexically driven. Mass concentrates at actions ≥2 executed hops from issue-named symbols and on `CANDIDATES` / edit-set decisions in multi-function tasks.
2. P2: `δ^∅ ≈ δ^G` on single-function tasks (nodes suffice) but not on multi-function tasks.
3. P3: training on c beats training on δ^G (plain OPSD) on held-out multi-function tasks. Subtracting the rewired teacher removes "more facts" credit that otherwise rewards copying names from the privileged block.

**Kill test (≤0.3 GPU-h scoring + CPU tracing).**
- Inputs: 40 trajectories (raw r2/r3 traces if retrievable from Kaggle outputs, GUESSED; otherwise 40 fresh rollouts ≈0.6 GPU-h, which would take the week's whole remainder). Trace F2P on CPU.
- Controls built in: GF = `δ^∅`, RW = `δ^ρ`, and the null = c computed from two independent rewires, `δ^{ρ1} − δ^{ρ2}`.
- Kill if any of:
  - (K1) the distribution of |c_t| on action tokens is indistinguishable from the ρ1−ρ2 null (KS p > 0.1): the teacher ignores edges;
  - (K2) per-action c does not separate actions whose observations reveal a gold function (AUROC < 0.6);
  - (K3) `δ^∅` explains ≥90% of the variance of `δ^G`: node identity, not structure. Fall back to plain OPSD, which is crowded.

**Publishable even if it fails.** "Self-teachers treat privileged graphs as node lists." The rewire-teacher is a reusable instrument for *measuring* whether an LLM uses edges, at one prefill per view. It is a direct, cheaper and finer-grained answer to the contested anomaly 4.

## R4. ALIAS: Observation-equivalent action marginal likelihood (tool-surface invariance)

**Mechanism (M3b).** When tools overlap, the policy's mass and the outcome credit split across syntactically different actions that deliver the same information. In our setting:
- `get_code_neighbors(X)` ≈ `grep -rn "X("` plus reading the definition;
- `search_similar_code(name)` ≈ `grep -n "def name"`.

Graph tools are optional and may be absent on hidden repos. Observed usage is 0 graph calls in v5 (VERIFIED, `behaviour_per_task.json`) and 45/129 tasks with any graph call in v2 (VERIFIED). The deployable goal is an agent whose competence is defined at the level of the **information act**, invariant to which tools exist.

**Signal.**
- `info(o)` = the set of (file, symbol, line-span) facts parsed deterministically from an observation.
- Rule-based aliasers rewrite each logged information action into candidates across tools: graph tool, grep variants, `sed -n`, `read_file` ranges, `python -c` with `ast`. They are executed at the same replayed repo state.
- `C(a_t) = {a′ : J(info(o′), info(o_t)) ≥ 0.8 and tokens(o′) ≤ 1.5·tokens(o_t)}`.
- Loss on action tokens: `L = −w_τ · log Σ_{a′∈C(a_t)} π_θ(a′ | s_t)`, using sequence log-probs of each alias's action text. Its gradient is posterior-weighted, so it reinforces whichever alias the model already prefers instead of forcing the teacher's syntax.
- **Tool-surface dropout:** 50% of training contexts drop the graph tools from the system prompt, and C is restricted to available tools.
- `w_τ` = outcome advantage. Tests enter through outcome; command execution builds the equivalence oracle.

**PEFT and offline loop.** Default config. It replaces the plain RFT loss on information actions; edit and submit tokens keep the standard loss. Online GRPO: compute the per-token advantage at the class level, i.e. share the group advantage across aliases in the same class (INFERRED, needs a derivation check).

**Compute.**

| Step | Cost |
|---|---|
| Alias execution | CPU, ~5 aliases × ~30 actions × 300 trajectories ≈ 45k cheap commands ≈ a few CPU-h |
| Scoring | Training computes alias log-probs: ~5 short action sequences per step, a small overhead (<10%) with prefix sharing (INFERRED) |
| Training | Same as RFT: 300 × 12k ≈ 1.5–2.5 h per iteration |

**Why it keeps W4A16 near SOTA.** It *loosens* the imitation target relative to SFT (marginal over aliases), so it moves the model less than RFT does. It directly targets the tool-call error channel that INT4 amplifies (Flat Score).

**Generalisation.** Any agent with overlapping tools: MCP-rich agents, KG agents with `Search` vs SPARQL. For graph reasoning it formalises "the graph interface does not matter, the information does" (GRA's selective-access finding, M2b) as a training objective.

**Surprising predictions.**
1. P1: ≥30% of Gemma's information actions have ≥2 executable aliases, and Gemma's within-class distribution has ≥1 bit of entropy. Splitting is real.
2. P2: in a RepoNavigator-style test on our harness (adding a *renamed duplicate* of `run_command` restricted to grep), plain RFT degrades but ALIAS-RFT does not.
3. P3: after ALIAS, |resolve(graph tools) − resolve(no graph tools)| < 1 pp, and graph-tool use rises only for hub-caller queries, where its observation is cheaper.

**Kill test (CPU + ≤0.3 GPU-h).**
- Build the aliasers and execute them for 300 logged actions in `scorer_site` envs; then score aliases by prefill.
- Kill if (K1) the alias rate is <20%, or (K2) the top alias holds ≥0.9 of class mass in ≥80% of classes.
- **GF:** aliases from grep/read only (no graph tools). **RW:** the graph tool backed by a degree-preserving rewire of `G_prov`. ≥90% of its observations must fall *outside* the classes, a sanity check that `info()` is fine-grained enough. If not, fix the extractor before any training.

**Publishable even if it fails.** K2 failing means tool redundancy does not split the policy, so RepoNavigator's drop must be credit dilution (M3a) or observation cost (M3c). That is a mechanism discrimination nobody has run.

## R5. SENSOR: tests as sensors (spectrum information gain)

**Mechanism.** The ideal observer chooses experiments that maximise information about the fix set (Phase 1(c2)). Test executions are *graph observations*: coverage spectra over `G_exec`. In SBFL, information comes from the differential between failing and passing executions on the graph.

Gemma under-experiments (6 reproductions among 47 wrong-never-tested, VERIFIED). F2 shows *naive* reproduction does not pay (2/17). The hypothesis is that **informative** experiments pay, and that the agent can be taught to run them.

**Signal.**
1. Replay each trajectory's mutating commands to reconstruct the repo state (the CERT replay machinery, `critic_design.md` §1a).
2. For each experiment command (pytest, python scripts importing repo code), re-execute it under a coverage tracer to get its spectrum over src functions.
3. Surrogate posterior over candidates: `p_t(v) ∝ exp(κ·Ochiai_t(v))`, with Ochiai from all experiments up to t (uniform before the first). κ is fit once on held-out tasks.
4. Truth-weighted gain `g_t = log p_t(F*) − log p_{t−1}(F*)`, where F* = necessary/gold set (privileged, offline). A set likelihood uses the mean over F*.
5. Advantage on the experiment's command tokens: `Ã_t = Â_τ + β·clip(g_t, −3, 3)`.

Tests enter in three places: as sensors (spectra), through F* (via test-verified gold), and as outcome.

**PEFT and offline loop.** Default config. Online GRPO: compute the spectrum in the rollout sandbox, a drop-in token credit.

**Compute.**

| Step | Cost |
|---|---|
| Replay + tracing | ~10–20 experiments per trajectory × 10–60 s → 300 trajectories ≈ 15–60 CPU-h (parallel CPU kernels, 1–2 days) |
| GPU | Same as RFT (generation 4–8 h, training 1.5–2.5 h per iteration) |

**Why it keeps W4A16 near SOTA.** On-policy; credit only on command tokens; small β.

**Generalisation.** Any domain where actions are experiments on a graph-structured system: KG `Search` calls (posterior over answers), CI pipelines, distributed-system traces. For graph reasoning it trains *active* structure acquisition, not passive structure use.

**Surprising predictions.**
1. P1: Σg per trajectory predicts resolve (AUROC) better than localiser function-hit (≈ chance, VERIFIED L1), and better than the number of tests run (negative, F2).
2. P2: g-trained agents run *fewer* but more discriminating experiments: reproduction scripts that hit the fault's executed neighbourhood, not whole-suite runs. Multi-function resolve rises.
3. P3: RW-computed Σg loses its predictive power, while GF Σg (pass/fail outcomes plus traceback names) keeps some. Structure carries the residual.

**Kill test (CPU ~1.5 days; 0 GPU-h if raw traces are retrievable, else ≤0.6 GPU-h for 30 fresh rollouts).**
- Kill if any of:
  - (K1) AUROC(Σg) ≤ max(AUROC function-hit, AUROC GF-Σg) + 0.05;
  - (K2) RW-Σg is within 0.03 of true Σg (no structural content);
  - (K3) <30% of trajectories contain any experiment. Then a prompt nudge must precede training; that is allowed, but the signal is then mostly synthetic.

**Publishable even if it fails.** "Information gain of agent experiments, measured on executed graphs, does not explain ~30B-agent success." The bottleneck is conversion (edit/commit), not observation. Together with F2 this closes off a whole family of "make the agent test more" interventions.

## R6. GCSR: Graph-compiled synthetic repos, with protocol/content factorisation

**Mechanism (M5b).** The SFT warm-start gain (SOG-R1 CWQ: base 35.3 → SFT 86.1 vs RL-only 63.0, VERIFIED) is mostly *protocol*: how to query, batch, page, verify and commit. It is not *content*. If so:
- warm-start data need not come from real repos or from teachers;
- procedurally compiled repos with controllable topology suffice;
- the synthetic distribution's topology statistics (depth, fan-in, hub ratio, dark-edge share) become the knob that controls transfer.

Our hidden repos are private (HQ), so content cannot transfer anyway. This is the regime where protocol learning is all that matters.

**Generator.**
1. Sample a directed graph H with typed edges: direct call, base-class dispatch, decorator wrap, registry/dict lookup, callback (the last four are "dark"). Either from a parametric family (depth D, branching, hub ratio) **or by copying the topology of a real repo's `G_exec`**.
2. Compile to a Python package. Each node becomes a function or method whose small deterministic body transforms its inputs and calls its children, so values flow along edges. Identifiers are drawn from real-repo vocabularies, giving realistic names with no semantic hints.
3. Add tests at the roots.
4. Inject 1–3 bugs with SWE-smith-style operators. Multi-node bugs are made **jointly necessary by construction** (a contract change that two nodes must agree on).
5. Write the issue from the failing symptom. It names the entry only; the GF variant names the bug node.
6. Size 200–3,000 functions, so navigation under 32k context matters.
7. Tools are exactly the harness tools plus optional graph tools backed by a deliberately degraded H (undirected, merged labels, as in `G_prov`).

**Objective.** Teacherless RFT warm start: Gemma's own successes from K=4 samples, weighted by outcome advantage. Reward = F2P pass, plus the CASE log score on the `CANDIDATES` line. A curriculum on D (reverse-curriculum flavour) addresses M5a too.

**Factorisation experiment (the paper result; next week, ~10–15 GPU-h).** Equal-token RFT warm starts:
- (a) real-repo SWE-smith/rebench trajectories;
- (b) GCSR with real-copied topology;
- (c) GCSR with a degree-preserving rewire of the same topologies (**RW control**);
- (d) GCSR depth-1 only (**GF control**: no traversal needed).

All are evaluated on held-out real repos and our dev set.

**Compute.**

| Step | Cost |
|---|---|
| Generator and validation | CPU |
| Rollouts | Synthetic tasks are short: ~30–80 rollouts/GPU-h (GUESSED); 400 rollouts ≈ 5–10 h |
| Training per arm | ~2–4 M tokens ≈ 1–2 h |

**Why it keeps W4A16 near SOTA.** On-policy RFT. The synthetic share is capped at 50% and mixed with real replay.

**Generalisation.** Model- and hardware-agnostic. Ideal for online GRPO: infinite, verifiable, difficulty-controllable tasks. A graph-ML-native benchmark generator where topology is the experimental variable.

**Surprising predictions.**
1. P1: (b) recovers ≥70% of (a)'s gain on held-out real repos.
2. P2: (b) ≈ (c) within 1 pp. Topology *fidelity* is irrelevant; the degree/depth distribution is what transfers.
3. P3: (d) recovers ≤30%. Multi-hop protocol is the active ingredient, and multi-node synthetic bugs move the real ≥3-gold dead zone more than real single-function data does.

**Kill test this week (CPU ~1.5 days + ≤0.6 GPU-h).**
- Generator v0 (DAG, three edge types, 1–2 bug nodes); 60 validated tasks at D ∈ {1, 2, 4} × size ∈ {300, 1,500}.
- Gemma zero-shot on 30 in the harness replica (8 parallel streams).
- Kill if (K1) success does not fall with D (Spearman > −0.1), since the gym would not exercise graph reasoning; or (K2) the GF variant (issue names the bug node) succeeds no more than the normal one.
- Calibrate, not kill: at D=1, success <30% (unnatural tasks) or at D=4 >80% (too easy).
- RW measurement: tasks compiled from requests' `G_exec` vs its degree-preserving rewire, both zero-shot.

**Publishable even if it fails.** If P1 fails, content (real API idioms) dominates the warm start. That warns synthetic-graph RL work (GraphWiz/G1-style) that transfer to real code graphs needs content, and it quantifies how much. If P2 fails, real call-graph topology has transferable structure beyond its degree sequence: a graph-ML finding in its own right.

---

## How the six fit together, and what to spend the 1.5 GPU-h on

- **One method, if the kill tests pass (INFERRED composition):**
  - **Data:** GCSR + ERC supply verifiable structural tasks at scale; real traces supply CASE labels.
  - **Credit:** outcome + SAST edge credit + SENSOR experiment credit, all additive token advantages.
  - **Likelihood:** ALIAS makes the information act, not the tool, the unit.
  - **Output:** CASE makes commit sets complete and calibrated.

  Every component is an additive term in `Ã_t` or a data source, so the same code runs offline (Kaggle) and online (GRPO).
- **This week, in order:**
  1. CPU: tracer + 150 CASE items; 300 ERC swaps; GCSR v0; alias executor; replay of r2/r3 traces if retrievable.
  2. GPU, ≈1.2 h total:
     - CASE list vs atomised: 0.3–0.4 h;
     - ERC zero-shot on 150 items × 3 contexts: 0.3–0.4 h;
     - SAST 4-view scoring of existing traces: 0.2–0.3 h, only if raw traces exist; otherwise spend it on GCSR zero-shot.
  3. ALIAS and SENSOR are CPU-first. Their GPU parts move to next week.
- **Unknowns to measure before trusting any compute number:** vLLM prefill throughput on 4×L4 W4A16 (GUESSED 3–6k tok/s); agent rollouts per GPU-h (GUESSED 40–80); whether raw r2/r3 trajectories are retrievable from Kaggle outputs (GUESSED yes).

---

## Search log

Every query this session, in order. Tool: alphaXiv (`answer_pdf_queries` / `discover_papers`).

1. **Paper read 2607.18481 (SOG-R1).** Queries: RL-only vs SFT-only vs SFT+RL on CWQ; why RL-only fails. Found Table 3 (Base 50.9/35.3/55.5; RL-only 80.9/63.0/60.7; SFT-only 89.6/86.1/88.8; SFT+RL 91.7/88.7/90.0), the text/table inconsistencies, the data-scale ablation, the Qwen3-4B replication, and a student prompt that demands examining every returned entity. → anomaly 5 corrected.
2. **Paper read 2608.29588 (CNY).** Queries: rewire setup and interpretation; what it preserves; no-walk baseline. Found rewire 76.6 → 73.0 ≈ preview+direct 73.4; walks/node 1.24 → 2.01; embedding-similar pick 22%; OPSD β-sweep; the recap prompt. → anomaly 4 reconciled reading.
3. **Discover:** "training LLMs to predict execution traces or dynamic call graphs of repository code, rewarded by running tests…"; keywords [execution trace prediction, call graph, RepoReasoner, SWE agent]; recency.
   - Returned RepoReasoner 2607.25996, ExecCritic 2609.09133, Socratic-SWE 2606.07412, GRADE 2606.22741, Neural Debugger 2603.09951, ExecVerify 2603.11226, "Understanding by Reconstruction" 2603.11103, and others (SNIPPET).
   - Closest to CASE/ERC labels: Neural Debugger and ExecVerify (execution-trace training, function-level). None seen does set-valued call-graph emission or rewire counterfactuals. Not read.
4. **Discover:** "LLM graph reasoning trained with counterfactual or perturbed graphs (edge rewiring…) as contrastive or invariance signal"; keywords [counterfactual graph, degree-preserving rewire, graph reasoning, GRPO].
   - Returned Counterfactual Self-Evolving Agents 2609.32870, GNN counterfactual-explanation papers (RACE 2609.37650, 2609.05113, 2608.12083), NormWorlds-CF 2607.03957, GRAIN 2608.27142, Visual Graph Scaffolds 2606.02673, GraphCert, GTA-RAG (SNIPPET).
   - No hit uses an executable degree-preserving rewire as training data or as a teacher baseline. Absence only.
5. **Paper read 2607.25996 (RepoReasoner).** Queries: task definition and ground truth; precision/recall numbers; context and error analysis. Found file-level call chains from dynamic pytest tracing, Table 2 including **Oracle recall 0.355 (GPT-4.1-Mini)**, the error categories, and 169 call-chain samples from 14 repos.
6. **Discover:** "LLMs predicting whether a code mutant is killed / which tests fail after a change, as a training signal"; keywords [predictive mutation testing, mutant, test outcome prediction, LLM].
   - Returned test-aware LLM mutant generation 2609.35841, RTM 2607.03223, intent-based mutation 2607.05149, Test vs Mutant 2602.08146, Code-A1 2603.15611, mutant-selection budget 2610.06055 (SNIPPET).
   - ERC's closest neighbours are mutation-testing ML. None seen uses degree-preserving call-target swaps as a graph-reasoning training distribution. Absence only.
7. **Discover:** "on-policy self-distillation where the teacher is the same model given privileged context… multi-turn agents or code"; keywords [on-policy self-distillation, privileged information, OPSD, agent]; recency.
   - Returned 2609.29051 (privileged info for multi-turn agents), PIVOT 2609.35303, RE-0 2609.32416, PivotOPD 2609.40285, PR-OPD 2609.36642, GUI-OPSD 2609.27307, **2609.25623 (What should a self-teacher see?)**, ASCENT 2610.05303, AdviSD, TISD 2609.30878, UOPD 2609.34036 (SNIPPET).
   - The field is crowded, so SAST's only claimed delta is the rewired-teacher baseline (structural credit).
8. **Paper read 2609.25623.** Queries: which privileged contexts were compared; whether more information hurts. Found that abstract contexts beat full solutions by 1.4–1.6 pts at 4B/8B, answer-only stays competitive, and initial KL does not order utility (math only, not code or graphs).

**Local computations (VERIFIED arithmetic):**
- F1 from `results/runs/r3_v5_graph/behaviour_per_task.json` and `results/runs/r2_v2_sed/per_task.jsonl`.
- F2 and graph-call counts from `r2_v2_sed/per_task.jsonl`.

**Not searched (budget):** RepoNavigator re-read (inherited VERIFIED); GRA re-read (inherited); OpenReview; direct check of the ISSTA'16 predictive-mutation-testing paper; RL's-Razor-style forgetting evidence for on-policy RFT (cited from memory as INFERRED only).
