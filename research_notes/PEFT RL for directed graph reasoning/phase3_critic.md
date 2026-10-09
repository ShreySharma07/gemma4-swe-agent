# Phase 3 critic: CASE, ERC, SAST, ALIAS, SENSOR, GCSR (8 Oct 2026)

**Role:** adversarial area chair plus ML-systems engineer, reading with fresh eyes.

**Inputs:**
- `inventor_phase1_2.md` (Phases 1 and 2);
- `docs/training_feasibility_4xl4.md`;
- `docs/HQ.md`;
- `results/runs/*`;
- `LLM graph reasoning for code/critic_judgement.md`.

Every search I ran is in the log at the end. Labels follow the inventor's file: VERIFIED, SNIPPET, INFERRED, GUESSED.

**Kill rules I applied:**
1. The idea does not fit **25 GPU-h in total**, counting data rollouts, teacher scoring, training, evaluation and both controls.
2. The kill test cannot be falsified. That covers vague thresholds, no pre-registered metric, a test that cannot separate the claimed mechanism from alternatives, a missing graph-free (GF) or degree-preserving rewire (RW) control, and confounds.
3. Prior art already exploits the same mechanism.

The top two ideas by **surprise** survive one extra round regardless of kills.

---

## Verdict summary

| Idea | GPU-h as specified | GPU-h narrowed | Falsifiability (0–3) | Surprise (0–5) | Prior-art risk | Verdict |
|---|---|---|---|---|---|---|
| **GCSR** | 35–65 | 16–21 | 1 | **4.0** | Medium | **SURVIVE-BY-SURPRISE** |
| **SAST** | 27–37 | 13–20 | 1 | **3.5** | **High** (mechanism) | **SURVIVE-BY-SURPRISE** |
| SENSOR | 25–35 | 15–20 | 1 | 3.0 | Medium | NARROW |
| CASE | 34–55 | 13–16 | 1 | 2.5 | Medium | NARROW |
| ERC | 34–45 | 7–9 | **0** | 2.5 | **High** | KILL as specified (a forecast-only salvage is possible but low value) |
| ALIAS | 23 if tree-masked; >100 if naive | n/a | 1 | 1.5 | High (MML) | **KILL** |

All six fail rule 1 or rule 2 as written. None is publishable as specified.

---

## 0. Cross-cutting findings (these change every budget)

**X1. Fixed costs the inventor's tables omit (charged to every training idea).**
- **Training-feasibility probe: ~1 GPU-h.** It has not been run (feasibility doc, "Nothing below has been run on an L4 yet"). Nothing trains until it passes.
- **Per-session overhead.** vLLM start is 480 s (VERIFIED, HQ Oct 7), plus environment restore. An FSDP load is about the same. Each generate→train iteration needs at least two sessions, so ≈0.3–0.5 h of overhead per iteration.
- **This week's 1.5 GPU-h** therefore covers *either* the probe plus one prefill-only kill test (≈0.4 h), *or* two kill tests and no probe. The inventor's plan (≈1.2 h of kill tests) silently assumes the probe is free.

**X2. Evaluation is the hidden budget sink.**
- A full dev run is 129 tasks.
- At GUESSED 40–80 rollouts/GPU-h, that is 1.6–3.2 h, plus ≈0.2 h overhead, so **≈1.8–3.4 h per arm**.
- Sequential r2 gives a floor: median task duration 93 s, i.e. ~38/h (VERIFIED, `r2_v2_sed/per_task.jsonl`).
- Four arms (base, method, GF-trained, RW-trained) cost **7–14 GPU-h of evaluation alone**, which is 30–55% of the 25 h.
- Every surviving design must push the GF and RW controls onto cheap item-level evaluations, and keep at most two arms for agent-level evaluation.

**X3. Statistical power: most agent-level predictions cannot be falsified at n=129.**
- Base resolve is ~20%. With ~15% discordant pairs, the SE of a paired difference is ≈3.4 pp, so the **minimum detectable effect is ≈7–9 pp**.
- On the ≥2-gold slice (65 tasks, 7 resolved, VERIFIED) the MDE is ≥12 pp.
- These predictions are therefore unfalsifiable as stated:
  - "within 1 pp" (GCSR P2, ALIAS P3);
  - "<1 pp" (ALIAS);
  - "multi-function resolve rises" (CASE P2, SAST P3, SENSOR P2).
- They must become TOST equivalence tests with a pre-registered margin of ≥6 pp, or move to dense item-level metrics with n ≥ 300.

**X4. Raw trajectories are not local (VERIFIED).**
- `results/runs/r2_v2_sed` and `r3_v5_graph` contain only per-task summaries (`per_task.jsonl`, `behaviour_per_task.json`, `task_results.jsonl` with 10 scalar fields).
- The inventor's "0 GPU-h if raw traces are retrievable" (SENSOR, SAST, ALIAS) is GUESSED and unsupported locally.
- **Budget assumption:** every replay-based kill test needs fresh rollouts.

**X5. Dev-set contamination.**
- Dev-set repos (VERIFIED): fastapi 67, **rich 48, requests 13, httpx 1**.
- CASE and ERC build their *training* items from requests, rich and httpx, which is 62/129 dev tasks. A trained arm evaluated on dev would have seen those repos' code, tests and call structure.
- Kill tests (zero-shot) may use them. **Training items must come from non-dev repos**, and dev evaluation must report rich/requests separately.

**X6. Resolved-replay cost.**
- Every "70% replay of resolved trajectories" mix needs ~4–5 rollouts per resolved trajectory (resolve ~20–25%).
- So 300 resolved trajectories cost ≈1,200–1,500 rollouts, i.e. **15–37 GPU-h**. That alone breaks the budget.
- Fix: replay with *base-model self-generated outputs on agent prompts* (KL anchor, no outcome filter), or cap resolved replay at ≤100.

**X7. Length.**
- 65/129 r2 trajectories exceed 14k prompt tokens, and 6 exceed 16k (VERIFIED).
- Any method that adds a privileged block (SAST +≤3k) or a `CANDIDATES` block (CASE) pushes a large share over the 16k training cap. That needs prefix trimming, which is a confound to report.

**X8. Prefill throughput is probably overstated.**
- Arithmetic: 2 × 31 B ≈ 62 GFLOP per token. At 4 × 121 TFLOPS × 20–40% MFU that gives **≈1.5–3k tok/s**, not the GUESSED 3–6k (INFERRED).
- I use 1.5–3k below.

---

## R1. CASE: Commit-completeness via Atomised Set Emission

### Scores

| Axis | Score / estimate |
|---|---|
| **GPU-h as specified** | Probe 1.0 · kill 0.4 · item generation 0.5 · **commit-item rollouts (≥400, omitted by the inventor) 5–10** · resolved replay (X6) 5–10 at a 100-trajectory cap · training 10M tok × 2 iterations 8.6–14 · GF and RW trained arms 8.6–14 · evaluation of 4 arms 7–14 → **≈34–55 h ✗** |
| **GPU-h narrowed** (trace items only; one iteration; 3 trained arms on 3.6M tok each; item-level evaluation on held-out repos; 1 agent evaluation pair) | 1.0 + 0.4 + 0.5 + 3 × (1.6–2.5) + 1.0 + 2 × (1.8–3.4) ≈ **13–16 h ✓** |
| Falsifiability | **1/3** |
| Surprise | **2.5/5**: recasting RepoReasoner's "multi-hop weakness" as an emission artefact is interesting, but "pointwise raises recall over listwise" is IR folklore |
| Prior-art risk | **Medium** |

### Why falsifiability is only 1

1. **K3 is broken by construction.**
   - The RW labels are reachability from the test node in a rewire of **G_exec(t)**. That graph contains only E(t) ∪ {test}.
   - Distractors are 0 under both labelings. The rewire can only *remove* positives, those stranded in detached cycles, which are few because every non-root node keeps in-degree ≥1.
   - So RW labels ≈ true labels. AUROC_true − AUROC_RW < 0.05 is near-certain **whatever the model does**: K3 kills every outcome and measures label-perturbation size, not structure use.
2. **K1 cannot separate M1a (termination prior) from "forced per-candidate attention" or plain yes-bias.**
   - Per-item binary queries inflate "yes" (well known for LLM verifiers), so recall↑ with precision↓ is the default expectation.
   - The "matched precision" operating point is a free parameter: the p-threshold over 5 levels.
3. **The list (L) format is underspecified.** Same pool? Same order? Same output budget? Different output lengths confound it.
4. **The training story is thin on RL.** Element-wise best-of-K recombination picks, per element, the sampled line with the highest S_v given the label. With K=4 and 5 confidence levels, some sample nearly always carries the right extreme, so this is **SFT on gold labels projected onto the model's support**. Call it that, or the "RL-style" claim will be attacked.

### Edge cases

| Case | Failure |
|---|---|
| Incomplete or unsound call graph | `sys.setprofile` is per-thread. httpx/anyio thread-pool calls are missed unless `threading.setprofile` is set, so false negatives get scored as negatives. C-level calls are invisible. Decorator wrappers appear as `wrapper` frames in the decorator's module, so identity must be keyed on the code object, not the name. |
| Hubs | E(t) is dominated by `__init__`, utils and logging (RepoReasoner's own error analysis). IDF weighting helps the reward but not the K1 metric. Report K1 on IDF-filtered positives too. |
| Huge E(t) | Real tests execute 50–500 functions, while the pool cap is ≤8k tokens (~40 candidates with 25-line bodies). Subsampling the positives makes "recall" a function of the sampler. Pre-register the sampler (stratified by depth). |
| New-code fixes | For commit items, a gold function that does not exist in the base graph cannot appear in `CANDIDATES` (7/129 dev tasks have 0 parseable gold functions, VERIFIED). The format needs a `NEW:` slot, or those items must be dropped and counted. |
| Recursion / cycles | Fine for trace labels. Depth stratification needs a shortest-path depth. |
| Flaky / weak tests | Trace labels are robust, since they are execution facts. Commit labels inherit regex-noisy gold (F1 is labelled "regex-noisy"). |
| >16k | `CANDIDATES` plus the diagnosis prefix exceeds 16k on ~half of the trajectories (X7). |
| Quantization | p is emitted as text over 5 levels, not read from logits, so this is robust to W4A16. ✓ |

### Prior art hit
- Pointwise vs listwise LLM ranking (IR literature, setwise ranking SIGIR'24, SNIPPET).
- Function-level localisation rankers (SweRank, LocAgent; inherited).
- "Exploration Structure in LLM Agents for Multi-File Change Localization" (2606.11976, SNIPPET, unread). It is the closest to the commit-set claim and the red team should read it.

### Verdict and fixes

**Verdict: NARROW.** As specified it is about twice the budget and its structural kill test (K3) kills every outcome. The format question is cheap and answerable.

Required fixes:
1. **Replace K3** with an input-side control. Show candidate bodies whose *call-sites* are permuted by a degree-preserving rewire of the repo's static call graph (text-level rewire), and/or recompute labels on a rewire of the **full repo graph including the distractors**.
2. **Add an EOS-suppression arm to K1.** Use the L format, ban the list-closing token (vLLM `logit_bias`), and force N=10 more items. Termination prior predicts that the extra items keep precision within 0.10 of the original list; perception failure predicts they collapse to the distractor base rate. This arm, not A vs L, discriminates M1a.
3. **Pre-register** the p≥0.5 threshold, the PR-AUC (both formats, with L scored by rank order), and identical pool, order and token budget for L and A.
4. **Train only on non-dev repos (X5).** Drop commit items and agent replay from the first paper. Measure transfer on the RepoReasoner call-chain set (169 items, cheap prefill) and with one dev evaluation pair.

---

## R2. ERC: Executable Rewire Counterfactuals

### Scores

| Axis | Score / estimate |
|---|---|
| **GPU-h as specified** | Probe 1.0 · kill 0.5 · forecast generation (20M prefill, ~5M unique after prefix cache) 0.6–1.0 · agentic rollouts 300 → 4–8 · training 4–6 · a trained control arm 4–6 · **evaluation of rewire sensitivity = (true G_prov, rewired G_prov, no graph tools) × (base, ERC) = 6 runs → 11–20** → **≈34–45 h ✗** |
| **GPU-h narrowed** (forecast-only, 2 arms, item-level evaluation) | 1.0 + 0.5 + 1.0 + 2 × (1.5–2.5) + 0.5 ≈ **7–9 h ✓** |
| Falsifiability | **0/3** |
| Surprise | **2.5/5** |
| Prior-art risk | **High** |

### Why falsifiability is 0

1. **K2 is near-tautological.**
   - By the reach–infect–propagate (RIP) condition, a test that never executes the swapped site cannot flip. So "is the site on G_exec(t)?" gets AUROC ≥0.80 by construction.
   - Worse, the forecast item lists **covering-test names** only. If only covering tests are candidates, reachability is constant and its AUROC is 0.5. K2 is undefined as written.
2. **K3 is tautological.** Randomising G_exec destroys coverage information by construction. It says nothing about either the model or the labels.
3. **K4 is an unfair comparison.** The reachability baseline uses privileged *dynamic coverage* that Gemma is never shown, so "BFS beats a 31B model" is guaranteed rather than a finding. Gemma must be shown the same G_exec edges, or the baseline must use static reachability only.
4. **Degree preservation is not shown to matter.**
   - Among covering tests, flips are decided by infection and propagation: does d compute something different from b *for these values*, and does that reach an assertion? That is value-semantic reasoning, not topology.
   - A single call-replacement mutant (a→d at one site, degrees change) gives the same kind of label.
   - The mechanism (M4b, "topology carries the label") is never isolated from "semantic difference between b and d".
5. **P2 is degenerate on this agent.** It predicts that rewire sensitivity rises after ERC, but v5 makes **0 graph calls** (VERIFIED). An agent that never queries G_prov has rewire sensitivity identically 0 before and after. P2 cannot be measured without a prompt that forces graph use, and that changes the deployed agent.

### Edge cases

| Case | Failure |
|---|---|
| Dynamic dispatch | Only statically resolvable direct call sites can be swapped (AST plus jedi). That samples **lexical** edges, the opposite of the dark edges the programme cares about (M1c/M2a). |
| Decorators | Swapping decorated callees swaps the wrappers, so registry and caching side effects create non-local flips. |
| Recursion | Swapping into a recursive callee gives infinite recursion. These items are RecursionError, a trivially detectable flip, or 300 s timeouts that blow up CPU cost. Filter and report them. |
| Type-incompatible yet "signature-compatible" | An immediate `TypeError` / `AttributeError` at the swap site makes the exception type a giveaway label. Drop flips whose first divergent frame is the swap site itself, or the "structural" items become syntax items. |
| Hubs | Hub swaps flip hundreds of tests, so the ½·#covering filter keeps few. Fine, but the census will be biased toward low-degree nodes: report the degree histogram of kept swaps. |
| Weak / flaky tests | Zero-flip ≠ equivalent: weak suites mislabel. Re-run flips 3×; report suite mutation score per repo. |
| Quantization | Not affected. |

### Prior art hit
- Predictive mutation testing (PMT, Zhang et al. ISSTA'16; GitHub `sei-pku/PredictiveMutationTesting`, SNIPPET).
- **Seshat**, which predicts full mutant×test **kill matrices** (TOSEM/ICSE'22, SNIPPET).
- **Contextual PMT** (Jain, Le Goues et al., mutant–test pair prediction with transformers, SNIPPET).
- Fine-grained PMT (2511.11999, SNIPPET).
- LLM equivalent-mutant detection (ISSTA'24, SNIPPET).
- "Method call replacement" is a standard mutation operator.
- The agentic "find the swap" variant is SWE-smith / BugPilot (2510.19898) bug synthesis.

What remains new is only the framing "a degree-preserving rewire is an executable graph-ML null model".

### Verdict and fixes

**Verdict: KILL as specified** (rules 2 and 3).

A salvage exists but is low value:
- forecast-only;
- Gemma shown the G_exec edges;
- an **equal-site single-replacement mutant** control to test whether degree preservation matters;
- flips at the swap site dropped;
- training repos disjoint from dev.

That salvage is a PMT study with an LLM, and its novelty is weak.

---

## R3. SAST: Structure-Ablated Self-Teacher (SURVIVE-BY-SURPRISE)

### Scores

| Axis | Score / estimate |
|---|---|
| **GPU-h as specified** | Probe 1.0 · rollouts 300 → 4–8 · teacher scoring 4 views × 300 × 12k = 14.4M prefill. The privileged block sits at the top of the system prompt, so **no prefix sharing across views** → 1.3–2.7 · training of 4 arms (outcome, OPSD δ^G, SAST c, GF δ^∅) 4 × (1.5–2.5) = 6–10 · evaluation of 4 arms 7–14 → **≈27–37 h ✗** |
| **GPU-h narrowed** (below) | **13–20 h ✓** |
| Falsifiability | **1/3** |
| Surprise | **3.5/5**. If c concentrates on ≥2-hop and commit decisions, and subtracting the rewired teacher beats plain OPSD, that is a direct, cheap instrument for the contested "do LLMs use edges?" question. If c ≈ null, "self-teachers read privileged graphs as node lists" is also a notable finding. |
| Prior-art risk | **High on mechanism** |

### Why falsifiability is only 1

1. **K1 cannot kill.** A KS test on thousands of action tokens rejects for any non-zero difference: p < 0.1 is effectively guaranteed. Kill tests need an **effect-size** criterion, not a p-value.
2. **Coherence confound.**
   - A degree-preserving rewire of `caller -> callee (file:line)` yields edges that are *visibly implausible*: the callee name never appears at that caller's call site.
   - The teacher can detect that the block is corrupted and **globally discount it**. c would then measure "trust in the privileged block", not edge use.
   - Nothing in the design separates the two.
3. **Node-set leak makes K3 likely to fire for the wrong reason.**
   - P(G) is "restricted to test → gold-node paths plus 1-hop neighbours", so **the node set itself points at the gold path**.
   - δ^∅ (node list) carries most of the information, and c ≈ 0 because the design made wiring redundant, not because the model ignores edges.
4. **Agent-level P3 (beats OPSD on multi-function tasks) is under-powered** (X3).

### Edge cases

| Case | Failure |
|---|---|
| New-code fixes | Gold is absent from G_exec(F2P), so P(G) holds no target and the teacher view is uninformative. Stratify them out and count them. |
| Huge traces | The ≤150-edge truncation has to choose edges. Choosing by "paths to gold" is the leak above. |
| Dynamic dispatch / decorators | The trace resolves them ✓. But a rendered `wrapper -> f` edge confuses the teacher; collapse functools wrappers. |
| Cycles / recursion | A self-edge survives in G but the rewire rejects self-loops, so ρG differs systematically on recursive nodes. Allow self-loops in both, or exclude them from both. |
| >16k | Teacher views of 14k + ≤3k are fine in vLLM (32k). Student training needs trimming on ~5–10% of trajectories (X7). |
| Quantization | c is a difference of two W4A16 log-probs for the same tokens: deterministic weights, but vLLM batch and length nondeterminism is ~1e-2 nats/token (INFERRED). That is the same order as the per-token c after β scaling (CNY: |βδ| ≈ 0.013–0.027 nats). The ρ1−ρ2 null must be computed **in the same batch layout**. The student is trained on HQQ/NF4 dequant in the HF stack, so the importance ratio r_t has a small kernel mismatch: report the zero-adapter r_t distribution. |

### Prior art hit (new this session)
- **CREDIT** (2605.11613, abstract VERIFIED this session). It decomposes the OPSD token reward as pointwise MI and subtracts a **batch-contrastive teacher baseline**, i.e. the teacher conditioned on *other inputs'* privileged context, to remove input-generic credit. Evaluated on coding and tool use.
- **Contrastive SD** (2609.21561, abstract VERIFIED). Attractive (correct-solution teacher) minus repulsive (incorrect-solution teacher), so that shared behavioural shifts cancel.

SAST is precisely "contrastive OPSD where the negative context is a degree-preserving rewire". The mechanism (credit = teacher(true privileged context) − teacher(counterfactual privileged context)) is taken. The only delta is the **structure-specific negative** and its use as a measurement instrument.

### Verdict and what must change

**Verdict: SURVIVE-BY-SURPRISE.** Without the surprise rule it would be KILL (rule 3, plus a broken K1). It gets one more round only if all of the following change:

1. **Reposition.** Present it as *"CREDIT/Contrastive-SD with a structure-only negative: measuring and crediting edge use"*. Cite both, and make the **instrument** (pointwise edge-use per action) the headline. Training is secondary.
2. **De-leak the privileged block.**
   - Node set = the k-hop executed neighbourhood of the **failing test's entry** (k=3), with sibling branches that do *not* reach gold. Selection must be independent of the gold set.
   - Then δ^∅ cannot locate gold and only the wiring can.
   - Pre-register: the node-list teacher's AUROC for gold-revealing actions ≤0.6.
3. **Kill the coherence confound with two extra negatives** (prefill only, so cheap):
   - (i) a **plausible rewire**: swap only among callees whose names occur in the same caller file, so edges stay lexically plausible;
   - (ii) a **dose-response**: rewire fractions 0.25 / 0.5 / 1.0.

   Edge use predicts c rising monotonically with the rewired fraction *on actions whose path crosses rewired edges only*. Global discounting predicts c rising on all actions equally.
4. **Replace KS with effect sizes.** Kill if the per-trajectory Σ|c| on action tokens is <2× the 95th percentile of the ρ1−ρ2 null; or if c separates gold-revealing actions with AUROC <0.65 (CI from bootstrap over trajectories, n ≥ 40); or if δ^∅ explains ≥80% of δ^G variance after de-leaking.
5. **Budget, narrowed.**

   | Step | GPU-h |
   |---|---|
   | Probe | 1.0 |
   | 200 rollouts, shared by all arms (offline) | 2.5–5 |
   | 6 views (G, ρ1, ρ2, plausible-ρ, ∅, none) on 200 trajectories, ≈14M prefill | 1.3–2.6 |
   | 3 trained arms (outcome-only, OPSD δ^G, SAST c) | 3 × (1–1.7) = 3–5 |
   | Evaluation: base, OPSD, SAST on dev | 3 × (1.8–3.4) = 5.4–10 |
   | **Total** | **≈13–20** |

   The GF and RW *training* arms are dropped because the GF and RW teachers are measured in the instrument. Say so.

   Put the privileged block **after** the system prompt and issue, so that those prefixes are shared across views.
6. **This week (≤0.6 GPU-h, only if the probe is deferred):** 20 fresh rollouts plus 6-view scoring is the instrument kill test. Otherwise do it in week 2.

---

## R4. ALIAS: Observation-equivalent action marginal likelihood

### Scores

| Axis | Score / estimate |
|---|---|
| **GPU-h as specified** | Probe 1.0 · fresh rollouts for the kill test (X4) 0.2–0.4 · alias prefill 0.3. **Training:** true MML needs a forward pass for every alias *with its full prefix*. FSDP/HF has no prefix sharing at 16k with checkpointing, so naive cost is ≈30 actions × 5 aliases × ~8k tokens × 300 trajectories ≈ **360M tokens ≈ 150–250 h ✗**. The inventor's "<10% overhead with prefix sharing" is INFERRED and wrong for this stack. With 4-D tree masks, or an EM approximation (frozen-posterior weights from vLLM prefix-cached prefill, then sampled alias substitution), cost ≈ RFT. P2 then needs rollouts in a duplicate-tool environment, 2 arms × 2 tool surfaces of evaluation, and the X6 replay: **≈23–30 h** |
| Falsifiability | **1/3** |
| Surprise | **1.5/5** |
| Prior-art risk | **High** |

### Why falsifiability is only 1
- K1 and K2 test the premise, with clear thresholds ✓.
- The mechanism (splitting hurts RL) is tested only by P2, which needs training.
- P3 "<1 pp" is unfalsifiable (X3).
- The RW control (a rewired graph tool falls outside the alias classes) is an extractor sanity check, not a mechanism control.
- No GF arm separates "graph tools" from "any tool".

### Edge cases and premise problems
- **Relevance to graph reasoning is weak.**
  - v5 makes **0 graph calls** and v2 has 66 `search_similar_code` calls against 3,435 `run_command` (VERIFIED).
  - At J ≥ 0.8, a graph-tool observation and a grep observation rarely carry the same (file, symbol, line) fact sets.
  - So the classes will be dominated by **shell-syntax variants** (`grep -rn` vs `grep -nr`, `sed -n` vs `read_file`). Gemma is probably peaked on its habitual syntax, so K2 likely kills.
  - Even if it does not, the paper is about tool-surface invariance, not directed-graph reasoning.
- **Observation drift.** A substituted alias changes o_t, but the later context was conditioned on the original o_t. With J=0.8, 20% of the facts differ, which creates off-policy inconsistency the loss ignores.
- **Quantization.** Alias mass comes from long-sequence log-probs. Int4 noise is small relative to the ≥1-bit entropy claim. Not a blocker.

### Prior art hit
- Maximum marginal likelihood over spurious / equivalent programs (Guu et al. 2017, "Bridging RL and MML", SNIPPET).
- MAPO (Liang et al. 2018, INFERRED from memory).

Marginalising credit over observation-equivalent actions is the semantic-parsing MML mechanism applied to tools.

### Verdict and fixes
**Verdict: KILL** (rules 1 and 3, low relevance).

Keep the K1/K2 alias **census** (CPU plus 0.3 h prefill) as a diagnostic paragraph for the RepoNavigator anomaly (M3a vs M3b discrimination) in another paper.

---

## R5. SENSOR: tests as sensors (spectrum information gain)

### Scores

| Axis | Score / estimate |
|---|---|
| **GPU-h as specified** | Probe 1.0 · kill-test rollouts (X4: raw traces absent) 0.4–0.8 for 30 · rollouts 300 → 4–8 · training of 4 arms (outcome, SENSOR, GF-Σg, RW-Σg; rollouts shared) 6–10 · evaluation of 4 arms 7–14 → **≈25–35 h ✗** |
| **GPU-h narrowed** | Correlational phase: one fresh dev-size run (129 rollouts) 1.8–3.4. Then 200 shared rollouts 2.5–5, 3 arms 3–5, evaluation of 2 arms plus base 5.4–10, probe 1.0 → **≈15–20 h ✓** |
| Falsifiability | **1/3** |
| Surprise | **3/5**. "Information gain of the agent's own experiments predicts success where localisation does not, and can be trained" would redirect the "make agents test more" literature, which F2 says fails naively. |
| Prior-art risk | **Medium** |

### Why falsifiability is only 1
1. **Outcome leakage.**
   - Experiments run *after an edit* on a correct fix turn F2P tests from fail to pass. That shifts Ochiai mass and therefore g.
   - Σg then correlates with resolve **by construction**. K1 would pass for the wrong reason.
2. **Dose confound.** Σg grows with the number of experiments and with trajectory length. Neither is controlled, and F2 says #tests is neutral to negative.
3. **n=30 cannot test K1.** At ~20% resolve that is ~6 positives, giving an AUROC 95% CI of about ±0.2. The 0.05 margin is meaningless.
4. **The GF definition is circular.** "GF Σg = pass/fail outcomes + traceback names" still uses traceback frames, which are a graph path. It needs a strict node-only version: per-experiment pass/fail only, uniform over files.

### Edge cases

| Case | Failure |
|---|---|
| Sparse experiments | Only **40/129** r2 trajectories ran any test, and only **17** reproduced before editing (VERIFIED). Restricted to pre-edit (fix 1), g ≡ 0 on ~85% of trajectories. The token-credit signal is extremely sparse, so K3's 30% threshold sits right at the boundary. |
| New-code fixes | F* is not on the executed graph, so p_t(F*) has no support and log p is undefined. Drop and count them (≥7/129, plus feature additions). |
| Hubs | Ochiai down-weights them ✓. A uniform prior over all src functions makes log p(F*) dominated by pool size: normalise per task. |
| Flaky tests | A pass/fail flip without a code change poisons the spectra. Re-run and mark unstable tests. |
| Dynamic dispatch | Tracing handles it ✓. **Subprocess-launched code** (CLI tests, `python script.py`) needs `sitecustomize` tracing, otherwise its spectra are empty. |
| Quantization | None (CPU signal) ✓. |

### Prior art hit
- **FLINT** (Yoo, Harman, Clark, TOSEM 2013: information-theoretic, entropy-reduction test prioritisation for fault localisation; INFERRED from memory, unread).
- **RL test selection maximising fault-diagnosis capability** (Zhao et al., 2501.02216 / TOSEM, SNIPPET).

Using the SBFL-posterior entropy change as *token credit for an LLM agent's self-chosen experiments* is the new part.

### Verdict and fixes
**Verdict: NARROW.**

1. Compute g **only for experiments executed on the unedited (buggy) state**: before the first edit, or replayed on a reverted tree.
2. Report a partial AUROC controlling for #experiments and trajectory tokens (logistic regression with both covariates; K1 = coefficient of Σg with a 95% CI excluding 0 and ΔAUROC ≥0.05 over covariates-only).
3. Use n ≥ 129 (one dev-size rollout set).
4. Strict GF (pass/fail only).
5. RW: recompute Ochiai on spectra mapped through a degree-preserving rewire of G_exec, with node relabelling that preserves the coverage-count distribution.
6. Training waits until the correlational phase passes.

---

## R6. GCSR: Graph-compiled synthetic repos (SURVIVE-BY-SURPRISE)

### Scores

| Axis | Score / estimate |
|---|---|
| **GPU-h as specified** | Probe 1.0 · kill 0.6 · factorisation: 4 arms × 400 rollouts → (a) needs real-repo environments, 5–10 · (b), (c), (d) at a GUESSED 40–80/h → 15–30 · training 4 × (1–2) · evaluation of 5 runs (base + 4) × (1.8–3.4) = 9–17 → **≈35–65 h ✗** |
| **GPU-h narrowed** (below) | **≈16–21 h ✓** |
| Falsifiability | **1/3** |
| Surprise | **4/5**. "Synthetic, content-free, topology-controlled repos recover most of the real-SWE warm-start gain, and topology *fidelity* is irrelevant" would cut against the field's spend on real-repo environment synthesis (SWE-smith, R2E-Gym, Davinci-env). Its failure mode ("content dominates, and here is how much") is also informative. |
| Prior-art risk | **Medium** |

### Why falsifiability is only 1
1. **P2 ("(b) ≈ (c) within 1 pp") is a null result** that n=129 cannot confirm (X3, MDE ≈7–9 pp). As written it "passes" on noise.
2. **K1 is under-powered.** Spearman > −0.1 on 30 tasks (6 cells × 5) has SE ≈0.18, so it fires or not by chance.
3. **Arms differ in success rate.** Equal tokens is specified ✓, but RFT on own successes means harder arms yield fewer distinct successful trajectories. That confounds the arm with data diversity, and with difficulty via the curriculum.
4. **The RW arm is not a clean null.** A degree-preserving rewire of a DAG can create **cycles**. A value-flow compiler must reject or bound them, so (c) becomes an acyclicity-constrained rewire whose depth distribution shifts. Since P3 says depth is the active ingredient, (b) vs (c) confounds fidelity with depth.

### Edge cases

| Case | Failure |
|---|---|
| Cycles / recursion | As above. Real G_exec has recursion, so the compiled semantics need bounded recursion depth, or the tasks hang. |
| Dark edges | Registry, decorator and callback compiled with names drawn from real vocabularies "with no semantic hints". That removes exactly the content signal real repos have, so P1 (≥70%) is likely to fail. Pre-register that this is acceptable. |
| Hubs | Controllable ✓, but the hub ratio must match real G_exec (report a KS test on the degree distribution against requests/rich). |
| Huge graphs | 200–3,000 functions is fine. The ≤16k training cap limits trajectory length; synthetic trajectories are short ✓. |
| Weak tests | Root-only tests may not distinguish 2-node jointly-necessary bugs from partial fixes. Verify each multi-node bug: fixing any strict subset must still fail (an IFG-style necessity check, CPU). |
| New-code fixes | Absent by construction. That is a transfer gap: report real-dev performance split by "gold exists in the base graph". |
| Quantization | None ✓. |

### Prior art hit
- SWE-smith; BugPilot (2510.19898, SNIPPET); Synthetic repo-level bug dataset (ACM 3744916.3764562, SNIPPET); Davinci-env (2603.13023, SNIPPET, cited in 2608.09802); R2E-Gym / SWE-Gym (inherited).
- Synthetic graph-reasoning RL: GraphWiz and G1 (inherited).

None seen compiles **controlled topologies into executable repos** and factorises protocol from content. The red team should search "procedurally generated codebase" plus "transfer" specifically.

### Verdict and what must change

**Verdict: SURVIVE-BY-SURPRISE** (falsifiability and budget would otherwise kill it). One more round only if:

1. **Cut to three arms that share everything but topology:**
   - (b) real-copied topology;
   - (c) rewired topology, **depth-matched**: accept a rewire only if the depth and longest-path distribution stays within ±1 of the original;
   - (d) depth-1.

   Arm (a), real repos, is **borrowed from the main-line RFT run** if one happens anyway. Otherwise P1 is dropped and the paper is (b) vs (c) vs (d).
2. **Equalise** the number of distinct successful trajectories *and* tokens per arm. Subsample the easier arms down.
3. **Dense, high-n evaluation instead of resolve:**
   - multi-function localisation recall (`CANDIDATES` log score);
   - first-gold-step on ≥300 **held-out real-repo trace items** (CASE items from non-dev repos, prefill-cheap, ≈0.3 h per arm);
   - plus one dev resolve run for the best arm only.

   P2 becomes TOST with margin 0.05 on recall, powered by n ≥ 300.
4. **K1:** ≥90 zero-shot tasks (30 per depth). Kill if the logistic depth slope's 95% CI includes 0. Also verify joint necessity on CPU before any GPU spend.
5. **Budget, narrowed.**

   | Step | GPU-h |
   |---|---|
   | Probe | 1.0 |
   | Kill (90 short tasks at ~8 parallel streams) | 1.0–1.5 |
   | 3 arms × 200 rollouts, synthetic and short | 3 × (2.5–5) = 7.5–15, **must measure first; if >4 h/arm, cut to 120** |
   | 3 × training | 3 × (0.6–1) |
   | Item-level evaluation | 3 × 0.3 |
   | 1 dev run | 1.8–3.4 |
   | **Total** | **≈16–21** |

   It fits only if synthetic rollouts reach ≥60/GPU-h. Measure that in the kill test and abort the factorisation if the rate is <40/h.

---

## Recommended use of the 1.5 GPU-h this week

1. **CPU, all week:**
   - GCSR generator v0 plus CPU necessity checks;
   - CASE trace items **on non-dev repos**, plus a separate zero-shot set on requests/rich;
   - the SAST de-leaked neighbourhood builder;
   - the SENSOR pre-edit replay harness.
2. **GPU, one session:** the **training probe (1.0 h)**, because it gates every idea. Then **either** the CASE L / A / EOS-suppressed zero-shot (≈0.4 h, prefill-heavy, no rollouts), **or** a 30-task GCSR throughput-and-depth pilot (≈0.5 h). That pilot only calibrates rollouts/GPU-h, which decides whether GCSR's narrowed budget is real.
3. **Not this week:** SAST and SENSOR, which need fresh rollouts (X4).

---

## Search log

| # | Query / action | Tool | Result relevant to judging |
|---|---|---|---|
| 1 | "LLM predict which tests fail after mutation per-test kill matrix prediction Seshat predictive mutation analysis" | firecrawl web | Seshat (kill matrix, ACM 3510417); Contextual PMT (Jain/Le Goues); fine-grained PMT 2511.11999; PMT pitfalls (Springer 2026). → ERC prior art (SNIPPET) |
| 2 | "pointwise vs listwise LLM recall set-valued answers under-generation …" | firecrawl web | IR pointwise/listwise/setwise rerankers (SIGIR'24 setwise; EMNLP'24 FIRST). → CASE format effect is IR folklore (SNIPPET) |
| 3 | "self-distillation privileged context counterfactual corrupted context baseline token credit contrast teacher …" | firecrawl web | awesome-on-policy-distillation list: CREDIT 2605.11613, Contrastive SD 2609.21561, CSCR 2607.27888, Privileged-but-Biased 2608.04794, OP2SD 2608.09228. → SAST mechanism prior art |
| 4 | Scrape arXiv 2605.11613 (CREDIT) | firecrawl scrape | Abstract VERIFIED: OPSD reward = pMI; batch-contrastive teacher baseline isolates input-specific credit; coding and tool-use benchmarks |
| 5 | Scrape arXiv 2609.21561 (Repulsive/Attractive teachers) | firecrawl scrape | Abstract VERIFIED: contrastive SD = attraction to the correct-solution teacher minus repulsion from the incorrect one; shared behavioural shifts cancel |
| 6 | "procedurally generated synthetic Python repositories from graph for training SWE agents RL bug injection" | firecrawl web | BugPilot 2510.19898; Synthetic repo-level bug dataset (ACM); SWE-Gym. No topology-compiled repos (absence only) |
| 7 | "marginal likelihood over equivalent tool calls agent training credit splitting redundant tools …" | firecrawl web | Guu et al. 2017 "Bridging RL and MML"; ToolRL. → ALIAS mechanism = MML (SNIPPET) |
| 8 | "reward agent experiments information gain spectrum-based fault localization Ochiai RL LLM debugging …" | firecrawl web | LLM4FL (TMLR); "Automatically Learning a Precise Measurement for Fault Diagnosis Capability of Test Cases" (TOSEM 3712189, RL test selection). → SENSOR prior art |
| 9 | Exact title search for #8 | firecrawl web | arXiv 2501.02216 confirmed (SNIPPET) |
| 10 | "LLM predictive mutation testing fine-tuned predict mutant killed by test case 2025" | firecrawl web | LLM equivalent-mutant detection (ISSTA'24), PRIMG, Meta's LLM mutation testing, PMT repo (SNIPPET) |
| 11 | "synthetic codebase generator controllable call graph depth train LLM agent navigation transfer …" | firecrawl web | Codebase-Memory, GraphCodeAgent, LocAgent. No controlled-topology generator (absence only) |
| 12 | OpenReview search API: "executable call graph rewire counterfactual"; "atomized set prediction recall list LLM"; "synthetic repositories graph compiled SWE agent"; "observation equivalent actions marginal likelihood agent" (2 s pacing) | curl api2.openreview.net | Only EICR (counterfactual reasoning, unrelated) and SWE-bench Pro reviews were relevant. **Note:** `notes?id=` lookups now return a 403 ChallengeRequiredError, so most forum titles could not be resolved; search results are mostly review notes |
| 13 | "arxiv 2026 procedurally generated repositories synthetic codebases multi-hop bug localization training agents …" | firecrawl web | Active-SWE 2608.04682; Davinci-env 2603.13023 (via 2608.09802); awesome-code-agents issue listing **2606.11976 "Exploration Structure in LLM Agents for Multi-File Change Localization"** (CASE neighbour, unread) |
| 14 | "LLM fault localization multi-location fixes under-reporting recall per-function binary classification vs list generation …" | firecrawl web | RGFL 2601.18044; "On the Role of FL Context for LLM-based Program Repair". Nothing on emission-termination (absence only) |

**Local computations (VERIFIED arithmetic, this session):**
- Dev repo counts (fastapi 67 / rich 48 / requests 13 / httpx 1).
- Any test run 40/129; repro before edit 17/129.
- Max prompt >14k in 65/129 and >16k in 6/129.
- Median task duration 93 s.
- Gold-function buckets 0/1/2/≥3 = 7/57/20/45.

Source files: `results/runs/r2_v2_sed/per_task.jsonl`, `task_results.jsonl`, `summary.json`.

**Absent locally:** raw r2/r3 trajectories.

**Not done (budget):** full reads of CREDIT, 2606.11976, Contextual PMT and FLINT. The red team should read CREDIT §method to check whether it already tries structured or graph negatives.
