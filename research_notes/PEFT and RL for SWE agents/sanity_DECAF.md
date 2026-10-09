# Sanity check: DECAF against the current literature (8 Oct 2026)

Independent reviewer pass. Scope: does DECAF (critic_judgement.md R4.1–R4.4; report §"DECAF moves dependency credit offline") address a real, open field gap (G2), and what did it miss? Judged against the literature, not our notebook results.

Labels: **VERIFIED** = I read the relevant section of the primary text this session. **SNIPPET** = abstract, search summary or listing only. **INFERRED** = my reasoning, no direct source.

**Verdict in one paragraph.** G2 *as worded* ("offline methods only weight whole trajectories; nobody joins step credit and offline training for SWE") is **false**. Orchard-SWE already does offline, step-level-credited SFT on SWE trajectories (LLM-judged TD values, rise-segment selection on 32k unresolved trajectories, +1.9 pp on SWE-bench Verified). SWE-Prime (LLM segment scores) and SWE-Lego/Nebius (deterministic error-turn masks) already put step-level weights into offline SWE SFT. ABSeeker's ABC-SFT already does "trace backward from the verifiable answer, reweight each turn's SFT loss" (search agents). A **narrower gap survives**: *deterministic, execution-derived (dependency/coverage) step credit used offline for SWE agents, validated against counterfactual ground truth*. Two conceptual problems are more serious than the prior-art ones. (1) Pure dependency credit gives **zero credit to the final verification test run**, so DECAF's objective pushes *against* its own headline metric "tested-after-last-edit". (2) "Coverage, Not Targeting" and "Credit Without Ground Truth" show that targeted step credit under terminal verifiers is often no better than a **matched-concentration shuffled control**, and that credit comparisons not matched on effective sample size measure dose, not credit. Tier B lacks both controls. Its process metrics are also confounded by λ_z and DART, which target the same metrics directly. NEW-2 (coverage-resolved E\*) is likely to be marginal, because DepGPO already tracks **line-level writes**, and E vs E\* differ only on edits that survive into the final state but are not executed. **Recommended change:** turn DECAF into a *credit-audit + offline-credit* study. Use hunk-ablation (interventional) certification instead of coverage as the write-credit signal, text-provenance reads, an explicit verification-step class, and arms {uniform, shuffled-matched, LLM-TD (Orchard-style), DepGPO-file, DECAF} at matched ESS.

---

## 1. Prior art the previous passes did not find

### Takeaway
G2 is **dead as stated** and **survives only narrowed**. Offline step-level credit already exists for SWE (Orchard: LLM-TD; SWE-Prime: LLM segment scores; SWE-Lego/Nebius: deterministic error masks) and for agents generally (IPR, Q-SFT, Digi-Q, SWEET-RL, ABSeeker ABC-SFT, Agentic-DPO). No work found that uses *deterministic dependency/coverage* credit offline for SWE, so that narrowed junction is still open. DepGPO is six days old and has no visible citers yet. No OpenReview entry matched.

### Cited Findings

**SWE work that already uses step-level weights offline (kills G2's wording)**
- **Orchard-SWE (2605.15040), §3.3 "Off-policy Learning: SFT with credit assignment."** Per-step credit is c_t = V(s_{t+1}) − V(s_t), "a lightweight LLM-based variant of temporal-difference value estimation". V comes from the teacher, which is shown "the full trajectory together with the gold test outcome". Sparse key steps are annotated and the rest interpolated. For unresolved trajectories, the loss is restricted to "action tokens inside the extracted rise segments" (c_t ≥ 0.05), with history kept as context. 32,536 unresolved trajectories are used. **VERIFIED** — [Orchard HTML](https://arxiv.org/html/2605.15040v3)
- Orchard ablation: resolved-only SFT 59.3% → + credit-assignment SFT 61.2% on SWE-bench Verified (+1.9). The extra 32K rise-segment trajectories are *added*, so data dose is confounded with credit, and there is no "unresolved without credit" arm. Backbone is Qwen3.5-35B-A3B. **VERIFIED** — [Orchard HTML §3.8](https://arxiv.org/html/2605.15040v3); [abstract](https://arxiv.org/abs/2605.15040)
- SWE-Prime: an LLM scores 20-step segments against the gold patch, and low-score segments are masked from the loss (threshold 7/10). This is offline step(segment)-level credit for SWE SFT. **VERIFIED in sibling note** (`rl_and_offline_substitutes.md` l.134, [SWE-Prime](https://www.alphaxiv.org/abs/2608.27449)); not re-read here.
- SWE-Lego step-level error masking (deterministic regex on environment errors; +2.0–2.2 pp) and Nebius error-turn masking in RFT. These are **deterministic, offline, step-level** weights for SWE, but only binary and only negative. **VERIFIED in sibling notes** (`sft_data_recipes.md` l.186–189; `rl_and_offline_substitutes.md` l.124)
- FLARE (2609.23808, SWE): RADAR runs an *offline* "reverse causal error chain analysis … via backward tracing" from global signals ("patch differences and failed test cases") to step-level labels. The labels train a GRM. In post-training, the GRM gives "process-supervised reranking scores for SFT" (trajectory selection: top 3,000 of 18,000) and "step-level dense rewards for RL". Reported +19.13% relative in SFT and +9.19% in RL. It is the **LLM-judged analogue of DECAF's backward trace**, on SWE, but its SFT use is trajectory-level selection. **VERIFIED** — [FLARE](https://www.alphaxiv.org/abs/2609.23808)
- A²Agent: advantage-weighted MLE with turn-level rewards (LoRA r16). The sibling note says the weights "can be computed from saved rollouts" but elsewhere lists it under "online" add-ons. Whether it is a third offline step-credit SWE instance is **unresolved** (sibling note, not re-checked) — [A²Agent](https://www.alphaxiv.org/abs/2608.29831)

**Offline step-credit for LLM agents in general (the base rate DECAF must cite)**
- IPR, "Watch Every Step!" (EMNLP 2024): Monte-Carlo step rewards; "iteratively refines agent policy through offline optimization on contrastive action pairs". Covers WebShop, ALFWorld and InterCode-SQL. This is *iterated-batch, offline, step-level* for agents. **SNIPPET** — [arXiv 2406.11176](https://arxiv.org/abs/2406.11176); [ACL Anthology](https://aclanthology.org/2024.emnlp-main.93)
- Q-SFT (offline Q-learning via an SFT objective, multi-turn) and Digi-Q (learned Q-functions for device-control agents trained offline). **SNIPPET** (listing/abstract) — [Q-SFT](https://www.alphaxiv.org/abs/2411.05193); [Digi-Q](https://www.alphaxiv.org/abs/2502.15760)
- SWEET-RL (Meta, 2025): a critic with training-time privileged information (reference solutions) gives step-level rewards for multi-turn collaborative *coding* and design (ColBench). +6% absolute over other multi-turn RL. I could not confirm from the abstract whether training is offline. **SNIPPET** — [arXiv 2503.15478](https://arxiv.org/abs/2503.15478)
- ABSeeker / ABC (2608.05102): "Answer-Backtracked Clue Recovery … traces back from the answer to recover intermediate clues", then "ABC-SFT, which reweights the loss of each turn". Search agents, Qwen3.5-4B. This is structurally the same as DECAF's "backward trace from verifier evidence → per-step SFT weights", in another domain. **SNIPPET** (abstract) — [ABSeeker](https://arxiv.org/abs/2608.05102)
- Agentic-DPO (step-level, expert-anchored, ≈ GRPO on τ-bench), already cited by DECAF. **VERIFIED in sibling note**

**Deterministic/provenance step credit (all online, which confirms the other half of G2)**
- DepGPO (2610.03634): the endpoint E is file-level, but **write tracking is already line-level**. Lines come from per-command diff hunks, and Contrib(c) = fraction of c's written lines reaching E. Read credit needs a *stdout edge*, i.e. a typed value printed by the read and textually reused later: q(r) = Σ β^{d} Contrib(c). Ownership is "not rebased after later insertions or deletions". Online GRPO; 3.8 s/trajectory; 61% of 1,376 groups have identical rewards. **VERIFIED** — [DepGPO full text §3.2–3.3, App. A.3](https://www.alphaxiv.org/abs/2610.03634)
- STAMP (2607.11172, deep search): "first-exposure attribution traces each supported citation back to the action that first surfaced it", using sign-preserving advantage modulation in GRPO. +2.0/+5.5/+3.0 on BrowseComp/-ZH/xbench-DS. Online. **SNIPPET** — [STAMP](https://arxiv.org/abs/2607.11172)
- EGCA (ICLR 2026 workshop, single-turn code): executes candidate and reference "under identical instrumentation", and localises GRPO advantage to the earliest trace divergence. +3.1 HumanEval, +1.5 MBPP. This is execution-trace credit for code, but online and single-turn. **SNIPPET** — [EGCA](https://arxiv.org/abs/2603.16158)
- Also online: RTMC (SWE-bench, +3.2 over GRPO, rollout trees), Groupwise Agentic Grading for code agents (2609.32577), SWE-MILE, VICT, AgentLTL, CRR, SHARPO, GraphHCA, AdaStep, TRACE. **SNIPPET** — [RTMC](https://www.alphaxiv.org/abs/2604.11037); [2609.32577](https://www.alphaxiv.org/abs/2609.32577)
- SCAD (2610.03372) mixes cross-rollout prefix-tree credit with teacher distillation. Search/multimodal, not SWE. **VERIFIED** (abstract + intro) — [SCAD](https://www.alphaxiv.org/abs/2610.03372)

**Validity audits of step credit (directly relevant to Tier A/B design)**
- "Credit Without Ground Truth" (2608.19760, ALFWorld) audits LLM-judge, implicit log-prob and confidence credit against *executed replay*. "None of the three signals shows reliable fidelity beyond its marginal-matched shuffled control." Only 30.5% of decision points have a non-zero replay contrast. In a 7-arm pre-registered training study, no arm reliably beats the untrained policy. Quote: "Comparisons of credit rules must match effective sample size, or they measure dose, not credit." **SNIPPET** (abstract) — [arXiv 2608.19760](https://arxiv.org/abs/2608.19760)
- "Coverage, Not Targeting" (2609.02417) defines verifier information density V_d = k/C. Terminal verifiers sit at low V_d: ≈0.15 on τ²-bench, with k=1 in 98% of rollouts. There, concentrating advantage on progress turns ≈ random turns, and uniform credit is the bar to beat. A matched-concentration shuffled control is proposed as the mandatory check. **SNIPPET** (abstract) — [arXiv 2609.02417](https://arxiv.org/abs/2609.02417)

**OpenReview / citation graph**
- No OpenReview (ICLR 2026/2027, NeurIPS 2026) entry matched "dependency credit + offline fine-tuning". The only openreview.net hit was an unrelated ICLR 2026 submission on offline vs online fine-tuning. **SNIPPET** — [search result](https://openreview.net/pdf/3a031437412c255f006f1be528dde5e805d12892.pdf)

### Inferences
- G2 must be restated as: *"Offline step credit for SWE exists only as LLM-judged values (Orchard, SWE-Prime, FLARE) or binary error masks (SWE-Lego, Nebius). Deterministic execution/dependency credit exists only online (DepGPO, VICT, STAMP, EGCA). Nobody has used execution-derived credit offline for SWE, or compared it against LLM-judged offline credit."* **INFERRED**
- Orchard is now the closest prior art (closeness ≈2.5/3): same setting (SWE, offline SFT, step-level credit, unresolved trajectories), different credit source. A reviewer will ask "why not Orchard-style LLM-TD with your own model?" DECAF must include that arm. **INFERRED**
- ABSeeker ABC-SFT means the *mechanism* ("backward trace from verifier evidence → per-turn SFT weights") is not new in general. Novelty rests on the deterministic execution-derived source in SWE. **INFERRED**

### Gaps
- A DepGPO citation graph is not yet meaningful (posted 2 Oct 2026). Semantic Scholar was not queried. A concurrent "offline DepGPO" cannot be excluded.
- Full texts of SWEET-RL, Credit Without Ground Truth, Coverage-Not-Targeting and ABSeeker were not read. Their training setup (offline vs online) is from abstracts.
- Whether A²Agent's turn-level advantage-weighted MLE is offline was not checked.

---

## 2. Conceptual soundness of coverage-resolved dependency credit

### Takeaway
Write credit from "surviving lines executed by F2P tests" is sound for *localised, single-site edits on resolved trajectories*. It is systematically wrong in five situations that are common in SWE: verification steps, non-reused informative reads, negative information, omission-type fixes on failed runs, and multi-location fixes. NEW-2's marginal value over DepGPO is likely small. Importance weights here mostly correct a *quantisation/kernel mismatch*, not policy drift, and can be dropped or must be measured.

### Cited Findings
- DepGPO's read credit exists only when the read's output is *textually reused* (stdout edge) on a path to a positive-credit write; "a read with no path to any positive-credit write receives zero". **VERIFIED** — [DepGPO §3.3](https://www.alphaxiv.org/abs/2610.03634)
- DepGPO's own appendix says "long terminal trajectories … often contain exploration, discarded edits, repeated inspections, and failed intermediate attempts", and that dependency credit separates these from commands "connected to the verified state". By design, exploration that does not feed writes gets no credit. **VERIFIED** — [DepGPO App. F](https://www.alphaxiv.org/abs/2610.03634)
- Terminal verifiers expose a small fraction of the causal chain (V_d ≈ 0.15–0.4; k=1 in 98% of rollouts), and targeted credit is "second-order" there. **SNIPPET** — [2609.02417](https://arxiv.org/abs/2609.02417)
- Coverage-based fault localisation is known to degrade on code-omission faults, and coverage spectra are "over-inclusive": executed code that does not affect the output is noise. **SNIPPET** — [SBFL survey 1607.04347](https://alphaxiv.org/abs/1607.04347); [MCFL (missing code)](https://research.buaa.edu.cn/en/publications/mcfl-improving-fault-localization-by-differentiating-missing-code/); [A Case Against Coverage-Based Program Spectra](https://www.inf.u-szeged.hu/~beszedes/research/SGH23.html)
- Rollout (vLLM) and training (FSDP) stacks give "significantly different token probabilities" even with identical weights, which turns on-policy into off-policy. Truncated IS is the standard fix. Token-level IS bias grows with length; sequence-level IS variance grows exponentially. Quantised rollouts make it worse (QaRL). **SNIPPET** — [Rollout-Training Mismatch, NeurIPS'25](https://neurips.cc/virtual/2025/126728); [Yao et al. blog](https://fengyao.notion.site/off-policy-rl); [Li & Liu blog](https://yingru.notion.site/Part-1-Why-Mismatch-Breaks-LLM-RL-295211a558b780b8bb4cc8e367d906d0); [QaRL 2604.07853](https://arxiv.org/pdf/2604.07853)

### Inferences (each a concrete failure case)
1. **Verification steps get zero credit, which contradicts the headline metric.** A `pytest` run after the last edit is a read whose output feeds no later write, so its raw credit is 0. With unit-mean step factors it is *down-weighted* relative to edits. DECAF claims "+10 pp tested-after-last-edit", but its credit rule pushes the other way. The submit/finish step has the same problem. Fix: a deterministic "verification" class (test run after the last mutating action, on resolved trajectories) with credit ≥ mean. **INFERRED**
2. **Informative reads with no textual reuse:** reading the issue's traceback, the existing tests, docs, or a caller to understand a contract. Line-span overlap with E\* catches reads of *executed* source lines, but not test files (unless they are in the F2P set), docs or error outputs. Credit is biased toward "read the line you then edited", which rewards shallow localisation behaviour. **INFERRED**
3. **Negative information:** a read or run that rules out a hypothesis ("bug is not in X") is causally useful but structurally invisible to dependency credit. Under `Σ_{w_t>0}` it is trained less, so the model learns to explore less. That may be good (fewer zero-information actions) or bad (premature commitment). The proposal does not say which or measure it. **INFERRED**
4. **Fixes that add code:** coverage on the *final* commit covers new lines, so resolved runs are fine. On **failed** runs, a missing guard or branch is an omission. The executed set on the final state does not point to where code should have been, so "partial-F2P usable credit" (KA2) is weakest exactly where the failure lies. **INFERRED** (SBFL omission literature above)
5. **Multi-location fixes:** per-line credit is additive. A hunk necessary only jointly with another is credited the same as a redundant one. The IFG joint-necessity work already in this project is the right tool here, not coverage. **INFERRED**
6. **NEW-2 is probably marginal.** DepGPO already tracks written *lines*. E (files the verifier reads) vs E\* (lines executed) differ only for lines that (a) survive to the end, (b) lie in an imported file, and (c) are not executed by F2P tests. Repro scripts, scratch files and reverted edits get zero under both. On resolved trajectories, surviving lines are mostly the patch itself, which F2P executes. KA1 may well fire. **This is checkable on existing traces in minutes of CPU, before any GPU batch:** fraction of final-diff lines not covered by F2P. **INFERRED**
7. **Importance weights.** In round 1 the learner initialises at the behaviour policy, so ρ should be ≈1. But behaviour log-probs come from **W4A16 vLLM** while training runs on an **NF4/bf16 QLoRA** stack, so ρ ≠ 1 at θ = θ_b. The IS term would then mostly inject quantisation and kernel noise. Over 30–60-turn trajectories, any non-token-level product explodes. Recommendation: measure ρ at initialisation on the training stack (free diagnostic). If its spread is comparable to the policy drift expected after one epoch, drop IS in round 1 and use token-level TIS (cap ≈2) in round 2 only. **INFERRED** (mismatch literature above)
8. **Credit is defined against the hidden F2P tests**, which the agent never saw. That is fine for training labels (privileged info, as in Orchard/SWEET-RL). But F2P tests are narrow, so credit inherits their blind spots (V_d is low). **INFERRED**

### Gaps
- No published measurement of how often SWE agent final diffs contain imported-but-unexecuted lines. This is the quantity that decides NEW-2.

---

## 3. Do Tier A and Tier B actually test the claim?

### Takeaway
Tier A measures *agreement with gold-patch functions*, a proxy for correctness, not *contribution*. Tier B's primary metrics are confounded by λ_z and DART, and it lacks the two controls the recent credit-audit literature treats as mandatory: a **matched-concentration shuffled credit** arm and **effective-sample-size matching**. As designed, a Tier B "pass" would not show that dependency credit caused anything.

### Cited Findings
- Credit audits should use executed replay (contribution), not correctness labels. Credit rules should be compared with marginal-matched shuffled controls and at matched effective sample size. **SNIPPET** — [2608.19760](https://arxiv.org/abs/2608.19760)
- Matched-concentration shuffled control is negative on 8/8 seeds on BFCL; random-turn concentration ≈ progress-turn concentration. **SNIPPET** — [2609.02417](https://arxiv.org/abs/2609.02417)
- Orchard's +1.9 compares runs that differ in data dose (32K vs 64K trajectories), the same confound. **VERIFIED** — [Orchard §3.8](https://arxiv.org/html/2605.15040v3)

### Inferences — what to change
**Tier A (credit validity)**
- Add a **contribution ground truth** that is cheap in SWE because sandboxes reset deterministically:
  - *Writes:* hunk ablation, i.e. revert each surviving hunk (and pairs, for multi-location fixes) and rerun F2P on CPU. This is interventional necessity and reuses IFG Stage A.
  - *Reads:* small executed replay on a subsample (~100 decision points). Blank the read's observation, resample K=4 next actions from the same policy, and compare outcomes.
- Score **every** credit source against this ground truth: uniform, shuffled-matched, DepGPO-file, DECAF E\*, and LLM-TD (Orchard-style, using Gemma itself or the teacher with the gold outcome). Gold-patch-function precision stays as a secondary metric.
- Replace the KA1 threshold logic. The question becomes "does E\* beat E *and* shuffled on contribution-AUROC", not "is E within 5 pp".
- Run the zero-GPU pre-check (§2.6) first.

**Tier B (method)**
- Add **arm A′ = A + λ_z + DART + anchor (no step credit)**. Then B − A′ isolates credit. Today KB1 (B vs A) mostly measures λ_z/DART on NPR. KB3 gestures at this but is secondary.
- Add **arm S = shuffled-matched credit** (same per-trajectory weight histogram, permuted over steps) and **arm L = LLM-TD credit** (the Orchard analogue). Without S, any B > A′ result can be explained by concentration or dose.
- Match **effective sample size** (Σw)²/Σw² or effective token count, not raw tokens.
- Add **"tested-after-last-edit"** as a check that the credit rule does not *reduce* it (§2.1).
- **Power:** 40 tasks with ~5 arms is too few for resolve; process metrics at n=40 are fine only for 1–2 pre-registered contrasts (B vs A′, B vs S). Apply Holm correction across the 3 process metrics. Use ≥2 rollout seeds per task in eval to reduce per-task variance. Report effects as paired per-task differences.
- **Mechanism metric that does not depend on λ_z:** credited-step NLL shift on held-out *own* trajectories. Does training raise likelihood of contribution-positive actions more than shuffled-credit training does?

**Missing baselines, as a list:** trajectory-weighted SFT (have: A); RFT (have: C); **A′ (no-credit with λ_z/DART)**; **shuffled-matched credit**; **LLM-judge step credit (Orchard-TD / SWE-Prime segments / FLARE-style backward chain)**; **SWE-Lego/Nebius error-turn masking** (deterministic, trivial baseline); **DepGPO file-level credit offline** (Tier A only today; add to Tier B if KA1 is ambiguous); random-step credit is subsumed by shuffled-matched.

### Gaps
- No public SWE number for how often step credit beats a shuffled control. That is exactly the analysis contribution DECAF could make.

---

## 4. Generality beyond this competition

### Takeaway
The general idea, offline credit from *verifier provenance*, transfers. Coverage is the SWE/Python instantiation. Coverage tooling exists for all major languages, so cost is not the blocker. What blocks transfer is (a) line-level read logging, which depends on the scaffold, and (b) low verifier density in non-code domains.

### Cited Findings
- Provenance-style credit already exists online in terminal (DepGPO: files/ports/processes), search (STAMP: citation first-exposure; ABSeeker: answer-backtracked clues) and embodied/web (VICT: verifier atoms). **VERIFIED/SNIPPET** as above.
- DepGPO's tracer is limited by path aliasing (symlinks unresolved), no ownership rebasing after line shifts, and whole-file fallbacks for binary or oversized writes. **VERIFIED** — [DepGPO App. A](https://www.alphaxiv.org/abs/2610.03634)

### Inferences
- **Other languages:** line coverage is mature for Java (JaCoCo), JS/TS (Istanbul/c8), Go (`go test -cover`), C/C++ (gcov/llvm-cov) and Rust (llvm-cov). The costs are instrumented builds for compiled languages and flaky coverage for multi-process tests. Multi-language SWE benchmarks would make a good generality check. **INFERRED** (tool availability from general knowledge, not searched)
- **Line-level reads** depend on the agent viewing explicit ranges (`sed -n`, `str_replace_editor view [a,b]`). `cat` of whole files or grep with context makes read spans coarse, so read credit falls back toward file level. Our scaffold assumption is already GUESSED in the report (E1). **INFERRED**
- **Non-code agents:** no coverage analogue exists. The portable part is "dependency credit computed offline into weighted SFT" (NEW-1), which in search is already ABC-SFT. So DECAF's generality claim should be "execution-derived credit for agents whose verifier is executable code", not "agents in general". **INFERRED**

### Gaps
- No evidence on how the coverage-resolved read set behaves for non-Python test suites (e.g. Java integration tests that load whole frameworks).

---

## 5. Verdict: what was missed, does the gap survive, what must change

### Takeaway
DECAF missed Orchard (offline step-credit SWE SFT), FLARE (offline backward-causal step labels on SWE), ABSeeker ABC-SFT (backward-traced per-turn SFT weights), IPR/Q-SFT/Digi-Q/SWEET-RL (offline step credit for agents), and the two credit-audit papers that set the control standard. G2 survives only narrowed to *deterministic execution-derived* offline credit for SWE. The strongest version is an **audit-first** paper: which step-credit sources actually track contribution in SWE trajectories, and does training on them beat shuffled credit at matched dose? DECAF's machinery becomes one arm.

### Cited Findings
- See §1–§3. Key new citations: [Orchard 2605.15040](https://arxiv.org/html/2605.15040v3), [FLARE 2609.23808](https://www.alphaxiv.org/abs/2609.23808), [ABSeeker 2608.05102](https://arxiv.org/abs/2608.05102), [Credit Without Ground Truth 2608.19760](https://arxiv.org/abs/2608.19760), [Coverage, Not Targeting 2609.02417](https://arxiv.org/abs/2609.02417), [IPR 2406.11176](https://arxiv.org/abs/2406.11176), [SWEET-RL 2503.15478](https://arxiv.org/abs/2503.15478), [Q-SFT](https://www.alphaxiv.org/abs/2411.05193), [Digi-Q](https://www.alphaxiv.org/abs/2502.15760), [STAMP 2607.11172](https://arxiv.org/abs/2607.11172), [EGCA 2603.16158](https://arxiv.org/abs/2603.16158), [RTMC 2604.11037](https://www.alphaxiv.org/abs/2604.11037), [rollout-training mismatch](https://neurips.cc/virtual/2025/126728).

### Inferences — concrete changes (ordered by importance)
1. **Rewrite G2 and NEW-1.** Use "offline *execution-derived* step credit for SWE vs offline *LLM-judged* step credit (Orchard/SWE-Prime/FLARE)". Cite Orchard as the nearest offline SWE work and ABSeeker as the nearest mechanism. Delete "offline methods only weight whole trajectories".
2. **Fix the verification-step contradiction.** Add a deterministic verification class (test run after the last mutating step, on R=1 trajectories) and a "reproduction" class (a run that reveals the failing traceback whose identifiers appear in later edits).
3. **Swap the core certificate from coverage to intervention for writes.** Hunk ablation with rerun F2P handles multi-location (joint) necessity and over-inclusive coverage, and it reuses IFG. Keep coverage only as a cheap prefilter. For reads, use **text provenance**: n-gram/identifier overlap between an observation and later write content or commands. This generalises DepGPO's typed-value edges and covers tracebacks, tests and docs.
4. **Make Tier A a contribution audit** against hunk ablation and replay, with a shuffled-matched control and an LLM-TD arm. That is a publishable *analysis* section even if Tier B never runs, and it scores on the rubric's "rigorous analysis of existing methods".
5. **Tier B arms:** A′ (no credit, same λ_z/DART), S (shuffled-matched), L (LLM-TD), B (DECAF), all at matched ESS. Pre-register B vs A′ and B vs S only.
6. **Drop or measure IS** (§2.7). Treat round 1 as weighted BC.
7. **Run the zero-GPU pre-check** of the E vs E\* difference on existing traces before spending the 1.5 GPU-h batch.

**Stronger design variant: "CERT-SFT" (counterfactual-certified step SFT).**
- Same offline loop, with credit from three deterministic certificates:
  - (i) hunk-ablation necessity for writes;
  - (ii) text-provenance for reads;
  - (iii) an explicit verification class.
- The headline claim becomes falsifiable against the field's new standard: *execution-certified credit beats shuffled-matched and LLM-TD credit on contribution-AUROC (Tier A), and on paired process metrics at matched ESS (Tier B)*.
- It is cheaper than DECAF: no line-level read-span logging is required, and it reuses IFG Stage A.
- It is also the only variant where a negative result is informative for the field. "Execution-certified credit ≈ shuffled credit on SWE" would extend Coverage-Not-Targeting's low-V_d finding to SWE.

**Does the gap survive?** Yes, narrowed, at about 1–1.5/3 novelty for the method and about 2/3 for the audit. The novelty is the *source* of the credit and the *audit*, not "offline step credit".

### Gaps
- Not verified: whether any ICLR 2027 submission (deadline ~late Sep 2026, not yet public) does offline DepGPO.
- Not checked: SWE-Gym, Comet-9B, P2T and SWE-Lego full texts for step weights beyond the error masking already known. Sibling notes indicate trajectory-level selection only for SWE-Gym/P2T/Comet.

---

## Search log

| # | Tool | Query / target | Outcome used |
|---|---|---|---|
| L1 | Local | `reports/PEFT and RL for SWE agents.md` (DECAF sections), `critic_judgement.md` R4.1–R4.4 + Round-4 search log, `rl_and_offline_substitutes.md`, `sft_data_recipes.md` (grep) | Proposal, prior searches Q27–Q38, SWE-Prime/SWE-Lego/Nebius masking facts |
| S1 | alphaXiv discover | "offline training of LLM agents with step-level credit … PRM-weighted SFT, step-level AWR, critic-weighted BC, Q-SFT, Digi-Q, offline turn-level DPO/KTO" | Q-SFT, Digi-Q, ArCHer, TRACE, VETTA, Agentic-DPO, 2607.05458, trajectory-graph advantages, FLARE, Critic Experience Bank, Function-Level Execution Feedback, SHARPO, **SCAD** |
| S2 | alphaXiv discover (recency) | "step-level credit assignment for SWE/coding agents using dependency tracing, data flow, program slicing, test coverage; incl. offline/SFT" | DepGPO, SWE-MILE, Groupwise Agentic Grading 2609.32577, GraphHCA, Coverage-Not-Targeting, SCAD, My FAULT, pivotal decisions, Fail-Fast, multi-harness RL, CREDO, SWE-Prime |
| F1 | alphaXiv full text | SCAD 2610.03372 | Prefix-tree credit + distillation; search/multimodal; not SWE |
| F2 | alphaXiv full text | FLARE 2609.23808 | Offline backward causal-chain step labels on SWE → GRM; SFT = trajectory selection; RL = step rewards |
| S3 | alphaXiv discover | "execution trace / test coverage as process supervision or step-level reward for code; offline preference with execution feedback" | 2608.23632, ExecuCritic, ExecCritic, 2608.31077, early reward prediction; no coverage-based offline step credit |
| S4 | alphaXiv discover (recency) | "offline step-/turn-weighted SFT for multi-turn agents from self-rollouts" | AdaStep, SHARPO, GraphHCA, TRACE, ABSeeker, AHEAD, PG-SFT. Mostly online |
| W1 | WebSearch | SWEET-RL step-level critic training-time information | SWEET-RL 2503.15478 (ColBench, privileged critic) |
| W2 | WebSearch | "Watch Every Step" IPR | IPR, EMNLP'24, offline step-level contrastive pairs |
| W3 | WebSearch | execution-grounded credit assignment GRPO code | EGCA 2603.16158 (ICLR'26 workshop) |
| W4 | WebSearch | openreview offline step-level credit SWE agent weighted fine-tuning | **Orchard 2605.15040**, **Credit Without Ground Truth 2608.19760**, ABSeeker, RTMC, DRACO, AgentBrew |
| F3 | WebFetch | arXiv abs 2605.15040 | Orchard-SWE abstract (69.7%/73.0%, credit-assignment SFT) |
| F4 | WebFetch | arXiv abs 2608.19760 | Audit vs executed replay; placebo-level; ESS-matching rule |
| F5 | WebFetch | arxiv.org/html/2605.15040v3 (0–100k) | §3.3 LLM-TD credit, rise segments, binary loss selection |
| F6 | WebFetch | same, offset 100k | §3.8 ablation 59.3 → 61.2 |
| F7 | alphaXiv full text + grep | DepGPO 2610.03634 | Line-level writes, file-level E, stdout-edge read credit, limitations |
| W5 | WebSearch | SBFL omission faults missing code coverage limitation | SBFL survey, MCFL, "case against coverage spectra" |
| F8 | WebFetch | arXiv abs 2608.05102 | ABSeeker ABC-SFT per-turn reweighting |
| W6 | WebSearch | training-inference mismatch vLLM FSDP truncated IS | NeurIPS'25 rollout-training mismatch, TIS, QaRL |
| W7 | WebSearch | site:openreview.net dependency credit offline fine-tuning verifier read set | No match; VICT, Coverage-Not-Targeting, STAMP, VPR |
| F9 | WebFetch | arXiv abs 2503.15478 | Abstract only; offline/online not stated |
| F10 | WebFetch | arXiv abs 2607.11172 | STAMP first-exposure provenance credit, GRPO |
| F11 | WebFetch | arXiv abs 2609.02417 | V_d, uniform beats targeting, shuffled control |

**Not searched:** Semantic Scholar citation API for DepGPO/DRIFT; full texts of SWEET-RL, ABSeeker, 2608.19760, 2609.02417; Digi-Q/Q-SFT details; AgentPRM; Multi-SWE-bench coverage tooling.
