# Audit offline SWE credit against patch necessity

**Bottom line.** Sanity checks against current research narrowed all three earlier candidates, and only one direction came out stronger than it went in. **Z** (zero-information actions) lost its "first ≥30B mechanism" claim to TACT and its harness-free framing to TVCACHE and the OpenHands StuckDetector, and falls from 21 to about **17–19**. **DECAF** lost its gap statement to Orchard-SWE, FLARE and SWE-Prime, all of which already put step credit into offline SWE SFT (**16**). **SCAD-R** lost its method claim to Memento No More, OPHSD and HAD, and the earlier report had misapplied PSP's lesson to it (**13–14**). DECAF's sanity reviewer proposed the survivor: **CERT**, an interventional step-credit audit. CERT runs hunk-level Shapley necessity on the final patch with deterministic test reruns, then uses `git blame` to trace that necessity back to the trajectory steps that wrote it. It scores the published offline SWE credit sources against that yardstick: Orchard LLM-TD, FLARE backward chains, DepGPO provenance, SWE-Lego/Nebius error masks, SWE-Prime segment scores and SRFT critic masks. CERT survived an adversarial critic and an OpenReview-focused red team, but **only narrowed**. The ICLR 2027 submission *Counterfactual Supervision* already scores LLM credit against execution-derived gold on SWE-rebench (closeness 2.5). TRIM already owns the edit-removal instrument (2.0). 2608.22960 already audits judges by replay on SWE-bench Verified (2.0). What remains is a different estimand, deterministic and policy-independent necessity of *surviving* writes, applied to the offline sources nobody has audited (RT-O1–O7, RT-A1, RT-A3, CD-W3, CD-W11). It scores **≈19.5/25 (range 17–20)**, level with or slightly ahead of Z. The verdict is **GO for a 2-day, CPU-only, $0 falsification test** on 100 SWE-smith trajectories, with kill criteria K0–K3. If CERT dies, **Z is the fallback** and CERT's measured null-write rate folds into Z's analysis. **Nothing in this report has been run.**

Search-log convention, used for every novelty or absence claim: **Z-S1…S25** = `sanity_Z.md`; **D-L1, D-S1…S4, D-W1…W7, D-F1…F11** = `sanity_DECAF.md`; **SR-S1…S22** = `sanity_SCAD-R.md`; **CD-L1–L2, CD-W1…W11, CD-F1…F10** = `critic_design.md` (main pass plus kill round 2); **RT-O1…O7** (OpenReview API), **RT-A1…A4** (alphaXiv), **RT-W1…W9**, **RT-F1…F4**, **RT-FC1** (Firecrawl PDF) = `prior_art_redteam.md`. C-Q, R and X IDs refer to the earlier report *PEFT and RL for SWE agents*. "Not found" always means absence in the indexes searched: alphaXiv, web search, and for the first time the OpenReview search API. Novelty for anything newer has a short shelf life.

---

## Round A: every earlier candidate had a closer neighbour than the earlier report found

The earlier report ranked Z first (21), DECAF and SCAD-R tied (16), and said OpenReview was never searchable. Three independent sanity reviewers re-checked each candidate against the field. Each one found a closeness-2 to 2.5 paper the kill rounds had missed, plus at least one error in the earlier report.

### Z: TACT and harness-level caches take the concept and the certificate

**Findings.** **TACT** defines *overacting* as a step whose action "returns an observation already in 𝒦t". It works on SWE-bench Verified with **Gemma-4-26B-A4B**. Hidden states separate overacting linearly (AUC ≈ 0.9), and training-free steering adds **+4.8 pp** resolve on Gemma and +5.8 pp on Qwen3.5-27B, with up to 26% fewer steps ([TACT](https://arxiv.org/abs/2605.05980); Z-S2, Z-S4, Z-S12). That is a judge-labelled superset of Z's label, studied at representation level, in the same task family, on a Gemma-4 sibling. Z's "first real-agent mechanism at ≥30B" claim does not survive a 27B→31B distinction.

The state certificate also exists at the systems level. **TVCACHE** allows a tool-cache hit only when the full tool history matches, "guaranteeing identical environment state" ([TVCACHE](https://arxiv.org/abs/2602.10986); Z-S19). The **OpenHands StuckDetector** is on by default and halts on 4+ identical action–observation pairs ([docs](https://docs.openhands.dev/sdk/guides/agent-stuck-detector); Z-S9, Z-S21). That also right-censors loops in the Nebius corpus Z planned to use. Z's own sanity pass suggested this check (INFERRED, unverified).

The red team's earlier "no mechanism papers" (R23) was wrong. Four rival accounts exist: induction-head copying ([2505.13514](https://arxiv.org/abs/2505.13514)), retrieval-failure fallback ([2504.01100](https://arxiv.org/abs/2504.01100)), errors in learning and risk aversion ([2512.12895](https://arxiv.org/abs/2512.12895)), and belief deviation ([2510.12264](https://arxiv.org/abs/2510.12264)) (Z-S3, Z-S17).

The earlier report's relevance number was also mis-cited. The "79–98% of tasks" covers three behaviours, two of them not exact repeats. The exact-repeat (ReTest) slice costs only **0.83–5.39%** of spend ([2609.30725](https://arxiv.org/html/2609.30725); Z-S13). SABER predicts small resolve effects from non-mutating deviations ([SABER](https://arxiv.org/abs/2512.07850); Z-S6). Finally, "provably identical output" fails for flaky tests, polling, exogenous edits and post-compaction re-reads (Z-S11, Z-S18).

**Decision.** Z survives as a junction: a deterministic, judge-free label with a legitimate-repeat guard, plus a test of whether weight-level training beats a **same-certificate harness mask**. No paper does state-certified offline repeat negatives (Z-S2, Z-S15, Z-S20, Z-S25; earlier C-Q23, R2, R8, R14). Re-scored to **≈17–19**. **Next step:** Z becomes the fallback, carrying the H-mask baseline and the renamed "certified observation-redundant action" definition.

### DECAF: offline step credit for SWE already exists, and its credit rule fights its own metric

**Findings.** The DECAF gap, "offline methods only weight whole trajectories", is **false as worded**. **Orchard-SWE** §3.3 already runs offline SFT with per-step credit c_t = V(s_{t+1}) − V(s_t). A teacher shown the gold outcome annotates the values, and the loss is restricted to "rise segments" on 32,536 unresolved trajectories ([Orchard](https://arxiv.org/html/2605.15040v3); D-W4, D-F3, D-F5). Its +1.9 pp (59.3 → 61.2) adds 32k trajectories, so dose and credit are confounded (D-F6). **FLARE**'s RADAR traces backward from patch differences and failed tests to step-level labels offline. Those labels train a reward model used for SFT selection and RL rewards ([FLARE](https://www.alphaxiv.org/abs/2609.23808); D-F2). SWE-Prime masks low-scoring 20-step segments ([SWE-Prime](https://www.alphaxiv.org/abs/2608.27449)). SWE-Lego and Nebius mask error turns deterministically ([SWE-Lego](https://www.alphaxiv.org/abs/2601.01426)).

Two credit audits now set the control standard. **Credit Without Ground Truth** (2608.19760) found no credit signal beating its marginal-matched shuffled control on ALFWorld. Only **30.5%** of decision points had a non-zero replay contrast, and the paper states that unmatched comparisons "measure dose, not credit" ([2608.19760](https://arxiv.org/abs/2608.19760); D-W4, D-F4). **Coverage, Not Targeting** (2609.02417) shows that under terminal verifiers, concentrating credit on progress turns ≈ random turns, and makes a matched-concentration shuffled control mandatory ([2609.02417](https://arxiv.org/abs/2609.02417); D-F11).

Inside the design, pure dependency credit gives **zero credit to the final verification test run**, so DECAF's loss pushes against its own headline metric, "tested-after-last-edit". Its NEW-2 coverage refinement is probably marginal, because DepGPO already tracks line-level writes ([DepGPO](https://www.alphaxiv.org/abs/2610.03634); D-F7).

**Decision.** The gap narrows to "deterministic, execution-derived offline step credit for SWE, validated against interventional ground truth". No such work was found (D-S1–S4, D-W4, D-W7). The reviewer's recommended rewrite was an **audit-first** study: hunk ablation instead of coverage for writes, an explicit verification class, and arms {uniform, shuffled-matched, LLM-TD, DepGPO, certified} at matched effective sample size. They named it **CERT-SFT**. **Next step:** CERT goes to Round B as a candidate.

### SCAD-R: same-model hint internalization is two years old, and the PSP defence was wrong

**Findings.** **Memento No More** (MNM) already distils state-triggered corrective hints into the *same* Llama-3.1-70B through a LoRA, for multi-turn tool agents. It uses scripted mistake detectors, DAgger-style rounds, KL loss and hint dropout of 0.9 ([MNM](https://www.alphaxiv.org/abs/2502.01562); SR-S14, SR-S15). **OPHSD** self-distils a deterministic programmatic harness into the same model and removes it at inference, though only for single-turn tasks ([OPHSD](https://www.alphaxiv.org/abs/2605.08741); SR-S18, SR-S19). **HAD**'s harness is deterministic and uses almost SCAD-R's exact templates ("Step {t} of {cap}", "did not work here (attempt N)", "Nothing new has been observed for {k} steps") ([HAD](https://www.alphaxiv.org/abs/2610.02858); SR-S18).

The earlier report said "privileged information belongs in the sampler, not the loss; this lesson shapes SCAD-R". That gets PSP backwards. SCAD-R *strips* the notes from the training context, which makes it cross-view context distillation. Its only real protection is information symmetry, which zeroes PSP's irreducible-gap term ([PSP](https://www.alphaxiv.org/abs/2609.29051); SR-S11). Symmetry does not guarantee learnability.

**OP²SD** shows privileged-distillation gains often come from **context-induced teacher behaviour**: an *unrelated* worked solution matches or beats the real one in 8 of 9 groups ([OP²SD](https://www.alphaxiv.org/abs/2608.09228); SR-S4, SR-S6). The NUS dissection agrees: the best reference view adds only +1.30 pp and does not survive Holm correction ([2609.20612](https://www.alphaxiv.org/abs/2609.20612); SR-S12). SCAD-R has no placebo-note arm.

**Decision.** SCAD-R dies as a method (closeness to MNM ≈ 2.5). It survives only as a "tell vs punish" arm inside Z, with a placebo note and a system-prompt rule as controls. No placebo-note control in agent harness distillation was found (SR-S1–S22). Re-scored to **13–14**. **Next step:** none on its own.

---

## Round B: CERT wins the rubric because it needs no GPU and fails informatively

**Findings.** The official rubric credits Novelty that includes "highlighting important properties of existing methods". That lets an audit paper compete with a method paper. Before the red team, the critic scored the four candidates (all GUESSED):

| Criterion | **CERT (audit, ±T1)** | Z (post-sanity) | DECAF | SCAD-R |
|---|---|---|---|---|
| Novelty | 3 (4 with T1). First interventional yardstick for SWE training-credit sources | 2–3 | — | 1 |
| Quality / generality | 4. 2–3 generators, 2 scaffolds, synthetic + real bugs | 3 | — | — |
| Relevance | 4. Tests the credit Orchard, FLARE, SWE-Prime and DepGPO actually use | 3 | — | — |
| Verifiability | 5. Deterministic, CPU-only, MIT/CC-BY data | 5 | — | — |
| Clarity | 4. One yardstick, one stratum, one table | 4 | — | — |
| **Total** | **20 (18–21)** | **17–19** | **16** | **13–14** |

CERT has three structural advantages. First, it **does not depend on Phase 0b** (31B training on 4×L4, E5 in the earlier report), which is still unverified. Z Stage 2, DECAF Tier B and the mainline LoRA all do. Second, its labels are deterministic test reruns that anyone can re-execute. Third, its likeliest failures are themselves findings. "Survival is enough on SWE writes" or "necessity reduces to a diff rule" is still a one-page result.

It also reuses work already designed. DECAF's sanity reviewer pointed out that hunk ablation reuses **IFG Stage A** from the graph report *LLM graph reasoning for code*. IFG enumerates function-level subsets of the *gold* patch to find jointly necessary functions, and IBugDC's divisibility definition carries over directly. CERT runs the same subset-apply-and-test loop on *agent* patches.

**Decision.** Pick CERT, framed as an audit. Training is optional. **Next step:** an adversarial critic attacks the design: the ground truth, data, metrics and power.

---

## Round C: the critic made the ground truth honest about what it cannot see

### Three findings reshaped the design

**Findings.**

First, the ground truth is **partly circular with DepGPO**. On resolved trajectories, "necessary hunk → blamed back to the step that wrote it" is nearly DepGPO's line-level write credit. The two can disagree only on lines that **survive but are not needed**, and on jointly necessary or redundant hunks. So the audit is informative only on those strata.

Second, **the write-side instrument has prior art**. TRIM (2607.18161) traces trajectory edits into the final patch and minimises it with validation runs, removing 17.9–32.9% "CodeSlop" ([TRIM](https://arxiv.org/abs/2607.18161); CD-W7, CD-F6). CAR (2606.08275) proposes do()-replay plus Monte-Carlo Shapley for step credit, but only on synthetic causal models, and reports LLM-judge step accuracy of about 14% on Who&When ([CAR](https://arxiv.org/abs/2606.08275); CD-W1, CD-F1). No hunk-Shapley paper turned up (CD-W3). Novelty therefore rests on using interventional patch credit **as a yardstick for training-credit sources**, not on ablation or Shapley.

Third, **read credit cannot be interventional for public corpora**. The generating models (Claude 3.7, Qwen3-Coder-480B) cannot be resampled offline within budget. The ground truth is therefore interventional for writes, provenance-level for reads, and undefined for negative information.

### Pre-registered ground truth and step classes C1–C6

Let P be the final patch restricted to non-test source files, split into units (hunks merged to function scope).

- **Value function.** v(S) = (fraction of F2P tests passing with only S applied) × 1[no P2P test newly fails].
- **Unit credit φ_u.** Exact Shapley for k ≤ 8 units (≤256 runs), 200-permutation Monte-Carlo above that. Necessity and IBugDC-style minimal sufficient sets are reported alongside.
- **Step write-credit.** g_t = Σ_u φ_u · b_{t,u}, where b is the share of unit u's lines that `git blame` attributes to step t in a replayed trajectory.

Redundant hunks are why Shapley rather than single-hunk necessity: if either of two hunks alone fixes the bug, single-hunk ablation gives both zero, while Shapley gives each 0.5.

| Class | Definition | Label strength |
|---|---|---|
| **C1** | Mutating step with g_t > ε | Interventional positive |
| **C2b** | Mutating step whose lines survive in P but g_t ≤ ε | **Interventional zero (the headline negative)** |
| C2a | Mutating step with no surviving lines (superseded or reverted) | Survival zero |
| C2-trivial | Writes only to tests or unimported scratch files | Trivial zero, excluded from the headline |
| C3 | Read whose observation first introduces a ≥3-identifier n-gram later used in a C1 write | Provenance positive (heuristic) |
| C4 | Test or repro run after the last C1 write | Verification (own class) |
| C5 | ⊂ C3: run whose traceback identifiers appear in later C1 writes | Provenance positive |
| C6 | Everything else, including exploration and negative information | **Unlabelled, never negative** |

The critic also handled the failure cases DECAF's reviewer raised:

- **Verification runs** get their own class (C4). That fixes DECAF's contradiction, and every source is reported on how much credit it puts on C4.
- **Weak F2P tests** are restated as "contribution to the verified outcome". AgentLens measures **10.7%** lucky passes, ranging 0.5–23.2% by model ([AgentLens](https://arxiv.org/abs/2605.12925); CD-F4).
- **Flaky tests** are handled by rerunning the full and empty patches 3× and each subset 2×.
- **Bash-based writes** are handled by replaying every mutating command with a commit after each step. A trajectory is dropped if its replayed diff ≠ its recorded patch.
- **Expected coverage** (GUESSED): about 20–30% of steps get interventional labels, a further 15–25% get provenance positives, and 45–60% stay unlabelled.

### Trajectory sources and licences

**SWE-smith** is primary. The trajectory card is **MIT**, has 76k rows and has a `model` column with three values ([SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories); CD-F8). The tasks span 128 repos ([SWE-smith](https://arxiv.org/abs/2504.21798); CD-F3), with one commit per repo. **SWE-MiniSandbox** caches all of them as venvs in **13.5 GB vs 295 GB** of Docker, with evaluation parity ([MiniSandbox](https://arxiv.org/html/2602.11210v1); CD-F7). Docker is not needed.

**Nebius SWE-rebench-OpenHands** is the generality corpus. It is CC-BY-4.0, generated by Qwen3-Coder-480B on OpenHands, with real issues. It is gated on a fidelity check, because its per-instance `install_config` will not cover apt-level dependencies ([SWE-rebench card](https://huggingface.co/datasets/nebius/SWE-rebench); CD-F2; [trajectories card](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories); Z-S22). Its fidelity is GUESSED at 60–70%.

Four corpora were rejected:

- SERA is soft-verified, so it has no ground truth.
- SWE-ZERO is execution-free.
- SWE-Gym needs about 6 TB of images.
- R2E-Gym's card carries no licence (CD-W8, CD-W9; sibling notes).

**Licence rule.** Auditing Claude-generated trajectories is fine. Any training stage uses only the Nebius (Qwen) slice or our own Gemma slice.

### Metrics, controls and power

The headline metric is **within-writes AUROC, C1 vs C2b**, reported as the *incremental* AUROC each source adds over a logistic **type + recency + survival baseline**. The critic called that baseline "the real bar". The mandatory control is a **type-stratified shuffled-matched** source. Secondary metrics:

- **null-mass**: the share of credit on C2a ∪ C2b ∪ C2-trivial;
- **verification share**: credit on C4 relative to uniform;
- per-trajectory **R-precision** and **Kendall τ_b**;
- **read-tier precision only**, with no recall, since C6 is unlabelled.

Statistics use a task-clustered bootstrap (≥2,000 reps) and Holm correction across sources on the headline metric. The sources are uniform, both shuffled controls, the baseline, DepGPO-offline, the error-turn mask, Orchard LLM-TD, FLARE-style chains and SWE-Prime segments. Gemma-4-31B is the judge, with a frontier judge on 150 trajectories to answer the "weak judge" objection (≤$15, GUESSED).

**Power (GUESSED inputs).** About 300 trajectories gives roughly 600 C1 and 450 C2b steps. With a design effect of ~1.5, the **minimum detectable ΔAUROC is ≈ 0.06**, or ≈ 0.08 on 150 Nebius trajectories. That is enough to ask whether any source beats the trivial baseline, but not to rank close LLM-judge variants, and the paper must say so.

**Labelling budget (GUESSED).** About 2–4 CPU-minutes per trajectory, or ~700–1,200 per 12 h Kaggle CPU session. **Engineering, not CPU, is the bottleneck.**

### Training stays optional

**T1** runs only if sources disagree by ≥ 0.06 incremental AUROC. It uses a small Gemma (E4B-class) with QLoRA on ~300 Nebius trajectories. There are four arms (uniform, shuffled CERT, best LLM judge, CERT), all matched on ESS = (Σw)²/Σw². The primary outcome is held-out **credit-targeted NLL shift**, Δlog-lik(C1) − Δlog-lik(C2b). Resolve rate is not an outcome: n ≈ 50 gives an SE of about 6 pp against expected effects of 1–2 pp. Cost is ~10–14 GPU-h, GUESSED.

**Decision.** The audit stands alone at **20/25**. The first-pass kill criteria were:

- **K0 fidelity**: below 80% replay-exact trajectories with 3/3 F2P;
- **K1 non-triviality**: below 15% of trajectories with a C2b step or a non-additive unit;
- **K2 label density**;
- **K3 saturation**: baseline AUROC ≥ 0.95;
- **K4**: an informational judge check.

**Next step:** a prior-art red team, this time including OpenReview.

---

## Round D: OpenReview showed the audit genre is crowded, but not on this estimand

### OpenReview is reachable through its API, not its website

**Findings.** Every earlier report flagged OpenReview as unsearchable. The red team found a working route. The term-search API `api2.openreview.net/notes/search?term=…&source=forum` answered **without login** and returned ICLR 2027 submissions (posted up to about 29 Sep 2026), 2026 workshop papers and ACL ARR submissions, with abstracts. It returned HTTP 429 after about 20 rapid calls, and pacing fixed that.

Three routes failed:

- `notes?id=` returned **403 ChallengeRequiredError**;
- WebFetch of forum and PDF pages hit a browser challenge;
- the JS search UI loads only a shell.

**Firecrawl's PDF parser did read one submission PDF in full (23 pages)**. Web search with `site:openreview.net` does not index ICLR 2027 submissions at all (RT-W1–W3, RT-W5, RT-O1–O7, RT-FC1).

Across **28 API queries**, no submission matches CERT's combination. About ten ICLR 2027 submissions sit on the "audit step credit against executed replay" axis. ICLR 2027 reviews (due Nov 2026) and per-note pages stay unreadable, and keyword search can miss a submission titled without "credit", "process reward" or "SWE".

### Seven papers that narrow CERT

**Counterfactual Supervision for Credit Assignment in LM Agents** (ICLR 2027 submission, QzGGstYqbH; read in full, VERIFIED) is the closest, at **closeness 2.5** ([PDF](https://openreview.net/pdf?id=QzGGstYqbH); RT-O1, RT-FC1).

- **Gold labels.** Before each turn, the environment is restored, **12 same-policy alternatives** are sampled, and each action gets **16 continuations**. Gold is a Monte-Carlo advantage.
- **Corpus.** 300 trajectories and 5,192 turns from Qwen3.5-4B/9B on SWE-rebench with mini_swe. The annotation corpus requested **792,208 continuations**.
- **What it scores.** An evolved LLM retrospective selector, against permutation controls. Within-trajectory permutation drops weighted sign agreement from **67.1% to 58.7%**.
- **Training.** Online PPO-style, single-run, with a shuffled arm. It reports +4.52/+4.79 pp over PPO on SWE-bench Pro.
- **Behavioural finding.** 136 of 143 successful trajectories contain an MC-negative turn, and semantic "redundant" labels agree with gold only 56.8–68.3% of the time. That is direct evidence that SWE-Lego/SRFT-style semantic error labels ≠ execution credit.
- **What it does not do.** No patch reversion, no audit of Orchard, FLARE, DepGPO, SWE-Lego/Nebius or SWE-Prime, no offline SFT, and no ESS matching.

**TRIM** (2.0 as an instrument) removes surviving edit sequences, then files, then edit actions, accepting a removal only if tests still pass. On 333 SWE-agent trajectories from SWE-bench Verified it removes **20.0%** of slop on pre-existing files, preserves correctness in 99.1% of cases, and matches DD-Hunk (32.9 vs 31.5%) with about 1.9× fewer validations ([TRIM](https://www.alphaxiv.org/abs/2607.18161); RT-A3, RT-F4; SNIPPET). Its purpose is patch minimisation, not credit.

**What Process Evaluation of Coding Agents Actually Measures** (2608.22960, 2.0) estimates step effects by replay on SWE-bench Verified file localisation. **0 of 190** step effects survive FDR correction. Full-trace judges show collider bias: verdicts shift later (+0.537) in 59/70 episodes ([2608.22960](https://www.alphaxiv.org/abs/2608.22960); RT-A2, RT-F2; SNIPPET). This paper gives CERT its best motivation: stochastic replay credit is underpowered on SWE, and deterministic necessity has no sampling variance.

**Credit Without Ground Truth (ICLR 2027 version) vs ACRE** now disagree.

- The CWGT revision (Me3WCeeMWW) softened the arXiv "placebo-level" result to "modestly positive" judge alignment, with training utility "unestablished" ([OpenReview](https://openreview.net/forum?id=Me3WCeeMWW)).
- ACRE (HbNTIXwoeU, ARR) finds that credit fidelity tracks held-out gain (cross-row Spearman **0.886**) across six signals on ALFWorld ([OpenReview](https://openreview.net/forum?id=HbNTIXwoeU)) (RT-O4).
- SWE is a second domain in which to adjudicate the two.

**CEDAR** (lw07yfdkn8) shows that verifier-defined credit can equal the changed-position set, so a plan-diff baseline recovers it fully ([OpenReview](https://openreview.net/forum?id=lw07yfdkn8)). This is the most dangerous critique for CERT: maybe necessity is "just the diff".

**CRR** (2609.33875, NeurIPS 2026) forks four decision points per SWE trajectory. It *announces* a 1.4M-tuple dataset "for verifier evaluation", which is CERT's use case, but has not released it ([CRR](https://www.alphaxiv.org/abs/2609.33875); RT-F1; no release link found in CD-W10, CD-F9).

**TraceProbe** (2607.06184) already labels writes deterministically as SURVIVED or REVERTED on 2,500 trajectories. It runs no ablations, but SURVIVED becomes a baseline CERT must beat ([TraceProbe](https://www.alphaxiv.org/abs/2607.06184); RT-F3, CD-F10).

Other papers to cite or audit:

- **RMPO**: leave-one-out deletion credit, ρ = 0.842 with planner progress ([ac6BotAkVF](https://openreview.net/forum?id=ac6BotAkVF)).
- **Individual Credit, Joint Effects**: 37.5% of credited actions move against their credit after the update, so credit fidelity ≠ update fidelity ([H4QZsZe2eb](https://openreview.net/forum?id=H4QZsZe2eb)).
- **CARVE**: offline replay-verified gradient admission, 60.9 → 63.8 on Code/SWE ([8yNUqdO1KV](https://openreview.net/forum?id=8yNUqdO1KV)).
- **Step Rejection Fine-Tuning (SRFT)**: offline critic masks on SWE, +3.7 vs +2.4 for RFT ([RndJzeZV81](https://openreview.net/forum?id=RndJzeZV81)). It becomes a sixth audited source.
- The wildfire replay benchmark, where critic credit RMSE is 8.8–15.6× that of zero credit despite R² 0.91–0.95 ([yZzXZX35EZ](https://openreview.net/forum?id=yZzXZX35EZ)) (all SNIPPET; RT-O2, RT-O5–O7).

None of the agent-PRM benchmarks uses execution-derived SWE step labels. SWE-PRM, SWE-Shepherd and SWE-TRACE use taxonomy, heuristic or rubric labels (RT-W8, RT-A2). No SWE step-level training-data attribution exists (RT-A4).

### What is still open

**Findings.** As "the first interventional audit of SWE step credit", CERT is **dead**: Counterfactual Supervision and 2608.22960 got there first. As the narrower claim below, it is **still open in every index searched**. No paper turns final-patch hunk necessity into per-step credit and uses it to benchmark other credit sources. The search terms included "patch hunk necessity", "hunk ablation", "revert edits rerun tests", "necessary edits patch minimal" and "edit provenance agent trajectory" (RT-O1, RT-O2, RT-O6, RT-A3, CD-W3, CD-W7). No paper scores any of the published offline SWE credit sources against execution-derived labels (RT-O1–O7, RT-A1, D-S1–S4, D-W4, D-W7). The narrower claim has three parts:

1. deterministic, policy-independent necessity;
2. applied to the six offline SWE sources;
3. with explicit trivial-baseline and identifiability controls.

The red team ranked five reviewer objections:

1. the ground truth is not credit (it misses exploration, and CEDAR suggests it reduces to the diff);
2. it is circular with DepGPO;
3. TRIM already did the instrument and Counterfactual Supervision did the audit;
4. single-hunk patches make necessity trivial;
5. necessity is the wrong target for RL or SFT weighting.

**Decision.** Novelty falls from 3 to roughly 1.5–2 in the red team's view, and 2.5 after the critic narrows the claim in Round E. **Next step:** the critic answers each objection or kills the direction.

---

## Round E: the narrowed audit survives at 19.5, ahead of Z by a hair

**Findings.** The critic compared CERT with Counterfactual Supervision axis by axis and judged the differences **real, not cosmetic**:

- **Estimand.** CERT asks which written tokens the verified outcome *depended on*. CS asks which choice *raised expected return*. CERT needs about 10 deterministic CPU test runs per trajectory; CS needed 792k continuations.
- **Target.** CERT audits the six offline sources SFT pipelines actually consume, while CS audits one evolved selector.
- **Corpus.** CERT uses teacher corpora (Claude 3.x and Qwen3-Coder-480B on SWE-agent and OpenHands), while CS uses on-policy 4B/9B rollouts.

The appealing reframing "do necessity and replay advantage disagree?" is **rejected as a headline**. It needs policy access, CRR's records are unreleased, and 2608.22960 shows 8 replays per prefix are too few. No paper compares the two estimands (CD-W11), so the question is open but out of budget. It is pre-registered as a conditional appendix if CRR releases its data before 25 Oct.

**Narrowed headline (pre-registered).**

> On resolved SWE trajectories from ≥2 teachers (≥2 scaffolds if Nebius passes K0), deterministic patch-necessity credit identifies **surviving-but-unnecessary writes** that survival-, diff- and provenance-based credit cannot see. We measure how much imitation weight each published offline SWE credit source (Orchard-TD, FLARE, DepGPO, SWE-Lego/Nebius, SWE-Prime, SRFT) places on interventionally null writes, against a TraceProbe-style survival/diff baseline and a type-stratified shuffled control.

Read-tier provenance labels (C3, C5) are **dropped from every metric that scores a source**. They appear only descriptively, which removes the DepGPO circularity at its root.

| # | Objection | Answer and design change |
|---|---|---|
| 1 | Ground truth is not credit: it misses exploration, reads are heuristic, and CEDAR says it reduces to the diff | Claims cover **write credit for imitation** only; reads are descriptive. Identifiability becomes a kill criterion (K2). The headline stratum is *surviving* writes, where TraceProbe's SURVIVED label is constant by construction and cannot score anything |
| 2 | Circular with DepGPO | DepGPO's write credit = surviving-line share, which is constant across C1/C2b in the headline stratum. Its expected AUROC there is ≈ 0.5. That is the finding ("survival ≠ necessity"), not circularity. No provenance edge used in a label is used by any scored source |
| 3 | TRIM did the instrument; CS did the audit | Reuse and cite TRIM's EDIT-SEQ → FILE → ACTION hierarchy as the instrument. The contribution is the estimand, the offline sources, ESS-matched update fidelity, and ACRE/CWGT adjudication (T1 only). Pre-step: read TRIM's full text to see whether its per-step removability already gives the C2b rate. If it does, cite it and present the cross-source audit as the contribution |
| 4 | Low variance: single-hunk patches dominate | K1 requires **absolute** support (≥ 40 C2b steps in 100 trajectories), reported first in the paper. TRIM's 20% slop on Verified suggests C2b is non-empty. SWE-smith bugs are simpler than Verified, which is the risk |
| 5 | Necessity is the wrong target | Offline SFT decides *which tokens to imitate*, and imitating null edits is what produces CodeSlop. The claim is limited to SFT weighting, not RL advantage. T1's outcome becomes **update fidelity** at matched ESS, following H4QZsZe2eb. No behavioural gain is claimed by 5 Nov |

**Revised first test** (≈ 2 days, Kaggle CPU, $0). The procedure has six steps:

1. Two 30-minute pre-steps: read the TRIM full text, and check whether CS or CRR data has been released.
2. Build venvs for 5–6 SWE-smith repos with fast tests.
3. Sample 100 resolved Claude-3.7 trajectories, at least 50 of them with ≥ 2 mutating source steps.
4. Replay each with commit-per-step, then run the full and empty patch 3×, exact Shapley, blame and class labels.
5. Fit the diff-feature baseline.
6. **Required:** an Orchard-TD-style judge on 50 trajectories, using Gemma-4-31B on the AI Studio free tier.

| ID | Measures | Kill if | Then |
|---|---|---|---|
| **K0 fidelity** | Replayed diff == recorded patch AND full patch passes F2P 3/3 | < 80% | Fix the parser once, then try Nebius. If both fail, kill |
| **K1 support** | C2b steps plus non-additive units (Shapley ≠ necessity) | < 40 C2b steps OR < 15% of trajectories with ≥ 1 | Intervention ≈ survival on SWE writes. Kill |
| **K2 identifiability** (CEDAR) | AUROC of a non-interventional diff baseline on C1 vs C2b (hunk size, new vs existing file, comment/print/log-only lines, edit position, modal edited file) | ≥ 0.90 | "Necessity reduces to a diff rule." Kill the audit; keep the rule as a one-paragraph finding |
| **K3 headroom** | Judge's incremental AUROC over K2's baseline, 50 trajectories | Informational | ≥ 0.9 → "LLM credit already tracks necessity" (positive). ≈ shuffled → "LLM credit spends X% on null writes" |

**Rubric re-score** (INFERRED/GUESSED):

| Criterion | Before | After |
|---|---|---|
| Novelty | 3 | **2.5** |
| Quality | 4 | 4 (3 if Nebius fails) |
| Relevance | 4 | 4 |
| Verifiability | 5 | 5 |
| Clarity | 4 | 4 |
| **Total** | 20 | **≈ 19.5 (17–20)** |

That compares with Z ≈ 17–19, DECAF 16 and SCAD-R 13–14.

**Decision.** **GO for the 2-day CPU test**, time-boxed to ≤ 10 working hours, 0 GPU-h and $0. Stop at the end of day 2 whatever the state. No judge beyond the 50-trajectory check and no T1 until K0–K2 pass.

- **If all pass:** scale to 300 SWE-smith trajectories (2 models), then run Nebius K0 on 30 instances, then 150 Nebius trajectories. Judges for every source run on L4 (~4 GPU-h, GUESSED). Writing starts by 25 Oct. T1 runs only if sources disagree by ≥ 0.06 and small-Gemma QLoRA works.
- **If CERT dies** (K0 on both corpora, K1 or K2): **fall back to Z**. If K0 passed, the measured null-write rate per teacher becomes a second waste type in Z's analysis ("zero-necessity writes" beside "zero-information reads/repeats"), so the two days are not sunk. A K2-only failure becomes a CEDAR-style paragraph in Z.

**Next step:** Shrey's go/no-go.

---

## Where ideas died, and what each death taught

| Idea or claim | Killed or narrowed by | Lesson |
|---|---|---|
| Z: "first ≥30B mechanism of agent repeats" | TACT, Gemma-4-26B, same task family (Z-S2, Z-S12) | A 27B→31B step is not a novelty axis. A mechanism claim must discriminate among rival hypotheses (copying, fallback, risk aversion, belief deviation) |
| Z: training as the obvious fix for a deterministic label | TVCACHE, StuckDetector, Feedback That Backfires' decoder ban (Z-S9, Z-S16, Z-S19) | If a label is computable at inference, the reviewer asks "why train?" The same-certificate harness mask is the killer baseline |
| Z: "provably zero information"; "79–98% of tasks" | Flaky tests, compaction, polling; ReTest cost 0.83–5.39% (Z-S11, Z-S13, Z-S18) | Information is relative to the agent's context, not just the environment. Cite cost share, not task share |
| DECAF's gap "nobody does offline step credit for SWE" | Orchard-SWE, FLARE, SWE-Prime, SWE-Lego (D-W4, D-F2, D-F5) | The open seam is the *source* of credit (execution-derived vs LLM-judged), not offline step credit itself |
| DECAF Tier B as designed | 2608.19760, 2609.02417 (D-F4, D-F11) | Without shuffled-matched controls and ESS matching, a credit comparison measures dose |
| DECAF dependency credit as a training rule | Zero credit on final verification runs (D §2) | A credit rule must not train against the behaviour you report. Verification needs its own class |
| SCAD-R as a method; "PSP lesson shapes SCAD-R" | MNM, OPHSD, HAD templates; PSP misread (SR-S11, SR-S14–S19) | Stripping a note from context *is* loss-side distillation. Information symmetry is the only defence, and a placebo arm is mandatory (OP²SD) |
| CERT as "first interventional SWE credit audit" | Counterfactual Supervision (RT-FC1), 2608.22960 (RT-F2) | The genre "audit credit against executed ground truth" is crowded in ICLR 2027. Only the estimand and the audited sources are new |
| CERT's edit-ablation instrument as a contribution | TRIM (RT-A3, CD-F6) | Reuse the instrument and claim the yardstick use |
| CERT read-tier credit in scored metrics | DepGPO circularity; no offline resampling (CD §1) | Never score a source with labels built from the same edges it uses. Keep heuristic labels descriptive |
| "Necessity vs replay advantage" as headline | CRR unreleased; CS's 792k continuations; 0/190 FDR (CD-W10, CD-F9, CD-W11) | An open question is not a feasible one. It stays a conditional appendix |
| Resolve rate as a training outcome | SE ≈ 6 pp at n ≈ 50 vs 1–2 pp effects (CD §4) | Use dense, paired, rollout-free outcomes (credit-targeted NLL, update fidelity) |

---

## What is verified and what is guessed

| Item | Status |
|---|---|
| TACT overacting on SWE-bench Verified with Gemma-4-26B; +4.8 pp by steering | **VERIFIED** (Z-S12, full text) |
| ReTest cost share 0.83–5.39%; StuckDetector default-on at 4 repeats; TVCACHE prefix-match certificate | **VERIFIED** (Z-S13, Z-S21, Z-S19) |
| Orchard §3.3 offline LLM-TD credit SFT; +1.9 pp dose-confounded; FLARE offline backward chains | **VERIFIED** (D-F5, D-F6, D-F2) |
| SWE-Prime segment masks, SWE-Lego/Nebius error masks | **VERIFIED in sibling notes**, not re-read |
| 2608.19760 (30.5% non-zero contrast, ESS rule); 2609.02417 (shuffled control) | **SNIPPET** (abstracts) |
| MNM, OPHSD, HAD templates, OP²SD, PSP mechanism | **VERIFIED** (SR-S6, SR-S11, SR-S15, SR-S18, SR-S19) |
| Counterfactual Supervision design, corpus, 67.1% → 58.7% permutation drop, no offline sources or patch reversion | **VERIFIED** (RT-FC1, 23-page PDF) |
| TRIM numbers (20.0%, 99.1%, 32.9 vs 31.5, 1.9×); 2608.22960 (0/190, collider bias) | **SNIPPET** (alphaXiv reports) |
| CWGT-ICLR'27 softening; ACRE Spearman 0.886; CEDAR; RMPO; CARVE; SRFT; H4QZsZe2eb | **SNIPPET** (OpenReview abstracts via API) |
| CRR dataset announced, not released | **VERIFIED as absence** (CD-W10, CD-F9) |
| OpenReview API reachable unauthenticated; per-note and PDF pages challenge-gated; Firecrawl PDF works | **VERIFIED** (tool behaviour, RT-O1–O7, RT-W5, RT-FC1) |
| No hunk-necessity → step-credit audit; no audit of offline SWE sources vs execution labels; no necessity-vs-advantage comparison | **VERIFIED as absence** in indexes searched (RT-O1–O7, RT-A1, RT-A3, CD-W3, CD-W7, CD-W11); ICLR'27 reviews and NeurIPS'26 non-arXiv papers not searchable |
| SWE-smith trajectories MIT, 76k rows, 3 models; MiniSandbox 13.5 GB vs 295 GB with parity; Nebius CC-BY-4.0 | **VERIFIED** (CD-F8, CD-F7, CD-F2, Z-S22) |
| AgentLens 10.7% lucky passes | **VERIFIED** (CD-F4) |
| C2b is non-empty at ≥ 40 steps per 100 trajectories (K1) | **GUESSED**; TRIM's 20% slop on Verified is the only prior |
| Diff baseline stays < 0.90 AUROC (K2) | **GUESSED**; CEDAR is the prior against |
| Replay fidelity ≥ 80% (K0); Nebius fidelity 60–70% | **GUESSED** |
| Class coverage (C1/C2 20–30%, C3–C5 15–25%, C6 45–60%); median k 2–3 | **GUESSED** |
| 2–4 CPU-min per trajectory; 700–1,200 per 12 h session | **GUESSED** |
| MDE ΔAUROC ≈ 0.06 (300 traj) / 0.08 (150) | **GUESSED** inputs, INFERRED arithmetic |
| T1 cost 10–14 GPU-h; judge cost ≤ $15; L4 judge ~4 GPU-h | **GUESSED** |
| Plain venvs suffice on Kaggle (no chroot); identity of SWE-smith's third model; Kwai 66k licence | **Not checked** |
| All rubric scores | **INFERRED/GUESSED** |
| Any CERT result | **Nothing has been run** |

---

## Conclusion

The sanity rounds exposed a pattern. Each earlier candidate was a *method* whose novelty rested on an unsearched corner, and the 2026 literature filled every corner: TACT for Z, Orchard and FLARE for DECAF, MNM for SCAD-R. CERT survives for a different reason. It does not try to beat those methods. It measures them, on an axis the crowded ICLR 2027 audit genre has skipped: what the *realised artefact* depended on, rather than what a policy's alternatives would have returned. That is also why its failure modes are cheap and publishable. If necessity collapses to survival (K1) or to a diff rule (K2), the field learns that the expensive LLM-judged credit in Orchard, FLARE and SWE-Prime is chasing a target a regex already hits.

Two things should change how the project searches from here. OpenReview is no longer a blind spot: the API route works, so every novelty claim in the final paper must be re-run through it, and through the TRIM and CRR citation graphs, right before submission. And the paper's novelty now depends on two facts it does not control: whether TRIM's full text already reports per-step removability, and whether CRR releases its records. Both are 30-minute checks scheduled before day 1.

---

## Decision needed from Shrey

**Nothing in this report has been run.** Five calls, in order of urgency:

1. **Approve the CERT 2-day CPU test?** It is capped at ≤ 10 working hours, 0 GPU-h and $0, with K0–K2 as hard kills and the two 30-minute pre-steps (TRIM full text; CS/CRR release check). It does not touch the GPU quota or Phase 0b.
2. **Approve sending 50 SWE-smith trajectories (MIT, Claude-generated) to the Gemma AI Studio free tier** for the required K3 judge check? This is the only external API use in the test. A frontier-judge subsample (≤ $15, GUESSED) is deferred until K0–K2 pass and needs a separate yes.
3. **Licence policy.** Audit on Claude-generated SWE-smith trajectories, but train (T1, if ever) only on the Nebius Qwen slice or our own Gemma outputs. Confirm.
4. **Sequencing against Z and IFG.** CERT, IFG Stage A (graph report) and Z Stages 0–1 all want CPU days before the 5 Nov freeze. The recommendation is CERT first, because it is the current top score and needs no GPU. Z Stage 1 can share the Oct GPU slack, and the subset-apply-and-test runner should be written once for both CERT and IFG.
5. **Paper shape.** If CERT passes, it is the headline, writing starts by 25 Oct, and Z drops to the alternate. If it fails, Z is the headline with CERT's null-write rate as an analysis section. Ties go to the earliest entry, so either way submit as soon as the evidence allows, ahead of the 12 Nov deadline.
