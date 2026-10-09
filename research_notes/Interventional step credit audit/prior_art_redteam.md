# Prior-art red team: CERT-SFT, an interventional step-credit audit for offline SWE-agent training (8 Oct 2026)

Role: hostile reviewer trying to find the paper that already did this. Scope: the direction as briefed, which has three parts.
- (i) Ground truth from hunk/function reversion of the final patch plus F2P re-runs, traced back to steps through a provenance graph.
- (ii) An audit of existing SWE step-credit sources against that ground truth, with ESS-matched and shuffled controls.
- (iii) Optional ESS-matched offline LoRA arms.

Labels:
- **VERIFIED**: I read the relevant primary text this session (full text or the relevant section).
- **SNIPPET**: abstract, OpenReview abstract field, alphaXiv AI report, or search summary only.
- **INFERRED**: my own reasoning.

Inherited and not re-run: the search logs of `sanity_DECAF.md` (L1, S1–S4, W1–W7, F1–F11), `sanity_Z.md`, `PEFT and RL for SWE agents/critic_judgement.md` (Q1–Q38), `LLM graph reasoning for code/critic_judgement.md` (T1–T10) and `prior_art_redteam.md` (R1–R8). Orchard, FLARE, DepGPO, IBugDC, Credit Without Ground Truth (arXiv), Coverage-Not-Targeting, STAMP, EGCA and SWE-Prime are taken as known.

**Bottom line.**
- **OpenReview is no longer a blind spot.** Its search API (`api2.openreview.net/notes/search`) answered unauthenticated and returns ICLR 2027 submissions with abstracts. One PDF was read in full through Firecrawl.
- **No paper does CERT-SFT's exact combination.** That combination is deterministic hunk-necessity ground truth traced to steps, used to audit the *published SWE offline credit sources*, with ESS-matched offline SFT.
- **Each ingredient now has close prior art, and the genre "audit step credit against executed ground truth" is crowded.** Most dangerous:
  1. ICLR 2027 submission *Counterfactual Supervision for Credit Assignment in Language Model Agents*. Execution-derived gold turn credit on SWE-rebench coding trajectories. LLM credit scorers are evaluated against it with permutation controls. A shuffled-credit arm. Training on top.
  2. *TRIM* (2607.18161). Trajectory-guided counterfactual removal of agent edits with test validation on SWE-bench Verified. This is CERT's ground-truth instrument.
  3. *What Process Evaluation of Coding Agents Actually Measures* (2608.22960). Replay-based causal step attribution on SWE-bench Verified that shows LLM judges have collider bias and that step effects are underpowered.
  4. ICLR 2027 resubmission of *Credit Without Ground Truth* and the ARR submission *ACRE*. Credit-fidelity audit protocols.
- **The framing survives only narrowed** (closeness verdict at the end).

---

## Q1. OpenReview: what was reachable, and what it contains

### Takeaway
OpenReview's term-search API worked without login and indexed ICLR 2027 submissions (posted up to about 29 Sep 2026), ICML/NeurIPS/COLM 2026 workshops, and ACL ARR 2026 submissions. Three routes failed with a browser challenge: per-note fetch by ID, forum and PDF pages through WebFetch, and the JS search UI. Firecrawl did fetch one OpenReview PDF. Across 28 API queries, no submission matches CERT-SFT exactly. About ten ICLR 2027 submissions sit on the "audit step credit against executed replay" axis, and one of them is on coding agents.

### Cited Findings
**Reachability (VERIFIED by direct tool behaviour this session)**

| Route | Result |
|---|---|
| `openreview.net/search?...` via WebFetch | Page loads only a "Loading" shell (JS). No results. |
| `api2.openreview.net/notes/search?term=…&source=forum` (curl / urllib) | **Works.** Returns titles, venue, abstracts and keywords, about 25–40 hits per term. Hit HTTP 429 after about 20 rapid calls; pacing fixed it. |
| `api2.openreview.net/notes?id=…` | **403 ChallengeRequiredError**. |
| `openreview.net/pdf?id=QzGGstYqbH` via WebFetch | "Verifying your browser" page. Unreadable. |
| Same PDF via Firecrawl scrape (pdf parser) | **Works**: 23 pages, status 200. |
| WebSearch `site:openreview.net …` (2 queries) | Only older or general items (iStar ICLR'26, SWE-RM, counterfactual contribution analysis NeurIPS'23). The ICLR 2027 submissions are not in the web index. |

**ICLR 2027 / 2026-venue submissions on the audit axis (all SNIPPET: OpenReview abstract field)**
- **Credit Without Ground Truth: Claim-Matched Evaluation of Step-Level Credit with Finite Executed Replay** (ICLR 2027 submission, forum Me3WCeeMWW). This is a revised version of arXiv 2608.19760.
  - Setting: ALFWorld, two independently seeded replay banks, 27,520 continuations over 172 turns. Cross-bank replay-contrast Pearson 0.736, pooled Spearman 0.330.
  - Judge alignment is "modestly positive with an unresolved archived-control increment". Training utility is "unestablished".
  - This is softer than the arXiv v1 "placebo-level" claim, so the null result has been weakened. — [OpenReview Me3WCeeMWW](https://openreview.net/forum?id=Me3WCeeMWW)
- **ACRE: Auditing Step-Level Credit Assignment in Reinforcement Learning for Language Agents** (ACL ARR Aug 2026, HbNTIXwoeU).
  - Protocol: measures agreement of a declared, update-aligned credit observable with a "task-grounded reference". On ALFWorld, six credit signals span fidelity −0.037 to 0.789.
  - Fidelity tracks the three-seed held-out gain ranking (cross-row Spearman 0.886). DAPO has the largest gain despite low fidelity.
  - This contradicts the "credit is placebo" reading of 2608.19760. — [OpenReview HbNTIXwoeU](https://openreview.net/forum?id=HbNTIXwoeU)
- **Same Groups, Different Values: Decomposing Step-Level Credit Assignment for LLM Agents** (ICLR 2027, fBEcJ6y8dK). A component-resolved audit of group-relative step-credit estimators, with replay-verified trajectories and a variance decomposition. Two text environments, no code. — [OpenReview fBEcJ6y8dK](https://openreview.net/forum?id=fBEcJ6y8dK)
- **CEDAR: An Identifiability Audit for Verifier-Defined Credit in Plan Revision** (ICLR 2027, lw07yfdkn8).
  - Finding: in 12,288 cases, verifier-defined credit equals the changed-position set, so a plan-diff baseline recovers it fully.
  - Relevance: the same "is your credit label just the diff?" critique applies to CERT (see Q5). — [OpenReview lw07yfdkn8](https://openreview.net/forum?id=lw07yfdkn8)
- **RMPO: Replay-Measured Turn-Level Credit** (ICLR 2027, ac6BotAkVF).
  - Method: credit by restoring state and applying a localized change. Operators are **leave-one-out action deletion**, prefix progress and same-state replacement. Settings: ALFWorld, WebShop, Search-QA.
  - Deletion credit correlates with symbolic-planner progress (ρ = 0.842). Online, critic-free. — [OpenReview ac6BotAkVF](https://openreview.net/forum?id=ac6BotAkVF)
- **Measuring Counterfactual Credit in a Replayable Wildfire-Suppression Benchmark** (ICLR 2027, yZzXZX35EZ).
  - Uses replay to grade critics' credit. Critic credit RMSE is 8.8–15.6× that of zero credit, even though factual-return R² is 0.91–0.95.
  - Relevance: a strong template for "credit that predicts return is still wrong per action". — [OpenReview yZzXZX35EZ](https://openreview.net/forum?id=yZzXZX35EZ)
- **Individual Credit, Joint Effects** (ICLR 2027, H4QZsZe2eb). In GiGPO, 37.5% of actions with nonzero credit move against their credit after the update, because of other actions' gradients and Adam momentum. Relevance to arm (iii): credit fidelity does not equal update fidelity. — [OpenReview H4QZsZe2eb](https://openreview.net/forum?id=H4QZsZe2eb)
- **CARVE: Replay-Verified Gradient Admission for Offline Agent Controllers** (ICLR 2027, 8yNUqdO1KV).
  - Method: in *offline* agent training, gates each logged decision by replayability and support before it gets actor gradient.
  - Code/SWE success goes from 60.9 to 63.8 with Llama-3.1-70B. An offline, replay-verified, per-step gradient-eligibility decision on SWE. — [OpenReview 8yNUqdO1KV](https://openreview.net/forum?id=8yNUqdO1KV)
- **HarnessSHAP** (ICLR 2027, koYG5gGmCP). Shapley step attribution, with an LLM-judge harness tuned against replay-derived endpoint errors. Domains: API-Bank, HotpotQA, ToolSandbox, Who&When. No code. — [OpenReview koYG5gGmCP](https://openreview.net/forum?id=koYG5gGmCP)
- SWE-side credit methods that are additional sources CERT would have to audit or cite (all SNIPPET):
  - SWE-MILE (uLcqi3rdO9)
  - LLM Value Functions for Credit Assignment (LVRL, U28Rqi5ouc): frozen-LLM value differences, +8.6 pp on SWE-bench Verified, "Monte-Carlo analysis confirms" the critic
  - FocalRL (cWQOoP3ZJp): LLM locator, then state restore, then local repair groups, on SWE-bench Verified
  - ECHO-Graph (mXgWi29erk): provenance DAG credit on SWE-Bench Pro
  - DepGPO ICLR'27 version (FMJWTvDXOP)
  - DeRL-SWE (DL4C'26, MisdGC6Jlh)
  - TRACE-SWE (0YxaeiAIje): execution-evidence step pruning for SWE SFT
  - **Step Rejection Fine-Tuning** (DL4C'26, RndJzeZV81): a critic LLM masks erroneous steps in unresolved SWE trajectories offline. +3.7 versus +2.4 for RFT on SWE-bench Verified.
  - BSR (d6w1RydYch)
  - CRR (NeurIPS 2026, below)

### Inferences
- The ICLR 2027 cycle contains at least ten "credit audited against replay" submissions. A reviewer will treat "we audit step credit against executed ground truth" as a known genre, not a contribution. **INFERRED**
- ACRE and the CWGT revision now disagree on whether fidelity predicts gain. That disagreement is an opening: SWE is a second domain where it could be adjudicated. **INFERRED**

### Gaps
- Per-note fetch and comments/reviews were blocked (challenge), so ICLR 2027 reviews (due Nov 2026) cannot be read. Only one submission PDF was read.
- The API term search is keyword-based. A submission titled without "credit", "process reward" or "SWE" could be missed. NeurIPS 2026 main-track papers are not on OpenReview as public submissions, so only accepted or arXiv'd ones would show.

---

## Q2. Papers that evaluate step credit for code/SWE agents against an interventional or execution-derived ground truth

### Takeaway
Two papers do this on SWE tasks. Neither uses hunk-necessity ground truth or audits the SWE offline credit sources.
- **Counterfactual Supervision (ICLR 2027 submission)**: policy-relative Monte Carlo replay gold on SWE-rebench. It scores LLM credit selectors against that gold with permutation controls and a shuffled arm.
- **What Process Evaluation of Coding Agents Actually Measures (2608.22960)**: replay-based SCAE on SWE-bench Verified file localization. It shows judges have collider bias and that 0/190 step effects survive FDR.

### Cited Findings
**Counterfactual Supervision for Credit Assignment in Language Model Agents (ICLR 2027 submission, QzGGstYqbH). Read via Firecrawl, VERIFIED unless marked.**
- **Gold labels.** Before each observed action the environment is restored. K = 12 same-policy alternatives are sampled, and each action, including the observed one, gets M = 16 continuations. Gold is Â^MC = Q̂(a_t) − mean_j Q̂(a′_j) under a correctness-plus-efficiency utility (−1 fail, +1 efficient voluntary success).
- **Corpus.** 300 trajectories / 5,192 turns from Qwen3.5-4B/9B before and after RL, on SWE-rebench with the mini_swe scaffold. The 185-source annotation corpus requested 792,208 continuations.
- **Credit sources scored against gold.** A frozen DeepSeek-v4.1-Flash retrospective selector, with prompts evolved by a Codex agent against gold feedback. Metrics: negative precision/recall, high-impact weighted sign agreement, NDCG@3, MAE, Pearson. High-impact sign agreement rises from 71.29% (initial prompt) to a higher value after development (exact figure not extracted).
- **Locality controls.** "Within-trajectory permutation… Weighted sign agreement falls from 67.1479% to 58.6562%, and NDCG@3 from 0.534319 to 0.417728." Permuting within trajectory halves gives 59.62%. Disagreements concentrate where |Â^MC| < 0.10 (789 of 1,074).
- **Behavioural audit.** 136 of 143 successful trajectories contain at least one MC-negative turn (32.20% of their turns). Only 78 of 244 "redundant" turns are MC-negative. Sign agreement of behaviour categories with gold is 56.8–68.3%. Point for CERT: semantic error labels (the SWE-Lego / SRFT style) ≠ execution credit.
- **Training.** Online, PPO-style clipped objective. Selector scores are the whole advantage (no critic, no GAE). Arms: Full, **Shuffled**, No success-negatives, No negatives. Baselines: PPO, CriticSearch, CAPO. SWE-bench Pro +4.52 / +4.79 pp over PPO at 4B/9B.
- **Gaps they acknowledge.** Single-run training; they do not isolate "MC-guided development from changes in prompts and inference multiplicity". The locality controls "test prediction, not long-run shuffled-credit training".
- **What it does not do.**
  - No hunk or patch reversion, and no provenance graph.
  - No audit of Orchard LLM-TD, FLARE, DepGPO, SWE-Lego/Nebius masks or SWE-Prime.
  - No offline SFT, and no ESS matching between credit arms.
  - Gold is policy-relative advantage, not policy-independent necessity.
  - **VERIFIED** (absence within this PDF) — [OpenReview PDF](https://openreview.net/pdf?id=QzGGstYqbH)

**What Process Evaluation of Coding Agents Actually Measures (2608.22960). SNIPPET: alphaXiv AI report; header claims ICLR 2027.**
- Defines step-level causal effect as prefix-conditioned θ_t. SCAE estimates it by replay (8 replays × 190 prefixes × 38 instances), with an intervention branch where the policy is near-deterministic.
- Data: 499 file-localization episodes from SWE-bench Verified, Codex CLI. Outcome = recall of the gold file set.
- "0 of 190 intervention-estimated step effects survived BH-FDR correction". Uncertainty is task-level (instance identity explains a 0.640 share of residual replay variance).
- Full-trace LLM judges show collider bias: verdicts shift later (+0.537) when downstream steps are visible, in 59/70 episodes. 0.900 of top-1 verdicts land on gold-file steps.
- Not done: patch editing, training, credit-source comparison beyond judges. — [alphaXiv 2608.22960](https://www.alphaxiv.org/abs/2608.22960)
- The "ICLR 2027" header cannot be an acceptance as of 8 Oct 2026. Treat it as a template artefact or an under-review submission. **INFERRED**

**Counterfactual Rollout Replay, CRR (2609.33875, "Accepted at NeurIPS 2026"). VERIFIED (full text grep).**
- Forks k = 4 decision points per on-policy SWE trajectory and uses R(τ) − R(τ̃) as the step advantage. Online GRPO.
- Appendix N: a dataset of about 1.4M counterfactual tuples over 3,638 tasks will be released, "useful as direct supervision for downstream PRM training, for **verifier evaluation**, and for studying the structure of credit assignment". They pre-announce exactly the evaluation use CERT targets, but have not done it. — [alphaXiv 2609.33875](https://www.alphaxiv.org/abs/2609.33875)

### Inferences
- CERT can no longer claim to be "the first to score SWE step credit against an execution-derived ground truth". Counterfactual Supervision did that on SWE-rebench and found LLM-credit sign agreement of about 67% against a permutation baseline of about 59%. **INFERRED**
- 2608.22960 gives CERT its best *motivation*: stochastic replay credit is underpowered on SWE. Deterministic hunk necessity is a cheap, low-variance, policy-independent alternative ground truth. That is the defensible delta. **INFERRED**

### Gaps
- I did not read the Counterfactual Supervision tables for the post-development selector sign agreement, or the shuffled arm's benchmark numbers. Only the Table-4 probe row ("Shuffled +0.0002 / +0.0166") was seen.

---

## Q3. Hunk/edit ablation or counterfactual replay as the source of step credit

### Takeaway
The ground-truth instrument is pre-empted. **TRIM** (2607.18161) reconstructs the trajectory's surviving edit sequences, removes edit sequences, then files, then atomic edit actions, in reverse trajectory order. It accepts a removal only if tests still pass, on SWE-bench Verified and kernel repair, with a DD-Hunk baseline. That is "revert hunks / trace to trajectory edit steps / rerun tests". Its purpose is patch minimization, not credit or training.

Step-deletion credit in agent RL already exists elsewhere: RMPO's leave-one-out action deletion (ALFWorld etc.), and CRR's and FocalRL's state-restore replay (SWE). IBugDC is already known.

### Cited Findings
- **TRIM.** Minimal behaviour-preserving patch AP\*. CodeSlop = len(AP) − len(AP\*).
  - Traj_R keeps only edits that survive into the final patch.
  - Hierarchical counterfactual search (EDIT SEQ, then FILE, then EDIT ACTION). TRIM-G is one-minimal.
  - On SWE-bench Verified, 333 SWE-agent trajectories: 20.0% ΔSlop on pre-existing files, and oracle correctness preserved for 99.1%.
  - Matches DD-Hunk (32.9 vs 31.5% ΔSlop) at about 1.9× fewer validations.
  - **SNIPPET** (alphaXiv AI report of full text) — [alphaXiv 2607.18161](https://www.alphaxiv.org/abs/2607.18161)
- RMPO: leave-one-out action deletion and same-state replacement as credit operators. Deletion correlates with planner progress at ρ = 0.842. **SNIPPET** — [OpenReview ac6BotAkVF](https://openreview.net/forum?id=ac6BotAkVF)
- FocalRL: an LLM locator finds consequential errors; the method restores state before each one and samples local continuations scored by repair rubrics. Covers SWE-bench Verified. **SNIPPET** — [OpenReview cWQOoP3ZJp](https://openreview.net/forum?id=cWQOoP3ZJp)
- Counterfactual Supervision and CRR: same-state alternative-action replay on SWE (Q2). **VERIFIED**
- No paper found that turns *final-patch hunk necessity* into *per-step credit* via provenance (write → hunk → test), nor one that uses that as a benchmark for other credit sources. Absence across Q1 API terms ("patch hunk necessity", "hunk ablation", "revert edits rerun tests", "necessary edits patch minimal", "edit provenance agent trajectory") and two alphaXiv discovers. **VERIFIED as absence in the indexes searched**

### Inferences
- A reviewer can call CERT's ground truth "TRIM's edit-action necessity, relabelled as credit". The real addition is (a) extending credit from surviving writes to the reads/searches that informed them, through provenance, and (b) using that as an audit target. Part (a) is the weakest link, because it is a heuristic (see Q5). **INFERRED**
- TRIM shows that Traj_R-guided search is cheaper than DD-Hunk. CERT should reuse TRIM's hierarchy rather than claim an algorithm. **INFERRED**

### Gaps
- I did not verify whether TRIM reports any per-step statistics (for example, the fraction of edit sequences that are removable) that would already serve as a "necessity rate" baseline.

---

## Q4. PRM benchmarks for agents, and data attribution for trajectory steps

### Takeaway
None of the agent-PRM benchmarks found uses execution-derived step labels for SWE.
- SWE-PRM: human/taxonomy labels for inefficiency.
- SWE-Shepherd: heuristic action rewards.
- SWE-TRACE: rubric PRM.
- AgentProcessBench, ToolPRMBench, CUARewardBench, PRMBench, ProcessBench: human or rule labels, not SWE.

No step-level training-data-attribution work for SWE-agent SFT was found. This sub-gap is open but probably not worth pursuing.

### Cited Findings
- AgentPRM (2511.08325): promise/progress step rewards for WebShop/browser tasks. **SNIPPET** — [alphaXiv](https://www.alphaxiv.org/abs/2511.08325)
- AgentProcessBench (2603.14465): 1,000 trajectories with 8,509 human-labelled steps, tool-use agents. **SNIPPET** — [arXiv](https://arxiv.org/html/2603.14465v1)
- ToolPRMBench (ICLR'26 VerifAI-2 workshop; 2601.12294): step-level test cases from tool benchmarks. **SNIPPET** — [arXiv](https://arxiv.org/abs/2601.12294)
- SWE-PRM (2509.02360; SEA@NeurIPS'25): taxonomy-guided inference-time PRM, 40.0 → 50.6 on SWE-bench Verified. Labels are not execution-derived. **SNIPPET** — [OpenReview wyrcoDNaGO](https://openreview.net/forum?id=wyrcoDNaGO)
- SWE-Shepherd (2604.10493): "heuristic approach to compute rewards" for action-level annotations. **SNIPPET** — [arXiv](https://arxiv.org/html/2604.10493v1)
- SWE-TRACE (2604.14820): rubric PRM. **SNIPPET** — [arXiv](https://arxiv.org/html/2604.14820v1)
- PRMBench (2501.03124) and ProcessBench (2412.06559) are math only. **SNIPPET** — [alphaXiv PRMBench](https://www.alphaxiv.org/abs/2501.03124), [alphaXiv ProcessBench](https://www.alphaxiv.org/abs/2412.06559)
- Trajectory attribution benchmarks: AgentDojo/Agent3Sigma (2608.06909, not coding), A³Bench (query-conditioned attribution, 2609.33676), LongRCA, Who&When Pro, DeFA, CausalFlow, Causal Agent Replay, REFLECT. These cover failure attribution, mostly LLM or human labels, and none is SWE patch-grounded. **SNIPPET** — [alphaXiv 2608.06909](https://www.alphaxiv.org/abs/2608.06909), [OpenReview jy1eXGt35T](https://openreview.net/forum?id=jy1eXGt35T)
- TraceProbe (2607.06184): **deterministic effect labels** (SURVIVED / REVERTED / FAILED / JUSTIFIED / OFF-ANCHOR) on 2,500 SWE-bench Verified trajectories, with gold-patch anchor sets and hunk-to-function mapping. These are diagnostics, not credit, but "SURVIVED write" is a cheap baseline CERT's ground truth must beat. **SNIPPET** — [alphaXiv 2607.06184](https://www.alphaxiv.org/abs/2607.06184)
- Data attribution: DRIFT, STRIDE, Influcoder and sketched metadifferentiation are example-level or token-level, with no agent-step SWE use. "Data-Weight Directions Rotate Along Training: Horizon-Matched Attribution for GUI Trajectories" (ICLR 2027 submission) is GUI only. **SNIPPET** — [alphaXiv DRIFT](https://www.alphaxiv.org/abs/2606.18307); OpenReview tzJ6kz4bDq

### Inferences
- "First SWE step-credit benchmark with execution-derived labels" is still *literally* unclaimed as a benchmark artefact. But Counterfactual Supervision has a 300-trajectory gold corpus and CRR is releasing 1.4M tuples, so this claim has a short shelf life. **INFERRED**

### Gaps
- I did not check whether SWE-Gym's verifier data or Nemotron-style SWE datasets include per-step execution labels.

---

## Q5. Verdict: closest prior work, remaining delta, reviewer objections

### Takeaway
No single paper kills CERT-SFT, but the field has caught up from three sides at once. The "audit with interventional ground truth" framing **survives only if narrowed** to the following, presented as a measurement paper rather than a method:
- (a) **deterministic, policy-independent** patch-necessity ground truth (low variance, unlike replay, per 2608.22960);
- (b) the **specific offline SWE credit sources** nobody has audited (Orchard LLM-TD, FLARE, DepGPO, SWE-Lego/Nebius masks, SWE-Prime, SRFT);
- (c) **ESS-matched offline SFT** arms.

### Closeness table (0 = unrelated, 3 = already did it)

| Prior work | Venue / date | What overlaps | Closeness | Label |
|---|---|---|---|---|
| Counterfactual Supervision for Credit Assignment in LM Agents | ICLR 2027 submission (QzGGstYqbH) | SWE-rebench coding agents; execution gold turn credit; LLM credit scorers scored vs gold; permutation controls; shuffled arm; training | **2.5** | VERIFIED |
| TRIM (2607.18161) | arXiv Jul 2026 | Trajectory-guided counterfactual removal of edits + test validation on SWE-bench Verified; DD-Hunk baseline | **2.0** (ground-truth instrument) / 0.5 (audit) | SNIPPET |
| What Process Evaluation of Coding Agents Actually Measures (2608.22960) | arXiv Aug 2026 ("ICLR 2027" header) | Replay causal step attribution on SWE-bench Verified; judges audited (collider bias) | **2.0** | SNIPPET |
| Credit Without Ground Truth (2608.19760 + ICLR'27 Me3WCeeMWW) | arXiv / ICLR 2027 submission | Audit framing, shuffled controls, ESS rule, training pilot; ALFWorld only | **2.0** (framing) | SNIPPET |
| ACRE | ACL ARR Aug 2026 | Credit-fidelity audit protocol; fidelity predicts gain; ALFWorld | **1.5** | SNIPPET |
| CRR (2609.33875) | NeurIPS 2026 | SWE fork credit; announces dataset for verifier evaluation | **1.5** | VERIFIED |
| RMPO | ICLR 2027 submission | Leave-one-out action-deletion credit, validated vs planner (ρ = .842) | **1.5** | SNIPPET |
| IBugDC (FSE'24) | known | Hunk-subset enumeration | 1.5 (ground truth only) | inherited |
| Coverage, Not Targeting (2609.02417) | arXiv | Uniform-is-the-bar, shuffled-matched control | 1.5 | inherited SNIPPET |
| CARVE | ICLR 2027 submission | Offline, replay-verified per-decision gradient admission, Code/SWE | 1.0 | SNIPPET |
| Step Rejection FT (SRFT) | ICML'26 DL4C | Offline LLM-critic step masks on SWE (another source to audit) | 1.0 | SNIPPET |
| LVRL, FocalRL, ECHO-Graph, SWE-MILE, DepGPO-ICLR'27 | ICLR 2027 submissions | SWE step-credit methods (sources, not audits) | 0.5–1 | SNIPPET |
| TraceProbe (2607.06184) | arXiv | Deterministic SURVIVED/REVERTED write labels on SWE | 1.0 | SNIPPET |
| AgentPRM / AgentProcessBench / ToolPRMBench / SWE-PRM / SWE-Shepherd | various | Agent PRM benchmarks; labels not execution-derived for SWE | 0.5 | SNIPPET |

### Remaining delta (what CERT can still claim)
1. **A different estimand.** Policy-independent *necessity* of the final artefact, traced to steps, versus the policy-relative advantage of Counterfactual Supervision, CRR and CWGT. It is deterministic and needs no continuation sampling. That avoids the 0/190-significant power failure reported in 2608.22960. **INFERRED**
2. **An audit of the offline SWE credit sources actually used in SFT**: Orchard LLM-TD, FLARE RADAR chains, DepGPO line/file provenance, SWE-Lego/Nebius error masks, SWE-Prime segment scores, SRFT critic masks. None of these has been scored against any execution-derived ground truth (absence across all logs). **VERIFIED as absence in the indexes searched**
3. **ESS-matched offline LoRA arms** on SWE. Counterfactual Supervision's training is online and single-run with no ESS matching. CWGT/ACRE are ALFWorld only. **INFERRED**
4. **Adjudicating ACRE versus CWGT** (does fidelity predict gain?) in a second domain. **INFERRED**

### Likely reviewer objections (ranked)
1. **"Your ground truth is not credit."**
   - Necessity of final-patch hunks says nothing about exploration, diagnosis, or recovery steps. Counterfactual Supervision found diagnosis turns are 38% MC-negative and redundant turns only 32% negative.
   - Read/search credit flows only through a provenance heuristic, which is itself an unvalidated credit rule.
   - Per CEDAR's identifiability argument, write-credit ground truth may reduce to "which edits survived" (TraceProbe's SURVIVED label). A trivial surviving-write baseline could then score near-perfectly.
   - Mitigation: report the surviving-write and diff-only baselines explicitly, and restrict claims to *write and informing-read* credit. **INFERRED**
2. **"Circularity with DepGPO."** DepGPO's write credit *is* line-level provenance to the verifier read set. A provenance-traced ground truth will mechanically favour provenance-based credit. The audit must hold out the provenance edges used by the ground truth, or use a different read-trace method. **INFERRED**
3. **"TRIM already did the instrument, and Counterfactual Supervision already did the audit on SWE."** Novelty then rests on the junction and on the offline-SFT arms. If the audit shows agreement or no difference, the paper has no result (the briefed gating already anticipates this). **INFERRED**
4. **"Low variance in the ground truth."** Many resolved trajectories have single-hunk patches; the project's own notes say Gemma dead-zone patches are often single-edit. Hunk necessity is then trivial (all-necessary), so audit power comes only from multi-hunk cases (about 50% of D4J multi-hunk bugs are divisible per IBugDC). Report the support size up front. **INFERRED**
5. **"Policy-independent necessity is the wrong target for RL/SFT weighting"** (2608.22960's three-level argument). Answer: for offline SFT, the target is which tokens to imitate, and necessity is arguably the right estimand there. Make that argument explicitly. **INFERRED**

### Does the "audit with interventional ground truth" framing survive?
**Partially (about 1.5–2 on novelty).**
- As "first interventional audit of step credit for SWE agents": **no**. Counterfactual Supervision (ICLR'27 submission) and 2608.22960 pre-empt it.
- As "deterministic patch-necessity ground truth vs the offline SWE credit sources used in SFT, with ESS-matched offline arms and explicit trivial-baseline/identifiability controls": **yes, still open as of 8 Oct 2026** in every index searched.

Must-cite list: Counterfactual Supervision (QzGGstYqbH), TRIM, 2608.22960, CWGT (both versions), ACRE, CRR, RMPO, CEDAR, IBugDC, TraceProbe, SRFT, CARVE. **INFERRED**

### Gaps
- Reviews of the ICLR 2027 submissions are not yet public, and per-note pages are challenge-gated.
- Full texts of TRIM, 2608.22960 and ACRE were not read in this session; their details come from AI reports and abstracts.

---

## Search log

| # | Tool | Query / target | Outcome |
|---|---|---|---|
| L1 | Local | `sanity_DECAF.md`, `sanity_Z.md` (grep + §1/search log), PEFT `critic_judgement.md` (grep), graph `critic_judgement.md` + `prior_art_redteam.md` (grep) | Inherited prior art and prior queries; OpenReview had been unsearched in all of them |
| A1 | alphaXiv discover (recency) | "evaluating step-level credit / PRMs for code/SWE agents against interventional or execution-derived ground truth (reverting hunks, counterfactual replay)" | CRR, SWE-MILE, CWGT, CREDO, ASCT, VICT, ExecuCritic, FLARE, A²Agent, Fail-Fast, Rubric-to-Code |
| A2 | alphaXiv discover | "PRM benchmark / step verifiers for agents with execution/intervention labels; step influence in trajectories" | AgentPRM, PRMBench, ProcessBench, AgentProcessBench, 2608.06909, Causal Agent Replay, CausalFlow, DeFA, REFLECT, SkillShapley, **2608.22960**, TrajMark, Who&When Pro, LongRCA, **2607.06184** |
| F1 | alphaXiv full text | CRR 2609.33875 (saved, grepped) | NeurIPS 2026; online; App. N dataset for verifier evaluation |
| F2 | alphaXiv report | 2608.22960 | SCAE replay on SWE-bench Verified; judge collider bias; 0/190 FDR |
| F3 | alphaXiv report | 2607.06184 TraceProbe | Deterministic SURVIVED/REVERTED labels |
| W1 | WebFetch | openreview.net/search?term=step-level credit assignment SWE agent | JS shell only; unreachable |
| W2 | WebSearch (ext.) | site:openreview.net step-level credit assignment software engineering agent counterfactual | iStar, COPPER, counterfactual contribution analysis, STEPORLM; nothing SWE |
| W3 | WebSearch (ext.) | site:openreview.net process reward model SWE-bench step labels execution ground truth | SWE-RM, SWE-RL, FreePRM; nothing matching |
| O1 | OpenReview API search | credit assignment software engineering agent; step-level credit audit; counterfactual credit coding agent; process reward SWE agent; hunk ablation | DeRL-SWE, SWE-MILE, CEDAR, ACRE, CWGT-ICLR'27, Same Groups, **Counterfactual Supervision**, SWE-AaaV, AdaStep, etc. |
| O2 | OpenReview API search | credit assignment SWE-bench; process reward coding agent; step credit ground truth; counterfactual replay agent credit; credit assignment uniform random baseline agent; revert edits rerun tests; influence function agent trajectory; data attribution agent trajectories; step-level verifier benchmark agent; process reward benchmark agents; offline credit assignment SFT agent; SWE agent trajectory step reward; necessary edits patch minimal; credit assignment coding agents evaluation | LVRL, FocalRL, ECHO-Graph, RMPO, CARVE, HarnessSHAP, wildfire replay benchmark, TRACE-SWE, SWE-PRM, ToolPRMBench, CUARewardBench, ABSeeker-ICLR'27, DepGPO-ICLR'27, A³Bench, etc. |
| O3 | OpenReview API notes?id= | 14 forum IDs | 403 challenge; switched to title-term search to retrieve abstracts |
| O4 | OpenReview API (title search, abstracts) | CWGT-ICLR'27, CEDAR, ACRE, Same Groups, Counterfactual Supervision, SWE-AaaV, RMPO, LVRL, TRACE-SWE | Abstracts read (SNIPPET) |
| O5 | OpenReview API (abstracts) | HarnessSHAP, A³ attribution, EGCA-self-evolving, CARVE, wildfire replay, DeRL-SWE, DepGPO-ICLR'27, ECHO-Graph, failed-trajectory repair, SWE-PRM, SWE-eval, FocalRL | Abstracts read; one 429 then recovered |
| O6 | OpenReview API search (paced) | patch hunk necessity; edit provenance agent trajectory; step weighting fine-tuning coding agent; credit fidelity; shuffled credit control; SWE process reward benchmark; step labels SWE trajectories; key steps SWE trajectories; trajectory pruning SWE fine-tuning; process supervision software engineering agents; offline credit software engineering; counterfactual step coding agent | Individual Credit Joint Effects, State-Grounded credit, SOAP, DSC-Gate, SRFT, TRAJEVAL, BSR, TrajFix; no hunk-credit paper |
| O7 | OpenReview API (abstracts) | Individual Credit Joint Effects, State-Grounded DCE, SOAP, DSC-Gate, Learning after Execution, SRFT, TRAJEVAL, BSR, Barriers to CCA | Abstracts read |
| W4 | WebSearch | "Counterfactual Supervision for Credit Assignment in Language Model Agents" | No arXiv version found (CausalFlow, C3, CCPO returned instead) |
| W5 | WebFetch | openreview.net/pdf?id=QzGGstYqbH | Browser challenge; unreadable |
| FC1 | Firecrawl scrape (pdf parser) | openreview.net/pdf?id=QzGGstYqbH | **23-page PDF read** (grep + §2, §4, §6, §7, App. B, D.1) |
| A3 | alphaXiv discover (recency) | "reverting hunks / edits of an agent's final patch and rerunning tests…; patch minimization / edit necessity for SWE trajectories" | **TRIM 2607.18161**, minimal-edit fidelity, Refine-After-Generation, SWE-Prime, Fail-Fast |
| A4 | alphaXiv discover | "training data attribution / influence at step level of agent trajectories for SFT" | 2608.06909, SkillShapley, A³, DRIFT, STRIDE, PG-SFT; no SWE step-level TDA |
| W6 | WebFetch | arXiv abs 2607.18161 | TRIM abstract (paraphrased) |
| W7 | WebFetch | arXiv abs 2608.06909 | Trajectory attribution benchmark; AgentDojo/Agent3Sigma; not coding |
| F4 | alphaXiv report | TRIM 2607.18161 | Hierarchical counterfactual edit removal; SWE-bench Verified; DD-Hunk baseline |
| W8 | WebSearch (ext.) | PRM benchmark SWE agent trajectories step-level labels SWE-Shepherd / SWE-PRM / SWE-TRACE | SWE-Shepherd heuristic labels; SWE-TRACE rubric PRM; SWE-PRM taxonomy; AgentProcessBench, ToolPRMBench, CUARewardBench |
| W9 | WebSearch | ACRE auditing step-level credit … arXiv | No arXiv version of ACRE found; CWGT mirrors |

**Not searched or not reachable:**
- ICLR 2027 reviews and per-note pages (challenge-gated).
- NeurIPS 2026 main-track papers not posted to arXiv.
- Semantic Scholar or Google Scholar citer graphs of TRIM, CRR and 2608.19760.
- Full texts of TRIM, 2608.22960, ACRE, RMPO and CARVE.
- An arXiv twin of the Counterfactual Supervision submission (none found).
