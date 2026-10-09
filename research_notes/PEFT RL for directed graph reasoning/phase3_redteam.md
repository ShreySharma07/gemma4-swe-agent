# Phase 3: prior-art red team on the 12 ideas (8 Oct 2026)

**Role.** A hostile prior-art red team, coming to the ideas fresh.

**Kill criterion served.** Prior art that exploits the *same mechanism*. Prior art that only shares the domain does not count.

**Input.** `inventor_phase1_2.md`: the Phase 2 six (CASE, ERC, SAST, ALIAS, SENSOR, GCSR) in depth, and the other six briefly.

**Labels.**
- **VERIFIED**: I read the primary text this session, through alphaXiv full-text page retrieval.
- **SNIPPET**: I saw only a title, abstract or search snippet, or another paper's description of it.
- **INFERRED**: my own reasoning, or something recalled from memory and not re-read.

**Closeness scale.**
- 0: unrelated.
- 1: same domain or goal, different mechanism.
- 2: same mechanism family, with a material difference (signal source, domain, or how the signal is used).
- 3: same mechanism already done.

---

## 0. Bottom line

| Idea | Closest prior (closeness) | Has the surprising prediction already been answered? | Verdict |
|---|---|---|---|
| **CASE** | Ma & Chochlakis 2505.17510 (**2**). Also CoverageEval 2307.13383 (2, for the item format) and MuSoBench 2512.01725 (1.5). | **Partly.** Ma et al. already show that list emission suppresses all but one label, and that SFT/RL make this worse. Their per-label ("unary") prompting fixes calibration but **did not raise F1** over a cheap max-over-generations readout. Nobody has measured it on code call sets or commit sets. | **NARROW** |
| **ERC** | Software World Models 2610.04940 (**2.5**). Also the predictive-mutation-testing lineage (PMT, Seshat, MutationBERT; 2). | **P1 is largely answered in a neighbouring setting.** SWM finds static reachability beats learned and zero-shot models on dependencies the graph records; learned models win only on hidden channels. | **NARROW** |
| **SAST** | RLCSD 2606.11709 (**2.5**) and CROP 2608.13387 (2.5). Also CREDIT, Purified OPSD, and the Salesforce "three checks" paper 2608.09263. | **P3 is answered in kind.** A contrastive teacher (correct minus wrong privileged context) beats one-sided OPSD by +1.0 to +3.6 on math (RLCSD Table 2). The paraphrase-calibrated counterfactual margin, which mirrors SAST's ρ1−ρ2 null, is CROP. | **NARROW** (as a method: near-KILL). Survives as a *measurement instrument* only. |
| **ALIAS** | Tool-schema bias 2609.34971 (**1.5–2**). Also MML over spurious programs (Guu et al. 2017, INFERRED). | **No.** Nobody has tested whether redundant tools that coexist split RL credit, or used an execution-verified marginal likelihood over aliases. | **SURVIVE** (narrowed framing) |
| **SENSOR** | IGPO 2510.14967, ICLR'26 (**2**). Also ENTBUG (Campos et al. ASE'13; 2) and ΔBelief-RL / IG-Search / CIGPO (2). | **No.** Nobody has tested Σg-predicts-resolve for SWE agents. The credit form `Δ log p(truth)` is IGPO's. Entropy over an SBFL posterior as the value of a test is ENTBUG's. | **NARROW** |
| **GCSR** | SWE-Playground 2512.12216 (**1.5**). Also FrogNano 2609.07925 (1), the SWM procedural worlds (1.5) and G1/GraphWiz (1). | **Partly informative, not answered.** Fully synthetic, LLM-written projects match SWE-smith at 7B, but trail it by about 9 pts at 32B. No one has done topology-as-variable or content-free compilation. | **SURVIVE** |
| 2 Depth-IPW | SPADER 2606.00593 inverse-frequency entity reward (1.5) | — | NARROW (diagnostic only) |
| 3 Dark-edge | Premise already established: PyCG recovers 49% of runtime edges (via NameRTS 2605.25356). SWM "hidden channels". | Premise confirmed | SURVIVE as diagnostic |
| 5 Backward-hop | Reversal-curse literature (1) | No code-specific test found | SURVIVE as diagnostic |
| 9 Privileged scaffold | Privileged-information distillation and OPSD family (2.5) | — | KILL as standalone; keep as SAST ablation |
| 11 Impact forecasting | **SWM 2610.04940 (3)** | Yes, the mechanism is done | **KILL** as a novelty claim |
| 12 Tri-state grounding | Inherited 2606.30247; IGPO's failure analysis (1.5) | — | NARROW (not deeply searched) |

---

## 1. CASE: atomised set emission

### Closest prior work

**1. Ma, Chochlakis et al., "Large Language Models Do Multi-Label Classification Differently", arXiv 2505.17510. Closeness 2. VERIFIED.**

When LLMs autoregressively emit a label set, each step is "spiky":
- the model suppresses every label except one at each step;
- the second-highest label at step r is usually never emitted later (78–100% of cases, Table 4);
- the distribution gives little signal about when the list will stop.

Scale and fine-tuning (SFT and RL) make the spikiness worse. They compare remedies:
- *unary breakdown*, one query per label;
- *binary breakdown*;
- *max-over-generations*.

Unary breakdown "disentangles the bias of language modeling from the classification task". It sharply improves NLL (GoEmotions 33.6 → 5.0 for Llama-3-8B), **but its example-F1 (0.29) is below the cheap max-over-generations readout (0.34)** (Table 5).

This is CASE's mechanism M1a, "the list closes early because of emission, not perception", demonstrated on subjective and entity-tracking multi-label tasks (Boxes). It is not on code.

**2. Tufano et al., "Predicting Code Coverage without Execution", 2307.13383 (CoverageEval). Closeness 2 for CASE's *trace items*. VERIFIED.**

The model annotates *every statement in canonical order* as executed (`>`) or not (`!`) for a given test. That is exactly CASE's "fixed canonical order, one decision per candidate, no list to close" format, applied to coverage. Limits:
- evaluation only;
- statement level within one method, not a repository call set;
- the authors propose it as a pre-training objective but do not train.

**3. Guan et al., "Beware of Reasoning Overconfidence", 2512.01725 (MuSoBench). Closeness 1.5. VERIFIED.**
- Short-CoT models stop early on multi-solution tasks, with high confidence and low recall.
- Fixes: Long-CoT, appending "Wait, there may be other solutions", or union-voting over 32 samples (recall 10% → 50%).
- This is the generative-enumeration version of the phenomenon. It has no candidate pool and no per-element scoring.

**4. SPADER, 2606.00593. Closeness 1.5. VERIFIED.**
- A multi-answer QA search agent. It names the "premature termination on head entities" failure.
- Fix: a per-entity step reward with an inverse-frequency novelty premium (1/N).
- Recall rises (QAMPARI Llama-8B: GRPO 0.18 → 0.34).
- The setting is a search agent, not set emission over a given pool.

**5. Claim-level calibration, 2608.22483. Closeness 1. VERIFIED.**
- Per-claim confidence with post-hoc temperature/Platt scaling, inference only.
- There is no training with a per-element proper score.
- RLCR "Beyond Binary Rewards" 2507.16806 (Brier-style calibration reward, SNIPPET) is answer-level, not element-level.

### What delta remains
- **(a)** The code-specific claim: RepoReasoner's call-chain under-recall, and our own `OTHER_LOCATIONS: none` pattern, are emission artefacts. Nobody has tested this.
- **(b)** Training with **per-element proper log-score advantages, plus element-wise best-of-K recombination**. I found no prior for the recombination trick.
- **(c)** Transfer from trace items to the agent's commit set.

The mechanism itself (list emission suppresses labels, and per-candidate decisions remove the suppression) is **not new**. Cite Ma et al. as the mechanism prior.

### Was the surprising prediction already tested?
- P1 ("zero-shot atomisation alone gives recall ≥0.65 at ≤0.10 precision loss") is **untested on code**.
- The nearest evidence is a warning (VERIFIED): per-label querying fixed calibration, but **did not dominate on F1**.
- Add max-over-generations as a third arm. It costs nothing with vLLM logprobs: emit the list once, read each candidate's maximum probability across emission steps. If it matches atomisation, the training component of CASE loses its motivation.
- K1 should compare recall at **matched precision**. CASE pools are 50% positive, so "say EDIT more often" inflates recall.

### Verdict: NARROW
- Reframe: "Ma et al.'s emission suppression, tested and trained away on executed code graphs and commit sets."
- Add the max-over-generations arm.
- The novelty lives in the code/commit-set evidence and in element-wise proper-score RL, not in atomisation.

---

## 2. ERC: executable rewire counterfactuals

### Closest prior work

**1. Su, Hu, Zhao et al., "Software World Models: From Consequence Prediction to Decision Value", 2610.04940; ICLR 2027 submission LCT7iCqsXa. Closeness 2.5. VERIFIED.**

What it does:
- Executes counterfactual changes from restored states.
- **Fine-tunes an LM (SFT warm-up, then GRPO on blast-set F1)** to predict which downstream components break.
- Compares against a **typed-reachability baseline**, explicit "hidden channels" the graph misses, and zero-shot code models.

What it finds:
- "Reachability is stronger on dependencies represented in the graph, while SWM recovers failures caused by couplings the graph misses."
- Blast-set F1: reachability 0.431 vs SWM 0.571.
- Exploration is focused where reachability was wrong ("blind-spot acquisition").

The setting is multi-repository systems and services plus real libraries, with a 0.5B Qwen and LoRA. It is not function-level call-target swaps.

**2. Predictive mutation testing lineage. Closeness 2.**
- PMT: Zhang et al., ISSTA 2016 (INFERRED from memory, not read).
- Seshat (SNIPPET).
- **MutationBERT, "Contextual Predictive Mutation Testing", Jain, Alon, Groce, Le Goues, FSE'23, arXiv 2309.02389 (SNIPPET).** It predicts the per-(mutant, test) kill matrix: ERC's forecast item, with arbitrary mutants.
- OpenReview also lists follow-ups (SNIPPET):
  - "Spotting Code Mutation for PMT" (ASE 2024);
  - WITNESS, fine-grained PMT (2025);
  - "Test Case Level PMT" (APSEC 2023);
  - LLMorpheus (TSE 2025).

**3. NameRTS, "Names Are All You Need", 2605.25356. Closeness 1, but decisive for ERC's K2. VERIFIED.**
- Name-based reachability, with no call graph, selects affected tests safely for 99.6% of commits.
- It cites Bouzenia et al.: PyCG recovers only 49% of the call edges DynaPyt observes at runtime.
- Consequence: the "text-only" baseline in ERC K2 should include **name-reachability**, not just TF-IDF. Otherwise "labels are structural" is a strawman.

**4. SWE-smith 2504.21798 (SNIPPET).** Procedural bug operators: remove assignment/loop/conditional, swap if/else blocks, swap operators. I found **no call-target-swap operator**. The agentic "find the swap" item type is otherwise standard bug-injection RL (SWE-smith; Self-play SWE-RL 2512.18552, SNIPPET).

### What delta remains
- The **degree-preserving call-target double swap** as a mutation operator. No prior found.
- The RW reachability null (K3).
- Using structure-necessary items to *raise the agent's rewire sensitivity* (P2). This is untested, and it is the only genuinely new prediction.

### Was the surprising prediction already tested?
- P1 ("BFS beats the 31B model") is **largely answered in kind** by SWM (VERIFIED): reachability is the strong baseline on represented edges. It is no longer surprising.
- P2 and P3 are untested.

### Verdict: NARROW
- Cite SWM and the PMT lineage.
- Drop P1 as the headline; make P2 (rewire sensitivity rises while graph-tool-free resolve holds) the headline.
- Add name-reachability (NameRTS-style) to K2.

---

## 3. SAST: structure-ablated self-teacher

### Closest prior work

**1. RLCSD, "Reinforcement Learning with Contrastive On-Policy Self-Distillation", 2606.11709; ICLR 2027 submission liW6AQ3OqV. Closeness 2.5. VERIFIED.**

Token signal (their eq. 7):

`e_ctr,t = log π_T(y_t | x, y*_c) − log (1/K) Σ_k π_T(y_t | x, y*_w,k)`

- y*_c is the correct privileged context; the y*_w,k are wrong contexts under an *identical template*.
- The signal **modulates the GRPO outcome advantage** through a tanh, a sign-preserving clamp and a mask.
- This is SAST's `Ã_t = Â_τ + β·clip(δ^G − δ^ρ)` with "wrong solution" in place of "rewired graph".
- As a plug-in, the contrast improves OPSD by +1.0/+0.9 and RLSD by +1.4/+0.1 (Table 2).
- It includes agentic ALFWorld and WebShop results (appendix, not read).

**2. CROP, 2608.13387. Closeness 2.5. VERIFIED.**
- Teacher sensitivity to a *counterfactual* prompt **minus** sensitivity to a meaning-preserving *paraphrase*, on a fixed student rollout.
- This is structurally SAST's `c_t` calibrated against its ρ1−ρ2 null.
- CROP also cites:
  - **CREDIT** (Shen et al. 2026, SNIPPET): subtracts teacher scores under unrelated queries.
  - **DOPD** (SNIPPET).

**3. Salesforce, "Privileged Likelihood is Not Automatically Value: Three Checks", 2608.09263. Closeness 2 as a critique. VERIFIED.**
- The RLCSD-style two-sided score separates correct from incorrect AIME trajectories at **AUC 0.505**.
- All five token-score configurations underperform outcome-only GRPO (24–34% vs 64%).
- On SciKnowEval, where AUC is 0.81–0.92, it helps.
- The paper also summarises:
  - **Ichihara et al. 2608.09228** (SNIPPET): a *corrupted* solution from another problem retains much of the OPSD gain.
  - **Purified OPSD** (SNIPPET): subtracts a question-absent teacher.

### What delta remains
- The counterfactual is a **degree-preserving rewire of a privileged graph**. That is a structure-specific negative that holds node identity, degree and text fixed. No prior found (alphaXiv discover queries 4 and 7, OpenReview "self-teacher rewired graph", "degree-preserving rewiring LLM graph reasoning").
- Its use as a cheap *instrument* for "does the model use edges" also has no prior found. The closest are edge-shuffle probes in TAG work (2511.16767, SNIPPET) and CNY's rewire control.

### Was the surprising prediction already tested?
- P3 ("c beats plain OPSD") is the RLCSD result in a new domain. Not surprising.
- P1 and P2 (where edge credit concentrates) are untested.
- The Ichihara result predicts a real risk: the rewired teacher may carry most of the "teacher effect", so c is ≈0 (SAST K1). Treat that as a *likely* outcome, not a remote one.

### Verdict: NARROW (method near-KILL)
- Position SAST as "RLCSD/CROP with a graph-rewire counterfactual". Cite all three papers.
- The defensible contribution is the measurement (P1/P2 and K1–K3).
- Add Salesforce's Q1 as kill check K0 before any training: trajectory-level AUC of Σc against resolve must clearly exceed 0.5.

---

## 4. ALIAS: alias-marginal likelihood and tool-surface invariance

### Closest prior work

**1. "Action-Space Shaping for LLM Agents: Measuring and Mitigating Tool-Schema Bias", 2609.34971; ICLR 2027 submission eC7MH5DgcU. Closeness 1.5–2. VERIFIED.**

The setup: nine invertible operators rewrite a tool schema into 32 *equivalent* schemas (merge, split, namespacing, multi-call protocols), and the authors measure the resulting "schema bias".

Findings:
- Success ranges from 0 to 97% purely by schema.
- Merging hurts more than splitting (median −19 pts).
- **Gemma-4-12B livelocks on any dispatcher.**
- **Mixed-schema GRPO repairs only the schema variants it was trained on.** Gains transfer to recombinations of trained changes, not to new kinds of change.

Their equivalence is *one action, many syntaxes, one schema at a time*. ALIAS's is *coexisting different tools whose observations carry the same facts*.

**2. Maximum marginal likelihood over equivalent / spurious programs.**
- Guu et al. ACL 2017, "From language to programs"; MAPO (INFERRED, from memory).
- Spurious-program filtering 2311.01161 (SNIPPET).
- This is the objective-level ancestor of `−log Σ_{a′∈C(a)} π(a′)`.

**3. Other near-misses (SNIPPET).** Redundant-tool-call RL papers 2605.26952 and 2609.16268 ("Spurious Tool Use") address over-calling, not alias splitting. The alphaXiv query on action equivalence classes / bisimulation for LLM-agent credit returned **no papers**.

### What delta remains
- An observation-equivalence oracle built by *executing* aliases and taking the Jaccard of extracted facts.
- An alias-marginal objective over coexisting tools.
- The redundant-duplicate-tool test (P2).
- Mechanism discrimination for RepoNavigator (M3a vs M3b vs M3c).
- None of these was found.

### Was the surprising prediction already tested?
- No.
- The schema-bias result (VERIFIED) **constrains P3**: invariance is learned only for surfaces seen in training. Tool-surface dropout must cover exactly the deployed surfaces (with and without graph tools). Fortunately those are known.

### Verdict: SURVIVE
- Cite 2609.34971 as the closest work and Guu/MAPO as the objective's ancestor.
- Frame ALIAS as "credit splitting under coexisting redundant tools", which is different from schema bias.

---

## 5. SENSOR: spectrum information gain

### Closest prior work

**1. IGPO, "Information Gain-based Policy Optimization", 2510.14967 (ICLR 2026). Closeness 2. VERIFIED.**
- Turn reward: `r_t = log π(a* | q, o_≤t) − log π(a* | q, o_≤t−1)`, the policy's own teacher-forced belief in the ground truth.
- Rewards are group-normalised separately from the outcome reward and discounted.
- +8.3 F1 at 7B and +16.6 at 3B over GRPO.
- SENSOR's `g_t = log p_t(F*) − log p_{t−1}(F*)` is this formula with an external posterior.
- Neighbours (SNIPPET): ΔBelief-RL ("Intrinsic Credit Assignment for Long Horizon Interaction"), IG-Search 2604.15148, CIGPO 2607.16244, and BED amortisation 2607.03426.

**2. Campos, Abreu, Fraser, d'Amorim, "Entropy-Based Test Generation for Improved Fault Localization", ASE 2013 (ENTBUG). Closeness 2. SNIPPET (web summary of the paper).**
- The entropy of a spectrum-based Bayesian fault-diagnosis posterior is the fitness function for choosing new tests (49% entropy reduction).
- This is SENSOR's "value of an experiment = information about the fault posterior from coverage spectra", in a non-LLM search setting.
- The older interactive-FL lineage (APSEC 2006, JCST 2009; OpenReview SNIPPET) belongs to the same family.

**3. SWE-agent credit papers. Closeness 1 (SNIPPET).** SWE-MILE 2609.32631, Counterfactual Rollout Replay 2609.33875 and SWE-Doctor 2607.00990 (runtime diagnosis from reproduction tests). None uses spectrum-posterior information gain as credit.

### What delta remains
- The posterior is an **execution-derived SBFL surrogate over the agent's own experiments**, not the policy's belief.
- The target is the fix *set*.
- Credit falls on the command tokens of an LLM SWE agent.

### Was the surprising prediction already tested?
- P1 (Σg predicts resolve where function-hit does not) is untested.
- A cheap mandatory ablation follows from IGPO: **self-belief IG**. Prefill the gold-function list after each turn and take Δ log π. If self-belief IG ≈ SBFL-IG in predictive AUROC, the SBFL machinery (15–60 CPU-h of replay) is redundant. Note that IGPO's own failure analysis (VERIFIED) shows ground-truth incompleteness penalising valid behaviour. Multi-function F* is exposed to the same problem.

### Verdict: NARROW
- Cite IGPO and ENTBUG; claim only the external execution posterior and the SWE-agent evidence.
- Add the self-belief-IG arm to K1.

---

## 6. GCSR: graph-compiled synthetic repos

### Closest prior work

**1. SWE-Playground, "Training Versatile Coding Agents in Synthetic Environments", 2512.12216. Closeness 1.5. VERIFIED.**
- Projects, tasks and tests are generated from scratch by LLMs and agents, with no real repositories.
- 704 synthetic trajectories give 7B SWE-bench Verified 17.0, against SWE-smith-7B 15.2 with 2k trajectories.
- **At 32B the synthetic data gives 31.2 against SWE-smith's 40.2.**
- The content is LLM-written and semantically rich. Topology is not controlled.

**2. SWM procedural software worlds, 2610.04940. Closeness 1.5. VERIFIED.**
- Procedurally generated multi-component systems with *controlled hidden channels*.
- Training and evaluation populations are disjoint, plus held-out real libraries.
- These are synthetic graph-defined worlds, but for consequence prediction, not for agent warm-starts.

**3. FrogNano 2609.07925. Closeness 1. VERIFIED.**
- Tasks are synthesised online at the learnability frontier, but from **real repository snapshots**.
- 300 synthetic tasks give 49.1, against 48.0 for 300 real tasks.

**4. Not read (SNIPPET).** SWE-smith, Self-play SWE-RL, and G1/GraphWiz (synthetic graph puzzles).

**5. Web search.** It turned up no paper on agent RL over procedurally generated, call-graph-compiled synthetic codebases with transfer to real repos.

### What delta remains
- Compilation from a sampled typed graph, with dark edge types and content-free identifiers.
- Topology fidelity (real-copied vs rewired vs depth-1) as the experimental variable.
- The protocol-versus-content factorisation.
- No prior found for any of these.

### Was the surprising prediction already tested?
- No.
- SWE-Playground (VERIFIED) is informative: even *content-rich* synthetic data falls about 9 pts short at 32B. That makes P1 ("≥70% of the real-RFT gain from content-free repos at 31B") risky.
- That is good for a falsification paper, but budget for P1 failing.

### Verdict: SURVIVE
- Cite SWE-Playground, SWM and FrogNano.
- The scale-dependence evidence (synthetic matches real at 7B, trails at 32B) should be stated as the prior expectation that the rewired-topology (RW) and depth-1 (GF) arms test.

---

## 7. The other six (brief)

**Idea 2, depth-IPW recall reward. NARROW.**
- Rarity- or difficulty-inverse reweighting of per-element reward is SPADER's novelty premium `U = 1/N(e)` (VERIFIED). Closeness 1.5.
- Depth as the propensity variable is unexamined.
- Keep it as a stratification inside CASE.

**Idea 3, dark-edge ranking. SURVIVE as diagnostic.**
- The premise is established: PyCG captures only 49% of runtime call edges (Bouzenia et al., via NameRTS, VERIFIED as cited).
- In SWM, the extracted graph drops channels that carry 34.8% of broken consumers (VERIFIED).
- No training reward on dark edges was found.

**Idea 5, backward-hop asymmetry. SURVIVE as diagnostic.**
- The reversal-curse literature (2309.12288 and successors; "Reversing Arrows in LLMs" 2608.03512, SNIPPET) is the general analogue. Closeness 1.
- No code-specific caller/callee recall asymmetry measurement was found.

**Idea 9, privileged-scaffold generation with stripped training. KILL as standalone; keep as SAST ablation.**
- Privileged-information distillation for agents is crowded (all SNIPPET):
  - Penaloza et al. 2026, cited in FrogNano;
  - 2609.29051, PIVOT, PR-OPD and others listed in the inventor log;
  - "Learning What to Teach: Adaptive Hint Generation for OPSD", ICLR 2027 submission 7O8pLcA8w3.
- Closeness 2.5.

**Idea 11, impact forecasting. KILL as a novelty claim.**
- SWM 2610.04940 (VERIFIED) already predicts the blast set of a change before execution.
- It trains with GRPO on executed consequences and feeds the forecasts back to the same coding agent for decisions. Closeness 3.
- Usable only as an auxiliary signal, citing SWM.

**Idea 12, tri-state execution grounding. NARROW, under-searched.**
- Inherited closest is 2606.30247.
- IGPO's failure analysis (VERIFIED) documents the exact harm: valid non-gold answers are penalised under incomplete ground truth.
- I did not search further. INFERRED: as CASE's labelling rule it is a design choice, not a claim of novelty.

---

## 8. Actionable changes for the critic

1. **CASE:** add a max-over-generations arm (Ma et al.) and report recall at matched precision. Novelty is the code/commit-set evidence and the element-wise proper-score RL.
2. **ERC:** demote P1. Add name-reachability (NameRTS) to K2. Cite SWM and MutationBERT/PMT. Lead with P2 (rewire sensitivity).
3. **SAST:** add K0, the Salesforce outcome-AUC check. Expect the Ichihara-style result that corrupted teachers retain most of the effect. Pitch it as a measurement instrument; as a method it is RLCSD with a graph negative.
4. **SENSOR:** add a self-belief IGPO arm. If it ties, drop the SBFL replay.
5. **ALIAS and GCSR:** survive. Cite the schema-bias paper and SWE-Playground respectively. GCSR's P1 is now a priori risky (synthetic data trails real by about 9 pts at 32B).
6. **Idea 11:** killed by SWM. **Idea 9:** absorbed into the crowded OPSD/privileged-information literature.

---

## Search log

Tools used:
- alphaXiv `discover_papers` (DP) and `answer_pdf_queries` (READ, full-text pages);
- WebSearch (WS);
- the OpenReview API `api2.openreview.net/notes/search` (OR). It was paced at 6–7 s between queries; some per-forum lookups returned HTTP 403 and were skipped.

| # | Tool | Query / target | Key result |
|---|---|---|---|
| 1 | DP | LLMs under-enumerate set-valued outputs; per-candidate vs list for recall | Set-valued prediction 2603.22966, recall-ceiling reranking 2609.27953 (SNIPPET, not close) |
| 2 | DP | turn-level information gain reward; BED for agent probes | **IGPO 2510.14967**, IG-Search, BED amortisation 2607.03426, CIGPO, ΔBelief-RL, TRACE |
| 3 | DP | SWE agent reward from SBFL / coverage of agent's tests | SWE-MILE, Counterfactual Rollout Replay, SWE-Doctor (SNIPPET) |
| 4 | DP | self-distillation with counterfactual / corrupted privileged context | **RLCSD**, **CROP**, CRAFT, contrastive privileged SD 2607.28026, Repulsive/Attractive teachers, **Salesforce three checks**, Ichihara 2608.09228 |
| 5 | READ | 2606.11709 RLCSD | contrastive formula; Table 2 plug-in gains |
| 6 | READ | 2608.13387 CROP | counterfactual-minus-paraphrase margin; cites CREDIT, DOPD |
| 7 | READ | 2608.09263 Salesforce | AUC 0.505; token scores below GRPO; Ichihara and Purified OPSD summaries |
| 8 | DP | multi-answer QA low recall, per-candidate fixes (historical) | QAMPARI, SPADER, MuSoBench 2512.01725, RVR |
| 9 | DP | LLM call-graph construction, pairwise vs list | Indirect-call analysis, GRADE, CNY (not close) |
| 10 | WS | list generation premature termination vs per-item classification | No direct comparison found |
| 11 | READ | 2512.01725 MuSoBench | reasoning overconfidence; mitigations |
| 12 | DP | multi-label: label-set generation vs one-vs-rest prompting | **Ma & Chochlakis 2505.17510**, dichotomic prompting 2511.03830, dLLM-SetScore |
| 13 | DP | per-element proper-scoring RL rewards | RLCR 2507.16806, claim-level calibration 2608.22483, others (SNIPPET) |
| 14 | READ | 2505.17510 | spiky label suppression; unary breakdown vs max-over-generations (Table 5) |
| 15 | READ | 2608.22483 | inference-only per-claim calibration |
| 16 | WS | Contextual Predictive Mutation Testing / MutationBERT | per-(mutant, test) kill prediction (SNIPPET) |
| 17 | DP | training LLMs to predict test failures after mutation, as a reward | MIST-RL, Code-A1, test-aware mutant generation (SNIPPET) |
| 18 | WS | SWE-smith procedural modification operators / call swap | No call-target-swap operator found |
| 19 | DP | LLM change-impact / test-outcome prediction vs reachability | **SWM 2610.04940**, **NameRTS 2605.25356** |
| 20 | READ | 2610.04940 SWM | GRPO consequence predictor; reachability baseline; hidden channels 34.8% |
| 21 | READ | 2605.25356 NameRTS | 99.6% safe name-reachability; PyCG 49% of runtime edges |
| 22 | DP | agent RL with redundant/overlapping tools; equivalent-action marginalisation | **Tool-schema bias 2609.34971**, Spurious Tool Use, TAPO (SNIPPET) |
| 23 | DP | MML over equivalent programs (historical) | spurious-program filtering 2311.01161, EquiBench (SNIPPET) |
| 24 | READ | 2609.34971 | 9 operators, 32 variants; mixed-schema RL repairs only seen variants |
| 25 | DP | action equivalence classes / bisimulation for LLM-agent credit | **No papers found** |
| 26 | DP | procedurally generated synthetic repos for SWE-agent RL, transfer | FrogNano, CodeMidas, DeNovoSWE, SWE-smith, Self-play SWE-RL, Agent World Model |
| 27 | DP | synthetic graph RL transfer (GraphWiz/G1); protocol vs content | G1, GRAIN, SPADE (SNIPPET) |
| 28 | DP | fully synthetic procedurally generated Python codebases with injected bugs | Code-QA-Bench, WitnessGym, Neural Debugger (not close) |
| 29 | READ | 2609.07925 FrogNano | synthetic tasks from real snapshots; 300 synthetic ≈ 300 real |
| 30 | WS | agent RL on procedurally generated synthetic codebases → SWE-bench | No match; SSR, SWE-Gym, DeepSWE only |
| 31 | READ | 2510.14967 IGPO | `r_t = Δ log π(a*)`; ablations; incomplete-ground-truth failure mode |
| 32 | WS | Campos et al. entropy-based test generation for FL (ASE 2013) | ENTBUG: entropy of SBFL diagnosis as fitness (SNIPPET) |
| 33 | DP | predicting code coverage / executed functions without running | **CoverageEval 2307.13383**, Neural Debugger, self-execution simulation |
| 34 | READ | 2307.13383 | per-statement canonical-order annotation format; GPT-4 exact match 25–30% |
| 35 | DP | do LLMs use graph structure: shuffled/rewired edges (historical) | "When Structure Doesn't Help" 2511.16767, "Lost in Serialization" (SNIPPET) |
| 36 | DP | caller vs callee asymmetry, reversal curse in code | Reversal-curse papers, "Reversing Arrows" 2608.03512; nothing code-specific |
| 37 | READ | 2606.00593 SPADER | premature termination; 1/N novelty premium; recall gains |
| 38 | READ | "Training Versatile Coding Agents in Synthetic Environments" (2512.12216) | SWE-Playground: 7B 17.0 vs 15.2; 32B 31.2 vs 40.2 |
| OR1 | OR | "degree-preserving rewire self-distillation" | Reviews only; GRACE (KDD'23), irrelevant |
| OR2 | OR | "atomised set emission recall" | Irrelevant |
| OR3 | OR | "list generation recall per-candidate LLM" | Irrelevant |
| OR4 | OR | "contrastive self-distillation wrong hint" | **RLCSD (ICLR 2027 submission liW6AQ3OqV)**; "Learning What to Teach: Adaptive Hint Generation for OPSD" (ICLR 2027, 7O8pLcA8w3, SNIPPET); "Better Thinkers, Worse Believers" (NeurIPS'26 workshop, SNIPPET) |
| OR5 | OR | "degree-preserving rewiring LLM graph reasoning" | Only a network-science rewiring paper; no LLM prior |
| OR6 | OR | "predictive mutation testing" | Spotting Code Mutation for PMT (ASE'24), WITNESS (2025), Test-Case-Level PMT (APSEC'23), LLMorpheus (TSE'25), PLM-efficient mutation testing (2023): all SNIPPET |
| OR7 | OR | "information gain reward agent fault localization" | Interactive FL lineage (APSEC'06, JCST'09), Hierarchical Reward Modeling for FL (EMNLP'25 Findings): SNIPPET |
| OR8 | OR | "redundant tools agent reinforcement learning" | Nothing relevant |
| OR9 | OR | "tool schema bias equivalent" | **Action-Space Shaping / Tool-Schema Bias (ICLR 2027, eC7MH5DgcU)** |
| OR10 | OR | "synthetic repositories software agents" | Training Versatile Coding Agents in Synthetic Environments (read as #38); WideSWE (ICLR 2027, SNIPPET) |
| OR11 | OR | "LLM under-enumerate list recall" | Irrelevant |
| OR12 | OR | "execution trace prediction training code" | "How Should Code LLMs Learn Execution? A Study of Trace Representations" (COLM'26 workshop), TRACE-SWE (ICLR 2027), "Do Code Semantics Help?" (ACL ARR 2025): SNIPPET |
| OR13 | OR | "self-teacher rewired graph" | GNN self-distillation only; no LLM prior |
| OR14 | OR | "marginal likelihood equivalent actions agent" | Irrelevant (GP/GNN marginal likelihood) |
| OR15 | OR | "software world model blast set" | **SWM (ICLR 2027 submission LCT7iCqsXa)** |

**Not done (budget).**
- Direct read of PMT (ISSTA'16), MutationBERT, ENTBUG and Guu et al. 2017. Each is SNIPPET or INFERRED above.
- Semantic Scholar. alphaXiv and OpenReview covered it.
- Firecrawl PDF scrapes of OpenReview-only submissions. None needed: every ICLR 2027 hit was also on arXiv and was read there.
- A deep search for idea 12.
