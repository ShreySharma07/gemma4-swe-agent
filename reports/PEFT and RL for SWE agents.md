# Offline step credit is the open SWE gap

**Bottom line.** Offline methods now match online RL on math and single-turn code. On multi-turn SWE agents, nobody has run that comparison. The sharpest open gap sits at a junction. Every deterministic step-credit method for agents is an *online* GRPO advantage modifier, and every offline method that rivals online RL uses *trajectory-level* weights only. No SWE work joins the two (critic C-Q31, C-Q32, C-Q35, C-Q36). An adversarial critic and a prior-art red team ran four kill rounds over about 40 candidates. They killed or narrowed most of them, including state-conditional copy suppression (SCCS) as a method claim, the k=2 verifier, self-RFT, gold-hinted distillation, offline DAgger, a repro-test writer and system-prompt distillation. Two things survive. One is a training architecture, **DECAF** (Dependency- and Execution-Certified Advantage Fine-tuning). It moves dependency credit offline, into importance-weighted LoRA SFT that fits Kaggle 4×L4 and serves on the W4A16 Gemma-4-31B. The other is a sharper, cheaper paper: **Z, zero-information actions**. Z is the SCCS line reframed at field level. Its question is whether in-context copying drives agent loops at 31B, and whether a guarded penalty labelled by environment state removes those loops without suppressing legitimate repeats. On the official rubric, **Z scores 21/25, DECAF 16 and SCAD-R 16**. The recommended paper has Z as its headline and DECAF's credit-validity test (Tier A) as an analysis section. Both have pre-registered kill experiments that cost **≤4 GPU-h (Z) and ≤1.5 GPU-h (DECAF Tier A)**, plus CPU work. **None of the experiments has been run.** The engineering mainline still carries the leaderboard: render-exact, filtered teacher-trajectory SFT. All research bets ride on top of it as extra loss terms.

Search-log convention used throughout: **C-Q1…C-Q38** = critic searches (`critic_judgement.md`); **R1…R26** = red-team searches (`prior_art_redteam.md`); **X-Q1…X-Q18 / X-R1…X-R9** = cross-domain log (`cross_domain_ideas.md`). Every "not found" below is absence in alphaXiv's arXiv index plus web search. OpenReview (ICLR-2027 submissions) was never searchable.

---

## Eight field-level gaps survive scrutiny, and only two are worth a paper

The user's brief was to work at the level of the field, not at the level of our own runs. The critic's Round 4 (R4.1) mapped what is still open in PEFT and offline/RL-style tuning for SWE agents, then attacked each gap. The table gives the result. Our Gemma runs appear only as the testbed.

| # | Gap | What exists | Searches | Status after attack |
|---|---|---|---|---|
| **G1** | Matched-compute offline / iterated-batch vs online RL for multi-turn SWE at ≥30B | On math and code, offline now matches or beats GRPO. G2D (short GRPO warm-up, then offline DPO) reaches ≥ GRPO at ~¼ compute ([G2D](https://www.alphaxiv.org/abs/2605.21266)). DRIFT's offline importance-weighted SFT ≈ multi-turn RL (ICML'26) ([DRIFT](https://www.alphaxiv.org/abs/2605.31455)). REGEN's offline replay recycling ≈ MOPD ([REGEN](https://www.alphaxiv.org/abs/2607.19450)). SWE papers (Nebius, SWE-Gym, Agent-RLVR) have no online arm | C-Q27, C-Q29, C-Q30, C-Q37 | **Narrowed: open for SWE only.** G2D attributes the offline–online gap to "data informativeness", which has never been tested on long-horizon SWE |
| **G2** | Step-level credit without online rollouts for long-horizon SWE | DepGPO traces read–write dependencies back from the verifier's read set and gains **+3.2–10 pts over the best RL baseline** on Terminal-Bench, but only online and with file-level reads ([DepGPO](https://www.alphaxiv.org/abs/2610.03634)). VICT, AgentLTL and CRR are also online ([VICT](https://www.alphaxiv.org/abs/2608.28128); [AgentLTL](https://www.alphaxiv.org/abs/2607.02599); [CRR](https://www.alphaxiv.org/abs/2609.33875)). DRIFT, REGEN and G2D use trajectory weights only, and DRIFT supervises only the terminal turn | C-Q31–C-Q38 | **Open at the junction.** Both halves exist separately |
| **G3** | Teacher-free correction at the agent's own states | DAgger-SWE, OEC and PivotOPD need a stronger teacher. Hinted self-distillation falls below base ([PSP](https://www.alphaxiv.org/abs/2609.29051)). AED's offline action-DPO was null ([AED](https://www.alphaxiv.org/abs/2609.40111)) | C-Q9, C-Q11–C-Q14, X-Q2 | Open |
| **G4** | Internalising harness behaviours that cannot ship | Harness-Zero, HAT (SWE-QA only), HAD ([Harness-Zero](https://www.alphaxiv.org/abs/2609.24974); [HAT](https://www.alphaxiv.org/abs/2610.01235); [HAD](https://www.alphaxiv.org/abs/2610.02858)) | X-Q3 | Narrowed: open only for a same-model deterministic harness, SWE *resolution*, ≥30B |
| **G5** | Train-on-X-bit, serve-on-a-different-int4 LoRA for agents | ProjQ, QA-LoRA and Scale-QLoRA all use the same format for training and serving. vLLM reuses one adapter on AWQ and GPTQ bases without measuring fidelity ([ProjQ](https://www.alphaxiv.org/abs/2606.00494); [vLLM example](https://docs.vllm.ai/en/v0.12.0/examples/offline_inference/lora_with_quantization_inference/)) | C-Q5, C-Q8, R9, R12, R13, R19–R21 | Open but low-value (a measurement) |
| **G6** | Post-training ≥30B SWE agents on 24 GB cards with no online RL | Every SWE RL/SFT paper uses A100/H100/H200 nodes. FrogNano is 4B | C-Q28 | Open, but a constraint rather than a contribution |
| **G7** | Offline proxies on own visited states, validated against closed loop | Gold-history next-turn gains do not transfer, and Gemma-3 strict tool accuracy fell after SFT ([2609.21187](https://www.alphaxiv.org/abs/2609.21187)) | C-Q6, C-Q7, C-Q21 (no papers) | Open (methodology) |
| **G8** | Usable offline negatives for agents | AED offline DPO null. Agentic-DPO is non-SWE ([Agentic-DPO](https://www.alphaxiv.org/abs/2607.10601)). DITTO is text-only. AKBE's offline variant over-suppresses necessary calls ([AKBE](https://www.alphaxiv.org/abs/2605.26952)) | C-Q1, C-Q24, C-Q26, R1–R3 | Narrowed |

**Kill round 4a.** It removed G5 (a measurement table), G6 (a constraint), G7 (a methods section, not a headline) and G4 (the delta left by HAT, Harness-Zero and HAD is thin). **G2 is the top gap. G1 is the framing that makes G2 matter. G3 and G8 plug into G2's negative and recovery channels.** The rest of the report explains how the candidates were narrowed to fit those gaps.

Why G2 matters for anyone with commodity hardware, not just us: DepGPO itself reports that **61% of K=4 GRPO groups have zero advantage**, and its dependency trace costs only **3.8 s of CPU per trajectory** ([DepGPO](https://www.alphaxiv.org/abs/2610.03634), C-Q38). The expensive part of credit assignment is the online loop, not the credit. Online SWE RL, meanwhile, runs at 64 H100s for 6 days (DeepSWE, 23.0 → 42.2) ([DeepSWE](https://www.together.ai/blog/deepswe)) or 16 nodes of 8 H200s (Nebius, RFT 20.5 → RL 39.0) ([Nebius](https://arxiv.org/abs/2508.03501)). That is two to three orders of magnitude above ~30 L4-hours a week.

---

## What the literature already settles under the 4×L4 constraint

Four facts from the survey notes bound every candidate.

**Offline substitutes on SWE are thin but real.** One round of RFT on self-rollouts with error-turn masking took Qwen2.5-72B from 11.4 to 20.5, about a third of the full RFT+RL gain ([Nebius](https://arxiv.org/abs/2508.03501)). Offline trajectory DPO after RFT added only +1.6 pp at 72B ([Agent-RLVR](https://arxiv.org/abs/2506.11425)). Naive self-RFT *hurt*: SWE-Gym's OpenHands agent fell from 15.3 to 8.7 on Lite, and iterated RFT saturated after two rounds (32B LoRA 19.0 → 19.7) ([SWE-Gym](https://arxiv.org/abs/2412.21139)). The best "offline ≈ online" result, Agentic-DPO (step-level, expert-anchored, 1.6× SFT cost vs 13.6× for GRPO), has never been run on SWE. Removing its SFT anchor cost 14–18 points ([Agentic-DPO](https://www.alphaxiv.org/abs/2607.10601)).

**Data curation beats volume at our scale.** SWE-Prime's top-10% process-filtered, segment-masked SFT beat full resolved-trajectory SFT (Qwen3-Coder-30B-A3B 51.0 → 53.2; GLM-4.7-Flash 41.4 → 51.4) ([SWE-Prime](https://arxiv.org/abs/2608.27449)). SWE-Lego's step-level error masking added about +2 pp ([SWE-Lego](https://www.alphaxiv.org/abs/2601.01426)). SERA's ordered truncation beat random truncation, 43.0 vs 37.5, and removing reasoning text collapsed results from 41.0 to 23.0 ([SERA](https://www.alphaxiv.org/abs/2601.20789)). Nebius found that *filtering out* looping trajectories increased looping, so loops must stay in the data as negatives or masked context ([Nebius](https://arxiv.org/abs/2508.03501)).

**PEFT variant matters far less than servability.** vLLM rejects DoRA and any adapter bias outright ([vLLM peft_helper](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/lora/peft_helper.py)). The only controlled agent comparison found shows DoRA ≈ LoRA (+1.3 on BFCL at 1B) ([MoRAgent](https://www.alphaxiv.org/abs/2512.21708)). Plain LoRA with `bias="none"` on all linear layers at r=16–32 is the default. Merging into a 4-bit base can lose up to 39 pp ([Scale-QLoRA](https://www.alphaxiv.org/abs/2609.04526)), but the scorer serves unmerged adapters, so merging is irrelevant.

**Gemma-4 training has a hard blocker.** Gemma-4's global-attention layers use head_dim 512, and FlashAttention caps at 256. Training crashes on step 1 unless attention is set per layer type, for example FA2 for sliding layers and SDPA for full layers ([SkyRL #1564](https://github.com/NovaSky-AI/SkyRL/issues/1564)). Whether SDPA picks a memory-efficient kernel for head_dim 512 on sm89 is unmeasured. If it falls back to the math kernel, a 32k global layer would materialise ~64 GB and run out of memory (INFERRED arithmetic, `peft_methods_l4.md`). The same note estimates **1k trajectories × 16k tokens ≈ 7–11 GPU-h per epoch** at 400–650 tokens/s across 4×L4. Both are GUESSED until Phase 0b runs.

---

## Four kill rounds turned 40 candidates into one paper and one architecture

### Round 1: twenty candidates, thirteen killed

**Findings.** The critic scored 20 candidates on novelty, feasibility, payoff and falsifiability (0–3 each; any 0 kills). The facts it rested on were computed from our own files. Exact repeats occur in **102/129** v2 tasks. Malformed arguments occur in only **9/129** (the oft-quoted 61% malformed edits is v5, not the frozen v2 scaffold). "Wrong fix, never tested" occurs in 47/129 (`results/runs/r2_v2_sed/per_task.jsonl`, `summary.json`). Survivors were step-level offline DPO (wounded by AED's null), unlikelihood on exact repeats, format self-repair (engineering only), the quantisation-matched adapter (measurement only), partial-fix negatives, and own-state proxy validation (evaluation only).

**Decision.** Combine survivors. **Next step:** Round 2 combinations.

### Round 2: combinations, attacked again

**Findings.** C1 combined unlikelihood with own-state evaluation into **SCCS**. Its negatives are labelled by *state equivalence*: an identical call with no intervening mutating action has a provably identical observation, so it is a zero-information action. Three attacks narrowed it. Contextual Drag shows targeted SFT costs **−22 to −28 pp** when the context is correct, so a guard class is mandatory ([Contextual Drag](https://www.alphaxiv.org/abs/2602.04288), C-Q20, C-Q22). 2609.30725 already defines "ReTest" via inter-patch windows, so the *label* is known; its mitigation is prompts and skills, and training is named as future work ([2609.30725](https://www.alphaxiv.org/abs/2609.30725), C-Q24, C-Q25). AKBE's offline signal over-suppressed necessary calls, so λ must be small ([AKBE](https://www.alphaxiv.org/abs/2605.26952), C-Q26). C-Q23 found no paper that trains on zero-information actions. C2 (information-symmetric recovery SFT) died as research: AED, Fission-GRPO and ContractRL cover it. C3 (partial-fix negatives) and C4 (serve-faithful adapter) survived as runner-up and third.

**Decision.** SCCS survives, narrowed to "state-equivalence repeat labels as offline LoRA negatives at own states, plus a copy-mechanism test at 31B." **Next step:** pre-register falsifiers.

### Round 3: SCCS picked, with a four-stage falsifier

**Findings.** The mechanism premise borrows from Feedback That Backfires: after a failure, repeat probability rises from **0.06 to 0.54**, and 83% of the effect is driven by the failed call's surface form. That paper covered only 135M–1.7B models ([Feedback That Backfires](https://www.alphaxiv.org/abs/2608.23651); no ≥30B replication found, C-Q19). The pick was falsifiable at four levels (K0–K3, detailed below).

**Decision.** SCCS is the Round-3 top pick. **Next step:** hand to the red team.

### Red-team round: DITTO and aTTT narrow SCCS

**Findings.** The red team found a closeness-2 hit the critic missed. **DITTO** (NeurIPS 2022) already measures the self-reinforcement/copy effect, trains a penalty on repeating the previous occurrence relative to that occurrence's probability, and frames necessary repetitions as positives against useless ones ([DITTO](https://www.alphaxiv.org/abs/2206.02369), R1, R5). DITTO's margin variant also over-penalised every repetition, which directly supports SCCS's small-λ plus guard design. **aTTT** applies repetition-aware LoRA updates *inside* SWE-bench Lite episodes (Qwen3.5-27B 57.8 → 62.7) and shows a decoding repetition penalty of 1.1 *hurts* (49.7 vs 50.7 on ALFWorld) ([aTTT](https://www.alphaxiv.org/abs/2607.03441), R3, R6). Agent-R reduces dead loops, but via revision SFT built with MCTS ([Agent-R](https://www.alphaxiv.org/abs/2501.11425), R8, R10). No paper builds a step negative whose label is **computable from the trajectory alone, with no reward, judge or teacher** (R2, R3, R8, R14). Partial-fix negatives met 2610.03984, which mutates patches into oracle-rejected variants for verifier RL but does not drop whole functions ([2610.03984](https://www.alphaxiv.org/abs/2610.03984), R18, R22). The NF4 → W4A16 study found no prior measurement and was judged "engineering note" (R9, R12, R13, R19–R21).

**Decision.** No hypothesis killed, but SCCS's novelty drops to about 1.5/3. Any paper must add a DITTO-loss arm and a decoding-penalty baseline. **Next step:** look outside the SWE literature for a bigger architecture.

### Cross-domain round: sixteen transplants, two killed on constraints

**Findings.** The divergent search produced 16 candidates (X-Q1–X-Q18). The strongest were these:

- **DART-style state-preserving perturb-and-return** (C1). Gemma's own failure actions (duplicate reads, rejected malformed calls) do not change environment state, so the "return" label is just the clean next action. This adapts PC-DAgger's calibrated, decaying noise ([PC-DAgger](https://www.alphaxiv.org/abs/2609.30462)). The main prior against it: Bench2Robust's vanilla SFT on injected failures gained nothing (24.0 vs 23.2) ([Bench2Robust](https://www.alphaxiv.org/abs/2608.11977)).
- **Same-model, information-symmetric harness distillation** (C2 → SCAD).
- **Weight surgery compiled into a LoRA** (C3). The public Gemma-4-31B "noloop" edit strips 1,062 L36 neurons, but residual 31B failures became "doom looping", and attribution was unstable (Spearman 0.122) ([One Neuron](https://www.alphaxiv.org/abs/2606.13705)).
- **Process-tag-conditioned SFT** (C4).
- **Program-derived data-flow-DAG loss weights** (C9). Note that LLM-annotated DAG *pruning* raised stuck-loop rates 2–3× when it broke the interaction format ([2609.18417](https://www.alphaxiv.org/abs/2609.18417)).

Test-time training (C15) and hypernetwork per-task LoRAs (C16) died on the declarative-scorer constraint.

**Decision.** Feed C1, C2 and C9 into the gap analysis. **Next step:** Round 4.

### Round 4: gaps first, then one architecture

**Findings.** Twelve new searches (C-Q27–C-Q38) established G1–G8 above. DepGPO (2 Oct 2026) is *concurrent* and online-only with file-level reads. DRIFT and REGEN are offline but trajectory-weighted. Joining them, with SWE-specific line-level reads, was not found (C-Q31, C-Q32, C-Q35, C-Q36).

**Decision.** Build **DECAF** (next section) as the architecture answer to G2, with SCCS folded in as its λ_z term, DART as its injector and SCAD-R as an optional sampler. **Next step:** score against the official rubric.

### Round 5: the rubric reverses the headline

**Findings.** Scored on the official criteria, the clean single insight beats the multi-component architecture. DECAF loses on Verifiability (six interacting parts, no online arm) and Clarity (hard to fit in ≤3,000 words). **Decision.** Headline Z; DECAF Tier A becomes an analysis section. **Next step:** the cheapest kill experiments.

---

## Where ideas died, and what each death taught

| Idea | Killed or narrowed by | Lesson |
|---|---|---|
| **SCCS as a method novelty** | DITTO (closeness 2), aTTT (1–2), Welleck UL (1) — R1, R3, R5, R6 | The loss is not new. Only the *label source* (environment semantics: no mutation in between ⇒ provably identical observation) and the 31B real-agent setting are. It survives reframed as Z and as DECAF's λ_z |
| **Verifier LoRA for k=2 selection** (H7) | Time budget: ~5 min/task in a 12 h cap, and v6 already hit ~10 h. SWE-Gym Best@2 is only +4.2 pp ([SWE-Gym](https://arxiv.org/abs/2412.21139)); C-Q10 | Verifiers pay at k≥8–16. At k≈1–2 they belong offline, as data filters and rewards, not at test time |
| **Self-RFT on own successes** (H12) | SWE-Gym 15.3 → 8.7; 32B LoRA iterated RFT +0.7 | Own successes on easy tasks amplify bias. Keep teacher/anchor data, cap per instance, weight rather than filter |
| **Gold-hinted self-distillation** (H14, OPSD/PivotOPD) | PSP: OPSD falls *below base* on SWE-bench Verified (Qwen3-4B 3.56 → 2.40) and fails on Gemma4-E4B ([PSP](https://www.alphaxiv.org/abs/2609.29051)); C-Q14 | Privileged information belongs in the *sampler*, not the loss. This lesson shapes SCAD-R's information-symmetric annotations |
| **Offline DAgger** (H13) | Already done on SWE (+3.9/+3.6 at 4B/8B with a Qwen3-Coder-30B teacher, [Revisiting DAgger](https://www.alphaxiv.org/abs/2605.12913)); needs a teacher stronger than Gemma-31B, not servable on L4; C-Q9, C-Q11 | Without a teacher, correction labels must be provably valid by construction (DART's state-preserving perturbations) or certified by execution (DECAF) |
| **Repro-test writer adapter, SFT only** (H8) | ExecCritic: SFT-only test validity 39.6%, and base self-tests *lower* resolve 61.2 → 57.3 ([ExecCritic](https://www.alphaxiv.org/abs/2609.09133)) | Bad tests produce false confidence. Testing behaviour must be rewarded *after an edit*, not as test volume |
| **Context distillation of the ~1k-token system prompt** | Cross-domain Q2: the prompt is 609 words (~3% of 32k), and the prompt is our one per-submission edit channel | Context distillation pays only for context that *cannot* ship: per-step harness annotations (SCAD) or content longer than the window |
| **Per-role adapters** (H6) | MoRAgent/MoRSE/ExRole exist; the two-agent v5 gained nothing on the LB | Role decomposition is not the bottleneck for this model |
| **Dropping looping trajectories** (H9) | Nebius §5.2: looping increased | Loops are signal. Mask or penalise them; do not drop them |
| **Pre-execution sentinel adapter** (H19) | Needs a per-step second model call; ADK cannot gate tool execution ([HiSentinel](https://www.alphaxiv.org/abs/2609.39957), C-Q12) | The deployment is declarative, so everything must be in the weights |
| **Gemma-native SERA-style data generation** (H18) | ~100 trajectories per session at ~5 min/task | Rollout-heavy data generation must be small and targeted (DECAF: K=4 on ~60 tasks) |
| **Information-symmetric recovery SFT** (C2, Round 2) | AED, Fission-GRPO, ContractRL | Only the 9/129 format case is deterministic. Kept as an engineering data slice |
| **Partial-fix negatives** (runner-up) | Narrowed by 2610.03984's mutation negatives (R18, R22) | Needs a random-hunk-drop control to show necessity labelling matters. Still gated on IFG Stage A |
| **NF4 → W4A16 adapter study** (third) | ProjQ/QA-LoRA same-format; vLLM reuse without measurement | A real gap, but a one-table measurement. Run it inside Phase 0b, not as a paper |
| **Test-time training, hypernetwork LoRAs** (C15, C16) | Declarative scorer; LatentSkill shows independent per-skill LoRAs underperform ([LatentSkill](https://www.alphaxiv.org/abs/2606.06087)) | Same constraint as the sentinel: weights only, fixed files |

---

## Z scores 21 of 25; DECAF and SCAD-R tie at 16

Scores are the critic's judgement (GUESSED) and assume the falsification experiments pass. The rubric's Novelty criterion explicitly credits "rigorous analysis of existing methods", which is why DECAF Tier A (does DepGPO's file-level read set collapse on Python repos?) earns a place in the paper even if Tier B never runs.

| Criterion | **Z: zero-information actions** | **DECAF** | **SCAD-R** |
|---|---|---|---|
| Novelty (incl. highlighting properties of existing methods) | **3.** First ≥30B real-agent test of the copy effect; Feedback That Backfires stops at 1.7B (C-Q19). The new piece is the state-equivalence label, not the loss (DITTO is closeness 2; R1, R5). | **3.** New junction (offline + step credit + coverage-resolved E\*), but DepGPO, DRIFT and REGEN each cover a half (C-Q35–C-Q38). A reviewer will say "offline DepGPO" | **2.** Harness-Zero, HAT and HAD are closeness 2 (X-Q3); the same-model, information-symmetric twist is thin |
| Quality / generality beyond the competition | **4.** The labelling rule is harness- and model-agnostic and reruns on any public trajectory corpus | **4.** Any agent with a test verifier, any model, commodity hardware | **3.** Annotations are scaffold-specific |
| Relevance to SE and agentic learning | **4.** Redundant actions affect 79–98% of tasks even in Claude Code and mini-SWE-agent ([2609.30725](https://www.alphaxiv.org/abs/2609.30725), C-Q25) | **4.** Addresses G1/G2, the field's "online RL is too expensive" problem | **3.** Practitioner value; less for learning theory |
| Verifiability | **5.** Deterministic labels; premise test is CPU plus ≤1.5 GPU-h; pre-registered kills; releasable code and labels | **3.** Six interacting components, no online arm, three arms plus ablations needed | **4.** One cheap with/without-annotation contrast (~1 GPU-h) |
| Clarity | **5.** One taxonomy table, one DITTO-style mechanism figure on real agent states, one ablation | **2.** Certifier, credit, negatives, DART and loss do not fit 3,000 words with evidence | **4.** One idea, one contrast |
| **Total** | **21** | **16** | **16** |

For reference, partial-fix negatives score about 13 (gated on IFG) and the NF4→W4A16 measurement about 13. DECAF ranks above SCAD-R on the tie because it wins on Quality and Relevance. That tie-break is our internal choice. The competition breaks ties between submissions by **earliest entry**, so a finished Z paper submitted early is worth more than a stronger paper submitted late.

**Recommended paper (≤3,000 words).** The headline is Z, in three sections: (1) a taxonomy of zero-information actions across our runs **plus ≥1,000 trajectories from a permissive public corpus**, such as Nebius SWE-rebench-OpenHands (CC-BY-4.0, [card](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)), at 0 GPU-h, which is what earns the Quality score; (2) the 31B copy-mechanism probe; (3) the guarded fix vs DITTO, plain unlikelihood and a decoding penalty. DECAF Tier A becomes a ~400-word section if it passes. DECAF Tier B and SCAD-R go to future work unless they finish before the Nov 5 freeze.

---

## DECAF moves dependency credit offline into weighted LoRA SFT

DECAF is an offline, iterated-batch post-training loop. It runs at most two rounds: SWE-Gym and DRIFT both saturate after about two refreshes. Rollouts, certification and training run in separate sessions, so nothing is online. The output is one plain PEFT LoRA served by the declarative v2 agent with no harness.

```
ROUND r (r ≤ 2)
┌─ GPU session A — ROLLOUTS  (vLLM 0.19.1, W4A16 + adapter_r; local runner + tracer; scorer untouched)
│    task pool: permissive repos with F2P/P2P tests, repo-disjoint from the dev set
│    K=4 rollouts/task at T≈0.7; per step log: call, observation, line-level workspace diff,
│    sed/grep line spans read, behaviour log-probs of emitted tokens   [REGEN replay record]
│    optional sampler add-on: SCAD-R annotations in observations      [Harness-Zero/HAT; PSP lesson]
│
├─ CPU session — CERTIFIER (deterministic, no model calls)
│    (1) verifier: F2P/P2P on final state → R ∈ {0,1} (+ F2P fraction)          [known]
│    (2) coverage tracer: lines/functions executed by F2P tests on base + final  [NEW-2]
│        → coverage-resolved verifier read set E*   ◄── same tooling as IFG Stage A
│    (3) dependency graph: line-level writes (diffs), line-level reads (sed/grep spans),
│        test-run reveals; backward trace from E* with β-decay → step credit c_t  [DepGPO rule]
│    (4) zero-information flag z_t: identical call, no mutating action since;     [NEW-3; DITTO lineage]
│        guard class g_t: identical call AFTER a write (e.g. re-run test)
│    (5) weights: Â = class-balanced, group-normalised outcome                     [REGEN/DRIFT]
│        w_t = Â · f(c_t), unit-mean step factors; abstain → uniform               [NEW-1: offline]
│    (6) DART injector: state-preserving perturbations (duplicate read + identical
│        output; rejected malformed call + parser error) spliced into positives;
│        label = clean next action; rate ∝ measured failure histogram, decays per round [PC-DAgger]
│
└─ GPU session B — TRAINING (FSDP-QLoRA, one adapter, r=16–32)
     L = Σ_{w_t>0} ρ̄_t · w_t · CE(a_t)                    truncated-IS weighted SFT   [REGEN]
       + λ_z Σ_{z_t} DITTO-relative penalty, λ_z ≈ 0.05     guard g_t kept as CE      [DITTO, AKBE]
       + λ_a CE(render-exact, SWE-Prime-filtered teacher slice)  anchor essential     [Agentic-DPO]
     → adapter_{r+1}

DEPLOY: one PEFT LoRA (bias="none", all-linear), declarative v2 agent, no harness.
```

| Piece | Source (closeness) | Status |
|---|---|---|
| Read–write dependency credit from the verifier read set | DepGPO (2: same idea, **online**, file-level reads, terminal tasks) | VERIFIED (C-Q38) |
| Verifier atoms → proof edges; no-op repeats get no credit | VICT (1–2: online, non-SWE) | VERIFIED (C-Q34) |
| Offline importance-weighted SFT ≈ online multi-turn RL | DRIFT (2: trajectory weights, math), ICML'26 | VERIFIED (C-Q37) |
| Replay records, asymmetric truncated IS, class-balanced sampling | REGEN (1: math/code/IF, 1.5B) | VERIFIED (C-Q29) |
| Guarded repeat penalty | DITTO (2 for this component) | VERIFIED (R5) |
| State-preserving perturb-and-return, calibrated decaying noise | DART / PC-DAgger (1: robotics) | VERIFIED (X-R8) |
| SFT anchor essential for step-level negatives | Agentic-DPO | VERIFIED |
| **NEW-1:** step-level dependency credit used offline in weighted SFT | — | Not found: C-Q31, C-Q32, C-Q35, C-Q36 |
| **NEW-2:** coverage-resolved verifier read set E\* for SWE | — | Not found: C-Q35. The claim that file-level E collapses because pytest imports the whole package is GUESSED |
| **NEW-3:** certified zero-information negatives with post-change guard in the same objective | DITTO/SCCS lineage | Narrowed (closeness 2) |

**Honest novelty: about 1.5–2 of 3.** Every component is published, and DepGPO (2 Oct 2026) is concurrent work. The contribution is the junction. The rebuttal to "offline DepGPO" rests on NEW-2, which Tier A tests directly. **The biggest weakness is the missing online arm:** we cannot show what fraction of online gains offline recovers on SWE. We can only cite DepGPO and SkyRL numbers from other models. That caps DECAF at a workshop or short paper.

**SCAD-R, the runner-up architecture.** During local data generation only, the runner appends deterministic annotations computed from visible state: "identical to call #k, nothing changed", "you edited X, no test since", "step n/40". The LoRA is trained on the annotated runs' actions with the annotations stripped. PSP's leak mechanism does not apply, because the annotations are re-derivable from context. If both survive, SCAD-R becomes DECAF's round-2 sampler, raising the yield of positive trajectories.

---

## The cheapest experiments that could prove each pick wrong

None of these has been run. GPU-h figures are GUESSED until Phase 0b measures training throughput.

### Z (SCCS reframed): four stages, ≤4 GPU-h total

| Stage | What | Cost | Kill criterion |
|---|---|---|---|
| **0 (CPU)** | Extend `analysis/trace_analyzer.py` with a repeat classifier: no-progress repeat (NPR), post-change repeat (PCR), post-error repeat. Report NPR share, tasks with ≥1 NPR, position, and resolved-vs-unresolved rates in length-matched call-index bins. Add the same taxonomy on ≥1,000 public trajectories | ~3 h CPU + ~1 h trace retrieval; public corpus ~0.5 day, 0 GPU-h | **K0:** NPR < 50% of exact repeats, **or** NPR in < 25% of tasks |
| **1 (no training)** | On ~300 NPR and ~150 PCR events, `prompt_logprobs` on served W4A16: L_act (real context), L_abl (earlier identical call and output replaced by a placeholder), L_pre (context cut before first occurrence); greedy one-step re-emission rate R0 | ~1.5 GPU-h | **K1:** median (L_act − L_abl) < log 2, **or** R0 < 30% |
| **2 (method)** | Two r=16 LoRAs, ~200 own states each, token-matched. A = SFT on own next non-repeated action + PCR repeats. B = A + NPR penalty (λ≈0.1). B′ = DITTO loss arm (red-team requirement). Evaluate on ≥150 held-out-repo NPR and ≥100 PCR states (train fastapi, test rich + requests) | ~2.5 GPU-h + ~4 h eng.; **needs Phase 0b** | **K2:** B − A < 15 pp or McNemar p > 0.05; also kill if A alone drives NPR < 10%. **K3 (harm):** PCR log-prob falls > 25% relative, parse validity drops > 5 pp, or clean-turn NLL worsens > 2% |
| **3 (closed loop)** | Base vs A vs B on a 40-task unseen-repo dev set | ~3 × 3.5 GPU-h | Required before any leaderboard claim; one-step metrics cannot see trajectory-level shift ([2609.21187](https://www.alphaxiv.org/abs/2609.21187)) |

Power for Stage 2 (GUESSED): with n=150 paired states and assumed R0 ≈ 60%, A ≈ 45%, B ≈ 25%, discordant pairs of 25–35% give power ≈ 0.9 for a 15 pp gap at α=0.05.

**Outcome branches.** A pass at K0–K3 leads to Stage 3, then the B-loss folds into the mainline SFT LoRA as an auxiliary term. Failing K0 or K1 kills Z as a method, but a **negative replication of Feedback That Backfires at 31B** is still a findings paragraph. Failing K2 (A ≈ B) keeps own-state SFT as a cheap mainline data slice, and the novelty claim dies. Failing K3 allows one retry with the guard upweighted and λ halved, then a kill. An ambiguous result (B − A of 5–15 pp, or p of 0.05–0.2) adds 150 states from r1/r3 traces (~0.5 GPU-h) before deciding, with **at most two λ settings**, to avoid fishing.

### DECAF: Tier A before any training, Tier B only if it passes

| Tier | What | Cost | Kill criteria |
|---|---|---|---|
| **A (credit validity)** | On r2 traces plus one fresh K=4 batch on 32 tasks, compute DepGPO file-level credit (E) and DECAF coverage-resolved credit (E\*). Measure the share of steps with non-zero credit, write-credit precision against gold-patch functions, read-credit recall of the read that showed the gold lines, and usable-credit coverage including partial-F2P failures. ~150–250 trajectories give ±7 pp CIs. The 129 public tasks are valid here because credit depends on traces and tests, not knowledge | ≤1.5 GPU-h (the K=4 batch) + CPU coverage in `scorer_site.tgz` | **KA1:** file-level E already gives ≤ 30% credited steps with precision within 5 pp of E\* (NEW-2 dies; DECAF reduces to "offline DepGPO"). **KA2:** usable non-uniform credit on < 60% of trajectories. **KA3:** write precision < 70% or read recall < 60% with E\* |
| **B (method)** | K=4 rollouts on ~60 training tasks (≈240 trajectories). Arms, token-matched with the same anchor: A = DRIFT-style trajectory weights; B = DECAF; C = RFT on successes. Evaluate on the 40-task unseen-repo dev set. Primary metrics are process metrics (closed-loop NPR rate, tested-after-last-edit, calls to first edit). n=40 detects only ~+15 pp of resolve, so resolve is reported, not tested | ~8 GPU-h (3.5 rollouts + 3 train + 1.5 eval) | **KB1:** B fails to beat A on ≥ 2 of 3 process metrics at p < 0.05. **KB2:** B's resolve < A's − 2 tasks. **KB3:** B without f(c_t) recovers ≥ 80% of B − A (the new piece is not the cause) |

**Outcome branches.** Passing A and B leads to one refresh round (~8 GPU-h) and a paper titled "Offline dependency-credited post-training of a 31B SWE agent on 4×L4". Failing KA1 only keeps NEW-1 with file-level E: the graph tie and NEW-2 die, but IFG is unaffected. Failing KA2 or KA3 kills DECAF; fall back to the mainline SFT. An ambiguous Tier B adds a second rollout seed (~3.5 GPU-h), with λ_z ∈ {0, 0.05} the only hyperparameter allowed. A Tier B failure is still a reportable negative for G1: "offline step credit ≠ online gains at 31B LoRA."

### SCAD-R: one step, ~1 GPU-h

At ~300 own NPR and post-edit states, compare next-action distributions with vs without the annotation (greedy + K=4). **Kill** if the annotation changes the action in < 30% of NPR states, or raises "run a test" after an unverified edit by < 10 pp. If the harness does not change behaviour in context, there is nothing to distil. This runs only if GPU slack remains.

---

## Nine engineering must-dos gate every research claim

These are not novel and are not scored for novelty. Most block the experiments above.

| # | Task | Cost | Why it blocks |
|---|---|---|---|
| **E1** | Retrieve r1–r3 raw traces from Kaggle outputs into `data/runs/` (gitignored; process by script only) | ~1 h | Every offline test above needs them; only `devset_v5` is on disk. If not retrievable, rerun v2 on 30 tasks (~2.5–3 GPU-h), doubling Z's cost |
| **E2** | **Gemma tool-call re-render check.** Render the exact turn-n+1 prompt that ADK 1.36.1 + vLLM 0.19.1 produce after an `edit_file` whose `old_string` contains `"`. Is history native `<\|"\|>` syntax or hybrid JSON? Is streaming on? CPU with the tokenizer chat template; GPU only if ambiguous | 2–3 h CPU | A forum report documents Gemma 4 "imitating malformed history" after a JSON/native re-render ([HF forum](https://discuss.huggingface.co/t/gemma-4-bug-fixes-and-research-request/176979), C-Q16). vLLM 0.19.1 fixed only streaming parse bugs. Training data must render identically to inference |
| **E3** | Port `use_scorer_env` + Python 3.13 fixes into `run_official_l4.py` | ~0.5 day | Local numbers were measured in the wrong environment |
| **E4** | Unseen-repo dev set: 40–60 gold-verified tasks from permissive recent repos, excluding fastapi/rich/requests and SWE-bench repos | 1–2 days CPU | The LB cannot see < 3 pp (1 task ≈ 0.8 pp); no adapter can be judged without it |
| **E5** | **Phase 0b training feasibility:** FSDP-QLoRA; SDPA memory probe on the **head_dim-512** global layers (FA2 for sliding, SDPA for full; check that the mem-efficient kernel is chosen on sm89); fused or chunked CE (one 32k × 262k-vocab fp32 logit tensor ≈ 34 GB); max sequence length; tokens/s; offline training wheelhouse; PEFT export check (`bias="none"`, 410 modules, rank, size) plus a serving probe. Fold the C4 NF4-vs-W4A16 NLL/validity table in here (~1 GPU-h) | ~1 day + 2–3 GPU-h | Blocks Z Stage 2, DECAF Tier B and all training. Unmeasured |
| **E6** | Render-exact data converter and filters: SWE-Lego error masking; SWE-Prime S_tool/S_redundancy/S_workflow; ordered truncation ≤ 16–24k; ≤ 3 per instance; git-hack filter; **licence ledger** | 1.5–2 days CPU | Mainline data. Prefer CC-BY Qwen-teacher sets (SWE-ZERO, Nebius OpenHands). Flag Claude/GPT-teacher terms (SWE-smith, SWE-Gym) and Gemma 4 output terms to Shrey. CoderForge and R2E-Gym carry no dataset-level licence ([CoderForge card](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview)) |
| **E7** | Mainline SFT LoRA #1 (1–2k trajectories, r=16–32) plus a small own-state slice (format self-repair; Z's B-loss if it passes) | 8–12 GPU-h/epoch + ~4 GPU-h eval | The largest expected LB lever |
| **E8** | Zero-training prompt A/B (one change): "rerun tests only after a code change; do not repeat a read you already did", plus a reflexive-repair line | ~3.5 GPU-h | The baseline any LoRA must beat. The repair prompt gave +5.9 on Gemma-4-31B INT4 ([Flat Score](https://www.alphaxiv.org/abs/2607.27275)); DevSkills cut ReTest cost by up to 41.7% ([2609.30725](https://www.alphaxiv.org/abs/2609.30725)) |
| **E9** | `validate_submission` plus a worst-case runtime check for every adapter submission | ~0.5 h each | Scorer-path rule |

A new item comes from this report: **trace-level line-read logging** (sed/grep spans per step) in the local runner. DECAF's line-level reads assume our v2 scaffold reads via `sed -n 'a,bp'`. That is plausible but GUESSED until E1's traces are checked.

**GPU plan to the Nov 5 freeze** (≈ 110–120 GPU-h: ~5.5 h until Oct 10, then ~30 h/week). The plan comes to ≈ 50 GPU-h:

| Item | GPU-h |
|---|---|
| E5 | 3 |
| Z Stages 1–2 | 4 |
| C4 (NF4 vs W4A16 measurement) | 1 |
| E7 | ≈ 16 |
| E8 | 3.5 |
| Three dev-set evaluations | ≈ 12 |
| Z Stage 3 | ≈ 10 |

DECAF Tiers A and B add ≤ 10 GPU-h from the remaining slack.

---

## DECAF and Interventional Fix Graphs share one coverage tracer

The earlier graph report picked **Interventional Fix Graphs (IFG)**. IFG claims that the executed call graph of the failing tests predicts which edited functions of a multi-function fix are *jointly necessary*. Its Stage A is a CPU-only kill experiment: a `sys.setprofile` tracer runs the F2P tests to record the executed graph, then every function-level subset of the gold patch is applied and tested. It costs about 1.5 working days and 0 GPU-hours and has four kill criteria (K1 labels, K2 graph decorative, K3 degenerate structure, K4 Gemma relevance). That report targeted the multi-function dead zone in our runs: **2 of 45 tasks with ≥3 gold functions resolved**.

The connection is direct, not decorative. **DECAF's coverage-resolved read set E\* *is* the executed graph of the F2P tests**, built with the same tracer in the same rebuilt scorer environment (`scorer_site.tgz`). Building the IFG Stage-A tracer therefore pays for most of DECAF Tier A's CPU side. The reverse also holds: if DECAF Tier A passes, the IFG paper gains a "graph used for training credit" section. The two picks fail independently. IFG's K2 tests whether the executed graph predicts *joint necessity among fix functions*. DECAF's KA1 tests whether line-level coverage *sharpens step credit* beyond file-level reads. Either can pass while the other fails. Partial-fix negatives, the old Round-3 runner-up, need IFG Stage A's necessity labels and a random-hunk-drop control against 2610.03984. They stay parked behind the IFG go/no-go.

---

## What is verified and what is guessed

| Item | Status |
|---|---|
| Exact repeats in 102/129 v2 tasks; repeat share 28.4%; mean repeats 10.35 unresolved vs 3.15 resolved; malformed args 9/129 (v2); 61% malformed edits is v5 | **VERIFIED** (our files). The causal link from repeats to failure is **GUESSED** (length confound: 34.3 vs 19.6 mean calls) |
| DepGPO, VICT and AgentLTL step credit are online-GRPO only; DepGPO reads are file-level | **VERIFIED** (C-Q33, C-Q34, C-Q38) |
| DRIFT, REGEN and G2D: offline ≈ or > online on math/code at ≤ 8B, trajectory-level weights | **VERIFIED** (C-Q29, C-Q30, C-Q37) |
| No SWE matched-compute offline-vs-online study; no offline step-credit SWE method; no zero-information-action training paper | **VERIFIED as absence** in the alphaXiv index and web (C-Q21, C-Q23, C-Q27, C-Q31, C-Q32, C-Q35, C-Q36); OpenReview not searched |
| DITTO closeness 2 to SCCS; aTTT online repetition-aware LoRA on SWE-Lite | **VERIFIED** (R5, R6) |
| PSP: hinted self-distillation below base on SWE; ExecCritic: SFT-only tests hurt; SWE-Gym self-RFT regression | **VERIFIED** (C-Q14; failure-targeted note; SWE-Gym PDF) |
| Gemma-4 head_dim-512 global layers break FlashAttention | **VERIFIED** (SkyRL #1564) |
| vLLM rejects DoRA and adapter bias | **VERIFIED** on vLLM main, not the 0.19.1 tag |
| Feedback That Backfires replicates at 31B (NPRs are in-context copies) | **GUESSED**; tested by Z Stage 1 |
| NPR ≥ 50% of exact repeats | **GUESSED**; tested by Z Stage 0 |
| An own-state one-step metric predicts closed-loop repeat reduction | **GUESSED**; gold-history metrics do not (2609.21187); Stage 3 checks it |
| File-level verifier read sets are uninformative for Python repos | **GUESSED**; tested by DECAF KA1 |
| Line-level reads recoverable from sed/grep spans | **GUESSED**, plausible |
| r1–r3 raw traces retrievable from Kaggle outputs | **GUESSED** |
| 31B QLoRA trains on 4×L4 at ≤ 12–16k tokens; ~400–650 tok/s | **GUESSED** (INFERRED arithmetic; Phase 0b unmeasured) |
| All effect sizes: Z ≥ 15 pp NPR one-step; DECAF ≥ +3 pp vs DRIFT-style, −40% NPR, +10 pp tested-after-edit; rubric scores | **GUESSED** |
| Gemma 4 outputs licence-clean for released training data | **GUESSED**; check the Gemma 4 terms |
| Unlikelihood original (Welleck 2019) numbers | **SNIPPET** via DITTO's tables (R1) |

---

## Conclusion

The kill rounds changed the question. The project started from "which PEFT/RL trick fixes Gemma's loops" and ended at a field-level asymmetry. The structural credit-assignment ideas that make online agent RL work are cheap to compute (seconds of CPU per trajectory) but have only ever been fed to the expensive online loop. The cheap offline objectives that now rival online RL have only ever used trajectory-level signal. That asymmetry, not our leaderboard, is what makes DECAF a research question. It is also why a negative DECAF result would still inform G1. The second lesson is about the rubric. A deterministic label with a mechanism probe and a guarded fix (Z) outscores a six-part architecture because the reviewer can check it. Z's generality comes from rerunning the label on public corpora, not from adding components.

Every "not found" here has a shelf life. DepGPO appeared six days before it was searched, and DITTO hid for a full critic round. The novelty claims must be stated as junctions (state-equivalence labels at own states at 31B; offline coverage-resolved step credit) and re-searched, including OpenReview and DITTO's and DepGPO's citation graphs, immediately before submission.

---

## Decision needed from Shrey

**None of the experiments in this report has been run.** Three go/no-go calls, in order of cost:

1. **Approve Z Stages 0–1 now?** The cost is ~4 h CPU plus ~1 h trace retrieval (E1), the 0-GPU public-corpus taxonomy (~0.5 day), and ~1.5 GPU-h of `prompt_logprobs`. Stage 1 fits inside the ~5.5 GPU-h left before the Oct 10 quota reset. A K0 or K1 failure still yields a findings paragraph.
2. **Build one coverage tracer for both IFG Stage A and DECAF Tier A?** It saves duplicate work. DECAF Tier A adds ≤ 1.5 GPU-h (one K=4 batch on 32 tasks). Approving this implies a "yes" on the pending IFG Stage-A decision from the graph report.
3. **Prioritise E2 (re-render check) and E5 (Phase 0b, including the head_dim-512 SDPA probe) this week?** Z Stage 2, DECAF Tier B and the mainline SFT LoRA are all blocked on E5. E2 decides whether training data and inference history match at all.

Two policy calls ride along:

- **The licence ledger.** Choose whether to admit Claude/GPT-teacher data (SWE-smith, SWE-Gym), or restrict training to CC-BY/Apache Qwen-teacher sets, and whether released data may include Gemma 4 outputs.
- **Paper shape and timing.** The recommendation is Z as headline plus DECAF Tier A as a section, submitted as early as the evidence allows. The award breaks ties by earliest entry.
