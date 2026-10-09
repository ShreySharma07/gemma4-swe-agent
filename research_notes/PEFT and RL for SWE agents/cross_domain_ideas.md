# Cross-domain transplants into offline LoRA tuning for the Gemma-4-31B SWE agent (8 Oct 2026)

Scope: divergent search for mechanisms from offline RL, imitation learning, robotics and control, curriculum learning, model merging, steering and representation engineering, mechanistic interpretability, distillation, test-time training, data attribution and graph credit assignment. Each must become a **plain PEFT LoRA** (no DoRA; at most 8 adapters, rank ≤128, total under 3 GiB) for `gemma-4-31b-it-qat-w4a16`. Training must happen offline on Kaggle 4×L4 (about 30 GPU-h/week, 12 h sessions). Only generate-then-train loops are allowed, for a few rounds at most. Deployment is a declarative YAML agent with a 32k context and about 5 min/task, scored on about 120 hidden tasks from private repos.

The measured failures this targets (from `critic_judgement.md`, VERIFIED there):
- exact repeats in **102/129** v2 tasks;
- a wrong fix that was never tested in **47/129** v2 tasks;
- budget overruns (v5 localiser);
- multi-function fixes: **2/45** solved with ≥3 gold functions;
- malformed arguments, mostly in v5 (**9/129** tasks in v2).

Localisation is not the bottleneck.

Labels:
- **VERIFIED**: read in the paper's full text, an alphaXiv full report, or our own repo files this session.
- **SNIPPET**: seen only in a search abstract or snippet.
- **INFERRED**: my reasoning, or a citation from memory that I did not re-read this session.

All 2026 arXiv items are preprints unless a venue is stated. Absence claims ("no paper found") rest on alphaXiv's arXiv index and web search only. I did not search OpenReview directly.

---

## Q1. Which cross-domain mechanisms can be turned into offline LoRA tuning for this agent, and what does the evidence say for each? (16 candidates)

### Takeaway

Six fields give viable transplants:
- robotics covariate-shift correction (DART-style perturb-and-return);
- harness, context and on-policy distillation;
- mechanistic weight surgery compiled into a LoRA;
- Decision-Transformer-style attribute conditioning;
- dependency-graph loss weighting;
- self-generated local corrections (DAgger without a teacher).

Three candidates die on the hard constraints:
- **test-time training**: there is no training at inference in a declarative scorer;
- **hypernetwork per-task LoRAs**: adapters are fixed files;
- **verifier or rollout-heavy methods**: the time budget rules them out.

The harness-distillation space filled up in Sep–Oct 2026 (Harness-Zero, Harness Annealing, Harness-Aware Distillation), so any idea there must be narrowed to what those papers leave open:
- a **same-model** teacher;
- **information-symmetric** annotations;
- an **SWE** setting at **31B 4-bit**;
- a deployment where the harness is **impossible**, not merely removed.

### Cited Findings

**Imitation learning and covariate shift**
- VERIFIED: on-policy expert corrections (OEC) for SWE agents. Rollouts start with the student and switch to a stronger expert at a random turn (U(0,30)). Results on SWE-bench Verified:
  - +14% / +13% relative over plain imitation at 7B / 32B;
  - 961 OEC + 961 BC trajectories resolve 23 more instances (+4.6 pp) than 2,000 BC;
  - masking the student (on-policy) segment is crucial at 32B (31.8% without masking vs 40.0% with);
  - training on fully on-policy trajectories "severely decreased" 32B performance;
  - removing repetition filtering costs 3.6 pp at 32B;
  - later switch points help slightly: 31.0 / 32.4 / 32.6% for switch turns 0–6 / 7–15 / 16–30.
  — [Lauffer et al. 2512.14895](https://www.alphaxiv.org/abs/2512.14895)
- VERIFIED: the same paper's filter removes trajectories with an identical action three times in a row, or a file read 20+ times in a row. It shows embedding-space covariate shift between student and expert growing sharply from about turn 10 to turn 30. — [2512.14895](https://www.alphaxiv.org/abs/2512.14895)
- VERIFIED: Policy-Calibrated DAgger:
  - estimates the policy's own action noise offline and noises expert intervention states inside that distribution;
  - labels each perturbed state with a corrective chunk that **returns to the expert trajectory**;
  - +2.8 pts over HG-DAgger (p=0.005) on a planar reacher, matching the best fixed noise in hindsight with no sweep;
  - 68.8% vs 57.4% HG-DAgger in a 3D lever-reaching task;
  - the best noise level shrinks as the policy improves (σ 8–16 at round 1 → 2–4 by round 4).

  It cites DART (Laskey et al., CoRL 2017) as the origin of noise injection. — [PC-DAgger 2609.30462](https://www.alphaxiv.org/abs/2609.30462)
- SNIPPET: "Is Online Interaction Necessary for Recovery?" argues for offline perturbation instead of online interaction in robot planning. — [2609.33049](https://www.alphaxiv.org/abs/2609.33049)
- VERIFIED: Bench2Robust (Amazon) injects controlled tool failures in τ²-bench/BFCL. Results on Qwen3-4B:
  - **vanilla SFT gave no recovery gain** (24.0 vs 23.2 under injection), and neither did vanilla GRPO (23.0);
  - their curriculum RL + structured runtime recovery context reached 45.9;
  - runtime context alone gave +16.8 pp;
  - RL without the context at inference gave +6.3 pp.

  Their reward zeroes episodes with ≥4 identical calls. — [Retry, Switch, or Abstain? 2608.11977](https://www.alphaxiv.org/abs/2608.11977)
- VERIFIED: Revisiting DAgger for SWE gains +3.9 / +3.6 at 4B / 8B with a Qwen3-Coder-30B teacher. (Read in a prior session and cited in `critic_judgement.md`.) — [2605.12913](https://www.alphaxiv.org/abs/2605.12913)

**Harness, context and on-policy distillation**
- VERIFIED: Harness-Zero (agent-as-harness distillation):
  - a reviewer model with a private "reference harness" PASSes or REPLACEs each student action in the target action space; the student is then LoRA-SFT'd (r=32, 65k sequence length);
  - on Qwen3.5-9B the macro-average goes base 23.3% → with evolved harness 41.7% → distilled, under the minimal harness, 44.3%;
  - it recovers 82.3% of 28 harness-induced behaviours, including "avoiding repeated failed calls";
  - a **weak reviewer is harmful** (it replaced 66% of steps on USPTO);
  - reviewing with oracle answers gives 15% vs 30% with the reference harness;
  - collection latency is ×2.4. — [2609.24974](https://www.alphaxiv.org/abs/2609.24974)
- VERIFIED (abstract + intro): Harness Annealing Training (HAT) combines explicit control supervision with a curriculum over teacher trajectories collected under progressively weaker harnesses.
  - It uses 9B and 35B models on **SWE-QA / SWE-QA-Pro**.
  - Annealed checkpoints running with tools alone come close to the starting checkpoint with the full harness.
  - For 9B, tools-only scores rise +5.01 / +3.09 pts.
  - "Further annealing does not uniformly improve performance." — [2610.01235](https://www.alphaxiv.org/abs/2610.01235)
- VERIFIED: Harness-Aware Distillation (HAD) contrasts the **same teacher's actions with and without harness information**. The harness is **kept** at deployment.
  - On ALFWorld it avoids stalls in 75.9% of episodes vs ≤64.6% for baselines, and escapes 59.7% of stalls vs 46.8% for the untrained student.
  - With the harness removed it still reaches 40.3% on unseen tasks.
  - On Gemma-4 (E2B student, 12B teacher) it scores 37.3% vs 32.1% for the best baseline.
  - It reads behaviour-monitoring feedback ("repeats an action without effect") partly into the weights. — [2610.02858](https://www.alphaxiv.org/abs/2610.02858)
- VERIFIED: On-Policy Context Distillation (OPCD) trains a context-free student on its own samples with token-level reverse KL (top-256 student tokens) to a frozen context-conditioned teacher.
  - System-prompt distillation (Llama-3.1-8B, medical): base 68.4, in-context 72.2, off-policy context distillation 75.2, OPCD 76.7. **Both distilled variants beat keeping the prompt in context.**
  - A frozen teacher beats self-distillation (Sokoban 53.9 vs 18.8), and self-distillation can diverge.
  - Raw traces as context *hurt* (70.5 vs 75.1 base); extracted "experience knowledge" helps. — [OPCD 2602.12275](https://www.alphaxiv.org/abs/2602.12275)
- VERIFIED: LatentSkill (a hypernetwork turns skill text into rank-8 LoRAs; Qwen3-8B; 8×H100):
  - ALFWorld +21.4 / +13.4 pts over in-context skills, with 63.9% fewer prefill tokens;
  - independently trained **per-skill LoRAs underperformed** (45.7 vs 52.9 in-context, seen split);
  - injection strength follows an inverted U (best α≈0.5–0.6; α=1.2 collapses to 22.9 / 8.2%);
  - long in-context skills *hurt* out of domain (WebShop 6.7 vs vanilla 10.3). — [2606.06087](https://www.alphaxiv.org/abs/2606.06087)
- SNIPPET: Parametric Skills trains a hypernetwork that maps free-form skills into LoRAs **for software-engineering tasks** (as described in LatentSkill's related work). Skill-to-LoRA trains one adapter per skill from skill-conditioned synthetic demonstrations. — [Parametric Skills 2606.30015](https://www.alphaxiv.org/abs/2606.30015); [Skill-to-LoRA 2606.16769](https://www.alphaxiv.org/abs/2606.16769)
- SNIPPET: "When Context Returns" studies what goes wrong when internalized context reappears at inference. — [2606.11627](https://www.alphaxiv.org/abs/2606.11627)
- VERIFIED (prior session, `critic_judgement.md`): hinted self-distillation (OPSD) falls below base on SWE-bench Verified (Qwen3-4B 3.56 → 2.40). Its conclusion: privileged information belongs in the sampler, not the loss. — [PSP 2609.29051](https://www.alphaxiv.org/abs/2609.29051)

**Mechanistic interpretability and steering**
- VERIFIED: "When Can One Neuron Fix Repetition Loops?" (Gemma-4):
  - **31B edit:** stripping 1,062 unique neurons (down_proj columns) at **layer 36**, selected by gradient×activation plus behavioural sweeps.
  - **Loop rates:** across four Gemma-4 models, held-out loops fall from 46/384 to 12/384, mainly in E4B and 31B, with no statistically detectable benchmark regression.
  - **Instability:** attribution is unstable (31B neuron-rank Spearman 0.122 across captures).
  - **31B residual failure:** at long budgets, 31B failures shift to "doom looping" (non-convergent self-correction).
  - **Repetition penalties** distort valid repetition.
  - **Antidoom FTPO baseline:** a preference method at detected loop boundaries (Liquid AI) gave 0/400 loops on Qwen3.5-4B when early-stopped. The full-epoch checkpoint degraded 6 of 7 capability benchmarks, and the same procedure gave no net gain on Gemma E4B.
  - **Released model:** the edited 31B is public as `edgerunner-ai/gemma-4-31B-it-noloop`.
  - **Scope:** the loops studied are enumeration-prompt loops, not agent tool-call loops. — [2606.13705](https://www.alphaxiv.org/abs/2606.13705)
- SNIPPET: activation-steering-to-weights work:
  - Steer2Edit converts steering directions into component-level weight edits ([2602.09870](https://www.alphaxiv.org/abs/2602.09870));
  - "Weight Updates as Activation Shifts" ([2603.00425](https://www.alphaxiv.org/abs/2603.00425));
  - the Halt Vector internalizes a causal steering intervention ([2608.28859](https://www.alphaxiv.org/abs/2608.28859));
  - Neural Procedural Memory does implicit activation steering for agents ([2606.29824](https://www.alphaxiv.org/abs/2606.29824));
  - "Does Fine-Tuning Undo Activation Steering?" ([2608.24988](https://www.alphaxiv.org/abs/2608.24988)).
- SNIPPET: "Suppression Sticks, Locality Is Fragile" audits task-vector negation in closed-loop VLA policies. Per the title, suppression persists but locality does not. — [2608.04692](https://www.alphaxiv.org/abs/2608.04692)

**Graph-structured credit and data refinement**
- VERIFIED: Dependency-Aware Trajectory Refinement:
  - an **LLM-annotated** round-level dependency DAG (GPT-5.4); out-degree-0 rounds are pruned before full-parameter SFT (Qwen3-VL-30B-A3B, multimodal QA);
  - leaf pruning: +1.7 pp over vanilla SFT;
  - LLM "critical path" deletion: −4 pp;
  - aggressive merges raised stuck-loop rates 2–3× because they broke the train/test interaction format. — [2609.18417](https://www.alphaxiv.org/abs/2609.18417)
- SNIPPET: graph or dependency credit assignment exists **only for online RL**:
  - dependency-aware policy optimization for terminal agents ([2610.03634](https://www.alphaxiv.org/abs/2610.03634));
  - graph-based credit assignment ([2605.26684](https://www.alphaxiv.org/abs/2605.26684));
  - GraphHCA ([2609.35084](https://www.alphaxiv.org/abs/2609.35084));
  - dependency-aware reward shaping ([2610.01207](https://www.alphaxiv.org/abs/2610.01207));
  - DeFA failure attribution ([2610.01256](https://www.alphaxiv.org/abs/2610.01256)).
- VERIFIED: Counterfactual Rollout Replay (CRR; **NeurIPS 2026**):
  - forks SWE sandboxes at high-entropy steps, rolls out an alternative action, and uses the return difference as the step advantage inside online GRPO;
  - 14B model, 41.7% vs 36.7% at equal wall-clock;
  - microbatch latency ×1.71;
  - 12% of forks use an immediately invalid alternative action. — [2609.33875](https://www.alphaxiv.org/abs/2609.33875)

**Offline RL, hindsight and data attribution**
- SNIPPET: hindsight relabelling for LLM agents (web and tool domains, not SWE): AgentHER ([2603.21357](https://www.alphaxiv.org/abs/2603.21357)) and "Spinning Straw into Gold" ([2607.04235](https://www.alphaxiv.org/abs/2607.04235)).
- SNIPPET: SWE-smith writes realistic issue text from a bug's diff, an F2P test and the test output, using an LM (NeurIPS 2025). — [SWE-smith](https://arxiv.org/pdf/2504.21798)
- SNIPPET: DRIFT uses the model's on-policy rollouts as influence-function validation targets for SFT data selection, with signed weighting by trajectory correctness (7B models). — [web search result](https://sotaverified.org/papers/260121996); Mechanistic Data Attribution — [PMLR v306](https://proceedings.mlr.press/v306/chen26cv.html)
- VERIFIED (prior session): AED's offline action-DPO pilot "establishes no recovery benefit". — [2609.40111](https://www.alphaxiv.org/abs/2609.40111)
- VERIFIED (prior session): Nebius found that filtering looping trajectories *increased* looping. — [2508.03501](https://arxiv.org/abs/2508.03501)

**Our own constraints (VERIFIED this session)**
- The v2 system prompt is 609 words / 3,985 bytes (`submissions/v2_sed/prompts/system.md`), roughly 0.8–1.1k tokens (INFERRED; not tokenized).
- A trajectory hits the 32k wall after about 20 full-size tool outputs, with no in-task compaction (`docs/HQ.md`).

### Inferences

Candidate cards follow. The novelty-search IDs (Q-numbers) refer to the search log in Q5. "Falsifier" means the cheapest test that can kill the idea.

**C1. DART-for-agents: state-preserving perturb-and-return augmentation (robotics → SWE)**
- **Mechanism:** inject Gemma's *own* failure types into the history of successful trajectories, and label with the clean trajectory's next action. Failure types: a duplicate of an earlier read call with its identical output; a malformed `edit_file` plus the real parser error; a failed `sed`/edit; a test rerun with no change.
- **Known pieces:** DART noise injection; PC-DAgger's calibrated noise with return-to-expert labels; OEC's covariate-shift evidence for SWE; SWE-Lego error masking and AED post-error SFT (prior notes); Feedback That Backfires (repeat probability 0.06 → 0.54 after a failure).
- **New piece:** perturbations are chosen to be **state-preserving**: a read-only duplicate, or a call that errors without changing files. So the clean next action is a *provably valid* label without any expert. Robotics needs a planner to compute the return path; here the return path is the identity.
  - The noise mix is **calibrated** to Gemma's measured failure histogram (PC-DAgger's idea): NPR share, malformed share, position in trajectory.
- **Targets:** exact-repeat loops (102/129), malformed args, post-failure covariate shift.
- **Feasibility offline on 4×L4:** high. Data is CPU-only (re-render public teacher trajectories plus own successes). One r16 LoRA is ≈2–3 GPU-h (INFERRED, pending Phase 0b).
- **Novelty searches:** Q2, Q12. Bench2Robust injects failures but trains with RL and finds vanilla SFT gave nothing. AED uses *real* errors. **No LLM-agent work using state-preserving synthetic perturbations with clean-continuation labels was found.**
- **Falsifier:**
  - Data: r2 traces (own states) for evaluation; ~300 public successful trajectories plus 3 perturbations each for training.
  - Arms: A = SFT on clean trajectories; B = A + perturbed copies, token-matched.
  - Metric: greedy one-step NPR re-emission at ≥150 held-out-repo *own* NPR states (from the critic's Stage 1), plus parse validity and held-out NLL.
  - Kill if B − A < 10 pp, or parse validity drops >3 pp.
  - Cost: ≈3 GPU-h.
  - Bench2Robust's null vanilla-SFT result is the main prior against this idea. Their SFT data was not state-preserving, and they did not report its details.

**C2. SCAD: same-model, information-symmetric harness distillation ("distil the harness you are not allowed to ship")**
- **Mechanism:** during *local* data generation only, our own runner (`run_official_l4.py`) appends deterministic annotations to observations:
  - "identical to call #k; no file changed since → output identical";
  - "you edited X and have not run a test since";
  - "step 31/40".

  Gemma runs with these annotations (teacher = same frozen base + annotation). The LoRA is trained on the resulting actions with the annotations stripped. Two variants:
  - (a) HAT-style SFT on annotated-run actions;
  - (b) OPCD-style token-level KL on student-visited states, using teacher logprobs with the annotation present.
- **Known pieces:** Harness-Zero, HAT, HAD, OPCD; PSP (privileged information in the loss fails).
- **New pieces:**
  1. The teacher is the **same model**: no stronger expert, which Harness-Zero shows is harmful when weak, but here the "reviewer" is a deterministic program.
  2. Annotations are **information-symmetric**: computable from the student's own visible context, so PSP's leak mechanism does not apply.
  3. The deployment harness is **impossible**: declarative ADK cannot run a loop detector, so internalization is the only route.
  4. SWE at 31B W4A16.
- **Targets:** NPR loops, untested fixes (47/129), budget overruns.
- **Feasibility:** medium. It needs a runner patch (our code, not the scorer's). Data costs about 3 GPU-h per 30–40 tasks at ~5 min/task.
- **Novelty searches:** Q3. Three close papers exist, so the novelty is **narrowed** to pieces (1)–(4).
- **Falsifier, Stage A (no training):**
  - Run v2 with and without annotations on the same 30 tasks.
  - Kill if annotations reduce the NPR count by <30% relative, **and** raise tested-after-last-edit by <10 pp. If the harness itself does not change behaviour, there is nothing to distil.
  - Cost ≈5–6 GPU-h (two 30-task runs). This also yields the training data.

**C3. Weight surgery compiled into a LoRA (mech interp → adapter)**
- **Mechanism:** the public 31B "noloop" edit zeroes 1,062 down_proj columns at L36. Write it as ΔW = −W_down[:,S]·E_Sᵀ, using our *dequantized W4A16* columns. Truncate it by SVD to rank ≤128 and ship it as a single-layer down_proj LoRA. Next step: redo the gradient×activation attribution on *agent* NPR states (repeat-call tokens) instead of enumeration prompts.
- **Known pieces:** 2606.13705 (Gemma-4 weight surgery), Steer2Edit, repetition neurons (cited therein).
- **New pieces:** (i) compiling sparse surgery into a servable low-rank adapter on a different (QAT int4) base; (ii) a test on tool-call loops.
- **Targets:** exact repeats.
- **Feasibility:** very high. **Zero training**: SVD on CPU, plus serving with prompt_logprobs.
- **Novelty searches:** Q14, Q15. No surgery-to-LoRA compilation was found.
- **Falsifier:**
  - Report the spectrum: fraction of ‖ΔW‖_F captured at r=128. (INFERRED risk: 1,062 orthogonal column selections may have a flat spectrum, so r=128 may keep too little.)
  - Measure greedy NPR re-emission at ~150 own states, with vs without the adapter.
  - Kill if the reduction is <5 pp, or if r=128 captures <50% of the energy *and* the full-rank merged edit (offline HF check) also fails.
  - Cost ≈1 GPU-h.
  - The paper itself says 31B residual failures become "doom looping", and that attribution is unstable. Strong prior that this transfers weakly.

**C4. Attribute-conditioned SFT with prompt-side conditioning (Decision Transformer / SteerLM → agents)**
- **Mechanism:** prefix each training trajectory with deterministic hindsight tags: `resolved=…`, `npr=…`, `tested_after_last_edit=…`, `steps=…`. Train on *all* data, including Gemma's own failed and looping runs. At inference, the declarative `system.md` requests the best tag values. That is the one channel we control at deployment.
- **Known pieces:** Nebius (filtering loops increases looping, so negatives carry signal); AgentHER (hindsight relabel). Decision Transformer and SteerLM are from memory, not re-read (INFERRED citation).
- **New piece:** conditioning on **process attributes** (loop count, test discipline) of SWE trajectories, with the condition delivered through the prompt of a declarative agent.
- **Targets:** loops, untested fixes, budget.
- **Feasibility:** high (plain SFT).
- **Novelty searches:** Q17. Nothing for SWE or agents.
- **Falsifier (controllability test):**
  - Train r16 on r1–r3 traces plus 500 public trajectories.
  - At inference, sweep the `npr` tag (0 vs 5) on 20 tasks.
  - Kill if NPR does not move monotonically with the tag (Spearman < 0.5 across tag values), or if the best-tag run is not better than untagged SFT.
  - Cost ≈4 GPU-h.

**C5. Same-model local corrections: "self-OEC" (DAgger without a teacher)**
- **Mechanism:** at Gemma's own failure states (NPR, malformed, post-error), fork the local sandbox and sample K=4 alternative actions at T≈0.8. Execute one step and keep alternatives that pass a deterministic progress oracle:
  - parses;
  - not an NPR;
  - observation is new;
  - preferably the trajectory then reaches a test.

  Optionally continue the branch to completion and keep resolving branches (OEC-style, with the student prefix masked).
- **Known pieces:** OEC (needs a stronger expert); CRR (forks, but online); Revisiting DAgger (needs a teacher).
- **New piece:** a **teacher-free** correction source (sampling plus deterministic oracle) used for offline SFT.
- **Targets:** loops, covariate shift.
- **Feasibility:** medium. Fork cost is real (CRR ×1.71). On L4 at ~34 tok/s it is limited to a few hundred states per session (INFERRED).
- **Novelty searches:** Q2, Q13.
- **Falsifier:**
  - At 150 own NPR states, measure the fraction where ≥1 of 4 samples (T=0.8) is a valid non-repeat.
  - Kill if <50%: no label supply.
  - Cost ≈1 GPU-h.

**C6. Long-context ("playbook") distillation (context distillation beyond the window)**
- **Mechanism:** distil a context that *cannot* fit at 32k into a LoRA, e.g. 20–60k tokens of worked recovery episodes and verified procedures from other repos. The teacher runs offline with a larger `max_model_len` for logprob scoring only.
- **Known pieces:** OPCD, LatentSkill, Parametric Skills (SWE), Doc-to-LoRA.
- **New piece:** a distilled context longer than the deployment window, for a declarative SWE agent.
- **Targets:** generic procedure; indirectly loops and testing.
- **Feasibility:** uncertain. A ≥64k teacher on 4×L4 W4A16 is unmeasured; Gemma-4 sliding-window layers may make it possible (INFERRED).
- **Novelty searches:** Q1, Q4. Parametric Skills already does SWE skills → LoRA, so novelty is limited to "longer than the window".
- **Falsifier:**
  - Does the teacher *with* the playbook beat no-playbook on 30 tasks?
  - Kill if Δresolve ≤ 0 and NPR is unchanged.
  - Cost ≈6 GPU-h, plus a feasibility probe of the long-context teacher.

**C7. Conditional steering LoRA (Contrastive Activation Addition → rank-1 gated adapter)**
- **Mechanism:**
  - Capture residual activations at matched states with vs without C2's annotation.
  - Fit v as the direction that detects "this exact call is already in context", and u as the annotation-induced shift.
  - Ship ΔW = u vᵀ on one layer's o_proj/down_proj.
- **Key constraint insight (INFERRED):** a constant steering vector is a *bias*. vLLM requires `bias="none"`, so plain CAA is **not servable**. Only input-gated low-rank updates are, and that is exactly what conditional steering needs.
- **Known pieces:** Steer2Edit; Weight-Updates-as-Activation-Shifts; Halt Vector. CAA itself is from memory.
- **New piece:** annotation-contrast steering for agents, compiled to a servable rank-1 adapter.
- **Targets:** loops.
- **Feasibility:** high (≈1 GPU-h activation capture).
- **Novelty searches:** Q11, Q15.
- **Falsifier:** same NPR-state metric as C1. Kill if <5 pp.

**C8. Anti-loop task negation (model merging → behaviour removal)**
- **Mechanism:** train a LoRA *on* Gemma's own NPR continuations (learning the loop), then ship it negated (scale −λ). Packaging is trivial: negate B.
- **Known pieces:** task-arithmetic negation (Ilharco et al., from memory); VLA negation audit, where suppression sticks but locality is fragile.
- **New piece:** behavioural negation against agent loops at 31B.
- **Targets:** loops.
- **Feasibility:** high (≈2 GPU-h).
- **Novelty searches:** Q10.
- **Falsifier:** NPR re-emission plus a guard on post-change repeats (PCR) plus parse validity. Kill if PCR log-prob falls >25% relative at any λ that gives ≥10 pp NPR reduction; the locality failure predicted by 2608.04692.
- Useful mainly as a **baseline** for SCCS / C1.

**C9. Data-flow-graph-weighted SFT (graph credit assignment → offline loss weights; graph-paper bonus)**
- **Mechanism:** build a *deterministic*, program-derived information-flow DAG for each SWE trajectory. Edge rules:
  - a read of file F lines a–b → a later edit touching those lines;
  - grep output naming a symbol → a later read of that symbol;
  - a failing test output → the next edit;
  - an edit → a test run.

  Use the DAG as **loss weights or masks, not pruning**. This keeps the interaction format, given 2609.18417's stuck-loop finding. Upweight nodes on paths into the final patch and its verification; zero-weight dead-end leaves and NPR duplicates.
  - This can be extended with IFG's executed call graph, weighting edits by executed-graph necessity.
- **Known pieces:** 2609.18417 (LLM-annotated DAG, pruning, QA); online-RL graph credit papers; IFG (our graph report).
- **New pieces:** (i) a program-derived DAG with no LLM annotator; (ii) used as SFT loss weights; (iii) SWE; (iv) a link to executed-graph necessity.
- **Targets:** loops (indirectly), multi-function fixes (via the IFG link).
- **Feasibility:** high. The DAG is CPU-only.
- **Novelty searches:** Q7, Q8.
- **Falsifier:**
  - First check DAG agreement with an LLM annotator on 50 trajectories: kept-node F1 ≥0.8.
  - Then compare graph-weighted SFT with vanilla SFT on held-out own-state metrics and patch-token NLL.
  - Kill if both deltas are <2%.
  - Cost ≈4 GPU-h.

**C10. Multi-locus bug composition plus k-locus curriculum (curriculum / self-paced → multi-function dead zone)**
- **Mechanism:** compose 2–3 SWE-smith-style single-function bugs on *executed-graph-adjacent* functions to synthesize multi-function tasks. Order training by number of loci, self-paced by Gemma's success.
- **Known pieces:** SWE-smith; IFG premise; curriculum and self-paced learning (from memory).
- **New piece:** graph-adjacency composition plus a k-locus curriculum.
- **Targets:** multi-function fixes (2/45).
- **Feasibility:** low by Nov 5. Without a teacher, Gemma's success on 3-locus tasks is about 4%, so few positive trajectories.
- **Novelty searches:** IFG log (graph report). Nothing new this session.
- **Falsifier:** count multi-function (≥2 gold functions) successes in public teacher datasets (CPU). Kill if <300 usable.

**C11. Own-failure-state influence selection (data attribution → trajectory selection)**
- **Mechanism:** score public trajectories by LoRA-gradient similarity (LESS/TracIn-style) to the *desired actions at Gemma's own failure states* (from C1/C5). Select the top-k for SFT.
- **Known pieces:** DRIFT (on-policy influence); LESS (from memory).
- **New piece:** the attribution target is own failure states of an agent, not task-validation loss.
- **Targets:** all failures, through data choice.
- **Feasibility:** medium-low. Gradient features at 31B cost an estimated 6–10 GPU-h for about 2k trajectories (INFERRED).
- **Novelty searches:** Q9 (alphaXiv: none), Q16 (web: DRIFT).
- **Falsifier:** top-500 by influence vs SWE-Prime-filtered 500 vs random 500, equal tokens. Kill if influence beats SWE-Prime by <3 pp on own-state metrics.

**C12. Hindsight issue relabelling (HER → SWE)**
- **Mechanism:** for Gemma's failed runs whose patch changes behaviour:
  - synthesize the issue that the patch *does* fix (SWE-smith-style issue writing from the diff plus a behaviour-diff test);
  - keep the trajectory as a positive demonstration for that relabelled goal, with NPR segments masked.
- **Known pieces:** HER (from memory); AgentHER; Spinning Straw; SWE-smith issue generation.
- **New piece:** hindsight relabelling of own SWE trajectories (none found).
- **Targets:** untested fixes (more own-positive data), covariate shift.
- **Feasibility:** medium (CPU plus test generation).
- **Novelty searches:** Q13, Q16.
- **Falsifier:** human or LLM audit of 30 relabelled trajectories. Kill if >40% read as goal-incoherent: exploration aimed at the old issue looks irrational for the new one. This is the known weakness of hindsight relabelling when actions encode the goal.

**C13. Q-SFT / implicit-Q with programmatic step rewards (offline value-based RL → LoRA)**
- **Mechanism:** use token-level Q-learning cast as SFT (Q-SFT, ILQL; from memory) with deterministic step rewards: −1 NPR, −1 malformed, +1 test after edit, plus outcome.
- **New piece:** offline value-based training for SWE tool agents with programmatic process rewards.
- **Feasibility:** low-medium. It needs several epochs; stability at 31B LoRA is unknown.
- **Novelty searches:** Q6. No LLM-agent Q-SFT hits returned.
- **Falsifier:** vs C1 at equal tokens on the NPR metric. Kill if not better.
- AED's null offline action-DPO result (prior notes) is a prior against offline value or preference signals for recovery.

**C14. Declarative recovery policy (robotics recovery controllers → second adapter)**
- **Mechanism:** a SequentialAgent where fixer adapter A runs to a step cap. Then a "verify/recover" sub-agent with adapter B (fresh context plus `output_key` summary) is trained on recovery and verification data.
- **Known pieces:** Recovery RL (from memory); our v4/v5 pipelines gained nothing on the LB (VERIFIED, CLAUDE.md).
- **Feasibility:** high mechanically. Payoff prior is low, given the v4/v5 history and the time budget.
- **Kill:** if the critic's H6 (per-role adapters) logic holds. **Low rank.**

**C15. Test-time training: killed**
- The scorer is declarative: no Python entrypoints, adapters are fixed files, ~5 min/task. In-task weight updates are impossible.
- The only offline analogue, per-repo adapters, cannot target private hidden repos.
- The search (Q18) found only test-time *compute* scaling for SWE.

**C16. Hypernetwork per-task LoRA (Text-to-LoRA / Doc-to-LoRA): killed**
- It needs runtime adapter generation; ≤8 fixed adapters are allowed.
- LatentSkill also shows independently trained per-skill LoRAs underperform in-context skills, so an offline "adapter bank" is weak too.

### Gaps
- No direct read of OpenReview ICLR-2027 submissions. Absence claims are arXiv-index only.
- Citations from memory, not re-read this session: DART original (cited inside PC-DAgger's reference list), Decision Transformer, SteerLM, CAA, task arithmetic (Ilharco), LESS, HER, Q-SFT/ILQL, Recovery RL.
- Bench2Robust's vanilla-SFT recipe details (what the SFT data contained) were not read. This matters for C1's prior.
- DRIFT was seen only via a web snippet; the paper was not read.
- Whether vLLM 0.19.1 serves a LoRA that targets **one layer's** down_proj only (C3, C7) is unverified. PEFT can express it; vLLM per-module coverage is INFERRED to work, but `rank_pattern` support is unknown (prior note).

---

## Q2. Context distillation of the long system prompt and skills into a LoRA: does it pay under the 32k cap?

### Takeaway

**Distilling our actual system prompt is low-payoff:**
- it is only ~1k tokens, about 3% of 32k;
- the prompt channel is the one deployment lever we keep;
- OPCD and LatentSkill evidence is ≤8B, single-turn or short-horizon.

Context distillation becomes valuable in two other forms:
1. **Distil the harness we cannot ship** (C2/SCAD): the "context" is per-step, state-dependent annotations that a declarative agent cannot produce.
2. **Distil content longer than the window** (C6).

The evidence says do it with a **frozen** same-base teacher (OPCD: frozen ≫ self), information-symmetric context (PSP), and moderate injection strength (LatentSkill's inverted U).

### Cited Findings
- VERIFIED: v2 `system.md` is 609 words / 3,985 bytes (this session). — local file `submissions/v2_sed/prompts/system.md`
- VERIFIED: the 32k wall is hit after about 20 full-size tool outputs, with no in-task compaction. — `docs/HQ.md` (local)
- VERIFIED: OPCD system-prompt distillation beats in-context prompting (e.g., 76.7 vs 72.2 on Llama-3.1-8B medical). Off-policy context distillation also beats in-context (75.2). Teacher-student with a frozen teacher ≫ self-distillation (53.9 vs 18.8 Sokoban), and self-distillation can diverge. OPCD better preserves out-of-distribution ability (IF-Eval) than off-policy context distillation. — [2602.12275](https://www.alphaxiv.org/abs/2602.12275)
- VERIFIED: in-context skills *hurt* out of domain (WebShop 6.7 vs 10.3 vanilla), while compiled skills helped. Over-strong injection collapses success (α=1.2). — [LatentSkill 2606.06087](https://www.alphaxiv.org/abs/2606.06087)
- VERIFIED: harness behaviours internalize unevenly: "inspect before edit" ~100%, pagination/read-back 40–50%. Procedural behaviour transfers better than domain knowledge (USPTO 30% distilled vs 38% with harness). — [Harness-Zero 2609.24974](https://www.alphaxiv.org/abs/2609.24974)
- VERIFIED: HAD keeps state tracking in the harness. Hiding state tracking dropped HAD 63.4 → 37.3%, while behaviour monitoring was partly internalized. — [2610.02858](https://www.alphaxiv.org/abs/2610.02858)
- VERIFIED (prior session): privileged-hint self-distillation fails on SWE. — [PSP 2609.29051](https://www.alphaxiv.org/abs/2609.29051)

### Inferences
- **Payoff arithmetic (INFERRED):**
  - Distilling the system prompt frees about 1k (plus maybe 1–2k of instruction-like tool docs, which ADK injects anyway) of 32k. That is roughly one extra tool output before the wall.
  - Our dominant failures (repeats, no testing) are not caused by a long prompt.
  - A LoRA-internalized prompt cannot be A/B-edited per submission the way `system.md` can. That costs us our cheapest iteration channel under the one-submission-per-day limit.
- **Why SCAD is the right form of context distillation here (INFERRED):**
  - HAD's finding that *state tracking* is the hardest part to internalize argues for annotations that are cheap re-derivations of visible context ("identical call already made at step k"). Hard global state belongs elsewhere.
  - Our NPR label is exactly such a re-derivation.
  - The 102/129 repeat prevalence gives a large target.
- **Practical recipe (INFERRED):**
  - Variant (a), SFT on annotated-run actions with annotations stripped, needs only our existing runner plus a patch. Run it first.
  - Variant (b), OPCD reverse-KL, needs teacher top-k logprobs. vLLM typically caps returned logprobs (about 20; unverified for 0.19.1) vs OPCD's top-256. So expect a degraded top-k KL, or use top-20 forward-KL on teacher-run tokens.
- **Risk ("When Context Returns"):** if the trained LoRA also sees a prompt containing the same rules, effects may double-apply. Hold the prompt fixed across arms.

### Gaps
- No study distils agent *harness annotations* with a same-model teacher on SWE at ≥30B.
- The vLLM 0.19.1 `max_logprobs` / prompt-logprobs top-k limit was not checked.
- The tool-schema token count in the scorer's rendered prompt was not measured.
- The "When Context Returns" full text was not read.

---

## Q3. Recovery-data augmentation for loops and covariate shift: what transfers from robotics and imitation learning?

### Takeaway

Robotics says covariate shift is fixed by **training on states the learner will actually visit, with labels that steer back to the nominal trajectory**: DAgger, DART, PC-DAgger. The noise should be **calibrated to the learner's own error distribution** and **shrink as the learner improves**.

SWE agents have a property robots lack: many of Gemma's failure actions (duplicate reads, rejected malformed calls) **do not change environment state**. The "return to the expert trajectory" label is therefore just the clean next action, which makes DART-style augmentation essentially free and provably label-correct (C1).

The strongest counter-evidence: Bench2Robust found **vanilla SFT on injected-failure data gave no gain**, and AED's offline DPO gave no recovery gain. Recovery may need on-policy states (C5, OEC), not synthetic ones.

### Cited Findings
- VERIFIED: covariate shift between student and expert grows over SWE trajectories even after behaviour cloning. OEC (partially on-policy, expert-corrected) beats pure BC, and pure on-policy data hurts at 32B. — [2512.14895](https://www.alphaxiv.org/abs/2512.14895)
- VERIFIED: calibrated noise around expert interventions, with return-to-trajectory labels, adds +2.8 pts over HG-DAgger and matches the best swept noise level. Optimal noise decreases over rounds. — [2609.30462](https://www.alphaxiv.org/abs/2609.30462)
- VERIFIED: tool-failure injection exposes a near-universal robustness gap (69/70 model-subset pairs degrade). Vanilla SFT and GRPO did not close it; curriculum RL plus runtime recovery context did. — [2608.11977](https://www.alphaxiv.org/abs/2608.11977)
- VERIFIED (prior session): Feedback That Backfires measured repeat probability 0.06 → 0.54 after a failure, 83% surface-driven, ≤1.7B models only. Contextual Drag: targeted SFT costs 22–28 pp when context is correct. Nebius: filtering loops increased looping. — see `critic_judgement.md` ([2608.23651](https://www.alphaxiv.org/abs/2608.23651), [2602.04288](https://www.alphaxiv.org/abs/2602.04288), [2508.03501](https://arxiv.org/abs/2508.03501))
- VERIFIED: in Gemma-4-31B, a static weight edit removed enumeration loops but residual failures became "doom looping". The FTPO preference baseline over-trained easily (the full epoch degraded 6/7 capability benchmarks on Qwen). — [2606.13705](https://www.alphaxiv.org/abs/2606.13705)

### Inferences
- **Design for C1 (INFERRED):**
  - Perturbation types and rates should match Gemma's measured distribution in r2 traces:
    - NPR position histogram;
    - malformed-call rate by tool;
    - post-error repeat rate.
  - Per PC-DAgger, decay the perturbation rate across generate-then-train rounds.
  - Always keep a clean, unperturbed copy, which Contextual Drag's −22–28 pp cost requires as a guard.
  - Include legitimate repeats (rerun the test *after* an edit) as positives.
- **Why C1 may beat Bench2Robust's null SFT (INFERRED):** their failures were *environmental*: stochastic API errors whose right response depends on hidden recoverability. Ours are *self-generated* and their right response is determined by visible context (the output is already known). That is a far easier supervised target.
- **Combine with critic's SCCS:** C1 supplies positive "return" labels at synthetic states; SCCS supplies negatives at own states. Together they form a DAgger-like bracket that needs no teacher. But keep one change per experiment.
- **C5 (self-OEC) is the on-policy complement** if C1's synthetic states fail to transfer to own states. That outcome would be the robotics lesson that synthetic noise ≠ policy noise.

### Gaps
- No ≥30B replication of Feedback That Backfires; the copy mechanism at 31B is untested (critic's Stage 1 measures it).
- No LLM-agent study measures whether synthetic-perturbation states and own visited states are close (e.g., embedding Fréchet distance as in OEC). This is a cheap diagnostic to add.

---

## Q4. Top 5 by novelty × payoff × feasibility, and why each could fail

### Takeaway

Ranking, with novelty / payoff / feasibility scored 0–3:

| Rank | Idea | N | P | F | Why |
|---|---|---|---|---|---|
| 1 | **C1 DART-for-agents** (state-preserving perturb-and-return) | 2 | 3 | 3 | Hits the most widespread failure (102/129); provably valid labels; no teacher; CPU data |
| 2 | **C2 SCAD** (same-model, information-symmetric harness distillation) | 2 | 3 | 2 | The only form of context distillation that pays here; targets loops, testing and budget together; novelty narrowed by Harness-Zero/HAT/HAD |
| 3 | **C3 Weight surgery → LoRA** | 2 | 1–2 | 3 | Zero-training falsifier in ≈1 GPU-h; new compile step; weak transfer prior |
| 4 | **C4 Attribute-conditioned SFT** with prompt-side tags | 2 | 2 | 3 | Uses failed and looping data instead of discarding it; exploits the prompt channel |
| 5 | **C9 Data-flow-graph-weighted SFT** | 2 | 1 | 3 | Ties tuning to graph structure for the paper; deterministic DAG; modest LB payoff |

Next in line: C5 self-OEC (the on-policy fallback for C1) and C7 conditional steering LoRA. C15/C16 are killed.

### Cited Findings
- See Q1–Q3 for the evidence behind each score. The key priors are:
  - OEC: on-policy correction works for SWE but needed a stronger expert ([2512.14895](https://www.alphaxiv.org/abs/2512.14895));
  - Bench2Robust: vanilla SFT on injected failures gave nothing ([2608.11977](https://www.alphaxiv.org/abs/2608.11977));
  - Harness-Zero / HAT / HAD: harness internalization works at 9–35B on non-SWE or SWE-QA tasks ([2609.24974](https://www.alphaxiv.org/abs/2609.24974), [2610.01235](https://www.alphaxiv.org/abs/2610.01235), [2610.02858](https://www.alphaxiv.org/abs/2610.02858));
  - noloop: weight surgery reduces Gemma-4 enumeration loops, but 31B shifts to doom looping ([2606.13705](https://www.alphaxiv.org/abs/2606.13705));
  - dependency-DAG refinement: +1.7 pp in QA, and format-breaking edits increase stuck loops ([2609.18417](https://www.alphaxiv.org/abs/2609.18417)).

### Inferences

**Why each could fail (honest):**

1. **C1**
   - Synthetic perturbations may not resemble Gemma's own looping states. Its loops arise after long, confusing contexts, not after a single injected duplicate. The model may learn "ignore the second identical block" without learning "don't emit the third".
   - Bench2Robust's null SFT result is a direct warning.
   - Teacher trajectories re-rendered into our format may themselves be off-distribution (OEC's covariate-shift result), diluting the effect.
   - The NPR metric may improve one step at a time while closed-loop resolve does not move: the 2609.21187 lesson, cited in the critic note.
2. **C2**
   - Annotations may not change Gemma's behaviour even in context. Feedback That Backfires shows error text can *increase* repeats, so Stage A may kill it.
   - Distilled behaviours may stay context-specific (HAT: further annealing is not uniformly better).
   - Data generation costs full agent runs (≈5 min/task), so only about 100–150 trajectories per session.
   - Reviewers could say it is "HAT with a programmatic harness", which caps paper novelty.
3. **C3**
   - The edit was selected on enumeration prompts. Agent tool-call repeats are probably in-context copying (induction-style), a different mechanism.
   - Rank-128 truncation of a 1,062-column edit may lose the effect.
   - Attribution instability (Spearman 0.12) means redoing it on agent states may not converge.
   - The 31B residual failure (doom looping) is closer to our "repeat with no progress" than tight loops are.
4. **C4**
   - Conditioning generalizes poorly to tag values that are rare in training. Good-process trajectories are rare in our own data, so it leans on public data.
   - Tags may be ignored if the LoRA learns the dominant mode.
   - At inference the model may "perform" low `npr` by superficially varying arguments: near-duplicates that dodge the exact-repeat metric. Measure near-repeats too.
5. **C9**
   - The program-derived DAG may miss reasoning dependencies: a read that shaped thinking without a lexical carry-over gets wrongly zero-weighted.
   - Loss reweighting at 1–2k trajectories may be within noise.
   - The LB cannot resolve the likely effect size (1 task ≈ 0.8 pp).
   - Its value is mostly the paper's graph story.

**Suggested order (INFERRED):**
1. Run C3's zero-training test and C2 Stage A first. They are the cheapest kills, ≈1 and ≈5 GPU-h, and Stage A doubles as data generation.
2. Then the C1 one-step test (≈3 GPU-h), sharing the critic's Stage-1 NPR state set.
3. C4 and C9 ride as auxiliary arms on the mainline SFT (E7) once Phase 0b is measured.

Total ≈15 GPU-h before any closed-loop evaluation, inside the critic's ~50 GPU-h plan slack.

### Gaps
- All effect sizes are GUESSED. None of the five has an SWE-specific precedent at ≥30B with LoRA.
- Phase 0b (31B QLoRA on 4×L4) is unmeasured, so every training cost above is provisional.

---

## Q5. Search log: what was searched and what came back

### Takeaway

27 queries and fetches: alphaXiv discovery and PDF reads, two web searches, and local repo reads. The searches confirmed:
- crowded areas: harness distillation, graph credit assignment (online RL), skill-to-LoRA;
- empty areas: state-preserving perturbation SFT for agents, surgery-to-LoRA compilation, attribute-conditioned process tags for SWE, SWE hindsight relabelling, influence selection on own failure states.

### Cited Findings

| # | Tool | Query / target | Result used |
|---|---|---|---|
| L1 | Local | `CLAUDE.md`, `docs/HQ.md` (grep), `critic_judgement.md` (full), headings of the 4 other notes, graph report sections, `submissions/v2_sed/prompts/system.md` (wc) | Constraints; failure stats; prompt = 609 words / 3,985 bytes |
| Q1 | alphaXiv discover | context distillation of prompts/skills into LoRA for agents | Doc-to-LoRA, **OPCD**, When Context Returns, **LatentSkill**, continual experience internalization, LoRA-as-memory, SKILL0, Skill0.5, K2P, CD as latent memory |
| Q2 | alphaXiv discover | noise injection / DART-style recovery data for LLM agents | **PC-DAgger**, **OEC 2512.14895**, perturbation-for-recovery 2609.33049, rollback correction, AgentRewind, Revisiting DAgger, RoG-DAgger, AED, retry-supervised value learning |
| Q3 | alphaXiv discover | distilling harness interventions into weights | **Harness-Zero**, **HAD**, **HAT**, distillation roadmap, diffusion OPCD, Harness-R1, recursive harness self-improvement, harness learning |
| Q4 | alphaXiv discover | Parametric Skills / Skill-to-LoRA SWE | Parametric Skills, Skill-to-LoRA, LatentSkill, COBRA-Skills, "New LoRA Skills Should Read but Never Write", SWE-Skills-Bench, skills-can-be-harmful |
| R1 | alphaXiv PDF | OEC 2512.14895 | Numbers in Q1/Q3 |
| R2 | alphaXiv PDF | LatentSkill 2606.06087 | Numbers in Q1/Q2 |
| R3 | alphaXiv PDF | OPCD 2602.12275 | Numbers in Q1/Q2 |
| R4 | alphaXiv report | Harness-Zero 2609.24974 | Numbers in Q1/Q2 |
| R5 | alphaXiv full text (grep) | HAT 2610.01235; HAD 2610.02858 | Abstract, intro and Gemma table |
| Q5 | alphaXiv discover | return-conditioned / hindsight / goal-conditioned BC for LLM agents | AgentHER, Spinning Straw, hindsight memory PRM, trajectory-relative hindsight distillation, HINT-SD, offline-RL harness control. No SWE return-conditioning |
| Q6 | alphaXiv discover | Q-SFT / IQL / CQL / AWR for LLM agents | Nothing LLM-agent-specific (robotics/offline-RL only) |
| Q7 | alphaXiv discover | graph-structured credit assignment for agent trajectories | Dependency-aware PO (terminal), **dependency-aware trajectory refinement**, graph-based credit assignment, GraphHCA, SHARPO, DeFA, dependency reward shaping, EDGE, MileGPO, VICT, TRACE. Credit-assignment papers are online RL; the one SFT paper uses an LLM-annotated DAG |
| R6 | alphaXiv PDF | 2609.18417 | Numbers in Q1 |
| Q8 | alphaXiv discover | induction / copy-suppression heads and repetition | Induction-head ablation, residual-stream repetition, LoopGuard, **One Neuron (Gemma-4)**, Wait-Wait-Wait looping |
| R7 | alphaXiv PDF | One Neuron 2606.13705 | 31B L36 edit, doom looping, FTPO comparison |
| Q9 | alphaXiv discover | influence functions for agent trajectory selection | **No papers found** |
| Q10 | alphaXiv discover | LoRA task negation | Per-parameter task arithmetic, MAAT, **VLA negation audit**, unmerge, etc. |
| Q11 | alphaXiv discover | steering vectors baked into LoRA / CAA / repetition | Steering-recovery-after-fine-tuning, Neural Procedural Memory, First-Order Steering, LocUS, Steer2Edit |
| Q12 | alphaXiv discover | synthetic perturbation of expert trajectories for agent recovery | AED, **Retry/Switch/Abstain**, PC-DAgger, AgentChaos, ToolMisuseBench, Kintsugi-VLA. No state-preserving perturbation SFT for LLM agents |
| R8 | alphaXiv PDF | PC-DAgger 2609.30462; Retry/Switch/Abstain 2608.11977 | Numbers in Q1/Q3 |
| Q13 | alphaXiv discover | hindsight relabel failed SWE patches with new issues | FC-SWE, XRepoSkill, H²SD, FailForge, **CRR**, Harness-R1, AED, RETRACE, RESKILL, Socratic-SWE. No SWE hindsight relabel |
| R9 | alphaXiv PDF | CRR 2609.33875 | Numbers in Q1 |
| Q14 | alphaXiv discover | compiling steering / sparse edits into LoRA | Steer2Edit, training-free task vectors, Weight-Updates-as-Activation-Shifts, First-Order Steering, Steer2Adapt, Halt Vector. No surgery→LoRA compilation |
| Q15 | (same as Q14) | — | — |
| Q16 | WebSearch | influence functions / data attribution for LLM agent fine-tuning 2026 | DRIFT (snippet), Mechanistic Data Attribution (PMLR) |
| Q16b | WebSearch | hindsight relabel failed SWE patches / backtranslation / SWE-smith | SWE-smith issue generation (snippet). No hindsight relabel |
| Q17 | alphaXiv discover | outcome / quality-tag-conditioned SFT for agent trajectories | AED, Harness-R1, Terminal-Universe, AgentBrew, DENSE, SWE-Prime, FailForge, PG-SFT. **No process-tag conditioning** |
| Q18 | WebSearch | test-time training LoRA SWE agent repo-specific | Only test-time *compute* scaling (Thinking Longer, Not Larger) and SWE-Eff-14B LoRA. No TTT |

### Inferences
- Novelty verdicts:
  - **Narrowed:** C2, C6, C9 (adjacent 2026 work exists).
  - **Open:** C1, C3, C4, C11, C12 (no adjacent work found).
  - **Killed on constraints:** C15, C16.

### Gaps
- Not searched:
  - OpenReview directly;
  - Semantic Scholar venue status for the 2026 preprints;
  - full texts of DRIFT, Parametric Skills, When Context Returns, Steer2Edit, AgentHER, Antidoom/FTPO (Liquid AI);
  - vLLM 0.19.1 `max_logprobs` and single-layer LoRA serving;
  - Bench2Robust's SFT data details.
- Q15 was a duplicate slot. Only 27 distinct actions were run.
