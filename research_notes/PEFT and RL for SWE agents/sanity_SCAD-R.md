# Sanity check: SCAD-R (same-model, information-symmetric harness distillation) against the current literature

Reviewer pass, 2026-10-08. Scope: the SCAD-R description in `reports/PEFT and RL for SWE agents.md` (Round 4, rubric table, "SCAD-R: one step, ~1 GPU-h"), `critic_judgement.md` R4 and `cross_domain_ideas.md` C2. The question here is the field, not our notebook.

Labels: **VERIFIED** means I read the primary text (full text or a structured full-paper report) in this pass. **SNIPPET** means title/abstract/search snippet only, or a reference seen inside another paper's bibliography. **INFERRED** means my reasoning. Query IDs S1–S22 refer to the Search log at the end.

**One-paragraph verdict.** The gap is **narrower than the report says, and one load-bearing argument is wrong**. Three close prior works were missed. (1) **Memento No More (MNM, 2502.01562)** already does iterative LoRA context distillation of *state-triggered* corrective hints into the *same* 70B model for multi-turn tool agents. Its mistake detectors are partly scripts (string match, AST checks), and it uses hint dropout (p=0.9) rather than full stripping. (2) **OPHSD (2605.08741)** self-distills a *deterministic programmatic harness* into the same model and removes it at inference, though only for single-turn tasks. (3) **HAD's harness (2610.02858)** is already deterministic and uses almost SCAD-R's exact note templates ("Step t of cap", "did not work here (attempt N)", "Nothing new observed for k steps"); HAD keeps the harness at deployment and uses a bigger teacher. The argument that fails is the claim that "PSP's lesson (privileged information in the sampler, not the loss) shapes SCAD-R". SCAD-R *strips* the notes from the training context, so it **is** cross-view context distillation in hard-label form, which is exactly what PSP warns against. SCAD-R is protected only by **information symmetry**, which does make PSP's irreducible-gap term zero. Symmetry does **not** guarantee *computational* realizability: counting steps toward a budget is the weak case. A 2026 dissection of OPSD shows most "privileged" gains can come from **context-induced teacher behaviour** rather than from the information. SCAD-R therefore needs a **placebo-note arm** and a **rules-in-system-prompt arm**, and has neither. The surviving contribution is an **analysis** that decomposes harness-note gains into information, salience and generic-context effects, and asks which note types internalize at 31B on SWE. That analysis is best framed as Z's "tell vs. punish" treatment arm, not as a standalone method paper.

---

## 1. Prior art the previous passes did not find

### Takeaway
Same-model internalization of per-step agent hints with LoRA is **not new**: MNM did it in 2025 at 70B. Deterministic-harness self-distillation is **not new**: OPHSD did it in 2026 for single-turn tasks. SCAD-R's note vocabulary is **already in HAD**. What remains new is SWE resolution, ≥30B W4A16, *harness impossible at deployment*, and *purely state-derived (no human-written) notes*. Each of these is a setting difference, not a mechanism difference.

### Cited Findings

**Closest: hint internalization for multi-turn agents (missed)**
- VERIFIED: MNM ("Memento No More: Coaching AI Agents to Master Multiple Tasks via Hints Internalization", Alakuijala et al., Aalto / System 2 AI) works as follows:
  - **Teacher.** Llama-3.1-70B-Instruct with hints `h_i(s)` in context.
  - **Student.** The same weights plus a LoRA, trained with forward KL to match `π_θ(a|s,h(s))` while seeing only `s`.
  - **Rounds 2+.** "Reviewers" (scripts using string matching / AST checks, or an LLM judge) detect the student's mistake states. Corrective hints are "state-specific but apply to all instances of the same error type". Corrected actions are sampled at those states and distilled, in an explicitly DAgger-like loop.
  - **Results.** ToolQA 97.9% and OfficeBench 90.3% without hints, vs GPT-4o 92.8 / 89.9 with long prompts.
  - **Ablations.** KL loss is slightly better than CE. **Hint dropout p=0.9 is optimal** (not full removal). Data balancing adds 1.4–1.5 pp.
  - **Limitation.** The hints are human-written. — [2502.01562](https://www.alphaxiv.org/abs/2502.01562) (S14, S15)
- SNIPPET: MNM describes itself as the first extension of context distillation to training LLM agents for interactive tasks. — [2502.01562](https://www.alphaxiv.org/abs/2502.01562) (S15)

**Deterministic-harness self-distillation (missed)**
- VERIFIED: On-Policy Harness Self-Distillation (OPHSD; Zhao, Ma, Zhang, PKU, 2605.08741):
  - **Harness.** A "deterministic, stateful program that wraps the language model". It is also fed privileged `z(x)`, such as a memory bank or reference solution.
  - **Teacher.** The frozen *same* model on the harness's terminal context.
  - **Student.** Reverse-KL on its own harness-free rollouts.
  - **Tasks.** Qwen3-8B; draft–verify classification and plan–solve math (single-turn).
  - **Results.** Beats OPSD by +2.82 pass@8 on average in math. Re-attaching the harness after training *hurts* classification (−7.19 USPTO).
  - **Gaps relative to SCAD-R.** No multi-turn agent and no SWE. — [2605.08741](https://www.alphaxiv.org/abs/2605.08741) (S18 citation → S19)
- VERIFIED: HAD's harness is "computed deterministically from the environment state and interaction history". Its templates include:
  - Budget: "Step {t} of {cap}" plus "Only {left} steps remain";
  - Failure: "'{action}' did not work here (this was attempt {N})";
  - Stall: "Nothing new has been observed for {k} steps".

  Thresholds are calibrated so that the lines fire on ≤5% of steps of successful validation episodes. — [2610.02858](https://www.alphaxiv.org/abs/2610.02858) (S18)

  This is SCAD-R's note set almost verbatim. HAD differs in keeping the harness at deployment and in using a larger teacher (12B → E2B on Gemma-4).
- SNIPPET: HAD's related work cites "Kim et al., 2026", which trains compact agents across varying harness configurations. Neighbouring work treats harness variation as a deployment reality:
  - CHART, a harness-rotation curriculum ([2609.22247](https://www.alphaxiv.org/abs/2609.22247));
  - HarnessBandit ([2609.13739](https://www.alphaxiv.org/abs/2609.13739)). (S18, S20)

**Feedback-conditioned self-teacher (missed as a lineage)**
- VERIFIED: SDPO (Hübotter et al., ETH/MIT/Stanford, 2601.20802):
  - **Teacher.** The current policy re-prompted with *environment feedback* (runtime errors, failing tests). It scores the student's own attempt token by token: `KL(π(·|x,y<t) ‖ sg π(·|x,f,y<t))`.
  - **Results.** LiveCodeBench v6: 48.8 vs GRPO 41.2 on Qwen3-8B. Gains grow with model scale.
  - **Scope.** Single-turn. "Long-horizon and agentic settings" are listed as future work. — [2601.20802](https://www.alphaxiv.org/abs/2601.20802) (S3, S5)

  SCAD-R's notes are a form of synthesized feedback `f`, so this is the parent mechanism.
- VERIFIED: SDAR (2605.15155):
  - **Setup.** Multi-turn OPSD with a skill-conditioned self-teacher, added to GRPO.
  - **Failure mode.** Naive multi-turn OPSD collapses. More than 50% of tokens carry a *negative* teacher–student gap.
  - **Fix.** A sigmoid gate on the gap keeps endorsements and attenuates rejections.
  - **Result.** ALFWorld 84.4 vs GRPO 75.0 (Qwen2.5-3B), with no skills at inference. — [2605.15155](https://www.alphaxiv.org/abs/2605.15155) (S3, S7)

**Mechanism dissections of privileged self-distillation (missed; directly threaten SCAD-R's attribution)**
- VERIFIED: "Privileged Solutions or Context-Induced Teacher Behavior?" (MBZUAI/RIKEN, 2608.09228):
  - **Design.** OP²SD gives the teacher a worked solution to an *unrelated* problem.
  - **Result.** OP²SD matches or beats OPSD in 8 of 9 model–benchmark groups (Qwen3-8B: +9 to +11 Avg@12 over OPSD).
  - **Controls.** A single fixed example works. A locally *corrupted* solution works. A trivial "1+1" context or an off-subject (physics) context fails.

  Conclusion: much of the gain is **context-induced teacher behaviour, not target-specific information**. — [2608.09228](https://www.alphaxiv.org/abs/2608.09228) (S4, S6)
- VERIFIED: "What Does Privileged Information Add to OPSD?" (NUS, 2609.20612):
  - Reference-free OPSD explains most of the Qwen3-1.7B gain.
  - The best reference view adds only +1.30 pp, and that effect does not survive Holm correction.
  - Effects reverse with the student's rollout mode.
  - Both models degrade by step 100. — [2609.20612](https://www.alphaxiv.org/abs/2609.20612) (S4, S12)
- VERIFIED: SAPD (2610.09665), a rollout-free, off-policy self-distillation method:
  - Step-aligned privileged info beats full-solution privileged info (47.87 vs 45.00 avg) and beats misaligned/shuffled steps (45.65).
  - It runs about 2× faster than OPSD/GRPO. — [2610.09665](https://www.alphaxiv.org/abs/2610.09665) (S1, S8)

  SCAD-R's per-step notes are "step-aligned" by construction, which is a point in its favour. SAPD also shows that offline (non-rollout) distributional self-distillation can be competitive. INFERRED relevance.

**Leak theory (re-read with the exact mechanism)**
- VERIFIED: PSP attributes the multi-turn OPSD failure to "an irreducible per-token gap I(Y_k; σ | X, Y_<k) > 0" (citing Yang et al. 2026a, RLSD). Under it, the student learns "the scaffolding of privileged competence, e.g., open confidently and act decisively" without the content. PSP's remedy trains *with the instruction kept in context*: "each group is trained under the context it was sampled from". PSP also distinguishes "information-symmetric" from information-asymmetric distillation. — [2609.29051](https://www.alphaxiv.org/abs/2609.29051) (S11)
- VERIFIED: "When Context Returns" (2606.11627):
  - Re-introducing the distilled context to the distilled student often *degrades* it ("context-induced degradation"), even on examples it already solved.
  - Proposed fix: No-Context Anchoring, which reduces harm in 12 of 14 configurations. — [arXiv 2606.11627](https://arxiv.org/abs/2606.11627) (S22)

**Classic context distillation, System-2 distillation, internalization theory**
- VERIFIED: "Distilling System 2 into System 1" (Meta FAIR, 2407.06023):
  - **Method.** The same model (Llama-2-70B-chat) is SFT'd on its own System-2 outputs, with the intermediates discarded.
  - **Works for** Rephrase-and-Respond (last-letter 30.0 → 98.0), System-2 Attention and Branch-Solve-Merge.
  - **Fails for** CoT on GSM8k (7.13 distilled vs 7.58 System 1; CoT itself 52.77).
  - Self-consistency filtering is critical. — [2407.06023](https://www.alphaxiv.org/abs/2407.06023) (S13, S17)
- VERIFIED: "Learning through Internalization" (Tsilivis, …, Srebro, 2606.20937):
  - Transformers internalize modular-counter semiautomata in-distribution via curricula that remove CoT tokens, but **OOD accuracy degrades progressively** (shortcut solutions).
  - Prime-order counters are harder.
  - A **mixture** of hinted and unhinted examples can replace the curriculum. — [2606.20937](https://www.alphaxiv.org/abs/2606.20937) (S13, S16)
- SNIPPET: the canonical context-distillation lineage appears in the reference lists of PSP and SAPD, which were read:
  - Snell et al. 2022 "Learning by Distilling Context";
  - Hsieh et al. 2023;
  - Lopez-Paz et al. 2016 "Unifying distillation and privileged information" (generalized distillation / LUPI);
  - Vapnik & Vashist 2009 (LUPI).

  Askell et al. 2021 coined "context distillation" (prior knowledge, not re-fetched). — [PSP](https://www.alphaxiv.org/abs/2609.29051), [SAPD](https://www.alphaxiv.org/abs/2610.09665)
- SNIPPET: Generative Prompt Internalization ([2411.15927](https://www.alphaxiv.org/abs/2411.15927)) targets agent prompts with role-swapped data synthesis. Its novelty for SCAD-R is low because it covers fixed prompts, not state-dependent notes. (S13, S14)

**Robotics / LUPI analogues**
- VERIFIED (abstract-level via proceedings page + search summary): ADVISOR (Weihs et al., NeurIPS 2021) defines the **imitation gap**: when the teacher uses information the student lacks, pure imitation is suboptimal. ADVISOR adaptively mixes IL and RL losses per state. — [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/9fc664916bce863561527f06a96f5ff3-Abstract.html) (S21)
- SNIPPET: Informed asymmetric actor-critic ([2509.26000](https://www.alphaxiv.org/abs/2509.26000)) and Student-Informed Teacher Training ([2412.09149](https://www.alphaxiv.org/abs/2412.09149)) adapt the privileged teacher toward what the student can realize. (S4)
- SNIPPET: LEAP ([2410.05434](https://www.alphaxiv.org/abs/2410.05434)) has LLM agents learn from *privileged AI feedback* through iterative fine-tuning. Recalled, not re-verified this pass: its paper discusses the trade-off between teacher privilege and student realizability. (S9)

**Budget/progress internalization**
- SNIPPET: several papers treat budget awareness:
  - Budget tracker in context: BATS ([2511.17006](https://www.alphaxiv.org/abs/2511.17006));
  - Budget internalized via RL: [2609.00813](https://www.alphaxiv.org/abs/2609.00813);
  - Agents are mostly not budget-aware: BAGEN ([2606.00198](https://www.alphaxiv.org/abs/2606.00198));
  - Progress-awareness training: [2606.14302](https://www.alphaxiv.org/abs/2606.14302). (S10)

  SCAD-R's "step n/40" note sits inside this cluster. None of these is SWE plus distillation as far as the snippets show.

### Inferences
- INFERRED, closeness ranking to SCAD-R (3 = same idea):
  - **MNM: 2.5.** Same model, LoRA, state-triggered hints, scripted mistake detection, multi-turn, ≥70B; only the hints are human-written and the domain is not SWE.
  - **OPHSD: 2.** Same model, deterministic harness, removed at inference; single-turn.
  - **HAD: 2.** Same notes, deterministic; harness kept, different teacher.
  - **Harness-Zero and HAT: 2**, as the report already says.
  - **SDPO: 1.5** (mechanism parent).
- INFERRED: the report's novelty score of 2/3 for SCAD-R is generous. With MNM and OPHSD on the table, the method novelty is ~1/3. Note the ~1/3 holds **only** under the "harness impossible at deployment, SWE, 31B W4A16" framing, which a reviewer will read as a setting change.
- INFERRED: the cross-domain note's claim "No study distils agent harness annotations with a same-model teacher" is false at the level of "agent hints" (MNM). It survives only for "*purely state-computed* annotations on *SWE* at ≥30B".

### Gaps
- OpenReview (ICLR 2027 submissions, NeurIPS 2026) could not be searched directly. A web search for OpenReview hits returned no ICLR-2026 page on agent scaffold distillation (S14). MNM's venue status is unknown.
- The "Kim et al., 2026" harness-configuration paper cited by HAD was not located by title.
- LEAP's realizability discussion was not re-read in this pass.
- No paper was found that tests a **placebo/irrelevant-note control** in *agent* harness distillation. The OP²SD-style control exists only for math.

---

## 2. Conceptual soundness: when does same-model distillation of a deterministic harness work?

### Takeaway
Information symmetry removes PSP's *information* leak, but three failure channels remain:
- **(a) computational unrealizability.** The student must compute the note inside one forward pass over a 32k context. This is easy for exact-repeat detection and hard for counting.
- **(b) form leak.** Actions that reference the note.
- **(c) covariate shift.** Notes prune states, so the student never sees long loops or near-duplicate loops.

The redundancy puzzle ("if the note is computable, why doesn't the model already use it?") has a clear answer. The failure is **salience/policy, not information**. That makes SCAD-R a behaviour-shaping method whose effect could equally come from a generic reminder, which is exactly what a placebo arm must rule out.

### Cited Findings
- VERIFIED: PSP's gap term is I(Y_k; σ | X, Y_<k). — [2609.29051](https://www.alphaxiv.org/abs/2609.29051)
- VERIFIED: HAD finds that hiding *state tracking* drops success 63.4 → 37.3%. *Behaviour monitoring* is partly internalized: hiding it gives 56.0 on unseen tasks. — [2610.02858](https://www.alphaxiv.org/abs/2610.02858)
- VERIFIED: Harness-Zero internalizes unevenly: "inspect before edit" ~100%, pagination/read-back 40–50%. (Prior session, cited in `cross_domain_ideas.md`.) — [2609.24974](https://www.alphaxiv.org/abs/2609.24974)
- VERIFIED: counter-type state tracking internalizes in-distribution but degrades OOD; mixtures help. — [2606.20937](https://www.alphaxiv.org/abs/2606.20937)
- VERIFIED: System-2 distillation fails when the scaffold carries genuine multi-step computation (CoT on GSM8k) and succeeds when it mostly re-focuses attention (S2A, RaR). — [2407.06023](https://www.alphaxiv.org/abs/2407.06023)
- VERIFIED: the PSP leak in practice: the student reproduces the *shape* ("declares done", closing call) without the content. — [2609.29051](https://www.alphaxiv.org/abs/2609.29051)
- VERIFIED (prior pass, `critic_judgement.md`): Feedback That Backfires shows repeat probability rising 0.06 → 0.54 after failure, 83% driven by surface form, at ≤1.7B. — [2608.23651](https://www.alphaxiv.org/abs/2608.23651)
- VERIFIED: OPSD gains can come from generic context-induced behaviour. — [2608.09228](https://www.alphaxiv.org/abs/2608.09228)
- VERIFIED: multi-turn self-distillation has >50% negative-gap tokens and collapses without gating. — [2605.15155](https://www.alphaxiv.org/abs/2605.15155)

### Inferences
- **The PSP argument is mis-stated in the report (INFERRED, high confidence).** The report says "Privileged information belongs in the *sampler*, not the loss. This lesson shapes SCAD-R." PSP's sampler keeps σ **in the training context**. SCAD-R removes it, which is Snell-style off-policy context distillation, the "target" use in PSP's taxonomy, done with hard labels. The correct defence is narrower. If σ_t = f(history_t) for a deterministic f, then I(Y; σ | X, Y_<k) = 0 and the *irreducible* gap vanishes; only a *learnability* gap remains. The report should say exactly this.
- **Per-note realizability (INFERRED):**
  - *"Identical to call #k, nothing changed since".* The repeat half is an exact-match lookup that induction heads do natively. "Nothing changed" requires knowing which tools mutate state: `sed -i` and `edit_file` do, while `cat` and `grep` do not. That is learnable from tool names. → **Likely internalizable.** This is also the case where the model demonstrably *has* the information and still repeats (copy bias), so the note works as a salience/policy intervention.
  - *"Edited X, no test since".* This needs a scan back over a variable-length history for the last mutating call and the last test call. It is moderately hard, and long contexts degrade it. → **Plausibly internalizable as a policy** ("after an edit, run tests") without computing the predicate exactly.
  - *"Step n/40".* This is modular counting of turns over up to 32k tokens, the weak case in the internalization literature (OOD degradation) and in HAD (state tracking does not internalize). → **Predicted not to internalize** as information. The likely result is that the student learns an *unconditional* drift (shorter episodes, earlier submission) instead of budget-conditional behaviour. PSP's "form without content" leak reappears here as "hurry without knowing when". Notes like this should be dropped, or tested as a separate arm with a pre-registered "does behaviour condition on n?" check.
- **Redundancy puzzle (INFERRED).** If the note is re-derivable, its in-context effect measures the gap between *what the model can compute* and *what its policy acts on*. Distillation can close that gap by changing the policy (Feedback That Backfires' copy-driven repeats). The same logic predicts that a **fixed generic reminder** ("do not repeat calls whose output you already have; test after edits") could produce much of the effect, through the context-induced-behaviour channel of 2608.09228. If it does, SCAD-R collapses into system-prompt distillation, which the report already killed as low-payoff.
- **Covariate shift (INFERRED).** Notes fire *after* the first repeat, so the training data contains "one repeat → recovery". It contains no 3+-repeat states and no paraphrased/near-duplicate loops, because the note does not fire on those. The 31B "doom-looping" residual after the noloop edit (One Neuron, cited in the report) suggests the model may move into near-duplicate loops. The evaluation needs a near-duplicate metric (e.g., normalized-command equality or observation-hash equality), not exact-match NPR only. DART-style perturbation (C1) is the natural complement.
- **Form leak (INFERRED).** The annotated teacher's thoughts and actions will sometimes cite the note ("as the harness noted…"). Stripped of the note, these become hallucinated references, PSP's Figure-1 failure in miniature. Such turns must be filtered or regenerated. Count how often they occur; this is a cheap diagnostic.
- **Hard-label vs KL (INFERRED from MNM, OPCD, SDAR).** MNM found KL slightly better and used p=0.9 dropout. OPCD and SDAR favour reverse KL. With vLLM's top-k logprob cap (≈20, unverified for 0.19.1), a truncated KL on teacher-run tokens is feasible. SFT on the teacher's sampled actions is the weakest variant.

### Gaps
- No study measures internalization rates separately for *information-symmetric* note types (repeat vs. untested-edit vs. budget) in SWE. This is the open analysis question.
- Whether Gemma-4-31B W4A16 can even detect "nothing changed since" correctly from context was never measured. A zero-training probe (ask the model "is this call identical to an earlier one with no mutation since?") would answer it at CPU-level cost.

---

## 3. Does the falsifier test the claims?

### Takeaway
No. Stage A ("do notes change the next action in context?") is a **necessary precondition only**. It cannot separate information from salience from generic context, and it says nothing about internalization. Its two written versions in our own notes also disagree. It needs three extra arms and a cluster-aware analysis. Total cost stays at ≤2 GPU-h for the one-step version.

### Cited Findings
- VERIFIED (local): the two versions of the falsifier disagree:
  - The report's version: ~300 NPR/post-edit states, one-step, ~1 GPU-h; kill if the action changes in <30% of NPR states **or** "run a test" rises by <10 pp.
  - `cross_domain_ideas.md` C2 Stage A: two closed-loop 30-task runs, ≈5–6 GPU-h; kill if NPR falls by <30% relative **and** tested-after-edit rises by <10 pp.

  The two differ in design (one-step vs closed loop), cost and logic (OR vs AND). — local files
- VERIFIED: off-subject or trivial teacher contexts fail while generic same-domain contexts succeed. So a placebo must be *plausible, same-domain, non-state-specific* text, not noise. — [2608.09228](https://www.alphaxiv.org/abs/2608.09228)
- VERIFIED: in Bench2Robust (Qwen3-4B), the runtime recovery context alone gave +16.8 pp, while vanilla SFT on injected failures gave nothing. — [2608.11977](https://www.alphaxiv.org/abs/2608.11977) (cited in `cross_domain_ideas.md`)
- VERIFIED: one-step metrics can miss trajectory-level shift (2609.21187, cited in the report). — report

### Inferences
- **Missing arms for Stage A (one-step, same states, same seeds; INFERRED):**
  1. **Rules in the system prompt** (the report's E8 text) with no per-step note. This is the deployable zero-training baseline.
  2. **Placebo per-step note.** Same position and length, generic same-domain reminder ("Reminder: avoid redundant calls; verify edits"). It fires on the same states, so it carries salience but no state-specific information.
  3. **Wrong-content note** (e.g., "identical to call #k" pointing to a *non*-identical call), testing whether the model reads the content at all.
  4. Optionally, a **few-shot recovery exemplar** in the system prompt.

  Decision rule: SCAD-R is worth distilling only if the real note beats the placebo **and** the system-prompt rule by a pre-registered margin (say ≥15 pp action-change on NPR states). If the placebo ≈ the real note, the mechanism is generic salience: ship the prompt rule (E8) and drop SCAD-R.
- **Internalization test (needed before any method claim; INFERRED).** At ~150 held-out NPR and post-edit states (different repos), compare five arms:
  - (i) base;
  - (ii) base + prompt rule;
  - (iii) LoRA distilled from real notes;
  - (iv) LoRA distilled from placebo notes;
  - (v) LoRA SFT on the *same* own trajectories with notes never shown (a data-selection/RFT control).

  Token-match all arms, and add Z's DITTO/penalty arm so the "tell vs punish" contrast is measured. Without (iv) and (v), any gain is confounded with generic context and with fine-tuning on filtered own data.
- **Power (INFERRED).** About 300 NPR states drawn from ~30–100 tasks are clustered, since NPRs repeat inside a few looping tasks. Use task-level cluster bootstrap or a mixed model, and report the effective n. With heavy clustering (ICC ~0.3, unmeasured), 300 states may be worth only ~60–100 independent units, which halves the power the report implies. Pre-register the per-task cap (e.g., ≤5 states per task).
- **Note-history confound (INFERRED).** In a real annotated run, notes accumulate in the history. The one-step test injects one note into an otherwise note-free history. The two conditions differ, so report both (single note vs. full annotated prefix).
- **Re-attachment check (INFERRED from 2606.11627).** After distillation, also evaluate the student *with* notes. Degradation there is a known failure ("context-induced degradation") and matters if any downstream scaffold adds similar messages.

### Gaps
- Gemma-4-31B's base rate of noticing exact repeats is unmeasured. Z's Stage 1 (prompt-logprob ablation of the earlier identical call) can share the same states and GPU budget. Run the two together.

---

## 4. Generality beyond this competition

### Takeaway
Moderate. The *setting* (a harness you can use while generating data but cannot ship) is real and becoming common as models are deployed into third-party or locked scaffolds. But the *mechanism* is already covered by general hint- and harness-distillation methods. The generalizable artifact is the **note-type internalizability analysis**, not the recipe.

### Cited Findings
- SNIPPET: production harnesses change often enough that harness-robust training is studied in CHART ([2609.22247](https://www.alphaxiv.org/abs/2609.22247)) and HarnessBandit ([2609.13739](https://www.alphaxiv.org/abs/2609.13739)). Model–harness fit varies across settings ([2610.00917](https://www.alphaxiv.org/abs/2610.00917)). (S20)
- VERIFIED: deterministic harness lines generalize across ALFWorld/WebShop/ScienceWorld with shared templates. — [2610.02858](https://www.alphaxiv.org/abs/2610.02858)
- VERIFIED: redundant actions affect 79–98% of tasks even in Claude Code and mini-SWE-agent (2609.30725, cited in the report). — report

### Inferences
- INFERRED: the "harness impossible" framing generalizes. Model vendors serve weights into scaffolds they do not control, and competitions and benchmarks lock the scaffold. That framing is a motivation, not a contribution, because MNM, OPHSD, Harness-Zero and HAT all also deploy without (or with less) harness.
- INFERRED: the transferable result would be a table answering "which deterministic note types internalize, at what rate, and does the effect exceed a placebo, on a 31B SWE agent." Any scaffold designer can use that table, and it matches the rubric's "rigorous analysis of existing methods" credit.

### Gaps
- No evidence on whether notes distilled under our v2 scaffold transfer to a different SWE scaffold (OpenHands, mini-SWE-agent). Test this cheaply with one-step states from the Nebius OpenHands trajectories, which Z already plans to use.

---

## 5. Verdict: what was missed, does the gap survive, what must change, and a stronger variant

### Takeaway
The gap **survives only as an analysis question**: which *information-symmetric* harness notes internalize, and is the gain information or salience, for SWE at ≥30B. As a *method*, SCAD-R is now closeness ~2.5 to MNM. It should drop from "runner-up architecture" to an **arm inside Z**. The rubric score should fall from 16 to about 13–14 (Novelty 2→1; others unchanged), unless it is reframed as below.

### Cited Findings
- See sections 1–3. The key sources are:
  - MNM [2502.01562](https://www.alphaxiv.org/abs/2502.01562);
  - OPHSD [2605.08741](https://www.alphaxiv.org/abs/2605.08741);
  - HAD harness table [2610.02858](https://www.alphaxiv.org/abs/2610.02858);
  - SDPO [2601.20802](https://www.alphaxiv.org/abs/2601.20802);
  - OP²SD [2608.09228](https://www.alphaxiv.org/abs/2608.09228);
  - 2609.20612;
  - PSP [2609.29051](https://www.alphaxiv.org/abs/2609.29051);
  - Learning through Internalization [2606.20937](https://www.alphaxiv.org/abs/2606.20937);
  - System-2 distillation [2407.06023](https://www.alphaxiv.org/abs/2407.06023);
  - When Context Returns [2606.11627](https://arxiv.org/abs/2606.11627).

### Inferences
**What the proposal missed (concrete):**
1. **MNM (2502.01562).** Same-model LoRA context distillation of state-triggered hints for multi-turn agents at 70B, with scripted mistake detection, DAgger rounds, KL loss and hint dropout of 0.9. This is the closest prior art, and our notes never cite it.
2. **OPHSD (2605.08741).** Same-model distillation of a *deterministic programmatic harness*, removed at inference.
3. **HAD's harness is deterministic, with the same note templates.** The report lists HAD only as "harness kept at deployment" and misses that its notes are SCAD-R's notes.
4. **SDPO lineage (2601.20802).** Feedback-conditioned self-teacher. SCAD-R notes are synthesized feedback, so the variant-(b) KL objective is SDPO/OPCD applied to multi-turn SWE.
5. **Context-induced-teacher-behaviour dissections (2608.09228, 2609.20612).** These require a placebo arm. Without one, a SCAD-R gain cannot be attributed to the information in the notes.
6. **The PSP argument is misapplied.** SCAD-R puts privileged context in the *loss* (by stripping). Its protection is information symmetry (zero irreducible gap), not "sampler placement". Learnability is not guaranteed (Learning through Internalization; HAD state tracking).
7. **The budget/step-counter note** is the type least likely to internalize, and the most likely to produce a form-without-content leak (unconditional hurrying).
8. **Form-leak filtering** (actions citing notes), **near-duplicate loop metrics**, **re-attachment degradation** (2606.11627) and **cluster-aware power** are all absent.
9. **Inconsistent falsifier specs** across the report and cross_domain C2: OR vs AND, one-step vs closed loop.

**Does the gap survive?** Partly, INFERRED:
- **Dead:** "first to distil agent harness hints with a same-model teacher".
- **Narrow but alive:** "first to separate information from salience in deterministic, information-symmetric harness notes, by note type, for a ≥30B SWE agent where the harness cannot ship". No paper found runs a placebo-note control in agent harness distillation (S1–S22).

**What must change:**
- Rewrite the PSP justification as information symmetry (I = 0) plus a learnability caveat.
- Drop or isolate the step-counter note.
- Add the placebo-note, wrong-content-note and system-prompt-rule arms to Stage A.
- Add the placebo-distilled and no-note RFT controls to the internalization test.
- Use hint dropout (p≈0.9, MNM) and KL on top-k teacher logprobs instead of hard-label stripping.
- Filter note-citing turns.
- Report near-duplicate loops and with-note re-attachment.
- Cite MNM, OPHSD and SDPO as closeness-2–2.5 prior art.

**Stronger variant ("Tell vs. punish", folded into Z), INFERRED:**
- **Setup.** Z already labels zero-information actions from environment state. The same label can drive two interventions on the same 31B states:
  - **(T) tell:** the deterministic note in context, distilled with dropout-KL (SCAD-R);
  - **(P) punish:** Z's guarded DITTO-style penalty.

  Controls: a placebo-note distillation, the system-prompt rule, and no-note own-state SFT.
- **Outcomes**, on held-out-repo NPR states, post-change-repeat (legitimate repeat) states, and near-duplicate loops:
  - action change;
  - legitimate-repeat preservation (Z's K3 harm metric);
  - near-duplicate migration.
- **Per note type.** Repeat, untested edit, and budget (the last as a pre-registered predicted failure).
- **Why it is stronger.** Every row reuses Z's labels and states, so it costs ≈1–2 GPU-h on top of Z. It produces an analysis result in both directions: if T ≈ placebo, harness notes act through salience, and the honest recommendation is "ship the prompt". It also gives Z a natural, well-motivated baseline the rubric rewards.

### Gaps
- MNM's exact KL-vs-CE margin and its per-hint internalization rates were not extracted (the report gives only "slightly better").
- Whether Gemma-4-31B W4A16 changes behaviour at all under in-context notes is the empirical crux and is unmeasured. Feedback That Backfires is the main prior that it might not, or might backfire.

---

## Search log

| ID | Tool | Query / target | Useful hits |
|---|---|---|---|
| S1 | alphaXiv discover | distilling scaffold/harness feedback, hints, system prompt into agent weights; context distillation, privileged info, self-distillation | Harness-Zero, HAT, HAD, PSP, distillation roadmap 2609.36630, SEED, **SAPD 2610.09665**, PR-OPD, Latent OPSD, ReSAIL |
| S2 | alphaXiv discover | feedback internalization; Reflexion; natural-language feedback distillation | RLTL;DR 2609.37633, SEED, continual experience internalization 2606.04703, Inductive Feedback MPD 2609.35390, AdviSD, Notes to Self, reflection-enhanced SD 2605.12741 |
| S3 | alphaXiv discover (historical) | RL via self-distillation with env feedback; OPSD privileged context | **SDPO 2601.20802**, Self-Distilled RLVR, SR-OPSD, RLCSD, **Self-Distilled Reasoner 2601.18734**, **SDAR 2605.15155**, Skill-SD |
| S4 | alphaXiv discover (historical) | LUPI, imitation gap, asymmetric actor-critic, teacher-student | Informed asymmetric AC, **2609.20612**, **2608.09228**, **π-Distill 2602.04942**, Rethinking PI in OPSD 2608.18271, HDPO, Student-Informed Teacher |
| S5 | alphaXiv read | 2601.20802 SDPO | Full report read |
| S6 | alphaXiv read | 2608.09228 OP²SD | Full report read |
| S7 | alphaXiv read | 2605.15155 SDAR | Full report read |
| S8 | alphaXiv read | 2610.09665 SAPD | Full text read |
| S9 | alphaXiv discover (historical) | LEAP privileged AI feedback, π-Distill, realizability | LEAP 2410.05434, π-Distill, AVSD, PHF, "Distill Skills into Weights" 2608.09826 |
| S10 | alphaXiv discover | budget awareness / loop detection internalization | 2609.00813, BAGEN, HAT, 2606.14302, BATS 2511.17006 |
| S11 | alphaXiv read (full text, grep) | 2609.29051 PSP: leak mechanism, sampler vs loss, info-symmetric | Sections 1–3 and related work read |
| S12 | alphaXiv read | 2609.20612 | Full report read |
| S13 | alphaXiv discover (historical) | System 2 distillation, context distillation, implicit CoT, prompt internalization | **2407.06023**, OPCD, CODI, **Learning through Internalization 2606.20937**, **When Context Returns 2606.11627**, Generative Prompt Internalization, SKILL0 |
| S14 | WebSearch | openreview ICLR 2026 context distillation agent scaffold internalize system prompt reminders | No OpenReview hit. Pointer to "agent coaching via hints" (**MNM 2502.01562**), GCD |
| S15 | WebFetch arXiv + alphaXiv read | 2502.01562 MNM | Full report read |
| S16 | alphaXiv read | 2606.20937 | Full report read |
| S17 | alphaXiv read | 2407.06023 | Full report read |
| S18 | alphaXiv read (full text, grep) | 2610.02858 HAD: harness determinism, templates, related work | Harness table, related work; led to **OPHSD 2605.08741**, Kim et al. 2026, SAGE-OPD, UCOB |
| S19 | alphaXiv read | 2605.08741 OPHSD | Full report read |
| S20 | alphaXiv discover (recency) | harness configurations; harness-robust agents; SWE scaffold-free | CHART, HarnessBandit, Harness Learning, CoTrace, "Thinking Outside the Box" 2609.39578, Model–harness fit 2610.00917 |
| S21 | WebSearch | "Bridging the Imitation Gap by Adaptive Insubordination" | NeurIPS 2021 ADVISOR |
| S22 | WebFetch | arXiv 2606.11627 abstract | Context-induced degradation, NCA, 12/14 |

**Not done / not possible:** a direct OpenReview venue search (no search access; S14 web search returned no OpenReview page); a forward-citation crawl of Snell 2022 / Askell 2021 (approximated through the related-work sections of PSP, SAPD and HAD); a full read of LEAP, π-Distill and CHART (snippet only).
