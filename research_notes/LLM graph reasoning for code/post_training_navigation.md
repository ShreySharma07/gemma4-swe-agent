# Post-training methods for code/graph navigation and evidence-backed diagnosis (as of 8 Oct 2026)

Scope: SFT, rejection sampling (RFT), RL, process rewards and self-distillation that teach LLMs to navigate code or graphs and produce evidence-backed diagnoses. Items already covered in `docs/literature_2025_2026.md` (LocAgent, ToolTrain, CodeScout, SWE-smith, SWE-ZERO, trajectory curation for LoRA, GRAIN, GraphDancer, SkillForge) are only referenced here when new details matter (e.g. CodeScout's exact reward and compute).

Labels: **[PR]** = peer-reviewed / accepted, **[PP]** = preprint. **VERIFIED** = read in the primary source this session. **INFERRED** = my reasoning, not stated by a source. Dates are arXiv first-version dates.

---

## Q1. What rewards exist for localisation and diagnosis, and which are verifiable and cheap without executing tests?

### Takeaway
Every strong 2026 localisation-RL paper uses an **execution-free, gold-patch-derived set-overlap reward** (file/module/function F1). The newest step is to make it **dense**: turn-level "newly discovered a gold location" rewards plus a **discovery→commit** reward (A²Agent), and level-wise rank rewards (C2C). None of these rewards checks *why* the agent got there. Execution-based process signals (counterfactual forks, CRR) work, but they cost ~1.7× rollout compute, which is out of reach for us.

### Cited Findings
**Outcome rewards on the final localisation (all execution-free)**
- CodeScout [PP, Mar 2026]: reward = F1_file + F1_module + F1_function, with gold sets parsed from the gold patch. The parser extends LocAgent's to catch added members, import/global edits, and to ignore docstring-only edits. The 14B run collapsed (agent exhausted steps without submitting) until an auxiliary binary "terminated within k turns" reward was added. Rewards for parallel tool calls *hurt*. No repo install or sandbox is needed, because localisation doesn't execute code. VERIFIED — [CodeScout](https://www.alphaxiv.org/abs/2603.17829)
- CodeScout trains on 39K SWE-smith instances across 128 repos, disjoint from eval repos. Results on SWE-bench Verified: 4B → file F1 68.5 / function F1 36.8; 14B → 68.6 / 40.3. RL used 8×H100. Their RL-algorithm ablation found "reward design and scaffold likely matter more than the specific RL algorithm": GSPO, Dr.GRPO and GRPO land within a few F1 points of each other. VERIFIED — [CodeScout](https://www.alphaxiv.org/abs/2603.17829)
- CodeScout argues that fixed top-K ranked outputs are ill-suited to RL reward, because extra predictions beyond the gold count go unscored. Variable-length set prediction scored by F1 balances precision and recall. VERIFIED — [CodeScout App. G](https://www.alphaxiv.org/abs/2603.17829)

**Dense / turn-level localisation rewards**
- A²Agent [PP, Aug 2026, SKKU] has a terminal reward and a turn-level reward:
  - Terminal reward r_T = (F1_file + F1_module + F1_func) + a **Discovery–Commit (DC) reward**. DC_g = |explored ∩ gold ∩ predicted| / |explored ∩ gold|, which is the fraction of gold locations the agent *saw* that it then *committed*.
  - Turn-level reward r_t = fraction of gold files, modules and functions **newly observed** at turn t.
  - Action-level advantages come from grouping turns across rollouts that share the same last-H actions (H=2), which acts as a state proxy without forking.
  - VERIFIED — [A²Agent](https://www.alphaxiv.org/abs/2608.29831)
- A²Agent results:
  - Qwen3-8B on SWE-bench Verified: file / module / function F1 = 73.3 / 48.0 / 41.0. Plain SFT and GRPO score 66.2 and 67.5 file F1.
  - Removing both components still beats GRPO by ~3.5 pp, so "turn-level reward alone provides substantially richer learning signals".
  - The discovery–commit gap is **>40% of failures across all baselines**.
  - Downstream, with a fixed Agentless repair back-end, resolve rate is 38.1% vs 31.8% for LocAgent localisation; oracle localisation gives 45.9%.
  - VERIFIED — [A²Agent](https://www.alphaxiv.org/abs/2608.29831)
- A²Agent's "RL" step is **advantage-weighted maximum likelihood**, not a clipped PPO/GRPO objective. Per-token weights w_t = f(A_t) multiply an SFT loss. Setup: LoRA r=16, 1,410 teacher (Qwen3-30B-A3B) trajectories for SFT warm-up, 8 rollouts per issue, ≤20 tool calls, 4× RTX Pro 6000. VERIFIED — [A²Agent App. F](https://www.alphaxiv.org/abs/2608.29831)
- C2C [PP, Sep 2026, Calgary] is a hierarchical RL localiser (file → function → line) over CodeBERT embeddings. It is not an LLM.
  - Reward: position-based rank weights [1.0, 0.8, 0.6, 0.4, 0.2], with −0.1 for wrong items ranked above the gold, gated level-by-level (function reward only if the file was correct).
  - On SWE-bench Verified, intermediate vs all-correct-sparse reward gives file MRR 0.41 vs 0.34, and line MRR 0.10 vs **0.00**.
  - Teacher-forced level-by-level training beat joint training.
  - VERIFIED — [C2C](https://www.alphaxiv.org/abs/2609.38402)

**Execution-based process signals (expensive)**
- Counterfactual Rollout Replay (CRR) [**PR, accepted NeurIPS 2026** per the arXiv v3 header]:
  - Method: fork the environment at ≤4 selected decision points, sample an alternative action, roll forward, and use R(realised) − R(counterfactual) as that step's advantage.
  - Results (14B, SWE-bench Verified): 41.7% vs 36.4% for vanilla GRPO. At equal wall-clock it is 41.7 vs 36.7 for extended GRPO. It stacks with a rubric PRM (SWE-TRACE) to 45.6%.
  - Cost: 321 vs 188 H100-h; the microbatch is 1.71× slower. The gain flips negative when fork cost is ≥4× native.
  - VERIFIED — [CRR](https://www.alphaxiv.org/abs/2609.33875)
- CRR cites other SWE process rewards: SWE-TRACE (rubric process reward models, arXiv 2604.14820), SWE-Shepherd (PRM, arXiv 2604.10493) and SWE-RM (execution-free feedback model, Shum et al. 2025). VERIFIED as citations only — [CRR related work](https://www.alphaxiv.org/abs/2609.33875); [CodeMidas refs](https://www.alphaxiv.org/abs/2609.22068)

**Patch-level verifiers without execution**
- Agentic Rubrics [**PR, ACL 2026 long**] scores patches against codebase-grounded rubrics without running tests. SWE-RL [**PR, NeurIPS 2025**] uses reference-patch similarity as its reward. VERIFIED as citations — [CodeMidas related work + refs](https://www.alphaxiv.org/abs/2609.22068)

**LLM-judge rewards on diagnosis text**
- NTEP-R [PP, Sep 2026] — vision-language models, not code:
  - A teacher distils each training sample into a "Necessary Tool-Evidence Path" of (goal, tool, required evidence) steps.
  - A frozen judge scores each call for pre-call goal alignment and post-call evidence uptake. Off-path and duplicate calls are penalised.
  - Results: tool calls drop from 3.11 to 1.55 at equal accuracy (69.7 vs 69.6 overall). The **goal-alignment-only variant collapsed** (5.36 calls, search accuracy 34.6).
  - Judge vs human agreement: 97%, κ=0.94.
  - VERIFIED — [NTEP-R](https://www.alphaxiv.org/abs/2609.03493)
- TRACE (advertising root-cause diagnosis) [PP, Sep 2026] — simulated, not code:
  - Reward = gated cause-correctness × slice-Jaccard + exact-attribution bonus + format. Weights 0.65 / 0.30 / 0.05.
  - The authors deliberately **do not reward the evidence field**: "Evidence is collected for interpretability … it does not affect the reward."
  - Results (Qwen3.5-35B-A3B): SFT on 1,200 rejection-sampled teacher trajectories → RL takes FullAttr@1 0.159 → 0.637 → 0.757. RL from base reaches only 0.434, with a 0.68 parse rate.
  - VERIFIED — [TRACE](https://www.alphaxiv.org/abs/2609.10315)

### Inferences
- **Cheap and verifiable for us (no execution):**
  - file / module / function F1 vs gold-patch locations;
  - nDCG@k if ranked output is wanted (I found no 2026 localisation-RL paper using nDCG as the reward; it is used as an eval metric);
  - A²Agent's turn-level discovery reward and DC reward, computable from tool-call logs plus gold sets;
  - path validity of a claimed chain against our (direction-repaired) graph, which is a set/graph lookup.
  
  All are O(1) per rollout once gold sets and the graph exist. INFERRED.
- **LLM-judge rewards** (NTEP-style) are cheap in GPU terms if Gemma judges itself, but add a judge-hacking surface. NTEP's ablation suggests pairing intent rewards with an outcome anchor and a duplicate penalty. INFERRED.
- **Discovery-commit fits our A1/A2 finding.** A²Agent's DC gap (>40% of failures) is the localisation analogue of GraphDancer's "evidence hit vs conversion". It is directly measurable on our ATIF traces with no training, which makes it a dense diagnostic metric for the paper. INFERRED.

### Gaps
- No public number found for nDCG-as-reward vs F1-as-reward for localisation.
- The SWE-TRACE, SWE-Shepherd and SWE-RM papers were not read directly. Their reward definitions are known only through citations.

---

## Q2. Is there work rewarding the evidence chain or reasoning path in SWE/code tasks (agent PRMs, step-level verification against static analysis, call-graph-checked rationales)?

### Takeaway
Yes for **process supervision of SWE trajectories**: gold-patch-distilled prerequisite graphs (P2T), rubric PRMs, counterfactual forks, and turn-level discovery rewards. **No** for rewarding an agent's *explicit claimed causal chain* (symptom → call path → fix) checked against a call graph or static analysis. That remains open as of 8 Oct 2026, as far as the searches below can establish.

### Cited Findings
- **P2T (Patches-to-Trajectories)** [PP, May 2026, NUS/MSRA] is the closest to "evidence-chain supervision" for SWE.
  - **Reverse phase:** a proposer–critic LLM pair distils the gold patch into a latent **process graph G\*** of fact nodes (static or dynamic "unlockers", e.g. view file, grep, run probe) and milestone nodes (repro script, root-cause analysis, fix plan, edit, validation), linked by prerequisite edges. A critic prunes "patch leakage" nodes.
  - **Forward phase:** sample blinded teacher continuations and score per-step progress against G\*. A **groundedness check** extracts referenced files and symbols and verifies each was already observed in the prefix (symbolic referential integrity), plus a neural "no premature claim" check. The shortest segment above an effectiveness floor is committed.
  - VERIFIED — [P2T](https://www.alphaxiv.org/abs/2605.21996)
- P2T results:
  - 1.8k SWE-Gym instances: SWE-bench Verified Pass@1 up to **+10.8 pts** over outcome-filtered SFT, with ~15% lower inference cost.
  - Removing the groundedness check costs **−7.2 pts** (Qwen2.5-Coder-32B: 50.4 → 43.2).
  - Compute-matched comparison: 4× rejection-sampling (~272 GPU-h) gives 43.2 vs P2T's 50.4 (~226 GPU-h).
  - Disclosing G\* to a blinded solver lifts Pass@1 from 29–35% to 94–97%.
  - The authors list "train a process reward model to score per-step compliance with G\*" as future work.
  - VERIFIED — [P2T](https://www.alphaxiv.org/abs/2605.21996)
- **A²Agent** rewards *observation* of gold entities per turn and commitment of observed gold. It does not reward the reasoning linking them. VERIFIED — [A²Agent](https://www.alphaxiv.org/abs/2608.29831)
- **CRR** gives step credit from executed counterfactuals. It needs no PRM, but it needs a reliable, cheap fork. VERIFIED — [CRR](https://www.alphaxiv.org/abs/2609.33875)
- **RC-OPD** (Root-Cause-Guided On-Policy Distillation) [PP, Oct 2026, math only]:
  - An external diagnoser locates the earliest substantive error in a failed student trajectory and writes a local repair.
  - The student continues from the repair to validate it (iterative counterfactual validation).
  - A frozen same-size teacher conditioned on the diagnosis supplies token-level KL targets for the error span. The valid prefix gets a down-weighted (λ=0.1) anchor-conditioned target.
  - Setup: LoRA r=64 on Qwen3-1.7/4/8B. Gains of +5.7–9.7% mean over OPSD.
  - Key analysis: with a reference solution in context, answer accuracy jumps to ~99%, but 23–25 pp of that is "reference borrowing" without derivation. Privileged hindsight can teach conclusion-copying.
  - VERIFIED — [RC-OPD](https://www.alphaxiv.org/abs/2610.03515)
- **NTEP-R** is a per-call evidence-path reward, verified by an LLM judge against a teacher-distilled necessary path (VLM domain). VERIFIED — [NTEP-R](https://www.alphaxiv.org/abs/2609.03493)
- **Gap searches:**
  - The alphaXiv query for RL rewards checking a claimed call-graph or dependency path against static analysis returned only non-matching items: CheckerBench, Dependency-Aware Policy Optimization for terminal agents, early terminal-reward prediction, OPD with verifiable reward, NTEP-R, SCOPE-RL.
  - An earlier phrasing (root cause / evidence / reasoning chain / static analysis / reward) returned **"No papers found"**.
  - VERIFIED (absence within the searched index) — see search log Q5.

### Inferences
- P2T's **symbolic referential-integrity check** ("every file or symbol referenced in the step must already appear in observed history") is a cheap, execution-free, rule-based evidence check. It transfers directly to a localiser's finding. Extending it from "was it observed" to "is the claimed edge real in the call graph" is the untried step. INFERRED.
- **Gameability.** A chain reward with no outcome anchor is gameable, as NTEP's goal-only collapse and RC-OPD's "borrowed conclusions" show. The chain reward should be gated on, or added to, gold-hit F1. INFERRED.

### Gaps
- No paper found that verifies agent-stated call paths with a static call graph as an RL or RFT signal.
- No PRM trained specifically on localisation/diagnosis steps for SWE.

---

## Q3. Synthetic task generation from repositories for localisation at scale: which gives dense supervision on unseen repos?

### Takeaway
For **localisation-only** supervision, the dominant 2026 recipe is to **reuse SWE-smith (bug injection, 128 repos, 39K instances after filtering) with gold-patch F1 and no execution at train time** (CodeScout). Execution-validated generators (R2E-Gym, SWE-smith validation, CodeMidas, Change2Task) cost Docker/test runs but guarantee observable symptoms. A **simulator-oracle** pattern (TRACE) gives objective diagnosis labels by construction. For code, the analogue is to inject a bug into an arbitrary (private-like) repo and keep the injected location as the hidden label.

### Cited Findings
- CodeScout trains purely on SWE-smith-derived localisation targets: 39K instances across 128 repos, none overlapping the eval repos. It generalises to SWE-bench Verified, Lite and Pro. VERIFIED — [CodeScout §4.1](https://www.alphaxiv.org/abs/2603.17829)
- SWE-smith [**PR, NeurIPS 2025**] synthesises test-breaking changes. R2E-Gym [**PR, COLM 2025**] generates tests and statements from commits. VERIFIED (venue via citations) — [CodeMidas refs](https://www.alphaxiv.org/abs/2609.22068); [SWE-smith](https://www.alphaxiv.org/abs/2504.21798); [R2E-Gym](https://www.alphaxiv.org/abs/2504.07164)
- CodeMidas [PP, Sep 2026, Xiaomi]:
  - Removes implemented functionality from repos, writes behavioural specs, and builds execution-grounded tests.
  - Yield: 5,545 tasks from 3,185 codebases (23 languages) out of 22,575 candidates.
  - GRPO gains on DeepSWE (+11.7) and SWE-bench Pro (+4.1).
  - **Quality > quantity:** a filtered 3k subset beat the unfiltered 8k set on all three evals.
  - Its comparison table summarises input requirements of other pipelines (SWE-rebench V2, SWE-Flow, SWE-Hub, MindForge).
  - VERIFIED — [CodeMidas](https://www.alphaxiv.org/abs/2609.22068)
- P2T notes that 64.7–70.2% of SWE-Gym instances give *no* supervision under outcome-filtered rejection sampling, because the teacher never solves them. Privileged-patch curation recovers supervision on those. VERIFIED — [P2T](https://www.alphaxiv.org/abs/2605.21996)
- The TRACE simulator–oracle pattern: sample an intervention, inject it, and generate the observations. An oracle verifier admits only episodes whose signal is recoverable from agent-visible data and stronger than confounders. The hidden label becomes an objective reward. VERIFIED — [TRACE](https://www.alphaxiv.org/abs/2609.10315)
- Other 2026 synthesis work found but **not read** (titles/abstracts only): BugPilot (complex bug generation, Oct 2025), Change2Task (Jul 2026), FrogNano (online task synthesis, 4B, Sep 2026), Socratic-SWE (Jun 2026), Hybrid-Gym (Feb 2026), SWE-Prime (Aug 2026). From the [alphaXiv discovery results](https://www.alphaxiv.org/abs/2510.19898).

### Inferences
- **Our density lever.** Hidden test repos are private and unseen, so we must train on *many repos*. Injected-bug localisation tasks give one exact, execution-free label per task, and can be generated on (a) SWE-smith's 128 repos and (b) any extra permissively licensed Python repos at near-zero cost. That yields thousands of eval instances, so function-F1 or Acc@k has a 95% CI of ±1–2 pp, versus ±6 pp for a 15% resolve rate on 129 tasks (√(0.15·0.85/129) ≈ 0.031). INFERRED (arithmetic mine).
- **Synthetic issue text can leak.** If issue text is LLM-written from the diff, it may name the mutated function, which inflates the "name shortcut" (Schrödinger / SHERLOC 58% finding). Generate the symptom from execution traces or failing-test output (SWE-smith style), or apply namespace remapping, to keep tasks structural. INFERRED.
- **Execution at data-generation time only.** For localisation training, execution is only needed to certify the bug is observable (TRACE's solvability check). It can be done once offline on CPU, not inside RL. INFERRED.

### Gaps
- No head-to-head study found comparing bug injection vs commit mining vs mutation *specifically for localisation generalisation to unseen repos*.
- BugPilot and Change2Task details not verified.

---

## Q4. What is known about LoRA on 4-bit quantised large models for agentic tool use, and about small-to-large transfer of RL findings?

### Takeaway
LoRA is not the bottleneck: rank-1 LoRA matches full RL, and LoRA beats full SFT for tool calling. Quantised-base RL is shown to work (QeRL: 32B on one H100, matching full fine-tuning on math). **No** published result covers LoRA-RL on a 4-bit ~30B model for *multi-turn code navigation*. Small-to-large evidence is mixed. Method rankings for tool-calling SFT vs GRPO stayed stable from 0.6B to 32B, but localisation-RL recipes needed scale-specific fixes (CodeScout).

### Cited Findings
- "LoRA Without Regret" (Thinking Machines blog, 29 Sep 2025; not peer-reviewed):
  - Rank-1 LoRA matches full-parameter GRPO for RL, because policy gradients carry ~1 bit per episode.
  - Attention-only LoRA underperforms, so include MLP layers.
  - The optimal LoRA LR is ~10× the full-fine-tuning LR.
  - VERIFIED (via secondary summaries + TRL docs) — [TRL docs](https://huggingface.co/docs/trl/main/en/lora_without_regret); [BERI summary](https://www.beri.net/learning/thinking-machines-lora-without-regret)
- QeRL [PP, arXiv 2510.11696, Oct 2025, MIT Han lab / NVlabs]:
  - NVFP4 quantisation + LoRA for RL, plus Adaptive Quantization Noise. Quantisation noise raises policy entropy and aids exploration.
  - First RL of a 32B model on a single H100-80GB; >1.5× rollout speedup; beats 16-bit LoRA and QLoRA.
  - Matches full fine-tuning on GSM8K/MATH (7B).
  - VERIFIED (project page/abstract) — [QeRL](https://hanlab.mit.edu/projects/qerl-beyond-efficiency----quantization-enhanced-reinforcement-learning-for-llms); [arXiv](https://arxiv.org/html/2510.11696v1)
- "SFT or RL for Tool-Calling Agents?" [PP, Sep 2026, Dialpad] — Qwen3 0.6B–32B:
  - **SFT with LoRA is best in-distribution in 15/18 settings.** GRPO wins 29/54 cross-dataset settings, but by <1 pt on average.
  - SFT→GRPO is rarely best. Dataset choice swings accuracy by up to 55 pts.
  - **LoRA beat full SFT** at 4B (88.0 vs 80.4 on xLAM).
  - The ordering is largely consistent across scale.
  - VERIFIED — [SFT or RL](https://www.alphaxiv.org/abs/2609.17848)
- CodeScout scale-specific behaviour:
  - 1.7B scored near zero F1 zero-shot and needed **RFT warm-start**: 4K perfect-F1 trajectories from the 14B model, then RL. RL still added gains after the SFT loss reached ~0.
  - 14B needed a termination reward to avoid collapse.
  - 4B and 14B reward curves were unsaturated at the end of training.
  - VERIFIED — [CodeScout](https://www.alphaxiv.org/abs/2603.17829)
- A²Agent limitation: experiments only on 4B and 8B "owing to computational resource constraints". VERIFIED — [A²Agent](https://www.alphaxiv.org/abs/2608.29831)
- TRACE: SFT→RL on a 35B MoE beat a prompted 122B sibling (0.757 vs 0.283). The authors read this as "the binding constraint is access to an effective post-training signal … rather than model scale". VERIFIED — [TRACE](https://www.alphaxiv.org/abs/2609.10315)
- Quantisation and agentic tool use: a three-model GGUF study (8-bit to 2-bit, Polish) reports that degradation thresholds replicate across models but failure modes do not. Not read in full. — [alphaXiv 2609.32042](https://www.alphaxiv.org/abs/2609.32042)

### Inferences
- **Compute feasibility.** CodeScout used 8×H100, A²Agent 4×96 GB, CRR 321 H100-h, and P2T ~226 GPU-h of curation. All exceed ~30 L4-h/week by 1–2 orders of magnitude. Online GRPO on Gemma-31B-4bit on L4×4 is unrealistic. The feasible shapes are:
  - (i) rejection-sampled SFT;
  - (ii) **offline advantage-weighted SFT** (A²Agent's objective is advantage-weighted MLE, so rollouts can be generated with vLLM and then trained on);
  - (iii) on-policy self-distillation with short contexts;
  - (iv) small-Gemma GRPO for the paper's ablation.
  
  INFERRED.
- **Quantisation mismatch risk.** Training QLoRA on an NF4 (bitsandbytes) base but serving on the W4A16 QAT compressed-tensors base means the adapter is fit to different quantisation errors. I found no study of this specific mismatch. It should be checked empirically: same adapter, eval on both bases. INFERRED.
- **Transfer from small to large.** The Dialpad study suggests method orderings (SFT vs RL) transfer across scale for tool calling. CodeScout shows reward-shaping needs (termination reward, warm-start) are scale-dependent. So run ablations on small Gemma, but re-validate reward shaping on 31B with a short RFT round. INFERRED.

### Gaps
- No paper found on LoRA RL for multi-turn repository navigation with a ≥30B 4-bit base.
- QeRL hardware support on Ada (L4) for NVFP4 kernels is unverified.
- "GRPO Training Dynamics for Small Language Models" (2609.39321) and "Scaling SFT-RL budget allocation small→large" (2609.01573) were found but not read.

---

## Q5. Which training ideas remain untried for code graphs? (with comparison grid and exact search log)

### Takeaway
The open space is **structural process supervision for localisation**. Nobody has done any of the following:
- (a) reward a claimed call-graph evidence chain;
- (b) do on-policy self-distillation with a privileged gold-location or graph-path teacher for code navigation;
- (c) use counterfactual forks for *read-only* localisation, where forking is free;
- (d) build a code NTEP (necessary evidence path) from gold patch + call graph without an LLM judge;
- (e) train with renaming + structural reward to make agents name-invariant.

Of these, (a)+(b)+(d) can be computed execution-free and offline. That fits L4×4 + LoRA.

### Comparison grid (recipes relevant to localisation / diagnosis / navigation)

| Recipe | Training signal | Exec needed? | Base / scale | Adapter | Headline result | Status | Fit for Gemma-31B-4bit, L4×4, ~30 GPU-h/wk |
|---|---|---|---|---|---|---|---|
| CodeScout | GSPO, Σ F1(file, module, func) + turn-limit bonus; RFT warm-start for 1.7B | No | Qwen3 1.7B/4B/14B | full? (not stated in read pages) | Verified file/func F1 68.6/40.3 (14B) | PP | Reward: yes. Online RL: no (8×H100) |
| A²Agent | SFT (1.4k teacher traj.) → advantage-weighted MLE; turn discovery + DC + F1 | No | Qwen3 4B/8B | LoRA r16 | Verified F1 73.3/48.0/41.0; DC gap >40% of failures | PP | **High**: offline-compatible weighted SFT |
| C2C | Hierarchical REINFORCE, rank-position rewards, level-gated | No | CodeBERT + small nets | n/a | Intermediate ≫ sparse reward (line MRR 0.10 vs 0.00) | PP | Concept only (reward shaping) |
| P2T | Patch → prerequisite graph G\*; per-step effectiveness + groundedness (symbol-observed) check; shortest-above-floor SFT curation | Partly (dynamic unlockers, final test gate) | Qwen2.5-Coder 14B/32B student, 480B teacher | full SFT (not stated as LoRA) | +10.8 Pass@1; −7.2 without groundedness | PP | Groundedness check: high. Full pipeline: too costly (~226 GPU-h) |
| CRR | Executed counterfactual forks → step advantage | Yes (forks + tests) | 14B | ? | 41.7 vs 36.4 (GRPO) | **PR NeurIPS'26** | Low for repair; **cheap for read-only localisation** (untried) |
| NTEP-R | Judge-checked per-call goal alignment + evidence uptake + duplicate penalty | No | 8B VLM | ? | Same accuracy, half the calls | PP | Medium (needs judge); idea transfers |
| TRACE | Simulator-oracle hidden label; SFT on 1.2k RS traj. → GRPO | Simulation only | 35B MoE | ? | 0.159→0.637→0.757 | PP | Pattern transfers (bug injection = intervention) |
| RC-OPD | On-policy self-distillation; diagnosis-conditioned teacher on error span | No | Qwen3 1.7–8B (math) | LoRA r64 | +5.7–9.7% over OPSD | PP | **High** in principle (teacher = same model + privileged context); untried for code |
| SFT-vs-RL tool calling | LoRA SFT vs GRPO vs SFT→GRPO | No | Qwen3 0.6–32B | LoRA r32 | SFT best in-dist 15/18; GRPO +<1 transfer | PP | Supports SFT-first under budget |
| QeRL | NVFP4 + LoRA RL + adaptive quant noise | No | up to 32B | LoRA | 32B RL on 1 H100; ≥ QLoRA | PP | Evidence 4-bit RL works; L4 kernel support unknown |
| GRAIN (prior notes) | Structure-invariance (graph-Jaccard) reward under isomorphic renaming | No | 4B | — | OOD gap 11.1→1.1% | PP | Template for structural reward |

### Cited Findings
- **(a) Untried: call-graph-checked evidence chains.**
  - No paper returned for "RL reward that checks an agent's claimed call-graph path … against static analysis" (keywords: call graph, reasoning path, code agent, reward, verifiable). The nearest hits were NTEP-R (VLM, judge-based) and Dependency-Aware Policy Optimization for terminal agents.
  - The earlier query (root cause, evidence, reasoning chain, static analysis, reward) returned "No papers found".
  - P2T's groundedness check verifies *observation* of referenced symbols, not graph edges.
  - Sources: [NTEP-R](https://www.alphaxiv.org/abs/2609.03493); [P2T](https://www.alphaxiv.org/abs/2605.21996)
- **(b) Untried for code: privileged on-policy self-distillation (OPSD).**
  - The OPSD family is active in math: OPSD, ROSD, DASH, PW-OPSD, AVSD, RC-OPD, SR-OPSD, Latent OPSD, SEED.
  - The only SWE use of privileged gold-patch information found is P2T, and it is for *offline SFT curation*, not on-policy token-level distillation.
  - Sources: [RC-OPD](https://www.alphaxiv.org/abs/2610.03515); [SEED](https://www.alphaxiv.org/abs/2607.14777); [SR-OPSD](https://www.alphaxiv.org/abs/2608.09745); [P2T](https://www.alphaxiv.org/abs/2605.21996)
- **(c) Untried: CRR-style forks for localisation.** CRR's limitation is fork and replay cost plus nondeterminism. Localisation needs neither installs nor execution (CodeScout clones the repo, no sandbox). The environment is read-only and deterministic, so forks cost only rollout tokens. A²Agent's history grouping is a fork-free *approximation* of this. Sources: [CRR](https://www.alphaxiv.org/abs/2609.33875); [CodeScout §3.1](https://www.alphaxiv.org/abs/2603.17829); [A²Agent §2](https://www.alphaxiv.org/abs/2608.29831)
- **(e) Untried: training-side name invariance.** The renaming/obfuscation query returned evaluation and attack papers plus encoder-level invariance pretraining, but no agent-localisation *training* with renaming + structural reward:
  - Jagged Frontier;
  - Lexical Superstition (whose prior-notes summary says fine-tuning on anonymised data doesn't close the gap);
  - obfuscation steering;
  - Invariant Pretraining (encoders).
  
  Sources: [Jagged Frontier](https://www.alphaxiv.org/abs/2608.18389); [Lexical Superstition](https://www.alphaxiv.org/abs/2609.26388); [Invariant Pretraining](https://www.alphaxiv.org/abs/2608.15412)

### Inferences — candidate untried recipes, ranked for our constraints (all INFERRED)
1. **Graph-Chain RFT / advantage-weighted SFT.** This uses the Kaggle budget (A²Agent's objective is offline-compatible).
   - Gemma-31B (vLLM, 4×L4) samples k=4–8 localiser rollouts per synthetic or real task. Each rollout emits a finding plus an explicit chain (symptom node → edges → fix node).
   - Score each rollout as: R = F1_func + λ₁·DC + λ₂·(fraction of chain edges present in the direction-repaired graph) + λ₃·(chain endpoint ∈ gold) − λ₄·(cited symbols never observed). The last term is P2T's referential-integrity check.
   - Train LoRA on token weights from group-relative advantages.
2. **Graph-privileged OPSD.**
   - Teacher = the same Gemma + LoRA with privileged context (gold function(s) + shortest graph path from issue-mentioned seed to gold). Student = issue only. Token-level KL on student rollouts.
   - RC-OPD warns that privileged context breeds "borrowed conclusions". Mitigate by giving the teacher the *path* (structure), not the answer name, and by restricting distillation to the first wrong navigation step.
   - Paper ablation candidate.
3. **Code-NTEP without a judge.**
   - The necessary evidence path is derived automatically from gold patch + call graph: which gold-function bodies, callers and tests must be *observed*.
   - Each tool observation is checked by string/AST match. Duplicate visits are penalised.
   - This gives a fully verifiable process reward.
4. **Free-fork counterfactual credit for read-only localisation.** Small-Gemma paper experiment only.
5. **Bug-injection simulator-oracle eval set.** TRACE pattern applied to many repos, with symptom text from failing-test traces, not the diff.
   - Gives the dense, well-powered metrics required: function-F1, Acc@k, DC gap, chain validity.
   - Report all with bootstrap CIs over ≥1k tasks on held-out repos, and resolve rate on the 109 stable local tasks only as a secondary metric.

### Gaps
- No direct evidence that any of (1)–(5) works.
- Whether the harness's localiser→editor handoff can carry a chain is unverified (see HQ open questions).

### Search log (exact queries used this session)
| # | Tool | Query / keywords | Purpose | Outcome |
|---|---|---|---|---|
| 1 | alphaXiv discover | Q: "RL or rejection-sampling post-training for code issue localization / fault localization with file or function level rewards"; kw [issue localization, reinforcement learning, reward, SWE]; after 2025-01-01 | Q1 | C2C, A²Agent, SWE-MILE, CRR, Loc2Repair, BLAgent, Groupwise Agentic Grading, FC-SWE, EffiHolmes, HyperFL |
| 2 | alphaXiv discover | Q: "process reward models or step-level verification for SWE / coding agents trajectories"; kw [process reward model, software engineering agents, step-level] | Q2 | CRR, SWE-MILE, FLARE, TRACE (turn-level, MSR), TRCA, LatentSift, P2T, process-eval levels, Hindsight Sentinel |
| 3 | alphaXiv PDF Q&A | A²Agent 2608.29831: reward, model/training, venue | Q1/Q4 | read |
| 4 | alphaXiv PDF Q&A | P2T 2605.21996: privileged supervision, execution, LoRA | Q2 | read |
| 5 | alphaXiv PDF Q&A | C2C 2609.38402: hierarchical rewards, data | Q1 | read |
| 6 | alphaXiv discover | Q: "synthetic task generation from repositories (bug injection, commit mining, mutation) for training issue localization or SWE agents without execution"; kw [bug injection, synthetic issues, localization, SWE-smith, R2E-Gym] | Q3 | SWE-smith, R2E-Gym, WitnessGym, Socratic-SWE, CodeMidas, Change2Task, Hybrid-Gym, FrogNano, SWE-Prime, FIM mid-training |
| 7 | alphaXiv discover | Q: "LoRA or QLoRA RL fine-tuning on 4-bit quantized large models for agentic tool use; does RL with LoRA match full fine-tuning; do small-model RL findings transfer to larger models"; kw [QLoRA, reinforcement learning, quantized, agent, GRPO] | Q4 | SFT-or-RL tool-calling study, One-Layer RL, GRPO dynamics SLMs, SFT-RL budget scaling, Agentic ESOpt, QLoRA fact learning, GGUF Polish agentic study |
| 8 | WebSearch | `QeRL quantization LoRA reinforcement learning NVFP4 32B single GPU` | Q4 | QeRL project page / arXiv |
| 9 | alphaXiv discover | Q: "rewarding the reasoning path or evidence chain in code agents: rationales verified against call graph or static analysis, grounded root-cause explanations, faithfulness rewards for debugging diagnoses"; kw [root cause, evidence, reasoning chain, static analysis, reward] | Q2/Q5 gap check | **No papers found** |
| 10 | alphaXiv discover | Q: "training LLMs to produce grounded explanations of bug root causes with verifiable or judge-based rewards on the reasoning, not just final location"; kw [fault localization, explanation, LLM-as-judge, reasoning reward] | Q2 | RC-OPD, microservice RCA studies, EviRCA, BugPilot, PRM survey, LogicReward; no code-localisation evidence-reward paper |
| 11 | alphaXiv discover | Q: "code localization agent training via self-distillation, rejection sampling fine-tuning, or on-policy distillation, generalization to unseen repositories"; kw [RepoNavigator, CodeScout, localization, self-distillation, rejection sampling]; after 2025-06-01 | Q5(b) | CodeScout, SEED, Latent OPSD, FailForge, ReSAIL, Small Agents w/ Semantic Search, OPSD variants, XRepoSkill; no OPSD-for-localisation |
| 12 | alphaXiv PDF Q&A | CRR 2609.33875: process reward computation, cost, results | Q1/Q2 | read |
| 13 | alphaXiv PDF Q&A | CodeMidas 2609.22068: synthesis method, verifier, results | Q3 | read |
| 14 | alphaXiv PDF Q&A | SFT-or-RL 2609.17848: findings, LoRA, scale | Q4 | read |
| 15 | alphaXiv PDF Q&A | CodeScout 2603.17829: reward, data, models | Q1/Q3 | read |
| 16 | alphaXiv discover | Q: "reinforcement learning reward that checks an agent's claimed call-graph path or dependency chain from symptom to fix location against static analysis"; kw [call graph, reasoning path, code agent, reward, verifiable] | Q5(a) gap check | No match. Nearest: NTEP-R, Dependency-Aware PO (terminal), early terminal reward prediction, OPD w/ verifiable reward, SCOPE-RL |
| 17 | alphaXiv discover | Q: "training code agents or localization models with identifier renaming / obfuscation augmentation to make them robust to names"; kw [identifier renaming, augmentation, code localization, training] | Q5(e) gap check | Only evaluation/attack/encoder papers (Jagged Frontier, Lexical Superstition, obfuscation steering, Invariant Pretraining) |
| 18 | alphaXiv PDF Q&A | NTEP-R 2609.03493: reward definition, results | Q2 | read |
| 19 | alphaXiv PDF Q&A | TRACE 2609.10315: rewards, domain, results | Q1/Q3 | read |
| 20 | alphaXiv PDF Q&A | RC-OPD 2610.03515: mechanism, domain, results | Q2/Q5(b) | read |
| 21 | WebSearch | `"LoRA Without Regret" Thinking Machines reinforcement learning rank 1` | Q4 | TRL docs + summaries |

Not searched this session, but worth running before the thesis lock:
- "self-distillation" + "SWE-bench" (exact);
- "evidence chain" + "localization" on Semantic Scholar;
- GitHub search for `call_graph` reward in SkyRL / verl example repos;
- the Nebius SWE-rebench and SWE-Gym dataset cards (not opened).
