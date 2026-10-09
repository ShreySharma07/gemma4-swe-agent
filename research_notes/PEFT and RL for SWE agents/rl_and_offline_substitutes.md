# RL for SWE agents and RL-like gains without online RL (as of 8 Oct 2026)

Scope: online RL recipes for SWE agents, their offline / iterated-batch substitutes, test-time substitutes (verifiers / reward models), offline reward sources, and gaps. The setting is gemma-4-31b-it (W4A16), LoRA only, Kaggle 4×L4, ~30 GPU-h/week, 12 h sessions, no online RL. Rollouts are generated with vLLM, saved, then trained on offline, possibly over a few rounds.

Already covered in `research_notes/LLM graph reasoning for code/post_training_navigation.md` and `critic_judgement.md`, so only referenced here: CodeScout, A²Agent (advantage-weighted MLE), CRR, P2T, Comet-9B, SWE-RL reward, QeRL, "SFT or RL for tool calling".

Labels:
- VERIFIED: read in the primary PDF or page this session.
- SNIPPET: from a search snippet or secondary summary only.
- INFERRED: my reasoning.
- PR = peer-reviewed (venue stated in the paper header). PP = preprint. Blog = tech blog.

**Headline grid** (SWE-bench Verified Pass@1 unless stated; "Δ vs SFT/RFT" is the online-RL gain over the best non-RL starting point in the same paper):

| Method (status) | Base | Start → after | What the non-RL baseline got | Compute | Offline-feasible for us? |
|---|---|---|---|---|---|
| SWE-RL (PR, NeurIPS'25) | Llama-3.3-70B, Agentless-Mini, single-turn | SFT 36.2 → RL 41.0 (best-of-500 reranked). Oracle-repair greedy: base 5.4, SFT 29.6, RL 34.8 | SFT recovers ~81% of the oracle-repair gain (INFERRED from the numbers) | 512 H100 × ~32 h | Reward (difflib patch similarity) yes. GRPO no |
| Nebius RFT→DAPO (PP 2508.03501) | Qwen2.5-72B, SWE-agent-like, multi-turn | 11.4 → RFT 20.5 → RL 35.7 → 39.0 | RFT keeps ~33% of the total gain (INFERRED) | 16 nodes × 8 H200 | The RFT step is exactly our setting |
| DeepSWE (Blog) | Qwen3-32B, R2E-Gym/OpenHands | 23.0 → 42.2. 59.0 with hybrid TTS@16 | No SFT used. Claims RL-only beats SFT-from-proprietary approaches | 64 H100 × 6 days, 4.5k tasks | No (online). Verifier recipe yes |
| SkyRL-v0 (Blog) | OpenHands-7B / Qwen3-14B | 11.0→14.6; 18.0→21.6 (~300 training samples) | – | not read | No |
| FC-SWE (PP 2610.07898) | Qwen3.5-4B, SWE-agent | 27.9 → GRPO 38.9 → FC-SWE 41.7 | – | TP4/PP2/CP2, 16 tasks × 8 chains/step | Idea (failure-conditioned retry data) can be done offline (INFERRED) |
| CER early-reward RL (PP, under review ICLR'27) | Qwen3.5-9B | 44.5 → TMax 49.7 → CER 51.6 | – | 52.7% fewer tokens than TMax | Rubric reward from offline rollouts: yes |
| Agent-RLVR (PP 2506.11425) | Qwen2.5-72B, Agentless-style | 9.4 → RFT-SFT 20.8 → **offline DPO 22.4** → +RM rerank @32: 27.8 | Offline DPO +1.6 over RFT | 4×8 H100, 10 h (72B) | **Yes**: one offline DPO iteration |
| SWE-Gym (PR, ICML'25) | Qwen2.5-Coder-32B, OpenHands | SFT(491 teacher traj) 7.0 → 20.6. Verifier Best@16 32.0 | On-policy RFT **hurt** (Lite 15.3 → 8.7). MoatlessTools iterative RFT: 7B 7.0→9.0→10.0, 32B (LoRA) 19.0→19.7 | 2–8 H100 | Yes. Negative evidence for naive self-RFT |
| R2E-Gym (PP 2504.07164) | Qwen2.5-Coder-32B | SFT(3,321 Sonnet traj) 34.4 → hybrid verifier Best@26 51.0 | EB-only / EF-only verifiers plateau ~43 | – | Verifier: yes. k=26 at test: no |
| SWE-RM (PP 2512.21919) | 30B-A3B RM | TTS: Qwen3-Coder-Flash 51.6 → 62.0. As RL reward: +3 pp over exec-only | – | 4 H100 nodes ~20 h, ~100k trajectories | RM too big to train. Lessons yes |
| SWE-Prime (PP 2608.27449) | Qwen3-Coder-30B-A3B / GLM-4.7-Flash / Qwen3-30B-A3B | Full resolved-traj SFT 51.0 / 41.4 / 36.8 → **10% filtered + segment-masked SFT 53.2 / 51.4 / 39.6** | Pure filtered BC | full SFT, 131k ctx | **Yes**: filtering + loss masking only |
| Agentic-DPO (PP 2607.10601, non-SWE) | Qwen3.5-2B/4B/9B | τ-bench 9B: SFT 21.7 → 41.4 vs online GRPO 40.0 | Matches GRPO | 1.6× SFT cost/step vs GRPO 13.6× | **Yes** (step-level, offline). Untested on SWE |
| Kimi-Dev (PP 2509.23045) | Qwen2.5-72B + 150B-token mid-train | Agentless RL 60.4. SWE-agent SFT (5k traj) 48.6 | RL prior only "slightly" better than SFT prior after e2e RL | huge | No |

---

## Q1. Online RL for SWE agents: what does it gain over SFT, and at what compute?

### Takeaway
Online RL gives large gains over the *starting* policy: +11 to +28 pp Pass@1 on SWE-bench Verified at 32–72B. Over a strong **SFT/RFT** starting point the gain is smaller: about +4.8 pp (SWE-RL), +1.6 pp offline-DPO-vs-RFT (Agent-RLVR), and +18.5 pp RFT→RL (Nebius), where the RFT start was weak. Every published online-SWE-RL run uses ≥64 H100/H200-class GPUs for days. That is 2–3 orders of magnitude above ~30 L4-h/week.

### Cited Findings
**SWE-RL** (PR, NeurIPS 2025)
- Single-turn Agentless-Mini repair, GRPO, reward = difflib similarity to the oracle patch (−1 for bad format). Llama3-SWE-RL-70B reaches 41.0% vs 36.2% for an SFT baseline (Llama3-SWE-SFT-70B, synthetic Magicoder-style CoT data). Both use best-of-500 reranking with 30 reproduction tests. VERIFIED — [SWE-RL](https://arxiv.org/abs/2502.18449)
- Oracle-file repair with greedy decoding: base 5.4 (12.2% correct format), SFT 29.6 (96.2% format), RL 34.8 (95.6% format). Continuous reward beat discrete exact-match (34.8 vs 29.0). VERIFIED — [SWE-RL §3.3, §3.6](https://arxiv.org/abs/2502.18449)
- Compute: 1,600 steps, batch 32 problems × 16 rollouts, 16k context, 512 H100 for ~32 h. RL improved out-of-domain benchmarks (e.g. MATH 63.2→73.7 strict), while SFT degraded them on average. VERIFIED — [SWE-RL §3.1, Table 3](https://arxiv.org/abs/2502.18449)
- Pass@1 from Agentless-Mini needs ~160–500 repair samples: 33.6 at 20 samples → 40.0 at 160 → 41.0 at 500. VERIFIED — [SWE-RL Fig. 4](https://arxiv.org/abs/2502.18449)

**Nebius multi-turn RL** (PP 2508.03501)
- Qwen2.5-72B-Instruct goes 11.4 → **RFT 20.5** → DAPO Stage 1 (65k) 35.7 → Stage 2 (131k) 39.0 Pass@1. Pass@10 goes 31.0 → 43.0 → 54.6 → 58.4. VERIFIED — [Golubev et al. Table 1](https://arxiv.org/abs/2508.03501)
- The RFT step:
  - 10 rollouts per task on 7,249 filtered SWE-rebench tasks give 6,548 successful trajectories.
  - One SFT epoch (lr 5e-6, 50 steps), with **assistant turns that triggered formatting errors masked out of the loss**.
  - Formatting errors were the dominant base-model failure.
  - VERIFIED — [§4.2, App. C](https://arxiv.org/abs/2508.03501)
- The RL step:
  - G=10 per task, synchronous on-policy, >100 iterations.
  - Reward = binary test pass + linear step-count penalty beyond a threshold.
  - Infrastructure: 16 H200 nodes (128 GPUs), full-parameter.
  - VERIFIED — [§4.3, App. D](https://arxiv.org/abs/2508.03501)
- Two findings that also apply offline:
  - (i) Dropping over-long trajectories (often loops) removes the negative examples of looping, so looping *increases*.
  - (ii) Rollout sampling with top-k/min-p turned on (a vLLM upgrade) broke importance-ratio validity and destabilised training.
  - VERIFIED — [§5.2](https://arxiv.org/abs/2508.03501)

**DeepSWE** (Blog)
- Qwen3-32B, GRPO++ (clip-high, no KL, no std-normalisation, leave-one-out, "compact filtering" of max-context/step/timeout trajectories), 4.5k R2E-Gym tasks, 64 H100 × 6 days.
- 42.2% Pass@1 (avg of 16), Pass@16 71.0%, 59% with hybrid TTS at K=16 (K=8 captures most of it).
- VERIFIED (blog page) — [DeepSWE blog](https://www.together.ai/blog/deepswe)
- Nebius lists DeepSWE's pre-RL base as 23.0%. VERIFIED (as citation) — [Golubev Table 2](https://arxiv.org/abs/2508.03501)

**SkyRL-v0** (Blog): with ~300 training samples, OpenHands-7B 11.0→14.6, Qwen3-8B 3.6→9.4, Qwen3-14B 18.0→21.6. A later SA-SWE-32B reached 39.4% (SkyRL-Agent, arXiv 2511.16108). SNIPPET — [SkyRL project page](https://sky.cs.berkeley.edu/project/skyrl/); [SkyRL-Agent](https://arxiv.org/abs/2511.16108v1)

**FC-SWE** (PP, NVIDIA, Oct 2026)
- Qwen3.5-4B Resolved@1: base 27.9 → GRPO 38.9 → FC-SWE 41.7. FC-SWE trains on recovery trajectories conditioned on the failed patch plus verifier feedback, with trajectory-local rewards.
- Removing the −5 token-advantage penalty on invalid tool calls / malformed thinking made online success collapse from 76.6% to 2.3% within 3 steps.
- Setup: 4,518 R2E-Gym tasks, 16 tasks × 8 chains per step.
- VERIFIED — [FC-SWE](https://arxiv.org/abs/2610.07898)

**Kimi-Dev** (PP)
- Agentless RL (Kimi k1.5 REINFORCE, outcome-only, drop pass@16=0 prompts, curriculum re-introduction, replay of recent positives) on a 150B-token mid-trained Qwen2.5-72B: 60.4% (workflow).
- SWE-agent SFT on 5,016 SWE-smith trajectories: 48.6% Pass@1.
- After end-to-end SWE-agent RL, the "RL prior" only *slightly* outperforms the "SFT prior". The mid-train-only prior collapsed to <2% within 10 RL steps because too few tasks were solvable (260/6,202).
- VERIFIED — [Kimi-Dev](https://arxiv.org/abs/2509.23045)

**Reference points** (all VERIFIED as citations in [Golubev Table 2](https://arxiv.org/abs/2508.03501), "Before → After"; the first four used teacher distillation, i.e. SFT)

| System | Before → After | Training |
|---|---|---|
| SWE-agent-LM-32B | 14.3 → 40.2 | SFT |
| Skywork-SWE-32B | 6.4 → 38.0 | SFT |
| R2E-Gym-32B | 7.0 → 34.4 | SFT |
| SWE-Gym-32B | 7.0 → 20.6 | SFT |
| DeepSWE | 23.0 → 42.2 | RL |
| SkyRL-14B | 18.0 → 21.6 | RL |

**Credit-assignment add-ons** (all online; covered in sibling notes): CRR 41.7 vs GRPO 36.4 (14B, PR NeurIPS'26), SWE-MILE (milestone potentials, PP 2609.32631), A²Agent turn-level rewards. VERIFIED (titles/abstracts) — [CRR](https://www.alphaxiv.org/abs/2609.33875); [SWE-MILE](https://www.alphaxiv.org/abs/2609.32631)

### Inferences
- At 32–72B, **distilled SFT (SWE-smith, Skywork, R2E-Gym: 34–40%) and online RL from scratch (DeepSWE 42.2, Nebius 39.0) land in the same band**. RL's distinctive edge over SFT shows up as (a) +2–5 pp on top of a good SFT start, (b) better Pass@k/TTS headroom, and (c) out-of-domain retention (SWE-RL). INFERRED
- Nebius is the cleanest "how much does offline RFT keep" datapoint for multi-turn SWE. One RFT round from self-rollouts recovered ~33% of the total gain ((20.5−11.4)/(39.0−11.4)), and most of it was format repair. Iterating RFT is what's untested at this scale (Q2). INFERRED
- Every published online run used ≥64 H100-class GPUs for days. Online RL is ruled out for us, consistent with the sibling note. INFERRED

### Gaps
- No paper reports online RL vs a compute-matched *iterated* RFT for multi-turn SWE at ≥30B.
- DeepSWE's exact H100-hours and its SFT ablation numbers were not read (blog only). SkyRL-Agent paper not read.
- Qwen3-Coder / DeepSeek-V3.x / GLM agentic-RL contributions are not separable from their pretraining. No ablation found.

---

## Q2. Offline / iterated-batch substitutes: what has been shown on multi-turn SWE agents, and how much of the online gain do they recover?

### Takeaway
On SWE agents specifically, the evidence for offline substitutes is thin.

What works, ordered by evidence strength:
- **Single-round RFT on self-rollouts with error-turn masking** (Nebius, +9.1 pp from 11.4).
- **Quality-filtered / segment-masked BC** (SWE-Prime: 10% of trajectories beats 100%, +2 to +10 pp over plain resolved-trajectory SFT).
- **Offline trajectory DPO after RFT** (Agent-RLVR: +1.6 pp over RFT at 72B).
- **Advantage-weighted MLE** for localisation (A²Agent, sibling note).

What fails: **naive on-policy self-RFT can hurt** in an OpenHands-style scaffold (SWE-Gym: 15.3 → 8.7 on Lite). Iterated RFT saturates after ~2 rounds (SWE-Gym MoatlessTools).

The strongest "matches online RL" result is **Agentic-DPO** (step-level, expert-anchored, 1.6× SFT cost), but it was shown on τ-bench / ToolBench / Mind2Web, not SWE.

### Cited Findings
**Rejection-sampling FT / filtered BC / expert iteration**
- Nebius RFT: one round on self-rollouts, 11.4 → 20.5. Masking error-triggering assistant turns from the loss was a deliberate design choice. VERIFIED — [Golubev §4.2](https://arxiv.org/abs/2508.03501)
- SWE-Gym self-improvement:
  - On-policy self-RFT failed on OpenHands: 868 self-sampled successes + 491 teacher trajectories took Lite from 15.3% to 8.7%.
  - On MoatlessTools, iterative RFT with 30 rollouts per task at T=1.0 gave 7B 7.0→9.0→10.0 and 32B (LoRA) 19.0→19.7→19.7. "After two iterations, further improvements are negligible."
  - Per-instance capping at 2 successes per task beat no cap (9.7 vs 9.3, 7B Lite), countering the easy-task bias.
  - VERIFIED — [SWE-Gym §4.2–4.3, Table 4, Table 6](https://arxiv.org/abs/2412.21139) (PR, ICML 2025)
- SWE-Gym SFT scaling: performance rose roughly log-linearly with teacher trajectories up to 491 with no saturation at 32B. VERIFIED — [SWE-Gym Fig. 1/5](https://arxiv.org/abs/2412.21139)
- R2E-Gym SFT: 32B improvement continued to 3.2k trajectories, while 14B saturated around 800. Training on thoughts + actions beat actions-only (34.4 vs 30.4). VERIFIED — [R2E-Gym §3.1, Fig. 2–3](https://arxiv.org/abs/2504.07164)
- SWE-Prime (filtered BC), selecting from 32,161 resolved Qwen3-Coder-480B trajectories:
  - Stage 1 scores trajectories on observe-edit-verify pattern, tool-call success percentile, adjacent-duplicate-call redundancy, git-history leakage, and patch scope vs gold (files and lines). It then picks the top 10% per HDBSCAN issue cluster.
  - Stage 2 has an LLM score semantic segments using the gold patch. Low-score segments stay in context but are masked from the loss (threshold 7/10).
  - VERIFIED — [SWE-Prime](https://arxiv.org/abs/2608.27449)
- SWE-Prime results (Verified):

  | Model | Full resolved SFT | Random 10% | SWE-Prime | Stage-1 only |
  |---|---|---|---|---|
  | Qwen3-Coder-30B-A3B | 51.0 | 41.2 | **53.2** | 50.2 |
  | GLM-4.7-Flash | 41.4 | – | **51.4** | – |

  Turns drop by 4–13. VERIFIED — [SWE-Prime Table 1](https://arxiv.org/abs/2608.27449)
- P2T: compute-matched, 4× rejection sampling (~272 GPU-h) gave 43.2 vs P2T privileged-curation 50.4 (~226 GPU-h). Also, 64.7–70.2% of SWE-Gym instances give *no* supervision under outcome-filtered RFT. VERIFIED (sibling note) — [P2T](https://www.alphaxiv.org/abs/2605.21996)

**DPO / preference optimisation on trajectories**
- Agent-RLVR (Scale AI, PP):
  - Method: one *offline* iteration on 817 environments. 16 rollouts per problem plus teacher-guided retries (Claude-3.7 plans / env-feedback / file-location hints, train time only). Pairs (passing trajectory, failing non-guided trajectory) are trained with trajectory-level DPO (lr 1e-6, 1 epoch) after an SFT phase on 20% of positives. 8,186 trajectories, 4,093 positive.
  - Results (Qwen2.5-72B): base 9.4 → RFT-SFT 20.8 → DPO 22.4. Guidance was critical: no-guidance DPO 19.8.
  - A Qwen2.5-Coder-32B pairwise RM trained on the same data, reranking 32 patches: 27.8. The authors hypothesise online RL would do better.
  - VERIFIED — [Agent-RLVR](https://arxiv.org/abs/2506.11425)
- Agentic-DPO (PP, JHU, non-SWE):
  - Method: at each *expert* state, sample K=4 one-step actions from the current student, drop parse failures and expert-equivalent actions, and take the highest-logprob wrong action as the negative. Loss = action-level DPO with a length-scaled β plus an SFT anchor (λ=0.5). R=5 negative-refresh rounds. No environment execution.
  - Policy-Preserving Augmentation renders the same decision under multiple tool schemas or with injected recovered-error histories.
  - VERIFIED — [Agentic-DPO §3](https://arxiv.org/abs/2607.10601)
- Agentic-DPO results:
  - τ-bench retail (Qwen3.5-9B): SFT 21.7, ETO (trajectory-DPO) 20.9±11.8, GRPO 40.0, Agentic-DPO 41.4. Agentic-DPO beats GRPO on StableToolBench at all scales.
  - Cost per step: Agentic-DPO 1.6× SFT, GRPO 13.6×.
  - Ablations: removing the SFT warm-up → 0% (parser-rejection floor); removing the SFT anchor −14 to −18 pts.
  - At 9B the best result came after **one** refresh round; smaller models needed more.
  - Stated limitation: it cannot discover states outside expert trajectories.
  - VERIFIED — [Agentic-DPO Tables 1–2, §4.6, §5](https://arxiv.org/abs/2607.10601)
- Agentic-DPO's related work lists the step-level/trajectory-level agent preference family: ETO (success/failure trajectory contrast), Step-DPO, DMPO, HPL, PivotRL (local on-policy rollouts at pivot states), and SWE-Lego / ATLaS (critical-step selection). VERIFIED as citations — [Agentic-DPO §2](https://arxiv.org/abs/2607.10601)

**Advantage-weighted / reward-weighted regression**
- A²Agent's "RL" is advantage-weighted MLE with LoRA r16, giving localisation F1 gains over GRPO. Its weights can be computed from saved rollouts. VERIFIED (sibling note) — [A²Agent](https://www.alphaxiv.org/abs/2608.29831)

**Offline value-based RL** (ArCHer / Digi-Q / ILQL style)
- No SWE-agent paper found. Searches returned only critics used at inference (Opera verbal critic, OpenHands Critic TD-trained) and token-credit theory (PACT). SNIPPET — [Opera](https://www.alphaxiv.org/abs/2609.33987); [CER related work on OpenHands Critic](https://arxiv.org/abs/2609.31995)

### Inferences
**Approximate gain recovered vs online RL** (INFERRED, cross-paper, noisy):

| Substitute | Recovers | Basis |
|---|---|---|
| One-round self-RFT | ~1/3 of total RFT+RL gain | Nebius 72B |
| RFT → offline trajectory DPO | maybe +1–2 pp over RFT | Agent-RLVR. No online arm to compare |
| Step-level expert-anchored DPO | ≈100% on tool-use benchmarks | Agentic-DPO; untested on SWE |
| Quality-filtered / segment-masked BC | +2 to +10 pp over plain resolved-traj SFT | SWE-Prime. Comparable in size to RL-over-SFT deltas (SWE-RL +4.8) |

**Recommended stack under our constraints** (INFERRED):
1. Rollouts with vLLM on our own tasks.
2. RFT with per-instance cap ≤2, error-turn masking, and SWE-Prime-style process filters (loop / duplicate-call / format-failure turns masked rather than trajectory dropped).
3. Optionally Agentic-DPO-style **step-level negatives** at the exact failure points we already measure: garbled `edit_file` args (61% in v5) and exact-repeat loops (28–48%). These are "plausible wrong one-step actions" that are cheap to sample offline (one step, no environment).
4. Advantage-weight with A²Agent/CodeScout localisation F1 if localisation is the target.

- Step 3 matches Agentic-DPO's operating regime (format/action-choice errors) better than "hard reasoning" errors. INFERRED
- The SWE-Gym self-RFT regression is a warning. Self-generated successes on easy tasks plus long, loopy trajectories can make OpenHands-style agents worse. Cap per instance, filter by process quality, and keep teacher/expert data in the mix. INFERRED
- Expect 1–2 useful rounds of iterated offline training, not many: SWE-Gym saturated after 2, and Agentic-DPO at 9B peaked after 1 refresh. INFERRED

### Gaps
- No direct online-RL vs offline-DPO/AWR comparison on multi-turn SWE at matched compute.
- No step-level DPO (Agentic-DPO / ETO / IPR-style) result on SWE-bench-style agents found.
- No offline Q-learning (ArCHer, Digi-Q, ILQL) on SWE agents found.
- KTO / ORPO / SimPO on SWE trajectories: not found.
- ETO / IPR / Agent Q originals were not re-read this session (known from training data only; not cited as findings).

---

## Q3. Test-time alternatives that substitute for RL (best-of-n + verifiers / reward models) under ~5 min/task and a 12 h total cap

### Takeaway
Learned verifiers give gains comparable to or larger than RL, but they need k≥8–16 full rollouts:

| Verifier | Gain | k |
|---|---|---|
| SWE-Gym ORM | +11.4 pp | k=16 |
| R2E-Gym hybrid | +16.6 pp | k=26 |
| SWE-RM | +10.4 pp | – |
| Agent-RLVR RM | +5.4 pp | k=32 |
| DeepSWE hybrid | +16.8 pp | k=16 |

At ~5–6 min/task sequential with a ~34 tok/s single stream, our scorer realistically allows k≈1–2. So the cheap versions matter:
- **early-prefix pruning / reranking** (CER matches the best baseline with 15–74% of tokens);
- a **LoRA YES/NO verifier adapter** on the same base (multi-LoRA allows it);
- verifier quality on a tiny k=2 choice.

### Cited Findings
- **SWE-Gym ORM**:
  - Qwen2.5-Coder-32B fine-tuned to emit YES/NO on full trajectories, 2,636 balanced trajectories (on- plus off-policy).
  - Best@k: 20.6 → 24.8 @2 → 29.8 @8 → 32.0 @16; Pass@16 42.8.
  - **LoRA verifier beat full-FT verifier** (29.8 vs 27.2 @8). Mixed on/off-policy data was best; off-policy-only plateaued around 22.
  - VERIFIED — [SWE-Gym §5.1](https://arxiv.org/abs/2412.21139)
- **R2E-Gym**:
  - Execution-based (generated tests + regression) and execution-free (14B LoRA r64 YES/NO, 5,700 trajectories) verifiers each saturate around 42.8–43.7%. Hybrid reaches 51.0 at k=26 (49.4 at k=16).
  - Removing the trajectory from the EF verifier input (patch only) drops it from 42.8 to 37.6. Agent thoughts bias the verifier.
  - For most problems <20% of generated tests discriminate. Toxic tests occur.
  - VERIFIED — [R2E-Gym §4](https://arxiv.org/abs/2504.07164)
- **SWE-RM** (Qwen3-30B-A3B RM, ~100k trajectories from ~400k rollouts):
  - TTS Qwen3-Coder-Flash 51.6→62.0, Max 67.0→74.6.
  - Two verifiers with the same TTS score can behave very differently in RL. AUC and calibration (ECE) also matter.
  - RMs trained on **<5k samples can *decline* as k grows** (OOD trajectories get spuriously high scores). A 500-sample RM had ECE 0.481 vs 0.067 at 100k. A 2:1 positive:negative ratio was best.
  - As RL reward, hybrid (RM + tests) beat tests-only by ~3 pp. A poorly calibrated RM badly hurt RL.
  - VERIFIED — [SWE-RM](https://arxiv.org/abs/2512.21919)
- Conflict on verifier data size: SWE-Gym got +11.4 pp from a 2,636-trajectory LoRA verifier, while SWE-RM reports <5k-sample RMs degrade with k. Likely due to different k ranges (≤16 vs ≤32), policies, and evaluation; not resolved. VERIFIED (both) — [SWE-Gym](https://arxiv.org/abs/2412.21139); [SWE-RM §4.2](https://arxiv.org/abs/2512.21919)
- **Agent-RLVR RM**: Qwen2.5-Coder-32B pairwise RM on patch + problem statement only, 500 steps; reranking 32 patches lifts 22.4 → 27.8. VERIFIED — [Agent-RLVR §3.1–3.2](https://arxiv.org/abs/2506.11425)
- **CER** (PP, under review ICLR 2027): task-adaptive rubrics judged on *unfinished prefixes* (cutoff at 20–80 steps), with an experience bank distilled from verified historical rollouts.
  - Nemotron-3-Ultra RM@8: 67.6 vs best baseline 63.4 (SWE-RM 61.6; OpenHands Critic ~58–60). CER@20 beats every full-trajectory baseline with 84.7% fewer tokens.
  - Qwen3.6-27B: 71.4 vs 69.4.
  - Oracle Best-of-8 is 76.2 / 78.6, so all verifiers leave a large gap.
  - VERIFIED — [CER Table 1](https://arxiv.org/abs/2609.31995)
- CER caveats:
  - The judge is length-biased: mean score rises from 0.43 to 0.85 over 10→100 steps with no change in labels, so it is unusable as a cross-step PRM.
  - Gold-file coverage predicts success with AUROC ≤0.608, so localisation alone is a weak outcome proxy.
  - Prefixes before editing/validation (≤20 steps) carry little signal.
  - Experience transfer is asymmetric: a weaker model's experience hurt a stronger model by 5–7 pp.
  - VERIFIED — [CER App. F.5–F.6, §6.1](https://arxiv.org/abs/2609.31995)
- **FC-SWE**: at matched two attempts, verifier-feedback-conditioned retry was *worse* than independent resampling for Base and GRPO policies, and better only after FC-SWE training. Note that its feedback came from the benchmark tests (not available at test time for us). VERIFIED — [FC-SWE §4.2, App. A](https://arxiv.org/abs/2610.07898)
- **SWE-RL**: test-based reranking needs many samples (20→160 samples: 33.6→40.0). VERIFIED — [SWE-RL Fig. 4](https://arxiv.org/abs/2502.18449)
- **SWE-TRACE** (rubric PRMs + heuristic TTS, PP 2604.14820) and **SWE-Shepherd** (PRM, 2604.10493): known via citations only. SNIPPET — [SWE-TRACE listing](https://www.alphaxiv.org/abs/2604.14820)

### Inferences
- Budget arithmetic (INFERRED; uses project facts from CLAUDE.md/HQ):
  - 12 h / ~120 tasks ≈ 6 min/task. At ~34 tok/s single stream that is ≈12k generated tokens per task *in total*, before tool time. k=16 full rollouts per task is impossible.
  - Feasible TTS shapes:
    - (a) k=2 via an ADK `ParallelAgent` of two short fixers, plus a judge sub-agent that selects via a **YES/NO verifier LoRA adapter** on the same base (multi-LoRA ≤8 adapters). Parallel requests may batch in vLLM at near single-stream latency.
    - (b) CER-style early pruning: abort or restart a trajectory whose prefix scores low by step ~30.
    - (c) Use the verifier offline only, as a data filter and reward for RFT/AWR (SWE-RM's "RL reward" use), which costs nothing at test time.
- Option (c) is the safest: training-time-only verifiers sidestep the scorer budget and the "skills untrusted" constraint. INFERRED
- The Pass@k/Best@k gap is large everywhere (e.g. SWE-Gym 42.8 vs 32.0 at k=16). Verifier quality at k=2 is plausibly the binding factor. A verifier trained on <3k of our own trajectories is at SWE-RM's "may not generalise" scale, so calibrate it on held-out tasks before trusting it. INFERRED

### Gaps
- No published SWE TTS result at k=2 with a per-task wall-clock cap of ~5 min.
- No result on a LoRA verifier adapter sharing the policy's 4-bit base.
- SWE-Shepherd / SWE-TRACE PRM numbers not read.

---

## Q4. Reward sources available offline (tests pass, localisation F1, tool-format validity, loop penalties)

### Takeaway
Rewards with SWE evidence, ranked:
1. **Test pass** (gold standard; needs env setup, available offline on CPU).
2. **Tool-format validity** as masking / negative advantage: essential, its removal collapses RL.
3. **Loop / length penalties** (Nebius step penalty; loops must be kept as negatives, not filtered).
4. **Patch similarity to gold** (SWE-RL difflib, continuous > discrete).
5. **Localisation F1** (CodeScout / A²Agent). It is a weak *outcome* predictor (AUROC ≤0.61), so use it as a shaping or auxiliary signal, not a proxy for resolution.
6. **Process-quality heuristics** (SWE-Prime): observe-edit-verify, tool-success rate, duplicate calls, git leakage, patch scope.
7. **Learned / rubric RMs** (SWE-RM, CER): only with calibration checks.

### Cited Findings
- Tests: binary fail-to-pass reward in Nebius / DeepSWE / FC-SWE / Kimi-Dev (outcome-only). VERIFIED — [Golubev §4.3](https://arxiv.org/abs/2508.03501); [Kimi-Dev §3.3](https://arxiv.org/abs/2509.23045)
- Format validity:
  - Nebius masks error-triggering turns in RFT.
  - FC-SWE gives a −5 advantage to invalid tool calls / malformed thinking. Without it, success collapsed from 76.6% to 2.3%.
  - SWE-RL uses −1 for bad format.
  - VERIFIED — [Golubev](https://arxiv.org/abs/2508.03501); [FC-SWE §3.5, §4.3](https://arxiv.org/abs/2610.07898); [SWE-RL Eq. 1](https://arxiv.org/abs/2502.18449)
- Loops / length:
  - Nebius applies a linear penalty beyond L_thr=10 steps (T_max 40/80). Filtering long (looping) trajectories increased looping.
  - SWE-Gym SFT reduced stuck-in-loop by 4.6–18.6 pp.
  - SWE-Prime's redundancy flag = any adjacent identical call.
  - VERIFIED — [Golubev §4.3, §5.2](https://arxiv.org/abs/2508.03501); [SWE-Gym Table 3](https://arxiv.org/abs/2412.21139); [SWE-Prime](https://arxiv.org/abs/2608.27449)
- Patch similarity: difflib continuous reward beat exact-match (oracle repair 34.8 vs 29.0). VERIFIED — [SWE-RL §3.6](https://arxiv.org/abs/2502.18449)
- Localisation F1: Σ F1(file, module, function) (CodeScout). Turn-level discovery and discovery→commit rewards (A²Agent). VERIFIED (sibling note) — [CodeScout](https://www.alphaxiv.org/abs/2603.17829)
- Caveat: the gold-target-file-touched fraction predicts terminal success only at AUROC ≤0.608 (peak at step 25). VERIFIED — [CER App. F.6](https://arxiv.org/abs/2609.31995)
- Learned RM as reward: requires AUC + calibration. A poorly calibrated RM degraded RL across 4 benchmarks (e.g. SWE-bench-Live 12.0 vs 22.4 hybrid). VERIFIED — [SWE-RM Table 5](https://arxiv.org/abs/2512.21919)
- Rubric early reward vs verifier: only 60.9% pairwise agreement, yet it produced comparable or better RL results (51.6 vs 49.7). VERIFIED — [CER §6.2](https://arxiv.org/abs/2609.31995)

### Inferences
- For our failure modes (61% garbled `edit_file` args, 28–48% exact-repeat calls, budget overruns, wrong-fix-never-tested), the cheapest high-value offline signals are all *programmatic* from traces: parse-validity per call, exact-repeat detection, "ran a test after the last edit", and "submitted within budget". These are deterministic and need no RM. INFERRED
- They map naturally onto Agentic-DPO negatives, SWE-Prime masks, and Nebius/FC-SWE penalties. INFERRED

### Gaps
- No study compares these reward sources head-to-head under offline AWR/DPO for SWE.
- No evidence on combining localisation F1 with test pass as an AWR weight at 30B.

---

## Q5. Gaps: which offline methods are untested for SWE agents (especially ~30B, LoRA)?

### Takeaway
As of 8 Oct 2026 the following were not found for multi-turn SWE agents:
- (1) step-level / action-level DPO (Agentic-DPO, ETO, IPR, Step-DPO);
- (2) KTO / ORPO / SimPO on SWE trajectories;
- (3) offline value-based RL (ArCHer, Digi-Q, ILQL);
- (4) any offline-vs-online comparison at matched compute on SWE;
- (5) any of the above with **LoRA on a ~30B 4-bit base**.

The closest SWE-specific offline evidence is Agent-RLVR (trajectory DPO, 72B full FT), Nebius RFT (72B full), SWE-Gym (32B, LoRA only for MoatlessTools RFT and the verifier), and SWE-Prime (30B-A3B MoE full SFT).

### Cited Findings
- LoRA appears in SWE work only as:
  - SWE-Gym's 32B MoatlessTools RFT and its verifier (LoRA beat full FT for the verifier);
  - R2E-Gym's 14B verifier (r64);
  - A²Agent (r16, 4–8B, localisation).
  - VERIFIED — [SWE-Gym](https://arxiv.org/abs/2412.21139); [R2E-Gym App. C.2](https://arxiv.org/abs/2504.07164)
- Agent-RLVR states its DPO is offline and one iteration, and that online/iterated variants are future work. VERIFIED — [Agent-RLVR §4.2](https://arxiv.org/abs/2506.11425)
- Agentic-DPO evaluates only StableToolBench, τ-bench, and Mind2Web. VERIFIED — [Agentic-DPO §4.1](https://arxiv.org/abs/2607.10601)

### Inferences
- A **paper-track opening** (INFERRED): "Step-level offline preference optimisation for SWE agents on a 31B 4-bit base with LoRA, using programmatic negatives (garbled-arg tool calls, exact-repeat calls, submit-without-test) at the agent's own failure states, compared against RFT and filtered-BC at matched L4-hours". All three cited ingredients exist separately:
  - Agentic-DPO: method.
  - SWE-Prime: process filters.
  - Nebius: error-turn masking.

  Their combination on SWE was not found.
- Iteration count: plan for 2 rounds (generate → train → regenerate → train), with per-instance caps. Evidence suggests diminishing returns after that. INFERRED

### Gaps
- OpenReview / NeurIPS 2026 camera-ready lists were not searched directly. Absence claims rely on alphaXiv/web search only.
- IPR, ETO, ArCHer, Digi-Q originals not re-read.
- Whether QeRL-style NVFP4 runs on L4 (Ada): unverified (carried over from the sibling note).

### Search log
| # | Tool | Query / target | Useful results |
|---|---|---|---|
| 1 | alphaXiv discover | "RL for SWE agents … offline substitutes like RFT, DPO on trajectories, step-level DPO, AW-SFT"; kw [SWE-bench, reinforcement learning, rejection sampling, DPO, SWE agent] | FC-SWE 2610.07898, SWE-MILE, CRR, **Agentic-DPO 2607.10601**, LEGO-RL, SWE-TRACE, **SWE-Prime 2608.27449**, Open-SWE-Traces, Socratic-SWE, SWE-MeM |
| 2 | alphaXiv discover (recency) | "offline RL or iterated batch self-improvement for multi-turn coding agents vs online GRPO"; kw [offline RL, GRPO, multi-turn, agent, coding] | PIVOT, self-practice OPSD, Agent Lightning v1, Critical-State RL, Mu-GRPO (off-policy GRPO), RODS (no SWE offline-vs-online comparison) |
| 3 | alphaXiv PDF | Agentic-DPO 2607.10601 | method, τ-bench vs GRPO, cost 1.6× vs 13.6× |
| 4 | alphaXiv PDF | SWE-Prime 2608.27449 | filtered/segment-masked BC table |
| 5 | alphaXiv PDF | FC-SWE 2610.07898 | GRPO vs FC-SWE, −5 invalid-call penalty collapse |
| 6 | alphaXiv PDF | Nebius 2508.03501 | RFT 20.5 → RL 39.0, compute, loop-filter finding, comparison table |
| 7 | alphaXiv PDF | SWE-Gym 2412.21139 | self-RFT failure, iterative RFT, ORM Best@k, LoRA verifier |
| 8 | alphaXiv PDF | R2E-Gym 2504.07164 | SFT 34.4, hybrid 51.0, verifier limits |
| 9 | alphaXiv discover | "DPO/KTO on SWE agent trajectories (step/trajectory level)" | Agentic-DPO only; no SWE step-DPO hit |
| 10 | alphaXiv discover | "reward model / verifier for selecting SWE patches (SWE-RM …)" | SWE-RM 2512.21919, Dockerless 2606.28436, CER 2609.31995, LLM-as-a-Verifier, SWE-ZERO→HERO |
| 11 | alphaXiv PDF | SWE-RM 2512.21919 | TTS gains, AUC/ECE, data scale, RL +3 pp |
| 12 | WebFetch | together.ai/blog/deepswe | 42.2 / 59, 64 H100 × 6 days |
| 13 | alphaXiv PDF | Kimi-Dev 2509.23045 | 60.4 agentless, 48.6 agent SFT, RL-prior vs SFT-prior |
| 14 | alphaXiv PDF | Agent-RLVR 2506.11425 | offline DPO 22.4 vs RFT 20.8; RM 27.8 |
| 15 | alphaXiv PDF | SWE-RL 2502.18449 | SFT 36.2 vs RL 41.0; compute; reward ablation |
| 16 | alphaXiv PDF | CER 2609.31995 | early reward TTS / RL; length bias; file-coverage AUROC 0.608 |
| 17 | alphaXiv discover | "offline value-based RL (Q-function, critic, ILQL, Digi-Q, ArCHer) or iterative RFT with LoRA for SWE agents" | No offline value-RL SWE paper; Opera, PACT, RRSI, HELIX, Polar |
| 18 | WebSearch | "SkyRL-v0 SWE-bench Verified OpenHands-7B 14B results" | SkyRL-v0 numbers (snippet), SkyRL-Agent 2511.16108 |
| – | Local read | CLAUDE.md, docs/HQ.md, sibling notes post_training_navigation.md / critic_judgement.md | context; CodeScout/A²Agent/CRR/P2T/QeRL facts reused, not re-searched |
