# Prior-art red team: SCCS, partial-fix negatives, NF4 to W4A16 adapter transfer (8 Oct 2026)

Role: hostile reviewer. The job was to find the paper that already did each hypothesis. Inputs: `critic_judgement.md` (its queries Q1–Q26 were not repeated) plus the new searches logged at the end.

Labels:
- **VERIFIED**: read in the paper's full text (alphaXiv PDF pages) or a primary doc page this session.
- **SNIPPET**: seen only in a search-result abstract or snippet.
- **INFERRED**: my reasoning, or recalled from memory and not re-checked.

Closeness scale: 0 = unrelated, 1 = same area but a different mechanism, 2 = same mechanism in a different setting (a reviewer will call it "X applied to Y"), 3 = scooped.

All 2026 arXiv items are preprints unless a venue is stated.

---

## Q1. Unlikelihood and repetition-suppression training: how close to SCCS?

### Takeaway
SCCS is not scooped, but **DITTO (Xu et al., NeurIPS 2022) is a closeness-2 hit that the critic missed.** DITTO already does three things:
- It measures the copy/self-reinforcement mechanism by feeding repeated content and watching the repeat probability rise.
- It trains a penalty on repeating the previous occurrence, relative to that occurrence's own probability.
- It explicitly frames useful repetitions as positives against penalised pseudo-repetitions, so the model learns to "distinguish between useful and useless contexts". That is the guard idea.

SCCS's defensible delta is narrower than the critic stated:
1. The negative label comes from **environment semantics** (an identical call with no state-mutating action in between has a provably identical observation), not from surface repetition.
2. The negatives are the agent's **own visited states**, offline, rather than synthetic repeated sentences.
3. The guard is a **separately labelled legitimate-repeat class** (post-change repeats).
4. The setting is a 31B 4-bit tool-calling agent.

A second near-neighbour is **aTTT (2607.03441)**. It applies repetition-aware LoRA updates *inside* SWE-bench Lite episodes.

### Cited Findings
- **Welleck et al. 2019, "Neural Text Generation with Unlikelihood Training"** (arXiv 1908.04319). It proposes token- and sequence-level unlikelihood to reduce dull, repetitive text. SNIPPET (alphaXiv listing). DITTO's tables report UL-token and UL-token+seq as baselines (Wikitext-103, greedy): Repetition-Sen 7.40% and 0.05% vs MLE 14.50%. MAUVE is 0.57 and 0.48 vs DITTO 0.77. VERIFIED via DITTO Table 1. **Closeness 1**: same objective family, text-level, no state conditioning. — [Welleck 2019](https://www.alphaxiv.org/abs/1908.04319); [DITTO](https://www.alphaxiv.org/abs/2206.02369)
- **DITTO / "Learning to Break the Loop"** (Xu et al., NeurIPS 2022, arXiv 2206.02369). VERIFIED. **Closeness 2.**
  - **Mechanism.** Repeating a sentence N times makes the repeat probability rise almost monotonically: the "self-reinforcement effect". Sentences with a higher initial probability reinforce more strongly. The authors' explanation is that the model "tries to find hints in context" and copies.
  - **Loss.** `−log(1 − |P(x_n,l) − λ·P*(x_{n−1},l)|)` with a stop-gradient on the previous repetition. It is trained on pseudo-repeated data, mixed 1:1 with MLE.
  - **Guard concept.** Necessary repetitions "can be viewed as positive samples, and our pseudo repetitive data can be viewed as negative samples… the model should learn to distinguish between useful and useless contexts". They contrast this with UL-token, which penalises "any repetitions (regardless of the necessity)".
  - **Over-suppression warning.** A margin variant learned a "cheap" solution that drove the repeat probability to ~0 after a single repetition, i.e. it "over-penalize[d]" every repetition.
  - Text only (Wikitext-103, CNN/DM). No agents, no state.
  - [Source](https://www.alphaxiv.org/abs/2206.02369)
- **aTTT, "No Time Like the Present: Agentic Test-Time Training"** (Kuaishou, arXiv 2607.03441). VERIFIED. **Closeness 1–2.**
  - **Method.** Online, per-episode LoRA updates served through vLLM's runtime LoRA API. The loss *down-weights* (does not penalise) tokens that fall in n-grams repeated from earlier updates: `w = max(w_min, 1/(1+f))`. This is a positive-loss reweighting, not unlikelihood.
  - **Results.** SWE-bench Lite: Qwen3.5-9B 35.6 → 38.4; Qwen3.5-27B 57.8 → 62.7 (3 seeds). ALFWorld up to +5.0.
  - **Useful control for us.** A decoding repetition penalty of 1.1 scored 49.7% vs the no-TTT baseline at 50.7% (ALFWorld, Qwen3.5-9B). So decoding penalties hurt.
  - **Repeat statistics.** Gemma-3-12B repeats its previous action in 7% of steps (45% within the last 3) on ALFWorld failures. Qwen3.5-4B repeats its previous action in 71%.
  - It does not condition on state change and does not use negatives.
  - [Source](https://www.alphaxiv.org/abs/2607.03441)
- **Implicit Unlikelihood Training** (arXiv 2101.04229) uses RL to regularise repetition. SNIPPET. **Closeness 1.** — [Source](https://www.alphaxiv.org/abs/2101.04229)
- **"Repetition In Repetition Out"** (arXiv 2310.10226) attributes degeneration to repetition in the training data. SNIPPET. **Closeness 0–1.** — [Source](https://www.alphaxiv.org/abs/2310.10226)
- **Search hit not followed up.** A web snippet attributes this result to a paper: "the corrective gain of a failure record… is negative for every instruction-tuned model tested… repeat probability… rises from 0.06 to 0.54". This matches Feedback That Backfires (2608.23651), already in the critic's table. SNIPPET. — [beri.net summary](https://www.beri.net/article/small-model-agent-repeats-failed-tool-call-harness-error-format)
- **Mechanistic induction-head/copying work on repetition:** the alphaXiv query returned **no papers**. That is an index result only.

### Inferences
- INFERRED: a reviewer will write "SCCS = DITTO/unlikelihood applied to tool calls". The rebuttal has to rest on the **state-equivalence label**. DITTO's negatives are *any* consecutive sentence repeat. SCCS's negatives are repeats that are provably zero-information given the environment, and its positives include identical-surface repeats that *are* informative (rerun the test after an edit). That is a sharper version of DITTO's useful/useless distinction, and it can be checked mechanically.
- INFERRED: the method section must add a **DITTO-style relative-decay loss arm** (B′) next to plain unlikelihood (B). Otherwise the reviewer asks "why not DITTO?". DITTO's finding that the margin variant over-penalises is direct support for SCCS's small-λ plus guard design.
- INFERRED: the Stage-1 premise test (log-prob with vs without the earlier identical call visible) is a single-step, real-agent analogue of DITTO's Figure 2 (repeat probability against number of repeats). Cite it as the template. The claimed novelty is then the agent setting and the 31B scale, not the probe design.
- INFERRED: aTTT is the most dangerous *competitor result*. It already shows that repetition-aware LoRA updates move SWE-bench Lite by +2.9 to +4.9. A reviewer will ask why offline SCCS beats or complements aTTT. The fair answer: aTTT needs per-episode training at serve time (1.9× wall clock) and is impossible under our Kaggle constraints, while SCCS is a static adapter.

### Gaps
- I could not reach the full text of Welleck 2019. Its numbers are taken via DITTO's baseline table.
- No paper was found that tests the self-reinforcement/copy effect on **tool calls** above 1.7B. Feedback That Backfires stops at 1.7B (critic note).
- OpenReview (ICLR 2027 submissions) was not searchable with these tools. A concurrent "DITTO for agents" submission cannot be excluded.

---

## Q2. Offline training of agents on negative or failed actions: how close to SCCS?

### Takeaway
The agent-tuning literature uses negatives at the **trajectory** level (NAT prefix tags, ETO DPO) or fixes loops through **positive revision data** (Agent-R). None uses a deterministic per-step "zero-information action" label as a negative. The closest is a loop-reduction result, not a method match: **Agent-R reduces repeated-action loops**, but only through SFT on spliced revision trajectories built with MCTS and a reward signal. **Closeness ≤1 for all.**

### Cited Findings
- **NAT, "Learning From Failure"** (Wang et al., arXiv 2402.11651). VERIFIED. **Closeness 0–1.**
  - It adds a "correctly / incorrectly" suffix to the query and trains plain SFT on both positive and negative trajectories, with loss on model tokens.
  - Prompt content does not matter: random strings work equally well (avg 64.04 vs 63.55).
  - Low-quality negatives hurt: −3.16 / −6.20 vs Vanilla.
  - Trajectory-level, math/QA, no repetition focus.
  - [Source](https://www.alphaxiv.org/abs/2402.11651)
- **Agent-R** (Fudan/ByteDance, arXiv 2501.11425). VERIFIED. **Closeness 1.**
  - It splices a bad prefix, a revision signal and a good continuation found by MCTS, then trains with SFT only.
  - Finding 4: training on revision trajectories "significantly reduces the occurrence of dead loops", measured by the count of repeated action sequences. Training on MCTS-optimal trajectories alone still loops.
  - Needs online MCTS rollouts with environment reward. No negatives, no state-equivalence label.
  - [Source](https://www.alphaxiv.org/abs/2501.11425)
- **ETO** is described in Agent-R as "first applies SFT… then uses DPO… with contrastive pairs of good and bad trajectories". It is trajectory-level. Agent-R reports that ETO-trained agents cannot self-correct once "stuck in a loop" (revision setting, Table 3). VERIFIED via Agent-R. **Closeness 1.** — [Agent-R](https://www.alphaxiv.org/abs/2501.11425)
- **Dependency-Aware Trajectory Refinement** (ShanghaiTech, arXiv 2609.18417). VERIFIED. **Closeness 1.**
  - It builds an LLM-annotated round-level dependency DAG. Its prompt lists "Redundant confirmation: Round j merely re-verifies a fact already established" as a no-edge pattern.
  - It **prunes** out-degree-0 rounds before SFT and does not use them as negatives. L1 gives +1.7 pp over vanilla SFT on multimodal QA.
  - A DPO appendix exists (Secs F–G) that I did not read.
  - Relevant as an "information-free round" concept, but the label comes from an LLM annotator, not environment state.
  - [Source](https://www.alphaxiv.org/abs/2609.18417)
- **Related training-based anti-redundancy hits, all closeness ≤1. SNIPPET unless noted.**
  - Co-Evolving Agents: failures as hard negatives (2511.22254) — [link](https://www.alphaxiv.org/abs/2511.22254)
  - Exploring Expert Failures (2504.13145) — [link](https://www.alphaxiv.org/abs/2504.13145)
  - AgentRefine (2501.01702) — [link](https://www.alphaxiv.org/abs/2501.01702)
  - TRUST (2606.06976), VERIFIED. Uncertainty-aligned GRPO for *when* to call a tool; no repetition — [link](https://www.alphaxiv.org/abs/2606.06976)
  - When2Tool / Probe&Prefill (2605.09252), VERIFIED. Inference-time probe; its SFT baseline "does not reliably reduce tool calls" — [link](https://www.alphaxiv.org/abs/2605.09252)
- **Agent-FLAN negative samples and IPR** were not re-read this session. My memory says Agent-FLAN adds negative samples for *hallucinated tool use* and IPR does step-level DPO with MC-estimated step rewards. INFERRED. Both are trajectory or step preference methods without a state-equivalence label.

### Inferences
- INFERRED: across NAT, ETO, IPR, Agent-R, AgentRefine, Agentic-DPO, AED and AKBE (the last three from the critic), nobody constructs a step negative whose label is **computable from the trajectory alone with no reward, judge or teacher**. That information symmetry is SCCS's cleanest novelty claim. State it explicitly in the paper.
- INFERRED: Agent-R's Finding 4 gives a reviewer the line "loops can be fixed with revision SFT". SCCS should show its arm A (own-state SFT on the next non-repeated action) against arm B. That is exactly a revision-style positive without MCTS. The critic's K2 kill criterion (A ≈ B) already covers it.

### Gaps
- IPR and Agent-FLAN full texts were not verified.
- Fission-GRPO (ACL 2026) and the "Redundant or Necessary?" benchmark (2605.29893) remain unread (carried over from the critic).

---

## Q3. Partial-patch and patch-completeness negatives for program repair: how close to the runner-up?

### Takeaway
No paper was found that builds preference negatives by **dropping one interventionally necessary function from a multi-function gold patch**. The nearest is **"Teaching Agents to Code Reliably" (Stanford/AWS, 2610.03984)**:
- It generates **mutated variants of patches** (boundary, value, logic, control-flow and *effect* faults) as oracle-rejected negatives, but uses them to reward a *test-writing verifier* with RL.
- Its SFT corpus includes "contrastive-edit examples pairing a correct repair with an incorrect candidate at the same location".

**Closeness 1–2.** A reviewer may call function-drop "just another mutation operator". The defensible delta is (a) the **necessity labelling** (an executed-graph or interventional check that the dropped function is required), (b) **multi-function granularity**, and (c) use in **policy** preference rather than verifier reward.

### Cited Findings
- **Teaching Agents to Code Reliably** (arXiv 2610.03984, 2 Oct 2026). VERIFIED. **Closeness 1–2.**
  - The harness "generates variants from the current patch using deterministic operators and a frozen model", covering "boundary, value, logic, control-flow, and effect faults".
  - The scoring set holds oracle-rejected variants and sibling repairs.
  - RL raises pass@1 on 270 held-out Verified issues from 35.2 (SFT) to 43.0 and verifier precision from 26.8 to 41.7.
  - Weighted SFT includes contrastive-edit pairs at the same location.
  - Backbone is Qwen3-Coder-30B-A3B; mutants come from frozen Claude Opus-4.8.
  - **No partial-subset (dropped-hunk or dropped-function) negatives are described.**
  - [Source](https://www.alphaxiv.org/abs/2610.03984)
- **Kimi-Dev** (arXiv 2509.23045). VERIFIED. **Closeness 0.** RL uses an "outcome-based reward only" (0/1 on ground-truth tests) plus "positive example reinforcement". No constructed negative patches. — [Source](https://www.alphaxiv.org/abs/2509.23045)
- **Multi-hunk repair is recognised as a problem, but only as analysis and benchmarks.** SNIPPET.
  - "Beyond Accuracy: Behavioral Dynamics of Agentic Multi-Hunk Repair" (arXiv 2511.11012) — [link](https://arxiv.org/pdf/2511.11012)
  - "Characterizing Multi-Hunk Patches" (ASE 2025): success declines with divergence and spatial dispersion — [link](https://conf.researchr.org/details/ase-2025/ase-2025-papers/64/Characterizing-Multi-Hunk-Patches-Divergence-Proximity-and-LLM-Repair-Challenges)
- **Also returned, with no partial-gold negatives in the snippets** (SNIPPET):
  - PAFT, minimal-edit preservation fine-tuning (2604.03113) — [link](https://www.alphaxiv.org/abs/2604.03113)
  - BoostAPR, dual reward models (2605.09134) — [link](https://www.alphaxiv.org/abs/2605.09134)
  - Step Rejection Fine-Tuning (JetBrains, 2605.10674) — [link](https://www.alphaxiv.org/abs/2605.10674)
  - Refine After Generation (2608.13292) — [link](https://arxiv.org/pdf/2608.13292)
  - SWE-Dev (2506.07636) — [link](https://www.alphaxiv.org/abs/2506.07636)
- **NAT-2** grades negatives by F1 into "almost wrong" vs "mostly correct" classes. That is graded-partial negatives in QA, not code. VERIFIED. **Closeness 0–1.** — [Source](https://www.alphaxiv.org/abs/2402.11651)

### Inferences
- INFERRED: the runner-up's novelty is intact but thin. Frame it as "**necessity-certified partial-fix negatives**" and cite 2610.03984 as the mutation-negative baseline. A strong paper would compare against randomly dropped hunks (not necessity-certified) as a control. If random drops work as well, the IFG link adds nothing.
- INFERRED: 2610.03984's own conclusion says "negatives must be manufactured rather than collected". That supports the general direction, so a reviewer will not reject the idea of constructed negatives.

### Gaps
- RepairLLaMA, PRefLexOR and RECAP full texts were not re-read. RECAP's direction comes from the IFG search, per the critic. The RECAP lookup by title failed this session.
- No OpenReview or ACL Anthology venue check.

---

## Q4. QLoRA adapter portability across quantisation formats: how close to the third hypothesis?

### Takeaway
I found no paper measuring an **NF4-trained, unmerged** LoRA served on a different int4 format (GPTQ/AWQ/compressed-tensors W4A16). The literature splits two ways:
- **Quantisation-aware initialisation or merging:** QA-LoRA, LoftQ, ProjQ (ICML 2026), Scale-QLoRA, GPTQ-intrinsic LoRA.
- **Practitioner warnings about merging into 4-bit**, with numbers behind a paywall.

vLLM's own example applies one LoRA to both AWQ and GPTQ TinyLlama bases without any fidelity measurement. **Closeness 1.** The hypothesis survives as a **measurement note**. A reviewer will call it engineering, not research.

### Cited Findings
- **ProjQ** (ICML 2026, PMLR 306; arXiv 2606.00494). VERIFIED. **Closeness 1.**
  - It shapes PTQ noise (GPTQ solver) into a low-rank subspace so a following LoRA can absorb it.
  - Compares against QLoRA (RTN and GPTQ-initialised), LoftQ and CALDERA at 2/3/4 bits. At 4-bit, Qwen2.5-7B commonsense: QLoRA 71.59, LoftQ 64.56, ProjQ 66.78.
  - Train and serve use the **same** quantised base. No cross-format transfer.
  - It also summarises QA-LoRA (ICLR 2024) as group-wise operators so the adapter "can be perfectly merged into the quantized weights for efficient INT4 inference".
  - [Source](https://www.alphaxiv.org/abs/2606.00494)
- **vLLM "LoRA with quantization inference" example** (docs v0.12.0). VERIFIED.
  - bitsandbytes uses `huggyllama/llama-7b` with `timdettmers/qlora-flan-7b`.
  - AWQ and GPTQ both use `TheBloke/TinyLlama-1.1B-Chat-v0.3-{AWQ,GPTQ}` with the **same** adapter `jashing/tinyllama-colorist-lora`.
  - Cross-format reuse is shown in practice, with no fidelity numbers.
  - [Source](https://docs.vllm.ai/en/v0.12.0/examples/offline_inference/lora_with_quantization_inference/)
- **vLLM forum.** SNIPPET.
  - bnb QLoRA serving needs the adapter fixed at load time, cannot hot-swap, and is "only Llama-based models". [Source](https://discuss.vllm.ai/t/support-for-deploying-4-bit-fine-tuned-model-with-lora-on-vllm/1186)
  - For GPTQ, the LoRA runs in a separate BF16 kernel added to the dequantised-base output. [Source](https://discuss.vllm.ai/t/how-lora-bf16-is-fused-into-a-gptq-8bit-model/786)
- **Kaitchup, "Don't merge your LoRA adapter into a 4-bit LLM".** Merging "may lead to a significant performance drop". The measured numbers are paywalled. VERIFIED (public part only). — [Source](https://kaitchup.substack.com/p/dont-merge-your-lora-adapter-into?open=false)
- **Practitioner blog.** "many people are training in bnb nf4, then merging to an unquantized base… and then re-quantizing". SNIPPET, low-authority source. — [Source](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026)
- **Other quantisation + PEFT items.** SNIPPET; all same-format.
  - Quantization-Aware Healing (2608.20953) — [link](https://www.alphaxiv.org/abs/2608.20953)
  - "Not All 4-bit Quantizers Are Equal" (2609.25014): quantiser choice changes a fine-tuned model's behaviour (PII leakage) — [link](https://www.alphaxiv.org/abs/2609.25014)
  - Scale-QLoRA and GPTQ-intrinsic LoRA are in the critic's table.

### Inferences
- INFERRED: "Not All 4-bit Quantizers Are Equal" is the closest *conceptual* support. The format of the 4-bit base changes downstream behaviour of a fine-tuned model. It concerns quantising *after* fine-tuning, not runtime adapters across formats.
- INFERRED: the expected reviewer reaction is "small, useful, workshop table". Novelty is real (a gap) but low-value unless the gap is large (≥3% NLL or ≥2 pp parse validity, per the critic's kill line).

### Gaps
- vLLM and PEFT GitHub issues on NF4-trained adapters with compressed-tensors bases were not found in this search. The GitHub-specific query returned only forum and docs pages.
- No study of long agent outputs (tool-call validity) under adapter format mismatch.

---

## Verdicts per hypothesis

| Hypothesis | Closest prior work (closeness) | Remaining delta | Likely reviewer reaction |
|---|---|---|---|
| **SCCS** | **DITTO 2206.02369 (2)**: mechanism probe + repeat penalty + useful/useless distinction, text only. Feedback That Backfires 2608.23651 (2, premise only, ≤1.7B, from critic). aTTT 2607.03441 (1–2): repetition-aware LoRA on SWE-bench Lite, online. Welleck UL (1). Agent-R (1). AKBE (1–2, from critic) | Environment-semantics, reward-free, information-symmetric step negatives (no mutation in between → zero-information call). Explicit post-change-repeat guard class. Offline own-state data. 31B W4A16 tool agent. Copy test on real agent states | "Incremental: DITTO/unlikelihood for tool calls." Survives **if** the paper (i) cites DITTO prominently, (ii) adds a DITTO-loss arm and a decoding-penalty baseline (aTTT shows the penalty hurts), and (iii) shows the guard keeps post-change repeats, which DITTO only argued informally. Novelty: **narrowed but alive (N = 1.5/3)** |
| **Partial-fix negatives** | Teaching Agents to Code Reliably 2610.03984 (1–2): mutation-variant negatives for verifier RL, plus same-location contrastive SFT pairs | Necessity-certified dropping of a whole function from a multi-function gold patch, used for policy preference | "Function-drop is another mutation operator." Needs a random-hunk-drop control to show necessity labelling matters. Novelty **alive (N = 2/3)**, feasibility still gated on IFG |
| **NF4 → W4A16 adapter transfer** | ProjQ / QA-LoRA / LoftQ / Scale-QLoRA (1): same-format train and serve. vLLM docs (1): reuse across formats without measurement | First measurement of an unmerged NF4-trained adapter on a W4A16 base for long agentic outputs | "Engineering note." Fine as one table. Novelty is a real gap but **low value (N = 1/3)** |

**Bottom line for the coordinator.** No hypothesis is killed. SCCS loses part of its novelty claim to DITTO. The critic's "Unlikelihood training as an objective: GUESSED citation" row should become a full related-work paragraph covering Welleck 2019, DITTO 2022 and aTTT 2026, and the method should add a DITTO-loss ablation arm.

---

## Search log (this session; does not repeat critic Q1–Q26)

| # | Tool | Query / target | Outcome |
|---|---|---|---|
| R1 | alphaXiv discover (historical) | "unlikelihood or negative training to reduce repetition and self-reinforcing loops… learning to break the loop, DITTO"; kw [unlikelihood training, repetition, self-reinforcement, DITTO, neural text degeneration] | **DITTO 2206.02369**, Welleck 1908.04319, Implicit UL 2101.04229, FOCUS&RePAIR, Repetition-In-Out 2310.10226, Holtzman 1904.09751 |
| R2 | alphaXiv discover | "training LLM agents with negative or failed trajectories offline: NAT, ETO, IPR, Agent-FLAN…" | AgentHER, ReflectRL, AED, Harness-R1, TRAJDEBUG, CausalFlow, etc. No step-level state-equivalence negatives |
| R3 | alphaXiv discover | "fine-tuning LLM agents to avoid repeating the same action or tool call… penalizing repeated actions" | **aTTT 2607.03441**, AKBE, **Dependency-Aware Refinement 2609.18417**, PATS, TACT, AgentLoop, infinite-loop study 2607.01641 |
| R4 | alphaXiv discover | "preference or contrastive training for program repair using incomplete or partially correct patches as negatives" | PAFT, Step-RFT, ContraFix, BoostAPR, ComPass, FC-SWE, ExecCritic, Refine-After-Generation. No partial-gold negatives |
| R5 | alphaXiv PDF | DITTO 2206.02369 | Loss, self-reinforcement, useful-vs-useless discussion, margin over-penalisation |
| R6 | alphaXiv PDF | aTTT 2607.03441 | Online repetition-aware LoRA TTT; SWE-Lite +2.9/+4.9; decoding penalty hurts |
| R7 | alphaXiv PDF | Dependency-Aware Refinement 2609.18417 | DAG leaf-prune of redundant rounds for SFT; not negatives |
| R8 | alphaXiv discover (<2026) | "recover from repeated or looping actions… Agent-R, AgentRefine; negative examples" | Agent-R, AgentRefine, NAT, Expert Failures, Co-Evolving hard negatives, structured reflection |
| R9 | alphaXiv discover | "LoRA adapter trained on one quantization format then served on a different quantized base" | Scale-QLoRA, **ProjQ**, GPTQ-intrinsic LoRA, QA-Healing, Not-All-4-bit-Quantizers |
| R10 | alphaXiv PDF | Agent-R 2501.11425 | Loop reduction via revision SFT (Finding 4); ETO description |
| R11 | alphaXiv PDF | NAT 2402.11651 | Prefix tagging; random strings work; NAT-2 graded negatives |
| R12 | alphaXiv PDF | ProjQ 2606.00494 | Same-format adapter-aware PTQ; no cross-format study |
| R13 | WebSearch | "QLoRA adapter trained bitsandbytes NF4 applied to GPTQ AWQ base model vLLM accuracy mismatch" | Blogs only; no measurement |
| R14 | WebSearch | "unlikelihood loss repeated tool calls LLM agent fine-tuning loop" | ToolScan RAC, SciAgentGym, FTB snippet, 2606.06976, 2605.09252 |
| R15 | WebSearch | "preference data incomplete patch rejected SWE-bench DPO partial fix multi-hunk negatives" | Multi-hunk analyses (2511.11012, ASE'25); no partial-gold DPO |
| R16 | alphaXiv PDF | TRUST 2606.06976 | Tool-call decision GRPO; not repetition |
| R17 | alphaXiv PDF | When2Tool 2605.09252 | Probe&Prefill; SFT does not reduce calls; not repetition |
| R18 | alphaXiv discover | "reward model or preference training for SWE agents using synthetic hard negative patches created by mutating or removing hunks from the gold patch" | **Teaching Agents to Code Reliably 2610.03984**, CATCH, hacktrace, reward-hack audits |
| R19 | WebSearch | "vllm github issue LoRA adapter GPTQ quantized base trained with QLoRA bitsandbytes different outputs" | vLLM forum and docs pages |
| R20 | WebFetch | docs.vllm.ai lora_with_quantization_inference | Same adapter on AWQ and GPTQ bases; no fidelity numbers |
| R21 | WebFetch | Kaitchup "don't merge your LoRA adapter into 4-bit" | Qualitative warning; numbers paywalled |
| R22 | alphaXiv PDF | 2610.03984 | Mutation-variant negatives for verifier RL; contrastive-edit SFT pairs |
| R23 | alphaXiv discover | "mechanistic cause of repetition via induction heads / in-context copying" | **No papers found** |
| R24 | alphaXiv discover | "DPO… SWE-bench… rejected patches incomplete fixes, Kimi-Dev, SWE-Dev…" | Kimi-Dev, SWE-Dev, Agentic-DPO, Open-SWE-Traces, SWE-Gate, AgentLens. No partial-gold |
| R25 | alphaXiv PDF | Kimi-Dev 2509.23045 | Outcome-only RL; no negatives |
| R26 | alphaXiv PDF | "RECAP program repair" by title | Lookup failed (no match) |

**Not searched:**
- OpenReview ICLR-2027 submissions;
- Semantic Scholar citation graph of DITTO (the most likely place for a "DITTO for agents" follow-up);
- IPR and Agent-FLAN full texts;
- vLLM and PEFT GitHub issue trackers directly.

Absence claims rest on alphaXiv's index and web search only.
