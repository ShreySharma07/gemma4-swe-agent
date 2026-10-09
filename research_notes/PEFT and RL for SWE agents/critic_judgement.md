# Critic judgement: PEFT and offline RL-style tuning for the Gemma-4-31B SWE agent (8 Oct 2026)

Role: adversarial reviewer (harsh area chair + pragmatic Kaggle grandmaster). Inputs: the four notes in this folder, `reports/LLM graph reasoning for code.md` (IFG pick), `CLAUDE.md`, `docs/{HQ,plan,findings,experiments}.md`, `results/runs/*/summary.json` and `r2_v2_sed/per_task.jsonl` (aggregates only). No experiment was run; nothing but this file was written.

Labels: **VERIFIED** = read in a primary source or computed from our own result files this session. **GUESSED** = my inference or an effect size I am predicting. Paper status: all 2026 arXiv items are preprints unless stated.

> **Read Round 4 first (R4.1–R4.8, at the end of this file).**
> - Field-level gaps are in R4.1. The DECAF architecture is in R4.2.
> - Under the **official paper rubric (R4.8)** the final ranking is:
>   1. **Z**: zero-information actions, i.e. the SCCS line reframed field-level (21/25);
>   2. **DECAF** (16/25);
>   3. **SCAD-R** (16/25).
> - Recommended paper: headline Z, plus DECAF Tier A as an analysis section.

## Bottom line

- **Top pick (H\*): State-conditional copy suppression (SCCS).** Gemma's exact repeats are mostly *no-progress repeats*: the same read-only call re-issued with no edit in between, so its output is guaranteed identical and carries zero new information. Train a small LoRA at the agent's **own** visited states, with an unlikelihood negative on the verbatim no-progress repeat and an SFT guard on legitimate repeats (re-running a test after an edit). Prediction (GUESSED): versus a matched SFT-only control, greedy one-step no-progress re-emission on held-out repos falls by ≥15 pp, with no loss on legitimate repeats.
- **Why this one:** it hits the failure that is actually widespread in the frozen v2 scaffold. Exact repeats occur in **102/129 tasks**. Malformed arguments, by contrast, occur in only **9/129** (VERIFIED, `r2_v2_sed/per_task.jsonl`). It needs no teacher, no privileged information, no reward model and no online RL. Its labels are deterministic and computable from the student's own context. That sidesteps the failure of hinted self-distillation that PSP documents (2609.29051).
- **Cheapest falsifier:** Stage 0 (CPU, ~3 h) plus Stage 1 (no training, ~1.5 GPU-h of prompt-logprobs on the served W4A16 model) can kill the premise. Stage 2 (~2.5 GPU-h, two r=16 LoRAs) can kill the method.
- **Honest ceiling:** this is a workshop-to-short-paper contribution. The largest leaderboard lever is still the engineering mainline: render-exact, filtered teacher-trajectory SFT. SCCS rides on top of it as an extra loss term.
- **Runner-up:** partial-fix negatives for multi-function fixes. This is IFG-linked, so it serves the graph paper.
- **Third:** a serve-faithful adapter measurement (NF4-trained adapter vs the W4A16 serving base).
- **Killed: 13 of 20 candidates.** Notably: verifier LoRA at k=2 (no time budget), per-role adapters, offline DAgger (needs a teacher), and hinted self-teacher distillation (PSP shows it falls *below base* on SWE-bench).

## Facts the judgement rests on

| Fact | Source | Status |
|---|---|---|
| v2 (frozen scaffold) exact-repeat share 28.4%, loops in 59.7% of tasks | `r2_v2_sed/summary.json` | VERIFIED |
| v2 tasks with ≥1 exact repeat: 102/129. Mean repeats: unresolved 10.35, resolved 3.15. Confounded by length: mean calls 34.3 vs 19.6 | `per_task.jsonl` aggregate, this session | VERIFIED |
| v2 tasks with malformed tool args: **9/129** (70 bad edit calls in 8 tasks). The 61% malformed-edit figure is from **v5**, not v2 | `per_task.jsonl`, `docs/findings.md` | VERIFIED |
| v2 wrong-fix-unverified 47/129; tested after last edit 14.7% | summary | VERIFIED |
| Multi-function: 2/45 tasks with ≥3 gold functions resolved | graph report | VERIFIED (as reported there) |
| Raw r1–r3 traces are **not** on this machine (`data/runs/` holds only `devset_v5`) | `ls`, this session | VERIFIED; whether they can be retrieved from Kaggle outputs is GUESSED |
| GPU: ~5.5 h left until Oct 10, then ~30 h/week → roughly 110–120 GPU-h until the Nov 5 freeze | HQ, Oct 7 | VERIFIED / arithmetic |
| Phase 0b (31B QLoRA training on 4×L4) is **not yet measured** | plan.md | VERIFIED |

---

## Round 1: 20 candidates, scored 0–3 (N = novelty, F = feasibility by Nov 5, P = payoff on LB and paper, Fa = falsifiability with cheap dense metrics). Any 0 kills.

Engineering must-dos are listed separately at the end. They are needed, not novel, and not scored for novelty.

| # | Candidate | N | F | P | Fa | Verdict, plainly |
|---|---|---|---|---|---|---|
| H1 | Step-level offline DPO with programmatic negatives at own failure states (Agentic-DPO → SWE) | 1 | 2 | 2 | 2 | **Survive, wounded.** Agentic-DPO exists (non-SWE). AED's offline action-DPO pilot "establishes no recovery benefit" (2609.40111). It needs a *positive* action at our own states, which we lack without a teacher. |
| H2 | Unlikelihood on exact-repeat calls at post-failure states | 2 | 3 | 2 | 3 | **Survive.** No SWE or agent training paper found (Q2, Q20, Q23, Q24). Risk: the contextual-drag trade-off (2602.04288): suppressing repeats can suppress useful reuse. |
| H3 | Format self-repair LoRA (deterministically re-serialise own malformed calls; loss only on the corrected call) | 1 | 3 | 1 | 3 | **Survive as engineering.** ParamBench PBT, SWE-Lego error masking and AED post-error SFT cover it. **Payoff is low in v2: 9/129 tasks.** |
| H4 | Error-turn masking + SWE-Prime filtering on converted public trajectories | 0 | 3 | 3 | 2 | **Killed as research** (Nebius, SWE-Lego, SWE-Prime). **Kept as the mainline baseline (E7).** |
| H5 | Quantisation-matched LoRA (train on exact W4A16 codes via GPTQ/HQQ) | 1 | 1 | 1 | 3 | **Survive as a measurement only.** Scale-QLoRA and GradCodeS cover merged or fully low-bit adapters, not runtime adapters across int4 formats. PEFT has no compressed-tensors path, and Gemma-4 support in GPT-QModel/HQQ is unknown. |
| H6 | Per-role adapters (localiser / fixer) | 0 | 2 | 0 | 2 | **Killed.** MoRAgent, MoRSE and ExRole exist. The scaffold is frozen at single-agent v2, and the two-agent v5 gained nothing on the LB. |
| H7 | Verifier LoRA for k=2 selection | 1 | 0 | 1 | 2 | **Killed.** Two rollouts plus a judge do not fit ~5 min/task within a 12 h cap (v6 already hit ~10 h). SWE-Gym Best@2 is only +4.2 pp even with k=2. |
| H8 | Repro-test writer adapter (SFT only) | 1 | 1 | 1 | 2 | **Killed.** ExecCritic: SFT-only test validity is 39.6%, and base self-tests *lower* resolve (61.2 → 57.3). Base-fail/gold-pass data generation is too costly by Nov 5. |
| H9 | Anti-loop by dropping looping trajectories | 0 | 3 | 1 | 2 | **Killed.** Nebius §5.2: filtering looping trajectories *increased* looping. Folded into H4's masking. |
| H10 | Adapter concatenation (format ⊕ trajectory, rank sum ≤128) | 0 | 3 | 1 | 2 | **Killed as research.** Exact linear algebra. Kept as a packaging option. |
| H11 | Ordered truncation / prefix-fit for long trajectories | 0 | 3 | 2 | 2 | **Killed as research** (SERA). Recipe item in E6. |
| H12 | Self-RFT on own successes on unseen repos | 0 | 2 | 1 | 2 | **Killed.** SWE-Gym self-RFT regressed (15.3 → 8.7); 32B LoRA RFT gained +0.7. |
| H13 | Offline DAgger: relabel own states with a teacher | 0 | 0 | 2 | 2 | **Killed.** Done on SWE (2605.12913, +3.6–3.9 pts at 4/8B), and it needs a stronger teacher than Gemma-31B, which is not servable offline on L4 within $50. |
| H14 | Hinted self-teacher distillation (OPSD / offline PivotOPD with gold-patch hints) | 1 | 2 | 0 | 2 | **Killed.** PSP (2609.29051): OPSD falls *below base* on SWE-bench Verified (Qwen3-4B 3.56 → 2.40), and it also fails on Gemma4-E4B (AppWorld). |
| H15 | Partial-fix negatives: full gold ≻ necessary-function-dropped patch (IFG-linked) | 2 | 1 | 2 | 2 | **Survive.** No partial-gold negatives were found (IFG search T11; RECAP goes the opposite direction). Feasibility hinges on the IFG go/no-go. |
| H16 | Budget/turn-aware SFT (submit by step k) | 1 | 2 | 1 | 2 | **Killed as separate research.** v2 budget exhaustion is 1.6%. The overrun was a v5 localiser problem. Short-trajectory selection goes into H4. |
| H17 | Validate own-state teacher-forced metrics as a proxy for closed-loop behaviour | 1 | 3 | 1 | 3 | **Survive as the evaluation component, not standalone.** 2609.21187 shows that *gold-history* next-turn gains do not transfer (Gemma-3 tool exact-accuracy even fell). No paper tests *own-state* metrics (Q21: none found). |
| H18 | Gemma-native execution-free data generation on many permissive repos (SERA-style) | 1 | 0 | 2 | 2 | **Killed.** At ~5 min/task on L4, generation yields ~100 trajectories per session. |
| H19 | Test-time sentinel adapter (HiSentinel-style pre-execution REDIRECT) | 1 | 0 | 2 | 2 | **Killed.** It needs a per-step second model call. ADK declarative agents cannot gate tool execution, and doubled calls break the time budget. |
| H20 | "Rerun tests only after a change" + reflexive-repair prompt (zero training) | 0 | 3 | 1 | 3 | **Not research.** Cheap prompt A/B (E8). DevSkills cut ReTest cost up to 41.7% (2609.30725), and Flat Score's repair prompt gave +5.9 on Gemma-31B INT4. |

**Survivors:** H1, H2, H3 (engineering), H5 (measurement), H15, H17 (evaluation).

---

## Round 2: combine known pieces plus one new piece, then attack again

| Combo | Known pieces | One new piece | New attack (search) | Result |
|---|---|---|---|---|
| **C1: State-conditional copy suppression (SCCS)** = H2 + H17 (+ H1 as an ablation arm) | Feedback That Backfires: repeat probability 0.06 → 0.54 after a failure; 83% surface-driven; ≤1.7B only. SWE-Prime redundancy flag. Nebius: keep loops as negatives. Agentic-DPO: SFT anchor is essential. ReTest/SubRetrv detectors (2609.30725). | **Negatives labelled by state-equivalence:** an identical call with no intervening mutating action has a provably identical observation, so it is a zero-information action. The label is deterministic and computable from the student's own context (information-symmetric). Training uses unlikelihood at the agent's own states plus an SFT guard on post-change repeats. Tested on a 31B W4A16 base, offline. | Q20: Contextual Drag shows reset-SFT costs **−22 to −28 pp** when the context is useful → requires a guard. Q24: 2609.30725 already defines ReTest via inter-patch windows (the *label* is known); its mitigation is prompts/skills, and "use behavior diagnoses … for agent training" is its stated future work. Q24: AKBE's *offline* signal over-suppressed necessary calls → requires a guard and a small λ. Q23: no zero-information-action training paper. | **Survives, narrowed.** Novelty = using state-equivalence repeat labels as offline LoRA negatives at own states, plus a copy-mechanism test at 31B. The detector itself is not new. |
| C2: Information-symmetric recovery SFT = H3 + H1 | AED post-error actor SFT; SWE-Lego masking; ParamBench | Targets are restricted to those derivable from the student's view | Q15/Q18: AED already conditions on masked error context; Fission-GRPO (ACL'26) and ContractRL cover recovery with RL | **Killed as research.** Only the payoff-limited format case (9/129 tasks) is truly deterministic. Kept as engineering (E7 data slice). |
| C3: Partial-fix negatives = H15 + IFG labels | IBugDC, Hunk4J, RECAP, A²Agent (graph report) | Preference pairs where a necessary node is dropped, built from executed-graph necessity labels | IFG search log T11 (no hit); no new hit this session | **Survives as runner-up**, conditional on IFG Stage A. |
| C4: Serve-faithful adapter = H5 + H17 | Scale-QLoRA (merge loss; conventions differ by about 1 pp), GradCodeS, Flat Score (INT4 doubles Gemma-31B tool-name hallucination) | Gap in runtime-adapter transfer NF4 → W4A16 for long agent outputs | Q5/Q8: only merged or fully-low-bit studies | **Survives as third** (measurement, cheap, needed anyway). |

---

## Round 3: final pick

### H\*: State-conditional copy suppression (SCCS)

**Falsifiable claim.** Gemma-4-31B-W4A16 in the v2 scaffold re-issues *no-progress repeats* (NPR): a call identical in tool and normalised arguments to an earlier call, with no state-mutating action between them (no `edit_file`, `write_file`, or mutating `run_command` such as `sed -i`, `>`, `git apply`, `pip`). The claim has two parts.

- **(a) Premise.** NPRs are ≥50% of exact repeats and are driven by in-context copying. Replacing the earlier identical call and its output with an abstract placeholder lowers the repeat's log-probability by ≥log 2 (median, paired).
- **(b) Method.** An r=16 LoRA trained at own states with an NPR-unlikelihood term plus an SFT guard compares against a token-matched SFT-only control on held-out-repo states:
  - lowers greedy one-step NPR re-emission by **≥15 pp absolute**;
  - keeps post-change repeats ("rerun test after edit") within **−25% relative** log-probability;
  - keeps tool-call parse validity within **−5 pp**.

  Downstream (GUESSED, Stage 3 only): exact-repeat share 28% → ≤15%; tested-after-last-edit up; resolve **+2–4 pp** on an unseen-repo dev set. The LB cannot resolve this; 1 task ≈ 0.8 pp.

**Known pieces + the new piece**

| Piece | Source | Status |
|---|---|---|
| Failed call in context raises re-emission; 83% from surface form; harness ban 31% → 8% | Feedback That Backfires (2608.23651), 135M–1.7B only | VERIFIED (prior note + abstract) |
| Erroneous context drags later outputs; targeted SFT helps partly but costs −22 to −28 pp on correct-context use | Contextual Drag (2602.04288) | VERIFIED |
| Redundancy flag (adjacent identical calls) as a *filter* | SWE-Prime (2608.27449) | VERIFIED |
| Keep loops as negatives; filtering loops increases looping | Nebius (2508.03501) | VERIFIED |
| Step-level negatives need an SFT anchor (−14 to −18 pts without it) | Agentic-DPO (2607.10601) | VERIFIED |
| ReTest = rerun without patch update (inter-patch windows); SubRetrv; mitigation by prompts/skills | 2609.30725 | VERIFIED |
| Offline efficiency signals over-suppress necessary calls | AKBE offline variant (2605.26952) | VERIFIED |
| Error-masking keeps the error in context, with no special post-error loss | SWE-Lego (2601.01426) | VERIFIED |
| Hinted self-distillation leaks unseen information and falls below base on SWE | PSP (2609.29051) | VERIFIED |
| Unlikelihood training as an objective | Welleck et al. 2019 (from memory, not re-read) | GUESSED citation, must check |
| **NEW:** state-equivalence (zero-information) repeat labels used as **offline LoRA negatives at the agent's own states**, with a post-change guard; copy-mechanism test on a 31B 4-bit SWE agent | — | Not found in Q2, Q15, Q20, Q21, Q23, Q24 (alphaXiv + web). arXiv-index absence only; OpenReview not searched. |

**VERIFIED vs GUESSED**

| Item | Status |
|---|---|
| Repeats are widespread in v2 (102/129 tasks) and more frequent in unresolved tasks (10.4 vs 3.2 mean) | VERIFIED; causal link GUESSED (length confound) |
| Share of exact repeats that are no-progress (vs post-change) | GUESSED (≥50%); measured in Stage 0 |
| Copy-driven at 31B (Feedback That Backfires replicates at scale) | GUESSED; it is untested above 1.7B (Q19) |
| r=16 LoRA can shift one-step behaviour with ~200 examples | GUESSED (ParamBench shows large format shifts on Gemma-4-12B with LoRA) |
| One-step own-state metric predicts closed-loop repeat reduction | GUESSED; 2609.21187 warns that gold-history metrics do not. Ours is own-state, which is untested (Q21). Stage 3 is the check |
| Effect on resolve | GUESSED (+2–4 pp); underpowered at n≈40–60 |
| 31B QLoRA trains on 4×L4 at 8k tokens | GUESSED until Phase 0b (E5) |
| r1–r3 raw traces retrievable from Kaggle outputs | GUESSED |

### Cheapest experiment that could prove H\* wrong

**Data.**
- r2 (v2_sed) raw traces: 129 tasks, the frozen scaffold, temperature 0.2. Use r1/r3 only for robustness.
- If the traces cannot be retrieved: v2 on 30 tasks (~2.5–3 GPU-h). This doubles cost.
- Split by repo for Stage 2: train on fastapi states, evaluate on rich + requests states. These repos are public, but the metric is behavioural, not about knowledge.

**Stage 0: CPU, ~3 h (+1 h trace retrieval).**
- Extend `analysis/trace_analyzer.py` with a repeat classifier: NPR / post-change repeat (PCR) / post-error repeat.
- Report:
  - NPR share of exact repeats;
  - tasks with ≥1 NPR;
  - NPR position;
  - NPR rate in resolved vs unresolved tasks, length-matched by call-index bins.
- **K0 (kill):** NPR <50% of exact repeats, **or** NPR in <25% of tasks. The target is then small: stop and put the effort into the mainline SFT.

**Stage 1: no training, ~1.5 GPU-h (vLLM 0.19.1, W4A16, scorer settings, `prompt_logprobs`).**
- For ~300 NPR events (≤3 per task) and ~150 PCR events, compute:
  - `L_act` = log p(c_t | real context);
  - `L_abl` = same, with the earlier identical call and its output replaced by "[earlier: same command, N lines output omitted]";
  - `L_pre` = log p(c_t | context cut before the first occurrence).
- Also run greedy one-step generation at each NPR state, to check that the model re-emits it at T=0 (base re-emission rate R0).
- **K1 (kills the copy-suppression framing):** median (`L_act` − `L_abl`) < log 2, or R0 < 30%. Repeats are then not in-context copying, or the model does not reproduce them greedily. Fall back to plain filtered SFT and report a negative replication of Feedback That Backfires at 31B, which is still a paper paragraph.

**Stage 2: method test, ~2.5 GPU-h (needs E5 done).**
- Two LoRAs, r=16, all-linear, NF4 of dequantised W4A16, ≤8k-token contexts, ~200 train states each, token-matched:
  - **A (control):** SFT on the agent's own next *non-repeated* action from the same trajectory (own-hindsight, information-symmetric), plus SFT on PCR repeats.
  - **B:** A + unlikelihood on the NPR call (λ ≈ 0.1; AKBE found small λ best).
- Evaluate on ≥150 held-out-repo NPR states and ≥100 PCR states, served on the scorer stack with the adapter.
- **Metrics:**
  - M1 greedy NPR re-emission (B vs A, paired McNemar);
  - M2 PCR log-prob change vs base (guard);
  - M3 parse validity of the one-step output;
  - M4 held-out NLL on clean successful turns (≤+2%).
- **Power:** n=150 paired states, assuming base R0 ≈60%, A ≈45%, B ≈25% (GUESSED). Discordant pairs ≈25–35% give power ≈0.9 for a 15 pp difference at α=0.05.
- **K2 (kill the new piece):** B − A < 15 pp, or McNemar p > 0.05. Also kill if A alone already drives NPR below 10%: the negative is then unnecessary and plain own-state SFT suffices.
- **K3 (kill for harm):** PCR log-prob falls >25% relative, parse validity drops >5 pp, or M4 worsens >2%.

**Why an offline one-step metric is acceptable here, and where it stops.**
- The states are the agent's *own* visited states, not gold history, and the failure is local by definition: one action at one state.
- 2609.21187's negative result is about gold-history turns, so it does not directly apply.
- But one-step metrics cannot see the trajectory-level shift once the adapter changes earlier actions. That is why Stage 3 is required before any LB claim.

**Hours.**
- Stages 0–1: ≈4 h CPU/engineering + ≈1.5 GPU-h. This is the cheapest falsifier.
- Stage 2: ≈2.5 GPU-h + ≈4 h engineering.
- **Total ≤4 GPU-h**, of which only Stage 1 runs before Phase 0b.

**Next step per outcome**
- **Pass (K0–K3 clear):** Stage 3 closed-loop on a 40-task unseen-repo dev set: base vs A vs B, ~3 × 3.5 GPU-h. Then fold the B-loss into the mainline SFT LoRA (E7) as an auxiliary term. Paper: mechanism (Stage 1) + method (Stage 2) + closed loop (Stage 3).
- **Fail at K0/K1:** drop SCCS. Write up the 31B non-replication or the repeat taxonomy as a short findings section. Move to the runner-up if IFG is a go; otherwise put all GPU into the mainline SFT.
- **Fail at K2 (A ≈ B):** keep own-state SFT (A) as a cheap data slice in the mainline. The novelty claim dies.
- **Fail at K3 (harm):** retry once with the PCR guard upweighted and λ halved. If it still harms, kill.
- **Ambiguous (B − A of 5–15 pp, or p between 0.05 and 0.2):** add 150 states from r1/r3 traces (no new rollouts, ~0.5 GPU-h eval) before deciding. Do not tune λ more than twice. That would be fishing.

### Runner-up: partial-fix negatives for multi-function fixes (C3, serves the IFG paper)

**Claim.** On failed multi-function tasks, ≥50% of Gemma's patches edit a strict non-empty subset of the gold functions. Gemma is not already indifferent between full and partial fixes. DPO pairs (full gold ≻ necessary-function-dropped gold) from training repos raise held-out multi-function resolve by +5 pp (GUESSED).

**Cheapest kill.**
1. CPU, ~3 h: from r2/r3 patches and `gold_funcs`, classify failed multi-function tasks as subset / disjoint / superset.
2. ~0.5 GPU-h: on the served model, compare log p(full-gold `edit_file` sequence) vs log p(partial) at the fixer's last state for ~60 pairs.

**Kill if** the subset share is <40%, or Gemma already prefers the full gold in ≥70% of pairs. **Depends on** the IFG Stage-A go.

### Third: serve-faithful adapter measurement (C4)

**Claim.** An adapter trained on NF4-of-dequantised-W4A16 loses measurable fidelity on the served W4A16 base: held-out token NLL gap ≥3%, or tool-call parse validity gap ≥2 pp. Exact-code (HQQ/GPTQ) training closes at least half of that gap.

**Cheapest kill.** Reuse the first Phase-0b adapter. Compute NLL and validity on 200 held-out turns in HF-NF4 vs vLLM-W4A16 (`prompt_logprobs`). ~1 GPU-h. **Kill (train on NF4 and stop worrying) if** the NLL gap is <1% and the validity gap is <1 pp. **Paper value:** one table that fills a stated gap (no runtime-adapter cross-format study for agent outputs).

---

## Engineering must-dos (ordered; not novel; needed regardless)

| # | Task | Cost | Why now |
|---|---|---|---|
| E1 | Retrieve r1–r3 raw traces from Kaggle outputs into `data/runs/` (gitignored); process by script only | ~1 h | Every offline test above needs them; they are not on disk |
| E2 | **Gemma tool-call round-trip check.** Render the exact prompt that ADK 1.36.1 + vLLM 0.19.1 produce for turn n+1 after an `edit_file` whose `old_string` contains `"`. Is history re-rendered as native `<|"|>` syntax or as hybrid JSON? Is streaming on? Use the tokenizer chat template on CPU; confirm on GPU only if it is ambiguous | 2–3 h CPU | If history is corrupted, the model learns in-context to emit broken calls (HF forum; 0.19.1 fixed only streaming parse bugs). Training data must be rendered identically |
| E3 | Port `use_scorer_env` + Python 3.13 fixes into `run_official_l4.py` | ~0.5 day | Local numbers were measured in the wrong environment |
| E4 | Unseen-repo dev set: 40–60 gold-verified tasks from permissive, recent repos, excluding fastapi/rich/requests and SWE-bench repos | 1–2 days CPU | No adapter can be judged without it; the LB cannot see <3 pp |
| E5 | Phase 0b training feasibility: FSDP-QLoRA, SDPA head_dim-512 memory probe, fused CE, max sequence length, tok/s, offline training wheelhouse, PEFT export (bias none, 410 modules, rank, size) + serving probe | ~1 day + 2–3 GPU-h | Blocks Stage 2 and all training |
| E6 | Render-exact data converter + filters: SWE-Lego error masking, SWE-Prime S_tool/S_redundancy/S_workflow, ordered truncation ≤16–24k, ≤3 per instance, git-hack filter, licence ledger (prefer CC-BY Qwen-teacher sets such as SWE-ZERO/Nebius; flag the Claude/GPT-teacher terms of SWE-smith/SWE-Gym to Shrey) | 1.5–2 days CPU | The mainline data |
| E7 | **Mainline SFT LoRA #1** (1–2k trajectories, r=16–32), plus a small own-state slice (format self-repair H3; SCCS if it passes) | 8–12 GPU-h/epoch + ~4 GPU-h eval | The largest expected LB lever |
| E8 | Zero-training prompt A/B (one change): "rerun tests only after a code change; do not repeat a read you already did" + a reflexive-repair line | ~3.5 GPU-h local | Cheap baseline the LoRA must beat |
| E9 | `validate_submission` + worst-case runtime for every adapter submission | ~0.5 h each | Scorer-path rule |

GPU plan to Nov 5 (≈110–120 h): E5 3 + SCCS Stages 1–2 4 + C4 1 + E7 ≈16 + E8 3.5 + three dev-set evals ≈12 + Stage 3 ≈10 ≈ **50 GPU-h**. This leaves roughly half the quota as slack for a second SFT epoch or a reroll.

---

## Search log (this session, in order)

| # | Tool | Query / target | Outcome used |
|---|---|---|---|
| Q1 | alphaXiv discover | "step-level or action-level DPO for SWE agents using negatives sampled at the agent's own failure states (malformed tool calls, repeated actions)"; kw [step-level DPO, SWE agent, negative actions, tool call] | FC-SWE, LiteSearch-VL Step-DPO, FailForge, **AED 2609.40111**, SWE-MILE, Feedback That Backfires, CPA, Invocation-Level Reliability, From Evidence to Action, AgentDrift. No SWE step-DPO |
| Q2 | alphaXiv discover | "unlikelihood training or negative examples to stop repetitive action loops in LLM agents"; kw [unlikelihood, repetition, agent loops, repeated actions] | StateWise, **Sentry 2610.02994**, residual-stream repetition, ResonAct, AgentLoop, LoopGuard, FOCUS&RePAIR, Bayesian repetition penalty, One-Neuron. No agent unlikelihood training |
| Q3 | alphaXiv PDF | Sentry 2610.02994 | Test-time playbook. SWE-bench Lite 0.20 → 0.30 (Qwen3.5-9B); progress-failure recovery on SWE only 34% |
| Q4 | alphaXiv PDF | AED 2609.40111 | Post-error actor SFT with masked error; **offline action-DPO pilot: no recovery benefit**; action-only repair +6.67 (WebShop-lite, single seed) |
| Q5 | alphaXiv discover | "LoRA fine-tuning directly on the deployed int4 quantized weights … train-serve quantization mismatch"; kw [LoRA, W4A16, GPTQ, quantization mismatch, QLoRA] | Scale-QLoRA, GPTQ-intrinsic LoRA, **GradCodeS 2608.30908**, AQLoRA, double-binary, QATFactory, LoftQ, L4Q. No runtime-adapter cross-format agent study |
| Q6 | alphaXiv discover | "does next-action likelihood or held-out loss on agent trajectories predict downstream agent success"; kw [next-action prediction, held-out loss, agent trajectories, SWE-bench] | **2609.21187**, certified selective eval, early-outcome audit, monitoring, trajectory UQ, self-retrospection, automata |
| Q7 | alphaXiv PDF | 2609.21187 | Gold-history next-turn gains do not transfer to closed loop; Gemma-3 strict tool accuracy fell after SFT |
| Q8 | alphaXiv PDF | GradCodeS 2608.30908 | Fully low-bit code-space tuning; QLoRA-4Merge −13.4; not runtime adapters, no agents |
| Q9 | alphaXiv discover | "relabeling an agent's own failed states with corrected actions from a privileged teacher (DAgger-style offline imitation) for SWE agents" | **Revisiting DAgger 2605.12913**, **PivotOPD 2609.40285**, Sentry, **HiSentinel 2609.39957**, AED, FC-SWE, Latent OPSD, **PSP 2609.29051**, SWE-MILE, FailForge |
| Q10 | alphaXiv discover | "LoRA verifier or reward model adapter sharing the policy base model to select between a small number of SWE agent patches (best-of-2)" | LatentSift, CER, Fail-Fast, CRR, SWE-MILE, SWE-Serve, cross-benchmark RL, SWE-Prime. No k=2 shared-base LoRA verifier |
| Q11 | alphaXiv PDF | Revisiting DAgger 2605.12913 | SWE DAgger +3.9/+3.6 (4B/8B); Qwen3-Coder-30B teacher; iterated batches of 512; 4×A100 |
| Q12 | alphaXiv PDF | HiSentinel 2609.39957 | 1.7B pre-execution sentinel; Qwen3-Coder 30 → 44% on SWE-V Mini; needs a per-step external model |
| Q13 | alphaXiv PDF | PivotOPD 2609.40285 | Online RL + OPD; SWE +3.2 (Nemotron); privileged self-teacher |
| Q14 | alphaXiv PDF | PSP 2609.29051 | OPSD below base (SWE-V Qwen3-4B 3.56 → 2.40); privileged information belongs in the sampler, not the loss |
| Q15 | alphaXiv discover | "training an agent so it does not repeat a just-failed tool call: recovery data where a failed call and its error stay in context …" | AED, PivotOPD, Sentry, FTB, Recova, Retry/Switch/Abstain, One Readout, SENTINEL-RL, AgentRewind, FailForge, CausalFlow |
| Q16 | WebSearch | "vLLM gemma4 tool parser fix quotes arguments chat template re-render tool_calls issue 0.19" | 0.19.1 fixed streaming tool-call JSON/delimiter bugs; re-render corruption risk persists (HF forum); vLLM issue #39468 |
| Q17 | alphaXiv discover | "separate LoRA adapters per sub-agent role (localizer, fixer, tester) for SWE agents vs one shared adapter" | MoRSE, PReCache, ExRole, SLAaaT, Skill-to-LoRA, CodeForge-MA |
| Q18 | alphaXiv discover | "fine-tuning on the model's own malformed tool calls repaired by a deterministic parser/re-serializer …" | ContractRL, Harness-R1, RESKILL, VeriHarness, AutoSaddler, Repair-not-Improvement. No SWE |
| Q19 | WebSearch | "Feedback That Backfires repeat failed tool call larger models 30B replication fine-tuning" | No ≥30B replication; Fission-GRPO (ACL 2026, online), structured reflection 2509.18847 |
| Q20 | alphaXiv discover | "training language models to suppress copying a previous erroneous output that is still in context …" | **Contextual Drag 2602.04288**, Copy Less Ground More, Blind Resampling > Self-Repair, DRY sampling, Diagnosis Before Recovery, self-correction blind spot |
| Q21 | alphaXiv discover | "do offline metrics computed on an agent's own rollout states … predict closed-loop agent performance after fine-tuning" | **No papers found** |
| Q22 | alphaXiv PDF | Contextual Drag 2602.04288 | Reset-SFT gains under erroneous context but costs −22 to −28 pp with correct context |
| Q23 | alphaXiv discover | "penalizing or labeling agent actions whose observation is identical to an earlier observation (zero information gain …)" | **No papers found** |
| Q24 | alphaXiv discover | "redundant or repeated tool calls in coding agent trajectories as a training penalty or negative" | AKBE 2605.26952, DAG refinement, RideWay, Redundant-or-Necessary benchmark 2605.29893 (title only), Fail-Fast, STAPO, TACO, TokenCast, SWE-Prime, **2609.30725** |
| Q25 | alphaXiv PDF | 2609.30725 | ReTest via inter-patch windows; SubRetrv; DevSkills cut cost up to 41.7%; training is named as future work |
| Q26 | alphaXiv PDF | AKBE 2605.26952 | Offline variant over-suppresses necessary calls; best λ is small (0.05) |
| L1 | Local | CLAUDE.md, docs/*, four notes, graph report §top pick, `results/runs/*/summary.json`, `r2 per_task.jsonl` (aggregates: repeats 102/129 tasks, malformed args 9/129) | Facts table |

**Not searched (gaps):**
- OpenReview ICLR-2027 submissions;
- Semantic Scholar venue checks;
- the unlikelihood original (Welleck 2019; cited from memory);
- the full text of the "Redundant or Necessary?" benchmark (2605.29893);
- Fission-GRPO full text;
- the actual vLLM 0.19.1 `gemma4` chat template.

Absence claims rest on alphaXiv's arXiv index and web search only.

---

# Round 4: field-level final judgement (8 Oct 2026)

Brief: judge at the level of open research gaps and design one training *architecture*. Our runs are a testbed, not the motivation.

New inputs:
- `prior_art_redteam.md`: DITTO and aTTT narrow SCCS; 2610.03984 partly anticipates partial-fix negatives; the NF4→W4A16 study is measurement-only.
- `cross_domain_ideas.md`: 16 ideas. Its top five are DART recovery injection, same-model harness distillation, surgery→LoRA, tag-conditioned SFT, and graph-weighted SFT.

New searches are Q27–Q38, listed in the log below.

## R4.1 Field-level gaps in PEFT and offline/RL-style tuning for SWE agents (after kill rounds 4a/4b)

Status: **Open** = no work found. **Narrowed** = adjacent work exists, but our exact setting is open. **Closed** = done.

| # | Gap | Evidence (search IDs) | Status after attack |
|---|---|---|---|
| **G1** | **Matched-compute offline/iterated-batch vs online RL for multi-turn SWE agents at ≥30B** | For math and single-turn code, offline now matches or beats GRPO: G2D, short GRPO warm-up then offline DPO, ≥ GRPO at ~¼ compute (2605.21266, Q30); DRIFT, offline importance-weighted SFT ≈ multi-turn RL (ICML'26, 2605.31455, Q37); REGEN, offline replay recycling ≈ MOPD (2607.19450, Q29). For SWE agents, Nebius, SWE-Gym and Agent-RLVR have no online arm (Round 1 notes); Q27 found no SWE comparison | **Narrowed: open for SWE only.** Math/code results say the offline–online gap is "data informativeness", which has never been tested on long-horizon SWE |
| **G2** | **Step-level credit assignment without online rollouts for long-horizon SWE** | Every deterministic or structural step-credit method is an *online GRPO advantage modifier*: DepGPO (read–write dependency to the verifier read set, Terminal-Bench, +3.2–10 pts over the best RL baseline; 2610.03634, Q38), VICT (verifier atoms → proof edges; 2608.28128, Q34), AgentLTL (LTL compliance reward; 2607.02599, Q33), CRR (forks; online), SWE-MILE. The offline methods (DRIFT, REGEN, G2D) use **trajectory-level** weights only; DRIFT even supervises only the terminal turn for credit reasons (Q37) | **Open at the junction.** Both halves exist separately. Joining them for SWE was not found (Q31, Q32, Q35, Q36) |
| **G3** | **Teacher-free correction at the agent's own states** | DAgger-SWE, OEC and PivotOPD need a stronger teacher. Hinted self-distillation falls below base (PSP). AED's offline action-DPO was null | Open (cross-domain note C1/C5, Round 2) |
| **G4** | **Internalising scaffold/harness behaviours that cannot ship** | Harness-Zero, HAT (SWE-QA only), HAD; offline harness-control RL 2607.05458 (SNIPPET) | **Narrowed.** Open only for same-model deterministic harness, SWE *resolution*, ≥30B |
| **G5** | **Quantised-base LoRA for agents (train on X-bit, serve on a different int4)** | ProjQ, QA-LoRA, Scale-QLoRA are same-format; vLLM docs reuse adapters across formats without measuring fidelity (red team) | Open but low-value (measurement) |
| **G6** | **Post-training ≥30B SWE agents on commodity hardware (24 GB cards, no online RL)** | FrogNano is 4B with online synthesis; ES coevolution and edge-LoRA work are small models (Q28). Every SWE RL/SFT paper uses A100/H100/H200 nodes | Open, but mostly a *constraint*, not a contribution by itself |
| **G7** | **Offline proxies on own visited states validated against closed loop** | Gold-history next-turn metrics fail (2609.21187); Q21 found nothing for own-state metrics | Open (evaluation methodology) |
| **G8** | **Usable offline negatives for agents** | AED offline action-DPO null; Agentic-DPO is non-SWE; DITTO is text-only; AKBE offline over-suppresses | Narrowed |

**Kill round 4a (gaps that are not worth a paper alone).**
- **G5:** a measurement table only.
- **G6:** a constraint, not a contribution.
- **G7:** a methodology section, not a headline.
- **G4:** HAT/Harness-Zero/HAD leave only a thin delta.

**Survivors: G2 (top), G1 (the framing that makes G2 matter), G3 and G8 (they plug into G2's negative and recovery channels).**

## R4.2 Architecture: DECAF (Dependency- and Execution-Certified Advantage Fine-tuning)

**One sentence.** An offline, iterated-batch post-training loop for SWE agents. It moves *deterministic dependency credit* (until now an online-GRPO-only device) into an *importance-weighted SFT objective* that fits 4×L4 LoRA. It resolves the verifier's read set with the **executed call graph of the fail-to-pass tests**, and it adds certified zero-information negatives and state-preserving recovery injection.

### Components (text diagram)

```
ROUND r  (r ≤ 2; SWE-Gym/DRIFT: gains saturate after ~2 refreshes)
┌─ GPU session A: rollouts (vLLM 0.19.1, W4A16 + adapter_r, local runner + tracer; scorer untouched)
│    training-task pool: permissive repos with F2P/P2P tests, repo-disjoint from dev set
│    K=4 rollouts/task at T≈0.7; log per step: call, observation, workspace diff (line-level),
│    sed/grep line spans read, behaviour log-probs of emitted tokens  (REGEN-style replay record)
├─ CPU session: certifier (deterministic, no model calls)
│    (1) verifier: F2P/P2P on final state → R ∈ {0,1} (+ F2P fraction for ties)
│    (2) coverage tracer: functions/lines executed by F2P tests on base + final commit
│        → coverage-resolved verifier read set E*   ◄── same infrastructure as IFG Stage A
│    (3) dependency graph: line-level writes (diffs), line-level reads (sed/grep spans),
│        test-run reveals; backward trace from E* (DepGPO rule, β-decay) → step credit c_t
│    (4) zero-information flag z_t: identical call, no mutating action since → provably
│        identical observation; guard class g_t: identical call AFTER a write (re-run test)
│    (5) weights: Â = class-balanced, group-normalised outcome (REGEN/DRIFT);
│        w_t = Â·f(c_t) with unit-mean step factors (DepGPO); abstain → uniform
│    (6) DART injector: state-preserving perturbations (duplicate read + identical output,
│        rejected malformed call + parser error) spliced into positive trajectories; label
│        = clean next action; rate ∝ measured failure histogram, decays per round (PC-DAgger)
└─ GPU session B: training (FSDP-QLoRA r=32, one adapter)
     L = Σ_{w_t>0} ρ̄_t·w_t·CE(a_t)                       (REGEN truncated-IS weighted SFT)
       + λ_z Σ_{z_t} DITTO-relative penalty, λ_z ≈ 0.05     (bounded; guard g_t kept as CE)
       + λ_a CE(render-exact, SWE-Prime-filtered teacher slice)  (Agentic-DPO: anchor essential)
     → adapter_{r+1}
Deployment: one PEFT LoRA (≤1 GiB at r=32+), declarative v2 agent, no harness. Shippable.
```

### Known pieces + new pieces

| Piece | Source (closeness) | Status |
|---|---|---|
| Read–write dependency credit traced from the verifier read set, redistributing group advantage | DepGPO 2610.03634 (2: same credit idea, **online**, file-level reads, terminal tasks) | VERIFIED |
| Verifier atoms → proof edges; no-op repeats get no credit | VICT 2608.28128 (1–2: online, non-SWE) | VERIFIED |
| Offline importance-weighted SFT ≈ online multi-turn RL; refresh round +0.7 | DRIFT 2605.31455 (ICML'26; 2: trajectory weights, math) | VERIFIED |
| Replay records with behaviour NLL; asymmetric truncated IS; class-balanced sampling | REGEN 2607.19450 (1: math/code/IF, 1.5B) | VERIFIED |
| Offline preference after short warm-up ≥ GRPO; the gap is informativeness | G2D 2605.21266 (1: math) | VERIFIED |
| Guarded repeat penalty | DITTO 2206.02369 (2 for this component) | VERIFIED via red team |
| State-preserving perturb-and-return; calibrated, decaying noise | DART/PC-DAgger 2609.30462 (1: robotics) | VERIFIED via cross-domain note |
| SFT anchor needed for step-level negatives | Agentic-DPO 2607.10601 | VERIFIED |
| Executed call graph of failing tests as structure | IFG (our graph report), TraceEval/RepoReasoner | VERIFIED as cited there |
| **NEW-1:** step-level dependency credit used **offline** (iterated batch, weighted SFT), not as an online advantage | — | Not found: Q31, Q32, Q35, Q36 |
| **NEW-2:** a **coverage-resolved verifier read set E\*** for SWE. DepGPO's file-level E collapses for Python test suites that import the whole package (GUESSED). Line-level reads come from our sed/grep scaffold | — | Not found (Q35) |
| **NEW-3:** certified zero-information negatives with a post-change guard, inside the same weighted objective | DITTO and SCCS lineage | Narrowed (DITTO, closeness 2) |

**Graph tie: genuine.** E\* *is* the executed call graph of the F2P tests, and the backward trace runs over a dependency graph. This shares the IFG Stage-A coverage tooling, and it gives the IFG paper a training-side use for its graph.

## R4.3 Falsifiable headline claim and reviewer comparisons

**Claim (effect sizes GUESSED).** On the same Gemma-4-31B rollouts and token budget, two DECAF rounds give the outcomes below on a repo-disjoint, scorer-env dev set.

| Outcome | DECAF vs DRIFT-style trajectory-weighted SFT | DECAF vs RFT (successes only) |
|---|---|---|
| Resolve | **≥ +3 pp** | **≥ +4 pp** |
| No-progress-repeat rate | ≥ 40% relative reduction | — |
| Tested-after-last-edit | ≥ +10 pp | — |
| Compute | ≤ 25 GPU-h total on 4×L4 | — |

**Mechanism claim (cheaper, central).** Coverage-resolved step credit concentrates weight on the steps that write F2P-executed lines and on the reads that supported them. This is measured offline by credit precision against gold-patch functions.

**What a reviewer will compare against (and our answer).**

| Baseline | Number to cite | Our arm |
|---|---|---|
| Online GRPO with dependency credit | DepGPO +3.2–10 pts over the best RL baseline (9–27B, Terminal-Bench) | Not runnable here. Report it as the upper reference and claim **offline feasibility on 4×L4**, not parity |
| Online SWE RL generally | SkyRL-14B +3.6; DeepSWE 23→42 (64 H100 × 6 days); Nebius RFT→RL +18.5 | Same: cited only |
| Offline trajectory DPO after RFT | Agent-RLVR +1.6 over RFT (72B) | Optional arm (needs pairs; AED was null) |
| Trajectory-weighted offline SFT | DRIFT (math) | **Arm A**: the primary control |
| RFT on own successes | Nebius 11.4→20.5; SWE-Gym self-RFT −6.6 | **Arm C** |
| Filtered BC | SWE-Prime 10% > 100% | Anchor slice shared by all arms (so not a separate win) |
| Step-level DPO with student negatives | Agentic-DPO ≈ GRPO (τ-bench) | Cited; it needs expert states we lack |
| Guarded repeat penalty | DITTO | Ablation: remove the λ_z term |

## R4.4 Cheapest experiment that could falsify DECAF

**Tier A: credit validity (CPU-first, ≤1.5 GPU-h). Run before any training.**
- **Data.** The 129 public tasks are a *mechanism* testbed: gold tests exist and the memorisation risk does not matter for credit validity.
  - Rollouts: existing r2 traces (if retrievable) plus one fresh K=4 batch on 32 tasks. The batch is the GPU cost (~1.5 GPU-h at ~8 concurrent agents; GUESSED).
  - Coverage of F2P tests is computed in the rebuilt scorer env (`scorer_site.tgz`, CPU).
- **Procedure.** Compute DepGPO file-level credit (E = files the verifier read) and DECAF coverage-resolved credit (E\*) for every trajectory.
- **Metrics.**
  - (i) Fraction of steps with non-zero credit.
  - (ii) Write-credit precision: credited edit steps touching gold-patch functions, on resolved trajectories.
  - (iii) Read-credit recall: whether the read that displayed the gold lines is credited.
  - (iv) Usable-credit coverage, including failed trajectories with partial F2P.
- **Kill criteria.**
  - **KA1:** file-level E already gives ≤30% credited steps with precision within 5 pp of E\*. NEW-2 is then dead, and DECAF reduces to "offline DepGPO".
  - **KA2:** usable non-uniform credit on <60% of trajectories. The fallback would dominate.
  - **KA3:** write precision <70% or read recall <60% with E\*. The credit is noise.
- **Power.** ~150–250 trajectories. A 95% CI half-width of about ±7 pp on the precision estimates is adequate.

**Tier B: method test (~8 GPU-h; only if Tier A passes and Phase 0b is done).**
- **Data.** Training: K=4 rollouts on ~60 training tasks (≈240 trajectories, ~3.5 GPU-h). Evaluation: the unseen-repo dev set E4, 40 tasks.
- **Arms (token-matched, same anchor slice):** **A** = DRIFT-style trajectory weights; **B** = DECAF (A + dependency redistribution + λ_z + DART); **C** = RFT on successes. Train ~1 GPU-h per arm at r=16, ≤12k tokens.
- **Primary metrics (process; resolve is underpowered at n=40):**
  - closed-loop no-progress-repeat rate per task;
  - tested-after-last-edit;
  - calls to first edit;
  - plus resolve and the own-state one-step metrics from Round 3.
- **Power.** Per-task paired process metrics (Wilcoxon, n=40) detect a ~0.5 SD shift at power ≈0.85. For resolve, n=40 detects only ~+15 pp, so resolve is reported, not tested.
- **Kill criteria.**
  - **KB1:** B fails to beat A on ≥2 of 3 process metrics at p<0.05.
  - **KB2:** B's resolve < A's resolve − 2 tasks.
  - **KB3:** removing the redistribution (B without f(c_t)) recovers ≥80% of B−A on all process metrics. The new piece is then not the cause.
- **GPU-h.** Tier A ≈1.5, Tier B ≈8 (3.5 rollouts + 3 train + 1.5 eval). Total ≤10, inside the ~60 GPU-h slack in Round 3's plan.

**Next step per outcome**
- **Pass (A and B):** run round 2 (one refresh, ~8 GPU-h). Fold DECAF into the mainline LoRA (E7). Paper: "Offline dependency-credited post-training of a 31B SWE agent on 4×L4", with credit-validity + three-arm table + refresh curve + cited online numbers.
- **Tier A fails at KA1 only:** keep the offline-credit piece (NEW-1) with file-level E. The graph tie and NEW-2 die; IFG is unaffected.
- **Tier A fails at KA2/KA3:** DECAF dies. Fall back to the runner-up (below) and the mainline SFT.
- **Tier B ambiguous (one of three metrics, or effects of 0.2–0.5 SD):** add a second rollout seed on the same 60 tasks (~3.5 GPU-h) before deciding. No hyperparameter fishing beyond λ_z ∈ {0, 0.05}.
- **Tier B fails:** report the negative ("offline step credit ≠ online gains at 31B LoRA"). That is still informative for G1, and it is a short paper.

## R4.5 Runner-up architecture: SCAD-R (same-model, information-symmetric harness distillation with recovery injection)

- **Mechanism.** During local data generation, the local runner (not the scorer) appends deterministic annotations computed from visible state:
  - "identical to call #k, nothing changed";
  - "you edited X, no test since";
  - "step n/40".

  The privileged information stays in the **sampler** (PSP's lesson). Train on the annotated runs' actions with the annotations stripped. This is legitimate because the annotations are information-symmetric (re-derivable from context), so PSP's leak mechanism does not apply. Add DART injection.
- **Prior art.** Harness-Zero, HAT and HAD (closeness 2), OPCD, PSP.
- **Gaps closed.** G4 + G3.
- **Cheapest falsifier.** About 1 GPU-h, one step:
  - At ~300 own NPR and post-edit states, compare next-action distributions with vs without the annotation (greedy + K=4).
  - Kill if the annotation changes the action in <30% of NPR states, or raises "run a test" after an unverified edit by <10 pp. If the harness does not change behaviour in context, there is nothing to distil.
- **If it passes:** the 30-task Stage A (~5–6 GPU-h) doubles as training data.
- **Relation to DECAF.** SCAD-R is a *sampler* add-on to DECAF's rollout session, raising positive-trajectory yield. Run it as DECAF's round-2 sampler if both survive.

## R4.6 VERIFIED vs GUESSED

| Item | Status |
|---|---|
| DepGPO, VICT and AgentLTL step credit are online-GRPO only; DepGPO reads are file-level | VERIFIED (Q33, Q34, Q38) |
| DRIFT, REGEN and G2D: offline weighted SFT/DPO ≈ or > online RL on math/code at ≤8B; trajectory-level weights | VERIFIED (Q29, Q30, Q37) |
| No SWE matched-compute offline-vs-online study; no offline step-credit SWE method | VERIFIED as absence in the alphaXiv index (Q27, Q31, Q32, Q35, Q36); OpenReview not searched |
| File-level verifier read sets are uninformative for Python repos (pytest imports the package) | GUESSED; this is Tier A KA1 |
| Line-level read tracking is possible from our sed/grep scaffold | GUESSED, plausible (v2 reads via `sed -n 'a,bp'`); verify on traces |
| r2 raw traces retrievable; 8-way concurrent rollouts give ~6× single-stream throughput on 4×L4 | GUESSED |
| Phase 0b (31B QLoRA on 4×L4) feasible at ≤12k tokens | GUESSED (unmeasured) |
| Effect sizes (+3 pp vs DRIFT-style, −40% NPR, +10 pp tested-after-edit) | GUESSED |
| The 129 public tasks are valid for *credit-validity* testing despite memorisation | GUESSED but low-risk (credit depends on traces and tests, not knowledge) |
| Gemma outputs are licence-clean for released training data | GUESSED; check the Gemma 4 terms |

## R4.7 Honest novelty verdict and paper ceiling

- **Novelty: about 1.5–2 / 3.** Every component is published:
  - dependency credit (DepGPO, 2 Oct 2026, *concurrent*);
  - offline weighted SFT (DRIFT, REGEN);
  - guarded repeat penalties (DITTO);
  - perturb-and-return (DART).

  The contribution is the junction: offline + step-level + SWE coverage-resolved credit + commodity hardware. A reviewer will say "offline DepGPO for SWE". Our rebuttal rests on NEW-2 (coverage-resolved E\*, which Tier A tests directly) and on demonstrating it at 31B on 4×L4.
- **Biggest weakness:** **no online arm.** We cannot show what fraction of online gains offline recovers (G1) on SWE. We can only cite DepGPO/SkyRL numbers from other models and benchmarks. A main-track reviewer will flag this.
- **Paper ceiling:**
  - workshop or short paper (NeurIPS/ICLR agent, efficient-ML or SE workshops) if Tiers A and B pass;
  - main track only with a matched online comparison we cannot afford;
  - Tier A plus the graph tie also gives the IFG paper a "graph used for training credit" section.
- **Leaderboard ceiling:** DECAF rides on the mainline SFT LoRA and is a few-pp lever at best. 1 task ≈0.8 pp, so the LB cannot confirm it. Dev-set process metrics are the evidence.
- **Ranking change vs Round 3:**
  - SCCS becomes DECAF's λ_z component and is no longer the headline.
  - Partial-fix negatives stay an IFG-dependent option, now explicitly anticipated by 2610.03984.
  - NF4→W4A16 is a measurement-only table.

## R4.8 Paper-rubric scoring (official criteria; the score decides the ranking)

The award goes to the 3 highest-point submissions. Criteria: Novelty (new insight, *including rigorous analysis of existing methods*), Quality (generality beyond the competition), Relevance (impact on SE and agentic learning), Verifiability, Clarity. Each pick must fit in ≤3,000 words.

Scale: 1–5 per criterion; the total is out of 25. Scores are my judgement (GUESSED) and assume the falsification experiments pass. Where a pick dies at a kill criterion, its paper becomes the negative result described in its "next step".

**Candidate Z: Zero-information actions in SWE agents, i.e. mechanism + guarded fix.** This is the Round 3 SCCS pick reframed at field level. Its question: *is in-context copying the mechanism of agent loops at 31B, and does a state-certified, guarded penalty remove it without hurting legitimate repeats?*

| Criterion | Score | One-line justification |
|---|---|---|
| Novelty | 3 | First ≥30B real-agent test of the copy effect (Feedback That Backfires stops at 1.7B). The new piece is the state-equivalence label, not the loss: DITTO is closeness 2 |
| Quality | 4 | The labelling rule (identical call, no mutating action in between) is harness- and model-agnostic. The analysis can be rerun on any public trajectory corpus (SWE-smith, Nebius, Open-SWE-Traces) and any open model on commodity GPUs |
| Relevance | 4 | Redundant actions affect 79–98% of tasks even in Claude Code and mini-SWE-agent (2609.30725). Loops are a field-wide failure, not ours alone |
| Verifiability | 5 | Labels are deterministic. The premise test is CPU plus ≤1.5 GPU-h of prompt-logprobs. Kill thresholds are pre-registered. Code and labels can be released |
| Clarity | 5 | One taxonomy table, one mechanism figure (DITTO-style curve on real agent states), one three-arm ablation. Fits easily in 3,000 words |
| **Total** | **21** | |

**Candidate D: DECAF** (R4.2, the offline dependency-credited architecture).

| Criterion | Score | One-line justification |
|---|---|---|
| Novelty | 3 | New junction (offline + step credit + coverage-resolved E\*), but DepGPO, DRIFT and REGEN each cover a half. A reviewer will say "offline DepGPO" |
| Quality | 4 | It applies to any agent with a test verifier, any model, and commodity hardware. Offline iterated-batch training is the generalisable selling point |
| Relevance | 4 | It addresses G1/G2, the field's open "online RL is too expensive" problem for SWE agents |
| Verifiability | 3 | Six interacting components and no online arm. Attributing gains needs three arms plus ablations, and the full pipeline is heavy to reproduce |
| Clarity | 2 | Hard to explain the certifier, credit, negatives, DART and the loss in ≤3,000 words without cutting the evidence |
| **Total** | **16** | |

**Candidate S: SCAD-R** (R4.5, same-model information-symmetric harness distillation).

| Criterion | Score | One-line justification |
|---|---|---|
| Novelty | 2 | Harness-Zero, HAT and HAD are closeness 2. The information-symmetric, same-model twist is the delta, and it is thin |
| Quality | 3 | It works for any declarative or "can't-ship-the-harness" deployment, but the annotations are scaffold-specific |
| Relevance | 3 | Useful for practitioners. Smaller impact on agentic-learning theory |
| Verifiability | 4 | The with/without-annotation contrast is simple and cheap (~1 GPU-h one-step test) |
| Clarity | 4 | One idea, one contrast, one table |
| **Total** | **16** | |

(For reference: partial-fix negatives ≈ 13 (N2 Q3 R3 V3 C2), gated by IFG. The NF4→W4A16 measurement ≈ 13 (N2 Q3 R2 V3 C3; it fails to rise above a workshop table).)

**Ranking by rubric**
1. **Z (21).**
2. **D (16).** It ties S on points; it wins on Quality and Relevance, the criteria the coordinator said to favour.
3. **S (16).**

This **reverses Round 4's headline.** The rubric rewards a clean, generalisable insight with deterministic verification more than a multi-component architecture.

**Recommended paper shape (≤3,000 words).**
- Headline: **Z**. Sections:
  1. a taxonomy of zero-information actions across our runs **plus one public trajectory corpus**, which gives the generality evidence;
  2. the 31B copy-mechanism probe;
  3. a guarded fix vs DITTO, plain unlikelihood and a decoding penalty (aTTT reports the decoding penalty hurts).
- **DECAF Tier A** (coverage-resolved vs file-level verifier read sets: a rigorous analysis of an existing method's property, which the rubric explicitly credits) becomes one ~400-word section if it passes.
- DECAF Tier B and SCAD-R go to "future work" unless they finish by the Nov 5 freeze.

**Cost and order.** Z's Stages 0–2 (≤4 GPU-h) come first. DECAF Tier A (≤1.5 GPU-h) runs in parallel because it shares the trace and coverage tooling. SCAD-R's one-step test (~1 GPU-h) runs only if GPU slack remains.

**One addition Z needs for the Quality score:** run the Stage-0 taxonomy (CPU only) on ≥1,000 trajectories from a permissive public corpus, e.g. Nebius SWE-rebench-OpenHands (CC-BY). This shows that the zero-information-action rate and its link to failure hold beyond Gemma and beyond our harness. Cost: ~0.5 day of conversion and analysis, 0 GPU-h.

## Search log, Round 4 (extends Q1–Q26; red-team R1–R26 and cross-domain Q1–Q18 were read, not re-run)

| # | Tool | Query / target | Outcome used |
|---|---|---|---|
| L2 | Local | `prior_art_redteam.md`, `cross_domain_ideas.md` (full) | DITTO, aTTT, 2610.03984 narrowings; C1–C16 ideas |
| Q27 | alphaXiv discover (recency) | "compute-matched comparison of offline or iterated-batch training … versus online RL … multi-turn SWE or coding agents" | Category-aware iterative expert training 2609.23377, **REGEN 2607.19450**, FC-SWE, Agentic-DPO, **G2D 2605.21266**, SWE-MILE, SWE-Prime, offline-to-online 2607.11720, Socratic-SWE, Fail-Fast, offline RL code 2605.28409. No SWE matched comparison |
| Q28 | alphaXiv discover (recency) | "post-training LLM coding or SWE agents on consumer or low-memory GPUs (24GB) … QLoRA on quantized base" | FrogNano 4B 2609.07925, FC-SWE, memory post-training, KVMem, WEFT, ES coevolution 2608.02391, edge LoRA memory. Nothing ≥30B on 24 GB |
| Q29 | alphaXiv PDF | REGEN 2607.19450 | Offline replay recycling ≈ MOPD (1.5B; math/code/IF); 4×L40S; 5× throughput; asymmetric TIS + class-balanced sampling |
| Q30 | alphaXiv PDF | G2D 2605.21266 | Short GRPO warm-up + offline DPO ≥ GRPO on MATH at ~¼ compute (7–8B, LoRA r16); the gap is informativeness |
| Q31 | alphaXiv discover (recency) | "step-level rewards for SWE agents computed deterministically from environment state (workspace diffs, file hashes, test status) rather than a learned PRM" | CRR, SWE-MILE, **VICT**, grounded checklist, **AgentLTL**, rubric-to-code, DRACO, TRCA, coverage-not-targeting. All online or rubric-based |
| Q32 | alphaXiv discover (recency) | "offline counterfactual branching … fork sandbox … step-level training labels without online RL" | CRR, belief-shift branching, CIPO, EPIG-Tree, BranPO, CREDO, headroom-drift replay, direct diversity optimisation. All online |
| Q33 | alphaXiv PDF | AgentLTL 2607.02599 | Deterministic LTL compliance score as reward (GRPO, synthetic tools); Gemma-4-31B compliance 0.789 → 0.961 under block-and-warn |
| Q34 | alphaXiv PDF | VICT 2608.28128 | Verifier atoms + proof edges, online GRPO; ALFWorld/WebShop/τ; no-op repeats get no credit |
| Q35 | alphaXiv discover | "using test coverage or the executed call graph of fail-to-pass tests to assign step-level credit to edits in SWE agent trajectories" | **DepGPO 2610.03634**, SWE-MILE, coverage-not-targeting, CRR, hacktrace, ExecCritic, 2610.03984, cross-view. No coverage-resolved credit |
| Q36 | alphaXiv discover (recency) | "offline advantage-weighted regression or reward-weighted fine-tuning with step-level weights on saved multi-turn agent rollouts … coding agents" | Offline harness-control RL 2607.05458 (SNIPPET), **DRIFT 2605.31455**, offline code post-training collapse 2609.11956 (SNIPPET), REGEN, reward-weighted FT 2506.06964, PG-SFT, LEGO-RL, TRACE 2607.13988. No offline step-credit SWE method |
| Q37 | alphaXiv PDF | DRIFT 2605.31455 | ICML'26; offline exp(R/β) weights ≈ online multi-turn RL; terminal-step-only supervision; refresh +0.7 |
| Q38 | alphaXiv PDF | DepGPO 2610.03634 | Read–write dependency credit to the verifier read set E; online GRPO; Terminal-Bench +3.2–10 over best RL baseline; file-level reads; 3.8 s CPU/trajectory; 61% of K=4 groups have zero advantage |

**Not searched:**
- OpenReview ICLR-2027 submissions (a concurrent "offline DepGPO" cannot be excluded);
- full texts of 2607.05458, 2609.11956 and TRACE 2607.13988;
- the Semantic Scholar citation graph of DepGPO/DRIFT.
