# Roadmap (reset 6 Oct 2026): notebook-style, post-training first

Scaffold iteration plateaued: v2 and v5 scored 0.08 on the LB, and v6 scored 0.00. From here, the scaffold is frozen at v2 (the best LB result, single agent, no skills). The effort moves to data → post-training → final pipeline. Every phase ends with an entry in `experiments.md`.

Compute: Kaggle only (GPU L4×4 for eval/training, about 30 h/week; CPU for data work). $50 is a reserve.

## Phase 0: Infra checks (Oct 6–9). These decide what is possible
- [ ] **LoRA serving check.** Train a tiny throwaway PEFT LoRA (rank 8, a few steps) on the bf16 or QAT weights. Confirm vLLM 0.19.1 serves it on the W4A16 base through `adapter_manifest`, and that the agent can select it with `adapter:` in agent.yaml.
  - Go → post-training is viable. No-go → re-plan.
- [ ] **Training feasibility on L4×4 (96 GB).** QLoRA on 31B: memory, tokens/s, max sequence length (target 16–24k), checkpoint export to PEFT safetensors.
- [ ] **Skills on the scorer (optional, costs 1 LB submission).** Probe = v2 agent whose first action is a skill call, and which submits an empty patch if that call fails. 0.08 means skills work; 0.00 means they don't.

## Phase 1: Data loading + EDA (Oct 7–10) → `notebooks/01_data_eda`
- Tasks:
  - repo mix
  - issue length / type (bug vs feature)
  - gold patch size (lines, files, hunks)
  - test count
  - creation dates (vs Gemma's training cutoff → memorisation risk)
- Repos: size, file counts, graph coverage.
- Difficulty strata: single-file/single-hunk vs multi-file. Which strata does v2 solve?
- Local env stability: which tasks' gold patches fail locally (exclude them from denominators).

## Phase 2: Baseline + failure analysis (Oct 9–12) → `notebooks/02_baseline_failures`
- Consolidate r1 (baseline), r2 (v2) and r3 (v5) from `results/runs/`.
  - Failure-stage funnel: found file → found lines → edited → tested → resolved.
  - Tool-call format errors.
  - Loops; budget use.
- Renamed dev set: build it, run v2 on it, and compare with the public tasks. This tests the memorisation hypothesis (expected: a large drop).

## Phase 3: Techniques one at a time (Oct 12–16)
- Evaluate on the dev set, starting from v2 (no skills): budget, verification instruction, temperature, thinking on/off, compact output formats.
- Keep only changes that show positive paired gains.

## Phase 4: Post-training data (Oct 14–22) → `notebooks/03_post_training_data`
1. **Tool-format data.** Turn every observed malformed call into a correct target call: cheap, high-value SFT data.
2. **Own successful trajectories** (rejection sampling). Use v2/v5 successes, plus extra sampled runs on the dev set at temperature 0.6–0.8.
3. **Public teacher trajectories** (SWE-Gym, SWE-smith, Nebius SWE-agent trajectories), converted to our tool schema. Check licences.
4. **Synthetic localisation data, no execution needed.** Commit history of public Python repos → (issue → gold function → call path) examples.
- Hold out repos so we can measure generalisation to unseen repos.

## Phase 5: Post-training (Oct 20 – Nov 5) → `notebooks/04_lora_training`
- **SFT LoRA #1: tool-format and workflow reliability.** Target: malformed calls → ~0, fewer loops.
- **SFT LoRA #2: rejection-sampled successful trajectories** (STaR / RFT style).
- Optional: per-role adapters (localizer vs fixer), or small-scale RL on the localisation reward.
- Evaluate each adapter on the dev set, then spend one LB submission on the best.

## Phase 6: Final pipeline + paper (Nov 5–12, then to Dec 2) → `notebooks/05_final_pipeline`
- Freeze experiments by Nov 5. Paper due Nov 12.
- Paper novelty process: failure mechanism → literature gap → one idea → ablation.
  - Current candidate: evidence-backed diagnosis briefs trained with a call-graph reward (no execution needed), evaluated on a memorisation-controlled dev set.
  - Must be checked against the literature (LocAgent, ToolTrain, CodeScout, SWE-RL, SHERLOC).
- Main track: pick 2 finals by Dec 2.
