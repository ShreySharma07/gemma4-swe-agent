# CERT-SFT design critique and fix (8 Oct 2026)

Reviewer stance: a harsh NeurIPS area chair who also runs SWE-agent infrastructure. Judged against current norms (2608.19760, 2609.02417, CAR 2606.08275, TRIM 2607.18161), not against our notebook runs.

Labels: **VERIFIED** = I read the primary page this session (abstract or card). **SIBLING** = verified in an earlier note in this repo. **SNIPPET** = search summary only. **GUESSED** = my estimate, to be measured in the first test.

## Verdict (one paragraph)

The idea survives as an **audit paper**, not as a method paper. Three findings from this pass change the design.

1. **The ground truth as proposed is partly circular with one of the audited sources.** On resolved trajectories, "necessary hunk → blame back to the step that wrote it" is almost the same computation as DepGPO's line-level write credit. The only places they can disagree are lines that **survive but are not needed** and **jointly necessary or redundant hunks**. The audit is informative only on those strata, so the headline metric must be stratified to them, and the first test must check that they are not empty.
2. **The write-side machinery has prior art.** TRIM (2607.18161) already traces trajectory edits into final-patch regions and minimises the patch with validation runs (17.9–32.9% "CodeSlop" removed). CAR (2606.08275) already proposes do()-replay plus Monte-Carlo Shapley for step credit, but only on synthetic SCMs. Novelty therefore rests on **using interventional patch credit as the yardstick for training-credit sources in SWE**, not on hunk ablation or on Shapley.
3. **Read credit cannot be interventional for public corpora.** The generating models (Claude 3.7, Qwen3-Coder-480B) cannot be resampled offline within budget, so read and exploration credit is at best a provenance heuristic. The paper must say plainly that its ground truth is **interventional for writes, provenance-level for reads, and undefined for negative information**.

Corpus decision: **SWE-smith** is the primary corpus. It has one commit per repo, venv caches already exist (SWE-MiniSandbox: 13.5 GB for 128 repos, no Docker), trajectories are MIT, and the corpus contains ≥2 generating models. **Nebius SWE-rebench-OpenHands** (CC-BY-4.0, Qwen3-Coder-480B, a different scaffold, real bugs) is the generality corpus, gated on a fidelity check. Training stage: **optional, small, and scored on a likelihood mechanism metric**, not on resolve rate. The audit stands alone. Cheapest test: 2 days of Kaggle CPU on 100 SWE-smith trajectories, with four kill criteria (§5).

---

## 1. Ground-truth validity

### 1a. Attacks on "necessary hunk via single-hunk ablation + provenance"

| Failure case | What the proposed GT does | Severity | Fix |
|---|---|---|---|
| Jointly necessary hunks (A defines a helper, B calls it, or a two-site fix) | Single-hunk revert breaks the tests for each one, so both look "necessary". This happens to be right. Disjunctive cases are where it goes wrong (below) | Low | Shapley over units handles both cases |
| Redundant hunks (A or B alone fixes the bug) | Neither is necessary, so **both get 0**. Wrong | **High** | Shapley gives 0.5 each. Banzhaf as a robustness check |
| Surviving but useless lines (CodeSlop: debug prints, defensive guards, refactors) | Correctly 0 under ablation | — | This is the stratum where GT ≠ survival credit. Make it the headline stratum (C2b) |
| Superseded attempts (edit, test fails, edit overwritten) | 0 (no surviving line) | Medium (debatable) | Separate class C2a. For SFT, zero imitation credit is defensible (Orchard TD would also give it ≤0). Report it, but not as the headline |
| Informational reads not copied into edits (reading the issue, a caller, docs) | 0, or missed | **High** | Do not call this zero. Use the provenance tier (C3) for positives and leave the rest **unlabelled** (C6), never negative |
| Negative information (ruling out file X) | Invisible | **High, unfixable offline** | Declare it out of scope. All read-side metrics are restricted to *positive-label precision*, never "credit on unlabelled = error" |
| Verification runs (pytest after the last edit) | 0: there is no write, and the run did not change the patch | High: it pushes training against testing | Deterministic class C4. Excluded from the write-tier metrics and reported separately: "what share of each source's credit lands on verification steps" |
| Reproduction runs whose traceback names the later-edited symbol | Credited only if text provenance links it | Medium | C5 = subset of C3, identified by traceback-identifier overlap with C1 writes |
| Weak F2P tests (a "lucky pass", 10.7% in AgentLens 2605.12925) | φ measures contribution to *the verifier*, not correctness | Medium | State the GT as "contribution to verified outcome". Secondary check: share of φ-mass inside gold-edited functions (in SWE-smith, gold = inverse of the bug patch) |
| Flaky tests | Spurious necessity | Medium | Rerun full and empty patches 3×, and each subset 2×. Drop a trajectory if full/empty disagree. Flag a unit if its φ changes |
| Revert produces a SyntaxError or patch-apply failure | Spurious necessity | Medium | Units = hunks merged up to function scope. If a subset fails to apply, merge the conflicting units. Count SyntaxError subsets as v=0 and report their share |
| Test-file and repro-script edits in the final patch | Trivially null (the harness resets test files; scratch files are not imported) | Inflates easy negatives | Mark as C2-trivial. Exclude from the headline stratum, otherwise any "is it in src/" rule wins |
| Bash-based writes (`sed -i`, `python - <<EOF`, `cat >`) | Blame needs the write replayed | Engineering | Replay every mutating command in the venv sandbox, `git commit` after each step, then `git blame` the final patch. Drop a trajectory if the replayed final diff ≠ the recorded `patch` |

### 1b. Precise ground-truth definition (pre-register this)

Let P be the final patch restricted to non-test source files, split into units u₁…u_k (diff hunks, merged to function scope; k>8 → group by function).

- **Value function.** v(S) = (fraction of F2P tests passing with only S applied) × 1[no P2P test newly fails]. Graded is primary; the binary all-F2P version is secondary.
- **Unit credit.** φ_u is exact Shapley for k ≤ 8 (≤256 runs; median k is small, GUESSED 2–3) and 200-permutation Monte-Carlo otherwise. Also report necessity v(P)−v(P∖u) and minimal sufficient sets (IBugDC-style divisibility).
- **Step write-credit.** g_t = Σ_u φ_u · b_{t,u}, where b_{t,u} = the share of u's lines that `git blame` attributes to step t in the replayed trajectory.
- **Step classes** (multi-level; only C1/C2b are interventional):

| Class | Definition | Label strength |
|---|---|---|
| C1 | Mutating step with g_t > ε | Interventional + |
| C2b | Mutating step whose lines survive in P, but g_t ≤ ε | **Interventional 0 (headline negative)** |
| C2a | Mutating step with no surviving lines (superseded or reverted) | Survival 0 |
| C2-trivial | Writes only to tests or unimported scratch files | Trivial 0 |
| C3 | Non-mutating step whose observation first introduces a token n-gram (≥3 identifier tokens, absent from the issue and earlier observations) that appears in a C1 write | Provenance + (heuristic) |
| C4 | Test or repro execution after the last C1 write | Verification (own class) |
| C5 | ⊂C3: an execution whose traceback identifiers appear in later C1 writes | Provenance + |
| C6 | Everything else (exploration, negative information) | **Unlabelled, not negative** |

- **Expected coverage** (GUESSED; measured in §5). Mutating steps are 20–30% of steps, so that much gets interventional labels. C3/C4/C5 positives are a further 15–25%. C6 is 45–60%. This is still more than 2608.19760's 30.5% of decision points with a defined, non-zero contrast. It must be reported as one-sided on the read side.

---

## 2. Trajectory source

| Corpus | Licence | Generators / scaffold | Env without Docker? | Fit |
|---|---|---|---|---|
| **SWE-smith tasks + `SWE-bench/SWE-smith-trajectories`** | MIT (card, VERIFIED) | `model` column has 3 values: Claude 3.7 Sonnet (VERIFIED) plus 2 others (SIBLING: 3.5 Sonnet, gpt-4o). SWE-agent. The splits `tool`/`xml`/`ticks` are renderings, not separate scaffolds (INFERRED) | **Yes.** One commit per repo, so one venv per repo. SWE-MiniSandbox caches all 128 repos in 13.5 GB vs 295 GB of Docker, with eval parity (14.8 vs 13.4, 40.1 vs 38.4) (VERIFIED). The bug patch is applied as a plain `git apply` | **Primary.** 76k rows; SIBLING says 21.5k resolved. Caveat: many bugs are single-function, so stratify by k |
| **Nebius `SWE-rebench-openhands-trajectories`** | CC-BY-4.0 (SIBLING) | Qwen3-Coder-480B, OpenHands. Real issues, 32k resolved | **Partly.** `install_config` is per instance and the card says frameworks may parse it (VERIFIED). It needs a per-instance venv, and apt-level deps will fail. Fidelity is GUESSED at 60–70% | **Generality corpus**, gated on K0. Long (avg 64 turns, ~47k tokens), so LLM-judge cost is higher |
| SWE-Gym OpenHands-SFT | MIT | gpt-4o / Claude 3.5, OpenHands | Poor: one image per task, ~6 TB, no install specs (SNIPPET, via MiniSandbox) | Skip |
| R2E-Gym SFT | **No licence on card** (SIBLING) | Claude | Docker images only (SNIPPET) | Skip |
| SWE-ZERO / SWE-Hero | CC-BY-4.0 | Qwen3-Coder-480B, OpenHands | ZERO is **execution-free**: no tests were run in the trajectories, and environments come via SWE-rebench-like images | Skip ZERO. Hero is possible later |
| SERA | Unverified | GLM, SWE-agent | **Soft-verified, with no F2P tests** | Unusable: no verifier, so no GT |
| Kwai mini-swe-agent-plus 66k | Unchecked | Unknown model, different scaffold, SWE-smith tasks | Same as SWE-smith | Optional third scaffold if the licence is OK |
| Own Gemma-31B on the 54 scorer-faithful tasks | ours | Gemma, ADK harness | Yes (subprocess + scorer env, HQ) | Small N (GUESSED ~20–60 resolved), costs GPU-h, and depends on the unrepaired `run_official_l4.py`. **Appendix only** |

**Labelling budget** (GUESSED; measured in §5). Per trajectory: replay of about 10 commands, then ~(2^k + 2 reruns) ≈ 8–12 targeted F2P runs at 5–30 s each, so ~2–4 CPU-min. A 4-vCPU Kaggle CPU session at 4-way parallelism gives ~60–100 trajectories per hour, i.e. ~700–1,200 per 12 h session. **CPU is not the bottleneck. Engineering time is**: the replay parser for each scaffold plus env setup for each repo. Realistic in 2 weeks at ~5 h/day: **300 SWE-smith (≥2 models) + 150 Nebius** (if K0 passes), restricted to trajectories with ≥2 mutating source steps.

**Generality (≥2 models/scaffolds): feasible.** Two models are available inside SWE-smith for free (same scaffold). The Nebius corpus adds a third model and a second scaffold on real bugs. If Nebius fidelity fails, generality drops to "2 models, 1 scaffold", and Quality falls by 1 point.

**Licence flag for the training stage.** Training Gemma on Claude/GPT outputs raises provider-ToS questions (SIBLING). Train only on the Nebius (Qwen, open weights) or own-Gemma slice. Using Claude trajectories for the audit only is fine.

---

## 3. Audit metrics and statistics

**Sources audited.** All are computed per step and normalised to sum to 1 per trajectory.

| Source | How to compute here |
|---|---|
| Uniform | 1/T |
| Shuffled-matched (global) | Permute each source's own weights within the trajectory |
| **Shuffled-matched (type-stratified)** | Permute within step type (write, read, run). **This is the mandatory control**: step type alone predicts most of the GT |
| **Type + recency + survival baseline** | Logistic model on {is_write, lines_survive, position/T, is_test_file}. **This is the real bar.** DepGPO ≈ this baseline plus stdout edges |
| DepGPO-offline | Reimplementation of Contrib(c) + stdout-edge read credit (§3.2–3.3 of the paper) |
| Error-turn mask (SWE-Lego/Nebius) | 0 on error-triggering turns, 1 otherwise |
| Orchard LLM-TD | Judge sees the trajectory + gold outcome, annotates V at key steps, interpolates, c_t = ΔV |
| FLARE-style backward chain | Judge traces backward from the patch and tests to the steps that caused it |
| SWE-Prime segment score | Judge scores 20-step segments against the gold patch |

Judges: Gemma-4-31B on Kaggle L4 for all trajectories (~3–4 GPU-h, GUESSED). A frontier API judge on a 150-trajectory subsample (≤$15, GUESSED pricing), to pre-empt the "your judge is weak" objection.

**Metrics**, pre-registered in this order:

1. **Headline: within-writes AUROC, C1 vs C2b** (surviving null writes), pooled. Each source is reported as Δ over the type+recency+survival baseline, i.e. the *incremental* AUROC from adding the source to that logistic model. This is the only fully interventional, non-trivial contrast.
2. **Null-mass:** the share of a source's credit that falls on C2b ∪ C2a ∪ C2-trivial. Directly interpretable as "imitation weight spent on unnecessary edits".
3. **Verification share:** credit on C4 relative to uniform. This tests the DECAF contradiction for every source.
4. **R-precision** per trajectory (k = |C1|). Mean over trajectories with |C1|≥1 and ≥1 negative.
5. **Kendall τ_b** between source and g_t on mutating steps only. Per trajectory, then mean.
6. **Read tier, secondary and flagged as heuristic:** precision of each source's top-k non-mutating steps against C3∪C5. No recall metric, because C6 is unlabelled.

**Statistics.**
- Task-clustered bootstrap (resample tasks; ≥2,000 reps), done separately per corpus and per generator.
- Holm correction across sources within metric 1 only.
- Per-trajectory vs pooled: pooled AUROC is primary, since many trajectories have ≤1 C2b step. Per-trajectory metrics are reported on the subset where both classes exist, with that subset size stated.
- Stratify by k (1 vs ≥2 units) and by bug type (SWE-smith procedural / LM-rewrite / PR-mirror / combined).

**Power** (GUESSED inputs). Assume ~300 trajectories, about 2 C1 and 1–2 C2b each, so ~600 positive / ~450 negative. SE(AUROC) ≈ 0.017. Paired difference between two sources (ρ≈0.5) gives SE ≈ 0.017, and a clustering design effect of ~1.5 gives ≈ 0.021. **Minimum detectable ΔAUROC ≈ 0.06** at 80% power, α=0.05. With 150 Nebius trajectories, MDE ≈ 0.08. That is adequate for "does any source beat the trivial baseline", but not for ranking close LLM-judge variants. Say so.

---

## 4. Training stage: needed or not?

**Decision: the audit stands alone. The rubric explicitly credits "highlighting important properties of existing methods".** Training is an optional T1 that runs only if the audit shows ≥0.06 incremental-AUROC disagreement between sources.

| Item | Design |
|---|---|
| Model | **Small Gemma (E4B-class), QLoRA**, not 31B. Phase 0b (31B training on 4×L4) is unverified, and the paper must not depend on it |
| Data | Nebius slice only (licence), ~300 resolved trajectories, truncated to 16k tokens |
| Arms (4) | (U) uniform; (S) type-stratified shuffled CERT weights; (L) best LLM-judge source; (C) CERT g_t + C3/C4 classes. **All matched on ESS = (Σw)²/Σw² per trajectory**, rescaled to identical effective token count |
| Primary outcome | **Held-out credit-targeted NLL shift**: on held-out trajectories, Δlog-lik of C1 actions minus Δlog-lik of C2b actions, relative to the base model. Needs no rollouts, gives dense paired data, and is powered at ~100 held-out trajectories |
| Secondary | Δlog-lik on C4 (verification). Resolve rate is **not** an outcome: E4B resolve is near floor, and n≈50 eval tasks give SE ≈ 6 pp vs expected effects of 1–2 pp (Orchard's +1.9 is dose-confounded) |
| Seeds | 2 per arm |
| Cost | ~6M tokens/epoch, ~1–1.5 GPU-h per arm-seed on one L4 (GUESSED), 4 arms in parallel → **~10–14 GPU-h total**, ≈ half a week's quota |
| Pre-registered contrasts | C vs S, and C vs L. Nothing else |

If T1 runs, it adds roughly +1 Novelty and +0.5 Relevance. If it is skipped, the paper is still complete.

---

## 5. Kill criteria and cheapest first test (≈2 days, Kaggle CPU only, $0)

**Procedure** (N = 100 resolved trajectories from `SWE-smith-trajectories`, `tool` split, Claude 3.7):
1. Pick 5–6 SWE-smith repos with fast test suites (<30 s for targeted F2P). Build one venv per repo at the SWE-smith commit on Kaggle CPU with internet.
2. Sample 100 resolved trajectories, stratified so that ≥50 have ≥2 mutating source steps.
3. Replay: apply the bug patch, replay each mutating command, commit after each step, check the replayed diff == the `patch` field.
4. Full and empty patch × 3 runs (flake check). Units, then exact Shapley (k≤8). Blame gives g_t and the class labels.
5. Compute the type+recency+survival baseline. Optional on day 2: a Gemma-31B-free-tier (AI Studio) Orchard-TD judge on 50 of the trajectories.

**Kill thresholds:**

| ID | Measures | Kill if | Then |
|---|---|---|---|
| K0 fidelity | replayed diff == recorded patch AND full patch passes F2P 3/3 | <80% of trajectories | Fix the replay parser once. If it still fails, switch to Nebius or kill |
| K1 non-triviality | share of trajectories with ≥1 C2b step OR ≥1 redundant/joint unit (Shapley ≠ necessity) | <15% | **Intervention ≈ survival.** DepGPO-style provenance already *is* the GT and there is nothing to audit on writes. Kill CERT and rank Z first |
| K2 label density | trajectories with both C1 and C2b | <25% | Pooled metrics only. Need N≈2× to hold power. Re-plan only if the budget allows |
| K3 saturation | type+recency+survival baseline AUROC on C1 vs C2b | ≥0.95 | No headroom for any source to differ, so the audit is uninformative. Kill |
| K4 (optional) judge | Orchard-TD incremental AUROC over baseline on the 50-trajectory subset | — (informational) | Either result is publishable: "LLM-TD adds nothing over survival" or "LLM-TD tracks interventional credit". It sets the T1 gate |

**Outcomes:**
- All pass → scale to 300 SWE-smith (2 models), then the Nebius K0 on 30 instances, then 150 Nebius; judges on the L4 (~4 GPU-h).
- K1 or K3 fail → write up the negative result (≈1 page, "survival is enough on SWE writes") inside the Z paper's analysis section; do not pursue CERT.
- K0 fails on both corpora → kill.

---

## 6. Rubric score (1–5, INFERRED/GUESSED)

| Criterion | CERT (audit, ±T1) | One-line justification |
|---|---|---|
| Novelty | 3 (4 with T1) | First interventional yardstick for SWE training-credit sources. TRIM, CAR, IBugDC and 2608.19760 each own a piece, so this is junction novelty |
| Quality / generality | 4 (3 if Nebius fails) | 2–3 generators, 2 scaffolds, synthetic + real bugs. Independent of the competition harness |
| Relevance | 4 | Directly tests the credit used by Orchard, FLARE, SWE-Prime and DepGPO. SE + agentic learning |
| Verifiability | 5 | Deterministic, CPU-only, public MIT/CC-BY data, re-runnable labels. Every claim maps to a test run |
| Clarity | 4 | One yardstick, one headline stratum, one table. The class taxonomy needs care |
| **Total** | **20 (range 18–21)** | |

Compared with Z ≈17–19, DECAF 16 and SCAD-R ≈13–14, CERT is now **first or tied-first**. It has the lowest GPU risk, since it does not depend on Phase 0b, and it gives an informative negative under K1/K3. Main risks: the scaffold replay engineering (K0), and the possibility that SWE writes are too easy (K3). Both are settled by the 2-day test.

---

## Search log

| # | Tool | Query / target | Outcome used |
|---|---|---|---|
| L1 | Local | `sanity_DECAF.md` (full), `sft_data_recipes.md` (grep: licences, sizes), `rl_and_offline_substitutes.md` (grep), `docs/HQ.md` (grep: scorer env, subprocess, 54 tasks, L4), `reports/LLM graph reasoning for code.md` (grep: IFG, IBugDC), `sanity_Z.md`/`critic_judgement.md` (scores) | Prior art set, dataset table, env facts, prior rubric scores |
| W1 | WebSearch | counterfactual step importance SWE agent trajectory ablation replay attribution 2026 | **CAR 2606.08275** (do()-replay + MC Shapley, synthetic only), SWE-Replay 2601.22129 (test-time, not attribution) |
| F1 | WebFetch | arxiv.org/abs/2606.08275 | VERIFIED: synthetic SCMs only; Who&When LLM-judge step accuracy ≈14% |
| W2 | WebSearch | SWE-bench evaluation without Docker conda local venv SWE-rebench install config | Official SWE-bench requires Docker; SWE-rebench fork reads `install_config` |
| F2 | WebFetch | HF card nebius/SWE-rebench | VERIFIED: CC-BY-4.0, `install_config`/`requirements`/`environment` fields, 21k+ instances, parse-your-own guidance |
| W3 | WebSearch | Shapley value patch hunks test outcome attribution program repair | No hunk-Shapley paper found; Hunk4J/PolyHunk adjacent |
| W4 | WebSearch | SWE-smith environment one Docker image per repository same commit | Docker per repo, pinned commit per repo |
| F3 | WebFetch | arxiv.org/abs/2504.21798 | 50k instances, 128 repos (abstract) |
| W5 | WebSearch | step-level attribution benchmark SWE-bench decisive step (Who&When, TRAIL, AgentRx) | RootSE/TrajAudit, LongRCA, Traverse, TRAJDEBUG, AgentLens, all *failure* attribution and human/LLM-labelled, not interventional |
| W6 | WebSearch | "Credit Without Ground Truth" 2608.19760 | 50 trajectories, Qwen2.5-7B, ALFWorld; 30.5% non-zero contrast; ESS rule |
| W7 | WebSearch | SWE agent edits surviving into final patch provenance credit audit | **TRIM 2607.18161** (trajectory-guided patch minimisation), Coherence Collapse 2603.24631 |
| F4 | WebFetch | arxiv.org/abs/2605.12925 (AgentLens) | VERIFIED: 10.7% lucky passes (0.5–23.2% by model) |
| F5 | WebFetch | arxiv.org/abs/2607.04419 (Agent Step Value) | PubMed QA, not SWE; not relevant beyond citation |
| F6 | WebFetch | arxiv.org/abs/2607.18161 (TRIM) | VERIFIED: minimises trajectories, CodeSlop −17.9 to −32.9%, ~½ the validation cost of delta debugging |
| W8 | WebSearch | SWE-Gym / R2E-Gym / SWE-smith without docker pip install | **SWE-MiniSandbox 2602.11210**, SWE-World 2602.03419; SWE-Gym ~6 TB one image per task |
| F7 | WebFetch | arxiv.org/html/2602.11210v1 | VERIFIED: venv + chroot caches; SWE-smith 13.5 GB vs 295 GB; eval parity with Docker; code on GitHub |
| F8 | WebFetch | HF card SWE-bench/SWE-smith-trajectories | VERIFIED: MIT, 76k rows, `resolved`/`model`(3 values)/`patch`/`instance_id` |
| W9 | WebSearch | allenai SERA trajectories licence soft verified | Soft-verified (no tests), so unusable; dataset licence unverified |

**Not checked:** the full texts of TRIM and CAR (how TRIM's edit groups map to steps); whether SWE-MiniSandbox's `chroot` works on Kaggle without CAP_SYS_CHROOT (plain venvs without chroot should suffice); the Kwai 66k licence; Kaggle CPU concurrent-session limits; the third `model` value in SWE-smith-trajectories; OpenReview for concurrent "interventional credit audit for SWE".

---

## Final judgement after red team (8 Oct 2026, kill round 2)

Input: `prior_art_redteam.md`. The new threats are Counterfactual Supervision (CS, ICLR'27 QzGGstYqbH, closeness 2.5), TRIM (2.0), 2608.22960 (2.0), CWGT-ICLR'27 and ACRE (in disagreement with each other), CEDAR, CARVE, SRFT, CRR, TraceProbe and RMPO.

### K-R1. Does CERT survive Counterfactual Supervision? **Yes, narrowed. Novelty drops from 3 to about 2.5.**

| Axis | Counterfactual Supervision | CERT (revised) | Delta: real or cosmetic? |
|---|---|---|---|
| Estimand | Policy-relative MC advantage Â = Q̂(a_t) − mean Q̂(a′). It needs 12 alternatives × 16 continuations per turn (792k continuations) | **Policy-independent necessity of the realised artefact**: Shapley over hunks, blamed to writes. About 10 deterministic CPU test runs per trajectory | **Real.** A different question ("which written tokens did the verified outcome depend on" vs "which choice raised expected return"). 2608.22960 shows replay effects are underpowered on SWE (0/190 survive FDR). Necessity has no sampling variance |
| What is audited | One evolved LLM retrospective selector | **The published offline SWE SFT credit sources**: Orchard LLM-TD, FLARE chains, DepGPO provenance, SWE-Lego/Nebius masks, SWE-Prime segments, SRFT critic masks | **Real.** No one has scored any of these against execution-derived labels (red team, absence across ~40 searches) |
| Trajectories | On-policy Qwen3.5-4B/9B, mini_swe | **Teacher corpora that SFT methods actually consume** (Claude 3.x and Qwen3-Coder-480B; SWE-agent and OpenHands) | Real, but modest |
| Training | Online PPO, single run, no ESS matching | Optional offline LoRA, ESS-matched, scored on **update fidelity** (H4QZsZe2eb: credit fidelity ≠ update fidelity) | Real only if T1 runs |
| Ground-truth cross-check | — | Necessity vs replay advantage on the same steps | **Not feasible by Nov 5** (below) |

**The "two ground truths disagree?" reframing is rejected as the headline and kept as a conditional extension.**
- A replay ground truth on our trajectories needs policy access. CS needed 792k continuations, and 2608.22960 shows that 8 replays per prefix are too few to give significant effects.
- CRR's 1.4M fork records are *announced, not released*. I found no HF card or link (F9, W10). They are also Qwen3-14B on-policy trajectories on Docker-based SWE-bench Verified/Live/rebench.
- **Conditional pre-registration:** if CRR releases its records before 25 Oct, compute necessity on their resolved realised trajectories whose SWE-rebench `install_config` builds without Docker. Report sign agreement between "is a write at a fork point necessary" and its R(τ)−R(τ̃) at matched points. This becomes an appendix, never the headline.
- No paper found that names necessity-vs-advantage disagreement as an object of study (W11; CVT-RL uses a necessity framing in RL, but no comparison). The reframing is open, but out of budget.

### Revised headline claim (pre-registered)

> On resolved SWE trajectories from ≥2 teachers (≥2 scaffolds if Nebius passes K0), deterministic patch-necessity credit identifies **surviving-but-unnecessary writes** that survival-, diff- and provenance-based credit cannot see. We measure how much imitation weight each published offline SWE credit source (Orchard-TD, FLARE, DepGPO, SWE-Lego/Nebius, SWE-Prime, SRFT) places on interventionally null writes, against a survival/diff baseline (TraceProbe-style SURVIVED) and a type-stratified shuffled control.

The read-tier provenance labels are **dropped from every metric that scores a source**. They appear only as descriptive statistics. This removes the DepGPO circularity.

### Reviewer objections 1–5 from the red-team note

| # | Objection | Answer / design change |
|---|---|---|
| 1 | "Ground truth is not credit: it misses exploration and diagnosis; reads use a heuristic; CEDAR says it reduces to the diff" | Claims are restricted to **write credit for imitation**. Reads are descriptive only. Identifiability is now a kill criterion: K2 (below) kills the study if a diff/syntax baseline recovers C1 vs C2b. The headline stratum is *surviving* writes only, so the TraceProbe SURVIVED label is constant there by construction and cannot score anything |
| 2 | "Circular with DepGPO" | DepGPO's write credit = surviving line share, which is constant across C1/C2b in the headline stratum. Its expected within-stratum AUROC is therefore about 0.5. That is the finding ("survival ≠ necessity"), not circularity. No provenance edge used in a label is used by any scored source |
| 3 | "TRIM did the instrument, CS did the audit" | Reuse TRIM's EDIT-SEQ → FILE → ACTION hierarchy and cite it as the instrument. Claims = estimand + offline sources + ESS-matched update fidelity + ACRE/CWGT adjudication (T1 only). Before day 1, read the TRIM full text (30 min) to check whether its per-step removability rate already gives the C2b rate. If it does, cite it and present our contribution as the cross-source audit |
| 4 | "Ground truth is low-variance; single-hunk patches dominate" | K1 now requires *absolute* support (≥40 C2b steps in 100 trajectories), not just a share. Report the support size first in the paper. Prior: TRIM removes 20% of lines on pre-existing files in SWE-bench Verified (SNIPPET), so C2b is probably non-empty. SWE-smith bugs are simpler than Verified, which is the risk |
| 5 | "Necessity is the wrong target for SFT/RL" | Argue explicitly that offline SFT decides *which tokens to imitate*, and that imitating null edits is what produces CodeSlop. The claim is limited to SFT weighting, not RL advantage. T1's outcome is reframed as update fidelity (does source-X weighting move held-out likelihood toward C1 over C2b at matched ESS?). Behavioural gain is not claimed by Nov 5 |

### Revised first test (2 days, Kaggle CPU, $0)

Same procedure as §5, plus two 30-min pre-steps:
- (i) read the TRIM full text for per-step removability;
- (ii) check whether the CS/CRR data has been released.

The day-2 judge check is now **required**: 50 trajectories, Orchard-TD-style prompt, Gemma-4-31B on the AI Studio free tier.

| ID | Measures | Kill if | Then |
|---|---|---|---|
| K0 fidelity | replayed diff == recorded patch AND full patch passes F2P 3/3 | <80% | Fix the parser once, then try Nebius. If both fail, kill |
| K1 support | among the 100: C2b steps (surviving, Shapley ≤ ε) plus non-additive units (Shapley ≠ necessity) | <40 C2b steps OR <15% of trajectories with ≥1 | Intervention ≈ survival on SWE. Kill |
| K2 identifiability (CEDAR) | AUROC of a non-interventional baseline on C1 vs C2b: hunk size, new-vs-existing file, comment/print/log-only lines, edit position, file = modal edited file | ≥0.90 | "Necessity reduces to a diff rule." Kill the audit; keep the rule as a one-paragraph finding |
| K3 headroom | the same judge's incremental AUROC over K2's baseline (50 trajectories) | Informational | Either outcome is reportable. If ≥0.9, the headline becomes "LLM credit already tracks necessity" (positive). If ≈ shuffled, it becomes "LLM credit spends X% on null writes" |

### Rubric re-score (INFERRED/GUESSED)

| Criterion | Before red team | After | One line |
|---|---|---|---|
| Novelty | 3 | **2.5** | CS (SWE audit vs execution gold) and TRIM (instrument) pre-empt the genre. The delta is estimand + offline sources |
| Quality | 4 | 4 (3 if Nebius fails) | Unchanged |
| Relevance | 4 | 4 | Audits the credit sources people actually use for SWE SFT |
| Verifiability | 5 | 5 | Deterministic, CPU, public data |
| Clarity | 4 | 4 | Narrower claim, simpler story |
| **Total** | 20 | **≈19.5 (range 17–20)** | **Ties Z (≈17–19), slightly ahead.** DECAF 16, SCAD-R ≈13–14 |

### Decision

- **GO for the 2-day CPU first test.**
  - Time-box: ≤10 working hours, 0 GPU-h, $0. Stop at the end of day 2 whatever the state.
  - Pre-steps: TRIM full text, plus a check for CS/CRR data release.
  - Do not build any judge beyond the 50-trajectory free-tier check, and do not start T1, until K0–K2 pass.
- **If all pass:** scale to 300 SWE-smith (2 models), then the Nebius K0 on 30 instances, then 150 Nebius. Judges for all sources on the L4 (~4 GPU-h). Start writing by 25 Oct. T1 runs only if the sources disagree by ≥0.06 incremental AUROC and Phase 0b-independent small-Gemma QLoRA is working.
- **If CERT dies (K0 on both corpora, K1, or K2):** fall back to **Z** (zero-information actions) as the paper.
  - If K0 passed, fold CERT's measured null-write rate (surviving-but-unnecessary edits per trajectory, by teacher) into Z's analysis section as a second waste type ("zero-necessity writes" beside "zero-information reads/repeats"). That keeps the 2 days from being sunk.
  - If K2 alone fails, the diff-rule result is a CEDAR-style SWE finding worth one paragraph in Z.

### Search log (extension)

| # | Tool | Query / target | Outcome used |
|---|---|---|---|
| L2 | Local | `prior_art_redteam.md` (full) | CS, TRIM, 2608.22960, CWGT/ACRE, CEDAR, CARVE, SRFT, CRR, TraceProbe, RMPO; objections 1–5 |
| W10 | WebSearch | CRR 2609.33875 dataset release huggingface counterfactual tuples | No release found; only a secondary write-up |
| F9 | WebFetch | alphaxiv.org/abs/2609.33875 | Qwen3-14B; SWE-bench Verified/Live/rebench; up to 4 fork points (high-entropy, state-changing); **no release link or licence** |
| W11 | WebSearch | necessity-based vs counterfactual advantage step credit disagreement, "but-for" causation, LLM agents | No paper compares the two estimands. CCPO, C3, CRAFT (advantage-style) and CVT-RL (necessity framing, RL) are adjacent |
| F10 | WebFetch | alphaxiv.org/abs/2607.06184 (TraceProbe) | Deterministic SURVIVED/REVERTED/JUSTIFIED labels, 2,500 trajectories, 5 configs. Appears to run no tests or ablations, and is not used for credit. Off-anchor fraction not reported numerically |

**Not checked (round 2):** the TRIM full text (scheduled as pre-step i), whether the CS gold corpus has been released, and the CVT-RL details.
