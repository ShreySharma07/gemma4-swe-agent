# Independent training methodology review — RS2/T1/LP1/LP2/G1

**Audit date:** 2 November 2026 (IST). **Evidence cutoff:** local `main`, `8f8dd657467709ab27eb76eae43e6408986ecaa6`. No fetch was performed: this is the latest locally verified state, not a claim that remote main cannot be newer. The working tree initially had only the pre-existing untracked `data/runs/atul_r2_s1/` evidence. Existing sources, manuscript, results, datasets, Git index/history and Kaggle resources were not modified. No notebook, inference, training, benchmark replay or leaderboard submission was launched. Only new Markdown reports are repository additions.

**Evidence labels:** **VERIFIED** = independently inspected code or recomputed from available records, not a rerun of the experiment; **REPORTED** = documented result whose underlying measurement was not reproduced; **INFERENCE** = interpretation supported but not uniquely established; **HYPOTHESIS** = proposal requiring testing; **BLOCKED** = evidence/access insufficient. An arithmetic verification does not turn a recorded test classification into a proven causal diagnosis. Full repository paths and the shorthand path aliases below refer to the cutoff commit; local-only evidence paths are not public download links. No private issue bodies, fixture contents, patches, traces or credentials are reproduced.

## 1. Exact experimental state and verdict

**INFERENCE — The pipeline has made real engineering progress, but behavioral benefit remains unproven. Another generic two-epoch training run is not justified by the current loss curves.** First audit labels and splits, preserve the exports, and consider evaluating the existing lower-dose checkpoint against a frozen base on fresh behavioral tasks. This is a proposal, not a launched experiment or authorization.

| Experiment | Result and independent status | Evidence |
|---|---|---|
| R2 | **VERIFIED:** 301+300 disjoint tasks, 176+174=350/601 resolved (58.24%). Combined singles 278/370, pairs 64/174, triples 8/57. | `results/runs/r2_s{0,1}/task_results.jsonl` and s1 provenance/audit metrics. |
| RS1 | **REPORTED:** 350 successful episodes, 261 core; 295/350 exceed 8k tokens; median 13,071.5 tokens; 925 windows. Old test-after-edit labels are unreliable. | `results/runs/rs1/rftset_report.json`; `kaggle/rftset/rft_core.py:59–107,174–200`. |
| RS2 v2 | **REPORTED counts, with implemented rules inspected:** 350 resolved → 344 eligible → 203 train, 40 validation, 101 capped out. Train: 153 singles, 44 pairs, 6 triples; 79 recognized pytest-after-last-edit episodes. **BLOCKED:** actual NPZ and episode manifest unavailable locally. | `results/runs/rs2/rs2_report.json`; `rs2_core.py:21–29,137–203`. |
| PA1 | **REPORTED hardware probe:** real-token forward-loss differences .00117 and .00004; random-token truncated/full gradient cosine .0932, norm ratio .1256. | `results/runs/pa1/prefix_probe_l4_v3.json`. Forward agreement is not gradient agreement. |
| LC1 | **REPORTED hardware controls:** v7 StreamCE loss difference 2e−7, gradient cosine .999993. Attention/HF cosine .0187, while HF+ulp/HF cosines .0048 and −.0032 on one 4k sequence. | `results/runs/lc1/l4_v7_noise_control.json`, earlier v3/v6 and CPU reports. |
| T1 | **VERIFIED recomputation from recorded metrics:** 50 numbered optimizer steps; 4,954,947 processed tokens; validation .1011 → .0669 → .0666 (−34.12%); mean train loss .075612 → .049356. Median 344.5 tokens/s; peak 17.47 GiB; maximum preclip gradient norm 18.355. | `results/runs/t1/l4_v1_summary.json`; two 410-module/820-tensor exports reported. |
| LP1 | **VERIFIED arithmetic:** W4A16 base .08663, raw .05075 (−41.42%), copied K→V .05092 (−41.22%). Copy worse on 29/40 episodes; exact sign p=.0064266. | `results/runs/lp1/l4_v1_report.json`. These are teacher-forced successful validation episodes, not repair rollouts. |
| LP2 | **REPORTED measured losses:** NF4 full .06661, V-base .06637, full-again .06661, base .10109. Removing the global V delta improves loss about .36%. | `results/runs/lp2/l4_v1.json`; zero-K control in `smoke_v1.json`. |
| G1 | **VERIFIED:** base 40/95, raw LoRA 42/95; both 36, new 6, lost 4, neither 49; exact McNemar p=.75390625. Overflows 12 versus 18, with 8 shared; overflow discordance p=.1795654. | `results/runs/g1/l4_v1_report.json`; both 95-row arms and paired IDs. |
| v18 leaderboard | **REPORTED:** submitted November 2 as ref 57892829. **BLOCKED/PENDING:** actual score. | `docs/HQ.md:3–6`; no independently verified score receipt. |

**VERIFIED — Training-unit reconciliation:** RS2's 590 training and 143 validation chunks are rendering artifacts, not T1's training unit. `trainkit/data.py:15–18` reads `ep_starts` and float weights, ignoring chunk indices for whole-episode training. T1 uses full episodes, maximum length 26,831 tokens, and zero overlength drops. It trains 200 of 203 episodes per epoch, dropping a random remainder, for 25 steps per epoch. `worker.py:169–171` invokes the default prefix=0 full loss. It does not use PA1's frozen-prefix truncated gradient. Describing T1 as 733 independent chunk examples would be incorrect.

## 2. Dataset quality and selection

### 2.1 What 203 episodes can support

**VERIFIED:** 153/203 training episodes are single-site (75.37%); 50/203 are multi-site (24.63%), including only six triples. Boltons and sqlparse supply 100/203 episodes (49.26%); adding more-itertools gives 133/203 (65.52%). There are nine training libraries, with only two inflection episodes. These counts are recorded aggregates, not an independent reconstruction from raw RS2 tokens.

**INFERENCE:** 203 successful trajectories suffice to demonstrate optimization and can teach syntax or common actions. They do not establish that general repair reasoning, verification, or coupled multi-site coordination was learned. Mutants can share files, APIs, tests and lexical templates; millions of tokens are not millions of independent decisions. The data are conditioned on the base policy already succeeding. They may retain incorrect intermediate actions, weak self-reproducers, redundant checks and lucky final outcomes.

**REPORTED:** train call mix is exploration 56.5%, run 21.9%, edit 7.5%, submit 7.3%, pytest 2.7%, cleanup 1.9%, scratch write 1.6%. Recognized pytest-after-last-edit appears in 79/203 episodes (38.92%). **INFERENCE:** a test invocation is neither a successful test nor evidence that the final source state was checked. Call frequency is not optimized token-loss mass; increasing a 2.7% action share arbitrarily is not a scientific objective.

**INFERENCE:** synthetic-to-real shift is substantial. Factory tasks often combine independent mutations; real A5 co-changes frequently share identifiers and interface contracts. Rewritten issues may retain function clues and template style. More examples of easy independent single-site successes will not automatically teach coordinated repairs. Cleaner issue descriptions and harder task construction are separate interventions and should be tested separately.

### 2.2 Holdout design versus a genuinely untouched test

**VERIFIED in source:** `rs2_core.py:21,149–163` holds out all toolz/pathspec episodes from gradient training. G1 evaluates all 95 tasks from these same libraries (`lora_gate.py:27,38–60`). The implemented design avoids direct training on those libraries. **BLOCKED:** raw `rs2.npz` and `episodes.json` are absent locally, so the actual deployed token-level split, exact training IDs, snapshot-family duplication and pretraining contamination cannot be certified.

**VERIFIED:** RS1's per-episode rows identify 40 resolved toolz/pathspec episodes without drift or sub-agent use, and all 40 appear in G1. **INFERENCE:** they are the RS2 validation candidates implied by the implemented eligibility rules. Actual RS2 episode IDs remain **BLOCKED** without its manifest/NPZ; the matching count is not proof of binary-level identity.

**VERIFIED post-hoc arithmetic on those candidate IDs:** G1 base/LoRA results are **36/40 versus 35/40**. On the remaining 55, they are **4/55 versus 7/55**, with four new wins and one loss (exact McNemar p=.375). These post-hoc subsets are descriptive, not confirmatory tests. The large difficulty difference is expected when validation episodes were selected from prior successes.

**INFERENCE:** G1 is held out from gradient training, but is not an untouched test after T1 checkpoint decisions and LP1/LP2 serving choices used the same 40 episodes/libraries. This is selection and adaptive evaluation, not an accusation of direct train/gate leakage. Future variants should use these libraries as development data and reserve fresh task/repository families for confirmation. New-60 has also repeatedly guided scaffold changes and is not pristine.

## 3. Weighting, masks and trajectory fidelity

**VERIFIED improvements:** RS2 retains float token weights (`data.py:32–45`) and keeps masked actions in context. It masks superseded submits, some cleanup, repeated calls without an intervening edit, stray-file actions and stray-carrying submits. Key edit/pytest/submit steps receive weight 1, other steps .5; multi-site first-edit-through-submit steps receive ×3; pytest-after episodes ×2; stray episodes ×.5; final step weight capped at 3 (`rs2_core.py:59–89`). Library caps prioritize pytest and multi-site examples. The earlier v1 flat-weight/span bug is documented as fixed; cap 30→50 was a post-hoc data change.

**VERIFIED remaining algorithmic risks, distinct from unmeasured behavioral impact:**

1. **Span mismatch restores flat positive weights.** At `rs2_core.py:176–181`, a mapping mismatch replaces all span weights with `W_OTHER * ep_weight`. The report records 11/243 episodes (4.53%). This can restore supervision on actions the intended masks would remove. **BLOCKED:** affected IDs, harmful actions and effective loss share cannot be recovered without the manifest/tokens. Do not claim that all 11 actually contain harmful masked actions. Quarantine or explicitly validate them before future training.
2. **Mixed edit/test commands can be mistagged.** `tag_step():47–53` checks pytest before mutation. A safe synthetic command that writes a source file and then runs pytest is labeled `pytest`, not `edit`. It may therefore fail to reset last-edit chronology or repeat masks. Regex matches also do not establish that a write succeeded; denied/failed commands remain intent.
3. **Only the first tool call is weighted per assistant message.** `step_weights():61–63` assumes one call, while `rft_core.py:92–107` can preserve multiple calls. **BLOCKED:** frequency in the actual 243 episodes is unknown. Assert one-call messages or implement and audit call-level alignment before claiming correctness.
4. **Scratch labels can be wrong.** Filename and extension heuristics can mark legitimate additions as stray. A command containing pytest can fail collection or merely quote it. Episode-level boosts may reward verification rituals rather than successful verification of the last actual source mutation.
5. **Many changes are bundled.** Masking, weighting, capping, context rendering and data filtering changed together. No ablation identifies which component helped or harmed behavior.

**HYPOTHESIS:** first audit all 11 mismatches and report effective weighted loss by thinking, exploration, edit, test and submit. `tokens_by_tag` counts rendered positions before final weighting and includes both splits; it is not a direct decomposition of the optimized training loss. Preserve wrong intermediate context where later actions depend on it, but do not reward demonstrably incorrect actions indiscriminately.

## 4. Model and optimizer assessment

### 4.1 Configuration and global normalization

**VERIFIED from code and reports:** the base is the unquantized QAT checkpoint quantized to NF4 with double quantization and bf16 compute (`model.py:20–32`). LoRA rank 16, alpha 32, dropout 0 and no bias target q/k/v/o and gate/up/down projections (`config.py:13,25–29`, `model.py:35–43`). The report records 410 linears and 122,429,440 trainable parameters. Batch size is eight episodes: four ranks × accumulation two. Learning rate is 1e−4, five warmup steps then cosine; clipping 1.0, weight decay 0, seed 0. Two epochs produce 50 optimizer steps, with exports at step 25 and final.

**VERIFIED mathematical consistency:** `data.py:65–68` computes total label weight across every rank and micro-step. `worker.py:167–176` divides each sequence's weighted CE by that global norm. `fix_lora.py:44–86` sums gradients across ranks, computes a global clipping norm, updates fp32 master shards and gathers bf16 model weights. No additional division by world size or accumulation count is needed. The earlier per-rank normalization and boolean weight-collapse problems are fixed in this code; they should not be presented as current T1 defects.

**INFERENCE:** fp32 masters prevent loss of small optimizer updates, but local gradients still accumulate in bf16 before being upcast for reduction. This is not fp32 gradient accumulation. T1's finite recorded losses/norms do not independently prove all custom backward paths exact. Large reported norms are preclip values, not evidence clipping failed. Peak 17.47 GiB slightly misses the ≤17 GiB prediction but fits the L4 memory envelope.

### 4.2 Full-context and memory engineering

**VERIFIED code:** whole episodes are never head-truncated; episodes over 28,672 tokens are excluded. T1 excluded none for length. The frozen tied embedding/head moves to host RAM; weighted softcapped CE streams over vocabulary tiles (`fix_head.py:19–101`). MLP and norms use checkpointing; sliding attention uses local masks and global attention is grouped by KV head; unused full-length KV storage is disabled (`fix_attention.py:31–105`). Long activation checkpoints use unpinned host offload (`worker.py:115–120`) after earlier pinned-memory pressure; the head embedding remains pinned for transfer.

**REPORTED hardware validation:** LC1 attention unit gradients agree with fp32 references at about .99999 cosine. v7's streamed CE has nonzero trained tokens and .999993 gradient cosine. An earlier v3 CE result had zero trained tokens and NaNs, so it was vacuous; later controls replaced it. These records were inspected, not rerun.

**INFERENCE — Numerical equivalence is narrower than “exact training.”** HF-versus-HF-plus-ulp gradients can be nearly orthogonal on one 4k sequence, as can the custom attention path. This weakens a naive single-sequence bf16 gradient-direction acceptance criterion. It does not prove that all implementations have equivalent expected gradients or training dynamics. T1 demonstrates optimization, not universal numerical parity. Describe it as mathematically full backpropagation with recorded agreement tests and rounding sensitivity. Multi-sequence aggregate gradient checks on deterministic small fp32 models and mask boundaries would be valuable CPU validation work; this audit did not train a model.

**VERIFIED code-level robustness issue, not an observed T1 failure:** `launcher.py:18–27` consumes stdout until EOF before `proc.wait(timeout=...)`; that timeout does not itself bound a process keeping stdout open. T1 completed with rc=0 and 50 steps. This does not invalidate T1 or justify relaunch; it is a future process-deadline review item, with no source change authorized here.

### 4.3 Dose and checkpoint selection

**VERIFIED:** epoch 1 already reduces validation loss 33.83%. Epoch 2 changes .0669 to .0666, only .448% relative, while mean training loss falls 34.73%. **INFERENCE:** additional memorization is plausible, not proven; validation saturation may also reflect objective or distribution limits. Neither “a third epoch cannot help” nor “train longer because loss still falls” follows.

**HYPOTHESIS:** a lower-dose checkpoint might retain the useful distribution shift with less policy drift. Step 25 already exists. Behavioral superiority cannot be inferred from the .0003 loss difference; no step-25 served loss or behavioral comparison is available. Once a behavioral set selects checkpoints, it must be treated as development data.

## 5. Serving compatibility

**VERIFIED:** LP1 computes weighted next-token likelihood using vLLM 0.19.1 prompt logprobs on the W4A16 serving base (`lora_probe.py:29–59`). Its offline engine has prefix caching off, 1024-token prefill chunks and one sequence, rather than scorer rollout settings. It is a useful serving-likelihood check, not an end-to-end behavioral gate.

**VERIFIED:** the raw adapter preserves a 41.42% likelihood improvement on these 40 episodes, versus 34.12% on the NF4 training path. Catastrophic likelihood-transfer failure is not supported. **INFERENCE:** this does not prove NF4/W4A16 mismatch harmless on failed-policy states, rare actions or different contexts. A lower base CE does not by itself identify fewer rounding errors as its cause; engine paths and quantization both differ.

**VERIFIED code and ablations:** in ten global layers, training uses K/V sharing such that k_proj LoRA affects V. Serving's packed V slot can lack that delta unless copied (`kvdup.py:1–9,45–65`). LP1 raw .05075 beats copy .05092 on 29/40 episodes; LP2 V-base .06637 beats full .06661 on NF4, with full-again matching full. Copying K→V is therefore not an obvious next fix. The cause remains unresolved; raw serving is an empirical choice, not exact train/serve functional parity. `export.py:21–22` now documents raw export and v18's main agent declares `adapter: rs2_lora`.

**REPORTED:** final exports have 820 tensors, about 245 MB and valid shapes; HQ says the submitted package is byte-identical to G1's and gives adapter hash prefix 2b22c3f3. **BLOCKED:** local binaries and full raw rank logs are unavailable for independent shape, hash, all-rank NaN or submitted-package verification. Code and committed summaries do not substitute for binary validation.

## 6. Loss versus behavior: paired diagnosis

**VERIFIED:** G1 has 95 matched IDs, both solved 36, LoRA-only six, base-only four, neither 49. Exact McNemar p=.75390625 establishes neither improvement nor equivalence. A descriptive paired task bootstrap (seed 20261102, 10,000 draws) yields a net-gain interval **−4.21 to +8.42 percentage points** around +2.11 pp. It does not account for all mutant-family/library dependence, and only two libraries were tested.

| Changed task | Outcome base→LoRA | What can be concluded from records |
|---|---|---|
| `f1_pathspec_m24` | fail→pass | Base overflow; new success, not proof of better local logic. |
| `f1_toolz_001` | fail→pass | Neither arm has a fatal error; patch cause unavailable. |
| `f1_toolz_m05` | fail→pass | Near call cap: 27 versus 28 calls. |
| `f1_pathspec_m11` | fail→pass | Base timeout; LoRA duration 729.49s but resolved. Duration includes differing overhead; do not infer identical hard cutoffs. |
| `f1_pathspec_m21` | fail→pass | No fatal error; calls 21→28. |
| `f1_toolz_m00` | fail→pass | No fatal error; calls 24→23. |
| `f1_toolz_000` | pass→fail | LoRA timeout; calls 19→24. |
| `f1_pathspec_010` | pass→fail | LoRA overflow. |
| `f1_pathspec_m14` | pass→fail | LoRA overflow. |
| `f1_pathspec_003` | pass→fail | No fatal error; 28 calls in both; patch/test cause unavailable. |

**VERIFIED:** singles 26→24/35, pairs 12→15/43, triples 2→3/17; pathspec 19→19/47, toolz 21→23/48. These are small post-hoc strata, not proof of a multi-site training effect. Median calls fall 24→22, but median time rises 328.39→349.37s (+6.39%), time-cap proxies 2→4, empty patches 8→9 and error records 15→22. Call-cap proxies 28→22 are not actual BudgetExceeded denial events.

### 6.1 Context ceiling and competing explanations

**VERIFIED:** v14/v18 reserve 8192 output tokens from a 32768 context, leaving **24576 prompt tokens**. G1 errors explicitly reject at least 24577+8192=32769. HQ's “24572 = 32768−8192” is literally four tokens off; 24572 is near the ceiling. A large returned tool output can make the next rejected prompt absent from successful-request metrics.

**VERIFIED:** overflows are 12 versus 18 with eight shared; discordances 4 versus 10 give exact p=.179565. Two of four regressions are LoRA overflows, but one gain recovers a base overflow. **INFERENCE:** a shared structural ceiling and policy-induced growth can coexist. Nonsignificance is not evidence that the excess is “mostly noise,” nor that the adapter cannot affect overflow risk.

**REPORTED:** HQ's 190-trace diagnosis gives paired peak prompt +239 tokens, completion −16, output +289 characters and steps −1; sub-agent unused. **BLOCKED:** G1 traces and `ovf_rows.json` are absent locally, so thinking, returned tests, actual source mutations, repeat calls and growth metrics cannot be independently reproduced. `analysis/context_overflow.py` uses serialized observation-character proxies and message characters, not a clean thinking-token decomposition. Do not conclude “LoRA thinks longer” from available records.

Ranked explanations:

1. **INFERENCE:** objective and selection mismatch is strongly supported. Teacher-forced likelihood of successful trajectories rewards predictable formatting, thinking and common actions; it does not directly optimize decisions at failed states.
2. **INFERENCE:** synthetic-to-real and coupled-repair shift is substantial, with uneven libraries and very few successful triples.
3. **INFERENCE:** checkpoint dose could matter because validation saturates, but a behavioral checkpoint comparison is missing.
4. **HYPOTHESIS:** mask fallback and tags may teach harmful intermediate actions. Code gaps are verified; their impact is not.
5. **INFERENCE:** catastrophic adapter loading/likelihood mismatch is unlikely on G1; subtle state-dependent parity remains open.
6. **HYPOTHESIS:** larger fresh samples might reveal a modest benefit or regression. G1 is small and calibrated, so the loss gain cannot decide which.

## 7. Three next strategies, ranked by scientific value

All costs are **rough estimates of Kaggle L4×4 quota-hours**, extrapolated from T1 (~9.6 for two epochs) and G1 (~7.4 for 190 task runs). They are not bills, reserved quota or guaranteed upper bounds. Every experiment below is a proposal requiring separate approval; no notebook is generated.

### 7.1 Priority 1: outcome-grounded verification and corrective targets

**HYPOTHESIS:** supervision at a failed-test decision point, followed by a corrected source edit and a returned passing test, improves repair more than imitating every intermediate action in successful runs. A4/A12 and RS2's mask gaps motivate this, but do not prove it.

**Dataset changes:** audit existing successes and failures first; separate returned tests from denied calls, collection failures and weak self-reproducers. Quarantine span mismatches and review legitimate stray files. Keep full prefix context, mask known wrong/superseded actions, and preserve context that later steps reference. Corrective targets not already available require separately authorized recovery/branch rollouts and independent tests. Never fabricate them from gold or train on held-out fixtures.

**Method and controls:** low-dose full-context SFT, current RS2 versus one changed outcome-validated mask/weight policy, matched target-token budget. Include base and the existing step-25 checkpoint; decompose edit/test/submit/thinking losses. Preference training is a later option only with validated pairs from comparable states; an episode's pass/fail label alone is not sufficient preference supervision.

**Evaluation:** fresh real issues plus held-out coupled synthetic families. Primary: paired resolved rate; secondary: successful test after last actual source edit, overflows, denials, empties and time. **Proposed gate:** ≥5 net wins/95 and exact p<.1 as a screen, then fresh confirmation at p<.05; no >2 pp overflow increase or >15% median slowdown. Fail if only likelihood/ritual frequency improves. Report CIs regardless of gate.

**Estimated compute:** CPU audit 0 GPU; one training epoch ~5–6 quota-hours; 95-task paired gate ~7–9. New corrective rollouts would add ~10–25+ depending on coverage, separate from this estimate. **Risks:** verification rituals, data leakage, expensive tests, action miscredit and overfitting success. Do not purchase new rollout data until the CPU audit establishes useful existing contrastive cases.

### 7.2 Priority 2: lower-dose checkpoint, then context-budget behavior

**HYPOTHESIS:** epoch 1 retains useful learning with less drift; separate context-aware action/output discipline could protect the 24576 ceiling. These are two factors, not one bundled fix.

**Dataset and method:** first use the already-trained base/step-25/final models with identical prompts, sampling and reservation. No retraining is needed. Freeze the checkpoint choice before testing a separate budget policy. Only later consider low-dose SFT on genuine late-context action decisions; do not teach unconditional early submission or treat failed tests as a signal to stop repairing.

**Controls:** dose first; bounded output or budget policy second with checkpoint fixed. Do not reduce the 8192 reservation under thinking without verifying applicable constraints and valid response behavior. Earlier forced-early-edit bundles failed; they are not support for repeating an unconditional rule.

**Evaluation and gate:** fresh paired tasks under scorer concurrency 1/8-minute settings, with factory concurrency 6/12-minute results separate. Pre-register a noninferiority margin and CI for resolves plus overflow/time improvement; p>.05 is not equivalence. A small budget-policy screen could require fewer overflows and no more than one net lost resolve/60, followed by larger confirmation.

**Estimated compute:** zero training for checkpoint comparison; three arms×95 ~11–14 quota-hours, two arms ~7–9. A later one-epoch policy ablation adds ~5–6 only if justified. **Risks:** development-set overfitting, premature submission, loss of useful reasoning and conflating checkpoint choice with prompt changes.

### 7.3 Priority 3: coupled multi-site and issue-shift coverage

**HYPOTHESIS:** success on repairs with shared identifiers/interfaces transfers better than adding independent single-line mutation demonstrations. R2 triples 8/57 and only six RS2 training triples reveal a gap, not a proven remedy.

**Dataset:** balance repositories and mutation families; obtain jointly necessary coupled changes and separately vary terse/natural/feature-style issues. Verify gold/empty/necessity for task validity without leaking hidden fixtures or gold text into training prompts. Hold out snapshot and fault families; reuse existing authorized evidence first.

**Method and controls:** fixed lower-dose SFT with equal weighted target tokens: current stratified data versus same-size coupled enrichment. Separate data enrichment from weight changes and issue rewriting. Do not also raise calls, change serving settings and increase epochs.

**Evaluation and gate:** fresh real issues and coupled tasks, stratified by size/site count/repository. A proposed confirmatory bar is ≥10 pp coupled-task gain with family-clustered CI excluding zero and no >3 pp single-site regression. No threshold is a forecast of success.

**Estimated compute:** new rollouts ~15–30+ quota-hours, training ~5–8, paired gate ~7–10. This may exceed a weekly account budget. **Risks:** synthetic shortcuts, family leakage, regression on singles and deadline distraction. Stronger as future work than a rushed addition to the current paper.

## 8. Recommended next experiment, power and paper inclusion

**PROPOSAL ONLY:** complete the CPU mask/label/provenance audit, then evaluate the existing step-25 export against base on a frozen fresh behavioral set. The final checkpoint can serve as a development reference if separately budgeted. This answers an actionable dose question without a new training run. Another generic increase in rank, learning rate, epochs or context range is unsupported.

**INFERENCE — Approximate power:** at G1's discordance fraction 10/95≈.105, a normal heuristic for 80% power at two-sided α=.05 gives `(1.96+.84)^2*.105/.05^2≈330` independent tasks for a 5 pp effect, and about 2060 for a 2 pp effect. This is not exact McNemar power; family/library dependence can raise requirements. Rough G1 scaling gives ~26 quota-hours for 330 task pairs. A 95-task gate can screen larger effects but cannot reliably confirm +2 pp. Do not repeatedly reuse it until significance appears.

**INFERENCE — Another training run:** not justified as “more of T1.” A single-factor retraining ablation can become justified after masks/splits are audited and a measurable behavioral hypothesis is frozen. It remains a proposal and requires separate authorization.

**INFERENCE — Paper placement:** keep R2/RS2/LC1/T1/LP1/LP2 engineering detail in this report or a future training paper. Add one brief paragraph to the current diagnostic paper qualifying earlier imitation-collapse claims: the newer adapter learns likelihood and serves, but G1 is inconclusive. Do not transform a counterfactual-diagnostics paper into an underpowered LoRA performance paper. The latest leaderboard score remains pending/unverified.

## 9. Actions and evidence limits

### Shrey should now

1. Preserve and provide authorized copies/manifests of RS2 NPZ/episode IDs, tokenizer/template, configs, step-25/final exports and hashes. Obtain the actual v18 score receipt. HQ records a leaderboard submission before the planned new-60 gate; disclose the decision-rule deviation rather than implying full compliance.
2. Review all 11 span mismatches, multi-call messages, combined edit/pytest commands and stray masks before any new data build. This audit changes none of that code.
3. Treat toolz/pathspec as development libraries after checkpoint/serving selection. Define fresh repository/fault-family splits before future comparisons.
4. Prefer existing-checkpoint evaluation over more training. Freeze paired metrics, overflow/time guards and realistic power claims.
5. Qualify “NF4 mismatch harmless,” “overflow mostly noise,” “all fixes exact,” “no one verifies” and universal imitation-failure statements.

### Atul can independently do without GPU

1. Reconcile aggregate IDs/counts/hashes and maintain a public claim ledger without publishing raw private episodes.
2. Adjudicate the ten A12v cases in the companion paper review when exact hybrid patches are available, separating message contracts, composition and target behavior.
3. Audit actual tokens/masks for the 11 mismatches; establish mutation/test tag precision on denials, dynamic writes and multi-call messages.
4. Check task/library/snapshot-family overlap across train, validation, gate and proposed future tests. Leave pretraining contamination unresolved unless evidence supports a claim.
5. Supply proposed manuscript wording and citation corrections in review reports, then help freeze evidence by November 5. No new GPU job is needed for the current paper's revisions.

**VERIFIED scope:** reviewed rft_core, rs2_core; trainkit data/config/model/worker/export, all five fix modules, prefix/probe, main and launcher; lora_gate/lora_probe/kvdup; and v18 agent/sampling declarations. No pipeline entry point was executed. CPU checks used JSON arithmetic, seeded bootstraps and isolated pure AST/tagging helpers. No Torch model or optimizer step was run.

**BLOCKED evidence:** raw RS2 NPZ/episode manifest, full G1 traces/patches, A12v hybrids, adapter binaries, complete T1 rank logs and actual v18 score were not local. Derived loss ratios reproduce recorded numbers, not hardware measurements. One R2 shard archive cannot substitute for G1 or the other shard. No Kaggle files were downloaded or permissions changed for this audit.

**VERIFIED original-audit preservation:** all 894 pre-existing tracked-file hashes matched after the audit, and only these two Markdown reviews were created. The original audit performed no staging, commit, push, merge, branch change or PR. The subsequent user-authorized review-branch publication is documented below and does not authorize experiments.


## Publication corrections and privacy review — 2 November 2026

This appendix records only changes to this review before its user-authorized, review-only GitHub PR. The opening “No fetch was performed” describes the original audit, not this publication pass. The publication pass fetched `origin`, confirmed `main` and `origin/main` both at `8f8dd657467709ab27eb76eae43e6408986ecaa6`, and uses a dedicated review branch. The scientific evidence cutoff remains unchanged. No source paper, training implementation, experiment output or Kaggle resource was changed.

| Correction | Reason and scope |
|---|---|
| T1 — Distinguish records from independent measurement | RS1/RS2 aggregate dataset counts and PA1/LC1/LP2 hardware measurements are now explicitly REPORTED. T1 step counts/ratios remain VERIFIED only as recomputation from recorded metrics. No hardware experiment was repeated. |
| T2 — Separate candidate-ID arithmetic from split identity | The 40 RS1/G1 candidate IDs and subgroup outcomes are observed; their identity with the actual RS2 validation set is an INFERENCE pending the raw manifest/NPZ. No direct training-leakage claim is made. |
| T3 — Preserve historical audit restrictions accurately | Changed the closing preservation statement to past tense and documented this subsequent authorized review-branch publication. Future experiment proposals still require separate approval. |
| T4 — Clarify source references and private-data boundaries | Added the path/access legend. Aggregate outcomes and already-public case IDs may be reviewed here; raw trajectories, patches, fixtures and training binaries are neither attached nor published. This report has no external Markdown citation URLs to repair. The companion report's public literature links were rechecked. |

**VERIFIED publication privacy check:** no credentials, private task/fixture bodies, non-public patches, raw traces or dataset payloads appear in this report. Proposals remain hypotheses or interpretations; missing raw data, binary checks and leaderboard evidence remain blocked. Source code and experiment artifacts are unchanged. The PR is review-only and must not be merged automatically.


### Source-path and access legend

Short references such as `paper_draft.md`, `paper_outline_find_vs_finish.md`, `references.bib` and `thesis_found_not_finished.md` resolve under `research_deliverables/`. `necessity.py`, `lora_gate.py`, `lora_probe.py` and `kvdup.py` resolve under `kaggle/eval_l4/evalkit/`; `rft_core.py` and `rs2_core.py` under `kaggle/rftset/`; `data.py`, `worker.py`, `model.py`, `config.py`, `export.py`, `launcher.py` and `fix_*.py` under `kaggle/train/trainkit/`. A12v JSON filenames resolve under `results/runs/a12v/`, and A3/A4 label/case files under `results/runs/a3a4/`. Explicit `data/runs/` paths describe workstation evidence, not guaranteed public artifacts. Citations to these files disclose provenance, not permission to redistribute their private inputs.
