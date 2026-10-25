# Novelty check: post-first-edit "obligation frontier" for SWE agents (2026-10-23)

Sources: alphaXiv discover_papers + answer_pdf_queries (full-text reads where marked READ), WebSearch.
The arXiv export API returned 503 during the session, so abstracts were checked through alphaXiv instead.
"Abstract only" means the claim was checked against the alphaXiv listing snippet, not the full text.
Anything marked UNVERIFIED comes from my prior knowledge and was not confirmed with a tool in this session.

## Area 1: graph/dependency change-impact and edit propagation in LLM coding

| Paper | ID / venue | Method | Key number | Status |
|---|---|---|---|---|
| CodePlan (Bairi et al.) | 2309.12499 (FSE 2024) | Seed edit, then incremental dependency graph, change *may-impact* analysis, then an adaptive plan graph whose nodes are literally "code edit **obligations**" discharged by the LLM | 5/6 repos pass validity vs 0/6 for build/pyright-repair baselines; GPT-4-32k; C# migration + Python temporal edits | READ |
| CoEdPilot | 2408.01733 (2024) | Learned edit-relevance plus project-wide search to recommend the next edit locations and contents after prior edits (IDE, interactive) | -- | abstract only |
| Coeditor | 2305.18584 (ICLR 2024, per my recollection; UNVERIFIED) | Multi-round auto-editing conditioned on contextual prior changes | -- | abstract only |
| Next Edit Prediction | 2508.10074 (2025) | Predict the location and content of the next edit from interaction history | -- | abstract only |
| NES | 2508.02473 (2025) | Low-latency next-edit suggestion trained on historical editing trajectories | -- | abstract only |
| Software World Models (SWM) | 2610.04940 (Oct 2026) | Predict a change's **blast set** over a typed System-Surface Graph; typed-reachability baseline; LM fine-tuned (GRPO) on *executed* consequences | blast-set F1 0.431 (reachability) to 0.571; synthetic + real libraries; cross-repo consumers, not SWE-bench | READ |
| Semantically-seeded graph-propagated impact analysis (vision) | 2606.18855 (2026) | Vision paper on CIA across artifacts | -- | abstract only |
| GrACE, Overwatch | -- | **Not found** in this session's searches. Do not cite without locating them. | -- | not found |

CodePlan vocabulary is dangerously close to ours: "change may-impact analysis", "derived specifications", "edit obligation", typed edges (Calls/CalledBy, Overrides/OverriddenBy, Instantiates, Imports/ImportedBy, Uses/UsedBy, Construct/ConstructedBy, BaseClassOf), and per-change-type rules (signature change => callers plus overrides; field change => users, constructors, sub/super-classes). It is also explicitly *post-seed-edit*.
CodePlan differs from our thesis in the following ways:
(a) Its setting is pervasive repo-level tasks (migration, temporal edits) with **given seed edits**, not issue resolution from an NL issue.
(b) CodePlan **drives** the LLM: a planner owns control flow and the LLM is called per block. It does not inject passive context into an autonomous agent.
(c) The graph is purely static. Its limitations section names dynamic/Python dependencies as future work, so there is no execution calibration.
(d) There is no find-vs-finish diagnosis, no test-necessity ground truth, and no shuffled-topology control.
(e) It uses GPT-4-32k, not a small quantised model.
(f) There is no SWE-bench evaluation.

Verdict for area 1: the **core mechanism (post-edit impact analysis that generates co-change obligations over a typed dependency graph) is already done by CodePlan (2023/FSE'24)**. We cannot claim "obligation closure" or "change-impact analysis after the first edit" as new. SWM (Oct 2026) adds learned and executed impact prediction, which makes "execution-calibrated" partially pre-empted, although SWM targets downstream services rather than in-repo co-change obligations for issue resolution.

## Area 2: incomplete or partial multi-location patches

| Paper | ID | Finding | Status |
|---|---|---|---|
| ChainSWE | 2607.02606 (2026) | Names an "undershoot" failure: the agent edits only issue-named files and omits supporting refactors. Of 663 downstream failures, 318 (48%) are chain errors, and under-edits outnumber over-edits about 9:1 among attributable ones. Concrete multi-file examples include MONAI 5 to 2 files and amaranth 4 to 2 files | READ |
| Incomplete Software Changes dataset | 2609.34317 (2026) | 45.7% of human bug-inducing changes are incomplete (Apache, BIC/BFC). 49.5% miss 1 file and 89.4% miss 5 or fewer. Co-change rules recover 69.7% | READ |
| MultiFixer | 2607.26591 (2026) | Coordinator-proposer multi-agent APR for multi-hunk bugs | abstract only |
| SiblingRepair | 2605.06209 (2026) | Embedding/TF-IDF sibling discovery plus simultaneous/iterative LLM repair. On Defects4J it fixed 23 sibling and 76 multi-hunk bugs. For 61% of sibling bugs, failing tests do not cover all siblings | READ |
| MULocBench | 2509.25242 (2025) | Best file-level Acc@5 is 35.2% (all locations required), which is far lower than on SWE-bench Lite | READ |
| Are "Solved Issues" Really Solved? | 2503.15223 (2025) | Plausible patches diverge behaviourally from gold. Exact rates are UNVERIFIED | abstract only |
| Investigating Test Overfitting on SWE-bench | 2511.16858 (2025) | Overfitting to observed tests | abstract only |
| Code Understanding is a Bottleneck (CABRA) | 2610.10610 (2026) | Synthetic call-graph "Add Parameter/Add Return" propagation tasks: frontier agents stay near-perfect on function traversal with N up to 200. On SWE-bench Verified, understanding tool calls predict failure better than lines edited | READ |

Verdict for area 2: incompleteness and undershoot are **documented** (ChainSWE, the human-change dataset) and multi-hunk repair methods exist (SiblingRepair, MultiFixer). However, I found **no paper that quantifies, on single-issue SWE tasks, that failed patches are a strict subset of the required co-change set given that the gold file was reached.** Our "find but don't finish" decomposition is open as a measured claim. CABRA is a caution: strong agents do *not* fail at propagation in clean synthetic call graphs, so our claim should be scoped to small/quantised models and real repos.

## Area 3: "localization is not the bottleneck"

| Paper | ID | Finding | Status |
|---|---|---|---|
| From Finding to Fixing | 2610.25826 (Oct 2026) | 6 FL methods, 3 LLMs, 16,248 trajectories. FL does not significantly improve repair: 16/18 settings have McNemar p≥0.05 and none survive Holm correction. **No-FL gold-file reach is 88.6–99.4%.** Gold file adds only +0.8pp on Verified. Bug-cause text helps more (+4.4pp) | READ |
| SWE-Explore | 2606.07297 (2026) | Agents have high HitFile (~0.65) but low line recall (0.14–0.19). Missing core context is the dominant failure, with a threshold effect between 50% and 75% of core regions | READ |
| SHERLOC | 2606.24820 (2026) | Structured diagnostic localization fed to SWE-agent/OpenHands. Gains are not uniform (per 2610.25826) | abstract + secondary |
| Loc2Repair | 2606.30963 (2026) | File-level localization hints for repo repair. Cited inside 2610.25826 and not opened | secondary |
| CodeAnchor | 2606.26979 (ISSTA 2026) | Grep-first agents already reach 83.2% Func@5 on Lite (LocAgent 59.5%) | READ |

Verdict for area 3: **"localization isn't the bottleneck" is already established** (Finding-to-Fixing, Oct 19 2026, with the same McNemar framing). Our ~97% gold-file-seen-in-failures is a replication in a new regime (31B W4A16 model, private-repo-like setting). It is not novel as a headline, so we should cite it and build on it. The open part is *what replaces localization*. Finding-to-Fixing points to "bug cause" information. We point to post-edit co-change completeness. SWE-Explore's "line-recall, not file-hit" finding is compatible with and supportive of our claim.

## Area 4: test-necessity ground truth

| Paper | ID | Method | Status |
|---|---|---|---|
| TRIM | 2607.18161 (2026) | Trajectory-guided hierarchical counterfactual removal of *agent* patch edits with test re-validation. Baseline is DD-Hunk delta debugging. Removes 17.9–32.9% of CodeSlop (20% of source lines on SWE-bench Verified) | READ |
| When Models Edit Too Much | 2609.04061 (2026) | Minimal-edit fidelity of LLM repairs | abstract only |
| From Finding to Fixing | 2610.25826 | A single reference patch misses alternative locations: modified files differ in 11.6% of successful-patch pairs | READ |

Verdict for area 4: test-validated removal (DD over hunks, TRIM) exists, but it is applied to **agent patches to remove slop**. I found no paper that applies leave-one-function-out to **gold** patches against FAIL_TO_PASS tests to define the *necessary co-change set*, or that uses it as ground truth for an impact/obligation predictor. This is **partially novel**: the technique is standard and the use is new. Caveat from 2610.25826: alternative valid patch locations exist, so necessity is relative to the gold patch.

## Area 5: passive context injection at or after edit time

| Paper / system | ID | What | Status |
|---|---|---|---|
| CodeAnchor | 2606.26979 (ISSTA 2026) | **Passive injection** of static call, inheritance, config and data-flow tags as in-source comments. Motivation: agents ignored an optional call-graph tool (low use in a 20-task pilot), which mirrors our graph-tool finding. Gains: +2.2pp Func@5, −1.6 rounds, run-to-run variance halved, Pass@1 +3.4pp on medium repos. Evaluates **localization only**, and the tags are static and pre-edit (offline) | READ |
| LARGER | 2605.16352 (2026) | Lexically anchored repository-graph retrieval for localization | abstract only |
| Does a Language Server Save Tokens? | 2608.13568 (2026) | LSP vs grep for agent retrieval | abstract only |
| OPRD | 2610.21496 (2026) | On-policy reminder distillation for constraint and tool-feedback adherence | abstract only |
| PAUSE | 2610.29870 (2026) | Search-free steering *before the first source edit* (pre-edit) | abstract only |
| SWE-agent linter-on-edit, Aider repo-map, OpenHands | -- | Known tooling, not re-verified this session. SWE-agent rejects syntactically broken edits via a linter (UNVERIFIED detail) | UNVERIFIED |

Verdict for area 5: passive structural injection is **done for localization** (CodeAnchor, with the same "agents don't call graph tools" rationale). Post-edit lint and type feedback exists in tooling. I found **no paper that injects impact or caller obligations triggered by the agent's own edit**, computed from the edit diff, into an autonomous SWE agent. That post-edit trigger is the open slot. CodeAnchor is the strongest prior to cite and contrast with: same delivery channel, opposite timing and purpose.

## Overall assessment

Prior work by component:

| Component of our thesis | Prior work | Status |
|---|---|---|
| Post-edit change-impact, obligation generation over typed dependency graph | CodePlan | **Done** (planner-driven, seed-edit tasks, static) |
| Learned or executed impact prediction | SWM 2610.04940 | Partially done (cross-service blast set) |
| Localization not the bottleneck | 2610.25826, SWE-Explore | **Done**. Ours is a replication |
| Undershoot / incomplete edits exist | ChainSWE, 2609.34317 | Documented. Not decomposed as find-vs-finish on single tasks |
| Passive structural injection | CodeAnchor | Done for localization, pre-edit |
| Test-necessity ground truth for gold co-change set | none found (TRIM is the nearest) | **Open** |
| Shuffled-frontier topology control | none found | **Open** |
| Memorisation × impact-graph via renaming | none found | **Open** (weak: our own docs note renaming keeps structure) |
| Small/quantised 31B open model, private-repo setting | none found | Open (setting delta) |

Safe novelty claim, phrased narrowly: an **edit-triggered, passive** obligation frontier, inside an **autonomous** SWE agent, computed by impact analysis on the agent's own diff, with the following properties:

- **Ground truth:** test-necessity (leave-one-out on gold vs FAIL_TO_PASS).
- **Evaluation metric:** the conditional "finish rate given find".
- **Control:** a shuffled-topology frontier, so gains can be attributed to graph structure rather than extra tokens.
- **Setting:** a small quantised open model.

Do NOT claim the following as novel: change-impact/obligation propagation (CodePlan), "localization is not the bottleneck" (Finding-to-Fixing), or passive graph injection (CodeAnchor). "Execution-calibrated" overlaps with SWM conceptually, so claim it only as "calibrating static edges with test/coverage traces for in-repo co-change", and cite SWM.
