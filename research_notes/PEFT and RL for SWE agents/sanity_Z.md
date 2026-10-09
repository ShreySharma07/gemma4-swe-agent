# Sanity check: Proposal Z (zero-information actions) against the current literature

Reviewer: independent sanity pass (8 Oct 2026). Scope: is Z a real, open, field-level gap, and what did the previous passes (critic Q1–Q38, red-team R1–R26) miss?

**Labels.**
- **VERIFIED**: I read the primary source in this session. That means the abstract via the research index, or the full text via WebFetch where noted.
- **SNIPPET**: seen only in a search-engine summary or snippet.
- **INFERRED**: my reasoning.

**Proposal under review.** A "zero-information action" (ZIA) is an exact tool call re-issued with no intervening environment-state change, so its output is claimed to be "provably identical". Z has three parts:
1. a ZIA taxonomy and prevalence study, on our runs plus ≥1,000 Nebius SWE-rebench-OpenHands trajectories;
2. a 31B "mechanistic" copy probe (`L_act` vs `L_abl` vs `L_pre`);
3. a guarded, state-certified unlikelihood LoRA vs DITTO, plain unlikelihood and a decoding repetition penalty.

It was scored 21/25 in critic Round 4.

---

## Q1. Prior art the earlier passes did not find

### Takeaway
No paper does what Z proposes in full: state-certified exact-repeat negatives, trained offline into a LoRA, with a legitimate-repeat guard, for SWE agents. But every component now has closer neighbours than the critic and red team recorded:
- **TACT** already defines "overacting" as an action that returns an observation already in the agent's knowledge set. It works on SWE-bench Verified with **Gemma-4-26B**, gives a representation-level account, and applies a training-free fix (steering).
- Mechanistic repetition work exists, and the red team's R23 "no papers found" is wrong. It also points to **competing mechanisms** to in-context copying.
- Harness-level state-certified dedup and caching already exist (TVCACHE, the OpenHands StuckDetector, CORVUS, Feedback That Backfires' decoder ban).
- Training-time repeat penalties already exist in agentic RL (IACM-RL, AKBE, EAPO, HDPO/Metis).

The gap survives only as the junction "deterministic state certificate + offline negative training + guard". It must beat a **zero-training harness mask that uses the same certificate**.

### Cited Findings

**A. The closest new neighbour: TACT (missed as an analysis; the red team listed its title only in R3)**
- TACT defines *overacting* as a step whose action "returns an observation already in 𝒦t" or fires "without adequately processing recent observations". Labels come from a GPT-5.4 LLM judge (κ = 0.78 vs humans on 300 steps). Hidden states separate overacting from calibrated steps linearly (AUC ≈ 0.9). Activation steering lifts the resolve rate on SWE-bench Verified, Terminal-Bench 2.0 and CLAW-Eval:
  - +5.8 pp on Qwen3.5-27B and +4.8 pp on **Gemma-4-26B-A4B-it**;
  - up to 26% fewer steps;
  - Qwen on SWE-bench Verified: 67.0% / 75.2 steps → TACT-Gate 73.3% / 65.3 steps.
  - **VERIFIED** (abstract + full text, first 100k chars). — [TACT, arXiv 2605.05980](https://arxiv.org/abs/2605.05980)
- TACT does **not** discuss in-context copying or induction heads. It does not compare against fine-tuning, repetition penalties or harness loop detection. **VERIFIED** (full text, first 100k chars). — [TACT](https://arxiv.org/html/2605.05980)
- Implication (INFERRED):
  - TACT's "observation already in 𝒦t" is a *semantic, judge-labelled superset* of Z's ZIA.
  - Z's "first ≥30B real-agent mechanism" novelty claim is weakened: TACT already gives a representation-level account at 26–27B in the same task family, with a Gemma-4 sibling. 31B vs 27B is not a meaningful scale distinction.
  - Z's remaining delta is (i) a deterministic, judge-free label; (ii) a specific *copying* hypothesis; (iii) a weight-level (LoRA) fix. **TACT must be cited and, ideally, reproduced as a baseline.**

**B. Mechanistic repetition and copying: the red team's R23 "no papers found" is incorrect**
- **Induction-head toxicity.** Repetition curse is attributed to induction heads dominating output logits during repetition; the paper proposes attention-head regularisation as mitigation. **VERIFIED** (abstract). — [Wang et al., 2505.13514](https://arxiv.org/abs/2505.13514)
- **"Repetitions are not all alike."** ICL-induced repetition relies on a dedicated network of attention heads. *Naturally occurring repetition* "emerges early and lacks a defined circuitry". It attends disproportionately to low-information tokens, i.e. "a fallback behavior when relevant context cannot be retrieved". **VERIFIED** (abstract). — [Mahaut & Franzon, 2504.01100](https://arxiv.org/abs/2504.01100)
- **"Why do reasoning models loop?"** Loops come from "errors in learning". The paper names two mechanisms: (1) risk aversion, where the model favours an easy cyclic action when progress is hard to learn; (2) an inductive bias toward temporally correlated errors. Larger models loop less, and distilled students loop more than their teachers. Temperature is only a stopgap. **VERIFIED** (abstract). — [Pipis et al., 2512.12895](https://arxiv.org/abs/2512.12895)
- **Belief deviation (ICLR 2026).** Agents' beliefs drift from the true state, "leading to loss of state awareness and uninformative or repetitive actions". T³ truncates trajectories at excessive deviation. **VERIFIED** (abstract); the ICLR 2026 oral status is **SNIPPET**. — [T³, 2510.12264](https://arxiv.org/abs/2510.12264)
- **Repetition neurons vs induction heads.** The paper reports strategies to reduce repetitive outputs while keeping ICL. **VERIFIED** (abstract). — [2507.07810](https://arxiv.org/abs/2507.07810)
- **Hapax.** Omitting the loss on tokens predictable by induction heads weakens copying but keeps abstractive ICL. **VERIFIED** (abstract). — [In-Context Learning Without Copying, 2511.05743](https://arxiv.org/abs/2511.05743)
- Implication (INFERRED): Z's "is in-context copying the mechanism?" question has **four live rival hypotheses** in the literature: induction-head copying, retrieval-failure fallback, errors-in-learning/risk aversion, and belief deviation. The current Stage-1 probe separates only "copying vs not". Z becomes a much better paper if the probe is designed to discriminate among them (see Q3).

**C. SWE-trajectory prevalence and behaviour studies (missed or under-used)**
- **2609.30725.** The definitions and numbers, full text read:
  - **ReTest** = "rerunning identical tests without an updated patch", detected via inter-patch windows; the last run in a window is kept and earlier ones are flagged.
  - ReTest share of tasks / cost share: Claude Code 49.67% / 0.83%; MSA+Sonnet 4.6 69.00% / 3.17%; MSA+MiniMax-M3 83.00% / 5.39%; MSA+Qwen-3.5 Plus 66.00% / 3.43%.
  - The headline "79–98% of tasks" covers **any of three behaviours**, two of which (SubRetrv, SimScrpt) are **not exact repeats**. Together they make up 6.86–22.75% of cost.
  - Named ReTest causes include "truncated or ambiguous test output".
  - **VERIFIED** (full text, first 100k chars). — [2609.30725](https://arxiv.org/html/2609.30725)
  - The critic's Relevance line ("Redundant actions affect 79–98% of tasks") **overstates the exact-repeat slice**. The exact-repeat (ReTest) cost share is **≈1–5%**. INFERRED from the table above.
- **SABER.** It splits SWE-bench Verified and τ-bench traces into mutating and non-mutating steps. Each extra deviation in a *mutating* action cuts the odds of success by up to 92–96%. Deviations in non-mutating actions have "little to no effect". **VERIFIED** (abstract). — [SABER, 2512.07850](https://arxiv.org/abs/2512.07850)
  - INFERRED: Z's target class is non-mutating repeats, so SABER predicts a **small resolve-rate effect**. Z's honest outcome is probably cost and steps, not resolve rate.
  - SABER also shows the mutating/non-mutating split that Z's certificate rests on is an established analysis axis. Cite it.
- **Cross-framework behaviour analysis.**
  - 64,380 SWE-bench runs from 126 configurations and 43 frameworks.
  - With the LLM held fixed, swapping the framework flips the *sign* of behaviour→outcome effects on most signals.
  - Framework explains 64% of between-configuration variance in mean turns, against 10% for the LLM.
  - The paper concludes that single-framework behavioural findings "warrant cross-configuration validation".
  - **VERIFIED** (abstract). — [2605.18332](https://arxiv.org/abs/2605.18332)
- **RedundancyBench.** Redundant-step detection is hard: the best method scores 24.88%. **VERIFIED** (abstract). — [2605.29893](https://arxiv.org/abs/2605.29893). INFERRED: Z's exact-repeat certificate is the high-precision, low-recall corner of this problem. Position it that way.
- **P2T.** It curates SFT data so each step is "information-bearing rather than redundant or looping", giving up to +10.8 Pass@1 on SWE-bench Verified with 15% lower inference cost. **VERIFIED** (abstract). — [P2T, 2605.21996](https://arxiv.org/abs/2605.21996). INFERRED: this is the strong "just filter loops from SFT data" baseline. Z's arm A only partly covers it.
- **Other SWE studies that also report looping and redundancy as failure markers.** SWE-PRM ([2509.02360](https://arxiv.org/abs/2509.02360), taxonomy-guided PRM on SWE-bench Verified, 40.0 → 50.6), Fail-Fast ([2608.03222](https://arxiv.org/abs/2608.03222)), Failure-as-a-Process (1,794 annotated Terminal-Bench trajectories, 3 scaffolds; [2607.09510](https://arxiv.org/abs/2607.09510)) and AgentDiet ([2509.23586](https://arxiv.org/abs/2509.23586)). **VERIFIED** (abstracts).

**D. Harness-level, state-aware and caching solutions (the missing baselines)**
- **OpenHands StuckDetector.**
  - It is enabled by default (`stuck_detection=True`).
  - It flags "the same action produces the same observation repeatedly (4+ times)", the same action with errors 3+ times, and ping-pong patterns (6+ cycles). It "can automatically halt execution".
  - Actions are compared by tool name, content and thought; observations by content.
  - **VERIFIED.** — [OpenHands docs](https://docs.openhands.dev/sdk/guides/agent-stuck-detector)
  - A GitHub issue notes that legitimate polling trips the 4-repeat rule, which made the thresholds configurable. **SNIPPET.** — [OpenHands SDK issue #762](https://github.com/OpenHands/software-agent-sdk/issues/762)
- **TVCACHE.** "Naively caching [tool] outputs … is incorrect since tool outputs depend on the environment state". A cache hit is allowed only when "the agent's full tool history matches a previously executed sequence, guaranteeing identical environment state", with hit rates up to 70%. **VERIFIED** (abstract). — [TVCACHE, 2602.10986](https://arxiv.org/abs/2602.10986)
  - INFERRED: the *concept* of a state certificate for identical tool output is already in the systems literature. Z's certificate (no mutating action since the earlier call) is a weaker, heuristic version of TVCACHE's prefix match. TVCACHE also implies the trivial harness fix: serve the cached output and tell the agent.
- **CORVUS.** Append-only trajectories freeze file-read snapshots. When files change "through agent edits or concurrent human modifications", snapshots go stale and agents "redundantly re-read files". A synchronised file registry gives 9–50% fewer input tokens and up to 37% fewer reasoning cycles on SWE-PolyBench Verified and SWE-bench Pro, at comparable pass rates. **VERIFIED** (abstract). — [CORVUS, 2607.22711](https://arxiv.org/abs/2607.22711)
- **ARC.** It stores tool observations in an ID-addressable log, so the agent can recall content "without re-executing the corresponding tools". **VERIFIED** (abstract). — [ARC, 2607.25066](https://arxiv.org/abs/2607.25066)
- **Feedback That Backfires.** Both remedies below are zero-training baselines Z must run:
  - Replacing the verbatim failed call with a runtime description removes 76% of the inversion.
  - "Making previously-failed strings unreachable at the decoder acts on the same term".
  - A "do not repeat" instruction does nothing. Retrying from a clean context is the worst harness.
  - **VERIFIED** (abstract). — [2608.23651](https://arxiv.org/abs/2608.23651)

**E. Training-time repeat and redundancy penalties (the red team found AKBE only)**
- **IACM-RL.** It reports "reducing infinite loops" in long-horizon tool invocation (BFCL-V3, τ²-Bench). **VERIFIED** (abstract). Its reward includes a consecutive-identical-call "loop short-circuit" penalty (−2.0) and −0.5 per excess duplicate call. **SNIPPET.** — [IACM-RL, 2608.02110](https://arxiv.org/abs/2608.02110)
- **EMNLP 2025 Industry RL planning paper.** It has a negative repetition reward `R_repeat` "to discourage redundant tool calls". **SNIPPET** (PDF not parseable; title and definition unverified). — [ACL Anthology 2025.emnlp-industry.116](https://aclanthology.org/2025.emnlp-industry.116.pdf)
- **EAPO and HDPO/Metis.** EAPO applies difficulty-aware penalties on redundant tool calls. HDPO/Metis argues that scalar tool penalties either suppress essential tool use or get washed out by advantage normalisation. **VERIFIED** (abstracts). — [EAPO 2606.02132](https://arxiv.org/abs/2606.02132); [HDPO 2604.08545](https://arxiv.org/abs/2604.08545)
- **Targeted Negative Training.** It updates a model to avoid unwanted outputs "while minimally changing model behavior otherwise". This is a negative-training loss family adjacent to unlikelihood and DITTO, and a candidate extra loss arm. **VERIFIED** (title + abstract fragment via citation graph). — [2406.13660](https://arxiv.org/abs/2406.13660)

**F. Web, embodied and other domains**
- WebArena-Lite human audit of 102 failed GPT-5.5 trajectories: "frequent scrolling loops". **VERIFIED** (abstract). — [2610.01491](https://arxiv.org/abs/2610.01491)
- Embodied agents are "frequently trapped in repetitive loops or issuing ineffective commands". The paper studies early exit. **VERIFIED** (abstract). — [2505.17616](https://arxiv.org/abs/2505.17616)
- dLLM agents "repeatedly fall into retry loops". **VERIFIED** (abstract fragment). — [2609.38536](https://arxiv.org/abs/2609.38536)
- Infinite agentic loops studied as a static-analysis bug class: 68 confirmed IALs in 47 projects. **VERIFIED** (abstract). — [IAL-Scan, 2607.01641](https://arxiv.org/abs/2607.01641)

**G. Citation-graph checks**
- The research index's citer list for DITTO (2206.02369) and for Welleck UL (1908.04319) returned mostly text-degeneration and decoding work: Penalty Decoding, the LZ penalty, Antislop, the production repetition study 2512.04419 and Self-Contrastive Training. It returned **no agent or tool-call application**. **VERIFIED** (tool output).
- Caveat: the index's citer list for DITTO did **not** include aTTT (2607.03441), which the red team showed cites the area. So the citer graph is **incomplete**, and absence there is weak evidence. INFERRED.
- OpenReview could not be queried directly. Web searches restricted to OpenReview and ICLR 2026 surfaced only LoopTool (withdrawn; unrelated to action loops), ARPO and the belief-deviation paper. **SNIPPET.** ICLR 2027 and NeurIPS 2026 submissions are not indexed by my tools.

### Inferences
- Novelty must be re-scored down. The concept (TACT, belief deviation), the certificate (TVCACHE, SABER's split, OpenHands StuckDetector), training-time repeat penalties (IACM-RL, AKBE, EAPO) and the copying mechanism (2505.13514, 2504.01100) each have direct precedents. What is still unclaimed:
  - (a) a deterministic, judge-free exact-repeat label with an explicit legitimate-repeat guard;
  - (b) a test of whether weight-level training adds anything over a harness enforcing the same label;
  - (c) a discriminating mechanism test among copying, fallback, risk aversion and belief deviation at ~30B on real agent states.
- The strongest reviewer attack is no longer "this is DITTO for tool calls". It is now: **"If the label is deterministic and computable at inference time, why train? A harness ban or cache does it exactly."** The proposal currently has no answer or baseline for this.

### Gaps
- I could not search OpenReview ICLR 2027 or NeurIPS 2026 submissions directly. A concurrent "unlikelihood on redundant agent actions" submission cannot be excluded.
- The full text of IACM-RL and the EMNLP-industry `R_repeat` paper were not read, so the details of their repeat penalties are SNIPPET-only.
- I did not verify whether the Nebius corpus (OpenHands v0.54.0) ran with the StuckDetector active.

---

## Q2. Is "no intervening state change ⇒ zero information" true?

### Takeaway
Not as stated. "Provably identical output" fails for:
- non-deterministic commands (time, randomness, network, test-order or hash-seed flakiness, timing lines);
- exogenous or hidden state (background processes, concurrent human edits, caches such as pytest `--lf`);
- calls that are themselves non-idempotent;
- the agent's *context* no longer containing the earlier output (compaction, truncation, condensers).

It is also an information claim relative to the *environment*, not the *agent*. A re-read after compaction or long-context decay carries information the agent no longer has. The definition needs four explicit conditions and a rename (below).

### Cited Findings
- **Flaky tests are common.** In a Python study, the tests that reproduce as flaky are "dominated by flakiness from network interactions and randomness". **VERIFIED** (abstract). — [Shaker-for-Python, 2609.25528](https://arxiv.org/abs/2609.25528)
- The state of practice for triaging a test failure is to *re-run* it: "if a test fails and then passes, it is flaky by definition". **VERIFIED** (abstract). — [230,439 Test Failures Later, 2401.15788](https://arxiv.org/abs/2401.15788)
  - INFERRED: an identical test rerun with no patch change is sometimes exactly the *correct* epistemic action.
- **Order-dependent flakiness** from shared files or shared mocking state is common in JS/Jest. **VERIFIED** (abstract). — [2501.12680](https://arxiv.org/abs/2501.12680)
- **Polling a long-running job** legitimately repeats the identical action and observation, and trips OpenHands' 4-repeat rule. **SNIPPET.** — [OpenHands issue #762](https://github.com/OpenHands/software-agent-sdk/issues/762)
- **Workspace state can change without an agent action.** SWE-Touch benchmarks agents in shared workspaces where users modify code mid-task; CORVUS cites "concurrent human modifications" as a cause of stale snapshots. **VERIFIED** (abstracts). — [SWE-Touch, 2608.02499](https://arxiv.org/abs/2608.02499); [CORVUS](https://arxiv.org/abs/2607.22711)
- **Tools are not atomic.** Real systems show "timeouts after dispatch, delayed …" behaviour. **VERIFIED** (abstract fragment). — [Verified Tool Calls, 2608.02645](https://arxiv.org/abs/2608.02645)
- **Agents assume a stationary context** and ignore elapsed time. **VERIFIED** (abstract fragment). — [Temporally Blind, 2510.23853](https://arxiv.org/abs/2510.23853)
- **Compaction removes context.**
  - Compaction can "silently remove" in-context constraints: violations rise from 0% to 30% after compaction. **VERIFIED** (abstract). — [Governance Decay, 2606.22528](https://arxiv.org/abs/2606.22528)
  - Coding-agent compaction is now trained into policies, so trajectories increasingly contain post-compaction states. **VERIFIED** (abstract). — [AutoCompact, 2610.02163](https://arxiv.org/abs/2610.02163)
  - CompactionRL trains GLM-4.5-Air on SWE-bench with compaction. **VERIFIED** (abstract). — [CompactionRL, 2607.05378](https://arxiv.org/abs/2607.05378)
- **Truncation.** 2609.30725 lists "truncated or ambiguous test output" among ReTest causes. **VERIFIED** (full text).
- **Natural repetition as retrieval fallback.** It looks like "a fallback behavior when relevant context cannot be retrieved". **VERIFIED** (abstract). — [2504.01100](https://arxiv.org/abs/2504.01100). INFERRED: some "zero-information" repeats are the model compensating for long-context retrieval failure. Suppressing them without restoring the content may hurt.

### Inferences (proposed refined definition)
Rename to **"observation-redundant action" (ORA)** or "certified zero-*environmental*-information action". Then label a_t as ORA iff all of the following hold:
1. **Same call.** a_t equals an earlier a_s after normalisation (whitespace, flag order). Report exact-string and normalised variants separately.
2. **Output still in context.** o_s is still verbatim and untruncated in the context the model saw at step t. That means no compaction, condenser drop, or harness truncation of o_s. Otherwise label the step "re-acquisition", which belongs to the guard class.
3. **State certificate**, ranked by strength:
   - (i) sandbox snapshot hash (repo tree + relevant process table) equal at s and t, as in TVCACHE-style prefix matching;
   - (ii) no intervening action outside a *read-only whitelist*, **and** a_s itself read-only or idempotent (a repeated `sed -i`, `pip install`, `git stash` or `pytest --lf` is not a ZIA even with nothing in between);
   - (iii) the current "no mutating action" heuristic.
   Report which tier each label used. Public corpora usually support only tier (ii) or (iii), so "provably" must be dropped for them.
4. **Determinism certificate.** Replay a_s twice in a restored sandbox and require identical output after masking timing and timestamp lines. Exclude, or put in the guard class, commands touching network, time, randomness, background jobs or flaky tests. Where replay is impossible (most public corpora), use a conservative denylist and report sensitivity.

Also state explicitly that **information ≠ decision value**. Re-emitting content near the end of context may help a long-context model. The guard therefore needs a third class besides post-change repeats: "re-acquisition after compaction or long distance". The paper should report ZIA rate as a function of distance (s → t in tokens).

### Gaps
- No study I found measures how often SWE-agent repeats are output-identical on replay. Z can contribute this cheaply (tier-1 replay on a sample of ~200 repeats), which would be a genuine verifiability plus.

---

## Q3. Would the experiment design test the claims?

### Takeaway
Partly. The one-step offline design can show that the LoRA lowers greedy re-emission. It cannot show that (a) training beats a zero-training harness that uses the same certificate, (b) copying is the mechanism rather than a rival hypothesis, (c) the agent does not just switch to near-duplicate evasions, or (d) anything changes end-to-end. The 40-task closed-loop check is underpowered for resolve rate.

### Cited Findings
- **Zero-training remedies that act on the same term.** Decoder unreachability and runtime descriptions remove 76% of the inversion, while instruction-only prompts fail. **VERIFIED** (abstract). — [2608.23651](https://arxiv.org/abs/2608.23651)
- **Training-free steering.** TACT reaches +4.8 to +5.8 pp resolve and up to 26% fewer steps. **VERIFIED.** — [TACT](https://arxiv.org/abs/2605.05980)
- **Prompt rules.** DevSkills robustly cut cost in 6 of 8 settings (7.88–41.73%), with one −1.50 pp Pass@1 drop. **VERIFIED** (full text). — [2609.30725](https://arxiv.org/html/2609.30725)
- **Decoding penalty.** A repetition penalty of 1.1 hurt (49.7 vs 50.7, ALFWorld). VERIFIED by the red team, not re-checked here. — [aTTT 2607.03441](https://arxiv.org/abs/2607.03441)
- **Over-penalisation risk.** Scalar penalties either over-suppress or vanish under advantage normalisation; HDPO/Metis argues this in the RL setting. **VERIFIED** (abstract). — [HDPO](https://arxiv.org/abs/2604.08545)
- **Small expected effect on success.** Non-mutating deviations barely move success. **VERIFIED.** — [SABER](https://arxiv.org/abs/2512.07850)
- **Single-harness findings may flip sign across frameworks.** **VERIFIED.** — [2605.18332](https://arxiv.org/abs/2605.18332)

### Inferences (concrete design changes)
1. **The killer baseline: a state-certified harness mask (H-mask).**
   - How it works: at inference, when the certificate holds, ban the exact repeated call string (logit mask, or reject-and-resample), or return "output identical to step k; state unchanged" without executing (H-cache, in the TVCACHE / Feedback That Backfires style).
   - What it tests: whether the LoRA teaches something the mask cannot, i.e. choosing a *better* next action rather than merely not-the-repeat.
   - Pre-registered kill: **if H-mask ≥ LoRA-B on closed-loop steps and cost at equal resolve rate, the training claim dies.** The paper then becomes "measurement + harness fix", which is still publishable.
2. **Add these baselines:**
   - the OpenHands StuckDetector at its defaults;
   - the DevSkills-style prompt rule;
   - loop-filtered SFT data (P2T/SWE-Prime style), which strengthens arm A;
   - TACT steering if feasible on Gemma-4-31B, or else a cited comparison;
   - DITTO, plain UL, and the decoding penalty (already planned).
   - Optionally a Targeted Negative Training arm.
3. **Measure displacement and evasion.** Add a near-duplicate rate (normalised-command equality, flag permutations, `pytest x` ↔ `python -m pytest x`, and the SimScrpt-style Jaccard ≥ 0.6 from 2609.30725), plus "semantically redundant" re-reads (subsumed retrieval). Unlikelihood on exact strings invites surface evasion. INFERRED.
4. **Make the mechanism probe discriminating** (currently `L_act` / `L_abl` / `L_pre`):
   - **Paraphrase counterfactual.** Show the earlier call in a different surface form with the same output. If the repeat probability stays high, the driver is semantic or belief-based, not surface copying. This mirrors Feedback That Backfires' surface vs semantic split.
   - **Length-matched placeholder control.** The current ablation also shortens context and removes evidence.
   - **Repeat probability vs number of prior repeats** (the DITTO self-reinforcement curve) on real states.
   - **Distance and compaction stratification** to test the retrieval-fallback hypothesis (2504.01100).
   - **A head-level test**, to earn the word "mechanistic": induction-head scoring and ablation (2505.13514 method) on the 31B in HF/bnb, or on Gemma-4-26B-A4B if memory forbids.
   - Without head-level work, call it a **behavioural copy test**, not a mechanistic probe.
5. **Closed-loop power.**
   - Problem: 40 tasks gives a resolve-rate SE of about √(0.3·0.7/40) ≈ 7 pp. That cannot detect the 2–5 pp effects SABER predicts. INFERRED.
   - Primary endpoints: per-task steps, tokens, and ORA rate (continuous, paired by task).
   - Resolve rate: report as a non-inferiority bound (e.g. −3 pp margin) on ≥150 tasks, or drop the claim.
6. **Guard set.** Build ~50–100 positive states where repeats are *correct*: flaky-test triage, polling a background job, transient network retry, re-read after compaction. Report keep-rates there. This is the real test of "without suppressing legitimate repeats", and it is more convincing than PCR log-probs.
7. **Corpus confound.** Nebius trajectories come from OpenHands v0.54.0 with Qwen3-Coder-480B, max 100 turns. VERIFIED from the [dataset card](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories). If its StuckDetector was active, which is the default in current OpenHands, runs of ≥4 identical action-observation pairs were **halted**. That right-censors loop length and bakes "loop ⇒ failure" into the corpus. INFERRED; must be checked.
   - Fix: report ZIA statistics with and without stuck-terminated runs, and add a corpus from a scaffold without such a detector (SWE-agent/SWE-smith or mini-swe-agent).
8. **Report cost share, not just task share.** 2609.30725's exact-repeat (ReTest) cost share is about 1–5%. VERIFIED. The Relevance argument should cite that number honestly.

### Gaps
- Feasibility of head-level analysis on Gemma-4-31B within 4×L4 memory was not assessed.

---

## Q4. Is the generality claim credible beyond this competition?

### Takeaway
Partly. The certificate idea generalises to agents with **inspectable, mostly deterministic state**: coding and terminal sandboxes, τ-bench-style read/write APIs, and ALFWorld. It does not generalise cleanly to web agents. There, state changes exogenously, actions like scroll change the viewport without progress, and "frequent scrolling loops" are *not* exact zero-state-change repeats. The current plan (one extra corpus, one scaffold) is below the bar that 2605.18332 sets for behavioural claims.

### Cited Findings
- Framework identity, more than the LLM, drives behaviour-outcome relations, and signs flip across configurations. **VERIFIED.** — [2605.18332](https://arxiv.org/abs/2605.18332)
- τ-bench and SWE-bench Verified support a clean mutating/non-mutating split. **VERIFIED.** — [SABER](https://arxiv.org/abs/2512.07850)
- WebArena-Lite failures show frequent scrolling loops. **VERIFIED.** — [2610.01491](https://arxiv.org/abs/2610.01491)
- Web-agent world models must handle "repeated elements across observations" and focus on state differences. **VERIFIED** (abstract). — [WMA, 2410.13232](https://arxiv.org/abs/2410.13232)
- Embodied and ALFWorld agents loop. aTTT reports previous-action repeat rates of 7% (Gemma-3-12B) and 71% (Qwen3.5-4B) on ALFWorld failures (VERIFIED by red team). — [aTTT](https://arxiv.org/abs/2607.03441); [2505.17616](https://arxiv.org/abs/2505.17616)
- Loops appear in tool-calling RL (BFCL, τ²): IACM-RL (VERIFIED abstract).

### Inferences
- Reframe the claim as: "for tool agents whose environment exposes a state certificate". Show it on **≥3 SWE scaffolds** (OpenHands/Nebius, SWE-agent/SWE-smith, mini-swe-agent or Claude-Code-style traces if licensed) and **≥2 generating models**, plus **one non-SWE domain with a clean certificate**: τ-bench (read-only tools are labelled) or ALFWorld (deterministic simulator).
- Leave web agents to future work, with a stated reason: the certificate needs a DOM/state hash, and viewport actions break the action/state equivalence.
- The probe and LoRA results are on one model (Gemma-4-31B W4A16). Generality there can only be argued by the CPU-side taxonomy across corpora, so say so.

### Gaps
- I did not check which public SWE trajectory corpora carry enough information (full observations, untruncated) for tier-2 certification.

---

## Q5. Verdict: what Z missed, does the gap survive, what must change

### Takeaway
**The gap survives, narrower. Z is still a reasonable headline, but 21/25 is inflated.** A fair re-score, assuming the changes below:

| Criterion | Old | New | Why |
|---|---|---|---|
| Novelty | 3 | 2–3 | TACT (concept, SWE, Gemma-4, representation-level), TVCACHE (state certificate), repetition-mechanism papers, RL repeat penalties |
| Quality | 4 | 3 | Rises to 4 only with multi-scaffold plus a non-SWE certificate domain |
| Relevance | 4 | 3 | Exact-repeat cost share ≈1–5%; SABER predicts a small success effect |
| Verifiability | 5 | 5 | Kept, if replay certification and code are released |
| Clarity | 5 | 4 | Rival mechanisms and the extra baselines add length |
| **Total** | **21** | **≈17–19** | Still ≥ DECAF (16), so the ranking probably holds; the margin is thin |

All scores are INFERRED and GUESSED.

### What the proposal missed (concrete)
1. **TACT** ([2605.05980](https://arxiv.org/abs/2605.05980)) defines overacting as an action returning an already-known observation, on SWE-bench Verified with Gemma-4-26B. It uses a linear residual-stream signal and a training-free steering fix (+4.8 to +5.8 pp). This is the closest prior art and is absent from the analysis.
2. **The repetition-mechanism literature** contradicts the red team's R23 null. Each paper gives a different account of where repetition comes from:
   - induction-head toxicity ([2505.13514](https://arxiv.org/abs/2505.13514));
   - natural repetition as circuit-less retrieval fallback ([2504.01100](https://arxiv.org/abs/2504.01100));
   - loops from errors in learning and risk aversion ([2512.12895](https://arxiv.org/abs/2512.12895));
   - belief deviation ([2510.12264](https://arxiv.org/abs/2510.12264)).
3. **State-certified dedup and caching already exist at the harness and systems level**: TVCACHE ([2602.10986](https://arxiv.org/abs/2602.10986)), the OpenHands StuckDetector (on by default; [docs](https://docs.openhands.dev/sdk/guides/agent-stuck-detector)), CORVUS ([2607.22711](https://arxiv.org/abs/2607.22711)), ARC ([2607.25066](https://arxiv.org/abs/2607.25066)), and Feedback That Backfires' decoder ban ([2608.23651](https://arxiv.org/abs/2608.23651)). **The proposal has no zero-training, same-certificate baseline.**
4. **Training-time repeat penalties already exist in agent RL**: IACM-RL ([2608.02110](https://arxiv.org/abs/2608.02110), SNIPPET for the penalty details), the EMNLP-25 industry `R_repeat` (SNIPPET), EAPO and HDPO. Z's delta is *offline + state-certified + guarded*, not "penalising repeats".
5. **The relevance number is mis-cited.** "79–98% of tasks" covers three behaviours, two of them non-exact. The exact-repeat (ReTest) slice is 49.67–83% of tasks but only **0.83–5.39% of cost** ([2609.30725](https://arxiv.org/html/2609.30725)).
6. **SABER** ([2512.07850](https://arxiv.org/abs/2512.07850)): non-mutating deviations have little effect on success, so Z should expect efficiency gains, not resolve-rate gains.
7. **Cross-framework sign flips** ([2605.18332](https://arxiv.org/abs/2605.18332)): a one-corpus generality check is not enough.
8. **The definition is unsound as "provable"**: flaky tests ([2609.25528](https://arxiv.org/abs/2609.25528), [2401.15788](https://arxiv.org/abs/2401.15788)), polling (OpenHands #762, SNIPPET), exogenous edits ([SWE-Touch](https://arxiv.org/abs/2608.02499)), non-idempotent calls, compaction ([2606.22528](https://arxiv.org/abs/2606.22528), [AutoCompact](https://arxiv.org/abs/2610.02163)) and truncation.
9. **Corpus confound.** The Nebius corpus comes from OpenHands v0.54.0. A default StuckDetector would right-censor exact loops (INFERRED; verify).
10. **Loop-filtered SFT is a strong data baseline**: P2T ([2605.21996](https://arxiv.org/abs/2605.21996)), +10.8 Pass@1.

### What must change (minimum set)
1. **Rename and redefine.** Use "certified observation-redundant action" with the four conditions in Q2: same call, output still in context, a tiered state certificate, and determinism by replay. Drop "provably" unless tier-1 replay is done. Add guard classes for post-change repeats, flaky/poll/retry, and re-acquisition after compaction.
2. **Add H-mask / H-cache as the pre-registered killer baseline** (Q3.1), plus the StuckDetector, a DevSkills prompt, loop-filtered SFT, and TACT (or a cited comparison). The paper's central claim becomes: *does weight-level training beat enforcing the same certificate at the harness?*
3. **Rewrite the mechanism section as a discriminating test** among copying, fallback, risk aversion and belief deviation: paraphrase counterfactual, length-matched control, repeat-count curve, distance stratification. Do at least one head-level ablation, or call it behavioural.
4. **Change the endpoints.** Primary: steps, tokens and ORA rate, plus near-duplicate displacement. Resolve rate becomes non-inferiority on ≥150 tasks. Add the legitimate-repeat guard suite.
5. **Generality.** Use ≥3 SWE scaffolds and ≥2 generating models, report with and without stuck-terminated runs, and add one non-SWE certificate domain (τ-bench or ALFWorld). Scope web agents out explicitly.
6. **Related work** must cover TACT, TVCACHE, the OpenHands StuckDetector, Feedback That Backfires' decoder ban, SABER, 2605.18332, the four mechanism papers, IACM-RL/AKBE/EAPO/HDPO, DITTO, Welleck UL and aTTT.

### Gaps
- Concurrent OpenReview submissions (ICLR 2027, NeurIPS 2026) remain unchecked.

---

## Search log

| # | Tool | Query / target | Outcome |
|---|---|---|---|
| S0 | Local read | Report, critic_judgement R4 + SCCS stages, prior_art_redteam | Proposal details; prior queries Q1–Q38, R1–R26 not repeated |
| S1 | research related_papers (citers) | DITTO arXiv 2206.02369, intent "repetition suppression applied to agents/tool calls" | Text-degeneration citers only (2512.04419, 2307.01542, 2310.14971, 2505.13514, 2504.12608 …); no agent applications; aTTT absent → index incomplete |
| S2 | research search (≥2024) | "LLM agents repeating the same action loops redundant tool calls" | IAL-Scan 2607.01641, Feedback That Backfires 2608.23651, **TACT 2605.05980**, RedundancyBench 2605.29893, early-exit 2505.17616, AgentLoop 2609.33315, dLLM loops 2609.38536 |
| S3 | research search (≥2024) | "induction heads in-context copying repetition mechanism agent trajectories" | **2505.13514**, **2504.01100**, 2507.07810, 2511.05743 (contradicts red-team R23 null) |
| S4 | research inspect ×5 | 2605.29893, 2605.05980, 2504.01100, 2512.12895, 2505.17616 | Abstracts verified |
| S5 | WebSearch (extended) | "openreview ICLR 2026 LLM agent repetitive actions loops tool calls training" | Belief-deviation (ICLR'26 oral, snippet), ARPO, LoopTool (withdrawn); no OpenReview hit on action-loop training |
| S6 | research search (≥2024-06) | "software engineering agent trajectories analysis repeated commands redundant actions failure modes SWE-bench" | SWE-PRM, **SABER 2512.07850**, **2605.18332**, Failure-as-Process 2607.09510, **P2T 2605.21996**, SWE-Touch 2608.02499, AgentLens, Fail-Fast |
| S7 | research search (≥2024) | "information gain value of information pruning redundant actions LLM agent observation unchanged state" | PABU, InfoPO, BAVT, AgentDiet, AgentLoop, CGDP 2605.07042; no state-certified pruning |
| S8 | research inspect ×5 | 2512.07850, 2605.18332, 2607.09510, 2605.21996, 2505.13514 | Abstracts verified |
| S9 | WebSearch | "OpenHands StuckDetector repeated action observation pairs loop detection" | Docs, issue #762, source study 2609.00006 |
| S10 | research search | "web agent loop detection repeated clicks stuck navigation LLM WebArena repetitive actions" | WebArena-Lite audit 2610.01491 (scrolling loops), cycle detection 2511.10650, WMA 2410.13232, real-time failure detection 2608.02464 |
| S11 | research search | "flaky tests nondeterminism rerun same test different outcome coding agents" | 2609.25528, 2401.15788, 2501.12680 |
| S12 | WebFetch | arxiv.org/html/2605.05980 (TACT) | Overacting definition, numbers, no copying/baseline comparisons |
| S13 | WebFetch | arxiv.org/html/2609.30725 | ReTest definition; task-share and cost-share table; truncation as cause; DevSkills effects |
| S14 | research related_papers (citers) | Welleck UL 1908.04319, intent "UL applied to agents/actions" | No agent applications; TNT 2406.13660, LZ penalty, Antislop |
| S15 | research search (≥2024-06) | "unlikelihood loss on bad actions fine-tuning LLM agent negative steps tool use" | SRFT 2605.10674, EEF, EAPO, AgentLTL, STAPO; no step-level UL on repeated actions |
| S16 | research inspect | 2608.23651 | Decoder-unreachable and description remedies; prompt fails |
| S17 | research search | "belief deviation RL active reasoning LLM agents repetitive uninformative actions" | T³ 2510.12264, information self-locking 2603.12109, ReBel |
| S18 | research search (≥2025) | "context compaction … coding agent re-reads files after compaction forgetting" | Governance Decay 2606.22528, AutoCompact 2610.02163, CompactionRL 2607.05378, **CORVUS 2607.22711**, ARC 2607.25066 |
| S19 | research search (≥2024-06) | "tool call result caching memoization deduplicate identical tool calls agent harness" | **TVCACHE 2602.10986**, ToolCaching 2601.15335, AgentCheck |
| S20 | WebSearch (extended) | openreview.net agent "redundant tool calls" OR "repeated tool calls" penalty training 2026 | IACM-RL penalty snippet, EMNLP-25 industry `R_repeat` snippet, OpenReward, EAPO; no OpenReview page |
| S21 | WebFetch | docs.openhands.dev agent-stuck-detector | Patterns, thresholds, equality, default-on, halt |
| S22 | WebFetch | HF nebius/SWE-rebench-openhands-trajectories | 67,074 traj., CC-BY-4.0, Qwen3-Coder-480B, OpenHands v0.54.0, 100 turns, failures included |
| S23 | research inspect | 2608.02110 (IACM-RL) | Abstract verified (loop reduction) |
| S24 | WebFetch | aclanthology 2025.emnlp-industry.116.pdf | Unparseable PDF; nothing verified |
| S25 | research search (≥2025) | "reward penalty for duplicate identical tool calls RL agent loop short-circuit" | EAPO, AKBE, PROVE, HDPO/Metis, SVRL |

**Not searched or not reachable:** OpenReview ICLR 2027 and NeurIPS 2026 submission lists; a Semantic Scholar or Google Scholar citer graph (I used the research index's citer graph as a substitute, and it is known to be incomplete); full texts of IACM-RL, SABER and the EMNLP-25 industry paper; OpenHands v0.54.0 eval config for stuck detection.
