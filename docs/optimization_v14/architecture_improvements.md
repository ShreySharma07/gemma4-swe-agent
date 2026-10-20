# Architecture improvements ranked by evidence

## Preserve the verified baseline

The baseline is the exact files in `submissions/v14_thinking/`: fixed Gemma 4 31B W4A16 base, no adapter, temperature .2, top_p .95, top_k 40, thinking enabled/budget4096, context32768, output8192, 8 min, 28 chargeable calls, 80 turns, command timeout240s. The measured gain is bundled. Do not reset to v7 for the main comparator or silently alter thinking/budgets while testing output management.

A declaration in `agent.yaml` does not grant a custom runtime. The competition-facing scaffold is declarative; source notes say custom Python entrypoints are unavailable and skill scripts are unverified. `evalkit/runner.py` controls the **local** harness compaction but modifying it would not demonstrate scorer deployability. Current official rules could not be freshly read (see roadmap). All architecture proposals remain conditional on the actual supported interface.

## Priority 1: bound generated command output at its source

**Mechanism:** one prompt-policy change specifying narrow symbol reads/searches, bounded output for tests, explicit test exit status, and a small error summary plus retrievable source/output location when results are long. Start with a proposed 1,200-character stdout/stderr display target per call; preserve necessary source regions and the actual failing assertion. Request a larger region when justified. Do not silently truncate the middle of code and then train or edit as though it were visible.

**Evidence:** eight boundary-identical context exceptions, zero recorded compactions; many unique outputs contribute tens of thousands of serialized characters per failure. The old prompt's ~80-line advice alone did not prevent this. The exact per-task scenario savings are in [failure analysis](failure_analysis.md); they measure removable excess characters under a hypothetical cap, not validated token savings or model behavior.

**Expected savings:** prompt-token reduction is `tokens(original retained output) − tokens(abbreviated retained output)` under the exact serving template. That quantity is not measurable from per-step aggregate metrics alone, so no numeric token forecast is claimed. Each 5,000-character stream bounded to 1,200 removes up to 3,800 characters before escaping/metadata. Token and quality savings must be measured by tokenizer replay before rollout. Reducing empty boilerplate and repeated stacks should preserve useful evidence better than uniform head-only clipping.

**Risks:** hiding the only failing assertion, confusing a pipeline's exit with pytest's exit, losing type definitions or caller relationships, and introducing more rereads. Existing explicit limits remain. The prompt must ask for actual test exit codes and source location references. Do not create a fixed mandatory test ritual. CPU acceptance requires preserving selected assertion/API/error evidence in every reviewed replay and showing no fabricated test pass.

**Complexity:** low for prompt prototype, moderate for robust diagnostic rendering tests. Runtime support is likely possible via existing run_command behavior, but a prompt instruction is not enforcement. Future GPU cost is evaluation only; no training cost. Primary falsifier: lower displayed output without fewer overflows or with degraded resolved count/evidence use.

## Priority 2: compact factual task state

**Mechanism:** retain issue constraints, implicated symbols/callers, current source diff, attempted edits, latest actual test result, unresolved hypotheses and remaining budget in ≤600–1,000 tokens. The size is a proposed cap. It is factual context, not generated private reasoning. Update at a meaningful phase transition rather than after every call.

**Observed cases:** multi-file compatibility/security features lose continuity at the context boundary; scratch-only or empty final patches need retained diff status. This does not prove prior compaction erased state; none occurred.

**Savings:** a summary only saves tokens if it **replaces** history in a supported fresh context. Adding a ledger to an ever-growing history increases tokens. As a hypothetical budget example, replacing 10,000 historical tokens with a 1,000-token state saves about 9,000 tokens before retrieval and prompt overhead; no such saving is measured here. A file-based ledger may cost extra charged calls and its own output history. Use hashes/paths for recoverable snippets, but retain the exact code needed to justify the next edit.

**Interface choices:** a declarative SequentialAgent with output_key/include_contents:none is documented in the repository, but prior localizer/fixer pipelines overran budget and scored poorly. Consider a bounded single handoff only if scorer schema permits it; do not introduce another unconstrained localizer. A mere prompt telling the same agent to forget history is not compaction.

**Complexity:** moderate for memory schema, high for reliable supported fresh-context transition and guardrails. No dataset required for initial inference ablation. Falsifier: invariant omissions, more rereads, extra calls or lower resolution with no context benefit.

## Priority 3: conditional diff and error recovery

**Mechanism:** before termination, use existing evidence to distinguish source changes, scratch-only changes and failed replacement attempts. When a command fails, interpret its actual error: grep exit1 may mean no match; assertion failure requires narrowing the replacement; a missing API argument may be an incomplete feature, not an environment excuse. Run a relevant check if needed and feasible. Do not repeatedly rerun unchanged commands or force editing at a fixed call number.

**Evidence:** ten failures have empty or scratch-only final patches; 141 CommandErrors and ten budget denials; partial constructor/symbol propagation appears in collection failures. Existing test-after-edit metrics are unreliable because Python edits and probes were missed. New audit records broader verification activity without treating a reproduction success as the hidden verifier.

**Savings:** avoids redundant commands/output; exact achievable tokens unknown. Measured repeated identical output is an upper bound for exact-output dedup alone, not all repetitive reasoning. Require state-aware transitions rather than hard bans (a repeated test is valid after a source change).

**Changes/data:** prompt-only policy first; recovery examples from independent tasks later. Never edit benchmark tests to force a pass. Complexity low/moderate; training unnecessary initially. Falsifier: more nonempty but incorrect patches, forced premature edits, or increased test frequency without resolution.

## Priority 4: source/API evidence retrieval, selectively

**Mechanism:** read the definition plus callers/configuration propagation when task scope demands it. For constructor/API features, inspect public API, delegated router/dependency path, return/schema behavior and consumers. Preserve a compact contract table for required arguments/status/header behavior. The current v14 prompt recommends search_similar_code with issue keywords, while docs/findings.md says this tool is an exact qualified-node lookup with no text encoder. Correcting that unsupported query advice is a small isolated candidate; four graph calls cannot establish its score impact. Source-local exact-name graph neighbors may help a known symbol when present; use normal narrow search when issue names do not resolve.

**Evidence:** 59/60 gold-file exposure means broad localization is near saturation; exposure is not understanding. Failures touch only one part of multi-file compatibility, SSE, security and dependency changes. v14 makes only four graph proposals, so the current data cannot estimate graph benefit. Earlier graph-heavy agents had overhead, exact-name lookup failures and unverified skill execution; KT1 was ambiguous, and the sibling-separation CPU pilot failed its preregistered bar.

**Savings:** selective retrieval may avoid broad outputs, but expands context if irrelevant neighbors are retained. No quantified benefit is claimed. Complexity moderate; no new graph runtime until a CPU audit shows marginal relevant evidence over narrow text retrieval. Falsifier: added calls/context without new contract evidence or matched-budget resolution gains.

## Priority 5: v14-compatible SFT, conditional on data

Use Design A anchored with authentic Design B recovery, after provenance, state-conditioning, effective-loss and full-rollout gates pass. Do not immediately train all available public data. No numeric score forecast. Planning cost: at historical ~280 input tokens/s, 1M scheduled input tokens is ~0.99 session hours before loading/validation/export; historical load varied from ~0.14 to ~0.62h, so a small 1M-token training pass might require roughly 1.5–2.5 session h /3–5 quota h with overhead. This is a scenario, not a quote or authorization. Separately budget serving and independent paired evaluation.

Complexity high because data/oracles, not LoRA export, are the unresolved prerequisite. Generalization may improve from repository diversity or worsen through shortcuts; both require new independent tasks. Reject a candidate if it passes one-step gates but increases early test frequency, empty patches or repeated calls in full rollouts.

## Priority 6: verified preference/rejection selection

Design C differs materially from SFT but has weak current support: no sufficiently diverse matched verified preference corpus is established. It requires reliable target-environment rewards and same-state comparability. CPU pair audits cost no GPU, but fresh candidate rollouts and preference training multiply resources. Estimate only after accepted pair counts and sequence lengths are known; unbounded online RL is rejected. Falsifier: labels fail environment consistency checks or rank superficial length/tool habits rather than issue resolution.

## Context mechanisms compared

| Mechanism | Relevant observed weakness | Expected saving and information risk | Feasibility |
|---|---|---|---|
| Early genuine compaction | History rejected at 24,577 input | Replacement can save large history volume; can lose exact edits/assertions | Strong engineering hypothesis; scorer hook unverified, local runner change alone insufficient |
| Exact-output dedup | Duplicate responses in some context cases | At most measured duplicate-output characters; pointers can lose self-contained evidence | Requires retained accessible content and supported history manipulation |
| Narrow output/head+tail+diagnostic extraction | Large unique outputs | Measured character-cap scenario; token savings pending tokenizer replay; middle failures may be lost | Best minimal prompt prototype |
| Factual ledger/fresh handoff | Multi-file state and patch retention | Savings only when replacing history; otherwise extra tokens | Declarative route known, budget and scorer checks needed |
| Dynamic output reservation | Fixed8192 reserve reduces prompt room | Potential extra prompt room equals reserve reduction; can truncate valid thinking/tool calls | Deferred: thinking-enabled completion distribution differs from v7; v11 reduction did not establish a gain |
| Dependency-aware retrieval | Partial multi-file repair | Could reduce broad reads or add noise; no measured delta | Use known-symbol targeted retrieval; avoid generic graph expansion |

The next concrete authorized artifact is this audit/handoff. Submission changes and runtime/adapter implementation remain future actions requiring the user's explicit permission.

## Generalization and data dependence by candidate

| Candidate | Dataset change | Generalization expectation to test | Competition dependency |
|---|---|---|---|
| Bounded output | None | Process rule may transfer across repos; source/error formats may defeat uniform caps | Existing commands; preserve actual failures and offline operation |
| State replacement | None initially | Repository-neutral facts may transfer; memory may omit unfamiliar invariants | Genuine supported fresh context, extra-call/time limits |
| Conditional recovery | None for prompt ablation; later authentic recovery corpus | Could improve unseen API/tool failures; forced rules can harm hard issues | No benchmark-test edits, no external tools at inference |
| Selective retrieval | None initially | Caller contracts may transfer; exact-name graph availability differs by repo | Graph existence/schema and matched tool budget |
| Verified SFT | Designs A/B | Diversity may improve transfer; template imitation may still regress | Accessible licensed data, fixed base/adapter format, offline serving |
| Verified preferences | Design C matched pairs | Depends on reward validity outside collection repos | Oracle reliability, teacher/data terms, rollout/training resources |
