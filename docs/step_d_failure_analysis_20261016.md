# Step D failure analysis — October 16, 2026

## Finding and confidence

**The immediate failure was absence of repository editing, rather than successful fixes disappearing during patch collection.** Step D’s recorded commands investigate code, run tests, or construct diagnostic reproductions. No `write_file` call or identifiable repository-source mutation appears in the 114 traces. All 81 explicit `submit_patch` responses successfully report **zero patch bytes and zero changed files**. All 114 final results independently report zero patch bytes.

This is strong evidence of a failure to transition from diagnosis/reproduction to implementation under the Step C adapter. It does **not** identify the underlying training defect conclusively. The leading upstream hypothesis is a learned behavior/control-flow regression associated with the public-data SFT, potentially involving window composition or train/serve rendering. Token-level supervision and counterfactual model behavior have not been verified in this analysis.

Budget and context failures amplify the problem, but do not explain it universally: **77 tasks reached verifier execution without a top-level agent error and still had empty patches.** The existing serving health checks test activity, not whether the agent edits source code or responds correctly to an empty submission.

**Scope:** retrospective, read-only analysis of preserved artifacts and source, using Python standard-library JSON, archive, AST, shell-token and binary-array inspection. No notebook, D0, retraining, inference, local PyTorch/model workload, leaderboard submission, credential access or Git push was performed. A/B/C/D outputs were not modified. Scratch analysis was confined to `/tmp/step_d_diagnosis_20261016/`; this document is the durable diagnosis. Proposed tests and fixes below were **not implemented or run**.

## 1. Evidence and methodology

Project instructions, the complete post-training guide, HQ, experiment log, completed D report, and D validation/behavior summaries were reviewed. The roadmap was also read as required by `CLAUDE.md`.

Primary local evidence:

- [Step D final report](step_d_d0_20261016_report.md), local-only validation (`data/runs/atul_step_d_20261016/validation_summary.json`), and earlier behavior summary (`data/runs/atul_step_d_20261016/behavior_summary.json`).
- Local-only D task results (`data/runs/atul_step_d_20261016/extracted/eval_v8_conv_lora/rep0/merged/task_results.jsonl`), all 114 traces, task logs and 77 verifier logs in that same `merged/` directory; original archive in `outputs/eval_v8_conv_lora.tgz`.
- Step B’s original `eval_v8_lora_smoke.tgz`, read directly in memory, plus its eight task results, configuration and submission archives in `data/runs/atul_step_b_20261015_v1/`.
- Historical v7 summaries for all five 54-task repetitions under `data/runs/rs_v7_valid/x/eval_v7_noedit/rep*/analysis/summary.json`, and the earlier 129-task summary under `data/runs/eval_v7_full/x/`.
- Captured B/D evalkit source and zipped submission configuration; preserved C `trainkit/` source, A/C reports and NPZ inputs; current conversion/rendering/tokenization code.

Raw experimental evidence remains local and is intentionally excluded from this documentation branch. For brevity, artifact paths below are relative to `/home/atul-baliyan/Desktop/gemma4-swe-agent/`. Task IDs are identifiers for reproduction, not quoted issue contents. No raw commands, model replies, logs or trajectory excerpts are reproduced.

### Counting correction: proposals are not executions

The original report’s 5,684 `run_command` count describes **recorded model tool-call proposals**. Eight tasks have a large first-turn batch, and many proposed calls never executed. A second issue affects the old analyzer: it associates observations with calls on the same step, whereas a multi-call response can have its first observation on the agent step and subsequent observations on following system steps. It also only recognizes `edit_file`/`write_file` as edits, missing Python edits through `run_command`.

This analysis traversed calls and responses in saved order, keeping a pending-call queue. The two completed multi-call batches contain exactly as many observation records as proposed calls; their ordered outcomes end with the expected budget rejections. These observations lack explicit response call IDs, so this is an **ordered-association reconstruction**, rather than an independent execution transcript. It reconciles exactly with the harness totals:

| Measurement | Count | Interpretation |
|---|---:|---|
| Recorded `run_command` proposals | 5,684 | Includes unexecuted and budget-rejected calls |
| Recorded `submit_patch` calls | 81 | All have their own successful empty-patch response |
| `run_command` response records | 4,380 | Includes budget rejections |
| Budget-rejected `run_command` responses | 389 | Tool did not execute the proposed command |
| Non-budget `run_command` responses | 3,991 | 2,817 successful; 1,174 CommandError responses |
| Calls counted by harness in six batched timeout tasks without saved observations | 115 | Charged-call count known; individual completion/outcomes not retained |
| Sum of task-results `tool_calls` | **4,106** | **3,991 + 115**; consistent with submit being exempt from this budget |
| Proposals without a saved observation | 1,304 | All in those six batched timeout traces; not 1,304 proven executions |

The previous analyzer’s 1,142 CommandError / 174 BudgetExceeded counts undercount the ordered observations; revised counts are **1,174 / 389**. Original files and summaries remain unchanged. Category counts such as “commands mentioning pytest” remain counts of proposals unless explicitly marked otherwise. Saved traces do not support reconstructing every unsaved command outcome.

## 2. Answers to the main failure questions

### 2.1 Why were all final patches empty?

The observable path is:

1. The agent reads/searches or runs tests and reproduction diagnostics.
2. It does not issue an identifiable repository-source edit.
3. In 81 tasks, it explicitly submits and receives an empty-diff result.
4. In the other 33, it exits on context failure or a batched session timeout before explicit submission.
5. The final evaluator records zero patch bytes in every task.

All 77 tasks with no top-level agent error explicitly submitted an empty patch. Four additional tasks submitted empty patches and subsequently failed: one context overflow, two turn limits and one session timeout. The 33 without explicit submission are the other 27 context failures and six batched timeouts. Thus missing `submit_patch` calls or exceptions alone cannot account for the result.

This establishes the **proximal behavior failure**. It does not prove whether defective data, rendering, optimization, or exposure to an unfamiliar policy state caused it.

### 2.2 Did agents try to edit, or only investigate and test?

No repository fix was identified. Tool selection is unusually narrow: **5,684 `run_command`, 81 `submit_patch`, and zero calls to every other declared tool**, including `write_file`, `get_status`, graph tools and the analyzer sub-agent. Commands mentioning pytest/unittest occur in **109/114 tasks**, including 74 of the 77 normally completed tasks. These are agent-side test commands, distinct from the final verifier tests.

A broad mutation-candidate sweep included Python write methods, open-for-write patterns, shell redirection, shell file mutators, patch commands and Git restoration. Its 54 candidate proposals reduce to:

| Task | Candidate count | Inspection finding |
|---|---:|---|
| `requests_7205` | 27 | Temporary reproduction-fixture writes, followed by temporary-file cleanup; **not source edits**. Ordered responses: 12 success, 15 BudgetExceeded. |
| `rich_3454` | 25 | A read-only regex/printing diagnostic. A comparison operator caused the broad redirection heuristic to flag it. All 25 responses are shell errors, exit 2. No write operation in its parsed Python body. |
| `requests_7315`, `requests_7433` | 1 each | Generated command-label/redirection-like syntax in huge batches; no write operation identified and no corresponding saved response. Not evidence of a successful edit. |

The narrower 27 “edit-intent” hits in the earlier behavior summary all belong to the temporary fixture case. They were false positives for **repository editing**. A broader lexical filesystem-write sweep finds write/unlink tokens only in that same temporary-fixture task. No shell source-mutator or Git revert command was found.

These checks are substantially stronger than the original edit regex, but remain a static audit of recorded commands. They cannot prove the absence of every possible side effect of arbitrary imported code, nor reconstruct lost sandbox files after teardown.

### 2.3 Were editing tools unavailable or incorrectly called?

**`edit_file` was intentionally absent**, as in the successful v7 design. That is not a new D configuration defect. The root agent declares `write_file`, and its prompt explicitly describes exact-snippet replacement through a Python script in `run_command`. Both editing routes were available by configuration; D simply never called `write_file` and did not generate such a source-replacement script.

There are no attempted `edit_file` calls, unknown edit-tool errors, `write_file` argument errors or permission-denied write responses to explain a universal failure. All recorded `run_command` argument mappings use the expected `command` key. Two proposals put a literal command label inside the command **value**, illustrating that valid argument keys do not guarantee a usable shell command.

Actual C adapter routing is visible, rather than inferred solely from attachment: **all 4,251 D trace steps with `model_name` specify `conv_lora`**. B’s corresponding 317 steps also specify `conv_lora`, backed by its different Step A adapter. The D root configuration declares the adapter; the unused analysis sub-agent does not. Correct routing and file identity do not guarantee good learned behavior.

A separate inference smoke would be required to prove every declared tool is callable with the C policy, but no evidence here supports a missing editing-tool interface as the main cause.

### 2.4 Did `run_command` make any successful modifications?

**Temporary fixture writes: evidence yes. Repository-source modifications: none identified.** The 12 successful fixture calls in `requests_7205` precede that batch’s budget rejections. Their code creates a temporary file, writes reproduction data, and cleans it up. Their success must not be presented as 12 attempted code fixes or as source changes lost by the collector.

By contrast, B contains seven successful Python source-edit commands across three tasks, with nonempty final patches. Its eighth edit proposal was rejected by the tool budget. This supplies an observed example of the intended v7 editing path, without running new code.

### 2.5 Were edits lost, reverted or excluded by collection?

There is **no affirmative evidence of that mechanism in D**:

- No source-targeted editing command was found to precede a disappearing diff.
- No source reversion, Git reset/checkout/restore or source cleanup command was identified.
- All 81 live submission results already report zero patch bytes and zero changed files, before archive packaging.
- No patch files exist, consistent with the final zero-byte results rather than an archive that omitted known nonempty patches.
- B/D `runner.py`, `compat.py`, `submission.py`, `server.py`, `main.py` and `report.py` are byte-identical. Their `submission_src.zip` file sets and contents are also byte-identical.
- Under that same evaluation code, B retained three source patches: 587 bytes (`fastapi_15280`), 536 (`requests_7328`) and 2,845 (`httpx_3672`). The requests task retained its patch even without an explicit `submit_patch` call, evidence of an end-of-task capture path in that run.

This weighs strongly against a generic broken collector. It does not exclude an unobserved D-specific sandbox defect absolutely: final worktrees and complete per-execution filesystem deltas were not retained. The installed official collector implementation was not available among the inspected local files; no claim about all of its filtering rules is made. An independent CPU collector replay is a proposed test, not a completed test.

### 2.6 Why submit an empty patch?

The tool accepts a zero-length diff and reports it as an ordinary successful tool response. The prompt asks the agent to check positive patch size, but that instruction is not enforced by the activity health checks. D ignored that feedback:

- **81 submissions in 81 tasks; all `status=ok`, `patch_size=0`, `files_changed=0`.**
- Only four have any subsequent tool activity; none subsequently creates a source patch.
- Seventy-seven have a later text-only agent response.
- A conservative phrase audit finds explicit completion/fix claim patterns in 15 task final messages despite empty patches; this is a linguistic indicator, not a success metric. For example, sampled `fastapi_11194` contains such a claim after an empty submission.

Likely explanation: “finish/submit” became a learned endpoint detached from the state of the repository and the negative empty-diff feedback. Training conversion synthesizes submission feedback from the teacher’s stored patch, rather than executing the converted trajectory and measuring its actual diff. That is a concrete data-validation gap and a plausible contributor, **not proof** that the synthetic responses were wrong in the 200 actual training tasks.

### 2.7 Did loops, context or budgets prevent editing?

They prevented additional opportunities, but did not erase successful code fixes.

- **66/114 tasks** record an identical command at least three times. There are 1,748 repeated command proposals beyond the first occurrence, including 825 consecutive command repeats. The original all-tool repeat share, 1,748/5,765 = **30.3%**, is still a valid proposal statistic. It is not an execution repetition rate.
- Eight first-turn batches contain **55, 153, 161, 235, 240, 250, 252 or 253 calls**. Six end in session timeout without saved observations; two overflow context after the responses. The agent cannot adapt later calls in these pre-generated batches to earlier tool feedback.
- All **28 context failures have the identical error text**, reporting **24,577 input tokens + 8,192 completion allowance = 32,769**, against a 32,768 limit. This is a concrete one-token request-budget boundary. It suggests a truncation/reservation edge worthy of audit; the saved files do not identify which component introduced it. It is not proof that 8,192 completion tokens were generated.
- Stored prompt-token maxima are from completed trace events, not necessarily the rejected request. For example, a large first-turn batch can have a low recorded prompt size and still overflow when its accumulated responses are sent back. The median maximum over all D trace steps is 16,634 tokens.
- **389 budget-rejection responses occur in six tasks**, including the two 100-turn failures. In `fastapi_14786`, after an empty submission, the agent issues another 59 test proposals that receive BudgetExceeded responses. Giving more calls would likely prolong the same behavior unless policy or control flow changes.
- Seven task-level session timeouts and two turn-limit errors explain nine task failures; 28 context failures explain another 28. They cannot explain the other 77 empty patches.

Do not treat context expansion, more tools, or larger budgets as a demonstrated fix. None supplies the missing source-edit behavior.

### 2.8 Did LoRA change editing or tool-use behavior?

The preserved runs show a marked **association**, especially on B’s same eight IDs: successful source editing appears under the Step A adapter and disappears under Step C. Calls remain correctly routed to the adapter, and source/evalkit configuration is identical. D instead makes agent-side test proposals on all eight, while B has none containing pytest/unittest. This is consistent with a learned policy shift toward reproduction and inspection.

It is **not a controlled estimate** of the effect of “training”: A and C used different data, different numbers of optimizer steps, separate stochastic runs and different training overlap. D’s new 60 tasks have no fresh baseline. The evidence cannot isolate data conversion, windowing, optimization, quantization transfer, or random generation effects as the unique upstream cause.

## 3. Representative failures: reproducible sample

Selection uses only task results and trace structure, not issue content or success claims. Choose the lexicographically first ID in each of these strata: single-call context failure; timeout; turn limit; verifier exits 1, 2 and 124; normally completed empty submission. Add the single-call task with the largest identical-command count (lexicographically largest ID on a tie), the two exceptional large-batch context cases, the read-only mutation false positive, and B’s three nonempty-patch IDs. Deduplicate. This produces the following **12 tasks**.

“Recorded/harness” distinguishes proposed tool calls, including submit, from the result’s charged call counter. Verifier `-1` means not run. Each sample’s final patch size is zero.

| Task | Category | Recorded / harness | Verifier | Private inspection finding |
|---|---|---:|---:|---|
| `fastapi_13713` | Single-call context failure | 28 / 28 | -1 | 25 read/search proposals, three test proposals, no repeated identical command and no edit; context growth can occur without an exact-repeat loop. |
| `requests_6644` | Batched session timeout | 161 / 27 | -1 | Entire first response queues 161 calls, 148 mentioning tests; no observations survived timeout. No source edit found; outcomes of individual queued calls cannot be asserted. |
| `fastapi_14786` | Turn limit and strongest single-call repetition | 100 / 40 | -1 | Empty submission followed by 59 budget-rejected test calls. One identical command appears 60 times. Never edits. |
| `fastapi_11194` | Normal completion, verifier exit 1, empty submission | 41 / 40 | 1 | 38 read/search and two test proposals, one command repeated 11 times, then zero-byte submission and a completion claim. No edit. |
| `fastapi_14262` | Normal completion, collection failure | 41 / 40 | 2 | 38 read/search and two test proposals, no identical command repeated; empty submission. Collection failure coexists with, but does not cause, absent editing. |
| `rich_3480` | Normal agent completion, verifier timeout | 41 / 40 | 124 | Reads dominate; one command appears 34 times. Empty submission precedes the verifier timeout. |
| `requests_7205` | Batched context failure and apparent writes | 55 / 40 | -1 | 27 temporary-fixture proposals; 12 successful responses followed by 15 budget rejections. Fixture writes/cleanup, no source replacement. |
| `rich_3454` | Repeated shell failure, mutation false positive | 41 / 40 | 1 | Same read-only regex diagnostic fails 25 times with shell exit 2, then empty submission. Python body contains no file write. |
| `requests_7505` | Batched context and budget failure | 240 / 40 | -1 | 240 first-turn calls; first 40 responses include 32 command errors, remaining 200 are budget rejections. No edit. |
| `requests_7328` | Same ID solved in B | 41 / 40 | 1 | D: 33 read/search and seven test proposals, empty submit. B: one successful source replacement, 536-byte patch, resolved. |
| `httpx_3672` | Same ID solved in B | 41 / 40 | 1 | D: seven test proposals, 33 command errors, no edit, empty submit. B: five successful source edits and a 2,845-byte patch, resolved. |
| `fastapi_15280` | Same ID edited but unresolved in B | 41 / 40 | 1 | D: 34 command errors, no edit. B: successful source edit, nonempty 587-byte patch, still unresolved. Editing and correctness are distinct gates. |

These findings describe actual generated actions and recorded outcomes. The read/test classifications are lexical; the targeted mutation cases additionally received shell-token/Python-AST inspection without executing their code. No sample was used to infer correctness from the agent’s narrative.

## 4. Comparison with earlier runs

### Historical v7, same original 54-task membership

Only the five aggregate v7 analysis summaries were locally available in the inspected historical folders; their raw traces/per-task results were not. This limits independent verification and paired statistics. The original 129-task run uses a different set and environment, so it is not used as a matched performance comparison.

| Metric | Historical v7, five passes × old54 | D restricted to old54 |
|---|---|---|
| Resolved | 16, 16, 16, 16, 18 | 0/54 |
| Empty patches | 29.6–42.6% | 100% |
| `write_file` calls | 2–5 per pass | 0 |
| Explicit submission | 77.8–87.0% of tasks | 38/54 = 70.4% |
| Tasks with ≥3 identical calls | 50.0–64.8% | 31/54 = 57.4% |
| Context overflow | 0–3.7% | 12/54 = 22.2% |
| Context pressure | Median 0.35–0.40 in stored analyzer summaries | Median maximum prompt 17,966 tokens, approximately 0.548 of 32,768 |

The last row is an approximate comparison: median-of-task maxima recomputed for D versus the old analyzer’s stored context-pressure statistic. It is not an exact measure of failed-request length. Likewise, the analyzer’s near-zero “tested after last edit” values for historical v7 are **not reliable evidence of no Python editing**, since its edit detector omits `run_command` edits. Use nonempty patches as the stronger historical edit outcome.

Repetition was already common before C and D’s old54 loop rate lies within the historical range. Therefore the diagnosis is not “LoRA universally increased exact repetition.” The clearest differences are **loss of nonempty patches, narrower tool choice, and more context failures**. A paired McNemar statistic for historical old54 cannot be reconstructed from aggregate summaries alone.

### Step B versus D on B’s eight exact IDs

| Metric | B: Step A adapter, eight tasks | D: Step C adapter, same eight IDs |
|---|---:|---:|
| Nonempty patches | 3/8 | 0/8 |
| Resolved | 2/8 | 0/8 |
| Successful source-edit commands identified | 7 across three tasks | 0 |
| `write_file` | 0 | 0 |
| `run_command` proposals | 244 | 393 |
| Agent commands mentioning pytest/unittest | 0 | 214, on all eight tasks |
| Final verifier reached | 7/8 | 5/8 |
| `submit_patch` proposals / tasks | 65 / 7 | 5 / 5 |
| Context errors | 0 | 2 |
| Exact-repeat share of recorded calls | 68.7% | 12.6% |

B’s submission count is dominated by **59 empty submissions in `rich_2943`**, not broad success. B itself has poor feedback handling and repetition. D’s lower repeat share on these eight IDs does not mean better behavior: it performs more varied diagnostics without implementing a fix. Both runs’ successful source patches or lack thereof are stronger evidence than their narrative summaries.

The two resolution discordances are B-only wins and there are zero D-only wins. A descriptive two-sided exact McNemar calculation on these eight IDs gives **p=0.50** (two discordant pairs). This small, overlapping-training smoke sample is not a controlled generalization comparison or evidence of a significant causal effect. No comparison on D’s new60 can be made because D0 was never run.

## 5. Ranked causal assessment

| Rank | Candidate cause | Supporting evidence | What remains unproven |
|---|---|---|---|
| 1 — high confidence as the immediate mechanism | Investigation/testing policy fails to emit a source edit or condition completion on a diff | No source-edit command, no `write_file`, 81 explicit empty submissions, all 77 normally completed tasks empty. B can edit through identical scaffolding. | Why this behavior was learned; counterfactual base/final-checkpoint generations are unavailable. |
| 2 — leading upstream hypothesis | Public-data SFT learns unsuitable phase/control-flow behavior; windowing or token/response boundaries contribute | D replaces B’s edit scripts with tests/reads on the same IDs; eight giant pre-feedback batches. C has 122 updates versus A’s five and uses converted external data rather than own-agent traces. Windowing repeats the original task prefix but drops preceding action/state history. | Actual per-window edit-token distribution, boundary/stop-token supervision, and whether training arrays faithfully encode the intended trajectories. |
| 3 — strong contributor, not universal cause | Budget-feedback noncompliance, batching and a context reservation boundary consume remaining opportunities | 389 rejected calls, six batches with no observations after timeout, two large batches that overflow, all 28 overflows at 24,577 + 8,192. Some calls continue after empty submission. | Which component creates the one-token boundary and how much a corrected budget policy improves edits. |
| 4 — plausible upstream contributor | Over-adaptation/optimization or NF4-to-served transfer damages a formerly useful policy | C’s held-out loss falls 0.7718 → 0.2882 while code-fix behavior collapses; rank-16 LoRA targets all 410 language modules, lr 1e-4. Existing guide records base-format disagreement. | A loss curve does not diagnose overfitting; no base/C-checkpoint counterfactual, per-category loss, or delta-strength test exists. A used the same serving transfer path successfully for some edits. |
| 5 — low support as main cause | Editing tools missing, adapter not routed, or generic collector discards edits | Contrary evidence: `write_file` declared; D steps route to `conv_lora`; final C hash verified; B/D scaffold/evalkit identical and B retains three patches. No D source edit is found to be lost. | A narrowly scoped D-only filesystem/collector defect cannot be ruled out without retained worktrees or a replay. |

### Concrete static training risks, not established root causes

1. **Conversion fabricates execution outcomes.** `posttrain/convert_public.py:199` converts editor operations into Python commands, but synthesizes their success from teacher response text. At line 229 it synthesizes submission patch size from the original stored patch. The converter does not replay the rewritten commands on the source commit and compare the resulting diff. Earlier quoting round-trip checks are not equivalent to full filesystem/patch fidelity.
2. **Training windows can lose the edit’s causal state.** `posttrain/render_atif.py:107` retains the original task message while moving to later windows without a summary of prior edits or completed actions. Later test/finish windows can therefore present “fresh issue → verify/finish” targets. The precise number and supervised-token weight of such windows is unknown without the original rendered text/tokenizer. The source can also emit an overlength singleton, and `tokenize_windows.py:45` silently slices to 8,192; whether that affected the actual C data has not been established.
3. **Train/serve tool declarations are not completely identical by construction.** `render_atif.py:89` explicitly omits the declared analyzer `agent_tool` schema. The actual v7 agent includes it. This is a real prompt-fidelity gap, but omission of this rarely used sub-agent does not alone explain why the still-declared `write_file` and shell-edit route vanish.
4. **Loss masking needs token-level confirmation.** `model_spans` uses textual delimiter searches, and tokenization assigns labels using token start offsets. These are places to audit tool-result exclusion, end-of-call/turn boundaries and special-token offsets. No mask corruption has been demonstrated. A stdlib NPZ audit finds 1,017 windows, 7,372,758 input tokens, 1,170,308 supervised tokens, masks containing only 0/1, no zero-supervision windows, and the expected 974/43 train/validation split. These structural checks do not establish that the *right* tokens are supervised.
5. **Held-out language loss is an insufficient release gate.** Every C validation point can improve while action selection, empty-patch recovery and source editing deteriorate. The five serving checks only require some results, some tools, saved traces, a non-error result and a test execution. They place no constraint on source edits, nonempty diffs, batching, or feedback compliance.

The raw converted 200 trajectories, rendered text windows and original tokenizer vocabulary were not located in the inspected local project/default tokenizer paths. The available NPZ and source establish structure and intended transformations, but cannot support a decoded, per-action supervision audit here. Obtain those exact preprocessing artifacts before asserting a specific data or mask bug. Captured C `data.py`, `worker.py`, `model.py` and `export.py` match current repository source byte-for-byte; this does not establish provenance for every preprocessing file.

## 6. Recommended direction and falsifiable next tests

**Prioritize the training-data/rendering audit and a source-edit behavioral gate before any new evaluation or retraining.** A full D0 baseline would not explain why no edits were generated, and the remaining quota is below its safety gate. Larger call/context budgets are particularly poor first remedies.

All rows below are proposals requiring authorization before implementation or experiments. GPU estimates are planning ranges in **Kaggle quota-hours**, using the observed approximately 2× rate. A GPU number is not a request to start work. Costs exclude any full retraining unless stated.

| Priority / proposed change or test | Expected benefit | Supporting evidence and falsifiable criterion | Code/data changes needed | CPU-only? | Future GPU estimate |
|---|---|---|---|---|---|
| P1: Replay converted teacher edits against disposable source snapshots and compare final diffs | Detect invalid “successful edit” targets and false submission feedback before more training | Converted edit success/submission size are synthetic. Require each claimed successful edit to actually change the intended tracked source file and final diff to match the intended teacher patch, with tolerated differences explicitly defined. | A separate data-audit utility; exact original trajectories, source commits and snapshots. Do not alter original training data in place. | Yes; no model execution. Snapshots may need retrieval. | **0** |
| P1: Audit exact rendered windows, masks and train/serve prompts | Test the strongest unverified explanations for batching and missing edits | Require byte/token comparisons for a two-call conversation, tool responses masked out, correct response/turn boundaries, and accounting of how many supervised tokens/actions are edits versus reads/tests/finish. Reject silent overlength slicing. | Restore original tokenizer and rendered JSONL; add an isolated audit, then fix renderer/tokenizer only if a discrepancy is demonstrated. Include the actual compiled tool list and analyzer schema. | Yes, tokenizer/data processing without PyTorch or model weights. | **0** |
| P1: Reconstruct window-state/phase distribution and redesign only if warranted | Prevent fresh-task windows that teach immediate verify/finish without an edit | Count first supervised action per window, edit-containing windows, prior-state-dependent targets and repeated test/finish tokens. Hypothesis is weakened if edits are well represented and every later target has sufficient context. | Data selection/window packing or explicit prior-state summaries; preserve original data and produce a separately versioned candidate. | Yes for audit, design and data validation. Training is separate. | **0** for audit |
| P2: Independent shell-edit → diff → submission replay using official collector code | Close the remaining infrastructure uncertainty cheaply | In disposable worktrees, an existing-file Python edit must produce a positive diff and captured patch; a temporary fixture must not. Test untracked/staged/reverted cases separately. | Obtain the exact preserved scorer implementation/wheel; isolated regression fixtures, no changes to official scorer semantics. | Yes, no model or Kaggle notebook needed. | **0** |
| P2: Validate request-size reservation and limit batching in a future candidate | Avoid one-token overflows and pre-generated feedback-blind batches | Assert input + output reserve + safety margin ≤32,768 for every recorded/request fixture; reject requests violating it. Require one tool call before awaiting observation in a short behavioral probe. | Inspect actual request construction/token accounting; candidate prompt/decoding controls only where supported. A smaller output ceiling is an ablation, not a proven remedy. | Reservation arithmetic/request-fixture tests: yes. Learned compliance: no. | **0** static; roughly **0.4–1.0** for a narrowly bounded future generation probe including startup |
| P2: Add a cheap pre-evaluation behavior gate | Catch another 114-empty-patch candidate before a full run | Require at least one real tracked-source edit, nonempty patch feedback and recovery from an intentionally empty diff on a small fixed set; measure read→edit→verify→submit, batch size and repeats. This is a rejection gate, not a guaranteed performance fix. | Separate candidate preflight/reporting plus prompt criteria; do not silently change official `submit_patch` semantics or count narrative claims. | Gate code/fixtures: yes. Model policy validation: no. | **0** for static gate; roughly **1.2–2.0** for a future 4-task-per-arm base/C probe, one pass each, bounded and separately approved |
| P3: Compare base, an early C checkpoint and final C at fixed decision prefixes | Separate policy damage/over-adaptation from a universally bad scaffold | Hold prompts, tools and decision points fixed; measure next action, valid source-edit proposals, feedback use and batching. Earlier checkpoint restoring edits would support a training-strength explanation; failure at every checkpoint would weaken it. | A new bounded inference-only probe with verified routing and unchanged original adapters. Checkpoint selection must be specified in advance, not swept automatically. | Prefix preparation and action scoring: yes; generation: no. | Roughly **0.5–1.5** for a small fixed-prefix probe; not a full issue-solving evaluation |
| P4: Retrain only after identifying a data/objective defect | Restore edit-conversion behavior rather than optimize already-misleading aggregate loss | Rebuilt data must pass CPU fidelity checks; held-out evaluation must report edit-action/category loss and behavior gates, not just mean CE. Compare one corrected change at a time. | Depending on evidence: phase-balanced/causally valid windows, correct rendering/masks, or less aggressive adaptation; specify new candidate and retain C unchanged. | Data/config checks: yes. Training: no. | About **1.4–3.0** for a future 40–80-window pipeline probe based on A; serving validation costs extra. A full C-sized repeat previously used **15.64** quota h. |

Interpret the generation-probe estimates cautiously: vLLM startup alone was 437 s in D, full sandbox tasks can take 4.5 minutes each, and restoration/availability adds overhead. Fixed-prefix generation is not proof of end-to-end solving. No proposed GPU test should begin merely because its estimate appears to fit the last 2.88 h.

**Recommended authorization boundary:** first obtain the exact preprocessing artifacts and approve the CPU-only audits/replays. Use their results to choose one specific corrective change or one small pre-registered generation comparison. Do not restore `edit_file`, increase budgets, select another checkpoint, resubmit D, run D0, or retrain on the basis of this document alone.

## 7. Reproduction and evidence limits

To reproduce the sample, apply the selection rules in §3 to D’s task results and traces. Count repeated commands using the exact `command` string, without normalization. Count recorded call proposals separately from `task_results.tool_calls`; pair ordered observations using the pending queue described in §1. The 106 single-call-per-step tasks require no multi-call pairing assumption. Inspect B directly from its original tar archive and compare source-zip members and six evalkit files byte-for-byte. Read historical v7 summaries without treating their aggregate percentages as per-task records. Do not execute any saved command during analysis.

SHA-256 evidence fingerprints:

| Artifact | SHA-256 |
|---|---|
| D `merged/task_results.jsonl` | `d7251a0087abb0871768c40683a2a2be6f1ce20af9c1fdb3c08a0a6aa2c906c1` |
| D `outputs/eval_v8_conv_lora.tgz` | `fef49857ff7a0c5faa221b77c47f7b34493bfc952c458b0f4a67ddbdf44f70b1` |
| B `eval_v8_lora_smoke.tgz` | `4d7c0cfb77f7b7f1735552d61d289beacf4335811535d82cd8b99d5c334af44e` |
| Original `train_public_200.npz` | `b3f51bc449c31de2a3b163f98c5fdec2725b4427bde15625af7b4c75447d2488` |

D used the previously verified C adapter hash `a116b6be160fffcc98cdd2a4ab6cfaffafcfc54ac6709a8a5d95d95ebab544c6`; its serving identity is not in dispute. No model outputs were regenerated. No retained final sandbox worktrees, full execution transcript for the six aborted batches, decoded C training windows or fresh new60 baseline were available. Those gaps bound the conclusions above.

**Disposition:** diagnosis complete; no fix, experiment or new submission authorized by this analysis. D0 remains unexecuted. Await authorization.
