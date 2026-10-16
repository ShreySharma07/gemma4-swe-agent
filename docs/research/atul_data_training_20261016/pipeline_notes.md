# Agent 2 — conversion, rendering and supervision audit

Atul Baliyan research contribution, 16 October 2026. Read the complete Step D diagnosis before investigating; also reviewed CLAUDE, roadmap, experiment history, HQ, training guide and C/D reports. This is static source/artifact inventory research, not implementation or training. No recorded agent commands were executed, no torch/model loaded, and no 114-task investigation repeated.

## Main finding

The source contains real fidelity hazards, but none is yet demonstrated to have caused C's missing-edit policy. Exact C action distribution, edit-token share and response/stop-token coverage remain unavailable without the historical tokenizer and rendered examples. Existing NPZ structure checks do not establish semantic label correctness. The strongest next action is recover those artifacts, compare their rendering with the exact serving stack, and inspect state/action weights before training.

### 1. Conversion supplies claims of execution, not execution evidence

`posttrain/convert_public.py:145–241` maps teacher tool output into new tool responses without executing the newly generated command against its base snapshot.

- `create` always emits `status=ok` and byte count from the proposed contents (lines 195–198), even if the teacher result represents a failed creation. Original OpenHands creation semantics and the destination `write_file` overwrite semantics also need comparison.
- `str_replace` calls a generated assert-and-replace script and labels it successful when the teacher output contains a phrase (199–207). The script requires exactly one occurrence. Success is not established by the separately reported string quoting round-trip check.
- `insert` synthesizes an `IndexError` on unsuccessful teacher feedback (208–213), while its generated Python uses `list.insert` (80–84). Python clamps out-of-range insertion positions; it does not generate that index error. This is a static semantic mismatch for rejected insert indices, not a measured frequency in C.
- Bash conversion defaults missing exit codes to zero (166–173). Missing observations can therefore become successful empty-output commands. `obs_by_id` indexes by ID without detecting duplicates, and missing records are replaced by empty strings (145–154).
- Paths are directly interpolated into Python quoted strings and shell commands (74–84, 183–193), without a serialization/quoting layer. Quotes, backslashes or shell-significant filenames can change behavior. Frequency in C unknown.
- Absolute root replacement and interpreter rewriting (136–143) also change text in commands/observations, not just structured paths. Stateful shell assumptions, cwd changes, helper files and environment setup require replay.
- Final submission is fabricated from `row.model_patch`, independently of the filesystem created by converted calls (227–233). The converter neither checks a tracked source diff nor confirms patch/test equivalence. A nonempty stored patch does not validate a converted trajectory.
- Filtering checks resolved/submit, excluded repositories and call budget; it does not prohibit test-only edits or prove successful source mutation. Current system instructions explicitly prohibit tests changes. Thus teacher success can still conflict with destination policy.
- Original assistant tool-call batches are serialized into alternating call/response steps. That inserts observations before later calls originally proposed without feedback. Actual C prevalence is unknown; preserve original decision-group IDs rather than assuming each call was reactive.

**Contrary evidence:** The recorded 50-trajectory pilot passed tool argument drift and edit-literal round-trip checks; the converter does retain many genuine edit programs. C used resolved teacher tasks. These weaken a claim that all edit targets are invalid, but cannot establish environment fidelity.

**Cheapest falsification:** Restore exact rows and snapshots, replay *converted* edit operations in disposable isolated worktrees, preserve before/after tracked-file hashes and diffs, compare teacher and converted final patch semantics, and separately validate synthetic success/error labels. Start with all distinct editor operations and missing-footer/failed-call cases, then audit all selected 200 tasks before admitting them to training. A complete replay with faithful observations strongly weakens this causal hypothesis.

**Compute/gate:** Zero GPU. String/AST validation minutes; full tests/snapshot setup may take hours to days of CPU and storage. Quarantine any example with fabricated success contradicting replay, missing observation/ID, source-less final diff, unexplained changed file, or unmatched patch. Require 100% of retained claimed edits and final submissions to have replay provenance. This verifies data, not that retraining will help.

### 2. State loss is definite by construction; harmful phase bias is unmeasured

`posttrain/render_atif.py:107–127` creates each window from the initial task message plus a consecutive segment of assistant/tool steps. Earlier reads, edits, failing tests and remaining budgets do not survive unless coincidentally repeated in the segment. Later windows therefore reuse a fresh-task prompt while targeting later-stage behavior. The code offers no prior-diff or state capsule. This is an established information omission; actual frequency of test/finish-first windows or missing indispensable edit context is unknown.

The renderer preserves an assistant-plus-response step within a normal window. That is useful and should be retained. However an oversized prefix/singleton is emitted rather than rejected (120–124); tokenization silently slices every sequence at 8,192 (tokenize_windows.py:45). This can remove response closures, editing actions or final state in general. However the exact C NPZ maximum length is **8,191**, verified in this research with stdlib binary-array parsing. Given the inspected slicing implementation, any clipped row would be exactly 8,192 tokens. Thus hard clipping by this tokenizer is excluded for C conditional on matching preprocessing code provenance; retain it as a future robustness defect, not a leading explanation of this run.

Length estimation uses `tok(trial)` with default special-token behavior; final tokenization explicitly uses `add_special_tokens=False` (tokenize_windows.py:37). Depending on the exact tokenizer configuration, render length and actual training length can disagree. This requires checking the historical tokenizer, not assuming duplicate BOS tokens entered training.

**Cheapest falsification:** Reconstruct each window's first target phase, prior action dependencies, observed file state and first-edit position; report count and effective objective weight of windows that begin test/submit without showing the prior edit. Confirm the original renderer/tokenizer provenance; C has no 8,192-token clipped row. For future data, reject any row with actual length >8,192. If every later target is adequately grounded and no relevant targets truncate, the principal windowing explanation is weakened.

**Correction alternatives:** Prefer verified short complete workflows fitting the 8k envelope; for longer workflows use recent observation history plus an environment-derived state capsule containing actual changed-source diff, known file excerpts, latest test outcome and remaining budget. Summaries must be computed from observations, contain no future solution information, and reflect a state the serving policy can actually receive. Training on arbitrary new summaries without a matching inference policy creates another format mismatch. Decision-point records can supervise one next action while preserving complete available context, avoiding repeated loss on old targets; they may cost more prefix tokens. Do not blindly discard all early localization data or oversample final submission text.

**Compute/gate:** Zero-GPU audit and preprocessing. Require zero silent truncations, intact response/action boundaries in every retained window, and documented sufficient context for all dependent actions. Source-edit decision coverage should be measured before selecting a numeric mixture. Any retraining is separate; a 40–80-window probe is roughly 1.4–3 quota-hours using A as a reference, plus serving validation, and is not currently authorized.

### 3. Response association and mask correctness need separate tests

`render_atif.py:34–56` creates one assistant message for each tool call; when processing a response it attaches it to the **last assistant message's first call ID**, rather than the response's resolved call ID. For a multi-call step, an early response can be assigned to the last call; once a tool message has been appended, subsequent responses fail the `msgs[-1].role == assistant` condition and are dropped. This is a definite general renderer defect. Converted public traces normally alternate singleton calls and responses, which protects that path; therefore it cannot be called an established C batching cause. Own traces and original batching preservation require care.

Malformed call plus response deletion (38–41) can also hide a causally relevant failed attempt before a valid recovery. Dropping a failed action is not the same as correcting its syntax while preserving its observed consequence.

`model_spans` (63–83) searches literal role/response delimiters. It excludes entire response blocks including their delimiters and retains model-text/call spans through the turn-close marker. Whether a tool-response opener is model-emitted at serving and must receive loss requires exact template/parser contract inspection. Delimiter-like text in code or observations and incomplete response blocks need adversarial fixtures. `tokenize_windows.py:39–44` labels by token start offset; tokens spanning a span boundary and special tokens with unusual offsets require inspection.

The downstream next-token shift is internally consistent by static inspection: `TrainData.example` uses tokens 1..T as labels and masks those same positions (`data.py:26–33`); `LossWrap` uses hidden states 0..T-1 (`model.py:84–87`). This is contrary evidence to an obvious one-token shift bug, not a numeric proof of all mask/gradient correctness.

The official published Gemma template inspected is revision `0def573f1183891a2c86e0261db7291dd1545400`: [primary source](https://huggingface.co/google/gemma-4-31B-it/blob/0def573f1183891a2c86e0261db7291dd1545400/chat_template.jinja). It treats assistant/tool chains as continuing model turns and can emit a tool-response opener when a call has no supplied response. **This is reference behavior, not evidence that this exact revision generated C or served D.** Do not replace historical artifacts with current main and call the result a reproduction.

**Falsification/gate:** CPU-only reference fixtures for single call, two calls, multiple responses, error/recovery, empty submission/recovery, final text and code containing reserved-looking strings. Require exact ID association and no missing/orphan responses; compare rendered bytes and IDs to the actual served template; manually annotate every emitted call/open/close/stop token and compare mask. Require zero supervised environment-result tokens and complete supervision of all intended model-generated boundaries. Do not train environment observations merely to teach the model to expect success. A clean exact-artifact audit weakens masking/boundary causality. Cost: zero GPU, minutes to hours once dependencies/tokenizer are recovered, without importing torch.

### 4. Tool/prompt equivalence is incomplete

`declared_tools` (render_atif.py:89–99) intentionally omits analyzer agent_tool. The destination root declares it (`submissions/v7_noedit/agent.yaml:12–14`). Tool schemas are manually preserved JSON and selected by YAML regex, not exported from the compiled request. Missing `--agent` uses all schemas, including disallowed read/edit tools; exact preprocessing CLI is needed to prove C selected the right list. `render` also falls back to all tools for an empty list (103).

Training sets enable_thinking false (86); server default is true (`kaggle/eval_l4/evalkit/server.py:32`). The guide says ADK overrides it for the deployed request. **Do not label the differing default a demonstrated mismatch:** capture the effective request parameters. Compare actual model prompt after ADK/LiteLLM normalization, schema order/text, thinking flags, tool parser, BOS, continuation markers and stops. B's ability to edit using the same scaffold weighs against a universally unusable serving prompt.

**Falsification/gate:** A CPU compiled-request fixture with the pinned historical packages and tokenizer must produce equivalent tokenized input/continuation at the relevant decision points; any intentional differences documented individually. Zero GPU; no package upgrades to current versions during reproduction.

### 5. Objective weighting: token shares alone are insufficient

`MaskedCE.forward` returns mean loss over supervised positions in one window (`model.py:50–70`). The worker backpropagates that mean divided by fixed accumulation, and FSDP averages ranks (`worker.py:110–111`). Therefore the effective objective is approximately **equal-window mean CE**, not corpus-token mean CE. For category a, compute both its raw token share and its nominal objective mass:

`mass(a) = sum_over_scheduled_windows(n_supervised_tokens_of_a_in_window / n_supervised_tokens_in_window)`.

Normalize the masses over categories; account for repeated windows, equal rank averaging, optimizer accumulation, last partial accumulation and clipped gradients. This is the nominal loss coefficient, not measured gradient influence. Short final windows can carry disproportionate mass relative to their token counts, and long edit scripts need not dominate merely because they contain more tokens. Validation explicitly reweights each window loss by supervised token count (`worker.py:69–83`), so validation and training aggregate different measures. Neither averaging convention is inherently incorrect; the choice should match intended behavioral weighting.

The last step has one rather than two microsteps and still divides by two; it has half the normal accumulation weight. Group-of-four scheduling drops 2 train and 3 validation windows, already documented in C. These small edge effects are lower-priority than universal missing edits and should not be inflated into its cause.

**Required action audit:** distinct trajectories/tasks/repos; counts of locate/read/reproduce/source-edit/test-file-edit/temp-write/test/repair/submit/final-text; successful versus failed **replayed** source modifications; first edit position; pre/post-edit test counts; positive/empty submission states; per-window phase, prior-state dependencies, token counts and loss mass; one decision versus many calls; dropped/truncated records. Separate edit syntax, actual edit execution and semantic correctness.

**Falsification/gate:** Restore exact windows and map each label token to an action/state; compute both token and window-weighted distributions. If editing/recovery is well covered with grounded contexts and appropriate effective mass, do not assert read/test domination. Change phase weighting only when this analysis identifies an imbalance. Zero GPU, minutes to hours. Keep aggregate validation loss plus phase/category loss and later behavioral gates; do not select a model solely by lower CE.

## Recovery manifest

The exact original names/locations of several missing preprocessing files are unknown; inventing filenames would imply provenance we do not have. Current search of project, `/tmp` and default Hugging Face cache did not find C rendered JSONL, original selected raw/converted trajectories, or Gemma tokenizer assets. Existing primary evidence explicitly reports these missing.

| Required artifact | Available / missing | Recovery or regeneration and acceptance test |
|---|---|---|
| Exact C NPZ, summary and schedule inputs | Available: `data/runs/train_public_200.npz`, `.json`; copies in corrected C download/private-dataset preparation | Preserve SHA-256 `b3f51bc449c31de2a3b163f98c5fdec2725b4427bde15625af7b4c75447d2488`; compare each array and dtype, not merely zipped NPZ bytes (archive metadata can differ). |
| Exact selected 200 raw rows, trajectory IDs, full patches, base commits and source dataset revision | Missing locally | Recover preprocessing machine/cache/export and original selection manifest. NPZ task IDs recover task membership, but not the exact teacher trajectory among multiple runs per task. Pinned public export can regenerate a candidate only when selection identity/order is known. |
| Converter output ATIF and convert_report, exact template trace, source hashes and CLI flags | Missing locally | Recover original staging directory. Otherwise replay pinned source conversion from exact raw rows and template; preserve each file hash. Current code does not prove historic converter provenance. |
| Rendered JSONL including text, char spans, role/window IDs and n_tokens | Missing locally | Recover original output first. Regenerate using original converter artifacts, exact system prompt/agent config/tool schemas and tokenizer version; compare regenerated ids/masks/task/val/starts arrays exactly with C. |
| Historical tokenizer vocabulary/merges/config/added tokens/special-token mappings/chat_template; transformers/tokenizers versions | Missing locally | Recover original `/tmp/g4tok` from preparer's machine or its pinned upstream model revision. Training model attachment's tokenizer is not automatically the preprocessing tokenizer. If original revision unavailable, record unresolved provenance; do not call current-model regeneration exact. |
| Per-step source state, replay logs, changed tracked files and final diff | Not created/preserved by converter | Regenerate from pinned repo snapshots in isolated CPU replay after authorization. Store actual tool responses, exit codes, cwd/env, per-operation before/after hashes. Test replay needs dependency/container lock and verifier identity. |
| Runtime effective requests and compiled tools/template from D | Partial: submission/evalkit/config preserved; full request/template provenance not established | Recover exact wheel/model attachment versions and compiled CPU request fixtures; confirm per-request thinking and truncation behavior. No need to repeat D's 114 tasks. |

API page sampling is not enough for reproducibility: `hf_rows` shuffles offsets with a seed but uses unpinned public pages and skips pages after retries. Dataset updates or transient failures alter selection. Resume logic likewise depends on pre-existing output files. Record source revision, row hashes/IDs, page/export hashes, deterministic order, every exclusion, selected IDs and resume state. If exact source cannot be recovered, name the new candidate separately and retain C's provenance gap.

## Priority order and minimum release contract

1. Recover artifacts and complete CPU token/action/state audit. This directly tests Step D rank-2 hypotheses at zero GPU cost; lack of artifacts is an evidence blocker, not permission to guess statistics.
2. Replay selected converted mutations and submission state. Require faithful actual source edits and negative-feedback records before accepting targets.
3. Propose one correction based on the failing audit, with unchanged control and explicit separately authorized tiny training/serving probe. Do not bundle weighting, template, optimizer and dataset changes.

For future behavior probes, source edit → nonempty patch is the first gate; test correctness and generalization remain distinct later gates. Include an intentionally empty diff and a failed edit whose next actions must recover. Measure source mutations, true diff, stopping after feedback, proposal batch size and post-edit targeted testing. All completed normal tasks with empty patches is an immediate stop, even if CE and activity health checks pass. Budget/context changes alone do not supply the missing behavior.


## Additional exact NPZ audit (stdlib only)

This extends the prior structural audit to supervision weighting; it does not decode or classify actions. Reads used zipfile, ast, array, hashlib and random only. No torch/numpy/model import. Exact NPZ SHA matches the diagnosis. All 1,017 window lengths: minimum 2,799, median 7,661, maximum **8,191**, total 7,372,758. No row reaches the tokenizer clipping threshold.

Supervised tokens per window: min **23**, median **895**, max **4,726**, total **1,170,308**. Histogram: 1–99 = **49** windows; 100–499 = **365**; 500–999 = **109**; 1,000–1,999 = **273**; ≥2,000 = **221**. There are no zero-supervision windows. Under the preserved seed-0 shuffle and four-rank drop-last schedule, 972 training windows contain **1,114,033** supervised tokens; min 23, median 879, max 4,726. A token in the shortest-supervision window has approximately **205.5×** the nominal within-window coefficient of a token in the largest-supervision window (4,726/23), before optimizer/rank/accumulation, gradient clipping and content differences. This does NOT show which actions receive that weight. Do not assume the 23-token windows are submissions; they cannot be identified semantically here.

Current source SHA-256 (not proof these preprocessing sources were historically used):

| File | SHA-256 |
|---|---|
| posttrain/convert_public.py | 0c4853833876a2ea48ed7f24427da99d3b4a219fbd3369d745788888ef4d7a06 |
| posttrain/render_atif.py | ac3b2edaf2c668c4902e1a54820c13e89b026ab3a91ed66289b925409f90ef6d |
| posttrain/tokenize_windows.py | 6070dc4d36c0c783215914eee25e7e57dcf4beba971a9efa32154ebfcfcf06df |
| kaggle/train/trainkit/data.py | 4150786d8e1c785ed9a3c43b92fc46bb8f6f84dd5d46088b1482cfd8b560651b |
| kaggle/train/trainkit/model.py | c62bc225816094e1938cc4aa34b0c6c587f022eaf73e4d64bdce04e59cecbfee |
| kaggle/train/trainkit/worker.py | ed2339c27c02c3d5aaf195ed34042c03a26cb1586345e4a51ad01b3000d46763 |

Cwd persistence: converter forwards bash commands and rewrites absolute roots, but does not reconstruct the teacher shell's cwd/environment state before subsequent relative commands. The destination schema says each command runs in the repository workspace. Whether actual teacher traces rely on persistent `cd`/exports and whether the runtime preserves them must be established from exact rows and tool implementation. It is a replay requirement, not a proven C failure.
