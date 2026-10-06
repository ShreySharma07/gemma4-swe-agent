# Map 2 — Baseline findings (run 1, 1 Oct 2026; all 129 traces analysed)

**Run.** Scorer-faithful local run on Kaggle L4×4:
- vLLM 0.19.1 with the W4A16 checkpoint, TP=4
- official harness, sequential tasks, 4.5 min / 40 calls, offline agent
- agent = organisers' sample prompt + analyzer sub-agent, no adapters, temp 0.2, thinking off
- the same config scored **0.00 on the public LB**

## Headline
- **15/129 resolved locally (11.6%).**
- **32 fastapi tasks could not pass in our local environment whatever the patch.** Verification crashed at test collection with `No module named 'inline_snapshot'` (25 tasks) or `'dirty_equals'` (9). These are test-only deps that the subprocess sandbox lacks.
- **On the 97 verifiable tasks: 15/97 = 15.5%.**
- Fixed for the next run: the sandbox venv sees host site-packages via `_host_env.pth`, so `run_official_l4.py` now pip-installs both packages (no-deps). The analyzer flags `env_unverifiable`.

## Where it fails — 97 verifiable tasks (first failing stage)
| Stage | Tasks |
|---|---|
| **Empty patch: never made a successful edit** | **41** (42%) |
| ↳ never even attempted an edit | 31 |
| ↳ edit attempts all failed (malformed args 5, other 5) | 10 |
| Edited gold file, wrong fix, never tested | 27 |
| Saw gold, edited a different file | 8 |
| Never found gold / found file not lines / wrong file | 21 |
| Wrong fix, tested locally | 5 |
| **Resolved** | **15** |

- **Localisation is not the bottleneck.** Gold lines were seen in 70/97 tasks, but P(resolved | saw gold lines) is only 20.0% (vs 3.7% when they weren't seen).
- **Verification is almost absent.** Only 7/97 ran a test after their last edit (2 of those resolved).

## Root cause of the 41 empty-patch tasks: read loops
Tasks that never edit vs the rest (means per task):

| Metric | No-edit tasks | Others |
|---|---|---|
| Malformed `read_file` calls | **11.0** | 3.0 |
| `read_file` calls | 15.7 | 10.4 |
| Exact-repeat tool calls | **26.5** | 14.2 |
| Context resets (prompt tokens drop >40%) | **2.05** | 1.12 |
| Share with ≥1 malformed read | 61% | 36% |

Mechanism:
1. Gemma's tool calls arrive with garbled argument keys: `start_line"` (732×), `end_line"` (572×), or end_line without start_line. That is **47% of all 1,608 read_file calls**, in 41% of tasks.
2. `read_file` silently ignores the unknown keys and returns lines 1–150 again.
3. The agent re-reads the same top of the file.
4. History is compacted at ~14k prompt tokens (each read ≈1.4k tokens), and after each reset the agent repeats the same grep + read sequence.
5. Result: 48% of all tool calls are exact repeats, and the budget runs out before any edit.

**Caveat:** among tasks that do edit, malformed reads barely change the resolve rate (13.3% vs 17.3%, small n). Their effect is on *getting to an edit at all*.

## Second tool-format bug: malformed `edit_file` arguments
- 438 edit calls in 17 tasks had code fragments leaking into argument keys. Example: `old_string` content containing `\"` breaks the call encoding, so ADK reports "mandatory input parameters are not present: old_string" and the agent retries in a loop.
- Likely source: Gemma-4 call serialisation of strings containing quotes, and/or the vLLM gemma4 parser.
- **Natural LoRA target: a tool-call-format reliability adapter.**

## Other facts
- **Graph tools are nearly unused/misused.** 415 `search_similar_code` calls (46 of them natural-language queries returning nothing), 4 `get_code_neighbors`, 1 `get_code_subgraph`. Analyzer sub-agent called 13 times in 129 tasks.
- **CommandError (380) is mostly benign.** 213 are grep with no match (exit 1); 20 are pytest runs.
- **Stray repro files in patches** (18+ tasks) don't break scoring; resolved tasks carry them too.
- 8 tasks hit the 4.5-min timeout and 2 hit the 100-turn cap (≈7.8% hit a budget).
- **Local 15.5% (verifiable) vs LB 0.00 for the identical config is unexplained.** Hypotheses:
  - (a) hidden repos lack graph files while the prompt pushes graph tools
  - (b) public-repo familiarity
  - (c) larger hidden repos amplify the read loops
  - (d) the hidden tasks' test deps

## Experiment v2 (one change): remove the broken read channel
- `read_file` is removed from the root agent's tools.
- The prompt line "using `read_file` or search tools" becomes "with `run_command` using `grep -n` and `sed -n 'START,ENDp' path` (≤~80 lines per read)".
- The analyzer gets read-only `run_command` instead of `read_file`.
- Everything else is identical.
- **Predictions:** fewer no-edit tasks (41 → fewer), fewer exact repeats, fewer context resets, higher resolve rate on verifiable tasks.
- Artefacts: `run_official_l4.py` (VARIANT="v2_sed", plus the test-deps fix) and `submission_v2_sed.zip` for the LB.
- Next after v2: the amnesia (resets) and the verification gap, then the edit-format LoRA.

---
# Run 2 — v2_sed (3 Oct 2026): `read_file` removed, test deps fixed (all 129 verifiable)

**Leaderboard: 0.08** (baseline 0.00).

**Local: 26/129 resolved (20.2%).**
- On the 97 tasks verifiable in both runs: baseline 15 → v2 **21**. Both runs solved 13, only the baseline solved 2, only v2 solved 8. Exact McNemar p ≈ 0.11, so the direction is clear but this alone is not significant at n=97; the LB move from 0.00 to 0.08 agrees with it.
- The 32 tasks that were env-broken in run 1: v2 solved 5 of them.

| Metric | Baseline | v2 |
|---|---|---|
| Empty patches (97 common tasks) | 41 | **20** |
| Tasks with loops | 77% | 60% |
| Exact-repeat call share | 48% | **28%** |
| Tasks with context resets | 74% | 49% |
| Gold lines seen | 70% | 80% |
| P(resolved \| saw gold lines) | 15.6% (20.0% on verifiable) | 24.3% |
| Tested after last edit | 7% | 14.7% |
| Malformed read calls | 47% of reads | 0 (tool removed) |
| Malformed edit calls | 438 in 17 tasks | 70 in 8 tasks |

**New dominant failure: wrong fix, never tested — 47/129 (36%).**
- Only 9 of those 47 ran any test at all, and only 6 reproduced the bug before editing.
- They used a median of 37 of 40 tool calls, so the budget ran out before verification.
- Of the 26 resolved tasks, 9 tested after their last edit.

**Next, v3_verify (one change):** v2 plus a mandatory budget plan in the system prompt.
1. ≤20 calls exploring, without re-reading lines already seen.
2. Edit by call 25.
3. Run the one existing test file related to the change.
4. Fix and re-run once if the change broke it.
5. Submit; scratch files go in /tmp.

**Predictions:** tested-after-edit ↑ substantially, wrong-fix-unverified ↓, resolve ↑. **Risk:** forcing an early edit may lower the quality of edits on hard tasks.

---
# Graph tools diagnosis (4 Oct 2026)

**Why `search_similar_code` returned nothing.** It is not a semantic search.
- `embed()` is a dictionary lookup of the query string among the node names in the repo's `.npz`. There is no text encoder.
- The installed swegemma has no `resolve_node_name` (graph_probe: AttributeError), so a short name is never mapped to a full node name.
- Result: only an exact qualified node name works, e.g. `fastapi.dependencies.utils.solve_dependencies`.

**Evidence from both runs (all 481 search calls):**
| Query type | Non-empty results |
|---|---|
| Sentence / keywords | 0/329 |
| Bare identifier | 0/132 |
| Dotted path | 3/20 (the misses were wrong paths) |

- `get_code_neighbors` with full dotted names worked 3/4 times.
- The graph files themselves load fine: about 3.7k nodes for fastapi, every node with a 256-dim embedding.
- The v4 localizer prompt actually told the agent to use short names, which is wrong.

**v5_graph = v4 + structural lookup.**
- New `code-map` skill: a stdlib `ast` script run in the task sandbox via `run_skill_script`. It does not depend on graph files, so it also works if hidden repos have none. Commands:
  - `find NAME` → qualified node name, file:line range, signature
  - `callers NAME`, `callees NAME`, `outline FILE`
- Localizer prompt: get the qualified name with `find`, then pass that exact name to `get_code_neighbors` and `search_similar_code` (used for OTHER_LOCATIONS). Never pass sentences.
- Tested through ADK's real skill wrapper on fastapi and rich: correct results in 0.1–0.4 s. Compiles with the official compiler.
- `trace_analyzer` now counts graph results (non-empty vs empty) and skill runs/errors.
- **Predictions vs v4:**
  - non-empty graph results go from about 0 to the majority of graph calls
  - fewer grep calls before the first gold-file read
  - OTHER_LOCATIONS filled more often
- Risk: the 4 extra skill tools add tool-choice noise.

---
# Run 3: v5_graph (5 Oct 2026). LB 0.08, the same as v2

**Local: 33/129 (v2: 26).**
- Gained 12 tasks, lost 5. McNemar p = 0.14, so this is not significant.
- By repo: fastapi 16 → 21, rich 9 → 11, requests 1 → 1.

**Behaviour:**
- Graph tools: **0 calls**. The agent used the `code-map` skill instead: 84 tasks, 186 calls, 90% returned useful output.
- **The localizer overruns its budget.** It averaged 48 calls (the prompt says ≤25), and 76 tasks used ≥40.
  - The fixer was left with fewer than 10 calls in 75 tasks.
  - Resolved tasks: localizer 35 / fixer 13.5 calls. Unresolved: 53 / 20.
- **The localizer loops:** 47.6% of its calls are exact repeats, and 66% of tasks had a context reset.
- **Malformed `edit_file` calls:** 351 of 571 edit calls (61%), concentrated in 14 tasks (5 never produced a valid edit). The `old_string` content leaks into the argument keys.
- Timeouts at 4.5 min: 21 (v2: 6). Median calls 75; 56 tasks hit about 80 calls. Total agent time 6.45 h.
- Tested after the last edit: 34/129 (v2 ≈ 19).
- The fixer always edited the file the diagnosis named (84/84).
- The localizer's confidence means little: 83 "high", only 31 of those resolved.

**Fidelity bug found in our runner.** It compacted context with interval 5 and threshold 14336.
- The official scorer uses interval 15 and threshold 32768 (HARNESS_README §7.2). That means effectively no compaction inside a task.
- Fixed in `run_official_l4.py`.
- Replaying v5 with no compaction: 11–29 tasks would overflow the context, but only 3–5 of them were resolved. Overflow explains a small part of the local-vs-LB gap, not most of it.

**Main conclusion: our local set no longer predicts the LB.**
- Local went 15 → 26 → 33 while the LB went 0.00 → 0.08 → 0.08.
- On about 120 hidden tasks, 0.08 is about 10 tasks, with a standard error of about 2.5 percentage points. The LB cannot see gains of a few points either.
- Likely cause: the local repos (fastapi, rich, requests) are public and Gemma has probably seen them in training. The hidden repos are private.
- We need a dev set that predicts the hidden set before tuning further.

---
# Plan after v5 (5 Oct 2026): v6_trace + a renamed dev set

**v6_trace (submission_v6_trace.zip).**
- **Localizer:** its only tool is the `code-map` v2 skill.
  - Commands: search (each hit shows its enclosing function), show, find, trace, callers (2 levels), callees, path (shortest call chain), similar (same-file / sibling-class overrides plus identifier overlap), tests, outline.
  - **Hard budget:** a per-task counter in /tmp makes the script refuse after 22 calls and order the handoff.
  - Prompt method: ANCHOR → TRACE (entry point down / helper up) → READ → SCOPE.
- **Handoff brief:**
  - SUMMARY, FILE, LINES, FUNCTION
  - CURRENT_CODE: the exact lines, verbatim
  - CALL_CONTEXT: the entry → function chain
  - ROOT_CAUSE, FIX steps, OTHER_LOCATIONS
  - TEST_COMMAND, CONFIDENCE plus evidence
- **Fixer:** confirm the code once, edit with 1–6 line `old_string`s (no docstrings), use a python-heredoc fallback after a garbled edit call, verify once, always submit.
- Budget: 64 calls / 4.5 min / 130 turns.
- Tested on real fastapi and rich checkouts through ADK's skill wrapper. Compiles with the official compiler.

**Renamed dev set (build_devset.py, Kaggle CPU).**
- For each fastapi / rich / requests task:
  - Rebuild the workspace with the official extract code.
  - Rename with same-length, word-boundary replacements: fastapi→glimmer, FastAPI→Glimmer, rich→luma, requests→conduits.
  - Apply the same renames to the code, issue text, gold patch and test patch.
  - Rebuild git history as a single commit.
- Verify each task with the official verifier: the gold patch must pass and an empty patch must fail. Tasks that fail are dropped.
- Sanity check on rich HEAD: 954 of 957 tests still pass after renaming. The 3 failures are SVG/hash snapshot tests.
- The runner's `DEVSET=True` mode evaluates on it.
- **The hypothesis to test:** if Gemma relies on memorised public code, v5 on the renamed set scores well below 33/129, closer to the LB.
