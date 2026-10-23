# Submissions

Each folder is an unzipped submission (zip the *contents*: `python tools/make_zip.py submissions/<name>`).

| Name | Idea | Local (129) | LB |
|---|---|---|---|
| baseline_v1 | organisers' sample agent with raised budgets | 15 | 0.00 |
| v2_sed | no `read_file`; grep/sed reads | 26 | **0.08** |
| v2b_budget80 | v2 + 80 calls | – | – |
| v3_verify | v2 + mandatory budget plan + one targeted test | – | – |
| v4_pipeline | localizer → fixer SequentialAgent | – | – |
| v5_graph | v4 + code-map skill | 33 | 0.08 |
| v7_noedit | v2 minus `edit_file` (61.5% of its calls drift); edits via python snippet-replace or write_file | – | **0.15** (current base) |
| v6_trace | skill-only localizer + rich brief | – | 0.00 (~10 h; skill likely broken on scorer) |
| v9_calls70 | v7 + eval_config max_tool_calls 40→70, max_turns 100→160 (time limit unchanged) | **0.12** (ref 57218504): no gain vs v7 |
| v10_edit_test | v9 + system.md 'Call Budget Checkpoints': edit by ~call 20, test the changed code's test file before submit_patch, submit by ~call 60 | superseded by v10b (v9 base did not help) |
| v10b_edit_test | v7 + system.md 'Call Budget Checkpoints' (40 calls): edit by ~call 15, test before submit_patch, submit by ~call 35 | **0.10** (ref 57253361): worse than v7 |
| v11_out4096 | v7 + sampling max_output_tokens 8192→4096 (avoid the 32,769 context boundary) | **0.10** (ref 57295077): near-identical to v7, so LB noise is large |
| v13_submit_tool | v7 + one prompt line: `submit_patch` is a tool, never run it via `run_command` (v7 shell-called it in 32/60 new-60 tasks and 45/129 full-run tasks, one wasted call + turn each) | – | **0.10** (ref 57383089): no measurable change vs v7 |
| v14_thinking | v13 + thinking ON (budget 4096, include_thoughts) + public budgets (8 min, 28 calls, 240 s cmd timeout, 80 turns). Top public 0.22–0.24 recipe; forum 129-task study: thinking off 23 vs on 48–51 | **new-60: 30/60 vs v7 23/60** (22 both, 8 v14-only, 1 v7-only; McNemar p=0.039); empty patches 8% vs 25%, repeats 6.5% vs 29%, overflow 13% vs 3%; 4.0 h for 60 tasks | **0.17** (ref 57425260): best so far (v7 0.15) |
| v15_spec | v14 + max_output_tokens 8192→6144 (8 context overflows on new-60) + prompt from the public 0.24 notebook (honghanhhh, adapted): issue-as-specification, parallel implementations, scratch files in /tmp, `git status --short && git diff` before submit, no graph tools (search_similar_code can return >100k chars) | – | **ERROR** (ref 57470215): "Notebook Threw Exception" after ~4 h. Suspect: 6144 output cap truncates a long thinking+call response (v14 max 5,913 tokens; thinking budget is soft) |
| v15b_spec8k | v15's prompt with v14's max_output_tokens 8192 (isolates the prompt; tests whether 6144 caused v15's error) | new-60 not run (Atul's run cancelled before start) | **0.08** (ref 57509358): ran without error (so v15's crash was the 6144 cap), but well below v14's 0.17 |
| v16_small_output | v14 + one prompt section 'Keep Tool Output Small': never `cat` files, bound searches with `\| head -n 20`, long output `\| tail -n 40` (v14's 8 overflow tasks: 47k tool chars/task vs 20k, 44% from `cat`) | **new-60 (Atul): 27/60 vs v14 30/60** (26 both, 1 v16-only, 4 v14-only, p=0.375); overflows 8→5, cat calls 99→59 | not submitted (no gain) |
| v17_superagent | exact copy of the public recipe `jvnhonglei/gemma-4-superagent-reproduction` (credited): `read_file` + `edit_file`, spec-first prompt with call plan (edit by call 12), no code_analyzer, thinking 4096, temp 0.2, 8 min / 28 calls / 240 s | – | submitted Oct 23 (pending) |
