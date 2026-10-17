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
