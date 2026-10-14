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
