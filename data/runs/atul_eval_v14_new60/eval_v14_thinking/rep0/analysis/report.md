# Map 2 — failure analysis

Tasks: 60 · resolved: 30

## Where it fails (first failing stage)
| stage | tasks |
|---|---|
| resolved | 30 |
| fix:wrong_fix_unverified | 16 |
| localize:saw_gold_edited_elsewhere | 8 |
| commit:saw_gold_never_edited | 5 |
| localize:edited_wrong_file | 1 |

## Key rates
| metric | value |
|---|---|
| evidence_hit_file_pct | 98.3 |
| evidence_hit_lines_pct | 91.7 |
| P(resolved|saw gold lines) | 54.5 |
| P(resolved|never saw gold lines) | 0.0 |
| P(edited gold|saw gold lines) | 83.6 |
| median_step_first_gold_lines | 3 |
| tool_calls_median | 24.0 |
| errors_total | 151 |
| error_recovery_pct | 85.4 |
| tasks_with_loops_pct | 21.7 |
| tasks_with_malformed_read_file_pct | 0.0 |
| read_file_calls_malformed_pct | None |
| P(resolved|malformed reads) | None |
| P(resolved|no malformed reads) | 50.0 |
| tasks_with_context_resets_pct | 0.0 |
| repeated_identical_calls_share_pct | 6.5 |
| grep_no_match_errors (benign) | 90 |
| tasks_stray_files_in_patch_pct | 18.3 |
| tasks_with_nudges_pct | 1.7 |
| tasks_overflow_pct | 13.3 |
| tasks_budget_exhausted_pct | 0.0 |
| explicit_submit_pct | 88.3 |
| empty_patch_pct | 8.3 |
| repro_before_edit_pct | 5.0 |
| tested_after_last_edit_pct | 3.3 |
| graph_tool_use_pct | 6.7 |
| nl_graph_queries_total | 0 |
| graph_results_nonempty_total | 3 |
| graph_results_empty_total | 1 |
| skill_runs_total | 0 |
| skill_errors_total | 0 |
| skill_use_pct | 0.0 |
| context_pressure_median | 0.535 |

## Tool errors
| type | count |
|---|---|
| CommandError | 141 |
| BudgetExceeded | 10 |

## Tool mix
| tool | calls |
|---|---|
| run_command | 1221 |
| write_file | 78 |
| submit_patch | 69 |
| get_status | 7 |
| get_code_neighbors | 3 |
| search_similar_code | 1 |
