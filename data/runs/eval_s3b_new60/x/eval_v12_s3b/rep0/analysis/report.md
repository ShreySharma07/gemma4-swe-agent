# Map 2 — failure analysis

Tasks: 60 · resolved: 9

## Where it fails (first failing stage)
| stage | tasks |
|---|---|
| localize:never_found_gold_file | 24 |
| localize:found_file_not_lines | 16 |
| resolved | 9 |
| commit:saw_gold_never_edited | 6 |
| fix:wrong_fix_unverified | 5 |

## Key rates
| metric | value |
|---|---|
| evidence_hit_file_pct | 60.0 |
| evidence_hit_lines_pct | 33.3 |
| P(resolved|saw gold lines) | 45.0 |
| P(resolved|never saw gold lines) | 0.0 |
| P(edited gold|saw gold lines) | 70.0 |
| median_step_first_gold_lines | 4.0 |
| tool_calls_median | 39.5 |
| errors_total | 424 |
| error_recovery_pct | 26.2 |
| tasks_with_loops_pct | 53.3 |
| tasks_with_malformed_read_file_pct | 0.0 |
| read_file_calls_malformed_pct | None |
| P(resolved|malformed reads) | None |
| P(resolved|no malformed reads) | 15.0 |
| tasks_with_context_resets_pct | 0.0 |
| repeated_identical_calls_share_pct | 47.4 |
| grep_no_match_errors (benign) | 187 |
| tasks_stray_files_in_patch_pct | 0.0 |
| tasks_with_nudges_pct | 0.0 |
| tasks_overflow_pct | 23.3 |
| tasks_budget_exhausted_pct | 0.0 |
| explicit_submit_pct | 73.3 |
| empty_patch_pct | 76.7 |
| repro_before_edit_pct | 0.0 |
| tested_after_last_edit_pct | 0.0 |
| graph_tool_use_pct | 0.0 |
| nl_graph_queries_total | 0 |
| graph_results_nonempty_total | 0 |
| graph_results_empty_total | 0 |
| skill_runs_total | 0 |
| skill_errors_total | 0 |
| skill_use_pct | 0.0 |
| context_pressure_median | 0.51 |

## Tool errors
| type | count |
|---|---|
| CommandError | 424 |

## Tool mix
| tool | calls |
|---|---|
| run_command | 1910 |
| submit_patch | 45 |
