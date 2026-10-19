# Map 2 — failure analysis

Tasks: 60 · resolved: 23

## Where it fails (first failing stage)
| stage | tasks |
|---|---|
| resolved | 23 |
| fix:wrong_fix_unverified | 16 |
| commit:saw_gold_never_edited | 7 |
| localize:never_found_gold_file | 6 |
| localize:saw_gold_edited_elsewhere | 4 |
| localize:edited_wrong_file | 2 |
| localize:found_file_not_lines | 2 |

## Key rates
| metric | value |
|---|---|
| evidence_hit_file_pct | 86.7 |
| evidence_hit_lines_pct | 76.7 |
| P(resolved|saw gold lines) | 50.0 |
| P(resolved|never saw gold lines) | 0.0 |
| P(edited gold|saw gold lines) | 82.6 |
| median_step_first_gold_lines | 2.0 |
| tool_calls_median | 29.0 |
| errors_total | 352 |
| error_recovery_pct | 44.0 |
| tasks_with_loops_pct | 48.3 |
| tasks_with_malformed_read_file_pct | 0.0 |
| read_file_calls_malformed_pct | None |
| P(resolved|malformed reads) | None |
| P(resolved|no malformed reads) | 38.3 |
| tasks_with_context_resets_pct | 1.7 |
| repeated_identical_calls_share_pct | 28.8 |
| grep_no_match_errors (benign) | 254 |
| tasks_stray_files_in_patch_pct | 20.0 |
| tasks_with_nudges_pct | 1.7 |
| tasks_overflow_pct | 3.3 |
| tasks_budget_exhausted_pct | 1.7 |
| explicit_submit_pct | 88.3 |
| empty_patch_pct | 25.0 |
| repro_before_edit_pct | 10.0 |
| tested_after_last_edit_pct | 1.7 |
| graph_tool_use_pct | 1.7 |
| nl_graph_queries_total | 0 |
| graph_results_nonempty_total | 1 |
| graph_results_empty_total | 0 |
| skill_runs_total | 0 |
| skill_errors_total | 0 |
| skill_use_pct | 0.0 |
| context_pressure_median | 0.39 |

## Tool errors
| type | count |
|---|---|
| CommandError | 352 |

## Tool mix
| tool | calls |
|---|---|
| run_command | 1538 |
| submit_patch | 54 |
| write_file | 4 |
| search_similar_code | 1 |
| code_analyzer_agent | 1 |
