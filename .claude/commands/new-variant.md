Create a new submission variant with exactly ONE change. Argument: base variant + description of the change, e.g. `v2_sed lower temperature to 0`.

1. Copy `submissions/<base>/` to `submissions/<new_name>/` and apply only the requested change.
2. Write the hypothesis and predicted effect at the top of the new row in `docs/experiments.md` (local and LB columns empty).
3. Validate with `python tools/validate_submission.py submissions/<new_name>`. It must compile.
4. Compute the worst-case scorer runtime: `max_time_minutes × 120` plus about 1 min per task overhead. Refuse if it exceeds ~10.5 h.
5. If the change relies on scorer-side execution not yet proven on the LB (skills, adapters, new config fields), say so and propose a cheap probe first.
6. `python tools/make_zip.py submissions/<new_name>` → `dist/`.
7. For local evaluation: upload `submissions/` as a Kaggle dataset and set `SUBMISSION_DIR` in `kaggle/eval_l4/run_official_l4.py`.
