# gemma4-swe-agent

Post-training Gemma 4 31B (QAT W4A16) into an offline software-engineering agent for the Kaggle **Google – Gemma 4 Developer Agent** competition. The agent resolves GitHub issues in unseen repositories on 4×L4 GPUs.

- Status, roadmap, and every experiment: [`docs/plan.md`](docs/plan.md), [`docs/experiments.md`](docs/experiments.md)
- Findings so far: [`docs/findings.md`](docs/findings.md)
- Working with Claude Code: [`CLAUDE.md`](CLAUDE.md)

## Layout
```
submissions/   agent configs (declarative ADK YAML) for every submitted variant
kaggle/        Kaggle scripts: scorer-faithful L4 eval, renamed dev-set builder, probes
analysis/      trace analyzer, run comparison, dataset/graph analyses
tools/         submission validation and zipping
results/runs/  small per-run artefacts
docs/          HQ decision log, plan, experiments, findings, literature
notebooks/     notebook-style pipeline (EDA → baseline → data → training → final)
```

## Quick start
1. Local evaluation: upload `kaggle/eval_l4/run_official_l4.py` to a Kaggle notebook (GPU L4×4, Internet off). Attach the competition data, the `metric/gemma-4-developer-agent-wheelhouse` dataset and the Gemma 4 31B W4A16 model, set `VARIANT` or `SUBMISSION_DIR`, then Save & Run All.
2. Analyse: `python analysis/compare_runs.py data/runs/<a>/merged data/runs/<b>/merged`.
3. Submit: `python tools/make_zip.py submissions/<name>`, then upload the zip.

License: MIT.
