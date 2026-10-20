# CPU research handoff — v14

Start with [executive_summary.md](executive_summary.md), which includes the thirteen direct decisions and links to the eight requested reports. The research is retrospective; proposed submission/data/runtime/training changes are not implemented.

## Files created locally

Eight requested reports:

- `executive_summary.md`
- `failure_analysis.md`
- `dataset_audit.md`
- `dataset_vNext_design.md`
- `architecture_improvements.md`
- `experiment_matrix.md`
- `competition_roadmap.md`
- `research_paper_opportunities.md`

CPU tooling and evidence:

- `audit.py`: read-only archive/trace audit; writes derived results in this directory only.
- `test_audit.py`: deterministic synthetic tests for edit recognition, read/test distinctions, nested failures, unknown/denied outcomes, response IDs and ordered observations.
- `validate.py`: stdlib NPZ structure/mask/split audit, input-hash preservation and v7/v14 paired statistics. No model or tokenizer loading.
- `input_manifest.json`: 763 preserved input-file hashes taken before analysis tooling was written. Covers selected reports/source/JSON/NPZ/archives/patches, not every file on disk (notably excludes `.log` and very large files). Original tracked changes are additionally checked with Git diff.
- `trace_audit.json`: all60 tasks, including per-step token metrics, intent/verification summaries and all30 failures; no raw thought text or command bodies copied here.
- `dataset_measurements.json`: new archive action/opening/transition measurements with explicit classifier definitions.
- `validation_results.json`: structural validation, input preservation and paired statistical results.
- `diagnostic_quarantine.json`: new-60 IDs marked retrospective-only, with no gold patches or training examples.
- `README.md`: reproduction and scope.

No existing file was modified. All files are new, untracked and uncommitted. Temporary generation helper `/tmp/generate_v14_failure_report.py` was also created to assemble the failure cards; it is CPU-only and outside the repository.

## Reproduce on CPU

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 docs/optimization_v14/audit.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s docs/optimization_v14 -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 docs/optimization_v14/validate.py
```

Only the Python standard library is needed. Scripts parse JSON, tar/zip/NPY bytes and Python AST. They never execute trace commands, import torch, load model weights, run benchmark tests, regenerate training inputs, call a model API or launch a notebook. Re-running replaces only their **new derived files** in this directory. Do not run them against changed inputs and silently accept a new baseline: validate.py fails if an original manifest hash changes.

Observed CPU validation:

- 14 deterministic tests pass.
- All60 v14 results/traces/logs inspected;30 unresolved cases documented; all8 context exceptions individually summarized.
- All1,379 proposed v14 calls associated using explicitly disclosed ordered same-tool fallback; zero unmatched observations under that reconstruction. This is not independent execution-ID proof.
- All four NPZs pass CRC, shape/length, binary-mask, positive-label, window-length and internal task-split checks. Semantic masks and tokenizer fidelity rely on preserved E0/E2 evidence and are not newly re-tokenized.
- All763 input hashes still match.
- v7/v14 exact-ID paired result reproduces23→30,22both/8gains/1loss/29neither,p=.0390625.
- Tokenizer replay, independent environment oracle runs, compiler/runtime support for a new architecture, current official rules and latest leaderboard outcome remain unverified prerequisites. They are not reported as completed tests.

## Evidence hierarchy and limitations

Measured original task results outrank old planning notes. New independent byte/array/trace findings are explicitly distinguished from preserved-report measurements and future hypotheses. Gold exposure, lexical mutation intent and adjacent-call transitions are fallible observability metrics; unknown outcomes remain unknown. Direct semantic understanding, complete patched-file syntax, hidden reasoning-token use and causal attribution are not inferred from those metrics.

The current public cohort is heavily FastAPI-concentrated, repeatedly evaluated and unsuitable for a new fair final generalization claim. No new independent dataset, preference corpus, trained adapter, submission or runtime change is produced here.

## Git and resource boundary

Initial status was clean; remote HEAD/main matched local HEAD through read-only `git ls-remote`. No fetch, stage, commit, push, merge, rebase, branch/worktree creation/switch, reset, clean or PR was performed. No GPU/model work, Kaggle notebook launch or leaderboard action was performed.

Final expected status:

```text
?? docs/optimization_v14/
```

Stop after analysis and wait for explicit user permission before any Git write or GPU experiment.
