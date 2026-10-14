"""
evalkit: scorer-faithful local evaluation of a submission on Kaggle 4x L4 (harness swegemma 0.2.11 / adk-submission 0.2.13).

Modules, in the order the notebook uses them:
  config      Settings (one JSON file shared by every step) and fixed paths
  util        run.log logger + shell helper
  envsetup    ScorerEnv: restore the Py3.12 scorer venv, upgrade changed wheelhouse wheels, add test deps (stdlib only:
              runs in the notebook's own Python 3.13 kernel)
  inputs      Inputs: competition / dev-set tasks, snapshots, served model
  submission  Submission: unpack the submission under test, declared model, adapters, eval budget
  server      ScorerServer: vLLM with the scorer's settings + model registry; offline sandbox
  runner      stratified task order + ChunkedEvaluation (official Evaluator, sequential, rep-major repeats)
  report      Report: merge chunks per rep, pass@k summary, trace analysis, package
  main        orchestration of everything after the env step (runs inside the venv)
Source of truth: kaggle/eval_l4/evalkit/. make_eval_notebook.py turns it into the competition notebook.
"""
