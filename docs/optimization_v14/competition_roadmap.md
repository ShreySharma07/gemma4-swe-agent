# Competition roadmap and stop boundary

## Current verified state

Local HEAD and read-only remote HEAD/main matched `125ce5a3e35bf6c4254611cf2d0c07b347d5a10b` during this task. Initial `git status --short` was empty. No fetch/merge/branch/worktree operation was performed. The current directory is left on its original branch. All new research is uncommitted under this directory; original HQ/experiments/submissions/data remain unchanged.

Latest robust local outcome: v14 30/60 versus v7 23/60. The latest repository commit notes v14 was submitted to the leaderboard and pending; the older `docs/experiments.md` row refers to the independent Atul evaluation and says not submitted for that run. These records describe different activities. Do not resolve them by inventing a v14 leaderboard score. v7's historical0.15 and v13's recorded0.10 remain historical observations, not live standings.

On October20 the official [rules](https://www.kaggle.com/competitions/gemma-4-developer-agent/rules) and [leaderboard](https://www.kaggle.com/competitions/gemma-4-developer-agent/leaderboard) pages were requested through web browsing; they returned no readable body. The overview was discoverable. **Current rules and latest leaderboard outcome were not independently verified.** No authenticated competition action was attempted. Treat architecture/data constraints below as repository-documented until fresh official text is captured before an experiment/deployment decision.

Repository sources (`CLAUDE.md`, HQ) describe fixed31B, offline sequential4×L4, 12h total, declarative submission, no custom Python entrypoint, adapter rank≤128/count≤8/total<3GiB, and public external data/teacher-output allowances subject to terms. Historical deadlines are November12 paper and December2 final; verify them from readable official material before planning around them. A prior note saying ~14k compaction is superseded by actual runner settings and zero-compaction traces.

## Ranked engineering direction

1. **Correct evidence and output handling.** Use the new classifier, acknowledge wrapt misattribution, and preserve exact baseline files. Prototype bounded output with relevant diagnostic/code retention. This has the lowest intervention cost and strongest observed architectural failure channel.
2. **Freeze an independent small rollout gate.** Obtain task/snapshot/environment provenance and leakage exclusions. Full rollouts reveal default test loops and source-edit failure that one-step gates miss.
3. **Request one bounded inference-only paired smoke experiment.** Use the E1 proposal; no automatic launch follows this report. Check fresh quota and current scorer support first.
4. **Add state memory/recovery only as isolated ablations.** Demonstrate actual context replacement before claiming compaction. Reject unconstrained localizer overhead and fixed edit schedules.
5. **Curate verified v14-compatible recovery data.** Do this only if inference evidence leaves a bottleneck that training can plausibly address and authentic eligible data exists. No S3b scaling or unchanged retraining.
6. **Confirm on independent repositories, then consider publication/deployment.** Budget is materially larger than the historical remaining balance. Resolve scorer runtime and rules before a leaderboard attempt.

## One-week CPU plan for one engineer

| Day | Deliverable and acceptance criterion | Dependency |
|---|---|---|
| 1 | Review failure cards and output-cap information-loss cases; preserve all source/test/API facts in reviewed samples | Current reports and raw evidence |
| 2 | Prototype bounded command-output policy and factual-state schema outside active submissions; exact tokenizer/template replay if available | No model generation required; verify permissible source/template availability |
| 3 | Check actual declarative/scorer API support, compiler schema and fresh-context semantics using source/CPU compilation | Readable official rules and installed compiler/source; report missing access rather than assuming support |
| 4 | Build new task provenance inventory and fork/near-duplicate groups; exclude all diagnostic/gate tasks | Eligible licensed independent task sources |
| 5 | Freeze development/final split manifests and environment-oracle validation procedure | Snapshot/test provenance; do not execute resource-consuming oracles without a concrete approved scope |
| 6 | Produce corpus preview/reject ledger and effective-loss/action audit for Designs A/B using already authorized data only | Authentic recovery availability; stop accepting invented/synthetic success |
| 7 | Review one isolated experiment proposal, exact settings, fresh accounting and hard caps; prepare a concrete review package | User's separate GPU permission remains required |

This is an estimated plan, not a scheduled automation or a commitment that independent tasks/licenses will materialize in seven days. This assignment ends with the handoff; future implementation beyond the local audit is not executed.

## Next approval proposal

Prefer **8 independent tasks ×2 arms**, frozen v14 versus v14+one bounded-output section, complete rollouts. Expected aggregate1.2–1.8 session h/2.4–3.6 quota h; proposed hard aggregate2.8h/5.6quota h and at least1h fresh reserve. Exact setup/watchdogs and live quota must be checked before approval. Reject this proposal if CPU evidence-retention or scorer-interface checks fail. See [full protocol](experiment_matrix.md) for operational stops and small-sample limitations.

No new GPU job is authorized. The last13.037684h balance is stale and was not refreshed because no launch is proposed for execution now. Do not spend it or launch a new notebook to verify it.

## Longer-term strategy and rejected work

Protect v14's demonstrated nonempty-patch and lower-repeat behavior. Seek deployable, matched-budget repair gains rather than the largest local score on familiar FastAPI tasks. Train only after verifying state-dependent data and rollout behavior, then independently test repository generalization. Pair graph evidence retrieval with actual contract repair, not a gold-localization proxy already near saturation.

Reject blind output/thinking reduction, larger call caps, broad graph stages, unverified skill runtimes, loss-based release decisions, fabricated DPO rewards, benchmark-trace training and repeated leaderboard tuning. Do not claim a future gain or guarantee a paper novelty result from this retrospective audit.

**STOP:** no Git write, commit, stage, push, merge, rebase, branch/worktree change, PR, training, model inference, Kaggle notebook launch or leaderboard action follows this document without explicit user authorization.
