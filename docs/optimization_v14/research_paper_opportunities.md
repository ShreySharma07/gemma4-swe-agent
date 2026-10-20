# Research paper opportunities and limits

## Best-supported contribution

The strongest existing result is a negative systems/post-training finding: **imitation fit and single-decision gates can improve while complete coding-agent behavior collapses**. Step C has much lower validation loss but Step D resolves0/114 with114 empty patches. S3/S3b improve grounding/decision diagnostics, yet S3b resolves9/60 against v7's23/60. A reproducible audit links those outcomes to observable state/window/action exposure defects without claiming a fully identified cause.

A second supported observation is the distinction between **seeing the repair location and implementing a correct repair**. v14's reported gold-file exposure is59/60 and line exposure55/60; only30 resolve. The exposure metrics are heuristic and gold-based, so the paper should not call them measured code understanding. Partial API propagation and scratch-only final diffs provide concrete failure modes beyond localization.

The context result is technically precise: all eight v14 exceptions reject at least24577 input+8192 reserved output against32768; no compaction events are recorded. Report both input headroom and nominal context. Per-task metrics describe completed requests and do not recover the rejected prompt exactly. The wrapt correction illustrates why startup warnings must not be treated as causal environmental explanations.

Sources: [C/D/E2 reports](../e0_e2_audit_20261016.md), [S1/S3 gate report](../atul_step1a_1b_step2_gate_report_20261018.md), [S3b handoff](../s3b_handoff_20261018.md), current CPU artifacts and [v14 report](../atul_v14_new60_report_20261019.md). Historical notes stating training/generalization remains untested are superseded where later evaluation exists.

## Connection to graph/code-reasoning research

Existing execution-graph research records only17% overlap between executed repository edges and provided graph edges, and weak same-class/constructor coverage; the KT1 repair pilot was ambiguous/instrument-limited, while the sibling-separation CPU pilot failed its preset threshold. These are negative or limited findings, not evidence that a new graph architecture improves task resolution. Review `docs/experiments.md`, `analysis/exec_graph_analysis.py`, `analysis/kt1_graph_repair.py` and the original graph research notes before presenting them.

A viable new question is: **does explicit source-contract evidence improve cross-file repair when file localization is already high?** Compare narrow text retrieval to matched-budget symbol/caller retrieval, measuring genuinely new required evidence and eventual resolved issues. Dynamic execution evidence, graph edges and source text must be separated experimentally. Do not frame adding a call graph alone as novel or infer causality from gold reachability.

Another viable question is **whether conditional training exposure predicts rollout failures better than global target-token shares**. S3b's63.1% edit token share coexists with frequent test-first supervision and pytest-heavy complete trajectories. Test the opening/phase mechanism using authentic data and matched effective-loss dose; the current correlation is motivating evidence only.

## Claims established versus open

| Claim | Status | Additional evidence needed |
|---|---|---|
| Training/export/adapter serving is feasible on the documented stack | Established for recorded runs | Reproducibility/environment disclosure |
| Lower imitation loss suffices for issue resolution | Rejected by these counterexamples | No broader universal claim needed |
| State loss and scratch-target exposure exist in Step C | Established by E0/E2 and source audit | Share exact provenance and masks |
| Those defects uniquely caused collapse | Unproven | Controlled data ablations and matched dose |
| S3b one-step gate guarantees healthy complete rollouts | Rejected in this observed case | More independent task/gate replications for generality |
| Test-first weighting caused S3b collapse | Plausible, unproven | Opener/conditional-transition ablation, fixed optimization/scaffold |
| Bounded output/compaction improves resolution | Hypothesis | Scorer-feasible CPU preservation checks plus independent paired rollouts |
| v14 thinking alone improves coding | Unproven; bundled changes | Thinking/prompt/budget ablations, matched tasks/settings |
| Graph retrieval improves repair beyond localization | Unproven | New independent matched-budget retrieval experiment |
| Public-task gains generalize to private repositories | Unproven | Truly independent repository groups, frozen evaluations |

## Experiments with publication value

1. Preregister output-bounding versus state-replacement versus control, verify the actual scorer interface, and measure independent resolved issues plus retained-evidence failures. Publish negative outcomes too.
2. Hold scaffold/optimization/effective-loss dose fixed; compare successful demonstrations with authentic recovery additions, and separately vary test-first supervision. Full-rollout gates precede sealed final evaluation.
3. Compare exact-symbol graph/caller evidence with equivalent-token source retrieval on multi-file contract tasks. Report execution/source graph coverage, irrelevant retrieval and tool overhead alongside repair accuracy.
4. Test whether a multi-step/full-rollout diagnostic predicts adapter collapse better than isolated-prefix accuracy using multiple independently trained adapters, without selecting the rubric on final outcomes.

Use task-level paired statistics, repository-group uncertainty, explicit environment-invalid criteria and complete provenance. Include all failed runs and configuration changes; do not count four samples from one prefix as four independent issues. The existing three public repos and repeated diagnostic set limit external validity.

Novelty is **not verified** here: no fresh systematic primary-literature review was performed. Before a paper claims an original method, compare to the cited graph/retrieval, agent-training and long-context work from repository literature maps, then update with primary papers. Prior-art gap claims, causal mechanism claims and hidden-score forecasts would currently be unsupported. The most defensible near-term output is an auditable negative-result/methodology study, strengthened by one controlled independent intervention if later approved.
