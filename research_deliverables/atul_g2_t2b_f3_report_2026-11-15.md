# G2 F3 evaluation, Base vs T2b

The private Kaggle evaluation ran as notebook version 1 on four NVIDIA L4 GPUs with internet disabled.

Kaggle terminal status was **COMPLETE**. Pre-launch available quota was 30.000 h, and terminal available quota was 20.614 h. Account used quota changed by 9.386 h.

Source baseline commit was `5d6b1a295fd8e1afbff63d7dd401180d3a481d51`. The submitted notebook was generated from an isolated copy of the committed F3 notebook. Its evaluation config already selected `t2b_lora`, but the embedded submission archive still declared `adapter: t2a_lora` in `agent.yaml`. The correction changed that declaration to `adapter: t2b_lora`. Every other notebook byte outside the embedded archive and every other archived file matched the committed notebook. The recovered manifests confirmed a Base arm with no adapter and a T2b arm with only `t2b_lora`. The preflight matched the existing T2b training version 1 adapter to the training receipt by SHA-256.

Scheduling budget was 10.50 h, with a 10.75 h watchdog. The paired blocks used concurrency 6, block size 24 and a 12 min task cap.

Paired outcomes are available for **101/101** tasks. Base completed 101/101 and T2b completed 101/101.

| Cohort | Pairs | Both resolved | Base only | T2b only | Neither |
|---|---:|---:|---:|---:|---:|
| All observed pairs | 101 | 38 | 9 | 6 | 48 |
| Sensitivity cohort | 100 | 37 | 9 | 6 | 48 |

The sandbox dry run failed its gold-patch check on one task. Both arms resolved that task in the recovered GPU report. The primary result uses all 101 observed pairs: Base **47/101 (46.5%)** and T2b **44/101 (43.6%)**. The 100-task row applies the planned dry-run exclusion as a sensitivity check. The exclusion does not change the discordant pairs or the decision.

Paired net change was **-3**. The exact two-sided McNemar p-value was **0.607239** on the observed pairs.

Resolution with the dry-run exclusion was Base **46/100 (46.0%)** and T2b **43/100 (43.0%)**. A 95% paired task bootstrap interval for T2b minus Base was [-11.0, 4.0] percentage points.

The registered F3 rule calls for stopping this line.

| Arm | Resolved | Median task seconds | Blocks |
|---|---:|---:|---:|
| base | 47 | 333.57 | 5 |
| lora | 44 | 403.27 | 5 |

Reported arm counters and block metrics are preserved below.

```json
{
  "arms": {
    "base": {
      "resolved": 47,
      "n": 101,
      "task_sec_median": 333.57,
      "time_cap_hits": 9,
      "call_cap_hits": 49,
      "empty_patch": 18,
      "errors": 11
    },
    "lora": {
      "resolved": 44,
      "n": 101,
      "task_sec_median": 403.27,
      "time_cap_hits": 14,
      "call_cap_hits": 52,
      "empty_patch": 19,
      "errors": 9
    }
  },
  "blocks": [
    {
      "arm": "base",
      "block": 0,
      "tasks": 24,
      "done": 24,
      "resolved": 15,
      "wall_min": 22.7,
      "tasks_per_h": 63.37,
      "task_sec_median": 278.61
    },
    {
      "arm": "lora",
      "block": 0,
      "tasks": 24,
      "done": 24,
      "resolved": 15,
      "wall_min": 29.0,
      "tasks_per_h": 49.72,
      "task_sec_median": 403.27
    },
    {
      "arm": "base",
      "block": 1,
      "tasks": 24,
      "done": 24,
      "resolved": 11,
      "wall_min": 29.9,
      "tasks_per_h": 48.18,
      "task_sec_median": 353.44
    },
    {
      "arm": "lora",
      "block": 1,
      "tasks": 24,
      "done": 24,
      "resolved": 8,
      "wall_min": 34.2,
      "tasks_per_h": 42.16,
      "task_sec_median": 382.82
    },
    {
      "arm": "base",
      "block": 2,
      "tasks": 24,
      "done": 24,
      "resolved": 9,
      "wall_min": 31.3,
      "tasks_per_h": 45.98,
      "task_sec_median": 353.0
    },
    {
      "arm": "lora",
      "block": 2,
      "tasks": 24,
      "done": 24,
      "resolved": 8,
      "wall_min": 38.5,
      "tasks_per_h": 37.43,
      "task_sec_median": 467.91
    },
    {
      "arm": "base",
      "block": 3,
      "tasks": 24,
      "done": 24,
      "resolved": 12,
      "wall_min": 32.9,
      "tasks_per_h": 43.78,
      "task_sec_median": 364.26
    },
    {
      "arm": "lora",
      "block": 3,
      "tasks": 24,
      "done": 24,
      "resolved": 13,
      "wall_min": 34.0,
      "tasks_per_h": 42.4,
      "task_sec_median": 417.3
    },
    {
      "arm": "base",
      "block": 4,
      "tasks": 5,
      "done": 5,
      "resolved": 0,
      "wall_min": 9.1,
      "tasks_per_h": 32.9,
      "task_sec_median": 427.4
    },
    {
      "arm": "lora",
      "block": 4,
      "tasks": 5,
      "done": 5,
      "resolved": 0,
      "wall_min": 8.7,
      "tasks_per_h": 34.61,
      "task_sec_median": 372.47
    }
  ]
}
```

Runtime log contains 0 watchdog markers, 13 context-error string matches and 0 CUDA OOM string matches. These are log mentions, which can repeat for a single event.
The evaluator reported 4.7 h through cleanup and exited with code 0.

All 15 recovered files matched the SHA-256 checksums in the version 1 receipt. The recovered aggregate result JSON has SHA-256 `3455add8e739648d49beea11e701b64bff6b37cf77c5309445e9756f0718a90a`.
