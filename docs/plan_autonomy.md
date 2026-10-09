# Autonomous plan (from 9 Oct 2026): measure first, then build the method on what the data shows

Shrey gave full autonomy on 9 Oct. This plan applies the lessons in `reports/` (why six paper designs failed):
- freeze one direction;
- run before designing;
- treat closeness-2 as normal and judge ideas by demonstrated advantage;
- time-box every idea to a kill test of 2 days or less;
- use critics as gates on experiments, not on ideas.

## Frozen paper direction (until Oct 23): execution graphs of the competition's own tasks
**Why this one:**
- It uses assets other teams don't have measured: the provided (undirected, single-label) graph, Gemma-4-31B traces, and the scorer-faithful environment (54/128 tasks pass gold).
- It runs on CPU now.
- It feeds three earlier ideas at once: IFG necessity, CERT credit, and BMM's imagined-vs-executed measurement.

**Instrument (Oct 9–11, CPU):** `kaggle/exec_graph/`. For each scorer-faithful task:
1. Trace the fail-to-pass tests in three modes:
   - empty patch: the failing run, giving the executed call graph;
   - gold patch;
   - leave-one-gold-function-out patches: necessity.
2. Join the traces with the provided graph, an AST call graph and the issue text.

**Measurements (Oct 11–13), looking for a surprise:**
- M1: what share of executed edges the provided graph contains, with direction and edge kind.
- M2: where the culprit sits on the executed path.
- M3: necessity structure of multi-function fixes.
- M4: how well trace-based localisation (Ochiai over failing vs passing tests) and issue-text BM25 do, compared with Gemma's v5 localiser, which we have per-task results for.

**Decision gate (Oct 14):** choose the method around the strongest surprise. If nothing surprising shows up, the instrument plus the measurements become the paper's analysis core, and the method is the cheapest one the data supports.

**Kill-test window (Oct 14–21):** one method, one kill test of 2 days or less, with GPU from Oct 10.

## Predictions recorded before any execution-graph data (9 Oct, so "surprise" is defined in advance)
**Definition:** a result is a *surprise* if it lands outside these ranges.

| ID | Measurement | Prediction |
|---|---|---|
| M1 | Share of executed workspace caller→callee edges (fail-to-pass tests, empty mode) whose endpoints are joined by an edge in the provided graph, direction ignored | 55–85% |
| M1b | Share of executed edges where the provided graph has either endpoint at all | ≥ 90% |
| M2 | Gold function executed by at least one failing test in empty mode | 70–90% of tasks |
| M2b | Median call depth from the test function to the gold function in the executed graph | 2–4 |
| M3 | Among multi-function fixes, LOO patches where dropping one function still resolves (that function is not necessary) | 10–35% |
| M4 | Function-level spectrum localisation (Ochiai: failing F2P runs vs passing P2P tests) puts gold in the top 5 | 40–70% of tasks |
| M4b | Issue-text BM25 over function bodies puts gold in the top 5 | 20–40% |
| M4c | Gemma v5 localiser named the gold function (`results/runs/r3`) | ~48% (52/109 known) |

## First results (9 Oct, v1/v2 traces, frozen-frame filter applied in analysis; v3 re-trace pending)

| ID | Prediction | Result | Verdict |
|---|---|---|---|
| M1 | 55–85% | **17%** of executed repo edges are provided-graph edges (95% of endpoints mapped) | **Surprise.** Same-class `self.m()` calls (3,156) and dunder/protocol calls (1,842) are 0% covered; same/cross-file calls about 50%. The provided graphs contain **no sibling-method edges at all** (static check) and attribute `self.m()` to the class node (4,830 cases) |
| M2 | 70–90% | 69% gold executed by a failing test | At the edge |
| M2b | 2–4 | median depth 1 | Mild surprise: unit tests mostly call the gold function directly |
| M3 | 10–35% | **46%** (19/41) of gold functions in multi-function fixes are not needed for the tests | Surprise (consistent with IBugDC's "divisible" bugs) |
| M4 | 40–70% | SBFL top-5 47%, top-1 16% | As predicted |
| M4b | 20–40% | BM25 top-5 40% | Top of range |
| M4c | ~48% | 48% | As predicted |

### Pre-registered kill test KT-1 (fixed before running; CPU, ≤1 day)
**Hypothesis:** the supplied graph is weak for issue→fix navigation because it lacks intra-class and protocol calls. Repairing exactly those edges, statically, will make the fix reachable from issue-named seeds.

**Arms** (all on the 129 public tasks' provided graphs):
- **P:** provided graph.
- **R:** P ∪ AST-resolved `self.m()` / `cls.m()` / `super().m()` edges ∪ class→`__init__`/dunder protocol edges.
- **D:** degree-preserving rewire of R (control).
- **X:** R with the repaired edges added at random (same count) (control).

**Metric:** share of tasks whose gold node is reachable within ≤2 hops (undirected, as an agent would traverse) from the clean issue seeds of A2.

**Thresholds:**
- PASS if R − P ≥ +10 pp **and** R − X ≥ +5 pp **and** R − D ≥ +5 pp.
- FAIL if R − P < +5 pp.
- In between is ambiguous.

**Validation of R:** R recovers ≥ 70% of the same-class and dunder edges executed in the traces.

## Main track (parallel)
- **Oct 10 (quota reset):** push the training probe v3 (`kaggle/probes/check_l4`, about 1 GPU-h). The throughput sets the whole SFT plan.
- Then the Gemma tool-call re-render check, and an SFT LoRA for tool-format reliability and loops, if the probe gives at least 300 tok/s.
- Leaderboard submissions only for scorer-path-validated variants (1 per day).

## Rules for myself
- No heavy runs on the Mac.
- Every experiment gets a pre-registered metric and threshold in `docs/experiments.md` before results.
- No direction change without a failed kill test.
