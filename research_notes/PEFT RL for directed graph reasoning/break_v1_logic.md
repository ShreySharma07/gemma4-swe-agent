# Break report on architecture v1 (GRAFT): logic and edge cases (Breaker 1, 8 Oct 2026)

**Target:** `architecture_v1.md` (read in full, including A1–A20), with `phase3_critic.md` and `phase3_redteam.md` skimmed for what has already been raised.
**Method:** for each component, a small worked counter-example (code or graph sketch) where the logic fails, misleads, or can be gamed. Every claim here is **INFERRED** (my reasoning) unless marked otherwise. Nothing was run.
**Severity scale:**
- **fatal**: invalidates a pre-registered hypothesis or the reward as specified;
- **major**: biases a headline comparison or opens a likely exploit;
- **minor**: noise, a reporting gap, or an easy patch.

Breaks are numbered **K-xx** so v2 can cite them.

---

## 0. Bottom line

- **The set reward is gameable as written (fatal, K-30).** The trajectory set term averages over (listed ∪ gold), and unlisted non-gold nodes are not scored. So padding the CANDIDATES block with easy negatives at p=0.05 dilutes any gold penalty. Worked numbers in §5 show that **omitting all gold plus padding scores better than honest listing**.
- **The compiled world has a topology-invariant solution path (fatal for H2/H3 logic, K-04, K-12).** Dense visible unit contracts with exact expected values reduce most localisation to "find the deepest failing contract". O1 body mutations of about 40 fixed templates reduce to template-anomaly spotting. Both work equally in arms (b), (c) and (d). Training on them makes (b)≈(c) true for reasons unrelated to topology, and pulls (b)−(d) towards 0.
- **The rewire (c) changes more than "which node feeds which" (major, K-20 to K-24).** It destroys module locality, import structure, coverage-depth correlation and class-hierarchy coherence. DPCS (O2) swaps also leave an import or locality fingerprint. So (b)>(c) can come from a lexical *file-locality prior* that transfers to real code, rather than from graph-reasoning fidelity.
- **Arm (d) is not a clean graph-free control (major, K-26 to K-28).** It changes bodies, length and success rate. Its issue text effectively names the bug node, and its item training teaches a lexical rule that should hurt deep strata on real items. H2 can therefore pass because (d) is *harmed*, while (b) = base.
- **Inference is pseudo-replicated (major, K-50).** There is one training run per arm, while the CI is bootstrapped over items and 8 repo clusters. H3 (TOST) and H2 can pass on training-seed noise. Δ_struct (H4) is mechanically coupled to H1.

---

## 1. Compiler (C1 tracer, C2 topology, C3 compiler)

### K-01 "Depth" in a union-of-tests graph is mostly 1 — **major** (violates the A8/A16 spirit and the H2 premise)
C2 takes the union of `G_exec` over *the whole test suite* and measures depth from test roots. Real suites are mostly *unit* tests that call internal helpers directly.

Sketch (requests-like):
```
test_session_send  ─▶ Session.send ─▶ HTTPAdapter.send ─▶ _build_conn ─▶ _parse_url
test_parse_url     ─▶ _parse_url                       (unit test)
test_build_conn    ─▶ _build_conn                      (unit test)
```
The shortest-path depth of `_parse_url` from *any* root is 1, not 4. With hundreds of unit tests, most nodes sit at depth 1–2. Consequences:
- the "bug depth ∈ {3, 4, ≥5}" curriculum cannot be sampled, or is sampled only from untested corners;
- the (b) vs (d) contrast shrinks;
- each real test becomes a root test, so the compiled root tests already pin most nodes.

**Fix.** Define roots as *integration-level* tests only (for example, tests whose E(t) spans ≥3 modules), or use public-API entry nodes. Delete the unit-test roots from the source graph before computing depth. Report the (b) depth histogram on CPU *before* B2. This is a 0 GPU-h check.

### K-02 SCC condensation creates mega-hubs; "<3% edges lost" is the wrong statistic — **major** (A16)
Real repos contain large SCCs. Examples (INFERRED): rich's renderable protocol (`Console.render` ↔ `__rich_console__` of Table, Panel, Padding ↔ `render_lines`), recursive-descent parsers, visitor and AST walkers, and JSON encoders.

Collapsing a 30-node SCC into one node:
1. makes a node whose in- and out-degree is the *union* of 30 functions' degrees, i.e. a hub that does not exist in the source;
2. forces one body template to fold 30+ child values;
3. removes the only place where recursion depth (a value-dependent traversal) lives.

The fraction of edges lost can be <3% while the fraction of *nodes* in the largest SCC is 15%. The node fraction, and the degree of the condensed node, are the statistics that matter.

**Fix.**
- Report |largest SCC| / |V| and the condensed node's degree.
- Drop source graphs with largest SCC >5% of |V|, or compile SCCs as bounded recursion with a fuel argument (`def f(x, fuel=3)`), which keeps the edges and terminates.
- Do not count SCC-internal edges as "lost <3%". Count them as structurally replaced.

### K-03 The BFS frontier truncates the deep tail — **minor** (A8)
"Induced connected subgraph of 200–1,500 nodes grown by BFS from test roots" cuts the graph at the BFS frontier. Frontier nodes become artificial leaves, their out-degree drops to 0, and the longest paths (the very tail the curriculum wants) are cut first.

**Fix.** Grow by *task* (all descendants of a sampled integration root, capped by size) rather than by global BFS. Report the out-degree of frontier nodes versus the source graph.

### K-04 Dense visible unit contracts make localisation topology-invariant — **fatal for the H2/H3 logic** (A19 is satisfied too well; A8)
C3 attaches unit tests, with *exact expected values from the gold program*, to a random 40% of internal nodes, and makes them visible. Running `pytest -x` on the visible file then shows:
- the bug node's own contract fails, if it has one;
- every contracted *ancestor* fails;
- everything else passes.

The bug is "the failing contract with no failing contracted descendant". That is a one-shot read of pytest output, with no traversal.

Worked example (depth 5 by the root metric):
```
root_test ─▶ n1 ─▶ n2 ─▶ n3 ─▶ n4 ─▶ n5*   (bug in n5)
contracts visible on: n2, n5      → pytest fails test_n2, test_n5
                                   → "deepest failing" = n5 → done, effective depth 0
```
- P(bug node has a visible contract) ≈ 0.4 × (1 − holdback). In those tasks the root depth is irrelevant.
- Otherwise the effective search distance is the hops from the *deepest failing visible contract* to the bug, which is ~1/0.4 ≈ 2.5 hops in expectation, whatever the root depth.

This strategy works identically on (b), (c) and (d). If it dominates the successes, every arm learns it, and so:
- (b)≈(c) holds trivially, and the H3 "pass" says nothing about topology;
- the H2 gap shrinks;
- K1's negative depth slope can still appear (see K-08), so the gate does not catch it.

**Fix.**
- The difficulty variable must be **distance from the nearest failing *visible* test to the bug** (call it d_obs), not root depth. Stratify, curriculum and depth-match (b)/(c) on d_obs.
- Sample the visible-contract set *first*, then place bugs only at nodes with d_obs ≥ the target. Disclose that the "bug nodes are uncontracted" prior is learnable, and report how strongly it is learned.
- Add a **no-LLM baseline solver** per arm: deepest-failing-contract plus a spectrum ranking. Report the fraction of tasks it localises. If it localises >50% of tasks in (b), the task is not exercising traversal, and K1 must be re-run against d_obs.

### K-05 Value cancellation and masking along diamonds — **major** (selection effect hidden behind C5)
Templates are deterministic arithmetic, string and fold operations, so masking is common:
```python
def a(x): return b(x) - c(x)      # diamond a→b→d, a→c→d
def b(x): return d(x) + 1
def c(x): return d(x) + 3
def d(x): return x * 7            # O1 bug: x * 8   → a(x) unchanged (= -2)
```
Other maskers: `% m`, `min`/`max` saturation, `len(s)` after a string transform, `dict.get(k, default)`, and `sorted(set(...))`.

C5(ii) drops tasks where *no* F2P fails, but it keeps tasks where the bug is observable on only *some* paths. That creates two silent selection effects:
1. **Diamond-heavy topologies (real (b) has more diamonds than (c), see K-22) get more tasks dropped.** The surviving (b) tasks are biased toward non-diamond regions, so the arms are not paired in bug location even though C2 pairs them in graph.
2. **The surviving bugs are observable along the path with the fewest maskers.** On those paths, local reading of one template is often sufficient.

**Fix.**
- Report the C5 drop rate per arm and per depth.
- Pair tasks across (b) and (c) only on *bug sites that survive in both*, or re-draw until per-arm drop rates match within ±5 pp.
- Ban non-injective templates (mod, min/max, len) on the path from bug to the F2P test, or tag them and stratify.

### K-06 Hubs make bugs trivial and make the P2P guard vacuous — **major** (A20)
Take a hub `_norm(x)` with in-degree 180, as in logging or validation helpers. In a value-flow compiler *every* edge carries a value, so an O1 bug in `_norm`:
- fails almost every root test;
- leaves P2P ≈ ∅, so the −0.5 P2P guard never fires;
- is localised by "the bug is the highest-in-degree node on the intersection of failing coverage" (spectrum, no reading).

In real code, most hub edges (logging, `__init__`, validation that returns None) do **not** carry values, so this observability profile does not transfer.

**Fix.**
- Make a configurable share of edges *effect-only*. The return value is ignored or used only in a guard, matching the real share, which C1 can estimate from whether call results are used.
- Exclude bug nodes whose test coverage exceeds 30% of root tests, or stratify by coverage.
- Require |P2P| ≥ 5 per task.

### K-07 The compiled dark edges are not dark — **major** (A20, transfer)
| Compiled form | One-hop grep that finds it |
|---|---|
| dispatch: `self.m()` with one override | `grep -n "def m("` lists base and override. With one subclass it is unambiguous |
| registry: `REGISTRY["k"](x)` | `grep -n '"k"'` finds `REGISTRY["k"] = f` |
| decorator: wrapper calls `fn(*a)` | `grep -n "@deco"` lists the wrapped functions |
| callback: `run(cb=f)` | `grep -n "cb=f\|run("` |

Real dark edges are harder: `getattr(obj, name)`, entry points, signals, `__getattr__`, framework routing (FastAPI dependency injection), and `**kwargs` forwarding. The compiled "dark" stratum is one grep away from lexical. So C9's "dark" stratification on *real* items measures a harder thing than training ever exercised.

**Fix.**
- Compile dispatch with ≥3 sibling overrides, where the executed one is determined by a constructor argument chosen elsewhere in the graph.
- Compile registry keys that are *computed* (`REGISTRY[prefix + kind]`).
- Add a `getattr(module, name_str)` form.
- Report a "grep-distance" for each dark edge.

### K-08 Long chains exceed the 12k trajectory cap, so K1 passes on budget rather than reasoning — **major** (A4, G2 interpretation)
A depth-8 chain across 8 modules costs, with `read_file` capped at 150 lines:
- ≈1.5–2k tokens per hop including tool overhead;
- 8 hops ≈ 12–16k, which exceeds the 12k cap before any edit.

Deep tasks therefore fail on **budget**, so K1's negative depth slope is guaranteed and says nothing about traversal. Round-2 depth ≥5 successes are then mostly the K-04 shortcut (no reading). The curriculum thus selects shortcut trajectories exactly in the bucket meant to teach deep traversal.

A further mismatch: ADK compaction (~14k) never triggers under a 12k cap, so training never contains compaction while deployment does.

**Fix.**
- In K1, treat a turn-cap or token-cap hit as a *separate* outcome (competing risks), and regress success on d_obs while controlling for tokens consumed.
- Size repos so the gold path's file reads fit in ≤50% of the cap.
- Report the fraction of depth ≥5 successes that read every file on the gold path.

### K-09 Static edge ≠ executed edge in the tracer's own labels (mocks, branches, name collision) — **major** (A9, C9 labels)
1. **Mocks.** `mock.patch("pkg.client.send")` means the body shows `self.send(req)`, but the executed callee is a MagicMock. The label is "not called" even though the displayed edge says "called". An edge-following model is penalised on real items.
2. **Branches.** `if sys.platform == "win32": _win_impl()`. Distractors "matched on file and degree" are very often exactly these statically reachable but unexecuted siblings. On them, static graph reasoning *cannot* score, so the item tests value and branch reasoning, not traversal.
3. **Lexical type by name.** "lexical = callee name followed by `(` appears in the caller's body". `self.get(` matches `Session.get`, `dict.get` and `Headers.get`, so dispatch edges whose method name appears textually are labelled lexical. The dark share is under-measured (A20), and the eval's lexical/dark strata are contaminated.

**Fix.**
- Drop eval tests that patch pool functions, or exclude the patched targets.
- Label each distractor as *statically reachable from the test or not*, using name-reachability over the pool (NameRTS-style), and stratify AP by it.
- Type an edge as lexical only if the name at the call site resolves *uniquely* in the pool.

### K-10 No dead code and no untested nodes in compiled repos — **minor** (transfer)
Every compiled node is reachable from a root (BFS growth), so every synthetic item distractor is executed by *some* test. On real items, distractors include dead or never-tested code with tell-tale names (`_deprecated_*`, `__repr__`, platform shims). Training teaches "called by this test vs another"; real eval also rewards "is this ever called". The two skills are not the same.

**Fix.** Compile 10–20% unreachable nodes (dead code) and 10% nodes reachable only from non-F2P tests. Match the real distractor composition, which C9a can measure.

---

## 2. Bug operators (C4), with focus on DPCS (O2)

### K-11 DPCS leaves an import and locality fingerprint — **major/fatal for "purely topological"** (A8, A19)
The layout is copied from the source repo, so most real edges are intra-module or go to a few util modules. A swap (a→b, c→d) ⇒ (a→d, c→b) between *type-compatible callees anywhere* usually makes a call cross a module boundary that was never crossed.
- If imports are generated from the **bugged** graph: `mod_a.py` gains `from mod_c_area import d`. That is an import used once, from a module `mod_a` never otherwise touches. `grep import` plus "odd one out" localises both sites without traversal.
- If imports are generated from the **gold** graph: `NameError: d` immediately, with a traceback naming the site. That is the same giveaway as the TypeError the critic removed.

```python
# mod_render.py (bugged)
from .mod_render_util import _pad          # used by 6 functions
from .mod_http_codec import _chunk_size    # NEW: only use in the file → the swap site
def render_line(s): return _chunk_size(s)  # was _pad(s)
```

**Fix.**
- Restrict O2 swaps to callee pairs **in the same module, or in modules already imported by both callers**. Equivalently, preserve the (caller module, callee module) pair.
- Alternatively, compile all calls through a namespace object (`api.f(x)`) so imports carry no information. That makes everything uniform in all arms.
- Report the AUC of a "single-use import" heuristic for locating O2 sites. It must be ≈0.5.

### K-12 Generator learning: template anomaly detection solves O1 without the graph — **fatal for H2/H3 logic** (A5, A8)
With about 40 deterministic templates reused across thousands of nodes, a LoRA learns each template's canonical surface form, e.g. `fold_mod` always uses `% 97` and `acc += v * 3`. An O1 mutation (`% 98`, `<` ↔ `<=`, `+1`) yields a body that is **not an instance of any template**.

The policy "grep for non-canonical constants and operators" localises O1 bugs, which are 50% of tasks, with zero traversal. It works identically in (b), (c) and (d), so it inflates (b)≈(c) and shrinks (b)−(d).
```python
# canonical instance (seen 400× in training)      # mutated (O1)
def nb(x): return sum(v * 3 for v in x) % 97      def nb(x): return sum(v * 3 for v in x) % 98
```

**Fix.**
- **In-distribution mutations.** Every template takes per-instance random constants and operators drawn from a family. A bug replaces the body with *another valid instance of the same family* (different constant or operator), so the bugged body is indistinguishable from a correct one without its contract or tests.
- Measure it with a "body-only discriminator" probe. The base or trained model is shown single bodies and asked "is this mutated?". AUC must be ≈0.5 before arms are trained.

### K-13 DPCS where the two sites are one function — **major** (H6 is gamed)
"Pick two call sites a→b and c→d" does not require a ≠ c. If a = c (one caller calls both b and d), the swap becomes:
```python
def a(x):  return combine(b(x), d(x))   →   return combine(d(x), b(x))
```
This is a single-function, adjacent-token argument swap, i.e. a classic *local* bug. The commit set has size 1, and H6 ("both sites listed EDIT") is satisfied by listing one node.

**Fix.** Require a ≠ c, preferably in different files, with graph distance(a, c) ≥ 2. Report H6 only on such tasks.

### K-14 DPCS is ill-defined for registry and decorator edges — **major** (C4 semantics)
- **Registry.** The edge is `REGISTRY["k"](x)`. "Swapping callees" is implemented either by editing the call-site keys (two sites) or by swapping the dict entries `{"k1": f, "k2": g}` → `{"k1": g, "k2": f}` (**one** site, which affects *all* callers of both keys). The gold fix, and therefore the commit-set size, depends on an implementation choice, and the one-site variant is not "two-site jointly necessary".
- **Decorator.** The "edge" is wrapper→wrapped. Swapping means moving `@deco` lines, which also changes every caller of the wrapped functions. Degree preservation at the type level does not imply local wiring semantics.
- **Dispatch.** Swapping which override `self.m()` reaches means changing a class base or a method name. Either can be one edit with many effects.

**Fix.** Restrict O2 to **lexical and callback** edges, where a call site is an unambiguous single token. Give dark-edge bugs their own operator with an explicitly defined gold, and report them as a separate stratum.

### K-15 Value-range giveaways survive type compatibility — **major** (the TypeError giveaway in another form)
Type and arity match, but the *caller's template is calibrated to the child's value range*:
```python
def a(x): return TABLE[b(x)]        # b returns 0..7 by construction
# after swap a calls d, which returns 0..10_000 → IndexError at a, traceback names the line
def c(k): return MAP.get(d(k), 0)   # d returns keys in MAP; b returns ints → silently 0
```
- The first site crashes with a traceback that points to it (one-site localisation).
- The second site is silently masked (K-05). The necessity check then drops the task, or keeps it only when the other path is observable.

**Fix.** Use total templates only (`TABLE[v % len(TABLE)]`) and require that **no** F2P failure in O2 tasks is an exception. Report the exception-vs-assert ratio per operator.

### K-16 Kwarg, local-name and docstring leaks — **minor/major** (implementation-dependent)
- If call sites use keywords derived from the callee's parameters (`b(width=...)`), a swap to d with different parameter names raises TypeError. If locals are named after callees (`v_b = d(x)`), the mismatch is lexical.
- Docstrings emitted from templates describe the computation and expose O1 mutations.

**Fix.** Positional calls only. Local names drawn independently of the callee. No docstrings, or docstrings sampled from the identifier pool.

### K-17 Necessity is checked on gold subsets, not on all fixes; alternative valid fixes are penalised — **major** (C5/C7 coupling)
C5(iii) shows that *gold's* strict subsets fail. It does not show that every passing fix touches both sites. For DPCS, an agent can edit callee bodies instead: make `d` behave like `b` when called from `a`'s context, or route through a shim. If both callees have in-degree 1 after the swap, a body-edit fix at {b, d} passes all F2P and P2P.

The set term then scores {a, c} as gold, so the agent lists b and d as EDIT p=0.95 and is **penalised heavily for a correct, different fix**. R's two terms disagree, and the agent is taught to avoid valid alternatives.

**Fix.** Define the set gold as *any minimal passing edit set* (enumerate on CPU: try reverting each candidate's swap or body). Alternatively, score the set term only when f_F2P < 1, or tri-state any listed node whose edit is part of a passing patch.

### K-18 Hardcoding the issue's root-test case earns partial reward — **major** (reward hack, C3/C7)
The issue text is generated from the failing root test's assertion (expected vs actual). `f_F2P` is a *fraction*. An agent writes:
```python
def entry(x):
    if x == [3, 1, 4]: return 1729     # value copied from the issue
    ...
```
- The visible root test passes.
- P2P stays intact, because the branch fires only on that input.
- With F2P = {root test, 1 held-back unit}, R ≈ 0.5.

In arm (d), where entry is the bug node, this is often the *cheapest* path. K=4 groups will reinforce it whenever honest attempts fail.

**Fix.** The compiler has the gold program, so generate **hidden F2P tests on fresh random inputs** (property-style, expected values from gold) that never appear in the issue. Make the F2P reward binary (all pass) for the success filter, keeping the fraction only as a shaped term. Add a static check that rejects patches containing literals copied from the issue.

### K-19 "Swap is a no-op" cases are filtered, but the filter biases toward distinguishable callees — **minor**
If b(x) = d(x) on the test inputs (the same template family with the same children), C5(ii) or (iii) drops the task. The survivors are swaps between *visibly different* callees: different templates and different value ranges, exactly the K-15 regime. The difficulty of the surviving O2 tasks is lower than the operator suggests.

**Fix.** Report the drop rate. For survivors, sample callee pairs from the *same* template family with different constants (K-12 fix), so they differ only through tests.

---

## 3. Rewire control (c): what else it changes (A8)

### K-20 Coverage per node and spectrum difficulty change; degree-preserving ≠ coverage-preserving — **major**
Worked example. Two subsystems share a utility U.
```
(b) real:   T1→A→B→U      T2→D→E→U       A→C   D→F
            in-deg U = 2; U reached by {T1, T2}; depth(U)=3
            bug in U → fails {T1,T2}; ∩ coverage(T1,T2) = {U}  → trivially localised by spectrum
(c) swap (B→U, D→F) ⇒ (B→F, D→U):
            T1→A→B→F, A→C        T2→D→U, D→E
            in-deg U still 2? (E→U, D→U) yes; out-degrees unchanged
            U reached by {T2} only; depth(U)=2
            bug in U → fails {T2}; candidates {D,E,U} → spectrum ambiguous
```
Per-type degrees are preserved, and depth shifted by one, which is within the ±1 tolerance. Yet the *number of roots reaching U*, the test coverage of U, and the spectrum ambiguity all changed.

In real code, deep shared utilities are covered by many tests: coverage and depth are positively correlated. A random rewire decorrelates them. Bug difficulty therefore differs systematically between (b) and (c), in a direction that depends on the graph. Success-count equalisation does not fix that, because the *skill* each arm's successes require differs.

**Fix.**
- Report, per arm: the distribution of #roots reaching each node; |desc(v)| and |anc(v)|; spectrum-localisability, i.e. the rank of the bug under an Ochiai score computed by a no-LLM script.
- Add "Ochiai rank of bug" to the (b)/(c) task-matching criteria, alongside depth.

### K-21 Modularity, file locality and imports are destroyed, so (b)>(c) for a lexical reason — **major/fatal for interpreting an H3 failure**
The layout is "the same partition reused for (c)". Real call graphs are modular: most edges are intra-file. A double-edge swap among the |E| edges of one type is approximately uniform, so after 10·|E| attempts the intra-module edge fraction falls to roughly Σ_m (|m|/|V|)², e.g. 5% for 20 equal modules, down from a real ~50–70% (INFERRED).

In (b), "the callee is probably in this file or a file this file imports" is a correct, cheap prior. It transfers to real repos, where it is also true. In (c) it is false. So:
- **(b)>(c) on real items** can come from a file-locality prior. The architecture would report this as "real topology carries transferable structure beyond degree and depth, a graph-ML finding". It is closer to a code-layout heuristic.
- **The import graph in (c)** is dense and random, with more imports per module. That changes tokens, read_file content and trajectory length, all of which are confounds.

**Fix.** Use a **block-preserving, level-preserving swap**: accept (a→b, c→d) ⇒ (a→d, c→b) only if module(b) = module(d), or the (module(a), module(b)) and (module(c), module(d)) pair counts are preserved (degree-corrected stochastic-block null). Also require depth(b) = depth(d) (see K-23). Report modularity Q and the intra-file edge fraction for (b) and (c). If v2 also wants a "locality destroyed" arm, it is a fourth arm, not a silent property of (c).

### K-22 Motifs: diamonds, feed-forward loops and transitive triangles drop — **major** (A8, interacts with K-05)
Real code has many a→b, a→u, b→u triangles: a function and its helper both call a shared utility. Rewiring reduces these toward the random-graph expectation. Fewer diamonds means:
- fewer masking opportunities (K-05), so a different C5 drop rate and different surviving bugs;
- fewer multiple paths from root to bug, so less path ambiguity;
- a different answer to "how many distinct paths must be read".

**Fix.** Report triangle, diamond and FFL counts per arm (A8 already asks for this). Either add a motif-count acceptance band to the rewire, or state H3 as "fidelity beyond {degree, depth, block, motif} profile".

### K-23 Depth-matching by KS p > 0.2 is statistically wrong and creates selection — **major**
1. The per-node depth histogram is discrete with heavy ties. The KS p-value is invalid there (conservative and unpredictable).
2. **Power depends on |V|.** At |V|=1,500 the test rejects tiny shifts, so many source graphs are dropped. At |V|=200 it accepts large shifts.
3. **Dropping the source graph from all arms selects** the graphs whose depth is insensitive to rewiring: shallow, star-like, hub-dominated. On those graphs (b) and (c) are alike *by construction*. That biases H3 toward passing.

**Fix.** Make depth preservation *exact by construction*. Only propose swaps with depth(a) = depth(c) and depth(b) = depth(d), using longest-path layering. Level-preserving swaps keep each node's layer, and a cycle cannot form because edges always go to a deeper layer. Report how many source graphs this drops (it should be ~0) instead of a KS filter.

### K-24 Dispatch and decorator rewires need impossible class structures, so (c) code looks different — **major**
A rewired dispatch edge a→d means `self.m()` inside `a`'s class must resolve to `d`. If d is a method of an unrelated class, the compiler must invent inheritance: mixins, multiple inheritance, or classes with one method. (c) repos then have systematically more classes, deeper MROs and stranger class graphs. That is a lexical and length signature of the arm, and it changes difficulty.

Similarly, rewiring decorator edges independently of the callers of the wrapped function breaks the real coupling "callers of f actually call wrapper(f)".

**Fix.** Rewire only lexical and callback edges (dark edges stay as in (b)), or rewire dispatch only among methods within the same class hierarchy. Report the class count and MRO depth per arm.

### K-25 Cases where (b)≈(c) holds for reasons unrelated to "topology does not matter" — summary
| Mechanism | Why it gives (b)≈(c) | Detect it by |
|---|---|---|
| K-04 contract bisection | The learned strategy needs no traversal | The no-LLM deepest-failing-contract solver's localisation rate |
| K-12 template anomaly | O1 is half the tasks and is solved by surface form | The body-only discriminator AUC |
| Format and calibration dominate the item gain (K-41) | All arms get the same item format and share | Teacher-forced readout (K-41 fix); gain on the GF context alone |
| Too little data | ~15–40 successes per arm; the LoRA moves mostly from items | Training-seed spread (K-50) versus the 0.03 margin |
| Eval depth is mostly 1 | Real items with depth-1 positives readable from the test body | AP stratified by depth; does H3 hold within depth ≥3 only? |
| K-23 selection | The kept graphs are rewire-insensitive | Compare the drop set with the kept set |

Cases where **(b)>(c) for confound reasons:** K-21 locality and imports; K-24 class weirdness; K-20 coverage and difficulty; K-31 equalisation by repo size; K-05 differential C5 drop.

---

## 4. Graph-free control (d)

### K-26 (d) changes bodies, length and success rate, not just depth — **major** (H2 attribution)
In a depth-1 forest, most nodes are leaves. The template "consumes the node's inputs and its children's return values" therefore degenerates to inputs-only templates. The **body distribution differs** from (b): fewer folds and fewer child-combining templates. That violates "everything except topology fixed" (C3).

(d) trajectories are also shorter: read the test, read one function, fix. Equal-token subsampling then means (d) contributes **more distinct trajectories**, or (b) is cut further. Either way trajectory diversity is confounded with depth.

**Fix.** Use the same template mix as (b), with leaf nodes taking "virtual children" (constants drawn like child values). Equalise **trajectory count and token count** separately, and report both.

### K-27 In (d) the issue effectively names the bug node — **major**
The issue names the entry function called by the failing root test. In (d), that entry *is* the bug node, or its single parent. So (d) ≈ the K2 "GF-named" condition. DPCS in (d) swaps which function a *test* calls, so the failing test literally shows the swap. (d) is a localisation-given control, *plus* a different code distribution.

**Better control (d′).** Use the **same graphs as (b)**, with the issue naming the bug node(s): localisation given, traversal unnecessary, topology identical. This isolates "traversal demand" from "graph shape". Keep (d) only if the budget allows, as a secondary arm.

### K-28 (d)'s item training teaches a lexical rule that should transfer *negatively* — **major** (H2 passes for the wrong reason)
In depth-1 compiled repos, the call set of a test is almost exactly "names that appear in the test body", except for dark edges. Training items reward CALLED for names in the test and p=0.05 for everything else.

On real items with depth ≥2 positives, (d) will under-score them relative to base. H2 = AP(b) − AP(d) can pass with **AP(b) = AP(base)**, entirely from (d)'s harm.

**Fix.**
- Pre-register H2 as a conjunction: AP(b) − AP(base) > 0 **and** AP(b) − AP(d) > 0. Report AP(d) − AP(base) by depth stratum.
- Interpret H2 only in the depth-1 stratum, where (d)'s rule is valid, plus the overall result.

**Can the model still exploit structure in (d)?** Yes, in two ways. "At most one leaf helper" gives depth-2 paths. Dark edges from test roots (a registry call inside a test) are odd, test-specific idioms the model can learn. Neither is large, but (d) is not graph-free. It is "shallow and test-local".

---

## 5. Readout and reward (C7, C9)

### K-30 The set score is diluted by padding, and omission plus padding beats honesty — **fatal** (the C7 reward)
`S̄(τ)` is the **mean** of S_v over (listed ∪ gold). Unlisted non-gold nodes are not scored. Per-element values after dividing by |log 0.05| = 3.0:

| Element | p | S_v |
|---|---|---|
| gold | 0.95 | −0.017 |
| gold | 0.8 | −0.074 |
| gold | 0.5 | −0.231 |
| gold, omitted | (0.05) | −1.000 |
| non-gold | 0.05 | −0.017 |
| non-gold | 0.2 | −0.074 |

Task with |gold| = 2:

| Policy | S̄ |
|---|---|
| Honest: 2 gold at 0.8 + 3 examined non-gold at 0.2 | (5 × −0.074)/5 = **−0.074** |
| Pad: 2 gold at 0.5 + 200 irrelevant at 0.05 | (2 × −0.231 + 200 × −0.017)/202 = **−0.019** |
| **Omit all gold** + 200 padding at 0.05 | (−2 − 3.4)/202 = **−0.027**, still better than honest |

The "one line per function the agent has examined" rule cannot be enforced; at best it can be checked against read_file calls. Even "examined" padding works, because `grep -rn def` examines everything.

The η-term makes this worse (K-31). The Stage-A filter "all-gold-listed" also *selects* long lists (K-33).

**Fix.**
- Score a **fixed set not under policy control**: gold ∪ a compiler-sampled matched negative set of size 2·|gold|, plus any listed nodes.
- Use a *sum normalised by |gold|*: `S = (Σ_{v∈gold} S_v + Σ_{v∈listed∖gold} S_v) / |gold|`. Padding then costs, and omission costs 1/|gold| per gold.
- Separately cap the CANDIDATES length (e.g. ≤ 3·|examined files|).
- Put a unit test on the reward: the honest policy must strictly beat every padding and omission policy over the K-30 grid.

### K-31 The η per-line advantage reinforces *names*, punishes moderate gold, and leaves omission free — **major**
`Ã_t = Â_τ + η(S_v − S̄_v^grp)` is applied to *all tokens of the line*, including the qualname, and S̄_v^grp averages only over samples that **listed** v.
1. A non-gold v listed at 0.05 by one sample and at 0.5 by another gets a positive advantage on its *name tokens*. That raises P(emit this name), i.e. it reinforces padding.
2. A gold v listed at 0.5 when peers said 0.95 gets a negative per-line advantage. A sample that **omits** v gets no per-line penalty, since there are no tokens. Locally, omission dominates moderate-confidence listing, which is exactly the under-enumeration (72% `none`) the term was built to fix.
3. If only one sample lists v, S_v − S̄_v^grp = 0, so a rare deep-gold discovery gets **zero** per-line credit.

**Fix.**
- Apply η only to the **probability tokens** (`p=0.xx`), never to the qualname or label tokens.
- Score omitted gold in the group baseline as p=0.05, so S̄_v^grp is averaged over all K samples.
- Give the listing decision its own term, tied to the K-30 fixed set.

### K-32 EDIT|SKIP is redundant with p and inconsistent states are unscored — **minor**
`SKIP p=0.95` is ambiguous. AP and the reward read p, so the label tokens are free and drift. **Fix:** define p := P(EDIT) and drop the label, or reject inconsistent lines (treat as omitted).

### K-33 The RFT filter "all-gold-listed" selects long CANDIDATES blocks — **major** (Stage A)
P(all gold listed) rises with list length. Filtering on it without a precision condition makes Stage-A SFT targets systematically longer and padded. **Fix:** the filter requires S (K-30 fixed-set version) ≥ a threshold, not "all-gold-listed".

### K-34 Δ_struct (RW context) confounds edge use with implausibility detection and memorisation — **major** (H4)
1. **The rewired real code is detectably wrong.** `return _parse_header(width, height)`, where the shown `def _parse_header(raw: bytes)` takes one argument. `self.console.flush_buffer(x)` where `console` has no such method. The base model can notice corruption and *fall back to name and test priors*, ignoring bodies, so its AP(RW) is barely affected: small Δ_struct for "distrust", not for "no edge use".

   The trained arm saw only content-free code, where names never matched semantics. It learns that implausibility carries no information, so it follows displayed edges literally and gets a large Δ_struct. H4 then passes because **plausibility checking was trained away**, not because edge use was trained in.
2. **Memorisation.** S_eval repos are popular OSS that Gemma likely saw in pretraining. In RW, the displayed edges conflict with memory. Base trusts memory, giving small Δ_struct. "Trust displayed over memory" again looks like structure use.
3. **Is the test body rewired?** It is not specified. If it is not, depth-1 positives stay lexically visible and Δ_struct probes only depth ≥2. If it is, depth-1 dominates Δ_struct. The H4 prediction for (d) flips depending on this choice.
4. **Mechanical coupling to H1.** If AP(RW) floors at a name-prior level P0 for every model, then Δ_struct = AP(real) − P0, and any H1 gain appears as an H4 gain. H4 is then not independent evidence.

**Fixes.**
- Use the plausible-ρ rewire on real items too: only swap callees with *compatible arity* and in *files imported by the caller*.
- Run a "corruption detection" probe (ask whether the code was tampered with) on real vs RW; it should be near chance.
- Specify and *stratify* by test-body rewired yes/no.
- Report a **no-LLM name-reachability oracle** (BFS over displayed bodies from the test) in real, RW and GF. Normalise: `Δ̂_struct = (AP_real − AP_RW) / (AP^oracle_real − AP^oracle_RW)`.
- Report Δ_struct conditional on H1-matched items, or test H4 on the residual after regressing Δ_struct on AP(real).
- Measure memorisation with Δ_GF. High AP(GF) means memory, so prefer less popular S_eval repos.

### K-35 Distractors can be statically reachable from the test, so the item is not a graph item — **major** (C9 construct validity)
"Matched on file and degree" means many distractors are same-file siblings reachable through branch-guarded calls (K-09). On those, perfect static traversal gives no separation. On statically unreachable distractors, a BFS script gives AP≈1. Base AP≈0.55 sits in an unknown mixture of the two.

**Fix.** Label each distractor as static-reachable or not. Report AP separately: the static-unreachable stratum measures traversal; the reachable stratum measures dynamic and branch reasoning. Report the BFS oracle on both.

### K-36 The 5-level p and matched precision — **minor** (A13)
With 5 levels, "recall at matched micro-precision" has only 4 achievable thresholds, so matching requires interpolation, which is a free parameter. A trained model collapsing to {0.05, 0.95} changes tie structure. That is harmless for expected AP under random tie-breaking, but it inflates variance. **Fix:** use the teacher-forced level distribution (K-41) as a continuous score, and keep the verbalised p as the deployed readout only.

### K-37 H6 is gameable by padding EDIT — **major**
"Both sites listed EDIT at p≥0.5" is satisfied by listing every examined node as EDIT p=0.5. **Fix:** H6 = both sites at p≥0.5 **and** ≤ k non-gold nodes at p≥0.5 (or use AP over the K-30 fixed set). Report precision.

---

## 6. Offline RL stage (C8)

### K-40 Recomputing π_A in the trainer removes the *visible* mismatch, not the bias — **major** (A12)
Samples come from π_A^vLLM (W4A16 kernels plus punica LoRA, bf16 accumulation). The surrogate treats them as samples from π_A^train (HQQ g32). "r = 1 exactly at epoch 1" hides that the importance weight `π_train/π_vLLM` is ≠1, and that weight is ignored.

The A12 gate uses the **median** |Δ| ≤ 0.05 nats per token. The harm comes from the **tail**: low-probability tokens sampled at T=1.0, where kernel differences are largest. Summed over a 12k-token trajectory, a 0.5% tail at |Δ| ≥ 1 nat dominates. This is the known train/infer mismatch failure (truncated-IS literature, INFERRED).

"Exact served grid" is also doubtful:
- If W4A16 is GPTQ with group 128, act-order (g_idx) and asymmetric zero-points, a fresh HQQ g32 fit to the dequantised weights chooses its own step per group of 32. It reproduces the g128 grid exactly only when each group-of-32 spans the full 16 levels.
- The ≥97% next-token agreement threshold permits a per-token KL that is not small at T=1.

**Fix.**
- Keep the vLLM log-probs and apply **truncated importance weights** `w_t = min(π_train/π_vLLM, C)`, with C≈2, per token. Mask tokens with |log w_t| > 1.
- Gate A12 on **p99** and on the per-sequence sum, not the median.
- Make the trainer's frozen base an *exact* dequantisation of the served GPTQ tensors (custom dequant, max-abs weight difference = 0), not an HQQ refit. Verify it by tensor diff, not by next-token agreement.

### K-41 Spurious transfer from format compliance — **major** (H1)
On real items, the base model in format A may skip lines, reorder, merge, truncate at the token budget, or emit `p=0.6`. The scoring of missing or invalid lines is unspecified (p=0? a tie? drop?). The trained arms all see the identical format at ≤30% of their tokens, so they emit complete, well-formed blocks. H1 then rises from **parse completeness**, and (b), (c) and (d) rise alike, which also feeds K-25.

**Fix.**
- Use a **teacher-forced readout** for every model. Prefill `"<qualname>: CALLED p="` per candidate and read the log-probs over the 5 level tokens. The score is the expectation. Format compliance is then impossible to differ on.
- Report the parse-failure rate for the free-generation A readout separately.

### K-42 KL estimator and length weighting — **minor**
- k3 is an unbiased estimator of KL(π_θ‖π_A) only under samples from π_θ. Here the samples come from π_A (off-policy, and from 2 epochs).
- The 1/|T| global token mean weights long failing trajectories more than short successes.

Neither is fatal at β=0.02. **Fix:** report the KL with the importance-weighted form, and use a per-group token normaliser (Dr.GRPO uses a constant).

### K-43 Zero-variance groups turn Stage B into label SFT — **minor/major**
At depth ≥4 most K=4 groups are all-fail. "Zero-variance groups still carry the η-term", so the gradient there is purely the per-line calibration term toward compiler labels. Stage B is then largely supervised calibration on synthetic labels, which is a different claim from "RL on traversal". **Fix:** report the gradient-norm share from η vs Â; ablate η=0 in one arm if the budget allows.

### K-44 DPO fallback pitfalls — **major**
1. **Per-arm triggering breaks pairing.** G5 is evaluated per arm. If (c) triggers DPO and (b) does not, the arms differ in objective. **Fix:** the fallback triggers for *all* arms if any arm trips it, pre-registered.
2. **DPO drops the set term.** The pairs carry no item log-score objective, so the readout H1–H4 measure is no longer optimised.
3. **Long multi-turn pairs share prefixes up to divergence.** The gradient concentrates on the few tokens after divergence. "Length-normalised" DPO is not DPO, since the reference-ratio form changes. It is a SimPO-like hybrid, and the β=0.1 calibration does not carry over.
4. **Likelihood displacement.** Chosen-trajectory log-probs typically fall too, so the success rate can drop while the margin grows. **Fix:** add an SFT term on τ⁺ (RPO-style) and monitor the absolute log π(τ⁺).
5. **Few pairs.** About 20 tasks × P(mixed group) gives perhaps 5–15 pairs per arm, too few to estimate anything. **Fix:** pool over item samples, which do have pairs per element.

### K-45 Debugging (b) before the other arms is researcher freedom — **minor**
G4 says "debug before training the other arms" after seeing (b). Any hyperparameter fix tuned on (b) favours (b). **Fix:** freeze the hyperparameters on a *fourth*, throwaway synthetic family, or apply any change to all arms and log it.

### K-46 Equalisation by repo size and by success count — **major** (paired design)
- "If an arm falls below the floor, recalibrate *that arm's* repo size" means the arms differ in |V| and file count, i.e. not only in topology.
- Subsampling successes to the minimum count does not match *which* tasks were solved. A hard arm's successes come from its easiest tasks, so per-depth composition differs across arms.

**Fix.** Recalibrate size **jointly** for all arms (or raise K for all arms). Subsample successes **stratified by d_obs bucket** so that every arm has the same per-bucket counts.

---

## 7. Transfer (A5, A7)

### K-47 Content-free training teaches "ignore names", which should transfer negatively — **major** (A5)
In compiled repos, identifiers carry zero information (`parse_header` computes a modular fold), so the optimal policy learns to down-weight names. On real repos names are the single strongest cue: the NameRTS result is 99.6% safe selection by names alone (VERIFIED by the red team).

Example: a test calls `client.get(url)`. The candidates include `Client._send_handling_redirects` and `Client._redirect_url`. Base uses names; a trained model that discounts names ranks them lower unless the displayed edges are complete. They are not: `_send_handling_redirects` is reached via `self.send` → `_send_handling_auth`, which may be truncated beyond line 25.

**Prediction:** H1 negative or ~0 on the lexical stratum, positive only on the deep-and-dark stratum.

**Fix.**
- Add a *name-preserving* condition: identifiers sampled with weak semantic correlation to the template family (e.g. `sum_*` for folds) at a controlled mutual information. That is a single v1.1 change, as A7 already anticipates.
- At minimum, pre-register H1 per stratum (lexical vs dark, depth), and make the lexical-stratum non-inferiority explicit.

### K-48 Spurious H1 gains without any graph skill — summary
| Spurious source | Mechanism | Guard |
|---|---|---|
| Format and parse compliance | K-41 | Teacher-forced readout |
| Base-rate learning | Training and eval items both have 33% positives; the trained model learns "≈1/3 CALLED". AP is rank-based, but the log score and matched-precision recall improve | Report AP only for the headline; vary the eval base rate (25% / 33% / 50%) on a subset |
| Length and truncation | Synthetic bodies are ≤25 lines, so nothing is truncated; real ones are truncated. The trained model learns "absence of call in body ⇒ NOT", which is wrong on truncated real bodies | Mark truncation (`# … 40 more lines`); stratify by "positive's call site truncated" |
| Hub prior | In synthetic graphs, hubs are reached by most tests | Distractors are degree-matched (good). Also report the IDF-filtered AP, already planned |
| Shared item operator | The RW eval context uses the same rewire operator that generated arm (c)'s training graphs | Report (c)'s AP(RW) separately; do not pool |

### K-49 Eval items are hub- and integration-test dominated — **minor**
G3 requires ≥5 positives per item on ≥90% of items. That drops small unit tests and keeps integration tests, whose E(t) is dominated by `__init__`, properties and helpers (the critic's hub row). The primary AP then rewards hub recognition. **Fix:** make the primary metric the IDF-filtered AP, or cap hub positives (≤2 per item) in the sampler.

---

## 8. Statistics and inference

### K-50 Pseudo-replication: one training run per arm — **major** (H2, H3; A10, A15)
The treatment (an arm) is applied once. The CI is bootstrapped over 300 items in 8 repos, so it captures *evaluation* noise, not *training* noise. LoRA runs with ~1.5M tokens and 15–40 successful trajectories have seed spreads plausibly ≥0.02–0.04 AP (INFERRED).
- **H3 (TOST, ±0.03)** can "pass" because the item-level CI is narrow while the true arm effect is unidentifiable. D2 drops even the single replicate.
- **H2** can pass or fail on one lucky seed.
- The cluster bootstrap with **8 clusters** is anti-conservative (well known for few clusters, INFERRED), which narrows the CI further.

**Fix.**
- Run ≥3 seeds per arm, paying for it by cutting tasks (the item evaluation is cheap) and by using shared rollouts with different shuffle and init, as in B11 but for all arms.
- Use a hierarchical bootstrap over (seed, repo, item).
- Use a wild-cluster or randomisation test for 8 clusters, or ≥12 repos.
- If only one seed is affordable, H3 must be **reported as "not distinguishable at n_seed=1"** and never as equivalence.

### K-51 The H6 synthetic family is one repo — **minor**
100 items from one held-out topology family form one cluster. McNemar assumes independence. **Fix:** use ≥3 held-out families, or report it descriptively only.

---

## 9. Assumption-to-break map

| Assumption | Breaks |
|---|---|
| A4 (band with variance) | K-08, K-43 |
| A5 (content-free transfer) | K-12, K-47, K-48 |
| A6 (AP proxy) | K-35, K-09 |
| A7 (protocol unchanged) | K-04, K-07, K-47 |
| A8 (rewire changes only wiring) | K-11, K-20, K-21, K-22, K-23, K-24 |
| A9 (trace completeness) | K-09 |
| A10 / A15 (noise, clusters) | K-50, K-51 |
| A12 (kernel mismatch small) | K-40 |
| A13 / A14 (levels, yes-bias) | K-32, K-36, K-41 |
| A16 (SCC, DAG) | K-01, K-02, K-03 |
| A19 (DPCS solvable by reasoning) | K-04, K-11, K-13, K-14, K-15, K-17 |
| A20 (dark share, hubs) | K-06, K-07, K-09 |
| Unlisted (the reward is proper and ungameable) | K-18, K-30, K-31, K-33, K-37 |
| Unlisted (the arms differ only in topology) | K-05, K-26, K-27, K-46 |

---

## 10. The five most damaging breaks

1. **K-30 + K-31: the set reward rewards padding and tolerates omission (fatal).** The mean over listed ∪ gold, the tri-state exemption and η on name tokens together make padding with easy negatives dominate honest listing. Even dropping all gold beats honesty in the worked example. This directly undermines the CANDIDATES mechanism and H6.
2. **K-04 + K-12: topology-invariant solution paths (fatal for H2/H3 logic).** Visible exact contracts give "deepest failing contract" localisation. Fixed templates give surface-anomaly localisation for O1 (50% of tasks). Every arm learns the same non-graph skill, so (b)≈(c) and (b)≈(d) hold for the wrong reason, and K1 can still pass on budget (K-08).
3. **K-21 (with K-11, K-20, K-24): the rewire changes locality, imports, coverage and class structure (major, near-fatal for H3 interpretation).** (b)>(c) would be read as "real topology is transferable graph structure" when it may be a file-locality and import prior. DPCS itself leaves an import fingerprint.
4. **K-27 + K-28 (with K-26): arm (d) is a localisation-given, different-content control that likely harms deep strata.** H2 can pass with (b) = base.
5. **K-50: pseudo-replicated inference with 8 clusters.** One seed per arm plus an anti-conservative cluster bootstrap means TOST "equivalence" (H3) and H2 can be produced by training noise. Δ_struct is mechanically coupled to H1 (K-34).

Close runners-up:
- K-18: hardcoding the issue value earns partial F2P.
- K-01: union-of-tests depth is ≈1.
- K-41: format-compliance transfer.
- K-40: tail mismatch ignored.

## 11. Minimum fix set for v2

| # | Fix | Kills |
|---|---|---|
| F1 | **Reward.** Score a fixed set (gold ∪ compiler-sampled matched negatives ∪ listed). Normalise the sum by |gold|. Apply η only to p tokens, with omitted gold counted at p=0.05 inside the group baseline. Use an S-threshold (not "all-gold-listed") in the RFT filter. Add a reward unit test showing honest > every padding and omission policy | K-30, K-31, K-33, K-37 |
| F2 | **Hidden F2P on fresh random inputs** generated from gold. Binary success for filtering. Reject patches containing literals from the issue | K-18 |
| F3 | **Difficulty = d_obs** (hops from the nearest failing *visible* test to the bug). Place bugs after sampling the contract set. Report the no-LLM contract/spectrum solver rate per arm; it must be well below the agent's rate before arms are trained | K-04, K-08, K-20 |
| F4 | **In-distribution mutations.** Per-instance randomised template constants and operators; a bug is another valid instance of the same family. The body-only discriminator AUC must be ≈0.5 | K-12, K-19 |
| F5 | **DPCS constraints.** a ≠ c, graph distance ≥ 2. Same module, or modules already imported, or namespace calls. Positional args. Total templates. Lexical and callback edges only. Set gold = any minimal passing edit set | K-11, K-13, K-14, K-15, K-16, K-17 |
| F6 | **Rewire (c) = level- and block-preserving swap.** depth(b) = depth(d), depth(a) = depth(c), and preserved (module, module) counts. Dark edges are not rewired. Report modularity, motifs, #roots per node, \|desc\|, class count and the C5 drop rate per arm | K-21, K-22, K-23, K-24, K-05 |
| F7 | **Replace (d) with (d′):** the same graphs as (b) with bug-named issues. Pre-register H2 as (b) > base **and** (b) > (d′). Report (d′) − base by stratum | K-26, K-27, K-28 |
| F8 | **Teacher-forced level-logprob readout** for A on all models. Name-reachability oracle in all contexts. Normalised Δ_struct. Plausible-ρ on real items. Stratify by test-body rewired and by static reachability of distractors | K-34, K-35, K-41, K-36 |
| F9 | **≥3 seeds per arm**, hierarchical bootstrap, and a few-cluster-valid test (or ≥12 repos). H3 is reported as equivalence only if the seed-inclusive CI fits the margin | K-50 |
| F10 | **Truncated IS weights from the vLLM log-probs.** A12 gated on p99. Exact dequant base, not an HQQ refit. The DPO fallback triggers for all arms jointly | K-40, K-44 |
| F11 | **CPU-only source-graph hygiene before B2.** Integration-only roots. SCC-node-fraction cap or bounded recursion. Effect-only edges at the real share. Hub bug cap. Report everything before spending GPU | K-01, K-02, K-06 |

F1–F3 and F11 are CPU or code changes with no GPU cost. F9 is the only fix that materially costs GPU. Paying for it by cutting tasks per arm is preferable to keeping an H3 that cannot be interpreted.
