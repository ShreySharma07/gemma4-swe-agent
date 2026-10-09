# Breaker 1 (logic and edge cases): Behavioural Map-Matching v1, 9 Oct 2026

**Target.** `architecture_v1.md` (BMM), including A1–A20. I also skimmed `phase3_critic.md` and `phase3_redteam.md`, and I do not repeat breaks they already raised unless v1's fix fails to close them.

**Labels.**
- **WORKED**: a small constructed example with arithmetic. The repos are real, but the snippets are illustrative, not quoted from a task.
- **INFERRED**: my reasoning.
- Nothing was run.

**Severity.**
- **Fatal**: a headline claim can pass or fail for a reason that is not the claimed mechanism, and v1 has no arm that would show it.
- **Major**: it biases a primary contrast or a stratum by an amount comparable to the predicted effect (MDE ≈0.02–0.045).
- **Minor**: it adds noise, cost or a reporting gap.

Breaks are numbered L1–L46 and grouped by the seven areas in the brief. The five most damaging breaks, with the minimum v2 fixes, are in §8.

---

## 1. Imagined paths

### L1. Granularity mismatch: the state space holds only functions, but guesses and fix sites are not always functions (Major; A8, and gold mapping §2.2.4)

- **Coarse guesses.** Take the guess `STEP 2: requests.utils | none`. s_name uses subtoken Jaccard against `qualname_v`.
  - Jaccard({requests, utils}, {requests, utils, quote}) = 2/3.
  - Jaccard({requests, utils}, {requests, utils, get, encoding, from, headers}) = 2/6.
  - So a module-level or class-level guess sends emission mass to the function in that scope with the **shortest name**. This is a systematic length bias, not a neutral spread over the scope.
  - A class-level guess `HTTPAdapter` gives an equal Jaccard to all ~12 methods, minus the length bias.
- **Fix sites that are not functions.** Requests' `Session.__attrs__` (the pickling list) is a class attribute. Many fixes in sympy, astropy and django edit class-body tables, module-level regexes, `__all__`, `Meta` options or dataclass field defaults.
  - §2.2.4 maps hunks "to existing functions" and has no rule for a hunk in class-body or module top-level code. Such a task is in neither the on-path, off-path nor new-code stratum. It is scored as an unreachable gold or silently dropped.
- **Fix.**
  - Add *scope states* (class body, module top level) as HMM states that emit with a hierarchical name score. Each scope state has transitions from the functions that read the scope's attributes (HEAP attribute-read edges).
  - Replace Jaccard with a containment score normalised by guess length (|n ∩ q| / |n|), with a small length prior.
  - Add a **non-function-hunk stratum** and report its size in week 1.

### L2. Plausible hallucinations are adversarial, not random (Major; A8, and the P1 noise model)

- **WORKED.**
  - Issue: "Session ignores `no_proxy` from the environment when a proxies dict is passed."
  - Guess: `requests.sessions.Session.merge_proxies` (this function does not exist).
  - The near-misses are `Session.rebuild_proxies`, `Session.merge_environment_settings`, `utils.get_environ_proxies` and `utils.should_bypass_proxies`. Each shares one or two subtokens.
  - The best-scoring wrong node shares the most tokens, and in a well-factored repo it is a *sibling* of the culprit.
- **Two consequences.**
  - **Name error and graph error are correlated.** Siblings are adjacent in G, so the transition model cannot separate them either. Sibling confusion is the error mode BMM is supposed to fix, and it is the one the graph is worst at fixing.
  - **The positive control's noise is too easy** (see L35). P1 noises 50% of names, presumably by random substitution. Random substitutes are far in G, so the transitions recover the path easily.
- **Fix.**
  - Measure the hallucination type on CAL-tune: non-existent, sibling, wrong module, or exact.
  - Make the P1 noise model **sample from that measured distribution** (sibling substitution), not uniformly.

### L3. Framework callbacks break the chain: no transition leaves ∅ (Major; A11, A13)

- **WORKED (fastapi-like, or pytest/pluggy, sphinx `app.connect`, django signals, click).**
  - Imagined path: `routing.APIRoute.get_route_handler → (starlette dispatch) → dependencies.utils.solve_dependencies → …_validate_value`.
  - The repo calls starlette, and starlette calls back into a repo closure. G has `get_route_handler → ∅` (external), but **no ∅ → repo edge**, because the callback is invoked from third-party code.
  - The only way back into the repo is the teleport ρ·π(v), and π is BM25. After any external hop, BMM is "BM25 prior × name emission", which is X-BAG with a prior.
- **Who this hits.**
  - Plugin and hook architectures: pytest is in SWE-bench Verified, and so is sphinx with its event system.
  - Django: its middleware and ORM are internal, but `signals.send`, template-tag registration and URL resolver callbacks route through registries.
- **Fix.**
  - Add ∅ → v transitions for functions that HEAP sees **passed as a value** into an external call or registered with a decorator. These are callback-registered nodes, weighted by registration count.
  - Report the share of true-path edges that pass through ∅.

### L4. One-step and two-step paths: decoding is name lookup, and C3 is a no-op (Major; A8, A13)

- For T = 1: γ_1(v) ∝ π(v)·b_v(o_1). That is exactly B-NAMES with a BM25 prior (plus the effect term). It has no structure.
- For T = 2: X-FL ("keep first and last") is **identical** to BMM, and X-SHUF changes nothing a symmetric score can see.
- The prompt asks for 2–8 steps. Typical issues ("`DataFrame.to_csv` drops the index name") yield an entry plus a culprit, with padding in between.
- If 40% of paths have T ≤ 2, the C3 contrasts are diluted by 40% of mechanical zeros, and the "SHUF ≈ BMM and DPR ≈ BMM → bag" kill fires on a population where the ablation cannot act.
- **Fix.**
  - Pre-register C3 on the stratum **T ≥ 3 in the majority of the k samples**, and report what fraction of items are ablation no-ops.
  - Report C1 by path-length stratum.

### L5. Fix sites that no execution path reaches (Major; A6)

v1 adds new-code and off-path strata. Three gaps remain.

- **(a) Missing branch.** The gold is the dispatcher that lacks `elif isinstance(x, Decimal)`. It is on-path, but its effect under the issue input is "falls through and returns None" or "raises TypeError from a generic fallback".
  - The **observable effect belongs to the fallback**, which is *downstream*, not to the gold.
  - The imagined path ends at the fallback, and decoding ranks it first. Gold is one step *upstream* of where the evidence points.
  - **Fix:** a secondary readout that also scores the *predecessors* of the posterior mode (a "culprit is the caller of the symptom" rule), reported separately.
- **(b) New registration.** The fix adds a new `@register(Decimal)` handler. It is "new-code" and scored at file level, while every other item is scored at function level.
  - **Fix:** the headline must exclude the new-code stratum. Do not pool its file-level RR into MRR.
- **(c) Ambiguous off-path rule.** "If a hunk adds a call into a function that is not on any executed path" is ambiguous. Is the gold off-path, or only the callee?
  - **Fix:** define off-path as *no gold-mapped function is entered in the base F2P run*.

### L6. Reversed narration: issues describe symptom → cause (Minor-major; A13)

- Bug reports often read "X crashes; this is because Y calls Z with…". The model may emit steps in symptom-first order, which is callee → caller.
- v1 logs a reversed-order variant but does not decode it.
- On a directed graph, a reversed path has near-zero transition likelihood, so mass leaks into δ and ρ. That makes the path a bag, and SHUF ≈ BMM.
- **Fix.** Decode both orientations and take the per-sample max likelihood. This is one change to the decoder, applied to every arm. Report the reversed fraction.

---

## 2. HMM decoding

### L7. Stutter degeneracy: δ beats A₁ for any caller with many call sites (Major; A13, §3.1)

A₁(u→v) is the mean over u's call sites of p(v | s), so each callee of a 25-site function gets at most 1/25.

**WORKED.**
- Settings: 1 − ε − ρ − δ = 0.7; δ = 0.1 (a plausible TPE value).
- Entry E = `Session.request`, with 25 sites. A(E→M) = 0.7 × 0.04 = 0.028.
- Hallucinated step o₂ = "Session.merge_proxies": b_E(o₂) = 0.010 (shares "session"); b_M(o₂) = 0.020 for M = `merge_environment_settings`.
- Move: 0.028 × 0.020 = 5.6e-4. Stay: 0.1 × 0.010 = 1.0e-3. **The path stays on E.**
- Moving needs b_M/b_E > δ/A ≈ 3.6. A hallucinated middle step almost never gives that ratio.

**Consequence.** Exactly in the regime BMM exists for (wrong names), the posterior collapses onto the well-named entry. "Structure corrects names" (M1) is predicted *false* by the model's own algebra whenever callers have many sites.

**Fix.**
- Replace the stutter with **null-emission skip states** (profile-HMM style): an observation may be "not a node" with probability η₀, rather than "the same node again".
- Normalise transitions per call site, not per function: use site-level states or p(v | u) = Σ_s p(s | u)·p(v | s), with p(s | u) from execution counts.
- Report the tuned δ against the median A₁ along true paths.

### L8. Missing edges are reachable only through the BM25 teleport (Major; A11, A13)

- When the true edge u→v is absent from G (unresolved dynamic dispatch, a `__getattr__` proxy, a lazy import inside a function), the only routes to v are ε·A₁² (a 2-hop route that does not exist either) and ρ·π(v).
- π is softmax(BM25/T). **A culprit with low BM25 is reachable after a missing edge only with probability ≈ ρ·π(v) ≈ 0.**
- So the structural correction works only where BM25 already works, which is the same population as the B-BM25 baseline. C1′ against BM25 is capped by construction.
- **Fix.**
  - Teleport to a *mixture* of π and a uniform or embedding prior, with weights tuned.
  - Report **true-path edge recall of G**: the share of traced F2P path edges in G, per stratum. This is CPU-only, week 1.

### L9. Hubs absorb forward mass, and ε·A₁² through hubs turns the skip into a teleport (Major; §3.1)

- **WORKED (sympy).** `sympify`, `Basic.__new__`, `_sympify` and `as_coeff_Mul` have in-degrees in the thousands. π·A^t accumulates on high in-degree nodes.
- With weak names, max_t γ_t(sympify) is large for every issue, and sympify ranks in the top 10 for most sympy tasks. Sympy is about 15% of SWE-bench Verified.
- The ε·A₁² skip goes *through* hubs. Every caller of `sympify` (thousands) gets 2-hop edges to every callee of `sympify`. nnz(A₁²) ≈ Σ_v indeg(v)·outdeg(v), which explodes, and the skip becomes a quasi-uniform jump. This is a semantic blur, not only a compute cost.
- Other hubs: django `cached_property.__get__`, `check_array` in sklearn, matplotlib `_api.check_in_list`.
- **Fix.**
  - Exclude intermediates above an in-degree percentile from A₁². Better, take the skip only over HEAP `wraps` / decorator edges, which are what it was introduced for (G-A4).
  - Divide the posterior by the stationary hub mass, scoring γ_t(v)/π_∞(v)^λ with λ tuned. B-PPR needs the same normalisation, to stay one-change.
  - Report hub share in the top 10.

### L10. Emission is a posterior, not a likelihood: hybrid-HMM double counting and sibling dilution (Major; §3.4)

- s_name(n, v) is a **softmax over v**, so it is P(v | n), not P(n | v). Multiplying it by π(v) and the transitions counts the "popularity" of v twice when names are informative, and it has no clear meaning when they are not.
- **Sibling dilution.** If the guess is `_print_Pow` and the printer family has ~300 `_print_*` methods across classes, one softmax shares the mass. A unique-named node gets full mass for an equally good match. So culprits in large method families (sympy printers, django `Field.get_prep_value` overrides, sklearn `fit`) are penalised relative to unique helpers.
- **Fix.**
  - Use a likelihood-style score: exp(τ·sim(n, v)) / Z_n, *without* normalising over v. Equivalently, divide the softmax by a prior over v (the hybrid-HMM correction).
  - Tune τ under that form.
  - Report the family-size stratum.

### L11. max_t aggregation is order-blind and favours entry points (Fatal for the C3 reading; Major for MRR)

- score(v) = mean over paths of max_t γ_t(v).
- Entry steps are the best-determined: issue-named, high π, high name match. Their γ is often near 1, while the culprit's last-step γ is spread.
- So when entry ≠ gold, the **entry ranks first and gold ranks second at best**. RR is ≤ 0.5 for structurally correct decodes.
- max_t also discards *which* step the mass came from. Shuffling the order (X-SHUF) changes γ only through transition smoothing. With name-dominated emissions, max_t γ_t is close to the per-step emission ranking, whatever the order.
  - X-SHUF ≈ BMM is then an artefact of the readout, not evidence that sequence is uninformative.
- The kill rule "SHUF ≈ BMM and DPR ≈ BMM" can therefore fire *because of the aggregator*.
- **Fix.**
  - Pre-register a **position-weighted readout**: weight w_t rises towards the culprit end, with the shape tuned on CAL-tune, or use the final-step posterior.
  - Run C3 on both readouts.
  - Report "rank of gold excluding the decoded entry node" as a diagnostic.

### L12. When BMM provably reduces to PPR-like or bag scoring (Fatal for interpretation if undiagnosed)

Three exact reductions.

1. **Uninformative observations.** All s_name are equal (hallucinated tokens with zero overlap, and the cosine is flat after τ) and e_t = none. Then b_v(o_t) is constant, the backward messages are 1, and γ_t = π·A^t. So score(v) = max_t (π A^t)(v).
   - This is a **truncated random walk with restart from π**, which is a PPR variant with a max over t in place of a geometric sum.
   - BMM differs from B-PPR only by the aggregator, and C1 measures max vs sum.
2. **Teleport-dominated tuning.** If TPE sets ρ near 1 because CAL-tune transitions are noisy (or leaked, see L28), A ≈ 1·πᵀ and γ_t ∝ π·b(o_t). That is X-BAG times a prior.
   - C1 can still pass, because BMM uses the LLM's names and B-PPR does not (L33). C3 then fires its "bag" kill.
   - v1's decision rules then disagree: C1 says "technique survives" while C3 says "it is a bag".
3. **T = 1.** See L4.

**Fix.**
- Report the tuned (ρ, δ, ε) and an **effective-structure diagnostic**: the KL divergence between BMM's posterior and X-BAG⊗π per item.
- Pre-register: if the median KL is below a threshold, C1 is reported as "bag + prior", whatever its sign.

### L13. Self-loops and recursion interact with δ (Minor)

- Recursive functions (sympy `_eval_*`, `simplify`; AST `visit` → `generic_visit` → `visit`) have a real A₁(u→u) edge plus δ.
- Visitor cycles with HEAP dispatch edges (`visit → visit_X`, |C(s)| ≈ 200) form high-mass strongly connected clusters, and both effects in L7 and L9 are amplified there.
- **Fix:** cap total self-transition at max(δ, A₁(u→u)) instead of summing them, and treat dispatch fan-out as one site (L7 fix).

### L14. Effect term rewards "omnivore" functions (Major; §3.4)

- log b ∋ β·Σ_{j∈e_t} log((1−η)q̃_{v,j} + η). Unstated effects carry no penalty, and there is no base-rate normalisation.
- So the score rises monotonically in every stated q. A large function that does everything (big `__init__`, `main`, `process_request`, a 300-line `fit`) with many raise sites, I/O and self-writes matches every stated effect set at least as well as a small culprit.
- **WORKED.** The issue states {ValueError, writes self attr}. Small culprit `_parse_offset`: q = (0.9, 0.05). `Parser.__init__`: q = (0.6, 0.95). With η = 0.05:
  - culprit: log 0.905 + log 0.0975 = −2.43;
  - `__init__`: log 0.62 + log 0.95 = −0.53.
  - The `__init__` wins by 1.9 nats per step × β.
- **Fix.**
  - Use a likelihood ratio: log(q̃_{v,j}/q̄_j), with q̄_j the repo base rate, which is an IDF for effects.
  - Add a weak penalty for strongly predicted effects that the issue does not state, tuned.
  - Report the size correlation of the effect term (Spearman of effect score vs LOC).

---

## 3. Effect signatures

### L15. The 10%-frequency label erases exceptional behaviour, the behaviour issues report (Fatal for C2b, Major for C2a; A7, A9)

- Labels are "observed in ≥10% of a function's traced calls". Exception bits are inherently rare-input.
- **WORKED.** `validate_offset(s)` is called 206 times in the suite: 200 with valid input and 6 with invalid input that raise ValueError. 6/206 = 2.9%, so **the label is 0**.
- g_φ therefore learns that a `raise ValueError` under an `if` is *not* a ValueError-raiser unless the suite hammers the error path. Execution calibration **lowers** q exactly on validators and guards. X-ASTEFF (rule fires → q = 1) keeps them.
- C2b (execution calibration adds) is then predicted **negative for a structural reason**, not because execution knowledge is useless.
- **Fix.**
  - Label "ever observed in ≥2 distinct tests" (possibility), with a separate frequency bit if wanted.
  - Or condition labels on the call-context class (error-path vs normal-path inputs).
  - Re-run the A9 gate on that label.

### L16. Issues report buggy, anti-normal effects, and the alphabet has no polarity (Fatal for C2a on its natural population)

There are three failure shapes, and the alphabet expresses none of them.

- **(a) "Should raise but doesn't."** "`parse('2021-02-30')` returns a date instead of raising." The culprit's *current* behaviour lacks ValueError, which is the bug. The generator, given the effect alphabet, writes `ValueError` (the word is in the issue). The emission then **promotes functions that do raise ValueError**, the culprit's well-behaved siblings, and demotes the culprit.
- **(b) "Raises X but shouldn't."** The culprit *does* raise, so the emission is aligned. But its traced label under the passing suite is "rarely raises" (L15), so q is low.
- **(c) "Returns the wrong value".** This is the modal SWE-bench bug, and the 16 bits do not represent it.

So the behaviour channel is aligned only with (b) and needs possibility labels to work even there.

**Fix.**
- Make the generator emit `+effect` (observed) and `−effect` (expected but missing) separately.
- Score −effect with a *contract* model: siblings or overridden base methods raise, but v does not.
- Stratify C2a by polarity (a)/(b)/(c), and pre-register C2a on (b) plus (a) with the contract score.

### L17. Behaviour is lexical in disguise, and R2 renaming makes C2a near-tautological (Fatal for the "name-free" claim)

- **Issue side.** The generator writes `KeyError` when the issue says "KeyError".
- **Code side.** g_φ's strongest feature is a `raise KeyError` site, or a subscript on a parameter. The effect channel is close to **grep for builtin exception tokens in bodies, matched against exception tokens in prose**.
- **Under R2:**
  - repo identifiers are renamed; builtin exception names, string literals (`raise ValueError("Invalid timezone offset")`), dunders and docstring English survive;
  - X-NOEFF's name channel sees only qualname + signature + docline (renamed);
  - BMM-full gains a body-derived channel that renaming did not touch.
- So "behaviour adds more under renaming" = "an unrenamed second channel helps when the first is renamed". That is true of *any* body-token channel, behavioural or not.
- **Fix.** Add two one-change controls and make C2a beat them:
  - **X-LEXEFF:** the effect term replaced by a binary "body contains a `raise <C>` / `except <C>` token for a class C named in the issue" (no learning, no execution).
  - **X-NOEFF+BODY:** the name emission augmented with BM25 over body tokens (renamed in R2 like everything else).
  - C2a holds only if BMM − X-LEXEFF > 0 and BMM − X-NOEFF+BODY > 0 under R2.

### L18. R1/R2 renaming leaves a large, uneven residual name channel (Major; C4, §2.3)

What survives scope-aware renaming:
- dunders (`__getattr__`, `__eq__`);
- overrides of external APIs (`logging.Handler.emit`, `Thread.run`, `json.JSONEncoder.default`, django `Field.to_python` when django is not the repo);
- **string-built dispatch** (`getattr(self, "visit_" + type(node).__name__)`, `"_print_" + cls.__name__`, sympy `"_eval_is_" + fact`), which *breaks* if renamed and is therefore probably left alone;
- kwargs names passed through `**kwargs`;
- `setattr` strings, django lookup names (`"iexact"`), string literals, and docstring and comment English.

**R1 issue-side renaming is ill-defined.** LibCST cannot parse prose. Token replacement in the issue renames English words that collide with identifiers ("request", "session", "headers", "index"). That destroys prose semantics, or, if skipped, leaves partial matches.

**Fix.**
- Rename only in code spans (backticks, code blocks) for the issue in R1, and report the collision rate.
- Publish the residual-name share per repo: the share of node-text tokens left unrenamed.
- Run C4 on repos with residual share <X.

### L19. A coarse alphabet gives huge ties for the modal effects; a fine one gives hierarchy ambiguity (Major)

- **Coarse.** "returns None" and "writes self attr" are true of thousands of functions (every `__init__`, setter or visitor). When an issue states only those, the effect term splits nodes into two giant classes. If ~40% of functions write self attributes, the bit carries ≈1.3 bits of information at best.
- **Fine.** dateutil `ParserError(ValueError)` and requests `InvalidURL(RequestException, ValueError)`. The issue says "ValueError" while the code raises a repo-defined subclass. The two bits ("ValueError", "repo-defined") never meet. `except LookupError` covers KeyError and IndexError, but the c_{v,j} rule needs class j literally.
- **Fix.**
  - Score raises by the exception **MRO**: a repo exception that subclasses ValueError fires both bits.
  - The except-coverage rule uses issubclass.
  - Add an effect-informativeness stratum (the issue states at least one bit with repo base rate <10%).

### L20. Cross-repo transfer: CAL utilities do not share EVAL's idioms (Major; A9)

- The CAL list (attrs, toolz, boltons, more-itertools, …) holds small, pure utilities. EVAL repos express effects through **in-repo wrappers**:
  - matplotlib `_api.warn_external`, `_api.warn_deprecated`;
  - sklearn `warnings.warn(..., ConvergenceWarning)` behind `_check_*` helpers;
  - astropy `AstropyWarning`;
  - django `logger = logging.getLogger("django.request")` and `self.log(...)`, plus I/O through the cursor or storage APIs, not `open`.
- **Warnings are not propagated** (§3.2: exceptions only). For "matplotlib now emits a deprecation warning when…", the only node with a high warn bit is `_api.warn_deprecated`, a hub. The issue's effect points at the hub, not the culprit. That is L9 and L14 combined.
- **Fix.**
  - Propagate warn and I/O bits one hop through wrapper functions that HEAP or AST identifies (a function whose body is essentially a `warnings.warn` / `logging` call).
  - Add a per-repo "effect wrapper" pass.
  - Report g_φ AUROC on **EVAL** pre-existing-test traces per repo, not only on CAL hold-out.

### L21. Training-label selection: only functions with ≥5 traced calls (Major; A9)

- Bugs live disproportionately in under-tested code. g_φ is trained on the well-tested subset.
- The culprits g_φ must score are out of distribution: the feature distribution of rarely-called functions differs (more error handling, odd paths).
- **Fix.** Report g_φ calibration binned by the call count of EVAL functions (from pre-existing traces). Weight training by inverse call-count propensity.

### L22. Pre-existing-test leakage in E-TR (Minor-major)

- "Pre-existing tests not touched by the test patch" still include tests that **exercise the buggy path** with benign inputs, the same file as the F2P test, and parametrised siblings of the F2P case.
- That is legitimate inference-time information. But E-TR's coverage concentrates on the gold's file, so E-TR's gain partly reflects "the developer's test file sits next to the bug".
- **Fix.** Report E-TR also with the F2P test's **file** excluded from tracing.

---

## 4. Graph builder

### L23. HEAP cannot import django-style modules without configured settings (Major; A11)

- `django.db.models`, `django.contrib.*` and most of django need `settings.configure()` and `apps.populate()` at import, otherwise ImproperlyConfigured or AppRegistryNotReady.
- A "stub importer" that swallows the error leaves the module half-initialised. Model metaclasses never run, so no ORM descriptors or registered lookups are created, and HEAP sees none of the dynamic edges it exists for.
- Django is ~46% of SWE-bench Verified, so A11 (≥80% import) is at risk on the largest EVAL repo.
- **Fix.** A per-repo bootstrap recipe: django runs `settings.configure(INSTALLED_APPS=[all contrib])` + `django.setup()`, and sphinx creates a dummy app to load extensions. Report HEAP import success per repo, not pooled.

### L24. Import-time HEAP misses lazy imports, PEP 562 `__getattr__` modules and plugins (Minor-major; A11)

- Function-body imports (`def f(): from .backend import x`) create edges only when f runs.
- PEP 562 lazy modules (sklearn, scipy-style `lazy_loader`) resolve on attribute access.
- Entry-point plugins (pytest `pytest11`, sphinx extensions via `setup(app)`) load at app start, not at import.
- **Fix.** Add a static pass over function-local imports (cheap), and touch every `__all__` name to trigger PEP 562. Report plugin-dependent edges as a separate edge kind.

### L25. X-STAT conflates edge recall and transition dilution (Major for its readout)

- HEAP adds override, dispatch and registry candidates, which **raises |C(s)|** at polymorphic sites and lowers A₁ on each true edge (L7).
- X-STAT, with a sparser G, can therefore *win* through sharper transitions even though it misses edges. Or it can lose through recall. The sign of X-STAT − BMM mixes the two.
- **Fix.** Report two numbers: true-path edge recall (L8), and the mean log A along true paths, for static and for static ∪ HEAP.

### L26. DPR on small or modular repos barely rewires, and A is recomputed on fabricated features (Fatal for the C3 DPR kill)

- **Too few swaps.** Degree- *and* module-preserving rewiring in a module of 6 functions where one has out-degree 5 has almost no valid swaps. The DPR graph ≈ G, so X-DPR ≈ BMM. The "DPR ≈ BMM → not structure" kill fires because rewiring did nothing.
- **Fabricated features.** "A recomputed by f_θ" on rewired edges needs features (resolution kind, receiver-type certainty, module distance) that do not exist for fabricated edges.
  - If they are set to "fuzzy", f_θ assigns low p, and DPR transitions become flat. DPR then loses like X-UNIF, **because calibration was destroyed**, not because the topology is wrong. That makes C3 pass for a confounded reason.
  - If features are copied from the swapped edge, the per-edge confidence pattern is preserved, which is fine, but it must be stated.
- **Fix.**
  - Carry each edge's p(c | s) with the edge in the swap (preserve the site-level weight multiset).
  - Report **edge overlap** |E_G ∩ E_DPR|/|E_G| per repo, and require ≤10% overlap or exclude the repo from C3.
  - Report how many true-path edges survive rewiring.

### L27. Static and HEAP choice changes which tasks look hard (Minor)

- The "polymorphic-site share" stratum is computed on G, so it differs between X-STAT and BMM. Per-arm strata are not comparable.
- **Fix.** Define every stratum on the *traced* path (execution facts), never on G.

---

## 5. Ground truth

### L28. CAL-tune leaks into the CAL-train traces, inflating β, P1 and the transfer hopes (Fatal for A14 and the P1 gate)

- f_θ and g_φ are trained on traces of the **CAL-train repos' suites**, and CAL-tune is 100 tasks **from the same repos**.
- If the suites are traced at a commit *after* the tasks' fixes (HEAD), the traces include those tasks' F2P tests. q̃ and A on CAL-tune are then near-oracle for the gold functions.
- TPE then raises β (trusting effects) and lowers ρ (trusting transitions), and P1 (run on CAL-tune) passes easily. None of this transfers to EVAL. This is the exact failure A14 says it will "report" but cannot diagnose.
- Even at a pre-fix commit, g_φ was trained on those repos' functions. In-repo q is better calibrated than out-of-repo q.
- **Fix.**
  - Split CAL **by repo**: CAL-train repos (traces, f_θ, g_φ) and CAL-tune repos (tasks only, with g_φ applied out-of-repo as on EVAL).
  - If that is impossible, trace each CAL repo at a commit **older than every CAL-tune task from that repo**.

### L29. F2P fails before reaching gold: import, collection and argument-binding failures (Major; A6, §2.2.5)

- A very common SWE-bench pattern is that the test patch uses the **new API**:
  - `from pkg.utils import new_helper`, which raises ImportError at collection;
  - `f(x, strict=True)`, which raises TypeError during argument binding, **before PY_START of f**;
  - `obj.new_attr`, which raises AttributeError.
- The base-commit F2P run then never enters gold. "First entry into a gold function" is undefined, or worse, it is a benign earlier call (next item), and the task is labelled off-path for a reason unrelated to the bug.
- **Fix.**
  - Classify base failures: collection/import, binding (TypeError naming the gold's own signature), assertion, or exception-in-repo.
  - Binding failures count as on-path at the gold, with a path ending at the call site.
  - Collection failures form their own stratum.

### L30. "First entry" picks a fixture call, not the bug call (Major; §2.2.5, §4.4)

- The gold may be called during setup (fixtures, `setUp`, module import) long before the failing assertion. The first-entry stack is the fixture's path, not the issue's scenario.
- The M-layer metrics (step hit, first divergence) are then computed against the wrong reference.
- **Fix.** Use a **differential trace**: take the gold invocation whose callee subtree or return value differs between the base and gold-patched runs. Fall back to the last entry before the failing assertion.

### L31. The true path's entry is where the *test* enters, not where the *issue* enters (Major for M1)

- Unit tests often call internals directly (`test__parse_header` calls `_parse_header`). The traced true path is then 1 step long.
- An imagined path from the public API (`requests.get → … → _parse_header`) "diverges at step 1" although it is correct.
- **Fix.**
  - Align imagined and true paths **suffix-first** from gold, and score whether the imagined path is a *valid walk* in the traced call tree ending at gold.
  - Report the exact-step metrics only where the test entry is a public API.

### L32. Test-shaped gold: testing frameworks and in-package test utilities (Major for pytest, django and sympy items)

- pytest-dev/pytest's whole product lives in `src/_pytest/` and `testing/`. Django has real gold sites in `django/test/` (`override_settings`, `TestCase`, `Client`). Sympy has `sympy/testing/`. Sklearn has `utils/_testing.py`.
- v1's rules interact badly:
  - "tests/docs dropped" from the skeleton;
  - ranking over "non-test repo functions";
  - tracer exclusion of "callees in test files".
- If "test" means a path heuristic, these golds are **invisible to the generator, unrankable and stripped from traces**.
- Tracing pytest under pytest also gives an outer harness and an inner `pytester` run. Frames from the system under test look like harness frames.
- **Fix.**
  - Define test files by the repo's discovery config (`testpaths`, `python_files`, django's `tests/` tree), not by substrings.
  - Run pytest-repo tasks with the inner session in a subprocess, traced separately.
  - Report how many golds the "test" rule removes.

### L33. Flaky and environment-dependent survival skews repo mix (Minor-major; A4)

- 2-of-2 runs miss hash-seed-dependent (`PYTHONHASHSEED`), time-dependent and float-BLAS-dependent tests, and matplotlib image comparisons that depend on the freetype version.
- Environment survival will be uneven: sympy and pure-Python repos survive, while C-extension astropy, matplotlib and older sklearn versions fail.
- The headline n ≥ 300 could be ~40% sympy. Sympy is the hub-heaviest repo (L9), so the result says more about sympy than about BMM.
- **Fix.**
  - Pin `PYTHONHASHSEED` and use 3 runs with different seeds.
  - Make the **repo-clustered bootstrap the primary CI**, not a sensitivity check.
  - Cap any repo at 25% of the headline stratum.

### L34. The traceback-free filter selects anti-effect, new-feature items (Major for C2a and C1 power)

- Removing tracebacks removes most issues that state an exception. Exceptions are 9 of 16 bits, so the traceback-free stratum is enriched for "wrong value" and "please add support" issues. That raises the new-code share and lowers effect coverage, so **traceback-free and effect-covered are anti-correlated by construction**.
- The regex also *keeps* issues that paste only the last line (`KeyError: 'content-type'`), with no "Traceback" header and no File lines. Those carry an exception class and a key literal, which BM25 and the effect channel both get for free.
- **Fix.**
  - Also strip bare `^\w+(Error|Exception|Warning): ` lines, or record them as a separate "exception-line" stratum.
  - Report the joint distribution of traceback-free × effect-covered × new-code before setting n.

---

## 6. Controls and claims

### L35. C1 passes for a reason other than decoding: B-PPR does not use the LLM's output (Fatal)

- B-PPR is "PPR from π, **no imagined path**". BMM uses Gemma's 2–8 named symbols. Gemma's names can be right for reasons unrelated to decoding:
  - the skeleton lists them;
  - memorisation of SWE-bench Verified patches (A15).
- **WORKED.** If Gemma names the gold exactly in 35% of tasks (the M1 prediction is 25–40% per step), then B-NAMES alone ranks gold at 1 in roughly that fraction. B-PPR, given only BM25 and no names, cannot.
- C1 = BMM − B-PPR then credits "the technique" with the value of **reading the LLM's answer**.
- The critic's control was "PPR-anchor × BM25 emission". v1 replaced it with "PPR, no imagined path", which drops the emission and so drops the one-change property.
- **Fix.** C1's primary control must be **X-BAG⊗PPR**: the same imagined observations, the same name and effect emissions, π and A used only as a PPR prior, no sequence. Equivalently, BMM with the order-aware forward–backward replaced by "PPR(π) × Π_t max-over-steps emission".
  - BMM − X-BAG⊗PPR is the decoding claim.
  - BMM − B-PPR becomes a secondary "value of the imagination" contrast.

### L36. The positive control passes but the real system cannot succeed, or fails for graph reasons (Major)

- **E-ORA passes trivially.** It uses the F2P true path (ending at gold by construction) with 50% of names noised and F2P effects. The last observation is often gold's own name.
  - Real paths end at the wrong sibling (L2) or the downstream symptom (L5a), and they carry polarity-inverted effects (L16). E-ORA therefore tells nothing about the regime the real system runs in. P1 passing is necessary, not informative.
- **E-ORA can also fail because of G.** If true-path edges are missing from G (L8), the oracle path leaks through ρπ, and P1 (hit@5 ≥0.80) fails. The rule "if P1 fails, no C-claim is read" then **kills the paper because of graph recall**, not the decoder.
- **Fix.** Split P1:
  - **P1a, decoder sanity:** oracle path projected onto G (shortest G-path between consecutive traced nodes), noise drawn from the measured hallucination distribution (L2). Gate ≥0.80.
  - **P1b, graph adequacy:** the edge recall of L8, reported, not gated.
  - Add **P1c, realistic oracle:** the measured M1 error profile injected into true paths. This is the ceiling the real system can approach.

### L37. X-RAND's "≤ B-PPR" passes trivially (Minor)

- X-RAND gives names *exactly* matching random-walk nodes from π. The emissions confidently point at wrong nodes, so it will lose to PPR almost surely.
- That shows only that confident wrong evidence hurts, not that BMM's gain needs the right path.
- **Fix.** Noise X-RAND's names to the same exact-hit rate as real imagined paths. Better, use **paths imagined for a different issue in the same repo** (a matched-distribution null).

### L38. C3 SHUF and DPR confounds (Major)

- SHUF ≈ BMM can come from:
  - the max_t aggregator (L11);
  - T ≤ 2 items (L4);
  - reversed narration (L6), where ordered decoding is already wrong;
  - the δ collapse (L7), where nothing moves anyway.
- DPR ≈ BMM can come from the failed rewiring of L26. DPR < BMM can come from the calibration destruction of L26.
- **Fix.** Apply the fixes to L4, L6, L7, L11 and L26 before C3 is read, and report C3 only on T ≥ 3 items with DPR overlap ≤10%.

### L39. Tuning-budget parity is not capacity parity (Minor-major; §3.4)

- 200 TPE trials over 8 parameters (BMM) vs 2 (PPR), on 100 CAL-tune tasks. BMM can overfit CAL-tune, which is also leaky (L28).
- If BMM transfers worse, C1 is understated. If CAL-tune resembles EVAL, it is overstated.
- **Fix.**
  - Use nested CV on CAL-tune (5 folds), and report the CV-estimated MRR alongside the tuned MRR for every arm.
  - Hold out 30 CAL-tune tasks for the A16 SD estimate. SDs measured on the tuning set are optimistic.

### L40. The skeleton selection policy is a hidden channel (Major; G5 unspecified)

- "≤6k tokens" cannot hold django (~30k functions) or sympy (~40k). Something chooses which ~600 defs appear.
- If the selection is BM25- or embedding-ranked against the issue, the generator's names are partly B-BM25's output. BMM then inherits a retrieval baseline through the prompt, which B-PPR gets only via π.
- If it is file-order truncation, gold visibility varies wildly by repo.
- **Fix.**
  - Specify the policy.
  - Report **gold-in-skeleton** rate per arm-relevant stratum.
  - Add a stratum split on it. C1 on gold-not-in-skeleton items is the honest "structure corrects hallucination" test.

### L41. The ranking universe includes vendored and generated code (Minor)

- `requests/packages`, `astropy/extern`, `sklearn/externals`, sympy's ANTLR-generated parsers and django migrations add thousands of nodes.
- They enlarge hub and tie classes unevenly across arms. BM25 rarely ranks generated parsers, while π·A^t may.
- **Fix.** Exclude vendored and generated paths by a fixed rule, and report it.

---

## 7. Study B (LATENT)

### L42. Negative-set composition leaks the headline stratum's label (Fatal for B1)

- In the headline stratum, positives are *call-time-only, no-name-mention* traced pairs. **By definition they are not static candidates.**
- Negatives are "(site executed ≥5×, candidate never observed)". Candidates come from C(s), the static ∪ HEAP candidate set, and static resolution is largely by *name*.
- So negatives are mostly pairs where v's name **appears** at u's call site. A probe that detects "v's name occurs in u's body" predicts *negative* and separates the classes with AUROC well above 0.5, with zero runtime knowledge.
- **Fix.** Within the headline stratum, draw negatives from the same lexical class: pairs (u, v) co-present in the window, no name mention, not in C(s), with u executed ≥5× and v executed in the same test run but never called from u. Report the name-mention baseline *within* that matched set.

### L43. Overrides share names, so "no name mention" is not lexical-free (Major)

- `super().save()` → `Base.save` (call-time, via MRO). The caller's body mentions "save", which is v's own name.
- If "no name mention" is checked against the qualname, the pair passes. A bare-name check excludes it, and much of the stratum disappears.
- Registry pairs (`@register("json")` vs `get_handler("json")`) share **string literals**.
- **Fix.**
  - Add a lexical-overlap baseline: shared bare method name, shared string literals, and token Jaccard between bodies.
  - Stratify by "shared bare name". Report B1 on the no-shared-name, no-shared-literal subset.

### L44. Two random orders do not remove the causal-attention asymmetry (Major)

- With causal attention, h_u can encode v only when v precedes u. Over two random orders, about half of the positive pairs are attention-possible in one order, and some in neither.
- Averaging the two orders' *scores* gives a mixture. A directed bilinear probe is still tested on pooled pairs where the mechanism is absent half the time.
- Packing whole files or classes as contiguous chunks also keeps **within-chunk distance** informative after chunk-order shuffling. Methods of one class stay adjacent, and override and `self.` pairs live inside one class.
- **Fix.**
  - Report AUROC conditioned on (v before u) separately; the critic asked for this and v1 dropped it.
  - Shuffle at **function** granularity, with class headers duplicated per method.
  - Make the position baseline include the order indicator and its interaction with distance.

### L45. Repo-order and memorisation leakage across CAL and EVAL (Minor-major)

- EVAL repos (django, sympy, sklearn) are far more present in Gemma's pretraining than the small CAL repos. Call-time edges of famous repos may be *memorised* as text facts ("`QuerySet.filter` calls `_filter_or_exclude`").
- B1 on EVAL then measures recall of documentation, not inference from the window. That is fine as a fact, but it is not "runtime knowledge in context".
- **Fix.** Add a **context-ablation** arm: h_u computed with v *absent* from the window, scored against a v embedding from another window. If AUROC holds, the probe reads memory, not the window. Also report B1 on post-cutoff CAL repos.

### L46. AUROC-only NF4 → W4A16 transfer misses calibration shift (Minor)

- T-LAT feeds probe **probabilities** into A. A linear probe under a quantisation-induced residual shift can keep AUROC (rank-invariant to monotone shift) while its probabilities drift a lot. AUROC drop ≤0.03 passes, and T-LAT is miscalibrated.
- "Dequantised W4A16 in HF" is not the vLLM Marlin kernel path either.
- **Fix.**
  - Report ECE and the fitted temperature shift alongside AUROC.
  - Recalibrate (Platt) on a 20k-token W4A16 slice.
  - State that transfer is to "W4A16 dequantised in HF", not to served activations.

---

## 8. The five most damaging breaks, and the minimum fixes for v2

| Rank | Break | Why it is the most damaging | Minimum v2 fix |
|---|---|---|---|
| **1** | **L35. C1's control does not see the imagined path** (with L12) | The one technique claim can pass because BMM reads Gemma's names, including memorised ones, while B-PPR reads nothing. No arm in v1 isolates *decoding* from *reading the answer*. | Primary C1 control = **X-BAG⊗PPR** (same imagined observations, same emissions, PPR prior, no sequence). Demote BMM − B-PPR to secondary. Report the effective-structure KL diagnostic (L12) and the tuned ρ/δ. |
| **2** | **L15 + L16. Effect labels encode normal behaviour at ≥10% frequency; issues report rare, buggy and polarity-inverted behaviour** | The novel piece's emission is *mis-signed* on the "should raise but doesn't" population and *zeroed* on validators. C2b is structurally predicted negative; C2a is diluted or reversed. | Possibility labels (observed in ≥2 tests), not frequency labels. Generator emits ±effects; −effects are scored by a contract model (siblings and base methods). Stratify C2a by polarity, and use likelihood-ratio effect scores with base rates (L14). |
| **3** | **L17. Behaviour is lexical in disguise, and R2 leaves exception tokens and literals unrenamed** | C2a ("behaviour adds more under renaming") is near-tautological. Any unrenamed body-token channel would show it, so the "name-free" headline cannot be distinguished from grep. | Add **X-LEXEFF** (exception-token grep) and **X-NOEFF+BODY** (body BM25 inside the name emission). C2a must beat both under R2. Publish the residual-name share per repo (L18). |
| **4** | **L7 + L8 + L11. Stutter collapse, BM25-only teleport and the max_t readout** | Together they make "structure corrects wrong names" impossible in the model's own algebra: hallucinated middle steps stay on the entry; missing edges reach only BM25-favoured nodes; the readout ranks the entry above the culprit and is blind to order, which also corrupts C3. | Null-emission skip states in place of δ; site-normalised transitions; teleport to a π + uniform/embedding mixture; a position-weighted (culprit-end) readout pre-registered with C3 on T ≥ 3 items. Report true-path edge recall of G in week 1. |
| **5** | **L28 + L36. CAL-tune leakage and an oracle control that validates the wrong regime** | Hyperparameters and the P1 gate are fit where q and A are near-oracle. Then P1 either passes vacuously, or it fails on graph recall and kills the paper for the wrong reason. A14 cannot be diagnosed. | Split CAL by repo (train repos vs tune repos), or trace at commits older than every tune task. Split P1 into P1a (G-projected oracle with sibling-noise from measured hallucinations, gated), P1b (edge recall, reported) and P1c (realistic oracle ceiling). Use nested CV for tuning (L39). |

**Next in line, which v2 should also fix cheaply:**
- L26: DPR overlap and calibration carry-over, or C3's DPR leg is uninterpretable.
- L29 and L30: base-failure classification and a differential-trace gold entry, or the on-path stratum and M1 are mislabelled.
- L42: a matched negative set for Study B, or B1 is a name-mention detector.
- L40: specify the skeleton policy and report gold-in-skeleton.
- L32: a test-file definition by discovery config, for the pytest and django.test golds.

**Not covered here.** The A20 prior-art search (log-to-path, symptom-guided path inference) was out of scope for this logic breaker.
