# Break of architecture v2 (BMM with signed behavioural evidence): logic, edge cases, method, stats, compute. 9 Oct 2026

**Target.** `architecture_v2.md`, read in full, including the §11 change log and assumptions A1–A20. I checked it against `break_v1_logic.md` (L1–L46) and `break_v1_method.md` (M§). The environment facts come from `docs/training_feasibility_4xl4.md` and `docs/HQ.md`.

**Labels.**
- **WORKED**: a constructed example with arithmetic. The repos are real; the snippets are illustrative.
- **INFERRED**: my reasoning.
- **GUESSED**: a number that a kill test must measure.
- **KNOWN**: a public fact I did not re-check today.

Nothing was run.

**Severity.**
- **Fatal**: a primary claim, or the pre-registered fallback, can pass or fail for a reason other than the claimed mechanism.
- **Major**: it biases a primary contrast or stratum by about the predicted effect, or it overruns the budget by 1.5× or more.
- **Minor**: noise, cost or a reporting gap.

Breaks are numbered **N1–N30**.

---

## 0. Verdict in five lines

1. v2 is a real improvement. The controls now see the imagined path, the headline no longer depends on execution yield, and Study B and JS are gone.
2. **Three v1 fixes are renamed, not fixed.**
   - The base-rate likelihood ratio cancels algebraically, so L14 (omnivores) is untouched (**N1**).
   - Insert states recreate the L7 stutter (**N8**).
   - X-BAG⊗PPR is a log-product "naive Bayes over all steps", not a one-change bag, so L35 comes back in a different form (**N14**).
3. **The core claim C-BEH is not powered.** Its pass rule is an intersection of four conditions; joint power is ≈0.3–0.5 at the assumed SDs (**N22**). K3's pass threshold (≥20%) is below what §6's own power table assumes (**N6**).
4. **The C-USE log₂-rank win can come from the tail.** B-DIRECT\*'s tail is BM25 by construction, so BMM can "beat" Gemma at ranks 50–2,000, which no agent reads (**N21**).
5. **Person-days are ≈26 central, not 15.5** (**N27**). GPU is ≈11.6 NH central, not 9.9; GPU fits, people-time does not. The plan is salvageable with ~8 targeted changes and a ~12–13 pd floor (§8).

---

## 1. Were v1's breaks fixed, renamed or worsened?

| v1 break | v2 action | Verdict | Where |
|---|---|---|---|
| L7 stutter | Insert states, h_ψ site propensity | **Renamed.** An insert emits background and keeps the position, which is a stutter with a fixed η₀ = 0.15. The move threshold is now ℓ > 1.48 nats instead of a 3.6× ratio | N8 |
| L8 BM25-only teleport | π = 0.5 BM25(G0) + 0.25 uniform + 0.25 CRE | **Partly renamed.** The uniform share per node is 0.25/30k ≈ 8e-6, which is e^−8.7 below a BM25 favourite. Reachability is now "BM25 or CRE favourites" | N10 |
| L9 hubs | π_∞^λ, skip limited to sub-hubs | **Partly fixed.** π_∞ is ill-defined on a call DAG, and ∅ → callbacks creates a new hub | N11, N12 |
| L10 posterior vs likelihood | Unnormalised sim | **Fixed**, but the insert/match calibration is now arbitrary, because τ sets it | N9 |
| L11 entry readout | w_t ∝ (t/T)² | **Fixed only when the decoder moves.** When inserts hold the entry (N8), the culprit-end weight lands on the entry | N8 |
| L12 effective-structure diagnostic | KL ≥ 0.05 nats | **Ill-defined.** It is the KL against an unnormalised score | N15 |
| L14 omnivores | Base-rate likelihood ratio, μ penalty | **Not fixed.** q̄_j cancels between nodes, and μ = 0.1 with a 0.8 gate is too weak | N1 |
| L15 frequency labels | Possibility labels | **Fixed for validators. A new problem appears:** passing suites encode *intended* raises, while bug reports describe *untested* raises | N3 |
| L16 polarity | ± grammar, peer contract | **Partly fixed.** Peer coverage is low for plain functions, and the "−" attachment is ambiguous | N2, N4 |
| L17 lexical behaviour | X-LEXEFF, X-NOEFF+BODY | **Weakened.** X-LEXEFF lacks MRO, propagation and the non-exception bits, and it is asymmetrically crippled under R2 for repo exceptions | N16, N17 |
| L19 informativeness | Base rate <10% stratum | **Wrong quantity.** The base rate cancels; what matters is the dynamic range of q̃ among plausible states | N1, N6 |
| L28 CAL leakage | Split by repo | **Fixed.** CAL-tune ∩ EVAL repos still needs an explicit rule (SWE-Gym repos sit inside SWE-rebench) | N19 |
| L35 C1 control blind | X-BAG⊗PPR | **Renamed.** The control now sees the path but changes the evidence algebra (log-product over steps) as well as the sequence | N14 |
| L36 P1 regime | P1a/b/c | **Fixed, but P1a is gated and debugged on EVAL-exec tasks,** which means tuning on test | N13 |
| L40 hidden skeleton retrieval | Explicit G0 | **Fixed. New:** R1 reuses the original-issue G0, which leaks the unrenamed issue | N25 |
| M§1 execution EVAL | Static EVAL | **Fixed**, but HEAP needs importable installs for all ~700 snapshots, so the env problem returns through HEAP | N24 |
| M§3.2 skeleton | G0 | **Fixed.** G0 tree prompts are 10–20k tokens for django/sympy, not 4k | N28 |
| M§4.1 C2b | Bit-level | **Fixed for power. External validity is new:** bit labels describe intended raises (N3) |  |
| M§0 scope | 15.5 pd | **Undercounted ≈1.7×** | N27 |

---

## 2. Signed evidence (the novel piece)

### N1. The base-rate "likelihood ratio" is a per-step constant, so L14's omnivore problem is untouched (Fatal for the "IDF-like" fix; Major for C-BEH; A6, A9)

The E⁺ term is β·Σ_j log[((1−η)q̃_vj + η)/q̄_j]. q̄_j is a single number per (repo, bit). For a fixed step t, every match state's log-emission shifts by the same −β·Σ_j log q̄_j.
- In forward–backward this constant multiplies every match state at step t, so it **cancels in every comparison between two functions**.
- In X-BAG⊗PPR (additive) it is a constant shift of all scores, with zero effect on ranking.

**WORKED (v1's L14 numbers, re-run under v2).** The issue states {+ValueError, +writes self}, with η = 0.05.

| Node | q̃ (ValueError, self-write) | E⁺ log-score |
|---|---|---|
| Culprit `_parse_offset` | (0.9, 0.05) | log 0.905 + log 0.0975 − log q̄_VE − log q̄_self |
| `Parser.__init__` | (0.6, 0.95) | log 0.62 + log 0.9525 − (same constants) |

- The difference is still **1.90 nats per step × β in favour of `__init__`**. q̄ never enters.
- The μ penalty fires only on *exception* bits with q̃ > 0.8 that the step does not state. Self-write is not an exception bit. If `__init__` has q̃_TypeError = 0.5, μ adds nothing.
- Even when it fires, 0.1·log(0.9/0.05) = 0.29 nats, which is 15% of the gap.

**What q̄ actually does.** It changes match states relative to the **non-effect states**: inserts (background), ∅, and scope states (whose q̃ is undefined in v2).
- Whenever the reported bit is rare with q̄_j < η = 0.05 (e.g. NotImplementedError, returns-self, AssertionError in a repo with few asserts), **every** match state, including q̃ = 0, gets a positive bonus of log(0.05/q̄_j). For q̄_j = 0.005 that is +2.3 nats × β per step.
- Insert states and ∅ then lose at that step whatever the evidence. So q̄ acts as an **uncontrolled match-vs-insert switch**, not an IDF.
- q̄_j has **no floor** in the E⁺ denominator. Isotonic calibration can output exact zeros, and a bit absent from a small rebench repo gives q̄_j = 0, which is a division by zero.

**Fix.**
1. Make the omnivore correction node-specific. One option normalises per node: score bit j by log q̃_vj − log E_{v'∈N(v)}[q̃_v'j], i.e. contrast against *v's candidate neighbourhood* (the site peers or same-file functions), not the repo mean.
2. Simpler, and pre-registerable: add a size-matched penalty −μ′·Σ_{j∉E} q̃_vj over **all** bits, not only exceptions with q̃ > 0.8. Tune μ′ in place of one of the fixed constants.
3. Floor q̄_j at η.
4. Report Spearman(effect score, LOC) as v2 promises. **Pre-register: if ρ > 0.3, the effect term is size-confounded, and C-BEH is read only on the size-matched stratum.**

### N2. Polarity attachment: "expected but not seen" names a behaviour, not the function that lacks it (Major; A7, A8)

The grammar attaches −E to a *step*. Three common shapes put the missing behaviour on different functions.

| Shape | Example | Where the "−" belongs | What the − term does |
|---|---|---|---|
| (i) Missing check in the culprit | "`parse('2021-02-30')` returns a date instead of raising" | Culprit `_parse_ymd` (it should raise) | Works **if** the culprit has peers that raise |
| (ii) Missing call | Django: "`ModelForm` doesn't run `validate_unique` for …". The culprit `_post_clean` lacks a *call*. The raising function `validate_unique` behaves correctly | The culprit is the **caller**. The function that "should raise" (`validate_unique`) has q̃ ≈ 1 | The first E⁻ term **penalises `validate_unique`** (log(0.05/0.95) = −2.9) and is near-neutral (+0.1) for `_post_clean`. Mild help, by accident |
| (iii) Swallowed exception | "Errors in the callback are silently ignored". The culprit has a bare `except: pass` | The culprit has the try/except. Its *callee* raises | G1 writes −RuntimeError (or similar) on some step. The callee (q̃ high) is penalised; the culprit (q̃ low, because the exception is caught) gets +0.1 |

**The dynamic range of the first E⁻ term is lopsided.**
- q̃ = 0 gives log(1.05/(1.05 − q̄)) ≈ +q̄ ≈ +0.05.
- q̃ = 1 gives log(0.05/(1.05 − q̄)) ≈ −3.0.

So E⁻ is almost purely "**demote raisers**". All positive signal comes from the peer term.

**Peer coverage for plain functions is near zero.**
- peer_j(v) = mean q̃ over the HEAP override family ∪ other candidates at v's call sites ∪ same-name methods in sibling classes.
- A module-level function called through statically resolved sites has |C(s)| = 1, so it has no peers.
- In SWE-bench, django and sympy golds are mostly methods (peers likely). In sklearn/astropy/xarray/rebench utilities, plain functions are common.
- **A8 (≥50% of − steps have peers) is GUESSED, and if it fails, the "−" half of the novel piece is mostly a demote-raisers prior.** v2 does not define the value when peers are absent; I assume the term drops.

**Fix.**
- Report E⁻ performance split by "peer exists".
- Add a **missing-call** reading for −E: score the *callers* of high-q̃_j functions that have a call path to them but did not reach them. This is v1's L5a predecessor readout, specialised to −E.
- Audit A7 against **traced** effects on EVAL-exec (did the culprit's executed behaviour match the polarity at the step G1 attached it to?), not only against the issue text. Agreement with the issue measures parsing, not attachment.

### N3. Possibility labels from passing suites learn *intended* raises; bug reports describe *untested* raises (Major → Fatal for "execution-trained" in C-BEH's sentence; A9)

- A possibility bit fires if the effect is observed in ≥2 distinct tests **of a passing suite**. Suites test *designed* error behaviour: validators raising ValueError, `NotImplementedError` in abstract methods, deprecation warnings.
- The modal "+exception" bug is an *accidental* raise: an unguarded subscript, attribute access on None, a TypeError from an unexpected type. By definition the suite never exercised it, otherwise it would have been caught.

**WORKED.**
- CAL-train holds 10,000 functions with a `d[key]` subscript on a parameter. Perhaps 1% raise KeyError in ≥2 tests (an explicitly tested `KeyError` contract).
- g_φ learns *P(KeyError | subscript feature) ≈ 0.01–0.03*.
- The AST rule says "subscript on a parameter → KeyError possible" (q = 1).
- On an EVAL issue "+KeyError when the header is missing", the culprit is the function with the unguarded subscript, and g_φ gives it q̃ ≈ 0.02. The validator that *deliberately* raises `KeyError("missing header")` gets q̃ ≈ 0.9.
- **Execution training moves mass from the accidental raiser (the culprit) to the deliberate raiser (its well-behaved sibling).** This is v1's L16(a) in a new form.

**C2b at bit level** (macro-AP on CAL hold-out) is measured on the same intended-raise labels. It can pass (g_φ predicts *intended* raises better than AST rules) while hurting task-level C-BEH. The bit-level test has no external validity for issues.

**Fix.**
- Train a second label set from **injected faults or mutation**: call each CAL function with type- and None-perturbed arguments (a cheap fuzz harness on CAL functions with simple signatures, CPU) and record *possible* raises. Use possibility = intended ∪ accidental.
- Or, minimally, pre-register that g_φ is used only for *intended-raise* bits (repo-defined, NotImplementedError, ValueError raised by `raise`), and that AST rules supply the implicit raises (KeyError, IndexError, AttributeError, TypeError).
- Report C2b on **EVAL-exec F2P traces** (the bug input's actual raises): ~100 positives, small but on the right distribution.

### N4. The LLM's polarity from prose: loose statements map to wrong or uninformative bits (Major; A6, A7)

| Issue wording (traceback-free) | Likely G1 emission | Problem |
|---|---|---|
| "`bulk_create` crashes on mixed-case columns" | `+repo-defined` or `+TypeError` (guessed) | The class is not stated. The traceback that named it was the reason the issue was *excluded* from traceback-free. The guess is noise with high confidence |
| "`autodoc_typehints='description'` doesn't show the return type" | `~value`, or invented `−returns None` | ~value is neutral, so the issue contributes nothing (fine). An invented −returns None demotes every function that returns None, i.e. ~40% of nodes |
| "This gives the wrong result for negative inputs" | `~value` | Neutral; it lands outside the stratum unless G1 invents codes |
| "It should warn but doesn't" | `−warns` | Matplotlib/astropy warn through wrappers. With 1-hop propagation, the culprit (which lacks the call) has q̃ ≈ 0, so the E⁻ first term gives +0.05. Everything rests on the peers |
| "Raises `ImproperlyConfigured`" (prose, no traceback) | `+repo-defined` | The repo-defined bit pools all repo exceptions. In django, q̄_repo is high and the bit has little dynamic range |

**The stratum is defined from G1 output** (≥1 informative ± effect in ≥3 of 10 samples). With an alphabet and glosses in the prompt, temperature 0.7 and 10 samples, G1 will **emit codes on many issues that state none**. The stratum then fills with invented effects, which dilutes Δ toward 0. v2 measures A7 precision against the issue on 50 items, which gives a ±0.13 CI.

**Fix.**
- Add an explicit "no effect stated" option in the grammar.
- Require the G1 effect to cite an issue span (`+KeyError @"raises KeyError"`), and drop codes whose cited span fails a substring check (CPU, free).
- Define the C-BEH stratum as G1-coded **and** span-verified.
- Report the invented-code rate on the regex-negative issues.

### N5. "Repo-defined" is one bit for all repo exceptions (Minor-major; A6)

- `ValidationError`, `FieldError`, `ImproperlyConfigured` and `SuspiciousOperation` all fire the same bit.
- MRO closure helps only when a repo exception subclasses a builtin in the alphabet (`ValidationError` subclasses `Exception` only).
- In django, q̄_repo-defined is high, so the issue that names `FieldError` gets the same evidence for every function that raises any django exception.

**Fix.** Make repo exceptions **per-class bits created on demand**: when a step names a class C that resolves in the repo, emit bit "raises C (MRO-closed)" from the AST rule and g_φ's generic "raises repo exc" probability times an AST indicator for C. This is cheap and removes a large tie class.

### N6. K3 gates the wrong quantity, at the wrong threshold, at the wrong time (Major; A6)

**Wrong threshold.**
- §6 sizes C-BEH at 200–300 tasks (33–50% of 600), giving power ≈0.75 per log-rank contrast.
- K3 *passes* at ≥20%, which is 120 tasks. At 120 tasks the stratum has ≈5 per repo, so DE ≈ 1.2 and n_eff ≈ 100.
- SE = 0.9/√100 = 0.09, so z = 0.25/0.09 = 2.78. Power at Holm α = 0.0167 (z_crit 2.39) is Φ(0.39) ≈ **0.65 per contrast**, and lower jointly (N22).

**Wrong quantity.**
- "Informative = base rate <10%" is irrelevant to ranking by N1. What matters is the dynamic range of q̃_j among the plausible candidates (G0 files).
- At Day 3–4 q̄ is not yet computable: g_φ needs CAL traces (week 1), and EVAL graphs need HEAP. Only AST-rule base rates are available.

**Wrong time.** The stratum is defined from **G1 output**, not issue regex. The regex proxy can be off in either direction (N4 inflation; L34 deflation, since tracebacks are stripped).

**My prior (GUESSED).**
- About 35–40% of SWE-bench issues have tracebacks. Of the remainder, ~20–25% name an exception class in prose, ~3–5% state a warning, and ~5% state "should raise / instead of raising".
- If bare exception lines are excluded from the headline, the regex share is ≈15–22%. That lands in or just above K3's ambiguous band.

**Fix.**
- Set K3 to pass at ≥35%, ambiguous at 20–35%, and fail at <20%, to match §6.
- Measure it twice: Day 3 by regex (CPU), and at K9 by span-verified G1 output on the 50-task pilot.
- Pre-register that if the pilot stratum is <150 tasks projected, C-BEH becomes a secondary estimate, and the headline row switches **before** EVAL decoding.

### N7. h_ψ propensity down-weights the error-handling sites where bugs live (Major; A10)

- p(s | u) ∝ h_ψ(s) = P(site executed in the CAL test suite).
- Error-path sites (`if not valid: return self._handle_invalid(x)`, `except E: self._recover()`) are rarely executed in suites.

**WORKED.**
- `u` has a main-path site with h = 0.9 and an error-path site with h = 0.1.
- An issue about the invalid-input path must route u → `_handle_invalid`. The transition is ×9 smaller than to the main-path callee, so emissions must supply ≥2.2 nats to overcome it.
- That is exactly the regime where the effect channel (+exception) is supposed to help, but the transition prior pulls toward the happy path.

**Fix.**
- Use a mixture p(s | u) = ½·uniform + ½·h_ψ.
- Or condition on the step's polarity: when E⁺ ∪ E⁻ contains an exception, use 1 − h_ψ for sites inside `if`/`except` blocks guarding raise/return.
- Report the true-path h_ψ distribution on EVAL-exec.

---

## 3. Decoder

### N8. Insert states re-create the L7 stutter, and L11's entry bias comes back with them (Fatal for "structure corrects wrong names"; A10, A13)

The insert state (v, ins) emits background (log-ratio 0) and keeps the position at v. That is "stay on v" with a fixed probability η₀ = 0.15 and a neutral emission.

**WORKED (v1's L7 setting).**
- Entry E = `Session.request` has 25 sites with similar h_ψ.
- Move probability = (1 − η₀ − ε − ρ)·p(s|E)·p(M|s) ≈ 0.75 × 0.04 × 1 = 0.030 (assuming ρ = 0.05).
- Stay-by-insert = 0.15 × e⁰.
- **Move iff ℓ_M(o₂) > ln(0.15/0.030) = 1.61 nats.**
- The hallucinated step `Session.merge_proxies` vs `merge_environment_settings`: containment |{merge}|/|{merge, proxies}| = 0.5, so the name term is 0.5·τ. With τ ≈ 2–3 that is 1.0–1.5 nats, which is **below threshold**. The path stays on E.
- v1's threshold was a 3.6× ratio (1.28 nats) *relative to E's own emission*. v2's is 1.61 nats *absolute*. **This is not easier.**

**Readout consequence.**
- If steps 2–4 insert at E, then γ_t(E) is high at t = 2–4. With w_t ∝ (t/T)², E receives weight (4 + 9 + 16)/30 ≈ 0.97 of the readout. **The culprit-end weighting puts its weight on the entry.**
- v2 does not say whether (v, ins) posterior counts toward v's score. If it does, L11 is back. If it does not, inserted steps contribute nothing, and the culprit's score comes only from steps that moved.

**Fix.**
- Count insert posterior as **unassigned** (no readout mass).
- Make an insert "consume the observation and advance to the *best-scoring callee neighbourhood*": a delete-and-jump in which the observation is skipped but the position may advance one hop. In profile-HMM terms, add **delete states** (advance position without emission) as well as inserts. v2 has inserts only, and a delete is exactly what a hallucinated *middle* step needs.
- Tune η₀ on CAL-tune instead of fixing it. That makes 5 tuned parameters, so drop λ (N11).
- Pre-register P1a with **sibling-substituted middle steps**. If hit@5 < 0.8 because the path stays on the entry, that is this break.

### N9. An unnormalised name similarity makes τ set two things at once (Major; A12)

- sim ∈ [0, 1] is ≥ 0 for every node: containment plus a CRE cosine that is rarely below 0.2–0.3 for any code.
- So α·τ·sim gives every match state a positive log-ratio over the background, and **τ simultaneously sets name sharpness and the match-vs-insert threshold** (N8).
- Tuning τ on CAL-tune (small repos, short names) and applying it to django (30k nodes) shifts both at once. log|V| normalisation fixes the sharpness, not the insert threshold.

**Fix.** Centre sim per step: sim′ = sim − median_v sim over the G0-file nodes. A random node then has log-ratio ≈ 0, and the insert threshold becomes meaningful.

### N10. The mixed teleport: reachability is now "BM25 or CRE favourites", not "any node" (Major; A10)

π = 0.5·softmax(BM25 boosted to G0) + 0.25·uniform + 0.25·CRE.

**WORKED (django, |V| ≈ 30k).**
- The uniform share per node is 0.25/30,000 = 8.3e-6.
- A BM25 favourite gets ≈0.5 × 0.1 = 0.05. The gap is ln(0.05/8.3e-6) = **8.7 nats**.
- CRE softmax over 30k nodes is peaked as well. A non-top node gets ~1e-5.
- After a missing edge, a low-BM25, low-CRE culprit needs ≈8.7 nats of emission evidence to win the teleport. The name channel can supply that only on an exact name hit, and then the graph was not needed. The effect channel's maximum is ≈3 nats × β per bit.
- **L8 survives**: missing-edge recovery works only for nodes that a retrieval baseline already ranks well.

**Under R2:** BM25 and CRE are both renamed, but G0 file paths are not. The teleport collapses to "uniform within G0 files", which is the useful case, but only when the gold file ∈ G0.

**Fix.**
- Make the uniform component **uniform over the 1-hop and 2-hop graph neighbourhood of the decoded posterior so far** (a local teleport), not over V.
- Or tune the mix weight.
- Report "gold reached only via teleport" per arm.

### N11. π_∞ "of the transition-only chain" is undefined on a call graph (Major; L9 fix)

- A call graph is close to a DAG with sinks (leaf functions, ∅). The transition-only chain (no teleport) is reducible, and its stationary distribution concentrates on absorbing leaves, or is not unique.
- Leaves have no sites, so their row mass goes to insert/ε only. Unless that is renormalised, the matrix is not even stochastic.
- With teleport included, π_∞ depends on ρ and on the per-issue π, so it is not a hub prior.

**Also, the direction is wrong for gold.** HQ A1 (VERIFIED, 129 tasks) found gold more central than random (degree AUC 0.78). Dividing by π_∞^λ penalises gold on average. λ will be tuned toward 0, which wastes one of the four tuned parameters.

**Fix.** Define the hub mass as PageRank with a uniform restart at a fixed 0.15 on the **undirected** graph. That is well-defined and issue-independent. Fix λ = 0 by default, and report hub share in the top 10. If the share is >30% for sympy, switch λ on, pre-registered.

### N12. The global ∅ → callback transition is a new hub teleport (Major; L3 fix)

- "Callbacks passed to external calls" include `sorted(key=f)`, `map(f, …)`, `functools.partial`, `defaultdict(factory)`, `re.sub(repl=f)` and `atexit`. Sympy and sklearn pass functions to `map` and `key=` everywhere.
- With **one** ∅ state, every external call (`json.dumps`, `re.compile`, `np.asarray`, which is most functions) can transition ∅ → any callback, weighted by registration count. That is a global jump to frequently registered functions: hubs again.

**Fix.**
- Make ∅ **site-local**: ∅_s → only the callbacks passed *at that site* (or registered to the same registry).
- Keep a global ∅ only for framework registries (signals, URL conf, sphinx events, pytest hooks), with mass split by registry.

### N13. P1a is gated and debugged on EVAL-exec, so the decoder is tuned on test tasks (Major; L36 fix)

- K6 says: "hit@5 ≥0.80 on EVAL-exec … Debug the decoder; no EVAL GPU until it passes."
- EVAL-exec is a subset of EVAL. Iterating decoder design until an EVAL subset passes is design-on-test, even if the scored quantity is an oracle path. The decoder fixes (η₀, ε, skip rules) change EVAL results.

**Fix.** Gate P1a on the **~40 executed CAL-tune tasks** v2 already plans. Read P1a/P1c on EVAL-exec once, after freeze, report-only.

### Both-orientation decoding (L6 fix): not double counting, with one wrinkle (Minor)

- Taking the max over the two orientations is a selection, not a sum, so there is no double counting.
- **But it changes which end the culprit weight lands on.** For T = 2 in a directed call graph, typically only one orientation has an edge, so the max selects the direction with the edge, and w_t then weights its callee. That is sensible.
- For "+exception" crash issues, the callee is often the *raiser* (a validator or a library), not the culprit (L5a). The predecessor readout is only secondary.
- **Fix.** Pre-register the predecessor readout as the primary readout on the "+exception" polarity stratum, chosen on CAL-tune. Also give X-SHUF and X-BAG the orientation max, so they are not handicapped.

### Tuning ≤4 parameters with nested CV: feasible (Minor)

- Decodes for β, τ and ρ must be recomputed; λ is readout-only.
- A 6³ grid × 100 tasks × k = 10 × 2 orientations × T ≈ 5 × nnz ≈ 3e5 gives ≈0.1–0.3 s per sample, so ≈2–5 kernel-hours per arm.
- Nested CV reuses the cached decodes, so it is cheap.
- About 12 arms × (original + R2) ≈ 50–120 kernel-hours. **The central 30 kernel-hours for decoding is low.** Use a coarse grid (4³) or Bayesian optimisation; the 100-task SE (≈0.03 MRR) cannot tell finer grid points apart anyway.

---

## 4. Controls

### N14. X-BAG⊗PPR is a log-product over steps, so C-DEC changes the evidence algebra as well as the sequence (Fatal for C-DEC's one-change claim and for the BER fallback; L35)

score_BAG(v) = κ·log PPR(v) + Σ_t w_t ℓ_v(o_t). That is a **product of every step's likelihood on the same node**: a node scores high only if it matches *all* steps.

BMM instead assigns each step to a *different* node (a posterior mixture). The bag therefore:
- **rewards omnivore and entry nodes** whose names share tokens with several steps (`Session.request` vs steps "session", "request", "proxies");
- **penalises a culprit that matches only the last step.** Its E⁻ and μ terms on *other* steps (which describe other functions) are summed onto it.

**WORKED.**
- Steps: (1) `Session.request`; (2) `merge_environment_settings`; (3) `get_environ_proxies | −no_proxy-handling`, i.e. +returns None with ~value on the culprit.
- The culprit `should_bypass_proxies` matches step 3 at ℓ = +2.0 and steps 1–2 at ℓ ≈ +0.3 each (the sim floor). Its bag score is ≈ w₃·2.0 + small.
- `Session.request` gets ℓ = +3.0 at step 1, +0.6 at step 2 and +0.3 at step 3, for a bag score of ≈ w₁·3 + w₂·0.6 + w₃·0.3.
- With w = (1, 4, 9)/14: culprit ≈ 1.39 + 0.11 = 1.50; entry ≈ 0.21 + 0.17 + 0.19 = 0.57. Here the bag *does* pick the culprit, thanks to w_t.
- Now add an E⁻ term on step 2 that penalises functions returning None, with the culprit at q̃_None = 0.9: −3.0 × w₂ = −0.86. The culprit falls to ≈0.64, roughly tied with the entry.
- **In the bag, evidence about one step's function leaks onto every node.** BMM's step assignment prevents that.

So C-DEC can pass because a product-of-experts bag is a poor aggregator, not because *sequence* matters. And the BER fallback (row 2 of §0.5) *is* this arm, so the fallback headline would rest on a mis-specified scorer.

**Fix (exact one-change).** X-BAG⊗PPR = **the same HMM with A replaced by the rank-1 matrix 𝟙·PPR_πᵀ**. Every step is drawn i.i.d. from the PPR prior, with the same insert and ∅ states, the same orientation max, the same readout Σ_t w_t γ_t / π_∞^λ, and no κ.
- Equivalently, BMM at ρ = 1 with π := PPR_π.
- Each step then gets its own posterior, softmax_v(ℓ_v(o_t) + log PPR(v)), and the readout sums them.
- This is the "mixture bag". **C-DEC = BMM − this.** The KL diagnostic becomes well-defined (N15). BER becomes a proper scorer.

### N15. The KL ≥ 0.05 nats diagnostic is not well-defined (Minor-major; L12)

- X-BAG⊗PPR's score is an unnormalised log-score. Turning it into a distribution needs a temperature, and KL scales with it, so 0.05 nats has no meaning.
- Over 30k nodes, the KL is dominated by the tail mass.

**Fix.** With N14's fix both are posteriors (KL defined). Better still, use a rank statistic: the median Jaccard of the top-10 between BMM and X-BAG. Pre-register "C-DEC reported as bag + prior if the median top-10 Jaccard ≥0.8".

### N16. X-LEXEFF is a weak grep, so C-BEH can pass on MRO, propagation and the five non-exception bits (Major → Fatal for the "behaviour, not grep" reading; L17)

X-LEXEFF = "body contains `raise C` / `except C` / `warnings.warn` for a class C named in the step". Compared with BMM's q̃, it lacks:
1. **MRO closure.** `raise InvalidURL` does not match "ValueError".
2. **Exception propagation** (≤2 hops) and the wrapper propagation for warn (1 hop).
3. **Implicit raises** (subscript, attribute, call).
4. **Five bits entirely**: returns None, returns self, mutates an argument, writes a self attribute, plus the I/O feature.

C-BEH = BMM − X-LEXEFF therefore credits "behaviour" with deterministic static analysis (MRO, call-graph propagation, AST rules). The novel claim is "per-function effect signatures from a cross-repository execution-trained model", and **only C2b (g_φ vs AST rules) isolates the learned part**. At task level that contrast was underpowered (M§4.1: +0.015 MRR, power 0.17–0.58).

**Fix.** Split the C-BEH claim honestly into two contrasts:
- **C-BEH (static behaviour vs lexical):** X-ASTEFF − X-LEXEFF⁺, where LEXEFF⁺ = LEXEFF with MRO closure and the same propagation. This is "typed effect inference beats token grep".
- **C-BEH-learned (descriptive):** BMM − X-ASTEFF at task level, plus C2b at bit level.

Reword the sentence to say "execution-trained" only if both are positive.

### N17. Under R2, X-LEXEFF is blinded on repo exceptions while BMM's repo-defined bit survives (Major; L17)

- R2 renames repo identifiers, including repo exception classes (`InvalidURL` → `C_8f2`). The issue is not renamed and still says "InvalidURL".
- X-LEXEFF looks for `raise InvalidURL` in a renamed body and finds nothing.
- BMM resolves `raise C_8f2` to a repo exception class (AST/MRO, name-independent), fires the repo-defined bit, and matches the step's "+repo-defined".
- **C-BEH under R2 then contains the exact L17 tautology through the repo-exception door.**

**Fix.** Give X-LEXEFF the pre-rename identity map for exception classes (the renamer knows it), or a binary "raises any repo exception" mapped to the repo-defined bit. Then report C-BEH separately on builtin-only and repo-exception steps.

### N18. X-NOEFF+BODY swaps a channel rather than ablating one, and its query is unspecified (Major; L17)

- **It changes two things:** β = 0 *and* body-BM25 is added. The contrast asks "effects instead of body tokens", not "effects beyond body tokens".
  - A complementary-but-weaker effect channel fails this test even if it adds value.
  - The natural claim is additive: (BMM + BODY) − (X-NOEFF + BODY), which is exactly one change (β = 0).
- **The query is not specified.**
  - If it is the step name n_t, under R2 G1's names come from the renamed skeleton, so they match renamed bodies only by the renamed tokens.
  - If it is the issue text, the control gains issue-level information that BMM's name channel lacks. That is a confound in either direction.
- **The fusion weight is unspecified.** It is a fifth tuned parameter.

**Fix.**
- Primary C-BEH = (BMM + BODY) − (BMM + BODY with β = 0), with query = step name + the issue sentence G1 cites, the same in both arms.
- Keep "BMM − X-NOEFF+BODY" as a secondary contrast.
- Tune the body weight once on CAL-tune and share it.

### Rewire, null and other controls (Minor)

- **X-DPR, degree- and module-preserving with ≤10% overlap.**
  - With module-pair preservation, most module pairs have 1–3 edges, so no valid swap exists. Expect few repos to pass K7 (≥15 repos is GUESSED and probably fails).
  - Under DPR, consecutive steps are no longer adjacent, so mass goes to teleport and insert, and X-DPR ≈ X-BAG with teleport. It largely re-measures C-DEC.
  - **Fix:** relax to degree-preserving within the *package* (top-level directory), not the module. State that DPR is a C-DEC replicate.
- **X-NULL** (paths imagined for another issue in the same repo):
  - If the other issue touches the same files (common in django `query.py`), the null is not null.
  - Its paths were generated at a *different base commit*, so some names do not exist at the target commit.
  - **Fix:** choose the donor issue with gold files disjoint from the target's and a commit within ±1 minor version. Reuse existing G1 outputs (0 GPU).
- **Scope states only in BMM's universe.** B-DIRECT\* lists functions. If class-body golds are rankable only by graph arms, the comparison is unfair. **Fix:** map scope-state golds to "class X (body)" tokens that every arm can output, or report the scope-gold stratum separately and exclude it from C-USE.

---

## 5. EVAL construction, contamination, G0

### N19. Pool arithmetic: SWE-bench supplies ≈225 tasks; ≥375 must come from rebench/Loc-Bench, which overlap the CAL-tune sources (Major; A3, A5)

**KNOWN approximate counts** (Verified / Lite): django 231/114, sympy 75/77, sphinx 44/16, matplotlib 34/23, sklearn 32/23, xarray 22/5, astropy 22/6, pytest 19/17, pylint 10/6, seaborn 2/4, flask 1/3. requests is excluded as a dev repo.

**WORKED.** Take the union, apply traceback-free 0.65 × non-new-code 0.88 = 0.57 (M§1.3 central), and cap at 40:
- django 40, sympy 40, sphinx ≈31, matplotlib ≈28, sklearn ≈27, pytest ≈18, astropy ≈15, xarray ≈14, pylint ≈8, seaborn ≈3, flask ≈2.
- **Total ≈226 from 11 repos.**

**Consequences.**
- ≥375 tasks from ≥14 repos must come from SWE-rebench and Loc-Bench. Most rebench repos contribute few tasks, so 25 repos is easy, but n_eff depends on the size distribution. That is fine.
- **Source is confounded with repo size and with contamination.** SWE-bench repos are large, old and memorised. Rebench repos are small and newer, with LLM-filtered issue texts. Log₂-rank ranges also differ (≈15 bits for django vs ≈9 for a 500-function repo). The clustered bootstrap absorbs the variance but not the interpretation. **Report every primary by source.**
- **Overlap and duplication.**
  - Lite ∩ Verified ≈ 90–100 tasks: dedupe by instance_id.
  - SWE-Gym's ~11 repos (pandas, moto, mypy, dask, dvc, conan, pydantic, monai, …) also appear in SWE-rebench.
  - Loc-Bench was built from GitHub issues in 2024–25 and may share repos and PRs with rebench.
  - **Dedupe by (repo, PR number), and by issue-text near-duplicates (MinHash ≥0.8).**
  - State the rule that CAL-tune repos are excluded from EVAL. v2 says "CAL repos", which is ambiguous for CAL-tune.

### N20. Memorisation: R2 does not remove it, it leaks through G0 paths and doclines, and it biases C-USE toward null (Major; A15)

- **G0 sees unrenamed file paths** (R2 renames code only). For Verified, Gemma has very likely seen the fix commits (KNOWN: 2012–2023 GitHub history). File-level recall from memory survives R2 fully.
- **Doclines are unrenamed.** sim uses cosine(CRE(n), CRE(signature + docline)). A memorised original name `add_filter` emitted by G1 matches the renamed node through its docline "Add a single filter to the query". Function-level memorisation leaks into R2.
- **The direction of bias.**
  - B-DIRECT\* outputs the memorised gold directly.
  - BMM dilutes the memorised final step with path posteriors and w_t over other steps.
  - So memorisation **favours B-DIRECT\* over BMM**, which biases C-USE toward null on SWE-bench and in its favour on post-cutoff tasks.
  - For C-BEH and C-DEC, memorisation is mostly shared by both arms, but it **compresses Δ**: when G1 names the gold exactly, every arm finds it.
- **Fix (cheap).**
  - Add **B-RECALL**: Gemma given the issue + repo name only (no tree, no skeleton) → top-10 functions, about 600 × 1k tokens (≈0.15 NH).
  - Its hit@10 per source is the **memorisation index**. Stratify every primary by B-RECALL hit vs miss. That has far more power than pre/post-cutoff (n ≈ 100).
  - Strip doclines from R2 node text, or rename identifiers inside doclines.

### N21. Shared G0: everything is conditional on G0, but not capped equally (Major for C-USE; A14)

- **G0 recall.** Agentless-style file localisation with top-10 files is GUESSED at 85–90% on Verified (memorised) and 65–80% on rebench/Loc-Bench.
- **G0 is not a uniform cap.**
  - LLM-list arms (B-DIRECT\*) can rank only skeleton symbols in the top 10, then a BM25 tail.
  - Graph arms can reach out-of-G0 files through edges, but only with the teleport handicap of N10.
  - The gold-not-in-G0 stratum is where C-USE is decided, and it is also where log₂-rank differences are largest.
- **C-USE's log₂-rank compares BMM's graph tail with B-DIRECT\*'s BM25 tail.**
  - **WORKED.** A gold at rank 400 under BM25 and rank 60 under BMM gives Δlog₂ = −2.7, although neither arm puts it in the top 10.
  - If 40% of tasks are "neither in the top 10", with a mean tail Δ of −1.0, the mean Δlog₂ = −0.4. **That is exactly C-USE's predicted effect, produced entirely by tail ordering** that no agent will ever read (HQ: agents read ≤10 candidates in ≈2 calls).
- **Fix.**
  - C-USE primary = **hit@10 / MRR**, or log₂-rank **truncated at 100**.
  - Or tail-fill B-DIRECT\* with **X-BAG⊗PPR** (the same evidence), so the tails differ only in the top-10 head.
  - Report C-USE separately on gold-in-G0 and gold-not-in-G0.

### Static gold edge cases (Minor)

- **Multiple gold functions** (Verified median ≈1–2). v2 does not say whether the rank is the min over golds or a mean. Min favours arms that spread mass. **Pre-register min rank** (first relevant) with all-gold recall@10 as secondary.
- **Hunks on a `def` line or decorator, deletion-only hunks, and hunks between two methods.** These need a rule: assign to the nearest enclosing scope, or to the following def for decorators.

---

## 6. Executed subset via image filesystems on Kaggle (A4)

### N24. Feasibility of `crane export` without Docker (Major for EVAL-exec and HEAP; A4, A11)

**What works (INFERRED).**
- CPU kernels have Internet. `crane` is a static Go binary (pip-free download).
- SWE-bench instance images (`swebench/sweb.eval.x86_64.<id>`, KNOWN) contain `/opt/miniconda3/envs/testbed` and `/testbed`.
- The conda env bundles its own libs, so extracting **only `/opt/miniconda3` and `/testbed`** next to Kaggle's `/opt/conda` should run with the host glibc, which is newer and backward compatible.

**What breaks.**
1. **Do not untar to `/`.** M§1.4 suggested `tar -C /`. The image is Ubuntu 22.04, and overwriting the kernel's `/usr` and `/lib` kills the session. Extract selected prefixes only. The conda prefix is hard-coded, so it must land at exactly `/opt/miniconda3`.
2. **Docker Hub pull limits** (KNOWN; exact numbers changed in 2025, re-check). Anonymous pulls are rate-limited per IP, and **Kaggle egress IPs are shared across users**, so 429s are likely. A Docker Hub token in Kaggle Secrets is needed, and **Shrey must enter it himself.** At about 100 pulls per 6 h authenticated, 150 instances needs ≥1 extra session.
3. **Flattened export loses layer sharing.** That means 150 × 1–3 GB ≈ 150–450 GB transferred. Process images serially and delete each one. The ephemeral disk size of a Kaggle CPU kernel is **unverified**; `/kaggle/working` is 20 GB. Plan per-image peak ≈6 GB.
4. **Network-off sandbox for HEAP.** `unshare -n` needs CAP_SYS_ADMIN, which Kaggle containers lack (INFERRED). Use a socket monkeypatch in-process, which is "best effort", and say so.
5. **HEAP needs importable installs for all ~700 snapshots, not 150.**
   - sklearn, astropy and matplotlib do not import without built extensions (`_check_build`).
   - Either HEAP also uses images (~226 SWE-bench snapshots, plus rebench images, which are available, KNOWN), or a per-commit `build_ext` takes 5–20 min (M§1.2).
   - CPU: 600 × (pull + extract 5 min + import walk 5–30 min) ≈ **100–350 kernel-hours**, against v2's central 36 + 40 = 76. HEAP central is ≈2×.

**settrace overhead.** With `f_trace_lines = False`, the global trace function still runs on every call, so call-heavy sympy runs 3–10× slower. F2P tests are seconds long. 150 × (3 validation + base/gold differential traces ×3) × 1–5 min ≈ 25–75 kernel-hours, close to v2's 30 central. That is fine.

**Fix.**
- Run K1 with the selective-prefix recipe and an authenticated pull.
- Count HEAP via images on the SWE-bench share.
- Pre-register "HEAP only where an image or pure-Python install exists". The rest use AST + Jedi, reported per repo (already rung 9).

---

## 7. Statistics: is anything headline actually powered?

### N22. C-BEH's pass rule is an intersection of four conditions; joint power ≈0.3–0.5 (Fatal for "core claim is powered"; A13)

**C-BEH passes iff** both log-rank CIs exclude 0 (vs X-NOEFF+BODY and vs X-LEXEFF) **and** both MRR lower bounds > −0.01.

**WORKED (v2's own SDs, n_eff 120, Holm α = 0.0167 so z = 2.39).**

| Condition | Arithmetic | Power |
|---|---|---|
| Log-rank, each contrast | SE = 0.9/√120 = 0.082; z = 0.25/0.082 = 3.04 | **0.74** |
| MRR non-inferiority, each contrast | SE = 0.15/√120 = 0.0137; pass iff Δ̂ > −0.01 + 2.39·0.0137 = 0.0227; true Δ = 0.03 → Φ((0.03 − 0.0227)/0.0137) = Φ(0.53) | **0.70** |
| MRR NI at 95% instead (z = 1.96) | — | 0.83 |
| **Joint** | 0.74² × 0.70² = 0.27 if independent. Positive correlation between the contrasts (same BMM, overlapping controls) raises it | **≈0.35–0.5** |

- **If the true ΔMRR is 0**, the NI condition passes only ≈5% (98.3% bound) to ≈11%. That is correct for NI, but it means C-BEH needs a real MRR gain, while the table calls MRR "underpowered and not claimed".
- **At K3's minimum pass (120 tasks, n_eff ≈ 100), joint power ≈ 0.25–0.4.**

### N23. C-DEC and C-USE power

**C-DEC.** Log-rank power ≈0.99. MRR NI: SE = 0.15/√280 = 0.009, pass iff Δ̂ > 0.0115, true 0.03 → 0.98. Plus the KL rule.
- **C-DEC is the only powered primary**, provided ICC ≈ 0.05.

**C-USE.** Log-rank: SE = 1.5/√280 = 0.09, z = 4.4 → ≈0.99 (v2 says 0.95). MRR NI: SE = 0.3/√280 = 0.018, pass iff Δ̂ > −0.02 + 0.043 = 0.023, predicted +0.02 → Φ(−0.17) = **0.43**.
- **C-USE joint power ≈0.4.** Its log-rank power comes from the tail (N21).

**ICC sensitivity.** The Δ between arms of one method family varies by repo with HEAP import success, g_φ transfer and residual-name share. ICC = 0.15 is plausible (GUESSED). Then DE = 1 + 23·0.15 = 4.45 and n_eff ≈ 135 for 600:
- C-DEC log-rank power ≈ 0.85;
- C-USE MRR NI ≈ 0.3;
- C-BEH log-rank per contrast ≈ 0.5.

**25 clusters.** The percentile cluster bootstrap undercovers by a few points at G = 25 with unequal sizes (INFERRED). Use a **wild cluster bootstrap (Rademacher) or a BCa cluster bootstrap**, and report the cluster count per stratum. The C-BEH stratum may span only ~15–20 repos.

**Holm over 3 with intersection–union.** This is valid (IU needs no extra α). But the "one variant only" fused-by-RRF rule makes the **tail BM25-dominated in every arm**. That shrinks log-rank Δ for C-BEH and C-DEC toward 0, while the SD assumptions came from unfused reasoning. **Measure SDs on the fused metric in the pilot.**

**Fix for N22/N23.**
- Make **log₂-rank (truncated at 100) the single primary metric** per claim. Report MRR and hit@10 as descriptive with CIs, and drop the MRR NI conditions from the pass rules. Or keep NI at 95% with margin −0.02.
- For C-BEH, use **one** control (the N16/N18-fixed (BMM + BODY) − β = 0 contrast). Report X-LEXEFF⁺ as C-BEH-static.
- Raise the C-BEH stratum to ≥200 by including the exception-line stratum (bare `KeyError: …` lines) in the C-BEH test only. The effect channel and BM25 both see the token, and X-LEXEFF⁺ controls for that.

---

## 8. Budget recount

### 8.1 GPU (NH)

| Line | v2 central | Recount opt / central / pess | Why |
|---|---|---|---|
| P0v | 0.4 | 0.3 / 0.4 / 0.5 | — |
| G0 (870 tasks) | 0.6 | 0.6 / **0.9** / 1.3 | Repo-tree prompts are ≈10–20k tokens for django/sympy/sklearn (≈900–1,500 non-test .py paths × ~12 tokens), not ~4k. Mean ≈6k × 870 = 5.2M prefill |
| G1 (870 × k = 10) | 2.7 | 1.8 / 2.7 / 4.2 | Agrees: ~6M prefill (prefix shared across samples) + ~1.7M decode |
| G2 R2 + R1 | 1.6 | 1.2 / **1.7** / 2.5 | + R1 must re-run G0 on the renamed issue (N25) |
| B4 B-DIRECT\* + B-RERANK | 1.3 | 1.0 / **1.6** / 2.4 | 1,600 B-DIRECT calls (3 variants on CAL-tune, original + R2 on EVAL, pilot). Variant (ii) has a ~500-token decode. Rerank windows ≈10k prefill × 600, and R2 is unspecified (it should be run). Assumes B-DIRECT is co-scheduled with G1 to share the skeleton prefix; otherwise +0.8 |
| B-RECALL memorisation probe (new, N20) | — | 0.1 / 0.15 / 0.25 | 600 × ~1k |
| G4 Qwen3 (300) | 1.2 | 1.4 / **2.0** / 3.0 | C-BEH is under R2, so the second model needs R2 too (doubles G1). Plus G0, staging and a separate start |
| G5 E4B (150) | 0.3 | 0.3 / 0.4 / 0.6 | The same R2 doubling, if it is to say anything about C-BEH |
| Session overhead | (in lines) | — | ≈9 manual L4 sessions × 0.25 ≈ 2.25 NH. Partly inside the lines; I add 0.3 for under-count |
| Reserve | 1.8 | 1.5 / 1.8 / 2.8 | — |
| **Core total** | **9.9** | **8.5 / 11.6 / 17.6** | — |

**GPU verdict.** It fits the 25 NH paper cap even at pessimistic. Against ≈120 quota-hours to Nov 5 shared with LoRA training (7–11 h per epoch, 12-h evals; HQ), it is ≈10–15%. **GPU is not the binding constraint.**

### 8.2 CPU (kernel-hours)

| Line | v2 central | Recount central (pess) | Why |
|---|---|---|---|
| Installs + rootfs | 36 | **70** (150) | HEAP needs installs or images for ~700 snapshots, with compiled repos (N24) |
| HEAP walks | 40 | 60 (120) | Django/sphinx bootstraps; per-snapshot walks |
| EVAL-exec | 30 | 35 (75) | Differential traces ×3 |
| AST + Jedi | 50 | 50 (100) | — |
| CAL traces | 18 | 18 (48) | — |
| Embeddings | 14 | 14 (40) | — |
| Renaming | 10 | 15 (30) | R2 over ~700 snapshots, consistent across files |
| Decoding + nested CV | 30 | **60** (120) | ≈12 arms × 2 views × grid (§3) |
| **Total** | **228** | **≈322 (683)** | At 5 concurrent kernels, ≈65 (≈137) wall-hours of babysat sessions |

### N27. Person-days: ≈26 central, not 15.5 (Fatal for the schedule; A20)

| Work | v2 central | Recount opt / central / pess | Why |
|---|---|---|---|
| Day-1 counts, Pathidea, rootfs probe | 1.0 | 1.0 / 1.5 / 2.0 | Docker Hub auth, selective extraction, prefix paths |
| EVAL builder over 4 sources | 1.5 | 2.0 / **2.75** / 3.5 | 4 schemas; dedup by PR and MinHash; checkouts of 25+ repos at ~700 commits; per-repo test-discovery configs (pytest.ini, setup.cfg, tox, pyproject, django runtests); scope-state mapping; vendored rule; G0 tree builder |
| Installs, HEAP, bootstraps | 2.0 | 3.0 / **4.0** / 5.5 | Precedent: `build_devset.py` took ~2 days for 3 repos and still yielded 42% (HQ). Here: 25+ repos, old Pythons, compiled exts, images |
| AST resolver, Jedi, graph | 1.5 | 2.0 / 2.5 / 3.0 | MRO, self/cls/super, aliases, local imports, scope states, site-local ∅ |
| Tracers, CAL traces, differential EVAL-exec traces | 2.5 | 3.0 / 3.5 / 4.5 | Two tracers, base-failure classification, 10 CAL repo envs |
| g_φ / h_ψ / f_θ + audits | 1.0 | 1.5 / **2.0** / 2.5 | + the 50-function fingerprint audit (~0.5) + the A7 polarity audit (~0.3) |
| Decoder + ~12 arms + Euler + nested CV | 2.0 | 2.5 / **3.0** / 4.0 | Profile HMM with inserts (and deletes, N8), orientations, π_∞, scope states, readouts, DPR rewiring, P1a/b/c projection |
| Prompts, vLLM, ~9 manual L4 sessions | 1.5 | 1.5 / 2.0 / 2.5 | — |
| Renaming R1/R2 + residual share | 1.0 | 1.5 / **2.0** / 3.0 | Consistent cross-module LibCST renaming at django scale (imports, `__all__`, string references), plus renamed graphs and embeddings |
| Embedding baselines | 0.5 | 0.5 / 0.5 / 0.8 | — |
| Stats, figures, kill reports | 1.0 | 1.0 / 1.25 / 1.5 | — |
| **Missing lines** | 0 | 1.0 / **1.75** / 2.5 | B-RERANK pipeline, Qwen staging, dataset release packaging, P1 hallucination profile, K3 manual calibration |
| **Pre-freeze total** | **15.5** | **≈20 / ≈26.75 / ≈35** | — |

**Calendar check.** Oct 9 → Nov 5 is 27 days at 5 h. 26.75 pd is ≈100% of all remaining days, with **zero** for the main track. The ladder's trigger ("projected > 16 pd") will fire in week 1.

**Floor recount.** Static EVAL builder 2.75 + AST-only graph 2.5 + AST q (no tracer, no CAL) 0.5 + decoder and core controls 2.5 + prompts and sessions 1.5 + R2 1.5 + stats 1.0 + EVAL-exec 60 (tracer) 1.5 ≈ **13.75 pd**, not 10.

---

## 9. Collapse table (§0.5): is every outcome publishable under the rubric?

Rubric: novelty incl. analysis, generality, relevance, verifiability, clarity.

| Row | Verdict | Problem |
|---|---|---|
| 1. All three pass | Publishable (technique, closeness 2, emission 1.5) | Joint probability is low: C-BEH ≈0.35–0.5 × C-USE ≈0.4, even if the effects are real (N22, N23) |
| 2. C-BEH ✓, C-DEC ✗ → BER | **Not tested as pre-registered.** C-BEH is measured on BMM, not on BER. BER (= X-BAG⊗PPR) has no one-change controls of its own, and as specified it is the mis-specified log-product scorer (N14) | Pre-register BER's own contrasts: BER − BER(β = 0) + BODY, and BER − BER-LEXEFF⁺. These cost CPU only |
| 3. C-DEC ✓, C-BEH ✗ | Weak: "Euler on LLM names", with no technique novelty claimed | Novelty rests on M1 analysis, closeness 2 (RepoReasoner, SWE-Flux). Workshop-grade |
| 4. Collapse → measurement paper | Weak-to-moderate. M1 on 120–150 tasks (60 if K1 fails) with one model largely replicates RepoReasoner's 0.80/0.31 | Strengthen it with the memorisation index (N20) and *typed* divergences by edge kind. That is analysis novelty |
| **Missing: C-DEC ✓, C-BEH ✓, C-USE ✗** | **The most likely outcome given the power.** "The technique beats its ablations but not asking Gemma directly". A reviewer reads it as negative on relevance | Pre-register this row: "a structured localiser that is more robust under R2 (C4) and in gold-not-in-G0 than direct prompting", using the gold-not-in-G0 and R2 strata |
| **Missing: K3 ambiguous (10–20%)** | C-BEH runs with a stated MDE that is underpowered | Pre-register that it is reported as an estimate and that the headline is row 3, decided before EVAL GPU |

**Verdict.** Rows 1–2 are publishable if the fixes land. Row 2 needs its own controls. Rows 3–4 are publishable only as analysis papers, and become clearly stronger with the memorisation index and gold-not-in-G0 analyses. Two likely outcomes are missing from the table.

---

## 10. Other breaks (short)

- **N25 (Minor-major).** R1 reuses G0 from the *original* issue ("paths are unrenamed"). But R1 renames the issue's code spans, so G0's input changes. Reusing the original G0 leaks unrenamed issue identifiers into the R1 condition. **Fix:** re-run G0 for R1 (+0.1 NH).
- **N26 (Minor).** Exception propagation (≤2 hops) gives entry and caller functions high q̃ for every exception raised within 2 hops. The effect channel is then flat along the true chain and cannot separate the culprit from its callers. It helps only across branches. **Fix:** state this in the method and evaluate effect gains by "culprit is the raiser" vs "culprit is a caller".
- **N28 (Minor).** The G0 prompt size for big repos (≈10–20k tokens) leaves room under 32k, but it must exclude tests by discovery config (L32 rule) consistently, or test files crowd the tree.
- **N29 (Minor).** v2 does not define scope-state emissions for effects (q̃ for a class body). If they are treated as background, N1's base-rate switch interacts with them. **Fix:** give scope states q̃ from their own AST (class-body raises are rare, so ≈0) and state it.
- **N30 (Minor).** DEV (118 competition tasks: fastapi, rich, requests, httpx) is "untuned in-domain". With 4 repos and heavy fastapi weight it is not a generality readout. Label it as such.

---

## 11. The five most damaging breaks, and the minimal final changes

| Rank | Break | Why it matters most | Minimal change |
|---|---|---|---|
| **1** | **N27. Person-days ≈26.75 central vs 15.5; the floor is ≈13.75, not 10** | The plan as written consumes every remaining day and leaves the main track nothing. The ladder fires in week 1 anyway, so pre-plan it | **Start at a "v2-lean" scope.** EVAL static (SWE-bench + rebench, Loc-Bench only if K0 is short). AST + Jedi graph. HEAP only where images exist. q from AST rules + g_φ only if K4 passes by Oct 17. R2 only (drop R1). Drop G5, and keep G4 at 150 with R2. EVAL-exec 60. Target ≈17–19 pd |
| **2** | **N14 + N15. X-BAG⊗PPR is a log-product, not a one-change bag; KL is ill-defined** | C-DEC can pass because of aggregation algebra, and the BER fallback rests on a mis-specified scorer | X-BAG⊗PPR := **the same HMM with A = 𝟙·PPR_πᵀ** (BMM at ρ = 1 with PPR restart), the same inserts/readout, no κ. Diagnostic = top-10 Jaccard ≥0.8 |
| **3** | **N22 + N21 + N6. Core claim underpowered; C-USE log-rank decided in the tail; K3 threshold inconsistent** | The "core claim never descoped" passes with probability ≈0.35–0.5 even if true. C-USE can "win" at ranks no agent reads | One control per claim; drop the MRR NI conditions (descriptive); log₂-rank **truncated at 100** as the single primary; B-DIRECT\* tail-filled by X-BAG⊗PPR; K3 pass ≥35% (on span-verified G1 output at K9); include the exception-line stratum in C-BEH |
| **4** | **N1 + N3 + N16 + N17 + N18. The signed-evidence algebra and its controls** | The base rate cancels (omnivores untouched). Execution labels learn intended raises, not bug raises. X-LEXEFF is a weak grep and is crippled under R2 on repo exceptions. X-NOEFF+BODY swaps channels. C-BEH can pass on static analysis and renaming asymmetry | Node-local contrast (or an all-bit μ′ penalty) instead of q̄; floor q̄. AST rules for implicit raises, g_φ for intended raises (or fault-injected labels). C-BEH = (BMM + BODY) − (β = 0). LEXEFF⁺ with MRO, propagation and the pre-rename exception identity. Split C-BEH-static vs C-BEH-learned |
| **5** | **N8 + N10 + N7 + N12. Decoder degeneracies renamed, not removed** | Inserts recreate the stutter and the entry readout. The teleport still reaches only retrieval favourites. h_ψ biases toward the happy path. Global ∅ is a callback hub. "Structure corrects wrong names" is again false by the model's own algebra in the hallucinated-middle-step regime | Add **delete states**; insert posterior gets no readout mass. Centre sim per step. Local (neighbourhood) uniform teleport. p(s\|u) = ½ uniform + ½ h_ψ. Site-local ∅_s. Gate P1a on executed CAL-tune, not EVAL-exec (N13) |

**Also cheap and worth doing:**
- **B-RECALL memorisation index** (N20, 0.15 NH), plus stripping doclines in R2.
- Dedup across sources by (repo, PR) and MinHash; explicit CAL-tune ∩ EVAL repo exclusion (N19).
- Re-run G0 for R1 if R1 survives (N25).
- Selective-prefix rootfs extraction with an authenticated pull (N24).
- Pre-register the two missing collapse rows and BER's own controls (§9).
- Define π_∞ as undirected PageRank, with λ = 0 by default (N11).

### Salvageable?

**Yes, conditionally.** The architecture's core (an imagined (symbol, effect) path decoded on a call graph against a mixture-bag control) is sound once N14 is fixed. The novel piece (polarity-aware effects) is testable once N1, N3, N16, N17 and N18 are fixed.

What is not salvageable as written is the **schedule** (N27) and the **claim that the core is powered** (N22). Both are fixed by cutting scope to v2-lean and simplifying the pass rules, not by more machinery.

**Expected value.** Even after all fixes, the most likely publishable outcome is row 3, or the missing "beats ablations, not direct prompting" row, with M1 and the memorisation index as the analysis contribution. Pre-register that framing now, so the paper is not rewritten after the data arrive.

No file other than this one was modified.
