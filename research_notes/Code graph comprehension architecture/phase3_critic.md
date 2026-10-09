# Phase 3 critic: six code-graph techniques (LATENT, HEAP, HYTRACE, DECONV, EFFECT, CHAIN), 9 Oct 2026

**Role.** I review as an adversarial area chair and ML-systems engineer, reading the material fresh. Inputs:
- `inventor_phase1_2.md` (Phases 1–2);
- the anomaly files `scout_graph_generation.md` (G-An) and `scout_embedding.md` (E-An);
- `docs/training_feasibility_4xl4.md`;
- the measured p0b row in `docs/experiments.md` and HQ (Oct 9).

**Labels.** VERIFIED means read in a primary source or in the repo. SNIPPET means search-summary only. INFERRED means my own reasoning. My search log is CR-n, at the end.

---

## 0. Ground rules I enforced, and two global defects

### 0.1 The unit of "GPU-h" is ambiguous, and the inventor mixes units
- In P1, the inventor writes "about 0.25 h on one 4×L4 node, so about 1–1.5 GPU-h". That converts node-hours to L4-hours (×4).
- The roll-up then adds node-hours (vLLM runs, LoRA epochs).
- Kaggle quota counts **session hours on the 4×L4 node**. **I count everything in node-hours (NH), and I take 25 GPU-h = 25 NH.**
- If the budget means L4-hours (25 L4-h = 6.25 NH), only the zero-shot HYTRACE kill plus the LATENT kill fit, and nothing that trains does. **The caller must settle this before any plan is approved.**

### 0.2 Throughput facts the inventor's numbers ignore (VERIFIED, p0b / HQ)

| Fact | Value |
|---|---|
| NF4 load from bf16 QAT | 705 s ≈ 12 min, per session or per process |
| Layer-split training throughput | 4k at 100 tok/s; **8k at 82 tok/s**; 16k OOMs on GPU3 |
| FSDP throughput | **unmeasured** (the stage crashed) |
| NF4-vs-W4A16 agreement | **unmeasured** (probe bug) |
| vLLM start | ~8 min (480 s) |
| Single-stream decode | ~34 tok/s |
| Host RAM | 188.7 GB, 48 CPUs |

**The inventor's "about 1.5k tok/s for HF forward passes" is unsupported.**
- A layer-split forward runs one GPU at a time. Training at 82 tok/s ≈ fwd + bwd + recompute ≈ 4× a forward pass, so layer-split forward-only is roughly **250–350 tok/s** at 8–12k (INFERRED).
- The only route to about 1–1.5k tok/s is **one NF4 replica per L4, data-parallel**:
  - 31B × 0.5 B ≈ 15.5 GB of NF4 weights, plus 2.8 GB bf16 embeddings, plus forward activations, gives about 19–21 GB of 24 GB at ≤8k (INFERRED). That is tight, and **unmeasured**.
  - bitsandbytes NF4 GEMMs are slow.
- **Every forward-pass budget below is given at 300 tok/s (pessimistic) and 1.2k tok/s (optimistic).** A 0.5 NH forward-only throughput probe is mandatory before LATENT or EFFECT spend anything.

### 0.3 Statistical power: every "+2–3 pp" kill threshold on the 118 dev tasks is inside the noise
- With n ≈ 93 (the no-name stratum) and about 20 discordant pairs, the SE of a paired hit@5 difference is about √20/93 ≈ **4.8 pp**.
- A 3 pp threshold therefore cannot be a kill criterion (INFERRED arithmetic).
- **Global fix.** Every downstream threshold becomes "paired-bootstrap 95% CI lower bound > 0, and point estimate ≥ X".
- Evaluation is extended to **≥300 public Python issue tasks on non-dev repos** (e.g. SWE-bench Lite/Verified minus fastapi, rich, requests and httpx). This is cheap for anything zero-shot or CPU.
- The 118 dev tasks stay as the in-domain readout.

### 0.4 Fixed candidate pools must not include gold by construction
- "Fixed pool of 50 candidates" appears in P2, P3, P5 and P6, and the pool construction is never specified.
- If gold is appended to a BM25 or graph pool, every arm is inflated, and the inflation is unequal: graph arms benefit from a graph-built pool.
- **Fix.** The primary metric is **rank of gold among all functions in the repo** (MRR, hit@k). Pools are reported only as secondary, built identically for all arms, with gold-in-pool rate reported.

### 0.5 Consistent alpha-renaming does not hurt BM25 the way P5 claims (logic error)
- If renaming is applied consistently to code *and* issue, exact-token BM25 is nearly invariant. Only English-word matches between issue prose and identifiers are lost.
- The claim "BM25 drops ≥10 pp while effect drops ≤2 pp" is therefore either false by construction or describes inconsistent renaming (code only).
- **Fix.** Define two renaming regimes:
  - **R1, consistent code+issue**: tests reliance on *semantic* name content. It hurts LLM readers, not BM25.
  - **R2, code-only with issue untouched**: the realistic "issue uses user vocabulary" case. It hurts BM25.

  Every technique reports both.

### 0.6 The quantisation transfer question applies to every probe on hidden states
- The served model is W4A16 (sym-int4, g32) in vLLM. Probes would be trained on NF4 states (HF path) or on bf16-dequantised W4A16 states.
- Only the second gives exactly the served weights, and it needs 62 GB of bf16 split across 4 GPUs (layer-split, about 250 tok/s, INFERRED).
- **Required check, about 0.2M tokens.** Train the probe on NF4 states and test it on dequantised-W4A16 states for the same windows. Report the AUROC drop.
- vLLM cannot easily export mid-layer states, so **no hidden-state probe is usable inside the competition scorer**. Hidden-state probes (LATENT, EFFECT heads) are paper artifacts, not inference tools for hidden repos. That is acceptable for the paper track, but it must be stated.

---

## 1. LATENT: execution-supervised latent graph readout

| Item | Value |
|---|---|
| GPU-h (NH), pessimistic 300 tok/s / optimistic 1.2k tok/s | Throughput probe 0.5 · kill 1.2M tok: 1.3 / 0.5 (incl. 12 min load) · full study +1.8M tok: 1.7 / 0.4 · R1/R2 renamed copies +0.5M: 0.5 / 0.15 · positive control +0.2M: 0.2 / 0.05 · NF4→W4A16 transfer check, 0.2M bf16 layer-split: 0.3 / 0.3 · **verbal baseline H3 (not counted by the inventor)**: vLLM with prefix caching over ~250 windows × 12k ≈ 3M prefill + start: 0.8 / 0.5 · two replication models: 1.0 / 0.6 · **dev forward "129 × 30k = 4M tok" (inventor: 3 GPU-h): 13.3 / 3.3, which must be cut**. **Total without dev forward: 6.3 / 3.0 NH. With dev forward: 19.6 / 6.3, which is infeasible beside anything else** |
| Falsifiability (0–3) | **1** as written; 3 after the fixes below |
| Surprise (0–5) | **4** (either outcome is informative: "the latent graph is static-lexical" explains G-A4 over-extension; "call-time edges are decodable but not verbalised" is new) |
| Prior-art risk | **Medium.** The method (edge probes with Hewitt–Liang and positional controls) is exactly 2605.25310 on tool-call graphs (CR-1, CR-6). Type probes on code models exist (TypeProbe 2607.08339, CR-3), as do AST/data-flow probes (Troshin & Chirkova; AST-Probe). No probe of code-LLM residuals for **repository runtime call edges** was found in CR-1/3/6 (weak absence). Closeness 2–2.5 |
| Generality | High for the claim (any model with open weights; any language with a tracer). Nil for scorer-time use (no mid-layer states from vLLM) |

**Logic and edge-case failures.**
1. **The "runtime-only" stratum is mis-defined. "Lexical baseline ≈ 0.5 by construction" is false.**
   - "Runtime-only = executed, neither lexical nor statically resolved" lumps two groups:
     - (a) edges whose callee *name appears* at the site but static resolution failed: `self.m()` via inheritance, `super().m()`, same-name methods in several classes;
     - (b) edges with **no name mention**: registry dispatch, `getattr(obj, name)`, callbacks, decorator wrappers, `__call__`, dunder protocol methods (`__eq__`, `__iter__`, `__enter__`).
   - The name-mention baseline is far above 0.5 on (a).
   - **Fix.** Use a strict hierarchy computed on CPU first: **lexical ⊂ static (Pyright/PyCG) ⊂ HEAP-resolvable ⊂ call-time-only**. Report AUROC per stratum. The headline stratum is *call-time-only, no name mention*. **Pre-gate:** ≥300 positive pairs across ≥3 test repos, or that stratum cannot carry a headline.
2. **Hard negatives from "static-only (resolved but never executed)" are mostly coverage holes, not non-edges.** At about 70% line coverage, many such edges are real but untested, so the probe learns "is this code tested".
   - **Fix.** Negatives are (site, candidate) pairs where the **call-site instruction executed ≥5 times and that candidate was never observed from it**: the same proxy HEAP uses.
   - Report the result also with random same-window negatives.
3. **Causal-attention order confound.**
   - Mean-pooled h_u can only encode v if v precedes u in the window.
   - Packing "in repo order" makes direction correlate with position, so an asymmetric bilinear probe can learn "earlier→later".
   - **Fix.**
     - Pack each window in two random orders and average.
     - Report AUROC split by (v before u) vs (v after u).
     - Add A17's **position-only baseline** (it reached 0.792 there, E-A17). The kill must be relative to it.
4. **The DPR-permuted-label control is the wrong one-change control for an interaction claim.** It catches node propensity but not "the probe adds h_u and h_v features".
   - **Fix.** Add an **additive probe** s = a·h_u + b·h_v (no interaction) as the one-change control against the bilinear probe. Keep the Hewitt–Liang control for selectivity.
5. **Co-presence and distance confound.** Labelled pairs are co-present by design, and true edges sit closer in the window (2-hop packing).
   - **Fix.** Add a token-distance-only baseline, and stratify AUROC by distance bins.
6. **Registries and decorators.**
   - For `@register("x")` / `REG[key]()`, the key string co-occurs lexically, so the probe may "read" it. That is legitimate inference, but it is not runtime-only in spirit.
   - HEAP resolves it without an LLM. **The comparison that matters is probe vs HEAP on the HEAP-unresolved stratum.**
7. **Monkeypatching in tests** (pytest `monkeypatch`, `mock.patch`): traced edges go to mocks.
   - **Fix.** Drop edges whose callee lives in test files or `unittest.mock`, and drop sites executed only under patched contexts.
8. **Flaky or weak tests.** Only edges observed in ≥2 runs are used. Seeded reruns are on CPU and free.
9. **Hidden repos.** Not applicable at inference (see 0.6).
10. **Huge repos.** Windows of ≤12k tokens cover 2-hop clusters only. Cross-window pairs are never labelled, so the claim is limited to within-window. State it.
11. **Non-Python.** A JS replication needs a tracer and a static baseline (Jelly). That is realistic CPU work.
12. **Quantisation.** See 0.6. The transfer check is mandatory.
13. **Verbal baseline H3** must use ≥2 prompts and logprob scoring (not sampled yes/no), otherwise "probe > verbal" is a prompt artifact.

**Verdict: SURVIVE-BY-SURPRISE, narrowed.** The question is new and either answer is publishable, but the kill test as written cannot separate runtime knowledge from name, position and coverage artifacts.

**Required fixes.**
- The stratum hierarchy and its pre-gate (1).
- Executed-site negatives (2).
- Order randomisation plus the position-only and distance-only baselines (3, 5).
- The additive-probe control (4).
- Drop the dev forward, or restrict it to ≤0.5M tokens of candidate neighbourhoods.
- Measure throughput first.
- Thresholds: call-time-only AUROC ≥0.70 **and** ≥ position-only + 0.08 **and** ≥ additive + 0.03, each with a CI over ≥3 repo-split resamples.
- **Positive control:** "A calls B" text ≥0.95 (kept).
- **Negative positive control:** a probe on a random-init model of the same architecture must fail on the call-time stratum.

---

## 2. HEAP: an import-time reflective graph builder

| Item | Value |
|---|---|
| GPU-h | **0** for construction and the CPU evaluation. A downstream LLM arm, if any, reuses HYTRACE lines |
| Falsifiability | **2**: thresholds exist, but the baseline is under-specified, and so is the denominator |
| Surprise | **2** (an engineering decomposition; the interesting number is "share of runtime edges resolvable at import time vs only at call time") |
| Prior-art risk | **Medium.** PoTo (ECOOP 2025) already runs **concrete evaluation in the interpreter** inside a Python points-to call-graph builder and beats PyCG on call graphs (CR-4). An LLM-hallucination detector builds its KB "via library introspection" (2601.19106, CR-8). Jedi and griffe introspect imported ("compiled") objects (INFERRED from tool knowledge, not searched). TamiFlex covers Java. SemNav's lazy Pyright resolution (G-A6) is the strong static competitor. No full **import-heap call-graph builder evaluated against traced edges** was found |
| Generality | Reflective dynamic runtimes (Python, Ruby, JS with care). Not needed in statically typed languages, where static resolution is already strong. That is honest scope, not a flaw |

**Logic and edge-case failures.**
1. **Strawman baseline.**
   - "Static" in the kill test is an in-house tree-sitter/AST resolver plus PyCG "where it terminates".
   - MRO and `super()` resolution are *statically* computable, and Pyright and Jedi do it. The predicted gains in the `super`/MRO category may only measure the weak baseline.
   - **Fix.** Baselines are **Pyright go-to-definition per site** (as SemNav does), PyCG, and PoTo where it runs. HEAP's increment is measured over the **union** of static tools.
2. **Wrong denominator.**
   - 59% of PyCG's missing callees are **built-in-type methods** (G-A1). These are C callees such as `str.strip`, with no comprehension value for localisation, and HEAP only resolves them when types are literal or annotated, as static tools can.
   - **Fix.** Use a **repo-internal callee** denominator for the headline, and report built-ins separately.
3. **Import side effects** (untrusted code). Importing runs module top-level code: Django needs `settings` configured; CLI modules parse `sys.argv`; some modules open sockets, read env vars or spawn threads.
   - In the competition sandbox this is no worse than running tests, which agents already do.
   - For a general tool it is a security property: run it in a container with network off and `HOME` and `tmp` isolated, and say so.
4. **Missing deps and C extensions.** Imports fail, which leaves partial heaps.
   - **Fix.** The import-failure rate per repo is a reported metric (the inventor has this).
   - Add a **fallback stub importer** (a `sys.meta_path` finder that returns `MagicMock` modules for missing third-party deps). Measure the precision cost, because mocks poison isinstance checks and registries.
5. **Lazy imports.** Module `__getattr__`, `importlib.util.LazyLoader` and `lazy_loader` (scientific stack) leave registries unpopulated until attribute access. Forcing access triggers more side effects.
   - Plugin registries filled via **entry points** (`importlib.metadata.entry_points`, pluggy) or on first use are invisible at import.
   - **Report a "filled-at-first-use" category** from the trace diff.
6. **Metaprogramming.** `dataclass`, `attrs`, `namedtuple`, `Enum` and `exec` generate functions with `co_filename` `<string>` and no source span. They need mapping to the defining class, or they become orphan nodes.
   - Metaclasses with `__call__` and `__init_subclass__` are handled only if the walker records them as edges, not only as facts.
7. **Monkeypatching at import** (e.g. `gevent.monkey`, `six.moves`) rewrites stdlib objects, so canonical names lie. Detect them by comparing `id()` against a clean interpreter.
8. **Hidden repos.** HEAP works only if the environment is installed, and it is in SWE images. **It is the only technique here that runs on hidden repos without an LLM and without tests.** That is its main value.
9. **Huge repos.** Importing all modules is O(seconds–minutes); memory is fine (188 GB). Importing *test* modules must be excluded or isolated (fixtures, `conftest` side effects).
10. **New-code fixes.** Not relevant to the builder.
11. **Precision proxy.** The site must be the instruction offset (`sys.monitoring` gives it), not the line. Several calls per line otherwise create fake false positives.

**Verdict: SURVIVE (narrowed).** It is cheap, falsifiable, the only inference-time builder that needs no execution of tests, and it is the necessary substrate for LATENT's strata and HYTRACE's graph. Novelty alone is thin: the paper value is the **import-time-vs-call-time decomposition** across about 12 repos against strong static baselines.

**Required fixes.**
- Pyright/PyCG/PoTo union as the baseline.
- A repo-internal denominator.
- Pre-registered categories: MRO/super, re-export, decorator-wrap, registry, metaclass-generated, filled-at-first-use, call-time-only.
- **Kill** if HEAP recovers <15% of repo-internal edges missed by the static *union*, or edge precision drops >3 pp, with a CI over repos.
- Downstream: use full-repo ranks (0.4), and add a DPR arm preserving in/out degree and module.

---

## 3. HYTRACE: issue → imagined runtime path, decoded onto the graph with an HMM

| Item | Value |
|---|---|
| GPU-h (NH) | **Zero-shot (the headline):** vLLM start 0.13 + (118 dev + ≥300 public tasks) × ~7k prefill ≈ 2.9M tokens + k=5 sampled paths per task (~0.3M decode tokens, batched) + R1/R2 renamed arms ≈ **1.5–2.0 NH**. **LoRA arm (optional):** pseudo-issue generation, 1,200 items on vLLM, ≈0.5 (not counted by the inventor); training at 7.8M tok/epoch is **26 NH per seed at the measured 82 tok/s layer-split rate**, or 3.3–5.4 NH at the unmeasured FSDP 400–650; 3 seeds plus 12-min loads plus export and eval ≈ 12–18 NH. **Infeasible as specified.** Cut to ~600 items × 3k-token skeleton ≈ 1.8M tok/seed, which gives 3 seeds ≈ 6–7 NH, and only if FSDP ≥400 tok/s |
| Falsifiability | **1** as written (no anchor-graph control, underpowered); 3 after the fixes |
| Surprise | **4** ("structure corrects names: imagined paths help even when ≥50% of steps are wrong"; map-matching for hallucinated code paths) |
| Prior-art risk | **Low–medium.** Executed-trace localisation (IssueExec, LLM4FL, OrcaLoca) and LLM search over call graphs (CoSIL 2503.22424) are adjacent (CR-2). HyDE-style query rewriting exists. **HMM map-matching** (Newson & Krumm 2009, GPS) is the obvious methodological ancestor, and must be cited. No HMM/Viterbi decoding of LLM-imagined symbol paths was found (CR-2, CR-7, CR-9) |
| Generality | High. The decoder is model- and language-agnostic, and a frontier model can be swapped in. No execution at inference. The graph needs only static plus HEAP |

**Logic and edge-case failures.**
1. **Missing the critical one-change control.**
   - Decoded vs names-only changes the *graph*, the *path prior* and the *aggregation* at once.
   - The entry step is usually the issue-named public API, so the HMM may reduce to "graph neighbourhood of the issue anchor, reranked by name similarity". That is known: RepoGraph/LocAgent-style expansion, and personalised PageRank.
   - **Fix: add arms.**
     - **PPR-from-anchors × BM25 emission (no imagined path)**: the main control.
     - **HMM with step order shuffled**: does sequence matter?
     - **HMM with only the first and last imagined steps**: does the middle of the path carry information?
     - DPR (kept, with in/out degree and module preserved).
   - The headline claim is decoded > PPR-anchor, not decoded > names-only.
2. **Power.** See 0.3. 93 no-name dev tasks cannot resolve 3 pp. Use ≥300 public tasks, with CI-based kills.
3. **Pool.** See 0.4.
4. **New-code fixes.** Where the fix adds a missing call or branch, the culprit may not be on the *executed* path at all: the bug is that a function is *not* called. The HMM, constrained to edges, then misses it by design.
   - **Report a stratum:** gold on the traced F2P path vs off it. That fraction (SNIPPET-adjacent: in SBEST 78% of buggy methods were within 0.34 calls of the stack trace, CR-5) bounds the method.
5. **Decorators, `__getattr__` and generators** truncate paths (G-A4). The HMM needs HEAP `wraps`/`dispatch` edges, so a skip-transition ε is essential and must be tuned on non-dev repos only.
6. **Training-data leaks (LoRA arm).**
   - The pseudo-issue is generated from failure output, which contains **test names** (`test_parse_header_empty` → culprit `parse_header`).
   - The leak checker must reject culprit-name tokens and **any token of the failing test's name that matches a skeleton symbol on the true path**.
   - The skeleton has no bodies, so the mutation itself cannot leak (good).
7. **Huge repos.** The skeleton is ≤6k tokens. Truncation makes hallucinated names more frequent, which is exactly where the decoder should help. Report gain vs skeleton coverage.
8. **Hidden repos.** Fully compatible: no execution is needed, given static+HEAP.
9. **Non-Python.** Compatible given a directed graph.
10. **Seeds.** The zero-shot headline needs no training seeds, but **sampling seeds** (k paths, 3 sampling seeds) are the unit for variance.
11. **Positive control** (true traced path with names noised at 50% → culprit ≥80%) is good. Add a **negative control**: a random path of matched length decoded on the true graph must not beat PPR-anchor.

**Verdict: SURVIVE-BY-SURPRISE (zero-shot only).** The LoRA arm is NARROW and gated: it runs only if FSDP is measured at ≥400 tok/s and zero-shot passes.

**Required fixes.**
- PPR-anchor, shuffled-order and random-path controls.
- ≥300 public tasks with paired-bootstrap CIs.
- Full-repo ranks.
- An on-path/off-path stratum.
- **Kill** if decoded − PPR-anchor has a CI lower bound ≤0 on the no-name stratum, or decoded ≈ DPR, or shuffled ≈ ordered (sequence then carries nothing, and the method is just a bag-of-names anchor expansion).

---

## 4. DECONV: inverting graph smoothing in the supplied embeddings

| Item | Value |
|---|---|
| GPU-h | **0** (CPU sparse solves; the small text encoder runs on CPU) |
| Falsifiability | **2.** The DPR inversion is sharp, but the threshold is underpowered (65 multi-function tasks; an MRR SE of about 0.04 against a 0.03 threshold). Ill-posedness is not addressed |
| Surprise | **2** |
| Prior-art risk | **High for the method.** Blind identification and deconvolution of graph filters is established: Segarra et al., IEEE TSP 2017 (CR-10), robustness to graph perturbation (IEEE SPL 2025), exact recovery (Signal Process. 2025) (CR-11); GDN (NeurIPS 2021). Only the *application* (un-smoothing a shipped code-embedding release, plus an NL bridge) is new |
| Generality | Low. The object is one competition's embeddings; the "GNN-smoothed releases" audience is small |

**Logic and edge-case failures.**
1. **The generator model is probably wrong.**
   - Linear X = g(Â)H holds only if the release is a linear propagation of fixed features.
   - If it came from a trained GNN with nonlinearities and weights, or from an LLM featuriser over graph walks (E-A18's previous-token mixing), linear inversion recovers an arbitrary linear image. CKA with TF-IDF can then be fitted by λ alone.
   - **Fix.** First test linearity: is the residual of X ≈ Σ_k c_k Â^k X̃ small on held-out nodes?
2. **The supplied Â may not be the generating graph.** The supplied graph is undirected with mixed labels (HQ). Inverting with a near-miss graph behaves like the DPR arm.
3. **Ill-posed.** With low-pass eigenvalues near 0, Tikhonov restores only mid frequencies. "Ĥ(DPR) < X" is nearly guaranteed for *any* wrong-graph inversion, so it is **not a sharp signature of the mechanism**. The informative comparison is against generic high-pass baselines: X − ÂX, Soft-ZCA, and centring.
4. **Execution grounding is weak.** Labels are gold edits; the inventor admits this. That fails the brief's "execution as ground truth".
5. **Competing explanation 3b** (co-edited functions are graph siblings) predicts that un-smoothing *hurts* co-edit retrieval. The test has an ambiguous sign, so both outcomes can be narrated.
6. Hidden repos: fine (CPU). Non-Python: as general as the release. Quantisation: not applicable.

**Verdict: KILL as a technique or headline. Keep it as a 1-day forensic control.** The method is prior art, execution plays no role, and the audience is narrow. The **useful residue** is a hygiene step (identify the smoothing and report Soft-ZCA/high-pass as baselines) inside whichever paper uses the supplied embeddings.

---

## 5. EFFECT: an execution-labelled effect parser with directed propagation

| Item | Value |
|---|---|
| GPU-h (NH) | Issue parsing ≈ 0.3 (vLLM start + ~420 short prompts). Effect heads cost +0 **only if LATENT's forward runs**; standalone they need their own forward over train repos (≈1–4 NH, see 0.2). Coverage kill: CPU + 0.3 |
| Falsifiability | **1** as written (no static-effect control; an inference-time label leak; wrong renaming logic); 3 after the fixes |
| Surprise | **3** (a shared, name-free behaviour vocabulary between issues and code; the coverage statistic is new) |
| Prior-art risk | **Medium-low.** Exception-driven localisation (EXPECT, ASE 2024), stack-trace plus coverage localisation (SBEST), and execution-feature diagnosis (2502.18664) are close (CR-5). Exception prediction exists (ThrowBench, CWM). No issue-effect ↔ function-effect-signature retrieval was found. A tool-effect evidence-graph paper exists for agents ("Beyond Tool-Call Success", ICLR 2027 submission, CR-6), but not for code localisation |
| Generality | Medium-high. Exceptions, mutation and I/O exist in every language. Name-free by construction |

**Logic and edge-case failures.**
1. **Inference-honest labels.**
   - The issue describes the **buggy behaviour under the issue's scenario**. Effects traced under the existing, passing suite describe *normal* behaviour. Tracing F2P tests at evaluation time is a **leak**, because hidden repos have no F2P tests.
   - **Fix.**
     - **Coverage** uses only base-commit *pre-existing* tests (inference-available) or the probe or static predictor.
     - The F2P trace is used **only** as the oracle positive control.
2. **Missing one-change control: static/syntactic effects.**
   - A cheap AST extractor gets most of the 16 bits without any execution: `raise X` / re-raise, `self.attr =`, `arg.append` / `arg[k] =`, bare `return` / `return None`, `logging.` / `warnings.warn` / `open(`.
   - BM25 also already sees `raise KeyError` tokens.
   - **Fix.** Add the arm **AST-effect (no execution)**. The execution-labelled claim lives only in the difference between traced or probe effects and AST effects.
3. **Renaming logic error.** See 0.5. Use R1/R2.
4. **Low resolution.** A 16-bit signature over about 4k nodes gives huge tie classes, so "effect alone" MRR is meaningless. Report tie-aware ranks, and fuse only by RRF with an explicit tie-break.
5. **Propagation over-approximates.** "Union of callee escapes minus caught types" ignores conditional raises and broad `except Exception`, and it over-propagates along chains. That is the same 0.99 → 0.19 compounding (G-A3).
   - **Fix.** Propagate along traced or HEAP edges only, with depth ≤2.
   - DPR propagation remains the control.
6. **Effects driven by mocks or monkeypatching in tests** (I/O mocked away). The exclusions are as for LATENT item 7.
7. **Hidden repos.** Static and probe effects work; traced effects need the suite run. Disclose it.
8. **Quantisation.** As 0.6, for the probe heads.

**Verdict: NARROW. It survives as a component (an HMM emission feature in HYTRACE, and a LATENT head), not as a standalone headline.**

**Required fixes.**
- The AST-effect control.
- Inference-honest coverage.
- R1/R2 renaming.
- **Kill** if coverage <25% with inference-honest labels, or traced/probe effects ≤ AST effects (CI) on fused MRR.

---

## 6. CHAIN: call-site probability distributions in place of binary edges

| Item | Value |
|---|---|
| GPU-h | **0** |
| Falsifiability | **2.** The one-change control (per-edge BCE, same features) is good. The "chain" definition is unspecified, and so is the coverage bias |
| Surprise | **2** |
| Prior-art risk | **High.** GRAPHIA ("Call Me Maybe", JS, 2506.18191) already predicts **per-call-site probability distributions over candidate targets** and ranks them (CR-12). Also: AutoPruner, CGPruner, Mir et al. (per-edge); Jikes RVM `WeightedCallTargets`; the Oracle patent on probabilistic call-graph construction; Jarvis's chain precision (G-A3). The remaining novelty is "site-normalised vs per-edge on *chain* precision" plus "mass-guided LLM context expansion" |
| Generality | High (any language with a tracer) |

**Logic and edge-case failures.**
1. **Frequency is not validity.**
   - The cross-entropy target is the per-site callee *frequency* in the test suite. That is driven by test loops and fixtures, not by which callees are possible.
   - Megamorphic sites (visitors, registries, plugin dispatch) get split mass, so true paths through them get tiny product mass. That penalises precisely the dynamic-dispatch paths where bugs hide.
   - **Fix.** Train on **set membership** (observed ≥1 time, normalised within the site), not raw frequency, and compare both.
2. **"Chain" is undefined.**
   - **Fix.** Pre-register: a chain is a simple path of ≤L hops from a test-entry function; it is valid iff it occurs as a contiguous subsequence of some traced call stack.
   - Untested-but-real chains count as FP equally for all arms. State the bias.
3. **The per-edge baseline can learn site normalisation** from the fan-out feature |C(s)|. The predicted gap may vanish for a trivial reason.
   - **Fix.** Add a third arm: per-edge BCE plus post-hoc within-site renormalisation. Only the joint softmax vs that arm isolates "learned site-correlation".
4. **"2-hop > 1-hop with mass" downstream** is underpowered on dev (see 0.3).
5. Hidden repos: fine (CPU, no tests at inference). Monkeypatching and flaky tests: as for LATENT.

**Verdict: KILL as a standalone paper (crowded, low surprise). NARROW into a component:** the HMM transition model for HYTRACE (ablation: uniform vs CHAIN transitions), and dispatch-mass splits in HEAP. The chain-precision decomposition can be a section if cheap.

---

## 7. Summary table

| Technique | NH (pess / opt) | Falsif. as written → fixed | Surprise | Prior-art risk | Generality | Verdict |
|---|---|---|---|---|---|---|
| LATENT | 6.3 / 3.0 (dev forward cut) | 1 → 3 | **4** | Medium (2605.25310 method; TypeProbe) | High (claim), nil at scorer time | **SURVIVE-BY-SURPRISE (narrowed)** |
| HEAP | 0 | 2 → 3 | 2 | Medium (PoTo hybrid concrete eval; introspection KBs) | Dynamic languages | **SURVIVE (narrowed)** |
| HYTRACE | 1.5–2 zero-shot; +6–7 LoRA (gated) | 1 → 3 | **4** | Low–medium (IssueExec, CoSIL, HyDE; HMM map-matching ancestor) | High | **SURVIVE-BY-SURPRISE (zero-shot)** |
| DECONV | 0 | 2 | 2 | High (blind graph deconvolution) | Low | **KILL** (keep as a control) |
| EFFECT | 0.3 (+forward if standalone) | 1 → 3 | 3 | Medium-low | Medium-high | **NARROW** (component) |
| CHAIN | 0 | 2 → 3 | 2 | High (GRAPHIA per-site distributions) | High | **KILL standalone → NARROW component** |

**Top 2 by surprise: LATENT (4) and HYTRACE (4).** They survive one extra round. What must change:
- **LATENT:**
  - the stratum hierarchy lexical ⊂ static ⊂ HEAP ⊂ call-time, with a ≥300-pair pre-gate;
  - executed-site negatives;
  - order randomisation with position-only and distance-only baselines;
  - the additive-probe control;
  - the NF4 → W4A16 transfer check;
  - throughput measured first;
  - the dev forward dropped.
- **HYTRACE:**
  - PPR-from-anchor as the main control, plus shuffled-order and random-path controls;
  - ≥300 public non-dev tasks with paired-bootstrap CIs;
  - full-repo ranks;
  - an on-path/off-path stratum;
  - LoRA gated on measured FSDP ≥400 tok/s, cut to about 1.8M tok/seed × 3 seeds.

---

## 8. A coherent combination: "graph as decoder" (HEAP → CHAIN-lite → HYTRACE, with EFFECT emissions), plus LATENT as its mechanism study

**Architecture (one paper, one claim: *execution-calibrated structure is most useful as a decoder that corrects an LLM's imagined execution, not as context*).**
1. **Generation (CPU, no tests at inference).** Static union (Pyright/PyCG), plus HEAP edges (`wraps`, `dispatch`, `override`, `reexport`), giving a directed typed graph.
2. **Graded transitions (CPU).** CHAIN-lite gives p(callee | site), trained on traced *membership* from non-dev repos. Ablation: uniform transitions.
3. **Parsing (one vLLM pass).** Gemma imagines `entry → … → culprit` from issue plus skeleton.
4. **Decoding (CPU).**
   - HMM forward–backward over the graph.
   - Emissions = name/docstring similarity **⊕ EFFECT match** (the issue-effect query against AST effects, or probe effects where available).
   - Node posteriors are the ranking.
5. **Mechanism study (LATENT).**
   - On the same strata (lexical / static / HEAP / call-time-only), ask whether Gemma's residual stream encodes the edges that its imagined paths get wrong.
   - Prediction linking the two halves: imagined-path step errors concentrate on strata where probe AUROC ≈ position baseline.

**Budget (NH):**

| Line | NH |
|---|---|
| Throughput probe | 0.5 |
| LATENT | 3–6.3 |
| HYTRACE zero-shot, all arms (incl. EFFECT and CHAIN ablations, which are CPU-side) | ≈2 |
| Optional HYTRACE LoRA, 3 seeds | 6–7 |
| **Total** | **≈ 5.5–9 NH without LoRA; ≈ 12–16 NH with it** |

That fits 25 NH with slack for one unplanned rerun. It does **not** fit if 25 means L4-hours: then use HYTRACE zero-shot plus the LATENT kill only.

**Why this combination and not others.**
- HEAP and CHAIN are weak alone (low surprise, crowded) but supply exactly what the HMM needs: a directed graph with calibrated transitions.
- EFFECT is weak alone (ties, coverage) but is a name-free emission, so it is what makes the decoder survive R1/R2 renaming.
- LATENT explains *why* the generator needs a decoder.
- Each component has its own one-change ablation inside the same evaluation harness. Generality comes from the decoder being model- and language-agnostic.

**Not combined:** DECONV, which serves only as a hygiene baseline if the supplied embeddings are used.

---

## Search log (critic)

| ID | Where | Query | Hit relevant to the verdict |
|---|---|---|---|
| CR-1 | WebSearch | probing LLM hidden states call graph repository caller callee linear probe 2026 | 2605.25310 (tool-call graph probes; Hewitt–Liang + positional controls; "tracks topology not identifiers"); 2601.17593; 2606.17648 (unread). No repo call-graph probe. SNIPPET |
| CR-2 | WebSearch | LLM hypothesized execution path issue localization call graph Viterbi | IssueExec 2607.17286, CoSIL 2503.22424, LLM4FL 2409.13642, BluesFL 2605.17290. No HMM/Viterbi decoding. SNIPPET |
| CR-3 | alphaXiv discover | probing code models execution / call relations | TypeProbe 2607.08339 (type representations probed from code-model residuals); Latent Programming Horizons 2607.05188; neural debugger 2603.09951. No call-graph probe. SNIPPET |
| CR-4 | WebSearch | PoTo hybrid Andersen points-to Python ECOOP 2025 | **PoTo: static + concrete interpreter evaluation; PoToCG beats PyCG.** HEAP-adjacent mechanism. SNIPPET |
| CR-5 | WebSearch | side-effect signature exception behaviour bug report localisation | SBEST 2405.00565, EXPECT (ASE'24), 2502.18664 execution features. No effect-signature ↔ issue retrieval. SNIPPET |
| CR-6 | OpenReview API (6 terms, 3 s pacing) | call graph probe; runtime call graph LLM; probabilistic call graph; dynamic call graph Python; imagined execution localization; graph deconvolution embeddings | 2605.25310 also an ARR Aug 2026 submission; "Beyond Tool-Call Success: Dynamic Evidence Graphs for Runtime Effect Certification" (ICLR 2027 submission; agent tools, not code); PoTo; Scalpel; DyPyBench; blind graph deconvolution (SPL 2025, Signal Process. 2025). Nothing on imagined-path decoding |
| CR-7 | WebSearch | LLM hallucinated API names map matching HMM code graph | StepwiseGraphFix (SANER'26); **2601.19106 KB via library introspection** (HEAP-adjacent); MARIN. No HMM. SNIPPET |
| CR-8 | (same as CR-7) | — | 2601.19106 logged as HEAP prior-art risk |
| CR-9 | OpenReview API | hidden markov model code localization; effect signature bug localization | Nothing relevant |
| CR-10 | WebSearch | blind identification of graph filters deconvolution Segarra | Segarra, Mateos, Marques, Ribeiro, IEEE TSP 2017 (1604.07234). **DECONV method is prior art.** SNIPPET |
| CR-11 | OpenReview API (in CR-6) | graph deconvolution embeddings | Blind deconvolution of graph signals, robustness (SPL 2025); exact and stable recovery (Signal Process. 2025) |
| CR-12 | WebSearch | call site target distribution learned probabilities call graph | **GRAPHIA, 2506.18191: per-call-site probability over candidate targets (JS, GNN).** Oracle US10719314; Jikes WeightedCallTargets. CHAIN prior-art risk high. SNIPPET |

**Not searched; the red team should check:**
- Jedi and griffe dynamic-inspection internals, as HEAP prior art;
- Jin & Rinard (ICML 2024) emergent program-state probes, as LATENT framing;
- Newson & Krumm (2009) HMM map-matching, for HYTRACE's citation;
- the full text of PoTo's concrete-evaluation scope (whether it resolves user-code registries or only library calls).

No files other than this one were modified.
