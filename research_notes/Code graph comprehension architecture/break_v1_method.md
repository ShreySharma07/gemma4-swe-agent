# Break of architecture v1 (BMM): methodology, statistics, compute, feasibility. Breaker 2, 9 Oct 2026

**Target:** `architecture_v1.md` (read in full).
**Environment facts used:** `docs/training_feasibility_4xl4.md`, `docs/HQ.md`, the p0b row in `docs/experiments.md`.

**Labels.**
- **VERIFIED**: in our repo docs.
- **KNOWN**: public facts about SWE-bench, uv and Kaggle that I am confident of but did not re-check today.
- **EST**: my estimate.

Nothing was run.

**Severity scale.**
- **S1**: kills the plan or its headline as written.
- **S2**: a reviewer-visible flaw, or a budget or time overrun of 1.5× or more.
- **S3**: fixable detail.

---

## 0. Context the architecture ignores (S1, cross-cutting)

- **Calendar.** Today is Oct 9. The experiment freeze is **Nov 5**, 27 days away. The paper is due Nov 12. Shrey works about 5 h a day, solo, and the main track (Dec 2) is still the primary goal (HQ).
- **GPU.** The quota is 30 h a week (VERIFIED), so about 105–120 h remain before the freeze. The main track needs this same quota:
  - LoRA training is 7–11 h per epoch even at FSDP speeds that are still unmeasured. At the measured layer-split speed it is ~27 h per 1k×8k trajectories (HQ, p0b).
  - Every scorer-faithful local eval is up to 12 h.
  - The 25 NH paper cap is therefore ≈20–25% of all remaining GPU time, and it competes with the thing we are trying to win.
- **Human time.** I size the v1 engineering below. It is **≈32–45 person-days at 5 h/day**:

| Work item | Days (EST) |
|---|---|
| No-Docker SWE-bench + SWE-rebench env harness for ~90 (repo, version) pairs + rebench per-task envs, 2×2 validation | 6–9 (precedent: `build_devset.py` took v1→v5 over ~2 days for 3 repos and still got 54/128) |
| Tracer (calls, sites, exceptions, return class, fingerprints, globals, I/O), both 3.12 `sys.monitoring` and <3.12 fallback | 3–4 |
| Static union: Pyright per-site go-to-def via LSP, PyCG, PoTo, all merged | 3–5 (PoTo is a research artefact; PyCG fails on large repos) |
| HEAP sandboxed import + heap walk | 2–3 |
| CHAIN-lite (LightGBM) + g_φ + calibration + 3 seeds | 2 |
| HMM decoder + 14 X/E/T arms + TPE tuning for every arm | 3–4 |
| LibCST R1/R2 renaming over repo + skeleton + issue | 2–3 |
| Baselines: CoSIL port to vLLM, SRE-S/L, CRE, B-CTX | 3–4 |
| Study B (window packing, NF4 extraction, probes, 6 controls, transfer check) | 5–7 |
| Writing + figures | 3–4 |
| **Total** | **32–45** |

That is 1.2–1.7× the calendar **if the main track got zero hours**.

**Fix:** cut the architecture to a floor that fits about 15 person-days (§7).

---

## 1. Data feasibility on Kaggle CPU without Docker

### 1.1 What our own evidence says (VERIFIED, HQ)

| Env | Gold pass |
|---|---|
| Host-package env (competitor) | 109/129 = 85% |
| Scorer-like env, highest wheel versions (ours, v5) | **54/128 = 42%** |

- By repo: rich 39/48, fastapi 10/67, requests 5/13.
- The failures were pins (starlette 1.6.0), missing test plugins (pytest-httpbin, inline_snapshot) and snapshot tests (pygments).
- Neither number is a version-pinned per-commit env, which is what v1 proposes. The right prior lies between them.
- The *class* of failure (test plugins, transitive pins, fixtures that need network) recurs in every repo.

### 1.2 SWE-bench specifics that break "uv per repo version" (KNOWN; S1 for part of the pool)

**Candidate pool.**
- Verified (500) ∪ Lite (300) is ≈700 unique tasks; minus psf/requests that leaves ≈685.
- **Django ≈45% and sympy ≈15% of Verified.**
- The other ≈40% is spread over astropy, matplotlib, scikit-learn, xarray, sphinx, pytest, pylint, seaborn and flask.

**Compiled repos (astropy, matplotlib, scikit-learn), ≈20% of the pool.**
- The harness specs build them in conda, with C/Cython extensions **rebuilt per commit** (5–20 min on 4 cores).
- They pin old numpy and scipy, and matplotlib needs freetype/qhull.
- Without conda and Docker the yield is poor (EST 20–45%).

**xarray.**
- It is a conda `environment.yml` with netCDF/HDF5 stacks. EST yield 30–50%.

**Old Pythons.**
- Django 3.0–3.2 specs use Python 3.6.
- uv's managed CPython builds do not go that low (verify; I believe 3.7/3.8+).
- Bumping the interpreter changes deprecation warnings, `DeprecationWarning`-as-error settings and ordering, and some F2P flips.

**sphinx (≈8%).**
- Old versions need old Jinja2/docutils/markupsafe pins that modern resolvers reject. EST 40–60%.

**Pure-Python Django, sympy, pytest and pylint.**
- These are the good case: EST 70–85%.

**SWE-rebench (≈500 needed to reach 1,200).**
- The install recipes were validated inside their Debian images, often with apt packages.
- The Kaggle CPU image has many system libraries, so the yield is plausibly 50–70%.
- But **each task needs its own env** (different repos and commits), so env cost is per task, not per version.

**sys.monitoring.**
- It needs Python ≥3.12.
- **Almost no SWE-bench EVAL env is ≥3.12.** EVAL tracing therefore runs on the `sys.setprofile` fallback, which brings the bug in §2.1.

### 1.3 Yield funnel (EST)

| Stage | Opt | Central | Pess |
|---|---|---|---|
| Candidates | 1,200 | 1,200 | 1,200 |
| Env installs and F2P fail→pass, 2 of 2 | 65% | 50% | 35% |
| Traceback-free by the strict regex | 70% | 65% | 60% |
| Gold in an *existing* function (not new-code) | 90% | 88% | 85% |
| **Headline survivors** | **491** | **343** | **214** |
| …of which Django + sympy | ~55% | ~60% | ~70% |

- **≥300 has roughly 55–60% probability** (EST).
- The survivors concentrate in 2 repos, which hurts generality and inflates clustered variance (§4.3).

### 1.4 Time, CPU, disk, network (EST)

**Kaggle CPU kernel.** 4 vCPU, ~30 GB RAM, 12 h per session, 20 GB `/kaggle/working` output. Concurrency of 5 is A1 and unverified.

**Per-task wall time on one kernel:**
- checkout + editable install: 0.5–2 min pure Python, 5–20 min compiled;
- 4 validation runs at 0.5–2 min each (Django `runtests.py` start-up ≈5–10 s; sympy is slow);
- 1 traced F2P run at 1–5 min (§2);
- static union + HEAP: 5–30 min (see C3);
- **failures cost the most**, because they run until timeouts.

**Central ≈12 min per candidate, against v1's 10 CPU-min (= 2.5 kernel-min).** That is a ≈5× undercount before static analysis.

**Disk.**
- Env tarballs: ≈90 (repo, version) pairs × 0.2–2 GB ≈ **40–80 GB** (xarray and sklearn are the big ones).
- Rebench envs, at ≈300 MB × 500 = 150 GB, cannot be cached. They must be built, run and deleted in the same session, keeping only the traces.
- Moving 40–80 GB of env tarballs between kernels through Kaggle datasets is itself slow and quota-bound.

**Network.** 100–500 MB of wheels per env, ≈50–150 GB in total. This is fine on CPU kernels but slow on old sdists that need building.

**Fix (S1 → S2): separate "headline EVAL" from "executed subset".**
1. **C1, C2a and C2b need no execution on EVAL.**
   - Their ground truth is the gold functions from the patch, a static mapping.
   - g_φ and f_θ are trained on CAL repos.
   - Execution on EVAL only serves M, E-ORA, E-TR, the on-path stratum and Study B strata.
2. **Headline EVAL**: ≥600 traceback-free tasks with static gold mapping.
   - Sources: Verified ∪ Lite ∪ SWE-rebench ∪ Loc-Bench (LocAgent), with **≤40 tasks per repo** and ≥25 repos.
   - HEAP needs only an *importable* install, which is easier than passing tests. Where import fails, that repo uses static-only G, reported per repo.
3. **Executed subset**: ~120–150 tasks from pure-Python repos (Django, sympy, pytest, pylint, modern rebench), for M, E-ORA and on-path.
   - Better source: **extract the official SWE-bench instance images' rootfs** without Docker (`crane export` / `skopeo` + tar to `/`, since Kaggle runs as root).
   - Then run `/opt/miniconda3/envs/testbed/bin/python` directly. This reuses the harness's validated envs and avoids uv entirely.
   - Cost: 1–3 GB per image with shared layers. Probe it on 3 images first (0 GPU, ~2 h).

---

## 2. Tracing cost and correctness

### 2.1 setprofile cannot see exceptions (S1 for E-ORA, oracle effects and true-path effects on EVAL)

- Under `sys.setprofile`, a frame that exits by an exception emits **`'return'` with `arg=None`**. This is indistinguishable from `return None`.
- The "raises X" bits and the "returns None" bit are conflated on every pre-3.12 EVAL env, which is all of SWE-bench.
- Exception classes need `sys.settrace` `'exception'` events: a per-frame local trace function with `f_trace_lines = False`. That is 2–4× heavier than setprofile.

**Fix:**
- Use `settrace` (calls + exceptions only) on <3.12.
- CAL-train must be ≥3.12 `sys.monitoring` with RAISE/PY_UNWIND.
- Run a **cross-tracer agreement check** on one CAL repo traced both ways.

### 2.2 Overhead (EST)

**Call-heavy code** (sympy, Django ORM, pyparsing) does 10⁷–10⁸ Python calls per minute of tests.

| Tracer | Overhead |
|---|---|
| A Python-level callback per event | ≈0.3–1 µs |
| `sys.monitoring` with counting only | 2–4× |
| `setprofile`/`settrace` with dict updates | 5–15× |
| + the ≤50-call arg/`self.__dict__` fingerprints (hashing before and after) | +1.5–3× on first calls |

- **CALL cannot return `DISABLE`** after the first sighting, because that would hide polymorphic targets at the same site. The overhead does not decay.
- **Module-global writes via "dict-version diff" are not implementable.**
  - `ma_version_tag` is not exposed to Python and is deprecated in 3.12.
  - Snapshotting module dicts per call is prohibitive.
  - **Fix:** take this bit from AST (`global`, `STORE_GLOBAL`, module-attribute stores) only, and say so. It then contributes nothing to C2b.
- **Shallow fingerprints** miss nested mutation and `__slots__` classes. The "mutates arg" and "writes self attr" labels are therefore biased low. Report a manual audit of 50 functions.

### 2.3 Volume

- **F2P traces** (true path, oracle): 1 traced run per task, 1–5 min. **Cheap.**
- **CAL-train full suites** (12 repos): plain run 1–15 min × 5–15× overhead ≈ **6–48 kernel-hours**. Fine.
  - Hypothesis-heavy suites (attrs, cattrs) explode and need an example cap.
- **E-TR "pre-existing tests" on EVAL, per task: infeasible.**
  - Django's full suite is ≈15k tests, 10–20 min plain, so 2–5 h traced.
  - sympy's full suite is 30–60 min plain, so 5–15 h traced.
  - × 300 tasks = **600–3,000 kernel-hours.**
  - **Fix:**
    - trace once per (repo, version) at one commit and map by qualname (≈60 runs, 60–480 kernel-hours);
    - or trace only the test modules co-located with the BM25 top-20 files;
    - or drop E-TR. It is a secondary arm.
- **Trace size.** Aggregated (caller, offset, callee, count) plus effect-bit counts is ≤100 MB per repo. **Raw event logs would be 10–100 GB.** Aggregate in-process and never log raw events (S3).

---

## 3. GPU budget recount (NH = one 4×L4 session-hour)

### 3.1 Throughput model

**Single stream: 34 tok/s** (VERIFIED). The 4.3 GB W4 shard per GPU at 300 GB/s L4 bandwidth gives a ≈70 tok/s bandwidth ceiling, so TP-over-PCIe overhead already halves it.
- No NVLink. Each decode step needs 120 all-reduces (60 layers × 2).
- That gives a ≈3–6 ms latency floor per step (EST).

**Batched decode:**
- **KV per sequence (EST):**
  - sliding layers: 50 layers × 16 KV heads × 256 × 2 × 2 B ≈ 0.82 MB/token, capped at the 1,024-token window ≈ 0.84 GB;
  - global layers: 10 × 4 × 512 × 1 (k = v) × 2 B ≈ 41 KB/token, × 7k ≈ 0.29 GB;
  - ≈1.1 GB per 7k sequence.
  - At util 0.80 there is ≈10–12 GB of KV room per GPU, so ≈35–45 concurrent sequences.
  - This assumes vLLM 0.19's hybrid allocator frees out-of-window blocks for Gemma-4. **Verify.**
- **Aggregate:** EST 250–500 tok/s central. The arch's 400–800 is optimistic; nothing batched has been measured.

**Prefill:**
- 62 GFLOP/token + attention, at 25–35% of 484 TFLOPS: ≈1.5–2.5k tok/s.
- The arch's 2k is plausible.
- Prefix caching helps **only within a task** (k samples, seeds). The skeleton differs per commit.

**HF forward:**
- p0b *training* layer-split measured 100 tok/s at 4k and 82 at 8k (VERIFIED).
- Forward-only is ≈3× that, so **≈250–300 tok/s at 4k, lower at 12k windows** because the head_dim-512 global layers run on the mem-efficient SDPA kernel.
- 4 NF4 replicas (≈18 GB per GPU) might fit forward-only, but that is **unproven**.

**Per session:** 8 min vLLM start (VERIFIED) + scorer_env restore and model mount ≈ 3–6 min + the manual UI start (HQ: L4 cannot be requested from the CLI). That is **≈0.25 NH per vLLM session** and **≈0.35 NH per HF NF4 session** (12-min load, VERIFIED).

### 3.2 Hidden GPU item: the skeleton does not fit in 6k tokens (S1)

A "module→class→def(sig)+1st doc line" skeleton is far over budget for the big repos:

| Repo | Skeleton size (EST) |
|---|---|
| Django (≈30k defs) | ≈250–400k tokens |
| sympy | ≈300k+ |
| scikit-learn | ≈150k |

**≤6k tokens means a hidden retrieval stage.** Either:
- (a) a BM25-selected file subset, which conditions the generator on the same π that B-PPR uses, confounds C1 and must be shared with every LLM baseline; or
- (b) an Agentless-style file-selection call first, which is +1 call per task (≈+30% G1 tokens) and is itself a localiser that must be reported.

This is also where most of the LLM's localisation power will come from, so the paper must own it.

**Fix:**
- Make it explicit as G0: file selection from the repo tree (≈3–5k tokens), then skeletons of the top-N files.
- Give the identical G0 output to B-PPR's π (restart mass on the selected files) and to the new B-DIRECT baseline (§5).

### 3.3 Line-by-line recount

| Line | v1 central | Opt | **Central** | Pess | Justification |
|---|---|---|---|---|---|
| P0 HF forward probe | 0.5 | 0.5 | 0.75 | 1.0 | p0b crashed 2 of 4 stages; assume one rerun |
| P0v vLLM *batched* throughput probe (missing) | — | 0.3 | 0.4 | 0.5 | Every G-line rests on an unmeasured 400–800 tok/s |
| G0 file selection (missing) | — | 0.3 | 0.5 | 0.8 | ~600 tasks × 4k prefill + 100 decode |
| G1 main generation (518 tasks × 15 samples) | 1.6 | 1.5 | 2.2 | 3.5 | Prefill 3.6M at 1.5–2.5k = 0.4–0.7 h. Decode 1.9M at 250–500 = 1.1–2.1 h. +2 starts. Format-failure retries (thinking off, strict STEP grammar) ~10% |
| G2 R1 + R2 (300 tasks, 1 seed) | 1.0 | 0.9 | 1.3 | 2.0 | No prefix sharing across views: 4.2M prefill + 0.75M decode |
| G2′ R2 at 3 seeds (C2a is primary under R2) | — | 0.5 | 0.8 | 1.2 | Otherwise the primary contrast has 1/3 of the generator averaging the others get |
| G3 effect parse | 0.2 | 0.1 | 0.2 | 0.3 | — |
| G4 Qwen3-32B-AWQ / Qwen3-Coder-30B-A3B | 0.8 | 0.8 | 1.2 | 2.0 | Separate vLLM start. The ~17–19 GB weights must first be staged as a Kaggle dataset from a CPU kernel (20 GB output limit, tight). Qwen thinking must be suppressed (`/no_think`). Chat-template or parser mismatch → one rerun |
| G5 Gemma-4-E4B (hidden inside the reserve in v1) | (0.2) | 0.2 | 0.3 | 0.5 | Must be its own line, not reserve |
| B1 CoSIL, 100 tasks | 1.8 | 1.2 | 2.0 | 3.0 | ≤20 calls × 10–15k context. Even with prefix caching ≈3–5M new prefill + 0.6M decode. Calls within a task are sequential, so batch across tasks |
| B2 SweRankEmbed-Large | 0.8 | 0.6 | 1.0 | 1.8 | 25M tokens × 14 GFLOP ≈ 350 PFLOP → ≈0.6 h at 170 TFLOPS effective, plus 4 replica loads. ×2 if R2 bodies are re-embedded (they must be) |
| B3 graph-as-context, 150 tasks | 0.5 | 0.4 | 0.6 | 0.9 | — |
| **B4 B-DIRECT + LLM reranker over BM25 top-50 (missing, §5)** | — | 0.6 | 0.9 | 1.4 | 300 tasks × (~8k + ~12k) prefill, short decode |
| L1 LATENT | 2.5 | 1.5 | **4.0** | 7.0 | 3.7M tok / 250–300 tok/s = 3.4–4.1 h + 2–3 NF4 loads. Unclear whether "two random orders" is inside the 3.7M; if not, ×1.6. Opt assumes the 4-replica path works |
| L2 NF4→W4A16 check | 0.3 | 0.3 | 0.4 | 0.6 | Needs vLLM *and* HF in one session; the p0b version of this stage crashed |
| L3 verbal baseline | 0.7 | 0.5 | 0.7 | 1.0 | Pair count unspecified |
| L4 LATENT on a 2nd model | 0.6 | 0.6 | 1.0 | 1.5 | New model's HF weights staged, new layer map |
| J1 JS | 0.6 | 0.4 | 0.6 | 1.0 | GPU is not the problem; CPU and engineering are (§6) |
| R reserve | 1.5 | 1.5 | 2.5 | 4.0 | The history is under-counting by 2× (L5). 24.5 h of quota went to probes in one week (VERIFIED) |
| **Total** | **13.4** | **12.9** | **21.6** | **34.5** | |

**Verdict.**
- The central case fits 25 NH with little slack.
- **The pessimistic case breaks the cap by ~40%.**
- Study B alone (L1–L4 + P0) is **6.9 NH central**, a third of the paper's GPU.
- The v1 "floor paper" is ≈4.5–7 NH; I recount it at **6–8 NH**.

**Qwen weights.** Kaggle Models hosts Qwen3 checkpoints (verify the int4/AWQ variants). Official AWQ is on HF (`Qwen/Qwen3-32B-AWQ`). Staging needs a CPU kernel with Internet plus a dataset upload.
- Qwen fits 4×L4 easily: ~5 GB per GPU at TP4.
- vLLM 0.19.1 supports Qwen3.
- Risk is moderate, but an AWQ run uses different kernels and quantisation than Gemma's W4A16, which is a confound that has to be stated.

---

## 4. Statistics

### 4.1 MDE (paired, normal approximation, 80% power)

**Formula:** MDE = (z₁₋α/2 + 0.842) · SD/√n. The Holm step-down levels are α = 0.0167, 0.025 and 0.05.

| Contrast | Predicted | SD assumed | n = 300: MDE at α = 0.0167 / 0.05 | Power for the predicted Δ (α = 0.0167 / 0.05) |
|---|---|---|---|---|
| C1 BMM − PPR | +0.05 | 0.25 | 0.047 / 0.040 | ≈0.85 / 0.91 |
| C1 if SD = 0.30 | +0.05 | 0.30 | 0.056 / 0.049 | ≈0.70 / 0.81 |
| C2a under R2 | +0.04 | 0.12 | 0.022 / 0.019 | >0.99 |
| C2a if SD = 0.18 | +0.04 | 0.18 | 0.034 / 0.029 | ≈0.89 / 0.94 |
| **C2b BMM − ASTEFF** | **+0.015** | 0.12 | 0.022 / 0.019 | **≈0.41 / 0.58** |
| C2b if SD = 0.18 | +0.015 | 0.18 | 0.034 / 0.029 | **≈0.17 / 0.28** |

- **C2b is under-powered by design (S1 for the "execution-calibrated" sub-claim).** The predicted effect is below its own MDE, so a null is uninformative.
- **Fix:**
  - Move C2b to a secondary estimate with a CI; or
  - raise n to ≥600 (possible only with the execution-free EVAL of §1). At n = 600 the MDE is 0.016 (SD 0.12) and power is ≈0.6–0.75; or
  - test C2b at the **effect-bit level** on CAL hold-out (AUROC, thousands of functions), where it is properly powered, and keep the task-level readout descriptive.
- **The SD of nested ΔRR is probably >0.12.** A β change that moves gold between rank 1 and rank 3 gives a ΔRR of 0.67 on that task. If 20–30% of tasks change top-10 order, SD ≈0.15–0.2 (EST).

### 4.2 MRR resolution

- In Django-size repos, typical gold ranks are 50–2,000 (RR 0.0005–0.02).
- **ΔMRR is driven almost entirely by the 25–40% of tasks with a top-10 hit.** The effective n is ~100, and the ΔRR distribution is heavy-tailed and zero-inflated. A bootstrap handles that, but power is lower than the normal approximation says.
- **Fix:** pre-register **mean log₂-rank** (or hit@10) as a co-primary sensitivity metric. It has resolution in the long tail, and the decoder's prior and transitions mostly act there.

### 4.3 Clustering (S2)

- Survivors are ~60% Django + sympy. With unequal cluster sizes and an ICC of ΔRR by repo of 0.05, the design effect is ≈1 + (m̃−1)·ρ, with m̃ being the size-weighted mean. With Django near 180 tasks, **DE ≈ 2–4, so the clustered MDE is 1.4–2× the task-level one.**
- With ~12 clusters, a repo-clustered bootstrap is itself unstable.
- **Fix:**
  - cap tasks per repo at 40;
  - use ≥25 repos (rebench + Loc-Bench);
  - report the repo-macro mean as the primary or co-primary;
  - make the clustered bootstrap the primary inference, not a sensitivity check.

### 4.4 Tuning leakage and fairness (S2)

- **g_φ and f_θ are trained on CAL-train repos, and CAL-tune tasks come from the same repos.** On CAL-tune, g_φ is in-sample (optimistic), so TPE learns too large a β and too much trust in A. On EVAL both are out of repo.
  - This *biases C2a, C2b and C1 toward null or harm*, and makes CAL-tune P1 pass too easily.
  - **Fix:** cross-fit. For each CAL-tune task, use g_φ and f_θ trained leave-that-repo-out.
- **CAL repos are small** (attrs, toolz: 0.5–3k functions) while EVAL repos are big (10–30k). Hyperparameters that depend on repo size (T, τ, ρ, teleport) and the measured SDs (A16) do not transfer.
  - **Fix:** normalise scores by log|V|, and include 2–3 large CAL repos.
  - Or measure SDs on a **burned 50-task EVAL pilot**, excluded from the headline.
- **Is 100 SWE-rebench tasks from 12 small pure-Python CAL repos available?** Probably not; small libraries have few qualifying PRs. **Check this on Day 1.**
  - It needs no execution (static gold), so it can be counted in an hour.
- **TPE over 200 trials, 8 dims (BMM) vs 2 (PPR).**
  - The MRR SE on 100 tuning tasks is ≈0.03. Many trial differences are noise, so BMM gets winner's-curse overfitting and PPR saturates.
  - That direction is fair to the baseline, but it wastes BMM's power.
  - **Fix:** fix ε, δ and η a priori; tune ≤4 params; use a 5-fold CV estimate on CAL-tune.
- **C1 is stated "alone and BM25-fused"**, which is two tests under one Holm slot. Pre-specify one (fused), or Holm is over 4 primaries.

### 4.5 Seeds, generator variance, equivalence (S3)

- "k = 5 × 3 seeds" at temperature 0.7 is just k = 15. The seed-level spread is fine to report, but it is not a separate variance source.
- **Item-averaging over 15 samples absorbs generator noise into item noise.** The task bootstrap is then valid, but it does *not* give "rerun the generator" uncertainty.
  - **Fix:** add a 2-level bootstrap (tasks, then samples) for C1 only.
- **C3 kills use "≈"** ("SHUF ≈ BMM and DPR ≈ BMM"). These need TOST equivalence margins (e.g. ±0.01 MRR), or the kill can never fire at this n.
- **Contamination (A15).**
  - Post-cutoff rebench tasks outside CAL and dev number perhaps 50–150. At that n the MDE for C1 is ≈0.07–0.11, so "the gain exists only pre-cutoff" cannot be detected.
  - Say this explicitly, or enlarge the post-cutoff pool.

---

## 5. Baselines

### 5.1 Runnable offline at this scale?

| Baseline | Status | Issue |
|---|---|---|
| CodeRankEmbed (137M) | Yes, CPU | `trust_remote_code` (nomic-bert) must be vendored for offline use. ~25M unique-body tokens/view at ~1–3k tok/s on 4 vCPU ≈ 3–7 kernel-hours per view; ×3 views (orig, R1, R2) |
| SweRankEmbed-Small | Yes, CPU | Same. **Check the licence** (Salesforce research releases are often CC-BY-NC). Fine as a baseline, but it must not enter the open-source pipeline or `s_name` |
| SweRankEmbed-Large (7B) | Yes, GPU | Weights must be staged (~15 GB) |
| CoSIL | Probably | It was published with open 32B models, so Gemma-31B is plausible. It outputs a short ranked list, not a full ranking. **Define the tail fill** (BM25 order) and report hit@k as its primary readout. 100 tasks gives an MDE of ≈0.08: "≈ CoSIL" is not a claim you can make |
| GraphLocator | Cite only | — |
| **B-PPR** | Fairness | It shares π, G and A. But **C1 = BMM − PPR adds an LLM.** It is not a one-change control for "decoding", because any gain could be the LLM reading issue + skeleton. Decoder-specific evidence is B-NAMES and X-BAG, which v1 made secondary |

### 5.2 Missing baselines a reviewer will demand (S1 for B-DIRECT)

1. **B-DIRECT: the same prompt (G0 + skeleton + issue), asking Gemma for a ranked top-10 culprit list.**
   - This is Agentless-style function localisation at matched tokens, tail-filled with BM25.
   - **If BMM ≈ B-DIRECT, the imagined path + HMM adds nothing over asking directly.** This is the comparator C1 should be, or a fourth primary.
   - Cost: ~0.3 NH.
2. **An LLM listwise reranker over BM25/SRE-Small top-50** (SweRankLLM-style, using Gemma). It is the strongest cheap pipeline in the SweRank paper. ~0.6 NH.
3. **LocAgent (2503.09089)**, a graph-guided LLM localiser on a heterogeneous code graph (contain/import/invoke/inherit). Its Loc-Bench is also a ready, partly post-cutoff EVAL source.
   - Run it at ≤20 calls on 100 tasks, or cite its function-level Acc@k on Lite with a matched-subset caveat.
   - **v1 does not mention it, and it is the closest graph-reasoning localiser.**
4. **Published SOTA numbers** (SweRank, LocAgent, CoSIL, OrcaLoca, Agentless function-level Acc@5/10 on Lite).
   - Report on the matched Lite subset with our metric, or explain why they are not comparable (pools vs full-repo, traceback filter).
5. HyperFL / a fine-tuned issue encoder: "cite, not run" is acceptable. SweRankEmbed already covers learned issue→function retrieval.

**Fix:** B-DIRECT and the LLM reranker replace B3 (graph-as-context) and B2 (SRE-Large) in the core set. Net GPU ≈ 0.

---

## 6. Rubric fit and word budget

| Criterion | v1 status | Break |
|---|---|---|
| Novelty incl. insight | Closeness 2 for the emission. Insight = "behaviour + structure correct imagined names" | Diluted across BMM + HEAP + CHAIN + Study B + measurement. The insight is only clean if C2a survives under R2 *and* B-DIRECT is beaten |
| Quality / generality | 2nd generator, E4B, JS stretch | **JS is not credible:** a JS tracer, Jelly integration, a JS effect alphabet, JS CAL repos, JS envs ≈ 7–10 person-days. Drop it. 2nd model + E4B is credible if staging works. **Repo generality is the real gap:** 60% Django/sympy. The ≥25-repo capped EVAL fixes this better than JS |
| Relevance | Graph reasoning + localisation: strong | — |
| Verifiability | Pre-registration hash, CIs | A no-Docker env harness with 40–60% yield is not reproducible by others. **Fix:** release generations, graphs, effect tables and traces as a Kaggle dataset; the headline must not depend on rerunning envs (it won't under §1's fix) |
| Clarity, ≤3,000 words | 9 baselines, 14 X/E/T arms, 3 views, 7 strata, 4 claims + M + Study B (B1, B2, 6 controls) | **Does not fit.** 3,000 words holds one method section (~700), one setup (~400), one main table, one ablation table and one measurement figure, plus ~600 words of analysis. **Study B is a second paper.** Cut it to one paragraph or drop it |

---

## 7. The 5 most damaging breaks, with the minimal v2 fix

| # | Break | Sev | Minimal v2 fix |
|---|---|---|---|
| 1 | **Scope vs calendar and quota.** 32–45 person-days of engineering against 27 days to freeze, shared with the main track. 21.6 NH central (34.5 pess) against a quota the main track also needs | S1 | Cut Study B (L1–L4, ≈6.9 NH, 5–7 days), JS, E-TR, PoTo/PyCG (Pyright ∪ HEAP only), B-CTX and SRE-Large. Target: ≈15 person-days, ≈9–11 NH |
| 2 | **Execution-validated EVAL is the riskiest step and the headline does not need it.** Yield 214–491 (central 343), 60% Django/sympy. CPU is undercounted ~5–15×. Py3.6 and compiled repos fail. setprofile cannot see exceptions | S1 | Headline EVAL = static gold mapping, ≥600 traceback-free tasks, ≤40 per repo, ≥25 repos (Verified ∪ Lite ∪ rebench ∪ Loc-Bench). Executed subset of 120–150 via SWE-bench image rootfs extraction (probe on 3 images first), for M, E-ORA and on-path only. Use settrace (exceptions) on <3.12 |
| 3 | **The ≤6k skeleton is impossible for Django/sympy/sklearn.** A hidden retrieval stage confounds C1 | S1 | An explicit G0 file-selection call. Its output is shared by B-PPR's π, B-DIRECT and the reranker. Report G0's own file-hit as a stage |
| 4 | **The C1 comparator is wrong.** PPR has no LLM. B-DIRECT, an LLM reranker and LocAgent are missing | S1 | Primary C1 = BMM (fused) − B-DIRECT (fused). Keep PPR as the decoder-mechanism control (secondary). Add the LLM reranker. Cite LocAgent/SweRank on matched Lite |
| 5 | **Statistics.** C2b is below its own MDE (power 0.17–0.58). Clustering gives DE 2–4. g_φ is in-sample on CAL-tune (leak). Small-repo CAL-tune SDs don't transfer. C1 counts twice | S1/S2 | Demote C2b to an effect-bit-level test on CAL hold-out plus a task-level estimate. Cross-fit g_φ/f_θ on CAL-tune. Clustered bootstrap + repo-macro as primary. One C1 variant. Co-primary log-rank. TOST margins for C3. SDs from a burned 50-task EVAL pilot |

**Also fix (S2/S3):**
- the module-global effect bit → AST-only;
- aggregate traces in-process;
- B-CoSIL readout = hit@k only;
- SweRankEmbed licence check;
- verify whether "two orders" is inside L1's tokens;
- E4B as its own line;
- G2′ R2 at 3 seeds;
- a 2-level bootstrap for C1;
- count CAL-tune task availability on Day 1;
- normalise hyperparameters by log|V|.

---

## 8. Corrected budget tables

### 8.1 GPU (NH)

**v1 as written, recounted:** 12.9 / **21.6** / 34.5 (opt / central / pess). Details in §3.3.

**Minimal v2** (Study B, JS, B2 and B3 cut; G0 and B4 added; EVAL ~600 tasks):

| Line | Opt | Central | Pess |
|---|---|---|---|
| P0v vLLM batched-throughput probe (folded into the first G session) | 0.3 | 0.4 | 0.5 |
| G0 file selection, ~800 tasks (600 EVAL + 118 DEV + ~100 CAL-tune) | 0.4 | 0.6 | 0.9 |
| G1 generation, 800 tasks × k = 10 (2 seeds × 5) | 1.8 | 2.7 | 4.2 |
| G2 R2 (all 600, k = 10) + R1 (150, k = 5) | 1.1 | 1.6 | 2.4 |
| G3 effect parse | 0.1 | 0.2 | 0.3 |
| G4 Qwen3 generator, 300 tasks | 0.8 | 1.2 | 2.0 |
| G5 Gemma-4-E4B, 150 tasks | 0.2 | 0.3 | 0.5 |
| B4 B-DIRECT + LLM reranker | 0.6 | 0.9 | 1.4 |
| B1 CoSIL, 100 tasks (hit@k only) | 1.2 | 2.0 | 3.0 |
| R reserve | 1.5 | 2.0 | 3.0 |
| **Total** | **8.0** | **11.9** | **18.2** |
| (Optional, only if ≥8 NH remain after G2) Study B kill only: P0 + L1-kill 1.2M + L2 | 1.5 | 2.4 | 3.5 |

### 8.2 CPU (kernel-hours, 4-vCPU Kaggle kernel; × 4 = vCPU-h)

| Line | v1 (implied) | v1 recount: Opt / Central / Pess | v2 minimal: Opt / Central / Pess |
|---|---|---|---|
| Env build per (repo, version) + per-task rebench envs | ≈50 total for everything | 30 / 60 / 120 | Import-only installs for HEAP 15 / 30 / 60; rootfs extraction 3 / 6 / 12 |
| Validation 2×2 + traced F2P (1,200 candidates; v2: 150) | — | 120 / 240 / 400 | 15 / 30 / 50 |
| Static union per surviving snapshot | — | 83 / 167 / 333 (PyCG timeouts) | Pyright ∪ HEAP, ~650 snapshots: 50 / 100 / 200 |
| HEAP import walks | — | 17 / 33 / 67 | 20 / 40 / 80 |
| E-TR pre-existing-suite traces | — | 60 / 180 / 480 (per version), or 600–3,000 (per task) | Dropped |
| CAL-train suite traces (12 repos) | — | 6 / 18 / 48 | 6 / 18 / 48 |
| Embeddings (CRE + SRE-S × 3 views) | — | 7 / 14 / 40 | 7 / 14 / 40 |
| LibCST renaming | — | 5 / 10 / 20 | 5 / 10 / 20 |
| Decoding + TPE for every arm | — | 10 / 30 / 80 | 10 / 25 / 60 |
| **Total kernel-hours** | **≈50 (200 CPU-h)** | **338 / 752 / 1,588** | **131 / 273 / 570** |
| Wall time at 5 concurrent kernels | 10–15 h | 68 / 150 / 318 h | 26 / 55 / 114 h |

- **CPU is free, but not free of babysitting.** Every 12-h session expiry and every dataset hand-off is a manual step.
- v1's CPU line is undercounted **≈7–30×**.
- The minimal v2 cuts it by ≈2.7× and removes the S1 yield risk from the headline.

### 8.3 Person-days

| | Opt | Central | Pess |
|---|---|---|---|
| v1 | 32 | 38 | 45 |
| Minimal v2 | 13 | 16 | 20 |

**Of the 27 days left, minimal v2 is the only version that leaves the main track any time.**
