# Evaluating LLM reasoning over code graphs and repository comprehension (as of Oct 2026)

Labels used throughout:
- **VERIFIED**: I read the primary page (arXiv abstract, alphaXiv full text, or official blog) in this session.
- **SNIPPET**: seen only in a search-engine summary. Plausible but not checked against the primary source.
- **INFERRED**: my own reasoning or arithmetic, not a claim from any source.
- Venue status is given as *peer-reviewed (venue)* or *preprint*.

Project context: we have 129 public tasks from 4 well-known repos (fastapi, rich, requests, httpx), likely memorised by Gemma. A hidden set of about 120 tasks comes from unseen private repos. Local gains stopped transferring. A renamed dev set was dropped because renaming keeps the structure. Already covered elsewhere, so not repeated here: Schrödinger's Repo, Jagged Frontier, Face/Off, SHERLOC, GraphDancer.

---

## Q1. Which benchmarks test repo-level structural questions rather than end-to-end resolve?

### Takeaway
A small but growing set of benchmarks asks direct structural questions with machine-checkable ground truth. The three most useful for us:
- **TraceEval** (2026): recover runtime call structure, labels from execution tracing, with a pipeline that turns *any* repo into new instances.
- **RepoReasoner** (FSE 2026): call-chain prediction from pytest tracing.
- **CodeQueries** (2022): CodeQL-derived semantic queries, single-hop and multi-hop.

Most other "repo comprehension" benchmarks are free-text QA graded by an LLM judge or rubric: SWE-QA, CoReQA, StackRepoQA, SWE-Atlas QnA, RepoProbe. They are sparse (one item per question) and mostly built on famous repos. Localisation benchmarks (LocBench) give dense-ish ranked metrics but are still tied to issues.

### Cited Findings

**Structural / graph-ground-truth benchmarks**
- **TraceEval** (*preprint*, submitted 10 May 2026). Task: recover a program's runtime call structure from source code. **VERIFIED** — [arXiv 2605.11006](https://arxiv.org/abs/2605.11006)
  - Labels are "mechanically witnessed by validation execution", not made by static tools or human annotators.
  - Size: 10,583 programs (2,129 test / 8,454 train) from 1,600+ repos, in Python, JS and Java.
  - Ships a reproducible pipeline that "converts any open-source repository into new verified benchmark instances".
  - Results: Claude-Opus-4.6 reaches 72.9% avg F1. A fine-tuned Qwen2.5-Coder-32B reaches 71.2% F1, a +55.6 lift from fine-tuning.
- **RepoReasoner** (*peer-reviewed, Proc. ACM Softw. Eng. / FSE 2026*). **VERIFIED** — [alphaXiv 2607.25996](https://www.alphaxiv.org/abs/2607.25996)
  - Two tasks: Output Prediction (858 samples) and Call Chain Prediction (169 samples). Built from 525 test functions in 169 test files across 14 repos.
  - Call-chain ground truth comes from dynamic tracing of pytest runs (a `hunter`-based plugin that records file-level call transitions).
  - Key pattern: **high precision, low recall** on call chains. Example: GPT-4.1-Mini gets 81.4% precision but 31.8% recall, i.e. weak multi-hop dependency understanding.
- **CodeQueries** (Google; 2022, so pre-2025). **SNIPPET** — [arXiv 2209.08372](https://arxiv.org/abs/2209.08372); [Google Research](https://research.google/pubs/codequeries-a-dataset-of-semantic-queries-over-code)
  - 133,456 Python examples from 52 public CodeQL queries (15 multi-hop, 37 single-hop).
  - Answers are code spans, with positive and negative examples. Context is file-level, not repo-level.
- **CG-Bench** (LMPL @ ICFP/SPLASH 2025). Described as the first manually built call-graph-understanding benchmark: 104 call sites from 7 C/C++ projects. **SNIPPET** — [researchr](https://conf.researchr.org/details/icfp-splash-2025/lmpl-2025-papers/3/CG-Bench-Can-Language-Models-Assist-Call-Graph-Construction-in-the-Real-World-)
- **"SemBench"**: function reachability (3,548 problems), data dependency (1,996) and dead-code detection. **SNIPPET**, and the source URL could not be pinned down (it appeared in a search summary next to an IJCAI 2026 proceedings PDF). Treat as unverified. — [search hit, IJCAI'26 PDF](https://www.ijcai.org/proceedings/2026/0064.pdf)
- **Earlier call-graph study**: LLMs for type and call-graph analysis in Python/JS, measured against PyCG/HeaderGen micro-benchmarks. The search summary says PyCG itself has ~99% precision and ~70% recall. **SNIPPET** — [arXiv 2410.00603](https://arxiv.org/pdf/2410.00603)

**Repo QA (mostly free-text, judge-graded)**
- **SWE-QA**: 576 QA pairs from 11 popular repos. Taxonomy built from 77,100 crawled issues. Covers intention understanding, cross-file reasoning and multi-hop dependency analysis; includes SWE-QA-Agent. **SNIPPET**; a listing under ACL 2026 Findings suggests *peer-reviewed (Findings of ACL 2026)*. — [arXiv 2509.14635](https://arxiv.org/html/2509.14635v1); [ACL Anthology preview](https://preview.aclanthology.org/ingest-acl/2026.findings-acl.402/)
- **CoReQA**: QA built from GitHub issues and comments of 176 repos in 4 languages, judged by an LLM on 5 aspects. **SNIPPET** — [arXiv 2501.03447](https://arxiv.org/pdf/2501.03447)
- **StackRepoQA** ("Beyond Code Snippets", *preprint* Mar 2026, with an ACM DOI listed). **VERIFIED** — [arXiv 2603.26567](https://arxiv.org/abs/2603.26567)
  - 1,318 Stack Overflow questions with accepted answers, across 134 Java projects.
  - "High scores often result from verbatim reproduction of Stack Overflow answers rather than genuine reasoning."
  - Structural signals improved accuracy, but it stayed limited.
- **RepoProbe** (*peer-reviewed, ASE 2026*). **VERIFIED** — [arXiv 2608.04783](https://arxiv.org/abs/2608.04783)
  - Questions come from GitHub Discussions rather than Issues, chosen to avoid pattern-matching on error logs.
  - **Checklist-based verification**: answers are decomposed "into atomic, verifiable facts". This is a dense-scoring design.
  - Reports an "Edit Bias": models write code instead of analysing the architecture.
  - The abstract does not give task or checklist counts.
- **SWE-Atlas Codebase QnA** (Scale AI, 2026; leaderboard/benchmark, not peer-reviewed). **SNIPPET** — [Scale Labs](https://labs.scale.com/papers/sweatlas); [HF dataset](https://huggingface.co/datasets/ScaleAI/SWE-Atlas-QnA/blob/main/README.md)
  - 124 QnA tasks from 11 *well-known public* repos in Go, Python, C and TS.
  - Agents must run software and trace execution across files.
  - Grading mixes programmatic checks with rubrics.
- **LongCodeBench**: LongCodeQA plus LongSWE-Bench at context lengths from 32K to 1M tokens. Tests long-context QA, not graph structure. **SNIPPET** — [alphaXiv benchmark page](https://www.alphaxiv.org/benchmarks/stanford-university/longcodebench)
- **RepoQA**: "Searching Needle Function". 500 tasks from 50 repos in 5 languages, with needle placed at 10%–100% depth. Retrieval by description, not structural reasoning. ICML 2024 workshop. **SNIPPET** — [mlanthology](https://mlanthology.org/icmlw/2024/liu2024icmlw-repoqa)

**Localisation**
- **LocBench / Loc-Bench_V1** (from LocAgent, *peer-reviewed ACL 2025*). **SNIPPET** — [arXiv 2503.09089](https://arxiv.org/pdf/2503.09089); [RepoNav 2609.08355](https://arxiv.org/pdf/2609.08355)
  - Real issues from 5 Python repos, collected after model cutoffs. Covers bugs, features, performance and security issues.
  - Ground truth is the sets of files, classes and functions to modify.
  - Metrics: file Acc@{1,3,5}, module/function Acc@{5,10}, plus Recall, NDCG, Precision and MAP.
- **Other graph-based localisers evaluated on LocBench or SWE-bench-Lite localisation**: LARGER (2605.16352), RepoNav (2609.08355), SweRank (2505.07849), FastCode (2603.01012). **SNIPPET** (titles and links only) — [LARGER](https://arxiv.org/pdf/2605.16352), [SweRank](https://arxiv.org/pdf/2505.07849)

**Adjacent (graph reasoning, not code)**
- **DW-Bench**: data-warehouse graph topology reasoning, 1,046 questions in 13 subtypes. **SNIPPET** — [arXiv 2604.18964](https://arxiv.org/html/2604.18964)
- **GABench**: 10,400 graph-analysis tasks with verifiable ground truth, built from 13 graph datasets. **SNIPPET** — [arXiv 2608.01684](https://arxiv.org/html/2608.01684v1)

### Inferences
- **INFERRED**: For our "who calls X / impact / path" needs, the best templates are TraceEval (generator) and RepoReasoner (task format plus P/R metric). Both derive ground truth from *execution traces*, not from the agent's own graph. That avoids circularity if we later score chains against a repaired static graph.
- **INFERRED**: RepoReasoner's high-precision, low-recall pattern looks like the same failure mode as our premature stopping. Models name a few correct hops and miss the rest. Recall on hops is the metric that matters.
- **INFERRED**: The free-text QA benchmarks are poor dev-set proxies for us. They use famous repos (contamination), judge-based scoring (noisy), and one item per question (sparse). RepoProbe's atomic-checklist scoring is the exception worth copying.

### Gaps
- **Multi-SWE-bench** and **CodeRAG-Bench** were not checked in this session. No search hit gave primary details, so their structural content is unknown here.
- No verified counts for the size of SWE-QA's multi-hop subset, or for RepoProbe's checklist items per question.
- "SemBench" could not be traced to a primary source.

---

## Q2. Contamination-free / post-cutoff / unseen-repo protocols: what do they show about memorisation vs structural skill?

### Takeaway
There are three families of protocol:
1. **Temporal freshness**: SWE-rebench, SWE-bench-Live, LocBench.
2. **Semantic-preserving rewrites**: RepoReasoner I/O rewriting, plus the already-covered rename and layout studies.
3. **Pipelines that mint new verified instances from arbitrary repos**: TraceEval, SWE-rebench, SWE-bench-Live.

All three show memorisation inflation, but the measured gaps vary a lot (about 4 pp to 18 pp depending on model and benchmark). None of them uses *private* repos, so even "fresh" tasks come from famous public projects.

### Cited Findings
- **SWE-rebench** (*peer-reviewed, NeurIPS 2025*). **VERIFIED** — [arXiv 2505.20411](https://arxiv.org/abs/2505.20411)
  - 21,000+ interactive Python tasks, collected by an automated pipeline with a continuous supply of fresh tasks. Leaderboard at swe-rebench.com.
  - Comparing against SWE-bench Verified shows that "performance of some language models might be inflated due to contamination". The abstract gives no model-specific numbers.
- **Claim: DeepSeek-V3 scores 39.7% on SWE-bench Verified vs 21.3% on fresh tasks.** **SNIPPET from an aggregator blog**, not confirmed in the primary abstract. Treat with caution. — [mindstudio blog](https://www.mindstudio.ai/blog/swe-rebench-benchmark-decontaminated-tests-model-inflation)
- **SWE-bench-Live** (Microsoft; arXiv 2505.23419, *preprint* at time of search result). **SNIPPET** — [arXiv 2505.23419](https://arxiv.org/abs/2505.23419)
  - 1,319 instances from 93 repos (Jan 2024–Apr 2025), adding about 50 verified instances per month. Lite subset of 300.
  - Reports a "substantial performance gap" vs static SWE-bench.
- **RepoReasoner I/O rewriting**: rewrite test inputs and expected outputs while keeping the logic. **VERIFIED** — [alphaXiv 2607.25996](https://www.alphaxiv.org/abs/2607.25996)
  - Performance drops across all models. DeepSeek-R1 goes from 69.1% to 65.0% Pass@1.
- **StackRepoQA**: high scores partly come from verbatim reproduction of Stack Overflow answers. This is the first evidence of memorisation in repo-level QA. **VERIFIED** — [arXiv 2603.26567](https://arxiv.org/abs/2603.26567)
- **Seen vs unseen real-world class-level code generation**: nearly identical performance on seen and unseen partitions. LLMs reach 84–89% on synthetic benchmarks vs 24–35% on real code. **SNIPPET** — [arXiv 2510.26130](https://arxiv.org/pdf/2510.26130)
  - This conflicts with the memorisation findings above. It is a generation task, not localisation.
- **TraceEval's repo→instance pipeline** is the only one found that targets *structural* (call-structure) questions and can be pointed at an arbitrary repo. **VERIFIED** — [arXiv 2605.11006](https://arxiv.org/abs/2605.11006)

### Inferences
- **INFERRED**: Temporal freshness removes *task* leakage but not *repo* leakage. A fresh issue on fastapi is still on a repo whose code Gemma has memorised. Our hidden set is a different distribution shift: unseen repo, not just an unseen task. The SWE-rebench / Live gaps are therefore a *lower bound* on our shift.
- **INFERRED**: The most faithful proxy for "private repo" is code that post-dates Gemma's training cutoff *and* is obscure. Concretely: small, low-star PyPI packages created after the cutoff, or repos we write and never push. Run a TraceEval/RepoReasoner-style tracing pipeline on them to mint structural probes.
- **INFERRED**: Procedurally generated synthetic repos give unlimited, guaranteed-unseen structure. But the class-level study (84–89% synthetic vs 24–35% real) warns that synthetic difficulty does not track real difficulty. They are best used as a *structural-skill* probe, not as the main dev-set proxy.
- **INFERRED**: Before tuning further, validate any new dev set by checking that it **rank-correlates with hidden-LB scores across past submissions**. This is the only direct test of proxy validity available to us.

### Gaps
- I found no benchmark built from genuinely private or closed-source repos with published numbers. That case is unstudied in public.
- I found no procedurally generated code-graph benchmark (synthetic repos with controlled call-graph topology) in 2025–2026. The search returned only general graph benchmarks (GABench, GraphInfer-Bench).
- I did not verify SWE-bench-Live's final venue or its specific numeric gaps.

---

## Q3. Dense per-claim metrics and power: how many items are needed to detect a 3–5 pp effect?

### Takeaway
Dense metrics exist in pieces:
- **Ranked localisation metrics** (LocBench: Acc/Recall/NDCG/MAP@k).
- **Per-hop P/R on call chains** (RepoReasoner, TraceEval F1).
- **Atomic-fact checklists** (RepoProbe).
- **Evidence-hit vs conversion** (GraphDancer, already covered).

For power, a paired 4 pp difference needs roughly 500–1,500 independent items, depending on how often the two systems disagree. Dense items within a task help only by a design-effect factor. At realistic intra-task correlation, 129 tasks × 20 probes gives only about 250–900 effective items. So density plus *more repos* is required.

### Cited Findings
- **Miller / Anthropic, "Adding Error Bars to Evals"** (*preprint* arXiv 2411.00640, 2024). **VERIFIED** — [Anthropic research](https://www.anthropic.com/research/statistical-approach-to-model-evals); [arXiv 2411.00640](https://arxiv.org/abs/2411.00640)
  - Report the SEM with a 95% CI (±1.96×SEM).
  - Cluster SEs on the unit of randomisation. Clustered SEs "can be over three times as large as naive standard errors".
  - Resample chain-of-thought answers per question to reduce within-question variance.
  - Use paired differences: frontier models' per-question correctness correlates at "between 0.3 and 0.7".
  - Do power analysis.
- **Claim: "500 questions at 50% baseline needs ~9 pp gap for 80% power."** **SNIPPET only.** The Anthropic page I fetched contains no numeric power example, so treat it as unverified. — [search summary](https://arxiv.org/pdf/2411.00640)
- **Clustering in paired experiments**: clustering at the unit level instead of the pair level can inflate the false-positive rate of a nominal 5% t-test to as much as 16.5%. Cluster at the pair level. **SNIPPET** — [HAL hal-03873897](https://hal-sciencespo.archives-ouvertes.fr/hal-03873897)
- **Localisation metric families**: file/module/function Acc@k, Recall@k, NDCG, MAP (LocBench). **SNIPPET** — [LocAgent 2503.09089](https://arxiv.org/pdf/2503.09089)
- **Per-hop metric**: call-chain F1 plus exact match, showing a precision/recall asymmetry. **VERIFIED** — [RepoReasoner](https://www.alphaxiv.org/abs/2607.25996)
- **Atomic-fact checklist** replaces scalar judge scores and improves reliability. **VERIFIED** — [RepoProbe 2608.04783](https://arxiv.org/abs/2608.04783)
- **Project's own baseline**: on about 120 hidden tasks at a score near 0.08, SE ≈ 2.5 pp. **VERIFIED**: matches `docs/findings.md`, and sqrt(0.08·0.92/120) = 0.025.

### Power arithmetic
All numbers here are **INFERRED**: my own calculation, two-sided α=0.05, power 0.80, z-sum ≈ 2.80. Paired McNemar-style approximation: n ≈ 7.84 · d_disc / Δ², where d_disc is the fraction of items on which the two systems disagree.

**Paired comparison, items needed by effect size and disagreement rate:**

| Effect Δ | disagree 10% | disagree 20% | disagree 30% | Unpaired, p≈0.5, per arm |
|---|---|---|---|---|
| 3 pp | 872 | 1,744 | 2,616 | 4,361 |
| 4 pp | 491 | 981 | 1,472 | 2,453 |
| 5 pp | 314 | 628 | 942 | 1,570 |

**Effective n with k dense probes per task.** Design effect DE = 1 + (k−1)ρ, where ρ is the intra-task correlation.

| k probes/task | ρ | DE | eff. n from 129 tasks | eff. n from 400 tasks |
|---|---|---|---|---|
| 10 | 0.1 | 1.9 | 679 | 2,105 |
| 10 | 0.3 | 3.7 | 349 | 1,081 |
| 20 | 0.3 | 6.7 | 385 | 1,194 |
| 20 | 0.5 | 10.5 | 246 | 762 |

### Inferences
- **INFERRED**: The ~120-task hidden LB cannot resolve 3–5 pp, and neither can our 129-task outcome-only dev set. Only paired, dense, many-repo probe sets can.
- **INFERRED**: Target about 300–500 tasks × 10–20 structural probes, drawn from **≥30 unseen repos**. Analyse with paired differences and SEs clustered by repo, which is the true randomisation unit for repo generalisation, not by task.
- **INFERRED**: Probe correlation within a *repo* will likely exceed correlation within a *task*. So the number of repos, not probes, is the binding constraint for "generalises to unseen repos" claims.
- **INFERRED**: Resampling each probe 3–5 times (Miller Rec. #3) cheaply reduces within-item variance for stochastic agents.

### Gaps
- No published estimates of intra-task or intra-repo correlation (ρ) for localisation or structural probes. The ρ values above are assumptions.
- I found no paper that reports power analysis specifically for SWE or localisation benchmarks.

---

## Q4. Which evaluation gaps remain?

### Takeaway
I found **no benchmark that checks whether a model's *stated* causal or evidence chain (symptom → path → fix) is actually valid in the call graph** of the repo. Existing work does one of three things:
- (a) asks the model to *predict* a call chain as the answer (RepoReasoner, TraceEval);
- (b) uses the call graph as a *search space* for the localiser (CoSIL);
- (c) scores free-text explanations with judges or checklists (RepoProbe, CoReQA).

Chain validity as a *faithfulness* metric for agent rationales is open.

### Cited Findings
- **CoSIL** formulates localisation as iterative LLM search over a dynamically built call graph. The graph is a tool, not an evaluation of rationale validity. **SNIPPET** — [arXiv 2503.22424](https://arxiv.org/pdf/2503.22424)
- **ConFL** (*preprint* 2026): explainable concurrent fault localisation. Candidate explanations include call chains plus hypotheses, scored for support. This is a method, not a benchmark metric. **SNIPPET** — [arXiv 2608.02974](https://arxiv.org/pdf/2608.02974)
- **General CoT-faithfulness work** (Lanham et al.; EMNLP 2025 unlearning-steps paper) measures faithfulness by intervention, not by checking against a program graph. **SNIPPET** — [Anthropic PDF](https://www-cdn.anthropic.com/827afa7dd36e4afbb1a49c735bfbb2c69749756e/measuring-faithfulness-in-chain-of-thought-reasoning.pdf); [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.504.pdf)
- **AST-derived vs LLM-extracted graphs for codebase Graph-RAG**: relevant to which graph to use as the validity oracle. **SNIPPET** (title only) — [arXiv 2601.08773](https://arxiv.org/pdf/2601.08773)

### Inferences
Gap list. All items are **INFERRED** from the absence of evidence in the searches logged below.
1. **No chain-validity / faithfulness metric for SWE localisation.** A metric like "fraction of claimed edges that exist in a trace-verified call graph, and whether the chain endpoint hits gold" would be new. GRAIN does this only for toy graphs.
2. **No private / unseen-repo structural benchmark.** Every benchmark found uses public, mostly popular repos. "Fresh" means fresh tasks, not fresh code.
3. **No dense per-task probe sets attached to SWE tasks.** For example, no set that pairs each SWE-bench-style issue with 10–20 auto-generated "who calls / reachable / impacted" probes on the same snapshot.
4. **No calibration metrics** were found for localisation or structural QA. None of the benchmarks report ECE or confidence-weighted recall.
5. **No synthetic-topology code benchmark.** Nothing controls graph depth, fan-in and hub structure to separate structural skill from lexical cues.
6. **Intra-repo correlation and power are unreported**, so benchmark authors cannot say what effect sizes their benchmarks can detect.

### Gaps
- These are absence-of-evidence claims from about 14 searches. A dedicated alphaXiv full-text sweep for "chain validity" or "path faithfulness" in code might still find something. Not done, to stay within the tool budget.

---

## Search log
All queries were run on 2026-10-08. Tools: WebSearch (standard or extended) and WebFetch.

**Searches**
1. `benchmark call graph reasoning LLM repository "who calls" impact analysis 2026` (extended). Found TraceEval, LARGER, DW-Bench, GraphInfer-Bench, SWE-Atlas.
2. `SWE-QA repository-level code question answering benchmark`. Found SWE-QA (2509.14635) and the ACL 2026 Findings listing.
3. `LocBench Loc-Bench code localization benchmark recall@k`. Found LocAgent, RepoNav, LARGER, SweRank, FastCode.
4. `SWE-rebench contamination-free decontaminated evaluation results`. Found SWE-rebench and the mindstudio aggregator.
5. `synthetic procedurally generated repository benchmark code navigation unseen codebase memorization LLM` (extended). Found StackRepoQA and the class-level real-vs-synthetic paper. No procedural code-graph benchmark.
6. `"Adding Error Bars to Evals" statistical power clustered standard errors paired differences`. Found Miller 2411.00640 and the HAL pair-clustering paper.
7. `SWE-bench-Live monthly fresh issues benchmark results contamination`. Found SWE-bench-Live 2505.23419.
8. `RepoQA "search needle function" long context code understanding; CodeQueries semantic queries dataset`. Found RepoQA and CodeQueries.
9. `arXiv 2026 benchmark LLM code graph question answering call chain data flow repository structural queries static analysis ground truth` (extended). Found GABench, TraceEval, and AST-vs-LLM-KG Graph-RAG.
10. `faithfulness of LLM explanations bug localization reasoning chain verified against call graph` (extended). **This is the gap check for chain validity.** Found CoSIL, ConFL and CoT-faithfulness papers. No benchmark that scores stated chains against a call graph.
11. `LongCodeBench CoReQA CodeRAG-Bench repository comprehension benchmark`. Found CoReQA, LongCodeBench and RepoProbe. No hit for CodeRAG-Bench.
12. `SWE-Atlas codebase QnA benchmark Scale AI`. Found SWE-Atlas QnA.
13. `dependency reasoning benchmark LLM "call graph" questions repository 2026 arXiv caller callee reachability`. Found RepoReasoner, CG-Bench and "SemBench".

**Fetches**
1. https://arxiv.org/abs/2608.04783 (RepoProbe)
2. https://arxiv.org/abs/2603.26567 (StackRepoQA)
3. https://arxiv.org/abs/2605.11006 (TraceEval)
4. https://www.anthropic.com/research/statistical-approach-to-model-evals (Miller)
5. https://arxiv.org/abs/2505.20411 (SWE-rebench)
6. https://www.alphaxiv.org/abs/2607.25996.md (RepoReasoner)

**Local computation**: the power and design-effect table, computed with a Python one-liner.

**Not searched (budget)**: Multi-SWE-bench, CodeRAG-Bench primary pages, NeurIPS 2026 D&B track listings, HF dataset cards beyond SWE-Atlas.
