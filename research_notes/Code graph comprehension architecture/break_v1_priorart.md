# Breaker 3 (prior art and novelty): Behavioural Map-Matching (BMM), architecture v1, 9 Oct 2026

**Target.** `architecture_v1.md` §0.3, the claimed novel piece: an LLM imagines an issue's execution as a sequence of (symbol guess, observable effect) steps. An HMM decodes that sequence onto a repository call graph. The emissions score *behaviour*: per-function effect signatures predicted from source by a model trained on executed traces of other repos. No tests run at inference.

**What I read first.** `architecture_v1.md` (all of it, including A20), `phase3_redteam.md` (RT-1..RT-48) and `phase3_critic.md` (CR-1..CR-12). I did not repeat their queries. This note searches the gap the architect flagged (A20): log- and symptom-based path reconstruction. It also searches four families neither phase-3 log covered: behavioural code search, learned effect and exception prediction, LLM imagined traces, and repo-level "predicted vs executed path" measurement.

**Labels.**
- **VERIFIED**: I read the primary text, either the abstract via the arXiv API or Firecrawl research index, or the PDF text extracted locally.
- **SNIPPET**: from a search-engine summary, a bibliographic record or an OpenReview hit only.
- **INFERRED**: my reasoning.

**Closeness scale.** The same as RT:
- **0**: unrelated.
- **1**: same goal, different mechanism.
- **2**: same mechanism in a different domain or input, or partly done.
- **3**: same mechanism done, in substance, for this setting.

---

## 0. Bottom line

| Unit | Closest prior(s) (closeness) | Verdict |
|---|---|---|
| **(a) BMM overall**: imagined (symbol, effect) path → HMM on the call graph → node posterior ranking | Euler (FSE'19) (**2**); LocAgent's "reconstruct the execution flow" prompt (**1.5–2**); SherLog / Pensieve / Pathidea / CrashLocator / 2506.19045 / COCA (**1.5–2**); RoG/GCR and Newson–Krumm (2, already in RT) | **SURVIVE, narrowed.** No closeness-3 prior. Every *ingredient* has a closeness-2 neighbour, and the composition does not exist |
| **(b) Behavioural emissions**: effect-signature likelihood per hidden state | CrashTracker 2408.12070, exception-thrown summaries (**1.5**); EXCEPT / SBEST (1.5, RT); FaR-Loc 2509.20552, failing-behaviour description → dense retrieval (**1.5**); behaviour-by-I/O code search (Podgurski & Pierce'92, Satsy, Jiang SCAM'18) (**1**); Desc2Fix 2607.18550 (**1**) | **SURVIVE.** This is the least-crowded piece. No prior scores a structured multi-effect signature as an emission against effects stated in prose |
| **(c) Execution-trained effect predictor g_φ** | IPA-GNN runtime-error prediction 2203.03771 (**2** for the exception bits); CodeFlow 2408.02816 (**2**); TRACED 2306.07487 (1.5); D-REX / CodeHunter (1.5, static labels); JPure (1, labels only) | **NARROW.** "Predict whether and which exception a program raises, learned from execution" is done. What remains is the *per-function multi-effect* alphabet (mutation, return identity, global writes, warn, I/O), *cross-repo* transfer on real libraries, and calibration for use as an emission. Treat it as a component, not a claim |
| **(d) Imagined-path vs executed-path measurement** | RepoReasoner 2607.25996, call-chain prediction vs pytest-traced chains (**2**); SWE-Flux 2609.28449 (**1.5**); TraceEval 2605.11006 (**1.5**); REval / CoCoNUT (1, function level) | **NARROW → SURVIVE.** Repo-level "LLM-predicted vs traced call chain" now exists, but its input is a *test function* and its output is a *file set*. "From *issue prose*, function-level, against the *F2P* executed path, with divergences typed by edge kind" is unoccupied |

**Most defensible one-sentence novelty claim** (claim only after C1/C2a pass):
> *We are the first to decode an LLM's prose-derived imagined execution of a traceback-free issue onto a repository call graph using emissions that score* behaviour, *meaning per-function effect signatures predicted by a cross-repository, execution-calibrated model. We show that this beats graph propagation and names-only decoding, most of all under code-side renaming. We also measure against fail-to-pass executions where, and on which edge kinds, the imagined paths diverge.*

---

## 1. Closest priors, by family

### 1.1 Log- and symptom-based execution-path reconstruction (the A20 gap)

| Prior | What it does | Closeness to BMM | Label |
|---|---|---|---|
| **SherLog** (Yuan et al., ASPLOS'10) | From a failure's run-time log messages plus source, infers which code paths "must or may" have executed, with no re-execution. Log messages are the starting points | **2** for "reconstruct a path from sparse observations, with static analysis connecting the gaps". Its observations are *real* log lines tied to logging statements (an exact anchor), and inference is symbolic and deterministic. There is no learned or probabilistic emission and no LLM | SNIPPET (abstract page + survey) |
| **Pensieve** (Zhang et al., SOSP'17) | Event chaining: from logs, bytecode *and a description of the failure symptom*, it follows control and data dependences backwards to build a partial chain of causally dependent events that leads to the symptom | **1.5–2.** It is the closest "symptom → causal chain on program structure" ancestor. The chain is built backwards by rules over real logs, the goal is reproduction, and there is no probabilistic decoding of a hypothesised sequence | SNIPPET |
| **Pathidea** (A. R. Chen, T.-H. Chen, S. Wang, TSE 48(8) 2022) | "Improving IR-based bug localization by re-constructing execution paths using logs" in bug reports | **2** for the *localisation* use: path reconstruction from report content combined with IR ranking. Its observations are log snippets pasted into reports, not imagined steps. **I could not read the abstract (S2 rate-limited, not on arXiv).** The mechanism is inferred from the title | SNIPPET (DBLP record) |
| **CrashLocator** (Wu, Zhang, Cheung, Kim, ISSTA'14, Distinguished Paper) | Expands a partial observation (the crash stack) along the *static call graph* into approximate failing traces, then scores functions by suspiciousness. Top-1/5/10 = 50.6 / 63.7 / 67.5% on Mozilla | **1.5–2**: "complete a partial path on the call graph, then rank". The observation is an executed stack, the expansion is by call depth with no emission model, and no prose is involved | SNIPPET (+ arXiv 1404.4100 summary) |
| **2506.19045** (Saboor Yaraghi et al., TSE'26) | Maps one failure log to statements and estimates the execution trace (≈90% F1), then has an LLM rank faulty locations | 1.5 (as RT). Real log, test code | VERIFIED (abstract) |
| **COCA** 2503.23051 | From the data in issue reports, it extracts code snippets and *reconstructs execution paths*, then has an LLM localise components | **1.5–2.** It is the closest *issue-report* path reconstruction. The paths are reconstructed from logs and stack data in the report by static analysis, not imagined, and are used as LLM context rather than decoded | VERIFIED (abstract) |
| TELLER 2608.01975 | Aligns log lines to reconstructed per-request call-chain trees | 1 (traces collected) | VERIFIED (abstract) |

**Verdict on A20.** The family stays at **≤2**, as the architect estimated. Every member observes real execution artefacts: log lines, stack frames or crash stacks. Each anchors those observations to code with an exact or near-exact map (log statement, frame name) and reconstructs paths deterministically. **BMM's delta** has three parts: (i) observations that are *imagined from prose* with no runtime artefact, (ii) a *soft, learned behavioural* observation model in place of exact anchors, and (iii) exact probabilistic inference rather than rules. (INFERRED)

**Hostile point.** This family shows that "reconstruct the execution path from symptoms, statically" is a 15-year-old SE idea. Reviewers from ICSE/FSE will see BMM as "SherLog/Pensieve with an LLM as the log". The paper must cite SherLog, Pensieve, Pathidea and CrashLocator in the first related-work paragraph, not only map-matching.

### 1.2 Natural-language step sequences grounded onto a program-execution graph

| Prior | What it does | Closeness | Label |
|---|---|---|---|
| **Euler** (Chaparro et al., ESEC/FSE'19, 1906.07107) | Identifies a bug report's steps to reproduce (S2R), *a sequence of NL steps*. It matches each step to program states and GUI interactions "represented in a graph-based execution model" built by app exploration. Matching uses textual similarity (term-level LCS). When a matched state is far from the current state, it "selects the shortest path between the current state and the state where the interaction occurs", inferring missing steps. Steps that match several components are flagged ambiguous; steps that match none are flagged as vocabulary mismatch | **2.** It is the closest *structural* ancestor of BMM's decode: an NL-described step sequence aligned onto a program-execution transition graph, with gap filling along graph paths. The differences are a GUI-state graph rather than a call graph, greedy step-by-step matching with shortest-path fill rather than an HMM, names and text only (no behaviour emissions), human-written steps rather than LLM-imagined ones, and a goal of report quality rather than localisation | **VERIFIED** (PDF text) |
| Yakusu, ReCDroid, AdbGPT (S2R → GUI action reproduction) | Search the GUI state graph to realise NL steps | 1.5 (same pattern, GUI, reproduction) | SNIPPET (cited in Euler; not re-read) |
| LadyBug 2508.05085 | Combines a recorded UI reproduction trace with report text for file ranking | 1 (executed trace) | VERIFIED (abstract) |
| GALA 2604.08089 | Aligns a UI graph extracted from screenshots to call graphs to ground visual elements in functions | 1.5 (graph-to-graph alignment for localisation; no sequence decoding) | VERIFIED (abstract) |

**Consequence.** Euler shows that "NL steps → alignment on an execution graph → infer the missing steps along paths" exists in SE. BMM's decoder is therefore neither new to SE (Euler) nor to ML (Newson–Krumm, RoG/GCR). Cite Euler as the SE ancestor of the decode. (INFERRED)

### 1.3 LLM "imagined trace" or "mental execution" for localisation

| Prior | What it does | Closeness | Label |
|---|---|---|---|
| **LocAgent** (ACL'25), localisation prompt | The prompt's Step 3 is "Reconstruct the execution flow": identify the main entry points that trigger the issue, trace function calls, class interactions and sequences of events, and identify potential breakpoints. Step 2 says to use the "described execution flow" to tell apart modules with similar names. Grounding is by agent tool calls on the code graph | **1.5–2** for the *imagined path from the issue* half. The imagined flow is free-text chain of thought consumed by the agent. It is never emitted as a structured (symbol, effect) sequence, never decoded probabilistically, and has no behavioural channel | **VERIFIED** (PDF prompt text) |
| CoSIL 2503.22424; SWE-Debate 2507.23348; GraphLocator 2512.22469 | Iterative LLM call-graph search; fault-propagation traces built *by graph traversal* from issue-matched entities; a causal issue graph grown from symptom vertices | 1.5–2 (as RT). SWE-Debate's chains are graph-first, not LLM-first. GraphLocator's "symptom vertices" are located by LLM matching, not by effect signatures | VERIFIED (abstracts) |
| SemLoc 2603.29109 | "Structured grounding of free-form LLM reasoning": LLM-inferred properties are bound to typed program anchors, then checked at runtime | 1.5 for "turn free-form LLM hypotheses into a structured IR grounded on program structure". It *executes* the program, and it works on single programs | VERIFIED (abstract) |
| CogniGent 2601.12522; SHERLOC 2606.24820; OrcaLoca; SpecRover/AutoCodeRover | Hypothesis-driven agentic localisation with call-graph root-cause analysis | 1 | VERIFIED (abstracts; OrcaLoca and SpecRover SNIPPET) |
| HyDE-style generation-then-retrieve (critic) | The LLM imagines the target, then retrieval follows | 1 | (CR) |

**Consequence.** "Ask the LLM to imagine the execution flow from the issue" is a *prompting step already shipped* in LocAgent, a SWE-bench-era system. BMM cannot claim the imagined path itself. It can claim (i) emitting the path as a typed (symbol, effect) observation sequence and (ii) decoding it exactly against a calibrated graph rather than consuming it as context. (INFERRED)

### 1.4 HMM / Bayesian / graphical-model fault localisation on program structure

| Prior | What it does | Closeness | Label |
|---|---|---|---|
| Dietz, Dallmeier, Zeller & Scheffer, NeurIPS'09 | A graphical model of transitional patterns over code positions, trained on passing traces. The least likely transition in a failing trace is the bug | 1.5: probabilistic sequence model over code-position transitions for localisation. The observations are executed traces, and nothing is imagined or decoded onto a graph | SNIPPET |
| Probabilistic program dependence graph (Baah, Podgurski & Harrold, ISSTA'08/TSE'10); Error Flow Graph (Feng & Gupta, PASTE'10); SmartFL (ICSE'22); **Prosecutor** (OOPSLA'26, Error Propagation Graph + counterfactual branch flips) | Bayesian or factor-graph inference over program dependence structure, using test executions | 1 (inference over program graphs, but over executed spectra; no LLM observation sequence) | SNIPPET |
| HMM FL for integration testing (Ge, Nakajima & Pantel) | Component fault-state HMMs | 1 | SNIPPET |
| ViterbiPlanNet (CVPR'26) | Differentiable Viterbi over a procedural-knowledge graph of action transitions, decoding the hidden action sequence from observed events | 1.5 for the decoder idea in another domain. No LLM, no code | SNIPPET |

Exact searches found **no HMM or Viterbi decoding of hypothesised symbol sequences onto a code graph**: BR3-3, BR3-14 and BR3-20, plus OpenReview "Viterbi code localization" (BR3-21). This confirms RT-28/32 and CR-9 with new terms (absence).

### 1.5 Behavioural or semantic code search and effect-based localisation (unit b)

| Prior | What it does | Closeness to behavioural emissions | Label |
|---|---|---|---|
| **CrashTracker** 2408.12070 | Static *exception-thrown summaries* (ETS) per framework exception, followed by data tracking from crash to buggy candidates, then LLM explanation | **1.5**: a structured, per-site exception-semantics summary used to localise. It needs a crash stack, covers exceptions only, uses no learned multi-effect signature and decodes no sequence | VERIFIED (abstract) |
| **FaR-Loc** 2509.20552 | An LLM turns the failed test and stack into a "description of failing behavior", then dense retrieval finds "methods with similar functional behavior" among covered methods | **1.5**: retrieval by *described behaviour*. The behaviour is a free-text embedding, not a typed effect signature. It needs a failing test and coverage | VERIFIED (abstract) |
| EXCEPT 2201.00736; SBEST 2405.00565; ExLocator; EXPECT (ASE'24) | Localisation by exception type or source, or trace proximity | 1.5 (RT/CR) | SNIPPET |
| Behaviour-by-example code search: Podgurski & Pierce'92 ("behavioral sampling"); Satsy (Stolee); Jiang et al. SCAM'18 (SMT over path constraints for I/O examples) | Retrieve methods whose *behaviour* satisfies a query | **1**: same "search by behaviour, not names" goal. The queries are I/O examples, the mechanism is SMT or execution, and the setting is not localisation | SNIPPET |
| Desc2Fix 2607.18550 | LLMs extract structured "behavioral anchors" (reproduction steps, API/exception cues, expected vs actual) from reports and measure alignment with tests and patches | **1** (a measurement; it supports BMM's parse step and coverage statistic; no localisation) | VERIFIED (abstract) |
| Chaparro et al. ICSME'17 (observed behaviour in reports for query reformulation) | Uses report OB/EB sentences to reformulate IR queries | 1 (from memory, not re-searched; INFERRED) | INFERRED |

**Remaining delta (INFERRED).** No prior combines:
- a fixed, name-free *effect alphabet* (exception classes, return identity, mutation, self/global writes, warn, I/O);
- per-function *predicted* probabilities over that alphabet;
- a *likelihood* term against effects stated in issue prose;
- a *sequence* decoder that multiplies those likelihoods along a connected path.

The closest (CrashTracker, FaR-Loc) need an executed failure artefact and use one channel (exceptions) or free text.

### 1.6 Learned effect and exception prediction (unit c)

| Prior | What it does | Closeness to g_φ | Label |
|---|---|---|---|
| **IPA-GNN runtime-error prediction** 2203.03771 (Bieber et al.) | An interpreter-inspired model that "models exception handling" and predicts *whether and which kind* of runtime error a program raises, statically, with labels from execution. It localises errors without location labels | **2** for g_φ's 9 exception bits: learned static prediction of raised exception kind, supervised by execution. It works on whole competitive-programming programs, has no non-exception effects and no per-function cross-repo use | VERIFIED (abstract) |
| **CodeFlow** 2408.02816 | Learns static CFG plus *dynamic dependencies from execution traces* to predict coverage and runtime errors | **2** (execution-trained behaviour predictor; statement level; no effect alphabet) | VERIFIED (abstract) |
| TRACED 2306.07487 | Execution-aware pre-training; estimates execution paths and variable values statically | 1.5 | VERIFIED (abstract) |
| D-REX (Java exception-type prediction, 81% top-1); CodeHunter (AAAI'25); ThEx | Static exception-type prediction with labels from code, not executions | 1.5 | SNIPPET |
| ThrowBench 2503.04241; 2604.16320; Cerberus 2512.21431 | LLM prediction of thrown exceptions (benchmarks or prompting) | 1 | VERIFIED (abstracts) |
| JPure (dynamic purity, Dallmeier); observation-based purity patents | Side-effect labels from executions; no learning | 1 | SNIPPET |
| LExecutor 2302.02343 | Neural value prediction trained on executions so that code can run | 1 | VERIFIED (abstract) |

Searches for a *learned side-effect / mutation / purity predictor trained on dynamic labels* came back **empty** (BR3-15, BR3-27: absence). The exception sub-alphabet is closeness 2. The mutation, return-identity, global-write, warn and I/O bits have no learned prior found. They are, however, the bits an AST rule set captures best (the critic's X-ASTEFF point), so their *execution-calibrated* increment is the most at risk of being empirically null. (INFERRED)

### 1.7 LLM-predicted vs executed path measurement (unit d)

| Prior | What it measures | Closeness | Label |
|---|---|---|---|
| **RepoReasoner** 2607.25996, *Call Chain Prediction* | Given a *target test function* and a file pool (signal + distractors), predicts the list of files invoked when the test runs. Ground truth comes from **pytest dynamic tracing** (the `hunter` library). Models have high precision (≈0.80) but low recall (≈0.31); best oracle F1 0.663; best exact match ≈0.21 | **2**: repo-level, LLM-predicted vs executed call chain, Python. Its input is a *test*, not an issue; it is file-level and set-scored; it does not decode onto a graph; it types no divergences | **VERIFIED** (abstract + HTML task section) |
| SWE-Flux 2609.28449 | Repo-level runtime QA (control flow, exceptions, inter-procedural, state) with gold answers from instrumented test runs. Best 37%; inter-procedural reasoning is weak | 1.5 | VERIFIED (abstract) |
| TraceEval 2605.11006 | Execution-verified call-graph recovery from source, 10.6k programs in 3 languages, plus a train split (Qwen2.5-Coder-32B tuned to 71.2 F1) | 1.5. Also relevant to C3/infrastructure: it is a ready *execution-verified call-edge* training set that competes with CHAIN-lite's data story | VERIFIED (abstract) |
| REval, CoCoNUT, CodeMind, E.T. 2503.05703 | Function-level execution-path or state prediction | 1 | SNIPPET / VERIFIED (E.T. abstract) |

**Remaining delta (INFERRED).** Issue prose → function-level ordered path → compared with the *F2P executed* path. Report the first-divergence position, typed by HEAP edge kind (decorator, registry, MRO, `__getattr__`, generator, re-export), on traceback-free issues. RepoReasoner's "high precision, low recall" result is a **prior expectation** for M1 (the imagined path finds a few correct anchors and misses multi-hop middles). The paper must cite it, and it lowers M1's surprise.

---

## 2. Verdicts

### (a) BMM overall: **SURVIVE (narrowed), combined closeness 2**
- No prior has (i) the LLM emit a typed imagined path, (ii) decode it with exact HMM inference on a calibrated call graph, (iii) with behavioural emissions. **VERIFIED absence** at the abstract level across ≈30 queries. This is not proof of absence; see §5.
- The four closeness-2 ancestors are in four different communities:
  - Euler (SE, NL steps on an execution graph);
  - SherLog, Pensieve and Pathidea (systems/SE, symptom → path);
  - Newson–Krumm (GIS, HMM decode);
  - RoG/GCR (KGQA, LLM plan → graph).
- That makes BMM a defensible *synthesis*, but a reviewer can write "incremental combination" unless C2a (behaviour adds, especially under R2) is positive. **If C2a dies, BMM collapses to closeness 2.5** (Euler + LocAgent + map-matching), which is too thin for a novelty claim. (INFERRED)

### (b) Behavioural emissions: **SURVIVE, closeness 1.5**
- This is the true novel piece. It is the only unit with no closeness-2 prior in the localisation setting.
- Effect-signature-likelihood emissions are new in two respects: (i) they match effects stated in prose to *predicted* function effects, and (ii) the matches compound along a path.

### (c) Execution-trained effect predictor: **NARROW, closeness 2 (exception bits) / 1.5 (overall)**
- IPA-GNN and CodeFlow already learn static runtime-error and behaviour prediction from execution labels.
- Claim only "a cross-repository, per-function, multi-effect predictor calibrated for use as an emission". The C2b result (vs AST rules) is its *only* evidence. If C2b fails, drop "execution-calibrated" from the headline sentence.

### (d) Imagined-vs-executed measurement: **NARROW → SURVIVE, closeness 2**
- RepoReasoner occupies "LLM call chain vs traced chain at repo level".
- BMM's version (issue → function path vs F2P path, with divergence typing) is distinct and still publishable.
- It must be framed as extending RepoReasoner and SWE-Flux to *issue-conditioned* paths, not as a first measurement.

---

## 3. Must NOT claim

1. The first to have an LLM imagine or reconstruct an issue's execution flow. LocAgent's prompt does this (VERIFIED). CoSIL, GraphLocator and SWE-Debate build chains.
2. The first to reconstruct execution paths from failure symptoms without execution: SherLog, Pensieve, Pathidea, CrashLocator, 2506.19045, COCA.
3. The first to align a natural-language step sequence onto a program-execution graph with path gap filling: Euler (VERIFIED).
4. HMM/Viterbi/map-matching, or probabilistic/Bayesian fault localisation over program structure, as new: Newson–Krumm; Dietz et al. NeurIPS'09; PPDG; SmartFL; Prosecutor.
5. The first exception- or effect-driven localisation: EXCEPT, SBEST, CrashTracker (ETS), ExLocator, EXPECT.
6. The first static prediction of raised exceptions learned from execution: IPA-GNN 2203.03771, CodeFlow 2408.02816; and the first execution-aware code representations (TRACED).
7. The first measurement of LLM-predicted vs executed call paths at repository level: RepoReasoner, SWE-Flux, TraceEval.
8. "Search by behaviour rather than names" as a new idea: Podgurski & Pierce'92, Satsy, Jiang SCAM'18, FaR-Loc.
9. "Test-free" or "execution-free" localisation as new: LLMAO (test-free FL), SHERLOC, Agentless. Also, BMM's HEAP *does* execute module top-level code, so call it "no tests at inference", not "no execution".
10. Execution-verified call-edge data as a contribution: TraceEval releases a train split.

---

## 4. What the paper should add because of this search (INFERRED)

- **Related work, first paragraph:** SherLog → Pensieve → Pathidea → CrashLocator → COCA/2506.19045 ("path from symptoms"). Then Euler ("NL steps on an execution graph"). Then Newson–Krumm and RoG/GCR (decoder and plan-then-ground). Then LocAgent ("imagined flow as prompt"). Then IPA-GNN and CodeFlow (execution-trained behaviour prediction). Then RepoReasoner and SWE-Flux (measurement).
- **One cheap extra baseline that this search makes obvious:** **Euler-style greedy alignment.** Match each imagined step to its best node by name+effect, fill gaps by shortest path on G, and rank by visit. It is one change from BMM (greedy vs HMM inference) and isolates "exact probabilistic decoding" from "align-and-fill". X-BAG and X-SHUF do not cover it. CPU cost is zero.
- **A CrashLocator-style baseline on traceback-free issues is not applicable**, since there is no stack. But on the *traceback* stratum (reported, not headline), CrashLocator-style stack expansion on G is the fair comparator, not BM25.
- **Use RepoReasoner's precision/recall split** for M1 (per-step precision vs path recall), so the numbers are comparable.

---

## 5. Gaps (not searched or not resolved)

- **Pathidea's abstract was not read** (Semantic Scholar returned 429 three times; IEEE is not fetched). Its closeness of 2 is inferred from the title. If it reconstructs paths from *NL* descriptions rather than log snippets, it rises to 2.5 for (a). **This is the highest-priority follow-up read.**
- Pensieve and SherLog were read only at abstract or summary level. Neither uses an LLM or prose-only input (SNIPPET).
- Euler's follow-ups (Chaparro's group, 2020–2026; e.g. LLM-based S2R → GUI-graph matching such as AdbGPT, and "BURT") were not swept. One of them could pair an LLM with a probabilistic alignment on a GUI graph, which would raise (a) to 2–2.5 *in the GUI domain*.
- ICSE/FSE 2026–27 papers not yet on arXiv, and ICLR 2027 submissions beyond the search API, cannot be excluded.
- CodeMonkeys and Agentless were checked only by summary. Neither is known to emit structured imagined paths (SNIPPET).

---

## Search log (BR3-n)

Tools:
- **AX**: the arXiv export API (https).
- **OR**: OpenReview `api2.openreview.net/notes/search`, 7 s pacing.
- **WS**: WebSearch (standard or extended).
- **WF**: WebFetch.
- **FCR**: Firecrawl research paper search.
- **PDF**: PDF text extracted locally with pypdf.
- **S2**: the Semantic Scholar API, which returned 429 throughout.

| ID | Tool | Query / target | Result used |
|---|---|---|---|
| BR3-1 | AX | `abs:"execution path" AND abs:log AND abs:reconstruct`; `abs:"hidden markov" AND abs:"fault localization"`; `abs:"bug report" AND abs:"execution path"`; `abs:"side effect" AND predict AND code AND learn`; `abs:"call graph" AND abs:viterbi`; `exception … "bug report" … localization`; `"issue localization" AND simulate`; `"map matching" AND program` | 4/1/1/9/0/0/2/4 hits, all irrelevant. **Absence** of a Viterbi-on-call-graph paper |
| BR3-2 | WS | SherLog error diagnosis, execution path inference | SherLog ASPLOS'10 (SNIPPET); LogEnhancer |
| BR3-3 | WS ext | bug report symptom → code location without stack trace, LLM execution path inference | LLMLoc, ConFL, DEVLoRe, 2506.19045; no imagined-path decoder |
| BR3-4 | WS | HMM fault localization call graph program trace | Dietz et al. NeurIPS'09; Ge et al. HMM-FL; Prosecutor OOPSLA'26; Feng & Gupta EFG; no HMM-on-call-graph (absence) |
| BR3-5 | WS | Pensieve event chaining | Pensieve SOSP'17 (SNIPPET); Anduril SOSP'24 |
| BR3-6 | WS | Prosecutor Bayesian counterfactual FL | EPG + branch flips, Defects4J (SNIPPET) |
| BR3-7 | WS ext | LLM predicts trace from bug report, aligns to call graph | IssueExec, LLM4FL, AgentFL, LocAgent, CoSIL; **no predict-then-align** |
| BR3-8 | WS | code search by behaviour / execution semantics / side effects | Podgurski & Pierce behavioural sampling; no effect-indexed retrieval (absence) |
| BR3-9 | WS | neural exception prediction learned from execution | D-REX, CodeHunter, ThEx, CatchAll, **IPA-GNN 2203.03771**, LExecutor |
| BR3-10 | OR ×8 | log execution path reconstruction; bug report execution path; fault localization hidden markov; behavioral code search; side effect prediction code; exception prediction neural; imagined execution localization LLM; symptom fault localization LLM | Noise. Wang et al. ICSE'08 (execution info for duplicate reports); SmartFL; exception-flow analysis SCAM'17; nothing on imagined-path or effect decoding |
| BR3-11 | WS | Pathidea | Not found by name |
| BR3-12 | WS ext | LLM hypothesised call chain grounded against the call graph, SWE-bench 2026 | SHERLOC, CoSIL, LocAgent, CORTEXA; none |
| BR3-13 | WS | 2506.19045 | Confirms RT-29 |
| BR3-14 | AX | `ti:Pathidea`; `ti:"execution paths" AND ti:logs`; `"bug report" "call graph" hypothes*`; `issue "call chain" LLM localization`; effect + FL + NL; `ti:"execution semantics" retrieval`; exception learned dynamic Python | CogniGent 2601.12522; ABB 2604.25700; LExecutor; nothing closer |
| BR3-15 | AX id_list | 2601.12522, 2604.25700, 2203.03771, 2512.22469 | Abstracts VERIFIED |
| BR3-16 | WS ext | repo-level benchmark: LLM call path vs executed trace | **SWE-Flux 2609.28449, RepoReasoner 2607.25996, TraceEval 2605.11006** |
| BR3-17 | WS | "Pathidea" Chen TSE | DBLP: TSE 48(8):2905–2919, 2022 (SNIPPET) |
| BR3-18 | WS | Dietz et al. NeurIPS'09 | Bernoulli graph model over transitional patterns (SNIPPET) |
| BR3-19 | AX id_list + S2 | 2609.28449, 2607.25996, 2605.11006, 2606.24820; S2 "Pathidea" | Abstracts VERIFIED; S2 429 |
| BR3-20 | WS ×3 | learned purity / side effects from dynamic labels; ThrowBench; observed-behaviour bug localisation, exception matching | JPure, purity patents (**no learned predictor; absence**); ThrowBench numbers; nothing academic on raise-site matching |
| BR3-21 | WS ×3 | Euler S2R execution model; log-to-logging-statement FL with LLM; LLM plan grounding Viterbi KG | Euler; FALCON ICSE'25, LogConfigLocalizer ISSTA'24; **ViterbiPlanNet CVPR'26**; no HMM-grounded LLM KG path |
| BR3-22 | WF + PDF | arXiv 1906.07107 (Euler) | **VERIFIED: S2R sequence matched to a graph-based execution model; shortest-path gap filling; LCS similarity** |
| BR3-23 | WF + PDF | LocAgent ACL'25 PDF | **VERIFIED prompt: "Reconstruct the execution flow … trace function calls … sequences of events"** |
| BR3-24 | S2 ×3 | Pathidea; CrashLocator; SmartFL | 429 (failed) |
| BR3-25 | FCR | LLM imagines/predicts execution path from bug report, aligned onto call graph | LLmiRQ, CoSIL, SieveFL, AutoFL, **GALA**, CogniGent, ConFL, **FaR-Loc**, GenLoc, **CrashTracker**, **SemLoc**, FlexFL, 2506.19045, BAP, RGFL (abstracts VERIFIED via index) |
| BR3-26 | FCR | predict function side effects / exceptions from source, trained on runtime traces | **CodeFlow 2408.02816**, IPA-GNN, ThrowBench, CRANE-LLM, Cerberus, E.T., TRACED, LExecutor, Neural Change Prediction 2606.03378 (abstracts VERIFIED) |
| BR3-27 | FCR | runtime-behaviour symptoms in issue text → code, no stack traces | IssueExec, LadyBug, **Desc2Fix 2607.18550**, SweRank, IQLoc, GraphLocator; no effect-signature matching |
| BR3-28 | WS ×3 | OrcaLoca/SpecRover/CodeMonkeys; CrashLocator; I/O-example behavioural code search | OrcaLoca (distance-aware pruning), SpecRover (spec inference); **CrashLocator ISSTA'14**; Satsy, Jiang SCAM'18 |
| BR3-29 | FCR | LLM hypothesised trace + HMM/Viterbi/beam alignment to program graph | TraceRepair, TraceCoder, GALA, DebugRepair, AutoTrace, PathFix; **none decode imagined sequences** |
| BR3-30 | OR ×5 | Viterbi code localization; symptom → root cause code graph; execution-free FL issue; effect type inference learned; behaviour signature function retrieval | Noise. LLMAO "LLMs for test-free FL"; SmartFL ICSE'22; no hit |
| BR3-31 | AX id_list | 2503.23051, 2507.23348, 2408.02816, 2607.18550 | COCA, SWE-Debate, CodeFlow, Desc2Fix abstracts VERIFIED |
| BR3-32 | WF | arXiv HTML 2607.25996 (RepoReasoner) | **VERIFIED**: input = target test function + file pool; output = file list; truth from pytest `hunter` tracing; precision ≈0.80, recall ≈0.31, oracle F1 0.663 |

No file other than this one was modified.
