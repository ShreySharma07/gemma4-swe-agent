# Breaker 3b (prior art, gap closure): BMM v2 "polarity-aware behavioural emissions", 9 Oct 2026

**Target.** `architecture_v2.md` §1 novel piece. Each step of an LLM's imagined execution path carries the effects the issue *reports* (E⁺) and the effects it *expected but did not see* (E⁻). These are scored as likelihood ratios against repo base rates q̄_j, using per-function effect probabilities q̃_v from a cross-repo, execution-trained model g_φ. The E⁻ term also has a peer-contrast part, peer_j(v): "siblings raise, v does not" (§3.3). The scores are compounded along the path by exact HMM decoding on the call graph.

**Scope.** This pass closes the five gaps that `break_v1_priorart.md` §5 left open. I did not repeat BR3-1..32.

**Labels.**
- **VERIFIED**: I read the primary text (the PDF text extracted locally, or the abstract via the arXiv API or the Firecrawl research index).
- **SNIPPET**: from a search summary, slides or a bibliographic record only.
- **INFERRED**: my reasoning, or recall not re-checked.

**Closeness scale** (as in RT/BR3):
- **0**: unrelated.
- **1**: same goal, different mechanism.
- **2**: same mechanism in another domain or input, or partly done.
- **3**: done in substance for this setting.

---

## 0. Bottom line

| # | Gap | Closest prior (closeness to the **novel piece**; closeness to BMM overall) | Verdict for that gap |
|---|---|---|---|
| 1 | Pathidea | Chen, Chen & Wang, TSE 48(8) 2022 (**0.5**; **2**). VERIFIED, PDF | It reconstructs paths **only from logs and stack traces** pasted in reports, never from NL description. It does not rise to 2.5 |
| 2 | S2R / LLM on GUI or state graphs | **Zhang, Tawsif, Ryu, Yu & Halfond, FSE'24, "Mobile Bug Report Reproduction via Global Search on the App UI Model"** (**1**; **2.5 for the decoder**). VERIFIED, PDF. Also AstroBR 2502.04251 (1; 2), BugScribe 2604.01148 (1; 1) | **New closeness-2.5 ancestor for the decode.** It is a Markov model over (step, UI-event) pairs, with DP top-k best paths and missing-step gap filling. It is not on a call graph, and its emissions are text only |
| 3 | Negative evidence / "expected but not observed" | In SE: Chaparro et al. ICSME'17 / EMSE'19, OB/EB query reformulation (**1**); software MBD (Wotawa et al.; CFaults) (**1.5**); **Chucky**, CCS'13, peer-contrast missing checks (**2** for the peer_j(v) term). Outside SE: **QMR-DT/Quickscore** noisy-OR positive and negative findings (**2**); **possibilistic MBD**, Cayrac–Dubois–Prade, effects "certainly present / absent" (**2**) | **Polarity is not new as an inference idea.** No SE localisation prior scores code against *issue-stated, expected-but-absent effects* (absence) |
| 4 | LR / base-rate normalised scoring of code | **Liblit et al. PLDI'05, Increase(P) = Failure(P) − Context(P)** (**2**: base-rate-normalised scoring of code predicates); Sisman & Kak MSR'12, prior × IR posterior (1.5); IDF/BM25 (1.5; the arch itself calls E⁺ "IDF-like"); Landsberg "Doric" probabilistic FL (1) | **Base-rate-normalised scoring of code elements is not new.** What is new is its *object*: a learned per-function effect probability against prose-stated effects |
| 5 | 2025–26 SWE localisation using OB/EB | 2607.07593 and 2607.09553 (measurements: EB matters to agents) (**1**); BLAgent 2605.17965 ("behavioral signals" query transform) (**1**); Desc2Fix (1); FaR-Loc (1.5) | **Absence.** No SWE-era localiser parses issues into signed effects and scores functions with them |

**Final verdict on the novel piece: NARROW (not KILL).**
- **Closeness of the novel piece: 1.5 → 2.** Every sub-mechanism now has a closeness-2 neighbour:
  - signed findings with likelihoods: QMR-DT, possibilistic MBD;
  - base-rate-normalised scoring of code elements: Liblit's Increase;
  - "peers do X, v does not": Chucky;
  - probabilistic sequence alignment of NL report steps onto a program graph with missing-step fill: Zhang et al. FSE'24.
- **Whole architecture: 2 → 2–2.5.** The decoder's SE ancestor moves from Euler (greedy) to Zhang et al. FSE'24 (Markov model + DP).
- **What no prior has, all at once** (absence at abstract level, ≈25 new queries):
  - (i) effects **stated in issue prose** with polarity;
  - (ii) scored against **predicted, execution-trained per-function effect probabilities** over a fixed effect alphabet, not text similarity;
  - (iii) as **likelihood ratios against the repo's base rates**;
  - (iv) **compounded along a call-graph path** by exact inference;
  - (v) for **issue localisation** with no tests at inference.

**Updated most-defensible one-sentence claim** (only if C-BEH passes):
> *We turn the observed and expected-but-unobserved behaviour stated in an issue into signed observations of an LLM-imagined execution path, and score each against per-function effect probabilities predicted by a cross-repository, execution-trained model (as likelihood ratios to repository base rates). Decoding them on the call graph improves function-level localisation over the same path decoded with names and body text alone, most of all under code-side renaming.*

Drop "first" for polarity, likelihood ratios and decoding individually. The only "first" that survives is the conjunction (i)–(v), and only at abstract-level absence.

---

## 1. Pathidea (gap 1)

**Source.** A. R. Chen, T.-H. Chen, S. Wang, "Pathidea: Improving Information Retrieval-Based Bug Localization by Re-Constructing Execution Paths Using Logs", IEEE TSE 48(8) 2022 (online 2021). I read the author PDF (petertsehsun.github.io/papers/PathideaTSE2021.pdf): abstract, §3 approach and Algorithm 1. **VERIFIED.**

**How it works:**
- **Input to path reconstruction:**
  - log snippets, found by regex (timestamp + verbosity level + fully-qualified class name);
  - stack-trace lines (`at …(X.java:n)`), extracted from the bug report.
- **The NL description is not used for paths.** Summary and description feed only the VSM/IR score, which uses segmented files of 800 tokens, BRTracer-style.
- **Path reconstruction:**
  - map log lines to methods/files and verify they exist in the repo;
  - build a method-level call graph from JavaParser ASTs;
  - `findPathsBetween(log_i, log_{i+1})` does **BFS between consecutive log anchors** and records every connecting path;
  - deduplicate repeated log templates (loops, threads).
- **Scoring:**
  - LogScore: 1/rank for stack frames ≤10, 0.1 beyond that, and α = 0.1 for log-snippet files;
  - PathScore(f) = β·N(VSM(f, br)), with β = 0.2, for files on any path. It is a **binary on-path boost** of the IR score;
  - file-level only.
- **Evaluation:** 1,273 reports *with logs*, 8 Java systems. MAP +8–21% and MRR +5–21% over BRTracer.

**Closeness.**
- **To BMM overall: 2**, confirmed. The shape matches: an ordered sequence of anchors from report content, gaps filled along the static call graph, then used for localisation.
- **To the novel piece: 0.5.**
  - The anchors are exact (log statement → class).
  - There is no emission model and no behaviour or effect channel.
  - There is no polarity.
  - The path is a hard set boost, not a posterior.

**Delta (INFERRED).** BMM's observations are imagined from prose with no runtime artefact. Its anchors are soft (learned effect probabilities plus names, not log-template identity). Its gap filling is probabilistic (propensity-weighted transitions and insert states, not unweighted BFS enumeration). It works at function level on traceback-free issues, which is exactly the population Pathidea excludes.

**Consequence.** The v1 worry (NL-description paths → 2.5) is **resolved negatively**. Pathidea stays in related-work paragraph 1, described precisely as "log-anchored BFS path reconstruction for IRBL".

## 2. Steps-to-reproduce and LLMs on GUI/state graphs (gap 2)

| Prior | Mechanism | Closeness (novel piece; BMM overall) | Label |
|---|---|---|---|
| **Zhang, Tawsif, Ryu, Yu, Halfond, FSE'24** (PACMSE 1(FSE) Art. 117, doi 10.1145/3660824) | "Redefines bug report reproduction as a **Markov model**" (PRM). States are (S2R step, UI event) pairs. Transitions go s_i → s_{i+1} only if e_y is reachable from e_x in the UI transition graph. The transition "probability" is a step–widget **text-embedding similarity** (max over label, content-desc and resource-id) plus 1/shortest-path distance. **Dynamic programming top-k most-probable paths** give a global match. **Missing steps** are filled by shortest paths before the first step and between consecutive matched events, and by fixed-depth exploration after the last step. 94% reproduction on 72 reports, 93% on reports with missing steps | **1; 2.5 for the decoder.** It is essentially Newson–Krumm map-matching of NL report steps onto a program-derived graph. It differs from BMM in five ways: (a) GUI transition graph, not call graph; (b) human-written S2R, not LLM-imagined; (c) text-similarity emissions only (no behaviour, no polarity); (d) heuristic additive scores, not normalised probabilities; (e) the goal is reproduction, not localisation | **VERIFIED** (PDF §3.1–3.3) |
| AstroBR 2502.04251 (Chaparro group) | An LLM extracts S2Rs and maps them to GUI interactions in a dynamically built execution model. It traverses that model for the best path covering all steps and suggests missing steps | 1; 2 (LLM + graph path, greedy or LLM-guided, GUI) | VERIFIED (abstract); the path detail is SNIPPET |
| BugScribe 2604.01148 | Grounds the report in UI execution info to generate OB/EB/S2R | 1; 1 (OB/EB polarity exists as report *structure*, but nothing is scored against code) | VERIFIED (abstract) |
| BURT 2209.10062 | A chatbot that guides OB/EB/S2R reporting against an app execution model | 0.5; 1 | VERIFIED (abstract) |
| TreeMind 2509.22431; BugRepro 2505.14528; LTGDroid 2603.29623; AdbGPT 2306.01987; ReBL 2407.05165; CrashTranslator 2310.07128 | LLM + MCTS, RAG + UI transition graph, visual-effect pre-assessment, prompting, feedback loops. All execute the app | 0.5–1; 1–1.5 | VERIFIED (abstracts) |
| GIFdroid 2112.04128 | Video keyframes → UTG states → execution trace | 0.5; 1.5 (sequence → graph states) | VERIFIED (abstract) |
| Yakusu (ISSTA'18), ReCDroid (ICSE'19), ReproBot (ISSTA'23) | NLP or ontology extraction of S2R, then guided dynamic exploration or RL | 0.5; 1–1.5 | SNIPPET / INFERRED (recall, not re-read) |
| GALA 2604.08089 (from BR3) | UI graph ↔ call graph alignment | 1; 1.5 | VERIFIED (BR3) |

**Absence.** No work maps NL bug-report steps onto a **call graph** (rather than a GUI or state graph) with probabilistic sequence alignment (BR3b-2, -6, -7, -9). Zhang et al. FSE'24 is the first SE paper I found that frames report-step alignment as a Markov model solved by DP. **It replaces Euler as the decoder's closest SE ancestor**, and it pre-empts "probabilistic decoding of report steps with missing-step inference" as an SE idea. (INFERRED)

**Action for the paper (INFERRED).**
- Cite FSE'24 next to Euler and Newson–Krumm.
- X-NOEFF+BODY is in substance "the FSE'24 PRM transplanted to call graphs, with LLM-imagined steps". Name it that way in the paper so reviewers see the right ancestor already ablated.

## 3. Negative evidence: "expected but not observed" (gap 3)

| Prior | What it does with E⁻-like evidence | Closeness to the polarity component | Label |
|---|---|---|---|
| Chaparro, Florez & Marcus, ICSME'17 ("Using observed behavior to reformulate queries…") and EMSE 2019 ("Using bug descriptions to reformulate queries…", doi 10.1007/s10664-018-9672-z) | They split reports into title / OB / EB / S2R and use the parts as IR queries. Title + OB is best overall. Adding EB or S2R helps where present. EB is **more query text**, not evidence of absence | **1** (polarity as report structure; no signed scoring) | SNIPPET |
| BEE / BURT / DEMIBUD / ImproBR 2604.26142 / BugScribe | Detect, extract or generate OB/EB/S2R | 0.5 | SNIPPET / VERIFIED (abstracts) |
| Desc2Fix 2607.18550 (BR3) | "expected vs actual behavior" anchors measured for alignment with tests and patches | 1 | VERIFIED |
| Model-based software debugging: Mateis/Stumptner/Wotawa, cs/0011027 and cs/0309030; CFaults 2407.09337 | Consistency-based diagnosis: a component model predicts values, and an observation that contradicts the expected output yields conflicts and minimal diagnoses. Needs executions or specs | **1.5** ("expected vs observed" as the core diagnostic signal over program components; logical, not probabilistic; no prose) | VERIFIED (abstracts) |
| **Cayrac, Dubois & Prade, "Practical MBD with qualitative possibilistic uncertainty"** 1302.4936 (UAI'95) | Component fault modes have effects that are "more or less certainly present (or absent)". Anomalies propagate along component links, and diagnoses are scored on consistency with present/absent observations | **2** (signed effect signatures per component + propagation along structure, in device diagnosis) | VERIFIED (abstract) |
| **QMR-DT + Quickscore** (Shwe et al. 1991; Heckerman UAI'89) | Bipartite noisy-OR disease → finding network. **Positive and negative findings** both enter the posterior. Negative findings factorise cheaply (cost is linear in m⁻) | **2.** BMM's E⁻ first term, log[(1−q̃_vj+η)/(1−q̄_j+η)], *is* a noisy-OR negative-finding likelihood ratio | SNIPPET |
| **Chucky** (Yamaguchi et al., CCS'13) | Compares each function with similar "neighbour" functions. Functions that resemble their neighbours but lack one check score highest | **2 for peer_j(v)**: "siblings raise, v does not" is Chucky's missing-check anomaly, applied to effects instead of checks, and conditioned on the issue | SNIPPET (paper PDF + summaries) |
| Engler et al., "Bugs as deviant behavior" (SOSP'01) | Infers beliefs from what most code does and flags deviants | 1.5 | INFERRED (recall) |
| SBFL passing-test coverage (Tarantula/Ochiai e_p terms); Barinel | Passing executions count as evidence against an element | 1 (negative evidence = coverage by passing runs, not absent effects) | INFERRED |
| SemLoc 2603.29109 (BR3); FLACK 2102.10152 (counterexample vs satisfying instances, Alloy) | Violated semantic constraints form a spectrum; contrast of failing vs passing instances | 1–1.5 | VERIFIED (abstracts) |
| 2608.04591 "When Absence Is Evidence" | LLM negative reasoning under coverage completeness | 0.5 (a useful caution, see the hostile point) | VERIFIED (abstract) |

**Verdict on polarity.**
- "Absent expected manifestations lower the likelihood of hypotheses that would produce them" is textbook diagnosis (QMR-DT, possibilistic MBD).
- "Peers do it, v does not" is Chucky and Engler.
- The surviving delta is the **source** (issue prose, LLM-parsed per imagined step) and the **object** (execution-trained per-function effect probabilities on a code graph) **in issue localisation**. No SE prior found does that (BR3b-3, -4, -10, -11, -13). (INFERRED)

**Hostile point (INFERRED; it touches the logic, so I flag it only).**
- Under the noisy-OR reading, E⁻ "expected W, saw none" *lowers* the posterior of functions likely to emit W. For **omission faults**, the culprit is often exactly the function that should have emitted W.
- The peer term (Chucky-style) points the right way for omissions. The first term may point the wrong way.
- 2608.04591 also shows LLMs over-close "absent" claims.
- **Recommendation:** report C-BEH with the E⁻ first term and the peer term ablated separately. If the first term hurts, the polarity story becomes the peer-contrast story (closeness 2, Chucky).

## 4. Likelihood-ratio and base-rate normalised scoring of code elements (gap 4)

| Prior | Mechanism | Closeness to "LR vs repo base rate" | Label |
|---|---|---|---|
| **Liblit, Naik, Zheng, Aiken & Jordan, "Scalable Statistical Bug Isolation", PLDI'05** | Increase(P) = Failure(P) − Context(P). Context is the background failure rate at the predicate's site, and predicates are ranked by a lower confidence bound. Liblit's slides call it a "formal correspondence to likelihood ratio testing" | **2.** Base-rate-normalised evidence scoring of code elements. Here the evidence is executed predicates and the base rate is site reach | SNIPPET (slides + abstract) |
| Sisman & Kak, MSR'12 (version-history priors in IR bug localisation) | A prior defect-proneness distribution over files, combined with IR into a posterior | 1.5 (Bayesian IR localisation; the prior is a base rate over files, not over effects) | SNIPPET |
| IDF in VSM/BM25 IRBL (BugLocator, BRTracer, Pathidea) | Rarity-weighting of query terms across the corpus | 1.5. BMM's E⁺ term log(q̃/q̄) is a **soft IDF over an effect alphabet**, and arch §3.3 itself labels it "IDF-like" | INFERRED |
| Landsberg, "Doric" / probabilistic FL (HVSV'16) | A classical-probability foundation for SBFL; the measure returns probabilities | 1 | SNIPPET |
| Xie et al., theoretical SBFL formula analysis; Naish et al. | Optimality of formulas under assumptions. No LR derivation of Ochiai/DStar was found | 1 | SNIPPET |
| IssueExec 2607.17286 | Information-theoretic analysis: tests reduce localisation uncertainty by 7.73 bits | 1 | VERIFIED (abstract) |
| Jonsson et al. QRS'16; BayesFLo 2403.08079 | Bayesian component classification from reports; Bayesian FL for combinatorial tests | 1 | SNIPPET |

**Verdict.** Scoring code elements by evidence normalised against a background rate (Liblit) and by rarity (IDF) is established. BMM can claim only its **specific instantiation**: base rates q̄_j computed per EVAL repo from *predicted* effect probabilities, applied to prose-stated effects.

## 5. 2025–2026 SWE localisation using OB/EB parsed from issues (gap 5)

| Prior | Relation | Closeness | Label |
|---|---|---|---|
| 2607.07593 "What Makes a Good Bug Report for an AI Agent?" | Ablations on SWE-bench Pro: the models depend on localisation cues **and expected behaviour** | 1 (measurement; supports BMM's premise that EB carries signal) | VERIFIED (abstract) |
| 2607.09553 "Writing Bug Reports for Software Repair Agents" | Sentence-level annotation of SWE-bench Verified (OB, EB, S2R, localisation cues, fixes) vs agent success | 1 (a ready OB/EB sentence annotation for K3's effect-statement proxy) | VERIFIED (abstract) |
| BLAgent 2605.17965 | "Dual-perspective query transformation capturing structural and behavioral signals" from reports, then agentic rerank | 1 | VERIFIED (abstract) |
| FaR-Loc 2509.20552; SemNav 2609.31176; GraphLocator 2512.22469; RGFL 2601.18044; XAgent 2609.09769; SHERLOC | Behaviour description → dense retrieval; issue-conditioned semantic cards; symptom vertices; reasoning explanations; execution-guided | 1–1.5 | VERIFIED (abstracts) |

**Absence.** No 2025–26 localiser extracts signed (observed / expected-not-seen) effects from issues and scores functions by them (BR3b-5, -12, -14, -15).

**Use.** 2607.09553's OB/EB sentence labels on SWE-bench Verified can validate the K3 effect-statement proxy for free (INFERRED). Check licence and availability.

---

## 6. Must NOT claim (additions to v2 §1.1 items 1–10)

11. Probabilistic (Markov-model/DP) alignment of NL bug-report steps onto a program-derived graph, with missing-step inference, as new: **Zhang et al. FSE'24** (VERIFIED); AstroBR; Euler.
12. Using the observed-vs-expected behaviour structure of bug reports for localisation as new: Chaparro et al. ICSME'17/EMSE'19; BugScribe; Desc2Fix; 2607.07593.
13. Negative findings or absent-expected-effect evidence as a new inference idea: QMR-DT/Quickscore; possibilistic MBD (Cayrac–Dubois–Prade); MBD software debugging (Wotawa et al.; CFaults).
14. Base-rate-normalised or likelihood-ratio scoring of code elements as new: Liblit et al. PLDI'05 (Increase vs Context); Sisman & Kak MSR'12; IDF-weighted IRBL.
15. "Should do X like its peers but doesn't" contrast as new: **Chucky** (CCS'13); Engler et al. SOSP'01.
16. Refine item 2. Pathidea reconstructs paths **from logs and stack traces in reports** by BFS between consecutive log anchors, giving a file-level IR boost. Do not describe it as using NL descriptions, and do not claim "first to combine report-derived execution paths with IR localisation" (VERIFIED).

**Related-work order update.**
1. SherLog → Pensieve → **Pathidea** (log-anchored BFS) → CrashLocator → COCA/2506.19045.
2. Euler → **Zhang et al. FSE'24 (Markov/DP S2R alignment)** → AstroBR.
3. Newson–Krumm and RoG/GCR.
4. **Diagnosis with signed findings: QMR-DT, possibilistic MBD, software MBD.**
5. **Liblit Increase, IDF** (base-rate scoring).
6. **Chucky / Engler** (peer contrast).
7. LocAgent.
8. IPA-GNN / CodeFlow.
9. RepoReasoner / SWE-Flux.

## 7. Residual gaps

- QMR-DT/Quickscore, Liblit's exact Increase notation, Chucky and Engler are SNIPPET or recall. They are classic papers and their mechanism is not in doubt; read them before writing.
- The full texts of Chaparro EMSE'19 and the Yakusu/ReCDroid/ReproBot papers were not read. None is expected above closeness 1.5 for the novel piece.
- Semantic Scholar returned 429 again. Google Scholar and ACM DL were not crawled. ICSE/FSE 2027 camera-ready papers not yet on arXiv cannot be excluded.
- I found no SE work on "omission-fault localisation from expected-but-absent effects" in issues. That is the strongest remaining open niche, and also the sign-semantics risk in §3.

---

## Search log (BR3b-n)

Tools:
- **AX**: arXiv export API.
- **OR**: OpenReview `api2.openreview.net/notes/search`, 8 s pacing.
- **WS**: WebSearch.
- **WF**: WebFetch.
- **FCS**: Firecrawl web search.
- **FCR**: Firecrawl research paper search.
- **PDF**: local pypdf extraction.
- **S2**: Semantic Scholar API.

| ID | Tool | Query / target | Result used |
|---|---|---|---|
| BR3b-1 | WS | Pathidea IRBL re-constructing execution paths logs abstract | Not found by name; Dao et al. (execution info + IRBL) noted |
| BR3b-1a | S2 | search "Pathidea bug localization execution paths logs" | 429 |
| BR3b-1b | FCS | `"Pathidea" bug localization execution paths logs Chen` | ACM/IEEE/author pages; **author PDF URL** |
| BR3b-1c | PDF | petertsehsun.github.io/papers/PathideaTSE2021.pdf | **VERIFIED**: abstract; §3 log regex; Algorithm 1 BFS between consecutive logs; PathScore = β·N(VSM) |
| BR3b-2 | FCR | LLM bug report S2R mapped to GUI transition graph or call graph, probabilistic alignment | TreeMind, BugRepro, ReBug, ViBR, LTGDroid, GIFdroid, GALA, **AstroBR**, CrashTranslator, BugScribe, AdbGPT, ReBL, Desc2Fix (abstracts) |
| BR3b-3 | FCR | Issue localisation using expected vs observed behaviour, SWE-bench (≥2024-06) | 2607.09553, 2607.07593, SweRank, XAgent, IssueExec, OrcaLoca, etc.; no signed-effect scoring |
| BR3b-4 | FCR | Negative-evidence FL, expected behaviour not observed | FLACK, SBEST, SemLoc, LLMAO, Inforence, 2608.04591; no SE absent-effect scoring |
| BR3b-5 | WS | SBFL likelihood ratio / Bayesian interpretation of Ochiai/DStar | No LR derivation found; Landsberg Doric; Xie et al. theory |
| BR3b-6 | WS | Bayesian IR bug localisation prior, Bayesian network | **Sisman & Kak MSR'12**; Jonsson QRS'16; BayesFLo |
| BR3b-7 | WS | Chaparro/Marcus OB/EB query reformulation ICSME'17/EMSE | EMSE'19 doi + finding (title+OB best); ICSME'17 citation; 2512.07022 |
| BR3b-8 | AX ×6 | `"expected behavior" AND localization`; `"observed behavior" AND "bug report"`; `"negative evidence" AND fault`; `absence AND "fault localization" AND expected`; `"likelihood ratio" AND "fault localization"`; `"base rate" AND localization AND code` | 2607.09553, 2607.07593, BURT, ImproBR, BugScribe; the last four queries returned 0 relevant (**absence**) |
| BR3b-9 | AX id_list | 2512.07022, 2209.10062, 2502.04251, 2604.01148 | Abstracts VERIFIED |
| BR3b-10 | WS | Bug-report reproduction steps aligned to GUI model with HMM/Viterbi/probabilistic alignment | **Zhang et al. FSE'24 (Markov model + DP)**; AstroBR; EBUG; FUSION |
| BR3b-11 | OR ×6 | expected behavior bug localization; negative evidence localization; missing behavior bug report; observed expected behavior issue localization; likelihood ratio localization code; steps to reproduce call graph | Noise (IRBL titles, RLocator, BigIssue, AutoPruner); **no hit** |
| BR3b-12 | WF + PDF | par.nsf.gov/servlets/purl/10544604 (FSE'24) | **VERIFIED**: PRM states (step, event); C1/C2 transitions; similarity + 1/distance "probability"; DP top-k; missing-step fill |
| BR3b-13 | WS | QMR-DT noisy-OR negative findings, Quickscore | Heckerman UAI'89 and model description (SNIPPET) |
| BR3b-14 | WS | Liblit Increase / Failure / Context | PLDI'05 and slides: "formal correspondence to likelihood ratio testing" (SNIPPET) |
| BR3b-15 | FCR | LLM extracts actual vs expected behaviour from issue, matched to function behaviour (≥2025) | FaR-Loc, XAgent, Time Travel, Desc2Fix, RGFL, RepoLens, CoSIL, COCA, ImproBR, Issue2Test; **absence** |
| BR3b-16 | FCR | Model-based software debugging, expected vs observed values | CFaults, Mateis/Stumptner/Wotawa (cs/0011027, cs/0309030), **possibilistic MBD 1302.4936** (abstracts) |
| BR3b-17 | FCR | Bayesian posterior over code locations from LLM-extracted symptom likelihoods (≥2024) | GenLoc, LIDL, SemNav, **BLAgent**, SHERLOC, GraphLocator, SweRank+; **absence** |
| BR3b-18 | WS | Chucky missing checks, neighbour functions | Chucky CCS'13: neighbour-contrast anomaly (SNIPPET) |

No file other than this one was modified. Downloaded PDFs were kept in separate temporary directories outside the project.
