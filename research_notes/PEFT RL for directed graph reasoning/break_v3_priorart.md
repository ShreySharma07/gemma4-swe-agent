# Break v3: hostile prior-art red team on the null-order dial (8 Oct 2026)

**Role.** Fresh, hostile prior-art red team. The target is the single [CLAIMED] piece of `architecture_v3_final.md` (§0, §1 Family B, §2.2).

**The claim under test.** LoRA training on execution-labelled counterfactual items, generated from a **nested family of graph null models**, acts as a dial over which topology statistics an LLM relies on when reasoning over code call graphs. The family is:
- near: degree + level + module;
- mid: degree + level;
- far: degree;
- leak: edge count.

The statistics are module locality, depth level and local motifs. Reliance on a statistic falls **when, and only when**, the training null destroys it. Reliance is read out by prior-vs-displayed-edge conflict probes on held-out real code.

**Kill criterion.** Prior art that uses the same mechanism. Prior art that only shares the domain does not count.

**Inputs read:**
- `architecture_v3_final.md`, in full;
- `break_v2_priorart.md`, in full;
- `break_v1_priorart.md`: bottom line and search log.

I did not repeat their searches. CRPO, CAD/Teney/PairCFR, NormWorlds-CF, 2112.00476 (abstract), CNY, Huang TMLR'24, NLGIFT, 2604.18092, PhantomEnvs, RepoMirage and Spurious Tool Use are inherited at their stated labels.

**Labels.**
- **VERIFIED**: I read the primary PDF text this session (alphaXiv `answer_pdf_queries`).
- **SNIPPET**: abstract or search-result text only, or another paper's description of the work.
- **INFERRED**: my own reasoning, or recalled from memory and not re-read this session.

**Closeness scale.**
- 0: unrelated.
- 1: same domain or goal, different mechanism.
- 2: same mechanism family, with a material difference.
- 3: same mechanism already done.

---

## 0. Bottom line

| Item | Result |
|---|---|
| **Verdict on the dial claim** | **NARROW.** Not KILL: no prior trains any model (LLM, GNN or otherwise) on a *nested family of graph null models* and reads out *statistic-specific* reliance. Not SURVIVE as written either, because the underlying mechanism is textbook in other modalities. The mechanism is: training data that *destroys or decorrelates* one cue lowers reliance on that cue, measured by cue-conflict probes, with a dose-response. The same applies to a nested family of statistic-preserving corruptions used as training corpora. |
| **Closest priors** | (1) **Sinha et al. 2021, "Masked LM and the Distributional Hypothesis"** (2104.06644, EMNLP 2021). **Closeness 2.** VERIFIED. A *nested family of n-gram-preserving shuffles* (M1 to M4, plus a unigram-resample "leak" baseline M_UG) is used as **training corpora**. Non-parametric minimal-pair probes show that order sensitivity grows with the preserved order. This is the text analogue of "nested null family as training data, plus a probe of what the model relies on". (2) **Geirhos et al. ICLR 2019 / Hermann, Chen & Kornblith NeurIPS 2020** (1911.09071). **Closeness 2.** Hermann is VERIFIED; Geirhos is INFERRED. Cue-destroying augmentations shift reliance, measured by cue-conflict stimuli; the effects are additive and dose-dependent. (3) **Early Cue Precision** (2606.30344). **Closeness 2.** VERIFIED. Graded cue-label decorrelation p ∈ {0.1 … 1} gives a monotone dose-response in conflict accuracy, and later shortcut-rich fine-tuning overwrites it. (4) **Null-model data augmentation for graph classification** (2112.00476 / TNSE'24). **Closeness 2**, inherited. Null models of several orders are used as GNN training data, but label-preserving and with no reliance readout. |
| **Next ring** | **Code over Words / LCV** (2601.18352): SFT on counterfactual pairs, where the same state carries contradictory displayed rules, cuts LLM prior-reliance (ΔP, prior vs context); code domain, LoRA; VERIFIED; **1.5**. **Kim et al.** (2510.02370): training-data statistics set prior-vs-context conflict preference; VERIFIED; **1.5**. **Shortcut Before Circuit** (2608.24460): cue reliance is *underdetermined* when cues are coextensive in training, with seed variance up to 0.88; VERIFIED; **1.5**. **Fly-connectome nulls** (2609.39248): what a degree-preserving or boundary-preserving null preserves decides what evolved agents rely on, with a dose series; VERIFIED; **1.5**. **Hermann et al. 2023, "Foundations of Shortcut Learning"** (2310.16228): reliance depends on predictivity *and* availability; SNIPPET; **1.5** |
| **What survives** | (a) The *corruption family is a nested hierarchy of graph null models* (dK-style), applied to real traced call graphs. (b) The *reliance readout is per statistic family* (Loc / Lev / Mot), with a pre-registered break/keep design matrix. (c) The **specificity half** ("only when"): keep arms do not move. No prior tests specificity across a nested family. Sinha measures one aggregate (order sensitivity); Hermann/Geirhos measure one cue (texture). (d) The domain: LLM, real code, displayed-edge conflict probes, and execution labels. |
| **Most defensible sentence** | §4 |

**Required changes to v3:**
- Retag the dial from [CLAIMED] mechanism to **[NARROWED] instrument plus specificity test**.
- Cite Sinha 2021, Geirhos 2019 / Hermann 2020, 2606.30344, 2601.18352, 2510.02370 and 2608.24460.
- Add the two design consequences in §5. They are new threats to HS-spec, so they are not only citations.

---

## 1. Closest prior work, with deltas

### 1a. Sinha, Jia, Hupkes, Pineau, Williams, Kiela, "Masked Language Modeling and the Distributional Hypothesis: Order Word Matters Pre-training for Little". arXiv 2104.06644, EMNLP 2021. Closeness 2. VERIFIED (pp. 1, 3–6, 8, 19, 22)

**What it does:**
- Pre-trains RoBERTa-base on corpora permuted by F_n, a shuffle that preserves n-grams, for n = 1…4 (M1–M4).
- Baselines: M_UG, sampled from the unigram distribution and keeping only unigram frequency (the analogue of a "leak" arm); M_N (natural, the analogue of U); M_NP (no positions).
- 3 pre-training seeds per corpus.
- Non-parametric minimal-pair probes (Linzen, Gulordava, Marvin & Linzen): the probability of the good vs the bad focus word is the reliance-on-order readout. "With each step from M1 to M4, the difference between probabilities of correct and incorrect focus words increases, albeit marginally" (App. L).
- Finding: GLUE is barely hurt, but non-parametric probes separate M_N from all shuffled models.

**Same as the dial:**
- a *nested* family of statistic-preserving null corpora used as **training data**, with the far end at "first-order statistics only";
- each arm's behaviour on minimal pairs is read as reliance on the destroyed structure.

**Different:**
1. **Label-preserving pretraining corpora, not answer-flipping counterfactual items.** The model never learns that the displayed structure determines the label.
2. **One aggregate readout** (word-order sensitivity), not per-statistic reliance. There is no break/keep matrix and no specificity test.
3. Text n-grams, not graph nulls (degree/level/module); MLM pretraining from scratch, not LoRA on a 31B model; no real-code conflict probe.

**Implication.** "Nested statistic-preserving nulls as training data, ordered by preserved order, with a minimal-pair readout" is **done** in NLP. v3 cannot claim the *dial idea* in general, only its graph-null instantiation and its per-family specificity.

### 1b. Geirhos et al. ICLR 2019 (Stylized-ImageNet; INFERRED, cited via the next two papers) and Hermann, Chen & Kornblith, "The Origins and Prevalence of Texture Bias in CNNs". NeurIPS 2020, 1911.09071. Closeness 2. VERIFIED (pp. 1–2, 5–6, 17)

**What it does:**
- Augmentations that **destroy one cue** (colour distortion, blur, noise) lower texture reliance. Random crops, which destroy global shape, *raise* it.
- Reliance is read on **cue-conflict stimuli** (shape bias on GST).
- Effects are **additive** (Table 3: 19.5% → 25.8% → 30.7% → 36.1% → 48.7% → 55.2%) and dose-dependent in minimum crop area (App. B).
- Quote (p. 9): "the statistics of the training dataset are the most important factor."

**Same as the dial:**
- the mechanism itself: the training-data operator destroys statistic s, so reliance on s falls on conflict probes;
- a graded family of operators with a dose-response;
- the training operator destroying cue A raises reliance on cue B. That is cue substitution, the exact threat to v3's "only when" (§5.1).

**Different:**
- Vision, label-preserving augmentation, a single pair of cues (shape/texture);
- no nested null hierarchy;
- no specificity test across three families.

### 1c. Park et al., "Early Cue Precision Shapes Visual Shortcut Learning". arXiv 2606.30344. Closeness 2. VERIFIED

**What it does:**
- Training cue-label match probability p ∈ {0.10, 0.25, 0.5, 0.75, 1.0} produces a monotone dose-response in conflict accuracy (CIFAR frozen probes 0.569 → 0.114) and in texture-choice rate.
- **End-to-end, one epoch of shortcut-rich fine-tuning overwrites the low-p benefit** (conflict accuracy 0.305 → 0.047).
- The distinction "degraded but predictive ≠ decorrelated" maps directly onto v3's null-vs-leak contrast.

**Same as the dial:** graded decorrelation of a cue in training data, with a conflict-probe readout and a dose-response.

**Different:** vision, one cue, label-preserving, no graph nulls.

**Bearing on v3:** v3's M arms are 50% real + 50% null, so the broken statistic is *decorrelated* (p ≈ 0.5–0.75), not removed. 2606.30344 predicts partial, not full, reliance drops. It also predicts that a later real-topology training stage (P arm, or deployment SFT) would erase them.

### 1d. Null-model data augmentation for graph classification. 2112.00476 / TNSE 2024. Closeness 2 (inherited; abstract VERIFIED by v3 N5)

**What it does.** Null models of several orders serve as training augmentation for GNN graph classification. Citers checked this pass (Semantic Scholar, 6 citers: AGSOA, Mix-Key, multi-level augmentation for low-degree graphs, hypergraph homophily, community hiding, molecular substructure expert): **none** uses nulls as a reliance dial and none involves an LLM. SNIPPET (titles/abstracts).

**Delta.** Label-preserving; no reliance readout; no specificity.

### 1e. Next ring (closeness 1.5)

| Prior | ID | Label | Mechanism | Delta / relevance |
|---|---|---|---|---|
| **Code over Words / LCV** (Xu et al., PKU) | 2601.18352 | VERIFIED | LoRA SFT (Qwen2.5-7B, ≈600 pairs) on **counterfactual pairs**: identical grid, contradictory displayed rules. This suppresses "semantic inertia" (prior over context), measured by ΔP = P(logic) − P(prior) on conflict scenarios | Prior-vs-displayed conflict readout, plus counterfactual SFT to make an LLM follow what is displayed. Same goal and readout family as v3 Family A / E2. No graph nulls, no dial. Cite it as prior art for "counterfactual SFT reduces prior reliance on conflict probes" |
| **How Training Data Shapes the Use of Parametric and In-Context Knowledge** (Kim et al.) | 2510.02370 | VERIFIED | Corpus statistics (repetition, 1–10% inconsistency, Zipf skew) causally set prior-vs-context conflict preference; post-training on noisy or clean data reshapes it | Data statistics set reliance, with knowledge-conflict probes. It shows that *1% inconsistency* flips preference. That warns v3 that small changes to the real/null mix may dominate |
| **Shortcut Before Circuit** (Liao & Liang) | 2608.24460 | VERIFIED | Cues that are coextensive in training leave the objective indifferent; the per-cell rule readout varies by up to 0.879 across seeds (SE 0.025) | Direct threat to HS-spec (§5.2). In keep arms, Loc/Lev/Mot cues that remain coextensive with the label are underdetermined, so seed variance in reliance could exceed v3's σ_seed = 0.03 |
| **Null-model treatment of the sensory-motor boundary** (Park, IonLabs) | 2609.39248 | VERIFIED | Agents are evolved on connectome vs nulls (column shuffle, Maslov–Sneppen degree-preserving, boundary-preserving). The null's preserved set decides which shortcut (direct olfactory→motor) the learner exploits. Dose series; olfaction-ablation reliance readout | The closest "nested graph nulls change what a learner relies on" finding. But the null is the learner's *architecture*, not its training data. The learner is an evolved rate model. Reliance is on a sensory channel, not on input topology statistics |
| **On the Foundations of Shortcut Learning** (Hermann, Mobahi, Fel, Mozer) | 2310.16228 | SNIPPET | Which feature a model uses depends on *predictivity* and *availability* | Theory behind the dial. It predicts that removing the predictivity of one family raises reliance on *available* remaining families, so the "only when" claim is at risk |
| OraclePhys | 2608.17162 | VERIFIED | The supervision answer form is the controlled variable in LoRA SFT; FE-oracle counterfactual probes | Same "make what fine-tuning teaches an experimental variable" framing; the knob is the label form, not the data null. **1** |

**Checked and not close (closeness ≤ 1):**
- FiLoRA 2602.02060: controllable feature reliance via LoRA *architecture*, not data.
- 2607.16298: feature-reliance evaluation.
- 2608.14121 and 2603.08270: GNN shortcut mitigation via explanations or spurious-correlation learning.
- 2605.08405: in-context graph belief vs circuitry.
- Code knowledge-conflict papers 2510.19116, 2604.09515, 2606.30810: API-version conflicts, evaluation or steering, no structural dial.
- 2609.26388 Lexical Superstition (inherited).
- 2604.04033: connectome vs sparse random controls.

---

## 2. Search for the exact combination: none found

No paper found does any of the following:
- train any neural model on **dK-series / configuration-model / hierarchical graph nulls** to *control* reliance (dK web search; Semantic Scholar citers of Orsini et al. 2015, 185 citers with ML-keyword filter, only "Discovering Wiring Patterns Influencing NN Performance" (architecture graphs) and a federated-learning initialisation paper);
- use a graph null family as **LLM** training data;
- test **statistic-specific specificity** ("reliance on f falls iff f is broken") across a nested family in any modality.

The closest specificity-like evidence: Hermann 2020 shows that crops raise texture bias while colour/blur/noise lower it, which is cue-specific direction. Sinha 2021 shows a monotone dial in a single aggregate.

---

## 3. Remaining delta, stated precisely

1. **Corruption operator.** A *nested hierarchy of graph null models* (directed double swaps with level/module constraints, plus an edge-count leak) on real traced call graphs. The preserved set is formally specified, and rewired fractions J are pre-registered. This is new as a *training* operator for LLMs. Its text analogue (n-gram-preserving shuffles) and graph-GNN analogue (label-preserving null augmentation) exist.
2. **Answer-flipping, execution-verified labels on the null worlds.** The break statistics are made *non-predictive of the label*, not merely corrupted in the input. Sinha, Hermann and 2112.00476 are all label-preserving. 2606.30344 manipulates cue–label correlation directly but in vision. *This is the substantive mechanism difference from Sinha 1a.*
3. **Per-family reliance readout plus specificity test** (break/keep design matrix, HS-spec TOST) on held-out real code, via displayed-edge conflict probes. Conflict probes per se are standard: Geirhos cue-conflict, knowledge-conflict, LCV ΔP, CNY rewire. Their use as a **three-family, pre-registered specificity matrix** is not found.
4. **Domain.** LLM (31B, LoRA, W4A16) over code call graphs. Domain alone would not count, but it carries items 1–3.

---

## 4. Most defensible one-sentence novelty claim

> **"We use a nested family of graph null models (degree / +level / +module, with an edge-count leak), applied to real traced call graphs, as answer-flipping, execution-labelled counterfactual training data for an LLM. Pre-registered prior-vs-displayed-edge conflict probes on real code then test whether reliance on each topology-statistic family falls specifically for the families the training null destroys: a graph-null, per-statistic extension of cue-decorrelation training (Geirhos/Hermann; Sinha et al.'s nested n-gram shuffles) that no prior has run on graphs or LLM code reasoning."**

Short form: *"A null-model dial: nested graph nulls as counterfactual LLM training data, with a per-statistic specificity test of which topology shortcuts are removed."*

**Must not be claimed:**
- that training on cue-destroying data reduces reliance on that cue (Geirhos 2019; Hermann 2020; 2606.30344);
- that a nested family of statistic-preserving nulls as training data is new (Sinha 2021 for text; 2112.00476 for GNN graphs);
- that counterfactual SFT reduces prior-over-displayed reliance in LLMs (LCV 2601.18352; Faithfulness-QA, inherited);
- that conflict probes are a new readout.

---

## 5. Consequences for v3 (prior-art driven threats to HS-spec, not only citations)

1. **Cue substitution (Hermann 2020; Hermann 2023 availability).**
   - **Threat.** When the mid null destroys Loc, Lev and degree become *relatively* more predictive in the M_mid training mix. Reliance on Lev in M_mid may then **rise above U**. That violates the HS-spec TOST (keep arms ≈ U, margin 0.06) in the *opposite* direction from distrust.
   - **Changes.**
     - Pre-register HS-spec as **two-sided**.
     - Add a secondary prediction "keep-family reliance in break arms ≥ U".
     - Compute, on CPU (K-C3 extension), each family's *conditional* predictivity in every arm's training mix. The dial's honest form is "reliance tracks training-mix predictivity per family", which K-C3's classifier can predict before any GPU spend.
     - If the predicted predictivities already show substitution, the "only when" wording must be replaced by "reliance tracks per-family predictivity".
2. **Underdetermination of coextensive cues (2608.24460).**
   - **Threat.** In keep arms, Loc, Lev and Mot are often coextensive with each other and with the displayed path. Which one a seed latches onto may vary across seeds far beyond σ_seed = 0.03. That would sink HS-spec power (≈0.7 now) and inflate the HS-β SEs.
   - **Change.** Pre-register a pilot σ_seed for *Reliance_f* specifically, not AP. Use 2 seeds of U and M_mid at G0+ on a 100-item E2 slice. Rescale MDEs before L5.
3. **Decorrelation, not removal (2606.30344).** M arms are 50% real + 50% null, so the broken family keeps p ≈ 0.5–0.75 predictivity in the mix.
   - Predicted reliance drops should be **partial**. The −0.20 for HS-Loc is GUESSED and probably optimistic.
   - Add an **N_mid** (pure-null) point as the p ≈ chance end. That is Q1 option B, which this strengthens.
   - State that a later real-topology fine-tune may overwrite the dial (2606.30344 Table 3). An optional "dial durability" check: M_mid → +P items for 1 epoch, then re-read reliance.
4. **Cite LCV (2601.18352) beside H1/H2.** It is the closest prior showing that counterfactual-pair LoRA SFT lowers LLM prior-over-displayed reliance on conflict probes. H1 ("M_mid > base on displayed-edge accuracy") is therefore *expected* from prior art. The novel test is H2 plus Family B.
5. **Cite Sinha 2021 as the nested-null precedent** in the §0 delta table, with closeness 2. Add M_UG ↔ M_leak as the explicit analogue. Their finding that parametric probes miss what non-parametric probes reveal supports v3's use of behavioural conflict probes over representation probes.

---

## Search log

**Tools:**
- alphaXiv `discover_papers` (DP) and `answer_pdf_queries` (READ);
- OpenReview `api2.openreview.net/notes/search` (OR), via a script paced at 7 s. I did not attempt per-note fetches; the v1/v2 logs report HTTP 403 for them;
- Semantic Scholar Graph API (S2);
- WebSearch (WS);
- Firecrawl search (FC).

| # | Tool | Query / target | Key result |
|---|---|---|---|
| 1 | DP | Training on null-model / dK-series / max-ent ensembles to control structural shortcut reliance | 2608.14121 (GNN shortcut XAI), 2604.04033, 2605.23285, **2609.39248**, SCL-GNN; nothing training an LLM or NN on nested nulls |
| 2 | DP | Targeted augmentation destroying one spurious feature; graded augmentation as a dial over reliance | FiLoRA 2602.02060 (architecture), 2607.16298, CoBA, SAGE; not close |
| 3 | DP | Pre-training on n-gram-order-preserving shuffles (nested statistic-preserving nulls) | 2203.10995, 2603.19427 (word-order spectrum), 2407.12034; led to Sinha 2021 |
| 4 | DP | LLM graph reasoning across graph generators / shortcuts | GTA, GraphEcho, 2604.18092 (inherited), 2605.08405; nothing same-mechanism |
| 5 | READ | **2104.06644 Sinha et al.** | **Nested F_n shuffles (n = 1–4) + unigram resample as pretraining corpora; non-parametric minimal-pair probes; monotone in n** |
| OR1 | OR | null model training data shortcut | **rnHqWs63l0 / 6Ex850lPzI** (null-model augmentation, inherited), Infusion; nothing new |
| OR2 | OR | dK-series graph neural network | DP-graph privacy (dK as graph model); nothing |
| OR3 | OR | configuration model augmentation graph | Nothing |
| OR4 | OR | maximum entropy graph ensemble training | Network-science max-ent; nothing ML-training |
| OR5 | OR | degree preserving randomization training language model | Degree-correlation rewiring (network science); nothing |
| OR6 | OR | texture shape cue conflict augmentation dial | Edge-deformation shape-bias augmentation; cue-decomposition; SAM texture bias (generic cue-conflict family) |
| OR7 | OR | knowledge conflict code LLM counterfactual | Domain-grounded counterfactual augmentation for knowledge conflicts (mL2znR6ezq; SNIPPET title; text, not structure) |
| OR8 | OR | graph reasoning LLM generalization synthetic graph generators | **NLGIFT** (aM736ItkQ9 / cBAXrIZNh3; inherited) |
| OR9 | OR | statistic preserving shuffle pretraining | Noise |
| OR10 | OR | controllable shortcut reliance augmentation | LLM smart augmentation for shortcuts, misinformation shortcuts, COMI; generic |
| S2-1 | S2 | Citers of arXiv:2112.00476 | 6 citers; none an LLM, dial or reliance readout |
| S2-2 | S2 | Citers of Orsini et al. 2015 (DOI 10.1038/ncomms9627), ML-keyword filter over 185 | "Discovering Wiring Patterns Influencing NN Performance" (ECML'22; architecture), DFL initialisation; nothing |
| 6 | DP | Code LLM knowledge conflict / context faithfulness / counterfactual code | 2510.19116, 2604.09515, 2606.30810, NovelAPIBench, Faithfulness-QA (inherited); API-version conflicts only |
| 7 | DP | LLM graph priors vs displayed edges; counterfactual graph probes | 2605.08405, 2606.09484, 2511.16767 (inherited), 2604.18092; nothing training-side |
| 8 | READ | **1911.09071 Hermann et al.** | **Cue-destroying augmentations shift shape bias additively; crops raise texture bias; dataset statistics dominate** |
| 9 | DP | Family of perturbations preserving different statistics as training data; impossible languages | **2608.24460**, 2606.30815 (impossible languages), 2609.32991 |
| 10 | READ | **2608.24460 Shortcut Before Circuit** | Coextensive cues give an underdetermined rule readout; seed spread 0.879 |
| 11 | WS | dK-series / dk-random graphs + GNN training / augmentation / shortcut | Only Orsini 2015 network-science pages; no ML use |
| 12 | READ | **2510.02370 Kim et al.** | Data statistics set prior-vs-context conflict preference |
| 13 | READ | **2609.39248 fly-connectome nulls** | Null's preserved set decides the learner's shortcut; dose series; architecture-null, not data |
| 14 | DP | Degree-preserving / configuration-model rewired graphs in LLM fine-tuning | Graph rewiring for over-squashing; SLASH; 2606.15633; **OraclePhys 2608.17162**; nothing |
| 15 | READ | 2608.17162 OraclePhys | Answer-form knob, LoRA, FE-oracle counterfactual probes (closeness 1) |
| 16 | FC | "null model(s)" + fine-tuning LM + graph training data shortcut | Generic fine-tuning guides only |
| 17 | FC | Surrogate-data hierarchy (FT/IAAFT) as NN training data | Surrogate testing literature only |
| 18 | DP | Code LLM fine-tuning on counterfactual programs to follow displayed code / call graphs | **Code over Words 2601.18352**, Lexical Superstition (inherited), CFG fine-tuning; nothing on call-graph nulls |
| 19 | DP | Feature predictivity varied in synthetic data; cue conflict | **2310.16228 Foundations of Shortcut Learning**, **2606.30344**, 2607.10116, 2605.12945 |
| 20 | READ | **2601.18352 LCV** | Counterfactual-pair LoRA SFT; ΔP prior-vs-logic probe; inverse scaling |
| 21 | READ | **2606.30344 Early Cue Precision** | Graded cue-label decorrelation; conflict-accuracy dose-response; fine-tuning overwrites |

**Not done:**
- Primary reads of Geirhos 2019 (INFERRED), Hermann 2023 2310.16228 (SNIPPET), the full text of 2112.00476 (abstract only, inherited), Kallini et al. "Mission: Impossible Language Models" (INFERRED as closeness ≤ 1.5: a perturbation family for learnability, not reliance) and the Mahadevan 2006 dK-series paper.
- OpenReview per-note fetches (skipped; known 403).
- A GraphArena check (an evaluation benchmark; not expected to bear on a training-side dial).
