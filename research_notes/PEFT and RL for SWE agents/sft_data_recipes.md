# SFT data and recipes for SWE agents (as of 8 Oct 2026)

Scope: public trajectory datasets and SFT recipes for open ~7–32B SWE agents, judged against our constraints:
- gemma-4-31b-it, served W4A16; LoRA only;
- Kaggle 4×L4 training, ~30 GPU-h/week; no online RL;
- ≤32k-token trajectories with no in-task compaction;
- tools run_command / read_file / edit_file / write_file / submit_patch / get_status in the ADK/Gemma tool-call format;
- hidden test = ~120 tasks from unseen private Python repos;
- open-source release required.

Already covered in `research_notes/LLM graph reasoning for code/post_training_navigation.md` and `docs/literature_2025_2026.md` (SWE-smith, SWE-ZERO, 2607.17205, CodeMidas, CodeScout, P2T, A²Agent). They are deepened here only where new numbers matter.

**Labels**
- Status: **[PR]** = peer-reviewed or accepted; **[PP]** = preprint / blog / dataset card.
- **VERIFIED** = read in the primary source (PDF or dataset card) this session.
- **SNIPPET** = taken from a search result, forum post or secondary summary, not the primary paper.
- **INFERRED** = my reasoning; no source states it.
- All "pp" are absolute percentage points on SWE-bench Verified (SWE-V) unless stated otherwise.

---

## Q1. Public trajectory datasets: size, teacher, scaffold, licence, Python share, length, success filter, and conversion cost

### Takeaway
There is now far more permissively licensed SWE trajectory data than we could ever train on: >1M trajectories, mostly CC-BY-4.0, MIT or Apache-2.0, Python-heavy, and with Qwen3-Coder-480B / GLM / Claude teachers.

The binding constraints for us are not supply. They are:
1. **Length.** Typical modern trajectories average 30–130 turns and roughly 20–70k tokens. Our 32k cap discards or truncates a large share.
2. **Scaffold mismatch.** Everything is OpenHands, SWE-agent or mini-swe-agent format. All of it must be re-rendered into Gemma-4's native tool-call syntax and our six tools.

The best-fitting sources are those with short trajectories and function-calling tools:
- **SWE-smith trajectories** (MIT, ~30 turns);
- **SWE-Gym** (MIT, ~19 turns, ~19k tokens);
- **nvidia SWE-ZERO** (CC-BY-4.0, 10–30 turns, execution-free);
- the ≤32k subset of **nebius SWE-rebench-OpenHands** (CC-BY-4.0).

### Cited Findings

**Comparison grid (all VERIFIED unless the cell says otherwise)**

| Dataset (HF id) | Size | Teacher | Scaffold / tool format | Licence (HF card) | Python | Avg length | Success filter | Best published student result | Status |
|---|---|---|---|---|---|---|---|---|---|
| SWE-Gym `SWE-Gym/OpenHands-SFT-Trajectories` | 491 resolved (from 2.4k Python tasks, 11 repos) | gpt-4o, Claude-3.5-Sonnet | OpenHands CodeAct (bash + str_replace_editor), function calling | MIT | 100% | ~19 turns, ~19k tokens | resolved only | Qwen2.5-Coder-32B 7.0→20.6 (+13.6) | **PR ICML 2025** |
| SWE-smith `SWE-bench/SWE-smith-trajectories` | 5,017 train set; HF repo now 49,897 total / 21,513 successful | Claude 3.7 Sonnet (+3.5 Sonnet, gpt-4o) | SWE-agent: bash, str_replace_editor, submit; function calling | MIT | 100% (128 Python repos) | 58 messages ≈ 29 actions; HF card avg 30.2 turns; **23.26% >32k tokens** (per SERA) | resolved only, ≤3 per instance | Qwen2.5-Coder-32B → 40.2 | **PR NeurIPS 2025 D&B** (via citations) |
| Nebius `nebius/SWE-rebench-openhands-trajectories` | 67,074 total / 32,161 resolved, 1,823 repos, 3,792 issues | Qwen3-Coder-480B-A35B | OpenHands v0.54, function calling; `arguments` stored as a JSON string | CC-BY-4.0 | ~100% (SWE-rebench Python) | avg 64.3 turns (max 100); resolved avg **47,171 tokens** (per 2607.17205) | flag `resolved`; also agent-generated-test metrics | Qwen3-Coder-30B-A3B 44.8→51.0 (full resolved SFT) | PP (card) |
| Nebius `nebius/SWE-agent-trajectories` | 80,036 / 13,389 resolved | Qwen2.5-72B, Llama3-70B | SWE-agent, **no** function calling | CC-BY-4.0, but the card adds a Llama 3.1 licence notice | Python | resolved avg 31.3 steps, **8.4k tokens** | `target` flag | 72B agent+verifier 40.6 | PP |
| SWE-Lego `Lego-X/SWE-Lego-Real-Data` + `-Synthetic-Data` | 32k tasks, 3,251 repos; 14.1k resolved + 4.0k semi-resolved trajectories | Qwen3-Coder-480B | OpenHands v0.53 (think, str_replace_editor, bash) | Apache-2.0 | "Python-centric" | avg 62.5 turns (max 100) | resolved + semi-resolved; git-hack filtered | Qwen3-8B 42.2, Qwen3-32B 52.6 (full SFT, 128k ctx) | PP |
| SWE-ZERO `nvidia/SWE-Zero-openhands-trajectories` | 318k (paper: 300k from 150k PRs, 3.5k repos) | Qwen3-Coder-480B | OpenHands (str_replace_editor, execute_bash, think, finish), **execution forbidden** | CC-BY-4.0; permissive-repo only | Python | 10–30 turns; ~40% fewer tokens than execution-based runs | not outcome-filtered; rule filters | 7B 46.8 / 14B 54.5 / 32B 57.5 | PP |
| SWE-Hero `nvidia/SWE-Hero-openhands-trajectories` | 34k on card (13.2k used in paper) | Qwen3-Coder-480B | OpenHands with execution | CC-BY-4.0 | Python | 20–60 turns | not filtered by resolved | Qwen2.5-Coder 7B 52.7 / 14B 60.8 / 32B 62.2 after SWE-ZERO | PP |
| Open-SWE-Traces `nvidia/Open-SWE-Traces` | 207,489 at v1.0, cut to **151k** after git-hack removal (Aug 2026); v1.1/1.2 add DeepSeek-V4-Flash, Qwen3.6/3.8-27B, mini-swe-agent | MiniMax-M2.5 (thinking), Qwen3.5-122B (non-thinking) | OpenHands + SWE-agent; `role/content/tool_calls` schema | CC-BY-4.0; MIT/Apache/BSD repos | **23.2%** (48k trajectories) | avg 58–131 assistant turns; 234–313 tokens per turn | resolved ≈40.6%; drops incomplete, empty, test-tampering and tool-error trajectories | Qwen3-Coder-30B-A3B 61.7 | PP |
| Nemotron `nvidia/Nemotron-SWE-v1` | 51k (R2E-Gym tasks) | Qwen3-Coder-480B | OpenHands | CC-BY-4.0 (+BSD subsets) | Python (R2E-Gym) | n/a | n/a | n/a | PP card |
| Nemotron `nvidia/Nemotron-SFT-SWE-v3` | 237,970 | mixed | OpenHands, SWE-agent, mini-SWE-agent | CC-BY-4.0 (+Apache/MIT/BSD); "ready for commercial use" | not stated | n/a | n/a | n/a | PP card |
| CoderForge `togethercomputer/CoderForge-Preview` | 258k raw (SWE-Rebench 77k, SWE-smith 148k, R2E 33k); `filtered_reward1` 155k | not on card | **single scaffold, fixed tool set** | **no dataset-level licence**; a per-row `license` column | mostly Python (source sets) | ~47–54k tokens/trajectory (INFERRED from byte sizes; see Inferences) | `reward` field | Qwen3-32B 23.0→59.4 | PP card |
| Scale-SWE `AweAI-Team/Scale-SWE-Distilled` | 71k trajectories, 3.5B tokens; 20k released task instances | DeepSeek v3.2 | not on card | badge CC-BY-4.0, but API `license` field empty | Python | ≈49k tokens/trajectory (INFERRED: 3.5B/71k) | n/a | Qwen3-30B-A3B → 64.0 | PP |
| R2E-Gym `R2E-Gym/R2EGym-SFT-Trajectories` | 3.3k valid (4.6k tasks, 10 repos) | Claude | R2E-Gym / OpenHands-like | **no licence on card** | Python | n/a | resolved | Qwen2.5-Coder-32B 34.4 | **PR COLM 2025** (via citation) |
| SWE-Fixer `internlm/SWE-Fixer-Train-110K` | 110k issues, 856 repos | GPT-4o-distilled | **Agentless 2-stage (retrieve + edit), not multi-turn** | MIT | Python | n/a | none | 72B → 32.8 | **PR ACL Findings 2025** (via citation) |
| SERA (allenai, Open Coding Agents) | 200k+ (121 SWE-smith repos); per-repo specialisation sets of 46–54k | GLM-4.5-Air, GLM-4.6 | SWE-agent (view, edit, submit, bash), full reasoning traces | not checked (gap) | Python | "majority exceed 32k" for specialisation data | soft-verified or unverified | Qwen3-32B 49.5 at 32k context / 54.2 at 64k | PP |
| Kwai `SWE-smith-mini_swe_agent_plus-trajectories-66k` | 65,994 (all successful) | unknown | mini-swe-agent-plus, **no function calling** | n/a (cited in Nebius card) | Python | avg 34.3 turns | all successful | n/a | SNIPPET (Nebius card table) |
| DeepSWE, SWE-Dev, CodeScout | — | — | — | — | — | — | — | DeepSWE-32B 42.2 (RL); SWE-Dev-32B 36.6 (RL); CodeScout = localisation RL | No SFT trajectory release found for these; see the prior notes for CodeScout |
| Kimi-Dev | No new SWE-agent data; adapted with the 5,016 SWE-smith trajectories | — | SWE-agent | — | — | 64k training context | — | Kimi-Dev-72B 48.6 (agentic) | PP (arXiv v3) |

**Sources:**
- [SWE-Gym](https://www.alphaxiv.org/abs/2412.21139) — PDF header "PMLR 267, 2025".
- [SWE-smith](https://www.alphaxiv.org/abs/2504.21798)
- [Nebius OpenHands card](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)
- [Nebius SWE-agent card](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
- [SWE-Lego](https://www.alphaxiv.org/abs/2601.01426) and its [card](https://huggingface.co/datasets/Lego-X/SWE-Lego-Real-Data)
- [SWE-ZERO/HERO](https://www.alphaxiv.org/abs/2604.01496) and the cards ([Zero](https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories), [Hero](https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories))
- [Open-SWE-Traces](https://www.alphaxiv.org/abs/2606.16038) and its [card](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)
- [Nemotron-SFT-SWE-v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3); [Nemotron-SWE-v1](https://huggingface.co/datasets/nvidia/Nemotron-SWE-v1)
- [CoderForge](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview)
- [Scale-SWE](https://huggingface.co/datasets/AweAI-Team/Scale-SWE-Distilled)
- [SERA](https://www.alphaxiv.org/abs/2601.20789)
- [Kimi-Dev](https://www.alphaxiv.org/abs/2509.23045)
- Licences pulled from the HF API (`/api/datasets/<id>`) this session.

**Supporting facts**
- In the SWE-Lego table of public datasets, valid trajectories are: R2E-Gym 3.3k, SWE-Gym 491, SWE-smith 5k, SWE-Lego 14.1k + 4.0k. SWE-Fixer and SWE-Swiss release only sub-trajectories. VERIFIED — [SWE-Lego Table 1](https://www.alphaxiv.org/abs/2601.01426)
- Open-SWE-Traces scrubs "git hacking" with an AST-based TrajectoryScanner that bans `git log`, `reflog`, `blame` and similar. SWE-Lego and SWE-ZERO also strip future commits. Their reason: older trajectories (and older SWE-V numbers) may be inflated by solution leakage. VERIFIED — [Open-SWE-Traces §2.4](https://www.alphaxiv.org/abs/2606.16038); [SWE-Lego](https://www.alphaxiv.org/abs/2601.01426); [SWE-ZERO §3](https://www.alphaxiv.org/abs/2604.01496)
- The Nebius card warns that tool-call `arguments` are serialised as strings, and recommends deserialising them before applying the chat template so that training matches inference formatting. VERIFIED — [card](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)
- CoderForge's own card lists a limitation: "generated all trajectories using a single scaffold and fixed tool set… may perform worse when deployed with different scaffolds, tools, prompt templates, or tool-call formats". VERIFIED — [card](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview)
- Cross-harness penalty. Training on one harness and evaluating on the other costs **1.0–4.6 pp** on SWE-V and up to 4.6 pp on Multilingual (e.g. MOH→MSWEA 56.0→53.9 thinking; MSWEA→MOH 59.5→55.3 no-think). Training on SWE-agent-style data transferred better. VERIFIED — [Open-SWE-Traces Table 5](https://www.alphaxiv.org/abs/2606.16038)
- Gemma 4's native tool-call syntax is `<|tool_call>call:func{key:<|"|>value<|"|>,…}<tool_call|>`, with unquoted keys and `<|"|>` string delimiters. It is not OpenAI JSON. SNIPPET — [HF forum: Gemma 4 bug fixes](https://discuss.huggingface.co/t/gemma-4-bug-fixes-and-research-request/176979); [vLLM gemma4 tool-parser docs](https://docs.vllm.ai/en/latest/api/vllm/tool_parsers/gemma4_utils/)

### Inferences
- **Conversion map to our schema** (INFERRED):
  - OpenHands / SWE-agent `execute_bash` / `bash` → `run_command`.
  - `str_replace_editor view` → `read_file`, or (matching our v2 scaffold, which removed read_file) `run_command "sed -n 'a,bp' f"`.
  - `str_replace` → `edit_file(old_string, new_string)`; `create` → `write_file`.
  - `finish` / `submit` → `submit_patch`.
  - `think` → fold into assistant text.
  - `insert` and `undo_edit` have no direct target: rewrite as `edit_file` or drop the trajectory.
  - Observations must be re-rendered in our harness's output style and **capped at 5,000 chars**.
  - mini-swe-agent / bash-only data (Kwai 66k, Open-SWE-Traces v1.2) needs heredoc/sed edits turned into `edit_file`. That is lossy, so deprioritise it.
  - Difficulty: SWE-agent ≈ OpenHands (easy, mechanical, ~1 day of scripting); mini-swe-agent (medium); SWE-Fixer / Agentless (hard; not agentic).
- **Length screen** (INFERRED, numbers from the table):
  - SWE-Gym (~19k) and Nebius SWE-agent (8.4k resolved) mostly fit.
  - SWE-smith fits ~77%.
  - Nebius OpenHands resolved (avg 47k) and Open-SWE-Traces mostly do **not** fit without truncation.
  - SWE-ZERO's 10–30 turns likely fit best among the large sets.
  - Expect ≥50% loss on the OpenHands-480B sets.
- **CoderForge length estimate** (INFERRED): computed from the tokenized config (int32 `input_ids` + int64 `labels` + the template string ≈ 15.5 B/token). SWE_Rebench 64.2 GB/77k ≈ 54k tokens. SWE_Smith ≈ 47k. R2E ≈ 47k. Treat as rough.
- **Licence ranking for a must-open-source winner** (INFERRED):
  - Cleanest: MIT (SWE-Gym, SWE-smith, SWE-Fixer), Apache-2.0 (SWE-Lego), CC-BY-4.0 (Nebius OpenHands, NVIDIA sets), all of which require attribution.
  - Avoid or check: Nebius SWE-agent-trajectories (Llama 3.1 output-licence notice), CoderForge and R2E-Gym (no dataset-level licence), SERA (unchecked).
  - Teacher ToS: Claude/GPT-4o outputs (SWE-Gym, SWE-smith) carry provider terms on training competing models. Qwen3-Coder / GLM / MiniMax / DeepSeek open-weight teachers avoid this. This is a legal question, not a research one; flag it for the owner.
- **Format risk is ours, not the data's.** Every dataset must be rendered through *Gemma-4's own chat template, exactly as ADK + vLLM 0.19.1 re-render history at inference*. Any mismatch (JSON vs `<|"|>` strings) is trained in. INFERRED.

### Gaps
- No per-trajectory token-length distributions were computed for SWE-ZERO, Nemotron-v3 or SWE-Lego. **Next step:** stream 1k rows of each, tokenise with the Gemma-4 tokenizer, and report the fraction ≤24k and ≤32k after re-rendering.
- Not checked: SERA's licence; the R2E-Gym trajectory licence (card empty); the CoderForge teacher; the Nemotron-SFT-SWE-v3 composition and Python share.
- No public dataset uses Gemma-4's native tool format or our 6-tool ADK schema, so conversion is unavoidable.

---

## Q2. Recipes: RFT / STaR / expert iteration, distillation, loss masking, >32k handling, mixing, epochs, data scaling

### Takeaway
Under SFT, teacher distillation dominates self-improvement. Gains are roughly log-linear in trajectory count with no saturation reported up to 8k (Skywork) or 150k (SWE-ZERO).

For a LoRA budget the most useful levers are cheap ones:
1. **Mask, don't drop.** Step-level error masking: +2.0–2.2 pp. Segment-level masking: +3–8 pp.
2. **Select hard, diverse, low-retry trajectories** (SWE-Prime: 10% beats 100%).
3. **Truncate ordered, not random.** Keep trajectories whose first ≥88–95% of steps fit; ordered vs random truncation is 43.0 vs 37.5.
4. **Cap ≤2–3 trajectories per instance.**
5. **Keep reasoning text.** Removing it took SERA 41.0 → 23.0.

On-policy self-improvement (RFT on the agent's own rollouts) has **failed or given ≤1–3 pp** in the only SWE agent tests found.

### Cited Findings

**Teacher distillation vs self-improvement**
- SWE-Gym, OpenHands scaffold. The 32B was fine-tuned on 868 on-policy successes plus the 491 teacher trajectories. SWE-bench Lite **dropped from 15.3 to 8.7**: "self-improvement is not yet working". VERIFIED — [SWE-Gym §4.2](https://www.alphaxiv.org/abs/2412.21139)
- SWE-Gym, MoatlessTools scaffold (short, constrained workflow). Iterative RFT used 30 rollouts per task at T=1.0. Results:
  - 7B: 7.0 → 9.0 → 10.0 on Lite.
  - 32B (LoRA r64, LR 5e-4, 5 epochs): 19.0 → 19.7, then no further gain.
  - **Per-instance cap of 2** beat no cap (9.7 vs 9.3, 7B), preferring shorter trajectories.
  - VERIFIED — [SWE-Gym Tab. 4, 6, App. B.3](https://www.alphaxiv.org/abs/2412.21139)
- SWE-smith: resolved-only RFT beats unfiltered random trajectories at every size (100 → 1.6k: 14.3/22.4/27.8/30.1/33.4 vs 10.2/19.7/18.3/23.4/27.8). Easy, repeatedly solved instances hurt, so they capped at 3 per instance. VERIFIED — [SWE-smith Fig. 23, §4](https://www.alphaxiv.org/abs/2504.21798)
- Self-distillation is advocated by our earlier SkillForge reading ("skills are model-specific"). No SWE paper found shows multi-turn RFT on a 30B-class *own-policy* agent beating teacher SFT. See the Q4 gaps. INFERRED from the absence of results.

**Data scaling (how many trajectories for how many pp)**

| Source | Student | Points (trajectories → SWE-V %) | Status |
|---|---|---|---|
| Skywork-SWE | Qwen2.5-Coder-32B, full FT, 3 epochs, 32k ctx | 0→6.4, 125→10.8, 250→16.1, 500→18.4, 1k→20.8, 2k→25.2, 4k→28.8, 6k→32.0, 8.2k→38.0; "log-linear… no signs of saturation" | PP |
| SWE-smith | Qwen2.5-Coder-32B | 100→14.3, 200→22.4, 400→27.8, 800→30.1, 1.6k→33.4, 5k→40.2 | PR |
| SWE-Gym | Qwen2.5-Coder-32B | 491 → 20.6 (+13.6), "no signs of saturation at 491" | PR |
| SWE-ZERO | 14B, execution-free | 4k PRs→42.1 … 32k→46.4 … 150k→49.1; fastest gains 4k–32k | PP |
| 2607.17205 | Qwen2.5-Coder-7B **QLoRA** r64, 32k ctx | Held-out CE loss: 500→1k −12.7%, 1k→2k −6.5 to −9.1%. Top-quality vs random gap <1% (n.s.) at 500–1k, 3.6% (p=0.016) at 2k. **No resolve rate** (7B ≈ 0) | PP |
| Kimi-Dev | Qwen2.5-72B priors | SWE-smith subsets 100 → 5,016. A strong prior reaches the base model's best pass@1 with 2²³ tokens (~100 trajectories) vs 1.5×2²⁸ tokens | PP |
| SERA | Qwen3-32B full FT | Power law, R² > 0.95, mean prediction error 0.4%; repo specialisation matches the teacher at ~8k samples per repo | PP |

Sources: [Skywork-SWE Fig. 1](https://www.alphaxiv.org/abs/2506.19290); [SWE-smith Fig. 1/23](https://www.alphaxiv.org/abs/2504.21798); [SWE-Gym Fig. 1](https://www.alphaxiv.org/abs/2412.21139); [SWE-ZERO Fig. 4](https://www.alphaxiv.org/abs/2604.01496); [2607.17205 Tab. 2–3](https://www.alphaxiv.org/abs/2607.17205); [Kimi-Dev §4.2](https://www.alphaxiv.org/abs/2509.23045); [SERA §4, §6](https://www.alphaxiv.org/abs/2601.20789). All VERIFIED.

**Selection / filtering (quality vs quantity)**
- SWE-Prime ([PP], Aug 2026) starts from Nebius's 32,161 resolved trajectories.
  - Stage 1 keeps the top 10% per cluster on:
    - observe→edit→verify pattern;
    - tool-call success rate;
    - no adjacent identical calls;
    - no git hacking;
    - patch quality vs gold.
  - Stage 2 masks the loss on low-value 20-step "segments" (they stay as context).
  - Results:

    | Student | Full resolved SFT (100%) | Random 10% | SWE-Prime (10%) | Stage 1 only |
    |---|---|---|---|---|
    | Qwen3-Coder-30B-A3B | 51.0 | 41.2 | **53.2** | 50.2 |
    | GLM-4.7-Flash | 41.4 | — | **51.4** | — |
    | Qwen3-30B-A3B | 36.8 | — | **39.6** | — |

  - Retention ratio sweep: 5% → 46.2, 10% → 50.2, 20% → 49.6, 30% → 49.2.
  - Behaviour: redundancy down by up to 0.188, tool success up by up to 0.094, 4–13 fewer turns.
  - VERIFIED — [SWE-Prime Tab. 1, Fig. 4–5](https://www.alphaxiv.org/abs/2608.27449)
- Including unresolved trajectories can help when the data is plentiful. Open-SWE-Traces, resolved-only → full: SWE-V 55.3→58.1 (think), 57.7→59.8 (no-think); Multilingual +7 pp. VERIFIED — [Open-SWE-Traces Tab. 5](https://www.alphaxiv.org/abs/2606.16038)
  - SWE-Lego: adding 4k semi-resolved (correct localisation, failed tests) 41.0→42.2; removing low-quality resolved 40.4→41.0. VERIFIED — [SWE-Lego Tab. 5](https://www.alphaxiv.org/abs/2601.01426)
  - SERA: unverified vs hard-verified trajectories are statistically indistinguishable (Kruskal-Wallis p=.066). VERIFIED — [SERA §5.1](https://www.alphaxiv.org/abs/2601.20789)
  - **Conflict:** SWE-smith found resolved-only > random unfiltered. INFERRED reconciliation: verification matters less than teacher quality and cleanliness, and the SWE-smith "no filter" arm also had more broken trajectories.
- Difficulty is not a strong lever. SWE-smith difficulty-binned SFT sets (scores 2/4/6/8) gave 12.4/10.8/13.6/12.2 (7B). VERIFIED — [SWE-smith §4.1](https://www.alphaxiv.org/abs/2504.21798)
- SWE-Lego's length-based curriculum (easy <50 turns → medium → hard) adds only +0.4/+0.8. VERIFIED — [SWE-Lego Tab. 6](https://www.alphaxiv.org/abs/2601.01426)
- In 2607.17205 the best single filter is **error-retry rate** (action → error → similar action), matching the full composite (Δ<0.2%). VERIFIED — [2607.17205 §4.5](https://www.alphaxiv.org/abs/2607.17205)

**Loss masking**
- Standard practice is assistant-only loss: tool outputs and observations are masked (SWE-ZERO/HERO, 2607.17205, SWE-Prime). VERIFIED — [SWE-ZERO §3.2](https://www.alphaxiv.org/abs/2604.01496); [2607.17205 §3.4](https://www.alphaxiv.org/abs/2607.17205)
- **Step-level error masking** (SWE-Lego): regex on the environment's error messages, then mask the *agent turn that caused the error* while keeping it as context. Errors from running repro/test scripts are not masked.
  - Masking alone (with curriculum): 8B 40.2→42.2, 32B 50.4→52.6.
  - Masking + curriculum vs neither: 39.4→42.2 and 48.8→52.6.
  - VERIFIED — [SWE-Lego §3.1, Tab. 6](https://www.alphaxiv.org/abs/2601.01426)
- **Segment-level masking** (SWE-Prime Stage 2): +3.0 to +7.8 pp over Stage 1 alone. VERIFIED — [SWE-Prime Tab. 1](https://www.alphaxiv.org/abs/2608.27449)
- Kimi-Dev's mid-training *did* train on observations (masking only the system prompt) as "world modeling". That was ~10B tokens of synthetic agentic data, not a LoRA-scale recipe. VERIFIED — [Kimi-Dev App. A](https://www.alphaxiv.org/abs/2509.23045)

**Handling >32k trajectories**
- SERA:
  - 23.26% of SWE-smith and 24.83% of its own data exceed 32k tokens.
  - **Ordered truncation** (choose trajectories with the highest "truncation ratio" <1, keep the first steps) scored **43.0 ± 1.9** vs **40.6 ± 0.7** for fully-fitting-only and **37.5 ± 0.5** for random truncation (p=.009).
  - The best partition was truncation ratio ≈0.95; they scale only down to 0.88.
  - The model's final steps are "typically focused on the redundant task of submitting".
  - VERIFIED — [SERA §5.3, Tab. 4](https://www.alphaxiv.org/abs/2601.20789)
- Skywork notes that trajectories >50 rounds exceed 32k and planned sequence parallelism to reach 128k. SWE-Gym kept only ≤32k trajectories for verifier training. VERIFIED — [Skywork §5](https://www.alphaxiv.org/abs/2506.19290); [SWE-Gym fn. 8](https://www.alphaxiv.org/abs/2412.21139)
- Most SOTA SFT recipes train at 128k (SWE-Lego, SWE-ZERO/HERO, Open-SWE-Traces, SWE-Prime 131k). Their numbers therefore do **not** directly transfer to a 32k budget. VERIFIED (configs) — see papers above.
- Execution-free trajectories (SWE-ZERO) are ~40% shorter and span 10–30 turns. VERIFIED — [SWE-ZERO §3.1](https://www.alphaxiv.org/abs/2604.01496)
- Compression edits (other domain):
  - Dependency-DAG "leaf prune" removes dead-end rounds (failed calls, abandoned queries): +1.7 pp accuracy with ~30% fewer inference tokens vs vanilla SFT.
  - LLM "critical-path" deletion *hurt* (−4 pp).
  - Merging rounds into multi-call turns that the inference loop never emits raised stuck/loop rates 2–3×.
  - Multimodal QA, not code: [PP], VERIFIED — [2609.18417](https://www.alphaxiv.org/abs/2609.18417)
- Summarisation-in-trajectory (AutoCompact, 2610.02163) exists but conflicts with our no-compaction harness. Not read. SNIPPET — [alphaXiv listing](https://www.alphaxiv.org/abs/2610.02163)

**Mixing, staging and reasoning traces**
- Real + synthetic mix (SWE-Lego):
  - The dataset gives +25.6 pp over base Qwen3-32B (23.2→48.8).
  - Adding synthetic SWE-smith-style bugs per repo keeps raising the resolve rate.
  - Real PRs touch 3.7 files / 138 lines; synthetic ones 1.0 file / 18.8 lines.
  - VERIFIED — [SWE-Lego Tab. 3, Fig. 1b/3](https://www.alphaxiv.org/abs/2601.01426)
- Two-stage, execution-free then execution-based (SWE-ZERO→HERO): 62.2 vs 55.7 for HERO-only (32B), and Multilingual 44.1 vs 30.8. VERIFIED — [SWE-ZERO §4.2](https://www.alphaxiv.org/abs/2604.01496)
- Multilingual data does not dilute Python: Python-only → full raised SWE-V 54.9→58.1 (think). VERIFIED — [Open-SWE-Traces Tab. 5](https://www.alphaxiv.org/abs/2606.16038)
- Reasoning traces:
  - Removing reasoning traces from GLM-4.5-Air trajectories took Qwen3-32B from **41.0 to 23.0**. VERIFIED — [SERA Tab. 8](https://www.alphaxiv.org/abs/2601.20789)
  - Open-SWE-Traces found /no_think ≥ /think at 30B-A3B (61.7 vs 59.3). VERIFIED — [Open-SWE-Traces Tab. 3](https://www.alphaxiv.org/abs/2606.16038)
  - INFERRED reconciliation: visible "thought" text before an action matters; long hidden CoT is optional.

**Epochs and learning rates (as reported)**

| Recipe | Epochs | LR | Context | Other |
|---|---|---|---|---|
| SWE-Gym | ≤5 | 1e-4 full FT; LoRA r64 at 5e-4 | 32k | — |
| SWE-smith | 3 | 5e-5 | 32k | — |
| Skywork-SWE | 3 | 5e-5 | 32k | — |
| SWE-Lego | 4 | 1e-4 (8B), 5e-5 (32B) | 128k | resolve 7.6/14.6/33.4/38.4/42.2 after epochs 0–4 |
| SERA | 3 | 1e-5 | 32k | — |
| SWE-ZERO/HERO | ≤3 | 1e-5 peak | 128k | — |
| 2607.17205 (QLoRA) | 2 | 2e-5 | 32k | r64, α128, NF4, 8-bit paged AdamW, ~150 A100-h for 16 runs at 7B |

VERIFIED — respective papers linked above.

### Inferences
- **Budget translation** (INFERRED, low confidence):
  - Skywork trained 8.2k × 3 epochs at 32k on 8×H800 in 12 h (~96 H800-h, full FT).
  - 4×L4 bf16 throughput is roughly 1/6–1/10 of 8×H100-class per hour, and QLoRA on a 31B base costs more per token than LoRA on a bf16 base.
  - So ~30 L4-h/week likely supports on the order of **1–3k trajectories × 1–2 epochs at ≤24k tokens**.
  - Per Skywork/SWE-smith curves at 32B, that is the regime where 1k → 2k still gives +3–4 pp per doubling.
  - **Measure tokens/s on the L4 rig before planning** (plan Phase 0 item).
- **Recipe for us** (INFERRED):
  - (a) Convert SWE-smith (MIT) + SWE-Gym (MIT) + the ≤32k / ordered-truncated subset of Nebius-OpenHands and SWE-ZERO (CC-BY-4.0).
  - (b) Filter on error-retry, adjacent-repeat, git-hack and tool-success.
  - (c) Cap 2–3 per instance and prefer repo diversity.
  - (d) Assistant-only loss **plus** SWE-Lego error-masking.
  - (e) Ordered truncation at ratio ≥0.88.
  - (f) 2 epochs.
  - (g) Then one round of RFT on our own Gemma successes on held-out repos as an add-on, not a replacement, given SWE-Gym's negative self-improvement result.
- 2607.17205's "quantity beats quality below ~1k" (QLoRA 7B, CE proxy) and SWE-Prime's "10% beats 100%" (30B MoE, full FT, resolve rate) conflict on the surface. They are in different regimes: 500–2k vs 3.2k-of-32k. **At our budget (≤3k) both say: take the cleanest ~3k, not a random 3k.** INFERRED.

### Gaps
- No SWE-agent study of **LoRA** SFT at 30B+ reporting resolve rate vs number of trajectories. The 32B scaling curves are full fine-tunes; 2607.17205 is 7B with a CE proxy.
- No published RFT/STaR result for a ≥30B *tool-calling* (not Moatless) SWE agent using its own successes.
- No study of ordered truncation vs DAG leaf-pruning for code trajectories.

---

## Q3. Tool-format reliability: what reduces malformed calls and loops?

### Takeaway
Three separable fixes, in order of cost:
1. **Make sure the serialisation round-trip is correct.** Gemma-4's malformed-argument loops are documented as a *protocol-boundary* bug in chat templates and vLLM parsers, not a weight problem.
2. **SFT on format-correct trajectories filtered for tool errors and repeats.** This sharply raises format validity (Gemma-4-12B tool-param exact match 0–6% → 63–73% after LoRA SFT on BFCL), and it reduces loops (SWE-Gym: −5.6 to −18.6 pp stuck-in-loop).
3. **Constrained decoding.** It guarantees structural validity but cannot fix semantic errors. Under our declarative submission we probably cannot enable it on the scorer.

Repetition loops are only partly fixed by SFT. SWE-smith's SFT model still had ≥10-long repeats in >25% of trajectories, and scaffold anti-repeat interventions "do not seem to improve resolve rates".

### Cited Findings
- **Gemma 4 protocol-boundary failure.** Gemma emits a correct native call. The stack converts it to OpenAI `tool_calls` with JSON-string arguments, then re-renders it into the Gemma prompt as hybrid syntax (e.g. `call:write_file{{"path":…}}`). The model then "imitates malformed history".
  - Fixes listed: update template, tokenizer and backend; disable streaming for tool calls; never feed malformed turns back; add retry caps and duplicate suppression.
  - A Jun 20 2026 update says "VLLM v23 finally fixed the issues". vLLM issues #38946, #39380, #41967 are cited.
  - SNIPPET (forum post, not peer-reviewed) — [HF forum](https://discuss.huggingface.co/t/gemma-4-bug-fixes-and-research-request/176979)
- The vLLM gemma4 tool parser handles `key:<|"|>value<|"|>` with "fallback to heuristic key-value extraction", and tolerates `key: "value"` from some templates. SNIPPET — [vLLM docs](https://docs.vllm.ai/en/latest/api/vllm/tool_parsers/gemma4_utils/)
- **SFT fixes most format errors on Gemma 4.** ParamBench / BFCL / API-Bank with LoRA (attention-only) on **Gemma-4-12B**:
  - BFCL EM 5.7 → 63.1 (SeedSFT) → 73.1 (probe-filtered bootstrapped training, PBT).
  - xLAM 0.4 → 35.6 → 74.2.
  - Averaged over 5 models, "SeedSFT fixes most of the format errors" (19.7 → 51.6 EM).
  - PBT adds +8.0 EM by self-training on the model's own calls, filtered by a hidden-state linear probe (AUC 0.986 in-domain, 0.71–0.77 cross-dataset).
  - Among frontier models' failed calls, only 2.1% violate the schema; the rest are value errors.
  - [PP], VERIFIED — [2608.03071](https://www.alphaxiv.org/abs/2608.03071)
- **Constrained decoding** (Outlines / XGrammar) raised schema validity from 78.6–92.9% to 100% on 0.6–4B models. XGrammar cost 4–8 ms compile and ~2–4% throughput. Content accuracy improved but "instruction-semantic" failures (e.g. a missing second call) are CD-resistant. [PP], VERIFIED — [2609.23742](https://www.alphaxiv.org/abs/2609.23742)
- **Trajectory curation for format:**
  - SWE-ZERO prunes trajectories without exactly one tool call per turn, or with ≥3 `str_replace_editor` errors.
  - Open-SWE-Traces drops "repeated tool usage errors" and illegal concurrent calls.
  - SWE-Prime scores tool-success percentile and rejects adjacent identical calls.
  - VERIFIED — [SWE-ZERO §3.1](https://www.alphaxiv.org/abs/2604.01496); [Open-SWE-Traces §2.3](https://www.alphaxiv.org/abs/2606.16038); [SWE-Prime §Stage 1](https://www.alphaxiv.org/abs/2608.27449)
- **Teacher rollout hygiene (SWE-Lego).** Rollouts were made "w/o handling malformed tool errors" vs with it, measured as avg turns 66.6 vs 62.5 and valid-trajectory rate 29% vs 30%. Pruning ineffective tools: 65.5 turns / 27%. VERIFIED — [SWE-Lego Tab. 4](https://www.alphaxiv.org/abs/2601.01426)
- **Error masking** removes the gradient on malformed or failed calls but keeps the recovery as context (+2 pp). VERIFIED — [SWE-Lego §3.1](https://www.alphaxiv.org/abs/2601.01426)
- **Loops after SFT:**
  - SWE-Gym SFT cut "stuck in loop" (same action 3× in a row) on Verified: 7B 39.6→21.0, 14B 32.1→21.3, 32B 29.4→23.8.
  - SWE-smith: >25% of SWE-agent-LM-32B trajectories contain a ≥10-long repetitive sequence vs <4% for Claude 3.7. A length-10 repeat ↔ 89% failure. 73% of the longest repeats are `str_replace_editor view` (scrolling instead of searching). "Simple interventions from the agent scaffold can mitigate repetitive actions, but do not seem to improve resolve rates".
  - Open-SWE-Agent loops in 5–10% of MOpenHands cases.
  - VERIFIED — [SWE-Gym Tab. 3](https://www.alphaxiv.org/abs/2412.21139); [SWE-smith §4.2](https://www.alphaxiv.org/abs/2504.21798); [Open-SWE-Traces §3.2](https://www.alphaxiv.org/abs/2606.16038)
- 2607.17205 reports **0% well-formed `<tool_call>` JSON** after QLoRA SFT on Nebius data at all 16 configs. They attribute it to class imbalance (tool_call 0–3% of assistant turns vs bash/edit). The 300-token generation budget also truncated outputs, so this is partly a measurement artefact. VERIFIED — [2607.17205 §5.2, §4.2](https://www.alphaxiv.org/abs/2607.17205)
- Other related work found but not read:
  - ContractRL (field-level tool-call repair RL);
  - CAST (critique-aware supervision for long-horizon tool calling);
  - EDGEGEN (edge-case synthesis);
  - GrammarRL.
  - SNIPPET — [alphaXiv results](https://www.alphaxiv.org/abs/2610.00328), [CAST](https://www.alphaxiv.org/abs/2608.30147)

### Inferences
- **Our 61% malformed edit_file rate is most likely a serialisation round-trip bug.** Code containing `"` in `old_string` is precisely where JSON-string vs `<|"|>` delimiters collide. Before any LoRA, dump the exact prompt text vLLM 0.19.1 receives for turn n+1 after a quoted edit. If history is re-rendered as hybrid syntax, the model is being *taught in-context* to emit broken calls. A LoRA trained on clean history cannot fully override corrupted in-context examples. INFERRED (high priority, cheap).
- **"Correction" data (malformed → correct target).** Train on contexts that *contain* a malformed call plus the error message, with the loss only on the corrected next call. This is SWE-Lego's error masking applied to our own failure traces, and it teaches recovery without imitating the error. No SWE paper isolates this, so it is untested. INFERRED.
- **Constrained decoding on the scorer is likely unavailable.** The submission is declarative YAML, and vLLM server flags are organiser-controlled. LoRA plus clean data is the only lever. INFERRED; verify whether `eval_config.yaml` exposes `guided_decoding` / `tool_choice`.
- **Loops.** Data hygiene (drop adjacent-repeat trajectories, prefer grep over scroll) plus a hard harness repeat-guard are complementary. Expect SFT to cut loop rate by roughly a quarter to a half (SWE-Gym), not to zero. INFERRED.

### Gaps
- No paper measures malformed-call rate before and after SFT **for long SWE trajectories with string-heavy code arguments**. ParamBench is short API calls.
- Unverified whether vLLM 0.19.1 (the scorer's version) contains the gemma4 template/parser fixes. The forum says the fix landed in vLLM "v23".
- No SWE study of explicit self-correction data, i.e. synthetic error → recovery pairs as a separate data slice.

---

## Q4. Transfer to unseen repos, and how much survives LoRA on a quantised base

### Takeaway
- **Repo diversity is the main generalisation lever.** Performance rises roughly log-linearly with the number of training repos (SWE-smith).
- **Repo-specific data beats general data** (+4–8 pp; SWE-smith SymPy, SERA). We cannot exploit that for private hidden repos at training time.
- **Same-repo train/eval inflates results** (Skywork warning). Our local dev numbers on fastapi/rich/requests are suspect.
- **Cross-harness costs 1–5 pp,** so train in our exact harness format.
- LoRA vs full FT and the quantised base: there is **no SWE-agent study** of LoRA on a 4-bit 30B-class base. Indirect evidence (verifier LoRA > full FT; tool-calling LoRA > full SFT; FP8 serving "minimal effects") is encouraging. The NF4-train / W4A16-serve mismatch is untested.

### Cited Findings
- **Number of repos.** At 700 trajectories (7B), 5→25→50→100 repos gave 10.3→11.5→12.9→15.1% ("approximately logarithmic"). VERIFIED — [SWE-smith Fig. 5](https://www.alphaxiv.org/abs/2504.21798)
- **Scaling unique instances vs repos.** SWE-Gym compared scaling trajectories (dedup), unique instances, and repositories at the 32B scale. VERIFIED (figure only; no numeric deltas extracted) — [SWE-Gym Fig. 9](https://www.alphaxiv.org/abs/2412.21139)
- **Repo specialisation:**
  - SWE-smith: SymPy-only 700 trajectories gave 7B 13.6→21.2 on SymPy with a small general drop. Specialist stage on 32B: 33.3→42.4.
  - SERA: 8k per-repo samples gave Django 52.2, SymPy 51.1 vs general 45.6 / 48.9. Specialised vs general +4.3 pp (SNR 5.6). Data mix α (share of repo data) "is the most important factor".
  - VERIFIED — [SWE-smith §4.1](https://www.alphaxiv.org/abs/2504.21798); [SERA §4.3, Tab. 7](https://www.alphaxiv.org/abs/2601.20789)
- **Leakage warning.** "Training and testing on pull request instances from the same repository can lead to inflated performance." VERIFIED — [Skywork §5](https://www.alphaxiv.org/abs/2506.19290)
- **Cross-family brittleness.** The SWE-smith recipe gives 32.6% on Qwen2.5-32B but 25.3% on Qwen3-32B; "the method may have been unintentionally optimized for the earlier model family". Run-to-run std 0.5–3.0% (median 1.2%); single-seed ablations can flip. VERIFIED — [SERA §6](https://www.alphaxiv.org/abs/2601.20789)
- **Cross-language/task transfer:**
  - Python-only SWE-ZERO→HERO reaches 44.1% on Multilingual.
  - SWE-smith-only SFT gave almost no Multilingual gain (8.4 vs 6.5 base).
  - Kimi-Dev: Agentless-trained priors generalise better to SWE-bench-Live and Multilingual when SFT data is scarce.
  - VERIFIED — [SWE-ZERO](https://www.alphaxiv.org/abs/2604.01496); [SWE-smith App. F.4](https://www.alphaxiv.org/abs/2504.21798); [Kimi-Dev App. G.4](https://www.alphaxiv.org/abs/2509.23045)
- **LoRA evidence in SWE / tool use:**
  - SWE-Gym verifier: LoRA (Unsloth) 29.8@8 vs full FT 27.2@8.
  - SWE-Gym MoatlessTools 32B policy trained with LoRA r64. The policy served in FP8 "found to have minimal effects".
  - VERIFIED — [SWE-Gym §5.1.1, App. B.3](https://www.alphaxiv.org/abs/2412.21139)
  - QLoRA r64 NF4 on 7B is viable for SWE trajectory SFT (CE drops 0.91→0.37), but resolve rate was not measured. VERIFIED — [2607.17205](https://www.alphaxiv.org/abs/2607.17205)
  - Attention-only LoRA SFT lifted Gemma-4-12B tool-parameter EM roughly 4–10×. VERIFIED — [2608.03071](https://www.alphaxiv.org/abs/2608.03071)
  - From the prior notes: LoRA beat full SFT for tool calling at 4B (88.0 vs 80.4); "LoRA Without Regret" says include the MLP layers. VERIFIED earlier — [SFT or RL](https://www.alphaxiv.org/abs/2609.17848)
- **Quantised-base adapters.** Relevant 2026 methods exist but were not read: GPTQ-intrinsic LoRA (2606.01412), Scale-QLoRA (adapter merging for 4-bit microscaling, 2609.04526), AQLoRA (2608.23816). None is evaluated on agentic SWE. SNIPPET — [GPTQ-intrinsic LoRA](https://www.alphaxiv.org/abs/2606.01412); [Scale-QLoRA](https://www.alphaxiv.org/abs/2609.04526)

### Inferences
- **For unseen private repos** (INFERRED):
  - maximise repo count (SWE-Lego 3,251 repos, Nebius 1,823, SWE-ZERO 3.5k) over trajectories per repo;
  - hold out whole repos for dev evaluation;
  - exclude fastapi/rich/requests and SWE-bench repos from training (leakage plus memorisation).
- **SERA-style test-free generation is viable for us** (INFERRED). Its pipeline is a vague bug prompt on a random function → teacher rollout → synthetic PR → second rollout → line-recall soft verification. It needs no tests, so it could generate Gemma-format data on *many additional permissive Python repos*. The specialisation advantage itself cannot be used on hidden repos.
- **Quantisation mismatch is the largest unquantified risk** (INFERRED). Options: (a) train on bf16 or QAT weights; (b) train on NF4. Then measure the same adapter on the W4A16 base in vLLM. Phase 0 showed serving works but measured no quality effect. A cheap check: per-token CE on 200 held-out trajectories under bf16 vs W4A16 with the same adapter.
- **Expect smaller SWE-V-style gains than the 32B full-FT papers** (INFERRED):
  - Gemma-4-31B starts far higher than Qwen2.5-Coder-32B (~7%), so headroom differs.
  - The papers use 128k context; we have 32k.
  - Hidden repos are unseen.
  - A realistic target is format/loop fixes plus a few pp, not +30 pp.

### Gaps
- No paper reports how much of a full-FT SWE gain survives with LoRA on a 4-bit base. **Query to log:** "QLoRA vs full fine-tuning SWE-bench resolve rate 32B agent trajectories".
- No study of adapter transfer from NF4-trained to GPTQ/compressed-tensors W4A16 serving. **Query:** "LoRA adapter trained on NF4 served on GPTQ AWQ base accuracy drop".
- No Gemma-family SWE-agent SFT result found at all. **Query:** "Gemma SWE-bench fine-tuning agent trajectories".

---

## Q5. What is untested (and queries to run next)

### Takeaway
The open space for a cheap, publishable contribution under our constraints:
1. LoRA SFT scaling for a 30B-class quantised base at a 32k budget.
2. Correction-pair data for tool-call arguments in long code-edit trajectories.
3. Ordered truncation vs DAG leaf-pruning for SWE trajectories.
4. Train-time render fidelity (native Gemma tool syntax) as an explanatory variable for malformed calls.
5. RFT on own successes *after* teacher SFT for a ~30B ADK agent, evaluated on held-out private-like repos with ≥3 seeds.

### Cited Findings
- Evaluation noise is large relative to typical gains (median std 1.2%; ~4 seeds needed to detect 2 pp at SNR ≥ 2). VERIFIED — [SERA Tab. 10](https://www.alphaxiv.org/abs/2601.20789)
- 2607.17205 explicitly leaves open replication on other corpora (SWE-smith, R2E-Gym), ≥5k scale, and "the tool-call format failure". VERIFIED — [2607.17205 §6](https://www.alphaxiv.org/abs/2607.17205)
- SERA leaves open whether soft verification still suffices at larger scale. VERIFIED — [SERA §9](https://www.alphaxiv.org/abs/2601.20789)
- Gap searches:
  - The alphaXiv discover query for "malformed function call / self-correction / repetition loops / constrained decoding" returned "No papers found".
  - A broader "tool-call format validity" query returned only general tool-calling papers (ParamBench, CD-vs-scale, ContractRL, CAST). None were SWE trajectories.
  - VERIFIED (absence within index) — see search log #9 and #10.

### Inferences
- With ~120 hidden tasks (1 task ≈ 0.8 pp) and one LB submission per day, **adapter selection must rely on dense local proxies.** Candidates:
  - held-out CE (2607.17205);
  - malformed-call rate;
  - loop rate;
  - line-level recall (SERA reports r=0.898 correlation with resolve rate across 1,036 evaluations);
  - resolve rate on held-out repos as secondary.
  
  INFERRED.

### Gaps (queries to log for the next session)
1. "LoRA vs full fine-tuning SWE-bench resolve rate trajectories 32B" (alphaXiv + Semantic Scholar).
2. "Gemma 4 SWE-bench agent fine-tune" (HF models, GitHub).
3. "vLLM 0.19 gemma4 tool parser fix quotes PR" (GitHub vllm-project issues #38946, #39380, #41967), to confirm whether the scorer's vLLM has the fix.
4. "SERA open coding agents dataset license" (HF allenai collection).
5. "R2E-Gym SFT trajectories license", "CoderForge-Preview license teacher model".
6. "Nemotron-SFT-SWE-v3 composition Python share token length".
7. "self-correction tool call error recovery SFT agent trajectories" (try ACL Anthology / Semantic Scholar; alphaXiv returned nothing).
8. "AutoCompact learning when to compact coding agents" (2610.02163): read, in case compaction-free behaviour can be distilled.
9. "GPTQ-intrinsic LoRA" (2606.01412) and "Scale-QLoRA" (2609.04526): read for the quantised-base adapter risk.
10. "git hacking trajectories SWE" (2609.06780, cited by the Open-SWE-Traces card) for the contamination filter list.

### Search log (this session)
| # | Tool | Query / target | Purpose | Outcome |
|---|---|---|---|---|
| 1 | alphaXiv discover | "open-source SWE agent trajectory datasets for SFT of 7B–32B models (SWE-Gym, SWE-smith, SWE-rebench, R2E-Gym, Skywork-SWE, SWE-Dev, SWE-Fixer, Kimi-Dev, Nemotron)" | Q1 | Open-SWE-Traces, SWE-Lego, SWE-ZERO/HERO, SWE-Prime, SWE-Gym, SWE-smith, R2E-Gym, SWE-rebench V2, daVinci-Env, Kimi-Dev, Socratic-SWE, SWE-TRACE, SWE-Master |
| 2 | alphaXiv discover | "training data and recipes to fix malformed tool calls, repetition loops, long trajectories in SWE agent SFT; data scaling" (recency) | Q2/Q3 | SWE-Prime, Dependency-Aware Trajectory Refinement, PG-SFT, FLARE, 2607.17205, AutoCompact, ContractRL |
| 3 | alphaXiv PDF Q&A | SWE-Lego 2601.01426 | Q1/Q2 | read |
| 4 | alphaXiv PDF Q&A | Open-SWE-Traces 2606.16038 | Q1/Q2/Q4 | read |
| 5 | alphaXiv PDF Q&A | SWE-Prime 2608.27449 | Q2/Q3 | read |
| 6 | alphaXiv PDF Q&A | 2607.17205 (deepened) | Q2/Q3/Q4 | read |
| 7 | alphaXiv PDF Q&A | SWE-Gym 2412.21139; SWE-smith 2504.21798; SWE-ZERO 2604.01496; Skywork-SWE 2506.19290 | Q1/Q2/Q4 | read |
| 8 | Bash/HF API | `/api/datasets/<id>` for 13 datasets; `?search=` for SWE-Lego, SWE-Zero, SWE-Hero, R2E-Gym, Kimi-Dev, SWE-Dev, Nemotron-SWE, CoderForge, SERA, Scale-SWE, daVinci-Dev, OpenSWE, SWE-Mirror; README fetch for 15 cards | Q1 licences/sizes | licences/sizes in the grid; SERA search hit unrelated datasets |
| 9 | alphaXiv discover | "reducing malformed function call arguments and repetitive action loops … self-correction … constrained decoding" | Q3 | **No papers found** |
| 10 | alphaXiv discover | "improving tool-call format validity and robustness … fine-tuning or decoding" (recency) | Q3 | SchemaFill, CD-vs-scale 2609.23742, GrammarRL, SLCA-GRPO, CAST, ParamBench 2608.03071, ToolRobustBench, EDGEGEN |
| 11 | alphaXiv discover | "LoRA adapter trained on QLoRA NF4 served on GPTQ/AWQ/W4A16; quantization-aware LoRA" | Q4 | GPTQ-intrinsic LoRA, Scale-QLoRA, AQLoRA, AGENTQ (none agentic SWE) |
| 12 | WebSearch | `vLLM gemma4 tool parser quotes arguments malformed tool call` | Q3 | HF forum thread, vLLM gemma4_utils docs |
| 13 | WebFetch | discuss.huggingface.co Gemma 4 bug-fixes thread | Q3 | protocol-boundary loop mechanism; "VLLM v23" fix claim |
| 14 | alphaXiv PDF Q&A | CD-vs-scale 2609.23742; ParamBench 2608.03071 | Q3/Q4 | read |
| 15 | alphaXiv PDF Q&A | SERA 2601.20789 (×2); DAG refinement 2609.18417; Kimi-Dev 2509.23045 | Q2/Q4 | read |

Not run (budget): GitHub repo READMEs for SWE-smith, SWE-Gym, R2E-Gym, OpenHands; the Nebius blog; the DeepSWE blog; SWE-Master and daVinci-Dev papers; Semantic Scholar venue checks for 2026 preprints.
