# PEFT methods for SWE / tool-using agents, and how to train a LoRA on Gemma-4-31B on 4× L4 (as of 8 Oct 2026)

Scope: which PEFT variants matter for agentic / tool-use SFT, whether each can ship as a plain PEFT LoRA that vLLM 0.19.1 serves on the W4A16 QAT base, the quantisation mismatch between training and serving, and the memory, throughput and engineering recipe for a 31B model on 4× L4 (24 GB, sm89, PCIe, offline, ~30 GPU-h/week, 12 h sessions).
Not repeated here: LoRA Without Regret, the Qwen3 tool-calling LoRA vs GRPO study and QeRL. These are in `research_notes/LLM graph reasoning for code/post_training_navigation.md`. Trajectory curation for LoRA (arXiv 2607.17205) is in `docs/literature_2025_2026.md`.

Labels:
- **[PR]** = peer-reviewed. **[PP]** = preprint. **[DOC]** = official docs or code.
- **VERIFIED** = I read it in the primary source this session.
- **SNIPPET** = search-result excerpt only; I did not read the full source.
- **INFERRED** = my own reasoning or arithmetic, not stated by any source.

Project facts reused below come from `docs/HQ.md` (Oct 7 entries):
- rank-8 adapter over 410 modules = 122.6 MB, so roughly 15.3 MB per unit of rank;
- two attention shapes: q 8192/16384 and k 4096/2048;
- vLLM serves synthetic PEFT adapters on the W4A16 base, and zero-LoRA output equals base output.

---

## Q1. Which LoRA variants help on agent/tool-use/code SFT, and which can ship as a plain PEFT LoRA that vLLM serves?

### Takeaway
Only adapters that reduce to **plain LoRA A/B matrices with `bias="none"`** can be shipped. These are vanilla LoRA, rsLoRA and LoRA+ (LoRA+ only changes the optimizer). PiSSA/OLoRA/LoRA-GA-style inits can also ship after conversion.
- **DoRA is rejected by vLLM outright.**
- VeRA, AdaLoRA-with-routing and Mixture-of-LoRA routers are not servable.

On agent/tool tasks, the only controlled PEFT-variant comparison I found shows DoRA ≈ LoRA (+1.3 pts on BFCL). That gap is small next to the gains from data and role decomposition. So the variant matters much less than the data and the target modules.

### Cited Findings
**Servability in vLLM (hard constraint)**
- vLLM's `PEFTHelper` validation rejects the following, quoting its error strings:
  - `use_dora`: "vLLM does not yet support DoRA."
  - `bias` other than "none": "Adapter bias is not supported."
  - `modules_to_save` other than None or ["classifier","score"].
  - `r` greater than `max_lora_rank`.
- vLLM accepts `use_rslora` and uses scaling `lora_alpha/√r`; otherwise the scaling is `lora_alpha/r`.
- VERIFIED [DOC, vLLM main branch, not the 0.19.1 tag] — [vllm/lora/peft_helper.py](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/lora/peft_helper.py)
- The vLLM LoRA user docs don't mention DoRA, rsLoRA, embeddings or quantized bases at all. Only the source code is authoritative. VERIFIED [DOC] — [vLLM LoRA docs](https://docs.vllm.ai/en/latest/features/lora.html)
- vLLM does not support LoRA on `embed_tokens` for GPTQ-quantized bases. SNIPPET — [vLLM forum](https://discuss.vllm.ai/t/support-for-deploying-4-bit-fine-tuned-model-with-lora-on-vllm/1186)

**PEFT-side support on quantized bases**
- PEFT trains LoRA on these quantized bases: bitsandbytes 4/8-bit, GPTQ (via GPT-QModel ≥7.0), AQLM, EETQ, HQQ, torchao int8 and INC FP8.
  - Merging is not possible for AQLM or INC.
  - torchao DoRA works only with int8.
  - **compressed-tensors is not listed.**
- VeRA supports bnb; AdaLoRA supports bnb and GPTQ; (IA)³ supports bnb.
- VERIFIED [DOC] — [PEFT quantization guide](https://huggingface.co/docs/peft/main/en/developer_guides/quantization)
- On LoftQ, the same guide says:
  - Target all linear layers ("all-linear") with NF4.
  - `replace_lora_weights_loftq` runs only one LoftQ iteration.
  - That one-step path supports only bnb 4-bit with safetensors.
- VERIFIED [DOC] — [PEFT quantization guide](https://huggingface.co/docs/peft/main/en/developer_guides/quantization)

**Evidence on agent/tool tasks**
- MoRAgent [PR, ICML 2025, PMLR 267; Huawei Noah's Ark / BUPT] splits an agent into reasoner, executor (tool-call) and summarizer roles, each with its own LoRA group, plus routers. All arms below are Llama-3.2-1B on BFCL with 0.16B trainable params and the same multi-role data:

  | Arm | BFCL average |
  |---|---|
  | Base | 27.5 |
  | LoRA | 65.5 |
  | **DoRA** | **66.8** |
  | Full SFT (1.24B params) | 76.0 |
  | MoR (role LoRAs) | 77.6 |

  Target modules were q,k,v,o,gate,up,down. Only models of 1–4B params were tested. VERIFIED — [MoRAgent, Table 6](https://www.alphaxiv.org/abs/2512.21708)
- MoRAgent's data-scaling numbers (BFCL, Llama-1B): 1k samples → 54.9, 5k → 60.0, 10k → 63.7, 50k → 73.4, 90k → 77.6. Even 1k samples gave +27 pts over base. VERIFIED — [MoRAgent, Table 7](https://www.alphaxiv.org/abs/2512.21708)
- MoRAgent found "directly applying LoRA for fine-tuning agents often yields performance that is significantly inferior to … full-parameter fine-tuning". It attributes this to learning several capabilities at once under a low-rank constraint. VERIFIED — [MoRAgent §1](https://www.alphaxiv.org/abs/2512.21708)
- Other 2025–26 PEFT-for-agent preprints surfaced but I did not read them: Skill-to-LoRA [PP, Jun 2026, CUHK] and CodeForge-MA (language-conditioned LoRA) [PP, Oct 2026]. The "Beyond Low-Rank…gradient decomposition" variant [PP, Sep 2026] is generic, not agentic. Titles only (SNIPPET) — [Skill-to-LoRA](https://www.alphaxiv.org/abs/2606.16769), [CodeForge-MA](https://www.alphaxiv.org/abs/2610.05481), [2609.37027](https://www.alphaxiv.org/abs/2609.37027)

**Comparison grid** (servability = with vLLM 0.19.1 on the W4A16 base)

| Method | What changes | Agent/code evidence | Plain-LoRA export / vLLM | Fits 31B on 4×L4? | Verdict for us |
|---|---|---|---|---|---|
| LoRA | A,B | MoRAgent BFCL 65.5 (1B) [PR]; LoRA Without Regret (see prior note) | Native ✔ (VERIFIED) | Yes with 4-bit base | **Default** |
| QLoRA (NF4 base) | Base stored in NF4 | Standard; no agent-specific study found | Adapter is plain LoRA ✔ | Yes, needed for memory | **Default storage**. Mismatch risk: see Q2 |
| rsLoRA | Scaling α/√r | No agent study found | ✔ vLLM honours `use_rslora` (VERIFIED) | Same as LoRA | Use if rank ≥64 (INFERRED: keeps LR stable across ranks) |
| LoRA+ | Higher LR on B | None found | ✔ (optimizer only; INFERRED) | Same | Free to try; low priority |
| DoRA | + magnitude vector m | +1.3 over LoRA on BFCL (1B) [PR] | ✘ **vLLM rejects `use_dora`** (VERIFIED) | Slower and more memory | **Don't use.** Folding m into a rank≤128 LoRA needs a lossy SVD of the full ΔW (INFERRED) |
| PiSSA / MiLoRA / OLoRA / CorDA | SVD-based init; base becomes residual | None on agents found | Convertible to a plain rank-2r LoRA on the original base via PEFT's conversion path (from memory of the PEFT docs, NOT re-verified). The conversion assumes a bf16 base, not W4A16 | Needs a bf16 SVD of 31B weights: heavy offline | Skip. Gains are small and the 2r rank costs adapter budget |
| LoRA-GA | Gradient-aligned init | None on agents | Same as PiSSA (base modified) | Needs extra gradient pass | Skip |
| LoftQ | Init to cancel NF4 error | None on agents | Plain LoRA, but it **compensates NF4 error, not W4A16 error** (INFERRED) | One-step path is cheap (VERIFIED: bnb-4bit only) | **Avoid** unless the training quant equals the serving quant |
| QA-LoRA / IR-QLoRA / L4Q / EfficientQAT | Group-wise / quant-aware merge into int4 | None on agents | The goal is a merged int4 model, not a runtime adapter, so irrelevant to our serving path (INFERRED) | n/a | Skip |
| VeRA | Shared frozen random A,B + vectors | None | ✘ not a LoRA layout (INFERRED) | Tiny | Skip |
| AdaLoRA | SVD-form, adaptive per-module rank | None | Convertible to LoRA (B=PΛ, A=Q), but gives a non-uniform `rank_pattern`. vLLM handling is UNVERIFIED | Extra overhead | Skip |
| MoR / MoLE (routed LoRAs) | Routers | MoR 77.6 vs LoRA 65.5 [PR] | ✘ routers not servable | — | Use the **servable analogue**: one adapter per ADK sub-agent (Q4) |

### Inferences
- The rejection list is verified on vLLM main, not the 0.19.1 tag. DoRA has never been supported in vLLM, so it is almost certainly rejected in 0.19.1 too. A one-line check against the wheelhouse copy of `vllm/lora/peft_helper.py` would settle it. (INFERRED)
- Every export should have `bias="none"`, no `modules_to_save`, and no `embed_tokens` / `lm_head` targets. Run the exact PEFT config through the probe (`kaggle/probes/probe_lora_serving.py`) before any real training. (INFERRED)
- MoRAgent's DoRA-vs-LoRA gap (+1.3) is on 1B models and short tool calls. Nothing suggests it is worth giving up servability for a 31B. (INFERRED)

### Gaps
- I found no study comparing LoRA variants on **SWE-style multi-turn trajectories** (SWE-bench / SWE-smith) at ≥14B. All variant comparisons on tool use are ≤8B and single-call or few-call (BFCL, ToolBench).
- I did not confirm whether vLLM 0.19.1 handles a PEFT `rank_pattern` / `alpha_pattern` (per-module ranks).

---

## Q2. Quantisation mismatch: train on an NF4/bf16 base, serve on the W4A16 QAT base

### Takeaway
- Runtime (unmerged) adapters on a quantized base are standard and work. Our own Oct 7 probe confirms the mechanics.
- I found **no controlled study** of an adapter trained on base precision X (NF4 or bf16) and served unmerged on a *different* int4 format Y (W4A16 compressed-tensors). The closest 2026 evidence says otherwise-equivalent quantizers can disagree by ~1 pt per task and that convention mismatches can be catastrophic. That evidence is for merged artifacts.
- The safest engineering choice is to **train against bit-identical serving weights**: load the compressed-tensors int4 codes and scales as-is into a dequant-on-the-fly linear (GPTQ or HQQ representation). The fallback is to **dequantize W4A16 to bf16 → NF4**, which double-quantizes. That path is not the original bf16 → NF4 either.

### Cited Findings
- Scale-QLoRA [PP, Sep 2026, Crusoe] studied NVFP4/MXFP4 bases and LoRA merging:
  - A naive dequantize → add LoRA → requantize merge loses **up to 39 pp** (NVFP4 Llama-8B Banking77: 93.0 → 55.6).
  - Merge-aware QAT-LoRA, which trains with an STE onto the deployment grid, is lossless (≤0.2 pp).
  - Six quantizer outputs from five codebases (HF, modelopt, compressed-tensors, the authors' own) **never agreed bit-exactly** across 224 linears.
  - A convention difference (HF quantize-on-load) cost −1.04 pp (paired McNemar p = 0.002).
  - VERIFIED — [Scale-QLoRA §1, App. B](https://www.alphaxiv.org/abs/2609.04526)
- In Scale-QLoRA's real vLLM 0.23 deployment table, an **unmerged 4-bit base + fp16 LoRA** scored 92.4 with **+24% decode latency**. Other rows scored 93.6–95.0. The rows used different adapters, so the authors say it is "not a controlled comparison". VERIFIED — [Scale-QLoRA Table 7](https://www.alphaxiv.org/abs/2609.04526)
- Scale-QLoRA's bf16 LoRA ceiling vs quantized-base adaptation: indistinguishable on classification (±2.6 pp), and up to ~4 pp behind on Spider text-to-SQL. Generative or structured-output tasks seem more sensitive to the base's precision. VERIFIED — [Scale-QLoRA Table 15](https://www.alphaxiv.org/abs/2609.04526)
- 2026 quantization-aware PEFT work exists: QATFactory [PP, Sep 2026, Together AI/UIUC/UT Austin] describes itself as a "deployment-aligned" QAT and distillation framework; QWHA [PP, Sep 2025]; AQLoRA [PP, Aug 2026]; plus a target-centric QAT survey [PP, Aug 2026]. Titles and abstracts only (SNIPPET) — [QATFactory](https://www.alphaxiv.org/abs/2609.39223), [QWHA](https://www.alphaxiv.org/abs/2509.17428), [AQLoRA](https://www.alphaxiv.org/abs/2608.23816), [QAT survey](https://www.alphaxiv.org/abs/2608.29667)
- Practitioner claim: adapters trained on a bf16 base can be applied to a quantized base at runtime as a delta, but "there is a quantization gap that requires adequate LoRA rank capacity". SNIPPET (vendor page, unverified) — [Friendli "Train on BF16, serve on NVFP4"](https://friendli.ai/models/MonumentalSystems/Holo-3.1-35B-A3B-lora-demo)
- A practitioner thread proposes deterministic LoRA training directly through the quantized GGUF serving weights. This is the same "train on the served codes" idea. SNIPPET — [HF forum](https://discuss.huggingface.co/t/deterministic-lora-training-directly-through-quantized-gguf-serving-weights/179118)
- PEFT has no compressed-tensors training integration. GPTQ (GPT-QModel) and HQQ are supported training backends. VERIFIED [DOC] — [PEFT quantization guide](https://huggingface.co/docs/peft/main/en/developer_guides/quantization)

### Inferences
The serving base is a symmetric int4 group-quantized ("W4A16") version of a QAT checkpoint. QAT means the model was trained to be robust at exactly that grid. Base-precision options, ranked by fidelity to the served base:
1. **Exact codes (best).** Unpack the compressed-tensors `weight_packed` + `weight_scale` (+ zero point if any) into:
   - a GPTQ-format checkpoint, so GPT-QModel + PEFT can train it; or
   - an HQQ layer with zero=8 and scale=s, which is exact for a symmetric uniform grid. Answer.AI's fsdp_qlora also supports HQQ.

   The adapter then sees the same W the scorer serves. The kernels differ (Marlin vs torch dequant), so outputs are close but not bit-identical. (INFERRED; **untested for Gemma-4**)
2. **bnb NF4 of the dequantized W4A16 weights.** Easiest path: FSDP-QLoRA, Unsloth and Axolotl all support it. But NF4's non-uniform 16 levels per 64-block cannot represent the uniform int4 grid exactly. This is a second rounding error, around the size of normal QLoRA error. (INFERRED)
3. **NF4 of the original bf16 (non-QAT) weights.** Avoid: the serving base also differs from it by the QAT update. (INFERRED)
4. **bf16 dequantized W4A16 (exact values, no second quantization).** 31B × 2 B ≈ 62 GB, which is 15.5 GB/GPU under FSDP on top of activations. Probably infeasible beyond ~4k tokens. (INFERRED, see Q3)

Other inferences:
- Measure the mismatch directly and cheaply. Train a small adapter under option 2. Then compute held-out trajectory loss and tool-call-validity on (a) the training-precision base in HF and (b) the scorer vLLM W4A16 base. A gap of more than a few % in token-level NLL means switching to option 1. (INFERRED)
- **LoftQ is actively harmful in our setting.** Its init cancels NF4 rounding error that does not exist on the served base. (INFERRED)
- Merging is irrelevant: the scorer serves unmerged runtime adapters. Scale-QLoRA's catastrophic merge losses therefore don't apply directly. Its "conventions differ by ~1 pp" finding is the relevant lower bound for mismatch noise. (INFERRED)

### Gaps
- No paper found that measures adapter **transfer across int4 formats** (NF4 → GPTQ/AWQ/compressed-tensors) for long generative or agentic outputs.
- Unknown whether Google ships a bf16 "QAT-unquantized" Gemma-4-31B checkpoint, as it did for Gemma 3. Such a checkpoint would make option 2 cleaner.
- Unknown whether GPT-QModel or HQQ support the Gemma-4 architecture in the transformers version on our Kaggle image.

---

## Q3. Memory, throughput and engineering for 31B QLoRA on 4×24 GB at 16k–32k tokens

### Takeaway
- Feasible, but only with: FSDP-sharded 4-bit base, gradient checkpointing, a **fused or chunked cross-entropy**, and per-layer attention backends (Gemma-4's global layers break FlashAttention).
- Realistic sequence-length targets are **≤16k tokens per sequence with plain FSDP**, or **24–32k with CPU-offloaded activation checkpoints or naive pipeline parallelism**.
- Rough throughput is a few hundred tokens/s across 4×L4. So **1k trajectories × 16k tokens ≈ 7–12 GPU-session hours per epoch**. 5k trajectories doesn't fit the weekly quota at full length.

### Cited Findings
**Gemma-4 specific blockers**
- Gemma-4 global-attention layers use `global_head_dim=512`, but FlashAttention-2, 3 and 4 are capped at head_dim ≤256. Training crashes on step 1 with "FlashAttention forward only supports head dimension at most 256".
  - The proposed fix is a per-layer-type `attn_implementation` dict (transformers v5): `{"sliding_attention": "flash_attention_2", "full_attention": "sdpa"}`.
  - Throughput and memory impact are not reported.
  - VERIFIED — [SkyRL issue #1564](https://github.com/NovaSky-AI/SkyRL/issues/1564)
- A search snippet says "26 of 30 layers sliding (head_dim 256), 4 global (head_dim 512)". That probably describes a different Gemma-4 variant; our 31B has 410 LoRA modules with two attention shapes. SNIPPET — [flash-attention issue #2427](https://github.com/Dao-AILab/flash-attention/issues/2427)
- Unsloth's Gemma-4 guide:
  - "31B QLoRA works with 22GB" (single GPU, short context; the example uses `max_seq_length = 8192`).
  - Recommends `r = lora_alpha` (e.g. 8/8 or 32/32).
  - Known bugs fixed in Unsloth: gradient-accumulation loss inflation; `use_cache=False` giving garbage logits; IndexError for 31B/26B-A4B when `num_kv_shared_layers = 0`.
  - The chat template and thinking flag must match between training and inference.
  - **The guide says nothing about multi-GPU.**
  - VERIFIED [DOC] — [Unsloth Gemma 4 fine-tuning guide](https://unsloth.ai/docs/models/gemma-4/train)

**Multi-GPU frameworks**
- Answer.AI FSDP+QLoRA trains a 70B model on 2× 24 GB GPUs. Their example ran at max seq length 512 and used just over 128 GB of CPU RAM, with CPU offload of parameters as an option. SNIPPET — [Answer.AI blog](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html), [fsdp_qlora README](https://github.com/AnswerDotAI/fsdp_qlora), [deep dive](https://www.answer.ai/posts/2024-03-14-fsdp-qlora-deep-dive.html), [Axolotl FSDP+QLoRA docs](https://docs.axolotl.ai/docs/fsdp_qlora.html)
- Axolotl sequence parallelism (ring-flash-attn):
  - Llama-3.1-8B QLoRA max sequence on RTX 4090s: 4,096 tokens on 1 GPU → 32,768 on 8 GPUs (linear scaling). On H100s: 17,408 → 129,024.
  - Constraints: flash attention is **required**, micro_batch_size = 1, and the SP degree must divide GPU count, sequence length and head count.
  - Composes with FSDP, DeepSpeed ZeRO 1–3, Liger, torch.compile and QLoRA.
  - VERIFIED — [Axolotl SP blog (HF)](https://www.huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl)
- TRL also documents context/sequence-parallel training options. SNIPPET — [TRL distributing training](https://huggingface.co/docs/trl/main/en/distributing_training)
- No public report found of fine-tuning Gemma-3-27B or Gemma-4-31B on L4 or other 24 GB cards with measured tokens/s. The only hits were inference sizing pages. SNIPPET — [willitrunai Gemma-3-27B on L4](https://willitrunai.com/es/can-run/gemma-3-27b-on-l4-24gb)

### Inferences
These are my arithmetic. Assumptions to verify on the real config: hidden size h ≈ 5376 and MLP intermediate ≈ 21.5k (Gemma-3-27B-like), 60 layers, vocab 262,144.

**Memory per GPU**
- **Base weights.** NF4 31B ≈ 15.5 GB plus ~1 GB of quantization constants, plus a ~2.8 GB bf16 embedding (262k × 5376 × 2 B).
  - FSDP over 4 GPUs: **~4.8 GB/GPU**.
  - DDP (Unsloth-style, a full copy per GPU): ~19–22 GB/GPU, which leaves no room for long sequences. **DDP is out for >4k tokens.**
- **Trainable params.** 122.6 MB at rank 8 is consistent with bf16 adapters. That gives ≈7.7M params per unit of rank, so r=64 ≈ 490M trainable params.
  - AdamW at ~16 B/param ≈ 7.8 GB, about **2 GB/GPU** FSDP-sharded.
  - r=16 is about 0.5 GB/GPU.
- **Logits are the trap.** One 32k sequence × 262k vocab × fp32 = **34 GB** (16k → 17 GB).
  - Use Liger fused-linear-CE or Cut-Cross-Entropy, or a chunked loss. These are **mandatory**, not an optimisation.
- **Activation checkpoints** (one hidden state per layer): ≈ h × 2 B × 60 layers ≈ 645 KB/token.
  - 16k tokens → ~10.6 GB/GPU; 32k tokens → ~21 GB/GPU under data parallelism.
  - Recompute peak for one MLP: ~16k × 21.5k × 2 B × ~3–4 tensors ≈ 2–3 GB at 16k, roughly double at 32k.
- **Global attention.** With SDPA on the head_dim-512 layers, everything depends on SDPA picking the memory-efficient kernel on sm89 for head_dim 512. If it falls back to the math kernel, a 32k global layer materialises 32 heads × 32k² × 2 B ≈ 64 GB → OOM. **This must be probed first.** Fallback: FlexAttention, or chunked/query-blocked attention for those layers.
- **Sequence-length budget per GPU.**
  - FSDP + checkpointing at 16k: ~4.8 + 2 + 10.6 + ~3 + ~1.5 (CUDA context and FSDP buffers) ≈ **22 GB**. Feasible but tight; use r ≤ 32 to gain margin.
  - At 32k: needs **CPU-offloaded checkpoints** (~21 GB per GPU of host RAM, so ~85 GB host RAM for 4 GPUs).
  - Alternative: **naive pipeline** (`device_map` across 4 GPUs, one sequence). Each GPU holds 15 layers: ~4.8 + 2 + 5.3 + ~6 GB peak ≈ 18 GB at 32k. Robust, but only ~25% GPU utilisation.
- **Sequence parallelism (ring/Ulysses)** looks attractive but Axolotl's SP **requires flash attention**, which fails on the 512-dim global layers. Treat SP as unavailable unless an SDPA- or flex-based SP path works for Gemma-4.

**Throughput**
- LoRA with checkpointing costs ≈6 × N FLOPs per token (forward + recompute + activation-grad backward, no frozen-weight grads) ≈ 1.9e11 FLOPs/token for 31B, plus attention overhead.
- 4× L4 at 121 TFLOPS dense bf16 peak each (NVIDIA spec, not re-verified) at 15–25% MFU (QLoRA dequant + PCIe FSDP) → **~400–650 tokens/s total**.
- FSDP communication is not the bottleneck at long sequences. Roughly 36 GB gathered per GPU per 32k-token sequence over PCIe takes ~2–3 s, against ~160 s of compute.

**GPU-hours (Kaggle quota counts session-hours)**

| Dataset (avg 16k tokens) | Tokens per epoch | Session-hours per epoch |
|---|---|---|
| 1k trajectories | 16M | **~7–11 h** (one 12 h session is borderline) |
| 5k trajectories | 80M | **~35–55 h** (more than a week of quota) |

- Loss-masking observations doesn't cut compute, because forward still runs over all tokens.
- Practical plan: 1–2k trajectories averaging ≤10k tokens (≈3–7 h/epoch). Checkpoint the adapter every ~30 min, since a 12 h session can be killed.

**Recipe (INFERRED; to be validated in Phase 0b)**
- **Stack:** transformers v5 + PEFT + TRL `SFTTrainer` (or Axolotl), with FSDP (FULL_SHARD, `use_orig_params`) and bnb NF4. Use `bnb_4bit_quant_storage=bfloat16` so FSDP can shard the 4-bit weights (Answer.AI's integration; quant_storage detail from memory).
- **Settings:** gradient checkpointing (reentrant=False), Liger or CCE fused loss, attention dict = FA2 for sliding layers and SDPA (mem-efficient) for full layers.
- **Batching:** micro-batch 1, packing off (or padding-free), assistant-only loss mask, max length 16k first.
- **LoRA config:** `target_modules` = all 7 projections, `bias="none"`, r = α = 16–32, `use_rslora` optional, LR ≈1e-4 to 2e-4, 1–2 epochs.
- **Export:** save PEFT safetensors, check every key matches the 410 served module names, and run the probe on the scorer stack.
- **Offline install:** everything installs from a pre-built wheelhouse, because L4 sessions are offline. This mirrors `scorer_env.tgz`, but as a separate *training* env (torch + transformers v5 + peft + trl + bitsandbytes + liger-kernel + flash-attn built for sm89).

### Gaps
- No measured Gemma-4-31B training numbers on L4: tokens/s, peak memory, or whether SDPA picks the mem-efficient backend for head_dim 512 on sm89.
- Kaggle host RAM on the L4×4 machine is unknown. It decides whether CPU-offloaded checkpoints at 32k are possible.
- Unsloth multi-GPU support for Gemma-4 isn't documented on its Gemma-4 page, so DDP-only is the conservative assumption.
- I could not confirm Gemma-4-31B's hidden size, intermediate size, layer count or vocab from a primary source this session. Every memory number above scales with them.

---

## Q4. Modules and rank for tool-format reliability vs reasoning; per-role adapters; merging and composition

### Takeaway
- Target **all 7 linear projections** at **moderate rank (16–32)** for the tool-format and workflow adapter. Formatting is a low-capacity behaviour, and the 3 GiB budget favours small adapters.
- Spend any extra rank on a fixer or trajectory adapter, not on format.
- Per-role adapters (localiser vs fixer) are the servable version of MoRAgent's role-specific LoRA groups, and MoRAgent's role decomposition beat a single LoRA by 12 pts.

### Cited Findings
- MoRAgent targets q,k,v,o,gate,up,down for every role.
  - Role-specific LoRA groups (reasoner, executor/tool-call, summarizer) score 77.6 vs 65.5 for a single LoRA at equal trainable params on BFCL (Llama-3.2-1B).
  - Role groups are combined by rule-based gates plus learned routers, which vLLM cannot serve.
  - The summarizer saturated with fewer LoRA modules (3–4) than reasoner and executor (4–5).
  - VERIFIED — [MoRAgent, §4.2, Tables 5–6](https://www.alphaxiv.org/abs/2512.21708)
- Unsloth recommends `lora_alpha == r` at least, and notes larger rank is more accurate but may overfit. VERIFIED [DOC] — [Unsloth Gemma 4 guide](https://unsloth.ai/docs/models/gemma-4/train)
- Adapter size scales linearly with rank: 122.6 MB at rank 8 (410 modules), so ~1 GB at rank 64. The cap is 3 GiB ≈ 3,220 MB in total. (Project fact, VERIFIED in `docs/HQ.md`, Oct 7)
- On merging: merging into the int4 base re-quantizes and can wipe out the adaptation (up to −39 pp). The scorer only serves runtime adapters, so merging into the base is neither needed nor safe. VERIFIED — [Scale-QLoRA](https://www.alphaxiv.org/abs/2609.04526)

### Inferences
- **Rank budget arithmetic.** The sum of ranks across all adapters must be ≤ ~210 (3,220 MB ÷ 15.3 MB). Example layouts: one r=64 + two r=32 + two r=16 ≈ 2.4 GB; or 2 × r=64 + 1 × r=64. Check `max_lora_rank` too: vLLM preallocates LoRA buffers up to max rank × max_loras, which takes KV-cache memory at gpu_mem 0.80. Whatever rank the scorer sets is a hidden cost.
- **Tool-format reliability.** The garbled `edit_file` and `read_file` argument keys are a token-level output-formatting problem. The loss should be concentrated on assistant tool-call tokens. Options: upweight tool-call JSON spans, or train a short "format" adapter on correct tool calls taken from the model's own successful trajectories. Fixing the format may well not need MLP modules. Attention-only vs all-linear at r=16 is a cheap ablation to run. (INFERRED)
- **Per-role adapters (localiser vs fixer).** These map naturally onto ADK sub-agents with `adapter:` per agent; serving routing is already proven. Costs: (a) twice the training runs; (b) each role gets fewer trajectories; (c) the risk of a role mismatch at handoff. Start with one shared adapter and split only if role-specific failure rates differ. Example: localiser budget overruns vs fixer garbled edits. (INFERRED)
- **Composition.** vLLM applies one LoRA per request, so composing at serving means a linear merge into a single adapter: concatenating A/B gives rank r1+r2 with exact sum semantics. Concatenation is exact and legal if the sum ≤128. TIES/DARE-style merging is lossy and unproven here. (INFERRED)

### Gaps
- I found no study isolating which modules (attention vs MLP) carry tool-call format vs multi-step reasoning in large (≥27B) models.
- No evidence found on per-role LoRA for **SWE** agents (localiser/fixer) specifically. MoRAgent covers tool-use and math with 1–4B models.

---

## Q5. Gaps: what is untested for PEFT on agentic SWE tasks

### Takeaway
The literature leaves our exact regime untouched: LoRA on ≥27B, long (16–32k) multi-turn SWE trajectories, a quantized base that differs from the serving base, and adapters served per sub-agent. Every key number (mismatch cost, max sequence length on L4, rank needed for format) has to come from our own Phase 0b probes.

### Cited Findings
- All PEFT-variant comparisons on agent/tool tasks that I found use ≤8B models and short tool calls (BFCL, StableToolBench). VERIFIED — [MoRAgent](https://www.alphaxiv.org/abs/2512.21708)
- The only cross-quantizer adapter evidence is for merged NVFP4/MXFP4 artifacts on classification, Spider and MBPP, not for runtime adapters or agent trajectories. VERIFIED — [Scale-QLoRA](https://www.alphaxiv.org/abs/2609.04526)
- Gemma-4 training-stack breakage (head_dim 512 vs FlashAttention) is still being worked around in open-source RL trainers. VERIFIED — [SkyRL #1564](https://github.com/NovaSky-AI/SkyRL/issues/1564)

### Inferences
Untested items, with the probe that would close each (all INFERRED):
1. **Mismatch cost:** NF4-of-dequantized-W4A16 vs exact-code GPTQ/HQQ training, measured as held-out NLL and tool-call validity on the scorer vLLM.
2. **SDPA backend for head_dim 512 at 16k/32k on sm89:** a 1-layer micro-benchmark measuring peak memory.
3. **Max sequence length and tokens/s for FSDP-QLoRA Gemma-4-31B on 4×L4:** one 30-min session sweeping 8k/16k/24k at r=16.
4. **Format adapter:** does an attention-only r=16 adapter fix garbled tool args as well as an all-linear one?
5. **Rank needed for trajectory SFT at ~1k trajectories** (r=16 vs 64), given the 3 GiB cap.
6. **Per-role vs shared adapter** on localiser/fixer failure modes.
7. **Whether vLLM 0.19.1 accepts `rank_pattern`, `use_rslora`**, and what `max_lora_rank` the scorer sets.

### Gaps
- **Search log** (every query run this session, in order):
  1. WebSearch: "vLLM LoRA DoRA support serving magnitude" → no DoRA info in docs; led to the source check.
  2. WebSearch: "Gemma 4 31B QLoRA fine-tuning memory Unsloth multi-GPU" → Unsloth Gemma-4 guide.
  3. WebSearch: "LoRA adapter trained on bf16 served on GPTQ AWQ quantized base mismatch" → vendor/forum snippets only.
  4. Firecrawl search: "FSDP QLoRA 70B two 24GB GPUs Answer.AI sequence length" → Answer.AI and Axolotl pages (snippets).
  5. WebFetch: unsloth.ai/docs/models/gemma-4/train.
  6. WebFetch: docs.vllm.ai/en/latest/features/lora.html.
  7. alphaXiv discover (failed: bad args) ×2, then retried:
  8. alphaXiv discover: keywords [LoRA quantization mismatch, adapter transfer quantized base, quantization-aware LoRA, QAT LoRA].
  9. alphaXiv discover: keywords [LoRA agent tuning, PEFT tool calling, LoRA SWE agent, DoRA PiSSA comparison code].
  10. WebFetch: raw vllm/lora/peft_helper.py (main).
  11. WebSearch: "Axolotl sequence parallelism ring attention QLoRA long context 32k multi-GPU".
  12. WebSearch: "Gemma 3 27B QLoRA L4 24GB fine-tune tokens per second sequence length" → no training benchmarks found.
  13. alphaXiv PDF Q&A: MoRAgent (2512.21708); Scale-QLoRA (2609.04526).
  14. WebFetch: Axolotl SP blog on HF.
  15. WebSearch: "Gemma 4 31B head_dim 512 flash attention not supported training".
  16. WebSearch: "PEFT LoRA training on compressed-tensors W4A16 checkpoint HuggingFace" → nothing specific.
  17. WebFetch: PEFT quantization developer guide; SkyRL issue #1564.
- **Queries not run** (suggested follow-ups):
  - "vllm 0.19 peft_helper use_dora" (pin the exact tag);
  - "GPTQModel Gemma4 support";
  - "HQQ FSDP LoRA Gemma";
  - "Kaggle L4x4 RAM";
  - "SDPA efficient attention head_dim 512 sm89";
  - "Gemma-4-31B config.json hidden_size";
  - "Liger kernel Gemma4";
  - "QATFactory LoRA" (full read);
  - "Unsloth multi-GPU Gemma 4 FSDP".
