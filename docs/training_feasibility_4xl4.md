# Can Gemma-4-31B (QAT W4A16) be LoRA-trained on 4× L4? Desk check, 8 Oct 2026

**Verdict: yes on paper, up to ~16k-token sequences without offloading. Nothing below has been run on an L4 yet.** The one thing that decides it is an empirical probe (~1 GPU-h, design at the end).

## Model facts (VERIFIED: HF `google/gemma-4-31B-it` config.json, our Oct 7 LoRA probe)
- 60 layers: 50 sliding-window (window 1024, head_dim 256, 16 KV heads) + 10 global (every 6th layer; head_dim **512**, 4 KV heads, `attention_k_eq_v` so global layers have no v_proj).
- hidden 5376, MLP 21504, vocab 262,144, tied embeddings, final logit softcap 30.
- ≈29.3 B linear params + 1.41 B embedding. 410 LoRA target modules (matches the probe).
- **Served base = compressed-tensors W4A16, symmetric int4, group size 32.** That is the same grid as Q4_0.
- Google also publishes **`google/gemma-4-31B-it-qat-q4_0-unquantized`** (Apache-2.0, 62.5 GB bf16). The README calls these "half-precision weights extracted from the QAT pipeline".
  - Rounding them to sym-int4/g32 should give the served weights. **Not verified bit-exact.**
  - Dequantizing the W4A16 file itself gives the exact served values with no download.

## Blockers checked
| Issue | Status | Evidence |
|---|---|---|
| FlashAttention caps head_dim at 256, so the global layers crash | **Avoidable.** Use PyTorch SDPA everywhere. | VERIFIED: PyTorch 2.10 `sdp_utils.cpp`. Flash limit is 256, and on sm86/89 training with head_dim >192 is blocked. The **memory-efficient backend has no head_dim upper bound**, only an alignment check, so head_dim 512 is eligible forward and backward. Runtime speed and memory at 512 are untested. |
| If SDPA falls back to the math kernel at 512 → 32×16k²×2 B ≈ 16 GB per layer → OOM | Must be confirmed in the probe | Force `torch.backends.cuda.enable_math_sdp(False)` in the probe so a fallback errors instead of silently OOMing |
| Mem-efficient backend has no native GQA | Fine | transformers repeats KV for SDPA; the global layers' 4 KV heads expand to 32 (~0.5 GB at 16k) |
| Sliding layers via SDPA use a dense mask (no window sparsity) | Costs speed, not feasibility | Attention ≈ +45% FLOPs over linear layers at 16k (INFERRED). FA2 for sliding layers is optional later; it isn't in our env |
| PEFT can't train on compressed-tensors | Work around | PEFT quant backends: bnb, GPTQ, HQQ, AQLM, EETQ, torchao (VERIFIED, notes in `research_notes/PEFT and RL for SWE agents/peft_methods_l4.md`). Path: dequantize W4A16 per tensor → bnb NF4 (`bnb_4bit_quant_storage=bfloat16` so FSDP can shard it). NF4 adds a second rounding error, so **measure the gap**. Exact alternative: HQQ 4-bit/g32/zero=8. |
| 262k-vocab logits: 16k × 262k × 4 B ≈ 17 GB | Must chunk | Chunked or fused CE with the 30.0 softcap (Liger fused-linear-CE, or a 2k-token manual chunk loop) |
| vLLM serving of the trained adapter | Done | Oct 7 probe: PEFT adapters with these 410 modules load and route on the W4A16 base. Export `bias="none"`, no `modules_to_save`, no embed/lm_head targets, no DoRA |
| Offline L4 session | Needs a training env tarball | Same pattern as `scorer_env.tgz`: build `train_env.tgz` (torch 2.10, transformers ≥5.5, peft, trl/accelerate, bitsandbytes, liger-kernel) on a CPU kernel with Internet |

## Memory per GPU, FSDP FULL_SHARD over 4 GPUs (INFERRED arithmetic)
| Item | 16k tokens | 24k | 32k |
|---|---|---|---|
| NF4 base shard (+scales, bf16 embedding) | 4.6 GB | 4.6 | 4.6 |
| LoRA r16 + AdamW (sharded) | 0.4 | 0.4 | 0.4 |
| Grad-checkpoint layer inputs: 60 × T × 5376 × 2 B | 10.6 | 15.9 | 21.1 |
| Recompute working set (one layer, mostly the MLP) | ~2.5 | ~3.7 | ~5 |
| FSDP all-gather buffers, chunked CE, CUDA context | ~2 | ~2 | ~2 |
| **Total** | **≈20 GB ✅** | ≈27 GB ❌ | ≈33 GB ❌ |

- 24–32k needs the checkpoints **offloaded to host RAM** (~16–21 GB per GPU, so 64–85 GB host RAM). Kaggle L4×4 host RAM is **unknown**; the probe records it.
- Other options: shorter trajectories with prefix-fit trimming (SERA kept the first ≥88–95% of steps and beat random trimming), or naive layer-split model parallel. Layer-split fits 32k easily but runs ~4× slower.

## Throughput and budget (INFERRED)
- Compute is ≈6 × 29.3 B ≈ 176 GFLOP per token (frozen base, with recompute) plus attention.
- 4 × 121 TFLOPS dense bf16 at 15–25% MFU gives **~400–650 tokens/s**.
- 1k trajectories × 16k tokens = 16 M tokens ≈ **7–11 h per epoch**, which is one 12 h session. 5k trajectories doesn't fit a week's quota.

## Probe: built 9 Oct (`kaggle/probes/train_feasibility/probe_train.py`)
- **Training base:** Kaggle hosts the bf16 QAT weights as `google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2`. They are quantised to NF4 on load, so there is no conversion step.
- **Add-on wheels:** peft 0.21.2 + bitsandbytes for the offline scorer env, from kernel `gemma4-train-wheels` (`kaggle/env/train_wheels/`, CPU). It checks they import next to torch 2.10 / transformers 5.13.1.
- **CPU smoke test:** passed locally (`PROBE_SMOKE=1`, tiny random Gemma-4 built from the real config). The LoRA module set and the exported key format match what vLLM served on Oct 7.
- **How to run:**
  1. Regenerate the notebook: `python3 kaggle/probes/train_feasibility/make_notebook.py`.
  2. Push: `.venv/bin/kaggle kernels push -p kaggle/probes/check_l4`. This **starts a GPU run**.
  3. The budget is 75 min (`PROBE_BUDGET_MIN`); later stages are skipped if time runs out.
- **Output:** `train_probe.json`.
- **v2 run (Oct 9, p0b):** partial GO; see experiments.md.
- **v3 (Oct 9):** each GPU stage runs in its own process, per-rank logs are kept, the loss runs inside the FSDP forward (custom chunked CE), 24k/32k use activation offload to host RAM, the bf16 adapter is exported one layer at a time, and the vLLM stage has a `__main__` guard. Smoke-tested on a Kaggle CPU kernel (`gemma4-train-probe-smoke`, built by `make_smoke_kernel.py`; FSDP on gloo, 2 ranks): train, offload, NF4 reference and export all pass, with 41/41 modules and PEFT key format. **Do not smoke-test on the Mac:** it crashed Python there.

Original plan:

## Probe to settle it (~1 GPU-h on L4×4, plus a CPU kernel to build `train_env.tgz`)
1. Load the text model only. Dequantize W4A16 → NF4 per tensor, FSDP over 4 GPUs, LoRA r16 all-linear, SDPA with the math backend disabled, gradient checkpointing, chunked CE.
2. Sweep sequence lengths 4k / 8k / 16k / 24k: record peak memory per GPU, tokens/s, and which SDPA backend ran on the head_dim-512 layers. Record host RAM.
3. **Mismatch check:** NLL and next-token agreement of the NF4 training model vs the W4A16 model in vLLM, on ~50 held-out trajectory turns, with the zero adapter.
4. Train 20 steps on 10 trajectories → export PEFT → load in the scorer vLLM (reuse `kaggle/probes/probe_lora_serving.py`). The adapter must route and change outputs.

Kill/redirect rules:
- head_dim-512 backward fails or OOMs at 8k → fall back to FlexAttention or a chunked-query attention for the 10 global layers.
- NF4-vs-W4A16 token agreement below ~97% → switch to exact HQQ g32.
- tokens/s below 250 → cap training to ~500 trajectories or 8k tokens.
