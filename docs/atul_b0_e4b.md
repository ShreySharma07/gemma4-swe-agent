# Atul: push B0 (E4B LoRA serving probe + 24-task E4B diagnosis), Nov 20

What it is: row B0 in `docs/experiments.md` (pre-registered). One L4×4 kernel, two steps:
1. `evalkit/lora_serve.py`: can vLLM serve a PEFT LoRA on the E4B W4A16 checkpoint through the scorer's server?
   Also measures E4B decode speed, base vs LoRA.
2. `evalkit.main`: `v26_e4b` (E4B, v14 scaffold) on 24 new-60 tasks at concurrency 1.

The CPU dry run already passed on Shrey's account (`gemma4-lora-serve-dry` v1: 258 target modules, both adapters discovered).

Cost: about 1.5 h wall ≈ 3 quota-h. Hard stop at 3.25 h wall ≈ 6.5 quota-h.

## Steps
```bash
git pull
```
```bash
kaggle kernels push -p kaggle/eval_l4/lora_serve_kernel
```
- The kernel id is `atulbaliyan/gemma4-lora-serve`.
- Inputs come from Shrey's account (already shared with you for G2): `gemma4-scorer-env`, `gemma4-devset-build`, `gemma4-test-overlays`, dataset `test-deps-wheels`.
- The model is public: `google/gemma-4/Other/gemma-4-e4b-it-qat-w4a16-ct/2`.
- If Kaggle gives T4 instead of L4, the env step aborts within minutes. Then start it from the web editor with Accelerator → GPU L4 ×4.

## When it finishes
```bash
kaggle kernels output atulbaliyan/gemma4-lora-serve -p data/runs/b0
```
- Send Shrey `lora_serve_report.json` and `summary.json`, or just tell him it finished.
- Please don't paste raw traces or logs: the tgz can be large. Claude analyses them from `data/runs/b0/`.
