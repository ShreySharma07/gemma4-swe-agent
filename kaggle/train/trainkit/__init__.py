"""
trainkit: LoRA SFT of gemma-4-31b-it on Kaggle 4x L4 with the recipe measured by the training probe (v4, Oct 10):
NF4 from the bf16 QAT weights, LoRA r16 on the 410 language-model linears, FSDP FULL_SHARD over 4 ranks,
memory-efficient SDPA (head_dim 512), loss inside the FSDP root, activation offload for long windows, per-layer export.

Modules, in the order the notebook uses them:
  config    TrainSettings (one JSON shared by every process) and fixed paths
  data      TrainData: the tokenized windows (.npz from posttrain/tokenize_windows.py), val split, per-rank schedule
  model     load NF4 + LoRA + FSDP; MaskedLoss (trained tokens only, Gemma softcap, chunked)
  export    per-layer gather of the LoRA weights -> PEFT adapter that vLLM serves
  worker    the torchrun entry point: train loop, validation loss, periodic + final export
  launcher  runs torchrun inside the scorer venv, keeps per-rank logs
  main      orchestration after the env step (runs inside the venv)
smoke=True runs the same code on CPU (gloo, 2 ranks) with a tiny random Gemma-4 built from the real config.
Source of truth: kaggle/train/trainkit/; make_train_notebook.py turns it into a notebook.
"""
