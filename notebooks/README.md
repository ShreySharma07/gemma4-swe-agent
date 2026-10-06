# Notebooks (planned, traditional flow)

| # | Notebook | Kaggle machine | Content |
|---|---|---|---|
| 01 | `01_data_eda` | CPU | tasks, repos, patches, tests, dates, graph coverage, difficulty strata, local env stability |
| 02 | `02_baseline_failures` | CPU (reads results) | r1/r2/r3 funnels, failure taxonomy, renamed dev set vs public tasks |
| 03 | `03_post_training_data` | CPU | tool-format pairs, rejection-sampled trajectories, public teacher trajectories, synthetic localisation data |
| 04 | `04_lora_training` | GPU L4×4 | QLoRA SFT runs, export PEFT safetensors, serving check |
| 05 | `05_final_pipeline` | GPU L4×4 | final agent + adapters, dev-set eval, submission zip |

Heavy steps write their outputs to Kaggle datasets so later notebooks only read results.
