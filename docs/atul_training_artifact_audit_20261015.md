# Atul training artifact audit — October 15 experiments

Audit performed 2026-10-15T18:34:35.958309+00:00 (October 16 local time). Requested report filename retains the experiment date.

**Final status: COMPLETE locally after recovery.** All required A/B/CPU/C artifacts are preserved. No required file remains missing or corrupt. Historical A v1 cannot be independently re-fetched through the available API; this is a remote-verification limitation, not a missing local adapter. No training, evaluation, notebook push, or leaderboard action was performed.

## Experiment inventory

| Experiment / notebook version | Local directory (`data/runs/`) | Expected and verified artifacts | Recovery / missing | Result |
|---|---|---|---|---|
| [Step A v1](https://www.kaggle.com/code/atulbaliyan/gemma4-train-lora) | `atul_step_a_20261015_v1/` | Reports/config, metrics, execution logs, 8 rank streams, final and serving adapters; original and corrected validation evidence; 51 local files | None / none | COMPLETE |
| [Step B v1](https://www.kaggle.com/code/atulbaliyan/check-l4) | `atul_step_b_20261015_v1/` | Summary/logs, 8 results and traces, 7 test logs, 3 patches, original TGZ and two ZIPs, serving evidence and 5 health checks; 43 local files | None / none | COMPLETE |
| [CPU smoke v1](https://www.kaggle.com/code/atulbaliyan/gemma4-train-smoke) | `atul_instrumentation_cpu_20261015/` | Report/metrics/logs, 4 rank streams; recovered configuration/input, tiny base and adapter files, source modules; 47 local files | 33 files / none | COMPLETE |
| [Step C v2](https://www.kaggle.com/code/atulbaliyan/gemma4-train-lora) | `atul_step_c_corrected_20261015_v2/` | Reports/config, metrics, execution logs, 8 rank streams, step40/80/120, final and serving adapters, validation reports; 92 local files | None / none | COMPLETE |

Version identity comes from saved submission/status records, distinct runtime configurations and export hashes. Current status checks report COMPLETE for all three notebook IDs. A has 5 training steps/40-window cap; C has 122 steps, smoke=false, max_windows=null, budget_h=10.5 and val/save every40. B serves the A hash. CPU uses the tiny 41-module model and two ranks.

## Remote comparison and limitations

Compared every nested output path from the current session output endpoint: B 37 files, CPU 41, C 46. Remote download response byte lengths match all corresponding local files after recovery. Notebook execution-log text also matches the returned session log for these runs. CPU staging downloads additionally match all pre-existing remote-output files byte-for-byte.

The installed CLI parses but does not forward `/version` for `kernels files` or `kernels output`. A request for `/1` therefore returned the current C listing; it was not used to validate or overwrite A. Explicit SDK `version_label` attempts returned HTTP404 for all four version requests. A retains its original listing, status history, notebook logs, original validation report and two hash-verified adapters. Older-version re-download remains unavailable through this interface.

The CLI file-list `size` values were implausibly small and its names omitted nested directories. They were not treated as artifact lengths. Instead the session-output endpoint supplied full paths, and streamed GET response headers supplied actual byte lengths. Remote headers did not supply a checksum; no remote SHA/MD5 comparison is claimed. Local SHA-256 hashes, recorded adapter hashes and archive checks provide integrity evidence. Signed download URLs were neither printed nor persisted.

## Step C adapter checks

All five files are exactly **244,982,352 bytes** each. Each passed 410 modules, 820 BF16 tensors, rank16 PEFT configuration, paired A/B keys, shape/byte-count consistency, contiguous non-overlapping offsets and exact complete file length. Every key/dtype/shape matches the validated A layout. All hashes match the original C validation report and each export is present in the training export manifest.

| Export | SHA-256 |
|---|---|
| `adapters/conv_lora` | `a116b6be160fffcc98cdd2a4ab6cfaffafcfc54ac6709a8a5d95d95ebab544c6` |
| `train_out/adapter_final` | `a116b6be160fffcc98cdd2a4ab6cfaffafcfc54ac6709a8a5d95d95ebab544c6` |
| `train_out/adapter_step120` | `b254701b4d6ef8eb787849299320b79c8d150b3f537aad14a5dbf415f930b59f` |
| `train_out/adapter_step40` | `6133f8d729eb9d7bf834d48b396eaa1c5c9108bab2a5f19a0d89af4a82bea726` |
| `train_out/adapter_step80` | `3aa824649b2be9328e0642def063b6524b580849d2e0e4106c9a00b849fe05a5` |

## Dataset, archives and logs

Approved `data/runs/train_public_200.npz`: 3,385,750 bytes; ZIP CRC check passed; SHA-256 `b3f51bc449c31de2a3b163f98c5fdec2725b4427bde15625af7b4c75447d2488` exactly matches approval.

Step B TGZ was read fully, including gzip CRC/end-of-stream; every regular member was read to its declared length. Both submission ZIP CRC checks passed. All eight trace JSONs parse and archived results match the standalone results exactly. The bundled serving adapter matches A (`09877873743ec96d0ef7d362e042ac512deb8331f1022bf5f5c02e4d1313d1a0`). All five health checks were re-read as true and adapter/vLLM startup markers are present.

The archive contains only three produced patch files and seven test logs: the eighth task hit the 100-turn budget. These are execution outcomes, not failed downloads. Its archived run.log is a 9,097-byte exact prefix of the 9,986-byte final log; packaging/completion lines were appended afterward. Both originals are retained.

The sole zero-byte file is the optional CPU `live_notebook.log` capture. Complete CPU notebook logs, run.log, metrics and both ranks’ stdout/stderr are present. It is retained as historical evidence and is not a missing execution log. A produced no intermediate checkpoint; C produced all three required checkpoints. All JSON/JSONL artifacts parse.

## Recovered files

Recovered from the completed CPU notebook into separate staging, verified response sizes, JSON/NPZ CRC and safetensors structure, then copied only absent paths using exclusive file creation. Existing reports were byte-compared and untouched. The tiny adapter contains 41 modules/82 tensors as expected, rather than the full-training 410/820. The tiny base model was inspected only as a file/header; no model or tensors were loaded.

- `data/runs/atul_instrumentation_cpu_20261015/adapters/instrumentation_smoke/adapter_config.json` — 343 bytes
- `data/runs/atul_instrumentation_cpu_20261015/adapters/instrumentation_smoke/adapter_model.safetensors` — 209,248 bytes
- `data/runs/atul_instrumentation_cpu_20261015/evalkit/__init__.py` — 1,128 bytes
- `data/runs/atul_instrumentation_cpu_20261015/evalkit/__pycache__/__init__.cpython-313.pyc` — 1,273 bytes
- `data/runs/atul_instrumentation_cpu_20261015/evalkit/__pycache__/config.cpython-313.pyc` — 3,114 bytes
- `data/runs/atul_instrumentation_cpu_20261015/evalkit/__pycache__/envsetup.cpython-313.pyc` — 14,755 bytes
- `data/runs/atul_instrumentation_cpu_20261015/evalkit/__pycache__/util.cpython-313.pyc` — 2,576 bytes
- `data/runs/atul_instrumentation_cpu_20261015/evalkit/config.py` — 2,262 bytes
- `data/runs/atul_instrumentation_cpu_20261015/evalkit/envsetup.py` — 8,301 bytes
- `data/runs/atul_instrumentation_cpu_20261015/evalkit/util.py` — 1,016 bytes
- `data/runs/atul_instrumentation_cpu_20261015/tiny_model/config.json` — 3,159 bytes
- `data/runs/atul_instrumentation_cpu_20261015/tiny_model/generation_config.json` — 235 bytes
- `data/runs/atul_instrumentation_cpu_20261015/tiny_model/model.safetensors` — 1,921,084 bytes
- `data/runs/atul_instrumentation_cpu_20261015/train_cfg.json` — 197 bytes
- `data/runs/atul_instrumentation_cpu_20261015/train_data.npz` — 385,141 bytes
- `data/runs/atul_instrumentation_cpu_20261015/train_out/adapter_final/adapter_config.json` — 343 bytes
- `data/runs/atul_instrumentation_cpu_20261015/train_out/adapter_final/adapter_model.safetensors` — 209,248 bytes
- `data/runs/atul_instrumentation_cpu_20261015/trainkit/__init__.py` — 1,190 bytes
- `data/runs/atul_instrumentation_cpu_20261015/trainkit/__pycache__/__init__.cpython-312.pyc` — 1,336 bytes
- `data/runs/atul_instrumentation_cpu_20261015/trainkit/__pycache__/config.cpython-312.pyc` — 2,585 bytes
- `data/runs/atul_instrumentation_cpu_20261015/trainkit/__pycache__/data.cpython-312.pyc` — 4,422 bytes
- `data/runs/atul_instrumentation_cpu_20261015/trainkit/__pycache__/export.cpython-312.pyc` — 3,985 bytes
- `data/runs/atul_instrumentation_cpu_20261015/trainkit/__pycache__/launcher.cpython-312.pyc` — 2,482 bytes
- `data/runs/atul_instrumentation_cpu_20261015/trainkit/__pycache__/main.cpython-312.pyc` — 5,791 bytes
- `data/runs/atul_instrumentation_cpu_20261015/trainkit/__pycache__/model.cpython-312.pyc` — 8,879 bytes
- `data/runs/atul_instrumentation_cpu_20261015/trainkit/__pycache__/worker.cpython-312.pyc` — 15,353 bytes
- `data/runs/atul_instrumentation_cpu_20261015/trainkit/config.py` — 1,883 bytes
- `data/runs/atul_instrumentation_cpu_20261015/trainkit/data.py` — 2,140 bytes
- `data/runs/atul_instrumentation_cpu_20261015/trainkit/export.py` — 2,141 bytes
- `data/runs/atul_instrumentation_cpu_20261015/trainkit/launcher.py` — 1,371 bytes
- `data/runs/atul_instrumentation_cpu_20261015/trainkit/main.py` — 3,264 bytes
- `data/runs/atul_instrumentation_cpu_20261015/trainkit/model.py` — 4,902 bytes
- `data/runs/atul_instrumentation_cpu_20261015/trainkit/worker.py` — 7,678 bytes

## Storage and remaining action

Four run directories plus the approved dataset contain **1,930,330,577 bytes (1.798 GiB)**. This is logical file size, including preserved duplicate logs/copies; excludes this audit’s staging/evidence directory and unrelated experiments. Free local disk: **259,326,668,800 bytes (241.52 GiB)** at audit time.

Recovery staging is retained under `data/runs/atul_artifact_audit_20261015/cpu_recovery_staging/`. Full per-file SHA-256 inventory, remote path/size comparisons and recovery evidence are in [audit summary](../data/runs/atul_artifact_audit_20261015/summary.json). No required artifact remains unavailable locally; only historical remote A verification and server-provided checksums are unavailable. This verifies preservation on this local disk, not a second-device backup.

C training remains PASS: validation 0.7718 → 0.3133 → 0.2909 → 0.2883 → 0.2882, 122/122 planned steps, global peak18.88GiB, steady throughput median295.6tok/s. Scheduler coverage972/974 train and40/43 validation windows is retained as a limitation. The C adapter has not yet been served or independently evaluated. Last recorded quota remaining is11.87h; this artifact-only audit consumed no GPU quota. Further experiments require explicit authorization.
