# Full T1 LoRA: corrected CPU preflight, 6 November 2026

**PASS.** Both v14 and full T1 compiled with the official scorer stack on the corrected private, GPU-disabled CPU notebook, version **1** of the distinct v2 notebook. The platform log has `DRY_PASS` and `paired_new60 --dry exit code 0`; the standalone JSON agrees with the logged report. This establishes CPU compilation and configuration validation. No task evaluation, inference server, GPU launch, training, conversion or competition submission occurred.

## Root cause in the first CPU preflight

The scorer environment initially contained swegemma 0.2.7; its wheelhouse upgrade installed **swegemma 0.2.11**, **adk-submission 0.2.13**, Google ADK 1.36.1 and LiteLLM 1.104.0. Both the official wheel and the installed v2 runtime report `task` as the only child of `swegemma.models`. The first preflight's import referred to an absent legacy module. Its child process exited 1 even though Kaggle marked the notebook COMPLETE.

The failing import was `from swegemma.models.registry import setup_gemma_model_registry`. The scorer raised `ModuleNotFoundError: No module named 'swegemma.models.registry'`.

This was a dry-run integration error before official compilation. It supplies no evidence against reusing raw T1 on W4A16.

## Repair and local verification

- Fixed canonical `data/runs/atul_full_lora_new60_20261104/paired_new60.py` first. Its old bytes are backed up as `cpu_fix_v2/paired_new60_before_fix.py`.
- Added `cpu_fix_v2/compiler_preflight.py`. It uses **`adk_submission.VllmServer.create_model_registry()`** as Shrey's `evalkit/server.py` does, without starting the server. It supplies `adk_submission.compile_submission()` the scorer's limits, real `SwegemmaContext.create_tools()` definitions, adapter manifests, and the evaluator's `resolve_swegemma_adapter` resolver. It imports the actual `swegemma.evaluate.Evaluator`; it does not substitute a fake scorer or compiler. An audit hook rejects network connections and subprocess launches during compilation.
- Regenerated **both** `cpu_fix_v2/dry_notebook_v2/` and `cpu_fix_v2/submitted_notebook_v2/` from the corrected canonical source and helper. Both notebooks' code cells compiled and their embedded code matched the authored files exactly. The old CPU and evaluation notebook artifacts were retained byte for byte.
- Local official-package compilation passed for both arms: base resolved to the W4A16 model with an empty adapter manifest, and full resolved to `openai/rs2_lora`. Local testing used the same four main package versions; the recovered Kaggle run verifies the actual restored scorer environment. The compiler, registry, evaluator and resolver source hashes match between the two environments.
- Verified all **60 original task IDs in order**, v14 settings, and the full arm's only source difference: the root adapter reference. The recovered raw adapter has **820 tensors / 410 modules**, rank 16, alpha 32, and SHA-256 `2b22c3f393f178f0c8885df99229a501da826ec1150bda83d8bfd7f4d3d56510`. The adapter config and weights match the originals. No synthetic global V copy was added.
- Existing launch/recovery scripts were syntax checked. The old GPU launch gate remains locked; the new CPU receipt is version-specific and does not populate its launch prerequisite.

## Recovery and preservation

Recovered and hashed **65 files** (437,478,168 bytes), plus the platform log, from the exact notebook version into a private recovery directory. Rechecked source, version, privacy, GPU-disabled settings and terminal status before/after recovery. Verified embedded source, base/full source ZIPs, the final packed ZIP against the recovered full submission tree, and the standalone compiler result against the platform log. No recovered Python, bytecode or archive member was executed. The notebook's trailing generic “no result file” display checks for a GPU summary, not the CPU dry report; the CPU report exists and independently passed.

The first full download broke during the large adapter transfer; its partial bytes and log remain preserved. Validated byte-range recovery completed the two large outputs without re-running the CPU notebook. The single CPU push returned a successful version-1 receipt with a `/code/` reference prefix that caused a local post-push assertion; independent source/version checks confirmed acceptance and no duplicate push was made.

**Available GPU quota: 30.0 hours**, with 0.0 used and 0.0 reserved at the final recovery check. Rehashed 2,401 pre-existing files: only the authorized canonical Python repair changed; its original is backed up. Original weights, v1 failure evidence, research, v14 and untracked R2 data are intact. No Git stage, commit or push and no monitor recreation occurred. Stop here for review before any separate GPU evaluation.
