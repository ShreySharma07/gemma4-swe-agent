"""
Write kaggle/probes/check_l4/check-l4.ipynb from probe_train.py and attach its inputs in kernel-metadata.json.

The probe re-execs itself inside the offline scorer env, which needs a real file, so the notebook writes it out and runs it.
`check-l4` is the competition notebook: only it gets L4 x4. Push with: .venv/bin/kaggle kernels push -p kaggle/probes/check_l4
"""
import base64
import json
import pathlib

HERE = pathlib.Path(__file__).parent
NB_DIR = HERE.parent / "check_l4"
BUDGET_MIN = 75  # remaining GPU quota this week is ~90 min including session start-up

src = base64.b64encode((HERE / "probe_train.py").read_bytes()).decode()
cell = f'''# Generated from kaggle/probes/train_feasibility/probe_train.py by make_notebook.py (do not edit here).
import base64, os, pathlib, subprocess, sys
SRC = {src!r}
p = pathlib.Path("/kaggle/working/probe_train.py")
p.write_bytes(base64.b64decode(SRC))
env = dict(os.environ, PROBE_BUDGET_MIN="{BUDGET_MIN}")
proc = subprocess.Popen([sys.executable, str(p)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end="", flush=True)
print("exit code", proc.wait())
'''
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": cell.splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
(NB_DIR / "check-l4.ipynb").write_text(json.dumps(nb, indent=1))

meta_path = NB_DIR / "kernel-metadata.json"
meta = json.loads(meta_path.read_text())
meta["model_sources"] = ["google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2",
                         "google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2"]
meta["kernel_sources"] = ["shreysharma07/gemma4-scorer-env", "shreysharma07/gemma4-train-wheels"]
meta["enable_internet"] = False
meta["machine_shape"] = "NvidiaL4"
meta_path.write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", NB_DIR / "check-l4.ipynb", f"({len(src) / 1e3:.0f} kB embedded)")
print(json.dumps(meta, indent=2))
