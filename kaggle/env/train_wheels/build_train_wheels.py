"""
Training add-on for the offline scorer env: peft + bitsandbytes wheels for Python 3.12, checked against scorer_env.tgz.

Why: L4 sessions are offline and scorer_env.tgz (kernel gemma4-scorer-env) has torch 2.10 + transformers 5.13.1 but no
peft / bitsandbytes. peft is pinned to the version smoke-tested locally against transformers 5.13.1.

Kaggle: CPU, Internet ON. Input: output of kernel gemma4-scorer-env (kernel_sources).
Output: /kaggle/working/train_wheels/*.whl + train_wheels_info.json. Consumer: kaggle/probes/train_feasibility.
"""
import glob
import json
import pathlib
import subprocess
import sys
import time

W = pathlib.Path("/kaggle/working")
OUT = W / "train_wheels"
OUT.mkdir(exist_ok=True)
VENV = pathlib.Path("/tmp/scorer_host")
PKGS = ["peft==0.21.2", "bitsandbytes"]
T0 = time.time()


def sh(cmd, timeout=None, check=False):
    print("$", cmd[:220], flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    print(((r.stdout or "")[-1500:] + "\n" + (r.stderr or "")[-2000:]).strip(), flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED rc={r.returncode}: {cmd[:120]}")
    return r


# 1. wheels for the venv's interpreter (cp312, manylinux x86_64), without deps: the venv already has them
sh(f"{sys.executable} -m pip download --no-deps --only-binary=:all: --python-version 3.12 --implementation cp "
   f"--platform manylinux_2_24_x86_64 --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 "
   f"-d {OUT} " + " ".join(f"'{p}'" for p in PKGS), timeout=900, check=True)
whls = sorted(glob.glob(f"{OUT}/*.whl"))

# 2. prove they install offline into the restored scorer env and import next to torch/transformers
env_tgz = sorted(glob.glob("/kaggle/input/**/scorer_env.tgz", recursive=True))
assert env_tgz, "attach the output of kernel gemma4-scorer-env (kernel_sources)"
sh(f"tar -xzf {env_tgz[0]} -C /", timeout=1800, check=True)
r = sh(f"{sys.executable} -m pip --python {VENV}/bin/python install --no-deps --no-index " + " ".join(whls), timeout=600)
pip_ok = r.returncode == 0
r = sh(f"{VENV}/bin/python -c 'import torch, transformers, peft, accelerate, bitsandbytes; "
       "from transformers import BitsAndBytesConfig, AutoModelForImageTextToText; "
       "print(torch.__version__, transformers.__version__, peft.__version__, accelerate.__version__, bitsandbytes.__version__)'")
info = {"wheels": [pathlib.Path(w).name for w in whls], "pip_into_venv_ok": pip_ok, "import_ok": r.returncode == 0,
        "versions": (r.stdout or r.stderr).strip()[-400:], "build_min": round((time.time() - T0) / 60, 1)}
(W / "train_wheels_info.json").write_text(json.dumps(info, indent=1))
print("INFO", json.dumps(info), flush=True)
if not info["import_ok"]:
    raise SystemExit("peft/bitsandbytes do not import in the scorer env")
