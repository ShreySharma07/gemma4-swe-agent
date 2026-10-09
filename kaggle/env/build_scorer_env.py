"""
Build the scorer's vLLM-host Python environment once, on a CPU kernel WITH Internet, for offline L4 kernels.

Why: L4 sessions in this competition have no Internet, the Kaggle image is now Python 3.13, and the wheelhouse is
cp312-only and not a consistent dependency set (google-adk vs vllm on starlette, xgrammar vs vllm on transformers),
so the scorer must install it unresolved. This script makes a uv Python 3.12 venv at /tmp/scorer_host with:
torch 2.10.0+cu128 -> every wheelhouse wheel --no-deps (exact scorer versions) -> only the missing requirements
(incl. requested extras such as fastapi[standard] -> uvicorn[standard] -> uvloop), never changing an installed version.

Kaggle: CPU, Internet ON. Input: dataset metric/gemma-4-developer-agent-wheelhouse.
Output: /kaggle/working/scorer_env.tgz (venv + uv's standalone Python, restored with `tar -xzf scorer_env.tgz -C /`).
Consumers attach this kernel via kernel_sources (probes/lora_serving, eval_l4).
"""
import glob
import json
import os
import pathlib
import subprocess
import sys
import time

W = pathlib.Path("/kaggle/working")
LOG = open(W / "build.log", "a", buffering=1)
VENV = pathlib.Path("/tmp/scorer_host")
T0 = time.time()


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")


def sh(cmd, timeout=None, check=False):
    log("$", cmd[:220])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = ((r.stdout or "")[-1500:] + ("\n" + r.stderr[-2500:] if (r.stderr or "").strip() else "")).strip()
    if out:
        log(out)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED rc={r.returncode}: {cmd[:120]}")
    return r


whl = sorted(glob.glob("/kaggle/input/**/swegemma-*.whl", recursive=True))
assert whl, "attach dataset metric/gemma-4-developer-agent-wheelhouse"
WH = pathlib.Path(whl[0]).parent

sh(f"{sys.executable} -m pip install -q uv", check=True)
sh(f"{sys.executable} -m uv venv --python 3.12 {VENV}", timeout=900, check=True)
uv = f"{sys.executable} -m uv pip install --python {VENV}/bin/python"
sh(f"{uv} --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 "
   "torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0", timeout=3600, check=True)
whls = " ".join(str(w) for w in sorted(WH.glob("*.whl")) if "cutlass" not in w.name.lower())
sh(f"{uv} --no-deps {whls}", timeout=1800, check=True)
sh(f"{uv} packaging", check=True)

MISSING = """
import importlib.metadata as md, json
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name as cn
ds = list(md.distributions())
have = {cn(d.metadata['Name']) for d in ds if d.metadata['Name']}
byname = {cn(d.metadata['Name']): d for d in ds if d.metadata['Name']}
miss, wanted = {}, {}  # wanted: package -> extras someone requested (e.g. fastapi[standard])
for d in ds:
    for raw in d.requires or []:
        q = Requirement(raw)
        if q.marker is None or q.marker.evaluate({'extra': ''}):
            wanted.setdefault(cn(q.name), set()).update(q.extras)
            if cn(q.name) not in have:
                miss.setdefault(cn(q.name), f'{q.name}{q.specifier}')
todo = [(n, e) for n, es in wanted.items() for e in es]
seen = set()
while todo:  # follow requested extras transitively
    name, extra = todo.pop()
    if (name, extra) in seen or name not in byname:
        continue
    seen.add((name, extra))
    for raw in byname[name].requires or []:
        q = Requirement(raw)
        if q.marker and q.marker.evaluate({'extra': extra}) and not q.marker.evaluate({'extra': ''}):
            todo += [(cn(q.name), e) for e in q.extras]
            if cn(q.name) not in have:
                miss.setdefault(cn(q.name), f'{q.name}{q.specifier}')
print(json.dumps(sorted(miss.values())))
"""
(W / "missing.py").write_text(MISSING)


def bare(req):
    for ch in "<>=!~":
        req = req.split(ch)[0]
    return req


filled = []
for rnd in range(10):
    out = subprocess.run([f"{VENV}/bin/python", str(W / "missing.py")], capture_output=True, text=True)
    miss = json.loads(out.stdout or "[]")
    if not miss:
        break
    log(f"round {rnd}: {len(miss)} missing: {miss}")
    r = sh(f"{uv} --no-deps " + " ".join(f"'{m}'" for m in miss), timeout=1800)
    if r.returncode != 0:  # a specifier with no matching release -> take any release of that package
        sh(f"{uv} --no-deps " + " ".join(f"'{bare(m)}'" for m in miss), timeout=1800)
    filled += miss
still = json.loads(subprocess.run([f"{VENV}/bin/python", str(W / "missing.py")], capture_output=True, text=True).stdout or "[]")

r = sh(f"{VENV}/bin/python -c 'import torch, vllm, swegemma, google.adk, adk_submission, uvloop, litellm; "
       "print(\"torch\", torch.__version__, \"vllm\", vllm.__version__, \"adk\", google.adk.__version__)'")
freeze = sh(f"{sys.executable} -m uv pip freeze --python {VENV}/bin/python").stdout
(W / "freeze.txt").write_text(freeze)
uv_python_dir = sh(f"{sys.executable} -m uv python dir").stdout.strip()
info = {"python": sh(f"{VENV}/bin/python --version").stdout.strip(), "import_check": (r.stdout or r.stderr).strip()[-500:],
        "import_ok": r.returncode == 0, "filled": filled, "still_missing": still, "uv_python_dir": uv_python_dir,
        "n_packages": len(freeze.splitlines()), "build_min": round((time.time() - T0) / 60, 1)}
(VENV / "install_info.json").write_text(json.dumps(info, indent=1))
(W / "install_info.json").write_text(json.dumps(info, indent=1))
log("INFO:", json.dumps({k: v for k, v in info.items() if k != "filled"}))
if not info["import_ok"]:
    raise SystemExit("env does not import -> not packing")

# pack venv + the uv-managed interpreter it points to, at their absolute paths
rel = [str(VENV).lstrip("/"), uv_python_dir.lstrip("/")]
comp = "pigz -1" if subprocess.run("command -v pigz", shell=True).returncode == 0 else "gzip -1"
sh(f"tar -C / -cf - {' '.join(rel)} | {comp} > {W / 'scorer_env.tgz'}", timeout=3600, check=True)
log(f"DONE: scorer_env.tgz {(W / 'scorer_env.tgz').stat().st_size / 1e9:.2f} GB in {(time.time() - T0) / 60:.1f} min")
