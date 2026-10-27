"""
Shared Kaggle-kernel header for the task factory: the official harness from the wheelhouse and the scorer's Python
environment for subprocess-sandbox verification. Copied from kaggle/devset/build_devset.py (sections 0 and 3); the
factory's build.sh prepends it to factory_core.py + factory_main.py. Expects OUT (pathlib.Path) to be defined.
"""
import asyncio
import glob
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def first(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None


# ------------------------------------------------------------------ 0. inputs + official harness
_all = sorted(glob.glob("/kaggle/input/**/tasks.jsonl", recursive=True))
_comp = [p for p in _all if (pathlib.Path(p).parent / "graphs").is_dir()]
tasks_path = (_comp or _all or [None])[0]
assert tasks_path, "attach the competition data"
DATA = pathlib.Path(tasks_path).parent
log("competition data:", DATA)


def ensure_harness():
    try:
        import swegemma  # noqa: F401
        from swegemma.evaluate import Evaluator  # noqa: F401
        return True
    except Exception as e:  # noqa: BLE001
        log("installing harness from wheelhouse:", type(e).__name__, e)
    wh = first("/kaggle/input/**/swegemma-*.whl")
    if not wh:
        log("wheelhouse dataset not attached -> cannot verify")
        return False
    skip = ("torch", "vllm", "nvidia", "triton", "cutlass", "xformers", "flashinfer", "flash_attn", "cupy")
    # the wheelhouse has cp312-only binaries; the Kaggle image may be newer (3.13 since Oct 2026), so install the
    # wheels this interpreter supports and fetch the same pinned versions of the rest from PyPI (Internet ON)
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    ok_tags = set(sys_tags())
    whls, pins = [], []
    for w in pathlib.Path(wh).parent.glob("*.whl"):
        if w.name.lower().startswith(skip):
            continue
        name, ver, _, tags = parse_wheel_filename(w.name)
        if tags & ok_tags:
            whls.append(str(w))
        else:
            pins.append(f"{name}=={ver}")
    log(f"wheelhouse: {len(whls)} compatible wheels; from PyPI: {pins}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall"] + whls)
    if pins:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps"] + pins)
        if r.returncode:  # exact pin missing for this Python -> take the nearest available release
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps"] + [p.split("==")[0] for p in pins])
    try:
        import importlib
        importlib.invalidate_caches()
        from swegemma.evaluate import Evaluator  # noqa: F401
        return True
    except Exception as e:  # noqa: BLE001
        log("harness import still failing:", type(e).__name__, e)
        return False


def ensure_ensurepip():
    # swegemma builds each task venv with ensurepip; without it the venv has no pip and the editable install of the
    # repo fails silently (`... || true`). Debian Pythons ship ensurepip separately (python3.X-venv).
    chk = [sys.executable, "-c", "import ensurepip"]
    if subprocess.run(chk, capture_output=True).returncode == 0:
        return True
    v = f"{sys.version_info.major}.{sys.version_info.minor}"
    log(f"ensurepip missing for python {v} -> apt-get install python{v}-venv")
    subprocess.run("apt-get update -qq", shell=True, capture_output=True)
    for pkg in (f"python{v}-venv", "python3-venv"):
        subprocess.run(f"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq {pkg}", shell=True, capture_output=True)
        if subprocess.run(chk, capture_output=True).returncode == 0:
            log("ensurepip OK via", pkg)
            return True
    log("WARNING: ensurepip still unavailable -> task venvs will have no pip")
    return False


SCORER_SITE = pathlib.Path("/tmp/scorer_site")


def fill_missing_requirements(site, rounds=4):
    """The competition wheels are not dependency-complete (pydantic 2.13 needs typing_inspection, absent from the
    wheels), so a wheels-only site can't import pydantic. Add only the missing requirements from PyPI; never change
    a version that is already present."""
    import importlib.metadata as md
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    added = []
    for _ in range(rounds):
        dists = list(md.distributions(path=[str(site)]))
        have = {canonicalize_name(d.metadata["Name"]) for d in dists if d.metadata["Name"]}
        missing = set()
        for d in dists:
            for raw in d.requires or []:
                req = Requirement(raw)
                if req.marker and not req.marker.evaluate({"extra": ""}):
                    continue
                if canonicalize_name(req.name) not in have:
                    missing.add(f"{req.name}{req.specifier}")
        if not missing:
            break
        fill = site.parent / "scorer_site_fill"
        shutil.rmtree(fill, ignore_errors=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(fill)] + sorted(missing))
        for item in fill.iterdir():
            if not (site / item.name).exists():
                shutil.move(str(item), str(site / item.name))
        added += sorted(missing)
    log(f"scorer site: added missing requirements from PyPI: {added}")


def use_scorer_env():
    """Make subprocess-sandbox verification see the scorer's Python environment instead of the Kaggle image's.

    In subprocess mode swegemma skips dependency setup and task venvs inherit host site-packages, so tests ran against
    Kaggle's starlette / pygments / requests (gold failed on ~70 original tasks). The Docker scorer instead has:
    python:3.13-slim + Dockerfile pip packages, the highest py313-compatible version of every competition wheel
    unpacked into site-packages, and `sandbox/setup.py --fast-path` (editable install + workspace_paths.pth).
    """
    import swegemma.evaluate as ev
    import swegemma.harness.container_setup as cs
    import swegemma.harness.verification as vf
    import swegemma.sandbox.subprocess as sp

    if not (SCORER_SITE / ".swegemma_scorer_site").exists():
        shutil.rmtree(SCORER_SITE, ignore_errors=True)
        SCORER_SITE.mkdir(parents=True)
        dockerfile_pkgs = ["pytest", "pytest-timeout==2.1.0", "typer", "pdm-backend", "setuptools", "wheel",
                           "poetry-core", "hatchling", "flit-core", "editables"]  # docker/Dockerfile.sandbox
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", str(SCORER_SITE)] + dockerfile_pkgs)
        base, large, _, _ = cs._inspect_wheels_dir(DATA / "wheels")
        tar = cs._build_unpacked_wheels_tar(list(base) + list(large.values()), "scorer_all", ".swegemma_scorer_site")
        subprocess.run(["tar", "--no-same-owner", "-xf", str(tar), "-C", str(SCORER_SITE)], check=True)
        log(f"scorer site: {len(base)} base + {len(large)} large wheels -> {SCORER_SITE}")
        fill_missing_requirements(SCORER_SITE)

    if not getattr(sp.SubprocessManager, "_scorer_env", False):
        orig_init, orig_start = sp.SubprocessManager.__init__, sp.SubprocessManager.start

        def init(self, *a, **kw):
            kw["system_site_packages"] = False
            orig_init(self, *a, **kw)

        def start(self, *a, **kw):
            sid = orig_start(self, *a, **kw)
            for spd in pathlib.Path(self._sandboxes[sid]["venv"]).glob("lib/python*/site-packages"):
                (spd / "_scorer_site.pth").write_text(f"{SCORER_SITE}\n")
            return sid

        sp.SubprocessManager.__init__, sp.SubprocessManager.start = init, start
        sp.SubprocessManager._scorer_env = True

    setup_py = DATA / "sandbox" / "setup.py"
    orig_deps = cs.install_test_dependencies

    def install_test_dependencies(docker, container_id, repo="", *, fast_path=True, config=None):
        if type(docker).__name__ not in {"SubprocessManager", "SubprocessSandbox"}:
            return orig_deps(docker, container_id, repo, fast_path=fast_path, config=config)
        flag = "--fast-path" if fast_path else "--no-fast-path"
        docker.exec(container_id, f"python3 {setup_py} {flag} --workspace /workspace {repo}".strip(), timeout=600)

    for mod in (cs, vf, ev):
        if hasattr(mod, "install_test_dependencies"):
            mod.install_test_dependencies = install_test_dependencies


