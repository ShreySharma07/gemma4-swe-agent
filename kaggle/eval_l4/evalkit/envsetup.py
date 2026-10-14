"""
ScorerEnv: the scorer's Python environment, offline. Runs in the notebook kernel (Python 3.13), stdlib only.

The wheelhouse is cp312-only and not a consistent dependency set, so the venv is built once on a CPU kernel with
Internet (kaggle/env/build_scorer_env.py -> scorer_env.tgz). Here it is restored, then every wheelhouse wheel whose
version differs from the venv (Oct 9: swegemma 0.2.11, adk-submission 0.2.13; their requirements are unchanged)
is installed with --no-deps, plus the test-only deps of the fastapi tasks (dataset test-deps-wheels).
"""
import json
import os
import pathlib
import re
import sys
import time

from .config import INPUT, SANDBOX_ENV, SANDBOX_PIP, SCORER_SITE, VENV
from .util import first, log

TEST_DEPS = re.compile(r"(inline_snapshot|dirty_equals|asttokens|executing)-")


def canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()


class ScorerEnv:
    def __init__(self, venv=VENV, n_gpus=4):
        self.venv, self.n_gpus = pathlib.Path(venv), n_gpus
        self.python = self.venv / "bin" / "python"

    def check_gpus(self):
        log.sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | head -2")
        gpus = [g for g in log.sh("nvidia-smi --query-gpu=name --format=csv,noheader").stdout.split("\n") if g.strip()]
        if len(gpus) < self.n_gpus:  # don't burn quota on the wrong machine
            raise SystemExit(f"need GPU L4 x{self.n_gpus}, got {gpus}")

    def restore(self):
        tgz = first(f"{INPUT}/**/scorer_env.tgz")
        assert tgz, "attach the output of kernel gemma4-scorer-env (kernel_sources)"
        log.sh(f"tar -xzf {tgz} -C /", timeout=1800, check=True)

    def installed(self):
        out = log.sh(f"{self.python} -c 'import importlib.metadata as m, json; "
                     "print(json.dumps({d.metadata[\"Name\"]: d.version for d in m.distributions()}))'",
                     check=True).stdout.strip().splitlines()[-1]
        return {canon(k): v.split("+")[0] for k, v in json.loads(out).items()}

    def stale_wheels(self):
        swegemma = first(f"{INPUT}/**/swegemma-*.whl")
        assert swegemma, "attach dataset metric/gemma-4-developer-agent-wheelhouse"
        have, stale = self.installed(), []
        for w in sorted(pathlib.Path(swegemma).parent.glob("*.whl")):
            if "cutlass" in w.name.lower():
                continue
            n, v = w.name.split("-")[:2]
            v = re.sub(r"\+?cu\d+$", "", v)
            if have.get(canon(n)) != v:
                log(f"wheelhouse {n} {v} != venv {have.get(canon(n))}")
                stale.append(w)
        return stale

    def test_dep_wheels(self):
        found = [pathlib.Path(p) for p in sorted(map(str, INPUT.rglob("*.whl"))) if TEST_DEPS.match(pathlib.Path(p).name)]
        if not found:
            log("WARNING: dataset test-deps-wheels not attached -> ~32 fastapi tasks will be unverifiable")
        return found

    def install(self, wheels):
        stage = pathlib.Path("/tmp/wh_stage")
        stage.mkdir(exist_ok=True)
        for w in wheels:  # pip rejects 'cu128' without '+' in a wheel's version
            name = w.name.replace("cu128", "+cu128") if ("cu128" in w.name and "+" not in w.name) else w.name
            if not (stage / name).exists():
                os.symlink(w, stage / name)
        whls = sorted(str(p) for p in stage.glob("*.whl"))
        if not whls:
            return
        r = log.sh(f"{sys.executable} -m pip --python {self.python} install --no-deps --no-index -q " + " ".join(whls),
                   timeout=1800)
        if r.returncode != 0:  # fallback: wheels are zip archives, unpack into site-packages
            site = log.sh(f"{self.python} -c 'import site; print(site.getsitepackages()[0])'").stdout.strip()
            for w in whls:
                log.sh(f"cd {site} && unzip -oq {w}", check=True)

    def ensure_pip(self):
        """uv venvs have no pip; the sandbox's per-task venvs need `python -m pip` (found via _host_env.pth) to
        install task packages. pip is pure Python, so copy the notebook Python's own pip into the venv (offline)."""
        if log.sh(f"{self.python} -m pip --version").returncode == 0:
            return
        import pip
        src = pathlib.Path(pip.__file__).parent
        site = pathlib.Path(log.sh(f"{self.python} -c 'import site; print(site.getsitepackages()[0])'",
                                   check=True).stdout.strip())
        log.sh(f"cp -r {src} {site}/ && cp -r {src.parent}/pip-*.dist-info {site}/", check=True)
        log.sh(f"{self.python} -m pip --version", check=True)

    def task_sandbox(self):
        """Task sandboxes run on this notebook's Python 3.13 (the scorer's sandbox is python:3.13-slim) with the
        scorer site built by gemma4-devset-build; pip is copied into its own dir (no ensurepip offline)."""
        import pip
        tgz = first(f"{INPUT}/**/scorer_site.tgz")
        assert tgz, "attach the output of kernel gemma4-devset-build (scorer_site.tgz)"
        SCORER_SITE.mkdir(parents=True, exist_ok=True)
        log.sh(f"tar -xzf {tgz} -C {SCORER_SITE}", timeout=1800, check=True)
        SANDBOX_PIP.mkdir(parents=True, exist_ok=True)
        src = pathlib.Path(pip.__file__).parent
        log.sh(f"cp -r {src} {SANDBOX_PIP}/ && cp -r {src.parent}/pip-*.dist-info {SANDBOX_PIP}/", check=True)
        py = os.path.realpath(sys.executable)
        env = {"python": py, "site": str(SCORER_SITE), "pip": str(SANDBOX_PIP)}
        pins = first(f"{INPUT}/**/overlays/pins.json")  # optional: kernel output gemma4-test-overlays
        if pins:
            root = pathlib.Path(pins).parent
            env.update(extras=str(root / "extras"), pins_root=str(root / "pins"), pins=json.loads(pathlib.Path(pins).read_text()))
            log(f"overlays: extras + pins for {len(env['pins'])} tasks from {root}")
        SANDBOX_ENV.write_text(json.dumps(env))
        sp = f"/tmp/sbx_check/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        log.sh(f"{py} -m venv --without-pip --clear /tmp/sbx_check && printf '%s\\n%s\\n' {SCORER_SITE} {SANDBOX_PIP} "
               f"> {sp}/_s.pth && /tmp/sbx_check/bin/python3 -c 'import sys, pytest, pip; print(\"task sandbox\", "
               "sys.version.split()[0], \"pytest\", pytest.__version__, \"pip\", pip.__version__)'", check=True)

    def verify(self):
        log.sh(f"{self.python} -c 'import sys, torch, vllm, swegemma, adk_submission, google.adk, "
               "importlib.metadata as m; print(sys.version.split()[0], \"torch\", torch.__version__, \"vllm\", "
               "vllm.__version__, \"swegemma\", m.version(\"swegemma\"), \"adk-submission\", "
               "adk_submission.__version__, \"adk\", google.adk.__version__)'", check=True)
        # import needs pytest's pluggy, which the sandbox venv installs per task; here only check presence
        log.sh(f"{self.python} -c 'import importlib.util as u; "
               "print(\"test deps\", {m: bool(u.find_spec(m)) for m in (\"inline_snapshot\", \"dirty_equals\")})'")

    def prepare_training(self, gpus=True):
        """Scorer venv + the training wheels (peft 0.21.2, bitsandbytes 0.50.2: kernel output gemma4-train-wheels)."""
        t = time.time()
        if gpus:
            self.check_gpus()
        self.restore()
        train = [pathlib.Path(p) for p in sorted(map(str, INPUT.rglob("train_wheels/*.whl")))]
        assert train, "attach the output of kernel gemma4-train-wheels (kernel_sources)"
        self.install(self.stale_wheels() + train)
        self.ensure_pip()
        log.sh(f"{self.python} -c 'import torch, transformers, peft, bitsandbytes; print(\"torch\", torch.__version__, "
               "\"transformers\", transformers.__version__, \"peft\", peft.__version__, \"bnb\", bitsandbytes.__version__)'",
               check=True)
        log(f"training venv ready in {(time.time() - t) / 60:.1f} min")

    def prepare(self, gpus=True):
        t = time.time()
        if gpus:
            self.check_gpus()
        self.restore()
        self.install(self.stale_wheels() + self.test_dep_wheels())
        self.ensure_pip()
        self.task_sandbox()
        self.verify()
        log(f"scorer venv ready in {(time.time() - t) / 60:.1f} min")
