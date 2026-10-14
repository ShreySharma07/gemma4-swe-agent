"""
Local-only shims for running the scorer's subprocess sandbox outside its Docker image. Nothing here changes what the
agent sees; each shim exists because a Kaggle notebook is not the scorer's container.

patch_sandbox_venv: the scorer venv's base Python is Debian's python3.12 without ensurepip (python3.12-venv is not
  installed and L4 sessions are offline). Debian's venv then calls sys.exit(1) inside venv.create(with_pip=True), and
  swegemma's SubprocessManager.start only catches Exception, so the SystemExit kills the whole evaluation. Turning it
  into an exception lets swegemma take its own fallback (venv without pip; pip comes from the host site-packages via
  _host_env.pth, see envsetup.ScorerEnv.ensure_pip).
"""
import contextvars
import venv

from .util import log

TASK = contextvars.ContextVar("evalkit_task", default=None)  # instance_id of the task whose sandbox is being started


def patch_sandbox_venv():
    if getattr(venv.create, "_evalkit", False):
        return
    orig = venv.create

    def create(*a, **kw):
        try:
            return orig(*a, **kw)
        except SystemExit as e:  # Debian venv without ensurepip
            raise RuntimeError(f"venv.create exited ({e}); ensurepip unavailable") from None

    create._evalkit = True
    venv.create = create
    log("compat: venv.create SystemExit -> RuntimeError (swegemma falls back to a venv without pip)")


def use_scorer_sandbox(data_dir):
    """Make each task sandbox look like the scorer's Docker sandbox (python:3.13-slim + Dockerfile pip packages +
    every competition wheel unpacked + sandbox/setup.py --fast-path), as in kaggle/devset/build_devset.py, where gold
    patches then pass 54/54 on the exec-graph tasks.

    - In subprocess mode swegemma skips test-dependency setup and task venvs inherit the HOST site-packages (here the
      harness's Py3.12 venv: no pytest, wrong versions). Instead: no system site-packages, a fresh Python 3.13 venv
      (the image's interpreter, matching python:3.13-slim and the cp313 wheels in scorer_site), and a .pth pointing at
      the restored scorer site (gemma4-devset-build output) plus a pip-only dir (Debian Python has no ensurepip).
    - install_test_dependencies runs the competition's sandbox/setup.py, like the scorer's container setup.
    """
    import json
    import pathlib
    import subprocess

    import swegemma.evaluate as ev
    import swegemma.harness.container_setup as cs
    import swegemma.harness.verification as vf
    import swegemma.sandbox.subprocess as sp

    from .config import SANDBOX_ENV

    env = json.loads(SANDBOX_ENV.read_text())
    if not getattr(sp.SubprocessManager, "_evalkit", False):
        orig_init, orig_start = sp.SubprocessManager.__init__, sp.SubprocessManager.start

        def init(self, *a, **kw):
            kw["system_site_packages"] = False
            orig_init(self, *a, **kw)

        def start(self, *a, **kw):
            sid = orig_start(self, *a, **kw)
            vdir = pathlib.Path(self._sandboxes[sid]["venv"])
            subprocess.run([env["python"], "-m", "venv", "--without-pip", "--clear", str(vdir)], check=True,
                           capture_output=True)
            # search order: this task's version pins -> test-only extras -> scorer site -> pip
            path = [f"{env['pins_root']}/{d}" for d in env.get("pins", {}).get(TASK.get(), [])]
            path += [env["extras"]] if env.get("extras") else []
            path += [env["site"], env["pip"]]
            for spd in vdir.glob("lib/python*/site-packages"):
                (spd / "_scorer_site.pth").write_text("\n".join(path) + "\n")
            return sid

        sp.SubprocessManager.__init__, sp.SubprocessManager.start = init, start
        sp.SubprocessManager._evalkit = True

    if not getattr(ev.Evaluator, "_evalkit_task", False):  # asyncio.to_thread copies the context into start()
        orig_eval = ev.Evaluator.evaluate_task

        async def evaluate_task(self, task, *a, **kw):
            TASK.set(task.instance_id)
            return await orig_eval(self, task, *a, **kw)

        ev.Evaluator.evaluate_task, ev.Evaluator._evalkit_task = evaluate_task, True

    setup_py = pathlib.Path(data_dir) / "sandbox" / "setup.py"
    assert setup_py.exists(), f"missing {setup_py}"
    orig_deps = cs.install_test_dependencies

    def install_test_dependencies(docker, container_id, repo="", *, fast_path=True, config=None):
        if type(docker).__name__ not in {"SubprocessManager", "SubprocessSandbox"}:
            return orig_deps(docker, container_id, repo, fast_path=fast_path, config=config)
        flag = "--fast-path" if fast_path else "--no-fast-path"
        docker.exec(container_id, f"python3 {setup_py} {flag} --workspace /workspace {repo}".strip(), timeout=600)

    for mod in (cs, vf, ev):
        if hasattr(mod, "install_test_dependencies"):
            mod.install_test_dependencies = install_test_dependencies
    log(f"compat: task sandboxes = {env['python']} venv + scorer site {env['site']} + {setup_py.name}"
        + (f" + extras + pins for {len(env.get('pins', {}))} tasks" if env.get("extras") else ""))
