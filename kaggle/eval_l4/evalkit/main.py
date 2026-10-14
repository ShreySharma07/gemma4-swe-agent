"""Everything after the env step. Runs inside the scorer venv: python -m evalkit.main (cwd /kaggle/working)."""
import time

import litellm

from .compat import patch_sandbox_venv, use_scorer_sandbox
from .config import Settings
from .inputs import Inputs
from .report import Report
from .runner import ChunkedEvaluation
from .server import ScorerServer, make_sandbox_offline, set_scorer_env
from .submission import Submission
from .util import log


def main():
    s = Settings.load()
    set_scorer_env()
    patch_sandbox_venv()
    litellm.drop_params = True
    inp = Inputs.locate(s.devset)
    use_scorer_sandbox(inp.data)
    sub = Submission().unpack()
    declared, adapters = sub.declared_model(), sub.adapters()
    log("declared model:", declared, "| adapters:", list(getattr(adapters, "adapters", {}) or {}))
    server = ScorerServer(inp.model_path, adapters).start()
    models = server.registry(declared)
    make_sandbox_offline()
    report = Report(s, inp)
    try:
        ChunkedEvaluation(s, inp, sub, models, adapters).run()
    finally:
        report.build()
        report.package()
        server.stop()
        log(f"RUN_DONE in {(time.time() - s.t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
