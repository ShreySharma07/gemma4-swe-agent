"""Submission: the agent under test, unpacked from submission_src.zip (written by the notebook's inputs cell)."""
import shutil
import zipfile

import yaml

from .config import SUBMISSION_ZIP, WORK
from .util import log


class Submission:
    def __init__(self, src_zip=SUBMISSION_ZIP, dest=WORK / "submission_under_test"):
        self.src_zip, self.dir = src_zip, dest

    def unpack(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        with zipfile.ZipFile(self.src_zip) as z:
            z.extractall(self.dir)
        self.fetch_adapters()
        shutil.make_archive(str(WORK / "submission"), "zip", self.dir)  # exactly what would be uploaded
        log("submission files:", sorted(str(p.relative_to(self.dir)) for p in self.dir.rglob("*") if p.is_file()))
        return self

    def fetch_adapters(self):
        """Adapters named in any agent YAML (`adapter: <name>`) but not shipped in the zip are copied from an attached
        training run's output (/kaggle/input/**/adapters/<name>/), so a trained LoRA is evaluated without re-uploading."""
        import re
        from .config import INPUT
        names = {m for p in self.dir.rglob("*.yaml") for m in re.findall(r"^\s*adapter:\s*(\S+)", p.read_text(), re.M)}
        for n in sorted(names):
            dest = self.dir / "adapters" / n
            if (dest / "adapter_model.safetensors").exists():
                continue
            found = sorted(INPUT.glob(f"**/adapters/{n}/adapter_model.safetensors"))
            assert found, f"agent declares adapter {n!r}: not in the submission and no attached adapters/{n}/"
            shutil.copytree(found[0].parent, dest)
            log(f"adapter {n} <- {found[0].parent}")

    def declared_model(self):
        """Exactly one model across all agents (scorer rule); swegemma.models.discovery is gone in 0.2.11."""
        from adk_submission import discover_declared_models
        declared = sorted({m.strip() for m in discover_declared_models(self.dir)})
        assert len(declared) == 1, f"submission must declare exactly one model, got {declared}"
        return declared[0]

    def adapters(self):
        from adk_submission import discover_adapters
        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
        return discover_adapters(str(self.dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)

    def eval_budget(self):
        """eval_config.yaml -> {timeout_seconds, max_time_minutes, max_tool_calls, max_turns}."""
        p = self.dir / "eval_config.yaml"
        ev = (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}
        ev = ev.get("evaluation", ev)
        turns = ev.get("max_turns", ev.get("max_llm_calls"))
        return {"timeout_seconds": ev.get("timeout_seconds"), "max_time_minutes": ev.get("max_time_minutes"),
                "max_tool_calls": ev.get("max_tool_calls"), "max_turns": int(turns) if turns is not None else None}
