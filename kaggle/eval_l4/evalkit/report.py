"""Report: merge chunk outputs per rep, pass@k summary, trace analysis, and one tarball for download."""
import json
import shutil
import sys

from .config import ANALYZER, WORK
from .util import log

PARTS = ("traces", "patches", "test_outputs", "logs")


class Report:
    def __init__(self, settings, inputs):
        self.s, self.inp = settings, inputs
        self.root = WORK / settings.label

    def merge_rep(self, rep_dir):
        merged = rep_dir / "merged"
        for part in PARTS:
            (merged / part).mkdir(parents=True, exist_ok=True)
        with open(merged / "task_results.jsonl", "w") as out:
            for ch in sorted(rep_dir.glob("chunk_*")):
                if (ch / "task_results.jsonl").exists():
                    out.write((ch / "task_results.jsonl").read_text())
                for part in PARTS:
                    for f in (ch / part).glob("*"):
                        shutil.copy(f, merged / part / f.name)
                shutil.rmtree(ch)
        return merged, [json.loads(l) for l in open(merged / "task_results.jsonl") if l.strip()]

    def analyze(self, merged, out):
        if ANALYZER.exists():
            log.sh(f"{sys.executable} {ANALYZER} --results {merged} --tasks {self.inp.tasks_path} --out {out}")

    def build(self):
        summary = {"name": self.s.name, "devset": self.s.devset, "reps": {}}
        solved = {}
        for rep_dir in sorted(self.root.glob("rep*")):
            merged, rows = self.merge_rep(rep_dir)
            for r in rows:
                solved.setdefault(r["instance_id"], []).append(bool(r.get("resolved")))
            summary["reps"][rep_dir.name] = {"resolved": sum(bool(r.get("resolved")) for r in rows), "n": len(rows)}
            self.analyze(merged, rep_dir / "analysis")
        summary["pass_at_k"] = {"tasks": len(solved), "any": sum(any(v) for v in solved.values()),
                                "all": sum(all(v) for v in solved.values())}
        (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
        log("SUMMARY", json.dumps(summary))
        return summary

    def package(self):
        log.sh(f"cd {WORK} && tar -czf {self.s.label}.tgz {self.s.label} task_ids_order.json run.log summary.json "
               f"&& rm -rf {self.s.label} submission_under_test; ls -la {WORK}")
