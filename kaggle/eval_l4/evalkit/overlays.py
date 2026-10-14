"""
Overlays: extra packages for the task sandboxes so more public tasks are verifiable (CPU kernel, Internet ON).

Why: 75/129 public tasks fail even with the gold patch in the scorer env (Oct 14 sweep): fastapi tests import
inline_snapshot / dirty_equals (31), python-multipart (4), old fastapi commits call Starlette APIs removed in 1.x (24), requests tests need
the pytest-httpbin plugin (7), rich colour snapshots changed with newer Pygments (7), `attr` missing (1). The leaderboard
scores private repos, so these public tasks only need to measure the agent reliably, not match the scorer.
Output (/kaggle/working/overlays):
  extras/      test-only packages for every task, resolved against the scorer site's versions (packages the site
               already has are removed, so only new ones are added)
  pins/<pkg>-<version>/   one package each, --no-deps
  pins.json    {instance_id: [pin dir, ...]}: only tasks that are invalid today, and only for PINNED repos; each
               package pinned to its latest release before the task's created_at (the env the tests were written in)
  report.json  what was built
Used by compat.use_scorer_sandbox (sandbox path: task pins -> extras -> scorer site -> pip); the gold/empty sweep
then decides which tasks are valid.
Run: python -m evalkit.overlays (no venv needed: plain pip in the notebook's Python 3.13, same as the sandbox).
"""
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import urllib.request

from .config import INPUT, SCORER_SITE, VALID_TASKS, WORK
from .util import first, log

EXTRAS = ["inline-snapshot", "dirty-equals", "pytest-httpbin", "httpbin", "attrs", "python-multipart"]
PINNED = {"fastapi": ["starlette"], "rich": ["pygments"]}
OUT = WORK / "overlays"


def canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()


class OverlayBuilder:
    def __init__(self, tasks, valid):
        self.tasks, self.valid = tasks, set(valid)
        self.report = {"extras": [], "pins": {}, "errors": []}

    def pip(self, *args):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *args],
                           capture_output=True, text=True)
        if r.returncode:
            self.report["errors"].append(" ".join(args)[:200] + " -> " + r.stderr[-400:])
        return r.returncode == 0

    def site_versions(self):
        """{canonical name: version} of the scorer site (restored from gemma4-devset-build's scorer_site.tgz)."""
        tgz = first(f"{INPUT}/**/scorer_site.tgz")
        assert tgz, "attach the output of kernel gemma4-devset-build (scorer_site.tgz)"
        SCORER_SITE.mkdir(parents=True, exist_ok=True)
        log.sh(f"tar -xzf {tgz} -C {SCORER_SITE}", timeout=1800, check=True)
        have = {}
        for d in SCORER_SITE.glob("*.dist-info"):
            m = re.match(r"(.+?)-([^-]+)\.dist-info$", d.name)
            if m:
                have[canon(m.group(1))] = m.group(2)
        return have

    def build_extras(self):
        have = self.site_versions()
        cons = WORK / "site_constraints.txt"
        cons.write_text("".join(f"{n}=={v}\n" for n, v in sorted(have.items())))
        ext = OUT / "extras"
        shutil.rmtree(ext, ignore_errors=True)
        for pkg in EXTRAS:  # one at a time: a resolution conflict in one must not block the others
            how = "constrained" if self.pip("--target", str(ext), "-c", str(cons), pkg) else (
                "unconstrained" if self.pip("--target", str(ext), pkg) else "FAILED")
            self.report.setdefault("extras_install", {})[pkg] = how
            log(f"extra {pkg}: {how}")
        for d in list(ext.glob("*.dist-info")):  # drop what the site already provides (same pinned version)
            name = canon(re.match(r"(.+?)-[^-]+\.dist-info$", d.name).group(1))
            if name in have:
                for line in (d / "RECORD").read_text().splitlines() if (d / "RECORD").exists() else []:
                    top = line.split(",")[0].split("/")[0]
                    p = (ext / top).resolve()
                    # only real top-level entries inside extras/ (RECORD also lists scripts as ../../bin/...)
                    if top in ("", ".", "..") or top.endswith(".dist-info") or p.parent != ext.resolve():
                        continue
                    shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
                shutil.rmtree(d, ignore_errors=True)
        self.report["extras"] = sorted(d.name for d in ext.glob("*.dist-info"))
        log("extras:", self.report["extras"])

    @staticmethod
    def releases(pkg):
        """[(version, upload time)] of non-yanked final releases, from PyPI's JSON API."""
        data = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=60))
        out = []
        for v, files in data["releases"].items():
            if not files or re.search(r"[a-zA-Z]", v) or all(f.get("yanked") for f in files):
                continue
            t = min(dt.datetime.fromisoformat(f["upload_time_iso_8601"].replace("Z", "+00:00")) for f in files)
            out.append((v, t))
        return sorted(out, key=lambda x: x[1])

    def build_pins(self):
        rel = {p: self.releases(p) for ps in PINNED.values() for p in ps}
        pins = {}
        for t in self.tasks:
            iid, repo_short = t["instance_id"], t["instance_id"].rsplit("_", 1)[0]
            if iid in self.valid or repo_short not in PINNED:
                continue
            when = dt.datetime.fromisoformat(str(t.get("created_at")).replace("Z", "+00:00"))
            if when.tzinfo is None:
                when = when.replace(tzinfo=dt.timezone.utc)
            for pkg in PINNED[repo_short]:
                before = [v for v, up in rel[pkg] if up <= when]
                if not before:
                    continue
                name = f"{pkg}-{before[-1]}"
                d = OUT / "pins" / name
                if not d.exists() and not self.pip("--no-deps", "--target", str(d), f"{pkg}=={before[-1]}"):
                    continue
                pins.setdefault(iid, []).append(name)
        (OUT / "pins.json").write_text(json.dumps(pins, indent=1))
        self.report["pins"] = {i: v for i, v in sorted(pins.items())}
        log(f"pins for {len(pins)} tasks:", sorted({n for v in pins.values() for n in v}))

    def build(self):
        OUT.mkdir(parents=True, exist_ok=True)
        self.build_extras()
        self.build_pins()
        (OUT / "report.json").write_text(json.dumps(self.report, indent=1))
        log("overlays built:", json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v)
                                          for k, v in self.report.items()}))


def main():
    tasks_path = next(p for p in sorted(INPUT.rglob("tasks.jsonl")) if (p.parent / "graphs").is_dir())
    tasks = [json.loads(l) for l in open(tasks_path) if l.strip()]
    valid = json.loads(VALID_TASKS.read_text()) if VALID_TASKS.exists() else []
    log(f"{len(tasks)} tasks, {len(valid)} valid today")
    OverlayBuilder(tasks, valid).build()


if __name__ == "__main__":
    main()
