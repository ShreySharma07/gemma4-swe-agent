"""Find vs finish: does the agent fail to *locate* the fix, or to *complete* it?

Reads trace_analyzer per_task.jsonl files (gold_files / gold_funcs / patch_files) plus the
agent's patch files, and splits each task into:
  found    = agent's patch touches >=1 gold file
  complete = agent's patch touches every gold file (file level)
  partial  = found but not complete
Function level is approximate: an agent patch "covers" a gold function if the name appears
in a hunk header context or in a changed `def`/`class` line.

usage: python analysis/find_vs_finish.py RUN_DIR [RUN_DIR ...]   (RUN_DIR = .../repN)
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HUNK = re.compile(r"^@@ [^@]* @@ ?(.*)$")
DEF = re.compile(r"^[+-]\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")
CTX_NAME = re.compile(r"(?:def|class)\s+([A-Za-z_]\w*)")


def patch_funcs(text):
    names = set()
    for line in text.splitlines():
        m = HUNK.match(line)
        if m:
            names.update(CTX_NAME.findall(m.group(1)))
            continue
        m = DEF.match(line)
        if m:
            names.add(m.group(1))
    return names


def py_gold(files):
    return [f for f in files if f.endswith(".py") and "/tests/" not in f and not f.startswith("tests/")]


def load(run):
    run = Path(run)
    rows = [json.loads(l) for l in open(run / "analysis" / "per_task.jsonl")]
    out = []
    for r in rows:
        if r.get("env_unverifiable"):
            continue
        gf = set(py_gold(r.get("gold_files") or []))
        if not gf:
            continue
        pf = set(r.get("patch_files") or [])
        pp = run / "merged" / "patches" / f"{r['instance_id']}.patch"
        pfun = patch_funcs(pp.read_text(errors="replace")) if pp.exists() else set()
        gfun = {g.split(".")[-1] for g in (r.get("gold_funcs") or []) if g and g != "<module>"}
        out.append(dict(
            id=r["instance_id"], resolved=bool(r.get("resolved")),
            n_gf=len(gf), n_gfun=len(gfun),
            saw=r.get("step_first_gold_file") is not None,
            found=bool(gf & pf), complete=gf <= pf,
            fun_cov=(len(gfun & pfun) / len(gfun)) if gfun else None,
            empty=bool(r.get("patch_empty")), overflow=bool(r.get("overflow")),
        ))
    return out


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "0/0"


def report(name, rows):
    n = len(rows)
    res = [r for r in rows if r["resolved"]]
    un = [r for r in rows if not r["resolved"]]
    print(f"\n## {name}: {len(res)}/{n} resolved")
    for label, sel in (("1 gold file", lambda r: r["n_gf"] == 1), (">=2 gold files", lambda r: r["n_gf"] >= 2),
                       ("<=1 gold func", lambda r: r["n_gfun"] <= 1), (">=2 gold funcs", lambda r: r["n_gfun"] >= 2)):
        s = [r for r in rows if sel(r)]
        print(f"  {label:16s} resolved {pct(sum(r['resolved'] for r in s), len(s))}")
    c = Counter()
    for r in un:
        if r["empty"]:
            c["empty patch"] += 1
        elif not r["found"]:
            c["patch misses every gold file"] += 1
        elif not r["complete"]:
            c["PARTIAL: some gold files untouched"] += 1
        elif r["fun_cov"] is not None and r["fun_cov"] < 1:
            c["PARTIAL: all gold files, some gold funcs untouched"] += 1
        else:
            c["all gold files+funcs touched, still wrong"] += 1
    print(f"  unresolved ({len(un)}), saw gold file in {pct(sum(r['saw'] for r in un), len(un))}:")
    for k, v in c.most_common():
        print(f"    {k:52s} {v}")
    multi = [r for r in rows if r["n_gf"] >= 2 and r["found"]]
    print(f"  multi-file tasks where a gold file was edited: complete {pct(sum(r['complete'] for r in multi), len(multi))}, "
          f"resolved when complete {pct(sum(r['resolved'] for r in multi if r['complete']), sum(r['complete'] for r in multi))}, "
          f"when partial {pct(sum(r['resolved'] for r in multi if not r['complete']), sum(not r['complete'] for r in multi))}")
    fc = [r for r in rows if r["found"] and r["n_gfun"] >= 2 and r["fun_cov"] is not None]
    full = [r for r in fc if r["fun_cov"] >= 1]
    part = [r for r in fc if r["fun_cov"] < 1]
    print(f"  >=2 gold funcs, gold file edited: full func coverage resolved {pct(sum(r['resolved'] for r in full), len(full))}, "
          f"partial coverage resolved {pct(sum(r['resolved'] for r in part), len(part))}")
    return rows


if __name__ == "__main__":
    for run in sys.argv[1:]:
        report(run, load(run))
