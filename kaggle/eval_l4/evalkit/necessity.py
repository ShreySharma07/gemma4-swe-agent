"""
A3/A4 (CPU, offline, no model): test-necessity of gold function groups and the completion counterfactual, through
the official Evaluator with a fake agent, exactly like the preflight sweep. Pre-registered in docs/experiments.md.
  A3  new-60 tasks whose gold touches >= 2 function groups: gold, empty, and gold minus group i (loo<i>).
  A4  every unresolved, non-empty agent patch (embedded runs) that touches >= 1 gold group and misses >= 1:
      agent   = the agent's patch alone (control, should stay unresolved)
      comp    = agent patch + gold hunks of the missed groups + gold module-level hunks (the completion)
      missed  = those gold hunks alone (control: was the agent's own part needed?)
Function groups use the exec-graph rule: each gold hunk maps to the innermost function(s) enclosing its changed lines.
Inputs: /kaggle/working/a3a4_payload.tgz (agent patches + per_task rows + old-54 LOO results), embedded by
make_eval_notebook.py --necessity. Output: /kaggle/working/necessity/*.jsonl|json and necessity_report.json.
Run inside the venv: python -m evalkit.necessity
"""
import ast
import json
import pathlib
import re
import shutil
import subprocess
import tarfile
import tempfile
import time

from .compat import patch_sandbox_venv, use_scorer_sandbox
from .config import VALID_TASKS, WORK, Settings
from .inputs import Inputs
from .preflight import gold_patches, verify
from .server import make_sandbox_offline, set_scorer_env
from .util import log

OUT = WORK / "necessity"
WS = pathlib.Path("/tmp/a3a4_ws")
PAYLOAD = WORK / "a3a4_payload.tgz"
IN = WORK / "a3a4_in"


# ------------------------------------------------------------------ gold function groups (exec-graph rule)
def split_patch(patch):
    """-> [(file_header_lines, [(old_start, old_len, hunk_lines)])] per file"""
    files, cur, hunk = [], None, None
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git") or (line.startswith("--- ") and (cur is None or cur[1])):
            cur = [[line], []]
            files.append(cur)
            hunk = None
        elif cur is None:
            continue
        elif line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))?", line)
            hunk = [int(m.group(1)), int(m.group(2) or 1), [line]]
            cur[1].append(hunk)
        elif hunk is None:
            cur[0].append(line)
        else:
            hunk[2].append(line)
    return files


def path_of(header):
    for line in header:
        if line.startswith("--- a/"):
            return line[6:].strip()
    m = re.match(r"diff --git a/(\S+)", header[0])
    return m.group(1) if m else None


def func_spans(src):
    out = []
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return out

    def walk(node, stack):
        for c in ast.iter_child_nodes(node):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append((c.lineno, c.end_lineno or c.lineno, ".".join(stack + [c.name])))
                walk(c, stack + [c.name, "<locals>"])
            elif isinstance(c, ast.ClassDef):
                walk(c, stack + [c.name])
            else:
                walk(c, stack)
    walk(tree, [])
    return out


def hunk_functions(path, root, hunk):
    f = root / path if path else None
    spans = func_spans(f.read_text(errors="replace")) if f is not None and f.exists() else []
    old, changed = hunk[0], []
    for line in hunk[2][1:]:
        if line.startswith("-"):
            changed.append(old)
            old += 1
        elif line.startswith("+"):
            changed.append(max(old - 1, 1))
        elif not line.startswith("\\"):
            old += 1
    fns = set()
    for ln in changed:
        inside = [(e - s, q) for s, e, q in spans if s <= ln <= e]
        fns.add(f"{path}::{min(inside)[1]}" if inside else f"{path}::<module>")
    return fns


def gold_groups(patch, root):
    """-> files, {group key: [(file_idx, hunk_idx)]}, sorted function-group keys (no <module>)"""
    files = split_patch(patch)
    groups = {}
    for fi, (hdr, hunks) in enumerate(files):
        p = path_of(hdr)
        for hi, h in enumerate(hunks):
            groups.setdefault(" | ".join(sorted(hunk_functions(p, root, h))), []).append((fi, hi))
    return files, groups, sorted(k for k in groups if "<module>" not in k)


def assemble(files, keep):
    out = []
    for fi, (hdr, hunks) in enumerate(files):
        kept = [h for hi, h in enumerate(hunks) if (fi, hi) in keep]
        if kept:
            out.append("".join(hdr) + "".join("".join(h[2]) for h in kept))
    return "".join(out)


def touched_functions(patch, root):
    out = set()
    for hdr, hunks in split_patch(patch):
        p = path_of(hdr)
        for h in hunks:
            out |= {k for k in hunk_functions(p, root, h) if "<module>" not in k}
    return out


def covers(group_key, funcs):
    for g in group_key.split(" | "):
        for f in funcs:
            if f == g or f.startswith(g + ".") or g.startswith(f + "."):
                return True
    return False


# ------------------------------------------------------------------ base workspaces + patch combination
def materialise(inp, t):
    dst = WS / t["instance_id"]
    if dst.exists():
        return dst
    import swegemma.sandbox.subprocess as sp
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.harness.container_setup import extract_snapshot
    mgr = sp.SubprocessManager(timeout_seconds=900)
    snap, base, patch = resolve_task_snapshot_paths(inp.snapshots, t["instance_id"], t["repo"])
    sid = mgr.start()
    try:
        extract_snapshot(mgr, sid, snap, patch_path=patch, base_snapshot_path=base)
        ws = pathlib.Path(mgr._sandboxes[sid]["workspace"])
        shutil.copytree(ws, dst, ignore=shutil.ignore_patterns(".git"), symlinks=True)
    finally:
        mgr.stop(sid)
    return dst


def git(cwd, *args, inp=None):
    return subprocess.run(["git", "-c", "user.email=a@a", "-c", "user.name=a", *args], cwd=cwd, input=inp,
                          capture_output=True, text=True)


def combine(root, patches):
    """Apply patches in order to a copy of the touched base files; -> (combined diff, how) or (None, why)."""
    text = "".join(patches)
    paths = set(re.findall(r"^(?:---|\+\+\+) [ab]/(\S+)", text, re.M))
    tmp = pathlib.Path(tempfile.mkdtemp())
    try:
        for p in paths:
            src = root / p
            if src.is_file():
                (tmp / p).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, tmp / p)
        git(tmp, "init", "-q")
        git(tmp, "add", "-A")
        git(tmp, "commit", "-qm", "base", "--allow-empty")
        how = []
        for pt in patches:
            if not pt.strip():
                continue
            pt = pt if pt.endswith("\n") else pt + "\n"
            r = git(tmp, "apply", "--whitespace=nowarn", "-", inp=pt)
            if r.returncode:
                r = subprocess.run(["patch", "-p1", "-F3", "--no-backup-if-mismatch", "-s"], cwd=tmp, input=pt,
                                   capture_output=True, text=True)
                if r.returncode:
                    return None, "apply failed"
                how.append("patch-fuzz")
            else:
                how.append("git-apply")
        git(tmp, "add", "-A")
        d = git(tmp, "diff", "--cached", "HEAD")
        return d.stdout, "+".join(how)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ main
def runs_from_payload():
    with tarfile.open(PAYLOAD) as tf:
        tf.extractall(IN, filter="data")
    runs = {}
    for pt in sorted(IN.rglob("per_task.jsonl")):
        s = str(pt)
        name = ("v14_new60" if "v14" in s else "v7_new60") if "new60" in s else \
            "v7_old54_" + re.search(r"rep\d", s).group(0)
        pdir = pt.parent.parent / "merged" / "patches"
        rows = [json.loads(l) for l in open(pt)]
        runs[name] = {r["instance_id"]: (r, (pdir / f"{r['instance_id']}.patch").read_text(errors="replace")
                                         if (pdir / f"{r['instance_id']}.patch").exists() else "") for r in rows}
    old_loo = {}
    if (IN / "xg" / "results.jsonl").exists():
        meta = {json.loads(l)["id"]: json.loads(l) for l in open(IN / "xg" / "tasks_meta.jsonl")}
        res = [json.loads(l) for l in open(IN / "xg" / "results.jsonl")]
        for iid, m in meta.items():
            g = m.get("gold_function_groups") or []
            loo = {r["mode"]: r["resolved"] for r in res if r["id"] == iid and r["mode"].startswith("loo")}
            old_loo[iid] = {k: (loo.get(f"loo{i}") is False) for i, k in enumerate(g)} if len(g) >= 2 else \
                {k: True for k in g}
    return runs, old_loo


def main():
    t0 = time.time()
    s = Settings.load()
    set_scorer_env()
    patch_sandbox_venv()
    inp = Inputs.locate(False, need_model=False)
    use_scorer_sandbox(inp.data)
    make_sandbox_offline()
    OUT.mkdir(parents=True, exist_ok=True)
    gold = gold_patches(inp)
    tasks = {t["instance_id"]: t for t in inp.load_tasks()}
    valid = set(json.loads(VALID_TASKS.read_text()))
    runs, old_loo = runs_from_payload()
    new60 = sorted({i for r in ("v7_new60", "v14_new60") for i in runs.get(r, {})} & valid)
    if s.max_tasks:  # smoke run: a few new-60 tasks and only the agent runs on them
        new60 = new60[: s.max_tasks]
        runs = {k: {i: v for i, v in r.items() if i in new60} for k, r in runs.items()}
    involved = sorted(({i for r in runs.values() for i in r} | set(new60)) & valid)
    log(f"runs {sorted(runs)}; new-60 valid {len(new60)}; tasks involved {len(involved)}")

    groups = {}
    for iid in involved:
        try:
            root = materialise(inp, tasks[iid])
            files, gmap, fg = gold_groups(gold[iid], root)
            groups[iid] = (root, files, gmap, fg)
        except Exception as e:  # noqa: BLE001
            log("materialise/groups failed", iid, type(e).__name__, str(e)[:200])
    (OUT / "groups.json").write_text(json.dumps({i: v[3] for i, v in groups.items()}, indent=1))
    log(f"groups built for {len(groups)} tasks in {(time.time() - t0) / 60:.1f} min")

    rows = []

    def run(label, patches, extra):
        if not patches:
            return {}
        if s.hours_left() < 0.3:
            log("budget: skipping", label)
            return {}
        t = time.time()
        r = verify(inp, patches, label)
        for iid, v in r.items():
            rows.append({"task": iid, "mode": label, **extra.get(iid, {}), "resolved": v["resolved"], "exit": v["exit"],
                         "error": v["error"]})
        with open(OUT / "results.jsonl", "w") as f:
            f.write("".join(json.dumps(x) + "\n" for x in rows))
        log(f"{label}: {sum(v['resolved'] for v in r.values())}/{len(r)} resolved in {(time.time() - t) / 60:.1f} min")
        return r

    # ---------------- A3: necessity on new-60
    multi = [i for i in new60 if i in groups and len(groups[i][3]) >= 2]
    run("gold", {i: gold[i] for i in multi}, {})
    run("empty", {i: "" for i in multi}, {})
    nec = {}
    for k in range(max((len(groups[i][3]) for i in multi), default=0)):
        pats = {}
        for i in multi:
            _, files, gmap, fg = groups[i]
            if k < len(fg):
                drop = set(gmap[fg[k]])
                pats[i] = assemble(files, {(fi, hi) for hunks in gmap.values() for fi, hi in hunks} - drop)
        r = run(f"loo{k}", pats, {})
        for i, v in r.items():
            nec.setdefault(i, {})[groups[i][3][k]] = not v["resolved"]
    for i in new60:
        if i in groups and len(groups[i][3]) == 1:
            nec[i] = {groups[i][3][0]: True}
    (OUT / "necessity_new60.json").write_text(json.dumps(nec, indent=1))

    # ---------------- A4: completion counterfactual
    cases = []
    for rname, tasks_r in runs.items():
        for iid, (r, apatch) in tasks_r.items():
            if iid not in groups or r.get("resolved") or not apatch.strip():
                continue
            root, files, gmap, fg = groups[iid]
            if len(fg) < 1:
                continue
            tf = touched_functions(apatch, root)
            covered = [g for g in fg if covers(g, tf)]
            missed = [g for g in fg if g not in covered]
            if not covered or not missed:
                continue
            module = {h for k, hs in gmap.items() if "<module>" in k for h in hs}
            keep = {h for g in missed for h in gmap[g]} | module
            missed_patch = assemble(files, keep)
            comp, how = combine(root, [apatch, missed_patch])
            if comp is None:  # retry without the module-level hunks
                missed_patch = assemble(files, {h for g in missed for h in gmap[g]})
                comp, how = combine(root, [apatch, missed_patch])
                how = (how or "") + "|no-module"
            nlab = nec.get(iid) or old_loo.get(iid) or {}
            cases.append({"run": rname, "task": iid, "covered": covered, "missed": missed,
                          "missed_necessary": [g for g in missed if nlab.get(g)], "necessity_known": bool(nlab),
                          "apply": how if comp is not None else "apply failed", "_agent": apatch,
                          "_comp": comp, "_missed": missed_patch})
    log(f"A4 cases: {len(cases)} ({sum(c['_comp'] is not None for c in cases)} combinable)")
    for rname in sorted({c["run"] for c in cases}):
        cs = [c for c in cases if c["run"] == rname]
        ex = {c["task"]: {"run": rname} for c in cs}
        run(f"agent:{rname}", {c["task"]: c["_agent"] for c in cs}, ex)
        run(f"comp:{rname}", {c["task"]: c["_comp"] for c in cs if c["_comp"] is not None}, ex)
        run(f"missed:{rname}", {c["task"]: c["_missed"] for c in cs}, ex)
    (OUT / "cases.json").write_text(json.dumps([{k: v for k, v in c.items() if not k.startswith("_")} for c in cases],
                                               indent=1))

    # ---------------- report (pre-registered outcomes)
    res = {(x["mode"], x["task"]): x["resolved"] for x in rows}
    for c in cases:
        c["agent_ok"] = res.get((f"agent:{c['run']}", c["task"]))
        c["comp_ok"] = res.get((f"comp:{c['run']}", c["task"]))
        c["missed_ok"] = res.get((f"missed:{c['run']}", c["task"]))
        c["pure"] = bool(c["comp_ok"]) and c["missed_ok"] is False

    def summ(cs):
        ok = [c for c in cs if c["comp_ok"] is not None and c["missed_ok"] is not None]
        n = len(ok)
        return {"cases": len(cs), "evaluated": n, "apply_failed": sum(c["_comp"] is None for c in cs),
                "agent_alone_resolved": sum(bool(c["agent_ok"]) for c in cs),
                "completion_resolved": sum(bool(c["comp_ok"]) for c in ok),
                "missed_only_resolved": sum(bool(c["missed_ok"]) for c in ok),
                "pure_incompleteness": sum(c["pure"] for c in ok),
                "completion_rate": round(sum(bool(c["comp_ok"]) for c in ok) / n, 3) if n else None,
                "pure_rate": round(sum(c["pure"] for c in ok) / n, 3) if n else None}

    multi_nec = sorted(i for i, d in nec.items() if sum(d.values()) >= 2)
    report = {
        "A3": {"new60_valid": len(new60), "multi_group": len(multi), "with_necessity_labels": len(nec),
               "tasks_with_>=2_necessary_groups": len(multi_nec), "ids": multi_nec,
               "gold_resolved": sum(bool(res.get(("gold", i))) for i in multi),
               "empty_resolved": sum(bool(res.get(("empty", i))) for i in multi)},
        "A4": {"pooled": summ(cases),
               "missed_a_necessary_group": summ([c for c in cases if c["missed_necessary"]]),
               **{r: summ([c for c in cases if c["run"] == r]) for r in sorted({c["run"] for c in cases})}},
        "hours": round((time.time() - t0) / 3600, 2)}
    (WORK / "necessity_report.json").write_text(json.dumps(report, indent=1))
    (OUT / "cases.json").write_text(json.dumps([{k: v for k, v in c.items() if not k.startswith("_")} for c in cases],
                                               indent=1))
    log("A3:", json.dumps(report["A3"]))
    log("A4 pooled:", json.dumps(report["A4"]["pooled"]))
    log("A4 missed a necessary group:", json.dumps(report["A4"]["missed_a_necessary_group"]))


if __name__ == "__main__":
    main()


# ================================================================== A12 site-swap (+ A5 typology), pre-registered Oct 24
# For every unresolved agent patch that edits >= 1 gold function group g: start from the gold-patched tree and replace
# g's function(s) with the agent-patched version (AST source segment, function level). Classify before testing:
# identical / conflict / coupled / clean; then run the fail-to-pass tests on clean + coupled swaps.
SWAP_OUT = WORK / "siteswap"
NORM_WS = re.compile(r"\s+")


def patched_texts(root, patches):
    """{path: text or None (deleted)} for every file the patches touch, after applying them to the base; or None."""
    text = "".join(patches)
    paths = set(re.findall(r"^(?:---|\+\+\+) [ab]/(\S+)", text, re.M))
    tmp = pathlib.Path(tempfile.mkdtemp())
    try:
        for p in paths:
            src = root / p
            if src.is_file():
                (tmp / p).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, tmp / p)
        git(tmp, "init", "-q")
        for pt in patches:
            if not pt.strip():
                continue
            pt = pt if pt.endswith("\n") else pt + "\n"
            if git(tmp, "apply", "--whitespace=nowarn", "-", inp=pt).returncode and subprocess.run(
                    ["patch", "-p1", "-F3", "--no-backup-if-mismatch", "-s"], cwd=tmp, input=pt,
                    capture_output=True, text=True).returncode:
                return None
        return {p: ((tmp / p).read_text(errors="replace") if (tmp / p).is_file() else None) for p in paths}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def diff_from_texts(root, texts):
    """Unified diff base -> texts (path -> new text or None)."""
    tmp = pathlib.Path(tempfile.mkdtemp())
    try:
        for p in texts:
            if (root / p).is_file():
                (tmp / p).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / p, tmp / p)
        git(tmp, "init", "-q")
        git(tmp, "add", "-A")
        git(tmp, "commit", "-qm", "base", "--allow-empty")
        for p, t in texts.items():
            f = tmp / p
            if t is None:
                f.unlink(missing_ok=True)
            else:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(t)
        git(tmp, "add", "-A")
        return git(tmp, "diff", "--cached", "HEAD").stdout
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def span_of(text, qual):
    for s_, e_, q in func_spans(text or ""):
        if q == qual:
            return s_, e_
    return None


def defined_names(text, skip=()):
    """Names a file defines OUTSIDE the line spans in `skip`: module- and class-level defs, imports and assignments,
    plus attributes assigned on self/cls anywhere (a field set in __init__ couples the methods that read it)."""
    out = set()
    try:
        tree = ast.parse(text or "")
    except (SyntaxError, ValueError):
        return out

    def inside(n):
        return any(a <= getattr(n, "lineno", -1) <= b for a, b in skip)

    def scope(body):
        for n in body:
            if inside(n):
                continue
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(n.name)
                if isinstance(n, ast.ClassDef):
                    scope(n.body)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                out.update((a.asname or a.name).split(".")[0] for a in n.names)
            elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    out.update(x.id for x in ast.walk(t) if isinstance(x, ast.Name))
    scope(tree.body)
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name) \
                and n.value.id in ("self", "cls") and not inside(n):
            out.add(n.attr)
    return out


def referenced_names(segment):
    import textwrap
    try:
        tree = ast.parse(textwrap.dedent(segment))
    except (SyntaxError, ValueError):
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.add(n.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in ("self", "cls"):
            out.add(n.attr)
    return out


def build_swap(root, gold_t, agent_t, base_t, group_key):
    """-> (class, patch or None, detail)"""
    funcs = [f for f in group_key.split(" | ")]
    funcs = [f for f in funcs if not any(f != o and f.startswith(o + ".") for o in funcs)]  # outermost only
    texts = dict(gold_t)
    same = True
    segs = []
    by_path = {}
    for f in funcs:
        path, qual = f.split("::", 1)
        by_path.setdefault(path, []).append(qual)
    for path, quals in by_path.items():
        g_text, a_text = gold_t.get(path), agent_t.get(path)
        if g_text is None or a_text is None:
            return "conflict", None, f"file missing in {'gold' if g_text is None else 'agent'} tree: {path}"
        g_lines, a_lines = g_text.splitlines(keepends=True), a_text.splitlines(keepends=True)
        repl = []
        for q in quals:
            gs, asp = span_of(g_text, q), span_of(a_text, q)
            if not gs or not asp:
                return "conflict", None, f"function {q} missing in {'gold' if not gs else 'agent'} tree"
            gseg, aseg = "".join(g_lines[gs[0] - 1:gs[1]]), "".join(a_lines[asp[0] - 1:asp[1]])
            same &= NORM_WS.sub("", gseg) == NORM_WS.sub("", aseg)
            repl.append((gs, a_lines[asp[0] - 1:asp[1]]))
            segs.append(aseg)
        for (s0, e0), new in sorted(repl, key=lambda x: -x[0][0]):
            g_lines[s0 - 1:e0] = new
        texts[path] = "".join(g_lines)
        try:
            compile(texts[path], path, "exec")
        except SyntaxError as e:
            return "conflict", None, f"py_compile: {e.msg} line {e.lineno}"
    if same:
        return "identical", None, ""
    agent_new = set()
    for p, t in agent_t.items():  # defined by the agent's edits OUTSIDE g
        skip = [span_of(t, q) for q in by_path.get(p, [])]
        agent_new |= defined_names(t, [x for x in skip if x]) - defined_names(base_t.get(p))
    gold_def = set().union(*(defined_names(t) for t in gold_t.values() if t)) if gold_t else set()
    refs = set().union(*(referenced_names(s_) for s_ in segs)) if segs else set()
    missing = sorted((refs & agent_new) - gold_def)
    patch = diff_from_texts(root, texts)
    return ("coupled" if missing else "clean"), patch, ",".join(missing[:5])


def a5_typology(groups, nec, gold, payload_dir):
    """Descriptive: oblig.py v2 relations between test-necessary groups of new-60 tasks with >= 2 of them."""
    import sys
    sys.path.insert(0, str(payload_dir / "tools"))
    import oblig
    rows, diffs = [], []
    for iid, labs in sorted(nec.items()):
        necs = [g for g, v in labs.items() if v]
        if len(necs) < 2 or iid not in groups:
            continue
        root, files, gmap, _ = groups[iid]
        idx = oblig.Index(str(root)).build()

        def keys(g):
            return [f.replace(".<locals>", "") for f in g.split(" | ")]

        def edited_idents(g):
            toks = set()
            for fi, hi in gmap.get(g, []):
                for line in files[fi][1][hi][2][1:]:
                    if line[:1] in "+-":
                        toks |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", line[1:]))
            return toks
        for a in necs:
            for b in necs:
                if a == b:
                    continue
                ka, kb = keys(a), keys(b)
                direct = sorted({t for x in ka if x in idx.defs for y, t, _ in idx.neighbours(x)
                                 if any(y == k or y.startswith(k + ".") or k.startswith(y + ".") for k in kb)})
                two = False
                for x in ka:
                    if x not in idx.defs:
                        continue
                    for y, _, _ in idx.neighbours(x):
                        for z, _, _ in idx.neighbours(y):
                            if any(z == k or z.startswith(k + ".") for k in kb):
                                two = True
                ca = {idx.defs[x].cls for x in ka if x in idx.defs}
                cb = {idx.defs[x].cls for x in kb if x in idx.defs}
                rows.append({"task": iid, "a": a, "b": b, "direct_edge_types": direct, "two_hop": two,
                             "same_file": ka[0].split("::")[0] == kb[0].split("::")[0],
                             "same_class": bool(ca & cb - {None}),
                             "shared_edited_identifiers": sorted(edited_idents(a) & edited_idents(b))[:8]})
        diffs.append(f"## {iid}\nnecessary groups: {necs}\n")
        for g in necs:
            body = "".join("".join(files[fi][1][hi][2]) for fi, hi in gmap.get(g, []))
            diffs.append(f"### {g}\n```diff\n{''.join(body.splitlines(keepends=True)[:40])}```\n")
    return rows, "\n".join(diffs)


def main_siteswap():
    import random as _r
    t0 = time.time()
    s = Settings.load()
    set_scorer_env()
    patch_sandbox_venv()
    inp = Inputs.locate(False, need_model=False)
    use_scorer_sandbox(inp.data)
    make_sandbox_offline()
    SWAP_OUT.mkdir(parents=True, exist_ok=True)
    gold = gold_patches(inp)
    tasks = {t["instance_id"]: t for t in inp.load_tasks()}
    valid = set(json.loads(VALID_TASKS.read_text()))
    runs, old_loo = runs_from_payload()
    nec_new = json.loads((IN / "a3" / "necessity_new60.json").read_text())
    cand = sorted({i for r in runs.values() for i, (row, p) in r.items()
                   if not row.get("resolved") and p.strip()} & valid)
    if s.max_tasks:
        cand = cand[: s.max_tasks]
    groups = {}
    for iid in sorted(set(cand) | {i for i, d in nec_new.items() if sum(d.values()) >= 2} & valid):
        try:
            root = materialise(inp, tasks[iid])
            groups[iid] = (root,) + gold_groups(gold[iid], root)
        except Exception as e:  # noqa: BLE001
            log("materialise/groups failed", iid, type(e).__name__, str(e)[:200])
    log(f"site-swap: {len(cand)} candidate tasks, groups for {len(groups)} in {(time.time() - t0) / 60:.1f} min")

    def necessary(iid, g, fg):
        lab = nec_new.get(iid) or old_loo.get(iid)
        if lab and g in lab:
            return bool(lab[g])
        return True if len(fg) == 1 else None

    swaps, gold_cache = [], {}
    for rname, rr in runs.items():
        for iid, (row, apatch) in rr.items():
            if iid not in cand or iid not in groups or row.get("resolved") or not apatch.strip():
                continue
            root, files, gmap, fg = groups[iid]
            covered = [g for g in fg if covers(g, touched_functions(apatch, root))]
            if not covered:
                continue
            if iid not in gold_cache:
                gold_cache[iid] = patched_texts(root, [gold[iid]])
            agent_t = patched_texts(root, [apatch])
            gold_t = gold_cache[iid]
            for g in covered:
                if gold_t is None or agent_t is None:
                    cls_, patch, det = "conflict", None, "patch does not apply"
                else:
                    paths = {f.split("::")[0] for f in g.split(" | ")} | set(agent_t)
                    base_t = {p: ((root / p).read_text(errors="replace") if (root / p).is_file() else None) for p in paths}
                    full_gold = {**{p: base_t.get(p) for p in paths}, **gold_t}
                    full_agent = {**{p: base_t.get(p) for p in paths}, **agent_t}
                    try:
                        cls_, patch, det = build_swap(root, full_gold, full_agent, base_t, g)
                    except Exception as e:  # noqa: BLE001
                        cls_, patch, det = "conflict", None, f"{type(e).__name__}: {str(e)[:120]}"
                swaps.append({"run": rname, "task": iid, "group": g, "necessary": necessary(iid, g, fg),
                              "class": cls_, "detail": det, "_patch": patch})
    log(f"swaps: {len(swaps)}; classes {dict((c, sum(x['class'] == c for x in swaps)) for c in ('identical', 'conflict', 'coupled', 'clean'))}")

    rows = []

    def run(label, patches, meta):
        if not patches or s.hours_left() < 0.3:
            return {}
        t = time.time()
        r = verify(inp, patches, label)
        for iid, v in r.items():
            rows.append({"task": iid, "mode": label, **meta.get(iid, {}), "resolved": v["resolved"], "exit": v["exit"],
                         "error": v["error"]})
        (SWAP_OUT / "results.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        log(f"{label}: {sum(v['resolved'] for v in r.values())}/{len(r)} resolved in {(time.time() - t) / 60:.1f} min")
        return r

    gold_ok = run("gold_rerun", {i: gold[i] for i in sorted({x["task"] for x in swaps})}, {})
    todo = [x for x in swaps if x["class"] in ("clean", "coupled") and gold_ok.get(x["task"], {}).get("resolved")]
    k = 0
    while todo:  # one patch per task per Evaluator call
        batch, rest = {}, []
        for x in todo:
            if x["task"] in batch:
                rest.append(x)
            else:
                batch[x["task"]] = x
        r = run(f"swap{k}", {t: x["_patch"] for t, x in batch.items()}, {})
        for t, x in batch.items():
            x["resolved"] = r.get(t, {}).get("resolved")
        if not r:
            break
        todo, k = rest, k + 1

    for x in swaps:
        x["gold_ok"] = bool(gold_ok.get(x["task"], {}).get("resolved"))
        x["wrong"] = (x.get("resolved") is False) if x["class"] in ("clean", "coupled") else None
    (SWAP_OUT / "swaps.json").write_text(json.dumps([{k_: v for k_, v in x.items() if k_ != "_patch"} for x in swaps],
                                                    indent=1))

    def rate(xs):
        xs = [x for x in xs if x["wrong"] is not None and x["gold_ok"]]
        return (sum(x["wrong"] for x in xs), len(xs))

    prim = [x for x in swaps if x["class"] == "clean" and x["necessary"]]
    w, n = rate(prim)
    by_task = {}
    for x in prim:
        if x["wrong"] is not None and x["gold_ok"]:
            by_task.setdefault(x["task"], []).append(x["wrong"])
    boots = []
    ts = list(by_task)
    rnd = _r.Random(0)
    for _ in range(10000):
        smp = [rnd.choice(ts) for _ in ts] if ts else []
        vals = [v for t in smp for v in by_task[t]]
        boots.append(sum(vals) / len(vals) if vals else 0)
    boots.sort()
    cls_counts = {c: sum(x["class"] == c for x in swaps) for c in ("identical", "conflict", "coupled", "clean")}
    report = {
        "swaps": len(swaps), "classes": cls_counts,
        "conflict_plus_coupled_share": round((cls_counts["conflict"] + cls_counts["coupled"]) / max(1, len(swaps)), 3),
        "primary_clean_necessary": {"wrong": w, "n": n, "rate": round(w / n, 3) if n else None,
                                    "ci95_task_bootstrap": [round(boots[250], 3), round(boots[9750], 3)] if ts else None},
        "task_level": {"tasks": len(by_task), "tasks_with_a_wrong_clean_necessary_swap": sum(any(v) for v in by_task.values())},
        "all_clean": dict(zip(("wrong", "n"), rate([x for x in swaps if x["class"] == "clean"]))),
        "coupled": dict(zip(("wrong", "n"), rate([x for x in swaps if x["class"] == "coupled"]))),
        "gold_rerun_failed_tasks": sorted(t for t, v in gold_ok.items() if not v.get("resolved")),
    }
    try:
        rows5, diffs = a5_typology({i: g for i, g in groups.items()}, nec_new, gold, IN)
        (SWAP_OUT / "a5_pairs.json").write_text(json.dumps(rows5, indent=1))
        (SWAP_OUT / "a5_diffs.md").write_text(diffs)
        report["A5"] = {"tasks": len({r_["task"] for r_ in rows5}), "pairs": len(rows5),
                        "same_file": sum(r_["same_file"] for r_ in rows5), "same_class": sum(r_["same_class"] for r_ in rows5),
                        "direct_any_edge": sum(bool(r_["direct_edge_types"]) for r_ in rows5),
                        "direct_call_edge": sum(any(t in ("CALLER", "CALLEE", "FWD", "FWDDOWN", "INST") for t in r_["direct_edge_types"]) for r_ in rows5),
                        "two_hop": sum(r_["two_hop"] for r_ in rows5),
                        "shared_edited_identifier": sum(bool(r_["shared_edited_identifiers"]) for r_ in rows5)}
        per = {}
        for r_ in rows5:
            per.setdefault(r_["task"], []).append(r_)
        macro = {}
        for key in ("same_file", "same_class", "two_hop"):
            macro[key] = round(sum(sum(bool(r_[key]) for r_ in v) / len(v) for v in per.values()) / len(per), 3)
        macro["direct_any_edge"] = round(sum(sum(bool(r_["direct_edge_types"]) for r_ in v) / len(v) for v in per.values()) / len(per), 3)
        macro["shared_edited_identifier"] = round(sum(sum(bool(r_["shared_edited_identifiers"]) for r_ in v) / len(v) for v in per.values()) / len(per), 3)
        report["A5"]["task_macro"] = macro
        report["A5"]["pairs_per_task"] = {t: len(v) for t, v in sorted(per.items(), key=lambda kv: -len(kv[1]))}
    except Exception as e:  # noqa: BLE001
        report["A5_error"] = f"{type(e).__name__}: {e}"
    report["hours"] = round((time.time() - t0) / 3600, 2)
    (WORK / "siteswap_report.json").write_text(json.dumps(report, indent=1))
    log("A12:", json.dumps({k_: report[k_] for k_ in ("classes", "conflict_plus_coupled_share", "primary_clean_necessary", "task_level")}))
    log("A5:", json.dumps(report.get("A5") or report.get("A5_error")))
