"""
Task factory v0 core (SWE-smith-lite): verifiable bug-fix tasks from pure-Python libraries the model has not been
evaluated on. Pure Python + git + pytest, no harness: importable on the Mac for a smoke test (smoke_local.py), and
concatenated into the Kaggle kernel by build.sh.

Per library: fetch a pinned release from GitHub -> baseline test outcomes -> AST mutations inside one function
(single) or two functions (pair) -> keep a mutation when 1..MAX_F2P baseline-passing tests fail and every other test
in those files still passes -> snapshot = mutated repo with the failing tests REMOVED (as with real hidden tests);
gold patch = revert of the mutation; test_patch = restore of the removed tests; issue = template from the failure.
"""
import ast
import copy
import hashlib
import io
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request

LIBS = [  # name, GitHub repo, pinned tag, package dir (relative to repo root), test dir; first ones are preferred
    {"name": "more-itertools", "gh": "more-itertools/more-itertools", "tag": "v11.2.1", "pkg": "more_itertools",
     "tests": "tests"},
    {"name": "sqlparse", "gh": "andialbrecht/sqlparse", "tag": "0.6.0", "pkg": "sqlparse", "tests": "tests"},
    {"name": "boltons", "gh": "mahmoud/boltons", "tag": "26.2.0", "pkg": "boltons", "tests": "tests"},
    {"name": "tabulate", "gh": "astanin/python-tabulate", "tag": "v0.10.0", "pkg": "tabulate", "tests": "test"},
    {"name": "inflection", "gh": "jpvanhal/inflection", "tag": "0.5.1", "pkg": "inflection", "tests": "."},
]
MAX_F2P = 10          # more failing tests than this = a global break, not an issue-sized bug
MAX_TEST_FILES = 2
PER_FUNCTION = 2      # kept singles per function (diversity)
GIT_ENV = dict(GIT_AUTHOR_NAME="dev", GIT_AUTHOR_EMAIL="dev@example.com", GIT_COMMITTER_NAME="dev",
               GIT_COMMITTER_EMAIL="dev@example.com", GIT_AUTHOR_DATE="2026-01-01T00:00:00",
               GIT_COMMITTER_DATE="2026-01-01T00:00:00")

PLUGIN = '''
import json, os
_R, _M = {}, {}
def pytest_collectreport(report):
    if report.failed:
        _R["<collect>::" + report.nodeid] = "error"
def pytest_runtest_logreport(report):
    if report.when == "call" or report.outcome != "passed":
        prev = _R.get(report.nodeid)
        out = "error" if report.when != "call" and report.outcome == "failed" else report.outcome
        if prev in (None, "passed"):
            _R[report.nodeid] = out
        if report.failed:
            _M[report.nodeid] = (report.longreprtext or "")[-2500:]
def pytest_sessionfinish(session, exitstatus):
    json.dump({"outcomes": _R, "messages": _M}, open(os.environ["FACTORY_OUT"], "w"))
'''


def sh(cmd, cwd=None, env=None, timeout=600):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def git_snapshot(root, msg="snapshot"):
    shutil.rmtree(root / ".git", ignore_errors=True)
    env = dict(os.environ, **GIT_ENV)
    for cmd in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"], ["git", "commit", "-q", "-m", msg]):
        sh(cmd, cwd=root, env=env)
    return sh(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()


# ------------------------------------------------------------------ fetch + baseline
def fetch(lib, dest):
    """Release tarball from GitHub -> dest (repo root, no .git)."""
    url = f"https://github.com/{lib['gh']}/archive/refs/tags/{lib['tag']}.tar.gz"
    data = urllib.request.urlopen(url, timeout=120).read()
    tmp = dest.parent / (dest.name + "_x")
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(tmp, filter="data")
    (top,) = list(tmp.iterdir())
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(str(top), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)
    for junk in (".github", "docs", "doc"):  # not needed by the tasks; keeps snapshots small
        shutil.rmtree(dest / junk, ignore_errors=True)
    return dest


def import_root(repo, lib):
    """Directory to put on sys.path so `import <pkg>` resolves to the repo copy (src/ layout or flat)."""
    for base in (repo / "src", repo):
        if (base / lib["pkg"]).is_dir() or (base / f"{lib['pkg']}.py").exists():
            return base
    raise FileNotFoundError(f"package {lib['pkg']} not found in {repo}")


def run_tests(repo, lib, plugin_dir, out_json, targets=None, timeout=900):
    """-> {"outcomes": {nodeid: passed|failed|error|skipped}, "messages": {nodeid: text}} (None on timeout)."""
    env = dict(os.environ, FACTORY_OUT=str(out_json), PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(import_root(repo, lib)), str(plugin_dir)]))
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "factory_plugin", "-p", "no:cacheprovider", "-o", "addopts=",
           "--timeout=30"] + (targets or [lib["tests"]])
    if out_json.exists():
        out_json.unlink()
    try:
        sh(cmd, cwd=repo, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return json.loads(out_json.read_text()) if out_json.exists() else None


def write_plugin(plugin_dir):
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "factory_plugin.py").write_text(PLUGIN)


def source_files(repo, lib):
    root = import_root(repo, lib)
    single = root / f"{lib['pkg']}.py"
    files = [single] if single.exists() else sorted((root / lib["pkg"]).rglob("*.py"))
    return [f for f in files if "test" not in f.name.lower()]


# ------------------------------------------------------------------ mutations (one site inside one function)
SWAP_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
SWAP_EXC = {"ValueError": "TypeError", "TypeError": "ValueError", "KeyError": "LookupError",
            "IndexError": "ValueError", "AttributeError": "TypeError", "RuntimeError": "ValueError"}


def functions(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def qualname(tree, fn):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and fn in node.body:
            return f"{node.name}.{fn.name}"
    return fn.name


def one_line(node):
    return getattr(node, "lineno", None) and node.lineno == node.end_lineno


def candidate_mutations(text):
    """-> [(op, func_qualname, (lineno, col, end_col), replacement)] on single-line expression spans,
    plus ('drop_*', func, (lineno, end_lineno), None) statement deletions."""
    tree = ast.parse(text)
    out = []
    for fn in functions(tree):
        q = qualname(tree, fn)
        if fn.end_lineno - fn.lineno < 2:
            continue
        inner = [n for stmt in fn.body for n in ast.walk(stmt)]
        for n in inner:
            new, op = None, None
            if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in SWAP_CMP and one_line(n):
                new, op = copy.deepcopy(n), "cmp_swap"
                new.ops = [SWAP_CMP[type(n.ops[0])]()]
            elif isinstance(n, ast.BoolOp) and one_line(n):
                new, op = copy.deepcopy(n), "boolop_swap"
                new.op = ast.Or() if isinstance(n.op, ast.And) else ast.And()
            elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not) and one_line(n):
                new, op = n.operand, "drop_not"
            elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)) and one_line(n) and \
                    not any(isinstance(x, (ast.Constant, ast.JoinedStr)) and not isinstance(getattr(x, "value", 0), int)
                            for x in (n.left, n.right)):
                new, op = copy.deepcopy(n), "arith_swap"
                new.op = ast.Sub() if isinstance(n.op, ast.Add) else ast.Add()
            elif isinstance(n, ast.Constant) and type(n.value) is int and 0 <= n.value <= 2 and one_line(n):
                new, op = ast.Constant(n.value + 1 if n.value < 2 else n.value - 1), "const_shift"
            elif isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call) and isinstance(n.exc.func, ast.Name) and \
                    n.exc.func.id in SWAP_EXC and one_line(n.exc.func):
                n2 = n.exc.func
                out.append(("exc_swap", q, (n2.lineno, n2.col_offset, n2.end_col_offset), SWAP_EXC[n2.id]))
                continue
            elif isinstance(n, ast.Call) and len(n.args) == 2 and not n.keywords and one_line(n) and \
                    all(isinstance(a, ast.Name) for a in n.args) and n.args[0].id != n.args[1].id:
                new, op = copy.deepcopy(n), "arg_swap"
                new.args = new.args[::-1]
            if new is not None:
                out.append((op, q, (n.lineno, n.col_offset, n.end_col_offset), ast.unparse(new)))
        for parent in [fn] + [n for n in inner if hasattr(n, "body") and isinstance(getattr(n, "body"), list)]:
            body = parent.body
            for s in body:
                if len(body) < 2:
                    break
                if isinstance(s, ast.If) and len(s.body) == 1 and isinstance(s.body[0], (ast.Raise, ast.Return)) \
                        and not s.orelse:
                    out.append(("drop_guard", q, (s.lineno, s.end_lineno), None))
                elif isinstance(s, ast.AugAssign):
                    out.append(("drop_update", q, (s.lineno, s.end_lineno), None))
    return out


def apply_mutation(text, mut):
    op, _, span, repl = mut
    lines = text.splitlines(keepends=True)
    if repl is None:  # statement deletion (whole lines)
        a, b = span
        del lines[a - 1:b]
        new = "".join(lines)
    else:
        ln, c0, c1 = span
        line = lines[ln - 1]
        b = line.encode()  # ast columns are UTF-8 byte offsets
        line = (b[:c0] + repl.encode() + b[c1:]).decode()
        lines[ln - 1] = line
        new = "".join(lines)
    try:
        compile(new, "<mutant>", "exec")
    except SyntaxError:
        return None
    return new if new != text else None


# ------------------------------------------------------------------ test removal (the hidden tests)
def split_nodeid(nodeid):
    path, *rest = nodeid.split("::")
    name = rest[-1].split("[", 1)[0] if rest else ""
    cls = rest[0] if len(rest) == 2 else None
    return path, cls, name


def remove_tests(test_text, names):
    """Delete the test functions/methods named (cls, name) including decorators; -> new text or None."""
    tree = ast.parse(test_text)
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and (node.name, m.name) in names:
                    spans.append((node, m))
    for m in tree.body:
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and (None, m.name) in names:
            spans.append((tree, m))
    if not spans:
        return None
    lines = test_text.splitlines(keepends=True)
    removed = {}
    for parent, _ in spans:
        removed[id(parent)] = removed.get(id(parent), 0) + 1
    # a class whose every member is removed keeps a `pass` where its first member was
    for parent, m in sorted(spans, key=lambda x: -x[1].lineno):
        start = min([m.lineno] + [d.lineno for d in m.decorator_list])
        indent = re.match(r"\s*", lines[m.lineno - 1]).group(0)
        last = isinstance(parent, ast.ClassDef) and removed[id(parent)] == len(parent.body) and m is parent.body[0]
        lines[start - 1:m.end_lineno] = [indent + "pass\n"] if last else []
    new = "".join(lines)
    try:
        ast.parse(new)
    except SyntaxError:
        return None
    return new


# ------------------------------------------------------------------ issue text (template; a model rewrite comes later)
def public_api(test_src, lib_defs):
    calls = re.findall(r"(?:\b[A-Za-z_][A-Za-z0-9_]*\.)?\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", test_src)
    hits = [c for c in calls if c in lib_defs and not c.startswith("_") and not c.startswith("assert")]
    return max(set(hits), key=hits.count) if hits else None


def failure_line(msg):
    """Last informative line of a pytest failure: 'E   ...' lines, else the exception line."""
    es = [l[1:].strip() for l in msg.splitlines() if l.startswith("E ")]
    es = [e for e in es if e and not e.startswith(("+", "-", "?", "Full diff", "Use -v"))]
    text = " ".join(es[:3]) if es else (msg.strip().splitlines() or [""])[-1]
    return re.sub(r"/[^\s:]*/(?=[A-Za-z_]+\.py)", "", text)[:400]


def make_issue(lib, f2p, messages, test_sources, lib_defs):
    apis, obs = [], []
    for nid in f2p:
        _, cls, name = split_nodeid(nid)
        api = public_api(test_sources.get(nid, ""), lib_defs)
        if api and api not in apis:
            apis.append(api)
        line = failure_line(messages.get(nid, ""))
        if line and line not in obs:
            obs.append(line)
    subject = f"`{apis[0]}`" if apis else f"`{lib['pkg']}`"
    scen = ", ".join(sorted({split_nodeid(n)[2].removeprefix("test_").replace("_", " ") for n in f2p})[:3])
    body = [f"{subject} returns the wrong result" + (f" ({scen})" if scen else ""), "",
            f"After upgrading `{lib['pkg']}`, {subject} no longer behaves as documented"
            + (f" when used for: {scen}." if scen else "."), ""]
    if len(apis) > 1:
        body += ["Other affected functions: " + ", ".join(f"`{a}`" for a in apis[1:4]) + ".", ""]
    if obs:
        body += ["What I see:", "", "```"] + obs[:3] + ["```", ""]
    body += ["Expected: the documented behaviour (it worked in the previous release)."]
    return "\n".join(body)


def lib_def_names(repo, lib):
    names = set()
    for f in source_files(repo, lib):
        try:
            names |= {n.name for n in ast.walk(ast.parse(f.read_text())) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        except SyntaxError:
            pass
    return names


def test_source(repo, nodeid):
    path, cls, name = split_nodeid(nodeid)
    p = repo / path
    if not p.exists():
        return ""
    text = p.read_text()
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return ast.get_source_segment(text, n) or ""
    return ""


# ------------------------------------------------------------------ search
def evaluate_mutants(repo, lib, base, muts, work, plugin_dir, files_text, log=print):
    """Apply each mutant (list of (rel_file, new_text)) in `repo`, run the suite, restore. -> rows."""
    rows = []
    for i, (meta, edits) in enumerate(muts):
        for rel, new in edits:
            (repo / rel).write_text(new)
        res = run_tests(repo, lib, plugin_dir, work / "mut.json")
        for rel, _ in edits:
            (repo / rel).write_text(files_text[rel])
        if res is None:
            rows.append(dict(meta, status="timeout"))
            continue
        oc = res["outcomes"]
        if any(k.startswith("<collect>") for k in oc):
            rows.append(dict(meta, status="collect_error"))
            continue
        broke = sorted(k for k, v in oc.items() if v in ("failed", "error") and base.get(k) == "passed")
        files = sorted({split_nodeid(k)[0] for k in broke})
        status = "equivalent" if not broke else "too_many" if len(broke) > MAX_F2P else \
            "too_many_files" if len(files) > MAX_TEST_FILES else "ok"
        rows.append(dict(meta, status=status, f2p=broke, test_files=files,
                         messages={k: res["messages"].get(k, "") for k in broke}))
        if (i + 1) % 25 == 0:
            log(f"  {lib['name']}: {i + 1}/{len(muts)} mutants, ok so far {sum(r['status'] == 'ok' for r in rows)}")
    return rows


def file_clean(base, test_file):
    """Every baseline test in this file passes or skips (the verifier runs whole files and needs 0 failures)."""
    return all(v in ("passed", "skipped") for k, v in base.items() if split_nodeid(k)[0] == test_file)


def build_task(repo, lib, tid, edits, f2p, messages, base, files_text, out_snap_dir, lib_defs):
    """Snapshot with mutation applied and F2P tests removed; gold = revert; test_patch = restore. -> task row."""
    work = repo.parent / f"task_{tid}"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(repo, work, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    test_sources = {n: test_source(repo, n) for n in f2p}
    by_file = {}
    for n in f2p:
        path, cls, name = split_nodeid(n)
        by_file.setdefault(path, set()).add((cls, name))
    orig_tests = {}
    for path, names in by_file.items():
        txt = (work / path).read_text()
        new = remove_tests(txt, names)
        if new is None:
            shutil.rmtree(work, ignore_errors=True)
            return None
        orig_tests[path] = txt
        (work / path).write_text(new)
    for rel, new in edits:
        (work / rel).write_text(new)
    base_commit = git_snapshot(work)
    for rel, _ in edits:  # gold = revert of the mutation
        (work / rel).write_text(files_text[rel])
    gold = sh(["git", "diff"], cwd=work).stdout
    sh(["git", "checkout", "-q", "--", "."], cwd=work)
    for path, txt in orig_tests.items():
        (work / path).write_text(txt)
    test_patch = sh(["git", "diff"], cwd=work).stdout
    sh(["git", "checkout", "-q", "--", "."], cwd=work)
    snap = out_snap_dir / f"{tid}.tgz"
    sh(["tar", "-czf", str(snap), "-C", str(work), "."])
    shutil.rmtree(work, ignore_errors=True)
    p2p = sorted(k for k, v in base.items() if v == "passed" and split_nodeid(k)[0] in by_file and k not in f2p)
    return {"instance_id": tid, "repo": f"{lib['gh']}", "base_commit": base_commit,
            "problem_statement": make_issue(lib, f2p, messages, test_sources, lib_defs),
            "patch": gold, "test_patch": test_patch, "hints_text": "", "created_at": "2026-10-28T00:00:00Z",
            "FAIL_TO_PASS": json.dumps(f2p), "PASS_TO_PASS": json.dumps(p2p), "environment_setup_commit": base_commit}


def run_library(lib, root, out_snap_dir, n_try=300, n_pairs=15, seed=0, log=print):
    """Whole pipeline for one library -> (tasks, report)."""
    rng = random.Random(seed)
    repo = fetch(lib, root / lib["name"])
    plugin_dir = root / "_plugin"
    write_plugin(plugin_dir)
    base_res = run_tests(repo, lib, plugin_dir, root / f"{lib['name']}_base.json")
    rep = {"lib": lib["name"], "tag": lib["tag"]}
    if base_res is None:
        return [], dict(rep, error="baseline timeout")
    base = base_res["outcomes"]
    rep["baseline"] = {s: sum(v == s for v in base.values()) for s in ("passed", "failed", "error", "skipped")}
    log(f"{lib['name']} baseline: {rep['baseline']}")
    if rep["baseline"]["passed"] < 20:
        return [], dict(rep, error="baseline has too few passing tests")
    files_text = {str(f.relative_to(repo)): f.read_text() for f in source_files(repo, lib)}
    cands = []
    for rel, text in files_text.items():
        try:
            for m in candidate_mutations(text):
                new = apply_mutation(text, m)
                if new:
                    cands.append(({"file": rel, "op": m[0], "func": m[1], "span": list(m[2])}, [(rel, new)]))
        except SyntaxError:
            continue
    rng.shuffle(cands)
    by_op = {}
    for c in cands:
        by_op.setdefault(c[0]["op"], []).append(c)
    picked, seen_fn = [], {}
    for rnd in range(3):  # round-robin over operators; at most rnd+1 tries per function in round rnd
        queues = {op: [c for c in q if c not in picked] for op, q in by_op.items()}
        while any(queues.values()) and len(picked) < n_try:
            for op in sorted(queues):
                while queues[op]:
                    c = queues[op].pop()
                    k = (c[0]["file"], c[0]["func"])
                    if seen_fn.get(k, 0) <= rnd:
                        picked.append(c)
                        seen_fn[k] = seen_fn.get(k, 0) + 1
                        break
    picked = picked[:n_try]
    rep["candidates"] = len(cands)
    rows = evaluate_mutants(repo, lib, base, picked, root, plugin_dir, files_text, log=log)
    rep["single_status"] = {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}
    ok = [r for r in rows if r["status"] == "ok" and all(file_clean(base, f) for f in r["test_files"])]
    per_fn, singles = {}, []
    for r in ok:
        k = (r["file"], r["func"])
        if per_fn.get(k, 0) < PER_FUNCTION:
            per_fn[k] = per_fn.get(k, 0) + 1
            singles.append(r)
    lib_defs = lib_def_names(repo, lib)
    tasks, slug = [], re.sub(r"[^a-z0-9]+", "_", lib["name"].lower())
    for j, r in enumerate(singles):
        cand = next(c for c in picked if c[0]["file"] == r["file"] and c[0]["span"] == r["span"] and c[0]["op"] == r["op"])
        t = build_task(repo, lib, f"fac_{slug}_{j:03d}", cand[1], r["f2p"], r["messages"], base, files_text,
                       out_snap_dir, lib_defs)
        if t:
            t["_meta"] = {"kind": "single", "ops": [r["op"]], "funcs": [f"{r['file']}::{r['func']}"]}
            tasks.append(t)
    # pairs: two kept singles in different functions -> both fixes needed (F2P = union)
    pairs = []
    pool = singles[:]
    rng.shuffle(pool)
    for a in pool:
        for b in pool:
            if len(pairs) >= n_pairs:
                break
            if (a["file"], a["func"]) >= (b["file"], b["func"]):
                continue
            f2p = sorted(set(a["f2p"]) | set(b["f2p"]))
            files = sorted({split_nodeid(k)[0] for k in f2p})
            if len(f2p) > MAX_F2P or len(files) > MAX_TEST_FILES or set(a["f2p"]) & set(b["f2p"]):
                continue
            ca = next(c for c in picked if c[0]["file"] == a["file"] and c[0]["span"] == a["span"] and c[0]["op"] == a["op"])
            cb = next(c for c in picked if c[0]["file"] == b["file"] and c[0]["span"] == b["span"] and c[0]["op"] == b["op"])
            if a["file"] == b["file"]:  # merge line-wise: both are single-line changes in different functions
                la, lb = ca[1][0][1].splitlines(keepends=True), cb[1][0][1].splitlines(keepends=True)
                orig = files_text[a["file"]].splitlines(keepends=True)
                if len(la) != len(orig) or len(lb) != len(orig):
                    continue
                merged = "".join(lb[i] if lb[i] != orig[i] else la[i] for i in range(len(orig)))
                edits = [(a["file"], merged)]
            else:
                edits = [ca[1][0], cb[1][0]]
            pairs.append((a, b, edits, f2p))
    pair_rows = evaluate_mutants(repo, lib, base, [({"pair": [a["func"], b["func"]]}, e) for a, b, e, _ in pairs],
                                 root, plugin_dir, files_text, log=log)
    np_ = 0
    for (a, b, edits, f2p), pr in zip(pairs, pair_rows):
        if pr["status"] not in ("ok", "too_many", "too_many_files"):  # timeouts / collection errors
            continue
        msgs = dict(a["messages"], **b["messages"])
        t = build_task(repo, lib, f"fac_{slug}_p{np_:02d}", edits, f2p, msgs, base, files_text, out_snap_dir, lib_defs)
        if t:
            t["_meta"] = {"kind": "pair", "ops": [a["op"], b["op"]],
                          "funcs": [f"{a['file']}::{a['func']}", f"{b['file']}::{b['func']}"]}
            tasks.append(t)
            np_ += 1
    rep.update(singles=len(singles), pairs=np_, tasks=len(tasks),
               ops={o: sum(o in t["_meta"]["ops"] for t in tasks) for o in sorted({o for t in tasks for o in t["_meta"]["ops"]})})
    return tasks, rep
