"""
Task factory v1 core (SWE-smith-lite): verifiable bug-fix tasks from pure-Python libraries the model has not been
evaluated on. Pure Python + git + pytest, no harness: importable on the Mac for a smoke test (smoke_local.py), and
concatenated into the Kaggle kernel by build.sh.

Per library: fetch a pinned release from GitHub -> baseline test outcomes -> AST mutations inside one function
(single) or two functions (pair) -> keep a mutation when 1..MAX_F2P baseline-passing tests fail and every other test
in those files still passes -> snapshot = mutated repo with the failing tests REMOVED (as with real hidden tests);
gold patch = revert of the mutation; test_patch = restore of the removed tests; issue = template from the failure.
v1 (after FS1: v14 solved 84% of v0 singles, 36% of pairs): a coverage map from the baseline run, so each mutant runs
only the test files that execute its lines, on parallel repo copies; sampling weighted to the hard operators; fewer
singles (hard operators first) and more multi-site tasks (2-3 functions, each fix needed); 14 more libraries.
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
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

LIBS = [  # name, GitHub repo, pinned tag, package, test dir (None = auto); new v1 libraries first, v0 ones last
    {"name": "cachetools", "gh": "tkem/cachetools", "tag": "v7.2.6", "pkg": "cachetools", "tests": "tests"},
    {"name": "toolz", "gh": "pytoolz/toolz", "tag": "1.2.0", "pkg": "toolz", "tests": "toolz/tests"},
    {"name": "isodate", "gh": "gweis/isodate", "tag": "0.7.2", "pkg": "isodate", "tests": None},
    {"name": "mistune", "gh": "lepture/mistune", "tag": "v3.4.0", "pkg": "mistune", "tests": "tests"},
    {"name": "pathspec", "gh": "cpburnz/python-pathspec", "tag": "v1.1.1", "pkg": "pathspec", "tests": None},
    {"name": "parse", "gh": "r1chardj0n3s/parse", "tag": "1.22.4", "pkg": "parse", "tests": None},
    {"name": "tabulate", "gh": "astanin/python-tabulate", "tag": "v0.10.0", "pkg": "tabulate", "tests": "test"},
    {"name": "marshmallow", "gh": "marshmallow-code/marshmallow", "tag": "4.3.1", "pkg": "marshmallow", "tests": "tests"},
    {"name": "sortedcontainers", "gh": "grantjenks/python-sortedcontainers", "tag": "v2.4.0", "pkg": "sortedcontainers",
     "tests": "tests"},
    {"name": "tomlkit", "gh": "python-poetry/tomlkit", "tag": "0.15.2", "pkg": "tomlkit", "tests": "tests"},
    {"name": "bidict", "gh": "jab/bidict", "tag": "v0.24.1", "pkg": "bidict", "tests": None},
    {"name": "humanize", "gh": "python-humanize/humanize", "tag": "4.16.0", "pkg": "humanize", "tests": "tests"},
    {"name": "inflection", "gh": "jpvanhal/inflection", "tag": "0.5.1", "pkg": "inflection", "tests": None},
    {"name": "natsort", "gh": "SethMMorton/natsort", "tag": "8.4.0", "pkg": "natsort", "tests": "tests"},
    {"name": "more-itertools", "gh": "more-itertools/more-itertools", "tag": "v11.2.1", "pkg": "more_itertools",
     "tests": "tests"},
    {"name": "sqlparse", "gh": "andialbrecht/sqlparse", "tag": "0.6.0", "pkg": "sqlparse", "tests": "tests"},
    {"name": "boltons", "gh": "mahmoud/boltons", "tag": "26.2.0", "pkg": "boltons", "tests": "tests"},
]
MAX_F2P = 10          # more failing tests than this = a global break, not an issue-sized bug
MAX_TEST_FILES = 2
MAX_F2P_MULTI, MAX_TEST_FILES_MULTI = 15, 3
PER_FUNCTION = 2      # kept singles per function (diversity)
OP_WEIGHT = {"cmp_swap": 3, "const_shift": 3, "drop_guard": 2, "drop_update": 2}  # FS1: hardest ops get more tries
HARD_FIRST = ["cmp_swap", "const_shift", "drop_guard", "drop_update", "drop_not", "boolop_swap", "arg_swap",
              "arith_swap", "exc_swap"]  # singles are kept in this order (FS1 solve rates, lowest first)
WORKERS = 4
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


def tests_dir(repo, lib):
    """The library's test directory (relative), auto-detected when LIBS says None."""
    if lib.get("tests"):
        return lib["tests"]
    for cand in ("tests", "test", f"{lib['pkg']}/tests", f"src/{lib['pkg']}/tests"):
        if (repo / cand).is_dir():
            return cand
    return "."


def import_root(repo, lib):
    """Directory to put on sys.path so `import <pkg>` resolves to the repo copy (src/ layout or flat)."""
    for base in (repo / "src", repo):
        if (base / lib["pkg"]).is_dir() or (base / f"{lib['pkg']}.py").exists():
            return base
    raise FileNotFoundError(f"package {lib['pkg']} not found in {repo}")


def run_tests(repo, lib, plugin_dir, out_json, targets=None, timeout=900, cov_file=None):
    """-> {"outcomes": {nodeid: passed|failed|error|skipped}, "messages": {nodeid: text}} (None on timeout).
    cov_file: also record per-test line coverage of the package (coverage.py dynamic contexts) for coverage_map."""
    env = dict(os.environ, FACTORY_OUT=str(out_json), PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(import_root(repo, lib)), str(plugin_dir)]))
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "factory_plugin", "-p", "no:cacheprovider", "-o", "addopts=",
           "--timeout=30"]
    if cov_file:
        env["COVERAGE_FILE"] = str(cov_file)
        root = import_root(repo, lib)
        cmd += [f"--cov={root / lib['pkg'] if (root / lib['pkg']).is_dir() else root}", "--cov-context=test",
                "--cov-report="]
    cmd += targets or [tests_dir(repo, lib)]
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
    return [f for f in files if "test" not in f.name.lower()
            and not {"tests", "test", "testing"} & set(f.relative_to(root).parts)]


def coverage_map(repo, cov_file):
    """-> {(rel_source_file, lineno): {test files whose tests executed that line}} from the baseline run."""
    from coverage import CoverageData
    data = CoverageData(basename=str(cov_file))
    data.read()
    out = {}
    for f in data.measured_files():
        try:
            rel = str(pathlib.Path(f).resolve().relative_to(repo.resolve()))
        except ValueError:
            continue
        for ln, ctxs in data.contexts_by_lineno(f).items():
            files = {c.split("::", 1)[0] for c in ctxs if "::" in c}
            if files:
                out[(rel, ln)] = files
    return out


def targets_for(cov, rel, span):
    """Test files covering the mutated lines (a deletion span covers its first line)."""
    lines = range(span[0], (span[1] if len(span) == 2 else span[0]) + 1)
    return sorted(set().union(*[cov.get((rel, ln), set()) for ln in lines]))


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
def evaluate_mutants(repos, lib, base, muts, plugin_dir, files_text, log=print, max_f2p=MAX_F2P,
                     max_files=MAX_TEST_FILES):
    """Apply each mutant (meta with "targets", list of (rel_file, new_text)) in one of the worker repo copies, run only
    its target test files, restore. -> rows in input order. Mutants with no covering tests are 'uncovered'."""
    free = list(repos)
    lock = threading.Lock()
    done = [0]

    def one(item):
        meta, edits = item
        if not meta.get("targets"):
            return dict(meta, status="uncovered")
        with lock:
            repo = free.pop()
        try:
            for rel, new in edits:
                (repo / rel).write_text(new)
            res = run_tests(repo, lib, plugin_dir, repo.parent / f"{repo.name}_mut.json", targets=meta["targets"],
                            timeout=300)
            for rel, _ in edits:
                (repo / rel).write_text(files_text[rel])
        finally:
            with lock:
                free.append(repo)
                done[0] += 1
                if done[0] % 50 == 0:
                    log(f"  {lib['name']}: {done[0]}/{len(muts)} mutants")
        if res is None:
            return dict(meta, status="timeout")
        oc = res["outcomes"]
        if any(k.startswith("<collect>") for k in oc):
            return dict(meta, status="collect_error")
        broke = sorted(k for k, v in oc.items() if v in ("failed", "error") and base.get(k) == "passed")
        files = sorted({split_nodeid(k)[0] for k in broke})
        status = "equivalent" if not broke else "too_many" if len(broke) > max_f2p else \
            "too_many_files" if len(files) > max_files else "ok"
        return dict(meta, status=status, f2p=broke, test_files=files,
                    messages={k: res["messages"].get(k, "") for k in broke})

    with ThreadPoolExecutor(len(repos)) as ex:
        return list(ex.map(one, muts))


def file_clean(base, test_file):
    """Every baseline test in this file passes or skips (the verifier runs whole files and needs 0 failures)."""
    return all(v in ("passed", "skipped") for k, v in base.items() if split_nodeid(k)[0] == test_file)


def build_task(repo, lib, tid, edits, f2p, messages, base, files_text, out_snap_dir, lib_defs):
    """Snapshot with mutation applied and F2P tests removed; gold = revert; test_patch = restore. -> task row."""
    for n in f2p:  # the verifier matches required names by prefix: a skipped/xfailed sibling would fail the task
        path, _, name = split_nodeid(n)
        if any(v == "skipped" and split_nodeid(k)[0] == path and re.match(re.escape(name) + r"($|[_\[\d])", split_nodeid(k)[2])
               for k, v in base.items()):
            return None
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
    # PASS_TO_PASS stays empty: the verifier already runs the whole test files and needs 0 failures, and listing
    # every passing node trips its prefix matching on xfailed tests (sqlparse test_python vs test_python_*, v0 run)
    return {"instance_id": tid, "repo": f"{lib['gh']}", "base_commit": base_commit,
            "problem_statement": make_issue(lib, f2p, messages, test_sources, lib_defs),
            "patch": gold, "test_patch": test_patch, "hints_text": "", "created_at": "2026-10-28T00:00:00Z",
            "FAIL_TO_PASS": json.dumps(f2p), "PASS_TO_PASS": "[]", "environment_setup_commit": base_commit}


def merge_edits(files_text, parts):
    """Combine single-site mutants (each [(rel, new_text)]) into one edit list. -> None if two touch the same file and
    either changes its line count (statement deletion) or both change the same line."""
    by_file = {}
    for edits in parts:
        for rel, new in edits:
            by_file.setdefault(rel, []).append(new)
    out = []
    for rel, news in sorted(by_file.items()):
        if len(news) == 1:
            out.append((rel, news[0]))
            continue
        orig = files_text[rel].splitlines(keepends=True)
        splits = [n.splitlines(keepends=True) for n in news]
        if any(len(s) != len(orig) for s in splits):
            return None
        merged = []
        for i, o in enumerate(orig):
            changed = [s[i] for s in splits if s[i] != o]
            if len(changed) > 1:
                return None
            merged.append(changed[0] if changed else o)
        out.append((rel, "".join(merged)))
    return out


def pick_singles(pool, n):
    """Hard operators first (FS1), with a per-operator quota so one operator cannot take every slot."""
    quota = {op: 2 * OP_WEIGHT.get(op, 1) for op in HARD_FIRST}
    kept, count = [], {}
    for r in sorted(pool, key=lambda r: HARD_FIRST.index(r["op"]) if r["op"] in HARD_FIRST else len(HARD_FIRST)):
        if len(kept) >= n:
            break
        if count.get(r["op"], 0) < quota.get(r["op"], 2):
            kept.append(r)
            count[r["op"]] = count.get(r["op"], 0) + 1
    return kept


def multi_candidates(pool, files_text, n, rng, max_uses=2):
    """Sets of 2 or 3 kept singles in different functions with disjoint failing tests (so every fix is needed),
    half pairs and half triples; each single is used in at most max_uses sets."""
    out, uses, seen = [], {}, set()
    for attempt in range(n * 60):
        if len(out) >= n:
            break
        k = 2 + attempt % 2
        free = [i for i in range(len(pool)) if uses.get(i, 0) < max_uses]
        if len(free) < k:
            break
        idx = tuple(sorted(rng.sample(free, k)))
        parts = [pool[i] for i in idx]
        if idx in seen or len({(r["file"], r["func"]) for r in parts}) < k:
            continue
        seen.add(idx)
        f2p = [set(r["f2p"]) for r in parts]
        union = sorted(set().union(*f2p))
        if sum(map(len, f2p)) != len(union) or len(union) > MAX_F2P_MULTI or \
                len({split_nodeid(x)[0] for x in union}) > MAX_TEST_FILES_MULTI:
            continue
        edits = merge_edits(files_text, [r["edits"] for r in parts])
        if edits is None:
            continue
        for i in idx:
            uses[i] = uses.get(i, 0) + 1
        out.append((parts, edits, union))
    return out


def run_library(lib, root, out_snap_dir, n_try=300, n_single=20, n_multi=30, seed=0, log=print):
    """Whole pipeline for one library -> (tasks, report)."""
    rng = random.Random(seed)
    repo = fetch(lib, root / lib["name"])
    plugin_dir = root / "_plugin"
    write_plugin(plugin_dir)
    cov_file = root / f"{lib['name']}.coverage"
    base_res = run_tests(repo, lib, plugin_dir, root / f"{lib['name']}_base.json", timeout=1800, cov_file=cov_file)
    rep = {"lib": lib["name"], "tag": lib["tag"], "tests_dir": tests_dir(repo, lib)}
    if base_res is None:
        return [], dict(rep, error="baseline timeout")
    base = base_res["outcomes"]
    rep["baseline"] = {s: sum(v == s for v in base.values()) for s in ("passed", "failed", "error", "skipped")}
    log(f"{lib['name']} baseline: {rep['baseline']}")
    if rep["baseline"]["passed"] < 20:
        return [], dict(rep, error="baseline has too few passing tests")
    cov = coverage_map(repo, cov_file) if cov_file.exists() else {}
    rep["covered_lines"] = len(cov)
    if not cov:
        return [], dict(rep, error="no coverage data")
    files_text = {str(f.relative_to(repo)): f.read_text() for f in source_files(repo, lib)}
    cands = []
    for rel, text in files_text.items():
        try:
            for m in candidate_mutations(text):
                new = apply_mutation(text, m)
                if new:
                    cands.append(({"file": rel, "op": m[0], "func": m[1], "span": list(m[2]),
                                   "targets": targets_for(cov, rel, m[2])}, [(rel, new)]))
        except SyntaxError:
            continue
    rep["candidates"] = len(cands)
    cands = [c for c in cands if c[0]["targets"]]  # no test executes the line: it cannot break anything
    rep["covered_candidates"] = len(cands)
    rng.shuffle(cands)
    by_op = {}
    for c in cands:
        by_op.setdefault(c[0]["op"], []).append(c)
    picked, seen_fn = [], {}
    for rnd in range(3):  # weighted round-robin over operators; at most rnd+1 tries per function in round rnd
        queues = {op: [c for c in q if c not in picked] for op, q in by_op.items()}
        while any(queues.values()) and len(picked) < n_try:
            for op in [o for o in sorted(queues) for _ in range(OP_WEIGHT.get(o, 1))]:
                while queues[op]:
                    c = queues[op].pop()
                    k = (c[0]["file"], c[0]["func"])
                    if seen_fn.get(k, 0) <= rnd:
                        picked.append(c)
                        seen_fn[k] = seen_fn.get(k, 0) + 1
                        break
    picked = picked[:n_try]
    repos = []
    for w in range(WORKERS):
        dst = root / f"{lib['name']}_w{w}"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(repo, dst, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
        repos.append(dst)
    try:
        rows = evaluate_mutants(repos, lib, base, picked, plugin_dir, files_text, log=log)
        rep["single_status"] = {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}
        pool, per_fn = [], {}
        for c, r in zip(picked, rows):
            if r["status"] == "ok" and all(file_clean(base, f) for f in r["test_files"]):
                k = (r["file"], r["func"])
                if per_fn.get(k, 0) < PER_FUNCTION:
                    per_fn[k] = per_fn.get(k, 0) + 1
                    pool.append(dict(r, edits=c[1]))
        rep["single_pool"] = len(pool)
        lib_defs = lib_def_names(repo, lib)
        tasks, slug = [], re.sub(r"[^a-z0-9]+", "_", lib["name"].lower())
        for j, r in enumerate(pick_singles(pool, n_single)):
            t = build_task(repo, lib, f"fac_{slug}_{j:03d}", r["edits"], r["f2p"], r["messages"], base, files_text,
                           out_snap_dir, lib_defs)
            if t:
                t["_meta"] = {"kind": "single", "ops": [r["op"]], "funcs": [f"{r['file']}::{r['func']}"]}
                tasks.append(t)
        # multi-site: 2-3 kept singles together; kept only if every part's failing tests still fail in the combination
        multis = multi_candidates(pool, files_text, int(n_multi * 1.5), rng)
        mrows = evaluate_mutants(
            repos, lib, base,
            [({"parts": [r["func"] for r in parts],
               "targets": sorted(set().union(*[r["targets"] for r in parts]) | {split_nodeid(x)[0] for x in union})},
              edits) for parts, edits, union in multis],
            plugin_dir, files_text, log=log, max_f2p=MAX_F2P_MULTI, max_files=MAX_TEST_FILES_MULTI)
        rep["multi_status"] = {s: sum(r["status"] == s for r in mrows) for s in sorted({r["status"] for r in mrows})}
        nm = 0
        for (parts, edits, union), mr in zip(multis, mrows):
            if nm >= n_multi or mr["status"] != "ok" or not set(union) <= set(mr["f2p"]):
                continue
            if not all(file_clean(base, f) for f in mr["test_files"]):
                continue
            msgs = {k: v for r in parts for k, v in r["messages"].items()}
            msgs.update({k: v for k, v in mr["messages"].items() if v})
            t = build_task(repo, lib, f"fac_{slug}_m{nm:02d}", edits, mr["f2p"], msgs, base, files_text, out_snap_dir,
                           lib_defs)
            if t:
                t["_meta"] = {"kind": "pair" if len(parts) == 2 else "triple", "ops": [r["op"] for r in parts],
                              "funcs": [f"{r['file']}::{r['func']}" for r in parts]}
                tasks.append(t)
                nm += 1
    finally:
        for d in repos:
            shutil.rmtree(d, ignore_errors=True)
    rep.update(singles=sum(t["_meta"]["kind"] == "single" for t in tasks), multis=nm, tasks=len(tasks),
               ops={o: sum(o in t["_meta"]["ops"] for t in tasks) for o in sorted({o for t in tasks for o in t["_meta"]["ops"]})})
    return tasks, rep
