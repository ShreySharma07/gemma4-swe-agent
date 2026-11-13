# ------------------------------------------------------------------ F3 driver: real bug-fix commits -> gate tasks
# (Kaggle CPU, Internet ON; pre-registered in docs/experiments.md, Nov 13). Evaluation only, never training.
# Per library: full clone -> fix-like commits that change 1-3 source .py files and >= 1 test file -> F2P = tests that
# pass at the commit and do not pass at the parent with the commit's tests -> snapshot = parent tree, gold = source
# diff, test_patch = test diff. Then factory_main's official verification (gold resolves, empty does not).
F3_LIBS = [
    {"name": "sqlglot", "gh": "tobymao/sqlglot", "pkg": "sqlglot"},
    {"name": "tinydb", "gh": "msiemens/tinydb", "pkg": "tinydb"},
    {"name": "pypika", "gh": "kayak/pypika", "pkg": "pypika"},
    {"name": "parso", "gh": "davidhalter/parso", "pkg": "parso"},
    {"name": "jsonpathng", "gh": "h2non/jsonpath-ng", "pkg": "jsonpath_ng", "pip": "jsonpath-ng"},
    {"name": "box", "gh": "cdgriffith/Box", "pkg": "box", "pip": "python-box"},
    {"name": "cerberus", "gh": "pyeve/cerberus", "pkg": "cerberus"},
    {"name": "voluptuous", "gh": "alecthomas/voluptuous", "pkg": "voluptuous"},
    {"name": "asttokens", "gh": "gristlabs/asttokens", "pkg": "asttokens"},
    {"name": "construct", "gh": "construct/construct", "pkg": "construct"},
    {"name": "mako", "gh": "sqlalchemy/mako", "pkg": "mako"},
    {"name": "markdownit", "gh": "executablebooks/markdown-it-py", "pkg": "markdown_it", "pip": "markdown-it-py"},
    {"name": "arrow", "gh": "arrow-py/arrow", "pkg": "arrow"},
    {"name": "glom", "gh": "mahmoud/glom", "pkg": "glom"},
    {"name": "isort", "gh": "PyCQA/isort", "pkg": "isort"},
    {"name": "pyflakes", "gh": "PyCQA/pyflakes", "pkg": "pyflakes"},
    {"name": "pycodestyle", "gh": "PyCQA/pycodestyle", "pkg": "pycodestyle"},
    {"name": "lark", "gh": "lark-parser/lark", "pkg": "lark"},
    {"name": "markdown", "gh": "Python-Markdown/markdown", "pkg": "markdown"},
]
for _l in F3_LIBS:
    _l["tests"] = None
LIBS = F3_LIBS  # the verification tail checks these against the scorer site
CFG = {"SINCE": "2022-01-01", "MAX_TRY": 60, "MAX_KEEP": 12, "MAX_SRC_FILES": 3, "MAX_SRC_LINES": 80, "MAX_F2P3": 15,
       "LIB_MIN": 35, "CONCURRENCY": 4, "SESSION_H": 8.0}
OUT = pathlib.Path("/kaggle/working/factory")
ROOT = pathlib.Path("/tmp/factory")
T0 = time.time()
shutil.rmtree(OUT, ignore_errors=True)
(OUT / "snapshots").mkdir(parents=True)
(ROOT / "clones").mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest", "pytest-timeout", "pytest-cov", "coverage",
                "hypothesis", "pytest-mock"])
for _l in F3_LIBS:  # the library's own dependencies (the repo copy shadows the installed package through PYTHONPATH)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", _l.get("pip", _l["name"])], capture_output=True)
comp_row = json.loads(open(tasks_path).readline())
PLUG = ROOT / "plugin"
write_plugin(PLUG)

FIX = re.compile(r"\b(fix|fixe[sd]|bug|issue|regression|crash|incorrect|wrong|handle|closes?|resolve[sd]?)\b|#\d+", re.I)
NOT_FIX = re.compile(r"\b(typo|docs?|bump|release|lint|format|refactor|typing|type hints?|changelog|ci|rename|deprecat)\b", re.I)
TEST = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")
IGNORABLE = re.compile(r"\.(md|rst|txt)$|(^|/)(CHANGE|NEWS|HISTORY|AUTHORS)[^/]*$|^docs?/", re.I)


def git(repo, *a, timeout=300):
    return sh(["git", *a], cwd=repo, timeout=timeout)


def classify(lib, files):
    """files: [(status, path)] -> (src, tests) or None when the commit is not a small source + test change."""
    src, tests = [], []
    pk = lib["pkg"]
    for st, p in files:
        if TEST.search(p):
            if not p.endswith(".py") or st not in ("M", "A"):
                return None
            tests.append(p)
        elif p.endswith(".py") and (p.startswith((pk + "/", "src/" + pk + "/")) or p in (pk + ".py", "src/" + pk + ".py")):
            if st != "M":
                return None
            src.append(p)
        elif not IGNORABLE.search(p):
            return None
    return (src, tests) if 1 <= len(src) <= CFG["MAX_SRC_FILES"] and tests else None


def run_at(repo, lib, tests, tag):
    try:
        targets = [t for t in tests if (repo / t).exists() and not t.endswith("conftest.py")]
        return run_tests(repo, lib, PLUG, ROOT / f"out_{lib['name']}_{tag}.json", targets=targets, timeout=300) if targets else None
    except Exception:  # noqa: BLE001  (package layout differs at this commit, ...)
        return None


def make_task(repo, lib, tid, sha, parent, src, tests, f2p, msgs_before, date):
    git(repo, "checkout", "-q", "-f", parent)
    git(repo, "clean", "-qfdx")
    work = ROOT / f"task_{tid}"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(repo, work, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    for junk in (".github", "docs", "doc"):
        shutil.rmtree(work / junk, ignore_errors=True)
    base_commit = git_snapshot(work)
    env = dict(os.environ, **GIT_ENV)

    def diff_with(paths):
        for p in paths:
            (work / p).parent.mkdir(parents=True, exist_ok=True)
            (work / p).write_text(git(repo, "show", f"{sha}:{p}").stdout)
        sh(["git", "add", "-A"], cwd=work, env=env)
        d = sh(["git", "diff", "--cached"], cwd=work).stdout
        sh(["git", "reset", "-q", "--hard"], cwd=work)
        sh(["git", "clean", "-qfd"], cwd=work)
        return d

    gold, test_patch = diff_with(src), diff_with(tests)
    if not gold.strip() or not test_patch.strip():
        shutil.rmtree(work, ignore_errors=True)
        return None
    sh(["tar", "-czf", str(OUT / "snapshots" / f"{tid}.tgz"), "-C", str(work), "."])
    shutil.rmtree(work, ignore_errors=True)
    body = git(repo, "show", "-s", "--format=%B", sha).stdout
    body = "\n".join(l for l in body.splitlines() if not re.match(r"\s*(co-authored-by|signed-off-by)\b", l, re.I)).strip()[:1500]
    fail = failure_line(msgs_before.get(f2p[0], "")) if msgs_before else ""
    name = f2p[0].split("::")[-1]
    issue = body + (f"\n\nObserved: `{name}` fails" + (f": {fail}" if fail else ".") )
    n_hunks = gold.count("\n@@ ")
    return {"instance_id": tid, "repo": lib["gh"], "base_commit": base_commit, "problem_statement": issue,
            "patch": gold, "test_patch": test_patch, "hints_text": "", "created_at": date,
            "FAIL_TO_PASS": json.dumps(f2p), "PASS_TO_PASS": "[]", "environment_setup_commit": base_commit,
            "_meta": {"kind": "single" if n_hunks <= 1 else "pair" if n_hunks == 2 else "triple", "ops": [], "issue": "commit",
                      "sha": sha, "src_files": len(src), "hunks": n_hunks, "f2p": len(f2p)}}


def mine(lib):
    rep = {"lib": lib["name"], "commits": 0, "fix_like": 0, "shape_ok": 0, "tried": 0, "tasks": 0, "drop": {}}
    t0 = time.time()

    def drop(why):
        rep["drop"][why] = rep["drop"].get(why, 0) + 1

    repo = ROOT / "clones" / lib["name"]
    r = sh(["git", "clone", "-q", f"https://github.com/{lib['gh']}", str(repo)], timeout=900)
    if r.returncode != 0:
        rep["error"] = "clone: " + r.stderr[-200:]
        return [], rep
    lines = git(repo, "log", "--no-merges", f"--since={CFG['SINCE']}", "--format=%H%x1f%P%x1f%cI%x1f%s").stdout.splitlines()
    rep["commits"] = len(lines)
    out = []
    for line in lines:
        if rep["tried"] >= CFG["MAX_TRY"] or len(out) >= CFG["MAX_KEEP"] or time.time() - t0 > CFG["LIB_MIN"] * 60:
            break
        sha, parent, date, subj = (line.split("\x1f") + ["", "", ""])[:4]
        if not parent or " " in parent or not FIX.search(subj) or NOT_FIX.search(subj):
            continue
        rep["fix_like"] += 1
        files = [tuple(l.split("\t")[:2]) for l in git(repo, "show", "--name-status", "--format=", sha).stdout.splitlines() if "\t" in l]
        cl = classify(lib, files)
        if not cl:
            continue
        src, tests = cl
        num = git(repo, "diff", "--numstat", parent, sha, "--", *src).stdout.split()
        if sum(int(x) for x in num if x.isdigit()) > CFG["MAX_SRC_LINES"]:
            continue
        rep["shape_ok"] += 1
        rep["tried"] += 1
        git(repo, "checkout", "-q", "-f", sha)
        git(repo, "clean", "-qfdx")
        after = run_at(repo, lib, tests, "b")
        if not after or not after["outcomes"]:
            drop("after_no_result")
            continue
        if any(v not in ("passed", "skipped") for v in after["outcomes"].values()):
            drop("after_not_clean")
            continue
        git(repo, "checkout", "-q", "-f", parent)
        git(repo, "clean", "-qfdx")
        git(repo, "checkout", "-q", sha, "--", *tests)
        before = run_at(repo, lib, tests, "a") or {"outcomes": {}, "messages": {}}
        f2p = sorted(n for n, v in after["outcomes"].items() if v == "passed" and before["outcomes"].get(n) != "passed")
        if not f2p:
            drop("no_f2p")
            continue
        if len(f2p) > CFG["MAX_F2P3"]:
            drop("too_many_f2p")
            continue
        t = make_task(repo, lib, f"f3_{lib['name']}_{len(out):03d}", sha, parent, src, tests, f2p, before["messages"], date)
        if t is None:
            drop("empty_diff")
            continue
        out.append(t)
    rep["tasks"] = len(out)
    shutil.rmtree(repo, ignore_errors=True)
    return out, rep


tasks, lib_reports = [], []
with ThreadPoolExecutor(4) as ex:
    futs = [(lib, ex.submit(mine, lib)) for lib in F3_LIBS]
    for lib, fu in futs:
        try:
            lt, rep = fu.result()
        except Exception as e:  # noqa: BLE001  one bad library must not stop the others
            lt, rep = [], {"lib": lib["name"], "error": f"{type(e).__name__}: {e}"[:400]}
        log("LIB", json.dumps(rep))
        lib_reports.append(rep)
        tasks += lt
log(f"mined {len(tasks)} candidate tasks in {(time.time() - T0) / 60:.1f} min")
VERIFY_ONLY = False
