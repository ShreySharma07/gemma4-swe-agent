"""code-map v2: read-only, relationship-first navigation of a Python repository (stdlib only).

Usage: args=[COMMAND, ...]
  search PATTERN        regex search; each hit shows file:line and the ENCLOSING function/class
  show FILE START END   print lines START..END (max 100) with line numbers and the enclosing symbol
  find NAME             definitions of NAME (bare, Class.method or dotted) -> qualified name, file:lines, signature
  trace NAME            one-shot relationship view: definition + callers + callees + overrides + tests
  callers NAME          who calls NAME, 2 levels up (call chains towards public entry points)
  callees NAME          what NAME calls, resolved to definitions with file:line
  path FROM TO          shortest call chain FROM -> ... -> TO (e.g. public API -> buggy helper)
  similar NAME          overrides / same-named methods / structurally similar functions (other places to fix)
  tests NAME            test files and test functions that reference NAME
  outline FILE          classes and functions of one file with line ranges
  status                calls used / remaining
Options: --tests (include test files where relevant)
"""
import ast
import collections
import hashlib
import inspect
import json
import os
import pickle
import re
import subprocess
import sys

BUDGET = 22  # code-map calls allowed per task before the localizer must hand off
SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "build", "dist", "__pycache__", ".tox", ".eggs",
             "site-packages", "docs", "docs_src", ".mypy_cache", ".pytest_cache"}
MAX_FILES = 6000
MAX_OUT = 3800
COMMON = {"get", "set", "append", "extend", "update", "items", "keys", "values", "pop", "join", "split", "strip",
          "format", "replace", "startswith", "endswith", "isinstance", "len", "str", "int", "dict", "list", "tuple",
          "print", "super", "getattr", "setattr", "hasattr", "type", "range", "enumerate", "zip", "sorted", "min",
          "max", "any", "all", "map", "filter", "open", "repr", "bool", "float", "add", "copy", "write", "read",
          "lower", "upper", "encode", "decode", "match", "search", "sub", "insert", "remove", "clear", "close",
          "next", "iter", "id", "hash", "abs", "sum", "round", "issubclass", "callable", "vars", "object", "set"}

_out = []


def emit(s=""):
    _out.append(str(s))


def flush():
    text = "\n".join(_out)
    if len(text) > MAX_OUT:
        text = text[:MAX_OUT] + "\n... (truncated; narrow the query)"
    print(text)


# ---------------------------------------------------------------- repo + per-task state
def git(args, cwd):
    try:
        r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def repo_root():
    cands = []
    for fr in inspect.stack():
        cwd = fr.frame.f_locals.get("_orig_cwd")
        if isinstance(cwd, str):
            cands.append(cwd)
    cands += [os.environ.get("CODE_MAP_ROOT"), "/workspace", os.environ.get("PWD"), os.getcwd()]
    for c in cands:
        if c and os.path.isdir(c) and not os.path.basename(c).startswith("tmp"):
            return git(["rev-parse", "--show-toplevel"], c) or c
    return os.getcwd()


def state_key(root):
    head = git(["rev-parse", "HEAD"], root)
    try:
        st = os.stat(os.path.join(root, ".git"))
        extra = f"{st.st_ino}-{st.st_ctime}"
    except OSError:
        extra = ""
    return hashlib.md5(f"{root}|{head}|{extra}".encode()).hexdigest()[:16]


def bump_counter(key, cmd):
    path = f"/tmp/.codemap_{key}.json"
    try:
        st = json.load(open(path))
    except Exception:  # noqa: BLE001
        st = {"n": 0}
    if cmd != "status":
        st["n"] += 1
    try:
        json.dump(st, open(path, "w"))
    except Exception:  # noqa: BLE001
        pass
    return st["n"]


# ---------------------------------------------------------------- index
def is_test(rel):
    parts = rel.replace("\\", "/").split("/")
    base = parts[-1]
    return any(p in ("tests", "test", "testing") for p in parts[:-1]) or base.startswith("test_") or \
        base.endswith("_test.py") or base == "conftest.py"


def module_name(rel):
    mod = rel[:-3].replace(os.sep, ".").replace("/", ".")
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    for prefix in ("src.", "lib."):
        if mod.startswith(prefix):
            mod = mod[len(prefix):]
    return mod


class Indexer(ast.NodeVisitor):
    def __init__(self, mod, rel, lines, idx):
        self.mod, self.rel, self.lines, self.idx = mod, rel, lines, idx
        self.stack = []  # (qual, kind)
        self.imports = {}

    def cur(self):
        return self.stack[-1][0] if self.stack else f"{self.mod}.<module>"

    def cur_class(self):
        for q, k in reversed(self.stack):
            if k == "class":
                return q
        return None

    def visit_Import(self, node):
        for a in node.names:
            self.imports[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]

    def visit_ImportFrom(self, node):
        base = node.module or ""
        if node.level:
            pkg = self.mod.split(".")
            if not self.rel.endswith("__init__.py"):
                pkg = pkg[:-1]
            pkg = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
            base = ".".join(pkg + ([base] if base else []))
        for a in node.names:
            self.imports[a.asname or a.name] = f"{base}.{a.name}" if base else a.name

    def _def(self, node, kind):
        q = f"{self.cur()}.{node.name}" if self.stack else f"{self.mod}.{node.name}"
        sig = self.lines[node.lineno - 1].strip()[:150] if node.lineno - 1 < len(self.lines) else ""
        doc = ast.get_docstring(node) or ""
        bases = [b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else "" for b in
                 getattr(node, "bases", [])]
        idents = set()
        if kind != "class":
            for n in ast.walk(node):
                if isinstance(n, ast.Name):
                    idents.add(n.id)
                elif isinstance(n, ast.Attribute):
                    idents.add(n.attr)
        self.idx["defs"].append(dict(q=q, short=node.name, kind=kind, rel=self.rel, start=node.lineno,
                                     end=getattr(node, "end_lineno", node.lineno), sig=sig,
                                     doc=doc.strip().split("\n")[0][:120], cls=self.cur_class(), bases=bases,
                                     test=is_test(self.rel), idents=sorted(idents)[:200]))
        self.stack.append((q, kind))
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self._def(node, "def")

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._def(node, "class")

    def visit_Call(self, node):
        f = node.func
        name = base = None
        if isinstance(f, ast.Name):
            name = f.id
        elif isinstance(f, ast.Attribute):
            name = f.attr
            if isinstance(f.value, ast.Name):
                base = f.value.id
        if name:
            self.idx["calls"].append((self.cur(), name, base, self.rel, node.lineno, self.mod, self.cur_class()))
        self.generic_visit(node)


def build_index(root):
    idx = {"defs": [], "calls": [], "imports": {}, "files": []}
    n = 0
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in SKIP_DIRS and not d.startswith("."))
        for fn in sorted(fns):
            if not fn.endswith(".py"):
                continue
            n += 1
            if n > MAX_FILES:
                break
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, root)
            idx["files"].append(rel)
            try:
                src = open(full, encoding="utf-8", errors="replace").read()
                tree = ast.parse(src)
            except Exception:  # noqa: BLE001
                continue
            mod = module_name(rel)
            v = Indexer(mod, rel, src.splitlines(), idx)
            v.visit(tree)
            idx["imports"][mod] = v.imports
    by_q = {d["q"]: d for d in idx["defs"]}
    by_short = collections.defaultdict(list)
    for d in idx["defs"]:
        by_short[d["short"]].append(d["q"])
    # resolve calls -> edges
    fwd, rev = collections.defaultdict(list), collections.defaultdict(list)
    for caller, name, base, rel, line, mod, cls in idx["calls"]:
        targets = []
        if base in ("self", "cls") and cls:
            cand = f"{cls}.{name}"
            if cand in by_q:
                targets = [cand]
            else:
                targets = [q for q in by_short.get(name, []) if by_q[q]["cls"]][:3]
        else:
            imp = idx["imports"].get(mod, {})
            if base and base in imp and f"{imp[base]}.{name}" in by_q:
                targets = [f"{imp[base]}.{name}"]
            elif not base and f"{mod}.{name}" in by_q:
                targets = [f"{mod}.{name}"]
            elif not base and name in imp and imp[name] in by_q:
                targets = [imp[name]]
            elif not base and name in imp:
                tail = imp[name].split(".")[-1]
                targets = [q for q in by_short.get(tail, []) if q.endswith("." + tail)][:2]
            elif name not in COMMON:
                c = by_short.get(name, [])
                targets = c[:3] if len(c) <= 3 else []
        caller_is_test = is_test(rel)
        targets = [t for t in targets if caller_is_test or not by_q[t]["test"]]
        for t in targets:
            if by_q[t]["kind"] == "class":
                init = f"{t}.__init__"
                t = init if init in by_q else t
            fwd[caller].append((t, rel, line))
            rev[t].append((caller, rel, line))
    idx["fwd"], idx["rev"], idx["by_short"] = dict(fwd), dict(rev), dict(by_short)
    idx["by_q"] = by_q
    return idx


def load_index(root, key):
    path = f"/tmp/.codemap_{key}.pkl"
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:  # noqa: BLE001
        pass
    idx = build_index(root)
    try:
        with open(path, "wb") as f:
            pickle.dump(idx, f)
    except Exception:  # noqa: BLE001
        pass
    return idx


# ---------------------------------------------------------------- helpers
def resolve(idx, name, tests=False):
    name = name.strip().strip("`'\"()").rstrip(":")
    by_q = idx["by_q"]
    if name in by_q:
        return [name]
    hits = [q for q in by_q if q.endswith("." + name)]
    if not hits and "." not in name:
        hits = list(idx["by_short"].get(name, []))
    if not hits:
        low = name.lower()
        hits = [q for q in by_q if q.lower().endswith("." + low)]
    if not tests:
        nt = [q for q in hits if not by_q[q]["test"]]
        hits = nt or hits
    hits.sort(key=lambda q: (by_q[q]["test"], q.count("."), len(q)))
    return hits


def loc(d):
    return f"{d['rel']}:{d['start']}-{d['end']}"


def enclosing(idx, rel, line):
    best = None
    for d in idx["defs"]:
        if d["rel"] == rel and d["start"] <= line <= d["end"] and (best is None or d["start"] >= best["start"]):
            best = d
    return best["q"] if best else module_name(rel)


def no_match(name):
    emit(f"no definition matching {name!r}. Try `search` with a regex, or a shorter name.")


def short_list(items, k, tests=True):
    out, seen = [], set()
    for it in items:
        if not tests and is_test(it[1]):
            continue
        key = it[0]
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out[:k]


# ---------------------------------------------------------------- commands
def cmd_search(root, idx, pattern, tests):
    try:
        rx = re.compile(pattern)
    except re.error:
        rx = re.compile(re.escape(pattern))
    hits = 0
    for rel in idx["files"]:
        if not tests and is_test(rel):
            continue
        try:
            lines = open(os.path.join(root, rel), encoding="utf-8", errors="replace").read().splitlines()
        except OSError:
            continue
        for i, l in enumerate(lines, 1):
            if rx.search(l):
                hits += 1
                if hits <= 30:
                    emit(f"{rel}:{i}  [{enclosing(idx, rel, i)}]  | {l.strip()[:120]}")
    if hits == 0:
        emit(f"no matches for {pattern!r}" + ("" if tests else " outside tests (add --tests to include tests)"))
    elif hits > 30:
        emit(f"... {hits - 30} more matches; make the pattern more specific")


def cmd_show(root, idx, rel, a, b):
    rel = rel.replace("/workspace/", "").lstrip("./") if not os.path.isabs(rel) else os.path.relpath(rel, root)
    try:
        lines = open(os.path.join(root, rel), encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        emit(f"cannot open {rel}")
        return
    a = max(1, int(a))
    b = min(len(lines), int(b), a + 99)
    emit(f"{rel} lines {a}-{b} of {len(lines)}  (inside {enclosing(idx, rel, a)})")
    for i in range(a, b + 1):
        emit(f"{i:5d}  {lines[i - 1]}")


def cmd_find(idx, name, tests):
    hits = resolve(idx, name, tests)
    if not hits:
        return no_match(name)
    for q in hits[:12]:
        d = idx["by_q"][q]
        emit(f"{q}  [{d['kind']}]  {loc(d)}  | {d['sig']}")
    if len(hits) > 12:
        emit(f"... {len(hits) - 12} more")


def tests_for(root, idx, short, k=5):
    out = []
    rx = re.compile(r"\b" + re.escape(short) + r"\b")
    for rel in idx["files"]:
        if not is_test(rel):
            continue
        try:
            src = open(os.path.join(root, rel), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        n = len(rx.findall(src))
        if n:
            out.append((n, rel))
    out.sort(reverse=True)
    return out[:k]


def cmd_trace(root, idx, name, tests):
    hits = resolve(idx, name, tests)
    if not hits:
        return no_match(name)
    q = hits[0]
    d = idx["by_q"][q]
    emit(f"DEFINITION  {q}  [{d['kind']}]  {loc(d)}")
    emit(f"  {d['sig']}" + (f"   # {d['doc']}" if d["doc"] else ""))
    if len(hits) > 1:
        emit("  other matches: " + ", ".join(hits[1:5]))
    if d["kind"] == "class":
        meths = [x for x in idx["defs"] if x["cls"] == q][:15]
        emit("  methods: " + ", ".join(f"{m['short']}:{m['start']}" for m in meths))
        sub = [x["q"] for x in idx["defs"] if x["kind"] == "class" and d["short"] in x["bases"]][:6]
        if sub:
            emit("  subclasses: " + ", ".join(sub))
    allc = idx["rev"].get(q, [])
    callers = short_list(allc, 12, tests)
    ntest = len({c for c, rel, _ in allc if is_test(rel)})
    emit(f"CALLERS ({len(callers)} in source" + (f"; {ntest} test functions also call it" if ntest and not tests else "") + ")")
    for c, rel, line in callers:
        emit(f"  {c}  @ {rel}:{line}")
    callees = short_list(idx["fwd"].get(q, []), 14)
    emit(f"CALLEES ({len(idx['fwd'].get(q, []))})")
    for c, rel, line in callees:
        cd = idx["by_q"].get(c)
        emit(f"  {c}  def {loc(cd) if cd else '?'}  (called at line {line})")
    sims = similar_list(idx, q, 5)
    if sims:
        emit("OVERRIDES / SIMILAR")
        for s, why in sims:
            emit(f"  {s}  {loc(idx['by_q'][s])}  ({why})")
    t = tests_for(root, idx, d["short"])
    emit("TESTS  " + (", ".join(f"{rel} ({n})" for n, rel in t) if t else "none reference it by name"))


def cmd_callers(idx, name, tests):
    hits = resolve(idx, name, tests)
    if not hits:
        return no_match(name)
    q = hits[0]
    emit(f"callers of {q}  ({loc(idx['by_q'][q])})")
    lvl1 = short_list(idx["rev"].get(q, []), 15, tests)
    if not lvl1:
        emit("  none found statically (may be used via a decorator, getattr, callback, or only from tests)")
    for c, rel, line in lvl1:
        emit(f"  <- {c}  @ {rel}:{line}")
        for c2, rel2, line2 in short_list(idx["rev"].get(c, []), 3, tests):
            emit(f"       <- {c2}  @ {rel2}:{line2}")


def cmd_callees(idx, name, tests):
    hits = resolve(idx, name, tests)
    if not hits:
        return no_match(name)
    q = hits[0]
    emit(f"{q}  ({loc(idx['by_q'][q])}) calls:")
    for c, rel, line in short_list(idx["fwd"].get(q, []), 25):
        cd = idx["by_q"].get(c)
        emit(f"  -> {c}  def {loc(cd) if cd else '?'}  (line {line})")
    if not idx["fwd"].get(q):
        emit("  (no calls to functions defined in this repository)")


def bfs(idx, srcs, dsts, graph, max_depth=7):
    dsts = set(dsts)
    prev = {s: None for s in srcs}
    frontier = list(srcs)
    for _ in range(max_depth):
        nxt = []
        for u in frontier:
            for v, rel, line in idx[graph].get(u, []):
                if v not in prev:
                    prev[v] = (u, rel, line)
                    if v in dsts:
                        chain, cur = [], v
                        while prev[cur] is not None:
                            p, r, l = prev[cur]
                            chain.append((p, cur, r, l))
                            cur = p
                        return list(reversed(chain))
                    nxt.append(v)
        frontier = nxt
    return None


def cmd_path(idx, a, b, tests):
    A, B = resolve(idx, a, True)[:6], resolve(idx, b, True)[:6]
    if not A or not B:
        return no_match(a if not A else b)
    chain = bfs(idx, A, B, "fwd")
    if chain:
        emit(f"call chain {a} -> {b}:")
        for p, c, rel, line in chain:
            emit(f"  {p}  --calls at {rel}:{line}-->  {c}")
        return
    chain = bfs(idx, B, A, "fwd")
    if chain:
        emit(f"no path {a} -> {b}, but {b} reaches {a}:")
        for p, c, rel, line in chain:
            emit(f"  {p}  --calls at {rel}:{line}-->  {c}")
        return
    ra = {x[0] for x in idx["rev"].get(A[0], [])}
    rb = {x[0] for x in idx["rev"].get(B[0], [])}
    common = sorted(ra & rb)[:5]
    emit(f"no static call chain between {A[0]} and {B[0]}" +
         (f"; both are called by: {', '.join(common)}" if common else ". Try `callers` on each."))


GENERIC = {"__init__", "__repr__", "__str__", "__call__", "__eq__", "__hash__", "__iter__", "__len__",
           "__getitem__", "__enter__", "__exit__", "__aenter__", "__aexit__", "__rich__", "__rich_console__",
           "__rich_measure__", "main", "run", "setup"}


def similar_list(idx, q, k):
    d = idx["by_q"][q]
    mod = d["q"].rsplit(".", 2 if d["cls"] else 1)[0]
    my_bases = set(idx["by_q"][d["cls"]]["bases"]) if d["cls"] in idx["by_q"] else set()
    out = []
    for o in idx["defs"]:
        if o["q"] == q or o["test"] or o["kind"] != d["kind"] or o["short"] != d["short"]:
            continue
        same_mod = o["rel"] == d["rel"]
        ob = set(idx["by_q"][o["cls"]]["bases"]) if o["cls"] in idx["by_q"] else set()
        sibling = bool(my_bases & ob) or (d["cls"] and idx["by_q"][o["cls"]]["short"] in my_bases if o["cls"] in idx["by_q"] else False)
        if d["short"] in GENERIC and not (same_mod or sibling):
            continue
        score = 3 if same_mod else 2 if sibling else 1
        out.append((score, o["q"], "same name, same file" if same_mod else "same name, sibling class" if sibling
                    else "same name elsewhere"))
    if d["kind"] != "class" and d["idents"]:
        a = set(d["idents"])
        for o in idx["defs"]:
            if o["q"] == q or o["test"] or o["kind"] == "class" or not o["idents"]:
                continue
            b = set(o["idents"])
            j = len(a & b) / max(1, len(a | b))
            if j >= 0.45:
                out.append((j, o["q"], f"shares {int(j * 100)}% of identifiers"))
    out.sort(key=lambda x: -x[0])
    seen, res = set(), []
    for s_, oq, why in out:
        if oq not in seen:
            seen.add(oq)
            res.append((oq, why))
    return res[:k]


def cmd_similar(idx, name, tests):
    hits = resolve(idx, name, tests)
    if not hits:
        return no_match(name)
    q = hits[0]
    res = similar_list(idx, q, 10)
    emit(f"places similar to {q}  ({loc(idx['by_q'][q])}):")
    for oq, why in res:
        emit(f"  {oq}  {loc(idx['by_q'][oq])}  ({why})")
    if not res:
        emit("  none found")


def cmd_tests(root, idx, name):
    short = name.strip().split(".")[-1]
    t = tests_for(root, idx, short, 8)
    if not t:
        emit(f"no test file mentions {short!r}")
        return
    for n, rel in t:
        fns = [d["short"] for d in idx["defs"] if d["rel"] == rel and d["short"].startswith("test")]
        emit(f"{rel}  ({n} mentions; {len(fns)} tests)  run: python -m pytest {rel} -x -q")


def cmd_outline(root, idx, rel):
    rel = rel.replace("/workspace/", "").lstrip("./")
    ds = sorted((d for d in idx["defs"] if d["rel"] == rel), key=lambda d: d["start"])
    if not ds:
        emit(f"no definitions indexed for {rel}")
        return
    emit(f"module {module_name(rel)}")
    for d in ds[:80]:
        depth = d["q"].count(".") - module_name(rel).count(".") - 1
        emit(f"{'  ' * depth}{d['short']}  {d['start']}-{d['end']}")


def main(argv):
    args = [a for a in argv if a != "--"]
    tests = "--tests" in args
    args = [a for a in args if a != "--tests"]
    if not args or args[0] not in ("search", "show", "find", "trace", "callers", "callees", "path", "similar",
                                   "tests", "outline", "status"):
        print(__doc__)
        return
    cmd = args[0]
    root = repo_root()
    key = state_key(root)
    n = bump_counter(key, cmd)
    if cmd == "status":
        print(f"code-map calls used: {n} of {BUDGET}")
        return
    if n > BUDGET:
        print(f"LOCALIZER BUDGET REACHED ({BUDGET} code-map calls). Do not call any more tools. "
              "Write your final diagnosis now, using what you have found.")
        return
    idx = load_index(root, key)
    try:
        if cmd == "search" and len(args) >= 2:
            cmd_search(root, idx, " ".join(args[1:]), tests)
        elif cmd == "show" and len(args) >= 4:
            cmd_show(root, idx, args[1], args[2], args[3])
        elif cmd == "show" and len(args) == 3 and "-" in args[2]:
            a, b = args[2].split("-", 1)
            cmd_show(root, idx, args[1], a, b)
        elif cmd == "find" and len(args) >= 2:
            cmd_find(idx, args[1], tests)
        elif cmd == "trace" and len(args) >= 2:
            cmd_trace(root, idx, args[1], tests)
        elif cmd == "callers" and len(args) >= 2:
            cmd_callers(idx, args[1], tests)
        elif cmd == "callees" and len(args) >= 2:
            cmd_callees(idx, args[1], tests)
        elif cmd == "path" and len(args) >= 3:
            cmd_path(idx, args[1], args[2], tests)
        elif cmd == "similar" and len(args) >= 2:
            cmd_similar(idx, args[1], tests)
        elif cmd == "tests" and len(args) >= 2:
            cmd_tests(root, idx, args[1])
        elif cmd == "outline" and len(args) >= 2:
            cmd_outline(root, idx, args[1])
        else:
            emit(f"bad arguments for {cmd!r}.")
            emit(__doc__)
    except Exception as e:  # noqa: BLE001
        emit(f"code-map error: {type(e).__name__}: {e}")
    if BUDGET - n <= 4:
        emit(f"[code-map: {BUDGET - n} calls left before you must write the diagnosis]")
    flush()


if __name__ == "__main__":
    main(sys.argv[1:])
