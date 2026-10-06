"""code-map: structural lookup over the repository using Python's ast (stdlib only, no graph files needed).

Commands (first positional argument):
  find NAME      definitions whose name matches NAME (bare `solve_dependencies`, `Class.method`, or a dotted path)
                 -> qualified graph name, file:line-range, signature
  callers NAME   call sites of NAME -> file:line, enclosing function, the source line
  callees NAME   names called inside the definition of NAME
  outline PATH   classes/functions of one file with line ranges

Options: --tests  include test files (excluded by default)   --limit N  max rows (default 25)
The qualified names printed (e.g. `fastapi.dependencies.utils.solve_dependencies`) are exactly the node names
that get_code_neighbors / search_similar_code / get_code_subgraph accept.
"""
import ast
import inspect
import os
import subprocess
import sys

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "build", "dist", "__pycache__", ".tox", ".eggs", "site-packages", "docs", "docs_src"}
MAX_FILES = 4000


def repo_root():
    for fr in inspect.stack():
        cwd = fr.frame.f_locals.get("_orig_cwd")
        if isinstance(cwd, str) and os.path.isdir(cwd):
            return git_top(cwd)
    for cand in (os.environ.get("CODE_MAP_ROOT"), "/workspace", os.environ.get("PWD"), os.getcwd()):
        if cand and os.path.isdir(cand) and not cand.startswith(("/tmp/tmp", "/var/folders")):
            return git_top(cand)
    return os.getcwd()


def git_top(path):
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path, capture_output=True, text=True,
                             timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return path


def is_test(rel):
    parts = rel.replace("\\", "/").split("/")
    base = parts[-1]
    return any(p in ("tests", "test", "testing") for p in parts[:-1]) or base.startswith("test_") or \
        base.endswith("_test.py") or base == "conftest.py"


def iter_py(root, tests):
    n = 0
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in SKIP_DIRS and not d.startswith("."))
        for fn in sorted(fns):
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, root)
            if not tests and is_test(rel):
                continue
            n += 1
            if n > MAX_FILES:
                return
            yield full, rel


def module_name(rel):
    mod = rel[:-3].replace(os.sep, ".").replace("/", ".")
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    for prefix in ("src.", "lib."):
        if mod.startswith(prefix):
            mod = mod[len(prefix):]
    return mod


def parse(full):
    try:
        with open(full, encoding="utf-8", errors="replace") as f:
            src = f.read()
        return ast.parse(src), src.splitlines()
    except Exception:  # noqa: BLE001
        return None, []


def defs(tree, mod):
    """Yield (qualname_without_module, node) for every class/function, with nesting."""
    stack = [(tree, "")]
    while stack:
        node, prefix = stack.pop()
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{prefix}.{ch.name}" if prefix else ch.name
                yield q, ch
                stack.append((ch, q))
            elif not isinstance(ch, (ast.expr,)):
                stack.append((ch, prefix))


def matches(query, mod, q):
    full = f"{mod}.{q}"
    return full == query or full.endswith("." + query) or q == query or q.split(".")[-1] == query


def signature(node, lines):
    line = lines[node.lineno - 1].strip() if node.lineno - 1 < len(lines) else ""
    return line[:160]


def call_name(call):
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def cmd_find(root, query, tests, limit):
    rows = []
    for full, rel in iter_py(root, tests):
        tree, lines = parse(full)
        if tree is None:
            continue
        mod = module_name(rel)
        for q, node in defs(tree, mod):
            if matches(query, mod, q):
                kind = "class" if isinstance(node, ast.ClassDef) else "def"
                rows.append(f"{mod}.{q}  [{kind}]  {rel}:{node.lineno}-{getattr(node, 'end_lineno', node.lineno)}"
                            f"  | {signature(node, lines)}")
    if not rows:
        print(f"no definition named {query!r}. Try a shorter name, or grep -rn \"{query.split('.')[-1]}\" .")
    for r in rows[:limit]:
        print(r)
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more")


def enclosing_map(tree):
    spans = []
    for q, node in defs(tree, ""):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((node.lineno, getattr(node, "end_lineno", node.lineno), q))
    return spans


def enclosing(spans, lineno):
    best = None
    for s, e, q in spans:
        if s <= lineno <= e and (best is None or s >= best[0]):
            best = (s, e, q)
    return best[2] if best else "<module>"


def cmd_callers(root, query, tests, limit):
    target = query.split(".")[-1]
    rows = []
    for full, rel in iter_py(root, tests):
        tree, lines = parse(full)
        if tree is None:
            continue
        spans = None
        mod = module_name(rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and call_name(node) == target:
                spans = spans if spans is not None else enclosing_map(tree)
                src = lines[node.lineno - 1].strip() if node.lineno - 1 < len(lines) else ""
                rows.append(f"{rel}:{node.lineno}  in {mod}.{enclosing(spans, node.lineno)}  | {src[:140]}")
    if not rows:
        print(f"no call sites of {target!r} found (it may be passed as a value, used via getattr, or only in tests;"
              f" try --tests or grep).")
    for r in rows[:limit]:
        print(r)
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more")


def cmd_callees(root, query, tests, limit):
    for full, rel in iter_py(root, tests):
        tree, lines = parse(full)
        if tree is None:
            continue
        mod = module_name(rel)
        for q, node in defs(tree, mod):
            if matches(query, mod, q) and not isinstance(node, ast.ClassDef):
                seen = []
                for ch in ast.walk(node):
                    if isinstance(ch, ast.Call):
                        nm = call_name(ch)
                        if nm and nm not in seen:
                            seen.append(nm)
                print(f"{mod}.{q}  {rel}:{node.lineno}  calls: {', '.join(seen[:limit * 2]) or '(nothing)'}")
                return
    print(f"no function named {query!r}")


def cmd_outline(root, path, limit):
    full = path if os.path.isabs(path) else os.path.join(root, path)
    tree, lines = parse(full)
    if tree is None:
        print(f"cannot parse {path}")
        return
    mod = module_name(os.path.relpath(full, root))
    out = []
    for q, node in defs(tree, mod):
        depth = q.count(".")
        out.append((node.lineno, f"{'  ' * depth}{q}  {node.lineno}-{getattr(node, 'end_lineno', node.lineno)}"))
    print(f"module {mod}")
    for _, s in sorted(out)[: limit * 4]:
        print(s)


def main(argv):
    args = [a for a in argv if a != "--"]
    tests = "--tests" in args
    limit = 25
    if "--limit" in args:
        i = args.index("--limit")
        try:
            limit = int(args[i + 1])
        except (IndexError, ValueError):
            pass
        del args[i:i + 2]
    args = [a for a in args if a != "--tests"]
    if len(args) < 2 or args[0] not in ("find", "callers", "callees", "outline"):
        print(__doc__)
        return
    cmd, query = args[0], args[1].strip().strip("`'\"()")
    root = repo_root()
    if cmd == "find":
        cmd_find(root, query, tests, limit)
    elif cmd == "callers":
        cmd_callers(root, query, tests, limit)
    elif cmd == "callees":
        cmd_callees(root, query, tests, limit)
    else:
        cmd_outline(root, query, limit)


if __name__ == "__main__":
    main(sys.argv[1:])
