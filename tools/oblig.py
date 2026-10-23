#!/usr/bin/env python3
"""oblig: obligation frontier for code agents. Python standard library only.

After an edit, lists the few other places that probably have to change as well: callers of a
changed signature, code that constructs a changed class, functions that forward the changed
parameters, methods that read fields the edited code writes, and overrides of the edited method.

  python3 oblig.py edit PATH  <<'EOF'      apply ONE exact replacement, then print obligations
  <<<<<<< OLD
  exact existing lines
  =======
  replacement lines
  >>>>>>> NEW
  EOF
  python3 oblig.py check                   obligations left open by the whole `git diff` (run before submit)
  python3 oblig.py show PATH::Qual.name    obligations if that symbol's signature changed

Run from the repository root (or pass --root DIR before the command).
"""
import ast
import math
import os
import re
import subprocess
import sys
from collections import defaultdict

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "build", "dist", "__pycache__", ".tox", ".mypy_cache",
             ".pytest_cache", "site-packages", ".eggs"}
DOC_DIRS = {"docs", "docs_src", "examples", "benchmarks"}
GENERIC = {"get", "set", "add", "pop", "copy", "read", "write", "close", "open", "run", "call", "update", "items",
           "keys", "values", "join", "format", "split", "strip", "append", "extend", "insert", "remove", "clear",
           "send", "start", "stop", "load", "dump", "dumps", "loads", "render", "parse", "init", "main", "setup"}

PLAIN_DUNDERS = {"__init__", "__repr__", "__str__", "__eq__", "__ne__", "__hash__", "__post_init__", "__new__",
                 "__len__", "__iter__", "__bool__", "__lt__", "__le__", "__gt__", "__ge__"}
# Edge weights (fixed a priori on 2026-10-23, before E0a; to be re-fit only on external data, E0b).
RULES = "v2"  # v1 = E0a (Oct 23); v2 adds forwarder->callee (FWDDOWN) and new-read->writer ("reads"), frozen before E0b
W = {"FWD": 1.0, "FWDDOWN": 0.9, "INST": 0.9, "CALLER": 0.85, "FIELD": 0.8, "OVR": 0.75, "FIELDW": 0.55,
     "CALLEE": 0.5, "OVRMOD": 0.5, "FUZZY": 0.3}
# Which edge types each kind of change propagates along.
ALLOW = {
    "sig": {"CALLER", "FWD", "FWDDOWN", "INST", "OVR", "OVRMOD", "FUZZY"},
    "reads": {"FIELDW"},
    "field": {"FIELD", "FIELDW", "INST", "FWD", "CALLER"},
    "contract": {"CALLER", "FWD", "OVR", "OVRMOD", "FUZZY"},
    "new": {"CALLEE", "FIELD", "OVR", "OVRMOD"},
    "body": {"OVR", "OVRMOD", "FIELD", "CALLEE", "CALLER"},
}
HOP_DECAY = 0.6
SAME_FILE, SAME_CLASS = 1.25, 1.25


# ----------------------------------------------------------------------------- indexing

class Def:
    __slots__ = ("key", "path", "qual", "name", "line", "end", "cls", "params", "varkw", "writes", "reads", "calls",
                 "deco", "node")

    def __init__(self, path, qual, node, cls):
        self.key, self.path, self.qual, self.name = f"{path}::{qual}", path, qual, qual.split(".")[-1]
        self.line, self.end, self.cls, self.node = node.lineno, getattr(node, "end_lineno", node.lineno), cls, node
        a = node.args
        self.params = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
        self.varkw = a.kwarg.arg if a.kwarg else None
        self.deco = {d.id if isinstance(d, ast.Name) else getattr(d, "attr", "") for d in node.decorator_list}
        self.writes, self.reads, self.calls = set(), set(), []


def module_of(path):
    p = path[:-3] if path.endswith(".py") else path
    parts = p.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if parts and parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts)


class _Visitor(ast.NodeVisitor):
    def __init__(self, idx, path):
        self.idx, self.path, self.stack, self.cls_stack, self.cur = idx, path, [], [], None

    def visit_ClassDef(self, node):
        qual = ".".join(self.stack + [node.name])
        bases = []
        for b in node.bases:
            if isinstance(b, ast.Subscript):  # Generic[T] / Base[int]
                b = b.value
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(b.attr)
        self.idx.classes[f"{self.path}::{qual}"] = bases
        self.idx.class_by_name[node.name].append(f"{self.path}::{qual}")
        self.stack.append(node.name)
        self.cls_stack.append(f"{self.path}::{qual}")
        prev, self.cur = self.cur, None
        self.generic_visit(node)
        self.cur = prev
        self.cls_stack.pop()
        self.stack.pop()

    def _func(self, node):
        qual = ".".join(self.stack + [node.name])
        cls = self.cls_stack[-1] if self.cls_stack and self.stack and self.cls_stack[-1].endswith("::" + ".".join(self.stack)) else None
        d = Def(self.path, qual, node, cls)
        self.idx.add_def(d)
        self.stack.append(node.name)
        prev, self.cur = self.cur, d
        self.generic_visit(node)
        self.cur = prev
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = _func

    def visit_Attribute(self, node):
        if self.cur is not None and isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
            (self.cur.writes if isinstance(node.ctx, (ast.Store, ast.Del)) else self.cur.reads).add(node.attr)
        self.generic_visit(node)

    def visit_Call(self, node):
        if self.cur is not None:
            f, recv, name = node.func, None, None
            if isinstance(f, ast.Name):
                name = f.id
            elif isinstance(f, ast.Attribute):
                name = f.attr
                v = f.value
                if isinstance(v, ast.Name):
                    recv = v.id
                elif isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "super":
                    recv = "super()"
                else:
                    recv = "?"
            if name:
                params = set(self.cur.params)
                fwd = {k.arg for k in node.keywords if k.arg and isinstance(k.value, ast.Name) and k.value.id == k.arg and k.arg in params}
                star = any(k.arg is None for k in node.keywords)
                self.cur.calls.append((recv, name, fwd, star, node.lineno))
        self.generic_visit(node)


class Index:
    def __init__(self, root):
        self.root = root
        self.defs, self.by_name, self.by_path = {}, defaultdict(list), defaultdict(list)
        self.classes, self.class_by_name = {}, defaultdict(list)
        self.imports = defaultdict(dict)   # path -> local name -> (module, attr or None)
        self.mod2path = {}
        self.tests = defaultdict(set)      # test file -> referenced names
        self.src = {}

    def add_def(self, d):
        self.defs[d.key] = d
        self.by_name[d.name].append(d.key)
        self.by_path[d.path].append(d.key)

    # .....................................................................
    def build(self, overrides=None):
        overrides = overrides or {}
        for dp, dns, fns in os.walk(self.root):
            dns[:] = [x for x in dns if x not in SKIP_DIRS and not x.startswith(".")]
            for fn in fns:
                if not fn.endswith(".py"):
                    continue
                full = os.path.join(dp, fn)
                rel = os.path.relpath(full, self.root).replace(os.sep, "/")
                self._add_file(rel, overrides.get(rel))
        for rel, text in overrides.items():
            if rel not in self.src and text is not None:
                self._add_file(rel, text)
        self._link()
        return self

    def _add_file(self, rel, text=None):
        if text is None:
            try:
                with open(os.path.join(self.root, rel), encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                return
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return
        parts = rel.split("/")
        is_test = any(p in ("tests", "test", "testing") for p in parts[:-1]) or parts[-1].startswith("test_") or parts[-1] == "conftest.py"
        if is_test:
            self.tests[rel] = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
                              {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            return
        if any(p in DOC_DIRS for p in parts[:-1]):
            return
        self.src[rel] = text
        self.mod2path[module_of(rel)] = rel
        mod = module_of(rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    pkg = mod.split(".") if rel.endswith("__init__.py") else mod.split(".")[:-1]
                    pkg = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                    base = ".".join(pkg + ([base] if base else []))
                for a in node.names:
                    self.imports[rel][a.asname or a.name] = (base, a.name)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    self.imports[rel][a.asname or a.name.split(".")[0]] = (a.name if a.asname else a.name.split(".")[0], None)
        _Visitor(self, rel).visit(tree)

    # ..................................................................... resolution
    def _top(self, path, name):
        k = f"{path}::{name}"
        if k in self.defs:
            return k
        if k in self.classes:
            return k
        return None

    def _resolve_import(self, path, name, depth=0):
        imp = self.imports.get(path, {}).get(name)
        if not imp or depth > 2:
            return None
        mod, attr = imp
        if attr is None:
            return ("module", mod)
        if mod + "." + attr in self.mod2path:
            return ("module", mod + "." + attr)
        p = self.mod2path.get(mod)
        if p:
            k = self._top(p, attr)
            if k:
                return ("def", k)
            return self._resolve_import(p, attr, depth + 1)
        return None

    def family(self, cls_key):
        """cls_key plus repo ancestors and descendants (by base-class name)."""
        if cls_key in self._fam_cache:
            return self._fam_cache[cls_key]
        fam, todo = {cls_key}, [cls_key]
        while todo:
            c = todo.pop()
            for b in self.classes.get(c, []):
                for bk in self.class_by_name.get(b, []):
                    if bk not in fam:
                        fam.add(bk)
                        todo.append(bk)
            cname = c.split("::")[1].split(".")[-1]
            for sub, bases in self.classes.items():
                if cname in bases and sub not in fam:
                    fam.add(sub)
                    todo.append(sub)
        self._fam_cache[cls_key] = fam
        return fam

    def _method(self, cls_keys, name):
        out = []
        for c in cls_keys:
            p, q = c.split("::", 1)
            k = f"{p}::{q}.{name}"
            if k in self.defs:
                out.append(k)
        return out

    def _ctor(self, cls_key):
        for c in [cls_key] + [x for x in self.family(cls_key) if x != cls_key]:
            k = f"{c}.__init__"
            if k in self.defs:
                return k
        return cls_key if cls_key in self.defs else None

    def resolve(self, d, recv, name):
        """-> list of (target key, is_instantiation, fuzzy)"""
        if recv is None:
            k = self._top(d.path, name)
            if k is None:
                r = self._resolve_import(d.path, name)
                k = r[1] if r and r[0] == "def" else None
            if k is None:
                cands = [c for c in self.class_by_name.get(name, [])] or \
                        [x for x in self.by_name.get(name, []) if "." not in self.defs[x].qual]
                if len(cands) == 1:
                    k = cands[0]
                elif 1 < len(cands) <= 3 and name not in GENERIC:
                    return [(self._ctor(c) or c, c in self.classes, True) for c in cands]
            if k is None:
                return []
            if k in self.classes:
                c = self._ctor(k)
                return [(c, True, False)] if c else []
            return [(k, False, False)]
        if recv in ("self", "cls") and d.cls:
            ks = self._method(self.family(d.cls), name)
            if ks:
                return [(k, False, False) for k in ks]
            if recv == "cls" and name == "__init__":
                c = self._ctor(d.cls)
                return [(c, True, False)] if c else []
        if recv == "super()" and d.cls:
            ks = self._method([c for c in self.family(d.cls) if c != d.cls], name)
            return [(k, False, False) for k in ks]
        if recv not in (None, "?", "self", "cls", "super()"):
            r = self._resolve_import(d.path, recv)
            if r and r[0] == "module":
                p = self.mod2path.get(r[1])
                k = self._top(p, name) if p else None
                if k:
                    return [(self._ctor(k), True, False)] if k in self.classes else [(k, False, False)]
            ck = self._top(d.path, recv) or (r[1] if r and r[0] == "def" else None)
            if ck and ck in self.classes:
                return [(k, False, False) for k in self._method([ck], name)]
        if name in GENERIC or name.startswith("__") or len(name) <= 3:
            return []
        cands = [x for x in self.by_name.get(name, []) if self.defs[x].cls]
        if 1 <= len(cands) <= 3:
            return [(c, False, True) for c in cands]
        return []

    # ..................................................................... edges
    def _link(self):
        self._fam_cache = {}
        self.rev = defaultdict(list)   # X -> [(Y, type, info)]  : Y depends on X
        self.fwd = defaultdict(list)
        self.indeg = defaultdict(int)
        for d in self.defs.values():
            # the self.__init__ / cls(...) inside class methods
            for recv, name, fwdp, star, line in d.calls:
                if recv == "cls" and name == "cls":
                    continue
                for tgt, inst, fuzzy in self.resolve(d, recv, name):
                    if tgt is None or tgt == d.key or tgt not in self.defs:
                        continue
                    t = "FUZZY" if fuzzy else ("FWD" if (fwdp or star) else ("INST" if inst else "CALLER"))
                    self.rev[tgt].append((d.key, t, line))
                    self.fwd[d.key].append((tgt, "FWDDOWN" if (fwdp or star) and not fuzzy else "CALLEE", line))
                    self.indeg[tgt] += 1
            # cls(...) / type(self)(...) in classmethods construct the class
            if d.cls:
                for recv, name, fwdp, star, line in d.calls:
                    if recv is None and name == "cls":
                        c = self._ctor(d.cls)
                        if c and c in self.defs and c != d.key:
                            self.rev[c].append((d.key, "FWD" if (fwdp or star) else "INST", line))
                            self.fwd[d.key].append((c, "FWDDOWN", line))
                            self.indeg[c] += 1
        # fields and overrides inside class families
        fam_of = {}
        for c in self.classes:
            fam_of[c] = min(self.family(c))
        writers, readers = defaultdict(set), defaultdict(set)
        meth_by_fam = defaultdict(lambda: defaultdict(list))
        for d in self.defs.values():
            if not d.cls:
                continue
            f = fam_of.get(d.cls, d.cls)
            for a in d.writes:
                writers[(f, a)].add(d.key)
            for a in d.reads:
                readers[(f, a)].add(d.key)
            if "." not in d.qual[len(d.cls.split("::")[1]) + 1:]:
                meth_by_fam[f][d.name].append(d.key)
        self.field_of = defaultdict(lambda: defaultdict(set))
        for (f, a), ws in writers.items():
            for w in ws:
                for r in readers.get((f, a), ()):
                    if r != w:
                        self.field_of[w][r].add(a)
        self.fieldw_of = defaultdict(lambda: defaultdict(set))
        for (f, a), rs in readers.items():
            for r in rs:
                for w in writers.get((f, a), ()):
                    if w != r:
                        self.fieldw_of[r][w].add(a)
        self.ovr = defaultdict(set)
        for f, by in meth_by_fam.items():
            for name, ks in by.items():
                if len(ks) > 1:
                    for k in ks:
                        self.ovr[k].update(x for x in ks if x != k)
        # same-name methods of classes in one module that are not related by inheritance
        self.ovrmod = defaultdict(set)
        by_path_name = defaultdict(list)
        for d in self.defs.values():
            if d.cls and d.qual.count(".") == 1 and d.name not in PLAIN_DUNDERS:
                by_path_name[(d.path, d.name)].append(d.key)
        for ks in by_path_name.values():
            if 1 < len(ks) <= 12:
                for k in ks:
                    self.ovrmod[k].update(x for x in ks if x not in self.ovr[k] and x != k)

    def neighbours(self, k):
        for y, t, line in self.rev.get(k, ()):
            yield y, t, line
        for y, t, line in self.fwd.get(k, ()):
            yield y, t, line
        for y, attrs in self.field_of.get(k, {}).items():
            yield y, "FIELD", ",".join(sorted(attrs)[:2])
        for y, attrs in self.fieldw_of.get(k, {}).items():
            yield y, "FIELDW", ",".join(sorted(attrs)[:2])
        for y in self.ovr.get(k, ()):
            yield y, "OVR", None
        for y in self.ovrmod.get(k, ()):
            yield y, "OVRMOD", None


# ----------------------------------------------------------------------------- change typing

def _defs_of(text, path):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {}
    idx = Index(".")
    idx.classes, idx.class_by_name = {}, defaultdict(list)
    _Visitor(idx, path).visit(tree)
    return {d.key: (d, ast.get_source_segment(text, d.node) or "") for d in idx.defs.values()}


def _shape(node, kind):
    return [ast.dump(n) for n in ast.walk(node) if isinstance(n, kind)]


def change_kinds(before_text, after_text, path):
    """{key: (kinds, new_params)} for every function whose source changed between the two texts."""
    b, a = _defs_of(before_text or "", path), _defs_of(after_text or "", path)
    out = {}
    for k, (d, seg) in a.items():
        if k not in b:
            out[k] = ({"new"}, [])
            continue
        d0, seg0 = b[k]
        if seg == seg0:
            continue
        kinds, newp = set(), []
        if d.params != d0.params or d.varkw != d0.varkw:
            kinds.add("sig")
            newp = [p for p in d.params if p not in d0.params]
        if d.writes != d0.writes:
            kinds.add("field")
        if d.reads - d0.reads:
            kinds.add("reads")
        if _shape(d.node, (ast.Return, ast.Raise, ast.Yield)) != _shape(d0.node, (ast.Return, ast.Raise, ast.Yield)):
            kinds.add("contract")
        out[k] = (kinds or {"body"}, newp)
    for k in b:
        if k not in a:
            out[k] = ({"sig"}, [])  # removed or renamed: callers break
    return out


# ----------------------------------------------------------------------------- frontier

def frontier(idx, changed, k=6, depth=2):
    """changed: {key: (kinds, new_params)} -> ranked [(key, score, type, via, info)] excluding changed keys."""
    best = {}
    for src, (kinds, newp) in changed.items():
        if src not in idx.defs:
            continue
        allow = set().union(*(ALLOW[x] for x in kinds))
        frontier_ = [(src, 1.0)]
        for hop in range(depth):
            nxt = []
            for node, s in frontier_:
                for y, t, info in idx.neighbours(node):
                    if t not in allow or y in changed or y not in idx.defs:
                        continue
                    # beyond the first hop only the forwarding chain and fields keep propagating
                    if hop > 0 and t not in ("FWD", "INST", "FIELD", "CALLER"):
                        continue
                    sc = s * W[t] * (HOP_DECAY if hop else 1.0)
                    dy, ds = idx.defs[y], idx.defs[src]
                    if dy.path == ds.path:
                        sc *= SAME_FILE
                    if dy.cls and dy.cls == ds.cls:
                        sc *= SAME_CLASS
                    sc /= math.sqrt(1 + idx.indeg.get(y, 0) / 20)
                    if sc > best.get(y, (0,))[0]:
                        best[y] = (sc, t, node, info, newp)
                    nxt.append((y, s * W[t]))
            frontier_ = nxt
    ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))
    return [(y, v[0], v[1], v[2], v[3], v[4]) for y, v in ranked[:k]]


def tests_for(idx, changed, k=3):
    names = {idx.defs[c].name for c in changed if c in idx.defs} | \
            {idx.defs[c].cls.split("::")[1].split(".")[-1] for c in changed if c in idx.defs and idx.defs[c].cls}
    names = {n for n in names if not n.startswith("__")}
    scored = sorted(((len(names & refs), p) for p, refs in idx.tests.items() if names & refs), reverse=True)
    return [p for _, p in scored[:k]]


REASON = {
    "FWD": "forwards args to {src}{np}",
    "INST": "builds {cls}(...){np}",
    "CALLER": "calls {src}",
    "FUZZY": "may call {src}",
    "FIELD": "reads self.{info} (set in {src})",
    "FIELDW": "sets self.{info} (read in {src})",
    "OVR": "same method in related class",
    "OVRMOD": "same method in sibling class",
    "CALLEE": "called by {src}",
    "FWDDOWN": "receives args forwarded by {src}{np}",
}


def render(idx, items, title, budget=900):
    lines, used = [title], len(title)
    for i, (y, sc, t, src, info, newp) in enumerate(items, 1):
        d, s = idx.defs[y], idx.defs[src]
        np_ = f" (new: {', '.join(newp[:3])})" if newp and t in ("FWD", "FWDDOWN", "INST", "CALLER") else ""
        cls = (s.cls.split("::")[1] if s.cls else s.qual)
        why = REASON[t].format(src=s.qual, np=np_, cls=cls, info=info or "")
        line = f"  {i}. {d.path}:{d.line} {d.qual} - {why}"[:120]
        if used + len(line) + 1 > budget:
            lines.append(f"  (+{len(items) - i + 1} more)")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


# ----------------------------------------------------------------------------- CLI

def _git(root, *args):
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def diff_changes(root):
    out = {}
    names = (_git(root, "diff", "--name-only", "HEAD") or "").split() + \
            (_git(root, "ls-files", "--others", "--exclude-standard") or "").split()
    for p in dict.fromkeys(names):
        if not p.endswith(".py") or "/tests/" in p or p.startswith("tests/"):
            continue
        before = _git(root, "show", f"HEAD:{p}") or ""
        try:
            with open(os.path.join(root, p), encoding="utf-8", errors="replace") as fh:
                after = fh.read()
        except OSError:
            after = ""
        out.update(change_kinds(before, after, p))
    return out


EDIT_RE = re.compile(r"<<<<<<< OLD\n(.*?)\n?=======\n(.*?)\n?>>>>>>> NEW", re.S)


def cmd_edit(root, path, block):
    m = EDIT_RE.search(block)
    if not m:
        return "EDIT FAILED: input must contain <<<<<<< OLD / ======= / >>>>>>> NEW markers"
    old, new = m.group(1), m.group(2)
    full = os.path.join(root, path)
    try:
        with open(full, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as e:
        return f"EDIT FAILED: {e}"
    n = text.count(old)
    if n != 1:
        hint = ""
        first = old.strip().splitlines()[0].strip() if old.strip() else ""
        if first:
            hits = [i + 1 for i, l in enumerate(text.splitlines()) if first in l][:5]
            hint = f"; first OLD line appears at lines {hits}" if hits else "; first OLD line not found"
        return f"EDIT FAILED: OLD text found {n} times in {path} (must be exactly once){hint}. Re-read the lines and copy them exactly."
    after = text.replace(old, new, 1)
    try:
        ast.parse(after)
    except SyntaxError as e:
        return f"EDIT REJECTED: result has a SyntaxError at line {e.lineno}: {e.msg}. File unchanged."
    with open(full, "w", encoding="utf-8") as fh:
        fh.write(after)
    line = text[: text.index(old)].count("\n") + 1
    msg = [f"EDIT OK {path}:{line} (-{old.count(chr(10)) + 1} +{new.count(chr(10)) + 1} lines)"]
    changed = diff_changes(root) or change_kinds(text, after, path)
    here = change_kinds(text, after, path)
    if here:
        idx = Index(root).build()
        items = frontier(idx, changed, k=5)
        what = ", ".join(f"{idx.defs[c].qual if c in idx.defs else c.split('::')[1]} [{'/'.join(sorted(here[c][0]))}]"
                         for c in list(here)[:3])
        msg.append(render(idx, items, f"OBLIGATIONS after editing {what}:") if items else
                   f"No structural obligations found for {what}.")
    return "\n".join(msg)


def cmd_check(root):
    changed = diff_changes(root)
    if not changed:
        return "CHECK: no source changes in git diff yet."
    idx = Index(root).build()
    items = frontier(idx, changed, k=8)
    out = [(f"CHECK: {len(changed)} changed function(s): " + ", ".join(c.split("::")[1] for c in list(changed)[:6]))[:150]]
    out.append(render(idx, items, "Open obligations (address each, or decide it needs no change):", budget=620) if items
               else "No open structural obligations.")
    t = tests_for(idx, changed)
    if t:
        out.append(("Existing tests that use the changed code: " + ", ".join(t))[:150])
    fac = "/tmp/facets.txt"
    if os.path.exists(fac):
        with open(fac) as fh:
            out.append("Issue facets you listed (each needs an edit): " + fh.read().strip()[:200])
    return "\n".join(out)[:1000]


def cmd_show(root, key):
    idx = Index(root).build()
    if key not in idx.defs:
        cands = [k for k in idx.defs if k.endswith(key) or k.split("::")[1] == key][:5]
        if len(cands) != 1:
            return f"unknown symbol {key}; candidates: {cands}"
        key = cands[0]
    items = frontier(idx, {key: ({"sig", "field", "contract"}, [])}, k=8)
    return render(idx, items, f"OBLIGATIONS if {key} changes:")


def main(argv):
    root = "."
    if len(argv) >= 2 and argv[0] == "--root":
        root, argv = argv[1], argv[2:]
    if not argv or argv[0] not in ("edit", "check", "show"):
        print(__doc__)
        return 2
    if argv[0] == "edit" and len(argv) == 2:
        print(cmd_edit(root, argv[1], sys.stdin.read()))
    elif argv[0] == "check":
        print(cmd_check(root))
    elif argv[0] == "show" and len(argv) == 2:
        print(cmd_show(root, argv[1]))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
