"""
K3c sibling-separation pilot (BMM Proposition 1), CPU only: no LLM, no traces, no tests run.
Spec: research_notes/Code graph comprehension architecture/architecture_v3_final.md, section 0.6.3, eq. (8), and the kill table.

Question: when names cannot tell the culprit function from a name-similar sibling, does the behaviour an issue states
(signed effects: reported E+ and expected-but-absent E-) separate them, using static AST effect probabilities q only?

PRE-REGISTERED (fixed before any data was seen, 9 Oct 2026)
  Population : SWE-bench Verified U Lite (dedup by instance_id), traceback-free issues, <=40 per repo (first 40 by
               instance_id), >=1 gold function (an existing function whose body the patch changes), >=1 regex effect code.
  Pairs      : each gold function x its top-5 non-gold siblings by name similarity, drawn from the same file or the same
               short name (family) in the same top-level package; test files excluded.
  Condition  : (8) at beta = 1, mu' = 0.1, eta = 0.05 (no tuning):
               D+ + D-NEG + D-PEER > mu' * sum_{j not in E} (q*_j - q'_j)
               D-PEER only when the sibling shares peers (same file).
  Metric     : sibling-separation rate = mean over pairs of 1[holds] (+0.5 for an exact tie, so chance = 0.5).
               95% CI from a repo-clustered bootstrap (2,000 resamples).
  Decision   : >= 0.60 proceed | 0.55-0.60 proceed, C-BEH flagged at risk | <= 0.55 C-BEH demoted before any GPU spend.
  Also reported (descriptive): strict rate, tie share, rate on untied pairs, by polarity, per repo, exclusions.
Simplifications vs the full model (stated, not tuned): AST q only (no learned g_phi, no 2-hop exception propagation),
  regex issue effects (no LLM, no span verifier beyond the regex match itself).

Kaggle: CPU, Internet ON. Output: /kaggle/working/k3c_result.json, k3c_pairs.csv, k3c_issues.csv.
"""
import ast
import builtins
import difflib
import io
import json
import math
import os
import pathlib
import random
import re
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict

W = pathlib.Path(os.environ.get("K3C_OUT", "/kaggle/working"))
REPOS = pathlib.Path(os.environ.get("K3C_REPOS", "/tmp/k3c_repos"))
W.mkdir(parents=True, exist_ok=True)
REPOS.mkdir(parents=True, exist_ok=True)
BETA, MU, ETA = 1.0, 0.1, 0.05
PER_REPO, TOPK, BOOT = 40, 5, 2000
LOW, MID, HIGH = 0.05, 0.6, 0.95
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ effect alphabet
BUILTIN_EXC = {n: c for n, c in vars(builtins).items() if isinstance(c, type) and issubclass(c, BaseException)}
FIXED_BITS = ["returns_none", "returns_self", "mutates_arg", "writes_self", "warns",
              "raise:KeyError", "raise:IndexError", "raise:AttributeError", "raise:TypeError"]
MUTATORS = {"append", "extend", "update", "pop", "remove", "insert", "clear", "setdefault", "add", "discard",
            "popitem", "sort", "reverse"}


def builtin_ancestors(name):
    c = BUILTIN_EXC.get(name)
    return [k.__name__ for k in c.__mro__ if k.__name__ in BUILTIN_EXC and k is not object] if c else [name]


def exc_closure(name, hier):
    """name + all ancestors (repo class hierarchy by simple name, then builtin MRO)."""
    out, todo, seen = [], [name], set()
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
        todo += hier.get(n, [])
        if n in BUILTIN_EXC:
            todo += builtin_ancestors(n)[1:]
    return out


# ------------------------------------------------------------------ issue effects (regex)
EXC_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Error|Exception|Warning))\b")
NEG_BEFORE = re.compile(r"(should(?: have)?|expected(?: [\w\s]{0,30}?)? to|must|ought to|doesn'?t|does not|did not|didn'?t|"
                        r"never|fails? to|no|without|instead of)\s+(?:[\w`'\"]+\s+){0,4}$", re.I)
NEG_AFTER = re.compile(r"^\W{0,3}(?:is|was)?\s*(?:not|never) (?:raised|thrown)", re.I)


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]


def issue_effects(text):
    """-> (E+, E-) as sets of effect codes, plus the matched spans (the 'span check' is the regex match itself)."""
    ep, em, spans = set(), set(), []
    for s in sentences(text):
        for m in EXC_RE.finditer(s):
            name = m.group(1)
            code = "warns" if name.endswith("Warning") else f"raise:{name}"
            before, after = s[max(0, m.start() - 80):m.start()], s[m.end():m.end() + 40]
            neg = bool(NEG_BEFORE.search(before)) or bool(NEG_AFTER.search(after))
            (em if neg else ep).add(code)
            spans.append((("-" if neg else "+") + code, s[max(0, m.start() - 40):m.end() + 20]))
        low = s.lower()
        if re.search(r"\breturns? `?none`?\b", low):
            neg = bool(re.search(r"(should|expected to|must)\s+(?:\w+\s+){0,2}returns? `?none", low))
            (em if neg else ep).add("returns_none")
            spans.append((("-" if neg else "+") + "returns_none", s[:120]))
        if re.search(r"\b(mutates?|modif(?:y|ies)|in[- ]place|changes the (?:original|input))\b", low):
            ep.add("mutates_arg")
            spans.append(("+mutates_arg", s[:120]))
        if re.search(r"\b(should|expected to)\s+(?:\w+\s+){0,2}warn", low):
            em.add("warns")
            spans.append(("-warns", s[:120]))
    em -= ep & em  # contradictory mentions in one issue: keep the observed reading
    return ep, em, spans


# ------------------------------------------------------------------ AST: functions, effects, hierarchy
def is_test_path(p):
    parts = p.split("/")
    name = parts[-1]
    return (any(x in ("tests", "test", "testing") for x in parts[:-1]) or name.startswith("test_")
            or name.endswith("_test.py") or name == "conftest.py")


class Funcs(ast.NodeVisitor):
    def __init__(self):
        self.out, self.stack, self.classes = [], [], {}

    def visit_ClassDef(self, n):
        self.classes[n.name] = [b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else ""
                                for b in n.bases]
        self.stack.append(n.name)
        self.generic_visit(n)
        self.stack.pop()

    def _fn(self, n):
        self.out.append((".".join(self.stack + [n.name]), n, bool(self.stack)))
        self.stack.append(n.name)
        self.generic_visit(n)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = _fn


def own_nodes(fn):
    """walk fn's body without descending into nested defs/classes/lambdas."""
    todo = list(fn.body)
    while todo:
        n = todo.pop()
        yield n
        for c in ast.iter_child_nodes(n):
            if not isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                todo.append(c)


def root_name(e):
    while isinstance(e, (ast.Attribute, ast.Subscript)):
        e = e.value
    return e.id if isinstance(e, ast.Name) else None


def effects_q(fn, hier):
    """static effect probabilities q_vj for one function (rules from architecture v3.1, eq. (6) q^AST)."""
    a = fn.args
    params = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs} | ({a.vararg.arg} if a.vararg else set()) \
        | ({a.kwarg.arg} if a.kwarg else set())
    params.discard("self")
    params.discard("cls")
    optional = set()
    for arg, d in zip((a.posonlyargs + a.args)[-len(a.defaults):] if a.defaults else [], a.defaults):
        if isinstance(d, ast.Constant) and d.value is None:
            optional.add(arg.arg)
    q = defaultdict(lambda: LOW)
    has_value_return = has_bare_return = False
    getter_roots = set()
    for n in own_nodes(fn):
        if isinstance(n, ast.Raise) and n.exc is not None:
            e = n.exc.func if isinstance(n.exc, ast.Call) else n.exc
            name = e.id if isinstance(e, ast.Name) else e.attr if isinstance(e, ast.Attribute) else None
            if name:
                for anc in exc_closure(name, hier):
                    q[f"raise:{anc}"] = HIGH
        elif isinstance(n, ast.Return):
            if n.value is None or (isinstance(n.value, ast.Constant) and n.value.value is None):
                has_bare_return = True
            else:
                has_value_return = True
                if isinstance(n.value, ast.Name) and n.value.id == "self":
                    q["returns_self"] = HIGH
        elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = n.targets if isinstance(n, (ast.Assign, ast.Delete)) else [n.target]
            for t in targets:
                if isinstance(t, (ast.Attribute, ast.Subscript)):
                    r = root_name(t)
                    if r == "self" and isinstance(t, ast.Attribute):
                        q["writes_self"] = HIGH
                    elif r in params:
                        q["mutates_arg"] = HIGH
        elif isinstance(n, ast.Call):
            f = n.func
            fname = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
            if fname == "warn":
                q["warns"] = HIGH
            if isinstance(f, ast.Attribute):
                r = root_name(f.value)
                if fname in MUTATORS and r in params:
                    q["mutates_arg"] = HIGH
                if fname == "get" and r:
                    getter_roots.add(r)
        if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Load):
            r = root_name(n.value)
            if r in params:
                q["raise:KeyError"] = max(q["raise:KeyError"], MID)
                q["raise:IndexError"] = max(q["raise:IndexError"], MID)
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load) and isinstance(n.value, ast.Name) \
                and n.value.id in optional:
            q["raise:AttributeError"] = max(q["raise:AttributeError"], MID)
    if has_bare_return or not has_value_return:
        q["returns_none"] = HIGH
    for k in ("raise:KeyError", "raise:IndexError"):  # parent classes of the implicit bits
        if q[k] >= MID:
            for anc in builtin_ancestors(k.split(":")[1])[1:]:
                q[f"raise:{anc}"] = max(q[f"raise:{anc}"], q[k])
    return dict(q)


# ------------------------------------------------------------------ data
def hf_parquet_urls(ds):
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{ds}/parquet", timeout=60) as r:
        meta = json.loads(r.read())
    urls = []
    for cfg in meta.values():
        for split, us in cfg.items():
            if split == "test":
                urls += us
    return urls


def load_tasks():
    import pandas as pd

    frames = []
    for ds in ("princeton-nlp/SWE-bench_Verified", "princeton-nlp/SWE-bench_Lite"):
        for u in hf_parquet_urls(ds):
            with urllib.request.urlopen(u, timeout=120) as r:
                frames.append(pd.read_parquet(io.BytesIO(r.read()))[["instance_id", "repo", "base_commit", "patch",
                                                                      "problem_statement"]])
    df = pd.concat(frames).drop_duplicates("instance_id").sort_values("instance_id")
    return df.to_dict("records")


def git(repo_dir, *args, binary=False):
    r = subprocess.run(["git", "-C", str(repo_dir), *args], capture_output=True, timeout=600)
    if r.returncode != 0:
        return None
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def ensure_repo(repo):
    d = REPOS / repo.replace("/", "__")
    if not (d / ".git").exists() and not (d / "HEAD").exists():
        t = time.time()
        subprocess.run(["git", "clone", "--quiet", "--no-checkout", f"https://github.com/{repo}.git", str(d)],
                       check=True, timeout=3600)
        log(f"cloned {repo} in {time.time() - t:.0f}s")
    return d


def changed_old_lines(patch):
    """{path: set(old line numbers touched)}; insertions anchor to the old line before them."""
    out, path, old = defaultdict(set), None, 0
    for line in patch.splitlines():
        if line.startswith("--- "):
            path = line[4:].strip()
            path = path[2:] if path.startswith("a/") else None if path == "/dev/null" else path
        elif line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+", line)
            old = int(m.group(1)) if m else 0
        elif path is None or line.startswith("+++"):
            continue
        elif line.startswith("-"):
            out[path].add(old)
            old += 1
        elif line.startswith("+"):  # insertion: anchor to the old line just before it (never the next one)
            out[path].add(max(old - 1, 1))
        else:
            old += 1
    return out


def index_commit(repo_dir, commit, cache={}):
    """parse every non-test .py file at the commit: functions (with nodes), class hierarchy."""
    key = (str(repo_dir), commit)
    if key in cache:
        return cache[key]
    files = [p for p in (git(repo_dir, "ls-tree", "-r", "--name-only", commit) or "").splitlines()
             if p.endswith(".py") and not is_test_path(p)]
    funcs, hier, by_short, parse_fail = {}, {}, defaultdict(list), 0
    for p in files:
        src = git(repo_dir, "show", f"{commit}:{p}")
        if src is None:
            continue
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError):
            parse_fail += 1
            continue
        v = Funcs()
        v.visit(tree)
        for name, node, is_method in v.out:
            fid = f"{p}::{name}"
            funcs[fid] = (p, name, node)
            by_short[name.rsplit(".", 1)[-1]].append(fid)
        for c, bases in v.classes.items():
            hier.setdefault(c, [b for b in bases if b])
    cache.clear()  # keep one commit in memory
    cache[key] = (funcs, hier, by_short, parse_fail)
    return cache[key]


def name_sim(a, b):
    sa, sb = a.rsplit(".", 1)[-1].lower(), b.rsplit(".", 1)[-1].lower()
    ta, tb = set(re.split(r"[_\W]+", sa)) - {""}, set(re.split(r"[_\W]+", sb)) - {""}
    jac = len(ta & tb) / max(1, len(ta | tb))
    return 0.5 * difflib.SequenceMatcher(None, sa, sb).ratio() + 0.5 * jac


# ------------------------------------------------------------------ condition (8)
def margin(qs, qp, ep, em, peers_shared):
    """LHS - RHS of eq. (8) for culprit q* = qs vs sibling q' = qp (beta = 1)."""
    g = lambda q, j: q.get(j, LOW)
    lhs = sum(math.log(((1 - ETA) * g(qs, j) + ETA) / ((1 - ETA) * g(qp, j) + ETA)) for j in ep)
    lhs += sum(math.log((1 - g(qs, j) + ETA) / (1 - g(qp, j) + ETA)) for j in em)
    if peers_shared:
        lhs += sum(math.log((g(qp, j) + ETA) / (g(qs, j) + ETA)) for j in em)
    stated = ep | em
    alphabet = (set(FIXED_BITS) | set(qs) | set(qp)) - stated
    rhs = MU * sum(g(qs, j) - g(qp, j) for j in alphabet)
    return BETA * lhs - rhs


def main():
    tasks = load_tasks()
    log(f"{len(tasks)} tasks (Verified U Lite)")
    excl = defaultdict(int)
    per_repo = defaultdict(int)
    pairs, issues = [], []
    for t in tasks:
        text = t["problem_statement"] or ""
        if "Traceback (most recent call last)" in text:
            excl["traceback"] += 1
            continue
        if per_repo[t["repo"]] >= PER_REPO:
            excl["repo_cap"] += 1
            continue
        ep, em, spans = issue_effects(text)
        if not (ep or em):
            excl["no_effect_code"] += 1
            continue
        per_repo[t["repo"]] += 1
        rd = ensure_repo(t["repo"])
        funcs, hier, by_short, parse_fail = index_commit(rd, t["base_commit"])
        if not funcs:
            excl["repo_index_failed"] += 1
            continue
        # gold = innermost function containing a changed old line
        gold, nonfunc = set(), 0
        by_path = defaultdict(list)
        for fid, (p, name, node) in funcs.items():
            by_path[p].append((fid, node))
        for p, lines in changed_old_lines(t["patch"]).items():
            for ln in lines:
                hits = [(node.end_lineno - node.lineno, fid) for fid, node in by_path.get(p, [])
                        if node.lineno <= ln <= (node.end_lineno or node.lineno)]
                if hits:
                    gold.add(min(hits)[1])
                else:
                    nonfunc += 1
        if not gold:
            excl["no_function_gold"] += 1
            continue
        # resolve repo exception names in the issue through the hierarchy so q and E speak the same codes
        n_pairs = 0
        for gid in sorted(gold):
            gp, gname, gnode = funcs[gid]
            short = gname.rsplit(".", 1)[-1]
            top = gp.split("/")[0]
            cands = {fid for fid, _ in by_path[gp]} | {fid for fid in by_short.get(short, [])
                                                       if funcs[fid][0].split("/")[0] == top}
            cands -= gold
            ranked = sorted(cands, key=lambda f: (-name_sim(gname, funcs[f][1]), f))[:TOPK]
            qs = effects_q(gnode, hier)
            for sid in ranked:
                sp, sname, snode = funcs[sid]
                qp = effects_q(snode, hier)
                m = margin(qs, qp, ep, em, peers_shared=(sp == gp))
                tie = abs(m) < 1e-9
                pairs.append({"instance_id": t["instance_id"], "repo": t["repo"], "gold": gid, "sibling": sid,
                              "same_file": sp == gp, "sim": round(name_sim(gname, sname), 3),
                              "polarity": "+" if ep and not em else "-" if em and not ep else "+-",
                              "margin": round(m, 4), "holds": float(m > 1e-9) + 0.5 * tie, "tie": tie})
                n_pairs += 1
        issues.append({"instance_id": t["instance_id"], "repo": t["repo"], "E+": sorted(ep), "E-": sorted(em),
                       "spans": spans[:6], "n_gold": len(gold), "nonfunc_changed_lines": nonfunc,
                       "n_pairs": n_pairs, "parse_fail_files": parse_fail})
        if len(issues) % 25 == 0:
            log(f"{len(issues)} issues, {len(pairs)} pairs, {time.time() - T0:.0f}s")

    # ---------------------------------------------------------------- statistics
    def rate(ps):
        return sum(p["holds"] for p in ps) / len(ps) if ps else float("nan")

    by_repo = defaultdict(list)
    for p in pairs:
        by_repo[p["repo"]].append(p)
    repos = sorted(by_repo)
    rng = random.Random(0)
    boots = []
    for _ in range(BOOT):
        sample = [p for r in (rng.choice(repos) for _ in repos) for p in by_repo[r]]
        boots.append(rate(sample))
    boots.sort()
    sep = rate(pairs)
    untied = [p for p in pairs if not p["tie"]]
    decision = ("proceed" if sep >= 0.60 else "proceed, C-BEH flagged at risk" if sep > 0.55
                else "C-BEH demoted to secondary before any GPU spend (R3/R6)")
    res = {
        "spec": {"beta": BETA, "mu_prime": MU, "eta": ETA, "topk_siblings": TOPK, "per_repo_cap": PER_REPO,
                 "q_levels": [LOW, MID, HIGH], "tie_rule": "0.5", "bootstrap": f"repo-clustered, {BOOT}"},
        "n_tasks_total": len(tasks), "exclusions": dict(excl), "n_issues": len(issues), "n_pairs": len(pairs),
        "n_repos": len(repos),
        "sibling_separation_rate": round(sep, 4),
        "ci95_repo_clustered": [round(boots[int(0.025 * BOOT)], 4), round(boots[int(0.975 * BOOT) - 1], 4)],
        "strict_rate": round(sum(p["holds"] == 1.0 for p in pairs) / max(1, len(pairs)), 4),
        "tie_share": round(sum(p["tie"] for p in pairs) / max(1, len(pairs)), 4),
        "rate_untied": round(rate(untied), 4), "n_untied": len(untied),
        "by_polarity": {k: {"n": len(v), "rate": round(rate(v), 4)} for k in ("+", "-", "+-")
                        for v in [[p for p in pairs if p["polarity"] == k]]},
        "by_same_file": {str(k): {"n": len(v), "rate": round(rate(v), 4)} for k in (True, False)
                         for v in [[p for p in pairs if p["same_file"] == k]]},
        "by_repo": {r: {"n": len(by_repo[r]), "rate": round(rate(by_repo[r]), 4)} for r in repos},
        "effect_code_counts": dict(sorted(((c, sum(c in (i["E+"] + i["E-"]) for i in issues))
                                           for c in {c for i in issues for c in i["E+"] + i["E-"]}),
                                          key=lambda kv: -kv[1])[:25]),
        "decision": decision,
        "minutes": round((time.time() - T0) / 60, 1),
    }
    (W / "k3c_result.json").write_text(json.dumps(res, indent=1))
    import csv

    for name, rows in (("k3c_pairs.csv", pairs), ("k3c_issues.csv", issues)):
        if rows:
            with open(W / name, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
    log("RESULT", json.dumps({k: res[k] for k in ("n_issues", "n_pairs", "sibling_separation_rate",
                                                     "ci95_repo_clustered", "tie_share", "rate_untied", "decision")}))


if __name__ == "__main__":
    main()
