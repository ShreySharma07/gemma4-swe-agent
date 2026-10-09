"""
KT-1 (pre-registered in docs/plan_autonomy.md): does repairing the supplied graph's missing intra-class / constructor
edges make the fix reachable from issue-named seeds, beyond random-edge and degree-preserving-rewire controls?

Arms: P provided | R = P + AST-resolved self./cls./super() method calls + constructor->__init__ | X = P + same number of
random edges | D = degree-preserving rewire of R. Metric: share of tasks whose gold node is within <=2 undirected hops
of a clean issue seed, expanding through no hub (top-1% degree in P). Same seeds, hubs, gold for every arm.
PASS: R-P >= +10 pp and R-X >= +5 pp and R-D >= +5 pp. FAIL: R-P < +5 pp. Validation: R holds >= 70% of the
executed same-class + constructor edges seen in the traces.
Light, pure Python. Usage: python3 analysis/kt1_graph_repair.py data/runs/xg
"""
import ast
import json
import random
import re
import sys
import tarfile
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exec_graph_analysis as ega  # noqa: E402

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/runs/xg")
ega.ROOT = ROOT
RNG = random.Random(0)

# ---- clean issue seeds (copied from verify_graph.py / gold_graph_analysis.py so this file needs no numpy)
TEMPLATE_LINE = re.compile(
    r"^\s*([-*]\s*\[[ xX]\]|#{1,6}\s|>\s)|"
    r"(searched|discussion|descriptive title|i already|i used the github search|"
    r"operating system|python version|fastapi version|pydantic version|to reproduce|"
    r"additional context|description|privileged|checklist)", re.I)
URL = re.compile(r"https?://\S+|\[[^\]]*\]\([^)]*\)")
GENERIC = set("""get post put patch delete head options run call main func function app api link server
response request data value item items name type obj args kwargs self cls result results
error errors test tests event events model models field fields config settings""".split())
PY_NOISE = set("""self cls none true false print str int float dict list set tuple bool bytes object type
return import from class def async await with lambda yield raise assert pass break continue
the and for this that when then have from into error errors issue issues example examples
python version versions docs documentation should would could value values default
test tests pytest file files line lines code bug fix fixes pull request main app""".split())
IDENT = [re.compile(p) for p in (r"`([^`\n]{2,80})`", r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\b", r"\b([A-Za-z_]\w*)\s*\(",
                                  r"\b([A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*)\b", r"\b([a-z][a-z0-9]*_[a-z0-9_]+)\b",
                                  r", in ([A-Za-z_]\w*)", r"\b([A-Z][a-z]{3,}[A-Za-z0-9]*)\b")]


def seeds_for(text, ids, by_last, hubs):
    text = URL.sub(" ", "\n".join(l for l in (text or "").splitlines() if not TEMPLATE_LINE.search(l)))
    toks = set()
    for pat in IDENT:
        for m in pat.finditer(text):
            for piece in re.findall(r"[A-Za-z_][\w.]*", m.group(1)):
                piece = piece.strip(".")
                if len(piece) >= 3 and piece.lower() not in PY_NOISE and piece.split(".")[-1].lower() not in GENERIC:
                    toks.add(piece)
    seeds = set()
    for t in toks:
        if t in ids:
            hits = {t}
        elif "." in t:
            hits = {n for n in ids if n.endswith("." + t)} or set(by_last.get(t.rsplit(".", 1)[-1], ()))
        else:
            hits = set(by_last.get(t, ()))
        if 0 < len(hits) <= 10:
            seeds |= hits
    return {s for s in seeds if not (s.startswith("tests.") or ".tests." in s or s.split(".")[0].startswith("test_"))
            and s not in hubs}


# ---- AST repair edges
def repair_edges(iid):
    """-> set of (caller_key, callee_key) for self./cls./super() method calls and Class(...) -> Class.__init__."""
    files = {}
    with tarfile.open(ROOT / "src" / f"{iid}.tgz") as t:
        for m in t.getmembers():
            if m.name.endswith(".py"):
                try:
                    files[m.name] = ast.parse(t.extractfile(m).read().decode("utf-8", "replace"))
                except (SyntaxError, ValueError):
                    pass
    classes = defaultdict(list)  # class name -> [(path, qual, methods set, base names)]
    for p, tree in files.items():
        def walk(node, stack):
            for c in ast.iter_child_nodes(node):
                if isinstance(c, ast.ClassDef):
                    q = ".".join(stack + [c.name])
                    meths = {x.name for x in c.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))}
                    bases = [b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else "" for b in c.bases]
                    classes[c.name].append((p, q, meths, bases))
                    walk(c, stack + [c.name])
                elif isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    walk(c, stack + [c.name, "<locals>"])
        walk(tree, [])

    def resolve(cls_entry, name, skip_self=False, depth=0):
        p, q, meths, bases = cls_entry
        if name in meths and not skip_self:
            return f"{p}::{q}.{name}"
        if depth > 6:
            return None
        for b in bases:
            cands = classes.get(b, [])
            if len(cands) == 1:
                r = resolve(cands[0], name, False, depth + 1)
                if r:
                    return r
        return None

    edges = set()
    for p, tree in files.items():
        def walk(node, stack, cls_entry):
            for c in ast.iter_child_nodes(node):
                if isinstance(c, ast.ClassDef):
                    q = ".".join(stack + [c.name])
                    ent = next((e for e in classes[c.name] if e[0] == p and e[1] == q), None)
                    walk(c, stack + [c.name], ent)
                elif isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    caller = f"{p}::{'.'.join(stack + [c.name])}"
                    for n in ast.walk(c):
                        if not isinstance(n, ast.Call):
                            continue
                        f = n.func
                        tgt = None
                        if isinstance(f, ast.Attribute) and cls_entry:
                            v = f.value
                            if isinstance(v, ast.Name) and v.id in ("self", "cls"):
                                tgt = resolve(cls_entry, f.attr)
                            elif isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "super":
                                tgt = resolve(cls_entry, f.attr, skip_self=True)
                        elif isinstance(f, ast.Name) and len(classes.get(f.id, [])) == 1:
                            tgt = resolve(classes[f.id][0], "__init__")
                        if tgt and tgt != caller:
                            edges.add((caller, tgt))
                    walk(c, stack + [c.name, "<locals>"], cls_entry)
        walk(tree, [], None)
    return edges


def reach2(adj, seeds, gold, hubs):
    dist = {s: 0 for s in seeds}
    q = deque(dist)
    while q:
        u = q.popleft()
        if dist[u] >= 2 or (u in hubs and dist[u] > 0):
            continue
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return any(g in dist for g in gold)


def rewire(adj, n_swaps_per_edge=10):
    E = sorted({tuple(sorted((a, b))) for a, bs in adj.items() for b in bs if a != b})
    S = set(E)
    E = list(E)
    for _ in range(n_swaps_per_edge * len(E)):
        i, j = RNG.randrange(len(E)), RNG.randrange(len(E))
        (a, b), (c, d) = E[i], E[j]
        if len({a, b, c, d}) < 4:
            continue
        e1, e2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
        if e1 in S or e2 in S:
            continue
        S -= {E[i], E[j]}
        S |= {e1, e2}
        E[i], E[j] = e1, e2
    out = defaultdict(set)
    for a, b in E:
        out[a].add(b)
        out[b].add(a)
    return out


def main():
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "tasks_meta.jsonl")}
    arms = {"P": 0, "R": 0, "X": 0, "D": 0}
    n, skipped, valid_hit, valid_tot, added = 0, defaultdict(int), 0, 0, []
    for iid, m in sorted(meta.items()):
        idx = ega.provided_index(m["base_commit"])
        if not idx or not (ROOT / "src" / f"{iid}.tgz").exists():
            skipped["no_graph_or_src"] += 1
            continue
        nodes, P, _ = idx
        P = {k: set(v) for k, v in P.items()}
        ids = set(nodes) | set(P)
        by_last = defaultdict(set)
        for x in ids:
            by_last[x.rsplit(".", 1)[-1]].add(x)
        deg = sorted(((len(P.get(x, ())), x) for x in ids), reverse=True)
        hubs = {x for _, x in deg[:max(1, len(deg) // 100)]}
        gold = {ega.map_key(k.strip(), idx) for g in m["gold_function_groups"] for k in g.split("|")} - {None}
        seeds = seeds_for(m["problem_statement"], ids, by_last, hubs)
        if not gold:
            skipped["gold_unmapped"] += 1
            continue
        if not seeds:
            skipped["no_seeds"] += 1
            continue
        new = set()
        for a, b in repair_edges(iid):
            ma, mb = ega.map_key(a, idx), ega.map_key(b, idx)
            if ma and mb and ma != mb and mb not in P.get(ma, ()):
                new.add((ma, mb))
        R = {k: set(v) for k, v in P.items()}
        for a, b in new:
            R.setdefault(a, set()).add(b)
            R.setdefault(b, set()).add(a)
        X = {k: set(v) for k, v in P.items()}
        pool = sorted(ids)
        for _ in range(len(new)):
            a, b = RNG.choice(pool), RNG.choice(pool)
            if a != b:
                X.setdefault(a, set()).add(b)
                X.setdefault(b, set()).add(a)
        D = rewire(R)
        n += 1
        added.append(len(new))
        for name, adj in (("P", P), ("R", R), ("X", X), ("D", D)):
            arms[name] += reach2(adj, seeds, gold, hubs)
        # validation against executed same-class + constructor edges (v-latest traces)
        tr = ega.load_trace(iid, "empty")
        if tr:
            for t, v in tr["tests"].items():
                if t == "__collect__" or v["outcome"] != "failed":
                    continue
                for c, d in v["edges"]:
                    if ega.is_test(c) or ega.is_test(d):
                        continue
                    name = d.split("::")[-1].split(".")[-1]
                    same_cls = c.split("::")[0] == d.split("::")[0] and c.split("::")[-1].split(".")[0] == d.split("::")[-1].split(".")[0]
                    if not (same_cls or name == "__init__"):
                        continue
                    mc, md = ega.map_key(c, idx), ega.map_key(d, idx)
                    if mc and md and mc != md:
                        valid_tot += 1
                        valid_hit += md in R.get(mc, ())
    rate = {k: round(v / n, 3) if n else None for k, v in arms.items()}
    d_rp, d_rx, d_rd = (rate["R"] - rate["P"]), (rate["R"] - rate["X"]), (rate["R"] - rate["D"])
    verdict = ("PASS" if d_rp >= 0.10 and d_rx >= 0.05 and d_rd >= 0.05 else "FAIL" if d_rp < 0.05 else "AMBIGUOUS")
    out = {"n_tasks": n, "skipped": dict(skipped), "reach2": rate, "R-P": round(d_rp, 3), "R-X": round(d_rx, 3),
           "R-D": round(d_rd, 3), "edges_added_median": sorted(added)[len(added) // 2] if added else 0,
           "validation_R_covers_executed_sameclass_ctor": round(valid_hit / valid_tot, 3) if valid_tot else None,
           "validation_n": valid_tot, "verdict": verdict}
    (ROOT / "kt1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
