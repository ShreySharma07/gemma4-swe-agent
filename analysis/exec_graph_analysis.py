"""
M1-M4 on the execution-graph instrument output (kaggle/exec_graph), against the predictions in docs/plan_autonomy.md.
Light, pure-Python. Usage:  python3 analysis/exec_graph_analysis.py data/runs/xg
"""
import ast
import io
import json
import math
import re
import statistics as st
import sys
import tarfile
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/runs/xg")
V5 = Path("results/runs/r3_v5_graph/behaviour_per_task.json")


def is_test(key):
    p = key.split("::")[0]
    if p.startswith("<") or "site-packages" in p:  # frozen stdlib / installed packages are not repo code
        return True
    parts = p.split("/")
    return any(x in ("tests", "test", "testing") for x in parts[:-1]) or parts[-1].startswith("test_") \
        or parts[-1] == "conftest.py" or "docs_src" in parts


def load_trace(iid, mode):
    f = ROOT / "traces" / f"{iid}__{mode}.json"
    return json.loads(f.read_text()) if f.exists() else None


def f2p_match(nodeid, f2p):
    return any(nodeid.endswith(x) or x.endswith(nodeid) or nodeid.split("::")[-1].split("[")[0] == x.split("::")[-1].split("[")[0]
               for x in f2p)


def functions_from_src(iid):
    """{key: (body tokens)} for every function in the base snapshot."""
    out = {}
    tf = ROOT / "src" / f"{iid}.tgz"
    if not tf.exists():
        return out
    with tarfile.open(tf) as t:
        for m in t.getmembers():
            if not m.name.endswith(".py"):
                continue
            src = t.extractfile(m).read().decode("utf-8", "replace")
            try:
                tree = ast.parse(src)
            except (SyntaxError, ValueError):
                continue
            lines = src.splitlines()

            def walk(node, stack):
                for c in ast.iter_child_nodes(node):
                    if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        q = ".".join(stack + [c.name])
                        body = "\n".join(lines[c.lineno - 1:(c.end_lineno or c.lineno)])
                        out[f"{m.name}::{q}"] = tokens(body + " " + q)
                        walk(c, stack + [c.name, "<locals>"])
                    elif isinstance(c, ast.ClassDef):
                        walk(c, stack + [c.name])
                    else:
                        walk(c, stack)
            walk(tree, [])
    return out


def tokens(s):
    toks = []
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s):
        toks.append(w.lower())
        toks += [p.lower() for p in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", w) if len(p) > 1]
    return toks


def bm25_rank(query, docs, k1=1.2, b=0.75):
    N = len(docs)
    avg = sum(len(d) for d in docs.values()) / max(1, N)
    df = Counter(t for d in docs.values() for t in set(d))
    q = Counter(tokens(query))
    scores = {}
    for k, d in docs.items():
        tf = Counter(d)
        s = 0.0
        for t in q:
            if tf[t]:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        scores[k] = s
    return scores


def rank_of(scores, gold):
    """pessimistic rank of the best gold key (ties counted against us)."""
    best = None
    for g in gold:
        if g in scores:
            s = scores[g]
            r = 1 + sum(v > s for v in scores.values()) + sum(v == s for k, v in scores.items() if k != g)
            best = r if best is None else min(best, r)
    return best


# ------------------------------------------------------------------ provided-graph id mapping
def module_of(path):
    m = path[:-3].replace("/", ".")
    for pre in ("src.",):
        if m.startswith(pre):
            m = m[len(pre):]
    return m[:-9] if m.endswith(".__init__") else m


def provided_index(commit):
    gp = next((p for p in (ROOT / "graphs").glob(f"*{commit}*")), None) if commit else None
    if not gp:
        return None
    g = json.loads(gp.read_text())
    nodes = {str(n.get("id") or n.get("name")) for n in g.get("nodes", [])}
    adj = defaultdict(set)
    for e in g.get("edges") or g.get("links") or []:
        s, t = str(e.get("source")), str(e.get("target"))
        adj[s].add(t)
        adj[t].add(s)
    by_tail = defaultdict(set)
    for n in nodes:
        parts = n.split(".")
        for k in range(1, min(4, len(parts)) + 1):
            by_tail[".".join(parts[-k:])].add(n)
    return nodes, adj, by_tail


def map_key(key, idx):
    nodes, adj, by_tail = idx
    path, q = key.split("::", 1)
    q = q.replace(".<locals>", "")
    full = f"{module_of(path)}.{q}"
    if full in nodes:
        return full
    parts = full.split(".")
    for k in range(min(4, len(parts)), 0, -1):
        cands = by_tail.get(".".join(parts[-k:]), set())
        if len(cands) == 1:
            return next(iter(cands))
        if cands and k >= 2:
            same_mod = [c for c in cands if parts[-k - 1:-k] and parts[-k - 1] in c]
            if len(same_mod) == 1:
                return same_mod[0]
    return None


def main():
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "tasks_meta.jsonl")}
    results = defaultdict(dict)
    for l in open(ROOT / "results.jsonl"):
        r = json.loads(l)
        results[r["id"]][r["mode"]] = r["resolved"]
    v5 = {x["id"]: x for x in json.loads(V5.read_text())} if V5.exists() else {}
    rows = []
    for iid, m in sorted(meta.items()):
        tr = load_trace(iid, "empty")
        row = {"id": iid, "repo": m["repo"], "gold_ok": results[iid].get("gold"), "empty_ok": results[iid].get("empty"),
               "traced": tr is not None and tr.get("monitoring")}
        gold = {k.strip() for grp in m["gold_function_groups"] for k in grp.split("|")}
        row["n_gold_funcs"] = len(gold)
        if tr:
            f2p = m.get("FAIL_TO_PASS") or []
            if isinstance(f2p, str):
                try:
                    f2p = json.loads(f2p)
                except json.JSONDecodeError:
                    f2p = [f2p]
            tests = {t: v for t, v in tr["tests"].items() if t != "__collect__"}
            failing = {t: v for t, v in tests.items() if v["outcome"] == "failed" and (not f2p or f2p_match(t, f2p))}
            passing = {t: v for t, v in tests.items() if v["outcome"] == "passed"}
            row["n_fail_tests"], row["n_pass_tests"] = len(failing), len(passing)
            exec_fail = set().union(*[set(v["nodes"]) for v in failing.values()]) if failing else set()
            row["M2_gold_executed"] = bool(gold & exec_fail)
            # M2b depth: BFS from each failing test's own function over its caller->callee edges
            depths = []
            for t, v in failing.items():
                g = defaultdict(set)
                for a, b in v["edges"]:
                    g[a].add(b)
                starts = [n for n in v["nodes"] if is_test(n) and n.split("::")[-1].split(".")[-1].startswith("test")]
                dist = {s: 0 for s in starts}
                dq = deque(starts)
                while dq:
                    u = dq.popleft()
                    for w in g[u]:
                        if w not in dist:
                            dist[w] = dist[u] + 1
                            dq.append(w)
                ds = [dist[k] for k in gold if k in dist]
                if ds:
                    depths.append(min(ds))
            row["M2b_depth"] = min(depths) if depths else None
            # M1 provided-graph coverage of executed non-test edges
            idx = provided_index(m.get("base_commit"))
            if idx:
                e_all = {tuple(e) for v in failing.values() for e in v["edges"] if not is_test(e[0]) and not is_test(e[1])}
                mapped = [(map_key(a, idx), map_key(b, idx)) for a, b in e_all]
                both = [(a, b) for a, b in mapped if a and b and a != b]
                row["M1_exec_edges"] = len(e_all)
                row["M1b_both_endpoints_in_graph"] = len(both) / len(e_all) if e_all else None
                row["M1_edge_in_graph"] = sum(b in idx[1][a] for a, b in both) / len(e_all) if e_all else None
                row["M1_edge_in_graph_given_mapped"] = sum(b in idx[1][a] for a, b in both) / len(both) if both else None
            # M4 SBFL (Ochiai) over executed non-test functions
            nf = len(failing)
            if nf:
                cover = Counter()
                for v in failing.values():
                    cover.update({n: 1 for n in v["nodes"]})
                pcover = Counter()
                for v in passing.values():
                    pcover.update({n: 1 for n in v["nodes"]})
                cand = {n for v in list(failing.values()) + list(passing.values()) for n in v["nodes"] if not is_test(n)}
                och = {n: cover[n] / math.sqrt(nf * (cover[n] + pcover[n])) if cover[n] else 0.0 for n in cand}
                row["M4_sbfl_rank"] = rank_of(och, gold)
        # M4b BM25 over all non-test functions in the snapshot
        funcs = {k: v for k, v in functions_from_src(iid).items() if not is_test(k)}
        if funcs:
            row["n_funcs"] = len(funcs)
            row["M4b_bm25_rank"] = rank_of(bm25_rank(m["problem_statement"], funcs), gold)
        # M3 necessity
        loos = {k: v for k, v in results[iid].items() if k.startswith("loo")}
        if loos:
            row["M3_loo_n"] = len(loos)
            row["M3_unnecessary"] = sum(loos.values())
        # M4c v5 localiser
        x = v5.get(iid)
        if x and x.get("diag"):
            fm = re.search(r"FUNCTION:\s*([\w\.]+)", x["diag"])
            fn = fm.group(1).split(".")[-1] if fm else None
            row["M4c_v5_named_gold"] = bool(fn and any(k.split("::")[-1].split(".")[-1] == fn for k in gold))
        rows.append(row)

    def share(key, pred=lambda v: bool(v)):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return (round(sum(pred(v) for v in vals) / len(vals), 3), len(vals)) if vals else (None, 0)

    def med(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return (round(st.median(vals), 3), len(vals)) if vals else (None, 0)

    loo_n = sum(r.get("M3_loo_n", 0) for r in rows)
    loo_u = sum(r.get("M3_unnecessary", 0) for r in rows)
    out = {
        "tasks": len(rows), "traced": sum(bool(r["traced"]) for r in rows),
        "gold_resolved": sum(bool(r["gold_ok"]) for r in rows), "empty_resolved": sum(bool(r["empty_ok"]) for r in rows),
        "M1 edge in provided graph (median over tasks) [pred 0.55-0.85]": med("M1_edge_in_graph"),
        "M1 given both endpoints mapped (median)": med("M1_edge_in_graph_given_mapped"),
        "M1b both endpoints in graph (median) [pred >=0.90]": med("M1b_both_endpoints_in_graph"),
        "M2 gold executed by a failing test [pred 0.70-0.90]": share("M2_gold_executed"),
        "M2b depth test->gold (median) [pred 2-4]": med("M2b_depth"),
        "M3 LOO unnecessary share [pred 0.10-0.35]": (round(loo_u / loo_n, 3) if loo_n else None, loo_n),
        "M4 SBFL top-5 [pred 0.40-0.70]": share("M4_sbfl_rank", lambda v: v <= 5),
        "M4 SBFL top-1": share("M4_sbfl_rank", lambda v: v <= 1),
        "M4b BM25 top-5 [pred 0.20-0.40]": share("M4b_bm25_rank", lambda v: v <= 5),
        "M4c v5 named gold [pred ~0.48]": share("M4c_v5_named_gold"),
    }
    (ROOT / "analysis.json").write_text(json.dumps({"summary": out, "rows": rows}, indent=1))
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
