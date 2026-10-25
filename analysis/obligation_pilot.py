"""Obligation-closure pilot on the execution-graph instrument (data/runs/xg).

Obligation set of a task = gold function groups whose leave-one-out run FAILS the fail-to-pass
tests (test-necessary). For tasks with >=2 necessary groups we ask, for every ordered pair
(anchor a, obligation o): is o within k hops of a in
  EXEC  = call edges executed by the gold-patched tests (undirected),
  PROV  = the competition's provided graph (undirected, mapped by name),
  FILE  = same file (lexical locality baseline)?
Then the same question for real agent runs: an agent that edited >=1 necessary group but missed
another one; would a k-hop frontier around what it edited have contained what it missed, and how
big is that frontier (the token cost of injecting it)?

usage: python analysis/obligation_pilot.py data/runs/xg RUN_DIR [RUN_DIR ...]
"""
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exec_graph_analysis as xa  # noqa: E402  (reuses provided_index / map_key / is_test)

ROOT = Path(sys.argv[1]) if __name__ == "__main__" else Path(__file__).resolve().parent.parent / "data/runs/xg"
xa.ROOT = ROOT
RUNS = sys.argv[2:] if __name__ == "__main__" else []


def groups(meta):
    return [[k.strip() for k in g.split("|")] for g in meta["gold_function_groups"]]


def necessary(results, iid, n):
    loo = {r["mode"]: r["resolved"] for r in results if r["id"] == iid and r["mode"].startswith("loo")}
    return [i for i in range(n) if loo.get(f"loo{i}") is False]


def exec_adj(iid):
    t = xa.load_trace(iid, "gold")
    adj = defaultdict(set)
    if not t:
        return None
    for v in t["tests"].values():
        for a, b in v.get("edges") or []:
            if xa.is_test(a) or xa.is_test(b):
                continue
            a, b = a.replace(".<locals>", ""), b.replace(".<locals>", "")
            adj[a].add(b)
            adj[b].add(a)
    return adj


def ball(adj, seeds, k):
    seen, q = set(seeds), deque((s, 0) for s in seeds)
    while q:
        u, d = q.popleft()
        if d == k:
            continue
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                q.append((v, d + 1))
    return seen


def norm(key):
    return key.replace(".<locals>", "")


HUNK = re.compile(r"^@@ [^@]* @@ ?(.*)$")
NAME = re.compile(r"(?:def|class)\s+([A-Za-z_]\w*)")


def agent_groups(patch_text, grps):
    """indices of gold groups the agent's patch touched (file + function-name match)."""
    files, names, cur = set(), defaultdict(set), None
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            cur = line[6:].strip()
            files.add(cur)
        m = HUNK.match(line)
        if m and cur:
            names[cur].update(NAME.findall(m.group(1)))
        elif cur and re.match(r"^[+-]\s*(?:async\s+)?def\s+", line):
            names[cur].update(NAME.findall(line))
    hit = set()
    for i, g in enumerate(grps):
        for key in g:
            path, q = key.split("::", 1)
            leaf = q.replace(".<locals>", "").split(".")[-1]
            if path in files and (leaf in names[path] or q == "<module>"):
                hit.add(i)
    return hit


def main():
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "tasks_meta.jsonl")}
    results = [json.loads(l) for l in open(ROOT / "results.jsonl")]
    pair = defaultdict(lambda: [0, 0])  # (graph,k) -> [hits, pairs]
    tasks = {}
    for iid, m in meta.items():
        grps = groups(m)
        if len(grps) < 2:
            continue
        nec = necessary(results, iid, len(grps))
        if len(nec) < 2:
            continue
        ex = exec_adj(iid)
        pidx = xa.provided_index(m["base_commit"])
        prov_adj = pidx[1] if pidx else {}
        tasks[iid] = (grps, nec, ex, pidx)
        for a in nec:
            for o in nec:
                if a == o:
                    continue
                A, O = [norm(k) for k in grps[a]], [norm(k) for k in grps[o]]
                same_file = any(x.split("::")[0] == y.split("::")[0] for x in A for y in O)
                pair[("FILE", 0)][0] += same_file
                pair[("FILE", 0)][1] += 1
                for k in (1, 2):
                    if ex is not None:
                        pair[("EXEC", k)][0] += bool(ball(ex, A, k) & set(O))
                        pair[("EXEC", k)][1] += 1
                    if pidx:
                        Am = [xa.map_key(x, pidx) for x in A]
                        Om = {xa.map_key(x, pidx) for x in O} - {None}
                        Am = [x for x in Am if x]
                        pair[("PROV", k)][0] += bool(Am and Om and (ball(prov_adj, Am, k) & Om))
                        pair[("PROV", k)][1] += 1
    print(f"tasks with >=2 test-necessary gold groups: {len(tasks)} -> {sorted(tasks)}")
    print("anchor->obligation pairs covered by a k-hop frontier:")
    for (g, k), (h, n) in sorted(pair.items()):
        print(f"  {g:4s} k={k}: {h}/{n}")

    # real agent runs: partial coverage of the obligation set
    agg = defaultdict(lambda: [0, 0])
    sizes = defaultdict(list)
    n_runs = n_partial = 0
    for run in RUNS:
        pt = {json.loads(l)["instance_id"]: json.loads(l) for l in open(Path(run) / "analysis" / "per_task.jsonl")}
        for iid, (grps, nec, ex, pidx) in tasks.items():
            if iid not in pt:
                continue
            pp = Path(run) / "merged" / "patches" / f"{iid}.patch"
            if not pp.exists():
                continue
            n_runs += 1
            hit = agent_groups(pp.read_text(errors="replace"), grps) & set(nec)
            miss = set(nec) - hit
            if not hit or not miss:
                continue
            n_partial += 1
            A = [norm(k) for i in hit for k in grps[i]]
            O = {norm(k) for i in miss for k in grps[i]}
            if ex is not None:
                for k in (1, 2):
                    b = ball(ex, A, k)
                    agg[("EXEC", k)][0] += bool(b & O)
                    agg[("EXEC", k)][1] += 1
                    sizes[("EXEC", k)].append(len(b) - len(A))
            if pidx:
                Am = [x for x in (xa.map_key(a, pidx) for a in A) if x]
                Om = {xa.map_key(o, pidx) for o in O} - {None}
                for k in (1, 2):
                    b = xa.ball(pidx[1], Am, k) if hasattr(xa, "ball") else ball(pidx[1], Am, k)
                    agg[("PROV", k)][0] += bool(b & Om)
                    agg[("PROV", k)][1] += 1
                    sizes[("PROV", k)].append(len(b) - len(Am))
    print(f"\nagent runs on these tasks: {n_runs}; partial (edited >=1 necessary group, missed >=1): {n_partial}")
    for key, (h, n) in sorted(agg.items()):
        s = sorted(sizes[key])
        med = s[len(s) // 2] if s else 0
        print(f"  {key[0]:4s} k={key[1]}: frontier contains a missed obligation {h}/{n}; median frontier size {med} nodes")


if __name__ == "__main__":
    main()
