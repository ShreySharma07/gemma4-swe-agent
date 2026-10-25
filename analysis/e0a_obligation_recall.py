"""E0a: obligation recall of a first-edit frontier, on old-54 (pre-registered in docs/experiments.md, Oct 23).

For every task with >=2 gold function groups and every ordered pair (anchor group a, target group t):
simulate the agent's first edit = the gold version of group a only, then ask each arm for a ranked list
of other functions and check whether group t appears in the top k.

Labels:  L1 = test-necessary groups (leave-one-out runs fail without them); L2 = all gold groups.
Arms:    TYPED (tools/oblig.py), UNTYPED (same graph, every edge type, uniform weights), PROV (provided
         graph BFS), EXEC (gold-trace call graph BFS), FILE-PROX (same file by line distance),
         BM25 (source similarity to the edited function), FILE-RANDOM (same file, random order).
Ties are broken at random and hit@k is averaged over 50 tie-breaks, for every arm.

usage: python analysis/e0a_obligation_recall.py [data/runs/xg] [--json out.json]
"""
import json
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
import exec_graph_analysis as xa  # noqa: E402
import oblig  # noqa: E402
import obligation_pilot as op  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
XG = Path(ARGS[0]) if ARGS else HERE.parent / "data/runs/xg"
OUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
xa.ROOT = op.ROOT = XG
KS = (1, 2, 4, 6, 8)
SEEDS = 50


def norm(k):
    return k.replace(".<locals>", "")


def match(node, group):
    for g in group:
        if node == g or node.startswith(g + ".") or g.startswith(node + "."):
            return True
    return False


def hit_at(ranked, target, k, rng_seeds=SEEDS):
    """ranked: [(item, score)] higher=better; P(target in top k) under random tie-breaking."""
    if not ranked:
        return 0.0
    hits = 0
    for s in range(rng_seeds):
        rnd = random.Random(s)
        order = sorted(ranked, key=lambda x: (-x[1], rnd.random()))
        hits += any(match(it, target) for it, _ in order[:k])
    return hits / rng_seeds


def sh(cwd, *cmd, inp=None):
    r = subprocess.run(cmd, cwd=cwd, input=inp, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def snapshot(iid):
    d = tempfile.mkdtemp()
    with tarfile.open(XG / "src" / f"{iid}.tgz") as tf:
        tf.extractall(d, filter="data")
    return d


def gold_after(base, patch):
    d = tempfile.mkdtemp()
    shutil.copytree(base, d, dirs_exist_ok=True)
    sh(d, "git", "init", "-q")
    rc, _ = sh(d, "git", "apply", "--whitespace=nowarn", "-", inp=patch if patch.endswith("\n") else patch + "\n")
    return d if rc == 0 else None


def anchor_edit(base, after, group):
    """Before/after text of the anchor's file with ONLY the anchor group's functions replaced by gold."""
    paths = {k.split("::")[0] for k in group}
    if len(paths) != 1:
        return None
    path = paths.pop()
    try:
        before = Path(base, path).read_text()
        gold = Path(after, path).read_text()
    except OSError:
        return None
    bdefs, gdefs = oblig._defs_of(before, path), oblig._defs_of(gold, path)
    text = before
    top = [norm(k) for k in group if norm(k) in bdefs and norm(k) in gdefs]
    # replace outermost functions only (nested ones come with them)
    top = [k for k in top if not any(k != o and k.startswith(o + ".") for o in top)]
    if not top:
        return None
    for k in top:
        old, new = bdefs[k][1], gdefs[k][1]
        if text.count(old) != 1:
            return None
        text = text.replace(old, new, 1)
    if text == before:
        return None
    return path, before, text


# ------------------------------------------------------------------------------------------ arms

def arm_typed(idx, changed):
    return [(y, sc) for y, sc, *_ in oblig.frontier(idx, changed, k=40)]


def arm_untyped(idx, changed):
    saved = (dict(oblig.W), {k: set(v) for k, v in oblig.ALLOW.items()}, oblig.SAME_FILE, oblig.SAME_CLASS)
    try:
        for t in oblig.W:
            oblig.W[t] = 1.0
        every = set(oblig.W)
        for kind in oblig.ALLOW:
            oblig.ALLOW[kind] = every
        oblig.SAME_FILE = oblig.SAME_CLASS = 1.0
        return [(y, sc) for y, sc, *_ in oblig.frontier(idx, {k: ({"sig"}, []) for k in changed}, k=40)]
    finally:
        oblig.W.clear()
        oblig.W.update(saved[0])
        oblig.ALLOW.clear()
        oblig.ALLOW.update(saved[1])
        oblig.SAME_FILE, oblig.SAME_CLASS = saved[2], saved[3]


def bfs_ranked(adj, seeds, exclude, maxhop=3):
    seen, front, out = set(seeds), list(seeds), []
    for hop in range(1, maxhop + 1):
        nxt = []
        for u in front:
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
                    if v not in exclude:
                        out.append((v, -hop))
        front = nxt
    return out


def arm_file(idx, anchor_keys, path):
    lines = [idx.defs[k].line for k in anchor_keys if k in idx.defs]
    if not lines:
        return []
    return [(k, -min(abs(idx.defs[k].line - l) for l in lines)) for k in idx.by_path.get(path, ())
            if k not in anchor_keys and not any(k.startswith(a + ".") for a in anchor_keys)]


def arm_file_random(idx, anchor_keys, path):
    return [(k, 0.0) for k, _ in arm_file(idx, anchor_keys, path)]


def arm_bm25(docs, anchor_src, anchor_keys):
    sc = xa.bm25_rank(anchor_src, docs)
    return [(k, s) for k, s in sc.items() if k not in anchor_keys and s > 0]


# ------------------------------------------------------------------------------------------ main

def main():
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(XG / "tasks_meta.jsonl")}
    results = [json.loads(l) for l in open(XG / "results.jsonl")]
    agg = {lab: defaultdict(lambda: defaultdict(list)) for lab in ("L1", "L2")}
    per_task = []
    skipped = []
    for iid, m in sorted(meta.items()):
        grps = [[norm(k) for k in g] for g in op.groups(m)]
        if len(grps) < 2:
            continue
        nec = set(op.necessary(results, iid, len(grps)))
        base = snapshot(iid)
        after = gold_after(base, m["patch"])
        if after is None:
            skipped.append((iid, "gold patch does not apply"))
            continue
        idx = oblig.Index(base).build()
        ex = op.exec_adj(iid)
        pidx = xa.provided_index(m["base_commit"])
        docs = {norm(k): v for k, v in xa.functions_from_src(iid).items() if norm(k) in idx.defs}
        for a, ga in enumerate(grps):
            ed = anchor_edit(base, after, ga)
            if ed is None:
                skipped.append((iid, f"anchor {ga[0]} not a single editable existing function"))
                continue
            path, before, text = ed
            changed = {k: v for k, v in oblig.change_kinds(before, text, path).items() if k in idx.defs}
            if not changed:
                skipped.append((iid, f"anchor {ga[0]} change not detected"))
                continue
            anchor_keys = set(changed)
            anchor_src = "\n".join(oblig._defs_of(text, path)[k][1] for k in anchor_keys if k in oblig._defs_of(text, path))
            arms = {
                "TYPED": arm_typed(idx, changed),
                "UNTYPED": arm_untyped(idx, changed),
                "FILE-PROX": arm_file(idx, anchor_keys, path),
                "FILE-RANDOM": arm_file_random(idx, anchor_keys, path),
                "BM25": arm_bm25(docs, anchor_src, anchor_keys),
                "EXEC": bfs_ranked(ex, list(anchor_keys), anchor_keys) if ex else [],
            }
            prov_targets = {}
            if pidx:
                am = [x for x in (xa.map_key(k, pidx) for k in anchor_keys) if x]
                arms["PROV"] = bfs_ranked(pidx[1], am, set(am)) if am else []
            for t, gt in enumerate(grps):
                if t == a:
                    continue
                labels = ["L2"] + (["L1"] if a in nec and t in nec and len(nec) >= 2 else [])
                row = {"task": iid, "anchor": ga[0], "target": gt[0], "labels": labels,
                       "kinds": sorted(set().union(*(v[0] for v in changed.values()))), "same_file": gt[0].split("::")[0] == path}
                for name, ranked in arms.items():
                    if name == "PROV":
                        tgt = [x for x in (xa.map_key(k, pidx) for k in gt) if x] if pidx else []
                        tgt = [t_.replace(".", "\x00") for t_ in tgt]
                        rk = [(it.replace(".", "\x00"), s) for it, s in ranked]
                    else:
                        tgt, rk = gt, ranked
                    for k in KS:
                        h = hit_at(rk, tgt, k) if tgt else 0.0
                        row[f"{name}@{k}"] = h
                        for lab in labels:
                            agg[lab][name][k].append(h)
                    sizes = [len(ranked)]
                    row[f"{name}_n"] = sizes[0]
                per_task.append(row)
        shutil.rmtree(base, ignore_errors=True)
        shutil.rmtree(after, ignore_errors=True)

    order = ["TYPED", "UNTYPED", "PROV", "EXEC", "FILE-PROX", "BM25", "FILE-RANDOM"]
    for lab in ("L1", "L2"):
        n = len(agg[lab]["TYPED"][1])
        tasks = len({r["task"] for r in per_task if lab in r["labels"]})
        print(f"\n## {lab}: {n} ordered pairs from {tasks} tasks  (hit@k, random tie-breaks averaged)")
        print("  arm          " + "  ".join(f"@{k:<5}" for k in KS))
        for name in order:
            if name not in agg[lab]:
                continue
            vals = [sum(agg[lab][name][k]) / max(1, len(agg[lab][name][k])) for k in KS]
            print(f"  {name:12s} " + "  ".join(f"{v:5.2f} " for v in vals))
    print(f"\nskipped anchors: {len(skipped)}")
    for s in skipped:
        print("  ", s)
    l1 = agg["L1"]
    t6 = sum(l1["TYPED"][6]) / max(1, len(l1["TYPED"][6]))
    best_other = max((sum(l1[a][6]) / max(1, len(l1[a][6])), a) for a in order[1:] if a in l1)
    gate = t6 >= 0.80 and t6 - best_other[0] >= 0.20
    print(f"\nGATE (E0a part): TYPED hit@6 on L1 = {t6:.2f} (need >= 0.80); best other = {best_other[1]} "
          f"{best_other[0]:.2f}; margin {t6 - best_other[0]:+.2f} (need >= +0.20) -> {'PASS' if gate else 'FAIL'}")
    if OUT:
        Path(OUT).write_text(json.dumps({"rows": per_task, "skipped": skipped}, indent=1))


if __name__ == "__main__":
    main()
