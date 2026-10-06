"""
Gemma 4 Developer Agent — dataset inspection script.

Run in a Kaggle notebook with the competition dataset attached:
    !python inspect_dataset.py
or paste into a cell. Override the data location with DATA_ROOT=/path if needed.

Prints a compact report and writes /kaggle/working/inspection_summary.json
(upload that JSON back to Claude).
"""

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("DATA_ROOT", "/kaggle/input"))
OUT = Path(os.environ.get("OUT_PATH", "/kaggle/working/inspection_summary.json"))
HASH_RE = re.compile(r"[0-9a-f]{40}")


def human(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def repo_short(repo):
    """'fastapi/fastapi' -> 'fastapi'"""
    return repo.split("/")[-1].lower()


# ---------------------------------------------------------------- 0. file tree
all_files = [p for p in ROOT.rglob("*") if p.is_file()]
dir_stats = defaultdict(lambda: [0, 0])  # rel dir (depth<=3) -> [count, bytes]
for p in all_files:
    rel = p.relative_to(ROOT)
    key = "/".join(rel.parts[:-1][:3]) or "."
    dir_stats[key][0] += 1
    dir_stats[key][1] += p.stat().st_size

print("=" * 70)
print(f"0. FILE TREE under {ROOT}  ({len(all_files)} files)")
print("=" * 70)
for d, (cnt, size) in sorted(dir_stats.items()):
    print(f"  {d:<55} {cnt:>6} files  {human(size):>9}")

# ---------------------------------------------------------------- 1. tasks
task_files = [p for p in all_files if p.name.endswith(".jsonl") and "task" in p.name.lower()]
tasks = []
for tf in task_files:
    with open(tf) as f:
        tasks += [json.loads(line) for line in f if line.strip()]

repo_counts = Counter(t["repo"] for t in tasks)
dates = sorted(t.get("created_at", "") for t in tasks if t.get("created_at"))

print("\n" + "=" * 70)
print(f"1. TASKS  ({len(tasks)} tasks from {[str(p.relative_to(ROOT)) for p in task_files]})")
print("=" * 70)
for repo, c in repo_counts.most_common():
    print(f"  {repo:<40} {c}")
if dates:
    print(f"  created_at range: {dates[0]}  ->  {dates[-1]}")
print(f"  tasks with hints: {sum(1 for t in tasks if t.get('hints_text', '').strip())}")


def patched_files(patch):
    return re.findall(r"^\+\+\+ b/(\S+)", patch or "", flags=re.M)


# ---------------------------------------------------------------- 2. graphs + embeddings
graph_files = [
    p for p in all_files
    if p.suffix == ".json" and "graph" in str(p.relative_to(ROOT)).lower()
]
emb_files = [p for p in all_files if p.suffix == ".npz"]

print("\n" + "=" * 70)
print(f"2. GRAPHS ({len(graph_files)})  /  EMBEDDINGS ({len(emb_files)})")
print("=" * 70)

graph_summary = {}
graph_nodes_by_repo = {}  # repo_short -> set(node ids)
for gp in sorted(graph_files):
    try:
        with open(gp) as f:
            g = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"  !! could not load {gp.name}: {e}")
        continue
    nodes = g.get("nodes", [])
    edges = g.get("edges") or g.get("links") or []
    meta = g.get("graph", {}) if isinstance(g.get("graph"), dict) else {}
    repo_name = str(meta.get("repo_name", ""))
    m = HASH_RE.search(repo_name) or HASH_RE.search(gp.stem)
    edge_types = Counter(
        str(e.get("type") or e.get("edge_type") or e.get("label") or "?") for e in edges
    )
    node_keys = Counter(k for n in nodes[:2000] for k in n.keys())
    ids = {str(n.get("id")) for n in nodes}
    graph_summary[gp.name] = {
        "path": str(gp.relative_to(ROOT)),
        "size": gp.stat().st_size,
        "repo_name": repo_name,
        "commit": m.group(0) if m else None,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "edge_types": dict(edge_types.most_common()),
        "node_fields": sorted(node_keys),
        "graph_attrs": sorted(meta.keys()),
    }
    # match graph to task repos by short name
    label = (repo_name or gp.stem).lower()
    for repo in repo_counts:
        if repo_short(repo) in label:
            graph_nodes_by_repo.setdefault(repo, set()).update(ids)
    print(f"  {gp.name:<50} nodes={len(nodes):>7} edges={len(edges):>8} "
          f"{human(gp.stat().st_size):>9}")
    print(f"      repo_name={repo_name}")
    print(f"      edge_types={dict(edge_types.most_common())}")
    print(f"      node_fields={sorted(node_keys)}  graph_attrs={sorted(meta.keys())}")

emb_summary = {}
for i, ep in enumerate(sorted(emb_files)):
    emb_summary[ep.name] = {"path": str(ep.relative_to(ROOT)), "size": ep.stat().st_size}
    if i == 0:  # peek inside the first one only
        try:
            import numpy as np
            try:
                z = np.load(ep, allow_pickle=False)
            except ValueError:
                z = np.load(ep, allow_pickle=True)
            emb_summary[ep.name]["arrays"] = {
                k: [list(z[k].shape), str(z[k].dtype)] for k in list(z.files)[:10]
            }
        except Exception as e:  # noqa: BLE001
            emb_summary[ep.name]["error"] = str(e)
for name, info in emb_summary.items():
    extra = f"  arrays={info['arrays']}" if "arrays" in info else ""
    print(f"  [emb] {name:<45} {human(info['size']):>9}{extra}")

covered = [r for r in repo_counts if r in graph_nodes_by_repo]
missing = [r for r in repo_counts if r not in graph_nodes_by_repo]
print(f"\n  task repos WITH a graph:    {covered}")
print(f"  task repos WITHOUT a graph: {missing}")

# ---------------------------------------------------------------- 3. commit check + patch coverage
print("\n" + "=" * 70)
print("3. GRAPH-vs-TASK COMMIT CHECK  +  GOLD-PATCH COVERAGE BY GRAPH")
print("=" * 70)

graph_commits = defaultdict(set)
for info in graph_summary.values():
    label = (info["repo_name"] or info["path"]).lower()
    for repo in repo_counts:
        if repo_short(repo) in label and info["commit"]:
            graph_commits[repo].add(info["commit"])

per_task = []
for t in tasks:
    repo = t["repo"]
    files = patched_files(t.get("patch"))
    py_files = [f for f in files if f.endswith(".py")]
    ids = graph_nodes_by_repo.get(repo, set())
    found = []
    for f in py_files:
        mod = f[:-3].replace("/", ".")
        cands = {mod, mod.removeprefix("src."), mod.removesuffix(".__init__")}
        if any(any(i == c or i.startswith(c + ".") for c in cands) for i in ids):
            found.append(f)
    row = {
        "id": t["instance_id"],
        "repo": repo,
        "base_commit_matches_graph": t["base_commit"] in graph_commits.get(repo, set()),
        "files": len(files),
        "py_files": len(py_files),
        "py_files_in_graph": len(found),
        "non_py_files": [f for f in files if not f.endswith(".py")],
    }
    per_task.append(row)

for r in per_task:
    print(f"  {r['id']:<30} commit_match={str(r['base_commit_matches_graph']):<5} "
          f"files={r['files']} py={r['py_files']} py_in_graph={r['py_files_in_graph']}"
          + (f" non_py={r['non_py_files']}" if r["non_py_files"] else ""))

n = len(per_task) or 1
print(f"\n  base_commit == graph commit: {sum(r['base_commit_matches_graph'] for r in per_task)}/{len(per_task)}")
print(f"  tasks whose every gold .py file is in the graph: "
      f"{sum(1 for r in per_task if r['py_files'] and r['py_files'] == r['py_files_in_graph'])}/{len(per_task)}")
print(f"  tasks touching NO .py file: {sum(1 for r in per_task if r['py_files'] == 0)}/{len(per_task)}")

# ---------------------------------------------------------------- 4. snapshots
snaps = [p for p in all_files if "snapshot" in str(p.relative_to(ROOT)).lower()]
print("\n" + "=" * 70)
print(f"4. SNAPSHOTS: {len(snaps)} files, {human(sum(p.stat().st_size for p in snaps))} total")
print("=" * 70)

# ---------------------------------------------------------------- save
summary = {
    "root": str(ROOT),
    "n_files": len(all_files),
    "tree": {d: {"files": c, "bytes": s} for d, (c, s) in dir_stats.items()},
    "n_tasks": len(tasks),
    "repo_counts": dict(repo_counts),
    "created_at_range": [dates[0], dates[-1]] if dates else None,
    "graphs": graph_summary,
    "embeddings": emb_summary,
    "task_repos_with_graph": covered,
    "task_repos_without_graph": missing,
    "per_task": per_task,
    "snapshots": {"n_files": len(snaps), "bytes": sum(p.stat().st_size for p in snaps)},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(summary, indent=2))
print(f"\nSaved -> {OUT}  (upload this file back to Claude)")
