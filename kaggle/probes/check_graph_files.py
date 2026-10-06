"""
Why did every search_similar_code call return 0 results? Checks the graph/embedding files exactly the way the
official harness resolves them (swegemma/graph/graph_utils.get_graph and embedding_utils.embed).

Run on Kaggle (CPU is enough) with the competition data attached:   !python check_graph_files.py
Prints a short report and writes /kaggle/working/graph_check.json. Upload the json (or paste the printed summary).
"""

import glob
import json
import os
import pathlib
import zipfile

tasks_path = sorted(glob.glob("/kaggle/input/**/tasks.jsonl", recursive=True))[0]
DATA = pathlib.Path(tasks_path).parent
G, E = DATA / "graphs", DATA / "embeddings"
tasks = [json.loads(l) for l in open(tasks_path) if l.strip()]


def harness_candidates(base, repo, commit, ext):
    short, slug = repo.rsplit("/", 1)[-1], repo.replace("/", "_")
    return [base / f"{short}_{commit}{ext}", base / f"{slug}_{commit}{ext}", base / f"{slug}{ext}",
            base / f"{short}{ext}", base / f"{repo}{ext}"]


def size(p):
    return p.stat().st_size if p.exists() else None


def npz_keys(p):
    try:
        with zipfile.ZipFile(p) as z:
            return [n[:-4] for n in z.namelist()]
    except Exception as e:  # noqa: BLE001
        return f"ERR {type(e).__name__}"


all_g = sorted(G.glob("*")), sorted(E.glob("*"))
print(f"graphs: {len(all_g[0])} files, {sum(1 for p in all_g[0] if p.stat().st_size == 0)} are 0 bytes")
print(f"embeddings: {len(all_g[1])} files, {sum(1 for p in all_g[1] if p.stat().st_size == 0)} are 0 bytes")
print("example names:", [p.name for p in all_g[1][:4]])

rows = []
for t in tasks:
    repo, commit, iid = t["repo"], t["base_commit"], t["instance_id"]
    r = {"id": iid}
    for kind, base, ext in (("graph", G, ".json"), ("emb", E, ".npz")):
        cands = harness_candidates(base, repo, commit, ext)
        chosen = next((p for p in cands if p.exists()), cands[0])
        task_named = base / f"{iid}{ext}"
        r[f"{kind}_chosen"] = chosen.name
        r[f"{kind}_chosen_bytes"] = size(chosen)
        r[f"{kind}_taskid_bytes"] = size(task_named)
    keys = npz_keys(E / r["emb_chosen"]) if r["emb_chosen_bytes"] else []
    r["emb_chosen_nkeys"] = len(keys) if isinstance(keys, list) else keys
    alt = E / f"{iid}.npz"
    akeys = npz_keys(alt) if size(alt) else []
    r["emb_taskid_nkeys"] = len(akeys) if isinstance(akeys, list) else akeys
    r["emb_key_examples"] = (akeys or keys)[:3] if isinstance(akeys or keys, list) else []
    rows.append(r)

n = len(rows)
dead_g = sum(1 for r in rows if not r["graph_chosen_bytes"])
dead_e = sum(1 for r in rows if not r["emb_chosen_bytes"])
fixable_g = sum(1 for r in rows if not r["graph_chosen_bytes"] and r["graph_taskid_bytes"])
fixable_e = sum(1 for r in rows if not r["emb_chosen_bytes"] and r["emb_taskid_bytes"])
print(f"\nTasks: {n}")
print(f"graph file the harness picks is missing/0 bytes: {dead_g}/{n}  (non-empty task-id copy exists: {fixable_g})")
print(f"embedding file the harness picks is missing/0 bytes: {dead_e}/{n}  (non-empty task-id copy exists: {fixable_e})")
print("sample rows:")
for r in rows[:5]:
    print(" ", r)
out = pathlib.Path("/kaggle/working/graph_check.json")
out.write_text(json.dumps({"n": n, "dead_graph": dead_g, "dead_emb": dead_e, "fixable_graph": fixable_g,
                           "fixable_emb": fixable_e, "rows": rows}, indent=1))
print(f"\nwrote {out}")
