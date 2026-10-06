"""
Why does search_similar_code return 0 results? Calls the OFFICIAL swegemma graph functions exactly as the tools do.
Kaggle notebook, CPU is fine. Inputs: competition data + gemma-4-developer-agent-wheelhouse.   !python probe_graph_tools.py
Prints a compact report (copy it back) and writes /kaggle/working/graph_probe.txt.
"""
import glob
import json
import pathlib
import subprocess
import sys
import traceback

out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

whl = sorted(glob.glob("/kaggle/input/**/swegemma-*.whl", recursive=True))
assert whl, "attach the wheelhouse dataset"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", whl[0]], check=True)
tasks_path = sorted(glob.glob("/kaggle/input/**/tasks.jsonl", recursive=True))[0]
DATA = pathlib.Path(tasks_path).parent
tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(tasks_path) if l.strip()}

from swegemma.graph import embedding_utils as eu, graph_utils as gu, retrieval_utils as ru  # noqa: E402

PROBES = [("fastapi_5624", ["SecurityScopes", "solve_dependencies", "security scopes propagation"]),
          ("fastapi_9555", ["solve_dependencies", "get_dependant"]),
          ("rich_3278", ["Console", "Segment"])]
for iid, queries in PROBES:
    t = tasks.get(iid)
    if not t:
        continue
    repo, commit = t["repo"], t["base_commit"]
    log(f"\n=== {iid} repo={repo} commit={commit[:10]}")
    try:
        g = gu.get_graph(repo, graph_dir=str(DATA / "graphs"), embeddings_dir=str(DATA / "embeddings"), base_commit=commit)
        nodes = list(g.nodes(data=True))
        with_emb = sum(1 for _, a in nodes if a.get("embedding") is not None)
        log(f"graph nodes={len(nodes)} edges={g.number_of_edges()} nodes_with_embedding={with_emb}")
        log("sample node:", nodes[0][0], {k: (str(v)[:60] if k != "embedding" else type(v).__name__) for k, v in nodes[0][1].items()})
        names, texts, mat = ru._extract_nodes_and_embeddings(g, repo, embeddings_dir=str(DATA / "embeddings"), base_commit=commit)
        log(f"_extract_nodes_and_embeddings: names={len(names)} matrix={getattr(mat, 'shape', None)}")
    except Exception as e:  # noqa: BLE001
        log("get_graph FAILED:", type(e).__name__, e); log(traceback.format_exc()[-800:]); continue
    for q in queries:
        try:
            res_node = gu.resolve_node_name(q, g)
            emb = eu.embed(q, repo, embeddings_dir=str(DATA / "embeddings"), base_commit=commit)
            res = ru.get_similar_nodes(node=q, repo_name=repo, k=5, graph_dir=str(DATA / "graphs"),
                                       embeddings_dir=str(DATA / "embeddings"), base_commit=commit)
            log(f"  q={q!r}: resolve->{res_node} | embed->{None if emb is None else emb.shape} | results={len(res)} "
                f"{[r['node_name'] for r in res[:3]]}")
            nb = list(g.successors(res_node))[:5] + list(g.predecessors(res_node))[:5] if res_node else []
            log(f"     neighbors sample: {nb}")
        except Exception as e:  # noqa: BLE001
            log(f"  q={q!r}: EXC {type(e).__name__}: {e}")
pathlib.Path("/kaggle/working/graph_probe.txt").write_text("\n".join(out))
print("\nwrote /kaggle/working/graph_probe.txt")
