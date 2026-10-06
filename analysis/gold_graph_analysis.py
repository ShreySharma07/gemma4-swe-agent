"""
Gemma 4 Developer Agent — gold-location graph analysis (no agent needed).

For every training task it finds the functions/classes the gold patch edits
("gold nodes") and the symbols mentioned in the issue text ("seed nodes"), then
answers four questions using only the provided call graphs + embeddings:

  Q1  Coverage     Are gold nodes in the graph? How connected are they?
  Q2  Reachability Can you walk from issue-mentioned symbols to the gold nodes
                   along call edges, and in how many hops?
  Q3  Embedding vs graph
                   Does search_similar_code(seed) surface gold nodes (top-10)?
                   Does get_code_neighbors(seed) (1 hop) surface them?
                   Are the two complementary? How does cosine similarity
                   vary with graph distance?
  Q4  Centrality   Are gold nodes more central (degree / PageRank / betweenness)
                   than a random node? (AUC; 0.5 = no signal)

Run in a Kaggle notebook with the competition dataset attached (CPU is fine):
    !python gold_graph_analysis.py
Optional env vars:
    DATA_ROOT=/kaggle/input/competitions/gemma-4-developer-agent
    OUT_DIR=/kaggle/working
    BETWEENNESS=0          # skip approximate betweenness (saves a few minutes)

Outputs (upload both back):
    OUT_DIR/gold_graph_summary.json   aggregate numbers
    OUT_DIR/gold_graph_per_task.json  per-task details for spot-checking
"""

import json
import os
import random
import re
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

random.seed(0)
np.random.seed(0)

ROOT = Path(os.environ.get("DATA_ROOT", "/kaggle/input"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
DO_BETWEENNESS = os.environ.get("BETWEENNESS", "1") != "0"
HASH_RE = re.compile(r"[0-9a-f]{40}")
EMB_TOPK = 10          # search_similar_code default k
NEIGHBOR_CAP = 50      # get_code_neighbors default max_neighbors
MAX_SEED_MATCHES = 10  # a mention matching more nodes than this is too ambiguous

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- files
def find_files():
    tasks = sorted(p for p in ROOT.rglob("*.jsonl") if "task" in p.name.lower())
    graphs = {}
    embs = {}
    for p in ROOT.rglob("*.json"):
        if "graph" in str(p).lower():
            m = HASH_RE.search(p.stem)
            if m:
                graphs[m.group(0)] = p
    for p in ROOT.rglob("*.npz"):
        m = HASH_RE.search(p.stem)
        if m:
            embs[m.group(0)] = p
    return tasks, graphs, embs


# --------------------------------------------------------------------------- patch parsing
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")


def parse_patch(patch):
    """Return list of hunks: {file, new_file, header_ctx, removed, added, context}."""
    hunks, cur_file, new_file, cur = [], None, False, None
    for line in (patch or "").splitlines():
        if line.startswith("--- "):
            new_file = line.strip() == "--- /dev/null"
            continue
        if line.startswith("+++ "):
            path = line[4:].strip()
            cur_file = None if path == "/dev/null" else re.sub(r"^b/", "", path)
            continue
        m = HUNK_RE.match(line)
        if m:
            cur = {"file": cur_file, "new_file": new_file, "header_ctx": m.group(5).strip(),
                   "removed": [], "added": [], "context": []}
            hunks.append(cur)
            continue
        if cur is None or not line:
            continue
        tag, body = line[0], line[1:]
        if tag == "-":
            cur["removed"].append(body)
        elif tag == "+":
            cur["added"].append(body)
        elif tag == " ":
            cur["context"].append(body)
    return [h for h in hunks if h["file"] and h["file"].endswith(".py")]


def meaningful(lines):
    out = []
    for ln in lines:
        s = ln.strip()
        if len(s) >= 4 and s not in {"pass", "else:", "try:", "return", "\"\"\"", "'''"} \
                and not s.startswith("#"):
            out.append(s)
    return out


# --------------------------------------------------------------------------- graph container
class RepoGraph:
    def __init__(self, gpath, epath):
        g = json.loads(Path(gpath).read_text())
        self.nodes = {}
        for n in g.get("nodes", []):
            nid = str(n.get("id"))
            self.nodes[nid] = n.get("text") or ""
        edges = g.get("edges") or g.get("links") or []
        self.out = defaultdict(set)
        self.inn = defaultdict(set)
        for e in edges:
            s, t = str(e.get("source")), str(e.get("target"))
            if s == t:
                continue
            self.out[s].add(t)
            self.inn[t].add(s)
        self.und = defaultdict(set)
        for s, ts in self.out.items():
            for t in ts:
                self.und[s].add(t)
                self.und[t].add(s)
        # all ids that appear anywhere
        self.ids = set(self.nodes) | set(self.und)
        # index by last component for mention resolution
        self.by_last = defaultdict(set)
        for nid in self.ids:
            self.by_last[nid.rsplit(".", 1)[-1]].add(nid)
            self.by_last[nid.rsplit(".", 1)[-1].lower()].add(nid)
        self._text_lines = {}
        self._comp = None
        self._centrality = None
        # embeddings
        self.emb_ids, self.emb_mat, self.emb_row = [], None, {}
        if epath is not None:
            try:
                z = np.load(epath, allow_pickle=False)
            except ValueError:
                z = np.load(epath, allow_pickle=True)
            keys = list(z.files)
            vecs = []
            for k in keys:
                v = np.asarray(z[k], dtype=np.float32).ravel()
                vecs.append(v)
            if vecs and len({v.shape for v in vecs}) == 1:
                M = np.stack(vecs)
                M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
                self.emb_ids, self.emb_mat = keys, M
                self.emb_row = {k: i for i, k in enumerate(keys)}

    def degree(self, n):
        return len(self.inn.get(n, ())), len(self.out.get(n, ()))

    def text_lines(self, n):
        if n not in self._text_lines:
            self._text_lines[n] = {ln.strip() for ln in self.nodes.get(n, "").splitlines()}
        return self._text_lines[n]

    # -------- module-scoped candidates for a file path
    def module_candidates(self, file_path):
        mod = file_path[:-3].replace("/", ".")
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        parts = mod.split(".")
        for i in range(len(parts)):  # longest suffix first
            pref = ".".join(parts[i:])
            cands = [nid for nid in self.nodes if nid == pref or nid.startswith(pref + ".")]
            if cands:
                return cands, pref
        return [], None

    # -------- components (undirected)
    def component_sizes(self):
        if self._comp is None:
            comp, sizes = {}, {}
            cid = 0
            for n in self.ids:
                if n in comp:
                    continue
                q, members = deque([n]), [n]
                comp[n] = cid
                while q:
                    u = q.popleft()
                    for v in self.und.get(u, ()):
                        if v not in comp:
                            comp[v] = cid
                            members.append(v)
                            q.append(v)
                sizes[cid] = len(members)
                cid += 1
            self._comp = (comp, sizes)
        return self._comp

    def bfs(self, sources, max_depth=None):
        dist = {s: 0 for s in sources if s in self.ids}
        q = deque(dist)
        while q:
            u = q.popleft()
            if max_depth is not None and dist[u] >= max_depth:
                continue
            for v in self.und.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    # -------- centrality (cached per graph)
    def centrality(self):
        if self._centrality is not None:
            return self._centrality
        ids = sorted(self.ids)
        c = {
            "in_degree": {n: len(self.inn.get(n, ())) for n in ids},
            "out_degree": {n: len(self.out.get(n, ())) for n in ids},
            "degree": {n: len(self.und.get(n, ())) for n in ids},
        }
        if nx is not None:
            G = nx.DiGraph()
            G.add_nodes_from(ids)
            for s, ts in self.out.items():
                for t in ts:
                    G.add_edge(s, t)
            try:
                c["pagerank"] = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-6)
            except Exception:  # noqa: BLE001
                pass
            if DO_BETWEENNESS:
                k = min(100, len(ids))
                c["betweenness"] = nx.betweenness_centrality(
                    G.to_undirected(as_view=True), k=k, seed=0, normalized=True)
        self._centrality = c
        return c


# --------------------------------------------------------------------------- gold nodes
def gold_nodes_for_task(task, RG):
    gold, details = set(), []
    for h in parse_patch(task.get("patch")):
        rec = {"file": h["file"], "method": None, "nodes": []}
        if h["new_file"]:
            rec["method"] = "new_file"
            details.append(rec)
            continue
        cands, pref = RG.module_candidates(h["file"])
        rec["module_prefix"] = pref
        if not cands:
            rec["method"] = "module_not_in_graph"
            details.append(rec)
            continue
        probe = meaningful(h["removed"])
        method = "removed_lines"
        if not probe:
            probe = meaningful(h["context"])
            method = "context_lines(addition_only)"
        best, best_score = [], 0
        for nid in cands:
            tl = RG.text_lines(nid)
            score = sum(1 for p in probe if p in tl)
            if score > best_score:
                best, best_score = [nid], score
            elif score == best_score and score > 0:
                best.append(nid)
        if best_score > 0:
            # most specific = shortest source text (method beats enclosing class/module)
            m = min(best, key=lambda n: len(RG.nodes.get(n, "")))
            rec["method"], rec["nodes"] = method, [m]
        else:
            ctx = h["header_ctx"]
            mm = re.search(r"(?:def|class)\s+([A-Za-z_]\w*)", ctx)
            hit = [n for n in cands if mm and n.rsplit(".", 1)[-1] == mm.group(1)] if mm else []
            if hit:
                m = min(hit, key=lambda n: len(RG.nodes.get(n, "")))
                rec["method"], rec["nodes"] = "hunk_header", [m]
            else:
                rec["method"] = "unmatched(module-level or new symbol)"
        gold.update(rec["nodes"])
        details.append(rec)
    return gold, details


# --------------------------------------------------------------------------- seed nodes (issue mentions)
PY_NOISE = set("""
self cls none true false print str int float dict list set tuple bool bytes object type
return import from class def async await with lambda yield raise assert pass break continue
the and for this that when then have from into error errors issue issues example examples
python version versions docs documentation should would could value values default
test tests pytest file files line lines code bug fix fixes pull request main app
""".split())

IDENT_PATTERNS = [
    re.compile(r"`([^`\n]{2,80})`"),                                 # inline code
    re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\b"),              # dotted names
    re.compile(r"\b([A-Za-z_]\w*)\s*\("),                              # calls
    re.compile(r"\b([A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*)\b"),              # CamelCase
    re.compile(r"\b([a-z][a-z0-9]*_[a-z0-9_]+)\b"),                    # snake_case
    re.compile(r", in ([A-Za-z_]\w*)"),                                # traceback frames
    re.compile(r"\b([A-Z][a-z]{3,}[A-Za-z0-9]*)\b"),                  # Capitalized (class names)
]


def extract_mentions(text):
    toks = set()
    for pat in IDENT_PATTERNS:
        for m in pat.finditer(text or ""):
            raw = m.group(1)
            for piece in re.findall(r"[A-Za-z_][\w.]*", raw):
                piece = piece.strip(".")
                if len(piece) >= 3 and piece.lower() not in PY_NOISE:
                    toks.add(piece)
    return toks


def resolve_mentions(tokens, RG):
    seeds, ambiguous, per_token = set(), [], {}
    for t in tokens:
        if t in RG.ids:
            hits = {t}
        elif "." in t:
            hits = {n for n in RG.ids if n.endswith("." + t)}
            if not hits:  # fall back to last component
                hits = set(RG.by_last.get(t.rsplit(".", 1)[-1], ()))
        else:
            hits = set(RG.by_last.get(t, ()))
        if not hits:
            continue
        if len(hits) > MAX_SEED_MATCHES:
            ambiguous.append(t)
            continue
        per_token[t] = sorted(hits)
        seeds |= hits
    return seeds, per_token, ambiguous


# --------------------------------------------------------------------------- embedding helpers
def emb_rank(RG, seed, target):
    """Rank (0 = top) of target when calling search_similar_code(seed). None if missing."""
    if RG.emb_mat is None or seed not in RG.emb_row or target not in RG.emb_row:
        return None
    sims = RG.emb_mat @ RG.emb_mat[RG.emb_row[seed]]
    sims[RG.emb_row[seed]] = -np.inf  # the tool would return the seed itself; ignore it
    return int((sims > sims[RG.emb_row[target]]).sum())


def cosine_by_hop(RG, n_sources=200, per_source=15):
    """Mean cosine similarity of node pairs bucketed by undirected call-graph distance."""
    if RG.emb_mat is None:
        return {}
    pool = [n for n in RG.ids if n in RG.emb_row]
    if len(pool) < 10:
        return {}
    buckets = defaultdict(list)
    for s in random.sample(pool, min(n_sources, len(pool))):
        dist = RG.bfs([s])
        vs = RG.emb_mat[RG.emb_row[s]]
        for t in random.sample(pool, min(per_source, len(pool))):
            if t == s:
                continue
            d = dist.get(t)
            key = "unreachable" if d is None else (str(d) if d <= 4 else "5+")
            buckets[key].append(float(vs @ RG.emb_mat[RG.emb_row[t]]))
        # also sample from the near neighbourhood so small distances are represented
        near = [t for t, d in dist.items() if 1 <= d <= 3 and t in RG.emb_row]
        for t in random.sample(near, min(per_source, len(near))):
            buckets[str(dist[t])].append(float(vs @ RG.emb_mat[RG.emb_row[t]]))
    return {k: {"n": len(v), "mean_cos": round(float(np.mean(v)), 4)} for k, v in buckets.items()}


# --------------------------------------------------------------------------- centrality helpers
def percentile(values_dict, node, universe):
    vals = [values_dict.get(u, 0.0) for u in universe]
    x = values_dict.get(node, 0.0)
    below = sum(1 for v in vals if v < x)
    equal = sum(1 for v in vals if v == x)
    return (below + 0.5 * equal) / max(len(vals), 1)  # = per-positive AUC contribution


# --------------------------------------------------------------------------- main
def main():
    task_files, graph_paths, emb_paths = find_files()
    tasks = []
    for tf in task_files:
        tasks += [json.loads(l) for l in tf.read_text().splitlines() if l.strip()]
    log(f"{len(tasks)} tasks, {len(graph_paths)} graphs, {len(emb_paths)} embeddings")

    cache = {}
    per_task = []
    cos_hop_by_repo = {}

    for i, task in enumerate(tasks):
        commit = task["base_commit"]
        if commit not in graph_paths:
            per_task.append({"id": task["instance_id"], "repo": task["repo"], "error": "no graph"})
            continue
        if commit not in cache:
            cache.clear()  # keep memory flat: tasks mostly have distinct commits
            cache[commit] = RepoGraph(graph_paths[commit], emb_paths.get(commit))
        RG = cache[commit]
        repo_short = task["repo"].split("/")[-1].lower()

        gold, hunk_details = gold_nodes_for_task(task, RG)
        tokens = extract_mentions(task.get("problem_statement", "") + "\n" + task.get("hints_text", ""))
        seeds, per_token, ambiguous = resolve_mentions(tokens, RG)
        comp, comp_sizes = RG.component_sizes()

        # Q1 structure of gold nodes
        gold_info = []
        for g in sorted(gold):
            din, dout = RG.degree(g)
            gold_info.append({
                "node": g, "in_deg": din, "out_deg": dout,
                "isolated": din + dout == 0,
                "component_size": comp_sizes.get(comp.get(g), 1),
                "in_embeddings": g in RG.emb_row,
            })

        # Q2 reachability (undirected hops, seeds -> any gold)
        direct = sorted(gold & seeds)
        dist = RG.bfs(seeds) if seeds else {}
        hops = [dist[g] for g in gold if g in dist]
        min_hops = min(hops) if hops else None

        # Q3 embedding vs graph (non-direct seeds only, simulates one tool call each)
        best_rank = None
        graph_1hop = False
        for s in seeds:
            nbrs = list(RG.und.get(s, ()))[:NEIGHBOR_CAP]
            if gold & set(nbrs):
                graph_1hop = True
            for g in gold:
                if g == s:
                    continue
                r = emb_rank(RG, s, g)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r

        # Q4 centrality percentiles
        cent = RG.centrality() if gold else {}
        universe_all = sorted(RG.ids)
        universe_lib = [n for n in universe_all if n.split(".")[0].lower() == repo_short] or universe_all
        cent_pct = {}
        for metric, vals in cent.items():
            cent_pct[metric] = {
                "all": [round(percentile(vals, g, universe_all), 4) for g in gold],
                "library": [round(percentile(vals, g, universe_lib), 4)
                            for g in gold if g in set(universe_lib)],
            }

        if task["repo"] not in cos_hop_by_repo and RG.emb_mat is not None:
            cos_hop_by_repo[task["repo"]] = cosine_by_hop(RG)

        per_task.append({
            "id": task["instance_id"], "repo": task["repo"],
            "n_hunks": len(hunk_details),
            "hunk_methods": [h["method"] for h in hunk_details],
            "gold": gold_info,
            "n_mention_tokens": len(tokens),
            "seeds": per_token, "ambiguous_mentions": ambiguous[:20],
            "n_seeds": len(seeds),
            "gold_directly_mentioned": direct,
            "min_hops_seed_to_gold": min_hops,
            "graph_1hop_hit": graph_1hop,
            "emb_best_rank": best_rank,
            "emb_top10_hit": best_rank is not None and best_rank < EMB_TOPK,
            "centrality_percentiles": cent_pct,
        })
        if (i + 1) % 10 == 0:
            log(f"processed {i + 1}/{len(tasks)}")

    summary = summarize(per_task, cos_hop_by_repo)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "gold_graph_per_task.json").write_text(json.dumps(per_task, indent=1))
    (OUT_DIR / "gold_graph_summary.json").write_text(json.dumps(summary, indent=2))
    print_report(summary)
    log(f"saved {OUT_DIR / 'gold_graph_summary.json'} and gold_graph_per_task.json")


# --------------------------------------------------------------------------- aggregation
def pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def summarize(per_task, cos_hop_by_repo):
    ok = [t for t in per_task if "error" not in t]
    S = {"n_tasks": len(per_task), "n_analyzed": len(ok)}

    methods = Counter(m for t in ok for m in t["hunk_methods"])
    tasks_with_gold = [t for t in ok if t["gold"]]
    S["A_gold_mapping"] = {
        "hunk_match_methods": dict(methods),
        "tasks_with_>=1_gold_node": len(tasks_with_gold),
        "tasks_with_no_gold_node": [t["id"] for t in ok if not t["gold"]],
        "gold_nodes_total": sum(len(t["gold"]) for t in ok),
    }

    golds = [g for t in ok for g in t["gold"]]
    degs = [g["in_deg"] + g["out_deg"] for g in golds]
    S["B_gold_structure"] = {
        "gold_nodes": len(golds),
        "isolated_pct": pct(sum(g["isolated"] for g in golds), len(golds)),
        "zero_in_degree_pct (nobody calls it)": pct(sum(g["in_deg"] == 0 for g in golds), len(golds)),
        "median_degree": float(np.median(degs)) if degs else None,
        "in_embeddings_pct": pct(sum(g["in_embeddings"] for g in golds), len(golds)),
        "tasks_where_ALL_gold_isolated": pct(
            sum(1 for t in tasks_with_gold if all(g["isolated"] for g in t["gold"])), len(tasks_with_gold)),
    }

    S["C_issue_mentions"] = {
        "tasks_with_>=1_resolved_seed_pct": pct(sum(t["n_seeds"] > 0 for t in tasks_with_gold), len(tasks_with_gold)),
        "median_seeds_per_task": float(np.median([t["n_seeds"] for t in tasks_with_gold])) if tasks_with_gold else None,
        "tasks_gold_directly_mentioned_pct": pct(
            sum(bool(t["gold_directly_mentioned"]) for t in tasks_with_gold), len(tasks_with_gold)),
    }

    def hop_bucket(h):
        if h is None:
            return "unreachable/no-seed"
        return str(h) if h <= 3 else "4+"
    hop_all = Counter(hop_bucket(t["min_hops_seed_to_gold"]) for t in tasks_with_gold)
    hop_repo = defaultdict(Counter)
    for t in tasks_with_gold:
        hop_repo[t["repo"]][hop_bucket(t["min_hops_seed_to_gold"])] += 1
    S["D_reachability_min_hops_seed_to_gold"] = {
        "all_tasks": dict(hop_all),
        "by_repo": {r: dict(c) for r, c in hop_repo.items()},
        "note": "0 = gold symbol named in the issue; undirected call-graph hops",
    }

    # Q3 complementarity on tasks with seeds but gold not directly mentioned
    elig = [t for t in tasks_with_gold if t["n_seeds"] > 0 and not t["gold_directly_mentioned"]]
    both = sum(t["graph_1hop_hit"] and t["emb_top10_hit"] for t in elig)
    g_only = sum(t["graph_1hop_hit"] and not t["emb_top10_hit"] for t in elig)
    e_only = sum(t["emb_top10_hit"] and not t["graph_1hop_hit"] for t in elig)
    neither = len(elig) - both - g_only - e_only
    ranks = [t["emb_best_rank"] for t in elig if t["emb_best_rank"] is not None]
    S["E_embedding_vs_graph"] = {
        "eligible_tasks (seeds, gold not named)": len(elig),
        "graph_1hop_hit_pct": pct(both + g_only, len(elig)),
        "embedding_top10_hit_pct": pct(both + e_only, len(elig)),
        "2x2": {"both": both, "graph_only": g_only, "embedding_only": e_only, "neither": neither},
        "embedding_best_rank_median": float(np.median(ranks)) if ranks else None,
        "cosine_by_graph_distance (per repo, sampled pairs)": cos_hop_by_repo,
    }

    metrics = sorted({m for t in ok for m in t["centrality_percentiles"]})
    cent = {}
    for m in metrics:
        allv = [p for t in ok for p in t["centrality_percentiles"].get(m, {}).get("all", [])]
        libv = [p for t in ok for p in t["centrality_percentiles"].get(m, {}).get("library", [])]
        cent[m] = {
            "AUC_vs_all_nodes": round(float(np.mean(allv)), 3) if allv else None,
            "AUC_vs_library_nodes": round(float(np.mean(libv)), 3) if libv else None,
            "n": len(allv),
        }
    S["F_centrality_AUC"] = {
        "metrics": cent,
        "note": "mean percentile of gold nodes = AUC; 0.5 = no better than random; "
                "gold = edited, not necessarily 'buggy' (includes features/refactors)",
    }
    return S


def print_report(S):
    print("\n" + "=" * 72)
    print("GOLD-LOCATION GRAPH ANALYSIS")
    print("=" * 72)
    for sec in ["A_gold_mapping", "B_gold_structure", "C_issue_mentions",
                "D_reachability_min_hops_seed_to_gold", "E_embedding_vs_graph", "F_centrality_AUC"]:
        print(f"\n--- {sec}")
        for k, v in S[sec].items():
            if isinstance(v, dict) and len(json.dumps(v)) > 110:
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"      {kk}: {vv}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
