"""
Gemma 4 Developer Agent — verification pass for the gold-location findings.

Checks three claims from gold_graph_analysis.py:

  V1  Edge symmetry      Is every `calls` edge stored in both directions?
                         If so, can the true direction be recovered from source text?
  V2  Embedding origin   Are the node embeddings derived from graph structure,
                         from code text, or both? (anisotropy, hop vs text
                         correlation, duplicate-code test, neighbour-mean test)
  V3  Clean seeds        Re-run issue -> gold reachability with de-noised mentions
                         (template text stripped, generic names dropped, test
                         nodes and hub nodes excluded), incl. 2-hop and hub-free BFS.

Needs gold_graph_analysis.py in the same folder (it reuses its parsers).
Run in a Kaggle notebook with the competition dataset attached (CPU is fine):
    !python verify_graph.py
Env vars: DATA_ROOT, OUT_DIR (same defaults as before).
Output: OUT_DIR/verify_summary.json (upload it back).
"""

import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gold_graph_analysis as gga  # noqa: E402

random.seed(0)
np.random.seed(0)
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return round(float((ra * rb).sum() / d), 3) if d else None


def short_name(nid):
    parts = nid.split(".")
    if parts[-1] == "__init__" and len(parts) > 1:  # constructor call looks like ClassName(
        return parts[-2]
    return parts[-1]


# =========================================================================== V1 symmetry
def v1_symmetry(graph_paths, latest_by_repo):
    log("V1: edge symmetry on all graphs")
    per_graph = []
    for commit, gp in sorted(graph_paths.items()):
        g = json.loads(Path(gp).read_text())
        nodes = {str(n.get("id")) for n in g.get("nodes", [])}
        edges = g.get("edges") or g.get("links") or []
        pairs, selfloops, parallel, dangling = set(), 0, 0, 0
        for e in edges:
            s, t = str(e.get("source")), str(e.get("target"))
            if s == t:
                selfloops += 1
                continue
            if (s, t) in pairs:
                parallel += 1
            pairs.add((s, t))
            if s not in nodes or t not in nodes:
                dangling += 1
        recip = sum(1 for (s, t) in pairs if (t, s) in pairs)
        per_graph.append({
            "graph": Path(gp).stem, "raw_edges": len(edges), "unique_directed_pairs": len(pairs),
            "reciprocal_pct": gga.pct(recip, len(pairs)), "self_loops": selfloops,
            "parallel_duplicates": parallel, "edges_with_endpoint_not_in_nodes": dangling,
        })

    # direction recovery from text on one graph per repo
    recovery = {}
    for repo, commit in latest_by_repo.items():
        g = json.loads(Path(graph_paths[commit]).read_text())
        text = {str(n.get("id")): n.get("text") or "" for n in g.get("nodes", [])}
        und = set()
        for e in g.get("edges") or g.get("links") or []:
            s, t = str(e.get("source")), str(e.get("target"))
            if s != t:
                und.add(tuple(sorted((s, t))))
        und = list(und)
        random.shuffle(und)
        und = und[:20000]
        c = Counter()
        examples = []
        for a, b in und:
            a_calls_b = bool(re.search(r"\b" + re.escape(short_name(b)) + r"\s*\(", text.get(a, "")))
            b_calls_a = bool(re.search(r"\b" + re.escape(short_name(a)) + r"\s*\(", text.get(b, "")))
            key = ("both" if a_calls_b and b_calls_a else
                   "one_direction" if a_calls_b or b_calls_a else "neither")
            c[key] += 1
            if key == "one_direction" and len(examples) < 5:
                caller, callee = (a, b) if a_calls_b else (b, a)
                examples.append(f"{caller}  ->  {callee}")
        n = sum(c.values())
        recovery[repo] = {
            "pairs_checked": n,
            "text_supports_exactly_one_direction_pct": gga.pct(c["one_direction"], n),
            "both_directions_pct": gga.pct(c["both"], n),
            "neither_pct (method refs, decorators, attribute calls...)": gga.pct(c["neither"], n),
            "examples": examples,
        }

    rp = [g["reciprocal_pct"] for g in per_graph if g["reciprocal_pct"] is not None]
    return {
        "graphs": len(per_graph),
        "reciprocal_pct_min_median_max": [min(rp), float(np.median(rp)), max(rp)] if rp else None,
        "graphs_fully_symmetric (>=99.9%)": sum(1 for x in rp if x >= 99.9),
        "total_self_loops": sum(g["self_loops"] for g in per_graph),
        "total_parallel_duplicates": sum(g["parallel_duplicates"] for g in per_graph),
        "total_dangling_edges": sum(g["edges_with_endpoint_not_in_nodes"] for g in per_graph),
        "direction_recovery_from_text": recovery,
        "per_graph_sample": per_graph[:5],
    }


# =========================================================================== V2 embeddings
SUBTOK = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def code_tokens(text):
    toks = []
    for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        for part in ident.split("_"):
            toks += [p.lower() for p in SUBTOK.findall(part) if len(p) > 1]
    return toks


def tfidf_matrix(texts):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(analyzer=code_tokens, sublinear_tf=True, min_df=1)
        return vec.fit_transform(texts)  # rows are L2-normalised
    except ImportError:
        return None


def v2_embeddings(graph_paths, emb_paths, latest_by_repo):
    log("V2: embedding origin, one graph per repo")
    out = {}
    for repo, commit in latest_by_repo.items():
        RG = gga.RepoGraph(graph_paths[commit], emb_paths.get(commit))
        if RG.emb_mat is None:
            out[repo] = {"error": "no embeddings"}
            continue
        ids = [n for n in RG.emb_ids if n in RG.ids]
        rows = np.array([RG.emb_row[n] for n in ids])
        E = RG.emb_mat[rows]
        Ec = E - E.mean(axis=0)
        Ec /= (np.linalg.norm(Ec, axis=1, keepdims=True) + 1e-8)
        idx = {n: i for i, n in enumerate(ids)}
        X = tfidf_matrix([RG.nodes.get(n, "") for n in ids])

        def emb_raw(i, j):
            return float(E[i] @ E[j])

        def emb_c(i, j):
            return float(Ec[i] @ Ec[j])

        def txt(i, j):
            return float(X[i].multiply(X[j]).sum()) if X is not None else None

        # anisotropy
        rnd = [(random.randrange(len(ids)), random.randrange(len(ids))) for _ in range(4000)]
        rnd = [(i, j) for i, j in rnd if i != j]
        aniso = {"mean_raw_cos_random_pairs": round(float(np.mean([emb_raw(i, j) for i, j in rnd])), 4),
                 "mean_centered_cos_random_pairs": round(float(np.mean([emb_c(i, j) for i, j in rnd])), 4)}

        # hop-bucketed pairs
        pairs = []
        for s in random.sample(ids, min(150, len(ids))):
            dist = RG.bfs([s])
            for t in random.sample(ids, min(20, len(ids))):
                if t != s:
                    pairs.append((idx[s], idx[t], dist.get(t)))
            near = [t for t, d in dist.items() if 1 <= d <= 3 and t in idx]
            for t in random.sample(near, min(10, len(near))):
                pairs.append((idx[s], idx[t], dist[t]))
        by_hop = defaultdict(lambda: {"emb_c": [], "txt": []})
        ec_all, tx_all, hop_all, ec_reach = [], [], [], []
        for i, j, d in pairs:
            key = "unreachable" if d is None else (str(d) if d <= 4 else "5+")
            e, x = emb_c(i, j), txt(i, j)
            by_hop[key]["emb_c"].append(e)
            if x is not None:
                by_hop[key]["txt"].append(x)
                ec_all.append(e)
                tx_all.append(x)
            if d is not None:
                ec_reach.append(e)
                hop_all.append(-d)
        hop_table = {k: {"n": len(v["emb_c"]),
                         "mean_centered_emb_cos": round(float(np.mean(v["emb_c"])), 4),
                         "mean_text_tfidf_cos": round(float(np.mean(v["txt"])), 4) if v["txt"] else None,
                         "spearman_emb_vs_text_within_bucket": spearman(v["emb_c"][:len(v["txt"])], v["txt"])
                         if v["txt"] else None}
                     for k, v in sorted(by_hop.items())}

        # duplicate-code test: identical source text in different nodes
        groups = defaultdict(list)
        for n in ids:
            norm = re.sub(r"\s+", " ", RG.nodes.get(n, "")).strip()
            if len(norm) >= 40:
                groups[norm].append(n)
        dup_pairs = []
        for members in groups.values():
            if 2 <= len(members) <= 12:
                for a in range(len(members)):
                    for b in range(a + 1, len(members)):
                        dup_pairs.append((idx[members[a]], idx[members[b]]))
        random.shuffle(dup_pairs)
        dup_pairs = dup_pairs[:2000]
        # control: isolated nodes may share a near-constant vector, which would fake "text similarity"
        iso = [idx[n] for n in ids if not RG.und.get(n)]
        iso_rnd = [(random.choice(iso), random.choice(iso)) for _ in range(2000)] if len(iso) > 1 else []
        iso_rnd = [(i, j) for i, j in iso_rnd if i != j]
        conn_dup = [(i, j) for i, j in dup_pairs if RG.und.get(ids[i]) and RG.und.get(ids[j])]
        dup = {"pairs": len(dup_pairs),
               "control_random_isolated_pairs_cos": round(float(np.mean([emb_c(i, j) for i, j in iso_rnd])), 4)
               if iso_rnd else None,
               "dup_pairs_both_connected": len(conn_dup),
               "dup_pairs_both_connected_cos": round(float(np.mean([emb_c(i, j) for i, j in conn_dup])), 4)
               if conn_dup else None,
               "mean_centered_emb_cos": round(float(np.mean([emb_c(i, j) for i, j in dup_pairs])), 4)
               if dup_pairs else None,
               "graph_connected_pct": gga.pct(
                   sum(1 for i, j in dup_pairs if ids[j] in RG.bfs([ids[i]], max_depth=6)), len(dup_pairs))
               if dup_pairs else None}

        # neighbour-mean test: is a node's vector ~ the mean of its neighbours' vectors?
        nm, rm = [], []
        cand = [n for n in ids if len([v for v in RG.und.get(n, ()) if v in idx]) >= 2]
        for n in random.sample(cand, min(1500, len(cand))):
            nb = [idx[v] for v in RG.und[n] if v in idx]
            m = Ec[nb].mean(axis=0)
            nm.append(float(Ec[idx[n]] @ m / (np.linalg.norm(m) + 1e-8)))
            rs = random.sample(range(len(ids)), len(nb))
            r = Ec[rs].mean(axis=0)
            rm.append(float(Ec[idx[n]] @ r / (np.linalg.norm(r) + 1e-8)))

        # top-1 embedding neighbour: is it a graph neighbour? text-similar?
        top1_graph, top1_text, top1_text_rand, iso_top1_text = [], [], [], []
        sample = random.sample(range(len(ids)), min(1500, len(ids)))
        for i in sample:
            sims = Ec @ Ec[i]
            sims[i] = -np.inf
            j = int(np.argmax(sims))
            top1_graph.append(ids[j] in RG.und.get(ids[i], ()))
            x = txt(i, j)
            if x is not None:
                top1_text.append(x)
                top1_text_rand.append(txt(i, random.randrange(len(ids))))
                if not RG.und.get(ids[i]):
                    iso_top1_text.append(x)

        out[repo] = {
            "graph": Path(graph_paths[commit]).stem,
            "nodes_with_embeddings": len(ids),
            "anisotropy": aniso,
            "spearman_emb_vs_text_all_pairs": spearman(ec_all, tx_all),
            "spearman_emb_vs_neg_hop_reachable_pairs": spearman(ec_reach, hop_all),
            "by_hop": hop_table,
            "duplicate_code_test": dup,
            "neighbour_mean_test": {
                "mean_cos_to_neighbour_mean": round(float(np.mean(nm)), 4) if nm else None,
                "mean_cos_to_random_set_mean": round(float(np.mean(rm)), 4) if rm else None},
            "top1_embedding_neighbour": {
                "is_graph_neighbour_pct": gga.pct(sum(top1_graph), len(top1_graph)),
                "mean_text_cos": round(float(np.mean(top1_text)), 4) if top1_text else None,
                "mean_text_cos_random_baseline": round(float(np.mean(top1_text_rand)), 4) if top1_text_rand else None,
                "isolated_nodes_mean_text_cos": round(float(np.mean(iso_top1_text)), 4) if iso_top1_text else None,
            },
        }
        log(f"  {repo} done")
    out["how_to_read"] = (
        "Structure-derived: centered cos drops with hop, duplicate-code pairs NOT similar, "
        "high neighbour-mean cos, top-1 often a graph neighbour. Text-derived: duplicate-code "
        "pairs ~1.0, strong emb-vs-text correlation even within a hop bucket, isolated nodes "
        "still get text-similar top-1 neighbours.")
    return out


# =========================================================================== V3 clean seeds
TEMPLATE_LINE = re.compile(
    r"^\s*([-*]\s*\[[ xX]\]|#{1,6}\s|>\s)|"
    r"(searched|discussion|descriptive title|i already|i used the github search|"
    r"operating system|python version|fastapi version|pydantic version|to reproduce|"
    r"additional context|description|privileged|checklist)", re.I)
URL = re.compile(r"https?://\S+|\[[^\]]*\]\([^)]*\)")
GENERIC = set("""
get post put patch delete head options run call main func function app api link server
response request data value item items name type obj args kwargs self cls result results
error errors test tests event events model models field fields config settings
""".split())


def clean_issue(text):
    lines = [ln for ln in (text or "").splitlines() if not TEMPLATE_LINE.search(ln)]
    return URL.sub(" ", "\n".join(lines))


def clean_seeds(text, RG, hubs):
    toks = gga.extract_mentions(clean_issue(text))
    toks = {t for t in toks if t.split(".")[-1].lower() not in GENERIC}
    seeds, _, _ = gga.resolve_mentions(toks, RG)
    return {s for s in seeds
            if not (s.startswith("tests.") or ".tests." in s or s.split(".")[0].startswith("test_"))
            and s not in hubs}


def bfs_avoiding(RG, sources, blocked):
    """BFS that may END on a blocked (hub) node but never expands through one."""
    dist = {s: 0 for s in sources if s in RG.ids}
    q = deque(dist)
    while q:
        u = q.popleft()
        if u in blocked and dist[u] > 0:
            continue
        for v in RG.und.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def v3_clean_seeds(tasks, graph_paths, emb_paths):
    log("V3: reachability with cleaned seeds")
    cache, rows = {}, []
    for task in tasks:
        commit = task["base_commit"]
        if commit not in graph_paths:
            continue
        if commit not in cache:
            cache.clear()
            RG = gga.RepoGraph(graph_paths[commit], emb_paths.get(commit))
            degs = sorted(len(RG.und.get(n, ())) for n in RG.ids)
            cut = degs[int(0.99 * (len(degs) - 1))] if degs else 10 ** 9
            hubs = {n for n in RG.ids if len(RG.und.get(n, ())) > max(cut, 10)}
            cache[commit] = (RG, hubs)
        RG, hubs = cache[commit]
        gold, _ = gga.gold_nodes_for_task(task, RG)
        if not gold:
            continue
        text = task.get("problem_statement", "") + "\n" + task.get("hints_text", "")
        seeds = clean_seeds(text, RG, hubs)
        d = RG.bfs(seeds) if seeds else {}
        dh = bfs_avoiding(RG, seeds, hubs) if seeds else {}
        hop = min((d[g] for g in gold if g in d), default=None)
        hop_nohub = min((dh[g] for g in gold if g in dh), default=None)
        best_rank = None
        for s in seeds:
            for g in gold - {s}:
                r = gga.emb_rank(RG, s, g)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
        rows.append({"id": task["instance_id"], "repo": task["repo"], "n_seeds": len(seeds),
                     "direct": bool(gold & seeds), "hop": hop, "hop_no_hubs": hop_nohub,
                     "emb_rank": best_rank, "n_hubs": len(hubs)})

    def bucket(h):
        return "unreachable/no-seed" if h is None else (str(h) if h <= 3 else "4+")
    n = len(rows)
    seeded = [r for r in rows if r["n_seeds"]]
    elig = [r for r in seeded if not r["direct"]]
    g1 = [r for r in elig if r["hop"] == 1]
    g2 = [r for r in elig if r["hop"] is not None and r["hop"] <= 2]
    e10 = [r for r in elig if r["emb_rank"] is not None and r["emb_rank"] < gga.EMB_TOPK]
    either = [r for r in elig if (r["hop"] is not None and r["hop"] <= 2)
              or (r["emb_rank"] is not None and r["emb_rank"] < gga.EMB_TOPK)]
    return {
        "tasks_with_gold": n,
        "seeded_pct": gga.pct(len(seeded), n),
        "gold_directly_named_pct": gga.pct(sum(r["direct"] for r in rows), n),
        "hop_hist_all_tasks": dict(Counter(bucket(r["hop"]) for r in rows)),
        "hop_hist_seeded_only": dict(Counter(bucket(r["hop"]) for r in seeded)),
        "hop_hist_seeded_hub_free_bfs": dict(Counter(bucket(r["hop_no_hubs"]) for r in seeded)),
        "eligible (seeded, gold not named)": len(elig),
        "graph_1hop_hit_pct": gga.pct(len(g1), len(elig)),
        "graph_<=2hop_hit_pct": gga.pct(len(g2), len(elig)),
        "embedding_top10_hit_pct": gga.pct(len(e10), len(elig)),
        "graph<=2hop_OR_emb_top10_pct": gga.pct(len(either), len(elig)),
        "by_repo_seeded_pct": {rep: gga.pct(sum(1 for r in rows if r["repo"] == rep and r["n_seeds"]),
                                            sum(1 for r in rows if r["repo"] == rep))
                               for rep in sorted({r["repo"] for r in rows})},
        "median_hubs_per_graph": float(np.median([r["n_hubs"] for r in rows])) if rows else None,
        "per_task": rows,
    }


# =========================================================================== main
def main():
    task_files, graph_paths, emb_paths = gga.find_files()
    tasks = []
    for tf in task_files:
        tasks += [json.loads(l) for l in tf.read_text().splitlines() if l.strip()]
    latest_by_repo = {}
    for t in sorted(tasks, key=lambda t: t.get("created_at", "")):
        if t["base_commit"] in graph_paths:
            latest_by_repo[t["repo"]] = t["base_commit"]
    log(f"{len(tasks)} tasks, {len(graph_paths)} graphs; repos: {list(latest_by_repo)}")

    S = {"V1_edge_symmetry": v1_symmetry(graph_paths, latest_by_repo),
         "V2_embedding_origin": v2_embeddings(graph_paths, emb_paths, latest_by_repo),
         "V3_clean_seed_reachability": v3_clean_seeds(tasks, graph_paths, emb_paths)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "verify_summary.json").write_text(json.dumps(S, indent=2))

    print("\n" + "=" * 72)
    for sec, body in S.items():
        print(f"\n### {sec}")
        for k, v in body.items():
            if k in ("per_task", "per_graph_sample"):
                continue
            print(f"  {k}: {json.dumps(v)[:600]}")
    log(f"saved {OUT_DIR / 'verify_summary.json'}")


if __name__ == "__main__":
    main()
