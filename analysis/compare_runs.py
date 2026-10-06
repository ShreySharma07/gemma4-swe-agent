"""Paired comparison of evaluation runs plus behaviour metrics from ATIF traces. Prints a compact summary only.

Usage:
  python analysis/compare_runs.py data/runs/r2_v2_sed data/runs/r3_v5_graph [more runs...] [--json out.json]
Each run dir must contain task_results.jsonl and (optionally) traces/*.json. The layout of an extracted
official_l4_*.tgz works as-is (pass .../merged).
"""
import argparse
import collections
import glob
import json
import math
import os
import re

GRAPH = {"search_similar_code", "get_code_neighbors", "get_code_subgraph"}
EDIT = {"edit_file", "write_file"}


def load_results(d):
    p = os.path.join(d, "task_results.jsonl")
    return {json.loads(l)["instance_id"]: json.loads(l) for l in open(p) if l.strip()}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)


def trace_metrics(d):
    rows = {}
    for f in glob.glob(os.path.join(d, "traces", "*.json")):
        iid = os.path.basename(f)[len("trace_"):-len(".json")]
        try:
            steps = json.load(open(f)).get("steps", [])
        except Exception:  # noqa: BLE001
            continue
        m = collections.Counter()
        seen = collections.Counter()
        by_author = collections.Counter()
        edited = False
        tests_after = 0
        for s in steps:
            obs = str((s.get("observation") or {}).get("content", "") or "")
            for tc in s.get("tool_calls") or []:
                name, args = tc.get("function_name"), tc.get("arguments") or {}
                author = (tc.get("extra") or {}).get("author") or (s.get("extra") or {}).get("author") or "?"
                by_author[author] += 1
                m["calls"] += 1
                key = name + json.dumps(args, sort_keys=True)
                seen[key] += 1
                m["repeats"] += seen[key] > 1
                if name in EDIT and not str(args.get("path", args.get("file_path", ""))).startswith("/tmp"):
                    m["edits"] += 1
                    if '"status": "ok"' in obs:
                        m["edits_ok"] += 1
                        edited, tests_after = True, 0
                    elif "mandatory input parameters" in obs:
                        m["edits_malformed"] += 1
                if name == "run_command" and edited and re.search(r"pytest|python\s+\S*test|/tmp/\w+\.py",
                                                                   str(args.get("command", ""))):
                    tests_after += 1
                if name in GRAPH:
                    m["graph_calls"] += 1
                    mm = re.search(r'"count": (\d+)', obs)
                    m["graph_nonempty"] += bool(mm and int(mm.group(1)) > 0)
                if name == "run_skill_script":
                    m["skill_calls"] += 1
                if name == "submit_patch":
                    m["submitted"] = 1
        m["tested_after_last_edit"] = int(tests_after > 0)
        rows[iid] = {"m": m, "by_author": by_author}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    names = [os.path.basename(os.path.normpath(r)) if os.path.basename(os.path.normpath(r)) != "merged"
             else os.path.basename(os.path.dirname(os.path.normpath(r))) for r in a.runs]
    res = [load_results(r) for r in a.runs]
    out = {"runs": {}}
    print("== outcomes")
    for n, R in zip(names, res):
        rr = list(R.values())
        calls = sorted(r.get("tool_calls", 0) for r in rr)
        row = {"n": len(rr), "resolved": sum(r["resolved"] for r in rr),
               "empty_patch": sum(r.get("agent_patch_size", 0) == 0 for r in rr),
               "median_calls": calls[len(calls) // 2] if calls else None,
               "timeouts": sum("timeout" in str(r.get("error", "")).lower() for r in rr),
               "agent_hours": round(sum(r.get("duration_seconds", 0) for r in rr) / 3600, 2)}
        out["runs"][n] = row
        print(f"{n:24s} " + "  ".join(f"{k}={v}" for k, v in row.items()))
    if len(res) > 1:
        print("== paired (McNemar, common tasks)")
        for i in range(len(res)):
            for j in range(i + 1, len(res)):
                com = set(res[i]) & set(res[j])
                b = sum(res[i][t]["resolved"] and not res[j][t]["resolved"] for t in com)
                c = sum(res[j][t]["resolved"] and not res[i][t]["resolved"] for t in com)
                p = mcnemar(b, c)
                print(f"{names[i]} vs {names[j]}: n={len(com)} only_{names[i]}={b} only_{names[j]}={c} p={p:.3f}")
                out.setdefault("paired", []).append({"a": names[i], "b": names[j], "n": len(com), "only_a": b,
                                                     "only_b": c, "p": p})
    print("== behaviour (from traces)")
    for n, d in zip(names, a.runs):
        tm = trace_metrics(d)
        if not tm:
            print(f"{n}: no traces")
            continue
        tot = collections.Counter()
        auth = collections.Counter()
        for v in tm.values():
            tot.update(v["m"])
            auth.update(v["by_author"])
        k = len(tm)
        b = {"tasks": k, "repeat_share": round(tot["repeats"] / max(1, tot["calls"]), 3),
             "edit_malformed_share": round(tot["edits_malformed"] / max(1, tot["edits"]), 3),
             "tested_after_last_edit": tot["tested_after_last_edit"], "submitted": tot["submitted"],
             "graph_calls": tot["graph_calls"], "graph_nonempty": tot["graph_nonempty"],
             "skill_calls": tot["skill_calls"],
             "calls_by_agent": {x: round(y / k, 1) for x, y in auth.most_common(4)}}
        out["runs"][n]["behaviour"] = b
        print(f"{n}: " + "  ".join(f"{x}={y}" for x, y in b.items()))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
