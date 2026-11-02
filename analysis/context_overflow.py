"""G1 context-overflow diagnosis: per-task context growth (prompt tokens, completions, tool outputs) per arm.
Usage: python3 -I analysis/context_overflow.py <dir with extracted eval_gate_*_b* blocks>
"""
import ast, glob, json, statistics as S, sys
root = sys.argv[1]
def lit(x):
    if isinstance(x, (dict, list)) or x is None: return x
    try: return ast.literal_eval(x)
    except Exception: return None
res = {}
for arm in ("base", "lora"):
    for f in glob.glob(f"{root}/eval_gate_{arm}_b*/rep0/merged/task_results.jsonl"):
        for l in open(f):
            r = json.loads(l); res[(arm, r["instance_id"])] = r
rows = []
for f in glob.glob(f"{root}/eval_gate_*_b*/rep0/merged/traces/*.json"):
    arm = f.split("eval_gate_")[1].split("_")[0]
    t = json.load(open(f)); iid = f.rsplit("trace_", 1)[1][:-5]
    ag = [s for s in t["steps"] if s.get("source") == "agent"]
    pt, ct, obs, think_chars, msg_chars, authors, cmds = [], [], [], 0, 0, {}, []
    for s in ag:
        m = lit(s.get("metrics")) or {}
        if m.get("prompt_tokens"): pt.append(m["prompt_tokens"]); ct.append(m.get("completion_tokens") or 0)
        o = lit(s.get("observation"))
        oc = len(str(o.get("content"))) if isinstance(o, dict) else len(str(s.get("observation") or ""))
        obs.append(oc)
        msg_chars += len(s.get("message") or "")
        e = lit(s.get("extra")) or {}
        authors[e.get("author")] = authors.get(e.get("author"), 0) + 1
        tc = lit(s.get("tool_calls")) or []
        for c in tc:
            cmds.append(c.get("function_name"))
    r = res.get((arm, iid), {})
    rows.append({"arm": arm, "id": iid, "ovf": "ContextWindow" in str(r.get("error")), "res": bool(r.get("resolved")),
                 "steps": len(ag), "max_pt": max(pt) if pt else 0, "sum_ct": sum(ct), "max_ct": max(ct) if ct else 0,
                 "obs_sum": sum(obs), "obs_max": max(obs) if obs else 0, "msg_chars": msg_chars,
                 "analyzer": sum(v for k, v in authors.items() if k and "analyzer" in k), "n_calls": len(cmds),
                 "last_growth": (pt[-1] - pt[-2]) if len(pt) > 1 else 0})
def med(xs): return round(S.median(xs)) if xs else None
keys = ["steps", "max_pt", "sum_ct", "max_ct", "obs_sum", "obs_max", "msg_chars", "analyzer", "n_calls", "last_growth"]
for sel, name in ((lambda r: True, "all"), (lambda r: r["ovf"], "overflow"), (lambda r: not r["ovf"], "no-overflow")):
    for arm in ("base", "lora"):
        rs = [r for r in rows if r["arm"] == arm and sel(r)]
        print(f"{name:12s} {arm:4s} n={len(rs):3d}", {k: med([r[k] for r in rs]) for k in keys})
# paired per task: lora - base
both = {}
for r in rows: both.setdefault(r["id"], {})[r["arm"]] = r
d = {k: [v["lora"][k] - v["base"][k] for v in both.values() if len(v) == 2] for k in keys}
print("paired median lora-base", {k: med(v) for k, v in d.items()})
print("paired lora>base share", {k: round(sum(x > 0 for x in v) / len(v), 2) for k, v in d.items()})
ov = [r for r in rows if r["ovf"]]
print("overflow: obs_max>20k chars", {a: sum(r["obs_max"] > 20000 for r in ov if r["arm"] == a) for a in ("base", "lora")},
      "max_ct>4000", {a: sum(r["max_ct"] > 4000 for r in ov if r["arm"] == a) for a in ("base", "lora")})
json.dump(rows, open(f"{root}/../ovf_rows.json", "w"))
