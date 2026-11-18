"""Compare v17 (temp 0.2) and v27 (temp 1.0) on the 8 smoke tasks: long-step repetition severity and non-ok tool call types.
Usage: python3 -I smoke_compare.py <v17 merged> <v27 merged> <valid_tasks.json>"""
import json, sys, os, zlib, re, collections

def steps(root, tid):
    return json.load(open(os.path.join(root, "traces", f"trace_{tid}.json")))["steps"]

def long_steps(root, tid):
    out = []
    for s in steps(root, tid):
        if s.get("source") != "agent": continue
        ct = (s.get("metrics") or {}).get("completion_tokens") or 0
        if ct < 2048: continue
        msg = s.get("message") or ""; b = msg.encode()
        lines = [l.strip() for l in msg.splitlines() if len(l.strip()) > 20]
        dup = 1 - len(set(lines)) / len(lines) if lines else 0
        top = collections.Counter(lines).most_common(1)[0][1] if lines else 0
        z = len(zlib.compress(b, 9)) / max(1, len(b))
        out.append((ct, round(dup, 3), top, round(z, 3), len(lines)))
    return out

def err_types(root, tid):
    c = collections.Counter()
    for s in steps(root, tid):
        if s.get("source") != "agent": continue
        for tc in s.get("tool_calls") or []:
            try: o = json.loads((s.get("observation") or {}).get("content") or "")
            except Exception: o = {}
            if o.get("status") == "ok": continue
            fn = tc.get("function_name")
            e = str(o.get("error") or o.get("message") or o.get("stderr") or o.get("output") or "")[:200]
            e = re.sub(r"[0-9]+", "N", e)
            e = re.sub(r"'[^']*'|\"[^\"]*\"", "Q", e)
            # keep a short category only
            cat = e.split("\n")[0][:70]
            c[(fn, str(o.get("status")), o.get("exit_code"), cat)] += 1
    return c

v17, v27, ids = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
print("== long steps (completion tokens, dup share, top-line repeats, zlib ratio, lines)")
for tid in ids:
    print(tid)
    print("   v17:", long_steps(v17, tid))
    print("   v27:", long_steps(v27, tid))
print("\n== non-ok tool calls by (tool, status, exit_code, first line of error, digits->N, quotes->Q)")
for name, root in (("v17", v17), ("v27", v27)):
    tot = collections.Counter()
    per = {}
    for tid in ids:
        c = err_types(root, tid); tot.update(c); per[tid] = sum(c.values())
    print(name, "per task:", per)
    for k, n in tot.most_common(25): print(f"   {n:3d}  {k}")
