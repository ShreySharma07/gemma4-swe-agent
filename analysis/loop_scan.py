"""Thinking-loop scan for v27-smoke (Nov 18). Selection: `python3 analysis/loop_scan.py <v17 merged dir>` prints the 8 task ids (seed 0).
Metrics: `python3 analysis/loop_scan.py <merged dir> <valid_tasks.json>` prints per-task long steps (>= 2,048 completion tokens),
repetitive ones (duplicate-line share >= 0.2 or zlib ratio < 0.25), overflow, malformed edit_file calls, non-ok tool calls, time."""
import json, sys, zlib, glob, os, random, statistics as st
def task_stats(root, ids=None):
    res = {json.loads(l)["instance_id"]: json.loads(l) for l in open(os.path.join(root, "task_results.jsonl"))}
    out = {}
    for f in glob.glob(os.path.join(root, "traces", "trace_*.json")):
        tid = os.path.basename(f)[6:-5]
        if ids and tid not in ids: continue
        r = res.get(tid, {}); nlong = nrep = 0; dups = []; ed = ed_bad = calls_err = calls = 0; gen = 0; peak = 0
        for s in json.load(open(f))["steps"]:
            if s.get("source") != "agent": continue
            m = s.get("metrics") or {}; ct = m.get("completion_tokens") or 0; gen += ct; peak = max(peak, m.get("prompt_tokens") or 0)
            for tc in s.get("tool_calls") or []:
                calls += 1
                try: o = json.loads((s.get("observation") or {}).get("content") or "")
                except Exception: o = {}
                calls_err += o.get("status") != "ok"
                if tc.get("function_name") == "edit_file":
                    ed += 1; keys = set((tc.get("arguments") or {}).keys())
                    ed_bad += (o.get("status") is None) or not keys <= {"filepath", "old_string", "new_string"}
            if ct >= 2048:
                msg = s.get("message") or ""; b = msg.encode()
                lines = [l.strip() for l in msg.splitlines() if len(l.strip()) > 20]
                dup = 1 - len(set(lines)) / len(lines) if lines else 0
                nlong += 1; dups.append(dup); nrep += (dup >= 0.2 or len(zlib.compress(b, 9)) / max(1, len(b)) < 0.25)
        out[tid] = dict(res=bool(r.get("resolved")), over="ContextWindowExceeded" in str(r.get("error")), err=(str(r.get("error"))[:60] if r.get("error") else None),
                        nlong=nlong, nrep=nrep, dup=round(st.median(dups), 3) if dups else None, edit=ed, edit_bad=ed_bad,
                        calls=calls, calls_err=calls_err, gen=gen, peak=peak, dur=r.get("duration_seconds"))
    return out
if __name__ == "__main__":
    root = sys.argv[1]; ids = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    if ids is None:  # selection mode
        loop = ["fastapi_13920", "fastapi_14262", "fastapi_14266", "fastapi_14360", "fastapi_14430", "fastapi_14448", "fastapi_5624", "rich_3469", "rich_3782"]
        ctrl = ['fastapi_12942', 'fastapi_14297', 'fastapi_14301', 'fastapi_14303', 'fastapi_14349', 'fastapi_14361', 'fastapi_14458', 'fastapi_14463', 'fastapi_14492', 'fastapi_14605', 'fastapi_14786', 'fastapi_14873', 'fastapi_5077', 'fastapi_9555', 'requests_7433', 'requests_7502', 'requests_7505', 'rich_3296']
        ids = random.Random(0).sample(sorted(loop), 6) + random.Random(0).sample(sorted(ctrl), 2)
        print(json.dumps(ids)); sys.exit()
    t = task_stats(root, set(ids))
    for k in ids: print(k, t.get(k))
    L = [k for k in ids[:6]]; A = ids
    agg = lambda key, sel: sum((t[k][key] or 0) for k in sel if k in t)
    print(f"loop tasks: long steps {agg('nlong', L)}, repetitive {agg('nrep', L)}, overflow {sum(t[k]['over'] for k in L if k in t)}, resolved {sum(t[k]['res'] for k in L if k in t)}")
    print(f"all 8: edit_file {agg('edit', A)} (malformed {agg('edit_bad', A)}), tool calls {agg('calls', A)} (non-ok {agg('calls_err', A)}), resolved {sum(t[k]['res'] for k in A if k in t)}, "
          f"mean dur {st.mean(t[k]['dur'] or 0 for k in A if k in t):.0f}s, generated {agg('gen', A)}, peak-prompt median {st.median(t[k]['peak'] for k in A if k in t):.0f}")
