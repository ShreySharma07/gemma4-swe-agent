"""
Run in the same Kaggle notebook after run_official_l4.py finishes:   !python pack_for_claude.py
Makes /kaggle/working/for_claude.zip (usually a few MB). Upload that one file to Claude.

Contents:
  run.log, task_results.jsonl, task_ids_order.json, analysis/ (if the analyzer ran), patches/,
  traces_digest.jsonl.gz  -> every trace compressed to its essentials (one line per tool call: who, tool,
                             args, status/error, output head, tokens), so all tasks fit,
  full_traces/            -> complete ATIF traces for up to 8 representative tasks
                             (resolved, wrong fix, empty patch, longest, most errors).
"""

import gzip
import json
import pathlib
import zipfile

W = pathlib.Path("/kaggle/working")
M = W / "merged"
OUT = W / "for_claude.zip"


def short(x, n):
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return s if len(s) <= n else s[:n] + f"...[+{len(s) - n}]"


results = {}
if (M / "task_results.jsonl").exists():
    for line in open(M / "task_results.jsonl"):
        if line.strip():
            r = json.loads(line)
            results[r["instance_id"]] = r

digest, stats = [], []
for tp in sorted((M / "traces").glob("trace_*.json")):
    iid = tp.stem[len("trace_"):].replace("__", "/")
    t = json.loads(tp.read_text())
    rows, n_err = [], 0
    for s in t.get("steps", []):
        ex = s.get("extra") or {}
        et = ex.get("event_type")
        m = s.get("metrics") or {}
        if s.get("tool_calls"):
            obs = (s.get("observation") or {}).get("content", "")
            try:
                o = json.loads(obs) if isinstance(obs, str) else obs
                o = json.loads(o["result"]) if isinstance(o, dict) and isinstance(o.get("result"), str) else o
            except Exception:  # noqa: BLE001
                o = obs
            status = o.get("status") if isinstance(o, dict) else None
            n_err += status == "error"
            for tc in s["tool_calls"]:
                rows.append({"t": round(float(ex.get("elapsed_s") or 0), 1), "who": (tc.get("extra") or {}).get("author"),
                             "tool": tc.get("function_name"), "args": short(tc.get("arguments"), 400),
                             "status": status, "err": o.get("error_type") if isinstance(o, dict) else None,
                             "out": short(o, 600), "ptok": m.get("prompt_tokens"), "ctok": m.get("completion_tokens")})
        elif et:
            rows.append({"t": round(float(ex.get("elapsed_s") or 0), 1), "event": et,
                         "text": short(s.get("message") or "", 1500 if et in ("final", "text", "thinking") else 300),
                         "ptok": m.get("prompt_tokens"), "ctok": m.get("completion_tokens")})
    r = results.get(iid, {})
    digest.append({"instance_id": iid, "result": r, "final_metrics": t.get("final_metrics"), "steps": rows})
    stats.append((iid, bool(r.get("resolved")), (r.get("agent_patch_size") or 0) > 0, len(rows), n_err))

pick = []
def add(cands):
    for c in cands:
        if c not in pick:
            pick.append(c)
            return

add([s[0] for s in stats if s[1]])                                  # a resolved task
add([s[0] for s in stats if not s[1] and s[2]])                     # wrong fix (patch but failed)
add([s[0] for s in stats if not s[2]])                              # empty patch
add([s[0] for s in sorted(stats, key=lambda s: -s[3])])             # longest
add([s[0] for s in sorted(stats, key=lambda s: -s[4])])             # most tool errors
for group in ([s[0] for s in stats if s[1]], [s[0] for s in stats if not s[1] and s[2]], [s[0] for s in stats if not s[2]]):
    if len(pick) < 8:
        add(group[1:])

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("run.log", "task_ids_order.json"):
        if (W / name).exists():
            z.write(W / name, name)
    if (M / "task_results.jsonl").exists():
        z.write(M / "task_results.jsonl", "task_results.jsonl")
    for p in (W / "analysis").glob("*") if (W / "analysis").exists() else []:
        z.write(p, f"analysis/{p.name}")
    for p in (M / "patches").glob("*.patch"):
        z.write(p, f"patches/{p.name}")
    z.writestr("traces_digest.jsonl.gz", gzip.compress("\n".join(json.dumps(d, default=str) for d in digest).encode()))
    for iid in pick:
        tp = M / "traces" / f"trace_{iid.replace('/', '__')}.json"
        if tp.exists():
            z.write(tp, f"full_traces/{tp.name}")
print(f"{len(digest)} traces digested, full traces for {pick}")
print(f"-> {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)  upload this file to Claude")
