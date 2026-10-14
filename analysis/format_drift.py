"""
Step 1 (Measure): tool-call format drift in Gemma-4 agent trajectories (ATIF traces, e.g. traces2/ = v5_graph run).

Each tool call in an ATIF trace has already been through vLLM's gemma4 parser. When Gemma drifts from its native
format (`key:<|"|>value<|"|>`) to JSON-ish quoting, the parser turns code fragments or quoted names into argument
keys. So drift is visible in the parsed keys/values. Classes (one call can have several):
  code_key      a key that is not an identifier (code fragment, quote, space, newline)  -> string delimiters lost
  quoted_key    a key that is a quoted identifier, e.g. `"start_line"` / `start_line"`  -> JSON-style keys
  extra_key     an identifier key the tool does not accept (e.g. run_command{run_command:..})
  missing_req   a required argument is absent
  quoted_value  a string value wrapped in literal quotes, e.g. path:"a.py"
A call is CLEAN if it has none of these. Also reported: whether the harness answered with an error, drift vs
context length (prompt tokens) and per agent role, and resolve rate for tasks with / without drift.
Usage: python3 analysis/format_drift.py traces2 [results/runs/r3_v5_graph/behaviour_per_task.json]
"""
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = {  # tool -> (required, optional), from swegemma 0.2.7 tool signatures
    "run_command": ({"command"}, set()),
    "read_file": ({"filepath"}, {"start_line", "end_line"}),
    "write_file": ({"filepath", "content"}, set()),
    "edit_file": ({"filepath", "old_string", "new_string"}, {"allow_multiple"}),
    "submit_patch": (set(), set()),
    "get_status": (set(), set()),
    "get_code_neighbors": ({"node"}, {"edge_type", "max_neighbors"}),
    "search_similar_code": ({"query"}, {"k"}),
    "get_code_subgraph": ({"nodes"}, set()),
    "load_skill": ({"skill_name"}, set()),
    "list_skills": (set(), set()),
    "run_skill_script": ({"skill_name", "file_path"}, {"args"}),
}
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def classify(name, args):
    req, opt = SCHEMA.get(name, (None, None))
    if req is None:
        return {"unknown_tool"}
    cls = set()
    if not isinstance(args, dict):
        return {"non_dict_args"}
    for k, v in args.items():
        bare = k.strip("\"' ")
        if not IDENT.match(k):
            cls.add("quoted_key" if IDENT.match(bare) else "code_key")
        elif k not in req | opt:
            cls.add("extra_key")
        if isinstance(v, str) and len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'" and "\n" not in v:
            cls.add("quoted_value")
    keys = {k.strip("\"' ") for k in args}
    if req - set(args):
        cls.add("missing_req")
    return cls


def main():
    tdir = Path(sys.argv[1] if len(sys.argv) > 1 else "traces2")
    beh = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/runs/r3_v5_graph/behaviour_per_task.json")
    resolved = {x["id"]: x.get("resolved") for x in json.loads(beh.read_text())} if beh.exists() else {}
    per_tool = defaultdict(Counter)
    by_ctx = defaultdict(Counter)
    by_role = defaultdict(Counter)
    task_drift = {}
    err_after = Counter()
    examples = defaultdict(list)
    n_calls = 0
    for f in sorted(tdir.glob("*.json")):
        iid = f.stem.replace("trace_", "")
        d = json.loads(f.read_text())
        steps = d.get("steps", [])
        drift_calls = 0
        for i, s in enumerate(steps):
            for tc in s.get("tool_calls") or []:
                name, args = tc.get("function_name"), tc.get("arguments")
                cls = classify(name, args)
                n_calls += 1
                clean = not cls
                per_tool[name]["calls"] += 1
                per_tool[name]["clean"] += clean
                for c in cls:
                    per_tool[name][c] += 1
                    if len(examples[(name, c)]) < 3:
                        examples[(name, c)].append({"task": iid, "keys": list(args)[:4] if isinstance(args, dict) else args})
                ptok = ((tc.get("extra") or {}).get("usage") or {}).get("prompt_tokens") or 0
                b = "<8k" if ptok < 8000 else "8-16k" if ptok < 16000 else "16-24k" if ptok < 24000 else ">=24k"
                by_ctx[b]["calls"] += 1
                by_ctx[b]["drift"] += not clean
                role = (tc.get("extra") or {}).get("author", "?")
                by_role[role]["calls"] += 1
                by_role[role]["drift"] += not clean
                drift_calls += not clean
                # harness reply: the next observation carrying this tool's result
                for s2 in steps[i:i + 3]:
                    obs = s2.get("observation")
                    if obs and (obs.get("extra") or {}).get("tool_name") == name:
                        body = obs.get("content") or ""
                        is_err = '"status": "error"' in body or "error" in body[:80].lower()
                        err_after[("drift" if not clean else "clean", is_err)] += 1
                        break
        task_drift[iid] = drift_calls

    def pct(a, b):
        return round(100 * a / b, 1) if b else None

    out = {"n_traces": len(task_drift), "n_calls": n_calls,
           "drift_calls_pct": pct(sum(n_calls - 0 for _ in [0]) and sum(v["calls"] - v["clean"] for v in per_tool.values()), n_calls),
           "per_tool": {t: {"calls": c["calls"], "clean_pct": pct(c["clean"], c["calls"]),
                            **{k: c[k] for k in ("code_key", "quoted_key", "extra_key", "missing_req", "quoted_value",
                                                 "unknown_tool", "non_dict_args") if c[k]}}
                        for t, c in sorted(per_tool.items(), key=lambda kv: -kv[1]["calls"])},
           "drift_by_context": {b: {"calls": c["calls"], "drift_pct": pct(c["drift"], c["calls"])}
                                for b, c in sorted(by_ctx.items())},
           "drift_by_role": {r: {"calls": c["calls"], "drift_pct": pct(c["drift"], c["calls"])} for r, c in by_role.items()},
           "harness_error_rate": {k: pct(err_after[(k, True)], err_after[(k, True)] + err_after[(k, False)])
                                  for k in ("clean", "drift")},
           "tasks_with_drift": sum(v > 0 for v in task_drift.values())}
    if resolved:
        w = [resolved.get(i) for i, v in task_drift.items() if v > 0 and i in resolved]
        wo = [resolved.get(i) for i, v in task_drift.items() if v == 0 and i in resolved]
        out["resolve_rate_with_drift"] = f"{sum(map(bool, w))}/{len(w)}"
        out["resolve_rate_without_drift"] = f"{sum(map(bool, wo))}/{len(wo)}"
    out["examples"] = {f"{t}:{c}": v for (t, c), v in examples.items()}
    Path("results/format_drift_" + tdir.name + ".json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    main()
