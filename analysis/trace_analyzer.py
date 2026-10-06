"""
Map 2 — trace analyzer for swegemma results (official harness, ATIF v1.7 traces).

Usage:
    python trace_analyzer.py --results baseline_tpu --tasks tasks.jsonl [--out analysis]

Inputs (from `swegemma eval` / Evaluator results dir):
    traces/trace_<id>.json   ATIF v1.7 trajectory (steps: source/message/tool_calls/observation/extra)
    task_results.jsonl       resolved / error / tool_calls / duration per task
    patches/<id>.patch       the agent's final diff
Gold (from tasks.jsonl `patch`): gold files, changed line ranges, enclosing function names.

For every task it labels WHERE the trajectory failed, WHAT broke along the way, and whether it RECOVERED:
  stages  understand -> localize (saw gold file / saw gold lines) -> reproduce -> edit (gold file?) -> verify -> submit
  breaks  tool errors by type, edit failures, command timeouts, nudges, compaction, loops, hallucinated paths,
          natural-language graph queries, context pressure (max prompt tokens)
  recovery  after each tool error: did the agent succeed with the same tool within 3 steps? (+ cost in steps)
Outputs: <out>/per_task.jsonl, <out>/summary.json, <out>/report.md
"""

import argparse
import collections
import json
import pathlib
import re
import statistics

TOOLS_READ = {"read_file"}
TOOLS_EDIT = {"edit_file", "write_file"}
GRAPH_TOOLS = {"get_code_neighbors", "search_similar_code", "get_code_subgraph"}
SED_RANGE = re.compile(r"sed\s+-n\s+['\"]?(\d+)\s*,\s*(\d+)p")
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")
FUNC_IN_HEADER = re.compile(r"(?:def|class)\s+([A-Za-z_]\w*)")
CONTEXT_LIMIT = 32768
SCHEMA = {"read_file": {"filepath", "start_line", "end_line"}, "edit_file": {"filepath", "old_string", "new_string", "allow_multiple"},
          "write_file": {"filepath", "content"}, "run_command": {"command"}, "get_code_neighbors": {"node", "edge_type", "max_neighbors"},
          "search_similar_code": {"query", "k"}, "get_code_subgraph": {"nodes"}, "submit_patch": set(), "get_status": set()}


# ------------------------------------------------------------------ gold
def parse_gold(patch: str):
    files, cur = {}, None
    for line in (patch or "").splitlines():
        if line.startswith("+++ b/"):
            cur = line[6:].strip()
            files[cur] = {"ranges": [], "funcs": set(), "new_file": False}
        elif line.startswith("--- /dev/null") and cur is None:
            pass
        elif cur and (m := HUNK.match(line)):
            start, length = int(m.group(1)), int(m.group(2) or 1)
            files[cur]["ranges"].append((start, start + max(length, 1) - 1))
            if f := FUNC_IN_HEADER.search(m.group(5) or ""):
                files[cur]["funcs"].add(f.group(1))
        elif cur and line.startswith("+") and not line.startswith("+++"):
            if f := re.match(r"\+\s*(?:async\s+)?def\s+([A-Za-z_]\w*)", line):
                files[cur]["funcs"].add(f.group(1))
    for f in files.values():
        f["funcs"] = sorted(f["funcs"])
    return files


def patch_files(patch_text: str):
    return re.findall(r"^\+\+\+ b/(\S+)", patch_text or "", flags=re.M)


# ------------------------------------------------------------------ trace helpers
def deep_json(x, depth=3):
    """Tool results are JSON strings, sometimes wrapped (e.g. {'result': '<json>'}); unwrap to a dict if possible."""
    for _ in range(depth):
        if isinstance(x, str):
            try:
                x = json.loads(x)
                continue
            except Exception:  # noqa: BLE001
                return x
        if isinstance(x, dict) and len(x) == 1 and isinstance(next(iter(x.values())), str):
            inner = next(iter(x.values()))
            try:
                x = json.loads(inner)
                continue
            except Exception:  # noqa: BLE001
                return x
        return x
    return x


def obs_text(step):
    o = step.get("observation") or {}
    c = o.get("content", "")
    return c if isinstance(c, str) else json.dumps(c)


def flatten_steps(traj):
    steps = list(traj.get("steps", []))
    for sub in traj.get("subagent_trajectories", []) or []:
        steps += flatten_steps(sub)
    return steps


def norm_path(p):
    p = (p or "").strip()
    for pre in ("/workspace/", "workspace/", "./"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p.lstrip("/")


def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


# ------------------------------------------------------------------ per-task analysis
def analyze(trace, task, result, agent_patch):
    gold = parse_gold(task.get("patch", ""))
    gold_files = set(gold)
    steps = flatten_steps(trace)
    ev = []   # flattened events: one per tool call, with its observation
    nudges, compactions, texts = 0, 0, 0
    max_prompt = 0
    ptoks = []
    for s in steps:
        et = (s.get("extra") or {}).get("event_type")
        m = s.get("metrics") or {}
        max_prompt = max(max_prompt, int(m.get("prompt_tokens", 0) or 0))
        if m.get("prompt_tokens"):
            ptoks.append(int(m["prompt_tokens"]))
        if et == "continuation_nudge":
            nudges += 1
        elif et == "compaction":
            compactions += 1
        elif et in ("text", "final"):
            texts += 1
        calls = s.get("tool_calls") or []
        res = deep_json(obs_text(s)) if s.get("observation") else None
        for i, tc in enumerate(calls):
            ev.append({"name": tc.get("function_name"), "args": tc.get("arguments") or {},
                       "author": (tc.get("extra") or {}).get("author", ""),
                       "obs": res if i == len(calls) - 1 else None,
                       "obs_raw": obs_text(s) if i == len(calls) - 1 else ""})

    f = {"instance_id": task["instance_id"], "repo": task["repo"], "resolved": bool(result.get("resolved")),
         "harness_error": result.get("error"), "duration_s": result.get("duration_seconds"),
         "n_tool_calls": len(ev), "gold_files": sorted(gold_files),
         "gold_funcs": sorted({fn for g in gold.values() for fn in g["funcs"]}),
         "nudges": nudges, "compactions": compactions, "max_prompt_tokens": max_prompt,
         "context_pressure": round(max_prompt / CONTEXT_LIMIT, 2)}
    f["tool_counts"] = dict(collections.Counter(e["name"] for e in ev))
    f["by_author"] = dict(collections.Counter(e["author"] or "?" for e in ev))

    first_gold_file = first_gold_lines = first_gold_func = first_edit = first_gold_edit = None
    first_test = last_edit = None
    errors, recoveries, err_types = [], [], collections.Counter()
    edit_fail = cmd_timeout = hallucinated = nl_graph = graph_nonempty = graph_empty = skill_runs = skill_err = 0
    repro_before_edit = False
    tests_after_last_edit, last_test_ok = 0, None
    seen_calls = collections.Counter()
    for k, e in enumerate(ev):
        name, a, o, raw = e["name"], e["args"], e["obs"], e["obs_raw"] or ""
        key = json.dumps([name, a], sort_keys=True)
        seen_calls[key] += 1
        status = o.get("status") if isinstance(o, dict) else None
        etype = o.get("error_type") if isinstance(o, dict) else None
        if status == "error":
            err_types[etype or "?"] += 1
            errors.append(k)
            if etype == "FileEditError":
                edit_fail += 1
            if etype == "TimeoutExceeded":
                cmd_timeout += 1
            msg = (o.get("error_message") or "").lower()
            if "not found" in msg or "no such file" in msg or "does not exist" in msg:
                hallucinated += 1
        path = norm_path(a.get("filepath", ""))
        cmd = a.get("command", "") if name == "run_command" else ""
        text_blob = raw + " " + cmd + " " + json.dumps(a)
        # localization: gold file seen?
        if first_gold_file is None and any(g in text_blob for g in gold_files):
            first_gold_file = k
        # gold lines seen (read_file ranges or sed -n ranges)
        if first_gold_lines is None and status != "error":
            rng = None
            if name in TOOLS_READ and path in gold_files:
                s0 = a.get("start_line") or 1
                e0 = a.get("end_line") or (s0 + 149)
                rng = (int(s0), int(e0))
                gr = gold[path]["ranges"]
            elif cmd:
                for g in gold_files:
                    if g in cmd and (m := SED_RANGE.search(cmd)):
                        rng, gr = (int(m.group(1)), int(m.group(2))), gold[g]["ranges"]
                        break
                    if g in cmd and re.search(r"\bcat\s", cmd):
                        rng, gr = (1, 10 ** 6), gold[g]["ranges"]
                        break
            if rng and any(overlaps(rng, r) for r in gr):
                first_gold_lines = k
        if first_gold_func is None and f["gold_funcs"] and any(fn in raw for fn in f["gold_funcs"]):
            first_gold_func = k
        # repro / tests
        is_test = bool(re.search(r"\bpytest\b|unittest", cmd))
        is_repro = bool(re.search(r"python3?\s+(/tmp/|-c\s)", cmd)) or (name == "write_file" and path.startswith("tmp/"))
        if first_edit is None and is_repro:
            repro_before_edit = True
        if is_test and first_test is None:
            first_test = k
        # edits
        if name in TOOLS_EDIT and not path.startswith("tmp/"):
            if first_edit is None:
                first_edit = k
            last_edit = k
            tests_after_last_edit, last_test_ok = 0, None
            if path in gold_files and first_gold_edit is None and status != "error":
                first_gold_edit = k
        elif is_test and last_edit is not None:
            tests_after_last_edit += 1
            last_test_ok = status == "ok"
        if name == "search_similar_code" and len(str(a.get("query", "")).split()) > 2:
            nl_graph += 1
        if name in GRAPH_TOOLS:
            o = e["obs"] if isinstance(e["obs"], dict) else {}
            if (o.get("count") or 0) > 0 or (name == "get_code_subgraph" and o.get("status") == "ok"):
                graph_nonempty += 1
            else:
                graph_empty += 1
        if name == "run_skill_script":
            skill_runs += 1
            if isinstance(e["obs"], dict) and (e["obs"].get("error") or e["obs"].get("stderr")):
                skill_err += 1
    # recovery after errors: same tool succeeds within the next 3 calls
    for k in errors:
        tool = ev[k]["name"]
        rec = next((j - k for j in range(k + 1, min(k + 4, len(ev)))
                    if ev[j]["name"] == tool and isinstance(ev[j]["obs"], dict) and ev[j]["obs"].get("status") == "ok"),
                   None)
        recoveries.append(rec)
    loops = sum(1 for c in seen_calls.values() if c >= 3)
    malformed = sum(1 for e in ev if e["name"] in SCHEMA and set(e["args"]) - SCHEMA[e["name"]])
    malformed_reads = sum(1 for e in ev if e["name"] == "read_file" and (set(e["args"]) - SCHEMA["read_file"]
                          or ("end_line" in e["args"] and "start_line" not in e["args"])))  # range silently lost
    resets = sum(1 for a0, b0 in zip(ptoks, ptoks[1:]) if a0 > 8000 and b0 < 0.6 * a0)
    grep_nomatch = sum(1 for e in ev if e["name"] == "run_command" and isinstance(e["obs"], dict)
                       and e["obs"].get("status") == "error" and "grep" in str(e["args"].get("command", ""))
                       and (e["obs"].get("details") or {}).get("exit_code") == 1)
    repeated_calls = sum(c - 1 for c in seen_calls.values() if c > 1)
    submitted = any(e["name"] == "submit_patch" for e in ev)
    pfiles = patch_files(agent_patch)
    stray = [p for p in pfiles if not p.endswith(".py") or re.search(r"(^|/)(repro|reproduce|debug|test_issue)", p)]
    f.update({
        "step_first_gold_file": first_gold_file, "step_first_gold_lines": first_gold_lines,
        "step_first_gold_func": first_gold_func, "step_first_edit": first_edit, "step_first_gold_edit": first_gold_edit,
        "step_first_test": first_test, "repro_before_edit": repro_before_edit,
        "tests_after_last_edit": tests_after_last_edit, "last_test_ok": last_test_ok,
        "errors_total": len(errors), "error_types": dict(err_types), "edit_failures": edit_fail,
        "cmd_timeouts": cmd_timeout, "hallucinated_paths": hallucinated, "nl_graph_queries": nl_graph, "graph_nonempty": graph_nonempty, "graph_empty": graph_empty, "skill_runs": skill_runs, "skill_errors": skill_err,
        "loops_3x": loops, "malformed_tool_args": malformed, "malformed_read_file": malformed_reads,
        "n_read_file": f["tool_counts"].get("read_file", 0), "context_resets": resets,
        "grep_no_match_errors": grep_nomatch, "repeated_identical_calls": repeated_calls, "recovered_errors": sum(r is not None for r in recoveries),
        "recovery_steps_mean": round(statistics.mean([r for r in recoveries if r]), 2) if any(recoveries) else None,
        "graph_calls": sum(f["tool_counts"].get(t, 0) for t in GRAPH_TOOLS),
        "analyzer_calls": sum(v for k2, v in f["tool_counts"].items() if "analyzer" in (k2 or "")),
        "submitted_explicitly": submitted, "patch_files": pfiles, "patch_empty": not agent_patch.strip(),
        "patch_touches_gold": bool(set(pfiles) & gold_files), "stray_files_in_patch": stray,
        "overflow": "context" in str(result.get("error") or "").lower() or max_prompt > 0.95 * CONTEXT_LIMIT,
        "budget_exhausted": "budget" in str(result.get("error") or "").lower(),
    })
    f["failure_stage"] = label(f)
    return f


def label(f):
    """First stage at which the trajectory went wrong (ordered like the pipeline)."""
    if f["resolved"]:
        return "resolved"
    if f["patch_empty"]:
        if f["step_first_gold_file"] is None:
            return "localize:never_found_gold_file"
        if f["step_first_gold_lines"] is None:
            return "localize:found_file_not_lines"
        if f["step_first_edit"] is None:
            return "commit:saw_gold_never_edited"
        return "edit:edits_failed_or_reverted"
    if not f["patch_touches_gold"]:
        return "localize:edited_wrong_file" if f["step_first_gold_file"] is None else "localize:saw_gold_edited_elsewhere"
    if f["tests_after_last_edit"] == 0:
        return "fix:wrong_fix_unverified"
    return "fix:wrong_fix_verified_locally"


# ------------------------------------------------------------------ aggregate
def pct(n, d):
    return round(100 * n / d, 1) if d else None


def summarize(rows):
    n = len(rows)
    by_stage = collections.Counter(r["failure_stage"] for r in rows)
    eh = [r for r in rows if r["step_first_gold_lines"] is not None]
    no_eh = [r for r in rows if r["step_first_gold_lines"] is None]
    err_types = collections.Counter()
    for r in rows:
        err_types.update(r["error_types"])
    tot_err = sum(r["errors_total"] for r in rows)
    s = {
        "n_tasks": n, "resolved": sum(r["resolved"] for r in rows),
        "failure_stage": dict(by_stage.most_common()),
        "evidence_hit_file_pct": pct(sum(r["step_first_gold_file"] is not None for r in rows), n),
        "evidence_hit_lines_pct": pct(len(eh), n),
        "P(resolved|saw gold lines)": pct(sum(r["resolved"] for r in eh), len(eh)),
        "P(resolved|never saw gold lines)": pct(sum(r["resolved"] for r in no_eh), len(no_eh)),
        "P(edited gold|saw gold lines)": pct(sum(r["patch_touches_gold"] for r in eh), len(eh)),
        "median_step_first_gold_lines": statistics.median([r["step_first_gold_lines"] for r in eh]) if eh else None,
        "tool_calls_median": statistics.median([r["n_tool_calls"] for r in rows]) if rows else None,
        "tool_mix": dict(sum((collections.Counter(r["tool_counts"]) for r in rows), collections.Counter()).most_common()),
        "error_types": dict(err_types.most_common()), "errors_total": tot_err,
        "error_recovery_pct": pct(sum(r["recovered_errors"] for r in rows), tot_err),
        "tasks_with_loops_pct": pct(sum(r["loops_3x"] > 0 for r in rows), n),
        "tasks_with_malformed_read_file_pct": pct(sum(r["malformed_read_file"] > 0 for r in rows), n),
        "read_file_calls_malformed_pct": pct(sum(r["malformed_read_file"] for r in rows), sum(r["n_read_file"] for r in rows)),
        "P(resolved|malformed reads)": pct(sum(r["resolved"] for r in rows if r["malformed_read_file"] > 0),
                                           sum(r["malformed_read_file"] > 0 for r in rows)),
        "P(resolved|no malformed reads)": pct(sum(r["resolved"] for r in rows if r["malformed_read_file"] == 0),
                                              sum(r["malformed_read_file"] == 0 for r in rows)),
        "tasks_with_context_resets_pct": pct(sum(r["context_resets"] > 0 for r in rows), n),
        "repeated_identical_calls_share_pct": pct(sum(r["repeated_identical_calls"] for r in rows), sum(r["n_tool_calls"] for r in rows)),
        "grep_no_match_errors (benign)": sum(r["grep_no_match_errors"] for r in rows),
        "tasks_stray_files_in_patch_pct": pct(sum(bool(r["stray_files_in_patch"]) for r in rows), n),
        "tasks_with_nudges_pct": pct(sum(r["nudges"] > 0 for r in rows), n),
        "tasks_overflow_pct": pct(sum(r["overflow"] for r in rows), n),
        "tasks_budget_exhausted_pct": pct(sum(r["budget_exhausted"] for r in rows), n),
        "explicit_submit_pct": pct(sum(r["submitted_explicitly"] for r in rows), n),
        "empty_patch_pct": pct(sum(r["patch_empty"] for r in rows), n),
        "repro_before_edit_pct": pct(sum(r["repro_before_edit"] for r in rows), n),
        "tested_after_last_edit_pct": pct(sum(r["tests_after_last_edit"] > 0 for r in rows), n),
        "graph_tool_use_pct": pct(sum(r["graph_calls"] > 0 for r in rows), n),
        "nl_graph_queries_total": sum(r["nl_graph_queries"] for r in rows),
        "graph_results_nonempty_total": sum(r["graph_nonempty"] for r in rows),
        "graph_results_empty_total": sum(r["graph_empty"] for r in rows),
        "skill_runs_total": sum(r["skill_runs"] for r in rows),
        "skill_errors_total": sum(r["skill_errors"] for r in rows),
        "skill_use_pct": pct(sum(r["skill_runs"] > 0 for r in rows), n),
        "context_pressure_median": statistics.median([r["context_pressure"] for r in rows]) if rows else None,
        "by_repo": {repo: {"n": len(v), "resolved": sum(x["resolved"] for x in v)}
                    for repo, v in collections.defaultdict(list, {}).items()},
    }
    repos = collections.defaultdict(list)
    for r in rows:
        repos[r["repo"]].append(r)
    s["by_repo"] = {k: {"n": len(v), "resolved": sum(x["resolved"] for x in v)} for k, v in repos.items()}
    return s


def report_md(s):
    lines = ["# Map 2 — failure analysis", "", f"Tasks: {s['n_tasks']} · resolved: {s['resolved']}"
             + (f" · env-unverifiable (missing test deps): {s['env_unverifiable_tasks']} · resolved on verifiable: "
                f"{s['verifiable_only']['resolved']}/{s['verifiable_only']['n_tasks']}" if s.get("env_unverifiable_tasks") else ""), "",
             "## Where it fails (first failing stage)", "| stage | tasks |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s["failure_stage"].items()]
    lines += ["", "## Key rates", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s.items() if not isinstance(v, dict) and k not in ("n_tasks", "resolved")]
    lines += ["", "## Tool errors", "| type | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s["error_types"].items()]
    lines += ["", "## Tool mix", "| tool | calls |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s["tool_mix"].items()]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args()
    res_dir, out = pathlib.Path(a.results), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = {t["instance_id"]: t for t in map(json.loads, open(a.tasks)) if t}
    results = {}
    if (res_dir / "task_results.jsonl").exists():
        for line in open(res_dir / "task_results.jsonl"):
            if line.strip():
                r = json.loads(line)
                results[r["instance_id"]] = r
    rows = []
    for tp in sorted((res_dir / "traces").glob("trace_*.json")):
        iid = tp.stem[len("trace_"):].replace("__", "/")
        if iid not in tasks:
            continue
        pp = res_dir / "patches" / f"{iid}.patch"
        row = analyze(json.loads(tp.read_text()), tasks[iid], results.get(iid, {}),
                      pp.read_text() if pp.exists() else "")
        lg = res_dir / "test_outputs" / f"{iid}.log"
        row["env_unverifiable"] = bool(lg.exists() and re.search(
            r"No module named '(inline_snapshot|dirty_equals)'", lg.read_text(errors="replace")))
        rows.append(row)
    with open(out / "per_task.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    s = summarize(rows)
    clean = [r for r in rows if not r["env_unverifiable"]]
    if len(clean) < len(rows):
        s["env_unverifiable_tasks"] = len(rows) - len(clean)
        s["verifiable_only"] = summarize(clean)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    (out / "report.md").write_text(report_md(s))
    print(report_md(s))


if __name__ == "__main__":
    main()
