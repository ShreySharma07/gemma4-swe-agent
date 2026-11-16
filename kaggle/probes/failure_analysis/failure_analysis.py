"""FA1: where does the base agent (v14_thinking, 28 calls) fail on unseen repos?

CPU only. Reads the base arm of the two G2 gate runs (kernel outputs attached as inputs):
  held-out   shreysharma07/gemma4-lora-gate-t2a      95 tasks (toolz, pathspec; synthetic bugs)
  real-bug   atulbaliyan/gemma4-lora-gate-t2b-f3     101 tasks (11 unseen libraries; real fixes)
Each eval_gate_base_b*.tgz already holds analysis/per_task.jsonl (trace_analyzer labels),
merged/task_results.jsonl and merged/traces. This script only aggregates them and adds how each
run ended (call cap, time cap, voluntary submit) and what happened after the first submit_patch.

The gate ran 6 tasks at once with a 12 min cap, so durations are NOT scorer durations
(scorer: one task at a time, 8 min). Call counts are comparable; times are reported as shares only.
Prints summaries only; writes failure_analysis.json and fa_per_task.jsonl.
"""
import collections
import json
import pathlib
import statistics
import tarfile

CAP = 28
INPUT = pathlib.Path("/kaggle/input")
WORK = pathlib.Path("/kaggle/working")
SETS = {"gemma4-lora-gate-t2a": "held-out", "gemma4-lora-gate-t2b-f3": "real-bug"}


def set_of(path):
    s = str(path)
    # the f3 slug contains neither the t2a slug nor vice versa
    for slug, name in SETS.items():
        if slug in s:
            return name
    return None


def first_submit(trace):
    """(tool calls before the first submit_patch, elapsed at it, elapsed at end, llm steps after it)."""
    n_calls, at, t_submit, after = 0, None, None, 0
    last_t = 0.0
    for s in trace.get("steps") or []:
        ex = s.get("extra") or {}
        if ex.get("elapsed_s") is not None:
            last_t = float(ex["elapsed_s"])
        if at is not None and s.get("source") == "agent":
            after += 1
        for tc in s.get("tool_calls") or []:
            if tc.get("function_name") == "submit_patch" and at is None:
                at, t_submit = n_calls, last_t
            n_calls += 1
    return at, t_submit, last_t, after


def ending(row, res):
    err = str(res.get("error") or "").lower()
    if "timeout" in err or "time" in err:
        return "time cap"
    if row.get("overflow") and "context" in err:
        return "context overflow"
    if err:
        return "other error"
    # the harness counter (paid calls) decides; the trace count includes free calls
    used = res.get("tool_calls")
    capped = used >= CAP if used is not None else row["n_tool_calls"] >= CAP
    if capped and not row["submitted_explicitly"]:
        return "call cap, no submit"
    if capped:
        return "call cap, submitted"
    if row["submitted_explicitly"]:
        return "submitted early"
    return "stopped, no submit"


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def table(title, counter, total):
    print(f"\n{title} (n={total})")
    for k, v in counter.most_common():
        print(f"  {v:4d}  {100 * v / total:5.1f}%  {k}")


def summarise(name, rows):
    n = len(rows)
    ok = [r for r in rows if r["resolved"]]
    bad = [r for r in rows if not r["resolved"]]
    print("\n" + "#" * 70 + f"\n{name}: {len(ok)}/{n} resolved ({100 * len(ok) / n:.1f}%)")
    out = {"n": n, "resolved": len(ok)}

    c = collections.Counter(r["failure_stage"] for r in bad)
    table("Failure stage (first thing that went wrong), failures only", c, len(bad))
    out["failure_stage"] = dict(c)

    for label, grp in (("resolved", ok), ("failed", bad)):
        c = collections.Counter(r["ending"] for r in grp)
        table(f"How the run ended, {label}", c, len(grp))
        out[f"ending_{label}"] = dict(c)

    # the question for the call budget: capped failures that had already found the gold lines
    capped = [r for r in bad if r["ending"].startswith("call cap")]
    out["failed_capped"] = len(capped)
    out["failed_capped_found_lines"] = sum(r["step_first_gold_lines"] is not None for r in capped)
    out["failed_capped_touched_gold"] = sum(r["patch_touches_gold"] for r in capped)
    out["failed_capped_empty_patch"] = sum(r["patch_empty"] for r in capped)
    early = [r for r in bad if r["ending"] == "submitted early"]
    out["failed_early_submit"] = len(early)
    out["failed_early_touched_gold"] = sum(r["patch_touches_gold"] for r in early)
    out["failed_early_tested"] = sum((r.get("tests_after_last_edit") or 0) > 0 for r in early)
    print("\nBudget questions (failures)")
    print(f"  hit the call cap: {len(capped)}; of these saw the gold lines: {out['failed_capped_found_lines']}, "
          f"patch touches gold file: {out['failed_capped_touched_gold']}, empty patch: {out['failed_capped_empty_patch']}")
    print(f"  submitted before the cap: {len(early)}; patch touches gold file: {out['failed_early_touched_gold']}, "
          f"ran a test after the last edit: {out['failed_early_tested']}")

    print("\nCalls used (p25 / median / p75 / max)")
    for label, grp in (("resolved", ok), ("failed", bad)):
        xs = [r["n_tool_calls"] for r in grp]
        print(f"  {label:9s} {q(xs, .25)} / {q(xs, .5)} / {q(xs, .75)} / {max(xs) if xs else None}")
        out[f"calls_{label}"] = [q(xs, .25), q(xs, .5), q(xs, .75), max(xs) if xs else None]
    for key, text in (("step_first_gold_lines", "first view of gold lines"), ("step_first_edit", "first edit"),
                      ("calls_before_submit", "first submit_patch")):
        for label, grp in (("resolved", ok), ("failed", bad)):
            xs = [r.get(key) for r in grp if r.get(key) is not None]
            print(f"  call index of {text:26s} {label:9s} n={len(xs):3d}  median {q(xs, .5)}  p75 {q(xs, .75)}")
            out[f"{key}_{label}"] = {"n": len(xs), "median": q(xs, .5), "p75": q(xs, .75)}

    # fast exit: what the agent spends after its first submit
    sub = [r for r in rows if r.get("calls_before_submit") is not None and r.get("t_end")]
    shares = [(r["t_end"] - r["t_submit"]) / r["t_end"] for r in sub if r["t_end"] > 0]
    extra_calls = [r["n_tool_calls"] - r["calls_before_submit"] - 1 for r in sub]
    out["after_submit"] = {"n": len(sub), "time_share_median": round(statistics.median(shares), 3) if shares else None,
                           "time_share_p75": round(q(shares, .75), 3) if shares else None,
                           "tool_calls_after_median": q(extra_calls, .5), "tool_calls_after_p75": q(extra_calls, .75),
                           "llm_steps_after_median": q([r["steps_after_submit"] for r in sub], .5),
                           "resubmitted": sum(r["tool_counts"].get("submit_patch", 0) > 1 for r in sub)}
    print(f"\nAfter the first submit_patch (n={len(sub)}): {json.dumps(out['after_submit'])}")

    # reliability problems, failures vs resolved
    print("\nReliability (share of tasks with at least one)")
    for key in ("malformed_tool_args", "loops_3x", "edit_failures", "cmd_timeouts", "nudges", "overflow"):
        a = sum(bool(r.get(key)) for r in ok)
        b = sum(bool(r.get(key)) for r in bad)
        print(f"  {key:22s} resolved {100 * a / max(1, len(ok)):5.1f}%   failed {100 * b / max(1, len(bad)):5.1f}%")
        out[key] = {"resolved": a, "failed": b}
    ctx = [r["max_prompt_tokens"] for r in rows]
    out["max_prompt_tokens"] = {"median": q(ctx, .5), "p90": q(ctx, .9), "max": max(ctx)}
    print(f"  max prompt tokens: median {q(ctx, .5)}, p90 {q(ctx, .9)}, max {max(ctx)} (limit 32768)")

    c = collections.Counter()
    tot = collections.Counter()
    for r in rows:
        tot[r["repo"]] += 1
        c[r["repo"]] += r["resolved"]
    out["by_repo"] = {k: [c[k], tot[k]] for k in tot}
    print("\nBy repo: " + ", ".join(f"{k.split('/')[-1]} {c[k]}/{tot[k]}" for k in sorted(tot)))
    return out


def main():
    tgzs = sorted(INPUT.rglob("eval_gate_base_b*.tgz"))
    print("archives:", [str(p.relative_to(INPUT)) for p in tgzs])
    rows_by = collections.defaultdict(dict)
    for p in tgzs:
        name = set_of(p)
        if name is None:
            print("skip (unknown set):", p)
            continue
        dest = WORK / "x" / name / p.stem
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(p) as t:
            t.extractall(dest, filter="data")
        for rep in dest.rglob("rep0"):
            results = {}
            for line in open(rep / "merged" / "task_results.jsonl"):
                if line.strip():
                    r = json.loads(line)
                    results[r["instance_id"]] = r
            for line in open(rep / "analysis" / "per_task.jsonl"):
                if not line.strip():
                    continue
                row = json.loads(line)
                iid = row["instance_id"]
                res = results.get(iid, {})
                tp = rep / "merged" / "traces" / f"trace_{iid.replace('/', '__')}.json"
                at = t_sub = t_end = None
                after = 0
                if tp.exists():
                    at, t_sub, t_end, after = first_submit(json.loads(tp.read_text()))
                row.update(calls_before_submit=at, t_submit=t_sub, t_end=t_end, steps_after_submit=after,
                           ending=ending(row, res), set=name)
                rows_by[name][iid] = row
    report = {}
    allrows = []
    for name in ("held-out", "real-bug"):
        rows = list(rows_by.get(name, {}).values())
        if not rows:
            print(f"\n!! no rows for {name}")
            continue
        report[name] = summarise(name, rows)
        allrows += rows
    if len(report) == 2:
        report["pooled"] = summarise("pooled", allrows)
    (WORK / "failure_analysis.json").write_text(json.dumps(report, indent=1))
    keep = ("instance_id", "set", "repo", "resolved", "failure_stage", "ending", "n_tool_calls", "calls_before_submit",
            "step_first_gold_lines", "step_first_edit", "patch_touches_gold", "patch_empty", "tests_after_last_edit",
            "max_prompt_tokens", "malformed_tool_args", "loops_3x", "harness_error")
    with open(WORK / "fa_per_task.jsonl", "w") as fh:
        for r in allrows:
            fh.write(json.dumps({k: r.get(k) for k in keep}) + "\n")
    import shutil
    shutil.rmtree(WORK / "x", ignore_errors=True)
    print("\nwrote failure_analysis.json, fa_per_task.jsonl")


if __name__ == "__main__":
    main()
