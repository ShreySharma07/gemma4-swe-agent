"""OT1: how many output tokens does one model step need? (check before max_output_tokens 8192 -> 4096)

CPU only. Reads saved traces: the base arm (v14_thinking) of both G2 gates and v17_superagent on new-60.
Per step: completion tokens (thinking + answer + tool call, all count against max_output_tokens).
Prints summaries only; writes output_tokens.json.
"""
import collections, json, pathlib, shutil, tarfile

INPUT, WORK = pathlib.Path("/kaggle/input"), pathlib.Path("/kaggle/working")
SOURCES = [("gemma4-lora-gate-t2a", "eval_gate_base_b*.tgz", "v14 held-out"),
           ("gemma4-lora-gate-t2b-f3", "eval_gate_base_b*.tgz", "v14 real-bug"),
           ("gemma4-eval-v17-new60", "eval_v17_superagent*.tgz", "v17 new-60"),
           ("gemma4-eval-v24-smoke", "eval_v24_nothoughts*.tgz", "v24 smoke (6 tasks)")]
EDGES = (1024, 2048, 3072, 4096, 6144, 8000)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def main():
    report = {}
    for slug, pat, name in SOURCES:
        tgzs = [p for p in sorted(INPUT.rglob(pat)) if slug in str(p)]
        steps, tasks, growth = [], {}, []
        for p in tgzs:
            dest = WORK / "x" / slug / p.stem
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(p) as t:
                t.extractall(dest, filter="data")
            for mdir in dest.rglob("merged"):
                res = {}
                if (mdir / "task_results.jsonl").exists():
                    for line in open(mdir / "task_results.jsonl"):
                        if line.strip():
                            r = json.loads(line)
                            res[r["instance_id"]] = r
                for tp in sorted((mdir / "traces").glob("trace_*.json")):
                    iid = tp.stem[len("trace_"):]
                    r = res.get(iid) or res.get(iid.replace("__", "/")) or {}
                    big, maxp, n = 0, 0, 0
                    all_steps = json.loads(tp.read_text()).get("steps") or []
                    # v2: is thinking kept in the history? prompt growth after a step vs what that step generated
                    ms = [s for s in all_steps if (s.get("metrics") or {}).get("prompt_tokens")]
                    for s0, s1 in zip(ms, ms[1:]):
                        m0, m1 = s0["metrics"], s1["metrics"]
                        tcs0 = s0.get("tool_calls") or []
                        vis = len(str(s0.get("message") or "")) + sum(len(json.dumps(tc.get("arguments") or {})) for tc in tcs0)
                        obs = len(json.dumps(s0.get("observation") or ""))
                        growth.append((int(m0.get("completion_tokens") or 0), int(m1["prompt_tokens"]) - int(m0["prompt_tokens"]), vis, obs))
                    for s in all_steps:
                        m = s.get("metrics") or {}
                        c = int(m.get("completion_tokens") or 0)
                        maxp = max(maxp, int(m.get("prompt_tokens") or 0))
                        if not c:
                            continue
                        n += 1
                        tcs = s.get("tool_calls") or []
                        tool = tcs[0].get("function_name") if tcs else "(text only)"
                        arg = len(json.dumps(tcs[0].get("arguments") or {})) if tcs else 0
                        steps.append((c, tool, arg, len(str(s.get("message") or ""))))
                        big = max(big, c)
                    tasks[iid] = {"max_step": big, "max_prompt": maxp, "resolved": bool(r.get("resolved")),
                                  "error": str(r.get("error") or "")[:60], "steps": n}
        if not steps:
            print(f"\n!! {name}: no traces found ({len(tgzs)} archives)")
            continue
        cs = [s[0] for s in steps]
        out = {"archives": len(tgzs), "tasks": len(tasks), "resolved": sum(t["resolved"] for t in tasks.values()),
               "steps": len(cs), "completion_median": q(cs, .5), "p90": q(cs, .9), "p99": q(cs, .99), "max": max(cs)}
        print("\n" + "#" * 60 + f"\n{name}: {out}")
        print("steps at or above N output tokens / tasks with such a step (resolved among them)")
        out["at_or_above"] = {}
        for e in EDGES:
            ns = sum(c >= e for c in cs)
            ts = [t for t in tasks.values() if t["max_step"] >= e]
            out["at_or_above"][e] = {"steps": ns, "tasks": len(ts), "resolved": sum(t["resolved"] for t in ts)}
            print(f"  >= {e:5d}: {ns:5d} steps ({100 * ns / len(cs):5.2f}%)   {len(ts):3d} tasks ({sum(t['resolved'] for t in ts)} resolved)")
        bigs = [s for s in steps if s[0] >= 3072]
        c = collections.Counter(s[1] for s in bigs)
        out["big_step_tools"] = dict(c)
        print("  steps >= 3072 by first tool:", dict(c))
        if bigs:
            # argument chars ~ the visible part; the rest is thinking
            print("  of those: median tool-argument chars", q([s[2] for s in bigs], .5), "median message chars", q([s[3] for s in bigs], .5))
            out["big_step_arg_chars_median"] = q([s[2] for s in bigs], .5)
        mp = [t["max_prompt"] for t in tasks.values()]
        ov = [t for t in tasks.values() if "context" in t["error"].lower()]
        out["max_prompt"] = {"median": q(mp, .5), "p90": q(mp, .9), "max": max(mp)}
        out["overflow_tasks"] = len(ov)
        out["tasks_prompt_over_20k"] = sum(x >= 20000 for x in mp)
        print(f"  largest prompt per task: median {q(mp, .5)}, p90 {q(mp, .9)}, max {max(mp)}; overflow errors {len(ov)}; tasks over 20k: {out['tasks_prompt_over_20k']}")
        # steps whose visible output and tool result are small: growth should be small unless thinking is kept
        print("  prompt growth after a step with < 2000 result chars, by tokens generated:")
        out["growth"] = {}
        for lo, hi in ((0, 200), (200, 600), (600, 1500), (1500, 3000), (3000, 99999)):
            g = [x for x in growth if lo <= x[0] < hi and x[3] < 2000 and x[1] > 0]  # v3: no visible-chars filter (the message field holds the thinking text)
            if g:
                out["growth"][f"{lo}-{hi}"] = {"n": len(g), "generated_median": q([x[0] for x in g], .5), "growth_median": q([x[1] for x in g], .5),
                                               "visible_chars_median": q([x[2] for x in g], .5), "result_chars_median": q([x[3] for x in g], .5)}
                print(f"    generated {lo:4d}-{hi:5d}: n={len(g):4d}  generated median {q([x[0] for x in g], .5):5d}  growth median {q([x[1] for x in g], .5):5d}"
                      f"  visible chars {q([x[2] for x in g], .5)}  result chars {q([x[3] for x in g], .5)}")
        pos = [x for x in growth if x[1] > 0]
        out["growth_per_step_median"] = q([x[1] for x in pos], .5)
        out["generated_total_share_of_growth"] = round(sum(x[0] for x in pos) / max(1, sum(x[1] for x in pos)), 3)
        print(f"  growth per step: median {out['growth_per_step_median']}; all generated tokens / all growth = {out['generated_total_share_of_growth']}")
        report[name] = out
    (WORK / "output_tokens.json").write_text(json.dumps(report, indent=1))
    shutil.rmtree(WORK / "x", ignore_errors=True)
    print("\nwrote output_tokens.json")


main()
