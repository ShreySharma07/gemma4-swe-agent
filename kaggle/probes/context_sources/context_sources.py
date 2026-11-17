"""CS1: what fills the context? Tool results by kind, vs the model's own output.

CPU only. Saved traces: v17_superagent on new-60, and the v14_thinking base arm of both G2 gates.
Each agent step records the prompt size before it was generated, the tokens it generated and the
tool result it received. With thinking kept (v14, v17) the next prompt is
    prompt[i+1] = prompt[i] + generated[i] + result_tokens[i] + overhead.
result tokens are estimated by fitting  growth - generated = a * result_chars + b  per data set;
a and b are printed. Results are classified by tool and, for run_command, by command kind.
Prints summaries only; writes context_sources.json.
"""
import collections, json, pathlib, re, shutil, tarfile

INPUT, WORK = pathlib.Path("/kaggle/input"), pathlib.Path("/kaggle/working")
SOURCES = [("gemma4-eval-v17-new60", "eval_v17_superagent*.tgz", "v17 new-60"),
           ("gemma4-lora-gate-t2a", "eval_gate_base_b*.tgz", "v14 held-out"),
           ("gemma4-lora-gate-t2b-f3", "eval_gate_base_b*.tgz", "v14 real-bug")]
KINDS = [("test", r"\bpytest\b|unittest|tox\b|nox\b"),
         ("search", r"\bgrep\b|\brg\b|\bag\b|git grep|\bfind\b.*-name|ack\b"),
         ("read (shell)", r"\bsed -n\b|\bcat\b|\bhead\b|\btail\b|\bnl\b|\bawk\b"),
         ("git diff/status/log", r"\bgit (diff|status|log|show)\b"),
         ("python snippet", r"\bpython3? -c\b|\bpython3? - <<|<<\s*'?EOF"),
         ("listing", r"\bls\b|\btree\b|\bfind\b"),
         ("run script", r"\bpython3? [\w./-]+\.py\b")]


def kind(name, args):
    if name != "run_command":
        return name or "?"
    cmd = str((args or {}).get("command", ""))
    for k, rx in KINDS:
        if re.search(rx, cmd):
            return "cmd: " + k
    return "cmd: other"


def as_obj(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            import ast
            try:
                return ast.literal_eval(x)
            except Exception:
                return x
    return x


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def load(slug, pat):
    tasks = []
    for p in sorted(INPUT.rglob(pat)):
        if slug not in str(p):
            continue
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
                steps = [as_obj(s) if not isinstance(s, dict) else s for s in json.loads(tp.read_text()).get("steps") or []]
                agent = []
                seen = set()
                reads = collections.defaultdict(set)
                for s in steps:
                    m = as_obj(s.get("metrics")) or {}
                    if not isinstance(m, dict) or not m.get("prompt_tokens"):
                        continue
                    tcs = as_obj(s.get("tool_calls")) or []
                    obs = as_obj(s.get("observation")) or {}
                    content = obs.get("content", "") if isinstance(obs, dict) else str(obs)
                    if not isinstance(content, str):
                        content = json.dumps(content)
                    tc = tcs[0] if tcs else {}
                    name, args = tc.get("function_name"), as_obj(tc.get("arguments")) or {}
                    key = json.dumps([name, args], sort_keys=True, default=str)
                    repeat = bool(tcs) and key in seen
                    seen.add(key)
                    reread = False
                    if name == "read_file" and isinstance(args, dict):
                        f = str(args.get("filepath", ""))
                        try:
                            lo, hi = int(args.get("start_line") or 1), int(args.get("end_line") or (int(args.get("start_line") or 1) + 149))
                        except Exception:
                            lo, hi = 1, 150
                        lines = set(range(lo, hi + 1))
                        reread = len(lines & reads[f]) > 0.5 * len(lines)
                        reads[f] |= lines
                    err = '"status": "error"' in content[:200] or "Error" in content[:120]
                    agent.append({"prompt": int(m["prompt_tokens"]), "gen": int(m.get("completion_tokens") or 0),
                                  "chars": len(content) if tcs else 0, "kind": kind(name, args) if tcs else "(no tool)",
                                  "repeat": repeat, "reread": reread, "error": bool(tcs) and err})
                if agent:
                    tasks.append({"iid": iid, "resolved": bool(r.get("resolved")),
                                  "overflow": "context" in str(r.get("error") or "").lower(), "steps": agent})
    return tasks


def fit(tasks):
    xs, ys = [], []
    for t in tasks:
        st = t["steps"]
        for a, b in zip(st, st[1:]):
            g = b["prompt"] - a["prompt"] - a["gen"]
            if a["chars"] and g > 0:
                xs.append(a["chars"]); ys.append(g)
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return a, my - a * mx, n


def summarise(name, tasks):
    a, b, n = fit(tasks)
    print("\n" + "#" * 70 + f"\n{name}: {len(tasks)} tasks, {sum(t['resolved'] for t in tasks)} resolved, "
          f"{sum(t['overflow'] for t in tasks)} overflow; fit: tokens = {a:.3f} x chars + {b:.0f} (n={n})")
    out = {"tasks": len(tasks), "fit": [round(a, 4), round(b, 1)]}
    for label, grp in (("all tasks", tasks), ("overflow tasks", [t for t in tasks if t["overflow"]]),
                       ("resolved tasks", [t for t in tasks if t["resolved"]])):
        if not grp:
            continue
        base = sum(t["steps"][0]["prompt"] for t in grp)
        gen = sum(s["gen"] for t in grp for s in t["steps"][:-1])
        tok = collections.Counter(); calls = collections.Counter(); per = collections.defaultdict(list)
        rep = rr = errt = big = 0
        for t in grp:
            for s in t["steps"][:-1]:
                if not s["chars"]:
                    continue
                k = max(0.0, a * s["chars"] + b)
                tok[s["kind"]] += k; calls[s["kind"]] += 1; per[s["kind"]].append(k)
                rep += k if s["repeat"] else 0
                rr += k if s["reread"] else 0
                errt += k if s["error"] else 0
                big += k if k >= 1500 else 0
        res = sum(tok.values()); tot = base + gen + res
        print(f"\n  {label} (n={len(grp)}): context added = {tot / len(grp):,.0f} tokens per task on average")
        print(f"    system + issue {100 * base / tot:5.1f}%   model output {100 * gen / tot:5.1f}%   tool results {100 * res / tot:5.1f}%")
        print(f"    of tool results: exact repeats {100 * rep / max(1, res):.1f}%, read_file re-reads {100 * rr / max(1, res):.1f}%, "
              f"error results {100 * errt / max(1, res):.1f}%, single results >= 1,500 tokens {100 * big / max(1, res):.1f}%")
        rows = {}
        for k, v in tok.most_common():
            rows[k] = {"share": round(v / res, 3), "calls": calls[k], "median": round(q(per[k], .5)), "p90": round(q(per[k], .9))}
            print(f"    {k:24s} {100 * v / res:5.1f}% of results   calls {calls[k]:5d}   tokens/call median {q(per[k], .5):6.0f}  p90 {q(per[k], .9):6.0f}")
        out[label] = {"base": base / tot, "gen": gen / tot, "results": res / tot, "repeat": rep / max(1, res),
                      "reread": rr / max(1, res), "error": errt / max(1, res), "big": big / max(1, res), "kinds": rows}
    return out


report = {}
for slug, pat, name in SOURCES:
    tasks = load(slug, pat)
    if not tasks:
        print(f"!! {name}: no traces"); continue
    report[name] = summarise(name, tasks)
(WORK / "context_sources.json").write_text(json.dumps(report, indent=1))
shutil.rmtree(WORK / "x", ignore_errors=True)
print("\nwrote context_sources.json")
