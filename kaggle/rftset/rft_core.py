"""
RS1: R2 factory rollouts -> thinking-ON training set (Kaggle CPU; pre-registered in docs/experiments.md, Oct 31).
Inputs (attached): R2 shard outputs (eval_rollout_s{0,1}.tgz), factory datasets v0/v1 (task_meta), the served model
dir (tokenizer.json + chat_template.jinja), submission v14 (embedded).
Per resolved trajectory of the main agent:
  - messages exactly as served: system.md, task prompt, then per step an assistant turn with the step's thinking as
    `reasoning_content` + its tool call, and the tool response. Gemma-4's template keeps thinking for every model turn
    after the last user turn, i.e. all of an agent run (checked: the rendered prompt tracks vLLM's logged
    prompt_tokens with a constant offset, so past thoughts are in the served context).
  - fidelity: rendered prompt tokens vs logged prompt_tokens at every step, for each candidate prefix (tool list with
    the code_analyzer AgentTool, ADK identity line); the variant with the smallest |gap| is used and the gap reported.
  - tags: stray files in patch, tests after last edit, tool calls, analyzer calls, drifted calls, overflow, src/lib/kind.
Outputs (/kaggle/working/rftset/): episodes.jsonl (full trajectories + render + train spans), windows_8k.jsonl
(render_atif-compatible windows for kaggle/train/trainkit), rftset_report.json.
"""
import glob
import json
import pathlib
import statistics
import tarfile

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

OUT = pathlib.Path("/kaggle/working/rftset")
WORK = pathlib.Path("/tmp/rftset")
MAX_WINDOW = 8192
AGENT = "swe_noedit_agent"
ANALYZER = {"name": "code_analyzer_agent",
            "description": "Analyzes repository source files and symbol graphs to locate root causes.",
            "parameters": {"type": "object", "properties": {"request": {"type": "string"}}, "required": ["request"]}}


# ------------------------------------------------------------------ Gemma-4 template + tokenizer (no transformers)
class Gemma:
    def __init__(self, model_dir):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, sort_keys=False: json.dumps(
            x, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)

        def _raise(m):
            raise jinja2.exceptions.TemplateError(m)
        env.globals["raise_exception"] = _raise
        d = pathlib.Path(model_dir)
        tpl = (d / "chat_template.jinja").read_text() if (d / "chat_template.jinja").exists() else \
            json.loads((d / "tokenizer_config.json").read_text())["chat_template"]
        self.tpl = env.from_string(tpl)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))

    def render(self, messages, tools, gen=False):
        return self.tpl.render(messages=messages, tools=tools, add_generation_prompt=gen, enable_thinking=True,
                               bos_token="<bos>", eos_token="<eos>")

    def n(self, text):
        return len(self.tok.encode(text, add_special_tokens=False).ids)


def model_spans(text):
    """char spans the model wrote (thinking, tool calls, text) inside model turns, minus embedded tool responses."""
    spans, i = [], 0
    while True:
        a = text.find("<|turn>model\n", i)
        if a < 0:
            return spans
        a += len("<|turn>model\n")
        b = text.find("<turn|>", a)
        b = len(text) if b < 0 else b + len("<turn|>")
        j = a
        while j < b:
            r = text.find("<|tool_response>", j, b)
            if r < 0:
                spans.append([j, b])
                break
            if r > j:
                spans.append([j, r])
            e = text.find("<tool_response|>", r, b)
            j = b if e < 0 else e + len("<tool_response|>")
        i = b


# ------------------------------------------------------------------ trajectories
def episode(trace, system):
    """-> (messages, per-step logged prompt_tokens [None for steps without an LLM call], drifted call count)."""
    steps = trace["steps"]
    task = next(s["message"] for s in steps if s.get("source") == "user")
    msgs, logged, drift = [{"role": "system", "content": system}, {"role": "user", "content": task}], [], 0
    for s in steps:
        if s.get("source") != "agent" or (s.get("extra") or {}).get("author", AGENT) != AGENT:
            continue
        logged.append(((s.get("metrics") or {}).get("prompt_tokens"), len(msgs)))
        tcs = s.get("tool_calls") or []
        if not tcs:
            msgs.append({"role": "assistant", "content": s.get("message") or ""})
            continue
        for t in tcs:
            if t["function_name"] != ANALYZER["name"] and classify(t["function_name"], t.get("arguments")):
                drift += 1
        msgs.append({"role": "assistant", "reasoning_content": s.get("message") or "",
                     "tool_calls": [{"id": t["tool_call_id"], "type": "function",
                                     "function": {"name": t["function_name"], "arguments": t.get("arguments") or {}}}
                                    for t in tcs]})
        obs = s.get("observation") or {}
        for k, res in enumerate(obs.get("results") or ([obs] if obs else [])):
            c = res.get("content")
            msgs.append({"role": "tool", "tool_call_id": tcs[min(k, len(tcs) - 1)]["tool_call_id"],
                         "content": c if isinstance(c, str) else json.dumps(c)})
    return msgs, logged, drift


def gaps(g, msgs, logged, tools):
    """logged - rendered prompt tokens at each step that has a logged count."""
    return [lp - g.n(g.render(msgs[:k], tools, gen=True)) for lp, k in logged if lp]


def windows(g, msgs, tools):
    """<= MAX_WINDOW windows at assistant-step boundaries: head (system + task) + consecutive steps (render_atif)."""
    head, steps, cur = msgs[:2], [], []
    for m in msgs[2:]:
        if m["role"] == "assistant" and cur:
            steps.append(cur)
            cur = []
        cur.append(m)
    if cur:
        steps.append(cur)
    out, cur = [], []
    for st in steps:
        if cur and g.n(g.render(head + [m for s in cur + [st] for m in s], tools)) > MAX_WINDOW:
            out.append(head + [m for s in cur for m in s])
            cur = [st]
        else:
            cur.append(st)
    if cur:
        out.append(head + [m for s in cur for m in s])
    return out


def lib_of(i):
    return i.split("_", 1)[1].rsplit("_", 1)[0]


def main(submission_dir, tools_base):
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    model_dir = next(pathlib.Path(p).parent for p in sorted(glob.glob("/kaggle/input/**/tokenizer.json", recursive=True))
                     if "w4a16" in p.lower())
    g = Gemma(model_dir)
    system = (submission_dir / "prompts" / "system.md").read_text()
    meta = {}
    for p in glob.glob("/kaggle/input/**/task_meta.jsonl", recursive=True):
        meta.update({json.loads(l)["id"]: json.loads(l) for l in open(p)})
    shards = {}
    for tgz in sorted(glob.glob("/kaggle/input/**/eval_rollout_s*.tgz", recursive=True)):
        name = pathlib.Path(tgz).stem.replace("eval_", "")
        if name in shards:
            continue
        with tarfile.open(tgz) as tf:
            tf.extractall(WORK, filter="data")
        root = WORK / f"eval_{name}" / "rep0"
        shards[name] = root
        print("shard", name, root, flush=True)
    assert len(shards) == 2, f"expected 2 R2 shards, got {sorted(shards)}"

    per_task, traces = {}, {}
    for name, root in shards.items():
        for l in open(root / "analysis" / "per_task.jsonl"):
            r = json.loads(l)
            per_task[r["instance_id"]] = r
        for p in (root / "merged" / "traces").glob("trace_*.json"):
            traces[p.stem.replace("trace_", "")] = p
    resolved = sorted(i for i, r in per_task.items() if r.get("resolved"))
    print(f"{len(per_task)} tasks, {len(resolved)} resolved, {len(traces)} traces", flush=True)

    # fidelity: pick the prefix variant (tool list / identity line) whose rendered prompt matches vLLM's counts
    ident = f'You are an agent. Your internal name is "{AGENT}".'
    variants = {
        "tools": (system, tools_base),
        "tools+analyzer": (system, tools_base + [{"type": "function", "function": ANALYZER}]),
        "tools+analyzer+identity": (ident + "\n\n" + system, tools_base + [{"type": "function", "function": ANALYZER}]),
    }
    probe = resolved[:: max(1, len(resolved) // 12)][:12]
    fid = {}
    for vn, (sysv, tl) in variants.items():
        gs = []
        for i in probe:
            msgs, logged, _ = episode(json.loads(traces[i].read_text()), sysv)
            gs += gaps(g, msgs, logged, tl)
        fid[vn] = {"median": statistics.median(gs), "min": min(gs), "max": max(gs), "n": len(gs)}
    best = min(fid, key=lambda v: abs(fid[v]["median"]))
    sysv, tools = variants[best]
    print("FIDELITY", json.dumps(fid), "->", best, flush=True)

    rows, n_win, stats = [], 0, []
    with open(OUT / "episodes.jsonl", "w") as fe, open(OUT / "windows_8k.jsonl", "w") as fw:
        for i in resolved:
            pt, m = per_task[i], meta.get(i, {})
            msgs, logged, drift = episode(json.loads(traces[i].read_text()), sysv)
            text = g.render(msgs, tools)
            tags = {"stray_files": pt.get("stray_files_in_patch") or [], "tests_after_last_edit": pt.get("tests_after_last_edit") or 0,
                    "tool_calls": pt.get("n_tool_calls"), "analyzer_calls": pt.get("analyzer_calls") or 0,
                    "drifted_calls": drift, "max_prompt_tokens": pt.get("max_prompt_tokens"),
                    "duration_s": pt.get("duration_s"), "src": i[:3], "lib": lib_of(i), "kind": m.get("kind"),
                    "ops": m.get("ops", [])}
            # core = what an RFT run should train on by default: no stray files, no drift, main agent only
            tags["core"] = not tags["stray_files"] and not drift and not tags["analyzer_calls"]
            g_last = gaps(g, msgs, logged[-1:], tools) if logged and logged[-1][0] else []
            nt = g.n(text)
            row = {"id": i, **tags, "n_tokens": nt, "gap_last_step": g_last[0] if g_last else None,
                   "messages": msgs, "text": text, "train_spans": model_spans(text)}
            fe.write(json.dumps(row) + "\n")
            for w, wm in enumerate(windows(g, msgs, tools)):
                wt = g.render(wm, tools)
                fw.write(json.dumps({"id": i, "role": AGENT, "window": w, "core": tags["core"], "text": wt,
                                     "train_spans": model_spans(wt), "n_tokens": g.n(wt)}) + "\n")
                n_win += 1
            rows.append({k: v for k, v in row.items() if k not in ("messages", "text", "train_spans")})
            stats.append(nt)
    core = [r for r in rows if r["core"]]

    def count(rs, key):
        out = {}
        for r in rs:
            for v in (r[key] if isinstance(r[key], list) else [r[key]]):
                out[v] = out.get(v, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))
    report = {
        "fidelity": fid, "variant": best, "resolved": len(rows), "core": len(core),
        "dropped_from_core": {"stray_files": sum(bool(r["stray_files"]) for r in rows),
                              "drift": sum(r["drifted_calls"] > 0 for r in rows),
                              "analyzer": sum(r["analyzer_calls"] > 0 for r in rows)},
        "tested_after_last_edit": sum(r["tests_after_last_edit"] > 0 for r in rows),
        "episode_tokens": {"median": statistics.median(stats), "p90": sorted(stats)[int(0.9 * len(stats))], "max": max(stats)},
        "over_8k": sum(t > MAX_WINDOW for t in stats), "windows_8k": n_win,
        "tool_calls_median": statistics.median(r["tool_calls"] for r in rows),
        "core_by": {k: count(core, k) for k in ("src", "kind", "lib")},
        "gap_last_step": {"median": statistics.median(x for x in (r["gap_last_step"] for r in rows) if x is not None)},
        "rows": rows,
    }
    (OUT / "rftset_report.json").write_text(json.dumps(report, indent=1))
    print("RS1 REPORT", json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
