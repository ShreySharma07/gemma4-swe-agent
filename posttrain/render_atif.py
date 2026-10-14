"""
ATIF trajectory -> Gemma-native training windows (docs/post_training_guide.md, step 2 "Data").

- Renders with Gemma-4's own chat template. enable_thinking must match the deployed config: our agents set
  thinking_budget: 0, so since adk-submission 0.2.13 the request carries enable_thinking=false (default here).
- One conversation per agent role (`extra.author`), so multi-agent traces (localizer / fixer) become separate samples.
- Drifted tool calls (analysis/format_drift.py classes) are dropped together with their tool response, so the adapter
  never sees or learns them; everything else is kept verbatim.
- Loss mask: only model-written spans (`<|tool_call>...<tool_call|>` and model text) inside model turns;
  system/user turns and the embedded `<|tool_response>...<tool_response|>` blocks are context only.
- Windows <= MAX_TOKENS (8k, the measured training envelope): system + task prefix, then as many consecutive model
  steps as fit; a step never straddles two windows.
Output JSONL rows: {"id", "role", "window", "text", "train_spans": [[char_start, char_end], ...], "n_tokens"}.
Usage: python posttrain/render_atif.py TRACE_DIR OUT.jsonl --system PROMPT.md [--tokenizer DIR] [--only-resolved FILE]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from format_drift import classify  # noqa: E402

MAX_TOKENS = 8192
TOOLS = json.loads((Path(__file__).parent / "tool_schemas.json").read_text())


def conversations(atif):
    """-> {author: [messages]} with drifted calls (and their responses) removed."""
    steps = atif.get("steps", [])
    task = next((s.get("message") for s in steps if s.get("source") == "user" and s.get("message")), "")
    convs, dropped, pending = {}, 0, {}
    for s in steps:
        for tc in s.get("tool_calls") or []:
            author = (tc.get("extra") or {}).get("author", "agent")
            msgs = convs.setdefault(author, [{"role": "user", "content": task}])
            name, args = tc.get("function_name"), tc.get("arguments")
            if classify(name, args):  # drift -> drop the call and remember to drop its response
                dropped += 1
                pending[tc.get("tool_call_id")] = None
                continue
            msgs.append({"role": "assistant", "tool_calls": [{"id": tc.get("tool_call_id"), "type": "function",
                                                              "function": {"name": name, "arguments": args or {}}}]})
            pending[tc.get("tool_call_id")] = (author, name)
        obs = s.get("observation")
        if obs:
            for res in (obs.get("results") or [obs]):
                cid = res.get("source_call_id") or res.get("tool_call_id")
                target = pending.pop(cid, "missing") if cid else "missing"
                if target is None:
                    continue  # response to a dropped call
                author = (obs.get("extra") or {}).get("author") or (target[0] if target != "missing" else "agent")
                msgs = convs.setdefault(author, [{"role": "user", "content": task}])
                if msgs and msgs[-1]["role"] == "assistant" and msgs[-1].get("tool_calls"):
                    msgs.append({"role": "tool", "tool_call_id": msgs[-1]["tool_calls"][0]["id"],
                                 "content": res.get("content") if isinstance(res.get("content"), str) else json.dumps(res.get("content"))})
        if s.get("source") == "agent" and s.get("message") and not s.get("tool_calls"):
            author = (s.get("extra") or {}).get("author", "agent")
            convs.setdefault(author, [{"role": "user", "content": task}]).append({"role": "assistant", "content": s["message"]})
    return convs, dropped


def model_spans(text):
    """char spans the model wrote: tool calls and model text, excluding embedded tool responses."""
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


THINKING = False  # matches thinking_budget: 0 in our sampling.yaml (adk-submission 0.2.13 -> enable_thinking=false)


def declared_tools(agent_yaml):
    """Tool declarations in the agent's own order (agent.yaml `tools:`), so train prompts match serve prompts.
    agent_tool sub-agents are declared to the model too, but their schema text is not reconstructed yet."""
    import re
    names = re.findall(r"^  - (\w+)\s*$", Path(agent_yaml).read_text(), re.M)
    by_name = {t.get("function", t).get("name"): t for t in TOOLS}
    missing = [n for n in names if n not in by_name]
    if "agent_tool:" in Path(agent_yaml).read_text():
        print("WARNING: agent_tool sub-agent declared by the agent is not in the rendered tool list", file=sys.stderr)
    assert not missing, f"no schema for {missing}"
    return [by_name[n] for n in names]


def render(tok, system, msgs, tools=None):
    return tok.apply_chat_template([{"role": "system", "content": system}] + msgs, tools=tools or TOOLS,
                                   tokenize=False, enable_thinking=THINKING)


def windows(tok, system, msgs):
    """split a conversation into <= MAX_TOKENS windows at assistant-step boundaries."""
    head, steps, cur = msgs[:1], [], []
    for m in msgs[1:]:
        if m["role"] == "assistant" and cur:
            steps.append(cur)
            cur = []
        cur.append(m)
    if cur:
        steps.append(cur)
    out, cur = [], []
    for st in steps:
        trial = render(tok, system, head + [m for s in cur + [st] for m in s])
        if len(tok(trial)["input_ids"]) > MAX_TOKENS and cur:
            out.append(head + [m for s in cur for m in s])
            cur = [st]
        else:
            cur.append(st)
    if cur:
        out.append(head + [m for s in cur for m in s])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("out")
    ap.add_argument("--system", required=True, help="system prompt file of the agent role being trained")
    ap.add_argument("--tokenizer", default="/tmp/g4tok")
    ap.add_argument("--only-resolved", default=None, help="behaviour/task_results json with {id, resolved}")
    ap.add_argument("--role", default=None, help="keep only this author (e.g. fixer_agent)")
    ap.add_argument("--agent", default=None, help="agent.yaml whose declared tools (and order) the prompt uses")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    system = Path(a.system).read_text()
    tools = declared_tools(a.agent) if a.agent else None
    global render
    _render = render
    render = lambda tok, system, msgs: _render(tok, system, msgs, tools)  # noqa: E731
    keep = None
    if a.only_resolved:
        keep = {x["id"] for x in json.loads(Path(a.only_resolved).read_text()) if x.get("resolved")}
    n_rows = n_tok = dropped_total = 0
    with open(a.out, "w") as f:
        for p in sorted(Path(a.trace_dir).glob("*.json")):
            iid = p.stem.replace("trace_", "")
            if keep is not None and iid not in keep:
                continue
            convs, dropped = conversations(json.loads(p.read_text()))
            dropped_total += dropped
            for role, msgs in convs.items():
                if a.role and role != a.role:
                    continue
                for w, wm in enumerate(windows(tok, system, msgs)):
                    text = render(tok, system, wm)
                    nt = len(tok(text)["input_ids"])
                    f.write(json.dumps({"id": iid, "role": role, "window": w, "text": text,
                                        "train_spans": model_spans(text), "n_tokens": nt}) + "\n")
                    n_rows += 1
                    n_tok += nt
    print(json.dumps({"windows": n_rows, "tokens": n_tok, "dropped_drift_calls": dropped_total}))


if __name__ == "__main__":
    main()
