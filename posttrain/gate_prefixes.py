"""
Behaviour-gate prefixes (docs/finetune_strategy_plan.md §4): fixed decision points cut from v7's own runs, chosen
before any training, on tasks S1 does not train on (v7's 5 old-54 reps + its 129-task run). A candidate adapter must act at these points at least
as well as base v7 (evalkit/gate.py serves base + adapters and asks each for its next call).

Types (pass rule for the model's next call):
  file_in_view  v7 just read the file it then edited, right before v7's first edit -> an edit
  test_failed   after an edit, a pytest run failed                               -> an edit or a read (not submit, not
                                                                                   a re-run of the same test)
  test_passed   after an edit, a pytest run passed                               -> submit_patch
  early         after the first 2 calls                                          -> a targeted read/search (not pytest,
                                                                                   not submit)
Every response must contain exactly ONE tool call (Step D emitted 55–253 at once).
Output: JSON list of {id, type, task, rep, messages (OpenAI chat format, task first; the gate adds system + tools)}.
Usage: python posttrain/gate_prefixes.py OUT.json [--per-type 8]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_atif  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "data" / "runs" / "rs_v7_valid" / "x" / "eval_v7_noedit",  # 5 reps over the old 54 valid tasks
           ROOT / "data" / "runs" / "eval_v7_full" / "x" / "eval_v7_noedit"]  # v7 over all 129 tasks (more test points)
S1 = ROOT / "data" / "runs" / "s1_own"
NOISE = re.compile(r"Error in sitecustomize; set PYTHONVERBOSE for traceback:\nModuleNotFoundError: No module named 'wrapt'\n?")


def clean(content):
    """Remove the local sandbox's `wrapt` startup warning (absent on the scorer)."""
    try:
        j = json.loads(content)
    except Exception:  # noqa: BLE001
        return content
    for d in (j, j.get("details") if isinstance(j.get("details"), dict) else {}):
        for k in ("stderr", "error_message", "stdout"):
            if isinstance(d.get(k), str):
                d[k] = NOISE.sub("", d[k])
    return json.dumps(j, ensure_ascii=False)


def kind(call):
    f = call["function"]
    if f["name"] != "run_command":
        return f["name"]
    c = re.sub(r"^cd /workspace\s*&&\s*", "", str(f["arguments"].get("command", "")).strip())
    if "PYEOF" in c and "write_text(" in c:
        return "edit"
    if re.match(r"(python3? -m )?pytest|python3? -m unittest", c):
        return "test"
    if re.match(r"(sed -n|cat |grep|find|ls|rg|head|tail)\b", c):
        return "read"
    return "other"


def status(content):
    try:
        return json.loads(content).get("status")
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-type", type=int, default=8)
    a = ap.parse_args()
    s1_tasks = {f.stem[len("trace_"):].rsplit("__", 1)[0] for f in S1.glob("trace_*.json")}
    found = {"file_in_view": [], "test_failed": [], "test_passed": [], "early": []}
    for rep in [r for src in SOURCES for r in sorted(src.glob("rep*"))]:
        for t in sorted((rep / "merged" / "traces").glob("trace_*.json")):
            task = t.stem[len("trace_"):]
            if task in s1_tasks:
                continue
            convs, _ = render_atif.conversations(json.loads(t.read_text()))
            msgs = next(iter(convs.values()), [])
            for m in msgs:
                if m["role"] == "tool":
                    m["content"] = clean(m["content"])
            calls = [(i, m["tool_calls"][0]) for i, m in enumerate(msgs) if m["role"] == "assistant" and m.get("tool_calls")]
            resp = {i: msgs[i + 1]["content"] for i, _ in calls if i + 1 < len(msgs) and msgs[i + 1]["role"] == "tool"}

            def add(typ, cut):
                if all(p["task"] != task for p in found[typ]):  # one prefix per task per type
                    found[typ].append({"id": f"{typ}:{task}:{rep.parent.parent.parent.name}/{rep.name}", "type": typ, "task": task,
                                       "rep": f"{rep.parent.parent.parent.name}/{rep.name}",
                                       "messages": msgs[:cut]})
            if len(calls) >= 3:
                add("early", calls[2][0])
            ks = [kind(c) for _, c in calls]
            if "edit" in ks:
                e = ks.index("edit")
                if e > 0 and ks[e - 1] == "read":
                    path = re.search(r'pathlib\.Path\("([^"]+)"\)', calls[e][1]["function"]["arguments"]["command"])
                    prev = calls[e - 1][1]["function"]["arguments"].get("command", "")
                    if path and path.group(1) in prev and status(resp.get(calls[e - 1][0])) == "ok":
                        add("file_in_view", calls[e][0])
                for j in range(e + 1, len(calls) - 1):
                    if ks[j] == "test":
                        typ = "test_passed" if status(resp.get(calls[j][0])) == "ok" else "test_failed"
                        add(typ, calls[j + 1][0])
                        break
    picked = [p for typ in found for p in found[typ][:a.per_type]]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(picked, ensure_ascii=False, indent=0))
    print({typ: f"{min(len(v), a.per_type)} of {len(v)} available" for typ, v in found.items()},
          "| excluded S1 tasks:", len(s1_tasks))


if __name__ == "__main__":
    main()
