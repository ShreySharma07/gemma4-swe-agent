"""
Behaviour gate (docs/finetune_strategy_plan.md §4): before any task evaluation, ask base v7 and each candidate
adapter for their NEXT call at fixed decision points cut from v7's own runs (posttrain/gate_prefixes.py), with v7's
exact system prompt and tool declarations, and score the kind of call each one makes.

  file_in_view -> edit            test_failed -> edit or read (not submit, not the same test again)
  test_passed  -> submit_patch    early       -> read/search (not pytest, not submit)
  every response: exactly one tool call
One vLLM server (the scorer's settings) serves the base model and every adapter at once; n samples per prefix at
v7's sampling settings. A candidate passes when its edit rate at file_in_view and its overall pass rate are >= base,
and it never batches calls. Output: /kaggle/working/gate_report.json.
Run inside the venv: python -m evalkit.gate
"""
import json
import pathlib
import re
import shutil
import urllib.request

from .config import INPUT, WORK, Settings
from .inputs import Inputs
from .server import ScorerServer, set_scorer_env
from .util import log

PREFIXES = WORK / "gate_prefixes.json"
GATE_SYSTEM = WORK / "gate_system.md"
GATE_TOOLS = WORK / "gate_tools.json"


def kind(call):
    f = call.get("function", {})
    if f.get("name") != "run_command":
        return f.get("name") or "none"
    try:
        args = json.loads(f.get("arguments") or "{}")
    except Exception:  # noqa: BLE001
        return "malformed"
    c = re.sub(r"^(#[^\n]*\n)+", "", str(args.get("command", "")).strip())  # a leading reasoning comment is fine
    c = re.sub(r"^cd /workspace\s*&&\s*", "", c)
    if "PYEOF" in c and "write_text(" in c:
        return "edit"
    if re.match(r"(python3? -m )?pytest|python3? -m unittest", c):
        return "test"
    if re.match(r"(sed -n|cat|grep|find|ls|rg|head|tail)\b", c):  # "cat \b" never matched `cat -n` (fixed Oct 18)
        return "read"
    return "other"


def passes(typ, k, prev_cmd, cmd):
    if typ == "file_in_view":
        return k == "edit"
    if typ == "test_failed":
        return k in ("edit", "read") and cmd != prev_cmd
    if typ == "test_passed":
        return k == "submit_patch"
    return k == "read"  # early


def as_api(messages):
    """OpenAI chat format: earlier tool-call arguments must be JSON strings (the prefixes store dicts)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{**tc, "function": {**tc["function"], "arguments": tc["function"]["arguments"]
                                if isinstance(tc["function"]["arguments"], str)
                                else json.dumps(tc["function"]["arguments"], ensure_ascii=False)}}
                               for tc in m["tool_calls"]]
            m.setdefault("content", None)
        out.append(m)
    return out


def chat(base_url, model, messages, tools, n):
    body = {"model": model, "messages": messages, "tools": tools, "tool_choice": "auto", "n": n,
            "temperature": 0.2, "top_p": 0.95, "top_k": 40, "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def main():
    s = Settings.load()
    set_scorer_env()
    inp = Inputs.locate(False)
    names = [n for n in (s.gate_adapters or []) if n]
    sub = WORK / "gate_submission"
    shutil.rmtree(sub, ignore_errors=True)
    for n in names:  # adapters from attached training outputs: /kaggle/input/**/adapters/<name>/
        found = sorted(INPUT.glob(f"**/adapters/{n}/adapter_model.safetensors"))
        assert len(found) == 1, f"adapter {n!r}: expected exactly one attached copy, found {found}"
        shutil.copytree(found[0].parent, sub / "adapters" / n)
    from adk_submission import discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
    adapters = discover_adapters(str(sub), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS) if names else None
    server = ScorerServer(inp.model_path, adapters).start()
    base_url = server.server.base_url
    served = [m["id"] for m in json.load(urllib.request.urlopen(f"{base_url}/models"))["data"]]
    base_id = next(m for m in served if m not in names)
    log("served:", served, "| base:", base_id)
    system = GATE_SYSTEM.read_text()
    tools = json.loads(GATE_TOOLS.read_text())
    prefixes = json.loads(PREFIXES.read_text())
    report = {"models": {}, "n": s.gate_samples, "prefixes": len(prefixes)}
    for label, model in [("base", base_id)] + [(n, n) for n in names]:
        rows, by_type = [], {}
        for p in prefixes:
            msgs = as_api([{"role": "system", "content": system}] + p["messages"])
            prev = next((m["tool_calls"][0]["function"]["arguments"].get("command") for m in reversed(p["messages"])
                         if m.get("tool_calls")), None)
            try:
                out = chat(base_url, model, msgs, tools, s.gate_samples)
            except Exception as e:  # noqa: BLE001
                rows.append({"id": p["id"], "error": f"{type(e).__name__}: {e}"[:200]})
                continue
            for ch in out["choices"]:
                calls = ch["message"].get("tool_calls") or []
                k = kind(calls[0]) if calls else "no_call"
                cmd = None
                if calls:
                    try:
                        cmd = json.loads(calls[0]["function"].get("arguments") or "{}").get("command")
                    except Exception:  # noqa: BLE001
                        cmd = None
                ok = len(calls) == 1 and passes(p["type"], k, prev, cmd)
                rows.append({"id": p["id"], "type": p["type"], "kind": k, "n_calls": len(calls), "pass": ok,
                             "call": (calls[0]["function"].get("name", "") + " " + (cmd or ""))[:300] if calls
                             else (ch["message"].get("content") or "")[:300]})
                t = by_type.setdefault(p["type"], {"pass": 0, "n": 0, "kinds": {}})
                t["pass"] += ok
                t["n"] += 1
                t["kinds"][k] = t["kinds"].get(k, 0) + 1
        scored = [r for r in rows if "pass" in r]
        report["models"][label] = {
            "pass_rate": round(sum(r["pass"] for r in scored) / max(1, len(scored)), 3),
            "edit_rate_file_in_view": round(by_type.get("file_in_view", {}).get("pass", 0)
                                            / max(1, by_type.get("file_in_view", {}).get("n", 0)), 3),
            "batched_responses": sum(r["n_calls"] > 1 for r in scored),
            "no_call_responses": sum(r["n_calls"] == 0 for r in scored),
            "errors": sum("error" in r for r in rows), "by_type": by_type, "rows": rows}
        log(f"GATE {label}: pass {report['models'][label]['pass_rate']} | edit@file_in_view "
            f"{report['models'][label]['edit_rate_file_in_view']} | batched {report['models'][label]['batched_responses']}")
    base = report["models"]["base"]
    for n in names:
        m = report["models"][n]
        m["verdict"] = "PASS" if (m["edit_rate_file_in_view"] >= base["edit_rate_file_in_view"]
                                  and m["pass_rate"] >= base["pass_rate"] and m["batched_responses"] == 0
                                  and m["errors"] == 0) else "FAIL"
        log(f"VERDICT {n}: {m['verdict']}")
    (WORK / "gate_report.json").write_text(json.dumps(report, indent=1))
    server.stop()


if __name__ == "__main__":
    main()
