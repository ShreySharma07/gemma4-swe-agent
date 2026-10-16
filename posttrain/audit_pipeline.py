"""
CPU audits of the Step C training data (docs/atul_data_training_research_20261016.md, experiments E0 and E2).
Tokenizer only; no model, no torch. Run with a Python that has `transformers` (tokenizers) installed.

  e0  Provenance: re-render the converted traces and re-tokenize them with the current code and the given tokenizer,
      then require exact equality with the preserved windows JSONL and the NPZ Step C trained on.
  e2  What the windows supervise: per supervised action, its kind (read / test / edit / ...), its window position, whether
      the window starts mid-run (no earlier steps), whether an edited file was shown earlier in the same window, the
      share of supervised tokens per kind, the first supervised action of every window, and boundary/mask checks
      (tool responses never supervised, every call span closed, consecutive calls without a response = batching).
Usage:
  python posttrain/audit_pipeline.py e0 --traces data/runs/public_200 --windows data/runs/public_200_windows.jsonl \
      --npz data/runs/train_public_200.npz --tokenizer data/g4tok --out data/runs/audit_e0
  python posttrain/audit_pipeline.py e2 --windows data/runs/public_200_windows.jsonl --tokenizer data/g4tok \
      --out data/runs/audit_e2
"""
import argparse
import collections
import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYSTEM = HERE.parent / "submissions" / "v7_noedit" / "prompts" / "system.md"
AGENT = HERE.parent / "submissions" / "v7_noedit" / "agent.yaml"
CALL = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.S)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def e0(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    re_windows, re_npz = out / "rerendered_windows.jsonl", out / "retokenized.npz"
    r = subprocess.run([sys.executable, str(HERE / "render_atif.py"), a.traces, str(re_windows), "--system", str(SYSTEM),
                        "--agent", str(AGENT), "--tokenizer", a.tokenizer], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    old = [json.loads(l) for l in open(a.windows)]
    new = [json.loads(l) for l in open(re_windows)]
    diffs = [i for i, (x, y) in enumerate(zip(old, new)) if x != y]
    r = subprocess.run([sys.executable, str(HERE / "tokenize_windows.py"), str(re_windows), str(re_npz), "--tokenizer",
                        a.tokenizer, "--val-frac", "0.05"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    import numpy as np
    p, q = np.load(a.npz), np.load(re_npz)
    arrays = {k: bool(p[k].shape == q[k].shape and (p[k] == q[k]).all()) for k in p.files}
    traces = sorted(Path(a.traces).glob("trace_*.json"))
    report = {
        "windows_rows": {"preserved": len(old), "rerendered": len(new), "rows_differing": len(diffs),
                         "first_diffs": diffs[:5]},
        "npz_arrays_equal": arrays, "npz_sha256": {"preserved": sha(a.npz), "retokenized": sha(re_npz)},
        "inputs_sha256": {"windows_jsonl": sha(a.windows), "tokenizer.json": sha(Path(a.tokenizer) / "tokenizer.json"),
                          "chat_template.jinja": sha(Path(a.tokenizer) / "chat_template.jinja"),
                          "system.md": sha(SYSTEM), "agent.yaml": sha(AGENT),
                          "traces_manifest": hashlib.sha256("".join(f"{t.name}:{sha(t)}\n" for t in traces).encode()).hexdigest(),
                          "n_traces": len(traces)},
        "code_sha256": {f: sha(HERE / f) for f in ("convert_public.py", "render_atif.py", "tokenize_windows.py",
                                                    "tool_schemas.json")},
    }
    report["verdict"] = ("REPRODUCED: preserved windows and NPZ regenerate exactly from the converted traces"
                         if not diffs and len(old) == len(new) and all(arrays.values())
                         else "NOT REPRODUCED: see windows_rows / npz_arrays_equal")
    (out / "e0_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def kind(name, args):
    """Coarse action kind from the rendered call (Gemma native args: key:<|"|>value<|"|>)."""
    if name != "run_command":
        return name
    cmd = re.sub(r'<\|"\|>', "", args)
    cmd = re.sub(r"^command:", "", cmd).strip()
    c = re.sub(r"^cd /workspace\s*&&\s*", "", cmd)
    if "PYEOF" in c and "write_text(" in c:
        return "edit (snippet)"
    if re.match(r"(python3? -m )?pytest|python3? -m unittest", c):
        return "test (pytest)"
    if c.startswith(("python", "python3")):
        return "run python script"
    if c.startswith("cat -n") or c.startswith("cat "):
        return "read (cat -n)"
    if c.startswith("sed -n"):
        return "read (sed -n)"
    if re.match(r"(grep|find|ls|rg)\b", c):
        return "search (grep/find/ls)"
    if c.startswith("git "):
        return "git"
    return "other shell"


def edited_path(name, args):
    if name == "write_file":
        m = re.search(r'filepath:<\|"\|>(.*?)<\|"\|>', args)
        return m.group(1) if m else None
    m = re.search(r'pathlib\.Path\("([^"]+)"\)', args)
    return m.group(1) if m and "write_text(" in args else None


def e2(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = [json.loads(l) for l in open(a.windows)]
    sup_tok = collections.Counter()      # supervised tokens per action kind
    acts = collections.Counter()         # supervised actions per kind
    first_act = collections.Counter()    # first supervised action of each window
    first_act_cont = collections.Counter()
    pos_kind = collections.defaultdict(collections.Counter)  # window index -> kinds
    checks = collections.Counter()
    edits_unseen, edits_total = 0, 0
    for r in rows:
        text, spans = r["text"], r["train_spans"]
        cont = r["window"] > 0
        checks["windows"] += 1
        checks["continuation_windows"] += cont
        shown = set()  # paths whose content appears in a tool response earlier in this window
        first = True
        for s, e in spans:
            seg = text[s:e]
            if "<|tool_response>" in seg or "<tool_response|>" in seg:
                checks["SUPERVISED_TOOL_RESPONSE"] += 1
            if seg.count("<|tool_call>") != seg.count("<tool_call|>"):
                checks["UNCLOSED_CALL_SPAN"] += 1
            calls = CALL.findall(seg)
            if len(calls) > 1:
                checks["spans_with_multiple_calls"] += 1
            if not calls:
                k = "final text" if seg.strip() else "empty"
                n = len(tok(seg, add_special_tokens=False)["input_ids"])
                sup_tok[k] += n
                acts[k] += 1
                if first:
                    first_act[k] += 1
                    if cont:
                        first_act_cont[k] += 1
                    first = False
                continue
            n = len(tok(seg, add_special_tokens=False)["input_ids"])
            for name, args in calls:
                k = kind(name, args)
                acts[k] += 1
                sup_tok[k] += n / len(calls)
                pos_kind[r["window"]][k] += 1
                if first:
                    first_act[k] += 1
                    if cont:
                        first_act_cont[k] += 1
                    first = False
                p = edited_path(name, args)
                if p and k in ("edit (snippet)", "write_file"):
                    edits_total += 1
                    if k == "edit (snippet)" and not any(p in x for x in shown):
                        edits_unseen += 1
            # what the following tool response shows (for the "edited file shown earlier" check)
            nxt = text.find("<tool_response|>", e)
            resp = text[e:nxt] if nxt > 0 else ""
            for name, args in calls:
                c = re.sub(r'<\|"\|>', "", args)
                for m in re.finditer(r"(?:cat -n|sed -n '[^']*'|cat)\s+([\w./-]+\.py)", c):
                    if '"status": "ok"' in resp:
                        shown.add(m.group(1))
        # batching: two calls with no response between them anywhere in the window
        if re.search(r"<tool_call\|>\s*<\|tool_call>", text):
            checks["windows_with_back_to_back_calls"] += 1
    tot_tok = sum(sup_tok.values())
    report = {
        "windows": checks["windows"], "continuation_windows (start mid-run, earlier steps absent)": checks["continuation_windows"],
        "mask_checks": {k: checks[k] for k in ("SUPERVISED_TOOL_RESPONSE", "UNCLOSED_CALL_SPAN", "spans_with_multiple_calls",
                                                "windows_with_back_to_back_calls")},
        "supervised_actions_by_kind": dict(acts.most_common()),
        "supervised_token_share_by_kind": {k: round(v / tot_tok, 3) for k, v in sup_tok.most_common()},
        "first_supervised_action_all_windows": dict(first_act.most_common()),
        "first_supervised_action_continuation_windows": dict(first_act_cont.most_common()),
        "kinds_by_window_index": {str(w): dict(c.most_common(5)) for w, c in sorted(pos_kind.items())},
        "snippet_edits_whose_file_was_not_shown_earlier_in_the_same_window": f"{edits_unseen}/{edits_total}",
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e2_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p0 = sub.add_parser("e0")
    p0.add_argument("--traces", required=True)
    p0.add_argument("--windows", required=True)
    p0.add_argument("--npz", required=True)
    p0.add_argument("--tokenizer", default="data/g4tok")
    p0.add_argument("--out", required=True)
    p2 = sub.add_parser("e2")
    p2.add_argument("--windows", required=True)
    p2.add_argument("--tokenizer", default="data/g4tok")
    p2.add_argument("--out", required=True)
    a = ap.parse_args()
    {"e0": e0, "e2": e2}[a.cmd](a)


if __name__ == "__main__":
    main()
