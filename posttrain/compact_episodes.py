"""
S3 compactor (docs/finetune_strategy_plan.md): converted public runs -> ONE grounded window per run.

E2 found that Step C's windows taught source edits without the edited code in view (79%), that 80% of windows
started mid-run, and that 56% of the loss was scratch-script writing. A compact episode fixes all three:

  task message (v7 template)
  -> up to 2 searches whose output names an edited file (how it was located)
  -> for each edited source file: its latest read before the first edit, narrowed to `sed -n 'a,bp' path` around
     the region the edit replaces (real file lines reconstructed from the teacher's `cat -n` output)
  -> the successful source edits (snippet-replace scripts), in order
  -> one run of an EXISTING test after the final edit (`... 2>&1 | tail -n 30`), if the teacher ran one
  -> submit_patch, then a short final message
Dropped: new scratch files (write_file: repro/test/verification scripts) and runs of them, failed edit attempts,
everything else. The teacher's short decision text (convert_public.py --reasoning) is kept as a leading `# ...`
comment in the command, because Gemma's template renders an assistant message's text AFTER its tool call.
An episode is rejected if any kept edit's old text is not visible in an earlier kept read (grounding), or if the
rendered window exceeds 8,192 tokens.
S3b options (behaviour gate, Oct 18: S3 submitted after FAILED tests 28/32 and edited at early prefixes 10/32, because
every S3 episode was search -> read -> edit -> passing test -> submit):
  --feedback   keep real test results: each failed existing-test run between edits (real status and output, the
               command as the teacher ran it), and the final run only if it passed; runs of test files the teacher
               edited are dropped (those edits are not in the episode). Teaches fail -> fix, pass -> submit.
  --explore N  keep the teacher's last N searches/reads of other files before the first edit (real outputs).
  --pre-test   keep the teacher's last existing-test run before the first edit (real result, pass or fail), so a
               test is not always followed by submit: before the fix it leads to reading/editing.
Defaults reproduce S3 exactly.
Usage: python posttrain/compact_episodes.py CONVERTED_DIR OUT_DIR [--tokenizer data/g4tok] [--feedback] [--explore N]
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_atif  # noqa: E402

AUTHOR = "swe_noedit_agent"
MAX_TOKENS = 8192
SYSTEM = Path(__file__).resolve().parents[1] / "submissions" / "v7_noedit" / "prompts" / "system.md"
AGENT = Path(__file__).resolve().parents[1] / "submissions" / "v7_noedit" / "agent.yaml"
OLD = re.compile(r'^old = (r?)("""|\'\'\')(.*?)\2\nnew = ', re.S | re.M)
NEW = re.compile(r'^new = (r?)("""|\'\'\')(.*?)\2\nassert ', re.S | re.M)
PATH = re.compile(r'pathlib\.Path\("([^"]+)"\)')
CD = re.compile(r"^cd /workspace\s*&&\s*")
TEACHER_ENV = re.compile(r"/opt/conda/envs/testbed/lib/python3\.\d+/site-packages")  # teacher's env -> sandbox's
NUM = re.compile(r"^\s*(\d+)\t(.*)$")


class Reject(Exception):
    pass


def cmd_of(call):
    return str((call.get("arguments") or {}).get("command", ""))


def is_test_path(p):
    return p.startswith("tests/") or "/tests/" in p or Path(p).name.startswith("test_") or p.endswith("_test.py")


def parse(trace):
    """-> task message, [(call dict, response str, reason str|None)] in order."""
    steps = trace["steps"]
    task = steps[0]["message"]
    resp = {}
    for s in steps:
        o = s.get("observation")
        if o:
            for r in o.get("results") or []:
                resp[r["source_call_id"]] = r["content"]
    out = []
    for s in steps:
        for tc in s.get("tool_calls") or []:
            out.append((tc, resp.get(tc["tool_call_id"], ""), s.get("message")))
    final = next((s["message"] for s in reversed(steps) if s.get("source") == "agent" and not s.get("tool_calls")
                  and s.get("message")), "")
    return task, out, final


def ok(r):
    try:
        return json.loads(r).get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


def stdout(r):
    try:
        d = json.loads(r)
    except Exception:  # noqa: BLE001
        return ""
    return d.get("stdout") or (d.get("details") or {}).get("stdout") or ""


def read_target(cmd):
    """Path of a file read with `cat -n PATH [| sed -n 'a,bp']`, else None."""
    c = CD.sub("", cmd.strip())
    m = re.match(r"cat -n (\S+)(\s*\|\s*sed -n '\d+,(\d+|\$)p')?\s*$", c)
    return m.group(1) if m else None


def file_lines(cat_out):
    """`cat -n` output -> {line number: text}."""
    lines = {}
    for line in cat_out.splitlines():
        m = NUM.match(line)
        if m:
            lines[int(m.group(1))] = m.group(2)
    return lines


def locate(lines, old):
    """First line number where `old` starts in the numbered lines, or None."""
    nums = sorted(lines)
    first = old.splitlines()[0] if old.strip() else ""
    want = old.splitlines()
    for i, n in enumerate(nums):
        if lines[n] == first and [lines.get(nums[j]) for j in range(i, min(i + len(want), len(nums)))] == want:
            return n
    return None


FILLER = re.compile(r"^(i'll help|i will help|great|perfect|excellent|good|now,? let me (run|check|test)|let me (run|try running))", re.I)
INFORMATIVE = re.compile(r"`|\w+\.\w+|\w+\(\)|\b(bug|issue|fix|problem|because|cause|should|wrong|missing|instead)\b", re.I)


def with_reason(cmd, reason):
    """Prefix the teacher's decision as a shell comment, only if it says something about the code or the fix."""
    if not reason or len(reason) < 25 or FILLER.match(reason.strip()) or not INFORMATIVE.search(reason):
        return cmd
    return "# " + reason.replace("\n", " ").strip() + "\n" + cmd


def compact(trace, margin=25, feedback=False, explore=0, pre_test=False):
    task, calls, final = parse(trace)
    created = {(tc.get("arguments") or {}).get("filepath") for tc, _, _ in calls if tc["function_name"] == "write_file"}
    edits, reads = [], {}  # reads: path -> latest successful cat -n before each edit, as (index, lines)
    for i, (tc, r, reason) in enumerate(calls):
        if tc["function_name"] != "run_command":
            continue
        c = cmd_of(tc)
        t = read_target(c)
        if t and ok(r):
            reads.setdefault(t, []).append((i, file_lines(stdout(r))))
            continue
        if "PYEOF" in c and "write_text(" in c and ok(r):
            p = PATH.search(c)
            m, mn = OLD.search(c), NEW.search(c)
            if not p or not m or not mn:
                continue
            path = p.group(1)
            if path in created or is_test_path(path):
                continue  # edits to scratch or test files are not source fixes
            edits.append((i, path, m.group(3), tc, r, reason, mn.group(3)))
    if not edits:
        raise Reject("no successful source edit")
    kept = []  # (order index, call, response, reason)
    shown = {}  # path -> text shown in kept reads
    for path in dict.fromkeys(p for _, p, *_ in edits):
        first_edit = min(i for i, p, *_ in edits if p == path)
        olds = [o for _, p, o, *_ in edits if p == path]
        cands = [(i, ls) for i, ls in reads.get(path, []) if i < first_edit]
        hit = next(((i, ls, locate(ls, olds[0])) for i, ls in reversed(cands) if locate(ls, olds[0])), None)
        if not hit:
            raise Reject("edited file not read with the edited region visible")
        i, ls, start = hit
        spans = [(st, st + len(o.splitlines())) for o in olds for st in [locate(ls, o)] if st]  # all edits in this read
        lo, hi = max(1, min(a for a, _ in spans) - margin), max(b for _, b in spans) + margin
        rng = [n for n in sorted(ls) if lo <= n <= hi]
        body = "\n".join(ls[n] for n in rng) + "\n"
        cmd = f"sed -n '{rng[0]},{rng[-1]}p' {path}"
        kept.append((i, {"function_name": "run_command", "arguments": {"command": with_reason(cmd, calls[i][2])}},
                     json.dumps({"status": "ok", "stdout": body, "stderr": "", "exit_code": 0}, ensure_ascii=False)))
        shown[path] = body
        # how it was located: up to 2 earlier searches whose output names this file
        found = [(j, tc, r, rs) for j, (tc, r, rs) in enumerate(calls[:i])
                 if tc["function_name"] == "run_command" and re.match(r"(grep|find|rg|ls)\b", CD.sub("", cmd_of(tc)))
                 and path in stdout(r)]
        for j, tc, r, rs in found[-2:]:
            kept.append((j - 0.5, {"function_name": "run_command",
                                   "arguments": {"command": with_reason(CD.sub("", cmd_of(tc)), rs)}}, r))
    if explore:  # the teacher's investigation before the first edit: other files' searches and reads, as they ran
        first = min(e[0] for e in edits)
        edited = {e[1] for e in edits}
        inv = [(j, tc, r, rs) for j, (tc, r, rs) in enumerate(calls[:first]) if tc["function_name"] == "run_command"
               and ok(r) and j not in {k[0] for k in kept} and j - 0.5 not in {k[0] for k in kept}
               and (re.match(r"(grep|find|rg|ls)\b", CD.sub("", cmd_of(tc)))
                    or (read_target(cmd_of(tc)) and read_target(cmd_of(tc)) not in edited))]
        for j, tc, r, rs in inv[-explore:]:
            kept.append((j - 0.25, {"function_name": "run_command",
                                    "arguments": {"command": with_reason(CD.sub("", cmd_of(tc)), rs)}}, r))
    if pre_test:
        first = min(e[0] for e in edits)
        test_files = {(tc.get("arguments") or {}).get("filepath") for tc, _, _ in calls} | \
                     {m.group(1) for tc, _, _ in calls for m in [PATH.search(cmd_of(tc))] if m and is_test_path(m.group(1))}
        pre = [(j, tc, r, rs) for j, (tc, r, rs) in enumerate(calls[:first]) if tc["function_name"] == "run_command"
               and re.match(r"(python3? -m )?pytest", CD.sub("", cmd_of(tc))) and not any(f in cmd_of(tc) for f in test_files if f)]
        if pre:
            j, tc, r, rs = pre[-1]
            kept.append((j, {"function_name": "run_command", "arguments": {"command": with_reason(CD.sub("", cmd_of(tc)), rs)}}, r))
    seen_idx = set()
    kept = [k for k in kept if not (k[0] in seen_idx or seen_idx.add(k[0]))]  # a search can locate two files
    for i, path, old, tc, r, reason, new in edits:  # simulate the visible text as edits apply
        if old not in shown.get(path, ""):
            raise Reject("edit not grounded in a kept read")
        shown[path] = shown[path].replace(old, new, 1)
        kept.append((i, {"function_name": "run_command", "arguments": {"command": with_reason(cmd_of(tc), reason)}}, r))
    last_edit = max(e[0] for e in edits)
    if feedback:
        test_files = {(tc.get("arguments") or {}).get("filepath") for tc, _, _ in calls} | \
                     {m.group(1) for tc, _, _ in calls for m in [PATH.search(cmd_of(tc))] if m and is_test_path(m.group(1))}
        runs = [(j, tc, r, rs) for j, (tc, r, rs) in enumerate(calls) if j > min(e[0] for e in edits)
                and tc["function_name"] == "run_command" and re.match(r"(python3? -m )?pytest", CD.sub("", cmd_of(tc)))
                and not any(f in cmd_of(tc) for f in test_files if f)]
        edit_idx = sorted(e[0] for e in edits)
        for j, tc, r, rs in runs:
            later = [e for e in edit_idx if e > j]
            if later and not ok(r) and not any(j < j2 < later[0] for j2, *_ in runs):  # last failed run before a fix
                kept.append((j, {"function_name": "run_command", "arguments": {"command": with_reason(CD.sub("", cmd_of(tc)), rs)}}, r))
        final_runs = [x for x in runs if x[0] > last_edit]
        if final_runs and ok(final_runs[-1][2]):
            j, tc, r, rs = final_runs[-1]
            kept.append((j, {"function_name": "run_command", "arguments": {"command": with_reason(CD.sub("", cmd_of(tc)), rs)}}, r))
    tests = [] if feedback else [(j, tc, r, rs) for j, (tc, r, rs) in enumerate(calls) if j > last_edit and tc["function_name"] == "run_command"
             and re.match(r"(python3? -m )?pytest", CD.sub("", cmd_of(tc)))
             and not any(f in cmd_of(tc) for f in created if f)]
    if tests:
        j, tc, r, rs = tests[-1]
        out_lines = (stdout(r) or json.loads(r).get("error_message", "")).splitlines()[-30:]
        cmd = CD.sub("", cmd_of(tc)) + " 2>&1 | tail -n 30"
        kept.append((j, {"function_name": "run_command", "arguments": {"command": with_reason(cmd, rs)}},
                     json.dumps({"status": "ok", "stdout": "\n".join(out_lines) + "\n", "stderr": "", "exit_code": 0},
                                ensure_ascii=False)))
    kept.sort(key=lambda x: x[0])
    submit = next(((tc, r) for tc, r, _ in calls if tc["function_name"] == "submit_patch"), None)
    if not submit:
        raise Reject("no submit")
    files = len({e[1] for e in edits})
    sub = json.loads(submit[1]) if submit[1] else {"status": "ok"}
    sub["files_changed"] = files  # the teacher's count included the scratch files we dropped
    steps = [{"source": "user", "message": task}]
    for n, (_, call, r) in enumerate(kept + [(None, {"function_name": "submit_patch", "arguments": {}},
                                              json.dumps(sub, ensure_ascii=False))]):
        r = TEACHER_ENV.sub("/usr/local/lib/python3.13/site-packages", r)
        cid = f"call_{n + 1}"
        steps.append({"source": "agent", "tool_calls": [{"tool_call_id": cid, "function_name": call["function_name"],
                                                         "arguments": call["arguments"], "extra": {"author": AUTHOR}}]})
        steps.append({"source": "system", "observation": {"results": [{"source_call_id": cid, "content": r}],
                                                          "extra": {"author": AUTHOR,
                                                                    "tool_name": call["function_name"]}}})
    msg = re.split(r"\n\s*\n", final.strip())[0][:300] if final else "Fixed the issue and submitted the patch."
    steps.append({"source": "agent", "message": msg, "extra": {"author": AUTHOR}})
    return {"schema_version": "ATIF-v1.7", "agent": {"name": "compact_episodes", "source": trace.get("agent")},
            "source": trace.get("source"), "steps": steps}, len(kept) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("converted")
    ap.add_argument("out")
    ap.add_argument("--tokenizer", default="data/g4tok")
    ap.add_argument("--feedback", action="store_true", help="S3b: keep real failed-test -> fix and passing-final-test results")
    ap.add_argument("--pre-test", action="store_true", help="S3b: keep the last existing-test run before the first edit")
    ap.add_argument("--explore", type=int, default=0, help="S3b: keep the last N investigation calls before the first edit")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    system = SYSTEM.read_text()
    tools = render_atif.declared_tools(AGENT)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    why, kept_calls, tokens = collections.Counter(), [], []
    for f in sorted(Path(a.converted).glob("trace_*.json")):
        # over 8k: narrower reads (margin 8), then (S3b) fewer investigation calls
        tries = [(25, a.explore), (8, a.explore)] + [(8, k) for k in sorted({a.explore // 2, 0}) [::-1] if k < a.explore]
        try:
            for margin, explore in tries:
                ep, n = compact(json.loads(f.read_text()), margin=margin, feedback=a.feedback, explore=explore,
                                pre_test=a.pre_test)
                convs, _ = render_atif.conversations(ep)
                text = render_atif.render(tok, system, next(iter(convs.values())), tools)
                nt = len(tok(text, add_special_tokens=False)["input_ids"])
                if nt <= MAX_TOKENS:
                    break
            else:
                raise Reject("over 8192 tokens")
        except Reject as e:
            why[str(e)] += 1
            continue
        (out / f.name).write_text(json.dumps(ep, ensure_ascii=False))
        why["kept"] += 1
        kept_calls.append(n)
        tokens.append(nt)
    report = {"episodes": dict(why.most_common()), "calls_median": sorted(kept_calls)[len(kept_calls) // 2] if kept_calls else None,
              "tokens_median": sorted(tokens)[len(tokens) // 2] if tokens else None, "tokens_total": sum(tokens),
              "tokens_max": max(tokens) if tokens else None}
    (out / "compact_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
