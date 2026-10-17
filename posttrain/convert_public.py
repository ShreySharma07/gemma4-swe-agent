"""
Public OpenHands trajectories (nebius/SWE-rebench-openhands-trajectories, CC-BY-4.0) -> ATIF traces in v7_noedit's
exact call format, so posttrain/render_atif.py renders them like our own runs (docs/post_training_guide.md, Oct 13).

Kept: resolved runs, exit_status "submit", repos outside our held-out eval, <= MAX_CALLS v7 calls after conversion.
Mapping (OpenHands -> v7, responses rebuilt in swegemma 0.2.11's JSON, outputs cut at 5000 chars like the harness):
  execute_bash{command}            -> run_command; footer "[... exit code N ...]" -> exit code; output -> stdout
  str_replace_editor view (file)   -> run_command `cat -n path` (+ `| sed -n 'a,bp'` for view_range)
  str_replace_editor view (dir)    -> run_command `find path -maxdepth 2 -not -path '*/.*'`; the first root listing
                                      becomes the task message's workspace tree instead
  str_replace_editor create        -> write_file{filepath, content}
  str_replace_editor str_replace   -> run_command snippet-replace script (the form in v7's system prompt); a failed
                                      replacement -> the script's AssertionError
  str_replace_editor insert        -> run_command line-insert script
  think / task_tracker / prose     -> dropped (v7 writes no text before tool calls; thinking is off)
  finish{message}                  -> submit_patch, then the final text-only reply
  undo_edit, interactive input (is_input), timeouts -> trajectory dropped
Paths: /workspace/<repo>__<ver> -> /workspace (v7's repo root); the conda interpreter path -> python.
The task message is v7's harness template (taken from one of our v7 traces) with this issue's repo, problem
statement and tree.
Usage:
  python3 posttrain/convert_public.py --template TRACE.json --hf-sample 50 OUT_DIR      (datasets-server API sample)
  python3 posttrain/convert_public.py --template TRACE.json --jsonl ROWS.jsonl OUT_DIR  (rows from a parquet export)
"""
import argparse
import collections
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

MAX_CALLS = 40
MAX_CHARS = 5000  # swegemma HarnessLimits.max_stdout_chars
AUTHOR = "swe_noedit_agent"
HELD_OUT_REPOS = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}
DATASET = "nebius/SWE-rebench-openhands-trajectories"
FOOTER = re.compile(r"^\[(?:The command completed with exit code \d+\.|Current working directory: [^\]]*|"
                    r"Python interpreter: [^\]]*|Command finished with exit code \d+|"
                    r"Below is the output of the previous command\.)\]\s*$", re.M)
EXIT = re.compile(r"exit code (-?\d+)")


class Skip(Exception):
    """Trajectory can't be expressed faithfully in v7's tools."""


def ok(**kw):
    return json.dumps({"status": "ok", **kw}, ensure_ascii=False)


def cmd_result(stdout, exit_code, stderr=""):
    out, err = stdout[:MAX_CHARS], stderr[:MAX_CHARS]
    if exit_code != 0:
        return json.dumps({"status": "error", "error_type": "CommandError", "error_message": err or out,
                           "details": {"stdout": out, "stderr": err, "exit_code": exit_code}}, ensure_ascii=False)
    return ok(stdout=out, stderr=err, exit_code=0)


def pyquote(text):
    """Triple-quoted literal that reproduces `text` exactly inside a quoted heredoc."""
    if "PYEOF" in text:
        raise Skip("heredoc delimiter in text")
    raw = "\\" in text  # raw literal so backslashes in code survive; can't end with a backslash
    for q in ('"""', "'''"):
        if q in text or text.endswith(q[0]) or (raw and text.endswith("\\")):
            continue
        return ("r" if raw else "") + q + text + q
    raise Skip("unquotable text")


def replace_script(path, old, new):
    return (f"python3 - <<'PYEOF'\nimport pathlib\np = pathlib.Path(\"{path}\")\ns = p.read_text()\n"
            f"old = {pyquote(old)}\nnew = {pyquote(new)}\nassert s.count(old) == 1, s.count(old)\n"
            f"p.write_text(s.replace(old, new))\nPYEOF")


def insert_script(path, line, new):
    return (f"python3 - <<'PYEOF'\nimport pathlib\np = pathlib.Path(\"{path}\")\n"
            f"lines = p.read_text().splitlines(keepends=True)\nnew = {pyquote(new)}\n"
            f"lines.insert({int(line)}, new if new.endswith(\"\\n\") else new + \"\\n\")\n"
            f"p.write_text(\"\".join(lines))\nPYEOF")


class Template:
    """v7's task message, split around the parts that change per task."""

    def __init__(self, trace_path):
        steps = json.loads(Path(trace_path).read_text())["steps"]
        msg = next(s["message"] for s in steps if s.get("source") == "user")
        a = msg.index("Problem Statement:\n") + len("Problem Statement:\n")
        b = msg.index("\n\n## Task Budget")
        c = msg.index("```\n", msg.index("## Workspace Layout")) + 4
        d = msg.index("```", c)
        head = msg[:a]
        self.head = head[:head.index("repository ") + len("repository ")], head[head.index(".\n\nProblem Statement:"):]
        self.middle, self.tail = msg[b:c], msg[d:]

    def render(self, repo, problem, tree):
        return self.head[0] + repo + self.head[1] + problem.strip() + self.middle + "\n".join(tree) + "\n" + self.tail


def issue_text(user_msg):
    m = re.search(r"<issue_description>\n?(.*?)\n?</issue_description>", user_msg, re.S)
    if not m:
        raise Skip("no issue description")
    return m.group(1)


def tree_from_listing(listing, root):
    paths = []
    for line in listing.splitlines():
        line = line.strip()
        if not line.startswith(root + "/"):
            continue
        rel = line[len(root) + 1:].rstrip("/")
        if rel and not any(part.startswith(".") for part in rel.split("/")):
            paths.append("./" + rel)
    if not paths:
        raise Skip("empty root listing")
    return sorted(set(paths), key=str.lower)


def args_of(tc):
    a = tc["function"].get("arguments")
    return json.loads(a) if isinstance(a, str) else (a or {})


def short_reason(texts, limit=240):
    """The teacher's decision text before a call, shortened to one or two sentences (markdown headers and
    'Phase N' banners dropped). None if nothing is left."""
    lines = []
    for t in texts:
        for line in str(t or "").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or re.match(r"(\*\*)?phase \d", line, re.I) or line.startswith("```"):
                continue
            lines.append(re.sub(r"^[-*\d.)\s]+", "", line))
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if not text:
        return None
    sents = re.split(r"(?<=[.!?])\s+", text)
    out = sents[0]
    if len(out) < 80 and len(sents) > 1:
        out += " " + sents[1]
    return out[:limit].rstrip() + ("…" if len(out) > limit else "")


def convert(row, template, reasoning=False):
    """-> (ATIF dict, stats) or raises Skip. reasoning=True attaches the teacher's short decision text (think +
    prose since the previous kept call) to each kept call as the agent step's `message`."""
    tr = row["trajectory"]
    user = next(m for m in tr if m["role"] == "user")
    umsg = user["content"] if isinstance(user["content"], str) else json.dumps(user["content"])
    root = re.search(r"<uploaded_files>\s*(\S+)\s*</uploaded_files>", umsg)
    if not root:
        raise Skip("no workspace root")
    root = root.group(1).rstrip("/")

    def fix(text):
        text = text.replace(root + "/", "/workspace/").replace(root, "/workspace")
        return re.sub(r"/opt/conda/envs/testbed/bin/python3?", "python", text)

    def rel(path):
        p = fix(path)
        return p[len("/workspace/"):] if p.startswith("/workspace/") else p

    obs_by_id = {m.get("tool_call_id"): m for m in tr if m["role"] == "tool"}
    steps, tree, n_calls, final, pending = [], None, 0, None, []
    for m in tr:
        if m["role"] != "assistant":
            continue
        if isinstance(m.get("content"), str) and m["content"].strip():
            pending.append(m["content"])
        for tc in m.get("tool_calls") or []:
            name, a = tc["function"]["name"], args_of(tc)
            o = obs_by_id.get(tc.get("id"))
            obs = str((o or {}).get("content") or "")
            call = resp = None
            if name == "think":
                pending.append(a.get("thought", ""))
                continue
            if name == "task_tracker":
                continue
            if name == "finish":
                final = a.get("message") or "I have implemented and verified the fix, and submitted the patch."
                continue
            if name == "execute_bash":
                if a.get("is_input") in (True, "true"):
                    raise Skip("interactive input")
                if "timed out" in obs and "exit code" not in obs:
                    raise Skip("command timeout")
                codes = EXIT.findall(obs)
                out = FOOTER.sub("", obs).rstrip("\n")
                call = ("run_command", {"command": fix(a.get("command", ""))})
                resp = cmd_result(fix(out) + ("\n" if out else ""), int(codes[-1]) if codes else 0)
            elif name == "str_replace_editor":
                c, path = a.get("command"), a.get("path", "")
                if c == "view":
                    if fix(path).rstrip("/") == "/workspace" or path.rstrip("/") in (root, "/workspace"):
                        if tree is None and obs.startswith("Here's the files"):
                            tree = tree_from_listing(obs, root)
                            continue
                    if obs.startswith("Here's the files"):
                        body = "\n".join(l for l in obs.splitlines()[1:])
                        call = ("run_command", {"command": f"find {rel(path) or '.'} -maxdepth 2 -not -path '*/.*'"})
                        resp = cmd_result(fix(body) + "\n", 0)
                    elif obs.startswith("Here's the result of running `cat -n`"):
                        body = obs.split("\n", 1)[1] if "\n" in obs else ""
                        vr = a.get("view_range")
                        cmd = f"cat -n {rel(path)}"
                        if vr:
                            lo, hi = vr
                            cmd += f" | sed -n '{lo},{'$' if hi == -1 else hi}p'"
                        call, resp = ("run_command", {"command": cmd}), cmd_result(fix(body), 0)
                    else:
                        call = ("run_command", {"command": f"cat -n {rel(path)}"})
                        resp = cmd_result("", 1, f"cat: {rel(path)}: No such file or directory\n")
                elif c == "create":
                    text = a.get("file_text", "")
                    call = ("write_file", {"filepath": rel(path), "content": text})
                    resp = ok(filepath=rel(path), size=len(text.encode()))
                elif c == "str_replace":
                    call = ("run_command", {"command": replace_script(rel(path), a.get("old_str", ""),
                                                                      a.get("new_str", "") or "")})
                    if "has been edited" in obs:
                        resp = cmd_result("", 0)
                    else:
                        n = len(re.findall(r"line \d+", obs)) if "Multiple occurrences" in obs else 0
                        resp = cmd_result("", 1, "Traceback (most recent call last):\n  File \"<stdin>\", line 6, "
                                                 f"in <module>\nAssertionError: {n}\n")
                elif c == "insert":
                    call = ("run_command", {"command": insert_script(rel(path), a.get("insert_line", 0),
                                                                     a.get("new_str", ""))})
                    resp = cmd_result("", 0) if "has been edited" in obs else cmd_result(
                        "", 1, "Traceback (most recent call last):\n  File \"<stdin>\", line 4, in <module>\n"
                               "IndexError: list index out of range\n")
                else:
                    raise Skip(f"editor command {c}")
            else:
                raise Skip(f"tool {name}")
            n_calls += 1
            cid = f"call_{n_calls}"
            step = {"source": "agent", "tool_calls": [{"tool_call_id": cid, "function_name": call[0],
                                                       "arguments": call[1], "extra": {"author": AUTHOR}}]}
            reason = short_reason(pending) if reasoning else None
            if reason:
                step["message"] = fix(reason)
            pending = []
            steps.append(step)
            steps.append({"source": "system", "observation": {"results": [{"source_call_id": cid, "content": resp}],
                                                              "extra": {"author": AUTHOR, "tool_name": call[0]}}})
    if tree is None:
        raise Skip("no root listing")
    if final is None:
        raise Skip("no finish")
    n_calls += 1
    patch = row.get("model_patch") or ""
    steps.append({"source": "agent", "tool_calls": [{"tool_call_id": "call_submit", "function_name": "submit_patch",
                                                     "arguments": {}, "extra": {"author": AUTHOR}}]})
    steps.append({"source": "system", "observation": {"results": [{"source_call_id": "call_submit", "content": ok(
        patch_size=len(patch.encode()), files_changed=patch.count("\ndiff --git") + patch.startswith("diff --git"))}],
        "extra": {"author": AUTHOR, "tool_name": "submit_patch"}}})
    if n_calls > MAX_CALLS:
        raise Skip("over call budget")
    steps.append({"source": "agent", "message": fix(final)[:1500], "extra": {"author": AUTHOR}})
    task = template.render(row["repo"], fix(issue_text(umsg)), tree)
    atif = {"schema_version": "ATIF-v1.7", "agent": {"name": "convert_public", "source": DATASET},
            "source": {"trajectory_id": row.get("trajectory_id"), "instance_id": row["instance_id"]},
            "steps": [{"source": "user", "message": task}] + steps}
    return atif, n_calls


def hf_rows(n_target, seed=0, page=100, total=67074):
    """Random pages from the datasets-server API (sample only; full runs read the parquet export)."""
    rng = random.Random(seed)
    offsets = list(range(0, total - page, page))
    rng.shuffle(offsets)
    for off in offsets:
        url = (f"https://datasets-server.huggingface.co/rows?dataset={DATASET}&config=default&split=train"
               f"&offset={off}&length={page}")
        for attempt in range(6):  # the public API returns 5xx under load: back off, then skip the page
            try:
                page_rows = json.load(urllib.request.urlopen(url, timeout=180))["rows"]
                break
            except Exception as e:  # noqa: BLE001
                print(f"page {off}: {e}; retry {attempt + 1}", file=sys.stderr)
                time.sleep(10 * (attempt + 1))
        else:
            continue
        for r in page_rows:
            yield r["row"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--template", required=True, help="a v7 trace (ATIF) whose task message is the template")
    ap.add_argument("--hf-sample", type=int, default=None, help="convert until N kept, via datasets-server")
    ap.add_argument("--jsonl", default=None, help="rows as JSON lines")
    ap.add_argument("--one-per-task", action="store_true", help="keep at most one trajectory per instance_id")
    ap.add_argument("--reasoning", action="store_true", help="keep the teacher's short decision text before each call")
    ap.add_argument("--only-trajectories", default=None,
                    help="dir of converted traces: convert exactly those trajectory_ids (stop when all are found)")
    ap.add_argument("--save-rows", default=None, help="append the raw source rows of kept trajectories to this JSONL")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tpl = Template(a.template)
    only = None
    if a.only_trajectories:
        only = {json.loads(f.read_text())["source"]["trajectory_id"] for f in Path(a.only_trajectories).glob("trace_*.json")}
    rows = hf_rows(10 ** 9 if only else a.hf_sample) if (a.hf_sample or only) else (json.loads(l) for l in open(a.jsonl))
    why, kept, seen, calls, done = collections.Counter(), 0, 0, [], set()
    for f in out.glob("trace_*.json"):  # resume: traces already converted count as kept
        done.add(f.stem[len("trace_"):].rsplit("__", 1)[0])
    kept = len(done)
    for row in rows:
        seen += 1
        if not row.get("resolved") or "submit" not in str(row.get("exit_status")):
            why["unresolved/no submit"] += 1
            continue
        if row["repo"].lower() in HELD_OUT_REPOS:
            why["held-out repo"] += 1
            continue
        if only is not None and row.get("trajectory_id") not in only:
            why["not in --only-trajectories"] += 1
            continue
        if a.one_per_task and row["instance_id"] in done:
            why["task already kept"] += 1
            continue
        try:
            atif, n = convert(row, tpl, reasoning=a.reasoning)
        except Skip as e:
            why[str(e)] += 1
            continue
        (out / f"trace_{row['instance_id']}__{(row.get('trajectory_id') or str(seen))[:8]}.json").write_text(
            json.dumps(atif, ensure_ascii=False))
        kept += 1
        calls.append(n)
        done.add(row["instance_id"])
        if a.save_rows:
            with open(a.save_rows, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if only is not None and len(done) >= len(only):
            break
        if a.hf_sample and not only and kept >= a.hf_sample:
            break
    report = {"seen": seen, "kept": kept, "distinct_tasks": len(done), "skipped": dict(why.most_common()),
              "calls_median": sorted(calls)[len(calls) // 2] if calls else None}
    (out / "convert_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
