"""Mutation / test-verification metrics, independent of analysis/trace_analyzer.py (HEURISTIC).

Why: trace_analyzer.py only counts edit_file/write_file as edits. v7/v14/v16 edit through
run_command (python heredocs, sed -i, cat > file, ...), so its edit-chronology fields
(step_first_edit, tests_after_last_edit, last_test_ok, failure_stage "wrong_fix_unverified")
are wrong for those runs. This module re-derives them from the raw ATIF traces with a
separate, documented tagger, and keeps three things apart:
  1. ACTION CHRONOLOGY  - when source was mutated and when tests ran (from tool calls).
  2. TEST RETURN        - whether a test invocation *returned success* (from tool output),
                          as distinct from merely being invoked.
  3. FINAL-PATCH COVERAGE - what the submitted patch touches (from patches/<id>.patch);
                          reported in a separate block and never mixed with 1/2.
Every output is a heuristic regex tagger, not an execution-state oracle.

Usage:
  python analysis/mutation_test_metrics.py [--out DIR] [--show N] [INPUT ...]
  INPUT = eval_<name>.tgz (official_l4 eval output) or an extracted <name>/repN dir.
  With no INPUT, every /kaggle/input/**/eval_*.tgz is used (Kaggle CPU kernel).
Outputs: DIR/mutation_metrics_summary.json, DIR/mutation_metrics_per_task.jsonl.
Stdlib only.

----------------------------------------------------------------------------------------
HEURISTICS (all of them; change here and nowhere else)

Calls. Traces are flattened like trace_analyzer (steps, then subagent_trajectories); one
step's observation belongs to its last tool call. A call index k is the position of the
tool call in that flattened list.

Command splitting (run_command only).
  H0  Leaked chat-template tokens such as `<|"|>` (Gemma tool-call debris) are replaced by a
      space first, so they do not read as redirections.
  H1  Heredocs: a line containing `<<[-]['"]WORD['"]` starts a body that ends at the line
      equal (after strip) to WORD. The header line stays in the shell text; the body is
      attached to the header. A body is Python if the header runs `python[N.N] [-]` and is
      file content if the header redirects (`cat <<EOF > f`). bash/sh heredoc bodies are
      NOT re-parsed (documented miss).
  H2  Shell lines are split into segments on unquoted `;`, `&&`, `||`, `|`, `&` (but not the
      `&` of `2>&1`/`&>`). Each segment is tokenised with shlex (fallback: whitespace).
      Leading `VAR=val`, `env`, `time`, `timeout N`, `sudo`, `command` are skipped.
  H2b An unquoted `#` at the start of a word starts a shell comment that runs to the end of the
      line; comments are dropped (agents write prose like `# -> dict[str, ...]`).

Mutation targets (a "write" event: path, via, char position in the command).
  M1  edit_file / write_file: args filepath (or any arg key containing "path").
  M2  Shell redirection `>`, `>>`, `>|` to a target that is not /dev/*, an fd (`>&2`), or
      preceded by a digit (`2>`). Quoted text is blanked before matching.
  M3  `tee [-a] FILE`.
  M4  `sed` with `-i`/`-i.bak`/`--in-place`: file operands after the script
      (`-e`/`-f` scripts honoured). `perl` with `-i`/`-pi`: non-option operands after -e.
  M5  `mv SRC DST`, `cp SRC DST`, `install`: DST (last operand). `rm`/`unlink` operands are
      write events too (deleting a source file mutates source).
  M6  `patch`, `git apply`, `git am`, `git checkout -- F`/`git checkout F`, `git restore F`,
      `git stash`, `git reset --hard`: target = listed files after `--`, else UNRESOLVED.
  M7  Python code (heredoc body, `python -c CODE`, or an agent-written script later run as
      `python X.py`): `open(T, MODE)` / `io.open` / `codecs.open` with w/a/x/+ in MODE;
      `T.write_text(` / `T.write_bytes(` / `Path(T).write_text(`; `shutil.copy*|move(S, T)`;
      `os.replace|rename(S, T)`; `fileinput.input(..., inplace=True)`. T is resolved when it
      is a string literal or a variable assigned `V = Path("lit")` / `V = "lit.ext"` in the
      same code; otherwise the target is UNRESOLVED.
  M7b If any M7 target is UNRESOLVED (helper `def edit(path, ...)`, loop over a list of
      paths), the targets become every file-like string literal in that code (`"a/b.py"`,
      extensions py pyi md rst toml cfg ini txt json yaml), excluding /tmp. Read-only paths
      named in the same code would be over-counted; the via tag is `py:fallback-literal`.
  M8  A write is APPLIED if the call did not return status "error" and its exit_code is 0
      or unrecorded; or, for a call that also ran a test whose summary was found in the
      output (so a `&&` chain reached the test), regardless of the final exit code.
      Attempted-but-failed writes (e.g. the template's `assert s.count(old) == 1` failing)
      are counted separately and never enter the chronology.

Target classes (path normalised: strip /workspace/, /testbed/, /repo/, ./).
  C1  tmp       : /tmp/*, /dev/*, /proc/*, /var/tmp/*.
  C2  scratch   : basename matches SCRATCH_RE (repro*, reproduce*, debug*, verify*, check_*,
                  scratch*, temp*/tmp*, out/output*, fix_*/patch_*, test_issue*/test_repro*/
                  test_fix*/test_bug*/test_my*, issue*, try*, run_*) or extension in
                  SCRATCH_EXT (.log .out .patch .diff .orig .bak .rej .pyc), or a path the
                  agent later rm'd that is absent from the final patch (post-hoc rule).
  C3  test      : under tests?/ or testing/, or basename test_*.py / *_test.py / conftest.py.
  C4  source    : everything else with a repo-relative path.
  C5  unresolved: target could not be resolved (M6/M7).
  SOURCE MUTATION (primary) = applied write in C4. Sensitivity variant adds C5.

Test invocations.
  T1  Segment command word: `pytest`/`py.test` (any path), `python -m pytest|py.test|unittest`,
      `tox`, `nox`, or `python FILE.py` where FILE is a test file by C3 *and was not written
      by the agent earlier in the trajectory* (agent-written scripts are repro, not tests).
  T2  In Python code: `pytest.main(`, `unittest.main(`, or a subprocess call whose literal
      args contain "pytest".
  Things like `grep pytest`, `cat tests/x.py`, `find tests` are not invocations.

Test return (from observation stdout+stderr, status, exit_code).
  R1  pytest summary lines: the last line matching `N passed|failed|error(s)|skipped|...`
      together with `in N.NNs` (or `-q` style "N passed"), plus "no tests ran",
      "Interrupted: N error". unittest: `Ran N tests` then `OK` / `FAILED (`.
  R2  PASS  = a summary with passed > 0 and failed == errors == 0, no failing summary
              anywhere in the output, and (exit_code == 0, or exit code unrecorded, or the
              command contains a pipe/`;`/`||` that can mask the test's exit code).
      FAIL  = a failing summary, or no-tests-ran, or non-zero unmasked exit code with a
              summary, or a timeout / status error.
      UNKNOWN = no recognisable summary (e.g. output filtered or truncated at 5,000 chars).
      Only PASS counts as "successful returned verification". Invocation alone never does.

Chronology (per task, primary source mutations only).
  first_mut / last_mut = call index of first/last applied source mutation.
  test_after_last_mut  = a test invocation at a later call, or later in the same call
                         (char position after the last write in that call).
  pass_after_last_mut  = such a test returned PASS (R2).
  X1  exec_after_last_mut / exec_ok_after_last_mut: any code execution after the last
      mutation (a test, `python X.py`, `python -c`, or a python heredoc that wrote nothing),
      and whether one returned status ok with exit 0. This is "ran something", e.g. the
      agent's own repro script; it is NOT test verification and exit 0 is not correctness.

Final-patch coverage (separate block). From patches/<id>.patch and the gold file list in
analysis/per_task.jsonl (gold_files is computed from the gold patch only, not from
chronology): patch non-empty; touches >= 1 non-test gold .py file; touches all of them.
Tagger recall check: share of non-test files in the final patch that the tagger saw a
source/unresolved write to (suffix match).
"""

import argparse
import collections
import glob
import json
import os
import re
import shlex
import statistics
import sys
import tarfile
import tempfile

# ------------------------------------------------------------------ regexes
HEREDOC = re.compile(r"(?<!<)<<(?!<)-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
PY_HEADER = re.compile(r"\bpython[\d.]*\s+(?:-[A-Za-z]+\s+)*(?:-\s*)?<<")
QUOTED = re.compile(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"")
REDIRECT = re.compile(r"(?<![0-9&>])(?:>>|>\||>)\s*([^\s;&|<>()]+)")
SCRATCH_RE = re.compile(
    r"^(repro|reproduce|debug|verify|check_|scratch|temp|tmp|out\b|output|fix_|patch_|test_issue|test_repro"
    r"|test_fix|test_bug|test_my|issue|try|run_|my_)", re.I)
SCRATCH_EXT = {".log", ".out", ".patch", ".diff", ".orig", ".bak", ".rej", ".pyc"}
TEST_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)(test_[^/]*\.py|[^/]*_test\.py|conftest\.py)$")
PREFIX_WORDS = {"env", "time", "sudo", "command", "nice", "exec"}

# python write idioms (M7)
STRLIT = r"(?:[rRbBuUfF]{0,2})(?:\"\"\"(.+?)\"\"\"|'''(.+?)'''|\"([^\"\n]*)\"|'([^'\n]*)')"
ASSIGN_PATH = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*(?:pathlib\.)?(?:Pure)?Path\(\s*" + STRLIT + r"\s*\)", re.S)
ASSIGN_STR = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*" + STRLIT, re.S)
OPEN_CALL = re.compile(r"\b(?:io\.|codecs\.)?open\(\s*([^,()]+?)\s*,\s*(?:mode\s*=\s*)?" + STRLIT, re.S)
WRITE_TEXT = re.compile(r"(\b[A-Za-z_]\w*|(?:pathlib\.)?Path\(\s*[^()]*?\))\s*\.\s*write_(?:text|bytes)\(", re.S)
SHUTIL = re.compile(r"\b(?:shutil\.(?:copy2?|copyfile|move)|os\.(?:replace|rename))\(\s*[^,]+,\s*([^,()]+?)\s*[,)]")
FILEINPUT = re.compile(r"fileinput\.input\(([^)]*inplace\s*=\s*(?:True|1)[^)]*)\)")
FILE_LIT = re.compile(r"""["']((?:/workspace/)?[\w.\-]+(?:/[\w.\-]+)*\.(?:py|pyi|md|rst|toml|cfg|ini|txt|json|ya?ml))["']""")
DEBRIS = re.compile(r"<\|[^|<>\n]{0,20}\|>")  # leaked chat-template tokens such as <|"|> (H0)
PY_TEST_CALL = re.compile(r"\bpytest\.main\(|\bunittest\.main\(|subprocess\.[A-Za-z_]+\([^)]*['\"](?:-m\s*['\"]?\s*)?pytest")

# test summaries (R1)
PYTEST_SUMMARY = re.compile(
    r"^[=\-\s]*((?:\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?|rerun)(?:, )?)+)"
    r"(?:\s+in\s+[\d.]+\s*s.*)?\.?\s*=*\s*$", re.M)
PYTEST_NOTESTS = re.compile(r"\bno tests ran\b|\bInterrupted: \d+ errors?\b|ERROR: (?:file|module) or directory not found"
                            r"|ERROR: not found:|\bcollected 0 items\b", re.I)
UNITTEST_RAN = re.compile(r"^Ran (\d+) tests? in [\d.]+s", re.M)
UNITTEST_OK = re.compile(r"^OK\b", re.M)
UNITTEST_FAIL = re.compile(r"^FAILED \(", re.M)


# ------------------------------------------------------------------ trace helpers
def deep_json(x, depth=3):
    for _ in range(depth):
        if isinstance(x, str):
            try:
                x = json.loads(x)
                continue
            except Exception:  # noqa: BLE001
                return x
        if isinstance(x, dict) and len(x) == 1 and isinstance(next(iter(x.values())), str):
            try:
                x = json.loads(next(iter(x.values())))
                continue
            except Exception:  # noqa: BLE001
                return x
        return x
    return x


def flatten_steps(traj):
    steps = list(traj.get("steps", []))
    for sub in traj.get("subagent_trajectories", []) or []:
        steps += flatten_steps(sub)
    return steps


def calls_of(trace):
    out = []
    for s in flatten_steps(trace):
        tcs = s.get("tool_calls") or []
        obs = None
        if s.get("observation"):
            c = (s["observation"] or {}).get("content", "")
            obs = deep_json(c if isinstance(c, str) else json.dumps(c))
        for i, tc in enumerate(tcs):
            out.append({"name": tc.get("function_name"), "args": tc.get("arguments") or {},
                        "obs": obs if i == len(tcs) - 1 else None})
    return out


def norm_path(p):
    p = (p or "").strip().strip("'\"")
    for pre in ("/workspace/", "/testbed/", "/repo/", "workspace/", "./"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p


def classify(path):
    if not path or path == "?":
        return "unresolved"
    if re.match(r"^/(tmp|dev|proc|var/tmp)(/|$)", path):
        return "tmp"
    if path.startswith("/"):
        p = path.lstrip("/")  # other absolute path: judge by name
    else:
        p = path
    base = os.path.basename(p)
    ext = os.path.splitext(base)[1].lower()
    if SCRATCH_RE.match(base) or ext in SCRATCH_EXT:
        return "scratch"
    if TEST_PATH.search(p):
        return "test"
    return "source"


# ------------------------------------------------------------------ shell parsing
def logical_line_end(cmd, start):
    """-> (end, code_end): first newline outside quotes, and where an unquoted `#` comment starts (H2b)."""
    q, i = None, start
    while i < len(cmd):
        ch = cmd[i]
        if q:
            if ch == "\\" and q == '"':
                i += 2
                continue
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch == "\\":
            i += 2
            continue
        elif ch == "#" and (i == start or cmd[i - 1] in " \t;&|("):
            nl = cmd.find("\n", i)
            nl = len(cmd) if nl < 0 else nl
            return nl, i
        elif ch == "\n":
            return i, i
        i += 1
    return len(cmd), len(cmd)


def split_heredocs(cmd):
    """-> list of (offset, logical_line, body_or_None); a body is attached to its header line.
    Newlines inside quotes (multi-line `python -c "..."`) do not end a line (H1)."""
    out, off = [], 0
    while off < len(cmd):
        end, code_end = logical_line_end(cmd, off)
        line = cmd[off:code_end]
        line_off, off = off, end + 1
        blank = QUOTED.sub(lambda q: "_" * len(q.group(0)), line)
        m = next((h for h in HEREDOC.finditer(line) if blank[h.start()] == "<"), None)  # marker outside quotes
        if not m:
            out.append((line_off, line, None))
            continue
        delim, body = m.group(2), []
        while off < len(cmd):
            nl = cmd.find("\n", off)
            nl = len(cmd) if nl < 0 else nl
            raw = cmd[off:nl]
            off = nl + 1
            if raw.strip() == delim:
                break
            body.append(raw)
        out.append((line_off, line, "\n".join(body)))
    return out


def split_segments(line):
    """Split a shell line on unquoted ; && || | & -> list of (offset, text, sep_before)."""
    segs, cur, start, q, i = [], [], 0, None, 0
    while i < len(line):
        ch = line[i]
        if q:
            if ch == "\\" and q == '"' and i + 1 < len(line):
                cur.append(line[i:i + 2]); i += 2; continue  # noqa: E702
            if ch == q:
                q = None
            cur.append(ch); i += 1; continue  # noqa: E702
        if ch in "'\"":
            q = ch; cur.append(ch); i += 1; continue  # noqa: E702
        if ch == "\\" and i + 1 < len(line):
            cur.append(line[i:i + 2]); i += 2; continue  # noqa: E702
        if ch in ";|&":
            prev = line[i - 1] if i else ""
            nxt = line[i + 1] if i + 1 < len(line) else ""
            if ch == "&" and (prev == ">" or nxt == ">"):
                cur.append(ch); i += 1; continue  # noqa: E702
            width = 2 if (ch in "|&" and nxt == ch) else 1
            segs.append((start, "".join(cur)))
            i += width
            start, cur = i, []
            continue
        cur.append(ch)
        i += 1
    segs.append((start, "".join(cur)))
    return [(o, t) for o, t in segs if t.strip()]


def tokens(seg):
    try:
        toks = shlex.split(seg, comments=False, posix=True)
    except ValueError:
        toks = seg.split()
    # drop redirections and their targets
    out, skip = [], False
    for t in toks:
        if skip:
            skip = False
            continue
        if re.fullmatch(r"\d*(>>|>\||>|<<<|<<-?|<)&?\d*", t):
            skip = not re.search(r"&\d*$", t)
            continue
        if re.match(r"^\d*(>>|>|<)", t):
            continue
        out.append(t)
    while out and (re.fullmatch(r"[A-Za-z_]\w*=.*", out[0]) or out[0] in PREFIX_WORDS):
        out = out[1:]
    if out and out[0] == "timeout":
        out = out[1:]
        while out and (out[0].startswith("-") or re.fullmatch(r"[\d.]+[smhd]?", out[0])):
            out = out[1:]
    return out


def operands(toks, takes_value=()):
    ops, skip = [], False
    for t in toks[1:]:
        if skip:
            skip = False
            continue
        if t in takes_value:
            skip = True
            continue
        if t.startswith("-") and t != "-":
            continue
        ops.append(t)
    return ops


def is_python(word):
    return bool(re.fullmatch(r"(?:.*/)?python[\d.]*", word or ""))


# ------------------------------------------------------------------ python code writes (M7)
def lit(m, base=2):
    for g in m.groups()[base - 1:base + 3]:
        if g is not None:
            return g
    return None


def py_writes(code):
    """-> (list of (target_path_or_'?', idiom), has_test_call)."""
    env = {}
    for m in ASSIGN_PATH.finditer(code):
        env[m.group(1)] = lit(m)
    for m in ASSIGN_STR.finditer(code):
        v = lit(m)
        if v and m.group(1) not in env and re.search(r"[/.]\w{1,5}$|/", v) and "\n" not in v and len(v) < 200:
            env[m.group(1)] = v

    def resolve(expr):
        expr = (expr or "").strip()
        mm = re.fullmatch(r"(?:pathlib\.)?(?:Pure)?Path\(\s*(.*?)\s*\)", expr, re.S)
        if mm:
            expr = mm.group(1).strip()
        mm = re.fullmatch(r"[rRbBuUfF]{0,2}(['\"])(.*)\1", expr, re.S)
        if mm:
            return mm.group(2)
        if re.fullmatch(r"[A-Za-z_]\w*", expr):
            return env.get(expr, "?")
        return "?"

    writes = []
    for m in OPEN_CALL.finditer(code):
        mode = lit(m) or ""
        if re.search(r"[wax+]", mode):
            writes.append((resolve(m.group(1)), "py:open"))
    for m in WRITE_TEXT.finditer(code):
        writes.append((resolve(m.group(1)), "py:write_text"))
    for m in SHUTIL.finditer(code):
        writes.append((resolve(m.group(1)), "py:shutil/os"))
    for m in FILEINPUT.finditer(code):
        first = m.group(1).split(",")[0]
        writes.append((resolve(first), "py:fileinput"))
    if any(p == "?" for p, _ in writes):
        # M7b: unresolved target (helper `def edit(path, ...)`, loop over a list): fall back to the
        # file-like string literals in the same code, excluding /tmp.
        lits = sorted({x for x in FILE_LIT.findall(code) if not x.startswith("/tmp")})
        if lits:
            writes = [w for w in writes if w[0] != "?"] + [(x, "py:fallback-literal") for x in lits]
    return writes, bool(PY_TEST_CALL.search(code))


# ------------------------------------------------------------------ per-call tagging
def tag_command(cmd, scripts):
    """-> dict(writes=[(pos, path, via)], tests=[pos], created=[(path, content)], removed=[path],
    execs=[pos]) where execs are code executions (python/pytest), used only for the exec metric."""
    writes, tests, created, removed, execs = [], [], [], [], []
    cmd = DEBRIS.sub(" ", cmd)
    for line_off, line, body in split_heredocs(cmd):
        blank_line = QUOTED.sub(lambda q: "_" * len(q.group(0)), line)
        py_body = body is not None and bool(PY_HEADER.search(blank_line))
        for seg_off, seg in split_segments(line):
            pos = line_off + seg_off
            blank = QUOTED.sub(lambda q: "_" * len(q.group(0)), seg)
            # quoted targets are blanked; the raw text is read back at the same span
            for m in REDIRECT.finditer(blank):
                raw = seg[m.start(1):m.end(1)]
                if raw.startswith("&") or raw.startswith("/dev/"):
                    continue
                path = norm_path(raw)
                writes.append((pos + m.start(), path, "sh:redirect"))
                if body is not None and not py_body and "<<" in blank:
                    created.append((path, body))
            toks = tokens(seg)
            if not toks:
                continue
            w = os.path.basename(toks[0])
            if w == "tee":
                for t in operands(toks):
                    writes.append((pos, norm_path(t), "sh:tee"))
            elif w in ("sed", "gsed") and any(t == "--in-place" or re.fullmatch(r"-[A-Za-z]*i\S*", t) for t in toks[1:]):
                has_e = any(t in ("-e", "-f", "--expression") for t in toks)
                ops = operands(toks, takes_value=("-e", "-f", "--expression"))
                files = ops if has_e else ops[1:]
                for t in files:
                    writes.append((pos, norm_path(t), "sh:sed-i"))
            elif w == "perl" and any(re.fullmatch(r"-[A-Za-z]*i\S*", t) for t in toks[1:]):
                for t in operands(toks, takes_value=("-e", "-E")):
                    writes.append((pos, norm_path(t), "sh:perl-i"))
            elif w in ("mv", "cp", "install"):
                ops = operands(toks, takes_value=("-t", "-m", "-o", "-g"))
                if len(ops) >= 2:
                    writes.append((pos, norm_path(ops[-1]), "sh:" + w))
                    if w == "mv":
                        removed.append(norm_path(ops[0]))
            elif w in ("rm", "unlink"):
                for t in operands(toks):
                    writes.append((pos, norm_path(t), "sh:rm"))
                    removed.append(norm_path(t))
            elif w == "patch":
                writes.append((pos, "?", "sh:patch"))
            elif w == "git" and len(toks) > 1 and toks[1] in ("apply", "am", "checkout", "restore", "stash", "reset"):
                sub = toks[1]
                if sub == "reset" and "--hard" not in toks:
                    pass
                elif sub in ("checkout", "restore") and "--" in toks:
                    for t in toks[toks.index("--") + 1:]:
                        writes.append((pos, norm_path(t), "sh:git-" + sub))
                elif sub in ("checkout", "restore"):
                    ops = [t for t in operands(toks)[1:] if "." in t or "/" in t]
                    for t in ops:
                        writes.append((pos, norm_path(t), "sh:git-" + sub))
                else:
                    writes.append((pos, "?", "sh:git-" + sub))
            # tests (T1)
            if w in ("pytest", "py.test", "tox", "nox"):
                tests.append(pos)
                execs.append(pos)
            elif is_python(toks[0]) and not (py_body and "<<" in blank):
                rest = toks[1:]
                n_before = len(writes)
                if "-m" in rest and rest.index("-m") + 1 < len(rest) and rest[rest.index("-m") + 1] in ("pytest", "py.test", "unittest"):
                    tests.append(pos)
                elif "-c" in rest and rest.index("-c") + 1 < len(rest):
                    ws, t = py_writes(rest[rest.index("-c") + 1])
                    writes += [(pos, norm_path(p) if p != "?" else "?", via) for p, via in ws]
                    if t:
                        tests.append(pos)
                else:
                    script = next((x for x in rest if not x.startswith("-")), None)
                    if script and script.endswith(".py"):
                        sp = norm_path(script)
                        if sp in scripts:  # agent-written script: scan for writes, it is not a test suite
                            ws, t = py_writes(scripts[sp])
                            writes += [(pos, norm_path(p) if p != "?" else "?", "script:" + via) for p, via in ws]
                            if t:
                                tests.append(pos)
                        elif TEST_PATH.search(sp):
                            tests.append(pos)
                if len(writes) == n_before and rest:
                    execs.append(pos)  # X1: python run that wrote nothing
        if py_body:
            ws, t = py_writes(body)
            writes += [(line_off, norm_path(p) if p != "?" else "?", via) for p, via in ws]
            if t:
                tests.append(line_off)
            if not ws:
                execs.append(line_off)  # X1: python heredoc that wrote nothing
    return {"writes": writes, "tests": tests, "created": created, "removed": removed, "execs": execs}


def obs_fields(o):
    if not isinstance(o, dict):
        return None, None, "", None
    status = o.get("status")
    ec = o.get("exit_code")
    if ec is None and isinstance(o.get("details"), dict):
        ec = o["details"].get("exit_code")
    text = "\n".join(str(o.get(k) or "") for k in ("stdout", "stderr", "output", "error_message"))
    return status, ec, text, o.get("error_type")


def test_outcome(cmd, o):
    status, ec, text, etype = obs_fields(o)
    if etype == "TimeoutExceeded" or "timed out" in (text or "").lower()[:300] and status == "error":
        return "FAIL", True
    summaries = [m.group(1) for m in PYTEST_SUMMARY.finditer(text)]
    found = bool(summaries)
    bad = any(re.search(r"\d+ (failed|errors?)\b", s) for s in summaries)
    good = any(re.search(r"\d+ passed\b", s) for s in summaries)
    if PYTEST_NOTESTS.search(text):
        bad, found = True, True
    for m in UNITTEST_RAN.finditer(text):
        found = True
        tail = text[m.end():m.end() + 400]
        if UNITTEST_FAIL.search(tail) or int(m.group(1)) == 0:
            bad = True
        elif UNITTEST_OK.search(tail):
            good = True
    if not found:
        if status == "error" or (ec not in (None, 0)):
            return "FAIL", False  # non-zero exit with no summary: crashed/collection error (no summary found)
        return "UNKNOWN", False
    masked = bool(re.search(r"\||;|\|\|", QUOTED.sub("", cmd)))
    if bad:
        return "FAIL", True
    if good and (ec in (0, None) or masked):
        return "PASS", True
    return "FAIL", True


# ------------------------------------------------------------------ per-task
def analyze_task(trace, patch_text):
    calls = calls_of(trace)
    scripts, events, tags = {}, [], []
    attempted_src = 0
    removed_all = []
    for k, c in enumerate(calls):
        name, a, o = c["name"], c["args"], c["obs"]
        status, ec, _, _ = obs_fields(o)
        call_tags = []
        if name in ("edit_file", "write_file"):
            path = a.get("filepath") or next((v for kk, v in a.items() if "path" in str(kk).lower() and isinstance(v, str)), "?")
            path = norm_path(path) or "?"
            applied = status != "error"
            events.append({"k": k, "pos": 0, "kind": "write", "path": path, "via": name, "applied": applied})
            if name == "write_file" and isinstance(a.get("content"), str):
                scripts[path] = a["content"]
            call_tags.append(f"W:{name}")
        elif name == "run_command":
            cmd = str(a.get("command", ""))
            t = tag_command(cmd, scripts)
            test_ret = None
            if t["tests"]:
                outcome, has_summary = test_outcome(cmd, o)
                test_ret = outcome
                for p in t["tests"]:
                    events.append({"k": k, "pos": p, "kind": "test", "outcome": outcome})
            else:
                has_summary = False
            applied = (status != "error" and ec in (0, None)) or (bool(t["tests"]) and has_summary)
            for p, path, via in t["writes"]:
                events.append({"k": k, "pos": p, "kind": "write", "path": path, "via": via, "applied": applied})
            exec_ok = status != "error" and ec in (0, None)
            for p in t["execs"]:
                events.append({"k": k, "pos": p, "kind": "exec", "ok": exec_ok})
            for path, content in t["created"]:
                scripts[path] = content
            removed_all += [(k, r) for r in t["removed"]]
            if t["writes"]:
                call_tags.append("W:" + ",".join(sorted({v for _, _, v in t["writes"]})))
            if t["tests"]:
                call_tags.append("T:" + str(test_ret))
            elif t["execs"]:
                call_tags.append("X:" + ("ok" if exec_ok else "err"))
        tags.append(call_tags)

    pfiles = re.findall(r"^\+\+\+ b/(\S+)", patch_text or "", flags=re.M)
    removed_paths = {r for _, r in removed_all}
    for e in events:
        if e["kind"] != "write":
            continue
        cls = classify(e["path"])
        if cls == "source" and e["path"] in removed_paths and not any(pf.endswith(e["path"]) or e["path"].endswith(pf) for pf in pfiles):
            cls = "scratch"  # created and later removed by the agent, absent from final patch
        if e["via"] == "sh:rm" and cls == "source" and not any(pf.endswith(e["path"]) or e["path"].endswith(pf) for pf in pfiles):
            cls = "scratch"  # rm of a file not in the final patch: cleaning up, not a source change
        e["cls"] = cls
        if cls == "source" and not e["applied"]:
            attempted_src += 1

    def chron(classes):
        muts = [e for e in events if e["kind"] == "write" and e["applied"] and e["cls"] in classes]
        mut_calls = sorted({e["k"] for e in muts})
        if not muts:
            return {"n_mut_calls": 0, "first_mut": None, "last_mut": None, "test_after_last_mut": None,
                    "pass_after_last_mut": None, "last_test_after_outcome": None, "mut_paths": [],
                    "exec_after_last_mut": None, "exec_ok_after_last_mut": None}
        last_k = max(e["k"] for e in muts)
        last_pos = max(e["pos"] for e in muts if e["k"] == last_k)
        later = [e for e in events if e["k"] > last_k or (e["k"] == last_k and e["pos"] > last_pos)]
        after = [e for e in later if e["kind"] == "test"]
        ex = [e for e in later if e["kind"] in ("exec", "test")]
        return {"n_mut_calls": len(mut_calls), "first_mut": mut_calls[0], "last_mut": last_k,
                "test_after_last_mut": bool(after), "pass_after_last_mut": any(e["outcome"] == "PASS" for e in after),
                "last_test_after_outcome": after[-1]["outcome"] if after else None,
                "mut_paths": sorted({e["path"] for e in muts}),
                "exec_after_last_mut": bool(ex),
                "exec_ok_after_last_mut": any(e.get("ok") or e.get("outcome") == "PASS" for e in ex)}

    tests = [e for e in events if e["kind"] == "test"]
    test_calls = sorted({e["k"] for e in tests})
    outcome_by_call = {e["k"]: e["outcome"] for e in tests}
    prim = chron({"source"})
    sens = chron({"source", "unresolved"})
    seen = set(prim["mut_paths"]) | set(sens["mut_paths"])
    nontest_pf = [p for p in pfiles if classify(p) == "source"]
    covered = [p for p in nontest_pf if any(p.endswith(s) or s.endswith(p) for s in seen if s != "?")]
    return {
        "n_calls": len(calls),
        "n_test_calls": len(test_calls),
        "n_test_pass_calls": sum(outcome_by_call[k] == "PASS" for k in test_calls),
        "test_outcomes": dict(collections.Counter(outcome_by_call[k] for k in test_calls)),
        "n_src_write_attempted_not_applied": attempted_src,
        "n_unresolved_writes": sum(1 for e in events if e["kind"] == "write" and e["cls"] == "unresolved"),
        "write_via": dict(collections.Counter(e["via"] for e in events if e["kind"] == "write" and e["cls"] == "source")),
        "primary": prim, "sensitivity_incl_unresolved": sens,
        "patch_files": pfiles, "patch_src_files": nontest_pf, "patch_src_files_seen_written": covered,
        "_tags": tags, "_calls": calls,
    }


# ------------------------------------------------------------------ loading
def find_runs(inputs, workdir):
    """-> list of (run_name, rep_dir)."""
    runs = []
    for inp in inputs:
        if inp.endswith(".tgz") or inp.endswith(".tar.gz"):
            dest = os.path.join(workdir, os.path.basename(inp).split(".")[0])
            os.makedirs(dest, exist_ok=True)
            with tarfile.open(inp) as tf:
                keep = [m for m in tf.getmembers() if re.search(
                    r"rep\d+/(merged/(traces/|patches/|task_results\.jsonl)|analysis/per_task\.jsonl)", m.name)
                    and not m.name.startswith("/") and ".." not in m.name]
                try:
                    tf.extractall(dest, members=keep, filter="data")
                except TypeError:  # Python < 3.12
                    tf.extractall(dest, members=keep)
            inp = dest
        for rep in sorted(glob.glob(os.path.join(inp, "**", "rep*"), recursive=True)):
            if os.path.isdir(os.path.join(rep, "merged", "traces")):
                runs.append((os.path.basename(os.path.dirname(rep)), rep))
    return runs


def load_jsonl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(x) for x in open(p) if x.strip()]


def pct(n, d):
    return round(100.0 * n / d, 1) if d else None


def med(xs):
    return statistics.median(xs) if xs else None


def group_metrics(rows, key):
    n = len(rows)
    ch = [r[key] for r in rows]
    with_mut = [c for c in ch if c["n_mut_calls"] > 0]
    return {
        "n": n,
        "pct_ge1_mutation": pct(len(with_mut), n),
        "pct_test_after_last_mut": pct(sum(bool(c["test_after_last_mut"]) for c in ch), n),
        "pct_pass_after_last_mut": pct(sum(bool(c["pass_after_last_mut"]) for c in ch), n),
        "pct_test_after_last_mut_given_mut": pct(sum(bool(c["test_after_last_mut"]) for c in with_mut), len(with_mut)),
        "pct_pass_after_last_mut_given_mut": pct(sum(bool(c["pass_after_last_mut"]) for c in with_mut), len(with_mut)),
        "pct_exec_after_last_mut": pct(sum(bool(c["exec_after_last_mut"]) for c in ch), n),
        "pct_exec_ok_after_last_mut": pct(sum(bool(c["exec_ok_after_last_mut"]) for c in ch), n),
        "median_mut_calls": med([c["n_mut_calls"] for c in ch]),
        "median_mut_calls_given_mut": med([c["n_mut_calls"] for c in with_mut]),
        "last_test_after_outcome": dict(collections.Counter(str(c["last_test_after_outcome"]) for c in with_mut)),
    }


def other_metrics(rows):
    n = len(rows)
    return {
        "pct_any_test_invoked": pct(sum(r["n_test_calls"] > 0 for r in rows), n),
        "pct_any_test_pass": pct(sum(r["n_test_pass_calls"] > 0 for r in rows), n),
        "median_test_calls": med([r["n_test_calls"] for r in rows]),
        "test_call_outcomes": dict(sum((collections.Counter(r["test_outcomes"]) for r in rows), collections.Counter())),
        "src_writes_attempted_not_applied": sum(r["n_src_write_attempted_not_applied"] for r in rows),
        "unresolved_writes": sum(r["n_unresolved_writes"] for r in rows),
        "src_write_via": dict(sum((collections.Counter(r["write_via"]) for r in rows), collections.Counter())),
    }


def old_tagger(rows_old):
    """trace_analyzer per_task.jsonl fields, for comparison only (known-broken for run_command edits)."""
    n = len(rows_old)
    return {
        "n": n,
        "pct_first_edit_found": pct(sum(r.get("step_first_edit") is not None for r in rows_old), n),
        "pct_tests_after_last_edit": pct(sum((r.get("tests_after_last_edit") or 0) > 0 and r.get("step_first_edit") is not None
                                             for r in rows_old), n),
        "pct_last_test_ok": pct(sum(r.get("last_test_ok") is True for r in rows_old), n),
        "n_wrong_fix_unverified": sum(r.get("failure_stage") == "fix:wrong_fix_unverified" for r in rows_old),
        "failure_stage": dict(collections.Counter(r.get("failure_stage") for r in rows_old)),
    }


def final_patch(rows):
    n = len(rows)
    ne = [r for r in rows if r["patch_files"]]
    g = [r for r in ne if r["gold_py"]]
    return {
        "n": n, "pct_patch_nonempty": pct(len(ne), n),
        "n_nonempty_with_gold_py": len(g),
        "pct_touch_any_gold_py": pct(sum(bool(set(r["patch_files"]) & set(r["gold_py"])) for r in g), len(g)),
        "pct_touch_all_gold_py": pct(sum(set(r["gold_py"]) <= set(r["patch_files"]) for r in g), len(g)),
    }


def compact_tags(res, limit=40):
    lines = []
    for k, (c, tg) in enumerate(zip(res["_calls"], res["_tags"])):
        if k >= limit:
            lines.append(f"    ... {len(res['_calls']) - limit} more calls")
            break
        if c["name"] == "run_command":
            head = re.sub(r"\s+", " ", str(c["args"].get("command", "")))[:70]
        else:
            a = c["args"]
            head = str(a.get("filepath") or "")[:70]
        lines.append(f"    {k:3d} {c['name'][:12]:12s} {' '.join(tg) or '-':28s} | {head}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--out", default="/kaggle/working" if os.path.isdir("/kaggle/working") else "mutation_metrics_out")
    ap.add_argument("--show", type=int, default=5, help="tasks to print compact per-call tags for (validation)")
    args = ap.parse_args()
    inputs = args.inputs or sorted(glob.glob("/kaggle/input/**/eval_*.tgz", recursive=True))
    print("inputs:", inputs)
    os.makedirs(args.out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="mm_")
    runs = find_runs(inputs, work)
    seen_runs, summary, per_task_out, shown = set(), {}, [], []
    for run, rep in runs:
        key = f"{run}/{os.path.basename(rep)}"
        if key in seen_runs:
            continue
        seen_runs.add(key)
        merged = os.path.join(rep, "merged")
        results = {r["instance_id"]: r for r in load_jsonl(os.path.join(merged, "task_results.jsonl"))}
        old = {r["instance_id"]: r for r in load_jsonl(os.path.join(rep, "analysis", "per_task.jsonl"))}
        rows = []
        for tp in sorted(glob.glob(os.path.join(merged, "traces", "trace_*.json"))):
            iid = os.path.basename(tp)[len("trace_"):-len(".json")]
            if iid not in results:
                continue
            pp = os.path.join(merged, "patches", iid + ".patch")
            patch = open(pp, errors="replace").read() if os.path.exists(pp) else ""
            res = analyze_task(json.load(open(tp)), patch)
            gold = old.get(iid, {}).get("gold_files") or []
            res.update({"run": key, "instance_id": iid, "resolved": bool(results[iid].get("resolved")),
                        "gold_py": [f for f in gold if f.endswith(".py") and classify(f) == "source"]})
            rows.append(res)
        if not rows:
            continue
        s = {"n_tasks": len(rows), "n_resolved": sum(r["resolved"] for r in rows), "chronology": {}, "chronology_sensitivity_incl_unresolved": {},
             "test_activity": {}, "final_patch_coverage": {}, "old_tagger": {}}
        for lab, sel in (("all", lambda r: True), ("resolved", lambda r: r["resolved"]), ("unresolved", lambda r: not r["resolved"])):
            g = [r for r in rows if sel(r)]
            s["chronology"][lab] = group_metrics(g, "primary")
            s["chronology_sensitivity_incl_unresolved"][lab] = group_metrics(g, "sensitivity_incl_unresolved")
            s["test_activity"][lab] = other_metrics(g)
            s["final_patch_coverage"][lab] = final_patch(g)
            s["old_tagger"][lab] = old_tagger([old[r["instance_id"]] for r in g if r["instance_id"] in old])
        src_pf = sum(len(r["patch_src_files"]) for r in rows)
        s["tagger_recall_vs_final_patch"] = {
            "patch_src_files": src_pf,
            "seen_written_pct": pct(sum(len(r["patch_src_files_seen_written"]) for r in rows), src_pf),
            "tasks_nonempty_src_patch_but_no_primary_mutation": sorted(
                r["instance_id"] for r in rows if r["patch_src_files"] and r["primary"]["n_mut_calls"] == 0),
        }
        summary[key] = s
        # validation sample: first resolved, first unresolved non-empty, first empty-patch (per run), capped globally
        pick = []
        for cond in (lambda r: r["resolved"], lambda r: not r["resolved"] and r["patch_files"], lambda r: not r["patch_files"], lambda r: r["n_test_calls"] > 0):
            r = next((r for r in rows if cond(r) and r not in pick), None)
            if r:
                pick.append(r)
        miss = [r for r in rows if r["instance_id"] in s["tagger_recall_vs_final_patch"]["tasks_nonempty_src_patch_but_no_primary_mutation"]]
        shown += [("sample", r) for r in pick] + [("recall-miss", r) for r in miss[:2]]
        for r in rows:
            per_task_out.append({k: v for k, v in r.items() if not k.startswith("_")})

    print("\n==== VALIDATION: compact per-call tags (W:=write events, T:=test invocation + returned outcome) ====")
    for why, r in shown:
        p = r["primary"]
        print(f"\n[{why}] {r['run']} {r['instance_id']} resolved={r['resolved']} calls={r['n_calls']} "
              f"src_mut_calls={p['n_mut_calls']} first={p['first_mut']} last={p['last_mut']} "
              f"test_after={p['test_after_last_mut']} pass_after={p['pass_after_last_mut']} "
              f"mut_paths={p['mut_paths'][:4]} patch_files={r['patch_files'][:4]}")
        print(compact_tags(r))

    with open(os.path.join(args.out, "mutation_metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    with open(os.path.join(args.out, "mutation_metrics_per_task.jsonl"), "w") as f:
        for r in per_task_out:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print("\n==== SUMMARY (heuristic tagger) ====")
    for key, s in summary.items():
        print(f"\n## {key}: {s['n_resolved']}/{s['n_tasks']} resolved")
        for lab in ("resolved", "unresolved", "all"):
            c, o, t = s["chronology"][lab], s["old_tagger"][lab], s["test_activity"][lab]
            print(f"  {lab:10s} n={c['n']:3d} ge1mut={c['pct_ge1_mutation']} testAfter={c['pct_test_after_last_mut']} "
                  f"passAfter={c['pct_pass_after_last_mut']} medMut={c['median_mut_calls']} execAfter={c['pct_exec_after_last_mut']} execOkAfter={c['pct_exec_ok_after_last_mut']} "
                  f"anyTest={t['pct_any_test_invoked']} anyPass={t['pct_any_test_pass']} | OLD firstEdit={o.get('pct_first_edit_found')} "
                  f"testsAfter={o.get('pct_tests_after_last_edit')} unverified={o.get('n_wrong_fix_unverified')}")
        print("  recall vs final patch:", {k: (v if not isinstance(v, list) else len(v)) for k, v in s["tagger_recall_vs_final_patch"].items()})
    print("\nwrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
