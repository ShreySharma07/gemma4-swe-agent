"""
RS4: corrected episodes from failed R2 + R3 runs (appended after rs2_core.py by build.sh --rs4; pre-registered in
docs/experiments.md, Nov 13). A corrected episode = the failed run's own messages up to the cut, then our steps:
one exact-replace edit per gold hunk (v14's heredoc form), one run of the task's test file(s), submit_patch.
Every episode is replayed in the task snapshot and kept only if the result equals the gold patch and the tests pass.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

EDIT_TMPL = ("python3 - <<'PYEOF'\nimport pathlib\np = pathlib.Path(\"{path}\")\ns = p.read_text()\nold = \"\"\"{old}\"\"\"\n"
             "new = \"\"\"{new}\"\"\"\nassert s.count(old) == 1, s.count(old)\np.write_text(s.replace(old, new))\nPYEOF")
W_CORR_MULTI = 3.0
CORR_MAX_SHARE = 0.40


def parse_hunks(patch):
    """-> [(path, [(tag, line)])] per hunk; tag in ' ', '-', '+'."""
    out, path, cur = [], None, None
    for line in patch.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("--- ") or line.startswith("diff --git") or line.startswith("index "):
            continue
        elif line.startswith("@@"):
            cur = []
            out.append((path, cur))
        elif cur is not None and line[:1] in (" ", "-", "+"):
            cur.append((line[0], line[1:]))
        elif cur is not None and line == "":
            cur.append((" ", ""))
    return out


def snippet(lines, text):
    """Smallest old/new pair around the changed lines that is unique in the file text; None if there is none."""
    ch = [k for k, (t, _) in enumerate(lines) if t != " "]
    if not ch:
        return None
    lo, hi = ch[0], ch[-1]
    while True:
        old = "\n".join(l for t, l in lines[lo:hi + 1] if t != "+")
        new = "\n".join(l for t, l in lines[lo:hi + 1] if t != "-")
        if old.strip() and text.count(old) == 1:
            return old, new
        if lo == 0 and hi == len(lines) - 1:
            return None
        lo, hi = max(0, lo - 1), min(len(lines) - 1, hi + 1)


def tool_text(m):
    c = m.get("content") or ""
    try:
        d = json.loads(c)
        return str(d.get("stdout", "")) + str(d.get("content", "")) if isinstance(d, dict) else c
    except Exception:  # noqa: BLE001
        return c


def find_templates(eps):
    """Real v14 tool responses to copy the wrapper from: (edit with empty stdout, passing pytest, final submit)."""
    ed = py = sub = None
    for e in eps:
        ms = e["msgs"]
        for k, m in enumerate(ms[:-1]):
            if m["role"] != "assistant" or not m.get("tool_calls") or ms[k + 1]["role"] != "tool":
                continue
            fn = m["tool_calls"][0]["function"]
            c = str((fn.get("arguments") or {}).get("command", "")) if isinstance(fn.get("arguments"), dict) else ""
            try:
                d = json.loads(ms[k + 1]["content"])
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(d, dict):
                continue
            d = {a: b for a, b in d.items() if a != "budget_warning"}
            if fn["name"] == "submit_patch" and sub is None and k + 2 >= len(ms) - 1:
                sub = d
            elif fn["name"] == "run_command" and "stdout" in d and d.get("status") == "ok":
                if ed is None and "PYEOF" in c and "write_text" in c and d["stdout"] == "":
                    ed = d
                if py is None and "pytest" in c and "passed" in str(d["stdout"]) and "failed" not in str(d["stdout"]):
                    py = d
        if ed and py and sub:
            break
    return ed, py, sub


def replay(snap, cmds, gold, test_cmd):
    """-> (ok, why, test_output). Runs the edit commands in the snapshot, compares with the gold patch applied."""
    a, b = pathlib.Path(tempfile.mkdtemp(prefix="rs4a_")), pathlib.Path(tempfile.mkdtemp(prefix="rs4b_"))
    try:
        for d in (a, b):
            with tarfile.open(snap) as tf:
                tf.extractall(d, filter="data")
        env = dict(os.environ, PYTHONPATH=f"{a}{os.pathsep}{a / 'src'}", PYTHONDONTWRITEBYTECODE="1",
                   PATH=str(pathlib.Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
        for c in cmds:
            r = subprocess.run(["bash", "-c", c], cwd=a, env=env, capture_output=True, text=True, timeout=60)
            if r.returncode != 0 or r.stdout.strip():
                return False, "edit_failed", ""
        (b / "_gold.patch").write_text(gold)
        if subprocess.run(["git", "apply", "_gold.patch"], cwd=b, capture_output=True).returncode != 0:
            return False, "gold_apply_failed", ""
        for path in {p for p, _ in parse_hunks(gold)}:
            if (a / path).read_text() != (b / path).read_text():
                return False, "differs_from_gold", ""
        r = subprocess.run(["bash", "-c", test_cmd], cwd=a, env=env, capture_output=True, text=True, timeout=300)
        out = r.stdout[-5000:]
        if not re.search(r"\d+ passed", out) or re.search(r"\d+ (failed|error)", out):
            return False, "tests_not_passing", out[-300:]
        return True, "", out
    except Exception as e:  # noqa: BLE001
        return False, "replay_" + type(e).__name__, ""
    finally:
        shutil.rmtree(a, ignore_errors=True)
        shutil.rmtree(b, ignore_errors=True)


def build_corrected(per_task, traces, meta, system, resolved_eps, rep):
    tasks, snaps = {}, {}
    for p in glob.glob("/kaggle/input/**/tasks.jsonl", recursive=True):
        for l in open(p):
            t = json.loads(l)
            tasks.setdefault(t["instance_id"], t)
    for p in glob.glob("/kaggle/input/**/snapshots/*.tgz", recursive=True):
        snaps.setdefault(pathlib.Path(p).stem, p)
    ed_t, py_t, sub_t = find_templates(resolved_eps)
    rep["templates"] = {"edit": ed_t, "pytest": {k: (str(v)[:120] if k == "stdout" else v) for k, v in (py_t or {}).items()},
                        "submit": sub_t}
    funnel = {}

    def drop(why):
        funnel[why] = funnel.get(why, 0) + 1

    if not (ed_t and py_t and sub_t):
        rep["funnel"] = {"no_templates": 1}
        return []
    cands = []
    for tid in sorted(i for i, r in per_task.items() if not r.get("resolved")):
        drop("failed_runs")
        pt, m = per_task[tid], meta.get(tid, {})
        if tid not in tasks or tid not in snaps or tid not in traces:
            drop("x_missing_inputs")
            continue
        msgs, _, drift = episode(json.loads(traces[tid].read_text()), system)
        if drift or (pt.get("analyzer_calls") or 0):
            drop("x_drift_or_subagent")
            continue
        hs = parse_hunks(tasks[tid]["patch"])
        edits, bad = [], None
        with tarfile.open(snaps[tid]) as tf:
            names = {n[2:] if n.startswith("./") else n: n for n in tf.getnames()}
            for path, lines in hs:
                if path not in names:
                    bad = "x_file_not_in_snapshot"
                    break
                sn = snippet(lines, tf.extractfile(names[path]).read().decode("utf-8", "replace"))
                if sn is None:
                    bad = "x_no_unique_snippet"
                    break
                if any('"""' in s or "\\" in s for s in sn):
                    bad = "x_quotes_or_backslash"
                    break
                edits.append((path, sn[0], sn[1]))
        if bad or not edits:
            drop(bad or "x_no_hunks")
            continue
        need = {l.strip() for _, old, _ in edits for l in old.splitlines() if l.strip()}
        # cut = first tool-call step by which every old line has been shown; nothing written before it
        seen, cut, wrote, k_step = "", None, False, -1
        for k, mm in enumerate(msgs):
            if mm["role"] == "assistant" and mm.get("tool_calls"):
                k_step += 1
                fn = mm["tool_calls"][0]["function"]
                if tag_step(fn.get("name"), fn.get("arguments"), set()) in ("edit", "scratch_write", "submit"):
                    wrote = True
                    break
            elif mm["role"] == "tool":
                seen += "\n" + tool_text(mm)
                if all(n in seen for n in need) and (k + 1 == len(msgs) or msgs[k + 1]["role"] != "tool"):
                    cut = k + 1
                    break
        if cut is None:
            drop("x_wrote_before_lines_visible" if wrote else "x_lines_never_visible")
            continue
        f2p = tasks[tid]["FAIL_TO_PASS"]
        files = sorted({n.split("::")[0] for n in (json.loads(f2p) if isinstance(f2p, str) else f2p)})[:2]
        if not files:
            drop("x_no_test_file")
            continue
        cmds = [EDIT_TMPL.format(path=p, old=o, new=n) for p, o, n in edits]
        test_cmd = "python3 -m pytest " + " ".join(files) + " -q 2>&1 | tail -5"
        cands.append((tid, m, msgs[:cut], cmds, test_cmd, k_step + 1))
    funnel["reach_replay"] = len(cands)
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda c: replay(snaps[c[0]], c[3], tasks[c[0]]["patch"], c[4]), cands))
    out = []
    for (tid, m, prefix, cmds, test_cmd, n_prefix), (ok, why, test_out) in zip(cands, res):
        if not ok:
            drop("x_" + why)
            continue
        multi = m.get("kind") in ("pair", "triple")
        w = W_CORR_MULTI if multi else 1.0
        msgs, sw, n = list(prefix), [("prefix", 0.0)] * n_prefix, 0

        def add(name, args, resp, tag):
            nonlocal n
            n += 1
            cid = f"call_corr_{n}"
            msgs.append({"role": "assistant", "reasoning_content": "", "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(resp)})
            sw.append((tag, w))

        for c in cmds:
            add("run_command", {"command": c}, ed_t, "c_edit")
        add("run_command", {"command": test_cmd}, dict(py_t, stdout=test_out), "c_test")
        add("submit_patch", {}, sub_t, "c_submit")
        tags = [t for t, _ in sw]
        out.append({"id": tid, "lib": lib_of(tid), "kind": m.get("kind"), "src": tid[:3], "stray": [], "tags": tags,
                    "pytest_after_last_edit": True, "n_edits": len(cmds), "msgs": msgs, "sw": sw, "corrected": True,
                    "n_prefix": n_prefix, "issue": m.get("issue"), "split": "val" if lib_of(tid) in HOLDOUT else "train"})
    funnel["kept"] = len(out)
    rep["funnel"] = funnel
    rep["kept_train"] = sum(e["split"] == "train" for e in out)
    rep["kept_val"] = sum(e["split"] == "val" for e in out)
    tr = [e for e in out if e["split"] == "train"]
    rep["train_by"] = {k: {str(v): sum(str(e[k]) == str(v) for e in tr) for v in sorted({str(e[k]) for e in tr})}
                       for k in ("kind", "src", "issue", "lib")}
    rep["n_prefix_median"] = statistics.median([e["n_prefix"] for e in out]) if out else None
    return out
