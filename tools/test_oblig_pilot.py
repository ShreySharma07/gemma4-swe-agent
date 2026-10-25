"""Functional tests for tools/oblig.py on the 7 pilot tasks (real repo snapshots from the xg instrument).

For each task: copy the base snapshot into a temp git repo, then
  1. `edit` with the gold version of the first test-necessary function (OLD = base source, NEW = gold source)
     -> must print EDIT OK and an OBLIGATIONS block, and the file must change;
  2. `check` -> must list the changed function and stay under 1,000 characters;
  3. error paths: OLD text absent -> EDIT FAILED; an edit that breaks syntax -> EDIT REJECTED, file unchanged.
Recall is NOT asserted here (that is E0a); it is printed for information only.

usage: python tools/test_oblig_pilot.py [data/runs/xg]
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "analysis"))
import oblig  # noqa: E402
import obligation_pilot as op  # noqa: E402

XG = Path(sys.argv[1] if len(sys.argv) > 1 else HERE.parent / "data/runs/xg")
PILOT = ["fastapi_14986", "fastapi_15800", "rich_3052", "rich_3061", "rich_3468", "rich_3942", "rich_3953"]
OBLIG = str(HERE / "oblig.py")


def sh(cwd, *cmd, inp=None):
    r = subprocess.run(cmd, cwd=cwd, input=inp, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def make_repo(iid, dst):
    with tarfile.open(XG / "src" / f"{iid}.tgz") as tf:
        tf.extractall(dst, filter="data") if sys.version_info >= (3, 12) else tf.extractall(dst)
    sh(dst, "git", "init", "-q")
    sh(dst, "git", "add", "-A")
    sh(dst, "git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")


def gold_after(repo, patch):
    tmp = tempfile.mkdtemp()
    shutil.copytree(repo, tmp, dirs_exist_ok=True)
    rc, out = sh(tmp, "git", "apply", "--whitespace=nowarn", "-", inp=patch if patch.endswith("\n") else patch + "\n")
    return (tmp if rc == 0 else None), out


def func_src(root, key):
    path, qual = key.split("::", 1)
    qual = qual.replace(".<locals>", "")
    text = Path(root, path).read_text()
    defs = oblig._defs_of(text, path)
    hit = defs.get(f"{path}::{qual}")
    return hit[1] if hit else None


def main():
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(XG / "tasks_meta.jsonl")}
    results = [json.loads(l) for l in open(XG / "results.jsonl")]
    fails = skips = 0
    for iid in PILOT:
        m = meta[iid]
        grps = op.groups(m)
        nec = op.necessary(results, iid, len(grps))
        work = tempfile.mkdtemp()
        make_repo(iid, work)
        after, err = gold_after(work, m["patch"])
        if after is None:
            print(f"[{iid}] SKIP: gold patch does not apply: {err[:200]}")
            skips += 1
            continue
        anchor = grps[nec[0]][0]
        path = anchor.split("::")[0]
        old, new = func_src(work, anchor), func_src(after, anchor)
        ok = old is not None and new is not None and old != new and Path(work, path).read_text().count(old) == 1
        if not ok:
            print(f"[{iid}] SKIP: anchor {anchor} not editable as one block")
            skips += 1
            continue
        block = f"<<<<<<< OLD\n{old}\n=======\n{new}\n>>>>>>> NEW\n"
        before_text = Path(work, path).read_text()
        rc, out = sh(work, sys.executable, OBLIG, "edit", path, inp=block)
        e1 = "EDIT OK" in out and "OBLIGATIONS" in out or "No structural obligations" in out
        e1 = e1 and Path(work, path).read_text() != before_text
        rc, chk = sh(work, sys.executable, OBLIG, "check")
        anchor_q = anchor.split("::")[1].replace(".<locals>", "")
        e2 = "CHECK:" in chk and anchor_q.split(".")[-1] in chk and len(chk) < 1000
        rc, bad = sh(work, sys.executable, OBLIG, "edit", path,
                     inp="<<<<<<< OLD\nthis text is not in the file 123\n=======\nx\n>>>>>>> NEW\n")
        e3 = "EDIT FAILED" in bad
        cur = Path(work, path).read_text()
        first = new.splitlines()[0]
        rc, syn = sh(work, sys.executable, OBLIG, "edit", path, inp=f"<<<<<<< OLD\n{first}\n=======\n{first}(((\n>>>>>>> NEW\n")
        e4 = ("EDIT REJECTED" in syn or "EDIT FAILED" in syn) and Path(work, path).read_text() == cur
        others = [grps[i] for i in nec[1:]]
        shown = sum(any(k.split("::")[1].replace(".<locals>", "").split(".")[-1] in out for k in g) for g in others)
        status = all((e1, e2, e3, e4))
        fails += not status
        print(f"[{iid}] {'PASS' if status else 'FAIL'} edit={e1} check={e2} notfound={e3} syntax={e4} "
              f"| anchor {anchor_q}; other necessary groups named in edit output: {shown}/{len(others)}; check {len(chk)} chars")
        if not status:
            print("   edit out:", out[:400].replace("\n", " | "))
            print("   check out:", chk[:400].replace("\n", " | "))
            print("   syntax out:", syn[:200].replace("\n", " | "))
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(after, ignore_errors=True)
    print(f"\n{len(PILOT) - fails - skips}/{len(PILOT)} pilot tasks pass, {fails} fail, {skips} skipped")
    return 1 if fails or skips else 0


if __name__ == "__main__":
    sys.exit(main())
