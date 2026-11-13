"""H1 (T2 plan): how many failed R2 + R3 runs can become corrected training episodes? Kaggle CPU, stdlib only.
Pre-registered in docs/experiments.md (Nov 13). Outputs /kaggle/working/h1/: h1_report.json, h1_rows.jsonl."""
import glob, json, pathlib, re, statistics, tarfile

OUT = pathlib.Path("/kaggle/working/h1"); OUT.mkdir(parents=True, exist_ok=True)
WORK = pathlib.Path("/tmp/h1"); WORK.mkdir(parents=True, exist_ok=True)
HOLDOUT = ("toolz", "pathspec")


def lib_of(i):
    return i.split("_", 1)[1].rsplit("_", 1)[0]


tasks, meta, snaps = {}, {}, {}
for p in glob.glob("/kaggle/input/**/tasks.jsonl", recursive=True):
    for l in open(p):
        t = json.loads(l)
        tasks.setdefault(t["instance_id"], t)
for p in glob.glob("/kaggle/input/**/task_meta.jsonl", recursive=True):
    for l in open(p):
        m = json.loads(l)
        meta.setdefault(m["id"], m)
for p in glob.glob("/kaggle/input/**/snapshots/*.tgz", recursive=True):
    snaps.setdefault(pathlib.Path(p).stem, p)
per_task, traces, done = {}, {}, set()
for tgz in sorted(glob.glob("/kaggle/input/**/eval_rollout_s*.tgz", recursive=True)):
    key = ("r3" if "r3-rollout" in tgz else "r2", pathlib.Path(tgz).stem)
    if key in done:
        continue
    done.add(key)
    with tarfile.open(tgz) as tf:
        tf.extractall(WORK / key[0], filter="data")
    root = WORK / key[0] / key[1] / "rep0"
    for l in open(root / "analysis" / "per_task.jsonl"):
        r = json.loads(l)
        per_task[r["instance_id"]] = r
    for p in (root / "merged" / "traces").glob("trace_*.json"):
        traces[p.stem.replace("trace_", "")] = p
print("inputs", sorted(done), len(tasks), "tasks", len(meta), "meta", len(snaps), "snapshots", len(per_task), "runs", flush=True)


def hunks(patch):
    """-> [(path, old_text, header_name)] per hunk; old_text = context + removed lines."""
    out, path, cur, name = [], None, None, None
    for line in patch.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("@@"):
            if cur is not None:
                out.append((path, "\n".join(cur), name))
            m = re.search(r"@@.*@@.*?\b(?:def|class)\s+(\w+)", line)
            cur, name = [], (m.group(1) if m else None)
        elif cur is not None and line[:1] in (" ", "-") and not line.startswith("--- "):
            cur.append(line[1:])
    if cur is not None:
        out.append((path, "\n".join(cur), name))
    return out


def unique_hunks(tid, hs):
    if tid not in snaps:
        return None
    ok = True
    with tarfile.open(snaps[tid]) as tf:
        names = {n.lstrip("./"): n for n in tf.getnames()}
        for path, old, _ in hs:
            if path not in names:
                return False
            text = tf.extractfile(names[path]).read().decode("utf-8", "replace")
            ok = ok and text.count(old) == 1
    return ok


rows = []
for tid, pt in sorted(per_task.items()):
    if pt.get("resolved") or tid not in tasks or tid not in traces:
        continue
    m, hs = meta.get(tid, {}), hunks(tasks[tid]["patch"])
    names = [f.split("::")[-1].split(".")[-1] for f in (m.get("funcs") or [])]
    src = "meta"
    if not names:
        names, src = [h[2] for h in hs if h[2]], "hunk_header"
    steps = [s for s in json.loads(traces[tid].read_text())["steps"] if s.get("tool_calls")]
    first = {}
    for k, s in enumerate(steps):
        text = json.dumps(s.get("tool_calls")) + json.dumps(s.get("observation"))
        for n in names:
            if n not in first and re.search(r"\b" + re.escape(n) + r"\b", text):
                first[n] = k
    cut = max(first.values()) if names and len(first) == len(set(names)) else None
    uniq = unique_hunks(tid, hs)
    ftype = ("error" if pt.get("harness_error") or pt.get("overflow") else "empty" if pt.get("patch_empty") else
             "cap" if pt.get("budget_exhausted") else "wrong")
    held = lib_of(tid) in HOLDOUT
    rows.append({"id": tid, "src": tid[:3], "lib": lib_of(tid), "kind": m.get("kind"), "issue": m.get("issue"),
                 "names": names, "names_from": src, "n_hunks": len(hs), "n_steps": len(steps), "cut": cut,
                 "seen": len(first), "hunks_unique": uniq, "ftype": ftype, "held_out": held,
                 "usable": bool(cut is not None and uniq and not held)})
with open(OUT / "h1_rows.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)


def by(key, sel):
    return {str(v): sum(1 for r in sel if r[key] == v) for v in sorted({str(r[key]) for r in sel})} if sel else {}


def by(key, sel):  # noqa: F811
    out = {}
    for r in sel:
        out[str(r[key])] = out.get(str(r[key]), 0) + 1
    return dict(sorted(out.items()))


us = [r for r in rows if r["usable"]]
cuts = [r["cut"] for r in rows if r["cut"] is not None]
report = {
    "failed_runs": len(rows), "failed_by_src": by("src", rows), "failed_by_type": by("ftype", rows),
    "names_from": by("names_from", rows), "no_names": sum(not r["names"] for r in rows),
    "cut_exists": len(cuts), "cut_median": statistics.median(cuts) if cuts else None,
    "cut_p90": sorted(cuts)[int(0.9 * len(cuts))] if cuts else None,
    "hunks_unique": sum(r["hunks_unique"] is True for r in rows), "no_snapshot": sum(r["hunks_unique"] is None for r in rows),
    "held_out": sum(r["held_out"] for r in rows),
    "usable": len(us), "usable_multi": sum(r["kind"] in ("pair", "triple") for r in us),
    "usable_by": {k: by(k, us) for k in ("kind", "src", "issue", "ftype")}, "usable_libs": by("lib", us),
    "partial_seen_multi": sum(r["cut"] is None and r["seen"] > 0 and r["kind"] in ("pair", "triple") for r in rows),
}
(OUT / "h1_report.json").write_text(json.dumps(report, indent=1))
print("H1 REPORT", json.dumps(report), flush=True)
