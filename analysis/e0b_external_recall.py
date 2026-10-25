"""E0b: obligation recall of a first-edit frontier on EXTERNAL repos (pre-registered in docs/experiments.md, Oct 23).

SWE-rebench `filtered` tasks, repo-disjoint from ours, at most one task per repo, random order (seed 0), first
120 eligible tasks. Label = gold co-change: every pair of *existing* functions the gold patch changes.
Simulated first edit = the gold version of the anchor function only. Arms as in E0a, minus PROV/EXEC
(they need the competition's graphs/traces). Snapshots come from GitHub tarballs and are only parsed (ast),
never executed.

usage: python analysis/e0b_external_recall.py PARQUET CACHE_DIR OUT_DIR
"""
import io
import json
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
import exec_graph_analysis as xa  # noqa: E402
import oblig  # noqa: E402
from e0a_obligation_recall import KS, arm_file, arm_file_random, arm_typed, arm_untyped, hit_at  # noqa: E402

N_TASKS, SEED, MIN_F, MAX_F = int(__import__("os").environ.get("E0B_N", 120)), 0, 2, 8  # E0B_N only for smoke tests
OURS = {"fastapi/fastapi", "Textualize/rich", "psf/requests", "encode/httpx"}
EXCL = re.compile(r"(^|/)(tests?|testing|docs?|docs_src|examples|benchmarks)/|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$")
MAX_TGZ = 150 * 2 ** 20


def py_files(patch):
    return [f for f in re.findall(r"^\+\+\+ b/(\S+)", patch, re.M) if f.endswith(".py") and not EXCL.search(f)]


def prefilter(r):
    return r["repo"] not in OURS and 1 <= len(py_files(r["patch"])) <= 6 and len(re.findall(r"^@@ ", r["patch"], re.M)) >= 2


def py_only_patch(patch):
    parts = re.split(r"(?m)^(?=diff --git )", patch)
    keep = [p for p in parts if p.startswith("diff --git ") and re.match(r"diff --git a/\S+\.py b/\S+\.py", p)]
    out = "".join(keep)
    return out if out.endswith("\n") else out + "\n"


def fetch(task, cache):
    """GitHub tarball at base_commit -> cache/<id>/ with .py files only. Returns path or (None, reason)."""
    d = Path(cache) / task["instance_id"]
    if (d / ".ok").exists():
        return str(d), None
    url = f"https://codeload.github.com/{task['repo']}/tar.gz/{task['base_commit']}"
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read(MAX_TGZ + 1)
        if len(data) > MAX_TGZ:
            return None, "tarball too large"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for m in tf.getmembers():
                if not m.isfile() or not m.name.endswith(".py"):
                    continue
                rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
                if ".." in rel.split("/"):
                    continue
                out = d / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(tf.extractfile(m).read())
        (d / ".ok").write_text("ok")
        return str(d), None
    except Exception as e:  # network / format
        return None, f"fetch failed: {type(e).__name__}"


def apply_gold(base, patch):
    d = tempfile.mkdtemp()
    shutil.copytree(base, d, dirs_exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=d, capture_output=True)
    r = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], cwd=d, input=py_only_patch(patch),
                       capture_output=True, text=True)
    return d if r.returncode == 0 else None


def changed_functions(base, after, patch, idx):
    keys = {}
    for f in py_files(patch):
        try:
            b = Path(base, f).read_text(errors="replace")
        except OSError:
            b = ""
        try:
            a = Path(after, f).read_text(errors="replace")
        except OSError:
            a = ""
        for k, v in oblig.change_kinds(b, a, f).items():
            if k in idx.defs:
                keys[k] = v
    top = [k for k in keys if not any(k != o and k.startswith(o + ".") for o in keys)]
    return top


def main():
    import pyarrow.parquet as pq
    parquet, cache, outdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(parquet, columns=["instance_id", "repo", "base_commit", "patch"]).to_pylist()
    rows = sorted((r for r in rows if prefilter(r)), key=lambda r: r["instance_id"])
    random.Random(SEED).shuffle(rows)
    seen, queue = set(), []
    for r in rows:
        if r["repo"] not in seen:
            seen.add(r["repo"])
            queue.append(r)
    reasons = Counter()
    agg = defaultdict(lambda: defaultdict(list))
    per_task, pair_rows = [], []
    pos = 0
    pool = ThreadPoolExecutor(8)
    while len(per_task) < N_TASKS and pos < len(queue):
        batch = queue[pos:pos + 16]
        pos += len(batch)
        fetched = list(pool.map(lambda t: fetch(t, cache), batch))
        for task, (base, why) in zip(batch, fetched):
            if len(per_task) >= N_TASKS:
                break
            if base is None:
                reasons[why] += 1
                continue
            after = apply_gold(base, task["patch"])
            if after is None:
                reasons["gold patch does not apply"] += 1
                continue
            try:
                idx = oblig.Index(base).build()
                funcs = changed_functions(base, after, task["patch"], idx)
                if not (MIN_F <= len(funcs) <= MAX_F):
                    reasons[f"changed existing functions = {len(funcs)}"] += 1
                    continue
                docs = {}
                for k, d in idx.defs.items():
                    seg = xa.ast.get_source_segment(idx.src[d.path], d.node) or ""
                    docs[k] = xa.tokens(seg + " " + d.qual)
                pairs = []
                ok = True
                anchors = {}
                for a in funcs:
                    path = a.split("::")[0]
                    before = Path(base, path).read_text(errors="replace")
                    gold = Path(after, path).read_text(errors="replace")
                    bseg = oblig._defs_of(before, path).get(a, (None, None))[1]
                    gseg = oblig._defs_of(gold, path).get(a, (None, None))[1]
                    if not bseg or not gseg or before.count(bseg) != 1:
                        ok = False
                        break
                    text = before.replace(bseg, gseg, 1)
                    ch = {k: v for k, v in oblig.change_kinds(before, text, path).items() if k in idx.defs}
                    if not ch:
                        ok = False
                        break
                    anchors[a] = (path, ch, gseg)
                if not ok:
                    reasons["anchor not editable as one block"] += 1
                    continue
                for a, (path, ch, gseg) in anchors.items():
                    akeys = set(ch)
                    bm = xa.bm25_rank(gseg, docs)
                    arms = {
                        "TYPED": arm_typed(idx, ch),
                        "UNTYPED": arm_untyped(idx, ch),
                        "FILE-PROX": arm_file(idx, akeys, path),
                        "FILE-RANDOM": arm_file_random(idx, akeys, path),
                        "BM25": [(k, s) for k, s in bm.items() if k not in akeys and s > 0],
                    }
                    for t in funcs:
                        if t == a:
                            continue
                        pr = {"task": task["instance_id"], "repo": task["repo"], "anchor": a, "target": t,
                              "kinds": sorted(set().union(*(v[0] for v in ch.values()))),
                              "same_file": t.split("::")[0] == path}
                        for name, ranked in arms.items():
                            for k in KS:
                                h = hit_at(ranked, [t], k)
                                pr[f"{name}@{k}"] = h
                                agg[name][k].append(h)
                        pairs.append(pr)
                pair_rows.extend(pairs)
                per_task.append({"task": task["instance_id"], "repo": task["repo"], "n_funcs": len(funcs), "n_pairs": len(pairs)})
                print(f"[{len(per_task):3d}/{N_TASKS}] {task['instance_id']}: {len(funcs)} functions, {len(pairs)} pairs", flush=True)
            except Exception as e:  # a broken repo must not kill the run; it is counted
                reasons[f"error: {type(e).__name__}"] += 1
            finally:
                shutil.rmtree(after, ignore_errors=True)

    order = ["TYPED", "UNTYPED", "BM25", "FILE-PROX", "FILE-RANDOM"]
    n = len(agg["TYPED"][1])
    print(f"\n## E0b: {n} ordered pairs from {len(per_task)} tasks / {len({t['repo'] for t in per_task})} repos "
          f"(queue position {pos}/{len(queue)})")
    print("  arm          " + "  ".join(f"@{k:<5}" for k in KS))
    for name in order:
        vals = [sum(agg[name][k]) / max(1, len(agg[name][k])) for k in KS]
        print(f"  {name:12s} " + "  ".join(f"{v:5.2f} " for v in vals))
    by = defaultdict(list)
    for p in pair_rows:
        by[p["task"]].append(p)
    print("  task-macro @6: " + ", ".join(
        f"{a} {sum(sum(x[a + '@6'] for x in v) / len(v) for v in by.values()) / max(1, len(by)):.2f}" for a in order))
    ts = list(by)
    for a, b in [("TYPED", "UNTYPED"), ("TYPED", "BM25"), ("TYPED", "FILE-RANDOM"), ("UNTYPED", "BM25")]:
        out = []
        for s in range(2000):
            rnd = random.Random(s)
            xs = [x for t in (rnd.choice(ts) for _ in ts) for x in by[t]]
            out.append(sum(x[a + "@6"] - x[b + "@6"] for x in xs) / len(xs))
        out.sort()
        print(f"  {a} - {b} @6: 95% repo-cluster bootstrap [{out[50]:+.2f}, {out[1950]:+.2f}]")
    print("ineligible / skipped:", dict(reasons))
    t6 = sum(agg["TYPED"][6]) / max(1, n)
    best = max((sum(agg[a][6]) / max(1, n), a) for a in order[1:])
    gate = t6 >= 0.80 and t6 - best[0] >= 0.20
    print(f"\nGATE: TYPED hit@6 = {t6:.2f} (need >= 0.80); best other = {best[1]} {best[0]:.2f}; "
          f"margin {t6 - best[0]:+.2f} (need >= +0.20) -> {'PASS' if gate else 'FAIL'}")
    (outdir / "e0b_pairs.json").write_text(json.dumps({"pairs": pair_rows, "tasks": per_task, "ineligible": dict(reasons)}, indent=1))


if __name__ == "__main__":
    main()
