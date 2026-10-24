
# ---------------------------------------------------------------------------------------------------------------
# A8 (pre-registered Oct 24, docs/experiments.md): why does BM25 beat the structural frontier on E0b?
# Kaggle CPU + Internet. PAYLOAD (above) = results/runs/e0b/e0b_pairs.json + tools/oblig.py (tar.gz, base64).
# Re-downloads the SWE-rebench `filtered` parquet and the GitHub tarballs of the E0b tasks (parsed, never executed).
# Per pair: (i) identifier Jaccard anchor-after vs target-base, (ii) shared edited identifier, (iii) edit-pattern
# Jaccard of the two functions' changed-line tokens, (iv) same file. Then hit@6 per arm within strata + logit.
# ---------------------------------------------------------------------------------------------------------------
import base64
import io
import json
import math
import pathlib
import re
import sys
import tarfile
import urllib.request
from collections import defaultdict

import numpy as np

WORK = pathlib.Path("/kaggle/working")
IN = WORK / "in"
IN.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as tf:
    tf.extractall(IN, filter="data")
sys.path.insert(0, str(IN / "tools"))
import oblig  # noqa: E402

pairs_doc = json.loads(next(IN.rglob("e0b_pairs.json")).read_text())
pairs = pairs_doc["pairs"]
task_ids = sorted({p["task"] for p in pairs})
print(f"E0b pairs {len(pairs)} from {len(task_ids)} tasks")

PQ = WORK / "filtered.parquet"
if not PQ.exists():
    urllib.request.urlretrieve(
        "https://huggingface.co/datasets/nebius/SWE-rebench/resolve/main/data/filtered-00000-of-00001.parquet", PQ)
import pyarrow.parquet as pq  # noqa: E402
meta = {r["instance_id"]: r for r in pq.read_table(PQ, columns=["instance_id", "repo", "base_commit", "patch"]).to_pylist()
        if r["instance_id"] in set(task_ids)}
print("tasks found in parquet:", len(meta))

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
KW = {"self", "cls", "def", "return", "None", "True", "False", "import", "from", "class", "for", "while", "if", "else",
      "elif", "and", "not", "with", "try", "except", "raise", "lambda", "pass", "yield", "assert", "async", "await"}


def idents(s):
    return {t for t in IDENT.findall(s or "") if t not in KW}


def fetch(repo, sha):
    url = f"https://codeload.github.com/{repo}/tar.gz/{sha}"
    data = urllib.request.urlopen(url, timeout=120).read()
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith(".py"):
                rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
                files[rel] = tf.extractfile(m).read().decode("utf-8", "replace")
    return files


def parse_patch(patch):
    """{path: [(old_line or None, new_line or None, '+'/'-', text)]} for changed lines."""
    out, path, o, n = defaultdict(list), None, 0, 0
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
        elif line.startswith("--- "):
            continue
        elif line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", line)
            o, n = int(m.group(1)), int(m.group(2))
        elif path and line.startswith("-"):
            out[path].append((o, None, "-", line[1:]))
            o += 1
        elif path and line.startswith("+"):
            out[path].append((None, n, "+", line[1:]))
            n += 1
        elif path and not line.startswith("\\"):
            o += 1
            n += 1
    return out


def spans(text, path):
    return [(d.line, d.end, k) for k, (d, _) in oblig._defs_of(text, path).items()]


def innermost(sp, ln):
    inside = [(e - s, k) for s, e, k in sp if s <= ln <= e]
    return min(inside)[1] if inside else None


feats = {}
for iid in task_ids:
    m = meta.get(iid)
    if not m:
        continue
    try:
        base = fetch(m["repo"], m["base_commit"])
    except Exception as e:  # noqa: BLE001
        print("fetch failed", iid, type(e).__name__)
        continue
    ch = parse_patch(m["patch"])
    edit_tokens = defaultdict(set)
    for p, lines in ch.items():
        sb = spans(base.get(p, ""), p)
        for o, n, kind, txt in lines:
            if kind == "-" and o:
                k = innermost(sb, o)
                if k:
                    edit_tokens[k] |= idents(txt)
        # added lines: attribute to the enclosing function of the nearest preceding removed/context line in base
        last_old = 1
        for o, n, kind, txt in lines:
            if o:
                last_old = o
            elif kind == "+":
                k = innermost(sb, last_old)
                if k:
                    edit_tokens[k] |= idents(txt)
    srcs = {}
    for p, t in base.items():
        if p in ch:
            for k, (d, seg) in oblig._defs_of(t, p).items():
                srcs[k] = seg
    feats[iid] = (edit_tokens, srcs, base)
print("tasks with features:", len(feats))

rows = []
for pr in pairs:
    f = feats.get(pr["task"])
    if not f:
        continue
    edit_tokens, srcs, base = f
    a, t = pr["anchor"], pr["target"]
    ea, et = edit_tokens.get(a, set()), edit_tokens.get(t, set())
    ia = idents(srcs.get(a, "")) | ea
    tsrc = srcs.get(t)
    if tsrc is None:
        p_, q_ = t.split("::", 1)
        tsrc = (oblig._defs_of(base.get(p_, ""), p_).get(t, (None, ""))[1])
    it = idents(tsrc)
    jac = len(ia & it) / max(1, len(ia | it))
    pat = len(ea & et) / max(1, len(ea | et))
    rows.append({**{k: pr[k] for k in pr if "@6" in k}, "task": pr["task"], "same_file": pr["same_file"],
                 "ident_jaccard": jac, "shared_edit": int(bool(ea & et)), "pattern_jaccard": pat})
print("pairs with features:", len(rows))

ARMS = ["TYPED", "UNTYPED", "BM25", "FILE-PROX", "FILE-RANDOM"]
rep = {"n_pairs": len(rows), "strata": {}}


def table(label, sel):
    xs = [r for r in rows if sel(r)]
    rep["strata"][label] = {"n": len(xs), **{a: round(sum(r[f"{a}@6"] for r in xs) / max(1, len(xs)), 3) for a in ARMS}}
    print(f"  {label:34s} n={len(xs):4d} " + " ".join(f"{a} {rep['strata'][label][a]:.2f}" for a in ARMS))


print("\nhit@6 by stratum:")
table("all", lambda r: True)
table("shared edited identifier", lambda r: r["shared_edit"])
table("no shared edited identifier", lambda r: not r["shared_edit"])
qs = sorted(r["pattern_jaccard"] for r in rows)
c1, c2 = qs[len(qs) // 3], qs[2 * len(qs) // 3]
table(f"pattern sim low (<= {c1:.2f})", lambda r: r["pattern_jaccard"] <= c1)
table(f"pattern sim mid", lambda r: c1 < r["pattern_jaccard"] <= c2)
table(f"pattern sim high (> {c2:.2f})", lambda r: r["pattern_jaccard"] > c2)
table("same file", lambda r: r["same_file"])
table("cross file", lambda r: not r["same_file"])


def logit(y, X, iters=60):
    y, X = np.asarray(y, float), np.asarray(X, float)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ b)))
        H = X.T @ (X * (mu * (1 - mu) + 1e-9)[:, None])
        step = np.linalg.solve(H + 1e-8 * np.eye(len(b)), X.T @ (y - mu))
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    mu = 1 / (1 + np.exp(-(X @ b)))
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]) + 1e-8 * np.eye(len(b)))))
    return b, se


names = ["const", "ident_jaccard", "shared_edit", "pattern_jaccard", "same_file"]
X = [[1, r["ident_jaccard"], r["shared_edit"], r["pattern_jaccard"], int(r["same_file"])] for r in rows]
for arm in ("BM25", "TYPED"):
    y = [int(r[f"{arm}@6"] >= 0.5) for r in rows]
    b, se = logit(y, X)
    rep[f"logit_{arm}"] = {n: {"coef": round(float(c), 3), "p": round(math.erfc(abs(c / s) / math.sqrt(2)), 4)}
                           for n, c, s in zip(names, b, se)}
    print(f"\nlogit {arm}@6>=0.5 ~ features:", json.dumps(rep[f"logit_{arm}"]))
ns = rep["strata"]["no shared edited identifier"]
rep["prediction_check"] = {"bm25_minus_typed_no_shared_edit": round(ns["BM25"] - ns["TYPED"], 3),
                           "predicted": "< 0.10"}
print("\nprediction check:", rep["prediction_check"])
(WORK / "a8_report.json").write_text(json.dumps(rep, indent=1))
(WORK / "a8_rows.json").write_text(json.dumps(rows))
