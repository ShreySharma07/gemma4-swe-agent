
# ---------------------------------------------------------------------------------------------------------------
# A1: is the multi-site resolve gap explained by patch size?  (pre-registered in docs/experiments.md, Oct 23)
# Runs on Kaggle CPU. PAYLOAD (above) = per-task outcomes of our runs + exec-graph LOO results, tar.gz + base64.
# Gold patches come from the competition's secret/perfect_submission; issue text from tasks.jsonl.
# ---------------------------------------------------------------------------------------------------------------
import base64
import glob
import io
import json
import math
import pathlib
import random
import re
import tarfile

import numpy as np

WORK = pathlib.Path("/kaggle/working")
IN = WORK / "in"
IN.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as tf:
    tf.extractall(IN, filter="data")
print("payload files:", sorted(str(p.relative_to(IN)) for p in IN.rglob("*.jsonl")))

# ------------------------------------------------------------------ competition data
_all = sorted(glob.glob("/kaggle/input/**/tasks.jsonl", recursive=True))
_comp = [p for p in _all if (pathlib.Path(p).parent / "graphs").is_dir()]
tasks_path = (_comp or _all or [None])[0]
assert tasks_path, "attach the competition data"
DATA = pathlib.Path(tasks_path).parent
tasks = {}
for line in open(tasks_path):
    if line.strip():
        t = json.loads(line)
        tasks[t["instance_id"]] = t
sec = next((a / "secret" for a in [DATA, *DATA.parents] if (a / "secret").is_dir()), None)
gold = {}
if sec:
    import pandas as pd
    for ext in ("parquet", "csv"):
        f = sec / f"perfect_submission.{ext}"
        if f.exists():
            df = pd.read_parquet(f) if ext == "parquet" else pd.read_csv(f)
            for r in df.to_dict(orient="records"):
                iid = str(r.get("id") or r.get("instance_id") or "")
                val = r.get("prediction") or r.get("patch")
                if iid and isinstance(val, str):
                    gold[iid] = val
            break
for iid, t in tasks.items():
    if t.get("patch"):
        gold.setdefault(iid, t["patch"])
print(f"tasks: {len(tasks)}, gold patches: {len(gold)}")

EXCL = re.compile(r"(^|/)(tests?|testing|docs?|docs_src|examples)/|(^|/)test_[^/]*\.py$|conftest\.py$")


def patch_stats(p):
    files, cur, lines, hunks = set(), None, 0, 0
    for ln in p.splitlines():
        if ln.startswith("+++ b/"):
            cur = ln[6:].strip()
            if cur.endswith(".py") and not EXCL.search(cur):
                files.add(cur)
            else:
                cur = None
        elif cur and ln.startswith("@@"):
            hunks += 1
        elif cur and (ln.startswith("+") or ln.startswith("-")) and not ln.startswith(("+++", "---")):
            lines += 1
    return len(files), lines, hunks


# ------------------------------------------------------------------ LOO necessity (old-54)
meta = {json.loads(l)["id"]: json.loads(l) for l in open(next(IN.rglob("tasks_meta.jsonl")))}
res = [json.loads(l) for l in open(next(IN.rglob("results.jsonl")))]


def n_necessary(iid):
    g = meta.get(iid, {}).get("gold_function_groups") or []
    if len(g) < 2:
        return len(g) if g else None
    loo = {r["mode"]: r["resolved"] for r in res if r["id"] == iid and r["mode"].startswith("loo")}
    return sum(1 for i in range(len(g)) if loo.get(f"loo{i}") is False)


# ------------------------------------------------------------------ rows
RUNS = {}
for p in sorted(IN.rglob("per_task.jsonl")):
    s = str(p)
    name = ("v14_new60" if "v14" in s else "v7_new60") if "new60" in s else \
        "v7_old54_" + re.search(r"rep(\d)", s).group(0)
    RUNS[name] = p
rows = []
for run, p in RUNS.items():
    for line in open(p):
        r = json.loads(line)
        iid = r["instance_id"]
        if r.get("env_unverifiable") or iid not in gold:
            continue
        nf, nl, nh = patch_stats(gold[iid])
        if nf == 0:
            continue
        rows.append(dict(run=run, task=iid, repo=r["repo"], resolved=int(bool(r["resolved"])), n_files=nf,
                         lines=nl, hunks=nh, n_funcs=len(r.get("gold_funcs") or []),
                         issue=len(tasks.get(iid, {}).get("problem_statement", "") or ""),
                         multi_file=int(nf >= 2), n_nec=n_necessary(iid) if "old54" in run else None))
print(f"rows: {len(rows)} across runs {sorted(RUNS)}")


# ------------------------------------------------------------------ logistic regression (IRLS) + cluster-robust SE
def logit(y, X, cluster=None, iters=50):
    y, X = np.asarray(y, float), np.asarray(X, float)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ b)))
        W = mu * (1 - mu) + 1e-9
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H + 1e-8 * np.eye(len(b)), X.T @ (y - mu))
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    mu = 1 / (1 + np.exp(-(X @ b)))
    Hinv = np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]) + 1e-8 * np.eye(len(b)))
    if cluster is None:
        V = Hinv
    else:
        meat = np.zeros((len(b), len(b)))
        for c in set(cluster):
            idx = [i for i, cc in enumerate(cluster) if cc == c]
            g = X[idx].T @ (y[idx] - mu[idx])
            meat += np.outer(g, g)
        V = Hinv @ meat @ Hinv
    ll = float(np.sum(y * np.log(mu + 1e-12) + (1 - y) * np.log(1 - mu + 1e-12)))
    sep = bool(np.any(np.abs(b) > 15))
    return b, np.sqrt(np.diag(V)), ll, sep


def design(rs, with_multi=True, run_fe=False):
    repos = sorted({r["repo"] for r in rs})
    runs = sorted({r["run"] for r in rs})
    names = ["const"] + (["multi_file"] if with_multi else []) + ["log1p_lines", "log1p_issue"] + \
        [f"repo={x}" for x in repos[1:]] + ([f"run={x}" for x in runs[1:]] if run_fe else [])
    X = []
    for r in rs:
        v = [1.0] + ([r["multi_file"]] if with_multi else []) + [math.log1p(r["lines"]), math.log1p(r["issue"])]
        v += [1.0 if r["repo"] == x else 0.0 for x in repos[1:]]
        if run_fe:
            v += [1.0 if r["run"] == x else 0.0 for x in runs[1:]]
        X.append(v)
    return names, X


def p_norm(z):
    return math.erfc(abs(z) / math.sqrt(2))


def chi2_1_p(x):
    return math.erfc(math.sqrt(max(x, 0) / 2))


report = {"models": {}, "matched": {}, "strata": {}}


def fit(label, rs, run_fe=False, cluster_by_task=False):
    y = [r["resolved"] for r in rs]
    cl = [r["task"] for r in rs] if cluster_by_task else None
    names, X = design(rs, True, run_fe)
    b, se, ll1, sep = logit(y, X, cl)
    _, X0 = design(rs, False, run_fe)
    _, _, ll0, _ = logit(y, X0, cl)
    lr = 2 * (ll1 - ll0)
    i = names.index("multi_file")
    j = names.index("log1p_lines")
    out = dict(n=len(rs), events=int(sum(y)), separation=sep,
               multi_OR=math.exp(b[i]), multi_CI=[math.exp(b[i] - 1.96 * se[i]), math.exp(b[i] + 1.96 * se[i])],
               multi_wald_p=p_norm(b[i] / se[i]), LR_multi_given_size=lr, LR_p=chi2_1_p(lr),
               lines_OR=math.exp(b[j]), lines_p=p_norm(b[j] / se[j]))
    report["models"][label] = out
    print(f"\n[{label}] n={out['n']} events={out['events']} sep={sep}\n"
          f"   multi_file OR {out['multi_OR']:.2f} [{out['multi_CI'][0]:.2f}, {out['multi_CI'][1]:.2f}] "
          f"Wald p={out['multi_wald_p']:.4f}; LR(multi | size, issue, repo) = {lr:.2f}, p={out['LR_p']:.4f}\n"
          f"   log1p(lines) OR {out['lines_OR']:.2f} p={out['lines_p']:.4f}")


new60 = {k: [r for r in rows if r["run"] == k] for k in ("v7_new60", "v14_new60")}
for k, rs in new60.items():
    if rs:
        fit(k, rs)
pooled = new60["v7_new60"] + new60["v14_new60"]
if pooled:
    fit("pooled_new60_runFE_taskcluster", pooled, run_fe=True, cluster_by_task=True)
old = [r for r in rows if r["run"].startswith("v7_old54")]


# ------------------------------------------------------------------ size-matched comparison + permutation test
def matched(label, rs, key, caliper=0.5, k=3, perms=10000):
    by_task = {}
    for r in rs:
        by_task.setdefault(r["task"], []).append(r)
    T = []
    for task, v in by_task.items():
        x = v[0]
        if x[key] is None:
            continue
        T.append(dict(task=task, repo=x["repo"], ll=math.log1p(x["lines"]), treat=int(x[key] >= 2),
                      rate=sum(r["resolved"] for r in v) / len(v)))
    sets = []
    for t in [t for t in T if t["treat"]]:
        ctrl = sorted((c for c in T if not c["treat"] and c["repo"] == t["repo"] and abs(c["ll"] - t["ll"]) <= caliper),
                      key=lambda c: abs(c["ll"] - t["ll"]))[:k]
        if ctrl:
            sets.append((t, ctrl))
    if not sets:
        print(f"\n[{label}] no matched sets")
        return

    def gap(ss):
        return float(np.mean([np.mean([c["rate"] for c in cs]) - t["rate"] for t, cs in ss]))

    obs = gap(sets)
    rnd = random.Random(0)
    cnt = 0
    for _ in range(perms):
        ss = []
        for t, cs in sets:
            members = [t] + cs
            pick = rnd.randrange(len(members))
            nt = members[pick]
            ss.append((nt, [m for i_, m in enumerate(members) if i_ != pick]))
        cnt += gap(ss) >= obs - 1e-12
    out = dict(sets=len(sets), treated_rate=float(np.mean([t["rate"] for t, _ in sets])),
               control_rate=float(np.mean([np.mean([c["rate"] for c in cs]) for _, cs in sets])),
               gap_pp=100 * obs, perm_p_one_sided=cnt / perms,
               median_lines_treated=float(np.median([math.expm1(t["ll"]) for t, _ in sets])),
               median_lines_control=float(np.median([math.expm1(c["ll"]) for _, cs in sets for c in cs])))
    report["matched"][label] = out
    print(f"\n[{label}] matched sets={out['sets']}: multi {out['treated_rate']:.2f} vs size-matched single "
          f"{out['control_rate']:.2f} -> gap {out['gap_pp']:.0f} pp, permutation p={out['perm_p_one_sided']:.4f} "
          f"(median lines {out['median_lines_treated']:.0f} vs {out['median_lines_control']:.0f})")


for k, rs in new60.items():
    matched(k + "_multi_file", rs, "n_files")
matched("v7_old54_x5_multi_necessary", old, "n_nec")
matched("v7_old54_x5_multi_file", old, "n_files")

# ------------------------------------------------------------------ descriptive: resolve by multi x size tercile
for k, rs in list(new60.items()) + [("v7_old54_x5", old)]:
    if not rs:
        continue
    ls = sorted(r["lines"] for r in rs)
    q1, q2 = ls[len(ls) // 3], ls[2 * len(ls) // 3]
    tab = {}
    for r in rs:
        size = "small" if r["lines"] <= q1 else ("mid" if r["lines"] <= q2 else "large")
        key = f"{'multi' if r['multi_file'] else 'single'}_{size}"
        a = tab.setdefault(key, [0, 0])
        a[0] += r["resolved"]
        a[1] += 1
    report["strata"][k] = {"cut_lines": [q1, q2], **{kk: f"{v[0]}/{v[1]}" for kk, v in sorted(tab.items())}}
    print(f"\n[{k}] resolve by file-count x size tercile (lines <= {q1} / <= {q2} / larger):")
    for kk, v in sorted(tab.items()):
        print(f"   {kk:14s} {v[0]}/{v[1]}")

(WORK / "a1_report.json").write_text(json.dumps(report, indent=1))
(WORK / "a1_rows.json").write_text(json.dumps(rows))
print("\nwrote /kaggle/working/a1_report.json and a1_rows.json")
