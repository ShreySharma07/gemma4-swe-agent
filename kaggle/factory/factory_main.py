# ------------------------------------------------------------------ factory v0 driver (Kaggle CPU, Internet ON)
# Output: /kaggle/working/factory/ = tasks.jsonl (verified only), tasks_all.jsonl, snapshots/<id>.tgz, task_meta.jsonl,
# verify_{gold,empty}.jsonl, factory_report.json. Same task format as the competition data, so the eval runner can
# point EvalConfig.tasks_path / snapshots_dir here for on-policy rollouts.
CFG = {"N_TRY": 250, "N_PAIRS": 15, "WANT_LIBS": 3, "MIN_TASKS_PER_LIB": 10, "CONCURRENCY": 4}
OUT = pathlib.Path("/kaggle/working/factory")
ROOT = pathlib.Path("/tmp/factory")
T0 = time.time()
shutil.rmtree(OUT, ignore_errors=True)
(OUT / "snapshots").mkdir(parents=True)
ROOT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest", "pytest-timeout"])

comp_row = json.loads(open(tasks_path).readline())
log("competition task keys:", {k: type(v).__name__ for k, v in comp_row.items()})

# REUSE: verify an earlier run's tasks (attached as a kernel source) instead of rebuilding them (v0 hit the 12 h cap)
PRIOR = sorted(glob.glob("/kaggle/input/**/factory/tasks_all.jsonl", recursive=True))
tasks, lib_reports = [], []
if PRIOR:
    src = pathlib.Path(PRIOR[0]).parent
    log("REUSE prior build:", src)
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(src / "task_meta.jsonl")} if (src / "task_meta.jsonl").exists() else {}
    for l in open(src / "tasks_all.jsonl"):
        t = json.loads(l)
        t["PASS_TO_PASS"] = [] if isinstance(t.get("PASS_TO_PASS"), list) else "[]"  # see build_task
        t["_meta"] = {k: v for k, v in meta.get(t["instance_id"], {"kind": "?", "ops": []}).items() if k != "id"}
        shutil.copy(src / "snapshots" / f"{t['instance_id']}.tgz", OUT / "snapshots")
        tasks.append(t)
    prior_rep = src / "factory_report.json"
    lib_reports = json.loads(prior_rep.read_text())["libs"] if prior_rep.exists() else []
for lib in ([] if PRIOR else LIBS):
    if sum(r.get("tasks", 0) >= CFG["MIN_TASKS_PER_LIB"] for r in lib_reports) >= CFG["WANT_LIBS"]:
        break
    t = time.time()
    try:
        lt, rep = run_library(lib, ROOT, OUT / "snapshots", n_try=CFG["N_TRY"], n_pairs=CFG["N_PAIRS"], log=log)
    except Exception as e:  # noqa: BLE001  one bad library must not stop the others
        lt, rep = [], {"lib": lib["name"], "error": f"{type(e).__name__}: {e}"[:400]}
    rep["minutes"] = round((time.time() - t) / 60, 1)
    log("LIB", json.dumps(rep))
    lib_reports.append(rep)
    tasks += lt

meta = {t["instance_id"]: t.pop("_meta") for t in tasks}
for t in tasks:  # mirror the competition rows: same keys, same F2P/P2P type
    for k, v in comp_row.items():
        if k not in t:
            t[k] = "" if isinstance(v, str) else type(v)()
        elif k in ("FAIL_TO_PASS", "PASS_TO_PASS") and isinstance(v, list):
            t[k] = json.loads(t[k])
with open(OUT / "tasks_all.jsonl", "w") as f:
    f.writelines(json.dumps(t) + "\n" for t in tasks)
with open(OUT / "task_meta.jsonl", "w") as f:
    f.writelines(json.dumps(dict(id=i, **m)) + "\n" for i, m in meta.items())
log(f"built {len(tasks)} tasks in {(time.time() - T0) / 60:.1f} min")

# ------------------------------------------------------------------ official verification: gold resolves, empty does not
ensure_ensurepip()
verified, shadow = {}, {}
if tasks and ensure_harness():
    from adk_eval_core.tracing import SessionTrace
    from adk_submission import ModelRegistry
    from swegemma.config import EvalConfig
    from swegemma.evaluate import Evaluator

    use_scorer_env()
    for lib in LIBS:  # does the scorer site already ship the library? (an installed copy could shadow /workspace)
        r = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(SCORER_SITE)!r}); "
                            f"import {lib['pkg']} as m; print(m.__file__)"], capture_output=True, text=True)
        shadow[lib["name"]] = r.stdout.strip() or None
    log("in scorer site:", shadow)
    dummy = ROOT / "dummy_submission"
    dummy.mkdir(exist_ok=True)

    def run_mode(mode):
        async def fake_agent(self, task, snapshot_path, task_index, total_tasks, **kw):
            return (task.patch if mode == "gold" else ""), None, SessionTrace()
        Evaluator._run_agent_sandbox = fake_agent
        cfg = EvalConfig(tasks_path=OUT / "tasks_all.jsonl", snapshots_dir=OUT / "snapshots",
                         results_dir=ROOT / f"verify_{mode}", submission_dir=dummy, models=ModelRegistry(),
                         sandbox="subprocess", wheels_dir=DATA / "wheels", task_ids=[t["instance_id"] for t in tasks],
                         concurrency=CFG["CONCURRENCY"], display_mode="quiet", max_time_minutes=30, timeout_seconds=600)
        res = asyncio.run(Evaluator(cfg).run())
        with open(OUT / f"verify_{mode}.jsonl", "w") as f:
            for r in res.task_results:
                f.write(json.dumps({"id": r.instance_id, "resolved": bool(r.resolved), "exit": r.test_exit_code,
                                    "error": r.error, "tail": (r.test_output or "")[-1500:]}) + "\n")
        return {r.instance_id: bool(r.resolved) for r in res.task_results}

    t = time.time()
    gold = run_mode("gold")
    empty = run_mode("empty")
    log(f"verify: gold {sum(gold.values())}/{len(gold)}, empty {sum(empty.values())}/{len(empty)} "
        f"({(time.time() - t) / 60:.1f} min)")
    verified = {i: {"gold": gold.get(i), "empty": empty.get(i)} for i in meta}
else:
    log("harness unavailable -> tasks are unverified")

keep = [t for t in tasks if verified.get(t["instance_id"], {}).get("gold") and
        verified[t["instance_id"]].get("empty") is False]
with open(OUT / "tasks.jsonl", "w") as f:
    f.writelines(json.dumps(t) + "\n" for t in keep)
kept_ids = {t["instance_id"] for t in keep}
for t in tasks:
    if t["instance_id"] not in kept_ids and verified:
        (OUT / "snapshots" / f"{t['instance_id']}.tgz").unlink(missing_ok=True)


def lib_of(i):
    return i.split("_", 1)[1].rsplit("_", 1)[0]


report = {
    "config": CFG, "libs": lib_reports, "shadowed_by_scorer_site": shadow,
    "built": len(tasks), "verified": len(keep),
    "by_lib": {lib: {"built": sum(lib_of(i) == lib for i in meta), "verified": sum(lib_of(i) == lib for i in kept_ids),
                     "pairs_verified": sum(lib_of(i) == lib and meta[i]["kind"] == "pair" for i in kept_ids)}
               for lib in sorted({lib_of(i) for i in meta})},
    "gold_fail": sorted(i for i, v in verified.items() if not v["gold"]),
    "empty_pass": sorted(i for i, v in verified.items() if v["empty"]),
    "f2p_sizes": sorted(len(t["FAIL_TO_PASS"] if isinstance(t["FAIL_TO_PASS"], list) else json.loads(t["FAIL_TO_PASS"]))
                        for t in keep),
    "ops_verified": {o: sum(o in meta[i]["ops"] for i in kept_ids) for o in sorted({o for m in meta.values() for o in m["ops"]})},
    "sample_issues": {t["instance_id"]: t["problem_statement"] for t in keep[:: max(1, len(keep) // 4)][:4]},
    "snapshot_mb": round(sum(p.stat().st_size for p in (OUT / "snapshots").glob("*.tgz")) / 1e6, 1),
    "minutes": round((time.time() - T0) / 60, 1),
}
(OUT / "factory_report.json").write_text(json.dumps(report, indent=1))
log("FACTORY REPORT", json.dumps({k: v for k, v in report.items() if k != "sample_issues"}))
