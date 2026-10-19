"""Task order + ChunkedEvaluation: the official swegemma Evaluator, one task at a time, in budgeted chunks."""
import asyncio
import json
import random
import time

from .config import VALID_TASKS, WORK
from .util import log


def stratified_order(tasks, seed=0):
    """Shuffle within each repo, then interleave repos (largest first) so any prefix is a stratified sample."""
    rng = random.Random(seed)
    by_repo = {}
    for tk in tasks:
        by_repo.setdefault(tk["repo"], []).append(tk["instance_id"])
    pools = []
    for v in by_repo.values():
        rng.shuffle(v)
        pools.append(list(v))
    ids, total = [], sum(len(p) for p in pools)
    while len(ids) < total:
        for p in sorted(pools, key=len, reverse=True):
            if p:
                ids.append(p.pop())
    return ids


class ChunkedEvaluation:
    def __init__(self, settings, inputs, submission, models, adapters):
        self.s, self.inp, self.sub = settings, inputs, submission
        self.models, self.adapters = models, adapters
        self.budget = submission.eval_budget()
        self.root = WORK / settings.label
        ids = list(settings.task_ids) if settings.task_ids else stratified_order(inputs.load_tasks(), settings.seed)
        if settings.valid_only:  # tasks the scorer env can verify (gold passes, empty fails): see preflight sweep
            valid = set(json.loads(VALID_TASKS.read_text()))
            ids = [i for i in ids if i in valid]
        self.ids = ids[:settings.max_tasks] if settings.max_tasks else ids
        (WORK / "task_ids_order.json").write_text(json.dumps(self.ids, indent=1))
        log(f"{len(self.ids)} tasks x {settings.repeats} rep(s), budget {self.budget}")

    def config(self, chunk, results_dir):
        """EvalConfig with the scorer's limits, context cache and compaction (interval 15 / threshold 32768)."""
        from google.adk.agents.context_cache_config import ContextCacheConfig
        from google.adk.apps._configs import EventsCompactionConfig
        from swegemma.config import EvalConfig, build_submission_limits
        limits, gen = build_submission_limits()
        return EvalConfig(
            tasks_path=self.inp.tasks_path, snapshots_dir=self.inp.snapshots, results_dir=results_dir,
            submission_dir=self.sub.dir, models=self.models, sandbox="subprocess", wheels_dir=self.inp.data / "wheels",
            graph_dir=str(self.inp.data / "graphs"), embeddings_dir=str(self.inp.data / "embeddings"),
            **self.budget, limits=limits, generation_constraints=gen, adapter_manifest=self.adapters,
            context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
            events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
                                                            token_threshold=32768, event_retention_size=5),
            task_ids=chunk, concurrency=1, display_mode="quiet", verbose=False)

    def task_min(self):
        """The submission's per-task agent time limit (eval_config.yaml max_time_minutes)."""
        return float(self.budget.get("max_time_minutes") or 4.5)

    def run_chunk(self, rep, start):
        from swegemma.evaluate import Evaluator
        chunk = self.ids[start:start + self.s.chunk_size]
        t = time.time()
        try:
            res = asyncio.run(asyncio.wait_for(
                Evaluator(self.config(chunk, self.root / f"rep{rep}" / f"chunk_{start:03d}")).run(),
                timeout=max(self.s.chunk_timeout_min, len(chunk) * (self.task_min() + 3)) * 60))
            log(f"rep {rep} chunk {start // self.s.chunk_size}: {repr(res)[:300]} ({(time.time() - t) / 60:.1f} min)")
        except Exception as e:  # noqa: BLE001
            log(f"rep {rep} chunk {start // self.s.chunk_size} FAILED: {type(e).__name__}: {e}")

    def healthy(self, chunk_dir):
        """After the first chunk: abort on systemic failure instead of burning a 10 h session on garbage."""
        p = chunk_dir / "task_results.jsonl"
        rows = [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []
        traces = list((chunk_dir / "traces").glob("*.json"))
        checks = {"has_results": bool(rows),
                  "not_all_errors": any(not r.get("error") for r in rows),
                  "model_calls_tools": any((r.get("tool_calls") or 0) > 0 for r in rows),
                  "traces_saved": bool(traces),
                  # task_results.jsonl has no test output: it is in test_outputs/<id>.log; exit code -1 = not run
                  "tests_ran": any((r.get("test_exit_code") if r.get("test_exit_code") is not None else -1) >= 0
                                   for r in rows) and any((chunk_dir / "test_outputs").glob("*.log"))}
        log("HEALTH after first chunk:", json.dumps(checks),
            "| errors:", [str(r.get("error"))[:120] for r in rows if r.get("error")][:3])
        return all(checks.values())

    def run(self):
        """Rep-major: rep 0 covers every task before rep 1 starts, so a budget stop still leaves a full eval."""
        for rep in range(self.s.repeats):
            for start in range(0, len(self.ids), self.s.chunk_size):
                n = len(self.ids[start:start + self.s.chunk_size])
                if self.s.hours_left() < n * max(self.s.est_task_min, self.task_min() + 1) / 60:
                    log(f"STOP: {self.s.hours_left():.2f} h left (rep {rep}, task {start})")
                    return
                self.run_chunk(rep, start)
                if rep == 0 and start == 0 and not self.healthy(self.root / "rep0" / "chunk_000"):
                    log("ABORT: first chunk failed the health check; fix before a full run")
                    return
