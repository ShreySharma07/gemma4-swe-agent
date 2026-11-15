"""
TP1 teacher pilot (CPU + Internet, no GPU, no vLLM). Pre-registered in docs/experiments.md (Nov 16).
The submission under test (v14) runs unchanged through the official Evaluator, but the model registry maps the
declared model name to an open-weight teacher behind an OpenAI-compatible endpoint (Settings.teacher_base /
teacher_model), so the trajectories use our exact prompt, tools and budgets.
  - tasks: Settings.task_ids, taken from R2's rewritten issues (rollout_tasks.jsonl of the attached R2 outputs);
    the held-out gate libraries (toolz, pathspec) are refused
  - key: env TEACHER_API_KEY (set by the notebook from a Kaggle Secret); read once, removed from the environment
    before any task sandbox starts, never logged
  - network: everything except the teacher's API host goes to a dead proxy, as in the offline runs
  - usage: tokens and failed calls are counted with LiteLLM callbacks (the per-task rows carry calls, not tokens)
--dry: tasks, snapshots, submission and registry only; no key, no API call.
Output: /kaggle/working/teacher_report.json + teacher_rollout.tgz. Run inside the venv:
python -m evalkit.teacher_rollout [--dry]
"""
import json
import os
import pathlib
import sys
import threading
import time
import urllib.parse

from .compat import patch_sandbox_venv, use_scorer_sandbox
from .config import INPUT, TARGET_MODEL, WORK, Settings
from .factory_smoke import breakdown, phase
from .inputs import Inputs
from .server import make_sandbox_offline, set_scorer_env
from .submission import Submission
from .util import log

HOLDOUT = ("toolz", "pathspec")  # the G1/G2 gate set: never in training data
KEY_ENV = "TEACHER_API_KEY"
TASKS_OUT = WORK / "teacher_tasks.jsonl"
SNAPS = pathlib.Path("/tmp/teacher_snapshots")
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "failed": {}}
_LOCK = threading.Lock()


def lib_of(i):
    return i.split("_", 1)[1].rsplit("_", 1)[0]


def load_tasks(ids):
    """The requested tasks with R2's rewritten issues, their snapshots (symlinked into one dir) and factory meta."""
    bad = [i for i in ids if lib_of(i) in HOLDOUT]
    assert not bad, f"held-out gate tasks requested: {bad[:5]}"
    rows = {}
    for p in sorted(INPUT.rglob("rollout_tasks.jsonl")):
        for l in open(p):
            if l.strip():
                t = json.loads(l)
                if t["instance_id"] in ids:
                    rows.setdefault(t["instance_id"], t)
    missing = [i for i in ids if i not in rows]
    assert not missing, f"no rewritten issue for {missing[:5]} (attach gemma4-factory-rollout-s0 / -s1)"
    meta, snaps = {}, {}
    for tp in sorted(p for p in INPUT.rglob("tasks.jsonl") if (p.parent / "task_meta.jsonl").exists()):
        meta.update({json.loads(l)["id"]: json.loads(l) for l in open(tp.parent / "task_meta.jsonl") if l.strip()})
        snaps.update({s.stem: s for s in (tp.parent / "snapshots").glob("*.tgz")})
    SNAPS.mkdir(parents=True, exist_ok=True)
    for i in ids:
        assert i in snaps, f"missing snapshot for {i} (attach gemma4-factory-v0 / -v1)"
        (SNAPS / f"{i}.tgz").unlink(missing_ok=True)
        (SNAPS / f"{i}.tgz").symlink_to(snaps[i])
    tasks = [rows[i] for i in ids]
    with open(TASKS_OUT, "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in tasks)
    return tasks, meta


def _on_success(kwargs, response, start, end):
    try:
        u = getattr(response, "usage", None) or (response.get("usage") if isinstance(response, dict) else None)
        get = (lambda k: (u.get(k) if isinstance(u, dict) else getattr(u, k, None)) or 0) if u else (lambda k: 0)
        det = (u.get("prompt_tokens_details") if isinstance(u, dict) else getattr(u, "prompt_tokens_details", None)) if u else None
        cached = ((det.get("cached_tokens") if isinstance(det, dict) else getattr(det, "cached_tokens", None)) or 0) if det else 0
        with _LOCK:
            USAGE["calls"] += 1
            USAGE["prompt_tokens"] += get("prompt_tokens")
            USAGE["completion_tokens"] += get("completion_tokens")
            USAGE["cached_tokens"] += cached
    except Exception:  # noqa: BLE001  accounting must never break a rollout
        pass


def _on_failure(kwargs, response, start, end):
    try:
        e = kwargs.get("exception")
        name = f"{type(e).__name__}:{getattr(e, 'status_code', '')}" if e is not None else "unknown"
        with _LOCK:
            USAGE["failed"][name] = USAGE["failed"].get(name, 0) + 1
    except Exception:  # noqa: BLE001
        pass


def registry(s, declared, key):
    """The declared model name (and the target name) -> the teacher behind an OpenAI-compatible endpoint."""
    import litellm
    from adk_submission import ModelRegistry
    from adk_submission.resolvers.generation import install_litellm_reasoning_patch
    from google.adk.models.lite_llm import LiteLlm
    litellm.drop_params = True          # top_k, thinking config: dropped if the endpoint does not take them
    litellm.telemetry = False
    litellm.suppress_debug_info = True
    litellm.success_callback = [_on_success]
    litellm.failure_callback = [_on_failure]
    install_litellm_reasoning_patch()
    reg = ModelRegistry()
    for alias in dict.fromkeys([declared, TARGET_MODEL]):
        reg.register(alias, LiteLlm(model=f"openai/{s.teacher_model}", api_base=s.teacher_base, api_key=key,
                                    num_retries=6, timeout=300))
    return reg


def ping(s, key):
    """One tiny tool-call request: fails fast on a bad key, model id or missing tool support. Returns a short status."""
    import litellm
    tools = [{"type": "function", "function": {"name": "run_command", "description": "Run a shell command.",
                                                "parameters": {"type": "object", "required": ["command"],
                                                               "properties": {"command": {"type": "string"}}}}}]
    try:
        r = litellm.completion(model=f"openai/{s.teacher_model}", api_base=s.teacher_base, api_key=key, tools=tools,
                               messages=[{"role": "user", "content": "List the files in the current directory "
                                                                     "by calling run_command."}],
                               max_tokens=512, timeout=120, num_retries=2)
        msg = r.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        return {"ok": True, "tool_call": bool(calls), "tool": calls[0].function.name if calls else None,
                "finish_reason": r.choices[0].finish_reason}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}".replace(key, "<key>")}


def main(dry=False):
    s = Settings.load()
    key = os.environ.pop(KEY_ENV, "")  # out of the environment before any sandbox (the agent can run `env`)
    assert s.task_ids, "teacher pilot needs Settings.task_ids"
    assert s.teacher_model and s.teacher_base, "teacher_model / teacher_base not set"
    set_scorer_env()
    patch_sandbox_venv()
    comp = Inputs.locate(False, need_model=False)
    use_scorer_sandbox(comp.data)
    tasks, meta = load_tasks(list(s.task_ids))
    ids = [t["instance_id"] for t in tasks]
    sub = Submission().unpack()
    declared, adapters = sub.declared_model(), sub.adapters()
    assert not (getattr(adapters, "adapters", None) or {}), "teacher runs take a submission without adapters"
    budget = {**sub.eval_budget(), **({"max_time_minutes": s.max_time_minutes} if s.max_time_minutes else {})}
    report = {"teacher_model": s.teacher_model, "teacher_base": s.teacher_base, "declared": declared, "tasks": ids,
              "libs": {h: sum(lib_of(i) == h for i in ids) for h in sorted({lib_of(i) for i in ids})},
              "budget": budget, "concurrency": s.teacher_conc, "key_present": bool(key)}
    log("TEACHER setup", json.dumps({k: v for k, v in report.items() if k != "tasks"}))
    models = registry(s, declared, key or "DRY")
    if dry:
        report["dry"] = True
        (WORK / "teacher_report.json").write_text(json.dumps(report, indent=1))
        return
    assert key, f"{KEY_ENV} is empty: attach the Kaggle Secret to this notebook"
    report["ping"] = ping(s, key)
    log("TEACHER ping", json.dumps(report["ping"]))
    rows = []
    try:
        assert report["ping"]["ok"], "teacher endpoint not usable (see ping)"
        make_sandbox_offline()
        host = urllib.parse.urlparse(s.teacher_base).hostname
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = f"localhost,127.0.0.1,{host}"
        inp = Inputs(TASKS_OUT, comp.data, SNAPS, "")
        report["run"], rows = phase(s, inp, sub, models, adapters, "teacher", ids, s.teacher_conc, chunk=10)
    finally:
        ref = s.teacher_ref or {}
        res = {r["instance_id"]: bool(r.get("resolved")) for r in rows}
        n = max(len(rows), 1)
        calls = sorted(r.get("tool_calls") or 0 for r in rows)
        report["overall"] = {"resolved": sum(res.values()), "n": len(rows), **breakdown(rows, meta, budget),
                             "tool_calls_median": calls[len(calls) // 2] if calls else None}
        report["vs_gemma"] = {  # R2's result for the same task and prompt (v14 on the served Gemma)
            "gemma_resolved": {"n": sum(ref.get(i) is True for i in res), "teacher": sum(v for i, v in res.items() if ref.get(i) is True)},
            "gemma_failed": {"n": sum(ref.get(i) is False for i in res), "teacher": sum(v for i, v in res.items() if ref.get(i) is False)}}
        with _LOCK:
            report["usage"] = {**USAGE, "failed": dict(USAGE["failed"]),
                               "prompt_tokens_per_task": round(USAGE["prompt_tokens"] / n),
                               "completion_tokens_per_task": round(USAGE["completion_tokens"] / n),
                               "calls_per_task": round(USAGE["calls"] / n, 1)}
        (WORK / "teacher_report.json").write_text(json.dumps(report, indent=1))
        log("TEACHER REPORT", json.dumps({k: v for k, v in report.items() if k not in ("tasks", "run")}))
        log.sh(f"cd {WORK} && tar -czf teacher_rollout.tgz teacher_report.json teacher_tasks.jsonl run.log "
               f"$(ls eval_teacher*.tgz 2>/dev/null)")
        log(f"RUN_DONE in {(time.time() - s.t0) / 3600:.1f} h")


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
