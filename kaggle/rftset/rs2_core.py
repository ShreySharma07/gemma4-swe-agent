"""
RS2: R2 resolved runs -> max-context training chunks with per-step loss weights (Kaggle CPU; appended after rft_core.py
by build.sh --rs2; pre-registered in docs/experiments.md, Oct 31). Red-team findings applied (docs/HQ.md, Oct 31):
  - step tagger that sees v14's real edits (run_command heredocs / sed -i), unlike analysis/trace_analyzer.py
  - masks (weight 0, kept as context): superseded submits + their cleanup, repeats with no edit in between, steps that
    write or run stray files, drifted calls
  - weights: edit / pytest / final submit 1.0, everything else 0.5; multi-site first-edit -> submit x3;
    episodes x2 if pytest ran after the last edit, x0.5 if the patch kept stray files; step weight capped at 3
  - split: toolz + pathspec held out (val + gate); boltons / sqlparse capped at LIB_CAP train episodes, keeping
    pytest-after and multi-site episodes first
  - chunks: each episode tiled from the end into ~CHUNK-token targets, boundaries snapped to tool-response starts,
    each chunk's prefix = everything before it (PA1 prefix-cached training); prefill_calls = ceil(P / 8192)
Outputs (/kaggle/working/rs2/): rs2.npz (ids, weight, ep_starts, chunks), episodes.json, rs2_report.json.
"""
import math
import random
import re

import numpy as np

HOLDOUT = ("toolz", "pathspec")
LIB_CAP = {"boltons": 30, "sqlparse": 30}
CHUNK, CHUNK_MIN, CHUNK_MAX, MAX_PREFIX = 4096, 3072, 5120, 24576
W_KEY, W_OTHER, W_MULTI, W_PYTEST_EP, W_STRAY_EP, W_CAP = 1.0, 0.5, 3.0, 2.0, 0.5, 3.0
WRAPT = re.compile(r"Error in sitecustomize; set PYTHONVERBOSE for traceback:\\n|ModuleNotFoundError: No module named 'wrapt'\\n|"
                   r"Error in sitecustomize; set PYTHONVERBOSE for traceback:\n|ModuleNotFoundError: No module named 'wrapt'\n")
EDIT_CMD = re.compile(r"sed\s+-i|write_text\(|\.write\(|open\([^)]*['\"][wa]['\"]|cat\s*>{1,2}\s*\S|\btee\s+\S|git\s+apply|\bpatch\s+-p")
PATHS = re.compile(r"[\w./-]+\.(?:py|txt|cfg|toml|json|md|rst)")
SCRATCH = re.compile(r"^/tmp/|repro|reproduce|debug|scratch|^test_issue|^check_")


def is_scratch(path, stray):
    b = path.rsplit("/", 1)[-1]
    return bool(SCRATCH.search(path) or SCRATCH.search(b)) or b in stray or path in stray


def tag_step(name, args, stray):
    """-> one of edit, scratch_write, pytest, run, cleanup, submit, explore, other."""
    args = args if isinstance(args, dict) else {}
    if name == "submit_patch":
        return "submit"
    if name == "write_file":
        return "scratch_write" if is_scratch(str(args.get("filepath", "")), stray) else "edit"
    if name != "run_command":
        return "other"
    c = str(args.get("command", ""))
    if re.search(r"\bpytest\b|-m\s+unittest", c):
        return "pytest"
    if re.match(r"\s*(cd\s+\S+\s*&&\s*)?rm\s", c):
        return "cleanup"
    if EDIT_CMD.search(c):
        paths = PATHS.findall(c)
        return "scratch_write" if paths and all(is_scratch(p, stray) for p in paths) else "edit"
    if re.search(r"\bpython3?\b", c):
        return "run"
    return "explore"


def step_weights(msgs, kind, stray, ep_w):
    """Per assistant message: (tag, weight). Masks and weights as in the module docstring."""
    steps = [m for m in msgs if m["role"] == "assistant"]
    calls = [((m.get("tool_calls") or [{}])[0].get("function") or {}) for m in steps]
    tags = [tag_step(c.get("name"), c.get("arguments"), stray) if c else "other" for c in calls]
    edits = [i for i, t in enumerate(tags) if t == "edit"]
    submits = [i for i, t in enumerate(tags) if t == "submit"]
    last_submit = submits[-1] if submits else None
    first_edit = edits[0] if edits else None
    out, seen = [], {}
    for i, (t, c) in enumerate(zip(tags, calls)):
        key = (c.get("name"), repr(c.get("arguments")))
        prev = seen.get(key)
        seen[key] = i
        w = W_KEY if t in ("edit", "pytest", "submit") else W_OTHER
        if t == "submit" and i != last_submit:
            w = 0.0                                      # superseded submit
        elif t == "cleanup" and last_submit is not None and submits and i > submits[0]:
            w = 0.0                                      # submit -> rm repro -> resubmit ritual
        elif prev is not None and not any(prev < e < i for e in edits):
            w = 0.0                                      # exact repeat with nothing edited in between
        elif t == "scratch_write" and stray:
            w = 0.0                                      # writes a file that ends up stray in the patch
        elif t == "run" and stray and any(s.rsplit("/", 1)[-1] in str((c.get("arguments") or {}).get("command", "")) for s in stray):
            w = 0.0
        elif t == "submit" and stray:
            w = 0.0                                      # the stray-carrying final submit
        if w and kind in ("pair", "triple") and first_edit is not None and first_edit <= i <= (last_submit or i):
            w *= W_MULTI
        out.append((t, min(W_CAP, w * ep_w) if w else 0.0))
    return out, tags


def chunk_bounds(n_tok, cand):
    """Targets tiled from the end: [(P, end)], boundaries snapped to candidate token indices (tool-response starts)."""
    out, end = [], n_tok
    cand = sorted(set(cand))
    while end > 0:
        lo, hi = end - CHUNK_MAX, end - CHUNK_MIN
        if lo <= 0:
            out.append((0, end))
            break
        inside = [c for c in cand if lo <= c <= hi]
        b = min(inside, key=lambda c: abs(end - CHUNK - c)) if inside else end - CHUNK
        out.append((b, end))
        end = b
    return out[::-1]


def main_rs2(submission_dir, tools_base):
    out_dir = pathlib.Path("/kaggle/working/rs2")
    out_dir.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    model_dir = next(pathlib.Path(p).parent for p in sorted(glob.glob("/kaggle/input/**/tokenizer.json", recursive=True))
                     if "w4a16" in p.lower())
    g = Gemma(model_dir)
    system = f'You are an agent. Your internal name is "{AGENT}".' + "\n\n" + (submission_dir / "prompts" / "system.md").read_text()
    tools = tools_base + [{"type": "function", "function": ANALYZER}]   # RS1's best fidelity variant (gap 29 tokens)
    meta = {}
    for p in glob.glob("/kaggle/input/**/task_meta.jsonl", recursive=True):
        meta.update({json.loads(l)["id"]: json.loads(l) for l in open(p)})
    per_task, traces, done = {}, {}, set()
    for tgz in sorted(glob.glob("/kaggle/input/**/eval_rollout_s*.tgz", recursive=True)):
        name = pathlib.Path(tgz).stem.replace("eval_", "")
        if name in done:
            continue
        done.add(name)
        with tarfile.open(tgz) as tf:
            tf.extractall(WORK, filter="data")
        root = WORK / f"eval_{name}" / "rep0"
        for l in open(root / "analysis" / "per_task.jsonl"):
            r = json.loads(l)
            per_task[r["instance_id"]] = r
        for p in (root / "merged" / "traces").glob("trace_*.json"):
            traces[p.stem.replace("trace_", "")] = p
    assert len(done) == 2, f"expected 2 R2 shards, got {sorted(done)}"
    resolved = sorted(i for i, r in per_task.items() if r.get("resolved"))

    # pass 1: messages, tags, episode-level flags (fixed tagger)
    eps, wrapt_stripped = [], 0
    for i in resolved:
        pt, m = per_task[i], meta.get(i, {})
        msgs, _, drift = episode(json.loads(traces[i].read_text()), system)
        if drift or (pt.get("analyzer_calls") or 0):
            continue                                     # malformed calls / sub-agent: excluded (6 in RS1)
        for mm in msgs:
            if mm["role"] == "tool" and isinstance(mm.get("content"), str):
                new = WRAPT.sub("", mm["content"])
                wrapt_stripped += new != mm["content"]
                mm["content"] = new
        stray = set(pt.get("stray_files_in_patch") or [])
        _, tags = step_weights(msgs, m.get("kind"), stray, 1.0)
        edits = [k for k, t in enumerate(tags) if t == "edit"]
        pytest_after = bool(edits) and any(t == "pytest" for t in tags[edits[-1] + 1:])
        eps.append({"id": i, "lib": lib_of(i), "kind": m.get("kind"), "src": i[:3], "stray": sorted(stray), "tags": tags,
                    "pytest_after_last_edit": pytest_after, "n_edits": len(edits), "msgs": msgs,
                    "split": "val" if lib_of(i) in HOLDOUT else "train"})
    # library cap on train: keep pytest-after and multi-site episodes first
    rng = random.Random(0)
    for lib, cap in LIB_CAP.items():
        pool = [e for e in eps if e["lib"] == lib and e["split"] == "train"]
        rng.shuffle(pool)
        pool.sort(key=lambda e: not (e["pytest_after_last_edit"] or e["kind"] != "single"))
        for e in pool[cap:]:
            e["split"] = "capped"

    # pass 2: render, token weights, chunks
    ids_all, w_all, ep_starts, chunks, rows = [], [], [0], [], []
    tag_tok, n_mismatch = {}, 0
    for ei, e in enumerate(e for e in eps if e["split"] != "capped"):
        ep_w = (W_PYTEST_EP if e["pytest_after_last_edit"] else 1.0) * (W_STRAY_EP if e["stray"] else 1.0)
        sw, _ = step_weights(e["msgs"], e["kind"], set(e["stray"]), ep_w)
        text = g.render(e["msgs"], tools)
        spans = model_spans(text)
        if len(spans) != len(sw):  # one span per assistant step expected; else weight every span at W_OTHER
            n_mismatch += 1
            sw = [("other", W_OTHER * ep_w)] * len(spans)
        enc = g.tok.encode(text, add_special_tokens=False)
        ids, offs = enc.ids, enc.offsets
        w = np.zeros(len(ids), dtype=np.float32)
        starts = np.array([o[0] for o in offs])
        for (a, b), (t, wt) in zip(spans, sw):
            sel = (starts >= a) & (starts < b)
            w[sel] = wt
            tag_tok[t] = tag_tok.get(t, 0) + int(sel.sum())
        cand = [int(np.searchsorted(starts, k)) for k in (m.start() for m in re.finditer(re.escape("<|tool_response>"), text))]
        for P, end in chunk_bounds(len(ids), cand):
            if P > MAX_PREFIX:  # drop the earliest context beyond MAX_PREFIX: never happens for RS1 (max 27k)
                continue
            if w[P:end].sum() > 0:
                chunks.append((ei, P, end, math.ceil(P / 8192)))
        ids_all.append(np.asarray(ids, dtype=np.int32))
        w_all.append(w.astype(np.float16))
        ep_starts.append(ep_starts[-1] + len(ids))
        rows.append({k: v for k, v in e.items() if k != "msgs"} | {"ep_weight": ep_w, "n_tokens": len(ids),
                                                                    "trained_weight_sum": round(float(w.sum()), 1)})
    np.savez_compressed(out_dir / "rs2.npz", ids=np.concatenate(ids_all), weight=np.concatenate(w_all),
                        ep_starts=np.asarray(ep_starts, dtype=np.int64), chunks=np.asarray(chunks, dtype=np.int64),
                        val=np.asarray([r["split"] == "val" for r in rows], dtype=np.uint8))
    (out_dir / "episodes.json").write_text(json.dumps(rows))

    # report + drift check: share of trained weight by step type (vs v14's own call mix)
    tr = [r for r in rows if r["split"] == "train"]
    calls = {}
    for r in tr:
        for t in r["tags"]:
            calls[t] = calls.get(t, 0) + 1
    tot_calls = sum(calls.values())
    ch = np.asarray(chunks)
    tr_idx = {k for k, r in enumerate(rows) if r["split"] == "train"}
    ch_tr = [c for c in chunks if c[0] in tr_idx]
    report = {
        "episodes": {"resolved": len(resolved), "kept": len(eps), "train": len(tr), "val": sum(r["split"] == "val" for r in rows),
                     "capped_out": sum(e["split"] == "capped" for e in eps)},
        "tagger": {"call_mix_train": {k: round(v / tot_calls, 3) for k, v in sorted(calls.items(), key=lambda kv: -kv[1])},
                   "pytest_after_last_edit_train": sum(r["pytest_after_last_edit"] for r in tr),
                   "episodes_with_edit": sum(r["n_edits"] > 0 for r in rows), "span_mismatch": n_mismatch},
        "tokens_by_tag": tag_tok, "wrapt_lines_stripped_msgs": wrapt_stripped,
        "train_by": {k: {v: sum(r[k] == v for r in tr) for v in sorted({r[k] for r in tr})} for k in ("kind", "lib", "src")},
        "chunks": {"train": len(ch_tr), "val": len(chunks) - len(ch_tr),
                   "prefill_calls": {int(v): int((ch[:, 3] == v).sum()) for v in sorted(set(ch[:, 3].tolist()))},
                   "P_median": int(np.median(ch[:, 1])), "P_max": int(ch[:, 1].max()),
                   "S_median": int(np.median(ch[:, 2] - ch[:, 1])), "S_max": int((ch[:, 2] - ch[:, 1]).max())},
        "train_tokens": {"targets_S": int(sum(c[2] - c[1] for c in ch_tr)), "prefill_P": int(sum(c[1] for c in ch_tr)),
                         "weighted_trained": round(float(sum(w_all[c[0]][c[1]:c[2]].astype(np.float32).sum() for c in ch_tr)), 0)},
    }
    (out_dir / "rs2_report.json").write_text(json.dumps(report, indent=1))
    print("RS2 REPORT", json.dumps(report), flush=True)
