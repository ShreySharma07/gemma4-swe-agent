"""
Decision structure of training episodes (S3b design check, Oct 18): what the NEXT call is after a failed test, after a
passing test, and how many calls precede the first edit. The behaviour gate scores exactly these decisions, so the
data must make them depend on the observation (S3: every test passed and was followed by submit).
Usage: python posttrain/decision_stats.py TRACE_DIR [TRACE_DIR ...]
"""
import collections
import json
import re
import statistics
import sys
from pathlib import Path


def kind(fn, cmd):
    if fn != "run_command":
        return fn
    c = re.sub(r"^(#[^\n]*\n)+", "", cmd.strip())
    c = re.sub(r"^cd /workspace\s*&&\s*", "", c)
    if "PYEOF" in c and "write_text(" in c:
        return "edit"
    if re.match(r"(python3? -m )?pytest|python3? -m unittest", c):
        return "test"
    if re.match(r"(sed -n|cat|grep|find|ls|rg|head|tail)\b", c):  # "cat \b" never matched `cat -n` (fixed Oct 18)
        return "read"
    return "other"


def failed(content):
    try:
        d = json.loads(content)
    except Exception:  # noqa: BLE001
        return False
    return d.get("status") != "ok" or bool(re.search(r"\b\d+ failed\b|\bFAILED\b|\d+ errors?\b", d.get("stdout", "")))


def main():
    for d in sys.argv[1:]:
        after = {t: collections.Counter() for t in ("test_failed", "test_passed", "pre_edit_test_failed", "pre_edit_test_passed")}
        before_edit, n, with_fail = [], 0, 0
        for f in sorted(Path(d).glob("trace_*.json")):
            steps = json.loads(f.read_text())["steps"]
            resp = {r["source_call_id"]: r["content"] for s in steps for r in (s.get("observation") or {}).get("results", [])}
            seq = [(kind(tc["function_name"], str((tc.get("arguments") or {}).get("command", ""))), resp.get(tc["tool_call_id"], ""))
                   for s in steps for tc in s.get("tool_calls") or []]
            ks = [k for k, _ in seq]
            n += 1
            if "edit" in ks:
                before_edit.append(ks.index("edit"))
            fail_here = False
            for i, (k, r) in enumerate(seq[:-1]):
                if k == "test":
                    t = ("" if "edit" in ks[:i] else "pre_edit_") + ("test_failed" if failed(r) else "test_passed")
                    fail_here |= t == "test_failed"
                    after[t][ks[i + 1]] += 1
            with_fail += fail_here
        print(f"{d}: {n} episodes, {with_fail} with a failed test after an edit; calls before first edit: median "
              f"{statistics.median(before_edit) if before_edit else None}, p25/p75 "
              f"{statistics.quantiles(before_edit, n=4)[::2] if len(before_edit) > 3 else None}")
        for t, c in after.items():
            print(f"   next call after {t}: {dict(c.most_common())}")


if __name__ == "__main__":
    main()
