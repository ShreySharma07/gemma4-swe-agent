"""Submit the next LB candidate as soon as a pending submission scores, choosing between two pre-built branches.

The LB allows one pending submission per team and one per UTC day, and a run takes about 24 h to score, so every
hour between a score landing and the next submit is lost. This waits for <ref> to leave PENDING, reads its public
score and submits --hi (score >= --threshold) or --lo (lower score, or no score = timeout/error).

Usage (from the repo root, Kaggle CLI logged in):
  python3 tools/submit_when_scored.py --ref 58540046 --threshold 0.15 \
      --hi dist/v28_diet/submission.zip --hi-msg "v28_diet: ..." \
      --lo dist/v27_temp10/submission.zip --lo-msg "v27_temp10: ..." [--dry] [--hours 20]
Only run it after Shrey has said yes to the submission(s) it can make. --dry decides and prints, never submits.
"""
import argparse
import csv
import io
import pathlib
import subprocess
import sys
import time

COMP = "gemma-4-developer-agent"


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), *a, flush=True)


def submissions():
    out = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "--csv"],
                         capture_output=True, text=True, timeout=120).stdout
    lines = [l for l in out.splitlines() if l and not l.startswith("Next Page Token")]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--hi", required=True)
    ap.add_argument("--hi-msg", required=True)
    ap.add_argument("--lo", required=True)
    ap.add_argument("--lo-msg", required=True)
    ap.add_argument("--hours", type=float, default=20)
    ap.add_argument("--every-min", type=float, default=5)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    for z in (a.hi, a.lo):
        assert pathlib.Path(z).name == "submission.zip" and pathlib.Path(z).exists(), f"{z}: must exist and be named submission.zip"
    deadline = time.time() + a.hours * 3600
    while time.time() < deadline:
        try:
            rows = submissions()
        except Exception as e:  # noqa: BLE001
            log("CLI error:", type(e).__name__, e)
            time.sleep(60)
            continue
        mine = [r for r in rows if str(r.get("ref")) == str(a.ref)]
        pending = [r for r in rows if "PENDING" in str(r.get("status"))]
        if not mine:
            log(f"ref {a.ref} not in the listing (CLI login? columns: {list(rows[0]) if rows else []})")
        elif not pending:
            raw = (mine[0].get("publicScore") or "").strip()
            score = float(raw) if raw else None
            hi = score is not None and score >= a.threshold
            zip_, msg = (a.hi, a.hi_msg) if hi else (a.lo, a.lo_msg)
            log(f"ref {a.ref}: status {mine[0].get('status')}, score {score} -> {'HI' if hi else 'LO'}: {zip_}")
            if a.dry:
                log("dry run: not submitting")
                return 0
            r = subprocess.run(["kaggle", "competitions", "submit", "-c", COMP, "-f", zip_, "-m", msg],
                               capture_output=True, text=True, timeout=600)
            log("submit rc", r.returncode, (r.stdout + r.stderr).strip()[-600:])
            return r.returncode
        else:
            log(f"pending: {[p.get('ref') for p in pending]}; waiting")
        time.sleep(a.every_min * 60)
    log("gave up: still pending at the deadline")
    return 1


if __name__ == "__main__":
    sys.exit(main())
