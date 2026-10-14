"""Logger that writes to stdout and /kaggle/working/run.log, plus a logged shell helper. Stdlib only."""
import glob
import subprocess
import time

from .config import WORK


class Log:
    def __init__(self, path=WORK / "run.log"):
        self.f = open(path, "a", buffering=1)

    def __call__(self, *a):
        s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")

    def sh(self, cmd, timeout=None, check=False):
        self("$", cmd[:220])
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = ((r.stdout or "")[-1500:] + ("\n" + r.stderr[-1500:] if (r.stderr or "").strip() else "")).strip()
        if out:
            self(out)
        if check and r.returncode != 0:
            raise SystemExit(f"FAILED rc={r.returncode}: {cmd[:120]}")
        return r


log = Log()


def first(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None
