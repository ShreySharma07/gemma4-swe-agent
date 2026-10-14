"""Launcher: torchrun over the worker inside the scorer venv, with per-rank logs kept for post-mortems."""
import subprocess
import sys

from .config import OUT_DIR, WORK


def launch(model_path, data_path, nproc, timeout_s):
    logs = OUT_DIR / "rank_logs"
    logs.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(nproc), "--log-dir", str(logs),
           "--redirects", "3", "--tee", "3", "-m", "trainkit.worker", str(model_path), str(data_path)]
    print("$", " ".join(cmd), flush=True)
    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONPATH": str(WORK)}
    import os
    proc = subprocess.Popen(cmd, cwd=WORK, env={**os.environ, **env}, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
        return proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        return -9


def rank_tails(n=1500):
    """Last lines of each rank's stderr (a crashing rank's own traceback, which torchrun hides)."""
    out = {}
    for f in sorted((OUT_DIR / "rank_logs").glob("**/stderr.log")):
        t = f.read_text(errors="ignore")[-n:]
        if t.strip():
            out[str(f.relative_to(OUT_DIR))] = t
    return out
