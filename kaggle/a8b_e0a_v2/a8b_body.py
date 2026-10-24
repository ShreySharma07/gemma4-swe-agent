
# ---------------------------------------------------------------------------------------------------------------
# A8b (pre-registered as POST HOC, Oct 24): E0a re-scored with tools/oblig.py v2 (the rules frozen for E0b).
# Inputs: output of kernel shreysharma07/gemma4-exec-graphs (xg/: tasks_meta, results, traces, src, graphs).
# PAYLOAD = tools/oblig.py + analysis/{e0a_obligation_recall,exec_graph_analysis,obligation_pilot}.py.
# ---------------------------------------------------------------------------------------------------------------
import base64
import glob
import io
import pathlib
import runpy
import sys
import tarfile

WORK = pathlib.Path("/kaggle/working")
IN = WORK / "in"
IN.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as tf:
    tf.extractall(IN, filter="data")
meta = sorted(glob.glob("/kaggle/input/**/xg/tasks_meta.jsonl", recursive=True))
assert meta, "attach the output of kernel gemma4-exec-graphs"
xg = str(pathlib.Path(meta[0]).parent)
print("xg dir:", xg)
sys.path.insert(0, str(IN / "tools"))
sys.path.insert(0, str(IN / "analysis"))
import oblig  # noqa: E402
print("oblig rules:", oblig.RULES)
sys.argv = ["e0a_obligation_recall.py", xg, "--json", str(WORK / "e0a_v2_rows.json")]
runpy.run_path(str(IN / "analysis" / "e0a_obligation_recall.py"), run_name="__main__")
