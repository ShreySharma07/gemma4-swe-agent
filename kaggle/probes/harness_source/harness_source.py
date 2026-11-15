"""CPU probe: copy the Python sources of the scorer harness (swegemma, adk_submission) out of the competition
wheelhouse so they can be read. No install, no model. Output: harness_src.zip (sources) + index.json (file sizes)."""
import glob, json, pathlib, zipfile

W = pathlib.Path("/kaggle/working")
index = {}
with zipfile.ZipFile(W / "harness_src.zip", "w", zipfile.ZIP_DEFLATED) as out:
    for pat in ("swegemma-*.whl", "adk_submission-*.whl", "adk-submission-*.whl"):
        for whl in sorted(glob.glob(f"/kaggle/input/**/{pat}", recursive=True)):
            name = pathlib.Path(whl).name
            with zipfile.ZipFile(whl) as z:
                for n in z.namelist():
                    if n.endswith((".py", ".md", ".yaml", ".yml", ".json", ".txt", "METADATA")):
                        data = z.read(n)
                        out.writestr(f"{name}/{n}", data)
                        index[f"{name}/{n}"] = len(data)
(W / "index.json").write_text(json.dumps(index, indent=1))
print(len(index), "files", sum(index.values()), "bytes")
