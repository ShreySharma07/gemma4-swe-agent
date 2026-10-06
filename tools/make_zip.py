"""Zip a submission directory for Kaggle (files at the zip root).

Usage: python tools/make_zip.py submissions/v2_sed [dist/submission_v2_sed.zip]
"""
import pathlib
import sys
import zipfile

src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else f"dist/submission_{src.name}.zip")
out.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(src.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            z.write(p, p.relative_to(src))
print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KiB)")
