#!/bin/sh
# Assemble the F3 kernel: harness header + factory_core + f3_main + factory_main's verification tail -> kernel_f3/.
# Then: .venv/bin/kaggle kernels push -p kaggle/factory/kernel_f3
set -e
cd "$(dirname "$0")"
mkdir -p kernel_f3
python3 - > kernel_f3/build_factory.py <<'PY'
import ast, pathlib
print(pathlib.Path("harness_common.py").read_text())
src = pathlib.Path("factory_core.py").read_text()
print("# " + "-" * 66 + " factory_core.py")
print("".join(src.splitlines(keepends=True)[ast.parse(src).body[1].lineno - 1:]))
print(pathlib.Path("f3_main.py").read_text())
main = pathlib.Path("factory_main.py").read_text()
marker = 'meta = {t["instance_id"]: t.pop("_meta") for t in tasks}'
assert main.count(marker) == 1
print("# " + "-" * 66 + " factory_main.py (verification tail)")
print(main[main.index(marker):])
PY
python3 -c "import ast; ast.parse(open('kernel_f3/build_factory.py').read())"
cat > kernel_f3/kernel-metadata.json <<JSON
{
  "id": "shreysharma07/gemma4-task-factory-f3",
  "title": "gemma4-task-factory-f3",
  "code_file": "build_factory.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "competition_sources": ["gemma-4-developer-agent"],
  "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", "shreysharma07/test-deps-wheels"],
  "model_sources": [],
  "kernel_sources": []
}
JSON
echo "wrote kernel_f3/build_factory.py ($(wc -l < kernel_f3/build_factory.py) lines)"
