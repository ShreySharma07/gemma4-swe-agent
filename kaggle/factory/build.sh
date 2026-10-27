#!/bin/sh
# Assemble the factory kernel script: shared harness header + core + driver -> kernel/build_factory.py.
# Then: .venv/bin/kaggle kernels push -p kaggle/factory/kernel
set -e
cd "$(dirname "$0")"
mkdir -p kernel
{
  cat harness_common.py
  echo
  python3 - <<'EOF'
import ast, pathlib
src = pathlib.Path("factory_core.py").read_text()
mod = ast.parse(src)
body = src.splitlines(keepends=True)[mod.body[1].lineno - 1:]  # drop the module docstring, keep imports
print("# " + "-" * 66 + " factory_core.py")
print("".join(body))
EOF
  echo
  cat factory_main.py
} > kernel/build_factory.py
python3 -c "import ast; ast.parse(open('kernel/build_factory.py').read())"
cat > kernel/kernel-metadata.json <<'EOF'
{
  "id": "shreysharma07/gemma4-task-factory",
  "title": "gemma4-task-factory",
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
EOF
echo "wrote kernel/build_factory.py ($(wc -l < kernel/build_factory.py) lines)"
