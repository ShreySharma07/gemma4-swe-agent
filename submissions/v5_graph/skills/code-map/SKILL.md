---
name: code-map
description: Structural code lookup with Python ast. Find where a symbol is defined (with its exact graph node name), who calls it, what it calls, and a file outline. Works with bare names like solve_dependencies.
---
# code-map

Run with `run_skill_script(skill_name="code-map", file_path="scripts/code_map.py", args=[COMMAND, NAME])`.
`args` is a list of strings.

| Command | args example | Returns |
|---|---|---|
| find | `["find", "solve_dependencies"]` | qualified node name, `file:start-end`, signature |
| callers | `["callers", "solve_dependencies"]` | every call site: `file:line`, enclosing function, source line |
| callees | `["callees", "solve_dependencies"]` | names called inside that function |
| outline | `["outline", "fastapi/routing.py"]` | classes/functions of one file with line ranges |

- Add `"--tests"` to include test files; add `"--limit", "50"` for more rows.
- NAME can be bare (`get_dependant`), `Class.method`, or a dotted path.
- The qualified name printed by `find` (e.g. `fastapi.dependencies.utils.solve_dependencies`) is the exact `node` to pass to `get_code_neighbors` and `search_similar_code`.
