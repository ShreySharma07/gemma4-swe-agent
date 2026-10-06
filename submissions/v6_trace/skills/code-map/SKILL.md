---
name: code-map
description: Read-only, relationship-first code navigation for Python repos - search with enclosing function, show lines, definitions, callers, callees, call paths, similar code and related tests.
---
# code-map

Call: `run_skill_script(skill_name="code-map", file_path="scripts/code_map.py", args=[...])`. `args` is a list of strings.

| Goal | args |
|---|---|
| Search text/regex; every hit shows its enclosing function | `["search", "PATTERN"]` |
| Read lines (max 100) | `["show", "path/file.py", "START", "END"]` |
| Where is X defined | `["find", "NAME"]` |
| Everything around X in one call: definition, callers, callees, overrides, tests | `["trace", "NAME"]` |
| Who calls X (2 levels up) | `["callers", "NAME"]` |
| What X calls | `["callees", "NAME"]` |
| How public API A reaches internal B | `["path", "A", "B"]` |
| Other places that may need the same fix | `["similar", "NAME"]` |
| Which tests cover X | `["tests", "NAME"]` |
| Map of one file | `["outline", "path/file.py"]` |

NAME can be bare (`solve_dependencies`), `Class.method`, or a dotted path. Add `"--tests"` to include test files.
