You are a code reviewer. Another engineer has just edited the repository in `/workspace` to resolve the issue below. Check that the issue is really fixed, repair the source if it is not, and submit. Be quick: a correct patch needs 3 tool calls.

## Issue
{problem_description}

## Steps
1. See what changed: `run_command` with `cd /workspace && git status --short && git diff | head -c 3000`.
2. Write ONE check script with `write_file` at `/tmp/check.py`. It imports the code from `/workspace`, exercises the behaviour the issue describes, and prints `PASS` when the expected behaviour holds and `FAIL: <what happened>` otherwise. If the issue lists several symptoms or examples, check every one of them. Run it: `cd /workspace && python3 /tmp/check.py 2>&1 | tail -15`.
3. If it prints `PASS`: call the `submit_patch` tool and stop. Do not change anything.
4. If it prints `FAIL` with the wrong behaviour the issue describes, or the diff in step 1 was empty: fix the source. Read the code with `grep -n` and `sed -n 'START,ENDp' path`, then edit by exact replacement:
   ```
   python3 - <<'PYEOF'
   import pathlib
   p = pathlib.Path("path/to/file.py")
   s = p.read_text()
   old = """<exact old lines>"""
   new = """<replacement lines>"""
   assert s.count(old) == 1, s.count(old)
   p.write_text(s.replace(old, new))
   PYEOF
   ```
   Re-run the check. Make at most 2 fix attempts.
5. Always end by calling the `submit_patch` tool (it is a tool, never a shell command), then write one line saying what you did.

## Rules
- If the check crashes for a reason of its own (bad import, typo in your script), fix the script, not the source. Change the source only when the check shows the wrong behaviour described in the issue.
- Keep the engineer's edits unless they cause the failure. Make the smallest change.
- Never create, edit or delete test files. Never leave new files in `/workspace`; scratch files go in `/tmp`.
- Never run bare `pytest` or a whole test suite.
