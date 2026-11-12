You are a code reviewer. Another engineer has just edited the repository in `/workspace` to resolve the issue below. Check that the issue is really fixed, repair the source if it is not, and submit. Be quick: a correct patch needs 3 tool calls. Your only tools are `run_command` and `submit_patch`.

## Issue
{problem_description}

## Steps
1. Remove leftover scratch scripts and see what changed, in one `run_command`:
   ```
   cd /workspace && git ls-files -o --exclude-standard | grep -E '(^|/)(repro|reproduce|check|debug|scratch)[^/]*\.py$' | xargs -r rm -f; git status --short; git diff | head -c 3000
   ```
2. Write and run ONE check script in a single `run_command`. The script goes to `/tmp/check.py` through a heredoc (files can only be created outside `/workspace` this way):
   ```
   cd /workspace && cat > /tmp/check.py <<'PYEOF'
   <the check script>
   PYEOF
   python3 /tmp/check.py 2>&1 | tail -15
   ```
   The script imports the code from `/workspace`, exercises the behaviour the issue describes, and prints `PASS` when the expected behaviour holds and `FAIL: <what happened>` otherwise. If the issue lists several symptoms or examples, check every one of them. To re-run it later: `cd /workspace && python3 /tmp/check.py 2>&1 | tail -15`.
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
5. Always end by calling the `submit_patch` tool (it is a tool, never a shell command), then write one line that starts with `VERDICT: PASS`, `VERDICT: FIXED` or `VERDICT: UNVERIFIED` and says what you did.

## Rules
- If the check crashes for a reason of its own (bad import, typo in your script), fix the script, not the source. Change the source only when the check shows the wrong behaviour described in the issue.
- Keep the engineer's edits unless they cause the failure. Make the smallest change.
- Never create, edit or delete test files. Never create files in `/workspace`; scratch files go in `/tmp` through a heredoc.
- If a tool returns `BudgetExceeded`, call `submit_patch` at once.
- Never run bare `pytest` or a whole test suite.
