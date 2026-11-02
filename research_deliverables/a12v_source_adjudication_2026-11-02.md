# A12v source-level read of Atul's ten hybrids (Nov 2, 2026)

**Result:** 8 of the 10 hybrids fail because of an error inside the agent's own function. The other two are one message-wording failure (fastapi_14479) and one composition failure (fastapi_14609). One of the eight (rich_3676) also breaks an interface that gold's other sites rely on. The registered 43/51 is unchanged; this is a read of Atul's deliberately distinct-task sample, not an estimate for all 51.

**Reader and blinding.** Claude read these on Shrey's request, after Atul's tail-only read and with the stored labels visible, so the read is not blinded. Atul should check each verdict against the evidence named below.

**Method.**
- **Inputs:** for each case, the hybrid patch and the gold patch (from `data/runs/a12v/siteswap2/patches/` and `v2_gold_rerun/patches/`, both archived in `results/runs/a12v/hybrid_patches.tgz` and the A12v run). A `diff -u` of gold patch against hybrid patch shows exactly what the swap replaced. That replaced code was read against the failing assertions in the stored tail, and against the full log where the tail was cut.
- **Execution:** nothing was rerun.
- **Verdicts:**
  - *intrinsic*: the agent's function itself computes the wrong thing that the failing assertion checks;
  - *wording*: the logic matches gold and only a tested message string differs;
  - *composition*: the failure comes from the agent's function calling into code that gold's other sites changed;
  - *interface*: the agent's function violates a type or contract that gold's other sites depend on.

| Case (patch_file) | Swapped function | What the agent's function does differently | Failing evidence | Verdict |
|---|---|---|---|---|
| `104_v7_old54_rep3_fastapi_14986` | `FastAPI.setup` / inner `openapi` | Drops root_path from the schema entirely (adds only a comment). Gold injects it into a per-request copy. | `test_legitimate_root_path_still_appears`: `'/api/v1' in []` | intrinsic |
| `018_v14_new60_requests_7328` | `SessionRedirectMixin.resolve_redirects` | Removes the `resp.history = hist[1:]` assignment without adding gold's `resp.history = hist[:]`, so `history` is never set. It also rewrites the max-redirect check. | `len(e.response.history) == 5` gets 0, plus 2 more redirect-history tests | intrinsic (Atul: inconclusive from the tail; the source settles it) |
| `034_v7_new60_fastapi_14455` | `get_openapi_security_definitions` | Merges OAuth2 scopes inside the component definitions but still appends one operation-security entry per requirement. Gold merges per scheme at operation level. | operation `security` has 3 entries where 1 is expected. The `ERROR` is inline-snapshot's teardown echo of the same mismatch, not a second fault. | intrinsic |
| `098_v7_old54_rep2_rich_3506` | `Segment._split_cells` | Position estimate `int(cut/cell_length*len(text))` without gold's `- 1` and clamp | `cell_len('早乙女 ') == 5` gets 7; emoji split case | intrinsic |
| `121_v7_old54_rep4_rich_3061` | `Text.expand_tabs` | Keeps character-count tab positions and only changes the padding's style. Gold rewrote it to count cell width. | wide-character tab cases produce 4 spaces instead of 8; span ends wrong (5 cases) | intrinsic |
| `014_v14_new60_fastapi_14609` | `create_model_field` | Keeps the pydantic-v1 branch and calls `v1.ModelField`, which gold's other edits remove from `fastapi._compat.v1`. It also tests `isinstance(field_info, FieldInfo)` against v2's `FieldInfo`. | `AttributeError: module 'fastapi._compat.v1' has no attribute 'ModelField'` (2 failed, 2 errors) | composition; already flagged `posthoc_reverse_coupling` |
| `113_v7_old54_rep3_rich_4006` | `split_graphemes` | Skips a leading zero-width joiner, variation selector or control character (`index += 1`) instead of giving it a zero-width span | `[] == [(0, 1, 0)]` for `\x1b`, `️`, `‍` and their doubles (6 cases) | intrinsic |
| `006_v14_new60_fastapi_14479` | `analyze_param` | Same assertion and condition as gold; only the message differs ("Query param … must be of one of" vs "Query parameter … must be one of") | 4 `pytest.raises(match=…)` failures | wording |
| `124_v7_old54_rep4_rich_3675` | `Console.is_terminal` | Handles `TTY_COMPATIBLE == "1"` only, not `"0"`, and checks it after `FORCE_COLOR` | `TTY_COMPATIBLE=0` on a TTY still reports a terminal | intrinsic |
| `100_v7_old54_rep2_rich_3676` | `Traceback.extract` | Stores `notes` as `str(list)`, or `None` when absent. Gold stores a list, and gold's renderer does `for note in stack.notes`. | `test_notes`: `"['Hello', 'World']" == ['Hello', 'World']` (intrinsic). The other 12 tests hit `TypeError: 'NoneType' object is not iterable` in gold's renderer (interface). | intrinsic + interface |

**Tally.**

| Verdict | Count |
|---|---|
| intrinsic | 7 |
| intrinsic + interface | 1 |
| wording only | 1 |
| composition | 1 |

Atul's tail read agreed on 104, 034, 098, 121, 113 and 124, and on the concerns about 014 and 006. The source read settles 018 (intrinsic, where he had it as inconclusive) and 100 (intrinsic, with an added interface mismatch).

**Two more gaps in the static screens, beyond Atul's three synthetic ones:**
1. **Attribute of an imported module.** In fastapi_14609 the screens resolve the name `v1`, but not `v1.ModelField`, which gold's tree no longer has.
2. **Value or type contracts of fields** that gold's other sites consume. In rich_3676 the field `notes` is `None`/`str` in the agent's version, but gold's renderer needs a list.

Both are composition or interface failures that a passing static screen does not rule out.

**What this does not establish.**
- Verdicts are judged against gold's design and the target tests. An "intrinsic" error means the agent's function fails what those tests check, not that no other design could pass.
- Ten distinct-task cases chosen by Atul's hash rule are not a random sample of the 51 swaps, so no corrected rate is computed from them.
