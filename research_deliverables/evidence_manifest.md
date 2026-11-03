# Evidence manifest: "Found, Not Fixed"

Each claim in the paper maps to a pre-registration commit, a result commit and the aggregate files in this repository. Pre-registration rows live in `docs/experiments.md`. Raw agent trajectories and full test logs are restricted competition material: they stay in the authors' Kaggle kernel outputs and are not published here.

| Paper section | Analysis | Pre-registered | Result | Public files |
|---|---|---|---|---|
| §3 exposure, size and multi-file | A1 | `e7f559f` | `9d99618` | `results/runs/a1/a1_report.json` |
| §3 necessity rows, size control | A13 (+A7/A2) | `92dabb6` | `09085fc` | `results/runs/a13/a1_a13_a7_a2_report.json` |
| §3 necessity labels; §4.1 completion | A3/A4 | `62e4068` | `7beef0e` | `results/runs/a3a4/{necessity_new60,groups,cases,results}.json[l]` |
| §4.2 site swap | A12v | `0841797` | `d932acd` (post-hoc flag `3d0d091`) | `results/runs/a12v/{swaps_v2.json,siteswap2_report.json,results.jsonl,hybrid_patches.tgz}` |
| §4.2 ten-hybrid source read | — (not pre-registered) | — | `cb23c49` | `research_deliverables/a12v_source_adjudication_2026-11-02.md` |
| §5 internal recall | E0a | same commit as result | `e8e9db3` | `results/runs/e0a/e0a_rows.json` |
| §5 internal recall, frozen v2 rules | A8b (post hoc) | `7ad401b` | `d5845c4` | `results/runs/a8b/e0a_v2_rows.json` |
| §5 external recall | E0b | same commit as result | `672bc30` | `results/runs/e0b/e0b_pairs.json` |
| §6 adapter gate | G1 | `5238741` | `8085156` | `results/runs/g1/l4_v1_report.json` |

## Deviations from the pre-registrations

1. **A4 completion.** The registered completion adds the missed gold hunks and gold's module-level hunks. In 5 of 26 cases that patch did not apply, so we retried without the module-level hunks. These fallbacks are reported separately in §4.1. The completion runs kept exit codes only, not test output, so their failures are untyped.
2. **A12 → A12v.** An internal audit found A12's "clean" label invalid, so A12v was registered as a replacement before it was built. A12's result is reported only as validation history.
3. **A12v identity control.** The registered gold→gold self-swap was implemented as a textual re-splice of the same tree, not a run of the swap builder on gold against gold. So it checks span bounds, not the builder path.
4. **A12v post-hoc flag.** The reverse-coupling flag (42/50) was added after the smoke run. It is labelled post hoc, and the decision rule does not use it.
5. **A12v reading.** The registered second read (Atul) used output tails. The later source-level read of ten hybrids was done by an unblinded reader and was not pre-registered.
6. **E0a, E0b.** Each prediction was committed together with its result, not before it.
7. **Populations.** The counterfactual populations pool v7 and v14 runs as the registrations specify, but 41 of the 51 primary swaps come from the five v7 runs on old-54. We report this imbalance (§4 table) rather than reweighting.
