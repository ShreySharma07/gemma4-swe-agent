# Training plan T2 (Nov 14 – Dec 2)

Written Nov 13, 2026. Budget: 30 GPU quota-h per week per account (Shrey + Atul = 60), reset on Saturdays
(Nov 14, 21, 28). A 4×L4 session costs about 2 quota-h per wall hour. Leaderboard runs do not use this quota.

## Where we stand

| | Score |
|---|---|
| Leaderboard top (Nov 13) | 0.29 |
| Large group of teams | 0.24 |
| Ours: v17 / v14 | 0.18 / 0.17 |

With about 120 hidden tasks, the gap to 0.24 is about 7 tasks and the gap to 0.29 is about 13.
No plan can promise the top. This one is built so that each GPU hour tests whether training moves resolves at all,
and stops early if it does not.

## What the model lacks (measured)

1. **It finds the code and writes the wrong fix.** R2: 105 of 251 failures submitted a wrong patch with calls left.
   A second Gemma pass does not catch these (v19 line, closed Nov 12).
2. **It does not finish multi-site fixes.** R3: singles 71%, pairs 26%, triples 13%.
3. **It needs the issue to name things.** R3: full issues 46%, terse 24%.
4. **It runs out of context.** 8–18 of 60 new-60 tasks end in a 32k overflow; an overflow is a lost task.
5. **It has no time to spare.** v14 is within 0.8 h of the 12 h cap; any adapter adds about 11% decode time
   (LT1), and v18 (v14 + adapter) timed out on the leaderboard.

## Why T1 did not help, and what changes

T1 trained on runs the model already solved. Validation loss fell 41% but resolves did not move
(42/95 vs 40/95, p 0.75) and overflows rose (12 → 18). Imitating your own successes teaches little that is new.
Epoch 2 added 0.4% validation gain while train loss fell 35%.

T2 therefore changes the **signal**, not the trainer:

- **A. More and harder successes (RS3, ready).** 391 train episodes, 137 multi-site (RS2: 203 / 50), 10 new libraries.
- **B. Corrected failures (RS4, to build; free CPU).** The factory tasks have known gold patches. For a failed run
  that reached the gold function, keep the model's own trajectory up to that point and append the correct edit(s)
  in v14's own edit format, the targeted test, and the submit. Only the appended steps carry loss. This is the only
  data we have that shows the model the right fix on tasks it gets wrong, and all sites of a multi-site fix
  (lacks 1 and 2). Tasks with terse issues are over-sampled (lack 3).
- **C. Shorter episodes (in RS4).** Steps already masked as waste (repeats, superseded submits, stray-file work)
  are removed from the context instead of kept, so the trained policy is shorter (lacks 4 and 5).
- **One epoch.** Epoch 2 memorised in T1.

Unchanged from T1: NF4 base, LoRA r16 α32 on 410 linears, lr 1e-4, full-context episodes ≤ 28,672 tokens,
toolz + pathspec held out. One change per arm.

## Steps

### Week 1 (Nov 14–20): does the new signal move resolves?

| Step | What | Cost (quota-h) | Gate to continue |
|---|---|---|---|
| H1 | CPU: count failed runs that reached the gold function, and how many gold edits can be rendered exactly | 0 | ≥ 150 usable corrected episodes |
| RS4 | CPU: render RS3 + corrected failures + trimmed contexts | 0 | no episode over 28,672 tokens; report matches pre-registration |
| T2a | Train 1 epoch on RS3 (control: more data only). Shrey | ≈ 10 | no NaN; val loss below T1's 0.0666 |
| T2b | Train 1 epoch on RS4 (the new signal). Atul | ≈ 14 | same, plus loss on corrected-edit steps reported separately |
| LP2 | Serving probe of both adapters on the W4A16 base | ≈ 2.5 | val gain ≥ 15% on the served model |
| G2 | Held-out gate, 3 arms (base, T2a, T2b) on the 95 toolz + pathspec tasks, one server, interleaved | ≈ 11 | see decision rule |

Total ≈ 38 of 60. The remaining ≈ 22 pays for the timing run below.

**Decision rule for G2 (pre-registered here):**
- best arm ≥ base + 6 and McNemar p < 0.1 → timing run, then leaderboard;
- +2 to +5 → one confirmation on new-60 before any leaderboard run;
- ≤ +1 on both arms → training on factory data does not move resolves; stop training, finals are v14 and v17.

**Timing run (only if G2 passes):** best adapter on new-60 at the scorer's conditions (concurrency 1, 8 min cap),
≈ 8 quota-h. It goes to the leaderboard only if the projected 120-task time is ≤ 11.0 h and overflows are not
above v14's. If it is too slow, cut `max_time_minutes` 8 → 7 or the output-token cap and re-time.

### Week 2 (Nov 21–27): leaderboard and one iteration

- Submit the adapter variant (1 run per day; each takes about 10 h on the scorer).
- One follow-up training run, chosen by what G2 showed: more corrected failures if T2b > T2a; rank 32 or a second
  rollout round on the hardest tasks if both gained; nothing if neither did.
- If the leaderboard score does not rise above 0.18 despite a local gain, the factory tasks do not transfer;
  stop spending on training.

### Week 3 (Nov 28 – Dec 2): finals

- Final 1: the best adapter variant with a verified scorer run.
- Final 2: v17 (0.18), which is known to finish in time.
- No new training after Nov 29: a final needs a full scorer run with a day to spare.

## Risk plan (added Nov 13, before anything runs)

Each risk gets a check that runs **before** the GPU hours it protects, and a rule for what happens if the check fails.

### Risk 1: gains on factory tasks do not transfer to the hidden tasks

Factory tasks are small injected bugs; the hidden tasks are real issues in private repos.

| # | Action | Cost | Rule |
|---|---|---|---|
| 1a | **F3: a real-bug gate set (CPU).** Mine real bug-fix commits with tests from small libraries that are in neither the factory nor new-60; keep those where the test fails before the fix and passes after. Used for evaluation only, never training | 0 | ≥ 40 verified tasks by Nov 17 → F3 joins G2. Fewer → fall back to 1b |
| 1b | **new-60 as the real-issue arm of the gate.** Memorisation is the same in both arms of a paired test, so the paired difference is still a real-issue signal. Two repeats (120 pairs), because one run varies by about ±5 | ≈ 16 quota-h, replaces the single confirmation run | adapter must not lose on real issues: paired difference ≥ 0 |
| 1c | **Tilt the training set toward what real issues look like.** In RS4, corrected episodes are sampled multi-site and terse/no-name issues first; single-operator fixes from one library are capped | 0 | reported in the RS4 pre-registration |
| 1d | **Ask the leaderboard early.** The first adapter that passes the gate and the timing check is submitted in week 2, day 1, not polished further. Leaderboard runs cost no quota | 0 | LB ≤ 0.18 with a local gain ≥ +6 → factory data does not transfer; stop training |

The G2 rule changes accordingly: an adapter needs **both** a gain on the held-out factory tasks **and** no loss on the real-issue arm (F3 or new-60 ×2).

### Risk 2: the appended correct steps are ours, not Gemma's

| # | Action | Cost | Rule |
|---|---|---|---|
| 2a | **Execute every corrected episode (CPU).** Replay the appended edit commands in the task snapshot; keep the episode only if the resulting diff equals the gold patch and the hidden tests pass | 0 | unverified episodes are dropped, not repaired |
| 2b | **Copy the format from Gemma.** The appended edit uses the heredoc exact-replace form v14 itself uses in resolved runs; the test command is the one the model already ran in that episode when there is one | 0 | — |
| 2c | **Do not teach "skip thinking".** Appended steps have no thinking text, so loss is put only on the tool-call arguments, not on the first tokens of the turn where the model chooses between thinking and calling | 0 | — |
| 2d | **Cap the share.** Corrected episodes carry at most 40% of the total training weight | 0 | — |
| 2e | **Two validation sets.** Held-out on-policy episodes (RS3 val) and held-out corrected episodes. T2b must not be worse than T2a on the on-policy set | inside T2b | T2b on-policy val more than 5% above T2a's → corrected steps hurt; G2 runs T2a only |
| 2f | **10-task behaviour smoke before G2.** Compare T2b with base on thinking tokens per step, malformed calls, calls per task and overflows | ≈ 2 quota-h | thinking per step down by more than half, or malformed calls up → fix the rendering before G2 |

T2a (RS3 only) stays as the control arm throughout.

### Risk 3: an adapter that resolves more but times out scores zero

Facts: v14 finishes the scorer with under 0.8 h to spare; an adapter adds about 11% decode time (LT1), projected
+0.8 h; v18 timed out. A shorter per-task cap is a weak lever: on v14's new-60 run a 6-minute cap saves 2% of
time and puts 3 resolved tasks at risk, a 5-minute cap saves 6% and risks 4. An output-token cut scored lower on
the leaderboard once (v11: 0.10 vs 0.15).

| # | Action | Cost | Rule |
|---|---|---|---|
| 3a | **One number decides.** Time the adapter variant and v14 on the same tasks at the scorer's settings (concurrency 1, 8 min). Ratio = adapter mean task time / v14 mean task time | ≈ 8 quota-h (the timing run already planned; run both arms interleaved on 30 tasks each if quota is short) | **ratio ≤ 1.03** → may be submitted. Above → 3b |
| 3b | **Offsets, in this order, one at a time:** (1) the shorter trajectories that RS4 trains for: read calls per task and decode tokens per task from G2; (2) `max_tool_calls` 28 → 25; (3) thinking budget 4096 → 2048; (4) 6-minute cap | 0 for (1); each of the others needs a re-time | stop at the first that brings the ratio to ≤ 1.03 without losing more than 2 of 60 |
| 3c | **Test the offset on the leaderboard without the adapter.** In week 1 the daily leaderboard slots are free. Submit v14 with offset (2), then (3), each as its own run | 0 quota | an offset that scores ≥ 0.15 and finishes ≥ 1 h earlier is safe to combine with an adapter |
| 3d | **Never risk both finals.** One final is always v17 or v14 with a completed scorer run | 0 | fixed |
| 3e | **First adapter submission is the timing probe.** It is submitted on a day when a timeout costs nothing but that day's slot, at least 5 days before Dec 2 | 0 | timeout → apply the next offset in 3b |

### What this adds to week 1

| | quota-h |
|---|---|
| Plan as written (H1, RS4, T2a, T2b, LP2, G2) | ≈ 38 |
| 2f behaviour smoke | ≈ 2 |
| Real-issue arm in G2 (F3, or new-60 ×2 in place of the confirmation run) | ≈ 8–16 |
| Timing run (3a) | ≈ 8 |
| **Total** | **≈ 56–64 of 60** |

That is the whole week's budget. If it runs short, the timing run moves to the first day of week 2 (reset Nov 21);
the gate is not cut.

## Risks (summary)

- **Synthetic tasks.** Factory tasks are small injected mutations; the hidden tasks are real issues in private repos.
  Local gains have failed to transfer before. G2 uses unseen libraries, but only the leaderboard measures transfer.
- **Corrected episodes are not on-policy.** The appended steps are ours, not Gemma's. They carry no thinking text;
  loss is on the tool calls only. T2a is the control that shows whether this helps or hurts.
- **Noise.** new-60 varies by about ±5 between identical runs. Decisions use paired tests on 95 tasks, not single runs.
- **Time.** An adapter that resolves more but times out scores 0. The timing run is not optional.
