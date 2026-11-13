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

## Risks

- **Synthetic tasks.** Factory tasks are small injected mutations; the hidden tasks are real issues in private repos.
  Local gains have failed to transfer before. G2 uses unseen libraries, but only the leaderboard measures transfer.
- **Corrected episodes are not on-policy.** The appended steps are ours, not Gemma's. They carry no thinking text;
  loss is on the tool calls only. T2a is the control that shows whether this helps or hurts.
- **Noise.** new-60 varies by about ±5 between identical runs. Decisions use paired tests on 95 tasks, not single runs.
- **Time.** An adapter that resolves more but times out scores 0. The timing run is not optional.
