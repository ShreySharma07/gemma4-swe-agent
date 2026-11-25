# Final week (Nov 25 → Dec 2): how to use the last six LB slots

This plan follows `docs/strategy_20261119.md`. It is a recommendation; nothing here is decided until Shrey says so.

**The plan in three lines**
- No agent we can build this week has a plausible gain big enough for one LB reading to detect. So the slots go to more readings of the two E4B candidates, not to new variants.
- Most of those slots go to v32, the E4B with the T4 adapter. It runs at temperature 1.0, so every reading is a fresh draw, and it has the better local record.
- Finals: v17 for the 31B, plus whichever E4B family has the better mean, using its best single draw.

---

## 1. Where we stand (18:40 UTC Nov 25)

| Board | Top | Us | Gap |
|---|---|---|---|
| Main (31B) | 0.24 (18 teams) | v17 0.18 (22 of 120 tasks) | about 7 tasks |
| Small model (host data, Nov 24 23:56 UTC; 1,419 teams) | 0.12 (2 teams); 34 at 0.10, 178 at 0.08, 396 at 0.06 | v26 0.06, v32 0.06, v29 0.05 | 4–6 tasks |

**What happened since the Nov 19 plan**
- **31B:** v27 (temperature 1.0) scored 0.10 and v25 (thinking 2048) 0.13. The decode-diet ladder failed, so v17 is the 31B final.
- **E4B:** distillation (T3 → T4) moved local results from 4/24 to 6/24 (p = 0.63) but not the LB: v32 scored 0.06, the same as plain E4B (v26). This is the same weak local-to-LB transfer the 31B showed.
- **Runtime:** v32 finished inside the 12 h cap at a local mean of 191 s per task. Our 9 h runtime bar was too strict.

**What one LB reading can detect**
- At 0.06, the binomial standard deviation is about 2.6 tasks (0.02). At 0.18 it is about 4.2 tasks (0.035).
- One reading separates two variants only when they differ by about 5 tasks or more.
- Nothing we could build in six days has a plausible gain of 5 tasks:
  - v32 with `include_thoughts: false` targets v32's 8/24 overflows. On the 31B it ran 54% longer and resolved 0/6.
  - Cutting the output reservation to free context was tried on the 31B: v11 scored 0.10 and v15 crashed.
  - More calls would not help: v32 ran out of calls on only 3/24 tasks, and plain E4B submits with calls to spare.
  - T4's epoch-1 export has the same validation loss as its final export.

A slot used for a new variant would therefore produce a reading we can't interpret. A slot used for another reading of a candidate we already have helps us pick the finals.

## 2. Why repeat readings are worth a slot

Kaggle runs each submission once. The private board is what pays, and how it relates to the public run is still unread (the Evaluation tab, open since Nov 18). There are two cases:
- **(S) Private comes from the same run** as the public score, on the same tasks or a split of them. A lucky draw then carries over, so extra draws are worth having, and the best one should be the final.
- **(I) Private scores tasks the public run never saw**, or re-runs the finals. Draws then fall back to their family's mean, and extra readings only sharpen the means.

**Repeat readings help in both cases**, and they cost nothing beyond the slot: a bad draw is simply not picked. The final-pick rule in §4 works whichever case holds.

**Why v32 rather than plain E4B**
- **Fresh draws:** v26 runs at temperature 0.2. Our temperature-0.2 runs repeat their scores (the v17 family scored 0.18 on four readings), so another v26 reading may tell us little. v32 runs at temperature 1.0, so each reading is a genuine draw.
- **Better local record:** v32 resolved 6/24 locally against plain E4B's 4/24.
- **Proven runtime:** v32 has already finished inside the 12 h cap.

**Why no 31B draws:** the v17 family scored 0.17, 0.18, 0.18 and 0.18. At temperature 0.2 there is too little variance for a draw to close a 7-task gap.

Every repeat gets a one-word prompt change, so it is not byte-identical and counts as a fresh draw (rule from the Nov 19 plan). Each zip is compiled before it is submitted.

## 3. Slot calendar

Each run takes about 24 h to score, and only one can be pending, so each slot opens when the previous one scores. Every zip is built and compiled the day before.

| # | Opens (UTC) | Submission | Purpose |
|---|---|---|---|
| 1 | **Nov 25, before 23:59** | `v32r1` = v32 + one-word prompt change | Second reading of E4B + T4 adapter |
| 2 | Nov 26, about 19:00 | `v26r1` = v26 + one-word change | Second reading of plain E4B; tests whether it repeats its score |
| 3 | Nov 27 | `v32r2` | Third v32 reading |
| 4 | Nov 28 | `v32r3`, or `v26r2` if v26's mean leads | Whichever family leads on mean; v32 on a tie |
| 5 | Nov 29 | same rule | |
| 6 | Nov 30 | same rule | |
| (7) | Dec 1 | spare, only if a slot slipped | Scores around Dec 2 |
| — | Dec 2 | select the two finals | |

**Slot 1 must go in today.** The UTC day ends in about 5 h, and each lost day removes a slot. Building, compiling and zipping it takes about 20 minutes. No GPU is needed for any of this calendar.

## 4. Finals rule (to pre-register now)

- **31B final:** v17 (four readings at 0.17–0.18; nothing else comes close).
- **E4B final:** two steps.
  1. **Choose the family** with the higher mean over its readings: v26 family (v26, v26r…) or v32 family (v32, v32r…). Within 0.01 → v26, under the current rule; Shrey may choose v32 here on its local record.
  2. **Within that family, choose its best single draw.**

Under (S), step 2 picks the luckiest run we own in the stronger family. Under (I), every draw in a family has the same expected private score, so step 2 costs nothing. If the Evaluation tab shows (S), choose the best single E4B draw across both families instead.

## 5. Other work this week (no slots)

- **Open-source readiness.** Winners must release training and inference code under an OSI licence. To check:
  - the licence file
  - `trainkit` and the `rftset` builders run from the README
  - the data provenance notes
  - the submissions are reproducible from `submissions/`
- **Log each reading** in `docs/experiments.md` and `docs/HQ.md` as it lands, with the running mean per family.

## 6. Decisions needed from Shrey

1. **Approve the calendar**, including slot 1 (`v32r1`) today, before 23:59 UTC.
2. **How slots get submitted:**
   - *Auto-submit:* a local watcher submits the next pre-approved zip when the pending one scores. The Mac must stay awake.
   - *Manual:* you approve each slot as it opens. A slot that opens at about 19:00 UTC (00:30 IST) may then slip by hours.
3. **E4B tie-break** in §4 step 1: v26 (current rule) or v32.
4. **Read the Evaluation tab** (2 minutes): is the private board scored from the same run as the public board?
