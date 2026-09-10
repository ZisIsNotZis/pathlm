# 01 — M1 follow-ups: mixture re-entry (C4/C5), retry curves, needle redesign (X2v2)

- **Status:** claimed
- **Type:** implementation + experiment
- **Blocked by:** 04-m1-runs/issues/01-m1-runs
- **need-review:** true
- **need-test-cases:** true

## Issue

Three follow-ups from the M1 results (user-approved design):

1. **C4 + C5 — mixture re-entry** (design.md §3 amendment): re-entry state is a
   confidence-weighted accumulator over all rounds (init anchor w=1, detached
   weights) instead of overwrite. C4 = soft + mixture; C5 = token retry +
   mixture (the R1 lesson fix — discrete re-embed of the accumulated
   distribution, no error commitment). Gate: post-retry repair ≥ base pass;
   compare retry-curves against C3/R1.
2. **Retry curves**: repair accuracy with the loop forced to 1–4 rounds at
   inference (no retraining) — the recurrent-transformer question. Applied to
   C1/C3 checkpoints and built into the battery for retry runs.
3. **X2v2 — needle redesign** (signal starvation fix): multi-needle batches on
   real-text filler. K=8 needles per row carved into a real data window; query
   tail [mask, x_i, y_i] × K; (xn, yn) pairs chosen absent from the row's text;
   anchor_frac oversamples the anchor regime. 16× supervised needle signal.
   Eval reports accuracy per regime bucket (in-window / beyond / anchor).

## Acceptance criteria

- [ ] Mixture re-entry implemented with fail-capable tests (anchor persistence, per-transport forms)
- [ ] retry_curve in eval battery for retry runs
- [ ] C4, C5, X2v2 runs complete with gates answered
- [ ] Retry curves for C1/C3/C4/C5 recorded

## Comments

- 2026-09-08 — agent (pi, volc2/glm-5.3-flash) — created from M1 results + user design discussion (mixture re-entry proposed by user); claimed.
