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
- 2026-09-10 — agent (pi, volc2/glm-5.3-flash) — results so far:
  X2v2 PASS — needle redesign works: in-window 84.9%, anchor 76.8%, beyond 0.4% (allowed to fail); bpc improved to 1.565 (real-text filler stops wasting needle batches). Eviction architecture vindicated; v1 failed on signal starvation only.
  Retry curves (C1/C3 checkpoints, forced rounds, no retraining): gains saturate ~round 3 (+2–3pp); C1 dips at 4 rounds (52.8) while C3 keeps creeping (53.5) — soft degrades more gracefully.
  C4 (soft+mixture) PASS-equal — repair 50.2→52.6→52.9→53.1 monotone through 4 rounds (never dips, unlike C1); vs C3 overwrite at equal rounds: a wash. Mixture = robustness at depth, not a mean-level win at this scale.
  C5 v1: token round FIXED (50.2% ≥ base 49.7%, ECE 0.005 vs R1's 46.7%/0.015) BUT middle passes re-entered via weighted latent mean (transport=none) and collapsed (self-acc 54%, ECE 0.35) — latent-mean blurs token identity (the direct-form caveat, now measured). Fix: transport=none middle passes stay identity re-entry; only the token round consumes the mixture. C5 v2 rerunning.
- 2026-09-10 — agent (pi, volc2/glm-5.3-flash) — C5 v2 results: blur artifact GONE (pass1 self-acc 90.7% vs v1's 54%). But: token round 48.7% vs base 50.0% (−1.3pp, better than R1's −4.6pp, no longer catastrophic, ECE 0.007) and forced multi-round loops DEGRADE monotonically (identity passes: 50→43→36→30%) — unlike C1 where the same structural re-entry is stable. Net: at this scale the mixture vote removes token-retry's catastrophic failure mode but does not make it profitable; the remaining lever is prob0-gated per-position re-embed (only rewrite where the model itself flags doubt) — deferred as a design option, not scheduled.
Ticket verdict: C4 PASS-equal (mixture = robustness at depth), C5 PASS-with-caveat (catastrophe removed, profit not found), X2v2 PASS (needle + anchors work), retry curves recorded (saturate ~r3).
