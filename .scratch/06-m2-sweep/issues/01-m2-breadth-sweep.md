# 01 — M2 breadth sweep: I2, I3, I4, C2, L1, L3, L4

- **Status:** claimed
- **Type:** experiment
- **Blocked by:** 04-m1-runs/issues/01-m1-runs
- **need-review:** true
- **need-test-cases:** false

## Issue

The remaining main effects, all config-only (the ticket-03 goal): I2 `[mask]`
corruption, I3 embedding Gaussian noise, I4 pure-noise latent, C2 linear
transport + retry, L1 p_skip, L3 shuffle-locality 0.5, L4 p_redo. Gates per
the §2 table (PPL deltas vs B0's 1.5074; rounds curve for C2; locality sweep
for L3).

Eval change shipped with this ticket: repair() measures under the run's own
corruption element (was: forced wrong-token) — needed for I2/I3/I4; noise-only
runs report self_acc_all only (no flagged positions). locality_sweep added for
L3 (eval-time locality 0/0.25/0.5/1.0).

## Acceptance criteria

- [ ] 7 runs complete, results.json per run
- [ ] Summary table with gates vs B0
- [ ] C2 rounds curve recorded

## Comments

- 2026-09-10 — agent (pi, volc2/glm-5.3-flash) — created; claimed; configs + eval fix committed.
