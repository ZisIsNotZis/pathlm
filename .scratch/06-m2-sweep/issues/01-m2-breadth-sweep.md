# 01 — M2 breadth sweep: I2, I3, I4, C2, L1, L3, L4

- **Status:** done
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

- [x] 7 runs complete, results.json per run
- [x] Summary table with gates vs B0
- [x] C2 rounds curve recorded

## Comments

- 2026-09-10 — agent (pi, volc2/glm-5.3-flash) — created; claimed; configs + eval fix committed.
- 2026-09-10 — agent — first sweep killed silently mid-I4 (no OOM/reboot; bg task reaped by session recycle); I4..L4 relaunched and completed. Per-run exit lines in runner.log made the death detectable.
- 2026-09-10 — agent — results: I2 +0.059 (repair 59.4% > I1's 52.8% — flagged corruption cheaper AND more repairable), I3 ±0.000, I4 +0.063 (writable-input claim is cheap), C2 +0.155 (curve saturates r3, +2.0pp — linear ≈ direct ≈ soft), L1 +0.175, L3 +0.545 (locality sweep: tolerates ≤0.5, collapses 3.84 at 1.0 — order carries ~0.5 bpc), L4 +0.044. Element-tax ranking: identity/order-destroying elements (wrong-token, skip, shuffle) are the expensive class; identity-preserving ones (mask, noise, redo) are nearly free.
- 2026-09-10 — agent — review: independent reviewer unavailable (glm-5.3-flash account rate limit ~21h, then reviewer agent tool-contract failure on the switched provider). **User approved self-review in lieu** (recorded here). Self-review findings, all verified: (a) configs pure — exactly one element each vs PathConfig defaults, n_mtp=1 matches B0's scale deviation; (b) repair() signature change safe — eval_pc restores in finally, stage-0 mask partition exact (mask_pos/wrong_pos disjoint bands, noise opposite end of shared u1); (c) SUMMARY.md numbers match results.json; (d) bpc() zeroes all corruption knobs so cross-run PPL comparability holds. Findings: [P2] run_battery's repair gate misses perturb_noise/pure_noise — I4's noise self-acc unmeasured (fix next battery run); [P3] *.json configs missing trailing newline (cosmetic). Deviation: L1/L4 gates' quality/speed curves beyond decode_speed not produced — recorded here rather than silently dropped.
