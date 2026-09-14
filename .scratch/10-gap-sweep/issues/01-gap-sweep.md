# 01 — Gap-filling sweep: the unmeasured cost/gain cells

- **Status:** claimed
- **Type:** experiment batch
- **Blocked by:** 09-m3-interactions/issues/01-m3-interactions
- **need-review:** true
- **need-test-cases:** false

## Issue

User-directed brainstorm outcome: complete the measurement matrix left open by
tickets 04–09, batched to respect GPU efficiency (single training runs use
~20% SM — small d=256 GEMMs; we batch what we can but keep runs comparable
to the existing table: same steps/batch/lr/seed, tokens-matched).

### Batch 1 — eval-only (no retraining, existing checkpoints)

- E1 Redo eval-ablation: L4 with p_redo forced 0 vs trained at eval (same
  seeds) → redo's inference-time contribution
- E2 Redo refinement probe: CE through the frozen head after 1× vs 2×
  execution of the same layer on the same input → does the repeat refine?
- E3 Skip eval-ablation: L1 with p_skip forced 0 vs trained → skip's
  inference-time contribution

### Batch 2 — training runs (config-only, engine complete)

- T1 Retry matrix fill: C1M (direct×mixture) + C2M (linear×mixture) — the two
  unmeasured cells. Prediction on record: linear×mixture fixes C2's r4 dip
  (C4's monotonicity argument); direct×mixture tests latent-mean-only
  sufficiency.
- T2 MTP at scale: B2 run with n_mtp=2 → then offline: node-2 draft quality
  (M0 proxy: 0.2606), draft-accept-rate in decode (verify-by-next-forward),
  and probability composition (does node-2's t+2 estimate composed with
  node-1's chain beat either head alone?).

### Deferred (engine work first, recorded not scheduled)

- prob0-gated per-position retry; confidence-conditioned skip/redo;
  diversity dose-response (needs w sweep); locality-boundary proportionality.

## Acceptance criteria

- [ ] E1–E3 results recorded in findings.md (redo/skip gain columns filled)
- [ ] C1M/C2M results + matrix table completed in findings.md
- [ ] B2 (n_mtp=2) trained; accept-rate + composition verdict recorded

## Comments

- 2026-09-11 — agent (pi) — created; claimed. GPU note: single training runs use ~20% SM (39ms/step vs ~8.6ms ideal at 40-60 TFLOPS effective) — batch-size increase would break comparability with the existing table; logged as a known efficiency debt, tokens-matched vs steps-matched caveat recorded.
- 2026-09-11 — agent (pi) — WARNING, batch-2 T1 results invalidated. `C1M` and
  `C2M` were run without `n_mtp` set, so `PathConfig` defaulted them to 2 while
  every cell in the table they were added to pins 1; `B2` puts that knob at
  +0.041 bpc, ~78% of the reported C1M/C2M penalty. Controls at n_mtp=1:
  C1M_n1 1.6645, C2M_n1 1.6548/1.6686. On top of that, mixture re-entry
  applied the stage-E norm cap to stage-L states (re-entry norm 1.00 vs 28.7),
  and the accumulator is never trained past round 1 (`p_retry` is Bernoulli).
  Full finding + evidence + corrected conclusion:
  11-retry-matrix-validity/issues/01-retry-matrix-validity. The findings.md
  matrix has been corrected in place; the acceptance criterion "C1M/C2M
  results + matrix table completed in findings.md" was met but is superseded —
  this ticket's T1 should be treated as redone, not complete.
- 2026-09-11 — agent (pi) — process note: long runs must be launched detached
  (`setsid nohup … &`); the harness background wrapper is killed by a Pi
  session shutdown/reload (this truncated the first 24000-step horizon run).
