# 01 — M3 interaction runs + diversity-training probe

- **Status:** done
- **Type:** experiment
- **Blocked by:** 08-tts-probe/issues/01-test-time-scaling
- **need-review:** true
- **need-test-cases:** false

## Issue

The program's closing questions:

**M3 interaction runs** (§3 table, IX2/INT included per milestone plan):

- IX1: best-I (I2, mask) × C3 (repair-in-context) — repair payoff over corruption alone
- IX2: L1+L2+L3+L4 all on — do depth knobs compose? gate: PPL vs sum of main effects
- IX3: L3 × C1 — order-free layers + latent loop — rounds curve monotonic
- IX4: C1 with w_consistency on — does the consistency loss carry chaining? chain-depth curve
- INT: all elements on — the actual Path LM — full battery

**Diversity probe** (TTS verdict follow-up): one training run with a
path-disagreement loss term — does rewarding path diversity convert TTA from
"tracks the tax" to "beats the tax"? Measured by the same K-sweep as ticket 08.

Configs: interaction runs are config-only (engine complete). Diversity loss
needs a small engine addition: w_diversity knob, loss term on node-1
probability divergence across the K sampled paths of the same batch.

## Acceptance criteria

- [ ] IX1–IX4 + INT runs complete with results.json
- [ ] Interaction deltas vs sum-of-parts table
- [ ] Diversity probe: K-sweep on the diversity-trained checkpoint vs L1
- [ ] Final program verdict in docs/experiments.md

## Comments

- 2026-09-10 — agent (pi) — created; claimed. Sequencing: IX runs first (config-only, zero engine risk), diversity probe second (engine change → test → run).
- 2026-09-10 — agent — IX configs committed and IX1–IX4+INT launched. Diversity engine change shipped TDD-first (w_diversity knob, capped −JS between parallel paths, partner-path sampling; caught n_mtp=0 KeyError via existing M0 test — 36 tests). DIVL1 (p_skip 0.1 + w_diversity 0.1) queued behind the IX runs.
- 2026-09-11 — agent — all runs complete: IX1 ✅ sub-additive (+0.087 < 0.277 sum), IX2/IX3/INT ❌ shuffle-poisoned (2.2+ bpc), IX4 ➖ neutral. DIVL1 v3 (after 2 instability-fix rounds: float32 JS, grad-norm guard): K-sweep 1.6842→1.6227, gain −0.062 > L1's −0.047, beats L1's K8 ceiling by 0.023 — **diversity mechanism confirmed, tax still dominates at 13M**. SUMMARY.md written.
- **Status:** done
