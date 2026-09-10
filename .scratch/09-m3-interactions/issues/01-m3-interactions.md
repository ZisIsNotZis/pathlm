# 01 — M3 interaction runs + diversity-training probe

- **Status:** claimed
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
