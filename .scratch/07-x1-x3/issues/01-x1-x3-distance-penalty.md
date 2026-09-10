# 01 — X1/X3: attention distance penalty (+ redundancy under eviction)

- **Status:** claimed
- **Type:** experiment
- **Blocked by:** 06-m2-sweep/issues/01-m2-breadth-sweep
- **need-review:** true
- **need-test-cases:** false

## Issue

The last two main-effect runs. X1 = attention distance penalty
(`dist_pen=0.1`, additive bias `-pen*log(1+d)`, correctness dominates) —
locality prior tax, gate: PPL delta + per-head distance telemetry (some heads
must stay long-range). X3 = X1 + X2 (window 128, anchors 8, p_needle 0.1 —
X2's exact settings) — is the penalty redundant under eviction training?
Gate: comparison vs X1/X2.

Engine change (small): Block.forward computes explicit attention when
dist_pen > 0 (softmax over causal + `-pen*log(1+d)` bias — one softmax gives
both the biased attention and the per-head mean-distance telemetry in
aux["attn_dist"]; dist_pen == 0 stays the exact SDPA no-op). 34 tests green.

## Acceptance criteria

- [ ] X1/X3 runs complete with results.json
- [ ] Per-head distance telemetry recorded for X1/X3
- [ ] X3-vs-{X1,X2} redundancy verdict

## Comments

- 2026-09-10 — agent (pi) — created; claimed; dist_pen engine change + telemetry test (TDD) + configs committed. Self-review note: telemetry shape [B, H] = layers-averaged per-head mean distance; X1/X2 gate parts are covered by bpc + attn_dist + (X3) needle_acc.
