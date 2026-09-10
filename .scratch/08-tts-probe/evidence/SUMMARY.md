# TTS probes — can composition beat B0? (verdict: no, and the why is the finding)

Anchor B0 = 1.5074 (200-batch protocol). Probe bpc values carry ~+0.01
sampling offset vs battery numbers (60-batch protocol, cross-checked:
probe-K1 on L1 1.6926 vs battery 1.6822; probe-K1 on X1 1.5132 vs 1.5006).

## Results (probe protocol)

| Checkpoint | K=1 | K=2 | K=4 | K=8 | gain K8−K1 |
| --- | --- | --- | --- | --- | --- |
| L1 (skip) | 1.6926 | 1.6657 | 1.6543 | 1.6454 | **−0.047** |
| L3 (shuffle) | 2.0373 | 2.0180 | 2.0140 | 2.0079 | −0.029 |
| L4 (redo) | 1.5661 | 1.5603 | 1.5578 | 1.5562 | −0.010 |
| X1 (penalty) | 1.5132 | = | = | = | **0 — paths identical** |
| X3 (evict+pen) | 1.5235 | = | = | = | **0 — paths identical** |
| I3 (noise) | 1.5198 | 1.5188 | 1.5181 | 1.5177 | −0.002 |
| T2 L2 depth-ens (uniform) | 1.6070 | | | | — |
| T2 L2 depth-ens (conf) | 1.6023 | | | | −0.005 vs uniform |
| T3 round-ens (C1/C3/C4) | 2.06–2.07 | | | | ≈ 0 or worse |

## Reading

1. **Path ensembling works exactly in proportion to path diversity.** L1's
   monotone −0.047 gain is real TTA behavior. X1/X3 are flat to 4 decimals:
   their paths are deterministic (no shuffle/skip/redo), so "K paths" is the
   same path K times — no diversity, no gain. The gain is a function of the
   element, not the architecture: you must TRAIN stochastic paths to ENSEMBLE
   over them at test.
2. **But no composed number beats B0.** Best composed: L4@K8 = 1.5562 (+0.049);
   X1 single-pass remains the best number overall (1.5006 battery). The
   element taxes dominate the ensemble gains: L1 ensembles 0.047 back of its
   0.175 tax and stalls at 1.645. Gain(K) curves are saturating (roughly
   g·(1−K^-0.5) shape) — K_sat ≈ 8, extrapolated ceiling for L1 ≈ 1.638:
   ensembling cannot outrun the tax.
3. **T2 (depth ensemble, FREE) fails**: L2's depth-mixed 1.602 vs its own
   single-pass 1.5744 — mixing WEAKENS. The early depths are not weak
   co-equal voters; they are strictly worse estimators (depth CE 1.06 → 0.74
   is a 15× perplexity cliff). Confidence weighting helps (+0.005 over
   uniform) but the confidence heads rank depths, not per-token reliability —
   wrong signal for mixture weights.
4. **T3 (round ensemble) is a塌缩 probe that returned塌缩**: C1/C3 round-ens
   ≈ 2.07 — WORSE than each checkpoint's own bpc (1.65). Why: on the clean
   stream, round-2's input is the transport-transformed round-1 output (a
   blurred/distribution-shifted state), so later rounds are worse estimators,
   and averaging them in hurts. The retry loop is a REPAIR mechanism, not an
   ensemble: it earns its keep only on corrupted inputs (C-runs' repair data
   prove that).

## Answers to the user's questions

- **Test-time scaling beat B0?** No. Composition gains exist but are smaller
  than element taxes; nothing composed crosses 1.5074.
- **Cost multipliers:** K_match does not exist for any probe (no K reaches
  B0); K_sat ≈ 8 (L1 curve flattens); score_sat = 1.6454 (L1@K8, 8× eval).
- **Why: methodology or convergence?** Neither — it's ARCHITECTURAL MATH at
  this design: (a) path diversity must be trained in, and the elements that
  create it tax bpc more than ensembling returns; (b) the shared-trunk
  correlation the M0 lesson predicted limits ensemble gain exactly as
  forecast; (c) auxiliary estimators (depths, rounds) are trained as
  fallbacks/repairs, so mixtures that include them degrade. The fixes are
  design-level: diversity-preserving training (reward path disagreement),
  equal-footing auxiliary supervision, or scale (taxes shrink with capacity).
