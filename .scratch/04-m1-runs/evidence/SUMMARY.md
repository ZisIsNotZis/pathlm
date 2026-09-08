# M1 results — 7 runs, ~33 min total, 98M tokens each (1.1 epochs)

| Run | bpc (clean) | Δ vs B0 | Repair (corrupted pos) | Gate | Verdict |
|---|---|---|---|---|---|
| B0 | **1.5074** | — | — | bpc in published 1.4–1.6 range | **PASS** — external anchor holds; always-on MTP cost ≈ 0 |
| I1 | 1.6340 | +0.127 | **52.8%** (chance ~0.5%) | capacity tax modest; repair works | **PASS** |
| C1 | 1.6549 | +0.148 | 50.7% → **52.8%** (+2.1pp) | retry curve at depth | **PASS-marginal** — direct retry is NOT a no-op at 12L (corrects the M0 extrapolation: deep stacks do not reach a fixed point in one pass) |
| C3 | 1.6578 | +0.150 | 49.6% → **52.1%** (+2.5pp) | soft ≥ direct? | **TIE** — soft ≈ C1 (+0.4pp); repair-in-context shows no clear win at this scale; ensemble +0.0012 |
| R1 | 1.6775 | +0.170 | 51.3% → **46.7%** (−4.6pp) | token retry helps? | **FAIL** — hard discrete re-entry commits errors; conf degrades (ECE 0.001→0.015); needs prob₀-gated firing |
| L2 | 1.5744 | +0.067 | depth curve 1.062→0.742, plateau at ~depth 6 | exit payoff | **PASS** — ~half the depth within ~2% of full; exit threshold gives 1.4× decode speed |
| X2 | 1.6287 | +0.121 | needle acc **0.0 everywhere** (incl. in-window) | needle must not collapse | **FAIL** — needle convention not learned; training signal too sparse (1 query token per needle batch, 10% of batches) |

Decode speed: exit threshold 1.4× on every model (42→60 tok/s greedy); windowed X2 decode ≈ full attention speed at this scale (the win is bounded cache, not FLOPs here).

## Notes
- C1 vs M0: the "direct retry is information-free" argument holds per-operator, but a 12-layer stack does not converge in one pass — the +2.1pp repair gain is real and the retry×depth question survives.
- C3 ≈ C1: decorrelation via soft transport not confirmed; the correlated-voter concern stands.
- R1's failure mode is a design input for gating: token retry should fire only when prob₀ is low (inference) and the round should be trained conditionally, not always.
- X2's needle failure is a training-signal problem (1 supervised token per needle batch vs 98M tokens of natural text), not yet evidence against eviction. Redesign options: more queries per row, higher needle rate, real-text filler, curriculum on distance.
