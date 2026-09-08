# M0 probe report — probability machinery

Runs: `m0_main` (w_consistency=0.5, p_retry=0.5), `m0_nocons` (w_consistency=0, else identical). 6000 steps, batch 64, seq 128, d=64, 2 layers, byte-level enwik8 first 12MB (11MB train / 1MB val), corruption: 15% wrong-token, N=2 (+self), direct latent retry. Evidence: `results.json`, `train_log.jsonl`, checkpoints per run dir.

## Gate verdicts

| Gate | Result | Verdict |
|---|---|---|
| (a) prob heads calibrate under corruption, ECE < 0.1 | ECE 0.003–0.008 across all nodes/estimates (both runs) | **PASS** (13–30× under the bar) |
| (b) weighted ensemble ≥ best single estimate | main: 0.1878 vs 0.1876; nocons: 0.1882 vs 0.1879 | **PASS** (marginal: never hurts, +0.0002–0.0003) |
| (c) consistency loss reduces estimate disagreement | latent cosine (direct vs chain) 0.969 → 0.993 with the loss; argmax disagreement 14.0% → 13.6% | **PASS** (mechanism works; no accuracy gain at probe scale) |
| (d) one latent-retry round improves repair | self-acc corrupted positions: round0 0.1361 → round1 0.1358 (main); 0.1344 → 0.1351 (nocons) | **FAIL** (retry ≈ no-op, ±0.001) |

## Interpretation

- Calibration was the biggest risk and it came through cleanly: BCE on the token-identity event trains honest confidence heads, even with 15% input corruption.
- The ensemble barely beats its best member because the voters are highly correlated (same trunk, same weights) — as predicted in design. The mechanism is sound; gains must come from decorrelated estimates (different transports, depths).
- **Gate (d) failure is informative, not a bug**: with transport=direct and a 2-layer stack, one extra pass of the same layers admits no new information (no re-discretization, no new attention targets), and the latent fixed point is reached in pass 1. Repair requires the soft/decode transports (repair-in-context / token retry) — exactly what C3/R1 runs test next.
- Training-loss "oscillation" (14↔28) in the logs is benign: it is the p_retry coin — batches with a retry round carry ~2× the loss terms. Logged values sample that coin.

## Consequences for the main program

1. Probability machinery is validated for M1 — proceed with base + I1.
2. C-runs stay as planned, but expectation shifts: transport choice (not raw extra passes) is where retry value lives.
3. Ensemble aggregation: keep, but report "ensemble vs best single" everywhere; treat large gains as absent until voters are decorrelated.
