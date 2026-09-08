# M0 probe report — probability machinery (v2, after review fixes)

Runs: `m0_main` (w_consistency=0.5, p_retry=0.5), `m0_nocons` (w_consistency=0, else identical). 6000 steps, batch 64, seq 128, d=64, 2 layers (learned positional embeddings), byte-level enwik8 first 12MB (11MB train / 1MB val), corruption: 15% wrong-token (never `[mask]`, never the original token), N=2 (+self node 0), direct latent retry, fresh layer path per round. Eval corruption is seed-controlled.

**Revision note:** v1 of this report was REJECTED in review. Fixed since: (1) node k now supervised on the clean token t_{i+k} with a true self node k=0 (v1 had an off-by-one that silently dropped the self node — the review's critical finding); (2) repair metric now aligned with the corruption mask; (3) gate-(c) dispersion metrics computed inside `evaluate()` and stored in `results.json` (v1's numbers came from an uncommitted ad-hoc script); (4) learned positional embeddings added (v1 had none — the "standard AR decoder" base was set-permutation-only); (5) shift-sensitive and memorization-proof tests; (6) ensemble weights are normalized calibrated confidences (not raw-logit softmax); (7) separate confidence head for the chained node-2 estimate; (8) no final LayerNorm, so latents keep the embedding geometry end-to-end.

## Gate verdicts

| Gate | Result (main / nocons) | Verdict |
|---|---|---|
| (a) prob heads calibrate under corruption, ECE < 0.1 | ECE 0.004–0.011 across all nodes/estimates, both runs | **PASS** (≥9× under the bar) |
| (b) weighted ensemble ≥ best single node-2 estimate | 0.2614 vs 0.2606 / 0.2617 vs 0.2605 | **PASS** (marginal: +0.0008/+0.0012, never hurts) |
| (c) consistency loss reduces estimate disagreement | latent cosine 0.9882 vs 0.9525 (w/ vs w/o loss); argmax disagreement 22.0% vs 22.8% | **PASS** (mechanism works; no accuracy gain at probe scale) |
| (d) one latent-retry round improves repair | self-acc on corrupted positions: 0.1889 → 0.1897 (main); 0.1891 → 0.1885 (nocons) | **FAIL** (retry ≈ no-op, ±0.001 — now measured on the correctly aligned metric) |

## Reading the numbers

- node0 (self/repair): 87.3% overall; 18.9% on corrupted positions (~0.5% is chance) — the self node reconstructs corrupted tokens from context.
- node1 (next-token AR head): 35.2% at byte level — reasonable for a 1M-param probe on 11MB.
- Calibration was the biggest risk and came through cleanly: BCE on the token-identity event trains honest confidence heads under 15% input corruption.
- Ensemble barely beats its best member because voters share one trunk (correlated) — as predicted in design §5; real gains need decorrelated voters (different transports/depths), which the C-runs vary.
- **Gate (d) failure is structural, not a bug** (review finding 14, now the report's claim too): with transport=direct and the same 2-layer stack re-run on a detached latent, round 2 is a deterministic function of round 1 — no new information enters (no re-discretization, no fresh corruption of context), and the fixed point is reached in pass 1. Repair value must come from the soft/decode transports (repair-in-context) and the token retry — exactly what C3 and R1 test.
- The training-log loss alternation (two levels ≈ 2×) is the p_retry coin: batches with a retry round carry ~2× the loss terms.

## Consequences for the main program

1. Probability machinery validated for M1 — proceed with base + I1 (base config now includes the self node and positional embeddings).
2. C-runs unchanged, but expectation sharpened: transport choice, not raw extra passes, is where retry value lives.
3. Report "ensemble vs best single" everywhere; treat large ensemble gains as absent until voters are decorrelated.
