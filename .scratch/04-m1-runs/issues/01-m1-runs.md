# 01 — M1 main-effect runs

- **Status:** ready-for-agent
- **Type:** experiment
- **Blocked by:** 03-engine-m1/issues/01-engine-m1-extensions
- **need-review:** false (results review via report)
- **need-test-cases:** false

## Issue

Run the 7 M1 runs at base scale (d=256, 8 layers, seq 512, full enwik8, ~1–2 epochs), one config each, report the fixed metrics battery:

| Run | Config | Gate |
|---|---|---|
| B0 | base (clean, MTP[0..1], no paths) | bpc in published small-model range (~1.4–1.6); MTP always-on cost visible here |
| I1 | + 15% wrong-token corruption, self-repair on | PPL delta vs B0; repair accuracy |
| C1 | + direct transport + latent retry (p_retry sampled) | quality-vs-rounds curve at 8 layers (fixed-point question) |
| C3 | + soft transport + latent retry (corruption on) | repair accuracy jump vs I1; rounds curve |
| R1 | + stage-4 discrete token retry | repair accuracy; prob₀ ECE |
| L2 | + early exit, dense supervision | speedup at iso-PPL; quality-vs-depth curve |
| X2 | + eviction training + anchor slots | needle-in-haystack survival; no-re-prefill decode |

Report into `.scratch/04-m1-runs/evidence/`, one results.json + log per run, summary table at the end. 2 seeds for any run whose gate decision is within noise.

## Acceptance criteria

- [ ] 7 runs complete with committed configs and results
- [ ] Summary table: every gate pass/fail with numbers
- [ ] Token-matched and FLOPs-matched training cost reported per run

## Comments

- 2026-09-07 — agent (pi, volc2/glm-5.3-flash) — created; C3 promoted into M1 and C2 deferred to M2 per M0 lesson (retry is a transport experiment).
