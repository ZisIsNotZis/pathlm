# 01 — Test-time scaling probes: can composition beat B0?

- **Status:** done
- **Type:** experiment (eval-only)
- **Blocked by:** 07-x1-x3/issues/01-x1-x3-distance-penalty
- **need-review:** true
- **need-test-cases:** false

## Issue

The user's question: a regular transformer cannot do TTA — one weights, one
function, one shot. Ours is approximately path-marginal (many compute paths
estimate the same distribution). Can composition of probability at test time
beat B0's 1.5074, and at what compute multiple?

Probes (probe_tts.py, eval-only over existing checkpoints):

- **T1 path-ens**: average node-1 softmax over K sampled paths, K ∈ {1,2,4,8},
  on L1/L3/L4/X1/X3/I3 (the path-diverse checkpoints).
- **T2 depth-ens**: average per-depth node-1 predictions on L2
  (uniform + confidence-weighted). FREE — one forward emits all depths.
- **T3 round-ens**: average node-1 across retry rounds on the CLEAN stream
  (C1/C3/C4, K ∈ {2,4}) — mixture never tried on clean bpc.

Deliverables per the brainstorm: bpc(K) curves → K_match (test mult to match
B0), K_sat (saturation), score_sat; train_mult probes deferred until T-results
are in. Fitness of T3 caveat: C1/C3/C4 with p_retry but deterministic eval
paths may make rounds identical on clean data — if so, that IS the finding
(paths collapsed onto one function for these checkpoints).

## Acceptance criteria

- [x] probe runs, tts_results.json written
- [ ] K_match/K_sat/score_sat table vs B0 anchor
- [ ] verdict: which mechanism (if any) beats 1.5074, at what multiplier

## Comments

- 2026-09-10 — agent (pi) — created; claimed. Engine prep committed: config-gated depth-logit capture (T2, eval-only), needle-batch silent-ambiguity fix found on the way (exhausted pair-retry must skip, not place a colliding pair). probe_tts.py smoke-tested (L1 K=1 plumbing; L2 depth-mix works — uniform 1.604, conf 1.6009 on a 2-batch smoke slice). Full battery launched.
- 2026-09-10 — agent — **battery complete** (after 3 self-caught bugs: batch-summed CE, CPU tensors, train-tail eval leak — all fixed & committed). Verdict: composition gain tracks path diversity exactly (L1 −0.047 monotone; deterministic-path X1/X3 exactly flat); depth-ens free but degrades (1.602 vs own 1.574 — depths are fallbacks, not voters); round-ens degrades on clean streams (retry is a repair mechanism, not an ensemble). **Nothing beats B0.** K_match: none exists; K_sat ≈ 8; score_sat = 1.6454 (L1@K8). Why: architectural math — taxes > ensemble returns + shared-trunk correlation (M0 lesson held) + fallback-trained auxiliaries. Fixes are design-level (diversity-preserving training / equal-footing aux supervision / scale). SUMMARY.md + tts_results.json committed.
- **Status:** done
