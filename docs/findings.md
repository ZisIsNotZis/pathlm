# Findings — what the program measured about each mechanism

One section per elastic-grammar mechanism, ordered by the design's stage map.
Every number is a measured main effect on enwik8 (13M params, 12 layers, base
B0 = 1.5074 bpc; Δ = clean-eval bpc tax vs B0). Evidence: `.scratch/*/evidence/`
(tickets 04–09). Read with docs/experiments.md (the run table) — this file is
the conclusions layer.

## Redo (single-layer repeat, p_redo)

Δ +0.044 — nearly free. Re-running a layer 10% of the time costs ~3% relative
PPL. Cheapest depth-side element. No interference observed in combination
(IX2's failure is attributable to shuffle, not redo). Use freely where extra
per-layer compute is acceptable; it is the budget way to add depth adaptivity.

## Skip (p_skip)

Δ +0.175 alone — the most expensive depth element: skipping 10% of layer
executions costs more than any corruption type. BUT skip is the best
test-time-scaling substrate: sampled paths give real ensemble diversity
(bpc 1.6926 → 1.6454 at K=8, −0.047), and with diversity pressure training
(DIVL1) both the baseline (1.6842) and the gain (−0.062) improve, breaking the
plain-skip K=8 ceiling by 0.023. Reading: skip is expensive as insurance but
converts to TTA capability better than anything else measured.

## Early exit (dense per-depth supervision, w_dense_exit)

Δ +0.067; exit at confidence threshold = 1.4× decode speed at ~iso-PPL. Depth
curve plateaus around depth 6/12 — half the stack adds ~0.003 bpc. Depth
ENSEMBLING (averaging per-depth predictions) FAILS: early depths are strictly
worse estimators (depth CE 1.06 → 0.74 is a 15× perplexity cliff), and the
confidence heads rank depths, not per-token reliability — wrong signal for
mixture weights. Reading: dense supervision buys calibrated early exits
(deployment value), not ensemble voters.

## Shuffle (order-free layers, shuffle_locality)

Δ +0.545 alone — in a class of its own; layer ORDER carries ~0.5 bpc. Hard
boundary: trained at locality 0.5, it tolerates ≤0.5 at eval but collapses at
full random order (3.84 bpc). Composition poison: every interaction run
containing shuffle lands at 2.2+ bpc (IX2, IX3, INT), and it is specifically
antagonistic with repair — the retry loop cannot refine what shuffle scrambled
(repair drops 51% → 38%). Probable dead end for prediction quality at this
scale, exactly as suspected; its only measured value is the (large) robustness
it buys against layer dropout/failure scenarios, which nothing has needed yet.

## Future-token heads (MTP block, node k predicts t_{i+k})

Infrastructure cost ~0 (always on; keeps every measured delta pure). The
node-1 head IS the standard AR head. Extra heads (node-2, chain estimates)
produced no clean-stream ensemble value in the probes run (n_mtp=1 everywhere;
IX4's consistency loss on chained latents was neutral: 1.6589 ≈ C1's 1.6549).
The heads' real payoff measured so far is indirect: they carry the confidence
signals that gate exit/retry and the dense-exit supervision. Deeper MTP stacks
(n_mtp ≥ 2 with spec-style self-speculative decoding) remain untested.

## Latent retry (loop back through a transport)

Δ ~+0.15 as a unit (retry requires corruption to act on; C-runs carry
corrupt_wrong 0.15). Repair accuracy +2.0–2.5pp over the base pass at depth;
gains saturate by round 3 (C1 dips at 4). Transport flavor does not matter:
direct ≈ linear ≈ soft at this scale (C1 1.6549, C2 1.6621, C3 1.6578) — the
value is the extra pass, not the return channel's geometry. Retry is a REPAIR
mechanism, not an ensemble: averaging rounds on clean streams degrades (later
rounds re-enter through transformed, distribution-shifted states).

## Token retry (discrete re-entry) and the mixture fix

Ungated argmax re-embed (R1) is catastrophic: −4.6pp repair, ECE 0.001 → 0.015
— wrong commits re-embed as ground-truth-looking tokens. Confidence-weighted
mixture re-entry (design §3 amendment) removes the catastrophe (C5 v2: −1.3pp,
ECE 0.007) and gives monotone robustness through 4 rounds (C4: 50.2→53.1%),
but the token round still lands slightly below base on clean data. The
remaining lever, recorded but unscheduled: prob0-gated per-position re-embed
(rewrite only where the model itself flags doubt).

## Input corruption (the I family) — flagged beats silent

Mask replacement: Δ +0.059, repair 59.4%. Wrong-token: Δ +0.127, repair 52.8%.
Corruption with an explicit "I don't know" flag is BOTH cheaper and more
repairable than silent adversarial corruption — a design input for how any
writable-input interface should signal corruption. Embedding noise is FREE
(±0.000); pure-noise latents (full writable-input claim) cost only +0.063.
Retry × corruption composes sub-additively (IX1: +0.087 < 0.277 sum) — the one
demonstrated composition win, because the mechanisms share repair machinery.

## Attention distance penalty (X1) — a free regularizer

Δ −0.007 (within noise but positive-signed): a low-weight locality prior costs
nothing and slightly helps on local-structure-dominant text. Composes with
eviction training at +0.004 total and IMPROVES the anchor channel
(needle 76.8% → 93.3%). Per-head telemetry: mean attended distance spreads
26–43, no head collapses to local-only. Beyond-window needle "hits" under the
penalty are local-LM strength, NOT eviction-proof recall — do not read them as
retention.

## Eviction + anchors (X2) — the ring-buffer deployment story

Windowed attention with anchor slots trains stably without re-prefill; needle
recall through the anchor channel reaches 76.8% (93.3% with the penalty).
Decode with per-layer caches, early exit, and prob0-gated retry works at 1.4×
speed. KV-cache positions stay strictly increasing across retry rounds (the
P0-class bug class to keep testing).

## Cross-cutting laws

1. **Element-tax split**: mechanisms that destroy identity/order structure
   (wrong-token, skip, shuffle) are expensive (Δ 0.13–0.55); mechanisms that
   preserve or flag identity (mask, noise, redo, penalty) are nearly free
   (Δ ≤ 0.07).
2. **TTA gain ∝ trained path diversity**: deterministic-path checkpoints gain
   exactly 0 from K-path ensembling; diversity must be trained in, and the
   diversity-pressure knob (capped −JS between parallel paths, float32 +
   grad-norm guard under bf16) measurably improves both baseline and slope.
3. **Composition is selective**: sub-additive when mechanisms share machinery
   (corruption × repair); antagonistic when one destroys the structure the
   other needs (shuffle × everything).
4. **Auxiliaries are fallbacks, not voters**: depth prefixes and retry rounds
   are trained as repairs/exits; mixing them into predictions degrades. To get
   ensemble value from auxiliary estimators, supervise them as co-equal
   predictors (untested) rather than fallbacks.
5. **Stability engineering for reward terms**: any negative/reward loss term
   under bf16 autocast needs float32 computation and a pre-step gradient-norm
   gate (two diverged runs taught this).
6. **Scale is the untested variable**: every "tax dominates" verdict above is
   at 13M params; whether tax floors and ensemble gains scale favorably is the
   main open question.
