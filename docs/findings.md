# Findings — what the program measured about each mechanism

One section per elastic-grammar mechanism, ordered by the design's stage map.
Every number is a measured main effect on enwik8 (13M params, 12 layers, base
B0 = 1.5074 bpc; Δ = clean-eval bpc tax vs B0; gain = what the mechanism
buys, in its own currency). Evidence: `.scratch/*/evidence/` (tickets 04–09).
Read with docs/experiments.md (the run table) — this file is the conclusions
layer. Measured-gaps are stated explicitly per mechanism: a blank gain cell
means NOT MEASURED, not zero.

## Redo (single-layer repeat, p_redo)

Cost: Δ +0.044 — nearly free (re-running a layer 10% of the time costs ~3%
relative PPL); no interference in the measured combination (IX2's failure is
attributable to shuffle, not redo).
Gain: **negligible measured so far** — but the with/without comparison is
cheap and NOT yet done: (a) eval-time ablation (paths with p_redo forced to 0
vs the trained config) — one battery re-run, no retraining; (b) the deeper
per-layer question — does the second execution refine that layer's own
output? — needs a probe comparing CE-through-frozen-head after 1× vs 2× on
the same input. The theoretical payoffs (per-layer self-correction,
depth-adaptive compute at the LAYER level) are unexploited: redo was never
conditioned on layer identity or on confidence at that depth. Cheapest knob
on the board; the gain column is open and now has two concrete measurement
recipes attached.

## Skip (p_skip)

Cost: Δ +0.175 — the most expensive depth element (skipping 10% of layer
executions costs more than any corruption type).
Gain: **the best test-time-scaling substrate measured.** Sampled paths give
real ensemble diversity (1.6926 → 1.6454 at K=8, −0.047); with diversity-
pressure training (DIVL1) BOTH terms improve — baseline 1.6842, gain −0.062,
breaking the plain-skip K=8 ceiling by +0.023. Unmeasured: skip conditioned at
run-time on per-layer confidence (adaptive skip), and FLOP-matched comparisons
(skip should be credited per saved FLOP, not per run).

## Early exit (dense per-depth supervision, w_dense_exit)

Cost: Δ +0.067.
Gain: **1.4× decode speed at ~iso-PPL** (exit at confidence threshold); depth
curve plateaus at depth ~6/12 (half the stack adds ~0.003 bpc) — the exit
curve is shallow, so most of the speedup is nearly free of quality loss.
Measured NEGATIVE result: depth ENSEMBLING fails (1.602 mixed vs 1.5744
single-pass) — early depths are strictly worse estimators (depth CE 1.06 →
0.74, a 15× perplexity cliff) and the confidence heads rank depths, not
per-token reliability. Unmeasured: exit-threshold sweep (the 1.4× is one
operating point), spec-decoding on top of exits.

## Shuffle (order-free layers, shuffle_locality)

Cost: Δ +0.545 — in a class of its own; layer ORDER carries ~0.5 bpc. Hard
boundary: trained at locality 0.5 it tolerates ≤0.5 at eval but collapses at
full randomness (3.84). Composition poison: every combination containing
shuffle lands at 2.2+ bpc (IX2, IX3, INT) — specifically antagonistic with
repair (retry loop cannot refine scrambled state; repair 51% → 38%).
Gain: **the only measured value is failure-robustness** (layer dropout/
permuted-deployment scenarios), which nothing has needed yet. Probable dead
end for prediction quality at this scale, exactly as suspected. Unmeasured:
partial locality (<0.25) at 2–4× scale; order-canonicalization at eval
(sorted execution as a free "repair to canonical order" mode).

## Future-token heads (MTP block, node k predicts t_{i+k})

Cost: ~0 (always-on infrastructure; keeps every measured delta pure).
Gain: **indirect, measured** — the heads carry the confidence signals that
gate early exit (calibrated by dense supervision), the retry gates, and the
mixture votes; without them none of the elastic machinery has a control
signal. Direct prediction-side gain: **not demonstrated** — node-2/chain
estimates produced no ensemble value in the probes run, and the consistency
loss on chained latents was neutral (IX4: 1.6589 ≈ C1's 1.6549).
Measured gap → now scheduled as the top open item: **MTP accept-rate and
probability composition are untested.** At M0 scale (n_mtp=2) the offline
proxy exists: node2_direct acc 0.2606, node2_chain 0.2581 (vs node-1's 0.3516;
chance ~0.005) — node-2 drafts t_{i+2} at ~26%, a plausible spec-draft
quality. But NO M1-scale checkpoint has n_mtp=2, and the interesting question
is exactly the user's: does CHAINING the MTP heads and COMPOSING their
probabilities (node-2's t_{i+2} estimate folded with node-1's t_{i+1} → t_{i+2}
marginal) beat either head alone — measurable offline once one d=256 n_mtp=2
run exists, plus the deployable accept-rate metric (draft verified by the
next forward) in decode.

## Latent retry (loop back through a transport) — the 2×(transport × gating) matrix

The C/R runs form a matrix: 4 transports × {ungated overwrite, mixture vote}.
All C-runs carry corrupt_wrong 0.15 (retry is measured AS a repair unit — a
transport without corruption has nothing to repair).

| Run | transport | re-entry | Δ cost | repair base→r1 | forced 4-round curve | note |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | direct | overwrite | +0.147 | 0.507→0.528 | — | baseline loop |
| C2 | linear | overwrite | +0.155 | 0.499→0.519 | 0.499/0.519/0.523/0.518 | saturates r3, dips r4 |
| C3 | soft | overwrite | +0.150 | 0.496→0.521 | — | ≈ C1 — flavor irrelevant |
| C4 | soft | **mixture** | +0.146 | 0.501→0.526 | 0.501/0.526/0.529/**0.530** | monotone, never dips |
| C5 | none (token) | **mixture** | +0.178 | 0.500→0.431 | 0.500/0.431/0.361/0.303/**0.427** | token round recovers to −1.3pp vs base |
| R1 | none (token) | overwrite | +0.170 | 0.513→0.467 | — | UNGATED = catastrophic |

Readings: (1) transport flavor is irrelevant — direct ≈ linear ≈ soft within
+0.01; the value is the extra pass, not the return-channel geometry. (2)
Gating matters more than transport: mixture re-entry removes retry's failure
modes (R1's −4.6pp → C5's −1.3pp; C4's curve monotone to r4 where C2 dips).
(3) Retry is a REPAIR mechanism, not an ensemble: averaging rounds on clean
streams DEGRADES (later rounds re-enter through transformed, distribution-
shifted states). (4) L-loop × depth synergy: retry helps MORE at depth (C1's
+2.1pp at 12L vs ~0 at shallow stacks in M0) — deep stacks do not reach a
fixed point in one pass.

**Matrix coverage is NOT complete — the user is right.** Mixture re-entry is
implemented for ALL transports (design §3: soft → expected embedding of S,
linear → project the weighted latent mean, direct → weighted latent mean;
all three pinned by test_mixture_reentry_transport_forms) but only TWO cells
of the 2×4 were measured: C4 = soft×mixture and C5/R1 = token×{mixture,
overwrite}. Direct×mixture and linear×mixture have never run. Given that
gating mattered more than transport in the measured cells, the unmeasured
mixtures are the natural next fills — and the theory sharpens it: C2's
overwrite curve DIPS at r4 while C4's (soft×mixture) is monotone, so
linear×mixture should also fix the dip; direct×mixture tests whether the
weighted latent mean alone (no vocab projection) suffices. Two runs, ~10 min.
Also still open: prob0-gated per-position re-embed (per-position accept
decisions instead of the batch-level coin) — the deployable form of "gating
matters more than transport".

## Token retry (discrete re-entry) — see matrix rows C5/R1

Ungated argmax re-embed quantizes away uncertainty: wrong commits re-embed as
ground-truth-looking tokens (R1: −4.6pp, ECE 0.001 → 0.015). Mixture vote
(argmax of the accumulated distribution, anchor w=1) removes the catastrophe
(C5: −1.3pp, ECE 0.007) but the token round still lands below base on clean
data. Remaining lever, recorded but unscheduled: prob0-gated per-position
re-embed (rewrite only where the model itself flags doubt) — needs a
per-position accept mechanism, which is also the missing accept-rate
measurement above.

## Input corruption (the I family) — flagged beats silent

| Element | Δ cost | gain (repair acc at 15% corruption) |
| --- | --- | --- |
| `[mask]` replacement | +0.059 | 59.4% |
| wrong-token | +0.127 | 52.8% |
| embedding noise | ±0.000 | n/a (no flagged positions) |
| pure-noise latent | +0.063 | n/a |

Corruption with an explicit "I don't know" flag is BOTH cheaper and more
repairable than silent adversarial corruption — a design input for how any
writable-input interface should signal corruption. Noise is free (the norm
cap geometry absorbs it); the full writable-input claim (pure noise) costs
only ~4% relative PPL. Composition win: retry × corruption is SUB-ADDITIVE
(IX1 mask×soft-retry: +0.087 < 0.277 sum) — the mechanisms share repair
machinery, the one demonstrated composition gain.

## Attention distance penalty (X1) — a free regularizer

Cost: Δ −0.007 (within noise but positive-signed) — the only element that
IMPROVED clean bpc.
Gain: composes with eviction at +0.004 total; IMPROVES the anchor channel
(needle 76.8% → 93.3%); per-head telemetry (mean attended distance 26–43,
no head collapses local-only). Caveat: beyond-window needle "hits" under the
penalty are local-LM strength, NOT eviction-proof recall — do not read them
as retention.

## Eviction + anchors (X2) — the ring-buffer deployment story

Cost: Δ +0.121 for X2's original run (v2 needle redesign: +0.058 — the
improvement is measurement, not mechanism).
Gain: ring-buffer decode without re-prefill works; anchor-channel needle
recall 76.8% (93.3% with penalty); per-layer caches with early exit +
prob0-gated retry at 1.4× speed; KV positions strictly increasing across
retry rounds (P0-class bug class to keep testing).

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
