# Findings — what the program measured about each mechanism

One section per elastic-grammar mechanism, ordered by the design's stage map.
Every number is a measured main effect on enwik8 (13M params, 12 layers, base
B0 = 1.5074 bpc; Δ = clean-eval bpc tax vs B0; gain = what the mechanism
buys, in its own currency). Evidence: `.scratch/*/evidence/` (tickets 04–09;
ticket 11 corrects the retry matrix — read that section before quoting any
retry number). Read with docs/experiments.md (the run table) — this file is
the conclusions layer. Measured-gaps are stated explicitly per mechanism: a
blank gain cell means NOT MEASURED, not zero.

**Caveat on every bpc number in this file (ticket 11):** all runs are single
or double seed, and the two variance floors are run-to-run nondeterminism
0.0019 bpc and per-cell seed variance 0.0001–0.0138 bpc. Deltas below ~0.014
bpc — which includes most of the retry matrix and several main effects — are
not resolvable at this replication. All runs are also ~1.1 epochs with a
cosine LR still mid-anneal, so every "tax" may be a convergence-rate
difference rather than an asymptote difference; that is being measured
(ticket 11, D5).

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
**MEASURED (ticket 10, B2 = n_mtp=2 at M1 scale, bpc 1.5481 / Δ+0.041):**
- node-2 drafts t_{i+2} at **55.97% accuracy** (vs node-1's 68.8% on t+1;
  chance 0.5%) — much stronger than M0's 0.26; a real draft head.
- **Deployable accept signal measured**: node-2's draft agrees with node-1's
  verification 65.8% overall; **conditioned on node-1 being right, the draft
  is accepted 77.3%** — a usable spec-decode accept rate.
- **Naive probability composition FAILS**: folding P1(t_{i+1}) with
  P2chain(t_{i+2}) via product-marginal gives CE 3.96 vs node-2 alone 1.58.
  The product treats both distributions as marginals over the same variable —
  the correct chain needs node-2 CONDITIONED on node-1's sampled token
  (re-embed + re-predict), i.e. chain2-through-embedding, not through latents.
  Recorded as the designed-but-unbuilt conditional-chain head.
- **CONDITIONAL CHAIN BUILT AND MEASURED (ticket 12 P1, 6000 steps x 2 seeds):**
  DeepSeek-style concat+proj (`Linear(2d,d)` on [context latent h; embed(t_{i+1})])
  then T1. **Oracle (true t_{i+1}) t+2 acc 0.682 ≈ node-1's own t+1 acc 0.693** —
  the information is there and the conditioning uses it. **Deploy (node-1 argmax
  condition) 0.547 < direct 0.560**: ~31% draft error injects more noise than the
  conditioning recovers; accept rate 0.75 ≈ direct 0.77. Verdict: mechanism sound,
  no deployable gain at this scale — bottleneck is draft quality; re-test gated on
  a better t+1 draft (prob0-gated retry is the candidate). k=1 draft also gives NO
  throughput gain in greedy decode (the verify round is the next generation round);
  throughput needs chain expansion to node-3 (unbuilt).

## Latent retry (loop back through a transport) — the 2×(transport × gating) matrix

The C/R runs form a matrix: 4 transports × {ungated overwrite, mixture vote}.
All C-runs carry corrupt_wrong 0.15 (retry is measured AS a repair unit — a
transport without corruption has nothing to repair).

| Run | transport | re-entry | bpc | seeds | repair base→last | forced 4-round curve |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | direct | overwrite | 1.6549 | 1 | 0.507→0.528 | — |
| C1M_n1 | direct | **mixture** | 1.6645 / 1.6595 | 2 | 0.493→0.518 | 0.493/0.518/0.521/0.522 |
| C2 | linear | overwrite | 1.6621 | 1 | 0.499→0.519 | 0.499/0.519/0.523/0.518 |
| C2M_n1 | linear | **mixture** | 1.6548 / 1.6686 | 2 | 0.503→0.522 | 0.503/0.522/0.524/0.524 |
| C3 | soft | overwrite | 1.6584 | 4 | 0.496→0.521 | — |
| C4 | soft | **mixture** | 1.6535 | 1 | 0.501→0.526 | 0.501/0.526/0.529/**0.530** |
| R1 | token | overwrite | 1.6775 | 1 | 0.513→0.467 | — |
| C5 | token | **mixture** | 1.6856 | 1 | 0.500→0.431 | 0.500/0.431/0.361/0.303/**0.427** |

**CORRECTED (ticket 11, 2026-09-11). The cell ORDERING this section used to
state is WITHDRAWN — read this before quoting any number in it.**

Four defects invalidated the comparison (full detail + evidence:
`.scratch/11-retry-matrix-validity/`):

- **D2 — config confound.** `C1M`/`C2M` omitted `n_mtp`, defaulting to 2 while
  every cell they were tabulated against pins 1. The knob alone costs +0.041
  (B2 1.5481 vs B0 1.5074). Published C1M +0.194 / C2M +0.164 were therefore
  inflated; the controls at `n_mtp: 1` are C1M_n1 1.6645/1.6595 and C2M_n1
  1.6548/1.6686.
- **D1 — re-entry gauge.** `_mixture_reentry` applies the stage-E `norm_cap`
  (1.0) to stage-L states whose norm is 29–88, while `_transport(direct)`
  returns `h` uncapped. Measured: direct/linear mixture re-entry norm 1.00 vs
  overwrite 28.7; soft 0.90 (cap is a no-op). Direct/linear mixture cells
  therefore measured a ~30× rescale, not a latent mean, and the reported
  soft > linear > direct ordering is monotone in how much the cap destroys.
- **D3 — untrained accumulator.** `p_retry` is Bernoulli, so training never
  exceeds 2 passes; the accumulator only differs from overwrite at r≥2, i.e.
  only in the inference-time forced `rounds=k` loop. At r=1 soft mixture and
  soft overwrite are bit-identical (`max|d| = 23.3726, cos = 0.92109` both
  arms), so the soft arm could not have measured mixture at training depth.
- **D4 — variance.** Two floors, measured separately: run-to-run
  nondeterminism 0.0019 bpc (three same-seed 6000-step runs), seed variance
  0.0001 (C3) / 0.0050 (C1M_n1) / **0.0138** (C2M_n1). Seed variance is
  cell-dependent and for C2M_n1 alone exceeds the entire six-cell spread.

Values are in the table above. All cells `n_mtp: 1`; B0 = 1.5074 is itself
single-seed, so cell-vs-cell comparisons are B0-free but absolute Δ's carry
B0's unknown seed variance.

Revised readings (folding in what survives of the original four):

1. **No ordering among the latent cells is resolvable.** They span 0.011 bpc;
   C2M_n1's seed spread alone is 0.014, and its mixture-vs-overwrite delta
   *reverses sign between seed 0 and seed 1* (−0.0073 → +0.0065). Withdrawn:
   "soft×mixture is best, direct×mixture is worst" (compared confounded cells)
   and "re-enter through vocab space, not the latent mean" (single-seed winner).
2. **Among overwrite cells the transport flavor is unresolvable** (spread
   0.007 bpc). That describes the data; it is not a demonstrated equality.
   What this section supports is the value of the extra pass, not the
   return-channel geometry.
3. **The retry tax itself is solid**: every C cell lands 1.6535–1.6856 vs B0's
   1.5074, i.e. +0.146…+0.178, 10–100× any measured floor. Two passes through a
   12-layer stack with 15% corruption costs ~0.15 bpc. Quote this one.
4. **Gating's demonstrated value is on the repair side**, not clean bpc: R1's
   −4.6pp → C5's −1.3pp (~20× the 0.16pp repair sampling error), and the
   forced-round curves stop dipping (C2's r4 dip 0.523→0.518 removed in
   C2M_n1's 0.524→0.524).
5. **What direct/linear mixture measured is a rescale**, so those cells are not
   evidence against latent-mean mixing in principle — only against this gauge.
6. **Retry is a repair mechanism, not an ensemble**: averaging rounds on clean
   streams degrades; later rounds re-enter through transformed, distribution-
   shifted states.
7. **L-loop × depth synergy**: retry helps more at depth (C1's +2.1pp at 12L vs
   ~0 at shallow stacks in M0) — deep stacks do not reach a fixed point in one
   pass.

Still open: prob0-gated per-position re-embed (per-position accept decisions
instead of the batch-level coin). **Reading 3 measured (ticket 11 D5, 24000
steps ≈ 4.4 epochs):** the retry tax falls +0.364 (step 1000) → +0.122 (step
24000), |Δlast|/|Δfirst| = 0.33, but it *plateaus* around +0.12–0.13 from step
~3000 rather than decaying to zero — a real tax, ~20% smaller than the
published 1.1-epoch +0.147. Consequence for this whole file: every published
number is a 1.1-epoch snapshot (B0 itself: 1.5074 at 1.1 epochs vs 1.3459 at
4.4) and is an upper bound on the asymptote. Measured on the pre-D1/D3 engine
(Bernoulli retry, capped direct mixture) — i.e. the configuration the published
table describes; the fixed engine changes retry depth and must be re-measured.

**FIXED-ENGINE RETRY (ticket 12 P2, 12000 steps x 2 seeds) — D1 causally
confirmed:** on the repaired gauge the mixture run's round-2 corrupted-input
bpc improves monotonically — 2.71 → 2.33 → 2.19 → 2.09 → 2.03 → 1.98 → 1.95 →
**1.936** (2 seeds 1.943/1.932) — where the pre-fix engine diverged
(3.7 → 8.5 → 32.9 → 95.2 by step 21000). Round order is now the designed one:
r1 1.993 → r2 1.936 → r3 1.934 saturating — the retry round genuinely refines.
The pre-fix explosion was the gauge, not the mechanism. (probe_e2e's metric is
node-1 next-token accuracy on corrupted positions, NOT eval.repair's node-0
self-repair; not comparable.)

## Token retry (discrete re-entry) — see matrix rows C5/R1

Ungated argmax re-embed quantizes away uncertainty: wrong commits re-embed as
ground-truth-looking tokens (R1: −4.6pp, ECE 0.001 → 0.015). Mixture vote
(argmax of the accumulated distribution, anchor w=1) removes the catastrophe
(C5: −1.3pp, ECE 0.007). These repair effects are ~3–4.5pp against a repair
sampling error of 0.16pp (n≈98k corrupted positions) and survive ticket 11.
The bpc side does not: R1 1.6775 and C5 1.6856 sit only 0.009–0.017 above the
worst observed latent-transport cell (C2M_n1 seed 1, 1.6686), inside that
cell's own 0.0138 seed spread — so "the token round lands below base on clean
data" is suggestive, not established (single seed each). Remaining lever,
recorded but unscheduled: prob0-gated per-position re-embed (rewrite only
where the model itself flags doubt) — needs a per-position accept mechanism,
which is also the missing accept-rate measurement above.

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
