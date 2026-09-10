# Path LM — experiment program

Decoupled ablation program for `docs/design.md`. Principles: one element per run (everything else = base config); gates defined before running; token-matched and FLOPs-matched both reported; 2 seeds wherever deltas <5% are expected; no all-on run until the integration run.

## 0. Lessons from M0 (absorbed 2026-09-07)

- **Retry is information-free unless the return channel transforms the state.** Direct re-entry re-derives the same fixed point (gate d, measured at 2 layers). Retry experiments are transport experiments: repair value is expected from soft (repair-in-context) and discrete (token retry) channels, which re-embed a corrected or expected token. No new information enters a loop whose channel is the identity.
- **Corruption + direct retry does not repair** (measured) — the transport, not the corruption, is the active ingredient.
- **Ensemble voters sharing one trunk are too correlated for meaningful gains** — gains must come from decorrelated estimates (different transports/depths).
- **Hard discrete re-entry (token retry) is only safe gated by prob₀.** Ungated, it rewrites every position with the model's argmax: correct predictions are ~no-ops, but the ~49% wrong commits re-embed as tokens that look exactly like ground truth, erasing the uncertainty signal downstream attention could have discounted — and each wrong commit contaminates the context for every position that attends to it (measured: repair 51.3% → 46.7%, ECE 0.001 → 0.015, M1/R1). Direct/soft re-entry keep the uncertainty in the latent; quantization destroys it. Fix: fire per-position only where prob₀ is low (selective re-embed).
- Consequence: C3 (soft) and R1 (discrete) are promoted into M1; C1 (direct) is retained as the fixed-point question at 12 layers (does a deeper stack converge in one pass?); C2 (linear) is deferred to M2.

## 1. Base config (identity element)

Standard AR decoder: clean tokens, stage-1 identity, no shuffle/skip/redo/exit, transport unused, no retry, MTP outputs for [i, i+1] only, tied E=U. The MTP block and probability heads are always-on infrastructure (~free, and they keep every measured delta pure — readout machinery constant across runs).

## 2. Main-effect runs

Type I — input corruption (repair supervision active wherever corruption exists):

| Run | Element | Question | Gate |
|---|---|---|---|
| I1 | wrong-token replacement | capacity tax of token corruption | PPL delta vs base |
| I2 | `[mask]` replacement | mask vs wrong-token | PPL delta |
| I3 | +Gaussian noise on embedding | continuous corruption tax | PPL delta |
| I4 | pure-noise latent (bypass embedding) | full writable-input claim | PPL delta |

Type C — transport + latent retry (inherently coupled: a transport without a retry does nothing, so they are measured as one unit):

| Run | Element | Question | Gate |
|---|---|---|---|
| C1 | direct + latent retry | does the latent loop actually refine? | quality-vs-rounds monotonic |
| C2 | linear transport | vocab-subspace return | PPL delta; rounds curve |
| C3 | soft transport (+repair-in-context) | does repair-in-context pay? | PPL delta + repair accuracy |
| C4 | soft transport + **mixture re-entry** (design §3 amendment) | does accumulated mixture beat overwrite at equal rounds? | repair after retry ≥ base; retry-curve vs C3 |
| C5 | token retry + **mixture re-entry** (the R1 lesson fix: discrete re-entry on the accumulated distribution) | does mixture fix R1's error-commitment? | repair after token round ≥ base pass |

Type L — depth elasticity:

| Run | Element | Question | Gate |
|---|---|---|---|
| L1 | p_skip | skip tax | PPL delta; quality/speed curve |
| L2 | p_exit (dense supervision) | early-exit payoff | speedup at iso-PPL |
| L3 | shuffle-locality | order tolerance | PPL delta across locality levels |
| L4 | p_redo | per-layer repeat tax | PPL delta; rounds curve |

Type R — token retry:

| Run | Element | Question | Gate |
|---|---|---|---|
| R1 | stage-4 discrete retry | self-correction at token level | repair accuracy; ECE of prob_0 |

Type X — locality/eviction:

| Run | Element | Question | Gate |
|---|---|---|---|
| X1 | attention distance penalty (low weight, correctness dominates) | locality prior tax | PPL delta; per-head distance telemetry (some heads must stay long-range) |
| X2 | eviction training + anchor slots | ring-buffer deployment without re-prefill | needle-in-haystack must not collapse |
| X3 | X1 + X2 | is the penalty redundant under eviction training? | comparison vs X1, X2 |

Total: base + 15 main-effect runs.

## 3. Interaction runs

| Run | Combo | Question | Gate |
|---|---|---|---|
| IX1 | best-I × C3 | repair-in-context payoff over corruption alone | repair accuracy jump |
| IX2 | L1+L2+L3+L4 all on | do the depth knobs compose? | PPL delta vs sum of main effects |
| IX3 | L3 × C1 | order-free layers + latent loop | rounds curve monotonic |
| IX4 | C1 with consistency loss off | does the consistency loss carry chaining? | chain-depth curve |
| INT | all elements on | the actual Path LM | full metrics battery |

## 4. Milestones

- M1 (7 runs): B0, I1, C1, C3, R1, L2, X2. B0 doubles as the external sanity anchor (published small-model enwik8 range ~1.4–1.6 bpc for this size class). C3 is promoted per the M0 lesson — retry × soft transport is the first configuration where retry can add information. Any gate failure learned at ~30% of program cost.
- M2: remaining main effects (I2–I4, C2, L1, L3, L4).
- M3: interaction runs + integration run.

Engine prerequisites per run: B0/I1 need the eval battery (bpc, repair, speed); C1/C3/R1 need the transport implementations; L2 needs dense early-exit supervision; X2 needs eviction + anchor slots.

## 5. Scale and defaults (decided)

Dataset: **enwik8** (char-level) — chosen for comparability: it is the standard char-level benchmark with many published small-model baselines (bits/char), giving the base config an external sanity anchor; if convergence is too slow, TinyStories char-level is the fallback (less comparability, faster learning).

Scale (approved): char-level ASCII vocab (~100 printable), d=256, 8 layers, ~10–15M params, seq 512. Sanity anchor: a vanilla transformer of this size should land in the published ~1.4–1.6 bits/char range on enwik8; landing far above that means our always-on machinery has a real cost. Probe-scale (M0) as above.
Deviation (2026-09-07): 8 layers × 12d² cannot reach 10–15M params — the param count is the comparability-relevant number, so base scale is d=256, **12 layers, mlp_mult 6, ~13.1M params**.
Needle construction (amended 2026-09-10 to v2 — v1's random-filler single-needle form starved the convention of signal and measured 0.0 everywhere; see ticket 05): row = real-text window with K=8 needle pairs (x_i, y_i) carved in at random positions p_i (anchor_frac of them inside the first `anchors` positions), query tail [mask, x_1, y_1, …, mask, x_K, y_K]; (x_i, y_i) pairs are chosen absent from the row's text so the association is unambiguous. Node-1 accuracy at the x_i query rows, bucketed by regime: in-window (reachable), anchor (survives eviction via the anchor channel), beyond (evicted — ALLOWED to fail; the information is gone, that is the design, not a bug).

Defaults: λ_total=0.15 split among active I-elements; p_skip=p_redo=p_exit=0.1; shuffle-locality=0.5 in the L3 run; N=1 in the base config (k=0…1 — self + next-token; deeper MTP N>1 is an M2+ sweep, see §1); retry counts sampled geometrically during training (coverage), prob_0-threshold gated at inference; norm cap c = typical embedding norm. Corruption is part of the C1/C3/R1 units (a retry needs something to repair) — read their deltas against I1, not B0, to isolate the retry/transport element. Engine: PyTorch custom training loop (config-driven path sampling) — decided and implemented.
