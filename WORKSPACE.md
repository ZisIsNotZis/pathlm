# Workspace

## Status (2026-09-07)

- **Design:** frozen in `docs/design.md` (stage ring 0–4, cycles, aggregation principle, training recipe). One post-freeze amendment: shuffle-locality formula in §L (v1 was mathematically a no-op).
- **M0 probe: DONE, merged.** Micro-model (d=64, 2L, ~1M params, enwik8-12MB). Verdicts: (a) confidence calibration PASS (ECE 0.004–0.011); (b) ensemble ≥ best single PASS (+0.0008…+0.0015, 3 runs/2 seeds); (c) consistency loss tightens estimates PASS (cosine 0.9525→0.9882); (d) direct-transport retry FAIL (structural no-op). Evidence + report: `.scratch/02-engine-m0/evidence/REPORT.md`.
- **M1 engine extensions: DONE, merged** (ticket 03, review APPROVE round 2 after fixing a P0 decode-retry position bug + 4 P1s). 27 tests green. Transports linear/soft, dense early-exit supervision, eviction+anchors, stage-4 token retry, KV-cache decode (per-layer caches, exit threshold, prob0 retry), full eval battery, configs/*.json. Base scale: d=256, 12 layers, mlp_mult 6 = 13.08M params (deviation recorded in experiments.md).
- **M1 runs: DONE** (ticket 04) — B0 1.507 bpc (published anchor holds), I1 repair 52.8%, C1 retry +2.1pp, C3 ≈ C1, R1 token retry FAIL (ungated), L2 plateau at depth 6, X2 needle FAIL (signal starvation). Summary: `.scratch/04-m1-runs/evidence/SUMMARY.md`.
- **M1 follow-ups: DONE** (ticket 05) — mixture re-entry implemented (design §3 amendment); C4 PASS-equal (mixture = robustness at depth, monotone to 4 rounds), C5 PASS-with-caveat (token-round catastrophe removed, −1.3pp remains; profit needs prob0-gated per-position re-embed — recorded, unscheduled), X2v2 PASS (in-window 84.9% / anchor 76.8% / beyond 0.4%), retry curves saturate ~round 3. Evidence: `.scratch/05-m1-followup/evidence/`. 33 tests green.
- **M2 sweep: DONE** (ticket 06) — element-tax law: identity/order-destroying elements expensive (shuffle +0.545, skip +0.175, wrong-token +0.127), identity-preserving ones nearly free (noise ±0, redo +0.044, mask +0.059). L3's locality boundary: trained 0.5 collapses at eval 1.0 (3.84).
- **X1/X3: DONE** (ticket 07) — distance penalty FREE (−0.007, mild regularizer), composes with eviction (+0.004), anchor channel improves 76.8→93.3%.
- **TTS probes: DONE** (ticket 08) — composition gain ∝ path diversity (L1 −0.047); deterministic-path checkpoints gain 0; depth/round ensembles degrade (auxiliaries are fallbacks, not voters); nothing beats B0 — taxes dominate.
- **M3 + diversity: DONE** (ticket 09) — IX1 sub-additive win (+0.087 < 0.277 sum); composition law: shuffle is the poison (every shuffle combo lands 2.2+; order-freeness antagonizes repair); INT full grammar does not compose at 13M. DIVL1: diversity pressure (capped −JS, float32 + grad-norm guard) improves both baseline AND ensemble slope (−0.062 > −0.047, ceiling +0.023) — mechanism proven, tax still dominates at this scale.
- **Next:** program complete through M3. Open directions (user's call): scale-up (taxes vs diversity at 2–4× size), prob0-gated re-embed, diversity weight sweep / better reward shaping.

## Key lessons (do not re-derive)

- **Retry is information-free unless the return channel transforms the state** — direct re-entry re-derives the same fixed point (measured). Retry experiments ARE transport experiments; repair value lives in soft (repair-in-context) and discrete (token retry) channels.
- **Node semantics:** node k at row i predicts clean token t_{i+k}; node 0 = self/repair of the CURRENT token (source of prob₀, the universal control signal). Targets = the clean token stream itself — never a pre-shifted y (this exact off-by-one was a critical review find in M0).
- Ensemble voters sharing one trunk are too correlated for real gains — decorrelation must come from transports/depths.
- Forward signature: `model(tokens, paths, targets)` — `paths` is one sampled layer path PER round (fresh shuffle/skips each pass, via `sample_rounds`); corruption happens inside the model; eval corruption is seeded via `torch.manual_seed(1234)`.

## Conventions

- Char-level = byte-level enwik8 (205-ish vocab incl. [mask] slot; vocab logged in results.json).
- Tickets: Matt Pocock layout in `.scratch/NN-slug/`; evidence under `.scratch/NN-slug/evidence/`; data caches under `data/` (gitignored), raw enwik8 in `.tmp/enwik8`.
- Tests: `python3 -m pytest tests/ -q` (10 tests, all fail-capable). Review gate: fresh-context subagent reads diff + spec only; two rounds were needed for M0.
- Training runs go to background nohup with log polling; ~2.5 min per M0-size run on the RTX 4090, M1 runs est. 30–60 min.

## Open decisions

- None blocking M1. (Dataset/scale/engine decided: enwik8, d=256/8L/seq512, PyTorch custom loop.)
