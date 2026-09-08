# Workspace

## Status (2026-09-07)

- **Design:** frozen in `docs/design.md` (stage ring 0–4, cycles, aggregation principle, training recipe). One post-freeze amendment: shuffle-locality formula in §L (v1 was mathematically a no-op).
- **M0 probe: DONE, merged.** Micro-model (d=64, 2L, ~1M params, enwik8-12MB). Verdicts: (a) confidence calibration PASS (ECE 0.004–0.011); (b) ensemble ≥ best single PASS (+0.0008…+0.0015, 3 runs/2 seeds); (c) consistency loss tightens estimates PASS (cosine 0.9525→0.9882); (d) direct-transport retry FAIL (structural no-op). Evidence + report: `.scratch/02-engine-m0/evidence/REPORT.md`.
- **Next: M1** — tickets `.scratch/03-engine-m1/issues/01-engine-m1-extensions.md` (engine: transports linear/soft, dense early-exit supervision, eviction+anchor slots, eval battery, configs-as-files) then `.scratch/04-m1-runs/issues/01-m1-runs.md` (7 runs: B0, I1, C1, C3, R1, L2, X2). Both ready-for-agent, not claimed.

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
