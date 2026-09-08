# 02 — Engine skeleton + M0 probability-machinery probe

- **Status:** done
- **Type:** implementation
- **Blocked by:** 01-design/issues/01-design-spec
- **need-review:** true
- **need-test-cases:** true

## Issue

Implement the config-driven PathLM engine skeleton (PyTorch custom loop) and the M0 micro-model, then run the M0 probe per docs/experiments.md §4:

- micro-model: d=64, 2 layers, seq 128, N=2 (k=0…2), ~1M params, first ~10MB of enwik8
- (a) prob heads calibrate under corruption: ECE < 0.1
- (b) weighted ensemble ≥ best single path (never hurts)
- (c) consistency loss reduces composition disagreement
- (d) one retry round improves repair accuracy

Scope at M0: corruption (wrong token), MTP heads + per-component CE + confidence heads, consistency loss, direct latent retry. Transports (linear/soft/decode), corruption elements beyond wrong-token, and span mode are deferred to the main-effect runs; config raises NotImplementedError if set.

## Acceptance criteria

- [x] Config-driven path sampling (knobs plumbed, defaults = base config)
- [x] M0 training runs complete with logged metrics (2 configs × 2 seeds)
- [x] Probe report answers (a)–(d) with evidence under `.scratch/02-engine-m0/evidence/`
- [x] Fresh-context review passed (round 1: REJECT → all findings fixed; round 2: OK with notes → notes applied)

## Comments

- 2026-09-07 — agent (pi, volc2/glm-5.3-flash) — claimed; branch ticket-02-m0-probe.
- 2026-09-07 — agent (pi, volc2/glm-5.3-flash) — v1 probe done; review REJECT (critical: node targets off-by-one, self node missing; no positional encoding; 11 more findings). All fixed, probe re-run (v2).
- 2026-09-07 — agent (pi, volc2/glm-5.3-flash) — review round 2: OK with notes (same-pass disagreement metric, shuffle_locality no-op formula — fixed in config + design.md §L, gate (b) second seed, vocab_size logged, data cache untracked). Notes applied, merged to main. Gates: (a) PASS ECE 0.004–0.011, (b) PASS +0.0008..+0.0015 over 3 runs/2 seeds, (c) PASS cosine 0.9525→0.9882, (d) FAIL structural (direct-transport retry is a no-op — expected to be resolved by C3/R1 transports).
