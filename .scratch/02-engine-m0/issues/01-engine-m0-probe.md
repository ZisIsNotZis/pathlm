# 02 — Engine skeleton + M0 probability-machinery probe

- **Status:** claimed
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

- [ ] Config-driven path sampling (knobs plumbed, defaults = base config)
- [ ] M0 training run completes with logged metrics
- [ ] Probe report answers (a)–(d) with evidence under `.scratch/02-engine-m0/evidence/`
- [ ] Fresh-context review of the diff passed

## Comments

- 2026-09-07 — agent (pi, volc2/glm-5.3-flash) — claimed; branch ticket-02-m0-probe.
