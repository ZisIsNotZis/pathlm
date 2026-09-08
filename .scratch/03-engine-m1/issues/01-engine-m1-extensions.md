# 01 — Engine extensions for M1

- **Status:** claimed
- **Type:** implementation
- **Blocked by:** 02-engine-m0/issues/01-engine-m0-probe
- **need-review:** true
- **need-test-cases:** true

## Issue

Extend the engine so every M1 run is config-only (no code per run):

1. **Transports** (stage 3): implement `linear` (h → U·Eᵀ projection → re-enter at 1) and `soft` (h → softmax(·Uᵀ)·E → re-enter at 1) as retry re-entry transforms. M1 scope: the transform applies on re-entry only; the full within-pass "downstream attends to expected embeddings" mechanism is deferred to a later ticket.
2. **Dense early-exit supervision** (L2): run full depth, apply the MTP block at every depth d = 1…n, CE + confidence BCE on each — variance-free per design §6. Inference: exit at depth d when prob₀ clears a threshold; quality-vs-depth curve in eval.
3. **Eviction training + anchor slots** (X2): per batch, sample window W; attention may only see the last W positions plus a small set of anchor positions (exempt from eviction, long-range channel). Loss on all positions. Eval: needle-in-haystack must not collapse; decode must work without re-prefill.
4. **Eval battery**: bpc on the standard enwik8 split (first 90M train / last 10M eval), repair accuracy, needle-in-haystack, decode speed per path config, quality-vs-rounds and quality-vs-depth curves, ensemble-vs-best-single. Metrics written to results.json per run.
5. **Base-scale config**: d=256, 8 layers, seq 512, full enwik8; configs as committed files (`configs/*.json`), one per run.

## Acceptance criteria

- [ ] Each feature lands with a fail-capable test
- [ ] B0 smoke run (short) produces bpc in a sane range and all battery columns
- [ ] No M1 run requires engine changes

## Comments
- 2026-09-07 — agent (pi, volc2/glm-5.3-flash) — claimed; branch ticket-03-engine-m1.

- 2026-09-07 — agent (pi, volc2/glm-5.3-flash) — created from M0 lessons (see docs/experiments.md §0); scope fixed to config-only-runs goal.
- 2026-09-07 — agent (pi, volc2/glm-5.3-flash) — implemented: linear/soft transports (re-entry transforms), dense early-exit supervision (per-depth hook), eviction+anchors (attention mask), stage-4 token retry (discrete re-embed), KV-cache incremental decode (per-layer caches, eviction, exit threshold, prob0-driven retry), eval battery (bpc/repair/depth-curve/needle/speed), configs/*.json for all 7 runs. 23 tests green. Smoke run B0 60 steps: 13.08M params, bpc 4.15 (random 7.68), pipeline end-to-end OK. Deviations recorded in docs/experiments.md: base scale = 12 layers × mlp_mult 6 (8 layers cannot reach 10–15M); needle task = trained convention with purged filler; decode bounded by absolute pos-emb (total ≤ seq_len). Cache bug found by smoke: early exit diverges per-layer cache lengths — fixed with per-layer position lists.
- 2026-09-07 — agent (pi, volc2/glm-5.3-flash) — review round 1: REJECT (P0 decode-retry position reuse + 4 P1: ECE/ensemble unwired, conf-BCE missing from dense hook, R1 token-retry round unmeasured, anchor regime ~1% train / 0% eval; 5 P2s). All fixed: self.n restored in retry() (+ bookkeeping test), battery now emits ece_node0/node1 + ensemble_vs_best over per-round estimates, dense hook supervises conf BCE, repair() measures config-driven rounds incl. token-retry, needle anchor_frac=0.3 training + anchor-distance eval columns, window off-by-one aligned, cuda.synchronize guarded, token-retry path from seeded model RNG, experiments.md: corruption documented as part of C/R units + N=1 default resolved. 27 tests green; smoke re-run OK (bf16 autocast added — GPU is shared with a resident llama-server; OOM fixed).
