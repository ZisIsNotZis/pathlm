# 01 — Retry-matrix validity: mixture-reentry gauge + run comparability

- **Status:** claimed
- **Type:** defect fix + measurement correction
- **Related:** 10-gap-sweep/issues/01-gap-sweep (owns the T1 runs this ticket corrects)
- **need-review:** true (engine behaviour change)
- **need-test-cases:** true (regression test pinning the re-entry norm regime)

## Issue

The latent-retry matrix in `findings.md` (transports × {overwrite, mixture},
rows C1/C1M/C2/C2M/C3/C4/C5/R1) is not a valid comparison. Four independent
defects, all measured on the existing checkpoints:

### D1 — stage-E norm cap applied to stage-L states (the gauge defect)

`_mixture_reentry` ends every branch with `cap_norm(..., mcfg.norm_cap)`
(`pathlm/model.py:356-361`), but `norm_cap = 1.0` is the **stage-E embedding**
cap. Measured on the C1M checkpoint (16×512 enwik8 eval window):

| quantity | norm |
|---|---|
| embedding row mean | 1.19 |
| `E[t] + pos` (clean stage-1 input) | 1.00 |
| layer output, depth 12, per round | 28.7 → 48.4 → 68.1 → 88.2 |
| `_mixture_reentry(direct)` output | **1.00** |
| direct overwrite re-entry (`_transport` direct, uncapped) | **28.7** |
| `_mixture_reentry(linear)` output (‖proj_E h‖/‖h‖ ≈ 71.7 before the cap) | **1.00** |
| `_mixture_reentry(soft)` output | 0.90 (cap is a no-op) |

Consequences:

- Mixture re-entry divides the state by ~29–88× — it is a rescale, not a blend.
- `_transport(direct)` returns `h` **uncapped** (`pathlm/model.py:196`) while
  `_mixture_reentry(direct)` caps: the two arms do not measure the same thing
  at any round.
- The reported ordering soft > linear > direct is monotone in *how much the cap
  destroys* (soft's cap is a no-op; linear's projection amplifies 72× and is
  then crushed; direct's raw state is crushed), not in information preserved.
- Distinguishing evidence: for **soft**, mixture and overwrite are
  bit-identical at round 1 (`softmax(logits_0)@E` either way) — measured
  `max|d| = 23.3726, cos = 0.92109` in both arms. For **direct** they are not.

### D2 — batch-2 configs omitted `n_mtp` (comparability defect)

`configs/C1M.json` and `configs/C2M.json` do not set `n_mtp`, so `PathConfig`
defaulted them to `n_mtp = 2`, while every cell they are tabulated against
(`B0`, `C1`, `C2`, `C3`, `C4`, `C5`, `R1`) pins `n_mtp: 1`. `B2` measures that
knob alone: bpc 1.5481 vs B0's 1.5074 = **+0.0407**.

Control runs at `n_mtp: 1` (`evidence/C1M_n1`, `evidence/C2M_n1`):

| run | n_mtp | mix | bpc | Δ vs B0 |
|---|---|---|---|---|
| C1 (direct) | 1 | no | 1.6549 | +0.1475 |
| C1M as published | 2 | yes | 1.7010 | +0.1936 |
| **C1M_n1** | 1 | yes | **1.6645** | **+0.1571** |
| C2 (linear) | 1 | no | 1.6621 | +0.1547 |
| C2M as published | 2 | yes | 1.7089 | +0.2015 |
| **C2M_n1** | 1 | yes | **1.6548** | **+0.1474** |

The published `direct +0.194` decomposes into +0.0365 (n_mtp knob) and
+0.0096 (D1's gauge). Corrected, linear×mixture is *better* than its own
overwrite by 0.0073 — the reverse of the published reading.

### D3 — the accumulator is never trained where it acts

`p_retry` samples `n_retries = 1 if rng.random() < p_retry else 0`
(`pathlm/config.py:119`), so training never exceeds 2 passes. The accumulator
only differs from overwrite at r≥2, i.e. only in the inference-time forced
`rounds=k` eval loop. Every conclusion about r3/r4 re-entrant state quality is
extrapolation of an untrained rule, not a trained mechanism.

### D4 — no replication, noise floor unmeasured

`experiments.md` §Preamble mandates 2 seeds for expected sub-5% deltas; every
matrix cell is single-seed. Fixed-seed training is nondeterministic: two
identical 600-step runs (seed 0) diverge from the first logged step (step-200
loss 8.6881 vs 8.6275) and land at bpc 3.3108 vs 3.3122.

## Acceptance criteria

- [ ] D1 resolved: the re-entry gauge contract is decided and applied to BOTH
      `_transport` and `_mixture_reentry` (stage-1 ⇒ both capped; stage-L ⇒
      neither capped); a regression test pins the re-entry norm regime so a
      stage-E cap can never be silently applied to a stage-L state again
- [ ] D2 resolved: C1M/C2M re-run at `n_mtp: 1` recorded in evidence (done)
      and the published table corrected; configs pin every path knob explicitly
      rather than relying on defaults
- [ ] D3 resolved: retry depth sampled geometrically (mean ≥ 2) so the
      accumulator is trained at the rounds it is evaluated at
- [ ] D4 resolved: BOTH variance sources measured per cell — run-to-run
      nondeterminism (0.0019 bpc @ 6000 steps, same seed) and seed variance
      (measured 0.0001 to 0.0138 bpc, cell-dependent). Operative rule: no cell
      ordering is claimed unless it survives ≥2 seeds AND |Δ| exceeds that
      cell's measured seed spread — a single global floor is not sufficient
- [ ] D5 resolved: effectiveness convergence measured — the delta between two
      cells tracked over a 4× training horizon (24000 steps, shared LR
      schedule, ckpt-every 1000) via `probe_curve.py --compare`. A delta whose
      |Δ_last|/|Δ_first| → 0 is a convergence-rate difference, not a tax, and
      must be re-labelled as such in findings.md
- [ ] findings.md Latent-retry section reflects the corrected numbers and
      states which conclusions survive (large effects only) vs which are
      unresolvable at current replication

## Comments

- 2026-09-11 — agent (pi) — created; claimed. Origin: user challenge to the
  frozen Latent-retry section ("direct mixture worse than override does not
  make sense — are we sure this is done properly?"). Verified against
  checkpoints rather than the summary; D1–D4 found. Evidence for the D1
  geometry and the D2 controls staged under `evidence/`; probe scripts included
  so the geometry measurement is reproducible.
- 2026-09-11 — agent (pi) — D4 measured. Same-seed nondeterminism 0.0019 bpc;
  seed variance 0.0001 (C3) / 0.0050 (C1M_n1) / 0.0138 (C2M_n1) — cell-
  dependent, and for C2M_n1 larger than the whole six-cell latent spread
  (0.011). The linear mixture-vs-overwrite delta **reverses sign between seed 0
  and seed 1** (−0.0073 → +0.0065). Consequence: the ticket will NOT publish a
  reversed ranking; the corrected conclusion is "corrected values, no
  resolvable ordering among latent cells". A third confident reading would
  have been wrong again.
- 2026-09-11 — agent (pi) — D5 instrumented. `train_m1.py` gains
  `--ckpt-every`/`--ckpt-dir`; new `probe_curve.py` builds effectiveness-vs-step
  curves from intermediate checkpoints and `--compare` prints the step-wise
  delta with a shrinking/growing trend. Horizon runs B0/C1/C1M_n1 at 24000
  steps with a shared schedule are in flight; all new numbers live in `.tmp/`
  and are promoted here on completion (weights stay out of git — 31 committed
  `model.pt` blobs already account for ~1.6 GB of `.git`, recorded as tech
  debt; removal is a history rewrite and therefore user-gated).
- 2026-09-11 — agent (pi) — D5 first result (partial). The 24000-step horizon
  attempt was truncated by a Pi session reload (`bg_run` wrapper SIGTERM'd, not
  OOM); checkpoints survived to step 8000 (B0) / 5000 (C1, C1M_n1) and yielded
  a usable early trend via `probe_curve.py`. Retry tax (C1−B0) falls +0.3637 →
  +0.2135 → +0.1580 → +0.1487 → +0.1385 over steps 1000–5000
  (|Δlast|/|Δfirst| = 0.38), with decelerating decrements (−0.150, −0.056,
  −0.009, −0.010) — a large early component is convergence-rate, but it does
  not yet trend to zero. Mixture delta (C1M_n1−C1) shows no monotone trend
  (+0.0009…+0.0154), matching D4. Relaunched detached with `setsid nohup` for
  the full 24000; the annealed tail decides the asymptote. Curves:
  `evidence/curve_*_partial.jsonl`.
- 2026-09-11 — agent (pi) — process note: long runs must be launched detached
  (`setsid nohup … &`), not via the harness background wrapper, which a session
  shutdown/reload kills. Checkpoints, not the wrapper, are the durable
  artifact — which is why adding `--ckpt-every` before the long runs was worth
  doing first.
- 2026-09-11 — agent (pi) — model-stacking idea assessed with measurement, not
  judgement: solo 39-48 ms/step, 3-way concurrent 98-113 ms/step, aggregate
  ~1.1-1.3x, GPU 100% util, 5.75 GB/run capping concurrency at 3-4. Batch
  scaling (B0, 400 steps, solo): batch 32 = 30 ms/step / 1.83 us-token, batch
  64 = 45 ms/step / 1.37 us-token — so ~25% of per-step cost IS fixed overhead
  (launches/Python), which means the merged-op rewrite is not useless in
  principle. Verdict stands on cost/risk, not on impossibility: the same ~25%
  is capturable by raising batch size, with no grouped-GEMM rewrite and no
  correctness risk, at the price of changed optimisation dynamics.
