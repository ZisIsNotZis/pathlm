# Evidence — retry-matrix validity (ticket 11)

## D1 — mixture-reentry gauge geometry

Reproduce (GPU, needs `.tmp/enwik8` + `data/enwik8_full.npz`):

    python3 evidence/probe_mix2.py

`probe_mix.py` reports the round-1 logit comparison (soft: mixture ≡ overwrite);
`probe_mix2.py` reports re-entry norms and the vocab-span projection ratio.
Both load the published `C1M` checkpoint from
`.scratch/10-gap-sweep/evidence/C1M/model.pt` and swap only
`transport`/`reentry_mix` at eval time.

Measured (16×512 enwik8 eval window, `torch.manual_seed(1234)`):

| quantity | norm |
|---|---|
| embedding row mean | 1.19 |
| `E[t] + pos` | 1.00 |
| depth-12 layer output per round (direct) | 28.7 / 48.4 / 68.1 / 88.2 |
| `_mixture_reentry(direct)` | 1.00 |
| direct overwrite re-entry | 28.7 |
| `_mixture_reentry(linear)` | 1.00 (‖proj_E h‖/‖h‖ ≈ 71.7 pre-cap) |
| `_mixture_reentry(soft)` | 0.90 (cap no-op) |

Round-1 node-0 logits, soft transport: mix vs no-mix `max|d| = 23.3726`,
`cos = 0.92109` in both arms — identical, so the soft arm of the published
matrix could not have measured mixture at training depth.

## D2 — `n_mtp` controls

`C1M_n1` / `C2M_n1` are the published `configs/C1M.json` / `configs/C2M.json`
with `n_mtp: 1` forced (matching the rest of the table). Configs are
`.tmp/C1M_n1.json`, `.tmp/C2M_n1.json` (gitignored) — reproduce with:

    python3 train_m1.py C1M_n1 --config .tmp/C1M_n1.json --out-root .tmp/mixprobe

| run | n_mtp | mix | bpc |
|---|---|---|---|
| C1 | 1 | no | 1.6549 |
| C1M (published) | 2 | yes | 1.7010 |
| C1M_n1 | 1 | yes | 1.6645 |
| C2 | 1 | no | 1.6621 |
| C2M (published) | 2 | yes | 1.7089 |
| C2M_n1 | 1 | yes | 1.6548 |
| B0 | 1 | – | 1.5074 |
| B2 | 2 | – | 1.5481 |

## D4 — variance floors and per-cell seed spread (6000 steps)

Two distinct variance sources, measured separately. They are not
interchangeable and the second is ~10x the first.

**Run-to-run nondeterminism (same config, same seed).** `C3deta`/`C3detb` at
600 steps: step-0 loss identical (11.2572), step-200 loss 8.6881 vs 8.6275,
final bpc 3.3108 vs 3.3122. At 6000 steps (`C3`, `C3r_a`, `C3r_b`, all
`configs/C3.json` seed 0): bpc 1.6578 / 1.6596 / 1.6577 — **spread 0.0019**.

**Seed variance (different init + data order).** Same config, seed 1 vs
seed-0 mean:

| config | seed 0 | seed 1 | spread |
|---|---|---|---|
| C3 (soft, overwrite) | 1.6584 (n=3) | 1.6585 | **0.0001** |
| C1M_n1 (direct, mixture) | 1.6645 | 1.6595 | **0.0050** |
| C2M_n1 (linear, mixture) | 1.6548 | 1.6686 | **0.0138** |

Seed variance is cell-dependent (0.0001 to 0.0138) and for C2M_n1 alone
**exceeds the entire 0.011 bpc spread of the six latent cells**. There is no
single global floor: ordering claims need per-cell replicates.

**Sign reversal.** The mixture-vs-overwrite delta for linear transport flips
with the seed: C2 1.6621 (overwrite) vs C2M_n1 1.6548 (seed 0, mix better by
0.0073) and vs C2M_n1_s1 1.6686 (seed 1, mix worse by 0.0065). No ordering
between these cells is resolvable at n=1-2 seeds per cell.

Weights are deliberately not committed here (derived intermediates); the
committed evidence is `results.json` + `train_log.jsonl` per run. The 31
already-committed `model.pt` blobs (~1.6 GB of `.git`) are recorded as tech
debt in the ticket; removing them needs a history rewrite, which is
user-gated.

## Model-stacking assessment (measured, not judged)

Solo vs 3-way process concurrency: solo 39–48 ms/step, 3-way concurrent
98–113 ms/step, `nvidia-smi` 100% util, 5.75 GB/run capping concurrency at 3–4.
So there is no large idle pool.

Batch-scaling test (solo, sequential, clean GPU — `B0` config, 400 steps):

| batch | ms/step | tokens/step | µs/token |
|---|---|---|---|
| 32 | 30 | 16384 | 1.83 |
| 64 | 45 | 32768 | **1.37** |

Doubling batch costs only 1.5x time, so **~25% of per-step cost is fixed
overhead** (kernel launches / Python), not FLOPs. The `(M,B,T,d)` stacking idea
is therefore not worthless in principle — but the same overhead is capturable
by raising batch size, which needs no code change and no grouped-GEMM rewrite.
Verdict: do not rewrite; treat batch size as the lever, with the caveat that it
changes optimisation dynamics and therefore comparability.

## D5 — effectiveness convergence (interim, partial checkpoints)

First run of the 24000-step horizon runs was killed by a Pi session reload (not
a crash); it left checkpoints for B0 to step 8000 and C1/C1M_n1 to step 5000.
Those are enough for a real early trend (`curve_*_partial.jsonl`, built with
`probe_curve.py --batches 100`; bpc is the clean single-pass eval, so the extra
corruption/retry loss terms do not enter it — the delta below is genuine
generalisation gap, not training-loss bookkeeping).

**Retry tax (C1 − B0), same 24000-step LR schedule, paired eval batches:**

| step | B0 | C1 | delta |
|---|---|---|---|
| 1000 | 2.2289 | 2.5926 | **+0.3637** |
| 2000 | 1.8044 | 2.0179 | +0.2135 |
| 3000 | 1.6920 | 1.8500 | +0.1580 |
| 4000 | 1.6281 | 1.7768 | +0.1487 |
| 5000 | 1.5902 | 1.7287 | **+0.1385** |

`|Δ_last|/|Δ_first| = 0.38`, and the per-1000-step decrements are
−0.150, −0.056, −0.009, −0.010 — a fast early collapse followed by a
near-plateau around +0.14. So a substantial part of the published +0.147 tax is
a **convergence-rate difference**, but it does not (yet) look like it decays to
zero: the annealed tail (steps 18000–24000) decides whether the asymptote is
~0.13 or much smaller. Unresolved until the relaunch completes.

**Mixture delta (C1M_n1 − C1)**: +0.0009, +0.0047, +0.0031, +0.0154, +0.0132 —
no monotone trend, fluctuating across a 0.015 range. Consistent with the D4
conclusion that the direct-mixture penalty is at the noise level.

Caveat: n=1 seed per cell, 100-batch bpc subset. The delta is paired (identical
eval batches for both configs), which suppresses eval-sampling noise but not
seed variance.

## GPU note

Five concurrent 13M runs OOM at 23.5 GiB (~5.4–5.8 GiB each). Concurrency for
this ticket's runs is capped at 2–3.

Long runs are launched detached with `setsid nohup … &` — the harness `bg_run`
wrapper was killed by a Pi session shutdown/reload, which is what truncated the
first horizon attempt (checkpoints, not the wrapper, are the durable artifact).
