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

## D5 — effectiveness convergence

The interim answer (built from a run truncated at step 8000 by a Pi session
reload) pointed the same way and is superseded by the full 24000-step run
below; its curves are kept as `curve_*_partial.jsonl`.

### D5 final — 24000 steps (≈4.4 epochs)

24000 steps ≈ 4.4 epochs, one shared 24000-step LR cosine, ckpt-every 1000,
`setsid nohup`-detached, ~32–41 min per run. bpc via `probe_curve.py
--batches 100` on identical eval batches for both configs (paired). Results in
`B0long/`, `C1long/`, `C1M_n1long/`; curves in `curve_*_long.jsonl`.

**Retry tax (C1 − B0), pre-D1/D3 engine — the published configuration:**

| step | 1000 | 2000 | 3000 | 5000 | 10000 | 20000 | 24000 |
|---|---|---|---|---|---|---|---|
| Δ | +0.3644 | +0.2166 | +0.1498 | +0.1385 | +0.1325 | +0.1239 | **+0.1217** |

`|Δ_last|/|Δ_first| = 0.33`, and the curve **plateaus**: flat at +0.12–0.13
from step 3000 onward, with only a slow drift down thereafter. So the retry tax
is REAL, not a convergence-rate artifact — but ~20% smaller than the published
1.1-epoch +0.147.

**The more important number: B0 itself.** 1.5074 bpc at 1.1 epochs → **1.3459
at 4.4 epochs**. Every tax in findings.md is a 1.1-epoch snapshot and therefore
an **upper bound** on the asymptote; the rank ordering may survive, the
absolute values do not.

**Mixture delta (C1M_n1 − C1)**: −0.0046 → +0.0073, oscillating, no trend —
noise level, consistent with D4.

**Forced-round repair at 24000 steps**: C1 dips at r4 (0.5893 → 0.5893 →
0.5903 → 0.5875); C1M_n1 monotone (0.5907 → 0.5928 → 0.5933). The r4-dip
removal claim survives. Separately, pushing C1M_n1 to far-forced rounds in the
curve probe collapses repair (0.578 → 0.231) — the accumulator is untrained
past its trained depth (r≤1 here), which is D3's point made visible.

Caveat: n=1 seed per cell; bpc is a 100-batch subset (paired, so cell-vs-cell
deltas are clean, but absolute values are noisier than the 200-batch battery).

## GPU note

Five concurrent 13M runs OOM at 23.5 GiB (~5.4–5.8 GiB each). Concurrency for
this ticket's runs is capped at 2–3.

Long runs are launched detached with `setsid nohup … &` — the harness `bg_run`
wrapper was killed by a Pi session shutdown/reload, which is what truncated the
first horizon attempt (checkpoints, not the wrapper, are the durable artifact).
