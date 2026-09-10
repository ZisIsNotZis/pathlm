# M2 breadth sweep — 7 runs (I4 restarted after a silent external kill)

Base anchor B0 = 1.5074 bpc. Deltas are capacity taxes (clean-input bpc).

| Run | bpc | Δ | Key number | Verdict |
|---|---|---|---|---|
| I2 `[mask]` | 1.5661 | +0.059 | repair 59.4% (I1 wrong-token: 52.8%) | flagged corruption is cheaper AND more repairable than silent corruption |
| I3 emb noise | 1.5073 | ±0.000 | — | noise is free robustness (norm cap works) |
| I4 pure-noise | 1.5704 | +0.063 | 15% of positions carry NO token info | full writable-input claim costs only ~4% rel. PPL |
| C2 linear+retry | 1.6621 | +0.155 | repair +2.0pp, saturates r3 | linear ≈ direct ≈ soft — transport flavor doesn't matter; retry value is real but flavor-independent |
| L1 p_skip | 1.6822 | +0.175 | most expensive L element | skipping 10% of layer-executions costs more than corruption |
| L3 shuffle 0.5 | 2.0527 | +0.545 | loc sweep: 0→2.010, 0.25→2.013, 0.5→2.040, **1.0→3.844** | order robustness has a hard boundary: trained 0.5 tolerates ≤0.5, collapses at full random |
| L4 p_redo | 1.5515 | +0.044 | — | layer repeats are nearly free |

## Element-tax ranking (the breadth result)

CHEAP (≤0.07): redo, mask, noise, pure-noise — elements that preserve or flag token identity.
EXPENSIVE (0.13–0.55): wrong-token, skip, linear-retry, shuffle — elements that destroy identity or order structure. Shuffle is in a class of its own: layer ORDER carries ~0.5 bpc of value, and full-random order is unrecoverable.

## Notes
- I4's noise-robustness (self-acc through noise) was not measured — run_battery's repair gate checks only corrupt_wrong/corrupt_mask, missing perturb_noise/pure_noise. Minor; fix next time the battery runs.
- C2's retry gain (+2.0pp) matches C1 (+2.1pp) and C3 (+2.5pp): retry is flavor-independent at this scale.
