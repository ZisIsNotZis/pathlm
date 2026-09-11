# M3 interactions + diversity probe — the program's closing results

## Interaction runs

| Run | bpc | Δ vs B0 (1.5074) | Key numbers | Verdict |
| --- | --- | --- | --- | --- |
| IX1 mask × repair-in-context | 1.5944 | +0.087 | repair 57.2→58.3% | ✅ SUB-ADDITIVE — cheaper than I1(+0.127)+C3(+0.150) separately; the one genuine composition win |
| IX2 skip+exit+shuffle+redo | 2.287 | +0.780 | loc-1.0 → 3.48 | ❌ depth knobs interfere |
| IX3 shuffle × retry | 2.2293 | +0.722 | repair 35.8→38.4% (vs C1's 51.4) | ❌ order-freeness breaks repair |
| IX4 retry + consistency loss | 1.6589 | +0.151 | ≈ C1 (1.655) | ➖ consistency loss neutral on chaining |
| INT everything on | 2.2691 | +0.762 | all buckets degraded | ❌ full grammar does not compose at 13M |

**The composition law: shuffle is the poison.** Every combination containing
shuffle_locality lands at 2.2+ bpc. Order-freeness and self-repair are
ANTAGONISTIC — the retry loop cannot refine what shuffle scrambled, and
dense-exit calibration collapses on shuffled depths. Excluding shuffle, the
grammar composes (IX1 proves sub-additivity is achievable).

## Diversity probe (DIVL1 = p_skip 0.1 + w_diversity)

Three training attempts: v1 diverged at step 4600 (bf16 + negative-JS), v2
silent-NaN after step 2200 (grad explosion post-clamp), v3 stable with the
grad-norm guard. Engineering findings: reward terms need float32 + gradient
sanity gates under bf16 autocast.

| Model | K=1 | K=4 | K=8 | ensemble gain |
| --- | --- | --- | --- | --- |
| L1 (plain skip) | 1.6926 | 1.6543 | 1.6454 | −0.047 |
| **DIVL1** | **1.6842** | **1.6339** | **1.6227** | **−0.062** |

Diversity pressure improves BOTH the baseline (−0.008) and the ensemble slope
(−0.062 vs −0.047), and pushes through L1's K8 ceiling by 0.023. **Mechanism
confirmed: paying paths to decorrelate converts to test-time gain.** But the
absolute number (1.6227) still sits far above B0 — the tax remains dominant
at 13M params. The lever now has a proven knob; scale is the untested variable.

## Program-level verdict

1. The elastic grammar's deployment capabilities (repair, early-exit, eviction)
   are real and demonstrated. Its prediction-side cost is real too.
2. Composition works ONLY where mechanisms share machinery (IX1) and fails
   where they fight (anything × shuffle).
3. Diversity training is the first intervention that moved the
   tax-vs-ensemble math in the right direction on both terms.
