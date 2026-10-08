All checks pass. The task is complete.

## Summary

**Task**: Continue the interrupted M1 (carrier) GPU line — run the evaluation and write the judgment for the three-arm (none/plain/delta carry-write) × 2-seed experiment, whose *training* was already finished.

**What I found and did**
- Training artifacts were complete (6 checkpoints + logs), but the eval segment had never run and `config.json` still held `TO_FILL_AFTER_TRAINING` hashes.
- Ran the evaluation (`eval.sh` → `fill_config_hashes.py` + smoke + full battery) on the **same checkpoints and same eval stream**, producing `results.json`'s eval section. Deterministic across 3 runs (identical numbers; bitwise re-eval confirmed).
- Fixed **three wiring defects** in the line's own directory, each revealed by actually running the code:
  1. `run.py`: eval-time `carry_liveness` had no `scan_tele`/`layer_stats` → `alive` was spuriously `False`; changed the primary reading to the **training-time** liveness (the authoritative one) and kept the eval-time probe as `eval_recheck` (mirrors the parent 2618's pattern).
  2. `run.py`: `arm_accuracy_spread_across_seeds.*_pp` was computed from **fraction** accuracies (mislabeled by 100× — e.g. plain's range showed `0.342` instead of `34.2pp`, which could invert a stability reading); now correctly ×100 with `per_seed_pp` added.
  3. `reconcile_numbers.py`: was written against an earlier API and had never run — added mandatory explicit tolerances, element-wise list comparison, string/`None` handling, eval-matched distance recomputation, and `window_mask` read from the raw train log.

**Key result (in-distribution multi-query recall, 32768 queries, chance 1.5625%)**: none-arm ≈ chance (1.55%/1.65%); carry arms ≈36%. `delta−none` = +34.9/+34.3pp, paired-clustered CI lower >0, range 0.60pp → **stable**. `delta−plain` = +0.37pp (CI[−0.15,+0.90], crosses 0) / +34.06pp → **not stable** (plain is bistable: 36.1% vs 1.9%; delta range 0.51pp). Damage-source control valid; delta is the best arm on **both** seeds.

**Required one-liner**: 「delta 写优于 plain/无通道」**不成立（作为合取）**——拆开看「delta 优于无通道」成立，「delta 一贯显著优于 plain」不成立（seed0 在噪声内打平）；**delta 仍是三臂中最稳、且两种子上都最好的那一支**。

GPU lock discipline was respected: verified absent before first run; the Spec line demonstrably **waited** for my lock (it began training only after my eval released). Another line's files were left untouched.