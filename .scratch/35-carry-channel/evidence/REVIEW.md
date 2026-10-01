# 独立复核（契约 V，fresh context，只读）

## Review

I reviewed contract U's deliverables (`axes/M-carrier/M1-carrier-type/experiments/2618-carry-channel-13m/`) against the pre-registered criteria, the implementation (`pathlm/depth_ar.py`, `train_depth_ar.py`), the tests (`tests/test_depth_ar.py`), and the raw evidence (`results.json`, `train/*/train_log.jsonl`, `mutation_report.json`, `divergence.json`, `premise_check.json`, `run_log.txt`, `smoke/*`). I do not have shell access, so pytest was not run (left to the main agent).

### Checklist (SCHEMA §14)

**① Criteria line unchanged — PASS.** `README.md` §4 reproduces the contract's 判据 block verbatim (items 1–6), and explicitly marks the remainder as 摘要 ("以下为**摘要**，非逐字…"). No wording drift from `.scratch/35-carry-channel/issues/01-carry-channel.md` 契约 U.

**② Three reconciliations (SCHEMA §12) — PASS.** `analysis.md` §7.1 gives arithmetic/task/口径 legs. Spot-verified the arithmetic: `32·ln64/255 = 0.5219` ✓; n=4096 CI half-width ≈0.0090 ✓; diff CI `[7.16, 9.10]pp` = `results.json::comparisons.U2_vs_U0.diff_ci_95` ✓; U2-vs-U1rep diff CI `[5.10, 7.21]pp` reproduced from `two_proportion` ✓. Task leg states same task/seed/position set/mask; 口径 leg states snapshot + accuracy, "bpc 未用作判据".

**③ Control-arm 3 elements + same task — PASS.** `controls/README.md` §1–3 spell out, for each control, the input distribution, what it predicts, and its deployment meaning. `analysis.md` §7.1 confirms no "fake-context + true-target" control.

**④ Intervention liveness, every 200 steps — PASS.** Every `train_log.jsonl` row (61 per arm, 200-step cadence) carries a non-null `attn_carry` block with `state_fro / beta_mean / write_rel`, non-zero throughout (U1 `state_fro` grows 22.7→71.8; U2 bounded ~2.5–3.5). The terminal `carry_liveness` block's `write_rel=0 / n_chunks=1` artifact is honestly disclosed in `stability.md` §6.

**⑤ Evidence traceability — MOSTLY PASS (3 numbers fail).** ≥18 spot checks match exactly (`results.json` ↔ `analysis.md` ↔ `train_log.jsonl`): accuracies 0.013672/0.000732/0.094971, Δ +8.13pp/+9.424pp, corrupt 0.008057, params 13,536,768/13,801,064/+264,296, walls 11.12/13.48/16.26, peaks 2015.3/2169.5/2282.4, slopes −0.001738/+0.026008/−0.053230, K-sweep and distance-leg values, U1rep 0.033447 and Δ+6.152pp, answer-CE 4.184/1.4772/2.2524, activity means (U1 state_fro 50.80→71.84, β 0.307, write_rel 0.647; U2 read_scale 0.0917, read_share→0.7078) all recomputed from the logs. Mismatches listed under Findings.

**⑥ 口径标注 — PASS.** Accuracy-vs-bpc (bpc explicitly not used; CE reported instead), snapshot-at-12k stated, analytic vs measured memory separated (`memory_analytic` vs `memory_measured`).

**⑦ Regression/pytest — NOT RUN.** No shell available; flagged for the main agent. Expectation from `stability.md` §5: 277 passed (baseline 260 + 17). I verified the new tests exist and target real seams (17 `test_carry_*`/`test_mqar_*`), and that `tools/evalkit/longrange_tasks.py` is purely additive (+82 lines, one new function; verified via `watchdog_diff`), and `retention.py` is untouched.

### Special focus (this round)

**a) Causality — PASS (the fatal-bug risk is well covered).** The sequential reference (`carry_scan_sequential`) is causal by construction (position t uses only token t + carried `S`). The training path is the chunked WY form (`_carry_chunk_delta`), and `test_carry_chunked_matches_sequential` pins it to the sequential reference in fp64 at `<1e-12` for chunk ∈ {1,2,3,5,8,24} (chunk=24 = whole sequence in one chunk) ⇒ chunked is causal. `test_carry_channel_is_causal_only` perturbs `q/k/v/β` at t≥10 and asserts outputs at t<10 are `torch.equal` for chunk ∈ {4,1}; `test_carry_channel_causality_end_to_end` asserts logits at t<7 are bitwise unchanged. Read-after-write convention (`S_t` includes t's own write) is pinned by `test_carry_read_convention_includes_own_write`. Mutation M1 (release the intra-chunk `keep` mask by one ⇒ future leak) is caught red. No future-information path found.

**b) U0 ≈ chance premise — PASS (not "window failed").** `window_mask_probe.max_abs_logit_diff_masked_vs_open = 2.106155 > 0`; queried pair distance 241–255, `frac_outside_window = 1.0`; `local_prior = 0.0` (K=8); U0 acc 0.013672 with CI [0.010544, 0.017711] ⊇ chance 0.015625; and `test_carry_channel_carries_across_window_but_baseline_does_not` proves the baseline's pos-10 output is bitwise invariant to a change at pos-0 (window=4) while channel arms are not. U2 hitting 9.5% on the same trunk shows the task is learnable when info is reachable.

**c) U1/U2 bitwise equal params, unique variable = write rule — PASS.** `selfchecks.params_bitwise_equal_U1_U2` = true (188 tensors, empty name/shape/value diffs, 13,801,064 both); `trunk_bitwise_equal_carry_vs_U0` = true; `identity_start_logits_bitwise` both true. `test_carry_plain_and_delta_params_bitwise_equal` asserts the only differing config field is `carry_rule`. (Note: contract said "N 个记忆槽"; the delivered form is a single shared slot — an explicitly recorded, main-agent-approved §17.8 implementation decision in `README.md` §3 that applies identically to U1/U2, so single-factor holds.)

**d) Corrupt-source control — PASS (executed and correctly used).** `corrupt_source` present for all 3 arms × 3 K and all distance legs; `criteria.c4_corrupt_source_control` invalidates none; U2 drops 9.50%→0.81% (CI [0.005743,0.011293]), well below the pre-stated ">4×chance" invalidation bar; U0/U1 already at floor.

**e) bpc not the main criterion — PASS.** Main leg is accuracy + Wilson CI; `analysis.md` §5.1 explicitly reports "clean bpc 不适用" and uses answer-row CE; bpc is only mentioned to explain why it isn't used.

**f) Extrapolation / unsourced numbers — 4 issues (P2).** See Findings. Note the distance curve is explicitly *not* extrapolated (§6.3: "只描述形状，不做外推") — that part is done correctly.

### Findings

- **Finding: P2 — eval-battery wall time inconsistent across evidence.** `analysis.md:3` and `:292` say **2.57 min**; `README.md:111` says **2.04 min**; but `results.json::cost.wall_minutes_total_eval = 2.01` and `run_log.txt:35` prints "评测墙钟 2.01 min". Smallest fix: change analysis's 2.57 → 2.01 and README's 2.04 → 2.01. (Root cause: `reconcile_numbers.py:160`'s check labeled "评测墙钟 ≈ 2.57 min" actually asserts `eval_peak_gpu_mem_mb == 521.5`, so the mismatch was never caught.)
- **Finding: P2 — step-time numbers not present in the cited logs.** `analysis.md:151` / `stability.md:89` attribute "63.2 / 221.1 / 410.5 ms/step" to `smoke/smoke_u{0,1,2}/train_log.jsonl`, but those logs contain no ms/step field; only a cumulative `min` field exists, from which one derives ≈**43 / 208 / 396–410** ms/step. U1/U2 ≈ match (conclusion unchanged), but the U0 value (63.2 vs derivable 43) is off ~45%. Smallest fix: cite the derivable values or state the measurement source.
- **Finding: P2 — "earlier trial at step 3600, answer CE 3.5015" is unsourced.** `analysis.md:47`, `stability.md:60`, `divergence.py:5`. No preserved log shows 3.5015 at step 3600 (U1rep@3600=4.2409, U1@3600=4.2021, U2@3600=1.8232, U0@3600=4.411); in the preserved data 3.5015 is U1rep @ **step 5000**. Smallest fix: correct the step number or drop the unpreserved-trial claim.
- **Finding: P2 — code fingerprint mismatch for `pathlm/depth_ar.py`.** `mutation_report.json::targets["pathlm/depth_ar.py"]` = `72499b5b…` vs `results.json::code`/`MODEL.md` = `a141ef42…`, while `MODEL.md` claims "同一修订". The other two targets match. The mutated snippets still exist verbatim in the current file (M1–M5 remain locatable), so coverage is plausibly intact — the difference is likely the later `write_rel`-excluding-chunk-0 edit — but the mutation verification is not pinned to the fingerprinted revision. Smallest fix: re-run `mutate.py` on the current revision, or drop the "同一修订" claim.
- **Finding: P2 — forward-batch decomposition doesn't sum.** `analysis.md:292` writes "864 个前向批：3臂×3K×64批×2 + 3臂×6距离×16批×2" which sums to 1728; `run.py:369` defines `n_forward_batches = 64·3·3 + 16·6·3 = 864` (no ×2). Smallest fix: remove the "× 2（主+毁源）".

### Merge verdict

**OK with notes.** The pre-registered criteria (①②③④) are correctly evaluated, and the three highest-risk items (causality, parameter-equality, corrupt-source control) are all properly grounded in code + failing-testable units + mutation checks. All findings are P2 evidence-fidelity issues (stale/unsourced numbers and one fingerprint inconsistency); none change the judgment. `pytest tests/ -q` should be run by the main agent (no shell here).