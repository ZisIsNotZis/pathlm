# 独立复核 R1（ticket 45）：X1（抖动补点）/ X2（搬运×深度）

## Review

Scope: read-only review of X1 `axes/G-policy/G5-parallel-lines/experiments/2627-mechanism-jitter/` and X2 `axes/M-carrier/M1-carrier-type/experiments/2627-carry-x-depth/` against their `README.md`/`analysis.md`/`results*.json`, the judgement/reconcile/mutation code, tests, and the git working tree. No files changed.

### Correct (verified with evidence)

1. **X1 pre-registered rule is faithfully executed (§4 → code → verdict).** `README.md:74-86` writes the rule; `jitter.py:14-18,36-53` implements exactly `range ≥ 0.0144 → downgraded`, `range < 0.005 → upgraded`, else `on_threshold`; `config.json` (`jitter_rule`) and `tests/test_mechanism_jitter.py:50-59,81-100` pin the same constants. `results_13m.json` jitter: `range=0.0035`, `range_over_effect=0.2431`, `verdict.state=upgraded_to_robust` — consistent (0.0035 < 0.005). Faithful.
   - Limitation: "written **before** the run" cannot be verified with the available (non-commit-range) tools; I confirmed only internal consistency and faithful execution.

2. **X1 "weight re-eval bit-identical" evidence is real, not circular.** `run.py:170-190` re-loads the 2626 checkpoints (sha-256 asserted, `run.py:82-90`), re-runs the K-curve and compares the **fresh** values to the recorded `source_v2` constants; `results_13m.json::cross_run_reproduction` shows `l1 1.4089/1.4389, repr01_A 1.3945/1.4277`, `all_equal=true`. This does support "抖动只来自训练、不含评测".

3. **X2 2×2 interaction arithmetic is correct.** `carry_x_depth_arms.py:100-118` defines `interaction = (both−carry) − (depth−base)`; recomputed from raw counts:
   - recall (in-dist): base 242/16384, carry 5862, depth 241, both 5924 → `depth@carry_on = 0.3784pp`, `depth@carry_off = −0.0062pp`, **interaction = +0.3846pp** ✓ (`results.json::increments.train_task_indist`).
   - tf@4 (synth): 87.84/88.53/99.84/94.47 → `depth@carry_off = 12.00`, `depth@carry_on = 5.94`, **interaction = −6.06pp** ✓ (`…increments.tf4_synth_indist`).

4. **X2 "untested factor" declaration is sufficient.** `README.md §4b/§4c`, the scope declaration, `analysis.md §2`, `controls/README.md §5`, and `carry_x_depth_arms.py` docstring all explicitly state that `ragged 随机退出 + proj-fill` (the `fill_kv=True` non-dense recipe behind the 100M `tf@4=0.9726`) is **not tested** and forbid extrapolation; ENG2 is named as follow-up.

5. **Numbers ↔ results.json correspond (spot-checks ≥10, all pass).** X1: repr01_A K1 1.4277/K8 1.3945; repr01_B K1 1.4237/K8 1.3910; net_A 0.0144 / net_B 0.0179 / range 0.0035; clean_tax −0.0112/−0.0152; gain 0.0332/0.0327; paired CI [−0.0045,−0.0024]; battery 1.4169/1.4132 (Δ−0.0037); det control 1.3826==1.3826 gain 0; JS 0.0223/0.0237/0.0233; wall 27.4 min, peak 4995.8 MB, `repr_decorr_last` 0.004868, `skip_frac_mean` 0.8993, `gate_skips` 0; sha256 `488732cc…`/`5e81e193…`. X2: n=16384, hits 242/5862/241/5924; tf4 87.84/88.53/99.84/94.47; bpc 55.07/52.36/125.74/1322.98; params 13,536,768 / 13,801,064 (+264,296); `n_exit_ce` 1/1/13/13; answer-CE slopes −0.01065/−0.06219/−0.01137/−0.08063; eval cost 0.557 min / 572.4 MB. `reconcile_report_13m.json` 62/62 pass; `reconcile_report.json` 175/0 pass.

6. **Deliverables in git:** all README/analysis-listed artifacts of both lines are tracked (watchdog_diff shows no working-tree delta and only two unrelated `.scratch` untracked files). `*.pt` correctly absent (documented). The one exception is Finding 2.

### Finding: P1 — X1 §17.17 component-decomposition identity does not reproduce

- **Issue.** `README.md:80` states "净效应 = 集成增益（K1−K8）− clean 税（vs l1）", with `ensemble_gain = repr01@K1 − repr01@K8` and `clean_tax = repr01@K1 − l1@K1`. With the values in `results_13m.json::jitter.component_decomposition` this yields `0.0332 − (−0.0112) = 0.0444`, **not** the reported net `+0.0144`. The same wrong identity appears in `analysis.md §4` ("净效应（= 增益 − 税）") and `results_13m.json::jitter.component_decomposition.note`.
- **Evidence / root cause.** `net = l1@K8 − repr01@K8 = (repr01_gain − l1_gain) − clean_tax`, where `l1_gain = l1@K1 − l1@K8 = 1.4389 − 1.4089 = 0.0300` is a term the table omits. `(0.0332 − 0.0300) − (−0.0112) = 0.0144`. The `l1` arm itself has a K1−K8 ensemble gain that must be netted out.
- **Why it slipped through.** `reconcile_numbers.py:88-96` checks each component's *definition* but never checks the stated identity, so it reports 62/62 while the formula is false. Individual values and the `upgraded_to_robust` verdict are unaffected.
- **Smallest fix.** Restate the identity as `net = (ensemble_gain_repr01 − ensemble_gain_l1) − clean_tax` and add `l1`'s ensemble gain (0.0300) to the §4.2/§4 table; or explicitly define `ensemble_gain` as the *differential* (`repr01_gain − l1_gain`). Add one `ck(...)` line to `reconcile_numbers.py` asserting `net == (gain_A − gain_l1) − tax_A`.

### Finding: P2 — X2 declared deliverable `train_<arm>.log` is not in git

- **Issue.** `README.md §7` and `MODEL.md` (whose derived-artifact table is headed "全部入库") list `train_<arm>.log` (`train_base.log`, `train_carry.log`, `train_depth.log`, `train_both.log`) as delivered artifacts, and `analysis.md §7` cites them as the source of the training wall-clock. The repo-wide `.gitignore:29` (`*.log`) ignores them, so none are in git.
- **Impact is bounded:** the cited numbers survive in tracked `train/<arm>/results.json` (`wall_minutes` 4.97/9.88/5.67/9.66; peaks below). Fix: drop `train_<arm>.log` from the delivered-artifacts lists (or add `!axes/…/2627-carry-x-depth/train_*.log` if the raw stdout is truly wanted).

### Finding: P2 — X2 cited "training peak memory" does not match the tracked training file

- **Issue.** `README.md §6` and `analysis.md §7` give training peak memory as `2015 / 2282 / 2095 / 2361 MB`. The tracked `train/{base,carry,depth,both}/results.json` record `2015.3 / 2283.4 / 2095.3 / 2362.0`. The cited `2282`/`2361` match the **400-step smoke** run (`smoke/steps400/*/results.json`: 2282.5 / 2361.5), not the 24k training.
- **Fix:** relabel as smoke-run peaks, or correct to 2283/2362. (Both are far under the 5 GB line, so §20.2 compliance is unaffected; not in the reconcile CITED list.)

### Notes (P2, minor)

- X1 `analysis.md §9` says "按臂取/放 GPU 锁，sleep 65 后释放（§20.1）"; `train_all.sh` is single-arm and does not `sleep 65` before `release_lock`. Harmless wording, but inaccurate.
- `tests/test_mechanism_jitter.py` docstring calls the semantic mutation "M4", while `mutation_report.json` labels it "M5".
- §20.1/§20.2/§20.3 accounting is otherwise present and consistent: X2 per-arm lock + 65s windows (`train_arms.sh`), X1 whole-arm hold with 60s heartbeat; step-time measured before budgeting (`smoke/steps400`, 250-step 72.0 ms/step); peaks ≤4995.8 MB (X1) / ≤2362 MB (X2), `nvidia-smi` free recorded before/after for X1.
- Cannot verify (tool limits): commit ordering / whether `catalog.yaml`/`INDEX.md`/`axes/**/README.md` were touched — `watchdog_diff` does not cover committed ranges and no git-history tool is available; likewise code sha-256 fingerprints in `MODEL.md` were not recomputed.

### Merge verdict: **OK with notes**

Conclusions and all data are sound; the P1 is a required-deliverable formula that must be corrected before the wave is accepted, and the two P2s are report accuracy fixes.