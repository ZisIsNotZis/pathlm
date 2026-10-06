# 独立复核 R2（ticket 46）：E1（ENG2）/ G1（抖动补全）/ L1（T=2048）

I've completed the read-only review of the three experiment lines. Let me compile the findings.

## Review

**Scope / method.** Read-only review of the three committed experiments against `SCHEMA §14/§14b, §17.11/§17.14/§17.15/§17.16/§17.17/§17.18, §18–§23` and the E1/G1/L1 charters. `watchdog_diff` at reviewer-launch `HEAD 767ce19e` shows **no staged/unstaged changes and no untracked paths** for the repo and for each reviewed directory ⇒ the three directories and their evidence are committed (per-directory diffs empty). I could not execute `pytest`/`build_wiki`/`check_schema` (no shell) — see commands for the supervisor.

---

### ① E1 `2628-eng2-carry-ragged` — ENG2 default-path invariance & write-timing semantics

**Correct.**
- **Default path really is numerically untouched (verified from source, not just the claim).** The only config-validation change is *inside* `if cfg.carry_rule is not None:` (`depth_ar.py:1875-1898`, dense-guard removed); `DepthARModel._input_states` builds a slot only under `if self.cfg.carry_rule is not None:` (`:2141-2144`); default block class is the **imported, unmodified** `Block` (`:2044`, `model.py:59`, returns `(x, None)`); the non-dense forward's new telemetry capture (`:2212-2216`) writes only into `attn_carry_layers` and never alters numerics. `default_untouched` in `results.json` confirms `block_class=Block`, `has_carry_module/box=False`, `attn_carry_in_aux=False`, `n_params=41984`. Regression: `pytest_full.txt` = `549 passed`; `pytest_eng2.txt` `33 passed, 101 deselected` → 33+101 = **134** in `test_depth_ar.py` (the README's "134" is derivable, not fabricated).
- **Write-timing semantics are fully specified (what/when).** `README §9`, `analysis.md §3`, `CarryChannelBlock` docstring (`:1516-1527`) and `_input_states` docstring (`:2126-2138`) all state the same rule: the slot is scanned **once over h₀, before any layer/exit**, so **all positions (including exited ones) write their own h₀**; the read's effect-range is governed by the caller's `where(active,out,frozen)`. Single test `test_eng2_carry_write_timing_ignores_exit_depths` + mutation M5 pin it; `results.json::write_timing.slot_identical_full_vs_fixed/zero = true`.
- **Numbers match artifacts (≥10):** combos 4 entries with `read_share` 0.003395/0.003416/0.003651/0.003651; adapter consumption only on proj-fill `0.00286717`↔`0.00287`; smoke `params 14590568`, `peak 861.3 MB`, `wall 0.05 min`, `read_scale_absmean 0→0.000225`, `read_share 1e-06`, `n_layer_samples 12→240`, step 0/step 19 rows; `reconcile_report.json` `n_checks=75 n_failed=0` ↔ README "75/75"; mutation `5/5 caught_red`, `sha256_restored==sha256_original==45f94679…`.
- **Mutation harness is sound:** each mutation requires a unique anchor (`count(old)!=1 → SystemExit`) and restores bytes before re-greening.

**Finding: P2 — "逐位还原 v1" evidence is structural, not a numeric A/B.** `tests/test_depth_ar.py:3208 test_eng2_default_path_bitwise_v1` (and README §2/analysis §4) name the property "逐位" but the test only asserts *structural* absence (`Block` class, `carry_rule is None`, no `_carry_box`, no `attn_carry`); there is no frozen-v1 numeric golden comparison. Evidence: test body `:3208-3223`; no other v1-output comparison exists. This does **not** invalidate the claim — the source inspection above independently shows the default path is the same code — but the word "逐位/bitwise" overstates what the named test checks. Smallest fix: rename the test/claim to "default path uses the untouched `Block` and builds no channel structures", or add one numeric check against a hand-computed reference.

---

### ② G1 `2628-jitter-l1` — pre-registered rule faithfully executed

**Correct.**
- **Pre-registration honored.** `README §4.1` fixes `primary = combined_range = max(l1_jitter, repr01_jitter)`, thresholds `0.0144` (downgrade) / `0.005` (maintain). Executed arithmetic: `l1_jitter=|1.4163−1.4089|=0.0074`, `repr01_jitter=|1.3910−1.3945|=0.0035`, `combined_range=max=0.0074`, verdict `0.005 ≤ 0.0074 < 0.0144` ⇒ **`on_threshold`** (`results_13m.json:3439-3462`; `jitter_verdict` in `jitter_l1.py`). The secondary "取大 vs 加和" concern is benign: additive range `0.0109` also lands in `on_threshold` — no cross-threshold divergence, and the report says so. `reconcile_numbers.py` recomputes `combined_range` and `verdict.state` from the curve and matches. The verdict correctly **retracts** 2627's "稳健".
- Config single-factor: `2628/configs13m/l1.json` is visually byte-identical to `2626/configs13m/l1.json` (read both); `tests/test_jitter_l1.py::test_l1_config_byte_identical_to_v2` pins it; mutation M1 catches a changed `steps`.
- Numbers match artifacts (≥10): four `@K8` readings, `l1_jitter 0.0074`, `repr01_jitter 0.0035`, `combined_range 0.0074`, `worst_case 0.0109`, four net combos `0.0144/0.0179/0.0218/0.0253`, paired CIs `+0.0074 [0.0061,0.0088]` / `−0.0034 [−0.0045,−0.0024]`, battery `+0.0061/−0.0037`, `wall 16.8 min`, `peak 2616.2 MB`, `gate_skips 0`, deterministic gain `0.0` with 8/8 paths identical, cross-run 6/6, `train_tokens 393,216,000`, code sha256s.
- Mutation harness (M1–M4 file write/read/restore with sha256 check + re-green; M5 in-process monkeypatch, no file writes) is correct and `all_caught_and_restored=true`.

**Finding: P1 — `analysis.md` carries two numbers that contradict `results_13m.json`.** (a) `analysis.md:6` and `:250` claim the reconcile is **`67/67`**, but `reconcile_report_13m.json` has `n_checks=82, n_fail=0` (and `reconcile_numbers.py` produces those 82 checks; `progress.md:16` also says **82/82**). (b) `analysis.md:220` claims eval wall **`1.38 min`**, but `results_13m.json::cost.wall_minutes_total_eval = 1.44`. Both are stale values left behind after the module-rename re-eval (`progress.md` records "重跑评测刷新 `results_13m.json::code` 指纹") — the exact "脚本化改动未逐条核对" hazard the repo warns about. No scientific conclusion is affected (the drift is confined to the self-reported check count and one cost figure), which is why I stop at P1 rather than P0; but the repo rule "数字必须与 results.json/日志逐一对应" makes this a must-fix. **Smallest fix:** edit `analysis.md` `67/67`→`82/82` (two places) and `1.38 min`→`1.44 min`.

---

### ③ L1 `2628-longT-2048` — convergence-degree inequality & memory-line accounting

**Correct.**
- **Convergence-degree inequality explicitly declared.** Token budget is matched (49.152M/arm) so steps differ (24000/12000/6000/**3000**); `README §4⑤` and `analysis.md §2.3` state the unequal convergence, give the step table, the last-window loss slopes (full −0.5158 / win −0.4923 nats/1k, "远未收敛"), a same-step control (`full@T=1024` = 1.4991 @3000 vs `full@T=2048` = 1.5423), and the explicit restriction "跨点比较不得当同收敛读（禁外推）". §12 repeats it. This satisfies §16/§17.15/§20.3.
- **Window-arm memory-line deviation accounted per §20.2 three-piece.** `analysis §4`: ① per-arm peaks `1.906 / 5.910 GiB` (torch) and `2.53 / 6.60 GB` (nvidia-smi) — matches `results.json::arms` and `criterion_4_memory_wallclock`; ② pre/post free `19439 / 18915 MB` (matches `results.json` + `smoke/smoke.json`), margin ≫8 GB; ③ explicit "超礼貌线 18%" statement. Not reducing batch is justified by §20.2 rule 1 (keep recipe consistency across 4 points); rule 4 not triggered (margin >8 GB).
- Numbers match artifacts (≥10): `Δ2048 native = 0.004361` (=2.339413−2.335052) → band "不亏"; four points −0.015752/−0.015101/+0.009264/+0.004361; gate `gnorm_max 7.22/7.26`, skips 0; `attn islands 0.1696/0.2914` vs formula 0.5714; `visible_frac 0.749878`; `steps_for(2048)=3000`; `tokens 49,152,000`; selfcheck `0.0 / 20.82474`; `reconcile_report.json n_checks=77 n_fail=0` (report cites no count ⇒ no mismatch); mutation 4/4 caught, `sha256 before==after`.
- §20.1 respected: `run.py::stage_train_all` releases the lock per arm with a 60 s gap; each arm 2.67 min (<10 min); per-arm evidence copies (`-train_log/gate_log/probe.jsonl`, `-train_summary.json`) present despite `runs/`+`*.pt` being `.gitignore`d.

**No findings.**

---

### ④ Cross-line number↔artifact correspondence / ⑤ deliverables in git / ⑥ §20 accounting

**Correct.**
- Every deliverable path claimed by each README/analysis exists on disk; `watchdog_diff` reports **no untracked and no modified** files ⇒ all non-ignored deliverables are committed. The only absent files are explicitly-ignored weights (`smoke/master/model.pt`, `train13m/l1/model.pt`, `runs/**/model.pt`) and `runs/**`, each documented as not-in-repo with hashes in `MODEL.md`.
- §20: E1 runs one short locked smoke (0.05 min, 861 MB < 5 GB ⇒ §20.2 N/A); L1 releases per arm with ≥60 s (§20.1 ✓), 5.91 GiB deviation accounted (§20.2 ✓), no external slowdown so no §20.3 step shrink; G1 single-arm (see note).

**Finding: P2 — G1 README §12 claims eval is ">2min" but the artifact says 1.44 min.** `README.md:§12` ("评测为纯推理 >2min ⇒ 取锁") vs `results_13m.json::cost.wall_minutes_total_eval=1.44`. Harmless (locking was conservative), but it is another number not matching the artifact; fix while touching `analysis.md`.

**Finding: P2 (record-only per §20.1 item 4) — G1 holds the GPU lock for the whole single-arm run.** `train_all.sh` take/release lock around one 16.8-min training; §20.1 rule 1's "整程 <10 min" cannot be met for a single unsplittable arm (there are no inter-arm boundaries), and the README §12 documents the reasoning. It does not affect any conclusion.

---

## Merge verdict

**OK with notes.** The three lines are scientifically coherent, the pre-registered E1 write-timing / G1 jitter / L1 band criteria are faithfully executed, mutations are genuinely red-then-byte-restored, and — apart from the G1 stale figures — every reported number traces to `results.json`/logs. One **P1** (G1 `analysis.md` 67↔82 and 1.38↔1.44) should be corrected before release; the P2s are report-only.

**Supervisor must run (I have no shell):**
```
python3 -m pytest tests/ -q            # expect 549 passed (task baseline)
python3 tools/build_wiki.py --check
python3 tools/check_schema.py
```