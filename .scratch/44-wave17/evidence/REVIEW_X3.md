# 独立复核 X3（ticket 44）：V1（长 T 干净三点）/ V2（G5 机制收敛）

## Review

**Scope note / limitations.** I reviewed the on-disk artifacts for **V1 (`2626-longT-clean`)** and **V2 (`2626-mechanism-convergence`)** against their contracts, SCHEMA §14/§17.x/§20.x, and their own `results*.json`/logs. `watchdog_diff` shows **no working-tree delta vs reviewer-launch HEAD** and only two untracked `.scratch` ticket files, so the reviewed artifacts are committed; I **could not** inspect committed ranges, therefore I cannot prove the exact commit ids `c3bbead/87f1fc1/c72f381` (no commit-range artifact was supplied).

### Correct (verified with evidence)

- **① V1 stability precondition really holds.** `results.json` `criterion_2_stability_gate` = `full_1024_gnorm_max 7.25 < 100`, `full_1024_grad_skips 0`, `passed true` (results.json:749-756); per-arm `gnorm_monitor` agrees (7.25 / 0 at :563-565). The three-point caliber is stated honestly: half-budget snapshot `49,152,000 tok/arm` (README §6; `results.json` arms = 24000/12000/6000 × bs8 × T), old points on a separate list with the four caliber differences (analysis §7), and §17.11 no-extrapolation declarations (analysis §12). Point arithmetic checks: Δ = 1.631129−1.646881 = −0.015752; 1.639843−1.654944 = −0.015101; 1.712069−1.702805 = +0.009264 all match analysis §2.
- **② V2's convergence claim is arithmetically faithful.** `results_13m.json::convergence_gate.per_checkpoint["18000"].main_slope_per_1k = −0.001`, `mechanism_slope_per_1k = −0.0001`; X1's −0.060/1k reproduces from X1's own file (0.1357−0.3164)/3 (2622 `results_13m.json`:6555,7389). The schedule caveat is also factually correct: `train_m1.py:124` normalizes cosine by `steps`, so lr@6k = 0.868×base for a 24k run (analysis/README §2), and the @6k-vs-X1 non-comparability is stated in both analysis §5 and `results_13m.json::x1_6k_cross_check.schedule_note`. Core numbers reconcile: main +0.0346 (CI [0.0298,0.0407]), mechanism +0.0144 (CI [−0.0158,−0.0130]) from `criteria.*.ci`; four-point trajectory −0.0063/+0.0147/+0.0150/+0.0144 from `trajectory`.
- **③ ≥10 number↔artifact correspondences hold per line** (cost, step counts, equations, hashes): V1 tokens 49,152,000, wall/step times, peak 2.016/2.65 GiB, six `sha256` in MODEL.md ↔ `config.json::checkpoints`; V2 params 13,076,483, wall 22.0/22.3/28.8 min, eval 4.09 min, peaks 2613.9/2616.2/4995.8 MB, `repr_decorr` 0.79354→0.007126, code hashes in `results_13m.json::code` ↔ MODEL.md. Reconcile scripts pass (`reconcile_report.json` 170/170 for V1; `reconcile_report_13m.json` `n_fail: 0`; mutation reports 4/4 and M1–M6 caught-red + byte-identical restore).
- **④ Deliverables present/committed.** Every path claimed in V1 README §10 and V2 README §9 exists on disk, and **none appears in the untracked inventory** ⇒ all tracked in HEAD (weights `*.pt` correctly absent, `runs/` gitignored).
- **⑤ §20 accounting mostly correct.** §20.1 per-arm release with ≥60 s window is real (V1 `train_all_log.txt` "释放锁，留 60s 窗口（§20.1）" per arm; V2 `train_all.sh:44-71` take/release per arm + `sleep 65`). §20.3 step-time-first budgets are evidenced (V1 `smoke/step_time_probe.json` 33.0/33.8/48.1 ms → half-budget; V2 Δ 250-step smoke → 90 min).

### Findings

- **Finding: P1 — V2 analysis cites a reconcile count that does not match its own artifact.**
  Location: `axes/G-policy/.../2626-mechanism-convergence/analysis.md:6` and `:277` both say "**69/69**";
  Evidence: the cited file `reconcile_report_13m.json:3-4` is `"n_checks": 85, "n_fail": 0` (i.e. 85/85), and the committed `progress.md:16` also says "**85/85 pass**".
  Impact: a report number that fails "数字与 results.json/日志逐一对应" (§12/§14), even though the pass/fail verdict (0 fails) is unchanged.
  Smallest fix: replace `69/69` → `85/85` in the two `analysis.md` lines (text-only).

- **Finding: P1 — V1 §5 asserts a "唯一稳健的读数" that its own two-round data contradicts.**
  Location: `axes/A-mixing/.../2626-longT-clean/analysis.md` §5 (the lines right after the wall-clock table).
  Evidence: the same section records "两轮方向不一致 … 如实记「测不出方向」" and then "唯一稳健的读数：**窗口臂墙钟 ≥ 全因果臂**（`r_time` 1.07–1.19 > 1）". But the round-1 column of that very table gives win/full = 25.736/30.935 = **0.83** (T=256) and 23.165/23.639 = **0.98** (T=512) — i.e. the window is *faster*, not slower. The "1.07–1.19" range is round-2 only (the column is labeled "（轮2）"). The phrase "T=512/1024 方向各区不同" also mis-groups T=1024, whose direction is consistent (window slower in both rounds).
  Impact: a false robustness claim in a judge-criterion (④) section that contradicts the immediately preceding sentence; the primary Δbpc conclusion is unaffected, but the wall-clock statement should not stand as written.
  Smallest fix: drop the "唯一稳健的读数" sentence or relabel it "轮2 观测：窗口更慢（T=256/512 轮1 反而更快，故仍为测不出方向）".

- **Finding: P2 — V2's "60×" slope-reduction mixes different training phases.**
  Location: `analysis.md` §0/§8 ("末窗 −0.001/1k … X1 的 6k 窗口是 −0.060/1k，速率降了约 60×").
  Evidence: V2's own like-for-like @6k→end slope is `−0.0042/1k` (`results_13m.json::convergence_gate.per_checkpoint["6000"].main_slope_per_1k`), i.e. ~14× vs X1's early-phase −0.060/1k, not 60×. The −0.001/1k figure is the late (18k→24k) segment.
  Impact: magnitude overstated (conclusion "基本收敛" still holds from its own decelerating four points). Smallest fix: also cite the @6k→end slope, or label the 60× as cross-phase.

- **Finding: P2 — V2 §20.2 "跑前后各记 nvidia-smi memory.free" is not in any committed artifact.**
  Location: V2 README §6 requires it (§20.2 三件套); `train_all.sh:59-66` emits `pre-free`/`post-free`, but grepping the whole V2 dir finds **no** free-memory reading in `results_13m.json` / `run_log_13m.jsonl` (the driver log is `*.log`, gitignored).
  Impact: §20.2's three-piece accounting was not *triggered* (repr01 peak 4995.8 MB < 5 GB, i.e. 0.08 % under the decimal line), so no over-line deviation exists; only the contract's own record-keeping ask is unmet in-repo. Smallest fix: note the free readings (or state they are only in the driver log).

- **Finding: P2 — V1's "余量 >8GB" cites the eval round only.**
  Location: `analysis.md` §5 "跑前/跑后 = 23356 → 22827（余量 >8GB）".
  Evidence: that pair comes from the eval stage (`smoke/smoke.json`, `results.json:956-957`); per-arm training readings include `win128-256-clean` `free_after_mb = 5791` (`win128-256-clean-train_summary.json:58-59`), i.e. ~5.7 GB free during that arm. Not a §20.2 trigger (no over-line), but the blanket "余量 >8GB" framing does not hold for every arm. Smallest fix: scope the sentence to the eval round.

### Merge verdict
**BLOCK** — two P1 documentation-accuracy defects (V2 count mismatch; V1 false "robust reading"), both text-only fixes. All scientific claims, numbers, gates, calibers, §20.1 release discipline, and deliverable-in-git checks otherwise verify; remaining items are P2 notes. I ran no commands (read-only); the parent must run `python3 -m pytest tests/ -q` (baseline **504**) since I cannot.