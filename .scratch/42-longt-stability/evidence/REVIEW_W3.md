# 独立复核 W3（ticket 42）：重派复核 Y2/Y3（Y2/Y3 已于 2059511 补交）

I have enough evidence. Here is my review.

## Review

**Scope / method.** Read-only review of the two committed experiments against `SCHEMA §14/§14b, §17.11, §17.14, §17.16, §17.17, §17.18, §18–§23` and the ticket-42 W3 charter. `watchdog_diff` at reviewer-launch `HEAD fe000c40` shows **no tracked working-tree changes**; the only untracked paths are `.scratch/41-…`, `.scratch/42-…` and **`tests/test_a9_window_longrange.py`**. Both reviewed directories are committed (per-directory `watchdog_diff` empty). I could not run `pytest` (no shell) — see commands to run at the end.

### Correct (verified)

- **Y2 "不可归因" is faithfully executed.** `results.json::criteria.criterion_1_quality` keeps the raw band literally (`band_class_T1024 = "新算子更好（越界）"`, `sign_T1024 = "+"`, `T=1024` entry in `three_points_native`), and adds the degradation as a separate `interpretation`/`confound_note` (`"不可判（基准臂不稳定）"`, `three_point_shape = "更差 → 不亏 → 不可判"`). The criterion was not edited; the axis README carries the same. Evidence: `2623-longT-1024/results.json:290-305`, `analysis.md:60-78`, `reconcile_report.json:107-108`.
- **Y2 口径差异已标.** bs8 (Y2) vs bs32 (X3) is stated in README §6, `analysis.md §1/§2/§10`, `MODEL.md §口径`, `controls/README.md`, and `run.py:51-57`.
- **Y2 §20.1/§20.2 记账.** Per-arm release with ≥60 s window is evidenced in `train_all_log.txt:10` (`释放锁，留 60s 窗口（§20.1）`); OOM→bs8 decision and free-memory shortfall candidly recorded (`analysis.md §4`; `results.json criterion_3_memory` free 4624/4099 MB, peak 1.915 GiB < 5 GB).
- **Y3 main leg is in training distribution (§17.18).** Main leg = `assoc_recall_multi_batch` with K=8/Q=8/gap=14, i.e. the **same generator/layout** as training (`train_depth_ar.py:191`, `--synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14`). The position-extrapolation variants (`assoc_recall_batch`, `var_recall`) are demoted to **secondary** and retained (`results.json::secondary_single_query`, `::distance_leg`; `analysis.md §9`). Destructive-source control is valid: `full` 0.366455 → 0.014465 (≈chance). §17.17 correctly declared not-triggered; §17.15 answer-row CE slope reported (`answer_ce_slope`, `analysis.md §7`).
- **Numbers ↔ `results.json`/logs correspond for both** (spot-check ≥14, including costs, counts, equations, hashes):
  - Y2: Δnative 0.023419, release −0.398242, bpc 1.510446/1.487027, gnorm 1740.92/7.35, params 7159296, tokens 98304000, `visible_frac` 0.749756, ckpt sha `e2eeb2…`/`b25a50…`, code fp `3a0db55…`. `reconcile_report.json` 71 checks / 0 fail.
  - Y3: full 0.366455, win16 0.014404, win16_rep 0.014587, Δ 35.205 pp / z 81.153, params 13536768, wall 13.01/13.75/14.18, clean bpc 0.472276/1.896909, `visible_frac` 0.120866, ckpt sha `d80f4c…`/`525bd3…`/`c5988d…`. `reconcile_report.json` 87 checks / 0 fail.
- **Mutation verification is real** for both (`mutation_report.json`: all caught red, `restored_bitwise = true`); Y3 n=1 negative reproduced by `win16_rep`.

### Findings

- **Finding 1 — P0 — Y3 committed revision is missing a required deliverable (test file untracked).**
  `tests/test_a9_window_longrange.py` is **untracked at `HEAD fe000c40`** (present in `watchdog_diff` untracked inventory), yet Y3 names it as a deliverable: `README.md §7/§11` (“单测：`tests/test_a9_window_longrange.py`”), `MODEL.md §4`, and its committed `mutation_report.json` validates 4 mutations against test IDs in that exact file. Consequence: a clean checkout of the committed revision cannot reproduce Y3's failing-test/mutation validation, and `pytest tests/` runs 10 fewer tests than the wave's stated baseline. This is the **same defect class** the re-review is chartered to confirm fixed (prior P0 = “未提交”; ticket 41 recorded Y1/Y2/Y3 as “提交 `a142283`” yet Z4 found them uncommitted). **Smallest fix:** `git add tests/test_a9_window_longrange.py` (explicit path) + commit; then re-run `pytest tests/ -q`. (Y2's `tests/test_longT_1024.py` *is* tracked — the asymmetry is Y3-only.)
- **Finding 2 — P2 — Y3 doc/result numeric drift on eval wall-clock.** `README.md:164` explicitly cites `results.json::cost.wall_minutes_total_eval` as **1.03 min**, but that field is **1.04** (`results.json:4551`); `analysis.md:181` also says 1.03. Smallest fix: correct both to 1.04.
- **Finding 3 — P2 — Y2 raw per-step evidence not in the committed revision.** Y2's README §7/§10 list `<arm>-train_log.jsonl` / `<arm>-gate_log.jsonl` as deliverables, but they live under `runs/` (git-ignored; `.gitignore`), so the committed tree carries only `train_all_log.txt` (1 000-step samples) + `*-train_summary.json`. The `analysis.md §6` figure “最大采样 gnorm（128.85）” appears only in `runs/full-1024/train_log.jsonl` (git-ignored) and is **not** in `results.json` — unverifiable from committed artifacts. (Y3 kept `train/<arm>/train_log.jsonl` tracked, so this is a Y2 traceability asymmetry.) Also `run.py:7` module docstring is stale (“bs32 … 3000 步”) vs actual bs8 × 12000.
- **Finding 4 — P2 — Y3 minor doc blemishes.** `analysis.md §3` arithmetic reads “= 24×32×256×… = 12000×32×256 = 98,304,000” (garbled lead-in). `README.md §3` states “`source_inside_window == (d<window)` 逐档一致”, but the full arm (w=0) records `True`; `reconcile_numbers.py` implements `(d<w) if w>0 else True`, and the report itself prints `True vs True (w=0)` — the shorthand contradicts its own check.
- **Finding 5 — P2 — Y2 does not cite Z1's seed=1 evidence (consistent, but uncited).** Y2's committed artifacts still list “seed=1 双臂同配方重跑” as an untested next step; Z1's seed=1 result (`29578.57`) is cited only at the axis/ledger level, not inside `2623`. Numerically consistent (Y2 seed0 gnorm 1740.92 and Δ 0.023419 match Z1's seed0 references), but the run-level artifact remains “不可判” rather than “永久不可归因”.

### Merge verdict

**BLOCK** — on Finding 1 only (P0: Y3's required semantic-layer test, and thus its mutation validation, is not in the committed revision). All substantive results, accounts, and §17-relevant judgments for Y2 and Y3 are otherwise **OK**; Findings 2–5 are report-only P2. After `tests/test_a9_window_longrange.py` is committed, this becomes **OK with notes**.

**Commands the supervisor must run (I have no shell):**
- `python3 -m pytest tests/ -q` — confirm green and count; reconcile against the stated baseline **468** (expect 10 of those to be `tests/test_a9_window_longrange.py`, which is currently untracked).
- `git log --oneline -- tests/test_a9_window_longrange.py` — confirm it was never committed.
- `python3 axes/A-mixing/A6-mixing-operator/experiments/2623-longT-1024/reconcile_numbers.py` and `python3 axes/A-mixing/A9-old-info-retention/experiments/2623-window-on-longrange-13m/reconcile_numbers.py` — expect rc 0 (71/87 checks).

**Residual risks:** Y2's conclusion (`Δ1024` unusable) is n=1-seed0 plus Z1's seed1 at the axis level; the run-level artifact still reads “不可判”. Y2's raw logs are uncommitted, so its stability narrative rests on committed summaries + embedded strings rather than the raw series.