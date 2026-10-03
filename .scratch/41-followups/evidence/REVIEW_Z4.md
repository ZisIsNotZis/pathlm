# 独立复核 Z4（ticket 41）：Y1/Y2/Y3

## Review

Reviewed against committed revision `a142283` (watchdog_diff HEAD = `a142283e8ae3`), read-only. Target = ticket-40 Y1/Y2/Y3.

### Correct (verified good)

- **Y2 verdict is faithfully executed (focus ②).** `results.json::criteria.criterion_1_quality` keeps the literal band for the third point (`band_class` T=1024 = `"新算子更好（越界）"`, line 261) while the *interpretation* is downgraded — `run.py:687-700` appends `不可判（基准臂不稳定）` only to the display label, `interpretation:"不可判"`; the "不可归因" note and n=1/禁外推 are present (analysis §2/§6). The bs8-vs-X3-bs32 口径差异 is flagged in analysis §1 (⚠) and §10. ✔
- **Y3 measurement-fix keeps original readings as secondary (focus ③).** `results.json::secondary_single_query` retains `full=0.0 / win16=0.012207 / win16_rep=0.013672` with the explicit note "查询键在末位（位置外推）；full 的末位先验=预测 filler", plus `distance_leg`; analysis §9 documents the ERRATA. Main leg = multi-query, `full=0.366455` vs `win16=0.014404`, Δ=35.205pp, corrupt full→0.014465. ✔
- **Lock/memory accounting (focus ④).** Y1 `train_arms.runlog` shows per-arm 取锁/释放 + 65s windows with lock-wait logged; Y2 `train_all_log.txt` shows "释放锁，留 60s 窗口（§20.1）" between arms; Y3 `train_all.sh` implements per-arm release + `sleep 60`. Y2 §20.2 three-piece accounting present (per-arm peak 0.914/1.915 GiB, free 4624→4099 MB, explicit "≥8GB 不可满足" deviation). All peaks ≤5 GB net. ✔
- **Numbers match results.json (focus ⑤) — spot-checked 30+.** Y1: main 0.0/0.001465/0.156738/0.153320, cross 0.136719, in-dist no-last-block 0.257254/0.266741/0.271205/0.254464, staircase +0.949/+0.446/−1.674 net −0.279, bpc 13.069118/37.145279/58.307878/54.970002, slopes −0.09264/−0.07398/−0.06019/−0.06076, params −8.748%. Y2: 1.510446/1.487027/Δ0.023419, release −0.398242, gnorm 1740.92/7.35, peaks 0.914/1.915 + nvidia 1.48/2.53, attn island 8.148/28.798=0.2829 vs formula 0.4, visible_frac 0.749756, free 4624/4099, wall 9.02/10.02. Y3: 0.366455/0.014404/0.014587, corrupt 0.014465, Δ35.205, answer_rows 1.896909/5.999322, wall 13.01/13.75/14.18, peaks 1630.4/2015.3. Reconcile reports: Y1 142/0, Y2 71/0, Y3 87/0. Mutation: Y1 `all_caught_red=all_restored_bitwise=true`. ✔
- **Focus ① partly satisfied.** Y1 *does* flag the near-OOD main leg and reframe carry's gain as distance extrapolation: analysis §7B ("判决腿的距离 241–255 全部超出了训练分布的最大距离 240 ⇒ …不外推"), §8.1 ("判决腿是近分布外…不外推"), human-summary leads with the in-distribution null; "哪条腿为主" has an explicit reason (§4: 判决腿 = `assoc_recall_batch`, "与 2618/W3 同一件仪器"). ✔

### Finding: P0 — Y2 and Y3 are not in the committed revision

- **Location:** `axes/A-mixing/A6-mixing-operator/experiments/2623-longT-1024/` and `axes/A-mixing/A9-old-info-retention/experiments/2623-window-on-longrange-13m/`.
- **Evidence:** `watchdog_diff` reports **every** file in both dirs as *untracked*, and the scoped file query confirms it (e.g. `.../2623-longT-1024/analysis.md`, `.../2623-window-on-longrange-13m/analysis.md` → "Untracked files"); `.git/index` contains no `longT-1024` / `window-on-longrange` entry, while `2623-composable-arms-13m` (Y1) **is** tracked (grep hit; per-file diff = no working-tree change). Meanwhile the **committed** `STATUS.md` L15-16 marks Y2/Y3 ✅ and `COMMIT_EDITMSG` (=a142283) claims all three lines — i.e. the commit message/status over-claimed relative to actual content (the recurring "提交信息超出实际改动" pattern).
- **Impact:** Confirms the review premise is false for 2 of 3 lines: they cannot be reviewed "基于已提交修订", and their ✅ state-lights have no committed artifact. Per "fail 的结论不计入状态灯", Y2/Y3 conclusions must not count until committed.
- **Smallest fix:** commit the two dirs with **explicit paths** (not `-A`/`.`, per AGENTS.md), then re-run Z4 for Y2/Y3. Verify with `git ls-files <the two dirs>` (expected empty now).

### Finding: P2 — Y1's primary leg is OOD in *two* ways; only distance is flagged

- **Location:** `.../2623-composable-arms-13m/analysis.md` §4, §7B, §8.1.
- **Evidence:** The main ("判决") leg is `assoc_recall_batch` — single-query, query key at the **last position T-1** (`tools/evalkit/longrange_tasks.py:131-132,160`), while training uses `assoc_recall_multi_batch`, whose final positions (242-255) are **always filler** (`longrange_tasks.py:548-563`). Y1 flags only the *distance* OOD (241–255 > 240) and attributes the non-carry arms' 0% to "距离越界崩掉". But the sibling line Y3 (same batch) independently established that this exact single-query-末位 layout is a **positional-extrapolation artifact** (`2623-window-on-longrange-13m/analysis.md` §9; `results.json::secondary_single_query` full=0.0), which alone yields 0% for multi-query-trained arms irrespective of distance. Y1 does not mention the layout mismatch or this confound.
- **Impact:** The headline (carry gain is an extrapolation effect) is already hedged, so verdicts don't flip; but the §7B causal claim ("无记忆臂是距离越界崩掉") is under-supported, and Y1's promotion of the single-query leg to 主判据 is inconsistent with Y3's demotion of the same leg. 
- **Smallest fix:** add one sentence to §7B/§8.1 noting the single-query-末位 layout confound (align with Y3 §9), and state why the OOD leg is retained as primary.

### Finding: P2 — Y2 raw log carries a stale `band_class` label

- **Location:** `.../2623-longT-1024/train_all_log.txt` (`"band_class_T1024":"亏"`, and T=256=`亏`/T=512=`不亏`).
- **Evidence:** Current `run.py:640-647` returns `新算子更好（越界）` for positive Δ, matching `results.json` T=1024; the raw log shows `亏` (a non-directional legacy label, acknowledged in the `run.py` docstring "「亏」当时代指「窗口更差」；本点 Δ 为正，必须区分方向"). It is *not* a判据 change (the rule |Δ|<0.014 is unchanged and results.json matches current code), but the raw log and final artifact disagree on the displayed label.
- **Smallest fix:** regenerate the log (or annotate it as pre-fix) so the raw evidence matches the authoritative band label.

### Merge verdict: **BLOCK** (Y2/Y3), **OK with notes** (Y1)

Y1 is committed and passes substantively (focus ①–⑤ met; two P2 notes). Y2 and Y3 are **not in the committed revision**, so they cannot be accepted as reviewed at `a142283` and their state lights must not stand until committed. pytest (baseline 431) not run — no shell available; left to the parent.

Residual risk: if `watchdog_diff`/`.git/index` somehow diverge from the parent's true repo state, the P0 is a false positive; the one-line `git ls-files` check in the message resolves it. Y1's contribution numbers are read from committed `results.json` only.