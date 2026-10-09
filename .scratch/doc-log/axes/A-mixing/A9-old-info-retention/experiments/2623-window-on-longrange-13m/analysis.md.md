# 过程史 · 自 `axes/A-mixing/A9-old-info-retention/experiments/2623-window-on-longrange-13m/analysis.md` 迁出

> 性质：日志文档。原文为「文档变更记录」与「ERRATA」标题，SSOT 化时从事实文档删除，只留当前结论。

- §9 原有标题（去 ERRATA）：`## 9. ERRATA / 执行期发现的测量问题（**必须随结论一起说**）`

- 文末原有整节 P2 变更记录（删）：

```
## 13. P2 修正记录（后于复核）

> **来源**：ticket 43 V3「P2 收尾」，修正对象 = `.scratch/42-longt-stability/evidence/REVIEW_W3.md`
> 的 report-only P2（Finding 2 / Finding 4）。**只改文档表述；不改任何数字结论、band 分类、判据表述。**
> 本文件（`analysis.md`）与同目录 `README.md` 的改动逐条留痕如下。

| # | 文件:位置 | 改了什么 | 改成什么 | 为什么不影响结论 |
|---|---|---|---|---|
| 1 | `README.md` §10（评测实测） | 全电池 `1.03 min` | 全电池 `1.04 min` | 笔误：对齐权威产物 `results.json::cost.wall_minutes_total_eval = 1.04`（results.json:4551）。仅评测墙钟成本，非判据量 |
| 2 | `analysis.md` §11（实测墙钟） | 评测全电池 `1.03` min | 评测全电池 `1.04` min | 同 #1（同一数字的两处引用）；不涉任何判据/结论 |
| 3 | `analysis.md` §3 项 1（算术对账） | `= 步数×bs×T = 24×32×256×... = 12000×32×256 = 98,304,000` | `= 步数×bs×T = 12000×32×256 = 98,304,000` | 只删除行内残片 `24×32×256×... =`；闭式值 `98,304,000` 未变（12000×32×256），`reconcile_numbers.py` 该 check 仍 0 失败 |
| 4 | `analysis.md` §3 项 2（任务对账） | `source_inside_window == (d<window)` 逐档一致 | `source_inside_window == (d<window)`（`w=0` 的 `full` 臂恒 `True`）逐档一致 | 补注 `w=0` 特例，与实现 `(d<w) if w>0 else True`（`reconcile_numbers.py:90`）及报告原文 `True vs True (w=0)` 自洽；原简写与其自身 check 矛盾，语义与判定不变 |

**未改动 / 复核结论**：`results.json`、`reconcile_report.json`、`mutation_report.json` 与全部数字结论、band 分类、判据线 **一字未动**；REVIEW_W3 的其余 finding（P0=Finding 1 已由 `a8f440d` 关闭；Finding 3/5 属 Y2，见其目录的 P2 修正记录）。
```
