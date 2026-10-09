# axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/analysis.md · 过程史（自事实文档迁出）

> 性质：日志文档。此处是自事实文档 `axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/analysis.md` 迁出的**过程叙述**（日期锚定说明 / 更正 / ERRATA /
> 变更叠加 / 已废台账指代）。事实以该事实文档的当前版本为准；本文件只作可回溯记录。

## ERRATA（测量实现错误，已修）+ 已作废读数

- §1 对账表注释：「（见 §4 **ERRATA**：首版该对照因毁错区间而虚高，≈0.10）」→ 去该注释。
- §1 结论：「三道对账全部通过，无需撤回首论；**唯一 ERRATA 见 §4**（是测量实现错误…）。」→ 去 ERRATA 子句。
- §4 标题：「## 4. 混杂与口径声明（**含 ERRATA**）」→「## 4. 混杂与口径声明」。
- §4 子节「### ERRATA（测量实现错误，已修）」整节（首版毁源对照把源区间硬编码为 `[0,32)`、
  `src_pos=96/191` 的源未被碰到 ⇒ 读数虚高；`diagnose_copy_confound.py` 定位；`run.py` 改为按各档
  源区间毁源；**首版那组虚高读数已作废**）→ 从事实文档删除。
- 现状：当前口径下的毁源读数（0.0104–0.0173 ≈ chance）保留在 §1 对账表与 results.json。
