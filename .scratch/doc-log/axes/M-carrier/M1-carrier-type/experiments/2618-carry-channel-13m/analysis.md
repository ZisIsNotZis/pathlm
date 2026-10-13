# axes/M-carrier/M1-carrier-type/experiments/2618-carry-channel-13m/analysis.md · 过程史（自事实文档迁出）

> 性质：日志文档。此处是自事实文档 `axes/M-carrier/M1-carrier-type/experiments/2618-carry-channel-13m/analysis.md` 迁出的**过程叙述**（日期锚定说明 / 更正 / ERRATA /
> 变更叠加 / 已废台账指代）。事实以该事实文档的当前版本为准；本文件只作可回溯记录。

## 主 Agent 修正（复核 P2 / P2-1，2026-10-01）

- 原文末两条 blockquote：
  「> 主 Agent 修正（复核 P2-1）：评测墙钟以 `results.json::cost.wall_minutes_total_eval = 2.01` 与
  `run_log.txt` 为准（原 2.57 为陈旧值）。」
  「> 主 Agent 修正汇总（复核 P2，2026-10-01）：评测墙钟 2.01 min、步时改为可推导口径、更正一处无留存
  支持的 step 号、更正前向批分解、并如实标注变异验证与代码指纹的修订关系。**以上均不改变判读。**」
- 现状：两条更正注删除；当前值 2.01 min 已在本文件 §（成本）与开头复现行给出。

## 「已更正」一处 step 号

- 原文（§ 结论与下一步 第 4 点）：「更早一次试跑也已到 3.5015（留存数据里该值出现在 U1rep 的
  **step 5000**；原写 step 3600 无留存日志支持，已更正）。」
- 现状：事实文档改为「更早一次试跑也已到 3.5015（留存数据里该值出现在 U1rep 的 **step 5000**）。」；
  过程（原写 step 3600 无留存日志支持、已更正）迁出，当前值 step 5000 / 3.5015 保留在事实文档。
