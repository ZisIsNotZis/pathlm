# 2.2.1-residual-form · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/2-depth/2.2.1-residual-form/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/2-depth/2.2.1-residual-form/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-14T23:10:00+08:00

- actor：轴 2 深度轴 lead（ticket 61 · 复核 B 处置 writer）
- change：按复核 B 的第 1 轮独立判词（PASS）回填 `conclusion_review`：判词指针指向 `TICKET.md §9.18`；本轮未改任何测量数字与判据线。

## 2026-10-13T10:37:02+08:00

- actor：轴 2 §2.2 worker
- change：新建研究项（无旧格，MAPPING §2）：残差的形式 归 2.2 层间怎么聚合（残差族）；按 docs/research-schema.md 预登记目的/基线/判据/否证条件；与 8-engineering 的归一化位置工程句分工已在 gap 写明。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T22:46:16+08:00

- actor：WP-2652 worker（ticket 61 · 轴 2 wave 1 收口）
- change：WP1 完成并收口：四臂 12k 全训完并评测（base 1.5310 / postln 1.5284 / gated 1.5326 / scaled 1.5413），`python3 run.py --aggregate` 把 scaled 折进 results.json（未改任何测量值）；落 analysis.md（人话摘要 + 四臂表 + 逐深度表 + 参数/活性/稳定性列 + §12 三对账 + §13 三要素 + 口径标注 + 两面读法）与 MODEL.md（四 checkpoint sha256/体积/生成命令）。status 未开始 ⇒ 已完成（终质维不可排序、可截断性维可分，明确结论；非「手段不成立」故不判被否证、failure_analysis 保持空）。gaps 补齐单 seed / 无独立等参数对照 / post-LN 读出尺度混杂 / 与 2.1.1 联合未测 / 衰减形式未单列。独立复核未做，待 fresh 复核者（SCHEMA.md §14/§14b）。**方向纠正**：派单与 ticket §9.8 曾记「postln 中间深度 bpc 更低 / 更可读」，与落盘证据相反（postln 中间深度 bpc 大幅更高 = 更不可读；『更可读』的正面读数落在 scaled），analysis/project.yaml 按实测方向写。

## 2026-10-14T22:46:16+08:00

- actor：WP-2652 worker（ticket 61 · 轴 2 wave 1 收口）
- change：另：post-LN 全程 gnorm_max 19.81（base 7.86）如实报——未发散、无 grad skip、末窗 gnorm 均值反而更低（0.3925 vs 0.4263），判早期尖峰、不改判据不改判词。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
