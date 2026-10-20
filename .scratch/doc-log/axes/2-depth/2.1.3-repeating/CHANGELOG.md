# 2.1.3-repeating · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/2-depth/2.1.3-repeating/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/2-depth/2.1.3-repeating/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:37:02+08:00

- actor：轴 2 §2.1 worker
- change：按 MAPPING §4 / docs/structure.md §9 冻结 2.1.3 repeating 为 Backlog 研究项（status: 未开始）；本仓零经验、无旧格证据，仅登记目的/判据/出线条件，不投入执行预算。

## 2026-10-14T06:44:03+08:00

- actor：framework-adjudication-writer（ticket 57）
- change：落地 61-F4 裁决（不改数字）：确认与 2.2.5 权重共享 **不合并**，两格均维持 Backlog、gaps 互指，删去「待 Main 裁」的悬置口径。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T21:00:00+08:00

- actor：轴 2 深度轴 lead（ticket 61）
- change：C1 声明回填（不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，`falsification` 字段显式补写「用哪个带」（本项取 0.014 bpc 带（限配对 Δ、≥5 窗口））；此前本项引用噪声带却只在 `validation` 声明，`falsification` 无声明 ⇒ 按该节判据不可判。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
