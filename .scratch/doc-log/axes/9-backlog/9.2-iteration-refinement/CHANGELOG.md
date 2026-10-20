# 9.2-iteration-refinement · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/9-backlog/9.2-iteration-refinement/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/9-backlog/9.2-iteration-refinement/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T11:32:11+08:00

- actor：轴 9 worker（ticket 56 M2）
- change：新建 Backlog 研究项：按 MAPPING.md §1/§4 把旧格 F2（2511-retry-matrix-withdrawal）与 F4（2619-reconstruction-loss-13m）的 experiments 整目录迁入本项；按 55-review F-PROBLEMS F-01~F-22 修正目的、判据口径与失败归因（补目的与可判定判据、换掉分布外主腿、等参数对照）。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
