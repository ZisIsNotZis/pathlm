# 3.1.5-channel-sparsity · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/3-width/3.1.5-channel-sparsity/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/3-width/3.1.5-channel-sparsity/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:57:34+08:00

- actor：轴 3 §3.1 worker
- change：新建研究项（无旧格，MAPPING §2）：通道极化 / 稀疏激活 归 3.1 每 token 的激活宽度；按 docs/research-schema.md 预登记目的/基线/判据/否证条件，并把「活性对照列」与「与专家轴 4 的边界」写死。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T22:45:00+08:00

- actor：axis3-lead
- change：Phase B 登记：本项已落预登记契约 + 实验目录内 FFN 前向 patch 实现 + 单测（含等值门/变异门）；四个臂（`K50`/`K25`/`F50` 等）因 GPU 互斥锁饥饿一个都没跑，`status` 由 未开始 改 进行中、`experiments` 登记 2644；未跑项与下一步最小实验写入 gaps。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
