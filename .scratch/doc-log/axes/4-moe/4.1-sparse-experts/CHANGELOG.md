# 4.1-sparse-experts · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/4-moe/4.1-sparse-experts/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/4-moe/4.1-sparse-experts/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T11:06:24+08:00

- actor：轴 4 专家轴 worker
- change：由旧格 W2-expertization 迁入建立（MAPPING §1：W2 → 4.1；无自有 experiments）。W2 的另两面按 docs/structure.md §4 拆为 4.2 路由与负载均衡、4.3 专家粒度/异构。按 55-review W-03/W-04 不照抄「todo 待补做」的立项语气、把造轮子风险写进 gap；按 W-13 消除 status/prior_art_verdict 自相矛盾；按 W-17 成本口径改为等效理论量；status 记为未开始（本项命题无本地判决）。

## 2026-10-13T19:25:15+08:00

- actor：axis4-lead
- change：Phase A（ticket 65）：role_map 实名；sota_baseline 改为可核实 arXiv 出处（本轮联网核实标题）；gap 写死「相对 Switch/GShard 的增量」（小尺度三列同框 + 等 FLOPs/等参数双对照 + tokens-per-param 欠拟合列）；新开 WP0（等 FLOPs/等参数臂记账 + MoE 接线 pre-flight，已完成）并把 WP1 的臂名与证据路径具体化；status 未开始 → 进行中；新增 experiments 目录 2610-moe-arm-ledger-13m。

## 2026-10-14T06:44:03+08:00

- actor：framework-adjudication-writer（ticket 57）
- change：落地 65-F1 / 65-F2 / C1 / C2 裁决（不改数字、不删证据）：status 进行中 → 未开始（降为 Backlog，不在 13M 花算力做 MoE 判决，保留 WP0 记账）；`gaps` 记 65-F1（业界 MoE 跑分线指针）与 65-F2（GPU 协调）；`validation`/`falsification` 补「用哪个带」（C1）；`milestone.acceptance` 补实质验收（C2）。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
