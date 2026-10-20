# 4.2-routing-balance · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/4-moe/4.2-routing-balance/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/4-moe/4.2-routing-balance/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T11:06:24+08:00

- actor：轴 4 专家轴 worker
- change：新建研究项（无旧格，MAPPING §2）：路由与负载均衡 承接旧格 W2-expertization「负载均衡/路由坍缩是主要风险」的一面、按 docs/structure.md §4 单列。按 docs/research-schema.md 预登记目的/基线/判据/否证条件，并把「专家使用率分布」与「跨 seed 路由稳定性」写为必须报的活性对照列。

## 2026-10-13T19:25:15+08:00

- actor：axis4-lead
- change：Phase A（ticket 65）：role_map 实名；sota_baseline 改为可核实 arXiv 出处；gap 写死「相对 Switch aux loss / GShard capacity factor / expert-choice 的增量」（小尺度三元前沿 + 均衡必要性可失败判据）；status 未开始 → 进行中（前提探针已跑，见 conclusion）；按探针结果把 falsification 的「仍坍缩」分支降为「本轮探针口径下尚无先验可触发性」（而非「实际不可触发」）；按独立复核 `REVIEW-1.md` 把 conclusion 的主证改为熵斜率 + 持久偏斜，并注明使用率为跨 12 层池化口径。

## 2026-10-14T06:44:03+08:00

- actor：framework-adjudication-writer（ticket 57）
- change：落地 65-F1 / C1 / C2 裁决（不改数字、不删证据）：status 进行中 → 未开始（降为 Backlog，不在 13M 花算力做 MoE 判决，保留坍缩探针记录）；`gaps` 记 65-F1（业界 MoE 跑分线指针）；`validation`/`falsification` 补「用哪个带」（C1）；`milestone.acceptance` 补实质验收（C2）。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
