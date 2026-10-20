# 4.3-expert-granularity · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/4-moe/4.3-expert-granularity/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/4-moe/4.3-expert-granularity/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T11:06:24+08:00

- actor：轴 4 专家轴 worker
- change：新建研究项（无旧格，MAPPING §2）：专家粒度 / 异构 承接旧格 W2-expertization 选项中的「细粒度专家」「共享专家」两面、按 docs/structure.md §4 单列；也是出资人提出的「激活更小专家而非更大专家」子问题。按 docs/research-schema.md 预登记目的/基线/判据/否证条件，并把「总参数与激活算力逐位对齐」写死。

## 2026-10-13T19:25:15+08:00

- actor：axis4-lead
- change：Phase A（ticket 65）：role_map 实名；sota_baseline 改为可核实 arXiv 出处；gap 写死「相对 DeepSeekMoE 的增量」（固定总参数/激活算力下的连续粒度扫描，不是固定点配方重跑）。status 保持未开始（无本地工作）。

## 2026-10-14T06:44:03+08:00

- actor：framework-adjudication-writer（ticket 57）
- change：落地 65-F1 / C1 / C2 裁决（不改数字、不删证据）：status 保持未开始（降为 Backlog，不在 13M 花算力做 MoE 判决）；`gaps` 记 65-F1（业界 MoE 跑分线指针）；`validation`/`falsification` 补「用哪个带」（C1）；`milestone.acceptance` 补实质验收（C2）。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
