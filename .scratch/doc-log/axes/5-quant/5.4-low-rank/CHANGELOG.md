# 5.4-low-rank · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/5-quant/5.4-low-rank/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/5-quant/5.4-low-rank/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T20:09:47+08:00

- actor：axis5-lead
- change：轴 5 Phase A（ticket 66）：role_map 四角色实名（writer = axis5-w5.4）；sota_baseline 由 `记忆·未核` 升级为本轮联网核实的 DeepSeek-V2（arXiv:2405.04434，MLA + head_dim=128）；gap 划清增量 = 与头共享在同一「每位置字节」轴上的受控比较 + 把「须从零训练 + 上投影计算」的代价入账。

## 2026-10-13T11:06:24+08:00

- actor：轴 5 量化轴 worker
- change：新建研究项（无旧格，MAPPING §2）：低秩压缩（MLA 式） 承接旧格 R1-quantization / W3-kv-head-sharing 的 KV 压缩一面、按 docs/structure.md §5.4 单列（MLA 与头共享是两种不同机制）。按 docs/research-schema.md 预登记目的/基线/判据/否证条件，横轴统一「每位置 KV 字节」、上投影计算入账，并把与 5.1/7-references 的边界写死。

## 2026-10-14T06:44:03+08:00

- actor：framework-adjudication-writer（ticket 57）
- change：落地 66-F4 裁决：登记等参数臂归属——与 `5.1` 共用同一 WP（`5.1` 的 `WP3`），本项 `WP1` 引用之（`gaps` + `roadmap`）。不改数字。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
