# 5.2-weight-quant · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/5-quant/5.2-weight-quant/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/5-quant/5.2-weight-quant/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T20:09:47+08:00

- actor：axis5-lead
- change：轴 5 Phase A（ticket 66）：role_map 四角色实名（writer = axis5-w5.2）；sota_baseline 由 `记忆·未核` 升级为本轮联网核实的 arXiv 出处（GPTQ 2210.17323 / AWQ 2306.00978 / LLM.int8() 2208.07339）；gap 划清「相对 SOTA 的增量」= 13M 受控含标尺字节账 + 三类账（权重/KV/激活）分开报，并声明不重造 GPTQ/AWQ 算法；本地锚补「同一横轴宜用与 5.1 相同的 13M trunk = 1.483571，B0 = 1.5074 仅历史锚、不得跨锚比较」。

## 2026-10-13T11:06:24+08:00

- actor：轴 5 量化轴 worker
- change：新建研究项（无旧格，MAPPING §2）：权重量化 承接旧格 R1-quantization 的「权重 int8/int4」选项、按 docs/structure.md §5 单列。按 docs/research-schema.md 预登记目的/基线/判据/否证条件，字节口径含标尺（防逐值标尺式错误），并把与 5.1/5.3 的三类账边界写死。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T22:40:00+08:00

- actor：axis5-lead（ticket 66 Phase B）
- change：执行 WP1（实验 `2651-weight-quant-13m`：4 臂 × 5 窗口，CPU 口径）；写 conclusion/gaps/experiments/WP1 状态，status 未开始 → 已完成；记第 1 轮独立复核（OK with notes）与返工处置。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
