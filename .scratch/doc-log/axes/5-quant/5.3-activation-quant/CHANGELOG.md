# 5.3-activation-quant · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/5-quant/5.3-activation-quant/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/5-quant/5.3-activation-quant/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T20:09:47+08:00

- actor：axis5-lead
- change：轴 5 Phase A（ticket 66）：role_map 四角色实名（writer = axis5-w5.3）；sota_baseline 由 `记忆·未核` 升级为本轮联网核实的 arXiv 出处（SmoothQuant 2211.10438 / LLM.int8() 2208.07339 / Atom 2310.19102）；gap 划清增量 = 小尺度受控含标尺激活字节账 + 把「逐层标尺/离群通道」做成可失败活性判据，声明不重造 SmoothQuant 迁移算法；本地锚口径说明同 5.2。

## 2026-10-13T11:06:24+08:00

- actor：轴 5 量化轴 worker
- change：新建研究项（无旧格，MAPPING §2）：激活量化 承接旧格 R1-quantization 的「激活量化 / 混合精度」选项、按 docs/structure.md §5 单列。按 docs/research-schema.md 预登记目的/基线/判据/否证条件，并把「逐层激活标尺 / 离群通道分布」列为必须报的活性对照列。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T22:40:00+08:00

- actor：axis5-lead（ticket 66 Phase B）
- change：执行 WP1（实验 `2652-act-quant-13m`：4 臂 × 5 窗口 + 离群活性列，CPU 口径）；写 conclusion/gaps/experiments/WP1 状态，status 未开始 → 已完成；记第 1 轮独立复核（FAIL，3 blocker）与返工处置（补跑 24×16@seed2 单点、订正不实口径叙述、按 n_batches 门控锚检查）。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
