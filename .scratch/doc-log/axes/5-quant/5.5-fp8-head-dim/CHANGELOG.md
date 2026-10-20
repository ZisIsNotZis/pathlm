# 5.5-fp8-head-dim · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/5-quant/5.5-fp8-head-dim/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/5-quant/5.5-fp8-head-dim/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T20:09:47+08:00

- actor：axis5-lead
- change：轴 5 Phase A（ticket 66）：role_map 四角色实名（writer = axis5-w5.5）；sota_baseline 由 `记忆·未核` 升级为本轮联网核实的 fp8 格式出处（arXiv:2209.05433）与 DeepSeek-V2 head_dim=128（arXiv:2405.04434）；gap 划清增量 = fp8 与 int8+块标尺在同一字节轴同一 13M 设定下比质量（补 5.1 的 fp8 质量缺口）+「head_dim → G → 每位置字节」的解析与实测对账，并写明 head_dim=32 时省幅在 46.875% 处封顶。

## 2026-10-13T11:06:24+08:00

- actor：轴 5 量化轴 worker
- change：新建研究项（无旧格，MAPPING §2）：fp8 / 大 head_dim 承接旧格 R1-quantization 的「fp8 质量、更大 head_dim 点」缺口、按 docs/structure.md §5.5 单列。按 docs/research-schema.md 预登记目的/基线/判据/否证条件，与 5.1 的 int8 块标尺同轴（防跨口径混用，55-review W-08/R8）。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T20:52:00+08:00

- actor：axis5-lead
- change：C1 声明回填补漏（ticket 66 Phase B）：脚本扫描（`.scratch/66-axis5-quant/c1_band_declaration_scan.py`）发现 `milestone.acceptance` 引用「越噪声带」但未点明用哪条带 ⇒ 补一句「跨格 0.014 bpc 带（限配对 Δ、≥5 窗口，见 `falsification` 的 C1 声明；本格无自测在力带）」。只加声明，不改数字 / 不改判据 / 不改证据。

## 2026-10-14T22:40:00+08:00

- actor：axis5-lead（ticket 66 Phase B）
- change：执行 WP1（实验 `2653-fp8-vs-int8-13m`，CPU 口径，5 窗口配对 Δ）+ WP2 的解析与 `head_dim`=32 的 G 实测；写 conclusion/experiments/gaps，WP1 未开始 → 已完成，status 保持 `未开始` → 改为 `进行中`（WP2 的 `head_dim`=64 实测腿未完成 ⇒ milestone 未全闭）。记第 1 轮独立复核（OK with notes，blocker 1）与返工处置。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
