# 5.6-window-quant-retrieval · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/5-quant/5.6-window-quant-retrieval/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/5-quant/5.6-window-quant-retrieval/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T20:09:47+08:00

- actor：axis5-lead
- change：轴 5 Phase A（ticket 66）：role_map 四角色实名（writer = axis5-w5.6）；sota_baseline 的量化腿升级为本轮联网核实的 KIVI/KVQuant/Atom 出处；gaps 补「组合臂应对齐 5.1 的 2642 给出的 7 个同字节水平」（禁插值）。

## 2026-10-13T11:06:24+08:00

- actor：轴 5 量化轴 worker
- change：由旧格 R4-window-kv-quant-retrieval 迁入建立（MAPPING §1：R4 → 5.6；实验 2617-prior-art-hybrid-kv 整目录 git mv）。按 55-review 修正：旧「窗口几乎零代价」与其自家 ERRATA 冲突，改记为 bpc 损伤随保留预算单调；量化腿改指 5.1 的块标尺口径、不引用已作废的逐值标尺 int4 数字；判据改为「每位置字节 → 两腿质量」前沿。status 记进行中（组合前沿与检索腿仍未跑）。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T20:52:00+08:00

- actor：axis5-lead
- change：C1 声明回填补漏（ticket 66 Phase B）：脚本扫描（`.scratch/66-axis5-quant/c1_band_declaration_scan.py`）发现 `falsification`（§12.1 硬性项）引用 band / threshold 却未点明用哪条带 ⇒ 在 `falsification` 末补一句「用哪个带（C1 裁决）：0.014 bpc 带（限配对 Δ、≥5 窗口）为主判；长程检索召回等率类指标走配对 bootstrap / Wilson 95% CI（对照 chance = 0.0156 与同掩码地板）、不跨量纲借用 0.014；本格无自测在力带」。只加声明，不改数字 / 不改判据 / 不改证据。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
