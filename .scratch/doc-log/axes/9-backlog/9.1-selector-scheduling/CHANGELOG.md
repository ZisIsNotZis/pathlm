# 9.1-selector-scheduling · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/9-backlog/9.1-selector-scheduling/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/9-backlog/9.1-selector-scheduling/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T09:40:25+08:00

- actor：WP-M1.0 foundation
- change：原 G1/G3/G4 合并为 Backlog 研究项（G→Backlog）；证据保留在 archive/wiki-legacy/axes/G-policy/**，见 MAPPING.md。

## 2026-10-13T11:32:11+08:00

- actor：轴 9 worker（ticket 56 M2）
- change：按 MAPPING.md §1/§4 把旧格 G1/G3 的 experiments（2508-tts-probes / 2514-allocator-selectivity / 2515-slider-rung3 / 2517-slider-at-scale / 2522-gen-quality-signals）整目录迁入本项 experiments/；补 WP3（观测信号）与 WP4（校准）为手段；按 55-review G-PROBLEMS P1–P4/P9 修正判据与结论口径（前沿上移、禁工具 KPI、假设失效升格）。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
