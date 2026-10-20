# 3.2.1-readout-reparam · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/3-width/3.2.1-readout-reparam/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/3-width/3.2.1-readout-reparam/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:57:34+08:00

- actor：轴 3 §3.2 worker
- change：由旧格 W4-readout-geometry 迁入建立（MAPPING §1：W4 → 3.2.1；5 个实验整目录迁入）。按 55-review W-10 明确结论为「不采纳为通用质量机制、只补本 depth-AR 形态缺陷」，按 W-11 把判据从「bpc 降了」改为「等参数对照归因 + 可迁移性」；成本口径改为等效理论量（W-17）。status 由旧 judged 记为已完成（已到本地明确结论）。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
