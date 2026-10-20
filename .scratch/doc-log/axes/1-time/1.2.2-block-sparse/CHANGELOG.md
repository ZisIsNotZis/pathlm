# 1.2.2-block-sparse · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/1-time/1.2.2-block-sparse/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/1-time/1.2.2-block-sparse/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:29:52+08:00

- actor：轴 1 §1.2 worker
- change：由旧格 A6-mixing-operator 拆分迁移建立（A6 → 1.2.2 + 1.2.3，见 MAPPING §3）；experiments 2616-linear-attention-13m、2637-linear-attn-quality 整目录搬入 experiments/；按 55-review A-P11/P12/R2 拆开「可见模式（掩码）」与「合并算子（线性核）」，把旧「线性更差」降为带伤/已改核的负观测、不再当块稀疏结论；补真块稀疏掩码与无伤线性质量点两个 WP。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
