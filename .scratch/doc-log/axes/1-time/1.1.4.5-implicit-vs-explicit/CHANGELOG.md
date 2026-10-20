# 1.1.4.5-implicit-vs-explicit · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/1-time/1.1.4.5-implicit-vs-explicit/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/1-time/1.1.4.5-implicit-vs-explicit/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:15:48+08:00

- actor：轴 1 §1.1 worker
- change：由旧格 A9-old-info-retention 迁移建立（A9 → 1.1.4.5）；2 个实验目录整目录搬入 experiments/；按 55-review P14 把与 1.1.1 重复的部分归 1.1.1、本项聚焦 implicit vs explicit 对撞；按 55-review §6 补上「同仪器/同预算正面对撞」与「前缀可丢端到端协议」为缺口。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T23:17:51+08:00

- actor：axis1-lead-4
- change：判终局（ticket 62 §10.25）：status 进行中 → **已完成**。实验 2661（6 臂纯推理、预登记 8 条判据线逐条过、`full rc=0`）给出本项的正面对撞读数：质量腿 +0.100515（CI [0.094630, 0.106286]，补回窗口 16→全量差距的 80.1%）；可检索性腿在同索引硬子集上 10.226% vs implicit 0.704%/1.000%（差 +9.522pp，CI [8.336, 10.708]）；仪器前提（无通道臂逐位不敏感）与负对照（implicit 停地板）都成立；预登记 falsification 的两支都**未触发**。两读法（C2）：字面「大部分成立」（delta 写这一类 n=2、对照的 stock 全因果锚取自同族 2638/2639、本实验未含 full 臂），实质「explicit 是有希望的方向」**成立**。同时如实登记更严聚合口径`intervention_arms_all_pass=false` 的差异、完备臂 seed 0 的种子缺口、以及与 1.1.4.1 共有的 F8 归因缺口。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
