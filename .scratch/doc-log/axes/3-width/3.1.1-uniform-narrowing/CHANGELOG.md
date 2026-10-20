# 3.1.1-uniform-narrowing · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/3-width/3.1.1-uniform-narrowing/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/3-width/3.1.1-uniform-narrowing/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:57:34+08:00

- actor：轴 3 §3.1 worker
- change：由旧格 W1-active-width 迁入建立（MAPPING §1：W1 → 3.1.1；实验整目录迁入）。按 55-review W-06 剔除「读出切片当减宽判决」的错配证据、按 W-07 不再照抄「减宽=减容量」的业界结论、按 W-14 剔除不可核实的用户本地实验引用、按 W-17 把成本口径改为等效理论量；status 由旧 todo 记为未开始（本项命题无本地判决）。

## 2026-10-14T06:44:03+08:00

- actor：framework-adjudication-writer（ticket 57）
- change：落地 64-F1 / 64-F3 裁决（不改数字）：`method` 写死等 FFN 预算 / 等算力位点为剪枝类标准对照口径（64-F1）；`validation` 记预算不足时部分臂允许 n=1、按既定递减（64-F3）。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T22:45:00+08:00

- actor：axis3-lead
- change：Phase B 收口：2642（D192 uniform 减宽 + WD256 嵌套宽度 dropout，各 12k）执行完成并落盘，填 conclusion / status=已完成 / WP1+WP2=已完成 / experiments / ended，把 D128 未跑、单 seed、WD256 单训练等未测项写进 gaps；结论按 C2 两读法并列，并按复核 R-3.1.1 第 1 轮的 blocking 把「每 1% 算力代价的斜率差」改判为「落在 0.014 带内 ⇒ 本尺度不可排序」，第 2 轮复核 PASS。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
