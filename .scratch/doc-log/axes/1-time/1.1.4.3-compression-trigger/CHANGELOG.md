# 1.1.4.3-compression-trigger · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/1-time/1.1.4.3-compression-trigger/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/1-time/1.1.4.3-compression-trigger/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:15:48+08:00

- actor：轴 1 §1.1 worker
- change：由旧格 A10-compression-trigger 迁移建立（A10 → 1.1.4.3 压缩触发与优先级）；experiments 整目录搬入；按 55-review P15/P16/P23 修正——判据从「LRU 无收益」改为「固定 ρ 下重建质量 + ρ-质量曲线 + 同可见性对数预算」，并注明旧门槛算术不可达；补目的（同预算下丢失信息最少）。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T22:16:41+08:00

- actor：axis1-lead-3
- change：Phase C 收口：**不改 status（仍进行中）**——复核现盘证据后确认本项 `falsification` 的两条腿都不能按预登记口径判：① 依赖腿（bpc）重对齐后 content +0.0132 / +0.0092 < 0.014 带、lru 更差，方向清楚；② 召回腿的预登记门槛（+10pp）被 2615 §6.1 证明**算术不可达**，而按 §12.1 的率类口径（配对 CI 不含 0）**现盘证据答不了**（2615 未存逐针 CI）。故按「判据悬空、无证据的结论不入账」留 `进行中`，并把「重算配对 CI」写成 0 训练 token 的下一步最小实验（见 gaps 末条）。不改任何数字、不改判据线。

## 2026-10-14T22:32:30+08:00

- actor：axis1-lead-4
- change：按 ticket 62 §10.21 完成「保留腿配对 CI 重算（0 训练 token）」：新增 `recompute_pair_ci.py` / `pair_ci_report.json` / `PAIR-CI-NOTE.md` 并据此更新 gaps 末条——**结论 = 配对判据不可复原 ⇒ 本项仍不可判（维持 `进行中`，判据悬空不入账）**；并列报出非配对近似的补充读法（硬子集 n=2322 上全部含 0）与下一步（重跑评测腿并落盘逐针记录）。不改任何已判数字、不改判据线。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
