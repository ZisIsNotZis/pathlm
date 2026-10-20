# 2.3.2-layer-resolution · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/2-depth/2.3.2-layer-resolution/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/2-depth/2.3.2-layer-resolution/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:37:02+08:00

- actor：轴 2 §2.3 worker
- change：新建研究项（无旧格，MAPPING §2）：逐层分辨率 归 2.3 深度的形状；按 docs/research-schema.md 预登记目的/基线/判据/否证条件，并把「长程召回损伤对照列」与「与 1.2.4 的边界」写死。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T21:30:00+08:00

- actor：轴 2 深度轴 lead（ticket 61）
- change：本轮终局推进登记 blocker（不改判据线、不改数字）：本项最小判决需要引擎不具备的逐层分辨率算子（池化/上采样/层级合并），等效算力成本 ≈ 20–25 min GPU + 60–90 min 实现工时，超出本轮 3h 预算 ⇒ 未执行；gaps 补「下一步最小实验 + 等效算力成本」与「金字塔第四臂未纳入」两条，conclusion_review 记当前状态。

## 2026-10-14T23:10:00+08:00

- actor：轴 2 深度轴 lead（ticket 61 · 复核 B 处置 writer）
- change：按复核 B 的 non-blocking 2.3.2-N1 对平臂数（不改判据、不改数字、不改状态）：`milestone.output`/`acceptance` 与 `WP1.means`/`acceptance` 補写「本轮先三臂 {全长, 先粗后细, 层级式}，金字塔补臂成本 +6–8 min GPU」，与 `gaps` 末条（金字塔未纳入最小判决）一致；conclusion_review 回填第 1 轮独立复核 = PASS。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
