# 5.1-kv-quant · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/5-quant/5.1-kv-quant/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/5-quant/5.1-kv-quant/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T20:13:43+08:00

- actor：axis5-lead
- change：轴 5 Phase A（ticket 66）：role_map 四角色实名（owner/reviewer/integrator 对齐兄弟轴惯例，writer = axis5-w5.1）；sota_baseline 由 `记忆·未核` 升级为本轮联网核实的 arXiv 出处（KIVI 2402.02750 / KVQuant 2401.18079 / Atom 2310.19102）；gap 划清机制归属（`block` ≈ 业界 per-channel/分块标尺、`block_tok`/`kivi` ≈ KIVI 的 K 侧 per-token ⇒ 机制非增量，增量 = 部署字节账 + 严格前缀不变性 + 统一字节轴）。执行 WP2（实验 `2642`，纯 CPU 再分析）：独立字节账对账 `max|Δ| = 0`、跨家族同字节分组 7 个、参数量混杂拆开（−8.75%/−10.21%，解析=实测）、fp8 解析点补入同轴；新增 experiment 目录、WP2 置已完成、gaps 改为「等参数臂 / 逐字节水平补臂 / int4 组合归因」，并排 fresh 复核。纯 CPU 再分析未取 GPU 锁（SCHEMA.md §20 不适用），未改共享文件。

## 2026-10-13T11:06:24+08:00

- actor：轴 5 量化轴 worker
- change：由旧格 R1-quantization 迁入建立（MAPPING §1：R1 → 5.1；六个 kv 实验目录整目录 git mv）。R1 的权重/激活量化与 fp8/大 head_dim 缺口按 docs/structure.md §5 分到 5.2/5.3/5.5。按 55-review W-08 / R8 用块标尺（含标尺/staging 的部署字节）口径取代逐值标尺口径、不迁入已作废的 int4 跨家族数字；按 W-09 判据改为每位置字节-质量前沿；按 W-17 成本口径改为等效理论量。status 记已完成（KV 量化的块标尺口径已有本地判决）。

## 2026-10-14T06:44:03+08:00

- actor：framework-adjudication-writer（ticket 57）
- change：落地跨轴裁决（`FRAMEWORK-DECISIONS.md` C1/C2/66-F1/66-F2/66-F4，不改数字、不删证据）：`falsification` 补「用哪个带」声明（本格在力带 0.036324 为主判、0.014 为参考，C1/66-F1）；`milestone.acceptance` 与 `conclusion` 补「字面成立 / 实质成立」并列读法（C2）；`gaps` 记 66-F2（头族优势记为参数混杂、不算机制发现）与 66-F4（等参数臂 = 5.1/5.4 共用 WP）；`work_packages` 新增 WP3（等参数头共享臂，与 5.4 共用）。status 保持已完成（66-F3）。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
