# 2.1.1-early-exit · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/2-depth/2.1.1-early-exit/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/2-depth/2.1.1-early-exit/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T13:30:00+08:00

- actor：轴 2 深度轴 lead（Phase A）
- change：role_map 实名（owner/writer/reviewer/integrator 由「待指派」改为实际角色，reviewer 独立于 writer）；执行 WP1：逐 token 退出决策实装进生产解码器（pathlm/depth_ar.py::DepthARDecoder.step_auto）+ 新模块 pathlm/exit_policy.py（策略 / 序列级口径 / 校准 / 算力账）+ tests/test_axis2_exit_decision.py（19 条，全绿）；新增实验 2631-pertoken-exit-seqlevel（纯推理、CPU、不取 GPU 锁；样本与历史 depth_curve 同一 seed=2 约定）。结论：逐 token 门控被同算力固定截断支配 ⇒ status 由「进行中」改「被否证」，补 failure_analysis（结构问题为主因、设计问题为次因、非 innovation 本身失败）与新的 gaps/roadmap（WP5 省算力对手 / WP6 策略感知训练）。

## 2026-10-13T10:37:02+08:00

- actor：轴 2 §2.1 worker
- change：按 MAPPING §1 由旧格 C1-layers-per-token 迁移核心（C1 → 2.1.1）；并入 C5-refinement-operator（C5 的 2523c → WP4 精化门）与 C6-depth-supervision（2518/2524 → WP3 深度监督配方）作为手段，整目录 git mv 进 experiments/。按 55-review C-PROBLEMS 重写：命题改为序列级 e2e、剔除墙钟卖点、不照抄「引擎已具备逐 token 门控」与「出口资产只能由深度监督造出」两个错误结论。

## 2026-10-14T06:44:03+08:00

- actor：framework-adjudication-writer（ticket 57）
- change：落地跨轴裁决（`FRAMEWORK-DECISIONS.md` C1/C2/61-F1，不改数字、不删证据）：`validation`/`falsification` 补「用哪个带」声明（本格配对带逐窗口 SE ≤0.008 为主判、0.014 为参考，一致率类走配对 bootstrap，C1）；`milestone.acceptance` 与 `conclusion` 补「字面成立 / 实质否证」并列读法（C2）；`validation` 写死序列级口径 = teacher-forcing 在线决策整段 bpc（61-F1）。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
