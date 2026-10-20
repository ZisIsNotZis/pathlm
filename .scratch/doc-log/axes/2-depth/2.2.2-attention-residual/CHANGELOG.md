# 2.2.2-attention-residual · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/2-depth/2.2.2-attention-residual/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/2-depth/2.2.2-attention-residual/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:37:02+08:00

- actor：轴 2 §2.2 worker
- change：新建研究项（无旧格，MAPPING §2）：用 causal attention 代替残差 承接出资人 C3 批注从未测的设计（55-review C-P10）；按 docs/research-schema.md 预登记目的/基线/判据/否证条件，并把与 2.2.3 线性学习加权和的同预算对照钉死。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T22:00:00+08:00

- actor：WP-2653 worker（ticket 61 · 轴 2 wave 2）
- change：WP1 进行中：按 `.scratch/61-axis2-depth/TICKET.md` §9.2 WP-2653 落 `experiments/2653-interlayer-attn-13m/`（README 预登记 P1–P6 / config.json / run.py（支持 --arms/--no-train/--aggregate）/ interlayer_attn.py（实验内工厂替换，不改共享代码）/ smoke.py / test_2653_interlayer_attn.py / gpu_lock.py / run_remaining.sh / analysis.md / MODEL.md）；实装层间 causal attention 聚合（层索引因果：只取 states[0..k−1]）并实测三件事：①等值门（one-hot 强制 ⇒ 前向/损失与 stock 单残差逐位一致，带必红探针）；②活性（k ≥ 2 的层权重非退化）；③参数账（+8 / +196 616，含每 token 额外 MACs 18 432 / 5 261 312）。status 未开始 ⇒ 进行中；两臂训练未跑成（等锁超时，等效成本 2 × 98.3M token 已登记），判据线未改、未砍。

## 2026-10-14T23:10:00+08:00

- actor：轴 2 深度轴 lead（ticket 61 · 复核 B 处置 writer）
- change：按复核 B 的三个 non-blocking 处置（不改数字、不改判据、不改结论）：N1 `argument` 的参数倍率改「约 2.5 万倍」（196 616 / 8 = 24 577，原写 2.7 万与本句自引数字不符）；N2 活性区间限定词改「k ≥ 5 的层熵 ≈ 1.6–2.1（层 2–4 更低，熵 0.69–1.38）」，与 `smoke_report.json::B_liveness_gate.per_layer` 一致（`analysis.md` 同步）；N3 `2653/README.md` §7/§9 落盘清单删去并不存在的 `run_log.txt`，两臂未训的阻塞证据只在 `.scratch/61-axis2-depth/TICKET.md` §9.11。conclusion_review 回填第 1 轮独立复核 = PASS（判词由 lead 落盘）。

## 2026-10-19T21:31:45+08:00

- actor：WP-R2/R3 worker（ticket 61 · 轴 2 wave 3）
- change：两臂（attn-scalar / attn-qkv）训练并评测落地，P1 判词按预登记判据线给出（scalar −0.0082 不可排序、qkv −0.0177 反向可分），判据线与臂定义未改；据此改 status 进行中 → 被否证，补 failure_analysis（设计 / 结构 / innovation 分开）与 conclusion，gaps 换成本轮未测项（n≥2、scalar 塌缩机理、qkv 等参数剥离），WP1 进行中 → 已完成，roadmap 换成剩余路径。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
