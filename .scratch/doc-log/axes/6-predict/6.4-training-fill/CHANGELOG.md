# 6.4-training-fill · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/6-predict/6.4-training-fill/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/6-predict/6.4-training-fill/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:47:00+08:00

- actor：轴 6 §6.4 worker
- change：按 MAPPING §1 由旧格 H2-training-fill 迁入（H2→6.4）；其 experiments/2523e-fill-variants 整目录 git mv 进本项 experiments/。按 55-review H-PROBLEMS H-I7 修正：把目的从「一次前向输出几个位置」的附属项改为「训练时未来 / 已退出位置的填充语义」（服务变深度 AR / reentry 与多位置预测），判据改为 clean bpc（税）+ 填入位置可读性（该涨）+ 提交质量 rescore（OOD 污染该低），并钉「禁墙钟、matches-step 口径」。

## 2026-10-13T12:15:00+08:00

- actor：ticket 56 M2 integrator
- change：修正 status 自洽（独立复核 rev-axis6 D3）：顶层 status 由「已完成」改为「进行中」、ended 置空。依据 docs/research-schema.md §7（status 只回答「做完没有」）——本项 milestone 要求 13M matched-step n≥2，而 WP1 进行中、WP2 未开始、gaps 自述无 n≥2 复本，故「已完成」高报；测量数字与 conclusion 正文未动。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T20:52:40+08:00

- actor：axis6-lead（ticket 63 续 · C1 声明回填第二轮）
- change：C1 声明回填第二轮（不改数字 / 不改结论 / 不改证据）：按 docs/research-schema.md §12.1，本项 milestone.acceptance / objective_criteria / WP2.acceptance 引用 0.014 噪声带但未在同字段声明用哪个带，且 WP2 的 rescore 峰值属率类（跨量纲风险），现改为——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口）；rescore 峰值 / 命中率等率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用。

## 2026-10-14T22:54:52+08:00

- actor：axis6-lead
- change：Phase C 收口（`2672-fill-n2-13m`）：conclusion 新增 (f) n≥2 快照复本（fill>nofill 两种子同向、方向与 12k 档一致）、(g) proj-fill 排序**依赖训练步数**（2k 快照反转，不得外推）、(h) 等参数软/硬提交剥离成立；failure_analysis 填入（唯一未复现的是 proj-fill 排序，归因训练步数效应，非设计/innovation 失败）；milestone 的规模标签按落盘实测更正为 6,962,688（原文「13M」为措辞误标）；WP1/WP2 status → 已完成 并回填复核判词；experiments 增 2672；gaps 换成剩余项（12k 档 projfill 第二 seed、三臂旧读数、嵌入层分布检验未做、目录名与契约 §6 不一致）；status 进行中→已完成、ended 落盘。

## 2026-10-14T23:20:00+08:00

- actor：axis6-lead（Main 批准的规模标签更正）
- change：规模标签正名：预登记为「13M」，本仓 23e 协议与 2672 的**实测参数量 = 6,962,688**（d=256/L=8/h=8/mlp_mult=4）——仅正名、未改判据（判据 / 阈值 / 否证条件逐字未动）。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
