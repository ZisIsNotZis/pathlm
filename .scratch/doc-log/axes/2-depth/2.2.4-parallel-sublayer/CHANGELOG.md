# 2.2.4-parallel-sublayer · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/2-depth/2.2.4-parallel-sublayer/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/2-depth/2.2.4-parallel-sublayer/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:37:02+08:00

- actor：轴 2 §2.2 worker
- change：按 MAPPING §1 由旧格 C3-inter-layer-topology 拆出并行子层读数归 2.2.4（证据只指针 see 2.2.3，不复制文件）；按 55-review C-P18 把「质量面单测判有害」改为「需质量面+代价面同框」的研究项。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T21:41:51+08:00

- actor：WP-2651 续做 writer（ticket 61 §9.2；fresh 上下文）
- change：WP-2651（2.2.3+2.2.4 的 seed=1 复核）续做轮：按 WP 契约把 `experiments` 加指向 2.2.3 的 `2651-inter-layer-topology-s1-13m`（只指针，不复制文件）；该目录内 `run.py` 已修崩溃并加逐臂增量落盘。当前事实：`sequential` 臂（seed=1）已训完，其余三臂待 GPU 锁；质量面（clean bpc）与代价面同框的质量代价未齐，`conclusion` / `failure_analysis` / `conclusion_review` / `argument` / `gaps` 未落。

## 2026-10-14T22:34:18+08:00

- actor：WP-2651 收口 writer（ticket 61 §9.2；fresh 上下文）
- change：WP-2651 收口轮：2.2.4 判词落盘——`parallel`/`parallel-dual-ln` 相对 `sequential` 的质量代价可排序（s0 +0.0458 = 3.27× 带 / +0.0402 = 2.87× 带），代价面给出真实收益（层内串行链 2→1 段、每 token 串行步数 16→8、算力差 0.055%、等参数臂逐位等参）⇒ 不满足 falsification 的「代价面无收益」条件，判为**以可排序质量代价换层内并行度翻倍（场景依赖取舍）**；`status` 由「进行中」改「已完成」，补 `conclusion`（含 C2 两读法）、`argument`，`experiments` 加 `2651` 的 `analysis.md`；`gaps` 记入「seed=1 两臂因跨轴 GPU 锁饥饿未跑 ⇒ n=1（等效成本 196.6M token）」；WP1 `status` 改「已完成」。证据 = `2651-inter-layer-topology-s1-13m/{analysis.md,MODEL.md,results.json}` 的代价面同框表。独立复核（第 1 轮 fresh + 处置 + 第 2 轮 fresh，均非本项 writer）：第 1 轮判 B1（+0.0402 = 2.87× 带，曾被写成「均 >3× 带」）等 1 blocking + 8 non-blocking，已逐条处置（提交 f4c991f）；第 2 轮判「**blocking：无**」（B1 已消除、倍数/区间跨字段一致、无新账实不符）。历轮见 ticket 61 §9.13 与 §9.18 复核记录（`.scratch/61-axis2-depth/TICKET.md`；复核者无写文件能力，判词原文在对应 run output log）。

## 2026-10-14T22:57:00+08:00

- actor：WP-2651 复核处置 writer（ticket 61 §9.2；fresh 上下文）
- change：复核 A 处置（**只改文本与状态语义，未动任何测量数字**）：①倍数订正——`parallel-dual-ln` 的质量代价 +0.0402 = **2.87×** 0.014 带（原写「均 >3× 带」不成立：3×0.014 = 0.042 > 0.0402），`parallel` 的 +0.0458 = **3.27×** 带；`conclusion` / `conclusion_review` / `change_log` 与 2.2.3 的 `2651.../analysis.md` / `.scratch/61-axis2-depth/WP-2651-REPORT.md` / 既有资产 `2613.../analysis.md` 的同族措辞一并改为实数倍数；②`gaps[0]`「无自有实验（按 MAPPING §3 只指针 see 2.2.3）」是事实/指针而不是缺口 ⇒ 挪进 `argument`，`gaps` 删除该条；③按复核 A 的严格读法把 WP1 `status` 由「已完成」改回「进行中」——WP 自己的预登记 `acceptance`/`means` 要求 n≥2，本轮止于 n=1（缺臂与等效成本见 `gaps`）；item 级 `status = 已完成` 与结论、`acceptance` 文字均不变。判据线未改。

## 2026-10-19T21:50:00+08:00

- actor：WP-R5 执行者（复核 03d78d4 的 F-1 处置；fresh 上下文）
- change：复核 F-1 处置（**只改文本与状态语义，未动任何测量数字**）：`conclusion` / `conclusion_review` 的臂完整性改为事实——s1（`2651`）四臂齐（`sequential` 1.5288 / `parallel` 1.5710 / `parallel-dual-ln` 1.5662 / `dense` 1.5273，`results.json::completed_arms` 四臂齐、`pending_arms` 为空），本判由 n=1 改为 **n=2**，并补 s1 的质量代价读数（+0.0422 = 3.01× 带 / +0.0374 = 2.67× 带）；`gaps` 删除「n≥2 未完成 / 两臂未跑 / n=1」一条；`conclusion` 正文删除时点标注（`docs/research-schema.md` §8）；WP1 `status` 由「进行中」改「已完成」（其预登记 `means`/`acceptance` 要求 n≥2，现已满足）。判据线未改，代价面读数未改。

## 2026-10-19T22:14:00+08:00

- actor：WP-R7 收口执行者（窄复核 P2 文本/指针项；fresh 上下文）
- change：窄复核 P2 收口（**只改文本与指针，未动任何测量数字、判据线与预登记 `acceptance`**）：①`sota_baseline` 的本地锚改为「旧格 C3（+0.0458 / +0.0402），现由 `2613` s0 与 `2651` s1 复现，n=2」，消除同一文件里 n=1 与 n=2 并置；②`conclusion_review` 末尾的复核指针由指向一个从未落盘的复核判词文件（复核者无写文件能力）改为真实存在的 `ticket 61 §9.13` 与 `§9.18 复核记录`（`.scratch/61-axis2-depth/TICKET.md`；复核者无 shell 与写文件能力，判词原文在对应 run output log）；③`roadmap` 删除已过期的「复核通过后写 conclusion，更新 status」。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
