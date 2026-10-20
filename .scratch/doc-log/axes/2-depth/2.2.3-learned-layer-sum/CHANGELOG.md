# 2.2.3-learned-layer-sum · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/2-depth/2.2.3-learned-layer-sum/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/2-depth/2.2.3-learned-layer-sum/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:37:02+08:00

- actor：轴 2 §2.2 worker
- change：按 MAPPING §1 由旧格 C3-inter-layer-topology 并入 2.2.3 层间学习加权和（DenseFormer 式），`2613-inter-layer-topology-13m` 整目录 git mv 为本项主家证据（并行子层读数归 2.2.4，只指针不复制）；按 55-review C-P10 把「手段枚举、判据零区分度」改成有目的/可排序判据的研究项，status 由 judged 降为进行中（n=1/单快照）。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T21:41:51+08:00

- actor：WP-2651 续做 writer（ticket 61 §9.2；fresh 上下文）
- change：WP-2651（2.2.3+2.2.4 的 seed=1 复核）续做轮：修 `.../2651-inter-layer-topology-s1-13m/run.py` 从训练脚本产物 `results.json` 取 `bpc` 的崩溃（上一轮 `KeyError: 'bpc'`；`clean_bpc` 唯一来源＝`run_battery → depth_curve[-1]['bpc']`，缺键显式报错不静默填 0），并加逐臂增量落盘（每臂写 `results.json` + append `run_log.txt`）；`experiments` 加入本 WP 实验目录。当前事实：`sequential` 臂（seed=1）已训完，其余三臂待 GPU 锁；四臂齐后的判词、`conclusion` / `failure_analysis` / `conclusion_review` / `argument` / `gaps` 未落。

## 2026-10-14T22:34:18+08:00

- actor：WP-2651 收口 writer（ticket 61 §9.2；fresh 上下文）
- change：WP-2651 收口轮：本项判词由预登记 falsification 给出并落盘——`dense − sequential` 配对 Δ（s0 = −0.0008，不越 0.014 带）且逐深度可读性无改善（s0 中间深度 depth 1–7 Δ = +0.226…+0.544）⇒ 密集跨层线性聚合在本尺度无净增量，`status` 由「进行中」改「被否证」，补 `failure_analysis`（结构问题为主因 / 设计问题次因 / 非 innovation 本身失败 / 非实现口径假象）、`conclusion`（含 C2 字面成立-实质否证两读法）、`argument`；`experiments` 加 `2651` 的 `analysis.md`；`gaps` 记入「seed=1 三臂因跨轴 GPU 锁饥饿未跑（等效成本 294.9M token）⇒ n=1」与更大规模/单纯形约束/2.2.2 对照等缺口；WP1 `status` 改「已完成」。证据 = `2651-inter-layer-topology-s1-13m/{analysis.md,MODEL.md,results.json}`（seed=1 只有基线臂完成，另三臂 missing）。独立复核（第 1 轮 fresh + 处置 + 第 2 轮 fresh，均非本项 writer）：第 1 轮判 B1（+0.0402 曾被写成「>3× 带」，实为 2.87×）等 1 blocking + 8 non-blocking，已逐条处置（提交 f4c991f）；第 2 轮判「**blocking：无**」（B1 已消除、未发现新 blocking，仅剩 4 条 non-blocking：本字段旧文本、3.3 倍四舍五入、2613 改动的 change_log 线索、第 1 轮 N2）。历轮见 ticket 61 §9.13 与 §9.18 复核记录（`.scratch/61-axis2-depth/TICKET.md`；复核者无写文件能力，判词原文在对应 run output log）。

## 2026-10-14T22:58:00+08:00

- actor：WP-2651 复核处置 writer（ticket 61 §9.2；fresh 上下文）
- change：复核 A 处置（**只改文本与状态语义，未动任何测量数字**）：①`failure_analysis` 的中间深度换 seed 散度区间由 0.010–0.121 更正为 **0.006–0.121**（depth 1–7；0.010 是 depth 0 的值），倍数限定改为逐深度 2.5–46 倍（原写 2–5 倍只在 depth 3–6 成立）；同句在 `2651.../analysis.md::§9.1` 一并更正；②`2651.../analysis.md::§6.1` 删「均在界内」（log₂(206) = 7.69 是参考值而非理论上界）；③`2651.../README.md::§5` 的 s0 等值证据指针落到**实际存在**的 `run_log-attempt1-crash.txt:3`，并注明 `results.json::s0_anchor_remeasured` 待下次全跑落盘；`2651.../analysis.md::§1` 补一句下游读 `results.json` 必须执行 `train_tokens != 98,304,000 ⇒ 当 missing` 自检；④按复核 A 的严格读法把 WP1 `status` 由「已完成」改回「进行中」——预登记 `acceptance`/`means` 要求 n≥2，本轮止于 n=1（缺臂与等效成本见 `gaps`）；item 级 `status = 被否证` 与结论、`acceptance` 文字均不变。判据线未改。

## 2026-10-19T22:14:00+08:00

- actor：WP-R7 收口执行者（窄复核 P2 指针项；fresh 上下文）
- change：窄复核 P2 收口（**只改指针，未动任何测量数字、判据线与预登记 `acceptance`**）：本文件 `conclusion_review` 与本条末尾的同族复核指针由指向一个从未落盘的复核判词文件（复核者无写文件能力）改为真实存在的 `ticket 61 §9.13` 与 `§9.18 复核记录`（`.scratch/61-axis2-depth/TICKET.md`；复核者无 shell 与写文件能力，判词原文在对应 run output log）。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
