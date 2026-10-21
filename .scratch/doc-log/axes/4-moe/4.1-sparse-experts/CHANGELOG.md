# 4.1-sparse-experts · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/4-moe/4.1-sparse-experts/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/4-moe/4.1-sparse-experts/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T11:06:24+08:00

- actor：轴 4 专家轴 worker
- change：由旧格 W2-expertization 迁入建立（MAPPING §1：W2 → 4.1；无自有 experiments）。W2 的另两面按 docs/structure.md §4 拆为 4.2 路由与负载均衡、4.3 专家粒度/异构。按 55-review W-03/W-04 不照抄「todo 待补做」的立项语气、把造轮子风险写进 gap；按 W-13 消除 status/prior_art_verdict 自相矛盾；按 W-17 成本口径改为等效理论量；status 记为未开始（本项命题无本地判决）。

## 2026-10-13T19:25:15+08:00

- actor：axis4-lead
- change：Phase A（ticket 65）：role_map 实名；sota_baseline 改为可核实 arXiv 出处（本轮联网核实标题）；gap 写死「相对 Switch/GShard 的增量」（小尺度三列同框 + 等 FLOPs/等参数双对照 + tokens-per-param 欠拟合列）；新开 WP0（等 FLOPs/等参数臂记账 + MoE 接线 pre-flight，已完成）并把 WP1 的臂名与证据路径具体化；status 未开始 → 进行中；新增 experiments 目录 2610-moe-arm-ledger-13m。

## 2026-10-14T06:44:03+08:00

- actor：framework-adjudication-writer（ticket 57）
- change：落地 65-F1 / 65-F2 / C1 / C2 裁决（不改数字、不删证据）：status 进行中 → 未开始（降为 Backlog，不在 13M 花算力做 MoE 判决，保留 WP0 记账）；`gaps` 记 65-F1（业界 MoE 跑分线指针）与 65-F2（GPU 协调）；`validation`/`falsification` 补「用哪个带」（C1）；`milestone.acceptance` 补实质验收（C2）。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。

## 2026-10-21T20:35:06+08:00

- actor：研究级主 Agent（4.1 的 writer，按上一轮独立复核的五条 P1 修判词与验收）
- change：① `status` 被否证 → 已完成——`falsification` 第一条的第三个连词「且无法由使用率分布解释」不成立（坍缩在四条 MoE 臂上都登记在案、且与「容量兑现不了」在本格读数上分不开），按预登记决策树只走到「本尺度不可分、不外推」，判词落点改成「本尺度下不可排序；只有等参数旁证越带」；② milestone 验收第三条「专家使用率分布齐备且非坍缩」字面不成立（4/4 臂 `collapse_flagged = true`），在 `conclusion` ⑩ 做一次显式 C2 裁决（实质验收换成「坍缩已逐层量化并登记、且列为未排除的替代解释」），并同步 `acceptance_evidence` 第三条、`roadmap` 末条与 `atlas/triage.md` 的「里程碑达成」措辞；③ `ledger.json` / `preflight.json` 由第六次绑定代码重跑（纯 CPU、秒级、不碰契约文件，只刷新 `code_hashes` 三个字段），`analysis.md` 的等激活 MACs 引用改指 `results.json::ledger.checks`，并写明 `2610` 全目录 `bound_at = unbound`（定序不可判）；④ `2610/analysis.md` §4 的「朴素 top-k 路由不会自发坍缩」标为已被 `2611` §5 取代，只保留「池化口径看不见坍缩」；⑤ 登记等参数稠密相对 `dense-B0` 只换来 +0.00749（s0 记账值）/ +0.01487（s1 未配对均值之差）这条读数，把它写成「本预算下参数/算力缩放本身买不到越带质量」的并列解释候选，并写明它对本格区分力的限制与它对「坍缩全解释」的反证。判据线、0.014 带、臂清单、配对规则、评测口径、token 预算、步数、种子与全部训练读数一字未动。
