# 2.3.1-layer-capacity · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/2-depth/2.3.1-layer-capacity/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/2-depth/2.3.1-layer-capacity/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-14T23:40:00+08:00

- actor：轴 2 深度轴 lead（ticket 61 · 按 Main 裁决 b）
- change：按 Main 裁决 b 落盘（不改判词、不改数字）：① argument 增写「跨轴一致旁证」——引用轴 3 的 2641（同引擎同配方、等 FFN 预算、n=2，同得 H0），并明写「旁证不替代本项证据，本项判词只由本目录三臂本地实测支撑」；② gaps 的旁证条目改为「已采用为旁证」，并把 2.3.1 与 3.1.4 都动 mlp_hidden_sched 的重叠登记在案（该边界由 Main 在最终集成裁，本目录不合并、不改名、不改树）。

## 2026-10-13T10:37:02+08:00

- actor：轴 2 §2.3 worker
- change：新建研究项（无旧格，MAPPING §2）：逐层容量变化 归 2.3 深度的形状；按 docs/research-schema.md 预登记目的/基线/判据/否证条件，并把「固定总参数」与「与 3.1.2 的边界」写死。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T22:15:00+08:00

- actor：WP-2655 worker（ticket 61 · 轴 2 wave 2）
- change：WP1 进行中：按 `.scratch/61-axis2-depth/TICKET.md` §9.2 WP-2655 落 `experiments/2655-layer-capacity-13m/`（README 预登记 P1–P6 / config.json / run.py（支持 --arms/--no-train/--aggregate）/ smoke.py + smoke_report.json / test_2655_layer_capacity.py（8 passed）/ gpu_lock.py / run_remaining.sh / analysis.md / MODEL.md）；三臂改用纯 CLI `--mlp-hidden-sched`（E2，提交 4638882）表达逐层 FFN 隐藏宽，共享代码未改。实测：等预算门（三臂 Σh/FFN 参数/总参数/FFN MACs 逐位相等）、引擎等值门（显式 uniform schedule == 引擎缺省，state_dict 与前向逐位一致，带必红探针）、活性（逐层隐藏维 == schedule、FFN 相对幅度 > 0.47）全过；`uniform` 臂 clean bpc = 1.5291（与 2613 sequential-s0 差 +0.0037、与 2641 U4-s0 差 −0.0023，均 <0.005）。`front-big` / `growing` 两臂本轮一度等锁超时（等效成本 2 × 98.3M token 已登记），判据线未改、未砍（按契约砍臂不砍判据线）。

## 2026-10-14T22:56:00+08:00

- actor：WP-2655 收口 writer（ticket 61 · 轴 2 wave 2）
- change：WP1 收口为终局：`front-big` / `growing` 两臂在本窗口内训完并落盘（三臂完整性 as of 2026-10-14T22:53:04+08:00），`results.json` 汇总 P1–P6。判词：`front-big` clean bpc 1.5431（Δ = bpc_uniform − bpc_front-big = −0.0140，全精度 −0.0140016、窗间 SE 0.0031、逐窗 5/5 为负 ⇒ 可排序更差[边界]）、`growing` 1.5309（Δ = −0.0018、窗间 SE 0.0018 ⇒ 不可排序 H0）；H1（非均匀可排序更优）被否证，预登记 falsification 条件成立 ⇒ 本项 status 与 WP1 status 由「进行中」改「被否证」，补 `failure_analysis`（设计问题为主因：L=8 太浅、无中间深度监督、只重分配 FFN 隐藏宽；结构问题已由 P4/P5/P6 排除；不主张 innovation 本身失败）。改 `analysis.md`（人话摘要 + 三臂表 + 配对 Δ/SE + 三道门 + §12 三对账 + §13 对照三要素 + 字面/实质两读法 + 跨轴旁证标注）、`MODEL.md`（三臂 model.pt 的 sha256 与生成命令）。独立复核未做。

## 2026-10-14T23:10:00+08:00

- actor：轴 2 深度轴 lead（ticket 61 · 复核 B 处置 writer）
- change：按复核 B 的三个 non-blocking 处置（不改数字、不改判据、不改判词）：N1 在 `failure_analysis` 补「`front-big` 名义越带 1.6×10⁻⁶（方向为更差）+ `growing` 不越带 ⇒ 两臂都无正面增益；字面『不越带』一项由 `growing` 单独满足」；N2 移除实验根目录的三个重复产物（`uniform-s0-gate_log.jsonl` / `uniform-s0-shared_results.json` / `uniform-s0-train_log.jsonl`）——`cmp` 验证与 `runs/uniform-s0-12k/{gate_log.jsonl,results.json,train_log.jsonl}` 逐字节相同，不在 README 落盘清单内；N3 的 `TICKET §9.12`/`§9.15` 陈旧行属 lead 文件，本 writer 不改、已回报。conclusion_review 回填第 1 轮独立复核 = PASS（判词由 lead 落盘）。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
