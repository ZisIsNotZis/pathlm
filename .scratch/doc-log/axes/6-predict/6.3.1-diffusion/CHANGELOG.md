# 6.3.1-diffusion · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/6-predict/6.3.1-diffusion/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/6-predict/6.3.1-diffusion/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:47:00+08:00

- actor：轴 6 §6.3 worker
- change：新建研究项（无旧格，MAPPING §2）：diffusion（离散扩散 / 掩码扩散）归 6.3 整段并行生成（不止 AR）；按 docs/research-schema.md 预登记目的 / 基线 / 判据 / 否证条件，并把 tokens-per-forward（该涨）与每 token 等效 FLOPs（代价轴）分开写死。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T20:52:40+08:00

- actor：axis6-lead（ticket 63 续 · C1 声明回填第二轮）
- change：C1 声明回填第二轮（不改数字 / 不改结论 / 不改证据）：按 docs/research-schema.md §12.1，本项 milestone.acceptance 与 objective_criteria 引用 0.014 噪声带但未在同字段声明用哪个带，现补声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），tokens-per-forward / 一致率等率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用。

## 2026-10-14T23:20:00+08:00

- actor：axis6-lead
- change：Phase C（WP1 = `experiments/2674-mdiffusion-13m`，复用 6.3.2 harness）：conclusion 首次填入（K=1/4/8/16 的 Δ = +2.6815…+2.6873，CI 不含 0 且远越 0.014 带；提高 K 拉不回 ⇒ 契约 §4 否证条件整体触发；K=1 连接点显示变密度掩码训练比固定 r=0.25 差 1.61 bpc）；failure_analysis 填入（结构为主 + regime 成分）；milestone 规模标签按实测更正；gaps 换成剩余项（均匀核臂、调度扫描、L5 确定性、快照口径）；status 未开始→进行中（复核 mode=PENDING）。

## 2026-10-14T23:10:00+08:00

- actor：axis6-lead
- change：按独立复核 REVIEW-C1-631.md（PASS，blocker 0 / normal 4 / info 3）收口：status 进行中→**被否证**（契约 §4 否证条件整体触发：任一 K 越带 + 提高 K 拉不回）、ended 落盘；按 info-3 把「K 形状」明确降级为「不当作精细可分结论（极差 0.006 ≪ CI 半宽 0.11），主判只看 K=1」；按 N3 补 K=1 连接点的**显式统计裁决**（与 6.3.2 `fill_bidir_r25` 的 CI 不相交 ⇒ 可区分）；按 N1/N2/N4 把 RESUME 未收尾、L3 未交付、AR 侧逐 block 量未复制记入 `gaps`。数字与结论不变。

## 2026-10-14T23:35:00+08:00

- actor：axis6-lead（Main 批准的规模标签更正）
- change：规模标签正名：预登记为「13M」，本项复用的 2671 同 harness 的**实测参数量 = 12,813,312**（d=256/L=12/h=8/mlp_mult=6）——仅正名、未改判据；并同步 `work_packages[WP1].means` 的旧标签。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
