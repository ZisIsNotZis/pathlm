# 6.3.2-pad-fill · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/6-predict/6.3.2-pad-fill/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/6-predict/6.3.2-pad-fill/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:47:00+08:00

- actor：轴 6 §6.3 worker
- change：新建研究项（无旧格，MAPPING §2）：一次性批量 pad 填充（填空式）归 6.3 整段并行生成（不止 AR）；按 docs/research-schema.md 预登记目的 / 基线 / 判据 / 否证条件，并把「未填位置因果性」写成必须不变的硬门以防未来泄漏。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T20:52:40+08:00

- actor：axis6-lead（ticket 63 续 · C1 声明回填第二轮）
- change：C1 声明回填第二轮（不改数字 / 不改结论 / 不改证据）：按 docs/research-schema.md §12.1，本项 milestone.acceptance 与 objective_criteria 引用 0.014 噪声带但未在同字段声明用哪个带，现补声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），tokens-per-forward 等率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用。

## 2026-10-14T23:05:29+08:00

- actor：axis6-lead
- change：Phase C 收口（WP1 = `experiments/2671-padfill-13m`，实验目录内自带 harness）：conclusion 首次填入（(a) 填充质量代价 +1.07…+2.37 bpc、CI 不含 0 且远越 0.014 带；(b) r 形状不单调 ⇒ 预登记否证条件的「单调恶化」子条件未满足、按臂分判；(c) 可见性独立增益 +0.1635；(d) 不受未来泄漏硬门通过 + gate probe 可失败；(e) 代价轴；(f) 未测清单含 L5 确定性未验证）；failure_analysis 填入（设计/结构归因，非 baseline 错）；WP1 status 未开始→已完成；experiments 增 2671；milestone 规模标签按实测更正为 12,813,312；gaps 换成剩余项（r 非单调原因、L5 确定性、快照口径）；status 未开始→已完成、ended 落盘。

## 2026-10-14T23:12:00+08:00

- actor：axis6-lead
- change：按最终复核 REVIEW-C2-FINAL.md 的 N1 与 info-a 修：`conclusion_review` 与 WP1.review 的复核指针从悬空的 `REVIEW-C2-632.md`（本轴收尾脚本写入时用错文件名，实际报告是 C2-FINAL）改为实际报告路径；conclusion (a) 的收敛档参照补出处（6.2/2610 clean bpc 1.4836）。不改任何数字与结论。

## 2026-10-14T23:20:00+08:00

- actor：axis6-lead（Main 批准的规模标签更正）
- change：规模标签正名：预登记为「13M」，本仓 2671 的**实测参数量 = 12,813,312**（d=256/L=12/h=8/mlp_mult=6；两臂逐位相同）——仅正名、未改判据。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
