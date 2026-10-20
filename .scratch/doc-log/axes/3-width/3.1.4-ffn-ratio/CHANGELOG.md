# 3.1.4-ffn-ratio · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/3-width/3.1.4-ffn-ratio/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/3-width/3.1.4-ffn-ratio/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:57:34+08:00

- actor：轴 3 §3.1 worker
- change：新建研究项（无旧格，MAPPING §2）：前馈隐藏比 归 3.1 每 token 的激活宽度；按 docs/research-schema.md 预登记目的/基线/判据/否证条件，并把「d_model/层数不变」与「与 3.1.2 的边界」写死。

## 2026-10-13T18:02:28+08:00

- actor：axis3-lead
- change：Phase A 逐项复核：role_map 实名；把「等 FFN 预算」位点写死（层间臂必须与等宽臂对齐 Σ_l r_l ⇒ FFN 参数与每 token FFN 等效算力逐位相等，剪枝类按 SCHEMA.md §15 例外）；falsification 补 H0 判据；登记 2641 实验并把 WP1 置进行中。

## 2026-10-13T19:2x+08:00

- actor：axis3-lead
- change：第 1 轮独立复核（fresh、只读；revision e6d2db1）判 FAIL（窄范围）：B1–B3 三处数字/等式与落盘证据不符 + N1–N7 措辞口径类。已按复核者最小修逐条改正 analysis.md / MODEL.md（纯文字数字、未重训重评），并按复核者指出、不可由现有证据排除的两条反假设（共享前缀初始化的通道子集混杂；PYR-s1 的 |Δ|=0.0139 贴种子散布上界 0.0138）把 conclusion/failure_analysis/gaps 的措辞弱化并加限定；预登记契约 README.md 保持冻结不改，同名不同值的口径差异改在 analysis.md §5.2 说明。第 2 轮复核因时间盒未派 ⇒ 复核门未闭合（如实登记）。

## 2026-10-13T19:09:16+08:00

- actor：axis3-lead
- change：WP1 执行完成（2641：U4/U2/PYR/APYR × seed 0/1 共 7 臂 × 12k matched-step，按 §20 逐臂取放锁）：填 conclusion / failure_analysis（设计问题主因 + 结构问题次因，innovation 失败不适用）、status 置已完成、登记 ended 与 WP2，并把非单调 schedule、早退副产品、U4 单 seed 等未测项写进 gaps。

## 2026-10-14T06:44:03+08:00

- actor：framework-adjudication-writer（ticket 57）
- change：落地跨轴裁决（`FRAMEWORK-DECISIONS.md` C1/C2/64-F1/64-F3，不改数字、不删证据）：`validation`/`falsification` 补「用哪个带」声明（0.014 bpc 带限配对 Δ、≥5 窗口；同臂换 seed 散列只作参考、不作 |Δ| 上界，C1）；`milestone.acceptance` 与 `conclusion` 补「字面成立 / 实质成立或否证」并列读法（C2）；`validation` 写死等 FFN 预算/等算力位点标准对照（64-F1）与部分臂允许 n=1（64-F3）。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T21:35:00+08:00

- actor：axis3-lead
- change：复核门闭合：第 2 轮（builtin reviewer，无 shell）判 FAIL（窄范围：6 项执行类检查跑不了）；据此改派有 shell 的 fresh 复核者做第 2B 轮，判 **PASS**（REVIEW-2B.md：pytest 13 passed、Δ 重算逐位一致、B1–B3 已改、T5 初始化控制 6 组全 True、T6 契约 sha256 与 mtime 成立、T9 越界 0 且 summary.py --check OK）。`conclusion_review` 按 docs/research-schema.md §11.1 只写当前（最后一轮）判词 = PASS；历轮判词留在 .scratch/64-axis3-width/REVIEW-*.md 与 TICKET.md §9/§10.8。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
