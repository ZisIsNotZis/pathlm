# 6.3.3-latent-regression · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/6-predict/6.3.3-latent-regression/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/6-predict/6.3.3-latent-regression/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-14T10:02:06+08:00

- actor：axis6-lead
- change：WP1 最小判决执行并收口（Phase B）：实验 `experiments/2660-latentreg-vs-disc-13m` 入库；conclusion 落地（a）–（e）：L0 实现保真（λ=0 两臂逐位相同）、路径 (i) 命中率 +0.031（λ=1.0，CI 不含 0，两主干点/两种子/k∈{2,4} 同向）、路径 (ii) 草稿接受率 Δα₁ +0.0916 / Δτ +0.1464（CI 不含 0，保真硬门 = repro_direct1 与 2640 direct1 逐位相同）、机制限定（共享 tied 读出下两臂同参数化，独立自由度只是监督目标 ⇒ 增益应表述为「latent 匹配监督」而非「回归机制」）；gaps 换成剩余项（free-run bpc 未测 / 单一规模 / 边界未排开 / λ 网格只正向 / 与主干内联合训练头的 regime 差）；WP1 status 未开始→已完成；status 未开始→进行中（milestone 输出包的序列级 bpc 未齐）。

## 2026-10-13T10:47:00+08:00

- actor：轴 6 §6.3 worker
- change：新建研究项（无旧格，MAPPING §2）：latent regression 批量生成（估计得贪心）归 6.3 整段并行生成（不止 AR）；按 docs/research-schema.md 预登记目的 / 基线 / 判据 / 否证条件，并把「同参数量离散跳头对照 + 读出共享」写死以防把容量差异记成机制增益。

## 2026-10-14T06:44:03+08:00

- actor：framework-adjudication-writer（ticket 57）
- change：落地 63-F4 裁决（不改数字）：`method`/`milestone`/`validation`/`objective_criteria`/`falsification`/`cost_model` 写明 (i) 直接发 与 (ii) 当草稿再验证 两条路径各作一臂、分开报代价（前者无接受率、每 token 主干 FLOPs ≈ L/k；后者有接受率、≈ (1+k)/k×L ≈ L）。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T20:52:40+08:00

- actor：axis6-lead（ticket 63 续 · C1 声明回填第二轮）
- change：C1 声明回填第二轮（不改数字 / 不改结论 / 不改证据）：按 docs/research-schema.md §12.1，本项 objective_criteria 把「草稿接受率」与 bpc 并列写在 0.014 噪声带下（跨量纲借用）、milestone.acceptance 把 τ / tokens-per-forward 差写成「越 0.014 噪声带」，现改为——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口）；τ / tokens-per-forward / 草稿接受率等率类指标走 prompt 聚类配对 bootstrap / Wilson 95% CI、不跨量纲借用。

## 2026-10-14T21:42:13+08:00

- actor：axis6-lead
- change：Phase C 收口（WP2 `2670-freerun-bpc-13m`）：conclusion 新增 (f) 路径 (i) 的 free-run 序列级 bpc（自评 Δ = +0.74/+0.92 bpc；外部真值锚 Δ = +4.15/+5.07，同向且更大；候选臂流明显退化）； failure_analysis 首次填入（路径 (i) 手段不成立；设计问题为主 + 结构问题次之）；WP2 条目新增（含 REVIEW-C1-2670 判词）；experiments 增 2670；gaps 换成剩余项（k=4/温度采样未测、真值锚同族偏差、单一规模）；status 进行中→已完成、ended 落盘。

## 2026-10-14T22:30:00+08:00

- actor：axis6-lead
- change：按复核 #C2（`.scratch/63-axis6-predict/REVIEW-C2-633.md`，VERDICT FAIL、缺陷均为文档一致性类）返工：B1 去掉 conclusion (e) 里与 (f)/gaps 矛盾的「free-run 序列级 bpc 未测」分句；N2 真值锚 Δ 下界改为 +1.1701；N3 退化统计改为与同句同格式的均值区间 99.8–153.8；I4 真值锚数字只留 conclusion (f) 一处、conclusion_review 改指针式；I3 roadmap 去掉完成日期；I5 标注执行清单属 C1 轮。数字与结论均未变。

## 2026-10-14T23:12:00+08:00

- actor：axis6-lead
- change：按最终复核 REVIEW-C2-FINAL.md 的判词更新 `conclusion_review`：当前（最后一轮）判词由 C1 轮改为「#C2 FAIL → 返工 → C2-FINAL PASS」，并写明返工项已被核验到位、未引入新矛盾。不改数字与结论。

## 2026-10-14T23:20:00+08:00

- actor：axis6-lead（Main 裁决 B5.4-2）
- change：`innovation` 按实际机制改写（Main 批准）：由「回归到连续 latent 再贪心」改为「latent 匹配监督（把 z_j 拉向主干未来真值隐状态）」，并写明「在共享 tied 读出 + 同参数量下与离散分类头数学等价、同种子逐位相同」这一本仓可核的等价性（依据 2660 的 L0）。仅改写机制描述，数字 / 结论 / 判据未变。

## 2026-10-14T23:30:00+08:00

- actor：axis6-lead（Main 裁决 B5.4-2 的一致性补齐）
- change：`gap` 里「用同参数量对照把「回归到 latent」与「直接分类」的增益分开」一句与 `innovation` 的机制限定对齐（§11.2 兄弟字段一致性）：补写「本仓实测下两臂数学等价、真正分开的是监督目标」。仅措辞与限定，数字 / 结论 / 判据未变。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
