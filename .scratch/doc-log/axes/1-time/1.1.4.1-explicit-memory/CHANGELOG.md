# 1.1.4.1-explicit-memory · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/1-time/1.1.4.1-explicit-memory/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/1-time/1.1.4.1-explicit-memory/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:15:48+08:00

- actor：轴 1 §1.1 worker
- change：由旧格 M1-carrier-type 迁移建立（M1 → 1.1.4.1 显式记忆槽 / 线性状态 / 递归状态）；23 个实验目录整目录搬入 experiments/；按 55-review M-01/M-02/M-03/M-12 把目的改为「信息右移 + 质量-代价前沿」；status 由 judged 改为进行中（M-22）；按 M-04/M-05/M-06/M-08/M-09 把退化实现、缺业界对照、错误归因列入 gaps 与 WP；标题改用 docs/structure.md §12 的 exact wording（含 delta 规则族）。

## 2026-10-13T15:35:09+08:00

- actor：axis1-lead
- change：Phase A WP1（实验 2638-carry-frontier-13m）判：显式状态点被同预算纯窗口 win17 严格支配（内存更少、bpc 更好）；status 由进行中改为被否证（本项预登记否证条件 #2 触发），补 failure_analysis（设计/实现为主因、结构次因、非 innovation 失败）、重写 conclusion/conclusion_review、更新 gaps、experiments 新增 2638、WP1 status→已完成；role_map 实名。

## 2026-10-13T15:48:43+08:00

- actor：axis1-lead
- change：按 REVIEW-1（fresh 复核，判 FAIL/科学结论 PASS）修 B1–B4：B1 判据③ 判定线方向文字改为与自身公式及 SCHEMA §17.3 一致、B2 修 analysis 引数 0.005026、B3 补 win17/18 追加登记与预算超支记账、B4 把「假设被否证」与「milestone 交付范围被裁剪（交付未达标）」拆开记入 conclusion/failure_analysis④/gaps 首条与 WP1.acceptance；另修 N1（results.json 补 liveness 块）、N2（非退化硬断言 + 评测期掩码反事实探针）、N4（引用路径 smk24）、N7（conclusion 补 T=256；conclusion_review 回填复核判词）与 N8（实验目录内自包含单测 test_2638_frontier.py，不改共享 tests/）。

## 2026-10-13T15:58:00+08:00

- actor：axis1-lead
- change：按 REVIEW-2 修 R2-B1/B2 + N1–N8（科学结论不变、无需重跑 GPU）：MODEL.md 的 run.py 指纹按实际值重生成；README §7 训练清单补 5 臂；删事实文档里的变更叙事；WP1.means 加指向 gaps 的半句；判据④ 标注 seed1 跨 seed 口径；单测改实质断言；证据锚点前移到 round-2 落盘 commit @443cb44。

## 2026-10-13T16:10:00+08:00

- actor：axis1-lead
- change：按 REVIEW-3 修 R3-B1 + N1–N6（科学结论不变、无需重跑 GPU）：conclusion_review 回填 REVIEW-2/REVIEW-3 判词（消除「REVIEW-2 待派」与 change_log 的矛盾）；README 单测条数 5→6、悬空指针改指 results.json/2634、§4a 纯窗口曲线补 win17/18；单测 docstring 收窄钉子范围；analysis 去重复行；change_log 改时间正序。

## 2026-10-14T06:44:03+08:00

- actor：framework-adjudication-writer（ticket 57）
- change：落地跨轴裁决（`FRAMEWORK-DECISIONS.md` C1/C2/62-F2/62-F3，不改数字、不删证据）：`validation`/`falsification` 补「用哪个带」声明（本格自测带 σ_cell＝0.00166 为主判、跨格 0.014 bpc 带为参考，两带不得混用，C1）；`milestone.acceptance` 与 `conclusion` 补「字面成立 / 实质否证」并列读法（C2）；`method` 写死代价主轴 = 每位置等效 KV 字节、FLOPs 副轴（62-F2）；`gaps` 记「形态税不作为判据项」（62-F3）。

## 2026-10-14T09:00:00+08:00

- actor：axis1-lead-2
- change：第二轮（完整机制重测）：引擎侧 E1 交付逐层独立状态 + 内容写门 + 可选 γ（附加式、默认逐位不变、独立复核 PASS），本轴用同一配方训练 3 个新臂（pl16 / c16 chunk 匹配对照 / pl16d γ=0.99）并重画前沿（实验 2639）。结果：完备机制在等字节（393,216 B）上 1.527595 vs win32 1.595823（+0.068228）、在下包络上、补回窗口→全量差距的 80.1%；单共享槽实现仍被 win17 支配 ⇒ 上一轮的否证归因确实落在「该实现」。据此：status 由被否证改为进行中（milestone 两轴与业界对齐仍未交付）、failure_analysis 置空（未判失败，schema 未判失败时写空）、重写 conclusion/conclusion_review、重写 gaps（新增归因不可分/召回轴/等参数/多 scale/γ 扫/长上下文化）、experiments 新增 2639、WP1 补 2639 evidence、WP2 改为进行中（引擎+质量读数已交付，业界基线对照未做）、roadmap 更新；ended 清空。口径：主判据用等字节直接点比（不依赖插值）；敏感性（chunk、换评测窗口序、摊平口径）已在 results.json 并列报出。

## 2026-10-14T20:51:00+08:00

- actor：axis1-lead-3
- change：框架项 F9 默认处置（ticket 62 §10.3）：把本项 `milestone.acceptance` 与 `conclusion` 里 C2 落地文本的「同算力」措辞改为「同预算点（每位置等效字节）」，与判据体（`method` / `objective_criteria` / 实验 README §4③）统一到 62-F2 裁决的代价主轴；纯窗口与线性状态的每 token FLOPs 本来就不同，「同算力」是口径模糊的简写。不改数字、不改判据线、不改预登记判据本体，只改称呼。

## 2026-10-14T23:17:46+08:00

- actor：axis1-lead-4
- change：落地本轮 2661 的召回轴读数（WP-R1）：**status 仍为进行中**（milestone 字面还差「每预算点两轴 + 量化点同图 + 与业界实现对齐」，且完备臂缺种子重复）——不改数字、不改判据线。`conclusion` 追加召回轴段落（同索引硬子集 10.226%、对撞 +9.522pp、`Δbpc_far` +0.032965 且关通道归零、负对照停地板）；`experiments` 加 2661 目录；`gaps` 把「排队失败」那条换成本轮实测的两条下一步；`conclusion_review` 记本轮新增证据与其复核落点（§10.26）。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
