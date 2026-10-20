# 1.1.1-hard-window · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/1-time/1.1.1-hard-window/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/1-time/1.1.1-hard-window/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T09:40:25+08:00

- actor：WP-M1.0 foundation
- change：由旧格 A3-window-anchor 拆分建立（A3 → 1.1.1 hard 窗口 + 1.1.3 可见量上限）；内容按 55-review A-PROBLEMS P9/P14/P21 修正。

## 2026-10-13T10:15:48+08:00

- actor：轴 1 §1.1 worker
- change：迁入旧格证据：A3-window-anchor/experiments/** 与 A6-mixing-operator 的长 T 窗口系列（2622/2623/2624/2625/2626/2628/2629/2636）整目录搬入 experiments/；experiments 路径与锚点改为新家；补 WP4（长 T 同 step 曲线）；按 55-review A-P10/P11/P12/P14/P18/P19 修正目的与判据口径（窗口是手段、代价进前沿、原「长 T」命名歧义已改为「窗口代价随 T」）。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T20:51:00+08:00

- actor：axis1-lead-3
- change：按 `docs/research-schema.md` §12.1 的声明义务补 **`falsification` 字段**的判据带声明（ticket 62 §10.3 F11；此前 C1 回填只落在 `validation`，`falsification` 仍缺声明 ⇒ 判据不可判）：bpc 类走 0.014 bpc 带（限配对 Δ、≥5 窗口）、召回等率类走配对 bootstrap / Wilson 95% CI、不跨量纲借用 0.014；不改数字、不改结论、不改判据线。

## 2026-10-14T22:48:10+08:00

- actor：axis1-lead-4
- change：补记结构性依赖（ticket 62 §10.22）：本实验评测要求预登记的锚点臂 **两 seed 都在场**（4 臂配置实测 `run.py:394 KeyError: win16a_s1`；另修好 `make_config --require-new` 在缺臂时不落盘已训臂的缺陷）⇒ 下一步改为「①补训 `win16a_s1`（1 臂 × 98.3M token）→ ②回填 config → ③跑评测」，已落成一个后台 driver。不改数字、不改判据线。

## 2026-10-14T23:17:03+08:00

- actor：axis1-lead-4
- change：落地本轮两个 GPU 实验后的判读（WP-R2，实验 2662 rc=0）：**status 仍为进行中**（主正面主张「锚点召回」在 n=2 下不可判、WP1/WP3 未完成、KV 消融保留曲线未在本目录复算）——不改数字、不改判据线。改写 conclusion：用仓内自包含读数取代旧「锚点通道是真产物 needle 76.8→93.3」表述（那是另一台仪器的绝对召回跳变、可归因于距离惩罚而非锚点槽）；补 failure_analysis（三分归因：设计问题=窗口/锚点按定义只改可见集；结构问题=sink 与锚点在本仓是同一根杠杆；innovation 不适用）；experiments 加 2662 目录；gaps 换成本轮实测的三条下一步（补 seed、评测期锚点消融、补 MODEL.md）；WP1/WP3 进行中、WP4 已完成（2636 的验收项已满足）。

## 2026-10-14T23:41:09+08:00

- actor：axis1-lead-4
- change：按 REVIEW-9 的 P2 补 `2662/MODEL.md`（权重清单 + sha256 + 脚本指纹 + 口径提示；数字取自 `results.json`，权重 *.pt 仍不入库）⇒ 该交付缺口关闭，相应 gaps 条目删除。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
