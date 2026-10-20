# 1.2.4-dynamic-merge · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/1-time/1.2.4-dynamic-merge/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/1-time/1.2.4-dynamic-merge/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:29:52+08:00

- actor：轴 1 §1.2 worker
- change：新建研究项（无旧格，MAPPING §2）：动态合并：越往下层 token 越少 归 1.2 可见结构；对应 PROPOSED-STRUCTURE §4 的 T.4 缺口；判据钉在「同每 token 等效总 FLOPs」预算上，以区分「合并已处理位置」与「减少可见量」。

## 2026-10-14T20:51:00+08:00

- actor：axis1-lead-3
- change：按 `docs/research-schema.md` §12.1 的声明义务补 **`falsification` 字段**的判据带声明（ticket 62 §10.3 F11；此前 C1 回填只落在 `validation`，`falsification` 仍缺声明 ⇒ 判据不可判）：bpc 类走 0.014 bpc 带（限配对 Δ、≥5 窗口）、召回等率类走配对 bootstrap / Wilson 95% CI、不跨量纲借用 0.014；不改数字、不改结论、不改判据线。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
