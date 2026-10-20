# 2.1.2-skipping · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/2-depth/2.1.2-skipping/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/2-depth/2.1.2-skipping/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-13T10:37:02+08:00

- actor：轴 2 §2.1 worker
- change：按 MAPPING §1 由旧格 C2-layer-selection-order 并入 2.1.2 skipping（C2 → 2.1.2），整目录 git mv 两个实验目录进 experiments/；按 55-review C-P09 把「手段枚举」改成有目的/假设/判据的研究项，按 C-P18 不立「层序 shuffle」研究项（redo 归 2.1.3 repeating）。

## 2026-10-14T21:00:00+08:00

- actor：轴 2 深度轴 lead（ticket 61）
- change：C1 声明回填（不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，`falsification` 字段显式补写「用哪个带」（本项取 bpc 走 0.014 bpc 带（限配对 Δ、≥5 窗口）；跳过层占比 / 分布一致率等率类指标走配对 bootstrap / Wilson 95% CI）；此前本项引用噪声带却只在 `validation` 声明，`falsification` 无声明 ⇒ 按该节判据不可判。

## 2026-10-14T22:35:00+08:00

- actor：WP-2654 worker（ticket 61 §9.2）
- change：WP-2654 实装 + 预登记契约落盘（工厂替换 + 逐 token 恒等旁路 + 按臂增量落盘）；四臂训练因 GPU 锁未取到而未跑（该表述后由 23:10 条目按复核 B 的 B1 更正为「冒烟已跑、layerdrop-pt 活性红」。）

## 2026-10-14T23:10:00+08:00

- actor：轴 2 深度轴 lead（ticket 61 · 复核 B 处置 writer）
- change：按复核 B 的 blocking 2.1.2-B1 修正账实不符（不改任何测量数字、不改判据线、不改实现）：`2654/analysis.md` §0/§1/§3/§5 把「冒烟未执行」改为真实结论（2026-10-14T22:37:39 第 4 轮取到锁跑 20 步冒烟；`stock` / `skip-mid` / `layers6` 活性门过 lp = 8.0 / 6.0 / 6.0；`layerdrop-pt` 活性门红：实测 lp 8.0 ≠ 设计 7.2，`run_log.txt:2194`，断言 `run.py:451`；22:38:50 SMOKE FAILED，`run_log.txt:2195`）；`conclusion_review` 与 `gaps` 登记「实现未过活性门 + 资源未获得」，`gaps` 补等效算力账（每臂 98.3M token / 四臂 393.2M token）；WP1 的 `status` 维持「进行中」但推进状态改由 `gaps` 与本条说明（实装未过活性门 ⇒ 先修接线，再排 GPU；预登记 `acceptance` 文字未改）；`WP-2654-REPORT.md` 撤销「一次也没轮上」表述。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
