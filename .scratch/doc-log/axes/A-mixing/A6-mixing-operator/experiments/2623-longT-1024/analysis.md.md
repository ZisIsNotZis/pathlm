# 过程史 · 自 `axes/A-mixing/A6-mixing-operator/experiments/2623-longT-1024/analysis.md` 迁出

> 性质：日志文档。原文是「文档变更记录」（P2 修正记录）与「结论降级」标注，SSOT 化时从事实
> 文档删除，只保留当前结论。事实以 `analysis.md` / `results.json` 为准。

- §3 表格下方原有结论降级标注（去变更化，保留判读）：
  `**⚠️ 判读（主 Agent 裁决 2026-10-03：结论降级）**：`

- §11 原有 K10 更正尾注（删）：「（X3 已更正的是「ticket 曾把 T=1024 的 +2.8% 误当 T=512」）」

- 文末原有整节 P2 变更记录（删）：

```
## 12. P2 修正记录（后于复核）

> **来源**：ticket 43 V3「P2 收尾」，修正对象 = `.scratch/42-longt-stability/evidence/REVIEW_W3.md`
> 的 report-only P2（Finding 3 / Finding 5）。**只改文档/注释/引用并补原始日志入库；
> `results.json`、`reconcile_report.json`、`mutation_report.json` 与全部数字结论、band 分类、
> 判据线一字未动。**

| # | 文件:位置 | 改了什么 | 改成什么 | 为什么不影响结论 |
|---|---|---|---|---|
| 1 | `run.py`（模块 docstring，臂清单） | `bs32，seed0` / `3000 步（98.3M token）` / `3000 步（长 T 第三点主臂）` | `bs8，seed0`（附 §20.2 降 batch 说明）/ `12000 步` / `12000 步` | 陈旧描述（沿用了 bs32 计划）；实际执行 = **bs8 × 12000 = 98.3M token**（`config.json`、`train_all_log.txt`、`analysis.md §1` 表、`README.md §6` 一致）。§0 契约未钉步数；token 预算仍 98.3M，唯一变量（窗口）与结论不变 |
| 2 | `README.md` §1（命题起句） | `3000 步 / 98.3M token` | `12000 步 / 98.3M token` | 同 #1（同一陈旧步数的另一处）；token 预算与判据线未动 |
| 3 | `README.md` §7（落盘清单） | `<arm>-train_log.jsonl` / `<arm>-gate_log.jsonl`（原在 gitignore 的 `runs/` 下、未入库） | 同名文件 **入库到实验目录根**，并注明「逐 200 步原始日志，已入库」 | 只补证据、不改数字；使 `analysis.md §6` 引用的「最大采样 gnorm 128.85」可从**干净 checkout** 复核（见下） |
| 4 | 新增入库文件 | — | `full-1024-train_log.jsonl`、`full-1024-gate_log.jsonl`、`win512-1024-train_log.jsonl`、`win512-1024-gate_log.jsonl`（`runs/<arm>/` 的逐字节副本，sha256 逐对相等） | 原始证据（每 200 步一行）；`max sampled gnorm full = 128.85 @ step 9800` 现可直接复算。数值与 `results.json::convergence_and_stability` 的 `gnorm_max=1740.92`（逐步统计，非 200 步采样）不矛盾 |
| 5 | `analysis.md` 抬头 / §6 | 指向 `runs/<arm>/train_log.jsonl` | 指向已入库的 `<arm>-train_log.jsonl` | 路径修正，不改读数 |
| 6 | `analysis.md` 摘要·一句话 / §6·工程债线索 / §10 第 1 条与第 5 条未测项 | 「seed=1 双臂同配方重跑」列为**未测下一步**（原 5 处口径见 REVIEW_W3 Finding 5） | 补引 **Z1 `2624-longT-1024-seed1`** 已完成：`full` 臂 gnorm_max **29578.57**（seed0 1740.92，×17.0）、跳步 3、爆炸点 ~step 6200 ⇒ 爆炸**非 seed 特异、复现且更猛**，账本记该点「永久不可归因」 | 纯溯源引用；**本点判读不变**（Δ=+0.0234 仍只作「不可判」，band 分类照字面保留，判据线未动） |

**口径交底（防误读）**：`analysis.md §6` 的 **128.85** = 逐 200 步**采样**的最大 gnorm；
`results.json` 的 **1740.92** = 训练器逐步统计的 `gnorm_max`（两臂逐字段一致，`grad_skips=0`）。
两者均指向同一「full 臂后半段梯度尖峰」现象，不构成数字冲突；本实验的判读依据仍是 `results.json` 的 1740.92。
```

## 追加（本次 SSOT 补扫：已废台账指代）

- §摘要原有：`⇒ 账本记该点「永久不可归因」` → `⇒ catalog.yaml 记该点「永久不可归因」`。
- §11 原有：`账本据此把该点记为「永久不可归因」` → `catalog.yaml 据此把该点记为「永久不可归因」`。
