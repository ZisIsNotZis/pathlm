---
name: 2613-readout-reparam-dense-13m
ticket: 30-batch3（契约 F）
date: "2026-09-30"
grid: W4-readout-geometry
engine: dense-13m（train_depth_ar.py --dense）
budget: "gpu_wall_budget: 26 min（契约 F 原文；冒烟 ~1 + 两臂训练 ≈22 + 电池 ≈2）；峰值显存 ≤5GB（net）"
status: planned
---

# 2613 · 读出重参数化在普通 dense 模型上的可迁移性 · 13M

> 本文件即本实验的本地契约。判据线逐字抄自 `.scratch/30-batch3/issues/01-batch3.md`
> 「契约 F」，**未改**。原始结果：`results.json`；判读：`analysis.md`；
> 稳定性：`stability.md`；权重清单：`MODEL.md`。

## 0. 契约原文（逐字，不得修改）

> ## 契约 F：读出重参数化在普通 dense 模型上的可迁移性（结论收尾）
>
> - 目录：`axes/W-width/W4-readout-geometry/experiments/2613-readout-reparam-dense-13m/`
> - 文件所有权：只写本实验目录（复用既有 `--readout-linear-resid` 与 `--readout-gated` 开关）。
> - **干预**：在普通 dense 模型（无门、无退出、仅最终 CE）上，比较
>   `baseline`（tied 线性读出）vs `linear-resid`（h + P·(Q h)，等参数 2d²）。
>   （gated 臂可选，若预算允许则加第三臂。）
> - 动机：12k 快照下「读出重参数化」在 depth-ar 形态上给了 ≈0.08 的改善，但已被归因于
>   容量而非非线性。**若它在普通 dense 上也给 ≈0.08，则这是一个与我们的架构无关的通用
>   发现（可迁移）；若它只在 depth-ar 上有效，则说明它是「补足某种形态缺陷」。**
> - **判据（预登记）**：Δbpc 相对 baseline ≥ 0.014 ⇒ 可迁移（记为通用发现）；
>   |Δ| < 0.014 ⇒ 不可迁移（只在形态内有效）；Δ ≥ +0.014 ⇒ 负迁移。
>   同样必附两臂同方法 loss 斜率。
> - 成本：2 臂 × ≈13 min = ≈26 min。
> - 交付：同统一清单。

## 1. 命题

在**普通 dense 模型**（无门、无退出采样、仅最终深度 CE 的 stock decoder-only
形态）上，tied 线性读出前加一次**纯线性状态重参数化** `h' = h + P·(Q h)`
（新增 2·d² 参数，init Q~N(0,0.02)/P=0 单边零 ⇒ 严格恒等起步、非鞍点），
能否复现它在 depth-ar 形态上给出的 ≈0.08 clean-bpc 改善？

- 能（Δbpc_improvement ≥ 0.014）⇒ 与我们的架构形态无关的**通用发现**；
- 不能（|Δ| < 0.014）⇒ 它只是**补足 depth-ar 形态缺陷**，不是通用配方。

可证伪：Δ 是一个可测标量，判据是预登记的 0.014 线。

## 2. 单因素声明

**唯一变量 = 读出几何**（dense 形态的 tied 线性读出 vs tied 线性 + 等参数线性
重参数化）。两臂逐字段一致（逐字段对照表见 `analysis.md` §1 与
`results.json::config_equivalence`）：d=256 / L=8 / h=8 / seq=256 / mlp_mult=4 /
dropout=0 / norm_cap=1 / gate_bias_init=−2 / dense=True / topology=sequential /
fill_kv=True / proj_fill=False / readout_dims=None / exit_mode=uniform /
aux_weight=1.0 / final_weight=1.0 / shallow_stopgrad=False / dense_aux=False /
steps=12000 / batch_size=32 / lr=1e-3 / **seed=0** / train_tokens=98.3M /
vocab=206 / 数据切分（前 90M train、末 10M eval，词表来自全文件）。

基底 = 主流标准 decoder-only transformer + RAP 共享底座三条偏离
（`docs/design.md` §2）。**非必要性例外**：无——读出重参数化在 stock dense 形态上
可独立生效，不引入第二个非标准因素。

## 3. 指标表（逐项分析）

| 维度 | 指标 | 与谁比 |
|---|---|---|
| 解决什么问题 | 读出重参数化增益的**可迁移性** | depth-ar 形态（2611，Δ=−0.0888） |
| 代价 | Δ clean bpc（13M，同 token 预算，12k matched-step 快照） | baseline（tied 线性） |
| 有效性 | Δbpc_improvement 与 0.014 线的关系 | 预登记判据线 |
| 收益·用法 1（质量） | 相对同尺寸 dense 基线的收回幅度 | baseline 本体（税分母 = baseline） |
| 收益·用法 2（自由度） | transform 相对幅度 `||P(Qh)||/||h||` 逐深度 | 0（无变换） |
| 可控性 | prob0 ECE（最终深度） | baseline |
| 副产品 | tf@k（dense 为 proxy 口径，非训练出口资产） | baseline |
| **干预活性（必报）** | P/Q fro 范数、absmax、非零占比、transform 相对幅度 | 0（死干预） |
| **收敛对账（必报）** | 两臂**同方法**（末 25% 同一窗口、同一最小二乘）loss 斜率 | baseline |
| **稳定性（必报）** | gnorm max/中位数/尖峰、NaN、loss 突跳、逐深度 ECE | baseline |

**判读规则**：先过恒等起步门、参数账、干预活性门；不过先修实现再判机制。

## 4. 判据线（预先写下，逐字来自契约 F，不得修改）

定义 `Δbpc_improvement = bpc_baseline − bpc_linres`（正 = 改善；等价于契约里的
「Δbpc 相对 baseline 的改善量」）：

- **Δ ≥ +0.014 ⇒ 可迁移**（与我们的架构无关的通用发现）；
- **|Δ| < 0.014 ⇒ 不可迁移**（只在 depth-ar 形态内有效）；
- **Δ ≤ −0.014 ⇒ 负迁移**（dense 上更差）。

**必附**：两臂同方法（末 25% 同一窗口、同一最小二乘算法）loss 斜率，作为
「是否只是收敛加速」的收敛对账（SCHEMA §16）。若报增益，须补 2× 更长训练点。

方差界参照：Δ < 0.014 不可排序（`docs/methodology.md` §2）。

## 5. 对照与方差

- **baseline** = 同 seed 全新训练的 dense tied 线性臂（自包含双跑，非复用历史 run）；
- **linres** = 同 seed 全新训练的 dense + `--readout-linear-resid` 臂；
- 两臂**同一脚本、同一评测代码**下重测同一电池（口径见 §7）；历史 23d-dense-B-12k
  checkpoint 另作**引擎复现门**（`results.json::historical_reproduction`）；
- **seed 数**：1 seed（契约未授权补 seed），两臂同 seed=0；
- 方差界：Δ < 0.014 不可排序。

## 6. 成本与 scope

| 项 | 值 |
|---|---|
| `gpu_wall_budget` | **26 min**（契约 F；冒烟 ~1 + 两臂训练 + 电池） |
| 最小可判定版本 | 13M / 12k 步 / 1 seed / 只测 clean bpc + 同方法斜率 + 活性 + ECE + tf proxy |
| 峰值显存 | ≤5GB net（实测两臂 torch 峰值分配 ≈1.0–1.2GB，冒烟 `smoke/smoke.json`） |
| 可中断性 | 是（每 200 步落 log；中断可按 checkpoint 续） |
| 冒烟计划 | `smoke.py`：全尺寸恒等起步（逐位）+ 参数量账 + 20 步活性 + 步时/显存 |
| 预算不够时砍什么 | **砍可选第三臂 gated**、砍 eval-batches；**不砍判据线** |

规模纪律（SCHEMA §7）：单卡 RTX 4090 24GB 且被多任务共享，只做 13M 原理验证，
不外推规模/渐近。

## 7. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | char 级 enwik8（`data/enwik8_full.npz`，本地已有，不下载；前 90M train / 末 10M eval / 词表 206） |
| 入口脚本 | `run.py`（训练 + 电池 + 对账 + 判读）；冒烟 `smoke.py`；变异 `mutate.py`；单测 `test_dense_readout_reparam.py` |
| 代码改动 | **无引擎改动**（复用既有 `--readout-linear-resid` / `--readout-gated`；本实验只写本目录） |
| 可失败单测 | 本目录 `test_dense_readout_reparam.py`（5 条）：dense 关开关逐位还原 v1 / 恒等起步与单因素 state_dict / dense 精确公式与活性 / 参数量 2d² / dense 仅最终 CE 监督 + 一步逃离鞍点 |
| 变异验证 | `mutate.py` 四处故意破坏（忽略开关 / 退回零初始 / dense 全深度监督 / 公式多做一次）必须抓红——见 `mutation_report.json` 与 `analysis.md` §5 |
| 落盘 | `results.json`（seed 0）/ `results-seed1.json`（seed 1 复现）/ `run_log.txt` / `run_seed1.log` / `{arm}-s{seed}-train_log.jsonl` / `{arm}-s{seed}-gate_log.jsonl` / `config.json` / `analysis.md` / `stability.md` / `stability.json` / `MODEL.md` / `mutation_report.json` / `smoke/smoke.json` |

## 8. 失败决策树

| 症状 | 诊断顺序 | 处置 |
|---|---|---|
| 起步非恒等 / 参数量 ≠ 2d² | ① 实现（单测+mutation） | 先修实现再判机制 |
| 干预死（P/Q/transform ≈ 0） | ① 实现（鞍点） | 检查 init 是否退回双零；活性门必须非零 |
| \|Δ\| < 0.014 | ② 无头寸/不可迁移 | 报「不可迁移」：读出重参数化只在 depth-ar 形态内有效（如实写，不叠第二机制） |
| Δ 落在 [0.013, 0.015) 边界 | ③ 方差边界 | 报「落在方差界附近」，不裁决，附 n=1 声明 |
| Δ ≥ +0.014 可迁移 | — | 报通用发现，并**补 2×（24k）两臂**做收敛对账（SCHEMA §16） |
| 结果反向（dense 上变差 > 0.014） | ④ 设计 | 报「负迁移」，写清 dense 与 depth-ar 的读出路径差异 |
| 不可重复 | ⑤ 方差 | 补 seed；仍不可排序则撤回 |

## 9. 产出

`axes/W-width/W4-readout-geometry/experiments/2613-readout-reparam-dense-13m/`
下的 `README.md`(本文件) / `run.py` / `smoke.py` / `mutate.py` / `stability.py` /
`test_dense_readout_reparam.py` / `config.json` / `results.json` /
`results-seed1.json` / `analysis.md` / `stability.md` / `stability.json` /
`MODEL.md` / `mutation_report.json` / `smoke/` / `runs/`（权重不入库，见 `MODEL.md`）。

> 主 Agent 补注（复核 P2#2/#3）：契约原文在 Δ 符号上自冲突（同一段既写「Δbpc ≥0.014 可迁移」又写「Δ ≥ +0.014 负迁移」）；本处统一按 **Δ = bpc_baseline − bpc_arm**（正=改善）执行，两种读法下本实验结论相同（实测 |Δ| = 0.0055）。实测 GPU 墙钟 ≈27.5 min（含 seed1 复现，契约预算 26 min，+6%，属 n=1 负结果必须复现的硬纪律）。
