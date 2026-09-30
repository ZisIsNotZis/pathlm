---
name: 2611-equalparams-linear-control-13m
ticket: 29-verify-batch（契约 B）
date: "2026-09-30"
grid: W4-readout-geometry
engine: depth-ar（13M 档）
budget: "gpu_wall_budget: 18 min（冒烟 ~1 + 训练 ≈13 + 三臂电池 ≈2 + 余量）；峰值显存 ≤5GB（net）"
status: planned
---

# 2611 · 读出几何的等参数纯线性对照 · 13M · depth-ar

> 本文件即本实验的本地契约（判据线逐字抄自 `.scratch/29-verify-batch/issues/01-verify-batch.md`
> 「契约 B」，**未改**）。原始结果：`results.json`；判读：`analysis.md`；稳定性：`stability.md`。

## 1. 命题

2610 的 gated 残差读出相对 tied 线性读出改善 clean bpc **−0.0812**。
本实验问：这个增益是**参数量**（2·d² 的新增容量）带来的，还是**非线性**本身带来的？

判决臂 = **参数量与 gated 臂完全相同、但功能上纯线性**的对照：读出前做
`h' = h + P·(Q h)`（P、Q 各 d×d，无偏置；整体等价于一个线性映射 `(I + PQ) h`），
再走 tied 线性读出 `logits = h'·Eᵀ`。

- init：**Q ~ N(0, 0.02)、P = 0**（单边零）⇒ 起步 `transform = P(Qh) = 0`，
  与基底 A 臂**逐位相同**（严格恒等起步，单测钉住）；
- `∂L/∂P = grad ⊗ (Qh) ≠ 0` ⇒ **非鞍点**（与 gated 臂同款单边零；
  `P = Q = 0` 是鞍点，禁止——2610 已实测零初始 12k 步不逃逸）；
- 参数量 = 2·d² = 131,072，与 gated 臂**一字不差**。

## 2. 单因素声明

**唯一变量 = 读出几何**（gated 非线性 vs 等参数纯线性）。其余与 2610 gated 臂
**逐字段一致**（逐字段对照表见 `analysis.md` §1 与 `results.json::*_equivalence`）：
d=256 / L=8 / h=8 / seq=256 / mlp_mult=4 / dropout=0 / norm_cap=1 /
gate_bias_init=−2 / dense=False / exit_mode=uniform / fill_kv=True / proj_fill=False /
readout_dims=None / aux_weight=1.0 / final_weight=1.0 / shallow_stopgrad=False /
dense_aux=False / steps=12000 / batch_size=32 / lr=1e-3 / **seed=0** /
train_tokens=98.3M / vocab=206 / 数据切分（前 90M train、末 10M eval，词表来自全文件）。

基底 = RAP 共享底座（三条必要偏离见 `docs/design.md` §2），不引入第二个非标准因素。
**非必要性例外**：无。A 臂与 gated 臂按 SCHEMA §9 复用历史/既有 run，附
「实现层逐位等值（关闭开关单测）+ 配置层逐字段对照 + 口径标注」三道门。

## 3. 指标表（逐项分析）

| 维度 | 指标 | 与谁比 |
|---|---|---|
| 解决什么问题 | gated 增益的归因（参数量 vs 非线性） | 2610 gated 臂 |
| 代价 | Δ clean bpc（13M，同 token 预算，12k matched-step 快照） | A 臂（tied 线性） |
| 有效性 | 与 gated 的 Δ 差距（< 0.02 ⇒ 参数量解释） | gated 臂 −0.0812 |
| 收益·用法 1（质量） | 税收回幅度 vs 同引擎 dense 13M（1.5287） | A 臂 +0.3295 |
| 收益·用法 2（自由度） | null 能量占比 / functional_dirs / 有效读出秩 | A 臂 workspace 探针 |
| 可控性 | prob0 ECE、exit-4 ECE、（软提交 rescore） | A 臂 / gated 臂 |
| 副产品 | 出口资产 tf@4 / tf@6 是否保持 | A 臂 0.9694 / gated 0.9680 |
| **干预活性（必报）** | P/Q fro 范数、absmax、非零占比、读出变换相对幅度 | 0（死干预） |
| **稳定性（必报）** | gnorm max / 中位数 / 尖峰次数、NaN、loss 突跳、逐深度 ECE | A 臂 / gated 臂 |

**判读规则**：先过等值性门与干预活性门；两者不过 → 先修实现再判机制。

## 4. 判据线（预先写下，逐字来自契约 B，不得修改）

- 若该对照臂的 Δbpc ≈ gated 臂的 Δbpc（**差距 < 0.02**）⇒ **增益来自参数量而非
  非线性**，gated 结论降级为「加参数的同价效果」，写入 `analysis.md` 并在 catalog
  标注入账。
- 若该对照臂 **Δbpc ≥ −0.014**（无改善）而 gated **≤ −0.05** ⇒ **非线性本身有效**，
  结论加固。
- **稳定性报告（必交）**：两臂 + 本对照的 `gnorm` max / 中位数 / 尖峰次数、有无 NaN
  或 loss 突跳、按深度 ECE 表（可从既有日志与重跑日志出）。

方差界参照：Δ < 0.014 不可排序（`docs/methodology.md` §2）；判据阈值 0.02 > 0.014，
即可判定性成立。

## 5. 对照与方差

- **A 臂**（tied 线性，`.scratch/23-depth-ar/evidence/23c-H2-w0-12k`，同配置同 seed）
  = Δ 的参照；**gated 臂**（`../2610-.../runs/gated-12k-s0`，同配置同 seed）= 被检验
  的声称。两臂都在本脚本下用**当前评测代码重测**同一电池，口径对齐。
- **seed 数**：本对照臂 1 seed（契约未授权补 seed）；三臂同 seed=0。
- 方差界：Δ < 0.014 不可排序。

## 6. 成本与 scope

| 项 | 值 |
|---|---|
| `gpu_wall_budget` | **18 min**（冒烟 ~1 + 训练 ≈13 + 三臂电池 ≈2 + 余量） |
| 最小可判定版本 | 13M / 12k 步 / 1 seed / 只测 clean bpc + tf@4 + ECE + 活性 + 稳定性 |
| 峰值显存 | ≤5GB net（共享卡已占 8–12GB；批大小与 gated 臂完全相同 = 32） |
| 可中断性 | 是（每 200 步落 log；中断后可按 checkpoint 续） |
| 冒烟计划 | `smoke.py`：训练前恒等起步（逐位）+ 20 步后 P/Q/transform 非零（活性）+ 参数量 = A + 2·d² |
| 预算不够时砍什么 | 砍 eval-batches（20→8）、砍 workspace/generation 副产品；**不砍判据线** |

规模纪律（SCHEMA §7）：单卡 RTX 4090 24GB 且被多任务共享，只做 13M 原理验证，
不外推规模/渐近。

## 7. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | char 级 enwik8（`data/enwik8_full.npz`，本地已有，不下载；前 90M train / 末 10M eval / 词表 206） |
| 入口脚本 | `run.py`（三臂等值性门 + 训练 + 电池 + 判读 + 稳定性）；冒烟 `smoke.py` |
| 代码改动 | `pathlm/depth_ar.py`（新增 `readout_linear_resid` 配置与读出分支，缺省逐位还原 v1）、`train_depth_ar.py`（`--readout-linear-resid`）、`tests/test_depth_ar.py`（+6 可失败单测） |
| 可失败单测 | ① 开关关闭逐位还原 A 臂；② 恒等起步（P=0）逐位 == tied；③ 精确公式 + **纯线性叠加律** + 开关变异抓红；④ 活性门（∂L/∂P≠0、一步后 P≠0、随后 ∂L/∂Q≠0）；⑤ 参数量与 gated 臂相等；⑥ 与 `--readout-gated` 互斥 |
| 变异验证 | 三处故意破坏（忽略开关 / 退回 P=Q=0 鞍点 / 给线性臂塞非线性）必须抓红——见 `analysis.md` §5 |
| 落盘 | `results.json` / `train_log.jsonl` / `gate_log.jsonl` / `config.json` / `analysis.md` / `stability.md` / `MODEL.md` |

## 8. 失败决策树

| 症状 | 诊断顺序 | 处置 |
|---|---|---|
| 起步非恒等 / 参数量不等 | ① 实现 bug | 先修实现 + 跑变异验证；未排除前不判机制 |
| 干预死（P/Q/transform ≈ 0） | ① 实现（鞍点） | 检查 init 是否退回双零；活性门必须非零才判 |
| Δ_linear 与 Δ_gated 都在 ±0.014 内 | ② 无头寸 | 如实报「两读出几何在 13M 都无头寸」，升级规模再判（不叠加第二机制） |
| Δ_linear 与 Δ_gated 差距 ∈ [0.014, 0.02) | ③ 方差边界 | 报「落在方差界与判据阈值之间」，不裁决，附 n=1 声明 |
| 结果反向（linear 显著好于 gated） | ④ 设计/优化 | 报「线性重参数化已足够」并写清它是 untied 线性读出的等价物（见 analysis §6） |
| 结果不可重复 | ⑤ 方差 | 补 seed；仍不可排序则撤回 |

## 9. 产出

`axes/W-width/W4-readout-geometry/experiments/2611-equalparams-linear-control-13m/`
下的 `README.md`(本文件) / `run.py` / `smoke.py` / `config.json` / `results.json` /
`train_log.jsonl` / `analysis.md` / `stability.md` / `MODEL.md`。
