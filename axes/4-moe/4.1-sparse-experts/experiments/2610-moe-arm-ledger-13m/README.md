---
name: 2610-moe-arm-ledger-13m
ticket: 65
date: "2026-10-13"
grid: 4.1-sparse-experts
engine: "depth-ar（stock dense 对照 + 轴 4 自持 MoE 模块，未改共享 pathlm/**）"
budget: "本阶段 0 GPU 判决训练（无对照质量实验）；仅含一条单臂坍缩探针（§20 持锁）。等效算力见 §6"
---

# 2610-moe-arm-ledger-13m（ticket 65 · 4.1 稀疏激活专家）

> 主格：`4.1-sparse-experts`｜关联格：`4.2-routing-balance`（坍缩探针）、`4.3-expert-granularity`｜
> 引擎：`pathlm.depth_ar.DepthARModel`（dense=True）+ 本目录 `moe.py` 的 top-k MoE FFN 透明替换
> `Block.mlp`｜预算：本阶段 0 GPU 判决训练；仅一条单臂坍缩探针（见 §6）。

## 1. 命题

在 13M 基底（`configs/B0.json`：d=256, L=12, H=8, mlp_mult=6）与固定 token 预算
（6000 步 × batch 32 × seq 512 = 98,304,000 token）下，把每层 FFN 换成 top-k 稀疏专家
（**每 token 专家激活 MACs 与 dense 逐位相等**），能否用 **8×（top-1）/ 4×（top-2）的 FFN 参数**
（对应总参数 5.9× / 3.1×，因注意力/embedding/读出不变）换到 clean bpc 的可排序改善（> 0.014 噪声带），且不出现专家使用率坍缩。

**可证伪**：若在等 FLOPs 对照下 MoE 相对 dense-B0 的 clean bpc 不占优（越带变差或持平），
且无法用使用率分布解释 ⇒ 该尺度稀疏激活不产生新知识。本阶段（Phase A）只先钉死**臂定义与代价**并
做**接线 pre-flight + 坍缩可观测性探针**；判决留待下一阶段（§6 / `gaps`）。

## 2. 单因素声明

干预只改一个因素：**FFN 的专家化**（专家数 E、每 token 激活数 k、每专家宽 h_e）。
- 基底 = stock decoder-only（`DepthARModel(cfg, vocab)`，`dense=True`，顺序残差）。
- 注意力 / LayerNorm / embedding / 读出 / 层数 L / 残差接线 / 优化器 / 数据切分 / 步数 / seed 全不动。
- `4.1` 的臂 `aux_weight = 0`（**朴素 top-k，无均衡机制**——均衡属 `4.2`）；
  专家**均匀同宽**（细粒度 / 共享专家属 `4.3`）。
- 实现方式：`moe.py:MoEDepthARModel` 在 `super().__init__()` 之后把每层 `blk.mlp` 换成
  `TopKMoEFFN`——**不改共享 `pathlm/**`**（新模块 + 最少接线，`harness/SCHEMA.md` §22.7）。
  透明性由 `preflight` 的 **E=1,top-1 + 灌入 stock mlp 权重 ⇒ 与 dense 逐位一致**断言钉住。

## 3. 指标表（多维，逐项分析）

| 维度 | 指标 | 口径 | 方向 |
|---|---|---|---|
| 质量 | clean bpc（enwik8 末 10M 留出） | matched-step / 同 token 预算 | 该跌 / 可排序（>0.014） |
| 容量 | 总参数量、FFN 参数量、`ffn_params_x_dense` | 实测（`ledger.json`） | 该涨（这是机制本身） |
| 每 token 算力 | `expert_macs_per_token`（专家部分）、`ffn_macs_per_token`（含路由税） | 解析（§15 等 FLOPs 对照） | 专家部分**必须逐位相等** |
| 活性（§10） | 逐层/全局专家使用率分布：熵、归一化熵、最大占比 | 每 100 步 + 终评 | 该均衡（防坍缩；必须报） |
| 数据饥饿 | `tokens_per_param` | token 预算 / 总参数 | 必须报（解释容量收益是否被欠拟合吃掉） |
| 可训性 | 路由器梯度范数、单步 loss 下降、路由判别力（零权重专家区分） | pre-flight | 该非零 / 该真的选到不同专家（防「死干预」） |
| 代价 | 等效算力（FLOPs / layer-passes / tokens-per-forward） | 理论 | 禁墙钟 |

## 4. 判据线（预先写下，不许事后调整）

1. **等 FLOPs 臂**：`moe-top1-E8`（E=8,k=1,h_e=1536）与 `moe-top2-E8`（E=8,k=2,h_e=768）的
   `expert_macs_per_token` 相对 `dense-B0` 必须**恰为 1.0**（等专家激活算力）；路由器 `E·d` logits
   构成的「路由税」单列并报（本配置 +0.26%）。
2. **等参数对照**：`dense-eqparam-top1`（mlp_mult=48，~79.8M）与 `dense-eqparam-top2`
   （mlp_mult=24，~42.0M）的总参数须与对应 MoE 臂 **相对差 < 0.2%**。
3. **主判据**：等 FLOPs 下 `Δ = bpc(dense-B0) − bpc(moe-*)`；`Δ > 0.014` ⇒ 占优可排序；
   `|Δ| ≤ 0.014` ⇒ 不可排序；`Δ < −0.014` ⇒ 劣。n≥2 seed。
4. **活性判据**：专家使用率归一化熵须显著高于坍缩阈值。**本阶段探针先测「该尺度上坍缩是否可观测」**：
   若 1200 步内归一化熵始终 > 0.5（或最大占比 < 0.6）⇒ **本尺度「坍缩」不可观测**，则 `4.2` 的
   前提在本尺度**无对象**（须如实登记，而非沉默）。**注**：该阈值在随机初始化第 1 步即已满足（0.968/0.205），
   命中近乎必然 ⇒ 主证应为「1200 步内是否**下移**（斜率）」与「是否有持久偏斜」，阈值只作附带。
5. **必须不变**：因果性（stock 全因果注意力）、总参数记账口径（不把「减参数」记成「减算力」）。

## 5. 对照与方差

- 对照臂 = `dense-B0`（同尺寸 stock dense，归一为 100%）；等参数对照 = `dense-eqparam-*`。
- 方差带 = **0.014 bpc**（`harness/methodology.md` §2：nondeterminism 0.0019、种子散布 ≤0.0138）。
- seed 数：判决需 n≥2；本阶段探针单臂不报质量对照（只报活性）。

## 6. 成本（等效理论量；墙钟仅作过程指标，禁入 `cost_model`/判据）

- **每 token FFN 等效算力**：dense = 2·d·h = 786,432 MACs/层；MoE（等 FLOPs 臂）专家部分恰等，
  另加路由 E·d = 2,048 MACs/层（+0.26%）。
- **layer-passes**：三臂每 token 均 L=12（专家路由是同一前向内的稀疏 gather，不增顺序依赖）。
- **tokens-per-forward**：每步 1。
- **全实验等效训练算力**（下一阶段）：3 臂 × 98.3M token ×（每 token 与 dense-B0 同量级的前向+反向）
  ≈ 3× dense-B0 的一次训练；n≥2 ⇒ ×2。**实测步时须开跑前先测 20 步再定预算**（`harness/SCHEMA.md` §17.8）；
  本阶段 20 步实测（争用下）见 `timing_20.json` 的 `timing_process_metric`。
- **本阶段实际**：CPU pre-flight（秒级）+ 单臂 1200 步坍缩探针（§20 串行取锁）。

## 7. 数据与脚本清单

- 数据：`data/enwik8_full.npz`（前 90M train / 末 10M eval，vocab=206，byte-level）。
- `moe.py`：`TopKMoEFFN`（透明替换 `Block.mlp`）+ `MoEDepthARModel` + `copy_shared_from`。
- `run.py ledger|preflight|train`。
- 落盘：`ledger.json`、`preflight.json`、`results.json`、`smoke_*.json`、`timing_20.json`、
  `probe_collapse_1200.json`、`analysis.md`、`MODEL.md`。

## 8. 失败决策树（无效时的处置路径：测量 → 实现 → 规模 → 设计）

1. **测量**：`preflight` 的 E=1 等值断言失败 ⇒ 实现问题（不是机制问题），先修接线。
2. **实现**：等 FLOPs / 等参数对照不齐 ⇒ 记账错，回 `ledger` 重算，不得开跑。
3. **规模**：13M 上场间差异 ≤ 噪声带 ⇒ 报「该尺度不可分」，不外推（`harness/SCHEMA.md` §17.11 禁外推）。
4. **设计**：若等 FLOPs 下 MoE 一致劣且使用率分布健康（非坍缩）⇒ 判该尺度稀疏激活不产生新知识
   （`failure_analysis` 区分：结构问题 vs innovation 本身失败）。

## 9. 产出

`results.json`（含 `ledger` / `preflight` / `cpu_smoke` / `collapse_probe`）、`analysis.md`（判读）、
`MODEL.md`（权重不入库）。
