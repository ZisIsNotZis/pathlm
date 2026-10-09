---
id: C3-inter-layer-topology
axis: C
grid_dir: C-vertical/C3-inter-layer-topology
title: "层间拓扑（残差之外的连接）"
status: judged
judge: "clean bpc（同参数量）+ 训练稳定性 + 深度可截断性"
cost: "13M 单臂 ~25–40 min"
tax: "parallel +0.0458（1.5254 → 1.5712）/ parallel-dual-ln（等参数） +0.0402 / dense 层间加权和 −0.0008（中性）；12k 步 / 13M / seed0 / 98.3M token"
benefit: "（无）"
verdict: "首探无优势：**并行子层明确有害**（+0.0458，等参数对照 +0.0402 复现），**层间学习加权和中性**（−0.0008）。"
deps: []
evidence: ["axes/C-vertical/C3-inter-layer-topology/experiments/2613-inter-layer-topology-13m/analysis.md @b3c2484"]
---

<!-- AUTO:begin -->

# C3-inter-layer-topology · 层间拓扑（残差之外的连接）

> 轴：**C 垂直机制（深度）** ｜ 状态：**✅ judged**

> **判读**：首探无优势：**并行子层明确有害**（+0.0458，等参数对照 +0.0402 复现），**层间学习加权和中性**（−0.0008）。

## 1. 命题

层与层之间除了单条残差，还能怎么连。

## 2. 选项枚举（本格全部可能取值）

1. `单残差（pre-LN 顺序，本项目基底）`
2. `并行（attn ∥ MLP，GPT-J/PaLM 式）`
3. `密集（对全部历史层输出加权和，DenseFormer 式）`
4. `跨层注意力（层间用注意力连接，Hyper-Connections / feedback transformer 式）`
5. `跨层跳连（LayerSkip 式早退跳连）`
6. `无残差（纯前馈堆栈）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：（无）

**行业口径**：并行化（PaLM/GPT-J）已被大模型广泛采纳（吞吐收益）；本地实测其质量代价 +0.05（等参数对照复现）。
密集/跨层注意（DenseFormer、Hyper-Connections）报告同规模小幅增益；本地 dense 加权和为中性。

## 4. 结论与数字

- **税**：parallel +0.0458（1.5254 → 1.5712）/ parallel-dual-ln（等参数） +0.0402 / dense 层间加权和 −0.0008（中性）；12k 步 / 13M / seed0 / 98.3M token
- **收益**：（无）
- **判读**：首探无优势：**并行子层明确有害**（+0.0458，等参数对照 +0.0402 复现），**层间学习加权和中性**（−0.0008）。

## 5. 失败路径与原因

1. 并行子层（PaLM/GPT-J 式）在此规模明确有害：+0.0458，且用逐位等参数的 dual-LN 对照复现 +0.0402（排除参数量混杂）
2. 层间学习加权和（DenseFormer 式）中性 −0.0008（|Δ| < 0.014）：层权确实活着（fro 3.305、偏离恒等起步 1.361、浅层被放大 2.36×）但质量无收益
3. 收敛对账（同方法末 25% 窗斜率 0.0025–0.0045 nats/1k，极差 0.0020）⇒ 差异是水平差不是收敛速度差
4. **契约缺陷**：并列判据 tf@4 ≥ 0.95 对 stock dense 形态不适用（只监督最终深度，同配置 dense tf@4 = 0.444；四臂 0.31–0.46 全部不可达）→ 该判据零区分度，不作区分量
5. n=1 seed；dense 的层权无单纯形约束（≠ 原论文严格凸组合语义）；24k 收敛点未跑

## 6. 判据与判决实验设计

- **判据**：clean bpc（同参数量）+ 训练稳定性 + 深度可截断性

## 7. 成本估计

- 13M 单臂 ~25–40 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. axes/C-vertical/C3-inter-layer-topology/experiments/2613-inter-layer-topology-13m/analysis.md @b3c2484

## 备注

首探结论：**改层间连接在 13M/12k 上没有便宜可捡**。未测：跨层注意力（Hyper-Connections 式）、
跨层跳连、无残差、以及 24k/更大尺寸。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
