---
id: H3-composition
axis: H
grid_dir: H-horizon/H3-composition
title: "多位置预测的组合方式"
status: partial
judge: "同 FLOPs 的质量（等算力对照）+ 草稿接受率 + 组合置信度的校准（ECE）"
cost: "13M 单臂 ~25–30 min"
tax: "（组合机制本身的税未单独测；MTP 头的税见 H1）"
benefit: "oracle 链 0.682 ≈ node-1 0.693（机制上限成立）；给定 node-1 正确则接受 77.3%"
verdict: "组合的结构问题从未被测过：已测的是「条件链 deploy 输给直接头」，归因是草稿质量瓶颈而非结构。"
deps: ["H1-outputs-per-pass", "F4-training-signal", "G1-selector"]
evidence: [".scratch/12-overnight/evidence @216a00a", ".scratch/04-m1-runs/evidence @e38349f"]
---

<!-- AUTO:begin -->

# H3-composition · 多位置预测的组合方式

> 轴：**H 视界（一次前向产生几个位置）** ｜ 状态：**◐ partial**

> **判读**：组合的结构问题从未被测过：已测的是「条件链 deploy 输给直接头」，归因是草稿质量瓶颈而非结构。

## 1. 命题

直接头与多跳组合如何合并——这是用户提出的「自门控非线性组合」所在格。

## 2. 选项枚举（本格全部可能取值）

1. `只用直接头`
2. `链式条件（concat+proj，DeepSeek 式）`
3. `朴素概率乘积`
4. `DP lattice 置信加权（设计 §5 已定义，从未端到端测量）`
5. `自门控选组合深度（一次跳 k vs 跳 1 走 k 次，由 sigmoid 门决定）`
6. `beam 多线 + 重排`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：oracle 链 0.682 ≈ node-1 0.693（机制上限成立）；给定 node-1 正确则接受 77.3%

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：（组合机制本身的税未单独测；MTP 头的税见 H1）
- **收益**：oracle 链 0.682 ≈ node-1 0.693（机制上限成立）；给定 node-1 正确则接受 77.3%
- **判读**：组合的结构问题从未被测过：已测的是「条件链 deploy 输给直接头」，归因是草稿质量瓶颈而非结构。

## 5. 失败路径与原因

1. 朴素复合 P1·P2chain CE 3.96 vs node-2 单独 1.58 —— 数学错误（正确链必须条件在 node-1 的采样 token 上）
2. 可部署条件链 0.547 < direct 0.560，~31% 草稿错误、接受 0.75≈0.77；k=1 naive verify-next-round 零吞吐收益 → 瓶颈是草稿质量，不是组合结构
3. consistency loss 中性（IX4 1.6589 ≈ C1 1.6549）→「让组合被训练出来」的既有手段实测无效

## 6. 判据与判决实验设计

- **判据**：同 FLOPs 的质量（等算力对照）+ 草稿接受率 + 组合置信度的校准（ECE）

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：同 FLOPs 的质量（等算力对照）+ 草稿接受率 + 组合置信度的校准（ECE）
- 成本：13M 单臂 ~25–30 min

## 7. 成本估计

- 13M 单臂 ~25–30 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `H1-outputs-per-pass`
2. `F4-training-signal`
3. `G1-selector`

## 9. 复现手段

1. .scratch/12-overnight/evidence @216a00a
2. .scratch/04-m1-runs/evidence @e38349f

## 备注

空白格（用户新设计）。等 FLOPs 的判决实验：1 次前向 + 跳跃头预测 t+2
vs 2 次前向推到 t+2（已有数字：跳跃头 55.97% / 条件接受 65.8%，
连跑两次 ≈ node-1 68.8% → 跳跃头以一半算力买到约 81% 准确率）。
质量头寸已被花掉（d6−d12 = 0.0003）→ 组合的价值应在成本分配而非质量。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
