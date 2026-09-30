---
id: C3-inter-layer-topology
axis: C
grid_dir: C-vertical/C3-inter-layer-topology
title: "层间拓扑（残差之外的连接）"
status: stub
judge: "clean bpc（同参数量）+ 训练稳定性 + 深度可截断性"
cost: "13M 单臂 ~25–40 min"
tax: "—"
benefit: "—"
verdict: "—"
deps: []
evidence: []
---

<!-- AUTO:begin -->

# C3-inter-layer-topology · 层间拓扑（残差之外的连接）

> 轴：**C 垂直机制（深度）** ｜ 状态：**⬜ stub**

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

**本地**：零经验（无实验）。

**行业口径**：并行化（PaLM/GPT-J）已被大模型广泛采纳（吞吐收益）；
密集/跨层注意（DenseFormer、Hyper-Connections）报告同规模下的稳定小幅增益，
但未成为主流。

## 4. 结论与数字

- **税**：—
- **收益**：—

## 5. 失败路径与原因

（无）

## 6. 判据与判决实验设计

- **判据**：clean bpc（同参数量）+ 训练稳定性 + 深度可截断性

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：clean bpc（同参数量）+ 训练稳定性 + 深度可截断性
- 成本：13M 单臂 ~25–40 min

## 7. 成本估计

- 13M 单臂 ~25–40 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

（本地未做；复现命令待本格实验目录建立后补）

## 备注

本地零经验 + 与「同空间」哲学自洽（不引入新空间，只改连接拓扑）→
最便宜的真空白之一。用户 hint：「残差换成层间 attention」。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
