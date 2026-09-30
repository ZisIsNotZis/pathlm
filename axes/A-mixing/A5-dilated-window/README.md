---
id: A5-dilated-window
axis: A
grid_dir: A-mixing/A5-dilated-window
title: "膨胀 / 跨步窗"
status: stub
judge: "bpc + 远程检索保留（KV 消融）"
cost: "13M 单臂 ~25 min"
tax: "—"
benefit: "—"
verdict: "—"
deps: []
evidence: []
---

<!-- AUTO:begin -->

# A5-dilated-window · 膨胀 / 跨步窗

> 轴：**A 序列混合机制** ｜ 状态：**⬜ stub**

## 1. 命题

用固定稀疏模式覆盖更远范围而省算力。

## 2. 选项枚举（本格全部可能取值）

1. `跨步(k`
2. `stride s)`
3. `膨胀(dilation d)`
4. `分层（局部密集 + 每 k 一个全局）`
5. `学习式块选择`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地**：零经验（无实验）。

**行业口径**：Longformer/BigBird 式稀疏模式；现代长上下文多用块稀疏 + 学习式选择（NSA/MoBA）。

## 4. 结论与数字

- **税**：—
- **收益**：—

## 5. 失败路径与原因

（无）

## 6. 判据与判决实验设计

- **判据**：bpc + 远程检索保留（KV 消融）

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：bpc + 远程检索保留（KV 消融）
- 成本：13M 单臂 ~25 min

## 7. 成本估计

- 13M 单臂 ~25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

（本地未做；复现命令待本格实验目录建立后补）

## 备注

本地零经验。与 A6 学习式 top-k 算子的边界：固定模式 vs 内容选择。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
