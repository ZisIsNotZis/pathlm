---
id: M3-cross-step-assumption
axis: M
grid_dir: M-carrier/M3-cross-step-assumption
title: "跨步假设"
status: partial
judge: "质量 + 内存 + 不重 prefill 能力"
cost: "13M 单臂 ~25–40 min"
tax: "—"
benefit: "ring-buffer 式解码免重 prefill；变深度增量解码逐位一致"
verdict: "每 token 独立 + 「不重 prefill」已实现；段级递归零经验。"
deps: []
evidence: [".scratch/03-engine-m1/issues @e38349f"]
---

<!-- AUTO:begin -->

# M3-cross-step-assumption · 跨步假设

> 轴：**M 状态载体** ｜ 状态：**◐ partial**

> **判读**：每 token 独立 + 「不重 prefill」已实现；段级递归零经验。

## 1. 命题

相邻 token 之间的状态关系——每 token 独立算，还是段级递归。

## 2. 选项枚举（本格全部可能取值）

1. `每 token 独立（标准 AR）`
2. `段级递归（Transformer-XL 式）`
3. `全量重算`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：ring-buffer 式解码免重 prefill；变深度增量解码逐位一致

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：—
- **收益**：ring-buffer 式解码免重 prefill；变深度增量解码逐位一致
- **判读**：每 token 独立 + 「不重 prefill」已实现；段级递归零经验。

## 5. 失败路径与原因

（无）

## 6. 判据与判决实验设计

- **判据**：质量 + 内存 + 不重 prefill 能力

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：质量 + 内存 + 不重 prefill 能力
- 成本：13M 单臂 ~25–40 min

## 7. 成本估计

- 13M 单臂 ~25–40 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. .scratch/03-engine-m1/issues @e38349f

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
