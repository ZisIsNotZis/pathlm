---
id: P4-normalization
axis: P
grid_dir: P-modifiers/P4-normalization
title: "归一化位置（修饰符）"
status: partial
judge: "训练稳定性 + 残差流范数分布 + 是否与「同空间」前提兼容"
cost: "13M 单臂 ~25 min"
tax: "—"
benefit: "无 final LayerNorm 使 logits = h·Eᵀ 语义成立（readout scale 由 U 学）"
verdict: "pre-LN + 无 final norm 是「同空间」几何的前提，已固定使用。"
deps: []
evidence: [".scratch/03-engine-m1/issues @e38349f"]
---

<!-- AUTO:begin -->

# P4-normalization · 归一化位置（修饰符）

> 轴：**P 修饰符（不是轴）** ｜ 状态：**◐ partial**

> **判读**：pre-LN + 无 final norm 是「同空间」几何的前提，已固定使用。

## 1. 命题

LayerNorm 放在哪、是否有 final norm——决定残差流是否保持 embedding 几何。

## 2. 选项枚举（本格全部可能取值）

1. `pre-LN（本项目）`
2. `post-LN`
3. `QK-norm`
4. `无 final LayerNorm（本项目，同空间前提）`
5. `RMSNorm`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：无 final LayerNorm 使 logits = h·Eᵀ 语义成立（readout scale 由 U 学）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：—
- **收益**：无 final LayerNorm 使 logits = h·Eᵀ 语义成立（readout scale 由 U 学）
- **判读**：pre-LN + 无 final norm 是「同空间」几何的前提，已固定使用。

## 5. 失败路径与原因

（无）

## 6. 判据与判决实验设计

- **判据**：训练稳定性 + 残差流范数分布 + 是否与「同空间」前提兼容

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：训练稳定性 + 残差流范数分布 + 是否与「同空间」前提兼容
- 成本：13M 单臂 ~25 min

## 7. 成本估计

- 13M 单臂 ~25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. .scratch/03-engine-m1/issues @e38349f

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
