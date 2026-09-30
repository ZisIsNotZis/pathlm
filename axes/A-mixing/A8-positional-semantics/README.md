---
id: A8-positional-semantics
axis: A
grid_dir: A-mixing/A8-positional-semantics
title: "位置语义"
status: partial
judge: "bpc + 长度泛化（训练短测长）"
cost: "13M 单臂 ~25 min"
tax: "dist_pen（相对偏置族）−0.007"
benefit: "—"
verdict: "绝对位置嵌入是硬上界（解码总长 ≤ seq_len），相对偏置族已侧证。"
deps: []
evidence: [".scratch/07-x1-x3/evidence @7b47555", ".scratch/03-engine-m1/issues @e38349f"]
---

<!-- AUTO:begin -->

# A8-positional-semantics · 位置语义

> 轴：**A 序列混合机制** ｜ 状态：**◐ partial**

> **判读**：绝对位置嵌入是硬上界（解码总长 ≤ seq_len），相对偏置族已侧证。

## 1. 命题

位置如何被编码进注意力与状态。

## 2. 选项枚举（本格全部可能取值）

1. `绝对可学习位置嵌入（本项目）`
2. `相对偏置（本项目 dist_pen 属此族）`
3. `RoPE`
4. `插值/外推（YaRN/NTK）`
5. `无位置（序无关）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：—

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：dist_pen（相对偏置族）−0.007
- **收益**：—
- **判读**：绝对位置嵌入是硬上界（解码总长 ≤ seq_len），相对偏置族已侧证。

## 5. 失败路径与原因

1. 绝对位置嵌入使 ring-buffer 解码有硬上界（总 ≤ seq_len），且驱逐后位置递增必须持续测试（曾是 P0 类 bug）

## 6. 判据与判决实验设计

- **判据**：bpc + 长度泛化（训练短测长）

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：bpc + 长度泛化（训练短测长）
- 成本：13M 单臂 ~25 min

## 7. 成本估计

- 13M 单臂 ~25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. .scratch/07-x1-x3/evidence @7b47555
2. .scratch/03-engine-m1/issues @e38349f

## 备注

RoPE / 插值外推本地零经验；长上下文线必须处理本格。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
