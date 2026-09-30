---
id: R6-moe-scale
axis: R
grid_dir: R-references/R6-moe-scale
title: "MoE 规模轴"
status: reference
judge: "同 FLOPs 质量"
cost: "13M 单臂 ~25–40 min"
tax: "—"
benefit: "—"
verdict: "—"
deps: []
evidence: []
---

<!-- AUTO:begin -->

# R6-moe-scale · MoE 规模轴

> 轴：**R 参照线（必须打败的基线）** ｜ 状态：**🏭 reference**

## 1. 命题

同 FLOPs 更多参数——唯一被大规模验证的「容量-计算分离」路线。

## 2. 选项枚举（本格全部可能取值）

1. `top-k 路由`
2. `细粒度专家`
3. `共享专家`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地**：零经验（无实验）。

**行业口径**：Mixtral / DeepSeek-MoE 等已证明；与 W2 同格，此处作为「参照线」登记其判据线。

## 4. 结论与数字

- **税**：—
- **收益**：—

## 5. 失败路径与原因

（无）

## 6. 判据与判决实验设计

- **判据**：同 FLOPs 质量

## 7. 成本估计

- 13M 单臂 ~25–40 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

（本地未做；复现命令待本格实验目录建立后补）

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
