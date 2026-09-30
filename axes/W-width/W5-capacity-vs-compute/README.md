---
id: W5-capacity-vs-compute
axis: W
grid_dir: W-width/W5-capacity-vs-compute
title: "容量-计算分离"
status: judged
judge: "同 FLOPs 的质量 + 同参数量的 FLOPs"
cost: "13M 单臂 ~25 min"
tax: "容量削减：负结果（W1 −50.4%）"
benefit: "—"
verdict: "减容量不买灵活性；买灵活性必须走保容量减计算。"
deps: []
evidence: [".scratch/25-tax-attribution/evidence @2e86924"]
---

<!-- AUTO:begin -->

# W5-capacity-vs-compute · 容量-计算分离

> 轴：**W 宽度** ｜ 状态：**✅ judged**

> **判读**：减容量不买灵活性；买灵活性必须走保容量减计算。

## 1. 命题

哪些宽度侧干预是「减容量」（付质量）、哪些是「保容量减计算」。

## 2. 选项枚举（本格全部可能取值）

1. `容量削减（减宽/剪枝/裁头）`
2. `保容量减计算（稀疏激活/专家路由/条件计算）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：—

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：容量削减：负结果（W1 −50.4%）
- **收益**：—
- **判读**：减容量不买灵活性；买灵活性必须走保容量减计算。

## 5. 失败路径与原因

1. 把「减宽」当作灵活性来源是类别错误：它减容量，不减每 token 的固定成本结构

## 6. 判据与判决实验设计

- **判据**：同 FLOPs 的质量 + 同参数量的 FLOPs

## 7. 成本估计

- 13M 单臂 ~25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. .scratch/25-tax-attribution/evidence @2e86924

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
