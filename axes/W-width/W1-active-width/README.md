---
id: W1-active-width
axis: W
grid_dir: W-width/W1-active-width
title: "激活宽度"
status: judged
judge: "同参数量下 clean bpc + 状态能量在保留子空间中的分布（workspace 探针）"
cost: "13M 单臂 ~25 min；100M ~35 min"
tax: "结构切分读出（只读前 360 维，@100M）clean 2.1475、同引擎税 +0.6662、恢复 −50.4%"
benefit: "（无）"
verdict: "减宽 = 减容量，不买灵活性；结构化约束会把状态能量赶出被约束子空间。"
deps: ["W4-readout-geometry", "W5-capacity-vs-compute"]
evidence: [".scratch/25-tax-attribution/evidence @2e86924"]
---

<!-- AUTO:begin -->

# W1-active-width · 激活宽度

> 轴：**W 宽度** ｜ 状态：**✅ judged**

> **判读**：减宽 = 减容量，不买灵活性；结构化约束会把状态能量赶出被约束子空间。

## 1. 命题

每个 token 是否可以用更少的通道/宽度跑完。

## 2. 选项枚举（本格全部可能取值）

1. `全宽`
2. `减宽（训练时或推理时裁通道）`
3. `随机宽度（width dropout）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：（无）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：结构切分读出（只读前 360 维，@100M）clean 2.1475、同引擎税 +0.6662、恢复 −50.4%
- **收益**：（无）
- **判读**：减宽 = 减容量，不买灵活性；结构化约束会把状态能量赶出被约束子空间。

## 5. 失败路径与原因

1. 机制探针显示模型把状态能量整体迁出读出子空间：切片能量 depth0 0.85 → depth2 0.026 → ≈0；null 能量 100% 但 functional_dirs=176（workspace 被真用、读出空间被放弃）
2. 门动力学反转（layer 3–11 p90=1.0 vs 正常臂全层 ≤0.07）、depth 曲线尾部倒退（d7→d12 +0.012）
3. tied 表不对称漂移（读出梯度只打 E[:, :360]）：输入态能量比 0.85 vs 随机基线 0.50
4. 用户本地独立实验（减宽）同样「效果微妙」——与本文结论一致：减容量不买灵活性

## 6. 判据与判决实验设计

- **判据**：同参数量下 clean bpc + 状态能量在保留子空间中的分布（workspace 探针）

## 7. 成本估计

- 13M 单臂 ~25 min；100M ~35 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `W4-readout-geometry`
2. `W5-capacity-vs-compute`

## 9. 复现手段

1. .scratch/25-tax-attribution/evidence @2e86924

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
