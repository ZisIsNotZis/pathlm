---
id: A2-full-attention
axis: A
grid_dir: A-mixing/A2-full-attention
title: "全注意力基线"
status: judged
judge: "clean bpc（对照锚点）"
cost: "13M 单臂 ~5 min"
tax: "—（锚点本身）"
benefit: "13M B0 = 1.5074 bpc（1.1 epoch 快照）/ 1.3459（4.4 epoch 渐近）；100M B0 = 1.4612（1.09 epoch）"
verdict: "基底成立，外部锚在公开数字范围内。"
deps: []
evidence: ["archive/legacy-docs/findings.md @5495a3b", ".scratch/04-m1-runs/evidence/B0 @e38349f", ".scratch/24-depth-ar-100m/evidence @7cdcdaa"]
---

<!-- AUTO:begin -->

# A2-full-attention · 全注意力基线

> 轴：**A 序列混合机制** ｜ 状态：**✅ judged**

> **判读**：基底成立，外部锚在公开数字范围内。

## 1. 命题

不做任何可见性约束时的锚点质量。

## 2. 选项枚举（本格全部可能取值）

1. `全因果 softmax 注意力（无窗、无偏置、无锚）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：13M B0 = 1.5074 bpc（1.1 epoch 快照）/ 1.3459（4.4 epoch 渐近）；100M B0 = 1.4612（1.09 epoch）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：—（锚点本身）
- **收益**：13M B0 = 1.5074 bpc（1.1 epoch 快照）/ 1.3459（4.4 epoch 渐近）；100M B0 = 1.4612（1.09 epoch）
- **判读**：基底成立，外部锚在公开数字范围内。

## 5. 失败路径与原因

（无）

## 6. 判据与判决实验设计

- **判据**：clean bpc（对照锚点）

## 7. 成本估计

- 13M 单臂 ~5 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. archive/legacy-docs/findings.md @5495a3b
2. .scratch/04-m1-runs/evidence/B0 @e38349f
3. .scratch/24-depth-ar-100m/evidence @7cdcdaa

## 备注

所有税的基准。引用任何税之前必须确认对照口径（快照 / 渐近 / matched-step）
与引擎（同引擎税才是干净归属，见 24b）。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
