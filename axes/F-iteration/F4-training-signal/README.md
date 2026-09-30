---
id: F4-training-signal
axis: F
grid_dir: F-iteration/F4-training-signal
title: "迭代/重复的训练信号"
status: judged
judge: "目标量（组合一致性 / 重建质量）+ 税 + 是否反噬主任务"
cost: "13M 单臂 ~25 min"
tax: "consistency loss 中性（IX4 1.6589 ≈ C1 1.6549）；BF16 下 reward 类损失两次发散 → 必须 float32 + pre-step grad-norm gate"
benefit: "M0 微小模型上 consistency 收紧估计（cosine 0.9525→0.9882）"
verdict: "consistency loss 中性；重建/sufficiency 是唯一无人占位、且是压缩可恢复性的必要信号。"
deps: ["H3-composition", "A9-old-info-retention", "C6-depth-supervision"]
evidence: [".scratch/02-engine-m0/evidence/REPORT.md @27418ae", ".scratch/04-m1-runs/evidence @e38349f"]
---

<!-- AUTO:begin -->

# F4-training-signal · 迭代/重复的训练信号

> 轴：**F 迭代** ｜ 状态：**✅ judged**

> **判读**：consistency loss 中性；重建/sufficiency 是唯一无人占位、且是压缩可恢复性的必要信号。

## 1. 命题

靠什么损失让「重复」与「组合」真的有效。

## 2. 选项枚举（本格全部可能取值）

1. `无（纯 CE）`
2. `consistency loss（T₂ ≈ T₁∘T₁）`
3. `重建/sufficiency loss（只给摘要重建原文）`
4. `自蒸馏`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：M0 微小模型上 consistency 收紧估计（cosine 0.9525→0.9882）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：consistency loss 中性（IX4 1.6589 ≈ C1 1.6549）；BF16 下 reward 类损失两次发散 → 必须 float32 + pre-step grad-norm gate
- **收益**：M0 微小模型上 consistency 收紧估计（cosine 0.9525→0.9882）
- **判读**：consistency loss 中性；重建/sufficiency 是唯一无人占位、且是压缩可恢复性的必要信号。

## 5. 失败路径与原因

1. consistency loss 在真实规模不改善聚合质量（中性）→「让组合被训练出来」的既有手段无效
2. negative/reward 项在 bf16 autocast 下不稳定（两次发散教会）

## 6. 判据与判决实验设计

- **判据**：目标量（组合一致性 / 重建质量）+ 税 + 是否反噬主任务

## 7. 成本估计

- 13M 单臂 ~25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `H3-composition`
2. `A9-old-info-retention`
3. `C6-depth-supervision`

## 9. 复现手段

1. .scratch/02-engine-m0/evidence/REPORT.md @27418ae
2. .scratch/04-m1-runs/evidence @e38349f

## 备注

空白格：sufficiency/重建 loss 是「右移压缩」能成立的唯一训练信号——
模型必须被惩罚「编造自己丢掉的内容」。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
