---
id: W4-readout-geometry
axis: W
grid_dir: W-width/W4-readout-geometry
title: "读出几何"
status: judged
judge: "clean bpc（税）+ workspace 探针（null 能量占比 / functional_dirs）+ 校准（ECE）+ 传输与提交机制是否仍可用"
cost: "13M 单臂 ~15–25 min"
tax: "**读出重参数化 Δ clean bpc −0.0812**（gated，12k）/**−0.0888**（等参数纯线性对照，12k）→ 二者不可排序（差 0.0076 < 0.014）⇒ 归因容量/重参数化；**24k 复测 Δ = −0.0463（缩减 43%）**；tied 线性 = 基底；切分式紧凑读出 = 负结果（见 W1）"
benefit: "tf@4 0.9694 → **0.9680**（≥0.95 保持）、tf@6 0.9938 → 0.9933；ECE 0.0066 → **0.0051**；null 能量占比 0.2703 → **0.1952**（读出面看到更多状态，−7.5pp）；全深度逐深度支配 A 臂；参数量仅 +2·d² = **131072**（可 scale，非 vocab×hidden）"
verdict: "读出几何**可以**解耦（prob0/提交/传输不受损，同空间前提不冲突），但**增量归因于「读出重参数化 + 额外 2d² 参数」，不是非线性本身**：等参数纯线性对照 Δ=−0.0888 与 gated 的 −0.0812 **不可排序**（差 0.0076 < 0.014）；且增益随训练缩小（12k −0.0812 → 24k −0.0463，缩 43%，两臂仍未收敛）⇒ 更像**收敛加速/容量效应**而非最终质量提升。**不采纳为质量机制**；保留为「读出重参数化」候选，待更长训练与更大尺寸确认。"
deps: ["W1-active-width", "A9-old-info-retention", "H4-head-trunk-relation"]
evidence: [".scratch/23-depth-ar/evidence @1586745", ".scratch/25-tax-attribution/evidence @2e86924", "axes/W-width/W4-readout-geometry/experiments/2610-gated-readout-13m-depthar/analysis.md @cba2f39", "axes/W-width/W4-readout-geometry/experiments/2611-equalparams-linear-control-13m/analysis.md @ticket29", "axes/W-width/W4-readout-geometry/experiments/2612-gated-readout-24k-13m/analysis.md @ticket29"]
---

<!-- AUTO:begin -->

# W4-readout-geometry · 读出几何

> 轴：**W 宽度** ｜ 状态：**✅ judged**

> **判读**：读出几何**可以**解耦（prob0/提交/传输不受损，同空间前提不冲突），但**增量归因于「读出重参数化 + 额外 2d² 参数」，不是非线性本身**：等参数纯线性对照 Δ=−0.0888 与 gated 的 −0.0812 **不可排序**（差 0.0076 < 0.014）；且增益随训练缩小（12k −0.0812 → 24k −0.0463，缩 43%，两臂仍未收敛）⇒ 更像**收敛加速/容量效应**而非最终质量提升。**不采纳为质量机制**；保留为「读出重参数化」候选，待更长训练与更大尺寸确认。

## 1. 命题

状态到 token 概率的映射——决定「计算空间」与「读出可见空间」是否耦合。

## 2. 选项枚举（本格全部可能取值）

1. `tied 线性（U = Eᵀ，本项目基底）`
2. `解绑线性（独立 unembed）`
3. `gated 残差读出（h·Eᵀ + SiLU-gate(h)·W，保线性桥）`
4. `纯 MLP 读出`
5. `低秩读出`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：tf@4 0.9694 → **0.9680**（≥0.95 保持）、tf@6 0.9938 → 0.9933；ECE 0.0066 → **0.0051**；null 能量占比 0.2703 → **0.1952**（读出面看到更多状态，−7.5pp）；全深度逐深度支配 A 臂；参数量仅 +2·d² = **131072**（可 scale，非 vocab×hidden）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：**读出重参数化 Δ clean bpc −0.0812**（gated，12k）/**−0.0888**（等参数纯线性对照，12k）→ 二者不可排序（差 0.0076 < 0.014）⇒ 归因容量/重参数化；**24k 复测 Δ = −0.0463（缩减 43%）**；tied 线性 = 基底；切分式紧凑读出 = 负结果（见 W1）
- **收益**：tf@4 0.9694 → **0.9680**（≥0.95 保持）、tf@6 0.9938 → 0.9933；ECE 0.0066 → **0.0051**；null 能量占比 0.2703 → **0.1952**（读出面看到更多状态，−7.5pp）；全深度逐深度支配 A 臂；参数量仅 +2·d² = **131072**（可 scale，非 vocab×hidden）
- **判读**：读出几何**可以**解耦（prob0/提交/传输不受损，同空间前提不冲突），但**增量归因于「读出重参数化 + 额外 2d² 参数」，不是非线性本身**：等参数纯线性对照 Δ=−0.0888 与 gated 的 −0.0812 **不可排序**（差 0.0076 < 0.014）；且增益随训练缩小（12k −0.0812 → 24k −0.0463，缩 43%，两臂仍未收敛）⇒ 更像**收敛加速/容量效应**而非最终质量提升。**不采纳为质量机制**；保留为「读出重参数化」候选，待更长训练与更大尺寸确认。

## 5. 失败路径与原因

1. **零初始化是鞍点**（首轮 12k 干预恒为 0、fro=0，实测 Δ 仅为重跑噪声）：g(h)=h+silu(W1h)⊙(W2h) 在 W1=W2=0 处 ∂L/∂W1 = ∂L/∂W2 = 0；正确做法 = **单边零**（W1~N(0,0.02)/W2=0：严格恒等起步且 ∂L/∂W2 = silu(W1h)⊗h ≠ 0，200 步即逃逸）
2. vocab×hidden 的 MLP 读出**不可 scale**（真词表 100k+ 是瓶颈）→ 只作对照臂，不采纳
3. **增益来源不是非线性**：等参数纯线性对照（h' = h + P·(Q h)，2d² 参数）Δ = −0.0888 与 gated 的 −0.0812 不可排序 ⇒ 非线性无额外贡献，增益来自读出重参数化/容量（SCHEMA §15 归因规则）。
4. **增益随训练缩减**：Δ 12k −0.0812 → 24k −0.0463（缩 43%）；两臂 24k 仍未收敛（末 25% 窗斜率均 ≈ −0.011/1k）⇒ 快照口径，疑为收敛加速而非最终质量。
5. **单 seed、单配置（13M / 8L）**：量级与规模外推未做（按 methodology §10 需多档次小实验拟合）
6. 宽度约束是硬的：d=256 / V=205 时 null 空间仅 62 维、功能性读取 26 条；d=128 < V 时结构性为 0（无余量）
7. 结构性切分（只读前 r 维）→ 能量迁出读出子空间（W1）

## 6. 判据与判决实验设计

- **判据**：clean bpc（税）+ workspace 探针（null 能量占比 / functional_dirs）+ 校准（ECE）+ 传输与提交机制是否仍可用

## 7. 成本估计

- 13M 单臂 ~15–25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `W1-active-width`
2. `A9-old-info-retention`
3. `H4-head-trunk-relation`

## 9. 复现手段

1. .scratch/23-depth-ar/evidence @1586745
2. .scratch/25-tax-attribution/evidence @2e86924
3. axes/W-width/W4-readout-geometry/experiments/2610-gated-readout-13m-depthar/analysis.md @cba2f39
4. axes/W-width/W4-readout-geometry/experiments/2611-equalparams-linear-control-13m/analysis.md @ticket29
5. axes/W-width/W4-readout-geometry/experiments/2612-gated-readout-24k-13m/analysis.md @ticket29

## 备注

用户新想法所在格：读出不必线性映射到 unembedding 空间；SwiGLU/SiLU +
线性残差可保住 tied 桥（prob0/传输/提交仍走线性路径），只把质量读出解耦。
澄清：soft commit 的凸包安全不依赖读出线性度（Σp·E 恒在凸包内）；
真正会断的是「概率 = 状态在 embedding 基下的坐标」这一语义、
linear transport 的投影语义、以及 tied 表的梯度对称。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
