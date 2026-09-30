---
id: F1-iteration-object
axis: F
grid_dir: F-iteration/F1-iteration-object
title: "迭代对象"
status: partial
judge: "税 + 每轮边际收益曲线 + 成本（forwards/FLOPs）"
cost: "13M 单臂 ~25–40 min"
tax: "latent retry C 族 1.6535–1.6856 vs B0 1.5074 = +0.146…+0.178（10–100× 任何测得底）"
benefit: "门控重试在修复侧有真价值：低置信 15% 开火把 r1→r2 从 −0.0193 翻到 +0.0211（C1M +0.1284）；12 层上 retry 更值（L-loop × depth 协同）"
verdict: "前两项已收口；后四项零经验。"
deps: ["F2-return-channel", "F3-stopping", "G1-selector"]
evidence: [".scratch/11-retry-matrix-validity/VERDICTS.md @d6a9898", ".scratch/12-overnight/evidence @216a00a", ".scratch/14-allocator/evidence @9954425", ".scratch/17-slider-at-scale/evidence @25704b8"]
---

<!-- AUTO:begin -->

# F1-iteration-object · 迭代对象

> 轴：**F 迭代** ｜ 状态：**◐ partial**

> **判读**：前两项已收口；后四项零经验。

## 1. 命题

重复施加在什么之上。

## 2. 选项枚举（本格全部可能取值）

1. `同 token 状态（latent retry，本项目）`
2. `同段多遍（AR 多轮）`
3. `整段去噪（扩散式并行）`
4. `候选集（best-of-n）`
5. `隐式推理步（latent CoT / pause token，无输出监督）`
6. `权重（fast-weight / TTT）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：门控重试在修复侧有真价值：低置信 15% 开火把 r1→r2 从 −0.0193 翻到 +0.0211（C1M +0.1284）；12 层上 retry 更值（L-loop × depth 协同）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：latent retry C 族 1.6535–1.6856 vs B0 1.5074 = +0.146…+0.178（10–100× 任何测得底）
- **收益**：门控重试在修复侧有真价值：低置信 15% 开火把 r1→r2 从 −0.0193 翻到 +0.0211（C1M +0.1284）；12 层上 retry 更值（L-loop × depth 协同）
- **判读**：前两项已收口；后四项零经验。

## 5. 失败路径与原因

1. 6 个 latent 回传 cell 跨度仅 0.011 bpc < 种子散布 0.0138 → 全部排序结论撤回（ticket 11）：支持的是「多一遍」，不是「回传几何」
2. retry 曲线 ~3–4 轮饱和；token retry 未门控 −4.6pp（量化掉不确定性）
3. 门控重试的质量杠杆在 100M 消失（τ 轴平坦 ±0.002）

## 6. 判据与判决实验设计

- **判据**：税 + 每轮边际收益曲线 + 成本（forwards/FLOPs）

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：税 + 每轮边际收益曲线 + 成本（forwards/FLOPs）
- 成本：13M 单臂 ~25–40 min

## 7. 成本估计

- 13M 单臂 ~25–40 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `F2-return-channel`
2. `F3-stopping`
3. `G1-selector`

## 9. 复现手段

1. .scratch/11-retry-matrix-validity/VERDICTS.md @d6a9898
2. .scratch/12-overnight/evidence @216a00a
3. .scratch/14-allocator/evidence @9954425
4. .scratch/17-slider-at-scale/evidence @25704b8

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
