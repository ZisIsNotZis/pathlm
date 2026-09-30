---
id: C2-layer-selection-order
axis: C
grid_dir: C-vertical/C2-layer-selection-order
title: "选哪些层与执行顺序"
status: judged
judge: "clean bpc（税）+ 组合兼容性"
cost: "13M 单臂 ~25 min"
tax: "skip +0.175（最贵深度元素）；redo +0.044；层序 shuffle +0.545（全项目最贵元素）"
benefit: "skip 是最好的 TTS 基底（K=8 −0.047，DIVL1 再 +0.023）"
verdict: "元素税定律的原始出处：破坏型元素贵，保留型近免费。"
deps: []
evidence: [".scratch/06-m2-sweep/evidence @a03a579", ".scratch/09-m3-interactions/evidence @39b1838", ".scratch/08-tts-probe/evidence @aa97bcf"]
---

<!-- AUTO:begin -->

# C2-layer-selection-order · 选哪些层与执行顺序

> 轴：**C 垂直机制（深度）** ｜ 状态：**✅ judged**

> **判读**：元素税定律的原始出处：破坏型元素贵，保留型近免费。

## 1. 命题

层可以被跳过、重排、重复到什么程度。

## 2. 选项枚举（本格全部可能取值）

1. `只跑前缀`
2. `任意子集（skip）`
3. `重排（shuffle-locality）`
4. `重复（redo）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：skip 是最好的 TTS 基底（K=8 −0.047，DIVL1 再 +0.023）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：skip +0.175（最贵深度元素）；redo +0.044；层序 shuffle +0.545（全项目最贵元素）
- **收益**：skip 是最好的 TTS 基底（K=8 −0.047，DIVL1 再 +0.023）
- **判读**：元素税定律的原始出处：破坏型元素贵，保留型近免费。

## 5. 失败路径与原因

1. shuffle 硬边界：locality 0.5 训练、eval ≤0.5 可容忍、全随机崩到 3.84（p_skip 训练/eval 不一致的同类问题）
2. shuffle 是组合毒药：任何含 shuffle 的组合落到 2.2+ bpc，与修复拮抗（repair 51%→38%）
3. TTS：确定性路径 checkpoint 的 K-path 集成恰好得 0

## 6. 判据与判决实验设计

- **判据**：clean bpc（税）+ 组合兼容性

## 7. 成本估计

- 13M 单臂 ~25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. .scratch/06-m2-sweep/evidence @a03a579
2. .scratch/09-m3-interactions/evidence @39b1838
3. .scratch/08-tts-probe/evidence @aa97bcf

## 备注

层序携带 ~0.5 bpc —— 「连层的执行顺序都是先验」。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
