---
id: P6-data-construction
axis: P
grid_dir: P-modifiers/P6-data-construction
title: "数据与任务构造（修饰符）"
status: judged
judge: "结论在「任务依赖」上的稳健性"
cost: "与实验同价"
tax: "—"
benefit: "损坏增广支撑修复侧结论（I 族）；needle v2 构造可测（v1 信号饥饿测到 0.0）"
verdict: "任务构造本身是本项目最大的单点偏差来源（enwik8 局部统计极强）。"
deps: []
evidence: [".scratch/05-m1-followup/evidence @480d50b", ".scratch/07-x1-x3/evidence @7b47555"]
---

<!-- AUTO:begin -->

# P6-data-construction · 数据与任务构造（修饰符）

> 轴：**P 修饰符（不是轴）** ｜ 状态：**✅ judged**

> **判读**：任务构造本身是本项目最大的单点偏差来源（enwik8 局部统计极强）。

## 1. 命题

训练/评测数据如何构造——决定结论是否可迁移。

## 2. 选项枚举（本格全部可能取值）

1. `真实文本（char 级 enwik8）`
2. `损坏增广（wrong/mask/noise）`
3. `needle 构造`
4. `去噪任务`
5. `合成任务`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：损坏增广支撑修复侧结论（I 族）；needle v2 构造可测（v1 信号饥饿测到 0.0）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：—
- **收益**：损坏增广支撑修复侧结论（I 族）；needle v2 构造可测（v1 信号饥饿测到 0.0）
- **判读**：任务构造本身是本项目最大的单点偏差来源（enwik8 局部统计极强）。

## 5. 失败路径与原因

1. needle v1 随机填充构造使每 batch 只有 1 个监督 token → 信号饥饿，全桶 0.0（假失败）
2. enwik8 字符级局部统计极强 → 距离惩罚的 bpc 改善与长程检索退化可以同时发生（A4 陷阱）
3. prefill/解码的总长受绝对位置嵌入限制（A8）

## 6. 判据与判决实验设计

- **判据**：结论在「任务依赖」上的稳健性

## 7. 成本估计

- 与实验同价

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. .scratch/05-m1-followup/evidence @480d50b
2. .scratch/07-x1-x3/evidence @7b47555

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
