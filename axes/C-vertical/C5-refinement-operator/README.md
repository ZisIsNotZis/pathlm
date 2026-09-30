---
id: C5-refinement-operator
axis: C
grid_dir: C-vertical/C5-refinement-operator
title: "精化算子"
status: judged
judge: "深度-bpc 曲线单调性 + 出口质量 + 税"
cost: "13M 单臂 ~25 min"
tax: "carry 门 + 填充 + 退出堆栈合计是形态税的主体（同引擎 +0.4362 @100M）"
benefit: "门行为健康（浅层开、深层微开；@100M 全层从 carry 初始化收小）；门=逐通道，天然给出「哪些通道在精化」的可观测性"
verdict: "携带门是形态税的承载者之一，也是「任意深度可读」的结构来源。"
deps: ["C1-layers-per-token", "H2-training-fill"]
evidence: [".scratch/23-depth-ar/evidence @1586745", ".scratch/24-depth-ar-100m/evidence @7725891", ".scratch/25-tax-attribution/evidence @2e86924"]
---

<!-- AUTO:begin -->

# C5-refinement-operator · 精化算子

> 轴：**C 垂直机制（深度）** ｜ 状态：**✅ judged**

> **判读**：携带门是形态税的承载者之一，也是「任意深度可读」的结构来源。

## 1. 命题

每层如何把状态往前推一步。

## 2. 选项枚举（本格全部可能取值）

1. `残差 + MLP（标准）`
2. `携带门乘 delta（h+g⊙Refine(h)，本项目）`
3. `并行支路 + Merge（MoD 式）`
4. `条件化（AdaLN / 时间步条件）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：门行为健康（浅层开、深层微开；@100M 全层从 carry 初始化收小）；门=逐通道，天然给出「哪些通道在精化」的可观测性

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：carry 门 + 填充 + 退出堆栈合计是形态税的主体（同引擎 +0.4362 @100M）
- **收益**：门行为健康（浅层开、深层微开；@100M 全层从 carry 初始化收小）；门=逐通道，天然给出「哪些通道在精化」的可观测性
- **判读**：携带门是形态税的承载者之一，也是「任意深度可读」的结构来源。

## 5. 失败路径与原因

1. 「渐进开门」在 100M 未出现（门从 0.119 收小，等权/2× 同款，非配方现象）→ 结构假设之一未兑现
2. stop-grad 浅头（probe 式）恶化至 2.7129、浅层门不再打开 → 浅头 stop-grad 化被排除

## 6. 判据与判决实验设计

- **判据**：深度-bpc 曲线单调性 + 出口质量 + 税

## 7. 成本估计

- 13M 单臂 ~25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `C1-layers-per-token`
2. `H2-training-fill`

## 9. 复现手段

1. .scratch/23-depth-ar/evidence @1586745
2. .scratch/24-depth-ar-100m/evidence @7725891
3. .scratch/25-tax-attribution/evidence @2e86924

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
