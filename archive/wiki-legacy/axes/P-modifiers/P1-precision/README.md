---
id: P1-precision
axis: P
grid_dir: P-modifiers/P1-precision
title: "精度（修饰符）"
status: judged
judge: "训练稳定性（发散 → 必须可失败地抓）"
cost: "与实验同价"
tax: "—"
benefit: "fp32 readout 头是 prob0/传输/提交的地基；强制 fp32 让深度 logits 稳定"
verdict: "reward/判据类损失必须 float32 + pre-step grad-norm gate。"
deps: []
evidence: [".scratch/09-m3-interactions/evidence @39b1838"]
---

<!-- AUTO:begin -->

# P1-precision · 精度（修饰符）

> 轴：**P 修饰符（不是轴）** ｜ 状态：**✅ judged**

> **判读**：reward/判据类损失必须 float32 + pre-step grad-norm gate。

## 1. 命题

哪些部分必须高精度——决定机制能否稳定训练。

## 2. 选项枚举（本格全部可能取值）

1. `bf16 autocast（主体）`
2. `fp32 读出/损失头（本项目）`
3. `全 fp32`
4. `量化内核（见 R1）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：fp32 readout 头是 prob0/传输/提交的地基；强制 fp32 让深度 logits 稳定

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：—
- **收益**：fp32 readout 头是 prob0/传输/提交的地基；强制 fp32 让深度 logits 稳定
- **判读**：reward/判据类损失必须 float32 + pre-step grad-norm gate。

## 5. 失败路径与原因

1. bf16 autocast 下的 negative/reward 项两次发散（capped −JS 多样性压力）→ 必须 float32 + pre-step grad-norm gate

## 6. 判据与判决实验设计

- **判据**：训练稳定性（发散 → 必须可失败地抓）

## 7. 成本估计

- 与实验同价

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. .scratch/09-m3-interactions/evidence @39b1838

## 备注

这是「稳定性工程」定律（playbook 第 5 条）的出处。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
