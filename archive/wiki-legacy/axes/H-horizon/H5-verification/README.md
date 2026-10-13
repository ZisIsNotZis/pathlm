---
id: H5-verification
axis: H
grid_dir: H-horizon/H5-verification
title: "验证机制"
status: judged
judge: "逐位一致（vs 贪心）+ 端到端 tok/s + 质量不劣"
cost: "13M 单臂 ~25 min（含 5 个可失败单测）"
tax: "验证本身不额外加税（复用同一次前向）"
benefit: "Medusa 批量验证逐位 == 逐位置贪心（强制全接受/强制拒绝/窗口驱逐三个专门测试）；稳态 2.40 tok/fwd"
verdict: "批量验证可用且逐位一致；定稿不变量（全深度重打分）是本形态的正确性保障。"
deps: ["H1-outputs-per-pass", "C1-layers-per-token"]
evidence: [".scratch/13-four-experiments/evidence/mtp3 @9abc181", ".scratch/20-spec-scale/evidence @a864bcf"]
---

<!-- AUTO:begin -->

# H5-verification · 验证机制

> 轴：**H 视界（一次前向产生几个位置）** ｜ 状态：**✅ judged**

> **判读**：批量验证可用且逐位一致；定稿不变量（全深度重打分）是本形态的正确性保障。

## 1. 命题

草稿如何被验证——决定 spec 的正确性与成本。

## 2. 选项枚举（本格全部可能取值）

1. `不验证`
2. `批量验证（一次前向验证全部草稿）`
3. `置信门控验证`
4. `全深度重打分（定稿不变量）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：Medusa 批量验证逐位 == 逐位置贪心（强制全接受/强制拒绝/窗口驱逐三个专门测试）；稳态 2.40 tok/fwd

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：验证本身不额外加税（复用同一次前向）
- **收益**：Medusa 批量验证逐位 == 逐位置贪心（强制全接受/强制拒绝/窗口驱逐三个专门测试）；稳态 2.40 tok/fwd
- **判读**：批量验证可用且逐位一致；定稿不变量（全深度重打分）是本形态的正确性保障。

## 5. 失败路径与原因

1. @100M 逐位一致失效（fp 平局翻转，1/400 token）→ 正确性判据必须按规模设定
2. exit × spec 在本引擎不可组合（批量验证必须全深度，逐位置深度会使缓存失步）

## 6. 判据与判决实验设计

- **判据**：逐位一致（vs 贪心）+ 端到端 tok/s + 质量不劣

## 7. 成本估计

- 13M 单臂 ~25 min（含 5 个可失败单测）

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `H1-outputs-per-pass`
2. `C1-layers-per-token`

## 9. 复现手段

1. .scratch/13-four-experiments/evidence/mtp3 @9abc181
2. .scratch/20-spec-scale/evidence @a864bcf

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
