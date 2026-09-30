---
id: R5-bestofn-verifier
axis: R
grid_dir: R-references/R5-bestofn-verifier
title: "best-of-n + verifier"
status: reference
judge: "同 FLOPs 质量 + verifier 校准"
cost: "纯推理，分钟级"
tax: "—"
benefit: "本地侧证：共享主干的多线增益可忽略（G5）"
verdict: "—"
deps: []
evidence: [".scratch/08-tts-probe/evidence @aa97bcf"]
---

<!-- AUTO:begin -->

# R5-bestofn-verifier · best-of-n + verifier

> 轴：**R 参照线（必须打败的基线）** ｜ 状态：**🏭 reference**

## 1. 命题

用额外算力做候选与重排——test-time compute 的标准答案。

## 2. 选项枚举（本格全部可能取值）

1. `自一致性投票`
2. `学习式 verifier`
3. `规则 verifier`
4. `多线重排`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：本地侧证：共享主干的多线增益可忽略（G5）

**行业口径**：标准 test-time compute 手段；与 G5 的多线机制直接竞争。

## 4. 结论与数字

- **税**：—
- **收益**：本地侧证：共享主干的多线增益可忽略（G5）

## 5. 失败路径与原因

1. 深度集成失败（置信头不排序逐 token 可靠性）

## 6. 判据与判决实验设计

- **判据**：同 FLOPs 质量 + verifier 校准

## 7. 成本估计

- 纯推理，分钟级

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. .scratch/08-tts-probe/evidence @aa97bcf

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
