---
id: R5-bestofn-verifier
axis: R
grid_dir: R-references/R5-bestofn-verifier
title: "best-of-n + verifier"
status: todo
judge: "同 FLOPs 质量 + verifier 校准"
cost: "纯推理，分钟级"
tax: "—"
benefit: "本地侧证：共享主干的多线增益可忽略（G5）"
verdict: "**尚未核实／待立项**：业界基线结论无核实出处；本地仅有侧证——共享主干的多线增益可忽略（G5）。"
deps: []
evidence: [".scratch/08-tts-probe/evidence @aa97bcf"]
---

<!-- AUTO:begin -->

# R5-bestofn-verifier · best-of-n + verifier

> 轴：**R 参照线（必须打败的基线）** ｜ 状态：**⬜ todo**

> **判读**：**尚未核实／待立项**：业界基线结论无核实出处；本地仅有侧证——共享主干的多线增益可忽略（G5）。

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
- **判读**：**尚未核实／待立项**：业界基线结论无核实出处；本地仅有侧证——共享主干的多线增益可忽略（G5）。

## 5. 失败路径与原因

1. 深度集成失败（置信头不排序逐 token 可靠性）

## 6. 判据与判决实验设计

- **判据**：同 FLOPs 质量 + verifier 校准

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：同 FLOPs 质量 + verifier 校准
- 成本：纯推理，分钟级

## 7. 成本估计

- 纯推理，分钟级

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. .scratch/08-tts-probe/evidence @aa97bcf

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
