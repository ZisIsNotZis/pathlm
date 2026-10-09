---
id: F4-training-signal
axis: F
grid_dir: F-iteration/F4-training-signal
title: "迭代/重复的训练信号"
status: todo
judge: "目标量（组合一致性 / 重建质量）+ 税 + 是否反噬主任务"
cost: "13M 单臂 ~25 min"
tax: "consistency loss 中性（IX4 1.6589 ≈ C1 1.6549）；BF16 下 reward 类损失两次发散 → 必须 float32 + pre-step grad-norm gate"
benefit: "M0 微小模型上 consistency 收紧估计（cosine 0.9525→0.9882）"
verdict: "**已有**：consistency loss 中性；**重建/sufficiency 损失是压缩可恢复性的关键信号，但业界已有成熟前例（非空白格）**——我们的增量只能是「13M + 局部先验抗性任务 + 单因素受控」。**尚缺**：接显式载体（carry/摘要槽）× 训练信号是否有效（待补做）。"
deps: ["H3-composition", "A9-old-info-retention", "C6-depth-supervision"]
evidence: [".scratch/02-engine-m0/evidence/REPORT.md @27418ae", ".scratch/04-m1-runs/evidence @e38349f"]
---

<!-- AUTO:begin -->

# F4-training-signal · 迭代/重复的训练信号

> 轴：**F 迭代** ｜ 状态：**⬜ todo**

> **判读**：**已有**：consistency loss 中性；**重建/sufficiency 损失是压缩可恢复性的关键信号，但业界已有成熟前例（非空白格）**——我们的增量只能是「13M + 局部先验抗性任务 + 单因素受控」。**尚缺**：接显式载体（carry/摘要槽）× 训练信号是否有效（待补做）。

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
- **判读**：**已有**：consistency loss 中性；**重建/sufficiency 损失是压缩可恢复性的关键信号，但业界已有成熟前例（非空白格）**——我们的增量只能是「13M + 局部先验抗性任务 + 单因素受控」。**尚缺**：接显式载体（carry/摘要槽）× 训练信号是否有效（待补做）。

## 5. 失败路径与原因

1. consistency loss 在真实规模不改善聚合质量（中性）→「让组合被训练出来」的既有手段无效
2. negative/reward 项在 bf16 autocast 下不稳定（两次发散教会）
3. **重建/充分性损失的两臂实测（ticket 36-U2，13M/12k、合成关联回忆 + 窗口 16、无显式通道）**：罚则被完美学会（`recon.acc_symbol` 0.0049→0.9897，未训练头 0.0016）**但长程回忆没长**——λ=1.0 组 0.0137 vs λ=0 组 0.0161（**−0.244pp，CI 重叠**），同配置重跑更出现 4096 次提问 **0 命中** ⇒ 预登记判据「≥+3pp 且 CI 不重叠」**未过**，记「13M/12k 上不成立」。机制解释：该损失只要求「把刚掉出视野的那一格接力往前递」= 链式状态转移，不要求「背住整张关联表」；对照参照系：同任务同尺度**显式写通道**可达 0.0950（+8.13pp）⇒ **「给路径」优于「给信号」**。契约硬要求全部满足：λ=0 逐位等价（单测 + 运行期逐步 loss 一致）、fp32 头（autocast 关闭）、pre-step grad-norm gate（skips=0，gnorm_max 92.9/93.7/138.1）、活性每 200 步落盘

## 6. 判据与判决实验设计

- **判据**：目标量（组合一致性 / 重建质量）+ 税 + 是否反噬主任务

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：目标量（组合一致性 / 重建质量）+ 税 + 是否反噬主任务
- 成本：13M 单臂 ~25 min

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

**这条路业界已有成熟前例，不是空白格。**
压缩/重建类训练信号：Compressive Transformer（2019 的 compression loss）、
AutoCompressors / ICAE / RECOMP / 500xCompressor 类重建或蒸馏目标；DMC 报告 KV 压缩
需配继续预训练。（均标 `记忆·未核`——本环境联网不可用，需查原文；
见 `axes/M-carrier/M1-carrier-type/experiments/2617-prior-art-memory/PRIOR_ART.md`。）
⇒ 我们的增量只能是 **13M + 局部先验抗性任务（SCHEMA §19）+ 单因素受控 + 源对照臂**，
**不得**再声称「无人占位」。
（业界多数压缩工作不报「可重建性」⇒ 带检索腿的判据仍是我们的增量。）

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
