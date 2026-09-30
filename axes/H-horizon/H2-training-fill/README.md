---
id: H2-training-fill
axis: H
grid_dir: H-horizon/H2-training-fill
title: "训练时未来位置的内容填充"
status: judged
judge: "税 + 出口/填入位置的可读性 + 提交质量（rescore）"
cost: "13M 单臂 ~15–25 min"
tax: "mask/wrong-token 替换 +0.059 / +0.127；embedding noise ±0.000；纯噪声 latent +0.063；self-fill（fill）1.8582 vs nofill 1.9438（+0.0856）"
benefit: "proj-fill（逐层恒等初始化 adapter + fill）1.8356 = 13M matched-step 族内最优（全深度逐深度支配 fill）"
verdict: "保留式填充（self-fill）胜出；PAD 与错误 token 都更贵；软提交有 OOD 污染。"
deps: ["C1-layers-per-token", "H5-verification", "P1-precision"]
evidence: [".scratch/06-m2-sweep/evidence @a03a579", ".scratch/23-depth-ar/evidence @6b27b5e", ".scratch/24-depth-ar-100m/evidence @7cdcdaa"]
---

<!-- AUTO:begin -->

# H2-training-fill · 训练时未来位置的内容填充

> 轴：**H 视界（一次前向产生几个位置）** ｜ 状态：**✅ judged**

> **判读**：保留式填充（self-fill）胜出；PAD 与错误 token 都更贵；软提交有 OOD 污染。

## 1. 命题

右侧（未来/已退出）位置在训练时被什么内容填充。

## 2. 选项枚举（本格全部可能取值）

1. `不存在（纯 AR 无右侧）`
2. `PAD / [mask]`
3. `错误 token`
4. `self-fill（用自己上一轮的预测 / 已退出位置的冻结表征）`
5. `软 latent（Σp·E，soft 提交）`
6. `随机 latent`
7. `纯噪声（去噪式）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：proj-fill（逐层恒等初始化 adapter + fill）1.8356 = 13M matched-step 族内最优（全深度逐深度支配 fill）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：mask/wrong-token 替换 +0.059 / +0.127；embedding noise ±0.000；纯噪声 latent +0.063；self-fill（fill）1.8582 vs nofill 1.9438（+0.0856）
- **收益**：proj-fill（逐层恒等初始化 adapter + fill）1.8356 = 13M matched-step 族内最优（全深度逐深度支配 fill）
- **判读**：保留式填充（self-fill）胜出；PAD 与错误 token 都更贵；软提交有 OOD 污染。

## 5. 失败路径与原因

1. 首验判「填充是负分量 +0.167」→ 23e matched 步数下反转：是欠拟合伪影 + 对照混淆（当时 nofill 臂同时无随机退出）
2. soft 提交 OOD 污染 context 三度复现（rescore 峰值：6.9 vs hard 1.2 首验；8.42 vs 0.79 @100M；C 臂 3.36 vs 0.74）→ soft 默认待重验

## 6. 判据与判决实验设计

- **判据**：税 + 出口/填入位置的可读性 + 提交质量（rescore）

## 7. 成本估计

- 13M 单臂 ~15–25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `C1-layers-per-token`
2. `H5-verification`
3. `P1-precision`

## 9. 复现手段

1. .scratch/06-m2-sweep/evidence @a03a579
2. .scratch/23-depth-ar/evidence @6b27b5e
3. .scratch/24-depth-ar-100m/evidence @7cdcdaa

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
