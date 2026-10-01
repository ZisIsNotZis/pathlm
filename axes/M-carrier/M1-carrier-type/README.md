---
id: M1-carrier-type
axis: M
grid_dir: M-carrier/M1-carrier-type
title: "状态载体类型"
status: partial
judge: "保留（needle 消融）+ 内存/步长 + 吞吐"
cost: "13M 单臂 ~25–40 min"
tax: "—（KV 为基底）"
benefit: "per-layer KV + 位置散点填充实现了「变深度 + 不重 prefill」（与并行 forward 逐位一致）"
verdict: "只有 KV 载体；**ticket 34 造出「不靠语言先验」的长程天平并给出关键诊断：我们 13M 主干的长程能力是「分任务型」的——关联回忆 0.0061、变距离回忆 0.005–0.010（≈或低于 chance）⇒ 检索类彻底缺席；复制 0.1333（距离 223 仍 0.118，毁源→0.0125）与 induction 0.0586（毁源→chance）⇒ 真实但弱。**载体是「能否丢/重算」的决定因素。"
deps: ["M2-carrier-compression", "A9-old-info-retention", "F1-iteration-object"]
evidence: [".scratch/03-engine-m1/issues @e38349f", "axes/M-carrier/M1-carrier-type/experiments/2617-longrange-task-battery/analysis.md @ticket34", "axes/M-carrier/M1-carrier-type/experiments/2617-prior-art-memory/PRIOR_ART.md @ticket34"]
---

<!-- AUTO:begin -->

# M1-carrier-type · 状态载体类型

> 轴：**M 状态载体** ｜ 状态：**◐ partial**

> **判读**：只有 KV 载体；**ticket 34 造出「不靠语言先验」的长程天平并给出关键诊断：我们 13M 主干的长程能力是「分任务型」的——关联回忆 0.0061、变距离回忆 0.005–0.010（≈或低于 chance）⇒ 检索类彻底缺席；复制 0.1333（距离 223 仍 0.118，毁源→0.0125）与 induction 0.0586（毁源→chance）⇒ 真实但弱。**载体是「能否丢/重算」的决定因素。

## 1. 命题

跨步信息存在哪里。

## 2. 选项枚举（本格全部可能取值）

1. `KV cache（本项目唯一）`
2. `递归状态（SSM/RNN）`
3. `显式记忆槽`
4. `外部存储（检索）`
5. `权重（fast-weight）`
6. `无`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：per-layer KV + 位置散点填充实现了「变深度 + 不重 prefill」（与并行 forward 逐位一致）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：—（KV 为基底）
- **收益**：per-layer KV + 位置散点填充实现了「变深度 + 不重 prefill」（与并行 forward 逐位一致）
- **判读**：只有 KV 载体；**ticket 34 造出「不靠语言先验」的长程天平并给出关键诊断：我们 13M 主干的长程能力是「分任务型」的——关联回忆 0.0061、变距离回忆 0.005–0.010（≈或低于 chance）⇒ 检索类彻底缺席；复制 0.1333（距离 223 仍 0.118，毁源→0.0125）与 induction 0.0586（毁源→chance）⇒ 真实但弱。**载体是「能否丢/重算」的决定因素。

## 5. 失败路径与原因

1. 早期缓存 bug：逐位置早退导致 per-layer cache 长度失步（已修为 per-layer 位置列表；持续单测）
2. **长程检索能力缺席（ticket 34-A 仪器实测）**：关联回忆 0.0061、变距离回忆六档 0.005–0.010，均≈或低于 chance 且无距离依赖 ⇒ 此前所有「没搬」读数，至少相当部分来自**模型本来就没有该能力**，而非「机制搬不动」（SCHEMA §19）
3. **仪器落地（`tools/evalkit/longrange_tasks.py`）**：四任务 + chance/局部先验基线 + 距离曲线 + `recall_under_visibility`（复用 retention 掩码）+ 毁源对照臂；自证：局部先验基线 0.0098–0.0217（chance 上限 0.0469）、毁源/打乱答案均掉回 chance、前向逐位一致
4. 不确定项：induction 每点 n=128 不支持斜率；assoc/var 低分含分布外成分（**不得**外推成「训练也学不会」）；copy 13% 的机制未定位

## 6. 判据与判决实验设计

- **判据**：保留（needle 消融）+ 内存/步长 + 吞吐

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：保留（needle 消融）+ 内存/步长 + 吞吐
- 成本：13M 单臂 ~25–40 min

## 7. 成本估计

- 13M 单臂 ~25–40 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `M2-carrier-compression`
2. `A9-old-info-retention`
3. `F1-iteration-object`

## 9. 复现手段

1. .scratch/03-engine-m1/issues @e38349f
2. axes/M-carrier/M1-carrier-type/experiments/2617-longrange-task-battery/analysis.md @ticket34
3. axes/M-carrier/M1-carrier-type/experiments/2617-prior-art-memory/PRIOR_ART.md @ticket34

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
