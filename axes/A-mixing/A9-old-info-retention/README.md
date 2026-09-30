---
id: A9-old-info-retention
axis: A
grid_dir: A-mixing/A9-old-info-retention
title: "旧信息处理策略"
status: partial
judge: "KV 消融保留曲线（真删，测损伤与可重建性）"
cost: "13M 单臂 ~25 min"
tax: "硬丢 +0.058~0.121（A3）"
benefit: "锚豁免使 needle 76.8→93.3%（+距离惩罚）"
verdict: "保留 / 硬丢 / 锚豁免已测；压缩、递归、外置三类零经验。"
deps: ["M1-carrier-type", "M2-carrier-compression", "A10-compression-trigger"]
evidence: [".scratch/07-x1-x3/evidence @7b47555", ".scratch/13-four-experiments/evidence @9abc181"]
---

<!-- AUTO:begin -->

# A9-old-info-retention · 旧信息处理策略

> 轴：**A 序列混合机制** ｜ 状态：**◐ partial**

> **判读**：保留 / 硬丢 / 锚豁免已测；压缩、递归、外置三类零经验。

## 1. 命题

已经读过的远距离信息，留什么、丢什么、压成什么。

## 2. 选项枚举（本格全部可能取值）

1. `保留原始 KV`
2. `硬丢（驱逐）`
3. `锚点/ sink 豁免`
4. `压成 summary 槽（压缩记忆）`
5. `压成递归状态（SSM/RNN 载体）`
6. `外置检索（RAG / memorizing transformer）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：锚豁免使 needle 76.8→93.3%（+距离惩罚）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：硬丢 +0.058~0.121（A3）
- **收益**：锚豁免使 needle 76.8→93.3%（+距离惩罚）
- **判读**：保留 / 硬丢 / 锚豁免已测；压缩、递归、外置三类零经验。

## 5. 失败路径与原因

1. 损坏 × 精确复制目标冲突（三路豁免全败）——「保住信息」与「容错修复」是互相抢容量的两个目标

## 6. 判据与判决实验设计

- **判据**：KV 消融保留曲线（真删，测损伤与可重建性）

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：KV 消融保留曲线（真删，测损伤与可重建性）
- 成本：13M 单臂 ~25 min

## 7. 成本估计

- 13M 单臂 ~25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `M1-carrier-type`
2. `M2-carrier-compression`
3. `A10-compression-trigger`

## 9. 复现手段

1. .scratch/07-x1-x3/evidence @7b47555
2. .scratch/13-four-experiments/evidence @9abc181

## 备注

「右移压缩」主线落点。判据必须是保留曲线；入场费上界 = 硬窗的
+0.058~0.121（13M）。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
