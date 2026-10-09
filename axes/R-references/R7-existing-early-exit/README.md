---
id: R7-existing-early-exit
axis: R
grid_dir: R-references/R7-existing-early-exit
title: "既有早退实现（CALM / LayerSkip / MoD）"
status: todo
judge: "同 FLOPs 质量 + 出口资产 + 税"
cost: "13M 单臂 ~25 min"
tax: "本地对照：dense 全深度监督（LayerSkip 式）+0.067（13M）/ 剂量恶化 @100M（+1.73 corrupt）"
benefit: "本地形态的增量 = 逐通道 carry 门 + 随机退出深度 + 定稿不变量（全深度重打分）"
verdict: "**已有**：本地仅有对照——dense 全深度监督（LayerSkip 式）+0.067（13M）/ 剂量恶化 @100M（+1.73 corrupt）。**尚缺**：同 FLOPs dense 早退（CALM/MoD）基线下本形态增量是否成立（业界实现结论须核实出处）（待补做）。"
deps: []
evidence: [".scratch/18-exit-probe/evidence @83bf1ac", ".scratch/21-exit-at-scale/evidence @9fe4cf1"]
---

<!-- AUTO:begin -->

# R7-existing-early-exit · 既有早退实现（CALM / LayerSkip / MoD）

> 轴：**R 参照线（必须打败的基线）** ｜ 状态：**⬜ todo**

> **判读**：**已有**：本地仅有对照——dense 全深度监督（LayerSkip 式）+0.067（13M）/ 剂量恶化 @100M（+1.73 corrupt）。**尚缺**：同 FLOPs dense 早退（CALM/MoD）基线下本形态增量是否成立（业界实现结论须核实出处）（待补做）。

## 1. 命题

深度自适应的既有实现——我们的形态必须说清增量在哪。

## 2. 选项枚举（本格全部可能取值）

1. `CALM（对比路由，需双模型）`
2. `LayerSkip（自推测早退，dense 监督）`
3. `MoD（逐 token 路由深度）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：本地形态的增量 = 逐通道 carry 门 + 随机退出深度 + 定稿不变量（全深度重打分）

**行业口径**：三者均有公开结果；共同点：需要深度维监督，且都报告质量-算力折中。

## 4. 结论与数字

- **税**：本地对照：dense 全深度监督（LayerSkip 式）+0.067（13M）/ 剂量恶化 @100M（+1.73 corrupt）
- **收益**：本地形态的增量 = 逐通道 carry 门 + 随机退出深度 + 定稿不变量（全深度重打分）
- **判读**：**已有**：本地仅有对照——dense 全深度监督（LayerSkip 式）+0.067（13M）/ 剂量恶化 @100M（+1.73 corrupt）。**尚缺**：同 FLOPs dense 早退（CALM/MoD）基线下本形态增量是否成立（业界实现结论须核实出处）（待补做）。

## 5. 失败路径与原因

1. dense 监督与 retry 精炼拮抗（C6）

## 6. 判据与判决实验设计

- **判据**：同 FLOPs 质量 + 出口资产 + 税

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：同 FLOPs 质量 + 出口资产 + 税
- 成本：13M 单臂 ~25 min

## 7. 成本估计

- 13M 单臂 ~25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. .scratch/18-exit-probe/evidence @83bf1ac
2. .scratch/21-exit-at-scale/evidence @9fe4cf1

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
