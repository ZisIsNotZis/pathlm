---
id: C6-depth-supervision
axis: C
grid_dir: C-vertical/C6-depth-supervision
title: "训练期深度分布与监督密度"
status: todo
judge: "出口资产（tf@k 接受率）+ 税 + 中间深度可读性"
cost: "13M 单臂 ~25 min；100M ~32–70 min"
tax: "全深度等权 +0.067（13M）；dense 0.2 剂量的税为负（EX2 1.6097 vs 无 dense 1.6698，aux 起正则作用）"
benefit: "形态臂 tf@4 0.97 / tf@6 0.99；退火配方 C 等效验证吞吐 122.8 vs A 112.9 tok/s（+9%）"
verdict: "**已有**：出口资产只能由深度维监督造出来——这是本项目最有迁移价值的一条结论。**尚缺**：同引擎税 + 出口资产是否需门堆栈（与 C5 共用一次 eval）（待补做）。"
deps: ["C1-layers-per-token", "F2-return-channel"]
evidence: [".scratch/14-allocator/evidence @9954425", ".scratch/18-exit-probe/evidence @83bf1ac", ".scratch/21-exit-at-scale/evidence @9fe4cf1", ".scratch/24-depth-ar-100m/evidence @7725891"]
---

<!-- AUTO:begin -->

# C6-depth-supervision · 训练期深度分布与监督密度

> 轴：**C 垂直机制（深度）** ｜ 状态：**⬜ todo**

> **判读**：**已有**：出口资产只能由深度维监督造出来——这是本项目最有迁移价值的一条结论。**尚缺**：同引擎税 + 出口资产是否需门堆栈（与 C5 共用一次 eval）（待补做）。

## 1. 命题

训练时让哪些深度被监督——决定哪些深度在推理时可用。

## 2. 选项枚举（本格全部可能取值）

1. `仅最终深度`
2. `全深度并行监督（等权/加权）`
3. `随机退出深度（本项目）`
4. `退火课程（末段收紧）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：形态臂 tf@4 0.97 / tf@6 0.99；退火配方 C 等效验证吞吐 122.8 vs A 112.9 tok/s（+9%）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：全深度等权 +0.067（13M）；dense 0.2 剂量的税为负（EX2 1.6097 vs 无 dense 1.6698，aux 起正则作用）
- **收益**：形态臂 tf@4 0.97 / tf@6 0.99；退火配方 C 等效验证吞吐 122.8 vs A 112.9 tok/s（+9%）
- **判读**：**已有**：出口资产只能由深度维监督造出来——这是本项目最有迁移价值的一条结论。**尚缺**：同引擎税 + 出口资产是否需门堆栈（与 C5 共用一次 eval）（待补做）。

## 5. 失败路径与原因

1. 纯 dense trunk（无深度监督）没有任何浅层截断资产：tf@4 仅 0.120（vs 形态臂 0.97）
2. dense 全深度监督杀 retry 精炼（ticket 14 ALLOC）：整形与自适应是同一梯度的两面，两全证伪
3. 剂量响应随规模变化：dense 0.2 在 13M 温和、@100M corrupt 崩溃（+1.73）

## 6. 判据与判决实验设计

- **判据**：出口资产（tf@k 接受率）+ 税 + 中间深度可读性

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：出口资产（tf@k 接受率）+ 税 + 中间深度可读性
- 成本：13M 单臂 ~25 min；100M ~32–70 min

## 7. 成本估计

- 13M 单臂 ~25 min；100M ~32–70 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `C1-layers-per-token`
2. `F2-return-channel`

## 9. 复现手段

1. .scratch/14-allocator/evidence @9954425
2. .scratch/18-exit-probe/evidence @83bf1ac
3. .scratch/21-exit-at-scale/evidence @9fe4cf1
4. .scratch/24-depth-ar-100m/evidence @7725891

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
