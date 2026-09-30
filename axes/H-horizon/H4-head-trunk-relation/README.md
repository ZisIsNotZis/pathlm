---
id: H4-head-trunk-relation
axis: H
grid_dir: H-horizon/H4-head-trunk-relation
title: "头与主干的关系"
status: judged
judge: "税（挂头是否扰动主干）+ 头自身质量 + 校准"
cost: "13M 单臂 ~5–25 min；冻结主干可复用已有 checkpoint → 单臂 ~5 min"
tax: "probe 式头 @100M ≈0（1.5928 vs 1.5871）；停梯度使浅层读出贵（13M depth6 +0.93）"
benefit: "保 retry（−0.0098）且门控更强（−0.029 @100M）；MTP 头是 d×d 变换、无 vocab 参数（~free 的基础设施）"
verdict: "挂头是免费旁路——这是「主干不动的判断头可以免费」的实测出处。"
deps: ["C1-layers-per-token", "W4-readout-geometry"]
evidence: [".scratch/18-exit-probe/evidence @83bf1ac", ".scratch/21-exit-at-scale/evidence @9fe4cf1"]
---

<!-- AUTO:begin -->

# H4-head-trunk-relation · 头与主干的关系

> 轴：**H 视界（一次前向产生几个位置）** ｜ 状态：**✅ judged**

> **判读**：挂头是免费旁路——这是「主干不动的判断头可以免费」的实测出处。

## 1. 命题

额外的头（MTP/读出头/判断头）是联合训练还是挂在冻结主干上。

## 2. 选项枚举（本格全部可能取值）

1. `联合训练`
2. `冻结主干只训头`
3. `共享 embedding（tied）`
4. `独立解码头`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：保 retry（−0.0098）且门控更强（−0.029 @100M）；MTP 头是 d×d 变换、无 vocab 参数（~free 的基础设施）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：probe 式头 @100M ≈0（1.5928 vs 1.5871）；停梯度使浅层读出贵（13M depth6 +0.93）
- **收益**：保 retry（−0.0098）且门控更强（−0.029 @100M）；MTP 头是 d×d 变换、无 vocab 参数（~free 的基础设施）
- **判读**：挂头是免费旁路——这是「主干不动的判断头可以免费」的实测出处。

## 5. 失败路径与原因

1. probe 式头改善 retry 但浅层读出质量贵 → 「保 retry」与「exit 质量」是同一梯度的两面（与 C6 同一条定律）

## 6. 判据与判决实验设计

- **判据**：税（挂头是否扰动主干）+ 头自身质量 + 校准

## 7. 成本估计

- 13M 单臂 ~5–25 min；冻结主干可复用已有 checkpoint → 单臂 ~5 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `C1-layers-per-token`
2. `W4-readout-geometry`

## 9. 复现手段

1. .scratch/18-exit-probe/evidence @83bf1ac
2. .scratch/21-exit-at-scale/evidence @9fe4cf1

## 备注

最便宜的真空白：拿一个冻结的 stock 小模型只训跳跃头，
可一次排 3–5 个变体，且完全不需要训练主干。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
