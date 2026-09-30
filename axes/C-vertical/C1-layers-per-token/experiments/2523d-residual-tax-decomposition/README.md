---
name: 2523d-residual-tax-decomposition
ticket: 23d
date: "2026-09-28"
grid: C1-layers-per-token
engine: "depth-ar（12k matched）"
budget: "13.7 min"
---

# 2523d-residual-tax-decomposition（ticket 23d）

> 主格：`C1-layers-per-token`｜关联格：`C5-refinement-operator`｜引擎：depth-ar（12k matched）｜预算：13.7 min

## 命题

残税 +0.121 的成分

## 干预

dense-B-12k / H1-12k 两个对照臂

## 结果

- +0.121 = {dense 欠拟合 −0.209 + aux 残余 +0.108 + 门/填充固有 +0.222}（恒等闭合）
- matched-step 真税 @12k = +0.3295（较 3.6k 收缩 54%）
- 100M 4.4-epoch 外推 ≈ +0.21（区间 +0.15–0.23）

## 失败路径与原因

1. 「残税主要是堆栈固有」证伪：dense 锚点自身欠拟合掩盖近 2/3

## 证据与复现

- `.scratch/23-depth-ar/evidence @5d5a983`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
