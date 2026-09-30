---
name: 2519-asymptotic-tax
ticket: 19-asym-100m
date: "2026-09-18"
grid: A2-full-attention
engine: "100M，4.4 epoch，2 seeds"
budget: "4.4 epoch × 100M"
---

# 2519-asymptotic-tax（ticket 19-asym-100m）

> 主格：`A2-full-attention`｜关联格：`F3-stopping`｜引擎：100M，4.4 epoch，2 seeds｜预算：4.4 epoch × 100M

## 命题

渐近口径下税是否收缩

## 干预

4.4-epoch 训练 + 双 seed

## 结果

- 税 +0.126 → +0.082（曲线单调降，收缩 35%）
- 等训练口径 13M +0.122 → 100M +0.082——税真正随规模下降
- 能力溢价随训练扩大（corrupt gap 1.54→1.89）；B0_asym 1.341 ≈ 13M 渐近 1.346（data-limited）

## 失败路径与原因

1. 快照口径的「不随规模收缩」全是 1.1-epoch 上界 → 口径必须标注

## 证据与复现

- `.scratch/19-asym-100m/evidence @8401b5e`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
