---
name: 2523e-fill-variants
ticket: 23e
date: "2026-09-28"
grid: H2-training-fill
engine: "depth-ar（12k matched）"
budget: "~29 min 三臂"
---

# 2523e-fill-variants（ticket 23e）

> 主格：`H2-training-fill`｜关联格：`P3-param-insertion-init`｜引擎：depth-ar（12k matched）｜预算：~29 min 三臂

## 命题

KV 填充策略对决：fill vs ragged nofill vs proj-fill

## 干预

三分支同配置唯一变量 = 填充策略

## 结果

- fill 1.8582 胜 ragged nofill 1.9438（+0.0856 ≫ ±0.03 平局带）
- proj-fill（逐层恒等初始化 adapter）1.8356 = 族内最优，全深度逐深度支配 fill

## 失败路径与原因

1. 首验「填充是负分量」反转：欠拟合伪影 + 对照混淆（nofill 臂同时无随机退出）
2. GPU 冒烟抓出 CUDA SDPA head 维不广播

## 证据与复现

- `.scratch/23-depth-ar/evidence @6b27b5e`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
