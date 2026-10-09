---
name: 2524-depth-ar-100m
ticket: 24/24b
date: "2026-09-28"
grid: C6-depth-supervision
engine: "depth-ar @100M（98.3M tokens = 1.09 ep）"
budget: "24: 32 min；24b: 51 min 两臂"
---

# 2524-depth-ar-100m（ticket 24/24b）

> 主格：`C6-depth-supervision`｜关联格：`C1-layers-per-token`, `W4-readout-geometry`｜引擎：depth-ar @100M（98.3M tokens = 1.09 ep）｜预算：24: 32 min；24b: 51 min 两臂

## 命题

形态税随规模如何变化（快照口径）+ 干净归属

## 干预

24：主 run；24b：等权臂 + 同引擎 dense 两臂拆混杂

## 结果

- 税 +0.4563（clean 1.9175 − B0_100M 1.4612）→ 24b 拆解：形态 +0.44（96%）+ 引擎 +0.02（4%）+ 配方 ≈0
- 同引擎形态税 +0.4362；13M matched +0.31 → 100M +0.44（斜率为正）
- 出口资产变厚：tf@4 0.9726 / tf@6 0.9946；exit-4 3.1× 墙钟；门全层收小
- 副发现：出口资产是全深度监督产物（纯 dense trunk tf@4 0.12 vs 形态 0.97）

## 失败路径与原因

1. 票 24 判读混杂（跨引擎对照 + final 2× 配方）→ 必须 24b 拆混杂才可归因
2. soft OOD 污染跨规模复现（d12 soft 8.42 vs hard 0.79）

## 证据与复现

- `.scratch/24-depth-ar-100m/evidence @7cdcdaa`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
