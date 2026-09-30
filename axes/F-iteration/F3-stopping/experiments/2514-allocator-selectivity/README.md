---
name: 2514-allocator-selectivity
ticket: 14-allocator
date: "2026-09-16"
grid: F3-stopping
engine: "3 checkpoint 交叉验证"
budget: "纯推理"
---

# 2514-allocator-selectivity（ticket 14-allocator）

> 主格：`F3-stopping`｜关联格：`G1-selector`, `C6-depth-supervision`｜引擎：3 checkpoint 交叉验证｜预算：纯推理

## 命题

愿景核心命题：低置信位置的每 FLOP 收益是否更高

## 干预

按 prob0 只对最低 15% 位置开第二轮 vs 均匀

## 结果

- INT2：门控 +0.0211 vs 均匀 −0.0193
- C1M：门控 +0.1284 vs 均匀 +0.0570（2.2 倍）
- 收益集中在中低置信带（+0.04~+0.06），高置信带 ≈0/负

## 失败路径与原因

1. dense-exit 监督摧毁重试精炼（ALLOC 各桶全负）→ 「整形 vs 自适应」定律的第一次露头

## 证据与复现

- `.scratch/14-allocator/evidence @9954425`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
