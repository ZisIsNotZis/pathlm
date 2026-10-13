---
name: 2507-x1-x3-distance-penalty
ticket: 07-x1-x3
date: "2026-09-14"
grid: A4-soft-distance-penalty
engine: "m1（13M）"
budget: "~25 min/run"
---

# 2507-x1-x3-distance-penalty（ticket 07-x1-x3）

> 主格：`A4-soft-distance-penalty`｜关联格：`A3-window-anchor`｜引擎：m1（13M）｜预算：~25 min/run

## 命题

距离惩罚是否与驱逐冗余

## 干预

X1 = dist_pen 0.1；X3 = X1 + eviction

## 结果

- X1 1.5006（Δ−0.0068，唯一负税元素）
- X3 1.5117（+0.0043）；needle 窗口内 90.7 / 锚 93.3 / beyond 90.3
- per-head 距离 26–43（无 head 纯 local，但全部被推向 local）

## 失败路径与原因

1. beyond 90.3% 是 local-LM 强度，不是抗驱逐召回（假保留陷阱）

## 证据与复现

- `.scratch/07-x1-x3/evidence @7b47555`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
