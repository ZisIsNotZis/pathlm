---
name: 2523b-shallow-weight-sweep
ticket: 23b
date: "2026-09-28"
grid: P5-loss-weighting
engine: "depth-ar（13M 级）"
budget: "10.8 min"
---

# 2523b-shallow-weight-sweep（ticket 23b）

> 主格：`P5-loss-weighting`｜关联格：`C6-depth-supervision`｜引擎：depth-ar（13M 级）｜预算：10.8 min

## 命题

浅层 aux 降权能否把税归零

## 干预

w ∈ {1, 0.1, 0.02, 1→0.1 退火}

## 结果

- 全败且单调反向：depth-L 2.45→2.57→2.73（w 越低越差）
- 退火臂 2.4444 保底座但税仍 +0.71

## 失败路径与原因

1. 「浅层监督稀释主因」改判：浅层全深度监督是表示正则，不是稀释

## 证据与复现

- `.scratch/23-depth-ar/evidence @b8c9f86`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
