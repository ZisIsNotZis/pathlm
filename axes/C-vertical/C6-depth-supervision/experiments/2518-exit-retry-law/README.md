---
name: 2518-exit-retry-law
ticket: 18-exit-probe
date: "2026-09-18"
grid: C6-depth-supervision
engine: "m1（13M）"
budget: "~25 min/run"
---

# 2518-exit-retry-law（ticket 18-exit-probe）

> 主格：`C6-depth-supervision`｜关联格：`H4-head-trunk-relation`, `F1-iteration-object`｜引擎：m1（13M）｜预算：~25 min/run

## 命题

exit 与 retry 能否两全

## 干预

probe 式头（trunk stop-grad）vs dense 0.2 整形

## 结果

- probe 式头保 retry（−0.0098）但浅层读出贵（depth6 +0.93）
- dense 0.2 整形 exit 轴好（depth6 +0.199）但杀 retry（+0.0146）且 clean/corrupt 双升（1.6097/1.9740）

## 失败路径与原因

1. 两全证伪：整形与自适应是同一梯度的两面

## 证据与复现

- `.scratch/18-exit-probe/evidence @83bf1ac`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
