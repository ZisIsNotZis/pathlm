---
name: 2521-exit-scale-confirm
ticket: 21-exit-at-scale
date: "2026-09-18"
grid: H4-head-trunk-relation
engine: "100M"
budget: "~35 min/run"
---

# 2521-exit-scale-confirm（ticket 21-exit-at-scale）

> 主格：`H4-head-trunk-relation`｜关联格：`C6-depth-supervision`｜引擎：100M｜预算：~35 min/run

## 命题

probe vs dense 双 profile 在 100M 是否更锐利

## 干预

两 profile @100M

## 结果

- probe profile 零税（1.5928 vs 1.5871）且门控更强（−0.029）
- dense 0.2 只剩 clean 优势与 exit 轴，corrupt 崩溃（+1.73）

## 失败路径与原因

1. 剂量响应随规模恶化 → 13M 的温和剂量在 100M 变灾难

## 证据与复现

- `.scratch/21-exit-at-scale/evidence @9fe4cf1`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
