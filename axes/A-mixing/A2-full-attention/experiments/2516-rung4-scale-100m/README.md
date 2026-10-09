---
name: 2516-rung4-scale-100m
ticket: 16-rung4-scale
date: "2026-09-17"
grid: A2-full-attention
engine: "100M × 2 seeds"
budget: "~1–2 h/run"
---

# 2516-rung4-scale-100m（ticket 16-rung4-scale）

> 主格：`A2-full-attention`｜关联格：`G1-selector`, `G4-calibration`, `F2-return-channel`｜引擎：100M × 2 seeds｜预算：~1–2 h/run

## 命题

回环族的税与能力溢价在 100M 是否成立

## 干预

100M × B0/C1 × 2 seeds

## 结果

- 税 @100M = +0.126（13M +0.147 → 29M +0.174 → 100M +0.126，首次不随规模增长）
- 能力溢价：B0 损坏崩至 3.551、C1 修复 2.015→1.949、E2E 差距 ≥1.54 bpc
- prob0 校准存活（ECE ≤0.0091）

## 失败路径与原因

1. batch 32→8 混杂（显存限制）
2. 门控重试质量杠杆在此规模开始衰减

## 证据与复现

- `.scratch/16-rung4-scale/evidence @c520d65`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
