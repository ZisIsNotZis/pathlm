---
name: 2509-m3-composition-law
ticket: 09-m3-interactions
date: "2026-09-15"
grid: C2-layer-selection-order
engine: "m1（13M）"
budget: "~25 min/run"
---

# 2509-m3-composition-law（ticket 09-m3-interactions）

> 主格：`C2-layer-selection-order`｜关联格：`P1-precision`, `F4-training-signal`｜引擎：m1（13M）｜预算：~25 min/run

## 命题

组合律 + 多样性压力

## 干预

IX1（retry×corruption）/ 含 shuffle 组合 / DIVL1（capped −JS）

## 结果

- IX1 次可加 +0.087 < 0.277 之和（唯一已证组合增益）
- 含 shuffle 组合全部 2.2+ bpc，与修复拮抗（repair 51%→38%）
- DIVL1 同时改善 baseline 与集成斜率（−0.062 > −0.047，上限 +0.023），税 +0.023

## 失败路径与原因

1. bf16 autocast 下 negative/reward 项两次发散 → 必须 float32 + pre-step grad-norm gate

## 证据与复现

- `.scratch/09-m3-interactions/evidence @39b1838`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
