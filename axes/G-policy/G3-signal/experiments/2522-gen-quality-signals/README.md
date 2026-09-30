---
name: 2522-gen-quality-signals
ticket: 22-gen-quality
date: "2026-09-18"
grid: G3-signal
engine: "INT2_r3（纯推理）"
budget: "纯推理"
---

# 2522-gen-quality-signals（ticket 22-gen-quality）

> 主格：`G3-signal`｜关联格：`G4-calibration`｜引擎：INT2_r3（纯推理）｜预算：纯推理

## 命题

自由生成段的质量信号（prob0 失效后的替代）

## 干预

路径分歧度 js + distinct-2 + 人工标注

## 结果

- js 真实但不完备（AUC 0.66–0.69 > prob0 0.58–0.62；生成段 real 0.0114 < corrupt 0.0250 < random 0.0390）
- distinct-2 区分塌缩窗 AUC 0.9815，与 js 正交（rank_corr 0.14）
- 「13M prob0 生成段反向」n=6 未复现（n=1 假象）

## 失败路径与原因

1. prob0 在生成段压天花板（分不开 real/random）
2. js 在吸引子塌缩窗双向失灵（假阴 0.000 与假阳 0.373 并存）
3. greedy 全局重复偏置 → 参考水平必须相对化

## 证据与复现

- `.scratch/22-gen-quality/evidence @c6c356f`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
