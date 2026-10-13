---
name: 2506-m2-element-tax-law
ticket: 06-m2-sweep
date: "2026-09-14"
grid: C2-layer-selection-order
engine: "m1（13M）"
budget: "~25 min/run"
---

# 2506-m2-element-tax-law（ticket 06-m2-sweep）

> 主格：`C2-layer-selection-order`｜关联格：`P6-data-construction`｜引擎：m1（13M）｜预算：~25 min/run

## 命题

元素税定律：破坏 identity/order 的贵，保留/标记 identity 的近免费

## 干预

shuffle / skip / redo / wrong-token / mask / noise 各自的税

## 结果

- shuffle +0.545；skip +0.175；wrong +0.127；mask +0.059；redo +0.044；noise ±0.000
- shuffle 硬边界：locality 0.5 训练、eval ≤0.5 可容忍、全随机 3.84

## 失败路径与原因

1. p_skip 训练/eval 不一致（locality 边界）→ 训练分布之外的评估不可信

## 证据与复现

- `.scratch/06-m2-sweep/evidence @a03a579`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
