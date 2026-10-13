---
name: 2523c-three-ablations
ticket: 23c
date: "2026-09-28"
grid: C5-refinement-operator
engine: "depth-ar（13M 级）"
budget: "15.2 min"
---

# 2523c-three-ablations（ticket 23c）

> 主格：`C5-refinement-operator`｜关联格：`C6-depth-supervision`, `H2-training-fill`｜引擎：depth-ar（13M 级）｜预算：15.2 min

## 命题

税源的因果分解（欠拟合 / aux / 门×退出×填充）

## 干预

H1 dense+aux-only / H2 3.33× 训练 / H3 浅头 stop-grad

## 结果

- 欠拟合是主因：3.33× 训练把税 +0.713 → +0.121
- aux 在门控臂穿 trunk 的梯度是净收益；stop-grad 恶化至 2.7129
- 门×退出×填充堆栈 +0.429（门本身 +0.228）

## 失败路径与原因

1. 浅头 stop-grad 化被排除（浅层门不再打开、tf@4 0.973→0.791）

## 证据与复现

- `.scratch/23-depth-ar/evidence @88aa341`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
