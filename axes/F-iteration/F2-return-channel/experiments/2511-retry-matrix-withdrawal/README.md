---
name: 2511-retry-matrix-withdrawal
ticket: 11-retry-matrix-validity
date: "2026-09-15"
grid: F2-return-channel
engine: "m1（13M）"
budget: "~25 min/run"
---

# 2511-retry-matrix-withdrawal（ticket 11-retry-matrix-validity）

> 主格：`F2-return-channel`｜关联格：`F1-iteration-object`｜引擎：m1（13M）｜预算：~25 min/run

## 命题

retry 矩阵的排序是否成立（四缺陷审计）

## 干预

重跑 6 个 latent cell + 缺陷审计 D1–D6

## 结果

- 6 个 cell 跨度 0.011 bpc < 单格种子散布 0.0138 → 全部排序撤回
- 税本身可靠：C 族 1.6535–1.6856 vs B0 1.5074 = +0.146…+0.178
- 所有已发布税是 1.1-epoch 上界（+0.364@1k → +0.122@24k）

## 失败路径与原因

1. 撤回全部排序结论（含「soft×mixture 最好」）
2. D1 重入 gauge：direct/linear mixture 测到的是 ~30× rescale 而非 latent mean

## 证据与复现

- `.scratch/11-retry-matrix-validity/VERDICTS.md @d6a9898`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
