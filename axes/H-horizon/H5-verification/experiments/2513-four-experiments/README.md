---
name: 2513-four-experiments
ticket: 13-four-experiments
date: "2026-09-15"
grid: H5-verification
engine: "m1（13M）"
budget: "~25 min/run"
---

# 2513-four-experiments（ticket 13-four-experiments）

> 主格：`H5-verification`｜关联格：`F2-return-channel`, `A3-window-anchor`, `H1-outputs-per-pass`, `F4-training-signal`｜引擎：m1（13M）｜预算：~25 min/run

## 命题

四实验：逐位置门控 / anchor 豁免 / n_mtp=3 + Medusa / findings 重构

## 干预

①逐位置 latent 门控 ②三路 anchor 豁免 ③n_mtp=3 + 批量验证

## 结果

- ③ POSITIVE：1.68–1.90× 实测 tok/s，税仅 +0.0166（全项目唯一税小收益大）
- ② 三路豁免全败：anchor 召回 0.764→0.016/0.033/0.035（任务级豁免还把 bpc 1.680→1.997）
- ① FALSIFIED：direct 重入是固定点；状态掩码不省算力

## 失败路径与原因

1. 容错训练与精确复制目标冲突 → 用户裁决「无需模式信号」（目标恒为正确 token）
2. 两次 review 抓出：flaky gate test（未 seed）、needle_acc 评测泄漏（corrupt_mask 未清零）

## 证据与复现

- `.scratch/13-four-experiments/evidence @9abc181`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
