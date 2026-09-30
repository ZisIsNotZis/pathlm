---
name: 2525-tax-attribution
ticket: 25
date: "2026-09-29"
grid: W1-active-width
engine: "depth-ar @100M（三臂）"
budget: "GPU ~185 min（三臂）"
---

# 2525-tax-attribution（ticket 25）

> 主格：`W1-active-width`｜关联格：`C1-layers-per-token`, `W4-readout-geometry`, `P5-loss-weighting`, `G2-currency`｜引擎：depth-ar @100M（三臂）｜预算：GPU ~185 min（三臂）

## 命题

税能否被两个干预收回（结构切分 / 时长）

## 干预

B 臂结构切分 readout 360 / C 臂 48k + 冷却退火 / dense-48k 对照

## 结果

- B 臂否证且反向：clean 2.1475、恢复 −50.4%（能量迁出读出空间：切片能量 0.85→0.026→≈0）
- C 臂名义 +16.4%，同引擎 48k 锚拆解后形态税零收缩（+0.4372 ≈ +0.4362）
- C 对 A 改善 0.0746 与 dense 侧训练收益 0.0756 逐位镜像
- C 等效验证吞吐 122.8 vs A 112.9 tok/s（+9%）= 当前最优配方

## 失败路径与原因

1. 「等智商免费加速」正式证伪：税不可被收，只能被交易
2. 同引擎 dense 对照健康（dense-48k 1.4057 < B0 1.4612），排除「对照本身不健康」

## 证据与复现

- `.scratch/25-tax-attribution/evidence @2e86924`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
