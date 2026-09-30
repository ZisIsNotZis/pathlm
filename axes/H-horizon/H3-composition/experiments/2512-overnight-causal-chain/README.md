---
name: 2512-overnight-causal-chain
ticket: 12-overnight
date: "2026-09-15"
grid: H3-composition
engine: "m1（13M，2 seeds）"
budget: "夜间批"
---

# 2512-overnight-causal-chain（ticket 12-overnight）

> 主格：`H3-composition`｜关联格：`F2-return-channel`, `A3-window-anchor`, `G2-currency`｜引擎：m1（13M，2 seeds）｜预算：夜间批

## 命题

条件链是否可部署；D1 gauge 因果确认；needle×corruption 拮抗

## 干预

DeepSeek 式 concat+proj 条件链 / gauge 修复 / anchor 豁免

## 结果

- oracle 链 0.682 ≈ node-1 0.693；deploy 0.547 < direct 0.560（草稿质量瓶颈，~31% 草稿错误）
- gauge 修复后 mixture r2 单调 2.71→1.936（pre-fix 发散至 95.2）
- needle×损坏拮抗：85%→0.5%

## 失败路径与原因

1. 条件链机制成立但无部署收益（草稿质量瓶颈）→ 不采纳
2. naive verify-next-round k=1 零吞吐收益

## 证据与复现

- `.scratch/12-overnight/evidence @216a00a`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
