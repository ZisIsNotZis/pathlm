---
name: 2523-depth-ar-first
ticket: 23-depth-ar
date: "2026-09-28"
grid: C1-layers-per-token
engine: "depth-ar（d=128/256, L=4/8）"
budget: "GPU 11.4 min"
---

# 2523-depth-ar-first（ticket 23-depth-ar）

> 主格：`C1-layers-per-token`｜关联格：`C5-refinement-operator`, `C6-depth-supervision`, `W4-readout-geometry`, `H2-training-fill`｜引擎：depth-ar（d=128/256, L=4/8）｜预算：GPU 11.4 min

## 命题

深度自适应 AR 四性质是否成立（精化 / workspace / 出口 / 门）

## 干预

新形态首验

## 结果

- 精化单调成立（B: 6.250→2.4161）；出口半栈 TF 接受率 0.9667；门浅开深微开
- workspace 成立：d=256 null 62 维 / 26 条功能性读取；d=128 结构性 0
- 形态税 +0.6786（判据 |税|<0.05 证伪）

## 失败路径与原因

1. 首验归因（监督稀释 +0.512 / 填充 +0.167）后被 23b–23e 四轮修正（w 降权全败、填充反转）
2. soft 提交 OOD 污染 context（rescore 6.9 vs hard 1.2）

## 证据与复现

- `.scratch/23-depth-ar/evidence @1586745`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
