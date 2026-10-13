---
name: 2508-tts-probes
ticket: 08-tts-probe
date: "2026-09-15"
grid: G1-selector
engine: "m1（13M）"
budget: "纯推理"
---

# 2508-tts-probes（ticket 08-tts-probe）

> 主格：`G1-selector`｜关联格：`C2-layer-selection-order`, `C6-depth-supervision`｜引擎：m1（13M）｜预算：纯推理

## 命题

组合增益是否 ∝ 路径多样性；有什么能打败 B0

## 干预

K 路径集成（不同 skip/depth/round 组合）

## 结果

- 组合增益 ∝ 路径多样性（L1 −0.047）
- 确定性路径 checkpoint 集成恰得 0
- 深度/轮次集成退化（辅助输出是 fallback 不是 voter）

## 失败路径与原因

1. 没有任何组合打赢 B0 → 项目卖点从「质量」改为「可控性」

## 证据与复现

- `.scratch/08-tts-probe/evidence @aa97bcf`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
