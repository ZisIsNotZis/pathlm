---
name: 2520-spec-at-scale
ticket: 20-spec-scale
date: "2026-09-18"
grid: H1-outputs-per-pass
engine: "100M"
budget: "纯推理"
---

# 2520-spec-at-scale（ticket 20-spec-scale）

> 主格：`H1-outputs-per-pass`｜关联格：`H5-verification`, `G2-currency`｜引擎：100M｜预算：纯推理

## 命题

spec 的收益在 100M 是否保持

## 干预

接受级联测量 + 墙钟

## 结果

- 接受级联稳定：a₂ 0.80、a₃|a₂ 0.767↑、2.4 tok/fwd 不变
- 前向 compute-bound（宽度-3 成本 3.4–5.3× vs 13M 1.28×）

## 失败路径与原因

1. 墙钟收益消失（k=2 仅 1.15–1.29×、k=1 转负）→ 只剩 FLOPs 口径
2. fp 平局翻转首现（top-2 gap 0.008，1/400 token）→「逐位相等」是 13M 规模性质

## 证据与复现

- `.scratch/20-spec-scale/evidence @a864bcf`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
