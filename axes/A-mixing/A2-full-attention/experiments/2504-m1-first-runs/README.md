---
name: 2504-m1-first-runs
ticket: 04-m1-runs
date: "2026-09-14"
grid: A2-full-attention
engine: "m1（13M）"
budget: "30–60 min/run"
---

# 2504-m1-first-runs（ticket 04-m1-runs）

> 主格：`A2-full-attention`｜关联格：`P6-data-construction`, `F1-iteration-object`, `C6-depth-supervision`, `A3-window-anchor`, `H1-outputs-per-pass`｜引擎：m1（13M）｜预算：30–60 min/run

## 命题

首个 run 矩阵：B0 锚点 + 损坏修复 + retry + 早退 + token retry + needle

## 干预

7 个 run（B0/I1/C1/C3/R1/L2/X2）

## 结果

- B0 = 1.5074 bpc（外部锚成立）
- I1 损坏修复 52.8%；C1 retry +2.1pp，r1→r3 单调；早退 L2 深度 6 平台 + 1.4× 解码
- R1 token retry −4.6pp，ECE 0.001→0.015
- X2 needle 全桶 0.0（v1 构造问题）

## 失败路径与原因

1. token retry 未门控：argmax re-embed 量化掉不确定性，污染下游注意力
2. needle v1 信号饥饿（假失败，见 P6）

## 证据与复现

- `.scratch/04-m1-runs/evidence @e38349f`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
