---
name: 2502-engine-m0-probe
ticket: 02-engine-m0
date: "2026-09-14"
grid: A2-full-attention
engine: "m0（d=64, 2L, ~1M 参数）"
budget: "~2.5 min/run"
---

# 2502-engine-m0-probe（ticket 02-engine-m0）

> 主格：`A2-full-attention`｜关联格：`F2-return-channel`, `G4-calibration`, `G5-parallel-lines`｜引擎：m0（d=64, 2L, ~1M 参数）｜预算：~2.5 min/run

## 命题

微型探针四判据：prob0 校准 / 集成增益 / consistency loss / 直接传输 retry

## 干预

微型模型 + 四组判据

## 结果

- prob0 校准 PASS：ECE 0.004–0.011
- 集成 ≥ 最佳单路径 PASS：+0.0008…+0.0015（3 runs/2 seeds，边际）
- consistency loss 收紧估计 PASS：cosine 0.9525→0.9882
- 直接传输 retry FAIL：结构性 no-op

## 失败路径与原因

1. retry 在无状态变换的通道上是固定点——全项目第一条铁律的出处
2. 集成增益 +0.0008~+0.0015 已预示「共享主干 voter 太相关」（定律 2）

## 证据与复现

- `.scratch/02-engine-m0/evidence/REPORT.md @27418ae`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
