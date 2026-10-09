---
name: 2510-gap-sweep-b2-c1m
ticket: 10-gap-sweep
date: "2026-09-15"
grid: H1-outputs-per-pass
engine: "m1（13M）"
budget: "~25 min/run"
---

# 2510-gap-sweep-b2-c1m（ticket 10-gap-sweep）

> 主格：`H1-outputs-per-pass`｜关联格：`F2-return-channel`｜引擎：m1（13M）｜预算：~25 min/run

## 命题

补 B2（n_mtp=2）与 C1M/C2M 的缺口

## 干预

B2 / C1M / C2M + eval-only 消融

## 结果

- B2 1.5481（Δ+0.041）
- C1M 1.7010 / C2M 1.7089（含 D2 混杂：漏 n_mtp，税虚高，ticket 11 修正）
- eval-only：redo eval +0.009、skip eval 回收 +0.058

## 失败路径与原因

1. config 混杂（D2）：对照与实验臂的 n_mtp 不一致 → 税虚高

## 证据与复现

- `.scratch/10-gap-sweep/evidence @57d64a7`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
