---
name: 2505-m1-followups
ticket: 05-m1-followup
date: "2026-09-14"
grid: A3-window-anchor
engine: "m1（13M）"
budget: "30–60 min/run"
---

# 2505-m1-followups（ticket 05-m1-followup）

> 主格：`A3-window-anchor`｜关联格：`F2-return-channel`, `F1-iteration-object`, `P6-data-construction`｜引擎：m1（13M）｜预算：30–60 min/run

## 命题

mixture 重入 + 曲线饱和 + needle v2 重设计

## 干预

C4（mixture）/ C5（token retry + mixture）/ X2v2（needle v2）

## 结果

- C4 PASS-equal：mixture 提高深度鲁棒性，r1→r4 单调
- C5 PASS-with-caveat：token 灾难消除（−1.3pp vs R1 的 −4.6pp）
- X2v2 PASS：窗口内 84.9% / 锚 76.8% / beyond 0.4%
- retry 曲线 ~r3 饱和

## 失败路径与原因

1. needle 修好测量后才可判机制（先修测量再判机制）
2. token retry 残余 −1.3pp 需 prob0 门控（未做）

## 证据与复现

- `.scratch/05-m1-followup/evidence @480d50b`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
