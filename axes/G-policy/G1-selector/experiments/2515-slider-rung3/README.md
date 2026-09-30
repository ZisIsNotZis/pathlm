---
name: 2515-slider-rung3
ticket: 15-slider-rung3
date: "2026-09-16"
grid: G1-selector
engine: "m1（13M，INT2_r3 底座）"
budget: "纯推理"
---

# 2515-slider-rung3（ticket 15-slider-rung3）

> 主格：`G1-selector`｜关联格：`G2-currency`, `F3-stopping`, `G4-calibration`｜引擎：m1（13M，INT2_r3 底座）｜预算：纯推理

## 命题

分配器端到端：校准 → 双货币成本模型 → 求解器 → 验证解码 → 在线代理

## 干预

Slider 全链

## 结果

- fire 曲线 split gap ≤0.004；a2 成本预测误差 ≤±13%
- 严格占优点复现：k=1 τ=0.98 = 0.635 fwd/tok @2.0502 vs plain 1.0 @2.0545
- TF 窗口在线代理 ECE 0.0011–0.0079；单轮 mixture == overwrite

## 失败路径与原因

1. 自由生成段 prob0 劣化检测反向（当时判「吸引子混淆」，后被 ticket 22 证伪为 n=1 假象）
2. a2 部署口径与 TF 口径差 −13~−20%（必须双报）

## 证据与复现

- `.scratch/15-slider-rung3/evidence @257b6ec`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
