---
name: 2517-slider-at-scale
ticket: 17-slider-at-scale
date: "2026-09-17"
grid: G1-selector
engine: "INT_100M_s0（104.3M）"
budget: "纯推理 + 训练"
---

# 2517-slider-at-scale（ticket 17-slider-at-scale）

> 主格：`G1-selector`｜关联格：`H1-outputs-per-pass`｜引擎：INT_100M_s0（104.3M）｜预算：纯推理 + 训练

## 命题

分配器在 100M 是否自洽；门控杠杆是否存活

## 干预

Slider 全链 @100M

## 结果

- 全链自洽：成本预测 clean +0.7% / corrupt ≤11.1%；spec 严格占优复现 0.566 fwd/tok
- 堆叠溢价 +0.256（13M +0.309 收窄）

## 失败路径与原因

1. 门控重试质量杠杆在 100M 消失（τ 轴平坦 ±0.002）→ 第一个「机制价值随规模自衰减」信号
2. 自由生成段劣化检测恢复正向（13M 反向系 n=1 假象）

## 证据与复现

- `.scratch/17-slider-at-scale/evidence @25704b8`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
