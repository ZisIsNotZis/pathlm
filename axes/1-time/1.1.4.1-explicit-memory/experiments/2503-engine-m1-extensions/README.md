---
name: 2503-engine-m1-extensions
ticket: 03-engine-m1
date: "2026-09-14"
grid: M1-carrier-type
engine: "m1（d=256, 12L, 13.08M）"
budget: "—"
---

# 2503-engine-m1-extensions（ticket 03-engine-m1）

> 主格：`M1-carrier-type`｜关联格：`P4-normalization`, `A8-positional-semantics`, `F2-return-channel`, `H5-verification`｜引擎：m1（d=256, 12L, 13.08M）｜预算：—

## 命题

引擎扩展：transports / dense-exit / 驱逐 / token retry / KV 增量解码 / 评测电池

## 干预

实现并单测各机制

## 结果

- 27 测试绿；smoke B0 60 步 bpc 4.15（随机 7.68），端到端 OK
- per-layer KV + 位置列表支持变深度增量解码与驱逐

## 失败路径与原因

1. 冒烟抓出缓存 bug：逐位置早退使 per-layer cache 长度失步 → 修为 per-layer 位置列表（P0 类，持续单测）
2. needle v1 构造信号饥饿（每 batch 仅 1 个监督 token）→ 全桶 0.0 假失败

## 证据与复现

- `.scratch/03-engine-m1/issues/01-engine-m1-extensions.md @e38349f`

> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。
