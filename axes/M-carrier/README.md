<!-- AUTO:begin -->

# M · 状态载体

- **原语**：II（沿序列推进） × 记忆
- **命题**：跨步信息存在哪里、怎么压、跨步假设是什么。

## 定义

载体决定「能否丢、能否重算、能否检索」。KV cache 只是众多载体之一；
载体选择直接决定部署内存与保留能力。

## 子维度

M1 载体类型 · M2 载体压缩 · M3 跨步假设

## 本地总体结论

本地只有 KV 载体（+ eviction 压缩）。递归状态 / 显式记忆槽 / 外部存储 /
权重载体全零经验——这是「右移压缩」主线的落点。

## 格子索引

| 格子 | 状态 | 判读 | 税 |
|---|---|---|---|
| [`M1-carrier-type`](M1-carrier-type/README.md) | ◐ partial | 只有 KV 载体；其余零经验，但载体是「能否丢/重算」的决定因素。 | —（KV 为基底） |
| [`M2-carrier-compression`](M2-carrier-compression/README.md) | ◐ partial | 驱逐已测（价格见 A3）；量化/摘要/低秩零经验。 | 驱逐 +0.058~0.121（13M） |
| [`M3-cross-step-assumption`](M3-cross-step-assumption/README.md) | ◐ partial | 每 token 独立 + 「不重 prefill」已实现；段级递归零经验。 | — |

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
