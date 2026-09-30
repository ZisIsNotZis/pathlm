<!-- AUTO:begin -->

# W · 宽度

- **原语**：I（在同一状态上多花算力） × 参数
- **命题**：每 token 激活多少参数/通道、参数如何组织成专家、读出几何如何。

## 定义

宽度轴的两个方向截然不同：减宽 = 减容量（付质量），
稀疏宽 = 保容量减计算（MoE 族）。读出几何决定「计算空间」与
「读出可见空间」是否耦合。

## 子维度

W1 激活宽度 · W2 专家化 · W3 KV 头共享 · W4 读出几何 · W5 容量-计算

## 本地总体结论

本地只做了「减通道」这一个方向，且结论是负的：结构切分读出（只读前 360 维）
使模型把状态能量整体迁出读出子空间（切片能量 0.85→0.026→≈0），
clean 2.1475、恢复 −50.4%。用户本地独立实验（减宽）同样「效果微妙」。
专家化、KV 头共享、非线性读出全是零经验。

## 格子索引

| 格子 | 状态 | 判读 | 税 |
|---|---|---|---|
| [`W1-active-width`](W1-active-width/README.md) | ✅ judged | 减宽 = 减容量，不买灵活性；结构化约束会把状态能量赶出被约束子空间。 | 结构切分读出（只读前 360 维，@100M）clean 2.1475、同引… |
| [`W2-expertization`](W2-expertization/README.md) | ⬜ stub | — | — |
| [`W3-kv-head-sharing`](W3-kv-head-sharing/README.md) | ◐ partial | 本地只有 MHA；其余为零经验，但这是长上下文线的免费基线（必须先打败它）。 | —（基底为 MHA） |
| [`W4-readout-geometry`](W4-readout-geometry/README.md) | ✅ judged | tied 线性是「同空间」前提的地基，也是宽度约束的来源（d − 有效读出秩）；**gated 残差读出（单边零初始化）在 13M 收回 24.6% 形态税**（Δ−0.081… | **gated 残差读出 Δ clean bpc −0.0812**（1.8… |
| [`W5-capacity-vs-compute`](W5-capacity-vs-compute/README.md) | ✅ judged | 减容量不买灵活性；买灵活性必须走保容量减计算。 | 容量削减：负结果（W1 −50.4%） |

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
