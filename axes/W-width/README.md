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
专家化（W2）与非线性读出仍是零经验；KV 头共享（W3）已有本地实测
（GQA/MQA × int8/int4 的组合价签）。

> **既有做法情报**：[PRIOR_ART.md](PRIOR_ART.md)（含机器可读分诊表；来源标 `记忆·未核`，联网不可用）

## 格子索引

| 格子 | 状态 | 判读 | 税 |
|---|---|---|---|
| [`W1-active-width`](W1-active-width/README.md) | ✅ judged | 减宽 = 减容量，不买灵活性；结构化约束会把状态能量赶出被约束子空间。 | 结构切分读出（只读前 360 维，@100M）clean 2.1475、同引… |
| [`W2-expertization`](W2-expertization/README.md) | 🏭 reference（含业界结论） | — | —（未做本地实验） |
| [`W3-kv-head-sharing`](W3-kv-head-sharing/README.md) | ✅ judged | 参考线已补齐**：GQA（KV −75%）实测 damage **+0.0143**、同 seed 重跑 +0.0139（**跨在门槛两侧**，散度仅 0.0004 ⇒ 依预登… | clean bpc：mha **1.4820** / gqa **1.496… |
| [`W4-readout-geometry`](W4-readout-geometry/README.md) | ✅ judged | 读出几何**可以**解耦（prob0/提交/传输不受损，同空间前提不冲突），但**增量归因于「读出重参数化 + 额外 2d² 参数」，不是非线性本身**：等参数纯线性对照 Δ=… | **读出重参数化 Δ clean bpc −0.0812**（gated，1… |
| [`W5-capacity-vs-compute`](W5-capacity-vs-compute/README.md) | ✅ judged | 减容量不买灵活性；买灵活性必须走保容量减计算。 | 容量削减：负结果（W1 −50.4%） |

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
