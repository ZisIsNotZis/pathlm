<!-- AUTO:begin -->

# F · 迭代

- **原语**：I（在同一状态上多花算力）
- **命题**：对什么重复、用什么通道回传、何时停、靠什么训练信号让重复有效。

## 定义

与深度同属原语 I；区别在于重复的对象（同 token 状态 / 同段 / 整段 /
候选集 / 隐式推理步 / 权重）。回传通道决定「重复能否带来新信息」。

## 子维度

F1 迭代对象 · F2 回传通道 · F3 停止条件 · F4 训练信号

## 本地总体结论

回环时代的主战场，已收口：直接重入是固定点（结构性 no-op）；
6 个 latent 回传 cell 跨度仅 0.011 bpc < 种子散布 0.0138 → 全部排序撤回；
可靠的是「多一遍有价值，回传几何无价值」；soft 回传三度复现 OOD 污染。
保留的有效资产：选择性门控成立（低置信 15% 开火 +0.128 vs 均匀 +0.057），
预算求解器与双货币记账建成。去噪整段 / 隐式推理步 / 权重迭代零经验。

## 格子索引

| 格子 | 状态 | 判读 | 税 |
|---|---|---|---|
| [`F1-iteration-object`](F1-iteration-object/README.md) | ◐ partial | 前两项已收口；后四项零经验。 | latent retry C 族 1.6535–1.6856 vs B0 1… |
| [`F2-return-channel`](F2-return-channel/README.md) | ✅ judged | 直接回传是固定点（结构性 no-op）；软回传有 OOD 污染；离散回传必须门控。 | 通道税在 F1 的 C 族内（+0.146…+0.178） |
| [`F3-stopping`](F3-stopping/README.md) | ✅ judged | 选择性问题（愿景核心命题）获证；预算求解器端到端建成。 | 分配器本身不额外加税（单轮 mixture == overwrite，设计保证） |
| [`F4-training-signal`](F4-training-signal/README.md) | ✅ judged | consistency loss 中性；重建/sufficiency 是唯一无人占位、且是压缩可恢复性的必要信号。 | consistency loss 中性（IX4 1.6589 ≈ C1 1.… |

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
