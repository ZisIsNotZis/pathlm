<!-- AUTO:begin -->

# C · 垂直机制（深度）

- **原语**：I（在同一状态上多花算力）
- **命题**：同一位置的算力如何沿「层」组织——跑几层、怎么连、怎么共享、怎么精化。

## 定义

这条轴是「Pro 当 Flash 用」的实现地：逐 token 变深度、层间拓扑、权重共享、
精化算子、以及训练期深度分布（哪条路径被监督）。

## 子维度

C1 跑几层 · C2 选哪些层与顺序 · C3 层间拓扑 · C4 权重共享 ·
C5 精化算子 · C6 训练深度分布

## 本地总体结论

深度自适应形态（ticket 23–25）在本轴上是本项目最厚的资产：
逐 token 变深度 + 全深度并行监督 + carry 门 + 随机退出深度；
出口资产 tf@4 0.9726、d6−d12 0.0003、exit-4 3.1× 墙钟，
代价是同引擎形态税 +0.4362 @100M（96% 干净归属）。
元素税定律在本轴同样成立：skip +0.175、层序重排 +0.545（最贵）。
层间拓扑（C3）与权重共享（C4）本地零经验——这是最便宜的真空白。

## 格子索引

| 格子 | 状态 | 判读 | 税 |
|---|---|---|---|
| [`C1-layers-per-token`](C1-layers-per-token/README.md) | ✅ judged | 逐 token 变深度成立且是本项目最厚的资产；代价是底座质量税。 | 形态税 +0.6786（13M 首验）→ matched-step 真税 +… |
| [`C2-layer-selection-order`](C2-layer-selection-order/README.md) | ✅ judged | 元素税定律的原始出处：破坏型元素贵，保留型近免费。 | skip +0.175（最贵深度元素）；redo +0.044；层序 shu… |
| [`C3-inter-layer-topology`](C3-inter-layer-topology/README.md) | ⬜ stub | — | — |
| [`C4-weight-sharing`](C4-weight-sharing/README.md) | ⬜ stub | — | — |
| [`C5-refinement-operator`](C5-refinement-operator/README.md) | ✅ judged | 携带门是形态税的承载者之一，也是「任意深度可读」的结构来源。 | carry 门 + 填充 + 退出堆栈合计是形态税的主体（同引擎 +0.43… |
| [`C6-depth-supervision`](C6-depth-supervision/README.md) | ✅ judged | 出口资产只能由深度维监督造出来——这是本项目最有迁移价值的一条结论。 | 全深度等权 +0.067（13M）；dense 0.2 剂量的税为负（EX2… |

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
