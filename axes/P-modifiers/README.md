<!-- AUTO:begin -->

# P · 修饰符（不是轴）

- **原语**：横切（不改变路径空间）
- **命题**：只改变实现/成本/条件的决策：精度、条件化信号、参数插入与初始化、归一化位置、损失权重、数据构造。

## 定义

修饰符可以作用在任意轴上，因此不是轴；但它们决定实验口径与可比性，
所以必须成页（否则无法复现别人的实验）。

## 子维度

P1 精度 · P2 条件化信号 · P3 参数插入与初始化 · P4 归一化位置 · P5 损失权重 · P6 数据构造

## 本地总体结论

本地踩过两个坑（已上升为定律）：bf16 autocast 下的 reward/consistency 项
必须 float32 + pre-step grad-norm gate（两次发散）；读出必须 fp32 头。
正资产：carry 门负偏置初始化（恒等起步）、proj-fill 的逐层恒等初始化
adapter（13M 族内最优 1.8356 vs fill 1.8582 vs nofill 1.9438）。
无 final LayerNorm 是同空间前提的一部分。

## 格子索引

| 格子 | 状态 | 判读 | 税 |
|---|---|---|---|
| [`P1-precision`](P1-precision/README.md) | ✅ judged | reward/判据类损失必须 float32 + pre-step grad-norm gate。 | — |
| [`P2-conditioning`](P2-conditioning/README.md) | ◐ partial | 深度索引（逐深度读出）与置信度条件（门控）已用；噪声级/任务条件零经验。 | 门控/条件化是本形态结构的一部分（税见 C1） |
| [`P3-param-insertion-init`](P3-param-insertion-init/README.md) | ✅ judged | 恒等初始化 adapter 在 13M matched-step 族内最优——「从基线出发学 remap」是对的。 | proj-fill 1.8356 < fill 1.8582 < nofil… |
| [`P4-normalization`](P4-normalization/README.md) | ◐ partial | pre-LN + 无 final norm 是「同空间」几何的前提，已固定使用。 | — |
| [`P5-loss-weighting`](P5-loss-weighting/README.md) | ✅ judged | 浅层降权单调反向；退火是当前最优配方的组成部分。 | 浅层降权全败且单调反向（w∈{1,0.1,0.02,退火}：depth-L … |
| [`P6-data-construction`](P6-data-construction/README.md) | ✅ judged | 任务构造本身是本项目最大的单点偏差来源（enwik8 局部统计极强）。 | — |

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
