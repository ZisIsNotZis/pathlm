# 2.2 层间怎么聚合（残差族） · 分组

> 分组节点：只聚合子研究项，不写 `project.yaml`（`docs/structure.md` §13）。
> 轴：**2 深度轴** ｜ 用词以 `docs/structure.md` 为准。

普通 decoder-only transformer 用一条残差把上一层接到下一层。本分组问：**改层间的接法**
（残差的形式、用 causal attention 代替残差、对历史层做学习加权和、并行子层、权重共享）
是否能在质量-代价前沿上取得净增量。

## 子节点

| 编号 | 研究项 | 目录 | 粒度 | 一句话 |
|---|---|---|---|---|
| 2.2.1 | 残差的形式 | [`2.2.1-residual-form`](../2.2.1-residual-form/) | 研究项 | pre/post-norm、缩放/门控残差、残差随深度衰减 |
| 2.2.2 | 用 causal attention 代替残差 | [`2.2.2-attention-residual`](../2.2.2-attention-residual/) | 研究项 | 把各历史层输出 stack 起来做 causal attention 聚合 |
| 2.2.3 | 层间学习加权和（DenseFormer 式） | [`2.2.3-learned-layer-sum`](../2.2.3-learned-layer-sum/) | 研究项 | 每层接全部历史层的学习加权和 |
| 2.2.4 | 并行子层 | [`2.2.4-parallel-sublayer`](../2.2.4-parallel-sublayer/) | 研究项 | 一层内注意力与 MLP 并行读同一输入（PaLM/GPT-J 式） |
| 2.2.5 | 权重共享 | [`2.2.5-weight-sharing`](../2.2.5-weight-sharing/) | Backlog | 层间复用同一组权重（省参数轴） |

## 共同口径

- 对照 = 同尺寸 **stock 单残差全深度**（归一为 100%）+ 同参数/同算力对照臂。
- 判据 = clean bpc（序列级，该跌/可排序）+ 逐深度 bpc（中途截断可用性）+ 额外参数/额外 FLOPs（代价列）。
- **墙钟不进判据**；代价 = 等效理论量（FLOPs / layer-passes / tokens-per-forward），考虑 pipelining。
- 旧格 C3 的 **层间拓扑** 负结论按 `.scratch/56-restructure/MAPPING.md` §1 拆入 `2.2.3`（层间学习加权和，
  证据主家）与 `2.2.4`（并行子层，只指针 see `2.2.3`）；出资人 C3 批注提出的**跨层注意力**设计
  （从未测，55-review C-P10）由 `2.2.2` 承接。
