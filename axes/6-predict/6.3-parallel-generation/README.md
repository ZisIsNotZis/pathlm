# 6.3 整段并行生成（不止 AR） · 分组

> 分组节点：只聚合子研究项，不写 `project.yaml`（`docs/structure.md` §13）。
> 轴：**6 预测轴** ｜ 用词以 `docs/structure.md` 为准。

自回归逐 token 生成把「生成长度」直接变成「顺序前向次数」。本分组问：能不能让一次前向刷新
**整段（或一大段）**，把顺序依赖次数压到远小于序列长度 T。三条路线按「一次前向产出的位置从
哪来」区分：迭代去噪（diffusion）、一次填空（批量 pad 填充）、一次回归（latent regression）。

## 子节点

| 编号 | 研究项 | 目录 | 粒度 | 一句话 |
|---|---|---|---|---|
| 6.3.1 | diffusion（离散扩散 / 掩码扩散） | [`6.3.1-diffusion`](../6.3.1-diffusion/) | 研究项 | 从全 mask 出发迭代去噪 K 步，每步并行刷新整段（顺序前向由 T 降到 K） |
| 6.3.2 | 一次性批量 pad 填充（填空式） | [`6.3.2-pad-fill`](../6.3.2-pad-fill/) | 研究项 | 一次前向把所有待填位置填出（顺序前向 ≈ 1，无迭代修正） |
| 6.3.3 | latent regression 批量生成（估计得贪心） | [`6.3.3-latent-regression`](../6.3.3-latent-regression/) | 研究项 | 一次前向回归出未来 k 个位置的 latent，再各自经共享读出贪心 |

## 共同口径

- 对照 = 同尺寸 **stock 全因果 AR**（归一为 100%）；同一 token 预算、同一评测口径同框。
- 判据 = tokens-per-forward（一次前向产出的位置数，该涨）+ 每 token 顺序前向次数（该跌）+
  序列级 bpc / perplexity（该不跌，须越 0.014 噪声带才算可排序）。
- 每 token 等效 FLOPs 是**代价轴**、如实报：并行生成必然用更多并行算力换更少顺序前向，
  「FLOPs 涨」是该范式的定义性质，不是亏损发现（`SCHEMA.md` §10–§17 的判据取向）。
- **墙钟不进判据**；代价 = 等效理论量（FLOPs / layer-passes / tokens-per-forward），考虑 pipelining。
- 「手段」不另立研究项；三条路线都是**可独立收口、可独立否证**的问题，故各写一份 `project.yaml`
  （`docs/structure.md` §13.1/§13.3）。
