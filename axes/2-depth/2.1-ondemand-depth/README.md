# 2.1 按需深度 · 分组

> 分组节点：只聚合子研究项，不写 `project.yaml`（`docs/structure.md` §13）。
> 轴：**2 深度轴** ｜ 用词以 `docs/structure.md` 为准。

普通 decoder-only transformer 对每个 token 都跑满 L 层，而 token 难度不一样。本分组问：
能不能把「要不要多算」按 token 或按层决定，在**降低算力**（每 token layer-passes）的同时
保住**序列级**质量——即同一模型「Pro 当 Flash 用」，以及相反方向的「多算力换少参数」。

## 子节点

| 编号 | 研究项 | 目录 | 粒度 | 一句话 |
|---|---|---|---|---|
| 2.1.1 | early exit（最重要） | [`2.1.1-early-exit`](../2.1.1-early-exit/) | 研究项 | 逐 token 跑到够准就停（退出决策头 / 异构深度回看 / 深度监督配方 / 精化门） |
| 2.1.2 | skipping（跳层） | [`2.1.2-skipping`](../2.1.2-skipping/) | 研究项 | 跳过（或训练期随机丢）部分层，被跳层走恒等旁路 |
| 2.1.3 | repeating 循环 transformer：多算力换少参数 | [`2.1.3-repeating`](../2.1.3-repeating/) | Backlog | 同一组层参数重复前向多次，用算力换有效深度 |

## 共同口径

- 对照 = 同尺寸 **stock 全深度**（归一为 100%）+ 同预算最优固定深度；并须补「更浅 dense / MoD」省算力对手。
- 判据 = 序列级 perplexity/一致率（该不跌，须越 0.014 噪声带才算可排序）+ 每 token 平均 layer-passes 与跳过占比（该跌）。
- **墙钟不进判据**；代价 = 等效理论量（layer-passes / tokens-per-forward），考虑 pipelining。
- 「手段」不另立研究项：`2.1.1` 的 退出决策头 / 异构深度回看 / 深度监督配方 / 精化门 写进该 `project.yaml` 的 `work_packages`（`docs/structure.md` §13.3）。
