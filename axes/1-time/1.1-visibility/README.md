# 1.1 可见范围（窗口族）· 分组

> 分组节点：只聚合子研究项，不写 `project.yaml`（`docs/structure.md` §13）。
> 轴：**1 时间轴** ｜ 用词以 `docs/structure.md` 为准。

一次前向里，一个 token「能看到谁」。本分组回答「可见范围」这条线：
把每个 token 的可见集合从全历史压到更小，代价是多少、值不值。

## 子节点

| 编号 | 研究项 | 目录 | 粒度 | 一句话 |
|---|---|---|---|---|
| 1.1.1 | hard 窗口 | [`1.1.1-hard-window`](../1.1.1-hard-window/) | 研究项 | 滑动窗口 / sink / 锚点槽：硬性裁可见集合 |
| 1.1.2 | soft 衰减 | [`1.1.2-soft-decay`](../1.1.2-soft-decay/) | 研究项 | 距离加性/乘性偏置：让远端变贵而非不可见 |
| 1.1.3 | 可见量上限 | [`1.1.3-visibility-cap`](../1.1.3-visibility-cap/) | 研究项 | 每 token 最多能看多少（1.1.1 的拆分位，只指针） |
| 1.1.4 | 信息右移 | [`1.1.4-right-shift`](../1.1.4-right-shift/) | 分组 | 窗口丢掉的，怎么搬到近处 |

## 共同口径

- 对照 = stock 全因果注意力（`axes/7-references`），归一为 100%。
- 判据 = clean bpc（该跌）+ KV 消融保留曲线 / 长程召回（该涨/该跌）。
- 代价 = 等效理论量（FLOPs / layer-passes / tokens-per-forward），**禁墙钟**。
- 每项有且仅有一个 milestone，推进过程见各自 `project.yaml` 的 work_packages。
