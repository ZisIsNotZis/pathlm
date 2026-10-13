# 1.2 可见结构（非均匀的可见模式）· 分组

> 分组节点：只聚合子研究项，不写 `project.yaml`（`docs/structure.md` §13）。
> 轴：**1 时间轴** ｜ 用词以 `docs/structure.md` 为准。

一次前向里，一个 token「能看到谁」的**模式**不只有「最近 k 个」这一种：可以是带空洞的跨步采样、
预定的块/条带/局部+全局掩码、按内容选的 top-k，或随深度递减的合并 token。本分组回答这条线：
在每 token 可见量远小于 T 的前提下，哪种**非均匀可见结构**最值。

## 子节点

| 编号 | 研究项 | 目录 | 粒度 | 一句话 |
|---|---|---|---|---|
| 1.2.1 | dilated / 空洞 | [`1.2.1-dilated`](../1.2.1-dilated/) | 研究项 | 近端窗口 + 每 d 步取一个：同可见量下扩感受野 |
| 1.2.2 | 块稀疏 / 条带 / 局部+全局混合（Longformer 式） | [`1.2.2-block-sparse`](../1.2.2-block-sparse/) | 研究项 | 预定块稀疏/条带/局部+全局掩码；含线性核算子臂 |
| 1.2.3 | 内容选择 top-k | [`1.2.3-content-topk`](../1.2.3-content-topk/) | 研究项 | 按 q·kᵀ 选内容最像的 k 个键，不看位置 |
| 1.2.4 | 动态合并：越往下层 token 越少 | [`1.2.4-dynamic-merge`](../1.2.4-dynamic-merge/) | 研究项 | 层间合并 token，使深层每层处理更少位置 |

## 共同口径

- 对照 = stock 全因果注意力（`axes/7-references`），归一为 100%。
- 判据 = clean bpc（该跌/该不变）+ 长程召回（该涨）+ 每 token 等效 FLOPs 与每位置缓存字节（该跌）。
- **同预算对齐**：比较跨结构（dilated / 块稀疏 / 内容 top-k / 动态合并）时必须先对齐
  「每 token 可见键数」或「每 token 等效总 FLOPs」，否则不可比。
- 代价 = 等效理论量（FLOPs / layer-passes / tokens-per-forward），**禁墙钟**。
- 每项有且仅有一个 milestone，推进过程见各自 `project.yaml` 的 work_packages。

> 旧格 `axes/A-mixing/A6-mixing-operator`（混合算子）按 `.scratch/56-restructure/MAPPING.md` §3
> 拆入本分组：`2614-topk` → `1.2.3`；`2616-linear-attention`、`2637-linear-attn-quality` → `1.2.2`；
> 长 T 窗口系列归 `1.1.1`。旧目录待统一退役。
