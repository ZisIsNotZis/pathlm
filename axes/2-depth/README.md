# 2 深度轴

> 轴：**2 深度轴**（目录 `axes/2-depth/`）。用词、编号、目录以 `docs/structure.md` §12 为准，
> 每个研究项的字段契约见 `harness/research-schema.md`。

深度轴问「算力沿深度怎么花」：普通 decoder-only transformer 对每个 token 都跑满 L 层、
每层用同一种连接、每层容量相同；本轴把「要不要跑这层」「跑完怎么接回去」「深度的形状」
三件事分开研究。目标：用更少的计算量拿到**序列级**类似效果；代价口径 = 等效理论时间/计算量
（FLOPs / layer-passes / tokens-per-forward，考虑 pipelining），**不用墙钟**。

## 本目录的子树

| 编号 | 名称 | 目录 | 粒度 |
|---|---|---|---|
| 2.1 | 按需深度 | [`2.1-ondemand-depth`](2.1-ondemand-depth/) | 分组 |
| 2.1.1 | early exit（最重要） | [`2.1.1-early-exit`](2.1.1-early-exit/) | 研究项 |
| 2.1.2 | skipping（跳层） | [`2.1.2-skipping`](2.1.2-skipping/) | 研究项 |
| 2.1.3 | repeating 循环 transformer：多算力换少参数 | [`2.1.3-repeating`](2.1.3-repeating/) | Backlog |
| 2.2 | 层间怎么聚合（残差族） | [`2.2-layer-aggregation`](2.2-layer-aggregation/) | 分组 |
| 2.2.1 | 残差的形式 | [`2.2.1-residual-form`](2.2.1-residual-form/) | 研究项 |
| 2.2.2 | 用 causal attention 代替残差 | [`2.2.2-attention-residual`](2.2.2-attention-residual/) | 研究项 |
| 2.2.3 | 层间学习加权和（DenseFormer 式） | [`2.2.3-learned-layer-sum`](2.2.3-learned-layer-sum/) | 研究项 |
| 2.2.4 | 并行子层 | [`2.2.4-parallel-sublayer`](2.2.4-parallel-sublayer/) | 研究项 |
| 2.2.5 | 权重共享 | [`2.2.5-weight-sharing`](2.2.5-weight-sharing/) | Backlog |
| 2.3 | 深度的形状 | [`2.3-depth-shape`](2.3-depth-shape/) | 分组 |
| 2.3.1 | 逐层容量变化：前大后小 / 逐渐放大 | [`2.3.1-layer-capacity`](2.3.1-layer-capacity/) | 研究项 |
| 2.3.2 | 逐层分辨率：先粗后细、层级式（层级合并） | [`2.3.2-layer-resolution`](2.3.2-layer-resolution/) | 研究项 |

## 共同口径

- 对照 = 同尺寸 **stock 全深度**（归一为 100%）；「更浅 dense / 深度剪枝 / MoE」作为省算力对手。
- 判据 = 序列级 perplexity/一致率（该不跌/可排序）+ 每 token 平均 layer-passes 与跳过占比（该跌）。
- **墙钟不进判据**；代价一律等效理论量（FLOPs / layer-passes / tokens-per-forward），考虑 pipelining。
- 每项有且仅有一个 `milestone`，推进过程见各自 `project.yaml` 的 work_packages。

## 迁移说明

按 `.scratch/56-restructure/MAPPING.md`：旧格 `archive/wiki-legacy/axes/C-vertical/C1-layers-per-token` 迁入 `2.1.1`；
`C2-layer-selection-order` 并入 `2.1.2`；`C3-inter-layer-topology` 拆入 `2.2.3`（主家）与 `2.2.4`
（指针）；`C4-weight-sharing` 降为 `2.2.5`（Backlog）；`C5-refinement-operator`、`C6-depth-supervision`
作为 `2.1.1` 的手段（精化门 / 深度监督配方）并入其 experiments。`2.2.1` / `2.2.2` / `2.3.1` / `2.3.2`
无旧格、为本轮新建。旧 `archive/wiki-legacy/axes/C-vertical/**` 目录与其 README 已随旧树退役归档（归档只读保留）。

## 怎么读

- 状态与全表：`python3 tools/summary.py`；校验：`python3 tools/summary.py --check`。
- `status` 仅四值：未开始 / 进行中 / 已完成 / 被否证；「做到多少」写在 `conclusion` / `gaps`。
