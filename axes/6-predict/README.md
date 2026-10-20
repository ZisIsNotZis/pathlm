# 6 预测轴

> 轴：**6 预测轴**（目录 `axes/6-predict/`）。用词、编号、目录以 `docs/structure.md` §12 为准，
> 每个研究项的字段契约见 `harness/research-schema.md`。

预测轴问「一次前向产出几个位置、多出来的位置从哪来、怎么验证」。普通 decoder-only transformer
一次前向只产 1 个位置（自回归），生成长度 T 就要 T 次顺序前向。目标：用更少的**顺序前向次数**
（等效理论算力）拿到**序列级**类似效果；代价口径 = 等效理论时间/计算量
（FLOPs / layer-passes / tokens-per-forward，考虑 pipelining），**不用墙钟**。

## 本目录的子树

| 编号 | 名称 | 目录 | 粒度 |
|---|---|---|---|
| 6.2 | 多位置预测 = 投机解码 | [`6.2-speculative-decoding`](6.2-speculative-decoding/) | 研究项 |
| 6.3 | 整段并行生成（不止 AR） | [`6.3-parallel-generation`](6.3-parallel-generation/) | 分组 |
| 6.3.1 | diffusion（离散扩散 / 掩码扩散） | [`6.3.1-diffusion`](6.3.1-diffusion/) | 研究项 |
| 6.3.2 | 一次性批量 pad 填充（填空式） | [`6.3.2-pad-fill`](6.3.2-pad-fill/) | 研究项 |
| 6.3.3 | latent regression 批量生成（估计得贪心） | [`6.3.3-latent-regression`](6.3.3-latent-regression/) | 研究项 |
| 6.4 | 训练时未来位置的填充语义：pad / self-fill / 错误 token | [`6.4-training-fill`](6.4-training-fill/) | 研究项 |

> `6.2` 的 `work_packages` 用 `6.2.1 草稿头（应免费、不动主干）`、`6.2.2 概率多候选 / 树（+2 / +1+1 / 加权组合）`、
> `6.2.3 验证`——这三者是**同一个投机解码研究问题的手段**，不另立研究项（`docs/structure.md` §13.3）。

## 前提（§6.1 自回归，参照）

| 编号 | 名称 | 粒度 | 家 |
|---|---|---|---|
| 6.1 | 自回归（一次一个位置） | 参照 | `axes/7-references/README.md` |

**6.1 自回归**是**参照**（逐 token 贪心 = 全因果 stock 的输出流），本轴不写 `project.yaml`、不建目录
（`docs/structure.md` §13.4）；它登记在 `axes/7-references/README.md` 的参照台账，同时是本轴
`6.2` / `6.3` / `6.4` 的对照基线（归一为 100%）。

## 共同口径

- 对照 = 同尺寸 **stock 全因果逐 token 贪心**（归一为 100%）；投机解码另有诚实对手「1-step AR 两次前向」。
- 判据 = 平均接受长度 τ / tokens-per-forward（该涨）+ 序列级质量（该不跌）+ 每 token 顺序前向次数（该跌）。
- 每 token 等效 FLOPs 是**代价轴**、如实报：并行 / 多位置预测必然用更多并行算力换更少顺序前向，
  「FLOPs 涨」是该范式的定义性质，不是亏损发现。
- **墙钟不进判据**；代价一律等效理论量（FLOPs / layer-passes / tokens-per-forward），考虑 pipelining。
- 每项有且仅有一个 `milestone`，推进过程见各自 `project.yaml` 的 `work_packages`（每条含独立 `review`）。

## 迁移说明

按 `.scratch/56-restructure/MAPPING.md`：旧格 `H1-outputs-per-pass` 迁入 `6.2`（本体），
`H3-composition`、`H4-head-trunk-relation`、`H5-verification` 分别作为 `6.2` 的手段
（WP `6.2.2 概率多候选 / 树` / `6.2.1 草稿头` / `6.2.3 验证`）并入其 `experiments/`；
旧格 `H2-training-fill` 迁入 `6.4`。`6.3` 及其三个子项（diffusion / 一次性批量 pad 填充 /
latent regression 批量生成）无旧格、为本轮新建。`6.1` 自回归降为参照，登记在
`axes/7-references/README.md`。旧轴目录 `archive/wiki-legacy/axes/H-horizon/**` 及其 README 已随旧树退役归档（归档只读保留）。

## 怎么读

- 状态与全表：`python3 tools/summary.py`；校验：`python3 tools/summary.py --check`。
- `status` 仅四值：未开始 / 进行中 / 已完成 / 被否证；「做到多少」写在 `conclusion` / `gaps`。
- 旧格（`archive/wiki-legacy/axes/H-horizon/**`）及其 README 为**上一版结构**，证据已按 MAPPING 迁入本目录
  `experiments/`，旧目录退役完成（归档只读保留）。
