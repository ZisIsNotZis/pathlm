# 3 宽度轴

> 轴：**3 宽度轴**（目录 `axes/3-width/`）。用词、编号、目录以 `docs/structure.md` §12 为准，
> 每个研究项的字段契约见 `harness/research-schema.md`。

宽度轴问「算力沿宽度怎么花」：普通 decoder-only transformer 对每个 token、每一层、每个位置都用
同一宽度（同 hidden 维、同 FFN 隐藏比、激活全部通道），把每 token 的算力/参数均匀摊在所有 token 上。
本轴把「**每 token 的激活宽度**」（每个 token 用多少通道跑完）与「**读出**」（把隐藏状态变成预测的
那一层怎么设计）两件事分开研究。目标：用更少的计算量拿到**序列级**类似效果；代价口径 = 等效理论
时间/计算量（FLOPs / layer-passes / tokens-per-forward，考虑 pipelining），**不用墙钟**。

> **前沿口径**：宽度轴的正解不是「掉不掉质量」的单点，而是**质量-代价前沿**——以同尺寸 stock
> **全宽**为 100%，横轴 = 每 token 等效算力（FLOPs）/ 参数量，纵轴 = 序列级质量（clean bpc）。
> 各研究项都必须给**等参数 / 等 FLOPs 对照**，否则会把「容量变化」误读成「宽度分配」。

## 本目录的子树

| 编号 | 名称 | 目录 | 粒度 |
|---|---|---|---|
| 3.1 | 每 token 的激活宽度 | [`3.1-active-width`](3.1-active-width/) | 分组 |
| 3.1.1 | uniform 减宽 / width dropout | [`3.1.1-uniform-narrowing`](3.1.1-uniform-narrowing/) | 研究项 |
| 3.1.2 | 逐层宽度变化：金字塔 / 沙漏 | [`3.1.2-layer-width`](3.1.2-layer-width/) | 研究项 |
| 3.1.3 | 同层内位置相关宽度：前 1024 与后 1024 意义不同 | [`3.1.3-position-width`](3.1.3-position-width/) | 研究项 |
| 3.1.4 | 前馈隐藏比：FF/MLP 的 hidden ratio，及其逐层变化 | [`3.1.4-ffn-ratio`](3.1.4-ffn-ratio/) | 研究项 |
| 3.1.5 | 通道极化 / 稀疏激活：只激活部分通道 | [`3.1.5-channel-sparsity`](3.1.5-channel-sparsity/) | 研究项 |
| 3.2 | 读出 | [`3.2-readout`](3.2-readout/) | 分组 |
| 3.2.1 | 读出的重参数化：加非线性 / GLU 式 / 线性 | [`3.2.1-readout-reparam`](3.2.1-readout-reparam/) | 研究项 |
| 3.2.2 | 逐出口读出：early exit 或多出口时，每个出口怎么映射到词表 | [`3.2.2-per-exit-readout`](3.2.2-per-exit-readout/) | 研究项 |
| 3.2.3 | tied vs untied embedding | [`3.2.3-tied-untied`](3.2.3-tied-untied/) | 研究项 |

## 边界：不属于本轴的两件旧事

旧「宽度轴」把四件事混在一起（55-review `W-PROBLEMS.md` W-01/W-02）；本轴按 `docs/structure.md`
只收「每 token 激活宽度」与「读出」，另两件归属他轴：

| 旧格 | 内容 | 新家 |
|---|---|---|
| W2 专家化（MoE） | 稀疏路由把参数量与每 token FLOPs 解耦 = **保容量减计算** | 专家轴 `4`（`4.1` 稀疏激活专家） |
| W3 KV 头共享（GQA/MQA/MLA） | 省**推理显存/带宽**，不是「每 token 激活宽度」 | 参照台账 `axes/7-references/`（不立项） |

> W2/W3 的证据物理位置由轴 4 / 轴 7 的 worker 按 `.scratch/56-restructure/MAPPING.md` 迁走；
> 本轴不接收、不复制。

## 跨轴边界（须由 integrator 核不矛盾）

- `3.1.2` 逐层宽度变化 与 `2.3.1` 逐层容量变化（深度轴）在实现上重叠（都是「哪层大哪层小、总参数固定」）。
- `3.1.5` 通道极化 / 稀疏激活（通道粒度）与专家轴 `4` MoE（专家粒度的 token 级路由）相邻。
- `3.2.2` 逐出口读出（每出口怎么映射到词表）与深度轴 `2.1.1` early exit（跑不跑这层）相邻；跑层归
  `2.1.1`，出口→词表的形态归本轴。

## 迁移说明

按 `.scratch/56-restructure/MAPPING.md`：旧格 `W1-active-width` 迁入 `3.1.1`，
`W4-readout-geometry` 迁入 `3.2.1`（证据 `experiments/**` 整目录迁入对应新家）；旧格
`W2-expertization` 归轴 4、`W3-kv-head-sharing` 归轴 7（见上表）。`3.1.2` / `3.1.3` / `3.1.4` /
`3.1.5` / `3.2.2` / `3.2.3` 无旧格、为本轮新建（MAPPING §2）。旧轴目录 `archive/wiki-legacy/axes/W-width/**` 及其
README、`PRIOR_ART.md` 已随旧树退役归档（归档只读保留）。

## 怎么读

- 状态与全表：`python3 tools/summary.py`；校验：`python3 tools/summary.py --check`。
- `status` 仅四值：未开始 / 进行中 / 已完成 / 被否证；「做到多少」写在 `conclusion` / `gaps`。
- 旧格（`archive/wiki-legacy/axes/W-width/**`）及其 README 为**上一版结构**，证据已按 MAPPING 迁入本目录
  `experiments/`，旧目录退役完成（归档只读保留）。
