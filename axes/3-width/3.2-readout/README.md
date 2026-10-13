# 3.2 读出 · 分组

> 分组节点：只聚合子研究项，不写 `project.yaml`（`docs/structure.md` §13）。
> 轴：**3 宽度轴** ｜ 用词以 `docs/structure.md` 为准。

**读出** = 把隐藏状态变成预测的那一层，也就是 unembedding / 输出层。普通 decoder-only transformer
用一张 tied 线性表（输出权重 = 输入 embedding 的转置 U = Eᵀ），于是「概率 = 隐藏状态在 embedding
基下的坐标」。本分组问读出的三种形态怎么影响质量与耦合：重参数化（加非线性 / GLU 式 / 线性，
3.2.1）、多出口时每个出口怎么映射到词表（3.2.2）、输入/输出是否绑定（3.2.3）。

## 子节点

| 编号 | 研究项 | 目录 | 粒度 | 一句话 |
|---|---|---|---|---|
| 3.2.1 | 读出的重参数化：加非线性 / GLU 式 / 线性 | [`3.2.1-readout-reparam`](../3.2.1-readout-reparam/) | 研究项 | 在读出前加一层可学变换，解耦「计算空间」与「读出可见空间」 |
| 3.2.2 | 逐出口读出：early exit 或多出口时，每个出口怎么映射到词表 | [`3.2.2-per-exit-readout`](../3.2.2-per-exit-readout/) | 研究项 | 多出口时共享一张表还是每出口一张 |
| 3.2.3 | tied vs untied embedding | [`3.2.3-tied-untied`](../3.2.3-tied-untied/) | 研究项 | 输入 embedding 与输出读出分不分开 |

## 共同口径

- 对照 = **tied 线性读出**（stock，归一为 100%）；重参数化类必须给**等参数纯线性对照**
  （否则把重参数化/容量记成非线性）。
- 判据 = clean bpc（该跌 / 可排序，须越 `docs/methodology.md` §2 的 0.014 噪声带）+ 出口资产 tf@k 与校准 ECE（该不跌）。
- 参数量口径分开记：重参数化 = 2·d²（本仓可 scale）；解绑 = vocab×d（本仓 vocab=206、小词表 +52736，大词表不可 scale）。
- **墙钟不进判据**；代价一律等效理论量（FLOPs / layer-passes / tokens-per-forward），考虑 pipelining。
- 每项有且仅有一个 `milestone`，推进过程见各自 `project.yaml` 的 work_packages（每条含独立 `review`）。
