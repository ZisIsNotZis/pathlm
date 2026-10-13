# 1.1.4 信息右移 · 分组

> 分组节点：只聚合子研究项，不写 `project.yaml`（`docs/structure.md` §13）。
> 轴：**1 时间轴** ｜ 用词以 `docs/structure.md` 为准。

窗口族（1.1.1–1.1.3）丢掉的远距离信息，怎么**搬到近处**——目标是让前缀可以被真正丢掉，
而模型仍是 causal decoder。判据必须是保留/召回曲线 + 质量-代价前沿，不是「窗口是否无损」。

## 子节点

| 编号 | 研究项 | 目录 | 粒度 | 一句话 |
|---|---|---|---|---|
| 1.1.4.1 | 显式记忆槽 / 线性状态 / 递归状态（delta 规则族：DeltaNet / GLA / Mamba-2） | [`1.1.4.1-explicit-memory`](../1.1.4.1-explicit-memory/) | 研究项 | 写进固定状态、线性读 |
| 1.1.4.2 | 压缩：摘要槽 / 低秩投影 / pooling | [`1.1.4.2-compression`](../1.1.4.2-compression/) | 研究项 | 压更小再读 |
| 1.1.4.3 | 压缩触发与优先级 | [`1.1.4.3-compression-trigger`](../1.1.4.3-compression-trigger/) | 研究项 | 何时压、先压谁 |
| 1.1.4.4 | 外部检索 | [`1.1.4.4-external-retrieval`](../1.1.4.4-external-retrieval/) | 研究项 | 存外部索引、按内容检索 |
| 1.1.4.5 | implicit（掩码/窗口，已证退化）vs **explicit**（有希望的方向） | [`1.1.4.5-implicit-vs-explicit`](../1.1.4.5-implicit-vs-explicit/) | 研究项 | 两类右移机制正面对撞 |

## 共同口径

- 对照 = stock 全因果注意力（`axes/7-references`，归一为 100%）+ 同掩码无记忆地板 + 同预算窗口/量化点。
- 判据 = 长程召回 / 保留曲线（该涨）+ 每位置内存字节、每 token 额外 FLOPs（该跌）+ 质量-代价前沿。
- 必要条件的可证伪探针 = 毁源敏感度（删被评行之前的前缀，看 logits 是否变）。
- 代价 = 等效理论量（FLOPs / layer-passes / tokens-per-forward），**禁墙钟**。
- 既有巨人基线（Compressive Transformer / Memorizing Transformer / DeltaNet / GLA / Mamba-2 / 外部检索）必须作为要打平或超越的对手，不能只用自写对照。
