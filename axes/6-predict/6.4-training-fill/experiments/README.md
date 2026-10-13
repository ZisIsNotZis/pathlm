# 6.4 训练时未来位置的填充语义：pad / self-fill / 错误 token · 实验目录

本格每次判决占一个子目录（自包含）：`README.md`(契约) / 代码 / `results.json` / `analysis.md`(判读) / `MODEL.md`(权重哈希，权重本身不入库)。

命名：`YYMM-<干预>-<规模>-<引擎>`。

| 实验 | 来源 | 说明 |
|---|---|---|
| [`2523e-fill-variants`](2523e-fill-variants/README.md) | 目录内 README | 旧格 H2 迁入 → 填充策略对决：fill / ragged nofill / proj-fill（深度 AR，12k matched） |

> 旧格 `H2-training-fill` 按 `.scratch/56-restructure/MAPPING.md` §1 并入本项：其 `experiments/**`
> 整目录迁入本目录。旧格目录与其 README 已迁出 `axes/`、只读保留于
> `archive/wiki-legacy/axes/H-horizon/H2-training-fill`（旧树退役已由本轮 integrator 完成）。
> 填充语义的其余读数（mask / 错误 token / 软 latent / 纯噪声）以 `.scratch` 原始证据为锚
> （见 `project.yaml` 的 `experiments`），本目录只放仓内自包含实验。
