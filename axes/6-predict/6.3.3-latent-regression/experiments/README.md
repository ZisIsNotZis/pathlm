# 6.3.3 latent regression 批量生成 · 实验目录

本格每次判决占一个子目录（自包含）：`README.md`(契约) / 代码 / `results.json` / `analysis.md`(判读) / `MODEL.md`(权重哈希，权重本身不入库)。

命名：`YYMM-<干预>-<规模>-<引擎>`。

| 实验 | 来源 | 说明 |
|---|---|---|
| [`2660-latentreg-vs-disc-13m`](2660-latentreg-vs-disc-13m/README.md) | 目录内 README | WP1 最小判决：冻结 13M 主干上「latent（隐状态）匹配监督」vs 同参数量离散跳头；路径 (i) 直接发（teacher-forced 命中率）与路径 (ii) 当草稿再验证（α₁ / τ）各作一臂（63-F4 裁决） |
