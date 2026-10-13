# 1.1.4.2 压缩：摘要槽 / 低秩投影 / pooling · 实验目录

本格每次判决占一个子目录（自包含）：`README.md`(契约) / 代码 / `results.json` / `analysis.md`(判读) / `MODEL.md`(权重哈希，权重本身不入库)。

命名：`YYMM-<干预>-<规模>-<引擎>`。

| 实验 | 来源 | 说明 |
|---|---|---|
| [`2614-kv-ablation-retention-13m`](2614-kv-ablation-retention-13m/README.md) | 目录内 README | KV 删除保留仪器（量依赖，非压缩方法） |
| [`2615-retention-retrieval-leg-13m`](2615-retention-retrieval-leg-13m/README.md) | 目录内 README | 保留 + 检索腿仪器 |

> 这两个实验量的是「删 KV 的保留曲线」，不是任何压缩方法（55-review M-18）；本项自身的压缩实验（摘要槽 / 低秩投影 / pooling）尚未做，见 `project.yaml` 的 work_packages。
