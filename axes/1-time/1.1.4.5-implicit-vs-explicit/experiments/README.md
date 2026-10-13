# 1.1.4.5 implicit vs explicit · 实验目录

本格每次判决占一个子目录（自包含）：`README.md`(契约) / 代码 / `results.json` / `analysis.md`(判读) / `MODEL.md`(权重哈希，权重本身不入库)。

命名：`YYMM-<干预>-<规模>-<引擎>`。

| 实验 | 来源 | 说明 |
|---|---|---|
| [`2615-window-trained-retention-13m`](2615-window-trained-retention-13m/README.md) | 目录内 README | 窗口训练依赖检验（implicit 臂） |
| [`2623-window-on-longrange-13m`](2623-window-on-longrange-13m/README.md) | 目录内 README | 训练 window vs full 的合成长程判决（implicit 臂） |

> explicit 半边（写通道）的实现与读数见 `1.1.4.1`；本项聚焦两类机制的正面比较（55-review §6 第 2 条）。
