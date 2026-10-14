# 6.3.2 一次性批量 pad 填充（填空式） · 实验目录

本格每次判决占一个子目录（自包含）：`README.md`(契约) / 代码 / `results.json` / `analysis.md`(判读) / `MODEL.md`(权重哈希，权重本身不入库)。

命名：`YYMM-<干预>-<规模>-<引擎>`。

| 实验 | 来源 | 说明 |
|---|---|---|
| [`2671-padfill-13m`](2671-padfill-13m/README.md) | 目录内 README | WP1 最小判决：{AR, 双向填空 r=10/25/50%, 因果填空 r=25%} 同 harness 单因素；含「不受未来泄漏」硬门 + gate probe；非因果注意力由本目录自带 harness 提供（共享引擎硬编码因果，见该目录 `README.md` §2） |
