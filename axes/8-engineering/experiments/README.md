# 8 工程纪律 · 实验目录

本目录收**工程纪律**相关的实验（旧修饰符格 P5 的自有实验）。每次判决占一个子目录（自包含）：
`README.md`(契约) / 代码 / `results.json` / `analysis.md`(判读) / `MODEL.md`(权重哈希，权重本身不入库)。

| 实验 | 来源 | 说明 |
|---|---|---|
| [`2523b-shallow-weight-sweep`](2523b-shallow-weight-sweep/README.md) | 目录内 README | 旧格 P5 迁入 → 浅层 aux 降权扫描（w∈{1,0.1,0.02,退火}）：**全败且单调反向**（depth-L 2.45→2.57→2.73）⇒ 浅层全深度监督是表示正则而非稀释；退火臂 2.4444 保底座但税仍 +0.71 |

> 旧格 `P5-loss-weighting` 按 `.scratch/56-restructure/MAPPING.md` §1 迁入本台账：其
> `experiments/2523b-shallow-weight-sweep` 整目录 `git mv` 到本目录（一格一份家，不复制）。
> 旧格目录与其 README 仍保留在 `archive/wiki-legacy/axes/P-modifiers/P5-loss-weighting/**`，旧树退役已由本轮 integrator 完成（旧目录归档只读保留）。
