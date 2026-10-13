# 2.2.3 层间学习加权和（DenseFormer 式） · 实验目录

本格每次判决占一个子目录（自包含）：`README.md`(契约) / 代码 / `results.json` / `analysis.md`(判读) / `MODEL.md`(权重哈希，权重本身不入库)。

命名：`YYMM-<干预>-<规模>-<引擎>`。

| 实验 | 来源 | 说明 |
|---|---|---|
| [`2613-inter-layer-topology-13m`](2613-inter-layer-topology-13m/README.md) | 目录内 README | 旧格 C3 迁入：四臂层间接法（sequential / parallel / parallel-dual-ln / dense） |

> 旧格 `C3-inter-layer-topology` 按 `.scratch/56-restructure/MAPPING.md` §1 拆入 `2.2.3`（本项，主家）
> 与 `2.2.4`（并行子层，只指针 see 本目录、不复制文件）。
> 本实验的 `dense` 臂 = 层间学习加权和（本项主判据）；`parallel` / `parallel-dual-ln` 臂 = 并行子层（归 2.2.4）。
> 出资人 C3 批注提出的**跨层 causal attention** 设计本实验**未测**（旧 README 自认未测），由 `2.2.2` 承接。
> 旧格目录与其 README 仍保留在 `axes/C-vertical/C3-inter-layer-topology`，待 integrator 统一退役。
