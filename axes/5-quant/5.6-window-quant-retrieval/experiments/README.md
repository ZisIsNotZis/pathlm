# 5.6 组合：窗口 + 量化 + 检索 · 实验目录

本格每次判决占一个子目录（自包含）：`README.md`(契约) / 代码 / `results.json` / `analysis.md`(判读) / `MODEL.md`(权重哈希，权重本身不入库)。

命名：`YYMM-<干预>-<规模>-<引擎>`。

| 实验 | 来源 | 说明 |
|---|---|---|
| [`2617-prior-art-hybrid-kv`](2617-prior-art-hybrid-kv/PRIOR_ART.md) | 目录内 PRIOR_ART | 旧格 R4 迁入 → 混合层比例 / GQA·MLA / KV 预算 Pareto 的既有做法情报（外部条目 `记忆·未核`，只作设计方向、不作判据；§3 记「业界缺统一 KV 字节预算比较」） |

> 旧格 `R4-window-kv-quant-retrieval` 按 `.scratch/56-restructure/MAPPING.md` §1 迁入本项：其
> `experiments/2617-prior-art-hybrid-kv` 整目录迁入本目录（整目录 `git mv`，一格一份家，不复制）。
> 旧格目录与其 README 仍保留在 `archive/wiki-legacy/axes/R-references/R4-window-kv-quant-retrieval/**`，旧树退役已由本轮 integrator 完成（旧目录归档只读保留）。

## 指针（他线的本地证据，不复制、不重复登记）

- 窗口 / 锚点腿：`.scratch/07-x1-x3/evidence @7b47555`（原址不动，作锚点）。
- 保留曲线 / 检索腿天平：`axes/1-time/1.1.4.2-compression/experiments/2614-kv-ablation-retention-13m/analysis.md @c39dc88`
  （旧 M2-carrier-compression 的实验，按 MAPPING §1 已迁入 `1.1.4.2`；本项只以指针引用）。
- 量化腿字节口径：见 `5.1-kv-quant`（块标尺，`2632`）；本项**不得**引用已作废的逐值标尺 int4 数字。
