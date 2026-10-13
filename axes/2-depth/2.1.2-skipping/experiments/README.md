# 2.1.2 skipping（跳层） · 实验目录

本格每次判决占一个子目录（自包含）：`README.md`(契约) / 代码 / `results.json` / `analysis.md`(判读) / `MODEL.md`(权重哈希，权重本身不入库)。

命名：`YYMM-<干预>-<规模>-<引擎>`。

| 实验 | 来源 | 说明 |
|---|---|---|
| [`2506-m2-element-tax-law`](2506-m2-element-tax-law/README.md) | 目录内 README | 旧格 C2 迁入：元素税定律（旧 M2 引擎，p_skip 逐序列） |
| [`2509-m3-composition-law`](2509-m3-composition-law/README.md) | 目录内 README | 旧格 C2 迁入：组合律 + 多样性压力（旧 M2 引擎） |

> 旧格 `C2-layer-selection-order` 按 `.scratch/56-restructure/MAPPING.md` §1 并入本项（C2 → 2.1.2）；
> 其「层序 shuffle」子项按 55-review C-P18 不立研究项（数字留 playbook note），「重复（redo）」归 2.1.3 repeating。
> 两个实验的 `p_skip` 由 `sample_path` **逐序列/逐 batch** 采样（不是逐 token），且属旧 M2 引擎，
> 故本项的税数字须在 depth-ar 引擎按序列级判据重测（见 `project.yaml` 的 WP1）。
> 旧格目录与其 README 仍保留在 `axes/C-vertical/C2-layer-selection-order`，待 integrator 统一退役。
