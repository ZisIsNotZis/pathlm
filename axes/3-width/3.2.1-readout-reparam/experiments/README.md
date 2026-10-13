# 3.2.1 读出的重参数化：加非线性 / GLU 式 / 线性 · 实验目录

本格每次判决占一个子目录（自包含）：`README.md`(契约) / 代码 / `results.json` / `analysis.md`(判读) / `MODEL.md`(权重哈希，权重本身不入库)。

命名：`YYMM-<干预>-<规模>-<引擎>`。

| 实验 | 来源 | 说明 |
|---|---|---|
| [`2610-gated-readout-13m-depthar`](2610-gated-readout-13m-depthar/README.md) | 目录内 README | 旧格 W4 迁入 → WP1：GLU 式 gated 残差读出（12k，Δ clean bpc −0.0812） |
| [`2611-equalparams-linear-control-13m`](2611-equalparams-linear-control-13m/README.md) | 目录内 README | 旧格 W4 迁入 → WP1：等参数纯线性对照（Δ −0.0888 ⇒ 与 gated 不可排序，归因容量/重参数化） |
| [`2612-gated-readout-24k-13m`](2612-gated-readout-24k-13m/README.md) | 目录内 README | 旧格 W4 迁入 → WP2：24k 复测（Δ −0.0463，缩 43% ⇒ 疑收敛加速） |
| [`2613-readout-reparam-dense-13m`](2613-readout-reparam-dense-13m/README.md) | 目录内 README | 旧格 W4 迁入 → WP2：stock dense 可迁移性（Δ −0.0055 / −0.0026，差距 16× ⇒ 只补本形态缺陷） |
| [`2621-per-depth-readout-13m`](2621-per-depth-readout-13m/README.md) | 目录内 README | 旧格 W4 迁入 → WP3：逐深度读出被等参数「共享宽变换」对照否证（Δ −0.0759 vs −0.1313） |

> 旧格 `W4-readout-geometry` 按 `.scratch/56-restructure/MAPPING.md` §1 迁入本项：其 `experiments/**`
> 整目录迁入本目录（一格一份家，不复制）。旧格目录与其 README 仍保留在
> `axes/W-width/W4-readout-geometry/**`，待 integrator 统一退役。
> 结构性切分（只读前 r 维 → 能量迁出读出子空间）的负结果物理证据在本轴
> `3.1.1/experiments/2525-tax-attribution`（MAPPING 一格一份家），本项只作**指针引用**、不复制。
