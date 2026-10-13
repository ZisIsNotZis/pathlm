# 9.1 调度 / 选择器（原 G 轴）· 实验目录

本目录收**原 G 轴**（G1 路径选择器 + G3 观测信号）的实验。每次判决占一个子目录（自包含）：
`README.md`(契约) / 代码 / `results.json` / `analysis.md`(判读) / `MODEL.md`(权重哈希，权重本身不入库)。

| 实验 | 来源 | 说明 |
|---|---|---|
| [`2508-tts-probes`](2508-tts-probes/README.md) | 旧格 G1（原 F3 并入） | 失败路径明写「没有任何组合打赢 B0 → 项目卖点从质量改为可控性」——目的漂移的原始出处 |
| [`2514-allocator-selectivity`](2514-allocator-selectivity/README.md) | 旧格 G1（原 F3 并入） | 分配器选择性探针；收益集中在中低置信带（+0.04~+0.06）、高置信带 ≈0/负 |
| [`2515-slider-rung3`](2515-slider-rung3/README.md) | 旧格 G1 | Slider rung3；「严格占优点」Δbpc=0.0043 < 0.014 方差带、FLOPs 口径净亏 |
| [`2517-slider-at-scale`](2517-slider-at-scale/README.md) | 旧格 G1 | 100M 规模；门控重试质量杠杆消失、τ 轴平坦 ±0.002 ⇒ 核心假设在 100M 失效 |
| [`2522-gen-quality-signals`](2522-gen-quality-signals/README.md) | 旧格 G3 | 三信号（prob0 / js / distinct-2）正交监测栈；n=24 单权重单解码、人工标注，证据偏薄 |

> 旧格 `G1-selector`、`G3-signal` 按 `.scratch/56-restructure/MAPPING.md` §1 迁入本项（Backlog 研究项）：
> 其 `experiments/**` 的五个实验目录整目录 `git mv` 到本目录（一格一份家，不复制）。
> `.scratch/**/evidence` 原地不动，只作锚点。旧格目录及其 README 仍保留在 `axes/G-policy/**`，
> 待 integrator 统一退役。`G4-calibration` 无自有实验目录。
