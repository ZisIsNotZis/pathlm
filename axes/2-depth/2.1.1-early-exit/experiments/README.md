# 2.1.1 early exit（最重要） · 实验目录

本格每次判决占一个子目录（自包含）：`README.md`(契约) / 代码 / `results.json` / `analysis.md`(判读) / `MODEL.md`(权重哈希，权重本身不入库)。

命名：`YYMM-<干预>-<规模>-<引擎>`。

| 实验 | 来源 | 说明 |
|---|---|---|
| [`2523-depth-ar-first`](2523-depth-ar-first/README.md) | 目录内 README | 旧格 C1 迁入：深度自适应 AR 首验 |
| [`2523d-residual-tax-decomposition`](2523d-residual-tax-decomposition/README.md) | 目录内 README | 旧格 C1 迁入：残税成分分解 |
| [`2621-depth-draft-e2e`](2621-depth-draft-e2e/README.md) | 目录内 README | 旧格 C1 迁入：深度草稿端到端（自投机，不是早退） |
| [`2523c-three-ablations`](2523c-three-ablations/README.md) | 目录内 README | 旧格 C5 迁入 → WP4 精化门证据 |
| [`2518-exit-retry-law`](2518-exit-retry-law/README.md) | 目录内 README | 旧格 C6 迁入 → WP3 深度监督配方证据 |
| [`2524-depth-ar-100m`](2524-depth-ar-100m/README.md) | 目录内 README | 旧格 C6 迁入 → WP3 证据（100M 形态税与出口资产） |

> 旧格 `C1-layers-per-token`（核心）、`C5-refinement-operator`（精化门）、`C6-depth-supervision`
> （深度监督配方）按 `.scratch/56-restructure/MAPPING.md` §1 并入本项：C1 的实验整目录迁入，C5/C6 作为
> 本项 `work_packages` 的手段，其判决实验也迁入本目录（一格一份家，不复制）。
> 对照指针（R7 参照线，不迁入）：`axes/R-references/R7-existing-early-exit/experiments/2623-standard-early-exit-13m`（CALM / LayerSkip / MoD 标准实现，本项必须打败的基线）。
> 旧格目录与其 README 仍保留在 `axes/C-vertical/**`，待 integrator 统一退役。
