# 9 Backlog（不删，移出主线）

> 轴：**9 Backlog（不删，移出主线）**（目录 `axes/9-backlog/`）。用词、编号、目录以
> `docs/structure.md` §12 为准，每个研究项的字段契约见 `harness/research-schema.md`。
> Backlog 项保留**研究项形式**（`status: 未开始`），**证据一律保留、不删**（`docs/structure.md` §13.6）。

Backlog 收「不删、但已移出主线」的算力路径：它们成立与否尚不明确，或机制归属未定；
保留是为了不丢证据，不是为了继续投入。要移出 Backlog，必须先给出**可证伪的目的 + 可判定判据**
（见各 `project.yaml` 的 `purpose` / `objective_criteria` / `falsification`）。

## 本目录的子树

| 编号 | 名称 | 目录 | 粒度 |
|---|---|---|---|
| 9.1 | 调度 / 选择器（原 G 轴） | [`9.1-selector-scheduling`](9.1-selector-scheduling/) | Backlog |
| 9.2 | 迭代精化的其余（原 F 轴剩余） | [`9.2-iteration-refinement`](9.2-iteration-refinement/) | Backlog |

> §9 另点名的「循环 transformer（重复层）」见 [`2.1.3`](../2-depth/2.1.3-repeating/)、
> 「权重共享」见 [`2.2.5`](../2-depth/2.2.5-weight-sharing/)；「层间拓扑」的负结论已入
> [`2.2.3`](../2-depth/2.2.3-learned-layer-sum/) / [`2.2.4`](../2-depth/2.2.4-parallel-sublayer/)。

## 共同口径

- 对照 = 全量标准模型（归一为 100%）+ 同预算 / 同参数 / 同算力的诚实对手；结论表述为**前沿位置**，不是单点 Δ。
- 判据 = 相对最优固定路径（9.1）或同算力加深度（9.2）的**序列级**质量；口径 = 等效理论算力
  （FLOPs / layer-passes / tokens-per-forward），**墙钟不进判据**。
- 每项有且仅有一个 `milestone`，推进过程见各自 `project.yaml` 的 `work_packages`（每条含独立 `review`）。
- `status` 仅四值：未开始 / 进行中 / 已完成 / 被否证；Backlog 项当前一律 `未开始`。

## 迁移说明

按 `.scratch/56-restructure/MAPPING.md` §1 / §4：原 G 轴（`G1-selector` + `G3-signal`，`G4-calibration` 无自有实验）
合并为 Backlog 研究项 `9.1`；原 F 轴剩余（`F2-return-channel` + `F4-training-signal`）合并为 `9.2`。
各旧格 `experiments/**` 整目录 `git mv` 入对应新项的 `experiments/`（一格一份家，只搬不复制）；
`.scratch/**/evidence` 原地不动，只作锚点。旧轴目录 `archive/wiki-legacy/axes/G-policy/**`、`archive/wiki-legacy/axes/F-iteration/**`
及其 README 已随旧树退役归档（归档只读保留）。

## 怎么读

- 状态与全表：`python3 tools/summary.py`；校验：`python3 tools/summary.py --check`。
- 「做到多少」写在 `conclusion` / `gaps` 正文，不进状态枚举。
