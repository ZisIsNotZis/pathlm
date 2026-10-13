# controls — 对照臂与方差界（2620 · 容量轴在「收敛后」是否有效）

本实验的**每一个**格（3 臂 × 4 难度档 = 12 格）都同时跑下面几条对照。
原始读数在 `../results.json::grid[<arm>_K<K>]`，机检在 `../reconcile_report.json`。

## 1. 毁源对照臂（强制，判据 ③.1）

- **做法**：`tools/evalkit/longrange_tasks.py::corrupt_span(task, 0, 2K)` —— 把
  `[0, 2K)` 的 key/value 全部换成**同分布随机符号**，查询行 / 答案标签 / 距离**逐位不变**
  （`corrupt_span` 的 generator 固定 `seed=0`）。
- **为什么**：如果读数来自「搬过来的源信息」，毁掉源后必须掉回 chance；若仍高，
  说明读数来自位置/局部先验或仪器混杂 ⇒ **该格判读作废**（README §4③.1）。
- **操作化**：作废条件 = 毁源读数的 **Wilson CI 下界 > 4×chance**（= 0.0625）。
- **结果**：见 `../analysis.md` §5（12 格毁源读数与作废格数）。

## 2. chance（瞎猜基线）与局部先验

- **任务**：多查询关联回忆，词表 64 个符号，同一行内 key 互不相同
  ⇒ 答案集合大小 = 64 ⇒ **chance = 1/64 = 0.015625**（全档位相同）。
- 各格的 `local_prior`（用最近 8 个位置能抄到的符号做预测）逐格落盘在
  `../results.json::grid[<arm>_K<K>].local_prior` ⇒ 这是 SCHEMA §19「局部先验抗性」在
  评测侧的直接读数：若它 ≈ chance，说明窗口确实把可抄的信号掐断了。

## 3. 前提对照：查询键落在窗外

- 评测掩码 = 训练掩码 = `window_anchor_keep_mask(256, window=16, anchors=0)`。
- 每格落盘 `query_key_distance` 的 min / median / max；窗口只有 16 ⇒
  「不搬就真答不出」（读数见 `../analysis.md` §3）。

## 4. 训练侧重跑臂（判据 ③.2，本实验相对 U1 的新增对照）

- **做法**：`S16_rep` 与 `S16` **逐字段相同**（同 seed=0、同 24k 步、同任务、同窗口），
  只是 run 名不同；两者都在同一台机器上串行跑完。
- **为什么**：ticket 36 的 `2618`（plain 写臂双稳态：同配置两次 0.0007 / 0.0334）与
  `2619-reconstruction`（`B` vs `B_rep` 差 1.37pp、CI 不重叠）都实测到**多 pp 级的训练侧
  散度** ⇒ 单跑一次不足以支持 +3pp 的判据。本实验的冒烟期 600 步确定性探针
  （`../stability.md`）也已证明**训练非逐位确定**（step 200 loss 0.5104 vs 0.5106）。
- **预登记的解读规则**（README §4③.2，不是新判据）：
  若 `S16` 与 `S16_rep` 在主腿 `K=8` 上 CI **不重叠** ⇒ 记「训练侧散度 ≥ 判据线（3pp），
  本尺度不可分辨」，主判据的「成立」结论**降级为「不可复现 ⇒ 待定」**。
- **结果**：见 `../results.json::criteria.c3_2_training_divergence` 与
  `../analysis.md` §6 / `../stability.md` §3。

## 5. 外部横向参照：U1 的 10k 臂（**只读引用，不复现**）

- U1（`../2619-carry-difficulty-capacity/`）的 `D16`/`D128` 与本实验的 `S16`/`S128`
  **逐字段相同，唯一差别是步数**（10,000 vs 24,000）。
- 本实验**不重跑** 10k 臂（省 GPU），直接引用 U1 的 `results.json::grid` 作横向参照
  （同 seed、同配方、同评测口径），并把本实验 24k 跑在**前 10k 步**的 loss 曲线与 U1 的
  10k 跑逐点对照——那同时是一次**第三方重跑散度**读数（见 `../stability.md` §3）。

## 6. 对照的复用与不重复（SCHEMA §21）

- **不重训无通道基线**：ticket 35（`../2618-carry-channel-13m/`）已实测无通道臂在该任务上
  ≈ chance（0.0137 vs chance 0.015625）。本实验的对照是 **chance 线 + 毁源臂 + 局部先验 +
  重跑臂**（四者都对同一读数施压），不重复花 GPU。
- **方差界**：本仓方法学（`docs/methodology.md`）规定 Δ < 0.014 不可排序。
  本实验所有方向判断同时满足「Wilson 95% CI 不重叠」。
- **复现性**：主腿格（`S128/K=8`）前 8 个批同 seed 整批重跑，逐位一致
  （`../results.json::reproducibility::same_seed_rerun_max_abs_diff = 0.0`）。
