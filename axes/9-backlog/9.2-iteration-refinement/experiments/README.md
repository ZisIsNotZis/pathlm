# 9.2 迭代精化的其余（原 F 轴剩余）· 实验目录

本目录收**原 F 轴**（F2 回传通道 + F4 迭代 / 重复的训练信号）的实验。每次判决占一个子目录（自包含）：
`README.md`(契约) / 代码 / `results.json` / `analysis.md`(判读) / `MODEL.md`(权重哈希，权重本身不入库)。

| 实验 | 来源 | 说明 |
|---|---|---|
| [`2511-retry-matrix-withdrawal`](2511-retry-matrix-withdrawal/README.md) | 旧格 F2 | retry 矩阵撤回；只存在于上一时代引擎（legacy transport / retry / reentry），其通道税数字随 D1/D3 修订作废 |
| [`2619-reconstruction-loss-13m`](2619-reconstruction-loss-13m/README.md) | 旧格 F4 | 重建 / 充分性损失（无显式载体）；主判据腿是分布外单查询腿（距离 241–255，`harness/SCHEMA.md` §17.18 已永久降级），训练用多查询 / 判决用单查询 ⇒ 结论不可解释 |

> 旧格 `F2-return-channel`、`F4-training-signal` 按 `.scratch/56-restructure/MAPPING.md` §1 迁入本项
> （Backlog 研究项）：其 `experiments/**` 的两个实验目录整目录 `git mv` 到本目录（一格一份家，不复制）。
> `.scratch/**/evidence` 原地不动，只作锚点。旧格目录及其 README 仍保留在 `archive/wiki-legacy/axes/F-iteration/**`，
> 旧树退役已由本轮 integrator 完成（旧目录归档只读保留）。
