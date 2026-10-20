# 2620 · 对照臂与方差界

本目录记录本实验的**对照臂定义**（数值全部落在 `../results.json::controls` 与
`../results.json::points[*].recall_copy_source_corrupt`；此处只写口径与判据）。

## 1. chance（均匀猜测）

- `uniform_answer_set = 1 / sym_vocab = 1/64 = 0.015625`（答案集均匀）。
  另报模型原始词表的均匀基线 `1/vocab_size`（需要时）。
- 判据 C4 的**地板阈值** = `chance + 2·σ_wilson`（`σ` = chance 处的 Wilson 区间半宽）。

## 2. 局部先验地板（`local_prior_accuracy(k=8)`）

- 「只看最近 8 个 token 的最强预测器」在 copy / induction / var_recall 上的准确率。
- 三条腿都通过 `check_local_prior_resistance`（局部先验 < 3×chance）⇒ 答案不可由局部
  统计猜出（SCHEMA §19：长程结论的前提）。

## 3. 毁源对照（判据 C5，**逐臂**）

- 把 copy 主腿的**源片段整段**换成随机符号（其余逐位不变）；**源区间 = 该档自己的
  `[src_pos, src_pos + 32)`**，不是恒为 `[0, 32)`。
  > 教训（2619 `diagnose_copy_confound.py`）：若只毁 `[0,32)`，则 `src_pos=96/191`
  两档的源根本没被碰过，那个「毁源对照」会虚高（实测 fp16 虚高到 0.0990）。
- 判据：毁源后准确率必须落回地板（`≤ chance + 2σ`）。若某点仍显著高于地板，说明读数
  不是（只）来自被搬运的源信息，必须标注。

## 4. 方差与可控性

- bpc 方差带 = `0.014`（`harness/methodology.md`）：`|damage_bpc| < 0.014` 标「⚠不可排序」。
- 纯推理、无随机采样；同 seed 重跑逐位相同。
- **跨 run 可控性**：8 个参考臂与 `2619` 同任务同 seed 逐位对照
  （`results.json::reproducibility_vs_2619`，阈值 1e-5）。
