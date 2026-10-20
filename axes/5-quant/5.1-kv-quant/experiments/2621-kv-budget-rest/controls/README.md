# 2621 · 对照臂与方差界

本目录记录本实验的**对照臂定义**（数值全部落在 `../results.json::controls` 与
`../results.json::points[*].recall_copy_source_corrupt`；此处只写口径与判据）。

## 1. chance（均匀猜测）

- `uniform_answer_set = 1 / sym_vocab = 1/64 = 0.015625`（答案集均匀）。
- 判据 C4 的**地板阈值** = `chance + 2·σ_wilson`（`σ` = chance 处的 Wilson 区间半宽）。

## 2. 局部先验地板（`local_prior_accuracy(k=8)`）

- 「只看最近 8 个 token 的最强预测器」在 copy / induction / var_recall 上的准确率。
- 三条腿都通过 `check_local_prior_resistance` ⇒ 答案不可由局部统计猜出（SCHEMA §19：
  长程结论的前提）。实测：copy 局部先验 0.0146、induction 0.0156、var_recall 0.0195
  （均 ≈ chance ⇒ 合成的长程任务对局部先验有抗性）。

## 3. 毁源对照（判据 C5，**逐臂**）

- 把 copy 主腿的**源片段整段**换成随机符号（其余逐位不变）；**源区间 = 该档自己的
  `[src_pos, src_pos + 32)`**，不是恒为 `[0, 32)`。
  > 教训（2619 `diagnose_copy_confound.py`）：若只毁 `[0,32)`，则 `src_pos=96/191`
  > 两档的源根本没被碰过，那个「毁源对照」会虚高。
- 判据：毁源后准确率必须落回地板（`≤ chance + 2σ`）。本实验 27 臂的毁源读数全部落在地板
  （`reconcile_numbers.py` 第 3 组逐臂断言）。

## 4. 族基线与增量对照

- **族基线**：trunk 族 = `fp16`（1.483571）；head 族与 quant_head 族 = `mha`（1.482026）。
- **量化增量**（`quant_increment_ref`）：组合臂 − 同拓扑 fp16 臂，隔离「再叠一个量化因子」的代价：
  `gqa_int4`←`gqa`、`mqa_int4`←`mqa`、`win32_int8`←`win32`、`win64_int8`←`win64`、
  `win16_anchor16_int4`←`win16_anchor16`、`win32_anchor32_int4`←`win32_anchor32`。
- **窗口增量**（`window_increment_ref`）：`gqa_win64`←`gqa`、`gqa_win128`←`gqa`，
  隔离「在 GQA 上再叠一个窗口因子」的代价。
- 单因素锚（基线与量化/窗口单用）都在同表内重测，便于同轴对照。

## 5. 方差与可控性

- bpc 方差带 = `0.014`（`harness/methodology.md`）：`|damage_bpc| < 0.014` 标「⚠不可排序」。
- 纯推理、无随机采样；同 seed 重跑逐位相同。与 2619 的 14 个参考臂、2620 的 5 个组合臂
  重测**逐位一致**（`max_bpc_abs_diff = 0.0`）。
