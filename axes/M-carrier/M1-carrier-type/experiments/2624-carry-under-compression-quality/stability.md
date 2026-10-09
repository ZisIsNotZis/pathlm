# 2624 · 稳定性记录：判决腿（距离 241–255）在 6k 下的 run-to-run 方差

> 契约：`README.md`（§4b/§4c）；判读：`analysis.md`（§7D）；原始结果：`results.json`。
> 本文件是**方法学级发现**的落盘（主 Agent 裁决要求显式记录）。

## 0. 一句话

**同配置**（GQA2 + carry delta + window16，6k 步，seed=0，同引擎）的 `carry_gqa`，在我这轮
的主腿（单查询关联回忆、距离 **241–255**、native window16）上 = **0.93%**，而 Y1（ticket 40）
的 `gqa_carry_window` = **15.33%**——**差 15×**。四类解释全部排除后判定：
**该判决腿在 6k 步下 run-to-run 方差极大，不得作为此类比较的主判据。**

## 1. 观测

| 对象 | 配置 | 主腿读数（241–255，window16） | 全因果口径 | 训练分布内腿（107–240） | 来源 |
|---|---|---|---|---|---|
| 我的 `carry_gqa` | GQA2+carry+win16，6k，seed0 | **0.928%**（19/2048） | 9.91% | 35.72% | 本实验 `results.json` |
| Y1 的 `gqa_carry_window` | **同上**（逐字段相同） | **15.332%**（314/2048） | 13.67% | 34.77% | ticket 40 / 2623 `results.json` |

两者参数量同为 12,616,808、块类同为 `ComposableBlock`、训练日志的 `window_mask`
逐字段相同（`train_window=16/anchors=0/alive=True/visible_frac=0.120866`）、
`read_share` 同量级（0.754 vs 0.811）、`synth_answer_ce` 同量级（1.579 vs 1.594）。

## 2. 四类排除（执行时逐条查过）

1. **仪器 / 接线无罪**：用**本实验**的评测 harness（`run.py` + `quant_carry.py`）评
   **Y1 的** `gqa_carry_window/model.pt`（sha256 `48c4e39e…`）→ 主腿 **14.26%**（n=512），
   且 **released 前向** 与我的 **fp16 档** 读数**逐位相同**；我的 `fp32` 档与 released
   前向**逐位相等**（`CarryChannelBlock` / `ComposableBlock` 各一条自检，
   `results.json::instrument_selfcheck.fp32_bitwise` 全 True）。
   ⇒ 差异**不是**我的评测实现造成的。
2. **配置无罪**：同引擎（`pathlm/depth_ar.py` 自 ENG1 `3c2b6a9` 与 X3 `b96ac47` 之后未再
   改动；Y1 训练时与现在同一份引擎）、同 steps(6000)/bs(32)/lr(1e-3)/seed(0)、同
   `--train-window 16 --train-anchors 0`；两边的 `kv_heads.qkv_out_width=384`、
   `k_shape=[B,2,T,32]` 一致。
3. **能力无罪**：我的两条臂在**全因果**口径下 = 15.5%（MHA）/ 9.9%（GQA），与 Y1 的
   ~15%（`gqa_carry` 15.67% / `gqa_carry_window` 13.67%）**一致**⇒ 模型确实能在这个距离上
   找到 pair（只是经注意力，不必然经搬运）。
4. **非「没学会搬运」**：`read_share`、`state_fro`、答案 CE 均与 Y1 同量级 ⇒ 差异**只出现
   在施加 window16 掩码之后**——即「记忆能否跨窗把信息交给查询行」这一步的 run-to-run 方差。

## 3. 判定与后果

- **判定**：差异来自**训练随机性**（CUDA 非确定性跨 6k 步复合；训练数据流由独立
  `torch.Generator().manual_seed(0)` 决定、本应可复现，但实测两轮 checkpoint **sha256 不同**
  ——`carry_gqa` `02c390d8…` vs Y1 `gqa_carry_window` `48c4e39e…`）在**近分布外、未收敛**
  的判决腿上的放大。**不是 bug**。
- **后果**：
  1. **「距离 241–255」的判决腿不得作为此类（carry / GQA / 压缩）比较的主判据**——
     它越出训练距离分布（>240）且 6k 远未收敛，跨会话不可复现。
  2. 本实验的主结论（int8 vs fp16，**同一批权重、确定性评测**）**不受影响**（0.0pp，
     命中数逐位相同）——它是同一 checkpoint 内的比较，不含训练随机性。
  3. 本实验的次要结论（GQA vs MHA）**只能标 n=1**，且跨腿符号相反 ⇒ 不可排序。
  4. Y1（2623）在判决腿上的绝对水平（~15%）应理解为**一次抽样**，其「carry 把距离外推」
     的定性方向（无记忆 0%、有记忆 15%）跨会话可复现性**未被本线验证**。

## 4. 下一步最小实验（已排，≈26 min GPU；不在本线现在补，避免与 Z2 抢锁）

- **两臂 seed=1 复现**：`carry_mha` / `carry_gqa` 各以 seed=1 重训 6k，评测同三套口径，
  报主腿（241–255）与训练分布内腿（107–240）的**跨 seed 方差**（`mean ± sd`，n=2）。
- **预登记判据**：若两 seed 的主腿读数跨度 ≥ 5pp（或与 Y1 的 15% 仍差 ≥5×）⇒ **确认方差为
  主因，判决腿（241–255）永久降级为辅助腿**，此类比较一律以**训练分布内距离档**为主判据
  （主 Agent 已把此定性为契约口径修正）；若两 seed 都落在 ~1% 且与 Y1 稳健不等 ⇒ 需另查
  会话级因素（如卡上并发负载对 kernel 选择的影响）。
