# 2678 · 同可见键数下「空洞采样」对「连续窗口」（`1.2.1` 的 WP1 最小判决）· 13M · dense

> 本文件是**预登记契约**：判据线在本节写死，跑之前不得修改；读数落 `results.json`，
> 人话判读落 `analysis.md`。归属 `axes/1-time/1.2.1-dilated` 的 `work_packages[WP1]`。
> 判据带的来源是 `docs/methodology.md` §2（0.014 bpc 带 **仅限配对 Δ、≥5 窗口**）；
> **判据常数**（`band_bpc` / 距离档 / 掩码参数 / 配对规则）住在 `config.json`，与
> `run.py` 一起被 `tools/prereg_bind.py --contract README.md config.json run.py` 绑住。

## 1 要回答的判决问题

`1.2.1` 的命题：**每 token 只看固定 k 个键时，把这 k 个位置摊开到更远（空洞）比只看最近 k 个
更能保住远程信息吗**。这一格本仓零经验（旧 `A5-dilated-window` 格已归档、从未实装），
所以最小判决就是：在**每行可见键数逐行相等**的两臂之间，唯一差别 = 这 k 个位置**怎么挑**
（最近 k 个 vs 近端窗口 + 摊满整个前缀的空洞采样）。

一句话可证伪的形式：*在 13M / T=256 / 12k 步 / matched-step 快照下，`dil32` 相对 `win32`
（同 32 键）的长程召回不越配对 CI 的噪声带、且 clean bpc 的配对差落 0.014 带内 ⇒ 空洞采样
在本尺度无净增量、不成立。*

## 2 两臂的设计（唯一变量 = 可见集的挑法）

`k = 32`，`T = 256`，两臂每行可见键数**逐行相等**且都等于 `min(32, i+1)`：

| 臂 | 可见集（第 i 行） | 每行计数 | 最远可达距离 |
|---|---|---|---|
| `win32` | 最近的 32 个位置 `j ∈ [i−31, i]` | `min(32, i+1)` | 31（恒定） |
| `dil32` | 近端窗口 16 个 `j ∈ [i−15, i]` **+** 16 个空洞点均匀铺满前缀 `j ∈ [0, i−16]` | `min(32, i+1)` | `i`（`i ≥ 32` 时铺满整个前缀，最远 = 最远的可用位置） |

* `dil32` 的第 t 个空洞点 = `round(t·(P−1)/15)`，`P = i − 15` 是可用前缀长度、`t = 0..15`
  ⇒ 逐行 dilation = `(P−1)/15`（**行内均匀铺满**，不是固定 stride：固定 stride 在 `i < 136`
  时取不满 16 个点，会让两臂可见键数不等、破同预算对照）；
* 两臂在 `i ≤ 31` 的行**完全重合**（都退化成全因果，`i+1 ≤ 32` 个键）⇒ 差异只发生在
  `i ∈ [32, 255]`（256 行里的 224 行）；
* 掩码是**静态位置掩码**（与内容无关、与步数无关），构造在实验目录的 `run.py` 里（`DilatedBlock`
  把掩码 buffer 挂在自己的 `Block` 上并走 `Block.forward(attn_mask=...)` 的既有掩码路径）；
  **不改** `pathlm/**`、`train_depth_ar.py`、`tools/**`、`tests/**`。

## 3 配方（与参考臂逐字段一致）

| 字段 | 值 |
|---|---|
| 形态 | stock dense decoder-only（`dense=True`、`topology=sequential`、无退出/无 carry/无填充） |
| d / layers / heads / T / mlp_mult / dropout / norm_cap | 256 / 8 / 8 / 256 / 4 / 0.0 / 1.0 |
| steps / batch_size / lr / seed | 12000 / 32 / 1e-3 / 0 与 1 |
| 数据 | `data/enwik8_full.npz`（前 90M 训练 / 末 10M 评测，字符级 vocab 206） |
| 训练入口 | `train_depth_ar.train()`（**同一个训练函数**，不复制训练循环；模型对象由本实验构造） |
| 评测 | 20 批 × bs16、`window_seed=2` 的**配对** clean bpc（与 `2676` / `2614` 同一批位置、同一 batch 顺序） |

**复用臂（0 GPU）**：`win32` / `win32_s1` 直接复用 `2676-content-vs-window-13m/runs/win32{,_s1}`
（同配方、只换 `--train-window 32`），`full` 复用 `2614-topk-sparse-attn-13m/runs/full-12k-s0`。
等值门：本实验重算的 `win32` / `win32_s1` 逐批 bpc 必须与
`axes/1-time/1.2.3-content-topk/experiments/2676-content-vs-window-13m/results.json::per_batch_bpc`
**逐位相同**（口径未漂移的证据），不等即中止判决。

## 4 判据线（预登记，跑后不改）

Δ 方向统一（`SCHEMA.md` §17.3）：**Δ = 读数_对照 − 读数_实验**，"实验" = 空洞采样 `dil32`，
"对照" = 同 k 的连续窗口 `win32` ⇒ **`Δ > 0` = 空洞采样占优**。

| # | 判据 | 判定线 | 该涨 / 该跌 / 必须不变 |
|---|---|---|---|
| ① **主判（质量）** | `Δ = bpc_win32 − bpc_dil32`（20 批配对） | **0.014 bpc 带**（`config.json::band_bpc`，仅配对 Δ）：`≥ +0.014` = 空洞占优、`|Δ| < 0.014` = 无质量增量、`≤ −0.014` = 空洞更差 | 该涨或如实报无增量 |
| ② **主判（长程召回）** | 变距离回忆（`tools/evalkit/longrange_tasks.var_recall_batch`）在**远档** `d ∈ {64,128,192,240}` 上的 top-1 准确率差 `Δ = acc_win − acc_dil`（对照 − 实验，同本节开头的 Δ 方向） | **率类指标**：按批聚类的配对 bootstrap 95% CI（`config.json::recall_*`）；CI 不含 0 才算可分。**不跨量纲借用 0.014** | 该涨（空洞该更能取回远处信息） |
| ③ **近档对照** | 同任务 `d ∈ {16,32}` 的准确率差（`d = 16` 两臂都够得着；`d = 32` **超出 `win32` 的可达上界 31**（`results.json::alignment_checks.reach.win32_max`）⇒ 该档只作噪声参考，不构成「两臂都够得着却跌了」的对照） | 同 ② 的 CI 口径；对照只对 `d = 16` 成立 | 必须不跌（若 `d = 16` 就跌，说明差异不是「够得着」造成的） |
| ④ **必须不变**：可见键数逐行相等 | `kept_per_row` 逐行相等且 `= min(32, i+1)`；两臂 `mismatch_rows == 0`；`dil ⊆ causal`、`mask_offending_future_pairs == 0` | 机检，落盘 | 必须不变 |
| ⑤ **必须不变**：参数量与配方 | 四臂参数量逐位相等（`results.json::params_all_equal`）；dil 臂与 `win32` 臂**构造路径同源**（同一个训练函数、同 seed 构造主干，转换后各参数逐位不变：`runs/dil32-s*/results.json::conversion_check`）。**未**把 dil 的初始权重与 `win32` 的 checkpoint 逐位比过（两个 run 各自训练，checkpoint 必然不同；比的是同 seed 的构造初值，落盘列只有「转换前后逐位相同」） | 机检，落盘 | 必须不变 |
| ⑥ **参照线**：相对 `full` 的配对 Δ | 同 0.014 带；**只作「少看键的代价」参照线，不与 ① 混用** | 该跌（少看必掉质量） | — |
| ⑦ **申报不进判据** | 墙钟、峰值显存、步时、掩码最远距离列 | 逐臂申报 | — |

**n 与带的关系（预登记）**：`dil32` 跑 **seed 0 + seed 1** 两条，`win32` 的两 seed 复用 `2676`。
判词规则只有这一条：**两 seed 的 ① 配对 Δ 同向且各自越 0.014 带** ⇒ 判词按两 seed 出（n=2）；
**两 seed 不同向或任一落带内** ⇒ 判「本 n 下不可排序」，`status` 停「进行中」，不得把不一致
解释成噪声、不得对两 seed 取平均掩盖。② 的率类判据每 seed 各自出 CI，跨 seed 只报一致性。
**跨 seed 同臂之差 = 独立 run 之差，只写「参考线」，不得与 0.014 配对带混用**（`docs/methodology.md` §2）。

**否证线（预登记）**：② 的远档 CI 含 0（含全部远档）**且** ① 两 seed 都落带内 ⇒ 判
**「空洞采样在本尺度无净增量、不成立」**；若 ① 越带为负（空洞更差）⇒ 判「按位置最近仍占优」；
若 ② 的近档对照（③）就可分 ⇒ 判「差异不是感受野造成的」，① 不作机制判据。

### 4.1 追补读数（判词落定之后加的，两条都不改上面的判定线）

两条都是**同一条主判的边界读数**：它们回答「召回这条仪器有没有分辨力」与「训练信号能不能迫使
模型用远处信息」，不新增判定线、不换判定量。两条的常数都住在 `config.json::recall` /
`config.json::synth_recall`，应用代码在 `run.py`，与 ①–⑦ 一起被绑定。

**A · 仪器分辨力（`config.json::recall.n_batches`）**：在现有四臂 checkpoint 上把召回电池从
**8 批 × 32 样本/档**加密到 **64 批 × 32 样本/档**（`batch_size` 与 `task_seed` 不变，只加样本数）。
读法只有两条：

* 若 seed 0 的 `d64` / `d128` 两档在 8 倍样本下**仍在**（CI 不含 0），且某档的
  `best_over_chance` 越过 `config.json::recall.power_ratio_threshold`（5×）⇒ **仪器有分辨力**，
  ② 可以按「可分但不支持空洞」读，判词须回到 ① 与 ③ 的机制读法；
* 若 8 倍样本下可分档消失、四臂仍贴 chance（`best_over_chance` ≤ 2×）⇒ **仪器在本尺度仍无分辨力**，
  ② 给不出正向证据这一条**不变**，真实文本那一半的判词维持「无质量增量」。

**B · 合成关联回忆对撞（`config.json::synth_recall`）**：`win32` 与 `dil32` × **seed 0 与 seed 1**
四支，在**必须用远处信息**的任务（多查询关联回忆：key 全在序列最前 16 个位置，查询行只看最近 32
个键根本看不到 key）上同构造、同 seed 对、同 12000 步训练，再在**同一批**保留任务上按批配对比较
答案准确率（chance = 1/64）。判定规则（`config.json::synth_recall.rule`）：

* **两个 seed 各自**满足 `dil32` 准确率 ≥ 5× chance **且** `Δ = acc_win − acc_dil` 的配对 95% CI
  不含 0 且为负 ⇒ 训练信号确实迫使使用远处信息、且空洞路径能把它搬回来，按 n=2 入账；
* 只有一个 seed 满足 ⇒ 临时读数（n=1 单种子，未见重复），不作 n=2 结论；
* 两臂都 < 5× chance ⇒ 该合成任务对两条掩码都没有可用路径 ⇒ 这条读数**不能**区分「机制无效」
  与「本任务不需要」。

（这一对臂的差别是「**能不能覆盖到**」——窗口臂在距离 107–240 的 key 上根本看不到 key，不是
「同样覆盖下谁更省」；所以它只能回答「掩码有没有交付远处信息」。）

**披露**：A 的样本数是在读到 8 批那一版之后加密的（同一估计量、同一判定线，只是样本变多），
B 是本轮新加的同任务对撞。两格的**契约都没有先于旧读数**（见 `docs/plan.md` 的批次三行与
`prereg_binding.json::readings_existing_at_bind`），所以这两条读数的定序只从本次绑定起算。

## 5 失败决策树

| 症状 | 处置 |
|---|---|
| `dil32` 训练 NaN / 发散（gnorm 尖峰、跳步） | 先修数值腿再判机制，该臂不进判据 |
| 掩码不满足 `⊆ causal`、或逐行计数 ≠ `min(32,i+1)` | 实现缺陷，中止判决（不硬判） |
| 等值门：`win32` 逐批 bpc 复现不了 `2676` | 评测口径漂移，中止判决 |
| 单臂 timeout（>20 min）或锁等待超时 | 如实登记为「未跑」，不推断；该臂不计入判据 |
| 两 seed 不同向 | 判「本 n 下不可排序」，`status` 停「进行中」 |

## 6 预算

| 项 | 值 |
|---|---|
| GPU 预算 | 2 支新臂（`dil32` seed 0/1）×≈7.5 min + 冒烟≈1.5 min + 评测（bpc + 召回电池）≈4 min ≈ **21 min**（上限 30 min） |
| 单臂 timeout | 20 min（`train_arms.sh`） |
| 显存 | 逐臂实测 `peak_gpu_mem_mb`；礼貌线 5GB net（`SCHEMA.md` §20.2） |
| 可中断性 | 每 200 步落 `train_log.jsonl`；断点即重取锁重跑该臂 |
| §4.1 的 B | `win32` / `dil32` 各 12000 步 ≈ 7 min/臂（同一条训练路径，逐臂取/放锁）；A 不训练，只是同一条召回电池的样本数 8 → 64 |

## 7 产出

本目录：`README.md`（本文件）/ `config.json`（判据常数）/ `run.py`（训练 + 评测 + 判词）/
`train_arms.sh`（取锁驱动）/ `results.json` / `analysis.md` / `prereg_binding.json` /
`runs/dil32-s{0,1}/{model.pt,train_log.jsonl,gate_log.jsonl,results.json}` /
`runs/synth-{win32,dil32}-s{0,1}/{model.pt,train_log.jsonl,gate_log.jsonl,results.json}`（§4.1 的 B）。

## 8 复现 / 重跑

```bash
python3 tools/prereg_bind.py axes/1-time/1.2.1-dilated/experiments/2678-dilated-vs-window-13m \
  --contract README.md config.json run.py          # 训练之前绑定（缺绑定两处入口拒绝启动）
bash train_arms.sh                                  # 2 臂（逐臂取/放 GPU 锁，臂间 61 s）
python3 run.py --stage eval                         # 对齐核查 + 配对评测 + 召回电池 → results.json
for A in win32 dil32; do for S in 0 1; do
  python3 run.py --stage synth --arm $A --seed $S  # §4.1 B：合成关联回忆（两臂 × 两 seed）
done; done
python3 run.py --stage eval                         # 再跑一次：把两条 synth 读数汇入 results.json::synth_recall
```
