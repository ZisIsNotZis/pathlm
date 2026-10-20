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
| ② **主判（长程召回）** | 变距离回忆（`tools/evalkit/longrange_tasks.var_recall_batch`）在**远档** `d ∈ {64,128,192,240}` 上的 top-1 准确率差 `Δ = acc_dil − acc_win` | **率类指标**：按批聚类的配对 bootstrap 95% CI（`config.json::recall_*`）；CI 不含 0 才算可分。**不跨量纲借用 0.014** | 该涨（空洞该更能取回远处信息） |
| ③ **近档对照** | 同任务 `d ∈ {16,32}`（两臂都够得着）的准确率差 | 同 ② 的 CI 口径；只作「两臂在可达距离上本该一样」的对照 | 必须不跌（若这里就跌，说明差异不是「够得着」造成的） |
| ④ **必须不变**：可见键数逐行相等 | `kept_per_row` 逐行相等且 `= min(32, i+1)`；两臂 `mismatch_rows == 0`；`dil ⊆ causal`、`mask_offending_future_pairs == 0` | 机检，落盘 | 必须不变 |
| ⑤ **必须不变**：参数量与配方 | 四臂参数量逐位相等；dil 臂的初始权重与 `win32` 臂同 seed 逐位相同（只换掩码） | 机检，落盘 | 必须不变 |
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

## 7 产出

本目录：`README.md`（本文件）/ `config.json`（判据常数）/ `run.py`（训练 + 评测 + 判词）/
`train_arms.sh`（取锁驱动）/ `results.json` / `analysis.md` / `prereg_binding.json` /
`runs/dil32-s{0,1}/{model.pt,train_log.jsonl,gate_log.jsonl,results.json}`。

## 8 复现 / 重跑

```bash
python3 tools/prereg_bind.py axes/1-time/1.2.1-dilated/experiments/2678-dilated-vs-window-13m \
  --contract README.md config.json run.py          # 训练之前绑定（缺绑定两处入口拒绝启动）
bash train_arms.sh                                  # 2 臂（逐臂取/放 GPU 锁，臂间 61 s）
python3 run.py --stage eval                         # 对齐核查 + 配对评测 + 召回电池 → results.json
```
