# 2683 · 同可见键数下「块条带」对「均匀条带」（`1.2.2` 的第三个固定非均匀模式）· 13M · dense

> 本文件是**预登记契约**：判据线在下节写死，跑之前不得修改；读数落 `results.json`，
> 人话判读落 `analysis.md`。归属 `axes/1-time/1.2.2-block-sparse` 的 `work_packages[WP1]`。
> 判据带的来源是 `harness/methodology.md` §2（0.014 bpc 带 **仅限配对 Δ、≥5 窗口**）。

## 1 要回答的判决问题

`1.2.2` 的机制问题是「**预先固定、非均匀**的可见模式能不能在每 token 只看很少键的前提下覆盖
全序列」。这条机制在本格已有两个实例：均匀条带（`2677` 的 `win24a0`）与「局部窗口 + 最前 8 个
恒可见全局槽」（`2662` 的 `win16a8`，`2677` 判它**更差** 0.023）。两者之外，BigBird 式**块稀疏**
（local window + **按块取可见块**）在本格从未实装、无读数（`project.yaml::gaps` 第一条）。

本实验实现它的**确定性成员**——**块条带**：每个查询看最近 16 个位置，外加**一块** `BLOCK=8` 个键的
**块对齐**远块，那块落在自己所在块往前 `LAG=4` 个块的位置（距离 `[25 + r, 32 + r]`，`r = i mod 8`）。
它满足固定 / 非均匀 / **块粒度**三条（同一模式逐块平移，而不是逐行平移），且**逐行可见键数与条带臂
完全相等**（都 = `min(24, i+1)`，`n_visible_pairs` 同为 5868）——这是本实验能回答
「非均匀性值不值」而不是「少看键掉多少」的前提。

一句话可证伪的形式：*在 13M / T=256 / 12k 步 / matched-step 快照下，**同样花 24 个可见键**，
块条带的 clean bpc 相对同键数的均匀条带**正越 +0.014 配对带** ⇒ 块粒度这个非均匀性买到了质量；
**落带内** ⇒ 无质量增量（同字节 ⇒ 不值得采纳）；**负越 −0.014** ⇒ 块条带更差。*

## 2 单因素声明

**唯一变量 = 24 个可见键怎么分配**（均匀的最近 24 vs 最近 16 + 滞后 4 块的一块对齐 8 键）。
所有臂都是同一 stock dense decoder-only 骨架（`train_depth_ar.py --dense --topology sequential`，
无 carry 门、无退出、仅最终深度 CE、tied 读出），同一配方逐字段一致：
d=256 / L=12 / h=8 / T=256 / mlp_mult=6 / dropout=0 / norm_cap=1 / bs=32 / lr=1e-3 / 12k 步 /
98.3M token（enwik8 前 90M 训练、末 10M 评测、vocab 206）。机器可读副本见 `config.json`。

| 臂 | 来源 | 可见模式 | 每位置可见键数 `k_eff` | 每位置 KV 字节（fp16） | 参数量 |
|---|---|---|---|---|---|
| `full` | 复用 `2610-frozen-trunk-jumphead-13m/trunk.pt` | 全因果（上界参照） | 256 | 3,145,728 | 13,536,768 |
| `win24a0_s0` / `_s1` | 复用 `2677-local-global-vs-band-13m/train/` | **均匀条带**：最近 24（**主判对照**） | 24 | 294,912 | 13,536,768 |
| `block8l4_s0` / `_s1` | **本实验训练** | **块条带**：最近 16 + 滞后 4 块的一块对齐 8 键（**主判实验臂**） | 24 | 294,912 | 13,536,768 |
| `win16a8_s0` / `_s1` | 复用 `2662-hard-window-anchor-13m/train/` | 局部+全局：最近 16 + 最前 8 恒可见（同字节轴上的第三个模式，参照） | 24 | 294,912 | 13,536,768 |

- **可见键数逐行相等（硬约束，落盘核）**：条带臂走 `pathlm.depth_ar.window_anchor_keep_mask(T, 24, 0)`；
  块条带臂走本目录 `block_sparse.block_stripe_keep_mask(T)`（`local=16, block=8, lag=4`）。
  两臂逐行计数恒等于 `min(24, i+1)`（块条带的构造证明见 `block_sparse.py` 的模块文档：
  稳态两段不相交 = 16 + 8；头部远块被 clamp 到 0 号块、与本地带相接 ⇒ 恰为 `[0, i]`），
  本实验逐行核这个等式并落盘（`results.json::visible_keys_per_row`），不靠文字声明。
- **参数量相等自动满足**：掩码不引入参数（`params_added = 0`）、不改块类（都是 stock `Block`）。
- **lag 为什么取 4**：`LAG=4` 是与本地带不相交的**最小**滞后 ⇒ 远块尽量近、信息密度最高，
  是对「非均匀性值不值」这个主张**最有利**的设定。若在这个最有利的设定下都不占优，结论不能
  被解释成「远块放得太远」。
- **本实验不改共享文件**：`pathlm/**`、`train_depth_ar.py`、`tools/**`、`tests/**` 全部只读调用；
  唯一干预点 = `block_sparse.BlockStripeModel` 覆写 `DepthARModel._window_mask`（训练前向里构造
  可见掩码的唯一一处），新代码只在本目录（`block_sparse.py` / `train_block.py` / `train_arms.sh` /
  `run.py`）。

## 3 判据线（预登记，写死后不得改）

Δ 方向统一（`harness/SCHEMA.md` §17.3）：**Δ = bpc_对照 − bpc_实验**，本实验的「实验」= 块条带
（`block8l4`），「对照」= 同可见键数的均匀条带（`win24a0`）⇒ **Δ = bpc_win24a0 − bpc_block8l4，
正 = 块条带更好**。

| # | 判据 | 判定线 | 该涨 / 该跌 / 必须不变 |
|---|---|---|---|
| ① **主判（质量）**：同 `k_eff = 24` 下「块条带 vs 均匀条带」的 clean bpc 配对差 | 同一批评测窗口上的配对 Δ，对 **0.014 带**；**两个 seed 各自算一次**，两 seed 同向且都越带才算「可排序」；三态：`Δ ≥ +0.014` = 块条带占优、`|Δ| < 0.014` = 无质量增量、`Δ ≤ −0.014` = 块条带更差 | 该涨（非均匀该更好）或如实报无增量 / 更差 |
| ② **质量参照线**：每个 24 键臂相对 `full` 的配对 Δ | 同 0.014 带，只作「少看 232 个键的代价」参照，**不与 ① 混用** | 该跌 |
| ③ **成本（该跌）**：每位置 KV 缓存字节 | 解析量 `12288 B × k_eff`（fp16、MHA、d=256/L=12）：三个 24 键模式同为 294,912 B（相对 `full` −90.6%），且 ① 的两臂**同字节** ⇒ 若 ① 有差，差不能记在字节账上。**本实装是稠密 bool 掩码，省幅是结构上界，未兑现为实测**，如实标注 | 该跌 |
| ④ **必须不变**：因果性 | 两张掩码都 `⊆ causal` 逐位为真（`mask_offending_future_pairs == 0`） | 必须不变 |
| ⑤ **必须不变**：可见键数逐行相等 | 块条带与条带的逐行可见键数逐行相等，`visible_keys_per_row.mismatch_rows == 0`、两臂 `n_visible_pairs` 相等（5868） | 必须不变 |
| ⑥ **必须不变**：训练/评测掩码同源（三道） | (a) 模型自己的前向（块条带臂走 `BlockStripeModel._window_mask`）与 `masked_logits` 的显式掩码前向 `max_abs_logit_diff == 0`；(b) 训练期落盘的掩码 `sha256` 指纹 = 评测期构造的指纹；(c) 训练期掩码调用次数 `> 0`（否则是「开关开着、掩码没跑」的死干预，该臂作废） | 必须不变 |
| ⑦ **必须不变**：参数量与配方 | 七臂参数逐位相等（13,536,768）；配方逐字段一致（`results.json::arms` / `config.json::recipe`） | 必须不变 |
| ⑧ **复用臂等值门**：复用 checkpoint 的读数可复现 | 本实验重新评测 `full` / `win24a0_s0` / `win24a0_s1` / `win16a8_s0` / `win16a8_s1` 必须逐位复现 `2677` 落盘的 bpc（1.502689 / 1.599270 / 1.598631 / 1.622527 / 1.621799） | 必须不变 |
| ⑨ **申报不进判据**：墙钟、峰值显存、步时 | 逐臂申报；`harness/methodology.md` §3 禁把墙钟当判据 | — |

**长程召回腿（本实验不重跑，如实登记）**：本判据的率类腿（远块是否让远端内容可召回）不跨量纲借
0.014 带，需要按针配对的 CI；它已落盘在
`axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m/results.json`（指针，不转述）。
本实验的判词因此**只对质量腿（bpc）成立**，在 `analysis.md` / `conclusion` 里写清这条边界。

## 4 指标口径（操作性定义）

- **clean bpc**：`masked_logits(model, x, keep_mask)` → 因果移位 CE 的 token 平均（bits/char，
  越低越好）；`n_batches = 96`、`batch_size = 16`、`window_seed = 2`（与 `2677` / `2662` 同批数、
  同窗口序 ⇒ 臂间差是**配对差**）。**每臂在自身可见性下评测**：块条带用
  `block_stripe_keep_mask(256)`，条带用 `window_anchor_keep_mask(256, 24, 0)`，`full` 用 `None`。
- **配对统计**：逐窗口 `bits_per_window` 的配对 Δ 的均值 / 标准差 / 标准误 / t 值 / 同向窗口数。
- **可排序门**：bpc 类只用 0.014 带（配对、≥5 窗口）；**不跨量纲**借用给任何率类指标。
- **字节口径（同一条每位置字节轴）**：每位置可读 KV 字节 = `12288 B × k_eff`（fp16、MHA、
  d=256/L=12、12 层）。三个 24 键模式都 = 294,912 B/位置，`full` = 3,145,728 B/位置。
  主判两臂同字节 ⇒ 主判的差不能记在字节账上（判据③）。

## 5 预算与成本

| 项 | 值 |
|---|---|
| GPU 预算 | 2 支新臂（`block8l4` seed 0/1）× ≈9 min + 冒烟 ≈1 min + 评测 <4 min ≈ **23 min**（本实验自报上限 30 min）；复用臂只做评测 |
| 单臂命令超时 | ≤20 min（`train_arms.sh` 的 `timeout`） |
| 显存 | 逐臂实测 `peak_gpu_mem_mb`；礼貌线 5GB net（`harness/SCHEMA.md` §20.2） |
| 可中断性 | 每 200 步落 `train_log.jsonl`；断点即重取锁重跑该臂（驱动先清日志） |

## 6 失败决策树

| 症状 | 处置 |
|---|---|
| `block8l4` 训练 NaN / 发散 | 先修数值腿再判机制；该臂不进判据（重试上限见 `docs/plan.md` 的本格预算） |
| 掩码不满足 `keep ⊆ causal` | 实现/构造缺陷，中止判决 |
| 逐行可见键数不等（`mismatch_rows > 0`） | 对齐失败，判决作废（报「不可判」而不是硬判） |
| 训练期掩码调用次数 = 0 / 指纹不等 | 死干预或非同源，该臂作废并重训 |
| 复用臂读数不复现 `2677` | 评测口径不一致 ⇒ 先对账，不判 |
| 主判 `|Δ| < 0.014`（两 seed 都不越带） | 如实报「块粒度的非均匀性在本尺度无质量增量」，并把「同字节下的唯一余量是率类腿」写成边界 |
| 两 seed 方向相反 | 报「本 n 下不可判」，不取平均当结论 |

## 7 产出

本目录下 `README.md`（本文件）/ `config.json` / `block_sparse.py` / `train_block.py` /
`train_arms.sh` / `run.py` / `results.json` / `analysis.md` /
`train/block8l4_s{0,1}/{model.pt,train_log.jsonl,gate_log.jsonl,results.json,stdout.log}`。
`results.json` 里除质量读数外还有：`visible_keys_per_row`（判据④⑤）、`train_eval_mask_same`
（判据⑥）、`byte_axis_per_position`（判据③）、`reuse_gate`（判据⑧）、`measured_step` /
`train_declaration`（判据⑨，不进判据）。
训练期掩码自身的活性量落在每臂 `train/<arm>/results.json::block_sparse_mask`（`n_visible_pairs` /
`per_row_counts_equal_band24` / `fingerprint_sha256` / `mask_calls_during_training`）——引擎自报的
`window_mask` 字段描述的是 `(24, 0)` 条带计数，**不是**本臂掩码的证据。

## 8 复现 / 重跑

```bash
# 两臂训练（block8l4 的 seed 0/1；逐臂取/放 SCHEMA §20 GPU 锁，臂间 61 s；单臂 timeout 20 min）
bash train_arms.sh
# 评测 + 等值门 + 对齐核查 + 掩码同源门 + 申报量（写 results.json，<4 min GPU）
python3 run.py
```

- 冒烟（300 步、量步时、不进判据）：`MODE=smoke bash train_arms.sh`。
- 掩码自证（0 GPU）：`PYTHONPATH=. python3 block_sparse.py`。
- 墙钟与峰值显存见 `results.json::train_declaration`（复用臂的读数取自各自落盘文件，
  路径在 `source` 字段里）。
