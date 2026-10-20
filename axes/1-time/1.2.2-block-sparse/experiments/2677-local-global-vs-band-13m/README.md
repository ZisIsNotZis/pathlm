# 2677 · 同可见键数下「局部+全局」对「均匀条带」（1.2.2 的 WP1 最小判决）· 13M · depth-ar/dense

> 本文件是**预登记契约**：判据线在下节写死，跑之前不得修改；读数落 `results.json`，
> 人话判读落 `analysis.md`。归属 `axes/1-time/1.2.2-block-sparse` 的 `work_packages[WP1]`。
> 判据带的来源是 `docs/methodology.md` §2（0.014 bpc 带 **仅限配对 Δ、≥5 窗口**）。

## 1 要回答的判决问题

本格的机制问题是「**预先固定、非均匀的可见模式**能不能在每 token 只看很少键的前提下覆盖全序列」。
Longformer 式做法把有限的可见键预算切两半用：大部分给最近的连续位置（局部），少数给最前面几个恒可见的
**全局槽**（`window_anchor_keep_mask` 的 `anchors`），期望用全局槽换回对远端关键位点的可达性。

旧读数只能回答「少看键掉多少质量」：`win16a8`（24 键）对 `full`（256 键）差 0.119838 bpc
（`axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m/` 的落盘读数），
它把「少看键的代价」与「非均匀是否值」混在一起。本实验只拆后者。

一句话可证伪的形式：*在 13M / T=256 / 12k 步 / matched-step 快照下，**同样花 24 个可见键**，
「16 个最近位置 + 8 个全局槽」（`win16a8`）的 clean bpc 相对「24 个最近位置」（`win24a0`）
**不越 0.014 配对带** ⇒ 非均匀性在本尺度没有质量增量，把键预算切给全局槽买不到东西；
Δ ≥ +0.014 ⇒ 非均匀模式占优；Δ ≤ −0.014 ⇒ 全局槽是把预算花坏了。*

## 2 单因素声明

**唯一变量 = 24 个可见键**怎么分配（连续最近 24 个 vs 最近 16 个 + 最前 8 个恒可见槽）。
所有臂都是同一 stock dense decoder-only 骨架（`train_depth_ar.py --dense --topology sequential`，
无 carry 门、无退出、仅最终深度 CE、tied 读出），同一配方逐字段一致：
d=256 / L=12 / h=8 / T=256 / mlp_mult=6 / dropout=0 / norm_cap=1 / bs=32 / lr=1e-3 / 12k 步 /
98.3M token（enwik8 前 90M 训练、末 10M 评测、vocab 206）。

| 臂 | 来源 | 可见模式 | 每位置可见键数 `k_eff` | 每位置 KV 字节（fp16） | 参数量 |
|---|---|---|---|---|---|
| `full` | 复用 `2610-frozen-trunk-jumphead-13m/trunk.pt` | 全因果（上界参照） | 256 | 3,145,728 | 13,536,768 |
| `win16a8_s0` / `_s1` | 复用 `2662-hard-window-anchor-13m/train/` | **局部+全局**：最近 16 + 最前 8 恒可见 | 24 | 294,912 | 13,536,768 |
| `win24a0_s0` / `_s1` | **本实验训练** | **均匀条带**：最近 24（对角带，带宽 24） | 24 | 294,912 | 13,536,768 |
| `win192a8_s0` | 复用 `2616-window192-trained-13m/train/window192` | 局部+全局的另一剂量点（最近 192 + 8） | 200 | 2,457,600 | 13,536,768 |

- **可见键数逐行相等（硬约束，落盘核）**：两臂都走同一个仪器
  `pathlm.depth_ar.window_anchor_keep_mask(T, window, anchors)`：
  `keep[i,j] = (j ≤ i) ∧ ((i − j) < window ∨ j < anchors)`。对 `(16, 8)` 与 `(24, 0)`，
  逐行计数恒等于 `min(24, i+1)`（`i ≤ 23` 两臂的可行集合并集连续、`i ≥ 24` 两臂都恰好 24 个），
  本实验逐行核这个等式并落盘（`visible_keys_per_row`），不靠文字声明。
- **参数量相等自动满足**：掩码不引入参数（`params_added = 0`）、不改块类（两臂都是 stock `Block`）。
- **非必要性例外**：无。基底是主流标准 decoder-only transformer；全局槽是可见集干预，不是第二个因子。
- **本实验不改共享文件**：`pathlm/**`、`train_depth_ar.py`、`tools/**`、`tests/**` 全部只读调用；
  新代码只在本目录（`train_arms.sh` 取锁驱动 + `run.py` 评测与对账）。

## 3 判据线（预登记，写死后不得改）

Δ 方向统一（`SCHEMA.md` §17.3）：**Δ = bpc_对照 − bpc_实验**，本实验的「实验」= 局部+全局（`win16a8`），
「对照」= 同可见键数的均匀条带（`win24a0`）⇒ **Δ = bpc_win24a0 − bpc_win16a8，正 = 非均匀更好**。

| # | 判据 | 判定线 | 该涨 / 该跌 / 必须不变 |
|---|---|---|---|
| ① **主判（质量）**：同 `k_eff = 24` 下「局部+全局 vs 均匀条带」的 clean bpc 配对差 | 同一批评测窗口上的配对 Δ，对 **0.014 带**；**两个 seed 各自算一次**，两 seed 同向且都越带才算「可排序」；三态：`Δ ≥ +0.014` = 非均匀占优、`|Δ| < 0.014` = 无质量增量、`Δ ≤ −0.014` = 非均匀更差 | 该涨（非均匀该更好）或如实报无增量 / 更差 |
| ② **质量参照线**：`win16a8` / `win24a0` 相对 `full` 的配对 Δ | 同 0.014 带，只作「少看 232 个键的代价」参照，**不与 ① 混用** | 该跌 |
| ③ **成本（该跌）**：每位置 KV 缓存字节 | 解析量 `12288 B × k_eff`（fp16、MHA、d=256/L=12）：`k_eff = 24` ⇒ −90.6%（相对 `full`），且 ① 的两臂**同字节** ⇒ 若 ① 有差，差不能记在字节账上。**本实装是稠密 bool 掩码，省幅是结构上界，未兑现为实测**，如实标注 | 该跌 |
| ④ **必须不变**：因果性 | `keep ⊆ causal` 逐位为真（`mask_offending_future_pairs == 0`） | 必须不变 |
| ⑤ **必须不变**：可见键数逐行相等 | `win16a8` 与 `win24a0` 的逐行可见键数逐行相等，`visible_keys_per_row.mismatch_rows == 0` | 必须不变 |
| ⑥ **必须不变**：训练/评测掩码一致 | 训练与评测都调同一个 `window_anchor_keep_mask`（同一 `(window, anchors)`），落盘两处的 `n_visible_pairs` 相等 | 必须不变 |
| ⑦ **必须不变**：参数量与配方 | 五臂参数逐位相等（13,536,768）；配方逐字段一致（`results.json::arms`） | 必须不变 |
| ⑧ **复用臂等值门**：复用 checkpoint 的读数可复现 | 本实验重新评测 `win16a8_s0` 必须逐位复现 `2662` 的落盘 bpc | 必须不变 |
| ⑨ **申报不进判据**：墙钟、峰值显存、步时 | 逐臂申报；`docs/methodology.md` §3 禁把墙钟当判据 | — |

**长程召回腿（本实验不重跑，如实登记）**：本判据的率类腿（全局槽是否让远端内容可召回）不跨量纲借
0.014 带，需要按针配对的 CI；它在 `axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m/results.json`
已落盘（路径见下面这条指针），本实验**只指针引用，不重跑、不转述**。
本实验的判词因此**只对质量腿（bpc）成立**，在 `conclusion` 里写清这条边界。

## 4 指标口径（操作性定义）

- **clean bpc**：`masked_logits(model, x, keep_mask)` → 因果移位 CE 的 token 平均（bits/char，越低越好）；
  `n_batches = 96`、`batch_size = 16`、`window_seed = 2`（与 `2662` 的 `realtext_probe.win16_bpc` 同批数、
  同窗口序 ⇒ 臂间差是**配对差**）。**每臂在自身可见性下评测**：`win16a8` 用
  `window_anchor_keep_mask(256,16,8)`，`win24a0` 用 `(256,24,0)`，`full` 用 `None`。
- **配对统计**：逐窗口 `bits_per_window` 的配对 Δ 的均值 / 标准差 / 标准误 / t 值 / 同向窗口数。
- **可排序门**：bpc 类只用 0.014 带（配对、≥5 窗口）；**不跨量纲**借用给任何率类指标。

## 5 预算与成本

| 项 | 值 |
|---|---|
| GPU 预算 | 2 支新臂（`win24a0` seed 0/1）× ≈13 min + 冒烟 ≈1 min ≈ **27 min**（本实验自报上限 32 min）；复用臂只做评测 |
| 单臂命令超时 | ≤20 min（`train_arms.sh` 的 `timeout`） |
| 显存 | 逐臂实测 `peak_gpu_mem_mb`；礼貌线 5GB net（`SCHEMA.md` §20.2） |
| 可中断性 | 每 200 步落 `train_log.jsonl`；断点即重取锁重跑该臂（驱动先清日志） |

## 6 失败决策树

| 症状 | 处置 |
|---|---|
| `win24a0` 训练 NaN / 发散 | 先修数值腿再判机制；该臂不进判据 |
| 掩码不满足 `keep ⊆ causal` | 实现/构造缺陷，中止判决 |
| 逐行可见键数不等 | 对齐失败，判决作废（报「不可判」而不是硬判） |
| 复用臂读数不复现 `2662` | 评测口径不一致 ⇒ 先对账，不判 |
| 主判 `|Δ| < 0.014`（两 seed 都不越带） | 如实报「非均匀在本尺度无质量增量」，并把「全局槽的收益只在率类腿才可能出现」写成边界 |
| 两 seed 方向相反 | 报「本 n 下不可判」，不取平均当结论 |

## 7 产出

本目录下 `README.md`（本文件）/ `train_arms.sh` / `run.py` / `results.json` / `analysis.md` /
`train/win24a0_s{0,1}/{model.pt,train_log.jsonl,gate_log.jsonl,results.json,stdout.log}`。

## 8 复现 / 重跑

```bash
# 两臂训练（win24a0 的 seed 0/1；逐臂取/放 SCHEMA §20 GPU 锁，臂间 61 s；单臂 timeout 20 min）
bash train_arms.sh
# 评测 + 等值门 + 对齐核查 + 申报量（写 results.json，<3 min GPU）
python3 run.py
```

- 冒烟（300 步、量步时、不进判据）：`MODE=smoke bash train_arms.sh`。
- 墙钟与峰值显存见 `results.json::train_declaration`（三条复用臂的读数取自各自落盘文件，
  路径在 `source` 字段里）。
