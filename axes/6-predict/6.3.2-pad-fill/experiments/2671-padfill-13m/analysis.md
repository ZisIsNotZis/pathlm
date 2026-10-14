# analysis — 2671-padfill-13m（S2 全量扫描：5 臂 × 2 seed）

## 人话摘要

**Nk 快照口径：本实验 `S = 1500`，与仓库既有 6000 步档不可比；只读臂间增量与排序**（`SCHEMA.md` §20.3）。

我们在**同一 harness、同一 token 预算、同一结构**（唯一变量 = 注意力可见性 + 损失位置集）下，比较「一次前向把所有待填位置填出来」的填空臂和全因果 AR 参照：AR 逐 token 生成，填空臂一次前向填出全部 PAD。主判是**同一批待填位置 `M_r` 上的 bpc 差**（每字符比特数，越低越好；Δ = 填空 − AR，正 = 填空更差），带 0.014 bpc 参考带。
参照系：AR 臂全位置 teacher-forced bpc = 2.2704（越低越好；各种子 2.1842, 2.3565）。相比 AR，填空臂在待填位置上的质量代价：
- `fill_bidir_r10` Δ=+2.1453 bpc (95% CI [+2.0400,+2.2500], 越带, CI 不含 0)
- `fill_bidir_r25` Δ=+1.0726 bpc (95% CI [+1.0034,+1.1455], 越带, CI 不含 0)
- `fill_bidir_r50` Δ=+2.3735 bpc (95% CI [+2.2821,+2.4720], 越带, CI 不含 0)
- `fill_causal_r25` Δ=+1.2362 bpc (95% CI [+1.1822,+1.2903], 越带, CI 不含 0)

- **r 形状（L2）**：fill_bidir 的 Δ 随 r ∈ {10%,25%,50%} **不单调**（r=0.25 代价最小）（序列 r=0.10: +2.1453, r=0.25: +1.0726, r=0.50: +2.3735）。⇒ 契约 §6「随 r 单调恶化」子条件**未满足**。
- **可见性对照（L3）**：`fill_causal_r25 − fill_bidir_r25` Δ=+0.1635 CI[+0.1096,+0.2140] (CI 不含 0 ⇒ 可见性有独立增益)。
- **硬门（L0/L0b）**：L0 **通过**；L0b gate probe 两条 **均变红（仪器可失败）**。
- **禁墙钟（L5）**：`results.json` 墙钟语义字段 = []（0 命中）。

> 信息集不对称（必须同时读）：`fill_bidir` 的信息集 = 全部**非 M** 真 token（含右侧）+ 自身位置的 mask 占位；`ar_at_M_r` 的信息集 = 全部**左侧**真 token。两者**互不包含**，因此主判差**不是**「纯机制优劣」，而是两种可用信息下的质量差。

## 口径与配置

- matched-step：steps=1500、batch=32、T=512、lr=0.001、Adam(no weight decay)；模型种子 = [0, 1]
- 数据顺序 seed=2671（各臂同 token 预算）；训练 mask seed=2672；评测 mask seed=2674
- 评测：enwik8 val 的 64 个固定 offset × T=512 block；位置集 M_r：i ≥ 8 且以概率 r 取「被填」
- 主判 bootstrap：block 聚类配对，B=2000、seed=2673、percentile 95% CI；参考带 = 0.014 bpc
- 结构：d_model=256、n_layers=12、n_heads=8、mlp_mult=6、params=12,813,312

## 数字表：主判与参照 bpc

| 臂 | 种子 | attn | r | ar_at_M bpc | fill_M bpc | Δbpc | 95% CI | 越 0.014 带 |
|---|---|---|---|---|---|---|---|---|
| `fill_bidir_r10` | 0 | full | 0.10 | 2.22093 | 3.86745 | +1.64348 | [+1.54595,+1.75302] | 是 |
| `fill_bidir_r10` | 1 | full | 0.10 | 2.39539 | 5.04788 | +2.64708 | [+2.51438,+2.77734] | 是 |
| `fill_bidir_r10` | **合并** | — | — | — | — | **+2.14528** | **[+2.04002,+2.25005]** | 是 |
| `fill_bidir_r25` | 0 | full | 0.25 | 2.20235 | 3.43149 | +1.22497 | [+1.15316,+1.30046] | 是 |
| `fill_bidir_r25` | 1 | full | 0.25 | 2.37170 | 3.29578 | +0.92031 | [+0.84916,+0.99513] | 是 |
| `fill_bidir_r25` | **合并** | — | — | — | — | **+1.07264** | **[+1.00339,+1.14548]** | 是 |
| `fill_bidir_r50` | 0 | full | 0.50 | 2.16433 | 4.20803 | +2.04465 | [+1.97108,+2.12353] | 是 |
| `fill_bidir_r50` | 1 | full | 0.50 | 2.34445 | 5.04468 | +2.70235 | [+2.59160,+2.82799] | 是 |
| `fill_bidir_r50` | **合并** | — | — | — | — | **+2.37350** | **[+2.28213,+2.47203]** | 是 |
| `fill_causal_r25` | 0 | causal | 0.25 | 2.20235 | 3.51846 | +1.31107 | [+1.25440,+1.36959] | 是 |
| `fill_causal_r25` | 1 | causal | 0.25 | 2.37170 | 3.53802 | +1.16128 | [+1.10591,+1.21842] | 是 |
| `fill_causal_r25` | **合并** | — | — | — | — | **+1.23617** | **[+1.18219,+1.29026]** | 是 |

| AR 参照 | 种子 | ar_full bpc | ar_at_M_10 | ar_at_M_25 | ar_at_M_50 |
|---|---|---|---|---|---|
| `ar` | 0 | 2.18422 | 2.22093 | 2.20235 | 2.16433 |
| `ar` | 1 | 2.35653 | 2.39539 | 2.37170 | 2.34445 |

## L0–L5 逐条

- **L0 不受未来泄漏硬门**：`pass=True` (bidir 扰动 M 真值 max|Δ|=0.0, causal 右侧扰动 max|Δ|=0.0)
- **L0b 可失败性自检**：`both_red=True` (泄漏构造 max|Δ|=1.9860783815383911, causal 改 full max|Δ|=0.554871678352356)；证据 = `evidence/l0_l0b_probe_s2.txt`
- **L1 可读性/质量主判**：齐全（逐臂逐 seed 的 ar_full / ar_at_M / fill_M bpc + 配对 Δ + CI）
- **L2 r 形状**：`monotonic_worsening_with_r=False`
- **L3 可见性对照**：有
- **L4 代价轴**：报告齐全（等效理论量，禁墙钟）
- **L5 禁墙钟自检**：`_wall_clock_fields=[]`（0 命中）。
- **L5 确定性**：eval_rerun_identical=None；short_train_20step_checksum_identical=None。判：**未通过 / 未验证**（本 revision 未取得 GPU 锁独立重跑，未取得非 None 值）。该子项以浮点逐位相等判定，GPU 归约顺序可使其跨运行抖动（同 revision 曾同时观测到 True 与 False）；历次观测值见 `run_log.txt`。 错误：short_train_error=GPUUnavailable('GPU lock held by pid 2607909 > 3.0s; escalating')。 错误：eval_error=GPUUnavailable('GPU lock held by pid 2607909 > 3.0s; escalating')。

## §6 否证条件逐条

- `cond_A_not_viable`：未触发 —— bidir CI 全不含 0=True、|Δ| 全越 0.014 带=True、r 单调恶化=False、causal 更差=True
- `cond_B_indeterminable`：未触发 —— 至少一条 bidir 的 |Δ| ≤ 0.014 或 CI 含 0 ⇒ 本尺度下该 r 的质量代价不可分（不得写成「无损」）
- `cond_C_visibility_no_gain`：未触发 —— fill_bidir_r25 − fill_causal_r25 CI 含 0=False ⇒ 填空视图的可见性在本尺度无独立增益

**判读**：逐臂分判（见数字表）：bidir 主判差可分辨 —— CI 全不含 0=True、|Δ| 全越 0.014 带=True；r 单调恶化=False ⇒ 未满足契约 §6「不成立」条件的全部要求，按各臂读数分判。

## 边界与未测

- 结论只适用于本设定（1500 步 / char enwik8 / 交付 12.8M 结构 / 该 r 集合），不外推；S < 6000 ⇒ 快照口径，与仓库既有 6000 步档不可比。
- 模型权重不入库（无 ckpt）；同 seed 由初始化 + 固定数据/mask 生成器重建。
