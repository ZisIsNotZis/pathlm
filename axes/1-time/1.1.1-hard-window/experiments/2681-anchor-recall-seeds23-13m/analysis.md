# 2681 · 判读：锚点召回腿补到 n=4（判据、口径与读数位置）

> 契约 = 同目录 `README.md`（§5 判据线预登记、写下后未改；绑定见 `prereg_binding.json`，
> 覆盖 `README.md` + `run.py`）。原始数字 = `results.json` / `needle_raw.json`；
> 运行流水 = `run_log.txt`；权重 sha256 见 `results.json::checkpoints`。
> **本文件的分工**：判据、口径与人话解释写在这里；**每个数字都从 `results.json` 抄**，
> 本节以下每个「读数位置」都给出键路径，便于逐条核对。

## 0 读数状态（当前事实）

| 臂 | 训练 | 说明 |
|---|---|---|
| `win16a_s2`（窗口 16 + 锚点 8，seed 2） | **已完成**（`train/win16a_s2/model.pt`，rc=0） | 本实验第一个新臂 |
| `win16a_s3` / `win16_s2` / `win16_s3` | **未训练**（声明缺失，不是零） | GPU 让位：卡上另有更高优先级作业（`scripts/train_sft.py` 与 `llama-server`，合计占 20.6/24 GB）⇒ 等它们退出、空闲显存 ≥ 12 GB 后由 `wait_and_train.sh` 起训 |

因此：**本文件尚无判词**。判据①（复现门）与判据③（n=4 跨 seed 主判）都依赖 4 个 seed 齐备；
在齐备之前，任何「可判 / 不可判」的写法都是把一个未测量的量写成测量值。

## 1 这个实验在问什么（人话）

`2662` 量到过一件小事：给「只看最近 16 个字节」的模型再加**最前面 8 个字节恒可见**（锚点槽 / sink），
「扎在锚点区里的针」（这条短语能不能被接出来）的召回率比不扎针的地板高一点点——seed 0 高 2.592 个
百分点，seed 1 高 1.097 个百分点。问题是：这「高一点点」到底是真的，还是这条仪器在 1,003 根针上
本来就分不出 1–3 个百分点的差别？

`2662` 按预登记的门（「两个 seed 都要过两套 CI」）判了**不可判**。本轮要做的是把这件事**判死**：
要么把它压成「可判且为正」，要么写清是哪一级不够——**不是**再报一次同样的「不可判」。

关键的一步是先把「补 seed 能不能解决」算清楚（`README.md` §2 的功效表）：

- 每个 seed 只有 1,003 根锚点区针，地板召回 ≈ 45–47%，两边一比，未配对两比例差的 95% 半宽就是
  **±4.37 个百分点**；而效应只有 1–3 个百分点 ⇒ 单个 seed 自己永远过不了这道门（要 2,850–15,900
  根针/seed）。
- 所以旧门是一个**功效近乎 0 的合取式**：补 seed 只是往合取式里加约束项，n 再大也不会变可判。
  这不是「锚点没信号」，是「仪器在 1,003 根针上分不出来」。

于是本轮的判定单位改成**种子层**：看 4 个 seed 的 `Δ_pp` 均值、方向一致性，以及两个跨 seed
区间（t 区间与两级 bootstrap）的下界。判据线的常数、两套 CI 的实现、配对口径、方差带
**一项都没改**，旧读数一项都没删（`README.md` §5a 把改动范围写死）。

## 2 判据线对照（预登记 → 读数位置 → 判词）

| # | 判据 | 判定线（预登记） | 读数位置 | 判词 |
|---|---|---|---|---|
| ① | 复现门（仪器自检） | 复用臂 6 个 bpc 与 `win16a_s0/s1` 的 anchor/beyond 桶 `(n, planted_hits, control_hits)` 与 `2662` 逐位相等 | `criteria.c1_reproducer_2662.{bpc,bpc_all_equal,needle_buckets,needle_all_equal,passes}` | 待读数 |
| ② | per-seed 读数（口径不变，不作门） | 逐 seed `Δ_pp` + Newcombe + 配对 bootstrap，逐 seed 如实报 | `criteria.c2_per_seed_recall_unchanged_instrument.per_seed.<seed>` | 待读数 |
| ③ | **n=4 主判** | `≥3/4` seed 的 `Δ_pp` > 0；均值 > 0；**两个**跨 seed 区间下界都 > 0（t 区间 df=3 + 两级 bootstrap） | `criteria.c3_cross_seed_n4.{mean_delta_pp,two_level_bootstrap,fixed_seed_pooled,decidable}` | 待读数 |
| ④ | 负对照：窗外停地板 | 4 个锚点臂的 `beyond` 桶植入 vs 地板差 CI 含 0 | `criteria.c4_beyond_bucket_stays_at_floor.<arm>.ci_contains_0` | 待读数 |
| ⑤ | 毁源敏感度 | 变体 B（毁 `[8,64)`）锚点臂 `max|Δlogit| == 0`；变体 A（毁 `[0,64)`）锚点臂必然 `> 0` 且 `premise_holds=false` | `criteria.c5_beyond_anchor_bitwise_insensitive` / `criteria.c5_far_source.<arm>.{far_all,beyond_anchor}` | 待读数 |
| ⑥ | 因果性 + 掩码 | 前半段 `max|Δlogit| == 0`；`visible_pairs` = 5,868 / 3,976；参数量逐位相同 | `criteria.c6_causality_no_future_leak` / `criteria.c7_mask_nondegenerate_and_reuse_equal` / `selfchecks.*.mask` | 待读数 |
| ⑦ | 质量腿（n=4，旁证） | 4 个同 seed `price_bpc` + 跨 seed 均值与 t 区间；与**冻结**带 0.000983 并列 | `quality_leg.price.<seed>.price_bpc` / `quality_leg.price_cross_seed` | 待读数 |

## 3 口径（复查时必须先读这一节）

- **每臂在自己的训练可见性下评测**：锚点臂 `window_anchor_keep_mask(256,16,8)`（`k_eff=24`），
  纯窗臂 `(256,16,0)`（`k_eff=16`）。所以 `price_bpc` 是**「同预算、不同可见集」**的差，
  不是同掩码的差。
- **锚点桶的归属规则**：一律 `anchors=8`（`p<8` 的针记为 `anchor`），九臂同规则 ⇒ 同索引可比。
  `beyond` = 非锚针且**有任意一个位置不可见**（半删即被删）。
- **地板**：同一 generator 序列、`plant=False` 的配对控制臂；`run.py` 断言主臂与控制臂的针位
  `meta` 逐位相同。
- **两套 CI 并列**：Newcombe 两比例差（正态近似 / 未 pooling）+ 按针配对 bootstrap（2,000 次）。
  等率类指标**不借用** bpc 的 0.014 带。
- **方差带冻结**：`sigma_cell = 0.000983` 是 `2662` 的两 seed 极差，本实验**不重算**；4 个 seed
  的极差必然更宽，那一列只是描述量（`quality_leg.bands.seed_range_observed_n4`），不得回填成新带。
- **墙钟不进判据**（只作过程指标，见 `cost`）。

## 4 上游事实（本实验不重测、只引用）

| 事实 | 值 | 出处 |
|---|---|---|
| 锚点臂 vs 同 seed 纯窗臂的 clean bpc 差 | −0.005583（seed 0）/ −0.007294（seed 1） | `axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m/results.json::quality_leg.price` |
| 锚点桶召回差（旧门的读数） | +2.592pp / +1.097pp；`passes_both` 两 seed 均 false | 同上 `::criteria.c2_anchor_bucket_above_floor` |
| 评测期关掉锚点掉多少 | +0.024600 / +0.016607 bpc（0 训练 token） | `axes/1-time/1.1.1-hard-window/experiments/2679-anchor-eval-ablation-13m/results.json` |
| 窗口 16 → 全因果的总缺口 | 0.125421 bpc | `2662::quality_leg.gap_win16_full` |

## 5 下一步（本文件的下一次改写）

三臂训齐 → `make_config.py` → `run.py` → 把 §2 的每一行「待读数」换成从 `results.json` 抄来的
数字与判词，并把 §0 的读数状态换成「4/4 臂齐备」。
