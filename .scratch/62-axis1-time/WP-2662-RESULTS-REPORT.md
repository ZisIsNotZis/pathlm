# WP-2662 结果报告 · `1.1.1` 锚点臂自包含复算（纯推理判读）

> actor: **WP-R2 worker**（本任务唯一写入者）；lead: **axis1-lead-4**。
> 状态：**完成**（判读完成；未改任何预登记判据线，未改任何 `project.yaml`）。
> 契约：`.scratch/62-axis1-time/TICKET.md` §10.24 **WP-R2**；预登记判据线：`2662…/README.md` §4（未改）。
> 实验目录：`axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m/`
> 同目录交付：`2662…/analysis.md`（人话摘要 + 六问 + 逐条判据表）。
> 运行：`EVAL done rc=0`（`driver_all.log` @`2026-10-14T23:09:16+08:00`）、训练 `win16a_s1 rc=0`。
> **数字来源**：`results.json`（键路径逐条给出）/ `run_log.txt` / `driver_all.log`。权重清单 `MODEL.md` 未落盘（见 §6）。

---

## 0. 人话摘要

**锚点是「质量收益」，不是税；但它只是把最前面 8 个字节变得永远可见，搬不动远处；而「锚区召回
被抬高」在 2 seed 上没达到预登记门槛。**

- **质量腿**：同 seed 相比，锚点臂 bpc 更低——seed 0 `1.628110 → 1.622527`（**−0.005583**）、
  seed 1 `1.629093 → 1.621799`（**−0.007294**）；两种子**同向为正收益**，**都越本格自测带
  （0.000983）**、**都不越 0.014 跨格带**。量级 ≈ **0.34% / 0.45%**。
- **召回腿**：锚点桶（针植入 `p<8`）植入 vs 配对地板——seed 0 **+2.592pp**（Newcombe `[−1.774, 6.958]`
  含 0、配对 bootstrap `[1.196, 4.187]` 不含 0）、seed 1 **+1.097pp**（Newcombe `[−3.275, 5.468]`、
  配对 bootstrap `[−0.299, 2.592]` **都含 0**）⇒ 按预登记「两 seed 都过才算有区分力」
  **未达成**（`win16a_both_seeds_pass_both_ci=false`）⇒ **不可判**，非否证。
- **负对照**：窗外桶（beyond）两个 seed 都**停在地板**（CI 含 0）⇒ 锚点**没有把远处搬近**。
- **毁源敏感度（判据④）**：锚点臂对锚点槽**之外**的 `[8,64)` 被毁**逐位不敏感**（`max|Δlogit|=0`）；
  对含锚点槽的 `[0,64)` 敏感（`max|Δlogit|=6.32/6.11`，`premise_holds=false`，= 预登记物理事实）。
- **必须不变项**：因果性、掩码非退化、复用臂逐位复现 2634 —— **三条全过**。

---

## 1. 口径声明（不得跳过）

- **「同预算、不同可见集」比较，不是「同掩码」比较**：锚点臂（`win16a_*`）的 clean bpc 是在
  **它自己的训练可见性**（窗口 16 + 锚点 8，`k_eff=24`，`294,912 B/位置`）下测的；纯窗臂
  （`win16_*`）在**锚点 0**（`k_eff=16`，`196,608 B/位置`）下测的。两者共享**同一 token 预算**
  （12k 步 × bs32 × T256 = 98.3M token/臂），差的只有可见集。
- **两带并列、不混用**（C1 裁决）：bpc 类主判用**本格自测带** `quality_leg.bands.sigma_cell = 0.000983`
  （= 本实验自己的换种子极差，描述量、非方差上界），`0.014` 跨格带（`quality_leg.bands.cross_cell_band`）
  **并列作参考**；**不得**把 0.014 借给召回等率类。召回等率类只用 Newcombe / 配对 bootstrap 的 CI。
- **禁外推**：enwik8 / T=256 / 13M（d256×12×8×mlp6）/ 12k 步 / 窗口 16 + 锚点 8 / 该快照。

---

## 2. 质量腿（锚点的价格）

| seed | `bpc(win16a)` | `bpc(win16)` | `price_bpc` | 符号 | bpc 源键 |
|---|---|---|---|---|---|
| 0 | `quality_leg.bpc.win16a_s0 = 1.622527` | `…win16_s0 = 1.628110` | **−0.005583** | **质量收益** | `quality_leg.price.0.price_bpc` |
| 1 | `quality_leg.bpc.win16a_s1 = 1.621799` | `…win16_s1 = 1.629093` | **−0.007294** | **质量收益** | `quality_leg.price.1.price_bpc` |

- **符号与量级**：两种子**都为正收益**（`price.sign = "锚点是质量收益（更好）"`），量级
  ≈ **0.34% / 0.45%**。相对全因果的差距由 `gap_win16_full = 0.125421` 缩到
  `gap_win16a_full = 0.119838`（缩小 ≈ **4.45%**）。
- **双带并列（不混用）**：本格自测带 `sigma_cell = 0.000983`（锚点臂内 `per_kind.win16a = 0.000728`、
  纯窗臂内 `.win16 = 0.000983`，取最大）⇒ 两收益**越本格带可排序**（`sortable_vs_sigma_cell=true`）；
  0.014 跨格带 ⇒ **不越**（`sortable_vs_0.014=false`）。两带**并列报、不混用**。
- **成本（必须并报）**：收益来自**更大的可见集**——`k_eff 16→24`（+50%），每位置 KV
  `196,608 → 294,912 B`（+**98,304 B/位置**，= `rows[*].kv_bytes_per_position`）。
  ⇒ 正确表述 =「**锚点用 50% 更多可见键换到 ≈0.4% 的质量改善**」，**不是免费改善**。
- **未排除的解释**：无评测期锚点消融 ⇒ 无法区分「收益来自锚点槽本身」还是「训练见过更多键」。

---

## 3. 召回腿

### 3a. 锚点桶（针植入 `p<8`，`n=1003`）——主判据

| 臂 | 植入召回 | 地板召回 | `delta_pp` | Newcombe 95% (pp) | 配对 bootstrap (pp) | 两套都过 |
|---|---|---|---|---|---|---|
| `win16a_s0` | 0.481555 | 0.455633 | **+2.592** | **[−1.774, 6.958]** 含 0 | **[1.196, 4.187]** 不含 0 | ❌ |
| `win16a_s1` | 0.482552 | 0.471585 | **+1.097** | **[−3.275, 5.468]** 含 0 | **[−0.299, 2.592]** 含 0 | ❌ |
| `win16_s0`（对照） | 0.467597 | 0.467597 | 0.0 | [−4.367, 4.367] | [−1.097, 1.097] | ❌ |
| `win16_s1`（对照） | 0.467597 | 0.467597 | 0.0 | [−4.367, 4.367] | [−1.097, 0.997] | ❌ |

键路径：`needle_leg.per_arm_bucket_planted_vs_floor.<arm>.anchor.{delta_pp,newcombe_diff_ci95_pp,paired_bootstrap_pp,passes_newcombe,passes_paired_bootstrap,passes_both}`。

- **门**：`criteria.c2_anchor_bucket_above_floor.win16a_both_seeds_pass_both_ci = false`；
  `.intervention_arms_all_pass_newcombe = false`。
- **按预登记「两 seed 都过才算有区分力」⇒ 未达成**：seed 0 只在配对 bootstrap 上排除 0，
  seed 1 两套 CI 都含 0。结论写「**本 n 下不可判**」（方向为正、两种子同向，但幅度不足以越门）。
- **同索引硬子集旁证**（`hard_subset_same_index.anchor.n_intersection = 359`）：
  `win16a_s0` vs `win16_s0` = **+3.621pp CI [1.170, 6.072]**（`passes=true`）；
  `win16a_s1` vs `win16_s1` = **+1.114pp CI [−0.907, 3.135]**（`passes=false`）
  ⇒ 同向、同样「只有 s0 过」，不改变结论。

### 3b. beyond 桶负对照（窗外，`p≥8` 且有位置不可见）——关键区分

| 臂 | 植入召回 | 地板召回 | `delta_pp` | Newcombe 95% (pp) | CI 含 0 |
|---|---|---|---|---|---|
| `win16a_s0` | 0.466481 | 0.465505 | +0.098 | [−1.535, 1.73] | ✅ |
| `win16a_s1` | 0.467178 | 0.467735 | −0.056 | [−1.689, 1.577] | ✅ |

键路径：`criteria.c3_beyond_bucket_stays_at_floor.<arm>.{delta_pp,newcombe_diff_ci95_pp,ci_contains_0}`。
⇒ 锚点臂窗外召回**仍停地板**（两 seed `ci_contains_0=true`）⇒ **锚点保住的是锚区，不是把远处搬近**。

---

## 4. 毁源敏感度（判据④，锚点算不算信息右移）

**实测**：被评行 `[244,255)`，`rf = 12×(16−1) = 180`。
- **变体 B `beyond_anchor`（毁 `[8,64)`，保住锚点槽）**：锚点臂
  `far_source_probe.win16a_s0.beyond_anchor.{max_abs_logit_diff_on_eval_rows=0.0, delta_bpc_far=0.0, premise_holds=true}`；
  `win16a_s1` **同（0.0 / 0.0 / true）** ⇒ **对锚点槽之外的前缀被毁逐位不敏感**。
- **变体 A `far_all`（毁 `[0,64)`，含锚点槽）**：锚点臂
  `max_abs_logit_diff_on_eval_rows` = **6.319242**（s0）/ **6.108980**（s1）、`premise_holds=false`
  —— 这是**预登记写死的物理事实**（锚点槽在被毁区内，直接可见 ⇒ 不可能逐位不敏感），
  以 `require_no_overlap=False` 运行并记「不可判」。
- **汇总门**：`criteria.c4_beyond_anchor_bitwise_insensitive = true`、
  `criteria.c4_far_all_window_bitwise_insensitive = true`。

**读法**：变体 B 逐位不敏感 + §3b beyond 停地板 ⇒ 「**锚点 = 少量恒可见槽（直接注意力），
不是记忆通道**」成立。锚点臂对锚点槽内的 `[0,8)` 敏感（变体 A 的 `>0`）恰好证明它**直接看**锚点槽。

---

## 5. 必须不变项（逐条）

| 项 | 结果 | 键路径 / 值 |
|---|---|---|
| **因果性** | ✅ 5 臂全过 | `selfchecks.<arm>.causality.max_abs_logit_diff_prefix = 0.0`；`criteria.c5_causality_no_future_leak = true`；后半段确实变（`changed_suffix_max_abs_logit_diff` 24.6–29.9）⇒ 探针有抓力 |
| **掩码非退化** | ✅ 过 | `selfchecks.win16a_*.mask.{visible_pairs=5868, expected=5868, ok=true, nondegenerate=true}`；`win16_* = 3976`；均 < `causal_pairs=32896`；反事实 `mask_counterfactual.max_abs_logit_diff_masked_vs_open` 15.3–18.5（掩码活着） |
| **复用臂逐位复现 2634** | ✅ 过 | `quality_leg.reuse_equivalence.win16_s0.{bpc=1.62811, bpc_2634=1.62811, equal=true}`；`.win16_s1.{bpc=1.629093, bpc_2634=1.629093, equal=true}`；`criteria.c6_mask_nondegenerate_and_reuse_equal = true`。（独立对账 `2634/results.json::quality_leg.rt_none_s0.bpc_win16 = 1.62811` / `rt_none_s1 = 1.629093` 一致） |
| **参数量一致（锚点不引入参数）** | ✅ 过 | `selfchecks.*.n_params = 13536768`（5 臂相同） |

---

## 6. 缺项 / 交付缺口

- 本目录 **`MODEL.md` 未落盘**（`README.md` §8 规划了它）。权重 sha256 已在
  `results.json::checkpoints.<arm>.sha256`（5 臂 `sha256_verified=true`）与 `driver_all.log`
  （`[verify] … sha256_ok=True`）中，但独立权重清单缺失。**不影响本判读数字**，列为交付缺口。

---

## 7. 建议的 `1.1.1` `project.yaml` 字段文本（**仅建议，未改任何 `project.yaml`**）

> 依据：本 WP 只完成 `work_packages` 里的 **WP1（自包含复算）** 的一部分；主正面主张（锚点召回）
> 在预登记门槛下**未达成（不可判）**，质量腿与机制分类**成立**。其余 WP2/WP3/WP4 未开始。
> `status` 合法取值 = `未开始 / 进行中 / 已完成 / 被否证`（`docs/research-schema.md` §7）。

**建议 `status`: `进行中`**（保持不变）。理由一句：主正面主张未达预登记门槛、且 WP2/WP3/WP4 未做，
不能判 `已完成`；锚点召回只是「不可判」而非「被否证」，也不宜判 `被否证`。

**建议 `conclusion`（改写，替换现「锚点通道是真产物 needle 76.8%→93.3%」这一未经本口径复现的表述）：**

> 已有仓内自包含复算（`experiments/2662-hard-window-anchor-13m/`，rc=0）：锚点臂（窗口 16 + 锚点 8，
> seed 0/1）相对同 seed 纯窗臂（窗口 16 + 锚点 0）的 clean bpc 差 = **−0.005583 / −0.007294**
> （两种子同向 = **质量收益**，越本格自测带 0.000983、不越 0.014 跨格带），代价 = 每位置多
> 98,304 B KV（可见键 16→24）。锚点桶（针植入 `p<8`）植入 vs 配对地板 = +2.592pp / +1.097pp，
> 按预登记「两 seed 都过」门槛**未达成 ⇒ 召回腿在本 n 下不可判**（非否证）。窗外（beyond）桶
> 两 seed 均停地板 ⇒ 锚点保住的是锚区、**不是信息右移通道**（判据④变体 B 逐位不敏感佐证）。

**建议 `conclusion_review`：**

> WP1 自包含复算完成（2662，rc=0；因果性 / 掩码非退化 / 复用臂逐位复现 2634 三条必须不变项全过）。
> 旧主张「锚点通道是真产物 needle 76.8%→93.3%」**在新口径（配对无针地板 + 两种子门）下未复现**：
> 该旧数字是**另一台仪器**（needle v2 合成口径、无配对地板）上的**绝对召回**跳变，且来源可溯到
> 「距离惩罚」而非锚点槽本身（`.scratch/07-x1-x3/evidence/SUMMARY.md`：`76.8` 是 X2v2、`93.3` 是
> X2v2+距离惩罚的 X3）⇒ 不能直接与本次读数比。本口径下成立的是「**质量收益 + 锚区可召回 + 窗外停地板**」。
> 独立复核未做（本判读之后另派 fresh 只读复核，见 §10.24）。

**建议 `gaps`（增改）：**

> - 锚点召回腿在 n=2 下不可判（s1 两套 CI 均含 0）：需补 seed 2/3 或换更低先验地板（§9）。
> - 缺评测期锚点消融 ⇒ 质量收益无法区分「锚点槽本身」与「训练见过更多键」。
> - `2662` 目录 `MODEL.md` 未落盘（权重 sha256 已在 `results.json::checkpoints`）。
> - （原有）无仓内自包含复算——**WP1 已部分消除**（本次 2662 即仓内复算）。

**`experiments` 是否加本目录**：**建议加**一条
`"axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m"`。理由：它是本项
**第一条仓内自包含的窗口/锚点臂 `results.json`**（WP1 的产物），是本项主数字的当前权威锚点；
现 `experiments` 列表里只有长 T 子线的搬迁目录。**（仅建议，未改。）**

---

## 8. 一句话：这条结果改变了什么判断 + 边界

**改变**：1.1.1 旧结论把「锚点通道是真产物（needle 76.8%→93.3%）」当已确立事实——本次在
**仓内自包含口径**上把它拆成两半：**成立的是「锚点是质量收益、且锁定的是锚区（直接可见），
窗外搬不动」**；**未成立的是「锚点抬高召回」**（配对地板 + 两种子门下不可判）。
⇒ 锚点作为**质量手段**值，作为**信息右移手段**不值。
**边界**：仅覆盖 enwik8 / T=256 / 13M / 12k 步 / 窗口 16 + 锚点 8 / 2 seed；旧 76.8→93.3 出自
另一台仪器（合成口径、无配对地板，且跳变可归因于距离惩罚而非锚点槽），与本读数**不可直接比**；
召回不可判≠否证，加 seed 后方可定论。

---

## 9. 下一步最小实验（带成本）

1. **锚点臂加 seed 2/3** ⇒ 压窄锚点桶两套 CI（≈10 min 训练 + ≈1.5 min 评测 / 臂）。
2. **评测期锚点消融**（对 `win16a_*` 评测时 `anchors=0`）⇒ 归因质量收益（纯推理 ≈1.5 min）。
3. **更低先验地板**的针形式 ⇒ 提高召回仪器区分力（当前地板 45–47% 吃掉信号）。

---

## 10. 纪律与可核

- **未改**：`run.py` / `config.json` / `README.md` / `make_config.py` / 任何 `project.yaml`；
  未跑 GPU、未跑 git、未碰其它目录、未删文件。
- 本报告与 `2662…/analysis.md` 是本次**唯二**写入。
- 每个数字给出 `results.json` 键路径；率类只用 Newcombe / 配对 bootstrap / Wilson CI。
