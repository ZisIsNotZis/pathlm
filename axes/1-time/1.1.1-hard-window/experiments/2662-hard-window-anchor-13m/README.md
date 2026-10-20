# 2662 · hard 窗口 + 锚点臂：锚点通道自包含复算（13M · 窗口 16 + 锚点 8 · 2 seed）

> 归属：`axes/1-time/1.1.1-hard-window`（研究项 1.1.1「hard 窗口」）的 **WP1 自包含复算**。
> 契约来源：`.scratch/62-axis1-time/TICKET.md` §10.2 **WP-C2 · 2662**（Phase C 第三轮）。
> 入口：`train_arms.sh`（取/放 §20 GPU 锁，按臂）→ `make_config.py`（生成 `config.json` + sha256）→
> `run.py`（质量腿 + 锚点召回腿 + 毁源敏感度 + 因果性/掩码自检）；语义层 `realtext_probe.py`；
> 统计层 `needle_stats.py`；原始结果 `results.json`；判读 `analysis.md`；权重清单 `MODEL.md`。

## 1 命题

**把 1.1.1 的核心正面主张——「锚点槽是真产物（旧读数 needle 76.8% → 93.3%）」——从
`.scratch` 摘要升级为仓内可复核证据，并给出锚点的代价（价格）。**

一句话可证伪的形式：*在 enwik8（真实文本）上，训练两臂（窗口 16 + 锚点 8，seed 0/1），
训练与评测的可见性都用 `window_anchor_keep_mask(T,16,8)`（= 最近 16 个位置 + 最前面 8 个
恒可见锚点槽）。用 2634 的 needle v3 仪器把针植入锚点区（`p < 8`）⇒ 归属桶 = `anchor`。
若锚点臂的 `anchor` 桶植入召回**高于配对无针地板**（差值 95% CI 下界 > 0），而窗外
（`beyond` 桶）召回**仍停在地板**，则「锚点槽把锚点区的内容变得可召回」这一主张在真实
文本口径上成立；否则如实记「不可判/被否证」。*

> 注意（本 WP 的核心区分）：锚点槽 = **最前面 8 个位置恒可见**，不是把远处内容搬近的记忆通道。
> 因此「锚点抬高召回」与「锚点算不算信息右移」是两件事，后者由 §4 判据④（毁源敏感度）单独判定。

## 2 单因素声明

- **基底**：stock dense decoder-only trunk（tied 读出 + 无 final LayerNorm + 顺序残差），
  d256 × 12 层 × 8 头 × mlp6 × T256，与 2634 / 2615 / 2616 **同一规格**。
- **唯一变量（本 WP 新臂之间）= 训练 seed**；**唯一变量（锚点臂 vs 复用纯窗臂）= 可见性掩码里
  的 `anchors`**（8 vs 0）。其余逐字段一致：数据（enwik8 前 90M 训练 / 末 10M 评测）、
  12k 步、bs32、lr 1e-3、AdamW+cosine、`--exit-mode uniform`、`--train-window 16`、评测协议。
  **锚点槽不引入参数**（掩码只改可见集）⇒ 锚点臂与纯窗臂参数量逐位相同（13,536,768）。
- **必要性例外（显式记账）**：锚点是一个**结构干预**（改了可见集），不是第二个因子；
  它正是 1.1.1 的「sink/锚点槽」这一手段本身。
- 本实验**不改任何共享文件**（`pathlm/**`、`train_depth_ar.py`、`tools/**`、`tests/**`、
  `docs/**`、其它轴）：`realtext_probe.py` / `needle_stats.py` 是 2634 的**目录内副本**
  （唯一改动见 §5a），其余是新增。

## 3 臂规格（5 臂；对照臂全部复用现成 checkpoint）

| 臂 | 来源 | `--train-window` | `--train-anchors` | seed | 可见性掩码（评测用） | 参数量 |
|---|---|---|---|---|---|---|
| `win16a_s0` | **本 WP 训练** | 16 | **8** | 0 | `window_anchor_keep_mask(256,16,8)` | 13,536,768 |
| `win16a_s1` | **本 WP 训练** | 16 | **8** | 1 | 同上 | 13,536,768 |
| `win16_s0` | 复用 `2634/train/rt_none_s0/model.pt` | 16 | 0 | 0 | `window_anchor_keep_mask(256,16,0)` | 13,536,768 |
| `win16_s1` | 复用 `2634/train/rt_none_s1/model.pt` | 16 | 0 | 1 | 同上 | 13,536,768 |
| `full` | 复用 `2610-frozen-trunk-jumphead-13m/trunk.pt` | — | — | 0 | `None`（全因果） | 13,536,768 |

- 复用臂**只做评测**（eval-only）；其等值性由「`win16_s0`/`win16_s1` 的 clean bpc 与
  `2634-carry-realtext-13m/results.json` 落盘值**逐位相同**」佐证（判据⑥）。
- `full` 是全因果锚：质量与代价的**上界参照**（不是本 WP 的干预臂，无掩码 ⇒ 不参与弃源探针）。

## 4 判据线（**预登记**：写下后不得改）

| # | 判据 | 判定线 | 该涨/该跌/必须不变 |
|---|---|---|---|
| ① | **质量腿（价格）**：锚点臂相对同 seed 纯窗臂的 clean bpc 差 | 报 `price_bpc = bpc(win16a) − bpc(win16)`（正 = 锚点是质量税）；可排序 = 越本格自测带（见 §5b）；0.014 跨格带**并列**报、**不混用** | 该涨（或如实报税） |
| ② | **锚点召回（主判据）**：`win16a_*` 的 `anchor` 桶（针植入 `p<8`）植入召回 **高于配对无针地板** | 逐臂：差值 95% CI 下界 > 0；两个 seed 都过才算该腿有区分力。CI 用 **Newcombe 两比例差** + **按针配对的 bootstrap** 两种并列报（等率类不跨量纲借用 0.014） | 该涨 |
| ③ | **负对照**：`beyond` 桶（窗外，`p≥8` 且有位置不可见）两臂召回都**停在地板** | 逐臂：植入 vs 地板差值的 95% CI **含 0**（不可排序） | 该跌（负对照） |
| ④ | **毁源敏感度（负对照，必须跑）**：锚点臂对「锚点槽**之外**的前缀 `[8,64)` 被毁」逐位不敏感 | 被评行 `[244,255)` 上 `max|Δlogit| == 0`（`eval_lo − rf ≥ corrupt_hi`，`rf = n_layers·(window−1) = 180`） | 必须不变 |
| ⑤ | **因果性**：只改后半段 token ⇒ 前半段 logits 逐位不变 | `max|Δlogit_prefix| == 0` | 必须不变 |
| ⑥ | **自检**：掩码非退化 + 复用臂 bpc 逐位复现 2634 | `visible_pairs(16,8)=5868`、`(16,0)=3976`（< 32896）；`win16_s0=1.628110`、`win16_s1=1.629093`（逐位） | 必须不变 |

**判据④的「如实报」义务（本 WP 的一处必要偏离，跑之前写死）**：锚点槽在 `[0,8)`，
而标准毁源布局的被毁前缀是 `[0,64)`——**锚点槽落在被毁区内部**。锚点臂能直接看到被毁的
锚点槽，所以它对 `[0,64)` **不可能逐位不敏感**（`max|Δlogit| > 0`）。这是**物理事实**，
不是仪器故障。因此判据④按**两变体**跑并全部如实报出：

- **变体 A · `far_all`**：被毁 `[0,64)`（= 2634 的原布局）。窗口臂预期 `max|Δlogit| == 0`；
  **锚点臂预期 `> 0`**（它直接看得到被毁的锚点槽）。此时仪器的「可见集不得与被毁区重叠」
  硬断言对锚点臂**必然触发** ⇒ 变体 A 对锚点臂以 `require_no_overlap=False` 运行，
  并把重叠事实记为 `premise_holds=False`（**不是**给它放行，而是把「不可判」写出来）。
- **变体 B · `beyond_anchor`**：被毁 `[8,64)`（**保住锚点槽 `[0,8)` 完整**）。两臂预期
  `max|Δlogit| == 0`——因为 `[8,64)` 既不在窗口可达区 `[i−180,i]`（`i≥244 ⇒ [64,244]`）、
  也不在锚点区 `[0,8)`。**这是判定「锚点算不算信息右移」的关键区分**：锚点臂对锚点槽
  **之外**的远处内容逐位不敏感 ⇒ 锚点**没有搬运**远处内容，只把最前面 ≤8 个位置变得恒可见。

**判据④的读法（预登记）**：若变体 B 上锚点臂 `max|Δlogit| == 0` 且变体 A 上 `> 0`，
则「锚点 = 少量恒可见槽（直接注意力），不是记忆通道」成立；若变体 B 上锚点臂 `> 0`，
则锚点真的把锚点槽之外的内容也搬运了（机制异常，先修仪器/实现）。

### 4a. 操作性定义（不改判据，只把口径落定）

- **clean bpc（每臂都在其自己的训练可见性下评测）**：`realtext_probe.win16_bpc(model, eval_arr,
  keep_mask, n_batches=96, batch_size=16, window_seed=2)`——与 2634 同一函数、同一窗口序、
  同一批数 ⇒ **复用臂（`win16_s0/s1`）应与 2634 落盘值逐位相同**（判据⑥）。锚点臂用
  `keep_mask = window_anchor_keep_mask(256,16,8)`；`full` 用 `keep_mask=None`。
- **`anchor` / `beyond` / `window` 桶（`retention.needle_fate`）**：针 v3（`phrase_needle_batch`，
  `n_needles=8, phrase_len=6, anchor_frac=0.25, seed=3`）。**归属桶一律用 `anchors=8`**
  （把 `p<8` 的针判为 `anchor`）：只有**所有臂用同一归属规则**，「锚点臂的 anchor 桶」与
  「纯窗臂的 anchor 桶」才是同一组索引（否则是索引集不同造成的假象）。`beyond` = 非锚针且
  **有任意一个位置不可见**（`all` 语义，半删即被删）。
- **地板（配对无针控制）**：同一 generator 序列、`plant=False` 的配对控制臂（不植入短语 ⇒
  只剩语料先验）；`run.py` 已断言主臂与控制臂的针位 `meta` 逐位相同。
- **同索引硬子集（`needle_stats.hard_index_sets`）**：跨臂比较一律在**同一组索引**上做
  （`intersection` = 五臂 control 全答不出；`union` = 至少一臂答不出）。
- **`σ_cell`（本格自测带）**：本实验**自己的**换种子极差 = `max(|bpc(win16a_s0)−bpc(win16a_s1)|,
  |bpc(win16_s0)−bpc(win16_s1)|)`（2 样本极差 ⇒ **描述量**，不是方差上界，显式标注）。两带
  并列报（本格带为主判、0.014 跨格带为参考），**不混用**（C1 裁决）。
- **阈值归属（可排序）**：bpc 类用本格带；召回等率类用 CI 下界是否 > 0。**不许**把 0.014 借给召回率。

### 4b. 代价口径（每位置等效 KV 字节，fp16）

`kv_bytes(a) = n_layers × 2 × d_model × 2 × k_eff = 12288 × k_eff`（`k_eff` = 该臂每位置可见的
键数；锚点臂 `k_eff = 16 + 8 = 24`）。⇒ `win16 = 196,608 B`、`win16a = 294,912 B`、
`full = 3,145,728 B`。**锚点的价格 = 98,304 B/位置**（= 8 × 12288）。

## 5 先例配方（SCHEMA §18：锚点 8 不是本 WP 首创）

- `2615-window-trained-retention-13m`：`window32` 用 `--train-window 32 --train-anchors 8`。
- `2616-window192-trained-13m`：`window192` 用 `--train-window 192 --train-anchors 8`。
- `2634-carry-realtext-13m`：`rt_none_s0/s1` 用 `--train-window 16 --train-anchors 0`
  （本 WP 的纯窗复用臂来源）。
- `--train-anchors` 是 `train_depth_ar.py` 的**已有公开 CLI**（`pathlm.depth_ar
  .window_anchor_keep_mask`；训练/评测共用同一函数）⇒ **本 WP 不需要也不允许改共享文件**。

## 5a. 从 2634 复制并改动的两处（显式登记）

1. **`needle_stats.py`**：逐字复制；**新增** `paired_bootstrap_diff_pp(planted, control, ...)`
   （按针配对的 bootstrap CI；2634 只有 Newcombe 两比例差）。原函数一字不改。
2. **`realtext_probe.py`**：语义层逐字复制 2634；`far_source_probe` **新增 `corrupt_lo`
   参数**（默认 0 = 2634 原布局；被毁区由 `[0,corrupt_hi)` 变为 `[corrupt_lo,corrupt_hi)`），
   并把「可见集与被毁区重叠」这条硬断言改由 `require_no_overlap` 控制（默认 True = 2634 行为；
   判据④变体 A 对锚点臂传 False 并记录 `premise_holds=False`）。`probe_kwargs` 同步带上
   `corrupt_lo`。`win16_bpc` / `channel_off` / `paired_cluster_bootstrap_ci` 一字不改。

## 5a 交付口径备注（不改 §4 判据线）

- 判据② 的「CI 用 Newcombe 两比例差」在交付实现里是**正态近似（未 pooling）**变体（`tools/evalkit/carry_channel.py::two_proportion`，`diff ± 1.96·√(p_aq_a/n_a + p_bq_b/n_b)`）。两法在本次被判的所有格子上**同号、同「含 0 / 排除 0」**，故 §4 的判定线**未变**；差异见 ticket 62 §10.28。
- 判据② 同时并列报了**配对 bootstrap**（逐针配对、口径与 §12.1 一致），其结论与上一条一致/更严的情况已在 `analysis.md` 逐格标出。

## 6 训练配方与命令原文

```
python3 train_depth_ar.py win16a_s0 --d 256 --layers 12 --heads 8 --mlp-mult 6 \
  --seq-len 256 --dense --topology sequential --steps 12000 --batch-size 32 --lr 1e-3 \
  --seed 0 --exit-mode uniform --train-window 16 --train-anchors 8 --eval-batches 4 \
  --out-root axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m/train
```
（`win16a_s1` 同命令、`--seed 1`；驱动脚本 `train_arms.sh` 按臂取/放 §20 锁，臂间 ≥61 s。）
`train_depth_ar.py` 是共享入口，**未改一字**。

**revision**：本实验写到 README 时的 `git rev-parse HEAD` = `110f9e29adfbbedc9b91fb47bd08c2052da5e4ab`
（`2026-10-14T20:54:54+08:00`）。注：本 WP 边界纪律禁止跑 git 命令，此处仅执行 §4 明文要求的
**只读** `git rev-parse HEAD`；不做任何暂存/提交/local 写。

## 7 成本（§8 唯一预算数字 + 砍什么）

`gpu_wall_budget: 30 min`（含步时冒烟 + 2 臂训练 + 评测 + 对账）；总体墙钟 ≤55 min。
**步时**：先跑 300 步冒烟实测，再决定是否砍。**预算不够时砍什么**（SCHEMA §17.7/§20.3）：
**两臂一起**砍步数（12k → 6k，保持臂间同 token 预算并在报告首段声明）> 砍 needle 批数
（64 → 32）> 砍毁源变体。**不砍判据线、不砍臂、不砍 seed 数**。墙钟/吞吐**不进判据**。

## 8 数据与脚本清单

- 数据：`data/enwik8_full.npz`（前 90M 字符训练 / 末 10M 评测，vocab 206 含 `[mask]`）；
  加载入口 `pathlm.data.load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")`。
- 训练：`train_arms.sh`（→ `train/win16a_s{0,1}/{model.pt,train_log.jsonl,gate_log.jsonl,results.json}`）。
- 评测：`make_config.py`（生成 `config.json`，含逐臂 path + sha256）→ `run.py`
  （质量腿 + 锚点召回腿 + 毁源两变体 + 因果性/掩码自检）→ `results.json` / `run_log.txt`。
- 落盘：`train/`、`config.json`、`results.json`（**严格 JSON**，禁裸 NaN/Infinity）、`run_log.txt`、
  `analysis.md`、`MODEL.md`、`smoke/`。权重 `*.pt` 不入库；`MODEL.md` 记 sha256 / 体积。

## 9 失败决策树（测量 → 实现 → 规模 → 设计）

1. **测量**：判据⑥（复用臂 bpc 不复现 2634 逐位值）⇒ 先查窗口序 / 掩码 / 批数，再判。
2. **实现**：锚点掩码活性为 0（`visible_pairs` 不增）⇒ 开关死了，先修。
3. **规模**：`anchor` 桶召回与地板不可排序 ⇒ 该口径/该尺度上锚点**没被用起来**；
   按 SCHEMA §19 只读作「不划算/不可判」，**不得**写成「机制不可能」；下一步 = 更大步数、
   锚点档位（4/16）扫描、训练期针注入——列成待测项与成本。
4. **设计**：`anchor` 桶召回显著 > 地板 但 bpc 变差 ⇒ 区分「召回改善但质量税抵消」。

## 10 scope 声明

本实验**只做单轴单格**（1.1.1），基底恒为 stock 对照；组合默认留白。唯一「组合」= 同一干预
（锚点掩码）与种子轴的重复测量（方差量化），不构成新结构组合。`deps` 只记录耦合方向。
组合取舍属具体产品方 scope。**禁外推**：结论只适用于 enwik8 / 窗口 16 + 锚点 8 / 13M / 12k 步快照。

## 11 产出与复核（SCHEMA §14/§14b）

见 §8；writer 停止写入后另派 **fresh、独立上下文**只读复核者（非本 WP writer），**复核排在
全部产物落盘之后**，返工 ≤2 轮。
