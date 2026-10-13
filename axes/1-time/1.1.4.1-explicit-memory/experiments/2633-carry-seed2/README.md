# 2633 · 写入规则三臂 × 种子 2 补点（把「delta vs plain」判到 n=3）· 13M · dense + 窗口 16 · 合成任务

> 契约来源：主线一（信息右移）工作项 WI-01 的**第三次种子补点**；判据线**逐字**沿用
> `../2631-carry-reseed/README.md` §4 的预登记判据（②③④⑤），**未改一字**，只把种子上限
> 从 2 提到 3。
> 入口：`smoke_steptime.sh`（先测步时）→ `train_seed2.sh`（3 臂 × seed=2，按臂释放锁）→
> `run.py`（评测 + 三 seed 合并）；语义/统计层**复用** `../2631-carry-reseed/reseed_arms.py`
> （只读导入，同一函数、同一口径；详见 §9 的复用证据）；原始结果 `results.json`；判读
> `analysis.md`；权重清单 `MODEL.md`。

## 1. 命题

**「delta 写（只搬上次没搬的）一贯优于 plain 写（累加写）」这条判据（2631 判据③）
在 n=2 时不成立，但它不成立的原因尚不能定量：plain 在两跑上呈双稳态（36.11% / 1.91%，
极差 34.2pp），n=2 给不出方差，也判不出「再跑几个种子才能判」。**

一句话可证伪的形式：*在 2631 的同一装置（同配方、同任务、同评测腿）上补第三个训练种子
（seed=2），三臂（无通道 / plain 写 / delta 写）各一新跑；若 `delta − plain` 在**三个种子
的每一个**上都 ≥ +3pp 且配对聚类 95% CI 下界 > 0，则「delta 一贯优于 plain」记为跨种子
稳健；否则记为**不成立**，并给出 plain 跨种子散布的定量估计与「要判出该效应还需要多少
种子」的样本量计算。*

## 2. 单因素声明

- **基底**：stock dense decoder-only trunk（`DepthARModel(dense=True,
  topology="sequential")`：tied 读出 + 无 final LayerNorm + 顺序残差），
  13.54M（d256 × 12 层 × 8 头 × mlp6 × T256），与 2618/2631 **逐字段相同**。
- **唯一变量**：① `carry_rule` ∈ {无通道 `none` / `plain` 写 / `delta` 写}（后两者**逐位
  等参数**）；② 训练 `seed` = 2（本实验新增的一档；seed 0/1 复用 2631）。
- 其余逐字段一致：模型规格、训练配方、数据生成器、可见性掩码、步数、批大小、学习率、
  优化器、评测协议。逐字段对照见 `analysis.md` §3 与 `config.json::equality_vs_2631`。
- **必要性例外（显式记账）**：训练与评测都使用 `train_window=16 / train_anchors=0`
  的可见性掩码（既有开关，非第二个机制）——它制造「必须搬」的需求，三臂完全相同。

## 3. 干预定义

与 2631 完全同一实现（**本实验不改任何 `pathlm/**`、`train_depth_ar.py`、`tools/**`、
`tests/**` 文件**，只新增目录内代码）：

```
写入（每位置一次，来自输入流 h0）：k̂ = normalize(W_k h0)，v = W_v h0，β = σ(W_β h0 + b_β)
  plain： S_t = S_{t−1} + β_t v_t k̂_tᵀ
  delta： S_t = S_{t−1}(I − β_t k̂_t k̂_tᵀ) + β_t v_t k̂_tᵀ = S + β(v − S k̂)k̂ᵀ
读出（每层各一次）： a ← a_SDPA(窗口掩码) + γ_ℓ ⊙ mem_t ，mem_t = S_t q_t（读后写）
```

恒等起步（γ 单边零初始化）、参数量 +264,296（13,536,768 → 13,801,064）。

## 4. 判据线（**逐字**沿用 2631 契约 §4 / §4a 的预登记判据，未改）

> ① 主腿 = **分布内**多查询召回（Wilson CI + 配对/聚类 bootstrap，不得把每个查询当独立
> 二项样本）；② 三臂跨种子的**极差与方向一致性**；③ delta 是否仍最好（若不是，如实推翻）；
> ④ 毁源对照必须有效；⑤ 禁外推。
> ① 的判定线：`Δ(delta − none)` 聚类 95% CI 下界 > 0 且点估计 ≥ +5pp；
> ③ 的判定线：`Δ(delta − plain)` 聚类 95% CI 下界 > 0 且点估计 ≥ +3pp；
> ④ 的判定线：主臂聚类 CI 下界 > 4×chance = 6.25% 的臂，毁源后必须 ≤ 6.25%，否则作废。

### 4a. 本实验新增的**并列**读数（不改上面任何判定线）

- **⑥ 样本量估计**（任务下发文本要求）：「若不可判，给出需要多少样本才能判」——
  用 plain 臂跨 3 种子的样本 sd（n−1）作效应尺度的估计，给「检测到 Δ=3pp（判据③的判定线）
  所需种子数（80% 功效、α=0.05 双侧、t 近似）」；并注明该估计建立在「plain 跨种子近似
  正态」这一**未验证假设**上（n=3 不足以验证），只作规划用，**不得**当日志外结论。

## 5. 七臂规格

| 臂 | seed | `carry-write` | 来源 | 参数量 |
|---|---|---|---|---|
| `none_s0` / `plain_s0` / `delta_s0` | 0 | `none` / `plain` / `delta` | **复用 2631**（§9） | 13,536,768 / 13,801,064 / 13,801,064 |
| `none_s1` / `plain_s1` / `delta_s1` | 1 | 同上 | **复用 2631**（§9） | 同上 |
| `none_s2` | 2 | `none` | **本实验新训** | 13,536,768 |
| `plain_s2` | 2 | `plain` | **本实验新训** | 13,801,064 |
| `delta_s2` | 2 | `delta` | **本实验新训** | 13,801,064 |

数据 = 多查询关联回忆（`assoc-mqar` K=8 / Q=8 / gap=14，训练与评测同一生成器）；
12k 步 × bs32 × T256 = 98.3M token / 臂，七臂同 token 预算。

## 6. 成本（§8 的唯一预算数字 + 砍什么）

`gpu_wall_budget: 50 min`（含步时冒烟 + 三臂新训 + 评测 + 对账；含共享卡争用余量）。

**步时实测（本实验 `smoke/steps300`，与 2631 同款 300 步冒烟；稳态窗 `step 200→299`）**：
见 `analysis.md` §7 的逐臂表（原始日志在 `smoke/steps300/<arm>/train_log.jsonl`）。
12k 步估算是「本实验实测 ms/step × 12000」。

**预算不够时砍什么**（SCHEMA §17.7）：只砍**评测批数**（辅助腿/K 敏感性）；
**不砍判据线、不砍臂、不砍种子数、不砍 12k 步**（砍步数会破坏与 2631 复用臂的
token 预算一致性 ⇒ 违反 §9 的配置等值）。

## 7. 数据与脚本清单

- 数据：**不用 enwik8**。训练与评测都用 `tools/evalkit/longrange_tasks.py` 的合成生成器
  `assoc_recall_multi_batch`（K=8/Q=8/gap=14），生成器种子与模型种子无关 ⇒ 七臂配对可比。
- 训练：`python3 train_depth_ar.py <arm> --d 256 --layers 12 --heads 8 --mlp-mult 6
  --seq-len 256 --dense --topology sequential --steps 12000 --batch-size 32 --lr 1e-3
  --seed 2 --exit-mode uniform --train-window 16 --train-anchors 0 --synth-task assoc-mqar
  --synth-pairs 8 --synth-queries 8 --synth-gap 14 --carry-write {none,plain,delta}
  --out-root train`（驱动脚本 `train_seed2.sh`，按臂释放锁）。
- 评测：`python3 run.py`（复用 2631 的主腿/毁源/统计口径；合并七臂）。
- 落盘：`train/<arm>/{train_log.jsonl,gate_log.jsonl,results.json,model.pt}`（新臂）、
  `results.json`、`analysis.md`、`MODEL.md`、`reconcile_report.json`、`smoke/`。
- 权重 `*.pt` 不入库；`MODEL.md` 记 sha256 / 体积。

## 8. 既有做法（prior art，SCHEMA §18）

见 `../2631-carry-reseed/README.md` §8 与 `../2617-prior-art-memory/PRIOR_ART.md`
（MQAR / DeltaNet delta rule / Gated DeltaNet；`记忆·未核` 条目按原文标注）。本实验的
增量**不是**提出机制，而是**把一条已报告的机制排序（delta 优于累加写）放到多种子上
量化**：业界报告多为单种子，本线问的是该排序在 seed 方向上的稳定性，以及需要多少种子
才能判定它。**本地已知反证**：2618 首验的 plain 臂双稳态（0.07% / 3.34%）、2631 的
plain 跨种子极差 34.2pp。

## 9. 对照复用（SCHEMA §9 的四项证据）

复用 2631 的 seed0/seed1 六臂作对照，逐条满足 §9：

1. **实现层 equality**：本次**未改**任何引擎/仪器文件，预登记时的判定是「与 2631 运行时是
   **同一份代码**」。逐文件 sha256（本文件写入时记录）：
   `pathlm/depth_ar.py` `45f94679…c585`、`train_depth_ar.py` `2d1863bb…1f35`、
   `tools/evalkit/longrange_tasks.py` `4a02b93f…465b`、`tools/evalkit/carry_channel.py`
   `4dcd91fa…6753`——登记时刻与 2631 运行时相同（2631 提交 `a25261b` 之后**只有**
   `tools/evalkit/kv_quant.py` 变动，本线不导入它）。
   > **当前判读以 `analysis.md` §9 为准**（本节是实现层口径的**预登记**原文）：此后**其它并行线**
   > 对 `pathlm/depth_ar.py` / `train_depth_ar.py` 做了**默认关闭的加性改动**并入库
   > ⇒ 工作树的**字节**已不再等于 2631 运行时记录（`equality_report.json::working_tree_drift_files`、
   > `bytes_unchanged_since_2631_run = false`），§9 的判据因此落在**行为等价**上：
   > 基准 = 「2631 运行时版本」（按记录 sha256 定位到 commit `9f125cf`），与当前工作树比
   > **前向三档逐位 logits** + **训练脚本 1,001 步（6 个日志点）逐字段**，见
   > `engine_equivalence.json`。
2. **配置层 equality**：`config.json::equality_vs_2631` 逐字段对照 2631 的
   `config.json`+`train_arms.sh`（d/layers/heads/mlp/seq/batch/steps/lr/优化器/seed/token 预算/
   数据生成器/可见性掩码/退火），不等字段 = **无**（除 `seed`，这正是本实验新增的档）。
3. **口径标注**：`analysis.md` 首段标注「seed0/1 为 2631 的历史同配置对照（同一实现
   sha256 + 同 token 预算），seed2 为本次新训；三者同快照口径（12k 步 / 98.3M token）」。
4. **任一字段不等 ⇒ 回退自包含双跑**：若 §2 的逐字段对照出现不等，则**放弃复用**，
   重跑该臂（本实验的 `train_seed2.sh` 驱动可直接跑任意 seed）。

## 10. 失败决策树（测量 → 实现 → 规模 → 设计）

1. **测量**：`none_s2` 显著高于 chance ⇒ 查窗口掩码活性（可见位置数、局部先验读数）后重判。
2. **实现**：通道活性为 0（`read_share`/β/状态范数任一为 0）⇒ 先跑既有变异验证（2631 的
   M1–M8）排除死开关与未来泄漏。
3. **规模**：三臂都 ≈chance ⇒ 该任务/该尺度上「不划算」（SCHEMA §19：不得表述为「机制不可能」）。
4. **设计**：某臂主臂高、毁源后不掉 ⇒ 该臂读数作废（判据④）。

## 11. scope 声明

本研究**只做单轴单格实验**，基底恒为 stock 对照；**组合默认留白**（显式声明，不是遗漏）。
本线唯一的「组合」= 同一干预（写入规则）与**种子轴**的重复测量（方差量化），不构成新的
结构组合。`deps` 只记录已知耦合方向。组合取舍属具体产品方 scope。

## 12. 产出与复核（SCHEMA §14/§14b）

见 §7；本实验完成后另派 fresh-context 只读复核者，**复核排在全部产物落盘之后**。
