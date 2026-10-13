---
name: 2613-self-gated-composition-13m
ticket: 30-batch3
date: "2026-10-01"
grid: H3-composition
engine: 冻结 13.5M dense 主干（2610 trunk.pt）+ 只训组合头
budget: "gpu_wall_budget: 20 min（含冒烟 + 全跑 + 评测）；实测 6.7 min GPU / 峰值显存 335 MiB net"
status: judged
---

# 2613 · 自门控多跳组合（契约 30-G，H3 纯空白格）· 13M

## 0. 契约原文（逐字保留，判据线不许改）

> 【契约 G】自门控多跳组合（H3，用户提出的设计，纯空白）
>
> - 目录：`axes/6-predict/6.2-speculative-decoding/experiments/2613-self-gated-composition-13m/`
> - 文件所有权：可改 `train_jumphead.py`、`tests/test_jumphead.py`、本实验目录。
> - **冻结主干**：复用 `axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt`
>   （13.5M dense，12k 步，bpc 1.4836，t+1 acc 0.6979；核对 MODEL.md 的 sha256）。**只训头**。
> - **干预（唯一变量 = 组合结构）**，三个臂，**参数量对齐**：
>   - `direct`：一个 d→d 变换 T 作用一次 → 读出得 t+2；
>   - `composed`：**同一个 T 作用两次**（T∘T）→ 读出得 t+2（即「跳 1 走两次」，与 direct 参数完全相同）；
>   - `gated`：`h_mix = g ⊙ (T∘T)(h) + (1−g) ⊙ T(h)`，`g = sigmoid(G(h))` 逐通道；
>     G 单边零初始化（起步等价 direct 或 composed，须活性判据）。
> - **判据（预登记）**：
>   - 主判据：`gated` 的 t+2 命中（vs 真值）**≥ max(direct, composed) + 0.014** ⇒ 自门控组合有效；
>   - 并列（用户取向）：**草稿接受率**（= 头 argmax == 主干自身 argmax）≥ 0.90 ⇒ 可作草稿；
>   - 同时报 `oracle`（上界，不作判据）；
>   - 附加洞见：报 g 的分布（模型偏好一跳还是两跳）与逐桶命中率。
> - 成本：头训练 3 臂 × ≈2–5 min + 评测 ≈3 min ⇒ ≈20 min。
> - 交付：同统一清单。
>
> 统一纪律：单因素；对照逐字段等值（含 seed）；干预活性（SCHEMA §10）；
> 三道对账（§12）；对照臂三要素（§13）；判据取向 = 与自循环/诚实对手比、oracle 只作上界（§11）；
> 增益归因须过等参数对照（§15）；快照增益须做收敛对账（§16）。

**本地化说明（不改变判据线）**：
1. 契约的「三臂」按字面实现；额外加一个 **`direct_eq`（§15 等参数对照臂）**：MLP 隐层 3d + 逐通道偏置，
   参数量与 `gated` **逐位相同**（394,496）、但没有门、没有组合。它不是判据线的一部分，
   只用来回答「增益来自门控结构还是来自参数」。
2. 契约写「G 单边零初始化（起步等价 direct 或 composed）」——这两条在 sigmoid 门下不相容
   （σ(0)=0.5）。本实验按 **SCHEMA §10 的硬纪律取「单边零」**（输出侧 `fc2=0`、输入侧 `fc1~N(0,0.02)`），
   因此**起点是 direct 与 composed 的 50/50 混合，不是任一臂的逐位等价**。这是已知偏差，见 §4 与
   `analysis.md`；活性判据（g 分布、G 的范数）照报。

## 1. 命题

在同一个**冻结** 13.5M dense 主干表征上，预测 t+2 的三种组合结构——直接跳两步（T）、
跳 1 走两次（T∘T）、以及由逐通道 sigmoid 门在两者间插值——能否让**自门控组合**显著优于
两条固定路线？判据线预先写下：gated 的 t+2 命中 ≥ max(direct, composed) + 0.014。

> **命题边界**：本实验测的是**冻结表征上的重复潜在映射（T∘T）与门控**，**不回答**「先出 N+1 再以之为条件出 N+2 vs 直接出 N+2 vs 概率混合」；该命题由 **`2631-spec-three-arms`**（`../2631-spec-three-arms/analysis.md`）回答（结论：direct ≈ combined > chain）。

## 2. 单因素声明

唯一变量 = **组合结构**（T 作用一次 / 同一 T 作用两次 / 逐通道门控混合 / 等参数无门对照）。
主干、数据切分、位置集、批大小、步数、lr、seed、评测协议逐字段一致（§5 逐字段对照表）。
基底 = 主流标准 decoder-only transformer（`DepthARModel(dense=True)`）。
**非必要性例外**：无。头是冻结主干外的旁路，主干未被改动（bpc 逐位不变，`tests/test_jumphead.py` 钉住）。

## 3. 指标表（多维，逐项分析）

| 维度 | 指标 | 与谁比 | 结果 |
|---|---|---|---|
| 它解决什么问题 | 多跳预测的结构选择（一跳走两次 vs 直接跳两步） | — | 结构选择本身不给可排序增益 |
| 代价 | 参数量 / FLOPs | direct / composed 逐位相同；gated 多一个门（+0.13M） | 见 §5 |
| 有效性 | t+2 命中（argmax == 真值） | max(direct, composed) | +0.0020（2k）/ +0.0042（4k），< 0.014 ⇒ 中性 |
| 收益·用法 1（草稿） | 草稿接受率 = P(头 argmax == 主干自身 AR-1 argmax) | 主干自身（无真值参与） | 0.6113 < 0.90 ⇒ 不可作草稿 |
| 收益·用法 2（上界参考） | oracle（喂真 t+1）命中 | 上界，不作判据 | 0.6953 |
| 可控性 | 门开度 g 的分布 + 逐桶命中 + 选择性 | — | g 均值 0.82（强烈偏好两跳），逐桶命中单调（0.17 → 0.68） |
| 副产品 | 门控臂的两条分支各自命中（哪条被训练出来） | 单臂 direct | gated 的 direct 分支只有 0.262（伙伴分支 0.464）⇒ 门「选了一条、荒废了另一条」 |

## 4. 判据线（预先写下，逐字保留）

- 主判据：`gated` 的 t+2 命中 ≥ max(direct, composed) + 0.014 ⇒ 自门控组合有效；
- 并列：草稿接受率（= 头 argmax == 主干自身 argmax）≥ 0.90 ⇒ 可作草稿；
- oracle 仅作上界，不作判据；
- 附加：报 g 的分布（模型偏好一跳还是两跳）与逐桶命中率。

**判决口径**（照抄仓库方差界，不改线）：Δ ≥ +0.014 = 成功；|Δ| < 0.014 = 中性（不可排序）；
Δ ≤ −0.014 = 失败。判据线只有 README 这一处，代码 `compose_verdict()` 只做归类。

**口径说明：0.014（= 1.4pp）在本格是预登记的实用阈值，不是已测的准确率方差
（准确率与 bpc 是两个不同随机变量）；准确率的可排序性用二项 / Wilson 95% CI
（见 `docs/methodology.md` §2）。四臂差 ≪ 0.014。**

## 5. 对照与方差

- **direct / composed**：参数量逐位相同（262,912），只差「T 作用一次还是两次」——这是组合结构的等参数对照。
- **direct_eq**（§15 附加）：394,496 = gated 逐位相同，去掉门与组合。
- seed：头训练 2 seeds（0/1），跨 seed 极差 ≤ 0.0017；**阈值 Δ < 0.014 不可排序（预登记实用阈值，非已测准确率方差）**。
- 诚实对手（§11/§13）：主干自身 1-step AR（`ar1_deploy`，只替换紧邻位置 i+1）——接受率的分母是它，
  不是真值；oracle 只作上界。
- 复用（§9）：复用 2610 的 **主干权重**（sha256 已核对），不重训主干；头是本实验自训的。
  与 2610 的差异字段在 §7 逐条列出。

## 6. 成本与 scope

| 项 | 值 |
|---|---|
| gpu_wall_budget | **20 min**（含冒烟 + 全跑 + 评测；契约原文） |
| 实测 | 6.66 min GPU（12 次头训练 + 12 次评测），峰值显存 334.8 MiB net（冻结主干，共享卡安全） |
| 最小可判定版本 | 4 臂 × 1 seed × 2000 步 + 1 次评测（≈2.5 min） |
| 可中断性 | 是（每臂单独存盘；日志按步追加） |
| 冒烟计划 | 20 步 × 4 臂 × 2 seed + 2 批评测，实测步时与显存后再全跑（已执行） |
| 预算不够时砍什么 | 砍 2× 收敛点 → 砍 seed 1 → 砍 direct_eq；**不砍判据线** |

scope（写入本仓库的每个实验）：本研究**只做单轴单格实验**，基底恒为 stock 对照；
**不做组合实验**——组合留白是显式声明，不是遗漏。`deps` 只记录耦合方向。

## 7. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | char 级 enwik8（`data/enwik8_full.npz`，前 90M 训练 / 末 10M 评测），未下载新数据 |
| 冻结主干 | `../../H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt`（sha256 见 `MODEL.md`，运行时会自动核对） |
| 入口脚本 | `run.py`（薄封装）→ 仓库根 `train_jumphead.py --composition` |
| resolved config | `config.json`（含 seed / 数据切分 / 头超参 / 收敛点） |
| 可失败单测 | `tests/test_jumphead.py`（§30-G 段：T∘T 语义、混合公式、单边零起步+逃逸、变异鞍点、等参数对照、评测管线） |
| 原始产出 | `results.json` / `compose_log_*.jsonl`（每 200 步一点）/ `run.log` |
| 权重 | 不入库；`MODEL.md` 给哈希、体积、生成命令 |
| 复现命令 | `python3 axes/6-predict/6.2-speculative-decoding/experiments/2613-self-gated-composition-13m/run.py`（≈7 min GPU） |

## 8. 失败决策树（结果不好时按此顺序排除）

| 症状 | 诊断顺序 | 处置 |
|---|---|---|
| gated ≈ direct | 测量问题（门是死的 / 未激活） | 先看 `intervention.gated_g_at_eval`：g 是否偏离 0.5、fc2 是否逃逸（本实验 g 均值 0.82 ⇒ 门活着） |
| 三臂都 ≈ 0.47，互相差 < 0.014 | 结构在此规模/此主干上无头寸 | 换判据（成本口径）或换规模，不叠加第二个机制（本实验结论） |
| 命中 ≈ 0.5 或近随机 | 实现 bug（shift 方向 / 混合公式） | 变异验证 + 单测（`test_shift2_targets_are_t_plus_2`、`test_gated_mix_formula_is_per_channel_sigmoid_of_G`） |
| gated 高但 direct_eq 也高 | 增益来自参数而非结构 | 按 §15 归因于容量，不得写成门控的功劳（本实验属此列） |
| 主干 bpc 变了 | 冻结失败 | `assert_frozen` 断言（本实验 ok=True） |

## 9. 产出

`README.md`（本文件）/ `run.py` / `config.json` / `results.json` / `analysis.md` / `MODEL.md` /
`compose_log_*.jsonl` / `run.log`。
