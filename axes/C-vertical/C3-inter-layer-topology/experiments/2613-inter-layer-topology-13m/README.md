---
name: 2613-inter-layer-topology-13m
ticket: 30-batch3（契约 E）
date: "2026-09-30"
grid: C3-inter-layer-topology（纯空白格，首次本地探索）
engine: depth-ar `--dense`（stock decoder-only transformer，13M 档）
budget: "gpu_wall_budget: 40 min（冒烟 ~1 + 四臂训练 ≈26 + 四臂电池 ≈3 + 余量）；峰值显存 ≤5GB（net）"
status: planned
---

# 2613 · 层间拓扑（C3 纯空白格首探）· 13M · depth-ar/dense

> 本文件即本实验的本地契约。第 0 节把 `.scratch/30-batch3/issues/01-batch3.md` 的
> 「契约 E」**逐字**抄下（判据线一字未改）；第 1 节起是实现与执行细节。
> 原始结果：`results.json`；判读：`analysis.md`；稳定性：`stability.md`；权重清单：`MODEL.md`。

## 0. 契约原文（逐字，judge 不得修改）

> 【契约 E】层间拓扑（C3 纯空白首次本地探索）
> 契约：`.scratch/30-batch3/issues/01-batch3.md` 的「契约 E」。
> 实验目录：`axes/C-vertical/C3-inter-layer-topology/experiments/2613-inter-layer-topology-13m/`
> 可改：`pathlm/depth_ar.py`、`train_depth_ar.py`、`tests/test_depth_ar.py`、本实验目录。
>
> 背景：这是「地图」上的纯空白格。层与层之间除了单条残差连接，还有别的接法（并行子层、对历史层的加权和）。目标：在**普通 decoder-only transformer** 上，只改层间连接方式这一个因素，看质量是否真的改善。
> 三臂：sequential（基线，现有 Block）/ parallel（h = h + attn(ln h) + mlp(ln h)）/ dense（每层输入 = 对全部此前层输出的学习加权和，初始化为「只取当前层」以等价起步，必须报层权活性）。
> 配置：d=256 / n_layers=8 / mlp_mult=4 / 12k 步 / bs=32 / seq=256 / seed=0。
> 判据：任一相对基线改善 ≥0.014 为成功，|Δ|<0.014 中性，≥+0.014 失败；并列 tf@4 ≥0.95；必须附三臂同方法 loss 斜率。
>
> 【ticket 统一纪律（逐字）】单因素；对照逐字段等值（含 seed）；干预活性（SCHEMA §10）；
> 三道对账（§12）；对照臂三要素（§13）；判据取向 = 与自循环/诚实对手比、oracle 只作上界（§11）；
> 增益归因须过等参数对照（§15）；快照增益须做收敛对账（§16）。

## 1. 命题（一句话、可证伪）

在普通 decoder-only transformer 上，**只把层与层之间的接法**从顺序链式残差换成
（a）并行子层或（b）对所有此前层输出的学习加权和，12k 步 matched-step 快照下的
clean bpc 相对顺序残差基线改善 **≥ 0.014**（否则该格在这一形态/规模上无头寸）。

## 2. 单因素声明

**唯一变量 = 层间接线**。四个臂全部是 stock decoder-only transformer
（`train_depth_ar.py --dense`：无 carry 门、无随机退出、仅最终深度 CE、tied 读出），
其余逐字段一致（对照表见 `analysis.md` §1 与 `results.json::field_by_field`）：
d=256 / L=8 / h=8 / seq=256 / mlp_mult=4 / dropout=0 / norm_cap=1 / bs=32 / lr=1e-3 /
12k 步 / seed=0 / 98.3M token / enwik8 前 90M train、末 10M eval / vocab 206。

| 臂 | 接线 | 相对基线参数量 Δ |
|---|---|---|
| `sequential` | `h = h + attn(ln1 h)`；`h = h + mlp(ln2 h)`（现有 `Block`） | 0（基线） |
| `parallel` | `h = h + attn(ln h) + mlp(ln h)`，**单个共享 pre-LN**（契约公式 / PaLM-GPT-J 原式） | **−L·2d = −4096** |
| `parallel-dual-ln` | 同并行接线，但两个子层各带自己的 pre-LN、都读同一块输入 `h` | **0（与基线逐位相同）** |
| `dense` | 层 k 输入 = `Σ_{j<k} α_kj · h_j`（DenseFormer 式），α 初值 = 只取 `j=k−1`（= 顺序残差，起步逐位等价基线） | **+L(L+1)/2 = +36** |

- **非必要性例外**：无。基底就是主流标准 decoder-only transformer；三条 RAP 共享底座
  偏离（tied 读出 + 无 final LN、变深度填充语义、全深度重打分）在本实验的四臂间相同，
  不作为变量。
- **`parallel-dual-ln` 是补充臂（不改变主判据）**：契约公式把 `parallel` 写成**单个**
  `ln`，那是 PaLM/GPT-J 原式，参数比基线少 4096（−0.06%）。本批特别要求核对
  「等参数对照是否真的等参数」（§15 / 契约 H 复核项 ①），而**并行接线的等参数形式
  不唯一**，因此额外跑一个「两子层各带 pre-LN、参数量逐位等于基线」的变体，把
  并行接线的收益与参数账解耦。三个变体都在同一判据线之下（「任一变体」）。

## 3. 指标表（逐项分析）

| 维度 | 指标 | 与谁比 |
|---|---|---|
| 解决什么问题 | 层间拓扑（顺序 / 并行 / 层间加权和）在 stock 形态上的质量头寸 | 命题本身 |
| 代价 | Δ clean bpc（13M / 12k / seed0 / 98.3M token，matched-step 快照） | `sequential` 基线 |
| 有效性 | Δ vs 基线 + 方差界（Δ<0.014 不可排序） | SCHEMA §12 判据线 |
| 收益·用法 1（质量） | 同引擎 dense 13M matched 税分母（1.5287）下的税 | 历史 dense-B-12k |
| 收益·用法 2（口径） | 逐深度 clean bpc 曲线 + next-token acc | 基线 |
| 收益·用法 3（出口资产） | tf@k 接受率（tf@4/tf@6）、等效验证加速（FLOPs 口径） | 基线；并列判据 tf@4 ≥0.95 |
| 可控性/校准 | prob0 ECE、逐深度 ECE | 基线 |
| 副产品（内存） | 推理期需保留的层输出数（sequential/parallel = 1；dense = 全部此前层） | 结构性代价，见 analysis §5 |
| **干预活性（必报）** | dense：层权 α 的 fro/absmax/非零占比/与 identity 起步的最大偏离；parallel：分支增量范数 + 接线相对偏差 | 0（死干预） |
| **稳定性（必报）** | gnorm max/中位数/尖峰、NaN、loss 突跳、逐深度 ECE | `stability.md` |

**判读规则**：先过「实现/接线」门（单测 + 冒烟 + 参数量账）与「干预活性」门；
两者不过 → 先修实现再判机制。

## 4. 判据线（预先写下，逐字来自契约 E，不得修改）

- 主判据：**任一变体相对 `sequential` 的 clean bpc 改善 ≥ 0.014 ⇒ 成功**；
  `|Δ| < 0.014` ⇒ 中性；`Δ ≥ +0.014` ⇒ 失败（更差）。
- 并列：**`tf@4 ≥ 0.95`**（不破坏出口资产）。
- **必须附**：各臂**同方法**（末 25% 窗、同一最小二乘算法）loss 斜率，作为
  「是否只是收敛差异」的收敛对账（SCHEMA §16）。

方差界参照：Δ < 0.014 不可排序（`docs/methodology.md` §2）；判据阈值即方差界。

## 5. 对照与方差

- 四臂**自包含双跑**（同一脚本、同一代码、同一 seed、同一数据切分、同一评测电池），
  不复用历史 run 作判据臂 —— 不引入跨 run 混杂（SCHEMA §9）。历史 `23d-dense-B-12k`
  （同配置 sequential 接线）只作**交叉核对**（当前代码重测，须复现 1.5287）。
- **seed 数**：1 seed（契约未授权补 seed；Δ<0.014 不可排序已按此解读）。
- 方差界：0.014。

## 6. 成本与 scope

| 项 | 值 |
|---|---|
| `gpu_wall_budget` | **40 min**（冒烟 ~1.5 + 四臂训练 ≈26 + 四臂电池 ≈3 + 余量） |
| 最小可判定版本 | 13M / 12k 步 / 1 seed / clean bpc + tf 表 + 活性 + 稳定性（四臂） |
| 峰值显存 | ≤5GB net（实测见 `results.json::gpu_peak_by_arm_gb`；共享卡已占 8–12GB） |
| 可中断性 | 是（每 200 步落日志；`--skip-train` 可用已有 checkpoint 重算电池） |
| 冒烟计划 | `smoke.py`：参数量账 + 初始化逐位一致 + 恒等起步 + 三个非基线臂各 20 步 CLI 训练（接线/活性/步时/显存） |
| 预算不够时砍什么 | 砍 `--eval-batches`（20→8）、砍 `parallel-dual-ln` 补充臂；**不砍判据线** |

规模纪律（SCHEMA §7）：单卡 RTX 4090 24GB 且被多任务共享，GPU 任务串行；
只做 13M 原理验证，不外推规模/渐近。

## 7. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | char 级 enwik8（`data/enwik8_full.npz`，本地已有，不下载；前 90M train / 末 10M eval / 词表 206） |
| 入口脚本 | `run.py`（四臂训练 + 同口径电池 + 对账 + 判读 + 稳定性）；冒烟 `smoke.py` |
| 代码改动 | `pathlm/depth_ar.py`（新增 `topology` 配置 + `ParallelBlock` + `topology_liveness` + dense 层权 `layer_mix`；缺省 `sequential` 逐位还原 v1）、`train_depth_ar.py`（`--topology` + 落盘活性块）、`tests/test_depth_ar.py`（+9 可失败单测） |
| 可失败单测 | ① 缺省拓扑逐位还原 v1；② 并行公式逐位钉死 + 非 stock 路径守卫；③ 参数量账 + 初始化逐位一致（等参数臂）；④ 接线活性（并行 ≠ 顺序；两种并行变体可分）；⑤ dense 恒等起步逐位 == sequential + 改 α 必偏离；⑥ dense 层权公式手算对照；⑦ dense 层权非鞍点/一步逃逸；⑧ 拓扑守卫（需 dense、未知值、`_gated_step` 拒绝）；⑨ 活性块 `params_added` 与实测参数量增量逐位一致 |
| 变异验证 | 三处故意破坏（忽略拓扑开关 / 并行实现成链式 / dense 层权不参与前向）必须抓红——见 `analysis.md` §5 |
| 落盘 | `results.json` / `train_log.jsonl`（每臂） / `config.json` / `analysis.md` / `stability.md` / `MODEL.md` / `smoke/smoke.json` / `mutation_report.json` / `rederive_liveness.py`（§10 活性块字段名修正后的等价重算，幂等） |

## 8. 失败决策树

| 症状 | 诊断顺序 | 处置 |
|---|---|---|
| 当前代码不能复现历史 1.5287 / 单测红 | ① 实现 bug | 先修实现 + 跑变异验证；未排除前不判机制 |
| 干预死（dense α 停在 identity / 并行分支增量为 0） | ① 实现 | 检查 init 与接线；活性门不过不判机制 |
| 参数量账不符（等参数臂不等参数） | ① 实现 | 修到逐位相同再跑 |
| 全部变体 \|Δ\| < 0.014 | ② 该形态/规模无头寸 | 如实报中性；下一步换尺度/换判据（**不加第二个机制**） |
| Δ 落在 [0.014, 0.02) 且斜率差大 | ③ 收敛速度 vs 质量 | 报「快照口径 + 疑为收敛加速」，标 n=1；不裁决为质量结论 |
| 变体更差 ≥ +0.014 | ④ 设计在该设定下无效 | 记为否证，写清「在什么条件下才可能有效」 |
| 结果不可重复 | ⑤ 方差 | 补 seed；仍不可排序则撤回 |

## 9. 产出

`axes/C-vertical/C3-inter-layer-topology/experiments/2613-inter-layer-topology-13m/`
下的 `README.md`(本文件) / `run.py` / `smoke.py` / `mutate.py` / `rederive_liveness.py` /
`config.json` / `results.json` / `mutation_report.json` / `train_log.jsonl`（四臂） /
`analysis.md` / `stability.md` / `MODEL.md`。
