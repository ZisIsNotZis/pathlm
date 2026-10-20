---
name: 2651-inter-layer-topology-s1-13m
ticket: ticket 61 §9.2 WP-2651（2.2.3 + 2.2.4）
date: "2026-10-14"
grid: 2.2.3-层间学习加权和（DenseFormer 式）· 2.2.4-并行子层（一实验收两项）
engine: depth-ar `--dense`（stock decoder-only transformer，13M 档）
budget: "gpu_wall_budget: 45 min（冒烟 ~1 + 四臂训练 ≈27 + 四臂电池 + 余量）；峰值显存 ≤5GB net"
status: judged
---

# 2651 · 层间拓扑 seed=1 复核 + 并行子层代价面 · 13M · depth-ar/dense

> 本文件即本实验的本地契约。§4 的判据线**逐字**来自 `.scratch/61-axis2-depth/TICKET.md`
> §9.2 WP-2651 的预登记 acceptance（`date` = 该契约预登记时间；本轮执行不得事后调整）。
> 原始结果：`results.json`；判读：`analysis.md`；权重清单：`MODEL.md`。

## 0. 契约原文（ticket 61 §9.2 WP-2651，逐字，judge 不得修改）

> #### WP-2651 · 2.2.3 + 2.2.4 · 层间拓扑 n≥2 + 并行子层代价面
>
> - **落点**：`axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/`
>   （2.2.4 的证据指针指向同一目录）。
> - **what**：把 2613 的四个层间拓扑臂在 **seed=1** 上重跑（其余逐字段相同），并在同一目录内
>   给出**并行子层的代价面**（层内串行依赖长度、每层 FLOPs、参数量）与质量面同框。
> - **means**：`train_depth_ar.py --d 256 --layers 8 --heads 8 --seq-len 256 --mlp-mult 4
>   --dense --topology {sequential,parallel,parallel-dual-ln,dense} --steps 12000
>   --batch-size 32 --seed 1 --exit-mode uniform --eval-batches 20`；配置必须与 2613 s0
>   逐字段相同（只 seed 不同）。
> - **acceptance（预登记）**：①四臂齐备，`clean_bpc` + `tf@4` + 参数量落 `results.json`；
>   ②每臂活性（dense 的层权 norm 偏离恒等起步 >0；parallel 臂与 sequential 逐位不同的
>   前向已由单测钉住）；③2.2.3 判词：s0/s1 两 seed 的 **dense−sequential** 配对 Δ 是否越
>   0.014（不越 ⇒「无差异」可下；两 seed 同号且越带 ⇒「可排序」）；④2.2.4 判词：
>   parallel / parallel-dual-ln 相对 sequential 的质量代价（>0.014 即代价可排序）**与**
>   代价面同框（层内串行依赖 2 段→1 段；参数量 −2d 与逐位等参对照两臂并列）；
>   ⑤报「质量代价是否换到可兑现的等效收益」，等价于回答「并行子层在本尺度是否划算」。
> - **evidence**：上述实验目录（`results.json` / `analysis.md` / `MODEL.md` / `run_log.txt`）
>   + 2613 目录（s0 对照）。
> - **gpu_wall_budget**：45 min（含冒烟 + 4 臂 × 12k 步 + 评测）。
> - **砍什么（预算不够时）**：先砍 dense-s1（用 2613 s0 dense 作 n=1 记录并如实降级结论），
>   **不砍判据线**。

统一纪律（ticket 61 §9.1，逐字要点）：判据只用**等效理论算力**（layer-passes/token、FLOPs、
tokens-per-forward，考虑 pipelining），**墙钟/吞吐不得进判据或 `cost_model`**；bpc 类用
**0.014 bpc 带**（限配对 Δ、≥5 窗口）；对照臂必须自持（本实验自带 stock `sequential` 对照）；
干预臂必须有非零活性量并落盘；`run.py` 一条命令复现全臂；GPU 按 `harness/SCHEMA.md` §20 取锁、
§20.1 按臂释放。

## 1. 命题（一句话、可证伪）

在普通 decoder-only transformer 上，只把层与层之间的接法换成（a）并行子层或（b）对所有
此前层输出的学习加权和，在 13M / 12k 步 matched-step 快照下：2.2.3 问 **dense − sequential**
的质量差是否在 **seed=0（2613）与 seed=1（本实验）两个 seed 上一致地越 0.014 bpc 带**
（越 ⇒ 可排序；不越 ⇒ 无差异）；2.2.4 问 **parallel / parallel-dual-ln − sequential** 的质量
代价是否越 0.014 带，并在同一帧内给出层内串行依赖 / 每层 FLOPs / 参数量三条代价列。

## 2. 单因素声明

- **臂间唯一变量 = 层间接线**（`topology`），四臂逐字段一致（`results.json::field_by_field`）。
- **本实验相对 2613 的唯一变量 = 训练 seed（0 → 1）**，用于把 2.2.3/2.2.4 的 n=1 升到 n=2
  （`config.json::vs_2613` 逐字段对照；唯一不等字段 = `seed` 与实验目录名）。
- 基底 = 主流标准 decoder-only transformer（`train_depth_ar.py --dense`：无 carry 门、无随机
  退出、仅最终深度 CE、tied 读出）。三条共享底座偏离（tied 读出 + 无 final LN、变深度填充
  语义、全深度重打分）在四臂间相同，不作变量。
- **非必要性例外**：无。

| 臂 | 接线 | 相对基线参数量 Δ | 层内串行依赖 |
|---|---|---|---|
| `sequential` | `h = h + attn(ln1 h)`；`h = h + mlp(ln2 h)`（现有 `Block`） | 0（基线） | 2 段（attn → MLP） |
| `parallel` | `h = h + attn(ln h) + mlp(ln h)`，**单个共享 pre-LN**（PaLM/GPT-J 原式） | **−L·2d = −4096** | **1 段**（同读块输入） |
| `parallel-dual-ln` | 同并行接线，两子层各带自己的 pre-LN、都读同一块输入 `h` | **0（与基线逐位相同）** | **1 段** |
| `dense` | 层 k 输入 = `Σ_{j<k} α_kj·h_j`（DenseFormer 式），α 初值 = 只取 `j=k−1`（起步逐位等价基线） | **+L(L+1)/2 = +36** | 2 段 |

- `parallel-dual-ln` 是**等参数对照臂**（SCHEMA §15）：并行接线的等参数形式不唯一，故以
  「两子层各带 pre-LN、参数量逐位等于基线」的变体把并行收益与参数账解耦。

## 3. 指标表（逐项分析）

| 维度 | 指标 | 与谁比 |
|---|---|---|
| 解决什么问题 | 层间拓扑（顺序 / 并行 / 层间加权和）在 stock 形态上的质量头寸 | 命题本身 |
| 代价（质量面） | Δ clean bpc（13M / 12k / seed1 / 98.3M token，matched-step 快照） | `sequential` 基线 |
| 有效性 | Δ vs 基线 + 方差带 0.014（|Δ|<0.014 不可排序） | `harness/SCHEMA.md` §12 判据线 |
| 收益·用法 1（质量） | 同引擎 dense 13M matched 税分母（1.5287）下的税 | 历史 `23d-dense-B-12k` |
| 收益·用法 2（口径） | 逐深度 clean bpc 曲线 + next-token acc | 基线 |
| 收益·用法 3（出口资产） | tf@k 接受率（tf@4/tf@6）、等效验证加速（FLOPs 口径） | 基线；并列判据 tf@4 ≥0.95 |
| 可控性/校准 | prob0 ECE、逐深度 ECE | 基线 |
| **代价面（2.2.4 必报）** | 层内串行依赖长度（2 段→1 段）、每层 / 全栈每 token 等效 MACs、参数量 Δ | `sequential` |
| **干预活性（必报）** | dense：层权 α 的 fro/absmax/非零占比/与 identity 起步最大偏离；parallel：逐层接线相对偏差 + 分支增量范数 | 0（死干预） |
| **稳定性（必报）** | gnorm max/中位数/尖峰、NaN、loss 突跳、逐深度 ECE | `results.json::stability` |

**判读规则**：先过「实现 / 接线」门（单测 + 冒烟 + 参数量账）与「干预活性」门；两者不过
→ 先修实现再判机制。

## 4. 判据线（预先写下，逐字来自 ticket 61 §9.2，不得修改）

- **2.2.3**：s0/s1 两 seed 的 **dense − sequential** 配对 Δ 是否越 **0.014 bpc 带**；
  不越 ⇒「无差异」可下；两 seed 同号且越带 ⇒「可排序」。
- **2.2.4**：`parallel` / `parallel-dual-ln` 相对 `sequential` 的质量代价（**>0.014 即代价
  可排序**）**与**代价面同框（层内串行依赖 **2 段 → 1 段**；参数量 **−2d** 与**逐位等参对照**
  两臂并列）。
- **并列（契约原文）**：`tf@4` 落 `results.json`；每臂活性非零。
- 方差带参照：bpc 用 **0.014 bpc 带（限配对 Δ、≥5 窗口）**；判据阈值即方差带。
- 报「质量代价是否换到可兑现的等效收益」，等价于回答「并行子层在本尺度是否划算」。

## 5. 对照与方差

- 四臂**自包含双跑**（同一脚本、同一代码、同一 seed、同一数据切分、同一评测电池），
  不复用历史 run 作判据臂（`harness/SCHEMA.md` §9）。
- **n = 2 seed**：s0 = 2613（seed 0，`2613-inter-layer-topology-13m/results.json`），
  s1 = 本实验（seed 1）。s0 与本实验的评测口径一致性用**当前电池重测 s0 四条 checkpoint**
  证明（clean_bpc 与 2613 记录逐位一致）；实际存证在 `run_log-attempt1-crash.txt:3`
  （`[s0] remeasured all_bitwise_equal=True`），`results.json::s0_anchor_remeasured`
  待下次全跑落盘（本轮冒烟走 `--skip-s0-anchor`，该字段尚未生成）。
- 方差带 **0.014**；另报同 seed 内窗口级配对 Δ 的 SE 与两半样本（`results.json::paired_delta_s1`）。
- 历史 `23d-dense-B-12k`（同配置 sequential 接线）只作**交叉核对**（2613 已重测复现 1.5287）。

## 6. 成本与 scope

| 项 | 值 |
|---|---|
| `gpu_wall_budget` | **45 min**（冒烟 ~1 + 四臂训练 ≈27 + 四臂电池 + 三次 60s 插队窗口 + 余量） |
| 最小可判定版本 | 13M / 12k 步 / seed 0+1 / clean bpc + tf 表 + 活性 + 稳定性 + 代价面（四臂） |
| 峰值显存 | ≤5GB net（冒烟实测峰值 1.4–1.5GB/臂） |
| 可中断性 | 是（每 200 步落日志；`--skip-train` 可用已有 checkpoint 重算电池） |
| 冒烟计划 | `smoke.py`：参数量账 + 初始化逐位一致 + 恒等起步 + 三个非基线臂各 20 步 CLI（接线/活性/步时/显存） |
| 预算不够时砍什么 | 先砍 `dense` s1（用 2613 s0 dense 作 n=1 记录并如实降级结论）；**不砍判据线** |

规模纪律（`harness/SCHEMA.md` §7）：单卡 RTX 4090 24GB 且被多任务共享，GPU 任务串行（§20 锁）；
只做 13M 原理验证，不外推规模/渐近。

## 7. 既有做法（prior art，SCHEMA §18）

| 谁做过 | 为什么这么设计 | 实测结果（来源） | 我们的增量 |
|---|---|---|---|
| **DenseFormer**（2023，arXiv:2402.02622） | 每层输入 = 对**所有此前层输出**的学习加权和（`deep layer aggregation` 式），缓解逐层串联的信息瓶颈 | 同参数下训练更快、loss 更低；层权对推理内存有 +LLM 结构开销 | 13M 单因素受控：与 stock 单残差在同一引擎、同预算、n=2 seed 下比 clean bpc；α 无单纯形约束，精确语义差异已记 `gaps` |
| **PaLM**（2022）/ **GPT-J** | 层内「注意力 ∥ MLP」同读层输入，缩短层内串行依赖以提高硬件利用率（吞吐） | 业界采用作吞吐优化；质量代价通常小幅（论文口径含大量其它改动，非单因素隔离）——**记忆·未核** | 13M 单因素隔离：并行接线的**质量代价**与**等效代价面**（串行依赖/FLOPs/参数）同框，n=2 seed |
| 本地 **2613**（契约 E，旧格 C3 首探） | 上述两族接线的首次本地探索 | dense −0.0008（中性 n=1）、parallel +0.0458 / dual +0.0402（差，n=1） | 本实验 = seed=1 复核（n=2）+ 补 2.2.4 从未测的代价面 |

- 「PaLM/GPT-J 质量代价小幅」为**记忆·未核**（未检索）——只用于设计方向，不作判据。
- 未找到「13M、单因素隔离、只改层间接线、n≥2」的直接前例（检索词：DenseFormer、parallel
  sublayers、GPT-J architecture、PaLM parallel attention MLP）。

## 8. 失败决策树

| 症状 | 诊断顺序 | 处置 |
|---|---|---|
| 当前电池重测 s0 checkpoint ≠ 2613 记录 | ① 实现/口径 | 先修电池；未复现前不判跨 seed |
| 干预死（dense α 停在 identity / 并行分支增量为 0） | ① 实现 | 检查 init 与接线；活性门不过不判机制 |
| 参数量账不符（等参数臂不等参数） | ① 实现 | 修到逐位相同再跑 |
| 2.2.3 两 seed Δ 一正一负 / 至少一 seed 落带内 | ② 无头寸 | 如实报「无差异」，给 0.014 带与 seed 散布两列 |
| 2.2.4 并行两臂质量代价越带但无等效算力收益 | ④ 设计在该设定下无效 | 记「不划算」，写清等效收益只在墙钟/流水侧、不在本项目等效算力度量内 |
| 结果不可重复（两 seed Δ 反号） | ⑤ 方差 | 如实标不可排序；不裁决为质量结论 |

## 9. 产出

`axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/`
下的 `README.md`(本文件) / `run.py` / `smoke.py` / `config.json` / `results.json` /
`analysis.md` / `MODEL.md` / `run_log.txt` / 四臂 `*-train_log.jsonl` + `*-gate_log.jsonl` /
`smoke/smoke.json` / `runs/<arm>-12k-s1/`（权重，`.pt` 不入库）。

> 契约副本标注（`harness/SCHEMA.md` §17.6）：§0 为契约摘要，**判据线逐字保留**；原文见
> `.scratch/61-axis2-depth/TICKET.md` §9.2 WP-2651。
