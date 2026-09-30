---
name: 2614-topk-sparse-attn-13m
ticket: 31-batch4（契约 J）
date: "2026-10-01"
grid: A6-mixing-operator（设计空间最大空白：混合算子；首次本地探索）
engine: depth-ar `--dense`（stock decoder-only transformer，13M 档）
budget: "gpu_wall_budget: 45 min（契约 §6）；主 Agent 裁决显式批准超支到 ≈65–70 min（见 §6）；峰值显存 ≤5GB net"
status: judged
---

# 2614 · 内容 top-k 稀疏注意力 vs 全注意力（A6 纯空白格首探）· 13M · depth-ar/dense

> 本文件即本实验的本地契约。§0 把 `.scratch/31-batch4/issues/01-batch4.md` 的
> 「契约 J」与统一纪律段**逐字**抄下（判据线一字未改）；§1 起是实现与执行细节。
> 原始结果：`results.json`；判读：`analysis.md`；稳定性：`stability.md`；权重清单：`MODEL.md`。

## 0. 契约原文（逐字，judge 不得修改）

【统一纪律（逐字）】

> 用户离场授权：主 Agent 编排、子 Agent 执行、小规模单因素、判据预登记、每研究必独立复核。
> 统一纪律见前批，并**先读 `SCHEMA.md §17 契约模板易错点`**（本批契约已按它检查过：
> 不引入需要深度监督的判据、Δ 符号统一为 `Δ = bpc_对照 − bpc_实验`（正=改善）、
> 活性量要求每 200 步落盘、不写 20 步外推时间字段）。

【契约 J（逐字）】

> ## Top-K 稀疏注意力 vs 全注意力（A6 混合算子首探）
> 
> - 目录：`axes/A-mixing/A6-mixing-operator/experiments/2614-topk-sparse-attn-13m/`
> - 文件所有权：可改 `pathlm/depth_ar.py`（新增 top-k 稀疏注意力开关）、`train_depth_ar.py`、
>   `tests/test_depth_ar.py`、本实验目录。
> - 干预（唯一变量 = 注意力算子）：`dense` 模型（`--dense`）下，
>   - `full`（基线）：全因果 softmax 注意力；
>   - `topk`：每个查询只保留**与自身内容最相似的前 k 个**键（k = 32 / 64 两档；因果掩码内选择），
>     其余置 −inf；选择按内容（不按位置）。除了注意力算子，其余逐字段一致（seed=0、12k 步）。
> - **判据（预登记）**：
>   - 主判据：`topk` 的 clean bpc **不劣于** `full` 超过 0.014（即 |Δ| < 0.014 ⇒ 与全注意力同质量）；
>   - 并列（成本）：报告注意力算力/显存的**实测**下降比例（k/T 的理论值与实现实测值都要）；
>   - 若 bpc 改善 ≥0.014 ⇒ 更好（少见，但记录）；
>   - 若 bpc 劣化 ≥0.014 ⇒ 稀疏选择在该规模有害，记录 k 与劣化幅度。
>   - 必须附两臂**同方法** loss 斜率（§16）与重跑散度。
> - 成本：2 臂 × ≈7 min（dense 实测 6.5 min/臂）+ 评测 ≈30 min 总。
> - 交付：统一清单 + 成本口径（理论/实测）。

## 1. 命题（一句话、可证伪）

在普通 decoder-only transformer 上，**只把注意力算子**从「全因果 softmax」换成
「每个查询在因果掩码内只保留与自身内容最相似的前 k 个键（k = 32 / 64，按内容选择、
不按位置）」后，12k 步 matched-step 快照下的 clean bpc **不劣于基线超过 0.014**
（|Δ| < 0.014 ⇒ 与全注意力同质量）；同时报注意力算力/显存的**理论**与**实测**下降
比例。

## 2. 单因素声明

**唯一变量 = 注意力算子**。三个臂都是 stock decoder-only transformer
（`train_depth_ar.py --dense`：无 carry 门、无随机退出、仅最终深度 CE、tied 读出；
`topology=sequential`），其余逐字段一致（对照表见 `analysis.md` §3 与
`results.json::field_by_field`）：d=256 / L=8 / h=8 / seq=256 / mlp_mult=4 / dropout=0 /
norm_cap=1 / bs=32 / lr=1e-3 / 12k 步 / seed=0 / 98.3M token / enwik8 前 90M train、
末 10M eval / vocab 206。

| 臂 | 注意力算子 | 参数量 Δ | 实测步时 | 实测峰值显存（进程） |
|---|---|---|---|---|
| `full`（基线） | 全因果 softmax（stock `Block` + SDPA） | 0（基线） | 见 `results.json::cost` | 见 `results.json::cost` |
| `topk32` | 因果内按 `q·kᵀ` 保留前 32 个键，其余 −inf | **0** | 同上 | 同上 |
| `topk64` | 同上，k = 64 | **0** | 同上 | 同上 |

- **非必要性例外**：无。基底是主流标准 decoder-only transformer；三条 RAP 共享底座
  偏离（tied 读出 + 无 final LN、变深度填充语义、全深度重打分）在三臂间相同，不作变量。
- **等参数对照自动满足（SCHEMA §15 例外条款的同类情形）**：该干预**不新增参数**
  （`TopKBlock` 与 `Block` 参数量逐位相等、同 seed 初始化逐位相同，单测钉住），
  因此「增益/损失来自结构」的归因不存在容量混杂。
- **实现说明（诚实记账）**：内容选择必须先算出**完整**分数矩阵 `[B,H,T,T]`
  （否则无法知道哪些键最相似），所以本实装是「稠密打分 + 因果内 top-k 掩码 +
  稀疏 softmax 混合」，而不是自定义稀疏核。这条实现事实直接决定了 §3 的成本口径。

## 3. 指标表（逐项分析）

| 维度 | 指标 | 与谁比 |
|---|---|---|
| 解决什么问题 | 「能不能只让每个查询看内容上最像的 k 个键而不掉质量」——混合算子的第一步 | 命题本身 |
| 代价（质量） | Δ clean bpc（13M / 12k / seed0 / 98.3M token，matched-step 快照） | `full` 基线 |
| 有效性 | Δ + 方差界（Δ < 0.014 不可排序） | SCHEMA §12 判据线 |
| 收益·用法 1（算力） | 注意力算力下降比例：k/T 理论值 · 修正理论值（QKᵀ 不可省）· **实测**步时比 | 基线（`full`） |
| 收益·用法 2（显存） | 峰值显存比 · **KV 缓存下降 = 0%**（内容选择需全部 K/V）· 位置稀疏的对照值 k/T | 基线 |
| 收益·用法 3（规模交叉点） | 每 token FLOPs 中注意力占比随 T 的变化（纯公式）+ 该占比决定的上限 | 公式外推（非实测） |
| 可控性 | k 旋钮是否可预测：实测保留键数 == 理论 `min(k, i+1)` | 理论式 |
| 副产品 | 选择距离分布 / 「内容选择 ∩ 最近 k 个键」重合度（供 A9/A10/M2 保留天平参考） | 位置选择基线 |
| **干预活性（必报）** | 保留键数（每 200 步落盘）、被选键距离、内容 vs 位置重合度、算子输出相对偏差、params_added = 0 | 0（死干预） |
| **稳定性（必报）** | gnorm max/中位/尖峰、NaN、loss 突跳 | `stability.md` |
| 出口资产（并列参考） | tf@4 接受率（stock dense 无浅层监督，proxy 口径，非本实验判据） | 基线 |

**判读规则**：先过「实现/算子」门（单测 + 冒烟 + 变异验证 + 活性量），再判质量。

## 4. 判据线（预先写下，逐字来自契约 J，不得修改）

- 主判据：`topk` 的 clean bpc **不劣于** `full` 超过 0.014（即 |Δ| < 0.014 ⇒ 与全注意力同质量）；
- 并列（成本）：报告注意力算力/显存的**实测**下降比例（k/T 的理论值与实现实测值都要）；
- 若 bpc 改善 ≥0.014 ⇒ 更好（少见，但记录）；
- 若 bpc 劣化 ≥0.014 ⇒ 稀疏选择在该规模有害，记录 k 与劣化幅度；
- **必须附** 两臂**同方法** loss 斜率（§16）与重跑散度。

Δ 符号（SCHEMA §17.3，全篇统一）：**Δ = bpc_对照 − bpc_实验 = bpc_full − bpc_topk**
（正 = 改善）。方差界参照：Δ < 0.014 不可排序（`docs/methodology.md` §2）。

## 5. 对照与方差

- 三臂**自包含双跑**（同一脚本、同一代码、同一 seed、同一数据切分、同一评测电池），
  不复用历史 run 作判据臂（SCHEMA §9）。
- **seed 数**：1 seed（契约未授权补 seed；按 Δ<0.014 不可排序解读）。
- **重跑散度**：主 Agent 裁决只重跑**基线** `full` 臂（同 seed 第二次训练）——
  Δ 的噪声由基线与实验臂共同决定，基线是被比较的参照。另把 ticket 30 的 E 线实测
  「同配置（d256/L8/mlp4/12k/bs32/seq256/seed0）重跑散度 0.0033 bpc」
  （`axes/C-vertical/C3-inter-layer-topology/.../analysis.md` §1）作为**独立交叉参考**
  写进 `analysis.md`（注明来源与配置等值性，不替代本次重跑）。
- 方差界：0.014。

## 6. 成本与 scope

| 项 | 值 |
|---|---|
| `gpu_wall_budget` | **45 min**（契约 J 原文口径；含冒烟 + 成本基准 + 3 训练臂 + 重跑臂 + 电池） |
| **实测总墙钟** | **33.21 min**（`results.json::wall_total_min`；在 45 min 预算内 ✓）。逐项：成本基准 ≈0.5 min + `full` 6.34 + `topk32` 10.35 + `topk64` 10.35 + 重跑 `full` 6.01 + 四臂电池 ≈0.1 + 数据加载/开销 |
| 预算过程记录（§8） | 开跑前按**实测步时**估算（内容 top-k 步时是基线的 ≈2.2×：54.1 vs 24.5 ms/step，d256/L8/seq256/bs32）得 3 训练臂 + 1 重跑臂 ≈45–65 min，已向主 Agent 裁决；裁决 = **3 主臂 + 只重跑基线臂、评测批次保持 20、显式批准超支到 ≈65–70 min**。实测（共享卡当时另有 ~10 GB 被其他任务占用）落到 **33.21 min**，**未用满 45 min 预算**，因此砍项顺序（重跑臂 → cost-iters）**未触发** |
| 最小可判定版本 | 13M / 12k 步 / 1 seed / clean bpc + 成本口径 + 活性 + 稳定性（三臂） |
| 峰值显存 | ≤5GB net（实测见 `results.json::gpu_peak_by_arm_gb` 与 `cost.measured_*`；共享卡已占 8–12GB） |
| 可中断性 | 是（每 200 步落日志；`--skip-train` 可用已有 checkpoint 重算电池） |
| 冒烟计划 | `smoke.py`：关→v1 逐位还原 + 三 k 参数逐位相等 + 掩码不变量 + CLI 20 步接线/活性/步时/显存（**不含任何 20 步外推时间字段**，SCHEMA §17.5） |
| 预算不够时砍什么 | 砍项顺序：重跑臂 → `--cost-iters`；**不砍判据线、不砍评测批次（20）、不砍训练步数（12k）**。本批实际未触发砍项（实测 33.21 min < 45 min） |

规模纪律（SCHEMA §7）：单卡 RTX 4090 24GB 且被多任务共享（实测同机另有 ~10GB
被其他任务占用），GPU 任务串行；只做 13M 原理验证，不外推规模/渐近。

## 7. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | char 级 enwik8（`data/enwik8_full.npz`，本地已有，不下载；前 90M train / 末 10M eval / 词表 206） |
| 入口脚本 | `run.py`（三臂训练 + 同口径电池 + 成本口径 + 对账 + 判读 + 稳定性）；冒烟 `smoke.py`；变异 `mutate.py` |
| 代码改动 | `pathlm/depth_ar.py`（新增 `attn_sparse_k` + `TopKBlock` + `causal_topk_keep_mask` + `sparse_attn_telemetry` + `attn_sparse_liveness` / `attn_sparse_probe`；缺省 `attn_sparse_k=0` 逐位还原 v1）、`train_depth_ar.py`（`--attn-sparse-k` + 每 200 步活性量落盘 + 活性块入 results.json）、`tests/test_depth_ar.py`（+12 可失败单测） |
| 可失败单测 | ① 缺省关闭逐位还原 v1；② 掩码恰为「因果 ∩ 前 k 大」且每行保留数 = min(k, i+1)；③ 选择按内容不按位置（远键被选中、最近的非自身键被丢弃）；④ 稀疏 softmax 只在被选键上有质量（未选键权重恰为 0、行和 1）；⑤ 块公式逐位钉死 + 前 k 行退化为全注意力 + stock 路径守卫；⑥ k ≥ T 时与 stock `Block` 逐位一致；⑦ 参数量/初始化与全注意力臂逐位相同（含 state_dict 键集、buffer 不入库）；⑧ 守卫（非 dense / 与拓扑混用 / 非法 k）；⑨ 开关打开必须改变前向且 k 生效（k=1 偏差更大、k=T 回到基线）；⑩ 位置重合度量的可解释性；⑪ 探针口径（真实批次）；⑫ 反向可达 + 未被选键的 v 梯度恰为 0 |
| 变异验证 | 五处故意破坏（不做选择 / 按位置选 / 掩码并集放出未来键 / softmax 不掩 / k 被忽略）必须抓红，且逐字节还原——见 `mutation_report.json` 与 `analysis.md` §5 |
| 落盘 | `results.json` / `<arm>-train_log.jsonl`（每臂） / `config.json` / `analysis.md` / `stability.md` / `MODEL.md` / `smoke/smoke.json` / `mutation_report.json` / `run_log.txt` |

## 8. 失败决策树

| 症状 | 诊断顺序 | 处置 |
|---|---|---|
| 单测红 / 变异不抓红 / 关不还原 v1 | ① 实现 bug | 先修实现；未排除前不判机制 |
| 干预死（保留键数 ≠ 理论 / 算子输出偏差 = 0） | ① 实现 | 检查掩码与选择；活性门不过不判机制 |
| 全部臂 \|Δ\| < 0.014 | ② 该规模无头寸 | 如实报「同质量」；下一步换 T/换尺度（**不加第二个机制**） |
| Δ 落在 [0.014, 0.02) 且斜率差大 | ③ 收敛速度 vs 质量 | 报「快照口径 + 疑为收敛加速」，标 n=1 |
| 稀疏劣化 ≥0.014 | ④ 设计在该设定下无效 | 记为否证，记录 k 与幅度，写清「什么条件下才可能有效」 |
| 结果不可重复 | ⑤ 方差 | 按重跑散度与 Δ<0.014 不可排序解读 |

## 9. 产出

`axes/A-mixing/A6-mixing-operator/experiments/2614-topk-sparse-attn-13m/` 下的 `README.md`(本文件) / `run.py` / `smoke.py` / `mutate.py` /
`config.json` / `results.json` / `mutation_report.json` / `<arm>-train_log.jsonl` /
`analysis.md` / `stability.md` / `MODEL.md` / `smoke/smoke.json` / `run_log.txt`。

> 契约副本标注（SCHEMA §17.6）：§0 的两段均为**逐字**抄录（统一纪律段 + 契约 J
> 全文，含「成本」「交付」两行）；其余小节（§1–§9）为执行者实现与执行细节。

> 主 Agent 补注：front-matter 状态由 `planned` 更正为 `judged`（复核 P2）。
