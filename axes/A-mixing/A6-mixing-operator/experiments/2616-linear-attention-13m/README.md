---
name: 2616-linear-attention-13m
ticket: 33-batch6（契约 R）
date: "2026-10-02"
grid: A6-mixing-operator（混合算子；ticket 31-J 内容 top-k 之后的第二个算子）
engine: depth-ar `--dense`（stock decoder-only transformer，13M 档）
budget: "gpu_wall_budget: 45 min（含冒烟 + 成本口径 + 2 臂 + 1 重跑 + 全部评测；峰值显存 ≤5GB net）"
status: judged
---

# 2616 · 线性注意力 vs 全注意力（A6 混合算子续）· 13M · depth-ar/dense

> 本文件即本实验的本地契约。§0 把 `.scratch/33-batch6/issues/01-batch6.md` 的开头
> 统一纪律段与「契约 R」抄下：**判据线逐字（一字未改）**，其余段落标注「摘要」。
> 原始结果：`results.json`；判读：`analysis.md`；稳定性：`stability.md`；权重清单：`MODEL.md`。

## 0. 契约原文（判据线逐字，不许改）

【开头统一纪律（摘要，逐字范围见下）】

> 用户离场授权：主 Agent 编排、子 Agent 执行、小规模单因素、判据预登记、每研究必独立复核。
> 统一纪律同前批；**先读 `SCHEMA.md §17`（12 条易错点）**，尤其：
> 第 11 条（单点曲线不得累加外推；累计效应直接测）、第 12 条（压缩/稀疏类比较须同时对齐
> 「删除量」与「可见性对数」两套口径）、第 8 条（臂数=干预档数×(1+重跑)；成本按本实验实测步时）。
>
> 共用器材（必须断言 sha256）：
> - 全注意力 dense 基线：`axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt`
> - 仪器：`tools/evalkit/retention.py`（**只增不改**，既有单测必须继续绿）

（上段为逐字抄录。本实验对共用器材的处置见 §7：仪器**只读**、既有单测全绿；
`trunk.pt` 只作跨批定性参考，其 sha256 已断言，见 §7.3。）

【契约 R（逐字）】

> ## 契约 R：线性注意力 vs 全注意力（A6 混合算子续）
>
> - 目录：`axes/A-mixing/A6-mixing-operator/experiments/2616-linear-attention-13m/`
> - 文件所有权：可改 `pathlm/depth_ar.py`（新增线性注意力开关，缺省逐位还原 v1）、
>   `train_depth_ar.py`、`tests/test_depth_ar.py`、本实验目录。
> - 动机：ticket 31-J 已证「内容选择式 top-k」在 T=256 净亏（质量越界 + 实测 2.2× 慢、KV 不省）。
>   线性注意力不同：它**不需要全分数矩阵**，理论上同时降时与降内存，是本轴上最可能「不亏」的算子。
> - 干预（唯一变量 = 注意力算子）：`linear`（特征映射 φ(q)·φ(k) 的线性注意力；用 elu+1 或
>   cosformer 简化形式，实现方式在 README 里写明并单测钉住）vs `full`（基线）。
>   其余逐字段一致（`--dense`、seed=0、12k 步）。
> - **判据（预登记）**：
>   1. 质量：|Δbpc| < 0.014 ⇒ 与全注意力同质量；Δ ≥ +0.014 ⇒ 线性化在该规模有害（记录幅度）；
>   2. **成本交叉点（SCHEMA §17 第 9 条）**：必须报**理论**（注意力 FLOPs、KV 内存）与**实测**
>      （ms/step、峰值显存）两组数，并写出「在什么 T 上开始回本」；
>   3. 附：`linear` 的评测期依赖曲线（仪器）——看线性注意力是否天然更依赖近处。
> - 成本：2 臂 × ≈7–15 min + 评测 ≈20–40 min。
> - 交付：统一清单。

（判据线三条 + 成本行 + 目录/文件所有权为**逐字**；动机段为逐字；其余为摘要。）

## 1. 命题（一句话、可证伪）

在普通 decoder-only transformer 上，**只把注意力算子**从「全因果 softmax 注意力」换成
「归一化**线性核**注意力 `out_i = Σ_{j≤i}(φ(q_i)·φ(k_j))v_j / Σ_{j≤i}φ(q_i)·φ(k_j)`，
φ = elu(x)+1」（用累积状态实现 ⇒ 每层时间/显存对 T 线性）后：

1. 12k 步 matched-step 快照下的 clean bpc 与全注意力**同质量**（\|Δ\| < 0.014）；且
2. 报出注意力算力/KV 内存的**理论**降幅与 ms/step、峰值显存的**实测**值，
   并写出「在什么 T 上开始回本」；且
3. 给出 linear 臂的评测期依赖曲线（删键/只留最近 d），看它是否天然更依赖近处。

## 2. 单因素声明

**唯一变量 = 注意力算子**。两臂都是 stock decoder-only transformer
（`train_depth_ar.py --dense`：无 carry 门、无随机退出、仅最终深度 CE、tied 读出；
`topology=sequential`），其余逐字段一致（对照表见 `analysis.md` §3 与
`results.json::field_by_field`）：d=256 / L=8 / h=8 / seq=256 / mlp_mult=4 / dropout=0 /
norm_cap=1 / bs=32 / lr=1e-3 / 12k 步 / seed=0 / 98.3M token / enwik8 前 90M train、
末 10M eval / vocab 206。

| 臂 | 注意力算子 | 参数量 Δ | 实测步时（T=256，本实验实测） | 峰值显存（进程） |
|---|---|---|---|---|
| `full`（基线） | 全因果 softmax（stock `Block` + SDPA） | 0（基线） | 见 `results.json::cost` | 见 `results.json::cost` |
| `linear` | 归一化线性核 φ(q)·φ(k)，φ=elu+1，累积状态 | **0** | 同上 | 同上 |

- **非必要性例外**：无。基底是主流标准 decoder-only transformer；三条 RAP 共享底座
  偏离（tied 读出 + 无 final LN、变深度填充语义、全深度重打分）在两臂间相同，不作变量。
- **等参数对照自动满足（SCHEMA §15 的例外条款同类情形）**：该干预**不新增参数**
  （`LinearAttnBlock` 与 `Block` 逐位同构：同 seed 下参数量、`state_dict` 键集与每一个
  张量逐位相同，单测 `test_linear_params_and_init_identical_to_full_attention_block`
  钉住），因此不存在容量/重参数化混杂。
- **实现说明（诚实记账，§10）**：训练实装 = **分块并行**形式（块间累积状态、块内显式
  因果打分，chunk=64）；`chunk` 是**实现细节**不是第二个干预档（单测钉住换 chunk 不改
  语义）。评测期仪器路径（掩码前向）用等价的**显式打分**参考实现（O(T²)），两路径的
  一致性由单测 + `results.json::instrument_selfcheck` 量出（线性臂只有浮点累加顺序差）。

## 3. 指标表（逐项分析）

| 维度 | 指标 | 与谁比 |
|---|---|---|
| 解决什么问题 | clean bpc（每字符比特数，越低越好） | 同尺寸同配置的 `full` 臂 |
| 代价（质量） | \|Δbpc\| 与配对 SE、重跑散度 | 判据阈值 0.014（本项目方差界） |
| 代价（算力） | ms/step（整模型 fwd+bwd，同时刻同机） | 同 T 的 `full` 臂；随 T 的斜率 |
| 代价（显存） | 峰值显存（进程 + `torch.cuda.max_memory_allocated`） | 同上；**解析内存与实测分开报** |
| 有效性（理论） | 注意力 MACs/层/token、KV/状态元素数、交叉点 T | 全注意力的同口径公式 |
| 可控性 | 是否 O(T)、是否有 [T,T] 张量、chunk 是否只是实现细节 | 单测 + FLOP 计数 + 形状追踪 |
| 副产品 1 | 评测期依赖曲线（按距离单点 + 按累计预算） | `full` 臂同一仪器 |
| 副产品 2 | 检索腿（短语针召回，全可见 + recent:d=64 + 无针地板） | `full` 臂同一批针 |
| 副产品 3 | 同权重下两种算子的注意力质量分布（距离均值 / 落在最近 d 内的份额） | 同一模型权重 |
| **干预活性（必报）** | φ 实测量程 / 归一化分母 / 块间状态贡献份额 / 同权重换算子的输出偏差 | 每 200 步落盘 + 评测探针 |
| **稳定性（必报）** | gnorm / NaN / loss 突跳 | `stability.md` |

## 4. 判据线（预登记，逐字，一字未改）

> 1. 质量：\|Δbpc\| < 0.014 ⇒ 与全注意力同质量；Δ ≥ +0.014 ⇒ 线性化在该规模有害（记录幅度）；
> 2. **成本交叉点（SCHEMA §17 第 9 条）**：必须报**理论**（注意力 FLOPs、KV 内存）与**实测**
>    （ms/step、峰值显存）两组数，并写出「在什么 T 上开始回本」；
> 3. 附：`linear` 的评测期依赖曲线（仪器）——看线性注意力是否天然更依赖近处。

**符号约定（必须一次说清，SCHEMA §17.3）**：判据 1 的「Δ」在契约文面上有两种读法
（`Δ = bpc_对照 − bpc_实验`（正 = 线性更好，SCHEMA §17.3 的规范写法）与
`Δ = bpc_linear − bpc_full`（正 = 线性更差））。**本实验两个符号都报、都不修改判据线**：

| 名字 | 定义 | 正号含义 |
|---|---|---|
| `delta_bpc_schema` | `bpc_full − bpc_linear` | 线性更好 |
| `delta_bpc_contract` | `bpc_linear − bpc_full` | 线性更差（契约句面读法） |

结论只依赖 \|Δ\| 与方向，不依赖选哪一种符号约定；`analysis.md` §0 与
`results.json::verdict` 两个数并列给出。详见 `analysis.md` §11 的「契约口径分歧记录」。

## 5. 对照与方差（SCHEMA §6.5）

- 对照臂：`full`（同配置、同 seed、同步数、同数据流，**本实验内自包含双跑**）；
- 重跑臂：`linear` 同 seed 重训一遍（新机制的可复现性未验证过）⇒ 重跑散度；
- 跨批参照：ticket 31-J 的同配置 dense 基线 `1.5304`、其重跑 `1.5279`（散度 `0.0025`）
  ——**只作量级参考**（复用只省算力不省证据，SCHEMA §9）；
- 方差界：`Δ < 0.014` 不可排序（`docs/methodology.md` §2），n_seed = 1（契约未授权补 seed）；
  阈值近旁的噪声另用**同批配对差**（20 批，`paired_eval.py`）与重跑散度量化。

## 6. 成本（§7/§8 纪律）

- `gpu_wall_budget: 45 min`（含冒烟 + 成本口径 + 2 臂训练 + 1 重跑 + 全部评测）。
  契约 R 的成本行（逐字）是「2 臂 × ≈7–15 min + 评测 ≈20–40 min」；本实验按
  **本实验 20 步冒烟实测的步时**估（SCHEMA §17.8：不得借用别的实验的步时）：
  `full` 21.5 ms/step、`linear` 55.5 ms/step（T=256, bs32，`smoke/smoke.json`；
  正式成本口径见 `results.json::cost`，两处同一数量级、差异 = 共享卡抖动）。
- **实测墙钟（**组装跑**的 `results.json::wall_total_min`；主跑墙钟见 `run_log.txt` 与 `run_log*.txt`）**：
  `full` 臂 12k 步 **5.75 min**（进程墙钟 6.01 min，峰值显存 1.47 GB）；
  成本口径 ≈0.2 min；full 臂电池 + 依赖曲线 + 检索腿 ≈0.06 min；
  **linear 臂：三次尝试各 5–15 min 均停摆（未产出 checkpoint）** ⇒
  本实验的实际 GPU 花费 ≈ 6.5 min（成功部分）+ ≈28 min（停摆尝试，含 5 次诊断二分）。
  事故与二分证据见 `stall_diagnosis.json`、`analysis.md` §3、`stability.md`。
- **预算不够时砍什么（顺序）**：重跑臂 → 成本随 T 的点 → 检索腿批数；
  **不砍判据线、不砍评测批次**。本次实际执行按此顺序砍掉了**重跑臂**
  （30 min 收口边界 + 三次停摆吃掉预算），并在 `analysis.md` §7.2 如实记账。
- 显存：单卡 4090 24GB **共享**（运行时另有其他任务占 3.5–6.3 GB）；GPU 任务串行，
  峰值显存 net ≤5GB（实测见 `analysis.md` §6.1）。
- 可中断性：每个臂一个独立子进程 + 每 200 步一行日志；中断只损失当前臂。

## 7. 数据与脚本清单

### 7.1 数据

`data/enwik8_full.npz`（char 级 enwik8；前 90M train / 末 10M eval，词表由全文件统计 = 206），
本地已有，**不下载、不入库**。

### 7.2 入口与落盘

| 文件 | 内容 |
|---|---|
| `run.py` | 主入口：成本口径（理论 + 实测）→ 训练两臂 + 电池 → 逐字段表 → 仪器自检 + 依赖曲线 + 检索腿 → 重跑（预算内）→ 判读 |
| `smoke.py` / `smoke/smoke.json` | 20 步冒烟（v1 还原 / 参数账 / 活性 / O(T) 证据 / 因果 / 仪器路径 / CLI 接线 / 步时显存） |
| `mutate.py` / `mutation_report.json` | 6 处变异验证（抓红 + 逐字节还原） |
| `paired_eval.py` / `paired_eval.json` | 同批 20 个评测位置的配对差与 SE |
| `reconcile_numbers.py` / `reconcile_report.json` | 报告数字回算（每个报告数字与 `results.json` 逐一对账） |
| `config.json` / `results.json` | resolved config（含 seed）/ 全部电池与判读 |
| `<arm>-12k-s0-train_log.jsonl` | 每臂训练日志（每 200 步，含 `attn_linear` 活性量） |
| `runs/<arm>/` | 每臂 checkpoint、`results.json`、`stdout.log`（权重不入库） |
| `README.md` / `analysis.md` / `stability.md` / `MODEL.md` | 契约与判据线逐字副本 / 判读 / 稳定性 / 权重哈希 |

### 7.3 共用器材

- `tools/evalkit/retention.py`：**只读**（本实验未改一行，`git diff` 为空）；
  既有单测全绿（`pytest tests/ -q`，见 `analysis.md` §9）。
- `trunk.pt`（`axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/`）：
  本实验**不使用**它做对照（两臂都是本次新训，自包含双跑）；只在 `analysis.md` §8
  引用 ticket 32-M/2615 在该 checkpoint 上的 recent 曲线作**跨批定性参考**。
  断言：`sha256(trunk.pt) = 86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2`
  （与 2615 `config.json` 记录一致，`sha256sum` 实测核对通过）。

### 7.4 生成命令

```bash
cd /home/z/vibe/pathlm
EXP=axes/A-mixing/A6-mixing-operator/experiments/2616-linear-attention-13m
python3 $EXP/smoke.py                     # 20 步冒烟（先冒烟再全跑）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python3 $EXP/run.py --steps 12000 --seed 0 --budget-min 45 \
  > $EXP/run_log.txt 2>&1 &               # 后台；轮询 ≤110s 单次 sleep
python3 $EXP/run.py --skip-train --no-rerun --arms full \
      > $EXP/run_log_assembly.txt 2>&1 &   # 收口组装：成本口径 + 电池 + 依赖曲线 + 检索腿
python3 $EXP/paired_eval.py               # 配对差（<1 min）
python3 $EXP/mutate.py                    # 变异验证（纯 CPU；不要与训练并发）
python3 $EXP/reconcile_numbers.py         # 报告数字回算
```

## 8. 失败决策树（按此顺序排除）

| 症状 | 排除顺序 | 本实验的抓手 |
|---|---|---|
| bpc 越界/异常 | ① **实现** | 12 条可失败单测（`tests/test_depth_ar.py` 的 linear 段）+ 6 处变异全抓红 + 关闭开关逐位还原 v1 + O(T) 证据（FLOP 计数 ~线性） |
| | ② **测量** | 同批配对差 + SE + 重跑散度 + 与 ticket 31-J 同配置基线的交叉参照 |
| | ③ **规模** | 13M / 98.3M token / T=256 是本仓库标准小规模位点；结论只在该位点成立 |
| | ④ **设计** | 若前三者都排除，把结论落在「线性核在该设定下的头寸」并写清什么条件下才可能翻转 |
| 成本为负（实测更慢/更费显存） | ① 本实装开销（无融合核）| 用**注意力孤岛**把「算法本征」与「本实装」分开量 |
| | ② 算法本征 | FLOP/KV 的**解析**口径 + 交叉点公式 |

## 9. 产出

`results.json` / `<arm>-train_log.jsonl` / `run_log.txt` / `analysis.md`（人话摘要在前）/
`stability.md` / `MODEL.md`（权重 sha256 / 体积 / 生成命令）/ `smoke/smoke.json` /
`mutation_report.json` / `paired_eval.json` / `reconcile_report.json`。
