---
name: 2623-longT-1024
ticket: 40-Y2
date: "2026-10-03"
grid: A6-mixing-operator（长 T 第三点：T=1024）
engine: depth-ar `--dense`（stock decoder-only transformer，7.03M@T=512 → 7.16M@T=1024）
budget: "gpu_wall_budget: 60 min（含冒烟 + 成本基准 + 2 训练臂 + 电池 + 合成腿）；峰值 ≤5GB net"
status: judged
---

# 2623 · 长 T 第三点：T=1024（A6 混合算子的长上下文端）· 7.16M · depth-ar/dense

> 本文件即本实验的本地契约。§0 把 `.scratch/40-next-wave/issues/01-next-wave.md` 的「Y2」段
> **逐字**抄下（判据线一字未改）；§1 起是实现与执行细节。
> 原始结果：`results.json`；判读：`analysis.md`；权重清单：`MODEL.md`。

## 0. 契约原文（逐字）

> 【Y2 · GPU】长 T 第三点 T=1024（A6，≈40–60 min）
> 目录：`axes/1-time/1.1.1-hard-window/experiments/2623-longT-1024/`
> 干预：full vs 窗口 512（**同相对占比 50%**，§17.16），扩绝对位置表（参数增量实测）。
> 判据：① Δ1024 的带宽分类（与 Δ256/Δ512 并列成三点，**仍禁外推**）；② 注意力占比实测 vs 公式
> （X3 实测 5.41%@256→10.78%@512、公式 14.29%→25.00%）；③ 显存实测；④ 墙钟（两轮，方向不明如实记）。
> 若显存越 5GB net ⇒ 按 §20.2 记账或降 batch，但**须保持两臂同配方**。

**无额外主 Agent 裁决条目**（本波 Y2 契约即上方文字；与 ticket 39-X3 不同，本轮没有单点退化
缺陷需要裁决——见 §2 能力面分析）。相关规范直接引用 `harness/SCHEMA.md`：§17.11（禁外推）、
§17.14（判据基准唯一）、§17.16（同族两点须同相对刻度且非退化）、§20/§20.1/§20.2（锁/显存礼貌线）、
§22/§22.8（共享面与能力面）、§19（长程须过局部先验抗性）、§23（预算内极致七问）。

## 1. 命题（一句话、可证伪）

在 stock decoder-only transformer（`train_depth_ar.py --dense`）上，把训练序列长度从 **512 提到
1024**（唯一其他变化 = 位置表随 `seq_len` 线性增大），比较 **全因果注意力** vs **位置式窗口注意力**
（窗口 = T/2 = 512，与前两点同相对刻度 50%），12000 步 / 98.3M token matched-token 快照下：

1. 窗口相对 full 的 **Δbpc 在 T=1024 点上的带宽分类**（与前两点 Δ256/Δ512 并列，只描述）；
2. **注意力子层算力占比**在 T=1024 的**实测**值（与公式 `512T/(512T+786432)` = 40.0% 对照）；
3. 训练**峰值显存**是否越过 5GB net；
4. 墙钟（两轮实测，方向不明则如实记）。

**判据线逐字见 §4（预先写下，不得修改）。**

## 2. 单因素声明与能力面分析（SCHEMA §22.8）

### 2.1 单因素

- **唯一变量 = 训练期可见窗口**（`train_window` ∈ {0 关闭, 512}），T 在两臂间相同（1024）。
  与前两点（256/512）之间只改 `seq_len`（位置表随之线性变大，是 `seq_len` 的**必然后果**，不是第二开关）。
- 基底是**主流标准 decoder-only transformer**（dense：无 carry 门、无随机退出、仅最终深度 CE、tied 读出、
  `topology=sequential`）。三条 RAP 共享底座偏离在**两臂间完全相同**，不作变量。
- **实现说明**：窗口掩码是**位置式**——由 `pathlm.depth_ar.window_anchor_keep_mask` 生成
  `keep[i,j] = (j≤i) ∧ ((i−j) < w)`，训练侧（`DepthARModel.forward`）与评测侧（`retention.masked_logits`）
  **同一函数**（单测钉住）。
- **非必要性例外**：无新增开关、未改 `pathlm/` 与 `train_depth_ar.py`（见 §9）。

### 2.2 能力面分析（派发前逐条核对）

`grep -n "ValueError\|互斥" pathlm/depth_ar.py` 列出的与本组合相关的校验：

- `if cfg.train_window > 0 and not cfg.dense: raise` ⇒ 本实验用 `--dense` ✅
- `if cfg.train_window > 0 and cfg.topology != "sequential": raise` ⇒ 本实验用缺省 `sequential` ✅
- `attn_sparse_k ⊕ train_window`、`attn_linear ⊕ train_window`、`carry_rule ⊕ train_window`、
  `n_kv_heads<n_heads ⊕ train_window` ⇒ 本实验**全不开**这些算子（只开窗口），无互斥 ✅
- 与 X3 同组合（`dense + sequential + train_window`），X3 已四臂跑通 ⇒ **组合在引擎中存在** ✅
- 块类：`train_window > 0` 走 `dense=True` 的掩码装饰层（与 block 类正交的 `attn_mask`），
  与 X3 同路径；`window_anchor_keep_mask` 支持任意 `T`、`w < T`（`w=512 < T=1024`）✅
- **结论：契约要求的组合（full / 窗口 512 @ T=1024）在引擎中可达，无未测因子。**

### 2.3 同相对刻度与非退化（SCHEMA §17.16）

三点 = `window128@T=256` / `window256@T=512` / `window512@T=1024`，窗口占比**同为 50%**。
第三点 `window512@T=1024` **非退化**（`w=512 < T=1024`，与精确因果掩码不等）——用单测钉住。
**禁止**用任何退化点（如 `window1024@T=1024`，恒等）充当点。

## 3. 指标表（逐项分析）

| 维度 | 指标 | 与谁比 |
|---|---|---|
| 解决什么问题 | A6 的「T=256 净亏 / T=512 不亏」结论在 T=1024 是否延续；窗口算子的**长 T 端质量与算力** | A6 的 T=256/512 两点 |
| 代价（质量） | Δ clean bpc（native 工作点：窗口臂 = 自身训练掩码；full 臂 = 全因果） | 本点自己的 full |
| 代价（质量·并列） | release 口径 Δ（窗口臂在全可见下评测） | 同上 |
| 有效性 | Δ + 方差界（Δ < 0.014 不可排序，`harness/methodology.md §2`） | SCHEMA §12 判据线 |
| **收益 1（等效 FLOPs）** | 理论：窗口保留的因果对占比（`visible_frac`）、注意力 MACs 下降、整层 MACs 下降 | full 注意力 |
| **收益 2（墙钟）** | 实测：整模型步时比 `r_time = ms(窗口)/ms(full)`、峰值显存比（**如实报可能 >1**） | full 注意力（同机同时刻） |
| **算力占比（判据②）** | 实测：注意力孤岛 fwd+bwd 步时 / 整模型步时，在 T=1024；与公式 `512T/(512T+786432)` = 40.0% 对照 | 公式外推值（对照，非判据） |
| **显存（判据③）** | 训练峰值（`torch.cuda.max_memory_allocated` + `nvidia-smi` 子进程峰值） | 5GB net 硬预算 |
| 可控性 | 窗口旋钮是否可预测：`visible_frac` 实测 == 理论闭式（单测） | 理论式 |
| 副产品 | T=1024 相对 T=256/512 的**位置表参数增量**（实测） | K10 表 |
| **干预活性（必报）** | `window_mask_liveness`（每 200 步随日志）+ `window_mask_probe`（掩码开/关 logit 差 > 0） | 0（死干预） |
| **稳定性（必报）** | gnorm max/中位/尖峰、NaN、loss 突跳 | 各臂日志 |
| **长程腿（§19）** | 合成长程任务 `var_recall`（距离档 × 臂）的 top-1 准确率 + chance + 局部先验 | 同任务 chance / 局部先验 |

**判读规则**：先过「实现/算子」门（单测 + 冒烟 + 变异验证 + 活性量），再判质量。

## 4. 判据线（预先写下，逐字来自契约 Y2，不得修改）

- **① 质量（三点，只描述、禁外推）**：`Δbpc(T) = bpc_full(T) − bpc_window(T)`（SCHEMA §17.3：正 =
  窗口更好）。三点 = `T=256·window128` / `T=512·window256`（**来自 ticket 39-X3 的 2622，跨 run 引用**）
  与 `T=1024·window512`（本实验）。判读：给出 Δ1024 的**带宽分类**（|Δ| < 0.014 ⇒ 该点「不亏」），
  与 Δ256/Δ512 并列成三点。**只描述三点，不作任何外推/内插。**
- **② 注意力占比**：报 T=1024 的注意力孤岛/整模型**实测**步时比，与公式
  `attn_share(T) = 512T/(512T+786432)` 对照（T=1024 = **40.0%**）。与 X3 已实测的
  5.41%@256 → 10.78%@512 并列。**公式值只是对照，不是判据。**
- **③ 显存**：报每臂训练峰值（进程 + `nvidia-smi`）。**超过 5GB net ⇒ 如实记「越界」**，按 §20.2
  记账或降 batch（**两臂同配方**）。
- **④ 墙钟**：报两轮 `r_time`（preflight 与 eval 各一轮）；**两轮方向不明则如实记「测不出方向」**，
  不得把其中一轮当收益。窗口臂走稠密 bool 掩码 ⇒ 理论省 12.5% 等效 FLOPs **未必兑现**，分开报。

Δ 符号（SCHEMA §17.3，全篇统一）：`Δ = bpc_对照 − bpc_实验 = bpc_full − bpc_window`（正 = 窗口更好）。
方差界：Δ < 0.014 不可排序（`harness/methodology.md §2`）。

## 5. 既有做法（prior art，SCHEMA §18）

> 本环境**联网不可用**（与 `archive/wiki-legacy/axes/A-mixing/PRIOR_ART_LENGTH.md §0`、`PRIOR_ART.md §0` 一致）。
> 外部内容一律 **`记忆·未核`**，只作设计方向、**不得当判据**。

1. **谁做过**：滑动窗口注意力（Longformer / Mistral 的 SWA）是长上下文标准件；稀疏注意力
   （BigBird / Longformer）以子二次注意力换上下文长度。
2. **为什么这么设计**：注意力 FLOPs 随 T² 增长⇒长 T 上按位置丢弃远端键可把注意力算力降到 O(T·w)，
   在长上下文吞吐上占优。
3. **它的实测**：公开报告多为**长上下文卖点**（吞吐/可及长度），**不报短上下文净亏的下界**，也不报
   7.16M/单因素受控下的交叉点。`记忆·未核`，不可当判据。
4. **我们的增量**：A6 已在 **T=256 实测**稀疏（内容 top-k）**净亏**（k=32 Δ=−0.0166，成本 2.2×）；
   ticket 39-X3 在 **T=512 实测**窗口算子首次「不亏」（Δ=−0.0087 落带宽内，记交叉点首观测点，
   单点不外推）。本实验**把第三个点补到 T=1024**，检验「窗口税随 T 收窄」是否继续（**仍禁外推**）。

**检索词**（供将来可核证的 run）：`sliding window attention long context quality`、
`sparse attention short context perplexity net loss`、`attention share of FLOPs by sequence length`。

## 6. 成本与 scope

| 项 | 值 |
|---|---|
| `gpu_wall_budget` | **60 min**（含冒烟 + 成本基准 + 2 训练臂 + 电池 + 合成腿） |
| 预算过程（§8） | 开跑前用 `--stage preflight` **实测步时/显存**再锁定（禁按公式估）——实测值见 `results.json::cost` 与 `analysis.md` |
| 训练臂（2） | `full-1024`(12000 步) / `win512-1024`(12000 步)，均 **bs8**、seed0 |
| **token 口径** | **同 token 预算 98.3M**（与 2622 的 T=256/512 四点同口径：12000×8×1024）。另报步数 |
| **batch 决定（SCHEMA §20.2）** | 开跑前实测（`.scratch` 外临时探针，见 `smoke/smoke.json`）：**bs32 的窗口臂 OOM**（将需 ~6.7 GiB）。改 bs16 后**训练仍 OOM 一次**（窗口臂训练峰值 4.06 GiB；当时另一 agent 的 `pytest tests/` 与 Y1 的 eval 并发占卡 ⇒ 卡满）。最终两臂统一 **bs8**（同配方），token 预算不变。实测（bs16 探针）峰值：full 1.482 GiB / win512 **3.37 GiB**（bs8 实测峰值见 `results.json`）。**注**：`smoke/smoke.json` 的冒烟与步时为 **bs16 时点**的记录（最终训练为 bs8）；本机背景进程（两个 `llama-server`）常驻占 ~18.3 GiB ⇒ 全卡 free 仅 ~5 GiB，**§20.2 的「跑前后余量 ≥8GB」在本机不可满足**，如实记账、不碰别人进程 |
| 最小可判定版本 | 2 臂匹配训练 + native/release bpc + 成本实测 + 合成腿 |
| 峰值显存 | ≤5GB net（共享卡；先测再定）；**本实验用 bs8**（bs32/bs16 的窗口臂均 OOM），两臂同配方（§20.2 记账见上表「batch 决定」） |
| 可中断性 | 是（每 200 步落日志；`--skip-train` 可用已有 checkpoint 只跑电池） |
| 冒烟计划 | `--stage preflight`：两臂各 20 步（实测步时/显存/活性，**不含任何 20 步外推时间字段**，SCHEMA §17.5） |
| 预算不够时砍什么 | 砍项顺序：合成腿距离档 → release 口径 → `--cost-iters`；**不砍判据线、不砍第三点** |

规模纪律（SCHEMA §7）：单卡 RTX 4090 24GB 多任务共享，GPU 串行；只做 7.16M 原理验证，
**不外推**规模/渐近/交叉点。

## 7. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | char 级 enwik8（`data/enwik8_full.npz`，前 90M train / 末 10M eval / vocab 206） |
| 合成腿任务 | `tools/evalkit/longrange_tasks.py::var_recall_batch`（**未改工具**）；距离档 T=1024: {256,300,512,768,1000}；chance = 1/64；局部先验 k=8 |
| 入口脚本 | `run.py`（preflight / train / eval / report）；`smoke.py`（CPU 不变量）；`mutate.py`（单测变异） |
| 代码改动 | **无**——本实验不改 `pathlm/`、`train_depth_ar.py`、`tests/`（只**新增** `tests/test_longT_1024.py` 与实验目录文件） |
| 可失败单测 | 见 `tests/test_longT_1024.py`：非退化点、visible_frac 闭式、同相对刻度、位置表参数增量、占比公式、可变距离合成任务合法性 |
| 变异验证 | `mutate.py`：**monkeypatch** 方式破坏掩码/占比 ⇒ 必须抓红 ⇒ 撤销（**不写共享文件**，见 §9） |
| 落盘 | `results.json` / `config.json` / `<arm>-train_log.jsonl` / `<arm>-gate_log.jsonl`（逐 200 步原始日志，**已入库**，见 §10） / `analysis.md` / `MODEL.md` / `smoke/smoke.json` / `smoke/cpu_smoke.json` / `mutation_report.json` / `reconcile_report.json` |

## 8. 失败决策树

| 症状 | 诊断顺序 | 处置 |
|---|---|---|
| 单测红 / 变异不抓红 | ① 实现 | 先修实现；未排除前不判机制 |
| 干预死（`window_mask_probe` diff = 0 / `alive=False`） | ① 实现 | 活性门不过不判机制 |
| 两臂 `\|Δ\|` < 0.014 | ② 该 T 无头寸 | 如实报「窗口不亏」；**不加第二个机制** |
| 窗口劣化 ≥0.014 | ④ 设计在该设定下净亏 | 记为否证/下界，记录窗口档与幅度 |
| 显存 >5GB net | ③ 硬件 | 按 §20.2 记账或降 batch（**两臂同配方**）；不擅自改判据 |
| 结果不可重复 | ⑤ 方差 | 按 Δ<0.014 不可排序解读；n=1 |

## 9. 文件所有权与共享面声明（SCHEMA §22）

- 本实验**只写**本实验目录 + **新增** `tests/test_longT_1024.py`。
- **不写** `pathlm/depth_ar.py`、`train_depth_ar.py`、`tests/test_depth_ar.py`（共享实现）。变异验证
  用**进程内 monkeypatch**（§7），运行前后记录共享文件 sha256 断言未变。
- GPU 取锁遵守 §20/§20.1（多臂**按臂释放**，留 ≥60s 窗口）；峰值 ≤5GB net。

## 10. 产出

本目录下 `README.md`(本文件) / `run.py` / `smoke.py` / `mutate.py` / `config.json` /
`results.json` / `analysis.md` / `MODEL.md` / `reconcile_numbers.py` / `reconcile_report.json` /
`mutation_report.json` / `smoke/smoke.json` / `smoke/cpu_smoke.json` / `<arm>-train_log.jsonl`；
`tests/test_longT_1024.py`。

> 契约副本标注（SCHEMA §17.6）：§0 的 Y2 原文段为**逐字**抄录；§1–§10 为执行者实现/执行细节
> （未改判据线）。
