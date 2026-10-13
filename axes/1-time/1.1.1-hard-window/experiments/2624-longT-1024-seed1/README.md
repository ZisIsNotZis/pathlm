---
name: 2624-longT-1024-seed1
ticket: 41-Z1
date: "2026-10-03"
grid: A6-mixing-operator（长 T 第三点 T=1024 的 seed=1 双臂重跑）
engine: depth-ar `--dense`（stock decoder-only transformer，7,159,296 params）
budget: "gpu_wall_budget: 30 min（含冒烟 + 成本基准 + 2 训练臂 + 电池 + 合成腿）；峰值 ≤5GB net"
status: judged
---

# 2624 · 长 T 第三点 T=1024 的 **seed=1 双臂重跑**（A6 · 7.16M · depth-ar/dense）

> 本文件即本实验的本地契约。§0 把 `.scratch/41-followups/issues/01-followups.md` 的「Z1」段
> **逐字**抄下（判据线一字未改）；§1 起是实现与执行细节。
> 原始结果：`results.json`；判读：`analysis.md`；权重清单：`MODEL.md`。

## 0. 契约原文（逐字）

> ## Z1 · T=1024 seed=1 双臂重跑（≈20–30 min GPU）
> - 目录：`axes/A-mixing/A6-mixing-operator/experiments/2624-longT-1024-seed1/`
> - 判据（预登记）：① `full` 臂的 gnorm 爆炸（上轮 1740.9）在 seed=1 是否**复现**——
>   复现 ⇒ 记「**long-T 训练不稳定**」为工程债、该点**永久不可归因**；不复现 ⇒ 该点可给干净 Δ、**重判三点趋势**；
>   ② 两臂同配方（含 bs，须与 Y2 一致：bs8）；③ 必附答案 CE 斜率与显存；④ n=2 后仍禁外推。
> - 不要求两臂都重跑（可只跑 full@seed1 判爆炸复现性），但若预算允许跑双臂则给 Δ 的 n=2。

**主 Agent 补充（随派发给出，同样视为契约）**：
- GPU 取锁（按臂释放、臂间 ≥60s 窗口）；峰值 ≤5GB net（超线按 §20.2 三件套记账）；
  **先测步时再定预算**（§20.3：卡被外部占满时缩步数保臂间同刻度并声明快照口径，不砍臂/不换配方/不等卡）。
- 派发前做能力面分析（§22.8）；长程类一律用 `tools/evalkit/longrange_tasks.py` 合成任务。
- 单因素 + 逐字段对照（含 seed）；活性量每 200 步；每个语义开关可失败单测 + 变异验证（抓红且逐字节还原）。
- 三道对账 + 禁外推；数字与 `results.json`/日志逐一对应 + 自带 `reconcile_numbers.py`。
- 结束前 `pytest tests/ -q` 全绿（基线 **431**）；权重不入库；只写授权文件。

**相关规范**：`SCHEMA.md` §14/§14b、§17.11（禁外推）、§17.14（判据基准唯一）、§17.16（两点同刻度
非退化）、§17.17（净和不可排序给分量）、§18、§19、§20+§20.1+§20.2+§20.3、§21、§22+§22.8、§23。

## 1. 命题（一句话、可证伪）

在与 2623（ticket 40-Y2）**逐字段相同**的配方上（stock dense，d=256/L=8/h=8/mlp=4，bs8，
12000 步 / 98.3M token，lr 1e-3，T=1024，两臂仅 `train_window` ∈ {0, 512} 不同），
**只把 seed 从 0 改成 1**，问：

> seed0 的 `full-1024` 臂那个 `gnorm_max = 1740.92` 的梯度爆炸（而 `win512-1024` 只有 7.35），
> 是**训练不稳定**（会复现 ⇒ 长 T 工程债、该点永久不可归因），还是 **seed 特异**（不复现 ⇒
> 该点可给干净 Δ、可重判三点趋势）？

## 2. 单因素声明与能力面分析（SCHEMA §22.8）

### 2.1 单因素

- **本实验相对 2623 的唯一变化 = seed（0 → 1）**；**两臂之间**的唯一变化 = 训练期可见窗口
  （`train_window` ∈ {0, 512}），seed 两臂相同（= 1）。
- 基底是**主流标准 decoder-only transformer**（dense：无 carry 门、无随机退出、仅最终深度 CE、
  tied 读出、`topology=sequential`）。三条 RAP 共享底座偏离在两臂间完全相同，不作变量。
- 窗口掩码是**位置式**（`keep[i,j] = (j≤i) ∧ ((i−j) < w)`），训练（`DepthARModel.forward`）与
  评测（`retention.masked_logits`）共用 `pathlm.depth_ar.window_anchor_keep_mask`（单测钉住）。
- **非必要性例外**：无新增开关、未改 `pathlm/` 与 `train_depth_ar.py`（见 §9）。

### 2.2 能力面分析（派发前逐条核对，脚本 `capability_check.py`）

`grep -n "ValueError\|互斥" pathlm/depth_ar.py` 列出的与本组合相关的校验（**在
`DepthARModel.__init__` 内，不在 `DepthARConfig`**）：

- `train_window > 0 and not dense` ⇒ raise；本实验 `--dense` ✅
- `train_window > 0 and topology != "sequential"` ⇒ raise；本实验缺省 `sequential` ✅
- `attn_sparse_k ⊕ train_window`、`attn_linear ⊕ train_window`、`carry_rule ⊕ train_window`、
  `n_kv_heads<n_heads ⊕ train_window` ⇒ 本实验**全不开**这些算子，无互斥 ✅
- `--seed` 是一等参数（`train_depth_ar.py:545`），只改初始化与数据抽样 ⇒ 无组合限制 ✅
- **正向/反向构造实测**（`controls/capability_analysis.json`）：两臂正向构造成功（参数同为
  7,159,296）；`dense=False + train_window=512` 与 `topology=parallel + train_window=512`
  **被拒绝**（证明校验活着，不是死代码）。
- **结论：契约要求的组合（full / 窗口 512 @ T=1024 @ seed1）在引擎中可达，无未测因子。**

### 2.3 同相对刻度与非退化（SCHEMA §17.16）

三点 = `window128@T=256` / `window256@T=512` / `window512@T=1024`，窗口占比**同为 50%**；
第三点 `window512@T=1024` **非退化**（`w=512 < T=1024`，与精确因果掩码不等）——单测钉住。
**禁止**用退化点（如 `window1024@T=1024`，恒等）充当点。

## 3. 指标表（逐项分析）

| 维度 | 指标 | 与谁比 |
|---|---|---|
| 解决什么问题 | seed0 的「基准臂自身爆炸 ⇒ 不可归因」能否在 seed=1 下被判掉 | 2623（seed0）的同一臂 |
| **主判据（Z1 ①）** | `gnorm_max` 复现判定（R1 绝对量级 / R2 相对量级） | seed0 的 1740.92 vs 7.35 |
| 质量（Z1 ③） | Δbpc（native 工作点：窗口臂用自身训练掩码，full 臂全因果） | 本点自己的 full（§17.14 基准唯一） |
| 代价（质量·并列） | release 口径 Δ（窗口臂在全可见下评测） | 同上 |
| 有效性 | Δ + 方差界（Δ < 0.014 不可排序，`docs/methodology.md §2`） | SCHEMA §12 判据线 |
| **判据损失斜率（Z1 ③）** | 训练日志 loss 的末窗/晚第三窗/早窗斜率（nats/1k 步，§17.15/§20.3） | 各臂自己 |
| 显存（Z1 ③） | 训练峰值（`torch.cuda.max_memory_allocated` + `nvidia-smi` 子进程峰值） | 5GB net 礼貌线 |
| 种子活性（§10） | 同配置 seed0/seed1 的初始参数是否不同（`smoke/cpu_smoke.json`） | 0（死因子） |
| 干预活性（§10） | `window_mask_liveness`（每 200 步随日志）+ `window_mask_probe`（>0） | 0（死干预） |
| 稳定性（必报） | 逐臂 gnorm max/mean/尖峰位置、NaN、grad_skips | 各臂日志 |
| 算力（并列） | 注意力孤岛/整模型实测步时比、`r_time`、峰值比 | 公式 `512T/(512T+786K)` = 40.0% |
| 长程腿（§19） | 合成 `var_recall`（距离档 × 臂）top-1 准确率 + chance + 局部先验抗性 | 同任务 chance / 局部先验 |
| 组合/净和（§17.17） | 不适用（本实验非「净和不可排序」场景：主判据是复现判定，非净效应） | — |

**判读规则**：先过「实现/算子」门（单测 + 冒烟 + 变异验证 + 活性量），再判爆炸复现，最后才提质量。

## 4. 判据线（预先写下，逐字来自契约 Z1，不得修改）

### 4.1 预登记的判据点与**操作化阈值**

1. **复现判定（Z1 ①）**——seed0 的签名 = `full` gnorm_max **1740.92**、`win` **7.35**（比 236.9）。
   预先写死两条：
   - **R1（绝对量级）**：seed1 的 `full` `gnorm_max > 100`（seed0 的 1740.92 的同一量纲下界）。
   - **R2（相对量级）**：seed1 的 `full.gnorm_max > 20 × win.gnorm_max`。
   - **R1 ∧ R2 ⇒ `reproduced`**：「long-T 训练不稳定」记**工程债**，T=1024 点**永久不可归因**。
   - **¬R1 ∧ ¬R2 ⇒ `not_reproduced`**：seed0 那次归**seed 特异**，本点可给**干净 Δ**、
     **重判三点趋势**。
   - **恰过一个 ⇒ `partial`**：**保守**按「不可归因」处置，**不给**干净 Δ，需第三 seed 才能再判。
2. **两臂同配方（Z1 ②）**：`bs8`（与 Y2 一致）、12000 步、98.3M token、seed 两臂同为 1；
   逐字段列表对照（`config.json::arms[].fields`），只允许差 `train_window`。
3. **判据损失斜率 + 显存（Z1 ③）**：报训练日志 loss 的**末窗**（最后 ~800 步）、**晚第三窗**、
   **早窗**斜率（nats/1k 步），并报逐臂显存峰值。
   - **口径声明（§17.15/§20.3）**：本实验**没有合成「答案行」**——判据损失 = enwik8 CE
     （`bpc = CE/log2`），故「决定判据的那个量」即 enwik8 CE；因引擎不落中间 checkpoint，
     斜率取**训练日志的 loss 序列**（每 200 步一行）作为判据损失的代理，并另报**同一 checkpoint
     的评测流 bpc**（水平值）。**已如实声明该代理口径，不冒充「答案行 CE」。**
4. **n=2 仍禁外推（Z1 ④）**：本实验与 2623 构成 n=2（同配方、仅 seed 不同）⇒ 只给两个读数与
   均值，**不拟合趋势、不报置信区间、不外推**（SCHEMA §17.11）。
5. **主判据唯一（§17.14）**：质量 Δ 的基准 = **本点自己的 `full` 臂**（native 口径），
   不得与 X3 两点混用（X3 用 bs32，本线 bs8，**batch 不同 ⇒ 不可直接并列**，只作描述）。

### 4.2 符号与带宽

`Δ = bpc_full − bpc_window`（SCHEMA §17.3：正 = 窗口更好），全篇统一。
方差界：|Δ| < 0.014 ⇒ 该点「不亏（噪声带内）」（`docs/methodology.md §2`）。

## 5. 既有做法（prior art，SCHEMA §18）

> 本环境**联网不可用**。外部内容一律 **`记忆·未核`**，只作设计方向、**不得当判据**。

1. **谁做过**：滑动窗口注意力（Longformer / Mistral 的 SWA）是长上下文标准件；训练稳定性与
   「梯度尖峰 ⇒ loss 上冲」在长序列/低精度训练里是常见工程现象（公开报告多为对策，非受控对比）。
2. **为什么这么设计**：注意力 FLOPs 随 T² 增长 ⇒ 长 T 上按位置丢弃远端键把注意力算力降到
   O(T·w)；梯度尖峰则用 clip / 预热 / 重初始化处理。
3. **它的实测**：公开报告**不报** 7M 单因素、同 token 预算下「窗口 vs 全因果」的稳定性差异，
   也不报「同一 seed 是否复现」。`记忆·未核`，不可当判据。
4. **我们的增量**：A6 已在 T=256 实测稀疏（内容 top-k）**净亏**（k=32 Δ=−0.0166，成本 2.2×）；
   ticket 39-X3 在 T=512 实测窗口算子「不亏」（Δ=−0.0087，带内）；ticket 40-Y2 的 T=1024 点因
   **基准臂自身 gnorm 爆炸**判「不可归因」。本实验做的正是 Y2 明确留下的**唯一未测项**：
   同一配方**换 seed** 判爆炸复现性——这是文献里几乎不报的**单因素稳定性**观测。

**检索词**（供将来可核证的 run）：`sliding window attention long context quality`、
`gradient spike sequence length transformer training instability`、`seed variance small transformer`。

## 6. 成本与 scope

| 项 | 值 |
|---|---|
| `gpu_wall_budget` | **30 min**（含冒烟 + 成本基准 + 2 训练臂 + 电池 + 合成腿） |
| 预算过程（§8） | 开跑前用 `--stage preflight` **实测步时/显存**再锁定（禁按公式估）——实测值见 `results.json::cost` 与 `analysis.md` |
| 训练臂（2） | `full-1024-seed1`(12000 步) / `win512-1024-seed1`(12000 步)，均 **bs8**、**seed1** |
| **token 口径** | **同 token 预算 98.3M**（与 2622 的 T=256/512 四点、2623 的 seed0 点同：12000×8×1024）；另报步数 |
| 峰值显存 | ≤5GB net（共享卡；先测再定）；**本实验用 bs8**（bs32/bs16 的窗口臂在 2623 已 OOM）。本机背景进程（两个 `llama-server`）常驻占 ~18–19 GiB ⇒ 全卡 free 仅 ~2–5 GiB，**§20.2 的「跑前后余量 ≥8GB」在本机不可满足**，如实记账、不碰别人进程 |
| 可中断性 | 是（每 200 步落日志；`--skip-train` 可用已有 checkpoint 只跑电池） |
| 冒烟计划 | `--stage preflight`：两臂各 20 步（实测步时/显存/活性，**不含任何 20 步外推时间字段**，SCHEMA §17.5）+ bs8 步时基准 |
| **预算不够时砍什么（§20.3）** | 顺序：① 缩**步数**（`PATHLM_2624_STEPS`，两臂同刻度，并在报告首段声明「Nk 快照口径、与 2623 的 12k 不可比」）→ ② 合成腿距离档 → ③ release 口径 → ④ `--cost-iters`。**不砍臂、不换配方、不等卡、不砍主判据（爆炸复现）** |

**若 30 min 装不下双臂**：契约允许只跑 `full-1024-seed1` 判爆炸复现性（Z1 原文明确），
并在报告里写明这一取舍；但本机实测步时若允许，则跑**双臂**给 Δ 的 n=2。

规模纪律（SCHEMA §7）：单卡 RTX 4090 24GB 多任务共享，GPU 串行；只做 7.16M 原理验证，
**不外推**规模/渐近/交叉点。

## 7. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | char 级 enwik8（`data/enwik8_full.npz`，前 90M train / 末 10M eval / vocab 206） |
| 合成腿任务 | `tools/evalkit/longrange_tasks.py::var_recall_batch`（**未改工具**）；距离档 T=1024: {256,300,512,768,1000}；chance = 1/64；局部先验 k=8 |
| 入口脚本 | `run.py`（preflight / train / train-all / eval / report）；`smoke.py`（CPU 不变量）；`mutate.py`（变异）；`capability_check.py`（§22.8） |
| 代码改动 | **无**——本实验不改 `pathlm/`、`train_depth_ar.py`（只**新增** `tests/test_longT_1024_seed1.py` 与实验目录文件） |
| 可失败单测 | `tests/test_longT_1024_seed1.py`：seed 单因素、复现判据（3 档）、带宽边界、降速同刻度、非退化点、同相对刻度 |
| 变异验证 | `mutate.py`：进程内 monkeypatch 破坏掩码/带宽/复现判据/降速旋钮/seed ⇒ 必须抓红 ⇒ 撤销（不写共享文件） |
| 落盘 | `results.json` / `config.json` / `<arm>-train_log.jsonl` / `<arm>-gate_log.jsonl` / `analysis.md` / `MODEL.md` / `smoke/smoke.json` / `smoke/cpu_smoke.json` / `controls/capability_analysis.json` / `mutation_report.json` / `reconcile_report.json` |

## 8. 失败决策树

| 症状 | 诊断顺序 | 处置 |
|---|---|---|
| 单测红 / 变异不抓红 | ① 实现 | 先修实现；未排除前不判机制 |
| 干预死（`window_mask_probe` diff = 0 / `alive=False`） | ① 实现 | 活性门不过不判机制 |
| seed 因子死（seed0 与 seed1 初始参数相同） | ① 实现 | seed 无效 ⇒ 停下请示（实验前提不成立） |
| `partial`（复现判据恰过一个） | ② 判定粒度 | 保守记「不可归因」，不给干净 Δ；报告写明需第三 seed |
| 显存 >5GB net | ③ 硬件 | 按 §20.2 三件套记账（不改配方——改配方会破坏与 2623 的可比性） |
| 卡被外部占满（步时涨） | ③ 硬件（§20.3） | 缩步数、两臂同刻度、声明快照口径；**不砍臂/不换配方/不等卡** |
| 30 min 装不下双臂 | ④ 预算 | 按 §6 取舍：只跑 `full-1024-seed1` 判复现性，报告写明 |
| 结果不可重复/怪异 | ⑤ 方差 | 按 n=2 只描述、禁外推解读 |

## 9. 文件所有权与共享面声明（SCHEMA §22）

- 本实验**只写**本实验目录（`runs/`、`smoke/`、`controls/` 及根文件）+ **新增**
  `tests/test_longT_1024_seed1.py`。
- **不写** `pathlm/depth_ar.py`、`train_depth_ar.py`、`tests/test_depth_ar.py`（共享实现）。
  变异验证用**进程内 monkeypatch**（§7），运行前后记录共享文件 sha256 断言未变。
- 与 2623（seed0）的关系：**跨 run 引用**其 `results.json`（只读，见 §4.1 第 5 条与 §9.1）。
- GPU 取锁遵守 §20/§20.1（多臂**按臂释放**，留 ≥60s 窗口）；峰值 ≤5GB net。

### 9.1 对照复用规则（SCHEMA §9）

2623 的 seed0 点是**第二个 seed 的数据点**，不是「节省算力的对照臂」：

1. **实现层 equality**：同一 `train_depth_ar.py`/`pathlm/depth_ar.py`，且本线运行前后记录
   sha256（`config.json::code_fingerprints`）与 2623 的指纹比对；
2. **配置层 equality**：逐字段列表对照（d/layers/heads/mlp/bs/steps/lr/exit-mode/data/切分/
   vocab），**唯一不等字段 = seed**，在 `analysis.md` 显式列出；
3. **口径标注**：batch 与 X3 两点不同（bs8 vs bs32），三点只描述、禁外推；
4. 若任一字段不等（除 seed）⇒ 回退到自包含双跑（本实验本就是自包含双跑，seed0 只作第二点）。

## 10. 产出

本目录下 `README.md`(本文件) / `run.py` / `smoke.py` / `mutate.py` / `capability_check.py` /
`config.json` / `results.json` / `analysis.md` / `MODEL.md` / `reconcile_numbers.py` /
`reconcile_report.json` / `mutation_report.json` / `smoke/smoke.json` / `smoke/cpu_smoke.json` /
`controls/capability_analysis.json`；`tests/test_longT_1024_seed1.py`。

> 契约副本标注（SCHEMA §17.6）：§0 的 Z1 原文段为**逐字**抄录；§1–§10 为执行者实现/执行细节
> （未改判据线）。
