---
name: 2625-longT-stability
ticket: 42-W1
date: "2026-10-03"
grid: A6-mixing-operator（长 T 训练稳定性修复：工程债）
engine: depth-ar `--dense`（stock decoder-only transformer，7,159,296 params）+ 实验本地候选补丁驱动器
budget: "gpu_wall_budget: 70 min（含冒烟 + 诊断臂 + 候选筛查 + 两 seed 确认 + 短 T 不回归）；峰值 ≤5GB net"
status: judged
---

# 2625 · long-T 训练稳定性修复（A6 · 7.16M · depth-ar/dense）

> 本文件即本实验的本地契约。§0 把 `.scratch/42-longt-stability/issues/01-longt-stability.md`
> 的「W1」段**逐字**抄下（判据线一字未改）；§1 起是实现与执行细节。
> 原始结果：`runs/<arm>/results.json` 与 `runs/<arm>/train_log.jsonl`；
> 判读：`stability_report.md`；对账：`reconcile_report.json`；变异：`mutation_report.json`。

## 0. 契约原文（逐字）

> ## W1 · long-T 训练稳定性修复（≈50–70 min GPU）
> - 目录：`axes/1-time/1.1.1-hard-window/experiments/2625-longT-stability/`
> - 事实（已确证，两 seed）：T=1024 的 `full` 臂 gnorm 爆炸（seed0 1740.9 / **seed1 29578.6**，跳步 0/3，爆炸点 ~9800/~6200），
>   而 `win` 臂两 seed 均稳（7.35/7.66）。**这挡住了整条长 T 线（A6 的入场券）。**
> - 任务：**只诊断与修复稳定性**（不改任务/不改掩码语义）。候选手法（各为单因素，先 CPU/短冒烟筛查）：
>   grad clip 策略（当前阈值/是否 per-parameter）、warmup 长度、LR 调度、初始化（pos/embed 尺度）、
>   以及「是否与 T² 相关的 logits 尺度/attention 熵」有关。
> - **判据（预登记）**：① 找到**至少一个**使 `full@T=1024` 在 seed∈{0,1} 两跑均不再爆炸（gnorm_max < 100 且无跳步）
>   的最小改动；② 该改动**不改变短 T（256/512）的行为**（逐位或 Δbpc < 0.014 之一，须写明用哪个口径）；
>   ③ 明确写出「爆炸的机制解释候选」与「排除了哪些」；④ 失败也要报（若找不到，则把该工程债与已试清单写清，
>   并给出「要几次实验才能收口」的估计）。
> - 交付含：`stability_report.md`（机制假设→实验→结论表）+ 每个改动臂的 gnorm/loss 曲线数据。

**主 Agent 补充纪律（随派发给出，同样视为契约）**：
- GPU 取锁、**按臂释放并留 ≥60s 窗口**；峰值 ≤5GB net（超线按 §20.2 三件套记账）；
  **先测步时再定预算**（§20.3）。
- **派发前做能力面分析**（§22.8）：`grep -n "ValueError\|互斥" pathlm/depth_ar.py` 核对组合。
- 长程类用 `tools/evalkit/longrange_tasks.py` 合成任务；主判据难度档须在训练分布内（§17.18）。
- 单因素 + 逐字段对照（含 seed）；活性量每 200 步；每个语义开关可失败单测 + 变异验证（抓红且逐字节还原）。
- 三道对账 + 禁外推；数字与 `results.json`/日志逐一对应 + 自带 `reconcile_numbers.py`（报告落盘）。
- 结束前 `python3 -m pytest tests/ -q` 全绿（基线 **468**）；权重不入库；只写授权文件。
- **提交前验证 staged 列表**（AGENTS.md）。

**相关规范**：`SCHEMA.md` §14/§14b、§17.11（禁外推）、§17.14（基准唯一）、§17.16（两点同刻度非退化）、
§17.17（净和不可排序给分量）、**§17.18（判决腿须在训练分布内）**、§18、§19、§20+§20.1+§20.2+§20.3、
§21、§22+§22.8、§23（七问）。

## 1. 命题（一句话、可证伪）

在与 2623/2624 **逐字段相同**的长 T 配方（stock dense，d=256/L=8/h=8/mlp=4，bs8，T=1024，
lr 1e-3，12000 步 / 98.3M token，全因果）上，**只改一个训练/结构开关**，问：

> 是否存在**至少一个**单因素改动，使 `full@T=1024` 在 **seed0 与 seed1 两跑都**满足
> `gnorm_max < 100` 且 `grad_skips == 0`，且该改动在 T=256/512 上**不改变行为**？

已知事实（2623/2624，本实验的「前」态，跨 run 引用、不改数字）：

| 臂 | seed | `gnorm_max` | `gnorm_mean` | `grad_skips` | 爆炸点 |
|---|---|---|---|---|---|
| `full-1024` | 0 | **1740.92** | 6.533 | 0 | ~step 9800 |
| `full-1024-seed1` | 1 | **29578.57** | 22.946 | **3** | ~step 6200 |
| `win512-1024` | 0 | 7.35 | 0.562 | 0 | — |
| `win512-1024-seed1` | 1 | 7.66 | 0.585 | 0 | — |

## 2. 单因素声明与能力面分析（SCHEMA §22.8）

### 2.1 单因素

- **唯一变量 = 候选补丁**（每个候选一次只开一个）；两臂之间/两 seed 之间的其余字段逐位相同
  （T=1024、bs8、`--dense`、`topology=sequential`、`exit-mode uniform`、lr 1e-3、12000 步、
  enwik8 前 90M/末 10M、`torch.manual_seed(seed)`）。
- 候选（`stability_driver.py::parse_candidate`）：
  | 候选名 | 干预 | 施加点 |
  |---|---|---|
  | `none` | 基线（逐字段 = 2623/2624） | — |
  | `temp<τ>` | 注意力 logits 除以 τ（q ← q/τ） | 每个 block 的 `qkv` post-hook |
  | `qk-norm` | q/k 按 head 做 L2 归一（×√d_h，无新参数） | 同上 |
  | `wd<v>` | AdamW `weight_decay` 实参替换 | 进程内替换 `torch.optim.AdamW` |
  | `warmup<N>` | 预热从 200 步拉长到 N 步（余弦公式逐字复用） | 进程内替换 `LambdaLR` |
  | `clip<v>` | `clip_grad_norm_` 阈值替换 | 进程内替换 `clip_grad_norm_` |
  | `initpos<σ>` / `initemb<σ>` | 构造后重设 `pos_embed` / `embed` 初始化 std | 构造后 `nn.init.normal_` |
- 基底是**主流标准 decoder-only transformer**（dense：无 carry 门、无随机退出、仅最终深度 CE、
  tied 读出、`topology=sequential`）。
- **非必要性例外**：**不改任何共享文件**（`pathlm/`、`train_depth_ar.py`、`tests/test_depth_ar.py`
  一字不动）。候选补丁只打在实验本地驱动器里构造的**对象**上；`train()` 逐字复用（见 §7）。

### 2.2 能力面分析（派发前逐条核对）

`grep -n "ValueError\|互斥" pathlm/depth_ar.py` 共 76 条命中；与本组合相关的校验：

- `train_window > 0 and not dense` ⇒ raise；`train_window > 0 and topology != "sequential"` ⇒ raise
  —— 本实验 `--dense` + 缺省 `sequential` ✅
- `attn_sparse_k ⊕ train_window`、`attn_linear ⊕ train_window`、`carry_rule ⊕ train_window`、
  `n_kv_heads<n_heads ⊕ train_window` ⇒ 本实验全不开这些算子（full/win 两臂之外不加任何算子）✅
- 与 2623/2624 同组合（`dense + sequential`，full 臂 `train_window=0`）⇒ 已在引擎中跑通 ✅
- **候选补丁本身不新增引擎 flag**：`temp`/`qk-norm`/`init*` 是模块级 forward hook 与参数重初始化，
  `wd`/`warmup`/`clip` 是进程内实参替换 —— 都不触碰 `DepthARConfig` 的校验面 ✅
- **结论：契约要求的组合与全部候选在引擎中可达，无未测因子。**

### 2.3 同相对刻度与非退化（SCHEMA §17.16）

本实验不改 T=256/512 两点的相对刻度（`window128@T=256` / `window256@T=512`，均 50%）；
短 T 不回归检查只比对**同一 T 的同一臂**在补丁前后是否改变（见 §4.2）。

## 3. 指标表（逐项分析）

| 维度 | 指标 | 与谁比 |
|---|---|---|
| 解决什么问题 | 长 T 线（A6 的入场券）能否在 `full@T=1024` 上稳定训练 | 2623/2624 的 `full-1024`（两 seed） |
| **主判据（①）** | 逐臂 `gnorm_max`、`grad_skips`（预登记阈值 <100 且 =0） | 1740.92 / 29578.57 / 0 / 3 |
| **回归判据（②）** | 短 T（256/512）行为：逐位或 Δbpc（口径见 §4.2） | 2622 的 4 个臂（T=256/512） |
| **机制（③）** | 训练中逐层探针：残差流范数、`qkv` 权重范数、注意力 logits RMS/max/top1/熵 | 各臂自己的轨迹 |
| 代价（质量） | 末态 clean bpc（native 口径，同点自己的 full） | §17.14 基准唯一 |
| 稳定性（必报） | `gnorm` max/mean/尖峰位置、NaN、`grad_skips`、loss 突跳 | 各臂日志 |
| 显存 | 训练峰值（`torch.cuda.max_memory_allocated` + 跑前后 `nvidia-smi` free） | 5GB net 礼貌线 |
| 步时 | 实测 `seconds_per_step`（先测再定预算，§20.3） | 2623 的 45 ms/step |
| 长程腿（§19） | 合成长程任务 `var_recall`；**主难度档须在训练分布内**（§17.18） | 同任务 chance / 局部先验 |

**判读规则**：先过「实现/候选真的生效」门（单测 + 变异 + 活性量），再判主判据，最后才提质量。

## 4. 判据线（预先写下，逐字来自契约 W1，不得修改）

### 4.1 主判据（① 稳定性修复）

对**每个**候选，在 T=1024、bs8、seed∈{0,1} 上训练，读数取 `results.json::gnorm_monitor`：

- **通过** ⇔ 两 seed 均 `gnorm_max < 100` **且** `grad_skips == 0`；
- 通过者中按「改动量级最小」排序（`warmup<N>` 的 N 越小、`wd` 越接近 0.01、`temp` τ 越接近 1
  越「小」；`qk-norm` 是结构改动，量级最大）。
- **至少一个通过 ⇒ 判据①成立**；全部失败 ⇒ 走 §8 的失败路径（④）。

### 4.2 回归判据（② 短 T 不改变行为）

口径**预先写死**（二选一，须写明用哪个）：

- **逐位口径（优先）**：若候选在 T ≤ 512 的代码路径上是**恒等**（例如只对 T>512 生效的改动），
  则 T=256/512 的行为**逐位不变**，用「同一 seed 同一配置重跑，与 2622 已有 `train_log.jsonl`
  逐行逐字段相等」来证（`reconcile_numbers.py` 的第 4 道对账）。
- **Δbpc 口径**：若候选在 T ≤ 512 上确实改变计算，则用 2622 的 4 个臂作基准，在**同一评测流**
  （`window_seed=2`、20 批 × bs16）上报 `Δbpc = bpc_候选 − bpc_基准`；`|Δ| < 0.014` 记「不改变
  （噪声带内）」，`|Δ| ≥ 0.014` 记「改变」并如实报方向与幅度。

### 4.3 机制（③）与失败（④）

- **③**：报告必须写出「机制解释候选」清单与「排除记录」——每条排除都要有**证据**，
  并明确哪些解释**本轮未排除**（n=1/2、单一尺度，禁外推）。
- **④**：若②失败或全部候选①失败，把已试清单（候选 × seed × 步数 × 读数）与
  「要几次实验才能收口」的估计写进 `stability_report.md`。

### 4.4 禁外推与基准唯一

- 只描述本尺度（7.16M / 98.3M token / T=1024 / bs8）的观测；**不外推**到其它 T、其它规模、
  其它 batch。
- 质量 Δ 的基准 = **本点自己的 `full` 臂**（同 seed、同 T、同 bs）；不得与 X3（bs32）混用。

## 5. 既有做法（prior art，SCHEMA §18）

> 本环境**联网不可用**。外部内容一律 **`记忆·未核`**，只作设计方向、**不得当判据**。

1. **谁做过**：长序列训练中的「loss spike / 梯度爆炸」是公开的工程现象，常见对策是
   梯度裁剪、预热、QK-Norm（对 q/k 做归一）、logit soft-cap、权重衰减与重初始化；
   滑动窗口注意力（Longformer / Mistral SWA）是长上下文标准件。
2. **为什么这么设计**：注意力 logits 的**极值随可见键数增长**（T 越大，逐行 softmax 的
   候选越多、最大 logit 的尾部越靠右）⇒ 注意力分布可变得极尖，训练曲率上升；
   QK-Norm / 温度 / soft-cap 直接压住这个尺度；预热与小步长则避免早期进入尖锐区。
3. **它的实测**：公开报告给的是**对策与配方**，不报「同尺寸、同 token 预算、单因素、
   两 seed 复现」的稳定性对照。`记忆·未核`，不可当判据。
4. **我们的增量**：2623/2624 在 **7.16M / 98.3M token / T=1024 / bs8** 上实测「全因果臂
   seed0 与 seed1 都炸（1740.9 / 29578.6）、窗口臂两 seed 都稳」；本实验做的是文献里几乎
   不报的**单因素稳定性诊断与最小修复**，并给「改动是否波及短 T」的回归读数。

**检索词**（供将来可核证的 run）：`attention logit growth sequence length training instability`、
`QK-norm long context stability`、`loss spike transformer warmup gradient clipping`。

## 6. 成本与 scope

| 项 | 值 |
|---|---|
| `gpu_wall_budget` | **70 min**（诊断臂 + 候选筛查 + 两 seed 确认 + 短 T 不回归；含锁等待余量不计） |
| 预算过程（§20.3） | 开跑前 `--stage preflight` 实测步时：**44 ms/step**（T=1024，bs8，与 2623 的 45 ms/step 一致）⇒ 7000 步 ≈ 5.2 min、12000 步 ≈ 9 min |
| 臂数 | 1 诊断 + 4 筛查（seed1，6500 步）+ 2 确认（两 seed，12000 步）+ 短 T 检查 |
| **token 口径** | 确认臂 = 12000×8×1024 = **98.3M**（与 2623/2624 同）；筛查臂 = 6500 步（≈53M token，**快照口径**，只用于「是否爆炸」的判定，不报质量） |
| 峰值显存 | ≤5GB net（共享卡）；实测 full@T=1024 bs8 峰值 torch **0.91 GiB** |
| 可中断性 | 是（每 200 步落日志 + 探针；模型 checkpoint 结束时落盘） |
| 冒烟计划 | `--stage preflight`：1 臂 20 步（实测步时/峰值/探针，**不含任何 20 步外推时间字段**，§17.5） |
| 预算不够时砍什么（§20.3） | 顺序：① 候选筛查臂数（保留最有先验的）→ ② 筛查步数（≥6500，与爆炸点对齐）→ ③ 诊断臂。**不砍判据①的两 seed 确认、不换配方** |

规模纪律（SCHEMA §7）：单卡 RTX 4090 24GB 多任务共享，GPU 串行；只做 7.16M 原理验证，
**不外推**规模/渐近。

## 7. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | char 级 enwik8（`data/enwik8_full.npz`，前 90M train / 末 10M eval / vocab 206） |
| 训练入口 | `train_depth_ar.py::train`（**逐字复用**，未改一行）；实验本地驱动器 `stability_driver.py` 只做「构造 → 打补丁 → 调 train → 落探针」 |
| 驱动器保真度 | 同一 seed0/T=1024/bs8/`none` 的 step0 读数与 2623 逐位相同（loss 5.2979 / gnorm 7.28，见 `runs/preflight-20/train_log.jsonl`）；⚠️ **仅 step 0**——本任务的训练**非 run-to-run 逐位可复现**（同配方重跑 max\|Δloss\|=0.032、T=512 bpc 极差 0.0045），故全篇**不用逐位作判据**（见 `stability_report.md §4.1`） |
| 入口脚本 | `run.py`（preflight / diag / try / eval）、`stability_driver.py`（构造 → 打候选补丁 → 逐字调 `train` → 落探针）、`eval_bpc.py`（clean bpc）、`probe_attention_scale.py` / `probe_grad_scale.py` / `probe_checkpoint_scar.py`（CPU 诊断） |
| 代码改动 | **无**——不改 `pathlm/`、`train_depth_ar.py`（只新增本目录文件 + `tests/test_longT_stability.py`） |
| 可失败单测 | `tests/test_longT_stability.py`：候选解析、`temp` hook 真缩 q（且 τ=1 逐位还原）、`qk-norm` 后每头 L2 范数 == √d_h、warmup 调度在 N 步后与余弦逐位相等、探针键齐全、驱动器 step0 与 2623 逐位一致 |
| 变异验证 | `mutate.py`：进程内 monkeypatch 破坏候选 hook / 调度 / 探针 ⇒ 必须抓红 ⇒ 撤销（**不写共享文件**） |
| 落盘 | `runs/<arm>/{results.json,train_log.jsonl,probe.jsonl,model.pt(不入库)}` / `<arm>-train_summary.json` / `smoke/{attention_scale_probe,grad_scale_probe,ckpt_scar_probe,cpu_smoke}.json` / `stability_report.md` / `reconcile_report.json` / `mutation_report.json` / `config.json` / `MODEL.md` |

## 8. 失败决策树

| 症状 | 诊断顺序 | 处置 |
|---|---|---|
| 候选未生效（`temp`/`qk-norm` 前后读数逐位相同） | ① 实现 | 先修接线；未排除前不判机制 |
| 单测红 / 变异不抓红 | ① 实现 | 先修实现；未排除前不判机制 |
| 候选在 seed1 筛查就炸 | ② 候选无效 | 记「已试 + 失败读数」，换下一个候选（不改判据） |
| 全部候选①失败 | ④ 失败路径 | 报已试清单 + 收口所需实验数估计；**不得**改判据线 |
| 通过者改变短 T 行为（Δbpc ≥ 0.014） | ② 回归 | 如实判「判据②不成立」，该候选只记作「稳定性可行但波及短 T」 |
| 显存 >5GB net | ③ 硬件 | 按 §20.2 三件套记账（不改配方） |
| 卡被外部占满（步时涨） | ③ 硬件（§20.3） | 缩筛查步数、臂间同刻度；**不砍两 seed 确认、不换配方** |

## 9. 文件所有权与共享面声明（SCHEMA §22）

- 本实验**只写**本实验目录 + **新增** `tests/test_longT_stability.py`。
- **不写** `pathlm/depth_ar.py`、`pathlm/model.py`、`train_depth_ar.py`、`tests/test_depth_ar.py`
  （共享实现）。候选补丁全部在实验本地驱动器的进程内完成；变异验证用进程内 monkeypatch。
- 与 2623/2624/2622 的关系：**跨 run 只读引用**它们的 `results.json` / `train_log.jsonl`
  （基准与短 T 对照）。
- GPU 取锁遵守 §20/§20.1（**按臂释放 + ≥60s 窗口**）；峰值 ≤5GB net。

## 10. 产出

本目录下 `README.md`(本文件) / `stability_report.md` / `stability_driver.py` / `run.py` /
`eval_bpc.py` / `probe_attention_scale.py` / `probe_grad_scale.py` /
`probe_checkpoint_scar.py` / `smoke.py` / `mutate.py` /
`reconcile_numbers.py` / `config.json` / `reconcile_report.json` /
`mutation_report.json` / `curves.json` / `eval.json` / `MODEL.md` /
`smoke/*.json` / `runs/**`
（权重 `model.pt` 不入库）；`tests/test_longT_stability.py`。

> 契约副本标注（SCHEMA §17.6）：§0 的 W1 原文段与主 Agent 补充纪律为**逐字**抄录；
> §1–§10 为执行者实现/执行细节（未改判据线）。
