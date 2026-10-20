---
name: 2629-longT-2048-matched
ticket: W2
date: "2026-10-06"
grid: A6-mixing-operator（长 T 第四点 T=2048 · **同收敛复测**：两臂跑到 6000 步 = 与 T=1024 点同 step）
engine: depth-ar `--dense`（stock decoder-only transformer，7.42M@T=2048）+ 实验本地候选补丁驱动器（复用 2625 的 QK-Norm hook / 2628 的配方）
budget: "gpu_wall_budget: ≈15 min（含冒烟 + 两臂训练 + 电池 + 合成腿；训练本体 ≈10.7 min 是硬需求）"
status: judged
---

# 2629 · T=2048 同收敛复测（A6 · 7.42M · depth-ar/dense × qk-normfrom512 @6000 步）

> 本文件即本实验的本地契约。§0 把派发任务里的「W2 · GPU」段**逐字**抄下（判据线一字未改）；
> §1 起是实现与执行细节。原始结果：`runs/<arm>/results.json`、`runs/<arm>/train_log.jsonl`、
> `runs/<arm>/probe.jsonl`；判读：`analysis.md`；对账：`reconcile_report.json`；
> 变异：`mutation_report.json`；成本：`smoke/smoke.json`、`eval.json`；权重清单：`MODEL.md`。

## 0. 契约原文（逐字）

> 【W2 · GPU】T=2048 同收敛复测（≈10 min）
> 目录：axes/1-time/1.1.1-hard-window/experiments/2629-longT-2048-matched/
> 动机：45-L1 的 T=2048 是 **3000 步快照**（loss 1.54），而 T=1024 是 6000 步（loss 1.21）⇒ 四点**收敛度不等**，「窗口代价随 T 消失」不可作为同收敛结论。
> 干预：T=2048 的两臂（full / 窗口 1024）跑到 **6000 步**（与该系列其它点同收敛水平；先测步时，≈6–10 min），其余逐字段一致。
> 判据：① Δ2048@6000 的 band 分类（与 Δ2048@3000 对照，**两点只描述、禁外推**）；② 稳定性前置（gnorm<100、跳步 0）；③ 显存（若越线按 §20.2）；④ 明确回答「同收敛后『不亏』是否保持」。

**主 Agent 补充纪律（随派发给出，同样视为契约）**：
- GPU 取锁、按臂释放留 ≥60s；峰值 ≤5GB net（超线三件套记账）；先测步时再定预算（§20.3）。
- 长程类用 `tools/evalkit/longrange_tasks.py`；主判据难度档必须在训练分布内（§17.18）。
- 单因素 + 逐字段对照（含 seed）；活性量每 200 步；语义开关要有可失败单测 + 变异验证（抓红且逐字节还原）。
- 三道对账 + 禁外推；数字与 results.json/日志逐一对应 + 自带 reconcile_numbers.py。
- 结束前 `python3 -m pytest tests/ -q` 全绿（**基线 549**）；权重不入库；不碰 `catalog.yaml` / `INDEX.md` / `axes/**/README.md`（实验目录内手写 README 可以）。
- **提交前逐条核对**：报告/README 声称的每个交付物路径都要出现在 `git diff --cached --name-only` 里。
- 遇冲突/预算不够/锁异常/数字可疑 → 停下请示。

**相关规范**：`SCHEMA.md` §14/§14b、§17.11（禁外推）、§17.14（基准唯一）、§17.15（门禁落在任务损失）、§17.16、§17.17（净和不可排序给分量）、§17.18（判决腿须在训练分布内）、§18、§19、§20+§20.1+§20.2+§20.3、§21、§22+§22.8、§23（七问）。

## 1. 命题（一句话、可证伪）

在**与 2628 的 T=2048 点逐字段完全相同的配方**（stock dense，d=256/L=8/h=8/mlp=4，bs8，
seed0，lr 1e-3，`exit-mode uniform`，enwik8 前 90M/末 10M，两臂全部套 `qk-normfrom512`）上，
**只把训练步数从 3000 改到 6000**（= 2626 干净三点里 T=1024 点的步数，**matched-step**；
token 预算随之从 49.152M 变成 98.304M/臂）：

> ① T=2048 的 `Δbpc = bpc_full − bpc_window`（正 = 窗口更好）在 **6000 步**处落在哪个
> **带宽分类**（|Δ| < 0.014 ⇒ 不亏）？与 @3000 的读数**只作两点描述、禁外推**。
> ② **稳定性前置**：`full@T=2048@6000` 的 `gnorm_max < 100` 且跳步 0 成立吗（不成立 ⇒
> 本实验**无效**，先报）？③ 显存（若越 5GB 礼貌线按 §20.2 三件套）？④ 把 T=2048 补到
> 与 T=1024 **同 step** 后，「窗口**不亏**」是否保持？

已知事实（跨 run 只读引用）：
- **2628（45-L1）**：T=2048@**3000 步**，`Δ2048 = +0.004361`（full 2.339413 / window 2.335052）
  ⇒ 不亏（噪声带内）；`full@2048` 末 loss **1.5423**、`gnorm_max 7.22`、跳步 0；窗口臂峰值
  5.91 GiB（越 5GB 礼貌线，已按 §20.2 记账）。
- **2626（43-V1 干净三点）**：T=256/512/1024 分别为 24000/12000/**6000** 步（同 49.152M token/臂），
  Δ = −0.015752 / −0.015101 / **+0.009264**；`full@T=1024@6000` 末 loss **1.2146**、`gnorm_max 7.25`。

## 2. 单因素声明与能力面分析（SCHEMA §22.8）

### 2.1 单因素

- **唯一变量 = 训练步数**（3000 → 6000）。两臂（`train_window` ∈ {0, T/2 = 1024}）在**两臂之间**
  仍然只差这一个开关；T=2048 在两臂间相同。
- 与 2628 的点相比：`d/L/H/mlp/dense/topology/bs/lr/seed/qk_norm/数据/退出模式/窗口` **逐字段相同**，
  **只差 `steps`**（及其必然后果 token 总量 49.152M → 98.304M）。⇒ 与 2628 的点构成
  「同一个 T=2048 观察点的两个快照」，可作**两点描述**（§9 对照复用的配置层 equality）。
- **两臂**都套同一个**长度条件档** `qk-normfrom512`（`min_t=512`）。因 `seq_len=2048 > 512`，
  **两臂各挂 8 个 hook**（逐头 L2 归一 q/k，零新增参数）。
- 基底是**主流标准 decoder-only transformer**（dense：无 carry 门、无随机退出、仅最终深度 CE、
  tied 读出、`topology=sequential`）。三条 RAP 共享底座偏离在两臂间完全相同，不作变量。
- **实现说明**：窗口掩码是**位置式**（`keep[i,j] = (j≤i) ∧ ((i−j) < w)`），由
  `pathlm.depth_ar.window_anchor_keep_mask` 生成，训练侧与评测侧（`retention.masked_logits`）
  **同一函数**（单测钉住）。
- **非必要性例外**：**不改任何共享文件**（`pathlm/`、`train_depth_ar.py`、`tests/test_depth_ar.py`
  一字不动）。QK-Norm hook **从 2625 的实验本地驱动器只读引用**（单一实现来源），不复制。

### 2.2 能力面分析（派发前逐条核对）

`grep -n "ValueError\|互斥" pathlm/depth_ar.py` 命中 76 条；与本组合相关的校验：

- `train_window > 0 and not dense` ⇒ raise；`train_window > 0 and topology != "sequential"` ⇒ raise
  —— 本实验 `--dense` + 缺省 `sequential` ✅
- `attn_sparse_k ⊕ train_window`、`attn_linear ⊕ train_window`、`carry_rule ⊕ train_window`、
  `n_kv_heads<n_heads ⊕ train_window` ⇒ 本实验**全不开**这些算子（只开窗口）✅
- **`steps` 不在引擎校验面内**（它是训练循环参数，`train_depth_ar.train()` 的 `tcfg["steps"]`），
  故步数改动不触碰任何互斥校验；2626 的 T=1024 点已用同一路径跑过 6000 步 ✅
- **QK-Norm 本身不新增引擎 flag**：它是 `qkv` 的 forward hook（`min_t=512`），不触碰
  `DepthARConfig` 的校验面 ✅
- 正向/反向构造实测见 `controls/capability_analysis.json`：两臂正向构造成功（params 7,421,440）；
  `dense=False + train_window>0` 与 `topology=parallel + train_window>0` 被拒绝（证明校验活着）。
- **结论：契约要求的组合（full / 窗口 T/2 × T=2048 × 6000 步 × qk-normfrom512）在引擎中可达，
  无未测因子。**

### 2.3 同相对刻度与非退化（SCHEMA §17.16）

窗口点 = `window1024@T=2048`，窗口占比 **50%**（与 2626 三点、2628 同一点同一相对刻度）；
该点**非退化**（`w = 1024 < T = 2048`，与精确因果掩码不等）——单测钉住。**禁止**用
`window=T`（恒等）充当点。

## 3. 指标表（逐项分析）

| 维度 | 指标 | 与谁比 |
|---|---|---|
| 解决什么问题 | 2628 的 T=2048 是 3000 步快照 vs T=1024 是 6000 步 ⇒ **收敛度不等**；补到同 step 后「不亏」是否保持 | 2628 的 T=2048@3000 + 2626 的 T=1024@6000 |
| **主判据（①）** | `Δ2048@6000 = bpc_full − bpc_window`（native 工作点）与带宽分类 + 两点描述（@3000 / @6000） | 本点的 full（§17.14 基准唯一） |
| 代价（质量·并列） | release 口径 Δ（窗口臂在全可见下评测） | 同上 |
| **稳定性前置（②）** | `full@T=2048@6000` 的 `gnorm_max`、`grad_skips`（阈 <100 且 0） | 2628 的 7.22（同点@3000）/ 2626 的 7.25 |
| **显存（③）** | 两轮 `model_step` 实测 + 逐臂训练峰值 | 5GB net 礼貌线（超线按 §20.2 三件套） |
| **同收敛回答（④）** | 带分类是否保持 + 三个末 loss（2048@6000 / 2048@3000 / 1024@6000） | 同上 |
| 口径对照（辅助） | 与 2628 逐字段（只差 steps）、与 2626 的 T=1024 同 step | 只读引用 |
| 注意力占比（辅助） | 实测：注意力孤岛 fwd+bwd / 整模型步时 @T=2048；与公式对照 | 公式 `512T/(512T+786432)`（2048=0.5714） |
| 有效性 | Δ + 方差界（Δ < 0.014 不可排序，`docs/methodology.md §2`） | SCHEMA §12 判据线 |
| **干预活性（必报）** | `window_mask`（每 200 步随日志）+ `qk_norm_hooks`/`params_added`（结构活性）+ `steps` 落盘为 6000 | 0（死干预） |
| 稳定性（必报） | 逐臂 gnorm max/mean、尖峰、跳步、末窗 loss 斜率（§17.15） | 各臂日志 |
| 长程腿（§19） | 合成 `var_recall`（距离档 512/1024/1536）top-1 准确率 + chance + 局部先验抗性 | 同任务 chance / 局部先验 |

**判读规则**：先过「实现/算子」门（单测 + 变异 + 活性量），再判**稳定性前置②**，最后才提质量。

## 4. 判据线（预先写下，逐字来自契约 W2，不得修改）

Δ 符号（SCHEMA §17.3，全篇统一）：`Δ = bpc_对照 − bpc_实验 = bpc_full − bpc_window`
（**正 = 窗口更好**）。方差界：Δ < 0.014 不可排序（`docs/methodology.md §2`）。

- **① 质量（同收敛点）**：报 `Δ2048@6000` 与**带宽分类**（|Δ| < 0.014 ⇒ 「不亏（噪声带内）」，
  否则按符号分「窗口更好/更差（越界）」）；与 `Δ2048@3000`（2628，只读引用）**只作两点描述**，
  **禁外推/内插**（含「交叉点 T*」）。
- **② 稳定性前置（gate）**：`full@T=2048@6000` 的 `gnorm_max < 100` **且** `grad_skips == 0`。
  **不成立 ⇒ 本实验无效**（先如实报，再按 §8 失败决策树处置；**不得**事后改判据线）。
- **③ 显存实测**：报逐臂训练峰值（torch + `nvidia-smi` 子进程）与 `model_step` 实测；
  若 >5GB net（礼貌线）⇒ 按 §20.2 三件套记账（**优先保两臂配方一致**），或降 batch
  （**两臂同降**，并在口径声明里标注差异）。
- **④ 同收敛回答**：明确回答「把 T=2048 补到与 T=1024 同 step（6000）后，『窗口不亏』是否
  保持」；答案必须同时给 `Δ@3000`、`Δ@6000` 与三个末 loss（`2048@6000` / `2048@3000` /
  `1024@6000`），并说明「同 step ≠ 同 loss」，**不得**把同 step 说成同收敛/渐近。

## 5. 既有做法（prior art，SCHEMA §18）

> 本环境**联网不可用**。外部内容一律 **`记忆·未核`**，只作设计方向、**不得当判据**。

1. **谁做过**：滑动窗口注意力（Longformer / Mistral SWA）是长上下文标准件；QK-Norm
   （对 q/k 逐头归一）在长上下文 / 大模型训练里被用于压住注意力 logits 的极值。
2. **为什么这么设计**：注意力 logits 的极值随可见键数增长 ⇒ 长 T 上注意力可变得极尖、
   训练曲率上升；QK-Norm 把每行 logits 的尺度钉在上界 √d_h 附近，直接压住这个尺度。
3. **它的实测**：公开报告给的是**配方**，不报「同尺寸、同 token/step 预算、单因素、多点趋势」
   的稳定性对照。`记忆·未核`，不可当判据。
4. **我们的增量**：2626 给了 T=256/512/1024 的三点、2628 给了 T=2048@3000；本实验把 T=2048
   补到**与 T=1024 同 step（6000）**，判「窗口代价消失」在**同 step 口径**下是否保持——
   这正是对「收敛度不等」这条混杂的**直接控制**（业界不报这种单因素 matched-step 对照）。

**检索词**（供将来可核证的 run）：`QK-norm long context training stability`、
`sliding window attention crossover`、`matched-step vs matched-token comparison language model`。

## 6. 成本与 scope

| 项 | 值 |
|---|---|
| `gpu_wall_budget` | **≈15 min**（含冒烟 + 两臂各 6000 步训练 + 电池 + 合成腿；锁等待不计） |
| **训练本体（先测步时再定，§20.3）** | 两臂各 **6000 步**、bs8、T=2048；按 2628 **实测步时 0.0534 s/step** 估 = **≈10.7 min**（两臂）。派发里的「≈10 min」即指这一段；冒烟/评测为附加小项 |
| **口径声明** | 本点是 **matched-step@6000**：与 2626 的 T=1024 点同 step，但 token 预算 2×（98.304M vs 49.152M）⇒ 与既有点并列时标注为**混合口径** |
| 臂数 | **2**（= 1 点 × 2 臂）= 干预档数（§17.8；本实验干预档 = 步数 1 档 × 窗口 2 臂） |
| 峰值显存 | ≤5GB net（共享卡）；2628 同配方窗口臂峰值 **5.91 GiB**（越线 18%，已按 §20.2 记账）⇒ 预期同样越线 |
| 可中断性 | 是（每 200 步落日志 + 探针；模型 checkpoint 结束时落盘） |
| 冒烟计划 | `--stage preflight`：两臂各 20 步（实测步时/峰值/活性量，**不含任何 20 步外推时间字段**，§17.5）+ 成本基准（T=2048 × full/window） |
| 预算不够时砍什么（§8/§20.3） | 顺序：① 砍**辅助腿**（合成腿 → 注意力占比辅助测量）→ ② 若卡被外部占满（步时 >1.5×）则**停下请示**。**不砍步数（= 干预本身）、不砍臂、不换配方、不砍判据线** |

规模纪律（SCHEMA §7）：单卡 RTX 4090 24GB 多任务共享，GPU 串行；只做 7.42M 原理验证，**不外推**规模/渐近。

## 7. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | char 级 enwik8（`data/enwik8_full.npz`，前 90M train / 末 10M eval / vocab 206） |
| 训练入口 | `clean_driver.py <arm> …`（实验本地；**逐字调用** `train_depth_ar.train()`；默认 `--steps 6000`） |
| QK-Norm 来源 | **只读引用** 2625 的 `stability_driver.py::apply_attn_scale`（进程内 forward hook） |
| 驱动器保真度 | 与 2628 的引擎路径同数据流/同随机数消耗（同 bs8/qk-norm 配方；只多跑 3000 步） |
| 入口脚本 | `run.py`（preflight / train / train-all / eval / report / promote）、`clean_driver.py`、`capability_check.py` |
| 代码改动 | **无共享文件改动**——只新增本目录文件 + `tests/test_longT_2048_matched.py` |
| 可失败单测 | `tests/test_longT_2048_matched.py`：T=2048 两臂**各挂 8 hook、逐头 L2 范数 == √d_h、零新增参数**；两臂表与窗口非退化；**STEPS=6000 且与 2626 的 T=1024 同 step**；token 预算 = 2×49.152M；与 2628 的配方常量逐一相等；`StepProbe` 训练门 |
| 变异验证 | `mutate.py`：进程内 monkeypatch 破坏长度条件门 / 归一 / 窗口点 / 探针 / **步数干预** ⇒ 5/5 抓红 ⇒ 逐字节还原（**不写共享文件**） |
| 落盘 | `runs/<arm>/{results.json,train_log.jsonl,gate_log.jsonl,probe.jsonl,model.pt(不入库)}` / `<arm>-train_summary.json` / `smoke/smoke.json` / `eval.json` / `controls/capability_analysis.json` / `reconcile_numbers.py` + `reconcile_report.json` / `mutate.py` + `mutation_report.json` / `MODEL.md` |

## 8. 失败决策树

| 症状 | 诊断顺序 | 处置 |
|---|---|---|
| 构造失败 / 组合被引擎拒绝 | ① 实现 | 先修接线；未排除前不判机制 |
| 单测红 / 变异不抓红 | ① 实现 | 先修实现；未排除前不判机制 |
| `full@T=2048@6000` 炸（②不成立） | ④ 设计 | **本实验无效**：先如实报（含 gnorm 曲线），再立后续项；**不得**改判据 |
| 训练/评测逐步发散（loss NaN） | ①/③ | 记录首个 NaN 步；报出并请示（步数是干预、不能自行改） |
| 显存 >5GB net | ③ 硬件 | 按 §20.2 三件套记账（**优先保两臂配方**）；若余量不足 8GB 则排队等锁 |
| 卡被外部占满（步时涨 >1.5×） | ③ 硬件（§20.3） | **停下请示**（步数 = 干预本身，不能缩）；如实记账环境争用 |
| 锁等待超时 / 锁泄漏 | — | 上报并记录（§20） |

## 9. 文件所有权与共享面声明（SCHEMA §22）

- 本实验**只写**本实验目录 + **新增** `tests/test_longT_2048_matched.py`。
- **不写** `pathlm/depth_ar.py`、`pathlm/model.py`、`train_depth_ar.py`、`tests/test_depth_ar.py`
  （共享实现）。
- 与 2622/2623/2624/2625/2626/2628 的关系：**跨 run 只读引用**（干净三点 + QK-Norm hook + 2628 的
  @3000 读数）。
- GPU 取锁遵守 §20/§20.1（**按臂释放 + ≥60s 窗口**）；峰值 ≤5GB net。

## 10. 产出与既有做法

见 §5、§7。本目录下 `README.md`(本文件) / `clean_driver.py` / `run.py` / `capability_check.py` /
`analysis.md` / `reconcile_numbers.py` / `reconcile_report.json` / `mutate.py` /
`mutation_report.json` / `config.json` / `results.json` / `eval.json` / `MODEL.md` /
`smoke/smoke.json` / `controls/capability_analysis.json` / `runs/**`（权重 `model.pt` 不入库；
`runs/` 被 .gitignore 忽略）/ `<arm>-train_log.jsonl` + `<arm>-gate_log.jsonl` +
`<arm>-probe.jsonl`（逐臂证据的**可入库副本**，`run.py --stage promote` 生成）；
`tests/test_longT_2048_matched.py`。

> 契约副本标注（SCHEMA §17.6）：§0 的 W2 原文段与主 Agent 补充纪律为**逐字**抄录；
> §1–§10 为执行者实现/执行细节（未改判据线）。§0 原文里的「loss 1.54 / 1.21」是本题对该系列
> 读数的**简记**（精确值：2628 `full@2048@3000` = 1.5423；2626 `full@1024@6000` = 1.2146）。
