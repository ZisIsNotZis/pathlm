---
name: 2626-longT-clean
ticket: 43-V1
date: "2026-10-03"
grid: A6-mixing-operator（长 T 干净三点：QK-Norm 修复后的趋势）
engine: depth-ar `--dense`（stock decoder-only transformer，7.16M@T=1024）+ 实验本地候选补丁驱动器（复用 2625 的 QK-Norm hook）
budget: "gpu_wall_budget: 60–80 min（含冒烟 + 六臂训练 + 电池 + 合成腿）；峰值 ≤5GB net"
status: judged
---

# 2626 · 长 T 干净三点（A6 · 7.16M · depth-ar/dense × qk-normfrom512）

> 本文件即本实验的本地契约。§0 把 `.scratch/43-longt-clean/issues/01-longt-clean.md`
> 的「V1」段**逐字**抄下（判据线一字未改）；§1 起是实现与执行细节。
> 原始结果：`runs/<arm>/results.json`、`runs/<arm>/train_log.jsonl`、`runs/<arm>/probe.jsonl`；
> 判读：`analysis.md`；对账：`reconcile_report.json`；变异：`mutation_report.json`；
> 成本：`smoke/smoke.json`、`eval.json`；权重清单：`MODEL.md`。

## 0. 契约原文（逐字）

> ## V1 · 长 T 干净三点（≈60–80 min）
> - 目录：`axes/1-time/1.1.1-hard-window/experiments/2626-longT-clean/`
> - 动机：W1 已用 **QK-Norm（零参数）**切断第 1 层注意力 logit 爆炸（gnorm 1740.9/29578.6 → 7.25/7.58），
>   并推荐**长度条件档 `qk-normfrom512`**（T≤512 零 hook）。入场券已拿到 ⇒ 可给**干净三点**。
> - 干预：T ∈ {256, 512, 1024} × {full, 窗口=同相对占比 50%（128/256/512）}，**全部启用 `qk-normfrom512`**。
> - 判据（预登记）：① 各 T 的 Δbpc 与 band 分类（三点并列，**仍禁外推**）；
>   ② **稳定性前置**：full 臂 T=1024 的 gnorm_max 必须 <100 且跳步 0（否则实验无效：先报再修）；
>   ③ 注意力占比实测 vs 公式；④ 显存与墙钟（两轮，方向不明如实记）；
>   ⑤ **口径声明**：与 X3/Y2 的旧点不可直接并列（旧点未用 QK-Norm、Y2 还是 bs8）——须单独列表对比。

**主 Agent 补充纪律（随派发给出，同样视为契约）**：
- GPU 取锁、**按臂释放并留 ≥60s 窗口**；峰值 ≤5GB net（超线按 §20.2 三件套记账）；
  **先测步时再定预算**（§20.3）。
- **派发前做能力面分析**（§22.8）：`grep -n "ValueError\|互斥" pathlm/depth_ar.py` 核对组合。
- 长程类用 `tools/evalkit/longrange_tasks.py` 合成任务；主判据难度档须在训练分布内（§17.18）。
- 单因素 + 逐字段对照（含 seed）；活性量每 200 步；语义开关要有可失败单测 + 变异验证（抓红且逐字节还原）。
- 三道对账 + 禁外推；数字与 `results.json`/日志逐一对应 + 自带 `reconcile_numbers.py`（报告落盘）。
- 结束前 `python3 -m pytest tests/ -q` 全绿（基线 **490**）；权重不入库；**不碰** `catalog.yaml` / `INDEX.md` / `axes/**/README.md`。
- **提交前逐条验证**：把报告/README 里声称的交付物路径逐个 grep 进 `git diff --cached --name-only`（AGENTS.md）。
- 遇冲突 / 预算不够 / 锁异常 / 数字可疑 → 停下请示。

**相关规范**：`harness/SCHEMA.md` §14/§14b、§17.11（禁外推）、§17.14（基准唯一）、§17.16（两点同刻度非退化）、
§17.17（净和不可排序给分量）、**§17.18（判决腿须在训练分布内）**、§18、§19、§20+§20.1+§20.2+§20.3、
§21、§22+§22.8、§23（七问）。

## 1. 命题（一句话、可证伪）

在**同一个自洽配方**（stock dense，d=256/L=8/h=8/mlp=4，**bs8**，seed0，lr 1e-3，
`exit-mode uniform`，enwik8 前 90M/末 10M，**共同 49.15M token/臂**）上，把 T ∈ {256, 512, 1024}
× 训练期可见窗口 ∈ {full, T/2}（**同相对刻度 50%**）**全部套用长度条件档 `qk-normfrom512`**
（`min_t=512`；T ≤ 512 不挂 hook、T=1024 逐头归一 q/k），问：

> ① 窗口相对 full 的 `Δbpc = bpc_full − bpc_window`（正 = 窗口更好）在三点上各落在哪个**带宽分类**
> （|Δ| < 0.014 ⇒ 不亏）？② **稳定性前置**：`full@T=1024` 的 `gnorm_max < 100` 且跳步 0 成立吗
> （不成立 ⇒ 本实验**无效**，先报再修）？③ 注意力子层算力占比**实测**与公式 `512T/(512T+786432)`
> 差多少？④ 显存/墙钟两轮方向？

已知事实（2625-W1，本实验的入场券，跨 run 引用）：QK-Norm 把 `full@T=1024` 的 `gnorm_max`
从 1740.9（seed0）/29578.6（seed1）压到 **7.25/7.58**、跳步 0；长度条件档 `qk-normfrom512`
在 T=512 上 `n_qkv_hooks == 0`（结构性零 hook），实测 Δbpc = −0.0016。

## 2. 单因素声明与能力面分析（SCHEMA §22.8）

### 2.1 单因素

- **唯一变量 = 训练期可见窗口**（`train_window` ∈ {0 关闭, T/2}），T 在两臂间相同；
  **三点之间**只改 `seq_len`（位置表随之线性变大，是 `seq_len` 的**必然后果**，不是第二个开关）。
- **所有六臂**都套同一个**长度条件档** `qk-normfrom512`（`min_t=512`）——它是 2625-W1 的
  修复手段，**不是本实验的变量**。因 `min_t=512`：T=256/512 的四个臂 `n_qkv_hooks == 0`
  （前向逐算子 = stock 基线），只有 T=1024 的两个臂真的归一 q/k。这一点由 2625 单测
  `test_qk_norm_length_conditional_is_inert_at_or_below_min_t` 钉住，本实验另有专项单测。
- 基底是**主流标准 decoder-only transformer**（dense：无 carry 门、无随机退出、仅最终深度 CE、
  tied 读出、`topology=sequential`）。三条 RAP 共享底座偏离在**六臂间完全相同**，不作变量。
- **实现说明**：窗口掩码是**位置式**（`keep[i,j] = (j≤i) ∧ ((i−j) < w)`），由
  `pathlm.depth_ar.window_anchor_keep_mask` 生成，训练侧与评测侧（`retention.masked_logits`）
  **同一函数**（单测钉住）。
- **非必要性例外**：**不改任何共享文件**（`pathlm/`、`train_depth_ar.py`、`tests/test_depth_ar.py`
  一字不动）。QK-Norm hook **从 2625 的实验本地驱动器只读引用**（单一实现来源），不复制。

### 2.2 能力面分析（派发前逐条核对）

`grep -n "ValueError\|互斥" pathlm/depth_ar.py` 共 **76** 条命中；与本组合相关的校验：

- `train_window > 0 and not dense` ⇒ raise；`train_window > 0 and topology != "sequential"` ⇒ raise
  —— 本实验 `--dense` + 缺省 `sequential` ✅
- `attn_sparse_k ⊕ train_window`、`attn_linear ⊕ train_window`、`carry_rule ⊕ train_window`、
  `n_kv_heads<n_heads ⊕ train_window` ⇒ 本实验**全不开**这些算子（只开窗口），无互斥 ✅
- 与 2622/2623 同组合（`dense + sequential + train_window`）⇒ 已在引擎中跑通 ✅
- **QK-Norm 本身不新增引擎 flag**：它是 `qkv` 的 forward hook（`min_t=512` 时按 `cfg.seq_len`
  决定是否挂），不触碰 `DepthARConfig` 的校验面 ✅
- 正向/反向构造实测见 `controls/capability_analysis.json`：两臂正向构造成功；
  `dense=False + train_window>0` 与 `topology=parallel + train_window>0` 被拒绝（证明校验活着）。
- **结论：契约要求的组合（full / 窗口 T/2 × 三点 × qk-normfrom512）在引擎中可达，无未测因子。**

### 2.3 同相对刻度与非退化（SCHEMA §17.16）

三点 = `window128@T=256` / `window256@T=512` / `window512@T=1024`，窗口占比**同为 50%**；
三点都**非退化**（`w < T`，与精确因果掩码不等）——单测钉住。**禁止**用 `window=T`（恒等）充当点。

## 3. 指标表（逐项分析）

| 维度 | 指标 | 与谁比 |
|---|---|---|
| 解决什么问题 | QK-Norm 修复后，A6 的「T=256 净亏 / T=512 不亏 / T=1024 不可判」能否给出**干净三点** | 2622/2623 的旧点（**单独列表**，口径不同） |
| **主判据（①）** | `Δbpc(T) = bpc_full − bpc_window`（native 工作点）与带宽分类 | 各点自己的 full（§17.14 基准唯一） |
| 代价（质量·并列） | release 口径 Δ（窗口臂在全可见下评测） | 同上 |
| **稳定性前置（②）** | `full@T=1024` 的 `gnorm_max`、`grad_skips`（阈 <100 且 0） | 2623/2624 的 1740.9 / 29578.6 |
| **注意力占比（③）** | 实测：注意力孤岛 fwd+bwd / 整模型步时，随 T；与公式 `512T/(512T+786432)` 对照 | 公式外推值（对照，非判据） |
| **显存/墙钟（④）** | 两轮（preflight + eval）`model_step` 实测 ms/显存；arm 训练峰值 | 5GB net 礼貌线 |
| 有效性 | Δ + 方差界（Δ < 0.014 不可排序，`harness/methodology.md §2`） | SCHEMA §12 判据线 |
| **干预活性（必报）** | `window_mask`（每 200 步随日志）+ `qk_norm_hooks`/`params_added`（结构活性） | 0（死干预） |
| 稳定性（必报） | 逐臂 gnorm max/mean、尖峰、跳步、末窗 loss 斜率 | 各臂日志 |
| 长程腿（§19） | 合成 `var_recall`（距离档 × 臂）top-1 准确率 + chance + 局部先验抗性 | 同任务 chance / 局部先验 |
| 净和/分量（§17.17） | 不适用（非「净和不可排序」场景：主判据是逐点 Δ 的带宽分类） | — |

**判读规则**：先过「实现/算子」门（单测 + 变异 + 活性量），再判**稳定性前置②**，最后才提质量。

## 4. 判据线（预先写下，逐字来自契约 V1，不得修改）

Δ 符号（SCHEMA §17.3，全篇统一）：`Δ = bpc_对照 − bpc_实验 = bpc_full − bpc_window`
（**正 = 窗口更好**）。方差界：Δ < 0.014 不可排序（`harness/methodology.md §2`）。

- **① 质量（三点，只描述、禁外推）**：报三点 `Δbpc(T)` 与**带宽分类**
  （|Δ| < 0.014 ⇒ 「不亏（噪声带内）」，否则按符号分「窗口更好/更差（越界）」）。
  **只描述三点，不作任何外推/内插**（含「交叉点 T*」）。
- **② 稳定性前置（gate）**：`full@T=1024` 的 `gnorm_max < 100` **且** `grad_skips == 0`。
  **不成立 ⇒ 本实验无效**（先如实报，再按 §8 失败决策树处置；**不得**事后改判据线）。
- **③ 注意力占比**：报 T ∈ {256, 512, 1024} 的注意力孤岛/整模型**实测**步时比，与公式
  `attn_share(T) = 512T/(512T+786432)`（256=0.1429、512=0.2500、1024=0.4000）对照。
  **公式值只是对照，不是判据。**
- **④ 显存与墙钟（两轮）**：报 preflight 与 eval 两轮 `model_step` 实测；**两轮方向不明则如实记
  「测不出方向」**，不得把其中一轮当收益。报逐臂训练峰值（进程 + `nvidia-smi`）。
- **⑤ 口径声明（不可直接并列）**：本次干净三点（bs8 + qk-normfrom512 + 49.15M token）与
  旧点 **X3（2622，bs32、无 QK-Norm）/ Y2（2623/2624，bs8、无 QK-Norm，full 臂爆炸）**
  **不可直接并列**——须**单独列表对比**，并写明每一处口径差异。

## 5. 既有做法（prior art，SCHEMA §18）

> 本环境**联网不可用**。外部内容一律 **`记忆·未核`**，只作设计方向、**不得当判据**。

1. **谁做过**：滑动窗口注意力（Longformer / Mistral SWA）是长上下文标准件；QK-Norm
   （对 q/k 逐头归一）在长上下文 / 大模型训练里被用于压住注意力 logits 的极值（如
   vision transformer 的 QK-Norm、部分 LLM 的 qk-layernorm）。
2. **为什么这么设计**：注意力 logits 的极值随可见键数增长 ⇒ 长 T 上注意力可变得极尖、
   训练曲率上升；QK-Norm 把每行 logits 的尺度钉在上界 √d_h 附近，直接压住这个尺度。
3. **它的实测**：公开报告给的是**配方**，不报「同尺寸、同 token 预算、单因素、两点趋势」的
   稳定性对照。`记忆·未核`，不可当判据。
4. **我们的增量**：2625-W1 在 **7.16M / T=1024 / bs8** 上实测 QK-Norm 把 `gnorm_max`
   从 1.7×10³/3.0×10⁴ 压到 7.25/7.58；本实验在其修复之上给**修复后的干净三点趋势**，
   并**实测**注意力占比与墙钟（不靠公式外推）。

**检索词**（供将来可核证的 run）：`QK-norm long context training stability`、
`attention logit growth sequence length`、`sliding window attention crossover`。

## 6. 成本与 scope

| 项 | 值 |
|---|---|
| `gpu_wall_budget` | **60–80 min**（含冒烟 + 六臂训练 + 电池 + 合成腿；锁等待不计） |
| **token 预算（§20.3 先测步时再定）** | 实测 bs8 真实步时：T=256 **33.0 ms**、T=512 **33.8 ms**、T=1024 **48.1 ms**（`smoke/step_time_probe.json`）⇒ 98.3M 全预算约 **99 min**（超信封）⇒ **缩到共同半预算 49.15M/臂**（臂间同 token 预算，§8「砍步数、不砍判据线、不换配方」）：T=256→24000 步、T=512→12000 步、T=1024→6000 步，合计约 **50 min** 训练 |
| **口径声明** | 本实验为 **49.2M/臂的快照**；绝对水平与 98.3M 的旧点不可比（§17.11 禁外推） |
| 臂数 | **6**（= 3 点 × 2 臂）= 干预档数（§17.8） |
| 峰值显存 | ≤5GB net（共享卡）；冒烟实测 full@T=1024 **1.04 GiB**、win@T=1024 **~1.9 GiB** |
| 可中断性 | 是（每 200 步落日志 + 探针；模型 checkpoints 结束时落盘） |
| 冒烟计划 | `--stage preflight`：六臂各 20 步（实测步时/峰值/活性量，**不含任何 20 步外推时间字段**，§17.5）+ 成本基准（三点 × full/window） |
| 预算不够时砍什么（§8/§20.3） | 顺序：① 缩共同 token 预算（保臂间相等）→ ② 砍 release 口径/synth 距离档（**不砍判据线**）。**不砍臂、不换配方、不砍判据线** |

规模纪律（SCHEMA §7）：单卡 RTX 4090 24GB 多任务共享，GPU 串行；只做 7.16M 原理验证，
**不外推**规模/渐近。

## 7. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | char 级 enwik8（`data/enwik8_full.npz`，前 90M train / 末 10M eval / vocab 206） |
| 训练入口 | `clean_driver.py <arm> …`（实验本地；**逐字调用** `train_depth_ar.train()`） |
| QK-Norm 来源 | **只读引用** 2625 的 `stability_driver.py::apply_attn_scale`（进程内 forward hook） |
| 驱动器保真度 | 与 2623 的引擎路径同数据流/同随机数消耗；`none` 档的 step0 读数锚点见 `tests/test_longT_clean.py` |
| 入口脚本 | `run.py`（preflight / train / train-all / eval / report）、`clean_driver.py` |
| 代码改动 | **无共享文件改动**——只新增本目录文件 + `tests/test_longT_clean.py` |
| 可失败单测 | `tests/test_longT_clean.py`：长度条件档在 T ≤ 512 **零 hook 且前向逐位等于无 hook**；T > 512 时**逐头 L2 范数 == √d_h 且零新增参数**；六臂表与窗口非退化 |
| 变异验证 | `mutate.py`：进程内 monkeypatch 破坏长度条件门 / 归一 / 探针 ⇒ 必须抓红 ⇒ 逐字节还原（**不写共享文件**） |
| 落盘 | `runs/<arm>/{results.json,train_log.jsonl,gate_log.jsonl,probe.jsonl,model.pt(不入库)}` / `<arm>-train_summary.json` / `smoke/smoke.json` / `smoke/step_time_probe.json` / `eval.json` / `controls/capability_analysis.json` / `reconcile_numbers.py` + `reconcile_report.json` / `mutate.py` + `mutation_report.json` / `MODEL.md` |

## 8. 失败决策树

| 症状 | 诊断顺序 | 处置 |
|---|---|---|
| 构造失败 / 组合被引擎拒绝 | ① 实现 | 先修接线；未排除前不判机制 |
| 单测红 / 变异不抓红 | ① 实现 | 先修实现；未排除前不判机制 |
| `full@T=1024` 仍炸（②不成立） | ④ 设计 | **本实验无效**：先如实报（含 gnorm 曲线），再把「为何 qk-normfrom512 在 bs8/49M 下仍炸」立成后续项；**不得**改判据 |
| 训练/评测逐步发散（loss NaN） | ①/③ | 记录首个 NaN 步；缩 token 预算重跑（保臂间相等） |
| 显存 >5GB net | ③ 硬件 | 按 §20.2 三件套记账（不改配方） |
| 卡被外部占满（步时涨） | ③ 硬件（§20.3） | 已按 §20.3 用共同半预算；若再涨则进一步缩共同预算、保臂间相等 |
| 锁等待超时 / 锁泄漏 | — | 上报并记录（§20） |

## 9. 文件所有权与共享面声明（SCHEMA §22）

- 本实验**只写**本实验目录 + **新增** `tests/test_longT_clean.py`。
- **不写** `pathlm/depth_ar.py`、`pathlm/model.py`、`train_depth_ar.py`、`tests/test_depth_ar.py`
  （共享实现）。
- 与 2622/2623/2624/2625 的关系：**跨 run 只读引用**（旧点结果 + QK-Norm hook）。
- GPU 取锁遵守 §20/§20.1（**按臂释放 + ≥60s 窗口**）；峰值 ≤5GB net。

## 10. 产出与既有做法

见 §5、§7。本目录下 `README.md`(本文件) / `clean_driver.py` / `run.py` / `analysis.md` /
`reconcile_numbers.py` / `reconcile_report.json` / `mutate.py` / `mutation_report.json` /
`config.json` / `results.json` / `eval.json` / `MODEL.md` / `smoke/*.json` /
`controls/capability_analysis.json` / `runs/**`（权重 `model.pt` 不入库；`runs/` 被 .gitignore 忽略）/
`<arm>-train_log.jsonl` + `<arm>-gate_log.jsonl` + `<arm>-probe.jsonl`（逐臂证据的**可入库副本**）；
`tests/test_longT_clean.py`。

> 契约副本标注（SCHEMA §17.6）：§0 的 V1 原文段与主 Agent 补充纪律为**逐字**抄录；
> §1–§10 为执行者实现/执行细节（未改判据线）。
