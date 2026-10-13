---
name: 2636-longT-unified
ticket: 52-A6U
date: "2026-10-09"
grid: A6-mixing-operator（长 T 四点统一口径 + 四条腿「省不省」）
engine: depth-ar `--dense`（stock decoder-only transformer）+ 实验本地驱动器（复用 2625 的 QK-Norm hook）
budget: "gpu_wall_budget: 60 min（含冒烟 + 八臂训练 + 电池 + 一轮成本探针 + 合成腿）；峰值 ≤5GB net，超线按 SCHEMA §20.2 记账"
status: judged
---

# 2636 · 长 T 四点统一口径（A6 · 7.16M · depth-ar/dense × qk-normfrom512）

> 本文件即本实验的本地契约（判据线预登记，不得事后调整）。§0 是任务口径；
> §1 起是实现与执行细节。
> 原始结果：`runs/<arm>/results.json`、`runs/<arm>/train_log.jsonl`、`runs/<arm>/probe.jsonl`；
> 判读：`analysis.md`；对账：`reconcile_report.json`；变异：`mutation_report.json`；
> 成本：`smoke/smoke.json`、`eval.json`；权重清单：`MODEL.md`。

## 0. 本实验要解决的问题（承接 50-critical-audit 的 A 轴缺口）

审计（`.scratch/50-critical-audit/A-AUDIT.md` §2 P1-2、§4-②）对 A6 长 T 系列列出三重缺陷：

1. **口径不齐**：已有的四点（`2626`/`2628`/`2629`）token 预算与步数相互不一致——
   `2626` 的 T=256/512/1024 用 **49.15M token/臂**（步数 24000/12000/6000），
   `2629` 的 T=2048 用 **98.3M token/臂**（6000 步）；lr 以 `steps` 为分母
   （`train_depth_ar.py` 的 `LambdaLR`），故四点**退火刻度不等** ⇒ 四点混合口径，**禁止**当趋势外推。
2. **「省不省」无干净结论**：窗口在**当前实装下墙钟不省**（稠密 bool 掩码不兑现 FLOPs 节省），
   这句只在各实验内部各自说过，没有在同一次会话、对四点同时给出**质量 / 解析 FLOPs / 实测墙钟 /
   峰值显存**四条腿。
3. **线性注意力质量腿完全空白**（`2616`：三次训练期停摆，只有成本数据）。

本实验只针对缺口 1 与 2（缺口 3 见姊妹实验 `2637-linear-attn-quality`）。

## 1. 命题（一句话、可证伪）

在同一自洽配方（stock dense，d=256/L=8/h=8/mlp=4，**bs8**，seed0，lr 1e-3，`exit-mode uniform`，
enwik8 前 90M/末 10M，全部套长度条件档 `qk-normfrom512`，**共同步数 S=6000**）上，把
T ∈ {256, 512, 1024, 2048} × 训练期可见窗口 ∈ {full, w = T/2}（**同相对刻度 50%**）跑成
**四点同 step 口径**，问：

> ① 窗口相对 full 的 `Δbpc(T) = bpc_full(T) − bpc_window(T)`（正 = 窗口更好）在四点上的**形状**是什么
> （带宽分类，|Δ| < 0.014 记「不亏」）？它与既有 **同 token 预算（49.15M）** 的四点形状是否一致
> （这检验「窗口代价随 T 消失」是不是欠训练/退火刻度造成的假象）？
> ② 四条腿（质量 / 解析 FLOPs / 实测墙钟 / 峰值显存）里，窗口算子**哪条腿省、哪条不省**？
> ③ 稳定性前置（四点 full 臂 `gnorm_max < 100` 且跳步 0）是否成立？

## 2. 口径设计：为什么是「同 step」，token 预算必然不等（预登记声明）

- **不可兼得的事实**：对同一 batch 大小，`tokens = steps × bs × T`。固定 T 不同时，**不可能**同时
  满足「同 token 预算」与「同步数」。二者只能取一。
- **本实验主口径 = 同 step（S=6000）**：lr 调度是 `min((s+1)/200, 0.1+0.9·0.5·(1+cos(π·s/steps)))`，
  以 `steps` 为分母 ⇒ **同 steps = 同退火刻度 = 同收敛制度**，直接消掉审计点名的「24000/12000/6000
  步不等」混杂。代价：token 预算随 T 线性变化（12.29M / 24.58M / 49.15M / 98.30M），**如实分报**。
- **副口径 = 同 token（49.15M）**：直接引用既有四点（`2626` T=256/512/1024 + `2628` T=2048），
  作为**稳健性交叉核对**。它与主口径的步数不同（24000/12000/6000/3000），故只作并列描述，**不合并成单条曲线**。
- **为什么选 S=6000**：T=1024 与 T=2048 的既有臂本身就是 **6000 步**（`2626`/`2629`，同 bs8、同 seed0、
  同 `qk-normfrom512`）⇒ 本实验只需**补 T=256/512 两点**（4 臂）即可凑齐同 step 四点，成本最小、复用最省。
  复用条件见 §6（逐字段 equality）。

## 3. 单因素声明与能力面分析（SCHEMA §22.8）

### 3.1 单因素

- **唯一变量 = 训练期可见窗口**（`train_window` ∈ {0 关闭, T/2}），T 在两臂间相同。
- **四点之间**只差 `seq_len`（位置表随 `seq_len` 线性变大，是必然后果，不是第二个开关）。
- 基底是 stock decoder-only transformer（dense：无 carry 门、无随机退出、仅最终深度 CE、tied 读出、
  `topology=sequential`）。三条 RAP 共享底座偏离在八臂间完全相同，不作变量。
- 窗口掩码是**位置式**：`keep[i,j] = (j≤i) ∧ ((i−j) < w)`，由 `pathlm.depth_ar.window_anchor_keep_mask`
  生成，训练侧与评测侧（`retention.masked_logits`）同一函数（单测钉住）。
- **QK-Norm（长度条件档 `qk-normfrom512`，`min_t=512`）不是本实验的变量**：它是 `2625-W1` 的稳定性
  修复，被**固定**套在八臂上。因 `min_t=512`：T=256/512 的臂 `n_qkv_hooks == 0`（前向逐算子 = stock
  基线），只有 T=1024/2048 的臂真的逐头归一 q/k。该结构性后果由 `2625` 单测
  `test_qk_norm_length_conditional_is_inert_at_or_below_min_t` 钉住。
- **不改任何共享文件**（`pathlm/`、`train_depth_ar.py`、`tests/` 一字不动）。QK-Norm hook 从 `2625`
  的实验本地驱动器**只读引用**（单一实现来源）。

### 3.2 能力面分析（派发前逐条核对）

`grep -n "ValueError\|互斥" pathlm/depth_ar.py` 核对与本组合相关的校验：

- `train_window > 0 and not dense` ⇒ raise；`train_window > 0 and topology != "sequential"` ⇒ raise
  —— 本实验 `--dense` + 缺省 `sequential` ✅
- `attn_sparse_k ⊕ train_window`、`attn_linear ⊕ train_window`、`carry_rule ⊕ train_window`、
  `n_kv_heads<n_heads ⊕ train_window` ⇒ 本实验全不开这些算子（只开窗口）✅
- 与 `2622/2623/2626/2629` 同组合（`dense + sequential + train_window`）⇒ 已在引擎中跑通 ✅
- QK-Norm 本身不新增引擎 flag（`qkv` 的 forward hook），不触碰 `DepthDecodeConfig` 校验面 ✅

## 4. 判据线（预登记，逐字；不得事后调整）

**符号约定**：`Δ(T) = bpc_full(T) − bpc_window(T)`（**正 = 窗口更好**），全篇一致。

**主判据 C1（质量形状，同 step 四点）**：
- 读数 = 各 T 的 native bpc（full 臂 = 全因果；window 臂 = 自身训练掩码），同一评测流
  （enwik8 末 10M，`window_seed=2`，20 批 × bs16）。
- 分类 = 工程方差带 **0.014** 与**本格自测噪声 `σ_cell`**（由 §4 C4 得到）的双门限：
  `|Δ| < 0.014` ⇒ 「不亏（噪声带内）」，否则按符号记「窗口更好/更差（越界）」。
- 输出 = 四点形状（并报**同 token 副口径**四点形状，并列描述）。
- **不强制方向**：本判据只要求**如实报形状**，禁止把形状读作渐近或跨 T 外推。

**判据 C2（稳定性前置，入场券）**：
- 四点 full 臂的 `gnorm_max < 100` 且 `grad_skips == 0`。**四点全部成立**才认为本实验有效；
  任一点不成立 ⇒ 该点记为「不可归因」，不得进 C1 形状。

**判据 C3（四条腿「省不省」，逐 T 回答）**：
对每个 T 同时报：
1. **质量**：Δbpc（= C1 读数）；
2. **解析 FLOPs**：注意力 MACs/token/layer = `2·w·d`（窗口）vs `2·T·d`（full）；整层 all-in 降幅；
3. **实测墙钟**：同一次会话内 `ms/step(window) / ms/step(full)`（整模型 fwd+bwd，含 QK-Norm 配方）；
4. **峰值显存**：同一次会话内窗口 vs full 的 torch 峰值比。
- **预登记期望（可被推翻）**：解析 FLOPs **省**（注意力 −50% + all-in 随 T 增大而降幅变大）；
  墙钟与显存**不省**（当前实装走稠密 bool 掩码，物化 T×T 掩码与满分数矩阵）
  ⇒ 结论须写「哪条腿省、哪条不省」，并**明确坐实或推翻**「窗口墙钟不省」。
- 墙钟读数对环境敏感 ⇒ 报**两轮**（preflight / eval），方向不一致时如实记「测不出方向」。

**判据 C4（噪声门限本格自测）**：
- 用 seed1 重跑四臂（同配置，唯一差别 seed）⇒ 得**本格**单臂 bpc 跨 seed 极差与
  `Δ(T)` 的跨 seed 极差，作为 `σ_cell`。
- 若 seed1 因预算被砍（SCHEMA §8 允许砍臂/步数、**不砍判据线**），则**如实声明**并引用 `2625` 已测的
  本格噪声底（T=512 同配方两次重跑 clean bpc 极差 **0.0045**）作为**替代门限**，且 C1 分类只作描述。

**判据 C5（口径对照，不得混用）**：
- 同 step 四点与同 token 四点**分表报告**，明确标注各自的 steps/tokens；
- 与历史点（`2622` X3 的 bs32、`2626`/`2628` 半预算）逐字段列表对照，不等字段显式列出。

**必须不变（护栏）**：参数量（掩码零参数，`params_added == 0`）；因果性（窗口掩码对角恒开、
`window_anchor_keep_mask` 单测）；KV 缓存字节（窗口只改可见集，**结构上不减缓存形状**）。

## 5. 既有做法（prior art，SCHEMA §18）

| # | 谁做过 | 为什么这么设计 | 它的实测结果 | 我们的增量 |
|---|---|---|---|---|
| 1 | StreamingLLM（Xiao et al. 2023）| 保 sink + 滑窗即可在超长流上保质量 | 长度外推上不重训即稳 | 我们量的是**训练期**滑窗在 **13M 同 step** 下的质量税与成本四腿（同 step 四点形状，业界不报）|
| 2 | Mistral SWA（2024）/ Longformer / BigBird | 滑窗把注意力从 O(T²) 降到 O(T·w) | 长上下文吞吐明显占优 | 我们量的是**当前稠密掩码实装**下「理论省 vs 实测省」的落差（FLOPs 省、墙钟不省） |
| 3 | QK-Norm（Henry et al. 2020；Dehghani 2023 ViT-22B 复用）| 约束注意力 logits 上界，防熵塌缩/发散 | 提升大模型训练稳定性 | 我们在长 T 崩溃上定位到「第 1 层 logits 爆炸」并用**零参数** QK-Norm 修复（`2625`），修正有局部实测 |
| 4 | 线性注意力（Katharopoulos et al. 2020，elu+1） | 把 softmax 换成归一化线性核 ⇒ 每层 O(T)、缓存常数 | 短上下文质量持平、长上下文吞吐占优 | 见 `2637`（质量腿在本仓首次可判）|

知识来源：`记忆·未核`（无检索工具，按 SCHEMA §21 不作判据，仅作设计方向）。

## 6. 对照与复用（SCHEMA §9）

**T=1024/2048 的同 step 点从 `2626`/`2629` 只读复用**（省算力、不省证据）。复用条件逐条核对：

1. **实现层 equality**：八臂共用同一 `clean_driver.py`（本目录，复用 `2625` 的 QK-Norm hook 单一实现）；
   `window_anchor_keep_mask` 训练/评测同一函数；QK-Norm 长度条件档由单测钉住。
2. **配置层 equality**（逐字段列表）：`2626` 的 `full-1024-clean`/`win512-1024-clean` 与 `2629` 的
   `full-2048-clean`/`win1024-2048-clean` 的 resolved config 与本实验新臂对照，允许差的字段仅有
   `seq_len`/`train_window`；其余（d/layers/heads/mlp/dense/topology/bs8/lr/seed0/exit-mode/
   qk-normfrom512/数据切分/steps=6000）逐字段相等。不等字段在 `reconcile_report.json` 显式列出。
3. **口径标注**：`analysis.md` 标注「历史同配置对照（同 step 6000）+ 等值性证明 + 已知口径」。
4. 若出现字段不等 ⇒ 回退到自包含双跑（本实验已备 `--stage train` 可单臂重跑）。

## 7. 指标表（多维，逐项分析；SCHEMA §6）

| 维度 | 指标 | 对照 | 口径 |
|---|---|---|---|
| 质量（代价） | clean bpc（native） | 各 T 自己的 full 臂（诚实对手，同 step/T） | matched-step 快照 |
| 算力（理论） | 注意力 MACs/token/layer、整层 all-in | full 臂 | 解析公式（只作对照） |
| 算力（实测） | 整模型 ms/step、注意力子层孤岛 ms | 同 T 的 full 臂 | 同会话两轮 |
| 内存 | torch 峰值、nvidia-smi 子进程峰值 | 同 T 的 full 臂 | 同会话 |
| 稳定性 | `gnorm_max` / `grad_skips` | 阈值 100 / 0 | 逐 200 步 |
| 活性（§10） | `window_mask.alive`、`visible_frac`、`n_qkv_hooks`、`params_added` | — | 逐 200 步 |
| 收益与用法 | 部署含义 = 长上下文省算力的前置 | — | 禁外推 |

## 8. 成本（SCHEMA §7/§8；先测步时再定预算）

- `gpu_wall_budget: 60 min`（含冒烟、八臂训练、电池、一轮成本探针、合成腿）。
- 臂数（§17.8）= seed0 四臂 + seed1 四臂 = **8 臂**。按既有实测步时（bs8，套 QK-Norm）估：
  T=256 full/win ≈ 0.035/0.044 s/step、T=512 ≈ 0.042/0.035 s/step，6000 步 ⇒ seed0 约 16 min、
  seed1 约 16 min，加臂间 ≥60s 放锁窗口 ≈ 8 min ⇒ 训练 ≈ 40 min；评测/成本探针 ≈ 5 min。
- **预算不够时砍什么**（§8）：**先砍 seed1 四臂**（退化为引用 `2625` 噪声底，如实声明），
  **再砍合成腿**；**不砍判据线、不改配方、不缩 S**。

## 9. 数据与脚本清单（SCHEMA §6/§12）

- 数据集：`enwik8_full`（`data/enwik8_full.npz`；前 90M train / 末 10M eval；vocab 206）。
- 入口脚本：`run.py`（阶段 `preflight/train/train-all/eval/report/all`）；训练子进程 `clean_driver.py`。
- resolved config：`config.json`；原始结果：`results.json` / `eval.json` / `smoke/smoke.json`。
- 运行期引擎指纹：`run_fingerprints_backfill.json`（本次运行的**人工补记**：preflight 当时未自动落盘，
  由会话内证据（`2637/config.json` 等）反推；形状与 `run.py --stage preflight` 写的 `run_fingerprints.json`
  兼容、两者互不覆盖；report 优先读前者，缺失时回退到补记）。
- 对照与方差：`<arm>-train_summary.json`（八臂）+ `<arm>-train_log.jsonl`（逐臂轨迹，入库副本）。
- 对账：`reconcile_numbers.py` → `reconcile_report.json`；变异：`mutate.py` → `mutation_report.json`。
- 权重：`MODEL.md`（哈希 / 体积 / 生成命令；权重不入库）。

## 10. 失败决策树（SCHEMA §6，按测量 → 实现 → 规模 → 设计顺序）

1. **测量**：bpc 与 full/window 的差分经 `reconcile_numbers.py` 重算；仪器自检（释放版 vs 显式精确因果掩码，逐位一致）。
2. **实现**：QK-Norm hook 单测 + 变异验证（抓红且逐字节还原）；窗口掩码单测钉住非退化。
3. **规模**：S=6000 是共同半预算快照口径；若末窗斜率仍显著非零 ⇒ 只作快照，不称渐近。
4. **设计**：若窗口质量在长 T 仍越界 ⇒ 记「在该 T/尺度下窗口有税」，并写清「要什么条件才可能回本」。

## 11. 产出（SCHEMA §6）

`results.json`（统一四点表 + 四腿）/ 逐臂 `train_log.jsonl` / `analysis.md`（逐格表格，先人话后数据）/
`MODEL.md` / `eval.json` / `smoke/smoke.json` / `reconcile_report.json` / `mutation_report.json` /
`run_fingerprints_backfill.json`（运行期指纹的人工补记）。
