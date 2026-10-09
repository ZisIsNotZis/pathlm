---
name: 2637-linear-attn-quality
ticket: 52-A6L
date: "2026-10-09"
grid: A6-mixing-operator（线性注意力质量腿）
engine: depth-ar `--dense --attn-linear`（LinearAttnBlock，φ=elu+1 归一化线性核，分块 O(T) 实装）
budget: "gpu_wall_budget: 40 min（含冒烟四臂 + 训练四臂 + 诊断矩阵 + 电池）；峰值 ≤5GB net"
status: judged
---

# 2637 · 线性注意力质量腿（A6 · 7.16M · depth-ar/dense × LinearAttnBlock）

> 本文件即本实验的本地契约（判据线预登记，不得事后调整）。原始结果：`runs/<arm>/`、
> `eval.json`（**preflight 冒烟与诊断矩阵未跑 ⇒ 无 `smoke/smoke.json` / `diag_matrix.json`**，见 §8）；
> 判读：`analysis.md`；权重清单：`MODEL.md`。

## 0. 要解决的问题（承接 `50-critical-audit` 的 A 轴缺口与 `51-tree` WI-12）

`2616` 判「线性注意力在当前实装下不可行」：三次训练期在 step 800/~500/400 **停摆**
（faulthandler 栈落在 `loss.backward()`；停摆前 `den_min` 触 clamp 下限 1e-6、`phi_min` 触 0、
gnorm 尖峰 5.1e3 vs 基线 0.4–0.8）。审计（`A-AUDIT.md` §5-7）指出这是**工程债**（独立进程二分跑通
3000 步），不是机制级结论。**线性注意力的质量（bpc）至今完全空白**。

本实验把「质量腿」补上：要么让实装跑起来拿到 bpc，要么给出**完整停摆诊断**（触发步、栈、
诊断矩阵、要跑起来需要什么）。契约明令：**不得**用「业界不宜」或「未测」糊过去。

## 1. 命题（一句话、可证伪）

在同一配方（stock dense，d=256/L=8/h=8/mlp=4，T=256，**bs8**，seed0，lr 1e-3，`exit-mode uniform`，
enwik8 前 90M/末 10M，**共同步数 S=6000**）上，把注意力算子从**全因果 softmax** 换成
**归一化线性核**（φ = elu(x)+1，分块 O(T) 实装，**参数逐位相同、零新增参数**），问：

> ① `Δbpc = bpc_full − bpc_linear`（正 = 线性更好）落在哪个**带宽分类**（|Δ| < 0.014 ⇒ 不亏）？
> ② 原实装（`elu+1` + `1e-6` clamp）在 6000 步内是否**停摆**？若停摆，触发步、栈、诊断矩阵是什么？
> ③ **稳定化分母**档（给 qkv 挂逐头 L2 QK-Norm，零新增参数，把 φ 输入钉在有界区间 ⇒ φ 不下溢、
> `den` 不塌）能否消除停摆并给出质量读数？

## 2. 单因素声明与能力面分析（SCHEMA §22.8）

- **唯一变量 = 注意力算子**（`attn_linear` ∈ {False, True}）；T/bs/seed/lr/步数/数据全相同。
- `LinearAttnBlock` 与 `Block` **子模块创建顺序一致**（ln1→ln2→qkv→proj→mlp）⇒ 同 seed 下参数
  逐位相同、`params_added == 0`（单测钉住）。
- 线性注意力**只在 `dense=True` 且无 dist_pen/kv_from/gate_vec** 时开放（引擎显式 raise）；
  本实验 `--dense` ✅，其余开关全关。
- **稳定化分母档不是第二个机制变量，而是「可换实现」**（契约授权）：它只改 φ 的输入数值条件
  （QK-Norm 把 q/k 逐头 L2 归一，`φ` 输入 ∈ 有界区间，φ 恒 > 0），**不改核形式、不加参数**。
  为保持单因素，稳定化档下 **full 与 linear 两臂都挂同一 hook**（2×2 设计），故 QK-Norm 的影响
  可被差分掉。
- `grep -n "ValueError\|互斥" pathlm/depth_ar.py`：`attn_linear ⊕ train_window/attn_sparse_k/n_kv_heads/
  carry_rule` ⇒ 本实验只开 linear ✅。

## 3. 判据线（预登记，逐字；不得事后调整）

**符号约定**：`Δ = bpc_full − bpc_linear`（**正 = 线性更好**），全篇一致。

- **C1（质量，主判据）**：`Δ_no_qk`（原实装）与 `Δ_with_qk`（稳定化档）各按带宽 **0.014** 分类
  （|Δ| < 0.014 ⇒ 「不亏」，否则按符号记「线性更好/更差（越界）」）。**至少一档拿到 bpc** 才判 C1；
  两档都停摆 ⇒ C1「无数据」（不得记 pass）。
- **C2（成本，四腿）**：同会话实测 `ms/step`（linear/full 比值）、峰值显存、解析注意力 MACs/token/layer。
  预登记：**期望线性实装在此规格下更慢**（`2616` 实测 T=256 比值 2.58、T=2048 1.53），如实报。
- **C3（停摆，诊断）**：报原实装的停摆状态（触发步 / `faulthandler` 栈 / `den_min`/`phi_min` 轨迹）
  与**诊断矩阵**（chunk ∈ {32,64,256} × qk ∈ {none,global}，短程 1200 步）；若稳定化档跑通则记
  「停摆可被数值稳定化消除」，否则记「在当前实装下不可行」并给「要跑起来需要什么」。
- **必须不变（护栏）**：参数量（线性块与 full 块逐位同参）；`LinearAttnBlock` 的严格因果性
  （单测 `test_linear_attn_causal_only_no_future_leak`）；KV/状态形状解析式不减参数。

## 4. 既有做法（prior art，SCHEMA §18）

| # | 谁做过 | 为什么这么设计 | 它的实测结果 | 我们的增量 |
|---|---|---|---|---|
| 1 | Katharopoulos et al. 2020（Linear Transformers，elu+1）| 把 softmax 换成归一化线性核 ⇒ 每层 O(T)、缓存常数 | 短上下文质量持平、长上下文吞吐占优 | 本仓首次给出**质量 bpc**（此前只有成本数据 + 停摆诊断）|
| 2 | Qin et al. 2022（cosformer）/ 特征归一化族 | 非负 + 归一化特征映射改善表达 | 部分任务追平 softmax | 我们不做换核（那是另一个干预）；只做**分母数值稳定化**（QK-Norm 有界输入）|
| 3 | Gated DeltaNet / MQAR（线性注意力标准基准）| 用合成回忆任务判线性注意力的记忆能力 | 长程回忆是线性注意力的弱点 | 本实验只判质量 + 成本，召回腿留待后项 |

知识来源：`记忆·未核`（无检索工具，按 SCHEMA §21 不作判据，仅作设计方向）。

## 5. 成本（SCHEMA §7/§8；先测步时再定预算）

- `gpu_wall_budget: 40 min`（含冒烟、训练四臂、诊断矩阵、电池）。
- 臂数 = 2×2 = **4 臂** × 6000 步；`2616` 实测 linear T=256 bs32 ≈ 60 ms/step ⇒ bs8 更省。
  诊断矩阵另 6 条短程（1200 步）。
- **预算不够时砍什么**：**先砍诊断矩阵**（退化为只报原实装停摆证据）；**再砍稳定化档的双臂**
  （只留 linear 侧）；**不砍判据线、不改配方、不缩 S**。

## 6. 数据与脚本清单（SCHEMA §6/§12）

- 数据集：`enwik8_full`（`data/enwik8_full.npz`；前 90M train / 末 10M eval；vocab 206）。
- 入口脚本：`run.py`（阶段 `preflight/train/train-all/diag/eval/report/all`）；
  训练子进程 `linear_driver.py`（复用 2625 的 QK-Norm hook，单一实现来源）。
- resolved config：`config.json`；原始结果：`results.json` / `eval.json`；对账：`reconcile_numbers.py` →
  `reconcile_report.json`；变异：`mutate.py` → `mutation_report.json`；权重：`MODEL.md`。
  （**未产出**：`smoke/smoke.json`（preflight 未单独跑）、`diag_matrix.json`（停摆未复现，边际信息低）。）

## 7. 失败决策树（SCHEMA §6，测量 → 实现 → 规模 → 设计）

1. **测量**：bpc 经 `reconcile_numbers.py` 从 `eval.json` 重算；仪器自检（释放 vs 显式因果掩码逐位一致）。
2. **实现**：停摆诊断矩阵（chunk × qk）；`faulthandler` 栈；活性量轨迹（φ/den/gnorm）。
3. **规模**：只报 T=256/S=6000 一点，禁外推。
4. **设计**：若线性质量越界 ⇒ 记「该算子在该尺度有税」，并注明这是否与业界线性注意力结论冲突。

## 8. 产出（SCHEMA §6）

`results.json` / 逐臂 `train_log.jsonl` / `analysis.md`（先人话后数据）/ `MODEL.md` / `eval.json` /
`reconcile_report.json` / `mutation_report.json`。
**未产出**：`smoke/smoke.json`（preflight 未单独跑；四臂的 6000 步训练与 `gnorm_monitor` 已充当步时/活性面证据）、
`diag_matrix.json`（停摆未复现，边际信息低）——`run.py --stage preflight/diag` 可补。
