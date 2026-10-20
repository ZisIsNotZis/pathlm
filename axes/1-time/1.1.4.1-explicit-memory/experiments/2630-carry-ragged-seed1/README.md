# 2630 · carry × ragged 的 seed=1 复现（13M · 非 dense RAP 形态 · 合成 assoc-mqar）

> 契约来源：ticket 46 · W2 的**任务下发文本**（§4 逐字抄录，一字未改）。
> 本文件是该线的**执行契约**；§4a 落定操作口径、§4b 记录派发前的**能力面分析**
> （SCHEMA §22.8）。
> 入口：`train_arms.sh`（seed=1 训练两臂）+ `run.py`（评测电池 + 跨 seed 合并）+
> `smoke_steptime.sh`（先测步时）；语义层 `ragged_seed1_arms.py`（seed 命名 + 跨 seed
> 统计 + W2 判据②③；单因素配方**复用** 2629 的 `ragged_projfill_arms.py`）；单测
> `tests/test_carry_ragged_seed1_13m.py`；变异 `mutate.py`；对账 `reconcile_numbers.py`；
> 原始结果 `results.json`；判读 `analysis.md`；权重清单 `MODEL.md`；对照说明
> `controls/README.md`。

## 1. 命题

**46-W1（= 2629）是单 seed（seed=0）**：它测出「carry × 随机退出 × proj-fill」这条产品级
配方里 ① 分布内多查询召回无增益（+0.36pp，CI 重叠），② **越界腿**（单查询，距离 241–255，
训练分布外）carry 效应 **+6.05pp（n=2048）** —— 后者虽 CI 不重叠，但**样本小、只有一次
独立训练**。本线问：这两个读数是否**跨 seed 稳定**？

一句话可证伪的形式：*把 seed=1 的两臂（有/无 carry，其余逐字段与 2629 相同）与 2629 的
seed=0 合并为 n=2；若越界腿的 carry 效应在**两个 seed 上都为正且各自差值 CI 下界 > 0**，
则「该通道确实在搬远距离信息」记为**稳健**（仅该腿）；若分布内召回的 carry 效应在**两个
seed 上都落在 1.4pp 方差带内**，则「搬运通道对召回冗余」的结论**稳定**。*

## 2. 单因素声明

- **基底**：本仓**非 dense RAP 形态**（`DepthARModel(dense=False, topology="sequential")`
  = 逐通道携带门 + 随机退出深度 + 混合填充），13.9M（d256 × 12 层 × 8 头 × mlp6 × T256）。
  与 2629 的种子臂**逐字段相同**（除 seed）。
- **唯一变量 = `carry_rule` 的开关**（两臂恰好是 (关) / (delta 写)）；**跨 seed 层**的
  唯一变量 = 训练 seed（0→1），配方不变。

  | 臂 | seed | `carry_rule` | `--exit-mode` | `proj_fill` | `fill_kv` | `dense` | 块类 |
  |---|---|---|---|---|---|---|---|
  | `projfill_s1`（基准） | 1 | 关 | uniform（随机退出） | 开 | 开 | False | `Block` |
  | `projfill_carry_s1`（目标） | 1 | **delta** | uniform | 开 | 开 | False | `CarryChannelBlock` |
  | （复用）`projfill` | 0 | 关 | uniform | 开 | 开 | False | `Block` |
  | （复用）`projfill_carry` | 0 | **delta** | uniform | 开 | 开 | False | `CarryChannelBlock` |

- **单因素语义层复用 2629**：`model_config_kwargs` / `train_flags` /
  `expected_block_class` / `native_visibility` 直接 `import` 2629 的
  `ragged_projfill_arms.py`（`ragged_seed1_arms.py` 把 2629 目录加入 `sys.path`），
  **不复制**⇒ 两 seed 的配方不可能出现「定义漂移」。
- **训练/评测可见性一致**：两臂**都没有**训练期可见窗口（`train_window > 0` 与
  `dense=False` 硬互斥，见 §4b）⇒ 评测口径 = 全因果；长程压力来自**随机退出 + 填充**的
  逐位置可见性差异，不是「源 key 被推窗外」。
- **无「必要性例外」**：两臂的任务、token 预算、评测批、退出模式逐字段相同（含 seed=1）。

## 3. 干预定义（写清是「实现」而非「标签」）

本线**不改任何引擎代码**（不写 `pathlm/`、不写 `train_depth_ar.py`），只把既有开关
组合成 seed=1 的两臂；新增的只有本实验自己的语义层 `ragged_seed1_arms.py`（跨 seed 层）。

- **seed=1**：`train_depth_ar.py --seed 1` 同时决定**模型初始化**（`torch.manual_seed`）
  与**训练数据流**（`torch.Generator().manual_seed(seed)`）。
- **carry（`--carry-write delta`）**：单共享记忆槽
  `S_t = S_{t−1}(I − β k̂ k̂ᵀ) + β v k̂ᵀ`；每层读出 `a ← a + γ ⊙ mem_t`，γ 单边零初始化。
- **随机退出（`--exit-mode uniform`）**：逐位置采样 `d_i ∈ U{0..L}`（两臂同款）。
- **proj-fill（`--proj-fill`）**：`fill_kv` 的混合 KV 源在已退出位置上再经恒等初始化的
  逐层 adapter remap。
- **评测口径与 2629 完全一致（同一件仪器 + 同一批位置）**：`run.py` 通过 importlib 载入
  2629 的 `run.py`（`R2629`）复用 `recall_multi` / `recall_single_ood` /
  `per_depth_agreement` / `clean_bpc` / `read_train_log` / `drop_last_block`；数据生成器
  seed 与模型 seed **无关** ⇒ seed=0/1 在同一批位置 + 同一退出模式上**成对**可比。

## 4. 判据线（**逐字**抄录任务下发文本，未改一字）

> 【W2 · GPU】carry×ragged 的 seed=1 复现（≈21 min）
> 目录：axes/1-time/1.1.4.1-explicit-memory/experiments/2630-carry-ragged-seed1/
> 动机：46-W1 是单 seed；越界腿 +6.05pp（n=2048）虽 CI 不重叠但小样本。
> 干预：同配置（carry×随机退出×proj-fill）**seed=1 重跑两臂**（有/无 carry），与 46-W1 的 seed=0 合并为 n=2。
> 判据：① 报两 seed 的分布内召回差与越界腿差（含 CI 与极差）；② 若越界腿效应跨 seed 保持 ⇒ 记「机制确实在搬」为稳健（**仅该腿、不外推到召回**）；③ 分布内召回冗余结论是否稳定；④ 禁外推。

### 4a. 执行时的操作性定义（不改判据，只把口径落定）

- **判据①「分布内召回差」** = `recall_indist_random_exit`（多查询 assoc-mqar，随机退出 =
  训练同款；距离 107–240 ≤ 训练上限）的 carry 效应 = 目标 − 基准（pp）；并列给去伪影块
  （去掉距离 226–240）与全深度口径。**「越界腿差」** = `ood_single_query_full_depth`
  （单查询，距离 241–255 > 训练上限；与 2629 `increments.single_query_ood` 同口径 = 全深度）
  的 carry 效应；并列给随机退出口径。每个效应都附 `two_proportion` **差值 95% CI** 与
  **跨 seed 极差**（`pp_range`）。
- **判据②「越界腿效应跨 seed 保持」**（预登记，只此一腿）：
  1. 至少两个 seed；
  2. 每个 seed 效应 **同号为正**；
  3. 每个 seed 的**差值 CI 下界 > 0**（统计检验）。
  证据类型 = `ragged_seed1_arms.OOD_ROBUST_EVIDENCE`（**统计检验**）。**用词不得超出证据**：
  只记「该腿的 carry 效应跨 seed 保持」；**不外推到召回**（召回是另一条腿、另一个判据）。
- **判据③「分布内召回冗余结论是否稳定」**：每个 seed 的分布内 carry 效应**都在方差带
  1.4pp 内** ⇒ 「冗余」结论稳定。证据类型 = **量级**（`INDIST_STABLE_EVIDENCE`；
  `harness/methodology.md`：Δ<0.014 不可排序）。**不做等价比/非劣检验**，也不把「不显著」
  说成「相等」。
- **禁外推（判据④，§17.11）**：只含本次两点（seed=0/1）/ 13M / 6k 快照 / 无训练期窗口；
  **不得**推断其它 seed、其它尺度、其它退出模式、或「window × carry × 非 dense」格。
- **辅助（非 W2 预登记判据）**：出口资产 `tf@4`（synth 分布内主口径 + enwik8 跨分布代理）
  的跨 seed Δ——2629 §9 把 enwik8 代理 −4.71pp 列为「待确认」，本线顺带用 seed=1 检查它
  是否同号/同量级；**只作辅助，不构成判据**。
- **基准唯一（§17.14）**：每个 seed 内基准 = 同 seed 的无 carry 臂；跨 seed 判据只读
  **每 seed 的 carry 效应**，不与裸 stock 比。
- **收敛门禁（§17.15）**：答案行 CE（`synth_answer_ce`）末窗（末 1/4）斜率；仍陡降 ⇒
  只读臂间增量。

## 4b. 能力面分析（SCHEMA §22.8，派发前逐条做）

契约要求的每个 flag/组合，引擎是否支持？本线**与 2629 同配置**，能力面前提由 2629 已核对
（`grep -n "ValueError\|互斥" pathlm/depth_ar.py`）：

| 组合 | 引擎现状 | 本线 |
|---|---|---|
| `carry_rule` ⊕ `dense=False`（随机退出） | **ENG2 后允许** | `projfill_carry_s1` 臂 |
| `proj_fill` ⊕ 非 dense | **允许**（非 dense 前向 `kv_from = where(active, h, adapter(frozen))`） | 两臂 |
| `carry × 非 dense × 随机退出 × proj-fill`（四者齐开） | **可建可前向**（`capability_probe()` + 单测钉住） | 目标臂 |
| `train_window > 0` ⊕ `dense=False` | **仍硬互斥**（`raise ValueError`） | 未用（未测因子） |
| `proj_fill` ⊕ `fill_kv=False` | **报错**（`proj_fill requires fill_kv=True`） | 两臂均 `fill_kv=True` |

**seed 层的能力面**：`--seed` 进 `train_depth_ar.py` 的 `torch.manual_seed`（模型初始化）
与训练数据流 `Generator(seed)`；两臂与 2629 同款；无新 flag、无新组合 ⇒ 无新互斥面。

**未测因子（显式声明，禁止外推）**：本线**没有**训练期可见窗口（非 dense 路径不允许）
⇒ 强长程压力（源 key 被推窗外）在本线**不存在**。本线结论**不得**外推到
「window × carry × 非 dense」格。

> 本实验**不改引擎绕开互斥**（SCHEMA §22.8 禁止在未获授权时改引擎）。

## 既有做法（prior art，SCHEMA §18）

> **来源标注纪律**：本环境**联网检索不可用**（未注册 web / 检索工具）。外部条目一律标
> `记忆·未核`，只作设计方向、**不得当判据**；本地条目给文件路径。

| # | 谁做过 | 为什么这么设计 | 实测结果 | 我们的增量 |
|---|---|---|---|---|
| 1 | **本地 2629（46-W1）** `2629-carry-ragged-projfill/analysis.md` | 产品级组合（carry × 随机退出 × proj-fill）首次可测 | 分布内召回 +0.36pp（CI 重叠）；越界腿 +6.05pp（n=2048）；enwik8 tf@4 −4.71pp | 本线**只加一个 seed**（=1），把 n=1 变 n=2，检验这三条的跨 seed 稳定性 |
| 2 | **本地 2625** `2625-ood-leg-variance/analysis.md` | 量「近 OOD 判决腿」的 run-to-run 方差（同配置 n=3） | 该腿极差 10.84pp ≥ 结论效应 3pp ⇒ 永久降级为辅助腿（§17.18） | 本线的越界腿**正是**该腿族：2625 说明它在 GQA+window 配置下**高方差**；本线在 **carry×ragged（无 window）**配置上量它两 seed 的效应是否保持 |
| 3 | **run-to-run 方差 / seed 敏感性**（记忆·未核：小模型短训练的 seed 方差通常可达几个 pp；ML 复现文献常报单 seed 结论不可靠） | 单 seed 结论可能是初始化幸运 | 未核 | 本线把「单 seed 摘要」补成「两点 + 极差 + 每点 CI」，并按 §17.11 明确不外推 |

**若可检索应查**：`seed variance small transformer long-range recall`、
`replication crisis single-seed ML`、`delta rule memory channel OOD distance generalization`。

## 5. 对照与方差

- **基准 = 同 seed 的无 carry 臂**（`projfill_s1`，seed=1）；判据是**单因素对照**
  （目标 − 基准），不与裸 stock 比。
- **跨 seed 判据**：每 seed 一个效应 → 极差 / 均值 / 样本 sd；判据②用**统计检验**
  （差值 CI 下界 > 0），判据③用**量级**（1.4pp 方差带）。**不混用**。
- **方差带 = 0.014 bpc / 1.4pp 召回**（`harness/methodology.md`）。
- **局部先验地板** = `local_prior_accuracy(k=8)`（§19，应 ≈ chance）；**chance** = 1/64。
- **逐臂毁源对照**（§4 (1) 的护栏）：任一臂毁源后 CI 下界 > 4×chance ⇒ 该臂读数作废。
- **纯推理、无随机采样**：同 seed 评测应逐位相同；两臂（及两 seed）**成对**（同一批 +
  同一退出模式）。
- **逐字段对照表（seed=1 两臂；跨 seed 层只有 seed 变）**：

  | 字段 | `projfill_s1` | `projfill_carry_s1` | 2629 seed=0 两臂 |
  |---|---|---|---|
  | seed | 1 | 1 | 0 |
  | 训练数据 | assoc-mqar（K8/Q8/gap14） | 同 | 同 |
  | 步数×批×T | 6000×32×256 | 同 | 同 |
  | lr / 优化器 / schedule | 1e-3 / AdamW+cosine | 同 | 同 |
  | `exit_mode` / `fill_kv` / `proj_fill` / `final_weight` | uniform / True / True / 2.0 | 同 | 同 |
  | `dense` / `topology` | False / sequential | 同 | 同 |
  | **`carry_rule`** | **None** | **delta** | 同（None / delta） |

## 6. 成本（§8 的唯一预算数字 + 砍什么）

```
gpu_wall_budget: 21 min   # 含两臂训练 + 评测（纯推理）；步时先测后定
```

- **实测步时**（先测后定，§17.8/§20.3；`smoke/steps300/`，300 步）：
  `projfill_s1` **76.3** / `projfill_carry_s1` **134.4** ms/step（step0→299；末段 200→299 =
  72.7 / 151.5）⇒ 6k×2 推算 ≈ **21 min**（与任务下发一致）；**不借用** 2629 的步时。
- **实测墙钟**：`projfill_s1` 7.63 min；`projfill_carry_s1` 训练循环 12.70 min + 训练后
  诊断相位 ~47.7 min（**外部 GPU 争用**，非本仓负载，§20.3 记账）；等锁 11.27 min；评测
  0.466 min。⇒ 实测总墙钟约 83 min，**远超 21 min 预算**，主因是外部争用（见 `analysis.md §7`）。
- **步数预算的依据**：与 2629 **严格同预算**（6k×32×256 = 49.0M token）——本线是复现，
  **不砍步数**（保「与 seed=0 同刻度」，§20.3）。
- **预算不够时砍什么**（§17.7）：若实测步时把 21 min 撑破，**先砍评测批数**（非判据的
  辅助腿优先：clean bpc / enwik8 tf）> 再据实上报；**不砍判据线**（分布内召回 + 越界腿 +
  跨 seed 合并必须出）。
- **§20.1 锁记账**：`train_arms.sh` **按臂**取/放锁、臂间 65s 窗口（不整程独占）；
  评测 <1 min、峰值 <1GB，取一次短锁。

## 7. 数据与脚本清单

- **训练数据**（§19：不用 enwik8 当长程主任务）：
  `tools/evalkit/longrange_tasks.py::assoc_recall_multi_batch`（K=8 对、Q=8 查询、gap=14）。
- **评测**（合成长程任务）：`assoc_recall_multi_batch`（主判据腿 + 毁源对照）、
  `assoc_recall_batch`（越界腿，距离 241–255）。
- **clean bpc 腿**：`data/enwik8_full.npz` 的评测尾段（vocab 206）。
- **出口资产**：`pathlm/depth_ar.py::tf_acceptance`（enwik8）+ `run.py::per_depth_agreement`
  （synth 分布内）。
- **入口**：`train_arms.sh`（训练）、`eval.sh`（取锁 + 冒烟 + 全电池；内部跑
  `python3 run.py`）、`python3 run.py`（评测 + 跨 seed 合并，读 `config.json`）、
  `smoke_steptime.sh`（先测步时）。
- **落盘**：`results.json`（含 `cross_seed` 合并块）、`train/<arm>/{train_log.jsonl,
  gate_log.jsonl,results.json}`、`train_<arm>.log`、`train_arms.runlog`、
  `smoke_steptime.runlog`、`eval.runlog`、`analysis.md`、
  `MODEL.md`、`controls/README.md`、`reconcile_numbers.py → reconcile_report.json`、
  `mutate.py → mutation_report.json`、`smoke/`（`steps300/` 步时 + `results.json` 冒烟）、
  `pytest_full.txt` / `pytest_line.txt`。
- **测试**：`tests/test_carry_ragged_seed1_13m.py`（9 条可失败单测 + 变异 `mutate.py`）。

## 8. 失败决策树（测量 → 实现 → 规模 → 设计）

1. **测量**：跨 seed 合并是否读了**原始计数**（hits/n）而非 2629 的派生字段？两 seed 是否
   同一批位置 / 同一退出模式？chance/局部先验是否同值？→ 先修 `run.py` 的接线，**不得**动模型。
2. **实现**：seed=1 目标臂 `read_share` 恒 0（通道没被用）或训练日志缺 `attn_carry` →
   跑变异验证排除死开关；`read_scale` 仍单边零 = **步数**问题，如实记录、不外推。
3. **规模**：两臂都 ≈ chance ⇒ 该任务/该尺度上（无窗口）注意力本身没学会 ⇒ 只能读
   「臂间增量」，**不得**表述为「搬运不可能」（SCHEMA §19）。
4. **设计**：若 seed=1 的越界腿效应**反号或 CI 含 0** ⇒ **不得**记「稳健」（这正是
   §17.18 / 2625 的教训：该腿族可能高方差）；如实报「越界腿效应**不跨 seed 保持**」，
   并把 46-W1 的 +6.05pp 降级为「单 seed 观测」。

## 9. 产出与复核

- `analysis.md` 给**人话摘要 + 七问**（§23）+ 跨 seed 表格 + 判读；`MODEL.md` 给权重哈希
  与代码指纹；`controls/README.md` 给对照臂与变异说明。
- 本实验完成后按 SCHEMA §14（含 §14b 顺序屏障）另派 fresh-context **只读复核者**独立检查。

## scope 声明

本研究**只做单轴单格实验**，基底恒为 stock 对照；本实验是 M1 载体类型格内 **2629 的
同配置 seed=1 复现**（唯一变量 = 训练 seed；配方内唯一变量仍是 `carry_rule` 的开关）。
`attn_sparse_k` / `attn_linear` / `train_window` 与上述因子的交互仍受引擎互斥限制，是显式
留白、不是遗漏；「window × carry × 非 dense」是**显式未测因子**（§4b）。`deps` 只记录已知
耦合方向（M1 载体类型 ← 2629 产品级组合 / 2625 近 OOD 腿方差 / ENG2 引擎解耦）。
