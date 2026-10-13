# 2625 · 近 OOD 判决腿的方差量化（13M · 合成长程任务 · GQA+carry+window16 × seed{0,1,2}）

> 契约来源：`.scratch/42-longt-stability/issues/01-longt-stability.md` 的「**W2**」。
> 本文件是该线的**执行契约**；§4 判据线**逐字**抄录 W2，执行期间未改。
> §4b 记录执行前由主 Agent 落定的**两处操作口径**（预算 / 参照臂的归属），未改判据线。
> 入口：`train_arms.sh`（训练 seed=1/2 两臂）+ `run.py`（评测电池，复用 2624 的评测仪器）；
> 语义层 `ood_variance_arms.py`；单测 `tests/test_ood_leg_variance.py`；变异 `mutate.py`；
> 对账 `reconcile_numbers.py`；步时先测 `measure_step_time.sh`；冒烟 `smoke.sh`。
> 判读 `analysis.md`；权重清单 `MODEL.md`；对照说明 `controls/README.md`。

---

## 1. 命题

**上游 Z3（ticket 41）发现：一条「近 OOD」判决腿（单查询关联回忆，钥匙距离 241–255，
超过训练上限 240）在**同一配置**下两次独立训练差 **15×**（0.93% vs 15.33%）。**
这使该腿是否还能当主判据存疑。一句话可证伪的形式：

> *同配置（GQA2+carry+window16，6k，13M）**独立重训 n=3**，这条判决腿读数的**极差**
> 是否 ≥ 结论效应（3pp）？* 若是 ⇒ 该腿**永久降级为辅助腿**（§17.18）；
> 同时报**训练分布内腿（≤240）**作对照，看抖动是任务级还是距离档级。

## 2. 单因素声明

- **基底**：本仓 stock dense decoder-only trunk（13.5M：d256 × 12 层 × 8 头 × mlp6 × T256，
  tied 读出 + 顺序残差），`DepthARModel(dense=True, topology="sequential")`。
- **唯一变量 = 一次独立训练（seed）**：GQA(`n_kv_heads=2`) + carry(`delta`) + window(16, 0)
  是**常量**；seed ∈ {0(reuse), 1, 2}；其余逐字段相同（6k×32×256、lr 1e-3、AdamW+cosine、
  合成任务 assoc-mqar(K=8,Q=8,gap=14)）。
- **无「必要性例外」**；**不改引擎**（本线不写 `pathlm/`、不写 `tools/evalkit/`）。

| 臂 | seed | 权重来源 | 角色 |
|---|---|---|---|
| `gqa_carry_win_s0` | 0 | 复用 Z3（2624）`carry_gqa` | n=3 主集（已有） |
| `gqa_carry_win_s1` | 1 | **本线新训** 6k | n=3 主集 |
| `gqa_carry_win_s2` | 2 | **本线新训** 6k | n=3 主集 |
| `y1_gqa_carry_win_s0` | 0 | 复用 Y1（2623）`gqa_carry_window` | **参照**（第二个同配置 seed=0 独立 run，不并入 n=3） |

## 3. 干预定义（写清是「实现」而非「标签」）

- **GQA（`--n-kv-heads 2`）**：KV 头 8→2（`ComposableBlock` 的 KV 头投影，`enable_gqa=True`）；
  真进 SDPA 的 K/V 形状 `[B,2,T,32]`（活性量）。
- **carry（`--carry-write delta`）**：单共享记忆槽 `S_t = S_{t−1}(I − β k̂ k̂ᵀ) + β v k̂ᵀ`；
  每层读出 `a ← a + γ ⊙ mem_t`，γ 单边零初始化。
- **window（`--train-window 16 --train-anchors 0`）**：`attn_mask` 是与块类正交的装饰层，
  把源 KV 段 `[0,2K)` 推到窗外 ⇒ 不搬就答不出。
- **seed（唯一变量）**：`train_depth_ar.py --seed` 同时决定**模型初始化**（`torch.manual_seed`）
  与**训练数据流**（`torch.Generator().manual_seed(seed)`）。
- **评测口径**（三套，并报）：① 判决腿（单查询，距离 241–255，native window16）；
  ② 训练分布内腿（多查询，距离 107–240）；③ 全因果交叉口径（去窗口，距离 241–255）。

## 4. 判据线（**逐字**抄录 W2 契约；仅把已废的台账指代替换为「写入 `catalog.yaml`」）

> ## W2 · 近 OOD 判决腿的方差量化（≈26–40 min GPU）
> - 目录：`axes/M-carrier/M1-carrier-type/experiments/2625-ood-leg-variance/`
> - 动机：Z3 发现该腿（距离 241–255，**超过训练上限 240**，违反 §17.18）同配置两次独立训练差 **15×**（0.93% vs 15.33%）。
> - 干预：同一配置（GQA+carry+window16）**seed=1 与 seed=2 各一跑**（≈11 min/臂，按臂释放锁），
>   加上已有 seed=0 共 **n=3**；评测在**同一批位置**上做。
> - **判据（预登记）**：① 报三点（含 CI）与**极差**；② 若极差 ≥ 结论效应（如 ≥3pp）⇒ 该腿**永久降级为辅助腿**并写入 `catalog.yaml`；
>   ③ 同时报**训练分布内腿**（≤240）作为对照（若它也高方差，则该任务整体不可用于 n=1 判决）；④ 禁外推。

### 4a. 执行时的操作性定义（不改判据，只把口径落定）

- **极差** = `100×(max−min)`（pp）；**结论效应阈值** = **3pp**（W2 明文「如 ≥3pp」）。
- **方差带 = 1.4pp**（`docs/methodology.md`）只用于「可排序性」，**不**用于降级判定。
- **基准唯一（§17.14）**：n=3 主集 = {seed0, seed1, seed2}；参照臂**不并入**，只作跨会话参照。
- **基准臂 / 分量分解（§17.17）**：判决腿（超分布）与分布内腿分列，**不得**合并成「有效/无效」。
- **毁源对照**（强制）：任一臂毁源后 CI 下界 `> 4×chance` ⇒ 该行作废。
- **收敛门禁（§17.15）**：答案行 CE `synth_answer_ce` 末窗（4500→5999）斜率；仍陡降 ⇒ 只读臂间。

### 4b. 落定（**主 Agent 裁决**）

1. **n=3 = {复用 Z3 的 seed=0} + {本线新训 seed=1, seed=2}**：契约写「加上已有 seed=0」，
   指 2624（Z3）的 `carry_gqa`（同配置、同 6k 预算、同评测仪器），故 s0 复用其 checkpoint。
2. **另设一条参照臂** `y1_gqa_carry_win_s0`（Y1 的 `gqa_carry_window`）**不并入 n=3**：
   它与 s0 **同为 seed=0 却是独立一跑**，正好用来验「方差是否来自 seed 标签」；
   作为参照行单列，避免把「同一 seed 的两跑」伪装成「两个 seed」。
3. **预算**：沿用 6k（与 s0/参照臂严格同预算，§20.3），**不砍臂、不换配方**。

## 5. 指标表（多维，逐项分析）

| 维度 | 本线怎么量 | 有效性的边界 |
|---|---|---|
| 解决什么问题 | 量「同配置重训」在该判决腿上的抖动，判它能否作主判据（§17.18） | 只在 13M / 合成任务 / 6k 快照上 |
| 代价 | 两次 6k 训练（≈23 min GPU）+ 一次评测（≈1 min） | — |
| 有效性（vs 对照） | 分布内腿（≤240）+ 全因果交叉口径 + 参照臂（第二跑）+ 毁源 | 全部同批位置、同仪器 |
| 收益与用法 | 给「该腿降级」提供 n=3 证据；给「分布内腿可用」提供对照 | 用法只在本任务/本尺度上成立 |
| 可控性 | 唯一变量 = seed；逐字段对照见 §2 | — |
| 副产品 | 仪器一致性证据（对 s0/参照臂逐位复现 2624/2623 的既有读数） | SCHEMA §9 |

## 6. 成本（§8 的唯一预算数字 + 砍什么）

```
gpu_wall_budget: 40 min   # 含冒烟 + 两臂训练（§20.1 按臂释放）+ 评测
```

- **先测步时再定预算**（§17.8/§20.3）：执行前实测 200 步 = **129 ms/step**
  （卡上有外部负载：跑前 free 20741 MiB、util 56%），⇒ 6k ≈ 12.9 min/臂 ×2 ≈ 26 min。
  实际：s1 **12.26 min**、s2 **11.04 min**（见 `train_arms.runlog`）。
- **峰值显存（逐臂实测）**：训练 **2193.6 MB**（两臂同）< 5GB 礼貌线；评测 **580.4 MB**。
  跑前/跑后 `nvidia-smi memory.free` 见 `train_arms.runlog` 与 `analysis.md §4.6`。
- **§20.1 锁**：`train_arms.sh` **按臂**取/放、臂间 **65s** 窗口（时间戳见 `train_arms.runlog`）；
  **评测**为 §20.1 第 2 条**一次性并发例外**（纯推理、≤2min、峰值 <1GB、空闲 ≥3GB），未取锁。
- **预算不够时砍什么**（§17.7）：砍 seed 数（n=3 → n=2）> 砍距离档；**不砍判据线**。

## 7. 数据与脚本清单

- **训练数据**（§19：**不用 enwik8 当长程主任务**）：
  `tools/evalkit/longrange_tasks.py::assoc_recall_multi_batch`（K=8, Q=8, gap=14）。
- **评测任务**：
  - 判决腿（近 OOD）：`assoc_recall_batch`（单查询，K=8，距离 241–255，native window16）；
  - 训练分布内腿：`assoc_recall_multi_batch`（多查询，距离 107–240）；
  - 全因果交叉口径：`assoc_recall_batch`（keep_mask=None）；
  - 距离曲线（辅助）：`var_recall_batch`（d ∈ {16,32,64,128,192,255}）；
  - 毁源：`corrupt_span([0,16))`。
- **评测仪器**：复用 2624 的 `quant_carry.py`（与 2618/W3/2623 同一件仪器；`fp32` 档与释放版
  逐位相等，本线自检钉住）。**本线不新增语义层**。
- **入口**：`bash train_arms.sh`（训练两臂）、`python3 run.py`（评测，`config.json`）。
- **落盘**：`results.json`、`train/<arm>/{train_log.jsonl,gate_log.jsonl,results.json}`、
  `train_<arm>.log`、`train_arms.runlog`、`analysis.md`、`MODEL.md`、`controls/README.md`、
  `reconcile_numbers.py → reconcile_report.json`、`mutate.py → mutation_report.json`、`smoke/`。
- **测试**：`tests/test_ood_leg_variance.py`（12 条可失败单测 + 变异 `mutate.py`）。

## 8. 失败决策树（测量 → 实现 → 规模 → 设计）

1. **测量**：四条臂是否评同一批位置？→ 查 `eval_batch_seeds` 与 `reconcile` 的任务对账；
   不一致先修 `run.py`，**不得**动模型/判据。
2. **实现**：块类/参数量/活跃开关是否真？→ 跑变异排除死开关；`run.py` 对块类不符直接退出。
3. **规模**：读数都 ≈ chance ⇒ 该任务/尺度上通道本就没起作用（读 2618/2623 作前提）；
   **不得**表述为「机制不可能」（§19）。
4. **设计**：**本线不改设计**（只量方差）。

## 9. 产出与复核

见 §7 落盘清单；本实验完成后按 SCHEMA §14（含 §14b 顺序屏障）另派 fresh-context
**只读复核者**独立检查。

## 既有做法（prior art，SCHEMA §18）

> **来源标注纪律**：本环境**联网检索不可用**。外部条目一律标 `记忆·未核`，只作设计方向、
> **不得当判据**；本地条目给文件路径。详见 `analysis.md §7`。

| # | 谁做过（记忆·未核） | 为什么这么设计 | 实测结果 | 我们的增量 |
|---|---|---|---|---|
| 1 | ML 复现性/方差研究（记忆·未核） | 报告同配置重跑的读数抖动，界定最小可分辨效应 | 本仓 `docs/methodology.md` 已写方差带 1.4pp | 本线把「同配置重跑」用在**一条具体判决腿**上，量出 10.8pp 极差并据此**降级一条腿** |
| 2 | CUDA 非确定性（记忆·未核） | 同 seed 同数据仍得不同权重（原子归约） | 2624 `stability.md §3` 已观测同 seed sha256 不同 | 本线**量化**：四跑四个 sha256，判决腿 0.928→15.332% |
| 3 | 分布内 vs 外推评测（记忆·未核） | 训练分布外的指标不代表机制能力 | §17.18 已立为纪律 | 本线给出**实证**：超分布档不可排序、分布内档可排序 |

**若可检索应查**：`seed variance benchmark deep learning`、`CUDA nondeterminism training
reproducibility`、`out-of-distribution evaluation small transformer`。

## 能力面分析（SCHEMA §22.8，派发前逐条做）

`grep -n "ValueError\|互斥" pathlm/depth_ar.py` 列出全部校验，逐条核对本线需要的 flag 组合：

| 组合 | 引擎校验 | 本线 |
|---|---|---|
| `carry_rule` × `n_kv_heads<8` | ENG1 后允许（`ComposableBlock`，`depth_ar.py:1853-1860`） | ✅ 四条臂 |
| `n_kv_heads<8` × `train_window>0` | ENG1 后允许（窗口是正交装饰层） | ✅ 四条臂 |
| `carry_rule` × `train_window>0` | 允许（2618 已用） | ✅ 四条臂 |
| 三者齐开 | ENG1 后可建可跑（2622 钉住） | ✅ 四条臂 = `ComposableBlock` |
| `attn_sparse_k` / `attn_linear` ⊕ 上述任一 | 仍互斥 | 本线不用 |

⇒ **不存在不可达组合**；`run.py` 加载时逐臂断言块类 = `ComposableBlock`（不符即退出）。
本线**不改** `pathlm/` / `tools/evalkit/` / 任何共享文件。

## scope 声明

本研究**只做单轴单格实验**，基底恒为 stock 对照；本实验是 M1 载体类型格内、**判决腿方差**
这一方法学问题的量化（同配置 seed{0,1,2} + 参照臂），**不做组合实验**；**不改任何机制/设计**。
评测期 KV 量化（2624/Z3 的面）不在本线 scope。`deps` 只记录已知耦合方向
（M1 载体类型 ← 2624/Z3 的读数；与 2623/Y1 的判决腿仪器耦合）。
