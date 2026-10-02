# 2619 · 写入通道的难度—容量曲线（13M · 合成关联回忆）

> 契约来源：`.scratch/36-parallel/issues/01-parallel.md` 的「波二 **U1 线**」。
> 本目录实现它。**判据线在 §4 预先写下，读数之前不得修改。**
> 先 20 步冒烟（`smoke/`）再全跑；GPU 工作全程持 `/tmp/pathlm_gpu.lock`（SCHEMA §20）。

## 1. 命题（人话）

ticket 35 已经证明：**给模型一个显式的写入通道 + delta 写规则，它就能把「训练窗口外」
的信息搬过来用**（关联回忆准确率 0.0137 → 0.0950 = 6.08× chance；毁源后掉回 0.0081）。
但那只是一个**点**。本实验把「能/不能」变成**两条定量曲线**：

- **难度曲线**：让它要记的东西变多（关联对数 `K` ∈ {4,8,16,32}），准确率怎么变？
- **容量曲线**：让它记事的「本子」变大或变小（记忆槽状态宽度 `dh` ∈ {16,32,64,128}），
  准确率怎么变？

一句话可证伪的形式：*「在固定训练配方下，准确率随 `K` 单调下降、随 `dh` 单调上升
（或各自出现明确拐点）；且存在 ≥3 个可实测的「状态宽度 ÷ 需回忆信息量」交换点。」*

## 2. 单因素声明

- **基底**：本仓 stock dense decoder-only trunk（`DepthARModel(dense=True)`：tied 读出 +
  无 final LayerNorm + 顺序残差），13.5M（d256×12 层×8 头×mlp6×T256），**写规则恒为
  delta**（ticket 35 判入的那条）。
- **唯一变量 = 记忆槽的状态宽度 `dh`**（`--carry-state-dim`，新增开关，见 §3）。
  四臂**只差这一个字段**（其余逐字段一致，含 seed=0）。
- **注意力路径逐位不动**（权重/头数/掩码都不变）⇒ 曲线只能归因到状态宽度，不会与注意力
  混在一起。**单测 `test_carry_state_dim_decouples_state_from_attention_heads` 钉住**
  （主干张量逐位相同，只有 `carry.*` 与每层 `carry_scale` 不同）。
- **必要性与取舍（显式记账，SCHEMA §17.8）**：`n_heads` 也能改出 {16,32,64,128} 这组
  状态宽度（`dh = 256/n_heads`），但那样会**同时改注意力的头维度** ⇒ 破单因素归因。
  因此新增 `carry_state_dim`，把状态宽度与注意力头**解耦**；缺省 `None` = v1 口径
  （`dh = d_model/n_heads`），**与 ticket 35 的 U1/U2 逐位相同**（单测钉住）。
  代价：状态宽度变化时写门表 `beta_w` 形状随记忆槽头数变化，**参数量有 ±0.03% 的差**
  （实测 13,803,216 → 13,799,450，见 §7），不是严格等参数；已逐臂落盘并在判读时声明。
- **必要性例外（显式记账）**：训练与评测都用 `train_window=16 / train_anchors=0`
  的可见性掩码（ticket 32-O 既有开关），制造「必须搬」的需求。这不是第二个机制，而是
  本契约指定的任务设定（四臂完全相同）。
- **步数取舍（显式记账）**：ticket 35 用 12k 步；本实验受契约 `≈60 min` 预算约束，
  四臂统一用 **10,000 步**（实测 ≈82 ms/步 ⇒ 4 臂 ≈55 min）。损失仍未收敛
  （答案 CE 单调下降），因此本实验报的是**同一预算下的横向比较**，不是收敛值。

## 3. 干预定义（实现，不是标签）

**状态宽度 `dh`**（`pathlm/depth_ar.py::DepthARConfig.carry_state_dim`）：单共享记忆槽
`CarryChannel` 的每个头把 `dh` 维键映到 `dh` 维值，状态 `S ∈ R^{H_c×dh×dh}`，
其中 **记忆槽头数 `H_c = d_model / dh`**（与注意力头数 `n_heads = 8` 无关）。
总状态浮点 = `H_c·dh² = d_model·dh` ⇒ `dh` 翻倍、状态翻倍。

```
写入（每位置一次，来自输入流 h0）：k̂=normalize(W_k h0)、v=W_v h0、β=σ(W_β h0+b_β)
  delta：S_t = S_{t−1}(I − β_t k̂_t k̂_tᵀ) + β_t v_t k̂_tᵀ
读出（每层各一次）：a ← a_SDPA(窗口掩码)；a ← a + γ_ℓ ⊙ mem_t，mem_t = S_t q_t
```

- **读出约定**：位置 t 读到 `S_t`（含 t 自己的写入，read-after-write），与 ticket 35 同
  （单测 `test_carry_read_convention_includes_own_write`）。
- **因果性**：位置 t 只读 `S_{≤t}`（单测 `test_carry_channel_is_causal_only`）。
- **恒等起步**：每层读出增益 `γ_ℓ` 零初始化 ⇒ 起步逐位恒等于无通道基线。
- **读出接线的等价改写（逐位）**：读出从「注意力头空间相加」改写成「`[B,T,d]` 空间相加」
  （`γ` 按记忆槽头展开到各自 `dh` 通道）。缺省 `dh = d/n_heads` 时两者**逐位相同**
  （加法与 reshape 可交换）——单测 `test_carry_state_dim_default_and_explicit_n_heads_bitwise_equal`
  钉住；这也让容量轴换记忆槽头数时读回仍然成立。

## 4. 判据线（**预登记**，读数前写下）

主腿用**准确率口径**（不用 bpc——训练侧 bpc 噪声已接近方差界）：

1. **前提校验**：`dh=32 / K=8` 臂（= ticket 35 的 U2 口径）的准确率必须显著高于 chance
   （chance = 1/64 = 0.015625），否则说明通道没活起来 ⇒ 先查活性量再判。
2. **难度曲线**（固定 `dh=32`，`K ∈ {4,8,16,32}`，**4 个直接测点，禁止外推**）：
   必须**单调不升**（容许相邻点 CI 重叠 ⇒ 记「不可分辨」）**或**给出**明确拐点**。
3. **容量曲线**（固定 `K`，`dh ∈ {16,32,64,128}`，**4 个直接测点，禁止外推**）：
   在 `K=8`（训练同档）与 `K=32`（加压档）**各画一条**，同样要求单调或不出现拐点即如实报
   「该段与二项 CI 不可分」。**哪个 `K` 上出现拐点就把它当容量曲线的主腿**（预登记：
   若 `K=8` 全平而 `K=32` 有拐点，主腿取 `K=32`——容量压力只有在需求接近容量时才可见）。
4. **交换率实测点（≥3 点，禁止外推/拟合线外推）**：定义
   `K*(dh)` = 使该格准确率 Wilson CI **下界 > 4×chance**（= 0.0625）的最大 `K`。
   报出全部 `(dh, K*)` 与反向 `dh*(K)` 的实测点；**若某段不可分辨就写「不可分辨」**，
   **不得**由少数点拟合出一条线并外推。
5. **毁源对照臂（强制）**：每个格子都跑 `corrupt_span`（毁掉全部 key/value 对）。
   任何格子若毁源后仍显著高于 chance ⇒ **该格判读作废**。
6. **活性量（强制）**：每 200 步落盘 `state_fro` / `β_mean` / `write_rel` / `read_share`
   与读出增益；训练结束再跑 `carry_probe` 的反事实（关掉通道的偏差）。
7. **必须附**：四臂同方法 loss 斜率、解析/实测内存分开报、每格样本数与二项 CI。

## 5. 臂规格（唯一变量 = carry_state_dim）

| 臂 | 可见性 | 写入 | `dh` | 记忆槽头数 `H_c` | 状态浮点/样本 | 参数量 |
|---|---|---|---|---|---|---|
| `D16` | 窗口 16 + 锚点 0 | delta | 16 | 16 | 4,096 | 13,803,216 |
| `D32` | 同 | delta | 32 | 8 | 8,192 | 13,801,064 |
| `D64` | 同 | delta | 64 | 4 | 16,384 | 13,799,988 |
| `D128` | 同 | delta | 128 | 2 | 32,768 | 13,799,450 |

10,000 步 / bs32 / T256 / lr 1e-3 / AdamW+cosine / seed 0 / `assoc-mqar` K=8,Q=8,gap14。
**评测难度档**：`K=4`(Q=4) / `K=8`(Q=8，训练同档) / `K=16`(Q=8) / `K=32`(Q=8)
（`Q≤K` 是生成器硬约束；`K=4` 只能配 Q≤4）。四臂 × 四档 = **16 个直接测点**，
每格带毁源对照。

## 6. 既有做法（prior art，SCHEMA §18）

> **检索条件声明**：本环境**联网检索不可用**（上一轮 researcher 因 web 工具未注册而失败）。
> 下列全部为 **`记忆·未核`**，**只用于设计方向，不作为判据**（SCHEMA §18/§21(c)）。
> 更完整的盘点见同格子 `experiments/2617-prior-art-memory/PRIOR_ART.md`。

1. **谁做过**：
   - **MQAR / Zoology**（Arora et al.，约 2023，`记忆·未核`）：用**多查询关联回忆**合成任务
     做诊断，发现门控卷积/SSM 在关联回忆上远逊于 softmax 注意力；业界据此刻画
     **「MQAR 准确率 vs 状态大小 / 层数」曲线**——**这正是本实验的曲线原型**。
   - **DeltaNet / Gated DeltaNet / DeltaProduct**（约 2024–2025，`记忆·未核`）：把状态更新
     写成残差 `S←S+β(v−Sk)kᵀ`；报告在 MQAR 一类上优于 Mamba-2。
   - **Mamba-2 / SSM 的 state-size 消融**（`记忆·未核`）：标准做法就是扫 `d_state`。
   - **Compressive Transformer / ICAE / AutoCompressors**（`记忆·未核`）：显式记忆槽 +
     重建/充分性目标；动机是「有界内存下保住远距离」。
2. **为什么这么设计**：单状态模型「无限历史压进固定状态」在需要**精确回忆**的任务上会塌，
   塌的**边界**被业界反复描述为「**状态容量 ÷ 需回忆信息量**」的比值 ⇒ 所以业界把它画成
   **二维**（容量 × 难度）曲线，而不是一个点。
3. **它的实测结果**（**全部 `记忆·未核`，数字未核**）：
   - 一致观测：随需回忆的关联数上升，纯单状态模型准确率明显下降，softmax 注意力几乎不掉；
   - 状态维度增大 ⇒ 可回忆信息量上升（**线性还是次线性我不确定**，需查原文）；
   - **不确定**：现有公开工作多在 100M–1B 级、且不控制**局部先验抗性**。
4. **我们的增量在哪（⇒ §21 判 **(b) 做实验****）**：
   - **尺度**：13M（业界曲线多在更大规模）；
   - **任务**：局部先验抗性受控（窗口内无符号 ⇒ 局部先验退化到 chance，SCHEMA §19）；
   - **单因素受控**：只改状态宽度、**注意力逐位不动**（业界扫 state size 时通常整模型一起变）；
   - **交接**：ticket 35 已把「有没有通道/写规则」钉住，本实验接着钉「容量—难度」的定量关系；
   - **我们不声称**任何「首创」，也不复现业界结论当自己的判据。
5. **若可检索应查的关键词**：`MQAR Zoology associative recall state size curve`、
   `DeltaNet Gated DeltaNet state dimension ablation recall`、`Mamba-2 state size MQAR`、
   `linear attention memory capacity scaling law recall`、`associative memory capacity
   Hopfield delta rule orthogonal keys`。

## 7. 指标表（多维，逐项在 `analysis.md` 分析）

| 维度 | 量 |
|---|---|
| 它解决什么问题 | 把 ticket 35 的「能/不能」变成**定量关系**：给多少状态、能记多少对 |
| 代价 | 状态浮点 = 256·dh（与 T 无关）；参数量 ±0.03%；步时/显存（实测） |
| 有效性 | 16 格准确率（含 Wilson CI）、毁源对照、两条曲线、交换率实测点 |
| 收益与用法 | 「状态预算 → 可回忆对数」的实测换算表（产品侧容量规划的输入） |
| 可控性 | 每 200 步活性量；`carry_probe` 反事实 |
| 副产品 | 状态大小与 T 无关（有限状态的解析边界）；缺省口径逐位还原 v1 |

## 8. 成本（唯一预算数字 + 砍什么）

`gpu_wall_budget: 60 min`（含评测）。

| 项 | 计划 | 实测 |
|---|---|---|
| 单臂步时 | — | **≈82 ms/步**（600 步冒烟；600 步档实测 88/92/100/96 ms ⇒ 含预热的上界） |
| 单臂训练（10k 步） | 10–16 min | 见 `results.json::train_results.*.wall_minutes` |
| 评测电池 | ≈5 min | 见 `results.json::cost.wall_minutes_total_eval` |
| 峰值显存 | ≤5GB net | 见 `results.json::train_results.*.peak_gpu_mem_mb`（实测 ≈2.2–2.4 GB） |

**砍什么**（契约 §成本 + SCHEMA §17.7）：砍评测批数 > 砍 K 档（K=4 与 K=32 是曲线端点，
**不砍**）> **不砍判据线**。若实测单臂 >20 min ⇒ 先跑 `D16/D32/D128` 三臂并如实标
`D64` 顺延（本实验按此授权预留了 3 点退化路径）。

## 9. 数据与脚本清单

- 数据：**不用 enwik8**（契约要求）。训练用 `tools/evalkit/longrange_tasks.py::assoc_recall_multi_batch`
  （只增不改）；评测用同文件的 `assoc_recall_multi_batch` + `corrupt_span`，
  `tools/evalkit/carry_channel.py` 的统计工具。所有 random seed 显式。
- 训练：`python3 train_depth_ar.py <arm> --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense
  --steps 10000 --batch-size 32 --seed 0 --lr 0.001 --train-window 16 --train-anchors 0
  --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --carry-write delta
  --carry-state-dim {16,32,64,128} --out-root train`
- 评测：`python3 run.py`（本目录）。落盘：`train/<arm>/{train_log.jsonl,gate_log.jsonl,results.json,model.pt}`、
  `results.json`、`run_log.jsonl`、`analysis.md`、`MODEL.md`、`mutate.py` + `mutation_report.json`、
  `reconcile_numbers.py` + `reconcile_report.json`、`controls/`、`smoke/`。
- **权重不入库**（`*.pt` 已被 `.gitignore` 覆盖）。

## 10. 失败决策树（测量 → 实现 → 规模 → 设计）

1. **测量**：`dh=32/K=8` 不显著高于 chance ⇒ 先查窗口是否生效、活性量是否为零、
   距离分布是否真的把 key 放在窗外；先修设置再判。
2. **实现**：某臂 `read_share`/`β_mean`/`state_fro` 任一恒为 0 ⇒ 跑变异验证排除死开关。
3. **规模**：容量非零但各臂都 ≈chance ⇒ 解释为「该任务/该尺度上不划算」（SCHEMA §19），
   **不得**表述为「该机制不可能」。
4. **设计**：某格毁源后不掉 ⇒ 读数不是搬来的信息，该格作废。

## 11. scope 声明

本研究**只做单轴单格实验**，基底恒为 stock 对照；**不做组合实验**——组合留白是显式声明，
不是遗漏。`deps` 只记录已知耦合方向（M1 状态载体 ← 写规则 U2；A9 窗口提供「需求」侧）。
组合取舍属具体产品方 scope。

## 12. §21 分诊（本格）

| grid_id | 判定 | 一句话理由 | 若要写进账本的结论 |
|---|---|---|---|
| `M1-carrier-type` | **(b) 做实验** | 业界有 MQAR×state-size 曲线，但尺度更大、不控局部先验、且不把状态宽度与注意力头解耦 | 本实验给**13M + 局部先验抗性 + 单因素**下的容量—难度定量关系（含≥3 交换率实测点） |

## 13. GPU 锁与并发声明（SCHEMA §20/§22）

- 训练/评测全程持 `/tmp/pathlm_gpu.lock`（mkdir 锁 + `trap` 释放；长训练后台 + ≤110s 轮询，
  轮询期间保持持锁）。锁等待/泄漏/显存超预算一律写进最终汇报。
- **共享文件并发**（SCHEMA §22）：本实验的代码改动落在 `pathlm/depth_ar.py`
  （配置字段 + `CarryChannel`/`CarryChannelBlock` 两处）与 `train_depth_ar.py`
  （一个 CLI 参数 + 一处接线）。**这两处已被编排方提交（commit `b44c809`）**，
  与本轮 U2 线（F4 重建损失）的未提交改动**共存**；本实验**不再改动**共享文件
  （新逻辑全部落在本目录），共享文件 sha256 记进 `MODEL.md`。

## 14. 产出与复核

见 §9 落盘清单；本实验完成后按 SCHEMA §14 另派 fresh-context 只读复核者（契约 V）。
