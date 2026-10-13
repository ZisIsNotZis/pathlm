# 2623 · A9 合成长程判决：训练 window vs full（ticket 40 Y3）· 13M · dense

> 契约来源：`.scratch/40-next-wave/issues/01-next-wave.md` §「Y3」+ 开头统一纪律段。
> **副本标注**：§3 干预、§4 判据线（①–④）与 §2 判定取向段**逐字**抄录；其余小节为
> **摘要**（摘要处已注明）。判据线一律未改。
> 入口：`run.py`（评测电池，纯推理）；训练入口：仓库根 `train_depth_ar.py`（只加一个
> 既有开关 `--train-window`，**不改引擎**）；语义层：`lr_eval.py`（只增）；
> 仪器：`tools/evalkit/longrange_tasks.py`（任务与判分，只增不改）；
> 单测：`tests/test_a9_window_longrange.py`；变异验证：`mutate.py` + `mutation_report.json`；
> 原始结果：`results.json`；权重清单：`MODEL.md`；三道对账：`reconcile_numbers.py`。

## 1. 命题

**「窗口能否在真正的长程任务上不亏」——在合成关联回忆上训练两臂直接判。**

ticket 39-X3 的 §19 合成腿是**空结果**：四臂都训练在 enwik8 上，`var_recall` 是
**分布外**任务，13M 模型不会做关联回忆 ⇒ 四臂全在 chance 附近，**不能区分窗口与 full**
（X3 analysis §7 的亲口诊断）。本实验按 X3 §10 给出的下一步最小实验，**在合成关联回忆上
训练**，只在**训练/评测可见掩码**一个维度上分两臂：`full`（无掩码）vs `win16`（只看最近
16 个位置 + anchors 0）。这样源 key 训练时就落在窗外，模型若不靠显式搬运通道，就**没有**
把远处信息带到提问处的路径——判决因此有判别力。

## 2. 判定取向（**逐字**抄录契约 Y3 的动机段）

> X3 的 §19 合成腿是**空结果**（未训练的权重在分布外任务上不会回忆 ⇒ 四臂全 chance）。
> 必须**训练**才能判「窗口能否在真正的长程任务上不亏」。

「真正的长程任务」的含义（摘要）：答案**不可由局部先验猜出**（合成关联回忆；局部先验
基线见 §12 抗性检查），必须把远处的 (key,value) 记住并在提问处取出来才答得对（SCHEMA §19）。

## 3. 干预（**逐字**抄录契约 Y3 的干预段）

> - 目录：`axes/A-mixing/A9-old-info-retention/experiments/2623-window-on-longrange-13m/`
> - 干预（唯一变量 = 训练/评测掩码）：`full` vs `窗口 16 + anchors 0`，**在合成关联回忆上训练**（13M/12k）。

**实现（摘要）**：掩码 `keep[i,j] = (j ≤ i) ∧ ((i−j) < 16)`（anchors=0），对角恒开；
唯一实现点 = `pathlm.depth_ar.window_anchor_keep_mask`。**训练侧**
`DepthARModel._window_mask` 与**评测侧** `lr_eval.native_keep_mask` 调**同一函数**
（契约：训练掩码必须等于评测掩码，否则量的是别的东西）。`train_window=0`（`full` 臂）=
关闭 ⇒ 走 stock 因果 SDPA 释放版路径，逐位还原 v1（既有单测钉住）。

## 4. 判据线（**逐字**抄录契约 Y3 的判据段，未改一字）

> - 判据（预登记）：① 两臂的合成召回（主腿，准确率口径 + chance 基线 + 二项 CI）；
>   ② 毁源对照臂；③ 距离腿（key 落在窗外 ⇒ 不搬就答不出）；④ clean bpc 作辅助。
>   **注意**：这与 ticket 32-O（enwik8 上训窗口）不同——那是强局部先验任务。

### 执行时的操作性定义（**不改判据**，只把符号与「显著」落定）

- **主腿任务变体（详见 `analysis.md`）**：契约写「合成召回」未指定
  变体。本实验主腿用 **`assoc_recall_multi_batch`（多查询关联回忆，与训练**同一布局**、
  held-out 新键值）**——这是 fair 的口径。**次级的单查询变体** `assoc_recall_batch`
  与 `var_recall` 都把查询键放在**最后一个位置**，而多查询训练布局里**末位恒为 filler**
  ⇒ 对 `full` 臂是**位置外推**（它会把末位预测成 filler）：实测 `full` 在单查询布局
  0.0、在多查询布局 0.35 ⇒ 单查询/变距离读数**作次级如实报告**并附此证据，
  **不作主判据**（属 methodology §14「先排除测量问题」）。主/次两个读数都落 `results.json`。
- **口径**：每臂用**自己的训练掩码**评测（native；`full` = 释放版，`win16` = 窗口 16）。
  训练/评测同掩码是契约的「唯一变量」的前提；**不报**「win16 臂在全可见下」的口径，
  那会把源信息白白送给一个从未学过用它做查找的模型、且引入第二个变量。
- **chance**：答案集 `{0..63}` 上均匀猜 = `1/64 = 0.015625`（`longrange_tasks` 的
  headline 口径）；另在 results 里落 `chance_dict`（含原始词表口径 `1/206`）。
  二项 CI = Wilson 95%（`carry_channel.wilson_ci`）。
- **判据 ① 的「显著高于 chance」** = `accuracy CI 下界 > chance`；**「与 chance 不可
  区分」** = `accuracy CI 下界 ≤ chance`。
- **判据 ① 的「窗口伤长程」** = `full` 与 `win16` 的两比例差（pooled z）
  `delta_pp ≥ 10pp` 且两臂 Wilson CI **不重叠**（`two_proportion`）。
- **判据 ②（毁源对照）**：把源片段 `[0,2K)` 换成同分布新随机符号（`corrupt_span`）。
  「主臂读数来自源信息」= 毁源后准确率 CI **上界 < 主臂准确率**（且毁源后落回 chance）。
- **判据 ③（距离腿）**：主读数是**多查询主腿的 by-distance 曲线**（源全部落在窗外）；
  **次读数** `var_recall` 六档（`d ≥ 16` 的源 key 在 `win16` 下都在窗外），因位置外推
  **只作参考**。「不搬就答不出」= 窗外档与 chance 不可区分。逐档标 `source_inside_window`。
- **判据 ④（clean bpc 辅助）**：本实验权重**训练在合成任务上**，enwik8 对它**分布外**、
  不可判读（同 X3 §19 的诊断口径）⇒ 「clean bpc」在本实验**显式定义**为
  **任务分布上的干净下一 token CE**（bit/token，越低越好），在各臂 native 掩码下：
  全位置口径 + 判决行口径（判决行 CE 才是决定准确率的那个量，SCHEMA §17.15）。
  此为**辅助腿、无阈值门禁**。**基准唯一且显式**（§17.14）：只报这一种 clean bpc 定义。

## 5. 单因素声明与配置等值性（§9）

唯一因素 = **训练/评测可见掩码**（`full`：无掩码 vs `win16`：`window=16, anchors=0`）。
两臂同 runner（`train_depth_ar.train`）、同架构（`d256/L12/H8/mlp6`、`dense`、
`sequential`）、同数据（`assoc-mqar`）、同优化器/调度/步数（12k）/批大小（bs32）/seed（0）；
**参数量逐位相等**（掩码零参数）；`run.py` 运行期硬断言 `params_equal`。

**实现层等值性证据**：`tests/test_depth_ar.py::test_train_window_default_off_matches_v1_bitwise`
（关闭开关 ⇒ 与无掩码裸堆叠逐位一致，单测钉住）；本实验新增单测再钉住
「window=16@T=256 **非退化**」（`visible_frac < 1`）与「源 key 落在窗外」。

## 6. 既有做法（prior art，SCHEMA §18）

**检索方式**：本子代理**无 web 工具**，以下均标 `记忆·未核`；**不作判据**，只用于设计方向
（§18 允许「未找到直接前例」时必须写明检索词/来源，故此处如实标注来源等级）。

1. **谁做过**：
   - **Longformer**（Beltagy 等，2020）：**滑动窗口注意力** + 少量 global token；动机 =
   把注意力从 O(T²) 降到 O(T·w)。窗口外信息只能经 global token 或层叠传递。
   - **BigBird**（Zaheer 等，2020）：窗口 + 随机 + global 的稀疏注意力，理论上是
   序列函数的通用逼近器。
   - **StreamingLLM**（Xiao 等，2023）：**attention sink + 滑动窗口** 流式推理，声称
   窗口外旧 token 的 KV 可丢而不掉质量（但依赖 sink 缓存）。
   - **Memorizing Transformers**（Wu 等，2022）/ **Landmark Attention**（Mohtashami &
   Jaggi，2023）：给窗口/局部注意力**外挂检索或地标**来搬远处信息。
   - **MQAR / induction heads**：`Zoology`（Arora 等，2023）、`Repeat After Me`
    （Jelassi 等，2024）、induction heads（Olsson 等，2022）——合成关联回忆是判「长程
   检索是否学到」的标准天平。
2. **为什么这么设计**：窗口注意力是为了**省算力/省显存**（局部性先验：多数任务只需近处）；
   其代价是**信息搬运能力**——这正好落在本格 A9「旧信息处理策略」的核心。
3. **它的实测结果**：上述工作普遍报告「窗口省算力；**纯窗口**在需要远距离检索的任务上
   退化，需 global/sink/检索补足」——**同任务、同尺度、单因素受控的本地实测我们没有**；
   标注 `记忆·未核`，不作判据。
4. **我们的增量在哪（§21 分诊 = (b) 做实验）**：**13M 小尺度、合成弱局部先验任务、
   单因素（只改掩码）、同 token 预算**的受控判决——把「窗口是否伤长程」从业界「需要补足」
   的定性说法，落成**本尺度本任务**的可复核数字（且给出毁源/距离两类因果护栏）。

## 7. 能力面分析（SCHEMA §22.8，派发前必做）

对容器要求的每个 flag **逐条**核对引擎校验（最低成本检查 `grep -n "ValueError\|互斥"
pathlm/depth_ar.py`）：本实验只用 **`--train-window` / `--train-anchors`（既有开关）**，
不叠加 `attn_sparse_k` / `attn_linear` / `carry_*` / `n_kv_heads` / `topology`。
逐个两两组合核对：

- `train_window=16` 要求 `dense=True` 且 `topology="sequential"` —— 本实验满足（不冲突）；
- `train_window` 与拓扑 / `attn_sparse_k` / `attn_linear` 互斥 —— 本实验都没开；
- `train_anchors=0` 不与 `train_window>0` 冲突（只有 `anchors>0 且 window==0` 才报错）；
- `full` 臂 = `train_window=0`（缺省）⇒ 不触发任何窗口校验；
- 块类：窗口是**正交装饰层**（`attn_mask`），两臂块类都是 stock `Block`（`run.py` 自检
  打印块类，`params` 相等硬断言）。

**结论**：契约要求的组合在引擎里**存在**，无需降级/绕开（对照 W3 的 `GQA×carry×window`
事故——本实验不碰那三条）。

## 8. 指标表（多维，逐项）

| 维度 | 指标 | 与谁比 | 结果（见 `analysis.md`） |
|---|---|---|---|
| 解决什么问题 | 「窗口能否在真正的长程任务上不亏」的直接判决 | full vs win16 | 判据 ①③ |
| 代价 | 训练期窗口对**任务分布** clean bit/token 的税 | full 臂 | 判据 ④ |
| 有效性（引用） | 合成关联回忆准确率（K=4/8/16）+ chance + Wilson CI | full vs win16 | 判据 ① |
| 有效性（因果） | 毁源对照臂（源片段换掉后落回 chance） | 各臂自身 | 判据 ② |
| 有效性（距离） | 变距离回忆六档曲线 + 源是否在窗内 | full vs win16 | 判据 ③ |
| 可控性 | 同 seed 纯推理复现 + 训练 loss 斜率（同方法） | 阈值/方差界 | §analysis 稳定性 |
| 干预是否活着 | window 掩码活性量（可见因果对占比，每 200 步落盘）+ 功能探针 | 0 参数增量 | `results.json::selfchecks` |

## 9. 对照与方差

- **对照臂 = `full`（同配方、无掩码）**，反事实清晰：同一模型、同一数据，唯一差别是
  「训练/评测时能不能看见 16 步以外的 token」。
- **同一模型内的对照**：距离腿的参照物是**同一臂 native 掩码**下的窗内/窗外档；
  毁源对照的参照物是**同臂主腿**读数。
- **复现臂（`win16_rep`）**：为满足「n=1 负结果必须复现后入账」（methodology §2），
  `win16` 臂同 seed 整条重跑一次；它**不参与判据口径**（主臂仍是 `train/win16/model.pt`），
  只作训练侧噪声与结论复现的证据（预算不足时砍本臂，见 §10）。
- 方差界：Δ < 0.014 不可排序（methodology §2，用于 bpc）；准确率用 Wilson CI 口径。
  训练侧固定 seed 在本引擎上**不可逐位复现**（项目级已知事实，见 2615/stability 记录）。

## 10. 成本（唯一数字）

- 契约预算：**≈40 min GPU**。
- **实测步时（先测后定，SCHEMA §17.8）**：`smoke/steptime_probe.sh`（每臂 400 步，
  `smoke/probe_*/`）—— full **75.2** ms/step、win16 **73.7** ms/step（含启动开销；
  日志 200→399 步的**稳态**两者均 ≈**72 ms/step**，共享卡争用所致）。
- **训练实测**：full **13.01 min** / win16 **13.75 min** / win16_rep **14.18 min**
  （两臂逐字段同配方，只差掩码；复现臂可砍）。训练总 **≈41 min**（3 臂）/ ≈27 min（2 臂）。
- **评测实测**：全电池 **1.04 min**（纯推理；`results.json::cost.wall_minutes_total_eval`）。
- 峰值显存：full **1630.4 MB** / win16 **2015.3 MB**（**< 5GB net**，不触礼貌线；
  逐臂峰值见 `results.json::train_results`）。
- 可中断性：训练中可杀（重跑 ≈14 min）；评测纯推理、可重跑（≈1 min）。
- **预算不够时砍什么**：砍 **`win16_rep` 复现臂**（已按 `--arms` 分次补跑）→ 砍距离腿
  batch 数 → 砍 K sweep 的 K=4/16 档；**不砍判据线**。

## 11. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | **合成**多查询关联回忆（`assoc_recall_multi_batch`，训练）/ `assoc_recall_batch`、`var_recall_batch`（评测）；**未用 enwik8** |
| `full` 臂权重 | `train/full/model.pt`，sha256 见 `config.json`（运行时断言；不入库） |
| `win16` 臂权重 | `train/win16/model.pt`，sha256 见 `config.json` |
| `win16_rep` 权重 | `train/win16_rep/model.pt`（复现臂；可砍） |
| 训练入口 | `./train_all.sh`（取锁、按臂释放）；步时探针 `./smoke/steptime_probe.sh` |
| 评测入口 | `python3 run.py`（`--smoke` 缩批冒烟） |
| 对账 / 变异 | `python3 reconcile_numbers.py` / `python3 mutate.py` |
| 落盘 | `results.json` / `run_log.jsonl` / `mutation_report.json` / `reconcile_report.json` / `train/<arm>/{model.pt,results.json,train_log.jsonl}` |

## 12. 局部先验抗性（SCHEMA §19 前提）

`assoc_recall` / `var_recall` 的**局部先验基线**（只看最近 k=8 个 token 的最强预测器）
必须**显著低于**真值（`check_local_prior_resistance`）：`run.py` 落 `local_prior` 读数
（K_main=8 批）。若局部先验被抬到 chance 之上 ⇒ **构造失败**，先修构造再判读。

## 13. 失败决策树

| 症状 | 诊断 | 处置 |
|---|---|---|
| `full` 也≈chance | 任务在 13M/12k 上**学不动**（信号饥饿 / 难度） | 如实报**空结果**（同 X3 §19），写清「训练步数/规模不足」，**不**表述为「窗口不伤长程」 |
| `full` 高、`win16`≈chance | **窗口伤长程**（无通道就搬不动） | 判据 ①③ 成立；与 2618「给通道后 delta 回升到 0.095/6×chance（K=8）」并列，说明**修法是通道不是掩码** |
| 距离腿两臂都≈chance | 分布外（var_recall 与训练布局不同） | 与 X3 §19 同处理：报**测不出**，以关联回忆主腿为准 |
| 毁源后仍高 | 任务/仪器有混杂（答案可从局部抄到） | 先修构造；`corrupt_span` 必须真改源片段（单测钉住） |
| 曲线非单调、超噪声容差 | 测量问题 | 先修仪器/重跑；并在 analysis.md 登记该测量问题 |
| 两臂参数量不等 | 加载错误 | 排查 config；`params_equal` 自检会标红 |

**原则**：先排除测量问题与实现问题，再排除规模问题，最后才归因设计问题（methodology §14）。

## 14. 产出

`README.md`（本文件）/ `run.py` / `lr_eval.py` / `config.json` / `train_all.sh` /
`fill_config_hashes.py` / `results.json` / `analysis.md`（人话摘要在前 + 多维指标表 +
逐字段对照 + 三道对账 + 失败决策树 + 口径 + 实测墙钟与次数）/ `MODEL.md` /
`smoke/steptime_probe.sh` / `reconcile_numbers.py` / `mutate.py` / `mutation_report.json`；
权重 `train/*/model.pt` 不入库（哈希见 `config.json` / `MODEL.md`）。
