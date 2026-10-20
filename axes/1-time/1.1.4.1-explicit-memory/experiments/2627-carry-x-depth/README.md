# 2627 · delta 搬运通道 × dense 全深度监督（13M · 2×2 因子设计 · 合成长程任务）—— 实验契约

> 契约来源：`.scratch/44-wave17/issues/01-wave17.md` 的「**X2**」。
> 本文件是该线的**执行契约**；§4 判据线**逐字**抄录 X2。
> **§4c 记录与契约明文的唯一偏离**（深度因子落到 `--dense-aux` 而非 ragged 随机退出）
> 及其**主 Agent 裁决**——那是本线唯一的契约参数变更。
> 入口：`train_arms.sh`（训练四臂）+ `run.py`（评测电池）+ `smoke_steptime.sh`（先测步时）；
> 语义层 `carry_x_depth_arms.py`（2×2 因子定义）；单测 `tests/test_carry_x_depth_13m.py`；
> 变异 `mutate.py`；对账 `reconcile_numbers.py`；原始结果 `results.json`；
> 判读 `analysis.md`；权重清单 `MODEL.md`；对照说明 `controls/README.md`。

## 1. 命题

**delta 搬运通道（关联回忆 0.0137→0.0950）与深度自适应/早退（tf@4 0.9726、exit-4 3.1× 墙钟）
是两个独立测过的正面资产，但它们从未一起上过同一主干。把它们同时装上：长程召回与出口资产
各付多少、得多少？两者**能不能叠加**？**

一句话可证伪的形式：*把 `carry` 与 `depth` 两个因子排成 2×2 格子，`both` 臂的合成召回
是否等于两个单因子的**可加预测**（`carry + depth − base`）；若显著低于它（负交互项），
则记为「两资产不可叠加」。*

## 2. 单因素声明（2×2 因子设计）

- **基底**：本仓 stock dense decoder-only trunk（`DepthARModel(dense=True, topology="sequential")`：
  tied 读出 + 无 final LayerNorm + 顺序残差），13.5M（d256 × 12 层 × 8 头 × mlp6 × T256）。
- **唯一变量 = 两个因子的开关**（`carry` × `depth`，四臂恰好是 (F,F)(T,F)(F,T)(T,T) 全格子）。
  四臂按**同一份配方**逐字段训练（同 seed=0、同数据生成器、同步数/批大小/学习率/优化器、
  同评测协议），相邻 / 对角臂只差一个因子：

  | 臂 | `carry`（delta 写） | `depth`（dense 全深度监督） | 引擎位点 | 参数量 |
  |---|---|---|---|---|
  | `base`（基准） | 关 | 关 | `Block` | 13,536,768 |
  | `carry` | **开** | 关 | `CarryChannelBlock` | 13,801,064（+264,296） |
  | `depth` | 关 | **开**（`--dense-aux`） | `Block`（**与 base 逐位同架构**） | 13,536,768 |
  | `both`（目标） | **开** | **开** | `CarryChannelBlock`（**与 carry 逐位同架构**） | 13,801,064 |

- **`depth` 是训练信号因子，不是架构变量**：它 = 训练 `forward` 的 `dense_aux=True`
  （把监督从「仅最终深度 1 个出口 CE」扩到「全深度 L+1 个出口 CE、梯度穿 trunk」）。
  因此 `depth`/`both` 的 `DepthARConfig` 与各自不含深度因子的臂**逐字段相同**
  （`model_config_kwargs()` 不含它）；差异只体现在训练日志的出口 CE 条数（13 vs 1）。
- **训练/评测可见性一致**：四臂**全部**用 `window=16 / anchors=0`（训练与评测**同一个**
  `window_anchor_keep_mask`）。窗口是**制造长程需求**的那一步（源 key 段 `[0, 2K)` 被推到
  窗外 ⇒ 不搬就答不出，SCHEMA §19），**四臂同款、不是变量**。
- **无「必要性例外」**：四臂的任务、token 预算、评测批逐字段相同。

## 3. 干预定义（必须写清，因为它是「实现」而非「标签」）

四臂**不改任何引擎代码**（本线不写 `pathlm/`、不写 `train_depth_ar.py`），只是把两个既有
开关组合起来：

- **carry（`--carry-write delta`）**：单共享记忆槽，
  `S_t = S_{t−1}(I − β k̂ k̂ᵀ) + β v k̂ᵀ`；每层读出 `a ← a + γ ⊙ mem_t`，γ **单边零初始化**
  （起步恒等于无通道臂、非鞍点）。参数量 +264,296（`w_k/w_v/w_q/beta/gate_w/carry_scale`）。
- **depth（`--dense-aux`）**：dense 主干上开启**全深度并行监督**——`states[0..L]` 各出一个
  tied 读出 CE（depth 0 = 输入直连读出），总损失按 `(w·L + final_weight)` 归一保持量级
  （`shallow_weight=1.0 / final_weight=1.0` = 等权）。**0 参数**；启用后训练日志
  `depth_ce` 长度从 1 变 13。它产出的出口资产就是判据②要读的 `tf@4`。

## 既有做法（prior art，SCHEMA §18）

> **来源标注纪律**：本环境**联网检索不可用**（未注册 web / 检索工具）。外部条目一律标
> `记忆·未核`，只作设计方向、**不得当判据**；本地条目给文件路径。可检索时应查的关键词见
> 每块末尾。

| # | 谁做过（记忆·未核） | 为什么这么设计 | 实测结果（记忆·未核 / 本地） | 我们的增量 |
|---|---|---|---|---|
| 1 | **显式写入通道 / delta rule（fast weight）**：Schmidhuber 1992；Schlag 等 2021（delta rule）；Yang 等 2024（Gated DeltaNet） | 用固定状态把窗口外信息搬到可达范围 | 本地 2618：无通道 0.0137≈chance、delta 写 0.0950（6.08×） | **本线**在该正面结果上叠加**全深度监督**，问「深度监督会不会破坏搬运」 |
| 2 | **中间层监督 / 早退（layer-wise supervision / early-exit）**：深度监督（Deeply-Supervised Nets 2014）、SDM / 自推测解码（Zhang 等 2024） | 浅层出口可用 ⇒ 用浅层草稿、全栈验证加速解码 | 本地 ticket 24：tf@4 **0.9726**、exit-4 墙钟 **3.1×**（13M 模式复现 0.97–0.98）；ticket 23c H1：dense+aux 在 enwik8 有质量代价（3.6k +0.284 → 12k +0.108 bpc） | **本线**把该出口资产与搬运通道**装在同一主干上**，量其对合成召回的影响（正/负/可加） |
| 3 | **两机制组合的干扰**：一般化「辅助损失是否与记忆机制抢容量」（记忆·未核） | 多任务 / 多监督可能竞争同一表征容量 | 本地无直接前例 | **本线**是首次把 (carry × dense-aux) 排成 2×2 因子格子并给交互项 |

**若可检索应查**：`delta rule fast weight MQAR`、`layer-wise supervision early exit draft acceptance`、
`auxiliary losses interference capacity small transformer`、`carry/recurrent memory + intermediate supervision`。

## 4. 判据线（**逐字**抄录 X2 契约，一字未改）

> ## X2 · 搬运通道 × 深度自适应（≈40 min，13M）
> 目录：`axes/1-time/1.1.4.1-explicit-memory/experiments/2627-carry-x-depth/`
> 动机：两个独立正面资产——**delta 搬运通道**（关联回忆 0.0137→0.0950）与**深度自适应/早退**
> （tf@4 0.9726、exit-4 3.1× 墙钟）——**从未一起测过**；ENG1 已解除组合限制。
> 干预（同 token 预算、逐字段一致）：`dense baseline` / `+carry(delta)` / `+per-token depth(随机退出)`
> / `carry × depth`（四臂阶梯，**每步增量都要报**，§17.17）。
> 判据：① 合成关联回忆召回（**训练分布内档**，§17.18）+ 毁源对照；② 出口资产 tf@4；
> ③ 每步增量；④ 参数量/活性量；⑤ 若 carry×depth 出现负增量（互相干扰）⇒ 明确记为「两资产不可叠加」。

**执行时的操作性定义（不改判据，只把口径落定）**：

- **(1) 的「合成关联回忆召回（训练分布内档）」**：主判据腿 = **训练型多查询 `assoc-mqar`**
  （`assoc_recall_multi_batch`，K=8、Q=8、gap=14；查询-键距离 ≤240，**与训练同分布**，§17.18）。
  另跑单查询 `assoc_recall_batch`（判决行 `T-1`，距离 241–255 = **训练分布外**）作为**外推参照**，
  并跑全因果交叉口径。**毁源对照**：把源段 `[0, 2K)` 换成同分布新随机符号（`corrupt_span`），
  任一臂毁源后 **CI 下界 > 4×chance** ⇒ 该臂读数作废。
- **(2) 的「出口资产 tf@4」**：`tf@4` = depth-4 出口 argmax 与全栈 argmax 的一致率（TF 代理）。
  两个口径并报：**enwik8**（对齐 100M 的 0.9726）与 **synth 训练分布内**（多查询 assoc-mqar）。
- **(3) 的「每步增量」**：§17.17 的**分量分解** = 2×2 因子分解的四个单向增量
  （`carry@depth_off`、`depth@carry_off`、`carry@depth_on`、`depth@carry_on`）+ **交互项**
  + 净和；判据线用 **pp**，方差带 = 1.4pp。质量代价项（clean bpc 与最终深度 CE）单独报。
- **(4) 的「参数量/活性量」**：四臂参数量逐臂实测并声明（carry +264,296 = +1.95%；
  depth +0）——按 §15 说明是否混淆归因。活性量每 200 步：carry（`read_share` 等）与
  depth（出口 CE 条数 13/1 + 逐深度 CE）均写进同一行训练日志；`run.py` 逐臂核对并摘末行。
- **(5) 的「两资产不可叠加」**：交互项 = `(both − carry) − (depth − base)`；若 **< 0 且
  |交互项| > 1.4pp 方差带** ⇒ 明确记「两资产不可叠加（互相干扰）」。

## 4b. 能力面分析（SCHEMA §22.8，派发前逐条做）

契约要求的每个 flag 组合，引擎是否支持？`grep -n "ValueError\|互斥" pathlm/depth_ar.py`
列出的校验逐条核对：

| 组合 | 引擎现状 | 本线 |
|---|---|---|
| `carry_rule` ⊕ `dense=True` | carry **硬要求** `dense=True`（`depth_ar.py:1845-1848`：`写入通道只在 dense=True … 下验证`） | carry/both 臂 |
| `carry_rule` ⊕ `dense=False`（ragged 随机退出） | **硬互斥**（实测 `ValueError`） | 不可达 |
| `exit_depths`（ragged 随机退出）⊕ 计算 | 只在 `dense=False` 分支生效；**dense 分支完全忽略 `exit_depths`** | — |
| `dense=True` ⊕ `dense_aux`（全深度监督） | **允许**（`forward(dense_aux=True)`，`readouts = depth_logits`） | depth/both 臂 |
| `carry_rule` ⊕ `dense_aux` | **允许**（无互斥校验；实测可建可跑） | both 臂 |
| `carry_rule` ⊕ `n_kv_heads<n_heads` / `train_window>0` | ENG1 后才允许（`ComposableBlock`）——**本线不用** | — |

**关键实测（CPU 前向）**：
1. `dense+carry`，`exit_depths=full` 与 `exit_depths=rand` 的 loss **逐位相同**
   （5.5217419 vs 5.5217419，出口 CE 条数 1）⇒ `--exit-mode uniform` 在 dense 上是**死开关**；
2. dense 上 `dense_aux=True` 使出口 CE 条数从 1 变 13（监督真的扩容）；
3. `dense=False` + `carry_rule=delta` ⇒ **`ValueError`**（carry 只支持 dense）；
4. `dense=False` ragged：`exit_depths` 会改变 loss（5.3419 vs 5.3483），即「随机退出」只在
   非 dense 路径成立。

⇒ **契约字面的 `carry × per-token depth(随机退出)` 在引擎里不存在**（carry ⇒ dense ⇒ 忽略
exit_depths；随机退出 ⇒ 非 dense ⇒ 拒绝 carry）。ENG1 解除的是 `GQA × carry × window`，
**不包含这一对**。处置见 §4c。

## 4c. 与契约明文的唯一偏离：深度因子落到 `--dense-aux`（**主 Agent 裁决**）

**根因**（见 §4b）：契约把深度因子写成「per-token depth(随机退出)」，但它与 carry 在引擎里
**硬互斥**——carry 要求 `dense=True`，dense 主干**忽略** `exit_depths`；而 ragged 随机退出要求
`dense=False`，该路径**拒绝** `carry_rule`。

**裁决（选项 A，主 Agent 认可能力面分析）**：深度因子改用 **dense 兼容的 `--dense-aux`**
（dense 主干上的全深度并行监督，L+1 个出口 CE 穿 trunk）。四臂全部 dense + window16 +
同 token 预算 ⇒ 2×2 因子设计可建可跑，且能产出判据②要读的 `tf@4` 出口资产。

**四条记账**：
1. **显式声明未测因子**：本实验测的是「delta 搬运 × **dense 全深度监督**」；**未测**
   「ragged 随机退出 + proj-fill 的冻结/混合 KV 语义」——而那正是产出 100M `tf@4=0.9726`
   的配方（`fill_kv=True`，非 dense）。**禁止**外推到「任意深度自适应 × 搬运」。
2. **质量代价一起报**：dense-aux 在 enwik8 上有已知代价（ticket 23c H1 − denseB：
   3.6k +0.284 → 12k +0.108 bpc）；本实验在自己的 clean bpc 腿上给出代价项，并按 §17.17
   给对基准质量（bpc / 最终深度 CE）与对目标量（合成召回）两列增量。
3. **判据②的 tf@4 口径**：它在 dense-aux 上是「出口头是否可用」的读数；与 100M 的
   ragged+proj-fill 配方**不是同一实现**，比较时只作**同一家族的方向性参照**。
4. **后续项**（主 Agent 另立，不占本波）：`ENG2 · 让 carry 支持非 dense ragged 路径`——
   把 carry 的 dense 硬校验解耦、让写入通道在随机退出路径可用、并在混合填充语义下定义写入
   时机（新块类接线 + 组合可失败单测 + 变异）。

> 本实验**不改引擎绕开互斥**（SCHEMA §22.8 禁止在未获授权时改引擎）。

## 5. 对照与方差

- **基准 = `base` 臂**（两因子都关的 stock dense trunk；本仓「基底恒为 stock 对照」口径）。
  每个因子分解项都相对**自己的对照格子**：`carry@depth_off = carry − base`、
  `depth@carry_off = depth − base`、`carry@depth_on = both − depth`、
  `depth@carry_on = both − carry`、`interaction = (both − carry) − (depth − base)`。
- **方差带 = 0.014 bpc / 1.4pp 召回**（`harness/methodology.md`）；召回判据用 **Wilson 95% CI**
  与 `two_proportion` 的差值 CI。
- **局部先验地板** = `local_prior_accuracy(k=8)`（§19：合成任务上局部先验必须 ≈ chance）；
  **chance** = `1/sym_vocab` = 1/64 = 0.015625。
- **逐臂毁源对照**（§4 (1)）。
- **纯推理、无随机采样**：同 seed 评测应逐位相同。
- **逐字段对照表**（唯一变量 = 两因子开关）：

| 字段 | base | carry | depth | both |
|---|---|---|---|---|
| seed | 0 | 0 | 0 | 0 |
| 训练数据 | assoc-mqar | 同 | 同 | 同 |
| 步数×批×T | 6000×32×256 | 同 | 同 | 同 |
| lr / 优化器 | 1e-3 / AdamW+cosine | 同 | 同 | 同 |
| `train_window/anchors` | 16/0 | 16/0 | 16/0 | 16/0 |
| `carry_rule` | None | **delta** | None | **delta** |
| `dense_aux`（训练 forward） | False | False | **True** | **True** |
| 参数量 | 13,536,768 | 13,801,064 | 13,536,768 | 13,801,064 |

## 6. 成本（§8 的唯一预算数字 + 砍什么）

```
gpu_wall_budget: 40 min   # 含四臂训练 + 评测
```

- **实测步时**（先测后定，§17.8/§20.3；`smoke/steps400/`，400 步稳态）：
  base **40.6** / carry **76.7** / depth **48.1** / both **85.7** ms/step ⇒ 6k×4 推算 ≈25.1 min。
- **实测训练墙钟**（`train_<arm>.log`）：base **5.0** / carry **10.0** / depth **5.7** / both **9.7** min
  = **30.4 min**；加 3×65s 臂间窗口 ≈33.7 min；**评测电池 0.56 min**（纯推理，峰值 **572.4 MB**）。
  四臂训练峰值显存 **2015 / 2283 / 2095 / 2362 MB**（≪5GB net；源 `train/<arm>/results.json`
  的 `peak_gpu_mem_mb`，原写 2282/2361 误引 400 步冒烟值 ⇒ P2 修正见 `analysis.md` 文末）。
- **预算不够时砍什么**（§17.7）：砍**步数（四臂同步砍，不砍单臂）** > 砍评测批数/距离档；
  **不砍判据线**（主判据腿、四个单向增量 + 交互项、毁源对照、参数量/活性量、tf@4 都必须出）。
- **§20.1 锁记账**：`train_arms.sh` **按臂**取/放锁、臂间 65s 窗口（不整程独占）；本波另有
  GPU 线（X1 等）在同一锁上串行，等待时长记入 `train_*.log` 与 `analysis.md`。
- **§20.1 第 2 条并发例外登记**：冒烟（`smoke_steptime.sh` 的 25 步步时测量、`run.py --smoke`）
  为**纯推理/短训、墙钟 <1 min、峰值 <2.5GB**；若满足「`nvidia-smi` 空闲 ≥3GB」则按例外登记。

## 7. 数据与脚本清单

- **训练数据**（§19：**不用 enwik8 当长程主任务**）：
  `tools/evalkit/longrange_tasks.py::assoc_recall_multi_batch`（K=8 对、Q=8 查询、gap=14）。
- **评测**（合成长程任务）：`assoc_recall_multi_batch`（**主判据腿** + 毁源对照）、
  `assoc_recall_batch`（训练分布外外推参照）、`var_recall_batch`（距离腿）。
- **clean bpc 腿**：`data/enwik8_full.npz` 的评测尾段（vocab 206，与模型词表一致）。
- **出口资产**：`pathlm/depth_ar.py::tf_acceptance`（enwik8）+ `run.py::per_depth_agreement`
  （synth 分布内）。
- **入口**：`train_arms.sh`（训练）、`python3 run.py`（评测，`config.json`）、
  `smoke_steptime.sh`（先测步时）。
- **落盘**：`results.json`、`train/<arm>/{train_log.jsonl,gate_log.jsonl,results.json}`、
  `train_<arm>.log`、`analysis.md`、`MODEL.md`、`controls/README.md`、
  `reconcile_numbers.py → reconcile_report.json`、`mutate.py → mutation_report.json`、
  `smoke/`。
- **测试**：`tests/test_carry_x_depth_13m.py`（11 条可失败单测 + 变异验证 `mutate.py`）。

## 8. 失败决策树（测量 → 实现 → 规模 → 设计）

1. **测量**：主判据腿与单查询腿是否分列？chance/局部先验是否同值？毁源是否同一批？
   → 先修 `run.py` 的接线，**不得**动模型。
2. **实现**：carry 臂 `read_share = 0`（通道没被用）或 depth 臂出口 CE 条数仍为 1
   （`--dense-aux` 是死开关）→ 跑变异验证排除死开关；`read_scale` 仍单边零 = **步数**问题
   （6k 未逃逸），如实记录、不外推。
3. **规模**：四臂都 ≈ chance ⇒ 该任务/该尺度上机制本就没起作用（读 2618 的 U0/U2 作前提）；
   **不得**表述为「组合不可能」（SCHEMA §19）。
4. **设计**：某臂毁源后仍高 ⇒ 读数混杂、该臂作废；`both` 的负交互项若只来自 dense-aux 的
   质量代价（而非搬运被破坏）⇒ 用分量分解（召回 vs bpc）分开读，并在结论里写清是设计还是口径。

## 9. 产出与复核

- `analysis.md` 给**人话摘要 + 七问**（§23）+ 2×2 因子表格 + 判读；`MODEL.md` 给权重哈希与
  代码指纹；`controls/README.md` 给对照臂与变异说明。
- 本实验完成后按 SCHEMA §14（含 §14b 顺序屏障）另派 fresh-context **只读复核者**独立检查。

## scope 声明

本研究**只做单轴单格实验**，基底恒为 stock 对照；本实验是 M1 载体类型格内的**2×2 因子设计**
（delta 搬运通道 × dense 全深度监督），唯一变量 = 两个因子的开关。
`attn_sparse_k` / `attn_linear` 与上述因子的交互仍受引擎互斥限制，是显式留白、不是遗漏；
「ragged 随机退出 + proj-fill」因子（100M 出口资产的真配方）与本实验**不可组合**，是显式
未测因子（§4c，ENG2 跟进）。`deps` 只记录已知耦合方向（M1 载体类型 ← 2618 搬运通道线 /
ticket 24 出口资产线）。
