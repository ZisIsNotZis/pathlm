# 2617 · 合成长程任务电池（局部先验无法补全的长程天平原语）· 13M · dense

> 契约来源：`.scratch/34-brainstorm/issues/01-brainstorm.md` §「契约 A」+ 开头统一纪律段。
> **副本标注**：§4 判据线 (a)(b)(c) 与 §2「四个任务 / 三条基线」定义**逐字**抄录；
> 其余小节为**摘要**（并注明摘要处）。判据线一律未改。
> 入口：`run.py`；仪器：`tools/evalkit/longrange_tasks.py`；单测：`tests/test_longrange_tasks.py`；
> 原始结果：`results.json`；逐任务日志：`run_log.jsonl`；冒烟：`smoke/`；
> 对照臂说明：`controls/README.md`；权重清单：`MODEL.md`；判读：`analysis.md`。

## 1. 命题

**在「局部先验无法补全」的合成长程任务上，我们 13M 主干到底有没有长程能力？**

enwik8 的局部统计极强（SCHEMA §19）：13M 主干「不需要」真正的长程记忆也能靠语言
先验补强。所以「窗口外的信息没被搬到近处」这个既有读数，可能是**模型不需要**，
不一定是**机制搬不动**。本实验造一件**去掉局部先验混杂**的天平：四个合成任务的
答案**不可由局部先验猜出**——只有把远处的信息记住并在提问处取出来才答得对。
它们是下一批「训练实验」的预登记仪器；本实验只测**零样本基线**（不训练、不调参）。

## 2. 单因素声明

唯一被测量的东西 = **模型在合成长程任务上的 top-1 准确率**。本实验**不改权重、
不训练、不新增参数**（`results.json::intervention.params_added = 0 /
weights_changed = false / training_steps = 0`），基底 = 主流标准 decoder-only
transformer（`DepthARModel(dense=True)`，12 层 × 256 宽 × 8 头，tied 读出），
checkpoint / 数据流 / 评测窗口全部逐字段一致。无「必要性例外」。

**四个任务与三条基线（逐字抄录契约「交付四个任务」段 + 「三条基线」段）**：

> 交付四个任务（每个都要：可调距离/长度、答案**不可由局部先验猜出**、随机种子可控）：
> 1. **关联回忆**（associative recall）：K 对 (key, value)，随机 key/value 取自词表子集，末尾问 key 求 value；
> 2. **Induction**：给出随机序列前缀，后面重复其中一段，要求续出下一项；
> 3. **复制**（copy）：随机片段出现在远处，要求复制；
> 4. **变距离回忆**：单个 (key,value) 对，距离 d ∈ {16,32,64,128,192,255}，问 value。
>
> 每个任务必须给出：**随机基线（chance）**、**局部先验基线（用最近 k 个 token 的 n-gram/最近邻预测）**、
> 以及**距离-准确率曲线**的定义。

实现口径（摘要，落地细节，不改判据）：

- **token 约定**：符号（key/value/片段）取 `0..63` 的随机 byte id，只当随机符号用；
  间隙 `filler_token=64` 把序列填满；提问标记 cue = 主干 mask 槽 `205`（真实文本
  不出现，与本仓库既有 needle 构造同款）。三者不重叠 ⇒ 无「答案在 filler 里又出现
  一次」的混杂。
- **序列长度恒 = 256**（主干 `seq_len`）；查询行 = 该任务指定的行，读
  `logits[query_row]` 的 argmax 比对答案（top-1）。
- **局部先验窗口 k = 8**（`config.json::tokens.local_k`）。必须**严格小于**最小距离
  （16）；实测 k=16 时变距离回忆 d=16 的 value 恰落在窗口边缘、局部先验被抬到 0.5
  ——所以「k < 距离」是任务构造的必要条件，单测 `test_local_prior_window_must_be_below_min_distance` 钉住。
- **chance 两个口径**（必须分开报，SCHEMA §12 口径对账）：`uniform_answer_set =
  1/64 = 0.015625`（本电池 headline）与 `uniform_model_vocab = 1/206 = 0.004854`。
- 距离的定义：关联回忆 = `(T-1) - 2·query_index`（被问那一对离提问处多远）；
  induction = `(T-1) - (a+copy_len-1)`（源片段末端到提问行）；复制 =
  `(T-copy_len-1) - src_pos`（源片段到 cue 的间隔）；变距离回忆 = `(T-1) - src_pos`。

## 既有做法（prior art，SCHEMA §18）

> **来源标注纪律**：本节的结论**全部是「记忆（未核）」**——执行者（worker）本批
> **无 web 检索工具**，没有核到链接。它们**只用于设计方向，不得当判据**（SCHEMA §18）。
> 链接与逐条核实由契约 B（`2617-prior-art-memory`）的情报负责；若契约 B 核出与本节
> 冲突的事实，以契约 B 为准。

| # | 谁做过（记忆·未核） | 为什么这么设计（记忆·未核） | 实测结果（记忆·未核） | 我们的增量 |
|---|---|---|---|---|
| 1 | **MQAR / associative recall 合成基准**：Zoology 套件（Arora 等 2023）、Based（Arora 等 2024） | 需要一个「不靠语言先验、只靠把 (key,value) 记住并取回」的任务，才能公平比较高效注意力/SSM 的**召回**能力 | 注意力强的模型召回高；纯 SSM 无注意力时在合成关联回忆上明显弱，混合少量注意力层即可补上 | 我们**零样本**测一个 13M **字节级** LM（不训练）；并把「局部先验基线 + 距离曲线 + 源对照臂」做成可失败仪器 |
| 2 | **Induction heads**：Olsson 等 2022（Anthropic） | 解释「上下文里重复的片段如何被续写」这一 in-context 复制机制 | induction head 随规模/训练涌现，是 in-context learning 的主要机制之一 | 我们在**单窗口 256、距离到 224** 上量零样本 induction，并加「毁掉源片段」对照 |
| 3 | **copy / selective copying 任务**：经典任务，被 Mamba（Gu & Dao 2023）等用作「选择性记忆」判据 | 复制要求「记住远处片段再原样吐出」，是长程记忆的干净探针 | Mamba 声称用选择性状态能复制（在其设定下）；纯 LTI SSM 不能 | 我们量零样本 copy 的**距离-准确率曲线**（32→223），并加源对照臂 |
| 4 | **needle-in-a-haystack / 变距离检索**：RULER（Hsieh 等 2024）、LongBench 等 | 评估长上下文检索：把针埋在远处、问针 | 多数模型随距离/上下文长度退化 | 我们把「单 (key,value) 对 + 受控距离」压进**单窗口 256**，配局部先验基线；并与 `retention.recent_keep_mask` 复用（依赖曲线） |
| 5 | **n-gram / 最近邻局部基线**：标准做法 | 需要一个「不用长程信息时最强能猜到多少」的地板 | 在随机符号任务上，任何局部统计都退化为均匀猜测 | 我们把局部先验实现为「窗口 k 内最长后缀最近邻 + 词频回退」，并用单测证明它是**真预测器**（窗口内能答对），再证明它在四任务上退化 |

**要查什么才能确定**（留给契约 B 的检索词）：`MQAR associative recall synthetic benchmark`、
`Zoology recall suite`、`induction heads in-context learning`、`Mamba selective copying`、
`RULER long context benchmark`、`linear attention associative recall state size`。

## 3. 指标表（逐项分析）

| 指标 | 解决什么问题 | 代价 | 有效性（vs 同尺寸/同标的对照） | 收益与用法 | 可控性 | 副产品 |
|---|---|---|---|---|---|---|
| 四任务 top-1 准确率 | 「不靠局部先验，模型的真实长程能力」 | 纯推理 ~12 s / 390 MiB | vs chance（1/64）与局部先验（≈chance） | 下一批训练实验的预登记判据线 | 距离/长度/种子全可控 | 距离-准确率曲线 |
| 局部先验基线 | 证明「不靠长程就答不出」（判据 a） | 无（纯 Python） | 单测证明它在窗口内能答对（真预测器） | 每个新任务上线前必过 | 窗口 k 可调 | — |
| 源对照臂（毁掉源片段） | 证明读数来自源信息，不是位置/局部先验 | 一次额外前向 | copy 0.133→0.012、induction 0.059→0.016（均回 chance） | 支撑「弱但真实的长程能力」类结论 | span 显式 | — |
| 打乱答案（变异验证） | 证明准确率来自上下文 | 一次额外前向 | 0.008→0.016（回 chance） | 仪器可信度 | 种子可控 | — |
| retention 可见性曲线 | 「源落在可见窗外还答不答得出」 | 复用 `recent_keep_mask` | 全档 ≈ chance | 直接对齐既有依赖腿仪器 | keep_d 可调 | — |

## 4. 判据线（**逐字**抄录契约「判据」段，未改一字）

> - **判据（预登记）**：
>   - (a) 四个任务都能跑通并给出 chance 与局部先验基线；局部先验基线的准确率必须**显著低于**
>     理想真值（即证明「不靠长程就答不出」）——若某任务局部先验就能答对，说明任务构造失败，
>     必须重做该任务；
>   - (b) 在既有 13M dense 主干（复用 `2610` 的 `trunk.pt`，断言 sha256）上测出四个任务的
>     **基线准确率**（这是关键数字：**我们的模型到底有没有长程能力**）；
>   - (c) 对「变距离回忆」给出与 `retention.py` 兼容的依赖曲线接口（复用其删除逻辑，不重写）。

**执行时的操作性定义（不改判据，只把阈值落定）**：

- **(a) 的「显著低于真值」**：真值上界 = 1.0（任务在完整上下文下由构造决定可解）；
  判据 = `local_prior.accuracy < max(3·chance, 0.02)`（`check_local_prior_resistance`，
  ceiling = 0.046875），四个任务**全部通过**。
- **(b) 的「基线准确率」**：零样本 top-1，口径 chance = 1/64 = 0.015625
  （另报 1/206 = 0.004854）。**并额外加一条源对照臂**（`corrupt_span` 毁掉复制/induction
  的源片段）：若不加，无法区分「真的在搬运源信息」与「位置/局部先验命中」。
- **(c) 的「兼容」**：`recall_under_visibility` 直接调用
  `retention.recent_keep_mask(T, d)`（只保留最近 d 个位置的可见性），不重写删除逻辑。

## 5. 对照与方差

- **对照臂**（每一个都必须写明：输入是什么分布 / 要预测什么 / 部署中对应什么）：
  1. `chance`：答案集（64 符号）均匀猜测 1/64。输入分布 = 任务分布；要预测什么 =
     无信息的随机猜测；部署对应 = 「没有任何长程能力」的地板。
  2. `local prior`（k=8）：只看最近 8 个 token 的最强预测器，输入分布 = 窗口内可见
     token；要预测什么 = 仅凭局部证据猜答案；部署对应 = 「只用最近上下文」的对手。
  3. `source destroyed`（`corrupt_span`）：同任务，把复制/induction 的**源片段**替换为
     全新随机符号（其余逐位不变）。输入分布 = 同任务但源信息不存在；要预测什么 = 同答案；
     部署对应 = 「源信息不存在时仍能答对」的伪命中检测。
  4. `shuffle_answers`：同任务、同 token，把答案标签整体随机置换。部署对应 = 「准确率
     是否与上下文无关」的变异验证。
- **方差**：本实验是**确定性推理**（同 seed 重跑逐位一致，已验：两次全跑 results.json
  除 `cost` 外完全相同）。准确率的不确定度用二项标准误 `sqrt(p(1-p)/n)` 报；样本量
  n：关联回忆 3584、induction 512、复制 28672、变距离回忆 1152。
- **噪声界**：`Δ < 0.014` 是 bpc 口径的方差界；本实验读数是准确率，不用 bpc 界，
  改用 95% 二项置信区间与「源对照臂是否回 chance」判读（见 `analysis.md`）。

## 6. 成本

```
gpu_wall_budget: 25 min   # 契约上限；本实验纯推理，实测 11.5 s（共享 4090）
```

- 实测：墙钟 **11.5 s**，峰值显存 **390 MiB**，无训练步、无优化器状态、可中断
  （纯前向，随时可停）。
- **冒烟计划**：`python3 run.py --smoke`（1 批 / 缩小档数，实测步时与显存，**不外推时间**）。
- **预算不够时砍什么**（SCHEMA §17.7）：砍档数（`n_pairs` / `src_starts` / `src_positions`
  / `distances` 的枚举）与批量（`n_batches`），**不砍判据线**（(a) 局部先验抗性必须全过、
  (b) 四个基线准确率必须出、(c) retention 兼容接口必须跑通）。
- 契约 §8：本文件只给一个预算数字（25 min）。

## 7. 数据与脚本清单

- **数据**：合成（无外部数据集）。生成器在 `tools/evalkit/longrange_tasks.py`；
  随机源 = `torch.Generator().manual_seed(config.seed)`（seed=0）。
- **入口脚本**：`run.py`（`--smoke` 冒烟 / 缺省全电池）。
- **resolved config**：`config.json`（含 checkpoint 路径 + sha256、模型 config、token
  约定、四任务与曲线参数、seed）。
- **落盘文件**：`results.json`（原始结果）、`run_log.jsonl`（逐任务/逐 K 日志）、
  `smoke/results.json`（冒烟）、`analysis.md`、`controls/README.md`、`MODEL.md`。

## 8. 失败决策树（无效时的处置路径：测量 → 实现 → 规模 → 设计）

1. **测量**：重跑同 seed 是否逐位一致？局部先验窗口 k 是否 < 最小距离？（k≥距离会
   人为抬高基线，见 §2。）
2. **实现**：源对照臂（`corrupt_span`）是否回 chance？打乱答案（`shuffle_answers`）
   是否回 chance？仪器自检（释放版 vs 显式因果掩码）是否逐位相等？
3. **规模**：若是「答不出来」，先排除**尺度不足**（13M、256 窗口、字节级、零样本）；
   这是 SCHEMA §19 要求的默认解释——**「该任务/尺度上不划算」，不是「机制不可能」**。
4. **设计**：只有前三条都排除后，才允许怀疑任务构造本身（关联是否真的无歧义、
   局部先验是否真的看不到答案），并**重做该任务**（契约 (a)：构造失败必须重做）。

## 9. 产出

- `results.json`：`instrument`（自检）/ `intervention`（活性）/ `tasks`（四任务 ×
  chance/局部先验/模型 × 距离曲线）/ `controls`（源对照臂）/ `retention_curve` /
  `mutation` / `criteria`（(a)(b)(c)）/ `code`（四项 sha256）/ `cost`。
- `analysis.md`：人话摘要 + 逐格表格 + 逐条判读（失败也写）。
- `MODEL.md`：**无新权重**；复用 2610 trunk.pt（哈希见下）。
- `controls/README.md`：对照臂与变异验证说明。

## scope 声明

> 本研究**只做单轴单格实验**，基底恒为 stock 对照；**不做组合实验**——
> 组合留白是显式声明，不是遗漏。`deps` 字段只记录已知耦合方向。
> 组合取舍属具体产品方 scope。
