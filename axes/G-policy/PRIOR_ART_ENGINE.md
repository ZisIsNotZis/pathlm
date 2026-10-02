# PRIOR_ART_ENGINE — 推理引擎与调度情报（服务 G 轴）

> 轴：**G 策略与调度** ｜ 服务格子：`G1-selector`、`G2-currency`、`G5-parallel-lines`
> 角色：纯知识线（无 GPU，不取锁，不跑训练/评测，不改代码/测试）
> 依据：`SCHEMA.md` §18（既有做法节）、§19（局部先验抗性）、§21（三选一分诊）
> 本文件**只写情报与分诊，不是实验证据**；与 `axes/G-policy/PRIOR_ART.md` 互补——
> 那份覆盖 G1–G5 的**机制级**既有做法，本文覆盖**推理引擎与调度**的工程形态。

---

## 0. 检索条件声明（**必读，决定本文证据等级**）

**本环境联网检索不可用**（未注册任何 web / `web_search` / `source_check` 工具）。
因此本文**无法在线核证任何外部数字**，严格遵守 `SCHEMA.md` §18/§21：

- 凡来自记忆的论文/系统，一律标 **`记忆·未核`**，**只作设计方向，不得作判据**。
- 凡来自本仓库的数，标 **`本地实测`** 并给文件路径。
- 凡记不清的细节/数字，一律**降级为定性表述**，并标 `不确定`，写「要查什么才能确定」。
- **本文不含任何无法回溯的精确数字**；记忆中不确定的数字宁可省略或只给量级，不编造。

**给下一次可检索 run 的关键词清单**：见每块末尾的「关键词」行，以及 §6 汇总。

术语就地解释：
`draft`（草稿：用便宜方式先猜若干个 token）、`verify`（验证：用完整模型一次前向核对草稿）、
`accept rate`（接受率：草稿被完整模型采纳的比例）、`KV cache`（键值缓存，解码时复用的注意力状态）、
`launch-bound`（受核函数启动开销支配，宽度便宜）、`compute-bound`（受实际算力支配，宽度线性计价）、
`tree attention`（用一个注意力掩码一次性验证树状多条候选）、
`rollback`（回滚：多条候选被否后丢弃其 KV，恢复到分叉点）。

---

## 1. 投机解码的工程形态（草稿从哪来 / 怎么批量验证 / 加速比与质量代价 / 何时收益消失）

> 整块来源类型：**外部系统，`记忆·未核`**（除非另标 `本地实测`）。不确定处已降级。

### 1.1 各家草稿来源（谁做过 / 草稿从哪来）

| 系统 | 草稿从哪来（`记忆·未核`） | 为什么这么设计 |
|---|---|---|
| **Medusa**（约 2023–2024） | 在目标模型**顶部并联多个解码头**，各头独立预测不同未来位置的 token；可组成候选序列 | 不训练独立 draft 模型，只加轻量头，复用主干 |
| **Medusa-2 / Hydra** | Medusa 思路的改进：头带自回归依赖；Hydra 让头**条件于前一个头** | 修正「多头独立预测」与真实自回归分布的失配 |
| **EAGLE-1** | **特征级 draft**：在目标模型**最后一层隐状态**上接一个轻量自回归头，在**特征空间**外推再映射到 token | 特征比 token 更连续、更易外推，接受率更高 |
| **EAGLE-2** | 同上，但草稿**树/深度动态化**：按不确定性扩张或收缩候选树 | 固定草稿长度在不同上下文下浪费或不足 |
| **tree attention**（验证机制，非独立系统） | 不是草稿来源；是**批量验证**手段（见 1.2） | 一次前向验证多条候选路径 |
| **n-gram / prompt-lookup draft**（Prompt Lookup Decoding / REST 一类，`记忆·未核`） | **从上下文里检索 n-gram** 匹配，或从数据表检索续写 | **零额外参数、零训练**；对「有可复制片段」的任务近乎免费 |
| **self-speculative / LayerSkip**（Meta，约 2024） | **同一个模型**：用**早退**（前若干层）产生草稿，再用**余下层**验证；训练期加早退/层跳过监督 | 不需要第二个模型；草稿与验证器同分布、共享 KV |
| **Lookahead decoding / Jacobi 类**（`记忆·未核`） | 用固定/学习的 guess 并行迭代求解 | 不依赖 draft 模型，靠并行猜测迭代收敛 |

- `不确定`：EAGLE-2 的树是否严格「动态宽度」、Medusa-2/Hydra 的命名与年份边界。
  **要查**：EAGLE/EAGLE-2、Medusa 原始论文与官方 repo 的 draft 生成伪代码。
- **对我们的定位**：我们的 `n_mtp` 头本质是 **Medusa 式多头草稿**（`本地实测`，见 §1.4），
  `LayerSkip` 式早退草稿与我们的「逐 token 变深度」设想最接近。

**关键词**：`Medusa multiple decoding heads`, `EAGLE feature-level draft`, `EAGLE-2 dynamic draft tree`,
`prompt lookup decoding`, `REST retrieval-based speculative decoding`, `LayerSkip self-speculative early exit`,
`lookahead decoding Jacobi`, `Hydra decoding heads`。

### 1.2 验证怎么批量（tree attention / KV 一致性）

- **一次前向验证多条候选**：把同一分叉点的多条候选**拼成一个 batch 或一条树状序列**，
  用 **tree attention 掩码**让每个位置的注意力只看到「它的祖先路径」，于是**一次前向**给出
  所有候选节点的 logits；再按接受规则取最长被接受前缀。来源类型：**外部机制，`记忆·未核`**。
- **KV 一致性**：被接受的前缀其 KV 保留；**被拒绝的后缀其 KV 直接丢弃**（无需重算前缀，
  因为前缀已接受）。分页 KV（PagedAttention 一类，见 §2）让「丢弃」= 释放页，而非拷贝。
  来源类型：**外部机制，`记忆·未核`**。
- **与 greedy 的「逐位相等」**：在确定性硬件/核下，草稿接受后结果应与贪心逐位相同；
  但**核级差异会翻转近平局**——这是 `本地实测`（见 §1.4 第 4 条），不是纯理论。
- `不确定`：tree attention 的具体掩码构造、是否需自定义 CUDA kernel。**要查**：
  Medusa/EAGLE 的 attention mask 构造与官方实现。

**关键词**：`tree attention speculative verification`, `speculative decoding attention mask`,
`longest accepted prefix`, `KV cache rollback speculative`, `paged KV speculative decoding`。

### 1.3 公开加速比与质量代价（**全部 `记忆·未核`，仅方向，不引数字**）

- **加速比依赖「接受率 × 草稿成本 × 硬件/批大小」**：文献普遍报告在**单流、小批、
  访存受限**场景下有明显加速；加速比**随 batch 增大而萎缩**，在算力受限的批量解码里可能归零。
  确切数字未核——`不确定`各家在何种硬件/批大小下报的数。
- **质量代价（关键）**：投机解码设计上**理论上无损**（输出分布/贪心路径应与目标模型一致，
  接受规则保证这点）——因此「质量代价」通常**应≈0**，而非 bpc 意义上的税。
  `不确定`：是否有工作在**数值非确定性**下量化过「接受导致的输出分歧」。
- **草稿越自回归越准但越贵**：EAGLE 用特征外推、Medusa 用并联头，都是在「草稿质量 vs 草稿成本」
  上的不同取舍。`不确定`：各家的接受率/成本曲线。
- **对训练式草稿的额外约束**：草稿头/早退出口需要**专门训练或微调**，这是我们的
  `n_mtp` 已走过的路。

**关键词**：`speculative decoding speedup batch size`, `acceptance rate draft quality tradeoff`,
`speculative decoding lossless`, `speculative decoding compute bound`, `Spec-Bench`。

### 1.4 收益何时消失：compute-bound vs launch-bound 判据（**本地实测 + 外部原理**）

**外部原理（`记忆·未核`）**：解码通常是**访存受限**（memory/launch-bound）——每个 token 要读回
全部权重和 KV，而算力闲置；此时「一次读权重、验证 k 个 token」几乎免费，投机有利。
当 **batch 足够大、算力被打满（compute-bound）** 时，验证 k 个 token 的成本≈k 倍，
投机不再有便宜的宽度，收益消失甚至为负。判据即「**当前解码是访存受限还是算力受限**」。

**本地实测（同判据的独立复现）** `本地实测`：
来源 `.scratch/20-spec-scale/evidence/SUMMARY.md`、`spec_speed.json`。
- 13M（launch-bound）：宽度-3/宽度-1 前向成本 **1.28×**，k=2 稳态墙钟加速有正收益。
- 100M（compute-bound）：宽度-3/宽度-1 前向成本 **3.4–5.3×**（宽度线性计价）；
  k=2 墙钟加速收缩到 **1.15–1.29×**，**k=1 出现净亏损（0.80–0.96×，负）**。
- 结论：**接受率不随规模提升（级联稳定），墙钟收益随规模消失**；forwards 口径收益仍在
  （≈2.4 tok/forward）。
- 第 4 条（口径陷阱）：100M 起，batched SDPA 与逐位置 SDPA 的**核差异**在 top-2 gap≈0.008 的
  近平局处**翻转 argmax**（约 1 次/400 token）——「spec==greedy 逐位相等」是 13M 规模性质。

**与 §1.1–1.3 的关系**：外部原理与本地实测**同一判据**（compute vs launch-bound）；
我们是把该判据从「接受率叙事」剥离出来、按**硬件计价**重述（`本地实测`）。

---

## 2. 多线 / 树状调度：KV 一致性与回滚，以及「变深度/变异草稿」的对应实现

> 整块来源类型：**外部系统，`记忆·未核`**（除另标 `本地实测`）。

### 2.1 并行/投机系统如何管理 KV 一致性与回滚（`记忆·未核`）

| 系统 | KV 管理方式（`记忆·未核`） | 回滚/一致性做法 |
|---|---|---|
| **vLLM**（PagedAttention） | KV 切成**固定大小的块（page）**，逻辑连续、物理不连续 | 被拒草稿的 KV **释放对应 page** 即可，无需拷贝；前缀命中可直接复用 |
| **SGLang**（RadixAttention） | 用**前缀树（radix tree）**缓存共享前缀的 KV | 多请求/多候选共享同一前缀 KV；分叉处各自挂子树 |
| **tree-based 投机**（Medusa/EAGLE 类） | 候选树在**一次前向**内批量验证 | 采用**最长接受前缀**；被拒分支的 KV 丢弃；下一轮从接受点续 |
| **beam search 输出多线** | 每步保留 top-b 假设，各自带 KV | 需**分叉/回滚 KV**：不同假设在不同位置，缓存需按假设复制/恢复 |
| **continuous batching** | 不同请求在**跨步**层面动态拼批 | 每请求独立 KV，步进时增删请求；与投机叠加是工程难点 |

- 关键共识（`记忆·未核`）：**回滚不是「撤销计算」，而是「释放/忽略无效 KV 页」**；
  投机系统之所以高效，正因前缀被接受、只有后缀被丢——**回滚代价与拒绝长度成正比，
  与已接受前缀长度无关**。
- `不确定`：SGLang/vLLM 对**多线（beam）** 的 KV 回滚在最新版本里的具体实现与上限。
  **要查**：vLLM/SGLang 文档与论文、`RadixAttention` 原始描述。

**关键词**：`PagedAttention block KV rollback`, `RadixAttention prefix tree KV`,
`vLLM speculative decoding`, `SGLang speculative`, `beam search KV cache fork`,
`continuous batching KV management`。

### 2.2 「逐 token 变深度 + 变异草稿」在既有引擎里有没有对应实现（`记忆·未核`）

- **逐 token 变深度（early-exit / dynamic depth serving）**：有对应研究线——
  **LayerSkip / CALM / early-exit**（见 `axes/G-policy/PRIOR_ART.md` 块①）主张在推理时对
  「容易 token」早退、「难 token」走满深度。这**正是我们 G1 的「逐 token 变深度」设想**。
  `不确定`：这些系统是否在**生产引擎**（vLLM/SGLang/TensorRT-LLM）里有开箱即用的调度支持，
  还是停留在研究原型。
- **变异草稿（stochastic / diverse draft）**：**直接对应物很少**。多数投机系统追求**单一高接受率
  草稿**（确定性/贪心对齐），「**故意变异以产生多条多样路径**」与投机解码的目标**相反**
  （变异会降低接受率）。最接近的是「**多草稿 / 多 draft 模型**」与「多样本 + verifier 重排」
  （self-consistency / best-of-n，见 `axes/G-policy/PRIOR_ART.md` 块⑤），
  但那走的是**采样多条再重排**，不是「共享 trunk 的变异草稿树」。
- **判定**：**「逐 token 变深度 + 树状验证」组合**业界有拼图（变深度 + tree attention），
  但**「以去相关为核心的多线变异草稿」在既有引擎里没有现成实现** ⇒ 对我们的 G5 是**潜在真空白**
  （且真空白的原因见 §5：变异与投机的接受率目标相冲突）。

**关键词**：`early exit serving dynamic depth`, `layer skip inference scheduling`,
`dynamic depth transformer inference`, `diverse draft speculative decoding`,
`multi-draft speculative decoding`, `stochastic speculative tree`。

---

## 3. 调度器的经济学：有没有「预算约束下的最大化」的形式化

> 整块来源类型：**外部方法学，`记忆·未核`**。

| 既有做法 | 用什么「货币」 | 怎么处理「延迟 / 分位数」 | 与我们 G1/G2 的关系 |
|---|---|---|---|
| **ACT / PonderNet**（Graves 2016 / Banino 2021，`记忆·未核`） | **计算步数**（ponder / 期望步数）；PonderNet 用期望计算量 + 方差惩罚 | 只对**期望**计算量建模，**不建模延迟分位数** | G1「逐 token 变深度」的祖先；G2 的「forwards」货币 |
| **Selective prediction / rejection**（Geifman & El-Yaniv，`记忆·未核`） | **覆盖率（coverage）× 风险（risk）** | 给「覆盖率-风险」曲线，**不直接是延迟分位** | G1「输入无关阈值」族；G4 校准的下游 |
| **自适应投机预算 / 最优 γ**（adaptive speculation，`记忆·未核`） | **草稿 token 数 / 期望接受长度** | 目标多为**期望吞吐/延迟**，少数关注尾部 | G2「forwards vs 墙钟」；直接对标我们 Slider 的宽度旋钮 |
| **Test-time compute scaling / best-of-n 预算**（`记忆·未核`，见 R5） | **采样数 n / FLOPs** | 报「同 FLOPs 质量」，延迟常被忽略 | G5 多线的直接竞争物 |
| **服务调度（延迟 SLO / tail latency）**（vLLM 连续批、调度器一类，`记忆·未核`） | **墙钟、分位数（p50/p99）** | **以分位数为目标**（背压、抢占、分块预填充） | G2「延迟分位」货币的工程出处 |

- **有没有把「算力分配」形式化为「预算下最大化」的公开工作？** 有**部分对应**：
  - ACT/PonderNet 是「**在期望计算预算下最大化性能**」（学习式）；
  - selective prediction 是「**在覆盖率约束下最小化风险**」；
  - adaptive speculation / Spec-Bench 一类是「**在草稿预算下最大化吞吐**」。
  - **但用「逐 token 边际收益 + 分位数延迟」双目标求解分配**的**统一形式化**，
    我没找到直接对应的公开工作（`不确定`，`记忆·未核`）。最接近的拼图是
    「**服务调度器（分位数）**」+「**学习式计算分配（期望预算）**」两条**分立的**文献。
- **判定**：**「预算下最大化」作为一个抽象框架，业界各分量都有占位** ⇒ G1 的**离线求解器**
  部分应改 `reference` 口径；**用「延迟分位数」当硬约束、逐 token 分配计算**的**在线求解器**
  是可能的空白（需检索确认，见 §6）。
- `不确定`：PonderNet 是否直接最小化方差而非分位数；adaptive speculation 是否有开源实现。
  **要查**：PonderNet 原文、`Spec-Bench`、vLLM/SGLang 的延迟 SLO 调度文档。

**关键词**：`PonderNet expected computation`, `adaptive computation time Graves`,
`selective prediction coverage risk`, `adaptive speculative decoding gamma`,
`speculative decoding budget optimization`, `tail latency LLM serving scheduler`,
`test-time compute optimal allocation`。

---

## 4. 对 G1 / G2 / G5 的**直接含义**（§21 三选一分诊）

> 记法：(a) 业界已研究透 ⇒ 直接写结论、**不跑实验**；
> (b) 业界没有 / 结论不一致 / 我们设定实质不同 ⇒ **值得做实验**；
> (c) 查不到（含检索不可用）⇒ 只记方向，**不得**据此判 (a)。

```
G1-selector  | (a) | 投机解码的「草稿来源 + 树状验证」工程形态成熟（Medusa/EAGLE/LayerSkip/n-gram 各占位） | 结论：作为**工程管线**，草稿生成与批量验证已被业界做透；我们不重复实现，直接引用。来源类型：外部系统（记忆·未核）+ 本地 MTP 复核。
G1-selector  | (a) | 「逐 token 变深度」有 LayerSkip/CALM/ACT 等成熟占位（见 PRIOR_ART.md 块①） | 结论：**离线求解器 + 变深度**应改 reference 口径；本地 Slider 作复核。
G1-selector  | (b) | 「用延迟分位数当硬约束、逐 token 在线分配计算」的**统一求解器**未见直接前例（记忆·未核，未核证） | 值得做：这是 G1「在线多线调度器」的候选增量，但**先解决 §5 的去相关障碍**。
G2-currency  | (a) | 「投机/效率结论必须在显存带宽受限 vs 算力受限两种计价下双报，否则相反」已是共识 | 结论：**compute-bound vs launch-bound 判据**是通用纪律；本地 100M 数据作同判据复现。来源类型：外部原理（记忆·未核）+ 本地实测。
G2-currency  | (a) | 服务调度以「延迟分位数（p50/p99）」为目标货币有成熟工程出处（vLLM/SGLang 一类） | 结论：G2 的「延迟分位」货币**不是我们的新概念**，业界已用；应改 reference 口径。
G2-currency  | (b) | 「同一机制在 forwards/FLOPs/墙钟/分位**四种货币下同报**并解释分歧」未见单一公开工作做全 | 保留本地口径：双（多）货币并报是本地已证的强纪律（本地实测）。
G5-parallel-lines | (a) | 「多线需去相关、共享主干成员高度相关增益可忽略」是深集成/自一致的既有共识 | 结论：我们 G5 的**主结论应改 reference 口径**；本地数字作 13M 复核（见 PRIOR_ART.md 块⑤）。
G5-parallel-lines | (b) | 「**以去相关为目标的多线变异草稿树**」在既有引擎里**没有现成实现**（记忆·未核；且与投机接受率目标冲突） | 值得做：这是 G5 的真空白候选，也正是「在线多线调度器」的核心；但**第一障碍见 §5**。
```

**分诊要点**：
- **投机解码的工程形态（§1）整体是 (a)**——草稿来源、树状验证、KV 回滚、compute/launch 判据，
  业界均已做透 ⇒ **我们不重复造轮子，直接引用其结论**（`SCHEMA §21`）。
- **G1/G2 的「通用机制」多为 (a)**：变深度、预算最大化、分位数货币都有占位 ⇒ 改 `reference` 口径。
- **唯一可能的真空白集中在 G5 / G1「在线多线调度」两个交叠点**：即
  **「以去相关为核心的、逐 token 分配计算的多线变异草稿调度」**——(b) 候选，
  但须先在 §5 的障碍上取得证据才值得投入。

---

## 5. 单独一节：若这些工程结论成立，我们「多线并行调度」设想的**第一障碍**

**结论：第一障碍不是引擎复杂度，而是「去相关从哪来」——多线必须先真的不相关，
调度器才有东西可调度。** 这一点外部工程无法替我们解决（引擎只保证**高效验证**多线，
不保证**多线彼此不同**），且被本地实测**直接测到**。

### 为什么引擎层面「无碍」、障碍在别处

1. **引擎已替我们解决了「怎么同时跑多线并回滚」**：tree attention 一次前向批量验证多条候选，
   PagedAttention / RadixAttention 让被拒分支**只释放 KV 页**、前缀零重算（§2，`记忆·未核`）。
   ⇒ **如果多线真的多样，引擎开销是可控的**（回滚代价 ∝ 拒绝长度，∝ 已接受前缀）。
2. **但引擎不制造多样性**：投机系统恰恰**追求单一高接受率草稿**，与「变异出多条不同路径」
   目标**相反**（§2.2）。⇒ 「多线」的**多**必须来自模型/训练，而不是调度器。

### 本地已测的硬事实（`本地实测`，直接支撑）

- **共享 trunk 的 voter 太相关**：集成增益仅 **+0.0008 ~ +0.0015**（可忽略）。
  来源 `.scratch/08-tts-probe/evidence/SUMMARY.md`、`axes/G-policy/G5-parallel-lines/README.md`。
- **确定性路径 checkpoint 的 K-path 集成恰好得 0**：X1/X3 的多条「路径」逐位相同，
  K=1..8 完全持平——**beam search 产出的多线正落在这个死区**（无变异 ⇒ 无增益）。
  来源 `.scratch/08-tts-probe/evidence/SUMMARY.md`。
- **去相关必须练进去，且代价为主**：DIVL1（capped −JS + float32 + pre-step grad-norm gate）
  同时改善 baseline（−0.008）与集成斜率（−0.062 > −0.047），但**绝对数仍高于 stock**
  （多样性压力税约 +0.023）；13M 下**税仍主导**。
  来源 `.scratch/09-m3-interactions/evidence/SUMMARY.md`、`G5-parallel-lines/README.md`。
- **深度的「免费多线」也不成立**：深度集成（T2）比单线更差——早层不是「弱但可用的投票者」，
  而是**严格更差的估计器**；置信头只排序**深度**、不排序**逐 token 可靠性**。
  来源 `.scratch/08-tts-probe/evidence/SUMMARY.md`。

### 推论（我们的画像）

> **多线并行调度的第一障碍 = 去相关来源。** 引擎工程（tree attention + 分页 KV 回滚）是
> **已解决的、可借用的**部分；**未解决的是「让多条线真的不同，且不为此付超过集成收益的税」**。
> 本地 13M 证据：从 trunk 分叉的 voter 天然相关（增益可忽略）；确定性多线（beam）落在零增益死区；
> 去相关需**显式训练**（DIVL1 有效但税主导）。⇒ **在没有可训练的去相关机制前，
> 「多线并行调度器」是空中楼阁**：调度器再高效，也只是高效地跑一堆等价的线。

**因此对下一批实验的排序建议**（详见 §6）：**先证去相关来源可规模化（G5），再谈调度器（G1）**。

---

## 6. 给下一批实验的具体建议（服务 G 轴）

> 均基于 (b) 空白判定；每条注明「为什么不是已有工程的重复」与「先决障碍」。

1. **G5 优先：去相关的规模验证（最高优先）**
   - 命题：DIVL1 式去相关训练在 **100M**（而非 13M）能否让**集成增益 > 多样性税**？
     13M 下税 +0.023 主导，`不确定` 税是否随容量收缩。
   - 为什么不是重复：本地已有 13M 机理证据，但**规模是唯一未测变量**（`.scratch/09` 自述）。
   - 先决障碍：需训练多样性 ⇒ 取 GPU 锁（本知识线不执行）。
   - 产出判据：集成斜率（K8−K1）与绝对 bpc 的**双货币**并报（呼应 G2）。
2. **G5 次优先：把「去相关」对准引擎的真实验证单位**
   - 命题：把多样性奖励从「输出分布 JS」改成「**tree attention 下接受前缀的分歧度**」，
     是否更直接地转化为投机场景的收益？
   - 为什么不是重复：既有引擎只做**单草稿**（§2.2），「为多线变异优化草稿」是空白。
   - 注意：与投机接受率目标冲突 ⇒ 契约须显式声明这是「**放弃部分接受率换多样性**」的取舍。
3. **G1/G2（多在 §4 判 (a)，仅作收口）**
   - G1：把「阈值/门控/学习路由/变深度/离线求解器」统一**改 `reference` 口径**，本地 Slider 作复核；
     仅保留「在线多线调度器」为 (b)，且**依赖 G5 结论**。
   - G2：把「compute-bound vs launch-bound 双报 + 延迟分位数货币」写成**纪律条目**
     （业界已用），本地 100M spec 数据作同判据复现；不再单独实验。
4. **明确不做（避免误区）**
   - **不重复实现** Medusa/EAGLE/tree attention/paged-KV 回滚——这些是 (a)，直接引用（§21）。
   - **不把 beam search 当多线**——本地已证确定性多线落在零增益死区（§5）。

**核证优先级**（下一次可检索时）：① `adaptive speculative decoding` 是否有开源实现（决定 §3 的 (b) 强度）；
② `early exit serving` 是否在生产引擎落地（决定 §2.2 的 (a)/(b)）；③ `diverse/multi-draft speculative`
是否有公开工作（决定 G5 空白是否成立）。

---

## 7. 来源清单

**本地实测（本仓库，可直接核对）**
- `.scratch/20-spec-scale/evidence/SUMMARY.md`、`spec_speed.json` — compute/launch-bound 判据、
  接受级联、100M 墙钟收缩、核差异翻转 argmax。
- `.scratch/08-tts-probe/evidence/SUMMARY.md` — 共享 trunk 相关性、确定性 K-path 集成得 0、深集成失败。
- `.scratch/09-m3-interactions/evidence/SUMMARY.md` — DIVL1 去相关（税 +0.023、集成斜率 −0.062）。
- `axes/G-policy/G5-parallel-lines/README.md`、`G1-selector/README.md`、`G2-currency/README.md` — 格子结论与数字。

**外部（`记忆·未核`，只作设计方向，不引精确数字）**
- 投机解码：Medusa / Medusa-2 / Hydra / EAGLE-1 / EAGLE-2 / LayerSkip / prompt lookup (REST) /
  lookahead-Jacobi；tree attention；PagedAttention（vLLM）/ RadixAttention（SGLang）。
- 调度经济学：ACT（Graves 2016）/ PonderNet / selective prediction（Geifman & El-Yaniv）/
  adaptive speculative decoding / test-time compute scaling / 服务延迟 SLO 调度。
- 参考既有对账：`axes/G-policy/PRIOR_ART.md`（G1–G5 机制级既有做法）。

**不确定 / 未找到**
- 各家加速比的**精确数字**与适用 batch/硬件（一律未核）。
- EAGLE-2 动态树的严格构造、Medusa-2/Hydra 命名边界。
- 「以去相关为核心的多线变异草稿调度」是否有公开工作（G5 空白成立性，待检索）。
- 「延迟分位数硬约束 + 逐 token 分配」的统一求解器是否已有公开形式化（§3，待检索）。
- 本环境**联网不可用**，以上外部条目**均未在线核证**。
