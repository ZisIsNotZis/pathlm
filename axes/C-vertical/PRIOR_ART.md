# PRIOR_ART — C 轴（垂直机制 / 深度）既有做法与分诊

> 轴：**C · 垂直机制（深度）** ｜ 格子：`C1`–`C6` ｜ 产出：本文件（知识波，K1 线）
> 角色：researcher（只读 + 本文件）｜ 本波**不跑训练/评测、不取 GPU 锁**
> 依据：`SCHEMA.md` §18（既有做法节）、§19（局部先验抗性）、§21（不重复造轮子）
> 性质：**情报，不是实验证据**。无 `results.json`；本地数字一律给文件出处，外部一律 `记忆·未核`。

---

## 0. 检索条件声明（**必读，决定本文的证据等级**）

**本环境联网检索不可用**（`web_search` / `fetch_content` / `source_check` 等 web 工具
未注册；上一轮 researcher 子代理即因此失败——见本批 `.scratch/36-parallel/issues/01-parallel.md` 波一）。
因此本文**无法在线核证**任何外部数字。

据此降级并遵守：

- 凡来自我记忆的论文/系统，一律标 **`记忆·未核`**，且**只允许用作「设计方向」**（SCHEMA §18）。
- 凡来自本仓库的数，标 **`本地实测`** 并给文件路径。
- 凡不确定的标 **`不确定`** 并写「要查什么才能确定」。
- **不含任何我无法回溯的精确外部数字**：记忆中的数字一律降级为定性表述。宁可少写，不编造。
- 术语就地解释：`bpc`（每字符比特数，越低越好）、`税`（相对普通 decoder-only transformer 多付出的质量代价）、
  `tf@k`（用前 k 层当草稿、被全深度接受的比例，越高说明浅层越能当「草稿/出口」）、
  `early exit`（早退：浅层算够就用浅层结果）、`adaptive depth`（逐 token 变深度）、
  `per-token`（逐 token，而非逐序列）、`carry gate`（携带门：`h ← h + g⊙Refine(h)` 的逐通道门）、
  `deep supervision`（深度监督：中间层的辅助损失）、`LayerSkip`（早退训练+推理）。

**给下一次可检索 run 的关键词清单**见 §3。

---

## 0.5 人话摘要（先看这段）

这条轴问的是「同一位置的算力怎么沿**层**组织」。本次盘点（只查记忆、不能联网）得到三句话：

1. **「逐 token 变深度 / 早退 / 自适应算力」这个大方向业界早就有了**（Universal Transformer、
   CALM、LayerSkip、Mixture-of-Depths 等）⇒ 机制层面**不是我们的发明**，相关子结论应改成
   **reference 口径**（「业界结论、非本地实测」）。**权重共享**（Universal Transformer/ALBERT）同理，业界已透。
2. **我们真正独有的，是两个规模上量出来的「代价」**：自适应深度在本形态下有 **+0.44 bpc 的形态税**
   （100M，96% 干净归属），以及「**出口资产只能由深度监督造出来**」的定量对照（`tf@4` 0.12 vs 0.97）。
   业界多报「自适应深度近无损」，**我们的税与他们叙事有张力** ⇒ 属 (b) 类，保留本地账本。
3. **层间拓扑（C3）有明确冲突**：业界 `DenseFormer` 报小幅增益，我们同规模测到**中性（−0.0008）**；
   并行子层（PaLM/GPT-J 采用）我们测到**明确有害（+0.0458）**。冲突 ⇒ (b)，且**不得**外推为「业内方法错」。
   另有情报：跨层 KV 共享有占位（CLA/YOCO/MLKV），但**层特异分享**查不到直接前例——**要做先核证**。

---

## 1. 块①：既有做法条目（SCHEMA §18 格式，逐格）

### C1 每个 token 跑几层（judged）

- **谁做过**：**自适应算力**一整条线——`Adaptive Computation Time`（Graves 2016）、
  `Universal Transformer`（Dehghani 2018，循环深度 + ACT）、`CALM`（Schuster 2022，置信早退）、
  `LayerSkip`（Elhoushi 2024，早退训练 + 自推测解码）、`Mixture-of-Depths`（Raposo 2024，逐 token 路由深度）、
  `CoLT5`/`SkipDecode` 等（均 `记忆·未核`）。
- **为什么**：不是所有 token 都需要满深度算力；把省下的 FLOPs 换吞吐或换更长上下文。
- **结果（`记忆·未核`）**：自适应深度在**匹配质量**下可省算力；**出口质量依赖训练方式**（是否加早退损失）。
  **`本地实测`**：逐 token 变深度**成立**且是本项目最厚资产——`tf@4 = 0.9726`、`tf@6 = 0.9946`、
  `d6−d12 = 0.0003`（半栈 ≈ 全栈）、`exit-4` 墙钟 `3.1×`；代价是**同引擎形态税 `+0.4362 @100M`**
  （形态 96% / 引擎 4% / 配方 ≈0）。出处 `axes/C-vertical/C1-layers-per-token/README.md`、
  `.scratch/23-depth-ar`、`.scratch/24-depth-ar-100m`、`.scratch/25-tax-attribution`。
- **我们的增量**：**机制（逐 token 变深度）业界已有（(a) 部分，改 reference）**；但
  「**同引擎形态税 +0.44**」与「**d6−d12=0.0003 的出口资产**」是本地定量增量（(b)）。
  ⇒ 分诊 C1 = (b)：机制子结论可写 reference，本地税与出口数保留 judged。

### C2 选哪些层与执行顺序（judged）

- **谁做过**：`LayerDrop`（Fan 2019，训练期随机丢层）、`Reducing Transformer Depth`（Sajjad 2020）、
  结构化深度剪枝（Zhang 2022）、循环/重入深度（`Universal Transformer`、`Deep Equilibrium` 类）；
  层序重排/打乱我记忆中有零星研究，但**指认不了确切论文**（`不确定`，要查）（均 `记忆·未核`）。
- **为什么**：想在不改参数量的前提下，用「跳过/复用/重排」换自适应与省钱。
- **结果（`记忆·未核`）**：skip / layer-drop 在**训练后**大体可控（破坏连续性但可恢复）；
  层序**重排**的敏感度报道不一。**`本地实测`**：**元素税定律的原始出处**——
  `skip +0.175`、`redo +0.044`、**层序 `shuffle +0.545`（全项目最贵元素）**；shuffle 是**组合毒药**
  （任何含 shuffle 的组合落到 2.2+ bpc）。出处 `axes/C-vertical/C2-layer-selection-order/README.md`、
  `.scratch/06-m2-sweep`、`.scratch/09-m3-interactions`。
- **我们的增量**：**skip/layer-drop (a)**；但「**层序重排最贵（+0.545）**」这条**本地量化的元素税定律**
  业界不报（业界多报「层可剪」，不报「层序是强先验」）⇒ **C2 = (b)**。

### C3 层间拓扑（judged）

- **谁做过**：`并行子层`（attention ∥ MLP）——`GPT-J`、`PaLM` 采用（`记忆·未核`）；
  `DenseFormer`（Pagliardini 2024，对全部历史层输出学习加权和）；
  `Hyper-Connections`（Zhu 2024，多残差流/可学习连接）；跨层跳连（`LayerSkip`）；
  feedback transformer（`记忆·未核`）。
- **为什么**：改层间连接以期改善信息流、吞吐或深度可截断性。
- **结果（`记忆·未核`）**：并行化常因**吞吐**被采用，代价是**一定质量损失**（业界报告为可接受）；
  `DenseFormer`/`Hyper-Connections` 报**同规模小幅增益**。**`本地实测`**：**并行子层明确有害**
  （`+0.0458`，等参数 `dual-LN` 对照复现 `+0.0402`）；**层间学习加权和中性**（`−0.0008`，层权确实活着：
  fro 3.305、浅层被放大 2.36×）。出处
  `axes/C-vertical/C3-inter-layer-topology/experiments/2613-inter-layer-topology-13m/analysis.md @b3c2484`。
- **我们的增量**：**结论不一致**——业界 `DenseFormer` 报增益、我们测中性；⇒ **C3 = (b)**，
  保留 judged，并按 §19 只表述为「13M/12k 上无便宜可捡」。**未测**：跨层注意力、无残差、更大尺寸。

### C4 权重共享（stub）

- **谁做过**：`Universal Transformer`（Dehghani 2018，循环权重）、`ALBERT`（Lan 2019，跨层参数共享）、
  循环深度 / weight-tying 一系（`记忆·未核`）。
- **为什么**：深度方向复用同一组权重，省参数、以循环次数换推理深度。
- **结果（`记忆·未核`）**：共享可省大量参数、匹配质量往往需要**更多步/更难训练**，易不稳。
- **我们的增量**：本格**本地零经验**；与本项目「回环时代的重入（retry）」不同——那是 F 轴迭代，
  权重天然共享；本格要的是**层与层之间**的结构共享。⇒ 分诊 C4 = (a)：业界已透，直接写 reference。

### C5 精化算子（judged）

- **谁做过**：`残差 + MLP`（标准）、`Highway`（Srivastava 2015）、`ReZero`（Bachlechner 2020）、
  门控残差 / `GLU` 门、`Mixture-of-Depths`（MoD）、`AdaLN`（DiT，Peebles 2022）、`FiLM`（均 `记忆·未核`）。
- **为什么**：门控让极深网络可训；条件化（AdaLN）把深度/条件注入每层。
- **结果（`记忆·未核`）**：门控帮助**深网络训练稳定**；未见「携带门在**同空间精化**下给出任意深度可读」的
  直接报道。**`本地实测`**：carry 门是本形态税的主要承载之一，也是「任意深度可读」的结构来源；
  门行为健康（浅层开、深层微开）；@100M 门从 `0.119` **收小**（原假设「渐进开门」未兑现）。出处
  `axes/C-vertical/C5-refinement-operator/README.md`、`.scratch/23-depth-ar`、`.scratch/24-depth-ar-100m`、`.scratch/25-tax-attribution`。
- **我们的增量**：**通用门控 (a)**；但「**carry 门在同空间精化形态下的行为与税**」是本地设计增量（(b)）。
  ⇒ 分诊 C5 = (b)。

### C6 训练期深度分布与监督密度（judged）

- **谁做过**：**深度监督**——`Deeply-Supervised Nets`（Lee 2015）、`GoogLeNet` 辅助分类器；
  NLP 侧早退损失（`LayerSkip` 的 early-exit loss）、`CALM` 的置信损失、随机退出深度（均 `记忆·未核`）。
- **为什么**：让中间深度「可用」（能当草稿/出口），或正则化训练。
- **结果（`记忆·未核`）**：中间层辅助监督会**塑形浅层表征**；`LayerSkip` 报告早退 + 自推测可用。
  **`本地实测`**：出口资产**只能由深度维监督造出来**——纯 dense trunk `tf@4` 仅 `0.120`（vs 形态臂 `0.97`）；
  全深度等权 `+0.067 @13M`；dense 全深度监督**杀 retry 精炼**（「整形与自适应是同一梯度的两面」）；
  剂量响应随规模变化（dense 0.2 在 13M 温和、@100M corrupt 崩溃 `+1.73`）。出处
  `axes/C-vertical/C6-depth-supervision/README.md`、`.scratch/14-allocator`、`.scratch/18-exit-probe`、
  `.scratch/21-exit-at-scale`、`.scratch/24-depth-ar-100m`。
- **我们的增量**：**深度监督概念 (a)**；但「**tf@4 0.12 vs 0.97 的对照**」「**整形 vs 自适应互斥**」
  「**剂量响应随规模翻转**」是本地增量（(b)）。⇒ 分诊 C6 = (b)。

---

## 2. 块②：机器可读分诊表

> 判定 = **(a)** 业界已研究透（不跑实验，写 `status: reference` + 标「业界结论，非本地实测」）/
> **(b)** 需本地实测（业界没有 / 结论不一致 / 我们设定实质不同）/
> **(c)** 查不到（按 `记忆·未核` 记方向，**不得**据此判 (a)）。
> 格式：`grid_id | 判定 | 一句话理由 | 若要写进账本的结论`

| grid_id | 判定 | 一句话理由 | 若要写进账本的结论（标注来源类型） |
|---|---|---|---|
| C1-layers-per-token | **(b)** | 逐 token 变深度/早退/自适应算力业界已有（(a) 部分），但**同引擎形态税 +0.44**与出口资产是本地定量增量。 | 待写入账本 = **本地结论**：「逐 token 变深度成立，出口资产 tf@4 0.9726 / tf@6 0.9946 / d6−d12 0.0003；代价是同引擎形态税 +0.4362 @100M（形态 96% 干净归属）。」来源类型：**本地实测**。机制子结论可另立 reference 条目。 |
| C2-layer-selection-order | **(b)** | skip/layer-drop 业界有（(a) 部分），但「层序重排是最贵元素（+0.545）」的定量元素税定律业界不报。 | 待写入账本 = **本地结论**：「元素税定律：破坏顺序结构最贵（shuffle +0.545 > skip +0.175 > redo +0.044）；层序携带约 0.5 bpc 的先验。」来源类型：**本地实测**。 |
| C3-inter-layer-topology | **(b)** | **与业界不一致**：DenseFormer 报增益、我们测中性；并行子层业界采用、我们测明确有害。 | 待写入账本 = **本地结论**：「13M/12k 上改层间连接无便宜可捡：并行子层 +0.0458（等参数复现 +0.0402），层间学习加权和中性 −0.0008。」来源类型：**本地实测**（§19：不做「机制不可能」的外推）。 |
| C4-weight-sharing | **(a)** | 跨层权重共享（Universal Transformer / ALBERT）业界已系统研究，结论稳定。 | 「深度方向复用权重省参数、以循环次数换推理深度；匹配质量需更多步且更难训练。」来源类型：**业界结论，非本地实测**（`记忆·未核`）。本地：无数据。 |
| C5-refinement-operator | **(b)** | 通用门控/条件化业界已有（(a) 部分），但 carry 门在同空间精化形态下的行为与税是本地设计。 | 待写入账本 = **本地结论**：「携带门是形态税承载之一，也是『任意深度可读』的结构来源；浅层开、深层微开，@100M 门从 0.119 收小（渐进开门假设未兑现）。」来源类型：**本地实测**。通用门控另立 reference 条目。 |
| C6-depth-supervision | **(b)** | 深度监督概念业界有（(a) 部分），但「出口资产只能由深度监督造出」的定量对照与「整形 vs 自适应互斥」是本地增量。 | 待写入账本 = **本地结论**：「出口资产只能由深度维监督造出来：纯 dense trunk tf@4 0.120 vs 形态臂 0.97；且深度整形与自适应精炼是同一梯度的两面（两全被证伪）。」来源类型：**本地实测**。 |
| C-跨层 KV 分配（情报） | **(c)** | 跨层 KV 共享有占位（CLA/YOCO/MLKV 均匀模式），层特异/不等共享/同字节预算最优分配**查不到直接前例**。 | 若可检索应核证；**不得**据此判 (a)。方向记 `记忆·未核`。与本轴 C1/C3 相关（KV 预算是深度的成本面）。 |

**给主 Agent 的直接含义**：C4 可考虑改 `status: reference`（标「业界结论，非本地实测」）；
C1/C2/C3/C5/C6 保持 judged（均 (b)，含「业界已有机制 + 本地独有定量」两层，建议在账本里
把**机制子结论**与**本地数值**分开标注）。

---

## 3. 块③：若可检索应查什么（关键词清单）

| 对应块 | 关键词 |
|---|---|
| C1 变深度/早退 | `adaptive computation time`, `Universal Transformer`, `CALM confident adaptive language modeling`, `LayerSkip self-speculative early exit`, `Mixture-of-Depths token routing`, `SkipDecode`, `CoLT5` |
| C2 层选择/顺序 | `LayerDrop`, `reducing transformer depth`, `structured depth pruning`, `transformer layer permutation sensitivity`, `recurrent depth weight sharing`, `layer reordering ablation` |
| C3 层间拓扑 | `parallel attention MLP GPT-J PaLM`, `DenseFormer dense connections`, `Hyper-Connections residual streams`, `feedback transformer`, `LayerSkip skip connections` |
| C4 权重共享 | `ALBERT cross-layer parameter sharing`, `Universal Transformer recurrent`, `deep equilibrium transformer`, `weight tying layers quality` |
| C5 精化算子 | `highway networks`, `ReZero residual`, `gated residual transformer`, `AdaLN DiT adaptive layernorm`, `FiLM conditioning`, `Mixture-of-Depths gate` |
| C6 深度监督 | `deeply supervised networks`, `GoogLeNet auxiliary classifiers`, `LayerSkip early exit loss`, `intermediate layer supervision transformer`, `depth-wise auxiliary loss` |
| 跨层 KV 分配 | `CLA cross-layer attention`, `YOCO you only cache once`, `MLKV multi-layer KV sharing`, `layer-specific KV heads`, `KV budget allocation across layers` |

**核证优先级（若只有一个时间窗）**：`C1 LayerSkip / CALM 的税`与`C3 DenseFormer 的增益`——
前者决定我们「形态税 +0.44」相对业界叙事是否需要补「规模/任务差异」说明；
后者是我们**与业界冲突**的正面对照，必须核准 `DenseFormer` 的原报告基线。

---

## 4. 来源清单

**来源类型图例**：`官网/论文（含链接）` ｜ `记忆（未核）` ｜ `本地实测`。

**本地实测（可直接当证据；路径均在本仓库）**
- `axes/C-vertical/C1-layers-per-token/README.md` — tf@4 0.9726 / tf@6 0.9946；d6−d12 0.0003；exit-4 3.1×；税 +0.4362 @100M。
- `axes/C-vertical/C2-layer-selection-order/README.md` — skip +0.175 / redo +0.044 / shuffle +0.545。
- `axes/C-vertical/C3-inter-layer-topology/experiments/2613-inter-layer-topology-13m/analysis.md` — parallel +0.0458（等参数 +0.0402）/ dense −0.0008。
- `axes/C-vertical/C5-refinement-operator/README.md` — carry 门行为；@100M 门从 0.119 收小。
- `axes/C-vertical/C6-depth-supervision/README.md` — dense trunk tf@4 0.120 vs 0.97；全深度等权 +0.067；dense 0.2 @100M corrupt +1.73。
- `docs/playbook.md` 定律 6（整形 vs 自适应同一梯度两面）、定律 7（主干被改税就来）。

**记忆（未核，只可用于设计方向，不得当判据）**
- 自适应算力：Graves ACT / Universal Transformer / CALM / LayerSkip / Mixture-of-Depths / CoLT5 / SkipDecode
- 层选择：LayerDrop / Sajjad reducing depth / structured pruning / Deep Equilibrium
- 层间拓扑：GPT-J / PaLM parallel / DenseFormer / Hyper-Connections / feedback transformer
- 权重共享：ALBERT / Universal Transformer
- 精化算子：Highway / ReZero / GLU / AdaLN(DiT) / FiLM / MoD
- 深度监督：Deeply-Supervised Nets / GoogLeNet / LayerSkip early-exit loss
- 跨层 KV：CLA / YOCO / MLKV

**未找到 / 不确定（本 run 无法核）**
- 「层序重排（shuffle）敏感度」的确切公开研究出处（`不确定`，需查）。
- `DenseFormer` / `Hyper-Connections` 报的增益**确切数字与基线强度**（需查原文消融表）。
- `ALBERT` 跨层共享的**精确质量-步数权衡**（需查原文）。
- 「层特异 KV 头数 / 同 KV 字节预算最优分配」是否存在直接前例。

## 5. 我不确定的地方（诚实清单）

1. 本文所有外部内容均为**记忆**，**未在线核**；任何具体外部数字若被引用进结论都属违规。
2. 「DenseFormer 报小幅增益」「LayerSkip 用早退损失」我信心较高但**未核**——它们若为假，
   C3/C6 的 (a)/(b) 边界需重审。
3. 「层序重排研究」我**记不清确切论文**，只能标 `不确定`——不得据此写「业界已透」。
4. 我**无法排除**已存在与 C1「逐 token 变深度」设计高度重合、且报过税的工作。
5. 本文件未做任何训练/评测；无 `results.json`，**不是实验证据，只是情报**。
