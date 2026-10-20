# PRIOR_ART — 混合层比例 / GQA·MLA / KV 预算 Pareto 的既有做法（契约 C）

> 轴：**R 参照线（必须打败的基线）** ｜ 格子：`R4-window-kv-quant-retrieval` ｜ 实验：`2617-prior-art-hybrid-kv`
> 角色：researcher（只读 + 本文件）｜ 本批**不跑训练/长评测**
> 依据：`harness/SCHEMA.md` §18（既有做法节）与 §19（局部先验抗性）
> 服务对象：契约 C 的三问（混合层比例 / GQA·MLA 未发掘空间 / KV 预算参照线）

---

## 0. 检索条件声明（**必读，决定本文的证据等级**）

**本环境未注册任何 web / 检索工具**（`web_search`、`fetch_content`、`get_search_content`、
`source_check` 均不可调用；上一轮 researcher 子代理即因此失败——见
`.scratch/34-brainstorm/issues/01-brainstorm.md` 执行记录）。因此本文**无法在线核证**
任何外部数字。

按契约 C 授权条款，本文**降级为「记忆（未核）+ 本地实测」**，并遵守：

- 凡来自记忆的论文/系统，一律标 **`记忆·未核`**，且**只允许用作「设计方向」，
  不得作为判据**（SCHEMA §18）。
- 凡来自本仓库的数，标 **`本地实测`** 并给文件路径。
- 凡我不确定是否存在、或不确定细节的，标 **`不确定`** 并写「要查什么才能确定」。
- **本文不含任何我无法回溯的精确数字**：凡记忆中的数字一律降级为**定性表述**或标注
  「数字未核，需查原文」。宁可少写，不编造。
- `bpc`（每字符比特数，越低越好）、`税`（相对普通 transformer 的质量代价）、
  `KV`（注意力缓存：模型回头查的键值表）、`GQA`（若干查询头共用一份键值）、
  `MQA`（所有头共用一份键值）、`MLA`（把键值压成低秩潜向量再展开）——首次出现即在此解释。

**给下一次可检索 run 的关键词清单**见 §5。

---

## 1. 块①：纯线性注意力为何被判「不可行」，以及混合层比例曲线

### 谁做过 / 为什么这么设计（`记忆·未核`）

- **线性注意力的原始动机是「把 sequence 长度的平方开销压成线性」**（`Katharopoulos et al.,
  2020, Transformers are RNNs`）。做法：把 `softmax(QKᵀ)V` 改写成 `φ(Q)(φ(K)ᵀV)`——
  先算一个 `d×d_v` 的**固定大小状态**，每步只把它累加更新。代价：检索变成
  「往一个定长关联记忆里读写」，而不再是对全部历史的可微查找。
- **一整个家族都在这条线上**（`记忆·未核`）：`Performer`（随机特征近似 softmax）、
  `S4`/`Mamba`（状态空间模型 SSM）、`RWKV`、`RetNet`（带衰减的线性注意力）、
  `GLA`/`Gated DeltaNet`（门控/增量规则）。它们共同把历史压进**固定大小的状态**。
- **理论侧的限制**：`The Illusion of State in State-Space Models`（`Merrill et al.，约 2024`，
  `记忆·未核`）主张 SSM 在**状态追踪/置换复合**这类问题上受计算复杂度类限制，做不到
  「任意长度」的追踪。`不确定`：这条结论的强度与是否被后续工作削弱。

### 实测结果（定性，全部 `记忆·未核`，数字一律未核）

1. **诊断任务：MQAR（multi-query associative recall）**（`Zoology，Arora et al.，约 2023`）。
   构造 K 个「键→值」对，末尾问键求值。发现**门控卷积 / Hyena / 纯 SSM 在 MQAR 上明显
   逊于甚至远逊于 softmax 注意力**；且「需要回忆的关联数上升时，纯单状态模型准确率明显下降，
   而注意力几乎不掉」。这是业界判定「纯线性不可行」的**主要经验证据**。
2. **复制/长度外推**：`Repeat After Me`（`Jelassi et al.，约 2024`，`记忆·未核`）
   报告 SSM 在**复制与长度外推**上失败；`Based`（`Arora et al.，2024`）也把
   **滑动窗口注意力**加回去以补局部精确复制。`不确定`：确切任务与数字。
3. **共识结论**：**「把无限历史压进固定状态」在需要精确回忆的任务上会塌**；可回忆的信息量
   随状态维度/层数上升，但具体是线性还是次线性**我不确定**。⇒ 这就是**混合（hybrid）**的动机：
   用少量 softmax 注意力层补「精确回忆」这条腿。

### 混合配方用了什么比例、放在哪些层（`记忆·未核`，比例数字均未核，需查原文）

| 系统 | 混合方式 | 我记忆中的比例（**未核**） | 放置 |
|---|---|---|---|
| **Jamba**（AI21，约 2024） | Mamba + Transformer + MoE 交错 | **每个 8 层块约 1 个注意力层 + 7 个 Mamba**（注意力占比约 1/8） | 注意力层在每块固定位置 |
| **Zamba**（Zyphra，约 2024） | 全局共享注意力块 + Mamba | 注意力层很少，且**跨块共享权重** | 一个共享全局注意块 |
| **Griffin / Hawk**（DeepMind，约 2024） | 循环（RG-LRU）+ 局部注意力 | 混合 | 局部注意力 |
| **Samba**（约 2024） | Mamba + 滑动窗口注意力 | 混合 | 窗口注意力 |
| **Hymba**（NVIDIA，约 2024） | 注意力头 + SSM 头**并行**（不是层间交错） | 头级混合 | 同层并行 |
| **Qwen3-Next / MiniMax-01 / Hunyuan 等**（约 2024–2025） | 大批「线性/门控线性 + 少量全注意力」主干 | 全注意力占比小（约每若干层 1 个） | 层间交错 |

- **Jamba 的公开消融（`记忆·未核`）**：报告注意力层数量是**质量的关键旋钮**——去掉/减少
  注意力层质量明显下降；混合体在**同等规模下接近同尺寸 Transformer**，同时吞吐/长上下文更省。
  **确切比例-质量数字未核，必须查 Jamba 论文的 ablation 表。**
- **最接近「ratio sweep」的公开工作**（`记忆·未核`）：
  `An Empirical Study of Mamba-based Language Models`（NVIDIA，约 2024）比较
  **纯 Mamba / Mamba-2 / Transformer / 混合（在不同层插若干注意力层）**，报告
  **纯 Mamba 在多任务上弱于 Transformer，而「少量注意力层」的混合可追平**。
  `不确定`：它是否给出连续的比例扫描曲线；**要查该论文的「hybrid with N attention layers」表**。
- **`未找到`（明确）**：一篇用**同一主干、扫描「注意力层占比」从 0 到 1、给出连续质量曲线**
  的**独立受控研究**。业界多为工程配方，不是干净的 ratio sweep。这正是我们可占位的点。
- **`不确定`**：`Based` / `Gated DeltaNet` 等是否报告过「线性层占比 vs 退化」的曲线。

### 与我们的关系（我们的增量在哪）

- **用户的判断被业界经验证据支持**：纯线性不可行是**经验（MQAR）判定**，不是理论禁令；
  且混合是行业的既定收敛点。**我们不做纯线性**，做的是**「线性层占比 vs 退化」曲线**——
  对应本仓库已有的空白格 `W1-active-width` / 混合算子 (`A6-mixing-operator`)。
- **我们的真增量**：① 13M 小尺度；② **局部先验抗性任务**（SCHEMA §19——业界的 MQAR 是合成的，
  但通常不控制局部先验，也不在 char 级 LM 上给对照）；③ **同 seed、同尺寸、同配方**的
  单因素占比扫描（业界配方不可直接当对照）；④ 把曲线接在本仓库的**检索腿 + KV 消融保留曲线**
  天平上（`tools/evalkit/retention.py`），而不是只报 ppl。
- **必须记住的混杂**（`atlas/playbook.md` 定律 8）：**方向随规模翻转**。13M 上的占比曲线
  只能判**方向**，量级判决须放大。且按 §19，小模型对远距离 context 本就不敏感 ⇒
  「占比高才不退化」这类结论必须在**合成任务**上复核，不能只在 enwik8 上得。

### 可复现的判据建议

- **判据**：横轴 = **注意力层占比**（0 / 1/8 / 1/4 / 1/2 / 1）；纵轴三色 = ① clean bpc（质量）、
  ② 合成长程任务（`tools/evalkit/longrange_tasks.py`，契约 A）的回忆/复制准确率、
  ③ 解析 KV 字节/吞吐。**纯线性（占比 0）必须作为曲线端点**，用于复现「端点崩塌」。
- **对照**：同尺寸 dense 全注意力（占比 1）= 曲线右端自带对照。
- **活性量**：实际被激活的线性层数 / 状态维度；混合体的参数量与 FLOPs 随占比的变化。
- **不要**把「某模型在 MQAR 有名次」当判据；只认**我们自己同 seed 同配方的曲线**。

---

## 2. 块②：GQA / MQA / MLA 尚未发掘的空间

### 谁做过 / 为什么（`记忆·未核`）

- **MQA（Shazeer，2019）**：所有查询头共用**一份** K/V。动机：解码期 KV cache 是显存/带宽瓶颈，
  头数一减，cache 与带宽按比例降。报告**质量损失小**（`数字未核`）。
- **GQA（Ainslie et al.，2023）**：折中——把查询头分成 G 组，每组共用一份 K/V（G=1 即 MQA，
  G=头数即 MHA）。**GQA 论文的关键贡献是「uptraining」**：从已有 MHA checkpoint 转换，
  只需少量继续训练即可恢复质量。报告**近无损**（`数字未核，需查原文的 G 扫描表`）。
- **MLA（DeepSeek-V2，2024）**：把 K/V **联合压成一个低秩潜向量** `c`（维度远小于 `头数×头维`），
  推理时再**上投影**回每头的 K/V；同时把 RoPE 解耦到单独的小分量上以免破坏低秩结构。
  报告**在同等 KV cache（甚至更小）下质量不劣于 MHA**（`数字未核`）。
- **跨层共享（关键、部分对应我们在做的 W3 方向）**（`记忆·未核`）：
  - **CLA（Cross-Layer Attention，约 2024）**：**相邻层共用同一份 K/V**（层间共享）。
  - **YOCO / You Only Cache Once（约 2024）**：只在**一层**算并缓存全局 KV，其余层复用它。
  - **MLKV（Multi-Layer KV，约 2024）**：把 KV 头在**多个层之间**共享。
  - **这些说明「跨层不等共享 / 跨深度共享」在业界已有占位**——但绝大多数是**均匀/固定模式**
    （相邻对、每隔若干层），**几乎没有「按预算最优分配」的**（见下）。
- **`未找到`（明确）**：
  1. **「层特异 KV 头数」的公开工作**——即不同层给不同的 KV 头数（浅层多、深层少），
     按统一预算分配。我记忆里**没有**一篇以「每层 KV 头数不同」为核心贡献的工作；
     `要查`：`layer-wise KV head allocation`、`adaptive GQA heads`、`non-uniform KV sharing`。
  2. **「同一 KV 字节预算下的最优分配」的直接前例**——这正是我们 `W3` 想做的。我**未找到**
     把「量化位宽 / 头数 / 层数 / 保留比例」放进**同一字节预算**、求解最优分配的公开论文。
     这可能是**真空白**，但也可能只是我检索不到（`不确定`）。
- **`不确定`**：上述若干「2024 系统」的确切标题、作者、会议。

### 公开的 Pareto 数字（头数 vs 质量 vs KV 字节）

- `记忆·未核`：GQA 论文与后续工作**通常给「G 扫描 / 质量」表**，但**是否给「质量 vs KV 字节」
  的连续 Pareto 前沿**我不确定；多数只给若干离散点（MHA / GQA-G / MQA）。
- **`未找到`**：一张把 GQA / MQA / MLA 与**量化、驱逐**放在**同一 KV 字节横轴**上的统一 Pareto 图。
  ⇒ 我们的 `W3` 若给出「同字节预算 × 质量」的统一前沿，就是增量（见块③）。

### MLA 的效果与代价（定性）

- **收益**：KV cache 大幅压缩；**训练时**可把 K/V 的参数量也压小。
- **代价**（`记忆·未核`）：① 上投影矩阵带来额外**解码期计算**与实现复杂度（吸收进权重 vs
  未吸收两套算法）；② RoPE 解耦（decoupled RoPE）；③ **必须从零/大规模继续训练**，
  不能像 GQA 那样轻量 uptrain；④ 与「无 final LayerNorm / tied readout」等非标准底座
  组合时行为未知。

### 与我们的关系（我们的增量在哪）

- **`W3` 格子已实测 GQA/MQA 的代价**（`本地实测`，见 §4），**但没做「同字节预算下的最优分配」**。
  业界把「头共享」当**均匀配方**用；**我们的增量 = 把「每层 KV 头数 / 共享模式」当作
  一个可优化变量，在同一字节预算下求分配**——并把结果接在本仓库天平（质量 + 检索腿）上。
- **一个必须防的假象**（SCHEMA §17.12）：比较不同分配规则时，若只对齐「头数」而不对齐
  「实际 KV 字节 / 可见性对数」，字面优势可能是对齐口径造成的。**统一横轴必须是「每位置字节」**。
- **一个已知的规模陷阱**（`atlas/playbook.md` 定律 8）：13M 上「层特异分配」的增益可能放大后消失。
- **排除的解释清单**（写结果时要逐条排除）：增益是否只是**参数减少**（§15 等参数对照）、
  是否只是**收敛加速**（§16 更长训练点）、是否只来自**层内 vs 层间**这一种共享的差异。

### 可复现的判据建议

- **判据**：横轴 = **解析 KV 字节/位置**；纵轴 = clean bpc + 合成长程检索准确率。
  比三条曲线：① 均匀 GQA（现有）；② **层特异头数 / 不等共享**（新）；③ 与 **int8/int4 量化**
  在同一横轴（块③）。
- **等参数对照**：层特异分配会**改变参数量** ⇒ 必须给「去掉该分配、用同参数量均匀分配」的对照
  （SCHEMA §15），否则归因不清。
- **必须报活性量**：每层实际 KV 头数、实际 KV 字节、注意力头使用分布（是否有的 KV 头被闲置）。

---

## 3. 块③：业界有没有把「量化 / 头共享 / 驱逐 / 检索」放进**同一字节预算**比较？

**简短答案**：`记忆·未核` —— **我没找到一篇把四者放在同一字节预算下统一比较的工作**；
各家族各报各的压缩，跨家族统一口径是常见缺口。`不确定`（可能是检索不到）。

### 各家族各自的公开做法（`记忆·未核`）

| 家族 | 代表工作 | 它报的横轴 | 是否统一字节预算 |
|---|---|---|---|
| **量化** | KIVI、KVQuant、Atom 等 | 比特位宽 / 组大小 | 否（按 bit） |
| **头共享** | MQA / GQA / MLA / CLA / YOCO | KV 头数 / 压缩率 | 否（按头/潜维） |
| **驱逐** | H2O、Scissorhands、SnapKV、PyramidKV | 保留 token 数 / 预算 | 否（按 token 数） |
| **检索** | MemGPT、RAG、Unlimiformer | 召回/质量，成本在检索侧 | 否（不报字节） |
| **长上下文基准** | LongBench、RULER、NIAH（haystack） | 任务正确率 | **不统一字节**（是任务指标，不是预算） |
| **综述** | `A Survey on Efficient Inference for LLMs` / KV cache 压缩综述（约 2024） | 综述性质 | 提出缺口，但一般不重做统一实验 |

### 为什么统一口径难（`记忆·未核`，机制推断）

- 四者的成本轴不同（bit / 头 / token / 检索算力），**没有天然共同横轴**；
- 质量口径也不同（ppl / 长上下文任务 / 检索召回）⇒ 各家口径不可直接比。

### 与我们的关系（**直接决定 W3 / R1 / M2 三格判据线怎么定才公平**）

- **本次 W3 已经做了一次同预算对照**（`本地实测`，`archive/wiki-legacy/axes/W-width/W3-kv-head-sharing/README.md`）：
  在**同一「每位置 KV 字节」横轴**上，**−75% 档** GQA 代价 **+0.0143 bpc** vs int4 KV
  **+0.0457 bpc** ⇒ GQA 便宜 ≈3.2×；**−50% 档** int8 KV 仅 **+0.00015 bpc（近无损）**。
  这已经是一张**跨家族（头共享 vs 量化）同字节横轴**的小 Pareto——**这本身就是业界缺的东西，
  是我们的增量**。
- **公平判据线的定义（建议，交给主 Agent）**：所有 W3 / R1 / M2 的臂
  **统一以「解析 KV 字节/位置」为横轴**，纵轴至少两条腿：
  ① clean bpc（质量）；② 合成长程检索召回（保留腿，`tools/evalkit/retention.py`）。
  **禁止**用「压缩率」「保留头数」「比特数」当横轴直接互比（SCHEMA §17.12）。
- **必须同时报**（SCHEMA §17.12）：删了多少 / 压了多少，**以及掉了多少可见性对数**——
  驱逐类与头共享类的「可见性对数」不同（头共享不改可见性对数，驱逐改），这是两条不同的账。

### 可复现的判据建议

- **最小可判定实验**：把现有 R1（int8/int4）、W3（GQA/MQA）、M2（驱逐/保留曲线）的**点值**
  画到**同一「每位置字节」横轴**上，看谁在给定字节下 bpc 最低、召回最高。
  这是**零训练成本的重绘**，可直接作为下一批的「公平参照线总图」。

---

## 4. 块④：我们两条实测数 vs 业界印象（定性对照）

> 数据：`本地实测` — int8/int4 KV 见 `archive/wiki-legacy/axes/R-references/R1-quantization/README.md`
> （实验 `2614-kv-quant-eval-13m`）；GQA/MQA 见 `archive/wiki-legacy/axes/W-width/W3-kv-head-sharing/README.md`
> （实验 `2616-gqa-mqa-13m`）。**勿当外部证据**，此处仅做「与业界印象是否一致」的定性判断。

### (a) int8 KV 近无损 +0.00015 / −50%（`本地实测`）

- **与业界印象**：**高度一致**（`记忆·未核`）。8 位 KV 量化在业界普遍被报为**近无损 / 可用**；
  真正有争议的是 **2–4 bit**。我们的 int4 +0.0457、int8 +0.00015 两条价签的**相对关系**
  （int4 是 int8 的约 300× 损伤）符合「低比特才开始明显掉质量」的普遍印象。
  ⇒ 这条数**与业界印象一致**，可放心当参照线。

### (b) GQA −75% damage +0.0143 / MQA −87.5% +0.0118（`本地实测`）

- **与业界印象**：**大方向一致，但「便宜程度」偏保守**（`记忆·未核`）。
  - 业界印象（`记忆·未核`）把 GQA（尤其从 MHA uptrain 后）说成「近无损」；我们的 13M
    从零训练体落在 **≈1% bpc（+0.012~0.014）**，**正好压在方差界（0.014）上**——
    即「不是免费午餐，但也谈不上崩」。**这不是与业界矛盾**：我们的对照是**从零训练的小模型**，
    没有 uptrain，且 13M 尺度对容量削减更敏感（`atlas/playbook.md` 定律 8）。
  - **值得记的异常**：**MQA（−87.5%，参数更少）的损伤（+0.0118）反而略小于 GQA（+0.0143）**，
    与「头越少越差」的直觉相反。但两者只差 0.0025 ≪ 0.014 ⇒ **不可排序**（`本地实测` 已判），
    且 GQA 两次跨门槛（+0.0143 / +0.0139）⇒ 该差异在噪声内。**定性上不构成对业界印象的反例**，
    但提示：**小尺度上「头的数量」不是单调的质量旋钮**。`不确定`：放大后是否恢复单调（要查
    更大规模 GQA 扫描）。
- **一句话**：我们的量级（~1% bpc 换 75–87.5% KV）**与业界「GQA/MQA 便宜」的印象同向**，
  但我们的读数把它们从「近乎免费」拉回「贴门槛、需等参数对照与更长训练点才能定级」。

### 与我们的关系

- 这两条数**作为参照线**与业界印象不冲突，可以保留；但**不能**在报告里把它们说成
  「业界已证明 GQA 有代价」——业界印象更倾向于「近无损」，我们的保守读法是
  **13M/从零训练**这个特定条件下的结果（§19：小模型结论不得外推为普遍机制结论）。

---

## 5. 若可检索时应查的关键词（交给下一次有 web 工具的 run）

| 块 | 关键词 |
|---|---|
| ① | `Katharopoulos linear transformers are RNNs`, `Zoology MQAR associative recall`, `Repeat After Me copying state space models`, `Illusion of State state space models`, `Jamba attention layer ablation ratio`, `Zamba shared attention block`, `An Empirical Study of Mamba-based Language Models hybrid attention layers`, `Samba sliding window Mamba`, `Hymba parallel attention SSM`, `Based sliding window linear attention` |
| ② | `Ainslie GQA uptraining`, `Shazeer MQA fast transformer decoding`, `DeepSeek-V2 MLA low-rank KV`, `Cross-Layer Attention CLA`, `YOCO you only cache once`, `MLKV multi-layer KV sharing`, `layer-wise KV head allocation`, `adaptive GQA heads`, `non-uniform KV sharing` |
| ③ | `KV cache compression survey`, `unified KV cache budget benchmark`, `quality vs KV bytes Pareto`, `H2O heavy hitter`, `SnapKV PyramidKV`, `KIVI KVQuant 2-bit`, `LongBench RULER needle in a haystack` |
| ④ | `GQA lossless large model`, `MQA quality degradation small model`, `int8 KV cache lossless` |

**核证优先级（若只有一个时间窗）**：② 的 `cross-layer / layer-wise KV sharing`
与 ③ 的 `unified KV byte budget benchmark`——前者决定我们 `W3` 的「同字节最优分配」
是否真空白，后者决定三格判据横轴是否需引用他人口径。

---

## 6. 来源清单

**来源类型图例**：`官网/论文（含链接）`｜`记忆（未核）`｜`本地实测`。
（本 run 无检索工具 ⇒ **无一条「官网/论文（含链接）」**；所有外部条目均为 `记忆·未核`。）

**本地实测（可直接当证据）**
- `archive/wiki-legacy/axes/R-references/R1-quantization/README.md` ＋ 实验 `2614-kv-quant-eval-13m/analysis.md`
  — int8 KV +0.00015（近无损，−50%）；int4 KV +0.0457（−75%）；fp16 基线 1.483571。
- `archive/wiki-legacy/axes/W-width/W3-kv-head-sharing/README.md` ＋ 实验 `2616-gqa-mqa-13m/analysis.md`
  — mha 1.4820 / gqa 1.4963（+0.0143，重跑 +0.0139）/ mqa 1.4938（+0.0118）；
  解析 KV 12288 → 3072（−75%）/ 1536 B（−87.5%）；同 −75% 档 GQA ≈ int4 的 1/3 代价。
- `archive/wiki-legacy/axes/M-carrier/M2-carrier-compression/README.md` — 驱逐已测（+0.058~0.121@13M）；
  保留曲线 + 检索腿（`tools/evalkit/retention.py`）；「低损伤 ≠ 保住了」。
- `archive/wiki-legacy/axes/A-mixing/A3-window-anchor/README.md` — 锚点通道 76.8%→93.3%；硬丢前缀 +0.058~0.121。
- `atlas/playbook.md` 定律 8（规模翻转方向）、定律 9（判据设计）；`harness/SCHEMA.md` §17.10/§17.12/§18/§19。
- `.scratch/34-brainstorm/issues/01-brainstorm.md` — 本批动机与「联网工具不可用」的记录。

**记忆·未核（仅作设计方向，不得当判据）**
- 块①：Katharopoulos 2020；Performer；S4/Mamba；RWKV/RetNet；Zoology(MQAR)；Repeat After Me；
  Illusion of State；Jamba；Zamba；Griffin/Hawk；Samba；Hymba；Based；NVIDIA Mamba 经验研究。
- 块②：Shazeer MQA 2019；Ainslie GQA 2023；DeepSeek-V2 MLA 2024；CLA；YOCO；MLKV。
- 块③：KIVI/KVQuant/Atom；H2O/Scissorhands/SnapKV/PyramidKV；LongBench/RULER/NIAH；KV cache 综述。

**不确定 / 未找到（明确列出，防止后人误当已知）**
1. 混合层比例的**连续 ratio sweep 曲线**：`未找到`干净受控研究（Jamba/NVIDIA 有消融但非连续扫描）。
2. **层特异 KV 头数 / 不等共享**的公开工作：`未找到`（有跨层均匀共享 CLA/YOCO/MLKV）。
3. **「同一 KV 字节预算下的最优分配」**直接前例：`未找到`——可能真空白，也可能检索不到（`不确定`）。
4. 四家族（量化/头共享/驱逐/检索）**统一字节预算比较**：`未找到`单一工作。
5. MLA 的具体质量/KV 数字、GQA 的 G 扫描数字：`数字未核`，需查原文。
