# PRIOR_ART — 记忆 / 压缩 / 状态架构的既有做法（契约 B）

> 轴：**M 状态载体** ｜ 格子：`M1-carrier-type` ｜ 实验：`2617-prior-art-memory`
> 角色：researcher（只读 + 本文件）｜ 本批**不跑训练/长评测**
> 依据：`harness/SCHEMA.md` §18（既有做法节）与 §19（局部先验抗性）

---

## 0. 检索条件声明（**必读，决定本文的证据等级**）

**本 run 未注册任何 web / 检索工具**（本子代理可用工具仅 `read` / `write` /
`contact_supervisor`；`web_search`、`source_check` 均不可调用）。因此本文**无法在线核证**
任何外部数字。

按契约 B 的授权条款，本文**降级为「记忆（未核）+ 本地实测」**，并遵守：

- 凡来自我记忆的论文/系统，一律标 **`记忆·未核`**，且**只允许用作「设计方向」，
  不得作为判据**（SCHEMA §18）。
- 凡来自本仓库的数，标 **`本地实测`** 并给文件路径。
- 凡我不确定是否存在、或不确定细节的，标 **`不确定`** 并写「要查什么才能确定」。
- **本文不含任何我无法回溯的精确数字**：凡记忆中的数字我一律降级为**定性表述**或
  明确标注「数字未核，需查原文」。宁可少写，不编造。

**给下一次可检索 run 的关键词清单**（按块给出，见 §6）。

---

## 1. 块①：单状态极限（RNN / LSTM / GRU / Mamba / RWKV / RetNet / 线性注意力）

### 谁做过 / 为什么

- **线性注意力与 SSM 的「回忆缺陷」是一整条研究线**：`Zoology / MQAR`（Arora et al.,
  约 2023，`记忆·未核`）用**多查询关联回忆（MQAR）**合成任务做诊断，发现门控卷积、
  Hyena 等在**关联回忆**上远逊于 softmax 注意力，并指出「记忆容量 = 吞吐的对价」。
  `Repeat After Me`（Jelassi et al.，约 2024，`记忆·未核`）进一步报告**SSM 在复制/长度外推
  上失败**。`不确定`：两篇的确切标题、年份、会议与数字。
- **改进方向是「更会写的状态更新规则」而不是「更大的状态」**：delta 规则系
  （`DeltaNet`、`Gated DeltaNet`、`DeltaProduct`，约 2024–2025，`记忆·未核`）把状态更新写成
  「写入 `v − Wk` 的残差」，即**只搬状态里还没有的那部分**（见块④）。
- **纯线性/纯递归在工程上被「混合」取代**：Mamba / Mamba-2、RWKV-4/5/6、RetNet、
  Griffin/Hawk 等一系列工作最终都走向**「若干递归/线性层 + 少量 softmax 注意力层」**的混合
  （Jamba、Zamba、Samba、Qwen3-Next、MiniMax-01 等，`记忆·未核`）。**比例-质量曲线属契约 C**，
  本文不重复。
- **理论侧的限制**：`The Illusion of State in State-Space Models`（大约 2024，`记忆·未核`）
  主张 SSM 在计算复杂度类上受限（状态追踪类问题如置换复合做不了）。`不确定`：确切结论强度。

### 实测结果（定性，全部 `记忆·未核`）

1. **「无限历史压进固定状态」在需要精确回忆的任务上确实会塌**：所有上述工作都观测到
   「随需回忆的关联数上升，纯单状态模型的准确率明显下降，而注意力几乎不掉」。
2. **可回忆的信息量随状态维度单调上升**：业界用 **「MQAR 准确率 vs 状态大小 / 层数」曲线**
   作为标准量化手段；**具体是线性还是次线性我不确定**，`要查`：Zoology 的 MQAR 容量曲线、
   Mamba-2 关于 state size 的消融。
3. **改善顺序（从弱到强，按我记忆的共识排序）**：无门控线性 → 门控（decay）→ 选择式
   （Mamba 的输入相关 Δ/B/C）→ delta 规则（残差写入）→ delta + 门控（Gated DeltaNet）。
   `不确定`：这条排序的确切证据强度与是否对所有规模成立。

### 与我们的关系

- **它直接回答用户的混杂**：业界确实**观测到**「单状态在回忆任务上塌」，但**同时**发现
  塌的边界是**「状态容量 ÷ 需回忆信息量」的比值**——所以「搬不动」与「不需要」在业界也是
  靠**把需求做大**（MQAR 的 K 变大）来区分的。这正面支持用户的方向：**我们必须把「需要」
  做成一个可调的旋钮**（契约 A 的变距离/关联回忆任务的就是这个旋钮）。
- **它给出「极限形态」的对照物**：我们要做的单状态臂，业界已有大量现成实现与标准任务
  （MQAR），**不要自己发明任务**；站在肩膀上 = 复用 MQAR 式构造 + 我们自己的局部先验抗性要求。

### 可复现的判据建议

- **判据**：在合成长程任务（契约 A）上画**「需回忆关联数 / 距离 d」× 「准确率」曲线**，
  以**同尺寸 13M dense 全注意力**为对照。
- **不要**把「某模型在 MQAR 上有名次」当判据；只把**我们自己同 seed 同配方的曲线**当判据。
- 单状态臂的**必要活性量**：状态维度 d_state 与「能回忆的关联数上界」的关系（哪怕只是
  「d_state 翻倍，可回忆对数是否翻倍」的两点测量）。

---

## 2. 块②：显式压缩记忆（Compressive Transformer / Memorizing Transformer / RMT / Unlimiformer / Block-Recurrent / DNC·NTM）

### 谁做过 / 为什么（`记忆·未核`）

| 系统 | 它做了什么 | 它声称的动机 |
|---|---|---|
| **Transformer-XL**（Dai et al., 2019） | 段级递归：缓存**上一层**的隐藏状态当「记忆」，配相对位置编码 | 不做压缩；用**缓存**换更长有效上下文，免重算 |
| **Compressive Transformer**（Rae et al., 2019） | 两级记忆：近期原始 + **被压过的旧记忆**；压缩率 c（strided conv / 池化） | 内存有界的同时保住远距离；**且用了「压缩损失」辅助项** |
| **Memorizing Transformer**（Wu et al., 2022） | 可训练层 + **kNN 检索**非可微外部记忆 | 检索式保留，不压 |
| **Block-Recurrent Transformer**（Hutchins et al., 2022） | 跨块**递归状态**（一组状态向量），窗口注意力 | 把历史压进递归状态 |
| **RMT**（Bulatov et al., 2022 / 2023） | 段级递归 + **特殊 memory token** 携带跨段信息 | 把有效上下文推到 1000s→1M |
| **Unlimiformer**（Bertsch et al., 2023） | 给预训练 encoder-decoder 加 **kNN 跨注意力**，覆盖全部历史隐藏态 | 不改预训练权重即可无限长输入 |
| **AutoCompressors**（Chevalier et al., 2023） | 把长上下文压成 **summary 向量**，用**重建损失**训练 | 展平压缩 + 复用 |
| **ICAE**（Ge et al., 2024） | 把上下文压成少量 memory slot，**L 个 slot ≈ 长上下文**，用重建（自编码）训练 | 上下文压缩 |
| **DMC**（Nawrot et al., 2024，NVIDIA） | KV 动态压成单槽；**需要继续预训练/蒸馏** | 推理内存 |
| **NTM / DNC**（Graves et al., 2014 / 2016） | 可微外部记忆 + 读/写头 + **基于使用量的分配（LRU 式）** | 合成任务（copy / bAbI）上证明「练出来的记忆」可行 |

### 实测结果（`记忆·未核`，**数字一律未核**）

1. **压缩率与质量是明确的单调权衡**：压缩比越大，质量越掉；`Compressive Transformer`
   一类工作报告中低压缩率（**约 3–4× 量级**）在**同等内存预算**下几乎不亏甚至更好
   （等价于「同样预算下留更久」）。**确切压缩率与 bpc/ppl 数字我不核，必须查原文。**
2. **「压得动」的前提是训练信号，而不是结构本身**（这一条被反复观测，见块③）。
3. **外部可微记忆（NTM/DNC）在小合成任务上强，但极难训练、几乎没进主流 LM**：
   这是「显式搬运」路线最贵的历史教训。
4. **检索式（Memorizing Transformer / Unlimiformer）不掉质量但把成本转到 kNN 检索**：
   「质量保持」与「省钱」是两条独立的腿（与我们 `R4` 的教训一致）。

### 与我们的关系

- **我们的 `M2-carrier-compression` 空白格（摘要/低秩）全都在业界的现成盘点里**，
  我们不是从零造轮子，而是**在 13M / char-enwik8 / 局部先验抗性任务上做受控对照**。
- **回收一条本地警告**：我们已经在 `M2` 里付过学费——「低损伤 ≠ 保住了」
  （`archive/wiki-legacy/axes/M-carrier/M2-carrier-compression/README.md`）。这正是 Compression 系工作
  **必须带检索腿**的原因；业界很多压缩工作只报 ppl/质量，**不报可重建性**，其「几乎无损」
  结论在我们的天平上不可直接搬用（`本地实测` + 研究推断）。

### 可复现的判据建议

- **双边前沿**：横轴 = 每 token 的**字节预算**（不是「压缩率」这种自造口径），
  纵轴双色 = ① clean bpc/质量 ② **检索腿召回**（`tools/evalkit/retention.py`）。
- **对齐口径**：比较不同压缩方案时必须报「掉了多少可见性对数」（SCHEMA §17.12），
  否则会重演 `A10` 的字面优势假象。

---

## 3. 块③：**强制搬运的训练信号**（重建 / 充分性 / 信息率惩罚）—— 对应 `F4-training-signal` 空白格

> **本节是本文最重要的一节。结论先行：我们的 `F4` 空白格在业界**不是空白**，
> 而是一条 2019 年就开线、2023–24 又热起来的成熟路线。**

### 谁做过 / 为什么（`记忆·未核`）

1. **Compressive Transformer（Rae et al., 2019）**：除主任务 CE 外，加一条**压缩损失**：
   让模型**从压缩记忆里重建原始记忆**（其一对变体是用注意力去重建）。
   动机原话大意：**只有逼模型把被丢掉的内容记住，压缩才有意义**。
   —— 这与我们契约里写的「惩罚模型编造自己丢掉的内容」**是同一个思想**。
2. **AutoCompressors（Chevalier et al., 2023）**：用**重建损失**训练 summary 向量，
   并讨论「重建原始 token vs 重建关键信息」的差异。
3. **ICAE（Ge et al., 2024）**：压缩上下文到 memory slot，**自编码（重建）**训练；
   报告 L 个 slot 可在部分任务上替代长上下文。
4. **`500xCompressor` 一类超高压工作（约 2024）**：以重建目标追求极高压缩率。
   `不确定`：确切标题与数字。
5. **RECOMP（Xu et al., 2023）**：检索文档压缩器，明确对比**重建/抽取目标 vs 蒸馏大模型**，
   并报告**小模型 + 重建目标可以胜过蒸馏**。`不确定`：这一具体结论的强度。
6. **DMC（2024）**：KV 压缩**必须配继续预训练/蒸馏**，否则质量掉很多——
   反过来说明：**「压缩结构 + 无训练信号」的组合在 KV 侧被实测为不够**。
7. **信息率惩罚 / 瓶颈类**：`不确定` 是否能指认一篇「对记忆状态加信息率惩罚来做 LM 记忆」
   的标杆论文。`要查`：`information bottleneck` + `KV cache` / `memory slot`。

### 实测结果（`记忆·未核`）

- **一致结论：重建/充分性类的辅助损失是「激进压缩能成立」的必要条件之一**；
  在只报质量的作品里，去掉它通常显著掉点（尤其高压比）。
- **代价**：需要额外解码路径（算力/显存），且**辅助损失会与主任务抢容量**
  （与我们 `atlas/playbook.md` 定律 6「整形 vs 自适应是同一梯度的两面」同构）。
- **`不确定`**：在**小模型 / 合成任务 / 同 seed 受控**条件下，「有 vs 无 sufficiency loss」
  的净效应，我没有找到可直接引用的公开对照。

### 与我们的关系（**直接改变契约 D/F4 的措辞**）

- 我们课题页写「重建/sufficiency 是**唯一无人占位**」——**那是相对本项目而言**。
  **业界已有占位**，我们不该把它表述成全新构想（SCHEMA §18 要求）。
- **我们真正的增量**是：① 13M 小尺度；② **局部先验抗性任务**（SCHEMA §19，业界合成任务
  通常不控制局部先验）；③ **同 seed、同尺寸、单因素**的「有/无辅助损失」受控对照；
  ④ 把 sufficiency 损失接到**右移压缩**的载体上（而不是压缩器自身）。
- **风险预警**：`F4` 本地已测 consistency loss **中性**（`本地实测`：
  `archive/wiki-legacy/axes/F-iteration/F4-training-signal/README.md`）；sufficiency 损失在 bf16 下
  属于「负/奖励类」风险区，必须 float32 + pre-step grad-norm gate（SCHEMA §10、playbook 定律 5）。

### 可复现的判据建议

- **最小可判定实验**：两臂 A/B——同 13M、同 seed、同数据，A = 纯 CE，B = CE + λ·重建损失；
  **判据线必须落在「保留腿」上**（窗口外/远处针的**可重建性**），而非只落 bpc。
- **必须报活性量**：重建损失的下降轨迹 + 状态/槽位的实际使用率（SCHEMA §10）。
- **必须做「λ = 0 的逐位等价」单测**（实现层 equality，SCHEMA §9）。
- **反噬检查**：主任务 bpc 是否变差（`F4` 的 consistency loss 就是中性偏无收益的先例）。

---

## 4. 块④：LRU / 增量搬运 / 按访问频率淘汰

### 谁做过 / 为什么（`记忆·未核`）

1. **按访问频率/重要性淘汰 KV**：
   - **H2O（Heavy-Hitter Oracle, 2023）**：按累计注意力分数保留「重击者」；
     报告在较小 KV 预算下保持质量。
   - **Scissorhands（2023）**：基于「重要性在时间上持续」（persistence of importance）假设做保留。
   - **SnapKV / PyramidKV / Ada-KV 等（2024）**：用观测窗口选择要留的 token。
   - **`不确定`**：这些方法的**增益在更严格基线下的复现性**——我记得后续有工作质疑
     各类驱逐法在多样任务上的稳健性，但**指认不了确切论文**，必须查。
2. **LRU 式分配的直接祖先在可微外部记忆**：**NTM / DNC** 用**使用计数**做分配
   （least-used 优先、free list）——这就是「按访问频率淘汰」在神经网络里的原始形态。
3. **「只搬上次没搬的」= delta 规则**（**本文对用户 LRU 构想的最强对应物**）：
   线性 RNN 的状态更新写成 `S ← S + β(v − Sk)kᵀ`，即**先减掉状态已能预测的部分，只写入残差**。
   `DeltaNet` / `Gated DeltaNet` / `DeltaProduct`（约 2024–2025，`记忆·未核`）把它做成了
   可并行训练的实现，并报告在 **MQAR / 长程回忆**上优于 Mamba-2 一类。
   `不确定`：具体增益数字与是否在所有规模成立。
4. **`本地实测`（重要，且与业界叙事有张力）**：
   `archive/wiki-legacy/axes/A-mixing/A10-compression-trigger/README.md` —— 我们在 13M/256 快照上做
   distance / lru / content 三规则同等删除预算对照，**LRU 未测出可见收益**；
   其字面优势经对账后归因于「恰好删掉的可见性对数更少」，按对数重对齐后 **lru −0.0170
   （由正转负）**；且 `lru:k=256` 出现「bpc 伤害最低（+0.0885）但召回崩 4.5→22.8pp」
   的典型「低损伤 ≠ 保住了」。三种规则**确实删得不一样**（Jaccard 0.034/0.270/0.472）
   却没换来收益。

### 实测结果概览

- **业界（`记忆·未核`）**：频率/重要性类驱逐在**低预算**下被报告有效；
  但**基线强度与任务覆盖**是这些结论的主要争议点。
- **本地（`本地实测`）**：LRU 相对于距离规则的**净增益在 13M/256 上不存在**；
  「访问频率」这一维度在该配置下未测出信息量。

### 与我们的关系

- **这是「用户拿到的第二个混杂」**：`A10` 的负结论是在**局部先验极强的 enwik8** 上得的。
  按 SCHEMA §19，它只能解释为「**该任务/该尺度上不划算**」，**不得**说「LRU 机制不可能」。
- **delta 规则给了 LRU 一个可形式化的机器**：与其做「哪块 KV 该删」的调度（`A10` 已负），
  不如做「**状态写入残差**」的更新规则（块①/④ 的 delta 系）——这把「LRU」从**驱逐策略**
  变成**写入算子**，可直接在 13M 上做单因素对照。

### 可复现的判据建议

- **若做 delta-规则臂**：判据 = 合成长程任务（契约 A）上的回忆曲线 + clean bpc 税；
  对照 = 同尺寸 dense 全注意力 + **同参数量/同 FLOPs 的门控线性对照**（SCHEMA §15，
  否则会把「重参数化增益」误记为「delta 规则」）。
- **若重做 LRU 类调度**：必须在**局部先验抗性任务**上重测，并沿用 `A10` 的对账口径
  （同时报「删多少位置」与「掉多少可见性对数」）。

---

## 5. 块⑤：Attention sink / StreamingLLM（为什么必须留前几个 token）

### 谁做过 / 为什么（`记忆·未核`）

- **StreamingLLM（Xiao et al., 约 2023）**：发现**丢掉最前面几个 token 会让流式困惑度崩塌**；
  保留 **少数几个「attention sink」token + 滑动窗口**即可恢复，且**无需微调**即可流式解码。
  机制解释：softmax 注意力权重必须归一化，模型需要一个「**无处可去时把注意力放这里**」的
  无信息落脚点；初始 token 天然成为该落脚点。
- 相关工作：`When Attention Sink Emerges in Language Models`（约 2024）、
  `Massive Activations`（约 2024）——都把 sink 与**异常大的激活/范数**联系起来。
  `不确定`：确切标题、年份、以及「sink 是训练涌现还是结构必然」的定论。

### 实测结果

- **业界（`记忆·未核`）**：不加 sink 直接驱逐前缀会崩；加回少量 sink 就恢复——
  **代价极低、收益极大**，是少有的「免费」结论。
- **本地（`本地实测`）**：
  - `archive/wiki-legacy/axes/A-mixing/A3-window-anchor/README.md`：**锚点通道是真产物**——
    needle 召回 76.8%，加距离惩罚后 **93.3%**。
  - `archive/wiki-legacy/axes/A-mixing/A10-compression-trigger/README.md`：注意力质量分布里
    **位置 0 的 mass 9.374 = 第 25 百分位的 12.7 倍**，「最少被读」几乎全在中段
    —— 与业界 sink 描述**独立吻合**（我们的模型是 char-enwik8 的 13M，也出现了 sink）。
- **张力记录**：我们的 `A4` 软距离惩罚**「改变使用率，不改变依赖是否为零」**（`本地实测`）
  ——sink 属于「使用率」侧。对 clean bpc 的效应 −0.0068 落在 Δ<0.014 的不可排序带内
  （`harness/methodology.md`）⇒ 只能称「**未观察到超带损伤**」，不可据此排序为「改善」。

### 与我们的关系

- **给我们的「锚点」设计一个既有解释**：锚点不是我们的发明，而是业界已识别的
  **softmax 归一化落脚点**；我们的增量是**把它做成显式的、可豁免驱逐的槽位**并测了召回。
- **对契约 D（右移搬运）的直接含义**：任何「只留最近 k 个」的窗口都会**砍掉 sink**，
  这可能是窗口外信息崩掉的一个**可分离原因**——值得作为契约 D 的一个**最小可判定实验**
  （「窗口 + 显式 sink」vs「窗口」）。

### 可复现的判据建议

- **判据**：在 `A3` 既有 needle 电池上，比较「窗口」「窗口+sink」「窗口+锚点槽」三臂的
  召回腿 + clean bpc；**并报位置 0 的注意力质量**作为 sink 活性量（我们已有测量代码路径）。
- 不主张「sink 一定存在于所有规模」——只报**本 checkpoints 的实测**。

---

## 6. 若可检索时应查的关键词（交给下一次有 web 工具的 run）

| 块 | 关键词 |
|---|---|
| ① | `MQAR multi-query associative recall`, `Zoology efficient language models recall`, `Repeat After Me copying SSM`, `Mamba-2 state size recall`, `Gated DeltaNet MQAR`, `hybrid attention SSM ratio` |
| ② | `Compressive Transformer compression rate`, `Memorizing Transformer`, `Recurrent Memory Transformer 1M tokens`, `Unlimiformer`, `Block-Recurrent Transformer`, `NTM DNC differentiable memory`, `DMC dynamic memory compression ratio` |
| ③ | `Compressive Transformer compression loss reconstruction`, `AutoCompressors summary vectors reconstruction loss`, `ICAE in-context autoencoder`, `500xCompressor`, `RECOMP reconstruction vs distillation`, `sufficiency loss memory`, `information bottleneck KV cache` |
| ④ | `H2O heavy hitter oracle KV eviction`, `Scissorhands persistence of importance`, `SnapKV`, `KV cache eviction reproducibility`, `DeltaNet delta rule residual write`, `NTM usage-based allocation` |
| ⑤ | `StreamingLLM attention sink`, `When attention sink emerges`, `massive activations LLM` |

**核证优先级（若只有一个时间窗）**：`③ Compressive Transformer compression loss`
与 `① MQAR 容量曲线`——前者决定我们 `F4` 的措辞是否需要改，后者是我们
「需求旋钮」的行业标准。

---

## 7. 来源清单

**来源类型图例**：`官网/论文（含链接）` ｜ `记忆（未核）` ｜ `本地实测`。

**本地实测（可直接当证据）**
- `archive/dropped/axes/M-carrier/M3-cross-step-assumption/README.md` — 窗口 32 训练税 +0.0866、
  窗口 192 −0.0028；窗口外整桶召回 0.475954 vs 无记忆地板 0.444444（+3.2pp）；
  **整段都在窗口外的 494 根差恰好 0.000**；窗内阳性对照 +28.2pp。⇒ 「丢了」不是「搬了」。
- `archive/wiki-legacy/axes/A-mixing/A9-old-info-retention/README.md` — 同一结论的另一处记账（ticket 32-O）。
- `archive/wiki-legacy/axes/A-mixing/A3-window-anchor/README.md` — 锚点通道 76.8%→93.3%；硬丢 +0.058~0.121。
- `archive/wiki-legacy/axes/A-mixing/A10-compression-trigger/README.md` — LRU vs distance vs content 同预算对照；
  重对齐后 lru −0.0170；位置 0 mass 9.374 = 第 25 百分位 12.7×。
- `archive/wiki-legacy/axes/F-iteration/F4-training-signal/README.md` — consistency loss 中性。
- `archive/wiki-legacy/axes/M-carrier/M1-carrier-type/README.md`、`M2-carrier-compression/README.md` — 格子盘点。
- `atlas/playbook.md` 定律 7、9；`harness/SCHEMA.md` §17.10/§17.11/§17.12。

**记忆（未核，只可用于设计方向，不得当判据）**
- Compressive Transformer / Memorizing Transformer / RMT / Unlimiformer / Block-Recurrent /
  DNC / NTM / Transformer-XL / AutoCompressors / ICAE / 500xCompressor / RECOMP / DMC
- Zoology(MQAR) / Repeat After Me / Based / Mamba / Mamba-2 / RWKV / RetNet / Griffin·Hawk /
  DeltaNet / Gated DeltaNet / Illusion of State
- StreamingLLM / When attention sink emerges / Massive Activations
- H2O / Scissorhands / SnapKV / PyramidKV / Ada-KV

**未找到 / 不确定（本 run 无法核）**
- 「单状态可回忆信息量 vs 状态维度」的**精确函数形式**（线性/次线性）与常数。
- `Compressive Transformer` 等工作的**确切压缩率、bpc/ppl、对照基线强度**。
- **小尺度 + 局部先验抗性任务**上「有 vs 无 sufficiency loss」的公开受控对照（疑似空白，
  但**不能断言**——检索不可用）。
- H2O 一族增益在严格基线下的**复现性争议**的具体论文出处。
- 是否有「对记忆状态加信息率惩罚」的标杆论文。

---

## 8. 三条对我们设计的直接含义（给契约 D 的接口）

1. **别把「重建损失」当全新构想**：它是 2019 年就有的既有做法（Compressive Transformer），
   且 2023–24 在压缩记忆上被反复重做。我们的增量应写成
   「**小尺度 + 局部先验抗性任务 + 单因素受控**」，而不是「首次提出」。
2. **「LRU 式增量搬运」的最强既有形态不是驱逐策略，而是 delta 写入规则**
   （只写状态预测不出的残差）。这比「挑哪块 KV 删」更接近用户的原话，也更容易做单因素对照。
3. **sink 是既有解释**：窗口类设计必须先回答「你砍掉了 sink 吗」——这是一个便宜、
   高信息量的**可分离对照**（窗口 vs 窗口+sink）。

## 9. 我不确定的地方（诚实清单）

1. 本文所有外部内容均为**记忆**，**未在线核**；任何一个具体数字若被引用进结论都属违规。
2. 「Compressive Transformer 用了重建/压缩损失」这一条我**信心较高但未核**——它若为假，
   本节③的核心命题需重写。**这是最该优先核证的一条。**
3. delta 规则与「LRU/只搬没搬的」的对应关系是**我的解释**，不是论文原话。
4. 我**无法排除**已存在与我方 `F4` 设计高度重合、且恰好在小尺度/合成任务上做过的工作。
5. 本文件未做任何训练/评测；无 `results.json`，不是实验证据，**只是情报**。
