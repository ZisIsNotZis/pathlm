# PRIOR_ART — A 轴（序列混合机制）既有做法与分诊

> 轴：**A · 序列混合机制** ｜ 格子：`A2`–`A10`（`A1`、`A5`、`A7` 已移出研究树，见 `archive/README.md`）｜ 产出：本文件（知识波，K1 线）
> 角色：researcher（只读 + 本文件）｜ 本波**不跑训练/评测、不取 GPU 锁**
> 依据：`SCHEMA.md` §18（既有做法节）、§19（局部先验抗性）、§21（不重复造轮子）
> 性质：**情报，不是实验证据**。无 `results.json`；本地数字一律给文件出处，外部一律 `记忆·未核`。

---

## 0. 检索条件声明（**必读，决定本文的证据等级**）

**本环境联网检索不可用**（`web_search` / `fetch_content` / `source_check` 等 web 工具
未注册；上一轮 researcher 子代理即因此失败——见 `.scratch/34-brainstorm` 记录与
本批 `.scratch/36-parallel/issues/01-parallel.md` 波一说明）。因此本文**无法在线核证**任何外部数字。

据此，本文降级并遵守：

- 凡来自我记忆的论文/系统，一律标 **`记忆·未核`**，且**只允许用作「设计方向」，
  不得作为判据**（SCHEMA §18）。
- 凡来自本仓库的数，标 **`本地实测`** 并给文件路径。
- 凡我不确定是否存在或不确定细节的，标 **`不确定`** 并写「要查什么才能确定」。
- **本文不含任何我无法回溯的精确外部数字**：记忆中的数字一律降级为定性表述或标注「数字未核，需查原文」。宁可少写，不编造。
- 术语就地解释：`bpc`（每字符比特数，越低越好）、`税`（相对普通 decoder-only transformer 多付出的质量代价）、
  `KV`（注意力缓存：模型回头查的键值表）、`sink`（注意力落脚 token，常为首 token）、
  `prefill`（首次前向建缓存）、`ALiBi`（加性距离偏置，让「读得远」变贵）、
  `SWA`（sliding-window attention，滑动窗口注意力）、`Retention/保留腿`（真删掉信息后还能不能答对）。

**给下一次可检索 run 的关键词清单**见 §3。

---

## 0.5 人话摘要（先看这段）

我们这条轴问的是「一个位置能看见谁、用什么算子合并、怎么定位」。本次盘点（只查记忆、不能联网）得到三句话：

1. **我们本地测出来的「窗口 + 锚点/sink」这一族，机制层面业界早就做透了**（滑动窗口、注意力 sink、
   全局 token 都是行业标准件）⇒ 相关子结论应改成 **「业界结论」口径**（「业界结论、非本地实测」），
   不要再声称是本地发现。
2. **我们真正独有的，是「在局部先验极强的任务上，用『真删信息』的判据量出来的那些数」**——
   比如「只保留最近 32 个位置 = +0.212 bpc」这种**累计**代价、以及「低损伤 ≠ 信息保住了」。
   这几条业界不报，属于我们的增量，**保留为本地实测结论**。
3. **有冲突的格子最值得警惕**：我们测出「LRU 式驱逐无收益」「Top-K 稀疏在短上下文净亏」，
   而业界（H2O 等）常报告驱逐有效。这种「结论不一致」按 §21 属 (b) 类——**要做实验，且不能
   把我们的负结论说成「机制不可能」**（§19：只能说「该任务/该尺度上不划算」）。

---

## 1. 块①：既有做法条目（SCHEMA §18 格式，逐格）

> 每格四件事：**谁做过 / 为什么 / 结果 / 我们的增量在哪**。外部来源一律 `记忆·未核`。

### A1 因果性（todo）

- **谁做过**：扩散 / 并行解码 LM 一系——`SEDD`、`Plaid`、`DiffuLLaMA`、`LLaDA`，
  以及半自回归 / 块因果的 `Block Diffusion` 类（`记忆·未核`）。块因果与段内双向常见于
  检索 / 摘要类编码器（如 `记忆·未核` 的 span-mask 训练）。
- **为什么**：想去掉「必须从左到右」的约束，换取并行生成、可插入/编辑、可双向利用上下文。
- **结果（定性，`记忆·未核`）**：并行/扩散 LM 在**可并行与可编辑性**上占优，但在**困惑度 / 长文本质量**上
  同尺寸普遍仍逊于自回归；且近期扩散 LM 在扩大规模后差距缩小。`不确定`：确切差距与是否已被追平。
- **我们的增量**：本格**本地零经验**；本项目测过的「序无关」是**层执行顺序**（C2 的 shuffle-locality），
  **不是**注意力因果性——两格不可互引。若做，增量是 13M + char 级 + 单因素「打乱输入顺序」消融。

### A2 全注意力基线（judged）

- **谁做过**：`Vaswani et al., 2017`（原 Transformer）以来所有 AR LM 的默认形态（`记忆·未核`）。
- **为什么**：这是「无可见性约束」的定义性基线，不是研究假设。
- **结果（`本地实测`）**：13M `B0 = 1.5074 bpc`（1.1 epoch 快照）/ `1.3459`（4.4 epoch 渐近）；
  100M `1.4612`（1.09 epoch）。出处 `axes/A-mixing/A2-full-attention/README.md`、
  `archive/legacy-docs/findings.md`。
- **我们的增量**：无——这是**锚点本身**。行业与本地一致（基底成立）。

### A3 窗口 / 锚点 / sink（judged）

- **谁做过**：
  - 滑动窗口 `SWA`：`Longformer`（Beltagy 2020）、`BigBird`（Zaheer 2020）、
    `Mistral 7B` / `Gemma` 的局部窗口（`记忆·未核`）。
  - 注意力 sink：`StreamingLLM`（Xiao 2023，`记忆·未核`）——**丢掉最前面几个 token 会让流式
    困惑度崩塌，保留少量 sink + 滑动窗口即可无微调恢复**；机制解释是 softmax 归一化需要一个
    「无处可去时的落脚点」。
  - 全局/锚点 token：`Longformer` 全局注意力、`BigBird` 全局 token、`gist tokens`（Mu 2023，`记忆·未核`）。
- **为什么**：把「前缀可丢」变成**结构事实**，使解码可用 ring-buffer、免重 `prefill`，并把成本从
  `O(T²)` 压到 `O(T·w)`。
- **结果（`记忆·未核`）**：SWA 在多数任务上近无损，但**纯窗口会砍掉 sink 导致崩塌**，加回少量 sink 即恢复
  （代价极低）。**`本地实测`**：硬丢左前缀的 bpc 代价 `+0.058~0.121 @13M`；锚点通道使 needle 召回
  `76.8% → 93.3%`；单点距离曲线**不可累加外推**——
  只保留最近 **32** 个位置实测 `+0.212 bpc`（是单点读数的约 15 倍），64/128/192 分别 `+0.081/+0.020/+0.004`；
  窗口外整桶召回 `0.4%`（X2v2），即**信息是真的丢了**。出处 `axes/A-mixing/A3-window-anchor/README.md`、
  `.scratch/07-x1-x3/evidence @7b47555`。
- **我们的增量**：
  - **机制层：业界已有**（SWA + sink 是标准件）⇒ 见分诊表 A3 = (a)：该**机制结论**应改「业界结论」口径。
  - **本地独有**：①「按**累计**预算而非单点」报代价的方法学（SCHEMA §17.11 的来源）；
    ②「低损伤 ≠ 保住了」（检索腿）的正面实例；③ 13M/char 的定量税。这些**保留为本地实测结论**，
    但它们**不是「窗口机制」的发现**，而是「怎么测量」的发现。

### A4 软距离惩罚 / 相对偏置（judged）

- **谁做过**：`ALiBi`（Press, Smith, Lewis 2022，`记忆·未核`）——加性距离偏置，免位置嵌入且改善长度外推；
  `T5` 相对位置 bias（Raffel 2020）；`RoPE`（Su 2021）在长上下文外推族（`NTK-aware`、`YaRN`）。
- **为什么**：用**可微**的距离先验替代**硬**窗口，使长程依赖「变贵」而非「不可能」，并改善泛化。
- **结果（`记忆·未核`）**：相对偏置族在**长度外推**上有效，在**同分布质量**上大体中性或微正。
  **`本地实测`**：dist_pen（本项目 w=0.1 的 `−w·log(1+d)`）是**唯一负税元素**（`1.5006 vs B0 1.5074`，
  `−0.0068`），但**不创造保留**——它把全 head 一起推向局部（per-head 距离 26–43，无 head 保持长程分工），
  且**注意力偏置只改「从哪读」不改「写到哪」**。出处 `axes/A-mixing/A4-soft-distance-penalty/README.md`、
  `.scratch/07-x1-x3/evidence/attn_dist_telemetry.json @7b47555`。
- **我们的增量**：**业界有 ALiBi（(a) 部分）**；但「相对偏置改善质量却**不创造保留**、且在局部先验极强的
  enwik8 上制造 90.3% 的**假保留**命中」是**本项目在局部先验抗性任务上的增量（(b)）**（SCHEMA §19）。
  ⇒ 分诊 A4 = (b)：**ALiBi 本身改「业界结论」口径；「不创造保留」这条保留 judged**。

### A5 膨胀 / 跨步窗（todo）

- **谁做过**：固定稀疏模式——`Sparse Transformer`（Child 2019，strided）、`Longformer` 膨胀窗口、
  `BigBird`（窗口 + 随机 + 全局）；现代块稀疏 / `MInference`（`记忆·未核`）。
- **为什么**：用固定模式覆盖更远范围而省算力（`O(T·√T)` 或 `O(T)`）。
- **结果（`记忆·未核`）**：固定模式在**效率**上有效，但**检索/精确回忆**上常逊于全注意力或学习式选择。
- **我们的增量**：本格**本地零经验**；固定模式与「学习式选择」（A6 已测）的边界清楚。增量 = 13M 上单因素
  的「固定模式 vs 全注意力」税 + KV 消融保留腿（我们的天平）。

### A6 混合算子（judged）

- **谁做过**：`softmax` 注意力（基底）、内容选择式 **Top-K 稀疏**（`NSA`、`MoBA`，2025，`记忆·未核`）、
  **线性注意力 / kernel 化**（`Katharopoulos 2020`、`Performer`、`Linformer`）、
  **SSM / 递归扫描**（`Mamba`/`RWKV`/`RetNet`/`GLA`）、**卷积**（`Hyena`、`Based`）、
  **池化**（`PoolFormer`）、**MLP-Mixer**、**傅里叶**（`FNet`）（均 `记忆·未核`）。
- **为什么**：长上下文下把注意力的 `O(T²)` 成本压下来；不同算子是「成本-表达力」的不同折中。
- **结果（`记忆·未核`）**：长上下文**吞吐**上稀疏/线性/SSM 明显占优；**短上下文质量**上偏差不一；
  混合架构（注意力 + SSM 层交替，`Jamba`/`Zamba`）成为主流折中。**`本地实测`**：内容选择式 Top-K 在
  **T=256 短上下文上净亏**——质量越删越差（k=32 `Δ=−0.0166` 越 0.014 界、k=64 `Δ=−0.0095`），
  且成本**实测上升 2.2×**（24.34 → 54 ms/step）、显存 1.7×；线性注意力在当前实装**不可行**
  （三次尝试均在 `loss.backward()` 停摆、非确定性运行时问题）；线性注意力的**成本交叉点已实测**
  （`2.58@256 → 1.53@2048`，2048 内未回本）。出处
  `axes/A-mixing/A6-mixing-operator/experiments/2614-topk-sparse-attn-13m/`、
  `.../2616-linear-attention-13m/analysis.md @ticket33`。
- **我们的增量**：**算子层：业界已有（(a) 部分，改「业界结论」）**；但「**短上下文（T=256）上稀疏净亏**」
  这个**下界**业界不报（他们的卖点是长上下文）⇒ **A6 = (b)**，本地数值保留。注意 §17.9：
  「省算力」类干预必须报**成本交叉点**，不能只报理论降幅。

### A7 寻址方式（todo）

- **谁做过**：位置式（窗/膨胀）、内容式（softmax `QK` 相似度）、**学习式 top-k 选择**（`NSA`/`MoBA`）、
  块级选择、**检索式**（`RAG`、`kNN-LM` Khandelwal 2020、`RETRO` Borgeaud 2022、`Memorizing Transformer`）（`记忆·未核`）。
- **为什么**：决定「读谁」——位置先验还是内容相似度还是外部索引。
- **结果（`记忆·未核`）**：检索式在**事实召回**上有效但把成本与基础设施转到外部索引；
  学习式选择在长上下文省算力。
- **我们的增量**：本格**本地只有内容式 softmax 与位置式窗口**；学习式/检索式空白。⇒ 见分诊 A7 = (a)。

### A8 位置语义（todo）

- **谁做过**：绝对可学习/正弦（`GPT-2`/`Vaswani`）、相对偏置（`Shaw 2018`/`T5`/`ALiBi`）、
  `RoPE`（Su 2021）、**位置插值 / 外推**（`Chen 2023` PI、`NTK-aware`、`YaRN` Peng 2023）、
  无位置 `NoPE`（Kazemnejad 2023）（`记忆·未核`）。
- **为什么**：位置编码决定长度泛化能力。
- **结果（`记忆·未核`）**：`RoPE + 插值`是业界**长上下文标准件**；绝对位置嵌入有**硬长度上界**。
  **`本地实测`**：绝对位置嵌入使 ring-buffer 解码硬上界 = `seq_len`（本项目 P0 类 bug 教训）；
  相对偏置族（dist_pen）`−0.007`。
- **我们的增量**：RoPE / 插值外推**本地零经验**；本格是本地**部分**覆盖。⇒ 分诊 A8 = (a)：
  位置编码家族与长上下文外推业界已透，直接写「业界结论」，不重做。

### A9 旧信息处理策略（todo）

- **谁做过**：保留 / 硬丢 / 锚豁免（A3）；**压成 summary 槽**（`Compressive Transformer` Rae 2019、
  `AutoCompressors` Chevalier 2023、`ICAE` Ge 2024）；**压成递归状态**（`RMT` Bulatov、
  `Block-Recurrent` Hutchins、SSM）；**外置检索**（`RAG`/`RETRO`/`Memorizing Transformer`/`kNN-LM`）（`记忆·未核`）。
- **为什么**：把长历史压进有界载体，在内存受限下保住远距离信息。
- **结果（`记忆·未核`）**：压缩类必须配**训练信号**（重建/充分性损失）才成立（见 M 轴 `PRIOR_ART` 块③）；
  检索类不掉质量但把成本转到 kNN。**`本地实测`**：窗口训练的直接检验（ticket 32-O）**证伪「信息被搬到近处」**——
  窗口外针召回 `0.4588` vs 同掩码无记忆地板 `0.4545`（`+0.004`），而窗内 `0.6506` vs `0.4458`（`+0.205`）
  ⇒ **是「丢了」不是「搬了」**；但**显式通道 + delta 写入**在关联回忆任务上把准确率从 `0.0137`（≈chance）
  升到 `0.0950`（`6.08× chance`）⇒ 「给通道 + 给训练信号」不在此负结论覆盖内。出处
  `axes/A-mixing/A9-old-info-retention/README.md`、`.scratch/13-four-experiments/evidence @9abc181`。
- **我们的增量**：载体层（summary/递归/检索）**业界已有 ((a) 部分，改「业界结论」)**；
  「**靠掩码逼不出搬运、但给显式通道 + delta 可以**」是本地增量（(b)）。⇒ 分诊 A9 = (b)。

### A10 压缩触发与优先级（judged）

- **谁做过**：按访问频率/重要性驱逐 KV——`H2O`（Zhang 2023，heavy-hitter）、`Scissorhands`（Liu 2023）、
  `SnapKV`（Li 2024）、`PyramidKV`/`Ada-KV`、`StreamingLLM` 的驱逐（`记忆·未核`）。
- **为什么**：在固定 KV 预算下保住「重要」token，报告低预算下质量近无损。
- **结果（`记忆·未核`）**：这一类在**低预算**下被报告有效；但**基线强度与任务覆盖**是其主要争议点，
  我记得后续有工作质疑其稳健性，但**指认不了确切论文**（`不确定`，要查）。
  **`本地实测`**：distance / lru / content 三规则**同等删除预算**对照，**LRU 无可见收益**；
  字面优势经对账后归因于「恰好删掉的**可见性对数**更少」（N=8：distance 2012 对 vs lru 1562 对），
  按对数重对齐后 `lru −0.0170`（由正转负）；`lru:k=256` 出现「bpc 伤害最低但召回崩 4.5→22.8pp」的
  「低损伤 ≠ 保住了」。出处 `axes/A-mixing/A10-compression-trigger/experiments/2615-priority-shootout-13m/analysis.md @ticket32`。
- **我们的增量**：**与业界结论不一致（H2O 报有效，我们同预算下无收益）⇒ A10 = (b)**，保留 judged；
  并按 §19 只表述为「该任务/该尺度上不划算」。**另附（`记忆·未核`）**：跨层 KV 共享有占位
  （`CLA`/`YOCO`/`MLKV`，均均匀模式），但「层特异 KV 头数 / 不等共享 / 同字节预算最优分配」**未找到直接前例**
  （可能真空白，也可能检索不到）——**要做先核证**，不得据此判 (a)。

---

## 2. 块②：机器可读分诊表

> 判定 = **(a)** 业界已研究透（不跑实验，写 `status: todo` + 标「业界结论，非本地实测」）/
> **(b)** 需本地实测（业界没有 / 结论不一致 / 我们设定实质不同）/
> **(c)** 查不到（含检索不可用，按 `记忆·未核` 记方向，**不得**据此判 (a)）。
> 格式：`grid_id | 判定 | 一句话理由 | 拟写进 `catalog.yaml` 的结论`

| grid_id | 判定 | 一句话理由 | 拟写进 `catalog.yaml` 的结论（标注来源类型） |
|---|---|---|---|
| A2-full-attention | **(a)** | 全注意力是所有 AR LM 的**定义性基线**，非研究问题。 | 「无可见性约束的全因果 softmax 注意力是质量锚点。」来源类型：**业界标准 + 本地锚**（本地数值保留：B0 1.5074 / 1.3459 / 100M 1.4612）。 |
| A3-window-anchor | **(a)** | 窗口 + sink/全局 token 是行业标准件（Longformer/BigBird/StreamingLLM）；**机制结论业界已有**。 | 「滑窗注意力近无损；丢弃首 token 会崩，保留少量 sink 即恢复；全局/锚点 token 提供远距离通路。」来源类型：**业界结论，非本地实测**（`记忆·未核`）。**本地方法学增量另行保留**（累计预算代价 +0.212@keep32、检索腿「低损伤≠保住了」）——保留 judged，**不改**。 |
| A4-soft-distance-penalty | **(b)** | ALiBi/相对偏置业界有（(a) 部分），但「改善质量却不创造保留、且在局部先验任务上造假保留」是本地增量。 | 拟写进 `catalog.yaml` = **本地结论**：「软距离惩罚可微地改善 clean bpc（−0.0068，唯一负税），但『改变使用率』不等于『创造保留』——beyond-window 90.3% 命中是局部补全。」来源类型：**本地实测**。ALiBi 子结论可另立「业界结论」 条目。 |
| A6-mixing-operator | **(b)** | 算子层业界已有，但**短上下文（T=256）净亏的下界**业界不报；线性注意力本地实装不可行。 | 拟写进 `catalog.yaml` = **本地结论**：「内容选择式 Top-K 稀疏在 T=256 上净亏（质量越删越差、成本实测 2.2× 上升）；线性注意力在本实装不可行；成本交叉点在 ≥2048 之外。」来源类型：**本地实测**。算子家族本身另立「业界结论」 条目。 |
| A8-positional-semantics | **(a)** | 位置编码家族与长上下文外推（RoPE/插值/YaRN/NoPE）业界已透。 | 「RoPE + 位置插值是长上下文标准件；绝对位置嵌入有硬长度上界。」来源类型：**业界结论，非本地实测**（`记忆·未核`）。本地局部结论（绝对 PE 硬上界）可作 side note。 |
| A9-old-info-retention | **(b)** | 载体（summary/递归/检索）业界已有，但「掩码逼不出搬运、显式通道+delta 可以」是本地增量。 | 拟写进 `catalog.yaml` = **本地结论**：「靠掩码逼不出信息搬运（窗口外 +0.004≈地板）；显式记忆槽 + delta 写入可把关联回忆从 chance 提到 6.08× chance。」来源类型：**本地实测**。载体家族另立「业界结论」 条目。 |
| A10-compression-trigger | **(b)** | **与业界结论不一致**：H2O 等报重要性/频率驱逐有效，我们同预算下无收益（对账后归因于删得更省）。 | 拟写进 `catalog.yaml` = **本地结论**：「同等删除预算下，LRU/content 优先级相对距离规则无净收益；重对齐可见性对数后 lru 转负。该维度在 13M/256 快照上未测出信息量。」来源类型：**本地实测**（§19：只表述为「该任务/该尺度不划算」）。 |
| A10（附）跨层 KV 分配 | **(c)** | 跨层 KV 共享有占位（CLA/YOCO/MLKV 均匀模式），但「层特异/不等共享/最优分配」查不到直接前例。 | 若可检索应核证；**不得**据此判 (a)。方向记 `记忆·未核`。 |

**格子树变更**：`A1-causality`、`A5-dilated-window`、`A7-addressing` 已从研究树移除（删 / 并入 `A6-mixing-operator`，见 `archive/README.md`）；本表只列现存格。

---

## 3. 块③：若可检索应查什么（关键词清单）

| 对应块 | 关键词 |
|---|---|
| A1 因果性 | `diffusion language model perplexity vs autoregressive`, `LLaDA scaling`, `SEDD Plaid`, `block diffusion semi-autoregressive LM`, `boundary bidirectional attention retrieval` |
| A3 窗口/sink | `StreamingLLM attention sink streaming perplexity`, `Longformer sliding window`, `BigBird global tokens`, `gist tokens`, `attention sink emergence`, `massive activations` |
| A4 相对偏置 | `ALiBi length extrapolation`, `T5 relative position bias`, `distance decay attention quality`, `attention locality vs long-range retention` |
| A5 稀疏模式 | `Sparse Transformer strided attention`, `Longformer dilated window`, `BigBird block sparse`, `NSA native sparse attention`, `MoBA mixture of block attention`, `MInference` |
| A6 混合算子 | `linear attention retrieval failure MQAR`, `Performer Linformer`, `Mamba state capacity recall`, `hybrid attention SSM ratio`, `top-k sparse attention short context quality`, `sparse attention crossover perplexity` |
| A7 寻址 | `learned top-k token selection attention`, `RETRO retrieval augmented`, `kNN-LM`, `Memorizing Transformer`, `RAG long context recall` |
| A8 位置 | `RoPE position interpolation`, `NTK-aware scaling`, `YaRN`, `NoPE length generalization`, `absolute position embedding length limit` |
| A9 旧信息 | `Compressive Transformer`, `AutoCompressors`, `ICAE`, `Recurrent Memory Transformer`, `Block-Recurrent Transformer`, `Unlimiformer`, `Memorizing Transformer` |
| A10 触发/驱逐/跨层 | `H2O heavy hitter oracle`, `Scissorhands`, `SnapKV`, `PyramidKV Ada-KV`, `KV eviction reproducibility`, `CLA cross-layer attention`, `YOCO`, `MLKV cross-layer KV sharing`, `layer-specific KV head allocation` |

**核证优先级（若只有一个时间窗）**：`A10 H2O 驱逐复现性`与`A6 稀疏注意力短上下文质量`——
这两条决定我们**与业界冲突**的负结论是否需要改写措辞（(b) 的诚实性依赖于此）。

---

## 4. 来源清单

**来源类型图例**：`官网/论文（含链接）` ｜ `记忆（未核）` ｜ `本地实测`。

**本地实测（可直接当证据；路径均在本仓库）**
- `axes/A-mixing/A2-full-attention/README.md` — B0 = 1.5074/1.3459 bpc（13M）、100M 1.4612。
- `axes/A-mixing/A3-window-anchor/README.md` — 硬丢 +0.058~0.121；锚点 needle 76.8→93.3%；累计 keep32 +0.212。
- `axes/A-mixing/A4-soft-distance-penalty/README.md` — dist_pen −0.0068；per-head 距离 26–43；beyond-window 90.3% 假保留。
- `axes/A-mixing/A6-mixing-operator/README.md` — top-k k=32 Δ−0.0166 / k=64 Δ−0.0095；2.2× 慢；线性注意力不可行 + 交叉点。
- `axes/A-mixing/A8-positional-semantics/README.md` — 绝对 PE 硬上界；dist_pen −0.007。
- `axes/A-mixing/A9-old-info-retention/README.md` — 窗口训练「丢了不是搬了」；显式通道 delta 6.08× chance。
- `axes/A-mixing/A10-compression-trigger/README.md` — LRU 无收益；可见性对数对账；位置 0 mass 9.374。

**记忆（未核，只可用于设计方向，不得当判据）**
- 扩散/并行 LM：SEDD / Plaid / DiffuLLaMA / LLaDA / Block Diffusion
- 稀疏与长上下文：Longformer / BigBird / Sparse Transformer / Mistral SWA / StreamingLLM / NSA / MoBA / MInference
- 混合算子：Katharopoulos linear attention / Performer / Linformer / Mamba / RWKV / RetNet / GLA / Hyena / Based / PoolFormer / MLP-Mixer / FNet / Jamba / Zamba
- 位置：Shaw / T5 bias / ALiBi / RoPE / PI / NTK / YaRN / NoPE
- 旧信息/压缩/检索：Compressive Transformer / Memorizing Transformer / RMT / Block-Recurrent / Unlimiformer / AutoCompressors / ICAE / RAG / RETRO / kNN-LM
- 驱逐/跨层：H2O / Scissorhands / SnapKV / PyramidKV / Ada-KV / CLA / YOCO / MLKV

**未找到 / 不确定（本 run 无法核）**
- 「短上下文（≤512）上稀疏注意力是否净亏」的公开受控报告（我们的卖点疑似空白，但**不能断言**）。
- H2O 一族增益在严格基线/多样任务下**复现性争议**的确切论文出处。
- 「层特异 KV 头数 / 同 KV 字节预算最优分配」是否存在直接前例。
- 扩散 LM 与 AR 在**同尺寸同 token 预算**下的当前确切差距。

## 5. 我不确定的地方（诚实清单）

1. 本文所有外部内容均为**记忆**，**未在线核**；任何具体外部数字若被引用进结论都属违规。
2. 「StreamingLLM 保留 sink 即恢复」我信心较高但**未核**——它若为假，A3 的 (a) 判定需重审。
3. 「H2O 报低预算有效」是记忆，**未核**；A10 的「结论不一致」判定依赖此点。
4. 我**无法排除**已存在与我们 A4/A6 局部先验判定高度重合的工作。
5. 本文件未做任何训练/评测；无 `results.json`，**不是实验证据，只是情报**。
