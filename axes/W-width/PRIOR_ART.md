# PRIOR_ART — 宽度（激活宽度 / 专家化 / KV 头共享 / 读出几何）的既有做法

> 轴：**W 宽度** ｜ 格子：`W1-active-width` … `W4-readout-geometry`（`W5` 已从研究树删除，分类事实归 `W2`，见 `archive/README.md`）
> 角色：researcher（只读 + 本文件）｜ 本波**无 GPU、不跑训练/评测、不取锁**
> 依据：`SCHEMA.md` §18 / §19 / §21

---

## 0. 检索条件声明（**必读**）

**本环境联网检索不可用**。本文所有外部内容**无法在线核证**，一律标 **`记忆·未核`**，
只作设计方向、**不得当判据**。本地数标 `本地实测` + 路径。不编数字。
术语就地解释：**MoE**（混合专家：每 token 只激活少数专家，参数量与每 token 计算解耦）；
**GQA/MQA/MLA**（KV 头共享三种形态：分组共享 / 全共享 / 低秩压成潜向量）；
**KV**（注意力缓存：模型回头查的键值表）；**税**（相对普通 transformer 的质量代价）；
**bpc**（每字符比特数，越低越好）；**tied 读出**（输出层直接复用输入 embedding 的转置）。

**若可检索应查的关键词清单**见 §3。

---

## 1. 既有做法（按格）

### W1 — 激活宽度（减宽 / 宽度 dropout）

**谁做过 / 为什么（`记忆·未核`）**

- **结构化剪枝**一系：`LLM-Pruner`、`Sheared LLaMA`、`Minitron`（NVIDIA）、
  通道/宽度剪枝——动机是**减参数、减计算**，代价是质量。
  业界共识：**结构化剪枝比非结构化更掉质量**（硬件友好但损害大）。
- **弹性宽度**：`MatFormer`（约 2023–24）等做「一个模型嵌套多个宽度」，`width dropout` /
  `stochastic depth` 类正则——动机是**一次训练得到多档成本**。
- **共识**：**减宽 = 减容量，需要付出质量**；它**不买每 token 的固定成本结构**
  （除非改变 FLOPs 结构，如稀疏激活）。

**我们的增量**：13M 上的**机制探针**——「结构化约束会把状态能量整体迁出被约束子空间」
（切片能量 0.85 → 0.026 → ≈0，`本地实测`）。业界报「剪枝掉点」，**少报「能量迁出」这一机制**。

**「业界是否已有此结论」判定（`W1` 是 `status: judged`）**：
- 「减宽 = 减容量、要付质量」→ **业界已有**（结构化剪枝一系，`记忆·未核`）⇒
  **该条标「业界结论」口径**（来源类型 = 业界结论，非本地实测）。
- 「状态能量迁出被约束子空间」这一**机制解释** → **(b)** 本地增量，保留 `judged`。

**可复现的判据建议**：判据 = Δclean bpc + workspace 探针（null 能量占比 / functional_dirs）。
**必须**给**等参数/等 FLOPs 对照**（§15），否则会把「减容量」与「重参数化」混淆。

---

### W2 — 专家化（MoE）

**谁做过 / 为什么（`记忆·未核`）**

- **MoE 是「同 FLOPs 更多参数」唯一被大规模验证的路线**：`Switch Transformer`、
  `GShard`、`Mixtral`、`DeepSeek-MoE`（**细粒度专家 + 共享专家**成为主流配方）、
  `Qwen-MoE` 等。动机：把参数量与每 token 计算解耦，扩容量而非扩 FLOPs。
- **已知工程风险**：**负载均衡**与**路由坍缩**（专家不均衡）——有 aux-loss / expert-choice
  等大量解药工作。
- **细粒度专家**（`DeepSeekMoE`）：把大专家切小、激活更多小专家——与用户提出的
  「激活更小专家而非更大专家」的**子问题一致**。

**「业界是否已有此结论」判定（`W2` 是 `status: todo`）**：
主结论 **(a) 业界已研究透**——MoE 的「稀疏宽 = 保容量减计算」在大规模上被反复验证；
异构/细粒度/共享专家配方成熟 ⇒ **建议写 `status: todo`**（来源类型 = 业界结论，非本地实测），
**不必在 13M 跑实验**（13M 上 MoE 专家数量少、意义有限，SCHEMA §7 规模纪律）。
若**唯一**要做，需证明「13M + 局部先验抗性任务 + 单因素受控」是实质不同设定 ⇒ 目前判 **(b) 存疑**，
默认按 (a) 处理。

**可复现的判据建议（若真要做）**：同 FLOPs 的质量 + 负载均衡 + 路由稳定性 + 内存；
**必须**报活性量（专家使用率分布、是否坍缩）。

---

### W3 — KV 头共享（MQA / GQA / MLA）

**谁做过 / 为什么（`记忆·未核`）**

- `Shazeer 2019`（MQA）：所有查询头共用一份 K/V——降显存/带宽。
- `Ainslie et al. 2023`（GQA）：分组共享（G=1 即 MQA，G=头数即 MHA）；关键贡献是
  **uptraining**（从 MHA 转换少量继续训练即恢复质量），报告**近无损**。
- `DeepSeek-V2 2024`（MLA）：K/V 联合压成低秩潜向量，推理上投影；同等或更小 KV cache 下
  质量不劣，代价是**必须从零/大规模训练**。
- **跨层共享**：`CLA`、`YOCO`、`MLKV`（约 2024，`记忆·未核`）——相邻层/跨层共享统一 KV。

**我们的增量**：**同「每位置 KV 字节」横轴**上把 GQA/MQA 与 int8/int4 量化放在一起，
给出「−75% 档 GQA 比 int4 便宜 ≈3.2×」的**跨家族小 Pareto**；且报**从零训练 13M** 的代价
（贴方差界 0.014），而非 uptrain 后的近无损。**统一字节横轴的跨家族比较是业界缺口**。

**「业界是否已有此结论」判定（`W3` 是 `status: judged`）**：
- 「GQA/MQA 降 KV、质量近无损」→ **业界已有**（`记忆·未核`）⇒
  **该条标「业界结论」口径**（来源类型 = 业界结论，非本地实测）。
- 「从零训练 13M 下 GQA 代价≈1% bpc、不免费」与「同字节跨家族 Pareto」→ **(b)** 本地增量，
  保留 `judged`（本仓库 `R4` 的 `PRIOR_ART.md` 已详述，见 §4）。

**可复现的判据建议**：横轴统一为**解析 KV 字节/位置**；纵轴 = clean bpc + 合成长程检索召回
（`tools/evalkit/retention.py` / `longrange_tasks.py`）；**禁止**用「头数/比特数/压缩率」直接互比
（§17.12）；**必须**给等参数对照（代价混着容量差）。

---

### W4 — 读出几何（tied / 解绑 / gated / 低秩读出）

**谁做过 / 为什么（`记忆·未核`）**

- **权重绑定**：`Press & Wolf 2017`、`Inan et al. 2017`（Tying Word Vectors and Word
  Classifiers）——输入 embedding 与输出层共享权重，**省参数**，质量上通常中性或略优。
  解绑（独立 unembed）在大词表/小模型上有时更好，但**非标准增益**。
- **门控线性单元**：`GLU / SwiGLU`（Shazeer 2020、Noam Shazeer GLU variants）用作 FFN/读出，
  是表达力增强的通用手段——**但它是重参数化 + 容量，未必是「非线性本身」的贡献**。
- **我们的读出重参数化**（`h + P·(Qh)`）与 GLU 同族；业界**少见「为解耦计算空间与读出可见
  空间」而专门改读出几何**的工作。

**我们的增量**：**等参数纯线性对照**证明「增益归读出重参数化/容量、不归非线性」（§15）；
且**不可迁移到普通 dense**（dense 上模型主动不用该自由度）——**补本形态读出缺陷而非通用改进**。

**「业界是否已有此结论」判定（`W4` 是 `status: judged`）**：
- 「tied vs untied 读出：小模型上差异小、绑定省参数」→ **业界已有**（`记忆·未核`）⇒
  该子条可标「业界结论」口径。
- 「读出重参数化增益=容量非非线性、且只在本形态有效」→ **(b)** 本地受控新结论，保留 `judged`。

**可复现的判据建议**：判据 = Δclean bpc（税）+ workspace 探针 + ECE + 传输/提交是否仍可用；
**必须**给**等参数纯线性对照**（否则把重参数化记成非线性）；**必须**报收敛对账（§16）。

---

### W5 — 容量-计算分离

**谁做过 / 为什么（`记忆·未核`）**

- 概念框架：**「容量削减」（剪枝/减宽/裁头）vs「保容量减计算」（稀疏激活/专家/条件计算）**。
  业界共识：**要买灵活性必须走保容量减计算**；减容量不买每 token 的固定成本结构。

**我们的增量**：本仓库把这条共识**形式化为判据**（W1 负结果 + W4 对照）。

**「业界是否已有此结论」判定（`W5` 是 `status: judged`）**：**(a) 业界已研究透**——
这是剪枝 vs MoE 的基本分类（`记忆·未核`）⇒ **建议标「业界结论」口径**，
来源类型 = 业界结论，非本地实测。应写入句：
**「减容量（剪枝/减宽/裁头）付质量而不买灵活性；保容量减计算（稀疏激活/专家路由）才是
扩容量的正路（业界共识，`记忆·未核`）；本地 W1 负结果与之一致（`本地实测`）。」**

**可复现的判据建议**：同 FLOPs 的质量 + 同参数量的 FLOPs。

---

## 2. 机器可读分诊表

| grid_id | 判定 | 一句话理由 | 拟写进 `catalog.yaml` 的结论 |
|---|---|---|---|
| `W1-active-width` | (a)+b | 减宽=减容量付质量=结构化剪枝已知；能量迁出机制=本地增量 | 参照线句：「减宽/裁通道等价于减容量，需付质量；它不买每 token 固定成本结构的灵活性（`记忆·未核`，结构化剪枝一系）。」能量迁出机制保留 `judged` |
| `W2-expertization` | (a) | MoE 稀疏宽=保容量减计算，大规模成熟验证；13M 无实质新设定 | 参照线句：「用稀疏路由把参数量与每 token FLOPs 解耦（MoE）是保容量减计算的成熟路线；细粒度专家 + 共享专家为主流配方，负载均衡/路由坍缩是主要风险（`记忆·未核`，Switch/Mixtral/DeepSeek-MoE）。」⇒ 建议 `status: todo` |
| `W3-kv-head-sharing` | (a)+b | GQA/MQA 降 KV、近无损=业界已知；从零训练 13M 的代价 + 同字节跨家族 Pareto=本地增量 | 参照线句：「GQA/MQA/MLA 可把 KV 降数倍且质量近无损（uptrain；`记忆·未核`，Shazeer 2019 / Ainslie 2023 / DeepSeek-V2 2024）。」同字节 Pareto 与 13M 代价保留 `judged` |
| `W4-readout-geometry` | (a)+b | tied vs untied 小模型差异小=业界已知；读出重参数化=容量非非线性、只本形态有效=本地增量 | 参照线句：「输入/输出 embedding 绑定省参数、小模型质量差异小；解绑/门控读出属重参数化+容量手段（`记忆·未核`）。」本地受控归因保留 `judged` |

`W5-capacity-vs-compute` 已删；其保留内容「剪枝=减容量、MoE=保容量减计算」作为 `W2-expertization` 的分类事实。

---

## 3. 若可检索应查的关键词

| 格 | 关键词 |
|---|---|
| W1 | `structured pruning LLM-Pruner`, `Sheared LLaMA`, `Minitron width pruning`, `MatFormer elastic width`, `width dropout transformer`, `structured vs unstructured pruning quality` |
| W2 | `Switch Transformer MoE`, `GShard`, `Mixtral sparse MoE`, `DeepSeekMoE fine-grained shared expert`, `expert choice routing`, `load balancing aux loss MoE`, `MoE small scale 13M` |
| W3 | `Shazeer MQA 2019`, `Ainslie GQA uptraining`, `DeepSeek-V2 MLA`, `Cross-Layer Attention`, `YOCO you only cache once`, `MLKV multi-layer KV`, `unified KV byte budget benchmark` |
| W4 | `Press Wolf tying word vectors`, `Inan tying word vectors classifiers`, `Shazeer GLU variants`, `untied output embedding small model`, `readout reparameterization LM`, `weight tying quality` |

**核证优先级**：`W3 GQA uptrain 近无损数字`（决定 W3 措辞）与
`W2 MoE 在极小尺度的意义`（决定 W2 是否真需 (b)）。

---

## 4. 来源清单

**来源类型图例**：`官网/论文（含链接）`｜`记忆（未核）`｜`本地实测`。
（本波无检索工具 ⇒ **无「官网/论文（含链接）」**；外部条目均为 `记忆·未核`。）

**本地实测（路径见 `catalog.yaml` `W1`–`W4` 的 `local.evidence`）
- `W1-active-width`：结构切分读出只读前 360 维 @100M → clean 2.1475、恢复 −50.4%；
  切片能量 0.85→0.026→≈0。证据 `.scratch/25-tax-attribution`。
- `W2-expertization`：本地零经验（原 `R6-moe-scale` 已并入本格）。
- `W3-kv-head-sharing`：mha 1.4820 / gqa 1.4963（+0.0143）/ mqa 1.4938（+0.0118）；
  解析 KV 12288→3072（−75%）/1536 B（−87.5%）。证据 `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/analysis.md`。
- `W4-readout-geometry`：gated 12k Δ=−0.0812、等参数纯线性 Δ=−0.0888（不可排序）；
  24k Δ=−0.0463；dense 上 Δ=−0.0055/−0.0026。证据 `axes/W-width/W4-readout-geometry/experiments/2610~2613-*/analysis.md`。

**关联（他轴但同主题，已落盘）**
- `axes/R-references/R4-window-kv-quant-retrieval/experiments/2617-prior-art-hybrid-kv/PRIOR_ART.md`
  — 已详述 GQA/MQA/MLA 与「统一 KV 字节横轴」业界缺口；本文 W3 不重复展开。

**记忆·未核（仅作设计方向，不得当判据）**
- W1：LLM-Pruner、Sheared LLaMA、Minitron、MatFormer、width dropout / stochastic depth。
- W2：Switch Transformer、GShard、Mixtral、DeepSeek-MoE、expert-choice、aux-loss 均衡。
- W3：Shazeer MQA 2019、Ainslie GQA 2023、DeepSeek-V2 MLA 2024、CLA、YOCO、MLKV。
- W4：Press & Wolf 2017、Inan 2017、Shazeer GLU/SwiGLU 2020。
- W5：条件计算 / 容量-计算框架。

**不确定 / 未找到（防误当已知）**
1. 结构化剪枝在 **13M 字符级 LM** 上的具体掉点数字：`数字未核`。
2. 「读出重参数化」为解耦读出几何的公开工作：`未找到`（可能与 GLU 系混同）。
3. MoE 在 **13M / 单卡合成任务**上是否有意义的公开对照：`未找到`。
4. GQA/MQA 从**零训练小模型**（非 uptrain）的代价：`不确定` 是否有同设定公开数。

**诚实清单**：本文无任何训练/评测，无 `results.json`，**只是情报**；外部内容均未在线核证。
