# PRIOR_ART — R 参照线（R1–R7）

> 轴：**R 参照线（必须打败的基线）**（横切） ｜ 格子：`R1`–`R7`（**七格均 `status: todo`**）
> 角色：知识波（无 GPU，不取锁） ｜ 依据：`SCHEMA.md` §18 / §21
> **R 轴是「业界结论」口径：本文件直接给「业界已研究透」的结论条目**，
> 供账本以 `status: todo` + 「业界结论，非本地实测」写入。
> 本文件只写情报与分诊，**不是实验证据**。

---

## 0. 检索条件声明（必读）

**本环境未注册任何 web / 检索工具** ⇒ **无法在线核证任何外部数字**。遵守 §18/§21：
记忆类一律 **`记忆·未核`**、只作设计方向；本地数标 **`本地实测`** + 路径；不编数字。
R 轴条目均为**业界结论（记忆·未核）**——写入账本时**必须**标「业界结论，非本地实测」。

术语：`量化`（用更低比特表示权重/KV）、`KV`（注意力缓存）、`蒸馏`（用大模型教小模型）、
`剪枝`（删参数/结构）、`MoE`（专家混合：多个专家、每 token 只激活少数）、
`verifier`（给候选答案打分的判别器）、`early exit`（早退：简单样本提前出层）。

---

## 1. 结论条目（R1–R7，写入账本用）

> 每条格式：**结论句 | 来源类型 | 对我们的含义**。

### R1 量化

- **结论**：权重/激活/KV 的 int8 量化通常近无损、int4 有明显质量代价；KV 量化可近乎
  无损地把每位置 KV 字节减半。工业界标准做法，「Pro 当 Flash 用」的最强现成基线。
  **来源类型：业界结论（记忆·未核）**（GPTQ / AWQ / SmoothQuant / LLM.int8() / KIVI / KVQuant 等）。
- **本地复核（`本地实测`）**：int8 KV Δbpc +0.00015（近无损，12288→6144 字节）；int4 KV
  +0.0457（3072 字节）。来源：`axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/analysis.md`。
- **对我们的含义**：**任何「省内存/丢信息」的主张，至少要打败 int8 KV 的近无损 −50%**；
  int4 代价是另一条可比价签。

### R2 蒸馏

- **结论**：用大模型教小模型（logit 蒸馏 / 中间层蒸馏 / 数据蒸馏）是小模型的标准来源，
  常显著优于同尺寸从头训练。**来源类型：业界结论（记忆·未核）**（Hinton et al., 2015；
  DistilBERT；TinyBERT；MiniLLM 等）。
- **对我们的含义**：与本项目「同参数等智商」命题**直接竞争**；创新线必须先说明相对
  蒸馏小模型的增量。

### R3 剪枝 / 结构化稀疏

- **结论**：非结构化剪枝+重训可达高稀疏；**结构化剪枝（层/头/通道）收益强依赖任务**，
  且通常需重训；「减结构 = 减容量」是主因。**来源类型：业界结论（记忆·未核）**
  （SparseGPT、Wanda、movement pruning、structured pruning、lottery ticket 等）。
- **本地侧证（`本地实测`）**：结构化约束（只读前 360 维）→ 恶化 −50.4%，模型绕开被约束
  子空间。来源：`.scratch/25-tax-attribution/evidence`。
- **对我们的含义**：与 W5「容量-计算」分类直接相关；减宽/减结构要付容量代价。

### R4 窗口注意力 + KV 量化 + 检索

- **结论**：现成长上下文组合拳 = 滑动窗口注意力（+sink）+ KV 量化 + 检索增强（RAG）；
  是生产系统默认答案，在同等内存下质量可接受。**来源类型：业界结论（记忆·未核）**
  （Longformer、Mistral 滑窗、StreamingLLM、RAG/Lewis 2020、Memorizing Transformer）。
- **本地复核（`本地实测`）**：窗口部分 +0.058~0.121 bpc（13M）；「就用个窗口」的 bpc 代价
  几乎为零 ⇒ 压缩/摘要线必须证明「保留」而非「省钱」。来源：`.scratch/07-x1-x3/evidence`。
- **对我们的含义**：**右移压缩必须打败的对手**；关键判据线 = 同内存预算下的长上下文质量 +
  检索召回。

### R5 best-of-n + verifier

- **结论**：采样 n 个候选 + verifier/自一致性重排是同 FLOPs 提升质量的标准 test-time
  compute 手段。**来源类型：业界结论（记忆·未核）**（Cobbe et al., 2021 verifiers；
  self-consistency；PRM / Lightman et al., 2023）。
- **本地侧证（`本地实测`）**：共享主干的多线增益可忽略（G5）。来源：`.scratch/08-tts-probe/evidence`。
- **对我们的含义**：与 G5 多线机制直接竞争；判据 = 同 FLOPs 质量 + verifier 校准。

### R6 MoE 规模轴

- **结论**：MoE 在同等 FLOPs 下用更多参数提升质量，是**唯一被大规模验证**的
  「容量-计算分离」路线。**来源类型：业界结论（记忆·未核）**（Shazeer 2017 / Switch /
  GShard / Mixtral / DeepSeek-MoE / ST-MoE）。
- **对我们的含义**：与 W2 同格；作为参照线登记其判据线（同 FLOPs 质量）。

### R7 既有早退实现（CALM / LayerSkip / MoD）

- **结论**：三者均有公开结果；共同点 = **需要深度维监督**、都报告质量-算力折中。
  **来源类型：业界结论（记忆·未核）**（CALM: Schuster 2022；LayerSkip: Meta 2024；
  MoD: Raposo 2024）。
- **本地对照（`本地实测`）**：dense 全深度监督（LayerSkip 式）+0.067（13M）。来源：
  `.scratch/18-exit-probe/evidence`、`.scratch/21-exit-at-scale/evidence`。
- **对我们的含义**：我们的形态必须说清增量——逐通道 carry 门 + 随机退出深度 + 定稿不变量
  （全深度重打分）。

---

## 2. 机器可读分诊表（§21）

```
R1-quantization            | (a) | 权重/激活/KV 量化是工业界标准做法、结论一致（int8 近无损/int4 有损） | 结论：int8 量化近无损、int4 有明显代价；省内存主张须打败 int8 KV −50%。来源类型：业界结论（记忆·未核）+ 本地复核。status: todo。
R2-distillation            | (a) | 蒸馏是小模型的标准来源，业界成熟 | 结论：大模型蒸馏是「同参数等智商」的直接竞争者。来源类型：业界结论（记忆·未核）。status: todo。
R3-pruning-sparsity        | (a) | 剪枝（尤其结构化）收益依赖任务、需重训，业界成熟 | 结论：减结构=减容量，收益强依赖任务。来源类型：业界结论（记忆·未核）+ 本地侧证。status: todo。
R4-window-kv-quant-retrieval | (a) | 窗口+KV量化+检索是生产系统默认组合拳 | 结论：长上下文现成答案是滑窗(+sink)+KV量化+RAG；右移压缩必须打败它。来源类型：业界结论（记忆·未核）+ 本地侧证。status: todo。
R5-bestofn-verifier        | (a) | best-of-n + verifier 是同 FLOPs 标准 test-time compute 手段 | 结论：采样+重排是标准手段，与 G5 多线竞争。来源类型：业界结论（记忆·未核）。status: todo。
R6-moe-scale               | (a) | MoE 是同 FLOPs 更多参数的唯一大规模验证路线 | 结论：MoE 实现容量-计算分离。来源类型：业界结论（记忆·未核）。status: todo。
R7-existing-early-exit     | (a) | CALM/LayerSkip/MoD 均有公开结果，共同需深度监督 | 结论：早退实现需深度维监督、报告质量-算力折中。来源类型：业界结论（记忆·未核）+ 本地对照。status: todo。
```

**分诊要点**：R 轴**七格全部 (a)**——这正是「业界结论」类的定义。主 Agent 可直接
把上表「若要写进账本的结论」列以 `status: todo` + 「业界结论，非本地实测」写进
`catalog.yaml`（本地复核数保留在 `local` 块）。

---

## 3. 若可检索应查的关键词

| 格 | 关键词 |
|---|---|
| R1 | `GPTQ`, `AWQ`, `SmoothQuant`, `LLM.int8()`, `KIVI KV quantization`, `KVQuant`, `int4 KV cache perplexity` |
| R2 | `knowledge distillation Hinton`, `DistilBERT`, `TinyBERT`, `MiniLLM reverse KL`, `sequence-level KD` |
| R3 | `SparseGPT`, `Wanda pruning`, `movement pruning`, `structured pruning transformers`, `lottery ticket hypothesis` |
| R4 | `Longformer sliding window`, `Mistral sliding window attention`, `StreamingLLM`, `retrieval augmented generation Lewis`, `Memorizing Transformer` |
| R5 | `Cobbe verifiers`, `self-consistency`, `process reward model PRM Lightman`, `best-of-n reranking` |
| R6 | `Switch Transformer`, `GShard`, `Mixtral of experts`, `DeepSeek-MoE`, `ST-MoE` |
| R7 | `CALM early exit Schuster`, `LayerSkip Meta`, `Mixture-of-Depths Raposo`, `self-speculative decoding` |

**核证优先级**：`KIVI/KVQuant 的 int4 KV 质量代价`（R1 价签基准）与
`LayerSkip/CALM 的同尺寸对照`（R7，判断本地形态增量是否成立）。

---

## 4. 来源清单

**本地实测（可当证据）**
- `axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/analysis.md` — int8/int4 KV 价签。
- `.scratch/07-x1-x3/evidence` — 窗口部分 +0.058~0.121。
- `.scratch/25-tax-attribution/evidence` — 结构化约束 −50.4%。
- `.scratch/08-tts-probe/evidence` — 多线增益可忽略。
- `.scratch/18-exit-probe/evidence`、`.scratch/21-exit-at-scale/evidence` — 早退对照。
- `axes/R-references/R4-window-kv-quant-retrieval/experiments/2617-prior-art-hybrid-kv/PRIOR_ART.md` — 混合层比例/GQA·MLA 已有情报。

**记忆（未核，只可作设计方向）**
- R1–R7 全部论文/系统名见 §3 关键词。

**不确定 / 未找到**
- 各工作的**确切数字、对照基线强度**。
- R6 在 13M 尺度是否有公开对照（规模远小于 MoE 典型规模）。
