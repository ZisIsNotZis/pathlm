# PRIOR_ART — R 参照线（R1/R4/R7）

> 轴：**R 参照线（必须打败的基线）**（横切） ｜ 格子：`R1`、`R4`、`R7`
> （`R2`/`R3`/`R5`/`R6` 已从研究树删除，见 `archive/README.md`）
> 角色：知识波（无 GPU，不取锁） ｜ 依据：`SCHEMA.md` §18 / §21
> **R 轴是「业界结论」口径：本文件直接给「业界已研究透」的结论条目**，
> 供 `catalog.yaml` 以 `status: todo` + 「业界结论，非本地实测」写入。
> 本文件只写情报与分诊，**不是实验证据**。

---

## 0. 检索条件声明（必读）

**本环境未注册任何 web / 检索工具** ⇒ **无法在线核证任何外部数字**。遵守 §18/§21：
记忆类一律 **`记忆·未核`**、只作设计方向；本地数标 **`本地实测`** + 路径；不编数字。
R 轴条目均为**业界结论（记忆·未核）**——写入 `catalog.yaml` 时**必须**标「业界结论，非本地实测」。

术语：`量化`（用更低比特表示权重/KV）、`KV`（注意力缓存）、`蒸馏`（用大模型教小模型）、
`剪枝`（删参数/结构）、`MoE`（专家混合：多个专家、每 token 只激活少数）、
`verifier`（给候选答案打分的判别器）、`early exit`（早退：简单样本提前出层）。

---

## 1. 结论条目（R1/R4/R7，写进 `catalog.yaml` 用）

> 每条格式：**结论句 | 来源类型 | 对我们的含义**。

### R1 量化

- **结论**：权重/激活/KV 的 int8 量化通常近无损、int4 有明显质量代价；KV 量化可近乎
  无损地把每位置 KV 字节减半。工业界标准做法，「Pro 当 Flash 用」的最强现成基线。
  **来源类型：业界结论（记忆·未核）**（GPTQ / AWQ / SmoothQuant / LLM.int8() / KIVI / KVQuant 等）。
- **本地复核（`本地实测`）**：int8 KV Δbpc +0.00015（近无损，12288→6144 字节）；int4 KV
  +0.0457（3072 字节）。来源：`axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/analysis.md`。
- **对我们的含义**：**任何「省内存/丢信息」的主张，至少要打败 int8 KV 的近无损 −50%**；
  int4 代价是另一条可比价签。

### R2 / R3 / R5 / R6（已从研究树删除）

`R2-distillation`、`R3-pruning-sparsity`、`R5-bestofn-verifier`、`R6-moe-scale` 已按
`.scratch/51-tree/OPERATIONS.md` 删除（无本地增量、无基线角色、与 `W2` 重复）；
其保留内容「剪枝=减容量、MoE=保容量减计算」归 `W2-expertization`。
目录见 `archive/dropped/axes/R-references/`。

### R4 窗口注意力 + KV 量化 + 检索

- **结论**：现成长上下文组合拳 = 滑动窗口注意力（+sink）+ KV 量化 + 检索增强（RAG）；
  是生产系统默认答案，在同等内存下质量可接受。**来源类型：业界结论（记忆·未核）**
  （Longformer、Mistral 滑窗、StreamingLLM、RAG/Lewis 2020、Memorizing Transformer）。
- **本地复核（`本地实测`）**：窗口部分 +0.058~0.121 bpc（13M）；「就用个窗口」的 bpc 代价
  几乎为零 ⇒ 压缩/摘要线必须证明「保留」而非「省钱」。来源：`.scratch/07-x1-x3/evidence`。
- **对我们的含义**：**右移压缩必须打败的对手**；关键判据线 = 同内存预算下的长上下文质量 +
  检索召回。

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
R4-window-kv-quant-retrieval | (a) | 窗口+KV量化+检索是生产系统默认组合拳 | 结论：长上下文现成答案是滑窗(+sink)+KV量化+RAG；右移压缩必须打败它。来源类型：业界结论（记忆·未核）+ 本地侧证。status: todo。
R7-existing-early-exit     | (a) | CALM/LayerSkip/MoD 均有公开结果，共同需深度监督 | 结论：早退实现需深度维监督、报告质量-算力折中。来源类型：业界结论（记忆·未核）+ 本地对照。status: todo。
```

**分诊要点**：R 轴**七格全部 (a)**——这正是「业界结论」类的定义。主 Agent 可直接
把上表「拟写进 `catalog.yaml` 的结论」列以 `status: todo` + 「业界结论，非本地实测」写进
`catalog.yaml`（本地复核数保留在 `local` 块）。

---

## 3. 若可检索应查的关键词

| 格 | 关键词 |
|---|---|
| R1 | `GPTQ`, `AWQ`, `SmoothQuant`, `LLM.int8()`, `KIVI KV quantization`, `KVQuant`, `int4 KV cache perplexity` |
| R4 | `Longformer sliding window`, `Mistral sliding window attention`, `StreamingLLM`, `retrieval augmented generation Lewis`, `Memorizing Transformer` |
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
- R1/R4/R7 全部论文/系统名见 §3 关键词。

**不确定 / 未找到**
- 各工作的**确切数字、对照基线强度**。
