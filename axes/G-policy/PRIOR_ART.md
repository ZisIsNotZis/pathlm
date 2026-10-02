# PRIOR_ART — G 策略与调度（G1–G5）

> 轴：**G 策略与调度**（横切轴） ｜ 格子：`G1-selector`、`G2-currency`、`G3-signal`、
> `G4-calibration`、`G5-parallel-lines`（**五格均 `status: judged`**）
> 角色：知识波（无 GPU，不取锁） ｜ 依据：`SCHEMA.md` §18 / §19 / §21
> 本文件只写情报与分诊，**不是实验证据**。

---

## 0. 检索条件声明（必读）

**本环境未注册任何 web / 检索工具** ⇒ **无法在线核证任何外部数字**。遵守 §18/§21：
记忆类一律 **`记忆·未核`**、只作设计方向；本地数标 **`本地实测`** + 路径；不确定标
**`不确定`**；不编数字。

术语：`ECE`（置信度校准误差，越低越好）、`FLOPs`（浮点运算量）、
`route/gate`（路由/门控：决定一个 token 走哪条路径）、`verifier`（重排候选答案的判别器）。

---

## 1. 既有做法条目（§18 四问）

### 块①：路径选择器（对标 `G1-selector`）

| 系统 | 谁做过·做了什么 | 为什么 | 实测结果 | 我们的增量 |
|---|---|---|---|---|
| **Adaptive Computation Time（ACT）**（Graves, 2016，`记忆·未核`） | 让每个 token 按 learned halt 概率决定重复步数（ponder cost 惩罚） | 算力应随难度变化 | 在算法/记忆任务上有效，但难训、加正则 | 「逐 token 变深度」的祖先 |
| **Mixture-of-Depths（MoD）**（Raposo et al., 2024，`记忆·未核`） | 逐 token 路由「跳过/走这一层」+ 容量上限 | 同 FLOPs 更多有效深度 | 报告同 FLOPs 质量不劣 | 我们 G1 的「逐 token 阈值」格子对应 |
| **CALM**（Schuster et al., 2022，`记忆·未核`） | 对比路由：小模型决定何时用大模型 | 早退省算力 | 报告质量-算力折中 | 见 R7；我们 G1 已测的对照物 |
| **LayerSkip**（Meta, 2024，`记忆·未核`） | 自推测早退，dense 深度监督 | 训练即带早退出口 | 报告折中 | R7；本地 dense 监督对照 |
| **MoE 路由**（Shazeer 2017 / Switch / GShard，`记忆·未核`） | learned top-k gate 选专家 | 容量-计算分离 | 大规模验证有效 | G1 的「学习门」族 |
| **选择式预测**（SelectiveNet、Geifman & El-Yaniv，`记忆·未核`） | 阈值 + 覆盖率-风险权衡 | 只在高置信样本上出手 | 标准框架 | G1 的「输入无关阈值」族 |

- **本地实测（G1）**：Slider 端到端（校准 fire split gap ≤0.004 → 双货币成本模型
  预测 ≤±13% → 求解器 → 验证解码 → 在线代理）；在线多线调度器是空白。
  来源：`.scratch/15-slider-rung3/evidence`、`.scratch/17-slider-at-scale/evidence`。
- **判定**：**「阈值/门控/学习路由/早退」作为选择器，业界已做透（MoD / CALM / ACT / MoE）**。
  G1 的**通用结论（选择器本身不加税、可校准）应改 `reference` 口径**。真正可能留增量的是
  **在线多线调度器**——但那与 G5 同题。

### 块②：记账货币（对标 `G2-currency`）

| 既有做法（`记忆·未核`） | 结果 |
|---|---|
| 效率文献习惯报「同 FLOPs」或「同墙钟」 | 同一机制在两口径下可翻转 |
| speculative decoding 报 wallclock / acceptance | 墙钟收益依赖 batch/硬件 |

- **本地实测（G2）**：同一机制两口径分歧实例——spec 在 forwards 口径 0.55–0.70、
  FLOPs 口径 ≥1.23；@100M 墙钟 k=2 1.15–1.29× 而 FLOPs 口径有利。
  来源：`.scratch/15-slider-rung3/evidence`、`.scratch/20-spec-scale/evidence`。
- **判定**：**「单货币记账会得出相反结论、必须双报」是效率研究的通用纪律 ⇒ 改 `reference` 口径。**
  本地数字作复核。

### 块③：观测信号（对标 `G3-signal`）

| 信号 | 既有做法（`记忆·未核`） |
|---|---|
| 模型自评置信 | Kadavath et al., 2022 `Language Models (Mostly) Know What They Know`（P(True) 自评、校准） |
| 语义熵 | `Semantic Uncertainty`（Kuhn et al., 2023）用语义簇的熵估计不确定性 |
| 一致性/幻觉检测 | `SelfCheckGPT`（Manakul et al., 2023）多次采样查一致性 |
| logit/熵类 | 序列似然、perplexity、entropy 基线 |
| 重复/塌缩 | distinct-n、degeneration 检测（历史已知） |

- **本地实测（G3）**：三信号正交监测栈——prob0 管 token 对错（ECE 0.002–0.011）；
  路径分歧 js 管语义分歧（AUC 0.66–0.69）；distinct-2 管塌缩（AUC 0.9815，与 js 正交
  rank_corr 0.14）。来源：`.scratch/22-gen-quality/evidence`。
- **判定**：**单信号分量（自评置信 / 一致性 / 重复率）业界均有成熟对应物 ⇒ 各分量改
  `reference` 口径**；本地的**「三信号正交组合 + 生成段无单一信号够用」是组合工程结论**，
  可保留本地口径，但**不得声称信号本身首创**。

### 块④：校准（对标 `G4-calibration`）

| 既有做法（`记忆·未核`） | 结果 |
|---|---|
| **Guo et al., 2017** `On Calibration of Modern Neural Networks` | 深网普遍过自信；**温度缩放**等后处理可显著降低 ECE |
| **选择性预测/拒绝**（SelectiveNet、Geifman & El-Yaniv） | 校准质量决定阈值策略有效性 |
| 训练期校准（标签平滑、mixup、focal loss 等） | 影响 ECE；`数字未核` |

- **本地实测（G4）**：prob0 ECE 0.002–0.011（M0）→ ≤0.0091（100M）；fire 曲线 split gap ≤0.004。
  来源：`.scratch/02-engine-m0/evidence/REPORT.md`、`.scratch/16-rung4-scale/evidence`。
- **判定**：**「校准可后处理改善、且跨规模存活」是既有结论（温度缩放为代表）⇒ 改
  `reference` 口径。** 本地的「prob0 校准跨规模存活」作为复核保留。

### 块⑤：并行多线（对标 `G5-parallel-lines`）

| 系统 | 既有做法（`记忆·未核`） | 结果 |
|---|---|---|
| **Self-Consistency**（Wang et al., 2022） | 多次采样 + 多数投票 | 推理任务上稳定增益 |
| **Deep Ensembles**（Lakshminarayanan et al., 2017） | 独立训练多模型求平均 | **集成增益来自成员多样性/去相关** |
| **Best-of-n + Verifier**（Cobbe et al., 2021） | 采样 n 个 + verifier 重排 | 同 FLOPs 提升；见 R5 |
| **Beam search** | 确定性搜索多条路径 | 常与集成的「多线」混淆 |

- **本地实测（G5）**：共享 trunk 的 voter 太相关 ⇒ 集成增益仅 +0.0008~+0.0015（可忽略）；
  确定性路径 checkpoint 的 K-path 集成恰得 0；去相关来自 transports/depths 而非 trunk，
  且多样性必须练进去（DIVL1）。来源：`.scratch/08-tts-probe/evidence`、`.scratch/09-m3-interactions/evidence`。
- **判定**：**「集成需要多样性、同主干成员高度相关增益可忽略」是深集成既有共识 ⇒ 改
  `reference` 口径。** 本地数字作 13M 复核。

---

## 2. 机器可读分诊表（§21）

```
G1-selector        | (a) | 阈值/门控/学习路由/早退作为选择器，业界有 MoD/ACT/CALM/MoE 成熟占位 | 结论：路径选择器本身不加质量税，且校准后阈值策略有效。来源类型：业界结论（记忆·未核）+ 本地复核。口径改为 reference。
G1-selector        | (b) | 「在线多线调度器」业界少见且依赖去相关来源（与 G5 同题） | 结论（待测）：去相关必须来自 transports/depths 而非共享 trunk。
G2-currency        | (a) | 同一机制在 forwards/FLOPs/墙钟口径下结论分歧、必须双报 | 结论：效率结论必须双货币并报，单货币会给出相反结论。来源类型：业界方法学（记忆·未核）。
G3-signal          | (a) | 自评置信/一致性/重复率各分量的既有做法成熟（P(True)/SemanticEntropy/SelfCheckGPT/distinct-n） | 结论：各信号分量为业界既有，来源类型：业界结论（记忆·未核）。
G3-signal          | (b) | 「三信号正交组合 + 生成段无单一信号够用」是组合工程结论，业界无相同组合的公开受控对照 | 结论（本地）：prob0 管对错、js 管语义分歧、distinct-2 管塌缩，三者正交。
G4-calibration     | (a) | 温度缩放等后处理可降 ECE、深网过自信，是 Guo 2017 以来的既有结论 | 结论：置信信号可校准且校准质量决定阈值策略。来源类型：业界结论（记忆·未核）+ 本地跨规模复核。口径改为 reference。
G5-parallel-lines  | (a) | 集成增益来自成员多样性/去相关（Deep Ensembles、self-consistency） | 结论：共享主干的多线因成员相关而增益可忽略；去相关必须显式训练。来源类型：业界结论（记忆·未核）+ 本地复核。口径改为 reference。
G5-parallel-lines  | (b) | 「确定性路径 checkpoint 的 K-path 集成恰得 0」是本仓库可复现的量化观察 | 结论（本地）：beam search 产出的确定性多线落在零增益死区。
```

**分诊要点**：G 轴**五格主结论全部判 (a)** ⇒ 通用机制级结论应改 `reference` 口径；
本地数字保留为 13M 复核。G1/G5 的「在线多线调度」与「去相关来源」可留 (b) 追测。

---

## 3. 若可检索应查的关键词

| 块 | 关键词 |
|---|---|
| ① 选择器 | `Mixture-of-Depths routing`, `adaptive computation time Graves ponder`, `CALM early exit`, `LayerSkip self-speculative early exit`, `learned router top-k capacity`, `selective prediction coverage risk` |
| ② 货币 | `FLOPs vs wallclock efficiency accounting`, `speculative decoding wallclock acceptance`, `matched-step FLOPs comparison` |
| ③ 信号 | `Language Models Mostly Know What They Know`, `Semantic Uncertainty Kuhn`, `SelfCheckGPT`, `distinct-n degeneration` |
| ④ 校准 | `On Calibration of Modern Neural Networks Guo`, `temperature scaling ECE`, `selective classification calibration` |
| ⑤ 多线 | `self-consistency chain of thought`, `deep ensembles Lakshminarayanan`, `ensemble diversity correlation`, `best-of-n verifier Cobbe` |

**核证优先级**：`MoD 质量-算力表`（决定 G1 是否彻底 (a)）与
`deep ensembles 多样性-增益关系`（决定 G5 的 (a) 强度）。

---

## 4. 来源清单

**本地实测**
- `.scratch/15-slider-rung3/evidence`、`.scratch/17-slider-at-scale/evidence`、`.scratch/14-allocator/evidence` — Slider。
- `.scratch/20-spec-scale/evidence`、`.scratch/25-tax-attribution/evidence` — 双货币分歧。
- `.scratch/22-gen-quality/evidence` — 三信号栈。
- `.scratch/02-engine-m0/evidence/REPORT.md`、`.scratch/16-rung4-scale/evidence` — 校准。
- `.scratch/08-tts-probe/evidence`、`.scratch/09-m3-interactions/evidence` — 多线去相关。

**记忆（未核，只可作设计方向）**
- ACT / MoD / CALM / LayerSkip / MoE 路由 / SelectiveNet / Guo 2017 温度缩放 /
  Kadavath P(True) / Semantic Uncertainty / SelfCheckGPT / distinct-n /
  Self-Consistency / Deep Ensembles / Cobbe verifier。

**不确定 / 未找到**
- MoD/CALM 在**同尺寸 13M + 单因素受控**下的确切增益数字。
- 业界是否有与我们「三信号正交组合」等价的公开监测栈。
- 在线多线调度器是否有以「去相关来源」为核心的公开工作。
