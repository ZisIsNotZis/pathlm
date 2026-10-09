# PRIOR_ART — 迭代（重复对象 / 回传通道 / 停止条件 / 训练信号）的既有做法

> 轴：**F 迭代** ｜ 格子：`F2-return-channel` · `F4-training-signal`（`F1-iteration-object` 并入 `F2`、`F3-stopping` 并入 `G1-selector`，见 `archive/README.md`）
> 角色：researcher（只读 + 本文件）｜ 本波**无 GPU、不跑训练/评测、不取锁**
> 依据：`SCHEMA.md` §18 / §19 / §21

---

## 0. 检索条件声明（**必读**）

**本环境联网检索不可用**。外部内容**无法在线核证**，一律标 **`记忆·未核`**，只作方向、
**不得当判据**。本地数标 `本地实测` + 路径。不编数字。
术语就地解释：**latent retry**（把同一 token 的隐状态重跑一遍）；
**回传通道**（上一轮结果以什么形式回到下一轮）；**税**（相对普通 transformer 的质量代价）；
**bpc**（每字符比特数，越低越好）；**一致性损失**（consistency loss：逼 `T₂ ≈ T₁∘T₁`）；
**充分性/重建损失**（sufficiency/reconstruction：只给摘要要求重建原文）；
**TTT**（test-time training，把权重本身当迭代对象）；**latent CoT / pause token**
（无输出监督的隐式推理步）。

**若可检索应查的关键词清单**见 §3。

---

## 1. 既有做法（按格）

### 迭代对象（原 F1，已并入 F2-return-channel）

**谁做过 / 为什么（`记忆·未核`）**

- **同状态多遍（looped / recurrent depth）**：`Universal Transformer`（Dehghani 2018，
  共享权重 + ACT 自适应步数）、`recurrent depth`（Geiping et al. 2024）、
  `Huginn` 一类**隐式推理步**——共享层反复施加在状态上。**已知教训**：
  权重绑定的循环若**不做输入注入**，容易**退化为固定点**——这与我们本地「direct 重入 =
  结构性 no-op」直接对应。
- **整段去噪（扩散 LM）**：`Diffusion-LM`、`SEDD`、`Plaid` 等——整段并行去噪。
- **候选集（best-of-n）**：通用解码。
- **隐式推理步**：`pause token`（Goyal et al. 2024）、latent CoT。
- **权重迭代**：`fast weight`（Schmidhuber 1992；Schlag 2021）、`TTT`（Sun et al. 2024）。

**我们的增量**：13M / char-enwik8 上**收口了前两项**（latent retry 与 AR 多遍），
并给出**「多一遍有价值、回传几何无价值」**的受控结论（6 个 latent 回传 cell 跨度 0.011
< 种子散布 0.0138 ⇒ 排序撤回）。后四项（整段去噪 / 候选集 / latent CoT / TTT）本地零经验。

**「业界是否已有此结论」判定（迭代对象已并入 `F2`）**：
- 「无输入注入的循环重入退化为固定点」→ **业界已有**（looped/recurrent 模型的公认教训，
  `记忆·未核`）⇒ 该子条可标「业界结论」口径（来源类型 = 业界结论，非本地实测）。
- 「回传几何（linear/soft/hard cell）在 13M 上不可排序」→ **(b)** 本地受控结论，保留。
- 整段去噪 / latent CoT / TTT → **(a) 业界已研究**（各自独立成线，`记忆·未核`）；
  我们的增量只能是**13M + 局部先验抗性任务 + 单因素受控**（§19/§21）。若做，判 **(b)**。

**可复现的判据建议**：判据 = 税 + **每轮边际收益曲线** + 成本（forwards/FLOPs）；
**必须**做「无传输 vs 有传输」的**结构性 no-op 单测**；合成任务复核（§19）。

---

### F2 — 回传通道

**谁做过 / 为什么（`记忆·未核`）**

- **输入注入 / 门控循环**：Universal Transformer 每步把输入加回状态（避免固定点）；
  `DEQ`（深度均衡模型）与门控循环是「通道设计」的近亲。动机：让重复**带来新信息**。
- **软 vs 硬（量化）回传**：VQ / 离散潜变量的**软提交凸包安全**是通用结论；
  软提交外插到训练分布外（OOD）是已知风险。
- `不确定`：是否有工作系统比较「direct/linear/soft/hard/mask/verifier 反馈」这一组通道的税。

**我们的增量**：同引擎单因素扫描回传通道，得到**「direct 重入 = 固定点（结构性 no-op）」
「软回传 OOD 污染三度复现」「D1 gauge bug（~30× rescale）」**等本地结论。

**「业界是否已有此结论」判定（`F2` 是 `status: judged`）**：
- 「无传输的重入无价值 / 需要输入注入」→ **业界已有**（looped 模型教训，`记忆·未核`）⇒
  该子条可标「业界结论」口径。应写入句：**「循环/迭代模型必须让每轮有新信息（输入注入或
  门控），否则退化为固定点（`记忆·未核`，Universal Transformer 系）；本地 13M 实测
  direct 重入 = 结构性 no-op（`本地实测`）。」**
- 「软回传 OOD 污染」→ **(b)** 本地复现 + 具体量级，保留 `judged`。
- 通道组的系统单因素对照 → **(b)**（未找到直接前例）。

**可复现的判据建议**：判据 = 每轮 bpc 边际 + ECE + 是否结构性 no-op；
**必须**防 gauge/尺度 bug（本地 D1 教训）；离散回传**必须门控**。

---

### 停止条件（原 F3，已并入 G1-selector）

**谁做过 / 为什么（`记忆·未核`）**

- **自适应计算**：`Adaptive Computation Time`（Graves 2016）、`PonderNet`——按需要分配步数。
- **早退 / 置信门控**：`BranchyNet`、`PABEE`、`CALM`（Confident Adaptive Language Modeling，
  Schuster et al. 2022）——**用模型自身置信度决定是否继续算**。动机：省算力、按需。
  ⇒ 「按置信选择性开火」是**行业成熟概念**。已知风险：**生成段的置信信号不可靠**（校准漂移）。
- **预算求解器（多货币）**：把「质量-成本前沿」当带约束的分配问题——**少见端到端公开工作**。
- **收敛判据（Δ 小）**：通用。

**我们的增量**：**选择性问题获本地证**（低置信 15% 开火 +0.128 vs 均匀 +0.057，
固定点最优性复现）+ **预算求解器端到端建成**（成本模型预测误差 ≤±13%）。

**「业界是否已有此结论」判定（停止条件已并入 `G1-selector`，在本格预登记重跑）**：
- 「按置信度选择性计算/早退能省钱」→ **业界已有**（ACT / CALM 系，`记忆·未核`）⇒
  该子条可标「业界结论」口径（来源类型 = 业界结论，非本地实测）。
- 「门控选择性的质量杠杆在 100M 消失」与「多货币预算求解器」→ **(b)** 本地增量，保留 `judged`。

**可复现的判据建议**：判据 = 质量-成本前沿 + 门控选择性（按置信分桶的每 FLOP 收益）；
**必须**给门控的活性量（开火率分布）+ 自由生成段的置信失效检查（本地已见 G3 问题）。

---

### F4 — 迭代/重复的训练信号

**谁做过 / 为什么（`记忆·未核`）**

- **重建 / 充分性类损失**：`Compressive Transformer`（Rae et al. 2019，**压缩损失**：
  从压缩记忆重建原文）、`AutoCompressors`（Chevalier 2023，重建 summary）、
  `ICAE`（Ge 2024，自编码 slot）、`RECOMP`（Xu 2023，重建 vs 蒸馏）、`500xCompressor`、
  `DMC`（2024，KV 压缩需配继续预训练）。⇒ **这是一条 2019 年就开线、2023–24 又热起来的
  成熟路线，不是空白格**（详见 `axes/M-carrier/.../2617-prior-art-memory/PRIOR_ART.md` §3）。
- **一致性损失 / 自蒸馏**：`Born-Again Networks` 一类自蒸馏；「consistency training」在
  半监督/序列模型里是通用手段。`不确定`：是否有工作把 `T₂ ≈ T₁∘T₁` 一致性用于多步组合。
- **稳定性**：奖励/负类损失在 **bf16 autocast** 下易发散——通用工程教训
  （我们本地两次发散 → 必须 float32 + pre-step grad-norm gate）。

**我们的增量**：13M + **局部先验抗性任务**（§19）+ **单因素受控** + **源对照臂**。
**不得**再声称「无人占位」。

**「业界是否已有此结论」判定（`F4` 是 `status: judged`）**：
- 「重建/充分性损失是激进压缩能成立的关键信号」→ **业界已有成熟前例**（见上，`记忆·未核`）⇒
  **该条改「业界结论」口径**（来源类型 = 业界结论，非本地实测）。应写入句：
  **「重建/充分性类辅助损失是压缩可恢复性的关键信号（`记忆·未核`，Compressive Transformer
  2019 / AutoCompressors / ICAE / RECOMP / DMC 2024）；本地 F4 是 13M + 局部先验抗性任务 +
  单因素受控的增量检验，非首创。」**
- 「consistency loss 在真实规模中性（不改善聚合质量）」→ **(b)** 本地受控结论，保留 `judged`
  （业界多数压缩工作不报可重建性 ⇒ 带检索腿的判据仍是我们的增量）。

**可复现的判据建议**：两臂 A/B（纯 CE vs CE+λ·重建），**同 seed 同数据**；
判据**落在保留腿**（窗口外针的**可重建性/检索召回**），**不是 bpc**；
**必须** `λ=0 逐位等价`单测（§9）+ float32 + pre-step grad-norm gate（§10）；
报活性量（重建损失轨迹、槽位使用率）+ 反噬检查（主任务 bpc）。

---

## 2. 机器可读分诊表

| grid_id | 判定 | 一句话理由 | 拟写进 `catalog.yaml` 的结论 |
|---|---|---|---|
| `F2-return-channel` | (a)+b | 需输入注入才不 no-op=业界教训；软回传 OOD 污染与通道组对照=本地增量 | 参照线句：「循环式重复若无输入注入会退化为固定点；隐式推理步/扩散去噪/权重迭代均为既有路线（`记忆·未核`）。」回传几何不可排序保留 `judged`/`todo` |
| `F4-training-signal` | (a)+b | 重建/充分性损失=2019 起成熟路线（非空白格）；一致性中性=本地增量 | 参照线句：「重建/充分性类辅助损失是压缩可恢复性的关键信号（`记忆·未核`，Compressive Transformer 2019 / AutoCompressors / ICAE / RECOMP / DMC 2024）。」⇒ 措辞改「业界结论」，本地增量保留 `judged` |

---

## 3. 若可检索应查的关键词

| 格 | 关键词 |
|---|---|
| 迭代对象（原 F1→F2） | `Universal Transformer adaptive computation`, `recurrent depth Huginn input injection`, `looped transformer fixed point collapse`, `Diffusion-LM SEDD Plaid`, `pause token Goyal 2024`, `latent chain of thought`, `test-time training TTT Sun 2024`, `fast weights Schmidhuber` |
| F2 | `deep equilibrium model input injection`, `gated recurrence state update`, `soft vs hard latent quantization OOD`, `vector quantization straight-through`, `verifier feedback loop LM` |
| 停止条件（原 F3→G1） | `adaptive computation time Graves 2016`, `PonderNet`, `PABEE early exit`, `CALM confident adaptive language modeling`, `early exit calibration generation`, `compute allocation budget solver transformer` |
| F4 | `Compressive Transformer compression loss reconstruction`, `AutoCompressors summary reconstruction`, `ICAE in-context autoencoder`, `RECOMP reconstruction vs distillation`, `500xCompressor`, `DMC dynamic memory compression`, `consistency loss self-distillation`, `sufficiency loss memory` |

**核证优先级**：`F4 Compressive Transformer compression loss`（决定 F4 措辞）与
`迭代对象 looped model 固定点/输入注入`（决定 `F2` 参照线措辞）。

---

## 4. 来源清单

**来源类型图例**：`官网/论文（含链接）`｜`记忆（未核）`｜`本地实测`。
（本波无检索工具 ⇒ **无「官网/论文（含链接）」**；外部条目均为 `记忆·未核`。）

**本地实测（路径见 `catalog.yaml` `F2`/`F4` 的 `local.evidence`）**
- `F2-return-channel`（并 `F1`）：latent retry C 族 1.6535–1.6856 vs B0 1.5074（+0.146…+0.178）；
  6 个 latent 回传 cell 跨度仅 0.011 < 种子散布 0.0138（排序撤回）；direct 重入 = 固定点（结构性 no-op）。证据
  `.scratch/11-retry-matrix-validity/VERDICTS.md`、`.scratch/12-overnight`、`.scratch/14-allocator`。
- `G1-selector`（并 `F3`）：低置信 15% 开火 +0.0211 vs 均匀 −0.0193；C1M +0.1284 vs +0.0570；
  成本模型预测误差 ≤±13%。证据 `.scratch/14-allocator`、`.scratch/15-slider-rung3`、
  `.scratch/22-gen-quality`。
- `F4-training-signal`：consistency loss 中性（IX4 1.6589 ≈ C1 1.6549）；bf16 下 reward 类
  损失两次发散。证据 `.scratch/02-engine-m0`、`.scratch/04-m1-runs`。

**关联（他轴但同主题，已落盘）**
- `axes/M-carrier/M1-carrier-type/experiments/2617-prior-art-memory/PRIOR_ART.md` §3
  — 已详述重建/充分性类损失的前例；本文 F4 不重复展开，只引用其结论与措辞修正。

**记忆·未核（仅作设计方向，不得当判据）**
- 迭代对象（原 F1）：Universal Transformer（Dehghani 2018）、recurrent depth / Huginn（Geiping 2024）、
  Diffusion-LM / SEDD / Plaid、pause token（Goyal 2024）、fast weights（Schmidhuber 1992 /
  Schlag 2021）、TTT（Sun 2024）。
- F2：DEQ、gated recurrence、VQ 软提交。
- 停止条件（原 F3）：ACT（Graves 2016）、PonderNet、BranchyNet、PABEE、CALM（Schuster 2022）。
- F4：Compressive Transformer（Rae 2019）、AutoCompressors（Chevalier 2023）、ICAE（Ge 2024）、
  RECOMP（Xu 2023）、500xCompressor、DMC（Nawrot 2024）、Born-Again Networks。

**不确定 / 未找到（防误当已知）**
1. 「回传通道组」direct/linear/soft/hard/mask/verifier 的系统单因素对照：`未找到`。
2. 「一致性损失 `T₂ ≈ T₁∘T₁` 用于多步组合」的公开工作：`未找到`。
3. 多货币「质量-成本预算求解器」的端到端公开前例：`未找到`（疑为真空白，检索不可用不能断言）。
4. 各类前例的**确切数字**：`数字未核`，需查原文。

**诚实清单**：本文无任何训练/评测，无 `results.json`，**只是情报**；外部内容均未在线核证。
