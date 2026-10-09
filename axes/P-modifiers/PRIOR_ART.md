# PRIOR_ART — P 修饰符（P1–P6）

> 轴：**P 修饰符（不是轴）**（横切，不改变路径空间） ｜ 格子：`P1-precision`（judged）、
> `P2-conditioning`（todo）、`P3-param-insertion-init`（judged）、`P4-normalization`（todo）、
> `P5-loss-weighting`（judged）、`P6-data-construction`（judged）
> 角色：知识波（无 GPU，不取锁） ｜ 依据：`SCHEMA.md` §18 / §19 / §21
> 本文件只写情报与分诊，**不是实验证据**。

---

## 0. 检索条件声明（必读）

**本环境未注册任何 web / 检索工具** ⇒ **无法在线核证任何外部数字**。遵守 §18/§21：
记忆类一律 **`记忆·未核`**、只作设计方向；本地数标 **`本地实测`** + 路径；不编数字。

术语：`bf16/fp16/fp32`（低/全精度浮点）、`autocast`（自动混合精度）、
`loss scaling`（混合精度下放大损失防止下溢）、`LN`（LayerNorm，层归一化）、
`RMSNorm`（只按均方根缩放的 LN）、`LoRA`（低秩适配器：冻结主干、加低秩更新矩阵）、
`零初始化`（新参数初始化成 0）、`鞍点`（梯度同时为 0 的驻点）。

---

## 1. 既有做法条目（§18 四问）

### 块①：精度（对标 `P1-precision`）

| 系统 / 论文 | 谁做过·做了什么 | 为什么 | 实测结果 | 我们的增量 |
|---|---|---|---|---|
| **Mixed Precision Training**（Micikevicius et al., 2017，`记忆·未核`） | fp16 前向/反向 + **fp32 master weights** + **loss scaling** | 低精度省显存/带宽，但小梯度会下溢 | 报告质量与全精度持平；loss scaling 是关键 | 我们的「fp32 读出/损失头 + pre-step grad-norm gate」是其特例 |
| **bf16 训练**（`记忆·未核`） | bf16 动态范围大、无需 loss scaling | 更稳、更省 | 主流默认 | 本地踩坑：bf16 下 reward 项发散的既有解释 |
| **RLHF/奖励模型稳定性**（`记忆·未核`） | 奖励/对数损失常单独用 fp32、加梯度裁剪 | 对数/负项在低精度下发散 | 社区惯例 | 我们的「pre-step grad-norm gate」是工程化写法 |

- **本地实测（P1）**：bf16 autocast 下的 negative/reward 项（capped −JS 多样性压力）
  **两次发散** ⇒ 必须 float32 + pre-step grad-norm gate；fp32 读出头是 prob0/传输/提交地基。
  来源：`.scratch/09-m3-interactions/evidence`。
- **判定**：**「低精度训练需 loss scaling / fp32 主权重 / 对数损失高精度」是 Micikevicius 2017
  以来的既有结论 ⇒ P1 的通用结论应改「业界结论」口径。** 本地的「pre-step grad-norm gate」
  是可复现的本地工程细节，保留本地口径。

### 块②：条件化信号（对标 `P2-conditioning`）

| 系统 | 既有做法（`记忆·未核`） | 我们的增量 |
|---|---|---|
| **FiLM**（Perez et al., 2018） | 用外部信号做特征级仿射调制 | 与「置信度门控」同族 |
| **AdaLN / DiT**（Peebles & Xie, 2023） | 时间步/噪声级条件注入 LayerNorm | 我们测了深度索引条件 |
| **conditional computation / gating** | 门控按信号开合 | 本地的置信度门控 |
| **task conditioning**（prompt/prefix） | 任务前缀条件化 | 零经验 |

- **本地实测（P2）**：深度索引（逐深度读出，13–17 格单调）与置信度条件（门控）已用；
  噪声级/任务条件零经验。来源：`.scratch/23-depth-ar/evidence`。
- **判定**：**AdaLN/FiLM 是成熟既有做法 ⇒ 「深度/噪声级条件化」应改「业界结论」口径**；
  「逐深度读出给出可观测深度曲线」的本地细节保留。

### 块③：参数插入与初始化（对标 `P3-param-insertion-init`）

| 系统 | 谁做过·做了什么（`记忆·未核`） | 与我们的关系 |
|---|---|---|
| **LoRA**（Hu et al., 2021） | 冻结主干，加低秩 `BA` 更新；**B 初始化为 0（单边零）**，使起步等价基线 | **我们的「单边零初始化」与 LoRA 的 B=0 是同一手法** |
| **Prefix / Prompt Tuning**（Li & Liang, 2021） | 加可训练前缀参数 | P3 的「prefix 参数」族 |
| **ReZero**（Bachlechner et al., 2020） | 残差分支乘可学习标量初值 0，起步即恒等 | 「恒等起步」的既有形态 |
| **Fixup**（Zhang et al., 2019） | 残差分支缩放 + 特殊初始化 | 恒等起步/稳定训练 |
| **ControlNet zero-conv**（Zhang et al., 2023） | 新增卷积层**输出侧零初始化**，起步恒等 | 单边零的又一实例 |
| **FiLM/AdaLN 初始化** | 门控初始偏置设成使起步近恒等 | carry 门 bias = σ(−2) 同族 |

- **本地实测（P3）**：proj-fill（恒等初始化 adapter）13M matched-step 族内最优
  （1.8356 < fill 1.8582 < nofill 1.9438，全深度支配）；**恒等初始化 ≠ 零初始化**：
  全零初始化在乘性结构上是**鞍点**（两端梯度同时 0）⇒ 干预系统性失效而测试全绿。
  来源：`.scratch/23-depth-ar/evidence`、`axes/W-width/W4-readout-geometry/experiments/2610-gated-readout-13m-depthar/zero-init-saddle/`。
- **判定**：**「输出侧零初始化/单边零=恒等起步」是 LoRA/ReZero/ControlNet 已做透的技法
  ⇒ P3 的通用结论（恒等起步有效）应改「业界结论」口径。** 「全零=鞍点」是本地可复现诊断
  （业界对 LoRA 用 B=0 而非 A=B=0 正是同一原因），保留本地口径。

### 块④：归一化位置（对标 `P4-normalization`）

| 既有做法（`记忆·未核`） | 结果 |
|---|---|
| **Pre-LN** vs Post-LN（Xiong et al., 2020） | Pre-LN 更易训练、可省 warmup |
| **RMSNorm**（Zhang & Sennrich, 2019） | 只按均方根缩放，省算力、质量相当 |
| **QK-norm** | 稳定注意力 logits |
| **无 final LayerNorm** | 使 `logits = h·Eᵀ` 的「同空间」语义成立（本项目前提） |

- **本地实测（P4）**：pre-LN + 无 final norm 是「同空间」几何前提、固定使用；
  readout scale 由 U 学。来源：`.scratch/03-engine-m1/issues`。
- **判定**：**Pre-LN/RMSNorm 是既有标准 ⇒ 改「业界结论」口径**；「无 final norm 配合 tied
  readout」的接口结论保留本地口径。

### 块⑤：损失权重与调度（对标 `P5-loss-weighting`）

| 既有做法（`记忆·未核`） | 结果 |
|---|---|
| **Deep supervision / 辅助分类器**（GoogLeNet aux heads 等） | 浅层监督是表示正则 |
| **Loss annealing / 课程** | 末段收紧辅助权重常见 |
| **梯度裁剪**（Pascanu 2013 等） | 稳定性标准手段 |
| **多目标加权**（不确定性加权 Kendall 2018 等） | 自动调权 |

- **本地实测（P5）**：浅层降权**全败且单调反向**（w∈{1,0.1,0.02,退火}：depth-L 2.45→2.57→2.73）
  ⇒ 浅层全深度监督是表示正则而非稀释主因；退火（末 25% 1.0→0.3、final 2×→1.0）使 C 配方
  族内最优（1.8429）。来源：`.scratch/23-depth-ar/evidence`、`.scratch/25-tax-attribution/evidence`。
- **判定**：**深度监督/退火/裁剪是既有做法（(a) 通用部分）**；本地的「浅层降权单调反向」
  是**可复现的反直觉本地观察**，保留本地口径（业界未必有此对照）。

### 块⑥：数据与任务构造（对标 `P6-data-construction`）

| 既有做法（`记忆·未核`） | 结果 |
|---|---|
| **Needle-in-a-Haystack**（Kamradt, 2023） | 长上下文检索探针的标准构造 |
| **合成任务**（MQAR / copy / induction） | 诊断长程能力的标准 |
| **数据增广**（mask/noise/wrong） | 修复侧结论的标准 |
| **局部先验偏差** | `entropy`/`bpc` 强局部统计数据会掩盖长程退化（`记忆·未核`） |

- **本地实测（P6）**：needle v1 随机填充 ⇒ 信号饥饿测到 0.0（假失败）；enwik8 字符级局部
  统计极强 ⇒ 距离惩罚的 bpc 改善与长程检索退化可同时发生。来源：`.scratch/05-m1-followup/evidence`、
  `.scratch/07-x1-x3/evidence`。
- **判定**：**needle 与合成任务构造业界成熟 ⇒ 通用部分改「业界结论」口径**；
  「任务构造是本项目最大单点偏差来源」+「局部先验抗性」是本仓库本地诊断（对应 §19），保留本地口径。

---

## 2. 机器可读分诊表（§21）

```
P1-precision            | (a) | 混合精度需 loss scaling/fp32 主权重/对数损失高精度，是 Micikevicius 2017 以来既有结论 | 结论：低精度训练须配 fp32 主权重与（对数/奖励类损失的）高精度路径。来源类型：业界结论（记忆·未核）+ 本地复核。通用部分改「业界结论」。
P1-precision            | (b) | 「bf16 下 reward 项发散 ⇒ float32 + pre-step grad-norm gate」是本仓库可复现的失败-修复路径 | 结论（本地）：判据/奖励类损失必须 float32 + pre-step grad-norm gate，否则两次发散。
P2-conditioning         | (a) | AdaLN/FiLM 是成熟的深度/噪声级条件化技法 | 结论：条件化信号（深度索引/噪声级）有成熟既有实现。来源类型：业界结论（记忆·未核）。通用部分改「业界结论」。
P2-conditioning         | (b) | 「逐深度读出给出可观测深度曲线」是本地接口细节，噪声级/任务条件本地零经验 | 结论（待测）：噪声级/任务条件在本项目零经验。
P3-param-insertion-init | (a) | 「输出侧零初始化/单边零=恒等起步」是 LoRA(B=0)/ReZero/ControlNet 已做透的技法 | 结论：恒等初始化起步有效且不消耗随机数（可严格对照）。来源类型：业界结论（记忆·未核）+ 本地复核。通用部分改「业界结论」。
P3-param-insertion-init | (b) | 「全零初始化是乘性结构上的鞍点 ⇒ 干预死掉但测试全绿」是本仓库可复现诊断 | 结论（本地）：恒等初始化 ≠ 零初始化；须单边零（输出侧置零、输入侧随机）。
P4-normalization        | (a) | Pre-LN / post-LN / RMSNorm / QK-norm 是既有标准 | 结论：Pre-LN 更易训练、可与省 warmup 组合。来源类型：业界结论（记忆·未核）。status: todo 口径。
P4-normalization        | (b) | 「无 final LayerNorm 使 logits=h·Eᵀ 同空间语义成立」是本项目底座接口 | 结论（本地）：无 final norm 是同空间前提的一部分。
P5-loss-weighting       | (a) | 深度监督/退火/梯度裁剪是既有做法 | 结论：多目标加权与退火是既有手段。来源类型：业界结论（记忆·未核）。通用部分改「业界结论」。
P5-loss-weighting       | (b) | 「浅层降权全败且单调反向」是本仓库可复现的反直觉观察，业界未必有同对照 | 结论（本地）：浅层全深度监督是表示正则，降低浅层权重不会提升最终质量。
P6-data-construction    | (a) | needle-in-haystack 与合成任务（MQAR/copy/induction）是长程诊断的标准构造 | 结论：长程能力诊断用标准合成任务 + needle 探针。来源类型：业界结论（记忆·未核）。通用部分改「业界结论」。
P6-data-construction    | (b) | 「局部先验偏差使长程退化被掩盖」（§19）是本仓库本地诊断 | 结论（本地）：enwik8 局部统计极强，长程结论须在弱局部先验任务上复核。
```

**分诊要点**：P 轴**通用技法全部 (a)**——混合精度纪律、恒等初始化、Pre-LN、
深度监督、needle 构造都是业界已做透的工程常识 ⇒ 通用结论应改「业界结论」口径；
本地的**反直觉/可复现诊断**（全零鞍点、浅层降权反向、局部先验偏差、pre-step grad-norm gate）
保留 `本地实测` 口径，作为「本项目在同一通用技法上的具体踩坑记录」。

---

## 3. 若可检索应查的关键词

| 格 | 关键词 |
|---|---|
| P1 | `mixed precision training Micikevicius 2017`, `loss scaling fp16`, `bf16 training stability`, `reward model fp32 loss`, `log-sum-exp numerical stability` |
| P2 | `AdaLN DiT Peebles`, `FiLM conditioning Perez`, `conditional computation gating`, `depth conditioning` |
| P3 | `LoRA B initialized zero`, `ReZero residual scaling`, `Fixup initialization`, `ControlNet zero convolution`, `identity initialization adapter` |
| P4 | `Pre-LN post-LN Xiong 2020`, `RMSNorm`, `QK-norm`, `final layer norm tied embedding` |
| P5 | `deep supervision auxiliary loss`, `loss annealing schedule`, `gradient clipping Pascanu`, `uncertainty weighting Kendall` |
| P6 | `needle in a haystack Kamradt`, `MQAR synthetic recall`, `local statistics language model bias`, `copy induction synthetic task` |

**核证优先级**：`LoRA B=0 单边零初始化`（决定 P3 是否彻底 (a)）与
`mixed precision loss scaling`（决定 P1 通用部分是否彻底 (a)）。

---

## 4. 来源清单

**本地实测**
- `.scratch/09-m3-interactions/evidence` — bf16 下 reward 项两次发散。
- `.scratch/23-depth-ar/evidence` — 逐深度读出、proj-fill、浅层降权反向、退火。
- `axes/W-width/W4-readout-geometry/experiments/2610-gated-readout-13m-depthar/zero-init-saddle/` — 全零鞍点。
- `.scratch/03-engine-m1/issues` — 无 final LN 的同空间前提。
- `.scratch/25-tax-attribution/evidence` — 结构化约束/退火。
- `.scratch/05-m1-followup/evidence`、`.scratch/07-x1-x3/evidence` — needle 构造、局部先验偏差。

**记忆（未核，只可作设计方向）**
- Micikevicius 2017 mixed precision / FiLM / AdaLN(DiT) / LoRA / Prefix Tuning / ReZero /
  Fixup / ControlNet zero-conv / Pre-LN(Xiong 2020) / RMSNorm / deep supervision /
  gradient clipping / Kendall uncertainty weighting / needle-in-haystack / MQAR。

**不确定 / 未找到**
- 各技法在**同尺寸 13M + 单因素受控**下的确切数字。
- 「pre-step grad-norm gate」是否有等价的公开命名/实现。
