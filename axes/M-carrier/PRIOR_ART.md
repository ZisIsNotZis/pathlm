# PRIOR_ART — M 状态载体（M1–M3）

> 轴：**M 状态载体** ｜ 格子：`M1-carrier-type`（judged）、`M2-carrier-compression`（partial）、
> `M3-cross-step-assumption`（judged） ｜ 角色：知识波（无 GPU，不取锁）
> 依据：`SCHEMA.md` §18（既有做法）、§19（局部先验抗性）、§21（不重复造轮子三选一分诊）
> 本文件只写情报与分诊，**不是实验证据**，无 `results.json`。

---

## 0. 检索条件声明（决定本文证据等级，必读）

**本环境未注册任何 web / 检索工具**（与上一轮 researcher 失败同因，见
`.scratch/36-parallel/issues/01-parallel.md`）。因此本文**无法在线核证任何外部数字**，遵守：

- 凡来自记忆的论文/系统，一律标 **`记忆·未核`**，**只可作设计方向，不得当判据**（§18）。
- 凡来自本仓库的数，标 **`本地实测`** 并给文件路径。
- 不确定的标 **`不确定`**，并写「要查什么才能确定」。
- **不含任何无法回溯的精确数字**：记忆中的数字一律降级为定性表述。宁可少写，不编造。

术语（首次出现解释）：`KV`（注意力缓存：模型回头查的键值表）、
`bpc`（每字符比特数，越低越好）、`税`（相对普通 transformer 的质量代价）、
`MQAR`（多查询关联回忆：给若干「键→值」对，末尾问键求值的合成任务）、
`delta 规则`（状态更新写成 `S ← S·(I−βkkᵀ)+βvkᵀ`，即只写入状态还预测不出的残差）。

---

## 1. 既有做法条目（§18 四问）

### 块①：载体类型（对标 `M1-carrier-type`）

| 系统 / 论文 | 谁做过·做了什么 | 为什么这么设计 | 实测结果 | 我们的增量 |
|---|---|---|---|---|
| **Transformer-XL**（Dai et al., 2019，`记忆·未核`） | 段级递归：缓存**上一层**隐藏状态当记忆 + 相对位置编码 | 不做压缩，用缓存换更长有效上下文、免重算 | 推理长上下文质量优于朴素截断；`数字未核` | 我们测的是「不重 prefill 的有界内存」价签，同机制不同接口 |
| **Compressive Transformer**（Rae et al., 2019，`记忆·未核`） | 两级记忆：近期原始 + 被压过的旧记忆 | 内存有界同时保住远距离 | 中低压缩率下同预算几乎不亏；`数字未核` | M2 的摘要压缩空白格，见块② |
| **Memorizing Transformer**（Wu et al., 2022，`记忆·未核`） | 可训练层 + kNN 检索**非可微**外部记忆 | 检索式保留、不压 | 检索命中提升；质量保持但成本转到检索 | 外部存储载体空白格；成本腿可复用 |
| **RMT**（Bulatov et al., 2022/23，`记忆·未核`） | 段级递归 + 特殊 memory token 携带跨段信息 | 把有效上下文推到极长 | 长上下文报告有效；`数字未核` | 显式记忆槽载体空白格 |
| **NTM / DNC**（Graves et al., 2014/16，`记忆·未核`） | 可微外部记忆 + 读/写头 + 基于使用量的分配 | 合成任务上证明「练出来的记忆」可行 | 合成任务强，但极难训练、几乎没进主流 LM | 「难训练」是我们要排除的既有解释 |
| **线性注意力 / SSM 族**（Katharopoulos 2020、Mamba、RWKV、RetNet、GLA；`记忆·未核`） | 把 `softmax(QKᵀ)V` 改写成固定大小状态的递归 | 把 O(T²) 压成 O(T) | MQAR 上明显逊于 softmax 注意力；「需求变大时纯单状态塌、注意力几乎不掉」（Zoology，`记忆·未核`） | 「载体容量÷需回忆信息量」的旋钮（本地已用） |
| **delta 规则系**（DeltaNet / Gated DeltaNet / DeltaProduct，约 2024–2025，`记忆·未核`） | 状态更新写成「写入 `v − Sk` 的残差」，即只搬状态还没搬的 | 更会写的状态更新，而非更大的状态 | 报告在 MQAR / 长程回忆上优于 Mamba-2；`数字未核` | **这就是我们 ticket 35 的「delta 写」**——见下 |
| **现代 Hopfield**（Ramsauer et al., 2020，`记忆·未核`） | 把注意力解释成关联记忆检索 | 统一「检索=能量最小化」视角 | 单步检索可逼近多步；`数字未核` | 理论解释，非判据 |
| **Fast Weight Programmers**（Schmidhuber, 1992；`记忆·未核`） | 权重本身当可写载体 | fast-weight 载体原型 | 历史工作，规模小 | 「权重载体」空白格的祖先 |

**关键对账（针对 `status: judged` 的 M1）**：

- **本地实测（ticket 35）**：显式写入通道成立——关联回忆 + 窗口 16（key 100% 窗外）上，
  13M/12k 步：无通道 U0 = 0.0137（≈chance）→ **delta 写 U2 = 0.0950（6.08× chance）**，
  毁源后掉回 0.0081；plain 写 U1 双稳态（0.0007 / 0.0334）。
  来源：`axes/M-carrier/M1-carrier-type/experiments/2618-carry-channel-13m/analysis.md`。
- **业界（`记忆·未核`）**：delta 规则 > 无门控线性 / 单纯递增，是 2024–2025 的既定共识；
  「只搬残差」的写入算子在 MQAR 上被反复报告有效。
- **判定**：**「delta 写比 plain 写稳定有效」这一机制级结论，业界早已有（DeltaNet 系）。
  ⇒ 账本该条应改为 `status: reference` 口径**，标注「业界结论，非本地实测」，
  本地 ticket 35 只作**同尺度的独立复核**保留。
- **不得外推**：本地绝对水平仅 9.5%（13M/12k）⇒ 按 §19 只能读作「在该任务/该尺度上成立」。
- **不确定**：DeltaNet 系是否在所有规模上成立、与 plain 写的**净差**数字。`要查`：
  `Gated DeltaNet MQAR`、`DeltaNet delta rule residual write`。

### 块②：载体压缩（对标 `M2-carrier-compression`）

| 路线 | 既有做法（`记忆·未核`） | 为什么 | 我们的增量 |
|---|---|---|---|
| **KV 量化** | KIVI、KVQuant、SmoothQuant、LLM.int8() 等 | 低比特换内存/带宽 | 见 §3 R1；M2 可复用其坐标系 |
| **驱逐/窗口** | H2O、Scissorhands、SnapKV、PyramidKV、Ada-KV、StreamingLLM | 保留「重要」token | 本地已测驱逐价签（见下） |
| **摘要压缩** | Compressive Transformer、AutoCompressors、ICAE、500xCompressor | 把旧历史压成少量向量 | **M2 空白格**；业界有占位 |
| **低秩投影（MLA 式）** | DeepSeek MLA、CLA/MLKV/YOCO 跨层共享 | 联合压低秩潜向量 | W3 相关；M2 空白格 |
| **动态压缩** | DMC（NVIDIA，2024，`记忆·未核`） | KV 动态压成单槽 | 需继续预训练/蒸馏——「压缩无训练信号不够」的既证 |

- **本地实测（M2）**：驱逐 +0.058~0.121（13M）；保留曲线严格单调，单距离
  d=1 +3.245 / d=8 +0.090 / d=16 +0.017 / 恰在 d=32 ≤+0.004；**ERRATA**：单距离不得当
  可加外推——只保留最近 32 个实测累计 **+0.212 bpc**（单点 15 倍）。
  来源：`axes/M-carrier/M2-carrier-compression/experiments/2614-kv-ablation-retention-13m/analysis.md`。
- **审慎提示**：业界多数压缩工作**只报 ppl/质量、不报可重建性**；我们的天平发现
  「低损伤 ≠ 保住了」（`lru:k=256` 伤害最低 +0.0885 而召回 4.5→22.8pp）⇒ 引用业界
  「几乎无损」结论时**必须补检索腿**（§17.10）。
- **判定**：M2 的**量化/摘要/低秩**三支业界均有成熟占位 ⇒ 若只是复现，**无增量**（(a)）；
  真正可做的是**同一字节预算横轴**下的受控对照（U3 线在做）。

### 块③：跨步假设（对标 `M3-cross-step-assumption`）

| 系统 | 既有做法（`记忆·未核`） | 结果 |
|---|---|---|
| Transformer-XL | 段级递归 | 见块① |
| 滑动窗口注意力（Longformer、Mistral 等） | 每 token 只看最近 k 个 | 大窗口训练税近零，是主流默认 |
| StreamingLLM | 窗口 + attention sink | 丢最前缀会崩、加回 sink 即恢复 |

- **本地实测（M3）**：窗口 192 训练税 **−0.002813**（只能报「近零」，重跑 +0.000673）；
  窗口 32 训练税 **+0.086566**；窗口外信息是「丢了」不是「搬了」（整段窗外的 494 根差
  恰好 0.000）。来源：`axes/M-carrier/M3-cross-step-assumption/experiments/2616-window192-trained-13m/analysis.md`。
- **判定**：「大窗口训练税近零、小窗口要付质量」在业界是成熟共识（滑动窗口注意力）
  ⇒ M3 的**主结论应改 `status: reference` 口径**；本地的 13M 两档数字作为**独立复核**保留，
  且**不得用两点定趋势**（§17.11，悬崖形状不可判定）。

---

## 2. 机器可读分诊表（§21）

格式：`grid_id | 判定 | 一句话理由 | 若要写进账本的结论`

```
M1-carrier-type        | (a) | 「delta 写算子有效、plain 写不稳定」是 DeltaNet 系已证机制（记忆·未核） | 结论：显式写入通道 + delta 写（只搬残差）能稳定搬运信息，优于无残差的 plain 写。来源类型：业界结论（记忆·未核）+ 本地同尺度复核。口径改为 reference。
M1-carrier-type        | (b) | 13M char-enwik8 + 局部先验抗性任务上的绝对水平（9.5%）与 plain 双稳态，业界不报此尺度 | 结论（本地·保留本地实测口径）：13M/12k 上 delta 写 6.08× chance、毁源→chance；绝对水平仅 9.5%，不外推。
M1-carrier-type        | (c) | 递归状态/显式记忆槽/外部存储/权重载体我们零经验，且无法在线核业界最新进展 | 方向：四类载体各有成熟前例；若要写账本需先按 (a) 补业界条目或本地实测。
M2-carrier-compression | (a) | KV 量化、摘要压缩、低秩投影（MLA）业界均已做透 | 结论：KV int8 近无损、int4 有损；压缩≠保信息，须配可重建性判据。来源类型：业界结论（记忆·未核）。
M2-carrier-compression | (b) | 「同一每位置字节预算下的受控对照」业界多为工程配方、缺干净 sweep | 结论（待 U3 实测）：同一字节横轴上量化/驱逐/窗口的统一质量-召回前沿。
M2-carrier-compression | (b) | 驱逐的累计口径（只留最近 32/64/128/192 各档）是本仓库本地实测 | 结论（本地）：单距离曲线不得当可加外推；只留最近 32 = +0.212 bpc（单点 15 倍）。
M3-cross-step-assumption | (a) | 「大窗口训练税近零、小窗口付质量；丢前缀信息是『丢了』不是『搬了』」是滑动窗口注意力/StreamingLLM 的既定结论 | 结论：有界内存的价签——窗口越大越近零税；限制可见范围不产生右移搬运。口径改为 reference。
M3-cross-step-assumption | (b) | 训练侧只有 32 与 192 两点，悬崖位置不可判定（§17.11） | 待补 64/128 训练点后方可定位悬崖。
```

**分诊要点（给主 Agent）**：
- M1/M3 的**主结论**判为 (a) ⇒ `catalog.yaml` 对应 verdict 应加「业界已有」标注、
  状态口径改 `reference`；本地数字保留为「同尺度复核」，**不删除**。
- M2 的量化/摘要/低秩判 (a)，但「同一字节预算受控对照」判 (b)（U3 线在做）。

---

## 3. 若可检索应查的关键词

| 块 | 关键词 |
|---|---|
| ① 载体/写入 | `MQAR multi-query associative recall`, `Zoology efficient language models recall`, `Repeat After Me copying SSM`, `DeltaNet delta rule residual write`, `Gated DeltaNet`, `DeltaProduct`, `linear attention fixed state recall capacity`, `fast weight programmer`, `modern Hopfield associative memory` |
| ② 压缩 | `KIVI KV quantization`, `KVQuant`, `SmoothQuant`, `Compressive Transformer compression loss`, `AutoCompressors reconstruction loss`, `ICAE in-context autoencoder`, `500xCompressor`, `DMC dynamic memory compression`, `MLA DeepSeek low-rank KV`, `cross-layer attention KV sharing`, `H2O heavy hitter oracle KV eviction` |
| ③ 跨步 | `Transformer-XL segment recurrence`, `sliding window attention perplexity`, `StreamingLLM attention sink`, `sink tokens training emergence` |

**核证优先级**：`DeltaNet / Gated DeltaNet 的 MQAR 曲线`（决定 M1 是否彻底 (a)）与
`Compressive Transformer compression loss`（决定 M2 摘要支的既有占位）。

---

## 4. 来源清单

**本地实测（可当证据）**
- `axes/M-carrier/M1-carrier-type/experiments/2618-carry-channel-13m/analysis.md` — delta 写 6.08× chance、毁源→chance、plain 双稳态。
- `axes/M-carrier/M1-carrier-type/experiments/2617-longrange-task-battery/analysis.md` — 长程能力分任务型（关联回忆 0.0061 等）。
- `axes/M-carrier/M2-carrier-compression/experiments/2614-kv-ablation-retention-13m/analysis.md` — 保留曲线 + ERRATA（累计 +0.212）。
- `axes/M-carrier/M3-cross-step-assumption/experiments/2616-window192-trained-13m/analysis.md` — 窗口 192 −0.0028 / 窗口 32 +0.0866。

**记忆（未核，只可作设计方向）**
- Transformer-XL / Compressive Transformer / Memorizing Transformer / RMT / NTM·DNC /
  linear attention·Mamba·RWKV·RetNet·GLA / DeltaNet·Gated DeltaNet·DeltaProduct /
  modern Hopfield / Fast Weight Programmers / DMC / MLA。

**不确定 / 未找到**
- DeltaNet 系与 plain 写的**净差数字**与是否跨规模成立。
- 各压缩工作的**确切压缩率、bpc/ppl、对照基线强度**。
- 「无需训练信号即可压缩」是否有反例。
