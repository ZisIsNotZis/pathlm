# PRIOR_ART_CAPACITY — 长程记忆的**定量律**情报（K7）

> 轴：**M 状态载体** ｜ 角色：知识波（无 GPU、不取锁、不改代码/测试）
> 依据：`SCHEMA.md` §18（既有做法四问）、§19（局部先验抗性）、§21（三选一分诊）
> 目的：为**下一批 13M GPU 实验选参数**——什么旋钮该扫、什么已知无意义（避免重复造轮子）。
> 本文件**不是实验证据**，无 `results.json`；本地数标 `本地实测`，外部数一律 `记忆·未核`。

---

## 0. 检索条件声明（决定本文证据等级，必读）

**本环境未注册任何 web / 检索工具**（与既往 researcher run 同因）。因此：

- 凡来自记忆的论文/系统/数字，一律标 **`记忆·未核`**，**只可作设计方向，不得当判据**（§18）。
- 凡来自本仓库的数，标 **`本地实测`** 并给文件路径。
- 记不清的**一律降级为定性表述**，**绝不编造数字**；不确定处标 `不确定` 并给「要查什么」。
- 每块末尾给「若可检索应查的关键词」。

术语（首次出现就地解释）：`MQAR`（Multi-Query Associative Recall，多查询关联回忆：先给
K 对「键→值」，末尾问某键求值；答案不可由局部规律猜出）、`状态维度 d_state`（递归/线性
载体里那块固定大小记忆的宽度）、`chance 基线`（瞎猜命中率，通常 = 1/值词表大小）、
`delta 规则`（写入时先扣掉状态已能预测的部分，即 `S ← S(I−βkkᵀ)+βvkᵀ`）、
`税`（相对普通 transformer 的质量代价）。

---

## 0.5 人话摘要（先看这段）

**这份文件想回答什么。** 我们要开一批新的 13M 实验，问「一块固定大小的记忆，到底能记
多少东西、记多牢」。在设计参数之前，先查业界**有没有把这件事量化过**——哪些旋钮（状态
宽度、层数、需要回忆的关联条数）已经被扫过、用什么判据、结论是什么。目的不是抄，而是
**别把人家做透的东西再跑一遍**（§21）。

**最要紧的三句话。**
1. **「状态容量 ÷ 需回忆的信息量」是业界公认的量化横轴**：他们用 MQAR 类任务，把「需要
   回忆几条关联」当成旋钮，画「准确率 vs 状态大小 / 层数」曲线。**具体是线性还是次线性，
   我记不清 ⇒ 不写数字**，只记方向。（`记忆·未核`）
2. **delta 写族（只搬状态还没搬的那部分）比朴素累加稳定**，这是 2024–2025 的既定共识；
   **我们本地已经独立复核过**（13M：delta 9.5% vs plain 双稳态）。所以这一条在我方账本里
   应归 (a)「业界做透」⇒ 直接写结论，别当新发现重跑（§21）。
3. **真正该我们做的是**：把**状态宽度（我们代码里的 `carry_state_dim`）✕ 需要的关联条数 K**
   做成受控扫描，在**局部先验抗性任务**上画容量曲线。这正是业界曲线没在「13M + 字节级 +
   局部先验受控」这个组合上做过的地方 = (b)，值得跑。

**本地已有的两片拼图**（`本地实测`）：13M 零样本长程能力**分任务型**（复制 13.3% 真而弱、
关联回忆 0.006 ≈ 瞎猜）；加了 delta 写入通道后关联回忆从 1.37%（≈瞎猜）升到 **9.50%**
（6.08× 瞎猜）。但**容量旋钮 `carry_state_dim` 目前只跑过 20 步冒烟**，没有真正的容量曲线
——这就是下一批的靶子。

---

## 1. 状态容量 vs 可回忆信息量（能不能量化？）

### 1.1 谁做过 / 为什么（`记忆·未核`）

- **MQAR 系列**（`Zoology` / `Based`，Arora 等，约 2023–2024，`记忆·未核`）：把「多对随机
  (键→值) + 末尾问值」做成合成诊断任务，用来判定「线性/次二次模型在**精确回忆**上是否
  掉队」。**动机**：把 O(T²) 注意力换成固定状态时，状态成为瓶颈，需要一个**不靠语言先验**
  的探针把这个瓶颈量出来。
- **Repeat After Me**（Jelassi 等，约 2024，`记忆·未核`）：用「先给随机串、后要求原样重复」
  测复制/长度外推，报告 SSM 类在复制上失败。
- **Mamba / Mamba-2 / RWKV / DeltaNet 的公开消融**（`记忆·未核`）：把**状态维度 `d_state`
  / 头数 / 层数**当旋钮，报「回忆准确率 vs 状态大小」的曲线族。
- **理论侧**：线性注意力/SSM 的「可存储关联数 ~ 状态维度」是记忆容量的常见直觉
  （把 `S = Σ v kᵀ` 看作一张关联表，其秩受 d_state 限制）。**具体形式（线性/次线性/带常数）
  我不确定，不写数字。**（`记忆·未核`）

### 1.2 旋钮 / 样本 / 判据（**这三样是我们选参数的关键**，`记忆·未核`）

| 维度 | 业界常见做法（定性） | 不确定处 |
|---|---|---|
| **旋钮** | ① 需回忆的关联数 K（MQAR 的键值对数）；② 状态维度 `d_state`；③ 层数 / 头数；④ 序列长度 / 关联间距离 | 到几层几头为止、是否交互 |
| **样本量** | 训练样本量按「足够收敛的 token 数」给；**我不记得具体 token 数**，`不确定` | 收敛判据未核 |
| **判据** | **token 级准确率（argmax 命中率）**为主；有的报「是否越过某阈值」的拐点 | 阈值口径各异 |
| **chance 基线** | = **1 / 值词表大小**（值在词表子集里均匀采样） | 是否另设「局部先验」基线，多篇**不设** |

> **对我们最有信息量的一点**：业界 MQAR 通常**不控制局部先验**（§19）。也就是说，
> 「13M + 字节级 + **局部先验抗性任务**」这个组合，业界曲线**大概率没覆盖** ⇒ 属 (b)。

### 1.3 本地已有的「半张曲线」（`本地实测`）

- **任务电池（零样本，不训练）**：`axes/M-carrier/M1-carrier-type/experiments/2617-longrange-task-battery/analysis.md`
  —— 关联回忆 K=4/8/16 逐档 0.0039 / 0.0078 / 0.0059，**无随 K 的趋势，均 ≈ 或低于 chance**；
  复制 0.1333（距离 223 仍 0.118，毁源→0.0125）；induction 0.0586（毁源→chance）。
  判据含 **chance = 1/64** 与**局部先验基线**（k=8），并做了毁源/打乱答案变异验证。
- **写入通道（训练 12k 步）**：`axes/M-carrier/M1-carrier-type/experiments/2618-carry-channel-13m/analysis.md`
  —— 同一任务，delta 写 U2 的 **K 敏感性**：K=4 → **14.4%**、K=8 → **9.5%**、K=16 → **5.6%**，
  随 K 单调下降（与「容量 ÷ 需求」图像一致）；U0（无通道）三档均 ≈ chance。
  **注意**：这是**同状态宽度**下只动 K 的三点，**不足以定标容量律**（且 K=16 时主判据
  +4.40pp 未过线）。
- **容量旋钮已有实现但未跑**：`carry_state_dim`（`pathlm/depth_ar.py`，只读确认）支持
  16/32/64/128，**2619-carry-difficulty-capacity 目前只有 20 步冒烟**（`smoke/smoke_dh*/`），
  **无真实容量读数**。⇒ 下一批第一优先。

### 1.4 我们的增量（§18 第 4 问）

不是「首发 MQAR」，而是：**在 13M / 字节级 / 局部先验受控**的三重约束下，画
**`carry_state_dim` ✕ K 的准确率曲面**，对照**同尺寸 dense 全注意力**。业界的曲线在
「大模型 + 合成数据 + 不控局部先验」上，我们在「小模型 + 受控探针」上——**尺度与任务是
实质不同**（(b)），可以做。

**若可检索应查**：`MQAR multi-query associative recall`, `Zoology state size recall curve`,
`Based linear attention recall`, `Mamba-2 state size ablation`, `associative recall capacity
linear attention state dimension`, `Repeat After Me copying SSM`, `sequence length extrapolation recall`.

---

## 2. delta 规则族（写入式 / 为什么好 / 何时不如 attention）

### 2.1 各写法的**写入式**（`记忆·未核`，公式记个大意，细节标不确定）

| 系统 | 写入式（大意） | 它加的什么 |
|---|---|---|
| **plain 累加** | `S_t = S_{t-1} + β v kᵀ`（见到就加） | 无 |
| **DeltaNet** | `S_t = S_{t-1}(I − β k kᵀ) + β v kᵀ`（先扣掉状态已能预测的 `S k`，只写残差 `v − S k`） | 「只搬没搬的」 |
| **Gated DeltaNet** | 在 DeltaNet 上再乘一个**衰减/门控** `α`（近似遗忘） | 写入 + 遗忘 |
| **DeltaProduct** | 一个位置做**多步** Householder 式更新（乘积） | 更强的单步写入 |
| **TTT（test-time training）** | 把状态当一层线性网络，在**测试时**用自监督步做梯度更新 | 「写入 = 一步训练」 |

- **为什么比 plain 好**（`记忆·未核`，与本地机制读数吻合）：plain 累加会把**重复内容反复
  堆叠**，有限状态被很快写满、范数膨胀 ⇒ 有效容量下降；delta 先减预测再写，**同一块状态
  能装下更多不同**的关联。**本地实测印证**：`2618/analysis.md` 里 plain 状态范数从 50 涨到
  71.8 且仍在涨，delta 稳定在 ~3（2.93 均值 / 3.11 末值）。
- **什么时候 delta 也救不了 / 不如 attention**（`记忆·未核` + 推断）：**需要的关联数超出状态
  容量**时（K 大、`d_state` 小），任何固定状态写法都会塌，而 softmax 注意力是**精确检索**、
  不掉。这与 §1 的「容量 ÷ 需求」直接相连。
- **有没有「delta 比 attention 更好」的公开设定？** `不确定`。**我能确定的只是**：业界的比较
  主轴是**效率（O(T) vs 对称注意力）与长程回忆**，delta 族相对的是**其它线性/SSM**，
  而不是「在质量上普遍超过 softmax 注意力」。记忆中**没有**「同规模同数据上 delta 全面胜
  全注意力」的干净结论。**不要据此写「delta > attention」**。`要查`：`Gated DeltaNet vs
  attention quality`, `DeltaNet long context benchmark`.

### 2.2 本地对账（`本地实测`）

- delta 写 U2 = **9.50%**（CI 8.64–10.43）vs 无通道 U0 = **1.37%**（Δ +8.13pp，CI 不重叠），
  毁源后回 **0.81%**（≈ chance）⇒ 「信息真被搬到可检索处」成立。
- **但 plain 写双稳态**（同配置同 seed 两次 0.0007 / 0.0334）⇒ **「delta 稳定优于 plain」这一
  强表述不作为我方结论**；即便取最利于 plain 的那次（3.34%），delta 仍高 +6.15pp。
- **分诊**：「delta 写（只搬残差）优于无残差 plain 写」是 DeltaNet 系的既定机制（`记忆·未核`）
  ⇒ 在我方账本里归 **(a) 业界做透**，`status: todo`，本地数保留为**同尺度独立复核**。

**若可检索应查**：`DeltaNet delta rule residual write`, `Gated DeltaNet`, `DeltaProduct`,
`test-time training TTT linear attention`, `delta rule vs Hebbian accumulation capacity`.

---

## 3. 遗忘 / 覆盖（写入冲突）

### 3.1 有没有量化「写入冲突」导致的遗忘率？（`记忆·未核`）

- **没有我能引用的干净「遗忘率」数字**。`不确定`：是否有工作把「写入冲突频率」直接和
  「旧关联被覆盖的比率」做成曲线。常见的**定性**图像：**位置/键不冲突时线性注意力的关联
  存储会互相干扰（crosstalk），delta 与门控是缓解手段**。
- **掩盖（interference）**是「关联记忆容量受限」的标准解释：`S = Σ v kᵀ` 里，查询一个键会
  收到所有别的键的**串扰**；键越不正交、条数越多，串扰越大。**本地机制读数（`2618`）里
  delta 的「读分享 →0.7078」是这一侧的一个亲缘观测**，但**不等于**遗忘率。
- `不确定`：是否有公开的「遗忘率 vs K / d_state」曲线。`要查`：`associative memory
  interference capacity`, `catastrophic forgetting linear attention overwrite`.

### 3.2 LRU / 使用计数分配（NTM/DNC）在语言任务上为什么失败？

- **既有解释（`记忆·未核`）**：NTM/DNC 在**小合成任务**（copy、bAbI）上很强，但**极难训练、
  几乎没进主流 LM**。常见归因（**定性，均未核**）：① 读/写头与**使用计数分配**里有大量
  **近似不可微/离散**操作（LRU 式 free list、argmax 分配），梯度信号弱、训练不稳；
  ② 需要**精心调参**与课程；③ **长序列上的信用分配**困难；④ 引入的复杂度在小规模上
  换不回质量。`不确定`：这些归因各自的**证据强度**与是否有一篇**直接分析**。
- **本地实测（与业界叙事有张力，重要）**：`axes/A-mixing/A10-compression-trigger/README.md`
  —— 13M/256 快照上 distance / lru / content 三规则**同等删除预算**对照，**LRU 未测出可见
  收益**；字面优势经对账（按「掉了多少可见性对数」重对齐）后 **lru −0.0170（由正转负）**；
  `lru:k=256` 是「低损伤 ≠ 保住了」的又一实例（bpc 伤害最低 +0.0885 而召回 4.5→22.8pp）。
  按 §19，这只能读作**该任务/该尺度上不划算**，**不得**写成「LRU 机制不可能」。
- **概念桥**：用户「只搬上次没搬的」的 LRU 构想，其**最强的既有形态不是「挑哪块 KV 删」的
  驱逐策略，而是 §2 的 delta 写入算子**——把「淘汰」变成「写入时只写残差」。这条桥
  **已在 `2618` 上被本地验证有效**，比继续做驱逐调度更接近用户原话。

**若可检索应查**：`NTM DNC training instability analysis`, `differentiable memory hard to
train`, `LRU usage-based allocation language model`, `H2O eviction reproducibility`,
`KV cache eviction robustness critique`.

---

## 4. 对下一批 13M 实验的直接含义（该扫什么 / 什么已知无意义）

### 4.1 **该扫的旋钮**（按信息量排序）

| 优先级 | 旋钮 | 我们代码里的字段（只读确认） | 判据 | 理由 |
|---|---|---|---|---|
| **P0** | **状态宽度** 16/32/64/128 | `carry_state_dim` | 关联回忆准确率（argmax）+ chance + 局部先验 | 唯一的**容量律**横轴；已有实现，2619 只跑了冒烟 |
| **P0** | **需回忆的关联数 K** 4/8/16/32 | 数据生成器 `assoc-mqar`（pairs） | 同上 | 与 §1 的「容量 ÷ 需求」对齐；本地已有 K=4/8/16 三点 |
| **P1** | **每层读 vs 单共享槽** | 通道拓扑（`2618` 用单共享槽；per-layer-12 已实测出局） | 准确率 + 步时（ms/step） | per-layer-12 **本地实测 221/410 ms/step 出局**（算子派发开销），**别再试** |
| **P1** | **训练信号**（重建/充分性损失） | `recon_window`（+ λ） | 保留腿（可重建性），**不是**只看 bpc | `F4` 空白格；业界有占位（Compressive Transformer 谱系），本地 consistency loss 中性 |
| **P2** | 关联间距离 d | 任务电池 d ∈ {16…255} | 距离-准确率曲线 | 本地已知复制**平**、检索**缺席**；delta 通道六档全 > chance（最远 255 为 0.595） |

> **一个必须先修的验证问题**：2618 的 **plain 臂双稳态**（同 seed 0.0007 vs 0.0334）说明
> **13M/12k 步的训练可复现性不足**。容量扫描的第一件事不是加大规模，而是**多 seed × 固定
> 「是否逃出瞎猜线」的成功判据**，否则容量曲线的点会落在双稳态的两端、不可比（`2618` 结论④）。

### 4.2 **已知无意义 / 别重复造的轮子**（§21 分诊）

- **(a) 别再当新发现重跑**：
  - 「delta 写优于 plain 累加」——DeltaNet 系已证（`记忆·未核`），本地已复核 ⇒「业界结论」。
  - 「大窗口训练税近零、小窗口付质量；丢前缀是『丢了』不是『搬了』」——滑动窗口/StreamingLLM
    既定 ⇒ `M3` 主结论「业界结论」（本地 32→+0.0866 / 192→−0.0028 作复核）。
  - 「KV int8 近无损、int4 有损」「压缩 ≠ 保信息」——KV 量化谱系已做透 ⇒ `M2` 量化支「业界结论」。
  - **MQAR 任务本身**：不要自造新合成任务，复用 MQAR 式构造（本地 `tools/evalkit/longrange_tasks.py`）。
- **(b) 才该我们做**：状态宽度 ✕ K 的**容量曲面**（13M + 局部先验受控）；「有/无 sufficiency
  损失」在**小尺度 + 受控任务**上的受控对照；同一字节预算下的压缩前沿（U3 线在做）。
- **(c) 查不到**：写入冲突「遗忘率」曲线、NTM/DNC 失败的**直接分析**论文、delta 是否在
  某设定**胜过 attention** ⇒ 记 `记忆·未核`，**不得**据此判 (a)。

### 4.3 三条落地纪律（复用现有工具，别新造）

1. **chance 与局部先验基线必须并列报**（`2617` 已建立：chance=1/64，局部先验 k=8，ceiling=0.046875）。
2. **每个容量点必须配毁源对照臂**（`2618`：毁源后回 chance 才可信）＋二项 CI。
3. **单距离/单点不得当可加外推**（`M2` ERRATA 教训：只保留最近 32 实测累计 +0.212 bpc，
   是单点读数的 15 倍）。容量曲线同理，只报测过的点。

---

## 5. 本文件里「若被证实会改变我们实验设计」的**最该优先核证三条**

1. **「状态维度 ✕ 可回忆条数」的曲线形式（线性 or 次线性，及拐点位置）。**
   若业界已给出**跨规模的普适定律**（例如「可回忆对数 ≈ 常数 × d_state」在 13M–1B 上稳定），
   那我们只需**在 13M 上验一个点**即可外推，**不必扫 16/32/64/128 四点**；反之若发现
   **小模型严重偏离**，则容量曲线本身就是我们的**独有增量**（§9 定律 8「规模翻转方向」）。
   → **改变 P0 扫描的宽度与是否值得跑。**
   `查`：`MQAR state size scaling law`, `associative recall capacity vs model scale`。

2. **delta/门控类相对 softmax 注意力是否存在**任何**公开的「质量更好」设定（非仅效率）。**
   若**存在**，我们在混合比例（契约 C）上的设计空间要重排，delta 通道可能从「搬运探针」
   升级为**主力质量手段**；若**不存在**（我目前的记忆倾向），则 delta 在我们这里应只当
   **搬运机制**，质量结论一律以 attention 为上限。→ **改变 delta 通道的定位与后续实验配额。**
   `查`：`Gated DeltaNet vs attention language modeling quality`, `hybrid attention delta net ratio`。

3. **写入冲突「遗忘率」是否有可引用的量化曲线 / NTM·DNC 失败是否有直接分析论文。**
   若**有**，§3 可整段从「定性」升级为「结论」，我们**不必**再花实验测「写入冲突导致遗忘」，
   直接把结论写进 `M1`（§21 (a)）；若**无**，则「K 增大时 delta 通道的遗忘率」本身
   是**未被占位的可测靶**，可作为一个新判据。→ **决定 §3 是写结论还是排队做实验。**
   `查`：`memory overwrite interference rate`, `differentiable memory training failure analysis`。

> **核证优先级（若只有一个窗口）**：**第 1 条**（决定 P0 容量扫描的规模）＞ 第 3 条（决定 §3
> 是否需实验）＞ 第 2 条（影响更长远的混合比例设计）。

---

## 6. 关键词清单（交给下一次有 web 工具的 run）

| 块 | 关键词 |
|---|---|
| 容量律 | `MQAR multi-query associative recall`, `Zoology efficient language models recall`, `Based linear attention recall`, `Mamba-2 state size ablation`, `associative recall capacity linear attention`, `Repeat After Me copying SSM`, `state dimension recall scaling` |
| delta 族 | `DeltaNet delta rule residual write`, `Gated DeltaNet`, `DeltaProduct`, `test-time training TTT`, `delta rule vs accumulation capacity`, `Gated DeltaNet vs attention quality`, `hybrid attention SSM ratio` |
| 遗忘/覆盖 | `associative memory interference capacity`, `catastrophic forgetting linear attention overwrite`, `write conflict forgetting rate`, `NTM DNC training instability analysis`, `differentiable memory hard to train`, `LRU usage-based allocation language model`, `H2O eviction reproducibility critique` |
| 判据/基线 | `MQAR chance baseline value vocabulary`, `in-context recall probe local prior baseline`, `length extrapolation accuracy criterion` |

---

## 7. 来源清单

**本地实测（可直接当证据）**
- `axes/M-carrier/M1-carrier-type/experiments/2617-longrange-task-battery/analysis.md`
  — 关联回忆 0.0061、复制 0.1333（距离 223 仍 0.118）、induction 0.0586；chance=1/64、局部先验 k=8。
- `axes/M-carrier/M1-carrier-type/experiments/2618-carry-channel-13m/analysis.md`
  — delta 写 9.50% vs U0 1.37%（毁源→0.81%）；K=4/8/16 = 14.4/9.5/5.6%；plain 双稳态；状态范数 plain 50→71.8 vs delta ~3。
- `axes/M-carrier/M1-carrier-type/experiments/2619-carry-difficulty-capacity/smoke/` — `carry_state_dim` 16/32/64/128 **仅 20 步冒烟**，无真实读数。
- `axes/M-carrier/M2-carrier-compression/experiments/2614-kv-ablation-retention-13m/analysis.md` — 保留曲线 + ERRATA（只留最近 32 累计 +0.212）。
- `axes/A-mixing/A10-compression-trigger/README.md` — LRU vs distance vs content 同预算对照，重对齐后 lru −0.0170。
- `axes/M-carrier/PRIOR_ART.md`、`.../2617-prior-art-memory/PRIOR_ART.md` — 同轴前序情报（本文是其「定量律」细化）。
- `pathlm/depth_ar.py`（只读）— `carry_rule` / `carry_state_dim` / `carry_chunk` / `recon_window` 字段与校验。

**记忆（未核，只可用于设计方向，不得当判据）**
- Zoology / Based（MQAR）、Repeat After Me、Mamba / Mamba-2 / RWKV / RetNet / GLA、
  DeltaNet / Gated DeltaNet / DeltaProduct / TTT、NTM / DNC、H2O / Scissorhands / SnapKV、
  Compressive Transformer 谱系、The Illusion of State。

**不确定 / 未找到（本 run 无法核）**
- 容量律的**精确函数形式**与拐点；MQAR 各工作的**确切样本量/判据阈值**。
- 「delta 族在质量上胜过 attention」的公开干净设定（我倾向：不存在，但**不能断言**）。
- 写入冲突「遗忘率」的量化曲线；NTM/DNC 失败的**直接分析**论文。
- 各压缩/驱逐工作的**确切压缩率与基线强度**。
