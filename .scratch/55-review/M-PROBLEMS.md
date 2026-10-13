# M · 状态载体 — 独立批判性复核（问题台账）

> 复核者：fresh context，未参与该线任何实验/文档。
> 范围：`axes/M-carrier/**`、`catalog.yaml` 的 M* 条目、`pathlm/depth_ar.py` 的 carry 相关、相关 `.scratch`。
> 工作顺序：**先想（§0，先写，未看代码之前）→ 后看（§1 起）**。
> 只读复核 + 只写本文件；未改仓库其它任何文件。

---

## 0. 第一性原理（看代码之前先独立想清楚）

### 0.1 这条线想解决什么问题（痛点）

自回归 LM 生成第 t 个 token 时必须能用到前面所有 token 的信息。做到这一点的**默认手段**是把
每个位置的 K/V 全留着（KV cache），于是内存与每步算力随上下文长度线性增长。痛点 =
**「历史越长，账单越大」**，且这份账单里绝大部分位置其实再也用不到。

出资人给的目标词是「**信息右移**」：把远处（左侧）已经看过的信息，主动**搬到当前**（近处/
右侧），使**前缀可以被真正丢掉**——不是「丢掉了事」，而是「丢了，但该留下的已经搬走了」。
用本项目的话说：目标是让「**有界内存下的质量—代价前沿**」尽可能贴近「全量上下文」这条
100% 基准线，而**不是**追求绝对无损。

### 0.2 research topic 是什么（不是手段）

topic **不是**「载体类型」「窗口」「ALiBi」「GQA」「稀疏核」；那些是手段。真正的 topic 是：

> **在给定的「每位置内存字节 / 每步算力」预算下，用哪种状态表示可以把窗外（前缀里）的信息
> 搬到现在，使前缀可丢而质量与可检索性最接近全量注意力？这条「质量—代价前沿」长什么样？**

因此一格研究的落点必须是：**一个可被搬动的信息量（质量/检索）↔ 一份可计量的代价（字节/FLOPs）**，
两个轴都要有数，并且要和「全量注意力 = 100%」以及**同预算的既有省钱手段**（窗口、量化、驱逐、
摘要）放在同一张前沿图上比较。

### 0.3 应该引入什么机制

「显式右移」的机制必须回答四个正交问题，缺一条都不完整：

1. **写什么 / 何时写**：内容选择（写当前 hidden 的什么分量？是否只在信息新颖时才写？写入门是
   内容相关还是位置相关？）；
2. **更新式**：累加、带衰减门（`γS + ...`）、还是残差/delta（`S + β(v−Sk)kᵀ`）；
3. **读出**：按内容检索（槽注意力）还是按 query 的线性读（`S q`）；状态是**每层一个**还是全局一个；
4. **参数与预算**：状态大小、新增参数、与序列长度无关的 FLOPs。

其中 2 的 delta 规则是 DeltaNet / Gated DeltaNet 一系已经证过的机制；1、3、4 才是取舍空间。

### 0.4 代价应该怎么评（口径）

- 横轴：**每位置内存字节**（可与 fp16 KV 直接比）或每 token 额外 FLOPs；**不是**墙钟、也不是
  参数量百分比当唯一价签。
- 纵轴：**以全量注意力为标准锚的质量**（bpc / 任务准确率）**+ 可检索性**（远离检索才是
  「右移」的目的，bpc 只测「能不能补全」）。
- 参照系：全量注意力（100% 质量 / 100% 代价）、滑动窗口（各档）、KV 量化（int8/int4）、驱逐。
- **pareto frontier**：在横轴上取若干预算点，画出「最优可达质量」的包络。要回答的是
  「carry 载体落在包络的哪一段、是否被同预算的窗口/量化支配」。

### 0.5 业界的巨人（既有设计）

软注意力「退化成固定状态」这一族早已被摊开：

- 线性注意力（Katharopoulos 2020）、RetNet、GLA（门控线性注意力）；
- SSM：S4 → Mamba → Mamba-2（SSD，状态空间对偶）；
- **delta 规则系**：DeltaNet → Gated DeltaNet → DeltaProduct（写残差 `v − Sk`，2024–2025）；
- 混合架构：Jamba / Samba / Zamba / Griffin（少量线性层 + 若干全注意力层）；
- 外部记忆/显式槽：NTM/DNC、Memorizing Transformer（kNN）、Block-Recurrent、RMT、
  Compressive Transformer（压缩记忆 + compression loss）；
- MQAR（Zoology）是这一族的标准判据，且已给出「容量 vs 需要回忆的关联数」曲线。

⇒ 一个只加「显式写/读状态」的机制，**默认是站在巨人肩膀上**；能算本地增量的只有：具体设计
组合、「真实文本口径」、受控单因素读数、以及边界/否定结论。**必须与真实巨人基线对照**，
不能只用自己写的「plain 累加」当对照。

### 0.6 哪种候选更可能有效

- 纯单状态、无门控、全局共享 = 最弱，业界已反复报告「需求变大就塌」；
- **带内容相关写入门 + 衰减 + delta 更新 + 少量全注意力层混合** = 最可能有效；
- **读出必须是按内容的检索（或至少 query 相关的线性读）**，否则状态退化成「无损压缩不可能」的
  平均池化；δ 规则只在「写」这一侧帮忙。
- 关键预期：**必须有质量—代价前沿**；「不掉质量不可能」是错的读法，正确的是「每个预算点上
  能追回多少全量注意力的差距」。

### 0.7 现有方法对不对（预判，待 §1 起核对）

在看代码之前，据 topic 与巨人图景，我预判这条线会犯的典型错误是：**把「手段」当 topic、
只测「加通道的增量」而不测「前缀丢掉后的前沿」、没有与真实巨人对照、用退化实现（单槽）的
负读数去下机制结论**。下面逐条核对。

---

## 1. 问题台账

严重度：**致命** = 结论/方向不成立或不可采纳；**重要** = 显著削弱证据或浪费；**次要** = 一致性/残留。

---

### A. 目的与 topic 层（最致命）

**M-01｜M 轴没有「目的 → 假设（预期效用）→ 端到端验证」三元组，判据字段是手段清单**
- 为什么是问题：M 轴 `definition` 说了「载体决定能否丢/重算/检索」，但真正的判据 `judge`
  写成「保留（needle 消融）+ 内存/步长 + 吞吐」——这是三件**测量**，不是目标，也没有
  「相对全量注意力的质量—代价前沿」这根横轴；`judge` 里没有参照系、没有阈值、没有
  pareto 定义。全项目对 M 的验收因此无法回答出资人的核心问句「这玩意值不值」。
- 严重度：**致命**
- 证据锚点：`catalog.yaml:126-136`（M 轴 definition/subdims）、`catalog.yaml:909-918`
  （M1 `judge`/`cost`）、`axes/M-carrier/M1-carrier-type/README.md:6-7`。

**M-02｜topic 被写成「载体类型」枚举，把手段当研究对象；6 类选项只测了 2 类**
- 为什么是问题：研究 topic 应是「给定内存预算下哪种状态表示能把窗外信息搬到现在、前沿长什么样」，
  而不是「跨步信息存在哪里」这张选项表。枚举的 6 类里，KV 与显式记忆槽各有一个实现分支，
  递归/线性载体一次尝试即发散（未取得质量读数），外部存储/权重载体**零经验**；「窗口 + 线性载体」
  因引擎互斥**不可测**。⇒ 这个「类型」命题根本没有被单因素地比较过，却被标成 `judged`。
- 严重度：**致命**
- 证据锚点：`catalog.yaml:913`（options）、`axes/M-carrier/M1-carrier-type/README.md:30-34`
  （§2 选项枚举）、`axes/M-carrier/M1-carrier-type/experiments/2635-recursive-carrier-13m/analysis.md:1-30`、
  `axes/M-carrier/M1-carrier-type/README.md:135-137`（线性族权威叙事被推到 A 轴）。

**M-03｜「信息右移」的端到端判据缺失：测的是「窗口 16 上加通道的增量」，不是「丢掉前缀后离全量还差多少」**
- 为什么是问题：目的是「前缀可丢而不掉质量」。但真实文本实验是
  `bpc_none(窗口16)` vs `bpc_delta(窗口16)` 的臂间差（+0.0031/+0.0025），**不是**
  「丢掉前缀后 vs 保留全部 KV」的差距；全量注意力 1.4836 只被当「尺度感参考」（同目录
  analysis 自己注明口径不同、未重跑）。⇒ 实验测的是「加通道有没有一丁点增量」，而目的要求的是
  「有界内存点落在前沿哪里」。这是**口径与目的错位**，不是「结果弱」。
- 严重度：**致命**
- 证据锚点：`axes/M-carrier/M1-carrier-type/experiments/2634-carry-realtext-13m/analysis.md:15-17,133,144`、
  `axes/M-carrier/M1-carrier-type/experiments/2634-carry-realtext-13m/README.md:7-18`（命题只写「能不能搬进来」）。

**M-04｜机制是 DeltaNet/GLA 一族的复刻（重新造轮子），且造得明显更弱，无业界基线对照**
- 为什么是问题：实现的写/读就是已证机制——写
  `k̂=normalize(W_k h0)`、`v=W_v h0`、`β=σ(W_β h0+b_β)`，
  `delta: S←S(I−βk̂k̂ᵀ)+βvk̂ᵀ`；读 `mem_t=S_t q_t` 加进每层注意力输出。这**正是** DeltaNet /
  Gated DeltaNet 的残差写 + 线性读（catalog 自己承认「机制排序不是本地增量」）。问题在于：
  ① 对照只有自己写的 `plain`（同一套张量改个更新式），**没有任何真实业界实现**（GLA/Mamba-2/
  Gated DeltaNet）作对照；② 在 MQAR 家族任务上，DeltaNet 系在可比小规模通常能到接近满分，
  本实现在超训练距离档只到 **9.5%**、分布内 36%，明显弱于巨人。⇒ 只能得出「我们这个（退化）
  实现够不着」，不能得出「载体类型」层面的判断。是**站了肩膀但把轮子重造得更差**。
- 严重度：**致命**
- 证据锚点：`pathlm/depth_ar.py:1357-1405`（`_carry_chunk_plain/_carry_chunk_delta`）、
  `pathlm/depth_ar.py:1410-1464`（`carry_scan_*`）、`pathlm/depth_ar.py:1484-1527`（`CarryChannel.write_read`）、
  `axes/M-carrier/M1-carrier-type/experiments/2618-carry-channel-13m/analysis.md:1-60`、
  `catalog.yaml:906`（`prior_art_note` 自认机制非本地增量）。

---

### B. 机制/设计层

**M-05｜决定性设计妥协（单共享槽、只从 h0 写）由墙钟/引擎开销逼出，却被写成对「载体」的判断**
- 为什么是问题：`CarryChannel` 明确记录：per-layer 12 个状态实测 U1 207 ms/step、U2 396–425 ms
  （基线 42–51），超预算 2.4×，因此**降级为单共享槽**（一次扫描、写只来自输入流 h0、每层只留一个
  标量读出增益 γ）。这等于把「每层一个可写状态」这一族的表达力砍掉大半，而业界主流线性层是
  **逐层**的。结论「通道只带来 +0.2%」因此**至少部分是实现/预算的产物**，不能读成
  「显式载体在这一族里不行」。按 `AGENTS.md` §4.3，这正是「我们的方法/实现有问题」的典型，
  必须先排除再下结论。
- 严重度：**重要**（但影响结论）
- 证据锚点：`pathlm/depth_ar.py:1468-1477`、`axes/M-carrier/M1-carrier-type/experiments/2618-carry-channel-13m/analysis.md:137-168`。

**M-06｜「explicit 右移」的设计要素不完整：缺衰减门、层间状态、内容/时机的可选择写、重置**
- 为什么是问题：出资人问「怎么个 Explicit 法」。当前 explicit 只做了：每位置写（写内容 = h0 的
  线性像）、delta 更新、线性读、全局单状态、参数 +264k。**缺**：写入门的内容相关性/时机选择
  （β 是逐位置标量，不含「该不该写」的离散/稀疏选择）、衰减/遗忘因子（`γS`）、层间独立状态、
  状态重置/淘汰。设计稿 §2c 也只形式化了**写规则**一维。⇒ 这是「手段列举不完整」，还谈不上
  一个完整机制设计；而业界 Gated DeltaNet（门 + delta）恰好是这几项的合体。
- 严重度：**重要**
- 证据锚点：`pathlm/depth_ar.py:1484-1520`、`axes/M-carrier/M1-carrier-type/experiments/2617-design-rightcarry/DESIGN.md:150-200`
  （只覆盖 (a)–(e) 手段清单）、`axes/M-carrier/M1-carrier-type/experiments/2617-prior-art-memory/PRIOR_ART.md`（GLA/Gated DeltaNet 条目）。

**M-07｜「容量曲线不成立」与业界「容量是约束」的核心图像冲突，且只在退化实现上测**
- 为什么是问题：Zoology/MQAR 的公认图像是「可回忆关联数受状态容量限制」（「需求变大时纯单状态
  塌」）。本线却在 `dh=16/32/64/128`（状态 4096→32768，×8）上测出**准确率一动不动**，
  并据此写「该尺度加大状态不划算」。两种可能没被分开：① 瓶颈其实在**训练/任务/退化读出头**，
  不是容量；② 状态确实已经够用。实验自己说「最可能是瓶颈在训练而非容量」，但 catalog 把这条
  当成 M1 的结论之一。⇒ 与业界冲突时，应先怀疑实现/规模，而不是把「容量无效」写成结论。
- 严重度：**重要**
- 证据锚点：`axes/M-carrier/M1-carrier-type/experiments/2619-carry-difficulty-capacity/analysis.md:32-57,144`、
  `axes/M-carrier/M1-carrier-type/experiments/2620-capacity-converged/analysis.md:25-56`、`catalog.yaml:962`。

---

### C. 判据与口径层

**M-08｜「长程检索能力本身 ≈ chance」是零样本、分布外、未训练合成任务的读数，被当成了模型的「能力性质」**
- 为什么是问题：`2617` 电池在一个**从没为这些游戏训练过**的 13M 主干上取零样本读数（关联回忆
  0.0061、变距离回忆 0.0052），分析自己在「残差风险」里写明「assoc/var 低分含分布外成分，
  **不得**外推成『训练也学不会』」。但 `catalog.yaml` 的 M1 `failure` 把它写成
  「**长程检索能力缺席**……此前所有『没搬』读数，至少相当部分来自**模型本来就没有该能力**」，
  并作为解释其他负结果的支柱。零样本 OOD 读数**不能**证明「能力缺席」，只能说明「没训过的
  任务格式它不会」。⇒ 这条是错的归因，而且它被用来给一批历史负结论「洗白」，影响面大。
- 严重度：**致命**
- 证据锚点：`axes/M-carrier/M1-carrier-type/experiments/2617-longrange-task-battery/analysis.md:1,20,174-177`、
  `catalog.yaml:951`（failure 第 2 条）。

**M-09｜真实文本「+0.2%」的判读：实验在该任务上几乎没有区分力，正确的读法是「任务不需要」而非机制结论**
- 为什么是问题：三点叠加——
  ① 同目录 analysis 自己承认**训练末 loss 的种子间差（1.0551 vs 1.1448 = 0.090 nat）远大于
  臂间差（0.005–0.010）**，n=2 下臂间比较被种子方差支配；
  ② 收益绝对值 0.2%，seed1 的配对 CI **跨 0**；
  ③ 窗口 16 no-channel 已经 1.628，全注意力 1.4836，通道只**回收了 2%** 的截断损失。
  enwik8 字节级 next-token 的局部先验本就补全绝大部分（analysis 原话），**没有长程需求**。
  ⇒ 这个实验不能用于判断「显式载体有没有用」；它只说明「在一个不需要长程的任务上，加通道
  几乎不涨」。这是**任务/规模问题 + 设计无区分力**，不是「机制搬不动」。文档目前用
  「弱证据、可检索性未证」把两种解释混在一起。
- 严重度：**致命**
- 证据锚点：`axes/M-carrier/M1-carrier-type/experiments/2634-carry-realtext-13m/analysis.md:15-17,33,108-111,133,144`、
  `axes/M-carrier/M1-carrier-type/experiments/2634-carry-realtext-13m/results.json`（`bpc_win16`）。

**M-10｜「关掉通道 bpc 1.625 崩到 4.51」被当作通道价值的证据——实为「依赖」而非「增益」**
- 为什么是问题：无通道训练臂本身就能到 1.628。带通道臂在消融后崩到 4.51，只说明**这个模型
  把自己的计算搬到了通道上**（训练学到了依赖），并不说明通道给了净收益（净收益 = +0.2%）。
  文档用「模型把质量压在通道上」来强化「通道重要」，但正确的对照是 no-channel 臂的 1.628，
  两者几乎相等 ⇒ 通道是一条**冗余路线**，不是「不可替代的收益来源」。
- 严重度：**重要**
- 证据锚点：`axes/M-carrier/M1-carrier-type/experiments/2634-carry-realtext-13m/analysis.md:36-38,128-136`、
  `catalog.yaml:911`（tax 条「说明模型把质量压在通道上」）。

**M-11｜合成leg的核心读数在两个距离域间跳动（9.5% vs 36%），headline 混用口径**
- 为什么是问题：首验 `2618` 主腿用距离 **241–255（> 训练上限 240）**，得 9.5%（6.08×chance）；
  后续 `2631/2633` 用**训练分布内**（107–240）多查询，得 **36%**。同一机制在两个距离域差 3.8×。
  catalog 的 `benefit` 第 1 条（真实文本）与第 2 条（合成）并列，读者容易把 36%（分布内）与
  9.5%（越界）当同一件事。且越界腿后来被自己判「永久降级/不稳健」。⇒ label 必须随距离域分开，
  headline 不能只取最好那半。
- 严重度：**重要**
- 证据锚点：`axes/M-carrier/M1-carrier-type/experiments/2618-carry-channel-13m/analysis.md:66-84`、
  `axes/M-carrier/M1-carrier-type/experiments/2633-carry-seed2/analysis.md:15-31`、
  `catalog.yaml:948-960`。

**M-12｜全轴没有质量—代价前沿（每位置字节的横轴），carry 从未被放到与窗口/量化/驱逐同一预算轴**
- 为什么是问题：R1 已经建好「每位置 KV 字节 vs bpc」的预算图（fp16/int8/int4/头共享/窗口同轴），
  出资人明确要的就是这种 pareto。M1 只报「状态 8192 floats、参数 +1.95%、算力与 T 无关」，
  **没有**把 carry 的状态字节换算成「每位置字节」并与窗口/量化点同图；M2 的 `judge` 写了
  「双边前沿」但**没有任何实验**。⇒ 无法回答「值不值」。
- 严重度：**重要**
- 证据锚点：`catalog.yaml:915-917`（M1 tax 只报参数量）、`catalog.yaml:987-989`（M2 judge/verdict）、
  对照 `axes/R-references/R1-quantization/experiments/2631-kv-budget-causal/analysis.md`（同预算前沿图存在）。

---

### D. 结构 / 树 / 重复 / 浪费

**M-13｜M1 单格塞了 22 个实验目录（含 3 个引擎子项目），是扁平列表而不是树**
- 为什么是问题：出资人明确「研究是一棵树，不是扁平列表；每人的活要清晰、独立、无重复无浪费」。
  M1 一个格子下挂了 22 个目录，从引擎改造、设计、情报，到 9 个合成任务变体、4 个组合实验，
  没有分层、没有「问题→做法→子做法/成败」的树结构。读完 5451 行 analysis 还得不到一个判断。
- 严重度：**致命**
- 证据锚点：`axes/M-carrier/M1-carrier-type/experiments/` 目录列表（2503, 2616, 2617×3, 2618,
  2619, 2620, 2621, 2622, 2623, 2624×2, 2625, 2627, 2628, 2629, 2630×2, 2631, 2633, 2634, 2635）。

**M-14｜组合实验违反「单轴单格」，被登记为「例外」却正是出资人说的乱拆**
- 为什么是问题：`SCHEMA.md §4` 默认「只做单轴单格、组合留白」，且**明文把** M 线这几组
  （搬运×深度监督 2×2、GQA+carry+窗口阶梯、头共享×量化）登记成「已发生的例外」。
  但这些组合（`2627` carry×depth、`2621`/`2624` carry×量化、`2623`/`2624` carry×GQA×窗口、
  `2629`/`2630` carry×ragged×proj-fill）本质是**产品形态的组合取舍**，应属产品方 scope，
  不是载体研究格的一部分。把它们塞进 M1 让格子既长又偏题。
- 严重度：**重要**
- 证据锚点：`SCHEMA.md:81-86`、`axes/M-carrier/M1-carrier-type/experiments/2627-carry-x-depth/`、
  `.../2623-composable-arms-13m/`、`.../2629-carry-ragged-projfill/`、`.../2630-carry-ragged-seed1/`。

**M-15｜严重重复：同一合成 assoc-recall 任务被 9 个实验反复测，单位信息增量低**
- 为什么是问题：`2618/2619/2620/2623/2625/2629/2630/2631/2633` 都在同一 MQAR 家族合成任务上，
  改动多为 seed/距离/预算/组合的小扰动。反复 reseed（`2631` seed 重跑、`2633` 补 seed2、
  `2630` carry 的 seed1）是为了把一个「方向一致性」结论做实，但结论本身（delta 不塌、plain 塌）
  在 `2618` 就已有雏形。⇒ 这是「没有重复、没有浪费」的反例。
- 严重度：**重要**
- 证据锚点：上述各目录 `analysis.md`；`axes/M-carrier/M1-carrier-type/experiments/2631-carry-reseed/analysis.md`、
  `.../2633-carry-seed2/analysis.md`。

**M-16｜实验编号不唯一：同一格内 2617×3、2624×2、2630×2；跨轴 2614/2615/2618/2621/2631 等重复**
- 为什么是问题：目录名第一段是实验 ID，被当作可追溯主键。同一格出现两个 `2624-*`、两个 `2630-*`、
  三个 `2617-*`，跨轴又与 R1/A9 等撞号（例如 `2614` 在 M2 与 R1 各有一个、且含义不同）。
  ⇒ SSOT/复现锚点无法唯一定位。
- 严重度：**重要**
- 证据锚点：`axes/M-carrier/M1-carrier-type/experiments/{2617-design-rightcarry,2617-longrange-task-battery,2617-prior-art-memory}`、
  `{2624-carry-under-compression-quality,2624-composable-followups-13m}`、
  `{2630-carry-ragged-seed1,2630-tf4-samecaliber}`；`axes/M-carrier/M2-carrier-compression/experiments/2614-kv-ablation-retention-13m` vs
  `axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m`。

**M-17｜引擎/基础设施子项目被挂进「研究格」，手段混进研究树**
- 为什么是问题：`2503-engine-m1-extensions`、`2622-eng1-composable`、`2628-eng2-carry-ragged`
  是**为了让实验可组合而改引擎**的工程项。工程使能项有其价值，但不是「载体类型」这个研究格的
  研究结论；把它们与判决实验平铺在一起，正是「基线/参照/修饰符不是子项目」的同类错误。
- 严重度：**重要**
- 证据锚点：`.../2503-engine-m1-extensions/README.md`、`.../2622-eng1-composable/`、`.../2628-eng2-carry-ragged/`、
  `AGENTS.md`（ticket 55 §2「参照/基线混进研究列表」）。

---

### E. M2 与文档一致性 / 残留

**M-18｜M2「载体压缩」没有任何属于自己的压缩实验——它的 evidence 是「KV 删除代价」的仪器**
- 为什么是问题：M2 claim =「载体怎么压（内存换质量）」，但两个实验是 `2614-kv-ablation-retention`
  （评测期删 KV 的保留曲线，测**依赖**）与 `2615-retention-retrieval-leg`（加检索腿）。二者都在
  量「删掉记忆会掉多少」，**不是任何压缩方法（摘要/低秩/量化）**。真正的压缩实验都在 R1（量化）
  和 A3（驱逐）。⇒ M2 这个格子名不副实，没有能回答自己 claim 的证据。
- 严重度：**重要**
- 证据锚点：`axes/M-carrier/M2-carrier-compression/experiments/2614-kv-ablation-retention-13m/README.md:14-18`、
  `.../2615-retention-retrieval-leg-13m/README.md:14-18`、`catalog.yaml:985-1003`。

**M-19｜M2 verdict 陈旧：要求「按因果/流式口径重测（见 R1）」，而 R1 早已完成重测**
- 为什么是问题：M2 `verdict` 写「KV 量化有本地实测（R1/2614、2618/2621），但价签须按因果/流式
  量化口径重测（见 R1）」。但 R1 已完成 `2631-kv-budget-causal`、`2632-kv-block-scale`，并已
  **撤回旧逐值标尺口径**、给出块标尺部署口径（int8 省 46.875%）。⇒ 事实文档里留着一条已经
  被别处满足的待办，属于「事实文档不得有过期内容」。交叉引用「R1/2614、2618/2621」也与 R1
  实际目录（2614-kv-quant-eval、2619/2620/2621-kv-budget-*）对不上。
- 严重度：**重要**
- 证据锚点：`axes/M-carrier/M2-carrier-compression/README.md:11,22,46`、`catalog.yaml:993`、
  `axes/R-references/R1-quantization/experiments/2632-kv-block-scale/analysis.md`。

**M-20｜M2 没有目的/假设，「选项」又一次是手段清单**
- 为什么是问题：M2 的选项是「不压/量化/驱逐窗口/摘要压缩/低秩投影（MLA 式）」——全是手段；
  「摘要压缩、低秩投影（MLA 式）无经验（待补做）」没有写**假设**（预期创新效用）、没有写
  **端到端验证**（用什么横轴、什么参照系、前沿长什么样）。按出资人判据 = 只列手段没有目的。
- 严重度：**重要**
- 证据锚点：`catalog.yaml:987-989`、`axes/M-carrier/M2-carrier-compression/README.md:29-33,58-66`。

**M-21｜同一事实两个家：两份内容不同的 PRIOR_ART.md，且 catalog 的 `prior_art_source` 与 evidence 各指一份**
- 为什么是问题：`AGENTS.md §2.1` 要求「每类事实恰好一个家」。实际存在
  `axes/M-carrier/PRIOR_ART.md`（轴级）与 `axes/M-carrier/M1-carrier-type/experiments/2617-prior-art-memory/PRIOR_ART.md`
  （实验级），两份标题/范围/内容均不同；`catalog.yaml` 的 M1 `prior_art_source` 指向前者
  （`@e3b00ba`），而 M1 的 `evidence` 列表指向后者（`@e3b00ba`）。⇒ 矛盾、无法确定权威。
- 严重度：**重要**
- 证据锚点：`axes/M-carrier/PRIOR_ART.md:1-6` vs `axes/M-carrier/M1-carrier-type/experiments/2617-prior-art-memory/PRIOR_ART.md:1-6`；
  `catalog.yaml:906,975`。

**M-22｜M1 标 `judged`，但 `verdict` 是「弱证据/未证/降级为单种子观测」的堆叠，没有明确结论**
- 为什么是问题：`SCHEMA.md §1` 定义 `judged` = 已有明确结论；`todo` = 尚无明确结论。
  M1 verdict 自己写「可检索性未证」「越界腿降级为单种子观测」「质量收益≈0.2%、一个种子 CI 跨 0」
  「该实现数值发散，无质量读数」。这些是**边界与未证**，不是一个明确结论；同时又把
  「6.08×chance / 36% / 2.32×」等正读数并列。读者拿不到一个可采纳的判断。⇒ 状态枚举用错，
  或应把「已证/未证」拆成两条结论。
- 严重度：**重要**
- 证据锚点：`axes/M-carrier/M1-carrier-type/README.md:11`、`catalog.yaml:909-981`。

**M-23｜旧代号/旧 ticket 残留：M3 已并入却仍以 M3 记账；evidence 仍指向早期 `.scratch/03-engine-m1`**
- 为什么是问题：`原 M3-cross-step-assumption 已并入本格` 但仍用「M3 的有界内存价签随并入本格」
  叙述（事实文档应陈述式、不带已废台账指代）；M1 的 `evidence` 第 11 条仍是
  `.scratch/03-engine-m1/issues @e38349f`（旧 ticket 的日志，不是复现锚点）。
- 严重度：**次要**
- 证据锚点：`catalog.yaml:978-983`、`axes/M-carrier/M1-carrier-type/README.md:135-139` 与 §9 第 11 条。

**M-24｜上一版本引擎残留：仓库根并存 loop-era 脚本与 V2 引擎**
- 为什么是问题：M 轴实验实际用 `train_depth_ar.py` + `pathlm/depth_ar.py`（V2）。根目录同时留着
  `train_m0.py`、`train_m1.py`、`probe_*.py`（probe_tts/mtp/mtp3/spec/...）、`slider.py`、
  `train_jumphead.py` 等上一时代（loop-era，配 `pathlm/model.py`）的入口。出资人要求「上一版本、
  上一时代的残留内容完全 migrate 掉」；这里两套引擎并存，容易误用与误引。
- 严重度：**次要**
- 证据锚点：仓库根 `ls`（`train_m0.py`/`train_m1.py`/`probe_*.py`/`slider.py`）、`pathlm/model.py` 与
  `pathlm/depth_ar.py` 并存；`axes/M-carrier/M1-carrier-type/README.md` §9（用 `train_depth_ar.py`）。

**M-25｜数字口径护栏是补丁（黑名单）而非结构修复**
- 为什么是问题：`catalog.yaml` M1 带 `key_numbers_forbidden: ["2.53%", "约 2.2×"]`，说明同一批
  硬子集数字曾出现口径错误，靠**黑名单**封堵。这是结果性补丁；根因（判据定义分散、数字在多处
  手抄）未解决，同类错误仍会在新数字上复发。
- 严重度：**次要**
- 证据锚点：`catalog.yaml:964`。

---

## 2. 最致命的三条（若只修三处）

1. **M-01/M-03**：先给 M 轴写出「目的 → 假设 → 端到端验证」与**质量—代价前沿**（横轴 = 每位置
   字节 / 每 token 额外 FLOPs，纵轴 = 相对全量注意力的质量 + 可检索性，参照系 = 全量/窗口/
   量化）。没有这张图，「+0.2%」「未证」都无法被判断。
2. **M-04/M-05/M-06**：把机制从「退化单槽」修成一个**完整、且与业界同量级**的设计
   （逐层状态 / 门控衰减 / 内容相关写入门 / delta 更新），并**与真实 DeltaNet/GLA/Mamba-2 基线
   对照**；当前结论「载体只值 +0.2%」至少部分是墙钟妥协的产物，不能作机制结论。
3. **M-08/M-09**：撤销两处错误归因——「长程检索能力本身≈chance」（零样本 OOD）与
   「真实文本 +0.2% 说明搬得动/有用」；改为「该任务/该规模无区分力，尚不能判机制价值」，
   并把「任务侧需求」（必须丢掉前缀的构造）作为判据的前置条件。

---

## 3. 复核方法与边界

- 本文件只读复核：读了 `catalog.yaml` M*、`axes/M-carrier/**` 的 README/catalog 生成页/关键
  `analysis.md`/`README.md`、`pathlm/depth_ar.py` 的 carry 实现、`SCHEMA.md §4/§81-86`、
  `docs/design.md` 相关节、以及 `axes/R-references/R1-quantization` 的对照。
- 未重跑任何实验、未核任何外部论文原文（本环境无检索）；涉及业界数字处按项目纪律标为「既有做法/
  记忆·未核」，只用于设计方向判断，不作本地判据。
- 仅写入本文件；未改仓库其它文件。
