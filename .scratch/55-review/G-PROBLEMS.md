# G-PROBLEMS · G 轴（策略与调度 / Slider）独立批判性复核

> 复核者：fresh context，未参与该研究。前提：**这条线做的一塌糊涂**。
> 范围：`axes/G-policy/**`、`catalog.yaml` 的 G* 条目、`pathlm/slider.py`、相关 `.scratch`。
> 方法：先第一性原理独立思考（§0），再读实际代码/证据核对（§1 起）。只读；只写本文件。
> 出资人判据：目的 vs 手段（手段不是 research topic）／问题 + 假设 + 端到端验证／用「信息右移」等既有词／
> 找 pareto frontier 而非绝对无损／基线不是子项目／清掉上一版本残留。

---

## 0. 独立先验（读代码之前，第一性原理）

**它想解决的问题**：普通 decoder-Only transformer 推理时对每个输入花**固定**算力（全深度、全 token）。
若「简单输入」和「困难输入」能被区分，就可以把算力按难度分配：同质量省算力，或同算力换质量。
痛点是**质量-算力（质量-代价）折中**。

**research topic 应该是什么**：不是「谁选路径」（选择器），不是「用什么信号」（prob0/js），
更不是「怎么校准」——这些都是**手段**。可证伪的 topic 应是：
> 「**按输入/按位置自适应分配算力，能否把质量-算力 pareto 前沿相对『最优固定路径』上移？**」
（以全量固定路径为 100% 参照，画出前沿。）

**应该引入的机制**：一个 selector，把某观测（便宜的信号）映射到算力分配（每个 token 跑几层/几轮）。
真正的机制问题只有一个：**「低置信位置每 FLOP 的边际收益是否更高」**——若是，按置信分配值；若否，均匀即可。

**代价如何评判**：
- x 轴 = **端到端实际算力**（应双报 forwards 与 FLOPs，且以硬件无关的解析量为主）；
- y 轴 = **端到端质量**（bpc / 准确率，含方差带）；
- 对手 = ①「最优固定路径」（同平均算力下的常量深度/常量阈值）② oracle（上界）③ 随机（零假设）；
- 判据 = **前沿是否上移**，**不是**「工具自洽」（成本预测准、开火率可预测）。

**巨人（既有设计）**：ACT（Graves 2016）、MoD（Raposo 2024）、CALM（Schuster 2022）、
LayerSkip（Meta 2024）、MoE 路由、SelectiveNet/选择性预测、早退/级联、投机解码、温度缩放（Guo 2017）、
语义熵（Kuhn 2023）、SelfCheckGPT、distinct-n。→ **这条线的核心机制全部是既有设计**。

**哪种候选更可能有效**：真正可能有增量的是**「有验证兜底」的选择器**（投机解码式 accept/reject、
级联 deferral），因为它是「用便宜路径 + 后续信息纠错」；纯置信阈值（prob0 门控）能否买到质量，
是**经验问题且已知弱**（校准好 ≠ 选择有收益）。→ **信号/校准是选择器的手段，不构成独立研究项。**

**我的先验判断**：如果一条线把「校准好、成本模型准、开火率可预测」当成结论，而拿不出「前沿上移」，
那它做的就是**工程（造一个可预测的调度工具）**，不是研究。

**读代码后的核对结果**：下文 18 条问题，基本印证了上述先验——**topic 缺失、目的漂移到工具 KPI、
唯一「严格占优」是带内差、信号/校准被误立为独立格、双货币是方法学而非研究、大量上一版本残留。**

---

## 1. 问题清单

### P1 ·【致命】G 轴没有 research topic，只有手段

- **问题**：`catalog.yaml` 给 G 轴的命题是「谁选路径、观测什么信号、怎么校准」（`catalog.yaml:145`）；
  G1 的命题写成「**谁决定这条路怎么走。**」（`axes/G-policy/G1-selector/README.md:26`）；
  G3 命题「用什么信号判断…」（`G3-signal/README.md:26`）；G4 命题「置信信号是否可信」（`G4-calibration/README.md:26`）。
  三句全是**手段陈述**，没有一句写「要解决的痛点 / 目的 / 端到端效果」。
- **为什么是问题**：直接违反出资人判据「只列手段、没有目的 = 瞎研究；手段（窗口、ALiBi、稀疏核、GQA…）
  不是研究 topic」。G1 自己的 `verdict` 也承认判据「按原预登记…是否获证（待补做）」——连问题都没立住。
- **严重度**：致命
- **证据锚点**：`axes/G-policy/G1-selector/README.md:24-26,11`；`catalog.yaml:145,1010`；
  `axes/G-policy/G3-signal/README.md:26`；`axes/G-policy/G4-calibration/README.md:26`。

### P2 ·【致命】G1 没有任何端到端结果证明「前沿上移」；唯一「收益」是工具自洽 KPI

- **问题**：G1 的 `benefit` 写的是「Slider 端到端：校准（fire split gap ≤0.004）→ 双货币成本模型
  （预测 ≤±13%）→ 求解器 → 验证解码 → 在线代理（TF 窗口 ECE 0.0011–0.0079）」——
  这一串全是**工具链内部的自我一致性指标**（成本模型准不准、代理 ECE 低不低），
  **没有一个是「同算力下质量更好」**。整个 G 轴**找不到一句用质量-算力前沿表述的结论**。
- **为什么是问题**：出资人判据是「找 pareto frontier（以全量标准为 100%）」。把「工具有效」当作
  「研究有价值」是**偷换判据**（优化了一个不代表目标的中间指标）。
  讽刺的是：真正的原始前沿数据**存在**（`ticket 14` 的 `frontier_INT2.json` 里有 `passes_per_tok`
  × `clean/corrupt_bpc_gated` × `tok_s` 的网格，且 `issues/01` 明写实验 1(3) 要「产出质量-算力前沿曲线」），
  但 G1 的 verdict **没有引用它**，反而去报成本模型误差。
- **严重度**：致命
- **证据锚点**：`axes/G-policy/G1-selector/README.md:10,38,45`；`catalog.yaml:1018`；
  `.scratch/14-allocator/evidence/composition/frontier_INT2.json`（真前沿数据）；
  `.scratch/14-allocator/issues/01-step1-completion.md`（实验 1(3) 目标）。

### P3 ·【致命】Slider /「双货币记账」是方法学与工程，不是研究；「严格占优」用自造货币

- **问题**：G1 的核心 evidence 与「收益」几乎全部来自 Slider 这套**离线求解器 + 成本模型**。
  而「双货币（forwards/FLOPs）必须双报」原本是独立格 `G2-currency`，已被判为
  「**不是可证伪的研究问题，是记账纪律**」并应并入 `docs/methodology.md`（`methodology.md:49`
  已写「原研究格 G2-currency 的纪律，**非可证伪命题**」）。G1 却把这条纪律当核心卖点。
  更糟的是：`forwards` 是与硬件脱节、可被 batch 摊销「作弊」的计数——`slider.py` 的 docstring 自己承认
  「the two currencies disagree and both are reported」，而「严格占优」**只在 forwards 口径成立、
  在 FLOPs 口径净亏**。
- **为什么是问题**：① 把方法学纪律塞回研究格（出资人判据禁止）；② 用一个自我定义、不代表真实算力的货币
  宣布「严格占优」，是**口径错误**；③ 这正是「在做工程而不是研究」的直接证据。
- **严重度**：致命
- **证据锚点**：`pathlm/slider.py:2,9,28-34,63-71,86`（自述两货币分歧、v1/WORKSPACE 口径）；
  `docs/methodology.md:49-56`；`archive/dropped/axes/G-policy/G2-currency/CRITIQUE.md`（②③判定为方法学）；
  `.scratch/15-slider-rung3/evidence/SUMMARY.md`（§2 flop 口径净亏、§3 两口径分歧）。

### P4 ·【致命】「严格占优点复现」是假结论——Δbpc 在方差带内，不可排序

- **问题**：G1 多处写「**严格占优点复现**」（`README.md:10,38,45`；`catalog.yaml:1018`）。
  证据里的两个数：`k=1 spec τ=0.98` bpc **2.0502** vs plain **2.0545** ⇒ **Δbpc = 0.0043**。
  而 `docs/methodology.md` 的方差带是 **0.014（Δ<0.014 不可排序）**。也就是说 bpc 侧**根本无法排序**，
  「bpc 更好」不成立；能被说的只有「forwards 更少、bpc 在噪声内持平」——
  而后者在 FLOPs 口径下是**净亏**（见 P3）。
- **为什么是问题**：把**带内差**当结论写进状态灯，是本项目自己方法论明令禁止的自欺
  （`methodology.md:19-20`）。它不是「达到」，是「没测出来」。
- **严重度**：致命
- **证据锚点**：`.scratch/15-slider-rung3/evidence/SUMMARY.md` §3（2.0502 vs 2.0545）；
  `docs/methodology.md:19-20`（0.014 带）；`axes/G-policy/G1-selector/README.md:10`；
  `.scratch/doc-log/axes/G-policy/G1-selector/CRITIQUE.md` ③-2。

### P5 ·【重要】G3/G4 是 G1 的实现手段，被误立为独立格（重复计数 + 一节两格）

- **问题**：G3（信号）与 G4（校准）在定义上就是「选择器用什么观测、观测可不可信」——是 G1 的**实现手段**。
  它们却各自领了 `judged` 状态灯与独立 verdict。出资人判据：手段不是子项目。
  连项目自己的清单都已承认：G3「信号是 **G1 的实现手段 → 并入 G1**」，G4「校准是 G1 的实现手段 → 并入 G1」。
- **为什么是问题**：① 违反「目的 vs 手段」判据；② 同一批实验（`.scratch/22`、`.scratch/15`、`.scratch/02`、`.scratch/16`）
  的结论被摊在三个灯上，**同一事实多个家**，重复计数；③ 让树结构看起来有内容，实际只有一个（未成立的）问题。
- **严重度**：重要
- **证据锚点**：`.scratch/54-subproject-schema/INVENTORY.md`（G1/G3/G4 自评）；
  `catalog.yaml:1029-1060`；`axes/G-policy/G3-signal/README.md:26`；`axes/G-policy/G4-calibration/README.md:26`。

### P6 ·【重要】G3 的判据「零额外前向」对它自己的信号 js 不成立，且成本未记账

- **问题**：G3 的 `judge` 列了「**零额外前向**」（`G3-signal/README.md:7,56`），
  但 js（路径分歧度）的定义要求 **≥2 条路径各做一次 forward**（`pathlm/gen_quality.py:317-318`
  硬校验 `n_paths<2` 抛错，`:331-336` 循环每条各跑一次），是在 **TF 重评分**口径下算的、
  **不是解码循环里的在线量**。而 `catalog.yaml` 给 G3 的 `cost` 只写「纯推理，分钟级」（`catalog.yaml:1126`），
  **完全没记 2× 前向成本**。
- **为什么是问题**：判据与机制自相矛盾（声明零成本、实际 2×），且违反「代价怎么评」的记账纪律。
  一个需要 2× 前向才能在线观测的信号，其代价若不计入前沿比较，结论必然虚高。
- **严重度**：重要
- **证据锚点**：`axes/G-policy/G3-signal/README.md:7,56`；`pathlm/gen_quality.py:317-318,331-336`；
  `catalog.yaml:1126`；`.scratch/doc-log/axes/G-policy/G3-signal/CRITIQUE.md` ③-2。

### P7 ·【重要】G4「校准跨规模存活 = 最稳资产」过强、口径混杂、无自有实验

- **问题**：G4 `verdict`「校准跨规模存活是本项目最稳的资产之一」（`catalog.yaml:1058`）。
  但：① ECE 0.002–0.011 **全部来自 TF（teacher-forcing）窗口**；② 所谓「跨规模」是
  **M0（1M、2 层 d64 探针）** 与 **100M（INT 系）** 两个**不同架构/不同任务**的读数并列，
  不是同一模型的缩放曲线；③ **自由生成段已失效**（prob0 压 0.99+ 天花板、分不开 real/random）；
  ④ G4 **无自有实验目录**（`G4-calibration/experiments/README.md:11`「本格暂无实验目录」），
  `judged` 全靠借用他格（`.scratch/02`/`16`/`15`），违反 SCHEMA 的实验目录契约。
- **为什么是问题**：把「一个口径（TF 窗口）上的一个读数」升格成「最稳资产」，是同文件 failure 自相拉扯的过强措辞；
  而且「不同模型的各自读数」不是「跨规模存活」的证据。校准好本身也不证明选择器有质量收益（那是手段）。
- **严重度**：重要
- **证据锚点**：`axes/G-policy/G4-calibration/README.md:10-11,44,48-49`；`catalog.yaml:1058`；
  `axes/G-policy/G4-calibration/experiments/README.md:11`；
  `.scratch/doc-log/axes/G-policy/G4-calibration/CRITIQUE.md` ②③。

### P8 ·【重要】G1 无自有实验目录；evidence 与已删的 F3-stopping 完全重合

- **问题**：G1 的 `experiments/` 下四个目录（2508/2514/2515/2517）的 README **全部由
  `tools/backfill_legacy.py` 生成**、指向 `.scratch`，没有自包含的 `run.py / results.json / analysis.md / MODEL.md`，
  不满足 SCHEMA §3 的「每次判决一个自包含子目录」。且 G1 的 evidence
  （`.scratch/15`、`.scratch/17`、`.scratch/14`）与已并入/删除的 F3-stopping **完全重合**，
  结论句几乎一样 ⇒ **同一批实验被写进两个格的状态灯**。
- **为什么是问题**：① 不满足实验目录契约，结论不可独立复现；② 重复计数，树结构失真。
- **严重度**：重要
- **证据锚点**：`axes/G-policy/G1-selector/experiments/2515-slider-rung3/README.md`（末尾「由 backfill_legacy.py 生成」注释）；
  `.scratch/doc-log/axes/G-policy/G1-selector/CRITIQUE.md` ③-2（两格同证据）；
  `archive/dropped/axes/F-iteration/F3-stopping/`。

### P9 ·【重要】选择器的核心假设在 100M 失效，却没有升格为 G1 的结论/否证

- **问题**：G1 唯一自我声明的判据是「**每 FLOP 边际收益**」（`README.md:7,55`）——即「低置信位置每 FLOP 收益更高」。
  实测：13M 门控重试杠杆 −0.0043（带内，已不显著）→ **100M ≈0，τ 轴平坦 ±0.002**；
  且 100M 出现「轮次本身有价值（forced-round r1→r2 −0.066）但**选择性门控**低置信带（fire~7%）内价值 ≤0.001」的分离。
  也就是说 **G1 的机制假设在 100M 不成立**。但这被写进 `.scratch/17` 的「工程叙述」，
  G1 的 verdict 里只轻描淡写列进 failure，没把它当成「这条线的核心命题被否证」。
- **为什么是问题**：假设失效是最重要的研究结论，却被埋在工程叙述里；状态灯因此显得比实际乐观。
- **严重度**：重要
- **证据锚点**：`.scratch/17-slider-at-scale/evidence/SUMMARY.md` 判读 2；
  `axes/G-policy/G1-selector/README.md:7,48-49,55`；`catalog.yaml:1013-1014`。

### P10 ·【重要】项目「卖点从质量改为可控性」——目的已被放弃，剩下的是手段/副产品

- **问题**：`2508-tts-probes` 的失败路径明写：「**没有任何组合打赢 B0 → 项目卖点从「质量」改为「可控性」**」。
  此后 G 轴的全部 KPI 变成「旋钮可预测」（成本预测误差、fire split gap、开火率可预测性）——
  这是**工具的可用性**，不是研究结论。「可控性」本身还是手段。
- **为什么是问题**：出资人判据明确禁止把手段当目的。目的（质量/前沿）失败后，正确动作是**否证并关闭**该方向，
  而不是把目的换成「工具好用」继续维持状态灯。
- **严重度**：重要
- **证据锚点**：`axes/G-policy/G1-selector/experiments/2508-tts-probes/README.md`（失败路径与原因）。

### P11 ·【重要】G1 判据把「目的」与「工具 KPI」混成一列

- **问题**：G1 判据为「**每 FLOP 边际收益 + 选择器的校准（ECE）+ 开火率可预测性**」（`README.md:7,55`）。
  只有第一项是目的（前沿）；后两项分别是「信号可信度」（手段）与「工具可预测性」（工程 KPI）。
- **为什么是问题**：把手段与工具 KPI 塞进判据，等于允许用「工具好用」来「证明」研究有价值——
  这正是 P2 的病根。判据应是单一的：**同平均算力下质量（或同质量下算力）相对最优固定路径的比较**。
- **严重度**：重要
- **证据锚点**：`axes/G-policy/G1-selector/README.md:7,55-60`；`catalog.yaml:1010`。

### P12 ·【重要】与项目主线「信息右移」的定位混乱；G 轴自造了一套词

- **问题**：项目主线目的是**信息右移**（前缀可丢、AR 仍成立）。G 轴（策略/调度/自适应算力）与「信息右移」
  不是同一件事——它更接近「**省算力 / 按预算分配**」族。但 catalog 把它当一个横切轴并列，
  且 G1 被反复与 F3-stopping、已删的 G5 多线、投机线纠缠。同时 G 轴自造了
  「策略与调度 / 选择器 / 观测信号 / 校准」这套词，没有对齐出资人要求的统一表述。
- **为什么是问题**：出资人明确要求「用出资人的词（信息右移等），不要自造说法」。G 轴既没对齐主线，
  也没有写出自己的目的陈述（例如「固定预算下最大化质量」），于是退化成手段集合。
- **严重度**：重要
- **证据锚点**：`catalog.yaml:142-155`；`.scratch/54-subproject-schema/INVENTORY.md:22-24`（出资人要求用「信息右移」）；
  `.scratch/51-tree/WORK_ITEMS.md:55-57`（主线一是信息右移，G 不在其列）。

### P13 ·【次要】上一版本 / 上一时代残留（要求「完全 migrate 掉」）

- **问题**：确认存在下列残留：
  1. `pathlm/slider.py` 的 docstring 引用 **`docs/mental_model.md`**（§3/§1）与 **`WORKSPACE` convention /
     `v1 budget currency`**——而 `docs/mental_model.md` **已不存在**（现位于 `archive/legacy-docs/mental_model.md`），
     `WORKSPACE` 是更早时代的文档名。`slider.py:2,9,29,71,86` 均带此残留；
     `.scratch/14-allocator/evidence/composition/frontier_INT2.json` 的 `notes` 也引 `docs/mental_model.md §4`。
  2. `catalog.yaml:149` 事实文档里仍写「**原 G2**」（G2 已删）；`G1-selector/README.md:81` 仍写「**原 F3-stopping 已并入**」。
  3. 事实文档 `axes/G-policy/PRIOR_ART.md` 正文带「原 `G2-currency`」「原 `G5-parallel-lines`，已删」等**变更叙事**，
     违反「事实文档只写当前状态、变更史归 doc-log」的规矩（doc-log 本已有 `PRIOR_ART_ENGINE.md`）。
  4. `catalog.yaml:1049-1050` 重复注释 `# ==================== G4`（连续两行）。
- **为什么是问题**：出资人明确要求把上一版本/上一时代的内容**完全 migrate 掉，不留垃圾**。
- **严重度**：次要
- **证据锚点**：`pathlm/slider.py:2,9,29,71,86`；`docs/mental_model.md` 不存在（实际在 `archive/legacy-docs/mental_model.md`）；
  `catalog.yaml:149,1049-1050`；`axes/G-policy/G1-selector/README.md:81`；`axes/G-policy/PRIOR_ART.md`（正文叙事）。

### P14 ·【次要】G3 的强结论（AUC 0.98「正交栈」）证据薄

- **问题**：G3 的 `benefit` 写「distinct-2 管塌缩（**AUC 0.9815**，与 js 正交 rank_corr 0.14）、三信号正交监测栈成形」
  （`catalog.yaml:1137`）。但该 AUC 与「正交」都来自**同一批 24 个窗、单权重、单解码（greedy）、
  人工标注由子代理完成（非独立第三方）**，且预登记阈值规则**退化为「全部塌缩」**（连续谱无自然间隙）。
- **为什么是问题**：把一个 n=24、单权重、非独立标注的读数当跨模型/跨解码的稳定能力，
  并把「正交栈成形」当硬结论，证据偏薄（作者在原始证据里已诚实标注，但 catalog 的口径没有降级）。
- **严重度**：次要
- **证据锚点**：`catalog.yaml:1137`；`.scratch/22-gen-quality/evidence/SUMMARY_22b.md`（`label_source=manual`）；
  `.scratch/doc-log/axes/G-policy/G3-signal/CRITIQUE.md` ③。

### P15 ·【次要】G1 的 option 枚举与结论/其它轴混叠

- **问题**：G1 的 option 枚举为「固定 / 输入无关阈值 / 逐 token 阈值（prob0 门控）/ 学习门（carry gate）/ 离线求解器（Slider）」
  （`README.md:30-34`）。其中：① `学习门（carry gate）` 是 M/C 轴的**搬运通道机制**，不该算「选择器」；
  ② `逐 token 阈值` 与 `离线求解器` 不是并列选项——求解器就是**离线把逐 token 阈值的 τ 定出来**，
  两者是同一机制的两层（离线定参 + 在线门控）。
- **为什么是问题**：枚举混叠会让「哪些选项已测」失焦，也让「选择器」概念被别的轴的机制污染。
- **严重度**：次要
- **证据锚点**：`axes/G-policy/G1-selector/README.md:30-34`；`catalog.yaml:1009`。

### P16 ·【次要】G1 关于「选择性判据是否已跑」自相矛盾

- **问题**：G1 的 `verdict` 说按原预登记判据（分桶单调 + 低置信桶 ≥ 高置信桶 2×）「**尚缺…是否获证（待补做）**」
  （`README.md:11`）；但 `2514-allocator-selectivity` 的 README 又写「收益集中在中低置信带（+0.04~+0.06）、
  高置信带 ≈0/负」——看起来已经测过选择性。到底跑没跑，两处口径不一。
- **为什么是问题**：同一判据在目录里出现「已测」与「待补做」两种状态，破坏 SSOT。
- **严重度**：次要
- **证据锚点**：`axes/G-policy/G1-selector/README.md:11`；`axes/G-policy/G1-selector/experiments/2514-allocator-selectivity/README.md`；
  `.scratch/14-allocator/issues/01-step1-completion.md`（实验 1(2) 的判据定义）。

### P17 ·【次要】「严格占优」的对照不是「最优固定路径」，判据取向不干净

- **问题**：G1 把 spec `k=1` 与「plain（k=0, retry off）」对比来宣布占优。但 `k=1` spec 本身是**另一条机制**
  （投机解码 / H 轴），把它混进「选择器」的占优叙事里，对照就不再是「同一个选择器 vs 最优固定路径」，
  而是「投机解码 vs no-op」。并且口径是 forwards 而非端到端真实算力（见 P3）。
- **为什么是问题**：违反 `methodology.md:16`「与自循环比」——对照必须是同机制下的成本/无选择，而不是换一个机制。
- **严重度**：次要
- **证据锚点**：`.scratch/15-slider-rung3/evidence/SUMMARY.md` §3（对比表）；`docs/methodology.md:16`。

### P18 ·【次要】`check_schema.py` 对「无自有实验的 judged 格」不拦截

- **问题**：G3/G4 均 `judged` 但 G4 无实验目录、G3 只有 backfill 存根；`tools/check_schema.py` 只校验
  「status=judged 必须有 judge 与 evidence、evidence 路径存在/锚点合法」（`check_schema.py:7-10,148-167`），
  **不校验是否有自有实验目录**，于是这类「借证据」格可以长期绿过 CI。
- **为什么是问题**：机器校验的缺口让「结论无实验支撑」无法被自动发现，只能靠人工复核。
- **严重度**：次要
- **证据锚点**：`tools/check_schema.py:7-10,148-167`；`axes/G-policy/G4-calibration/experiments/README.md:11`。

---

## 2. 一句话总结

**G 轴（G1/G3/G4）把「自适应算力分配」这条本应有明确目的的线，做成了「一套可预测的调度工具」：
topic 只剩手段（选择器/信号/校准），结论只剩工具自洽 KPI（成本预测准、ECE 低、开火率可预测），
唯一的「严格占优」是方差带内的 Δbpc=0.0043，而选择器的核心假设（低置信位置每 FLOP 收益更高）
在 100M 已失效却未升格为否证。信号/校准被误立为独立格（应并入 G1），双货币记账是方法学而非研究，
并残留大量上一版本痕迹（mental_model.md / WORKSPACE / 原 G2 / 原 F3 / 原 G5）。**

**最致命的问题**：P1+P2+P3+P4 的组合——**没有目的、没有端到端 pareto 结果、把方法学/工程当研究、
用带内差宣布占优**。任何一条成立，该格就不该带状态灯；四条同时成立，说明整条 G 线需要**重开目的、
重设判据（质量-算力前沿）、把 G3/G4 并入 G1、并把 Slider 降到共享底座/方法学**。
