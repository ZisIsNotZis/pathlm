# F（迭代）轴 · 独立批判性问题台账

> 复核者：fresh context 独立复核（未参与 F 轴任何执行/写作）；只读 + 只写本文件。
> 范围：`axes/F-iteration/**`、`catalog.yaml` 的 F 段（F 轴 + F2 + F4）、相关 `.scratch/`。
> 基准：本 checkout 工作树（HEAD `ce2610a`；工作树有他人未提交改动，未触碰）。
> 方法：**先按第一性原理独立想清楚，再读代码/文档核对**；对既有 `CRITIQUE.md`/审计的结论
> 逐条独立验证，**不采信其说法**（下面标了「确认 / 修正 / 新增」）。
> 严重度：**致命** = 结论/结构不成立，必须重做或删除；**重要** = 会导致错误判断或返工；**次要** = 措辞/归属/可读性。

---

## 0. 先想后看：我的独立判断（写在读代码之前）

**这条线想解决什么问题（痛点）。** 语言模型每 token 的算力是「一次性、固定深度、每层不同权重」。
「迭代」问的是：**在同一个状态上多花几遍算力，能不能换到质量/灵活性**——即把「算力路径」变成
可重复、可按需追加的东西。真正的下游目的只可能是三类之一：
(a) **同参数下用更多算力换质量**（compute–quality 前沿，把「Pro 当 Flash 用」反过来）；
(b) **损坏/低质输入的修复**（鲁棒性）；
(c) **信息右移**（把旧位置的信息搬到当前，让前缀可丢）。
**这三类的判据完全不同**，混在一起就是瞎研究。

**research topic 应该是。** 一句可证伪的话，形如：「对同一状态重复施加一个（共享/回传的）算子、
并配一个让重复有效的信号，其**质量-算力前沿**严格优于把同样算力花在别处（加深度/加参数/加窗口）」。
**不是**「我们有哪几种回传方式」。

**应该引入的机制。** 一个**携带状态**（carried state）+ 一个**每轮注入新信息的算子**
（无注入 = 重入退化为固定点，这是 Universal Transformer/DEQ 的公论）+ **一个决定何时停/是否值得
再来一遍的判据**（否则多算力无意义）。训练侧要么让梯度穿过重复（unroll/回传），要么用辅助目标
补上「写-读」的梯度缺口——后者必须**配一个真正的载体（瓶颈）**才成立。

**代价该怎么评判。** 只能用 **pareto frontier**：横轴 = 实测算力（FLOPs 与墙钟/显存分开记，
含每轮额外开销），纵轴 = 质量（+ 目标能力）。参照系 = **全量标准模型 = 100%**；对照必须有
「同算力的加深度」「同参数的加宽/加层」「不迭代但同 FLOPs」三类，否则测到的只是「多花算力当然
更好/更差」。**绝对无损不存在**，要找的是前沿形状与斜率。

**巨人肩膀。** 同状态重复 = Universal Transformer(2018) / recurrent-depth(2024) / Huginn；
回传/门控 = DEQ、gated recurrence；软/硬潜变量 = VQ/straight-through；停止 = ACT(2016) /
PonderNet / CALM(2022)；分段多遍/去噪 = Diffusion-LM/SEDD/Plaid；权重迭代 = fast weight/TTT；
训练信号 = Compressive Transformer compression loss(2019) / AutoCompressors / ICAE / RECOMP / DMC；
候选集 = best-of-n。**F 轴这两格（F2/F4）几乎整条都在这张名单上**。

**哪种更可能有效。** 「多一遍 + 有外部新信息注入 + 显式载体 + 按需开火」比「裸回传几何」
更可能有效；「重建损失」只在有压缩瓶颈时才有意义。

**现有方法对不对？** 读完代码/证据后的判断见下。结论：**方向没错、topic 错位、判据不可判定、
证据链断裂、且大量内容属于上一时代引擎**。

---

## A. 目的 / topic 层（最致命）

### F-01【致命】F 轴与 F2/F4 都没有「目的」，命题本身就是一张手段清单
- **问题**：F 轴 `claim` =「对什么重复、用什么通道回传、何时停、靠什么训练信号让重复有效」
  （`catalog.yaml:113`）——四个都是**手段**；`subdims`（`:117`）、格子页 §1 命题（F2 在
  `catalog.yaml:854`、F4 在 `:880`）同样只写手段。整条轴**没有一句「为什么值得重复计算」**。
- **为什么是问题**：出资人的判据「只列手段、没有目的 = 瞎研究」。目的缺失直接导致：判据无法选
  （不知道该对什么涨）、对照组无法选、成本口径无法定、结论无法判定值不值。这不是措辞问题，是
  研究从根上没被定义。
- **根因（结构性）**：课题页模板**根本没有 purpose 字段**。`tools/build_wiki.py` 生成的九节
  只有「命题/选项/状态/结论/失败/判据/成本/依赖/复现」（`tools/build_wiki.py:106-166`），
  `catalog.yaml` 的必填键也没有 purpose（`tools/build_wiki.py:53`）。⇒ **任何格都写不出目的**，
  F 轴只是最极端的受害格。出资人已在 `ticket 54` 提出新 schema 含
  `purpose/innovation/expected_effect/validation`（`.scratch/54-subproject-schema/TICKET.md`），
  但 F 轴现状仍是老模板。
- **证据锚点**：`catalog.yaml:113,117,854,880`；`tools/build_wiki.py:53,106-166`；
  `.scratch/54-subproject-schema/INVENTORY.md:140-149`（自评也承认「回传/训练信号是手段，我没写假设」）。

### F-02【致命】F2/F4 的「目的」在三处互相矛盾，且没有一处写进 SSOT
- **问题**：同一批格子被三个不同目的各自表述：
  1. **项目树**把它归入「主线一 · 信息右移（把旧位置信息搬到现在）」
     （`.scratch/51-tree/WORK_ITEMS.md:55-57,145-174,533-534`：WI-05=F2、WI-06=F4 都标
     「信息右移」）；
  2. **F 轴自身定义**是「原语 I（在同一状态上多花算力）」（`catalog.yaml:112,114-116`）——
     与「信息右移」是两码事；
  3. **历史实际目的**是「损坏输入**修复**」——`archive/legacy-docs/findings.md:58-78`
     明写「retry 是**修复**不是集成」，`docs/playbook.md:64` 把 retry 归为
     「回环 retry / corruption 族」。
- **为什么是问题**：同一格三个目的 ⇒ 任何结论都无法被正确引用。「回传几何无价值」在「修复」目的下
  只说明「修复不需要挑通道」，**不能**推出「信息右移不需要挑通道」（信息右移要的是跨窗口搬运，
  机制不同）；反之亦然。这是「结论与目的错配」的根。
- **证据锚点**：`catalog.yaml:112-117`；`.scratch/51-tree/WORK_ITEMS.md:55,145-174,533-534`；
  `archive/legacy-docs/findings.md:58-78`；`docs/playbook.md:64`。

### F-03【致命】F2/F4 共用同一个「让迭代/搬运有效」的问题，但机制归属分属 C 轴与 M 轴
- **问题**：出资人/主 Agent 的倾向是「F2+F4 合并」（`.scratch/54-subproject-schema/INVENTORY.md:144,149`；
  `ticket 55` §2）。但独立看机制：
  - **F2（回传通道）**的唯一主题 = legacy 引擎的 **latent retry**（同 token 状态重入），它是
    **深度/自适应算力**（C 轴）的旋钮：重入门控就是 G1 的 selector，`deps=[G1-selector]`
    （`catalog.yaml:870`），且与 dense-exit 整形**拮抗**（`docs/playbook.md:54`；F3 干脆已并入
    G1，`.scratch/51-tree/OPERATIONS.md:233`）。
  - **F4（训练信号）**的唯一实验（2619）是「**无载体 dense 上，训练信号能不能替代显式通道搬信息**」，
    这是 **M1 载体线（信息右移）**的子实验：它的对照/参照全是 M1 的 carrier（`2619/analysis.md` §10
    用 `2618` 的 delta 写通道当参照），主 Agent 也把 WI-06 的依赖写成「WI-01 的显式载体就位」
    （`.scratch/51-tree/WORK_ITEMS.md:174`）。
  ⇒ **按现状合并 = 把 C 轴的旋钮和 M 轴的载体实验塞进同一个子项目**，反而制造新的错误归属。
- **建议（供裁决）**：F2 的真实家是「深度/迭代算力分配」（并入 C-vertical 或 G1，其负结果
  「无注入重入无新信息」当边界条件）；F4 的真实家是「信息右移」（并入 M1，作为「载体 × 训练信号」
  的选项）。**F2+F4 不是重复**，它们只是**都没有自己的目的**，所以看起来像重复。
- **证据锚点**：`catalog.yaml:870,895`；`docs/playbook.md:54`；
  `.scratch/51-tree/OPERATIONS.md:233`；`.scratch/51-tree/WORK_ITEMS.md:145-174`；
  `axes/F-iteration/F4-training-signal/experiments/2619-reconstruction-loss-13m/README.md:11`
  （「载体是无显式记忆槽的残差流」）。

---

## B. 判据 / 口径层

### F-04【致命】F2 的判据「每轮 bpc 边际」落在项目自定的不可排序带里 —— 判据不可判定
- **问题**：F2 `judge` =「每轮 bpc 边际 + 校准（ECE）+ 是否结构性 no-op」（`catalog.yaml:856`）。
  但本仓自己规定：**Δ < 0.014 bpc 不可排序**（`docs/methodology.md:19`），而**训练侧 bpc 噪声
  ~0.0156 已超该带**（`2619/README.md` §4「禁止」条、`2619/analysis.md` §3）。逐轮**边际**比
  总效应更小，必然落在噪声里。
- **为什么是问题**：判据一开始就不可判定 ⇒ 无论跑出什么都会是「不可排序」，等于没判据。而且
  F4 的实验自己写了「**禁止**把 bpc 当主判据」，F2 却仍用它——**同轴两格口径互斥**。
- **证据锚点**：`catalog.yaml:856,882`；`docs/methodology.md:17-26`；
  `axes/F-iteration/F4-training-signal/experiments/2619-reconstruction-loss-13m/README.md`（§4 禁止条）。

### F-05【致命】F2 判据完全不含「成本」，而 F 轴的全部意义就是「多花算力」
- **问题**：F2 判据只有质量侧（bpc/ECE/no-op），**没有** forwards/FLOPs/墙钟/显存任何一项；
  F2 的 `cost` 字段只是「单臂跑多久」的实验室预算，不是每 token 增量成本。
- **为什么是问题**：迭代研究的产出物必须是 **质量-算力前沿**（同参数下每轮多花多少、换多少质量）。
  没有成本轴 ⇒ 无法回答「这遍算力值不值」，只能得到「质量变好/变坏」，而那在多花算力下几乎必然。
  对比：F1（已归档）的判据还写着「成本（forwards/FLOPs）」（`archive/dropped/axes/F-iteration/
  F1-iteration-object/README.md` §6），**并入 F2 时把成本口径丢了** —— 这是迁移造成的倒退。
- **证据锚点**：`catalog.yaml:856,857`；`archive/dropped/axes/F-iteration/F1-iteration-object/README.md`（§6）。

### F-06【致命】F2 的税字段引用的是**已被本仓宣布作废**的引擎修订
- **问题**：F2 `local.tax` =「通道税在 F1 的 C 族内（+0.146…+0.178）」（`catalog.yaml:862`），
  这是 legacy 引擎上 retry 矩阵的 C 族数字。`ticket 11` 自己写：D1（gauge 缺陷使
  soft>linear>direct 是「cap 破坏量」的单调）、D3（训练从不采到 r≥2 ⇒ 累加器从未在它起作用的地方
  训练过）修完后，**「no retry number in findings.md survives as-is」**，所有 retry 数必须重测
  （`.scratch/11-retry-matrix-validity/issues/01-retry-matrix-validity.md`，D1/D3 段与
  「IMPORTANT scoping note」）。
- **为什么是问题**：① 数字作废；② 更根本的是**命名错**：「通道税」其实是「多跑一遍 retry 的税」
  （C 族 = transport 变体），把它标成「回传通道的税」会让读者以为不同通道本身有不同代价——而
  D1 恰恰证明测到的排序是钳位的产物，不是通道的代价。
- **证据锚点**：`catalog.yaml:862`；`.scratch/11-retry-matrix-validity/issues/01-retry-matrix-validity.md`
  （D1/D3/D6 + scoping note）；`archive/legacy-docs/findings.md:60-65`。

### F-07【致命·新增】F4 的主判据腿正是仓库自己「永久降级」的分布外腿 —— 直接违反 SCHEMA §17.18
- **问题**：2619 的 c0/c1/c2（主判据）用的是**单查询**关联回忆，`results.json::premise_check.
  query_key_distance` = `min 241 / median 247 / max 255`。而仓库在 M1 已实测并判：**「单查询关联回忆、
  距离 241–255、超训练上限 240」的腿，同配置 n=3 独立重训极差 10.840pp ≥ 结论效应 3pp ⇒ 永久降级为
  辅助腿**（`axes/M-carrier/M1-carrier-type/README.md:94`；`SCHEMA.md:327-333` 就是为此立的 §17.18）。
- **为什么是问题**：主判据的效应（预登记 +3pp）**远小于该腿自身的重跑方差（10.84pp）** ⇒
  无论读数是 −0.244pp 还是 +2pp 都不可解释，**「未过」这个结论本身不可信**；而 catalog 却把它当
  「13M/12k 上不成立」入账（`catalog.yaml:893`）。既有 F4 `CRITIQUE.md` 只提了 n=1 与重跑落坑，
  **没有**把这腿与 §17.18 的永久降级条款对上——这是本条的新增发现。
- **证据锚点**：`axes/F-iteration/F4-training-signal/experiments/2619-.../results.json::premise_check`；
  `axes/M-carrier/M1-carrier-type/README.md:94,30,36`；`SCHEMA.md:327-333`；`catalog.yaml:893`。

### F-08【重要·新增】F4 训练用「多查询」布局、判决却用被替换掉的「单查询」布局
- **问题**：2619 训练命令是 `--synth-task assoc-mqar --synth-pairs 8 --synth-queries 8`
  （`2619/README.md` §9），即 `assoc_recall_multi_batch`（Q=8，dist ≤ 240）；但主判据腿在
  `run.py:218,446` 调的是 `assoc_recall_batch`（**单查询** Q=1，dist 241–255）。
- **为什么是问题**：`tools/evalkit/longrange_tasks.py:502` 的 docstring 明确写：多查询版是
  **为训练侧**设计的，理由是单查询版「每行只有 1 个真正依赖记忆的监督 token ⇒ 12k 步里通道拿到的
  梯度份额约 1/255」= **信号饥饿**（会把「训练信号无效」测成假失败）。**用被判定会饿死信号的布局
  去判决一个「训练信号」实验**，是最不该犯的错；同时训练/判决布局不一致也让「同任务」的说法不成立。
- **证据锚点**：`2619/README.md` §9；`2619/run.py:218,446`；`tools/evalkit/longrange_tasks.py:502`
  （docstring「**训练侧构造**」「信号饥饿」）；`axes/M-carrier/M1-carrier-type/README.md:94`（分布内腿
  应取多查询、距离 107–240）。

### F-09【重要】F4 把「假设」当「结论」写进 SSOT，且实验文档自己已降级
- **问题**：`catalog.yaml:893` 写「同任务同尺度**显式写通道**可达 0.0950（+8.13pp）⇒ **「给路径」优于
  「给信号」**」——这是**结论口气**。但 2619 自己的 `analysis.md:40-45,257` 明确：两条读数**不是等参数
  对照**（通道 +264,296 vs 解码头 +52,736），只能作**假设**（SCHEMA §15）。同一条还被复制到 A9 的
  verdict（`catalog.yaml:960`：「远优于」），错误被二次传播到另一格。
- **为什么是问题**：同一格内「结论」与「限定」自相矛盾；且违约的这句话是 F4 唯一的正面产出，删掉后
  F4 的 verdict 只剩「未过 + 一个不可解释的 OOD 腿」。
- **证据锚点**：`catalog.yaml:893,960`；
  `2619/analysis.md:40-45,257`；`SCHEMA.md:217-226`（§15 等参数对照）。

### F-10【重要】F2/F4 的「零经验/下一步」是占位符，违反 SCHEMA §0「有内容的空」
- **问题**：F2/F4 的 §6「下一步判决实验」都只写「**干预：待定**」（`axes/F-iteration/F2-return-channel/
  README.md:65-67`；`.../F4-training-signal/README.md:57-59`），把 `judge` 原文抄一遍当「判据线」。
- **为什么是问题**：`SCHEMA.md:8-18` 规定 `todo` 格必须给出「选项枚举、判据、**判决实验设计**、成本」
  四件，且「未做的格子也必须给出**预测与判决实验设计**（不是留白）」。「待定」= 留白。
- **证据锚点**：F2 README:65-67；F4 README:57-59；`SCHEMA.md:8-18`。

### F-11【次要】「是否结构性 no-op」被当成判据线
- **问题**：F2 `judge` 里含「是否结构性 no-op」（`catalog.yaml:856`）——这是一个**实现正确性检查**
  （布尔），不是可判定的质量判据；把它与 bpc/ECE 并列写进判据线，会让「通道有效」被偷换成「通道不是 no-op」。
- **证据锚点**：`catalog.yaml:856`；`SCHEMA.md:140-156`（§10 活性判据本来就是**另一类**要求）。

---

## C. 结论 / 证据链层

### F-12【致命】「direct 重入 = 固定点（结构性 no-op）」是数学错误，被写成「全项目第一条铁律」
- **问题**：`catalog.yaml:865` 与 F2 verdict（`:859`）、轴 summary（`:119`）都写「直接回传是固定点
  （结构性 no-op）」。代码事实：`transport=="direct"` 时 `reentry = h`，随后仍执行
  `_run_layers(h, path)`（`pathlm/model.py:393-411` 与 `:441`），即把上一轮输出再喂进**该轮新采样的
  层路径** ⇒ 这是 `f∘f`，**一般 ≠ f**，不是「固定点」。M0 的原证据只是 **1M/2 层**探针上
  self-acc `0.1889 → 0.1897`（对照臂 `0.1891 → 0.1885`，±0.001，低于任何噪声带）
  （`.scratch/02-engine-m0/evidence/REPORT.md:15`），其「structural, not a bug」是**报告自己的断言**
  （同文件 `:23`），从未证明。
- **为什么是问题**：① 结论错：可站住的只是「direct 回传不注入外部新信息 ⇒ 在 M0 上第二轮无增益」
  （信息论意义），不是「结构性不可能」；② 被上升到「全项目第一条铁律」，污染了 A9/M1 等下游措辞；
  ③ 这条是 F2 的两个「已有」之一，抽掉它 F2 的 verdict 只剩一条作废的数字。
- **说明**：F2 `doc-log/CRITIQUE.md` 与 `.scratch/50-critical-audit/FMG-AUDIT.md:6-7` 已指过这点，
  **我独立确认代码事实**（`model.py:393-411,441`），但 SSOT（catalog/轴 README）**至今未改**。
- **证据锚点**：`pathlm/model.py:393-411,441`；`.scratch/02-engine-m0/evidence/REPORT.md:15,23`；
  `catalog.yaml:859,865,119`。

### F-13【致命】F2/F4 的 `evidence` 锚点**不指向产生数字的实验目录**，结论悬空
- **问题**：F4 verdict 引的是 `recon.acc_symbol 0.0049→0.9897`、两臂 `0.0137/0.0161`
  （`catalog.yaml:893`），但 F4 `evidence` 只有 `.scratch/02-engine-m0/evidence/REPORT.md` 与
  `.scratch/04-m1-runs/evidence`（`catalog.yaml:894`）——**都与重建损失无关**
  （`grep recon` 在 `04-m1-runs` 零命中）。真正产出这些数字的 `2619-reconstruction-loss-13m/` **不在
  evidence 里**。F2 同理：唯一实验 `2511-retry-matrix-withdrawal/` 不在 evidence 列表（`:869`），
  只靠 ticket 号间接指到。
- **为什么是问题**：`SCHEMA.md:61-79`/`§2` 要求 `judged` 格的数字有可复现锚点。现在任何人按 catalog
  都找不到这些数字的出处（数字只存在于 `.scratch` 的工单与一个未登记的实验目录里）。
- **证据锚点**：`catalog.yaml:893,894,869`；`find axes/F-iteration/**`（2619 完整存在，未被引用）；
  `grep -rl "recon" .scratch/04-m1-runs`（无）。

### F-14【重要】F2 的「实验」是一个没有数据的 backfill README，违反 SCHEMA §3 实验目录契约
- **问题**：`2511-retry-matrix-withdrawal/` 只有 `README.md`（front-matter `ticket: 11-...`），
  **没有** `run.py/config.json/results.json/analysis.md/MODEL.md/controls/`。文档自己标
  「本条目由 `tools/backfill_legacy.py` 生成」。
- **为什么是问题**：`SCHEMA.md:61-79` 规定实验目录六件套；没有原始数据/复现入口，这个「实验」不可复现，
  而它正是 F2 全部数字的来源（且数字来自作废引擎，见 F-06）。
- **证据锚点**：`axes/F-iteration/F2-return-channel/experiments/2511-retry-matrix-withdrawal/README.md`
  （全文）；`SCHEMA.md:61-79`。

### F-15【重要】F4 的结论建立在 n=1、且同配置两跑落入不同退化模式
- **问题**：2619 三臂各 n=1；`B`（0.013672）与 `B_rep`（0.000000）CI 不重叠，`analysis.md` §7 自认
  「同配置两次落在两个不同退化模式」，「有没有 1–2pp 的小效应在本尺度不可分辨」（§12-4）。
- **为什么是问题**：结合 F-07（该腿方差 10.84pp），整条 F4 结论**不可分辨正负**；catalog 却写
  「不成立」（`catalog.yaml:893`）。正确措辞应是「本尺度本腿不可判定」，并先换到分布内腿。
- **证据锚点**：`2619/analysis.md` §7,§12-4；`catalog.yaml:893`。

### F-16【次要】F4 把一个「带内差」当「已有结论」入账
- **问题**：F4 `local.tax` 写「consistency loss 中性（IX4 1.6589 ≈ C1 1.6549）」——Δ=0.0040 < 0.014
  不可排序带，「中性」结论成立，但它是 legacy 引擎（`configs/IX4.json`/`configs/C1.json`）上的点，
  且被当作 F4 的既有资产列出。
- **证据锚点**：`catalog.yaml:882`；`docs/methodology.md:17-26`。

---

## D. 造轮子 / 开倒车

### F-17【重要】F 轴主体是业界成熟路线，本仓自认 (a) 却仍留作待做研究格
- **问题**：`axes/F-iteration/PRIOR_ART.md` 自己给出分诊：F2 的「循环无输入注入退化」「软提交凸包」
  是 (a)（业界结论），F1 options 的整段去噪/候选集/隐式推理步/权重迭代是 (a)，F4 的重建损失是
  「2019 年就开线、2023–24 又热起来」的成熟路线（`PRIOR_ART.md` §1 各节与 §2 分诊表；`catalog.yaml:865`
  也把「无传输重入无价值」当「第一条铁律」，而 UT/DEQ 早就是公论）。
- **为什么是问题**：`SCHEMA.md:433-447`（§21）：(a) 类**不跑实验、直接写结论**，且**不得**进入
  `judged`；本格却把它们留在研究格/verdict 里当「本地发现」（「铁律」措辞）。
- **说明（修正既有审计）**：FMG-AUDIT 说 F2/F4 当前是 `judged`，**已过时**——现 `catalog.yaml:858,884`
  与格子页 front-matter 都是 `status: todo`。但 `.scratch/51-tree/OPERATIONS.md:86,88`、
  `PRIOR_ART.md`（「（`F2` 是 `status: judged`）」）仍写 judged ⇒ **状态三处冲突**（见 F-20）。
- **证据锚点**：`axes/F-iteration/PRIOR_ART.md`§1/§2；`SCHEMA.md:433-447`；`catalog.yaml:858,884`；
  `.scratch/51-tree/OPERATIONS.md:86,88`。

### F-18【次要】「记忆·未核」被当作「非空白格」的支撑，而 PRIOR_ART 自己说它不得当判据
- **问题**：`catalog.yaml:875,899` 的 `prior_art_note`/`notes` 用「记忆·未核」的业界前例支撑
  「非空白格/不得声称首创」；同一份 `PRIOR_ART.md` §0 写「一律标 `记忆·未核`，只作方向、**不得当判据**」。
- **证据锚点**：`catalog.yaml:875,899`；`axes/F-iteration/PRIOR_ART.md` §0；`SCHEMA.md:433-447`(c) 行。

---

## E. 上一版本 / 上一时代残留与结构

### F-19【致命】F2 的主题只存在于**上一时代引擎**，现共享底座里已不存在
- **问题**：F2 的全部对象（transport / retry / reentry / mixture）只在 `pathlm/model.py`（文档自述
  「PathLM model (**M0 skeleton**)」，`pathlm/model.py:1-14`）+ `train_m1.py` + `configs/C*.json`
  里；**现共享底座 `pathlm/depth_ar.py:3` 明写「纯 AR（无 corruption / 无 retry / 无 mask 站）」**。
  `docs/design.md:83-95` 的引擎迁移计划也「**尚未执行**」。
- **为什么是问题**：① F2 的 `cost`「13M 单臂 ~25 min」（`catalog.yaml:857`）暗示可在现天平上执行，
  实际不行（现底座没有 retry 站）；② F2 verdict 的「尚缺：…同批/同 seed/同 token 下…排序」是一条
  **需要重写引擎或复活旧引擎**的任务，被当成一次普通实验派发；③ 这是典型的「上一时代残留未 migrate」。
- **证据锚点**：`pathlm/depth_ar.py:3`；`pathlm/model.py:1-14`；`train_m1.py:1-20`；`configs/C1.json`；
  `docs/design.md:83-95`；`catalog.yaml:857`。

### F-20【重要】F2/F4 的状态、措辞在 catalog / README / OPERATIONS / PRIOR_ART 三处互相冲突
- **问题**：`catalog.yaml:858,884`、两格 front-matter = `todo`；`.scratch/51-tree/OPERATIONS.md:86,88`
  = `judged`；`axes/F-iteration/PRIOR_ART.md` =「（`F2` 是 `status: judged`）」。
- **为什么是问题**：`SCHEMA.md:19-42`/§1 只允许 `judged|todo`。同一格三个状态 ⇒ 任何「状态灯」读数
  都不可信，也直接违反「交接以目录下文档为准」。
- **证据锚点**：`catalog.yaml:858,884`；`axes/F-iteration/{F2-return-channel,F4-training-signal}/README.md:6`；
  `.scratch/51-tree/OPERATIONS.md:86,88`；`axes/F-iteration/PRIOR_ART.md`（分诊节）。

### F-21【重要】F2 verdict 要求「修 D6（gauge bug）」，但 D6 早已修好，且标错了缺陷名
- **问题**：`catalog.yaml:859` 写「**尚缺**：修 D6（gauge bug）后…真实通道排序（待补做）」。
  事实：D6 = 「linear transport 实为 `EᵀE`（放大 71.7×），不是投影」，现已改为真投影
  `pinv(U)@U`（`pathlm/model.py:165-172,271-306`，并 `VERDICTS.md`「D6 已修」）；而 gauge 缺陷是
  **D1**，也已修（同一函数共享 gauge）。⇒ ① 待办项已完成；② 「D6 = gauge bug」是错的（两个缺陷名混用）。
- **证据锚点**：`catalog.yaml:859`；`pathlm/model.py:165-172,271-306`；
  `.scratch/11-retry-matrix-validity/VERDICTS.md`（D6/D1 行）。

### F-22【重要】F2 的 options 把两个正交维度混在一起（F1 并入时留下的结构伤）
- **问题**：F2 `claim` =「一轮的结果**以什么形式回到下一轮**」（`catalog.yaml:854`），但 `options`
  （`:855`）里有 11 项，其中「同段多遍（AR 多轮）/整段去噪/候选集/隐式推理步/权重（TTT）」是
  **迭代对象**（原 F1），不是「回传形式」。备注也说「原 F1 已并入本格，迭代对象降为选项维」
  （F2 README 备注；`catalog.yaml:871-873`）。
- **为什么是问题**：把「重复施加在什么之上」与「结果以什么形式回来」压进一个 options 列表后，
  `judge`（每轮 bpc 边际）对「整段去噪/候选集」这类选项根本没有意义（它们不是逐轮重入）。
  这是「拆分不对」的实证：并入操作没给出新的判据，只制造了维度混淆。
- **证据锚点**：`catalog.yaml:854,855,871-873`；F2 README「备注」节。

### F-23【重要】F 轴把 G 轴/H 轴的资产算作自己的产出（资产越界）
- **问题**：F 轴 `local_summary`（`catalog.yaml:121-123`）把「**选择性门控成立**（低置信 15% 开火
  +0.128 vs 均匀 +0.057）、**预算求解器与双货币记账建成**」列为 F 轴保留资产——这两项是
  **G1-selector / G2-currency** 的资产（F3-stopping 已并入 G1，`.scratch/51-tree/OPERATIONS.md:233`；
  相关 funding 见 `catalog.yaml:1065` 指向 slider/rung 证据）；同段「soft 回传三度复现 OOD 污染」
  的锚点在 **H2**（`:866` 写「见 H2」）。
- **为什么是问题**：违反 AGENTS.md §2「每类事实恰好有一个家」。F 轴的「已收口」有一半是别人的产出，
  让 F 轴看起来比实际更接近交付。
- **证据锚点**：`catalog.yaml:121-123,866,1065`；`.scratch/51-tree/OPERATIONS.md:233`；
  `AGENTS.md` §2.1。

### F-24【次要】`deps` 字段与实验实际耦合不符
- **问题**：F2 `deps=[G1-selector]`（`catalog.yaml:870`）漏了与 **C 轴/深度**的已知拮抗
  （`docs/playbook.md:54`「整形杀 retry」，且 retry 门控=重入=深度相关）；F4
  `deps=[H3-composition, A9-old-info-retention, C6-depth-supervision]`（`:895`）漏了它实际依赖的
  **M1 载体**（WI-06 的依赖、2619 的参照通道都是 M1；`WORK_ITEMS.md:174`）。
- **证据锚点**：`catalog.yaml:870,895`；`docs/playbook.md:54`；`.scratch/51-tree/WORK_ITEMS.md:174`。

### F-25【次要】F4 `notes` 承诺「带检索腿的判据是我们的增量」，但实验从不测检索腿
- **问题**：`catalog.yaml:899-905` 的备注写「业界多数压缩工作不报『可重建性』⇒ **带检索腿的判据**
  仍是我们的增量」；但 2619 的主腿是合成关联回忆的 top-1 命中，**没有检索腿**（无 needle/RAG 腿），
  且 §12 声明「不跑 enwik8 腿」。
- **为什么是问题**：承诺的增量（检索腿）与实做不符，属判据与实验脱节。
- **证据锚点**：`catalog.yaml:899-905`；`2619/README.md` §12；`2619/analysis.md` §12。

### F-26【次要】F1/F3 的过程史归档位置与 F2/F4 不一致
- **问题**：F2/F4 的批判审计进了过程史家 `.scratch/doc-log/axes/F-iteration/**/CRITIQUE.md`，
  而 F1/F3 的 `PROBLEMS.md`/`CRITIQUE.md` 留在 `archive/dropped/axes/F-iteration/**`
  （事实树 `axes/**/PROBLEMS.md` 应迁 `.scratch/doc-log/`，见 `AGENTS.md` §2.2 与
  `.scratch/51-tree/PROBLEMS-TODOS.md` 头注）。
- **证据锚点**：`.scratch/doc-log/axes/F-iteration/**`；`archive/dropped/axes/F-iteration/{F1-iteration-object,F3-stopping}/`；
  `AGENTS.md` §2.2；`.scratch/51-tree/PROBLEMS-TODOS.md:1-10`。

---

## F. 命名 / 术语

### F-27【次要】用词与出资人词汇不统一，且 options 名与代码实现对不上
- **问题**：轴名「迭代/重复」，而项目主线词是「**信息右移**」；F2 options 的
  「soft 单纯形再量化 / 部分状态掩码 / hard 离散重嵌」在代码里对应的是
  `transport∈{none,direct,linear,soft}` + `token_retry` + `reentry_mix` + `retry_gate`
  （`pathlm/config.py:84-105`），名字对不上，无法逐项映射到可执行干预；options 还含
  「verifier 反馈」而**本仓不存在 verifier**（与 R5 审计结论一致）。
- **证据锚点**：`catalog.yaml:111,855`；`pathlm/config.py:84-105`；
  `.scratch/50-critical-audit/RP-AUDIT.md`（R5 条，本仓无 verifier）。

---

## G. 最致命的三条（给主 Agent 排序用）

1. **F-07（+F-08）**：F4 主判据腿是仓库自己**永久降级**的分布外腿（距离 241–255 超训练上限 240，
   n=3 重跑极差 10.84pp ≥ 3pp 效应），且训练用多查询、判决用单查询——**F4 的「不成立」结论不可信**，
   必须换到分布内多查询腿重做或撤下该结论（catalog `:893` 与 A9 `:960` 都要改）。
2. **F-01/F-02/F-03**：F 轴（含 F2/F4）**没有目的**，且同一批格子在三处被安上三个互斥目的；
   F2+F4「合并」会错把 C 轴旋钮与 M 轴载体实验塞进一格——**先定目的与归属，再谈合并/删留**。
3. **F-12/F-06/F-19**：F2 的两个「已有」——「结构性 no-op」（数学错误）与「retry 税」（作废引擎
   + 名不符实）——都站不住，且 F2 的主题只存在于上一时代引擎（`depth_ar.py:3` 无 retry 站）。
   F2 要么重做在现底座上，要么降为「已删除格」背景。

---

## H. 与既有复核的关系（区分确认 / 修正 / 新增）

- **确认**（我独立读到同样证据）：`doc-log/.../F2/CRITIQUE.md` 指出的 direct 非恒等、D6、税继承
  pre-D1/D3；`F4/CRITIQUE.md` 指出的 off-design（无载体）、非等参数、n=1、catalog 把假设当结论。
- **修正**：FMG-AUDIT 说 F2/F4 现为 `judged` —— 与当前 catalog 不符（现为 `todo`，见 F-20/F-17）。
- **新增**（既有审计未指出）：① F2/F4 缺目的是**模板层**缺陷（build_wiki 无 purpose 字段）；
  ② 三处目的互斥；③ F2+F4 的真机制归属分属 C/M（合并会造新错）；④ **F4 主腿是 §17.18 永久降级腿**；
  ⑤ 训练多查询/判决单查询的布局错配；⑥ F2/F4 的 `evidence` 指向错误的 `.scratch` 报告，实验目录未登记；
  ⑦ F2 的主题只存在于 legacy 引擎；⑧ F 轴把 G/H 资产算作自家产出；⑨ 状态三处冲突、D6 待办已完成却标错名。
