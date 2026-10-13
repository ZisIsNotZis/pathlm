# A 轴（序列混合机制 / 出资人命名：信息右移）独立批判性复核 · 问题台账

> 复核者：independent reviewer（fresh context，未参与该研究），`fork=false`，只读。
> 前提（出资人指令）：**「以该研究做的一塌糊涂为前提」**，先想后看，把**所有**问题写下来。
> 范围：`axes/A-mixing/**`、`catalog.yaml` 的 A 段（`axes.A` + `A2/A3/A4/A6/A8/A9/A10`）、相关 `.scratch` 证据。
> 本文件是**问题清单**，不是结论。未改仓库任何其它文件。
> 「严重度」= 致命（则 A 轴当前结论不可用）/ 重要（结论或方向有问题，需返工）/ 次要（措辞、结构、残留）。

---

## 0. 复核方法（先想，后看代码）

按出资人顺序执行：**读任何代码/文档之前**，先用第一性原理写下这条线应该是什么；写完才去核对仓库，
核对时**不以现有说法为准**（仓库文档会自证其说）。下面 §1 是读代码前的原始推理，§2 起是核对后的发现。

---

## 1. 独立第一性思考（读代码前写下，未参照仓库）

1. **要解决什么问题**：causal decoder 的长上下文推理中，KV cache / 每 token 计算随序列线性增长，
   且**历史信息被无差别保留**。痛点 = 长上下文推理的**成本随长度膨胀**，而其中大部分历史对当前
   预测无用。
2. **这条线的 research topic 应该是什么**（用出资人的词）：**信息右移**——如何**系统性地构造一种范式**，
   把前缀的信息**搬到右边**（写入可携带的载体/槽位），使得**前缀本身可以被物理丢弃**，而模型**仍是
   causal decoder**；损失不可避免，目标是**尽可能小**，且设计要 **explicit**（显式搬运），
   implicit 不算。
3. **应该引入什么机制**：**explicit 的写入/搬运/压缩**——专用 memory/summary 槽、跨步写的记忆矩阵、
   递归状态、显式检索，配合一个**迫使信息"必须被搬过去"的训练目标**（drop-invariance / 充分性损失：
   丢掉前缀后仍要能预测）。**滑窗掩码 / ALiBi 距离偏置 / RoPE / 稀疏核 / 线性核都不是右移**——
   它们只**衰减或丢掉**远端信息，不搬运、也不保证后缀自足。
4. **代价该怎么评**：**质量-代价 pareto frontier**。x 轴 = 推理代价（KV 字节 / 每 token FLOPs / 墙钟），
   y 轴 = 质量（相对**全量标准注意力 = 100%** 归一）。全注意力是右上角一个点；要找出在更低预算下
   质量最接近的可行点集合。**不是**去问"某手段是否无损"（无损不可能）。
   口径必须钉死：训练 context 与评测 context 是否一致？成本是训练成本还是**推理（增量解码）成本**？
   对照必须含：全注意力、纯窗口、以及"能保留全部信息"的 oracle 下界。
5. **巨人（既有设计）**：StreamingLLM / attention sinks、H2O、SnapKV、Compressive Transformer (Rae 2019)、
   Transformer-XL recurrence、Memorizing Transformer、Memory Transformer / RMT、Landmark Attention、
   gist tokens、AutoCompressors / ICAE、Infini-attention、Feedback Transformer、DeltaNet、
   RWKV/Mamba（线性递推）。问题 = 我们是在**这些肩膀上做 explicit 右移**，还是**重造**滑窗/ALiBi/稀疏/线性？
6. **哪种候选更可能有效**：**explicit 压缩 + 写通道 + 迫使使用的训练目标**最可能，因为它直接优化
   "丢前缀后预测仍准"，把损耗集中到一次可学的压缩，而不是逐 token 隐式衰减。
7. **现有方法对不对（预判）**：如果 A 轴整轴是 implicit 手段清单（window/anchor/sink/ALiBi/sparse/linear），
   它**根本没在研究所陈述的 topic**——它在研究"如何减少 attention 计算"（那是**成本手段**），
   **目的缺失**。判据若是 clean bpc（enwik8），那更测不到"信息是否被搬过前缀边界"（bpc 可被局部统计补全）。

> 以上为读代码前的判断。下面是核对结果——**基本被证实**，并另有若干结构性/口径/残留问题。

---

## 2. 致命问题（不修则 A 轴当前结论不可用）

### P1 · A 轴整体**丢失了目的**：维度定义里只有手段，没有「信息右移」
- **问题**：`catalog.yaml:26` 的 A 轴 `claim = 「一个位置从哪里、以什么算子、按什么寻址方式取信息。」`，
  `definition`（`:32`）= 「一个 token 能看见谁、用什么算子合并、怎么定位」。`subdims`（`:36`）= 窗口/偏置/
  算子/位置/旧信息/压缩触发——**全是手段**。整轴没有任何一句说"为什么要做"（让前缀可丢、AR 仍 causal）。
- **为什么是问题**：出资人判据第一条「目的 vs 手段：只列手段、没有目的 = 瞎研究」。A 轴的**定义本身**
  就把这条件限定成了手段清单，因此下面每一格都只能在手段层出结论（"窗口质量亏多少""ALiBi 有无净增量"），
  这正是"没目的所以测不到东西"的根源。
- **严重度**：**致命**。
- **证据锚点**：`catalog.yaml:26-45`（A 轴 block）；`axes/A-mixing/README.md`（由它渲染）。

### P2 · 「信息右移」这个 topic 在 A 轴**没有被当作 topic 研究**（目的-手段错位）
- **问题**：A 轴七个格研究的是**隐式的注意力成本/可见范围**：A3 硬窗+anchor、A4 距离偏置、A6 稀疏/线性
  算子、A8 位置编码、A10 驱逐优先级。**没有一格是 explicit 的"把前缀信息写入可丢弃载体"设计**。
  唯一 explicit 首验（记忆槽 + delta 写入）在 **M 轴 M1**（`catalog.yaml` M1 段、`.scratch/13-*`），
  A 轴只在 A9 的 failure 里引用它（`catalog.yaml:376` deps 也指向 M1/M2）。
- **为什么是问题**：出资人明令「需要 explicit 的信息往右移动的 design」。A 轴把"信息右移的**手段**"
  （窗口/偏置）当研究，却把 explicit 右移**放到别的轴**；等于 A 轴在研究**另一个 topic**（省 attention 算力），
  主线的落点不在 A 轴。
- **严重度**：**致命**。
- **证据锚点**：`catalog.yaml:214/247/277/330/383`；`axes/A-mixing/A9-.../README.md` failure 第 3-4 条；
  `catalog.yaml` M1 段。

### P3 · 判据用 clean bpc，**测不到「前缀可丢后信息还在不在」**
- **问题**：A2/A3/A4/A6/A8 的 judge 都是 clean bpc（`catalog.yaml:194/223/255/295/339`）。
  bpc 会被**局部统计补全**（本仓自己的 A4「假保留陷阱」：beyond-window 命中 90.3%，`catalog.yaml:262`）。
  用 bpc 判"信息右移成功与否"在原理上不成立。
- **为什么是问题**：能判"搬运"的判据是**丢前缀/删 KV 后的目标可测性 + 保留曲线 + 对无记忆地板/毁源对照**
  （只在 A9/2623、A10、M1 用了），A3/A4/A6/A8 **一律没用**。于是这些格即使"不亏"也证明不了信息被搬走。
- **严重度**：**致命**。
- **证据锚点**：`catalog.yaml:262`（假保留）、`axes/A-mixing/A9-old-info-retention/experiments/2623-window-on-longrange-13m/analysis.md` §0（win16 归零）、
  `axes/A-mixing/A6-mixing-operator/experiments/2636-longT-unified/analysis.md` §2（只用 bpc）。

### P4 · **没有 pareto frontier 口径**（出资人明确要求）
- **问题**：出资人要求「以全量标准注意力 = 100%，找出质量-代价前沿」。A 轴**没有**任何一处把质量 vs 预算
  画成前沿：A6 分别报 Δbpc 与成本三腿（`2636/analysis.md` §2/§5），但没有"同 KV/同算力预算下最优质量"
  的对照，也没有 100% 归一化；`pareto`/`前沿` 字样在 A 轴文件里**一次都没有**（仅 `catalog.yaml` 内
  W4/M2 段出现"Pareto/双边前沿"）。
- **为什么是问题**：没有前沿就没有回答"哪种设计更值"，只能得到孤立的"某手段不亏/亏多少"。
  出资人明说"不掉质量不可能，要找的是 pareto frontier"。
- **严重度**：**致命**。
- **证据锚点**：`axes/A-mixing/A6-mixing-operator/experiments/2636-longT-unified/analysis.md` §2/§5；
  `grep -rl pareto axes/A-mixing` = 空。

### P5 · 违反「不掉质量不可能」：仍在判「窗口/手段是否无损」
- **问题**：A3/A9 的判据写成"是否掉质量"（`A9-2615/analysis.md` 判据①、`catalog.yaml:360`）；
  A6 的 11 个实验（`2622`–`2637`）几乎全部在重判**"窗口相对全注意力亏不亏（Δ 是否越过 0.014 带）"**
  与**"窗口代价是否随 T 消失"**，结论是"n=1、两点可测地不同、**在本 n 下不可判定**"（`catalog.yaml:295`）。
- **为什么是问题**：花掉整批 GPU（2622–2637）重判一个 n=1、结论"不可判定"的"是否无损"问题，**决策价值≈0**；
  而它消耗的资源本应用于回答"前沿在哪"。这正是出资人批注 A6 的核心。
- **严重度**：**致命**。
- **证据锚点**：`catalog.yaml:295-315`（A6 超长 verdict）；`axes/A-mixing/A6-mixing-operator/experiments/{2622,2623,2624,2625,2626,2628,2629,2636}/`。

### P6 · 0.014 全局方差带**与其来源相反**，且被误用于**两臂之差**
- **问题**：`docs/methodology.md:19`「种子散布 ≤0.0138 ⇒ Δ<0.014 不可排序」。但该数的唯一来源
  `.scratch/11-retry-matrix-validity/evidence/README.md:69-71` 原话是「**There is no single global floor:
  ordering claims need per-cell replicates**」。且 0.0138 是**单臂**跨种子极差，而 A 轴判的是**两臂之差**
  （若独立，界应≈√2×0.0138≈0.0195）。
- **为什么是问题**：A 轴几乎所有"越界/带内"判定都紧贴这个界（A6 top-k `−0.0166`=1.19×、长T `−0.0158`=1.13×、
  A10 `+0.0159`=1.14×）；按两臂界读法，这些"越界"全部回落到带内。**整轴的排序结论建立在一个被自己来源
  否证的界上**。
- **严重度**：**致命**。
- **证据锚点**：`docs/methodology.md:19`；`.scratch/11-retry-matrix-validity/evidence/README.md:69-71`；
  `catalog.yaml:297`（top-k Δ）。

### P7 · A3/A4 两格的结论**没有仓内可独立复核的实验**，证据只指向 `.scratch` 与 legacy stub
- **问题**：A3 的证据是 `.scratch/07-x1-x3/evidence @7b47555`（不可复核的旧 markdown）、
  A4 的证据是 `.scratch/07-x1-x3/evidence/attn_dist_telemetry.json @7b47555`（`catalog.yaml:242/273`）。
  而 A3/A4 目录下的"实验"只是 legacy 回填 stub：`A3/experiments/2505-m1-followups/README.md`、
  `A4/experiments/2507-x1-x3-distance-penalty/README.md`，**无 `results.json`/`run.py`/`analysis.md`**。
- **为什么是问题**：A3 被当作"任何压缩/摘要求设计必须打败的价格上界"（`catalog.yaml:268` notes），
  这个上界数字却来自不可复核的旧 markdown；SSOT 无证据 = 不能当基线用。
- **严重度**：**致命**（对"上界"角色）。
- **证据锚点**：`catalog.yaml:242/273/268`；`axes/A-mixing/A3-window-anchor/experiments/2505-m1-followups/README.md`（stub）；
  `axes/A-mixing/A4-soft-distance-penalty/experiments/2507-x1-x3-distance-penalty/README.md`（stub）。

---

## 3. 重要问题

### P8 · A2「全注意力基线」仍是项目列表条目，且带 4 个 stub 实验目录（未 migrate）
- **问题**：出资人明示「基线不是项目，应去 scratch，不该在项目列表里」。现状：`catalog.yaml:190`
  仍是 `A2-full-attention`（`status: judged`），并挂了 `experiments/{2502,2504,2516,2519}/` 四个
  **只有 README 的 stub**（`catalog.yaml:196-204`）。
- **为什么是问题**：基线/参照进研究列表 = 结构错误；且它还在 `deps` 被别的格当"锚点"用，
  把"参照"伪装成"研究格"。
- **严重度**：**重要**（结构，出资人已明示）。
- **证据锚点**：`catalog.yaml:190-212`；`axes/A-mixing/A2-full-attention/experiments/*/README.md`。

### P9 · A3 标题即手段（窗口/锚点/sink），目的（信息右移）不在页面主位
- **问题**：A3 标题 `窗口 / 锚点 / sink`、claim「用硬性窗口限制可见范围」（`catalog.yaml:216-217`），
  `status: todo`。目的只在一句 note 里以"最接近右移压缩的**已测机制**"出现（`catalog.yaml:268`）。
- **为什么是问题**：出资人批注逐字：「窗口是手段，信息右移是目的……该项研究的目的是如何系统性地构造
  信息右移范式，使前缀可以丢弃」。现状 = 用手段当标题，把目的丢了；且出资人强调"结论不一定单一、
  要点是 explicit 右移 design"——A3 只测了 implicit 窗口，没有任何 explicit 方案。
- **严重度**：**重要**。
- **证据锚点**：`catalog.yaml:214-217, 268`。

### P10 · A4（ALiBi）是**手段**，其"本地无净增量"正是"没目的"的结果；且用错了 ALiBi 的口径
- **问题**：A4 标题 `软距离惩罚（ALiBi 式加性偏置）`（`catalog.yaml:248`），结论 `−0.0068`
  落在不可排序带内（`catalog.yaml:256`），出资人批：「手段本身有啥好研究的」。技术上加倍错误：
  ① 用**固定 w、全头共用**（丢掉 ALiBi 的 per-head slope ⇒ "全头被推向局部、长不出分工"是自设条件的
  必然，`catalog.yaml:263`）；② ALiBi 的真实卖点是**长度外推**，而 A4 只测**同分布 bpc**——
  那是 ALiBi 论文自己报的"中性"（无增量）。
- **为什么是问题**：把一个手段做成独立格、用错其核心口径、得到"无净增量"，是典型的**为手段而研究**。
- **严重度**：**重要**。
- **证据锚点**：`catalog.yaml:247-273`；`axes/A-mixing/A4-soft-distance-penalty/README.md` §1/§2/§5。

### P11 · A6「混合算子」：研究项/问题/假设不清，两个独立假设塞一格；「长 T」命名有歧义
- **问题**：出资人批：「你的研究项是啥？你的问题是啥，你的假设是啥？你瞎研究吗？」现状：
  A6 options 是 **10 种算子罗列**（`catalog.yaml:280-291`），无研究问题、无假设；把"内容选择稀疏"
  与"线性注意力"（两个独立假设）塞进一格；还塞进了已删格 A5/A7 的"选项维"。
  另：「长 T」（A6 大量使用，如 `2622/README.md:11`）其实是**序列长度**（`2622/run.py:268 T=spec["seq_len"]`），
  出资人质疑「T 是不是 Temperature、业界真用 T 吗」——命名非标准、与他处 T 冲突。
- **为什么是问题**：手段罗列 + 无假设 = 瞎研究；且用 T 表示序列长度会让结论不可读/易误读。
- **严重度**：**重要**。
- **证据锚点**：`catalog.yaml:277-291`；`axes/A-mixing/A6-mixing-operator/experiments/2622-longT-512/README.md:11`；
  `axes/A-mixing/A6-mixing-operator/experiments/2622-longT-512/run.py:268`。

### P12 · A6 把「位置窗口（A3 族）」当自己的主结论，机制错格
- **问题**：`2622`–`2629`/`2636` 的变量是 `train_window = T/2`（位置窗口），属 A3 族；
  catalog 现在**已自认**「机制属位置窗口（A3 族），在此作 A6 的同预算成本基线」（`catalog.yaml:305`），
  但 A6 的 verdict 仍大量以这些为结论（`catalog.yaml:295`）。
- **为什么是问题**：一格塞两条研究，且用别格的机制充当本格结论；正是出资人说的"拆分不对"。
- **严重度**：**重要**。
- **证据锚点**：`catalog.yaml:295-315`；`2622-longT-512/run.py`（唯一变量 `train_window`）。

### P13 · A8「位置语义」是手段（RoPE/插值），且本质是"长上下文工程前提"而非研究格
- **问题**：A8 标题/claim 是位置编码族（`catalog.yaml:330-338`），无目的；其真实价值是"长上下文线必须
  换 PE"这一**工程前提** + 解 ring-buffer 硬上界（`catalog.yaml:337`）。**本地零实验**
  （`axes/A-mixing/A8-positional-semantics/experiments/` 只有一个 `README.md`）。
- **为什么是问题**：同 A4 类——只写手段、没目的 = 瞎研究；且它已被业界做透（A8 自判 (a)）。
- **严重度**：**重要**。
- **证据锚点**：`catalog.yaml:330-346`；`axes/A-mixing/A8-positional-semantics/README.md` §1/§6。

### P14 · A9 与 A3 实质重复（同一族 + 同一判据）
- **问题**：A9 options = 保留/硬丢/锚豁免/压成 summary/递归状态/外置检索（`catalog.yaml:353-359`），
  A3 options = 滑窗/锚点/sink/全局 token（`catalog.yaml:218-223`）；二者判据都是"KV 消融保留曲线"
  （`catalog.yaml:224/361`）。purpose 也都是"旧信息如何处理"。
- **为什么是问题**：重复建格（出资人自评"并入 A3"）；且 A9 的"硬丢 +0.058~0.121"与 A3 的"价格上界"
  是同一件事。
- **严重度**：**重要**（结构）。
- **证据锚点**：`catalog.yaml:214-244` vs `:350-381`。

### P15 · A10 判据（ρ 曲线）≠ 实测量（驱逐优先级）；且与自家 PRIOR_ART 矛盾
- **问题**：A10 judge = 「保留曲线（固定压缩率 ρ 下的重建质量）+ 压缩率-质量曲线」（`catalog.yaml:389`），
  但实验量的是**同预算驱逐优先级三条规则对决**（`2615-priority-shootout`），**没有 ρ 曲线、没有重建**。
  notes 又写「业界无人占位（访问感知压缩）」（`catalog.yaml:416`），与自家 `PRIOR_ART.md` §A10
  （H2O/Scissorhands/SnapKV/PyramidKV/StreamingLLM）**直接矛盾**。
- **为什么是问题**：判据与实测错配 + 自相矛盾 = 结论不可托付。
- **严重度**：**重要**。
- **证据锚点**：`catalog.yaml:383-418`；`axes/A-mixing/PRIOR_ART.md` §A10。

### P16 · A10 判据门槛**算术不可达**（实验者已自认）
- **问题**：保留腿门槛 10pp，但全保留基线召回 0.6868，N=8 时 distance 已 0.6650，
  `0.6650+0.10 > 0.6868` ⇒ 门槛结构上达不到（`2615-priority-shootout-13m/analysis.md` §0/§6.1）。
- **为什么是问题**：判据在跑之前就不可达（无变化空间），跑出来的"未过线"没有信息量。
- **严重度**：**重要**。
- **证据锚点**：`axes/A-mixing/A10-compression-trigger/experiments/2615-priority-shootout-13m/analysis.md` §6.1。

### P17 · 整轴基本是**重造业界标准件**，且被判 (a) 后仍继续跑本地实验
- **问题**：窗口/sink（Longformer/BigBird/StreamingLLM）、ALiBi（Press 2022）、RoPE+PI/YaRN、
  稀疏（NSA/MoBA）、线性（Katharopoulos/Performer）、驱逐（H2O/SnapKV）——`PRIOR_ART.md` 逐格标了 (a)，
  却也逐个继续跑本地实验（A6 又跑了 11 个）：A6 的 top-k/线性、A3/A4 的窗口/偏置，全部是业界标准件。
- **为什么是问题**：出资人判据「是否站在巨人肩膀上、还是重新造轮子」；`SCHEMA §21`（不重复造轮子）
  与本轴的实际投入相反。**真正的巨人**（Compressive Transformer、Memorizing Transformer、RMT、
  AutoCompressors、ICAE、Infini-attention、Landmark Attention、gist tokens、DeltaNet）在 `PRIOR_ART.md`
  里只各占"`记忆·未核` 一行"，从未被当作 **explicit 右移**的基线来对齐/超越。
- **严重度**：**重要**。
- **证据锚点**：`axes/A-mixing/PRIOR_ART.md` §1/§2；`axes/A-mixing/PRIOR_ART_LENGTH.md` §2。

### P18 · A3 页头/轴 summary 与 A3 verdict **互相矛盾**
- **问题**：轴 `local_summary`「丢前缀在 bpc 上**很便宜**（+0.058~0.121 @13M）」（`catalog.yaml:39`），
  而 A3 verdict「丢前缀的代价……**并不便宜**（只保留最近 32 = +0.212）」（`catalog.yaml:221`）。
- **为什么是问题**：同一事实两个相反陈述同时出现在 SSOT 的不同表面。
- **严重度**：**重要**。
- **证据锚点**：`catalog.yaml:39` vs `:221`；`axes/A-mixing/README.md`。

### P19 · A6 top-k 的显著性口径含混
- **问题**：`catalog.yaml:297` 用「配对 20/20 批同向，t=−7.7」当真实性证据，而该 t 只量**评测批内配对差**，
  不含**训练重跑方差**。
- **严重度**：**重要**。
- **证据锚点**：`catalog.yaml:297`。

---

## 4. 次要问题

### P20 · A 轴 status 在 catalog 与 PRIOR_ART 之间不一致
- A3/A4/A6/A10 在 `catalog.yaml` 是 `status: todo`（`:226/256/294/391`），
  而 `PRIOR_ART.md` 各节标题写作「（judged）」并给 (a)/(b) 分诊。
- **证据锚点**：`catalog.yaml:226/256/294/391`；`axes/A-mixing/PRIOR_ART.md` §1。
- **严重度**：次要。

### P21 · A3/A9/A2 的证据锚指向不可复现的旧 `.scratch` / `archive` 资产
- `A2` evidence 指向 `archive/legacy-docs/findings.md @5495a3b`（上一时代资产，`catalog.yaml:204`）；
  `A3/A4/A8` 只锚 `.scratch/07-x1-x3/...`；`A9` 只锚两个 `.scratch`（`catalog.yaml:380`）。
- **严重度**：次要（但叠加 P7 = 致命）。
- **证据锚点**：`catalog.yaml:204/242/273/342/380`。

### P22 · A 轴 `local_summary` 跨轴泄漏
- `catalog.yaml:43` 写「M 轴的**显式记忆槽**已有首验，见 M1」——A 轴的目的落点在 M 轴被记录。
- **严重度**：次要（呼应 P2）。
- **证据锚点**：`catalog.yaml:37-45`。

### P23 · A10 备注称"业界无人占位"，与 `industry` 字段自陈矛盾
- `catalog.yaml:416`（notes）vs `catalog.yaml:392-394`（industry: Compressive/Memorizing/StreamingLLM）。
- **严重度**：次要（与 P15 同源）。
- **证据锚点**：`catalog.yaml:392-394, 416`。

---

## 5. 残留清单（上一版本 / 上一时代 / 未 migrate）

### R1 · `catalog.yaml` A 段 `prior_art_note` **整体被 `#` 注释掉且错位一格**
- 逐行事实（`grep -n prior_art_note catalog.yaml`）：
  - `catalog.yaml:212` 注释头写 `A3`，装的却是 **A2** 的文本「无可见性约束的全因果 softmax 注意力是质量锚点」；
  - `catalog.yaml:245` 注释头写 `A4`，装 **A3** 的文本「滑窗注意力近无损；丢弃首 token 会崩……」；
  - `catalog.yaml:348` 注释头写 `A9`，装 **A8** 的文本「RoPE + 位置插值……」；
  - `catalog.yaml:381` 注释头写 `A10`，装 **A9** 的文本「靠掩码逼不出信息搬运……」；
  - **A6 没有** `prior_art_note`。
- 因为整行以 `#` 开头 ⇒ **机器不可读**，`check_schema.py` 抓不到 ⇒ 页面上完全看不到"这是业界标准件"。
- **严重度**：**致命（数据完整性）**。
- **证据锚点**：`catalog.yaml:212,245,348,381`。

### R2 · A6 options 仍引用**已删除格**「原 A5-dilated-window / 原 A7-addressing」
- `catalog.yaml:289-290` 两个选项写着「（原 A5-dilated-window，作同预算对照）」「（原 A7-addressing）」，
  而 `A5`/`A7` 目录已移入 `archive/dropped/axes/A-mixing/`（`archive/README.md` 并入表）。
- **严重度**：**重要（残留）**。
- **证据锚点**：`catalog.yaml:289-290`；`archive/README.md`；`archive/dropped/axes/A-mixing/A5-dilated-window/`、`A7-addressing/`。

### R3 · A6 verdict 是**超长历史叙事**，违反 SSOT「只写当前状态」
- `catalog.yaml:295-315` 含大量「旧读法……**已由** 2636 统一口径」「本条只作**历史快照**」
  「ERrata/永久不可归因/ticket 40-Y2/41-Z1/42-W1/43-V1/45-L1/46-W2/52-A6U」等**日期化变更叙事**。
- 按 `AGENTS.md §2.1`「事实文档一律陈述式、当前状态，不带变更记录/ERRATA/历史快照」，这是残留。
- **严重度**：**重要（残留）**。
- **证据锚点**：`catalog.yaml:295-315`。

### R4 · `.scratch/doc-log/axes/A-mixing/**` 存在 `*.md.md` 双扩展名文件
- 例：`2614-.../analysis.md.md`、`2622-longT-512/README.md.md`、`2622-longT-512/MODEL.md.md`、
  `2616-.../stability.md.md`、`PRIOR_ART.md.md` 等（`find .scratch/doc-log/axes/A-mixing -name "*.md.md"`）。
- **严重度**：**次要（残留）**。
- **证据锚点**：`.scratch/doc-log/axes/A-mixing/**/*.md.md`。

### R5 · A2 的 stub 实验目录仍在项目树内
- `A2-full-attention/experiments/{2502-engine-m0-probe,2504-m1-first-runs,2516-rung4-scale-100m,2519-asymptotic-tax}/`
  各只有 `README.md`。
- **严重度**：**次要（与 P8 同源）**。
- **证据锚点**：`axes/A-mixing/A2-full-attention/experiments/`。

### R6 · A3 备注仍以「X2/X3」等旧代号叙事
- `axes/A-mixing/A3-window-anchor/README.md` §5「beyond-window 桶召回 0.4%（X2v2）」、
  `catalog.yaml:39`「检索当场归零（beyond 0.4%）」等，仍以旧实验代号 X1/X2/X3 表述（旧时代台账）。
- **严重度**：**次要（残留）**。
- **证据锚点**：`axes/A-mixing/A3-window-anchor/README.md` §5；`catalog.yaml:39`。

---

## 6. 缺什么（出资人要求的「问题 / 假设 / 判据」清单）

> 出资人：「缺的是什么问题、什么假设、什么 pareto 口径？」——逐条列出 A 轴**没有**的。

1. **缺「explicit 右移」的设计集**：没有一格把"写通道 + 迫使使用的损失 + 前缀可丢"作为机制来立项。
   现有 explicit 首验在 M1，A 轴只引用。
2. **缺 implicit vs explicit 的同仪器/同预算正面对撞**：`implicit 不怎么 work` 目前只有 M3/A9 的
   "掩码逼不出搬运"负结果（`catalog.yaml:376` failure 末条），**没有** implicit（窗口/偏置）与
   explicit（写通道）在**同一任务、同一物理预算、同一保留仪器**下的对照。
3. **缺「前缀可丢」的端到端验证协议**：判据停留在"训练期可见集 + clean bpc"；没有
   训练期右移 → 推理期**真丢前缀 KV** → 目标仍可测 + 与全量对照 + 质量-预算曲线 的协议。
4. **缺 AR 仍 causal 的验收条件**：没有任何一处检验"丢前缀后解码是否仍严格因果、是否长出非因果捷径"。
5. **缺变化空间前门**：`.scratch/53-design-audit/DISPATCH.md` §0 已提出强制前门（"必须有变化空间，否则
   把'设计无效'误读成'机制无效'"），但 A 轴契约**未逐条回答**（A6 的"稠密 bool 掩码 ⇒ 纸面省 FLOPs
   兑现不了"正是设计无效的一个实例，见 `DISPATCH.md` §1）。
6. **缺「explicit 右移」的既有巨人基线**：Compressive Transformer / AutoCompressors / ICAE / RMT /
   Infini-attention / Landmark / gist / DeltaNet 未被列为要打平/超越的 explicit 基线。
7. **缺「结论不唯一」的显式声明**：出资人说"结论并不一定单一、可能有很多种手段"；A 轴仍以"窗口"这类
   单一手段为主标题，未把"多种右移手段"作为枚举维度。

---

## 7. 出资人具体批注 · 逐条回应

| 批注 | 现状核对 | 结论 |
|---|---|---|
| **A2**：基线不是项目，应去 scratch | `catalog.yaml:190` 仍是 `A2-full-attention`（judged）+ 4 个 stub 实验目录 | **未 migrate**（P8/R5） |
| **A3**：窗口是手段，信息右移是目的；要 explicit design；结论不单一 | A3 标题/claim 仍是"窗口/锚点/sink"（`catalog.yaml:216`），目的只在 note；无 explicit 方案 | **未改**（P1/P2/P9） |
| **A4**：手段不是目的；"本地无净增量"因没目的 | A4 仍是独立格、options 是偏置手段族、结论"−0.0068 无净增量"；且用错 ALiBi 口径（无 per-head slope、无长度外推） | **未改**（P10） |
| **A6**：研究项/问题/假设是啥？长 T 是 Temperature？不掉质量不可能，要找 pareto | A6 是 10 种算子罗列、无假设；"长 T"=序列长度（非标准命名）；11 个实验在判"是否无损"、无 pareto | **未改**（P4/P5/P11） |
| **A8**（自查）：位置编码是手段，要写目的或删 | A8 仍是手段格、本地零实验、自判 (a) | **未改**（P13） |
| **A9**（自查）：保留/硬丢是手段，应并入 A3 | A9 与 A3 重复、判据相同 | **未改**（P14） |
| **A10**（自查）：压缩触发是手段，应并入 M2 或删 | A10 仍独立格、judge≠实测、与 PRIOR_ART 矛盾 | **未改**（P15/P16） |

---

## 8. 一句话总结

**A 轴把「信息右移」这个目的丢了**：整轴是隐式省算力手段（窗口/ALiBi/稀疏/线性/位置编码）的清单，
没有 explicit 右移设计，也没有 pareto frontier 口径与"前缀可丢"的端到端判据；
在此基础上还叠着"0.014 全局界被自己来源否证""基线 A2 未移出""catalog 的 prior_art_note 错位且被注释掉"
等结构与数据完整性问题。**最致命的一条是 P1+P2**：轴的定义里根本没有目的，导致下游每一格都只能在
手段层出"不亏/亏多少""无净增量"这类无决策价值的结论。
