# C 轴（深度/早退）独立批判性复核 · 问题清单

> 复核者：fresh context subagent（fork=false），未参与该线。**只读 + 只写本文件**。
> 任务前提（出资人）：**这条线做的一塌糊涂**，先想后看代码，先写下第一性原理判断，再核对代码/文档，
> 不被现有说法带歪，把所有问题写下来。范围：`axes/C-vertical/**`、`catalog.yaml` 的 C* 条目、
> `pathlm/depth_ar.py`、`pathlm/decode.py`、相关 `.scratch`。
> 严重度：**致命**（命题不成立/永远测不到点/结论错）/ **重要**（口径错、目的缺、判据错、重复、造轮子）
> / **次要**（措辞、命名、台账）。
>
> 说明：本轮独立复核与 `.scratch/50-critical-audit/C-AUDIT.md`（2026-10-02）部分结论重合；本文件
> **独立复现并逐条给证据锚点**，并**新增**：出资人 C1(a)(b)(c) 三问的答案审查、引擎张冠李戴、
> 代码残留（悬空文档引用）、C1 命题与口径的偷换分析。

---

## 0 · 先想后看：独立第一性原理判断（读代码前写下）

**它想解决什么问题（痛点）**：标准 decoder-only transformer 对每个 token 都跑满 L 层；但 token 的
难度不一样（常见续写几乎确定，困难 token 需要满深度）。把「要不要多算」按 token 决定，理论上能在
**同一模型内**把平均算力压下来而不改模型能力——即轴页说的「Pro 当 Flash 用」。

**它真正的 research topic（不是手段）**：不是「窗口/门/fill/ALiBi/并行子层」这些手段，而是
**「逐位置自适应深度的质量-代价前沿（Pareto frontier）长什么样」**：以全量 stock 满深度为 100%，
横轴 = 每 token 等效算力（layer-passes/token 或等效理论时间），纵轴 = 质量（序列级 ppl / 任务准确率，
归一化到 stock=100%）。问题必须能一句话说成「问题 + 假设（预期效用）+ 端到端验证方法」。

**该引入的机制**：(i) 一个**退出决策器**（解码期怎么知道该不该退、是否校准）；(ii) **异构深度下的
注意力语义**（前面 token 退了，后面的深 token 往前 attend 看到什么）；(iii) 一个**退出口/概率头**的
定义（固定 vs 每层；专用头 vs 复用读出）。这三件是同一个机制的三面，缺一即不成立。

**代价该怎么评（出资人已明确）**：「不掉质量不可能」——要找的是 **Pareto frontier**，不是绝对无损；
基线与参照（stock、R7 既有早退）**不是子项目**；不能用墙钟/单点快照当结论。

**巨人（必须站上去）**：自适应算力这一整条线早已有——`ACT`（Graves 2016）、`Universal Transformer`
（Dehghani 2018）、`CALM`（Schuster 2022，逐 token 置信早退）、`LayerSkip`（Elhoushi 2024，早退训练
+ 自投机）、`Mixture-of-Depths`（Raposo 2024，逐 token 路由 + 容量约束）。**「逐 token 变深度」机制
本身不是本仓发明**。若要占位置，只能在 (a) 更干净的代价账、(b) 异构深度 KV 语义或 (c) 与 MoE 路线的
比较上做增量。

**候选设计哪种更可能有效**：MoD 式「逐 token 路由 + 容量约束」比「任意深度早退」更可能**真的省算力**，
因为它保住 dense 批处理（top-k 选满一档），而任意深度早退会破坏批内密集计算（本仓自己的 2621/2623
已实测「省墙钟不省 FLOPs」；compute-bound 下收益还会消失）。换句话说，**本仓选的实现方式，恰好是
理论上最难兑现算力收益的那一种**。

**现有方法对不对（先验怀疑）**：(1) 命题要求「降算力 + 保**序列级**质量」，但本仓公开的收益数是
**单 token 一致率** `tf@4`——单 token 一致率天然高（深监督把它训到 0.97），且它量的是「浅层出口与
自家深层的 argmax 是否相同」，**不是正确性**，近同义反复；(2) 省算力与省墙钟是两回事，本仓两种口径
混用；(3) 没有一条与「训练一个更浅的 dense 模型 / MoE」这种真正的省算力对手比较。

**结论（先验）**：这条线的**命题从未被端到端检验过**，而它公开的「资产」是命题中最弱、最容易达到的
代理量。下面用代码/实验逐条核对。

---

## 1 · 问题清单

### 致命

**C-P01 · 核心命题（逐 token 早退降算力且保住**序列级**质量）从未做过一次端到端实验。**
- 问题：C1 的判据白纸黑字是「序列级 perplexity/一致率 vs 全深度 + 算力」，但**逐 token 门控退出的
  自回归序列级实验从来没跑过**。已有的 e2e 只有三种：固定退 4 层、均匀随机退、CALM 门控（且是
  R7 参照线的 13M，不是 C1 自己的判据实验）。
- 为什么是问题：命题的两半（降算力 / 保序列级质量）**一半都没测**；把一个从未跑过的命题写成
  「本轴最厚资产、已实现」，是把未验证当结论。出资人「不要最后发现根本没研究到点上去」正指此。
- 严重度：**致命**。
- 证据锚点：`axes/C-vertical/C1-layers-per-token/README.md`（`verdict` 的「尚缺（本线主判据，未测）
  ① …… 从未跑」）；`catalog.yaml` C1 `judge`/`verdict`；`axes/C-vertical/README.md`（「本地总体结论」
  把 tf@4/exit-4 当既定资产）。

**C-P02 · 偷换命题：用单 token 一致率 `tf@k` 当序列级质量，且 `tf@k` 量的是「自己 vs 自己」。**
- 问题：headline「收益」= `tf@4 0.9726 / tf@6 0.9946`。① 它是单 token 代理，判据自己声明单 token
  不够（`p^N` 累积：p=0.973、N=100 ⇒ 整段一致仅 ≈6.5%）；② `tf@k` = depth-k argmax == depth-L
  argmax，**是模型自己浅层与深层的一致性，不是与正确答案的一致性**；而训练对每个深度都做监督
  （随机退出 + 全深度读出），等于**把「浅层模仿深层」直接训进目标**，所以 `tf@k` 高是**近乎同义
  反复的构造结果**，不能证明「质量不掉」。
- 为什么是问题：命题要证的是「掉不掉质量」，`tf@k` 恰恰把「质量」偷换成「自洽」。这是本线最核心
  的逻辑漏洞，且 C1 的标题字段仍以它为 `benefit`。
- 严重度：**致命**。
- 证据锚点：`axes/C-vertical/C1-layers-per-token/README.md`（`benefit`/`verdict`）；`catalog.yaml`
  C1 `benefit`；`axes/C-vertical/C1-layers-per-token/experiments/2621-depth-draft-e2e/analysis.md`
  §1 口径表（`tf@k = depth-k argmax == depth-L argmax`）；`pathlm/depth_ar.py:2336-2363`（全深度读出
  + 随机退出监督）。

**C-P03 · 引擎张冠李戴：被当作「已有」的逐 token 退出门控，其实在上一代引擎里；真正的 depth-ar
生产解码器只支持固定退出深度。**
- 问题：C1 `verdict` 写「引擎已具备逐 token 退出门控（`pathlm/decode.py::_run_stack` 的
  `exit_threshold` 读 `sigmoid(model.conf(h))`）」。但 `decode.py` 的 `Decoder` 绑定的模型是
  `pathlm/model.py::PathLM`（旧 M0/M1 谱系，`docs/design.md` §5 把它列为「历史引擎」）。C1/C6 的
  实验全部用 `DepthARModel`（`pathlm/depth_ar.py`），它**根本没有 `conf` 属性**，其 `DepthARDecoder.step`
  的签名是 `step(inp, forced_depth)`——**逐位置强制一个退出深度，没有逐 token 退出判断**。
- 为什么是问题：本线声称的「机制成立」把两个引擎混为一谈。承载体（能跑本线 checkpoint 的 depth-ar
  解码器）**在解码期做不到逐 token 早退**；能做逐 token 早退的 `decode.py` 跑的是另一套模型。
- 严重度：**致命**。
- 证据锚点：`axes/C-vertical/C1-layers-per-token/README.md`（`verdict` 的「已有」句）；
  `pathlm/decode.py:2-18, 41-75`（`Decoder` 的 docstring 与 `_run_stack`）；`pathlm/decode.py:12`
  （`from .model import PathLM, cap_norm`）；`pathlm/model.py:161`（`self.conf` 只在 `PathLM`）；
  `pathlm/depth_ar.py:2706-2744`（`DepthARDecoder.step(inp, forced_depth)`）；`docs/design.md` §5
  （`pathlm/` 为「共享底座与历史引擎」）。

**C-P04 · 代价口径自相矛盾：判据明写「墙钟不进判据」，headline 却拿墙钟当卖点；而按真正的算力
口径，没有一条同时「降算力 + 保序列质量」。**
- 问题：① C1 `judge` 写「报告口径=layer-passes/token …… 端到端墙钟取决于引擎实现，不进判据」，
  但 `benefit`/轴页的卖点是「exit-4 **3.1× 墙钟**」「uniform-exit 1.8×」。② 按算力口径：
  深度草稿 **8.45 > 8.0**（更贵，FLOPs 负账）；固定退 4 = 4.0（省一半，但用的是 depth-4 出口，质量
  掉，非逐 token 门控）；CALM 门控退到 3.43–3.64 层，但「一致率 0.959」（仍是 token 级、且来自 R7
  参照线 13M，不是 C1 判据实验）。⇒ 没有一条点同时满足命题。
- 为什么是问题：用被自己排除的口径当结论，且真正的口径下命题不成立——这是「手段 vs 目的」错位。
- 严重度：**致命**。
- 证据锚点：`catalog.yaml` C1 `judge`（墙钟不进判据）；`axes/C-vertical/C1-layers-per-token/README.md`
  （`benefit` 的 3.1×/1.8×）；`axes/C-vertical/C1-layers-per-token/experiments/2621-depth-draft-e2e/analysis.md`
  §5（`8.45 vs 8.00`、FLOPs +5.6%）；`.scratch/52-r7-early-exit`（2623）`analysis.md` §4（CALM 3.43/8、
  一致率 0.959）。

### 重要

**C-P05 · 出资人问 (a)「解码期怎么知道该不该退出」——当前没有落地的决策机制，且未校准。**
- 事实：depth-ar 生产解码器（`DepthARDecoder`）**没有**退出判断；唯一的逐 token 退出器在参照实验
  `2623/exit_policies.py` 里，判据 = 每层读出后取 `logits.softmax(-1).max()`（top-1 概率）与逐深度
  阈值比较。仓库自认「退出预测头 `conf` 的校准未验（已测 ECE 只是输出头 prob0）」。
- 为什么是问题：这是早退的**核心决策问题**，本线的主判据实验要靠它，但它既没进生产引擎、也没校准
  证据；「该不该退」目前是「拿读出 top-1 概率拍一个阈值」。
- 严重度：**重要**。
- 证据锚点：`axes/C-vertical/C1-layers-per-token/README.md`（`verdict`「尚缺②」）；
  `axes/R-references/R7-existing-early-exit/experiments/2623-standard-early-exit-13m/exit_policies.py:150`
  （`_confidence = logits.softmax.max()`）、`:133-135`（逐层停靠）；`pathlm/depth_ar.py:2706`。

**C-P06 · 出资人问 (b)「前面 token 早退后，后面深 token 往前 attend 看到什么」——当前是 fill
（把冻结表征经本层 KV 投影塞回去），不是预测/补出；其质量代价从未测，且与旧引擎 no-fill 语义矛盾。**
- 事实：depth-ar 的 `DepthARDecoder._layer_kv` 对「没跑到本层」的位置，用 `frozen`（该 token 退出
  时的表征）过本层 `qkv(ln1(·))` 得到 K/V**填充**（`pathlm/depth_ar.py:2694-2703`）；训练侧对应
  `fill_kv` 路径（`:2278-2283`）。这既不是「用线性/非线性头预测缺失 token 的表征」，也不是「看不到」，
  而是「复用冻结表征 + 本层投影」。同时旧引擎 `decode.py` 是 **no-fill**（该位置在深层 KV 缺席，
  `decode.py:16-18`，训练侧对应 `no-fill ragged`，`depth_ar.py:2287`）。
- 为什么是问题：出资人问的正是「异构深度下前向注意力的语义」，而本线**两套语义并存**，且填充近似的
  质量代价**从未测**（C1 自认「尚缺③ 异构深度下深层 token 回看浅层冻结表征的质量代价未测」）。这
  直接决定早退命题成不成立（深层看到的浅层信息是失真的）。
- 严重度：**重要**。
- 证据锚点：`pathlm/depth_ar.py:2694-2703`（fill）、`:2278-2287`（fill vs no-fill ragged，注明
  no-fill 即 `decode.Decoder` 语义）；`pathlm/decode.py:16-18`（no-fill，docstring 自述「那些 cache
  永远看不到该 token（正确因果语义）」）；`axes/C-vertical/C1-layers-per-token/README.md`（「尚缺③」）。

**C-P07 · 出资人问 (c)「固定退出口 vs 每层都可退；概率头每层都有 vs 一个标准头」——代码两套实现
互相矛盾且未给对照。**
- 事实：① `decode.py`（旧引擎）用**专用置信头** `model.conf[0]`（`nn.Linear(d,1)`，训练为
  token-identity 的 BCE），且**每层复用同一个头**（`decode.py:67-68`，`model.py:161,366-371`）；
  ② `2623` 用**每层读出 top-1 概率**（per-depth readout，`exit_policies.py:133-135,150`）。
  两套「概率头」定义不同、量纲不同，仓库没有指出哪套是设计、也没有做过「专用头 vs 复用读出」的对照。
- 为什么是问题：出资人明确在问这个机制怎么定义；现状是**没有唯一设计**，出题人无法据此判断机制的
  可行性或代价。
- 严重度：**重要**。
- 证据锚点：`pathlm/decode.py:67-68`；`pathlm/model.py:161`、`:366-371`；
  `axes/R-references/R7-existing-early-exit/experiments/2623-standard-early-exit-13m/exit_policies.py:133-150`。

**C-P08 · 造轮子 + 未对齐巨人基线：逐 token 变深度/早退业界早已有，本仓既把它当本地发现卖，又
没有按业界口径（正常训练 + 轻量早退项）测税。**
- 事实：`PRIOR_ART.md` 自认机制层面 = `CALM/LayerSkip/MoD/Universal Transformer`（(a)），但 C1 的
  headline 把「机制成立 + 出口资产」写成本地最厚资产（`catalog.yaml` C1）。且训练配方把主目标
  （最终深度 CE）**归一摊薄到 1/(L+1)≈1/9**（`pathlm/depth_ar.py:2362-2363`），与业界「主干照常训，
  只加轻量早退项」的基线不同——而 2623 后来才发现「主目标保权反而更差」，即形态税确实与本仓特有配方
  强耦合。
- 为什么是问题：既重复巨人，又没有站上巨人的可比基线；「我们的税」对不上业界的价签。
- 严重度：**重要**。
- 证据锚点：`axes/C-vertical/PRIOR_ART.md` §1 C1（机制 (a)）；`catalog.yaml` C1 `verdict`；
  `pathlm/depth_ar.py:2362-2363`（`loss = loss / (w*(len(readouts)-1)+final_weight)`）；
  `.scratch/52-r7-early-exit`（2623）`analysis.md` §2（2×2 拆配方）。

**C-P09 · C2「选哪些层与顺序」是手段枚举、无独立目的，且证据跨代引擎 + p_skip 口径错 + 只有 self-review。**
- 事实：① C2 `命题`=「层可以被跳过、重排、重复到什么程度」——纯手段，没有「要解决什么/假设是什么/
  怎么端到端验证」（INVENTORY 自评也同意「并入 C1」）。② 税数字（skip +0.175 / shuffle +0.545）
  来自旧 M2 引擎 `.scratch/06-m2-sweep`；其 `p_skip` 由 `sample_path` **逐序列/逐 batch** 采样一次
  （整条路径丢同一批层），**不是逐 token**（`pathlm/config.py:150-166`）。③ 该格无独立复核，源工单
  明文「**User approved self-review in lieu**」。
- 为什么是问题：手段当研究、口径与「逐 token」主题不符（跨引擎不可比）、无独立复核 ⇒ 结论不可采信。
- 严重度：**重要**。
- 证据锚点：`axes/C-vertical/C2-layer-selection-order/README.md` §1；`catalog.yaml` C2；
  `pathlm/config.py:150-166`；`.scratch/06-m2-sweep/issues/01-m2-breadth-sweep.md:33`（self-review）。

**C-P10 · C3 层间拓扑：无目的、无假设、判据有缺陷，且**没有测出资人真正提出的设计**（把之前各层
结果 stack 起来做 causal attention），却已判 `judged`。**
- 事实：① C3 `命题`=「层与层之间除了单条残差还能怎么连」——手段，无目的/假设；② 用**质量面**宣布
  并行子层「明确有害」，而不测业界采用它的理由（吞吐）；③ 联合判据 `tf@4 ≥ 0.95` 对 stock dense
  **零区分度**（自认「契约缺陷」）；④ 只跑 13M / 单 seed / 12k 快照。
- 为什么是问题：出资人的 C3 批注是**一个明确假设**——「目的不就是提高正确率吗？比如把之前层的结果
  都拉出来 stack 在一起做 causal attention」。这正是选项里的「跨层注意力（Hyper-Connections /
  feedback transformer）」——**从未测**（README 自认「未测：跨层注意力、无残差」），却被判 `judged`。
  等于**没测对的东西就下了结论**。
- 严重度：**重要**。
- 证据锚点：`axes/C-vertical/C3-inter-layer-topology/README.md` §1/§5.4/§5.5/「备注（未测：跨层注意力）」
  ；`catalog.yaml` C3；出资人批注见 `.scratch/54-subproject-schema/INVENTORY.md` C3 条。

**C-P11 · C4 权重共享：与出资人判断一致（不省计算、只省显存，未来是 MoE），却仍留作 todo 且状态口径
自相矛盾。**
- 事实：出资人批注明确「研究意义不大，因为不省计算，只省显存，而未来都是 MoE 这种不 care 显存省计算
  路线」。仓库却把 C4 保留为 `todo`，轴页称「本地零经验——这是**最便宜的真空白**」；`PRIOR_ART.md`
  §2 分诊表给 C4 = **(a)**（业界已透），而 C4 格自己写「按 SCHEMA §21 **不判 (a)**（检索不可用）」
  ——**同一事实三处口径互相打架**。且 C4 的 `judge` 写「同 FLOPs bpc」，但权重共享不改 FLOPs，
  判据本身错位。
- 为什么是问题：无决策价值的项（不改「某路径值不值」的判断）不该占格子；状态/判据自相矛盾违反
  仓库自己的 SSOT 规则。
- 严重度：**重要**。
- 证据锚点：`axes/C-vertical/README.md` C4 行（「最便宜的真空白」）；`axes/C-vertical/C4-weight-sharing/README.md`
  （「不判 (a)」）；`axes/C-vertical/PRIOR_ART.md` §2（C4=(a)）、§1 C4；`catalog.yaml` C4 `judge`；
  出资人批注见 `INVENTORY.md` C4 条。

**C-P12 · 缺 Pareto 前沿视野：没有与真正的「省算力对手」（更浅的 dense / 深度剪枝 / MoE）比较，
只有相对自家 stock 的税。**
- 事实：出资人要求「以全量标准为 100%，质量-代价前沿长什么样」。C 轴现有对照只有「同深度 stock vs
  C1 形态」（拿税）与「同形态内 depth-bpc 曲线」，**没有**一个「用同样 token 预算训练一个更浅的 dense
  模型」或 MoE 的对照点。因此无法回答「用户想省一半算力时，该选自适应深度还是直接训小模型」。
- 为什么是问题：命题的价值主张只能在**前沿**上体现；缺外部对手 ⇒ 无法证明自适应深度优于「直接变小」
  （而这恰是 C4 批注里 MoE 路线的核心）。
- 严重度：**重要**。
- 证据锚点：`catalog.yaml` C1 `cost`/`judge`；`axes/C-vertical/README.md`；出资人 C4 批注
  （`INVENTORY.md`）；C 轴 `catalog.yaml` 无「浅 dense / 剪枝 / MoE」对照条目。

**C-P13 · C5/C6 实为 C1 的实现手段，重复成格；C6 结论「出口资产只能由深度监督造出（0.12 vs 0.97）」
证据对不成立。**
- 事实：INVENTORY 自评同意「C5 是 C1 的实现手段」「C6 是 C1 的实现手段」，但二者仍各自单列（C5、C6），
  与 C1 三格讲同一件事。C6 的 headline 证据 `dense trunk tf@4 = 0.12` 是**未训练 tied 读出的 proxy**
  （24b 自述「非训练出口资产」）；真正该跑的对照（`23d-H1-12k`：dense trunk + 全深度 aux）的 tf@4
  **从未跑** ⇒ 「只能由深度监督造出」的对照组缺失。且 C6 证据混两代引擎（`2518` 是 m1 引擎，
  `2524` 是 depth-ar）。
- 为什么是问题：拆分不对（重复/碎片）+ 最强结论证据错 + 跨代混杂。
- 严重度：**重要**。
- 证据锚点：`INVENTORY.md` C5/C6 条；`axes/C-vertical/C6-depth-supervision/README.md`（`verdict`「只
  能由深度维监督造出来」）；`.scratch/50-critical-audit/C-AUDIT.md` §4②（24b `tf_note`、H1-12k 缺 tf@4）；
  `axes/C-vertical/C6-depth-supervision/experiments/2518-exit-retry-law/README.md`（engine: m1）。

**C-P14 · 上一版本/上一时代的残留未 migrate（代码 + 文档）。**
- 事实：
  1. `docs/mental_model.md` **不存在**（已移入 `archive/legacy-docs/mental_model.md`），但
     `pathlm/decode.py:3-4,271`、`pathlm/config.py:57`、`pathlm/slider.py:2,71,86` 仍把它当现行文档引用
     ⇒ **悬空引用**（`grep`/`find` 可复现）。
  2. `pathlm/decode.py` docstring 仍写「`L2-trained models`」「the `X2 gate`」「`M1 limitation`」
     「latent-retry」等旧阶段黑话；`pathlm/__init__.py` 仍是「M0 skeleton」。
  3. C1 grid 的 100M 税归属「形态 96% / 引擎 4% / 配方 ≈0」被自认是「同引擎三列拆**混杂**口径」，
     却「**原样保留并标注**」——旧口径结论未清。
- 为什么是问题：出资人明令「代码/文档里上一版本、上一时代的残留内容要完全 migrate 掉」。
- 严重度：**重要**。
- 证据锚点：`find . -name mental_model*` ⇒ `archive/legacy-docs/mental_model.md`（本仓 docs/ 无）；
  `pathlm/decode.py:3-4,271`；`pathlm/config.py:57`；`pathlm/slider.py:2,71,86`；`pathlm/__init__.py:1`；
  `catalog.yaml` C1 `tax`（「原样保留并标注」）。

**C-P15 · 快照当结论、单点当定论、墙钟数字不可复现。**
- 事实：① C2/C3 只跑 13M / 单 seed / 12k(≈1 epoch) 快照即判 `judged`；② exit-4 墙钟倍率在同协议
  三次 run 间为 1.57× / 3.08× / 5.53×（1.5–5.5× 漂移），单臂内逐 prompt 散布 55%，唯一带交错控制的
  是 2621（1.86×）；③ C6 的「退火配方 +9% 等效验证吞吐」是**计算副产品**（加速因子本身 C<A，9% 全靠
  full-depth 墙钟差）。
- 为什么是问题：单次抽样在 1.5–5.5× 漂移下不可采信；快照不能当渐近结论。
- 严重度：**重要**。
- 证据锚点：`.scratch/50-critical-audit/C-AUDIT.md` §4①（三 run 倍率）；`axes/C-vertical/C3-inter-layer-topology/README.md`
  §5.5（n=1、24k 未跑）；`catalog.yaml` C1 `benefit`（3.1×）、C6 `benefit`（+9%）。

### 次要

**C-P16 · 判据量无变化空间（自己承认）。** C3 的 `tf@4 ≥ 0.95` 对 stock dense 零区分度（自认「契约
缺陷」）；多处「贴门槛/不可排序」。→ 判据设计失效。锚点：`C3 README §5.4`；`.scratch/53-design-audit/DISPATCH.md`
§1。

**C-P17 · 命名/目的表达：C1 的 `title`「每个 token 跑几层」是手段式标题，未陈述目的**（「逐 token
早退降算力且保序列级质量」）；同一概念多拼写（exit / tf@k / 接受率 / 一致率 / 退出资产）混用。锚点：
`catalog.yaml` C1 `title`；`axes/C-vertical/C1-layers-per-token/README.md` §1。

**C-P18 · 无决策价值/半边结论被立成结论。** C2 的「层序 shuffle」子项（+0.545）：没有部署会打乱层序，
结论无论正负都不改变「某算力路径值不值」（建议删，数字留 playbook note）；C3 的「并行子层有害」只测
质量面、不测吞吐面（业界采用它的理由是吞吐）。锚点：`catalog.yaml` C2/C3；`.scratch/50-critical-audit/C-AUDIT.md`
§1。

**C-P19 · 证据锚点不一致。** 同一证据路径挂不同 commit：`.scratch/24-depth-ar-100m/evidence` 在
`catalog.yaml` 一处 `@7725891`、另一处与来源实验 README `@7cdcdaa`；`.scratch/23-depth-ar/evidence`
同时挂 `@1586745`/`@6b27b5e`/`@b8c9f86`。⇒ 至少一个锚点是错的。锚点：`catalog.yaml` C1/C6 `evidence`；
`.scratch/50-critical-audit/C-AUDIT.md` §4③。

---

## 2 · 对出资人批注的逐条回应

**C1（核心）**：命题「逐 token 早退降算力且保住序列级质量」——**逻辑不成立/未被检验**（C-P01/C-P02/
C-P04）。具体三问：
- **(a) 解码期怎么知道该不该退出**：**未答**（C-P05）。生产 depth-ar 解码器没有退出门控；参照实验用
  读出 top-1 概率拍阈值，校准未验。
- **(b) 前面 token 早退后，后面深 token 往前 attend 看到什么 / 要不要补出来 / 线性或非线性头预测 /
  还是干脆看不到**：现状是 **fill**——用该 token 退出时的冻结表征过本层 KV 投影塞回（一个固定投影，
  **不是**学出来的预测头，也**不是**干脆看不到）；旧引擎则是 **no-fill**（干脆看不到）。**两套语义
  并存，且 fill 近似的质量代价从未测**（C-P06）。⇒ 出资人的三种选项，本仓**没有明确选一种**。
- **(c) 固定一个/多个退出口 vs 每层都能退；概率头每层都有 vs 一个标准线性/非线性头**：代码支持
  **每层都能退**，但「概率头」**两套矛盾实现**（旧引擎专用 `conf` 头每层复用 vs 参照线 per-depth
  读出 top-1），且**没有对照**（C-P07）。⇒ 只算半个答案。

**C2**：出资人说「顺序（shuffle）估计不靠谱；重复/跳过可能更靠谱」。核对：本仓**顺序/shuffle** 实测
最贵（+0.545，全项目最贵元素），**redo（重复）** 近免费（+0.044）、**skip（跳过）** +0.175——**数据
方向与出资人一致**。但问题在于：① 整个 C2 是**手段枚举、无目的/假设**（C-P09）；② 数字跨代引擎、
`p_skip` 是逐序列而非逐 token、只有 self-review（C-P09）；③ shuffle 子项无论正负都不改变决策，应删
（C-P18）。**结论**：不要为「层序 shuffle」立研究项；「重复/跳过」也只是手段，须挂到 C1 的「逐 token
深度分配」目的下、在 depth-ar 引擎重测。

**C3**：出资人说「目的就是提高正确率；比如不是残差，而是把之前层的结果都拉出来 stack 在一起做
causal attention」。核对：C3 **没有目的/假设**，用质量面判「并行子层有害」而不测吞吐，且**出资人提的
跨层注意力（stack 历史层 → causal attention）从未测就判了 `judged`**（C-P10）。⇒ 这一格需要**按出资
人的设计重做**，且补目的与假设，不能停在该处。

**C4**：出资人说「权重共享研究意义不大——不省计算只省显存，未来是 MoE」。核对：**完全同意**。仓库却
把 C4 留作 todo 并称「最便宜的真空白」，且状态口径自相矛盾（todo / (a) / 不判(a) 三处打架）（C-P11）。
⇒ 应**降参照/删**，不作为子项目。

---

## 3 · 一句话总判

**C 轴把「逐 token 早退」这个业界已有（CALM/LayerSkip/MoD）的机制，当成自家最厚资产来卖：它的核心
命题（降算力 + 保序列级质量）从未端到端检验过，公开的「资产」`tf@4` 只是单 token 自洽代理（近乎
同义反复），声称的「逐 token 退出门控」还长在上一代引擎里，而按真正的算力口径没有一条点同时满足命题；
出资人问的 (a)(b)(c) 三问分别「未答 / 两套语义并存且未测 / 半个答案」。** 真正该做的是：在 depth-ar
引擎里选定制码期的退出决策与异构深度注意力语义、把命题改成序列级 e2e、并补上「更浅 dense / MoE」这个
省算力对手，画出与全量标准 100% 相对的 Pareto 前沿。
