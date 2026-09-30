<!-- AUTO:begin -->

# INDEX · 设计空间状态灯

共 **51** 个格子 ｜ ✅ judged 32 ｜ ◐ partial 7 ｜ 🏭 reference 7 ｜ ⬜ stub 5

> 全部内容由 `tools/build_wiki.py` 从 `catalog.yaml` 生成；**不要手改本文件**。

## A · 序列混合机制（10 格）

[轴说明](axes/A-mixing/README.md) ｜ 一个位置从哪里、以什么算子、按什么寻址方式取信息。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`A1-causality`](axes/A-mixing/A1-causality/README.md) · 因果性 | ⬜ stub | — |
| [`A2-full-attention`](axes/A-mixing/A2-full-attention/README.md) · 全注意力基线 | ✅ judged | 基底成立，外部锚在公开数字范围内。 |
| [`A3-window-anchor`](axes/A-mixing/A3-window-anchor/README.md) · 窗口 / 锚点 / sink | ✅ judged | 丢前缀的代价**由距离与预算共同决定、并不便宜**（见 ERRATA：只保留最近 32 个 = +0.212 bpc）；锚点通道是真产物（76.8→93.3%）；检索腿显示「低损伤 ≠ 保住了」。 |
| [`A4-soft-distance-penalty`](axes/A-mixing/A4-soft-distance-penalty/README.md) · 软距离惩罚（ALiBi 式加性偏置） | ✅ judged | 唯一改善 clean bpc 的元素，但不创造保留——它改变使用率，不改变依赖是否为零。 |
| [`A5-dilated-window`](axes/A-mixing/A5-dilated-window/README.md) · 膨胀 / 跨步窗 | ⬜ stub | — |
| [`A6-mixing-operator`](axes/A-mixing/A6-mixing-operator/README.md) · 混合算子 | ✅ judged | 内容选择式 Top-K 稀疏注意力在 **T=256 上净亏**：质量越删越差（k=32 越界有害、k=64 只在噪声带内），成本**实测上升** 2.2×（慢且更占显存）——稀疏的收益要到长上下… |
| [`A7-addressing`](axes/A-mixing/A7-addressing/README.md) · 寻址方式 | ⬜ stub | — |
| [`A8-positional-semantics`](axes/A-mixing/A8-positional-semantics/README.md) · 位置语义 | ◐ partial | 绝对位置嵌入是硬上界（解码总长 ≤ seq_len），相对偏置族已侧证。 |
| [`A9-old-info-retention`](axes/A-mixing/A9-old-info-retention/README.md) · 旧信息处理策略 | ◐ partial | 保留 / 硬丢 / 锚豁免已测；压缩、递归、外置三类零经验。 |
| [`A10-compression-trigger`](axes/A-mixing/A10-compression-trigger/README.md) · 压缩触发与优先级 | ✅ judged | 「按访问频率优先压（LRU）」在同等删除预算下没有可见收益**——字面优势经对账后归因于「恰好删掉的可见性对数更少」，重对齐后消失（见 ERRATA）。压缩优先级这一维度在 13M/256 快照上… |

## C · 垂直机制（深度）（6 格）

[轴说明](axes/C-vertical/README.md) ｜ 同一位置的算力如何沿「层」组织——跑几层、怎么连、怎么共享、怎么精化。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`C1-layers-per-token`](axes/C-vertical/C1-layers-per-token/README.md) · 每个 token 跑几层 | ✅ judged | 逐 token 变深度成立且是本项目最厚的资产；代价是底座质量税。 |
| [`C2-layer-selection-order`](axes/C-vertical/C2-layer-selection-order/README.md) · 选哪些层与执行顺序 | ✅ judged | 元素税定律的原始出处：破坏型元素贵，保留型近免费。 |
| [`C3-inter-layer-topology`](axes/C-vertical/C3-inter-layer-topology/README.md) · 层间拓扑（残差之外的连接） | ✅ judged | 首探无优势：**并行子层明确有害**（+0.0458，等参数对照 +0.0402 复现），**层间学习加权和中性**（−0.0008）。 |
| [`C4-weight-sharing`](axes/C-vertical/C4-weight-sharing/README.md) · 权重共享 | ⬜ stub | — |
| [`C5-refinement-operator`](axes/C-vertical/C5-refinement-operator/README.md) · 精化算子 | ✅ judged | 携带门是形态税的承载者之一，也是「任意深度可读」的结构来源。 |
| [`C6-depth-supervision`](axes/C-vertical/C6-depth-supervision/README.md) · 训练期深度分布与监督密度 | ✅ judged | 出口资产只能由深度维监督造出来——这是本项目最有迁移价值的一条结论。 |

## H · 视界（一次前向产生几个位置）（5 格）

[轴说明](axes/H-horizon/README.md) ｜ 一次前向预测几个未来位置、这些位置在训练时被什么内容填充、输出如何组合与验证。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`H1-outputs-per-pass`](axes/H-horizon/H1-outputs-per-pass/README.md) · 每步输出位置数 | ✅ judged | 全项目税效比最好的机制；但墙钟收益是 13M/launch-bound 现象。 |
| [`H2-training-fill`](axes/H-horizon/H2-training-fill/README.md) · 训练时未来位置的内容填充 | ✅ judged | 保留式填充（self-fill）胜出；PAD 与错误 token 都更贵；软提交有 OOD 污染。 |
| [`H3-composition`](axes/H-horizon/H3-composition/README.md) · 多位置预测的组合方式 | ✅ judged | 自门控组合**被实测否决为主张**：门控混合 0.4738 仅比直跳 0.4672 高 +0.0020（<0.014 噪声带），且等参数无门对照 0.4785 更高 ⇒ 增益归容量不归门控；门不是… |
| [`H4-head-trunk-relation`](axes/H-horizon/H4-head-trunk-relation/README.md) · 头与主干的关系 | ✅ judged | 挂头零税坐实（Δbpc = 0.0000）。按修正后的诚实口径（vs 1-step AR 两次前向，只替换紧邻位置）**判中性**：对手 54.89%（≥ 算术下界 0.4822 ✓ 对账过），跳… |
| [`H5-verification`](axes/H-horizon/H5-verification/README.md) · 验证机制 | ✅ judged | 批量验证可用且逐位一致；定稿不变量（全深度重打分）是本形态的正确性保障。 |

## W · 宽度（5 格）

[轴说明](axes/W-width/README.md) ｜ 每 token 激活多少参数/通道、参数如何组织成专家、读出几何如何。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`W1-active-width`](axes/W-width/W1-active-width/README.md) · 激活宽度 | ✅ judged | 减宽 = 减容量，不买灵活性；结构化约束会把状态能量赶出被约束子空间。 |
| [`W2-expertization`](axes/W-width/W2-expertization/README.md) · 专家化（MoE） | ⬜ stub | — |
| [`W3-kv-head-sharing`](axes/W-width/W3-kv-head-sharing/README.md) · KV 头共享 | ✅ judged | 参考线已补齐**：GQA（KV −75%）实测 damage **+0.0143**、同 seed 重跑 +0.0139（**跨在门槛两侧**，散度仅 0.0004 ⇒ 依预登记规则该臂判读作废、… |
| [`W4-readout-geometry`](axes/W-width/W4-readout-geometry/README.md) · 读出几何 | ✅ judged | 读出几何**可以**解耦（prob0/提交/传输不受损，同空间前提不冲突），但**增量归因于「读出重参数化 + 额外 2d² 参数」，不是非线性本身**：等参数纯线性对照 Δ=−0.0888 与 … |
| [`W5-capacity-vs-compute`](axes/W-width/W5-capacity-vs-compute/README.md) · 容量-计算分离 | ✅ judged | 减容量不买灵活性；买灵活性必须走保容量减计算。 |

## F · 迭代（4 格）

[轴说明](axes/F-iteration/README.md) ｜ 对什么重复、用什么通道回传、何时停、靠什么训练信号让重复有效。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`F1-iteration-object`](axes/F-iteration/F1-iteration-object/README.md) · 迭代对象 | ◐ partial | 前两项已收口；后四项零经验。 |
| [`F2-return-channel`](axes/F-iteration/F2-return-channel/README.md) · 回传通道 | ✅ judged | 直接回传是固定点（结构性 no-op）；软回传有 OOD 污染；离散回传必须门控。 |
| [`F3-stopping`](axes/F-iteration/F3-stopping/README.md) · 停止条件 | ✅ judged | 选择性问题（愿景核心命题）获证；预算求解器端到端建成。 |
| [`F4-training-signal`](axes/F-iteration/F4-training-signal/README.md) · 迭代/重复的训练信号 | ✅ judged | consistency loss 中性；重建/sufficiency 是唯一无人占位、且是压缩可恢复性的必要信号。 |

## M · 状态载体（3 格）

[轴说明](axes/M-carrier/README.md) ｜ 跨步信息存在哪里、怎么压、跨步假设是什么。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`M1-carrier-type`](axes/M-carrier/M1-carrier-type/README.md) · 状态载体类型 | ◐ partial | 只有 KV 载体；其余零经验，但载体是「能否丢/重算」的决定因素。 |
| [`M2-carrier-compression`](axes/M-carrier/M2-carrier-compression/README.md) · 载体压缩 | ◐ partial | 驱逐已测（价格见 A3）；量化/摘要/低秩零经验。 |
| [`M3-cross-step-assumption`](axes/M-carrier/M3-cross-step-assumption/README.md) · 跨步假设 | ✅ judged | 有界内存的价签已测两档：**窗口 192 的训练税 −0.0028（近零）**、窗口 32 为 +0.0866；但 192 档**本就没有头寸**（未重训限制到 192 只 +0.0029），真正… |

## G · 策略与调度（5 格）

[轴说明](axes/G-policy/README.md) ｜ 谁选路径、用什么货币记账、观测什么信号、怎么校准、多线如何并行。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`G1-selector`](axes/G-policy/G1-selector/README.md) · 路径选择器 | ✅ judged | 前五种已建成或已测；在线多线调度器是空白（且障碍已测，见 G5）。 |
| [`G2-currency`](axes/G-policy/G2-currency/README.md) · 记账货币 | ✅ judged | 双货币必须双报——本项目多次出现口径分歧。 |
| [`G3-signal`](axes/G-policy/G3-signal/README.md) · 观测信号 | ✅ judged | 三信号正交监测栈成形；生成段没有单一信号够用。 |
| [`G4-calibration`](axes/G-policy/G4-calibration/README.md) · 校准 | ✅ judged | 校准跨规模存活是本项目最稳的资产之一。 |
| [`G5-parallel-lines`](axes/G-policy/G5-parallel-lines/README.md) · 并行多线机制 | ✅ judged | 共享主干的多线几乎不增益——瓶颈是相关性，不是算力。 |

## R · 参照线（必须打败的基线）（7 格）

[轴说明](axes/R-references/README.md) ｜ 每条创新线面对的现成行业答案——不打败它，创新就没有意义。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`R1-quantization`](axes/R-references/R1-quantization/README.md) · 量化 | 🏭 reference | — |
| [`R2-distillation`](axes/R-references/R2-distillation/README.md) · 蒸馏 | 🏭 reference | — |
| [`R3-pruning-sparsity`](axes/R-references/R3-pruning-sparsity/README.md) · 剪枝 / 结构化稀疏 | 🏭 reference | — |
| [`R4-window-kv-quant-retrieval`](axes/R-references/R4-window-kv-quant-retrieval/README.md) · 窗口注意力 + KV 量化 + 检索 | 🏭 reference | — |
| [`R5-bestofn-verifier`](axes/R-references/R5-bestofn-verifier/README.md) · best-of-n + verifier | 🏭 reference | — |
| [`R6-moe-scale`](axes/R-references/R6-moe-scale/README.md) · MoE 规模轴 | 🏭 reference | — |
| [`R7-existing-early-exit`](axes/R-references/R7-existing-early-exit/README.md) · 既有早退实现（CALM / LayerSkip / MoD） | 🏭 reference | — |

## P · 修饰符（不是轴）（6 格）

[轴说明](axes/P-modifiers/README.md) ｜ 只改变实现/成本/条件的决策：精度、条件化信号、参数插入与初始化、归一化位置、损失权重、数据构造。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`P1-precision`](axes/P-modifiers/P1-precision/README.md) · 精度（修饰符） | ✅ judged | reward/判据类损失必须 float32 + pre-step grad-norm gate。 |
| [`P2-conditioning`](axes/P-modifiers/P2-conditioning/README.md) · 条件化信号（修饰符） | ◐ partial | 深度索引（逐深度读出）与置信度条件（门控）已用；噪声级/任务条件零经验。 |
| [`P3-param-insertion-init`](axes/P-modifiers/P3-param-insertion-init/README.md) · 参数插入与初始化（修饰符） | ✅ judged | 恒等初始化 adapter 在 13M matched-step 族内最优——「从基线出发学 remap」是对的。 |
| [`P4-normalization`](axes/P-modifiers/P4-normalization/README.md) · 归一化位置（修饰符） | ◐ partial | pre-LN + 无 final norm 是「同空间」几何的前提，已固定使用。 |
| [`P5-loss-weighting`](axes/P-modifiers/P5-loss-weighting/README.md) · 损失权重与调度（修饰符） | ✅ judged | 浅层降权单调反向；退火是当前最优配方的组成部分。 |
| [`P6-data-construction`](axes/P-modifiers/P6-data-construction/README.md) · 数据与任务构造（修饰符） | ✅ judged | 任务构造本身是本项目最大的单点偏差来源（enwik8 局部统计极强）。 |

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
