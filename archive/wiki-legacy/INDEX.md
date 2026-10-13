<!-- AUTO:begin -->

# INDEX · 设计空间状态灯

共 **37** 个格子 ｜ ✅ judged 16 ｜ ⬜ todo 21

> 全部内容由 `tools/build_wiki.py` 从 `catalog.yaml` 生成；**不要手改本文件**。

## A · 序列混合机制（7 格）

[轴说明](axes/A-mixing/README.md) ｜ 一个位置从哪里、以什么算子、按什么寻址方式取信息。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`A2-full-attention`](axes/A-mixing/A2-full-attention/README.md) · 全注意力基线 | ✅ judged | 基底成立，外部锚在公开数字范围内。 |
| [`A3-window-anchor`](axes/A-mixing/A3-window-anchor/README.md) · 窗口 / 锚点 / sink | ⬜ todo | **已有**：丢前缀的代价**由距离与预算共同决定、并不便宜**（只保留最近 32 个 = +0.212 bpc）；锚点通道是真产物（76.8→93.3%）；检索腿显示「低损伤 ≠ 保住了」。**… |
| [`A4-soft-distance-penalty`](axes/A-mixing/A4-soft-distance-penalty/README.md) · 软距离惩罚（ALiBi 式加性偏置） | ⬜ todo | **已有**：唯一**未观察到超带损伤**的 clean bpc 元素（−0.0068，落在 Δ<0.014 不可排序带内），但不创造保留——它改变使用率，不改变依赖是否为零。**尚缺**：per… |
| [`A6-mixing-operator`](axes/A-mixing/A6-mixing-operator/README.md) · 混合算子 | ⬜ todo | **已有**：①内容选择式 Top-K 稀疏注意力在 **T=256 上净亏**（质量越删越差、成本实测上升 2.2×）；②**长 T 四点已统一到同一口径**（T=256/512/1024/20… |
| [`A8-positional-semantics`](axes/A-mixing/A8-positional-semantics/README.md) · 位置语义 | ⬜ todo | **已有**：绝对位置嵌入是硬上界（解码总长 ≤ seq_len）、相对偏置族（dist_pen −0.007，带内）已侧证。**尚缺**：RoPE、插值/外推（YaRN/NTK）、无位置（序无关… |
| [`A9-old-info-retention`](axes/A-mixing/A9-old-info-retention/README.md) · 旧信息处理策略 | ⬜ todo | **已有**：保留 / 硬丢 / 锚豁免已测（硬丢 +0.058~0.121；锚豁免使 needle 76.8→93.3%）。**尚缺**：压缩（summary 槽）、递归状态（SSM/RNN）、… |
| [`A10-compression-trigger`](axes/A-mixing/A10-compression-trigger/README.md) · 压缩触发与优先级 | ⬜ todo | **已有**：「按访问频率优先压（LRU）」在同等删除预算下没有可见收益——字面优势经对账后归因于「恰好删掉的可见性对数更少」，重对齐后消失；压缩优先级这一维度在 13M/256 快照上未测出信息… |

## C · 垂直机制（深度）（6 格）

[轴说明](axes/C-vertical/README.md) ｜ 同一位置的算力如何沿「层」组织——跑几层、怎么连、怎么共享、怎么精化。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`C1-layers-per-token`](axes/C-vertical/C1-layers-per-token/README.md) · 每个 token 跑几层 | ⬜ todo | **目标口径（本线）**：逐 token 早退必须在**降低算力**的同时保住**序列级**准确率（perplexity / 自回归一致率 vs 全深度）。单 token 一致率不足：一致率 p、… |
| [`C2-layer-selection-order`](axes/C-vertical/C2-layer-selection-order/README.md) · 选哪些层与执行顺序 | ✅ judged | 元素税定律的原始出处：破坏型元素贵，保留型近免费。 |
| [`C3-inter-layer-topology`](axes/C-vertical/C3-inter-layer-topology/README.md) · 层间拓扑（残差之外的连接） | ✅ judged | 首探无优势：**并行子层明确有害**（+0.0458，等参数对照 +0.0402 复现），**层间学习加权和中性**（−0.0008）。 |
| [`C4-weight-sharing`](axes/C-vertical/C4-weight-sharing/README.md) · 权重共享 | ⬜ todo | **已有**：无本地数据（业界结论检索不可用，`记忆·未核`）。**尚缺**：深度方向权重共享是否值（同参数量质量 / 同 FLOPs 质量 / tf@k）（待补做）。 |
| [`C5-refinement-operator`](axes/C-vertical/C5-refinement-operator/README.md) · 精化算子 | ⬜ todo | **已有**：携带门是形态税的承载者之一，也是「任意深度可读」的结构来源。**尚缺**：出口资产是否必须付门堆栈税（补 `23d-H1-12k` 的 tf@4/6 + gate-off 臂）（待补… |
| [`C6-depth-supervision`](axes/C-vertical/C6-depth-supervision/README.md) · 训练期深度分布与监督密度 | ⬜ todo | **已有**：出口资产只能由深度维监督造出来——这是本项目最有迁移价值的一条结论。**尚缺**：同引擎税 + 出口资产是否需门堆栈（与 C5 共用一次 eval）（待补做）。 |

## H · 视界（一次前向产生几个位置）（5 格）

[轴说明](axes/H-horizon/README.md) ｜ 一次前向预测几个未来位置、这些位置在训练时被什么内容填充、输出如何组合与验证。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`H1-outputs-per-pass`](axes/H-horizon/H1-outputs-per-pass/README.md) · 每步输出位置数 | ✅ judged | **已有（本地实测）**：n_mtp 1→2 税 +0.041、2→3 +0.017（递减）；13M/100M 接受级联稳定（2.37–2.40 tok/fwd）。**算力账**：一次前向送 `1… |
| [`H2-training-fill`](axes/H-horizon/H2-training-fill/README.md) · 训练时未来位置的内容填充 | ✅ judged | 保留式填充（self-fill）胜出；PAD 与错误 token 都更贵；软提交有 OOD 污染。 |
| [`H3-composition`](axes/H-horizon/H3-composition/README.md) · 多位置预测的组合方式 | ⬜ todo | **已有**：**本格命题限于「冻结表征上的重复潜在映射与门控」**：direct `T(h)` / composed `T(T(h))` / gated 混合，全部只被 CE(t+2) 监督。它… |
| [`H4-head-trunk-relation`](axes/H-horizon/H4-head-trunk-relation/README.md) · 头与主干的关系 | ✅ judged | 挂头零税坐实（Δbpc = 0.0000）。按修正后的诚实口径（vs 1-step AR 两次前向，只替换紧邻位置）**判中性**：对手 54.89%，跳跃头 46.67% = 对手的 **85%… |
| [`H5-verification`](axes/H-horizon/H5-verification/README.md) · 验证机制 | ✅ judged | 批量验证可用且逐位一致；定稿不变量（全深度重打分）是本形态的正确性保障。 |

## W · 宽度（4 格）

[轴说明](axes/W-width/README.md) ｜ 每 token 激活多少参数/通道、参数如何组织成专家、读出几何如何。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`W1-active-width`](axes/W-width/W1-active-width/README.md) · 激活宽度 | ⬜ todo | **已有**：减宽 = 减容量，不买灵活性；结构化约束会把状态能量赶出被约束子空间。**尚缺**：真正的减宽（推断期确定性裁通道 / 训练期 width dropout）的判决（待补做）。 |
| [`W2-expertization`](axes/W-width/W2-expertization/README.md) · 专家化（MoE） | ⬜ todo | **已有**：无本地数据（业界结论检索不可用，`记忆·未核`；原 R6-moe-scale 已并入本格——13M 无公开对照）。**尚缺**：13M 同 FLOPs MoE 是否 ≥ dense（… |
| [`W3-kv-head-sharing`](axes/W-width/W3-kv-head-sharing/README.md) · KV 头共享 | ⬜ todo | **已有**：GQA（KV −75%）实测 damage **+0.0143**、同 seed 重跑 +0.0139（**跨在门槛两侧**，散度仅 0.0004 ⇒ 依预登记规则该臂判读作废、**… |
| [`W4-readout-geometry`](axes/W-width/W4-readout-geometry/README.md) · 读出几何 | ✅ judged | 读出几何**可以**解耦（prob0/提交/传输不受损，同空间前提不冲突），但**增量归因于「读出重参数化 + 额外 2d² 参数」，不是非线性本身**：等参数纯线性对照 Δ=−0.0888 与 … |

## F · 迭代（2 格）

[轴说明](axes/F-iteration/README.md) ｜ 对什么重复、用什么通道回传、何时停、靠什么训练信号让重复有效。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`F2-return-channel`](axes/F-iteration/F2-return-channel/README.md) · 回传通道 | ⬜ todo | **已有**：直接回传是固定点（结构性 no-op）；软回传有 OOD 污染；离散回传必须门控。**尚缺**：修 D6（gauge bug）后同批/同 seed/同 token 下 direct/… |
| [`F4-training-signal`](axes/F-iteration/F4-training-signal/README.md) · 迭代/重复的训练信号 | ⬜ todo | **已有**：consistency loss 中性；**重建/sufficiency 损失是压缩可恢复性的关键信号，但业界已有成熟前例（非空白格）**——我们的增量只能是「13M + 局部先验抗… |

## M · 状态载体（2 格）

[轴说明](axes/M-carrier/README.md) ｜ 跨步信息存在哪里、怎么压。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`M1-carrier-type`](axes/M-carrier/M1-carrier-type/README.md) · 状态载体类型 | ✅ judged | **入口判读（本格当前结论，本地实测）**： 1. **通道活着且被读**：真实文本口径下 `read_share` 0.30–0.42、读出增益已逃离单边零、同权重关掉通道前向偏差 0.46、且… |
| [`M2-carrier-compression`](axes/M-carrier/M2-carrier-compression/README.md) · 载体压缩 | ⬜ todo | **已有**：驱逐已测（价格见 A3）；KV 量化有本地实测（R1/2614、2618/2621），但价签须按因果/流式量化口径重测（见 R1）。**尚缺**：摘要压缩、低秩投影（MLA 式）无经… |

## G · 策略与调度（3 格）

[轴说明](axes/G-policy/README.md) ｜ 谁选路径、观测什么信号、怎么校准。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`G1-selector`](axes/G-policy/G1-selector/README.md) · 路径选择器 | ⬜ todo | **已有**：路径选择器前五种已建成或已测（原 F3-stopping 已并入本格：停止条件与预算分配同一条线）。**尚缺**：按原预登记判据（分桶单调 + 低置信桶 ≥ 高置信桶 2×）照原样跑… |
| [`G3-signal`](axes/G-policy/G3-signal/README.md) · 观测信号 | ✅ judged | 三信号正交监测栈成形；生成段没有单一信号够用。 |
| [`G4-calibration`](axes/G-policy/G4-calibration/README.md) · 校准 | ✅ judged | 校准跨规模存活是本项目最稳的资产之一。 |

## R · 参照线（必须打败的基线）（3 格）

[轴说明](axes/R-references/README.md) ｜ 每条创新线面对的现成行业答案——不打败它，创新就没有意义。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`R1-quantization`](axes/R-references/R1-quantization/README.md) · 量化 | ⬜ todo | **已有**：本地已实测 KV 量化的部署口径价签与质量——**块标尺（沿 head_dim，G=32）** 下 int8 每位置 6528 B = fp16 的 ×0.53（省 46.875%）… |
| [`R4-window-kv-quant-retrieval`](axes/R-references/R4-window-kv-quant-retrieval/README.md) · 窗口注意力 + KV 量化 + 检索 | ⬜ todo | **已有**：本地仅覆盖窗口/锚点部分（税 +0.058~0.121 bpc；锚点通道 93.3%）。**尚缺**：按保留预算的窗口曲线 + 检索腿是否显著 >chance；量化部分按 R1 口径… |
| [`R7-existing-early-exit`](axes/R-references/R7-existing-early-exit/README.md) · 既有早退实现（CALM / LayerSkip / MoD） | ✅ judged | **本地实测（13M/enwik8/12k 步＝1 epoch 快照、每臂单 seed、n=1）**：三种标准实现都装好并跑通。① **CALM 式置信门控**：两套标定口径**并报** —— 保… |

## P · 修饰符（不是轴）（5 格）

[轴说明](axes/P-modifiers/README.md) ｜ 只改变实现/成本/条件的决策：精度、参数插入与初始化、归一化位置、损失权重、数据构造。

| 格子 | 状态 | 判读 |
|---|---|---|
| [`P1-precision`](axes/P-modifiers/P1-precision/README.md) · 精度（修饰符） | ✅ judged | reward/判据类损失必须 float32 + pre-step grad-norm gate。 |
| [`P3-param-insertion-init`](axes/P-modifiers/P3-param-insertion-init/README.md) · 参数插入与初始化（修饰符） | ✅ judged | 恒等初始化 adapter 在 13M matched-step 族内最优——「从基线出发学 remap」是对的。 |
| [`P4-normalization`](axes/P-modifiers/P4-normalization/README.md) · 归一化位置（修饰符） | ⬜ todo | **已有**：pre-LN + 无 final LayerNorm 是「同空间」几何的前提，已固定使用。**尚缺**：post-LN、QK-norm、RMSNorm 的对照判决（待补做）。 |
| [`P5-loss-weighting`](axes/P-modifiers/P5-loss-weighting/README.md) · 损失权重与调度（修饰符） | ✅ judged | 浅层降权单调反向；退火是当前最优配方的组成部分。 |
| [`P6-data-construction`](axes/P-modifiers/P6-data-construction/README.md) · 数据与任务构造（修饰符） | ✅ judged | 任务构造本身是本项目最大的单点偏差来源（enwik8 局部统计极强）。 |

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
