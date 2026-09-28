# Findings — mechanism conclusions ledger

弹性语法的逐机制结论账本（SSOT）。每机制：**税**（Δ clean bpc vs B0）|
**收益**（各自货币）| **裁决** | 证据。叙事、中间值、逐 run 表在
`docs/report.md`（retry 矩阵 + D2–D4 见 §3.7；Slider §3.9；规模 §4.6）；
运行表 `docs/experiments.md`。空白 = 未测，不是 0。

- 尺度 enwik8，13M（d=256/12L）；B0 = **1.5074 bpc**（1.1 epoch 快照，
  4.4 epoch 渐近 1.3459）。Δ = clean-eval bpc 税 vs B0。
- 证据 `.scratch/*/evidence/`（tickets 04–09、11–13）；引用 retry 数前先读 ticket 11。
- 所有 run 单/双 seed；**方差**（ticket 11）：nondeterminism 0.0019，种子方差 ≤0.0138——**Δ < ~0.014 不可排序**。
- 所有 run ~1.1 epoch、cosine LR 仍在退火中，**税可能是收敛率差异而非渐近差异**；
  **所有已发布绝对税是 1.1-epoch 上界**：retry 税 +0.364@1k step → +0.122@24k step，
  ~3k 起平台，比已发布 +0.147 小 ~20%（ticket 11 D5）。

## Redo (single-layer repeat, p_redo)

- 税：Δ **+0.044**（10% 层重复 ≈ +3% 相对 PPL）；测量过的组合无干扰（IX2 失败归因 shuffle）。
- 收益：**尚无正收益**（理论收益从未按层置信度条件化）；未做食谱：eval-only
  ablation、层内自校正 probe——**最便宜的旋钮，收益列开放**。

## Skip (p_skip)

- 税：Δ **+0.175** —— 最贵的深度元素。
- 收益：**最好的 TTS 基底**（K=8 −0.047；DIVL1 再 +0.023）。未测：按层置信度自适应 skip、FLOP-matched 记账。

## Early exit (dense per-depth supervision, w_dense_exit)

- 税：Δ **+0.067**（dense 1.0 整形）；**dense 0.2 剂量税为负**（EX2 1.6097 vs
  无 dense INT2_r3 1.6698——aux 深度监督起正则作用，corrupt 也好 0.08）。
- 收益：**~iso-PPL 下 1.4× 解码**（按置信阈值退出）；深度曲线 ~6/12 平台（半栈仅 ~0.003 bpc）。
- 负结果：深度**集成失败**——置信头只排序深度、不排序逐 token 可靠性。
- **exit×retry 权衡定案（ticket 18；100M 复测 ticket 21）**：整形与 retry 是同一
  梯度的两面——probe 式头（trunk stop-grad）保 retry，浅层读出贵；dense 整形
  exit 轴好但杀 retry。**100M 复现且更锐利**：probe 税≈0 且门控更强；dense 0.2
  在 100M 恶化为 corrupt 崩溃 + retry 灾难——**剂量响应随规模变化**（13M 温和、
  100M 灾难）。两全证伪，自适应 profile（probe）是 100M 明确赢家。conf 逐深度
  校准在训练分布上极好，阈值跨 profile 需重校准。

## Shuffle (order-free layers, shuffle_locality)

- 税：Δ **+0.545** —— 独一档；层序携带 ~0.5 bpc。
- 硬边界：locality 0.5 训练，eval ≤0.5 可容忍，全随机崩（3.84）；组合毒药：
  任何含 shuffle 的组合落到 2.2+ bpc，与修复拮抗（repair 51%→38%）。
- 收益：**唯一测到的是失败鲁棒性**（层 dropout/置换部署），至今无需求。
- 裁决：**预测质量上的死路**。未测：2–4× 规模 partial locality；eval 时 order-canonicalization。

## Future-token heads (MTP block, node k predicts t_{i+k})

- 税：always-on infra（保 Δ 纯净）；n_mtp 1→2 **+0.041**（B2 1.5481）；2→3 **+0.0166**。
- 收益（间接）：承载全部弹性机制的控制信号（早退置信、retry 门、mixture 票）。
  直接预测收益**未证实**（node-2/链无集成价值；consistency loss 中性）。
- 草稿头（ticket 10）：node-2 对 t+2 acc **55.97%**（node-1 t+1 68.8%，chance 0.5%）；
  朴素概率复合失败——正确链需 node-2 条件在 node-1 采样 token 上。
- **条件链（ticket 12，2 seed）**：DeepSeek 式 concat+proj。**oracle 0.682 ≈ node-1 自身
  0.693；deploy 0.547 < direct 0.560**——机制成立、无部署收益（瓶颈草稿质量）；需批量验证。
- **n_mtp=3 + Medusa 批量验证（ticket 13）**：一次 forward 验证全部草稿，13M 上
  **逐位 == 逐位置贪心**；a₂=0.844 / a₃|a₂=0.653 → 2.40 tok/fwd，稳态 **1.68×**(k=2)。
- **spec 缩放定案（ticket 20 @100M）**：接受级联稳定（a₂ 0.80，a₃|a₂ **0.767↑**，
  2.4 tok/fwd 不变）；前向 compute-bound（宽度-3 **3.4–5.3×**）→ 墙钟 k=2 仅
  1.15–1.29×、**k=1 负**。fp 平局翻转首现（top-2 gap 0.008 处翻 argmax）——逐位相等是 13M 规模性质。

## Latent retry (loop back through a transport) — the 2×(transport × gating) matrix

- 矩阵与逐 run 表在 **docs/report.md §3.7**。
- **裁决：无任何 cell 排序可分辨（ticket 11）。** 6 个 latent cell 跨度仅 0.011 bpc；
  C2M_n1 种子散布 **0.0138** 已超整个 cell 跨度，其 mixture-vs-overwrite delta 在
  seed 0/1 间**变号**。撤回"soft×mixture 最好"等排序结论；overwrite cells 内
  transport flavor 亦不可分辨——支持的是"多一遍"的价值，不是回传几何。
- **税本身可靠**：每个 C cell 1.6535–1.6856 vs B0 1.5074，即 **+0.146…+0.178**，
  10–100× 任何测得底；此数可引用。门控的已证价值在**修复侧**（retry 是修复不是
  集成：干净流上平均 rounds 退化）；L-loop×depth 协同：12L 上 retry 更值。
- **D1 因果确认（ticket 12 P2，12000步×2 seed）**：修复 gauge 后 mixture 的 round-2
  损坏输入 bpc **单调改善 2.71→1.936**（饱和）；pre-fix 引擎发散至 **95.2**——爆炸是
  gauge 不是机制。D5 测于 pre-D1/D3 引擎；fixed engine 改变 retry 深度，需重测。
- **逐位置 latent 门控（ticket 13 实验 1）被证伪，此路不通**：`retry_gate=τ` 开火但
  r2 bpc 恒为 direct 固定点值（direct 重入状态 == h，`torch.where` 无可切换）；linear
  ckpt 上 ~15% 开火使 r2 劣于 r1；gated 训练税未缩。**状态掩码不省算力，收益需稀疏
  计算**；仍 open 的是 prob0 门控逐位置 **re-embed**（token 重入）。

- **分配选择性成立（ticket 14，3 checkpoint 交叉验证）——愿景核心命题获证**：
  按 round-0 prob0 分桶测 r1→r2 收益，门控（最低 15% 位置开第二轮）vs 均匀：
  **INT2 +0.0211 vs −0.0193**；**C1M +0.1284 vs +0.0570**（2.2 倍）；门控把重试从
  净有害翻成净有益（2×→1.15×）。收益形状：中低置信带 +0.04~+0.06，高置信带
  ≈0/负。**限制：dense-exit 监督摧毁重试精炼** → 分配器底座不带 dense-exit；
  缺陷 D2/D3/D4 见 report §3.7。

## Slider — the runtime allocator (Rung 3, ticket 15)

- **端到端建成并验证**（`pathlm/slider.py`；证据 `.scratch/15-slider-rung3/evidence/`，
  底座 INT2_r3，clean 1.6698）：校准 → 双货币成本模型 → 求解器 → 验证解码 → 在线代理。
- **成本可预测**：fire 曲线 split gap ≤ 0.004；a2 部署口径校准后全线预测误差
  **≤ ±13%**（TF 估 a2 系统性低估，偏保守）。**双货币必须双报**（corrupt 下
  k=1 τ=0.98 = 0.635 fwd/tok @ 2.0502 严格占优 plain；flop 口径 +39.5% 换 −0.0043）。
- **在线代理**：TF 窗口 prob0 ≈ P(当前 token 正确)，ECE 0.0011–0.0079；跨 τ 与
  next-token acc 相关 r=0.91–0.98。
- 引擎无损失：单轮重试 mixture==overwrite（设计保证，oracle 证伪）。开放：exit
  不在前沿内；flop 口径 cost<1.0 不可达（需深度头）。

## Scale (Rung 4, ticket 16) — 100M, 2 seeds

- **税 @100M = +0.126**（B0 1.4612 vs C1 1.5871；种子配对 +0.118/+0.134）。
  13M +0.147 → 29M +0.174 → **100M +0.126：税首次不随规模增长**（混杂：
  batch 13M:32 → 100M:8）。
- **能力溢价持续**：同损坏下 B0 崩至 3.551（r2 反而 4.18）；C1 修复 2.015→1.949
  （r3 饱和）；**E2E 差距 ≥1.54 bpc**；ECE ≤0.0091，prob0 校准在 100M 存活。

## Slider @100M (ticket 17) — 分配器闭环上规模

- **全链在同一 100M 权重上自洽成立**（INT_100M_s0，INT2 协议、p_needle=0，
  clean 1.7175 vs B0_100M 堆叠溢价 +0.256，13M 同口径 +0.309 收窄）：
  成本预测 clean +0.7% / corrupt ≤11.1%；fire split gap ≤0.0028；prob0 ECE
  0.0014/0.0096；spec 严格占优复现（0.566 fwd/tok @ 2.0954 vs plain 2.0961）。
- **门控重试质量杠杆在 100M 消失**（τ 轴平坦 ±0.002；低置信带 ≤0.001、超出后
  为负）；轮次价值仍在训练侧（C1_100M forced-round r1→r2 −0.066）。
- **劣化检测方向修正（ticket 22 复核）**：原判“13M 自由生成段反向（吸引子混淆）→
  100M 恢复正向（0.984→0.948，开火率 0.9%→8.9%）——13M 失败是容量现象”；
  ticket 22 同权重同协议 n=6 复刻**未复现 13M 反向**（corrupt prob0 0.932 < clean
  0.996，开火 13.0%>0.6%）——原反向系 n=1 单 prompt 假象，方向本就正向且跨规模一致。

## Free-generation quality signals (ticket 22) — path disagreement

- **路径分歧度（同窗多条独立层路径 TF 重评分的成对 JS）是真实但不完备的生成段
  质量信号**（INT2_r3；证据 `.scratch/22-gen-quality/evidence/`）。判据 A 成立
  （互补路线）：prob0 高置信半区内 js 三分位错误率 0.12→0.37→0.48，js AUC
  0.66–0.69 > prob0 0.58–0.62（但 js 顶桶饱和，桶单调性主路线输给 prob0）。
  B 成立：生成段窗口 js 严格 real 0.0114 < corrupt 0.0250 < random 0.0390（prob0
  压在 0.955–1.00，分不开 real/random）。C 成立：同窗注入 0/7.5/15% wrong 合并
  js 单调 0.0251/0.0314/0.0375（15/18 窗单调）。
- **盲区：吸引子塌缩窗 js→0 且 agree=1.0**（注入 15% 也不动；随机 prompt+流损坏域
  js 反而 0.0084 < clean 0.0393）——js 只能作互补信号，不能单独监测塌缩。
- **ticket-15“13M prob0 反向”未复现**（同权重同协议 n=6：corrupt 0.9323 < clean 0.9961，方向转正）——原反向系 n=1 假象（账本已修正）。
- **塌缩域监测已定案（ticket 22b）：重复率成立**——distinct-2 区分塌缩/健康窗 **AUC 0.9815**（人工语义标注，非循环；契约阈值规则在 greedy 全局重复偏置下退化为全塌缩，24 窗全部低于自然 enwik8 水平 0.46）；塌缩窗 js 不升反降（0.035<0.046）且**双向失灵**：假阴（js=0.000 逐位一致）与假阳（字节平铺不确定循环 js=0.373）并存，与重复率仅 rank_corr 0.14 正交。监测栈定形：prob0 管 token 对错（塌缩域 0.976–0.985 盲）+ js 管分歧 + distinct-2 管塌缩（零额外前向）。

## Token retry (discrete re-entry) — see matrix rows C5/R1

- 未门控 argmax re-embed 量化掉不确定性（R1 **−4.6pp**，ECE 0.001→0.015）；mixture
  vote 消除灾难（C5 **−1.3pp**，ECE 0.007）。修复效应 ~3–4.5pp vs 0.16pp 采样误差
  （n≈98k），过 ticket 11。
- bpc 侧不成立：R1/C5 仅在 worst latent cell 上方 0.009–0.017，落在 0.0138 种子
  散布内——“token round 低于 base”仅 suggestive、未确立。

## Input corruption (the I family) — flagged beats silent

- 各元素 [税 | 15% 损坏下修复 acc]：`[mask]` 替换 **+0.059 | 59.4%**；wrong-token
  **+0.127 | 52.8%**；embedding noise **±0.000**（无 flag）；pure-noise latent **+0.063**。
- **带显式“I don't know” flag 的损坏既更便宜又更可修复**（vs 静默对抗性损坏）——
  writable-input 接口应显式标注损坏。组合胜：**retry × corruption 次可加**
  （IX1 +0.087 < 0.277 之和）——共享修复机器，唯一已证组合增益。

## Attention distance penalty (X1) — a free regularizer

- 税：Δ **−0.007**（在噪声内但正号）——唯一改善 clean bpc 的元素。收益：与驱逐
  组合共 +0.004；**改善 anchor 通道（needle 76.8%→93.3%）**；per-head 遥测距离 26–43。
  警告：penalty 下 beyond-window needle “命中”是 local-LM 强度，非抗驱逐召回。

## Eviction + anchors (X2) — the ring-buffer deployment story

- 税：Δ +0.121（原 X2；v2 needle 重设计 +0.058——改善是测量不是机制）。
  收益：ring-buffer 解码免重 prefill；anchor 通道 needle 召回 76.8%
  （+penalty 93.3%）；retry 轮间 KV 位置严格递增（P0 类 bug，须持续测试）。
- **损坏 × 复制 = 目标冲突，三路尝试全失败（ticket 13 实验 2）**：anchor 召回
  0.764（X2v2 无损坏）→ X2C 无豁免 **0.016**、X2C-v2 锚区位置豁免 `corrupt_spare_anchors`
  **0.033**（in_window 0.47%→0.52%、beyond 0.59%→0.67%，clean bpc +0.0044）、X2Cv3
  needle 批次整体豁免 `needle_corrupt_free` **0.035**（且 clean bpc 1.680→**1.997**）。
  位置豁免救不回 4%；任务级豁免也没救回且 bpc 更糟。
- **裁决：损坏鲁棒与精确复制在单模型里目标冲突，修复必须靠显式模式/通道信号**，
  不能靠位置或批次的损坏豁免（非锚 needle 仍低是全局"别抄"策略，单点豁免不够）。
- 附带：`needle_acc` 评测协议 bug（漏清 `corrupt_mask`）已修并有可失败单测钉住；verdict-neutral（阳性对照 X2v2 复现 0.8658/0.7637）。

- **用户裁决（2026-09-16）：无需模式信号**。设计原则——模型从不"保留错误"，
  **目标恒为正确的 token**（损坏只是输入侧增广）。实测限制：13M 下损坏训练
  会压低精确复制行为（X2Cv3 连干净 needle 批次也未恢复 0.05，clean bpc
  1.680→1.997）——属容量/训练规模限制。needle 保持 backlog。

## Diversity pressure (TTS, div_weight)

- 机制：并行路径间 capped −JS 压差（bf16 下 float32 + pre-step grad-norm gate）。
  收益：**同时改善 baseline 与 TTS 斜率**（DIVL1 −0.062 > plain −0.047）；税 Δ +0.023。
  确定性路径 checkpoint 从 K-path 集成恰得 0——多样性必须练进去。

## Cross-cutting laws

1. **元素税分割**：破坏 identity/order 结构的机制（wrong-token、skip、shuffle）贵
   （Δ 0.13–0.55）；保留或 flag identity 的（mask、noise、redo、penalty）~免费（Δ ≤ 0.07）。
2. **TTA 增益 ∝ 训练路径多样性**：确定性路径 ckpt 从 K-path 集成恰得 0；多样性必须
   练进去；DIVL1（capped −JS，float32 + grad-norm gate）同时改善 baseline 与斜率。
3. **组合有选择性**：共享机器时次可加（corruption × repair）；一方破坏另一方所需结构
   时拮抗（shuffle × everything）。
4. **辅助输出是 fallback 不是 voter**：深度前缀与 retry 轮按修复/退出训练，混进预测会
   退化。要集成价值需当 co-equal predictor 监督（未测）。
7. **整形 vs 自适应**（ticket 18/21）：trunk 被浅层目标塑形（exit 质量好）就毁
   retry 精炼；保 retry 就浅层贵——同一梯度的两面，以 profile 选择而非单模型
   两全。**剂量响应随规模变化**：dense 0.2 在 13M 温和、100M 灾难；probe 在
   100M 零税且门控更强。
5. **reward 项的稳定性工程**：bf16 autocast 下任何 negative/reward loss 需 float32 +
   pre-step grad-norm gate（两次发散教会）。
6. **规模**（Rung 4 定案 @100M，2 seeds）：税首次不增（+0.126）、能力溢价持续
   （E2E 差距 ≥1.54 bpc，B0 脆性加深）——13M 的"税主导"判决在 100M 不再成立。

## Depth-adaptive AR (ticket 23) — 方向修订后的新形态首验

- **形态 v1**（mental_model §0 方向，规格见 `.scratch/23-depth-ar/issues/01-sanity.md`）：
  纯 AR 无 corruption/retry；全深度并行监督（depth 0..L 各挂 tied-unembed fp32
  读出，等权 CE）；逐通道携带门 h_{k+1}=h_k+g⊙Refine(h_k)（偏置 −2 恒等起步）；
  训练随机退出深度 dᵢ~U{0..L}，已退出行深层 KV = kv_proj(冻结表征)（混合深度
  context）；提交 soft(Σp·E[t])/hard/latent。代码 `pathlm/depth_ar.py`；混合
  填充/解码器等价/门语义由 14 单测钉死（变异验证可失败）。
- **四性质 + 一判据**（enwik8 小模型，d=128/256，L=4/8，29.5M tokens，总 GPU
  11.4 min；证据 `.scratch/23-depth-ar/evidence/SUMMARY.md`）：
  - 深度精化 **成立**：depth-bpc 全 run 单调降（B: 6.250→2.4161；深层平台形态
    与 L2 一致）；门行为健康（浅层开 g1≈0.18–0.43、深层微开 0.04–0.11，无发散）。
  - **宽度-workspace 假设成立**：d=256>V=206 时状态把 ~10% 能量放在读出不可见
    null 空间（62 维）且 26 条 (batch,方向,±) 扰动响应 >δ —— 携带信息确被后续
    读取；d=128<V 时 null 维结构性为 0。探针口径 = null 方向 ±5% 扰动 → 后续
    depth-L logits 响应（当前行对照 3.4e-05 通过）。
  - **出口提交可用**：半栈出口（B depth-4/8）TF 草稿接受率 **0.9667**、depth-1
    已 0.80；18/18 深度×提交方式生成合法无发散。意外：**soft 提交的 OOD 期望
    嵌入污染 context**（生成文本 rescore 6.9 vs hard 1.2）——soft 默认待重验；
    latent 直通两极（塌缩自洽 0.52 / 漂移 8.5）符合预判。
  - **形态税判据证伪**：B depth-L 2.4161 vs dense-B 1.7375 = **+0.6786 ≫ 0.05**。
    归因 ablation（B-nofill：去随机退出/填充、保留全深度监督）→ **监督稀释
    +0.512 主因、混合填充扰动 +0.167 次因**。代码 bug 排除（语义单测钉死），
    属配方层证伪：**等权全深度监督不保底座**。
- **对 stage-2 的直接推论**：(a) 监督需深度加权/浅层降权/浅头 stop-grad（回环
  时代 probe profile 教训可复用）；(b) 混合填充 fill 变体（冻结 KV 直通 vs
  重投影）未测，值一个 ablation；(c) 出口轴（0.97@半栈）是本形态最强资产，
  自 spec 组合优先于深度均摊训练。
- **23b 监督加权 sweep（ticket 02，形态税归零搜索）**：浅层 aux 权重
  w∈{1, 0.1, 0.02, 1→0.1 退火}（4×sanity-B 配置，10.8 min）——**税归零全败且
  单调反向**（w 越低 depth-L 越差：2.45→2.57→2.73，税 +0.71→+0.99；退火臂
  2.4444 保底座但税仍 +0.71）。**归因改判**：浅层全深度监督是表示正则而非
  ticket-23 §4 所判「稀释主因」——carry 门乘性瓶颈/欠拟合升级为税主嫌
  （H1: dense+aux-only 测纯 aux 税；H2: 12k 步；H3: stop-grad 浅头）。
  soft OOD 污染复现（d4 7.4 vs hard 1.2）且幅度随浅头质量耦合。监督加权轴
  关闭，等权=已知最优。证据 `.scratch/23-depth-ar/evidence/`（23b-*，112 测试绿）。
