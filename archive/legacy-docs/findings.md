# Findings — mechanism conclusions ledger

弹性语法的逐机制结论账本（SSOT）。每机制：**税**（Δ clean bpc vs B0）|
**收益**（各自货币）| **裁决** | 证据。叙事、逐 run 表在 `docs/report.md`
（retry 矩阵 + D2–D4 见 §3.7；Slider §3.9；规模 §4.6）；运行表 `docs/experiments.md`；
深度自适应 AR 逐臂过程在 `docs/report_details.md` §E。空白 = 未测，不是 0。

- 尺度 enwik8，13M（d=256/12L）；B0 = **1.5074 bpc**（1.1 epoch 快照，4.4 epoch
  渐近 1.3459）。证据 `.scratch/*/evidence/`（索引 `docs/artifacts.md`）；
  引用 retry 数前先读 ticket 11。
- **方差**（ticket 11）：nondeterminism 0.0019，种子方差 ≤0.0138——**Δ < ~0.014
  不可排序**；所有 run ~1.1 epoch，**已发布绝对税是 1.1-epoch 上界**（D5：
  +0.364@1k → +0.122@24k；渐近口径 ticket 19）。

## Redo / Skip — 两个深度元素

- **Redo**：税 Δ **+0.044**（IX2 失败归因 shuffle）；**尚无正收益**（从未按层置信
  条件化）；eval-only ablation / 层内 probe 未做——最便宜旋钮，收益列开放。
- **Skip**：税 Δ **+0.175**——最贵深度元素；**最好的 TTS 基底**（K=8 −0.047，
  DIVL1 再 +0.023）。未测：按层置信自适应 skip、FLOP-matched 记账。

## Early exit (dense per-depth supervision, w_dense_exit)

- 税：Δ **+0.067**（dense 1.0 整形）；**dense 0.2 剂量税为负**（EX2 1.6097 vs
  无 dense 1.6698——aux 起正则作用，corrupt 也好 0.08）。
- 收益：**~iso-PPL 下 1.4× 解码**；深度曲线 ~6/12 平台（半栈仅 ~0.003 bpc）。
  负结果：深度**集成失败**——置信头只排序深度、不排序逐 token 可靠性。
- **exit×retry 权衡定案（ticket 18；100M 复测 ticket 21）**：整形与 retry 是同一
  梯度的两面，两全证伪——probe 式头（trunk stop-grad）保 retry（−0.0098）但浅层
  读出贵（depth 6 +0.93）；dense 整形 exit 轴好（+0.199）但杀 retry（+0.0146）。
  **100M 复现且更锐利**：probe 税≈0 且门控更强（−0.029）；dense 0.2 恶化为
  corrupt 崩溃（+1.73）——**剂量响应随规模变化**。自适应 profile（probe）是
  100M 明确赢家；阈值跨 profile 需重校准。

## Shuffle (order-free layers, shuffle_locality)

- 税：Δ **+0.545** —— 独一档；层序携带 ~0.5 bpc。硬边界：locality 0.5 训练，
  eval ≤0.5 可容忍，全随机崩（3.84）；组合毒药：任何含 shuffle 的组合落到 2.2+
  bpc，与修复拮抗（repair 51%→38%）。
- 收益：唯一测到的是失败鲁棒性。裁决：**预测质量上的死路**。
  未测：2–4× 规模 partial locality；eval order-canonicalization。

## Future-token heads (MTP block, node k predicts t_{i+k})

- 税：n_mtp 1→2 **+0.041**（B2 1.5481）；2→3 **+0.0166**。
- 收益：承载全部弹性机制的控制信号（早退置信、retry 门、mixture 票）+ spec 草稿。
  node-2 t+2 acc **55.97%**（node-1 68.8%，chance 0.5%）；朴素概率复合失败
  （CE 3.96 vs 1.58——正确链需 node-2 条件在 node-1 采样 token 上）。
- **条件链（ticket 12，2 seed）**：DeepSeek 式 concat+proj。oracle 0.682 ≈ node-1
  0.693；**deploy 0.547 < direct 0.560**——机制成立、无部署收益（瓶颈草稿质量）。
- **n_mtp=3 + Medusa 批量验证（ticket 13）**：一次 forward 验证全部草稿，逐位 ==
  逐位置贪心；部署 a₂=0.844 / a₃|a₂=0.653 → 2.40 tok/fwd，稳态 **1.68×**(k=2)。
- **spec 缩放定案（ticket 20 @100M）**：接受级联稳定（a₂ 0.80、a₃|a₂ **0.767↑**、
  2.4 tok/fwd 不变）；前向 compute-bound（宽度-3 **3.4–5.3×** vs 13M 1.28×）→
  **墙钟 k=2 仅 1.15–1.29×、k=1 负**。fp 平局翻转首现（top-2 gap 0.008，
  1/400 token）——逐位相等是 13M 规模性质。

## Latent retry (loop back through a transport)

- **裁决：无任何 cell 排序可分辨（ticket 11）**。6 个 latent cell 跨度仅 0.011 bpc
  < 单格种子散布 0.0138；撤回全部排序结论（含"soft×mixture 最好"）。支持的是
  "多一遍"的价值，不是回传几何。逐 run 表与 D1–D6 缺陷审计见 report §3.7。
- **税本身可靠**：C cell 1.6535–1.6856 vs B0 1.5074 = **+0.146…+0.178**，10–100×
  任何测得底。门控的已证价值在**修复侧**（retry 是修复不是集成）；L-loop×depth
  协同：12L 上 retry 更值。
- **D1 因果确认（ticket 12，2 seed）**：修复 gauge 后 mixture r2 单调 2.71→**1.936**
  （饱和）；pre-fix 发散至 **95.2**——爆炸是 gauge 不是机制。
- **逐位置 latent 门控（ticket 13）证伪，此路不通**：direct 重入状态 == h（固定点）；
  linear ~15% 开火使 r2 劣于 r1；状态掩码不省算力，需稀疏 gather/scatter。
  仍 open：prob0 门控逐位置 **re-embed**（token 重入）。
- **分配选择性成立（ticket 14，3 checkpoint 交叉验证）——愿景核心命题获证**：
  按 round-0 prob0 分桶，门控（最低 15% 开第二轮）vs 均匀：**INT2 +0.0211 vs
  −0.0193**；**C1M +0.1284 vs +0.0570**（2.2 倍）；收益在中低置信带 +0.04~+0.06，
  高置信带 ≈0/负。**限制：dense-exit 监督摧毁重试精炼**→ 底座不带 dense-exit。
- **Token retry（C5/R1）**：未门控 argmax re-embed 量化掉不确定性（R1 **−4.6pp**，
  ECE 0.001→0.015）；mixture vote 消除灾难（C5 **−1.3pp**）。修复效应 3–4.5pp vs
  0.16pp 采样误差；bpc 侧落在种子散布内，仅 suggestive。

## Slider — the runtime allocator (Rung 3–4, tickets 15–19)

- **端到端建成并验证**（`pathlm/slider.py`；底座 INT2_r3 clean 1.6698）：
  校准 → 双货币成本模型 → 求解器 → 验证解码 → 在线代理。**成本可预测**：
  fire split gap ≤ 0.004；a2 校准后预测误差 **≤ ±13%**。**双货币必须双报**
  （corrupt 下 k=1 τ=0.98 = 0.635 fwd/tok @ 2.0502 严格占优 plain；flop 口径
  +39.5% 换 −0.0043）。**在线代理**：TF 窗口 ECE 0.0011–0.0079；引擎无损失：
  单轮重试 mixture==overwrite（设计保证）。
- **@100M（ticket 17）**：全链自洽（堆叠溢价 +0.256，13M +0.309 收窄；成本预测
  clean +0.7% / corrupt ≤11.1%；spec 严格占优复现 0.566 fwd/tok @ 2.0954）；
  **门控重试质量杠杆在 100M 消失**（τ 轴平坦 ±0.002；轮次价值仍在训练侧
  r1→r2 −0.066）。
- **Scale 定案（ticket 16，2 seeds）**：税 @100M = **+0.126**（13M +0.147 →
  29M +0.174 → 100M +0.126，首次不随规模增长；混杂 batch 32→8）；能力溢价持续：
  B0 损坏崩至 3.551（r2 反而 4.18）、C1 修复 2.015→1.949（r3 饱和）、**E2E 差距
  ≥1.54 bpc**、ECE ≤0.0091。
- **渐近口径定案（ticket 19，4.4 epoch，2 seeds）**：税 +0.126 → **+0.082**
  （曲线单调降，收缩 35%）；**等训练口径 13M +0.122 → 100M +0.082——税真正随
  规模下降**（快照口径"不缩"系 1.1-epoch 上界）。能力溢价随训练**扩大**（corrupt
  gap 1.54→1.89）。B0_asym 1.341 ≈ 13M 渐近 1.346：data-limited。

## Free-generation quality signals (ticket 22/22b) — path disagreement

- **路径分歧度（同窗多条独立层路径 TF 重评分的成对 JS）是真实但不完备的生成段
  质量信号**（INT2_r3）：A 成立（互补：prob0 高置信半区内 js 三分位错误率 0.12→0.48、
  AUC 0.66–0.69 > prob0 0.58–0.62）；B 成立（生成段 js 严格 real 0.0114 < corrupt
  0.0250 < random 0.0390）；C 成立（注入剂量单调）。**盲区：吸引子塌缩窗 js→0**。
- **ticket-15"13M prob0 反向"未复现**（同权重同协议 n=6：corrupt 0.9323 < clean
  0.9961）——原反向系 n=1 假象（账本与 ticket 17 判读已修正）。
- **塌缩域监测（ticket 22b）：重复率成立**——distinct-2 区分塌缩/健康窗 **AUC
  0.9815**（人工标注）；塌缩窗 js 双向失灵（假阴 js=0.000 与假阳 js=0.373 并存，
  rank_corr 0.14 正交）。监测栈定形：**prob0 管 token 对错 + js 管分歧 +
  distinct-2 管塌缩**（零额外前向；greedy 全局重复偏置需相对化参考水平）。

## Input corruption (the I family) — flagged beats silent

- 各元素 [税 | 15% 损坏下修复 acc]：`[mask]` 替换 **+0.059 | 59.4%**；wrong-token
  **+0.127 | 52.8%**；embedding noise **±0.000**；pure-noise latent **+0.063**。
- **带显式"I don't know" flag 的损坏既更便宜又更可修复**（vs 静默对抗性损坏）。
  组合胜：**retry × corruption 次可加**（IX1 +0.087 < 0.277 之和）——共享修复
  机器，唯一已证组合增益。

## Attention distance penalty (X1) — a free regularizer

- 税：Δ **−0.007**——唯一改善 clean bpc 的元素。收益：与驱逐组合共 +0.004；
  **改善 anchor 通道（needle 76.8%→93.3%）**。警告：penalty 下 beyond-window
  needle "命中"是 local-LM 强度，非抗驱逐召回。

## Eviction + anchors (X2) — the ring-buffer deployment story

- 税：Δ +0.121（原 X2；v2 needle 重设计 +0.058——改善是测量不是机制）。
  收益：ring-buffer 解码免重 prefill；anchor 通道 76.8%（+penalty 93.3%）；retry
  轮间 KV 位置严格递增（P0 类 bug，须持续测试）。
- **损坏 × 复制 = 目标冲突，三路尝试全失败（ticket 13）**：anchor 召回
  0.764（X2v2 无损坏）→ 无豁免 **0.016** / 锚区位置豁免 **0.033** / needle 批次
  整体豁免 **0.035**（且 clean bpc 1.680→**1.997**）。裁决：修复必须靠显式
  模式/通道信号，不能靠位置或批次的损坏豁免。附带：`needle_acc` 评测泄漏
  （漏清 `corrupt_mask`）已修并有可失败单测钉住；verdict-neutral（阳性对照
  X2v2 复现 0.8658/0.7637）。
- **用户裁决（2026-09-16）：无需模式信号**——目标恒为正确 token，从不"保留错误"；
  13M 下损坏训练压低精确复制属容量/规模限制。needle 保持 backlog。

## Diversity pressure (TTS, div_weight)

- capped −JS 压差（bf16 下 float32 + pre-step grad-norm gate）。**同时改善
  baseline 与 TTS 斜率**（DIVL1 −0.062 > −0.047，上限 +0.023）；税 +0.023。
  确定性路径 checkpoint 从 K-path 集成恰得 0——多样性必须练进去。

## Cross-cutting laws

1. **元素税分割**：破坏 identity/order 结构的机制（wrong-token、skip、shuffle）贵
   （Δ 0.13–0.55）；保留或 flag identity 的（mask、noise、redo、penalty）~免费
   （Δ ≤ 0.07）。
2. **TTA 增益 ∝ 训练路径多样性**：确定性路径 ckpt 从 K-path 集成恰得 0；DIVL1
   （capped −JS）同时改善 baseline 与斜率。
3. **组合有选择性**：共享机器时次可加（corruption × repair）；一方破坏另一方所需
   结构时拮抗（shuffle × everything）。
4. **辅助输出是 fallback 不是 voter**：深度前缀与 retry 轮按修复/退出训练，混进
   预测会退化。
5. **reward 项的稳定性工程**：bf16 autocast 下任何 negative/reward loss 需 float32
   + pre-step grad-norm gate（两次发散教会）。
6. **整形 vs 自适应**（ticket 18/21）：trunk 被浅层目标塑形（exit 质量好）就毁
   retry 精炼；保 retry 就浅层贵——同一梯度的两面，以 profile 选择而非单模型两全；
   剂量响应随规模变化（dense 0.2：13M 温和、100M 灾难）。
7. **规模（回环族定案 @100M）**：快照税不增（+0.126）、渐近税随规模下降
   （+0.122→+0.082）、能力溢价随规模与训练扩大——**此经验不外推到深度自适应
   形态**（见下节：同引擎口径下斜率为正）。

## Depth-adaptive AR (tickets 23–25) — 方向修订后的新形态；「等智商免费加速」收口

> 方向与哲学演化见 `docs/mental_model.md` §0；逐臂判读过程见
> `docs/report_details.md` §E；证据 `.scratch/23~25-*/evidence/`；叙事见 `docs/retrospective.md` §5–§7。

- **机制四性质成立（ticket 23，13M 级，GPU 11.4 min）**：深度精化单调（B:
  6.250→2.4161）；宽度-workspace 假设成立（d=256>V null 62 维/26 条功能性读取，
  d=128 结构性 0）；出口提交可用（半栈 TF 接受率 **0.9667**）；门浅开深微开。
  代码 `pathlm/depth_ar.py`，语义由 14 可失败单测钉死。
- **形态税 @13M 归因链（+0.6786 → 23b/23c/23d/23e 四轮收敛）**：监督加权 sweep
  全败且单调反向（23b：w 越低越差 2.45→2.73——浅层监督是表示正则非稀释主因）；
  欠拟合主因（23c：3.33× 训练税 +0.713→+0.121；浅头 stop-grad 恶化排除）；matched
  真税 @12k = **+0.3295**（23d）；**fill 胜 nofill**（23e：1.8582 vs 1.9438——
  23 首验"填充负分量"系欠拟合伪影+对照混淆）、**proj-fill 新最优 1.8356**。
- **规模判决（ticket 24/24b @100M，同 token 预算 98.3M）**：快照税 **+0.4563**>
  13M matched +0.31——24b 拆混杂后干净归属：**形态 +0.44（96%）+ 引擎 +0.02（4%）
  + 配方 ≈0**（等权臂 1.9200 ≈ final-2× 1.9175；同引擎 dense-AR 1.4813 ≈ B0
  1.4612）；**同引擎形态税 +0.4362，13M +0.31 → 100M +0.44 斜率为正**。出口资产
  系全深度监督产物（dense trunk tf@4 0.12 vs 形态臂 0.97）。
- **归因收口（ticket 25，三臂 @100M 同引擎，判读表四格落第 4 格）**：H-短视
  （结构切分 readout 360/workspace 360）**否证且反向**——clean 2.1475、恢复
  **−50.4%**：模型把状态能量整体迁出读出空间（切片能量 depth0 0.85→depth2
  0.026→≈0、null 100% 但 functional_dirs=176）、门动力学反转、曲线尾部倒退——
  **涌现 workspace 优于结构强加切分**。H-时长（48k+冷却退火）名义 +16.4%（<50%
  线）且同引擎 48k 锚拆解**形态税零收缩**（+0.4372@48k ≈ +0.4362@24k，C 对 A
  改善 0.0746 与 dense 侧训练收益 0.0756 逐位镜像）——**税结构性存在，
  「等智商免费加速」正式证伪**。
- **交易终值**：+0.38~0.44 bpc 底座损失换 exit-4 ≈3.1× / uniform-exit ≈1.8× 墙钟、
  ≈1.5× 等效验证解码（FLOPs 口径，spec_cost_model 对账）与出口资产（三形态臂
  tf@4 ≥0.95 稳健）；当前最优配方 = C（48k+冷却，等效吞吐 122.8 tok/s vs A
  112.9，+9%）。**税只能被交易，不能被收。**
- **贯穿副发现**：soft 提交 OOD 污染 context 三度复现（rescore 峰值 8.4 vs hard
  0.8）——soft 默认待重验；门 @100M 全层收小（等权/2× 同款，非配方现象，Q8 剩
  规模候选）。
