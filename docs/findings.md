# Findings — mechanism conclusions ledger

弹性语法的逐机制结论账本（SSOT）。每机制：**税**（Δ clean bpc vs B0）|
**收益**（各自货币）| **裁决** | 证据。叙事、中间值、逐 run 表在
`docs/report.md`（retry 矩阵 + D2–D4 见 §3.7）；运行表 `docs/experiments.md`。
空白 = 未测，不是 0。

- 尺度 enwik8，13M（d=256/12L）；B0 = **1.5074 bpc**（1.1 epoch 快照，
  4.4 epoch 渐近 1.3459）。Δ = clean-eval bpc 税 vs B0。
- 证据 `.scratch/*/evidence/`（tickets 04–09、11–13）；引用 retry 数前先读 ticket 11。
- 所有 run 为单/双 seed；**方差**（ticket 11）：同种子 nondeterminism 0.0019；
  per-cell 种子方差 0.0001–0.0138。**Δ < ~0.014 不可排序**（含大部分 retry 矩阵
  与若干主效应）。
- 所有 run ~1.1 epoch、cosine LR 仍在退火中，**税可能是收敛率差异而非渐近差异**；
  **所有已发布绝对税是 1.1-epoch 上界**：retry 税 +0.364@1k step → +0.122@24k step，
  ~3k 起平台，比已发布 +0.147 小 ~20%（ticket 11 D5）。

## Redo (single-layer repeat, p_redo)

- 税：Δ **+0.044**（10% 层重复 ≈ +3% 相对 PPL）；测量过的组合无干扰（IX2 失败归因 shuffle）。
- 收益：**尚无正收益**；with/without 未做，两个食谱：(a) eval-only ablation（p_redo→0）；
  (b) 层内自校正 probe（同输入 1× vs 2× 的 frozen-head CE）。理论收益（层内自校正、
  层级自适应算力）从未按层身份/置信度条件化。
- 裁决：**最便宜的旋钮，收益列开放**。

## Skip (p_skip)

- 税：Δ **+0.175** —— 最贵的深度元素。
- 收益：**最好的 TTS 基底**。采样路径有真实多样性（K=8 1.6926→1.6454，−0.047）；
  多样性压力训练（DIVL1）两项同时改善：baseline 1.6842、增益 −0.062，比 plain-skip
  K=8 上限再 +0.023。
- 未测：按层置信度自适应 skip；FLOP-matched 比较（应按省下的 FLOP 记账）。

## Early exit (dense per-depth supervision, w_dense_exit)

- 税：Δ **+0.067**。
- 收益：**~iso-PPL 下 1.4× 解码**（按置信阈值退出）；深度曲线 ~6/12 平台（半栈仅 ~0.003 bpc）。
- 负结果：深度**集成失败**（mixed 1.602 vs single 1.5744）——浅层严格更差（depth CE
  1.06→0.74，15× PPL 悬崖），置信头只排序深度、不排序逐 token 可靠性。
- 未测：退出阈值扫描（1.4× 仅一个工作点）；退出上叠 spec-decoding。

## Shuffle (order-free layers, shuffle_locality)

- 税：Δ **+0.545** —— 独一档；层序携带 ~0.5 bpc。
- 硬边界：locality 0.5 训练，eval ≤0.5 可容忍，全随机崩（3.84）。
- 组合毒药：任何含 shuffle 的组合落到 2.2+ bpc（IX2/IX3/INT）；与修复拮抗（重试
  无法精修乱序状态；repair 51%→38%）。
- 收益：**唯一测到的是失败鲁棒性**（层 dropout/置换部署），至今无需求。
- 裁决：**预测质量上的死路**。未测：2–4× 规模 partial locality（<0.25）；eval 时
  order-canonicalization（排序执行当免费"修复到规范序"）。

## Future-token heads (MTP block, node k predicts t_{i+k})

- 税：always-on infra（保 Δ 纯净）；n_mtp 1→2 **+0.041**（B2 1.5481）；2→3 **+0.0166**
  （B3 1.5604 vs B2fresh 1.5438；1→2 为 +0.0407）。
- 收益（间接，已测）：承载全部弹性机制的控制信号（早退置信、retry 门、mixture 票）。
- 直接预测收益：**未证实**——node-2/链估计无集成价值；consistency loss 中性
  （IX4 1.6589 ≈ C1 1.6549）。
- 草稿头（ticket 10）：node-2 对 t+2 acc **55.97%**（node-1 t+1 68.8%，chance 0.5%，
  M0 仅 0.26）；与 node-1 验证一致 65.8%，**给定 node-1 正确则接受 77.3%**。朴素概率
  复合失败（CE 3.96 vs node-2 单独 1.58），正确链需 node-2 条件在 node-1 采样 token 上。
- **条件链（ticket 12 P1，6000步×2 seed）**：DeepSeek 式 concat+proj
  `Linear(2d,d)([h; embed(t_{i+1})])`+T1。**oracle（真 t+1）t+2 acc 0.682 ≈ node-1 自身
  0.693；deploy（node-1 argmax）0.547 < direct 0.560**（~31% 草稿错误），接受率
  0.75 ≈ direct 0.77。**裁决：机制成立、此规模无部署收益，瓶颈是草稿质量**；k=1 naive
  verify-next-round 零吞吐收益，需批量验证。
- **n_mtp=3 + Medusa 批量验证（ticket 13 实验 3）**：`decode.py::decode_spec` 一次 forward
  验证 node-2..n_mtp + node-1 下一 token，输出**逐位 == 逐位置贪心**（5 单测）。部署口径
  a₂=0.844 / a₃|a₂=0.653 → 2.40 tok/forward；稳态 enwik8 **1.68×**(k=2)/1.42×(k=1)、
  random **1.85×**(k=2)；宽度-3 前向成本仅 1.28×。**≥1.3× 判据通过。**

## Latent retry (loop back through a transport) — the 2×(transport × gating) matrix

- 矩阵（4 transport × {ungated overwrite, mixture vote}，全带 corrupt_wrong 0.15）与逐 run
  表已移到 **docs/report.md §3.7**。
- **裁决：无任何 cell 排序可分辨（ticket 11）。** 6 个 latent cell 跨度仅 0.011 bpc；
  C2M_n1 种子散布 **0.0138** 已超整个 cell 跨度，其 mixture-vs-overwrite delta 在
  seed 0/1 间**变号**（−0.0073 → +0.0065）。撤回"soft×mixture 最好、direct×mixture
  最差"与"经 vocab 空间重入而非 latent 均值"。overwrite cells 内 transport flavor
  亦不可分辨（跨度 0.007）——支持的是"多一遍"的价值，不是回传几何。
- **税本身可靠**：每个 C cell 1.6535–1.6856 vs B0 1.5074，即 **+0.146…+0.178**，
  10–100× 任何测得底；12 层栈两遍 + 15% 损坏 ≈ +0.15 bpc。此数可引用。
- 门控的已证价值在**修复侧**：R1 −4.6pp → C5 −1.3pp（~20× 0.16pp 修复采样误差），
  forced-round 曲线停止下探。retry 是修复不是集成：干净流上平均 rounds 退化。
  L-loop×depth 协同：12L 上 retry 更值（C1 +2.1pp；M0 浅栈 ~0）。
- **D1 因果确认（ticket 12 P2，12000步×2 seed）**：修复 gauge 后 mixture run 的 round-2
  损坏输入 bpc **单调改善 2.71→2.33→2.19→2.09→2.03→1.98→1.95→1.936**（2 seed
  1.943/1.932），pre-fix 引擎发散（3.7→8.5→32.9→**95.2**）；round 序 r1 1.993→r2 1.936→
  r3 1.934 饱和。爆炸是 gauge 不是机制（probe_e2e 指标为 node-1 下一 token acc on
  损坏位置，非 node-0 自修复）。D5 测于 pre-D1/D3 引擎；fixed engine 改变 retry 深度，需重测。
- **逐位置 latent 门控（ticket 13 实验 1）被证伪，此路不通**：`retry_gate=τ` 门控重入
  状态。fixed-engine direct ckpt 上 τ 从 0→100% 开火但 r2 bpc 恒为 C1M_s0 1.9353
  （r1 1.9928）/ C1_s1 1.9287（r1 1.9861）——direct 重入状态 == h，`torch.where` 无可
  切换。linear ckpt 上 ~15% 开火使 r2 劣于 r1（bpc 保留 C2M −0.15/C2 −0.48）；6000 步
  gated 训练税未缩（1.6884 vs 1.6860）。**状态掩码不省算力，收益需稀疏计算**；仍 open
  的是 prob0 门控逐位置 **re-embed**（token 重入，与已证伪的 latent 状态掩码不同）。

- **分配选择性成立（ticket 14 实验 1，3 checkpoint 交叉验证）——愿景核心命题获证**：
  按 round-0 prob0 分桶测 r1→r2 的 node-1 bpc 收益，门控（只对最低 15% 位置开第二轮）
  vs 均匀（全位置重试）：ALLOC(dense-exit) −0.0651 vs −0.0554；**INT2 +0.0211 vs
  −0.0193**；**C1M +0.1284 vs +0.0570**。门控把重试从净有害翻成净有益（计算 2×→1.15×）；
  C1M 同一第二轮收益翻 2.2 倍。收益形状：中低置信带 +0.04~+0.06，高置信带（87%
  token）≈0/负。**限制：dense-exit 监督摧毁重试精炼**（ALLOC 各桶全负）→ 分配器
  底座 = retry + corruption，**不带 dense-exit**。证据 `.scratch/14-allocator/evidence/`。
- 缺陷 D2/D3/D4 细节见 report §3.7。

## Slider — the runtime allocator (Rung 3, ticket 15)

- **端到端建成并验证**（`pathlm/slider.py` + 根探针；证据 `.scratch/15-slider-rung3/evidence/`，
  底座 INT2_r3 重训，clean bpc 1.6698 vs 已发布 1.6548，Δ 在重训非确定性内）：
  校准 → 双货币成本模型 → 求解器（预算/质量两模式）→ 验证解码 → 在线代理。
- **成本可预测**：fire 曲线 split gap ≤ 0.004；**a2 用部署口径校准后全线预测误差
  ≤ ±13%**（判据 ±15% 过）：k=0 −2.7~−7.6%，k=1 −4.6~−12.8%；TF 估 a2 会
  系统性低估部署接受率（0.59–0.64 vs 0.74–0.80，自生成文本自一致率更高），
  预测偏保守（真实成本 ≤ 预测）。
- **双货币必须双报**：forwards（§1 口径，spec 摊销后 0.55–0.64）与 flop（等价
  width-1 单位，spec ≥1.26）结论分歧——corrupt profile 的 k=1 τ=0.98 点在
  forwards 口径比 plain **少 37% 且 bpc 更好**（严格占优复现），flop 口径则是
  +39.5% 算力换 −0.0043 bpc。
- **在线代理（prob0）**：TF 窗口上 prob0 ≈ P(当前 token 正确)，ECE **0.0011**
  (corrupt) / 0.0079 (clean)；门控后有效代理 ECE 0.0018；跨 τ 与 next-token
  acc 相关 r=0.91–0.98。
- **负结果 ×2**：① 自由生成段的 mean-prob0 劣化检测反向（损坏把生成推进
  高置信重复吸引子，0.954→0.976）——prob0 代理只适用于 ingest/prefill 窗口；
  ② 单轮重试的 mixture-vs-overwrite 引擎差距不存在（共享 gauge 设计保证，
  单项 mixture == overwrite；差异只在 round-3+）。
- 开放：exit 不在前沿内（INT2 无 dense-exit 训练）；flop 口径 cost<1.0 不可达（需深度头）。

## Token retry (discrete re-entry) — see matrix rows C5/R1

- 未门控 argmax re-embed 量化掉不确定性：错提交以"真值样" token 重入（R1 **−4.6pp**，
  ECE 0.001→0.015）；mixture vote（累积分布 argmax，anchor w=1）消除灾难（C5 **−1.3pp**，
  ECE 0.007）。修复效应 ~3–4.5pp vs 0.16pp 采样误差（n≈98k），过 ticket 11。
- bpc 侧不成立：R1 1.6775 / C5 1.6856 仅在 worst latent cell 上方 0.009–0.017，
  落在该 cell 0.0138 种子散布内（各单 seed），故"token round 在干净数据上低于 base"
  仅 suggestive、未确立。
- open：prob0 门控逐位置 re-embed（只在模型自认不确定处重写），需逐位置 accept 机制。

## Input corruption (the I family) — flagged beats silent

- 各元素 [税 | 15% 损坏下修复 acc]：`[mask]` 替换 **+0.059 | 59.4%**；wrong-token
  **+0.127 | 52.8%**；embedding noise **±0.000 | n/a**（无 flag）；pure-noise latent
  **+0.063 | n/a**。
- **带显式"I don't know" flag 的损坏既更便宜又更可修复**（vs 静默对抗性损坏）——
  writable-input 接口应显式标注损坏。噪声免费（norm-cap 几何吸收）；纯噪声
  writable-input 只 ~4% 相对 PPL。
- 组合胜：**retry × corruption 次可加**（IX1 mask×soft-retry +0.087 < 0.277 之和）——
  共享修复机器，唯一已证组合增益。

## Attention distance penalty (X1) — a free regularizer

- 税：Δ **−0.007**（在噪声内但正号）——唯一改善 clean bpc 的元素。
- 收益：与驱逐组合共 +0.004；**改善 anchor 通道（needle 76.8%→93.3%）**；per-head
  遥测平均注意距离 26–43，无头塌成纯 local。
- 警告：penalty 下 beyond-window needle "命中"是 local-LM 强度，**不是**抗驱逐召回。

## Eviction + anchors (X2) — the ring-buffer deployment story

- 税：Δ +0.121（原 X2；v2 needle 重设计 +0.058——改善是测量不是机制）。
- 收益：ring-buffer 解码免重 prefill；anchor 通道 needle 召回 76.8%（+penalty 93.3%）；
  per-layer cache + 早退 + prob0 retry 下 1.4×；retry 轮间 KV 位置严格递增
  （P0 类 bug，须持续测试）。
- **损坏 × 复制 = 目标冲突，三路尝试全失败（ticket 13 实验 2）**：anchor 召回
  0.764（X2v2 无损坏）→ X2C 无豁免 **0.016**、X2C-v2 锚区位置豁免 `corrupt_spare_anchors`
  **0.033**（in_window 0.47%→0.52%、beyond 0.59%→0.67%，clean bpc +0.0044）、X2Cv3
  needle 批次整体豁免 `needle_corrupt_free` **0.035**（且 clean bpc 1.680→**1.997**）。
  位置豁免救不回 4%；任务级豁免也没救回且 bpc 更糟。
- **裁决：损坏鲁棒与精确复制在单模型里目标冲突，修复必须靠显式模式/通道信号**，
  不能靠位置或批次的损坏豁免（非锚 needle 仍低是全局"别抄"策略，单点豁免不够）。
- 附带：`needle_acc` 评测协议 bug 实为真 bug（`eval_pc` 只清 `corrupt_wrong`，漏
  `corrupt_mask`），已加可失败单测钉住；**verdict-neutral**——阳性对照 X2v2 复现
  **0.8658/0.7637**。证据 `.scratch/13-four-experiments/evidence/{anchor_exempt,needle_antagonism}/`。

- **用户裁决（2026-09-16）：无需模式信号**。设计原则——模型从不"保留错误"，
  **目标恒为正确的 token**（损坏只是输入侧增广；训练 targets 一直是干净流，
  实现已满足此原则）。实测限制：13M 下损坏训练会压低精确复制行为（X2Cv3 连
  干净 needle 批次也未恢复 0.05，且 clean bpc 1.680→1.997）——属容量/训练规模
  限制，非目标冲突。needle 保持 backlog；若未来需要精确复制，优先更大规模/
  更长训练或专门检索头，而非模式信号。

## Diversity pressure (TTS, div_weight)

- 机制：并行路径间 capped −JS 压差（bf16 下 float32 + pre-step grad-norm gate）。
- 收益：**同时改善 baseline 与 TTS 斜率**（DIVL1 −0.062 > plain −0.047，上限 +0.023）。
- 税：Δ +0.023。确定性路径 checkpoint 从 K-path 集成恰得 0——多样性必须练进去。

## Cross-cutting laws

1. **元素税分割**：破坏 identity/order 结构的机制（wrong-token、skip、shuffle）贵
   （Δ 0.13–0.55）；保留或 flag identity 的（mask、noise、redo、penalty）~免费（Δ ≤ 0.07）。
2. **TTA 增益 ∝ 训练路径多样性**：确定性路径 ckpt 从 K-path 集成恰得 0；多样性必须
   练进去；DIVL1（capped −JS，float32 + grad-norm gate）同时改善 baseline 与斜率。
3. **组合有选择性**：共享机器时次可加（corruption × repair）；一方破坏另一方所需结构
   时拮抗（shuffle × everything）。
4. **辅助输出是 fallback 不是 voter**：深度前缀与 retry 轮按修复/退出训练，混进预测会
   退化。要集成价值需当 co-equal predictor 监督（未测）。
5. **reward 项的稳定性工程**：bf16 autocast 下任何 negative/reward loss 需 float32 +
   pre-step grad-norm gate（两次发散教会）。
6. **规模是未测变量**：以上"税主导"判决均在 13M；税底与集成增益是否随规模改善是
   主要 open 问题。
