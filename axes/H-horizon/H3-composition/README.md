---
id: H3-composition
axis: H
grid_dir: H-horizon/H3-composition
title: "多位置预测的组合方式"
status: todo
judge: "同 FLOPs 的质量（等算力对照）+ 草稿接受率 + 组合置信度的校准（ECE）"
cost: "13M 单臂 ~25–30 min"
tax: "（头机制，挂冻结主干：零税；头的税见 H1）"
benefit: "省算力：等效 1.95×（H4 口径）；门是可观测量（g 分布 + 逐桶命中单调 0.17→0.68）。**部署口径（`2631-spec-three-arms`）**：direct 路线 tok/fwd **2.58**、墙钟加速比 **1.58×**（只作相对排序，绝对水平受外部占用抬高）、每 token 算力 **R=1.163（净亏）**（53.9M 点；旧 `c₃/(tok/fwd)=0.48` 口径已撤回）"
verdict: "**已有**：**本格命题限于「冻结表征上的重复潜在映射与门控」**：direct `T(h)` / composed `T(T(h))` / gated 混合，全部只被 CE(t+2) 监督。它**不回答**用户要的概率路径组合问题（`p(x_{N+2}|h)=Σ_z p(z|h)p(x_{N+2}|h,z)` vs `p_direct`）——`T∘T` 既非「N+1 再加一」、也非「直接 N+2」、更非两者的概率混合（无概率、第一遍不受 N+1 监督）。下述结论**只对该受限命题成立**，不得用作「先 N+1 再 N+2 被否证」。**在该受限命题下被判负**：门控混合 0.4738 仅比直跳 0.4672 高 +0.0020（<0.014 噪声带），且等参数无门对照 0.4785 更高 ⇒ 增益归容量不归门控；门不是「选择」而是**坍缩到连跳**（96.7% 位置 g≈0.82，被压掉的直跳分支只训到 0.262 vs 单训 0.467）。**本格命题的正式回答由 Spec 三臂给出**：`2631-spec-three-arms`（唯一变量 = 第 N+2 草稿位的分布来源：`direct` 从同一状态一次直接出 / `combined` 两条分布按概率混合 λ=0.5 / `chain` 先出 N+1 再以之为条件出 N+2）在 13.4M×2 seed 与 53.9M×1 seed 上一致给出 **direct ≈ combined（配对差 95% CI 含 0，不可排序）> chain（显著更差）**；53.9M 点 α₁/α₂|₁ = **0.844/0.872（direct）· 0.829/0.853（combined）· 0.657/0.826（chain）**、τ = **1.58/1.54/1.20**、tok/fwd = **2.58/2.54/2.20**、墙钟加速比 = **1.58/1.57/1.39**、每 token 算力 **R=(1+γ)/(1+τ) = 1.163/1.183/1.364**（53.9M 点；**均 >1 即净亏**，臂序 direct<combined<chain；旧「等效算力净账 c₃/(tok/fwd) = 0.48/0.49/0.56」把**墙钟比当算力比**，是**符号级错误，已撤回**）；三臂发出流与原生贪心**逐位相同**（门禁 9/9）。**2613 是背景实验**：它测的是冻结表征上的重复潜在映射 T∘T 与门控，回答的不是本命题。**α₁ 差的归因由 `2632-alpha1-attribution` 给出**（同一 checkpoint、只换 N+2 头/条件、不重训）：三臂**共用一份 checkpoint**（发出流逐位相同）⇒ **排除了「臂级独立训练」**；**训练期 chain 损失是否损害 node1 未测**（需无-chain 损失消融）。该排序是**前缀 regime 依赖**的：在**完全 in-distribution 的真实文本前缀**（enwik8 val，ctx≤160）上 `chain` 的 N+2 草稿与模型自身 AR 输出的**一致率反而更高**（0.842/0.871/0.852，direct 0.677/0.706/0.693，CI 不重叠），**但该 regime 不是部署 regime**；**部署（自生成）regime 下 direct 仍胜**。真实文本上的高接受率含**共享 T1 耦合**成分，且**两条机制（耦合 vs 条件分支 OOD）未分离**；**「循环」是未验证假设**（仅 4 个 prompt；§19 强局部先验未做低局部先验对照）。`chain` 的「更准」只在 **oracle 条件（部署不可得）** 下成立（0.59–0.61 > direct 0.45–0.47）；**部署条件（喂自己 v1）下 chain 质量 0.434–0.447 低于 direct 0.452–0.470 ⇒ 头优势不可部署兑现**；条件错时 chain 的真 t+2 质量从 0.67 崩到 0.09（direct 0.64→0.21）。**Q2**：**同 K 下** regime 决定排序；但**预登记 T1 的 cross-K 判据在 13m_s0（chain@4 0.831 ≥ direct@1 0.792）与 teacher-anchored（chain@1 0.842 ≥ direct@1 0.677）均触发「树可能翻转」，须做完整树实验才能定论**。**尚缺（待立项）**：完整树验证、温度>0 拒绝采样、无 chain 损失消融（判 ②）、把 `chain2` 换成独立条件块、§19 低局部先验对照、100M。"
deps: ["H1-outputs-per-pass", "F4-training-signal", "G1-selector"]
evidence: [".scratch/12-overnight/evidence @216a00a", ".scratch/04-m1-runs/evidence @e38349f", "axes/H-horizon/H3-composition/experiments/2613-self-gated-composition-13m/analysis.md @15eeac0", "axes/H-horizon/H3-composition/experiments/2631-spec-three-arms/analysis.md @d256c0a", "axes/H-horizon/H3-composition/experiments/2632-alpha1-attribution/analysis.md @ead6a74", "axes/H-horizon/H3-composition/experiments/2632-alpha1-attribution/results.json @ead6a74", "axes/H-horizon/H3-composition/experiments/2632-alpha1-attribution/diag_records.jsonl @ead6a74", "axes/H-horizon/H3-composition/experiments/2632-alpha1-attribution/probes_output.json @ead6a74"]
---

<!-- AUTO:begin -->

# H3-composition · 多位置预测的组合方式

> 轴：**H 视界（一次前向产生几个位置）** ｜ 状态：**⬜ todo**

> **判读**：**已有**：**本格命题限于「冻结表征上的重复潜在映射与门控」**：direct `T(h)` / composed `T(T(h))` / gated 混合，全部只被 CE(t+2) 监督。它**不回答**用户要的概率路径组合问题（`p(x_{N+2}|h)=Σ_z p(z|h)p(x_{N+2}|h,z)` vs `p_direct`）——`T∘T` 既非「N+1 再加一」、也非「直接 N+2」、更非两者的概率混合（无概率、第一遍不受 N+1 监督）。下述结论**只对该受限命题成立**，不得用作「先 N+1 再 N+2 被否证」。**在该受限命题下被判负**：门控混合 0.4738 仅比直跳 0.4672 高 +0.0020（<0.014 噪声带），且等参数无门对照 0.4785 更高 ⇒ 增益归容量不归门控；门不是「选择」而是**坍缩到连跳**（96.7% 位置 g≈0.82，被压掉的直跳分支只训到 0.262 vs 单训 0.467）。**本格命题的正式回答由 Spec 三臂给出**：`2631-spec-three-arms`（唯一变量 = 第 N+2 草稿位的分布来源：`direct` 从同一状态一次直接出 / `combined` 两条分布按概率混合 λ=0.5 / `chain` 先出 N+1 再以之为条件出 N+2）在 13.4M×2 seed 与 53.9M×1 seed 上一致给出 **direct ≈ combined（配对差 95% CI 含 0，不可排序）> chain（显著更差）**；53.9M 点 α₁/α₂|₁ = **0.844/0.872（direct）· 0.829/0.853（combined）· 0.657/0.826（chain）**、τ = **1.58/1.54/1.20**、tok/fwd = **2.58/2.54/2.20**、墙钟加速比 = **1.58/1.57/1.39**、每 token 算力 **R=(1+γ)/(1+τ) = 1.163/1.183/1.364**（53.9M 点；**均 >1 即净亏**，臂序 direct<combined<chain；旧「等效算力净账 c₃/(tok/fwd) = 0.48/0.49/0.56」把**墙钟比当算力比**，是**符号级错误，已撤回**）；三臂发出流与原生贪心**逐位相同**（门禁 9/9）。**2613 是背景实验**：它测的是冻结表征上的重复潜在映射 T∘T 与门控，回答的不是本命题。**α₁ 差的归因由 `2632-alpha1-attribution` 给出**（同一 checkpoint、只换 N+2 头/条件、不重训）：三臂**共用一份 checkpoint**（发出流逐位相同）⇒ **排除了「臂级独立训练」**；**训练期 chain 损失是否损害 node1 未测**（需无-chain 损失消融）。该排序是**前缀 regime 依赖**的：在**完全 in-distribution 的真实文本前缀**（enwik8 val，ctx≤160）上 `chain` 的 N+2 草稿与模型自身 AR 输出的**一致率反而更高**（0.842/0.871/0.852，direct 0.677/0.706/0.693，CI 不重叠），**但该 regime 不是部署 regime**；**部署（自生成）regime 下 direct 仍胜**。真实文本上的高接受率含**共享 T1 耦合**成分，且**两条机制（耦合 vs 条件分支 OOD）未分离**；**「循环」是未验证假设**（仅 4 个 prompt；§19 强局部先验未做低局部先验对照）。`chain` 的「更准」只在 **oracle 条件（部署不可得）** 下成立（0.59–0.61 > direct 0.45–0.47）；**部署条件（喂自己 v1）下 chain 质量 0.434–0.447 低于 direct 0.452–0.470 ⇒ 头优势不可部署兑现**；条件错时 chain 的真 t+2 质量从 0.67 崩到 0.09（direct 0.64→0.21）。**Q2**：**同 K 下** regime 决定排序；但**预登记 T1 的 cross-K 判据在 13m_s0（chain@4 0.831 ≥ direct@1 0.792）与 teacher-anchored（chain@1 0.842 ≥ direct@1 0.677）均触发「树可能翻转」，须做完整树实验才能定论**。**尚缺（待立项）**：完整树验证、温度>0 拒绝采样、无 chain 损失消融（判 ②）、把 `chain2` 换成独立条件块、§19 低局部先验对照、100M。

## 1. 命题

直接头与多跳组合如何合并——这是用户提出的「自门控非线性组合」所在格。

## 2. 选项枚举（本格全部可能取值）

1. `只用直接头`
2. `链式条件（concat+proj，DeepSeek 式）`
3. `朴素概率乘积`
4. `DP lattice 置信加权（设计 §5 已定义，从未端到端测量）`
5. `自门控选组合深度（一次跳 k vs 跳 1 走 k 次，由 sigmoid 门决定）`
6. `beam 多线 + 重排`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：省算力：等效 1.95×（H4 口径）；门是可观测量（g 分布 + 逐桶命中单调 0.17→0.68）。**部署口径（`2631-spec-three-arms`）**：direct 路线 tok/fwd **2.58**、墙钟加速比 **1.58×**（只作相对排序，绝对水平受外部占用抬高）、每 token 算力 **R=1.163（净亏）**（53.9M 点；旧 `c₃/(tok/fwd)=0.48` 口径已撤回）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：（头机制，挂冻结主干：零税；头的税见 H1）
- **收益**：省算力：等效 1.95×（H4 口径）；门是可观测量（g 分布 + 逐桶命中单调 0.17→0.68）。**部署口径（`2631-spec-three-arms`）**：direct 路线 tok/fwd **2.58**、墙钟加速比 **1.58×**（只作相对排序，绝对水平受外部占用抬高）、每 token 算力 **R=1.163（净亏）**（53.9M 点；旧 `c₃/(tok/fwd)=0.48` 口径已撤回）
- **判读**：**已有**：**本格命题限于「冻结表征上的重复潜在映射与门控」**：direct `T(h)` / composed `T(T(h))` / gated 混合，全部只被 CE(t+2) 监督。它**不回答**用户要的概率路径组合问题（`p(x_{N+2}|h)=Σ_z p(z|h)p(x_{N+2}|h,z)` vs `p_direct`）——`T∘T` 既非「N+1 再加一」、也非「直接 N+2」、更非两者的概率混合（无概率、第一遍不受 N+1 监督）。下述结论**只对该受限命题成立**，不得用作「先 N+1 再 N+2 被否证」。**在该受限命题下被判负**：门控混合 0.4738 仅比直跳 0.4672 高 +0.0020（<0.014 噪声带），且等参数无门对照 0.4785 更高 ⇒ 增益归容量不归门控；门不是「选择」而是**坍缩到连跳**（96.7% 位置 g≈0.82，被压掉的直跳分支只训到 0.262 vs 单训 0.467）。**本格命题的正式回答由 Spec 三臂给出**：`2631-spec-three-arms`（唯一变量 = 第 N+2 草稿位的分布来源：`direct` 从同一状态一次直接出 / `combined` 两条分布按概率混合 λ=0.5 / `chain` 先出 N+1 再以之为条件出 N+2）在 13.4M×2 seed 与 53.9M×1 seed 上一致给出 **direct ≈ combined（配对差 95% CI 含 0，不可排序）> chain（显著更差）**；53.9M 点 α₁/α₂|₁ = **0.844/0.872（direct）· 0.829/0.853（combined）· 0.657/0.826（chain）**、τ = **1.58/1.54/1.20**、tok/fwd = **2.58/2.54/2.20**、墙钟加速比 = **1.58/1.57/1.39**、每 token 算力 **R=(1+γ)/(1+τ) = 1.163/1.183/1.364**（53.9M 点；**均 >1 即净亏**，臂序 direct<combined<chain；旧「等效算力净账 c₃/(tok/fwd) = 0.48/0.49/0.56」把**墙钟比当算力比**，是**符号级错误，已撤回**）；三臂发出流与原生贪心**逐位相同**（门禁 9/9）。**2613 是背景实验**：它测的是冻结表征上的重复潜在映射 T∘T 与门控，回答的不是本命题。**α₁ 差的归因由 `2632-alpha1-attribution` 给出**（同一 checkpoint、只换 N+2 头/条件、不重训）：三臂**共用一份 checkpoint**（发出流逐位相同）⇒ **排除了「臂级独立训练」**；**训练期 chain 损失是否损害 node1 未测**（需无-chain 损失消融）。该排序是**前缀 regime 依赖**的：在**完全 in-distribution 的真实文本前缀**（enwik8 val，ctx≤160）上 `chain` 的 N+2 草稿与模型自身 AR 输出的**一致率反而更高**（0.842/0.871/0.852，direct 0.677/0.706/0.693，CI 不重叠），**但该 regime 不是部署 regime**；**部署（自生成）regime 下 direct 仍胜**。真实文本上的高接受率含**共享 T1 耦合**成分，且**两条机制（耦合 vs 条件分支 OOD）未分离**；**「循环」是未验证假设**（仅 4 个 prompt；§19 强局部先验未做低局部先验对照）。`chain` 的「更准」只在 **oracle 条件（部署不可得）** 下成立（0.59–0.61 > direct 0.45–0.47）；**部署条件（喂自己 v1）下 chain 质量 0.434–0.447 低于 direct 0.452–0.470 ⇒ 头优势不可部署兑现**；条件错时 chain 的真 t+2 质量从 0.67 崩到 0.09（direct 0.64→0.21）。**Q2**：**同 K 下** regime 决定排序；但**预登记 T1 的 cross-K 判据在 13m_s0（chain@4 0.831 ≥ direct@1 0.792）与 teacher-anchored（chain@1 0.842 ≥ direct@1 0.677）均触发「树可能翻转」，须做完整树实验才能定论**。**尚缺（待立项）**：完整树验证、温度>0 拒绝采样、无 chain 损失消融（判 ②）、把 `chain2` 换成独立条件块、§19 低局部先验对照、100M。

## 5. 失败路径与原因

1. **命题边界**：本格全部臂只吃 CE(t+2)、`T∘T` 无概率与条件监督 ⇒ **不能**回答用户「N+1 再加一 vs 直接 N+2 vs 概率混合」的问题；该问题由 Spec 三臂（`2631-spec-three-arms`）回答——**direct ≈ combined（不可排序）> chain（显著更差）**；本实验（2613）是背景实验
2. **主判据未过**：gated 0.4738 vs 阈值 max(direct 0.4672, composed 0.4718)+0.014 = 0.4858 ⇒ Δ = +0.0020
3. 草稿接受率 0.6113（< 0.90）⇒ 不可作草稿
4. **等参数无门对照 direct_eq（394,496 参数，与 gated 逐位相同）反而最高 0.4785** ⇒ 那 +0.0020 归容量、不归门控结构（SCHEMA §15）
5. 门**坍缩到单支**：起点 g≡0.5 → 均值 0.82（96.7% 位置偏两跳），被抑制的直跳分支退化到 0.262（单独训练为 0.467）⇒ 不是「选择组合深度」而是「偏好一支」
6. 2× 训练（4000 步）后 Δ = +0.0042 仍 < 0.014；快照口径（四臂同窗斜率 −0.060~−0.069/1k 仍在下降）
7. 契约缺陷（主 Agent 认领）：sigmoid 门 + 单边零 + 「起步等价 direct 或 composed」三者不可兼得（σ(0)=0.5）
8. **翻转适用范围受限**：teacher-anchored 前缀（完全 in-distribution 的真实文本）≠ 部署前缀；该口径下的高接受率含**共享 T1 耦合**成分，不得读成部署收益（`2632-alpha1-attribution/analysis.md` §5.2/§10）
9. **§19 强局部先验未做对照**：enwik8 上的「regime 依赖」结论不得外推为长程结论（需在随机键值回忆 / induction / 复制上复核）
10. **预登记 T1 的 cross-K 分支触发**（树可能翻转）**未实测**：只给了 top-k 召回探针，未做完整树验证

## 6. 判据与判决实验设计

- **判据**：同 FLOPs 的质量（等算力对照）+ 草稿接受率 + 组合置信度的校准（ECE）

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：同 FLOPs 的质量（等算力对照）+ 草稿接受率 + 组合置信度的校准（ECE）
- 成本：13M 单臂 ~25–30 min

## 7. 成本估计

- 13M 单臂 ~25–30 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `H1-outputs-per-pass`
2. `F4-training-signal`
3. `G1-selector`

## 9. 复现手段

1. .scratch/12-overnight/evidence @216a00a
2. .scratch/04-m1-runs/evidence @e38349f
3. axes/H-horizon/H3-composition/experiments/2613-self-gated-composition-13m/analysis.md @15eeac0
4. axes/H-horizon/H3-composition/experiments/2631-spec-three-arms/analysis.md @d256c0a
5. axes/H-horizon/H3-composition/experiments/2632-alpha1-attribution/analysis.md @ead6a74
6. axes/H-horizon/H3-composition/experiments/2632-alpha1-attribution/results.json @ead6a74
7. axes/H-horizon/H3-composition/experiments/2632-alpha1-attribution/diag_records.jsonl @ead6a74
8. axes/H-horizon/H3-composition/experiments/2632-alpha1-attribution/probes_output.json @ead6a74

## 备注

空白格（用户新设计）。等 FLOPs 的判决实验：1 次前向 + 跳跃头预测 t+2
vs 2 次前向推到 t+2（已有数字：跳跃头 55.97% / 条件接受 65.8%，
连跑两次 ≈ node-1 68.8% → 跳跃头以一半算力买到约 81% 准确率）。
质量头寸已被花掉（d6−d12 = 0.0003）→ 组合的价值应在成本分配而非质量。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
