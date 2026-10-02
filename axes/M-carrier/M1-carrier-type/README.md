---
id: M1-carrier-type
axis: M
grid_dir: M-carrier/M1-carrier-type
title: "状态载体类型"
status: judged
judge: "保留（needle 消融）+ 内存/步长 + 吞吐"
cost: "13M 单臂 ~25–40 min"
tax: "ticket 35 的通道为**旁路**（`+264,296` 参数，U1/U2 逐位相同；单共享槽在输入流上，每层用自己的 query 读）——为对齐 MQAR/线性注意力标准形态而选择（per-layer-12 实装实测 221/410 ms/step 出局，根因是算子派发/autograd 节点开销而非 FLOPs）；解析状态 8192 floats（与 T 无关）vs 实测峰值 2015/2170/2282 MB（分开报）"
benefit: "**delta 写（S ← S·(I−βkkᵀ)+βvkᵀ，「只搬上次没搬的」）在关联回忆上 6.08× chance、且毁源后崩到 chance** ⇒ 「信息被搬到可检索处」首次被直接证实；读取量证据：U2 读分享 →0.7078、U0 为 0；距离腿六档全部 > chance（最远 d=255 为 0.5947，曲线非单调、不外推）；per-layer KV + 位置散点填充仍支持「变深度 + 不重 prefill」（与并行 forward 逐位一致）"
verdict: "**载体类型首验（ticket 35）：显式写入通道成立**——在「关联回忆 + 窗口 16（key 100% 落在窗外）」的合成任务上，13M/12k 步：无通道 U0 = 0.0137（≈chance，前提校验过）→ **delta 写 U2 = 0.0950（6.08× chance，+8.13pp，CI 不重叠；毁源后掉到 0.0081 ⇒ 读数确来自被搬运的信息）**；plain 写 U1 双稳态（0.0007 / 0.0334）且比 U2 低 9.42pp ⇒ **「只搬增量」的 delta 写是稳定有效的那一支**。绝对水平仅 9.5%（13M/12k）⇒ 按 SCHEMA §19 不得外推为「业界做不到/做得到」。 同时 ticket 34 造出「不靠语言先验」的长程天平并给出关键诊断：13M 主干长程能力**分任务型**——关联回忆 0.0061、变距离回忆 0.005–0.010（≈或低于 chance）⇒ 检索类彻底缺席；复制 0.1333（距离 223 仍 0.118，毁源→0.0125）、induction 0.0586（毁源→chance）⇒ 真实但弱。"
deps: ["M2-carrier-compression", "A9-old-info-retention", "F1-iteration-object"]
evidence: [".scratch/03-engine-m1/issues @e38349f", "axes/M-carrier/M1-carrier-type/experiments/2617-longrange-task-battery/analysis.md @ticket34", "axes/M-carrier/M1-carrier-type/experiments/2617-prior-art-memory/PRIOR_ART.md @ticket34", "axes/M-carrier/M1-carrier-type/experiments/2618-carry-channel-13m/analysis.md @ticket35"]
---

<!-- AUTO:begin -->

# M1-carrier-type · 状态载体类型

> 轴：**M 状态载体** ｜ 状态：**✅ judged**

> **判读**：**载体类型首验（ticket 35）：显式写入通道成立**——在「关联回忆 + 窗口 16（key 100% 落在窗外）」的合成任务上，13M/12k 步：无通道 U0 = 0.0137（≈chance，前提校验过）→ **delta 写 U2 = 0.0950（6.08× chance，+8.13pp，CI 不重叠；毁源后掉到 0.0081 ⇒ 读数确来自被搬运的信息）**；plain 写 U1 双稳态（0.0007 / 0.0334）且比 U2 低 9.42pp ⇒ **「只搬增量」的 delta 写是稳定有效的那一支**。绝对水平仅 9.5%（13M/12k）⇒ 按 SCHEMA §19 不得外推为「业界做不到/做得到」。 同时 ticket 34 造出「不靠语言先验」的长程天平并给出关键诊断：13M 主干长程能力**分任务型**——关联回忆 0.0061、变距离回忆 0.005–0.010（≈或低于 chance）⇒ 检索类彻底缺席；复制 0.1333（距离 223 仍 0.118，毁源→0.0125）、induction 0.0586（毁源→chance）⇒ 真实但弱。

## 1. 命题

跨步信息存在哪里。

## 2. 选项枚举（本格全部可能取值）

1. `KV cache（本项目唯一）`
2. `递归状态（SSM/RNN）`
3. `显式记忆槽`
4. `外部存储（检索）`
5. `权重（fast-weight）`
6. `无`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：**delta 写（S ← S·(I−βkkᵀ)+βvkᵀ，「只搬上次没搬的」）在关联回忆上 6.08× chance、且毁源后崩到 chance** ⇒ 「信息被搬到可检索处」首次被直接证实；读取量证据：U2 读分享 →0.7078、U0 为 0；距离腿六档全部 > chance（最远 d=255 为 0.5947，曲线非单调、不外推）；per-layer KV + 位置散点填充仍支持「变深度 + 不重 prefill」（与并行 forward 逐位一致）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：ticket 35 的通道为**旁路**（`+264,296` 参数，U1/U2 逐位相同；单共享槽在输入流上，每层用自己的 query 读）——为对齐 MQAR/线性注意力标准形态而选择（per-layer-12 实装实测 221/410 ms/step 出局，根因是算子派发/autograd 节点开销而非 FLOPs）；解析状态 8192 floats（与 T 无关）vs 实测峰值 2015/2170/2282 MB（分开报）
- **收益**：**delta 写（S ← S·(I−βkkᵀ)+βvkᵀ，「只搬上次没搬的」）在关联回忆上 6.08× chance、且毁源后崩到 chance** ⇒ 「信息被搬到可检索处」首次被直接证实；读取量证据：U2 读分享 →0.7078、U0 为 0；距离腿六档全部 > chance（最远 d=255 为 0.5947，曲线非单调、不外推）；per-layer KV + 位置散点填充仍支持「变深度 + 不重 prefill」（与并行 forward 逐位一致）
- **判读**：**载体类型首验（ticket 35）：显式写入通道成立**——在「关联回忆 + 窗口 16（key 100% 落在窗外）」的合成任务上，13M/12k 步：无通道 U0 = 0.0137（≈chance，前提校验过）→ **delta 写 U2 = 0.0950（6.08× chance，+8.13pp，CI 不重叠；毁源后掉到 0.0081 ⇒ 读数确来自被搬运的信息）**；plain 写 U1 双稳态（0.0007 / 0.0334）且比 U2 低 9.42pp ⇒ **「只搬增量」的 delta 写是稳定有效的那一支**。绝对水平仅 9.5%（13M/12k）⇒ 按 SCHEMA §19 不得外推为「业界做不到/做得到」。 同时 ticket 34 造出「不靠语言先验」的长程天平并给出关键诊断：13M 主干长程能力**分任务型**——关联回忆 0.0061、变距离回忆 0.005–0.010（≈或低于 chance）⇒ 检索类彻底缺席；复制 0.1333（距离 223 仍 0.118，毁源→0.0125）、induction 0.0586（毁源→chance）⇒ 真实但弱。

## 5. 失败路径与原因

1. 早期缓存 bug：逐位置早退导致 per-layer cache 长度失步（已修为 per-layer 位置列表；持续单测）
2. **长程检索能力缺席（ticket 34-A 仪器实测）**：关联回忆 0.0061、变距离回忆六档 0.005–0.010，均≈或低于 chance 且无距离依赖 ⇒ 此前所有「没搬」读数，至少相当部分来自**模型本来就没有该能力**，而非「机制搬不动」（SCHEMA §19）
3. **仪器落地（`tools/evalkit/longrange_tasks.py`）**：四任务 + chance/局部先验基线 + 距离曲线 + `recall_under_visibility`（复用 retention 掩码）+ 毁源对照臂；自证：局部先验基线 0.0098–0.0217（chance 上限 0.0469）、毁源/打乱答案均掉回 chance、前向逐位一致
4. 不确定项：induction 每点 n=128 不支持斜率；assoc/var 低分含分布外成分（**不得**外推成「训练也学不会」）；copy 13% 的机制未定位
5. **容量×难度曲线（ticket 36-U1）**：难度曲线成立——K=4/8/16/32 准确率 0.535/0.366/0.107/0.056（分别 34.2×/23.4×/6.9×/3.6× chance），相邻档 95% CI 两两不重叠，**8→16 有明显拐点**（跌 25.8pp，而 16→32 仅跌 5.1）；**容量曲线不成立**——状态维度 4096→32768（×8）在 K=8/16/32 上准确率几乎不动（0.0573/0.0563/0.0563/0.0578，CI 全重叠），K=4 上最大状态反而更差（0.422 vs 0.531，−11pp，原因未定论）；四臂到 1 万步**均未收敛**（loss 斜率 −0.065~−0.051）⇒ 最可能是**瓶颈在训练而非容量**；按 SCHEMA §19 只能读作「该尺度加大状态不划算」，**不得**外推为「容量机制不可能」。活性已证：状态浮点随开关 4096→32768 精确变化、读出占比 0.67–0.75、**毁源后 16 格全部掉回 chance（最大 0.0173 vs 0.0156）**
6. **训练信号（ticket 36-U2，见 F4）**：重建/充分性损失被完美学会（重建命中 ≈1.00）但**长程回忆一点没长**（有 λ=1.0 0.0137 vs 无 λ 0.0161，−0.244pp、CI 重叠；重跑 4096 次提问 **0 命中**）⇒ 该尺度上「给路径（显式写通道 0.0950）」远优于「给信号（重建损失）」
7. **ticket 35 的边界**：绝对水平仅 9.5%（13M/12k 步）⇒ 只能读作「在该任务/该尺度上成立」；**plain 写（U1）双稳态**（同配置同 seed 两次 0.0007 vs 0.0334，答案行 CE 4.184 vs 2.252，重跑散度 +3.271pp）⇒ 判据③只在「两支 plain 读数」上成立，**强表述（delta 优于 plain）不作为结论**；距离曲线受训练位置结构混杂；未做 ④ 训练信号（重建/充分性损失）与 ③ 容量压力（更小状态）
8. **因果性护栏（本设计最致命的风险）已钉死**：分块 WY 形与顺序参考在 fp64 下 <1e-12 逐位一致（chunk ∈ {1,2,3,5,8,24}）、扰动 t≥10 时 t<10 输出 `torch.equal`、读后写约定有专测、变异 M1（释放块内 mask 造成未来泄漏）被抓住

## 6. 判据与判决实验设计

- **判据**：保留（needle 消融）+ 内存/步长 + 吞吐

## 7. 成本估计

- 13M 单臂 ~25–40 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `M2-carrier-compression`
2. `A9-old-info-retention`
3. `F1-iteration-object`

## 9. 复现手段

1. .scratch/03-engine-m1/issues @e38349f
2. axes/M-carrier/M1-carrier-type/experiments/2617-longrange-task-battery/analysis.md @ticket34
3. axes/M-carrier/M1-carrier-type/experiments/2617-prior-art-memory/PRIOR_ART.md @ticket34
4. axes/M-carrier/M1-carrier-type/experiments/2618-carry-channel-13m/analysis.md @ticket35

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
