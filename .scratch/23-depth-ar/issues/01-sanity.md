# 23 — 深度自适应 AR：小规模形态验证

- **Status:** done（待主会话审查；四性质 3+部分成立、形态税判据证伪+归因定位，见 evidence/SUMMARY.md）
- **Type:** 架构原型（新方向第一步）
- **Related:** mental_model §0（方向修订）、16/17/18/19（回环时代的对照基线）

## 目标

**证明代码没有 bug、理论正确**——不做规模主张。小模型（分钟级训练）上验证
深度自适应 AR 形态的四个核心性质，全部成立后才谈规模。

## 形态规格（本轮实现）

- 纯 AR 目标（**无 corruption 站 / 无 mask 站 / 无 retry**），enwik8。
- **全深度并行监督**：depth 0..L 全部算 CE（depth 0 = embedding 直接连
  tied-unembed 读出），tied unembed，fp32 头。
- **逐通道携带门**：h_{k+1} = h_k + g_k ⊙ Refine_k(h_k)，
  g_k = σ(Gate_k(h_k))，Gate 偏置初始化为 **carry（g≈0，恒等起步）**。
- 训练时**随机退出深度**：每个 position 采样一个退出深度 dᵢ，
  其 KV 用退出深度的表征填充到所有深层（混合深度 context，stage-2 精确化）。
  ——随机采样按均匀分布起步。
- 推理提交：softmax 软提交（e = Σp·E[t]）为默认 + 硬提交（argmax→embed）
  为推理通道；纯 latent 跳跃只作 ablation arm（预期漂移，如实记录）。

## 实验矩阵（全部小规模：enwik8，V=205）

| run | 配置 | 目的 |
|---|---|---|
| sanity-A | d=128, L=4, ~30M tokens | 代码正确性冒烟：深度曲线单调、门稳定、损失平衡 |
| sanity-B | d=256, L=8, ~30M tokens | 形态验证主点（workspace 有 51 维余量） |
| sanity-C | d=128, L=8 | **宽度-workspace 假设**：d<V 时 workspace≈0，深度增益应显著变差 |
| dense-B | d=256, L=8 纯 dense（无出口/门，同 token 预算） | 对照：形态的 tax/收益参照 |

每个 run 训练 ~10-20 min（4090），总 GPU ≤2h。

## 测量与判据

1. **depth-bpc 曲线**（每 run）：单调递减 = 精化成立；depth-L bpc vs dense-B
   的差 = 形态税（判据：|税| < 0.05，carry 门下预期 ≈0）。
2. **workspace 探针**：有效读出秩（depth 状态 logit-Jacobian 的数值秩 /
   正交扰动敏感度：扰动 h 的正交分量 ±ε，logits 变化 < δ 则该分量为 workspace）。
   判据：sanity-B 的 workspace 维数 > sanity-C（宽度假设）。
3. **出口提交**：每深度 exit→argmax→embed 提交后继续生成 128 token，
   序列合法、无发散；出口草稿 + 全栈验证的接受率（对照 MTP 2.40 tok/fwd
   量级即可，不强求）。
4. **门行为**：g_k 的均值随深度/训练的变化曲线（应从 0 逐步打开，
   不应全关或全开）；逐深度损失曲线无发散。
5. **代码正确性**：混合深度 context 的实现要有单测钉住（填充语义 =
   位置 i 的 KV 在其退出深度之后的所有层 = 其退出深度表征）。

## 判读与产出

- 四性质成立 → 形态地基确认，进入 Stage-2/规模规划（新 ticket）；
- 任一失败 → 如实证伪 + 假设定位（代码 bug / 假设错误分开陈述）。
- 产出：`.scratch/23-depth-ar/evidence/`（各 run results + depth 曲线 +
  workspace 探针 + SUMMARY.md 全数字表）；`pathlm/` 新模块（深度自适应
  训练/解码）；可失败单测；findings.md 新章（≤200 行）；WORKSPACE；
  **commit，不 push（主会话审查）**。

## 全局约束

- 仓库 /home/z/vibe/pathlm；策略 /home/z/vibe/AGENTS.md；语言中文/代码英文。
- GPU 单卡独占（空闲）；PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True；
  所有 GPU 步骤带超时；GPU 总预算 ≤2.5h。
- 测试基线：93 全绿，必须保持。
- 设计真值：docs/mental_model.md §0（先读）；docs/design.md 是回环时代
  冻结文档，仅供参考不作为本 ticket 规格。
- 实现注意：现有 depth_hook（dense-exit 监督）可扩展复用；混合深度填充
  需改 Decoder/训练 forward，additive 为主，回环时代的测试不许删。
