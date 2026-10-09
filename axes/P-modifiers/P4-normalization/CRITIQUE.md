# CRITIQUE — P4-normalization（独立批判审计，fork=false）

## ① 是什么
LayerNorm 放哪、有无 final norm、是否 QK-norm——决定残差流是否保持 embedding 几何（"同空间"前提）。

## ② 对着什么 sota，我们偏离在哪
业界标准：Pre-LN 更易训练（Xiong 2020）、RMSNorm 等价省算、QK-norm 稳定注意力 logits（`记忆·未核`）。**偏离**：本格把"无 final Norm"当成本项目的**设计前提**（不是测量），且 `status: partial`（已废除的状态）。

## ③ 五类问题逐条判定
- **开倒车：无。**
- **过程错误：有（缺实验 + 漏记本地结果）。**
  - `catalog.yaml:1358`（`verdict`）"pre-LN + 无 final norm 是『同空间』几何的前提，已固定使用"——**没有任何消融**（无 post-LN 臂、无"有 final norm"臂），`failure: []`，判据却写"训练稳定性 + 残差流范数分布"。
  - **漏记本仓自己的 QK-Norm 结果**：`catalog.yaml:334–360`（A6）记录 ticket 42-W1/43-V1 的修复——"**QK-Norm（逐头 L2 归一 q/k，零新增参数）**：把 logits 钉在 ±√32≈5.66、两 seed `gnorm_max` **1740.9→7.25 / 29578.6→7.58**、末态 clean bpc 差 0.096→**0.0005**"，并推荐长度条件档 `qk-normfrom512`（T≤512 零 hook、T=512 实测 Δ=−0.0016、全局档在 T=512 +0.0314 越界）。**QK-norm 正是 P4 的选项之一**（`catalog.yaml:1362` `options` 含 `QK-norm`），但这批结果**没有回填 P4**。
- **结论错误：无强结论**，但"已固定使用"把前提当结论；缺项（QK-norm）是实质缺口。
- **无意义：否。** 归一化位置影响稳定性与"同空间"契约（本仓多处依赖 `logits = h·Eᵀ`）。
- **值得做：是（回填 + 一个最小消融）。**

## ④ 建议动作：补做 / 回填
1. 把 A6 的 QK-Norm 结论写进 P4（含推荐档 `qk-normfrom512` 与短 T 越界事实），并在 P4 标"本地实测"。
2. 明确区分：**"无 final norm"是设计前提（前提 ⇒ 不是结论）**；"QK-Norm 抑制 logits 爆炸"是**本地实测结论**。
3. 可选最小消融：有/无 final norm（同参数量，13M/12k）以判断该前提是否有代价。

## ⑤ 正确的验证方式
- **该涨**：开 QK-Norm 后 `gnorm_max` 应从 O(1e3–1e4) 降到 <100、跳步应为 0（`catalog.yaml:334–360` 的既有读数）。
- **该跌**：全局 QK-Norm 在短 T 的 clean bpc（T=512 +0.0314 越界 ⇒ 必须用长度条件档）。
- **必须不变**：T≤512 时零 hook（`qk-normfrom512` 下 T=512 Δ=−0.0016 在带内）；`logits = h·Eᵀ` 的同空间语义。
