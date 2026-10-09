<!-- AUTO:begin -->

# H · 视界（一次前向产生几个位置）

- **原语**：II（沿序列推进） × 头
- **命题**：一次前向预测几个未来位置、这些位置在训练时被什么内容填充、输出如何组合与验证。

## 定义

MTP 族。本轴的核心不是「质量」（满深度与半栈答案的差距仅 0.0003），
而是成本：草稿从哪来、接受率多高、验证多贵。

## 子维度

H1 每步输出位置数 · H2 训练内容填充 · H3 组合方式 ·
H4 头与主干关系 · H5 验证机制

## 本地总体结论

税效比全项目最好的一条：n_mtp 1→2 税 +0.041、2→3 +0.0166（递减），
换 Medusa 批量验证 1.68–1.90× @13M；@100M 墙钟塌到 1.15–1.29×
（前向 compute-bound），只剩 FLOPs 口径。否证：朴素概率复合是
数学错误（CE 3.96 vs 1.58）；条件链 deploy 0.547 < direct 0.560
（草稿质量瓶颈，~31% 草稿错误）；consistency loss 中性。
冻结主干只训头（H4）实测 @100M 零税 → 最便宜的真空白。

> **既有做法情报**：[PRIOR_ART.md](PRIOR_ART.md)（含机器可读分诊表；来源标 `记忆·未核`，联网不可用）

## 格子索引

| 格子 | 状态 | 判读 | 税 |
|---|---|---|---|
| [`H1-outputs-per-pass`](H1-outputs-per-pass/README.md) | ✅ judged | 全项目税效比最好的机制；但墙钟收益是 13M/launch-bound 现象。 | n_mtp 1→2 +0.041（B2 1.5481）；2→3 +0.016… |
| [`H2-training-fill`](H2-training-fill/README.md) | ✅ judged | 保留式填充（self-fill）胜出；PAD 与错误 token 都更贵；软提交有 OOD 污染。 | mask/wrong-token 替换 +0.059 / +0.127；em… |
| [`H3-composition`](H3-composition/README.md) | ✅ judged | **本格命题限于「冻结表征上的重复潜在映射与门控」**：direct `T(h)` / composed `T(T(h))` / gated 混合，全部只被 CE(t+2) 监… | （头机制，挂冻结主干：零税；头的税见 H1） |
| [`H4-head-trunk-relation`](H4-head-trunk-relation/README.md) | ✅ judged | 挂头零税坐实（Δbpc = 0.0000）。按修正后的诚实口径（vs 1-step AR 两次前向，只替换紧邻位置）**判中性**：对手 54.89%，跳跃头 46.67% =… | probe 式头 @100M ≈0（1.5928 vs 1.5871）；停梯… |
| [`H5-verification`](H5-verification/README.md) | ✅ judged | 批量验证可用且逐位一致；定稿不变量（全深度重打分）是本形态的正确性保障。 | 验证本身不额外加税（复用同一次前向） |

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
