<!-- AUTO:begin -->

# R · 参照线（必须打败的基线）

- **原语**：横切
- **命题**：每条创新线面对的现成行业答案——不打败它，创新就没有意义。

## 定义

参照线不是「我们做的「轴」，而是「我们必须超越的对手」。它们决定了
每条创新线的判据线：例如右移压缩必须打败「就用个窗口」（其 bpc 代价
仅 +0.058~0.121）。

## 子维度

R1 量化 · R2 蒸馏 · R3 剪枝稀疏 · R4 窗口+KV量化+检索 · R5 best-of-n+verifier · R6 MoE · R7 既有早退实现

## 本地总体结论

本地对 R4 的窗口部分与 R5 的集成部分有侧证（X2/X3、TTS），
其余为行业口径。这些线本地零经验，但决定所有创新线的判据线。

> **既有做法情报**：[PRIOR_ART.md](PRIOR_ART.md)（含机器可读分诊表；来源标 `记忆·未核`，联网不可用）

## 格子索引

| 格子 | 状态 | 判读 | 税 |
|---|---|---|---|
| [`R1-quantization`](R1-quantization/README.md) | 🏭 reference（含业界结论） | — | **int8 KV：Δbpc +0.00015（近无损）**，每位置 122… |
| [`R2-distillation`](R2-distillation/README.md) | 🏭 reference（含业界结论） | — | — |
| [`R3-pruning-sparsity`](R3-pruning-sparsity/README.md) | 🏭 reference（含业界结论） | — | 本地侧证：结构化约束（只读前 360 维）→ 恶化 −50.4%（W1） |
| [`R4-window-kv-quant-retrieval`](R4-window-kv-quant-retrieval/README.md) | 🏭 reference（含业界结论） | — | 窗口部分：+0.058~0.121 bpc（13M）；检索/量化部分零经验 |
| [`R5-bestofn-verifier`](R5-bestofn-verifier/README.md) | 🏭 reference（含业界结论） | — | — |
| [`R6-moe-scale`](R6-moe-scale/README.md) | 🏭 reference（含业界结论） | — | — |
| [`R7-existing-early-exit`](R7-existing-early-exit/README.md) | 🏭 reference（含业界结论） | — | 本地对照：dense 全深度监督（LayerSkip 式）+0.067（13… |

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
