<!-- AUTO:begin -->

# R · 参照线（必须打败的基线）

- **原语**：横切
- **命题**：每条创新线面对的现成行业答案——不打败它，创新就没有意义。

## 定义

参照线不是「我们做的「轴」，而是「我们必须超越的对手」。它们决定了
每条创新线的判据线：例如右移压缩必须打败「就用个窗口」（其 bpc 代价
仅 +0.058~0.121）。

## 子维度

R1 量化 · R4 窗口+KV量化+检索 · R7 既有早退实现

## 本地总体结论

本地实测：R1 的 KV 量化价签（int8/int4）、R4 的窗口部分；其余为行业口径
（`记忆·未核`）。这些线决定所有创新线的判据线。

> **既有做法情报**：[PRIOR_ART.md](PRIOR_ART.md)（含机器可读分诊表；来源标 `记忆·未核`，联网不可用）

## 格子索引

| 格子 | 状态 | 判读 | 税 |
|---|---|---|---|
| [`R1-quantization`](R1-quantization/README.md) | ⬜ todo | **已有**：本地已实测 KV 量化的部署口径价签与质量——**块标尺（沿 head_dim，G=32）** 下 int8 每位置 6528 B = fp16 的 ×0.53（… | **部署口径（源 `2632-kv-block-scale`）**：**块标… |
| [`R4-window-kv-quant-retrieval`](R4-window-kv-quant-retrieval/README.md) | ⬜ todo | **已有**：本地仅覆盖窗口/锚点部分（税 +0.058~0.121 bpc；锚点通道 93.3%）。**尚缺**：按保留预算的窗口曲线 + 检索腿是否显著 >chance；量… | 窗口部分：+0.058~0.121 bpc（13M）；检索/量化部分零经验 |
| [`R7-existing-early-exit`](R7-existing-early-exit/README.md) | ⬜ todo | **已有**：本地仅有对照——dense 全深度监督（LayerSkip 式）+0.067（13M）/ 剂量恶化 @100M（+1.73 corrupt）。**尚缺**：同 F… | 本地对照：dense 全深度监督（LayerSkip 式）+0.067（13… |

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
