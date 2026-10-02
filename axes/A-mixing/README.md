<!-- AUTO:begin -->

# A · 序列混合机制

- **原语**：II（沿序列推进）
- **命题**：一个位置从哪里、以什么算子、按什么寻址方式取信息。

## 定义

这条轴决定「一个 token 能看见谁、用什么算子合并、怎么定位」。它同时是
长上下文成本所在（可见范围）与质量所在（算子表达力）。

## 子维度

A1 因果性 · A2 全注意力基线 · A3 窗口/锚点/sink · A4 软距离惩罚 ·
A5 膨胀跨步窗 · A6 混合算子 · A7 寻址方式 · A8 位置语义 ·
A9 旧信息处理 · A10 压缩触发与优先级

## 本地总体结论

本地覆盖了「窗口 + 锚点 + 软距离惩罚」这一族（X1/X2/X3）。核心结论：
丢前缀在 bpc 上很便宜（+0.058~0.121 @13M），但检索当场归零（beyond 0.4%）；
软距离惩罚是唯一改善 clean bpc 的元素（−0.007），但它不创造保留——
窗口外 90.3% 的「命中」是局部补全，是最典型的假保留陷阱。
压缩/摘要/递归/检索式载体在本项目零经验（M 轴亦然）。

> **既有做法情报**：[PRIOR_ART.md](PRIOR_ART.md)（含机器可读分诊表；来源标 `记忆·未核`，联网不可用）

## 格子索引

| 格子 | 状态 | 判读 | 税 |
|---|---|---|---|
| [`A1-causality`](A1-causality/README.md) | 🏭 reference（含业界结论） | — | — |
| [`A2-full-attention`](A2-full-attention/README.md) | ✅ judged | 基底成立，外部锚在公开数字范围内。 | —（锚点本身） |
| [`A3-window-anchor`](A3-window-anchor/README.md) | ✅ judged | 丢前缀的代价**由距离与预算共同决定、并不便宜**（见 ERRATA：只保留最近 32 个 = +0.212 bpc）；锚点通道是真产物（76.8→93.3%）；检索腿显示「低… | +0.121（原 X2 口径）/ +0.058（needle v2 重设计后… |
| [`A4-soft-distance-penalty`](A4-soft-distance-penalty/README.md) | ✅ judged | 唯一改善 clean bpc 的元素，但不创造保留——它改变使用率，不改变依赖是否为零。 | −0.0068（1.5006 vs B0 1.5074，唯一负税元素） |
| [`A5-dilated-window`](A5-dilated-window/README.md) | 🏭 reference（含业界结论） | — | —（未做本地实验） |
| [`A6-mixing-operator`](A6-mixing-operator/README.md) | ✅ judged | 内容选择式 Top-K 稀疏注意力在 **T=256 上净亏**：质量越删越差（k=32 越界有害、k=64 只在噪声带内），成本**实测上升** 2.2×（慢且更占显存）——… | k=32 Δ = **−0.0166**（越 0.014 界，配对 20/2… |
| [`A7-addressing`](A7-addressing/README.md) | 🏭 reference（含业界结论） | — | —（未做本地实验） |
| [`A8-positional-semantics`](A8-positional-semantics/README.md) | ◐ partial | 绝对位置嵌入是硬上界（解码总长 ≤ seq_len），相对偏置族已侧证。 | dist_pen（相对偏置族）−0.007 |
| [`A9-old-info-retention`](A9-old-info-retention/README.md) | ◐ partial | 保留 / 硬丢 / 锚豁免已测；压缩、递归、外置三类零经验。 | 硬丢 +0.058~0.121（A3） |
| [`A10-compression-trigger`](A10-compression-trigger/README.md) | ✅ judged | 「按访问频率优先压（LRU）」在同等删除预算下没有可见收益**——字面优势经对账后归因于「恰好删掉的可见性对数更少」，重对齐后消失（见 ERRATA）。压缩优先级这一维度在 1… | 依赖腿伤害（N=8/32/64）：distance +0.197/+0.64… |

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
