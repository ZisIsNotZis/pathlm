# CRITIQUE — R1-quantization（独立批判审计，fork=false）

## ① 这个课题是什么
把推理期的 KV cache 用更低比特存起来换内存/带宽。它在本仓的角色是**参照线**：任何"省内存/丢信息"的主张都必须先打败它。

## ② 它对着什么 sota/业界标准，我们偏离在哪
- **业界标准**（离线无法联网核证，按"待核假设"记，需在可联网时核 `ggml-quants` 的 `QK8_0`/`block_q8_0` 与 KIVI 的 K per-channel / V per-token）：KV 量化把**标尺按块/通道摊薄**——G 个连续值共享 1 个 fp16 标尺，标尺开销 = `2/G B/值`，**与序列长 T 无关**。代表实现：llama.cpp 的 KV 量化按 **head_dim 方向**每 32 个值一块（q8_0 = 34 B/32 值 = **1.0625 B/值**、q4_0 = 18 B/32 = **0.5625 B/值**）；KIVI 的 K per-channel / V per-token。
- **我们的偏离**：`tools/evalkit/kv_quant.py` 是**自写对称量化器**。2026-10-02 的 F1"因果化"把它改成**逐值标尺**：`kv_quant.py:127` 对 token 轴取 `cummax`，`scale` 形状 `[B,H,T,head_dim]`，即"每个值一个 fp16 标尺"（`kv_quant.py:33` 自述 int8 净 **3 B/值**）。
- **仓内自我反证**：同一仓里 per-channel 标尺的总开销是 **T 无关的 12288 B**（`2614-kv-quant-eval-13m/analysis.md:49`；`kv_quant.py:273` 的 `scale_bytes_total`），T=256 上只有 48 B/位置 ⇒ int8 ≈ 6192 B/位置 ≈ **−49.6%**。"标尺可摊薄"本仓原本就有，F1 把它降级成了逐值标尺。

## ③ 五类问题逐条判定
- **开倒车：有。** 自造口径（逐值标尺）与业界块/通道标尺不一致，据此得出与业界相反的结论。
- **过程错误：有。**
  - 无"标准块标尺"对照臂。`2631-kv-budget-causal/budget_caliber.py` 只给两种口径：**(a)** 逐值标尺、**(b)** "每 G 个**连续可见位置**"共享标尺 + **块内回填**；它把分块**只沿 token 轴**做，于是 (b) 被定义成"必须放弃前缀确定"。同一模块的 `mixed_axis_note` 自己写出："V 按 token 取 head_dim 维 amax ⇒ 严格因果且 +2/32 B/值"，紧接着断言"**K 侧做不到**"，未给理由。**块沿 head_dim 对 K 同样成立**（llama.cpp 即如此），此时 `+2/G B/值`、严格因果、无需回填/staging。
  - 口径混用：值字节口径（6144/3072）与部署口径（18432/15360）同处一个 `tax` 字段，直到"省内存"被推翻才拆分。
- **结论错误：有（本次审计最严重）。** `catalog.yaml:1205–1219` 写"**量化在任何拓扑上都是内存负收益**"、"旧的『近无损省内存』参照线作废"（`axes/R-references/R1-quantization/README.md` 同步）。
  **正确结论**：标准块/通道标尺下 KV int8 = **1.0625 B/值（省 46.9%）**、int4 = **0.5625（省 71.9%）** 且**严格前缀确定**——这两个数本仓自己算过（`2631/analysis.md:176`、`2631/table.md:266`）。所以"**int8 KV 近无损且省约一半**"这条参照线**应保留**；被推翻的只是"逐值标尺实现下量化省内存"。
- **无意义：否。** 所有省内存线的判据基准。
- **值得做：是（最高优先级重做）。**

## ④ 建议动作：重做
1. `tools/evalkit/kv_quant.py` 增加**块沿 head_dim** 的标尺（G∈{16,32,64}，G=32 主）与 **KIVI 式 K per-channel / V per-token**；逐值标尺**降为对照臂**。
2. 内存横轴统一"值+标尺"部署口径，逐臂报 `bytes_per_position` 与对 fp16 的倍率。
3. 复用 `2631` 的 27 臂重测；把**前缀不变性**单测打到新实现（现有 `causal_check.py` 应能对 token 轴实现抓红）。
4. 修 `catalog.yaml` R1 与 `axes/R-references/R1-quantization/README.md`、`2614/README.md`、`2621/README.md`，并补 `carry_quant.py:12`/`quant_carry.py:11` 的过期 docstring（L1-1 已挂账）。

## ⑤ 正确的验证方式
- **该跌**：解析每位置字节 → int8 ≈ 1.0625 B/值（−46.9%）、int4 ≈ 0.5625（−71.9%）；同拓扑 Pareto 上"量化点"应重新出现。
- **该涨**：质量损伤 → 块标尺取整误差上界 ≥ 逐值标尺；int4 Δbpc 应比 +0.0247 略大（旧账 +0.0457 是另一口径的上界）；int8 仍应 < 0.014。
- **必须不变**：① 前缀不变性两条 = True（head_dim 块天然满足；token 轴块必须为 False，这是抓红点）；② 非量化臂逐位不变；③ 毁源后召回掉回 chance；④ 同拓扑 fp16 基线 bpc。
