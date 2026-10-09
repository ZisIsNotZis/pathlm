---
id: R4-window-kv-quant-retrieval
axis: R
grid_dir: R-references/R4-window-kv-quant-retrieval
title: "窗口注意力 + KV 量化 + 检索"
status: todo
judge: "同内存预算下的长上下文质量 + 检索召回"
cost: "13M 单臂 ~25 min（窗口部分已测）"
tax: "窗口部分：+0.058~0.121 bpc（13M）；检索/量化部分零经验"
benefit: "锚点通道 93.3%；ring-buffer 免重 prefill"
verdict: "**已有**：本地仅覆盖窗口/锚点部分（税 +0.058~0.121 bpc；锚点通道 93.3%）。**尚缺**：按保留预算的窗口曲线 + 检索腿是否显著 >chance；量化部分按 R1 口径重测（业界基线结论须核实出处）（待补做）。"
deps: []
evidence: [".scratch/07-x1-x3/evidence @7b47555", "axes/M-carrier/M2-carrier-compression/experiments/2614-kv-ablation-retention-13m/analysis.md @c39dc88"]
---

<!-- AUTO:begin -->

# R4-window-kv-quant-retrieval · 窗口注意力 + KV 量化 + 检索

> 轴：**R 参照线（必须打败的基线）** ｜ 状态：**⬜ todo**

> **判读**：**已有**：本地仅覆盖窗口/锚点部分（税 +0.058~0.121 bpc；锚点通道 93.3%）。**尚缺**：按保留预算的窗口曲线 + 检索腿是否显著 >chance；量化部分按 R1 口径重测（业界基线结论须核实出处）（待补做）。

## 1. 命题

长上下文的现成组合拳——右移压缩必须打败的对手。

## 2. 选项枚举（本格全部可能取值）

1. `窗口 + sink`
2. `KV 量化`
3. `检索增强（RAG / memorizing）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：锚点通道 93.3%；ring-buffer 免重 prefill

**行业口径**：生产系统的默认答案。

## 4. 结论与数字

- **税**：窗口部分：+0.058~0.121 bpc（13M）；检索/量化部分零经验
- **收益**：锚点通道 93.3%；ring-buffer 免重 prefill
- **判读**：**已有**：本地仅覆盖窗口/锚点部分（税 +0.058~0.121 bpc；锚点通道 93.3%）。**尚缺**：按保留预算的窗口曲线 + 检索腿是否显著 >chance；量化部分按 R1 口径重测（业界基线结论须核实出处）（待补做）。

## 5. 失败路径与原因

1. 关键判据线：「就用个窗口」的 bpc 代价几乎为零 → 压缩/摘要线必须证明「保留」而非「省钱」

## 6. 判据与判决实验设计

- **判据**：同内存预算下的长上下文质量 + 检索召回

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：同内存预算下的长上下文质量 + 检索召回
- 成本：13M 单臂 ~25 min（窗口部分已测）

## 7. 成本估计

- 13M 单臂 ~25 min（窗口部分已测）

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. .scratch/07-x1-x3/evidence @7b47555
2. axes/M-carrier/M2-carrier-compression/experiments/2614-kv-ablation-retention-13m/analysis.md @c39dc88

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
