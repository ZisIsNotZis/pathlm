---
id: R4-window-kv-quant-retrieval
axis: R
grid_dir: R-references/R4-window-kv-quant-retrieval
title: "窗口注意力 + KV 量化 + 检索"
status: reference
judge: "同内存预算下的长上下文质量 + 检索召回"
cost: "13M 单臂 ~25 min（窗口部分已测）"
tax: "窗口部分：+0.058~0.121 bpc（13M）；检索/量化部分零经验"
benefit: "锚点通道 93.3%；ring-buffer 免重 prefill"
verdict: "—"
deps: []
evidence: [".scratch/07-x1-x3/evidence @7b47555"]
---

<!-- AUTO:begin -->

# R4-window-kv-quant-retrieval · 窗口注意力 + KV 量化 + 检索

> 轴：**R 参照线（必须打败的基线）** ｜ 状态：**🏭 reference**

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

## 5. 失败路径与原因

1. 关键判据线：「就用个窗口」的 bpc 代价几乎为零 → 压缩/摘要线必须证明「保留」而非「省钱」

## 6. 判据与判决实验设计

- **判据**：同内存预算下的长上下文质量 + 检索召回

## 7. 成本估计

- 13M 单臂 ~25 min（窗口部分已测）

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. .scratch/07-x1-x3/evidence @7b47555

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
