---
id: M2-carrier-compression
axis: M
grid_dir: M-carrier/M2-carrier-compression
title: "载体压缩"
status: partial
judge: "内存下降 × 质量/保留损失（双边前沿）"
cost: "13M 单臂 ~25 min"
tax: "驱逐 +0.058~0.121（13M）"
benefit: "ring-buffer 解码免重 prefill（内存有界）"
verdict: "驱逐已测（价格见 A3）；量化/摘要/低秩零经验。"
deps: ["A3-window-anchor", "W3-kv-head-sharing"]
evidence: [".scratch/07-x1-x3/evidence @7b47555", "axes/M-carrier/M2-carrier-compression/experiments/2614-kv-ablation-retention-13m/analysis.md @c39dc88"]
---

<!-- AUTO:begin -->

# M2-carrier-compression · 载体压缩

> 轴：**M 状态载体** ｜ 状态：**◐ partial**

> **判读**：驱逐已测（价格见 A3）；量化/摘要/低秩零经验。

## 1. 命题

载体怎么压（内存换质量）。

## 2. 选项枚举（本格全部可能取值）

1. `不压`
2. `量化（KV/权重）`
3. `驱逐窗口`
4. `摘要压缩`
5. `低秩投影（MLA 式）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：ring-buffer 解码免重 prefill（内存有界）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：驱逐 +0.058~0.121（13M）
- **收益**：ring-buffer 解码免重 prefill（内存有界）
- **判读**：驱逐已测（价格见 A3）；量化/摘要/低秩零经验。

## 5. 失败路径与原因

1. 驱逐 × 复制目标冲突（三路豁免全败，见 A3）
2. **保留曲线天平已建成**（ticket 31-I，`tools/evalkit/retention.py`）：三条曲线严格单调、同 seed 重跑最大差 4.94e-07；量出「依赖」随距离快速衰减（d=1 +3.245 / d=8 +0.090 / d=16 +0.017 / d≥32 ≤+0.004），长历史查询丢最远 8 个只 +0.0085（N=16 +0.0135），连续段删除 L=4 平均 +0.114 ⇒ **代价由距离决定、几乎与内容无关；安全丢弃半径 d*≈32**
3. **该仪器只量「依赖」不量「保留」**（SCHEMA §17 第 10 条）：低损伤也可能是模型本来就不用它——X3「假保留」的另一面。检索腿已排进 ticket 32 契约 M

## 6. 判据与判决实验设计

- **判据**：内存下降 × 质量/保留损失（双边前沿）

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：内存下降 × 质量/保留损失（双边前沿）
- 成本：13M 单臂 ~25 min

## 7. 成本估计

- 13M 单臂 ~25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `A3-window-anchor`
2. `W3-kv-head-sharing`

## 9. 复现手段

1. .scratch/07-x1-x3/evidence @7b47555
2. axes/M-carrier/M2-carrier-compression/experiments/2614-kv-ablation-retention-13m/analysis.md @c39dc88

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
