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
evidence: [".scratch/07-x1-x3/evidence @7b47555"]
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

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
