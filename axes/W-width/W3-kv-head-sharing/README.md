---
id: W3-kv-head-sharing
axis: W
grid_dir: W-width/W3-kv-head-sharing
title: "KV 头共享"
status: partial
judge: "质量（bpc）+ KV 内存 + 长上下文吞吐"
cost: "13M 单臂 ~25 min"
tax: "—（基底为 MHA）"
benefit: "—"
verdict: "本地只有 MHA；其余为零经验，但这是长上下文线的免费基线（必须先打败它）。"
deps: ["M2-carrier-compression", "A3-window-anchor"]
evidence: []
---

<!-- AUTO:begin -->

# W3-kv-head-sharing · KV 头共享

> 轴：**W 宽度** ｜ 状态：**◐ partial**

> **判读**：本地只有 MHA；其余为零经验，但这是长上下文线的免费基线（必须先打败它）。

## 1. 命题

用更少的 KV 头降低推理内存/带宽。

## 2. 选项枚举（本格全部可能取值）

1. `MHA（本项目）`
2. `MQA（1 个 KV 头）`
3. `GQA（G 个 KV 头）`
4. `MLA（低秩压缩 KV）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：—

**行业口径**：KV 内存降数倍、质量近无损，GQA/MQA/MLA 已被全行业采纳。

## 4. 结论与数字

- **税**：—（基底为 MHA）
- **收益**：—
- **判读**：本地只有 MHA；其余为零经验，但这是长上下文线的免费基线（必须先打败它）。

## 5. 失败路径与原因

（无）

## 6. 判据与判决实验设计

- **判据**：质量（bpc）+ KV 内存 + 长上下文吞吐

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：质量（bpc）+ KV 内存 + 长上下文吞吐
- 成本：13M 单臂 ~25 min

## 7. 成本估计

- 13M 单臂 ~25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `M2-carrier-compression`
2. `A3-window-anchor`

## 9. 复现手段

（本地未做；复现命令待本格实验目录建立后补）

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
