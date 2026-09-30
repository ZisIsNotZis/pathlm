---
id: A3-window-anchor
axis: A
grid_dir: A-mixing/A3-window-anchor
title: "窗口 / 锚点 / sink"
status: judged
judge: "KV 消融保留曲线（真删位置 j 的 KV 测损伤）——不是窗口外答对率"
cost: "13M 单臂 ~15–25 min"
tax: "+0.121（原 X2 口径）/ +0.058（needle v2 重设计后——改善是测量不是机制）"
benefit: "ring-buffer 解码免重 prefill；锚点通道 needle 76.8%（+距离惩罚后 93.3%）"
verdict: "丢前缀的代价**由距离与预算共同决定、并不便宜**（见 ERRATA：只保留最近 32 个 = +0.212 bpc）；锚点通道是真产物（76.8→93.3%）；检索腿显示「低损伤 ≠ 保住了」。"
deps: ["A4-soft-distance-penalty", "M2-carrier-compression", "G1-selector"]
evidence: [".scratch/07-x1-x3/evidence @7b47555", ".scratch/05-m1-followup/evidence @480d50b", ".scratch/13-four-experiments/evidence @9abc181"]
---

<!-- AUTO:begin -->

# A3-window-anchor · 窗口 / 锚点 / sink

> 轴：**A 序列混合机制** ｜ 状态：**✅ judged**

> **判读**：丢前缀的代价**由距离与预算共同决定、并不便宜**（见 ERRATA：只保留最近 32 个 = +0.212 bpc）；锚点通道是真产物（76.8→93.3%）；检索腿显示「低损伤 ≠ 保住了」。

## 1. 命题

用硬性窗口限制可见范围，把「前缀可丢」变成结构事实。

## 2. 选项枚举（本格全部可能取值）

1. `滑动窗口（window = k）`
2. `窗口 + 锚点槽（前 anchors 个位置永不被驱逐）`
3. `窗口 + sink token（attention sink 式保留首 token）`
4. `窗口 + 稀疏全局 token（每隔 k 个位置保留一个）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：ring-buffer 解码免重 prefill；锚点通道 needle 76.8%（+距离惩罚后 93.3%）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：+0.121（原 X2 口径）/ +0.058（needle v2 重设计后——改善是测量不是机制）
- **收益**：ring-buffer 解码免重 prefill；锚点通道 needle 76.8%（+距离惩罚后 93.3%）
- **判读**：丢前缀的代价**由距离与预算共同决定、并不便宜**（见 ERRATA：只保留最近 32 个 = +0.212 bpc）；锚点通道是真产物（76.8→93.3%）；检索腿显示「低损伤 ≠ 保住了」。

## 5. 失败路径与原因

1. needle v1 设计（随机填充单 needle）信号饥饿，全桶测到 0.0 → v2 重设计才可测（假失败的教训：先修测量再判机制）
2. beyond-window 桶召回 0.4%（X2v2）——信息是真的丢了，这是设计不是 bug；对抗性误解：不要用窗口外命中率证明保留
3. 损坏 × 精确复制三路豁免全败（无豁免 0.016 / 锚区豁免 0.033 / 任务级豁免 0.035，且 clean bpc 1.680→1.997）

## 6. 判据与判决实验设计

- **判据**：KV 消融保留曲线（真删位置 j 的 KV 测损伤）——不是窗口外答对率

## 7. 成本估计

- 13M 单臂 ~15–25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `A4-soft-distance-penalty`
2. `M2-carrier-compression`
3. `G1-selector`

## 9. 复现手段

1. .scratch/07-x1-x3/evidence @7b47555
2. .scratch/05-m1-followup/evidence @480d50b
3. .scratch/13-four-experiments/evidence @9abc181

## 备注

这是最接近「右移压缩」的已测机制，也是它的价格上界：
丢掉整个左前缀只花 +0.06~0.12 bpc。任何压缩/摘要求设计必须打败
「就用个窗口」——而窗口的 bpc 代价几乎为零。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
