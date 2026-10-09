---
id: G1-selector
axis: G
grid_dir: G-policy/G1-selector
title: "路径选择器"
status: todo
judge: "每 FLOP 边际收益 + 选择器的校准（ECE）+ 开火率可预测性"
cost: "13M 单臂 ~25–40 min"
tax: "—（选择器本身不加税）"
benefit: "Slider 端到端：校准（fire split gap ≤0.004）→ 双货币成本模型（预测 ≤±13%）→ 求解器 → 验证解码 → 在线代理（TF 窗口 ECE 0.0011–0.0079）；严格占优点复现"
verdict: "**已有**：路径选择器前五种已建成或已测（原 F3-stopping 已并入本格：停止条件与预算分配同一条线）。**尚缺**：按原预登记判据（分桶单调 + 低置信桶 ≥ 高置信桶 2×）照原样跑选择性问题是否获证（待补做）。"
deps: ["G3-signal", "G4-calibration"]
evidence: [".scratch/15-slider-rung3/evidence @257b6ec", ".scratch/17-slider-at-scale/evidence @25704b8", ".scratch/14-allocator/evidence @9954425"]
---

<!-- AUTO:begin -->

# G1-selector · 路径选择器

> 轴：**G 策略与调度** ｜ 状态：**⬜ todo**

> **判读**：**已有**：路径选择器前五种已建成或已测（原 F3-stopping 已并入本格：停止条件与预算分配同一条线）。**尚缺**：按原预登记判据（分桶单调 + 低置信桶 ≥ 高置信桶 2×）照原样跑选择性问题是否获证（待补做）。

## 1. 命题

谁决定这条路怎么走。

## 2. 选项枚举（本格全部可能取值）

1. `固定（架构决定）`
2. `输入无关阈值`
3. `逐 token 阈值（prob0 门控）`
4. `学习门（carry gate）`
5. `离线求解器（Slider）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：Slider 端到端：校准（fire split gap ≤0.004）→ 双货币成本模型（预测 ≤±13%）→ 求解器 → 验证解码 → 在线代理（TF 窗口 ECE 0.0011–0.0079）；严格占优点复现

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：—（选择器本身不加税）
- **收益**：Slider 端到端：校准（fire split gap ≤0.004）→ 双货币成本模型（预测 ≤±13%）→ 求解器 → 验证解码 → 在线代理（TF 窗口 ECE 0.0011–0.0079）；严格占优点复现
- **判读**：**已有**：路径选择器前五种已建成或已测（原 F3-stopping 已并入本格：停止条件与预算分配同一条线）。**尚缺**：按原预登记判据（分桶单调 + 低置信桶 ≥ 高置信桶 2×）照原样跑选择性问题是否获证（待补做）。

## 5. 失败路径与原因

1. 在线质量代理在自由生成段失效（prob0 压天花板）
2. 门控重试质量杠杆 @100M 消失（τ 轴平坦 ±0.002）

## 6. 判据与判决实验设计

- **判据**：每 FLOP 边际收益 + 选择器的校准（ECE）+ 开火率可预测性

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：每 FLOP 边际收益 + 选择器的校准（ECE）+ 开火率可预测性
- 成本：13M 单臂 ~25–40 min

## 7. 成本估计

- 13M 单臂 ~25–40 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `G3-signal`
2. `G4-calibration`

## 9. 复现手段

1. .scratch/15-slider-rung3/evidence @257b6ec
2. .scratch/17-slider-at-scale/evidence @25704b8
3. .scratch/14-allocator/evidence @9954425

## 备注

原 F3-stopping 已并入本格（目录见 `archive/dropped/axes/F-iteration/F3-stopping/`）：
「选择器与预算分配」是同一条线；F3 的选择性判据（分桶单调 + 低置信桶 ≥ 高置信桶 2×）在本格预登记重跑。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
