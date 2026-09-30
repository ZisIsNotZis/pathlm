---
id: F3-stopping
axis: F
grid_dir: F-iteration/F3-stopping
title: "停止条件"
status: judged
judge: "质量-成本前沿 + 门控选择性（按置信分桶的每 FLOP 收益）"
cost: "13M 单臂 ~25–40 min（含校准）"
tax: "分配器本身不额外加税（单轮 mixture == overwrite，设计保证）"
benefit: "分配选择性：按 prob0 只对最低 15% 开第二轮，INT2 +0.0211 vs 均匀 −0.0193；C1M +0.1284 vs +0.0570（2.2 倍）；收益集中在中低置信带（+0.04~+0.06），高置信带 ≈0/负；严格占优点复现（0.635 fwd/tok @2.0502 vs plain 1.0 @2.0545）；成本模型预测误差 ≤±13%"
verdict: "选择性问题（愿景核心命题）获证；预算求解器端到端建成。"
deps: ["G1-selector", "G2-currency", "G3-signal", "G4-calibration"]
evidence: [".scratch/14-allocator/evidence @9954425", ".scratch/15-slider-rung3/evidence @257b6ec", ".scratch/16-rung4-scale/evidence @c520d65", ".scratch/22-gen-quality/evidence @c6c356f"]
---

<!-- AUTO:begin -->

# F3-stopping · 停止条件

> 轴：**F 迭代** ｜ 状态：**✅ judged**

> **判读**：选择性问题（愿景核心命题）获证；预算求解器端到端建成。

## 1. 命题

何时停止迭代——固定、门控、收敛还是预算求解。

## 2. 选项枚举（本格全部可能取值）

1. `固定轮数`
2. `prob0 门控（逐 token）`
3. `收敛判据（Δ 小）`
4. `预算求解器（多货币）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：分配选择性：按 prob0 只对最低 15% 开第二轮，INT2 +0.0211 vs 均匀 −0.0193；C1M +0.1284 vs +0.0570（2.2 倍）；收益集中在中低置信带（+0.04~+0.06），高置信带 ≈0/负；严格占优点复现（0.635 fwd/tok @2.0502 vs plain 1.0 @2.0545）；成本模型预测误差 ≤±13%

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：分配器本身不额外加税（单轮 mixture == overwrite，设计保证）
- **收益**：分配选择性：按 prob0 只对最低 15% 开第二轮，INT2 +0.0211 vs 均匀 −0.0193；C1M +0.1284 vs +0.0570（2.2 倍）；收益集中在中低置信带（+0.04~+0.06），高置信带 ≈0/负；严格占优点复现（0.635 fwd/tok @2.0502 vs plain 1.0 @2.0545）；成本模型预测误差 ≤±13%
- **判读**：选择性问题（愿景核心命题）获证；预算求解器端到端建成。

## 5. 失败路径与原因

1. 自由生成段 prob0 检测失效（TF 窗口 ECE 0.0011–0.0079，生成段压天花板）→ 需要替代信号（见 G3）
2. 「13M prob0 在生成段反向」系 n=1 假象，n=6 复刻未复现（方向与 100M 一致）——n=1 负结果必须复现后入账

## 6. 判据与判决实验设计

- **判据**：质量-成本前沿 + 门控选择性（按置信分桶的每 FLOP 收益）

## 7. 成本估计

- 13M 单臂 ~25–40 min（含校准）

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `G1-selector`
2. `G2-currency`
3. `G3-signal`
4. `G4-calibration`

## 9. 复现手段

1. .scratch/14-allocator/evidence @9954425
2. .scratch/15-slider-rung3/evidence @257b6ec
3. .scratch/16-rung4-scale/evidence @c520d65
4. .scratch/22-gen-quality/evidence @c6c356f

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
