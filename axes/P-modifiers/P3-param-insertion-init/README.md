---
id: P3-param-insertion-init
axis: P
grid_dir: P-modifiers/P3-param-insertion-init
title: "参数插入与初始化（修饰符）"
status: judged
judge: "起步是否严格等价于基线（逐位还原）+ 最终质量"
cost: "13M 单臂 ~15–25 min"
tax: "proj-fill 1.8356 < fill 1.8582 < nofill 1.9438（族内最优，全深度逐深度支配 fill）"
benefit: "恒等初始化不消耗随机数 → 同 seed 下与 fill 臂逐位一致（可严格对照）；carry 门 bias = σ(−2)≈0.12 起步（恒等近似 + 梯度可流）；**单边零**是通用的安全恒等起步写法（输出侧置零 + 输入侧随机）"
verdict: "恒等初始化 adapter 在 13M matched-step 族内最优——「从基线出发学 remap」是对的。"
deps: []
evidence: [".scratch/23-depth-ar/evidence @6b27b5e", ".scratch/23-depth-ar/issues @1586745", "axes/W-width/W4-readout-geometry/experiments/2610-gated-readout-13m-depthar/zero-init-saddle/ @cba2f39"]
---

<!-- AUTO:begin -->

# P3-param-insertion-init · 参数插入与初始化（修饰符）

> 轴：**P 修饰符（不是轴）** ｜ 状态：**✅ judged**

> **判读**：恒等初始化 adapter 在 13M matched-step 族内最优——「从基线出发学 remap」是对的。

## 1. 命题

新参数插在哪、初始化成什么——决定「恒等起步」能否成立。

## 2. 选项枚举（本格全部可能取值）

1. `无`
2. `恒等初始化 adapter（proj-fill，本项目）`
3. `负偏置门（carry 初始化）`
4. `LoRA / 低秩适配器`
5. `prefix / prompt 参数`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：恒等初始化不消耗随机数 → 同 seed 下与 fill 臂逐位一致（可严格对照）；carry 门 bias = σ(−2)≈0.12 起步（恒等近似 + 梯度可流）；**单边零**是通用的安全恒等起步写法（输出侧置零 + 输入侧随机）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：proj-fill 1.8356 < fill 1.8582 < nofill 1.9438（族内最优，全深度逐深度支配 fill）
- **收益**：恒等初始化不消耗随机数 → 同 seed 下与 fill 臂逐位一致（可严格对照）；carry 门 bias = σ(−2)≈0.12 起步（恒等近似 + 梯度可流）；**单边零**是通用的安全恒等起步写法（输出侧置零 + 输入侧随机）
- **判读**：恒等初始化 adapter 在 13M matched-step 族内最优——「从基线出发学 remap」是对的。

## 5. 失败路径与原因

1. proj-fill 只在 13M matched-step 验证过，未上 100M
2. **恒等初始化 ≠ 零初始化**：全零初始化在乘性结构上是**鞍点**（两端梯度同时为 0）→ 干预系统性失效而测试全绿（详见 W4 的 12k 死干预事件与 SCHEMA §10 干预活性判据）
3. 契约里写「建议零初始化或极小值」这类把初始化取值判断下放给执行者的措辞会埋坑：必须给定值或给定判据

## 6. 判据与判决实验设计

- **判据**：起步是否严格等价于基线（逐位还原）+ 最终质量

## 7. 成本估计

- 13M 单臂 ~15–25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. .scratch/23-depth-ar/evidence @6b27b5e
2. .scratch/23-depth-ar/issues @1586745
3. axes/W-width/W4-readout-geometry/experiments/2610-gated-readout-13m-depthar/zero-init-saddle/ @cba2f39

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
