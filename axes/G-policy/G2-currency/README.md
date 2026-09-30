---
id: G2-currency
axis: G
grid_dir: G-policy/G2-currency
title: "记账货币"
status: judged
judge: "同一机制在不同货币下的结论是否分歧（分歧即必须双报）"
cost: "与实验同价（不额外训练）"
tax: "—"
benefit: "同一机制两口径分歧实例：spec 在 forwards 口径 0.55–0.70、FLOPs 口径 ≥1.23；@100M 墙钟 k=2 1.15–1.29× 而 FLOPs 口径有利；深度自适应 C 配方按 FLOPs 口径 122.8 vs 112.9 tok/s"
verdict: "双货币必须双报——本项目多次出现口径分歧。"
deps: []
evidence: [".scratch/15-slider-rung3/evidence @257b6ec", ".scratch/20-spec-scale/evidence @a864bcf", ".scratch/25-tax-attribution/evidence @2e86924"]
---

<!-- AUTO:begin -->

# G2-currency · 记账货币

> 轴：**G 策略与调度** ｜ 状态：**✅ judged**

> **判读**：双货币必须双报——本项目多次出现口径分歧。

## 1. 命题

算力/收益用什么计价。

## 2. 选项枚举（本格全部可能取值）

1. `forwards（每 token 前向次数）`
2. `FLOPs（等价 width-1 单位）`
3. `墙钟`
4. `内存`
5. `延迟分位`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：同一机制两口径分歧实例：spec 在 forwards 口径 0.55–0.70、FLOPs 口径 ≥1.23；@100M 墙钟 k=2 1.15–1.29× 而 FLOPs 口径有利；深度自适应 C 配方按 FLOPs 口径 122.8 vs 112.9 tok/s

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：—
- **收益**：同一机制两口径分歧实例：spec 在 forwards 口径 0.55–0.70、FLOPs 口径 ≥1.23；@100M 墙钟 k=2 1.15–1.29× 而 FLOPs 口径有利；深度自适应 C 配方按 FLOPs 口径 122.8 vs 112.9 tok/s
- **判读**：双货币必须双报——本项目多次出现口径分歧。

## 5. 失败路径与原因

1. 单货币记账会得出相反结论（spec 墙钟收益消失 vs FLOPs 收益仍在）
2. 「逐位相等」是 13M 规模性质，@100M fp 平局翻转（1/400 token）

## 6. 判据与判决实验设计

- **判据**：同一机制在不同货币下的结论是否分歧（分歧即必须双报）

## 7. 成本估计

- 与实验同价（不额外训练）

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. .scratch/15-slider-rung3/evidence @257b6ec
2. .scratch/20-spec-scale/evidence @a864bcf
3. .scratch/25-tax-attribution/evidence @2e86924

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
