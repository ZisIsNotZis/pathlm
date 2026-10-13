---
id: G3-signal
axis: G
grid_dir: G-policy/G3-signal
title: "观测信号"
status: judged
judge: "AUC / ECE / 与真实错误率的分层单调性 / 零额外前向"
cost: "纯推理，分钟级"
tax: "—"
benefit: "prob0 管 token 对错（ECE 0.002–0.011，跨规模存活）；路径分歧 js 管语义分歧（AUC 0.66–0.69 > prob0 0.58–0.62；生成段 real 0.0114 < corrupt 0.0250 < random 0.0390；注入剂量单调）；distinct-2 管塌缩（AUC 0.9815，与 js 正交 rank_corr 0.14）"
verdict: "三信号正交监测栈成形；生成段没有单一信号够用。"
deps: ["G1-selector", "G4-calibration"]
evidence: [".scratch/22-gen-quality/evidence @c6c356f"]
---

<!-- AUTO:begin -->

# G3-signal · 观测信号

> 轴：**G 策略与调度** ｜ 状态：**✅ judged**

> **判读**：三信号正交监测栈成形；生成段没有单一信号够用。

## 1. 命题

用什么信号判断「这个 token 难不难 / 这段生成好不好」。

## 2. 选项枚举（本格全部可能取值）

1. `prob0`
2. `路径分歧度（js）`
3. `重复率（distinct-2）`
4. `深度置信`
5. `未校准 logit gap`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：prob0 管 token 对错（ECE 0.002–0.011，跨规模存活）；路径分歧 js 管语义分歧（AUC 0.66–0.69 > prob0 0.58–0.62；生成段 real 0.0114 < corrupt 0.0250 < random 0.0390；注入剂量单调）；distinct-2 管塌缩（AUC 0.9815，与 js 正交 rank_corr 0.14）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：—
- **收益**：prob0 管 token 对错（ECE 0.002–0.011，跨规模存活）；路径分歧 js 管语义分歧（AUC 0.66–0.69 > prob0 0.58–0.62；生成段 real 0.0114 < corrupt 0.0250 < random 0.0390；注入剂量单调）；distinct-2 管塌缩（AUC 0.9815，与 js 正交 rank_corr 0.14）
- **判读**：三信号正交监测栈成形；生成段没有单一信号够用。

## 5. 失败路径与原因

1. prob0 在自由生成段压天花板（TF 窗口 ECE 0.0011–0.0079 但生成段分不开 real/random）
2. js 在吸引子塌缩窗双向失灵（假阴 js=0 与假阳 js=0.373 并存）
3. greedy 生成有全局重复偏置 → 监测参考水平必须相对化

## 6. 判据与判决实验设计

- **判据**：AUC / ECE / 与真实错误率的分层单调性 / 零额外前向

## 7. 成本估计

- 纯推理，分钟级

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `G1-selector`
2. `G4-calibration`

## 9. 复现手段

1. .scratch/22-gen-quality/evidence @c6c356f

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
