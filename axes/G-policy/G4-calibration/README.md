---
id: G4-calibration
axis: G
grid_dir: G-policy/G4-calibration
title: "校准"
status: judged
judge: "ECE（分桶）+ 阈值开火率可预测性 + 跨 profile 阈值迁移"
cost: "纯推理，分钟级"
tax: "—"
benefit: "prob0 ECE 0.002–0.011（M0）→ ≤0.0091（100M）；fire 曲线 split gap ≤0.004；TF 窗口在线代理 ECE 0.0011–0.0079"
verdict: "校准跨规模存活是本项目最稳的资产之一。"
deps: []
evidence: [".scratch/02-engine-m0/evidence/REPORT.md @27418ae", ".scratch/16-rung4-scale/evidence @c520d65", ".scratch/15-slider-rung3/evidence @257b6ec"]
---

<!-- AUTO:begin -->

# G4-calibration · 校准

> 轴：**G 策略与调度** ｜ 状态：**✅ judged**

> **判读**：校准跨规模存活是本项目最稳的资产之一。

## 1. 命题

置信信号是否可信——决定阈值策略与在线代理的有效性。

## 2. 选项枚举（本格全部可能取值）

1. `无`
2. `ECE gate（训练期验证）`
3. `温度/在线重校准`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：prob0 ECE 0.002–0.011（M0）→ ≤0.0091（100M）；fire 曲线 split gap ≤0.004；TF 窗口在线代理 ECE 0.0011–0.0079

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：—
- **收益**：prob0 ECE 0.002–0.011（M0）→ ≤0.0091（100M）；fire 曲线 split gap ≤0.004；TF 窗口在线代理 ECE 0.0011–0.0079
- **判读**：校准跨规模存活是本项目最稳的资产之一。

## 5. 失败路径与原因

1. 阈值跨 profile 需重校准（probe vs dense 两 profile）
2. token retry 的 argmax re-embed 会把 ECE 从 0.001 推到 0.015（量化掉不确定性）

## 6. 判据与判决实验设计

- **判据**：ECE（分桶）+ 阈值开火率可预测性 + 跨 profile 阈值迁移

## 7. 成本估计

- 纯推理，分钟级

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. .scratch/02-engine-m0/evidence/REPORT.md @27418ae
2. .scratch/16-rung4-scale/evidence @c520d65
3. .scratch/15-slider-rung3/evidence @257b6ec

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
