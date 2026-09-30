---
id: H4-head-trunk-relation
axis: H
grid_dir: H-horizon/H4-head-trunk-relation
title: "头与主干的关系"
status: judged
judge: "税（挂头是否扰动主干）+ 头自身质量 + 校准"
cost: "13M 单臂 ~5–25 min；冻结主干可复用已有 checkpoint → 单臂 ~5 min"
tax: "probe 式头 @100M ≈0（1.5928 vs 1.5871）；停梯度使浅层读出贵（13M depth6 +0.93）；冻结主干挂跳跃头 13M：Δbpc = **0.0000**（1.4836→1.4836，冻结断言通过）"
benefit: "保 retry（−0.0098）且门控更强（−0.029 @100M）；MTP 头是 d×d 变换、无 vocab 参数（~free 的基础设施）；宽头 ECE 0.0064 ≤0.01 且 10 桶严格单调（0.073→0.982）可作分配信号"
verdict: "挂头是免费旁路（probe 头 @100M 零税；冻结主干跳跃头 13M 零税 Δ=0.0000）。但**冻结主干上的跳跃头无质量头寸**：t+2 命中 0.4667 < 阈值 0.5617（= 0.81 × 直接 t+1），仅在**部署口径**（对手 = 真实 2 次前向 rollout 0.3128）以半成本反超。"
deps: ["C1-layers-per-token", "W4-readout-geometry", "H3-composition"]
evidence: [".scratch/18-exit-probe/evidence @83bf1ac", ".scratch/21-exit-at-scale/evidence @9fe4cf1", "axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/analysis.md @1181cd4"]
---

<!-- AUTO:begin -->

# H4-head-trunk-relation · 头与主干的关系

> 轴：**H 视界（一次前向产生几个位置）** ｜ 状态：**✅ judged**

> **判读**：挂头是免费旁路（probe 头 @100M 零税；冻结主干跳跃头 13M 零税 Δ=0.0000）。但**冻结主干上的跳跃头无质量头寸**：t+2 命中 0.4667 < 阈值 0.5617（= 0.81 × 直接 t+1），仅在**部署口径**（对手 = 真实 2 次前向 rollout 0.3128）以半成本反超。

## 1. 命题

额外的头（MTP/读出头/判断头）是联合训练还是挂在冻结主干上。

## 2. 选项枚举（本格全部可能取值）

1. `联合训练`
2. `冻结主干只训头`
3. `共享 embedding（tied）`
4. `独立解码头`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：保 retry（−0.0098）且门控更强（−0.029 @100M）；MTP 头是 d×d 变换、无 vocab 参数（~free 的基础设施）；宽头 ECE 0.0064 ≤0.01 且 10 桶严格单调（0.073→0.982）可作分配信号

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：probe 式头 @100M ≈0（1.5928 vs 1.5871）；停梯度使浅层读出贵（13M depth6 +0.93）；冻结主干挂跳跃头 13M：Δbpc = **0.0000**（1.4836→1.4836，冻结断言通过）
- **收益**：保 retry（−0.0098）且门控更强（−0.029 @100M）；MTP 头是 d×d 变换、无 vocab 参数（~free 的基础设施）；宽头 ECE 0.0064 ≤0.01 且 10 桶严格单调（0.073→0.982）可作分配信号
- **判读**：挂头是免费旁路（probe 头 @100M 零税；冻结主干跳跃头 13M 零税 Δ=0.0000）。但**冻结主干上的跳跃头无质量头寸**：t+2 命中 0.4667 < 阈值 0.5617（= 0.81 × 直接 t+1），仅在**部署口径**（对手 = 真实 2 次前向 rollout 0.3128）以半成本反超。

## 5. 失败路径与原因

1. probe 式头改善 retry 但浅层读出质量贵 → 「保 retry」与「exit 质量」是同一梯度的两面（与 C6 同一条定律）
2. **冻结主干上跳跃头质量不可达**：t+2 0.4667（宽头 0.5081）< 0.81×0.6934；决策树排除 off-by-one（shift 单测 + oracle 恒等式）/ 冻结失败（Δbpc=0 且 grad is None）/ 头映射（直接 t+1 0.6951 ≈ 主干 0.6934）/ 步数与容量（宽头仅 +4.1pp）→ 归因**设计无效**：冻结表征只被 t+1 目标塑形，t+2 映射不可达（回环时代 joint 的 55.97% 是表征可用性差异）
3. **判据设计教训（双口径）**：预登记口径用 oracle 两跳（喂真 t+1，= 单跳质量 0.6953）作分母，比真实对手严格得多；真实等 FLOPs 对手（自己预测的 t+1 再前向）= **0.3128**，跳跃头半成本反超 +15.4pp。两结论并存，以预登记为准
4. 对照 3（联合臂）口径缺陷：lr 3e-4 把主干改坏（bpc 1.48→1.85），非干净对照，仅记录

## 6. 判据与判决实验设计

- **判据**：税（挂头是否扰动主干）+ 头自身质量 + 校准

## 7. 成本估计

- 13M 单臂 ~5–25 min；冻结主干可复用已有 checkpoint → 单臂 ~5 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `C1-layers-per-token`
2. `W4-readout-geometry`
3. `H3-composition`

## 9. 复现手段

1. .scratch/18-exit-probe/evidence @83bf1ac
2. .scratch/21-exit-at-scale/evidence @9fe4cf1
3. axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/analysis.md @1181cd4

## 备注

最便宜的真空白：拿一个冻结的 stock 小模型只训跳跃头，
可一次排 3–5 个变体，且完全不需要训练主干。
13M 首验（ticket 28）：零税坐实，但**质量头寸不存在**——若再判，应**预登记 deploy rollout
为对手**（oracle 作上界附报），并考虑 H3 的联合/自门控组合路线（表征可用性才是瓶颈）。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
