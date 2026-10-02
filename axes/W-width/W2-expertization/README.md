---
id: W2-expertization
axis: W
grid_dir: W-width/W2-expertization
title: "专家化（MoE）"
status: reference
judge: "同 FLOPs 的质量（主）+ 负载均衡 + 路由稳定性 + 内存"
cost: "13M 单臂 ~25–40 min（路由器需要至少 2–4 专家才可判）"
tax: "—（未做本地实验）"
benefit: "业界结论（记忆·未核，非本地实测）：参照线句：「用稀疏路由把参数量与每 token FLOPs 解耦（MoE）是保容量减计算的成熟路线；细粒度专家 + 共享专家为主流配方，负载均衡/路由坍缩是主要风险（`记忆·未核`，Switch/Mixtral/DeepSeek-MoE）。」⇒ 建议 `status: reference`"
verdict: "—"
deps: []
evidence: ["axes/W-width/PRIOR_ART.md @ticket36"]
---

<!-- AUTO:begin -->

# W2-expertization · 专家化（MoE）

> 轴：**W 宽度** ｜ 状态：**🏭 reference（含业界结论）**

## 1. 命题

用稀疏路由把「参数量」与「每 token FLOPs」解耦。

## 2. 选项枚举（本格全部可能取值）

1. `无（稠密）`
2. `MoE top-k 路由`
3. `激活更少专家（top-1）`
4. `激活更小专家（细粒度专家）`
5. `共享专家 + 路由专家`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：业界结论（记忆·未核，非本地实测）：参照线句：「用稀疏路由把参数量与每 token FLOPs 解耦（MoE）是保容量减计算的成熟路线；细粒度专家 + 共享专家为主流配方，负载均衡/路由坍缩是主要风险（`记忆·未核`，Switch/Mixtral/DeepSeek-MoE）。」⇒ 建议 `status: reference`

**行业口径**：MoE 是「同 FLOPs 更多参数」唯一被大规模验证的路线（Mixtral/DeepSeek-MoE 系，
细粒度专家 + 共享专家成为主流配方）。负载均衡与路由坍缩是主要工程风险。

## 3b. 既有做法（ticket 36 分诊，SCHEMA §21）

**业界结论（非本地实测）**：参照线句：「用稀疏路由把参数量与每 token FLOPs 解耦（MoE）是保容量减计算的成熟路线；细粒度专家 + 共享专家为主流配方，负载均衡/路由坍缩是主要风险（`记忆·未核`，Switch/Mixtral/DeepSeek-MoE）。」⇒ 建议 `status: reference`

来源：`axes/W-width/PRIOR_ART.md @ticket36`（`记忆·未核`，联网不可用，待核）

## 4. 结论与数字

- **税**：—（未做本地实验）
- **收益**：业界结论（记忆·未核，非本地实测）：参照线句：「用稀疏路由把参数量与每 token FLOPs 解耦（MoE）是保容量减计算的成熟路线；细粒度专家 + 共享专家为主流配方，负载均衡/路由坍缩是主要风险（`记忆·未核`，Switch/Mixtral/DeepSeek-MoE）。」⇒ 建议 `status: reference`

## 5. 失败路径与原因

（无）

## 6. 判据与判决实验设计

- **判据**：同 FLOPs 的质量（主）+ 负载均衡 + 路由稳定性 + 内存

## 7. 成本估计

- 13M 单臂 ~25–40 min（路由器需要至少 2–4 专家才可判）

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. axes/W-width/PRIOR_ART.md @ticket36

## 备注

本地零经验。「激活更小专家而非更大专家」是用户提出的子问题（细粒度专家），与行业趋势一致。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
