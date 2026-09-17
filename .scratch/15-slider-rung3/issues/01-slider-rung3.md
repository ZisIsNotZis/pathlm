# 15 — Rung 3：Slider 端到端（预算求解 → 分配解码 → 在线质量代理）

- **Status:** done（Rung 3 完成）
- **Type:** mechanism (allocator / 分配策略)
- **Related:** 14-allocator（采纳账本 + 首个前沿点）、12-overnight（INT2）、mental_model §3/§5
- **目标（WORKSPACE §新会话入口）:** 第一步已完成，**下一步 = Rung 3**：Slider 端到端
  ——预算 b → 阈值求解 → 分配解码 → 在线质量代理，13M 上 4090 可行。

## 背景与缺口

ticket 14 已测出第一个前沿点（INT2：spec(k=1)+门控 τ=0.9 严格占优 plain），
但 Slider 还缺四件套（mental_model §3）：

1. **校准曲线**：校准集上 P(prob0 < τ) → 每个 τ 的开火率（成本可预测）。
2. **成本模型**：cost(k, τ) = 等价 width-1 forward 数 / token，
   由 fire_rate(τ)、spec 接受率 a₂、宽度倍率合成（v1 口径：每多一轮 ≈ 1 等价 forward）。
3. **求解器**：给定预算 b → 在前沿插值上选 (k, τ) 使 E[cost] ≤ b 且质量最优；
   或给定质量下限 → 最小成本。
4. **在线质量代理**：生成窗口的平均 prob0 ≈ 预测准确率（ECE 校准 ⇒），
   部署时无 ground truth 也能在线报质量、检测输入劣化。

## 计划

1. **重训 INT2_r3**：原 checkpoint 只存 `.tmp/p3/`（已随权重清理丢失），
   配置从 ticket 12 results.json 完整恢复（`.tmp/INT2.json`），24000 步 ≈ 41 min。
   重训后先对账：clean bpc vs 已发布 1.6548（种子非确定性 ~0.002–0.014 内即可用，
   前沿/校准/求解全部在新权重上自洽重测，不与旧权重数字混排）。
2. **pathlm/slider.py**（库，纯函数可测）：fire 曲线、成本模型、前沿插值、
   预算求解、可靠性分箱。decode.py 增量加 prob0_log 采集（additive，None 时零改动）。
3. **slider.py 探针**（根目录）：校准（eval 前半）→ 前沿网格（k∈{0,1} × τ 扫描，
   质量=门控 TF bpc clean+corrupt，成本=实测 forwards/tok + tok/s）→
   求解（b ∈ {1.0,1.1,1.25,1.5}）→ 验证解码（实测成本 vs 预测 ±容差）→
   代理验证（mean prob0 vs TF 实测准确率，clean vs corrupt 输入的在线检测）。
4. **tests/test_slider.py**：纯函数单测（不依赖 GPU/权重）。
5. **证据**：`.scratch/15-slider-rung3/evidence/`；SUMMARY.md；
   findings/report/mental_model/WORKSPACE 同步；commit。

## 判据（验收）

- **成本可预测**：求解 θ 在 held-out 解码上的实测 forwards/token 与预测差 ≤ ±15%。
- **质量可预测**：实测门控 bpc 与前沿插值预测一致（|Δ| ≤ 方差界 0.014）。
- **代理可用**：跨配置 corr(mean prob0, 实测 acc) ≥ 0.9；corrupt 输入的代理
  显著低于 clean（在线劣化检测）。
- **诚实记录**：exit 杠杆不进前沿（INT2 无 dense-exit 训练，质量未校准）；
  cost < 1.0 fwd/tok 的区域当前不可达（需 exit/深度头），如实写明。

## 全局约束

- 仓库 `/home/z/vibe/pathlm`；策略 `/home/z/vibe/AGENTS.md`。
- **语言**：docs/ticket/evidence/commit 中文；代码标识符英文。
- **测试**：`python3 -m pytest tests/ -q`（52 绿基线）。改引擎必须加能失败的测试。
- **收益规则**：bpc 优先 → 解码速度；每条报"收益值(vs 对照) + 推理成本"；
  双货币记账（等价 forwards 与 wall-clock 分开报，不混用）。
- **证据**：results/probe jsonl → `.scratch/15-slider-rung3/evidence/`；不提交 model.pt。
