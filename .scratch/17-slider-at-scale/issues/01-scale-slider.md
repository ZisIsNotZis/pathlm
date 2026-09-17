# 17 — Slider 上规模：INT profile @100M 的前沿复测

- **Status:** done（Slider @100M 完成）
- **Type:** scale study + allocator
- **Related:** 15-slider-rung3（Rung 3）、16-rung4-scale（100M 底座成立）
- **目标（WORKSPACE 下一步）:** INT profile @100M 的 Slider 前沿复测——把
  Rung 3 的端到端分配器搬到规模上，验证"Slider 底座可上规模"的完整闭环。

## 协议

1. **训练 INT_100M_s0**：configs/INT_100M.json（INT2 协议 @100M：
   n_mtp=2 + latent chain + 损坏 0.075+0.075 + soft transport + p_retry 0.5 +
   window/anchors；**偏离记录：p_needle=0**——needle 已裁决 backlog，与
   Slider 前沿无关且去除拮抗变量）。24000×8 首跑 OOM（blender 重启抢占余量，16.6+6.4GB≈23/23.5）；
   改 **32000 步 × batch 6**（token 预算 98.3M 不变），~2–2.5h。
2. **Slider 探针**：slider.py 直接复用（--config configs/INT_100M.json），
   校准→前沿→双货币求解→验证→代理，与 13M 版同一代码路径。
3. **判读**（vs 13M INT2_r3 前沿）：
   - 严格占优点是否保持（corrupt：spec+门控 vs plain）；
   - fire 曲线/校准在 100M 是否依旧可用（split gap、ECE）；
   - 成本预测误差（a2 部署口径校准后）；
   - tok/s 绝对值与 13M 对比（吞吐尺度）。
4. **产出**：`.scratch/17-slider-at-scale/evidence/`；SUMMARY；findings
   Slider 节补一行；report §3.9/§4.6 补记；WORKSPACE；commit。

## 风险

- 显存：n_mtp=2 + 多轮 loss backward，batch 8 下估 12–16GB（C1 单轮 ~8GB）；
  先 20 步冒烟，OOM 则按 playbook 降 batch（全体口径同步记录）。
- p_needle=0 偏离：13M INT2 协议含 needle；此处去除已裁决机制，结论限定在
  Slider 杠杆（k×τ）层面，不涉复制能力。
