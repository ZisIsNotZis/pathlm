# 16 — Rung 4：规模研究（100M，2 seeds，定案税/增益缩放）

- **Status:** in_progress
- **Type:** scale study
- **Related:** 15-slider-rung3（Rung 3 关闭）、12-overnight（§4.5 的 29M 单种子探针）
- **目标（WORKSPACE 新会话入口）:** Rung 4 = 规模。100M+ 训练，**2 seeds +
  更大跨度定案税/增益缩放**（13M→29M 的单种子结论：税不缩小、E2E 差距收窄
  因基线不那么脆——需要种子方差与更大跨度确认）。

## 协议

- **模型**：d=720 / 12L / 12 heads / mlp_mult 6 / seq 512 = **100.0M 参数**
  （configs/B0_100M.json、configs/C1_100M.json；resolved config 由 train_m1
  全量落盘，避免 29M 探针配置不可复现的教训）。
- **配置对**（与 §4.5 相同的最小协议）：
  - B0（plain 基线）× 2 seeds
  - C1（direct transport + corrupt_wrong 0.15 + p_retry 0.5）× 2 seeds
- **训练**：12000 步 / batch 16 / lr 1e-3 / cosine（token 预算 = 6000×32 不变；
  batch 32 在 100M+C1 下 OOM，全体同步降保可比）；~1.5–2h/run，
  4 runs 串行 ≈ 4–6h（后台链）。
- **判读**：
  1. retry 税（C1−B0 clean）随规模：13M +0.147 → 29M +0.174 → 100M ?
  2. E2E 损坏差距（B0 损坏 r2 − C1 损坏 r2）：13M 1.66 → 29M 1.13 → 100M ?
  3. 种子方差 @100M（之前从未测过）。
  4. 能力溢价守恒假说（29M 结论）在 100M 是否延续。
- **显存风险**：p_retry>0 的多轮 loss 单次 backward；13M 时最坏 20GB。
  先 20 步冒烟看峰值；OOM 则 batch 16 / steps 12000（token 预算不变，
  全部 4 runs 同步改，保持可比）。

## 产出

- `.scratch/16-rung4-scale/evidence/{B0_100M_s0,B0_100M_s1,C1_100M_s0,C1_100M_s1}/`
  （results.json + train_log.jsonl；model.pt 不入库）
- findings.md 规模节更新；report §4.5 → §4.6；WORKSPACE 状态；commit。
