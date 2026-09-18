# 20 — node-3+ spec 宽度缩放 @100M

- **Status:** in_progress
- **Type:** scale study（spec 机制）
- **Related:** 13-four-experiments（13M：a₂=0.844，a₃|a₂=0.653，2.40 tok/fwd，
  k=2 1.68× 稳态）、17-slider-at-scale（INT_100M，n_mtp=2）
- **目标：** spec 接受率与吞吐是否随规模上升（更大模型预测更准 → 接受率更高 →
  spec 增益放大）。13M 基线：k=2 稳态 1.68×/random 1.85×，2.40 tok/fwd。

## 协议

1. **INT3_100M_s0 训练**：INT_100M 协议 + n_mtp=3（13M 的 2→3 增量税 +0.017），
   32000×6，~1.5h。
2. **probe_spec**（宽度 1/2/3 微基准 + 稳态解码，random + enwik8 prompt，
   接受级联 + tok/forward），对照 13M B3 数字。
3. **判读**：a₂/a₃ 随规模方向；k=2 speedup vs 13M 的 1.68×；门控重试与
   n_mtp=3 的组合不在此 scope（13M 已证 retry 质量杠杆随规模自衰减）。

## 产出

`.scratch/20-spec-scale/evidence/`；findings MTP 节补行；report；WORKSPACE；commit。
