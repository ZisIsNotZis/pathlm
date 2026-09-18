# 18 — probe 式深度头：exit 杠杆校准 + exit×retry 共存（双边前沿）

- **Status:** done（EX1/EX2 双实验 + 权衡定案）
- **Type:** mechanism + engine
- **Related:** 14-allocator（dense-exit×retry 拮抗的发现）、15-slider-rung3、
  mental_model §4（probe 式深度头方案）
- **目标：** 重大节点——exit 杠杆获得校准质量，前沿向下突破 1.0 flop
  （当前不可达的"更便宜"侧）；dense-exit×retry 拮抗被 probe 式解掉；
  Slider 三杠杆（depth/retry/spec）齐全。

## 假设（可证伪）

1. **probe 式头（trunk stop-grad）不再破坏重试精炼**（ALLOC 的失败机制是
   dense-exit 的 trunk 梯度；probe 模式下 trunk 梯度恒等于无 dense 基线，
   已有单测钉住 `test_exit_probe_depth_supervision_leaves_trunk_untouched`）。
2. probe 头仍能把各深度的 prob0 校准好（解码 exit 门读中间深度置信）。
3. 代价：头在移动表征上追赶，深度曲线收敛可能滞后于 L2（对照 L2：半栈
   仅 +0.003 bpc）。

## 协议

1. **EX1 训练**：configs/EX1.json（INT2 协议 + w_dense_exit 1.0 + exit_probe），
   24000×32 seed 0，13M。对照：INT2_r3（同协议无 dense，ticket 15）与
   ALLOC（dense 非 probe，ticket 14）。
2. **测量**（probe_exit.py）：
   - depth 曲线（clean + corrupt 侧）：各深度 bpc——exit 质量轴；
   - exit 阈值扫描解码：mean_depth（flop 成本）+ tok/s + 实测 forwards；
   - **retry 共存性**：EX1 上 gate_probe 的门控质量 vs INT2_r3（ticket 15 数字）——
     重试杠杆若仍翻正/不劣化，共存成立；
   - **双边前沿**：exit 点（cost<1.0 flop，k=0 exit+门控可组合于 decode()）+
     plain + spec 点拼成完整前沿。
3. **判据**：retry 门控在 EX1 上保持 ≥ INT2_r3 收益的一半（ALLOC 是全负）；
   exit 点的 quality-cost 曲线单调可用（深度越浅质量越差、成本越低）。

## 产出

`.scratch/18-exit-probe/evidence/`；findings/report/mental_model 同步；
若共存成立 → mental_model §4 的"已知设计冲突与解法"结案。
