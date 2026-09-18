# Ticket 18 — probe 式深度头：exit×retry 权衡的显式测量

日期：2026-09-17。EX1 = INT2 协议 + w_dense_exit 1.0 + **exit_probe=True**
（trunk stop-grad，深度头只训头），24000×32，13M。对照：INT2_r3（无 dense）、
ALLOC（dense 非 probe，ticket 14，retry 被毁）、L2（dense 整形，exit 好）。

## EX1 底座

- clean bpc **1.7065**（INT2_r3 1.6698 → probe-dense 税 +0.037；L2 整形版税
  +0.067——probe 更便宜但非免费，耦合通道 = 深度 CE 训练共享 embed/输出头）
- 损坏 off 2.1139（INT2_r3 2.0545，+0.059）

## 结果

### 1. retry 共存：成立 ✓（本 ticket 核心假设被证实）

corrupt 门控质量（EX1 vs 参照 INT2_r3）：

| τ | EX1 gated bpc (off 2.1139) | fire | INT2_r3 (off 2.0545) |
|---|---|---|---|
| 0.5 | 2.1089 (−0.0050) | 7.2% | − |
| 0.9 | 2.1067 (−0.0072) | 13.8% | −0.0037 |
| 0.98 | **2.1041 (−0.0098)** | 23.3% | −0.0043 |

门控响应单调、收益超过参照（ALLOC 是全负）——**probe 式解掉了 dense-exit×retry
的拮抗**。副产物：逐深度 conf-BCE 训练把门信号新练了一遍，门控比参照更有效。

### 2. depth 质量轴：probe 不整形 trunk 的代价被显式测出

clean depth-bpc（p_skip=0）：1:3.60 → 6:2.55 → 11:1.71 → 12:1.62；
corrupt：1:3.85 → 6:2.95 → 11:2.12 → 12:2.02。
对照 L2（整形 trunk）半栈仅 +0.003 bpc——**"整形换 retry"是同一梯度的两面**：
trunk 被浅层目标塑形（L2，exit 好）就毁 retry；不塑形（probe，retry 活）就
浅层读出昂贵。EX1 的 exit 点存在但质量代价大（depth 6 = +0.93 bpc @ 0.5 flop）。

### 3. exit 门逐深度校准：corrupt 分布上极好，clean 浅层过自信

corrupt 输入：depth 1 conf 0.257 / acc 0.261；depth 12 0.605/0.601——按深度校准。
clean 输入浅层过自信（τ=0.95 也在 depth 1 退出）：头在损坏分布上训练，clean
浅层是其 OOD——**exit 阈值必须按部署 profile 校准**（正是 Slider 校准框架的
适用范围，非引擎缺陷）。

## 结论（EX1 部分）

1. **共存成立**：probe 式头保住 retry（甚至更好的门），核心假设证实；
   mental_model §4 的"已知设计冲突与解法"半案结案。
2. **双边前沿的便宜侧有质量代价**：exit 点（flop<1.0）可达但质量差
   （不整形 trunk 的必然）。
3. **剂量问题开放**：温和整形（EX2 = dense 0.2 非 probe）能否以小 retry 损伤
   换 L2 级 exit 质量——EX2 运行中，结果出来后本文件补判读。
