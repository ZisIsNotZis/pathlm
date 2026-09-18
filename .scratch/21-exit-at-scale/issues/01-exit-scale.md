# 21 — 整形 vs 自适应定律 @100M 复测（exit 双 profile）

- **Status:** in_progress
- **Type:** scale study（exit 机制）
- **Related:** 18-exit-probe（13M 定案：两全证伪，整形/retry 同一梯度两面）
- **目标：** ticket 18 的权衡在 100M 是否成立——
  1. EX1_100M（probe，trunk stop-grad）：retry 门控存活？
  2. EX2_100M（dense 0.2 温和整形）：retry 死 + exit 质量轴好 + aux 正则？
  3. 13M 的 dose 非敏感性（0.2 已杀 retry）在 100M 是否重现。

## 协议

- C1_100M 协议 + {w_dense_exit 1.0+exit_probe / w_dense_exit 0.2}，
  各 24000×8（与 ticket 16 同 token 预算口径），probe_exit.py 测量。
- 对照：C1_100M（同协议无 dense，ticket 16/19 数字）。

## 判据

- 定律成立 ⇔ 两 profile 的行为模式与 13M 定性一致（probe retry 活 / dense 死
  且 exit 轴好）。
- 任何方向性偏离 → 缩放定律修正（如 100M 容量下两全可能出现）。
