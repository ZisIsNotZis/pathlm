# 对照臂与变异验证（2622-eng1-composable）

本实验不比较「质量」，只验证**可测性**。因此对照不是「谁 bpc 低」，而是三类
「同任务、同权重、同输入」的机械对照：

## 1. 对照臂

| 对照臂 | 输入分布 | 要预测什么 | 部署中对应什么 |
|---|---|---|---|
| `default`（MHA、无 carry、无 window）= stock `Block` | CPU 随机 token `[2,12]`，vocab 10 | 逐深 next-token（本实验只用前向一致性） | 「什么都不开」的基底；ENG1 不许改它 |
| `full-window`（`train_window=12=seq_len`） | 同组合、同 seed、同权重 | 同目标 | 「窗口开关关掉」的对手；与窗口臂之比只含掩码 |
| 快路径 `GQABlock` / `CarryChannelBlock` | 同测试位点 | 同目标 | 既有单特性实现；ENG1 不许改块类 |

- **窗口对照**：`m_win` 与 `m_full` 同 seed、逐字段权重逐位相同（`train_window` 不消耗
  随机数）⇒ 二者 logits 若不同，差异只能来自**掩码**。这就是「窗口活着」的证据。
- **carry 对照**：读出增益单边零起步 ⩾ 恒等；把 `carry_scale` 填成同一常量后
  `read_share > 0`，把它置零会改变 logits。这就是「通道被读」的证据。

## 2. 变异验证（`mutate.py`）

在 `pathlm/depth_ar.py` 上做四处「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」，各对准一条组合：

| id | 破坏点 | 期望抓红的单测 |
|---|---|---|
| M1 | 可组合块的写入通道读出被丢掉（`a + scaled*0`） | `test_eng1_gqa_x_carry_composes_and_both_active`、`…all_compose` |
| M2 | KV 头投影的注意力忽略窗口掩码（`attn_mask=None`） | `test_eng1_gqa_x_window_composes_and_window_active`、`…all_compose` |
| M3 | 写入通道块忽略窗口掩码 | `test_eng1_carry_x_window_composes_and_both_active` |
| M4 | 可组合块前向忽略 `attn_mask` | `test_eng1_gqa_x_carry_x_window_all_compose` |

- 每处先记原文 sha256，跑指定单测（期望非零退出 = 红），再逐字节还原并核对 sha256。
- 结果：`mutation_report.json::all_caught_red` / `all_restored_bitwise`（见 `analysis.md`）。
- **只动 `pathlm/depth_ar.py`**（本线持有代码锁，SCHEMA §22；不碰 `train_depth_ar.py`
  与其它模块）。
