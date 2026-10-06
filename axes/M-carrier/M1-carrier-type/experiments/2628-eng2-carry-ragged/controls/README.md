# 对照臂与变异验证（2628-eng2-carry-ragged）

本实验不比较「质量」，只验证**可测性**。因此对照不是「谁 bpc 低」，而是四类
「同任务、同权重、同输入」的机械对照：

## 1. 对照臂

| 对照臂 | 输入分布 | 要预测什么 | 部署中对应什么 |
|---|---|---|---|
| `default`（`carry_rule=None`，非 dense）= stock `Block` | CPU 随机 token `[2,12]`，vocab 10 | 逐深 next-token（本实验只用前向一致性） | 「什么都不开」的基底；ENG2 不许改它 |
| `full-depth`（`exit_depths` 全 = L） | 同组合、同 seed、同权重 | 同目标 | 「退出开关关掉」的对手；与混合退出臂之比只含退出/填充语义 |
| `γ=0`（读出增益置零） | 同组合、同 seed、同权重 | 同目标 | 「通道读出关掉」的对手；通道臂与它之比只含记忆读出 |
| `adapter=identity` vs `换方向扰动` | 同组合、同 seed、同权重 | 同目标 | 「填充 KV 源 remap 关掉」的对手；扰动后 logits 必变 = adapter 被消费 |

- **退出对照**：`m(exit_depths=混合)` 与 `m(exit_depths=全深度)` 的 logits 必须逐位
  不同——否则该组合在引擎里是「死开关」（ticket 44-X2 已实测 dense 分支完全忽略
  `exit_depths`，正是本线要避免的形态）。
- **通道对照**：读出增益单边零起步 ⩾ 恒等；填成同一常量后 `read_share > 0`，把它
  置零会改变 logits。这就是「通道被读」的证据。
- **填充对照**：proj-fill 的 adapter 换方向扰动后 logits 必变——否则说明带通道的块
  没消费 `kv_from`（ENG2 前它直接报错）。
- **写入时机对照**：三种退出模式下 `_carry_box["mem"]` 必须**逐位相同**——写入发生在
  退出之前，槽内容与退出模式无关（本线的关键语义决定，见 `README.md` §9）。

## 2. 变异验证（`mutate.py`）

在 `pathlm/depth_ar.py` 上做五处「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」，各对准一条组合/
一条语义：

| id | 破坏点 | 期望抓红的单测 |
|---|---|---|
| M1 | 带通道的块忽略 `kv_from`（混合深度 KV 源失效） | `test_eng2_carry_x_random_exit_x_fill_consumes_frozen_kv` |
| M2 | 带通道的块忽略 `gate_vec`（逐通道门失效） | `test_eng2_carry_x_random_exit_composes`、`…x_ragged_layer_matches_manual_rule` |
| M3 | 带通道的块把记忆读出乘 0（通道成摆设） | `…x_ragged_layer_matches_manual_rule`、`…x_random_exit_composes` |
| M4 | 带通道的块忽略 `attn_mask`（ragged 掩码失效） | `…x_ragged_layer_matches_manual_rule` |
| M5 | 非 dense 前向把「已退出位置的写入」抹掉（槽内容错误地依赖退出模式） | `test_eng2_carry_write_timing_ignores_exit_depths`、`…x_ragged_layer_matches_manual_rule` |

- 每处先记原文 sha256，跑指定单测（期望非零退出 = 红），再逐字节还原并核对 sha256。
- 结果：`mutation_report.json::all_caught_red` / `all_restored_bitwise`（见 `analysis.md`）。
- **只动 `pathlm/depth_ar.py`**（本线持有代码锁，SCHEMA §22）。若共享卡上还有别的线
  在跑（各臂之间会 import 同一文件），变异在**仓库副本**上执行以避免争写——报告里
  的 `sha256_original` / `sha256_restored` 仍对应同一份字节内容。
