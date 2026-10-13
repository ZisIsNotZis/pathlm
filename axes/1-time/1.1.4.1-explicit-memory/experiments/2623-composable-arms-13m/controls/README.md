# 2623 · 对照臂与变异验证说明

## 1. 对照臂（逐臂写明：输入分布 / 要预测什么 / 部署对应）

| 对照 | 定义 | 输入分布 | 要预测什么 | 部署对应 |
|---|---|---|---|---|
| **基准 `stock`** | 无叠加的 stock trunk（`Block`，MHA / 无 carry / 无窗） | assoc-mqar 合成 token | 因果移位 next-token | 「什么都不加」的基底 |
| `gqa` | `stock` + GQA(2 KV 头) | 同 | 同 | KV cache 缩 8.75% 的部署形态 |
| `gqa_carry` | `+` 单共享槽 delta 写通道 | 同 | 同 | 用固定状态补长程 |
| `gqa_carry_window` | `+` 训练期可见窗口 16 | 同（**训练掩码 = 窗口 16**） | 同 | 长上下文窗口 + 搬运通道 + GQA |
| **毁源对照** | 评测时把源 KV 段 `[0, 2K)` 换成新随机符号 | 同分布、答案与上下文解耦 | 同 | 「读数是不是真来自被搬运的信息」的阴性对照 |
| **局部先验** | 只看最近 8 个 token 的最强预测器 | 同 | 同 | §19 的「不靠长程能猜多少」地板 |
| **chance** | 答案集上均匀猜 `1/64` | — | — | 下限 |

每臂都在 **native 可见性**（= 自己训练时的掩码）下评测，另跑一套**全因果交叉口径**；两套
口径的输入分布、要预测的目标、位置集完全相同，唯一差别是注意力的可见集。

## 2. 变异验证（`mutate.py` → `mutation_report.json`）

靶子 = 本实验自己的语义层 `arms.py`（引擎三特性本线不改，其活性/变异已由
`2622-eng1-composable/mutate.py` 改 `pathlm/depth_ar.py` 钉住）。逐处「故意破坏 ⇒ 单测抓红
⇒ 逐字节还原」：

| id | 破坏点 | 期望抓红的单测 |
|---|---|---|
| M1 | `native_visibility` 把目标臂的窗口吞掉（窗口臂退化为全因果） | `test_window_arm_is_nondegenerate_and_not_identity` |
| M2 | `staircase_increments` 把逐叠加步增量偷换成净和（§17.17 禁止只看净和） | `test_staircase_increments_are_adjacent_diffs` |
| M3 | `expected_block_class` 把 GQA×carry 仍报 `GQABlock`（假装 ENG1 未解除组合） | `test_each_arm_builds_and_lands_expected_block_class` |
| M4 | `native_visibility` 把窗口张到所有臂（基准臂被污染） | `test_non_window_arms_use_full_causal` |

要求：`all_caught_red == true` 且 `all_restored_bitwise == true`（原文 sha256 还原）。
结果见 `mutation_report.json`。

## 3. 前提校验（判据成立的条件，不是判据本身）

- 关联回忆 chance = 1/64 = 0.015625；局部先验（k=8）应 ≈ chance（§19 的任务构造）；
- 窗口臂：被问的 pair 的 key 距离（`query_key_distance`）应 `frac_outside_window = 1`
  （key 100% 落在窗外 ⇒ 不搬就答不出）；
- 前三条非窗口臂：`frac_outside_window` 由窗口=0 决定（全因果下「窗外」无定义，记 0）。
