# 2629-carry-ragged-projfill · 对照臂与变异验证

## 一、对照臂（每个写明：输入分布 / 要预测什么 / 部署对应）

两臂都是**非 dense RAP 形态**（逐通道门 + 随机退出 + 混合填充），唯一变量 = `carry_rule`。

| 臂 | 输入分布 | 要预测什么 | 部署对应 |
|---|---|---|---|
| `projfill`（基准） | 合成多查询 assoc-mqar（K8/Q8/gap14），逐位置退出深度 `d_i ∈ U{0..L}` | 逐位置 next-token（判决行 = 查询键行，答案是该 pair 的 value） | 「100M 出口资产配方（随机退出 + proj-fill），不开搬运通道」 |
| `projfill_carry`（目标） | **同一批** token + **同一退出模式**（seed 成对） | 同目标 | 同配方 **+** delta 搬运通道 |

- **判据是单因素对照**（目标 − 基准），**不与裸 stock（dense、无 proj-fill）比**——那会把
  形态件的代价算进 carry 的账。
- **毁源对照**：`corrupt_span(tb, 0, 16)` 把源 KV 段换成同分布新随机符号，其余（查询行/
  答案/距离）逐位不变。规则：**毁源后 CI 下界 > 4×chance ⇒ 该臂读数作废**。
- **局部先验地板**：`local_prior_accuracy(k=8)` 必须 ≈ chance（§19）。
- **并列口径**：`full_depth_deployment`（退出深度恒 = L）——它是部署全栈模式，但训练时
  从未见过全深度 ⇒ 严格 OOD，只作并列参考；主腿是 `in_distribution_random_exit`。

## 二、变异验证（`mutate.py` → `mutation_report.json`）

靶子 = 本实验自己的语义层 `ragged_projfill_arms.py`（**不改**任何共享引擎文件，SCHEMA §22）。
六处破坏点各对准一条单测，**故意破坏 ⇒ 抓红 ⇒ 逐字节还原**：

| id | 破坏点 | 期望抓红的下游测试 |
|---|---|---|
| M1 | `train_flags`：proj-fill 形态件被吞掉 | `test_train_flags_share_recipe_and_add_carry` |
| M2 | `model_config_kwargs`：carry 因子被吞掉 | `test_only_carry_varies_between_arms` |
| M3 | `single_factor_increment`：增量符号反转 | `test_single_factor_increment_arithmetic` |
| M4 | `native_visibility`：伪造训练期窗口 | `test_native_visibility_zero_for_both_arms` |
| M5 | `expected_block_class`：carry 臂仍报 `Block` | `test_expected_block_class_per_arm` |
| M6 | `destroy_source_valid`：毁源判据反转 | `test_destroy_source_validity_rule` |

实测：**6/6 抓红、6/6 逐字节还原**（`mutation_report.json::all_caught_red/all_restored_bitwise`）。

## 三、能力面回归钉（`tests/test_carry_ragged_projfill_13m.py`）

三条引擎依据（§22.8）：
1. `carry_rule` × 非 dense × 随机退出 × proj-fill **可建可前向**、通道遥测进 `aux`（ENG2 解锁）；
2. `train_window > 0` 与 `dense=False` **仍硬互斥**（本线没有训练期窗口的依据）；
3. `proj_fill` 要求 `fill_kv=True`（非 dense 前向显式校验）。

引擎自身的变异覆盖另有其人（`tests/test_depth_ar.py` §ENG2 / `2628-eng2-carry-ragged/mutate.py`）。
