# controls · 2627-carry-x-depth 对照臂索引

本文件列出本实验的对照臂、方差界与护栏，供复核者按 SCHEMA §3/§13/§14 逐条核对。

## 1. 对照臂（唯一变量 = `carry` × `depth` 两个因子的开关）

| 角色 | 臂 | 配置 | 用途 |
|---|---|---|---|
| **基准** | `base` | 两因子都关（stock dense + window16） | 所有因子分解的原点 |
| 单因子 | `carry` | + delta 搬运通道 | `carry@depth_off` |
| 单因子 | `depth` | + dense 全深度监督（`--dense-aux`） | `depth@carry_off` |
| **目标** | `both` | 两因子都开 | `carry@depth_on` / `depth@carry_on` / 交互项 |

- 四臂**同 token 预算**（`6k × 32 × 256 = 49.0M token`）、同 seed=0、同 window16、同评测协议。
- **基准唯一**：因子分解的每一项都相对它自己的对照格子（不是相对某个 oracle），符号方向
  在 §4c/`carry_x_depth_arms.py::factorial_increments` 定义一次。

## 2. 毁源对照臂（强制，判据①）

评测时把源段 `[0, 2K)`（= K 对 key/value 表）换成**同分布新随机符号**（`corrupt_span`），
查询行 / 答案 / 距离逐位不变。**判定**：任一臂毁源后 **CI 下界 > 4×chance** ⇒ 该臂读数
**作废**（说明读数不是来自被搬过去的源信息）。四臂逐臂读数见 `results.json::destroy_source_validity`。

## 3. 方差界与口径

- **方差带**：`0.014 bpc`（`docs/methodology.md`）——Δ < 0.014 不可排序；
  召回判据用 **1.4pp** 作参考带，正式用 **Wilson 95% CI** + `two_proportion` 差值 CI。
- **chance** = `1/sym_vocab` = 1/64 = 0.015625；**局部先验地板** = `local_prior_accuracy(k=8)`。
- **主判据腿在训练分布内**（§17.18）：训练型多查询 `assoc-mqar`，查询-键距离 ≤240；
  单查询腿（距离 241–255）降为**外推参照**，`results.json` 里显式标注
  `in_train_distribution: false`。

## 4. 语义开关的可失败单测 + 变异验证

- 单测：`tests/test_carry_x_depth_13m.py`（11 条）；变异：`mutate.py → mutation_report.json`。
- **变异验证（全部抓红 + 逐字节还原）**：

| id | 破坏点 | 靶子单测 |
|---|---|---|
| M1 | `train_flags` 吞掉深度因子（depth/both 不再带 `--dense-aux`） | `test_depth_factor_maps_to_dense_aux_train_flag` |
| M2 | `model_config_kwargs` 吞掉 carry 因子（四臂都变无 carry） | `test_carry_factor_is_the_only_config_variable` |
| M3 | `factorial_increments` 把 2×2 单向增量偷换成净和（§17.17 禁止） | `test_factorial_increments_are_factorial_not_staircase` |
| M4 | `native_visibility` 吞掉窗口（四臂退化全因果） | `test_native_visibility_is_window16_for_all_arms` |
| M5 | `expected_block_class` 让 carry 臂仍报 `Block` | `test_expected_block_class_per_arm` |

- **引擎侧开关**：`--dense-aux` 的语义（全深度监督，出口 CE 条数 1→13）由本实验单测
  `test_dense_aux_full_depth_supervision_is_live` 钉住；「dense 主干忽略 `exit_depths`」这条
  能力面事实由 `test_dense_trunk_ignores_exit_depths` 钉住；引擎自身的 `dense_aux` 变异覆盖
  见 `tests/test_depth_ar.py`（ticket 23c）。本线**不改**任何共享引擎文件（SCHEMA §22）。

## 5. 未测因子（显式留白，禁外推）

「ragged 随机退出 + proj-fill」因子（产出 100M `tf@4=0.9726` 的配方，`fill_kv=True`、非 dense）
与 carry **在引擎里硬互斥**（§4b/§4c），本实验**未测**；主 Agent 已立后续项 `ENG2` 跟进。
不得把本实验结论外推到「任意深度自适应 × 搬运」。
