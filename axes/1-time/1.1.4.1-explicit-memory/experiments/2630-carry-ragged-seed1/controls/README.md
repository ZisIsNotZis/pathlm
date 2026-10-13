# 2630-carry-ragged-seed1 · 对照臂与变异验证

## 一、对照臂（每个写明：输入分布 / 要预测什么 / 部署对应）

本线是 **2629（seed=0）的同配置 seed=1 复现**。每个 seed 内的对照是**单因素**
（目标 − 基准，唯一变量 = `carry_rule`）；跨 seed 层只换 seed。两臂都是**非 dense
RAP 形态**（逐通道门 + 随机退出 + 混合填充）。

| 臂 | seed | 输入分布 | 要预测什么 | 部署对应 |
|---|---|---|---|---|
| `projfill_s1`（基准） | 1 | 合成多查询 assoc-mqar（K8/Q8/gap14），逐位置退出深度 `d_i ∈ U{0..L}` | 逐位置 next-token（判决行 = 查询键行） | 「100M 出口资产配方（随机退出 + proj-fill），不开搬运通道」 |
| `projfill_carry_s1`（目标） | 1 | **同一批** token + **同一退出模式** | 同目标 | 同配方 **+** delta 搬运通道 |
| `projfill`（复用种子臂） | 0 | 同 | 同 | 2629 基准（seed=0） |
| `projfill_carry`（复用种子臂） | 0 | 同 | 同 | 2629 目标（seed=0） |

- **判据是单因素对照**（目标 − 基准），**不与裸 stock 比**——那会把形态件的代价算进
  carry 的账。
- **跨 seed 判据**：每 seed 一个 carry 效应；判据②用**差值 CI**（统计检验），判据③用
  **1.4pp 方差带**（量级）。证据类型分开写，不混用。
- **同批位置**：数据生成器 seed 与模型 seed 无关（`recall_multi` 用 2000 基准、
  越界腿用 4000 基准、tf synth 用 3000 基准）⇒ seed=0/1 的模型在**同一批 + 同一退出模式**
  上评测（成对）；两臂亦然。
- **毁源对照**：`corrupt_span(tb, 0, 16)` 把源 KV 段换成同分布新随机符号。规则：
  **毁源后 CI 下界 > 4×chance ⇒ 该臂读数作废**。
- **局部先验地板**：`local_prior_accuracy(k=8)` 必须 ≈ chance（§19）。
- **并列口径**：全深度（`full_depth_deployment`，退出深度恒 = L）——训练时从未见过全深度
  ⇒ 严格 OOD，只作并列参考；主腿是 `in_distribution_random_exit`。
- **越界腿**：单查询，距离 241–255（> 训练上限 240，§17.18）——**外推参照，非判据**；
  本线用它检验 2629 的 +6.05pp 是否跨 seed 保持。

## 二、变异验证（`mutate.py` → `mutation_report.json`）

靶子 = 本实验自己的语义层 `ragged_seed1_arms.py`（跨 seed 统计 + W2 判据②③）。
**不改**任何共享引擎 / 训练入口文件，**不改** 2629 的语义层（SCHEMA §22）。
六处破坏点各对准一条单测，**故意破坏 ⇒ 抓红 ⇒ 逐字节还原**：

| id | 破坏点 | 期望抓红的下游测试 |
|---|---|---|
| M1 | `pp_range`：极差变成 min−max（符号反转） | `test_pp_range` |
| M2 | `sign_consistent`：恒真（不再检查符号一致性） | `test_sign_consistency` |
| M3 | `ood_effect_robust`：丢掉「每 seed 差值 CI 下界 > 0」的统计检验 | `test_ood_robust_requires_significance` |
| M4 | `indist_redundancy_stable`：方差带判据反转（超带才算稳定） | `test_indist_redundancy_band` |
| M5 | `BASE_TO_S1`：seed=0 臂名映射被调换 | `test_arm_name_mapping` |
| M6 | `pp_sd`：样本 sd 被换成总体 sd（除 n 而非 n−1） | `test_pp_sd` |

实测：**6/6 抓红、6/6 逐字节还原**（`mutation_report.json::all_caught_red/all_restored_bitwise`）。

**卫生修一处**：变异脚本首轮 M5 曾被**陈旧的 `.pyc` 掩盖**（同一秒内等长改写 ⇒ Python
的（mtime 秒级, size）失效判据命中旧字节码）。`mutate.py` 现在每次跑子进程前
**删掉靶子的 `.pyc` 并设 `PYTHONDONTWRITEBYTECODE=1`**，保证每次从源码编译。此坑记在
脚本 docstring 里，供后续复现线复用。

## 三、单测（`tests/test_carry_ragged_seed1_13m.py`）

9 条可失败单测：臂名映射、单因素定义**复用** 2629、跨 seed 统计（极差 / 样本 sd / 符号）、
判据②（必须同时「同号为正」与「每 seed 差值 CI 下界 > 0」）、判据③（方差带）、两份
`config.json` 除 seed 外逐字段相同、2629 结果可被合并。
