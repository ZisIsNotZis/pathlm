# 2624 · 对照臂 / 方差界 / 变异验证 —— 说明

> 契约：`../README.md`；判读：`../analysis.md`；原始结果：`../results.json`；
> 对账：`../reconcile_report.json`（550 项，全过）；变异：`../mutation_report.json`。

## 1. 对照臂清单（本线共 7 臂）

| 角色 | 臂 | 与谁比 | 唯一差别 | 来源 |
|---|---|---|---|---|
| **基准**（恒为 stock，§15/§17.14） | `stock` | — | — | 2623 |
| ① **容量括号下端**（MHA） | `mha_mlp5` | `stock` | `mlp_mult` 6 → 5（−11.65% 参数） | 本线自训 6k |
| ① **被检验点** | `gqa` | 插值到 12.35M 的 MHA | `n_kv_heads` 8 → 2 | 2623 |
| ② **训练布局（带伪影）** | `stock`/`gqa`/`gqa_carry`/`gqa_carry_window` | 各自的 Q=7 读数 | **评测口径** Q=8 → 7 | 2623 |
| ③ **6k 点（无 carry / 有 carry）** | `gqa` / `gqa_carry` | 彼此 | `carry_rule` None → delta | 2623 |
| ③ **10k 点（无 carry / 有 carry）** | `gqa_10k` / `gqa_carry_10k` | 彼此 & 各自的 6k 版 | `carry_rule` None → delta（对）；`steps` 6000 → 10000（对间） | 本线自训 10k |

**对照可比性（§13）三要素**（每条腿都满足）：输入分布 = 同一生成器同一 seed 的同一批；
预测目标 = 同一位置集上的同一个下一 token；部署对应 = 同一个模型在同一个任务上的检索能力。
**同一条腿内各臂的 n / chance / local_prior 必须相等**（`reconcile_report.json` 的
`task.*.same_n / same_chance / same_local_prior` 逐条钉住）。

## 2. 方差界

| 量 | 带 | 来源 |
|---|---|---|
| 准确率差（pp） | **1.4pp** | `docs/methodology.md`；本线 `points.PP_BAND` |
| clean bpc | **0.014** | 同上；本线 `points.BPC_BAND` |
| 逐臂二项 CI | Wilson 95% | `tools/evalkit/carry_channel.py::wilson_ci` |
| 两臂比较 | `two_proportion`（pooled z + Newcombe 差值 CI） | 同上 |

**判读纪律**：差值 < 方差带 ⇒ 写「不可排序」，**不得**写「更好/更差」；且必须给
§17.17 的**分量分解**（对基准质量 + 对目标量两列，见 `analysis.md` §5.3）。

## 3. 毁源对照（每臂每条腿都跑）

- 构造：`longrange_tasks.corrupt_span(tb, 0, 2K)`——把源 KV 段 `[0, 2K)` 换成同分布新随机
  符号，**查询行/答案/距离逐位不变**（种子 `seed_base + i`，与主批同批）。
- 判据：毁源后 CI 下界 > 4×chance ⇒ 该臂读数作废。
- 本线结果：**21/21 通过**（3 条腿 × 7 臂）；分布内腿还额外做了**逐查询块**的毁源读数
  （`by_query_block_corrupt`）：Q=7 的每一块都掉到 0.39–2.93%（≈ chance 1.5625%），
  Q=8 末块从 100% 掉到 1.17% ⇒ 「越靠后越准」和「末块抬升」**都来自源信息**。

## 4. 变异验证（`mutate.py` → `mutation_report.json`）

靶子 = 本线自己的语义层 `points.py`（**不碰任何共享引擎文件**，SCHEMA §22）。
每处：改前记 sha256 → 跑指定单测（期望**红**）→ 逐字节还原 → 复查 sha256。

| id | 破坏点 | 期望被抓红的单测 | 结果 |
|---|---|---|---|
| M1 | `Q_LEAKFREE` 7 → 8（伪影条件复活） | `test_q8_last_block_is_unique_unqueried_q7_is_not` | 抓红 ✓ / 还原 ✓ |
| M2 | `unique_unqueried_last_block` 谓词取反 | 同上 | 抓红 ✓ / 还原 ✓ |
| M3 | `interp_at_target` 去掉括号检查（允许外推） | `test_interpolation_values_and_no_extrapolation` | 抓红 ✓ / 还原 ✓ |
| M4 | `capacity_slope` 退化括号不再报错 | `test_capacity_slope_rejects_degenerate_bracket` | 抓红 ✓ / 还原 ✓ |
| M5 | 臂表 `gqa_10k` 步数 10000 → 6000（③ 退化成单点） | `test_scale_pairs_same_recipe_same_ratio` | 抓红 ✓ / 还原 ✓ |

⇒ `all_caught_red=True`、`all_restored_bitwise=True`（5/5）。

## 5. 数字入账流程（§12 三道对账）

1. **算术对账**：`reconcile_numbers.py` 从 `results.json` 的**原始计数**重算
   准确率、逐块聚合、增量、容量斜率/插值/残差、bpc 差分、参数量差（不引用已四舍五入的值）。
2. **任务对账**：同腿内 n/chance/local_prior 相等；毁源同批同 span；**伪影谓词在生成的
   batch 上重算**（Q=8 全真 / Q=7 全假）；距离区间与训练分布分开。
3. **口径对账**：基准唯一（`stock`）+ ① 的插值基准显式；① 的「非逐位等参数」声明为真；
   ③ 两对同相对刻度（1.6667）；收敛门禁落在答案行 CE；活性量逐 200 步非零。
4. **`analysis.md` 逐条回填**：550 项对账里有 **164 项（82 个数字 × 2）**是
   「`analysis.md` 里逐字出现的 token → 从 token 文本解析出数值 → 与 `results.json` 比」，
   **并断言该 token 确实出现在 `analysis.md` 文本里**（抓「文本与结果不符」）。
   另有 **36 项仪器一致性**（`instr.*`）把本线与 2623 的四臂读数逐位对齐。
5. 任何一项 fail ⇒ 退出码非 0，必须先修再入账。

## 6. 未测 / 显式留白

- ① **不是逐位等参数臂**（容量括号 + 插值；原因与三个候选方案见 `../README.md §4b`）。
- ② 的**残余末块抬升机制未分离**（残余二选一消除 vs 位置/上下文效应）；
  Q=7 是「削弱伪影」而不是「完全无泄漏」，报告已按最保守口径（前 6 块）给数。
- ③ **两点都是快照**（答案行 CE 斜率仍为负），只读增量的保持性，不外推到更大预算。
- `attn_sparse_k` / `attn_linear` 与组合的交互仍受引擎互斥限制——引擎层留白，不属本线 scope。
