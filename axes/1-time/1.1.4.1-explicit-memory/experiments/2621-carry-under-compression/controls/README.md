# controls · 2621 的对照臂与方差界（SCHEMA §3 的 `controls/`）

本实验唯一变量 = **评测期注意力 K/V 的存储精度**。对照体系如下。

## 主对照（诚实对手）

| 角色 | 是什么 | 在哪 |
|---|---|---|
| **基线** | `fp16` KV（工业默认「全精度」口径，与 2614/2619/2620 同基线） | `results.json::main_leg.fp16` |
| 实现等值锚 | `fp32` 释放版前向（`quantize_kv` 恒等）——必须与模型自带前向**逐位相等** | `results.json::main_leg.fp32`、`selfchecks` |
| **处理臂** | `int8` KV（对称 per-channel） | `results.json::main_leg.int8` |

三臂加载**同一检查点**（sha256 逐次断言）、同任务、同位置集、同 seed（逐字段对照见
`README.md §6`）。

## 前提与护栏对照

| 对照 | 作用 | 读数 |
|---|---|---|
| chance | 均匀猜答案集（1/64 = 1.5625%） | `main_leg[*].chance` |
| 局部先验地板 | 只看最近 8 个 token 的最强预测器（3.125%） | `main_leg[*].local_prior` |
| **毁源对照臂** | 把源片段 `[0, 16)` 换随机符号（判据 ③） | `main_leg[*].corrupt_source`（三臂均 0.73% < chance） |
| 查询-key 距离分布 | 证明 key 100% 落在窗口外（需求真的存在） | `main_leg[*].query_key_distance.frac_outside_window = 1.0` |
| `2618` U0（无通道） | 「不搬就真答不出」的历史对照（ticket 35） | `../2618-carry-channel-13m/results.json`（0.0137 ≈ chance） |

## 方差界

- **bpc**：`0.014`（`harness/methodology.md`）——`|Δbpc| < 0.014` 标「⚠不可排序」。
  本实验 `int8 − fp16 = −0.0025` ⇒ 不可排序（测不出损伤）。
- **召回**：Wilson 95% CI；「崩」的判据线 = `Δ < −3pp` **且** CI 不重叠。
- **复现性**：同 seed 重跑 `fp16`/`int8` 主腿各一遍，`n_hits` 完全相同（`max_abs_diff = 0.0`）。

## 未纳入（显式声明）

- **GQA / MQA 头因子**：引擎硬互斥、本实验未测（见 `README.md §4b` 与 `analysis.md §4b`）；
  后续有界任务 `ENG1`。
- **`int4` 档**：2614 已知有明显损伤，叠加搬运通道未测（见 `analysis.md §9`）。
