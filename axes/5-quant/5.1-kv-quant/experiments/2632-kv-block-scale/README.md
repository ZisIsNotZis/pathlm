# 2632 · KV 量化标尺口径重测（块标尺 / KIVI 式 vs 逐值标尺）—— 实验契约

> **契约来源**：本文件的判据线落盘自 `config.json`（**运行前**写入，mtime 早于 `results.json`）的
> `criterion` / `schemes` / `cost` / `eval` / `reuse` 字段，以及 `.scratch/51-tree/WORK_ITEMS.md`
> 的 WI-08 与独立批判性审计（KV 量化把标尺做成逐值、据此得出与业界相反的内存结论）。
> 口径定义与判据线以 `config.json::criterion` 为准（下同）；本文件只是它的落盘副本。
>
> 入口：`run.py`（纯推理 + 汇总，**不训练**；`--charts-only` 可纯 CPU 重画表）；
> 语义层：`block_arms.py`（标尺方案感知的前向）、`scale_caliber.py`（值 + 标尺 + staging 的字节账）；
> 标尺算子：`tools/evalkit/kv_quant.py`（新增 `scheme` 分支，默认 `per_value` 向后兼容）；
> 单测：`tests/test_kv_block_scale.py`；变异：`mutate.py` → `mutation_report.json`。
> 判读：同目录 `analysis.md`；表：`table.md`；结果：`results.json`；日志：`run_log.txt`。

---

## 1. 命题

把 `2631` 的 **27 点 KV 预算图**在**块标尺**（标尺沿 `head_dim` 每 `G` 个值摊一把，`G=32` 为主）
下重测，并把 KV 量化的字节账从「值字节」改成**部署字节（值 + 标尺）**；回答可证伪的：

1. **前缀不变性**（`block` 沿 `head_dim`）是否成立——改未来 token / 改同批邻居，过去位置的读数不变？
2. **「8 位近无损且省约一半」这条参照线成立吗？** 在块标尺下每位置字节对 fp16 的倍率、质量 Δbpc 各是多少？
3. **在什么实现下不成立？** 逐值标尺（`2631` 的生产实现）与沿 token 轴的块（`block_tok` / KIVI 的 K 侧）各付什么代价？

## 2. 单因素声明

- 被测模型是**主流标准 decoder-only 变压器**（13.5M dense：12 层 × d256 × 8 头 × mlp6 × seq256，
  tied 读出、仅最终深度 CE），结构不改、参数不增（`params_added = 0`）、不训练，复用 2610/2616 的冻结
  checkpoint（`checkpoints.*.sha256` 运行时断言）。
- **唯一改动的一个因素** = KV int8/int4 的**标尺方案**：`2631` 的逐值标尺（`cummax` 版）→ 本实验的
  块标尺（沿 `head_dim`，`G=32` 为主）。checkpoint / 数据流 / 窗口序 / 任务 / 判据 / 27 个臂的
  `(kind, mode, window, anchors, value_bytes, positions_kept)` 与 `2631` 逐字段相同。
- **实现层 equality（SCHEMA §9）**：本实验 `scheme="per_value"` 的前向与 `2631` 的生产前向
  **逐位一致**（`results.json::instrument.scheme_forward_equality_vs_2631_per_value` 三条 True）；
  17 个非量化臂与 `2631` 逐位一致（`max|Δbpc| = 0`）——这是「唯一改动的因子只有标尺方案」的证据。
- 组合声明：组合臂（量化 × 窗口/锚点、量化 × 头共享）由 `2621` 的契约授权（SCHEMA §4 例外条款），
  本实验只是**换标尺口径重测**它们，没有新增组合。

## 3. 指标表（多维，逐项分析）

| 维度 | 指标 | 数据来源 |
|---|---|---|
| 横轴（值字节，对照） | `bytes_per_position` = `2·L·n_kv_heads_eff·head_dim·value_bytes·(kept/T)` | `2619` 判据 C1（逐字同一式，与 `2631` 逐位可比） |
| 横轴（**部署字节，本实验主口径**） | `2·L·n_kv_heads_eff·head_dim·(value_bytes + scale_bytes_per_value)·(kept/T)` | `scale_caliber.deployment_bytes`（与 `kv_quant.quantize_kv(scheme=...)` 一一对应，值字节部分逐点对账） |
| 门禁（判据 ①） | 前缀不变性两条（未来 token / 同批邻居） | `block_arms`（对生产前向检验）；`block` 必须 True，`block_tok`/`kivi` 的 K 侧如实报 False |
| 代价（质量腿 1） | clean bpc + `damage_bpc` = 本臂 − 本族基线；量化增量 = 本臂 − 同拓扑 fp16 | 同进程、同一路径重测（`b21.b19.bpc_of`） |
| 代价（质量腿 2） | 合成长程召回 copy/induction/var_recall + Wilson 95% + 逐源簇 | `tools/evalkit/longrange_tasks.py` |
| 参照系 | chance = 1/64 = 0.0156；局部先验（最近 8 token）= 0.0146；逐臂毁源对照 | 同仪器内置 |
| 收益（内存） | 每位置部署字节对 fp16 的倍率（**解析**）；实测显存单列；同价对手 fp8 解析（未测质量） | `scale_caliber` / `results.json::memory_convention` |
| **本格自测方差 / 排序带** | **在力带 = 单臂跨窗口 bpc 极差 0.036324**（标准口径 = 配对 Δ、≥5 窗口；本批 3 窗口 ⇒ 不满足）。3 点配对量（均值 +4.1e-05）只作描述 | `results.json::ranking_band` / `variance_self_measured[_paired]` |
| 活性 | 量化：`n_values` / **`n_scales`（== values/G）** / `changed_frac` / `rel_err` / `scale_min` / `scale_max`；掩码：挡掉的因果对数；头共享：真实 K/V 形状 | `results.json::points[*].activity` |
| 可控性 | ① 仪器自检（本路径 vs 既有仪器逐位）；② 非量化臂与 `2631` 逐位；③ `--charts-only` 可重画；④ §20 取锁/释放与前后空闲显存（机器可读） | `results.json::instrument` / `reuse` / `gpu_lock` |

## 4. 判据线（预先写下，不许事后调整）

**逐字抄自 `config.json::criterion`（运行前写入）：**

1. **① 前缀不变性必须成立**（主臂 `block(G=32)` 与 V per-token）：改未来 token / 改同批邻居，不改过去
   位置的读数；对违反的实现（`block_tok` / `kivi` 的 K 侧）如实报 False。
2. **② 诚实对照**：逐值标尺臂（`per_value`）与不量化臂（`fp16` / 同拓扑 fp16）必须同场。
3. **③ 部署字节账**：值 + 标尺（+ 非因果方案的 staging），不得再把标尺当免费。
4. **④ 质量腿**：clean bpc + 合成长程召回（Wilson CI）；**内存腿**：解析字节账 + 实测峰值。
5. **⑤ 结论必须回答**：『8 位近无损且省约一半』这条参照线成立吗？成立到什么程度？在什么实现下不成立？

**门禁**：①（主臂 `block G=32`）不成立 ⇒ 本实验作废，不得作为新价签。
**方差带（预登记口径 + 在力声明）**：**标准口径** = 同一对照臂对在多个评测窗口重测的**配对 Δ** 区间，
要求 **≥5 个窗口**；**本批只跑了 3 个（<5）⇒ 不满足**，故**在力排序带 = 单臂跨窗口 bpc 极差 0.036324**
（更保守）。本批所有「可排序 / 不可排序」判断一律用它；3 点配对量（均值 +4.1e-05、区间
[+9e-06, +1.01e-04]）只作描述，**不当带**（主表窗口的 Δ 恰为该 3 点 max ⇒ 当带会循环、不可证伪）。
**禁外推**：单点只描述；G 阶梯与 scheme 比较各点单独测（`config.json::criterion.no_single_point_extrapolation`）。
**召回地板规则**：`recall ≤ chance + 2σ_wilson(chance)` ⇒ `at_floor = true`（读作「该任务上不可分辨」，
不得写成「机制无效」，SCHEMA §19）。
**符号约定**：`damage_bpc = bpc_实验 − bpc_对照`（正 = 更差）。

## 5. 既有做法（prior art，SCHEMA §18）

| # | 谁做过 | 为什么这么设计 | 实测结果 | 我们的增量 |
|---|---|---|---|---|
| 1 | **llama.cpp 的 `q8_0` / `q4_0` 块量化**（KV cache 量化用）：每个 32 个连续值的块共享 1 个 fp16 scale，`scale = 块内 amax / qmax` | 省存储/带宽；块沿行的连续维摊薄 scale，使每值开销从 2 B 降到 `2/32` B | 生产系统默认做法；`memory（未核）` | **本地实测**：在本仓 13.5M 模型上量它对 bpc/召回的实际影响，并**证明它沿 head_dim 摊薄仍严格前缀确定** |
| 2 | **KIVI**（Liu et al., 2024, "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"）：**K cache per-channel、V cache per-token** 量化；沿 token 分组 + 残差 fp16 窗口 | K 有通道离群值（per-channel 保留分辨率）、V 的离群值在 token 上（per-token 更合适） | 论文口径 2-bit 近无损；`记忆·未核`（本机联网不可用，未核对原文/代码） | **本地实测**：KIVI 式在本仓的质量与**前缀不变性的代价**（K 侧沿 token 分组 ⇒ 块内过去值被后续位置改写） |
| 3 | `tools/evalkit/kv_quant.py`（本仓）的**逐值标尺**（`cummax` 沿 token 轴） | 「过去位置严格前缀确定」——位置 t 的 scale 只由 ≤ t 决定 | **本地实测**（`2631`）：每值 +2 B 标尺 ⇒ int8 净 3 B/值 = fp16 的 1.5× ⇒ 内存负收益 | **降为对照臂**：本实验证明可与「前缀确定」兼容的摊薄方式是沿 `head_dim`，不是沿 token |
| 4 | **fp8（E4M3/E5M2 式 KV 量化）**：1 B/值、浮点格式无显式标尺 | 8 位里存储最省的通用做法 | `记忆·未核`；本实验**未测质量** | **本地解析对照**：fp8 = 1.0 B/值 = fp16 的 ×0.50，**比 int8+块标尺（1.0625 B/值、×0.53）更省** ⇒ 界定「不能声称 int8+块标尺是最优 8 位方案」 |

**撤回**：`2631/budget_caliber.py::mixed_axis_note` 的「K 侧做不到非 token 轴摊薄」——本实验证明
K 沿 `head_dim` 分块（`block`）可摊到 `2/G` B/值**且保持严格因果**（详见 `analysis.md` §9）。

**检索说明**：本机联网不可用 ⇒ 第 1/2 条为 `记忆·未核`，本文件**只把它们当作设计方向与对照口径**，
不当判据；未核对的量化边界值（如 2-bit）不进入本实验的结论句。

## 6. 对照与方差

- **诚实对照（判据 ②）**：`per_value`（逐值标尺）与 `fp16` / 同拓扑 fp16 臂**同场**；同表还并列
  `2619/2621/2631` 的窗口 / 头共享族。
- **seed**：`config.json::seed = 7`（评测窗口序）；模型是冻结 checkpoint（单 seed 训练）⇒ 训练侧散布
  未知，本实验**不**据此宣称结构优劣。
- **自测方差与排序带**：标准口径 = 配对 Δ、**≥5 个窗口**；**本批只跑了 3 个（<5）⇒ 不满足**，
  故在力排序带 = **单臂跨窗口 bpc 极差 0.036324**（见 `table.md` 表 5 与 `results.json::ranking_band`）。
  3 点配对量只作描述（均值 +4.1e-05；**不以 max|Δ| 当带**）。召回的 Wilson CI 按池化计数
  （`2631` 复核 B2 已指出其偏窄，本文件沿用并把配对读法写进判读）。
- **重跑**：纯推理无随机采样，同 seed 重跑应逐位相同；`--charts-only` 可纯 CPU 重生成表。

## 7. 成本

- **GPU 墙钟预算：40 min**（`config.json::cost.gpu_wall_budget_min`，来源契约 WI-08）；
  GPU 工作实测见 `results.json::cost`（共享卡被外部占用时同代码步时变长；wall 从取锁成功后起算）。
- **显存预算 ≤ 5 GiB**；实测**峰值 1798.6 MiB**（`torch.cuda.max_memory_allocated`）。
- **§20 共享卡记账（机器可读）**：`results.json::gpu_lock` = `holder_pid` / `wait_seconds` /
  `free_mib_before` / `free_mib_after` / `release_ok` / `held_whole_run`。本实验整程 <10 min ⇒
  SCHEMA §20.1 允许整程持锁；峰值 > 1 GiB 不满足免锁例外，故必须留此记录。
  **读法**：`free_mib_before/after` 是 **nvidia-smi 的全卡空闲**读数（含**外部占用**），
  差值**不等于**本 run 足迹——本 run 足迹看 `peak_mem_mib_this_run`（= `cost.peak_mem_mib`）。
- **预算不够时砍什么**（预登记）：先砍 scheme 比较臂（保留 `per_value`/`block g16`/`block g32`/
  `block_tok g32`/`kivi`），再砍自测方差到 2 个 seed，保全部 27 点与判据线；**不砍判据线**。
- **可中断性**：`--charts-only` 只从 `results.json` 重画表（纯 CPU，不取 GPU 锁）。

## 8. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | `data/enwik8_full.npz`（enwik8 full，末 10M 评测尾段跑 clean bpc）；**不入库** |
| 合成任务 | `tools/evalkit/longrange_tasks.py`（copy / induction / var_recall）——SCHEMA §19 不用 enwik8 当长程主任务 |
| 入口 | `python3 run.py`（全跑）／`--smoke`／`--charts-only` |
| resolved config | `config.json`（继承 `2631` 的 arm/checkpoint/eval/task；本实验覆盖 contract/criterion/schemes） |
| 结果 | `results.json`（27 点 + 13 方案行 + 仪器自检 + 自测方差 + 部署口径 + 代码哈希） |
| 表 | `table.md`（表 1–5） |
| 判读 | `analysis.md`（人话摘要 + 正面对照表 + 三道对账 + 失败与局限） |
| 权重清单 | `MODEL.md`（冻结 checkpoint 哈希/体积/生成命令；权重不入库） |
| 单测 / 变异 | `tests/test_kv_block_scale.py`、`mutate.py` → `mutation_report.json`（8/8 抓红 + 逐字节还原） |
| 日志 | `run_log.txt`（逐行读数） |

## 9. 失败决策树

按此顺序排除（前一档不成立才进下一档）：

1. **测量**：主臂 `block` 前缀不变性若为 False ⇒ 先怀疑标尺归约轴（`--smoke` 复检）；非量化臂若不再
   逐位不变 ⇒ 横轴/数据流被动了，本实验作废。
2. **实现**：`scale_caliber.deployment_bytes` 的值字节与判据 C1 对不上 ⇒ 直接报错（不许出数）；
   变异未抓红 ⇒ 单测无效，先修单测。
3. **规模**：只有 13.5M / 单 T=256 / 单 seed / `head_dim=32` ⇒ 结论只作**这些点**的描述，不外推；
   `head_dim` 更大时 `G=32` 只是一行的一小段（分辨率更高），本实验不能外推该效应。
4. **设计**：若块标尺下 int4 的质量代价超自测带宽 ⇒ 结论为「int4 在本规模上有损」，回退到 int8 /
   头共享 / 窗口三条已测杠杆。

## 10. 产出

`results.json` / `run_log.txt` / `table.md` / `analysis.md` / `MODEL.md` / `mutation_report.json` /
`JOURNAL.md` / `README.md`（本文件）。权重与数据不入库（`.gitignore`）。
