# 2631 · 因果口径 KV 预算图（27 点）—— 实验契约

> **补录声明（重要）**：本实验在 2026-10-02 的 48 波次中被模型网关鉴权故障（401）中断，
> **运行已跑完、契约文件未落盘**。本文件于运行结束后**补齐目录契约的形状**（SCHEMA §3/§6 的
> 九节），内容逐字来源如下，**不新增、不调整任何判据**：
> ① `config.json`（运行前写入，mtime 早于 `results.json`）的 `causal_caliber` /
> `criterion` / `eval` / `controls` / `checkpoints` / `cost` / `reproducibility` 字段；
> ② `2621-kv-budget-rest/README.md` §4 的 V1 判据线（本实验逐字沿用，`config.json::contract_copy` 记明）；
> ③ `docs/REMEDIATION.md` §4 R1 与 `.scratch/49-adversarial-review/REPORT.md` A2（外部依据）。
> **判据线在 `config.json::causal_caliber.preregistered_criteria`，以那份为准**；本文件只是它的落盘副本。
>
> 入口：`run.py`（纯推理 + 汇总，**不训练**；`--charts-only` 可纯 CPU 重画表/图）；
> 语义层：`causal_check.py`（前缀不变性仪器 + 新旧并列）、`budget_caliber.py`（值 + 标尺的部署字节）；
> 单测：本目录 `test_causal_check.py` 与仓库 `tests/test_kv_budget_causal.py`；
> 变异：`mutate.py` → `mutation_report.json`（8/8 抓红 + 逐字节还原）。
> 判读：同目录 `analysis.md`；表：`table.md`；结果：`results.json`；日志：`run_log.txt`、`charts_only_rerun.runlog`。

---

## 1. 命题

把 `2621` 的 **27 点 KV 预算图**在 L1 修复后的**逐请求、因果/流式** KV 量化下**重测**，
回答三个可证伪的问题（`config.json::causal_caliber.preregistered_criteria`）：

1. 前缀不变性是否成立（改未来 token / 改同批邻居，过去位置的反量化值与过去 logits 不变）？
2. 逐臂的 Δbpc_因果 = bpc_新 − bpc_旧 的方向：旧（非因果）口径是**夸大**还是**缩小**了 int8/int4 的质量代价？
3. 头共享 × 量化的 Pareto 排序（同一「每位置字节」轴上）在新旧口径下是否改变？

附加判据 ④：若结论反转（如 int8 不再近无损 / int4 代价跨过方差带），如实报并**建议推翻旧价签**。

## 2. 单因素声明

- 被测模型是**主流标准 decoder-only 变压器**（13.5M dense：12 层 × d256 × 8 头 × mlp6 × seq256，
  tied 读出、仅最终深度 CE），结构不改、参数不增（`params_added = 0`）、不训练。
- **唯一改动的一个因素** = KV int8/int4 的**标尺约定**：`沿 batch+token 全序列取 amax` →
  `逐请求、因果、流式`（`tools/evalkit/kv_quant.py::quantize_kv` 的 `cummax` 版，L1 的 F1 修复，
  **非本实验代码**）。checkpoint / 数据流 / 窗口序 / 任务 / 判据 / 27 个臂的
  `(kind, mode, window, anchors, value_bytes, positions_kept)` 与 `2621` 逐字段相同
  （`config.json::causal_caliber.unchanged_relative_to_2621`）。
- **因果性是本实验自带的第二个观测对象**：严格前缀确定要付「每值一个 fp16 标尺」的钱
  ⇒ 部署口径的字节必须另算（`budget_caliber.py`；判读见 `analysis.md` §4）。
- 组合声明：本表里的组合臂（量化 × 窗口/锚点、量化 × 头共享、头共享 × 窗口）由 `2621` 的契约
  授权（SCHEMA §4 的 2026-10-02 例外条款），本实验只是**换口径重测**它们，没有新增组合。

## 3. 指标表（多维，逐项分析）

| 维度 | 指标 | 数据来源 |
|---|---|---|
| 横轴（判据 C1，唯一口径） | `bytes_per_position` = `2(K,V) · n_layers · n_kv_heads_eff · head_dim · value_bytes · (positions_kept / T)`，T = 256 | 解析计算（`gaps_arms`/`rest_arms`，与 2619/2620/2621 逐字同一式） |
| 横轴（新增，部署口径） | 值 + **标尺**：(a) 逐值标尺 = 值 + 2 B/值；(b) 分块标尺 = 值 + 2/G B/值（G = 32，另给 8/16/64/128 敏感性） | `budget_caliber.py`（与判据 C1 的乘子对账；不通过直接报错） |
| 门禁（判据 ①） | 前缀不变性两条：未来 token 不改过去 logits；同批邻居不改本请求 logits | `causal_check.py`（对**生产**量化器检验；旧实现会被抓红） |
| 代价（质量腿 1） | clean bpc（越低越好）+ `damage_bpc` = 本臂 − 本族基线 + Δbpc_因果 | 本实验同进程、同一路径重测（`2619.bpc_of`） |
| 代价（质量腿 2） | 合成长程召回 copy/induction/var_recall（越高越好）+ Wilson 95% + 逐源簇 | `tools/evalkit/longrange_tasks.py` |
| 参照系 | chance = 1/64 = 0.0156；局部先验（最近 8 token）= 0.0146；**逐臂毁源对照** | 同仪器内置 |
| 收益（内存） | 相对全精度基线的每位置字节（**解析**；实测显存单列） | 解析公式（值字节列 + 部署列） |
| 新旧对照 | Δbpc(新−旧) 逐臂 + 方向标签；非量化臂必须逐位不变 | `causal_check.compare_causal_vs_noncausal`（对 `2621/results.json`） |
| 可控性 | ① 仪器自检（本路径 vs 既有仪器逐位）；② 与来源实验逐位对照；③ `--charts-only` 可重画 | `results.json::instrument` / `cross_check_vs_source` |
| 活性 | 量化：`n_values` / `changed_frac` / `rel_err` / `scale_min` / `scale_max`；掩码：挡掉的因果对数；头共享：真实 K/V 张量形状 | `results.json::points[*].activity` |

## 4. 判据线（预先写下，不许事后调整）

**逐字抄自 `config.json::causal_caliber.preregistered_criteria`（运行前写入）：**

1. **① 前缀不变性必须成立**：改未来 token / 改同批邻居，不改过去位置的反量化值与过去 logits；
2. **② 逐臂报 Δbpc_因果 = bpc_新 − bpc_旧**，并明确旧（非因果）数字是**夸大了**还是**缩小了**
   int8/int4 的质量代价；
3. **③ 头共享 × 量化的 Pareto**：在每位置字节轴上，GQA/MQA×int8/int4 相对纯量化/纯头共享的排序，
   新旧口径是否改变；
4. **④ 若结论反转**（如 int8 不再近无损 / int4 代价跨过方差带），如实报并建议推翻旧价签。

**门禁**：① 不成立 ⇒ 本实验作废（实现未修好），不得作为新价签。
**方差带**：Δ < 0.014 不可排序（`config.json::criterion.variance_band_bpc`，逐字沿用 V1）。
**禁外推**：单点只描述；窗口/锚点臂的 `kept/T` 与 T 有关 ⇒ 不得外推到别的 T（`config.json::no_single_point_extrapolation`）。
**召回地板规则**：`recall ≤ chance + 2σ_wilson(chance)` ⇒ 记 `at_floor = true`（读作「该任务上不可分辨」，
不得写成「机制无效」，SCHEMA §19）。

## 5. 对照与方差

- **对照臂**：本族全精度基线（trunk 族 = `fp16`；head / 量化×头共享族 = `mha`）；
  同拓扑 fp16 臂（`config.json::quant_increment_ref`：gqa/mqa/win32/win64/win16_anchor16/win32_anchor32）；
  chance / 局部先验 / 逐臂毁源；**非量化臂在两条口径下逐位不变**（这才是「唯一因素」的证据）。
- **seed**：`config.json::seed = 7`（评测窗口序）；模型是冻结 checkpoint（单 seed 训练，
  `2610`/`2616` 各一次）⇒ 训练侧散布未知，本实验**不**据此宣称结构优劣。
- **方差界**：bpc Δ < 0.014 不可排序（预登记）；召回的 Wilson CI **按池化计数**，复核 B2 指出
  同一序列的多次查询不是独立二项样本 ⇒ CI 偏窄（实测 2.5–15.7 倍），`analysis.md` §6 已如实量化
  并改用逐源簇配对读法。
- **重跑**：`config.json::reproducibility` = 纯推理无随机采样，同 seed 重跑应逐位相同，
  阈值 `max_abs_delta_bpc ≤ 0.005`；本波实际用 `--charts-only` 重画（纯 CPU）验证可重生成。

## 6. 成本

- **GPU 墙钟预算：20 min**（`config.json::cost.gpu_wall_budget_min`，来源契约 W1「纯 eval ≈20 min」；
  SCHEMA §8 要求契约里只有一个数字）。**实测 2.288 min**（`run_log.txt`，27 点全跑 + 两腿）。
- **显存预算 ≤ 5 GiB**；实测评测期峰值 **462.5 MiB**（`torch.cuda.max_memory_allocated`）。
- **预算不够时砍什么**（预登记）：先砍 `var_recall`，再砍 `induction`，保 `copy` 主腿与全部预算点；
  **不砍判据线**。
- **冒烟**：`python3 run.py --smoke`（缩批量/档数，实测步时与显存，不外推时间）；产物在 `smoke/`。
- **可中断性**：`--charts-only` 可只从 `results.json` 重画表/图（纯 CPU，不取 GPU 锁）。
- **共享卡纪律**：GPU 任务串行（`/tmp/pathlm_gpu.lock`），按臂释放、臂间 ≥60 s。本实验的评测段
  已完成（实测 2.288 min、峰值 462.5 MiB，27 点全部落盘）；本波同批的其它 GPU 重跑线未执行。

## 7. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | `data/enwik8_full.npz`（enwik8 full，前 90M 训练 / 末 10M 评测；本实验只用评测尾段跑 clean bpc）；**不入库** |
| 合成任务 | `tools/evalkit/longrange_tasks.py`（copy / induction / var_recall，符号词表 64、copy_len 32、src_positions 0/96/191）——SCHEMA §19：不用 enwik8 当长程主任务 |
| 入口 | `python3 run.py`（全跑）／`python3 run.py --smoke`／`python3 run.py --charts-only` |
| resolved config | `config.json`（臂定义、横轴因素、判据、预算、checkpoint 哈希全在内） |
| 结果 | `results.json`（27 点 + 仪器自检 + 对照 + 新旧并列 + 来源对照 + 部署口径 + 代码哈希） |
| 表 / 图 | `table.md`（主表 + 表 2/3/4/5）、`chart.txt`、`chart.png` |
| 判读 | `analysis.md`（人话摘要 + 逐点表 + 部署口径答复 + 三道对账 + 统计口径） |
| 权重清单 | `MODEL.md`（4 个冻结 checkpoint 的哈希/体积/生成命令；权重不入库） |
| 单测 / 变异 | `test_causal_check.py`、`tests/test_kv_budget_causal.py`、`mutate.py` → `mutation_report.json` |
| 日志 | `run_log.txt`（27 点逐行读数）、`charts_only_rerun.runlog`（补列后重画） |

## 8. 失败决策树

按此顺序排除（前一档不成立才进下一档）：

1. **测量**：前缀不变性两条若为 False ⇒ 先怀疑量化器/仪器（判据 ① 的门禁，`--smoke` 复检）；
   非量化臂若不再逐位不变 ⇒ 横轴/数据流被动了，本实验作废。
2. **实现**：`budget_caliber.deployment_bytes` 与判据 C1 的乘子对不上 ⇒ 直接报错（不许出数）；
   变异未抓红 ⇒ 单测无效，先修单测。
3. **规模**：只有 13.5M / 单 T=256 / 单 seed ⇒ 结论只作**这些点**的描述，不外推。
4. **设计**：若 (b)（分块标尺）被批准而质量代价超带 ⇒ 结论为「KV 量化在本规模上不是省内存手段」，
   回到头共享/窗口两条已测杠杆。

## 9. 产出

`results.json` / `run_log.txt` / `charts_only_rerun.runlog` / `table.md` / `chart.txt` / `chart.png` /
`analysis.md` / `MODEL.md` / `mutation_report.json` / `README.md`（本文件）。
权重与数据不入库（`.gitignore`）。
