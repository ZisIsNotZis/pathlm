# 2651 · 权重量化最小判决（含标尺字节账）—— 实验契约（**运行前**写下）

> **契约来源**：`axes/5-quant/5.2-weight-quant/project.yaml::work_packages[WP1]`（`method` / `validation` /
> `objective_criteria` / `falsification` / `milestone` 逐字照做）+ 本目录 `config.json::criterion`
> （**运行前**写入，mtime 早于 `results.json`）。
> 口径定义与判据线以 `config.json::criterion` 为准；本文件是它的落盘副本与说明。
>
> 入口：`run.py`（纯推理 + 汇总，**不训练**；`--smoke` 冒烟、`--charts-only` 纯 CPU 重画表）；
> 语义层：`wq_arms.py`（权重假量化的唯一新逻辑）；
> 单测 / 变异：`selftest_wq.py` → `mutation_report.json`；
> 判读：`analysis.md`；表：`table.md`；结果：`results.json`；权重清单：`MODEL.md`；流水：`JOURNAL.md`。

---

## 1. 命题

在 **13.5M 冻结 trunk、纯推理、单因素**的设定下回答一个可证伪的问题：

> **把权重以 int8 / int4（含分组标尺）存放，每参数部署字节（含标尺）→ 序列级两腿质量
> （clean bpc + 合成长程召回）的前沿是什么？int8 / int4 是否落在占优侧？**

三条子问题（对齐 `project.yaml::objective_criteria`）：

1. **该跌**：每参数部署字节是否真的降了（值字节 **+ 标尺**，不得把标尺当免费）？
2. **该不跌 / 可排序**：clean bpc 的代价有多大（与 fp16 的**配对 Δ** 是否越 0.014 噪声带）？
3. **该保持**：合成长程召回（第二腿）是否不塌？

## 2. 单因素声明

- 被测模型是**主流标准 decoder-only 变压器**（13.5M dense：d256 / L12 / H8 / mlp6 / seq256，
  tied 读出、仅最终深度 CE）。结构不改、参数不增（`params_added = 0`）、**不训练**，
  复用 `2610` 的冻结 checkpoint（`checkpoints.trunk.sha256` 运行时断言）。
- **唯一改动的一个因素 = 权重矩阵的 `(bits, scheme, group)`**：`{fp16, int8 per-channel,
  int8 group-32, int4 group-32}`。KV / 激活 / 偏置 / LayerNorm / tied 读出**逐位不动**
  （不施加任何 KV 量化、不施加任何激活量化）。
- **实现层 equality**：`fp16` 臂**不装任何补丁** ⇒ 前向**逐位**等于 stock 释放版；
  `int8/int4` 臂只覆盖 `nn.Linear` 的实例 `forward`（假量化往返在前向内发生），
  `module.weight` 张量本身逐位不变 ⇒ 退出即逐位还原（`results.json::instrument` 留证）。
- 与 `5.1`（KV 量化）/ `5.3`（激活量化）划界：本项只压**权重**，另两类账不动，便于分开记账。
- **不做组合实验**（权重 × KV × 激活 的组合留白是显式声明，不是遗漏）。

## 3. 臂与字节账（横轴）

| 臂 | `bits` | `scheme` | `group` | 解析每参数字节 |
|---|---|---|---|---|
| `fp16` | 16 | `identity` | — | 2 |
| `int8_per_channel` | 8 | `per_channel` | — | `1 + 2·Σn_out/Σ(n_out·n_in)` |
| `int8_group32` | 8 | `group` | 32 | `1 + 2/32 = 1.0625` |
| `int4_group32` | 4 | `group` | 32 | `0.5 + 2/32 = 0.5625` |

- **对称、零点固定 0**（与 `tools/evalkit/kv_quant.py` 同口径）：`qmax = 2^(bits−1) − 1`，
  `scale = amax/qmax`，`q = clamp(round(w/scale), −qmax, qmax)`，`ŵ = q·scale`。
- **标尺 fp16 存储 = 2 B/把**（`SCALE_BYTES = 2.0`）。标尺沿**收缩维**（`in_features`）分组。
- **`G = 32` 的理由**：① 与 `5.1` 块标尺主臂同口径，便于跨轴并账；② 整除本模型全部被量化矩阵的
  `in_features ∈ {256, 1536}`；③ 与块量化（每块 32 个连续值共享一把 scale）的常见块长一致。
- **字节账与实现一一对应**（`wq_arms.scale_count`，并在 `activity_of` 内断言）：
  `per_channel` 标尺个数 = `n_out`；`group G` 标尺个数 = `n_out·n_in/G`（要求 `G | n_in`，否则报错）。
  逐矩阵列出 `{name, shape, n_params, n_scales, value_bytes, scale_bytes, total_bytes, bytes_per_param}`。
- **本模型的行长远大于 32** ⇒ `per_channel`（每行一把标尺）**字节更省**但标尺更粗，
  `group-32` 更细但字节更多；两者都在前沿上，谁占优由质量读数决定。解析阶梯（`ladder`）
  另报 G ∈ {8,16,32,64,128,256}，回答「多小的 G 会让 int4 的标尺顶到 fp16 之上」。
- **三个口径分开报**（`config.json::axis_definition`）：`scope_linear_all`（60 个 `nn.Linear`，主口径）
  / `scope_linear_live`（48 个活矩阵）/ `scope_whole_model`（把未量化的偏置/Norm/embedding 按 fp16 计进分母）。

## 4. 指标表（多维，逐项分析）

| 维度 | 指标 | 数据来源 |
|---|---|---|
| 横轴（主口径） | `bytes_per_param` = `bits/8 + 2·n_scales/n_params`（逐矩阵 + 按参数数加权总账） | `wq_arms.byte_account` |
| 代价（质量腿 1） | clean bpc + 配对 Δbpc = `bpc_臂 − bpc_fp16`（同窗口同批数据配对） | `b21.b19.bpc_of` |
| 代价（质量腿 1，独立路径） | `retention.masked_bpc`（第二条路径，逐臂逐位对照） | `tools/evalkit/retention.py` |
| 代价（质量腿 2） | 合成长程召回 copy / induction / var_recall + Wilson 95% + `at_floor` | `tools/evalkit/longrange_tasks.py` |
| 参照系 | chance = 1/64 = 0.015625；局部先验（最近 8 token）；逐臂毁源对照 | 同仪器内置 |
| 门禁（因果性 / 预测不变） | 前缀不变性两条（改未来 token / 改同批邻居，过去与同批邻居 logits 逐位不变） | `2631/causal_check.prefix_invariance_of_logits` |
| 活性 | 每臂 `changed_frac` / `rel_err` / `mean_abs_err` / `max_abs_err` / `max_rel_err` / 标尺个数对账 + 逐矩阵 | `wq_arms.activity_of` |
| 死权重 | `gates.*` 清零后 logits 是否逐位不变（决定它只进字节账/也进质量读数） | `wq_arms.dead_gate_probe` |
| 方差带 | **在力带 = 0.014 bpc**（配对 Δ、5 个窗口，标准口径满足）；单臂跨窗口极差只作描述 | `results.json::band` |
| 可控性 | ① 非量化臂逐位复原；② 双 bpc 路径逐位一致；③ 逐位还原；④ 变异 9/9 抓红；⑤ §20 取锁/释放记录 | `results.json::instrument` / `gpu_lock` |

## 5. 判据线（**运行前**写下，不许事后调整）

**逐字抄自 `config.json::criterion`（运行前写入）：**

**① 该跌**：每参数部署字节（含标尺；int8 ≈1 B/参数、int4 ≈0.5 B/参数 + 标尺）。
**② 该不跌 / 可排序**：clean bpc（须越 0.014 噪声带）。
**③ 该保持**：合成长程召回（第二腿不塌）。
**④ 必须不变**：因果性与非量化臂逐位不变；**必须报**：标尺/零点开销与分组大小。
**⑤ 否证**：若权重量化的 clean bpc 代价越噪声带且吃掉全部内存收益（每参数字节相近的点位上不占优）
⇒ 判该档在本尺度不成立；若分组标尺的字节开销使 int4 反占内存（每参数字节 ≥ fp16）⇒ 判该实现在该
分组下不可用；若非量化臂不能逐位还原 ⇒ 判实现有误（对照失效）。

**噪声带（预登记口径 + 在力声明，逐字照 `harness/research-schema.md` §12.1）**：
标准口径 = **配对 Δ**（同一对照臂对在同一批评测窗口重测，逐 seed 取 `bpc_臂 − bpc_fp16`），
要求 **≥5 个窗口**。**本批跑 5 个窗口**（`window_seed ∈ {2,3,4,5,6}`，每个 8×16 批量）
⇒ **满足标准 ⇒ 在力排序带 = 0.014 bpc**。判决规则 = 每个量化臂的 **`max|配对 Δbpc|` vs 0.014**：
≤ 带 ⇒ **不越带**（该臂的质量代价在该带内不可排序）；> 带 ⇒ **越带**（该臂在本尺度确有可排序的质量代价）。
另报的「单臂跨窗口 bpc 极差」是**另一口径的描述性读数**，**不参与判决、不得与 0.014 混用**。
配对 Δ 的均值 / 区间只作描述。

**召回规则**：Wilson 95%（池化计数）；`recall <= chance + 2·sigma_wilson(chance)` ⇒ `at_floor = true`
（读作「在该任务上不可分辨」，**不得**写成「机制无效」，SCHEMA §19）。**率类指标不借用 0.014 bpc 带**。

**门禁（G1–G6）**：G1 每个量化臂的干预必须活着（`changed_frac > 0`）；G2 `fp16` 臂必须与独立路径
逐位相等、且等于历史锚 1.483569；G3 字节账与实现一一对应；G4 取整误差界成立；G5 变异 9/9 抓红；
G6 `gates.*` 的死权重性质必须被逐位证明。任一不成立 ⇒ 对应结论作废。

**禁**：墙钟 / 吞吐 / 日期进判据或 `cost_model`（墙钟只作过程读数）；单点外推；跨口径混用。

## 6. 既有做法（prior art）—— **只作设计方向，不作判据**

| # | 谁做过 | 设计要点 | 我们的增量 / 划界 |
|---|---|---|---|
| 1 | **GPTQ**（Frantar et al. 2023, arXiv:2210.17323） | 逐层二阶误差补偿的 3–4 bit 训练后量化；group-wise scale 是常见配置 | **本项不重造 GPTQ**：不做 Hessian 误差补偿，只做**对称分组假量化**的受控前沿（13M、含标尺字节账、两腿质量） |
| 2 | **AWQ**（Lin et al. 2024, arXiv:2306.00978） | 按激活重要性保护显著通道（per-channel 缩放后分组量化） | **本项不重造 AWQ**：不做激活感知的通道缩放，只比 `per_channel` 与 `group-32` 两个标尺粒度 |
| 3 | **LLM.int8()**（Dettmers et al. 2022, arXiv:2208.07339） | 离群通道留 fp16（混合精度）；int8 主体 + 离群保护 | **本项不做离群保护** ⇒ 若 int4 越带，第一个嫌疑就是「未做离群保护」，见 `analysis.md` 局限 |
| 4 | **块量化**（每 G 个连续权重共享一把 fp16 scale，llama.cpp `q8_0`/`q4_0` 式） | 把标尺从逐值摊到 `2/G` B/值 | **本地实测** G=32 在本仓 13M trunk 上的质量代价，并**逐矩阵**核标尺个数（防 5.1 的 2631 那种「逐值标尺把省内存算成占内存」的口径错误） |

**检索说明**：arXiv 编号来自 `5.2-weight-quant/project.yaml::sota_baseline` 的本轮联网核实（HTTP 200）；
本文件只引用编号与一句方法方向，**不引用论文数字**，也不把它们当判据。

## 7. 对照与方差

- **诚实对照**：`fp16`（不量化）与三个量化臂**同场、同进程、同数据流、同窗口序、同任务**。
- **seed**：评测窗口序 `config.json::seed = 7`。模型是**冻结 checkpoint（单 seed 训练）**
  ⇒ 训练侧散布未知，本实验**不**据此宣称结构优劣，也不外推到别的尺度 / 别的 `in_features`。
- **配对窗口**：`window_seed ∈ {2,3,4,5,6}`（5 个窗口），每窗 8×16 批量；主表用
  `window_seed = 2` + 24×16（与历史锚同口径）。逐 seed 报该臂与 `fp16` 的配对 Δbpc。
- **双路径对账**：每臂都跑 `bpc_of`（2621/2619 路径）与 `retention.masked_bpc`（独立路径），
  要求逐位一致；不一致 ⇒ 仪器有问题、本实验作废。
- **重跑**：纯推理、无随机采样、无训练 ⇒ 同 seed 重跑应逐位相同（`--charts-only` 可纯 CPU 重画表）。

## 8. 成本

> **本轮改为 CPU 口径**（ticket 66 §13bis F5，GPU 锁被其它轴长期占用）：本节的 **GPU 墙钟预算 / §20 取锁记账条款不适用**——本 run 纯 CPU、不取 GPU 锁、无显存足迹（`results.json::caliber.is_cpu = true`、`gpu_lock` 为空）；锚比对按 `results.json::caliber` 的口径声明读（跨设备不构成判据）。

- **GPU 墙钟预算 30 min**（`config.json::cost.gpu_wall_budget_min`，来源本 WP1 ticket）；
  实测见 `results.json::cost`（wall 从取锁成功后起算；共享卡被外部占用时同代码步时更长）。
- **显存预算 ≤ 5 GiB**；实测峰值见 `results.json::cost.peak_mem_mib`（13M 模型应远低于）。
- **§20 共享卡记账（机器可读）**：`results.json::gpu_lock` =
  `holder_pid` / `wait_seconds` / `free_mib_before` / `free_mib_after` / `release_ok` /
  `held_whole_run` / `peak_mem_mib_this_run`。等锁上限 10 min（40×15 s）；等不到 ⇒ 不杀任何人的进程，
  写「锁被占，稍后重试/登记」并返回。`free_mib_before/after` 是 `nvidia-smi` 的**全卡空闲**读数
  （含**外部占用**），其差**不等于**本 run 足迹——本 run 足迹看 `peak_mem_mib_this_run`。
- **预算不够时砍什么**（预登记）：先砍配对窗口（5 → 3 个 seed；若真砍了，**必须**把在力带改成
  「单臂跨窗口 bpc 极差」并在 `config.json::criterion` 里声明两带不混用，且 0.014 只作参考），
  再砍 `var_recall` / `induction` 副腿；保 `copy` 主腿、四个臂与全部字节账；**不砍判据线**。

## 9. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | `data/enwik8_full.npz`（末 10M 评测尾段跑 clean bpc）；**不入库** |
| 合成任务 | `tools/evalkit/longrange_tasks.py`（copy / induction / var_recall）——SCHEMA §19 不用 enwik8 当长程主任务 |
| 入口 | `python3 run.py`（全跑）／`--smoke`／`--charts-only` |
| resolved config | `config.json`（`model`/`data`/`tokens`/`tasks`/`eval`/`checkpoints`/`controls`/`seed` 抄自 `2621-kv-budget-rest/config.json`，出处见 `provenance`） |
| 语义层 | `wq_arms.py`（权重假量化 + 字节账 + 活性量 + 死权重探针） |
| 结果 | `results.json`（4 臂 + 配对窗口 + 字节账逐矩阵 + 仪器自检 + §20 记账 + 代码哈希） |
| 表 | `table.md`（主表 + 字节账逐矩阵表 + 配对 Δ 表 + 召回表 + 解析阶梯） |
| 判读 | `analysis.md`（**人话摘要** + 表 + 三道对账 + 失败与局限 + 下一步） |
| 权重清单 | `MODEL.md`（冻结 checkpoint 哈希/体积；权重不入库） |
| 单测 / 变异 | `selftest_wq.py`（10 条单测 + M1..M9）→ `mutation_report.json`（9/9 抓红） |
| 日志 | `run_log.txt`（逐行读数 + 步时）、`JOURNAL.md`（过程流水，允许日期） |

## 10. 失败决策树

1. **测量**：`fp16` 臂不逐位等于 stock 锚（1.483569）或两条 bpc 路径不一致 ⇒ 先怀疑窗口序 / 数据流被动了
   ⇒ 本实验作废，不得作为新价签。**本轮为 CPU 口径**（见 §8）：与 GPU 锚的比较是**跨设备**、不构成判据，
   读法按 `results.json::caliber` 的口径声明（在力的门禁 = 两条 bpc 路径逐位相等，实测已过）。
2. **实现**：字节账与实现对不上（标尺个数 ≠ 解析式）⇒ `wq_arms.activity_of` 直接断言报错，**不许出数**。
   `selftest_wq.py` 的变异未抓红 ⇒ 单测无效，先修单测。
3. **规模**：只有 13.5M / 单 T=256 / 单 checkpoint / 单数据流 ⇒ 结论只作**这些点**的描述；
   `in_features` 更长时 `2/n_in` 更小、`2/G` 不变，本实验不外推该效应。
4. **设计**：`int4_group32` 若越 0.014 带 ⇒ 结论为「int4 在本规模有损」，回退 int8；
   `int8_*` 若越带 ⇒ 结论为「该尺度该项无收益」（在给定字节下），如实报，不辩解。

## 11. 产出

`results.json` / `run_log.txt` / `table.md` / `analysis.md` / `MODEL.md` / `mutation_report.json` /
`JOURNAL.md` / `README.md`（本文件）。权重与数据不入库（`.gitignore` 覆盖 `*.pt` 与 `data/`）。
