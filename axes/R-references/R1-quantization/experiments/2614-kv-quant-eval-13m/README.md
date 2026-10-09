# 2614 · KV 量化参照线（纯推理，不训练）· 13M · dense

> 契约来源：`.scratch/31-batch4/issues/01-batch4.md` §「契约 K」+ 开头统一纪律段。
> **副本标注**：§3 方法行与 §4 判据线**逐字**抄录（含原文换行）；其余小节为**摘要**
> 并注明。判据线一律未改。
> 入口：`run.py`；实现：`tools/evalkit/kv_quant.py`；单测：`tests/test_kv_quant.py`；
> 变异验证：`mutate.py` + `mutation_report.json`；稳定性专报：`stability.md`；
> 原始结果：`results.json`；逐 batch 活性量日志：`kv_quant_log.jsonl`；
> 全电池 stdout：`run_log.txt`；权重清单：`MODEL.md`。

## 1. 命题

**把 KV cache 存成 int8 / int4（评测期量化，不改权重、不训练），质量掉多少、内存省多少？**
这是参照线：**任何「省内存 / 丢信息」的主张（丢弃、压缩、窗口、蒸馏、稀疏）都必须
先打败这条现成路线。**

## 2. 单因素声明

唯一因素 = **KV cache 的数值精度**。被测量模型、checkpoint、数据流、评测窗口、
位置集、读出全部逐字段一致；不改权重、不训练、不新增参数
（`results.json::intervention.params_added = 0`）。基底 = 主流标准 decoder-only
transformer（`DepthARModel(dense=True)`：12 层 × 256 宽 × 8 头 × mlp6，seq256，
tied 读出、无 final LN）。无「必要性例外」。

**干预实现**：在注意力对 K/V 取 `ln1` 投影之后、`scaled_dot_product_attention`
之前，把 K/V 做一次**量化-反量化往返**（存低精度、读回 fp32 参与注意力）——
这正是 KV cache 的存储语义，不是事后对 logits 做手脚。query 恒 fp32。

**量化口径（摘要，落地细节不改判据）**：

- 布局 `[B, H, T, head_dim]`（`Block` 的 `transpose(1,2)`）。**量化口径**：**逐请求、严格因果 `cummax`**——位置 t 的
  scale = `max_{t'≤t}|x_{t'}|/qmax`，只用 ≤ 当前时刻的信息（batch 不参与归约），
  **不重编码、无 fp16 残差窗口**。
- **因果标尺的内存价（同 `2631-kv-budget-causal`）**：scale 逐位置变、无法摊薄 ⇒ 每个值
  要多存一个 fp16 标尺 ⇒ **int8 = 3 B/值（fp16 基线 2 B/值的 1.5×）、int4 = 1.25×**
  ⇒ 严格因果逐值标尺下量化是内存负收益。
- **对称**：零点固定 0，`qmax = 2^(bits-1)-1`（int8→127，int4→7），
  `q = clamp(round(x/scale), -qmax, qmax)`，反量化 `x̂ = q·scale`。
  往返误差上界 `|x̂ − x| ≤ 0.5·scale`（单测钉住）。
- `fp16` 档 = cast 到 fp16 再读回（**基线**，工业默认 KV cache 精度）；另有
  `fp32` 档 = 恒等（**释放版前向**，用于实现等值自检，逐位相等）。

## 3. 方法（**逐字**抄录契约「方法」行）

> - 方法：复用契约 I 的同一 checkpoint（核对 sha256），只改 KV cache 的数值精度：
>   `fp16`（基线）/ `int8`（对称 per-channel 量化）/ `int4`（同上）；测 Δbpc 与内存占用。
>   实现须是**评测期**量化（不改模型权重、不训练）。

## 4. 判据线（**逐字**抄录契约「判据」行，未改一字）

> - **判据（预登记）**：给出 int8/int4 的 Δbpc 与内存下降；若 int8 的 |Δbpc| < 0.014 ⇒
>   记「近无损」（写进参照线，后续所有压缩主张必须与之比较）；若 int4 劣化 ≥0.014 ⇒
>   记其代价。

**执行时的操作性定义（不改判据，只把符号落定）**：

- 符号统一为 SCHEMA §17.3：`Δ = bpc_对照 − bpc_实验`（正 = 改善）。本实验
  「对照」= `fp16`（基线），「实验」= `int8`/`int4`；量化只会抬高 bpc，故 Δ ≤ 0。
  同时报 `damage = bpc_实验 − bpc_对照 = −Δ`（正 = 变差）。
- 「内存下降」：KV cache 的**解析**每位置字节数与相对 fp16 的下降比
  （见 §6；评测期实现会立刻反量化，故 GPU 实测峰值不下降—— §7 口径标注）。

## 5. 对照与方差

- **对照 = 同一模型的 fp16 KV cache**（`mode="fp16"`，同一 `run.py` 同一电池）。
  另有 `fp32` 档与释放版 dense 前向**逐位相等**（`abs_diff = 0.0`），故三档与
  基线同引擎、同数据流、同窗口、同位置集——**不是**跨 run 对照（无 §9 复用问题）。
  锚：释放版 bpc = 1.483569 vs 2610 记录的 1.4836（差 3.1e-05，同序同口径）。
- 方差界：Δ < 0.014 不可排序（methodology §2）。本实验另测**跨评测窗口 seed**
  的配对 damage 散布作为噪声底（`results.json::noise_floor`）。
- 种子：评测窗口 seed = 2（与 2610 的 `depth_curve` 基线同序）；因果 seed = 0
  （推理无随机，`config.json::kv_quant.seed`）。复现性 = 同 seed 重跑。

## 6. 指标表（多维，逐项）

| 维度 | 指标 | 与谁比 | 结果（见 `analysis.md`） |
|---|---|---|---|
| 解决什么问题 | 「省 KV 内存」的现成参照线，供所有压缩主张作对手 | — | 已建成并通过实现等值自检 |
| 代价 | clean bpc 损伤 damage | 同一模型 fp16 KV 基线 | int8 **+0.00015**；int4 **+0.045661** |
| 有效性 | int8 是否近无损 | 0.014 方差界 | **近无损**（\|Δ\|=0.00015 < 0.014） |
| 收益·用法（内存） | KV cache 每位置字节（**部署口径 = 值 + 逐值 fp16 标尺**；旧列是值字节，见 `2631-kv-budget-causal`） | fp16 = 12288 B/位置 | int8 **18432 B/位置（fp16 的 ×1.50）**；int4 **15360 B（×1.25）** ⇒ **量化在内存轴上是负收益**（因果标尺逐位置变、无法摊薄） |
| 收益·用法（精度） | int4 的代价 | 0.014 方差界 | **劣化 0.0457 ≥ 0.014 ⇒ 记代价** |
| 可控性 | 复现性（同 seed 重跑 Δbpc 差） | 阈值 0.005 | **4.83e-07 ⇒ 完全可复现** |
| 副产品（量化误差活性） | rel_err（Σ\|err\|/Σ\|x\|） | fp32 恒等 = 0 | fp16 1.76e-4；int8 6.84e-3；int4 1.24e-1 |
| 干预是否活着 | changed_frac（数值被改变的比例） | fp32 = 0 | fp16/int8/int4 均 ≈ 1.0 |
| 副产品（实现等值） | fp32 档 == 释放版前向 | 逐位 | **bitwise equal** |

## 7. 成本（唯一数字 = 契约「成本」行）

- 契约写的预算是**纯推理 ≈10 min**。
- **实测**：墙钟 **53.1 s（0.89 min）**，45 次评测（每次 24×16 窗口），
  峰值显存 **184 MiB（net 132 MiB）**——远低于契约预算与 ≤5GB 净预算。
  共享卡（实测时已被他人任务占 ~6.5GB）串行运行，无排队。
- 可中断性：是（纯推理，无训练状态；重跑成本 <1 min）。
- 冒烟计划：`python3 run.py --smoke`（2 个窗口批次，实测步时/显存，
  **不外推时间字段**，遵循 SCHEMA §17.5）。预算不够时砍「噪声底 seed 数 →
  复现性重跑」，**不砍判据线**。

## 8. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | char 级 enwik8（`data/enwik8_full.npz`；前 90M train / 末 10M eval，本实验只用 eval 尾段）；vocab 206 |
| 被测量模型 | `2610-frozen-trunk-jumphead-13m/trunk.pt`（契约 I 的同一 checkpoint），sha256 `86953dcb…159d2`（运行时断言），13,536,768 参数 |
| 入口脚本 | `run.py`（`--smoke` 冒烟）；实现 `tools/evalkit/kv_quant.py` |
| resolved config | `config.json`（model / data / eval / kv_quant / memory / 复现性 / 噪声底 / 方差界） |
| 落盘 | `results.json`（各档 bpc/Δ/内存/活性量/判据）、`kv_quant_log.jsonl`（逐 batch 活性量）、`run_log.txt`（全电池 stdout）、`MODEL.md`（权重哈希） |

## 9. 口径标注

- **bpc**：clean bpc（每字符比特数，越低越好）= 因果移位 CE 的 token 平均；
  1.1-epoch **快照口径**（12k 步、98.3M tokens，未收敛），matched-step（同一
  checkpoint/同一数据流）。
- **内存**：**解析** KV cache 占用（部署关心的量）。评测期实现立刻反量化，
  **实测 GPU 峰值不下降**——不把它当「实测内存下降」报。
- **Δ 符号**：`Δ = bpc_fp16 − bpc_mode`（正 = 改善）；本实验 Δ ≤ 0。
- §16 收敛对账：本实验**不报任何增益**（量化只降不升），故不适用。
