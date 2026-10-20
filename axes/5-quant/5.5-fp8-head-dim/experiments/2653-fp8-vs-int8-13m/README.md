# 2653 · fp8 vs int8+块标尺（同一「每位置部署字节」轴）+ 大 head_dim 的标尺摊薄

研究项 `5.5-fp8-head-dim`（轴 5 量化轴）WP1 + WP2 的执行实验。**本文件是运行前预登记契约**；
判据线写死在 `config.json::criterion`（其 mtime 必须早于 `results.json`）。

## 0. 一句话命题

8 位存放有两条路：**定点 int8（需分组标尺）** 与 **浮点 fp8（E4M3/E5M2，可能免标尺）**。
本实验把二者放到**同一条「每位置部署字节」轴**上，在同一个 13M 受控设定下测 clean bpc 与合成长程召回，
回答：**fp8 在本尺度值不值得，它的「免标尺」前提是否成立**（补 `5.1` 的 fp8 质量缺口）。
WP2 另给 `head_dim → G → 每位置字节` 的**解析 + 实测**对账。

## 1. 单因素声明

- 唯一改动 = **KV cache 的 8 位存放格式**（`fp16` / `int8+块标尺 G=32` / `fp8 E4M3` / `fp8 E5M2`）
  与（WP2）**块标尺的组大小 G**。模型、数据、评测窗口序、任务、判据全部固定。
- 纯推理：**不训练、不改权重、不新增参数**（`params_added = 0`）；基底恒为 stock decoder-only。
- 组合留白：不做「量化 × 窗口 / 头共享 / 权重」等组合（组合是本仓的显式留白，不是遗漏）。

## 2. 设备口径（本 run 的硬约束）

**整程 CPU（`device: "cpu"`），全部臂同一设备，不取 GPU 锁。** 单卡 4090 被多轴共享，
`/tmp/pathlm_gpu.lock` 长期被其它轴占用 ⇒ 纯 CPU 推理按 `harness/SCHEMA.md` §20 不适用该锁。

> **跨设备不构成判据**：本 run **不**宣称与 GPU 历史锚 `1.483569` bpc 逐位相等（设备不同），
> 只并列报告并声明差异。判据一律用**同设备（CPU）、同窗口序**的**配对 Δ = bpc_臂 − bpc_fp16**。

## 3. 横轴（判据 C1 的唯一乘式）

```
每位置部署字节 = 2(K,V) · n_layers · n_kv_heads · head_dim · (每值字节) · (positions_kept / T)
```

- 每位置值数 = 2 · 12 · 8 · 32 · 1 = **6144**（与 G / head_dim 无关，只要 `n_kv_heads·head_dim = d_model`）。
- `fp16` = 2 B/值 ⇒ **12288 B/位置**（基线）；`int8 + 块标尺 G` = `1 + 2/G` B/值
  （G=32 ⇒ 1.0625 ⇒ **6528 B/位置**，与 `5.1` 逐项对账）；`fp8 免标尺` = 1 B/值 ⇒ **6144 B/位置**（= `5.1` 的解析价签）。
- 每个数字都由 `fp8_arms.py::deployment_ledger` 给**逐项因子**，并与 `5.1` 的 12288 / 6528 断言对账。

## 4. 臂

| 臂 | 每值字节 | 每位置字节 | 说明 |
|---|---|---|---|
| `fp16` | 2 | 12288 | 基线（KV 存 fp16） |
| `int8_block_g32` | 1.0625 | 6528 | 对照 = `5.1`/`2632` 的块标尺口径（`quantize_kv(scheme="block", G=32)`） |
| `fp8_e4m3` | 1.0 | 6144 | 主臂：E4M3 原样存放（**免标尺假设**），动态范围 ±448 |
| `fp8_e5m2` | 1.0 | 6144 | 主臂：E5M2 原样存放，动态范围 ±57344、尾数更短 |

辅助臂（回答「fp8 是否真免标尺」与 WP2 的 G 扫描）：
`fp8_e4m3_tokscale`（E4M3 + 一把**严格前缀确定**的 per-token fp16 标尺 ⇒ `1 + 2/head_dim` = 1.0625 ⇒ 6528 B/位置，
与 int8 块标尺同字节）、`int8_block_g8`（1.25 B/值 ⇒ 7680）、`int8_block_g16`（1.125 ⇒ 6912）。

## 5. 指标表

| 腿 | 量 | 该涨/该跌 | 口径 |
|---|---|---|---|
| 字节 | 每位置部署字节（含标尺） | 该跌 | 解析式 + 逐项因子，与 `5.1` 对账 |
| 质量 | clean bpc（越低越好） | 该不跌/可排序 | `bpc_of`，主表 `window_seed=2, 24×16` |
| 质量 | 合成长程召回 copy / induction / var_recall | 该不跌 | Wilson 95% CI + `at_floor`；chance = 1/64 = 0.0156 |
| 护栏 | 严格前缀不变性两条 | 必须 True | 改未来 token / 改同批邻居 ⇒ 过去位置 logits 不变 |
| 护栏 | 非量化/恒等臂逐位不变 | 必须成立 | `fp32` 臂：`bpc_of` vs `retention.masked_bpc` 逐位相等 |
| 护栏 | 干预活性 | 必须报 | `changed_frac` / `rel_err` / `n_scales` / `saturated_frac` |
| 活性 | 实现层 equality | 必须成立 | 本模块 fp32 前向 = `kv_quant` 生产前向；int8 块标尺 = `2632/block_arms` 逐位相等 |

## 6. 判据线（逐字，与 `config.json::criterion` 同源）

1. **该跌**：每位置部署字节（fp8 免标尺时 1 B/值；大 `head_dim` 下标尺摊薄 `2/G` B/值）；
   **该不跌/可排序**：clean bpc（fp8 vs int8+块标尺，须越 0.014 噪声带）；
   **必须不变**：严格前缀不变性两条、非量化臂逐位不变；**必须报**：fp8 是否需分组标尺、大 `head_dim` 下的 G 与省幅。
2. **否证条件**：若 fp8 的 clean bpc 代价越噪声带且其免标尺的字节优势被质量否定（同字节下不占优）⇒ 判 fp8
   在本尺度不值得；若 fp8 同样需要分组标尺 ⇒ 判其「免标尺」前提失败；若大 `head_dim` 下标尺摊薄不改变每位置字节
   （与解析不符）⇒ 判实测与解析口径不一致、须查明。
3. **带（C1 裁决）**：主判 = **跨格 0.014 bpc 带，限配对 Δ、≥5 窗口**（本 run 用 5 个窗口 seed `{2,3,4,5,6}`，
   均匀口径 `n_batches=2` ⇒ 满足）；本格自测在力带 = 单臂跨窗口 bpc 极差（如实给数值，**只作辅助描述**）；
   **两带不混用**。越带判定取**逐窗配对 Δ 的 `max|Δ|`**（非均值；披露与 `5.1`/§12.1 的一致性见
   `config.json::criterion.band_aggregation`）；判据要求的「fp8 vs int8+块标尺」**二选一判定见
   `analysis.md` §3.1**（`fp8_e4m3 − int8_block_g32` 的 `max|Δ| = 0.005363` ⇒ 不越带、不可分；
   `fp8_e5m2 − int8_block_g32` = 0.014496 ⇒ 越带）。
4. **地板规则**：`recall <= chance + 2·σ_wilson` ⇒ `at_floor = true`，读作「该任务上不可分辨」，
   不写成「机制无效」（`harness/SCHEMA.md` §19）。
5. **单点不外推**（`harness/SCHEMA.md` §17.11）：`head_dim=64` 的实测缺位时只给解析，不外推。

## 7. 既有做法（prior art）与增量

- fp8 数值格式：**FP8 Formats for Deep Learning**（Micikevicius et al. 2022，`arXiv:2209.05433`）——
  定义 E4M3/E5M2 及其动态范围与精度。本实验**不重造 fp8 格式**。
- 大 `head_dim` 设计点：DeepSeek-V2 MLA `head_dim=128`（`arXiv:2405.04434`）。**只作设计方向，不作判据。**
- vLLM / TensorRT-LLM 的 fp8 KV cache：`记忆·未核`——只作设计方向，不作判据。
- 本地基线：`5.1`/`2632` 的 int8 块标尺（1.0625 B/值、6528 B/位置、Δbpc vs fp16 = +0.000101）；
  trunk fp16 全因果基线（12288 B/位置）。
- **增量**：把 fp8 与 int8+块标尺放在**同一每位置字节轴**、同一 13M 受控设定下比质量
  （业界多在大尺度对各格式分别报，缺同轴小尺度对照），并给 `head_dim → G → 每位置字节` 的解析 + 实测对账。

## 8. 对照与方差

- 对照组 = `fp16` 臂（同设备、同窗口序）；方差用**配对 Δ**（5 窗口 seed）与**单臂跨窗口极差**两个量，
  并显式声明主判用 0.014 带。
- 召回地板：`chance = 1/64 = 0.0156`；每臂 n = 3 个源位置 × 1 批 × 16 请求 × 32 答案 = **1536 个答案**
  （最小可判样本；地板阈值由 `floor_threshold(n, chance)` 给）。

## 9. 成本

CPU 现测约 **2–5 s / 批（16×256）**（`.scratch/66-axis5-quant/evidence/cpu_eval_timing.py` 记 3.4 s/批；共享机上会漂）。
本 run 批数 ≈ 4 主臂 × 39 + 辅助 ≈ 210 批 ⇒ 目标 **≈10–15 min**；**不取 GPU 锁、无显存足迹**。
预算不够时的砍法（预登记）：先砍辅助臂（保 `fp8_e4m3_tokscale`，砍 int8 G 扫描），再砍配对腿批数，保主表 4 臂与判据线。

## 10. 数据与脚本清单

- 数据：`pathlm.data.load_enwik8_full`（`data/enwik8_full.npz` 的评测尾段；**不入库**）。
- checkpoint：`2610-frozen-trunk-jumphead-13m/trunk.pt`，sha256 `86953d…159d2`，运行时断言。
- harness（只读）：`axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest/run.py`（`b21`，
  内含 `2619-kv-budget-chart/run.py` 的 `bpc_of` / `masked_bpc` / `build_tasks` / `pooled_accuracy` / `floor_threshold`）、
  `2631-kv-budget-causal/causal_check.py`（前缀不变性两条）、`2632-kv-block-scale/block_arms.py`（int8 块标尺 equality 对照）、
  `tools/evalkit/kv_quant.py`（`quantize_kv` 的 scheme 分支）。
- 本实验新文件：`fp8_arms.py`（fp8 往返 + 字节账，唯一新逻辑层）、`run.py`（入口，逐臂持久化）、
  `selftest_fp8.py`（可失败单测 + 变异）、`config.json`。

## 11. 失败决策树

| 情形 | 处理 |
|---|---|
| 前缀不变性任一为 False | 该臂不得出数，实验作废（判据 gate） |
| `fp32` 恒等臂两条路径不逐位相等 / 实现层 equality 失败 | 仪器红，先修再跑 |
| `int8_block_g32` 无法逐位复现 `2632` 前向 | 口径漂移，停止并用 equality 定位 |
| CPU 预算不足 | 按 §9 的砍法降级，**如实登记**砍了哪条腿 |
| `head_dim=64` 训练腿拿不到 GPU 锁 / 未交付 | 按契约 §4 走 WP2 降级路径：解析表 + `head_dim=32` 的 G 实测扫描 + **显式登记未完成、不外推** |
| fp8 免标尺结论与质量冲突 | 如实报两臂读数与 `saturated_frac`，宣布「免标尺」前提成立或失败，二选一明确 |

## 12. 产出

`README.md`（本文件）· `config.json` · `fp8_arms.py` · `run.py` · `selftest_fp8.py` · `results.json`
（逐臂持久化：`partial` / `arms_completed` / `arms_pending`）· `table.md` · `analysis.md`（第一段人话摘要）·
`run_log.txt` · `MODEL.md` · `JOURNAL.md` · `wp2_hd64_status.json`（若训练腿走 detach 作业）。

## 13. 运行

```bash
python3 run.py                 # 全跑（4 主臂 + 3 辅助臂 + 仪器自检）
python3 run.py --smoke         # 冒烟：n_batches=1、只跑 fp16 + fp8_e4m3
python3 run.py --charts-only   # 只从已有 results.json 重画 table.md
python3 -m pytest selftest_fp8.py -q   # 单测 + 变异（可失败）
```
