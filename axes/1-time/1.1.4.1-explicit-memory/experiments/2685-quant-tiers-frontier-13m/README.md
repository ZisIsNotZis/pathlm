# 2685 · 把 int4 档与 carry 族的 int8 档叠到同一条「每位置部署字节 → 质量」图上 · 13M

> 归属：`axes/1-time/1.1.4.1-explicit-memory`（研究项 1.1.4.1「显式记忆槽 / 线性状态」的 WP1，服务里程碑验收第 4 条）。
> 入口：`run.py`（纯推理评测 + 前沿重算）；预登记契约 = **本文件 + `config.json` + `run.py` + `carry_quant.py`**；
> 原始读数 `results.json`；判读 `analysis.md`；本目录单测 `test_2685_tiers.py`（纯 CPU）。
> 复用仪器（只读导入、不复制实现）：`../../../../../axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale/block_arms.py`
> 的块标尺量化前向（纯窗臂）、`axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m/realtext_probe.py`
> 的配对 bootstrap（逐窗配对 CI）。臂表、判据常数、字节口径沿用 `../2681-int8-equalbytes-frontier-13m` 并显式列出本实验新补的格。

## 1 命题（一句话、可证伪）

`2680` / `2681` 只把 **int8** 块标尺档叠进「每位置部署字节 vs clean bpc」图，且 carry 族
（显式记忆槽 / 逐层独立状态）**只能用 fp16 档**（`CarryChannelBlock` 不在 `2632` 量化前向的支持面内）。
于是图上有两处缺口：**int4 档完全没有**；**carry 族没有自己的 int8 档**（纯窗族用的是它自己最省的档，
carry 用的是 fp16 档——这一不对称对 carry 不利，但「给 carry 同档以后它还占不占优」从未测过）。

**可证伪形式**：*给 `CarryChannelBlock` 写一条**只做推理期**的量化前向（KV 精度 × 状态精度两轴），
让 carry 点落到与纯窗点相同的每位置部署字节轴上；同时对同一批已训 checkpoint 补 int4 档*。

- 若 carry 的 **int8 全档**（KV int8 + 状态 int8，`carrypl16_s0` = **208,896 B**）在合并网格上
  存在 `total_bytes ≤ 208,896` 且 `bpc ≤ 该格 bpc − 0.014` 的点 ⇒ **该同档格被支配**，
  必须直接写进 `project.yaml::conclusion`；
- 若不存在 ⇒ 该同档格上不被支配，结论强度从「carry fp16 档不被支配」升到「carry 在自己最省的档上也不被支配」。

## 2 单因素声明

- **基底**：与 `2634` / `2638` / `2639` / `2680` / `2681` 逐字段同一 stock dense 规格（d256 × 12 层 × 8 头 × mlp6 × T256，
  13.54M / carry 臂 13.80M，tied 读出 + 无 final LayerNorm + 顺序残差），同一数据（enwik8 前 90M 字符训练 /
  末 10M 字符评测）、同一评测窗口序（`n_batches` 96 × bs16、`window_seed` 2）。
- **唯一变量 = 评测期的数值格式**：KV 精度 ∈ {不量化, int8, int4}、carry 状态精度 ∈ {不量化, int8, int4}，
  都是块标尺（沿 `head_dim`，G=32，2 B 标尺/32 值）。**0 训练 token、不改权重、不改训练入口**。
- **两轴分开动，不并成一个开关**：carry 臂报 `kv8`（只换 KV）、`kv8_state8`（在 `kv8` 之上再换状态）、
  `kv4`（只换 KV 到 int4）、`kv4_state4`（两轴都在 int4）⇒ 每一对相邻格的差只归因到一个轴。
- **本实验不改任何共享文件**（`pathlm/**`、`train_depth_ar.py`、`tools/**`）：新逻辑全部在本目录
  `carry_quant.py`；块内数学与量化算子都是只读导入。
- **状态量化的实现口径**（单因素 = 状态的**存储精度**）：`carry_quant.carry_scan_quant` 在**块边界**
  （`carry_chunk`=64 ⇒ T=256 下 4 块）把状态存一次、读一次（`S_stored = quant(S)`，下一块用反量化的
  `S`），块内计算仍是 fp32；末块之后不再往返（终态没有读者）。这不是「另一条扫描算法」：
  `state="none"` 时它与 `pathlm.depth_ar.carry_scan_chunked` 逐位相同，且整模型层在
  (KV 不量化, 状态不量化) 下与已入库仪器 `retention.masked_logits` 逐位相同——两处都由
  `test_2685_tiers.py` 与判据① 钉住。

## 3 字节账（口径在 `config.json::analytic`：G 与位宽）

每位置每键的 KV 值数 = `n_layers × 2 × d_model` = 6144；
每值字节 = fp16 **2**、int8 **1 + 2/32 = 1.0625**、int4 **0.5 + 2/32 = 0.5625**（与 `2632` / `2680` / `2681` 同一口径）。
每位置部署字节 = `k × 6144 × bytes_per_value[KV 档] + state_values × bytes_per_value[状态档]`。

| 格 | 纯窗 `win16` | `win32` | `win60` | `win128` | `carrypl16_s0`（状态 98,304 值） | `carry16_s0`（状态 8,192 值） |
|---|---|---|---|---|---|---|
| fp16（基线） | 196,608 | 393,216 | 737,280 | 1,572,864 | **393,216** | **212,992** |
| int8 | 104,448 | 208,896 | 391,680 | 835,584 | `kv8` **301,056** / `kv8_state8` **208,896** | `kv8` 120,832 / `kv8_state8` 113,152 |
| int4 | 55,296 | **110,592** | 207,360 | 442,368 | `kv4` 251,904 / `kv4_state4` **110,592** | `kv4` 71,680 / `kv4_state4` 59,904 |

两处**天然同字节**（不是插值，是两条独立部署账算到同一个数）：

- `carrypl16_s0@kv8_state8` = **208,896 B** ≡ `win32_s0@kv8` = **208,896 B**；
- `carrypl16_s0@kv4_state4` = **110,592 B** ≡ `win32_s0@kv4` = **110,592 B**。

## 4 臂（13 个 checkpoint = 9 条纯窗 + 4 条 carry，全部已在盘、本实验不训练）

| 臂 | 窗口 | seed | 跑的格 | 来源 |
|---|---|---|---|---|
| `win16_s0` / `win16_s1` | 16 | 0 / 1 | `fp16` / `kv8` / `kv4` | `2634` 复用 |
| `win17_s0` / `win18_s0` / `win24_s0` / `win32_s0` / `win128_s0` | 17 / 18 / 24 / 32 / 128 | 0 | `fp16` / `kv8` / `kv4` | `2638` 复用 |
| `win60_s0` / `win64_s0` | 60 / 64 | 0 | `fp16` / `kv8` / `kv4` | `2681` 复用（`sha256` 在本实验 `config.json` 里钉住） |
| `carrypl16_s0` | 16 + 逐层状态 | 0 | `fp16` / `kv8` / `kv8_state8` / `kv4` / `kv4_state4` | `2639` 复用 |
| `carrypl16d_s0` | 16 + 逐层状态 + γ=0.99 | 0 | 同上 | `2639` 复用 |
| `carry16_s0` / `carry16_s1` | 16 + 单共享槽 | 0 / 1 | 同上 | `2634` 复用 |

纯窗臂走 `2632` 的仪器（`masked_logits` / `ba.masked_kv_quant_logits`）；carry 臂走本目录
`carry_quant.quant_carry_logits`（KV 精度 × 状态精度可换）。

## 5 判据线（**预登记：写下后不得改**）

| # | 判据 | 判定线 | 该涨 / 该跌 / 必须不变 |
|---|---|---|---|
| ① | **仪器复现门（两道）** | (a) 13 条臂的 `fp16` 格与 `../2681-int8-equalbytes-frontier-13m/results.json::rows[*].bpc_fp16` **1e-6 内相等**；(b) 9 条纯窗臂的 `kv8` 格与同文件 `rows[*].bpc_int8` **1e-6 内相等**。carry 臂的 `fp16` 格走的是**新代码**（`carry_quant.py`），它与落盘值相等即「新前向是已入库仪器的忠实复刻」。任一不等 ⇒ 本实验作废（仪器 / 权重 / 窗口序变了），判据线不改 | **必须不变** |
| ② | **字节账（该跌）** | `results.json::byte_ledger` 用 `config.json::analytic` 的 G=32 与位宽解析：`6144 × 2 = 12,288`、`× 1.0625 = 6,528`、`× 0.5625 = 3,456` 逐项相等；`state_values` 与模型实读数（`carry_quant.num_state_values`）逐臂相等；§3 表里每个数都能由该公式重算 | 该跌 |
| ③ | **int4 质量腿（该不跌，如实报）** | 逐纯窗臂 `Δ_int4 = bpc(kv4) − bpc(fp16)`，逐窗配对 bootstrap（10k 次，seed 2685）给 95% CI；`|Δ_int4| < 0.014` ⇒ **「本尺度无质量增量」**（禁写「无信号」）；`≥ 0.014` ⇒ 量化在该臂上有可排序代价，如实报方向与幅度 | 该不跌 |
| ④ | **carry 族质量腿（该不跌，如实报）** | 逐 carry 臂报三对配对 Δ：`kv8_vs_fp16`（只换 KV）、`state8_vs_kv8`（只换状态）、`tier4_vs_fp16`（两轴都在 int4），各带 95% CI 与 0.014 带的关系；另报 `kv4_vs_fp16`（只换 KV 到 int4） | 该不跌 |
| ⑤ | **主判（同档不被支配）** | 在 47 点合并网格上（9 条纯窗臂 × 3 格 + 4 条 carry 臂 × 5 格），对 carry 的两个同档格各判一次：**int8 全档** `carrypl16_s0@kv8_state8`（208,896 B）与 **int4 全档** `carrypl16_s0@kv4_state4`（110,592 B）。被支配 ⇔ 存在 `total_bytes ≤ 该格字节` 且 `bpc ≤ 该格 bpc − 0.014` 的点。两向都如实报；若被支配必须直接写进 `project.yaml::conclusion` | 二选一如实报 |
| ⑥ | **决定性格必须报** | 报出与每格**同字节**的纯窗点及其差（`win32_s0@kv8` / `win32_s0@kv4`，天然同字节），报出该格的 `best_cheaper_or_equal` 与差；**禁止两点之间插值造点**；结论必须写明网格限定 | 必须报 |
| ⑦ | **往返一致 + 口径** | 同一条臂的 `fp16` 与各量化格**并存**入图（不许只入量化档）；横轴 = 每位置部署字节（值 + 标尺，`2639`/`2680`/`2681` 同一口径）；配对 Δ 用 0.014 bpc 带、σ_cell（0.00166）并列报出、两带不混用；`n=1` 的臂（除 `win16` 两 seed 与 `carry16` 两 seed）只能标「临时」；**墙钟不进判据** | 口径 |

## 6 否证条件

- 判据① 任一不成立 ⇒ 本实验作废（同一 checkpoint、同一窗口序、同一数学应逐位复现已入库仪器）。
- 判据⑤ 任一同档格出现非空支配集 ⇒ **显式状态在该预算点上被同预算纯窗口支配**，这是允许的判决，
  必须直接写进 `project.yaml::conclusion`，不得降级成「不可排序」。
- 判据⑤ 两格的最小差距都 `< 0.014` ⇒ 只许写「同一字节水平不可排序」，不许写成「carry 占优」。
- 判据③④ 若 `|Δ| ≥ 0.014` ⇒ 必须写明「这一档在该臂上有可排序质量代价」，不许把量化当免费。

## 7 成本

- 评测 13 臂 ×（3 或 5）格 = **47 次前向**，0 训练 token（`2680` 实测 ≈2 s/次、carry 扫描更慢）。
- 预算上界：**20 min GPU**（`config.json::budget.gpu_wall_budget_min`），整轮取一次 `/tmp/pathlm_gpu.lock`
  （`harness/SCHEMA.md` §20）。

## 8 复现

```bash
python3 tools/prereg_bind.py axes/1-time/1.1.4.1-explicit-memory/experiments/2685-quant-tiers-frontier-13m \
    --contract README.md config.json run.py carry_quant.py    # 契约文件清单（判据实现面一起绑）
python3 axes/1-time/1.1.4.1-explicit-memory/experiments/2685-quant-tiers-frontier-13m/run.py
python3 -m pytest axes/1-time/1.1.4.1-explicit-memory/experiments/2685-quant-tiers-frontier-13m/test_2685_tiers.py -q
```

## 9 scope 声明

单轴单格（1.1.4.1）。**禁外推**：结论只适用于 enwik8 / T=256 / 13M / 12k 步 / 块标尺 G=32 这一格，
以及本图列的 13 条已落盘 checkpoint（除 `win16` 与 `carry16` 各两 seed 外均为 n=1）；不外推到
T ≫ 256、别的标尺方案（`per_value` / `kivi` / `block_tok`）、别的位宽或 G。
carry 的状态量化按「块边界存一次读一次」实现，块内仍 fp32 ⇒ 本图不外推到「状态在每次写入后都量化」的实现。
