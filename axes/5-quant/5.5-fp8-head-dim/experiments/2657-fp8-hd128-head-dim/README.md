# 2657 · `head_dim=128` 的 KV 标尺摊薄（WP2 的第三点：字节轴往外推一格）

研究项 `5.5-fp8-head-dim`（轴 5 量化轴）**WP2** 的执行实验，也是 `atlas/triage.md` §2b 给本格点出的那个
单一动作（「`5.5` 后续：`head_dim`=128 腿」，3.0 档首位）。本文件是**运行前预登记契约**；判据线写死在
`config.json::criterion`（其 mtime 必须早于 `results.json`）。
**本实验只补 `head_dim=128` 这一个点**，不改 `2653`（WP1）与 `2655`（`head_dim=64`）的任何读数。

## 0. 一句话命题

块标尺每值摊 `2/G` B，`G` 的上限是 `head_dim` ⇒ 头维度越大省幅越接近 50%，但 `G` 变大时**一把标尺罩住
的值也变多**（尺度更粗）。`head_dim=32` 实测封顶 46.875%（`2651`/`2632`），`head_dim=64` 实测 48.4375%
（`2655`）。本实验把 `head_dim` 推到 128（`--heads 2`，其余配方与 `2616` mha 臂逐字段一致），问两件事：

1. **字节腿**：`G=128` 的每位置部署字节是否按解析落到 **6240 B/位置（49.21875%）**，实测标尺密度是否仍是
   `1/G`（⇒ 与解析 `2/G` 摊薄项逐位一致）；
2. **质量腿**：把标尺摊到 `G=128`（一把标尺罩 128 个值）**仍然近无损吗**？以及在同一「每位置字节」轴上，
   免标尺的 fp8 E4M3（6144 B/位置）与 int8 `G=128`（6240 B/位置）在 `head_dim=128` 上还有没有可分辨的质量差。

`fp8` 相对 int8 的**解析**字节优势随 `head_dim` 单调缩小：384 B/位置（hd32）→ 192 B（hd64）→ **96 B（hd128）**。
本实验判的是这条收敛在实测质量腿上是否仍然无代价。

## 1. 单因素声明

- 唯一改动 = **模型侧注意力头数**（`--heads 8 → 4 → 2` ⇒ `head_dim = 256/n_heads = 128`）与**块标尺的组大小
  `G`**（G ∈ {32, 64, 128}）。数据、评测窗口序、任务、判据、步数、批大小、`lr`、`d_model`、层数、`mlp_mult`
  全部固定，与 `2655` 逐字段一致。
- 评测阶段**不训练、不改权重、不新增参数**（`params_added = 0`）；基底恒为 stock decoder-only（MHA，无 GQA）。
- 组合留白：不做「量化 × 窗口 / 头共享 / 权重 / 激活」等组合（组合是本仓的显式留白，不是遗漏，
  `harness/SCHEMA.md` §4）。

## 2. 设备口径（本 run 的硬约束）

- **训练腿**：单卡 4090 共享 ⇒ 按 `harness/SCHEMA.md` §20 取 `/tmp/pathlm_gpu.lock`（协议只读复用
  `2632-kv-block-scale/run.py::GpuLock`，未复制改写）；等锁上限 600 s，等不到**重试排队**、**不杀他人进程**；
  两臂合计 ≈16 min > 10 min ⇒ **不整程持锁，每臂取一次锁、跑完即放**（§20.1）。单臂训练墙钟预算 ≤ 20 min。
- **评测腿**：`device: "cpu"`，**全部臂同一设备**、同窗口序、只报**配对 Δ**；按 §20 纯 CPU 推理不取 GPU 锁。
- **跨设备 / 跨 checkpoint 都不构成判据**：GPU 历史锚 `1.483569`、`2653` 的 CPU fp16 读数
  `1.4835687`（`head_dim=32`）、`2655` 的 `1.483031`（`head_dim=64`）都**不是**本实验的被测权重 ⇒
  只并列报告，**不声称逐位相等、不作判据**。

## 3. 横轴（判据 C1 的唯一乘式）

```
每位置部署字节 = 2(K,V) · n_layers · n_kv_heads · head_dim · (每值字节) · (positions_kept / T)
```

- 每位置值数 = `2 · 12 · 2 · 128 · 1 = **6144**`（与 hd32 的 `2·12·8·32`、hd64 的 `2·12·4·64` 同值，
  只要 `n_kv_heads · head_dim = d_model = 256`）。
- `fp16` = 2 B/值 ⇒ **12288 B/位置**（基线）；`int8 + 块标尺 G` = `1 + 2/G` B/值
  （G=128 ⇒ **1.015625 B ⇒ 6240 B/位置、省幅 49.21875%**；G=64 ⇒ 1.03125 ⇒ 6336、48.4375%；
  G=32 ⇒ 1.0625 ⇒ 6528、46.875%）；`fp8 E4M3 免标尺` = 1 B/值 ⇒ **6144 B/位置、省幅 50%**。
- 数字由只读复用的 `2653-fp8-vs-int8-13m/fp8_arms.py::deployment_ledger` / `bytes_per_value_with_scale` /
  `head_dim_cap` 给出（`config.json::analytic` 由这些函数生成，本文件不手抄）；`hd128_arms.py` 在运行时
  **重算并断言** `config.json::analytic` 与之逐位相同（生成物与事实源不许漂）。
- **解析 vs 实测的对账口径**（WP2 的验收量）：实测标尺密度 `n_scales/n_values`（探针批）× 2 B/把
  = 每值摊到的标尺字节；加值字节得实测每值字节，乘 6144 得实测每位置字节，与解析价签逐位比对。

## 4. 臂

| 臂 | 每值字节 | 每位置字节 | 省幅 | 说明 |
|---|---|---|---|---|
| `fp16` | 2 | 12288 | 0% | 基线／配对 Δ 的基准 |
| `int8_block_g128` | 1.015625 | 6240 | 49.21875% | **主臂**：`head_dim=128` ⇒ G=128 合法（每 token 沿 head_dim 一把标尺），即要实测的那个点 |
| `fp8_e4m3` | 1.0 | 6144 | 50% | 免标尺臂（动态范围 ±448）：与 G=128 同轴比较 |
| `int8_block_g64` | 1.03125 | 6336 | 48.4375% | 辅助 G 扫描：与 `2655` 的主臂**同字节**（6336 B/位置），跨 `head_dim` 的同字节对照点 |
| `int8_block_g32` | 1.0625 | 6528 | 46.875% | 辅助 G 扫描：与 `5.1`/`2632` 的 hd32 封顶点同字节（6528 B/位置） |

臂名与装配见 `config.json::arms` / `side_arms`；辅助臂**只跑 `scan` 口径**（6×16, seed=2），只用于 G 阶梯
内部的解析对账，不与其他臂混比、不进配对表。

## 5. 指标表

| 腿 | 量 | 该涨/该跌 | 口径 |
|---|---|---|---|
| 字节 | 每位置部署字节（含标尺） | 该跌 | 解析式 + 逐项因子；实测由 `n_scales/n_values` 反推 |
| 质量 | clean bpc（越低越好） | 该不跌/可排序 | `bpc_of`，主表 `window_seed=2, 24×16` |
| 质量 | 合成长程召回 copy / induction / var_recall | 该不跌 | Wilson 95% CI + `at_floor`；chance = 1/64 = 0.0156 |
| 护栏 | 严格前缀不变性两条 | 必须 True | 改未来 token / 改同批邻居 ⇒ 过去位置 logits 不变 |
| 护栏 | 非量化臂逐位不变 | 必须成立 | `fp32` 臂：`bpc_of` vs `retention.masked_bpc` 逐位相等 |
| 护栏 | 干预活性 | 必须报 | `changed_frac` / `rel_err` / `n_scales` / `saturated_frac` |
| 活性 | 实现层 equality | 必须成立 | int8 块标尺 G=128 **与** G=64 前向各自与 `2632/block_arms` 逐位相等 |

## 6. 判据线（逐字，与 `config.json::criterion` 同源；读数之前写死）

1. **该跌**：每位置部署字节（`G=128` ⇒ 6240 B/位置；fp8 免标尺 ⇒ 6144 B/位置；两者都须比 `2655` 的
   6336/6144 更低或持平）；**该不跌/可排序**：clean bpc（配对 Δ 须越 0.014 带才算可分辨）；
   **必须不变**：严格前缀不变性两条、非量化臂逐位不变；**必须报**：fp8 是否需分组标尺、`head_dim=128` 下的 G 与省幅。
2. **两个二选一判决**（缺一不算闭）：
   - **Q1（摊薄质量腿）**：`int8_block_g128` 相对同 checkpoint 的 `fp16`，5 窗口配对 Δ 的 `max|Δ|`
     **≤ 0.014** ⇒ 判「`G=128` 的摊薄在本尺度仍近无损」；**> 0.014** ⇒ 判「摊薄到 `G=128` 有可分辨的质量代价」
     （即「省幅趋近 50%」不是免费的，边界写在该 head_dim 上）。
   - **Q2（fp8 vs int8 同轴腿）**：`fp8_e4m3 − int8_block_g128` 的 5 窗口配对 Δ `max|Δ|`
     **≤ 0.014** ⇒ 判「`head_dim=128` 上两者质量不可分辨」（此时 fp8 的 96 B/位置优势站得住）；
     **> 0.014** ⇒ 判「该 byte 水平上 fp8 与 int8 质量可分辨」，方向按 Δ 符号报。
3. **否证条件**：若大 `head_dim` 下标尺摊薄不改变每位置字节（实测标尺密度偏离 `1/G`、与解析不符）⇒ 判
   「实测与解析口径不一致、须查明」；若 fp8 同样需要分组标尺 ⇒ 判其「免标尺」前提失败。
4. **带（C1 裁决）**：主判 = **跨格 0.014 bpc 带，限配对 Δ、≥5 窗口**（本 run 用 5 个窗口 seed `{2,3,4,5,6}`，
   均匀口径 `n_batches=2` ⇒ 满足）；本格**无自测在力带**；**两带不混用**。越带判定取逐窗配对 Δ 的
   `max|Δ|`（非均值）。**率类**（召回）单独定门 = Wilson 95% CI，**不跨量纲借用** 0.014
   （`harness/research-schema.md` §12.1）。
5. **地板规则**：`recall <= chance + 2·σ_wilson` ⇒ `at_floor = true`，读作「该任务上不可分辨」，
   不写成「机制无效」（§19）。
6. **单点不外推**（§17.11）：`head_dim=128` 的实测只描述该点，不外推到 `head_dim=256`、别的行宽或更长的 `T`。
7. **n≥2**：两条 seed 腿（`hd128-s0` seed 0 主腿 / `hd128-s1` seed 1 第二腿），各腿只判**自己**的配对 Δ；
   任一腿出现「越带」（负结论）时须两腿同向才算结论。

## 7. 既有做法（prior art）与增量

- fp8 数值格式：**FP8 Formats for Deep Learning**（Micikevicius et al. 2022，`arXiv:2209.05433`）——
  定义 E4M3/E5M2 及其动态范围与精度。本实验**不重造 fp8 格式**。
- 大 `head_dim` 设计点：DeepSeek-V2 MLA `head_dim=128`（`arXiv:2405.04434`）。**只作设计方向，不作判据。**
- vLLM / TensorRT-LLM 的 fp8 KV cache：`记忆·未核`——只作设计方向，不作判据。
- 本地链：`5.1`/`2632`（hd32、G=32 封顶 6528 B/位置）→ `2653`（hd32 上 fp8 vs int8 同轴：E4M3 不可分
  `max|Δ| = 0.005363`、E5M2 越带 `0.014496`）→ `2655`（hd64：G=64 ⇒ 6336 B/位置、实测 = 解析、
  fp8 − int8 `max|Δ| = 0.002279`）。
- **增量**：把「标尺摊薄 ⇒ 省幅趋近 50%」这条**解析**结论再往外推一格到 `head_dim=128`，并第一次在
  **`G` 大到 `head_dim`（一把标尺罩 128 个值）**时检验「摊薄是否仍然近无损」——业界典型只在固定头维度上
  报块标尺开销，缺「同一模型轴上把 `G` 顶到头维度」的小尺度受控点。

## 8. 对照与方差

- 对照组 = 同 checkpoint 的 `fp16` 臂（同设备、同窗口序）；方差用**配对 Δ**（5 窗口 seed）与
  **单臂跨窗口极差**两个量，并声明主判用 0.014 带。
- 召回地板：`chance = 1/64 = 0.0156`；每臂 n = 3 个源位置 × 1 批 × 16 请求 × 32 答案 = **1536 个答案**
  （最小可判样本；地板阈值由 `floor_threshold(n, chance)` 给）。

## 9. 训练（detach + 失败重试 + §20 按臂取锁）

```bash
# 起点（detach、失败 sleep 20 重试）
cd <expdir> && setsid nohup python3 hd128_train_driver.py >> train_driver.log 2>&1 &
```

- 臂 = `hd128-s0`（seed 0，主臂）→ `hd128-s1`（seed 1，`n≥2` 的第二 seed）。
  单臂命令 = `python3 <ROOT>/train_depth_ar.py hd128-sN --d 256 --layers 12 --heads 2 --mlp-mult 6 --dense
  --steps 12000 --batch-size 32 --seed N --lr 0.001 --eval-batches 24 --out-root <expdir>/runs`
  （逐臂落 `runs/hd128-sN/{cmd.txt,stdout.log,model.pt,results.json}`；机器可读状态见 `train_driver_state.json`）。
- **被切断后如何续跑**：`hd128_train_driver.py` 可重复启动——已有 `runs/hd128-sN/model.pt` 的臂**直接跳过**
  （`PATHLM_HD128_RETRAIN=1` 强制重训）；预算紧时 `touch STOP_AFTER_FIRST_ARM` 让驱动只跑第一臂。
- **先绑定再训练**：`python3 tools/prereg_bind.py <expdir>` 必须在本驱动开跑之前完成（驱动本身不取锁外的
  任何旁路；`train_depth_ar.py` 的 `--require` 门会拦）。
- 历史锚（只作预算记账）：同配方 12k 步墙钟 7.77–8.15 min/臂（`2655/runs/*/results.json::wall_minutes`）。

## 10. 评测（finisher 的一条命令）

```bash
cd /home/z/vibe/pathlm && python3 axes/5-quant/5.5-fp8-head-dim/experiments/2657-fp8-hd128-head-dim/hd128_arms.py
```

- 前置：`runs/hd128-s0/model.pt` 存在（`hd128_arms.py` 缺失时**显式报错**，不静默降级）。
- 输出：`results.json`（逐臂持久化：`points` / `side_points` / `partial` / `wp2_reconciliation` /
  `paired_delta` / `paired_delta_arm_vs_arm` / `instrument` / `checkpoint` / `code`）+ `table.md`（4 张表）。
- **本实验新增的那一个读数**：`paired_delta_arm_vs_arm` —— 直接把 `fp8_e4m3 − int8_block_g128` 的逐窗配对 Δ
  落盘（`2653`/`2655` 的「fp8 − int8」是判读时手算的，本 run 把它变成机检字段）。
- 辅入口：`--ledger-only`（只打印解析字节账，秒级、不装载模型）；`--charts-only`（从已有 `results.json` 重画表）；
  `--smoke --checkpoint-path … --checkpoint-sha256 …`（冒烟/仪器自检，`plumbing_only = true`，**不构成读数**）。
- **第二 seed 腿（`n≥2`）**：
  `python3 <expdir>/hd128_arms.py --seed-leg second_seed --out-dir <expdir>/hd128-s1`
  ——臂表 / 口径 / 判据线逐字不变，只把被测权重换成 `config.json::checkpoints.second_seed`
  （`runs/hd128-s1/model.pt`，`sha256` 预登记为 null ⇒ 评测时现算）；产出 `hd128-s1/results.json` +
  `hd128-s1/table.md`，`plumbing_only = false`、`checkpoint_role = "second_seed"`。

## 11. 数据与脚本清单

- 数据：`pathlm.data.load_enwik8_full`（`data/enwik8_full.npz` 的评测尾段；**不入库**）。
- checkpoint：`runs/hd128-s0/model.pt`（`config.json::checkpoints.trunk`；预登记时期望 sha256
  **声明缺失 = null**，评测时现算并落盘 `results.json::checkpoint.sha256`）。
- 只读复用（不在本目录复制改写；sha256 记在 `config.json::reuse`）：
  `2653-fp8-vs-int8-13m/fp8_arms.py`（fp8 往返 + 解析字节账 + `head_dim_ladder`/`head_dim_cap`）、
  `2653-fp8-vs-int8-13m/run.py`（`eval_arm` / `paired_deltas` / `build_tasks`）、
  `2621-kv-budget-rest/run.py`（内含 `2619`：`bpc_of` / `masked_bpc` / `pooled_accuracy` / `floor_threshold`）、
  `2631-kv-budget-causal/causal_check.py`（前缀不变性两条）、`2632-kv-block-scale/block_arms.py`（int8 块标尺
  equality）、`2632-kv-block-scale/run.py::GpuLock`（§20 锁协议）、`tools/evalkit/kv_quant.py`
  （`quantize_kv(scheme="block", G)`）。
- 本实验新文件：`config.json`（运行前契约）· `hd128_train_driver.py`（训练驱动）·`hd128_arms.py`
  （评测入口 + WP2 对账 + 表格）· `README.md`（本文件）· `analysis.md`（结论与对账）· `JOURNAL.md`（过程流水）。

## 12. 失败决策树

| 情形 | 处理 |
|---|---|
| 前缀不变性任一为 False | 该臂不得出数，实验作废（判据 gate） |
| `fp32` 恒等臂两条路径不逐位相等 / G=128 或 G=64 与 `2632` 前向不逐位相等 | 仪器红，先修再跑 |
| 拿不到 GPU 锁（等锁 600 s 超时） | 驱动记 `wait_lock_timeout` 并 `sleep 20` 重试；**不杀他人进程**、不降级配方 |
| 训练失败（rc≠0） | 驱动重试；连续失败则在 `train_driver_state.json` 记 `failed` 并保留 `stdout.log` 供诊断 |
| 预算不足 | 按 `config.json::cost.budget_cut_rule` 砍：辅助 G 扫描 → 配对腿批数 → 第二 seed；**不砍判据线** |
| 被测 trunk 的 clean bpc 显著劣于 hd64/hd32 的 trunk | 如实并列报告（本实验的判据是**同 checkpoint 内**的配对 Δ，不跨 checkpoint 比 bpc）；若 trunk 退化到不可读（bpc ≥ 8 = 均匀分布）⇒ 判「仪器不够：该 head_dim 的 trunk 未学到可测的语言结构」并落盘证据 |
| 实测每位置字节与解析不一致 | 判「实测与解析口径不一致、须查明」（本实验的直接否证条件），如实登记差异、不修改判据 |
| `hd128-s1` 未完成 | 主表按 `hd128-s0`（n=1）出数并**显式声明 n=1**；不外推方差 |

## 13. 产出

`README.md`（本文件）· `config.json` · `hd128_train_driver.py` · `hd128_arms.py` ·
`train_driver.log` + `train_driver_state.json` · `runs/hd128-s0/**`、`runs/hd128-s1/**`（权重不入库）·
`results.json` + `table.md`（seed 0 主表）· `hd128-s1/results.json` + `hd128-s1/table.md`（seed 1 腿）·
`analysis.md`（评测后）· `JOURNAL.md`（过程流水）。

## 14. 成本（只用等效理论量）

- **判据侧成本 = 等效理论量**：每位置部署字节（值数 × 每值字节）；**墙钟不进判据**，只作预算记账。
- 训练：≈8 min GPU/臂（同配方实测），两臂 ≈16–20 min，串行取锁。
- 评测：CPU 现测 ≈3.4 s/批（16×256；共享机会漂），≈135 批 ⇒ **≈8 min**，无显存足迹、无 GPU 锁。

## 15. 运行（速查）

```bash
python3 tools/prereg_bind.py <expdir>                 # 先绑定（判据先于读数）
python3 hd128_train_driver.py                          # 训练两臂（可重复启动续跑）
python3 hd128_arms.py                                  # 正式评测（3 主臂 + 2 辅助臂 + 仪器自检；seed 0 主表）
python3 hd128_arms.py --seed-leg second_seed --out-dir hd128-s1   # 第二 seed 腿（n≥2）
python3 hd128_arms.py --ledger-only                    # 解析字节账（秒级）
python3 hd128_arms.py --charts-only                    # 只重画 table.md
```
