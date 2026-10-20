# 2655 · `head_dim=64` 的 KV 标尺摊薄（WP2 的实测腿）

研究项 `5.5-fp8-head-dim`（轴 5 量化轴）**WP2** 的执行实验。本文件是**运行前预登记契约**；
判据线写死在 `config.json::criterion`（其 mtime 必须早于 `results.json`）。
**本实验只补 WP2 缺的那一条实测腿**（`head_dim=64`），不改 WP1（`2653-fp8-vs-int8-13m`）的任何读数。

## 0. 一句话命题

块标尺的开销是「每 G 个沿 `head_dim` 相邻的值摊一把 fp16 标尺」= `2/G` B/值，所以**头维度越大、
标尺摊得越薄、省幅越接近 50%**。本实验把模型换成 `head_dim=64`（`--heads 4`，其余配方与 `2616`
mha 臂逐字段一致），实测 `head_dim → G → 每位置字节 → clean bpc` 这条链，并与解析式对账：
**`head_dim=64`、G=64 ⇒ 每值 1.03125 B、每位置 6336 B、省幅 48.4375%**
（`head_dim=32` 时 G=32 封顶在 46.875%）。要验证的是「省幅趋近 50%」在**实测**上也成立，
而不是只停在解析价签上。

## 1. 单因素声明

- 唯一改动 = **模型侧 kv 头数**（`--heads 8 → 4` ⇒ `head_dim = 256/4 = 64`；`d_model/层数/mlp/seq/步数/批大小/seed`
  与 `2616` mha 臂逐字段一致）与**块标尺的组大小 G**（G ∈ {16, 32, 64}）。数据、评测窗口序、任务、判据全部固定。
- 评测阶段**不训练、不改权重、不新增参数**（`params_added = 0`，与 `2653` 同）；基底恒为 stock decoder-only。
- 组合留白：不做「量化 × 窗口 / 头共享 / 权重 / 激活」等组合（组合是本仓的显式留白，不是遗漏）。

## 2. 设备口径（本 run 的硬约束）

- **训练腿**：单卡 4090 共享 ⇒ 按 `SCHEMA.md` §20 取 `/tmp/pathlm_gpu.lock`（协议只读复用
  `2632-kv-block-scale/run.py::GpuLock`，未复制改写）；等锁上限 600 s，等不到**重试排队**、**不杀他人进程**；
  两臂合计 ≈18 min > 10 min ⇒ **不整程持锁，每臂取一次锁、跑完即放**（§20.1）。
- **评测腿**：`device: "cpu"`，**全部臂同一设备**、同窗口序、只报**配对 Δ**；按 §20 纯 CPU 推理不取 GPU 锁。
- **跨设备 / 跨 checkpoint 都不构成判据**：GPU 历史锚 `1.483569`（`2614/2610` 口径，`head_dim=32` 的 trunk）
  与 `2653` 在本机的 CPU fp16 读数 `1.4835687` 都不是本实验的被测权重 ⇒ 只并列报告，
  **不声称逐位相等、不作判据**。

## 3. 横轴（判据 C1 的唯一乘式）

```
每位置部署字节 = 2(K,V) · n_layers · n_kv_heads · head_dim · (每值字节) · (positions_kept / T)
```

- 每位置值数 = `2 · 12 · 4 · 64 · 1 = **6144**`（与 `head_dim=32` 的 `2·12·8·32` 同值，只要 `n_kv_heads·head_dim = 256`）。
- `fp16` = 2 B/值 ⇒ **12288 B/位置**（基线）；`int8 + 块标尺 G` = `1 + 2/G` B/值
  （G=64 ⇒ **1.03125 B ⇒ 6336 B/位置、省幅 48.4375%**；G=32 ⇒ 1.0625 ⇒ 6528、46.875%；G=16 ⇒ 1.125 ⇒ 6912、43.75%）；
  `fp8 E4M3 免标尺` = 1 B/值 ⇒ **6144 B/位置、省幅 50%**。
- 每个数字都由只读复用的 `2653-fp8-vs-int8-13m/fp8_arms.py::deployment_ledger` 给**逐项因子**，并与 `5.1` 的
  12288 / 6528 锚逐项断言对账（`config.json::analytic` 由该函数生成，本文件不手抄数字）。
- **解析 vs 实测的对账口径**（WP2 的验收量）：实测标尺密度 `n_scales/n_values`（探针批）× 2 B/把
  = 每值摊到的标尺字节；加值字节得实测每值字节，乘 6144 得实测每位置字节，与解析价签逐位比对。

## 4. 臂

| 臂 | 每值字节 | 每位置字节 | 省幅 | 说明 |
|---|---|---|---|---|
| `fp16` | 2 | 12288 | 0% | 基线／配对 Δ 的基准 |
| `int8_block_g64` | 1.03125 | 6336 | 48.4375% | **主臂**：`head_dim=64` ⇒ G=64 合法（每 token 沿 head_dim 一把标尺），即要实测的那个点 |
| `fp8_e4m3` | 1.0 | 6144 | 50% | 免标尺臂（动态范围 ±448）：与 G=64 同轴比较 |
| `int8_block_g32` | 1.0625 | 6528 | 46.875% | 辅助 G 扫描：与 `5.1`/`2632` 的 `head_dim=32` 封顶点**同字节**的对照 |
| `int8_block_g16` | 1.125 | 6912 | 43.75% | 辅助 G 扫描：把 `2/G` 项摊开，与解析对账 |

臂名与装配见 `config.json::arms` / `side_arms`；辅助臂**只跑 `scan` 口径**（6×16, seed=2），
只用于 G 阶梯内部的解析对账，不与其他臂混比、不进配对表。

## 5. 指标表

| 腿 | 量 | 该涨/该跌 | 口径 |
|---|---|---|---|
| 字节 | 每位置部署字节（含标尺） | 该跌 | 解析式 + 逐项因子；实测由 `n_scales/n_values` 反推 |
| 质量 | clean bpc（越低越好） | 该不跌/可排序 | `bpc_of`，主表 `window_seed=2, 24×16` |
| 质量 | 合成长程召回 copy / induction / var_recall | 该不跌 | Wilson 95% CI + `at_floor`；chance = 1/64 = 0.0156 |
| 护栏 | 严格前缀不变性两条 | 必须 True | 改未来 token / 改同批邻居 ⇒ 过去位置 logits 不变 |
| 护栏 | 非量化/恒等臂逐位不变 | 必须成立 | `fp32` 臂：`bpc_of` vs `retention.masked_bpc` 逐位相等 |
| 护栏 | 干预活性 | 必须报 | `changed_frac` / `rel_err` / `n_scales` / `saturated_frac` |
| 活性 | 实现层 equality | 必须成立 | int8 块标尺 G=64 前向 = `2632/block_arms` 逐位相等 |

## 6. 判据线（逐字，与 `config.json::criterion` 同源）

1. **该跌**：每位置部署字节（fp8 免标尺时 1 B/值；大 `head_dim` 下标尺摊薄 `2/G` B/值）；
   **该不跌/可排序**：clean bpc（fp8 vs int8+块标尺，须越 0.014 噪声带）；
   **必须不变**：严格前缀不变性两条、非量化臂逐位不变；**必须报**：fp8 是否需分组标尺、大 `head_dim` 下的 G 与省幅。
2. **否证条件**：若 fp8 的 clean bpc 代价越噪声带且其免标尺的字节优势被质量否定（同字节下不占优）⇒ 判 fp8
   在本尺度不值得；若 fp8 同样需要分组标尺 ⇒ 判其「免标尺」前提失败；**若大 `head_dim` 下标尺摊薄不改变
   每位置字节（与解析不符）⇒ 判实测与解析口径不一致、须查明**（本实验的直接否证条件）。
3. **WP2 验收句**：每位置字节随 `head_dim`/G 的**实测**（标尺密度 → `2/G`）与**解析**一致；质量不越噪声带；对账差异有解释。
4. **带（C1 裁决）**：主判 = **跨格 0.014 bpc 带，限配对 Δ、≥5 窗口**（本 run 用 5 个窗口 seed `{2,3,4,5,6}`，
   均匀口径 `n_batches=2` ⇒ 满足）；本格**无自测在力带**；**两带不混用**。越带判定取逐窗配对 Δ 的 `max|Δ|`
   （非均值）。**率类**（召回）单独定门 = Wilson 95% CI，**不跨量纲借用** 0.014（`docs/research-schema.md` §12.1）。
5. **地板规则**：`recall <= chance + 2·σ_wilson` ⇒ `at_floor = true`，读作「该任务上不可分辨」，不写成「机制无效」（§19）。
6. **单点不外推**（§17.11）：`head_dim=64` 的实测只描述该点，不外推到 `head_dim=128` 等未测点。

## 7. 既有做法（prior art）与增量

- fp8 数值格式：**FP8 Formats for Deep Learning**（Micikevicius et al. 2022，`arXiv:2209.05433`）——
  定义 E4M3/E5M2 及其动态范围与精度。本实验**不重造 fp8 格式**。
- 大 `head_dim` 设计点：DeepSeek-V2 MLA `head_dim=128`（`arXiv:2405.04434`）。**只作设计方向，不作判据。**
- vLLM / TensorRT-LLM 的 fp8 KV cache：`记忆·未核`——只作设计方向，不作判据。
- 本地基线：`5.1`/`2632` 的 int8 块标尺（`head_dim=32`、G=32 ⇒ 1.0625 B/值、6528 B/位置）；
  `2653` 的 fp8 vs int8 同轴判决（`head_dim=32`；E4M3 相对 int8 不可分 `max|Δ| = 0.005363`，
  E5M2 越带 `0.014496`）。
- **增量**：把「标尺摊薄随 `head_dim` 变细」这条**解析**结论在 `head_dim=64` 上做**实测**对账
  （业界通常只在固定头维度上报块标尺开销，缺「同一模型轴上改 head_dim」的小尺度受控点）。

## 8. 对照与方差

- 对照组 = 同 checkpoint 的 `fp16` 臂（同设备、同窗口序）；方差用**配对 Δ**（5 窗口 seed）
  与**单臂跨窗口极差**两个量，并声明主判用 0.014 带。
- 召回地板：`chance = 1/64 = 0.0156`；每臂 n = 3 个源位置 × 1 批 × 16 请求 × 32 答案 = **1536 个答案**
  （最小可判样本；地板阈值由 `floor_threshold(n, chance)` 给）。

## 9. 训练（detach + 失败重试 + §20 按臂取锁）

```bash
# 起点（本 run 已用该命令起训；detach、失败 sleep 20 重试）
cd <expdir> && setsid nohup python3 hp64_train_driver.py >> train_driver.log 2>&1 &
```

- 臂 = `hd64-s0`（seed 0，主臂）→ `hd64-s1`（seed 1，`n≥2` 的第二 seed）。
  单臂命令 = `python3 <ROOT>/train_depth_ar.py hd64-sN --d 256 --layers 12 --heads 4 --mlp-mult 6 --dense
  --steps 12000 --batch-size 32 --seed N --lr 0.001 --eval-batches 24 --out-root <expdir>/runs`
  （逐臂落 `runs/hd64-sN/{cmd.txt,stdout.log,model.pt,results.json}`；机器可读状态见 `train_driver_state.json`）。
- **被切断后如何续跑**：`hp64_train_driver.py` 可重复启动——已有 `runs/hd64-sN/model.pt` 的臂**直接跳过**
  （`PATHLM_HD64_RETRAIN=1` 强制重训），所以重跑驱动只会补没做完的臂；`hd64-s0` 已完成时可直接起评测（§10）。
  **预算紧时砍 `hd64-s1`**：先 `touch STOP_AFTER_FIRST_ARM` 再起驱动（或跑完 `hd64-s0` 后 touch），
  驱动会在起 `hd64-s1` 前显式登记 `cut_by_stop_file` 并退出。
- 历史锚（只作预算记账）：同配方 12k 步墙钟 8.44–9.06 min/臂（`2616-gqa-mqa-13m/MODEL.md`）。

## 10. 评测（finisher 的一条命令）

```bash
cd /home/z/vibe/pathlm && python3 axes/5-quant/5.5-fp8-head-dim/experiments/2655-fp8-hd64-head-dim/hp64_arms.py
```

- 前置：`runs/hd64-s0/model.pt` 存在（`hp64_arms.py` 会在缺失时**显式报错**，不静默降级）。
- 输出：`results.json`（逐臂持久化：`points` / `side_points` / `partial` / `wp2_reconciliation` /
  `paired_delta` / `instrument` / `checkpoint` / `code`）+ `table.md`（4 张表）+ 控制台进度。
- 辅入口：`--ledger-only`（只打印解析字节账，秒级、不装载模型）；`--charts-only`（从已有 `results.json` 重画表）；
  `--smoke --checkpoint-path … --checkpoint-sha256 …`（冒烟/仪器自检，`plumbing_only = true`，**不构成读数**）。
- **第二 seed 腿（`n≥2`）**：
  `python3 <expdir>/hp64_arms.py --seed-leg second_seed --out-dir <expdir>/hd64-s1`
  ——臂表 / 口径 / 判据线逐字不变，只把被测权重换成 `config.json::checkpoints.second_seed`
  （`runs/hd64-s1/model.pt`，`sha256` 预登记为 null ⇒ 评测时现算）；产出 `hd64-s1/results.json` +
  `hd64-s1/table.md`，`plumbing_only = false`、`checkpoint_role = "second_seed"`。
  两条腿分开落盘、不合并成一张主表（每腿的配对 Δ 都是对**该腿自己的** `fp16` 臂算的）。

## 11. 数据与脚本清单

- 数据：`pathlm.data.load_enwik8_full`（`data/enwik8_full.npz` 的评测尾段；**不入库**）。
- checkpoint：`runs/hd64-s0/model.pt`（`config.json::checkpoints.trunk`；预登记时期望 sha256
  **声明缺失 = null**，评测时现算并落盘 `results.json::checkpoint.sha256`）。
- 只读复用（不在本目录复制改写；sha256 记在 `config.json::reuse`）：
  `2653-fp8-vs-int8-13m/fp8_arms.py`（fp8 往返 + 解析字节账）、`2653-fp8-vs-int8-13m/run.py`（`eval_arm` / `paired_deltas` / `build_tasks`）、
  `2621-kv-budget-rest/run.py`（内含 `2619`：`bpc_of` / `masked_bpc` / `pooled_accuracy` / `floor_threshold`）、
  `2631-kv-budget-causal/causal_check.py`（前缀不变性两条）、`2632-kv-block-scale/block_arms.py`（int8 块标尺 equality）、
  `2632-kv-block-scale/run.py::GpuLock`（§20 锁协议）、`tools/evalkit/kv_quant.py`（`quantize_kv(scheme="block", G)`）。
- 本实验新文件：`config.json`（运行前契约）· `hp64_train_driver.py`（训练驱动）·
  `hp64_arms.py`（评测入口 + WP2 对账 + 表格）· `README.md`（本文件）· `wp2_status.json`（状态登记）·
  `analysis.md`（结论与对账）· `JOURNAL.md`（过程流水）·
  `f5_anchor_2653_precision.py`（F5 ③ 的 12 位单点，写 `f5-anchor-2653-rerun/precision_point.json`）。
- `f5-anchor-2653-rerun/`：F5 ③ 等价证据的原始证据子目录（`2653-fp8-vs-int8-13m/run.py --arms fp16` 的输出 +
  12 位单点），**不是本实验的读数**，仅供 `2653-fp8-vs-int8-13m/results.json::anchor_single_point_f5` 溯源。

## 12. 失败决策树

| 情形 | 处理 |
|---|---|
| 前缀不变性任一为 False | 该臂不得出数，实验作废（判据 gate） |
| `fp32` 恒等臂两条路径不逐位相等 / G=64 与 `2632` 前向不逐位相等 | 仪器红，先修再跑 |
| 拿不到 GPU 锁（等锁 600 s 超时） | 驱动记 `wait_lock_timeout` 并 `sleep 20` 重试；**不杀他人进程**、不降级配方 |
| 训练失败（rc≠0） | 驱动重试；连续失败则在 `train_driver_state.json` 记 `failed` 并保留 `stdout.log` 供诊断 |
| 预算不足 | 按 `config.json::cost.budget_cut_rule` 砍：辅助 G 扫描 → 配对腿批数 → 第二 seed；**不砍判据线** |
| 实测每位置字节与解析不一致 | 判「实测与解析口径不一致、须查明」（本实验的直接否证条件），如实登记差异、不修改判据 |
| `hd64-s1` 未完成 | 主表按 `hd64-s0`（n=1）出数并**显式声明 n=1**；不外推方差 |

## 13. 产出

`README.md`（本文件）· `config.json` · `hp64_train_driver.py` · `hp64_arms.py` ·
`train_driver.log` + `train_driver_state.json` · `runs/hd64-s0/**`、`runs/hd64-s1/**`（权重不入库）·
`results.json` + `table.md`（seed 0 主表）· `hd64-s1/results.json` + `hd64-s1/table.md`（seed 1 腿）·
`analysis.md`（评测后）· `wp2_status.json`（状态与接手命令）·
`JOURNAL.md`（过程流水）· `f5_anchor_2653_precision.py` + `f5-anchor-2653-rerun/**`（F5 ③ 证据）。

## 14. 成本（只用等效理论量）

- **判据侧成本 = 等效理论量**：每位置部署字节（值数 × 每值字节）；**墙钟不进判据**，只作预算记账。
- 训练：≈9 min GPU/臂（历史同配方实测），两臂 ≈18 min，串行取锁。
- 评测：CPU 现测 ≈3.4 s/批（16×256；共享机会漂），≈135 批 ⇒ **≈8 min**，无显存足迹、无 GPU 锁。

## 15. 运行（速查）

```bash
python3 hp64_arms.py                  # 正式评测（3 主臂 + 2 辅助臂 + 仪器自检；seed 0 主表）
python3 hp64_arms.py --seed-leg second_seed --out-dir hd64-s1   # 第二 seed 腿（n≥2）
python3 hp64_arms.py --ledger-only    # 解析字节账（秒级）
python3 hp64_arms.py --charts-only    # 只重画 table.md
python3 hp64_arms.py --smoke --checkpoint-path <pt> --checkpoint-sha256 <hex>   # 冒烟/plumbing
python3 hp64_train_driver.py          # 训练驱动（可重复启动续跑；detach 用法见 §9）
```
