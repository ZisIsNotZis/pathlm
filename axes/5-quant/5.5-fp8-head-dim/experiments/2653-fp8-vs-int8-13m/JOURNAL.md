# JOURNAL · 2653-fp8-vs-int8-13m（过程记录，允许日期流水）

> 日志是过程记录，不是权威；与 `results.json` / `analysis.md` 冲突时以事实文档为准。

## 2026-10-14 · 立项与预登记

- 依据 ticket 66 的派单契约 `.scratch/66-axis5-quant/DISPATCH-5.5.md` 执行 5.5 的 WP1 + WP2。
- **设备口径改为纯 CPU**（契约 §5 + 派单补充约束）：单卡 4090 被多轴共享，`/tmp/pathlm_gpu.lock`
  在 22:06 由 `holder_pid 2420657` 持有（占 4944 MiB）⇒ 纯 CPU 推理按 `harness/SCHEMA.md` §20 不适用 GPU 锁。
  全部臂同一设备（CPU），判据改用同设备、同窗口序的**配对 Δ**；与 GPU 历史锚 1.483569 只并列、不宣称逐位相等。
- 运行前落盘：`README.md`（预登记契约）、`config.json`（判据线写死，mtime 早于 `results.json`）、
  `fp8_arms.py`（新语义层：fp8 往返 + 字节账 + head_dim/G 阶梯）、`run.py`（入口，逐臂持久化）。

## 2026-10-14 · 实现过程中的三个实测发现（写进实现口径）

1. **E4M3 与 E5M2 的溢出语义不同**（torch CPU cast 实测）：E4M3（`fn`，无 inf）超 ±448 **饱和**到 ±448；
   E5M2 超 ±57344 **溢出为 inf**。故活性量除 `rel_err`/`changed_frac` 外还报 `saturated_frac`（`|x| > max` 的占比）。
2. **`--d 256 --heads 4` 的 head_dim=64 训练腿不需要改引擎**（`train_depth_ar.py` 的 flag 已支持）；
   该腿**未跑**（GPU 锁被占，见 `wp2_hd64_status.json`）⇒ 按契约 §4 走降级路径、不外推。
3. **CPU 批时 ≈1.7–5 s / 批（16×256）**（共享机上波动大；`.scratch/66-axis5-quant/evidence/cpu_eval_timing.py` 记 3.4 s）。
   据此把召回腿压到最小可判样本（copy 3 个源位置 × 1 批 = 1536 个答案），并把配对腿做成**均匀口径** 5 窗口 × 2 批。

## 2026-10-14 · 冒烟与仪器自检

- `python3 run.py --smoke` 走通全流程（fp16 + fp8_e4m3），并暴露并修掉三个实现缺陷：
  臂的 `head_dim` 不能塞进 `cfg["model"]`（会打断 `DepthARConfig`）、fp16 档的 `scheme` 必须是 `per_value`、
  以及初次 `render_table` 在 `instrument` 尚未生成时 KeyError。
- 正式 run 的仪器自检全绿：`bpc_of` = `retention.masked_bpc`（逐位）、本模块 fp32 前向 = `kv_quant` 生产前向（逐位）、
  `int8 块标尺 G=32` = `2632/block_arms` 前向（逐位）。
- `selftest_fp8.py` 11 项（含 4 个变异）全过；其中 E5M2 的「最近可表示值」最初写错（600 → 640 而非 1024），
  由测试抓出后改正 —— 测试确实有牙。

## 2026-10-14 · 第 1 轮复核处置（返工 writer，fresh 上下文）

复核报告：`.scratch/66-axis5-quant/REVIEW-5.5.md`（判词 `OK with notes`，1 blocker / 3 normal / 4 info）。
本节只处置派单点名的 **B1 / N3 / I2 / I4**（文档面改动，**未改任何读数与判决**）。

**独立复算（B1 + §3.1）**：命令 = 读 `results.json` 的 `points[*].bpc_paired_windows`（两臂在同一
`window_seed` 上相减，等价于两臂各自的 fp16 配对 Δ 相减），窗口 seed `{2,3,4,5,6}`、`n_batches=2`：
- `fp8_e4m3 − int8_block_g32` = +0.002229 / +0.002025 / +0.005363 / +0.001024 / +0.001833 ⇒ `max|Δ| = 0.005363 < 0.014` ⇒ **不越带（不可分）**。
- `fp8_e5m2 − int8_block_g32` = +0.011607 / +0.008847 / +0.008257 / +0.012915 / +0.014496 ⇒ `max|Δ| = 0.014496 > 0.014` ⇒ **越带**。
- `fp8_e4m3_tokscale − int8_block_g32` ⇒ `max|Δ| = 0.004532` ⇒ 同 6528 B/位置下不可分。
（复核者自算的 e4m3 值 0.005363 与本 writer 独立复算**一致**；e5m2 的 0.014496 由本 writer 补出。）
结论：二选一 = `fp8_e4m3` 无标尺相对 `int8_block_g32` **无越带差异（不可分）**，同轴字节再省 384 B/位置
（−5.9%）⇒ 按每位置字节取 `fp8_e4m3`；`fp8_e5m2` 相对 int8 **越带** ⇒ 同字节下不可用。已写入
`analysis.md §3.1`（并在 §0 结论句、§8 判据表、`README.md §6` 加指针）。

**N3 聚合口径声明**：本格越带判定 = 逐窗配对 Δ 的 `max|Δ|`（`run.py::paired_deltas` 的既有实现即此），
非均值；与 `harness/research-schema.md §12.1`「限配对 Δ、≥5 窗口」一致，与 `5.1/2632` 的「配对 Δ 的区间」一致。
披露：`fp8_e5m2` 的 5 窗均值 +0.011375 落在带内、仅 seed 6 越界 ⇒ 该判定依赖 max 聚合。已写入
`analysis.md §3.1` 与 `config.json::criterion.band_aggregation`（新增键，价值本身即「补声明」）。

**I2/I4 措辞**：§4.2 结论由「每位置字节随 G 的实测与解析逐位一致」改为「实测**标尺计数**与解析 `1/G`
逐位一致 ⇒ `2/G` 摊薄项成立；7680/6912/6528 仍是解析价签」；§0/§3 去掉「int8 最稳/买不到质量」的
带内排序含义，改为「质量不可分辨（不构成严格支配）」；§0 一句话显式写「`head_dim=64` 的 48.4375% 只有解析、不外推」。

**文件改动**（本目录内）：`analysis.md`、`README.md`、`config.json`、`JOURNAL.md`。
**注意**：`config.json` 因本处补声明 mtime 已晚于 `results.json`（`README.md:5` 的「config mtime 须早于
results.json」是对**运行前预登记线**的要求）；`config.json::criterion.verbatim_contract` 等**预登记数组未改**，
新增键 `band_aggregation` 自带括号注明「第 1 轮复核 N3 处置补声明」。`results.json::criterion` 是 run 时
生成的副本，不含该新增键（生成物未手改）。
**未处置（不在派单范围）**：N1（`fp8_scale_claim_rule` 字面触发 vs `analysis.md:26` 结论）、
N2（`selftest_fp8.py` 缺运行留证）、I1（`clipped_frac = 0.0` 默认值）、I3（trunk 锚 1.483571 vs 1.483569 两处）。
