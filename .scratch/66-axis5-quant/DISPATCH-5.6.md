# 派单契约 · 5.6「组合：窗口 + 量化 + 检索」WP1（writer，fresh 上下文）

> 由 axis5-lead 在 ticket 66 Phase B 落盘；派发时把本文件路径交给 worker（`worker` agent，fresh 上下文，单一写入者）。
> 边界：只写 `axes/5-quant/5.6-window-quant-retrieval/experiments/**`（可另写 `.scratch/66-axis5-quant/evidence/W-5.6-*.md`）。
> 禁止写/改：`axes/5-quant/*/project.yaml`（事实文档由 axis5-lead 写）、其它轴、`docs/**`、`pathlm/**`、
> `train_*.py`、`tools/**`、`tests/**`、任何共享文件。只读复用 `tools/evalkit/**` / `pathlm/**` / 其它实验目录
> （尤其 `axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest/**` 与 `2619`/`2631`）。
> 不动 git 索引（不 `git add`/`commit`）。需要引擎/工具改动 ⇒ **不要改**，写进返回报告的「阻塞/需求」段。

## 0. 交付
目录：`axes/5-quant/5.6-window-quant-retrieval/experiments/2654-window-retention-frontier-13m/`
回答：**按保留预算把窗口曲线补完整（keep32→192），并把它与窗口/锚点 × int8 块标尺组合放到同一
「每位置部署字节 → 两腿质量」天平的读数上。**（WP2 检索腿另派，不在本契约内。）

产物：`README.md`（**运行前**预登记契约）· `config.json`（**运行前**写死 `criterion`）· `run.py` ·
`win_arms.py`（如需要新语义层）· `results.json` · `table.md` · `analysis.md`（第一段「人话摘要」）·
`MODEL.md` · `run_log.txt` · `selftest_win.py`（可失败单测/变异）· `JOURNAL.md`。

## 1. 权威输入（各读一次）
- `axes/5-quant/5.6-window-quant-retrieval/project.yaml`（契约：`method`/`validation`/`objective_criteria`/
  `falsification`/`gaps`/`milestone` 逐字照做）。
- `axes/5-quant/5.1-kv-quant/project.yaml::conclusion`（块标尺口径事实：int8 块标尺 G=32 = 1.0625 B/值 =
  6528 B/位置；`2642` 的 7 个同字节水平 = 432/768/816/864/1536/1632/3072 B/位置，**组合臂应落在这些水平上，
  或明确另立水平并说明理由，禁用插值**）。
- 本地已落实的窗口/锚点与保留曲线证据：`.scratch/07-x1-x3/evidence`、
  `axes/1-time/1.1.4.2-compression/experiments/2614-kv-ablation-retention-13m/analysis.md`（M2 ERRATA：
  keep32 = +0.212、keep64 = +0.081 bpc）。
- harness 模板（照抄模式，别改）：`axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest/run.py` +
  `config.json`（**窗口/锚点臂与 `win*` 保留预算的现成分派就在这里**）、`2620-kv-budget-gaps/run.py`、
  `2619-kv-budget-chart/run.py`、`2632-kv-block-scale/run.py`（GPU 锁 + 部署字节账模板）。
- 纪律：`SCHEMA.md` §3/§4（组合留白与例外条款）/§6/§7/§8/§10/§14/§17.12/§20/§22/§23；
  `docs/research-schema.md` §12.1/§11.2。

## 2. 可复用 harness API（只读 import）
`b21 = <2621/run.py 载为模块>`（照抄 `2632/run.py` 的 `_load_module`）：`b21.b19.load_trunk(cfg, dev)`；
`b21.b19.bpc_of(logits_fn, eval_arr, T, n_batches, batch_size, window_seed, dev)`；`b21.b19.masked_bpc(...)`
（独立第二路径）；`b21.b19.build_tasks(cfg, smoke=False)`；`b21.b19.pooled_accuracy(fn, tasks["copy_by_pos"], dev)`
→ `{accuracy,n,wilson95}`；`b21.b19.floor_threshold(n, chance)`；`b21.bytes_per_position(spec, cfg)`；
`b21.make_logits_fn(model, spec, T, dev)`（窗口/锚点/组合的现成分派）；`b21.SOURCE`（每个历史臂的数字来源）。
窗口/锚点的语义层在 `2620-kv-budget-gaps/gaps_arms.py`（`window_anchor`）与 `2621/rest_arms.py`（`combo_mask`）；
可见性掩码工具在 `tools/evalkit/retention.py`（`recent_keep_mask` / `mask_stats` / `masked_bpc`）与
`pathlm.depth_ar.window_anchor_keep_mask`。数据 `pathlm.data.load_enwik8_full`（不入库）。chance = 0.0156。
trunk checkpoint：`axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt`，
sha256 `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2`，clean bpc 历史锚 **1.483569**
（`window_seed=2, n_batches=24, batch_size=16, T=256`）。

## 3. 设计（单因素：保留预算 / 组合）
- 臂：`full`（全因果基线 = 1.483569）· 窗口 `W ∈ {32, 64, 96, 128, 160, 192}`（**保留预算扫描**；已有落盘点的
  用 `b21.SOURCE` 指出的历史结果**只读复用**并在 `results.json::points[*].source` 注明，未跑的点**自己跑**）·
  窗口 + 锚点（`win32_anchor32` 等）· 窗口 × int8 块标尺（`win32_int8` / `win64_int8` …）。
- **量化腿口径**：一律 `5.1` 的**块标尺**（G=32，`tools/evalkit/kv_quant.py::quantize_kv(scheme="block")`）；
  **禁止**引用已作废的逐值标尺数字。
- **横轴**：每位置**部署字节**（值 + 2/G 标尺，含 `kept/T` 因子）——与 `5.1::2642` 同式；组合臂必须落在
  `2642` 的 7 个同字节水平之一或另立水平并说明。
- **两条质量腿**：clean bpc + 合成长程召回（copy / induction / var_recall，Wilson 95% CI，`at_floor`）。
  **必须两腿并报**（「低损伤 ≠ 保住了」：窗口 192 的 bpc 只掉 0.0042 但长程召回掉 25%）。
- **活性判据**：每臂实际保留位置数 / 挡掉的因果对数（`mask_stats`）、锚点覆盖的源比例。
- **护栏**：① 非量化臂逐位不变（`full` 臂与历史锚 + `masked_bpc` 独立路径逐位一致）；② 严格因果/同拓扑
  fp16 基线不变；③ **变异测试**（破坏掩码 / 破坏块标尺 ⇒ 断言报红；未抓住先修单测）。
- **判据带（C1）**：主判 = 跨格 **0.014 bpc 带**，限「配对 Δ、≥5 窗口」。⇒ 主表 `window_seed=2` +
  另跑 `seeds=[3,4,5,6]`（`n_batches` 可缩到 8）⇒ 5 窗口。做不到 ⇒ 显式声明本格自测在力带（给数值）并声明两带不混用。
  率类指标（召回）走 Wilson 95% CI，不借用 0.014。
- **必须回答（契约 `falsification`）**：① 窗口 bpc 损伤是否随保留预算单调（keep32 ≈ +0.212 → keep192 ≈ +0.004，
  与旧「窗口几乎零代价」判读冲突 ⇒ 旧判读被否证）；② 窗口 + 量化组合在**等字节**下是否优于单用其一；
  ③ 与 M2 ERRATA 的 keep32/keep64 数字是否一致（不一致要如实报并给证据）。

## 4. GPU 纪律（`SCHEMA.md` §20，逐字照做）
照抄 `2632/run.py` 的 `GpuLock`：`/tmp/pathlm_gpu.lock` `mkdir` 抢锁 + 陈旧锁自愈 + `holder_pid`/`heartbeat`
+ `trap` 释放；等锁上限 10 分钟（40×15s），等不到**不许杀别人进程**；整程 <10 min 可整程持锁，否则按臂释放
（本实验臂数多 ⇒ **按臂释放**，留 ≥60 s 窗口）。`results.json` 记 `gpu_lock = {holder_pid, wait_seconds,
free_mib_before, free_mib_after, release_ok, held_whole_run}` + `peak_mem_mib_this_run`。显存礼貌线 ≤5 GiB。
**GPU 墙钟预算 ≤40 min（只作过程预算）**；先 `--smoke`。锁拿不到 ⇒ 先做 CPU 可做的，再如实返回；**不得空转等锁**。

## 5. 验收（自检先满足）
- `README.md` = 运行前预登记契约：命题、单因素声明、指标表、判据线逐字、**既有做法 prior art**（生产默认组合拳
  `记忆·未核`；量化腿 KIVI `2402.02750` / KVQuant `2401.18079` / Atom `2310.19102`——只作设计方向、不作判据）、
  对照与方差、成本、数据与脚本清单、失败决策树、产出。`config.json::criterion` 运行前写入。
- `analysis.md`：人话摘要 + 表 + **三道对账**（① 复用点的数字与 `source` 一致；② `full` 与历史锚/独立路径一致；
  ③ 同字节水平与 `2642` 一致）+ 失败与局限（含「单 seed 冻结 checkpoint ⇒ 不宣称结构优劣」「结论只作这些点的描述」
  「**本项最高风险是把兄弟线的单点重绘当新发现 ⇒ 必须自跑组合臂**」）。
- `gaps`：未测项（含检索腿）写成「下一步最小实验 + 等效算力成本」。
- 落盘后跑 `python3 tools/check_tree.py` + `python3 tools/summary.py --check`（只读确认没破坏共享校验）。

## 6. 返回给 lead（≤15 行）
实验目录路径；臂清单与每臂 `bytes_per_position` / `clean bpc` / `recall_copy`；保留预算曲线是否单调（逐点数值）；
组合臂在等字节下是否优于单用其一；与 M2 ERRATA 的一致性；`full` 逐位复原是否成立；走完的否证路径与结果；
**结论一句话**；未测项与下一步；阻塞（含引擎改动需求）；GPU 锁与显存读数。

先把契约与代码写好、跑通、做完分析再返回；**不要中途返回半成品**。客观不可能或锁拿不到时按 §4 返回并如实登记 blocker。
