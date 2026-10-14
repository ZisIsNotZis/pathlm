# 派单契约 · 5.5「fp8 / 大 head_dim」WP1 + WP2（writer，fresh 上下文）

> 由 axis5-lead 在 ticket 66 Phase B 落盘；派发时把本文件路径交给 worker（`worker` agent，fresh 上下文，单一写入者）。
> 边界：只写 `axes/5-quant/5.5-fp8-head-dim/experiments/**`（可另写 `.scratch/66-axis5-quant/evidence/W-5.5-*.md`）。
> 禁止写/改：`axes/5-quant/*/project.yaml`（事实文档由 axis5-lead 写）、其它轴、`docs/**`、`pathlm/**`、
> `train_*.py`、`tools/**`、`tests/**`、任何共享文件。只读复用 `tools/evalkit/**` / `pathlm/**` / 其它实验目录。
> 不动 git 索引（不 `git add`/`commit`）。需要引擎/工具改动 ⇒ **不要改**，写进返回报告的「阻塞/需求」段。

## 0. 交付
目录：`axes/5-quant/5.5-fp8-head-dim/experiments/2653-fp8-vs-int8-13m/`
回答两个可证伪的问题：

- **WP1**：同一「每位置部署字节」轴上，**fp8（E4M3 / E5M2）** 相对 **int8+块标尺**的质量与字节取舍如何？
  （补 `5.1` 的 fp8 质量缺口：`5.1` 已解析出 fp8 = 1.0 B/值 = 6144 B/位置 = ×0.50，比 int8 块标尺
  1.0625 B/值 = 6528 B/位置 = ×0.53 更省，但**质量未测**。）
- **WP2**：`head_dim → G → 每位置字节 → clean bpc` 的**解析 × 实测**对账。

产物：`README.md`（**运行前**预登记契约）· `config.json`（**运行前**写死 `criterion`）· `run.py` ·
`fp8_arms.py`（语义层，新文件）· `results.json` · `table.md` · `analysis.md`（第一段「人话摘要」）·
`MODEL.md` · `run_log.txt` · `selftest_fp8.py`（可失败单测/变异）· `JOURNAL.md`。

## 1. 权威输入（各读一次）
- `axes/5-quant/5.5-fp8-head-dim/project.yaml`（契约：`method`/`validation`/`objective_criteria`/`falsification`/
  `milestone` 逐字照做；判据带已声明为跨格 0.014 bpc 带，限配对 Δ、≥5 窗口）。
- `axes/5-quant/5.1-kv-quant/project.yaml::conclusion`（块标尺口径的既有事实：int8 块标尺 G=32 = 1.0625 B/值、
  6528 B/位置、Δbpc vs fp16 = +0.000101；fp8 解析 6144 B/位置质量未测）。
- harness 模板（照抄模式，别改）：`axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale/run.py`（GPU 锁 +
  `bpc_of` + 召回 + `results.json` 结构 + `scale_caliber.py` 的部署字节账）与其 `scale_caliber.py`；
  `axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest/run.py` + `config.json`；
  `axes/5-quant/5.1-kv-quant/experiments/2619-kv-budget-chart/run.py`。
- 纪律：`SCHEMA.md` §3/§6/§7/§8/§10/§14/§20（GPU 锁）/§22/§23；`docs/research-schema.md` §12.1/§11.2。

## 2. 可复用 harness API（只读 import）
把 `2621` 目录加进 `sys.path`，用 `_load_module` 载入其 `run.py` 为模块 `b21`（照抄 `2632/run.py`）：
`b21.b19.load_trunk(cfg, dev)` → `(model, ck)`；`b21.b19.bpc_of(logits_fn, eval_arr, T, n_batches, batch_size,
window_seed, dev)`；`b21.b19.masked_bpc(...)`（独立第二路径）；`b21.b19.build_tasks(cfg, smoke=False)`；
`b21.b19.pooled_accuracy(fn, tasks["copy_by_pos"], dev)` → `{accuracy,n,wilson95}`；`b21.b19.floor_threshold(n, chance)`。
`cfg` 键：`model`/`data`/`eval`/`tasks`/`checkpoints`（从 `2621/config.json` 抄成你自己的）。
**KV 量化的现成算子**：`tools/evalkit/kv_quant.py::quantize_kv`（有 `scheme` 分支：`per_value` / `block`(G) /
`block_tok` / `kivi`）——**只读复用**，`block(G=32)` 即 `5.1` 的口径。fp8 需你自己在 `fp8_arms.py` 里实现
（假量化往返：`x → float8 舍入 → 反舍入`，用 `torch.float8_e4m3fn` / `torch.float8_e5m2`；若 CUDA 不支持
该 dtype 的运算，用纯 CPU/逐元素模拟实现并**如实登记实现口径**）。前缀不变性仪器可复用
`axes/5-quant/5.1-kv-quant/experiments/2631-kv-budget-causal/causal_check.py`（`prefix_invariance_of_logits`）。
数据 `pathlm.data.load_enwik8_full`（不入库）。chance = 0.0156。
trunk checkpoint：`axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt`，
sha256 `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2`，clean bpc 历史锚 **1.483569**
（`window_seed=2, n_batches=24, batch_size=16, T=256`；模型 12 层 × d256 × 8 头 × head_dim 32 × mlp6）。

## 3. WP1 设计（单因素：KV 的 8 位存放格式）
- 臂（KV 量化，权重/训练不动，`params_added = 0`）：`fp16`（基线，必须与 1.483569 逐位一致）·
  `int8_block`（块标尺，G=32，= `5.1` 口径对照）· `fp8_e4m3` · `fp8_e5m2`。
- **横轴**：每位置**部署字节** = `2·L·n_kv_heads·head_dim·(每值字节)·(kept/T)`。fp8：**实测它是否真能免标尺**
  （若你需要分组标尺才能过质量关，就如实报 `1 + 2/G` 并宣布「免标尺」前提失败）；int8 块标尺：
  `1 + 2/G`（G=32 ⇒ 1.0625）。**每个数值都给出解析式与逐项因子**，并与 `5.1` 的 6528 B/位置对账。
- **两条质量腿**：clean bpc（越低越好）+ 合成长程召回（copy / induction / var_recall，Wilson 95% CI，`at_floor`）。
- **护栏**：① 干预活性（K/V 真被改：`changed_frac`/`rel_err`/`n_scales`）；② 非量化臂逐位不变（与历史锚 +
  `masked_bpc` 独立路径逐位一致）；③ **严格前缀不变性两条**（改未来 token / 改同批邻居 ⇒ 过去位置的
  反量化值与 logits 不变）必须为 True；④ **变异测试**（破坏量化/标尺/位宽 ⇒ 断言报红；未抓住先修单测）。
- **判据带**：主表 `window_seed=2`（与历史可比）+ 另跑 `seeds = [3,4,5,6]`（可缩小 `n_batches` 至 8）⇒ 5 个窗口
  满足「配对 Δ、≥5 窗口」⇒ 用预登记的 **0.014 bpc 带**判 `max|配对 Δbpc|` 越带/不越带。**若做不到 ≥5 窗口**，
  必须在 `config.json::criterion` 显式声明「本格自测在力带 = 单臂跨窗口 bpc 极差（给数值）」并声明两带不混用。

## 4. WP2 设计（head_dim → G → 每位置字节）
- 用**解析**式给出 `head_dim ∈ {32, 64}` 时 `G ∈ {8,16,32,64}` 的每值字节 `1 + 2/G` 与每位置字节
  （`head_dim` 越大 G 上限越大 ⇒ 省幅趋近 50%）；把 `head_dim=32` 时省幅在 **46.875%** 处封顶（G=32）算清。
- **实测腿（必须真跑，不许只写解析）**：在 `head_dim=64` 上重跑同一组臂。构造方式：用引擎现成 flag 训练 1 个
  `--heads 4`（d=256 ⇒ head_dim=64）的 13M 模型，命令与 2616 同配方：
  `python3 train_depth_ar.py fp8-hd64-s0 --d 256 --layers 12 --heads 4 --mlp-mult 6 --dense --steps 12000
   --batch-size 32 --seed 0 --lr 0.001 --eval-batches 24 --out-root <expdir>/runs`
  （先确认能力面：`grep -n "ValueError\|互斥" pathlm/depth_ar.py`，并 `--smoke` 一次；若引擎不支持，**不要改引擎**，
  如实报「WP2 实测腿受阻 + 需求」）。然后在该模型上跑 `{fp16, int8_block G=64, fp8_e4m3}` 三臂，
  与解析对账（`head_dim=64` 的每位置字节应比 `head_dim=32` 更接近 50%）。
  训练墙钟 ≈9 min（**只作过程读数，不进 `cost_model`、不进判据**）；`cost_model` 只写等效理论量。
- **WP2 降级路径（若 GPU 锁 25 min 内拿不到或训练未交付）**：把 WP2 交付改为 ①`head_dim ∈ {32,64}` 的**解析表** +
  ②`head_dim=32` 上 **G ∈ {8,16,32} 的实测扫描**（CPU，零成本，用来验证 `2/G` 标尺摊薄项与每位置字节的对账）+
  ③显式登记「`head_dim=64` 的**实测腿未完成**（等锁/预算），故 5.5 的 milestone 中『大 head_dim 省幅与解析一致』
  只有解析 + head_dim=32 的实测支撑，`head_dim=64` 不外推」。**不得**把未跑的训练数字写成已测。
- 验收：`每位置字节随 head_dim/G 的实测与解析一致`；差异必须有解释；质量不越带（用同一 0.014 带）。

## 5. 算力纪律（**默认 CPU 口径**，见 ticket 66 §13 F5）
单卡 4090 被多轴共享，`/tmp/pathlm_gpu.lock` 长期被其它轴的训练占用（实测同一 holder 持有 >13 分钟、attempt 等锁 600s 超时）。
**WP1 一律走 CPU 口径，不取 GPU 锁**（纯 CPU 工作，`SCHEMA.md` §20 不适用）：
- 已实测成本（证据 `.scratch/66-axis5-quant/evidence/cpu_eval_timing.py`）：`load_model=1.2s`、**16×256 一批 = 3.4 s**
  ⇒ 主表 24 批 ≈ 83 s/臂、配对窗 4×8 批 ≈ 110 s/臂 ⇒ **4 臂 ≈ 13 min CPU**，可接受。
- **全部臂必须同设备**（同一 CPU 口径）；`results.json` 记 `device: "cpu"` 与 `caliber_note`：
  **跨设备（CPU vs GPU 历史锚 1.483569）不构成判据**；判据用同设备、同窗口序的**配对 Δ**
  （`pathlm.data.batch` 的数据流与设备无关 ⇒ 配对有效性不受影响）；把 `fp16` 臂的 CPU bpc 与 GPU 历史锚**并列**
  并显式声明该口径差异（设备不同 ⇒ 不宣称「逐位等于历史锚」，只宣称「同设备内部逐位/同口径」）。
- 轮询/判活用 `run_log.txt` + 完成横幅；**detach 起跑**，单条 bash 命令总时长 ≤180 s，轮询 `sleep` ≤60 s。
- **WP2 的训练腿**（`head_dim=64`）是唯一需要 GPU 的部分：写成**独立 detach 作业**排队（`nohup` + 等锁 + 后台 + 短轮询），
  预算 ≤25 min；等不到锁就按下面「WP2 降级」处理，**不得空转等锁、不得杀他人进程**。
  若走 GPU：照抄 `2632/run.py` 的 `GpuLock`（`mkdir` 抢锁 + 陈旧锁自愈 + `holder_pid`/`heartbeat` + `trap` 释放），
  `results.json` 记 `gpu_lock = {holder_pid, wait_seconds, free_mib_before, free_mib_after, release_ok, held_whole_run}`
  + `peak_mem_mib_this_run`，显存礼貌线 ≤5 GiB。

## 6. 验收（自检先满足）
- `README.md` = 运行前预登记契约：命题、单因素声明、指标表、判据线逐字、**既有做法 prior art**
  （fp8 格式 `arXiv:2209.05433`；DeepSeek-V2 MLA `head_dim=128` 见 `arXiv:2405.04434`；vLLM/TensorRT-LLM 的
  fp8 KV cache = `记忆·未核`——**只作设计方向，不作判据**；本项不重造 fp8 格式/推理框架）、对照与方差、
  成本、数据与脚本清单、失败决策树、产出。`config.json::criterion` 运行前写入（mtime 早于 `results.json`）。
- `analysis.md`：人话摘要 + 表 + **三道对账**（① 字节账与实现/`5.1` 一致；② fp16 与历史锚/独立路径一致；
  ③ 前缀不变性两条 + 配对 Δ 与带的关系）+ 失败与局限（含「单 seed 冻结 checkpoint ⇒ 不宣称结构优劣」
  「结论只作这些点的描述」「fp8 免标尺与否的实况」）。
- `gaps`：未测项写成「下一步最小实验 + 等效算力成本」。
- 落盘后跑 `python3 tools/check_tree.py` + `python3 tools/summary.py --check`（只读确认没破坏共享校验）。

## 7. 返回给 lead（≤15 行）
实验目录路径；臂清单与每臂 `bytes_per_position` / `clean bpc` / `recall_copy`；**fp8 是否真免标尺**；
前缀不变性两条是否为 True；配对 Δ 与所带结论；`head_dim=64` 的解析 vs 实测对账结果；`fp16` 逐位复原是否成立；
走完的否证路径与结果；**结论一句话**（该尺度该格式值不值）；未测项与下一步；阻塞（含引擎改动需求）；
GPU 锁与显存读数、训练作业是否交付。

先把契约与代码写好、跑通、做完分析再返回；**不要中途返回半成品**。客观不可能或锁拿不到时按 §5 返回并如实登记 blocker。
