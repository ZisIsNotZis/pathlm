# 2657 · 过程记录（JOURNAL）

本文件是**过程史**（日期流水），不是权威结论；结论见 `analysis.md`、原始数字见 `results.json`、
判据线见 `config.json::criterion`。冲突时以 `analysis.md` / `results.json` 为准。

## Phase 1（契约 + 彩排 + 绑定 + 起训）

时间：2026-10-20 22:55–23:05（+0800）。设备：契约与彩排在 **CPU**；训练腿占 **GPU**（按 §20 取
`/tmp/pathlm_gpu.lock`，单臂取放一次）。

1. 先确认格名：本仓 5 轴目录是 `axes/5-quant/`，本格是 `5.5-fp8-head-dim`（不是 `5-memory/5.5-precision`）。
   分诊表 `atlas/triage.md` §2b 给本格点出的单一动作 = **`head_dim`=128 腿**（3.0 档首位）。
2. 立新实验目录 `2657-fp8-hd128-head-dim`（`2656` 已被 `5.4-low-rank` 占用，取未被占用的下一个号），
   写 `README.md`（预登记契约）+ `config.json`（判据线与 `analytic` 字节账；`analytic` 由只读复用的
   `axes/5-quant/5.5-fp8-head-dim/experiments/2653-fp8-vs-int8-13m/fp8_arms.py` 生成，不手抄）。
3. **先跑彩排再绑定**（fail-loud：契约先于读数，但代码不许带病绑定）：用随机权重 checkpoint
   （`/tmp/hd128-smoke/random-hd128.pt`，不入库）跑两轮——
   ① `--smoke`（2 臂、1 批）验仪器自检；
   ② 把 `config.json` 复制到 `/tmp` 并把 `n_batches` 调小，跑**全部 5 臂 + 全表 + `paired_delta_arm_vs_arm`**；
   ③ 再用同一条 `/tmp` 配置跑 `--seed-leg second_seed` 验第二 seed 腿分支。
   三轮全部通过：实测/解析逐臂一致（12288 / 6240 / 6144 / 6336 / 6528）、前缀不变性 True/True、
   `int8` 块标尺 **G=128 与 G=64** 各自与 `2632/block_arms` 前向逐位相等。
4. `python3 tools/prereg_bind.py <expdir> --contract README.md config.json hd128_arms.py hd128_train_driver.py`
   ⇒ `bound_at = 2026-10-20T23:00:46+08:00`；随后立刻把 5 份契约文件提交并推送（`0ef3f1e`），
   读数尚未产生 ⇒ 「契约先于读数」有独立定序证据。
5. `23:01:04` 起训练驱动（detach）：`hd128-s0`（seed 0）取锁 0 s（`gpu free 10389 MiB`）；
   驱动按 §20.1 每臂取放一次锁、臂间 60 s 窗口。

## Phase 2（评测与判读）

时间：2026-10-20 23:11–23:47（+0800）。设备：评测整程 **CPU**（不取 GPU 锁）；训练腿占 GPU。

1. `23:10:12` `hd128-s0` 完成（rc=0、墙钟 547.7 s、sha256 `a41d5eaac208ed5f…`）。驱动按 §20.1 进入 60 s 臂间窗口，
   随后 `hd128-s1` 起训时 **GPU 锁被另一条研究线占着**（先 `2681-int8-equalbytes-frontier-13m/train_arms.sh`、
   再 `2681-anchor-recall-seeds23-13m/train_arms.sh`）⇒ attempt 1 等锁 600 s 超时、`sleep 20` 重试，
   attempt 2 在 `23:30:18` 拿到锁（`wait_seconds = 525.024`、`gpu free 3866 MiB`），`23:40:40` 完成
   （墙钟 622.3 s、sha256 `70ca528def94e2be…`）。**未杀他人进程、未降级配方。**
2. `23:11` 起 seed 0 主腿评测（detach，CPU）：`python3 hd128_arms.py` ⇒ 仪器自检
   `bpc_of = masked_bpc = 1.466302`（逐位相等）、fp32 前向 equality True、int8 块标尺 **G=128 / G=64** 与
   `2632` 前向逐位相等 ⇒ 主臂 3 / 辅助 2 全出数，`wall = 242.02 s`、`partial = False`、
   `wp2_reconciliation.all_consistent = True`。
3. `23:41` 起 seed 1 第二腿评测：`python3 hd128_arms.py --seed-leg second_seed --out-dir hd128-s1`
   ⇒ 仪器自检 `1.480934` 逐位相等、其余同 ⇒ `wall = 364.85 s`、`partial = False`、`all_consistent = True`
   （评测期机器上有并发作业，故比 seed 0 那趟慢，不影响配对口径）。
4. 两腿读数与判读写进 `analysis.md`；`project.yaml` 加 WP3、两条 `cost_rows`、`gaps`/`roadmap`/`conclusion_review`；
   `atlas/cost-table.md` 与 `atlas/repro.md` 按生成器重跑（手工产物不入库）。
