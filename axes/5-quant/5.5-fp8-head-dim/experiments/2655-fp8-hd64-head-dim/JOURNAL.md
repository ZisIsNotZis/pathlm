# 2655 · 过程记录（JOURNAL）

本文件是**过程史**（日期流水，允许正序/倒序），不是权威结论；结论见 `analysis.md`、原始数字见
`results.json`、判据线见 `config.json::criterion`。冲突时以 `analysis.md` / `results.json` 为准。

## Phase 2（评测与判读 · finisher）

时间：2026-10-14 23:30–23:44（+0800）。执行者：WP2 收尾 writer（单一写入者）。
设备：整程 **CPU**，不取 GPU 锁；训练驱动 `pid 2613938`（`hd64-s1` 排队中）同时在跑，未干涉。

1. `23:30:19` 起 **2655 主评测**（detach）：
   `cd <expdir> && setsid nohup python3 hp64_arms.py > f5_finisher_eval.log 2>&1 &`
   ⇒ `[装载]` sha256 `ce40e246614f2f50…`（`computed_at_eval`）、`head_dim=64`、评测尾段 1.2 s 就绪。
2. `23:30:37` 起 **2653 的 `fp16` 臂历史锚口径重跑**（detach，F5 ③ 的等价证据）：
   `python3 axes/…/2653-fp8-vs-int8-13m/run.py --arms fp16 --out-dir <expdir>/f5-anchor-2653-rerun`
   ⇒ `[自检]` 三条 equality 全 True；`[主臂] fp16 bpc=1.483570（158 s，24×16@seed2）`；
   `--arms` 不过滤 `side_arms`，故另跑出 3 个辅助臂（写入该 out-dir，不进 `2653/results.json`）；
   `23:39:38` 完成（9.1 min）。
3. `23:35:35` 起 **12 位精度单点**（detach，同一 `bpc_of` 路径）：
   `python3 <expdir>/f5_anchor_2653_precision.py` ⇒ `1.48357047091`（144.75 s），
   对 GPU 历史锚 `1.483569` 的 `|Δ| = 1.4709e-06`。
4. `23:41:37` **2655 主评测完成**：`wall=663 s`、主臂 3 / 辅助 2、`WP2 全臂一致=True`、
   `partial=False`；逐臂读数：`fp16 1.483031`、`int8_block_g64 1.483140`、`fp8_e4m3 1.484297`、
   `int8_block_g16(scan) 1.495037`、`int8_block_g32(scan) 1.494940`。
5. 落盘 `analysis.md`（人话摘要 + 主表 + 三道对账 + 判据逐条）、`table.md`（脚本生成）、
   `2653/results.json::anchor_single_point_f5`（**新键**，既有键逐键比对未变）+ `2653/analysis.md` §9 追加。
6. `hd64-s1` **未训练完成**：driver `attempt=2` 在 `waiting_lock`（`wait_lock_timeout` 600 s 后
   sleep 20 重试）⇒ 本 run 主表按 `hd64-s0` n=1 出数并显式声明（见 `analysis.md` §5.1）。

## Phase 1（契约 + 训练驱动 · 前一轮）

- `23:08:37` 起训练驱动 `hp64_train_driver.py`（detach、每臂取一次 §20 锁、失败 sleep 20 重试）。
- `23:09:22`–`23:17:13` `hd64-s0` 训练完成：12k 步、等锁 45 s、wall 470.7 s（7.77 min）、
  sha256 `ce40e246614f2f50…`、13 536 768 参数。
- 预登记契约（`README.md`）+ 判据线（`config.json::criterion`，mtime 早于 `results.json`）+
  评测入口 + 训练驱动 + 冒烟（`smoke-plumbing-2610trunk/`，`plumbing_only`，非读数）。
