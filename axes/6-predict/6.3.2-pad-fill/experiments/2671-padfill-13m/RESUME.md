# RESUME · 2671-padfill-13m（S2 续跑状态）

## 当前事实

- **10 个 run 全部训完并逐 run 落盘**：`ar` × 2 seed + 4 个填空臂（`fill_bidir_r10/r25/r50`、`fill_causal_r25`）× 2 seed；`results.json` 逐 run `done=true`、每 run 64 offset。
- **已完成**：`results.json` 的 `L0 / L0b / L1 / L2 / L3 / L4 / L5 / verdicts` 与 `analysis.md` 已生成（`finalize` 跑通、退出码 0、`_wall_clock_fields=[]`）。
- **已完成但降级（L5 确定性）**：本次收尾 `finalize` 的 L5 确定性子项为 `eval_rerun_identical=None`、`short_train_20step_checksum_identical=None` —— 最终 revision 未能取得 GPU 锁独立复跑（锁被同机 6.3.1 `wp6.3.1-mdiff` 占用，holder pid 2605023）。**判：确定性未通过 / 未验证**（不得写 PASS）。同 revision 更早一次取得锁的行记录于 `run_log.txt`：`eval_rerun_identical=True`、`short_train_identical=False`；该子项在两次运行间 True/False 抖动（浮点逐位相等判定，GPU 归约顺序敏感）。
- **未完成**：L5 确定性未在最终 revision 的锁窗口内独立复跑（GPU 锁竞争）。
- 过程事件（收尾者终止本 WP 遗留的 detached supervisor 循环、run.py 收尾修补清单）见 `run_log.txt`，不在此复述。

## 下一个确切命令

```bash
cd /home/z/vibe/pathlm/axes/6-predict/6.3.2-pad-fill/experiments/2671-padfill-13m
WP632_LOCK_MAX_WAIT=300 python3 -u run.py --stage all --out . --seeds 0 1 --steps 1500 --resume
```

- 该命令在 **GPU 锁空闲**时重算 L5 确定性并重写 `analysis.md` 与 `results.json`；不重训任何 run（gate + finalize 秒级，L5 需一次短 GPU 取锁）。
- `--resume` 跳过 `results.json` 中 `done=true` 的 run；某 seed 的填空臂全部完成时**不重训 AR**（配对读数已在 `results.json`）。
- GPU 按臂取放锁（`gpu_lock.py`）；锁被占则不杀他人进程，L5 子项如实记为未取得。

> **§20.1 让出窗口（Phase D 补记，axis6-lead）**：本实验的 driver 在臂间只间隔约 1s 重取 GPU 锁，**未**留 §20.1 要求的 ≥60s 让出窗口（Main 由其它轴的饥饿观测到）。该偏离只影响其它轴的等待时间，**不影响本目录任何判据**（本轴所有判据都是等效理论量、禁墙钟）。**若重跑本 driver：须在臂间插入 `sleep 60` 再取锁**；本目录已落盘的数字与 `MODEL.md` 的 sha256 绑定不回改。
