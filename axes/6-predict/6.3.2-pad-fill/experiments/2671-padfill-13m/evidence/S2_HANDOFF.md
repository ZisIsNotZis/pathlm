# S2 交接状态 · 2671-padfill-13m（时间盒到点时的落盘状态）

- 写入者：本目录当前**只有一个 writer**（后台 detached supervisor，2026-10-14 21:51 起）。
- supervisor 命令：`while :; do WP632_LOCK_MAX_WAIT=900 python3 -u run.py --stage all --out . --seeds 0 1 --steps 1500 --resume; ...; sleep 20; done`
- supervisor bash pid 与子进程 pid 见 `pgrep -af "run.py --stage all --out \."`。
- **完成判据**：`/tmp/wp632_s2_supervisor.log` 出现 `SUPERVISOR: run.py exit 0 - ALL DONE`，且该 python 进程消失；
  届时 `analysis.md`、`results.json::verdicts`、`results.json::L5` 才会由 `finalize()` 生成。
- **不要**在该进程存活期间另起 writer：两个进程会同时写 `results.json`（非原子 `json.dump`）。
  如需接管，先确认进程已退出，再按 `RESUME.md` 的 `--resume` 命令续跑。
- 逐 run 落盘：每完成一个 `(arm, seed)` 即重写 `results.json` 与 `RESUME.md`，并 append `train_log.jsonl`。

## 本次派发已落的证据（均在本目录）

| 交付物 | 状态 |
|---|---|
| `evidence/l0_l0b_probe_s2.txt` | 最终 revision 上重跑：L0 两条绿、gate probe 两条红；sha256 与 `padfill.py`/`run.py` 现文件一致 |
| `evidence/pytest_test_run_s2.txt` | `pytest test_padfill.py test_run_s2.py -v` → 25 passed |
| `mutation_report.json` / `evidence/mutation_run.txt` | 6 条变异全抓红、逐字节还原、还原后全绿；不含逐次耗时 |
| `results.json` | `config.steps = 1500`、64 个 offset、禁墙钟 `_wall_clock_fields` 待 finalize 计算 |
| `run_log.txt` | 墙钟日志（唯一允许出现墙钟处） |
