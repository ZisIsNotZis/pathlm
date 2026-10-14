# 派单契约 · 5.2 + 5.3 收尾 writer（fresh 上下文，一个 writer 顺序处理两包）

> 由 axis5-lead 落盘（ticket 66 Phase B）。背景：两包的前序 writer 均被子 Agent **30 分钟硬上限**切断，
> 但两包的 **run 是 detach 起跑的 ⇒ 仍会自行跑完并落盘**。你的任务是**按完成顺序收尾**（一次只动一包，
> 不得两包并行改文件）。纪律：`SCHEMA.md` §3/§6/§10/§14b/§20/§22/§23；`docs/research-schema.md` §11.2/§12.1。

## 边界
- **只写** `axes/5-quant/5.2-weight-quant/experiments/2651-weight-quant-13m/**` 与
  `axes/5-quant/5.3-activation-quant/experiments/2652-act-quant-13m/**`（顺序进行）。
- **禁止写/改**：`axes/5-quant/*/project.yaml`（事实文档由 axis5-lead 写）、其它轴、`docs/**`、`pathlm/**`、
  `train_*.py`、`tools/**`、`tests/**`、任何共享文件。**禁止** `git add/commit`。**禁止**杀他人进程、抢 GPU 锁。
- **只描述已跑出来的数字；没跑的臂一律写「未测」。不得伪造、不得插值。**

## 步骤
1. **判活**：`ps -eo pid,etimes,cmd | grep -E "2651-weight|2652-act" | grep -v grep`；
   `tail -40 <exp>/run_log.txt`。两包都是 **CPU 口径**（`device=cpu`，不取 GPU 锁）。
2. 哪个 run 先出现**完成横幅 / `results.json`**，就先收尾哪个（顺序：先 `results.json` + `table.md`，
   再 `analysis.md`，最后 `JOURNAL.md` + 两项校验命令）。
3. 若某包的 run **已死且无 `results.json`**：先用它自己的 `run_log.txt` 把**已完成臂**的读数落进
   `results.json`（标 `partial: true` + `arms_completed` / `arms_pending`，并注明「数字来自 run_log，
   按臂逐条转写」）——**能救多少救多少**；再只对已完成臂写 `analysis.md`。**不得**补造未跑臂。
   若仍有存活进程：**让它跑完**，用 `≤60s` 的 `sleep` 轮询（**单条 bash 命令总时长 ≤180s**），
   轮询间隙做 CPU 小事（搭表骨架、回读判据）。
4. `analysis.md`（两包各一份）必须含：
   - **人话摘要**（第一段）；表（每臂 `bytes_per_*` / `clean bpc` / `recall_copy` 的 Wilson CI 与 `at_floor`）；
   - **三道对账**：① 字节账与实现一一对应；② `fp16` 臂在**同设备内**逐位复原（`bpc_of` vs `masked_bpc`）
     + 与 GPU 历史锚 1.483569 的**并列**与口径声明；③ 配对 Δ（逐 seed）与 0.014 带的关系；
   - **设备口径声明**：本批为 CPU 口径（原因 = GPU 锁被其它轴长期占用）；**跨设备不构成判据**，判据只用
     同设备同窗口序的配对 Δ；若 `fp16` 的 CPU bpc 恰好等于 1.483569，如实写明「与历史锚在 6 位小数上一致」；
   - **5.3 必须解释 `|Δanchor|`**：若主表评测流的 `n_batches`/`batch_size`/`seed` 与 5.1 的 24×16@seed2 不同，
     显式声明「不同口径、不与 1.483569 直接比」，并把 24×16@seed2 的 `fp16` 值**也跑一遍并列**（约 82 s）；
     若口径相同而仍不一致 ⇒ 列为收口级缺陷、查明或如实登记为「原因未定的口径不一致」并降级结论；
   - 失败与局限（含「单 seed 冻结 checkpoint ⇒ 不宣称结构优劣」「结论只作这些点的描述」、`at_floor` 读法）；
   - **5.3 必须给「离群通道是否迫使部分层回退」的判定**（用 `revert_frac` / `outlier_idx` / `eff_bits` 等活性列）。
5. 跑 `python3 tools/check_tree.py` 与 `python3 tools/summary.py --check`（只读确认没破坏共享校验）。
6. **返回 ≤15 行**（两包各一段）：每包 目录 / 已完成臂与未跑臂 / `bytes_per_*` 与 `clean bpc`（+`recall_copy`）/
   配对 Δ 与越带判定 / `fp16` 逐位复原与历史锚关系 / 5.3 离群判定 / 走完的否证路径 / **结论一句话** /
   未测项与下一步 / 阻塞与口径声明。

预算：≤25 分钟墙钟。到点先保 `results.json`+`analysis.md` 落盘，再返回；**未完成的如实登记**。
