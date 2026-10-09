# M1 三步线的评测锁口径流水（从格目录事实文档迁出的过程史）

> 这是一份**日志**（`.scratch/doc-log/`，允许日期流水 / 已废口径）。三个格目录的事实文档
> （`axes/M-carrier/M1-carrier-type/experiments/{2633,2634,2635}-*/analysis.md` §20）只留
> 「**以本格 `lock_accounting.json` 为准 + 指针**」的口径；**逐次读数与跨格流水记在这里**
> （一格一家：跨格事实不在某一格的事实文档里）。与产物冲突时**以产物为准**。
>
> 值取自各提交里的 `lock_accounting.json`（git 可复核）；同一提交里三格可能各有一行。

| 提交时间（提交） | 格 | 驱动 | `mode` | 墙钟 s | 峰值 MB | 前后空闲 MiB | 备注 |
|---|---|---|---|---:|---:|---|---|
| 2026-10-10T01:32（提交 6cde724） | 2633-carry-seed2 | eval_concurrent.sh | section_20_1_2_short_job_exception | 174 | 701.6 | 15,752 → 15,286 | 首次并发评测（等锁超时后按 §20.1.2 插队） |
| 2026-10-10T01:32（提交 6cde724） | 2634-carry-realtext-13m | eval_concurrent.sh | section_20_1_2_short_job_exception | 79 | 317.9 | 14,654 → 15,752 | 同上（**未越 ≤120 s 门槛**） |
| 2026-10-10T01:32（提交 6cde724） | 2635-recursive-carrier-13m | eval_concurrent.sh | section_20_1_2_short_job_exception | 11 | 286.8 | 15,286 → 15,286 | 同上（**未越门槛**） |
| 2026-10-10T02:20（提交 3a3d33b） | 2633-carry-seed2 | eval_concurrent.sh | section_20_1_2_short_job_exception | 252 | 701.6 | 15,028 → 16,610 | 驱动加 §9 前置门后重跑 ⇒ **超 ≤120 s 门槛**，登记为一次性并发例外 |
| 2026-10-10T02:20（提交 3a3d33b） | 2635-recursive-carrier-13m | eval_concurrent.sh | section_20_1_2_short_job_exception | 10 | 286.8 | 15,023 → 15,023 | **未越门槛** |
| 2026-10-10T02:42（提交 4e445a5） | 2633-carry-seed2 | eval.sh | section_20_lock | 231 | 701.6 | 16,611 → 16,611 | 首次走整程持锁（`run.py` 进程内计时 176 s） |
| 2026-10-10T03:02（提交 feaa87a） | 2633-carry-seed2 | eval.sh | section_20_lock | 270 | 701.6 | 16,611 → 16,611 | P1 修复后重跑（进程内 214 s） |
| 2026-10-10T03:36（提交 580b60c） | 2633-carry-seed2 | eval.sh | section_20_lock | 340 | 701.6 | 15,804 → 15,802 | 六驱动统一走 `record` 后复跑 |
| 2026-10-10T03:36（提交 580b60c） | 2634-carry-realtext-13m | eval.sh | section_20_lock | 115 | 317.9 | 15,804 → 16,193 | 同上 |
| 2026-10-10T03:36（提交 580b60c） | 2635-recursive-carrier-13m | eval.sh | section_20_lock | 22 | 286.8 | 15,804 → 15,804 | 同上 |
| 2026-10-10T03:55（提交 e9f74f3） | 2633-carry-seed2 | eval.sh | section_20_lock | 250 | 701.6 | 17,670 → 18,059 | `record()` 加「跨文件读回断言」后复跑 |
| 2026-10-10T03:55（提交 e9f74f3） | 2634-carry-realtext-13m | eval.sh | section_20_lock | 108 | 317.9 | 18,054 → 17,668 | 同上 |
| 2026-10-10T03:55（提交 e9f74f3） | 2635-recursive-carrier-13m | eval.sh | section_20_lock | 19 | 286.8 | 18,058 → 18,054 | 同上 |

- §20.1.2 的字面门槛是「≤2 min（120 s）」：上表**超门槛的只有 2633 的两条**（174 s 首次并发、
  252 s 加 §9 前置门后）；2634 的 79 s 与 2635 的 11/10 s **都未越门槛**（此前 2634/2635 的
  §20 曾写「那两次都超门槛」，属**跨格误植**，已删）。
- 事实文档正文曾写死「231 s」一个数字，而同一格不同次运行有两份不同产物（231 s = 02:42 那次、
  270 s = 03:02 那次）⇒ 正文复制数字必然与**当时的**产物对不上（复核 P2-3 的由来）。处置：
  正文改为**纯指针**，数字只留产物与本节。
