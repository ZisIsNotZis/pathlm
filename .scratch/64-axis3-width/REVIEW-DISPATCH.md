# 轴 3 Phase B · 复核派单契约（fresh 复核者用）

> 前提：**复核必须派有 shell 的 `worker` agent 承担复核角色**（builtin `reviewer` 无 bash，见 TICKET §10.8）。
> 复核者纪律：只读被审产物 + **只写本 ticket 目录下的一份报告**（`.scratch/64-axis3-width/REVIEW-<项>.md`），
> 不改被审产物、不提交 git、不碰其它轴。复核排在 writer **停止写入之后**（`SCHEMA.md` §14/§14b）。
> 判词必须能 FAIL：跑不动的检查一律记「未跑 = 不计通过」。

## R-3.1.1（被审 = `axes/3-width/3.1.1-uniform-narrowing/experiments/2642-uniform-narrowing-13m/`）

必须实跑 / 自算（不得引用 writer 自报值）：
1. `python3 -m pytest test_2642_uniform_narrowing.py -q`（全绿，且**其中至少有 1 条**是本 WP 期内新加的、能变红的用例——例如 tuple 输出 hook 回归、或等值门的变异）。
2. 自写脚本从该目录的 `results_*.json` 重算：各臂 clean bpc、每 token 等效 MACs、参数量；核对与 `results.json` 的判据字段逐位一致。
3. **契约冻结**：`README.md` 的 mtime 与内容哈希；起跑时刻（`run_log.txt` 第一条 lock acquired / 首个 `runs/*-12k` 目录 mtime）晚于 `README.md` mtime；`README.md` 内的判据线（P1–P5）与 `analysis.md` 里报告的判据线一致（判据线不得被事后改）。
4. **等值门与干预真到达模型**：核 `D192`/`D128` 的实测 `d_model`/`n_heads`/FFN 隐藏维；核 `WD256` 的三档宽度评测确实是**同一 checkpoint 的切片**（而不是三个分别训练的模型）——要求 writer 的切片评测函数可复现：自己跑一次「同一 `model.pt` → 宽度 256/192/128」，核对 `results_*.json` 里的三档读数。
5. **评测确定性**：`--no-train` 重评与首评的 bpc 逐位一致（可自己跑一次，或核 writer 的 `reeval` 产物 + 日志）。
6. **数字保真**：抽 ≥5 个数字（含锚 `U4-s0 = 1.5314` 的引用、Δ 值、MACs、参数）与落盘 JSON / `2641` 的锚文件对照。
7. **口径纪律**：`analysis.md` 的判据段与 `results.json` 不得出现墙钟/吞吐作为判据；墙钟只允许出现在过程日志。
8. **越界扫描**：`git diff --name-only <该项起跑前的提交>..HEAD` 里只允许出现本项实验目录 + `.scratch/64-axis3-width/`（注意：并发其它轴的路径不计本项越界，按路径归属分类并附清单）。
9. `python3 tools/summary.py --check`（期望 OK：46 份）。

blocking（任一 ⇒ FAIL）：数字与落盘不符 / 判据线被事后改 / 「同一 checkpoint 多档」名不副实 / 等值门未真跑或恒真 / 结论越证据（把噪声带内的差异说成已分胜负）/ 本轴范围出现未说明改动。

## R-3.2.3（被审 = `axes/3-width/3.2.3-tied-untied/experiments/2643-tied-untied-13m/`）

1. pytest 全绿（含「解绑开关关闭时前向 == 引擎 stock readout」的等值门 + 能变红的变异）。
2. **复用门（`SCHEMA.md` §9）**：核 `T` 锚（`2641` 的 `U4-s0`）的逐位初始化对照结论是否真跑过（脚本 + 输出），
   以及 config 逐字段对照表；口径（评测批/步数/种子）与锚一致。
3. 自算 Δbpc(`U`−`T`)、Δbpc(`PR`−`T`)，核对 `results.json`；核 0.014 带的使用是否只用于 bpc 口径、率类（prob0 一致性）是否另走 CI（口径不跨量纲借用）。
4. 参数量：`U` 的 +52,736 与 `PR` 的 +14,784 是否与实测参数差逐位一致；「大词表档不可判」是否显式写出。
5. 数字保真抽 ≥5 个；`analysis.md` 首段人话摘要是否存在；口径无墙钟；越界扫描；`summary.py --check`。

blocking 同上；另：「prob0 一致性」若被当作 bpc 同量纲判据使用 ⇒ FAIL（跨量纲借用）。

## 组合复核（时间紧时）

一个 fresh 复核者可同时覆盖 R-3.1.1 与 R-3.2.3，但报告里必须**逐项给判词**（每项一行 `VERDICT: PASS/FAIL`），
且不得用一项的结论替另一项背书。
