VERDICT: PASS

# R-3.1.1 第 2 轮 fresh 独立复核（axis3-reviewer）

被审对象：`axes/3-width/3.1.1-uniform-narrowing/experiments/2642-uniform-narrowing-13m/`
（处置提交 `555d4ab` ← `1a77ddf` ← `04be624`；`README.md` = 冻结预登记契约）。
复核时刻：2026-10-14 22:43–22:45 +0800（`date` 实测）。复核者非本项 writer，上下文全新，未读第 1 轮报告（仅从派发摘要知晓 B1/N1/N2 的症状），第 3 项自算独立重做。

## ① 人话摘要

1. 上轮判 FAIL 的那个毛病——把「每省 1% 算力多付多少 bpc」的斜率差写成「减宽略优于砍 FFN」——现在改对了：文本明说该斜率差 0.000105 对应的 Δbpc 差只有 0.0132，落在项目自己的 0.014 噪声带里，**本尺度两只旋钮不可排序**，不再宣称谁赢。
2. 落盘的 `results.json` 与文字一致，且这组数字我自己从锚文件重算过一遍（1.5314 / 1.5709 → 0.0527、0.0395、差 0.0132），不是抄它的。另外两个小毛病（「结果缺失」文案与事实矛盾、判据文件里出现「≈8 min GPU」）都闭合了。
3. 契约没被事后改（README 与 `04be624` 逐字节相同、mtime 早于起跑），判据线没被移动（`VARIANCE_BOUND = 0.014`、P1–P5 定义逐字未变），处置提交只动了本实验目录与 `.scratch/64-axis3-width/`。**结论：PASS**，只剩 5 条不影响通过的措辞/覆盖率瑕疵。

## ② 逐项检查表

| # | 命令 | 真实输出摘要 | 结论 |
|---|---|---|---|
| 1 | `cd …/2642-uniform-narrowing-13m && python3 -m pytest test_2642_uniform_narrowing.py -q` | `24 passed in 6.04s` | 通过（全绿） |
| 2a | `grep -nE "略优\|略便宜\|不落下风\|占前沿优势" analysis.md` | 唯一命中 `analysis.md:146`：「…**不可排序**，既不能说 d_model 减宽不占前沿优势，也不能说它占优势」——这是**对第 1 轮缺陷的引述**（否定式），非方向性宣称 | 非 blocking |
| 2b | 同 pattern，`results.json` | 无命中 | 通过 |
| 2c | 同 pattern，`run.py` | 唯一命中 `run.py:1010`，位于 `if delta_bpc_gap < VARIANCE_BOUND … else:` 的 **else（越带才触发）分支** | 非 blocking |
| 2d（补充） | `grep -nE "优于\|占优\|前沿优势\|更便宜\|更好\|略" analysis.md results.json` | `analysis.md:26`「算力也是真便宜，且在斜率上与另一只旋钮**不可排序**」（「便宜」指 MACs −40.96%，事实量）、`analysis.md:145-146`（引述）、`:128`「不得据此宣称任一方更优」。`results.json` 无命中 | 当前结论句无带内方向性宣称 ⇒ 通过 |
| 3 | 自算：锚 `2641/results_{U4,U2}_s0.json` 的 `eval.clean_bpc` = 1.5314 / 1.5709；`results.json::P2.table['D192-s0'].clean_bpc` = 1.5841 | `Δ_D192 = 0.0527`、`Δ_U2 = 0.0395`、`gap = 0.0132 < 0.014 = True`；斜率复算 0.0527/40.957 = 0.001287、0.0395/28.368 = 0.001392 | 与落盘一致 |
| 3b | `results.json::P5_falsification.b_frontier_advantage` | `"不可排序（本尺度）：… D192=0.001287 vs U2(FFN 旋钮)=0.001392 的差，对应 Δbpc 差 0.0132 < 0.014 噪声带 ⇒ …不可排序；能确定的只是两者互不支配"` | 与自算一致 ⇒ 通过（B1 已闭合） |
| 4 | `grep -nE "min GPU\|分钟\|wall\|秒\|min\b" results.json` | 无命中（`arms_dropped` 已由「≈8 min GPU」改为「排程代价见 ticket 64 §10」） | 通过（N2 已闭合；残留时钟坐标见 N-1，非时间量） |
| 5 | `grep -nE "缺失" analysis.md results.json run.py` | `analysis.md`、`results.json` 均**无**「缺失」；仅 `run.py:1015/1016/1018/1019` 的**未触发分支 fallback 串**保留「U2 斜率缺失」「WD256 结果缺失」 | N1 已闭合（P5 段已按修补后落盘重写：`D128` 未跑如实登记在 `arms_dropped`） |
| 6 | `stat -c '%y' README.md` + `sha256sum README.md` vs `git show 04be624:…/README.md` | mtime `2026-10-14 20:55:45.69 +0800`；hash `c5e97de4…55c3f5`，与 `04be624` 版本**逐字节相同**；`run_log.txt` 首条 lock `21:08:53`、`runs/D192-s0-12k` mtime `21:57:28` | mtime 早于起跑 ⇒ 契约冻结未被破坏 |
| 7 | `git show 555d4ab --stat`、`git diff 04be624..555d4ab -- run.py`、两侧 `grep VARIANCE_BOUND` | `555d4ab` 仅动 5 文件（`analysis.md` 28±、`results.json` 8±、`run.py` 28±、`.scratch/…/REVIEW-3.1.1.md`、`TICKET.md`）；`run.py` diff 全在 P5 聚合段（P5b 带内判据 + `arms_dropped` 文案 + P5 else 文案）；`VARIANCE_BOUND = 0.014` 两侧逐字相同（第 57 行）；P1–P5 判据串 diff 无增删 | 判据线未移动；口径修补在派发授权范围内 |
| 8 | `git diff --name-only 04be624..555d4ab`；`git show 555d4ab --name-only`；`python3 tools/summary.py --check` | 见「④ 口径说明」：一个区间含 24 文件（跨轴2/3.1.5/3.2.3），但**处置提交自身**只含 5 文件；`summary.py --check` → `OK：46 个 project.yaml 全部符合 docs/research-schema.md` | 无越界改动（按处置提交口径）；树校验通过 |
| 9（补充） | `for f in README.md analysis.md results.json run.py`：worktree hash vs `git show 555d4ab:` | 四文件全部 `MATCH`；`sha256(run.py) = 01e49e76…f002e` == `results.json::code_hash_run_py` | 证据绑定到同一 revision；被审产物未被工作区浮动修改 |
| 10（补充） | `grep -nE "略优\|占优势\|优于" axes/3-width/3.1.1-uniform-narrowing/project.yaml` | 无命中（`前沿` 命中均为通用判据措辞） | 上游研究项 `project.yaml` 未复述带内方向性宣称 |

## ③ blocking / non-blocking

**Blocking：无（0 条）。**
逐条对照派发的 blocking 清单：带内方向性宣称 = 无；数字与落盘不符 = 无（自算逐项一致）；契约事后编辑 = 无；判据线移动 = 无；判据文件时间量 = 无（时间量口径命中 0）；越界改动 = 无。

**Non-blocking（5 条，均不影响 PASS）：**

- **N-1（措辞/口径）** `results.json:606`：`arms_dropped.D128` 含时钟坐标「按 **21:50** 判据（见 run_log.txt **21:49:40** 那条…）」。这是对某次决策点与日志行的**溯源坐标**（AGENTS §2.1 允许溯源日期），不是被测量的时间量/时长，故不判 blocking；但判据文件里出现裸时钟仍会让人误读，建议改为指向 `ticket 64 §10`／`run_log.txt` 而不带钟点。
- **N-2（悬空引用）** `results.json:606`：引用「**README §预算递减**」——README 无此节标题（§5 标题为「预算与显存」，砍臂次序在 `README.md:122`）⇒ 该指针不可解析（违反 AGENTS `resolvable`）。建议改为「README §5 预算与显存」。
- **N-3（契约与实现口径差）** `run.py:1002-1006`：新增的「`|Δ_D192 − Δ_U2| < 0.014` ⇒ 不可排序」分支，不在 `README.md:106` 预登记 P5b 的字面条件里（P5b 只写「否则若两臂斜率均 ≥ U2 ⇒ 写『不占前沿优势』」）。该分支方向是**取消宣称**（比原实现更弱），且属派发明确授权的「P5b 带内判据」口径修补；契约已冻结不得回改，故只记**口径差**，建议在 ticket 64 §10 留一句「实现比预登记更保守」的显式说明。
- **N-4（兜底文案）** `run.py:1010-1013`：else（越带）分支仍保留「不占前沿优势／至少一臂优于 FFN 旋钮（Δbpc 差越带）」方向性判词——当前落盘未走该分支（`gap=0.0132 < 0.014`），故不是当前结论句；但若日后补 `D128` 而两臂同向大幅越带，该判词会直接进结论，届时须复核它是否仍被预登记 P5b 支持。
- **N-5（测试覆盖）** `test_2642_uniform_narrowing.py` 中 `grep -nE "b_frontier_advantage|不可排序|delta_bpc_gap"` **无命中**：新增的带内判据无红-绿回归门（24 个用例无一触及）。属新增逻辑无变异保护，建议补一个「构造斜率差 < 带 ⇒ 断言输出含『不可排序』」的用例。

## ④ 未跑项与理由

- **未跑 GPU**：本轮为只读文档/判据复核，不重训、不重评；判据数字以**锚文件 + 落盘 per-arm JSON 自算**为准（已做）。理由：派发未要求重跑，且 GPU 为共享资源、重跑不改变被审文本。
- **未跑 `run.py --aggregate` 复算**：会覆盖 `results.json`（写入被审产物），与「只读被审对象」冲突 ⇒ 未跑，改以独立 Python 自算复现全部关键字段（结果一致）。此项**不计通过**，但 blocking 清单不依赖它。
- **`04be624..555d4ab` 越界扫描的口径说明**：该区间实际含 **24** 个路径（含 `axes/2-depth/*`、`axes/3-width/3.1.5`、`3.2.3`、`.scratch/61-axis2-depth` 等），原因是 `04be624..555d4ab` 之间夹着**其它工作流的中间提交**（`ae1952d`/`7f319f5`/`b0b905f`/`16d8190`）。派发里「该命令应只含本实验目录 + `.scratch/64-axis3-width/`」的预期在此不成立。按**处置提交自身**换算（`git show 555d4ab --name-only` = 5 文件：实验目录 3 + `.scratch/64-axis3-width/` 2）无越界。这是检查方法的口径问题，不是发现的缺陷。

## ⑤ 残留不确定

1. **`D128` 未跑**：P5b 的「不可排序」只在 d=192 一档上成立，斜率是否在更小宽度处反转仍无本地证据（`analysis.md` 已如实列为「需 D128 复核」）。当前结论是"本尺度不可排序"，不是「减宽与 FFN 旋钮等价」，措辞已守住这一点。
2. **噪声带外推**：0.014 带在 `0.0527` 这个量级上是 3.8 倍距，判「可分」稳健；但斜率差的判读是**在带内判不可排序**，其统计效力（n=1 臂、无配对 seed）本就不足以反驳任何一侧，只能算「现有证据不支持排序」。README 对此的口径我认为自洽。
3. **同目录 `run_log.txt` 有未提交修改**（`git status` 显示 ` M`），属过程日志的追加写入，未被列为被审产物；四份被审产物与 `555d4ab` 逐字节一致，故不影响本轮判定，但我未追查该追加由哪个进程产生。
4. **N-1/N-2 的整改属派发方决定**：本轮只读，未改任何被审产物；上述 5 条 non-blocking 是否要求整改，由主 Agent/用户裁定。
