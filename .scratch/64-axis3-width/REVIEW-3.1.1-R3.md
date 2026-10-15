VERDICT: PASS

人话摘要：
- D4 两条必修都真修了：悬空引用改成能解析的指针（README §5 + `README.md:122` 行内容确为「砍臂次序」），新增的「带内 ⇒ 不可排序」判据配了会变红的单测（我造了两个 mutant，实测分别把带内门与越带门打红）。
- 判据线没有被移动：`VARIANCE_BOUND=0.014` 原样，README 第 4 节 P1–P5 定义 git diff 为空（事后未编辑），results.json 只改了 `arms_dropped` 文案、`code_hash_run_py` 与 `generated_at`；P1–P5 其余字段与 `555d4ab` 版本语义等价。
- 数字与落盘一致：D192 的 Δbpc=1.5841−1.5314=+0.0527、算力节省 40.957%、斜率 0.001287，与 2641 锚和 results.json 的 P1/P2 自算吻合；`code_hash_run_py` == sha256(run.py)。
- 无 blocking，PASS 绑定到 `64dbb16` 的产物字节（当前 HEAD 已前进到 `e5613a9`，但该提交不碰 2642，实验目录工作区干净、与 `64dbb16` 逐字节相同）。

审计对象：revision `64dbb16`（2642 实验目录 = `run.py` / `results.json` / `test_2642_uniform_narrowing.py`）。
本轮为 fresh 独立复核（第 3 轮 / D4 重核），未看过第 1、2 轮报告；未修改被审产物、未提交 git（`git diff --cached --name-only | wc -l` = 0）。

## 逐项检查表

| # | 检查项 | 实跑命令 | 真实输出摘要 | 结论 |
|---|---|---|---|---|
| 1 | 单测全绿、用例总数 | `python3 -m pytest test_2642_uniform_narrowing.py -q`（实验目录内） | `27 passed in 6.44s`（另一独立次 `27 passed in 11.81s`） | PASS：27 例（24 旧 + 3 新）全绿 |
| 2① | 指针节标题存在 | `grep -n '^## ' README.md` | `118:## 5. 预算与显存` —— 字段引的 `§5「预算与显存」` **存在** | PASS |
| 2② | 引的行内容＝砍臂次序 | `sed -n '122p' README.md` | `- 预算不够时砍什么（SCHEMA.md §8）：先砍 **seed 数**（seed 1 一律不跑），再砍 **WP2**…**不砍判据线、不砍步数**` | PASS：确是砍臂次序条款 |
| 2③ | 残留裸钟点坐标 | `python3 -c "re.findall(r'[0-9]{1,2}:[0-9]{2}', arms_dropped.D128)"` | `CLOCK_HITS: []`；旧串 `预算递减` 亦已消失 | PASS。另：新指针引 `run_log.txt` 中 `接手者调度调整`，实测存在于 `run_log.txt:20` |
| 3 | N-5 红-绿门（gate probe，关键） | 见下方「mutant 证据」 | M1：`1 failed, 2 passed`（`test_p5b_in_band_cancels_directional_claim` 红）；M2：`1 failed, 2 passed`（`..._better` 红）；原目录复跑 `27 passed` | PASS：两组门都实测可红 |
| 4 | 判据线未被移动 | `git diff 555d4ab..64dbb16 -- <exp>/run.py`；`-- README.md` | run.py 仅两处 hunk：新增 `p5b_frontier_verdict` 函数（+26 行，含 docstring/注释）+ 调用点改写 + `arms_dropped` 文案；`README.md` diff **空**；`VARIANCE_BOUND = 0.014` 行未在 diff 中（run.py:57 原样） | PASS |
| 5 | N-3「更保守」方向仍在 | 读 `run.py:756–778` + `README.md:98–117` | 带内分支（run.py:771–775）返回串只含「不可排序（本尺度）…两者互不支配」，**无**「占/不占优势、优于、不落下风」；越带分支（run.py:777–778）才允许方向性判词。README §4 的 P5b 预登记**没有**带内分支 ⇒ 实现依旧比预登记更保守（N-3 维持，按要求不动） | PASS（记录不动） |
| 6 | 数字与落盘一致 / 语义等价 | 自算 vs `results.json::P1/P2/P5` | D192 Δbpc = 1.5841−1.5314 = **+0.0527** ✓；`macs_cut_pct` = 1−4364928/7392768 = **40.957** ✓；斜率 = 0.0527/40.957 = **0.001287** ✓；U2 斜率 = 0.0395/28.368 = **0.001392** ✓；`delta_bpc_gap` = |0.0527−0.0395| = **0.0132 < 0.014** ⇒ 带内 ⇒ 不可排序 ✓。以当前 `run.py` 实调 `p5b_frontier_verdict(...)` 得串与 `results.json::P5_falsification.b_frontier_advantage` **逐字相等**（`EQ: True`）。`results.json` 对 `555d4ab` 的 diff 仅 `arms_dropped` + `code_hash_run_py` + `generated_at` 三行 ⇒ P1–P5 其余字段语义等价。`code_hash_run_py` = `0b39c1bd…9ce2` == `sha256(run.py)`（MATCH: True） | PASS |
| 7 | 越界扫描 | `git show --name-only 64dbb16`；`python3 tools/summary.py --check` | 恰好 5 文件：`axes/…/2642-uniform-narrowing-13m/{results.json,run.py,test_2642_uniform_narrowing.py}` + `.scratch/64-axis3-width/{TICKET.md,supervise_arms.sh}`；校验 `OK：46 个 project.yaml 全部符合 docs/research-schema.md` | PASS |

## mutant 证据（N-5 gate probe，全程在 /tmp，未改仓库）

构造：`cp run.py /tmp/mutant/run.py`；测试文件同拷并把 `HERE` 固定为 `/tmp/mutant`、`ROOT` 固定为 `/home/z/vibe/pathlm`（否则 HERE 变了 `pathlm` 导入路径失效）。

- **M1**（把带内返回串改成方向性措辞 `d_model 减宽占前沿优势…`）：
  `cd /tmp/mutant && python3 -m pytest test_2642_uniform_narrowing.py -q -k p5b`
  → `1 failed, 2 passed, 24 deselected`；红例 `test_p5b_in_band_cancels_directional_claim`，断言 `'占前沿优势' not in txt` 抛 `AssertionError`。
- **M2**（把越带分支的 `worse = all(...)` 写死成 `worse = True`）：
  同命令 → `1 failed, 2 passed, 24 deselected`；红例 `test_p5b_out_of_band_keeps_directional_claim_better`，断言 `'至少一臂优于 FFN 旋钮' in txt` 失败。
- 复原确认：原目录 `python3 -m pytest test_2642_uniform_narrowing.py -q` → `27 passed in 6.44s`。

⇒ 三个新用例不是空断言：带内门的「防方向性措辞」与越带门的两条方向分支都各自被真实反例打红。

## blocking

无。

## non-blocking（带 file:line）

1. `run.py:761`：新增 docstring 里混入外文单词 `beide`（「把 beide 相减得到的 0.000105」），应为「两者」。不影响判据，纯文案。
2. `results.json:606`：新指针含**行号** `README.md:122`，当下可解析且内容正确，但 README 上方增删行即失效；稳定锚是 `§5「预算与显存」` 标题，行号宜降级为辅助（或改成不依赖行号的措辞）。
3. `results.json:606` 语义松配：被引的 README §5 条款列的是「先砍 seed 数、再砍 WP2」，而 D128 是**按 GPU 锁排队**放弃的；指针现在可解析且指向真实的砍臂次序，但这一刀的直接依据在新引的 `run_log.txt:20`（`接手者调度调整`，实测存在）而非 README。建议后续把「该条不是臂清单驱动」写明，避免读者以为 README 授权砍 D128。
4. N-1 残留复核：`grep -oE '[0-9]{1,2}:[0-9]{2}' results.json` 只剩 `results.json:619` 的 `generated_at: "2026-10-14T23:22:48+0800"`（溯源元数据，`SCHEMA.md` §3 允许）；判据相关字段已无裸钟点。按裁决登记不动。
5. revision 漂移（不影响本次结论）：复核期间另一轴在 `23:23:43` 提交 `e5613a9`，HEAD 已不是 `64dbb16`。`git diff --name-only 64dbb16..HEAD` 不涉 2642；实验目录 `git status --porcelain` 为空 ⇒ 本 PASS 所验产物字节 == `64dbb16` 版本。
6. N-4（越带分支文案）按要求未改：README 的 P1–P5 表述 diff 为空，run.py 越带分支文案原样（仅改由 `p5b_frontier_verdict` 承载、阈值改为形参 `bound` 且默认值同为 `VARIANCE_BOUND`）——判据线未动。

## 未跑项与理由

- 未重跑训练/评测臂（12k 训练 + 逐深度评测）：需 GPU 且本轮只审 D4 处置（文案 + 判据封装 + 单测），数字一致性已由**落盘原始读数自算**核对（`results_D192_s0.json`、2641 的 `results_U4_s0.json` / `results_U2_s0.json`），不需要重算。
- 未跑 `python3 -m pytest tests/ -q`（共享底座）：本轮改动只在单一实验目录内的实验文件与单测，未触共享底座；按 `SCHEMA.md` §22 的写入边界，越界项已由 `git show --name-only` 覆盖。
- 未核 WD256 三档原始读数（P3）：D4 处置未触及 P3 相关代码，且 `results.json` 对 `555d4ab` 的 diff 证明 P3 字段未变。

## 残留不确定

- README §4 的 P5b 预登记与实现（多出的「带内 ⇒ 不可排序」分支）之间的口径差是 Main 已裁决「记录不动」的 N-3；本 PASS 只核其**方向**（更保守、带内不出方向性判词）成立，不判该口径差本身是否应回写预登记。
- 行号型指针（non-blocking 2）在下一个改动 README 的提交里会静默失效；本轮无此提交，故不影响 PASS。
