VERDICT: FAIL

# REVIEW-3.1.1.md — 轴 3 · 研究项 3.1.1 · fresh 独立复核（R-3.1.1，有 shell）

> 复核者 = fresh 独立上下文 subagent（**有 bash**），cwd `/home/z/vibe/pathlm`；只读被审产物，只写本文件。
> 被审对象 = `axes/3-width/3.1.1-uniform-narrowing/experiments/2642-uniform-narrowing-13m/`（提交 `04be624` + `1a77ddf`）。
> 契约细节 = `.scratch/64-axis3-width/REVIEW-DISPATCH.md` §R-3.1.1。命令超时 240s；时间盒 ≤25 min。全部命令**实跑**，输出见下。

## ① 人话摘要（3 行）

1. **执行层检查全过**：pytest `24 passed`（含能抓红的 T3a–e 变异门与 tuple-hook 回归门）；我自算的 Δbpc / MACs 节省% / 每 1% MACs 代价 / 三档 bpc / 参数解析，与落盘 `results.json` 与 `results_*.json` **逐位一致**（含 1.5314 / 1.5841 / 1.6327 / 1.6178 / 0.0527 / 0.0486）；预登记 `README.md` 未被事后编辑；`1a77ddf` 的 P5 修补**没有移动判据线**；「同一 checkpoint 三档」为真（我在 CPU 上独立复跑 w=192/w=128 得 1.6327/1.7301，与落盘同）。
2. **但判 FAIL（1 条 blocking）**：`analysis.md` 与 `results.json` 把「每 1% 算力代价」的斜率差（D192 0.001287 vs U2 0.001392）写成**方向性结论**（「整体减宽比砍 FFN 略便宜 / 略优于 / 不落下风」，`results.json` 更写成「占前沿优势」），而**该斜率差所依赖的两个 Δbpc 之差 0.0527−0.0395 = 0.0132 < 本实验自订的 0.014 噪声带**——即把噪声带内的差异说成已分胜负，且 `results.json` 的强度还高于 `analysis.md` 自身（后者写「不能判…需 D128 复核」），两处自相矛盾。
3. **另有 2 条非阻塞**：① `analysis.md:139-140` 仍写「脚本按设计输出『D192/D128 结果缺失』」，与 lead 修补后落盘的 `results.json:608`「不成立（否证线被触发）」冲突（补丁未同步 `analysis.md`）；② `results.json:606` 的墙钟数字（≈8 min GPU）落在判据文件而非 `run_log.txt`。WD256 的「本尺度不成立」措辞**在其量级（0.0486 = 3.5×带）下可接受**，仅缺「单 seed/一条训练」字样。

## ② 逐项检查表（命令 / 真实输出摘要 / 结论）

| # | 检查 | 命令 | 真实输出摘要 | 结论 |
|---|---|---|---|---|
| 1 | pytest 全绿 + 能变红用例 | `python3 -m pytest test_2642_uniform_narrowing.py -q` | `........................  [100%]` / `24 passed in 5.82s` | **通过**（24 passed，无 skip ⇒ bf16/CUDA 门真跑） |
| 2 | 自算 Δbpc/MACs/斜率 vs 落盘 | 自写脚本读 `results_D192_s0.json`/`results_WD256_s0.json`/`2641/results_U4|U2_s0.json` | D192 Δ=0.0527 省40.957% 斜率0.001287；U2 Δ=0.0395 省28.368% 斜率0.001392；WD@192 Δ=0.1013 斜率0.002473；WD@128 Δ=0.1987 斜率0.002788 —— **与 `results.json` P1/P2/P3 逐位一致** | **通过** |
| 3 | 契约冻结（sha256/mtime） | `stat`/`sha256sum`/`git log --follow` | README mtime `2026-10-14 20:55:45`；sha256 `c5e97de4…`；`git log` 只有 `04be624` 一次触碰；起跑 `run_log.txt` 首条 lock `21:08:53`（D192-s0 正式臂 `21:49:48`，`runs/D192-s0-12k` mtime `21:57:28`） | **通过**（mtime 早于起跑 13 min） |
| 3b | 修补是否移动判据线 | `git show 1a77ddf -- run.py` | 只把 `if "D192" in p1 and "D128" in p1` 改为「按已跑臂」+ f-string + 新增 `arms_run`/`arms_dropped`；`VARIANCE_BOUND=0.014` 与 P1–P5 定义**未动**；正分支文案（「占前沿优势」）**修前已存在**于 `04be624` | **通过（未移动判据线）** |
| 4 | 「同一 checkpoint 多档」 | 读 `results_WD256_s0.json` + 自写 CPU 复跑切片评测 | 顶层 `model_pt_sha256=27a8f903…` 单值 = 实际 `runs/WD256-s0-12k/model.pt` sha256；我独立复跑 `build_sliced_model(sd,w)+depth_curve(w=192)` → **`1.6327`（match）**，`w=128` → **`1.7301`（match）** | **通过** |
| 4b | 「减宽干预真到达模型」 | 读落盘 + 自算解析式 | D192 实测 3,944,064 == 解析；解析 `4w²+2Tw`/`2rw²`/`w·V`：w=128/192/256 → 2,123,520/4,364,928/7,392,768 与落盘逐位同；WD256 三档 `no_full_width_tensor=true`、`hidden=[4w]×8` | **通过** |
| 5 | 评测确定性 | `diff` 工作树 vs `git show 04be624:` 的逐臂 JSON；独立 CPU 复跑 | 工作树 `results_D192_s0.json`/`results_WD256_s0.json` 与 `04be624` 版本 **JSON 完全一致**（reeval 于 22:30/22:32 重写过文件但内容不变）；我另一次独立进程复跑亦得同值 | **通过** |
| 6 | 数字保真（≥5） | 抽 1.5314/1.5841/1.6327/1.6178/0.0527/0.0486/40.96%/43.4% | 全部与 `2641/results_U4_s0.json`（1.5314）及本项落盘 JSON 对上 | **通过** |
| 7 | 口径纪律 | `grep -nE "墙钟|吞吐|wall|min|throughput"` on `analysis.md`/`results.json` | `analysis.md` 只在「纪律声明」处出现「墙钟/吞吐只进 `run_log.txt`」；`results.json:606` 出现「≈8 min GPU」 | **部分通过**（判据未用墙钟；见 ③N2） |
| 8 | 越界扫描 | `git diff --name-only 04be624..1a77ddf`；`git show --stat 04be624` | 区间 = `…/2642…/run.py`、`…/results.json`、`.scratch/64-axis3-width/TICKET.md`；`04be624` = 该实验目录 16 文件 | **通过**（无本轴未说明改动） |
| 9 | SSOT 校验 | `python3 tools/summary.py --check` | `OK：46 个 project.yaml 全部符合 docs/research-schema.md。`（exit 0） | **通过** |
| 10 | 结论是否越证据 | 读 `analysis.md` §0/§4.1/§5 vs `results.json` P2/P5b | 见 ③B1 | **不通过 ⇒ FAIL** |

### 1a. pytest 组成（`--collect-only`）
- 等值门：`test_t1_prefix_forward_at_full_width_matches_stock_bitwise`、`test_t1_prefix_forward_matches_stock_under_bf16_autocast`（2 条）、`test_t2_sliced_model_matches_prefix_forward_bitwise[128/192/256]`（3 条）。
- 参数/维数审计 T4：`…_is_really_narrower[128/192/256]`、`…_uniform_arm_params_match_analytic[256-8/192-6/128-4]`、`…_analytic_cost_matches_2641_recorded_values`（7 条）。
- **变异门 T3（必须抓红）**：`t3a_wrong_proj_row_slice`、`t3b_wrong_ln_slice`、`t3c_wrong_proj_column_slice`、`t3d_wrong_head_count`、`t3e_masking_style_slice`（5 条）——均以 `assert not torch.equal(mutated, reference)` 表达「错切必被 T1/T2 抓到」，非恒真（若错切与正确切片等值即红）。
- 本 WP 期新增回归门：`test_p4_liveness_hook_accepts_block_tuple_output`（tuple 输出 hook 回归；docstring 声明退回旧 hook 即 `AttributeError` 变红）。
- 因果性/复用/构造：`test_causality_prefix_and_sliced[256/192/128]`、`test_anchor_reuse_reads_2641_files_and_config_parity`、`test_arm_argv_differs_from_u4_only_in_width_fields`、`test_width_dropout_model_stock_init_and_prefix_forward`（6 条）。
- **能变红的证明方式**：T3a–e 用实跑列出「变异 ≠ 参考」，等价于 T1/T2 对这些错误构造会红；`test_p4_…tuple_output` 的红路径我**未做回退变异**（CUDA 依赖），未跑 = 不计通过该单条的红性，但「本 WP 期内新增、能变红」的要求已由 T3a–e 满足。

### 2a. 自算输出（节选）
```
D192 bpc=1.5841 delta=0.0527 cut=40.957% slope=0.001287
U2  delta=0.0395 cut=28.368% slope=0.001392
WD@128 bpc=1.7301 delta=0.1987 cut=71.276% slope=0.002788
WD@192 bpc=1.6327 delta=0.1013 cut=40.957% slope=0.002473
adjacent 128-192 0.0974  192-256 0.0149 ; WD@192-D192=0.0486 ; WD@256-U4=0.0864
analytic w=192: attn=1966080 ffn=2359296 readout=39552 total=4364928   (== 落盘)
```

### 4a. 独立切片复跑（CPU，写盘为零）
```
REPRO w=192: clean_bpc=1.6327 expect=1.6327 match=True
REPRO w=128: clean_bpc=1.7301 expect=1.7301 match=True
```

## ③ blocker 与 `file:line`

### blocking
- **B1 · 结论越证据（把噪声带内的差异写成方向性/优势结论）** — `analysis.md:26/28/103/125`、`results.json:609`。
  - 事实：`Δ_D192 = 0.0527`、`Δ_U2 = 0.0395`，**两者之差 = 0.0132 < 0.014**（本实验自订、且 `analysis.md` 全篇引用的噪声带）；斜率差 `0.001392−0.001287 = 0.000105`。把 ±0.014 代入 D192 的 40.957% 档即得斜率不确定度 ±0.000342（**> 该斜率差**）⇒ 排序不可分辨。
  - 越界处：`analysis.md:28`「⇒ 在从 U4 出发的射线上，**整体减宽比砍 FFN 略便宜**」；`:103`「D192 在这条射线上**略便宜（约 −8%）**」；`:125`「D192（0.001287）**略优于** U2（0.001392）⇒ …**减宽不落下风**」；`results.json:609`「…⇒ d_model 减宽在该档**占前沿优势**」。
  - 自相矛盾：`analysis.md:142-143` 自己写「⇒ **不能判**『d_model 减宽不占前沿优势』；但只有 d=192 一档，**需 D128 复核**」——与 `results.json:609` 的「占前沿优势」方向断言强度不一致。
  - 缺失的限定：**单 seed（n=1）**、**未收敛（12k 步 13M）**、以及**斜率差落在 0.014 带内**这条最关键的限定；预登记 P5b 只定义了「均 ≥ U2 ⇒ 不占优势」的负分支，未授权「占优势」正向判词。
  - 判定依据：`REVIEW-DISPATCH.md` §R-3.1.1 blocking「结论越证据（把噪声带内的差异说成已分胜负）」。

### non-blocking
- **N1 · 补丁未同步 `analysis.md`，事实文档自相冲突** — `analysis.md:139-140` 仍写「脚本按设计输出『D192/D128 结果缺失』，无法给出机器判定」，但 lead 于 `1a77ddf`（22:35:33）重聚合后的 `results.json:608` 写「不成立（否证线被触发）」（`analysis.md` mtime `22:32:44` < 补丁时刻）。**B1 之外的第二处 `analysis.md`↔`results.json` 不一致**；两文对同一事实冲突，须按 `AGENTS.md` §2.1 删一处或改指针。
- **N2 · 判据文件里出现墙钟数字** — `results.json:606` `arms_dropped` 含「≈8 min GPU / ≈3 min」（README §4 规定墙钟「只进 `run_log.txt`」）。它是**下一步实验的成本估计**、非本实验判据，故不判 blocking，但按契约字面应移入过程日志。
- **N3 · WD256 结论措辞** — `analysis.md:35/136`「弹性宽度在本尺度不成立」**量级可接受**（0.0486 = 3.5×0.014 带，噪声不可解释），但缺「**单 seed、一条训练**的负面读数」字样；不构成 FAIL。
- **N4 · 未跑 D128:0** — `results.json:606` 与 `run_log.txt:21:49:40` 如实记录放弃理由（GPU 锁排队超预算）与下一步成本，`arms_dropped` 诚实；不作为 blocker。

## ④ 未跑项与理由

- **`run.py --no-train` 的 GPU 重评未由我实跑**：`/tmp/pathlm_gpu.lock` 复核期间被 `wp2672-eval`（`2672-fill-n2-13m`）持有，按 `SCHEMA.md` §20 不得抢锁。**替代证据**：① 用**独立进程 + CPU** 复跑同 checkpoint 的切片评测，`w=192 → 1.6327`、`w=128 → 1.7301` 与落盘逐位同；② 工作树 `results_*.json` 与 `04be624` 版本 JSON 完全一致，而 reeval 曾于 22:30/22:32 重写这两个文件 ⇒ 两次独立评测同值。故「评测确定性」仍有独立证据，但不是派单原本的 `--no-train` 路径。
- **`test_p4_liveness_hook_accepts_block_tuple_output` 的红路径未做回退变异**（需改文件，属改被审产物）⇒ 该单条的「能变红」未实跑，不计通过。
- **`analysis.md` 的 `bpc_by_depth` 逐档**未全量比对；我独立复跑只对到 `clean_bpc`（`depth_curve()[-1]`）。
- **`run_log.txt` 有 2 行未提交追加**（`git status` = `M`）：见 ⑤2，属过程日志，不影响判定的产物哈希。
- **锚 U4/U2 的 2641 config 逐字段对照**未从 `config.json` 原点重跑（落盘 `anchor_config_parity.all_equal=true` + pytest `test_anchor_reuse_reads_2641_files_and_config_parity` 已跑，但我未独立读 2641 `config.json` 逐字段）。

## ⑤ 残留不确定

1. **B1 的度量选择属我的推断**：0.014 带明文限「**配对** Δ、≥5 窗口」，而「D192 斜率 vs U2 斜率」是**跨工作点**的两个派生比值，严格说该带不直接作用于它。我采用的判据是：斜率差所依赖的 Δ 之差（0.0132）小于本实验自订噪声带，且单 seed ⇒ 方向不可辨。若主 Agent 认为「互不支配 + 略 + 需 D128 复核」的措辞已足够，则 B1 可降为非阻塞——但 `results.json:609`「占前沿优势」与 `analysis.md:142`「不能判」之间的**自相矛盾**仍需处置。
2. **被审 revision 定位**：`04be624`（22:32:56）晚于 reeval（22:30/22:32）、`1a77ddf`（22:35:33）晚于 `analysis.md`（22:32:44）；两提交均只触及本实验目录（+ ticket），未见未说明改动。复核期间 HEAD 由 `1a77ddf` 位移到 `7f319f5`（**其它轴并发提交**），被审的 `README.md`/`analysis.md`/`run.py`/`results*.json`/`test_*.py` 已提交且工作树相对 HEAD 干净；**唯一未提交改动 = `run_log.txt` 追加 2 行**（`22:32:00 reeval runner done` / `22:34:00 reeval runner2 done`，发生在 `04be624` 提交之后），属过程日志收尾、非判据产物。
3. **pytest 的 CUDA 门**：复核期 GPU 正被他人持有，`test_p4_…tuple_output` 与 bf16 T1 仍实跑并通过（13M 级小模型，非训练）；该并发不改变读数，但记录在此。
