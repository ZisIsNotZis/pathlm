> 性质：独立复核报告（轴 3 · 研究项 3.1.4 · Phase A 遗留复核门的**第 2 轮**）。
> 复核者 = fresh 独立上下文 subagent（builtin reviewer，run `d0fe4f22-42ce-4d01-86ff-dabfafad9bfa`，2026-10-14T20:53 派、21:0x 结束）。
> **工具限制（复核者自述）**：其环境只有 read/grep/find/ls/watchdog_diff，**无 shell、无写文件工具** ⇒ 报告由 axis3-lead 从其最终回复**原样落盘**（未改一字，仅去掉了日志里穿插的工具输出行）。
> 判词 = **FAIL（窄范围：复核门未闭合 = 复核者无 shell，非发现新的实质缺陷）**。

## VERDICT: FAIL（复核门未闭合；原因 = 本复核者无 shell/执行工具，非发现新的实质缺陷）
Blocker 列表（就我能只读核验的范围）：
- **无新增实质 blocker**：第 1 轮 B1/B2/B3 三处**看起来已真改**（逐条证据见下）。
- **Blocker（流程/证据类，未跑不计通过）**：派单要求的 6 项「必须实际执行」的检查我**一项也跑不了**（无 bash/执行工具），其中 `watchdog_diff` 亦不可用（返回 “Reviewer-launch HEAD changed or is unavailable”）。按派单规则「未跑 = 不计通过」且「命令均跑过才 ⇒ PASS」⇒ 本轮不能给 PASS。
---
# 给 `.scratch/64-axis3-width/REVIEW-2.md` 的报告全文（我无写文件工具，请落盘）
VERDICT: FAIL（窄范围：复核门未闭合 = 复核者无 shell 工具；静态复核未发现新的实质 blocker）
## ① 人话摘要
1. 第 1 轮的 3 处硬伤（把 |Δ|=0.0139 说成「小于 0.0098 所以效应被噪声吃掉」、APYR-s1 第 6 层写 2.4699、参数差余项写 1,024）**在我读到的版本里都已按最小修改正**，结论也已改成「判 H0 是阈值规则的判词，不是效应已被排除」，并保留两条不可排除的反假设限定。
2. 7 臂数字（clean bpc、MACs、参数、Δ、逐层深度读数）与落盘 JSON **逐位对得上**，我手算的四个 Δ 与 `results.json::P1_P2_deltas_by_seed` 一致，两条 seed 的 H0 判词也一致。
3. **但我拿不到 PASS 的证据**：本环境只给我 read/grep/find/ls/watchdog_diff，**没有 shell**——pytest、重算脚本、T5 初始化控制脚本、`sha256sum`/`stat`/`git show`、`git diff`、`summary.py --check` 全部未跑 ⇒ 按派单「未跑 = 不计通过」，本轮判 FAIL，请另派**有 shell 的** fresh 复核者跑那 6 项。
## ② 逐项检查表（命令 / 真实输出摘要 / 结论）
| # | 要求 | 我实际做的 | 真实输出摘要 | 结论 |
|---|---|---|---|---|
| 1 | `pytest test_2641_ffn_ratio.py -q` | **未跑**（无 shell）。静态：逐个数 `test_2641_ffn_ratio.py` 的用例 = T1 两条 + T2（`sorted(SCHEDULES)` 参数化 4 条 + 等预算 1 条 + 前缀 1 条）+ T3 五条 = **13**；`gates_report.txt:1-6` 自报 `exit=0` / `13 passed in 4.99s` | 用例数 13 自洽 | **未跑=不计通过**（静态旁证，不替代实跑） |
| 2 | 自写脚本重算 Δ 并核对 `P1_P2_deltas_by_seed` / `P1_verdict_by_seed` | **脚本未跑**；用 read 取 7 个 `results_<arm>_s<seed>.json:68` 的 `clean_bpc` 手算 | U4-s0 1.5314 / U2-s0 1.5709 / U2-s1 1.5611 / PYR-s0 1.5730（落盘 `1.573`）/ PYR-s1 1.5750（`1.575`）/ APYR-s0 1.5644 / APYR-s1 1.5601；Δ_up = 1.5709−1.5730 = **−0.0021**、1.5611−1.5750 = **−0.0139**；Δ_down = **+0.0065**、**+0.0010**；四个 \|Δ\| 全 < 0.014 ⇒ 两条 `P1_verdict_by_seed` 均「不可排序（H0）」；`P4_same_sign_across_seeds = true`（两向都同号）。与 `results.json`（arms 块 :11–73、deltas :75–84、grep 确认 :81 = −0.0139、verdict :85–88、P4 :89）**逐位一致** | 静态一致；**脚本未跑=不计通过** |
| 3 | **B1 复核**：是否仍有伪命题 / 结论是否已弱化 | `grep -n '吃掉\|小于同臂换 seed\|0\.0098' axes/3-width/.../analysis.md project.yaml` | 唯一含「被噪声吃掉」处 = `project.yaml:18 conclusion_review`，那是**对第 1 轮缺陷的描述**（「B1：把「四条 Δ 全小于同臂换 seed 散布 0.0098」用作「分配效应被噪声吃掉」的论据…」），不是当前结论。伪命题已消失；`analysis.md:33-37`（§0 边界）、`:89`（§3-P4）、`:101-104`（§4）、`:175-176`（§7.1）、`:201-202`（§8）均改为「判 H0 是**阈值规则**判词、**不等于效应已被排除**」；`project.yaml::conclusion` ②③ 与 `::failure_analysis` ① 同改并保留两条限定 | **已改，成立** |
| 4 | **B2 复核**：APYR-s1 第 6 层 bpc | `read results_APYR_s1.json`（40–61 行逐行核对）+ grep | `results_APYR_s1.json:53 "depth": 6` → `:54 "bpc": 2.4684`；`results_APYR_s0.json:54 = 2.7209`；`analysis.md:110` 现写「APYR 则相反（第 6 层 **2.4684/2.7209** = s1/s0）」 | **已改，成立** |
| 5 | **B3 复核**：参数差余项 | 读 `analysis.md:157-161` + 手算 | 现写「参数差 = 6,962,688 − 4,861,440 = 2,101,248 = 2·(4−2)·256²·8（= 2,097,152）+ (4−2)·256·8（`lin1` 偏置，= **4,096**）」，并注明 `lin2` 偏置 8·d = 2,048 不随 r 缩放；2,097,152 + 4,096 = 2,101,248 ✓ | **已改，成立** |
| 6 | **T5 初始化控制（写脚本 import `run.build_model` 逐位核）** | **脚本未跑**。静态论证：`run.py:120-131 build_model` 先 `torch.manual_seed(seed)` 再以**恒定** `MLP_MULT=4` 构造，再 `narrow_ffn_prefix`（`run.py:96-119`）只做 `W[:h].clone()` / `bias[:h].clone()` / `W[:, :h].clone()`（**无 RNG 消耗**）；训练侧 `train_depth_ar.py:771 torch.manual_seed(args.seed)` → `:790 model = DepthARModel(cfg, vocab_size)`（被换成工厂），且 `run.py:shared_argv` 对**所有臂**都传 `--mlp-mult 4` ⇒ RNG 消耗与 schedule 无关。旁证测试：`test_...py:88-98`（同 seed 全宽 vs r=2，前缀行/列 + 新参数对象）、`:170-182`（U4 vs U2 的 qkv/proj/embed 逐位相同） | 构造性论证成立；**执行未跑=不计通过** |
| 7 | **T6 契约未被事后编辑 + mtime 先后** | **未跑**（无 `git show`/`sha256sum`/`stat`）。可核的只有二手：`MODEL.md` 哈希表列 `README.md dd1e8201…`、`test_2641_ffn_ratio.py 66761343…`、`train_arms.sh 353988a3…`（由 `collect_logs.py` 训练后 19:07 生成，属自报）；`TICKET.md` §7.2.1（:158-163 附近）自报 mtime 17:58:24 / 17:59:41 / 18:00:10 与 49f697f 逐位相同 | **未核 = 不计通过** | **未跑=不计通过** |
| 7b | 口径差异核对（契约表 vs 落盘 `ffn_params_analytic`） | 读 `2641/README.md` §3 表 + `results_U4_s0.json` | README 表列「FFN 参数（解析）」U4 = 8·2·4·256² = **4,194,304**（不含偏置）；落盘 `results_U4_s0.json:132 ffn_params_analytic = 4204544`、`params_ffn = 4204544`；2×2,103,296 = 4,206,592，−2,048 = **4,204,544** ✓（analysis §5.2 已说明同名列口径不同） | **与事实相符** |
| 8 | **T9 越界扫描 + `summary.py --check`** | **未跑**（无 git/无 shell）。静态替代：`find axes -name project.yaml` = **46** 个（12+10+8+3+6+5+2），与「OK：46 个 project.yaml」一致 | 未核越界 | **未跑=不计通过** |
| 9 | 抽样 5 个数字对落盘 | 读/grep 对照 | ① U4-s0 clean bpc 1.5314（`results_U4_s0.json:68`、`results.json:66`、analysis §2）✓ ② U2-s0 1.5709（`results_U2_s0.json:68`）✓ ③ APYR-s1 1.5601 + 第 6 层 2.4684 ✓ ④ U4 每 token FFN MACs 4,194,304 / `params_ffn` 4,204,544 / 总 MACs 7,392,768（`results_U4_s0.json:114-131`）✓ ⑤ Δ_up(s1) −0.0139（`results.json:81`）✓ | **5/5 保真** |
| 附 | 臂间 config 抽查 | 读 `runs/PYR-s1-12k/results.json` | `mlp_mult 4 / d_model 256 / n_layers 8 / n_heads 8 / seq_len 256 / dense true / dropout 0.0 / seed 1`；`runs/` 被 `.gitignore:8`（`runs/`、`*.pt`、`data/`、`*.log`）覆盖 ⇒ grep 看不到，只能逐个 `read` | 抽查 1/7 通过（第 1 轮已逐字段核 7/7） |
| 附 | N 项的顺带抽查 | grep | `analysis.md:137`「恰为其两倍」已加限定（N2 ✓）；`analysis.md` 中「前后读数」已不存在（N3 ✓）；「0.30–0.35」已不存在（N1 ✓）；**N4 只改了一半**：`:109`（§4）标了 `(seed 0)`，但 `:40`（§0 真信号条目）仍写「第 6 层 1.9727 vs 等宽 2.3338」未标 seed | 见 ③-P2 |
## ③ blocker 与 file:line
- **B-0（流程，唯一阻塞 PASS 的项）**：本复核者环境**无 shell/执行工具**（工具集 = read/grep/find/ls/watchdog_diff/contact_supervisor；`watchdog_diff` 还报 `Reviewer-launch HEAD changed or is unavailable`）。派单第 1/2/6/7/8 项的**执行**部分与 T9 全部未跑 ⇒ 依派单「未跑 = 不计通过」「全部命令跑过才 ⇒ PASS」，本轮判 FAIL。**不是**在产物里发现了新的数字/结论缺陷。
- **第 1 轮 3 处 blocker：静态核验均已真改**（`analysis.md:33-37/87/89/101-104/110/137/157-161/175-176/201-202`、`project.yaml:16-17`）：
  - B1 ⇒ `analysis.md:33-36`「PYR-s1 的 |Δ| = 0.0139 已贴在项目自认的种子散布上界（≤0.0138）⇒ 不能排除幅度约 0.014 的真实效应…散列不能当作 |Δ| 的上界」；`:104`「『不可排序』是**阈值规则**的判词」。
  - B2 ⇒ `analysis.md:110` = 2.4684/2.7209（落盘 `results_APYR_s1.json:54` = 2.4684；`results_APYR_s0.json:54` = 2.7209）。
  - B3 ⇒ `analysis.md:160` = `lin1` 偏置 **4,096**（2,097,152+4,096 = 2,101,248 ✓）。
- **P2（不改不影响本轮判词）**：`analysis.md:40`（§0「一个真信号」）仍引用未标 seed 的中间深度读数（真实是 seed 0：PYR-s0 L6 1.9727 / L7 1.7492），与 `:109` 已标 seed 的写法不一致；最小修 = 在 `:40` 括注「(seed 0)」。
- **P2（过程工单内部陈旧数字）**：`TICKET.md` §7.2.2 表的 `FFN 参数` = 2,099,200（= 2,097,152+2,048）与 §7.3 表 / SSOT 的 2,103,296 不一致；该表是 18:2x 的**中途快照**且工单允许日期流水 ⇒ 不构成事实冲突，但建议加一行「（中途快照，最终读数见 §7.3）」。
- **我未改动任何被审产物**（无写文件工具，仅只读复核）。
## ④ 未跑项与理由（一律不计通过）
| 未跑项 | 理由 | 交 supervisor 的精确命令（请有 shell 的复核者跑） |
|---|---|---|
| pytest 13 passed | 无 shell | `cd axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m && python3 -m pytest test_2641_ffn_ratio.py -q` |
| 独立重算 Δ / 判词 | 无 shell（我只做了读文件+手算，不替代脚本） | `python3 - <<'PY'` 读 7 个 `results_<arm>_s<seed>.json` 的 `eval.clean_bpc` 算 Δ_up/Δ_down 并与 `results.json::P1_P2_deltas_by_seed`、`P1_verdict_by_seed` 比对 |
| T5 初始化控制 | 无 shell | `import run; u4=run.build_model(run.SCHEDULES['U4'],206,0); … 逐位 assert trunk + FFN 前缀`（seed 0/1 × U2/PYR/APYR 共 6 组，逐层布尔）。静态论证我已给出（见 ②-6），但**未执行** |
| T6 契约未被事后编辑 + mtime | 无 git/sha256sum/stat；`watchdog_diff` 不可用 | `git show 49f697f:axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/{README.md,train_arms.sh,test_2641_ffn_ratio.py} \| sha256sum` 与工作树对照；`stat -c '%y %n' <三份契约文件> runs/*`（须全部早于起跑 18:00:37） |
| T9 越界扫描 | 无 git | `git diff --name-only 49f697f..HEAD \| grep -vE '^(axes/3-width/\|\.scratch/64-axis3-width/)' \| wc -l`（期望 0） |
| `summary.py --check` | 无 shell | `python3 tools/summary.py --check`（期望 `OK：46 个 project.yaml …`；我静态数到 `axes/**/project.yaml` = 46，只证数目自洽） |
| 工作树 delta / 被审 revision 一致性 | `watchdog_diff` 返回「Reviewer-launch HEAD changed or is unavailable」 | 重新 launch 复核者，或提供 `49f697f..HEAD` 的 diff 工件 |
## ⑤ 残留不确定
1. **被审 revision 不明**：`watchdog_diff` 不可用 ⇒ 我无法证明「我读的工作树 == 派单所指 revision」，也无法排除我读到的是第 2 轮修完后的工作树之外的其它状态（我读到的版本里 B1–B3 已修，故时间上应在第 1 轮修正之后）。
2. **契约冻结（T6）纯属未核**：`MODEL.md` / `TICKET.md` 的哈希与 mtime 都是 writer 自报（`collect_logs.py` 在 19:07 生成，即"训练后"），**不能证明** `README.md` / `train_arms.sh` / `test_2641_ffn_ratio.py` 与 `49f697f` 逐位相同。若有人事后编辑了契约文件，我的静态核对无法发现。
3. **`analysis.md:111`（§5.1）引用「初始化控制（独立核验，CPU）」的脚本**在实验目录内不存在（`ls` 无对应文件）——第 1 轮已记为 I2 并改为引用 T2/T3 承载；我确认现文已改为「由 `T2_…` 与 `T3_…` 两条测试承载」，但**这两条测试我未实跑**。
4. **越界风险未核**：若 `49f697f..HEAD` 触碰了 `axes/3-width/**` 与 `.scratch/64-axis3-width/**` 之外的文件，我无法发现。
5. 静态可核部分的**结论**：3 处 blocker 已真改、7 臂数字保真、Δ 与判词自洽、结论未越证据（H0 被明确标为阈值规则判词 + 两条不可排除的反假设限定在位）——**这些不构成 PASS**，只是「若另派有 shell 的复核者跑完 ④ 的 6 项且输出一致，本轮的 FAIL 原因即可消除」。
Mid-run steering from the parent orchestrator:
补充口径（不影响已有判据，只防误判）：本仓库此刻有**其它轴**的 lead/worker 在同一工作树并发写入（例如 `axes/2-depth/**`、`axes/5-quant/**`、`axes/6-predict/**`、`.scratch/6*/**`）。因此 T9 的 `git diff --name-only 49f697f..HEAD` 里出现非本轴的路径时，**不等于本项 writer 越界**。请这样判：① 只看 2641 实验目录与 `axes/3-width/**`、`.scratch/64-axis3-width/**` 相关的改动是否与 `e6d2db1..9e9b291` 这段（3.1.4 的处置提交）一致；② 非本轴路径若来自别的 ticket（看路径归属即可判断），记为「并发其它轴的既有工作，非本项越界」并附路径列表；③ 只有当**本轴的** `project.yaml`/实验产物出现未说明的改动时才计 blocker。其它判据不变。
Incorporate this guidance at the next safe point. Do not restart the task unless the guidance explicitly asks you to.
Incorporating the new T9 口径 at this safe point — I still have no git, but I can statically corroborate the concurrent-writers attribution.
ls: /home/z/vibe/pathlm/.scratch
grep: /home/z/vibe/pathlm/axes
01-design/
02-engine-m0/
03-engine-m1/
04-m1-runs/
05-m1-followup/
06-m2-sweep/

---
12-overnight/
13-four-experiments/
14-allocator/
15-slider-rung3/
16-rung4-scale/
17-slider-at-scale/
18-exit-probe/
19-asym-100m/
20-spec-scale/
21-exit-at-scale/
22-gen-quality/
23-depth-ar/
24-depth-ar-100m/
25-tax-attribution/
26-closing/
27-research-wiki/
28-pilot-batch1/
29-verify-batch/
30-batch3/
31-batch4/
32-batch5/
33-batch6/
34-brainstorm/
35-carry-channel/
36-parallel/
37-gaps/
38-auto-parallel/
39-close-gaps/
40-next-wave/
41-followups/
42-longt-stability/
43-longt-clean/
44-wave17/
46-wave19/
49-adversarial-review/
50-critical-audit/
50-doc-ssot/
51-tree/
52-a6-longt/
52-r7-early-exit/
53-design-audit/
54-subproject-schema/
55-review/
56-restructure/
57-program/
61-axis2-depth/
62-axis1-time/
63-axis6-predict/
64-axis3-width/
65-axis4-moe/
66-axis5-quant/
67-engine-changes/
doc-log/
README.md
4-moe/4.3-expert-granularity/project.yaml:57:   - date: "2026-10-13T11:06:24+08:00"
4-moe/4.3-expert-granularity/project.yaml:60:   - date: "2026-10-13T19:25:15+08:00"
4-moe/4.3-expert-granularity/project.yaml:63:   - date: "2026-10-14T06:44:03+08:00"
5-quant/5.6-window-quant-retrieval/project.yaml:70:   - date: "2026-10-13T20:09:47+08:00"
5-quant/5.6-window-quant-retrieval/project.yaml:73:   - date: "2026-10-13T11:06:24+08:00"
5-quant/5.6-window-quant-retrieval/project.yaml:76:   - date: "2026-10-14T06:59:49+08:00"
5-quant/5.6-window-quant-retrieval/project.yaml:79:   - date: "2026-10-14T20:52:00+08:00"
4-moe/4.2-routing-balance/project.yaml:59:   - date: "2026-10-13T11:06:24+08:00"
4-moe/4.2-routing-balance/project.yaml:62:   - date: "2026-10-13T19:25:15+08:00"
4-moe/4.2-routing-balance/project.yaml:65:   - date: "2026-10-14T06:44:03+08:00"
6-predict/6.2-speculative-decoding/project.yaml:102:   - date: "2026-10-14T10:01:27+08:00"
6-predict/6.2-speculative-decoding/project.yaml:105:   - date: "2026-10-13T17:55:00+08:00"
6-predict/6.2-speculative-decoding/project.yaml:108:   - date: "2026-10-13T17:50:00+08:00"
6-predict/6.2-speculative-decoding/project.yaml:111:   - date: "2026-10-13T10:47:00+08:00"
6-predict/6.2-speculative-decoding/project.yaml:114:   - date: "2026-10-14T06:44:03+08:00"
6-predict/6.2-speculative-decoding/project.yaml:117:   - date: "2026-10-14T06:59:49+08:00"
6-predict/6.2-speculative-decoding/project.yaml:120:   - date: "2026-10-14T20:52:40+08:00"
5-quant/5.5-fp8-head-dim/project.yaml:65:   - date: "2026-10-13T20:09:47+08:00"
5-quant/5.5-fp8-head-dim/project.yaml:68:   - date: "2026-10-13T11:06:24+08:00"
5-quant/5.5-fp8-head-dim/project.yaml:71:   - date: "2026-10-14T06:59:49+08:00"
5-quant/5.5-fp8-head-dim/project.yaml:74:   - date: "2026-10-14T20:52:00+08:00"
5-quant/5.4-low-rank/project.yaml:57:   - date: "2026-10-13T20:09:47+08:00"
5-quant/5.4-low-rank/project.yaml:60:   - date: "2026-10-13T11:06:24+08:00"
5-quant/5.4-low-rank/project.yaml:63:   - date: "2026-10-14T06:44:03+08:00"
5-quant/5.4-low-rank/project.yaml:66:   - date: "2026-10-14T06:59:49+08:00"
5-quant/5.3-activation-quant/project.yaml:56:   - date: "2026-10-13T20:09:47+08:00"
5-quant/5.3-activation-quant/project.yaml:59:   - date: "2026-10-13T11:06:24+08:00"
5-quant/5.3-activation-quant/project.yaml:62:   - date: "2026-10-14T06:59:49+08:00"
5-quant/5.2-weight-quant/project.yaml:56:   - date: "2026-10-13T20:09:47+08:00"
5-quant/5.2-weight-quant/project.yaml:59:   - date: "2026-10-13T11:06:24+08:00"
5-quant/5.2-weight-quant/project.yaml:62:   - date: "2026-10-14T06:59:49+08:00"
5-quant/5.1-kv-quant/project.yaml:88:   - date: "2026-10-13T20:13:43+08:00"
5-quant/5.1-kv-quant/project.yaml:91:   - date: "2026-10-13T11:06:24+08:00"
5-quant/5.1-kv-quant/project.yaml:94:   - date: "2026-10-14T06:44:03+08:00"
5-quant/5.1-kv-quant/project.yaml:97:   - date: "2026-10-14T06:59:49+08:00"
6-predict/6.3.3-latent-regression/project.yaml:58:   - date: "2026-10-14T10:02:06+08:00"
6-predict/6.3.3-latent-regression/project.yaml:61:   - date: "2026-10-13T10:47:00+08:00"
6-predict/6.3.3-latent-regression/project.yaml:64:   - date: "2026-10-14T06:44:03+08:00"
6-predict/6.3.3-latent-regression/project.yaml:67:   - date: "2026-10-14T06:59:49+08:00"
6-predict/6.3.3-latent-regression/project.yaml:70:   - date: "2026-10-14T20:52:40+08:00"
9-backlog/9.2-iteration-refinement/project.yaml:86:   - date: "2026-10-13T11:32:11+08:00"
6-predict/6.3.2-pad-fill/project.yaml:55:   - date: "2026-10-13T10:47:00+08:00"
6-predict/6.3.2-pad-fill/project.yaml:58:   - date: "2026-10-14T06:59:49+08:00"
6-predict/6.3.2-pad-fill/project.yaml:61:   - date: "2026-10-14T20:52:40+08:00"
4-moe/4.1-sparse-experts/project.yaml:68:   - date: "2026-10-13T11:06:24+08:00"
4-moe/4.1-sparse-experts/project.yaml:71:   - date: "2026-10-13T19:25:15+08:00"
4-moe/4.1-sparse-experts/project.yaml:74:   - date: "2026-10-14T06:44:03+08:00"
2-depth/2.3.2-layer-resolution/project.yaml:55:   - date: "2026-10-13T10:37:02+08:00"
2-depth/2.3.2-layer-resolution/project.yaml:58:   - date: "2026-10-14T06:59:49+08:00"
2-depth/2.3.1-layer-capacity/project.yaml:55:   - date: "2026-10-13T10:37:02+08:00"
2-depth/2.3.1-layer-capacity/project.yaml:58:   - date: "2026-10-14T06:59:49+08:00"
9-backlog/9.1-selector-scheduling/project.yaml:87:   - date: "2026-10-13T09:40:25+08:00"
9-backlog/9.1-selector-scheduling/project.yaml:90:   - date: "2026-10-13T11:32:11+08:00"
9-backlog/9.1-selector-scheduling/project.yaml:93:   - date: "2026-10-14T06:59:49+08:00"
2-depth/2.2.5-weight-sharing/project.yaml:56:   - date: "2026-10-13T10:37:02+08:00"
2-depth/2.2.5-weight-sharing/project.yaml:59:   - date: "2026-10-14T06:44:03+08:00"
2-depth/2.2.5-weight-sharing/project.yaml:62:   - date: "2026-10-14T06:59:49+08:00"
2-depth/2.2.5-weight-sharing/project.yaml:65:   - date: "2026-10-14T21:00:00+08:00"
2-depth/2.2.4-parallel-sublayer/project.yaml:55:   - date: "2026-10-13T10:37:02+08:00"
2-depth/2.2.4-parallel-sublayer/project.yaml:58:   - date: "2026-10-14T06:59:49+08:00"
1-time/1.2.4-dynamic-merge/project.yaml:55:   - date: "2026-10-13T10:29:52+08:00"
1-time/1.2.4-dynamic-merge/project.yaml:58:   - date: "2026-10-14T21:05:00+08:00"
3-width/3.1.1-uniform-narrowing/project.yaml:68:   - date: "2026-10-13T10:57:34+08:00"
3-width/3.1.1-uniform-narrowing/project.yaml:71:   - date: "2026-10-14T06:44:03+08:00"
3-width/3.1.1-uniform-narrowing/project.yaml:74:   - date: "2026-10-14T06:59:49+08:00"
2-depth/2.2.2-attention-residual/project.yaml:56:   - date: "2026-10-13T10:37:02+08:00"
2-depth/2.2.2-attention-residual/project.yaml:59:   - date: "2026-10-14T06:59:49+08:00"
1-time/1.2.3-content-topk/project.yaml:66:   - date: "2026-10-13T10:29:52+08:00"
1-time/1.2.3-content-topk/project.yaml:69:   - date: "2026-10-14T06:59:49+08:00"
3-width/3.1.3-position-width/project.yaml:55:   - date: "2026-10-13T10:57:34+08:00"
3-width/3.1.3-position-width/project.yaml:58:   - date: "2026-10-14T06:59:49+08:00"
3-width/3.1.4-ffn-ratio/project.yaml:71:   - date: "2026-10-13T10:57:34+08:00"
3-width/3.1.4-ffn-ratio/project.yaml:74:   - date: "2026-10-13T18:02:28+08:00"
3-width/3.1.4-ffn-ratio/project.yaml:77:   - date: "2026-10-13T19:2x+08:00"
3-width/3.1.4-ffn-ratio/project.yaml:80:   - date: "2026-10-13T19:09:16+08:00"
3-width/3.1.4-ffn-ratio/project.yaml:83:   - date: "2026-10-14T06:44:03+08:00"
3-width/3.1.4-ffn-ratio/project.yaml:86:   - date: "2026-10-14T06:59:49+08:00"
3-width/3.1.2-layer-width/project.yaml:56:   - date: "2026-10-13T10:57:34+08:00"
3-width/3.1.2-layer-width/project.yaml:59:   - date: "2026-10-14T06:59:49+08:00"
3-width/3.1.5-channel-sparsity/project.yaml:55:   - date: "2026-10-13T10:57:34+08:00"
3-width/3.1.5-channel-sparsity/project.yaml:58:   - date: "2026-10-14T06:59:49+08:00"
3-width/3.2.1-readout-reparam/project.yaml:91:   - date: "2026-10-13T10:57:34+08:00"
3-width/3.2.1-readout-reparam/project.yaml:94:   - date: "2026-10-14T06:59:49+08:00"
3-width/3.2.2-per-exit-readout/project.yaml:55:   - date: "2026-10-13T10:57:34+08:00"
3-width/3.2.2-per-exit-readout/project.yaml:58:   - date: "2026-10-14T06:59:49+08:00"
1-time/1.2.2-block-sparse/project.yaml:67:   - date: "2026-10-13T10:29:52+08:00"
1-time/1.2.2-block-sparse/project.yaml:70:   - date: "2026-10-14T06:59:49+08:00"
6-predict/6.4-training-fill/project.yaml:69:   - date: "2026-10-13T10:47:00+08:00"
6-predict/6.4-training-fill/project.yaml:72:   - date: "2026-10-13T12:15:00+08:00"
6-predict/6.4-training-fill/project.yaml:75:   - date: "2026-10-14T06:59:49+08:00"
6-predict/6.4-training-fill/project.yaml:78:   - date: "2026-10-14T20:52:40+08:00"
2-depth/2.1.2-skipping/project.yaml:70:   - date: "2026-10-13T10:37:02+08:00"
2-depth/2.1.2-skipping/project.yaml:73:   - date: "2026-10-14T21:00:00+08:00"
2-depth/2.1.1-early-exit/project.yaml:113:   - date: "2026-10-13T13:30:00+08:00"
2-depth/2.1.1-early-exit/project.yaml:116:   - date: "2026-10-13T10:37:02+08:00"
2-depth/2.1.1-early-exit/project.yaml:119:   - date: "2026-10-14T06:44:03+08:00"
2-depth/2.1.1-early-exit/project.yaml:122:   - date: "2026-10-14T06:59:49+08:00"
3-width/3.2.3-tied-untied/project.yaml:55:   - date: "2026-10-13T10:57:34+08:00"
3-width/3.2.3-tied-untied/project.yaml:58:   - date: "2026-10-14T06:59:49+08:00"
1-time/1.2.1-dilated/project.yaml:55:   - date: "2026-10-13T10:29:52+08:00"
[100 matches limit reached. Use limit=200 for more, or refine pattern]
已在新口径下重做 T9 的判定框架（执行部分仍跑不了）。以下为 `.scratch/64-axis3-width/REVIEW-2.md` 的**完整**内容（含对本轮新增口径的整合），请原样落盘。
---
VERDICT: FAIL（窄范围：复核门未闭合 —— 复核者无 shell/执行工具；静态复核未发现新的实质 blocker，3 处第 1 轮 blocker 已真改）
## ① 人话摘要
1. 第 1 轮的 3 处硬伤（把 |Δ|=0.0139 说成「小于 0.0098 ⇒ 效应被种子噪声吃掉」、APYR-s1 第 6 层写 2.4699、参数差余项写 1,024）**在我读到的版本里都已按最小修改正**，结论也已改为「判 H0 是阈值规则的判词，不是效应已被排除」，并保留两条不可排除的反假设限定。
2. 7 臂数字（clean bpc / MACs / 参数 / Δ / 逐层深度读数）与落盘 JSON **逐位一致**，我手算的四个 Δ 与 `results.json::P1_P2_deltas_by_seed`、两条 `P1_verdict_by_seed`（H0）一致。
3. **但我拿不到 PASS 的证据**：本环境只给我 read/grep/find/ls/watchdog_diff/contact_supervisor，**没有 shell**，且 `watchdog_diff` 本身报 “Reviewer-launch HEAD changed or is unavailable” ⇒ pytest、重算脚本、T5 脚本、sha256sum/stat/git、summary.py --check 全部未跑，按派单「未跑 = 不计通过」判 FAIL；请另派**有 shell 的** fresh 复核者跑 ④ 列的 6 项。
## ② 逐项检查表（命令 / 真实输出摘要 / 结论）
| # | 要求 | 我实际做的 | 真实输出摘要 | 结论 |
|---|---|---|---|---|
| 1 | `pytest test_2641_ffn_ratio.py -q` | **未跑**（无 shell）。静态：逐个数用例 = T1 两条 + T2（参数化 4 条 + 等预算 1 条 + 前缀 1 条）+ T3 五条 = **13**；`gates_report.txt:1-6` 自报 `exit=0` / `13 passed in 4.99s` | 用例数 13 自洽 | **未跑=不计通过** |
| 2 | 自写脚本重算 Δ，核 `P1_P2_deltas_by_seed` / `P1_verdict_by_seed` | **脚本未跑**；read 7 个 `results_<arm>_s<seed>.json:68` 的 `clean_bpc` 手算 | clean bpc：U4-s0 1.5314、U2-s0 1.5709、U2-s1 1.5611、PYR-s0 1.5730（落盘 `1.573`）、PYR-s1 1.5750（`1.575`）、APYR-s0 1.5644、APYR-s1 1.5601；Δ_up = **−0.0021 / −0.0139**、Δ_down = **+0.0065 / +0.0010**；四个 \|Δ\| < 0.014 ⇒ 两 seed 均「不可排序（H0）」；P4 同号 = true。与 `results.json`（arms :11–73、deltas :75–84〔grep 确认 :81 = −0.0139〕、verdict :85–88、P4 :89）逐位一致 | 静态一致；**脚本未跑=不计通过** |
| 3 | **B1 复核** | `grep -n '吃掉\|小于同臂换 seed\|0\.0098'` | 唯一「被噪声吃掉」在 `project.yaml:18 conclusion_review`，是**对第 1 轮缺陷的描述**（引号内的旧说法），非当前结论；`analysis.md:33-37`（§0 边界）、`:89`（P4）、`:101-104`（§4）、`:175-176`（§7.1）、`:201-202`（§8）已改为「H0 是**阈值规则**判词、**不等于效应已被排除**」；`project.yaml::conclusion` ②③ + `::failure_analysis` ① 同改，两条反假设限定在位 | **已改，成立** |
| 4 | **B2 复核** | read `results_APYR_s1.json` 逐行核 | `results_APYR_s1.json:53 depth 6` → `:54 bpc 2.4684`；`results_APYR_s0.json:54 = 2.7209`；`analysis.md:110` 现写「第 6 层 **2.4684/2.7209** = s1/s0」 | **已改，成立** |
| 5 | **B3 复核** | read `analysis.md:157-161` + 手算 | 现写「= 2·(4−2)·256²·8（2,097,152）+ (4−2)·256·8（`lin1` 偏置，= **4,096**）= 2,101,248」，并注明 `lin2` 偏置 8·d = 2,048 不随 r 缩放；2,097,152+4,096 = 2,101,248 ✓ | **已改，成立** |
| 6 | **T5 初始化控制脚本** | **未跑**。静态：`run.py:120-131 build_model` 先 `manual_seed(seed)`、再以**恒定** `MLP_MULT=4` 构造、再 `narrow_ffn_prefix`（`run.py:96-119`，只 `W[:h].clone()` / `W[:, :h].clone()`，**无 RNG 消耗**）；训练侧 `train_depth_ar.py:771 manual_seed` → `:790 DepthARModel(...)`（被工厂替换），且 `shared_argv` 对**所有臂**都传 `--mlp-mult 4` ⇒ RNG 消耗与 schedule 无关；旁证测试 `test_2641_ffn_ratio.py:88-98`（前缀行/列 + 新对象）、`:170-182`（U4 vs U2 的 qkv/proj/embed 逐位同） | 构造性论证成立；**执行未跑=不计通过** |
| 7 | **T6 契约未被事后编辑 + mtime** | **未跑**（无 git/sha256sum/stat）。仅二手：`MODEL.md` 哈希表（`README.md dd1e8201…`、`test_2641_ffn_ratio.py 66761343…`、`train_arms.sh 353988a3…`，由 `collect_logs.py` 训练后 19:07 生成）；`TICKET.md` §7.2.1 自报 mtime 17:58:24 / 17:59:41 / 18:00:10 与 49f697f 逐位相同 | **未核=不计通过** |
| 7b | 口径差异核对 | read `2641/README.md` §3 + `results_U4_s0.json` | README 表「FFN 参数（解析）」U4 = 8·2·4·256² = 4,194,304（**不含偏置**）；落盘 `results_U4_s0.json:132 ffn_params_analytic = 4204544`、`params_ffn = 4204544`；2×2,103,296 − 2,048 = **4,204,544** ✓（analysis §5.2 已说明同名不同值） | **与事实相符** |
| 8 | **T9 越界扫描**（按新增口径） | **未跑**（无 git）。静态替代与归属判断见下 | `find axes -name project.yaml` = **46**（12+10+8+3+6+5+2），与「OK：46 个 project.yaml」一致；各轴 `change_log` 存在 **2026-10-14T20:52–21:05** 的并发写入 | **未跑=不计通过**；且**不再把非本轴路径计为越界**（见 ③-T9 口径） |
| 9 | 抽 5 个数字对落盘 | read/grep | ① U4-s0 clean bpc 1.5314（`results_U4_s0.json:68`、`results.json:66`）✓ ② U2-s0 1.5709（`results_U2_s0.json:68`）✓ ③ APYR-s1 1.5601 + L6 2.4684 ✓ ④ U4 FFN MACs 4,194,304 / `params_ffn` 4,204,544 / 总 7,392,768（`results_U4_s0.json:114-131`）✓ ⑤ Δ_up(s1) −0.0139（`results.json:81`）✓ | **5/5 保真** |
| 附 | 臂间 config 抽查 | read `runs/PYR-s1-12k/results.json` | `mlp_mult 4 / d_model 256 / n_layers 8 / n_heads 8 / seq_len 256 / dense true / seed 1`；`runs/`、`*.pt`、`data/`、`*.log` 被 `.gitignore:8` 覆盖 ⇒ grep 跳过，只能逐文件 read | 抽查 1/7 通过（第 1 轮已逐字段核 7/7） |
| 附 | N 项顺带抽查 | grep | `analysis.md:137` 已给「恰为其两倍」加限定（N2 ✓）；「前后读数」「0.30–0.35」已消失（N3/N1 ✓）；**N4 半改**：`:109` 标了 `(seed 0)`，`:40`（§0 真信号）仍未标 | 见 ③-P2 |
## ③ blocker 与 file:line
- **B-0（唯一阻塞 PASS 的项，流程/证据类）**：本复核者环境**无 shell/执行工具**（工具集 = read/grep/find/ls/watchdog_diff/contact_supervisor），且 `watchdog_diff` 返回 “Reviewer-launch HEAD changed or is unavailable” ⇒ 派单第 1/2/6/7/8 项的**执行**部分全部未跑，T9 未跑；依派单「未跑 = 不计通过」「命令均跑过才 ⇒ PASS」判 FAIL。**不是**在产物里发现了新的数字/结论缺陷。
- **T9 口径（按主 Agent 补充口径，修订上一条报告）**：本工作树**有其它轴并发写入** ⇒ `git diff --name-only 49f697f..HEAD` 出现非本轴路径时**不等于本项 writer 越界**。判定改为：① 只核 2641 实验目录 / `axes/3-width/**` / `.scratch/64-axis3-width/**` 的改动是否与 `e6d2db1..9e9b291`（3.1.4 处置提交）一致；② 非本轴路径按归属记为「并发其它轴的既有工作」并附路径清单；③ **仅当**本轴 `project.yaml`/实验产物出现**未说明**的改动时才计 blocker。
  - 我能做的静态部分：`axes/3-width/**` 8 份 `project.yaml` 的 `change_log` 里，除 `3.1.4` 自身的 `2026-10-13T19:2x`（第 1 轮处置）与全轴统一的 `06:44:03 / 06:59:49`（ticket 57 的框架裁决 + C1 声明回填）外，**未见更晚的、未说明的条目**；`3.1.4` 的 `conclusion/failure_analysis/conclusion_review` 内容与「第 1 轮处置」自述**一致**（内容层面）。**但 commit 级一致性（`e6d2db1..9e9b291` 到底改了哪些文件）无 git 不可核**，故 ① 未成立为证据。
  - ② 并发路径清单（静态证据）：`.scratch/` 下并存 `61-axis2-depth/ 62-axis1-time/ 63-axis6-predict/ 65-axis4-moe/ 66-axis5-quant/ 67-engine-changes/` 等 ticket 目录；`axes/{1-time,2-depth,4-moe,5-quant,6-predict,9-backlog}/**/project.yaml` 的 `change_log` 含 `2026-10-14T20:52:00+08:00`（`5-quant/5.1–5.6`、`6-predict/6.2,6.3.2,6.3.3,6.4`）、`20:52:40`、`21:00:00`（`2-depth/2.1.2,2.2.5`）、`21:05:00`（`1-time/1.2.4`）等条目 ⇒ 并发写入属实（也解释了 `watchdog_diff` 的 HEAD 变化）。**这些不构成本项越界**。
- **第 1 轮 3 处 blocker：静态核验均已真改**：B1 → `analysis.md:33-36`（「|Δ| = 0.0139 已贴种子散布上界 ≤0.0138 ⇒ 不能排除幅度约 0.014 的真实效应…散列不能当 |Δ| 上界」）+ `:104`（「阈值规则判词」）+ `project.yaml::conclusion` ②③ / `::failure_analysis` ①；B2 → `analysis.md:110` = 2.4684/2.7209（落盘 `results_APYR_s1.json:54`）；B3 → `analysis.md:160` = 4,096（2,097,152+4,096 = 2,101,248 ✓）。
- **P2（不阻塞）**：`analysis.md:40`（§0「一个真信号」）仍用未标 seed 的中间深度读数（真实为 seed 0：PYR-s0 L6 1.9727 / L7 1.7492），与 `:109` 已标 seed 的写法不一致；最小修 = `:40` 括注「(seed 0)」。
- **P2（过程工单内部陈旧数字）**：`TICKET.md` §7.2.2 表 `FFN 参数` = 2,099,200（= 2,097,152+2,048）与 §7.3 表 / SSOT 的 2,103,296 不一致；该表是 18:2x 中途快照、工单允许日期流水 ⇒ 非事实冲突，建议加「（中途快照，最终读数见 §7.3）」。
- **我未改动任何被审产物**（无写文件工具，仅只读复核；亦未提交 git）。
## ④ 未跑项与理由（一律不计通过）
| 未跑项 | 理由 | 交 supervisor 的精确命令 |
|---|---|---|
| pytest 13 passed | 无 shell | `cd axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m && python3 -m pytest test_2641_ffn_ratio.py -q` |
| 独立重算 Δ / 判词 | 无 shell（我只做了 read + 手算，不替代脚本） | 读 7 个 `results_<arm>_s<seed>.json` 的 `eval.clean_bpc` 算 Δ_up/Δ_down，与 `results.json::P1_P2_deltas_by_seed`、`P1_verdict_by_seed` 比对 |
| T5 初始化控制 | 无 shell | `import run` → seed 0/1 下 U4 vs U2/PYR/APYR 共 6 组：逐层 assert trunk（embed/qkv/proj/LN）逐位相同 + `mlp[0].weight == U4[:h]`、`mlp[2].weight == U4[:, :h]`，输出逐层布尔 |
| T6 契约冻结 + mtime | 无 git/sha256sum/stat；`watchdog_diff` 不可用 | `git show 49f697f:axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/{README.md,train_arms.sh,test_2641_ffn_ratio.py} \| sha256sum` 对工作树；`stat -c '%y %n'` 三份契约文件 + `runs/*`（须全部早于起跑 18:00:37） |
| T9 越界扫描（新口径） | 无 git | `git diff --name-only e6d2db1..9e9b291`（只核本轴/本 ticket 路径是否与处置一致）；`git diff --name-only 49f697f..HEAD \| grep -vE '^(axes/3-width/\|\.scratch/64-axis3-width/)'`（结果按路径归属分类，**非本轴不计 blocker**） |
| `summary.py --check` | 无 shell | `python3 tools/summary.py --check`（期望 `OK：46 个 project.yaml …`；我静态数到 46，只证数目自洽） |
| 被审 revision / 工作树 delta | `watchdog_diff` = “Reviewer-launch HEAD changed or is unavailable” | 重新 launch 复核者，或提供 `e6d2db1..9e9b291` 的 diff 工件 |
## ⑤ 残留不确定
1. **被审 revision 不明**：`watchdog_diff` 不可用（HEAD 被其它轴并发推进）⇒ 我无法证明「我读的工作树 == 派单所指 revision」；只能说我读到的是 B1–B3 已修后的版本。
2. **契约冻结（T6）纯未核**：`MODEL.md`/`TICKET.md` 的哈希与 mtime 都是 writer 自报（19:07 训练后生成），**不能证明**三份契约文件与 `49f697f` 逐位相同；若有人事后编辑过契约文件（或有并发写者误碰本目录），我的静态核对无法发现。
3. **处置提交一致性（新口径 ①）只到内容层**：`analysis.md`/`project.yaml` 当前内容与第 1 轮处置自述一致，但 `e6d2db1..9e9b291` 的 commit 级文件清单未核 ⇒ 无法排除该段还改了别的本轴文件。
4. `analysis.md:111`（§5.1）现改为由 `T2_…`/`T3_…` 两条测试承载初始化控制（第 1 轮 I2 的处置），但**这两条测试我未实跑**。
5. 静态可核部分的结论：3 处 blocker 已真改、7 臂数字保真、Δ 与判词自洽、结论未越证据（H0 明确为阈值规则判词 + 两条不可排除的反假设限定在位）；**这些不构成 PASS**——只是「另派有 shell 的复核者跑完 ④ 的 6 项且输出一致后，本轮 FAIL 的原因即消除」。
