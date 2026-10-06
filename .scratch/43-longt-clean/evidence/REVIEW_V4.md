# 独立复核 V4（ticket 43）：W1（QK-Norm 修复）/ W2（判决腿方差）

## Review

**范围与方法.** 只读复核 ticket 42 的 W1（`axes/A-mixing/A6-mixing-operator/experiments/2625-longT-stability/`）与 W2（`axes/M-carrier/M1-carrier-type/experiments/2625-ood-leg-variance/`）。`watchdog_diff` 在 reviewer-launch `HEAD afdc0d08`：全仓仅两条 `.scratch/43-longt-clean/issues/` 为未跟踪；**两个目标目录的范围查询均返回“No working-tree changes”** ⇒ 我读到的文件内容 = 该修订的已提交内容。**限制**：`watchdog_diff` 不含已提交区间，我**无法**逐个核对 `d7f4628`/`25bc48f`/`52c2064` 的区间内容与提交信息，只能按“内容 + 是否入库”复核（见末尾待办命令）。我无 shell，未跑 pytest、未跑 `git ls-files`；改用 `.git/index` 路径匹配（含负对照）代替 `git ls-files`。

### Correct（逐条验证过）

**① QK-Norm 确为零新增参数 + 逐头 L2 归一。** `stability_driver.py::apply_attn_scale`（:54-92）只 `register_forward_hook`，全程不创建 `nn.Parameter`/buffer；hook 把 q/k `reshape(..., n_heads, dh)` 后对最后一维做 L2 归一、再乘 √dh 拼回（:68-80）⇒ 结构上不可能加参数。单测 `tests/test_longT_stability.py::test_qk_norm_gives_unit_scale_heads`（:114-138）断言**每个头**的 q/k 范数 == √(d/h)（atol 1e-4）。变异 `W1-M2`（`mutate.py:60-63`）把 hook 换成恒等 ⇒ 该单测抓红；`mutation_report.json` 记 6/6 抓红、`restored_bitwise=true`（前后 sha256 相同）。载体是 `--dense` stock 7,159,296 参数模型（与 2623/2624 基准同值，`2623-longT-1024/full-1024-train_summary.json`），共享文件未动（`MODEL.md §3` 四道 sha256；`depth_ar.py` 的 `3a0db55…` 与 W3 复核里独立记录的 Y2 指纹一致）。

**② “T≤512 零 hook”是结构性的，不是靠 Δbpc 推断。** 代码在 `min_t>0 and model.cfg.seq_len<=min_t` 时**提前 return**（`stability_driver.py:65-67`，`active=False, n_qkv_hooks=0`），根本不注册 hook ⇒ 前向/反向图与基线逐算子相同；单测把它钉死（`tests/test_longT_stability.py:67-84`：seq_len=16/min_t=512 时 `depth_logits` 逐位相等，min_t=8 时 `n_qkv_hooks==L`）。运行侧三处一致：cond 臂确实是 seq_len=512（`eval.json::arms."full-T512-cond512-s0".seq_len=512`、`full-T512-cond512-s0-train_summary.json`），其 `gnorm_max=6.90` ≈ 2622 基准 6.90（而全局 qk-norm 在 T=512 是 6.65），`reconcile_numbers.py:293-299` 还核了 `patch.n_qkv_hooks==0`。Δbpc=−0.001603 只作次级确认（在 run-to-run 噪声 0.004534 内），报告也确实这么写。

**③ W2 极差与产物一致；降级只落在超训练距离档。** 19/142/241（n=2048）⇒ 0.928/6.934/11.768%；我手算极差 10.840、均值 6.5433、sd(ddof=1) 5.4305，与 `results.json::main_leg_variance` 逐位一致；四跑 14.404/8.740/6.243 亦可复算。Wilson CI 手算 19/2048=[0.005947,0.014445] 与产物一致。参照臂**未并入** n=3（`verdict.primary_arms` / `main_leg_variance.names`）。降级范围正确：`verdict.leg` 钉死到该腿；`in_dist_leg.high_variance=false`；`analysis §5/§6`、`controls/README §1` 明写抖动来自**距离档（>240）**而非任务，并**显式禁止**把该腿与分布内腿合并成“有效/无效”；交叉口径（去窗口、241–255，极差 11.865pp）进一步支持“是距离本身”。仪器可回溯：2624 已提交的 `carry_gqa_fp16 = 19/2048 = 0.009277` 与本线 s0 逐位相同。

**④ 数字↔产物（抽查 >20，含成本/次数/等式/哈希）：W1** bpc 1.543248/1.543707、T256 1.535669(Δ+0.010280)、T512 1.605401(Δ+0.031381)、cond 1.572417(Δ−0.001603)、基准 1.525389/1.574020/1.510446/1.606426 ↔ `eval.json`；gnorm_max 7.25/7.58、mean 0.539/0.553、跳步 0、峰值 1.015GiB、free 21807→21812 ↔ `full-T1024-qknorm-s{0,1}-train_summary.json`；基准 1740.92/6.533/0 与 29578.57/22.946/3 ↔ 2623/2624 summary；**“同配方重跑 max|Δloss|=0.0320”我用已提交 `curves.json` 独立复算**：step0 两跑 loss 均 5.3344（逐位同），step200 = 2.5621 vs 2.5941 ⇒ Δ 0.0320 ✓；tokens 98,304,000（12000×8×1024）、诊断 57,344,000、logits 上界 √32=5.657 等式自洽；`curves.json` 含 8 个改动臂。**W2** sha256 四条（`02c390d8…`/`48c4e39e…`/`896e9e55…`/`23969f29…`）↔ `MODEL.md`/`config`；n_params 12,616,808；eval 峰值 580.4 MB；训练 12.26/11.04 min、峰值 2193.6 MB ↔ `train/gqa_carry_win_s{1,2}/results.json`；read_share 0.863936/0.709234、答案行 CE 1.6407/1.6039、斜率 −0.07141/−0.06297、gnorm_max 51.21/57.41 ↔ `train_activity`；距离曲线 d=255 = 6.25/46.88/56.25/89.06%（n=192）↔ `distance_leg`；KV 头形状 [8,2,256,32]。**变异/对账**：W1 `reconcile_report.json` 154 检查 0 失败、W2 132 检查 0 失败；W2 `mutation_report.json` 6/6 抓红且逐字节还原、目标 sha256 与 MODEL.md 一致。

**⑤ §20.1/§20.2。** W2 `train_arms.sh` 按臂取/放 + 臂间 65s，**已提交** `train_arms.runlog` 时间戳可见窗口（21:27:50 → 21:28:55）；W1 `run.py` 按臂 `gpu_acquire/gpu_release` + 臂间 `time.sleep(60)`（:179/189/206）。两者峰值（2193.6 MB / 1.015 GiB）均 <5GB ⇒ §20.2 三件套**未触发**；W1 的三件套（峰值 + free_before/after）还落在**已提交**的 `*-train_summary.json` 里。

**⑥ 声称的交付物是否入库。** 用 `.git/index` 路径匹配（负对照 `UNLIKELY_TOKEN_XYZ_123` → 无匹配，证明可判别）：W1 声明的非 `runs/` 交付物（`stability_report.md`、`curves.json`、`eval.json`、8 个 `*-train_summary.json`、`smoke/*.json`、`tests/test_longT_stability.py`、`mutation_report.json`…）**全部命中**；`runs/**` 已在报告首段显式声明“不入库”，并给了可入库副本 `curves.json`/`eval.json`。W2 的 `results.json`、`train/<arm>/{train_log.jsonl,gate_log.jsonl,results.json}`、`train_arms.runlog`、`controls/README.md`、`smoke/`、`MODEL.md`、`tests/test_ood_leg_variance.py` **全部命中**。唯一缺口见 Finding 1。

### Findings（全部 P2，report-only；不影响任何结论）

1. **P2 · W2 声明的一个交付物不在 `git ls-files`。** `README.md §7（落盘）` 列出 `train_<arm>.log`，但 `.gitignore` 有 `*.log`，且 `.git/index` 里查 `train_gqa_carry_win_s1.log`、`train_gqa_carry_win_s2.log` **均无匹配**（同法查 `train_arms.runlog`/`controls/README.md` 有匹配）⇒ 两条 stdout 日志未入库。影响很小（其内容与**已入库**的 `train/<arm>/train_log.jsonl` + `train_arms.runlog` 重叠，报告没有任何数字只依赖它）。最小修法：把 §7 改成“stdout 日志（不入库）”，或 `git add -f` 后补账。**这条正是上两轮 P0 的同类项，但缺的是冗余日志而非必需单测/证据文件，故按前次同类判例记 P2。**
2. **P2 · W2 报告↔结果的评测墙钟漂移。** `analysis.md:189` 写“评测墙钟 1.05 min”，`results.json:2573 cost.wall_minutes_total_eval = 1.115`（`provenance.run_log` 末条 t=66.92s=1.115min 自洽）。`reconcile_numbers.py` 只核了 580.4 MB，未覆盖该字段。最小修法：改为 1.115。
3. **P2 · W2 的 §20.2/§20.3 记账证据指错了文件。** `README.md:101` 说“跑前/跑后 `nvidia-smi memory.free` 见 `train_arms.runlog`”，但该 runlog 只有锁时间戳；`analysis.md:190` 的 20290/20292 MiB 与 `README.md:98` 的 20741 MiB/util 56%/129 ms/step 只存在于控制台输出（`measure_step_time.sh` 产物落 `/tmp`，未入库）。**峰值本身没问题**（2193.6 MB ≪ 5GB，§20.2 三件套未触发），但步时定预算（§20.3）链条在已提交修订上不可回溯。最小修法：把步时实测输出落一个 `smoke/steptime.json` 并改掉指针。
4. **P2 · W1 步时/墙钟与已提交 summary 不一致。** `stability_report.md:269-270` 写“seed0 纯训练 9.55 min / 12000 = 48 ms；seed1 10.8 min / 12000 = 54 ms”，而**已提交** `full-T1024-qknorm-s0-train_summary.json` 是 `wall_min 9.67 / seconds_per_step 0.04834`，s1 是 `11.0 / 0.05501`。报告里的数疑似取自未入库的 `runs/<arm>/results.json`（drivers 内计时，不含数据加载）。最小修法：统一到已提交 summary 的 9.67/11.0 min（48/55 ms）。
5. **P2 · W1 探针表 L2 行疑似抄写错误，且不可核。** `stability_report.md:204` 的 L2 行在 step 2000（23.53）与 4000（25.94）**与 L1 行完全相同**，而在 800/6000/6800 各不相同（10.93/26.33/27.57 vs 4.84/50.02/86.13）——两处连续精确相同要么是抄写错、要么是极不可能巧合；源 `runs/full-1024-s1-diag/probe.jsonl` 未入库，`reconcile_numbers.py` 也没核这两个点。该行只作背景（H1 的结论靠 L1 的 0.60→86.13 与其余层平台），不影响判据。最小修法：把该行退回原始 probe 核一遍，或把逐层表导成可入库副本。
6. **P2 · W1 的“驱动器保真度锚点”单测在干净检出下会静默 skip。** `tests/test_longT_stability.py:224-226`：证据文件（`2625/runs/preflight-20/train_log.jsonl` 与 `2623-longT-1024/runs/full-1024/train_log.jsonl`）都在被忽略的 `runs/` 下，缺文件时 `pytest.skip("证据文件缺失")` 而非失败 ⇒ 该“可失败单测”在 CI/干净检出上变成不计分的空断言（本地因两目录 `runs/` 都在，实际会跑）。最小修法：把 step0 的两行读数固化进一个入库的小 json/md，供该单测对比。
7. **P2 · W1 冒烟文件里确有“20 步派生时间字段”。** `README.md §7` 声称 preflight “**不含任何 20 步外推时间字段**（§17.5）”，但已提交 `preflight-20-train_summary.json` 有 `"seconds_per_step": 1.00284`（20 步、启动开销主导，真值 ~44 ms）。报告没引用它（用的是 300 步实测 44 ms/step），故无结论影响，但正是 §17.5 点名的误引风险。最小修法：冒烟摘要里去掉该字段或注明“20 步，含启动开销，不作步时口径”。
8. **P2 · W2 人话摘要的百分数与它自己给的参照不自洽。** `analysis.md §0`：“KV 头 8→2 ⇒ 省 **8.7%** 参数（12.62M vs 13.54M）”。括号里的两个数相减只支持 **6.80%**（919,960/13,536,768）；8.71% 对应的是**孤立** GQA 效应（2×(65,536−16,384)×12 = 1,179,648 / 13,536,768）。两个数都“真”，但括号支撑不了那句话。最小修法：写成“GQA 单独省 8.7%（1.18M/13.54M），叠加 carry 的 +0.26M 后净省 6.8%”。结论不受影响。
9. **P2 · W1 没有 §23 七问的逐条回答（W2 有）。** W2 `analysis.md §1` 有完整七问表；W1 只在 `stability_report.md:286` 标了“§23 第 6 问”。若本波要求 W1 也逐条落盘，补一张表即可（信息其实散在 §5/§6/§9 里）。

### Merge verdict

**OK with notes** —— 六项重点**全部实质成立**（①②③④⑤⑥），无 P0/P1；9 条均为 report-only P2，不改变任何判据、数字结论或状态灯。建议把 Finding 1/2/4/5/6 随 ticket 43-V3 的 P2 收尾一并处理。

**请 supervisor 跑的命令（我无 shell）：**
- `python3 -m pytest tests/ -q` —— 期望全绿（ticket 43 基线 490）；注意在干净检出上 `tests/test_longT_stability.py::test_driver_preflight_step0_matches_engine_2623` 会 skip（Finding 6）。
- `git ls-files axes/M-carrier/M1-carrier-type/experiments/2625-ood-leg-variance | grep -c '\.log$'`（期望 0）与 `git check-ignore -v axes/.../train_gqa_carry_win_s1.log`（确认 Finding 1）。
- `python3 axes/M-carrier/M1-carrier-type/experiments/2625-ood-leg-variance/reconcile_numbers.py`（期望 rc 0，132 检查）与 `python3 axes/A-mixing/A6-mixing-operator/experiments/2625-longT-stability/reconcile_numbers.py`（期望 rc 0，154 检查；会重写 `curves.json`/`eval.json`，需在工作树干净时跑）。
- `git show --stat d7f4628 25bc48f 52c2064` —— 核提交区间与提交信息（超出我可用工具范围）。

**残余风险：** ①内容按 `HEAD afdc0d08` 复核，未做区间级（per-commit）归属，提交信息未核；②W1 的 `runs/**`（逐臂 `results.json`/`probe.jsonl`/`train_log.jsonl`）不入库是**已声明**的，代价是干净检出无法重跑 W1 的 `reconcile_numbers.py`，且 Finding 5/6 只能凭表内/代码本身判断；③W2 的“分布内腿仍可用于 n=1 判决”是 n=3/6k 单点推断（报告已标禁外推）；④W1 的“T=1024 上 cond 与全局档同一计算”是结构论证（hook 代码相同、门条件成立），T=1024 的 cond 实跑当时未做（ticket 43-V1 正在补）。

> 交付方式说明（依 reviewer 政策）：本复核无阻塞决策需请示，故未发 `contact_supervisor`，完整报告按上文内联返回。