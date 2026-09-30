# 独立复核（契约 H，fresh context，只读）

## Review（契约 H · 独立复核 · 只读，fresh context）

**范围与方法**：只读检查 `axes/C-vertical/C3-inter-layer-topology/experiments/2613-inter-layer-topology-13m`（E）、`axes/W-width/W4-readout-geometry/experiments/2613-readout-reparam-dense-13m`（F）、`axes/H-horizon/H3-composition/experiments/2613-self-gated-composition-13m`（G）的交付物，逐字对照 `.scratch/30-batch3/issues/01-batch3.md` 契约 E/F/G，并回源核对被引用的历史锚点（2610 / 2611 / 23d-dense-B-12k）。**未运行任何命令**（无 shell），`watchdog_diff` 只覆盖相对 reviewer-launch HEAD 的工作区 delta，E 的引擎改动已提交、不可 diff（下条说明）。

### 逐条 pass/fail（SCHEMA §14 七项 + 本轮特别关注 a–e）

| # | 条目 | 结论 | 证据 |
|---|---|---|---|
| 1 | 判据线是否被改（对照契约原文） | **PASS ×3** | E：README §4「任一变体 Δ≤−0.014 成功 / \|Δ\|<0.014 中性 / Δ≥+0.014 失败；tf@4 ≥0.95；必须附同方法斜率」与契约逐条一致；F：|Δ|、±0.014 三条与契约同 (README §0 保留逐字原文)；G：主判据 `≥max(direct,composed)+0.014`、并列草稿接受率 `≥0.90`、oracle「不作判据」逐字保留，代码 `compose_verdict()` 只做归类。**三条判据线均未被改动。** |
| 2 | 三道对账（§12） | **PASS ×3** | E §6.1–6.4、F §6.1–6.3、G §5①②③ 均写全算术/任务/口径；抽查算术全部成立（见「Correct」）。三条线都报的是**非有利数字**（无改善/中性/否证），仍按 §12 做满。 |
| 3 | 对照臂三要素 + 同一任务定义 | **PASS（E 记 note）** | F §6.2、G §5② 有显式三要素表（输入分布 / 预测什么 / 部署对应）；E §6.2 是同口径叙述（四臂同一数据流、同一位置集、同一自移位目标），未做成表。均无跨臂位置集差异（G 四臂与对手共用同一 eval seed/批序，`ar1` 的 argmax 只算一次）。 |
| 4 | 干预活性非零、与日志一致 | **PASS（两处 note）** | E dense：α fro 3.3049 / absmax 2.3615 / 非零 1.0 / 偏离 identity 1.3615；**我用 results.json 里逐层 α 反算 √10.9220 = 3.3048 ✓**；smoke 里 20 步已逃逸 max_dev 0.001114（与 analysis §3.1 的 0.0011 一致），且 dense/sequential 第 0 步 loss 均 5.3131、gnorm 均 7.86 ⇒ 干预活着且起步逐位等价。F：P fro 1.451 / Q 0.8918（Q 从 5.1253 收缩）、变换相对幅度 0.0089–0.0124、smoke `p_fro 1.0903`（与 analysis「0.0→1.09」一致）。G：g 均值 0.8251(seed0)/0.8157(seed1) 起点 0.5、q10 0.577/0.560、逐桶命中单调 0.17→0.68。注① E/G 的活性块来自 checkpoint 终值，**训练日志里没有逐 200 步的活性序列**，无法「与日志逐点交叉核对」（G 的 `intervention.alive` 只检查 T 的非零率，未检查门；门活性靠 `gated_g_at_eval` 另证）。 |
| 5 | results.json ↔ analysis.md 数字逐一对应（抽查 ≥8） | **PASS** | 已核对 ≥15 对：E 1.5254/1.5712/1.5656/1.5246、Δ −0.0008/+0.0458/+0.0402、参数量 6,962,688/6,958,592/6,962,688/6,962,724、α 3.3049/2.3615/1.3615、四臂斜率 0.00251/0.00390/0.00452/0.00450、tf@4 0.4560/0.3486/0.3149/0.4538、税 −0.0033/+0.0425/+0.0369/−0.0041、步时 0.0325/0.0300/0.0325/0.0400、wall_total 27.12、峰值 1.47/1.43/1.43/1.53；F 1.5257/1.5312、Δ −0.0055、params 131,072、P/Q fro、幅度 0.0089–0.0124、斜率 0.00297/0.00179、seed1 1.5298/1.5324/Δ−0.0026/p 1.3467/q 0.8579；G 0.4672/0.4718/0.4738/0.4785、Δ+0.0020、accept 0.6113、ar1 0.5489、oracle 0.6953、trunk_t1 0.6934、naive 0.0684、params 262,912/394,496、g 0.8251/0.1548/0.5769/0.8669/0.9915、桶命中、4000 步 0.4788/0.4854/0.4896/0.4882、选择性 24,963@0.9043 / 5,194@0.7605。**未发现对不上的数字。** |
| 6 | 口径标注 | **PASS ×3** | 三条线均标 matched-step 快照口径（E：12k/98.3M token/seed0/n=1；F：12k + seed1 复现 + 「非渐近/非墙钟/非 FLOPs」；G：2000 头步快照 + 4000 头步收敛点 @ 冻结 12k 主干），并各自标了 tf@k 的 proxy 性质与 E 的 tf@4 不可达。 |
| 7 | pytest 全绿 / 既有实验未被破坏 | **无法验证（留给主 Agent）** | 我无 shell。静态计数：`tests/` 现有 **165** 个 `def test_`（基线 150 + E 9 + G 6），与 E 的「159 passed」、F 的「159 passed」自洽。**请主 Agent 跑 `python3 -m pytest tests/ -q`（期望 165 passed）**；另 `python3 tools/check_schema.py`、`python3 tools/build_wiki.py`（见 P2-8）。 |
| a | 等参数对照是否**逐位等参数** | **PASS** | 参数量逐位核对通过：E `parallel-dual-ln` 6,962,688 **=** `sequential`；`parallel` −4096 = −L·2d（少一个共享 pre-LN）；`dense` +36 = L(L+1)/2。G `direct` = `composed` = 262,912（同一个 `self.T` 实例，T∘T 不多参数）；`gated` = T 262,912 + G(fc1/fc2 d→d) 131,584 = **394,496** = `direct_eq`（3d 宽 MLP 6d²+4d + 逐通道偏置 d = 6d²+5d）。F 的 +131,072 = 2·d² 属「干预本身」，且 F 不声称增益（§15 不触发，analysis §7.7 已自陈）。E 的 smoke 给出 `nonzero_element_diff_vs_sequential` 全 0（共享张量逐位相同）。 |
| b | 是否用 oracle / 真值当判据 | **PASS** | E/F 判据是 clean bpc（无 oracle）。G 的**主判据是契约自己预登记的「t+2 命中 vs 真值」**，但 executor 明确把它标为质量口径、并同时给出 §11 的诚实量（草稿接受率 = 头 argmax == 主干自身 AR-1 argmax = 0.6113，对照物不是真值）；`oracle` 落在 `criteria.oracle.used_as_criterion=false`，只作上界。判决（中性/未达成）在两种口径下同向 ⇒ 不构成「oracle 当判据」。 |
| c | 快照增益是否给**同方法**斜率 | **PASS** | E 四臂同「末 25% 窗 + 同一 OLS」；F 两臂同「末 25%、16 点、9000–11999」；G 四臂同「每 200 步日志 + 末 25% 窗 + OLS」。G 还额外交了 2× 点（4000 步），并自陈 2000 步那行只有 3 点、噪声主导、不作排序证据。无「一臂 25% 窗、另一臂 50% 窗」。 |
| d | dense 层权 / 自门控 g 是否真活着 | **PASS** | dense α：36 个全非零、absmax 2.3615、对最早层的负权 −0.17~−0.27、α 起步 identity + 训练侧第 0 步逐位一致；g：起点恒 0.5（单边零，单测钉住）→ 终值 0.8251，`frac_prefer_two_hops 0.9668`，4000 步仍在移动（0.6934）。 |
| e | 有无「假上下文 + 真目标」式对照 | **PASS** | G 的诚实对手 `ar1_second_step_readout` 只替换紧邻位置 i+1（真值前缀 0..i 保留），且有 `test_ar1_matches_bruteforce_rowloop` / `test_ar1_prefix_keeps_ground_truth_not_whole_context` 钉住；G §5② 明确把 ticket 28 的「整段替换」版（0.3128）与诚实版（0.5489）区分开。四臂输入均为真值前缀教师强制、目标为 t+2，位置集相同、`ar1`/oracle 只算一次。 |

### Correct（已核实的正确项）

- **判据线未被改动**（E/F/G 逐条对照通过），且判决与证据一致：E = 无成功臂（`parallel` +0.0458、`parallel-dual-ln` +0.0402 否证；`dense` −0.0008 中性）；F = 不可迁移（−0.0055 / seed1 −0.0026，均 < 0.014）；G = 中性（+0.0020 / 4000 步 +0.0042，< 0.014）。
- **历史锚点回源核对全部为真**：2610 `jump_t2_hit` 0.4667、`trunk_t2_naive_hit` 0.0684、`trunk_t1_hit` 0.6934、oracle 0.6953；2611 `ar1_deploy_t2_hit` 0.5489、`accept_jump_t2` 0.5998、`delta_bpc_linear_vs_A` −0.0888；`23d-dense-B-12k` 1.5287/0.6894（E 用当前代码重测逐位复现 1.5287/0.6894）；2610 `trunk.pt` sha256 `86953dcb…59d2` 与 G 的 `config.json`/`MODEL.md` 一致、`sha256_verified: true`。
- **§15 归因做对了**：G 的等参数无门对照 `direct_eq` 0.4785 **高于** `gated` 0.4738 ⇒ 那 +0.0020 被正确归到参数而非门控结构；E 的并行族用逐位等参数的 `parallel-dual-ln` 复现同一负结论（两臂差 0.0056 < 0.014）；F 不触发 §15 并说明理由。
- **活跃量数值我独立重算通过**：α fro √(Σα²)=3.3048（报 3.304855）；E `w_fro 起步 √8 = 2.8284`、比例 0.0033/0.014=24%、dense 步时 +23%、并行 −8%、等效加速 1.456/1.5=0.9707 全部自洽；F Q 初始 fro 256×0.02=5.12 ✓。
- **冒烟/变异证据齐全**：E 三个非基线拓扑各 20 步 CLI 冒烟 + 参数量账 + 恒等起步逐位 + 接线偏差（parallel 0.497 vs sequential 0.0）+ 三处变异全抓红（`mutation_report.json` 记 sha256 前后一致，`79d3cc59…a5f07`）；F 四处变异抓红 + `common_params_bitwise_equal` / `identity_start_logits_bitwise_equal` / `linres_equals_gated`。

### Finding（无 P0/P1；以下为 P2，只报不改）

1. **P2 — E：契约副本标注「逐字」但实为摘要。** `…/2613-inter-layer-topology-13m/README.md` §0 自称「逐字抄下」，实际加了「背景：」段、删了「成本：3 臂 × ≈13 min」「交付」两行、`可改` 列表也压缩（少「新增 block 拓扑变体 + 开关」）。判据线内容本身一致（已逐条比对），故只是标注不准。最小修法：`§0` 改成直接引用契约原文，或把「逐字」改成「摘要（判据线逐字）」。
2. **P2 — F：判据线「逐字」标注与符号改写。** `…/2613-readout-reparam-dense-13m/README.md` §4 写「逐字来自契约 F」，但它把契约内部符号不一致（同一段里「Δbpc ≥0.014 ⇒ 可迁移」与「Δ ≥ +0.014 ⇒ 负迁移」冲突）显式改写为「Δ=改善量：≥+0.014 可迁移 / ≤−0.014 负迁移」。契约原文在 §0 完整保留，实测 |Δ|=0.0055 在两种读法下结论相同 ⇒ 无实质影响。最小修法：§4 改标「契约符号自冲突，此处按『Δ = bpc_baseline − bpc_arm』统一」。
3. **P2 — F：未报实测墙钟，且略超声明预算。** 契约预算 26 min；实测 `run_log.txt` 14.07 min + `run_seed1.log` 12.38 min ≈ 26.45 min，再加 `smoke.py` ≈1 min ⇒ ≈27.5 min（约 +6%）。加跑 seed1 是仓库硬纪律「n=1 负结果必须复现」要求，合理，但 README/analysis 里没有一条「实测 GPU 墙钟」记录（E 有 `wall_total_min 27.12`，G 有 `cost.gpu_wall_minutes 6.66`）。最小修法：README §6 增一行「实测 ≈27.5 min（含 seed1 复现）」。
4. **P2 — E：冒烟里有一个 10× 假数字。** `smoke/smoke.json::cli_effective.*.min_per_12k_extrapolated ≈ 60.85/60.9 min`，与实际全跑 6.5 min 差 10 倍（20 步里进程启动占主导）。analysis 未引用它（引用的是全跑 `seconds_per_step`），但该字段留在证据文件里容易被后手误引。最小修法：删掉该字段或标注「含启动开销，不可外推」。
5. **P2 — G：没有稳定性块/`stability.md`。** E/F 都有 gnorm/NaN/loss 突跳（E 还有专报），G 的 `results.json` 无 stability 字段、目录无 `stability.md`（12 次头训练、冻结主干，风险低：全目录无 `nan|inf`）。最小修法：补一段「12 次头训练 loss 全程有限、单调下降」的说明。
6. **P2 — G：`intervention.alive` 的定义比看上去弱。** `train_jumphead.py::run_composition` 里 `alive` 只断言四臂 T 的 `nonzero_frac > 0`，**没有查门**；门的活性实际由 `gated_g_at_eval`（g 0.8251 ≠ 起点 0.5）证明。结论无误，但读 `alive: true` 会误以为「门被子断言覆盖」。最小修法：`alive` 里加 `gated_g_at_eval.mean` 偏离 0.5 的条件。
7. **P2 — G：契约的「G 单边零初始化（起步等价 direct 或 composed）」在 sigmoid 门下不可满足。** executor 取 SCHEMA §10 的单边零 ⇒ 起步 = direct/composed 的 50/50 混合，已在 README §0/§4、analysis §7.1 显式记账，判据线未改。仅作对照可比性提示（gated 的起点既不等于 direct 也不等于 composed）。
8. **P2 — 索引导航卫生（属主 Agent 范畴）。** `axes/W-width/W4-readout-geometry/experiments/README.md` 的 **AUTO 块被手工加了一行**（2613 行）；同时 `axes/H-horizon/H3-composition/experiments/README.md` **仍缺** 2613 行。`experiments_page()` 是按文件系统扫目录生成的，所以前者的手改内容与生成结果恰好一致（该文件本身不会漂移），但后者会让 `python3 tools/build_wiki.py --check` 现在为红——这正是批次说明里「索引未刷新预期红」的情形，请主 Agent 跑一次 `build_wiki.py` 统一刷新（C3 那份已含新行）。
9. **P2 — E：整体判决用词（非数字问题）。** 契约的三档是**逐臂**定义，executor 的 `verdict` 写「失败（任一臂更差 ≥+0.014）」而在 §4 同时把 `dense` 记为中性——两种读法都在文里说清了，仅提示后续汇总（catalog/课题页）时不要把「E = 失败」读成「dense 也被否证」。
10. **P2 — 复核盲区：E 的引擎改动已提交，无法做 diff 级复核。** `pathlm/depth_ar.py` / `train_depth_ar.py` / `tests/test_depth_ar.py` 不在工作区 delta 里（`watchdog_diff` 只显示 `train_jumphead.py`、`tests/test_jumphead.py`、W4 `experiments/README.md`），因此我只核到**当前文件内容**（拓扑分支、`ParallelBlock`、`_layer_mix_input`、`topology_liveness`、v1 逐位还原路径与守卫）与测试存在性，**未复核提交区间**。若需 diff 级复核，请提供 E 的 commit 范围。

### 可疑数字清单（逐条结论）

| 数字 | 出处 | 结论 |
|---|---|---|
| `min_per_12k_extrapolated ≈ 60.9 min` | E `smoke/smoke.json` | **假数字**（启动开销主导，真实 6.5 min）；未被结论引用，建议删除 |
| tf@4 判据线 `≥0.95` | 契约 E 并列判据 | 阈值本身没错，但 stock dense 形态基线只有 0.444–0.456 ⇒ **该并列条件零区分度**，executor 如实报「四臂同不可达、不作区分量」，处理正确 |
| E `sequential` 1.5254 vs 历史 1.5287（Δ0.0033） | E §1 | 与 F 的 −0.0030 同量级，属联合跑底噪；两处都被用来标定「0.014 阈值有富余」，不构成矛盾 |
| F 20 步 smoke `loss_last` 3.3977 vs 3.6945 | F `smoke/smoke.json` | 步数极短的中间态，未用作证据，无风险 |
| G 2000 步斜率 +0.1619…+0.1800（/1k） | G results/analysis | executor **自己**判定「3 点、单批 loss、噪声主导、不可作排序证据」，改用 4000 步行 ✓ |
| G `0.4785 > 0.4738`（等参数无门对照更高） | G results | 真实且与 §15 归因方向一致（增益记容量、不记门控）✓ |

### 结论

**Merge verdict：E — OK with notes；F — OK with notes；G — OK with notes（合计：无 P0/P1 阻塞项，10 条 P2 报告项）。** 三条线的判据线未被改动、三道对账做满、等参数对照逐位成立、无 oracle 当判据、无「假上下文 + 真目标」对照、快照增益均有同方法斜率、干预（dense α / 自门控 g / 读出新自由度）均可证活着。需要主 Agent 承接的两件事：**（1）跑 `python3 -m pytest tests/ -q`（期望 165 passed）并确认既有实验未破；（2）跑 `python3 tools/build_wiki.py` 刷新 C3/W4/H3 索引（当前 H3 索引会让 `--check` 红）。**

按 reviewer 政策，我没有用 `contact_supervisor` 发完成汇报（该通道只用于 need_decision / interview_request / progress_update）；完整报告即本回复正文，请主 Agent 落盘。