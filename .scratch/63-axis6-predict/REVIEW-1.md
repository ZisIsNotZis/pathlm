## Review（WP 6.2.2 · 2640-tree-conf-13m · 独立复核 #1）

### 0. 复核对象 revision 与工具限制（先声明，影响下面每条的置信度）

| 文件 | sha256（**生产者自报，我无法重算**） | 来源 |
|---|---|---|
| `2640-tree-conf-13m/results.json` | `1302b6785630f32815651a02340ecd72b556c7e5ec44657311994ee74c13dbaa` | `.scratch/63-axis6-predict/run1-snapshot/README.md`（17:32 三方比对） |
| `2640-tree-conf-13m/cycle_log.jsonl` | `d2073370c7985c7b1bc0981e6d8a73231462674bbd2ea49ae0f54ef48a592290` | 同上 |
| `2640-tree-conf-13m/run_log.txt` | `ba48b06a0bcee2420c119c61a6c2687dee544225d1e30c29b60fb1a600a2984c` | 同上 |
| `2640-tree-conf-13m/spec_tree.py` | `71d98b2687b2cc36d362514bd8bcc014aa3818f4aad4ef14c83b56e9be8fb62f` | `mutation_report.json::targets`（变异前后逐字节还原值） |

- `git rev-parse HEAD`：**不可得**（本复核上下文无 shell、无 Git 权限）。工作树脏项清单：`watchdog_diff` 返回 `Reviewer-launch HEAD changed or is unavailable` ⇒ **无法给出**；我改用的替代证据见 C。
- 复核者：axis6-lead 派出的 fresh reviewer，未参与本 WP 任何写作，**只读**，未改任何文件。
- 复核时间：**无时钟可读**（无 shell）。按派发时序（`TICKET.md` §7：17:31 派 reviewer，末条记录 2026-10-13T17:32:11+08:00），本复核执行于 2026-10-13 **17:32+08:00 之后**；此时间戳**非机器核验**。
- 预算：硬盒 ≤25 min；实际约 20–25 min（估计，无时钟）。**实际耗时未能精确测量**。
- **本上下文没有 shell / 写文件工具**：因此「自己写脚本复算」「自己跑 `mutate.py`」「`sha256sum`」「`git diff`/`git status`」「`pytest`」这几类**必须落盘或执行**的证伪手段我**做不了**。我用①源码逐行读、②`results.json`/`criteria_fixed.json`/`recompute_check.json`/`cycle_log.jsonl` 原始数据的**手工算术一致性核对**、③**两套互相独立实现**的输出互印证（writer 的 `recompute_criteria.py` vs axis-lead 的 `recompute_check.py`）替代。凡我未能真正执行的条目，下面逐条标「未能自跑」，并给出补齐命令。

---

### 1. 发现清单

**F1 · normal · `analysis.md` §4 C1（与 `WP-6.2.2-LOG.md` T4 同错）：一个数值 10 倍错**
- 位置：`axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/analysis.md` §4「C1」小节（「探针点 0.8922 **落在**本臂 α₁ 的 95% CI 之外（CI 下界 0.9008 略高于探针点 0.00086）」，同段末「（下界差 0.00086）」）。
- 违反的预登记条目：`README.md` §4 C1（要求如实记录探针点与实测 CI 的关系）。
- 证据/复现：`results.json:20440-20442` → `13m_s1.tree_direct_k2.alpha1_ci = [0.900752, 0.942632]`；探针 = `2632-alpha1-attribution/probes_output.json:156` → `P3_free_gen_topk_vs_T2["2"]["direct"] = 0.8922`。差 = `0.900752 − 0.8922 = 0.008552 ≈ 0.0086`，**不是** 0.00086（差 10 倍）。同一错误数字出现在 `.scratch/63-axis6-predict/WP-6.2.2-LOG.md` T4。
- 影响：C1 的定性结论（探针点落在 CI 外、但两 CI 重叠、方向为「实测更高」）**不变**；只有这个「差多少」的量级错了（真实差距比文中写的大 10 倍，方向仍不触发 README 的 bug 判据）。
- 最小修法：把两处 `0.00086` 改为 `0.0086`（analysis.md 是事实文档，必改；LOG 是流水，可留痕但建议同改以免再次被引用）。

**F2 · normal · 代价轴口径未披露「K 行各带完整 [v1, c_i, d3] 链」⇒ `R=K(1+γ)/(1+τ)` 是**实现口径**、对真树注意力略偏保守（D1）**
- 位置：`analysis.md` §1（「抬 τ 的代价是每轮把 K 个候选送进同一次**批量**前向（K 倍 FLOPs）」）与 §5 P5 表；契约 `README.md` §4 P5（`B_verified` = 每轮候选行数 K）。
- 证据：`spec_tree.py:tree_forward` 的行构造 `rows = [[v1, c, d3] for c in cands]`（`spec_tree.py` 的解码主循环段，`decode_tree` 内），每行 3 个位置 ⇒ 每轮主干新位置数 = **3K**；共享前缀只是 KV 广播（不重算）。真·树注意力（节点共享）下不同节点数 = 1(root) + K(候选) + K(候选各自的 d3 子节点) = **2K+1**，即本实现比理想共享树多算 ≈1.2×（K=2：6 vs 5）到 ≈1.33×（K=4：12 vs 9）的位置。文中把 1.86×/3.65× 直接写成「树的代价」，**未声明这是本实现（K 行平铺）而非树结构的固有代价**。
- 定级理由（按题目「是否误导读者判断值不值」）：方向是**保守**（把树写得更贵），不夸大结论；且 P5 预登记为「如实报、不作判据」，所以不是 blocker。但读者与业界（Medusa 式树注意力）对比时会高估树的价格。
- 最小修法：§5 加一句「K 行平铺 ⇒ 每轮 3K 个主干位置（v1/d3 每行各算一次）；若按节点共享的树注意力（≈2K+1 个位置）算，FLOPs 倍数约为本文的 0.7–0.8 倍；τ 边际收益随 K 递减的结论在该修正下不变」（我按 §5 的 τ/R 数值算过：K=2→4 时理想口径成本比 ≈1.8×，而 tok/fwd 只涨 ≈2%，结论不翻）。

**F3 · normal · 研究项 SSOT（`6.2/project.yaml`）在本次复核 revision 上与新读数不自洽**
- 位置：`axes/6-predict/6.2-speculative-decoding/project.yaml` 的 `conclusion`（「**主判据（冻结主干 + 树 / 多候选 + 置信加权下 τ 抬多少、质量是否越带）待 6.2.2 补跑**，故 status = 进行中」）、`gaps` 首条「主判据未测（WP 6.2.2）…从未端到端跑」、`status: 进行中`；而 `experiments` 已含 `.../2640-tree-conf-13m`（17:32 登记）。
- 证据：新实验 `analysis.md` §4 已判 **P1/P2 成立**（`criteria_fixed.json` 3 个 checkpoint 全 `pass: true`；`recompute_check.json` 独立复算同判），P4 全过、C1 过；同一事实在两处冲突 ⇒ 按 `AGENTS.md` §2.1「同一事实若在两处冲突，以权威处为准」应改 SSOT。
- 说明：`TICKET.md` §1 步骤 ⑤（更新 `status/conclusion/…`）排在复核之后，所以这**很可能是父级的既定收尾动作**；但在被复核 revision 上它是一处**当前事实不一致**，必须随本轮复核结论一并改掉（含 `gaps` 首条、`status`、以及「53m_s0 只跑 3 臂 / P3 否证 / P4 全过 / C1 的 13m_s1 不确定性」）。
- 最小修法：复核通过后由 axis-lead 就地重写 `6.2/project.yaml::conclusion`（把「待补跑」改成已判结论 + 证据指针 + 尺度限定）、删/改 `gaps` 首条、`status` 按 `SCHEMA.md` §1 定（有明确结论 ⇒ `judged`）。

**F4 · normal（低严重度）· `analysis.md` 正文带日期流水**（题目 E 明确点名）
- 位置：`analysis.md` §2：「读数 = run #1（2026-10-13 16:54 落盘）」「run #2（前 worker 16:56 启动）被运行时连带杀死，未产出任何读数」「`run.py` 已在 16:55 修好取键」。
- 违反：`AGENTS.md` §2.1（事实文档「一律陈述式、当前状态；正文叙述不带日期，不带『原本是 A、后来改成 B』…」，流水家在 `.scratch/**`）。这里的**当前事实**（「读数 = `results.json`；判据以 `criteria_fixed.json` 为准，因 `results.json::criteria` 的配对差字段因键方向不可用」）必须保留，但 run#2 被杀、16:55 修码属过程叙事。
- 最小修法：压成一句当前事实 + 指针（流水已在 `.scratch/63-axis6-predict/WP-6.2.2-LOG.md` 记全）。

**F5 · info · C1 的 Wilson CI 用探针的 `P3_free_gen_overall.n`**
- 位置：`run.py::probe_recall`（`"n": int(blob["P3_free_gen_overall"]["n"])`）与 `recompute_criteria.py::probe_recall`；13m_s1 的 C1 判词只靠 0.0086 的边界比较。
- 证据：`probes_output.json` 的 `P3_free_gen_topk_vs_T2` 各 k 条目**只有比例、无 n**，`overall.n` = 337/334/311。用 overall n 做 k=2 的 Wilson CI 是**可得的最接近口径**，且文档写明了 n；但若 k=2 的实际评分轮数少于 overall，CI 会略窄。
- 建议：文中加半句「探针 CI 用 `P3_free_gen_overall.n`（探针文件未按 k 拆 n）」，或直接引用探针自带的 k=2 n（若有）。不影响 C1 判词。

---

### 2. A–E 逐条结论（只写我核过的）

#### A 实现是否符合声称 —— **PASS**（A5 的「跑变异」未自跑，见下）
- **A1 PASS**。`spec_tree.py::pw_mix`：`w_a = p1[v1]*pc[a]`、`w_b = pd[b]`、`q = (w_a*pc + w_b*pd)/(w_a+w_b)`；`combined_fixed` 走 `first_slot_distribution(..., lam=COMBINED_LAM)` 且 `COMBINED_LAM = 0.5`（常量在 `spec_tree.py` 顶部，单测 `test_single_arm_first_drafts_match_legacy_2631` 断言 `LEGACY.COMBINED_LAM == T.COMBINED_LAM == 0.5`）。**真数据佐证**：`cycle_log.jsonl:4127` 第一轮 `w_A = 0.10548`，而 `p1_v1 = 0.30127`、`pc_a = 0.35013` ⇒ `0.30127×0.35013 = 0.10548` ✓，且 `w_A ≠ pc_a` ⇒ `p1[v1]` 未被约掉。`fallback` 计数 0（`results.json` 各臂活动块 `pw_fallback` 全 false）。
- **A2 PASS**。`TOPK_ARMS = {"tree_direct_k2": ("direct",2), "tree_qpw_k2": ("pw",2), "tree_topk4": ("pw",4)}`；`tree_union2` = `dict.fromkeys([argmax pc, argmax pd])`（去重）。实跑 `mean_candidates_per_round`：union2 = **1.5745**（13m_s0）/1.5114（13m_s1），direct_k2 = 2.0，qpw_k2 = 2.0，topk4 = 4.0，与预登记一致；`cycle_log.jsonl:4129` 有 `n_candidates: 1` 的 union2 轮（去重生效）。
- **A3 PASS**。`decode_tree` 每轮 `n_forward += 1` 且 `rows = [[v1, c, d3] for c in cands]` ⇒ **一次 B=K 行前向**；`results.json` 中 `n_forward_total == n_rounds`：13m_s0 union2 930/930、direct_k2 945/945（`results.json:7968-7970` 与 `:9146` 对应块）、qpw_k2 945/945、oracle 872/872（`:9179`/`:10282`）、topk4 908/908；13m_s1 918/918、912/912、924/924、896/896、872/872；53m_s0 树臂同（`recompute_check.json` 的 `n_rounds_recomputed` 与 `results.json` 逐值一致）。**掩码合法性**：`tree_forward` 与生产 `pathlm/decode.py::_run_stack_multi`（:139-160）**逐句同构**——新 key 用 `tril`（行内因果）、`window<=0 且 anchors==0` 时 `mask[:, :L-K] = True`（前缀对每行可见）；且 `tree_forward` 对 `window>0/anchors!=0` **显式 `NotImplementedError`**（不静默近似，单测 `test_tree_forward_rejects_eviction_configs` 覆盖）。
- **A4 PASS**。`acc_i = next((i for i,c in enumerate(cands) if int(logs[i,0].argmax()) == c), None)`：候选必须等于**验证器自己**在该位置的 argmax，无「无条件接受」；`d3` 只用**接受行**索引 1 的 logits 判（真前缀链）。与生产 `decode_spec`（`decode.py:344-348`：`fnodes[1]["logits"][0,n_acc].argmax() != drafts[n_acc]` 跳出）语义一致。`oracle_accept` 用真值但只在臂表出现，`criteria` 里不含它作判据（`controls/README.md` 也写明不作判据）。
- **A5 代码层面 PASS / 运行层面未自跑**。`analysis.md`→`test_cache_writeback_matches_from_scratch_recompute` 确实对「接受行写回」与 `Decoder.step` 从头重算做 argmax 相等 + 缓存长度断言，**写出错行/错位置或未推进 `dec.n` 都会红**（`assert dec.n == n_prefix+1+n_acc == dec2.n`）；`write_back_accepted_row` 与生产 `decode_spec` 的 `truncate` 语义对齐（只写 `p0..p0+n_pos-1`、被拒后缀不追加、`dec.n` 重锚）。**但我没有执行变异/单测**（无 shell）⇒「这条测试真的会失败」只有代码推理，无运行证据。

#### B 数字可否独立复现 —— **PASS（带限制）**：全部数值我都能从落盘原始字段手工对上，且两套独立实现互印证；**我自己的复算脚本未能运行**
- **B1 PASS（手工一致性 + 双实现互证，非自跑）**。臂级三元组自洽（我的算术）：τ = `accepted_total / n_rounds`（13m_s0 direct1 1426/981 = 1.453619 ✓；chain1 1248/1161 = 1.074935 ✓；13m_s1 tree_direct_k2 1500/912 = 1.644737 ✓）；α₁ 与 α₂|₁ 自洽：τ = α₁(1+α₂|₁)（direct1 0.802243×1.8119 = 1.4536 ✓；tree_direct_k2(13m_s1) 0.919956×1.787843 = 1.64475 ✓）。两套实现（`criteria_fixed.json` 与 `.scratch/63-axis6-predict/recompute_check.json`）的臂级 α₁/τ 与 `results.json` **逐位一致**，其 CI 与 `analysis.md` §3 表逐值一致（我抽查了 13m_s0 全部 10 臂 + 13m_s1 全部 8 个非对照臂的 α₁ CI：例 tree_union2 0.8654/0.9157、tree_direct_k2 0.8576/0.9263、tree_topk4 0.9027/0.9596、oracle 1/1、combined_fixed 0.7776/0.8320）。
- **B2 PASS（同上限制）**。配对 CI 抽查：13m_s0 tree_direct_k2 vs `combined_fixed`（最强单臂）= [0.065686, 0.121718]（`recompute_check.json` 独立实现）↔ analysis 表 [0.0657, 0.1217] ✓；tree_topk4 = [0.108006, 0.14705] ✓；tree_qpw_k2 = [0.036417, 0.088642] ✓；13m_s1 tree_direct_k2 vs `combo_pw` 的 CI 在 `criteria_fixed.json` 内（未逐位抄录，我核了它的臂级 CI）；13m_s0 P3 配对 CI = [−0.023331, 0.011967]（`criteria_fixed.json:415-424`）↔ analysis §4 P3 表 [−0.0233, 0.0120] ✓。**13m_s1 的 tree_topk4 配对 CI 我没有逐位核**（时间盒内未取到该行）。
- **B3 PASS**。「取键缺陷」成立且**读数未变**：`results.json::paired_diffs` 只存**按写入顺序**的无向键，`combined_fixed_minus_tree_union2` / `combined_fixed_minus_tree_direct_k2` / `combined_fixed_minus_tree_qpw_k2` / `combined_fixed_minus_combo_pw` 等存在，而 `tree_union2_minus_combined_fixed` 这类**反方向键不存在**（我用键名枚举核过 13m_s0/13m_s1 两段）⇒ 旧代码直线取 `f"{tree}_minus_{best}"` 必然 miss；对应 `results.json::criteria` 里 P1/P2 的 `paired*` 全 null、`pass:false`，而 `delta_vs_best_single` 为正；`run.py::paired_dir()` 现在按方向取符号（`derived_from: -(...)`）。`criteria_fixed.json` 三个 checkpoint 的 `drift_vs_results_json` **全为 `{}`** ✓；`recompute_check.json` 的 `alpha1_reported` vs `alpha1_recomputed` 全部相同到 1e-6。
- **B4 PASS**。`native` 确实走 `pathlm.decode.decode`（`run.py::arm_pass` 的 NATIVE 分支取 `native_streams[pi][:n_new]`，而 `native_streams` 由 `decode(model, prompt, n_new + T.GAMMA, pcap)` 生成）——不是批量 teacher forcing 冒充；`run_ckpt` 另核 `prefix_consistent`（`n_new+γ` 与 `n_new` 前缀一致），`run_log.txt` 三个 checkpoint 均 `prefix_consistent=True`。无损性：`results.json` 每个臂的 `per_prompt[*].identical_to_native` 全 true、`first_mismatches: []`、`P4_lossless.all_arms_stream_identical: true`（3 个 checkpoint）。
- **B5 mutation_report 内容 PASS / 自跑未做**。`mutation_report.json`：3 条 `caught_red: true`、`all_restored_bitwise: true`，`sha256_restored == targets.sha256`（`71d98b26…8fb62f`）；三条补丁的原文片段在 `spec_tree.py` 中**各自唯一可定位**（`TOPK_ARMS` 那行、`acc_i = next((i for i, c ...` 表达式、`if arm == "combo_pw": return pw_mix(nodes)["q"]`），且正好对应预登记的 3 件事（M1 K=2→1 退化 / M2 无条件接受 → 无损性门禁 / M3 `combo_pw` 权重退化成等权 = `combined_fixed`）。**我未能复跑 `mutate.py`**（无 shell）。
  - info：`README.md` §6 自检 1（缓存正确性）**没有对应变异**——它的可失败性只有代码推理（见 A5），无变异/运行证据。

#### C 越界/违规 —— **PASS（符号级证据）+ 两项未能核**
- 共享引擎：`grep -n "2640-tree-conf|tree_forward|decode_tree" pathlm/` → **无匹配** ⇒ 树实现完全在实验目录内，`pathlm/**` 未新增钩子；`run.py`/`spec_tree.py` 对 `pathlm.decode`/`pathlm.model` 只读导入（`spec_tree.py` 顶部 `from pathlm.decode import Decoder, _slice_nodes, decode, decode_spec`）。
- GPU 锁（§20）：`results.json::lock_events` 6 条——`acquire 16:38:44 → release 16:44:49 (after 13m_s0) → acquire 16:45:50 → release 16:51:48 → acquire 16:52:48 → release 16:54:45`，两次让出窗口 = **61 s / 60 s**（`run_log.txt` 内有 `[lock] §20.1 让出窗口：sleep 60s`）；`[run] wrote results.json / cycle_log.jsonl (963s)` ⇒ 全程 ≈16 min，在 60 min 硬盒内；`waited_s: 0.0` 表示没抢/没杀别人进程。
- 禁墙钟判据：`results.json` 中 `wall|speedup|elapsed|duration|seconds|latency|throughput|tok_s|ms` 键**零命中**（只有 `timing_measured: false`、`timing_note`、`budget_min` 配置）；`analysis.md` 判定表内无墙钟数字；`6.2/project.yaml::cost_model` 纯解析量（唯一涉墙条目是 `gaps` 里「墙钟…不作判据」的声明）。
- 预登记未被事后放宽：**无法用 git 核**（无 Git）。可得旁证：`results.json::config.arms` 列出实跑的 10 臂，`criteria` 内含 C1 字段 ⇒ 追加臂与 C1 **在 run #1（16:38:44 起跑）之前就已在代码/契约里**；`WP-6.2.2-LOG.md` 记录了 axis-lead 中途追加 `tree_direct_k2`/`tree_qpw_k2`/C1 并「已并入 README.md」。**`README.md` 是否在 16:54 落盘后被改（判据是否被放宽）我无法证伪** ⇒ 见下「必须由 supervisor 跑的补齐命令」。
- **未核项①**：`git status --short` / `git diff --stat` 证明未改 `pathlm/**`、`docs/**`、其它轴、其它 `project.yaml` 实体内容——`watchdog_diff` 报 `Reviewer-launch HEAD changed or is unavailable`（不含已提交区间），我无 Git。**未核项②**：`README.md` 预登记判据的历史不可变性（同上）。⇒ C 的结论只覆盖「符号级 + 落盘读数」范围。

#### D 结论是否越证据 —— **PASS（1 条过度声明，见 F2；1 条数值错，见 F1）**
- **D1**：见 **F2**（未披露 K 行各带完整 3 位置链 ⇒ 成本口径偏保守、不是高估读者对树的估值，但未声明）。补充：即使按理想共享树折算，§5「τ 边际收益随 K 递减」的定性结论**仍成立**（我按表内数算过：理想口径 K=2→4 成本比 ≈1.8×，tok/fwd 仅 2.608→2.659）。
- **D2 PASS**。13m_s1 的处置**如实**：`analysis.md` §4 C1 明写「探针点 0.8922 **落在**本臂 95% CI 之外」，给出两 CI 重叠（探针 Wilson 上界 0.9211 > 本臂下界 0.900752 ✓ 我复算过 Wilson：13m_s0 [0.8194,0.8934]、13m_s1 [0.8544,0.9211] 与表一致）、方向为「实测更高」、并写「不判 C1 失败/不判实现 bug」+ 记为已知不确定性；`results.json::criteria_overall.13m_s1.C1_probe_inside_ci = false` 也对得上。唯一缺陷是差距数字（F1）。
- **D3 PASS**。P3 只否证「加权 vs 等权在本尺度有差」，明写「不宣称任何一方更优」，且给了活性量（逐轮分歧率 0.0566/0.0582 ≠ 0 ⇒ 两臂确实不同）；**没有**滑向「组合无价值」——§5 否证条件 3 写「部分触发」并解释读数（分歧率 ≠ 0）。附注：`README.md` §4 否证条件 3 的字面触发条件含「分歧率 ≈ 0」，严格说未触发；文中已如实说明并按判据精神裁定，属可接受。另：P1 的「最强单臂」按 checkpoint 实测决定 ⇒ 13m_s0 = `combined_fixed`、13m_s1 = `combo_pw`，跨 checkpoint 的 Δ 不是同一基线（表头已标，属预登记写法）。
- **D4 PASS**。每条结论都带（checkpoint / 臂 / n）：§3 三个表头写了模型/最强单臂、每行有臂与 CI，§4 每行有 Δ 与 CI；53m_s0 明写「只跑了 3 臂（native/direct1/tree_direct_k2），是规模点而非完整结论」，§1 也写明只跑了一个树臂；§6 逐条列未测（无温度>0、无 batch>1、无二级条件树、n=2 个 13M、53m_s0 未测 7 臂）并明令「不得读成业界做不到/做得到，不得外推到别的任务/规模/温度」。info：53m_s0 的「最强单臂 = direct1」实为该 checkpoint **唯一被测**单臂（表头「只跑了 3 臂」已兜住）。

#### E 文档一致性 —— **PASS（role_map / 无损性口径）+ 1 条不一致（F3）+ 1 条体例（F4）**
- `role_map`：5 份 `axes/6-predict/*/project.yaml` 均为四角色且实名（owner=axis6-lead、writer=各项 worker（6.2 为「axis6-w-6.2 …WP 6.2.2 由 ticket 63 派发的 WP worker 写入」）、reviewer=axis6-reviewer、integrator=tree-integrator）✓。
- `6.2::objective_criteria` 与 `conclusion` 的**无损性口径自洽**：判据写「13M 逐位相等硬门禁；更大规模已知 fp 平局翻转…按分歧率有界判、逐位相等不作跨规模判据」，`conclusion`(e) 同口径 ✓（这与 `TICKET.md` §3 记录的 (a) 处修正一致）。`cost_model` 纯解析量、无墙钟 ✓。
- 不一致：**F3**（`conclusion`/`status`/`gaps` 仍写「主判据待 6.2.2 补跑」）。
- 新实验已登记：`6.2/experiments/README.md:16` 有 `2640-tree-conf-13m` 行 ✓；`project.yaml::experiments` 含该目录 ✓。
- `analysis.md` 体例：**F4**（§2 带日期流水）。
- `criteria_fixed.json` 的 `source`/`reason` 字段把「读数 = run #1、判据走离线重算」写成了可核的当前事实 ✓。

---

### 3. 必须由 supervisor 执行的补齐命令（我无 shell，全部 `not-run`）

1. 范围/越界 + 预登记不变性：`git status --short`；`git diff --stat 01d25ff..HEAD -- pathlm docs axes/6-predict`（应只见 `axes/6-predict/**` 的 role_map 等文档改动）；`git log -p --follow -- axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/README.md`（确认 16:54 之后判据未被放宽）；`git diff` 确认工作树里 `pathlm/**` 无改动。
2. 自跑复算（B1/B2，钉住 §0 的 sha256）：`python3 axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/recompute_criteria.py`（`drift_vs_results_json` 应为空）；`.scratch/63-axis6-predict/recompute_check.py` 对该目录 `results.json` 复跑。
3. 单测与变异（A5/B5）：`python3 -m pytest axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/test_spec_tree.py -q`；`python3 axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/mutate.py`（并确认 `spec_tree.py` sha256 回到 `71d98b26…8fb62f`）。
4. 门禁：`python3 tools/summary.py --check`、`python3 tools/check_tree.py`、`python3 -m pytest tests/ -q`（`TICKET.md` 声称全绿，我无法核）。
5. 哈希自核：对上表 4 个文件跑 `sha256sum`，确认与我引用的生产者自报值一致。

---

### 4. 判词

`.scratch/63-axis6-predict/REVIEW-1.md`（本报告；因本上下文无写文件工具，由 runtime 落盘到该权威路径）。

在我**钉住的这个 revision 上、我实际检查过的范围内**：实现与声称一致（A 全过，A5 仅代码推理）、数字可从落盘原始字段复现且两套独立实现互证（B，唯 B5 未自跑）、无越界改动迹象（C 的符号级与锁/墙钟部分）、结论未越证据（D，除 F2 一处未披露与 F1 一处 10 倍数值错）、文档一致性除 F3/F4 外成立。发现 4 条 normal + 1 条 info，**无 blocker**；两项未核（C 的 git/工作树全量比对、B1/B2/B5 的自跑）已在 §0/§3 显式声明为限制，不构成 PASS 之外的隐含声明。

**VERDICT: PASS（blocker 0 条；2 项限制未核：①无 shell/git ⇒ 自跑复算与变异、工作树/Git 越界比对未做；②F1–F4 需在收口提交里修）**